"""Full test-set evaluation: Build Plan -> Improve loop (stop after 3 no-improve).

For each instance:
1. BuildPlan (up to 3 retries if invalid)
2. ImprovePlan loop until 3 consecutive rounds with no score improvement
3. Report best score per instance, with timing

Usage:
    python -m vllm.entrypoints.openai.api_server \
        --model /data/models/qwen3-8b-osm_v2_sft \
        --port 8000 --tensor-parallel-size 2 --max-model-len 32768

    python examples/qwen3-8B-osm_v2_atomic/eval_full_test.py \
        --url http://localhost:8000 \
        --model /data/models/qwen3-8b-osm_v2_sft \
        --workers 16 --max-turns 25 \
        --output output/eval_full_test.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import openai
import httpx
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
ROLL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROLL_ROOT))
sys.path.insert(0, str(ROLL_ROOT / "roll/pipeline/agentic/env/osm_v2/deps"))

import types
gem_mock = types.ModuleType("gem")
class _EnvBase:
    def reset(self, seed=None): pass
    def step(self, action): pass
    def close(self): pass
gem_mock.Env = _EnvBase
gem_mock.register = lambda *a, **k: None
sys.modules.setdefault("gem", gem_mock)

from roll.pipeline.agentic.env.osm_v2.env import OsmV2AtomicEnv, normalize_score
from roll.pipeline.agentic.env.osm_v2.deps.atomic_harness import write_plan

DEFAULT_SCAFFOLD = str(Path(__file__).resolve().parent / "data/scaffolds/osm_v2_terminus2.yaml")
TASK_TYPES = ["road_planning", "ev_charging", "urban_planning"]
print_lock = threading.Lock()


def _save_rollouts(rollout_dir, instance, rollouts):
    """Save per-instance rollout to a JSONL file for inspection."""
    if not rollout_dir:
        return
    task_dir = Path(rollout_dir) / instance["task"]
    task_dir.mkdir(parents=True, exist_ok=True)
    out_path = task_dir / f"{instance['instance_id']}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rollouts:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_test_instances(instances_root: str, n_per_task: int = None):
    instances = []
    for task in TASK_TYPES:
        task_dir = Path(instances_root) / "test" / task
        if not task_dir.exists():
            continue
        files = sorted(task_dir.glob("*.json"))
        if n_per_task:
            files = files[:n_per_task]
        for f in files:
            instances.append({"task": task, "path": str(f), "instance_id": f.stem})
    return instances


def load_instances_file(path: str, instances_root: str) -> list[dict]:
    """Load a fixed evaluation subset manifest.

    Supported formats:
    - JSON object with an "instances" list
    - JSON list of instance rows
    - JSONL with one instance row per line
    """
    manifest_path = Path(path)
    if not manifest_path.exists() and not manifest_path.is_absolute():
        manifest_path = ROLL_ROOT / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"instances file not found: {path}")

    if manifest_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = payload.get("instances", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"instances file must contain a list: {path}")

    instances = []
    for row in rows:
        task = row.get("task") or row.get("task_name")
        instance_id = row.get("instance_id") or row.get("id")
        instance_path = row.get("path") or row.get("instance_path")
        if not task or not instance_id:
            raise ValueError(f"instance row requires task and instance_id: {row}")
        if not instance_path:
            instance_path = Path(instances_root) / "test" / task / f"{instance_id}.json"
        instance_path = Path(instance_path)
        if not instance_path.exists() and not instance_path.is_absolute():
            instance_path = ROLL_ROOT / instance_path
        if not instance_path.exists():
            raise FileNotFoundError(f"instance path not found for {task}/{instance_id}: {instance_path}")
        item = {"task": task, "path": str(instance_path), "instance_id": instance_id}
        for key in ("difficulty", "difficulty_bucket", "demand_count", "candidate_count"):
            if key in row:
                item[key] = row[key]
        instances.append(item)
    return instances


def sanitized_config(args) -> dict:
    config = vars(args).copy()
    if config.get("api_key"):
        config["api_key"] = "***redacted***"
    return config


def save_results(output_path: str, args, results: list[dict], started_at: float, completed: int, total: int) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": sanitized_config(args),
        "completed": completed,
        "total": total,
        "running_time_sec": round(time.time() - started_at, 1),
        "results": results,
    }
    Path(output_path).write_text(json.dumps(output_data, indent=2, ensure_ascii=False))


def normalize_openai_base_url(url: str) -> str:
    """Accept either a server URL or an OpenAI-compatible /v1 base URL."""
    clean = url.rstrip("/")
    return clean if clean.endswith("/v1") else f"{clean}/v1"


def make_client(url: str, api_key: str, timeout: float) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=normalize_openai_base_url(url),
        api_key=api_key,
        timeout=timeout,
        http_client=httpx.Client(timeout=timeout, trust_env=False),
    )


def healthcheck_client(client: openai.OpenAI) -> list[str]:
    models = client.models.list()
    return [str(m.id) for m in models.data]


def call_llm(client, model, messages, max_model_len=32768, temperature=0.7, max_response_tokens=4096):
    total_chars = sum(len(m["content"]) for m in messages)
    est_tokens = total_chars // 3
    max_tokens = min(max_response_tokens, max(512, max_model_len - est_tokens - 256))
    if max_tokens < 512:
        raise ValueError(f"Context too long: ~{est_tokens} tokens, no room for output")
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def run_episode(client, model, env, seed, max_turns, temperature, max_response_tokens):
    obs, info = env.reset(seed=seed)
    system_prompt = info.get("env_instruction", "")
    ctx = info.get("atomic_context") or info.get("context") or {}
    task_intro = ""
    if ctx:
        task_intro = (
            f"Task: {ctx.get('task', 'unknown')}\n"
            f"Goal: {ctx.get('goal', '')}\n"
            f"Instance: {ctx.get('instance_id', '')}\n"
            f"Final plan path: {ctx.get('final_plan_path', 'outputs/final_plan.json')}\n"
            f"Finish command: {ctx.get('finish_command', '')}\n\n"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_intro + obs},
    ]
    total_reward = 0.0
    turns = 0
    terminated = False
    for turn_i in range(max_turns):
        turns = turn_i + 1
        try:
            action = call_llm(
                client,
                model,
                messages,
                temperature=temperature,
                max_response_tokens=max_response_tokens,
            )
        except Exception as e:
            logging.warning(f"LLM call failed turn {turns}: {e}")
            break
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        messages.append({"role": "assistant", "content": action})
        messages.append({"role": "user", "content": obs})
        if terminated or truncated:
            break

    final_score = info.get("final_score", 0.0)
    task_name = ""
    if hasattr(env, "export") and env.export and hasattr(env.export, "task"):
        task_name = env.export.task

    # Extract plan ids for improve loop
    plan_ids = []
    final_plan_path = None
    if hasattr(env, "env") and env.env and hasattr(env.env, "outputs_dir"):
        final_plan_path = env.env.outputs_dir / "final_plan.json"
    if final_plan_path and final_plan_path.exists():
        try:
            payload = json.loads(final_plan_path.read_text())
            if isinstance(payload, list):
                plan_ids = [str(x.get("action_id", x)) if isinstance(x, dict) else str(x) for x in payload]
            elif isinstance(payload, dict):
                for key in ("candidate_ids", "action_ids", "selected_candidate_ids"):
                    if isinstance(payload.get(key), list):
                        plan_ids = [str(x) for x in payload[key]]
                        break
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "final_score": final_score,
        "normalized": round(normalize_score(final_score, task_name), 4),
        "valid": info.get("valid", False),
        "turns": turns,
        "reward": round(total_reward, 4),
        "task_name": task_name,
        "plan_ids": plan_ids,
        "messages": messages,
    }


def run_instance(
    client,
    model,
    instance,
    instances_root,
    plan_bank_path,
    scaffold,
    max_turns,
    patience=3,
    rollout_dir=None,
    temperature=0.7,
    max_response_tokens=4096,
    ev_score_version=None,
):
    """Run Build + Improve loop for one instance. Returns summary dict."""
    import tempfile, uuid
    t_start = time.time()
    # Each instance gets its own run_root to avoid sandbox conflicts in concurrent mode
    instance_run_root = Path(tempfile.gettempdir()) / "roll_eval" / f"{instance['instance_id']}_{uuid.uuid4().hex[:6]}"
    instance_run_root.mkdir(parents=True, exist_ok=True)
    data_source = instance["path"]
    if instance["task"] == "ev_charging" and ev_score_version:
        payload = json.loads(Path(data_source).read_text())
        payload.setdefault("metadata", {})["ev_score_version"] = ev_score_version
        data_source = instance_run_root / "data_sources" / f"{instance['instance_id']}_{ev_score_version}.json"
        data_source.parent.mkdir(parents=True, exist_ok=True)
        data_source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        data_source = str(data_source)
    common = dict(
        data_source=data_source,
        isolation="local",
        step_timeout_seconds=30,
        workspace_prompt_config=scaffold,
        run_root=str(instance_run_root),
    )
    rollouts = []

    # Phase 1: BuildPlan with up to 3 retries
    build_result = None
    for attempt in range(1, 4):
        env = OsmV2AtomicEnv(atom="BuildPlan", max_turns=max_turns, **common)
        result = run_episode(
            client,
            model,
            env,
            seed=hash(data_source) + attempt,
            max_turns=max_turns,
            temperature=temperature,
            max_response_tokens=max_response_tokens,
        )
        env.close()
        rollouts.append({"phase": "BuildPlan", "attempt": attempt, "score": result["final_score"],
                         "valid": result["valid"], "turns": result["turns"], "messages": result["messages"]})
        if result["valid"] and result["plan_ids"]:
            build_result = result
            break

    if not build_result:
        elapsed = time.time() - t_start
        _save_rollouts(rollout_dir, instance, rollouts)
        return {
            "instance_id": instance["instance_id"],
            "task": instance["task"],
            "build_score": result["final_score"],
            "best_score": result["final_score"],
            "best_normalized": result["normalized"],
            "valid": False,
            "improve_rounds": 0,
            "total_turns": result["turns"],
            "elapsed_sec": round(elapsed, 1),
        }

    # Phase 2: ImprovePlan loop (create temp bank from BuildPlan result)
    current_plan_ids = build_result["plan_ids"]
    current_score = build_result["final_score"]
    best_score = current_score
    best_normalized = build_result["normalized"]
    no_improve_count = 0
    improve_rounds = 0
    total_turns = build_result["turns"]

    # Create a temp plan bank with the build result so ImprovePlan can sample from it
    import tempfile
    task_name = ""
    if hasattr(build_result, "get"):
        task_name = build_result.get("task_name", "")
    tmp_bank = Path(tempfile.mktemp(suffix=".jsonl"))
    bank_entry = {"candidate_ids": current_plan_ids, "score": current_score, "valid": True,
                  "task_name": task_name or instance["task"], "instance_id": instance["instance_id"]}
    tmp_bank.write_text(json.dumps(bank_entry) + "\n")

    while no_improve_count < patience:
        improve_rounds += 1
        # Update bank with latest best plan
        bank_entry = {"candidate_ids": current_plan_ids, "score": current_score, "valid": True,
                      "task_name": task_name or instance["task"], "instance_id": instance["instance_id"]}
        tmp_bank.write_text(json.dumps(bank_entry) + "\n")

        env = OsmV2AtomicEnv(atom="ImprovePlan", max_turns=max_turns, base_plan_bank_path=str(tmp_bank), **common)
        result = run_episode(
            client,
            model,
            env,
            seed=hash(data_source) + improve_rounds + 100,
            max_turns=max_turns,
            temperature=temperature,
            max_response_tokens=max_response_tokens,
        )
        env.close()
        total_turns += result["turns"]
        improved = result["valid"] and result["final_score"] > best_score
        rollouts.append({"phase": "ImprovePlan", "round": improve_rounds, "score": result["final_score"],
                         "valid": result["valid"], "turns": result["turns"], "improved": improved,
                         "messages": result["messages"]})

        if improved:
            best_score = result["final_score"]
            best_normalized = result["normalized"]
            if result["plan_ids"]:
                current_plan_ids = result["plan_ids"]
            current_score = result["final_score"]
            no_improve_count = 0
        else:
            no_improve_count += 1

    tmp_bank.unlink(missing_ok=True)

    elapsed = time.time() - t_start
    _save_rollouts(rollout_dir, instance, rollouts)
    return {
        "instance_id": instance["instance_id"],
        "task": instance["task"],
        "build_score": build_result["final_score"],
        "best_score": best_score,
        "best_normalized": best_normalized,
        "valid": True,
        "improve_rounds": improve_rounds,
        "total_turns": total_turns,
        "elapsed_sec": round(elapsed, 1),
    }


def print_summary(results):
    print(f"\n{'='*70}")
    print(f"  FULL TEST SET RESULTS ({len(results)} instances)")
    print(f"{'='*70}")

    valid = [r for r in results if r["valid"]]
    print(f"\n  Valid rate: {len(valid)}/{len(results)} ({100*len(valid)/max(len(results),1):.1f}%)")

    if valid:
        scores = [r["best_normalized"] for r in valid]
        print(f"  Best normalized score (valid): mean={sum(scores)/len(scores):.4f}, "
              f"min={min(scores):.4f}, max={max(scores):.4f}")

    all_elapsed = [r["elapsed_sec"] for r in results]
    print(f"  Avg time per instance: {sum(all_elapsed)/len(all_elapsed):.1f}s")
    print(f"  Total time: {sum(all_elapsed)/3600:.2f}h")

    avg_turns = sum(r["total_turns"] for r in results) / len(results)
    avg_improve = sum(r["improve_rounds"] for r in results) / len(results)
    print(f"  Avg total turns: {avg_turns:.1f}")
    print(f"  Avg improve rounds: {avg_improve:.1f}")

    print(f"\n  Per-task breakdown:")
    print(f"  {'Task':<20} {'Valid':<10} {'Score(raw)':<15} {'Score(norm)':<15} {'Avg Time':<12} {'Avg Rounds'}")
    print(f"  {'-'*20} {'-'*10} {'-'*15} {'-'*15} {'-'*12} {'-'*10}")
    for task in TASK_TYPES:
        task_results = [r for r in results if r["task"] == task]
        if not task_results:
            continue
        task_valid = [r for r in task_results if r["valid"]]
        valid_str = f"{len(task_valid)}/{len(task_results)}"
        if task_valid:
            raw_scores = [r["best_score"] for r in task_valid]
            norm_scores = [r["best_normalized"] for r in task_valid]
            raw_score_str = f"{sum(raw_scores)/len(raw_scores):.4f}"
            norm_score_str = f"{sum(norm_scores)/len(norm_scores):.4f}"
        else:
            raw_score_str = "N/A"
            norm_score_str = "N/A"
        avg_t = sum(r["elapsed_sec"] for r in task_results) / len(task_results)
        avg_r = sum(r["improve_rounds"] for r in task_results) / len(task_results)
        print(f"  {task:<20} {valid_str:<10} {raw_score_str:<15} {norm_score_str:<15} {avg_t:<12.1f} {avg_r:.1f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    default_model = os.environ.get("OSMV2_TEACHER_MODEL")
    parser.add_argument("--url", default=os.environ.get("DEEPSEEK_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", "empty"))
    parser.add_argument("--model", default=default_model, required=default_model is None)
    parser.add_argument("--instances-root", default="examples/qwen3-8B-osm_v2_atomic/data/city_tasks_osm_v2/instances")
    parser.add_argument("--plan-bank", default="examples/qwen3-8B-osm_v2_atomic/data/large_4000_500/bank/plans.jsonl")
    parser.add_argument("--scaffold", default=os.environ.get("OSM_V2_SCAFFOLD_CONFIG", DEFAULT_SCAFFOLD))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("OSMV2_TEMPERATURE", "0.7")))
    parser.add_argument("--max-response-tokens", type=int, default=int(os.environ.get("OSMV2_MAX_RESPONSE_TOKENS", "4096")))
    parser.add_argument("--request-timeout", type=float, default=float(os.environ.get("OSMV2_REQUEST_TIMEOUT", "300")))
    parser.add_argument("--ev-score-version", choices=["v3"], default=os.environ.get("OSMV2_EV_SCORE_VERSION"))
    parser.add_argument("--n", type=int, default=None, help="Instances per task (None=all)")
    parser.add_argument("--instances-file", default=None, help="Fixed subset manifest JSON/JSONL")
    parser.add_argument("--patience", type=int, default=3, help="Stop improve after N consecutive no-improve")
    parser.add_argument("--output", default="output/eval_full_test.json")
    parser.add_argument("--rollout-dir", default="output/rollouts", help="Dir to save per-instance rollout messages")
    parser.add_argument("--skip-healthcheck", action="store_true")
    args = parser.parse_args()

    if args.instances_file:
        instances = load_instances_file(args.instances_file, args.instances_root)
    else:
        instances = get_test_instances(args.instances_root, args.n)
    task_counts = {task: sum(1 for inst in instances if inst["task"] == task) for task in TASK_TYPES}
    print(f"Evaluating {len(instances)} instances")
    print(f"Task counts: {task_counts}")
    print(f"Workers: {args.workers}, Max turns: {args.max_turns}, Patience: {args.patience}")
    print(f"Model: {args.model}")
    print(f"API base URL: {normalize_openai_base_url(args.url)}")
    print(f"Scaffold: {args.scaffold}")
    print(f"Instances file: {args.instances_file or 'default full test ordering'}")
    print(f"EV score version override: {args.ev_score_version or 'default'}")
    print(f"Temperature: {args.temperature}, Max response tokens: {args.max_response_tokens}, Request timeout: {args.request_timeout}s")
    print(f"Output: {args.output}\n")

    client = make_client(args.url, args.api_key, args.request_timeout)
    if not args.skip_healthcheck:
        model_ids = healthcheck_client(client)
        print(f"API healthcheck OK: models={model_ids[:8]}\n")
    results = []
    completed = 0
    t_global_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_instance, client, args.model, inst,
                args.instances_root, args.plan_bank, args.scaffold, args.max_turns, args.patience,
                args.rollout_dir, args.temperature, args.max_response_tokens, args.ev_score_version
            ): inst
            for inst in instances
        }

        for future in as_completed(futures):
            inst = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                status = "VALID" if result["valid"] else "FAIL"
                with print_lock:
                    print(f"  [{completed}/{len(instances)}] {result['task']}/{result['instance_id']}: "
                          f"{status} score={result['best_normalized']:.4f} "
                          f"rounds={result['improve_rounds']} "
                          f"time={result['elapsed_sec']:.1f}s")
                save_results(args.output, args, results, t_global_start, completed, len(instances))
            except Exception as e:
                completed += 1
                with print_lock:
                    print(f"  [{completed}/{len(instances)}] {inst['task']}/{inst['instance_id']}: ERROR {e}")
                save_results(args.output, args, results, t_global_start, completed, len(instances))

    total_time = time.time() - t_global_start
    print(f"\nTotal wall-clock time: {total_time/3600:.2f}h")

    print_summary(results)

    # Save results
    output_data = {
        "config": sanitized_config(args),
        "completed": len(results),
        "total": len(instances),
        "total_time_sec": round(total_time, 1),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
