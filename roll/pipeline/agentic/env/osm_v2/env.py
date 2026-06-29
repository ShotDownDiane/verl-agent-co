"""OSM-v2 Atomic Workspace environment adapted for ROLL's gem.Env interface.

Wraps the STS OSMV2AtomicWorkspaceEnv and exposes the standard
reset/step/close contract expected by ROLL's agentic pipeline.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from gem import Env

from roll.pipeline.agentic.env.osm_v2.deps.atomic_harness import (
    load_export,
    score_value,
    selected_count,
    top_editable_candidates,
    write_plan,
)
from roll.pipeline.agentic.env.osm_v2.deps.scaffold import OSMV2WorkspaceScaffold, clip_text, stable_json_hash
from roll.pipeline.agentic.env.osm_v2.deps.workspace_env import OSMV2WorkspaceSandboxEnv


FINISH_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MAX_OBS_CHARS = 10000
OBS_TRUNCATE_THRESHOLD = 2500
OBS_KEEP_HEAD = 1000
OBS_KEEP_TAIL = 1000
OBS_TRUNCATE_MSG = "\n... [output too long, middle omitted. Use more targeted commands like grep/head/tail to inspect specific parts] ...\n"

# Per-task normalization: maps raw score to [0,1] with p5→0.2, p95→0.8
_TASK_SCORE_NORM = {
    "road_planning": {"low": -1.0, "high": 1.0},
    "ev_charging": {"low": 0.1283, "high": 0.8845},
    "urban_planning": {"low": -0.0094, "high": 0.4832},
}


def normalize_score(score: float, task_name: str) -> float:
    """Map raw evaluator score to [0,1] with p5→0.2, p95→0.8."""
    norm = _TASK_SCORE_NORM.get(task_name)
    if norm is None:
        return max(0.0, min(1.0, score))
    span = norm["high"] - norm["low"]
    if span <= 0:
        return 0.5
    normalized = 0.2 + 0.6 * (score - norm["low"]) / span
    return max(0.0, min(1.0, normalized))


def _roll_root() -> Path:
    """Return the ROLL project root directory."""
    env_val = os.environ.get("ROLL_DIR")
    if env_val and Path(env_val).exists():
        return Path(env_val).resolve()
    candidate = Path(__file__).resolve().parents[6]
    if (candidate / "roll").is_dir():
        return candidate
    return Path.cwd()


ROOT = _roll_root()
DEFAULT_PROMPT_CONFIG = ROOT / "examples/qwen3-8B-osm_v2_atomic/data/scaffolds/osm_v2_terminus2.yaml"

def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    text = str(path)
    for marker in ("/STS/", "/slime_osm_v2_atomic/"):
        if marker in text:
            relative = text.split(marker, 1)[1]
            candidate = ROOT / relative
            if candidate.exists():
                return candidate
    if not path.is_absolute():
        candidate = ROOT / path
        if candidate.exists():
            return candidate
    return path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _extract_command(text: str) -> str:
    """Extract one bash command from a model response."""
    tool_matches = re.findall(
        r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL | re.IGNORECASE
    )
    for raw in reversed(tool_matches):
        raw = raw.strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            args = payload.get("arguments") or payload.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"command": args}
            if isinstance(args, Mapping) and args.get("command") is not None:
                return str(args["command"]).strip()

    fence_matches = re.findall(
        r"```(?:bash|sh|shell)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if fence_matches:
        return fence_matches[-1].strip()

    command_match = re.search(
        r'"command"\s*:\s*"(?P<command>(?:\\.|[^"\\])*)"', text, flags=re.DOTALL
    )
    if command_match:
        try:
            return json.loads('"' + command_match.group("command") + '"').strip()
        except json.JSONDecodeError:
            pass

    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    for index, line in enumerate(lines):
        if line.strip().startswith(
            ("python ", "python3 ", "cat ", "ls ", "head ", "echo ", "mkdir ", "printf ")
        ):
            return "\n".join(lines[index:])
    return lines[-1]

def _extract_commands_terminus2(text: str) -> tuple[list[str], bool, bool]:
    """Parse Terminus-2 JSON response. Returns (commands_list, task_complete, json_parsed)."""
    payload = None

    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            pass

    if payload is None:
        fence = re.search(r"```(?:json)?\s*\n(.*?)```", text, flags=re.DOTALL)
        if fence:
            try:
                payload = json.loads(fence.group(1).strip())
            except json.JSONDecodeError:
                pass

    if payload is None:
        brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if brace:
            try:
                payload = json.loads(brace.group(0))
            except json.JSONDecodeError:
                pass

    if isinstance(payload, dict) and "commands" in payload:
        task_complete = bool(payload.get("task_complete", False))
        raw_cmds = payload.get("commands", [])
        commands = []
        for cmd in raw_cmds:
            if isinstance(cmd, dict):
                ks = cmd.get("keystrokes", "")
            elif isinstance(cmd, str):
                ks = cmd
            else:
                continue
            ks = ks.rstrip("\n")
            if ks:
                commands.append(ks)
        return commands, task_complete, True

    # Fallback: legacy single-command extraction
    legacy = _extract_command(text)
    if legacy:
        return [legacy], False, False
    return [], False, False


def _render_terminal_observation(results: list[dict[str, Any]], initial: bool = False) -> str:
    """Render command execution results as terminal-style output (Terminus-2 format)."""
    prompt = "root@workspace:/workspace#"
    if initial:
        header = f"Current terminal state:\nCurrent Terminal Screen:\n{prompt}"
    else:
        header = "New Terminal Output:"

    lines = [header]
    for r in results:
        cmd = r.get("command", "")
        output = r.get("output", "")
        lines.append(f"{prompt} {cmd}")
        if output:
            out_text = output.rstrip()
            if len(out_text) > OBS_TRUNCATE_THRESHOLD:
                out_text = out_text[:OBS_KEEP_HEAD] + OBS_TRUNCATE_MSG + out_text[-OBS_KEEP_TAIL:]
            lines.append(out_text)
    lines.append(prompt)
    return "\n".join(lines)


class OsmV2AtomicEnv(Env):
    """OSM-v2 atomic workspace env for ROLL's agentic pipeline.

    Wraps the STS workspace sandbox and exposes the gym-like interface
    that ROLL's TrajEnvManager / OsmV2EnvManager expects.

    Two modes of operation:
    1. Direct mode: pass `data_source` for a single fixed instance.
    2. Dataset mode: pass `prompt_data_path` (JSONL) and the env selects
       an instance on each reset(seed) via seed-based indexing.
    """

    def __init__(
        self,
        atom: str = "BuildPlan",
        data_source: str | None = None,
        data_source_path: str | None = None,
        prompt_data_path: str | None = None,
        max_turns: int = 15,
        run_root: str | None = None,
        isolation: str = "local",
        step_timeout_seconds: int = 30,
        workspace_prompt_config: str | None = None,
        base_plan_bank_path: str | None = None,
        base_sampling_policy: dict | None = None,
        editable_top_k: int = 80,
        max_obs_chars: int = MAX_OBS_CHARS,
        **kwargs: Any,
    ) -> None:
        self.default_atom = atom
        self.max_turns = max_turns
        self.run_root = Path(
            run_root or os.environ.get("OSM_ATOMIC_RUN_ROOT")
            or Path(tempfile.gettempdir()) / "roll_osm_v2_atomic"
        ).resolve()
        self.isolation = isolation
        self.step_timeout_seconds = step_timeout_seconds
        prompt_cfg = workspace_prompt_config or str(DEFAULT_PROMPT_CONFIG)
        self.prompt_config_path = _repo_path(prompt_cfg)
        self.scaffold = OSMV2WorkspaceScaffold.from_file(self.prompt_config_path)
        self.max_obs_chars = max_obs_chars
        self.editable_top_k = editable_top_k
        self.default_base_plan_bank_path = base_plan_bank_path
        self.base_sampling_policy = base_sampling_policy or {}
        self.extra_config = kwargs

        # Dataset mode: load all instances from prompt JSONL
        self._instance_rows: list[dict[str, Any]] | None = None
        if prompt_data_path:
            self._instance_rows = _load_jsonl(_repo_path(prompt_data_path))
            if not self._instance_rows:
                raise ValueError(f"No rows in prompt_data_path: {prompt_data_path}")

        # Direct mode: single fixed instance
        self._fixed_data_source: Path | None = None
        if not prompt_data_path:
            source = data_source or data_source_path
            if source is None:
                raise ValueError("Either data_source or prompt_data_path is required")
            self._fixed_data_source = _repo_path(source)

        # Episode state (set during reset)
        self.atom: str = atom
        self.env: OSMV2WorkspaceSandboxEnv | None = None
        self.export = None
        self.turn = 0
        self.done = False
        self.base_record: dict[str, Any] | None = None
        self.base_score: float | None = None
        self.base_plan_bank_path: str | None = base_plan_bank_path
        self.current_context: dict[str, Any] = {}

# --- PLACEHOLDER_METHODS ---

    def _resolve_instance(self, seed: int | None):
        """Resolve the instance config for this episode.

        In dataset mode, selects a row from prompt JSONL based on seed.
        In direct mode, uses the fixed data_source.
        """
        if self._instance_rows is not None:
            idx = (seed or 0) % len(self._instance_rows)
            row = self._instance_rows[idx]
            cfg = row.get("metadata", {}).get("config", {})
            self.atom = str(cfg.get("atom", self.default_atom))
            data_source = _repo_path(cfg.get("data_source") or cfg.get("data_source_path", ""))
            self.base_plan_bank_path = cfg.get("base_plan_bank_path", self.default_base_plan_bank_path)
            self.max_turns = int(cfg.get("max_turns", self.max_turns))
            if cfg.get("workspace_prompt_config"):
                self.prompt_config_path = _repo_path(cfg["workspace_prompt_config"])
                self.scaffold = OSMV2WorkspaceScaffold.from_file(self.prompt_config_path)
            return data_source
        else:
            self.atom = self.default_atom
            return self._fixed_data_source

    def get_instructions(self) -> str:
        return self.scaffold.system_prompt()

    def reset(self, seed=None):
        import uuid

        Env.reset(self, seed)
        self.turn = 0
        self.done = False
        self._last_plan_ids: list[str] = []
        self.base_record = None
        self.base_score = None

        data_source = self._resolve_instance(seed)
        if self.atom not in {"BuildPlan", "ImprovePlan"}:
            raise ValueError(f"Unknown atom: {self.atom}")

        self.export = load_export(data_source)
        run_id = f"roll_atomic_{self.atom}_{self.export.task}_{self.export.instance_id}_{seed}_{uuid.uuid4().hex[:8]}"
        self.env = OSMV2WorkspaceSandboxEnv(
            data_source,
            run_root=self.run_root,
            run_id=run_id,
            isolation=self.isolation,
            step_timeout_seconds=self.step_timeout_seconds,
        )
        self.env.reset()

        if self.atom == "ImprovePlan":
            self.base_record = self._sample_base_plan(seed)
            baseline_path = self.env.outputs_dir / "baseline_plan.json"
            write_plan(baseline_path, list(self.base_record["candidate_ids"]))
            write_plan(self.env.outputs_dir / "final_plan.json", list(self.base_record["candidate_ids"]))
            try:
                baseline_eval = self.env.evaluate_plan(baseline_path)
                self.base_score = score_value(baseline_eval)
                self.base_record["score"] = self.base_score
                self.base_record["score_detail"] = baseline_eval.get("score", {})
            except Exception:
                self.base_score = _as_float(self.base_record.get("score"), 0.0)
            self._last_plan_ids = sorted(str(x) for x in self.base_record["candidate_ids"])

        context = self._atomic_context()
        self.current_context = context
        (self.env.work_dir / "atomic_context.json").write_text(
            json.dumps(context, indent=2, ensure_ascii=False, default=str) + "\n"
        )

        initial_prompt = self.scaffold.render_initial_atomic_observation(
            template_vars=self._template_vars(),
            atomic_context=context,
            state=self._workspace_state(turn=0),
            char_limit=self.max_obs_chars,
        )
        observation = clip_text(
            "\n\n".join([initial_prompt, _render_terminal_observation([], initial=True)]),
            self.max_obs_chars,
        )
        info = {
            "env_instruction": self.scaffold.system_prompt(),
            "atomic_context": context,
            "run_dir": str(self.env.run_dir),
        }
        return observation, info

    def step(self, action: str):
        if self.env is None:
            raise RuntimeError("reset must be called before step")
        if self.done:
            return "Episode is already finished.", 0.0, True, False, {}

        self.turn += 1
        commands, task_complete, json_parsed = _extract_commands_terminus2(action)

        results: list[dict[str, Any]] = []
        for cmd in commands:
            result = self.env.execute(cmd)
            results.append({"command": cmd, "returncode": result.get("returncode"), "output": result.get("output", "")})

        if not commands:
            results = [{"command": "", "returncode": 2, "output": "No commands found in response.", "isolation": self.isolation}]

        last_result = results[-1] if results else {"returncode": 2, "output": ""}
        legacy_finish = self._is_finish(last_result)
        finished = task_complete or legacy_finish
        reached_max = self.turn >= self.max_turns

        if finished or reached_max:
            self.done = True
            reward, info = self._finalize()
            finish_reason = "submitted" if finished else "max_turns"
            info["finish_reason"] = finish_reason
            info["turns"] = self.turn
            info["command"] = "; ".join(cmd for cmd in commands) if commands else ""
            observation = _render_terminal_observation(results)
            metrics = self._build_metrics(info)
            info["metrics"] = metrics
            info["metrics_agg_mode"] = {"success": "last", "atomic_success": "last", "final_score": "last", "selected_count": "last"}
            terminated = True
            truncated = not finished
            return observation, reward, terminated, truncated, info

        observation = _render_terminal_observation(results)

        # Intermediate step shaping reward
        step_reward = 0.0
        # +0.005 for valid Terminus-2 JSON format
        if json_parsed and commands:
            step_reward += 0.005
        # +0.005 if all commands executed successfully
        if results and all(r.get("returncode") == 0 for r in results):
            step_reward += 0.005
        # +0.005 if plan was modified this step
        final_plan = self.env.outputs_dir / "final_plan.json"
        current_plan_ids = sorted(str(x) for x in self._plan_ids(final_plan))
        if current_plan_ids and current_plan_ids != self._last_plan_ids:
            step_reward += 0.005
        self._last_plan_ids = current_plan_ids

        info = {"finish_reason": "running", "turns": self.turn, "command": "; ".join(commands)}
        info["metrics"] = {"success": False, "atomic_success": False, "final_score": 0.0, "selected_count": 0}
        info["metrics_agg_mode"] = {"success": "last", "atomic_success": "last", "final_score": "last", "selected_count": "last"}
        return observation, step_reward, False, False, info

    def _is_finish(self, result: dict[str, Any]) -> bool:
        output = str(result.get("output", ""))
        lines = output.lstrip().splitlines()
        return bool(
            any(line.strip() == FINISH_SENTINEL for line in lines)
            and result.get("returncode") == 0
        )

    def _finalize(self) -> tuple[float, dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("reset must be called before finalizing")
        final_plan = self.env.outputs_dir / "final_plan.json"
        if final_plan.exists():
            try:
                evaluation = self.env.evaluate_plan(final_plan)
            except Exception as exc:
                evaluation = {"valid": False, "score": {"score": 0.0, "selected_count": 0}, "error": repr(exc)}
        else:
            evaluation = {"valid": False, "score": {"score": 0.0, "selected_count": 0}, "error": "missing"}

        final_score = score_value(evaluation)
        final_count = selected_count(evaluation)
        evaluator_valid = bool(evaluation.get("valid"))
        valid_non_empty = evaluator_valid and final_count > 0
        delta = final_score - self.base_score if self.base_score is not None else None

        final_candidate_ids = self._plan_ids(final_plan)
        baseline_ids = list(self.base_record.get("candidate_ids", [])) if self.base_record else []
        plan_changed = True
        if self.atom == "ImprovePlan":
            plan_changed = sorted(str(x) for x in final_candidate_ids) != sorted(str(x) for x in baseline_ids)

        task_name = self.export.task if self.export else ""
        normalized = normalize_score(final_score, task_name)

        if self.atom == "ImprovePlan":
            if not valid_non_empty or not plan_changed:
                reward = -0.2
            else:
                reward = normalized
        else:
            reward = normalized if valid_non_empty else -0.3

        atomic_success = bool(valid_non_empty and (self.atom != "ImprovePlan" or plan_changed))
        info = {
            "atom": self.atom,
            "task_name": self.export.task,
            "instance_id": self.export.instance_id,
            "valid": atomic_success,
            "atomic_success": atomic_success,
            "evaluator_valid": evaluator_valid,
            "plan_changed": plan_changed,
            "selected_count": final_count,
            "final_score": final_score,
            "base_score": self.base_score,
            "delta_score": delta,
            "reward": reward,
            "evaluation": evaluation,
        }
        return reward, info

    def _build_metrics(self, info: dict[str, Any]) -> dict[str, Any]:
        return {
            "success": bool(info.get("atomic_success")),
            "atomic_success": bool(info.get("atomic_success")),
            "final_score": info.get("final_score", 0.0),
            "selected_count": info.get("selected_count", 0),
        }

    def _plan_ids(self, path: Path) -> list[str]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(errors="replace"))
        except json.JSONDecodeError:
            return []
        if isinstance(payload, list):
            return [str(item.get("action_id", item)) if isinstance(item, dict) else str(item) for item in payload]
        if not isinstance(payload, dict):
            return []
        for key in ("candidate_ids", "action_ids", "selected_candidate_ids"):
            value = payload.get(key)
            if isinstance(value, list):
                return [str(item.get("action_id", item)) if isinstance(item, dict) else str(item) for item in value]
        return []

    def _sample_base_plan(self, seed: int | None) -> dict[str, Any]:
        import random

        if not self.base_plan_bank_path:
            raise ValueError("ImprovePlan requires base_plan_bank_path")
        rows = _load_jsonl(_repo_path(self.base_plan_bank_path))
        rows = [
            row for row in rows
            if row.get("valid", True)
            and row.get("candidate_ids")
            and row.get("task_name") == self.export.task
            and row.get("instance_id") == self.export.instance_id
        ]
        if not rows:
            raise ValueError(f"No compatible base plans for {self.export.task}/{self.export.instance_id}")
        rng = random.Random(seed if seed is not None else 0)
        return dict(rng.choice(rows))

    def _atomic_context(self) -> dict[str, Any]:
        current_plan = list(self.base_record.get("candidate_ids", [])) if self.base_record else []
        return {
            "atom": self.atom,
            "goal": self._goal_text(),
            "task": self.export.task,
            "instance_id": self.export.instance_id,
            "current_plan": current_plan,
            "current_score": self.base_score,
            "budget": float(self.export.budget) if self.export.budget is not None else None,
            "max_steps": int(self.export.max_steps) if self.export.max_steps is not None else None,
            "editable_candidates": top_editable_candidates(
                self.export, current_plan=current_plan, top_k=self.editable_top_k
            ),
            "finish_command": f"echo {FINISH_SENTINEL}",
            "final_plan_path": "outputs/final_plan.json",
        }

    def _goal_text(self) -> str:
        if self.atom == "BuildPlan":
            return "Create a valid non-empty initial plan at outputs/final_plan.json, evaluate it, then finish."
        return (
            "Improve the preloaded baseline plan in outputs/final_plan.json with a concrete "
            "candidate_ids edit. Keep the plan valid and non-empty, evaluate it, then finish."
        )

    def _template_vars(self) -> dict[str, Any]:
        metadata = getattr(self.export, "metadata", {}) or {}
        task_alias = str(self.export.task).split("_", 1)[0]
        return {
            "task": self.export.task,
            "task_name": self.export.task,
            "task_alias": task_alias,
            "instance_id": self.export.instance_id,
            "size_bucket": metadata.get("size_bucket"),
            "split": metadata.get("split", ""),
        }

    def _workspace_state(self, turn: int) -> dict[str, Any]:
        if self.env is None:
            return {}
        final_plan = self.env.outputs_dir / "final_plan.json"
        plan_exists = final_plan.exists()
        plan_ids = self._plan_ids(final_plan) if plan_exists else []
        return {
            "task_name": self.export.task,
            "instance_id": self.export.instance_id,
            "turn": turn,
            "max_turns": self.max_turns,
            "atom": self.atom,
            "plan_status": {
                "path": "outputs/final_plan.json",
                "exists": plan_exists,
                "non_empty": bool(plan_ids),
                "selected_count": len(plan_ids),
            },
            "base_score": self.base_score,
        }

    def render(self, mode=None) -> str:
        if self.env is None:
            return ""
        return json.dumps(self._workspace_state(turn=self.turn), indent=2)

    def close(self):
        self.env = None
