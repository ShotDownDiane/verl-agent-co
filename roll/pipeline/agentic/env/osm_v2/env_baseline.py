"""OSM-v2 Baseline environment — no atomic task split (BuildPlan/ImprovePlan).

Single-phase environment: the model gets 30 turns to explore the workspace and
produce the best possible plan at outputs/final_plan.json. No baseline plan is
preloaded (always starts from scratch). Used for ablation testing against the
atomic two-phase setup.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from gem import Env

from roll.pipeline.agentic.env.osm_v2.deps.atomic_harness import (
    load_export,
    score_value,
    selected_count,
    top_editable_candidates,
)
from roll.pipeline.agentic.env.osm_v2.deps.scaffold import OSMV2WorkspaceScaffold
from roll.pipeline.agentic.env.osm_v2.deps.workspace_env import OSMV2WorkspaceSandboxEnv
from roll.pipeline.agentic.env.osm_v2.env import (
    FINISH_SENTINEL,
    MAX_OBS_CHARS,
    _extract_commands_terminus2,
    _load_jsonl,
    _render_terminal_observation,
    _repo_path,
    _roll_root,
    normalize_score,
)

ROOT = _roll_root()
DEFAULT_PROMPT_CONFIG = ROOT / "examples/qwen3-8B-osm_v2_atomic/data/scaffolds/osm_v2_terminus2.yaml"


class OsmV2BaselineEnv(Env):
    """Single-phase baseline: 30 turns, no atom split, always build from scratch.

    Reward = evaluator score if valid plan exists, else -1.
    """

    def __init__(
        self,
        data_source: str | None = None,
        data_source_path: str | None = None,
        prompt_data_path: str | None = None,
        max_turns: int = 30,
        run_root: str | None = None,
        isolation: str = "local",
        step_timeout_seconds: int = 30,
        workspace_prompt_config: str | None = None,
        editable_top_k: int = 80,
        max_obs_chars: int = MAX_OBS_CHARS,
        **kwargs: Any,
    ) -> None:
        self.max_turns = max_turns
        self.run_root = Path(
            run_root or os.environ.get("OSM_ATOMIC_RUN_ROOT")
            or Path(tempfile.gettempdir()) / "roll_osm_v2_baseline"
        ).resolve()
        self.isolation = isolation
        self.step_timeout_seconds = step_timeout_seconds
        prompt_cfg = workspace_prompt_config or str(DEFAULT_PROMPT_CONFIG)
        self.prompt_config_path = _repo_path(prompt_cfg)
        self.scaffold = OSMV2WorkspaceScaffold.from_file(self.prompt_config_path)
        self.max_obs_chars = max_obs_chars
        self.editable_top_k = editable_top_k

        self._instance_rows: list[dict[str, Any]] | None = None
        if prompt_data_path:
            self._instance_rows = _load_jsonl(_repo_path(prompt_data_path))
            if not self._instance_rows:
                raise ValueError(f"No rows in prompt_data_path: {prompt_data_path}")

        self._fixed_data_source: Path | None = None
        if not prompt_data_path:
            source = data_source or data_source_path
            if source is None:
                raise ValueError("Either data_source or prompt_data_path is required")
            self._fixed_data_source = _repo_path(source)

        self.env: OSMV2WorkspaceSandboxEnv | None = None
        self.export = None
        self.turn = 0
        self.done = False

    def _resolve_instance(self, seed: int | None) -> Path:
        if self._instance_rows is not None:
            idx = (seed or 0) % len(self._instance_rows)
            row = self._instance_rows[idx]
            cfg = row.get("metadata", {}).get("config", {})
            data_source = _repo_path(cfg.get("data_source") or cfg.get("data_source_path", ""))
            self.max_turns = int(cfg.get("max_turns", self.max_turns))
            return data_source
        return self._fixed_data_source

    def get_instructions(self) -> str:
        return self.scaffold.system_prompt()

    def reset(self, seed=None):
        Env.reset(self, seed)
        self.turn = 0
        self.done = False
        self._last_plan_ids: list[str] = []

        data_source = self._resolve_instance(seed)
        self.export = load_export(data_source)
        run_id = f"roll_baseline_{self.export.task}_{self.export.instance_id}_{seed}_{uuid.uuid4().hex[:8]}"
        self.env = OSMV2WorkspaceSandboxEnv(
            data_source,
            run_root=self.run_root,
            run_id=run_id,
            isolation=self.isolation,
            step_timeout_seconds=self.step_timeout_seconds,
        )
        self.env.reset()

        observation = _render_terminal_observation([], initial=True)
        context = {
            "goal": "Create a valid high-scoring plan at outputs/final_plan.json, evaluate it, then finish.",
            "task": self.export.task,
            "instance_id": self.export.instance_id,
            "budget": float(self.export.budget) if self.export.budget is not None else None,
            "editable_candidates": top_editable_candidates(
                self.export, current_plan=[], top_k=self.editable_top_k
            ),
            "finish_command": f"echo {FINISH_SENTINEL}",
            "final_plan_path": "outputs/final_plan.json",
        }
        (self.env.work_dir / "context.json").write_text(
            json.dumps(context, indent=2, ensure_ascii=False, default=str) + "\n"
        )
        info = {
            "env_instruction": self.scaffold.system_prompt(),
            "context": context,
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
            results = [{"command": "", "returncode": 2, "output": "No commands found in response."}]

        last_result = results[-1] if results else {"returncode": 2, "output": ""}
        legacy_finish = FINISH_SENTINEL in str(last_result.get("output", "")) and last_result.get("returncode") == 0
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
            info["metrics_agg_mode"] = {"success": "last", "final_score": "last"}
            terminated = True
            truncated = not finished
            return observation, reward, terminated, truncated, info

        observation = _render_terminal_observation(results)

        # Intermediate step shaping reward
        step_reward = 0.0
        if json_parsed and commands:
            step_reward += 0.005
        if results and all(r.get("returncode") == 0 for r in results):
            step_reward += 0.005
        final_plan = self.env.outputs_dir / "final_plan.json"
        current_plan_ids = sorted(str(x) for x in self._plan_ids(final_plan))
        if current_plan_ids and current_plan_ids != self._last_plan_ids:
            step_reward += 0.005
        self._last_plan_ids = current_plan_ids

        info = {"finish_reason": "running", "turns": self.turn, "command": "; ".join(commands)}
        info["metrics"] = {"success": False, "final_score": 0.0}
        info["metrics_agg_mode"] = {"success": "last", "final_score": "last"}
        return observation, step_reward, False, False, info

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

        task_name = self.export.task if self.export else ""
        reward = normalize_score(final_score, task_name) if valid_non_empty else -0.3

        info = {
            "task_name": self.export.task,
            "instance_id": self.export.instance_id,
            "valid": valid_non_empty,
            "success": valid_non_empty,
            "evaluator_valid": evaluator_valid,
            "selected_count": final_count,
            "final_score": final_score,
            "reward": reward,
            "evaluation": evaluation,
        }
        return reward, info

    def _build_metrics(self, info: dict[str, Any]) -> dict[str, Any]:
        return {
            "success": bool(info.get("success")),
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

    def close(self):
        if self.env is not None:
            if hasattr(self.env, "close"):
                self.env.close()
            self.env = None
