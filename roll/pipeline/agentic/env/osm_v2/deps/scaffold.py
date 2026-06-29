"""Shared OSM-v2 workspace agent scaffold rendering utilities.

The mini-swe-agent evaluator and verl atomic environments use different runtime
loops, but they should expose the same workspace rules and prompt semantics to
the model.  This module keeps those prompt/rendering details in one place.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "osm_v2_workspace_scaffold_v1"
DEFAULT_FINISH_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def stable_json_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clip_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + f"\n...<elided {len(text) - limit} chars>...\n" + text[-tail:]


def _preview_mapping(payload: Mapping[str, Any], *, keys: tuple[str, ...]) -> dict[str, Any]:
    preview = {key: payload[key] for key in keys if key in payload}
    if preview:
        return preview
    return {key: payload[key] for key in list(payload)[:6]}


def _preview_list(items: Any, *, limit: int, keys: tuple[str, ...] | None = None) -> tuple[list[Any], int | None]:
    if not isinstance(items, list):
        return [], None
    preview: list[Any] = []
    for item in items[:limit]:
        if isinstance(item, Mapping) and keys is not None:
            preview.append(_preview_mapping(item, keys=keys))
        else:
            preview.append(item)
    truncated = len(items) - limit if len(items) > limit else 0
    return preview, truncated


def _compact_json_lists(payload: Any, *, limit: int = 12) -> tuple[Any, bool]:
    if isinstance(payload, list):
        changed = False
        values = []
        for item in payload:
            compact_item, item_changed = _compact_json_lists(item, limit=limit)
            values.append(compact_item)
            changed = changed or item_changed
        return values, changed
    if not isinstance(payload, dict):
        return payload, False

    compact: dict[str, Any] = {}
    changed = False
    id_list_keys = {"candidate_ids", "action_ids", "selected_candidate_ids"}
    for key, value in payload.items():
        if key in id_list_keys and isinstance(value, list):
            compact[f"{key}_count"] = len(value)
            compact[f"{key}_preview"] = value[:limit]
            compact[f"{key}_truncated"] = max(0, len(value) - limit)
            changed = True
            continue
        compact_value, value_changed = _compact_json_lists(value, limit=limit)
        compact[key] = compact_value
        changed = changed or value_changed
    return compact, changed


def compact_observation_output(output: str, *, limit: int = 12) -> str:
    stripped = output.strip()
    if not stripped or stripped[0] not in "[{":
        return output
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return output
    compact_payload, changed = _compact_json_lists(payload, limit=limit)
    if not changed:
        return output
    return json.dumps(compact_payload, indent=2, ensure_ascii=False, default=str)


def _render_template(template: str, variables: Mapping[str, Any]) -> str:
    try:
        from jinja2 import Environment
    except Exception:  # pragma: no cover - jinja2 is available in normal runner envs.
        rendered = template
        for key, value in variables.items():
            rendered = rendered.replace("{{" + key + "}}", str(value))
            rendered = rendered.replace("{{ " + key + " }}", str(value))
        return rendered

    env = Environment(autoescape=False)
    return env.from_string(template).render(**dict(variables))


def load_workspace_prompt_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt config must be a mapping: {path}")
    agent = payload.get("agent", {})
    model = payload.get("model", {})
    if not isinstance(agent, dict) or not isinstance(model, dict):
        raise ValueError(f"Prompt config must contain agent/model mappings: {path}")
    for key in ("system_template", "instance_template"):
        if not agent.get(key):
            raise ValueError(f"Prompt config missing agent.{key}: {path}")
    return payload


@dataclass(frozen=True)
class OSMV2WorkspaceScaffold:
    """Renderer for the shared mini-swe style OSM-v2 workspace scaffold."""

    prompt_config_path: Path
    prompt_config: dict[str, Any]
    prompt_config_hash: str

    @classmethod
    def from_file(cls, path: str | Path) -> "OSMV2WorkspaceScaffold":
        resolved = Path(path)
        return cls(
            prompt_config_path=resolved,
            prompt_config=load_workspace_prompt_config(resolved),
            prompt_config_hash=file_sha256(resolved),
        )

    @property
    def agent_config(self) -> dict[str, Any]:
        return dict(self.prompt_config.get("agent", {}))

    @property
    def model_config(self) -> dict[str, Any]:
        return dict(self.prompt_config.get("model", {}))

    def trace_info(self) -> dict[str, str]:
        return {
            "scaffold_schema_version": SCHEMA_VERSION,
            "scaffold_name": "mini-swe-agent-workspace",
            "prompt_config_path": str(self.prompt_config_path),
            "prompt_config_hash": self.prompt_config_hash,
        }

    def system_prompt(self) -> str:
        return str(self.agent_config.get("system_template", "")).strip()

    def render_instance_prompt(self, variables: Mapping[str, Any]) -> str:
        return _render_template(str(self.agent_config.get("instance_template", "")), variables).strip()

    def render_tool_observation(self, result: Mapping[str, Any]) -> str:
        template = str(self.model_config.get("observation_template", ""))
        output = compact_observation_output(str(result.get("output", "")))
        payload = {
            "returncode": result.get("returncode", 1),
            "output": output,
            "exception_info": result.get("exception_info") or "",
        }
        if template:
            return _render_template(template, {"output": payload}).strip()
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def render_format_error(self, error: str) -> str:
        template = str(self.model_config.get("format_error_template", "{{ error }}"))
        return _render_template(template, {"error": error}).strip()

    def _atomic_prompt_context(self, context: Mapping[str, Any]) -> dict[str, Any]:
        current_plan = context.get("current_plan")
        plan_preview, plan_truncated = _preview_list(current_plan, limit=12)
        editable_preview, editable_truncated = _preview_list(
            context.get("editable_candidates"),
            limit=10,
            keys=(
                "candidate_id",
                "action_id",
                "id",
                "site_id",
                "block_id",
                "rank",
                "score",
                "cost",
                "benefit",
                "type",
                "config",
            ),
        )
        return {
            "atom": context.get("atom"),
            "task": context.get("task"),
            "instance_id": context.get("instance_id"),
            "size_bucket": context.get("size_bucket"),
            "budget": context.get("budget"),
            "max_steps": context.get("max_steps"),
            "current_score": context.get("current_score"),
            "current_plan_selected_count": len(current_plan) if isinstance(current_plan, list) else None,
            "current_plan_preview": plan_preview,
            "current_plan_truncated": plan_truncated,
            "editable_candidate_count": len(context.get("editable_candidates", []))
            if isinstance(context.get("editable_candidates"), list)
            else None,
            "editable_candidates_preview": editable_preview,
            "editable_candidates_truncated": editable_truncated,
            "atomic_context_path": "work/atomic_context.json",
            "final_plan_path": context.get("final_plan_path"),
            "baseline_plan_path": context.get("baseline_plan_path"),
            "finish_command": context.get("finish_command"),
        }

    def atomic_objective_and_reward(self, context: Mapping[str, Any]) -> tuple[str, str]:
        atom = str(context.get("atom", ""))
        if atom == "BuildPlan":
            objective = (
                "In this episode, your goal is to create a valid, non-empty initial "
                "plan for the given workspace. The plan does not need to be globally "
                "optimal, but it should be executable, reasonable, and evaluated."
            )
        elif atom == "ImprovePlan":
            objective = (
                "In this episode, your goal is to improve the preloaded plan for the "
                "given workspace. Make and evaluate a concrete candidate_ids edit that "
                "aims to increase the score while keeping the plan valid and non-empty. "
                "An unchanged copy of the preloaded plan is valid but not a successful "
                "improvement."
            )
        else:
            objective = str(context.get("goal", "Solve the given workspace task."))
        return objective, ""

    def render_atomic_goal_text(self, context: Mapping[str, Any]) -> str:
        objective, _ = self.atomic_objective_and_reward(context)
        return objective

    def render_atomic_instance_prompt(self, variables: Mapping[str, Any], context: Mapping[str, Any]) -> str:
        text = self.render_instance_prompt(variables)
        default_goal = "Create a valid high-scoring plan at `outputs/final_plan.json`."
        atomic_goal = self.render_atomic_goal_text(context)
        if default_goal in text:
            return text.replace(default_goal, atomic_goal, 1)
        return "\n\n".join([text, "## Goal", atomic_goal])

    def render_atomic_task_block(self, context: Mapping[str, Any]) -> str:
        return self.render_atomic_goal_text(context)

    def render_workspace_state_card(self, state: Mapping[str, Any]) -> str:
        plan = state.get("plan_status", {}) if isinstance(state.get("plan_status"), Mapping) else {}
        evaluation = state.get("evaluation_status", {}) if isinstance(state.get("evaluation_status"), Mapping) else {}
        atomic = state.get("atomic_status", {}) if isinstance(state.get("atomic_status"), Mapping) else {}
        errors = evaluation.get("last_hard_errors") or []
        warnings = evaluation.get("last_warnings") or []
        return "\n".join(
            [
                "[Workspace State]",
                (
                    f"- plan: exists={plan.get('exists')} | non_empty={plan.get('non_empty')} | "
                    f"selected_count={plan.get('selected_count')} | needs_eval={plan.get('changed_after_last_eval')}"
                ),
                f"- evaluation: has_evaluated={evaluation.get('has_evaluated')} | valid={evaluation.get('last_valid')} | score={evaluation.get('last_score')}",
                f"- hard_errors: {errors[:3] if errors else 'none'}",
                f"- warnings: {warnings[:3] if warnings else 'none'}",
                f"- next: {state.get('next_required_action')}",
            ]
        )

    def render_initial_atomic_observation(
        self,
        *,
        template_vars: Mapping[str, Any],
        atomic_context: Mapping[str, Any],
        state: Mapping[str, Any],
        char_limit: int,
    ) -> str:
        text = "\n\n".join(
            [
                self.render_atomic_instance_prompt(template_vars, atomic_context),
                self.render_workspace_state_card(state),
            ]
        )
        return clip_text(text, char_limit)

    def render_step_observation(
        self,
        *,
        result: Mapping[str, Any],
        state: Mapping[str, Any],
        terminal: bool,
        final_info: Mapping[str, Any] | None,
        char_limit: int,
        persistent_context: str = "",
    ) -> str:
        parts = []
        if persistent_context:
            parts.append(persistent_context)
        command = str(result.get("command", "")).strip()
        if command:
            parts.append("[Last Bash Command]\n\n```bash\n" + clip_text(command, 1200) + "\n```")
        parts.extend(["[Tool Result]\n\n" + self.render_tool_observation(result), self.render_workspace_state_card(state)])
        if terminal and final_info is not None:
            compact_info, _ = _compact_json_lists(dict(final_info), limit=12)
            parts.append(
                "Terminal evaluation:\n\n```json\n"
                + json.dumps(compact_info, indent=2, ensure_ascii=False, default=str)
                + "\n```"
            )
        return clip_text("\n\n".join(parts), char_limit)
