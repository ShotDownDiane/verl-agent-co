"""Unified OSM v2 online sandbox.

This environment is the v2-aligned tool interface for Road, EV Charging, and
Urban Planning OSM instances. It consumes a CityTaskExport produced by
``instance_builder.py`` and uses the shared scoring function also used by
heuristic and DRL baselines.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ..common.export_schema import (
    CityTaskExport,
    candidate_to_mapping,
    export_from_mapping,
)
from ..common.schemas import CityStepResult
from .scoring import score_osm_v2_plan

TOOL_NAMES = frozenset({"query_state", "query_candidates", "run_python_analysis", "submit"})

DEFAULT_MAX_TOOL_CALLS = 40
DEFAULT_MAX_PYTHON_CALLS = 5
DEFAULT_MAX_SUBMIT_OPS = 15
DEFAULT_PYTHON_TIMEOUT = 10
DEFAULT_QUERY_CANDIDATE_LIMIT = 50
DEFAULT_STATE_DETAIL_LIMIT = 10
MAX_STDOUT_CHARS = 2000
MAX_STDERR_CHARS = 1000
MAX_OUTPUT_FILE_PREVIEW_CHARS = 1000
MAX_OUTPUT_FILE_PREVIEWS = 5


def _build_sandbox_preamble(sandbox_root: str) -> str:
    return (
        "import builtins as _b, os as _os, os.path as _op, io as _io\n"
        f"_SANDBOX_ROOT = _op.realpath({sandbox_root!r})\n"
        "def _check(p):\n"
        "    r = _op.realpath(str(p))\n"
        "    if not (r == _SANDBOX_ROOT or r.startswith(_SANDBOX_ROOT + _os.sep)):\n"
        "        raise PermissionError(f'Access denied: {p}')\n"
        "_orig_open = _b.open\n"
        "def _safe_open(f, *a, **kw):\n"
        "    _check(f); return _orig_open(f, *a, **kw)\n"
        "_b.open = _safe_open\n"
        "_orig_io_open = _io.open\n"
        "def _safe_io_open(f, *a, **kw):\n"
        "    _check(f); return _orig_io_open(f, *a, **kw)\n"
        "_io.open = _safe_io_open\n"
        f"_os.chdir({sandbox_root!r})\n"
    )


class OSMV2OnlineSandboxEnv:
    """Unified v2 sandbox with items + plan_id submit schema."""

    def __init__(
        self,
        data_source: Mapping[str, Any] | CityTaskExport,
        *,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_python_calls: int = DEFAULT_MAX_PYTHON_CALLS,
        max_submit_ops: int = DEFAULT_MAX_SUBMIT_OPS,
        python_timeout_seconds: int = DEFAULT_PYTHON_TIMEOUT,
        invalid_action_penalty: float = -0.5,
    ) -> None:
        self._data_source = data_source
        self._max_tool_calls = int(max_tool_calls)
        self._max_python_calls = int(max_python_calls)
        self._max_submit_ops = int(max_submit_ops)
        self._python_timeout = int(python_timeout_seconds)
        self._invalid_action_penalty = float(invalid_action_penalty)

        self._export: CityTaskExport | None = None
        self._plans: dict[str, list[str]] = {"main": []}
        self._tool_calls_used = 0
        self._python_calls_used = 0
        self._submit_ops_used = 0
        self._full_state_returned = False
        self._terminated = False
        self._truncated = False
        self._sandbox_dir: tempfile.TemporaryDirectory | None = None

    @property
    def export(self) -> CityTaskExport:
        if self._export is None:
            raise RuntimeError("Environment has not been reset.")
        return self._export

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    def close(self) -> None:
        if self._sandbox_dir is not None:
            self._sandbox_dir.cleanup()
            self._sandbox_dir = None

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self.close()
        if isinstance(self._data_source, CityTaskExport):
            self._export = self._data_source
        elif isinstance(self._data_source, (str, Path)):
            with Path(self._data_source).open() as f:
                self._export = export_from_mapping(json.load(f))
        else:
            self._export = export_from_mapping(self._data_source)
        self._plans = {"main": []}
        self._tool_calls_used = 0
        self._python_calls_used = 0
        self._submit_ops_used = 0
        self._full_state_returned = False
        self._terminated = False
        self._truncated = False
        self._sandbox_dir = tempfile.TemporaryDirectory(prefix="osm_v2_sandbox_")
        self._write_sandbox_files()
        return None, {"instance_id": self.export.instance_id, "task": self.export.task}

    def step(self, tool_call: dict[str, Any]) -> CityStepResult:
        if self._terminated or self._truncated:
            return CityStepResult(
                state=None,
                reward=0.0,
                terminated=True,
                truncated=self._truncated,
                info={"valid": False, "reason": "already_done"},
            )
        tool = str(tool_call.get("tool", "")).strip()
        args = tool_call.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        if tool not in TOOL_NAMES:
            return self._error_result(f"Unknown tool '{tool}'. Available: {sorted(TOOL_NAMES)}")

        self._tool_calls_used += 1
        if self._tool_calls_used > self._max_tool_calls:
            self._truncated = True
            return CityStepResult(
                state=None,
                reward=0.0,
                terminated=False,
                truncated=True,
                info={"valid": False, "reason": "tool_call_limit", "tool_result": {"error": "Tool call limit reached."}},
            )

        handler = {
            "query_state": self._handle_query_state,
            "query_candidates": self._handle_query_candidates,
            "run_python_analysis": self._handle_run_python,
            "submit": self._handle_submit,
        }[tool]
        return handler(args)

    def _handle_query_state(self, args: dict[str, Any]) -> CityStepResult:
        detail = bool(args.get("detail", False))
        plan_id = str(args.get("plan_id", "main"))
        result = {
            "task": self.export.task,
            "instance_id": self.export.instance_id,
            "metadata": self.export.metadata,
            "workspace_files": self._workspace_files(),
            "candidate_schema": self._candidate_schema(),
            "plan_id": plan_id,
            "working_plan": self._plan_summary(plan_id),
            "score_estimate": score_osm_v2_plan(self.export, self._plans.get(plan_id, [])),
            "budgets": self._budgets(),
            "candidate_count": len(self.export.candidate_actions),
            "demand_count": len(self.export.demand),
            "budget_info": {
                "budget": self.export.budget,
                "max_steps": self.export.max_steps,
            },
            "next_step_hint": self._next_step_hint(plan_id),
        }
        if "plan_ids" in args and isinstance(args["plan_ids"], list):
            result["plans"] = {
                str(pid): self._plan_summary(str(pid))
                for pid in args["plan_ids"]
            }
        if detail:
            result["candidates"] = self._candidate_list(limit=DEFAULT_STATE_DETAIL_LIMIT)
            result["demand"] = {
                "count": len(self.export.demand),
                "zones": self.export.demand[:DEFAULT_STATE_DETAIL_LIMIT],
            }
            result["existing_assets"] = self.export.existing_assets[:DEFAULT_STATE_DETAIL_LIMIT]
            result["detail_limit"] = DEFAULT_STATE_DETAIL_LIMIT
            self._full_state_returned = True
        return self._ok_result(result)

    def _handle_query_candidates(self, args: dict[str, Any]) -> CityStepResult:
        ids = args.get("candidate_ids")
        limit = args.get("limit")
        offset = int(args.get("offset", 0) or 0)
        candidate_map = {candidate.action_id: candidate for candidate in self.export.candidate_actions}
        if ids is None:
            candidates = self.export.candidate_actions[offset:]
            requested_limit = limit if isinstance(limit, int) and limit >= 0 else DEFAULT_QUERY_CANDIDATE_LIMIT
            effective_limit = min(int(requested_limit), DEFAULT_QUERY_CANDIDATE_LIMIT)
            candidates = candidates[:effective_limit]
            return self._ok_result(
                {
                    "candidates": [self._candidate_detail(candidate) for candidate in candidates],
                    "offset": offset,
                    "returned": len(candidates),
                    "total": len(self.export.candidate_actions),
                    "limit": effective_limit,
                    "truncated": offset + len(candidates) < len(self.export.candidate_actions),
                    "next_offset": offset + len(candidates),
                }
            )
        if not isinstance(ids, list):
            return self._error_result("candidate_ids must be a list")
        original_count = len(ids)
        ids = ids[:DEFAULT_QUERY_CANDIDATE_LIMIT]
        results = []
        for cid in ids:
            candidate = candidate_map.get(str(cid))
            if candidate is None:
                results.append({"candidate_id": str(cid), "error": "not found"})
            else:
                results.append(self._candidate_detail(candidate))
        return self._ok_result(
            {
                "candidates": results,
                "returned": len(results),
                "requested": original_count,
                "truncated": original_count > len(results),
                "limit": DEFAULT_QUERY_CANDIDATE_LIMIT,
            }
        )

    def _handle_run_python(self, args: dict[str, Any]) -> CityStepResult:
        if self._python_calls_used >= self._max_python_calls:
            return self._error_result(f"Python call limit reached ({self._max_python_calls}).")
        self._python_calls_used += 1
        code = args.get("code", "")
        if not isinstance(code, str) or not code.strip():
            return self._error_result("code must be a non-empty string")
        assert self._sandbox_dir is not None
        sandbox_root = self._sandbox_dir.name
        outputs_dir = os.path.join(sandbox_root, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        full_code = _build_sandbox_preamble(sandbox_root) + "\n" + code
        try:
            proc = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._python_timeout,
                cwd=sandbox_root,
            )
            output: dict[str, Any] = {"returncode": proc.returncode}
            if proc.stdout:
                output["stdout"] = proc.stdout[:MAX_STDOUT_CHARS]
                output["stdout_chars"] = len(proc.stdout)
                output["stdout_truncated"] = len(proc.stdout) > MAX_STDOUT_CHARS
            if proc.stderr:
                output["stderr"] = proc.stderr[:MAX_STDERR_CHARS]
                output["stderr_chars"] = len(proc.stderr)
                output["stderr_truncated"] = len(proc.stderr) > MAX_STDERR_CHARS
            output_files = []
            for fname in sorted(os.listdir(outputs_dir)):
                if len(output_files) >= MAX_OUTPUT_FILE_PREVIEWS:
                    break
                fpath = os.path.join(outputs_dir, fname)
                if os.path.isfile(fpath) and os.path.getsize(fpath) < 10000:
                    with open(fpath) as f:
                        content = f.read()
                        output_files.append(
                            {
                                "name": fname,
                                "content": content[:MAX_OUTPUT_FILE_PREVIEW_CHARS],
                                "chars": len(content),
                                "truncated": len(content) > MAX_OUTPUT_FILE_PREVIEW_CHARS,
                            }
                        )
            if output_files:
                output["output_files"] = output_files
            output["workspace_files"] = self._workspace_files()
            return self._ok_result(output)
        except subprocess.TimeoutExpired:
            return self._error_result(f"Python code timed out ({self._python_timeout}s).")
        except Exception as exc:
            return self._error_result(f"Python execution error: {exc}")

    def _handle_submit(self, args: dict[str, Any]) -> CityStepResult:
        op = str(args.get("op", "")).strip().lower()
        if op not in {"set", "add", "remove", "finish"}:
            return self._error_result('submit op must be one of: "set", "add", "remove", "finish"')
        plan_id = str(args.get("plan_id", "main"))
        self._plans.setdefault(plan_id, [])
        if op == "finish":
            return self._handle_finish(plan_id)
        if self._submit_ops_used >= self._max_submit_ops:
            return self._error_result(f"Submit operation limit reached ({self._max_submit_ops}).")
        self._submit_ops_used += 1
        ids = self._ids_from_items(args)
        if ids is None:
            return self._error_result("submit requires items as a list")
        if op == "set":
            self._plans[plan_id] = self._validated_ids(ids)["ids"]
        elif op == "add":
            validated = self._validated_ids(ids)["ids"]
            existing = set(self._plans[plan_id])
            self._plans[plan_id].extend([action_id for action_id in validated if action_id not in existing])
        elif op == "remove":
            remove_set = set(str(action_id) for action_id in ids)
            self._plans[plan_id] = [action_id for action_id in self._plans[plan_id] if action_id not in remove_set]
        validation = self._validate_plan(self._plans[plan_id])
        self._write_working_plan_file()
        return self._ok_result(
            {
                "status": "ok",
                "op": op,
                "plan_id": plan_id,
                "working_plan": self._plan_summary(plan_id),
                "validation": validation,
                "score_estimate": score_osm_v2_plan(self.export, self._plans[plan_id]),
                "next_step_hint": self._next_step_hint(plan_id),
            }
        )

    def _handle_finish(self, plan_id: str) -> CityStepResult:
        plan = self._plans.get(plan_id, [])
        if not plan:
            return self._error_result("Cannot finish with an empty plan.")
        validation = self._validate_plan(plan)
        if validation["hard_errors"]:
            return self._error_result("; ".join(validation["hard_errors"]))
        scores = score_osm_v2_plan(self.export, plan)
        self._terminated = True
        tool_result = {
            **scores,
            "final_plan": self._plan_summary(plan_id),
            "validation": validation,
        }
        return CityStepResult(
            state=None,
            reward=float(scores.get("score", 0.0)),
            terminated=True,
            truncated=False,
            info={"valid": True, "tool_result": tool_result, **scores},
        )

    def _ids_from_items(self, args: dict[str, Any]) -> list[str] | None:
        if "items" in args:
            items = args["items"]
            if not isinstance(items, list):
                return None
            ids = []
            for item in items:
                if isinstance(item, str):
                    ids.append(item)
                elif isinstance(item, Mapping):
                    if item.get("action_id") is not None:
                        ids.append(str(item["action_id"]))
                    else:
                        entity = item.get("entity") if isinstance(item.get("entity"), Mapping) else {}
                        assignment = item.get("assignment") if isinstance(item.get("assignment"), Mapping) else {}
                        ids.append(self._action_id_from_entity(entity, assignment))
                else:
                    ids.append(str(item))
            return [action_id for action_id in ids if action_id]
        for legacy_key in ("road_ids", "station_ids", "candidate_ids"):
            if legacy_key in args:
                value = args[legacy_key]
                return [str(item) for item in value] if isinstance(value, list) else None
        if "assignments" in args and isinstance(args["assignments"], list):
            ids = []
            for assignment in args["assignments"]:
                if not isinstance(assignment, Mapping):
                    continue
                ids.append(self._action_id_from_entity(
                    {"id": assignment.get("block_id")},
                    {"land_use": assignment.get("land_use")},
                ))
            return [action_id for action_id in ids if action_id]
        return []

    def _action_id_from_entity(self, entity: Mapping[str, Any], assignment: Mapping[str, Any]) -> str:
        entity_id = str(entity.get("id", ""))
        config_id = str(assignment.get("id", assignment.get("charger_config_id", "")))
        land_use = str(assignment.get("land_use", ""))
        for candidate in self.export.candidate_actions:
            payload = candidate.payload
            candidate_entity = payload.get("entity") if isinstance(payload.get("entity"), Mapping) else {}
            candidate_assignment = payload.get("assignment") if isinstance(payload.get("assignment"), Mapping) else {}
            if entity_id and str(candidate_entity.get("id", payload.get("site_id", payload.get("block_id", "")))) != entity_id:
                continue
            if config_id and str(candidate_assignment.get("id", payload.get("charger_config_id", ""))) != config_id:
                continue
            if land_use and str(candidate_assignment.get("land_use", payload.get("land_use", ""))) != land_use:
                continue
            return candidate.action_id
        return ""

    def _validated_ids(self, action_ids: list[str]) -> dict[str, Any]:
        valid = {candidate.action_id for candidate in self.export.candidate_actions if candidate.is_feasible}
        out = []
        skipped = []
        for action_id in action_ids:
            aid = str(action_id)
            if aid in valid and aid not in out:
                out.append(aid)
            else:
                skipped.append(aid)
        return {"ids": out, "skipped": skipped}

    def _candidate_efficiency(self, action_id: str, candidate_map: Mapping[str, Any]) -> tuple[float, float, float]:
        candidate = candidate_map[action_id]
        cost = float(candidate.cost)
        weight = candidate.estimated_effects.get("served_demand_weight", 0.0)
        try:
            served_weight = float(weight or 0.0)
        except (TypeError, ValueError):
            served_weight = 0.0
        efficiency = served_weight / max(cost, 1e-9)
        return efficiency, served_weight, cost

    def _repair_suggestions(
        self,
        action_ids: list[str],
        candidate_map: Mapping[str, Any],
        *,
        total_cost: float,
    ) -> list[dict[str, Any]]:
        suggestions: list[dict[str, Any]] = []
        selected = [aid for aid in action_ids if aid in candidate_map]
        ranked_low = sorted(
            selected,
            key=lambda aid: (
                self._candidate_efficiency(aid, candidate_map)[0],
                self._candidate_efficiency(aid, candidate_map)[1],
                -self._candidate_efficiency(aid, candidate_map)[2],
            ),
        )

        if self.export.budget is not None and total_cost > self.export.budget:
            excess = total_cost - float(self.export.budget)
            removed_cost = 0.0
            remove_ids = []
            for aid in ranked_low:
                remove_ids.append(aid)
                removed_cost += float(candidate_map[aid].cost)
                if removed_cost >= excess:
                    break
            if remove_ids:
                suggestions.append(
                    {
                        "reason": "budget_exceeded",
                        "excess_cost": round(excess, 6),
                        "remove_action_ids": remove_ids[:20],
                        "hint": "Use submit(op=remove) with these action_ids, or submit(op=set) without them.",
                    }
                )

        if self.export.max_steps is not None and len(selected) > self.export.max_steps:
            excess_steps = len(selected) - int(self.export.max_steps)
            remove_ids = ranked_low[:excess_steps]
            if remove_ids:
                suggestions.append(
                    {
                        "reason": "step_limit_exceeded",
                        "excess_steps": excess_steps,
                        "remove_action_ids": remove_ids[:20],
                        "hint": "Remove the lowest-efficiency selected actions until selected_count <= max_steps.",
                    }
                )

        return suggestions

    def _validate_plan(self, action_ids: list[str]) -> dict[str, Any]:
        candidate_map = {candidate.action_id: candidate for candidate in self.export.candidate_actions}
        hard_errors = []
        warnings = []
        total_cost = sum(candidate_map[aid].cost for aid in action_ids if aid in candidate_map)
        if self.export.budget is not None and total_cost > self.export.budget:
            hard_errors.append(f"Budget exceeded: {total_cost:.2f} > {self.export.budget:.2f}")
        if self.export.max_steps is not None and len(action_ids) > self.export.max_steps:
            hard_errors.append(f"Step limit exceeded: {len(action_ids)} > {self.export.max_steps}")
        if self.export.task == "ev_charging":
            sites = [str(candidate_map[aid].payload.get("site_id")) for aid in action_ids if aid in candidate_map]
            duplicate_sites = sorted({site for site in sites if sites.count(site) > 1})
            if duplicate_sites:
                hard_errors.append(f"Duplicate charging site configs are not allowed: {duplicate_sites[:10]}")
        if self.export.task == "urban_planning":
            blocks = [str(candidate_map[aid].payload.get("block_id")) for aid in action_ids if aid in candidate_map]
            duplicate_blocks = sorted({block for block in blocks if blocks.count(block) > 1})
            if duplicate_blocks:
                hard_errors.append(f"Duplicate block assignments are not allowed: {duplicate_blocks[:10]}")
        remaining_budget = None
        if self.export.budget is not None:
            remaining_budget = round(float(self.export.budget) - total_cost, 6)
        remaining_steps = None
        if self.export.max_steps is not None:
            remaining_steps = int(self.export.max_steps) - len(action_ids)
        repair_suggestions = self._repair_suggestions(
            action_ids,
            candidate_map,
            total_cost=total_cost,
        )
        return {
            "hard_errors": hard_errors,
            "warnings": warnings,
            "total_cost": round(total_cost, 6),
            "budget": self.export.budget,
            "remaining_budget": remaining_budget,
            "max_steps": self.export.max_steps,
            "remaining_steps": remaining_steps,
            "repair_suggestions": repair_suggestions,
        }

    def _candidate_list(self, *, limit: int) -> list[dict[str, Any]]:
        return [self._candidate_brief(candidate) for candidate in self.export.candidate_actions[:limit]]

    def _candidate_brief(self, candidate: Any) -> dict[str, Any]:
        payload = candidate.payload
        effects = candidate.estimated_effects
        return {
            "id": candidate.action_id,
            "action_type": candidate.action_type,
            "label": candidate.label,
            "cost": candidate.cost,
            "entity": payload.get("entity"),
            "assignment": payload.get("assignment"),
            "served_demand_count": len(effects.get("served_demand_ids", [])),
            "served_demand_weight": effects.get("served_demand_weight"),
            "expected_supply": effects.get("expected_supply"),
            "is_feasible": candidate.is_feasible,
        }

    def _candidate_detail(self, candidate: Any) -> dict[str, Any]:
        detail = candidate_to_mapping(candidate)
        detail["in_plans"] = {
            plan_id: candidate.action_id in action_ids
            for plan_id, action_ids in self._plans.items()
        }
        return detail

    def _plan_summary(self, plan_id: str) -> dict[str, Any]:
        action_ids = list(self._plans.get(plan_id, []))
        return {
            "plan_id": plan_id,
            "items": [{"action_id": action_id} for action_id in action_ids],
            "action_ids": action_ids,
            "count": len(action_ids),
            "score": score_osm_v2_plan(self.export, action_ids),
        }

    def _budgets(self) -> dict[str, Any]:
        return {
            "tool_calls_remaining": self._max_tool_calls - self._tool_calls_used,
            "python_calls_remaining": self._max_python_calls - self._python_calls_used,
            "submit_ops_remaining": self._max_submit_ops - self._submit_ops_used,
        }

    def _workspace_files(self) -> dict[str, str]:
        return {
            "city_state": "city_state.json",
            "candidate_schema": "candidate_schema.json",
            "candidate_details": "candidates.json",
            "candidate_summary": "candidate_summary.csv",
            "candidate_table": "candidates.csv",
            "working_plans": "working_plans.json",
            "outputs_dir": "outputs/",
        }

    def _candidate_schema(self) -> dict[str, Any]:
        return {
            "candidate_details_file": "candidates.json",
            "candidate_details_shape": "list[CandidateAction]",
            "top_level_fields": [
                "action_id",
                "action_type",
                "label",
                "payload",
                "cost",
                "estimated_effects",
                "is_feasible",
                "metadata",
            ],
            "payload_note": {
                "road_planning": "payload usually contains entity road segment information.",
                "ev_charging": "payload contains site_id, entity, and assignment charger config.",
                "urban_planning": "payload contains block_id, entity, and assignment land_use.",
            },
            "estimated_effects_note": (
                "served_demand_ids, served_demand_weight, expected_supply, and other task metrics "
                "are under candidate['estimated_effects'], not top-level fields."
            ),
            "candidate_summary_fields": [
                "action_id",
                "action_type",
                "cost",
                "served_demand_count",
                "served_demand_weight",
                "efficiency_weight_per_cost",
                "expected_supply",
                "entity_id",
                "site_id",
                "block_id",
                "assignment_id",
                "charger_config_id",
                "land_use",
                "is_feasible",
            ],
            "submit_item_shape": {"action_id": "<candidate action_id>"},
        }

    def _next_step_hint(self, plan_id: str) -> str:
        plan = self._plans.get(plan_id, [])
        validation = self._validate_plan(plan) if plan else None
        if not plan:
            return "Read candidate_summary.csv or candidates.json, rank feasible actions, then submit a complete plan with op=set."
        if validation and validation["hard_errors"]:
            return "Repair hard_errors before finish. Use submit op=set to replace the plan with a valid action_id list."
        if validation and validation.get("remaining_steps") is not None and validation["remaining_steps"] <= 0:
            return "The plan is valid and max_steps is fully used. Call submit with op=finish now."
        if self._python_calls_used >= self._max_python_calls:
            return "The plan is valid and Python analysis budget is exhausted. Call submit with op=finish unless a simple submit repair is required."
        return "If the plan is valid and satisfactory, call submit with op=finish; otherwise revise with op=set/add/remove."

    def _write_sandbox_files(self) -> None:
        assert self._sandbox_dir is not None
        root = self._sandbox_dir.name
        state = {
            "task": self.export.task,
            "instance_id": self.export.instance_id,
            "metadata": self.export.metadata,
            "budget": self.export.budget,
            "max_steps": self.export.max_steps,
            "candidate_count": len(self.export.candidate_actions),
            "demand_count": len(self.export.demand),
        }
        with open(os.path.join(root, "city_state.json"), "w") as f:
            json.dump(state, f, indent=2, default=str)
        with open(os.path.join(root, "candidate_schema.json"), "w") as f:
            json.dump(self._candidate_schema(), f, indent=2, default=str)
        with open(os.path.join(root, "candidates.json"), "w") as f:
            json.dump([candidate_to_mapping(c) for c in self.export.candidate_actions], f, indent=2, default=str)
        with open(os.path.join(root, "candidate_summary.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "action_id",
                "action_type",
                "cost",
                "served_demand_count",
                "served_demand_weight",
                "efficiency_weight_per_cost",
                "expected_supply",
                "entity_id",
                "site_id",
                "block_id",
                "assignment_id",
                "charger_config_id",
                "land_use",
                "is_feasible",
            ])
            for candidate in self.export.candidate_actions:
                brief = self._candidate_brief(candidate)
                payload = candidate.payload
                entity = brief.get("entity") if isinstance(brief.get("entity"), Mapping) else {}
                assignment = brief.get("assignment") if isinstance(brief.get("assignment"), Mapping) else {}
                cost = float(brief["cost"] or 0.0)
                served_weight = float(brief.get("served_demand_weight") or 0.0)
                writer.writerow([
                    brief["id"],
                    brief["action_type"],
                    cost,
                    brief["served_demand_count"],
                    served_weight,
                    served_weight / cost if cost > 0 else "",
                    brief.get("expected_supply"),
                    entity.get("id", ""),
                    payload.get("site_id", ""),
                    payload.get("block_id", ""),
                    assignment.get("id", assignment.get("charger_config_id", "")),
                    assignment.get("charger_config_id", payload.get("charger_config_id", "")),
                    assignment.get("land_use", ""),
                    brief["is_feasible"],
                ])
        with open(os.path.join(root, "candidates.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["action_id", "action_type", "label", "cost", "entity", "assignment", "served_demand_count", "served_demand_weight"])
            for candidate in self.export.candidate_actions:
                brief = self._candidate_brief(candidate)
                writer.writerow([
                    brief["id"],
                    brief["action_type"],
                    brief["label"],
                    brief["cost"],
                    json.dumps(brief.get("entity"), default=str),
                    json.dumps(brief.get("assignment"), default=str),
                    brief["served_demand_count"],
                    brief.get("served_demand_weight"),
                ])
        self._write_working_plan_file()
        os.makedirs(os.path.join(root, "outputs"), exist_ok=True)

    def _write_working_plan_file(self) -> None:
        if self._sandbox_dir is None:
            return
        with open(os.path.join(self._sandbox_dir.name, "working_plans.json"), "w") as f:
            json.dump(self._plans, f, indent=2, default=str)

    def _ok_result(self, tool_result: Any) -> CityStepResult:
        return CityStepResult(
            state=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"valid": True, "tool_result": tool_result},
        )

    def _error_result(self, message: str) -> CityStepResult:
        return CityStepResult(
            state=None,
            reward=self._invalid_action_penalty,
            terminated=False,
            truncated=False,
            info={"valid": False, "message": message, "tool_result": {"error": message}},
        )


__all__ = ["OSMV2OnlineSandboxEnv"]
