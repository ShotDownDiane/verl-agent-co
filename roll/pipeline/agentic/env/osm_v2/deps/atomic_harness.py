"""Atomic OSM-v2 planning harness.

This module is a small prototype for the "train finite, use infinite" idea:

- BuildPlan creates one initial valid plan in a clean local context.
- ImprovePlan proposes one local patch from the current plan and metrics.
- The harness evaluates each proposal and accepts only score-improving plans.

The default policies are deterministic heuristic policies. They are intended
as a CPU-only smoke implementation and as an interface target for later
SFT/RL-trained atomic policies.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from .city_tasks.common.export_schema import CityTaskExport, export_from_mapping
from .city_tasks.osm_v2.scoring import score_osm_v2_plan
from .workspace_env import OSMV2WorkspaceSandboxEnv

ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    """Resolve paths copied from another STS checkout."""

    path = Path(path)
    if path.exists():
        return path
    text = str(path)
    marker = "/STS/"
    if marker in text:
        candidate = ROOT / text.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return path


def load_export_from_config(config: Mapping[str, Any]) -> CityTaskExport:
    source = repo_path(config["env_config"]["data_source_path"])
    with source.open() as f:
        return export_from_mapping(json.load(f))


def load_export(source: str | Path | Mapping[str, Any] | CityTaskExport) -> CityTaskExport:
    if isinstance(source, CityTaskExport):
        return source
    if isinstance(source, Mapping):
        if "env_config" in source:
            return load_export_from_config(source)
        return export_from_mapping(source)
    with repo_path(source).open() as f:
        return export_from_mapping(json.load(f))


def candidate_ids(export: CityTaskExport) -> list[str]:
    return [candidate.action_id for candidate in export.candidate_actions if candidate.is_feasible]


def candidate_map(export: CityTaskExport) -> dict[str, Any]:
    return {candidate.action_id: candidate for candidate in export.candidate_actions}


def candidate_costs(export: CityTaskExport) -> dict[str, float]:
    return {candidate.action_id: float(candidate.cost) for candidate in export.candidate_actions}


def entity_key(export: CityTaskExport, action_id: str) -> str:
    candidate = candidate_map(export)[action_id]
    payload = candidate.payload
    if export.task == "ev_charging":
        return str(payload.get("site_id", action_id))
    if export.task == "urban_planning":
        return str(payload.get("block_id", action_id))
    return action_id


def entity_keys(export: CityTaskExport) -> dict[str, str]:
    cmap = candidate_map(export)
    out = {}
    for action_id in cmap:
        out[action_id] = entity_key(export, action_id)
    return out


def is_feasible_plan(export: CityTaskExport, plan: list[str]) -> bool:
    cmap = candidate_map(export)
    feasible_ids = {cid for cid, candidate in cmap.items() if candidate.is_feasible}
    if not plan:
        return False
    if any(action_id not in feasible_ids for action_id in plan):
        return False
    if len(plan) != len(set(plan)):
        return False
    if export.max_steps is not None and len(plan) > int(export.max_steps):
        return False
    costs = candidate_costs(export)
    if export.budget is not None and sum(costs.get(action_id, 0.0) for action_id in plan) > float(export.budget):
        return False
    if export.task in {"ev_charging", "urban_planning"}:
        keys = [entity_key(export, action_id) for action_id in plan]
        if len(keys) != len(set(keys)):
            return False
    return True


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def land_use_for_candidate(candidate: Any) -> str:
    payload = candidate.payload
    assignment = payload.get("assignment") if isinstance(payload.get("assignment"), Mapping) else {}
    return str(payload.get("land_use") or assignment.get("land_use") or "")


def site_id_for_candidate(candidate: Any) -> str:
    payload = candidate.payload
    return str(payload.get("site_id") or payload.get("parcel_id") or candidate.action_id)


def block_id_for_candidate(candidate: Any) -> str:
    payload = candidate.payload
    return str(payload.get("block_id") or payload.get("parcel_id") or candidate.action_id)


def candidate_rank_value(candidate: Any) -> tuple[float, float, float]:
    effects = candidate.estimated_effects
    served_weight = _as_float(effects.get("served_demand_weight"))
    served_count = len(effects.get("served_demand_ids", []) or [])
    expected_supply = _as_float(effects.get("expected_supply"))
    cost = max(float(candidate.cost or 0.0), 1.0)
    utility = served_weight + 0.25 * served_count + 0.1 * expected_supply
    return (utility / cost, utility, -cost)


def top_editable_candidates(
    export: CityTaskExport,
    *,
    current_plan: list[str],
    top_k: int = 30,
) -> list[dict[str, Any]]:
    selected = set(current_plan)
    ranked = sorted(
        [candidate for candidate in export.candidate_actions if candidate.is_feasible and candidate.action_id not in selected],
        key=candidate_rank_value,
        reverse=True,
    )
    rows = []
    for candidate in ranked[:top_k]:
        effects = candidate.estimated_effects
        rows.append(
            {
                "action_id": candidate.action_id,
                "cost": float(candidate.cost or 0.0),
                "rank_value": round(candidate_rank_value(candidate)[0], 6),
                "served_demand_weight": effects.get("served_demand_weight"),
                "served_demand_count": len(effects.get("served_demand_ids", []) or []),
                "expected_supply": effects.get("expected_supply"),
                "land_use": land_use_for_candidate(candidate),
                "site_id": site_id_for_candidate(candidate),
                "block_id": block_id_for_candidate(candidate),
            }
        )
    return rows


def write_plan(path: Path, candidate_id_list: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"candidate_ids": candidate_id_list}, indent=2, ensure_ascii=False) + "\n")


def score_value(evaluation: Mapping[str, Any]) -> float:
    score = evaluation.get("score", {})
    if not isinstance(score, Mapping):
        return 0.0
    return _as_float(score.get("score"), 0.0)


def selected_count(evaluation: Mapping[str, Any]) -> int:
    score = evaluation.get("score", {})
    if not isinstance(score, Mapping):
        return 0
    return int(_as_float(score.get("selected_count"), 0.0))


@dataclass
class AtomicContext:
    """Finite context passed to one atomic policy call."""

    atom: str
    task: str
    instance_id: str
    current_plan: list[str] = field(default_factory=list)
    current_score: float | None = None
    metric_breakdown: dict[str, Any] = field(default_factory=dict)
    editable_candidates: list[dict[str, Any]] = field(default_factory=list)
    budget: float | None = None
    max_steps: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "atom": self.atom,
            "task": self.task,
            "instance_id": self.instance_id,
            "current_plan": self.current_plan,
            "current_score": self.current_score,
            "metric_breakdown": self.metric_breakdown,
            "editable_candidates": self.editable_candidates,
            "budget": self.budget,
            "max_steps": self.max_steps,
            "notes": self.notes,
        }


@dataclass
class PlanProposal:
    candidate_ids: list[str]
    info: dict[str, Any] = field(default_factory=dict)


class BuildPlanPolicy(Protocol):
    name: str

    def build(self, export: CityTaskExport, context: AtomicContext) -> PlanProposal:
        ...


class ImprovePlanPolicy(Protocol):
    name: str

    def improve(self, export: CityTaskExport, context: AtomicContext) -> PlanProposal:
        ...


class GreedyBuildPlanPolicy:
    """Build a compact initial plan by greedy marginal score."""

    name = "greedy_build"

    def __init__(self, *, initial_step_ratio: float = 0.5, min_steps: int = 1) -> None:
        self.initial_step_ratio = max(0.05, min(float(initial_step_ratio), 1.0))
        self.min_steps = max(1, int(min_steps))

    def build(self, export: CityTaskExport, context: AtomicContext) -> PlanProposal:
        remaining = set(candidate_ids(export))
        selected: list[str] = []
        costs = candidate_costs(export)
        keys = entity_keys(export)
        used_keys: set[str] = set()
        total_cost = 0.0
        max_steps = int(export.max_steps or len(remaining) or 1)
        target_steps = max(self.min_steps, int(math.ceil(max_steps * self.initial_step_ratio)))
        target_steps = min(target_steps, max_steps)
        current_score = float(score_osm_v2_plan(export, selected)["score"])
        history = []

        while remaining and len(selected) < target_steps:
            best_id = None
            best_value = None
            for action_id in list(remaining):
                if export.task in {"ev_charging", "urban_planning"} and keys[action_id] in used_keys:
                    continue
                next_cost = total_cost + costs.get(action_id, 0.0)
                if export.budget is not None and next_cost > float(export.budget):
                    continue
                trial = selected + [action_id]
                trial_score = float(score_osm_v2_plan(export, trial)["score"])
                marginal = trial_score - current_score
                rank = candidate_rank_value(candidate_map(export)[action_id])
                value = (marginal / max(costs.get(action_id, 0.0), 1.0), marginal, trial_score, *rank)
                if best_value is None or value > best_value:
                    best_value = value
                    best_id = action_id
            if best_id is None:
                break
            selected.append(best_id)
            used_keys.add(keys[best_id])
            total_cost += costs.get(best_id, 0.0)
            remaining.remove(best_id)
            next_score = float(score_osm_v2_plan(export, selected)["score"])
            history.append({"action_id": best_id, "score": round(next_score, 6), "delta": round(next_score - current_score, 6)})
            current_score = next_score
            if best_value is not None and best_value[1] <= 0 and len(selected) >= self.min_steps and export.task != "urban_planning":
                break

        if not selected:
            selected = self._best_singleton(export)
        return PlanProposal(
            selected,
            {
                "policy": self.name,
                "initial_step_ratio": self.initial_step_ratio,
                "target_steps": target_steps,
                "greedy_history": history,
            },
        )

    def _best_singleton(self, export: CityTaskExport) -> list[str]:
        best_id = None
        best_score = -1e18
        for action_id in candidate_ids(export):
            if not is_feasible_plan(export, [action_id]):
                continue
            score = float(score_osm_v2_plan(export, [action_id])["score"])
            if score > best_score:
                best_score = score
                best_id = action_id
        return [best_id] if best_id is not None else []


class LocalSearchImprovePlanPolicy:
    """Propose the best single add/remove/swap patch in a bounded pool."""

    name = "local_search_improve"

    def __init__(self, *, candidate_limit: int = 120, max_trials: int = 4000) -> None:
        self.candidate_limit = max(1, int(candidate_limit))
        self.max_trials = max(1, int(max_trials))

    def improve(self, export: CityTaskExport, context: AtomicContext) -> PlanProposal:
        current = list(dict.fromkeys(context.current_plan))
        if not current:
            return PlanProposal([], {"policy": self.name, "reason": "empty_current_plan"})

        cmap = candidate_map(export)
        selected_set = set(current)
        ranked_candidates = sorted(
            [aid for aid in candidate_ids(export) if aid not in selected_set],
            key=lambda aid: candidate_rank_value(cmap[aid]),
            reverse=True,
        )
        pool = list(dict.fromkeys(current + ranked_candidates[: self.candidate_limit]))
        current_score = float(score_osm_v2_plan(export, current)["score"])

        operations: list[tuple[str, str | None, str | None]] = []
        operations.extend(("add", None, action_id) for action_id in pool if action_id not in selected_set)
        for out_id in current:
            operations.append(("remove", out_id, None))
        for out_id in current:
            operations.extend(("swap", out_id, in_id) for in_id in pool if in_id not in selected_set)

        best_plan: list[str] | None = None
        best_score = current_score
        best_edit: dict[str, Any] | None = None
        trials = 0
        for op, out_id, in_id in operations:
            if trials >= self.max_trials:
                break
            base = [aid for aid in current if aid != out_id]
            if in_id is not None and in_id not in base:
                trial = base + [in_id]
            else:
                trial = base
            if not trial:
                continue
            trials += 1
            if not is_feasible_plan(export, trial):
                continue
            trial_scores = score_osm_v2_plan(export, trial)
            if trial_scores.get("invalid_selected"):
                continue
            trial_score = float(trial_scores["score"])
            if trial_score > best_score + 1e-12:
                best_score = trial_score
                best_plan = trial
                best_edit = self._edit_info(export, current, trial, op=op, out_id=out_id, in_id=in_id)
                best_edit.update(
                    {
                        "before_score": round(current_score, 6),
                        "after_score": round(trial_score, 6),
                        "delta": round(trial_score - current_score, 6),
                    }
                )

        if best_plan is None:
            return PlanProposal(
                current,
                {
                    "policy": self.name,
                    "reason": "no_improving_single_edit",
                    "trials": trials,
                    "candidate_pool_size": len(pool),
                },
            )
        return PlanProposal(
            best_plan,
            {
                "policy": self.name,
                "edit": best_edit,
                "trials": trials,
                "candidate_pool_size": len(pool),
            },
        )

    @staticmethod
    def _edit_info(
        export: CityTaskExport,
        before: list[str],
        after: list[str],
        *,
        op: str,
        out_id: str | None,
        in_id: str | None,
    ) -> dict[str, Any]:
        before_set = set(before)
        after_set = set(after)
        added = [aid for aid in after if aid not in before_set]
        removed = [aid for aid in before if aid not in after_set]
        cmap = candidate_map(export)
        info: dict[str, Any] = {"op": op, "added": added, "removed": removed}
        if in_id and in_id in cmap:
            candidate = cmap[in_id]
            info["added_candidate"] = {
                "action_id": in_id,
                "cost": float(candidate.cost or 0.0),
                "land_use": land_use_for_candidate(candidate),
                "site_id": site_id_for_candidate(candidate),
                "block_id": block_id_for_candidate(candidate),
                "rank_value": round(candidate_rank_value(candidate)[0], 6),
            }
        if out_id and out_id in cmap:
            candidate = cmap[out_id]
            info["removed_candidate"] = {
                "action_id": out_id,
                "cost": float(candidate.cost or 0.0),
                "land_use": land_use_for_candidate(candidate),
                "site_id": site_id_for_candidate(candidate),
                "block_id": block_id_for_candidate(candidate),
                "rank_value": round(candidate_rank_value(candidate)[0], 6),
            }
        return info


class OSMV2AtomicHarness:
    """Open-ended composition loop around finite BuildPlan/ImprovePlan calls."""

    def __init__(
        self,
        data_source: str | Path | Mapping[str, Any] | CityTaskExport,
        *,
        build_policy: BuildPlanPolicy | None = None,
        improve_policy: ImprovePlanPolicy | None = None,
        run_root: str | Path | None = None,
        run_id: str | None = None,
        isolation: str = "local",
        step_timeout_seconds: int = 10,
        editable_top_k: int = 30,
    ) -> None:
        self.data_source = data_source
        self.export = load_export(data_source)
        self.build_policy = build_policy or GreedyBuildPlanPolicy()
        self.improve_policy = improve_policy or LocalSearchImprovePlanPolicy()
        self.run_root = Path(run_root or ROOT / "tmp/osm_v2_atomic_harness").resolve()
        self.run_id = run_id or f"{self.export.task}_{self.export.instance_id}_{uuid.uuid4().hex[:8]}"
        self.isolation = isolation
        self.step_timeout_seconds = int(step_timeout_seconds)
        self.editable_top_k = int(editable_top_k)
        self.env = OSMV2WorkspaceSandboxEnv(
            data_source,
            run_root=self.run_root,
            run_id=self.run_id,
            isolation=self.isolation,
            step_timeout_seconds=self.step_timeout_seconds,
        )
        self.trace: list[dict[str, Any]] = []

    def run(self, *, max_iters: int = 8, patience: int = 3, min_delta: float = 1e-9) -> dict[str, Any]:
        self.env.reset()
        trace_path = self.env.outputs_dir / "atomic_trace.jsonl"
        final_plan_path = self.env.outputs_dir / "final_plan.json"
        best_plan_path = self.env.outputs_dir / "best_plan.json"

        build_context = self._context(atom="BuildPlan", plan=[], evaluation=None)
        build = self.build_policy.build(self.export, build_context)
        write_plan(final_plan_path, build.candidate_ids)
        evaluation = self.env.evaluate_plan(final_plan_path)
        current_plan = list(build.candidate_ids)
        current_score = score_value(evaluation)
        best_plan = list(current_plan)
        best_score = current_score
        no_improve = 0
        build_reward = self._build_reward(evaluation)
        self._record(
            trace_path,
            {
                "atom": "BuildPlan",
                "context": build_context.to_jsonable(),
                "proposal": build.candidate_ids,
                "proposal_info": build.info,
                "evaluation": evaluation,
                "reward": build_reward,
                "accepted": bool(evaluation.get("valid") and selected_count(evaluation) > 0),
                "context_chars": len(json.dumps(build_context.to_jsonable(), ensure_ascii=False, default=str)),
            },
        )

        if evaluation.get("valid") and selected_count(evaluation) > 0:
            shutil.copy2(final_plan_path, best_plan_path)

        for iteration in range(1, int(max_iters) + 1):
            if no_improve >= int(patience):
                break
            improve_context = self._context(atom="ImprovePlan", plan=current_plan, evaluation=evaluation)
            proposal = self.improve_policy.improve(self.export, improve_context)
            proposal_path = self.env.outputs_dir / f"proposal_iter_{iteration:03d}.json"
            write_plan(proposal_path, proposal.candidate_ids)
            proposal_eval = self.env.evaluate_plan(proposal_path)
            proposal_score = score_value(proposal_eval)
            valid_non_empty = bool(proposal_eval.get("valid") and selected_count(proposal_eval) > 0)
            improved = bool(valid_non_empty and proposal_score > current_score + float(min_delta))
            reward = proposal_score - current_score if valid_non_empty else -1.0

            if improved:
                current_plan = list(proposal.candidate_ids)
                current_score = proposal_score
                evaluation = proposal_eval
                no_improve = 0
                shutil.copy2(proposal_path, final_plan_path)
                if current_score > best_score + float(min_delta):
                    best_score = current_score
                    best_plan = list(current_plan)
                    shutil.copy2(proposal_path, best_plan_path)
            else:
                no_improve += 1

            self._record(
                trace_path,
                {
                    "atom": "ImprovePlan",
                    "iteration": iteration,
                    "context": improve_context.to_jsonable(),
                    "proposal": proposal.candidate_ids,
                    "proposal_info": proposal.info,
                    "evaluation": proposal_eval,
                    "reward": round(reward, 6),
                    "accepted": improved,
                    "patience": no_improve,
                    "context_chars": len(json.dumps(improve_context.to_jsonable(), ensure_ascii=False, default=str)),
                },
            )

        final_evaluation = self.env.evaluate_plan(final_plan_path) if final_plan_path.exists() else {}
        summary = {
            "task": self.export.task,
            "instance_id": self.export.instance_id,
            "run_dir": str(self.env.run_dir),
            "build_policy": getattr(self.build_policy, "name", type(self.build_policy).__name__),
            "improve_policy": getattr(self.improve_policy, "name", type(self.improve_policy).__name__),
            "max_iters": int(max_iters),
            "patience": int(patience),
            "iterations_run": sum(1 for row in self.trace if row.get("atom") == "ImprovePlan"),
            "accepted_improvements": sum(1 for row in self.trace if row.get("atom") == "ImprovePlan" and row.get("accepted")),
            "initial_score": score_value(self.trace[0]["evaluation"]) if self.trace else 0.0,
            "final_score": score_value(final_evaluation),
            "best_score": best_score,
            "score_gain": round(best_score - (score_value(self.trace[0]["evaluation"]) if self.trace else 0.0), 6),
            "valid": bool(final_evaluation.get("valid")),
            "selected_count": selected_count(final_evaluation),
            "best_plan": best_plan,
            "final_evaluation": final_evaluation,
            "trace_path": str(trace_path),
            "best_plan_path": str(best_plan_path),
        }
        (self.env.outputs_dir / "atomic_harness_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n"
        )
        return summary

    def _context(
        self,
        *,
        atom: str,
        plan: list[str],
        evaluation: Mapping[str, Any] | None,
    ) -> AtomicContext:
        score_payload = evaluation.get("score", {}) if isinstance(evaluation, Mapping) else {}
        if not isinstance(score_payload, dict):
            score_payload = {}
        notes = [
            "finite-context atomic call",
            "global state is owned by workspace+harness",
            "proposal is evaluated by input/evaluate_plan.py and accepted only if score improves",
        ]
        return AtomicContext(
            atom=atom,
            task=self.export.task,
            instance_id=self.export.instance_id,
            current_plan=list(plan),
            current_score=score_value(evaluation) if evaluation is not None else None,
            metric_breakdown=dict(score_payload),
            editable_candidates=top_editable_candidates(self.export, current_plan=plan, top_k=self.editable_top_k),
            budget=float(self.export.budget) if self.export.budget is not None else None,
            max_steps=int(self.export.max_steps) if self.export.max_steps is not None else None,
            notes=notes,
        )

    @staticmethod
    def _build_reward(evaluation: Mapping[str, Any]) -> float:
        if not evaluation.get("valid") or selected_count(evaluation) <= 0:
            return -1.0
        return round(1.0 + score_value(evaluation), 6)

    def _record(self, trace_path: Path, row: dict[str, Any]) -> None:
        self.trace.append(row)
        with trace_path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str, sort_keys=True) + "\n")


__all__ = [
    "AtomicContext",
    "BuildPlanPolicy",
    "GreedyBuildPlanPolicy",
    "ImprovePlanPolicy",
    "LocalSearchImprovePlanPolicy",
    "OSMV2AtomicHarness",
    "PlanProposal",
    "load_export",
    "load_export_from_config",
    "repo_path",
]
