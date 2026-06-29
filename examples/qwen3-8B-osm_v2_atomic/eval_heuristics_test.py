#!/usr/bin/env python3
"""Evaluate greedy, SA, and ALNS heuristics on OSM-v2 test instances.

The evaluator is intentionally offline and deterministic: all algorithms call
the same shared scoring logic as the environment, but do not involve LLM or
workspace execution. Results are written as per-instance details plus grouped
summary tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OSM_V2_DEPS = REPO_ROOT / "roll/pipeline/agentic/env/osm_v2/deps"
if str(OSM_V2_DEPS) not in sys.path:
    sys.path.insert(0, str(OSM_V2_DEPS))

from city_tasks.common.export_schema import (
    CityTaskExport,
    export_from_mapping,
)
from city_tasks.osm_v2.scoring import score_osm_v2_plan


TASK_CONFIGS = {
    "road_planning": "test_road_planning.jsonl",
    "ev_charging": "test_ev_charging.jsonl",
    "urban_planning": "test_urban_planning.jsonl",
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    return [value]


def _demand_weight_map(export: CityTaskExport) -> dict[str, float]:
    weights: dict[str, float] = {}
    for idx, item in enumerate(export.demand):
        did = item.get("id", item.get("parcel_id", item.get("demand_id", idx)))
        weight = item.get("demand_weight", item.get("weight", 1.0))
        try:
            weights[str(did)] = float(weight)
        except (TypeError, ValueError):
            weights[str(did)] = 1.0
    return weights


def _served_ids(candidate: Any) -> list[str]:
    for source in (candidate.estimated_effects, candidate.payload):
        for key in (
            "served_demand_ids",
            "served_demand_zones",
            "covered_zones",
            "served_zones",
            "served_parcels",
            "covered_parcels",
            "served_interior_parcels",
        ):
            values = _as_list(source.get(key))
            if values:
                return [str(value) for value in values]
    return []


def _site_id(candidate: Any) -> str:
    payload = candidate.payload
    for key in ("site_id", "charging_site_id", "parcel_id", "block_id"):
        value = payload.get(key)
        if value is not None:
            return str(value)
    entity = payload.get("entity")
    if isinstance(entity, dict) and entity.get("id") is not None:
        return str(entity["id"])
    return candidate.action_id


def _block_id(candidate: Any) -> str:
    payload = candidate.payload
    for key in ("block_id", "parcel_id"):
        value = payload.get(key)
        if value is not None:
            return str(value)
    return candidate.action_id


def _land_use(candidate: Any) -> str:
    payload = candidate.payload
    value = payload.get("land_use")
    if value is not None:
        return str(value)
    assignment = payload.get("assignment")
    if isinstance(assignment, dict) and assignment.get("land_use") is not None:
        return str(assignment["land_use"])
    return ""


@dataclass(frozen=True)
class CandidateInfo:
    idx: int
    action_id: str
    cost: float
    key: str
    land_use: str
    mask: int
    rank_value: float


@dataclass
class PlanState:
    selected: frozenset[int]
    score: float
    obj: float
    cost: float
    cost_ratio: float
    detail: dict[str, Any]


class FastProblem:
    def __init__(self, export: CityTaskExport, *, repair_candidate_limit: int = 300):
        self.export = export
        self.task = export.task
        self.instance_id = export.instance_id
        self.repair_candidate_limit = max(1, int(repair_candidate_limit))
        self.max_steps = int(export.max_steps) if export.max_steps is not None else 10**9
        self.budget = float(export.budget) if export.budget is not None and export.budget > 0 else None
        self.cost_weight = float(
            export.metadata.get("urban_cost_weight", 0.2)
            if export.task == "urban_planning"
            else export.metadata.get("cost_weight", 0.5)
        )

        demand_weights = _demand_weight_map(export)
        self.demand_ids = list(demand_weights)
        self.demand_index = {did: i for i, did in enumerate(self.demand_ids)}
        self.demand_weights = [float(demand_weights[did]) for did in self.demand_ids]
        self.total_demand_weight = sum(self.demand_weights) or float(len(self.demand_weights) or 1)

        self.requirements = export.metadata.get("need_config", {})
        if not isinstance(self.requirements, dict):
            self.requirements = {}
        self.required_total = sum(int(v) for v in self.requirements.values()) if self.requirements else 0

        self.candidates: list[CandidateInfo] = []
        for candidate in export.candidate_actions:
            if not candidate.is_feasible:
                continue
            mask = 0
            for did in _served_ids(candidate):
                idx = self.demand_index.get(str(did))
                if idx is not None:
                    mask |= 1 << idx
            cost = float(candidate.cost or 0.0)
            served_count = mask.bit_count()
            expected_supply = _safe_float(candidate.estimated_effects.get("expected_supply"), 0.0)
            rank_utility = self._bit_weight(mask) + 0.25 * served_count + 0.1 * expected_supply
            key = _block_id(candidate) if export.task == "urban_planning" else _site_id(candidate)
            self.candidates.append(
                CandidateInfo(
                    idx=len(self.candidates),
                    action_id=candidate.action_id,
                    cost=cost,
                    key=key,
                    land_use=_land_use(candidate),
                    mask=mask,
                    rank_value=rank_utility / max(cost, 1.0),
                )
            )
        self.id_to_idx = {candidate.action_id: candidate.idx for candidate in self.candidates}
        self.all_indices = tuple(candidate.idx for candidate in self.candidates)

    def _bit_weight(self, mask: int) -> float:
        total = 0.0
        while mask:
            lsb = mask & -mask
            total += self.demand_weights[lsb.bit_length() - 1]
            mask ^= lsb
        return total

    def is_feasible_set(self, selected: frozenset[int]) -> bool:
        if not selected or len(selected) > self.max_steps:
            return False
        cost = 0.0
        keys = set()
        enforce_unique_keys = self.task in {"ev_charging", "urban_planning"}
        for idx in selected:
            candidate = self.candidates[idx]
            cost += candidate.cost
            if enforce_unique_keys:
                if candidate.key in keys:
                    return False
                keys.add(candidate.key)
        return self.budget is None or cost <= self.budget + 1e-9

    def can_add(self, selected: frozenset[int], idx: int) -> bool:
        if idx in selected or len(selected) >= self.max_steps:
            return False
        candidate = self.candidates[idx]
        if self.budget is not None:
            cost = sum(self.candidates[i].cost for i in selected) + candidate.cost
            if cost > self.budget + 1e-9:
                return False
        if self.task in {"ev_charging", "urban_planning"}:
            key = candidate.key
            for i in selected:
                if self.candidates[i].key == key:
                    return False
        return True

    def evaluate(self, selected: frozenset[int]) -> PlanState:
        if not selected:
            return PlanState(frozenset(), 0.0, 0.0, 0.0, 0.0, {"valid": False, "reason": "empty"})
        action_ids = [self.candidates[i].action_id for i in sorted(selected)]
        detail = score_osm_v2_plan(self.export, action_ids)
        score = float(detail["score"])
        cost = float(detail.get("total_cost", 0.0))
        cost_ratio = float(detail.get("cost_ratio", 0.0))
        if self.task == "urban_planning":
            obj = 0.6 * float(detail.get("requirement_score", 0.0)) + 0.3 * float(detail.get("service_coverage", 0.0))
        else:
            obj = float(detail.get("coverage", 0.0))
        valid = self.is_feasible_set(selected) and not detail.get("invalid_selected")
        detail = dict(detail)
        detail["valid"] = bool(valid)
        return PlanState(selected, score, obj, cost, cost_ratio, detail)

    def sorted_candidates(self) -> list[int]:
        return [
            candidate.idx
            for candidate in sorted(
                self.candidates,
                key=lambda c: (c.rank_value, -c.cost),
                reverse=True,
            )
        ]

    def repair_candidates(self) -> list[int]:
        return self.sorted_candidates()[: min(len(self.candidates), self.repair_candidate_limit)]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def greedy_plan(problem: FastProblem, start: frozenset[int] | None = None, candidate_order: list[int] | None = None) -> PlanState:
    selected = frozenset(start or ())
    current = problem.evaluate(selected) if selected else PlanState(frozenset(), -1e18, 0.0, 0.0, 0.0, {})
    candidates = candidate_order or list(problem.all_indices)

    while True:
        best_state: PlanState | None = None
        for idx in candidates:
            if idx in selected:
                continue
            trial = frozenset((*selected, idx))
            if not problem.is_feasible_set(trial):
                continue
            state = problem.evaluate(trial)
            if best_state is None or state.score > best_state.score + 1e-12:
                best_state = state
        if best_state is None:
            break
        if selected and best_state.score <= current.score + 1e-12:
            break
        selected = best_state.selected
        current = best_state

    if selected:
        return current

    best_single: PlanState | None = None
    for idx in candidates:
        trial = frozenset((idx,))
        if not problem.is_feasible_set(trial):
            continue
        state = problem.evaluate(trial)
        if best_single is None or state.score > best_single.score:
            best_single = state
    return best_single or problem.evaluate(frozenset())


def random_initial_plan(problem: FastProblem, rng: random.Random, candidate_pool: list[int] | None = None) -> PlanState:
    candidates = list(candidate_pool or problem.all_indices)
    rng.shuffle(candidates)
    selected: frozenset[int] = frozenset()
    for idx in candidates:
        trial = frozenset((*selected, idx))
        if problem.is_feasible_set(trial):
            selected = trial
    if selected:
        return problem.evaluate(selected)
    return problem.evaluate(frozenset())


def _random_add(problem: FastProblem, selected: frozenset[int], rng: random.Random, pool: list[int] | None = None) -> frozenset[int] | None:
    candidates = list(pool or problem.all_indices)
    rng.shuffle(candidates)
    for idx in candidates:
        trial = frozenset((*selected, idx))
        if problem.is_feasible_set(trial):
            return trial
    return None


def _random_remove(selected: frozenset[int], rng: random.Random) -> frozenset[int] | None:
    if not selected:
        return None
    out = rng.choice(tuple(selected))
    trial = set(selected)
    trial.remove(out)
    return frozenset(trial) if trial else None


def _random_swap(problem: FastProblem, selected: frozenset[int], rng: random.Random, pool: list[int] | None = None) -> frozenset[int] | None:
    if not selected:
        return _random_add(problem, selected, rng, pool)
    remove_idx = rng.choice(tuple(selected))
    base = set(selected)
    base.remove(remove_idx)
    candidates = list(pool or problem.all_indices)
    rng.shuffle(candidates)
    for in_idx in candidates:
        if in_idx in base:
            continue
        trial = frozenset((*base, in_idx))
        if problem.is_feasible_set(trial):
            return trial
    return frozenset(base) if base else None


def sa_plan(problem: FastProblem, *, iterations: int, seed: int, init: str = "greedy") -> PlanState:
    rng = random.Random(seed)
    if init == "random":
        current = random_initial_plan(problem, rng)
    else:
        current = greedy_plan(problem, candidate_order=problem.sorted_candidates())
    if not current.selected:
        return current
    best = current
    ranked_pool = problem.sorted_candidates()[: max(50, min(len(problem.candidates), 250))]
    t0 = max(0.02, min(0.25, abs(current.score) * 0.2 + 0.05))
    t_end = 0.001

    for step in range(max(1, iterations)):
        progress = step / max(1, iterations - 1)
        temp = t0 * ((t_end / t0) ** progress)
        op = rng.random()
        if op < 0.35:
            trial_set = _random_add(problem, current.selected, rng, ranked_pool)
        elif op < 0.6:
            trial_set = _random_remove(current.selected, rng)
        else:
            trial_set = _random_swap(problem, current.selected, rng, ranked_pool)
        if not trial_set or not problem.is_feasible_set(trial_set):
            continue
        trial = problem.evaluate(trial_set)
        delta = trial.score - current.score
        if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-9)):
            current = trial
            if trial.score > best.score + 1e-12:
                best = trial
    return best


def _destroy_random(problem: FastProblem, selected: frozenset[int], rng: random.Random) -> frozenset[int]:
    if len(selected) <= 1:
        return selected
    remove_n = rng.randint(1, max(1, min(len(selected) - 1, math.ceil(0.3 * len(selected)))))
    remaining = set(selected)
    for idx in rng.sample(tuple(selected), remove_n):
        remaining.remove(idx)
    return frozenset(remaining)


def _destroy_expensive(problem: FastProblem, selected: frozenset[int], rng: random.Random) -> frozenset[int]:
    if len(selected) <= 1:
        return selected
    remove_n = rng.randint(1, max(1, min(len(selected) - 1, math.ceil(0.3 * len(selected)))))
    ranked = sorted(selected, key=lambda i: problem.candidates[i].cost, reverse=True)
    return frozenset(ranked[remove_n:])


def _destroy_worst(problem: FastProblem, selected: frozenset[int], rng: random.Random) -> frozenset[int]:
    if len(selected) <= 1:
        return selected
    remove_n = rng.randint(1, max(1, min(len(selected) - 1, math.ceil(0.3 * len(selected)))))
    current = problem.evaluate(selected)
    impacts = []
    for idx in selected:
        trial_set = frozenset(i for i in selected if i != idx)
        if not trial_set:
            impact = current.score
        else:
            impact = current.score - problem.evaluate(trial_set).score
        impacts.append((impact, idx))
    impacts.sort()
    remove = {idx for _, idx in impacts[:remove_n]}
    return frozenset(i for i in selected if i not in remove)


def _repair_greedy(problem: FastProblem, selected: frozenset[int], rng: random.Random) -> PlanState:
    return greedy_plan(problem, start=selected, candidate_order=problem.repair_candidates())


def _repair_noisy_greedy(problem: FastProblem, selected: frozenset[int], rng: random.Random) -> PlanState:
    ranked = problem.repair_candidates()
    top = ranked[: max(20, min(len(ranked), 120))]
    rng.shuffle(top)
    order = top + [idx for idx in ranked if idx not in set(top)]
    return greedy_plan(problem, start=selected, candidate_order=order)


def alns_plan(problem: FastProblem, *, iterations: int, seed: int, init: str = "greedy") -> PlanState:
    rng = random.Random(seed)
    if init == "random":
        current = random_initial_plan(problem, rng)
    else:
        current = greedy_plan(problem, candidate_order=problem.sorted_candidates())
    if not current.selected:
        return current
    best = current
    destroy_ops = [_destroy_random, _destroy_expensive, _destroy_worst]
    repair_ops = [_repair_greedy, _repair_noisy_greedy]
    destroy_weights = [1.0 for _ in destroy_ops]
    repair_weights = [1.0 for _ in repair_ops]
    t0 = max(0.01, min(0.2, abs(current.score) * 0.15 + 0.03))
    t_end = 0.0005

    for step in range(max(1, iterations)):
        d_idx = _weighted_choice(destroy_weights, rng)
        r_idx = _weighted_choice(repair_weights, rng)
        partial = destroy_ops[d_idx](problem, current.selected, rng)
        if not partial:
            partial = current.selected
        candidate = repair_ops[r_idx](problem, partial, rng)
        progress = step / max(1, iterations - 1)
        temp = t0 * ((t_end / t0) ** progress)
        delta = candidate.score - current.score
        accepted = delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-9))

        reward = 0.0
        if candidate.score > best.score + 1e-12:
            best = candidate
            reward = 5.0
        elif accepted and delta > 0:
            reward = 2.0
        elif accepted:
            reward = 0.5

        if accepted:
            current = candidate
        destroy_weights[d_idx] = 0.85 * destroy_weights[d_idx] + 0.15 * max(reward, 0.1)
        repair_weights[r_idx] = 0.85 * repair_weights[r_idx] + 0.15 * max(reward, 0.1)
    return best


def _weighted_choice(weights: list[float], rng: random.Random) -> int:
    total = sum(max(w, 0.0) for w in weights)
    if total <= 0:
        return rng.randrange(len(weights))
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += max(w, 0.0)
        if acc >= r:
            return i
    return len(weights) - 1


def _resolve_instance_path(data_root: Path, config_row: dict[str, Any]) -> Path:
    instance_path = config_row.get("instance_path")
    if instance_path:
        candidate = data_root / str(instance_path)
        if candidate.exists():
            return candidate
    for key in ("data_source_path", "absolute_instance_path"):
        raw = config_row.get("env_config", {}).get(key) if key == "data_source_path" else config_row.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if path.exists():
            return path
        marker = "/city_tasks_osm_v2/"
        text = str(raw)
        if marker in text:
            candidate = data_root / text.split(marker, 1)[1]
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Cannot resolve instance path for {config_row.get('instance_id')}")


def _load_export(path: Path) -> CityTaskExport:
    with path.open() as f:
        return export_from_mapping(json.load(f))


def _result_row(
    *,
    algorithm: str,
    problem: FastProblem,
    state: PlanState,
    runtime_s: float,
    instance_path: Path,
) -> dict[str, Any]:
    detail = state.detail
    if problem.task == "urban_planning":
        obj_components = {
            "requirement_score": detail.get("requirement_score"),
            "service_coverage": detail.get("service_coverage"),
        }
    else:
        obj_components = {"coverage": detail.get("coverage")}
    return {
        "algorithm": algorithm,
        "task": problem.task,
        "instance_id": problem.instance_id,
        "instance_path": str(instance_path),
        "valid": bool(detail.get("valid", False)),
        "obj": round(state.obj, 6),
        "cost": round(state.cost, 6),
        "cost_ratio": round(state.cost_ratio, 6),
        "score": round(state.score, 6),
        "selected_count": int(detail.get("selected_count", 0) or 0),
        "runtime_s": round(runtime_s, 6),
        "candidate_ids": [problem.candidates[i].action_id for i in sorted(state.selected)],
        "obj_components": obj_components,
        "score_detail": detail,
    }


def _evaluate_one(payload: tuple[str, str, str, int, int, int, int, str]) -> list[dict[str, Any]]:
    task, config_json, data_root_str, seed, sa_iters, alns_iters, repair_candidate_limit, metaheuristic_init = payload
    config_row = json.loads(config_json)
    data_root = Path(data_root_str)
    instance_path = _resolve_instance_path(data_root, config_row)
    problem = FastProblem(_load_export(instance_path), repair_candidate_limit=repair_candidate_limit)
    rows = []
    algorithms = [
        ("greedy", lambda: greedy_plan(problem, candidate_order=problem.sorted_candidates())),
        ("sa", lambda: sa_plan(problem, iterations=sa_iters, seed=seed, init=metaheuristic_init)),
        ("alns", lambda: alns_plan(problem, iterations=alns_iters, seed=seed + 1000003, init=metaheuristic_init)),
    ]
    for algorithm, fn in algorithms:
        start = time.time()
        state = fn()
        rows.append(
            _result_row(
                algorithm=algorithm,
                problem=problem,
                state=state,
                runtime_s=time.time() - start,
                instance_path=instance_path,
            )
        )
    return rows


def _read_configs(config_dir: Path, tasks: list[str], limit_per_task: int = 0) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for task in tasks:
        cfg_path = config_dir / TASK_CONFIGS[task]
        task_rows = 0
        with cfg_path.open() as f:
            for line in f:
                if line.strip():
                    rows.append((task, json.loads(line)))
                    task_rows += 1
                    if limit_per_task > 0 and task_rows >= limit_per_task:
                        break
    return rows


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["task"], row["algorithm"]), []).append(row)
    out = []
    for (task, algorithm), items in sorted(groups.items()):
        scores = [float(r["score"]) for r in items]
        objs = [float(r["obj"]) for r in items]
        costs = [float(r["cost"]) for r in items]
        cost_ratios = [float(r["cost_ratio"]) for r in items]
        runtimes = [float(r["runtime_s"]) for r in items]
        out.append(
            {
                "task": task,
                "algorithm": algorithm,
                "n": len(items),
                "valid_rate": sum(1 for r in items if r["valid"]) / len(items),
                "obj_mean": statistics.mean(objs),
                "cost_mean": statistics.mean(costs),
                "cost_ratio_mean": statistics.mean(cost_ratios),
                "score_mean": statistics.mean(scores),
                "score_p50": _quantile(scores, 0.50),
                "score_p95": _quantile(scores, 0.95),
                "score_max": max(scores),
                "runtime_mean_s": statistics.mean(runtimes),
            }
        )
    return out


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_existing_details(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if not path.exists():
        return [], set()
    rows = []
    algs_by_instance: dict[tuple[str, str], set[str]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            key = (str(row["task"]), str(row["instance_id"]))
            algs_by_instance.setdefault(key, set()).add(str(row["algorithm"]))
    complete = {key for key, algs in algs_by_instance.items() if {"greedy", "sa", "alns"}.issubset(algs)}
    return rows, complete


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OSM-v2 heuristics on the test split.")
    parser.add_argument("--data-root", type=Path, default=Path("examples/qwen3-8B-osm_v2_atomic/data/city_tasks_osm_v2"))
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--tasks", default="road_planning,ev_charging,urban_planning")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sa-iters", type=int, default=1200)
    parser.add_argument("--alns-iters", type=int, default=350)
    parser.add_argument("--metaheuristic-init", choices=("greedy", "random"), default="greedy")
    parser.add_argument("--repair-candidate-limit", type=int, default=300)
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true", help="Do not skip completed instances in an existing output dir.")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    config_dir = (args.config_dir or (data_root / "configs")).resolve()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = sorted(set(tasks) - set(TASK_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("output/osm_v2_heuristic_test") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = _read_configs(config_dir, tasks, args.limit_per_task)
    details_jsonl = output_dir / "details.jsonl"
    existing_rows, complete_instances = ([], set()) if args.no_resume else _read_existing_details(details_jsonl)

    payloads = []
    for i, (task, row) in enumerate(configs):
        key = (task, str(row.get("instance_id")))
        if key in complete_instances:
            continue
        payloads.append(
            (
                task,
                json.dumps(row, ensure_ascii=False),
                str(data_root),
                args.seed + i * 9973,
                args.sa_iters,
                args.alns_iters,
                args.repair_candidate_limit,
                args.metaheuristic_init,
            )
        )

    print(
        f"Evaluating {len(payloads)}/{len(configs)} remaining instances x 3 algorithms "
        f"with {args.workers} workers; output_dir={output_dir}",
        flush=True,
    )
    all_rows: list[dict[str, Any]] = list(existing_rows)
    started = time.time()
    completed = 0
    with details_jsonl.open("a") as details_file:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_evaluate_one, payload) for payload in payloads]
            for future in as_completed(futures):
                rows = future.result()
                all_rows.extend(rows)
                for row in rows:
                    details_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                details_file.flush()
                completed += 1
                if completed % 25 == 0 or completed == len(futures):
                    elapsed = time.time() - started
                    progress = {
                        "completed_this_run": completed,
                        "remaining_this_run": max(0, len(futures) - completed),
                        "total_instances": len(configs),
                        "elapsed_s": round(elapsed, 3),
                        "output_dir": str(output_dir),
                    }
                    (output_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                    print(f"completed {completed}/{len(futures)} remaining instances in {elapsed:.1f}s", flush=True)

    all_rows.sort(key=lambda r: (r["task"], r["instance_id"], r["algorithm"]))
    summary = _summarize(all_rows)

    detail_fields = [
        "task",
        "instance_id",
        "algorithm",
        "valid",
        "obj",
        "cost",
        "cost_ratio",
        "score",
        "selected_count",
        "runtime_s",
        "instance_path",
    ]
    summary_fields = [
        "task",
        "algorithm",
        "n",
        "valid_rate",
        "obj_mean",
        "cost_mean",
        "cost_ratio_mean",
        "score_mean",
        "score_p50",
        "score_p95",
        "score_max",
        "runtime_mean_s",
    ]
    _write_csv(output_dir / "details.csv", all_rows, detail_fields)
    _write_csv(output_dir / "summary.csv", summary, summary_fields)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "data_root": str(data_root),
                "config_dir": str(config_dir),
                "tasks": tasks,
                "workers": args.workers,
                "seed": args.seed,
                "sa_iters": args.sa_iters,
                "alns_iters": args.alns_iters,
                "metaheuristic_init": args.metaheuristic_init,
                "repair_candidate_limit": args.repair_candidate_limit,
                "limit_per_task": args.limit_per_task,
                "instances": len(configs),
                "rows": len(all_rows),
                "elapsed_s": round(time.time() - started, 3),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print(f"Wrote {details_jsonl}", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
