"""Workspace-style OSM v2 sandbox for bash-native agents.

This module packages an OSM v2 instance as an isolated file workspace:

- ``input/`` is task data and helper scripts.
- ``work/`` is writable scratch space.
- ``outputs/`` is writable plan/evaluation output.

It is intended for mini-swe-agent style agents that operate through shell
commands instead of structured sandbox tool calls.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping

from .city_tasks.common.export_schema import CityTaskExport, export_to_mapping
from .city_tasks.osm_v2.online_sandbox_env import OSMV2OnlineSandboxEnv

MAX_OBSERVATION_CHARS = 2500
ROOT = Path(__file__).resolve().parents[1]


def _repo_path(path: str | Path) -> Path:
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


EVALUATE_PLAN_SOURCE = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def as_list(value: Any) -> list[Any]:
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


def load_plan(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [str(item.get("action_id", item)) if isinstance(item, dict) else str(item) for item in payload]
    if not isinstance(payload, dict):
        raise SystemExit("Plan must be a JSON object or list.")
    for key in ("candidate_ids", "action_ids", "selected_candidate_ids"):
        value = payload.get(key)
        if isinstance(value, list):
            return [str(item.get("action_id", item)) if isinstance(item, dict) else str(item) for item in value]
    items = payload.get("items")
    if isinstance(items, list):
        return [str(item.get("action_id", item)) if isinstance(item, dict) else str(item) for item in items]
    raise SystemExit("Plan JSON must contain candidate_ids, action_ids, selected_candidate_ids, or items.")


def demand_weight_map(export: Mapping[str, Any]) -> dict[str, float]:
    weights = {}
    for item in export.get("demand", []):
        did = item.get("id", item.get("parcel_id", item.get("demand_id")))
        if did is None:
            continue
        try:
            weights[str(did)] = float(item.get("demand_weight", item.get("weight", 1.0)))
        except (TypeError, ValueError):
            weights[str(did)] = 1.0
    return weights


def served_ids(candidate: Mapping[str, Any]) -> list[str]:
    for source in (candidate.get("estimated_effects", {}), candidate.get("payload", {})):
        if not isinstance(source, Mapping):
            continue
        for key in (
            "served_demand_ids",
            "served_demand_zones",
            "covered_zones",
            "served_zones",
            "served_parcels",
            "covered_parcels",
            "served_interior_parcels",
        ):
            values = as_list(source.get(key))
            if values:
                return [str(value) for value in values]
    return []


def site_id(candidate: Mapping[str, Any]) -> str:
    payload = candidate.get("payload", {})
    if not isinstance(payload, Mapping):
        payload = {}
    for key in ("site_id", "charging_site_id", "parcel_id", "block_id"):
        if payload.get(key) is not None:
            return str(payload[key])
    entity = payload.get("entity")
    if isinstance(entity, Mapping) and entity.get("id") is not None:
        return str(entity["id"])
    return str(candidate.get("action_id", ""))


def block_id(candidate: Mapping[str, Any]) -> str:
    payload = candidate.get("payload", {})
    if not isinstance(payload, Mapping):
        payload = {}
    for key in ("block_id", "parcel_id"):
        if payload.get(key) is not None:
            return str(payload[key])
    return str(candidate.get("action_id", ""))


def land_use(candidate: Mapping[str, Any]) -> str:
    payload = candidate.get("payload", {})
    if not isinstance(payload, Mapping):
        payload = {}
    if payload.get("land_use") is not None:
        return str(payload["land_use"])
    assignment = payload.get("assignment")
    if isinstance(assignment, Mapping) and assignment.get("land_use") is not None:
        return str(assignment["land_use"])
    return ""


def budget(export: Mapping[str, Any], total_cost: float) -> float:
    value = export.get("budget")
    if value is not None and float(value) > 0:
        return float(value)
    return max(total_cost, 1.0)


def valid_selected(export: Mapping[str, Any], action_ids: list[str]) -> tuple[list[Mapping[str, Any]], list[str]]:
    candidates = {
        str(candidate.get("action_id")): candidate
        for candidate in export.get("candidate_actions", [])
        if isinstance(candidate, Mapping)
    }
    selected = []
    invalid = []
    seen = set()
    for action_id in action_ids:
        aid = str(action_id)
        candidate = candidates.get(aid)
        if candidate is None or not candidate.get("is_feasible", False):
            invalid.append(aid)
            continue
        if aid in seen:
            continue
        seen.add(aid)
        selected.append(candidate)
    return selected, invalid


def score_subset(export: Mapping[str, Any], selected: list[Mapping[str, Any]], invalid: list[str]) -> dict[str, Any]:
    demand_weights = demand_weight_map(export)
    total_weight = sum(demand_weights.values())
    if total_weight <= 0:
        total_weight = float(len(export.get("demand", [])))
        demand_weights = {
            str(item.get("id", item.get("parcel_id", idx))): 1.0
            for idx, item in enumerate(export.get("demand", []))
        }
    served = set()
    total_cost = 0.0
    selected_sites = set()
    duplicate_sites = 0
    total_supply = 0.0
    for candidate in selected:
        total_cost += float(candidate.get("cost") or 0.0)
        sid = site_id(candidate)
        duplicate_sites += int(sid in selected_sites)
        selected_sites.add(sid)
        served.update(served_ids(candidate))
        effects = candidate.get("estimated_effects", {})
        payload = candidate.get("payload", {})
        supply = effects.get("expected_supply", payload.get("expected_supply", payload.get("charger_supply", 0.0)))
        try:
            total_supply += float(supply)
        except (TypeError, ValueError):
            pass
    served_weight = sum(demand_weights.get(did, 0.0) for did in served)
    coverage = served_weight / total_weight if total_weight > 0 else 0.0
    b = budget(export, total_cost)
    cost_ratio = total_cost / b if b > 0 else 0.0
    max_steps = export.get("max_steps")
    count_violation = max(0, len(selected) - int(max_steps)) if max_steps is not None else 0
    penalty = 0.05 * count_violation + 0.05 * duplicate_sites
    cost_weight = float(export.get("metadata", {}).get("cost_weight", 0.5))
    score = coverage - cost_weight * cost_ratio - penalty
    return {
        "score": round(score, 6),
        "coverage": round(coverage, 6),
        "weighted_coverage": round(coverage, 6),
        "cost_ratio": round(cost_ratio, 6),
        "served_demand_weight": round(served_weight, 6),
        "total_demand_weight": round(total_weight, 6),
        "unique_demand_served": len(served),
        "total_demand_zones": len(export.get("demand", [])),
        "selected_count": len(selected),
        "unique_site_count": len(selected_sites),
        "duplicate_site_violation": duplicate_sites,
        "count_violation": count_violation,
        "invalid_selected": invalid,
        "total_cost": round(total_cost, 6),
        "budget": round(b, 6),
        "expected_supply": round(total_supply, 6),
        "cost_weight": cost_weight,
    }


def score_ev_v2(export: Mapping[str, Any], selected: list[Mapping[str, Any]], invalid: list[str]) -> dict[str, Any]:
    import heapq
    cost_weight = float(export.get("metadata", {}).get("cost_weight", 0.01))
    demand_weights = demand_weight_map(export)
    total_weight = sum(demand_weights.values())

    total_cost = 0.0
    selected_sites = set()
    selected_node_ids = set()
    duplicate_sites = 0
    for candidate in selected:
        total_cost += float(candidate.get("cost") or 0.0)
        sid = site_id(candidate)
        duplicate_sites += int(sid in selected_sites)
        selected_sites.add(sid)
        nid = candidate.get("payload", {}).get("node_id")
        if nid is not None:
            selected_node_ids.add(int(nid))

    if not selected_node_ids or total_weight <= 0:
        b = budget(export, total_cost)
        return {
            "score": 0.0, "mean_travel_distance": 0.0, "total_cost": round(total_cost, 6),
            "budget": round(b, 6), "cost_ratio": round(total_cost / b, 6) if b > 0 else 0.0,
            "selected_count": len(selected), "unique_site_count": len(selected_sites),
            "duplicate_site_violation": duplicate_sites, "demand_served": 0,
            "demand_total": len(export.get("demand", [])), "invalid_selected": invalid,
            "cost_weight": cost_weight,
        }

    # Try precomputed distance matrix first
    dist_matrix = export.get("distance_matrix")
    if dist_matrix:
        # dist_matrix: {site_id: {demand_id: distance}}
        site_id_to_node = {}
        for c in export.get("candidate_actions", []):
            sid_c = c.get("payload", {}).get("site_id")
            nid_c = c.get("payload", {}).get("node_id")
            if sid_c and nid_c is not None:
                site_id_to_node[sid_c] = int(nid_c)
        active_sids = [sid for sid in selected_sites if site_id_to_node.get(sid) in selected_node_ids]
    else:
        active_sids = None

    demand_entries = {}
    for d in export.get("demand", []):
        entries = d.get("entry_node_ids", [])
        if entries:
            demand_entries[str(d.get("id", ""))] = entries

    if dist_matrix and active_sids:
        # Fast path: lookup from precomputed matrix
        total_weighted_dist = 0.0
        served = 0
        for did, w in demand_weights.items():
            min_d = float("inf")
            for sid in active_sids:
                d_val = dist_matrix.get(sid, {}).get(did)
                if d_val is not None and d_val < min_d:
                    min_d = d_val
            if min_d < float("inf"):
                total_weighted_dist += w * min_d
                served += 1
    else:
        # Slow path: Dijkstra
        graph = {}
        for e in export.get("edges", []):
            fn, tn = e.get("from_node"), e.get("to_node")
            if fn is None or tn is None:
                continue
            length = float(e.get("length_m", 1.0))
            graph.setdefault(fn, []).append((tn, length))
            graph.setdefault(tn, []).append((fn, length))

        def _dijkstra(source):
            dist = {source: 0.0}
            pq = [(0.0, source)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, float("inf")):
                    continue
                for v, w in graph.get(u, []):
                    nd = d + w
                    if nd < dist.get(v, float("inf")):
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
            return dist

        site_dists = [_dijkstra(nid) for nid in selected_node_ids]
        total_weighted_dist = 0.0
        served = 0
        for did, w in demand_weights.items():
            entries = demand_entries.get(did, [])
            if not entries:
                continue
            min_d = float("inf")
            for sd in site_dists:
                for e in entries:
                    if e in sd and sd[e] < min_d:
                        min_d = sd[e]
            if min_d < float("inf"):
                total_weighted_dist += w * min_d
                served += 1

    mean_d = total_weighted_dist / total_weight if total_weight > 0 else 0.0
    b = budget(export, total_cost)
    cost_ratio = total_cost / b if b > 0 else 0.0
    max_steps = export.get("max_steps")
    count_violation = max(0, len(selected) - int(max_steps)) if max_steps is not None else 0
    penalty = 0.05 * count_violation + 0.05 * duplicate_sites
    score = -mean_d - cost_weight * total_cost - penalty

    return {
        "score": round(score, 6),
        "mean_travel_distance": round(mean_d, 6),
        "total_cost": round(total_cost, 6),
        "budget": round(b, 6),
        "cost_ratio": round(cost_ratio, 6),
        "selected_count": len(selected),
        "unique_site_count": len(selected_sites),
        "duplicate_site_violation": duplicate_sites,
        "count_violation": count_violation,
        "demand_served": served,
        "demand_total": len(export.get("demand", [])),
        "invalid_selected": invalid,
        "cost_weight": cost_weight,
    }


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):
            return default
        return out
    except (TypeError, ValueError):
        return default


def ev_candidate_supply(candidate: Mapping[str, Any]) -> float:
    effects = candidate.get("estimated_effects", {})
    if not isinstance(effects, Mapping):
        effects = {}
    payload = candidate.get("payload", {})
    if not isinstance(payload, Mapping):
        payload = {}
    supply = effects.get("expected_supply", payload.get("expected_supply", payload.get("charger_supply")))
    value = safe_float(supply, -1.0)
    if value >= 0:
        return value
    chargers = payload.get("chargers", {})
    if isinstance(chargers, Mapping):
        return (
            safe_float(chargers.get("slow"), 0.0)
            + 3.0 * safe_float(chargers.get("medium"), 0.0)
            + 8.0 * safe_float(chargers.get("fast"), 0.0)
        )
    return 0.0


def ev_weighted_nearest_mean(
    distance_by_site: Mapping[str, Mapping[str, Any]],
    site_ids: set[str],
    demand_weights: Mapping[str, float],
) -> tuple[float, float]:
    total_dist = 0.0
    total_weight = 0.0
    for did, weight in demand_weights.items():
        best = min(
            (
                safe_float(distance_by_site.get(sid, {}).get(did), float("inf"))
                for sid in site_ids
            ),
            default=float("inf"),
        )
        if best < float("inf"):
            total_dist += weight * best
            total_weight += weight
    return (total_dist / total_weight if total_weight > 0 else 0.0, total_weight)


def score_ev_v3(export: Mapping[str, Any], selected: list[Mapping[str, Any]], invalid: list[str]) -> dict[str, Any]:
    metadata = export.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    demand_weights = demand_weight_map(export)
    total_demand_weight = sum(demand_weights.values())

    total_cost = 0.0
    duplicate_sites = 0
    selected_sites = set()
    site_infos = {}
    supply_unit = safe_float(metadata.get("ev_supply_unit_weight"), 50.0)

    for candidate in selected:
        total_cost += float(candidate.get("cost") or 0.0)
        sid = site_id(candidate)
        if sid in selected_sites:
            duplicate_sites += 1
            continue
        selected_sites.add(sid)
        supply = ev_candidate_supply(candidate)
        site_infos[sid] = {
            "supply": supply,
            "capacity_weight": max(0.0, supply * supply_unit),
        }

    b = budget(export, total_cost)
    cost_ratio = total_cost / b if b > 0 else 0.0
    max_steps = export.get("max_steps")
    count_violation = max(0, len(selected) - int(max_steps)) if max_steps is not None else 0

    cost_weight = safe_float(metadata.get("ev_cost_ratio_weight"), 0.2)
    queue_weight = safe_float(metadata.get("ev_queue_assignment_weight"), 0.15)
    distance_scale = max(1.0, safe_float(metadata.get("ev_distance_scale_m"), 2000.0))
    overload_penalty_weight = safe_float(metadata.get("ev_overload_penalty_weight"), 0.10)
    unserved_penalty_weight = safe_float(metadata.get("ev_unserved_penalty_weight"), 0.20)

    all_site_ids = set()
    for candidate in export.get("candidate_actions", []):
        if candidate.get("is_feasible", True):
            all_site_ids.add(site_id(candidate))
    distance_by_site = export.get("distance_matrix") or {}
    if not isinstance(distance_by_site, Mapping):
        distance_by_site = {}

    if not site_infos or total_demand_weight <= 0:
        penalty = 0.05 * duplicate_sites + 0.05 * count_violation
        return {
            "score": round(-cost_weight * cost_ratio - penalty, 6) if selected else 0.0,
            "score_version": "ev_v3",
            "coverage": 0.0,
            "weighted_coverage": 0.0,
            "service_score": 0.0,
            "distance_score": 0.0,
            "time_score": 0.0,
            "benefit_score": 0.0,
            "selected_mean_distance": 0.0,
            "ideal_mean_distance": 0.0,
            "mean_travel_distance": 0.0,
            "mean_time": 0.0,
            "served_demand_weight": 0.0,
            "assigned_demand_weight": 0.0,
            "total_demand_weight": round(total_demand_weight, 6),
            "total_capacity_weight": 0.0,
            "overload_ratio": 0.0,
            "unserved_ratio": 1.0 if total_demand_weight > 0 else 0.0,
            "total_cost": round(total_cost, 6),
            "budget": round(b, 6),
            "cost_ratio": round(cost_ratio, 6),
            "selected_count": len(selected),
            "unique_site_count": len(site_infos),
            "duplicate_site_violation": duplicate_sites,
            "count_violation": count_violation,
            "invalid_selected": invalid,
            "cost_weight": cost_weight,
        }

    assignments = {}
    loads = {sid: 0.0 for sid in site_infos}
    pressures = {sid: 0.0 for sid in site_infos}
    for _ in range(3):
        next_assignments = {}
        for did, weight in demand_weights.items():
            best_sid = None
            best_distance = float("inf")
            best_generalized = float("inf")
            for sid in site_infos:
                distance = safe_float(distance_by_site.get(sid, {}).get(did), float("inf"))
                if distance == float("inf"):
                    continue
                generalized = distance / distance_scale + queue_weight * pressures.get(sid, 0.0)
                if generalized < best_generalized:
                    best_generalized = generalized
                    best_sid = sid
                    best_distance = distance
            if best_sid is not None:
                next_assignments[did] = (best_sid, best_distance)
        loads = {sid: 0.0 for sid in site_infos}
        for did, (sid, _) in next_assignments.items():
            loads[sid] += demand_weights.get(did, 0.0)
        pressures = {}
        for sid, load in loads.items():
            capacity = max(site_infos[sid]["capacity_weight"], 1e-9)
            rho = load / capacity
            pressures[sid] = rho / max(1.0 - min(rho, 0.99), 1e-9)
        assignments = next_assignments

    assigned_weight = sum(demand_weights.get(did, 0.0) for did in assignments)
    selected_distance_total = sum(
        demand_weights.get(did, 0.0) * distance for did, (_, distance) in assignments.items()
    )
    selected_mean_distance = selected_distance_total / assigned_weight if assigned_weight > 0 else 0.0
    ideal_mean_distance, ideal_assigned_weight = ev_weighted_nearest_mean(distance_by_site, all_site_ids, demand_weights)
    if selected_mean_distance <= 0:
        distance_score = 1.0 if assigned_weight > 0 else 0.0
    elif ideal_assigned_weight > 0 and ideal_mean_distance > 0:
        distance_score = min(1.0, ideal_mean_distance / selected_mean_distance)
    else:
        distance_score = 0.0

    served_weight = 0.0
    overload_weight = 0.0
    total_capacity_weight = 0.0
    weighted_time = 0.0
    for sid, info in site_infos.items():
        load = loads.get(sid, 0.0)
        capacity = max(info["capacity_weight"], 1e-9)
        total_capacity_weight += info["capacity_weight"]
        served_weight += min(load, info["capacity_weight"])
        overload_weight += max(0.0, load - info["capacity_weight"])
        rho = load / capacity
        rho_capped = min(rho, 0.99)
        charge_time = load / capacity
        wait_time = rho_capped / max(2.0 * (1.0 - rho_capped), 1e-9)
        weighted_time += load * (charge_time + wait_time)

    service_score = min(1.0, served_weight / total_demand_weight) if total_demand_weight > 0 else 0.0
    mean_time = weighted_time / assigned_weight if assigned_weight > 0 else 0.0
    time_score = 1.0 / (1.0 + mean_time)
    overload_ratio = overload_weight / total_demand_weight if total_demand_weight > 0 else 0.0
    unserved_ratio = max(0.0, total_demand_weight - assigned_weight) / total_demand_weight if total_demand_weight > 0 else 0.0
    penalty = (
        0.05 * duplicate_sites
        + 0.05 * count_violation
        + overload_penalty_weight * overload_ratio
        + unserved_penalty_weight * unserved_ratio
    )
    benefit_score = 0.50 * service_score + 0.25 * distance_score + 0.15 * time_score
    score = benefit_score - cost_weight * cost_ratio - penalty

    return {
        "score": round(score, 6),
        "score_version": "ev_v3",
        "coverage": round(service_score, 6),
        "weighted_coverage": round(service_score, 6),
        "service_score": round(service_score, 6),
        "distance_score": round(distance_score, 6),
        "time_score": round(time_score, 6),
        "benefit_score": round(benefit_score, 6),
        "selected_mean_distance": round(selected_mean_distance, 6),
        "ideal_mean_distance": round(ideal_mean_distance, 6),
        "mean_travel_distance": round(selected_mean_distance, 6),
        "mean_time": round(mean_time, 6),
        "served_demand_weight": round(served_weight, 6),
        "assigned_demand_weight": round(assigned_weight, 6),
        "total_demand_weight": round(total_demand_weight, 6),
        "total_capacity_weight": round(total_capacity_weight, 6),
        "overload_ratio": round(overload_ratio, 6),
        "unserved_ratio": round(unserved_ratio, 6),
        "total_cost": round(total_cost, 6),
        "budget": round(b, 6),
        "cost_ratio": round(cost_ratio, 6),
        "selected_count": len(selected),
        "unique_site_count": len(site_infos),
        "duplicate_site_violation": duplicate_sites,
        "count_violation": count_violation,
        "demand_served": len(assignments),
        "demand_total": len(export.get("demand", [])),
        "invalid_selected": invalid,
        "cost_weight": cost_weight,
    }


def road_graph(export: Mapping[str, Any], selected_ids: set[str]) -> dict[int, list[tuple[int, float]]]:
    graph = {}
    for edge in export.get("edges", []):
        fn, tn = edge.get("from_node"), edge.get("to_node")
        if fn is None or tn is None:
            continue
        length = float(edge.get("length_m", 1.0))
        graph.setdefault(int(fn), []).append((int(tn), length))
        graph.setdefault(int(tn), []).append((int(fn), length))
    for candidate in export.get("candidate_actions", []):
        action_id = str(candidate.get("action_id"))
        if action_id not in selected_ids:
            continue
        payload = candidate.get("payload", {})
        if not isinstance(payload, Mapping):
            continue
        fn, tn = payload.get("from_node"), payload.get("to_node")
        if fn is None or tn is None:
            continue
        length = float(payload.get("length_m", candidate.get("cost", 1.0)))
        graph.setdefault(int(fn), []).append((int(tn), length))
        graph.setdefault(int(tn), []).append((int(fn), length))
    return graph


def road_dijkstra(graph: Mapping[int, list[tuple[int, float]]], source: int) -> dict[int, float]:
    import heapq
    dist = {int(source): 0.0}
    pq = [(0.0, int(source))]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def road_pair_stats(export: Mapping[str, Any], selected_ids: set[str]) -> dict[str, Any]:
    graph = road_graph(export, selected_ids)
    demand_entries = []
    for demand in export.get("demand", []):
        entries = [int(node_id) for node_id in demand.get("entry_node_ids", []) if node_id is not None]
        if entries:
            demand_entries.append(entries)
    n = len(demand_entries)
    if n < 2:
        return {
            "connected_pairs": 0,
            "total_pairs": 0,
            "connectivity": 1.0,
            "fully_connected": True,
            "mean_distance": 0.0,
        }
    total_pairs = n * (n - 1) // 2
    connected = 0
    total_dist = 0.0
    for i in range(n):
        combined = {}
        for source in demand_entries[i]:
            if source not in graph:
                continue
            for node, dist in road_dijkstra(graph, source).items():
                if node not in combined or dist < combined[node]:
                    combined[node] = dist
        for j in range(i + 1, n):
            min_dist = float("inf")
            for target in demand_entries[j]:
                if target in combined:
                    min_dist = min(min_dist, combined[target])
            if min_dist < float("inf"):
                connected += 1
                total_dist += min_dist
    fully_connected = connected == total_pairs
    return {
        "connected_pairs": connected,
        "total_pairs": total_pairs,
        "connectivity": connected / total_pairs if total_pairs > 0 else 1.0,
        "fully_connected": fully_connected,
        "mean_distance": total_dist / total_pairs if fully_connected and total_pairs > 0 else 0.0,
    }


def score_road_pairdist(export: Mapping[str, Any], selected: list[Mapping[str, Any]], invalid: list[str]) -> dict[str, Any]:
    selected_ids = {str(candidate.get("action_id")) for candidate in selected}
    all_candidate_ids = {
        str(candidate.get("action_id"))
        for candidate in export.get("candidate_actions", [])
        if candidate.get("is_feasible", True)
    }
    total_cost = sum(float(candidate.get("cost") or 0.0) for candidate in selected)
    b = budget(export, total_cost)
    cost_ratio = total_cost / b if b > 0 else 0.0
    metadata = export.get("metadata", {})
    cost_weight = float(metadata.get("road_cost_ratio_weight", 0.1)) if isinstance(metadata, Mapping) else 0.1
    max_steps = export.get("max_steps")
    count_violation = max(0, len(selected) - int(max_steps)) if max_steps is not None else 0
    penalty = 0.05 * count_violation

    stats = road_pair_stats(export, selected_ids)
    ideal_stats = road_pair_stats(export, all_candidate_ids)
    if not stats["fully_connected"]:
        distance_score = 0.0
        score = -1.0 + float(stats["connectivity"]) - cost_weight * cost_ratio - penalty
    else:
        mean_d = float(stats["mean_distance"])
        ideal_mean = float(ideal_stats["mean_distance"])
        if mean_d <= 0:
            distance_score = 1.0
        elif ideal_stats["fully_connected"] and ideal_mean > 0:
            distance_score = min(1.0, ideal_mean / mean_d)
        else:
            distance_score = 0.0
        score = distance_score - cost_weight * cost_ratio - penalty

    return {
        "score": round(score, 6),
        "fully_connected": bool(stats["fully_connected"]),
        "connectivity": round(float(stats["connectivity"]), 6),
        "connected_pairs": int(stats["connected_pairs"]),
        "total_pairs": int(stats["total_pairs"]),
        "mean_pair_distance": round(float(stats["mean_distance"]), 6),
        "all_candidate_mean_pair_distance": round(float(ideal_stats["mean_distance"]), 6),
        "distance_score": round(distance_score, 6),
        "cost_ratio": round(cost_ratio, 6),
        "total_cost": round(total_cost, 6),
        "budget": round(b, 6),
        "selected_count": len(selected),
        "demand_count": len([d for d in export.get("demand", []) if d.get("entry_node_ids")]),
        "count_violation": count_violation,
        "invalid_selected": invalid,
        "cost_weight": cost_weight,
    }


def score_urban(export: Mapping[str, Any], selected: list[Mapping[str, Any]], invalid: list[str]) -> dict[str, Any]:
    metadata = export.get("metadata", {})
    requirements = metadata.get("need_config", {}) if isinstance(metadata, Mapping) else {}
    if not isinstance(requirements, Mapping):
        requirements = {}
    counts = Counter()
    block_ids = set()
    duplicate_blocks = 0
    served = set()
    total_cost = 0.0
    for candidate in selected:
        total_cost += float(candidate.get("cost") or 0.0)
        bid = block_id(candidate)
        duplicate_blocks += int(bid in block_ids)
        block_ids.add(bid)
        lu = land_use(candidate)
        if lu:
            counts[lu] += 1
        served.update(served_ids(candidate))
    required_total = sum(int(v) for v in requirements.values()) if requirements else 0
    satisfied = 0
    missing = {}
    overbuilt = {}
    for lu, required in requirements.items():
        required_int = int(required)
        have = counts.get(str(lu), 0)
        satisfied += min(have, required_int)
        if have < required_int:
            missing[str(lu)] = required_int - have
        elif have > required_int:
            overbuilt[str(lu)] = have - required_int
    requirement_score = satisfied / required_total if required_total > 0 else 0.0
    demand_weights = demand_weight_map(export)
    total_weight = sum(demand_weights.values())
    served_weight = sum(demand_weights.get(did, 0.0) for did in served)
    service_coverage = served_weight / total_weight if total_weight > 0 else 0.0
    b = budget(export, total_cost)
    cost_ratio = total_cost / b if b > 0 else 0.0
    max_steps = export.get("max_steps")
    count_violation = max(0, len(selected) - int(max_steps)) if max_steps is not None else 0
    cost_weight = float(metadata.get("urban_cost_weight", 0.2)) if isinstance(metadata, Mapping) else 0.2
    violation_penalty = 0.05 * duplicate_blocks + 0.02 * sum(overbuilt.values()) + 0.05 * count_violation
    score = 0.6 * requirement_score + 0.3 * service_coverage - cost_weight * cost_ratio - violation_penalty
    return {
        "score": round(score, 6),
        "requirement_score": round(requirement_score, 6),
        "service_coverage": round(service_coverage, 6),
        "cost_ratio": round(cost_ratio, 6),
        "served_demand_weight": round(served_weight, 6),
        "total_demand_weight": round(total_weight, 6),
        "selected_count": len(selected),
        "unique_block_count": len(block_ids),
        "duplicate_block_violation": duplicate_blocks,
        "count_violation": count_violation,
        "counts_by_land_use": dict(counts),
        "missing_by_land_use": missing,
        "overbuilt_by_land_use": overbuilt,
        "invalid_selected": invalid,
        "total_cost": round(total_cost, 6),
        "budget": round(b, 6),
        "cost_weight": cost_weight,
    }


def validate(export: Mapping[str, Any], action_ids: list[str], scores: Mapping[str, Any]) -> dict[str, Any]:
    hard_errors = []
    total_cost = float(scores.get("total_cost", 0.0))
    budget_value = export.get("budget")
    if budget_value is not None and total_cost > float(budget_value):
        hard_errors.append(f"Budget exceeded: {total_cost:.2f} > {float(budget_value):.2f}")
    max_steps = export.get("max_steps")
    if max_steps is not None and int(scores.get("selected_count", 0)) > int(max_steps):
        hard_errors.append(f"Step limit exceeded: {scores.get('selected_count')} > {max_steps}")
    if scores.get("invalid_selected"):
        hard_errors.append(f"Invalid action ids: {scores['invalid_selected'][:10]}")
    if export.get("task") == "ev_charging" and int(scores.get("duplicate_site_violation", 0)) > 0:
        hard_errors.append("Duplicate charging site configs are not allowed.")
    if export.get("task") == "urban_planning" and int(scores.get("duplicate_block_violation", 0)) > 0:
        hard_errors.append("Duplicate block assignments are not allowed.")
    return {
        "hard_errors": hard_errors,
        "warnings": [],
        "total_cost": round(total_cost, 6),
        "budget": budget_value,
        "remaining_budget": round(float(budget_value) - total_cost, 6) if budget_value is not None else None,
        "max_steps": max_steps,
        "remaining_steps": int(max_steps) - int(scores.get("selected_count", 0)) if max_steps is not None else None,
    }


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: evaluate_plan.py PLAN_JSON [EXPORT_JSON]")
    plan_path = Path(sys.argv[1])
    export_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("input/export.json")
    export = json.loads(export_path.read_text())
    action_ids = load_plan(plan_path)
    selected, invalid = valid_selected(export, action_ids)
    if export.get("task") == "road_planning" and export.get("nodes"):
        scores = score_road_pairdist(export, selected, invalid)
    elif export.get("task") == "urban_planning":
        scores = score_urban(export, selected, invalid)
    elif export.get("task") == "ev_charging" and export.get("nodes"):
        metadata = export.get("metadata", {})
        if isinstance(metadata, Mapping) and str(metadata.get("ev_score_version", "")).lower() == "v3":
            scores = score_ev_v3(export, selected, invalid)
        else:
            scores = score_ev_v2(export, selected, invalid)
    else:
        scores = score_subset(export, selected, invalid)
    validation = validate(export, action_ids, scores)
    result = {
        "plan_path": str(plan_path),
        "valid": not validation["hard_errors"],
        "validation": validation,
        "score": scores,
        "candidate_ids": [str(candidate.get("action_id")) for candidate in selected],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    outputs = Path("outputs")
    if outputs.exists() and outputs.is_dir():
        with (outputs / "eval_history.jsonl").open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


VALIDATE_PLAN_SOURCE = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys

cmd = [sys.executable, "input/evaluate_plan.py", *sys.argv[1:]]
proc = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", capture_output=True)
if proc.stdout:
    payload = json.loads(proc.stdout)
    print(json.dumps({"valid": payload.get("valid"), "validation": payload.get("validation")}, indent=2, sort_keys=True))
if proc.stderr:
    print(proc.stderr, file=sys.stderr)
raise SystemExit(proc.returncode)
'''


README_TEMPLATE = """\
# OSM v2 Workspace Sandbox

You are solving one city-planning instance in a file workspace.

## Directories

- `input/`: read-only task data and helper scripts.
- `work/`: writable scratch files.
- `outputs/`: writable plans and evaluation history.

## Important files

- `input/city_state.json`
- `input/candidate_schema.json`
- `input/candidate_summary.csv`
- `input/candidates.json`
- `input/candidates.csv`
- `input/export.json`
- `input/evaluate_plan.py`
- `input/validate_plan.py`

## EV Charging Task

For EV charging tasks, each demand parcel is assigned to the nearest selected
charging station by road network distance. The score formula is:

    score = -mean_travel_distance - cost_weight * total_cost

Selecting fewer well-positioned sites often beats selecting many redundant ones.
The `candidate_summary.csv` includes `node_id` for each site's network location.
Use `input/export.json` to access the road network graph (nodes/edges) if needed.

## Road Planning Task

For road planning tasks, selected road candidates are added to the existing
road graph. The score first rewards connecting demand-cell pairs; once all
demand pairs are connected, it rewards lower mean shortest-path distance
between demand cells:

    disconnected: score = -1 + connectivity - cost_weight * cost_ratio
    connected:    score = distance_score - cost_weight * cost_ratio

`distance_score` compares the selected network's mean demand-pair distance
against the all-candidates network. Lower pair distance is better.

## Plan format

Write a JSON file such as `outputs/final_plan.json`:

```json
{
  "candidate_ids": ["road_00001", "road_00002"]
}
```

The same format works for road, EV, and urban tasks.

## Evaluation

Run:

```bash
python input/evaluate_plan.py outputs/final_plan.json
```

It prints validation and score metrics. It also appends to
`outputs/eval_history.jsonl`.

Keep command output compact. Do not print full candidate tables or full JSON
files. Prefer Python scripts that aggregate candidates and print selected IDs,
scores, costs, validation errors, and a few ranked alternatives.

Empty plans are not successful plans. Always keep and submit the best
non-empty valid plan found so far. For urban tasks, if the `need_config`
requirements cannot be fully satisfied, still submit the current best
non-empty valid plan rather than an empty plan.

## Finish

For mini-swe-agent, finish by running:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```

The runner should then read `outputs/final_plan.json` or the best non-empty valid plan
from `outputs/eval_history.jsonl`.
"""


class OSMV2WorkspaceSandboxEnv:
    """File-workspace OSM v2 sandbox with optional bubblewrap isolation."""

    def __init__(
        self,
        data_source: str | Path | Mapping[str, Any] | CityTaskExport,
        *,
        run_root: str | Path | None = None,
        run_id: str | None = None,
        isolation: str = "auto",
        step_timeout_seconds: int = 10,
    ) -> None:
        self.data_source = data_source
        self.run_root = (Path(run_root or tempfile.gettempdir()) / "osm_v2_workspace_sandbox").resolve()
        self.run_id = run_id or f"episode_{uuid.uuid4().hex}"
        self.isolation = isolation
        self.step_timeout_seconds = int(step_timeout_seconds)
        self.run_dir = self.run_root / self.run_id
        self.input_dir = self.run_dir / "input"
        self.work_dir = self.run_dir / "work"
        self.outputs_dir = self.run_dir / "outputs"
        self.logs_dir = self.run_dir / "logs"
        self.export: CityTaskExport | None = None

    def reset(self) -> Path:
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        self.input_dir.mkdir(parents=True)
        self.work_dir.mkdir()
        self.outputs_dir.mkdir()
        self.logs_dir.mkdir()

        online_env = OSMV2OnlineSandboxEnv(self._resolved_data_source())
        try:
            online_env.reset()
            self.export = online_env.export
            sandbox_dir = Path(online_env._sandbox_dir.name)  # noqa: SLF001
            for path in sandbox_dir.iterdir():
                if path.is_file():
                    shutil.copy2(path, self.input_dir / path.name)
            export_payload = self._export_to_mapping()
            (self.input_dir / "export.json").write_text(json.dumps(export_payload, indent=2, ensure_ascii=False, default=str))
        finally:
            online_env.close()

        (self.input_dir / "evaluate_plan.py").write_text(EVALUATE_PLAN_SOURCE)
        (self.input_dir / "validate_plan.py").write_text(VALIDATE_PLAN_SOURCE)
        (self.input_dir / "README.md").write_text(README_TEMPLATE)
        os.chmod(self.input_dir / "evaluate_plan.py", 0o755)
        os.chmod(self.input_dir / "validate_plan.py", 0o755)
        return self.run_dir

    def _export_to_mapping(self) -> dict[str, Any]:
        if self.export is None:
            raise RuntimeError("reset must be called before exporting workspace data")
        return export_to_mapping(self.export)

    def _resolved_data_source(self) -> str | Path | Mapping[str, Any] | CityTaskExport:
        if isinstance(self.data_source, (str, Path)):
            return _repo_path(self.data_source)
        return self.data_source

    def execute(self, command: str) -> dict[str, Any]:
        mode = self._isolation_mode()
        if mode == "bwrap":
            argv = self._bwrap_command(command)
        elif mode == "local":
            error = self._local_command_error(command)
            if error:
                return {"returncode": 126, "output": error, "isolation": mode}
            argv = ["bash", "--noprofile", "--norc", "-lc", command]
        else:
            return {"returncode": 125, "output": f"Unknown isolation mode: {mode}", "isolation": mode}

        try:
            proc = subprocess.run(
                argv,
                cwd=self.run_dir,
                env=self._safe_env(),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.step_timeout_seconds,
            )
            return {
                "returncode": proc.returncode,
                "output": self._clip(proc.stdout),
                "output_chars": len(proc.stdout),
                "isolation": mode,
            }
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            return {
                "returncode": 124,
                "output": self._clip(output + f"\nCommand timed out after {self.step_timeout_seconds}s."),
                "isolation": mode,
            }

    def evaluate_plan(self, plan_path: str | Path = "outputs/final_plan.json") -> dict[str, Any]:
        path = Path(plan_path)
        if not path.is_absolute():
            path = self.run_dir / path
        proc = subprocess.run(
            [sys.executable, str(self.input_dir / "evaluate_plan.py"), str(path), str(self.input_dir / "export.json")],
            cwd=self.run_dir,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.step_timeout_seconds,
        )
        if proc.returncode not in {0, 1}:
            raise RuntimeError(proc.stderr or proc.stdout)
        return json.loads(proc.stdout)

    def _isolation_mode(self) -> str:
        if self.isolation == "auto":
            return "bwrap" if shutil.which("bwrap") else "local"
        return self.isolation

    def _safe_env(self) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
            "HOME": str(self.run_dir),
            "PYTHONNOUSERSITE": "1",
            "PAGER": "cat",
        }

    def _bwrap_command(self, command: str) -> list[str]:
        binds = [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            "--dir", "/workspace",
            "--ro-bind", str(self.input_dir), "/workspace/input",
            "--bind", str(self.work_dir), "/workspace/work",
            "--bind", str(self.outputs_dir), "/workspace/outputs",
            "--tmpfs", "/tmp",
            "--proc", "/proc",
            "--dev", "/dev",
            "--chdir", "/workspace",
            "--unshare-net",
            "--clearenv",
            "--setenv", "PATH", os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
            "--setenv", "HOME", "/workspace",
        ]
        executable = Path(sys.executable).resolve()
        conda_root = executable.parents[1] if len(executable.parents) > 1 else executable.parent
        if conda_root.exists() and str(conda_root).startswith("/data/"):
            binds.extend(["--ro-bind", str(conda_root), str(conda_root)])
        binds.extend(["bash", "--noprofile", "--norc", "-lc", command])
        return binds

    def _local_command_error(self, command: str) -> str:
        """Best-effort local fallback. This is not a hard isolation boundary.

        The local mode is for Docker-based development when bubblewrap is not
        installed. It should prevent obvious cross-workspace mistakes and
        destructive operations, but it should not reject normal Python analysis
        code merely because the source contains words such as ``export``.
        """
        forbidden_patterns = [
            r"(^|[\s;&|])cd\s+\.\.",
            r"\.\./",
            r"(^|[\s;&|])rm\s+(-[^\s]*[rf][^\s]*|-r|-f|--recursive|--force)\b",
            r"(^|[\s;&|])(cat|head|tail|less|more|cp|mv|rm|sed|awk)\s+/(data|home|root|etc|proc|sys|var)\b",
            r"(^|[\s;&|])cat\s+/(proc/self/environ|etc/passwd|etc/shadow)\b",
            r"(^|[\s;&|])printenv\b",
            r"(^|[\s;&|])env\s*$",
        ]
        for pattern in forbidden_patterns:
            if re.search(pattern, command):
                return (
                    "Command rejected by local workspace guard. Use relative paths under input/, work/, and outputs/. "
                    "Install bubblewrap for hard filesystem isolation."
                )
        return ""

    @staticmethod
    def _clip(text: str) -> str:
        if len(text) <= MAX_OBSERVATION_CHARS:
            return text
        head = MAX_OBSERVATION_CHARS // 2
        tail = MAX_OBSERVATION_CHARS - head
        return text[:head] + f"\n...<elided {len(text) - MAX_OBSERVATION_CHARS} chars>...\n" + text[-tail:]


__all__ = ["OSMV2WorkspaceSandboxEnv"]
