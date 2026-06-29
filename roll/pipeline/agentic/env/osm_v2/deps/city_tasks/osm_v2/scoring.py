"""Shared scoring for OSM v2 city-task instances.

The functions here are intentionally deterministic and mapping-based so the
online sandbox, heuristic baselines, and DRL wrappers can evaluate exactly the
same plan under the same task definition.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

try:
    from env.city_tasks.common.export_schema import CityTaskExport
except ModuleNotFoundError:
    try:
        from city_tasks.common.export_schema import CityTaskExport
    except ModuleNotFoundError:
        from ..common.export_schema import CityTaskExport


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
    for item in export.demand:
        did = item.get("id", item.get("parcel_id", item.get("demand_id")))
        if did is None:
            continue
        weight = item.get("demand_weight", item.get("weight", 1.0))
        try:
            weights[str(did)] = float(weight)
        except (TypeError, ValueError):
            weights[str(did)] = 1.0
    return weights


def _candidate_maps(export: CityTaskExport) -> dict[str, Any]:
    return {candidate.action_id: candidate for candidate in export.candidate_actions}


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
    if isinstance(entity, Mapping) and entity.get("id") is not None:
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
    if isinstance(assignment, Mapping) and assignment.get("land_use") is not None:
        return str(assignment["land_use"])
    return ""


def _budget(export: CityTaskExport, total_cost: float) -> float:
    if export.budget is not None and export.budget > 0:
        return float(export.budget)
    return max(total_cost, 1.0)


def score_osm_v2_plan(export: CityTaskExport, action_ids: list[str]) -> dict[str, Any]:
    """Score a v2 plan represented by flattened candidate action ids."""

    candidates = _candidate_maps(export)
    valid_selected = []
    invalid_selected = []
    seen = set()
    for action_id in action_ids:
        aid = str(action_id)
        candidate = candidates.get(aid)
        if candidate is None or not candidate.is_feasible:
            invalid_selected.append(aid)
            continue
        if aid in seen:
            continue
        seen.add(aid)
        valid_selected.append(candidate)

    if export.task == "road_planning":
        return _score_road_pair_distance(export, valid_selected, invalid_selected)
    if export.task == "urban_planning":
        return _score_urban(export, valid_selected, invalid_selected)
    if export.task == "ev_charging" and str(export.metadata.get("ev_score_version", "")).lower() == "v3":
        return _score_ev_v3(export, valid_selected, invalid_selected)
    return _score_subset(export, valid_selected, invalid_selected)


def _score_subset(
    export: CityTaskExport,
    selected_candidates: list[Any],
    invalid_selected: list[str],
) -> dict[str, Any]:
    demand_weights = _demand_weight_map(export)
    total_weight = sum(demand_weights.values())
    if total_weight <= 0:
        total_weight = float(len(export.demand))
        demand_weights = {
            str(item.get("id", item.get("parcel_id", idx))): 1.0
            for idx, item in enumerate(export.demand)
        }

    served: set[str] = set()
    total_cost = 0.0
    selected_sites: set[str] = set()
    duplicate_sites = 0
    total_supply = 0.0

    for candidate in selected_candidates:
        total_cost += float(candidate.cost)
        sid = _site_id(candidate)
        if sid in selected_sites:
            duplicate_sites += 1
        selected_sites.add(sid)
        for did in _served_ids(candidate):
            served.add(str(did))
        supply = candidate.estimated_effects.get(
            "expected_supply",
            candidate.payload.get("expected_supply", candidate.payload.get("charger_supply", 0.0)),
        )
        try:
            total_supply += float(supply)
        except (TypeError, ValueError):
            pass

    served_weight = sum(demand_weights.get(did, 0.0) for did in served)
    coverage = served_weight / total_weight if total_weight > 0 else 0.0
    budget = _budget(export, total_cost)
    cost_ratio = total_cost / budget if budget > 0 else 0.0
    count_limit = export.max_steps
    count_violation = max(0, len(selected_candidates) - count_limit) if count_limit is not None else 0
    site_violation = duplicate_sites
    penalty = 0.05 * count_violation + 0.05 * site_violation

    cost_weight = float(export.metadata.get("cost_weight", 0.5))
    score = coverage - cost_weight * cost_ratio - penalty
    return {
        "score": round(score, 6),
        "coverage": round(coverage, 6),
        "weighted_coverage": round(coverage, 6),
        "cost_ratio": round(cost_ratio, 6),
        "served_demand_weight": round(served_weight, 6),
        "total_demand_weight": round(total_weight, 6),
        "unique_demand_served": len(served),
        "total_demand_zones": len(export.demand),
        "selected_count": len(selected_candidates),
        "unique_site_count": len(selected_sites),
        "duplicate_site_violation": duplicate_sites,
        "count_violation": count_violation,
        "invalid_selected": invalid_selected,
        "total_cost": round(total_cost, 6),
        "budget": round(budget, 6),
        "expected_supply": round(total_supply, 6),
        "cost_weight": cost_weight,
    }


def _score_urban(
    export: CityTaskExport,
    selected_candidates: list[Any],
    invalid_selected: list[str],
) -> dict[str, Any]:
    requirements = export.metadata.get("need_config", {})
    if not isinstance(requirements, Mapping):
        requirements = {}

    counts: Counter[str] = Counter()
    block_ids: set[str] = set()
    duplicate_blocks = 0
    served: set[str] = set()
    total_cost = 0.0

    for candidate in selected_candidates:
        total_cost += float(candidate.cost)
        bid = _block_id(candidate)
        if bid in block_ids:
            duplicate_blocks += 1
        block_ids.add(bid)
        land_use = _land_use(candidate)
        if land_use:
            counts[land_use] += 1
        for did in _served_ids(candidate):
            served.add(str(did))

    required_total = sum(int(v) for v in requirements.values()) if requirements else 0
    satisfied = 0
    missing: dict[str, int] = {}
    overbuilt: dict[str, int] = {}
    for land_use, required in requirements.items():
        required_int = int(required)
        have = counts.get(str(land_use), 0)
        satisfied += min(have, required_int)
        if have < required_int:
            missing[str(land_use)] = required_int - have
        elif have > required_int:
            overbuilt[str(land_use)] = have - required_int

    requirement_score = satisfied / required_total if required_total > 0 else 0.0
    demand_weights = _demand_weight_map(export)
    total_weight = sum(demand_weights.values())
    served_weight = sum(demand_weights.get(did, 0.0) for did in served)
    service_coverage = served_weight / total_weight if total_weight > 0 else 0.0
    budget = _budget(export, total_cost)
    cost_ratio = total_cost / budget if budget > 0 else 0.0
    count_limit = export.max_steps
    count_violation = max(0, len(selected_candidates) - count_limit) if count_limit is not None else 0
    cost_weight = float(export.metadata.get("urban_cost_weight", 0.2))
    violation_penalty = (
        0.05 * duplicate_blocks
        + 0.02 * sum(overbuilt.values())
        + 0.05 * count_violation
    )
    score = (
        0.6 * requirement_score
        + 0.3 * service_coverage
        - cost_weight * cost_ratio
        - violation_penalty
    )
    return {
        "score": round(score, 6),
        "requirement_score": round(requirement_score, 6),
        "service_coverage": round(service_coverage, 6),
        "cost_ratio": round(cost_ratio, 6),
        "served_demand_weight": round(served_weight, 6),
        "total_demand_weight": round(total_weight, 6),
        "selected_count": len(selected_candidates),
        "unique_block_count": len(block_ids),
        "duplicate_block_violation": duplicate_blocks,
        "count_violation": count_violation,
        "counts_by_land_use": dict(counts),
        "missing_by_land_use": missing,
        "overbuilt_by_land_use": overbuilt,
        "invalid_selected": invalid_selected,
        "total_cost": round(total_cost, 6),
        "budget": round(budget, 6),
        "cost_weight": cost_weight,
    }


# ---------------------------------------------------------------------------
# Road planning v2: pair-distance scoring
# ---------------------------------------------------------------------------

import heapq
from collections import defaultdict


def _build_road_graph(
    edges: list[dict[str, Any]],
    candidates: list[Any] | None = None,
    selected_ids: set[str] | None = None,
) -> dict[int, list[tuple[int, float]]]:
    """Build adjacency list from edges with from_node/to_node."""
    graph: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for e in edges:
        fn = e.get("from_node")
        tn = e.get("to_node")
        if fn is None or tn is None:
            continue
        length = float(e.get("length_m", 1.0))
        graph[fn].append((tn, length))
        graph[tn].append((fn, length))
    if candidates and selected_ids:
        for c in candidates:
            if c.action_id not in selected_ids:
                continue
            payload = c.payload
            fn = payload.get("from_node")
            tn = payload.get("to_node")
            if fn is None or tn is None:
                continue
            length = float(payload.get("length_m", 1.0))
            graph[fn].append((tn, length))
            graph[tn].append((fn, length))
    return dict(graph)


def _dijkstra_all(graph: dict[int, list[tuple[int, float]]], source: int) -> dict[int, float]:
    """Single-source shortest paths from source."""
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


def _road_pair_distance_stats(export: CityTaskExport, selected_ids: set[str]) -> dict[str, Any]:
    graph = _build_road_graph(
        export.edges,
        candidates=export.candidate_actions,
        selected_ids=selected_ids,
    )

    demand_entries: list[list[int]] = []
    for demand in export.demand:
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
        combined: dict[int, float] = {}
        for source in demand_entries[i]:
            if source not in graph:
                continue
            for node, dist in _dijkstra_all(graph, source).items():
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
    mean_distance = total_dist / total_pairs if fully_connected and total_pairs > 0 else 0.0
    return {
        "connected_pairs": connected,
        "total_pairs": total_pairs,
        "connectivity": connected / total_pairs if total_pairs > 0 else 1.0,
        "fully_connected": fully_connected,
        "mean_distance": mean_distance,
    }


def _score_road_pair_distance(
    export: CityTaskExport,
    selected_candidates: list[Any],
    invalid_selected: list[str],
    *,
    cost_weight_override: float | None = None,
) -> dict[str, Any]:
    """Score road plans by demand-pair shortest-path distance.

    Disconnected plans are ranked by connected demand-pair ratio on a small
    negative scale. Fully connected plans are ranked by how close their mean
    pair distance is to the all-candidates network, with a small cost-ratio
    penalty.
    """
    selected_ids = {candidate.action_id for candidate in selected_candidates}
    all_candidate_ids = {candidate.action_id for candidate in export.candidate_actions if candidate.is_feasible}
    total_cost = 0.0
    for candidate in selected_candidates:
        total_cost += float(candidate.cost)

    stats = _road_pair_distance_stats(export, selected_ids)
    ideal_stats = _road_pair_distance_stats(export, all_candidate_ids)
    budget = _budget(export, total_cost)
    cost_ratio = total_cost / budget if budget > 0 else 0.0
    cost_weight = (
        float(cost_weight_override)
        if cost_weight_override is not None
        else float(export.metadata.get("road_cost_ratio_weight", 0.1))
    )
    count_limit = export.max_steps
    count_violation = max(0, len(selected_candidates) - count_limit) if count_limit is not None else 0
    penalty = 0.05 * count_violation

    if not stats["fully_connected"]:
        distance_score = 0.0
        score = -1.0 + float(stats["connectivity"]) - cost_weight * cost_ratio - penalty
    else:
        mean_distance = float(stats["mean_distance"])
        ideal_mean = float(ideal_stats["mean_distance"])
        if mean_distance <= 0:
            distance_score = 1.0
        elif ideal_stats["fully_connected"] and ideal_mean > 0:
            distance_score = min(1.0, ideal_mean / mean_distance)
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
        "selected_count": len(selected_ids),
        "demand_count": len([d for d in export.demand if d.get("entry_node_ids")]),
        "count_violation": count_violation,
        "invalid_selected": invalid_selected,
        "cost_weight": cost_weight,
    }


def score_road_plan_v2(
    export: CityTaskExport,
    action_ids: list[str],
    *,
    cost_weight: float | None = None,
) -> dict[str, Any]:
    """Score a road plan using normalized demand-pair distance."""
    candidates = _candidate_maps(export)
    selected = []
    invalid = []
    seen = set()
    for action_id in action_ids:
        aid = str(action_id)
        candidate = candidates.get(aid)
        if candidate is None or not candidate.is_feasible:
            invalid.append(aid)
            continue
        if aid in seen:
            continue
        seen.add(aid)
        selected.append(candidate)
    return _score_road_pair_distance(export, selected, invalid, cost_weight_override=cost_weight)


# ---------------------------------------------------------------------------
# EV charging v2: nearest-assignment network distance scoring
# ---------------------------------------------------------------------------


def score_ev_plan_v2(
    export: CityTaskExport,
    action_ids: list[str],
    *,
    cost_weight: float = 0.01,
) -> dict[str, Any]:
    """Score an EV plan using nearest-assignment network distance.

    Each demand is assigned to the nearest selected station (by network distance).
    score = -mean_travel_distance - cost_weight * total_cost
    """
    candidates_map = _candidate_maps(export)
    selected_ids: set[str] = set()
    total_cost = 0.0
    selected_node_ids: set[int] = set()

    for aid in action_ids:
        c = candidates_map.get(str(aid))
        if c is not None and c.is_feasible and str(aid) not in selected_ids:
            selected_ids.add(str(aid))
            total_cost += float(c.cost)
            node_id = c.payload.get("node_id")
            if node_id is not None:
                selected_node_ids.add(int(node_id))

    if not selected_node_ids:
        return {
            "score": 0.0,
            "mean_travel_distance": 0.0,
            "total_cost": round(total_cost, 6),
            "selected_count": len(selected_ids),
            "demand_served": 0,
            "demand_total": len(export.demand),
        }

    graph = _build_road_graph(export.edges)

    demand_entries: list[tuple[str, list[int], float]] = []
    for d in export.demand:
        entries = d.get("entry_node_ids", [])
        weight = float(d.get("demand_weight", 1.0))
        did = str(d.get("id", ""))
        if entries:
            demand_entries.append((did, entries, weight))

    if not demand_entries:
        return {
            "score": 0.0,
            "mean_travel_distance": 0.0,
            "total_cost": round(total_cost, 6),
            "selected_count": len(selected_ids),
            "demand_served": 0,
            "demand_total": len(export.demand),
        }

    # Dijkstra from each selected site node
    site_dists: list[dict[int, float]] = []
    for node_id in selected_node_ids:
        site_dists.append(_dijkstra_all(graph, node_id))

    # Assign each demand to nearest selected site
    total_weighted_dist = 0.0
    total_weight = 0.0
    demand_served = 0

    for did, entries, weight in demand_entries:
        min_dist = float("inf")
        for sd in site_dists:
            for e in entries:
                if e in sd:
                    min_dist = min(min_dist, sd[e])
        if min_dist < float("inf"):
            total_weighted_dist += weight * min_dist
            total_weight += weight
            demand_served += 1

    if total_weight <= 0 or demand_served == 0:
        return {
            "score": 0.0,
            "mean_travel_distance": 0.0,
            "total_cost": round(total_cost, 6),
            "selected_count": len(selected_ids),
            "demand_served": 0,
            "demand_total": len(export.demand),
        }

    mean_travel_dist = total_weighted_dist / total_weight
    score = -mean_travel_dist - cost_weight * total_cost

    return {
        "score": round(score, 6),
        "mean_travel_distance": round(mean_travel_dist, 6),
        "total_cost": round(total_cost, 6),
        "selected_count": len(selected_ids),
        "demand_served": demand_served,
        "demand_total": len(export.demand),
        "cost_weight": cost_weight,
    }


# ---------------------------------------------------------------------------
# EV charging v3: capacity/time-aware service scoring
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _ev_candidate_supply(candidate: Any) -> float:
    supply = candidate.estimated_effects.get(
        "expected_supply",
        candidate.payload.get("expected_supply", candidate.payload.get("charger_supply")),
    )
    value = _safe_float(supply, -1.0)
    if value >= 0:
        return value
    chargers = candidate.payload.get("chargers", {})
    if isinstance(chargers, Mapping):
        return (
            _safe_float(chargers.get("slow"), 0.0)
            + 3.0 * _safe_float(chargers.get("medium"), 0.0)
            + 8.0 * _safe_float(chargers.get("fast"), 0.0)
        )
    return 0.0


def _ev_distance_table(export: CityTaskExport, site_ids: set[str]) -> dict[str, dict[str, float]]:
    matrix = getattr(export, "distance_matrix", {}) or {}
    if isinstance(matrix, Mapping) and matrix:
        out: dict[str, dict[str, float]] = {}
        for sid in site_ids:
            raw = matrix.get(sid, {})
            if not isinstance(raw, Mapping):
                continue
            out[sid] = {str(did): _safe_float(value, float("inf")) for did, value in raw.items()}
        return out

    site_nodes: dict[str, int] = {}
    for candidate in export.candidate_actions:
        if not candidate.is_feasible:
            continue
        sid = _site_id(candidate)
        node_id = candidate.payload.get("node_id")
        if sid in site_ids and node_id is not None and sid not in site_nodes:
            site_nodes[sid] = int(node_id)

    demand_entries: dict[str, list[int]] = {}
    for idx, demand in enumerate(export.demand):
        did = str(demand.get("id", demand.get("parcel_id", idx)))
        entries = [int(node_id) for node_id in demand.get("entry_node_ids", []) if node_id is not None]
        if entries:
            demand_entries[did] = entries

    graph = _build_road_graph(export.edges)
    out: dict[str, dict[str, float]] = {}
    for sid, node_id in site_nodes.items():
        dist = _dijkstra_all(graph, node_id)
        row: dict[str, float] = {}
        for did, entries in demand_entries.items():
            best = min((dist[e] for e in entries if e in dist), default=float("inf"))
            if best < float("inf"):
                row[did] = best
        out[sid] = row
    return out


def _ev_weighted_nearest_mean(
    distance_by_site: Mapping[str, Mapping[str, float]],
    site_ids: set[str],
    demand_weights: Mapping[str, float],
) -> tuple[float, float]:
    total_dist = 0.0
    total_weight = 0.0
    for did, weight in demand_weights.items():
        best = min(
            (
                _safe_float(distance_by_site.get(sid, {}).get(did), float("inf"))
                for sid in site_ids
            ),
            default=float("inf"),
        )
        if best < float("inf"):
            total_dist += weight * best
            total_weight += weight
    return (total_dist / total_weight if total_weight > 0 else 0.0, total_weight)


def _score_ev_v3(
    export: CityTaskExport,
    selected_candidates: list[Any],
    invalid_selected: list[str],
) -> dict[str, Any]:
    metadata = export.metadata if isinstance(export.metadata, Mapping) else {}
    demand_weights = _demand_weight_map(export)
    total_demand_weight = sum(demand_weights.values())

    total_cost = 0.0
    duplicate_sites = 0
    selected_sites: set[str] = set()
    site_infos: dict[str, dict[str, float]] = {}
    supply_unit = _safe_float(metadata.get("ev_supply_unit_weight"), 50.0)

    for candidate in selected_candidates:
        total_cost += float(candidate.cost)
        sid = _site_id(candidate)
        if sid in selected_sites:
            duplicate_sites += 1
            continue
        selected_sites.add(sid)
        supply = _ev_candidate_supply(candidate)
        site_infos[sid] = {
            "supply": supply,
            "capacity_weight": max(0.0, supply * supply_unit),
        }

    budget = _budget(export, total_cost)
    cost_ratio = total_cost / budget if budget > 0 else 0.0
    count_limit = export.max_steps
    count_violation = max(0, len(selected_candidates) - count_limit) if count_limit is not None else 0

    cost_weight = _safe_float(metadata.get("ev_cost_ratio_weight"), 0.2)
    queue_weight = _safe_float(metadata.get("ev_queue_assignment_weight"), 0.15)
    distance_scale = max(1.0, _safe_float(metadata.get("ev_distance_scale_m"), 2000.0))
    overload_penalty_weight = _safe_float(metadata.get("ev_overload_penalty_weight"), 0.10)
    unserved_penalty_weight = _safe_float(metadata.get("ev_unserved_penalty_weight"), 0.20)

    all_site_ids = {_site_id(candidate) for candidate in export.candidate_actions if candidate.is_feasible}
    needed_site_ids = set(site_infos) | all_site_ids
    distance_by_site = _ev_distance_table(export, needed_site_ids)

    if not site_infos or total_demand_weight <= 0:
        penalty = 0.05 * duplicate_sites + 0.05 * count_violation
        return {
            "score": round(-cost_weight * cost_ratio - penalty, 6) if selected_candidates else 0.0,
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
            "budget": round(budget, 6),
            "cost_ratio": round(cost_ratio, 6),
            "selected_count": len(selected_candidates),
            "unique_site_count": len(site_infos),
            "duplicate_site_violation": duplicate_sites,
            "count_violation": count_violation,
            "invalid_selected": invalid_selected,
            "cost_weight": cost_weight,
        }

    assignments: dict[str, tuple[str, float]] = {}
    loads = {sid: 0.0 for sid in site_infos}
    pressures = {sid: 0.0 for sid in site_infos}
    for _ in range(3):
        next_assignments: dict[str, tuple[str, float]] = {}
        for did, weight in demand_weights.items():
            best_sid = None
            best_distance = float("inf")
            best_generalized = float("inf")
            for sid in site_infos:
                distance = _safe_float(distance_by_site.get(sid, {}).get(did), float("inf"))
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
    ideal_mean_distance, ideal_assigned_weight = _ev_weighted_nearest_mean(
        distance_by_site,
        all_site_ids,
        demand_weights,
    )
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
        "budget": round(budget, 6),
        "cost_ratio": round(cost_ratio, 6),
        "selected_count": len(selected_candidates),
        "unique_site_count": len(site_infos),
        "duplicate_site_violation": duplicate_sites,
        "count_violation": count_violation,
        "demand_served": len(assignments),
        "demand_total": len(export.demand),
        "invalid_selected": invalid_selected,
        "cost_weight": cost_weight,
    }


def score_ev_plan_v3(export: CityTaskExport, action_ids: list[str]) -> dict[str, Any]:
    """Score an EV plan using capacity/time-aware service scoring."""
    candidates = _candidate_maps(export)
    selected = []
    invalid = []
    seen = set()
    for action_id in action_ids:
        aid = str(action_id)
        candidate = candidates.get(aid)
        if candidate is None or not candidate.is_feasible:
            invalid.append(aid)
            continue
        if aid in seen:
            continue
        seen.add(aid)
        selected.append(candidate)
    return _score_ev_v3(export, selected, invalid)
