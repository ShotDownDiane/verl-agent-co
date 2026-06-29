"""Deterministic export/import schema for digital city task instances.

The classes and helpers in this module intentionally depend only on the
standard library.  They provide a stable mapping-based contract that can carry
city-data-hub outputs for urban planning, EV charging, and road planning tasks
without importing city-data-hub runtime dependencies.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Mapping

from .schemas import TaskId

VALID_TASKS = frozenset({"urban_planning", "ev_charging", "road_planning", "poi_placement"})


@dataclass(frozen=True, slots=True)
class CandidateActionExport:
    """Serializable description of one candidate city-task action."""

    action_id: str
    action_type: str = ""
    label: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    estimated_effects: dict[str, Any] = field(default_factory=dict)
    is_feasible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CityTaskExport:
    """Serializable city task instance export.

    Fields are intentionally generic so the same contract can represent all
    city-data-hub-derived task families used by the city task environments.
    """

    task: TaskId
    instance_id: str
    city_id: str = ""
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    initial_metrics: dict[str, Any] = field(default_factory=dict)
    budget: float | None = None
    max_steps: int | None = None
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    zones: list[dict[str, Any]] = field(default_factory=list)
    demand: list[dict[str, Any]] = field(default_factory=list)
    existing_assets: list[dict[str, Any]] = field(default_factory=list)
    distance_matrix: dict[str, Any] = field(default_factory=dict)
    candidate_actions: list[CandidateActionExport] = field(default_factory=list)


def _copy_dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return dict(deepcopy(value))


def _copy_list(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return list(deepcopy(value))


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric")
    return float(value)


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    return value


def _required_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _optional_str(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def candidate_from_mapping(mapping: Mapping[str, Any]) -> CandidateActionExport:
    """Convert a mapping into a validated :class:`CandidateActionExport`."""

    if not isinstance(mapping, Mapping):
        raise TypeError("candidate action must be a mapping")

    action_id = _required_str(mapping.get("action_id"), "action_id")
    cost = mapping.get("cost", 0.0)
    if not isinstance(cost, Real) or isinstance(cost, bool):
        raise TypeError("cost must be numeric")

    is_feasible = mapping.get("is_feasible", True)
    if not isinstance(is_feasible, bool):
        raise TypeError("is_feasible must be a boolean")

    return CandidateActionExport(
        action_id=action_id,
        action_type=_optional_str(mapping.get("action_type", ""), "action_type"),
        label=_optional_str(mapping.get("label", ""), "label"),
        payload=_copy_dict(mapping.get("payload"), "payload"),
        cost=float(cost),
        estimated_effects=_copy_dict(
            mapping.get("estimated_effects"), "estimated_effects"
        ),
        is_feasible=is_feasible,
        metadata=_copy_dict(mapping.get("metadata"), "metadata"),
    )


def candidate_to_mapping(candidate: CandidateActionExport) -> dict[str, Any]:
    """Convert a candidate action export into a detached plain dictionary."""

    if not isinstance(candidate, CandidateActionExport):
        raise TypeError("candidate must be a CandidateActionExport")
    return {
        "action_id": candidate.action_id,
        "action_type": candidate.action_type,
        "label": candidate.label,
        "payload": deepcopy(candidate.payload),
        "cost": candidate.cost,
        "estimated_effects": deepcopy(candidate.estimated_effects),
        "is_feasible": candidate.is_feasible,
        "metadata": deepcopy(candidate.metadata),
    }


def export_from_mapping(mapping: Mapping[str, Any]) -> CityTaskExport:
    """Convert a mapping into a validated :class:`CityTaskExport`.

    Optional mapping/list fields default to empty containers.  All container
    fields are deep-copied to avoid mutation coupling with the input mapping.
    """

    if not isinstance(mapping, Mapping):
        raise TypeError("export must be a mapping")

    task = mapping.get("task")
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")

    candidate_actions_raw = (
        mapping["candidate_actions"] if "candidate_actions" in mapping else []
    )
    if candidate_actions_raw is None:
        candidate_actions_raw = []
    if not isinstance(candidate_actions_raw, list):
        raise TypeError("candidate_actions must be a list")
    candidate_actions = [
        candidate_from_mapping(candidate) for candidate in candidate_actions_raw
    ]
    action_ids = [candidate.action_id for candidate in candidate_actions]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("candidate action ids must be unique")

    return CityTaskExport(
        task=task,  # type: ignore[arg-type]
        instance_id=_required_str(mapping.get("instance_id"), "instance_id"),
        city_id=_optional_str(mapping.get("city_id", ""), "city_id"),
        seed=_optional_int(mapping.get("seed"), "seed"),
        metadata=_copy_dict(mapping.get("metadata"), "metadata"),
        initial_metrics=_copy_dict(mapping.get("initial_metrics"), "initial_metrics"),
        budget=_optional_float(mapping.get("budget"), "budget"),
        max_steps=_optional_int(mapping.get("max_steps"), "max_steps"),
        nodes=_copy_list(mapping.get("nodes"), "nodes"),
        edges=_copy_list(mapping.get("edges"), "edges"),
        zones=_copy_list(mapping.get("zones"), "zones"),
        demand=_copy_list(mapping.get("demand"), "demand"),
        existing_assets=_copy_list(mapping.get("existing_assets"), "existing_assets"),
        distance_matrix=_copy_dict(mapping.get("distance_matrix"), "distance_matrix"),
        candidate_actions=candidate_actions,
    )


def export_to_mapping(export: CityTaskExport) -> dict[str, Any]:
    """Convert a city task export into a detached plain dictionary."""

    if not isinstance(export, CityTaskExport):
        raise TypeError("export must be a CityTaskExport")
    return {
        "task": export.task,
        "instance_id": export.instance_id,
        "city_id": export.city_id,
        "seed": export.seed,
        "metadata": deepcopy(export.metadata),
        "initial_metrics": deepcopy(export.initial_metrics),
        "budget": export.budget,
        "max_steps": export.max_steps,
        "nodes": deepcopy(export.nodes),
        "edges": deepcopy(export.edges),
        "zones": deepcopy(export.zones),
        "demand": deepcopy(export.demand),
        "existing_assets": deepcopy(export.existing_assets),
        "distance_matrix": deepcopy(export.distance_matrix),
        "candidate_actions": [
            candidate_to_mapping(candidate) for candidate in export.candidate_actions
        ],
    }
