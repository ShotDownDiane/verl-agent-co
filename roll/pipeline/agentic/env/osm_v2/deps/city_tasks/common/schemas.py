"""Dependency-light schemas shared by digital city task environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TaskId = Literal["urban_planning", "ev_charging", "road_planning", "poi_placement"]


@dataclass(slots=True)
class CityAction:
    """An action exposed by a city task environment."""

    action_id: str
    action_type: str
    label: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CityState:
    """Task-agnostic state snapshot for a digital city environment."""

    task: TaskId
    instance_id: str
    step_idx: int
    entities: list[dict[str, Any]] = field(default_factory=list)
    available_actions: list[CityAction] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CityStepResult:
    """Result returned by one city environment transition."""

    state: CityState
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
