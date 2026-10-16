"""Shared reception interface; importing a case never compiles or executes it."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .profile_contracts import ProfileSelection


@dataclass(frozen=True)
class CaseSpec:
    case: Any
    layout: Any
    initial_values: Mapping[Any, Any]
    subjects: Mapping[str, Any]
    t_end: float
    dt: float
    invariants: Callable[[Any], dict[str, Any]]
    profile: str
    selection: ProfileSelection | None = None

    def __post_init__(self):
        if self.dt <= 0 or self.t_end <= 0:
            raise ValueError("a reception case needs a positive step and final time")
        if not self.initial_values and not self.subjects:
            from pops.time._program.finite_state import require_finite_state_program

            require_finite_state_program(self.case._time)
        elif not self.initial_values or not self.subjects:
            raise ValueError("initial bindings, subjects and a precise profile are required")
        if not self.profile:
            raise ValueError("a precise reception profile is required")


class MissingCapabilityError(RuntimeError):
    """A complete requested case cannot yet be built using the available primitives."""

    def __init__(self, model: str, *, required: tuple[str, ...], profile: str):
        self.model, self.required, self.profile = model, required, profile
        super().__init__(f"{model}: missing {', '.join(required)}; requested profile: {profile}")


def state_array(runtime_or_results, block: str, shape):
    """Collective native gather, or already gathered results, for reception only.

    Every MPI rank must invoke the invariant callable. A non-root gather may
    return None; the owning rank evaluates metrics and the test broadcasts them.
    This helper is not used by generated scientific equations.
    """
    import numpy as np

    value = (
        runtime_or_results[block]
        if isinstance(runtime_or_results, Mapping)
        else runtime_or_results.state_global(block)
    )
    return None if value is None else np.asarray(value).reshape(shape)


def non_owner_metrics():
    return {"passed": True, "metrics_owner": False}
