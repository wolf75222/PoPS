"""Coupled state/geometry SSA and explicit Reynolds projection policies."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any

from pops.identity.scalar import scalar_data
from pops.time._authoring import atomic_authoring
from pops.time._program.value_validation import (
    require_compatible_spaces, require_owned, require_top_level,
)
from pops.time.values import _resolve_handle


@dataclass(frozen=True, slots=True)
class MovingFieldProjection:
    """Explicit linear density trace and source measure quadrature.

    Face weights sample the physical cells below/above an oriented face.
    Measure weights sample independently stored old/new cell measures.
    Physics belongs to the supplied rate operators, never to this policy.
    """
    face_weights: tuple[Any, Any]
    source_measure_weights: tuple[Any, Any]
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        for label, weights in (("face", self.face_weights),
                               ("source measure", self.source_measure_weights)):
            if type(weights) is not tuple or len(weights) != 2:
                raise TypeError("%s projection needs exactly two explicit weights" % label)
            for weight in weights:
                if type(weight) not in (int, float, Fraction) or not math.isfinite(float(weight)):
                    raise TypeError("projection weights must be finite scalar literals")
                scalar_data(weight)
            if sum(Fraction(weight) for weight in weights) != 1:
                raise ValueError("%s projection weights must sum exactly to one" % label)

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": 1,
                "contract": "pops://moving-field-projection/linear@1",
                "face_weights": [scalar_data(value) for value in self.face_weights],
                "source_measure_weights": [scalar_data(value) for value in self.source_measure_weights]}


class _ProgramGeometry:
    @atomic_authoring
    def geometry_state(self, state: Any, *, evolution: Any, name: str | None = None):
        """Bind an accepted physical state to its explicit evolved geometry."""
        from pops.mesh import GeometryEvolution
        self._guard_mutable("bind state geometry")
        if self._current_region() != 0:
            raise ValueError("geometry_state must be declared in the top-level Program")
        state = require_owned(self, _resolve_handle(state), "geometry_state", vtype="state")
        require_top_level(self, state, "geometry_state")
        if type(evolution) is not GeometryEvolution:
            raise TypeError("geometry_state needs an exact GeometryEvolution")
        if state.op != "state" or state.state_ref is None:
            raise ValueError("geometry_state needs the actual accepted state instance")
        if state.space is None or state.space.centering != "cell" or \
                state.space.representation not in {"conservative", "cell_average"}:
            raise ValueError("moving state needs a complete cell-average StateSpace")
        if state.space.frame != evolution.frame_id:
            raise ValueError("moving state and coordinate law need the same exact frame")
        if any(clock != state.clock for expression in evolution.coordinate_map
               for clock in expression.time_clocks()):
            raise ValueError("coordinate law must use the exact owning physical-state clock")
        if state.state_ref in self._geometry_states:
            raise ValueError("a physical state has exactly one geometry binding")
        value = self._new("state_geometry", "geometry_state", (state,),
            {"schema_version": 1, "evolution": evolution.to_data()},
            name or "geometry_" + state.name, state.block, space=state.space,
            state_ref=state.state_ref, point=state.point)
        self._geometry_states[state.state_ref] = value
        return value

    @atomic_authoring
    def reynolds_update(self, geometry: Any, *, physical_rate: Any, projection: MovingFieldProjection,
                        geometry_tolerance: Any, source_rate: Any = None, at: Any, name: str | None = None):
        """Author one coupled finite-volume state/geometry candidate.

        Physical flux/source bodies retain their operator handles and original
        Equation. The endpoint coordinate law, density trace, source measure
        projection and complete interval are explicit serialized dependencies.
        """
        from pops.time.points import TimePoint
        self._guard_mutable("author a Reynolds update")
        geometry = require_owned(self, geometry, "reynolds_update geometry", vtype="state_geometry")
        require_top_level(self, geometry, "reynolds_update geometry")
        if geometry.op != "geometry_state" or self._geometry_states.get(geometry.state_ref) is not geometry:
            raise ValueError("Reynolds update requires the issued accepted geometry binding")
        state = geometry.inputs[0]
        physical_rate = require_owned(self, _resolve_handle(physical_rate), "Reynolds physical rate", vtype="rhs")
        if physical_rate.op not in {"principal_rate", "rhs"} or \
                (physical_rate.op == "rhs" and physical_rate.attrs.get("operator_handle") is None):
            raise ValueError("Reynolds physical flux needs its actual principal face evaluation")
        rates = (physical_rate,) if source_rate is None else (
            physical_rate, require_owned(self, _resolve_handle(source_rate), "Reynolds source rate", vtype="rhs"))
        for rate in rates:
            require_top_level(self, rate, "Reynolds rate")
            if rate.block != state.block or rate.state_ref != state.state_ref or rate.point != state.point:
                raise ValueError("Reynolds rates must sample the same exact state/clock/point")
            require_compatible_spaces(state.space, rate.space, "Reynolds rate", typed_pair=True)
            def samples(value):
                if value.op == "linear_combine" and value.attrs.get("physical_balance") is not None:
                    return tuple(leaf for item in value.inputs for leaf in samples(item))
                return value.inputs
            sampled = samples(rate)
            if not sampled or any(value is not state for value in sampled):
                raise ValueError("Reynolds evaluation requires complete single-state sampling")
        if type(projection) is not MovingFieldProjection:
            raise TypeError("Reynolds update needs an explicit MovingFieldProjection")
        if type(geometry_tolerance) not in (int, float, Fraction) or \
                not math.isfinite(float(geometry_tolerance)) or geometry_tolerance < 0:
            raise ValueError("geometry tolerance must be finite and nonnegative")
        if type(at) is not TimePoint or at != TimePoint(state.clock, step=1):
            raise ValueError("Reynolds update currently requires the complete root U.next interval")
        return self._new("state_geometry", "reynolds_update", (geometry, *rates),
            {"schema_version": 1, "projection": projection.to_data(),
             "geometry_tolerance": scalar_data(geometry_tolerance),
             "interval": {"begin": state.point.to_data(), "end": at.to_data(),
                          "physical_time_quadrature": "left_endpoint@1",
                          "swept_measure": "exact_endpoint_displacement@1"},
             "source_rate_input": None if source_rate is None else 2},
            name or "reynolds_" + state.name, state.block, space=state.space,
            state_ref=state.state_ref, point=at)


__all__ = ["MovingFieldProjection"]
