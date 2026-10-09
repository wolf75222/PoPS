"""Explicit clock-domain crossing for Program authoring."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time.points import StagePoint, TimePoint, point_clock
from pops.time._program.value_validation import require_owned
from pops.time._schedule.synchronization import relation_data
from pops.time.values import _resolve_handle

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramClocks(_ProgramBase):
    def stage(self, name: str, *, c: Any) -> StagePoint:
        """Declare a fresh stage at exact fraction ``c`` of this Program's local step.

        Reusing a name or an abscissa still creates a distinct stage. The name is a
        diagnostic label; the Program-local ordinal persists through graph rebuilds.
        Use ``StagePoint`` directly for partitioned or explicitly clocked coordinates.
        """
        self._guard_mutable("declare a temporal stage")
        point = StagePoint(
            name, {"main": TimePoint(self.clock, c)}, identity=self._next_stage_identity)
        self._next_stage_identity += 1
        return point

    def requested_dt(self) -> Any:
        """Read the numerical step request as a runtime Scalar, without evaluating it in Python."""
        self._guard_mutable("read the requested duration")
        return self._new("scalar", "requested_dt", (), {}, "requested_dt", None)

    def reached_duration(self, duration: Any) -> Any:
        """Declare the effective duration returned by this Program's candidate.

        All commits share this computed frontier. Earlier stages retain their
        requested-step coordinates. Version 1 permits one uniform cadence 1/1
        invocation with no spatial interval exchanges; unsupported compositions
        fail during preparation, rather than relabelling their measures.
        """
        from pops.time._program.value_validation import require_top_level
        self._guard_mutable("declare a computed temporal frontier")
        duration = self._canonical_value(duration)
        require_top_level(self, duration, "reached_duration")
        if duration.vtype != "scalar":
            raise TypeError("reached_duration requires a Scalar from this Program")
        if self._recording:
            raise ValueError("reached_duration requires the top-level Program region")
        if any(value.op == "reached_duration" for value in self._values):
            raise ValueError("reached_duration may be declared only once")
        return self._new("scalar", "reached_duration", (duration,),
                         {"schema_version": 1, "interval_rule": "no_spatial_exchanges"},
                         "reached_duration", duration.block)

    def synchronize(
        self, value: Any, *, at: Any, relation: Any, name: Any = None
    ) -> Any:
        """Transfer a value to another clock through one explicit typed relation."""
        value = _resolve_handle(value)
        require_owned(self, value, "Program.synchronize")
        if type(at) not in (TimePoint, StagePoint):
            raise TypeError("Program.synchronize at= must be an exact TimePoint or StagePoint")
        # A partitioned stage with distinct explicit/implicit abscissae is not one transfer point.
        # Force the caller to select ``stage.time_for(partition)`` instead of silently choosing one.
        if type(at) is StagePoint:
            _ = at.time
        target_clock = point_clock(at, "Program.synchronize")
        if value.clock == target_clock:
            raise ValueError("Program.synchronize requires distinct source and target clocks")
        return self._new(
            value.vtype,
            "synchronize",
            (value,),
            {
                "source_clock": value.clock.to_data(),
                "relation": relation_data(relation, source=value, target=at),
            },
            name,
            value.block,
            space=value.space,
            field_context=value.field_context,
            state_ref=value.state_ref,
            point=at,
        )

    def temporal_manifest(self) -> dict[str, Any]:
        """Return the canonical clock/history/schedule contract required by strict restart."""
        from pops.time._program.temporal_manifest import build_temporal_manifest

        return build_temporal_manifest(self)


__all__ = ["_ProgramClocks"]
