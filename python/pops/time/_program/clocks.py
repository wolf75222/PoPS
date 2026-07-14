"""Explicit clock-domain crossing for Program authoring."""
from __future__ import annotations

from typing import Any

from pops.time.points import StagePoint, TimePoint, point_clock
from pops.time._program.value_validation import require_owned
from pops.time._schedule.synchronization import relation_data
from pops.time.values import _resolve_handle


class _ProgramClocks:
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
                "relation": relation_data(relation),
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
