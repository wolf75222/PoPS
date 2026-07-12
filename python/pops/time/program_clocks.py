"""Explicit clock-domain crossing for Program authoring."""
from __future__ import annotations

from typing import Any

from pops.time.points import StagePoint, TimePoint
from pops.time.program_value_validation import require_owned
from pops.time.synchronization import relation_data
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


__all__ = ["_ProgramClocks"]
