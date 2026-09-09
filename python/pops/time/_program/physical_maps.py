"""Public stage-qualified maps; numerical work is compiled into native continuations."""
from __future__ import annotations

import hashlib
from typing import Any

from pops.time._authoring import atomic_authoring
from pops.time._program.value_validation import require_owned
from pops.time.values import ProgramValue, _resolve_handle


class _ProgramPhysicalMaps:
    @atomic_authoring
    def map(self, physical_map: Any, *, source: Any, target: Any) -> Any:
        """Map one materialized state into an explicit target stage at the same time.

        ``physical_map`` describes supports, axes and integration measures. Layouts and the
        authenticated native provider are bound later by ``pops.resolve``. Each invocation is an
        independent dependency and scratch result; neither accepted state is overwritten here.
        """
        from pops.identity.encoding import canonical_bytes
        from pops.mesh import LayoutMappingPort, LayoutRepresentation, PhysicalSupportMap
        from pops.time.points import StagePoint

        self._guard_mutable("author a physical map")
        if self._current_region() != 0:
            raise ValueError("Program.map requires a top-level region barrier")
        if type(physical_map) is not PhysicalSupportMap:
            raise TypeError("Program.map requires an exact PhysicalSupportMap")
        target = self._require_stage(target, "Program.map target")
        if target in self._time_stage_values:
            raise ValueError("Program.map target stage is already defined")
        source = _resolve_handle(source)
        if not isinstance(source, ProgramValue) or source.vtype != "state":
            raise TypeError("Program.map source must be a materialized State or StageHandle")
        require_owned(self, source, "Program.map source")
        if source.state_ref is None or source.block == target.block:
            raise ValueError("Program.map requires qualified source and target states in distinct blocks")
        source_time = source.point.time if type(source.point) is StagePoint else source.point
        if source_time != target.point.time:
            raise ValueError("Program.map source and target must have the same exact clock coordinate")
        from pops.time.references import canonical_handle
        physical_map.validate_ports(
            LayoutMappingPort(canonical_handle(source.state_ref), LayoutRepresentation.CELL_AVERAGE_V1),
            LayoutMappingPort(canonical_handle(target.state), LayoutRepresentation.CELL_AVERAGE_V1))
        declaration = {
            "schema_version": 1,
            "physical_map": physical_map.to_data(),
            "source_state": source.state_ref,
            "target_state": target.state,
            "source_point": source.point,
            "target_point": target.point,
        }
        identity_data = {
            "physical_map": physical_map.to_data(),
            "source_state": canonical_handle(source.state_ref).canonical_identity(),
            "target_state": canonical_handle(target.state).canonical_identity(),
            "source_point": source.point.to_data(),
            "target_point": target.point.to_data(),
        }
        invocation = "pops.program-map.v1::" + hashlib.sha256(canonical_bytes(identity_data)).hexdigest()
        declaration["invocation"] = invocation
        self._new("state", "layout_map_export", (source,), declaration,
                  "map_export_" + target.key, source.block, space=source.space,
                  state_ref=source.state_ref, point=source.point)
        state = self._time_states[self._time_state_key(target.block, target.state, target.clock)]
        shape = self._current_time_value(state)
        imported = self._new("state", "layout_map_import", (shape,), declaration,
                             "map_import_" + target.key, target.block, space=target.space,
                             state_ref=target.state, point=target.point)
        # The paired export/import edge is explicit channel data. Keeping the import's ordinary
        # inputs local allows exact per-layout slicing without duplicating a foreign numerical op.
        self._time_stage_values[target] = imported
        return imported
