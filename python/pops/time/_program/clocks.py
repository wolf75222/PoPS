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
    def attempt_start(self, state: Any) -> Any:
        """Immutable native macro-entry State, inside ``after_synchronization``.

        This is the beginning of the entire hierarchy attempt, including when
        refined levels took several provisional substeps.
        """
        self._guard_mutable("read the macro-entry State")
        if not self._post_sync_recording:
            raise ValueError("attempt_start requires an after_synchronization region")
        state = require_owned(self, _resolve_handle(state), "attempt_start", vtype="state")
        if state.op != "state" or state.block is None:
            raise ValueError("attempt_start requires a declared current State")
        return self._new("state", "attempt_start", (state,), {"schema_version": 1},
                         None, state.block, space=state.space, state_ref=state.state_ref,
                         point=state.point)

    def composite_pair(self, left: Any, right: Any, *, weights: Any) -> Any:
        """Physical-volume pairing on the finest active cells of immutable towers.

        Operands are a declared current State (the refluxed provisional tower)
        or its ``attempt_start`` snapshot. Weights are explicit finite component
        coefficients. This first contract requires Cartesian cell averages.
        """
        import math
        self._guard_mutable("pair native macro towers")
        if not self._post_sync_recording:
            raise ValueError("composite_pair requires after_synchronization")
        left = require_owned(self, _resolve_handle(left), "composite_pair", vtype="state")
        right = require_owned(self, _resolve_handle(right), "composite_pair", vtype="state")
        if left.op not in {"state", "attempt_start"} or right.op not in {"state", "attempt_start"}:
            raise ValueError("composite_pair requires immutable current/macro-entry State towers")
        if left.state_ref != right.state_ref or left.space != right.space or left.block != right.block:
            raise ValueError("composite_pair requires the same qualified State space")
        coefficients = tuple(float(value) for value in weights)
        if len(coefficients) != len(left.space.components) or not all(map(math.isfinite, coefficients)):
            raise ValueError("composite_pair requires one finite weight per State component")
        units = {left.space.units[i] for i, value in enumerate(coefficients) if value != 0.}
        if len(units) > 1:
            raise ValueError("composite_pair nonzero dimensionless weights require compatible component units")
        return self._new("scalar", "composite_pair", (left, right),
                         {"schema_version": 1, "weights": coefficients}, None, left.block)

    def composite_integral(self, name: str, expression: Any, *, over: Any) -> Any:
        """Integrate an authored scalar expression over finest active cell volumes.

        Field leaves are immutable current/macro-entry towers of ``over``;
        native Scalar leaves permit arbitrary scalar algebra and root iterations.
        A scratch field from one level cannot stand for a hierarchy-wide field.
        """
        from pops.time.expressions import encode_expressions
        self._guard_mutable("integrate an authored expression over native macro towers")
        if not self._post_sync_recording:
            raise ValueError("composite_integral requires after_synchronization")
        if type(name) is not str or not name:
            raise TypeError("composite_integral requires a nonempty name")
        over = require_owned(self, _resolve_handle(over), "composite_integral", vtype="state")
        if over.op != "state" or over.block is None:
            raise ValueError("composite_integral requires a declared current State as support")
        encoded, nodes, inputs = encode_expressions((expression,), self)
        if any(node[0] == "coefficient" for node in nodes):
            raise TypeError("composite_integral requires explicit Scalar captures, not ambient coefficients")
        for value in inputs:
            if value.vtype == "scalar":
                if value.block is not None and value.block != over.block:
                    raise ValueError("composite_integral Scalar belongs to another block")
                continue
            if value.vtype != "state" or value.op not in {"state", "attempt_start"}:
                raise ValueError("composite_integral requires immutable current/macro-entry State towers")
            if value.state_ref != over.state_ref or value.space != over.space or value.block != over.block:
                raise ValueError("composite_integral requires the same qualified State space")
        if all(value is not over for value in inputs):
            inputs = (*inputs, over)
        return self._new("scalar", "composite_integral", inputs,
                         {"schema_version": 1, "expressions": encoded, "expression_nodes": nodes,
                          "support_input": next(i for i, value in enumerate(inputs) if value is over)},
                         name, over.block, inherit_state_ref=False)

    def accept_increment(self, *, scale: Any, duration: Any, histories: Any = None) -> Any:
        """Atomically accept ``U0 + scale*(Ustar-U0)`` and its face measures.

        One direct post-reflux parent invocation owns every participating State.
        ``duration == scale*requested_dt`` must hold in native binary64 arithmetic.
        Source evaluations retain their original requested-step coordinates.
        The engine applies no physical model or numerical formula for ``scale``.
        """
        self._guard_mutable("accept an affine macro increment")
        if not self._post_sync_recording or len(self._recording) != 1:
            raise ValueError("accept_increment requires the direct after_synchronization region")
        scale, duration = self._canonical_value(scale), self._canonical_value(duration)
        for value in (scale, duration):
            require_owned(self, value, "accept_increment", vtype="scalar")
        if any(value.op == "accept_increment" for value in self._recording[-1]):
            raise ValueError("accept_increment may be declared only once")
        from .affine_history import history_contract
        rows = history_contract(self, histories)
        attrs = {"schema_version": 1, "interval_rule": "affine_macro_publication"}
        if rows: attrs["histories"] = rows
        return self._new("scalar", "accept_increment", (scale, duration), attrs,
                         "accepted_duration", duration.block)

    def interval(self, *, duration=None, role="reached"):
        """Issue begin/duration/end SSA for this uniform native invocation."""
        from pops.intervals import issue_interval
        return issue_interval(self, duration=duration, role=role)

    def history_interval(self, value):
        """Reconstruct the exact native publication window of a retained State."""
        from pops.intervals import issue_history_interval
        return issue_history_interval(self,value)

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
