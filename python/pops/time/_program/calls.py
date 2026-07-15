"""pops.time Program authoring mixin -- typed operator-call lowering.

Split out of :mod:`pops.time._program.operations` for the 500-line cap (ADC-550): the private
``_call`` lowering used by callable operator handles plus its helpers
(``_validate_schedule`` / ``_lower_call`` / ``_lower_coupled_rate`` /
``_check_call_args``). ``_ProgramCore`` mixes :class:`_ProgramCall` in while the public surface
stays operator-first (``rate(state)``).

Like ``program_core``, this stays free of any ``pops.codegen`` / ``_pops`` module-scope edge: the
``pops.model`` import is function-local.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.model.operators import OPERATOR_KINDS
from pops.time.operator_resolution import resolve_operator_handle
from pops.time._schedule.api import Schedule
from pops.time._schedule.ir import ScheduleDueIR, ScheduleDueKind
from pops.time.references import block_name
from pops.time._program.value_validation import require_owned, require_top_level
from pops.time.value_collections import _CoupledResult
from pops.time.values import ProgramValue

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramCall(_ProgramBase):
    """Private typed operator-call lowering used by callable operator handles."""

    def _call(self, operator: Any, *args: Any, name: Any = None, schedule: Any = None) -> Any:
        """Resolve, type-check and lower one exact operator handle."""
        from pops.model import OperatorHandle
        if not isinstance(operator, OperatorHandle):
            raise TypeError(
                "operator call requires the exact OperatorHandle returned by a model declarer; "
                "got %r" % (operator,))
        op = resolve_operator_handle(self, operator, where="operator call", values=args)
        operator_name = op.name
        operator_handle = operator
        if op.kind == "field_operator":
            raise TypeError(
                "model field-provider handles describe physical RHS contributions but are not "
                "Program solve authorities; register the physical FieldOperator with "
                "Case.field(...) and call the returned case-owned FieldHandle")
        self._check_call_args(op, args)
        self._validate_scheduled_reads(args, consumer="operator %r" % op.name)
        if schedule is not None:
            self._validate_schedule(op, schedule, args)
        result = self._lower_call(op, operator_handle, operator_name, args, name)
        # A coupled_rate has no single output ProgramValue (it returns a _CoupledResult): its per-block
        # spaces are tagged inside _lower_coupled_rate, and a schedule on the whole bundle is not
        # meaningful yet -- reject it with a clear message rather than leaking an AttributeError.
        if isinstance(result, _CoupledResult):
            if schedule is not None:
                raise ValueError(
                    "schedule= is not supported on a coupled_rate operator (%r) yet; schedule its "
                    "per-block consumers instead (ADC-457/458)" % (operator_name,))
            tagged = {}
            coupled = None
            for block, value in result.items():
                if value.inputs:
                    coupled = value.inputs[0]
                attrs = dict(value.attrs)
                attrs["operator_handle"] = operator_handle
                tagged[block] = self._replace_value(value, attrs=attrs)
            if coupled is not None:
                attrs = dict(coupled.attrs)
                attrs["operator_handle"] = operator_handle
                self._replace_value(coupled, attrs=attrs)
            return _CoupledResult(tagged)
        # Tag the result with the operator's declared output type (a Rate / FieldSpace /
        # LocalLinearOperator / StateSpace) so downstream ops can type-check the composition
        # (a Rate(U) cannot be combined with a State(V); an L: U -> U cannot drive a State(V)).
        attrs = dict(result.attrs)
        attrs["operator_handle"] = operator_handle
        if schedule is not None:
            attrs["schedule"] = schedule
            schedule.validate_site(clock=result.clock, point=result.point,
                                   where="schedule on operator %r" % op.name)
        result = self._replace_value(
            result, attrs=attrs, space=op.signature.output,
            field_context=result.field_context)
        return result

    def _validate_schedule(self, op: Any, schedule: Any, values: Any = ()) -> Any:
        """A schedule on an operator handle must be a Schedule; a caching policy (hold / accumulate_dt)
        requires the operator to be cacheable (Spec 3 criterion 27)."""
        if not isinstance(schedule, Schedule):
            raise TypeError(
                "schedule= expects an pops.time Schedule (always()/every(n)/...), got %r"
                % (schedule,))
        value_clocks = {
            value.clock for value in values if isinstance(value, ProgramValue)
        }
        expected_clock = next(iter(value_clocks)) if value_clocks else self.clock
        if len(value_clocks) > 1:
            raise ValueError(
                "schedule cannot govern inputs from different clocks; synchronize them first")
        if schedule.clock != expected_clock:
            raise ValueError(
                "schedule clock %r does not match operator evaluation clock %r"
                % (schedule.clock.name, expected_clock.name))
        due = schedule.trigger.native_schedule_due(
            where="schedule on operator %r" % op.name)
        if type(due) is not ScheduleDueIR:
            raise TypeError(
                "Trigger.native_schedule_due() must return an exact ScheduleDueIR")
        if due.kind is ScheduleDueKind.PROGRAM_PREDICATE:
            cond = due.predicate
            if isinstance(cond, ProgramValue):
                require_top_level(self, cond, "schedule when(cond)")
                if cond.vtype != "bool" or not any(v.id == cond.id for v in self._values):
                    raise ValueError(
                        "schedule when(cond): cond must be a previously authored Bool value")
                if cond.clock != schedule.clock:
                    raise ValueError(
                        "schedule when(cond): predicate and schedule must use the same clock")
        if schedule.needs_cache() and not op.capabilities.get("cacheable"):
            raise ValueError(
                "operator %r is not cacheable; cannot use schedule %s -- declare it with "
                "m.operator_capabilities(%r, cacheable=True)"
                % (op.name, type(schedule.off).__name__, op.name))

    @staticmethod
    def _validate_scheduled_reads(values: Any, *, consumer: str) -> None:
        """A non-trivial scheduled value is readable only with an explicit typed OffPolicy."""
        for value in values:
            if not isinstance(value, ProgramValue):
                continue
            source_schedule = value.attrs.get("schedule")
            if (source_schedule is not None and not source_schedule.is_always()
                    and source_schedule.off is None):
                raise ValueError(
                    "%s reads scheduled value %r without an explicit OffPolicy; construct it as "
                    "Schedule(trigger, off=Hold()/Skip()/Zero()/AccumulateDt()/Error())"
                    % (consumer, value.name))

    def _lower_call(self, op: Any, operator_handle: Any, operator_name: Any,
                    args: Any, name: Any) -> Any:
        # A typed call lowers through the private native RHS projection (self._rhs_primitive(...) /
        # self.source / ...): the public P.rhs reject never sees this internal lowering (the user
        # already invoked an exact operator handle), so there is one path and no re-entrancy
        # flag to keep. ADC-642: one decode -- a keyed dispatch over the shared OPERATOR_KINDS
        # vocabulary; each handler holds its arm body verbatim (grid_operator/local_rate share one).
        kind = op.kind
        handler = _LOWER_CALL_HANDLERS.get(kind)
        if handler is None:
            raise NotImplementedError(
                "operator kind %r is not yet lowerable (operator %r)" % (kind, operator_name))
        return handler(self, op, operator_handle, operator_name, args, name)

    def _lower_local_source(self, op: Any, _operator_handle: Any, operator_name: Any,
                            args: Any, name: Any) -> Any:
        fields = args[1] if len(args) > 1 else None
        source_name = op.lowering.get("source", operator_name)
        if source_name == "default":
            # The default source lives in m._source, not as a named source_term; reach it
            # through the source-only RHS path (byte-identical to flux=False,
            # sources=["default"]), since ctx.source(name) only resolves named source_terms.
            return self._rhs_primitive(name=name, state=args[0], fields=fields, flux=False,
                                       sources=["default"])
        return self._source(source_name, state=args[0], fields=fields)

    def _lower_rate(self, op: Any, _operator_handle: Any, operator_name: Any,
                    args: Any, name: Any) -> Any:
        # grid_operator (flux divergence only) and local_rate (flux + sources per op.lowering).
        fields = args[1] if len(args) > 1 else None
        if op.kind == "grid_operator":
            # Flux divergence only (no source): the default flux or a named flux_term.
            fluxes = None if operator_name == "flux_default" else [operator_name]
            return self._rhs_primitive(name=name, state=args[0], fields=fields, flux=True,
                                       sources=[], fluxes=fluxes)
        low = op.lowering
        # A multi-state Module retains the exact physical grid-operator identity in ``fluxes`` so
        # model lowering can install the right formula.  When that sole operator is explicitly the
        # native default, its Program evaluation must use rhs_into (the configured FV/Riemann route),
        # not the named centered-divergence kernel.
        program_fluxes = None if low.get("default_flux") is not None else low.get("fluxes")
        return self._rhs_primitive(name=name, state=args[0], fields=fields,
                                   flux=low.get("flux", True), sources=low.get("sources"),
                                   fluxes=program_fluxes)

    def _lower_local_linear_operator(self, op: Any, _operator_handle: Any,
                                     operator_name: Any, args: Any, name: Any) -> Any:
        result = self._linear_source(operator_name)
        contexts = [arg.field_context for arg in args
                    if getattr(arg, "vtype", None) == "fields" and arg.field_context is not None]
        if not contexts:
            return result
        from pops.time.field_context import merge_field_provenance
        return self._replace_value(
            result, field_context=merge_field_provenance(*contexts))

    def _lower_projection(self, op: Any, _operator_handle: Any, operator_name: Any,
                          args: Any, name: Any) -> Any:
        return self.project(name=name, state=args[0])

    def _lower_coupled_rate(self, op: Any, _operator_handle: Any, operator_name: Any,
                            args: Any, name: Any) -> Any:
        """Lower a coupled_rate operator to a coupled node plus one per-block rate projection.

        A coupled operator (collisions, ionization, ...) of arbitrary arity returns a typed
        ``RateBundle``; the callable handle returns a :class:`_CoupledResult` whose ``["electrons"]`` is the
        per-block rate (an RHS ProgramValue over that block) so it composes like any other RHS. The
        coupled-rate KERNEL codegen has landed (ADC-457): ``_emit_coupled_rate_kernel`` lowers the
        ``coupled_rate`` node to one multi-state ``for_each_cell`` and each ``coupled_rate_out``
        projects its block's rate scratch.
        """
        points = {argument.point for argument in args}
        if len(points) != 1:
            raise ValueError(
                "coupled operator %r inputs belong to different evaluation points; "
                "construct one explicit partitioned StagePoint or synchronize them first"
                % operator_name)
        bundle = op.signature.output                 # a model.RateBundle: block -> RateSpace
        input_blocks = {
            block_name(argument.block): argument.block
            for argument in args if getattr(argument, "block", None) is not None
        }
        missing = [name for name in bundle.keys() if name not in input_blocks]
        if missing:
            raise ValueError(
                "coupled operator %r outputs blocks %s but no matching typed input BlockHandle "
                "was supplied" % (operator_name, missing))
        blocks = [input_blocks[name] for name in bundle.keys()]
        base = name or operator_name
        coupled = self._new("rhs", "coupled_rate", tuple(args),
                            {"operator": operator_name, "blocks": list(blocks)},
                            base, args[0].block)
        outs = {}
        for output_name, blk in zip(bundle.keys(), blocks, strict=True):
            out = self._new("rhs", "coupled_rate_out", (coupled,),
                            {"operator": operator_name, "out_block": blk},
                            "%s_%s" % (base, output_name), blk,
                            space=bundle[output_name])
            outs[blk] = out
        return _CoupledResult(outs)

    def _check_call_args(self, op: Any, args: Any) -> Any:
        """Type-check callable-handle arguments against an operator's Signature: arity plus the vtype of
        each space-typed input (a StateSpace input wants a 'state' value, a FieldSpace input a
        'fields' value). Operator-valued inputs are not passed positionally. Clear error on mismatch.
        """
        expected = [t for t in op.signature.inputs
                    if getattr(t, "kind", None) in ("state", "field")]
        if len(args) != len(expected):
            raise ValueError(
                "operator %r expects %d argument(s) %s, got %d"
                % (op.name, len(expected), tuple(t.name for t in expected), len(args)))
        want_of = {"state": "state", "field": "fields"}
        for t, a in zip(expected, args, strict=True):
            want = want_of[t.kind]
            if not (isinstance(a, ProgramValue) and a.vtype == want):
                got = a.vtype if isinstance(a, ProgramValue) else type(a).__name__
                raise ValueError(
                    "operator %r argument for %s %r expects a %s value, got %s"
                    % (op.name, t.kind, t.name, want, got))
            require_owned(self, a, "operator %r argument" % op.name)
            # If the argument carries an operator-first space tag, its COMPLETE structural space
            # must match the declaration. Component order affects generated kernels, so name-only
            # equality would permit two different C++ programs to share one cache identity.
            arg_space = getattr(self._canonical_value(a), "space", None)
            if arg_space is None:
                raise ValueError(
                    "operator %r requires a typed %s %r; declare it through the authenticated "
                    "T.state(block[U]) route"
                    % (op.name, t.kind, t.name))
            if arg_space != t:
                raise ValueError(
                    "operator %r expects %s %r with components %r but got a value over %r "
                    "with components %r"
                    % (op.name, t.kind, t.name, getattr(t, "components", ()),
                       getattr(arg_space, "name", None), getattr(arg_space, "components", ())))


# ADC-642: the one operator-kind -> lowering dispatch, keyed on the shared OPERATOR_KINDS
# vocabulary. grid_operator and local_rate share _lower_rate (it inspects op.kind internally, as
# before). Field providers are physical contributions and are rejected above in favour of the
# Case-owned FieldHandle; the other intentionally-unlowered kinds (diagnostic /
# matrix_free_operator / residuals) have no row and fall through _lower_call's NotImplementedError
# catch-all. The assert makes an unwired kind fail loudly at import, not silently at runtime.
_LOWER_CALL_HANDLERS = {
    "local_source": _ProgramCall._lower_local_source,
    "grid_operator": _ProgramCall._lower_rate,
    "local_rate": _ProgramCall._lower_rate,
    "local_linear_operator": _ProgramCall._lower_local_linear_operator,
    "projection": _ProgramCall._lower_projection,
    "coupled_rate": _ProgramCall._lower_coupled_rate,
}
assert set(_LOWER_CALL_HANDLERS) <= set(OPERATOR_KINDS)

__all__ = ["_ProgramCall"]
