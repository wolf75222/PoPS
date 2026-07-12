"""pops.time Program authoring mixin -- typed operator-call lowering.

Split out of :mod:`pops.time.program_core` for the 500-line cap (ADC-550): the operator-first
``P.call`` front door plus the internal ``_call`` lowering and its helpers
(``_operator_call_name`` / ``_validate_schedule`` / ``_lower_call`` / ``_lower_coupled_rate`` /
``_check_call_args``). ``_ProgramCore`` mixes :class:`_ProgramCall` in, so ``Program`` exposes the
same methods and the IR built by ``P.call`` is byte-identical.

Like ``program_core``, this stays free of any ``pops.codegen`` / ``_pops`` module-scope edge: the
``pops.model`` import is function-local.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.model.operators import OPERATOR_KINDS
from pops.time.operator_resolution import resolve_operator_handle
from pops.time.schedule import Schedule
from pops.time.references import block_name
from pops.time.program_value_validation import require_owned, require_top_level
from pops.time.values import ProgramValue, _CoupledResult

if TYPE_CHECKING:
    from pops.time._program_contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramCall(_ProgramBase):
    """Typed operator-call lowering (``P.call`` and its internal ``_call`` path)."""

    def call(self, operator: Any, *args: Any, name: Any = None, schedule: Any = None) -> Any:
        """Call a typed operator by HANDLE (the operator-first level, Spec 5 sec.14.2.3).

        ``operator`` MUST be the :class:`pops.model.OperatorHandle` a declarer (``m.rate`` /
        ``m.field_operator`` / ``m.source_term`` / ``m.rate_operator`` / ``m.linear_source``)
        returned -- the one public path. A bare string operator NAME is REFUSED with a clear
        ``TypeError`` naming the handle path: a Program references an operator only by the typed
        handle, never by a free string (Spec 5 "one clean API", ADC-479 criterion 23).

        The handle retains its owner, kind and optional structural signature through exact registry
        resolution; only then does lowering use the registered operator name. Thus
        ``P.call(handle, ...)`` builds the byte-identical IR as the
        INTERNAL ``P._call(registered_name, ...)`` path (used by lowerers after validation and the
        operator-first lowering). Resolves the handle against the bound operator registry (see
        :meth:`bind_operators`), type-checks the arguments against the operator's ``Signature``, then
        lowers to the equivalent primitive op so the result is IDENTICAL to the matching PDE shortcut:
        a ``field_operator`` to ``solve_fields``, a ``local_source`` to ``source``, a
        ``grid_operator`` / ``local_rate`` to ``rhs``, a ``local_linear_operator`` to
        ``linear_source``, a ``projection`` to ``project``. A Program composes operators by signature,
        never by a hardcoded PDE category.
        """
        from pops.model import OperatorHandle
        if not isinstance(operator, OperatorHandle):
            if isinstance(operator, str):
                raise TypeError(
                    "P.call requires a typed operator handle, not the string %r; build it with "
                    "m.rate(...) / m.field_operator(...) / m.source_term(...) (any m.*_operator "
                    "declarer returns an pops.model.OperatorHandle)" % (operator,))
            raise TypeError(
                "P.call: operator must be an pops.model.OperatorHandle (from m.rate / "
                "m.field_operator / m.source_term / m.rate_operator / m.linear_source), got %r"
                % (operator,))
        return self._call(operator, *args, name=name, schedule=schedule)

    def _call(self, operator: Any, *args: Any, name: Any = None, schedule: Any = None) -> Any:
        """Internal operator-first call: resolve, type-check and lower an operator (str name OR
        handle). NOT a public surface -- it is the byte-identical lowering the public typed
        :meth:`call` delegates to, and the path the ``pops.lib.time`` macros, the board/operator
        handles and the re-entrant typed lowering use directly (a string token survives here only as
        an internal selector, undocumented in the public API)."""
        from pops.model import OperatorHandle
        operator_handle = None
        if isinstance(operator, OperatorHandle):
            op = resolve_operator_handle(self, operator, where="P.call", values=args)
            operator_name = op.name
            operator_handle = operator
        else:
            operator_name = self._operator_call_name(operator)
            from pops.time.operator_resolution import resolve_registered_operator
            op = resolve_registered_operator(
                self, operator_name, where="P._call", values=args)
        self._check_call_args(op, args)
        if schedule is not None:
            self._validate_schedule(op, schedule, args)
        result = self._lower_call(op, operator_name, args, name)
        # A coupled_rate has no single output ProgramValue (it returns a _CoupledResult): its per-block
        # spaces are tagged inside _lower_coupled_rate, and a schedule on the whole bundle is not
        # meaningful yet -- reject it with a clear message rather than leaking an AttributeError.
        if isinstance(result, _CoupledResult):
            if schedule is not None:
                raise ValueError(
                    "schedule= is not supported on a coupled_rate operator (%r) yet; schedule its "
                    "per-block consumers instead (ADC-457/458)" % (operator_name,))
            if operator_handle is None:
                return result
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
        if operator_handle is not None:
            attrs["operator_handle"] = operator_handle
        field_context = result.field_context
        if (operator_handle is not None and result.op == "solve_fields"
                and attrs.get("field") is not None):
            # _lower_field_operator authenticated a registry-issued field selector. Retain the exact
            # public handle (including an alias identity) in both node attrs and field provenance.
            attrs["field"] = operator_handle
            from pops.time.field_context import FieldContext
            field_context = FieldContext(
                operator_handle, field_context.stage_sources, field_context.outputs)
        if schedule is not None:
            attrs["schedule"] = schedule
        return self._replace_value(
            result, attrs=attrs, space=op.signature.output, field_context=field_context)

    def _operator_call_name(self, operator: Any) -> Any:
        """Normalize an internal :meth:`_call` operator selector to its registry name.

        Accepts only a plain ``str`` internal selector token. Handles are resolved by
        :func:`resolve_operator_handle` before this private seam is reached, so no typed identity can
        be reduced to a name here. The public string reject lives in :meth:`call`."""
        if isinstance(operator, str) and operator:
            return operator
        raise TypeError(
            "_call: internal operator selector must be a non-empty string, got %r"
            % (operator,))

    def _validate_schedule(self, op: Any, schedule: Any, values: Any = ()) -> Any:
        """A schedule= on P.call must be a Schedule; a caching policy (hold / accumulate_dt)
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
        if schedule.kind == "when":
            cond = schedule.params.get("cond")
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
                % (op.name, schedule.policy, op.name))

    def _lower_call(self, op: Any, operator_name: Any, args: Any, name: Any) -> Any:
        # A typed call lowers THROUGH the PRIVATE RHS builder (self._rhs_legacy(flux=...) /
        # self.source / ...): the public P.rhs reject never sees this internal lowering (the user
        # already used the typed P.call front door), so there is one public path and no re-entrancy
        # flag to keep. ADC-642: one decode -- a keyed dispatch over the shared OPERATOR_KINDS
        # vocabulary; each handler holds its arm body verbatim (grid_operator/local_rate share one).
        kind = op.kind
        handler = _LOWER_CALL_HANDLERS.get(kind)
        if handler is None:
            raise NotImplementedError(
                "P.call: operator kind %r is not yet lowerable (operator %r)" % (kind, operator_name))
        return handler(self, op, operator_name, args, name)

    def _lower_field_operator(self, op: Any, operator_name: Any, args: Any, name: Any) -> Any:
        # A multi-input field operator (e.g. fields_from_species over N species) is the COUPLED
        # multi-block field solve: every input species contributes to the one shared elliptic RHS
        # (Sum_s elliptic_rhs_s, the default phi). Route it to solve_fields_from_blocks so no
        # species is dropped -- a single-input field operator stays the historical single-block
        # solve_fields (named-field routing via the operator name as before).
        state_args = [a for a in args if getattr(a, "vtype", None) == "state"]
        if len(state_args) > 1:
            result = self.solve_fields_from_blocks(state_args, name=name)
            return self._replace_value(result, space=op.signature.output)
        field = None
        if operator_name != "fields_from_state":
            registry = self._operator_registries[args[0].block.model_owner_path]
            field = next(
                handle for handle in registry.declaration_index().records()
                if handle.local_id == operator_name)
        result = self._solve_fields(name=name, state=args[0], field=field)
        return self._replace_value(result, space=op.signature.output)

    def _lower_local_source(self, op: Any, operator_name: Any, args: Any, name: Any) -> Any:
        fields = args[1] if len(args) > 1 else None
        source_name = op.lowering.get("source", operator_name)
        if source_name == "default":
            # The default source lives in m._source, not as a named source_term; reach it
            # through the source-only RHS path (byte-identical to flux=False,
            # sources=["default"]), since ctx.source(name) only resolves named source_terms.
            return self._rhs_legacy(name=name, state=args[0], fields=fields, flux=False,
                                    sources=["default"])
        return self._source(source_name, state=args[0], fields=fields)

    def _lower_rate(self, op: Any, operator_name: Any, args: Any, name: Any) -> Any:
        # grid_operator (flux divergence only) and local_rate (flux + sources per op.lowering).
        fields = args[1] if len(args) > 1 else None
        if op.kind == "grid_operator":
            # Flux divergence only (no source): the default flux or a named flux_term.
            fluxes = None if operator_name == "flux_default" else [operator_name]
            return self._rhs_legacy(name=name, state=args[0], fields=fields, flux=True,
                                    sources=[], fluxes=fluxes)
        low = op.lowering
        return self._rhs_legacy(name=name, state=args[0], fields=fields,
                                flux=low.get("flux", True), sources=low.get("sources"),
                                fluxes=low.get("fluxes"))

    def _lower_local_linear_operator(self, op: Any, operator_name: Any, args: Any,
                                     name: Any) -> Any:
        result = self._linear_source(operator_name)
        contexts = [arg.field_context for arg in args
                    if getattr(arg, "vtype", None) == "fields" and arg.field_context is not None]
        if not contexts:
            return result
        from pops.time.field_context import merge_field_provenance
        return self._replace_value(
            result, field_context=merge_field_provenance(*contexts))

    def _lower_projection(self, op: Any, operator_name: Any, args: Any, name: Any) -> Any:
        return self.project(name=name, state=args[0])

    def _lower_coupled_rate(self, op: Any, operator_name: Any, args: Any, name: Any) -> Any:
        """Lower a coupled_rate operator to a coupled node plus one per-block rate projection.

        A coupled operator (collisions, ionization, ...) of arbitrary arity returns a typed
        ``RateBundle``; ``P.call`` returns a :class:`_CoupledResult` whose ``["electrons"]`` is the
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
        """Type-check ``P.call`` arguments against an operator's Signature: arity plus the vtype of
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
                    "operator %r requires a typed %s %r; bind its owner registry before "
                    "T.state(block, U), or declare typed state metadata"
                    % (op.name, t.kind, t.name))
            if arg_space != t:
                raise ValueError(
                    "operator %r expects %s %r with components %r but got a value over %r "
                    "with components %r"
                    % (op.name, t.kind, t.name, getattr(t, "components", ()),
                       getattr(arg_space, "name", None), getattr(arg_space, "components", ())))


# ADC-642: the one operator-kind -> lowering dispatch, keyed on the shared OPERATOR_KINDS
# vocabulary. grid_operator and local_rate share _lower_rate (it inspects op.kind internally, as
# before). The intentionally-unlowered kinds (diagnostic / matrix_free_operator / the residuals) have
# no row and fall through _lower_call's NotImplementedError catch-all. The assert makes an unwired
# kind fail loudly at import, not silently at runtime.
_LOWER_CALL_HANDLERS = {
    "field_operator": _ProgramCall._lower_field_operator,
    "local_source": _ProgramCall._lower_local_source,
    "grid_operator": _ProgramCall._lower_rate,
    "local_rate": _ProgramCall._lower_rate,
    "local_linear_operator": _ProgramCall._lower_local_linear_operator,
    "projection": _ProgramCall._lower_projection,
    "coupled_rate": _ProgramCall._lower_coupled_rate,
}
assert set(_LOWER_CALL_HANDLERS) <= set(OPERATOR_KINDS)

__all__ = ["_ProgramCall"]
