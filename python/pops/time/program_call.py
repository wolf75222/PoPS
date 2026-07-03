"""pops.time Program authoring mixin -- typed operator-call lowering.

Split out of :mod:`pops.time.program_core` for the 500-line cap (ADC-550): the operator-first
``P.call`` front door plus the internal ``_call`` lowering and its helpers
(``_operator_call_name`` / ``_validate_schedule`` / ``_lower_call`` / ``_lower_coupled_rate`` /
``_check_call_args``). ``_ProgramCore`` mixes :class:`_ProgramCall` in, so ``Program`` exposes the
same methods and the IR built by ``P.call`` is byte-identical.

Like ``program_core``, this stays free of any ``pops.codegen`` / ``_pops`` module-scope edge: the
``pops.model`` import is function-local.
"""
from pops.time.schedule import Schedule
from pops.time.values import Value, _CoupledResult


class _ProgramCall:
    """Typed operator-call lowering (``P.call`` and its internal ``_call`` path)."""

    def call(self, operator, *args, name=None, schedule=None):
        """Call a typed operator by HANDLE (the operator-first level, Spec 5 sec.14.2.3).

        ``operator`` MUST be the :class:`pops.model.OperatorHandle` a declarer (``m.rate`` /
        ``m.field_operator`` / ``m.source_term`` / ``m.rate_operator`` / ``m.linear_source``)
        returned -- the one public path. A bare string operator NAME is REFUSED with a clear
        ``TypeError`` naming the handle path: a Program references an operator only by the typed
        handle, never by a free string (Spec 5 "one clean API", ADC-479 criterion 23).

        The handle is a transparent alias for its ``.name``: it follows the EXACT same registry
        resolution + lowering, so ``P.call(handle, ...)`` builds the byte-identical IR as the
        INTERNAL ``P._call(handle.name, ...)`` path (used by the ``pops.lib.time`` macros and the
        operator-first lowering). Resolves the name against the bound operator registry (see
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

    def _call(self, operator, *args, name=None, schedule=None):
        """Internal operator-first call: resolve, type-check and lower an operator (str name OR
        handle). NOT a public surface -- it is the byte-identical lowering the public typed
        :meth:`call` delegates to, and the path the ``pops.lib.time`` macros, the board/operator
        handles and the re-entrant typed lowering use directly (a string token survives here only as
        an internal selector, undocumented in the public API)."""
        operator_name = self._operator_call_name(operator)
        if self._registry is None:
            raise ValueError("P.call(%r): no operators bound; call P.bind_operators(model) first"
                             % (operator_name,))
        op = self._registry.get(operator_name)  # clear KeyError on an unknown operator
        self._check_call_args(op, args)
        if schedule is not None:
            self._validate_schedule(op, schedule)
        result = self._lower_call(op, operator_name, args, name)
        # A coupled_rate has no single output Value (it returns a _CoupledResult): its per-block
        # spaces are tagged inside _lower_coupled_rate, and a schedule on the whole bundle is not
        # meaningful yet -- reject it with a clear message rather than leaking an AttributeError.
        if isinstance(result, _CoupledResult):
            if schedule is not None:
                raise ValueError(
                    "schedule= is not supported on a coupled_rate operator (%r) yet; schedule its "
                    "per-block consumers instead (ADC-457/458)" % (operator_name,))
            return result
        # Tag the result with the operator's declared output type (a Rate / FieldSpace /
        # LocalLinearOperator / StateSpace) so downstream ops can type-check the composition
        # (a Rate(U) cannot be combined with a State(V); an L: U -> U cannot drive a State(V)).
        result.space = op.signature.output
        if schedule is not None:
            result.attrs["schedule"] = schedule
        return result

    def _operator_call_name(self, operator):
        """Normalize an internal :meth:`_call` operator selector to its registry name.

        Accepts EITHER a plain ``str`` (an internal selector token, returned unchanged) OR an
        :class:`pops.model.OperatorHandle` (its ``.name`` is returned). A handle resolves through the
        identical registry lookup + lowering as its name, so the IR is byte-identical. Any other type
        is a clear ``TypeError``. The public string REJECT lives in :meth:`call`; this internal
        normalizer accepts the string the lowering and the lib.time macros pass."""
        from pops.model import OperatorHandle
        if isinstance(operator, OperatorHandle):
            return operator.name
        if isinstance(operator, str):
            return operator
        raise TypeError(
            "_call: operator must be a str name or an pops.model.OperatorHandle, got %r"
            % (operator,))

    def _validate_schedule(self, op, schedule):
        """A schedule= on P.call must be a Schedule; a caching policy (hold / accumulate_dt)
        requires the operator to be cacheable (Spec 3 criterion 27)."""
        if not isinstance(schedule, Schedule):
            raise TypeError(
                "schedule= expects an pops.time Schedule (always()/every(n)/...), got %r"
                % (schedule,))
        if schedule.needs_cache() and not op.capabilities.get("cacheable"):
            raise ValueError(
                "operator %r is not cacheable; cannot use schedule %s -- declare it with "
                "m.operator_capabilities(%r, cacheable=True)"
                % (op.name, schedule.policy, op.name))

    def _lower_call(self, op, operator_name, args, name):
        # A typed call lowers THROUGH the PRIVATE RHS builder (self._rhs_legacy(flux=...) /
        # self.source / ...): the public P.rhs reject never sees this internal lowering (the user
        # already used the typed P.call front door), so there is one public path and no re-entrancy
        # flag to keep.
        kind = op.kind
        if kind == "field_operator":
            # A multi-input field operator (e.g. fields_from_species over N species) is the COUPLED
            # multi-block field solve: every input species contributes to the one shared elliptic RHS
            # (Sum_s elliptic_rhs_s, the default phi). Route it to solve_fields_from_blocks so no
            # species is dropped -- a single-input field operator stays the historical single-block
            # solve_fields (named-field routing via the operator name as before).
            state_args = [a for a in args if getattr(a, "vtype", None) == "state"]
            if len(state_args) > 1:
                return self.solve_fields_from_blocks(state_args, name=name)
            field = None if operator_name == "fields_from_state" else operator_name
            return self.solve_fields(name=name, state=args[0], field=field)
        if kind == "local_source":
            fields = args[1] if len(args) > 1 else None
            if operator_name == "source_default":
                # The default source lives in m._source, not as a named source_term; reach it
                # through the source-only RHS path (byte-identical to flux=False,
                # sources=["default"]), since ctx.source(name) only resolves named source_terms.
                return self._rhs_legacy(name=name, state=args[0], fields=fields, flux=False,
                                        sources=["default"])
            return self.source(operator_name, state=args[0], fields=fields)
        if kind in ("grid_operator", "local_rate"):
            fields = args[1] if len(args) > 1 else None
            if kind == "grid_operator":
                # Flux divergence only (no source): the default flux or a named flux_term.
                fluxes = None if operator_name == "flux_default" else [operator_name]
                return self._rhs_legacy(name=name, state=args[0], fields=fields, flux=True,
                                        sources=[], fluxes=fluxes)
            low = op.lowering
            return self._rhs_legacy(name=name, state=args[0], fields=fields,
                                    flux=low.get("flux", True), sources=low.get("sources"),
                                    fluxes=low.get("fluxes"))
        if kind == "local_linear_operator":
            return self._linear_source(operator_name)
        if kind == "projection":
            return self.project(name=name, state=args[0])
        if kind == "coupled_rate":
            return self._lower_coupled_rate(op, operator_name, args, name)
        raise NotImplementedError(
            "P.call: operator kind %r is not yet lowerable (operator %r)" % (kind, operator_name))

    def _lower_coupled_rate(self, op, operator_name, args, name):
        """Lower a coupled_rate operator to a coupled node plus one per-block rate projection.

        A coupled operator (collisions, ionization, ...) of arbitrary arity returns a typed
        ``RateBundle``; ``P.call`` returns a :class:`_CoupledResult` whose ``["electrons"]`` is the
        per-block rate (an RHS Value over that block) so it composes like any other RHS. The
        coupled-rate KERNEL codegen has landed (ADC-457): ``_emit_coupled_rate_kernel`` lowers the
        ``coupled_rate`` node to one multi-state ``for_each_cell`` and each ``coupled_rate_out``
        projects its block's rate scratch.
        """
        bundle = op.signature.output                 # a model.RateBundle: block -> RateSpace
        blocks = bundle.keys()
        base = name or operator_name
        coupled = self._new("rhs", "coupled_rate", tuple(args),
                            {"operator": operator_name, "blocks": list(blocks)},
                            base, args[0].block)
        outs = {}
        for blk in blocks:
            out = self._new("rhs", "coupled_rate_out", (coupled,),
                            {"operator": operator_name, "out_block": blk},
                            "%s_%s" % (base, blk), blk)
            out.space = bundle[blk]                   # the per-block RateSpace, for type checks
            outs[blk] = out
        return _CoupledResult(outs)

    def _check_call_args(self, op, args):
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
            if not (isinstance(a, Value) and a.vtype == want):
                got = a.vtype if isinstance(a, Value) else type(a).__name__
                raise ValueError(
                    "operator %r argument for %s %r expects a %s value, got %s"
                    % (op.name, t.kind, t.name, want, got))
            # If the argument carries an operator-first space tag, its name must match the
            # operator's declared input space (a value over 'V' cannot feed an input typed 'U').
            arg_space = getattr(a, "space", None)
            arg_name = getattr(arg_space, "name", None)
            if arg_name is not None and arg_name != t.name:
                raise ValueError(
                    "operator %r expects %s %r but got a value over %r"
                    % (op.name, t.kind, t.name, arg_name))


__all__ = ["_ProgramCall"]
