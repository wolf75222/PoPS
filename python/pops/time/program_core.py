"""pops.time Program authoring mixin -- core builder ops.

State / field / RHS / source / apply construction (the operator-first builder core).
"""
from pops.time.program_base import _ProgramConstants
from pops.time.schedule import Schedule
from pops.time.values import (
    Value, _Coeff, _CoupledResult, _Operator, _resolve_handle, _state_base_name, _to_affine,
)


class _ProgramCore(_ProgramConstants):
    """State / field / RHS / source / apply construction (the operator-first builder core)."""

    # --- node construction ---
    def _new(self, vtype, op, inputs, attrs, name, block):
        for v in inputs:
            if isinstance(v, Value) and v.prog is not self:
                raise ValueError("IR value %r belongs to a different Program" % (v,))
        vid = self._next_id
        self._next_id += 1
        v = Value(self, vid, vtype, op, [i for i in inputs if isinstance(i, Value)],
                  attrs, name or ("%s%d" % (op, vid)), block)
        # Inside a control-flow recording scope (cond_fn / body_fn of a while_), ops go into the active
        # sub-block, NOT the flat self._values: a while body must RE-EXECUTE each iteration, so its ops
        # are owned by the while op and re-emitted in the loop, never walked once at the top level.
        if self._recording:
            self._recording[-1].append(v)
        else:
            self._values.append(v)
        return v

    def state(self, name="U", space=None, *, block=None):
        """Create typed temporal-version handles for one block state.

        Public Programs use ``P.state("U", block="plasma")`` and then compose ``U.n``,
        ``U.stage(k)``, ``U.next`` and ``U.prev``.  The old positional block form is not
        accepted as a public API: raw SSA state values are an internal lowering detail used
        by typed handles and ready-made library schemes.
        """
        if block is None:
            raise TypeError(
                "Program.state requires block= and returns typed temporal handles, e.g. "
                "P.state('U', block='plasma'). Internal scheme builders must use "
                "Program._state_value(block).")
        from pops.time.handles import TimeState
        return TimeState(self, block, name=name or "U", space=space)

    def _state_value(self, block, *, space=None, name=None):
        """Internal SSA value for the current state of ``block``.

        This is the only place that emits the ``state`` IR node consumed by C++ codegen.
        It is intentionally not public DSL: users work with temporal handles.
        """
        if not isinstance(block, str) or not block:
            raise ValueError("_state_value: block must be a non-empty string")
        v = self._new("state", "state", (), {}, name or block, block)
        v.space = space
        return v

    def _fields_from_state(self, name=None, state=None, field=None):
        """Internal field-solve builder used by typed operator lowering.

        This emits the ``solve_fields`` IR op consumed by the C++ Program codegen.
        It is intentionally private: user Programs call typed field operators through
        :meth:`call`, or use higher-level board sugar that lowers through this route.
        """
        # A defined temporal-version handle (U.stage(k) / U.next / U.prev) resolves to its Value
        # here so it composes wherever a State does; a plain Value / None / str is unchanged.
        name, state = _resolve_handle(name), _resolve_handle(state)
        if isinstance(name, Value) and state is None:
            name, state = None, name
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("_fields_from_state: a State value is required")
        if field is not None and not (isinstance(field, str) and field):
            raise ValueError("_fields_from_state: field must be a non-empty named elliptic field")
        # The attr is added only for a named field; the default field solve has no extra selector.
        attrs = {"field": field} if field is not None else {}
        return self._new("fields", "solve_fields", (state,), attrs, name, state.block)

    def solve_fields_from_blocks(self, states, name=None):
        """Solve the elliptic fields from the SIMULTANEOUS stage states of MULTIPLE blocks (spec
        \"Multi-blocs\"): a coupled Poisson where each listed block reads its own @p states[k] override
        at once, returning a FieldContext.

        RUNTIME (Spec 3 criterion 24, ADC-457): this is the multi-target coupled solve. It lowers to
        ``ctx.solve_fields_from_blocks(u_stages)``, a per-block pointer vector the native field solver
        (``System::solve_fields_from_blocks`` ->
        ``SystemFieldSolver::assemble_poisson_rhs_from_blocks``) assembles the system Poisson RHS from as
        ``Sum_s elliptic_rhs_s(U_s)`` reading EVERY listed block's stage state AT ONCE (a true
        simultaneous override, not a sequence of single-target solves). A block NOT listed contributes
        its live state. The listed states slot at their block index (the P.state declaration order), so
        the runtime sees each coupled block at its stage state into the one shared phi/aux.

        A per-block internal ``_fields_from_state(state=Ub)`` remains the right lowering when blocks
        advance in sequence (block b at its stage state, every other block at its live state); this op is for the
        SIMULTANEOUS case where multiple coupled blocks must each contribute their stage state at once."""
        if not (isinstance(states, (list, tuple)) and states):
            raise ValueError("solve_fields_from_blocks: a non-empty list of State values is required")
        seen = set()
        for s in states:
            if not (isinstance(s, Value) and s.vtype == "state"):
                raise ValueError("solve_fields_from_blocks: every entry must be a State value")
            if s.block in seen:
                raise ValueError("solve_fields_from_blocks: block '%s' listed twice" % s.block)
            seen.add(s.block)
        # The FieldContext is attached to the first listed block (an arbitrary but stable owner).
        return self._new("fields", "solve_fields_from_blocks", tuple(states), {}, name, states[0].block)

    # --- operator-first calls (Spec 2) -------------------------------------------
    def bind_operators(self, source):
        """Bind a typed operator registry so ``P.call`` can resolve and type-check operators.

        ``source`` is an ``pops.model.OperatorRegistry`` or any object exposing
        ``operator_registry()`` (for example ``pops.model.Module``). Returns ``self`` for
        chaining. The bound registry is build-time TYPE information only -- the codegen still reads
        the model passed to ``compile_problem``; operator-first Programs and the ``pops.lib.time``
        macros bind the module's operators here.
        """
        reg = source.operator_registry() if hasattr(source, "operator_registry") else source
        if not (hasattr(reg, "get") and hasattr(reg, "names")):
            raise TypeError("bind_operators: expected an OperatorRegistry or an object exposing "
                            "operator_registry(); got %r" % (source,))
        self._registry = reg
        return self

    def call(self, operator, *args, name=None, schedule=None):
        """Call a typed operator by HANDLE (the operator-first level, Spec 5 sec.14.2.3).

        ``operator`` MUST be a typed selector: either the :class:`pops.model.OperatorHandle` a
        physics declarer returned, or the :class:`pops.model.Operator` returned by
        ``pops.model.Module.operator(...)``. A bare string operator NAME is REFUSED with a clear
        ``TypeError`` naming the typed-object path: a Program references an operator only by a
        typed object, never by a free string (Spec 5 "one clean API", ADC-479 criterion 23).

        Resolves the handle against the bound operator registry (see :meth:`bind_operators`),
        type-checks the arguments against the operator's ``Signature``, then records an
        ``call`` node in the Program IR. Codegen lowers that typed call to the appropriate
        C++ ``ProgramContext`` operation. The public path no longer rewrites a call back into the
        private ``_rate_from_transport`` / ``_fields_from_state`` builders.
        """
        from pops.model import Operator, OperatorHandle
        if not isinstance(operator, (OperatorHandle, Operator)):
            if isinstance(operator, str):
                raise TypeError(
                    "P.call requires a typed operator handle/object, not the string %r; build it "
                    "with m.rate(...) / m.field_operator(...) / m.source_term(...), or keep the "
                    "pops.model.Operator returned by module.operator(...)" % (operator,))
            raise TypeError(
                "P.call: operator must be an pops.model.OperatorHandle or pops.model.Operator, got %r"
                % (operator,))
        return self._call(operator, *args, name=name, schedule=schedule)

    def _call(self, operator, *args, name=None, schedule=None):
        """Internal operator-first call: resolve, type-check and lower an operator (str name OR
        handle). NOT a public surface -- the string token survives here only as an internal selector
        for package macros and tests."""
        operator_name = self._call_name(operator)
        if self._registry is None:
            raise ValueError("P.call(%r): no operators bound; call P.bind_operators(model) first"
                             % (operator_name,))
        args = tuple(_resolve_handle(arg) for arg in args)
        op = self._registry.get(operator_name)  # clear KeyError on an unknown operator
        self._check_call_args(op, args)
        if schedule is not None:
            self._validate_schedule(op, schedule)
        result = self._lower_call(op, self._registry.id_of(operator_name), operator, operator_name,
                                  args, name, schedule=schedule)
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

    def _call_name(self, operator):
        """Normalize an internal :meth:`_call` operator selector to its registry name.

        Accepts EITHER a plain ``str`` (an internal selector token, returned unchanged), an
        :class:`pops.model.OperatorHandle`, or a :class:`pops.model.Operator` (their ``.name`` is
        returned). A typed selector resolves through the identical registry lookup + lowering as its
        name. Any other type is a clear ``TypeError``. The public string reject lives in
        :meth:`call`; this internal normalizer accepts package-internal registry keys."""
        from pops.model import Operator, OperatorHandle
        if isinstance(operator, (OperatorHandle, Operator)):
            return operator.name
        if isinstance(operator, str):
            return operator
        raise TypeError(
            "_call: operator must be a str name, pops.model.OperatorHandle, or pops.model.Operator, got %r"
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

    def _call_node(self, storage_vtype, output_vtype, operator_id, operator_handle, operator_name,
                   output, args, name, block, schedule=None):
        attrs = {
            "operator": operator_name,
            "operator_id": int(operator_id),
            "operator_handle": getattr(operator_handle, "name", operator_name),
            "output_type": repr(output),
            "output_vtype": output_vtype,
        }
        if schedule is not None:
            attrs["schedule"] = schedule
        return self._new(storage_vtype, "call", args, attrs, name, block)

    def _lower_call(self, op, operator_id, operator_handle, operator_name, args, name,
                    schedule=None):
        # A typed call is a first-class IR node: operator id + arguments + output type.  The Program
        # does not lower by operator kind here; GeneratedModule::Operators owns the numerical route.
        from pops.model import FieldSpace, LocalLinearOperator, RateBundle, RateSpace, StateSpace
        output = op.signature.output
        block = args[0].block if args else None
        if isinstance(output, FieldSpace):
            return self._call_node("fields", "fields", operator_id, operator_handle, operator_name, output,
                                   args, name, block, schedule=schedule)
        if isinstance(output, RateSpace):
            return self._call_node("rhs", "rate", operator_id, operator_handle, operator_name, output,
                                   args, name, block, schedule=schedule)
        if isinstance(output, LocalLinearOperator):
            return self._call_node("operator", "local_linear_operator", operator_id,
                                   operator_handle, operator_name, output, args,
                                   name or operator_name, block, schedule=schedule)
        if isinstance(output, StateSpace):
            return self._call_node("state", "state", operator_id, operator_handle, operator_name, output,
                                   args, name, block, schedule=schedule)
        if isinstance(output, RateBundle):
            return self._lower_coupled_rate(op, operator_name, args, name)
        raise NotImplementedError(
            "P.call: operator %r output type %r is not yet lowerable"
            % (operator_name, output))

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

    def _rate_from_terms(self, name=None, state=None, fields=None, *, terms=None, **legacy):
        """Internal typed-term lowering kept for package macros.

        The public Program API is operator-first: users call ``P.call(rate_handle, U, fields)``.
        This helper exists so internal scheme builders can still convert a typed term list onto the
        internal transport-rate IR without exposing a second user path.
        """
        if legacy or terms is None:
            extra = "".join(", %s=" % k for k in sorted(legacy))
            raise TypeError(
                "_rate_from_terms requires terms=, not the old flux=/sources=/fluxes= form%s"
                % extra)
        from pops.time._rhs_terms import terms_to_flux_sources
        flux, sources = terms_to_flux_sources(terms)
        return self._rate_from_transport(name=name, state=state, fields=fields,
                                         flux=flux, sources=sources)

    def _rate_from_transport(self, name=None, state=None, fields=None,
                             flux=True, sources=None, fluxes=None):
        """Internal RHS builder: ``R = -div F(U) + sum of the requested named sources`` from the
        compiled transport/source selectors. NOT a public surface -- ``pops.lib.time`` macros
        and operator-first lowering call it directly after typed validation.

        ``sources`` (ADC-425): ``None`` keeps ``-div F`` + the model's default/composite source;
        ``["default"]`` is the same explicitly; ``[]`` is FLUX ONLY (no default source); a list of
        named ``m.source_term`` names adds exactly those (plus the default iff ``"default"`` is in the
        list). ``None`` and ``[]`` are recorded DISTINCTLY in the IR. ``flux`` (ADC-430) toggles the
        ``-div F`` base: ``flux=False`` is SOURCE-ONLY (named ``fluxes`` are then rejected -- no flux
        to divide). ``fluxes`` (ADC-419): ``None``/``["default"]`` is the model's historical -div F; a
        list of NAMED ``m.flux_term`` assembles -div of their SUM (mixing ``"default"`` with named
        fluxes is rejected)."""
        state, fields = _resolve_handle(state), _resolve_handle(fields)
        if isinstance(name, Value):
            raise ValueError("rhs: pass state=/fields= by keyword (first arg is the debug name)")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("rhs: a State value is required (state=...)")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("rhs: fields must be a FieldContext from solve_fields")
        # Preserve None (default source included) DISTINCT from [] (flux only): the codegen routes on
        # whether "default" is requested.
        src = list(sources) if sources is not None else None
        attrs = {"flux": bool(flux), "sources": src, "fluxes": list(fluxes) if fluxes else None}
        inputs = (state, fields) if fields is not None else (state,)
        return self._new("rhs", "rhs", inputs, attrs, name, state.block)

    def linear_combine(self, name=None, expr=None):
        """Materialize an affine combination of State/RHS values into a new State. Accepts
        ``linear_combine(name, expr)`` or ``linear_combine(expr)``. The per-input coefficient
        polynomials in ``dt`` are recorded in ``attrs['coeffs']`` (aligned with ``inputs``)."""
        if expr is None and not isinstance(name, str):
            name, expr = None, name
        aff = _to_affine(expr)._merge()
        if not aff:
            raise ValueError("linear_combine: empty combination")
        block = None
        state_space = None
        for v, _ in aff:
            if v.vtype == "state":
                block = v.block
                if state_space is None:
                    state_space = v.space
                break
        if block is None:
            block = aff[0][0].block
        # Operator-first type check (Spec 2): every State/Rate term must live over ONE StateSpace.
        # Combining a Rate(U) with a State(V) (V != U) is a type error; untyped (legacy) terms skip.
        spaces = {nm for nm in (_state_base_name(v.space) for v, _ in aff) if nm is not None}
        if len(spaces) > 1:
            raise ValueError(
                "cannot combine values over different state spaces %s; a State and the Rate(state) "
                "added to it must share one StateSpace" % sorted(spaces))
        inputs = tuple(v for v, _ in aff)
        coeffs = [c.as_dict() for _, c in aff]
        out = self._new("state", "linear_combine", inputs, {"coeffs": coeffs}, name, block)
        out.space = state_space  # the combine result is a State over the same space
        return out

    # --- internal named-source / local-linear nodes -----------------
    @property
    def I(self):  # noqa: E743  -- the mathematical identity operator (matches the spec's P.I)
        """The identity operator, for building a local linear operator ``self.I - a * L`` (L a
        linear source). Consumed by `solve_local_linear`."""
        return _Operator(_Coeff({0: 1.0}), [])

    def _linear_source_value(self, name):
        """Internal value for local-linear lowering.

        This is intentionally underscored: user Programs must call typed operator handles. Existing
        package macros still use this while they are migrated to operator-first handles.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("_linear_source_value: name must be a non-empty string")
        return self._new("operator", "linear_source", (), {"linear_source": name}, name, None)

    def _source_value(self, name, state=None, fields=None):
        """Internal value for a named-source node."""
        state, fields = _resolve_handle(state), _resolve_handle(fields)
        if not isinstance(name, str) or not name:
            raise ValueError("_source_value: a non-empty source name is required")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("_source_value: a State value is required (state=...)")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("_source_value: fields must be a FieldContext from solve_fields")
        inputs = (state, fields) if fields is not None else (state,)
        return self._new("rhs", "source", inputs, {"source": name}, name, state.block)

    def _check_operator_state(self, l_value, state_value, where):
        """Operator-first type check (Spec 2): a LocalLinearOperator L: U -> U may only act on a State
        over U. Fires only when both carry space tags."""
        lop = getattr(l_value, "space", None) if isinstance(l_value, Value) else None
        dom = getattr(lop, "domain_name", None)
        st = _state_base_name(getattr(state_value, "space", None))
        if dom is not None and st is not None and dom != st:
            raise ValueError(
                "%s: operator maps %s -> %s but was applied to a State over %r"
                % (where, dom, getattr(lop, "range_name", dom), st))

    def apply(self, operator=None, state=None, fields=None, name=None):
        """Apply a linear-source operator to a state: ``LU = L_name(aux, params) U``. ``operator`` is
        a `linear_source` value (or its name). Returns an RHS-like value."""
        state, fields = _resolve_handle(state), _resolve_handle(fields)
        lname = self._linear_source_name(operator, "apply")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("apply: a State value is required (state=...)")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("apply: fields must be a FieldContext from solve_fields")
        self._check_operator_state(operator, state, "apply")
        inputs = (state, fields) if fields is not None else (state,)
        return self._new("rhs", "apply", inputs, {"linear_source": lname},
                         name or ("apply_" + lname), state.block)
