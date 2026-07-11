"""pops.time Program authoring mixin -- core builder ops.

State / field / RHS / source / apply construction (the operator-first builder core).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time.program_base import _ProgramConstants
from pops.time.program_call import _ProgramCall
from pops.time.program_rhs import _ProgramRhs
from pops.time.operator_resolution import resolve_operator_handle
from pops.time.references import (
    bind_field_reference, bind_program_block, bind_state_reference, block_name, field_name,
)
from pops.time.program_value_validation import (
    merge_state_spaces, rate_space_for, require_compatible_spaces,
    require_declared_state_space, require_owned, validate_input_regions,
)
from pops.time.values import (
    ProgramValue, _Affine, _Coeff, _Operator, _authoring_source_location, _resolve_handle,
    _to_affine,
)

if TYPE_CHECKING:
    from pops.time._program_contract import _ProgramBase
else:
    _ProgramBase = object

_UNCHANGED = object()


class _ProgramCore(_ProgramCall, _ProgramRhs, _ProgramConstants, _ProgramBase):
    """State / field / RHS / source / apply construction (the operator-first builder core).

    The typed operator-call lowering (``P.call`` / ``_call`` and helpers) is mixed in from
    :class:`pops.time.program_call._ProgramCall` (split for the 500-line cap, ADC-550).
    """

    # --- node construction ---
    def _new(self, vtype: Any, op: Any, inputs: Any, attrs: Any, name: Any, block: Any, *,
             space: Any = None, field_context: Any = None, state_ref: Any = None) -> Any:
        region = self._current_region()
        validate_input_regions(self, inputs, region, "IR op %r" % op)
        vid = self._next_id
        if name is None:
            name = "%s%d" % (op, vid)
        elif not isinstance(name, str) or not name:
            raise ValueError("IR op %r name must be a non-empty string or None" % op)
        self._next_id += 1
        source_location = _authoring_source_location() if self._capture_source else None
        value_inputs = [i for i in inputs if isinstance(i, ProgramValue)]
        if state_ref is None:
            input_refs = {item.state_ref for item in value_inputs if item.state_ref is not None}
            if len(input_refs) == 1:
                state_ref = next(iter(input_refs))
        v = ProgramValue(self, vid, vtype, op, value_inputs,
                         attrs, name, block,
                         space=space, source_location=source_location,
                         field_context=field_context, region=region, state_ref=state_ref)
        self._issued_values[id(v)] = v
        # Inside a control-flow recording scope (cond_fn / body_fn of a while_), ops go into the active
        # sub-block, NOT the flat self._values: a while body must RE-EXECUTE each iteration, so its ops
        # are owned by the while op and re-emitted in the loop, never walked once at the top level.
        if self._recording:
            self._recording[-1].append(v)
        else:
            self._values.append(v)
        return v

    def _canonical_value(self, value: Any) -> Any:
        """Return the current immutable record for ``value``'s SSA id.

        Builder completion (a named view, a schedule or a matrix-free apply body) replaces a record
        atomically instead of mutating it.  Existing external references remain usable because the
        SSA id is stable and every completion boundary canonicalizes through this helper.
        """
        if (not isinstance(value, ProgramValue) or value.prog is not self
                or self._issued_values.get(id(value)) is not value):
            return value
        for block in reversed(self._recording):
            for current in reversed(block):
                if current.id == value.id:
                    return current
        for current in reversed(self._values):
            if current.id == value.id:
                return current
        return value

    def _replace_value(self, value: Any, *, attrs: Any = _UNCHANGED, name: Any = _UNCHANGED,
                       space: Any = _UNCHANGED, field_context: Any = _UNCHANGED) -> Any:
        """Replace one builder-owned SSA record with a newly constructed immutable record."""
        if getattr(self, "_frozen", False):
            raise RuntimeError(
                "pops.time.Program %r is frozen: cannot replace an authored IR record" % self.name)
        if self._issued_values.get(id(value)) is not value:
            raise ValueError("cannot replace a ProgramValue not authored by this Program")
        current = self._canonical_value(value)
        if not isinstance(current, ProgramValue) or current.prog is not self:
            raise ValueError("cannot replace a ProgramValue owned by another Program")
        replacement = ProgramValue(
            self,
            current.id,
            current.vtype,
            current.op,
            current.inputs,
            current.attrs if attrs is _UNCHANGED else attrs,
            current.name if name is _UNCHANGED else name,
            current.block,
            space=current.space if space is _UNCHANGED else space,
            source_location=current.source_location,
            field_context=(current.field_context if field_context is _UNCHANGED else field_context),
            region=current.region,
            state_ref=current.state_ref,
        )
        self._issued_values[id(replacement)] = replacement
        for collection in list(reversed(self._recording)) + [self._values]:
            for index, candidate in enumerate(collection):
                if candidate.id == current.id:
                    collection[index] = replacement
                    for block, committed in tuple(self._commits.items()):
                        if committed.id == current.id:
                            self._commits[block] = replacement
                    if self._dt_bound is not None and self._dt_bound[1].id == current.id:
                        self._dt_bound = (self._dt_bound[0], replacement)
                    return replacement
        raise ValueError("ProgramValue #%d is not present in its Program" % current.id)

    def state(self, block: Any, state: Any) -> Any:
        """Declare one block-qualified temporal state family.

        The sole public form is ``T.state(block_handle, model_state_handle)``. The model-local state
        is authenticated by the block's authoritative case registry and immediately qualified as
        ``block[state]``. From this boundary onward the Program stores the qualified handle; neither
        the block nor the state is ever represented by a free string.
        """
        self._guard_mutable("declare a state")
        block, qualified_state = bind_state_reference(block, state)
        bind_program_block(self, block, where="Program.state")
        space = getattr(qualified_state, "space", None)
        if space is None:
            space = self._default_state_spaces.get(block.model_owner_path)
        require_declared_state_space(self, qualified_state, space)
        return self._time_state(block, qualified_state, space)

    def solve_fields(self, name: Any = None, state: Any = None, field: Any = None) -> Any:
        """Solve the elliptic fields from ``state`` and return a FieldContext. Accepts
        ``solve_fields(state)`` or ``solve_fields(name, state)``. Each call is a DISTINCT
        FieldContext (a stage's RHS must read the fields solved from its own state, never a stale
        global). ``field`` is the case-owned ``FieldHandle`` returned by ``Problem.add_field`` for a
        named elliptic solve; a string is never promoted into a field identity."""
        # A readable temporal handle (U.stage(k) / U.prev(lag)) resolves through Program-owned tables
        # here so it composes wherever a State does; a plain ProgramValue / None / str is unchanged.
        name, state = _resolve_handle(name), _resolve_handle(state)
        if isinstance(name, ProgramValue) and state is None:
            name, state = None, name
        if field is not None:
            if not (isinstance(state, ProgramValue) and state.vtype == "state"):
                raise ValueError("solve_fields: a State value is required")
            field = bind_field_reference(self, state.block, field)
        return self._solve_fields(name=name, state=state, field=field)

    def _solve_fields(self, name: Any, state: Any, field: Any = None) -> Any:
        """Internal typed field-solve builder used after handle authentication."""
        if not (isinstance(state, ProgramValue) and state.vtype == "state"):
            raise ValueError("solve_fields: a State value is required")
        attrs = {"field": field} if field is not None else {}
        # ADC-588: tag the value with a typed FieldContext (the "solve_fields returns a FieldContext"
        # contract, now a real object). The default problem exposes the historical phi/grad outputs;
        # a named field exposes its own single output. The context is build-time metadata only, NEVER
        # serialized as canonical provenance so validation, rewrites and cache identity agree.
        from pops.time.field_context import DEFAULT_FIELD_PROBLEM, FieldContext
        output_name = field_name(field) if field is not None else None
        outputs = ("phi", "grad_x", "grad_y") if field is None else (output_name,)
        context = FieldContext(
            field or DEFAULT_FIELD_PROBLEM, ((state.block, state.id),), outputs)
        default_field_space = self._default_field_spaces.get(state.block.model_owner_path)
        return self._new(
            "fields", "solve_fields", (state,), attrs, name, state.block,
            field_context=context, space=(default_field_space if field is None else None))

    def solve_fields_from_blocks(self, states: Any, name: Any = None) -> Any:
        """Solve the elliptic fields from the SIMULTANEOUS stage states of MULTIPLE blocks (spec
        \"Multi-blocs\"): a coupled Poisson where each listed block reads its own @p states[k] override
        at once, returning a FieldContext.

        RUNTIME (Spec 3 criterion 24, ADC-457): this is the multi-target coupled solve. It lowers to
        ``ctx.solve_fields_from_blocks(u_stages)``, a per-block pointer vector the native field solver
        (``System::solve_fields_from_blocks`` ->
        ``SystemFieldSolver::assemble_poisson_rhs_from_blocks``) assembles the system Poisson RHS from as
        ``Sum_s elliptic_rhs_s(U_s)`` reading EVERY listed block's stage state AT ONCE (a true
        simultaneous override, not a sequence of single-target solves). A block NOT listed contributes
        its live state. The listed states slot at their block index (the T.state declaration order), so
        the runtime sees each coupled block at its stage state into the one shared phi/aux.

        A per-block ``P.solve_fields(state=Ub)`` remains the right choice when the blocks advance in
        sequence (block b at its stage state, every other block at its live state); this op is for the
        SIMULTANEOUS case where multiple coupled blocks must each contribute their stage state at once."""
        if not (isinstance(states, (list, tuple)) and states):
            raise ValueError("solve_fields_from_blocks: a non-empty list of State values is required")
        seen = set()
        for s in states:
            if not (isinstance(s, ProgramValue) and s.vtype == "state"):
                raise ValueError("solve_fields_from_blocks: every entry must be a State value")
            if s.block in seen:
                raise ValueError("solve_fields_from_blocks: block '%s' listed twice" % s.block)
            seen.add(s.block)
        # A coupled solve carries EVERY exact block/state source. It has no singular block owner:
        # projecting onto states[0] would allow only that block to be checked and silently discard
        # the provenance of every other simultaneous override.
        from pops.time.field_context import DEFAULT_FIELD_PROBLEM, FieldContext
        context = FieldContext(
            DEFAULT_FIELD_PROBLEM,
            tuple((state.block, state.id) for state in states),
            ("phi", "grad_x", "grad_y"),
        )
        field_spaces = {
            self._default_field_spaces.get(state.block.model_owner_path) for state in states
        }
        field_spaces.discard(None)
        field_space = next(iter(field_spaces)) if len(field_spaces) == 1 else None
        return self._new(
            "fields", "solve_fields_from_blocks", tuple(states), {}, name, None,
            field_context=context, space=field_space)

    # --- operator-first calls (Spec 2) -------------------------------------------
    def bind_operators(self, source: Any) -> Any:
        """Bind a typed operator registry so ``P.call`` can resolve and type-check operators.

        ``source`` is an ``pops.model.OperatorRegistry`` or any object exposing
        ``operator_registry()`` (a ``dsl.Model`` / ``pops.model.Module``). Returns ``self`` for
        chaining. The bound registry is build-time TYPE information only -- the codegen still reads
        the model passed to ``compile_problem``; operator-first Programs and the ``pops.lib.time``
        macros bind the module's operators here.
        """
        self._guard_mutable("bind operators")
        reg = source.operator_registry() if hasattr(source, "operator_registry") else source
        if not (hasattr(reg, "get") and hasattr(reg, "names")):
            raise TypeError("bind_operators: expected an OperatorRegistry or an object exposing "
                            "operator_registry(); got %r" % (source,))
        owner = getattr(reg, "owner_path", None)
        if owner is None:
            raise ValueError(
                "bind_operators: registry must expose its authoritative OwnerPath owner_path")
        existing = self._operator_registries.get(owner)
        if existing is not None:
            if existing is reg:
                return self
            raise ValueError(
                "bind_operators: owner %s is already bound to a different registry" % owner)
        canonical_owner = owner.canonical()
        collision = next(
            (bound_owner for bound_owner in self._operator_registries
             if bound_owner.canonical() == canonical_owner), None)
        if collision is not None:
            raise ValueError(
                "bind_operators: distinct authoring registries claim canonical owner %s"
                % canonical_owner)
        self._operator_registries[owner] = reg
        inferred = []
        inferred_fields = []
        for operator_name in reg.names():
            signature = reg.get(operator_name).signature
            for candidate in signature.inputs:
                if getattr(candidate, "kind", None) == "state" and candidate not in inferred:
                    inferred.append(candidate)
            output = signature.output
            if getattr(output, "kind", None) == "field" and output not in inferred_fields:
                inferred_fields.append(output)
        state_space = inferred[0] if len(inferred) == 1 else None
        field_space = inferred_fields[0] if len(inferred_fields) == 1 else None
        self._default_state_spaces[owner] = state_space
        self._default_field_spaces[owner] = field_space
        # A model may finish declaring imposed aux fields after an early Program binding. Rebinding
        # before freeze widens already-authored FieldContext values to the registry's now-authoritative
        # complete FieldSpace by immutable SSA replacement; stale external references canonicalize by
        # id. This changes the IR hash and never silently keeps the earlier partial type.
        if field_space is not None:
            for value in tuple(self._values):
                if (value.vtype == "fields" and value.block is not None
                        and value.block.model_owner_path == owner
                        and value.space != field_space
                        and (value.space is None
                             or value.space.name == field_space.name)):
                    self._replace_value(value, space=field_space)
        return self

    def linear_combine(self, name: Any = None, expr: Any = None) -> Any:
        """Materialize an affine combination of State/RHS values into a new State. Accepts
        ``linear_combine(name, expr)`` or ``linear_combine(expr)``. The per-input coefficient
        polynomials in ``dt`` are recorded in ``attrs['coeffs']`` (aligned with ``inputs``).

        A combination whose terms are ALL ``scalar_field`` values materializes a ``scalar_field``
        instead (ADC-427: the condensed-Schur phi^{n+1} = phi^n + (1/theta)(phi^{n+theta} - phi^n)
        extrapolation over 1-component potentials). The State path is unchanged -- the scalar branch
        activates only when no State/RHS term is present, so an existing all-State combine serializes
        and hashes byte-identically. The two vtypes never mix in one affine (a scalar_field and a State
        are different grid shapes); the codegen lowers both through the same axpy/lincomb idiom."""
        if expr is None and not isinstance(name, str):
            name, expr = None, name
        raw = tuple((self._canonical_value(value), coeff)
                    for value, coeff in _to_affine(expr).terms)
        for value, _ in raw:
            require_owned(self, value, "linear_combine")
        aff = _Affine(raw)._merge()
        if not aff:
            raise ValueError("linear_combine: empty combination")
        if any(v.vtype == "scalar_field" for v, _ in aff) and not all(
                v.vtype == "scalar_field" for v, _ in aff):
            raise ValueError("linear_combine: scalar fields cannot mix with State/Rate values")
        # ADC-427: an affine whose terms are ALL scalar_field yields a scalar_field.  A solve_linear
        # result retains the block provenance of its rhs, while a raw P.scalar_field scratch is
        # unqualified (block=None).  Preserve the single known block across the combination: otherwise
        # a valid ``phi_next = phi + correction`` would lose its provenance and could not be committed
        # through the block-qualified endpoint.  Two distinct known blocks are a type error; silently
        # choosing the first would make a cross-block commit possible.  Unqualified scratch terms may
        # participate alongside one qualified value, but cannot manufacture a block on their own.
        if all(v.vtype == "scalar_field" for v, _ in aff):
            blocks = {v.block for v, _ in aff if v.block is not None}
            if len(blocks) > 1:
                raise ValueError(
                    "cannot combine scalar fields owned by different blocks %s"
                    % sorted(block_name(item) for item in blocks))
            block = next(iter(blocks), None)
            inputs = tuple(v for v, _ in aff)
            # A block-qualified scalar result can represent a one-component state (notably a
            # scalar-domain Krylov solve).  Preserve its single structural StateSpace through
            # combinations with unqualified scratch fields so the result remains commit-compatible.
            spaces = [v.space for v in inputs if v.space is not None]
            space = spaces[0] if spaces else None
            for candidate in spaces[1:]:
                require_compatible_spaces(space, candidate, "linear_combine scalar fields")
            coeffs = [c.to_polynomial() for _, c in aff]
            return self._new(
                "scalar_field", "linear_combine", inputs, {"coeffs": coeffs}, name, block,
                space=space)
        inputs = tuple(v for v, _ in aff)
        # Structural type errors outrank the secondary block-label mismatch.
        state_space = merge_state_spaces(inputs, "linear_combine")
        blocks = {value.block for value in inputs if value.block is not None}
        if len(blocks) > 1:
            raise ValueError(
                "linear_combine: cannot combine values owned by different blocks %s"
                % sorted(block_name(item) for item in blocks))
        block = next(iter(blocks), None)
        from pops.time.field_context import merge_field_contexts
        field_context = merge_field_contexts(inputs, "linear_combine")
        coeffs = [c.to_polynomial() for _, c in aff]
        return self._new(
            "state", "linear_combine", inputs, {"coeffs": coeffs}, name, block,
            space=state_space, field_context=field_context)

    # --- named sources / local linear operators (Phase 4 / ADC-403) ---
    @property
    def I(self) -> Any:  # noqa: E743  -- the mathematical identity operator (matches the spec's P.I)
        """The identity operator, for building a local linear operator ``self.I - a * L`` (L a
        linear source). Consumed by `solve_local_linear`."""
        return _Operator(_Coeff({0: 1}), [])

    def linear_source(self, operator: Any) -> Any:
        """Reference a model linear-source operator ``L`` (declared via ``m.linear_source`` /
        ``m.local_linear_map``), for operator algebra (``self.I - a * P.linear_source(L)``) or `apply`.
        ``operator`` MUST be the typed :class:`pops.model.OperatorHandle` the declarer returned
        (ADC-532 / ADC-625): a free string is REFUSED here with a ``TypeError`` naming the handle form;
        the lowering / lib.time macros lower through the ``_linear_source`` seam with the bare name, so
        the IR is byte-identical to the historical string form."""
        if isinstance(operator, str):
            raise TypeError(
                "linear_source: a free string %r is not accepted on the public route; pass the typed "
                "OperatorHandle the declarer returned (P.linear_source(handle))" % (operator,))
        resolved = resolve_operator_handle(
            self, operator, where="linear_source",
            expected_kinds="local_linear_operator")
        return self._linear_source(resolved.name)

    def source(self, operator: Any, state: Any = None, fields: Any = None) -> Any:
        """Evaluate one typed model source ``S(U, fields)`` on its own.

        ``operator`` is the exact :class:`pops.model.OperatorHandle` returned by
        ``m.source_term``. A free name is refused; the private :meth:`_source` seam carries the
        registry-local name after owner/kind/signature validation. Returns an RHS-like value.
        """
        if isinstance(operator, str):
            raise TypeError(
                "source: a free string %r is not accepted on the public route; pass the "
                "OperatorHandle returned by m.source_term(...)" % operator)
        state, fields = _resolve_handle(state), _resolve_handle(fields)
        resolved = resolve_operator_handle(
            self, operator, where="source", expected_kinds="local_source",
            values=tuple(value for value in (state, fields) if value is not None))
        source_name = resolved.lowering.get("source", resolved.name)
        if source_name == "default":
            return self._rhs_legacy(
                name=operator.name, state=state, fields=fields,
                flux=False, sources=["default"])
        return self._source(source_name, state=state, fields=fields)

    def _source(self, name: Any, state: Any = None, fields: Any = None) -> Any:
        """Private lowering seam for a registry-local source name."""
        state, fields = _resolve_handle(state), _resolve_handle(fields)
        if not isinstance(name, str) or not name:
            raise ValueError("_source: a non-empty source name is required")
        if not (isinstance(state, ProgramValue) and state.vtype == "state"):
            raise ValueError("source: a State value is required (state=...)")
        if fields is not None and not (isinstance(fields, ProgramValue) and fields.vtype == "fields"):
            raise ValueError("source: fields must be a FieldContext from solve_fields")
        field_context = None
        if fields is not None:
            from pops.time.field_context import require_field_read
            field_context = require_field_read(fields, state, "source")
        inputs = (state, fields) if fields is not None else (state,)
        return self._new(
            "rhs", "source", inputs, {"source": name}, name, state.block,
            space=rate_space_for(state.space), field_context=field_context)

    def _check_operator_state(self, l_value: Any, state_value: Any, where: Any) -> Any:
        """Operator-first type check (Spec 2): a LocalLinearOperator L: U -> U may only act on a State
        over U. Fires only when both carry space tags (P.call / T.state(block, U))."""
        lop = getattr(l_value, "space", None) if isinstance(l_value, ProgramValue) else None
        domain = getattr(lop, "domain", None)
        range_ = getattr(lop, "range", None)
        state_space = getattr(state_value, "space", None)
        if (domain is not None and range_ is not None and state_space is not None
                and (domain != state_space or range_ != state_space)):
            raise ValueError(
                "%s: operator maps %r -> %r but was applied to a State over %r; "
                "space compatibility is structural, not name-based"
                % (where, domain, range_, state_space))

    def apply(self, operator: Any = None, state: Any = None, fields: Any = None,
              name: Any = None) -> Any:
        """Apply a linear-source operator to a state: ``LU = L_name(aux, params) U``.

        ``operator`` MUST be a typed :meth:`linear_source` value or an
        :class:`pops.model.OperatorHandle` (ADC-625): a free string is REFUSED on this public route
        with a ``TypeError`` naming the handle form. Returns an RHS-like value."""
        if isinstance(operator, str):
            raise TypeError(
                "apply: a free string %r is not accepted on the public route; pass a typed "
                "P.linear_source(handle) value or the OperatorHandle" % (operator,))
        return self._apply(operator, state=state, fields=fields, name=name)
