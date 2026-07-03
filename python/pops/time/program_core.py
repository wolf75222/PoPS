"""pops.time Program authoring mixin -- core builder ops.

State / field / RHS / source / apply construction (the operator-first builder core).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time.program_base import _ProgramConstants
from pops.time.program_call import _ProgramCall
from pops.time.values import (
    Value, _Coeff, _Operator, _authoring_source_location, _resolve_handle,
    _state_base_name, _to_affine,
)

if TYPE_CHECKING:
    from pops.time._program_contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramCore(_ProgramCall, _ProgramConstants, _ProgramBase):
    """State / field / RHS / source / apply construction (the operator-first builder core).

    The typed operator-call lowering (``P.call`` / ``_call`` and helpers) is mixed in from
    :class:`pops.time.program_call._ProgramCall` (split for the 500-line cap, ADC-550).
    """

    # --- node construction ---
    def _new(self, vtype: Any, op: Any, inputs: Any, attrs: Any, name: Any, block: Any) -> Any:
        for v in inputs:
            if isinstance(v, Value) and v.prog is not self:
                raise ValueError("IR value %r belongs to a different Program" % (v,))
        vid = self._next_id
        self._next_id += 1
        v = Value(self, vid, vtype, op, [i for i in inputs if isinstance(i, Value)],
                  attrs, name or ("%s%d" % (op, vid)), block)
        if self._capture_source:
            v.source_location = _authoring_source_location()
        # Inside a control-flow recording scope (cond_fn / body_fn of a while_), ops go into the active
        # sub-block, NOT the flat self._values: a while body must RE-EXECUTE each iteration, so its ops
        # are owned by the while op and re-emitted in the loop, never walked once at the top level.
        if self._recording:
            self._recording[-1].append(v)
        else:
            self._values.append(v)
        return v

    def state(self, arg: Any = None, space: Any = None, *, block: Any = None) -> Any:
        """Reference the current conservative state of a block at the start of the step.

        Two forms (additive; the positional form is the historical one, byte-identical):

          - ``P.state("plasma")`` / ``P.state("plasma", space=...)`` (LEGACY) returns the State
            :class:`pops.time.values.Value` of block ``"plasma"`` -- the positional argument is the
            block name;
          - ``P.state("U", block="plasma")`` (Spec 5 sec.5.3.1) returns a
            :class:`pops.time.handles.TimeState` -- a family of typed temporal-version handles
            (``.n`` / ``.stage`` / ``.next`` / ``.prev``) for block ``"plasma"`` named ``"U"``.

        The handle form is detected SOLELY by the ``block=`` keyword (no legacy caller passes it),
        so the positional path is unchanged.

        @p space (Spec 2): the operator-first :class:`pops.model.StateSpace` this block instantiates.
        It is recorded for type checking (a State tagged with space U cannot be combined with a
        Rate(V), and an operator expecting state U cannot be called on it) and is NOT serialized into
        the IR. ``None`` keeps the legacy untyped state (no space checks)."""
        if block is not None:
            from pops.time.handles import TimeState
            return TimeState(self, block, name=arg or "U")
        if not isinstance(arg, str) or not arg:
            raise ValueError("state: block must be a non-empty string")
        v = self._new("state", "state", (), {}, arg, arg)
        v.space = space
        return v

    def solve_fields(self, name: Any = None, state: Any = None, field: Any = None) -> Any:
        """Solve the elliptic fields from ``state`` and return a FieldContext. Accepts
        ``solve_fields(state)`` or ``solve_fields(name, state)``. Each call is a DISTINCT
        FieldContext (a stage's RHS must read the fields solved from its own state, never a stale
        global). @p field (ADC-419) names a NAMED elliptic field (m.elliptic_field) to solve instead of
        the default Poisson coupling; its derived aux populate that field's named aux channel. The
        multi-field RUNTIME is DEFERRED: a non-None @p field lowers to a clear NotImplementedError
        (the IR records it so a program reads cleanly when the runtime lands)."""
        # A defined temporal-version handle (U.stage(k) / U.next / U.prev) resolves to its Value
        # here so it composes wherever a State does; a plain Value / None / str is unchanged.
        name, state = _resolve_handle(name), _resolve_handle(state)
        if isinstance(name, Value) and state is None:
            name, state = None, name
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("solve_fields: a State value is required")
        if field is not None and not (isinstance(field, str) and field):
            raise ValueError("solve_fields: field must be a non-empty named elliptic field")
        # The attr is added ONLY for a named field so a default solve_fields keeps its historical IR
        # (empty attrs) -> the .so cache key of an existing time program is byte-identical (no spurious
        # invalidation from this feature).
        attrs = {"field": field} if field is not None else {}
        v = self._new("fields", "solve_fields", (state,), attrs, name, state.block)
        # ADC-588: tag the value with a typed FieldContext (the "solve_fields returns a FieldContext"
        # contract, now a real object). The default problem exposes the historical phi/grad outputs;
        # a named field exposes its own single output. The context is build-time metadata only, NEVER
        # serialized into the IR -> the .so cache key stays byte-identical.
        from pops.time.field_context import DEFAULT_FIELD_PROBLEM, FieldContext
        outputs = ("phi", "grad_x", "grad_y") if field is None else (field,)
        v.field_context = FieldContext(field or DEFAULT_FIELD_PROBLEM, state.block, state.id, outputs)
        return v

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
        its live state. The listed states slot at their block index (the P.state declaration order), so
        the runtime sees each coupled block at its stage state into the one shared phi/aux.

        A per-block ``P.solve_fields(state=Ub)`` remains the right choice when the blocks advance in
        sequence (block b at its stage state, every other block at its live state); this op is for the
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
        v = self._new("fields", "solve_fields_from_blocks", tuple(states), {}, name, states[0].block)
        # ADC-588: typed FieldContext for the coupled solve -- the shared default Poisson, owned by
        # the first listed block, stage-sourced by that block's state (build-time metadata only).
        from pops.time.field_context import DEFAULT_FIELD_PROBLEM, FieldContext
        v.field_context = FieldContext(
            DEFAULT_FIELD_PROBLEM, states[0].block, states[0].id, ("phi", "grad_x", "grad_y"))
        return v

    # --- operator-first calls (Spec 2) -------------------------------------------
    def bind_operators(self, source: Any) -> Any:
        """Bind a typed operator registry so ``P.call`` can resolve and type-check operators.

        ``source`` is an ``pops.model.OperatorRegistry`` or any object exposing
        ``operator_registry()`` (a ``dsl.Model`` / ``pops.model.Module``). Returns ``self`` for
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

    def rhs(self, name: Any = None, state: Any = None, fields: Any = None, *,
            terms: Any = None, **legacy: Any) -> Any:
        """Build the typed right-hand side R from a list of TYPED ``terms`` (Spec 5 sec.14.2.4, the one
        public path, ADC-479 criterion 27).

        ``terms`` is the right-hand-side composition: a :class:`pops.numerics.terms.Flux` plus the
        source terms to fold in::

            R = P.rhs(U, fields=f, terms=[Flux(), electric])

        ``fields`` is the explicit FieldContext any field-dependent source reads (no implicit global
        aux). A ``Flux()`` in the list adds the ``-div F`` base (its absence -> source only); every
        source term -- a :class:`pops.numerics.terms.SourceTerm` / :class:`~pops.numerics.terms.\
LocalTerm`, an :class:`pops.model.OperatorHandle` from ``m.source_term``, or a plain source NAME
        string -- appends its name to the folded sources. So ``terms=[Flux(), electric]`` is
        ``-div F + electric``, ``terms=[Flux()]`` is flux only, ``terms=[electric]`` is the named
        source only, and ``terms=[]`` is the zero RHS.

        The legacy ``flux=``/``sources=``/``fluxes=`` boolean/name form is NOT a public path: it is
        REFUSED with a clear ``TypeError`` naming the ``terms=`` alternative (a Program composes the
        RHS only by typed terms). The byte-identical builder it used to expose survives ONLY as the
        internal :meth:`_rhs_legacy` (used by the ``pops.lib.time`` macros and the operator-first
        lowering); a non-term object in the list (e.g. a bare ``bool`` -- ``Flux()`` is a term, not a
        bool) raises a clear ``TypeError``."""
        # The legacy flux=/sources=/fluxes= boolean/name form is NOT public: name it explicitly in a
        # clear TypeError pointing at terms=, rather than letting CPython raise an opaque "unexpected
        # keyword argument". A bare P.rhs (no terms=) is the legacy default and is refused too.
        if legacy or terms is None:
            extra = "".join(", %s=" % k for k in sorted(legacy))
            raise TypeError(
                "P.rhs requires the typed terms= list, not the legacy flux=/sources=/fluxes= form%s; "
                "pass P.rhs(state=U, fields=f, terms=[Flux(), source]) (a pops.numerics.terms.Flux "
                "plus the source terms to fold in)" % extra)
        from pops.time._rhs_terms import terms_to_flux_sources
        flux, sources = terms_to_flux_sources(terms)
        return self._rhs_legacy(name=name, state=state, fields=fields, flux=flux, sources=sources)

    def _rhs_legacy(self, name: Any = None, state: Any = None, fields: Any = None, flux: Any = True,
                    sources: Any = None, fluxes: Any = None) -> Any:
        """Internal RHS builder: ``R = -div F(U) + sum of the requested named sources`` from the
        legacy ``(flux, sources, fluxes)`` triple. NOT a public surface -- the public typed
        :meth:`rhs` lowers ``terms=`` onto this byte-identically, and the ``pops.lib.time`` macros /
        the operator-first lowering call it directly (a flux/sources string token survives here only
        as an internal selector, undocumented in the public API).

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
        # Preserve None (legacy default = flux + default source) DISTINCT from [] (flux only): the
        # codegen routes on whether "default" is requested, and None is the legacy "default included".
        src = list(sources) if sources is not None else None
        attrs = {"flux": bool(flux), "sources": src, "fluxes": list(fluxes) if fluxes else None}
        inputs = (state, fields) if fields is not None else (state,)
        return self._new("rhs", "rhs", inputs, attrs, name, state.block)

    def linear_combine(self, name: Any = None, expr: Any = None) -> Any:
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

    # --- named sources / local linear operators (Phase 4 / ADC-403) ---
    @property
    def I(self) -> Any:  # noqa: E743  -- the mathematical identity operator (matches the spec's P.I)
        """The identity operator, for building a local linear operator ``self.I - a * L`` (L a
        linear source). Consumed by `solve_local_linear`."""
        return _Operator(_Coeff({0: 1.0}), [])

    def linear_source(self, operator: Any) -> Any:
        """Reference a model linear-source operator ``L`` (declared via ``m.linear_source`` /
        ``m.local_linear_map``), for operator algebra (``self.I - a * P.linear_source(L)``) or `apply`.
        ``operator`` MUST be the typed :class:`pops.model.OperatorHandle` the declarer returned
        (ADC-532 / ADC-625): a free string is REFUSED here with a ``TypeError`` naming the handle form;
        the lowering / lib.time macros lower through the ``_linear_source`` seam with the bare name, so
        the IR is byte-identical to the historical string form."""
        from pops.model import OperatorHandle
        if isinstance(operator, str):
            raise TypeError(
                "linear_source: a free string %r is not accepted on the public route; pass the typed "
                "OperatorHandle the declarer returned (P.linear_source(handle))" % (operator,))
        if not isinstance(operator, OperatorHandle):
            raise TypeError(
                "linear_source: expected an pops.model.OperatorHandle, got %r" % (operator,))
        return self._linear_source(operator.name)

    def source(self, name: Any, state: Any = None, fields: Any = None) -> Any:
        """Evaluate a single named model source ``S_name(U, fields)`` (``m.source_term``) on its own.
        Returns an RHS-like value (a dU/dt contribution) usable in linear combinations. Named sources
        are never summed implicitly; this requests exactly one."""
        state, fields = _resolve_handle(state), _resolve_handle(fields)
        if not isinstance(name, str) or not name:
            raise ValueError("source: a non-empty source name is required")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("source: a State value is required (state=...)")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("source: fields must be a FieldContext from solve_fields")
        inputs = (state, fields) if fields is not None else (state,)
        return self._new("rhs", "source", inputs, {"source": name}, name, state.block)

    def _check_operator_state(self, l_value: Any, state_value: Any, where: Any) -> Any:
        """Operator-first type check (Spec 2): a LocalLinearOperator L: U -> U may only act on a State
        over U. Fires only when both carry space tags (P.call / P.state(space=)); legacy skips."""
        lop = getattr(l_value, "space", None) if isinstance(l_value, Value) else None
        dom = getattr(lop, "domain_name", None)
        st = _state_base_name(getattr(state_value, "space", None))
        if dom is not None and st is not None and dom != st:
            raise ValueError(
                "%s: operator maps %s -> %s but was applied to a State over %r"
                % (where, dom, getattr(lop, "range_name", dom), st))

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

