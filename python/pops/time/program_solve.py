"""pops.time Program authoring mixin -- solve / commit / record ops.

Krylov solve_linear, histories, commits, board sugar (fields/define/solve) and records.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time.handles import (
    HistoryHandle, StageHandle, StateEndpointHandle, TimeState,
)
from pops.time.program_base import _ProgramConstants
from pops.time.program_commit_validation import validate_commit_many
from pops.time.program_diagnostics import _ProgramDiagnostics
from pops.time.program_transaction import atomic_authoring
from pops.time.references import block_name
from pops.time.program_value_validation import (
    require_compatible_spaces, require_top_level, structural_state_space,
)
from pops.time.value_metadata import positive_scalar_literal
from pops.time.values import ProgramValue, _Affine, _is_field_value, _resolve_handle

if TYPE_CHECKING:
    from pops.time._program_contract import _ProgramBase
else:
    _ProgramBase = object


def _lower_krylov_method(method: Any) -> Any:
    """Lower a typed Krylov descriptor to ``(scheme, options)`` (Spec 5 sec.7).

    ``method`` is a :mod:`pops.solvers.krylov` descriptor (``CG()`` / ``GMRES()`` /
    ``BiCGStab()`` / ``Richardson()``); its ``scheme`` is the C++ token (``"cg"`` ...) the
    runtime keys on, so the typed object lowers byte-identically to the historical string. A
    bare algorithm-selector string is REJECTED (Spec 5 forbids keeping the string form on the
    public surface); ``None`` defaults to the ``cg`` scheme.

    The descriptor's own ``max_iter`` (mandatory at descriptor construction, ADC-535) is metadata
    for the LinearProblem-lowering path; the program's ``P.solve_linear(max_iter=...)`` argument is
    the authoritative budget for THIS op. ADC-645: ``options`` carries the descriptor's optional
    ``rel_tol`` (supplies ``tol`` when the call site leaves it default) and ``omega`` (Richardson
    relaxation, baked at emit) -- absent when unset, so a default descriptor lowers as before.
    """
    if method is None:
        return "cg", {}
    if isinstance(method, str):
        raise TypeError(
            "solve_linear: method must be a typed pops.solvers.krylov descriptor "
            "(e.g. pops.solvers.krylov.GMRES() / CG() / BiCGStab() / Richardson()), not the "
            "string %r" % (method,))
    scheme = getattr(method, "scheme", None)
    if getattr(method, "category", None) != "solver" or not isinstance(scheme, str):
        raise TypeError(
            "solve_linear: method must be a pops.solvers.krylov descriptor "
            "(CG() / GMRES() / BiCGStab() / Richardson()); got %r" % (method,))
    options = dict(getattr(method, "options", None) or {})
    if "omega" in options and scheme != "richardson":
        raise ValueError(
            "solve_linear: omega only applies to Richardson() (the relaxation factor of "
            "pops::richardson_solve); got method %r" % (scheme,))
    return scheme, options


# Preconditioner schemes that lower to REAL C++ in the matrix-free Krylov path (Spec 5 sec.7, ADC-516):
#   - "identity":     the empty pops::ApplyFn{} (unpreconditioned; the historical default);
#   - "geometric_mg": one V-cycle of the wired pops::GeometricMG, emitted as a real ApplyFn callback.
# The planned-but-unwired schemes (jacobi / block_jacobi) carry available=False and have no native
# kernel yet; they are rejected with an HONEST "planned, not wired" message (a separate issue), never a
# transitional catch-all.
_WIRED_PRECOND_SCHEMES = frozenset({"identity", "geometric_mg"})


def _lower_preconditioner(preconditioner: Any) -> Any:
    """Lower a typed preconditioner descriptor to ``(scheme, precond_options|None)`` (Spec 5 sec.7).

    ``preconditioner`` is a :mod:`pops.solvers.preconditioners` descriptor
    (``preconditioners.Identity()`` / ``preconditioners.GeometricMG()`` ...); its ``scheme`` is the
    C++ token. A bare string is REJECTED; ``None`` defaults to ``Identity()`` (the unpreconditioned
    default). The geometric-multigrid preconditioner lowers to a real V-cycle ApplyFn; the planned
    jacobi / block_jacobi descriptors have no native kernel yet and are rejected with an honest
    "planned, not wired" message (out of scope -- a separate issue).

    ADC-644: a ``GeometricMG(...)`` with validated V-cycle-shape knobs returns its option dict; a
    default one returns ``None`` (the IR omits ``precond_options`` -> emitted V-cycle byte-identical).
    """
    if preconditioner is None:
        preconditioner = _preconditioners().Identity()
    if isinstance(preconditioner, str):
        raise TypeError(
            "solve_linear: preconditioner must be a typed pops.solvers.preconditioners "
            "descriptor (e.g. pops.solvers.preconditioners.Identity() / GeometricMG()), not the "
            "string %r" % (preconditioner,))
    scheme = getattr(preconditioner, "scheme", None)
    if getattr(preconditioner, "category", None) != "preconditioner" or not isinstance(scheme, str):
        raise TypeError(
            "solve_linear: preconditioner must be a pops.solvers.preconditioners descriptor "
            "(e.g. Identity() / GeometricMG()); got %r" % (preconditioner,))
    if scheme not in _WIRED_PRECOND_SCHEMES:
        # A catalogued-but-unwired preconditioner (jacobi / block_jacobi): no native C++ kernel yet.
        # An HONEST capability limit, not a transitional reject -- wiring it is tracked separately.
        raise NotImplementedError(
            "solve_linear: the %r preconditioner is planned, not wired yet (it needs a native C++ "
            "kernel); use preconditioners.Identity() or preconditioners.GeometricMG()" % (scheme,))
    options = getattr(preconditioner, "options", None)
    precond_options = dict(options) if options else None
    return scheme, precond_options


def _preconditioners() -> Any:
    """The pops.solvers.preconditioners catalog (imported lazily)."""
    from pops.solvers import preconditioners
    return preconditioners


class _ProgramSolve(_ProgramDiagnostics, _ProgramConstants, _ProgramBase):
    """Krylov solve_linear, histories, commits, board sugar (fields/define/solve) and records."""

    def solve_linear(self, name: Any = None, operator: Any = None, rhs: Any = None,
                     initial_guess: Any = None, method: Any = None, preconditioner: Any = None,
                     tol: Any = None, max_iter: Any = None, restart: Any = None) -> Any:
        """Solve the matrix-free linear system ``operator x = rhs`` with the runtime's Krylov loop and
        return the solution as a field. A state-domain operator with a State rhs returns a State;
        scratch/vector solves return a scalar_field. The iteration is DYNAMIC (C++-side, inside the loop):
        the IR only carries the operator (its apply lambda), the rhs, the initial guess, and the
        method / tolerance / iteration budget.

          - @p operator: a ``matrix_free_operator`` value (with a ``set_apply`` body);
          - @p rhs: the right-hand side -- a scalar_field or State value. A typed StateSpace must have
            exactly the operator's ``ncomp``; an untyped State is checked against the physical model
            component count during lowering;
          - @p initial_guess: warm start (defaults to zero);
          - @p method: a TYPED Krylov descriptor (``pops.solvers.krylov.CG()`` (SPD),
            ``BiCGStab()`` (general), ``Richardson()``, or ``GMRES()`` -- restarted GMRES(m), the
            robust choice for a NON-symmetric operator). A bare string is REJECTED (Spec 5
            sec.7); ``None`` defaults to ``CG()``;
          - @p preconditioner: a typed ``pops.solvers.preconditioners`` descriptor.
            ``Identity()`` (the unpreconditioned default) and ``GeometricMG()`` (one V-cycle of the
            wired geometric multigrid, for ``GMRES()`` / ``BiCGStab()`` only) lower to real C++; the
            planned ``Jacobi()`` / ``BlockJacobi()`` are rejected (no native kernel yet). A non-identity
            preconditioner with ``CG()`` / ``Richardson()`` is rejected (those loops have no
            preconditioner slot). A bare string is REJECTED; ``None`` defaults to ``Identity()``;
          - @p tol: relative L2 residual stop (> 0);
          - @p max_iter: iteration budget (REQUIRED, > 0: a dynamic solver loop with no budget is a
            configuration error -- ``pops::*_solve`` itself throws on a non-positive budget);
          - @p restart: GMRES restart length m (a positive int; defaults to 30). Ignored by the other
            methods; passing it to a non-gmres solve is rejected."""
        # Spec 5 sec.7: method / preconditioner are TYPED descriptors (pops.solvers.krylov /
        # pops.solvers.preconditioners). They lower to the SAME internal scheme tokens the runtime
        # always keyed on, so the IR / emitted C++ stay byte-identical to the historical string path;
        # a bare algorithm-selector string is rejected (the public string form is removed).
        operator = self._canonical_value(operator)
        method, method_options = _lower_krylov_method(method)
        preconditioner, precond_options = _lower_preconditioner(preconditioner)
        # ADC-645: the call-site tol stays the authoritative per-op budget; left default (None) it
        # falls back to the descriptor's optional rel_tol, then to the historical 1e-8 -- so a
        # default program resolves tol to the same 1e-8 and the IR node is byte-identical.
        if tol is None:
            tol = method_options.get("rel_tol", 1e-8)
        if not (isinstance(operator, ProgramValue) and operator.vtype == "matrix_free_op"):
            raise ValueError("solve_linear: operator must be a matrix_free_operator value")
        if operator.attrs["apply_block"] is None:
            raise ValueError("solve_linear: operator '%s' has no apply; call P.set_apply first"
                             % operator.name)
        if not _is_field_value(rhs):
            raise ValueError("solve_linear: rhs must be a scalar_field or State value (rhs=...)")
        if initial_guess is not None and not _is_field_value(initial_guess):
            raise ValueError("solve_linear: initial_guess must be a scalar_field or State value")
        if initial_guess is not None:
            unqualified_scratch = (
                initial_guess.vtype == "scalar_field" and initial_guess.block is None
                and initial_guess.space is None)
            if initial_guess.block != rhs.block and not unqualified_scratch:
                raise ValueError("solve_linear: rhs and initial_guess must belong to the same block")
            if not unqualified_scratch:
                require_compatible_spaces(
                    rhs.space, initial_guess.space, "solve_linear initial_guess", typed_pair=True)
        op_ncomp = int(operator.attrs["ncomp"])
        # The rhs / initial guess must carry at least the operator's component count: the solve runs on
        # an op_ncomp buffer. A scalar_field exposes its ncomp here; a State's n_cons is only known at
        # compile (against the model), so a State is accepted now and checked there.
        for label, fld in (("rhs", rhs), ("initial_guess", initial_guess)):
            if fld is None:
                continue
            state_space = structural_state_space(fld.space)
            if fld.vtype == "state" and state_space is not None:
                fld_ncomp = len(state_space.components)
                if fld_ncomp != op_ncomp:
                    raise ValueError(
                        "solve_linear: %s StateSpace has %d component(s) but the operator declares "
                        "ncomp=%d" % (label, fld_ncomp, op_ncomp))
                continue
            if fld.vtype != "scalar_field":
                continue
            fld_ncomp = int(fld.attrs.get("ncomp", 1))
            if fld_ncomp < op_ncomp:
                raise ValueError(
                    "solve_linear: %s has %d component(s) but the operator needs %d (a scalar_field "
                    "with ncomp >= the operator ncomp, or a State)" % (label, fld_ncomp, op_ncomp))
        if method not in self._KRYLOV_METHODS:
            raise ValueError("solve_linear: method must be one of %s; got %r"
                             % (sorted(self._KRYLOV_METHODS), method))
        # A non-identity preconditioner needs the runtime ApplyFn slot, which only the Krylov methods
        # that take one (BiCGStab / GMRES, generic_krylov.hpp) expose; pops::cg_solve / richardson_solve
        # have NO preconditioner parameter. This is an honest capability limit of the matrix-free path,
        # not a transitional reject.
        if preconditioner != "identity" and method not in ("gmres", "bicgstab"):
            raise ValueError(
                "solve_linear: preconditioning is not available for CG/Richardson in the matrix-free "
                "Krylov path; use GMRES() or BiCGStab()")
        tol_literal = positive_scalar_literal(tol, where="solve_linear: tol")
        if (max_iter is None or isinstance(max_iter, bool)
                or not isinstance(max_iter, int) or max_iter <= 0):
            raise ValueError("dynamic solver loops require max_iter")
        # restart is a gmres-only knob; the GMRES(m) basis size. Other methods have no restart concept,
        # so passing one to them is a config error (fail loud rather than silently ignore it).
        if method == "gmres":
            if restart is None:
                restart = self._GMRES_RESTART_DEFAULT
            elif isinstance(restart, bool) or not isinstance(restart, int) or restart <= 0:
                raise ValueError("solve_linear: restart must be a positive integer for gmres (got %r)"
                                 % (restart,))
        elif restart is not None:
            raise ValueError("solve_linear: restart only applies to method='gmres' (got method=%r)"
                             % (method,))
        inputs = (operator, rhs) if initial_guess is None else (operator, rhs, initial_guess)
        # restart is a positive int on the gmres path (validated above); the None union member the
        # checker infers is from the non-gmres branch, which takes the else arm of the ternary.
        restart_int = int(restart) if method == "gmres" else None  # pyright: ignore[reportArgumentType]
        attrs = {"method": method, "preconditioner": preconditioner, "tol": tol_literal,
                 "max_iter": int(max_iter), "has_guess": initial_guess is not None,
                 "ncomp": op_ncomp, "restart": restart_int}
        # ADC-644: the resolved V-cycle-shape options of a configured GeometricMG preconditioner. Added
        # ONLY when non-None (a default GeometricMG() lowers to None), so an unconfigured program's IR
        # hash / emitted source stays byte-identical (the attr is JSON-dumped into _serialize_node).
        if precond_options is not None:
            attrs["precond_options"] = precond_options
        # ADC-645: Richardson relaxation factor, added ONLY when the descriptor set it (a default
        # Richardson() program's IR hash / emitted source stays byte-identical: omega = 1 literal).
        if "omega" in method_options:
            attrs["omega"] = positive_scalar_literal(
                method_options["omega"], where="solve_linear: Richardson omega")
        # A state-domain solve over a State rhs returns a State, preserving the mathematical unknown's
        # block and StateSpace. Scalar/vector scratch solves remain scalar_field values. This keeps a
        # Newton update ``U + dU`` typed without an implicit scalar-field-to-State conversion.
        result_type = (
            "state" if operator.attrs["domain"] == "state" and rhs.vtype == "state"
            else "scalar_field")
        return self._new(
            result_type, "solve_linear", inputs, attrs, name, rhs.block, space=rhs.space)

    def commit(self, endpoint: StateEndpointHandle, state: ProgramValue) -> None:
        """Commit ``state`` to ``endpoint``, at most once for its block.

        The public form is exactly ``P.commit(U.next, U_next)``. ``U.next`` is a
        commit-only :class:`StateEndpointHandle`; a block-name string is not a public target and a
        stage handle is not implicitly resolved. ``state`` must already be a ProgramValue owned by
        this Program.

        A 1-component model's conservative state may be represented by a ``scalar_field`` (for
        example a ``solve_linear`` result); it is accepted and copied back into the block state by
        the final runtime ``ctx.lincomb``."""
        self._guard_mutable("commit a state")
        if not isinstance(endpoint, StateEndpointHandle):
            raise TypeError(
                "commit: target must be U.next (a StateEndpointHandle); block-name strings and "
                "stage handles are not public commit targets")
        endpoint = self._require_endpoint(endpoint, "commit")
        if isinstance(state, ProgramValue) and state.block != endpoint.block:
            raise ValueError(
                "commit: cross-block write: endpoint for block %r cannot receive a value owned by block %r"
                % (endpoint.block, state.block)
            )
        require_top_level(self, state, "commit")
        require_compatible_spaces(endpoint.space, state.space, "commit", typed_pair=True)
        return self._commit_state(endpoint.state, state)

    def _commit_state(self, state_ref: Any, state: ProgramValue) -> None:
        """Record one validated qualified-state commit."""
        self._guard_mutable("commit a state")
        from pops.model.handles import Handle
        if not isinstance(state_ref, Handle) or state_ref.kind != "state" \
                or not state_ref.is_instance:
            raise TypeError("_commit_state: target must be a block-qualified state Handle")
        block = state_ref.block_ref
        if not (isinstance(state, ProgramValue) and state.vtype in ("state", "scalar_field")):
            raise TypeError("_commit_state: a State (or scalar_field) ProgramValue is required")
        if state.prog is not self:
            raise ValueError("_commit_state: the State value belongs to a different Program")
        require_top_level(self, state, "_commit_state")
        if state.block != block:
            raise ValueError(
                "_commit_state: block %r cannot receive a value owned by block %r"
                % (block_name(block), block_name(state.block)))
        if state.state_ref is not None and state.state_ref != state_ref:
            raise ValueError(
                "_commit_state: state %s cannot receive a value derived from %s"
                % (state_ref.qualified_id, state.state_ref.qualified_id))
        if block not in self._state_spaces:
            raise ValueError(
                "_commit_state: block %r has no declared StateSpace" % block_name(block))
        require_compatible_spaces(
            self._state_spaces[block], state.space, "_commit_state", typed_pair=True)
        if state_ref in self._commits:
            raise ValueError("state %s committed more than once" % state_ref.qualified_id)
        if any(committed.block_ref is block for committed in self._commits):
            raise ValueError(
                "block %r already has a committed state; multi-state block storage is not "
                "supported by the current runtime" % block_name(block))
        self._commits[state_ref] = state

    def commits(self) -> Any:
        """Map of qualified state Handle -> committed State value (copy)."""
        return dict(self._commits)

    # --- board-like sugar (Spec 3): T.define / T.fields / T.solve / T.commit_many ---
    # These lower to the SAME primitive ops as the P.call / linear_combine /
    # solve_local_linear / commit style; they are blackboard notation, not a new IR.
    def op(self, operator: Any) -> Any:
        """Return callable board sugar for an exact typed operator handle.

        ``expl = P.op(explicit_rate)`` followed by ``expl(U, fields)`` builds the same IR as
        ``P.call(explicit_rate, U, fields)``. A free name is refused at creation; the closure retains
        the handle's owner, kind and signature until the ordinary typed call boundary.
        """
        from pops.time.operator_resolution import resolve_operator_handle
        resolve_operator_handle(self, operator, where="P.op")

        def _handle(*args: Any, value_name: Any = None) -> Any:
            return self.call(operator, *args, name=value_name)

        _handle.__name__ = operator.name
        return _handle

    def fields(self, name: Any, from_state: Any = None, from_states: Any = None,
               from_state_set: Any = None, operator: Any = None) -> Any:
        """Board sugar for a field solve selected by an optional typed operator handle.

        With ``operator=None`` this uses the generic default field solve. Otherwise ``operator`` must
        be the exact ``field_operator`` handle returned by the declaring model; strings never select
        a field route.
        """
        if from_state_set is not None:
            states = from_state_set.states()
        elif from_states is not None:
            states = list(from_states)
        elif from_state is not None:
            states = [from_state]
        else:
            raise ValueError("fields: provide from_state=, from_states= or from_state_set=")
        if operator is not None:
            from pops.time.operator_resolution import resolve_operator_handle
            resolve_operator_handle(
                self, operator, where="P.fields", expected_kinds="field_operator", values=states)
            return self.call(operator, *states, name=name)
        if len(states) == 1:
            return self.solve_fields(name, states[0])
        return self.solve_fields_from_blocks(states, name=name)

    def value(self, name: Any, expr: Any) -> Any:
        """Name an intermediate SSA value ``name`` from ``expr`` (ADC-561: the short named-value form).

        The lightweight spelling of the free-value case of :meth:`define`::

            U_star = T.value("rhs_star", U.n + T.dt * R_n)
            Q      = T.value("Q", U.n + 0.5 * T.dt * R_n + 0.5 * T.dt * R_star)

        Returns the named IR handle (a :class:`pops.time.values.ProgramValue`) so the value composes in the
        affine algebra. It lowers to the EXACT ``program.define(name, expr)`` path (an affine
        combination materializes via ``linear_combine``, a ``rate(U) == <expr>`` equation keeps its
        right-hand side, any other ProgramValue is named in place), so ``T.value(name, expr)`` produces the
        byte-identical IR as ``T.define(name, expr)`` -- and the SSA invariants (single definition, no
        redefine, use-before-define) are unchanged.

        ``name`` MUST be a non-empty string (an intermediate SSA value, never a mutation): the
        stage handles stay the ``T.define(U.stage(k), ...)`` door, while ``U.next`` is exclusively a
        commit destination. Passing either kind here is refused with the corresponding final API.
        """
        if isinstance(name, StateEndpointHandle):
            self._require_endpoint(name, "T.value")
            raise TypeError(
                "T.value: U.next is a commit-only StateEndpointHandle; "
                "use T.commit(U.next, value)")
        if isinstance(name, StageHandle):
            self._require_stage(name, "T.value")
        elif isinstance(name, HistoryHandle):
            self._require_history(name, "T.value")
        elif isinstance(name, TimeState):
            self._require_time_state(name, "T.value")
        if isinstance(name, (StageHandle, HistoryHandle, TimeState)):
            raise TypeError(
                "T.value(name, expr) names a free intermediate value: pass a string name. For a "
                "temporal stage use T.define(U.stage(k), expr).")
        if not isinstance(name, str) or not name:
            raise ValueError("T.value: name must be a non-empty string")
        return self.define(name, expr)

    def define(self, name: Any, value: Any = None) -> Any:
        """Name a value or assign a stage; a state endpoint is never definable.

        Two forms (additive; the ``(name, value)`` board form is unchanged):

          - ``P.define("U1", U0 + dt * k0)`` (board sugar) names a value: an affine combination of
            states materializes via ``linear_combine``, a ``rate(U) == <expr>`` equation keeps its
            right-hand side, and any other ProgramValue is named in place;
          - ``P.define(U.stage(1), U.n + dt * k0)`` assigns a generated SSA name to the
            stage and enforces single assignment. ``T.define(U.n, ...)`` raises because the
            current state is read-only; ``T.define(U.prev, ...)`` raises because history is
            policy-owned; and ``T.define(U.next, ...)`` raises because ``U.next`` is the
            commit-only destination of ``T.commit(U.next, value)``.

        The handle form is detected by the FIRST argument being a version handle; the legacy form
        keeps a string name.
        """
        if isinstance(name, StateEndpointHandle):
            self._require_endpoint(name, "T.define")
            raise TypeError(
                "T.define: U.next is a commit-only StateEndpointHandle; "
                "use T.commit(U.next, value)")
        if isinstance(name, StageHandle):
            return self._define_stage(name, value)
        if isinstance(name, HistoryHandle):
            self._require_history(name, "T.define")
            raise ValueError("history is produced by the history policy")
        if isinstance(name, TimeState):
            self._require_time_state(name, "T.define")
            raise ValueError(
                "T.define: pass a version handle (U.stage(k) / U.next), not the TimeState itself")
        if isinstance(name, ProgramValue):
            # The handle ``U.n`` is a State ProgramValue (the current state). No legacy define names a ProgramValue
            # (the board form always passes a string name), so a ProgramValue target can only be a misuse of
            # the read-only current state -- reject it with the spec message.
            raise ValueError("current state is read-only in Program")
        value = _resolve_handle(value)  # a bare defined handle as the rhs names its resolved ProgramValue
        from pops import math as _bm
        if isinstance(value, _bm.Equation):
            if not isinstance(value.lhs, _bm.TimeDerivative):
                raise ValueError("define(%r): an equation must read 'rate(U) == <rate expression>'"
                                 % (name,))
            value = value.rhs
        if isinstance(value, _Affine):
            return self.linear_combine(name, value)
        if isinstance(value, ProgramValue):
            return self._replace_value(value, name=name)
        raise TypeError(
            "define(%r): expected a ProgramValue, an affine combination, or a rate equation; got %r"
            % (name, value))

    @atomic_authoring
    def solve(self, name: Any, equation: Any) -> Any:
        """Board sugar for an implicit local solve ``(I -/+ a*L) @ unknown("x") == rhs``.

        Lowers to ``linear_combine`` (if the rhs is an affine combination) then
        ``solve_local_linear``; identical IR to writing those two calls by hand.
        """
        from pops import math as _bm
        if not isinstance(equation, _bm.Equation):
            raise TypeError("solve(%r): expected '(I - dt*C) @ unknown(\"x\") == rhs'" % (name,))
        lhs, rhs = equation.lhs, equation.rhs
        if not isinstance(lhs, _bm.OpApply):
            raise ValueError("solve(%r): left-hand side must be 'operator @ unknown(name)'" % (name,))
        if isinstance(rhs, _Affine):
            rhs = self.linear_combine(name + "_rhs", rhs)
        elif not (isinstance(rhs, ProgramValue) and rhs.vtype == "state"):
            raise ValueError("solve(%r): right-hand side must be a State or an affine of States"
                             % (name,))
        return self.solve_local_linear(name=name, operator=lhs.operator, rhs=rhs)

    def commit_many(self, mapping: Any) -> None:
        """Commit ``{Ua.next: Ua_next, Ub.next: Ub_next}`` as one atomic group.

        Every endpoint/value owner and block is checked before ``_commits`` changes.
        """
        self._guard_mutable("commit a state group")
        self._commits.update(validate_commit_many(self, mapping))

    # --- inspection / debug (Spec 3 section 33): show the lowering ---
