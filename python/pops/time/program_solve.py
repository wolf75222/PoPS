"""pops.time Program authoring mixin -- solve / commit / record ops.

Krylov solve_linear, histories, commits, board sugar (fields/define/solve) and records.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time.program_base import _ProgramConstants
from pops.time.values import StageStateSet, Value, _Affine, _is_field_value, _resolve_handle

if TYPE_CHECKING:
    from pops.time._program_contract import _ProgramBase
else:
    _ProgramBase = object


def _lower_krylov_method(method: Any) -> Any:
    """Lower a typed Krylov descriptor to its internal scheme token (Spec 5 sec.7).

    ``method`` is a :mod:`pops.solvers.krylov` descriptor (``CG()`` / ``GMRES()`` /
    ``BiCGStab()`` / ``Richardson()``); its ``scheme`` is the C++ token (``"cg"`` ...) the
    runtime keys on, so the typed object lowers byte-identically to the historical string. A
    bare algorithm-selector string is REJECTED (Spec 5 forbids keeping the string form on the
    public surface); ``None`` defaults to the ``cg`` scheme.

    Only the ``scheme`` is read here: the descriptor's own ``max_iter`` (mandatory at descriptor
    construction, ADC-535) is metadata for the LinearProblem-lowering path; the program's
    ``P.solve_linear(max_iter=...)`` argument is the authoritative budget for THIS op, so the
    ``None`` default returns the ``cg`` token directly without constructing a budgeted descriptor.
    """
    if method is None:
        return "cg"
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
    return scheme


# Preconditioner schemes that lower to REAL C++ in the matrix-free Krylov path (Spec 5 sec.7, ADC-516):
#   - "identity":     the empty pops::ApplyFn{} (unpreconditioned; the historical default);
#   - "geometric_mg": one V-cycle of the wired pops::GeometricMG, emitted as a real ApplyFn callback.
# The planned-but-unwired schemes (jacobi / block_jacobi) carry available=False and have no native
# kernel yet; they are rejected with an HONEST "planned, not wired" message (a separate issue), never a
# transitional catch-all.
_WIRED_PRECOND_SCHEMES = frozenset({"identity", "geometric_mg"})


def _lower_preconditioner(preconditioner: Any) -> Any:
    """Lower a typed preconditioner descriptor to its scheme token (Spec 5 sec.7).

    ``preconditioner`` is a :mod:`pops.solvers.preconditioners` descriptor
    (``preconditioners.Identity()`` / ``preconditioners.GeometricMG()`` ...); its ``scheme`` is the
    C++ token. A bare string is REJECTED; ``None`` defaults to ``Identity()`` (the unpreconditioned
    default). The geometric-multigrid preconditioner lowers to a real V-cycle ApplyFn; the planned
    jacobi / block_jacobi descriptors have no native kernel yet and are rejected with an honest
    "planned, not wired" message (out of scope -- a separate issue).
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
    return scheme


def _preconditioners() -> Any:
    """The pops.solvers.preconditioners catalog (imported lazily)."""
    from pops.solvers import preconditioners
    return preconditioners


class _ProgramSolve(_ProgramConstants, _ProgramBase):
    """Krylov solve_linear, histories, commits, board sugar (fields/define/solve) and records."""

    def solve_linear(self, name: Any = None, operator: Any = None, rhs: Any = None,
                     initial_guess: Any = None, method: Any = None, preconditioner: Any = None,
                     tol: Any = 1e-8, max_iter: Any = None, restart: Any = None) -> Any:
        """Solve the matrix-free linear system ``operator x = rhs`` with the runtime's Krylov loop and
        return the solution as a scalar_field. The iteration is DYNAMIC (C++-side, inside the loop):
        the IR only carries the operator (its apply lambda), the rhs, the initial guess, and the
        method / tolerance / iteration budget.

          - @p operator: a ``matrix_free_operator`` value (with a ``set_apply`` body);
          - @p rhs: the right-hand side -- a scalar_field, or (MVP) a 1-component State value;
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
        method = _lower_krylov_method(method)
        preconditioner = _lower_preconditioner(preconditioner)
        if not (isinstance(operator, Value) and operator.vtype == "matrix_free_op"):
            raise ValueError("solve_linear: operator must be a matrix_free_operator value")
        if operator.attrs["apply_block"] is None:
            raise ValueError("solve_linear: operator '%s' has no apply; call P.set_apply first"
                             % operator.name)
        if not _is_field_value(rhs):
            raise ValueError("solve_linear: rhs must be a scalar_field or State value (rhs=...)")
        if initial_guess is not None and not _is_field_value(initial_guess):
            raise ValueError("solve_linear: initial_guess must be a scalar_field or State value")
        op_ncomp = int(operator.attrs["ncomp"])
        # The rhs / initial guess must carry at least the operator's component count: the solve runs on
        # an op_ncomp buffer. A scalar_field exposes its ncomp here; a State's n_cons is only known at
        # compile (against the model), so a State is accepted now and checked there.
        for label, fld in (("rhs", rhs), ("initial_guess", initial_guess)):
            if fld is None or fld.vtype != "scalar_field":
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
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError("solve_linear: tol must be a positive number (got %r)" % (tol,))
        if max_iter is None or not isinstance(max_iter, int) or max_iter <= 0:
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
        return self._new("scalar_field", "solve_linear", inputs,
                         {"method": method, "preconditioner": preconditioner, "tol": float(tol),
                          "max_iter": int(max_iter), "has_guess": initial_guess is not None,
                          "ncomp": op_ncomp,
                          "restart": restart_int}, name, rhs.block)

    # --- multistep histories (ADC-406a) ---
    def history(self, name: Any, lag: Any = 1, ncomp: Any = None) -> Any:
        """Read a SYSTEM-OWNED history field carried across macro-steps: the value stored @p lag steps
        back (e.g. ``P.history("plasma.R", lag=1)`` is R_{n-1} for Adams-Bashforth). Returns a value
        usable in the affine algebra. The history is owned by the System (a HistoryManager), not the
        Program, so a later checkpoint slice can serialize it; reading it before it has ever been stored
        is a fail-loud runtime error (it must be written by `store_history` every step). @p lag must be
        a Python int >= 1.

        @p ncomp (ADC-427) sizes the ring's slots. ``None`` (the default) reads a STATE-typed value over
        block 0's ncomp -- the full-state multistep ring, byte-identical to the historical IR. An
        explicit ``ncomp=1`` reads a SCALAR_FIELD-typed value from a 1-component ring: the persistent
        1-component carry the condensed-Schur theta<1 stage needs for its cross-step phi^n (a lag-1
        potential kept across steps, not a full state). The narrower ring is declared with this ncomp in
        the codegen prelude, so a bare ``ctx.history`` read never widens it."""
        if not isinstance(name, str) or not name:
            raise ValueError("history: name must be a non-empty string")
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
            raise ValueError("history: lag must be a Python int >= 1 (got %r)" % (lag,))
        self._histories[name] = max(self._histories.get(name, 0), lag)
        if ncomp is None:
            # DEFAULT: a full-state ring, State-typed. attrs unchanged from ADC-406a so every existing
            # multistep history node serializes and hashes byte-identically (no ncomp key).
            return self._new("state", "history", (), {"history": name, "lag": int(lag)}, name, None)
        if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp < 1:
            raise ValueError("history: ncomp must be None or a Python int >= 1 (got %r)" % (ncomp,))
        # An explicit ncomp: a scalar-field ring (block=None, like P.scalar_field). Only ncomp=1 is a
        # meaningful narrow carry today (the phi^n potential); a wider explicit ncomp is a State the
        # default already covers, so require 1 here (fail loud rather than silently alias the state ring).
        if ncomp != 1:
            raise ValueError(
                "history: an explicit ncomp must be 1 (the 1-component carry); the full-state ring is "
                "the default ncomp=None (got %r)" % (ncomp,))
        self._histories_ncomp[name] = int(ncomp)
        return self._new("scalar_field", "history", (),
                         {"history": name, "lag": int(lag), "ncomp": int(ncomp)}, name, None)

    def store_history(self, name: Any, value: Any) -> Any:
        """Store @p value (a State/RHS field) into the CURRENT slot of history @p name at the end of the
        step (rotated to lag 1 on the next step). A multistep scheme stores its current RHS so the next
        step can read it back via `history`. The history is System-owned; this is a side-effecting op
        (no value). @p value must be a State/RHS field of the Program."""
        if not isinstance(name, str) or not name:
            raise ValueError("store_history: name must be a non-empty string")
        if not _is_field_value(value):
            raise ValueError("store_history: value must be a State/RHS field (got %r)" % (value,))
        if value.prog is not self:
            raise ValueError("store_history: the value belongs to a different Program")
        self._histories.setdefault(name, 1)
        return self._new("state", "store_history", (value,), {"history": name}, name, value.block)

    def keep_history(self, timestate: Any, depth: Any, cold_start: Any = None,
                     checkpoint_policy: Any = None) -> Any:
        """Keep a ring of past states for a :class:`pops.time.handles.TimeState` (Spec 5 sec.5.3.1).

        Records the ring ``depth`` and the ``cold_start`` policy on the handle and lowers a
        ``store_history("<block>.<name>", U.n)`` so the System-owned ring is populated every step.
        After this, ``U.prev(lag)`` (for ``lag <= depth``) reads the lagged state via ``P.history``.
        ``cold_start`` defaults to :class:`pops.time.history.CopyCurrent` (seed every slot with the
        current state on step 0, the historical behavior). Returns the lowered ``store_history`` node.

        @p checkpoint_policy (ADC-626) is a typed history-persistence policy
        (:class:`pops.time.Dense` / :class:`~pops.time.Interval` / :class:`~pops.time.Revolve`)
        selecting which ring slots a checkpoint STORES; the remaining slots are recomputed at restart
        by deterministic replay of the installed compiled Program (bit-identical to a dense-restored
        ring). ``None`` (the default) resolves to :class:`~pops.time.Dense` (the whole-ring historical
        behaviour). The policy is validated against @p depth at author time (loud on incoherence)."""
        from pops.time.handles import TimeState
        if not isinstance(timestate, TimeState):
            raise ValueError(
                "keep_history: a TimeState handle is required (P.state('U', block=...))")
        if timestate.program is not self:
            raise ValueError("keep_history: the TimeState belongs to a different Program")
        return timestate._keep_history(depth, cold_start, checkpoint_policy)

    def commit(self, block: Any, state: Any = None) -> Any:
        """Replace the current state of ``block`` with ``state`` at the end of the step. Each block
        is committed AT MOST once; read-only blocks need no commit.

        Two forms (additive; the positional ``(block, state)`` form is unchanged):

          - ``P.commit("plasma", U_next)`` (LEGACY) commits a State value to a named block;
          - ``P.commit(U.next)`` (Spec 5 sec.5.3.1) commits a single typed version handle to its own
            block (``commit(handle.block, handle.value)``). The version must have been defined
            (``T.define(U.next, ...)``) first; an undefined handle raises.

        @p state is normally a State value; a 1-component model's conservative state doubles as a
        scalar field, so a ``scalar_field`` (e.g. a ``solve_linear`` solution) is also accepted and
        copied back into the block state at commit (the final ``ctx.lincomb`` in the lowered body)."""
        from pops.time.handles import _Version
        if isinstance(block, _Version) and state is None:
            version = block
            return self.commit(version.block, version.value)  # version.value raises if undefined
        state = _resolve_handle(state)  # P.commit("blk", U.next) also resolves a defined handle
        if not (isinstance(state, Value) and state.vtype in ("state", "scalar_field")):
            raise ValueError("commit: a State (or scalar_field) value is required")
        if state.prog is not self:
            raise ValueError("commit: the State value belongs to a different Program")
        if block in self._commits:
            raise ValueError("block '%s' committed more than once" % block)
        self._commits[block] = state

    def commits(self) -> Any:
        """Map of committed block -> committed State value (copy)."""
        return dict(self._commits)

    # --- board-like sugar (Spec 3): T.define / T.fields / T.solve / T.commit_many ---
    # These lower to the SAME primitive ops as the P.call / linear_combine /
    # solve_local_linear / commit style; they are blackboard notation, not a new IR.
    def op(self, name: Any) -> Any:
        """Return a callable board handle for a bound operator: ``expl = P.op("explicit_rate")``
        then ``expl(U, fields)`` builds the same IR as ``P.call(rate_handle, U, fields)``. The
        board handle names the operator at creation (``P.op(name)``), so its call lowers through the
        INTERNAL ``P._call`` -- the name is an internal selector, not the public handle-only path."""
        def _handle(*args: Any, value_name: Any = None) -> Any:
            return self._call(name, *args, name=value_name)
        _handle.__name__ = str(name)
        return _handle

    def fields(self, name: Any, from_state: Any = None, from_states: Any = None,
               from_state_set: Any = None, operator: Any = None) -> Any:
        """Board sugar for a field solve. Lowers through the internal ``P._call(operator, ...)`` when
        a named operator is bound, else to ``P.solve_fields`` (single state) or
        ``P.solve_fields_from_blocks`` (the board names the operator here; ``_call`` is the internal
        selector path, not the public handle-only ``P.call``)."""
        if from_state_set is not None:
            states = from_state_set.states()
        elif from_states is not None:
            states = list(from_states)
        elif from_state is not None:
            states = [from_state]
        else:
            raise ValueError("fields: provide from_state=, from_states= or from_state_set=")
        named = operator is not None and operator != "fields_from_state"
        if len(states) == 1:
            if named and self._registry is not None:
                return self._call(operator, states[0], name=name)
            return self.solve_fields(name, states[0])
        if named and self._registry is not None:
            return self._call(operator, *states, name=name)
        return self.solve_fields_from_blocks(states, name=name)

    def value(self, name: Any, expr: Any) -> Any:
        """Name an intermediate SSA value ``name`` from ``expr`` (ADC-561: the short named-value form).

        The lightweight spelling of the free-value case of :meth:`define`::

            U_star = T.value("rhs_star", U.n + T.dt * R_n)
            Q      = T.value("Q", U.n + 0.5 * T.dt * R_n + 0.5 * T.dt * R_star)

        Returns the named IR handle (a :class:`pops.time.values.Value`) so the value composes in the
        affine algebra. It lowers to the EXACT ``program.define(name, expr)`` path (an affine
        combination materializes via ``linear_combine``, a ``rate(U) == <expr>`` equation keeps its
        right-hand side, any other Value is named in place), so ``T.value(name, expr)`` produces the
        byte-identical IR as ``T.define(name, expr)`` -- and the SSA invariants (single definition, no
        redefine, use-before-define) are unchanged.

        ``name`` MUST be a non-empty string (an intermediate SSA value, never a mutation): the
        temporal-VERSION handles (``U.stage(k)`` / ``U.next``) stay the ``T.define(handle, ...)`` door,
        and ``T.define(U.next, value)`` remains the commit-facing definition. Passing a version handle
        here is refused pointing at ``T.define``.
        """
        from pops.time.handles import TimeState, _Prev, _Version
        if isinstance(name, (_Version, _Prev, TimeState)):
            raise TypeError(
                "T.value(name, expr) names a free intermediate value: pass a string name. For a "
                "temporal version use T.define(U.stage(k) / U.next, expr).")
        if not isinstance(name, str) or not name:
            raise ValueError("T.value: name must be a non-empty string")
        return self.define(name, expr)

    def define(self, name: Any, value: Any = None) -> Any:
        """Board sugar to name a value, or lower a typed temporal-version handle (Spec 5 sec.5.3.1).

        Two forms (additive; the ``(name, value)`` board form is unchanged):

          - ``P.define("U1", U0 + dt * k0)`` (board sugar) names a value: an affine combination of
            states materializes via ``linear_combine``, a ``rate(U) == <expr>`` equation keeps its
            right-hand side, and any other Value is named in place;
          - ``P.define(U.stage(1), U.n + dt * k0)`` / ``P.define(U.next, ...)`` (handle form) lowers
            the same way through this method (with a generated name) and binds the resulting Value
            onto the version handle, enforcing SSA single assignment. ``T.define(U.n, ...)`` raises
            (the current state is read-only) and ``T.define(U.prev, ...)`` raises (history is
            produced by the history policy).

        The handle form is detected by the FIRST argument being a version handle; the legacy form
        keeps a string name.
        """
        from pops.time.handles import TimeState, _Prev, _Version
        if isinstance(name, (_Version, _Prev)):
            timestate = name._timestate
            return timestate._define(name, value)
        if isinstance(name, TimeState):
            raise ValueError(
                "T.define: pass a version handle (U.stage(k) / U.next), not the TimeState itself")
        if isinstance(name, Value):
            # The handle ``U.n`` is a State Value (the current state). No legacy define names a Value
            # (the board form always passes a string name), so a Value target can only be a misuse of
            # the read-only current state -- reject it with the spec message.
            raise ValueError("current state is read-only in Program")
        value = _resolve_handle(value)  # a bare defined handle as the rhs names its resolved Value
        from pops import math as _bm
        if isinstance(value, _bm.Equation):
            if not isinstance(value.lhs, _bm.TimeDerivative):
                raise ValueError("define(%r): an equation must read 'rate(U) == <rate expression>'"
                                 % (name,))
            value = value.rhs
        if isinstance(value, _Affine):
            return self.linear_combine(name, value)
        if isinstance(value, Value):
            value.name = name
            return value
        raise TypeError(
            "define(%r): expected a Value, an affine combination, or a rate equation; got %r"
            % (name, value))

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
        elif not (isinstance(rhs, Value) and rhs.vtype == "state"):
            raise ValueError("solve(%r): right-hand side must be a State or an affine of States"
                             % (name,))
        return self.solve_local_linear(name=name, operator=lhs.operator, rhs=rhs)

    def commit_many(self, mapping: Any, fields: Any = None) -> Any:
        """Atomically commit several coupled blocks (Spec 3). ALL entries are validated before any
        commit, so a partial or double commit of a coupled group is rejected as a unit and no block
        is left half-committed. ``fields`` (optional) is validated as a coherent FieldContext but is
        RESERVED: the IR commit has no fields slot yet (the runtime association lands with ADC-457)."""
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("commit_many: a non-empty {block: State} mapping is required")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("commit_many: fields must be a FieldContext from solve_fields")
        for block, state in mapping.items():
            if not (isinstance(state, Value) and state.vtype in ("state", "scalar_field")):
                raise ValueError("commit_many: block %r needs a State value" % (block,))
            if state.prog is not self:
                raise ValueError("commit_many: the State for %r belongs to a different Program"
                                 % (block,))
            if block in self._commits:
                raise ValueError("block '%s' committed more than once" % (block,))
        for block, state in mapping.items():
            self._commits[block] = state

    def state_set(self, name: Any, mapping: Any) -> Any:
        """Build a :class:`StageStateSet` -- a coherent set of stage states for a field solve."""
        return StageStateSet(name, mapping)

    def record(self, name: Any, value: Any) -> Any:
        """Record a scalar diagnostic (board sugar over :meth:`record_scalar`).

        ``value`` is a Program scalar -- a reduction result such as ``P.sum(U)`` or
        ``P.norm2(U)`` (the runtime value of a generic invariant). The automatic
        reduction of an arbitrary ``integral(expr)`` over a per-cell expression is a
        follow-up (it needs the scheduler / a generated reduction kernel, ADC-458)."""
        if not (isinstance(value, Value) and value.vtype == "scalar"):
            raise ValueError(
                "record(%r): value must be a Program scalar (e.g. P.sum / P.norm2); got %r"
                % (name, value))
        return self.record_scalar(name, value)

    def check_invariant(self, name: Any, before: Any = None, after: Any = None,
                        tolerance: Any = 1e-10) -> Any:
        """Record the drift of a generic invariant between two stages (board diagnostic).

        ``before`` / ``after`` are Program scalars (reduction results); the recorded
        diagnostic ``"<name>_drift"`` is ``after - before``. ``tolerance`` is carried as
        metadata for a later assertion stage (the scheduled runtime check is ADC-458)."""
        if not (isinstance(before, Value) and before.vtype == "scalar"
                and isinstance(after, Value) and after.vtype == "scalar"):
            raise ValueError(
                "check_invariant(%r): before/after must be Program scalars" % (name,))
        drift = after - before
        out = self.record_scalar(name + "_drift", drift)
        out.attrs["tolerance"] = float(tolerance)
        return out

    # --- inspection / debug (Spec 3 section 33): show the lowering ---
