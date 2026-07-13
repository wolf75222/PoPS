"""pops.lib.time.multistep -- Adams-Bashforth and BDF (Backward Differentiation Formula) schemes.

Exports: adams_bashforth, adams_bashforth2, bdf.
Private helpers: _AB_WEIGHTS, _bdf_local_linear, _bdf_implicit_flux.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from ._helpers import (
    _DEFAULT_SOURCES, _at_point, _block_label, _commit, _operator_handle, _source_names,
    _stage_rhs, _time_state, _typed_rhs, program_macro,
)
from .euler import forward_euler as _forward_euler_macro


def _forward_euler(P: Any, temporal: Any, sources: Any, flux: Any) -> None:
    # AB1 degenerates to Forward Euler; reuse the local euler macro for byte-identical IR.
    _forward_euler_macro(P, temporal, sources=sources, flux=flux)


def _history(P: Any, name: Any, lag: Any, temporal: Any, space: Any) -> Any:
    """Read one typed full-state/rate history slot for a preset."""
    return P.history(
        name, lag=lag, space=space, block=temporal.block, state_ref=temporal.state)


# Adams-Bashforth weights b_j on R_{n-j} (j = 0..order-1), per order (ADC-423). AB1 is Forward Euler.
_AB_WEIGHTS = {
    1: (1,),
    2: (Fraction(3, 2), Fraction(-1, 2)),
    3: (Fraction(23, 12), Fraction(-16, 12), Fraction(5, 12)),
}


@program_macro
def adams_bashforth(P: Any, block: Any, state: Any = None, order: Any = None, *,
                    sources: Any = _DEFAULT_SOURCES,
                    flux: Any = True) -> Any:
    """Adams-Bashforth, explicit ``order``-step, over the System-owned history ring (ADC-406a / ADC-423):

        R_n     = R(U)
        U^{n+1} = U + dt * sum_{j=0}^{order-1} b_j * R_{n-j}
        store_history(block.R, R_n)

    ``order`` selects the classic AB weights b_j:
      - **AB1** == Forward Euler (b = 1), with NO history (it never reads or stores the ring);
      - **AB2** == (3/2, -1/2) on (R_n, R_{n-1});
      - **AB3** == (23/12, -16/12, 5/12) on (R_n, R_{n-1}, R_{n-2}).

    COLD START: the store of R_n is recorded BEFORE the lag reads, and the runtime fills EVERY history
    slot on the FIRST store, so step 0 reads R_{n-j} = R_0 for all j and the recurrence degenerates to a
    single Forward-Euler step (U^1 = U^0 + dt*R_0, since sum_j b_j = 1). From step ``order-1`` on it is
    the true AB recurrence; in between it runs the same partially-filled ring the runtime exposes. This
    is deterministic and exact; an offline reference mirrors it (FE-fill cold start then AB). The history
    name is ``"<block>.R"`` (the block's previous RHS).

    AB1 keeps Forward Euler's exact IR (no history op); AB2 keeps the historical ``"ab2_step"`` combine
    so a pre-ADC-423 AB2 program's ``.so`` cache key is byte-identical."""
    temporal = _time_state(P, block, state)
    label = _block_label(temporal)
    if isinstance(order, bool) or not isinstance(order, int) or order not in _AB_WEIGHTS:
        raise ValueError("adams_bashforth: order must be an int in %s (got %r)"
                         % (sorted(_AB_WEIGHTS), order))
    b = _AB_WEIGHTS[order]
    if order == 1:  # AB1 == Forward Euler: no history, identical IR to forward_euler.
        _forward_euler(P, temporal, sources, flux)
        return
    name = label + ".R"
    step_name = "ab2_step" if order == 2 else ("ab%d_step" % order)
    U = temporal.n
    R_n = _stage_rhs(P, U, sources, flux, name="ab_current", offset=0)
    # Store R_n FIRST (so the first store cold-start-fills the ring), then read R_{n-j} = lag j.
    P.store_history(name, R_n)
    expr = U + (P.dt * b[0]) * R_n
    for j in range(1, order):
        expr = expr + (P.dt * b[j]) * _history(P, name, j, temporal, R_n.space)
    _commit(P, temporal, P.linear_combine(
        step_name, expr, at=temporal.next.point))


@program_macro
def adams_bashforth2(P: Any, block: Any, state: Any = None, *,
                     sources: Any = _DEFAULT_SOURCES, flux: Any = True) -> Any:
    """Adams-Bashforth 2, a thin back-compat alias for ``adams_bashforth(P, block, 2)`` (ADC-423).

    Kept so existing callers and the historical ``"ab2_step"`` IR are unchanged: this lowers to the
    SAME IR as before (R_n stored first, R_{n-1} read at lag 1, weights 3/2 / -1/2)."""
    adams_bashforth(P, block, state, order=2, sources=sources, flux=flux)


def _bdf_local_linear(P: Any, temporal: Any, order: Any, linear_source: Any, sources: Any,
                      flux: Any) -> Any:
    """The cell-LOCAL linear-source BDF fast path (the historical lowering): the BDF system is
    block-diagonal, so ``(c0*I - dt*L) U^{n+1} = rhs`` is solved per cell by `P.solve_local_linear`.

      - **BDF1** (backward Euler): ``(I - dt*L) U^{n+1} = U^n [+ dt R]``;
      - **BDF2**: ``(I - (2/3) dt L) U^{n+1} = (2/3)(2 U^n - 1/2 U^{n-1}) [+ dt R]`` over the System
        history ring, with a BDF1 cold start (the first store fills every slot -> U^{n-1} = U^n)."""
    label = _block_label(temporal)
    U = temporal.n
    fields = P.solve_fields(U) if flux else None
    # Optional EXPLICIT flux/source RHS folded into the BDF right-hand side (lagged at U^n).
    R = (_typed_rhs(P, U, fields=fields, sources=sources, flux=flux)
         if (flux or sources) else None)

    def _with_explicit(expr: Any) -> Any:
        return (expr + P.dt * R) if R is not None else expr

    if order == 1:  # (I - dt*L) U^{n+1} = U^n [+ dt R]
        rhs = P.linear_combine(
            label + "_bdf1_rhs", _with_explicit(1 * U), at=temporal.next.point)
        operator = P.I - P.dt * P.linear_source(linear_source)
        out = P.solve_local_linear(
            name=label + "_bdf1_step", operator=operator, rhs=rhs, fields=fields)
        _commit(P, temporal, out)
        return out
    # BDF2: (3/2 I - dt*L) U^{n+1} = 2 U^n - 1/2 U^{n-1} [+ dt R], over the history ring.
    name = label + ".U"
    P.store_history(name, U)                       # store U^n first (cold-start fills the ring)
    U_nm1 = _history(P, name, 1, temporal, U.space)
    rhs = P.linear_combine(
        label + "_bdf2_rhs", _with_explicit(2 * U - Fraction(1, 2) * U_nm1),
        at=temporal.next.point)
    operator = P.I - (P.dt * Fraction(2, 3)) * P.linear_source(linear_source)
    # Divide both sides by 3/2: (I - (2/3) dt L) U^{n+1} = (2/3)(2 U^n - 1/2 U^{n-1} [+ dt R]).
    rhs = P.linear_combine(label + "_bdf2_rhs_scaled", Fraction(2, 3) * rhs)
    out = P.solve_local_linear(
        name=label + "_bdf2_step", operator=operator, rhs=rhs, fields=fields)
    _commit(P, temporal, out)
    return out


def _bdf_implicit_flux(P: Any, temporal: Any, order: Any, sources: Any, flux: Any, ncomp: Any,
                       newton_max: Any, krylov_tol: Any,
                       krylov_max: Any, krylov_restart: Any, eps: Any) -> Any:
    """The IMPLICIT-FLUX BDF lowering (ADC-431): a matrix-free Newton-Krylov solve of the coupled
    nonlinear system, composed PURELY from existing IR primitives (no new C++ stepper).

    The implicit BDF step solves ``F(U^{n+1}) = 0`` with::

        BDF1:  F(U) = U - U^n            - dt*rhs(U)
        BDF2:  F(U) = U - (4/3)U^n + (1/3)U^{n-1} - (2/3)*dt*rhs(U)

    (BDF2 reads ``U^{n-1}`` from the System history ring with a BDF1 cold start.) ``rhs(U) = -div F(U)
    [+ sources]`` is the SAME hyperbolic residual the explicit schemes use, so the flux couples the
    cells through its stencil and the Newton system is GLOBAL. Every residual re-solves the fields from
    its own iterate; the finite-difference Jacobian likewise re-solves them from ``U^k + h v`` so field
    coupling is part of the implicit operator, never a stale ``U^n`` side channel.

    Newton's method (the outer loop) is a fixed `static_range` unroll of @p newton_max iterations -- each
    iteration is independent IR (its own matrix-free operator + Krylov solve), which the codegen lowers
    at the top level (the install-time apply lambda the Krylov loop needs cannot live inside a runtime
    while/range body). Each iteration:

      1. ``R^k = rhs(U^k)`` (one rhs evaluation; also the frozen base of the matvec FD);
      2. ``F^k = U^k - U^n_terms - c*dt*R^k`` (the residual; ``c = 1`` BDF1, ``c = 2/3`` BDF2);
      3. solve ``J dU = -F^k`` with GMRES (J nonsymmetric), J applied matrix-free via `rhs_jacvec`
         (``J v = v - c*dt * d(rhs)/dU v``, a finite-difference Jacobian-vector product around U^k);
      4. ``U^{k+1} = U^k + dU``.

    The final residual norm ``||F||`` is recorded as the diagnostic ``"<block>.bdf_residual"`` (read via
    ``sim.program_diagnostic``). @p ncomp is the block component count (1 by default -- a scalar model
    like inviscid Burgers / linear advection; pass the model's n_cons for a multi-component block)."""
    c = 1 if order == 1 else Fraction(2, 3)
    label = _block_label(temporal)
    U0 = temporal.n
    endpoint = temporal.next.point
    # Snapshot U^n into a scratch: the commit writes this block's runtime state IN PLACE at the very
    # end, so the lagged term must read this frozen copy (not the live state) -- otherwise the post-commit residual
    # diagnostic would read U^{n+1} as U^n. The Newton-loop residuals (before the commit) would be correct
    # either way; the snapshot keeps every residual (loop + diagnostic) reading the true U^n.
    Un = P.linear_combine(label + "_bdf_Un", 1 * U0)
    if order == 2:
        name = label + ".U"
        P.store_history(name, U0)                   # store U^n (cold-start fills the ring)
        U_nm1 = _history(P, name, 1, temporal, U0.space)

    def _un_terms() -> Any:
        # The lagged (constant-in-Newton) part of the residual: U^n for BDF1, (4/3)U^n - (1/3)U^{n-1}
        # for BDF2 (the constant-state coefficients of the BDF residual normalized to a unit U^{n+1}).
        if order == 1:
            return 1 * Un
        return Fraction(4, 3) * Un - Fraction(1, 3) * U_nm1

    src = list(sources) if sources is not None else None
    field_coupled = src is None or "default" in src
    # The Newton unknown is physically a conservative State even for a one-component model. Runtime
    # storage is still an ncomp-wide MultiFab; declaring the state domain preserves the Program type
    # of the Krylov correction instead of relying on a hidden scalar_field -> State conversion.
    kind = "state"

    def _residual(P: Any, Uk: Any, tag: Any) -> Any:
        # F^k = U^k - U^n_terms - c*dt*rhs(U^k); returns (F^k, R^k) so the matvec can reuse R^k.
        fields_k = _at_point(P, P.solve_fields(Uk), endpoint) if field_coupled else None
        Rk = _at_point(P, P._rhs_legacy(
            name="%s_R" % tag, state=Uk, fields=fields_k, flux=flux, sources=src), endpoint)
        Fk = P.linear_combine(
            "%s_F" % tag, _un_terms() * -1 + Uk - (c * P.dt) * Rk, at=endpoint)
        return Fk, Rk

    def _newton_step(P: Any, Uk: Any, k: Any) -> Any:
        tag = "%s_bdf%d_n%d" % (label, order, k)
        Fk, Rk = _residual(P, Uk, tag)
        negF = P.linear_combine("%s_negF" % tag, -1 * Fk)
        A = P.matrix_free_operator("%s_J" % tag, domain=kind, range_=kind,
                                   ncomp=ncomp)

        def apply(P: Any, out: Any, v: Any) -> Any:
            # J v = v - c*dt * d(rhs)/dU v, matrix-free FD around the frozen iterate U^k (r0 = R^k).
            return P.rhs_jacvec(out, v, iterate=Uk, r0=Rk, c_dt=(c * P.dt), eps=eps, flux=flux,
                                sources=sources, field_coupled=field_coupled)

        from pops.solvers import krylov
        from pops.time import FailRun
        P.set_apply(A, apply)
        dU = P.solve_linear(name="%s_dU" % tag, operator=A, rhs=negF,
                            method=krylov.GMRES(max_iter=krylov_max),
                            tol=krylov_tol, max_iter=krylov_max,
                            restart=krylov_restart).consume(action=FailRun())
        return P.linear_combine("%s_next" % tag, Uk + dU, at=endpoint)

    # Outer Newton loop: a fixed unroll of newton_max iterations (each independent top-level IR).
    Uk = U0
    for k in range(newton_max):
        Uk = _newton_step(P, Uk, k)
    # Record the final residual norm for diagnostics (sim.program_diagnostic("<block>.bdf_residual")).
    Ffinal, _ = _residual(P, Uk, "%s_bdf%d_final" % (label, order))
    P.record_scalar(label + ".bdf_residual", P.norm2(Ffinal))
    _commit(P, temporal, Uk)
    return Uk


@program_macro
def bdf(P: Any, block: Any, state: Any = None, order: Any = None, *,
        linear_source: Any = None,
        sources: Any = _DEFAULT_SOURCES, flux: Any = True, ncomp: Any = 1,
        newton_max: Any = 20, krylov_tol: Any = 1e-10,
        krylov_max: Any = 200, krylov_restart: Any = None, eps: Any = 1e-7) -> Any:
    """Backward Differentiation Formula, IMPLICIT ``order``-step (ADC-423 / ADC-431).

    Two lowerings share this entry point, selected by whether an implicit @p linear_source is named:

      - **implicit FLUX** (the default, ADC-431): ``F(U^{n+1}) = 0`` for the coupled nonlinear system
        ``U - U^n - dt*rhs(U)`` (BDF1) / ``U - (4/3)U^n + (1/3)U^{n-1} - (2/3)dt*rhs(U)`` (BDF2) is
        solved by a matrix-free Newton-Krylov iteration -- ``rhs(U) = -div F [+ sources]`` couples the
        cells through the flux stencil, so the Jacobian ``J = I - c*dt*d(rhs)/dU`` is GLOBAL and applied
        matrix-free by a finite-difference Jacobian-vector product (`P.rhs_jacvec`); each Newton step
        solves ``J dU = -F`` with GMRES (J nonsymmetric). The outer Newton loop is a fixed author-time
        unroll of EXACTLY @p newton_max iterations: there is no runtime Newton convergence test and no
        Newton tolerance parameter. The final ``||F||`` is recorded as ``"<block>.bdf_residual"`` so
        callers can assess convergence honestly. This is
        a pure-macro composition of existing primitives (matrix_free_operator + solve_linear + the affine
        algebra + history) -- no new C++ runtime stepper.

      - **cell-local linear SOURCE** (the fast path, ADC-423): when @p linear_source is the typed
        handle of a model ``m.linear_source`` ``L``, the BDF system is block-diagonal and
        ``(c0*I - dt*L) U^{n+1} = rhs``
        is solved per cell by `P.solve_local_linear` (no Newton / Krylov). @p flux / @p sources then add
        an EXPLICIT flux/source RHS lagged at U^n (as in an IMEX explicit partition).

    @p order is 1 (backward Euler) or 2 (BDF2, over the System history ring with a BDF1 cold start).
    @p ncomp is the block component count for the implicit-flux path (1 for a scalar model such as
    inviscid Burgers / linear advection; pass the model's n_cons for a multi-component block).
    @p newton_max is the positive, fixed number of author-time-unrolled Newton updates (not a maximum
    guarded by a hidden tolerance); @p krylov_tol / @p krylov_max / @p krylov_restart configure each
    GMRES inner solve; @p eps is the relative finite-difference step of the Jacobian-vector product.
    The implicit-flux Jacobian currently linearizes only the default flux with either its default source
    (``sources=None`` / ``["default"]``) or no source (``sources=[]``); named source terms are rejected
    until the matrix-free apply can emit their perturbed-state kernels."""
    temporal = _time_state(P, block, state)
    if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2):
        raise ValueError("bdf: order must be the int 1 or 2 (got %r)" % (order,))
    if linear_source is not None:
        # ADC-532: linear_source is a typed OperatorHandle (m.linear_source / m.local_linear_map),
        # not a name string. _bdf_local_linear passes it to P.linear_source, which accepts a handle.
        linear_source = _operator_handle(linear_source, "linear_source")
        return _bdf_local_linear(P, temporal, order, linear_source, sources, flux)
    # The implicit-flux Newton-Krylov path (ADC-431): a flux-less BDF with no implicit term is a no-op.
    if not flux:
        raise ValueError(
            "bdf with flux=False needs a cell-local implicit linear_source (there is no implicit term to "
            "solve); pass linear_source=<m.linear_source handle> for the relaxation BDF, or flux=True "
            "for the implicit-flux Newton-Krylov BDF")
    implicit_sources = _source_names(P, temporal.n, sources)
    named_sources = [source for source in implicit_sources if source != "default"]
    if named_sources:
        raise NotImplementedError(
            "bdf implicit-flux Jacobian cannot linearize named sources %r yet; use sources=[] for "
            "flux-only, sources=['default'] for the default composite RHS, or a cell-local "
            "linear_source path" % named_sources)
    if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp < 1:
        raise ValueError("bdf: ncomp must be a positive int (the block component count); got %r"
                         % (ncomp,))
    if isinstance(newton_max, bool) or not isinstance(newton_max, int) or newton_max < 1:
        raise ValueError("bdf: newton_max must be a positive int (got %r)" % (newton_max,))
    return _bdf_implicit_flux(P, temporal, order, implicit_sources, flux, ncomp, newton_max,
                              krylov_tol, krylov_max, krylov_restart, eps)
