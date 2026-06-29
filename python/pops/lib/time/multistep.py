"""pops.lib.time.multistep -- Adams-Bashforth and local-linear BDF schemes.

Exports: adams_bashforth, adams_bashforth2, bdf.
Private helpers: _AB_WEIGHTS, _bdf_local_linear.
"""

from ._helpers import _opcall, _stage_rate
from .euler import forward_euler as _forward_euler_macro


def _forward_euler(P, block, rhs_operator, fields_operator):
    # AB1 degenerates to Forward Euler; keep the implementation single-sourced.
    _forward_euler_macro(P, block, rhs_operator=rhs_operator, fields_operator=fields_operator)


# Adams-Bashforth weights b_j on R_{n-j} (j = 0..order-1), per order (ADC-423). AB1 is Forward Euler.
_AB_WEIGHTS = {
    1: (1.0,),
    2: (1.5, -0.5),                       # 3/2, -1/2
    3: (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0),
}


def adams_bashforth(P, block, order, *, rhs_operator, fields_operator=None):
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

    AB1 uses Forward Euler directly and therefore records no history op. AB2 uses ``"ab2_step"`` as
    its stable step node name."""
    if isinstance(order, bool) or not isinstance(order, int) or order not in _AB_WEIGHTS:
        raise ValueError("adams_bashforth: order must be an int in %s (got %r)"
                         % (sorted(_AB_WEIGHTS), order))
    b = _AB_WEIGHTS[order]
    if order == 1:  # AB1 == Forward Euler: no history, identical IR to forward_euler.
        _forward_euler(P, block, rhs_operator, fields_operator)
        return
    name = block + ".R"
    step_name = "ab2_step" if order == 2 else ("ab%d_step" % order)
    U = P._state_value(block)
    R_n = _stage_rate(P, U, rhs_operator=rhs_operator, fields_operator=fields_operator, tag="ab_0_")
    # Store R_n FIRST (so the first store cold-start-fills the ring), then read R_{n-j} = lag j.
    P.store_history(name, R_n)
    expr = U + (P.dt * b[0]) * R_n
    for j in range(1, order):
        expr = expr + (P.dt * b[j]) * P.history(name, lag=j)
    P.commit(block, P.linear_combine(step_name, expr))


def adams_bashforth2(P, block, *, rhs_operator, fields_operator=None):
    """Adams-Bashforth 2, a named convenience macro for ``adams_bashforth(P, block, 2)``.

    It stores ``R_n`` first, reads ``R_{n-1}`` at lag 1, and applies the classic weights
    ``3/2`` and ``-1/2``."""
    adams_bashforth(P, block, 2, rhs_operator=rhs_operator, fields_operator=fields_operator)


def _bdf_local_linear(P, block, order, *, implicit_operator, rhs_operator=None, fields_operator=None):
    """Cell-local BDF over typed operator handles.

    The BDF system is block-diagonal because ``implicit_operator`` returns a
    ``LocalLinearOperator``. Optional explicit terms are supplied by ``rhs_operator`` and are lagged at
    ``U^n``. The optional ``fields_operator`` is evaluated once from ``U^n`` and reused by both the
    explicit rate and local-linear operator when their signatures require fields.

      - **BDF1** (backward Euler): ``(I - dt*L) U^{n+1} = U^n [+ dt R]``;
      - **BDF2**: ``(I - (2/3) dt L) U^{n+1} = (2/3)(2 U^n - 1/2 U^{n-1}) [+ dt R]`` over the System
        history ring, with a BDF1 cold start (the first store fills every slot -> U^{n-1} = U^n)."""
    U = P._state_value(block)
    fields = _opcall(P, fields_operator, U, value_name="%s_bdf_fields" % block) \
        if fields_operator is not None else None
    lin = _opcall(P, implicit_operator, fields, value_name="%s_bdf_L" % block)
    # Optional explicit rate folded into the BDF right-hand side, lagged at U^n.
    R = None
    if rhs_operator is not None:
        if fields is not None:
            R = _opcall(P, rhs_operator, U, fields, value_name="%s_bdf_R" % block)
        else:
            R = _opcall(P, rhs_operator, U, value_name="%s_bdf_R" % block)

    def _with_explicit(expr):
        return (expr + P.dt * R) if R is not None else expr

    if order == 1:  # (I - dt*L) U^{n+1} = U^n [+ dt R]
        rhs = P.linear_combine(block + "_bdf1_rhs", _with_explicit(1.0 * U))
        operator = P.I - P.dt * lin
        out = P.solve_local_linear(name=block + "_bdf1_step", operator=operator, rhs=rhs, fields=fields)
        P.commit(block, out)
        return out
    # BDF2: (3/2 I - dt*L) U^{n+1} = 2 U^n - 1/2 U^{n-1} [+ dt R], over the history ring.
    name = block + ".U"
    P.store_history(name, U)                       # store U^n first (cold-start fills the ring)
    U_nm1 = P.history(name, lag=1)                 # U^{n-1} (== U^n on step 0 -> BDF1 cold start)
    rhs = P.linear_combine(block + "_bdf2_rhs", _with_explicit(2.0 * U - 0.5 * U_nm1))
    operator = P.I - (P.dt * (2.0 / 3.0)) * lin
    # Divide both sides by 3/2: (I - (2/3) dt L) U^{n+1} = (2/3)(2 U^n - 1/2 U^{n-1} [+ dt R]).
    rhs = P.linear_combine(block + "_bdf2_rhs_scaled", (2.0 / 3.0) * rhs)
    out = P.solve_local_linear(name=block + "_bdf2_step", operator=operator, rhs=rhs, fields=fields)
    P.commit(block, out)
    return out


def bdf(P, block, order, *, implicit_operator, rhs_operator=None, fields_operator=None):
    """Backward Differentiation Formula over typed operator handles.

    This ready-made macro is intentionally local-linear and operator-first:

      - ``implicit_operator`` is a typed handle returning ``LocalLinearOperator(U, U)``;
      - ``rhs_operator`` is an optional typed explicit rate handle, lagged at ``U^n``;
      - ``fields_operator`` is an optional typed field operator handle evaluated from ``U^n``.

    It does not accept legacy string selectors. Non-local implicit transport BDF belongs in a future
    descriptor with a real operator handle / matrix-free apply contract, not in this ready macro.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2):
        raise ValueError("bdf: order must be the int 1 or 2 (got %r)" % (order,))
    return _bdf_local_linear(P, block, order, implicit_operator=implicit_operator,
                             rhs_operator=rhs_operator, fields_operator=fields_operator)
