"""pops.lib.time.imex -- IMEX (implicit-explicit) time-stepping schemes.

Exports: imex_local, imex_local_linear.
"""
from __future__ import annotations

from typing import Any

from ._helpers import _exact_coefficient, _opcall, _operator_handle, program_macro


@program_macro
def imex_local(P: Any, block: Any, *, linear_source: Any, sources: Any = ("default",),
               flux: Any = True, theta: Any = 1) -> Any:
    """IMEX with an EXPLICIT flux/source and an IMPLICIT cell-local linear source (ADC-423).

    One step of a theta-implicit splitting of ``dU/dt = R_explicit(U) + L U`` where ``L`` is a named
    model ``m.linear_source`` (e.g. a Lorentz operator) solved cell by cell:

        R   = R_explicit(U)                                     (P.rhs: -div F + the named sources)
        U^{n+1} = (I - theta*dt*L)^{-1} (U + dt*R)              (P.solve_local_linear)

    The explicit part is assembled with `P.rhs` (flux + the requested named @p sources, on the fields
    solved from U); the implicit part is the local solve of ``(I - theta*dt*L) U^{n+1} = U + dt*R``
    via `P.solve_local_linear`, exactly the predictor half of the codebase's predictor-corrector
    pattern (``test_time_local_solve``). At ``theta == 1`` this is backward Euler on the L term and
    forward Euler on R; ``theta == 0`` would drop the implicit solve (use `forward_euler` instead) and
    is rejected. @p linear_source is the typed handle of the model ``m.linear_source`` /
    ``m.local_linear_map``; @p theta the implicitness of the L term (0 < theta <= 1)."""
    linear_source = _operator_handle(linear_source, "linear_source")
    theta = _exact_coefficient(theta, "imex_local: theta")
    if not (0 < theta <= 1):
        raise ValueError(
            "imex_local: theta must be in (0, 1] (got %r); theta == 0 is fully explicit -- use "
            "forward_euler instead" % (theta,))
    U = P.state(block)
    fields = P.solve_fields(U) if flux else None
    R = P._rhs_legacy(state=U, fields=fields, flux=flux, sources=list(sources))
    rhs = P.linear_combine(block + "_imex_rhs", U + P.dt * R)
    operator = P.I - (theta * P.dt) * P.linear_source(linear_source)
    out = P.solve_local_linear(name=block + "_imex_step", operator=operator, rhs=rhs, fields=fields)
    P._commit_block(block, out)
    return out


@program_macro
def imex_local_linear(P: Any, block: Any, *, explicit_operator: Any, implicit_operator: Any,
                      fields_operator: Any = None, theta: Any = 1,
                      state_space: Any = "U") -> Any:
    """Generic IMEX with an explicit rate and an implicit local linear operator (Spec 2).

    One theta-implicit step of ``dU/dt = R(U[, fields]) + L([fields]) U``::

        U^{n+1} = (I - theta dt L)^{-1} (U^n + dt R)

    composing the typed ``explicit_operator`` and ``implicit_operator`` handles (and an optional
    ``fields_operator`` handle). Each is a :class:`pops.model.OperatorHandle` from an ``m.*``
    declarer, not a name string. Requires ``P.bind_operators(module)``.
    """
    theta = _exact_coefficient(theta, "imex_local_linear: theta")
    if not (0 < theta <= 1):
        raise ValueError("imex_local_linear: theta must be in (0, 1]")
    explicit_operator = _operator_handle(explicit_operator, "explicit_operator")
    implicit_operator = _operator_handle(implicit_operator, "implicit_operator")
    if fields_operator is not None:
        fields_operator = _operator_handle(fields_operator, "fields_operator")
    u = P.state(block)
    fields = _opcall(P, fields_operator, u, value_name="fields") if fields_operator else None
    r = _opcall(P, explicit_operator, u, fields, value_name="R")
    lin = _opcall(P, implicit_operator, fields, value_name="L")
    q = P.linear_combine("imex_rhs", u + P.dt * r)
    u1 = P.solve_local_linear("imex_step", operator=P.I - theta * P.dt * lin, rhs=q, fields=fields)
    P._commit_block(block, u1)
    return u1
