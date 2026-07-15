"""Generic moment-model builder: index helpers and build_moment_model.

Symbols are re-exported via python/pops/lib/moments/__init__.py.

The symbolic IR primitives (``Const`` / ``sqrt`` / ``abs_``) come from
:mod:`pops._ir` at module scope (the IR is lightweight and lib may import it).
The public blackboard model is imported LAZILY inside
:func:`build_moment_model` so the symbolic helpers remain importable without
constructing the compiler-facing model layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

from pops._ir.expr import Const as _Const
from pops._ir.ops import sqrt as _sqrt, abs_ as _abs_


def moment_indices(order: Any) -> list:
    """Canonical list of (p, q) with p + q <= order: q outer, p inner, increasing."""
    if order < 1:
        raise ValueError("moments: order >= 1 required (order %r)" % (order,))
    return [(p, q) for q in range(order + 1) for p in range(order + 1 - q)]


def moment_names(order: Any) -> list:
    """Canonical names 'M{p}{q}' aligned with moment_indices(order)."""
    return ["M%d%d" % pq for pq in moment_indices(order)]


def _pow(e: Any, k: Any) -> Any:
    """e**k by repeated multiplication (k >= 0; e a DSL Expr or a number)."""
    if k == 0:
        return 1.0
    r = e
    for _ in range(k - 1):
        r = r * e
    return r


def _is_zero(e: Any) -> bool:
    # NUMERIC zero (int/float) or SYMBOLIC zero (ir.Const(0.0)): a closure may return
    # either one; both drop the term from the generated flux (dead primitive not emitted).
    if isinstance(e, (int, float)):
        return float(e) == 0.0
    return isinstance(e, _Const) and e.value == 0.0


@dataclass(frozen=True)
class MomentFluxExpressions:
    """Generic symbolic result of closing one Cartesian moment hierarchy."""

    moments: dict[tuple[int, int], Any]
    x: tuple[Any, ...]
    y: tuple[Any, ...]


def moment_flux_expressions(author: Any, variables: Any, order: Any, closure: Any, *,
                            robust: bool = False, eps_m00: Any = 1e-12,
                            eps_cov: Any = 1e-12) -> MomentFluxExpressions:
    """Build closure-local algebra independently of the surrounding Model facade.

    ``author`` only needs ``primitive(name, expression)``.  This deliberately small protocol
    lets the host expression evaluator and the blackboard ``physics.Model`` share the
    exact M -> C -> S -> closure -> flux construction without model-name dispatch.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError("moment_flux_expressions: order must be an int >= 2")
    idx = moment_indices(order)
    supplied = tuple(variables)
    if len(supplied) != len(idx):
        raise ValueError(
            "moment_flux_expressions: order %d requires %d variables, got %d"
            % (order, len(idx), len(supplied))
        )
    M = dict(zip(idx, supplied, strict=True))

    def floor(nm: Any, x: Any, eps: Any) -> Any:
        return author.primitive(nm, ((x + eps) + _abs_(x - eps)) / 2.0)

    M00 = floor("M00f", M[(0, 0)], eps_m00) if robust else M[(0, 0)]
    u = author.primitive("u", M[(1, 0)] / M00)
    v = author.primitive("v", M[(0, 1)] / M00)
    mn = {pq: (1.0 if pq == (0, 0) else M[pq] / M00) for pq in idx}

    C = {(0, 0): 1.0, (1, 0): 0.0, (0, 1): 0.0}
    for degree in range(2, order + 1):
        for q in range(degree + 1):
            p = degree - q
            expr: Any = None
            for i in range(p + 1):
                for j in range(q + 1):
                    coef = float(comb(p, i) * comb(q, j) * (-1) ** (p - i + q - j))
                    term = coef * _pow(u, p - i) * _pow(v, q - j)
                    if (i, j) != (0, 0):
                        term = term * mn[(i, j)]
                    expr = term if expr is None else expr + term
            C[(p, q)] = author.primitive("C%d%d" % (p, q), expr)

    C20 = floor("C20f", C[(2, 0)], eps_cov) if robust else C[(2, 0)]
    C02 = floor("C02f", C[(0, 2)], eps_cov) if robust else C[(0, 2)]
    sx = author.primitive("sx", _sqrt(C20))
    sy = author.primitive("sy", _sqrt(C02))
    standardized = {"S20": 1.0, "S02": 1.0}
    for (p, q), central in C.items():
        if p + q >= 2 and (p, q) not in ((2, 0), (0, 2)):
            standardized["S%d%d" % (p, q)] = author.primitive(
                "S%d%d" % (p, q), central / (_pow(sx, p) * _pow(sy, q)))

    from .closures.protocol import apply_local_closure

    top = apply_local_closure(closure, order, standardized)
    closed = dict(C)
    for key, expression in top.items():
        p, q = int(key[1]), int(key[2])
        closed[(p, q)] = (
            0.0 if _is_zero(expression) else author.primitive(
                "C%d%d" % (p, q), expression * _pow(sx, p) * _pow(sy, q))
        )

    highest = {}
    for q in range(order + 2):
        p = order + 1 - q
        expr = None
        for i in range(p + 1):
            for j in range(q + 1):
                central = closed.get((i, j))
                if central is None or _is_zero(central):
                    continue
                term = float(comb(p, i) * comb(q, j)) * _pow(u, p - i) * _pow(v, q - j)
                if not (isinstance(central, float) and central == 1.0):
                    term = term * central
                expr = term if expr is None else expr + term
        highest[(p, q)] = author.primitive("M%d%d" % (p, q), M00 * expr)

    def raw(index: tuple[int, int]) -> Any:
        return M[index] if index in M else highest[index]

    return MomentFluxExpressions(
        M,
        tuple(raw((p + 1, q)) for p, q in idx),
        tuple(raw((p, q + 1)) for p, q in idx),
    )


def build_moment_model(name: Any, order: Any, closure: Any, blocks: Any = None,
                       exact_speeds: bool = True, robust: bool = False, eps_m00: Any = 1e-12,
                       eps_cov: Any = 1e-12, sources: Any = None, roe: bool = False,
                       frame: Any = None) -> Any:
    """2D moment model with an arbitrary closure: flux and intermediates GENERATED.

    @p order: max order of the transported moments (order=2 -> 6 variables, order=4 -> 15).
    @p closure: callable S -> dict 'S{p}{q}' of the standardized moments of order order+1
       (ALL keys p+q = order+1 required; values DSL Expr or numbers -- a numeric zero
       removes the term from the generated flux). S holds the let-bound standardized moments
       for 2 <= p+q <= order, with S20 = S02 = 1.0 exact (standardization identities).
    @p blocks: block structure of the Jacobian for the eigenvalue solve (pass-through to
       m.wave_speeds_from_jacobian; default full matrix). Ignored if exact_speeds=False.
    @p exact_speeds: True = exact wave speeds by autodiff of the flux + per-cell numeric
       eigenvalues (faithful riemann='hll'). False = the caller sets m.eigenvalues /
       m.wave_speeds itself (e.g. a bring-up bound).
    @p robust: True = smooth floors max(x, eps) = ((x+eps)+|x-eps|)/2 on M00 (division) and
       C20/C02 (sqrt) -- differentiable (diff(Abs)), so compatible with exact_speeds. False =
       the bare path, faithful to the guard-free references (may produce NaN on a degenerate
       state).
    @p sources: callable (m, M) -> list of Expr (aligned with moment_indices), wired through
       m.source; M = dict (p, q) -> conservative variable. See lorentz_sources.
    @p roe: True = also emit the generic Roe dissipation (m.roe_from_jacobian): the FULL flux
       Jacobian at the arithmetic-mean interface state is eigendecomposed (|A| via the matrix-sign
       kernel pops::roe_abs_apply, spectral-radius Rusanov fallback), making riemann='roe' available
       for the moment system (no fluid roles / pressure needed). Additive to exact_speeds (which
       still provides max_wave_speed for the CFL dt). Emitted by the production backend.
    @p frame: a typed Cartesian frame exposing the ``x`` and ``y`` axes; ``None`` selects
       :class:`Cartesian2D`.
    @return the canonical :class:`pops.physics.Model`, ready to attach to a Problem."""
    from pops.frames import Cartesian2D
    from pops.math import ddt, div
    from pops.physics import Density, Model

    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError("build_moment_model: order must be an int >= 2 "
                         "(standardization relies on C20/C02; order %r)" % (order,))
    selected_frame = Cartesian2D() if frame is None else frame
    axes = getattr(selected_frame, "axes", None)
    if (not isinstance(axes, tuple) or len(axes) != 2
            or tuple(getattr(axis, "name", None) for axis in axes) != ("x", "y")):
        raise TypeError("build_moment_model frame must expose the typed Cartesian axes x and y")

    m = Model(name, frame=selected_frame)
    state = m.state(
        "U", components=tuple(moment_names(order)), roles={"M00": Density()})
    expressions = moment_flux_expressions(
        m, tuple(state), order, closure,
        robust=robust, eps_m00=eps_m00, eps_cov=eps_cov)
    flux = m.flux(
        "transport",
        frame=selected_frame,
        state=state,
        components={axes[0]: expressions.x, axes[1]: expressions.y},
    )

    if exact_speeds:
        m.wave_speeds_from_jacobian(blocks=blocks)
    if roe:
        m.roe_from_jacobian()
    rhs = -div(flux)
    if sources is not None:
        source = m.source(
            "source", on=state, value=sources(m, expressions.moments))
        rhs = rhs + source
    m.rate("transport", equation=ddt(state) == rhs)
    return m


__all__ = [
    "MomentFluxExpressions", "build_moment_model", "moment_flux_expressions",
    "moment_indices", "moment_names",
]
