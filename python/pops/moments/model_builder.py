"""Generic moment-model builder: index helpers and build_moment_model.

Symbols are re-exported via the central :mod:`pops.moments` package.

The symbolic IR primitives (``Const`` / ``sqrt`` / ``abs_``) come from
:mod:`pops.ir` at module scope.
The formula carrier is imported lazily inside :func:`build_moment_model`: it is the public
``pops.physics.Model`` board authoring facade, which lowers directly to ``pops.model.Module``.
"""
from math import comb

from pops.ir.expr import Const as _Const
from pops.ir.ops import sqrt as _sqrt, abs_ as _abs_


def moment_indices(order):
    """Canonical list of (p, q) with p + q <= order: q outer, p inner, increasing."""
    if order < 1:
        raise ValueError("moments: order >= 1 required (order %r)" % (order,))
    return [(p, q) for q in range(order + 1) for p in range(order + 1 - q)]


def moment_names(order):
    """Canonical names 'M{p}{q}' aligned with moment_indices(order)."""
    return ["M%d%d" % pq for pq in moment_indices(order)]


def _pow(e, k):
    """e**k by repeated multiplication (k >= 0; e a DSL Expr or a number)."""
    if k == 0:
        return 1.0
    r = e
    for _ in range(k - 1):
        r = r * e
    return r


def _is_zero(e):
    # NUMERIC zero (int/float) or SYMBOLIC zero (ir.Const(0.0)): a closure may return
    # either one; both drop the term from the generated flux (dead primitive not emitted).
    if isinstance(e, (int, float)):
        return float(e) == 0.0
    return isinstance(e, _Const) and e.value == 0.0


def build_moment_model(name, order, closure, blocks=None, exact_speeds=True,
                       robust=False, eps_m00=1e-12, eps_cov=1e-12, sources=None, roe=False):
    """2D moment model with an arbitrary closure: flux and intermediates GENERATED.

    @p order: max order of the transported moments (order=2 -> 6 variables, order=4 -> 15).
    @p closure: callable S -> dict 'S{p}{q}' of the standardized moments of order order+1
       (ALL keys p+q = order+1 required; values DSL Expr or numbers -- a numeric zero
       removes the term from the generated flux). S holds the let-bound standardized moments
       for 2 <= p+q <= order, with S20 = S02 = 1.0 exact (standardization identities).
    @p blocks: block structure of the Jacobian for the eigenvalue solve (pass-through to
       m.wave_speeds_from_jacobian; default full matrix). Ignored if exact_speeds=False.
    @p exact_speeds: True = declare the closed order-2 Gaussian characteristic-speed bound
       carried by the Module. Higher-order exact-speed generation is deliberately not exposed
       through this generic builder until a typed exact-speed descriptor can declare the full
       realizability/closure contract.
    @p robust: True = smooth floors max(x, eps) = ((x+eps)+|x-eps|)/2 on M00 (division) and
       C20/C02 (sqrt) -- differentiable (diff(Abs)), so compatible with exact_speeds. False =
       the bare path, faithful to the guard-free references (may produce NaN on a degenerate
       state).
    @p sources: callable (m, M) -> list of Expr (aligned with moment_indices), wired through
       m.source; M = dict (p, q) -> conservative variable. See lorentz_sources.
    @p roe: currently reserved for a typed moment Roe descriptor. Passing True raises before
       codegen instead of exposing a half-routed path.
    @return public ``pops.physics.Model`` ready for ``to_module()`` / ``compile_problem``."""
    from pops import math as _math
    from pops.physics import Model as _PhysicsModel
    if order < 2:
        raise ValueError("build_moment_model: order >= 2 required (standardization relies "
                         "on C20/C02; order %r)" % (order,))
    if blocks is not None:
        raise ValueError(
            "build_moment_model(blocks=...): block-Jacobian exact speeds are not part of the "
            "Module-native moments builder; provide a typed wave-speed descriptor instead")
    if exact_speeds and order != 2:
        raise ValueError(
            "CartesianVelocityMoments(order=%d, exact_speeds=True) is not supported by the "
            "Module-native generic builder. Use exact_speeds=False or a provided model with a "
            "typed wave-speed descriptor." % order)
    if roe:
        raise ValueError(
            "CartesianVelocityMoments(roe=True) requires a typed moment Roe descriptor; the "
            "generic builder does not expose a partial Roe route")
    idx = moment_indices(order)
    m = _PhysicsModel(name)
    roles = {}
    names = moment_names(order)
    if "M00" in names:
        roles["M00"] = "density"
    if "M10" in names:
        roles["M10"] = "momentum_x"
    if "M01" in names:
        roles["M01"] = "momentum_y"
    state = m.state("U", components=names, roles=roles)
    cons = tuple(state)
    M = dict(zip(idx, cons))

    def floor(nm, x, eps):
        # max(x, eps) = ((x + eps) + |x - eps|) / 2: smooth floor, expressible in the AST.
        return m.primitive(nm, ((x + eps) + _abs_(x - eps)) / 2.0)

    M00 = floor("M00f", M[(0, 0)], eps_m00) if robust else M[(0, 0)]
    u = m.primitive("u", M[(1, 0)] / M00)
    v = m.primitive("v", M[(0, 1)] / M00)

    # normalized raw moments m_pq = M_pq / M00 (no let: each used once)
    mn = {pq: (1.0 if pq == (0, 0) else M[pq] / M00) for pq in idx}

    # --- central moments: binomial transform, derived in a loop ---
    # C_pq = sum_{i<=p, j<=q} comb(p,i) comb(q,j) (-u)^(p-i) (-v)^(q-j) m_ij
    C = {(0, 0): 1.0, (1, 0): 0.0, (0, 1): 0.0}
    for s in range(2, order + 1):
        for q in range(s + 1):
            p = s - q
            expr = None
            for i in range(p + 1):
                for j in range(q + 1):
                    coef = float(comb(p, i) * comb(q, j) * (-1) ** (p - i + q - j))
                    t = coef * _pow(u, p - i) * _pow(v, q - j)
                    if (i, j) != (0, 0):
                        t = t * mn[(i, j)]
                    expr = t if expr is None else expr + t
            C[(p, q)] = m.primitive("C%d%d" % (p, q), expr)

    # --- standardization: S_pq = C_pq / (sx^p sy^q); S20 = S02 = 1 by construction ---
    C20 = floor("C20f", C[(2, 0)], eps_cov) if robust else C[(2, 0)]
    C02 = floor("C02f", C[(0, 2)], eps_cov) if robust else C[(0, 2)]
    sx = m.primitive("sx", _sqrt(C20))
    sy = m.primitive("sy", _sqrt(C02))
    S = {"S20": 1.0, "S02": 1.0}
    for (p, q), c in C.items():
        if p + q >= 2 and (p, q) not in ((2, 0), (0, 2)):
            S["S%d%d" % (p, q)] = m.primitive("S%d%d" % (p, q),
                                              c / (_pow(sx, p) * _pow(sy, q)))

    # --- closure (the ONLY physics) then de-standardization C'_pq = S'_pq sx^p sy^q ---
    top = closure(S)
    want = {"S%d%d" % (p, order + 1 - p) for p in range(order + 2)}
    if set(top) != want:
        raise ValueError("moments: the closure must return exactly the keys %s "
                         "(got %s)" % (sorted(want), sorted(top)))
    Call = dict(C)
    for key, e in top.items():
        p, q = int(key[1]), int(key[2])
        Call[(p, q)] = (0.0 if _is_zero(e)
                        else m.primitive("C%d%d" % (p, q), e * _pow(sx, p) * _pow(sy, q)))

    # --- reconstruction of the order order+1 raw moments: inverse binomial ---
    # m_pq = sum_{i<=p, j<=q} comb(p,i) comb(q,j) u^(p-i) v^(q-j) C_ij
    Mtop = {}
    for q in range(order + 2):
        p = order + 1 - q
        expr = None
        for i in range(p + 1):
            for j in range(q + 1):
                cij = Call.get((i, j))
                if cij is None or _is_zero(cij):
                    continue
                t = float(comb(p, i) * comb(q, j)) * _pow(u, p - i) * _pow(v, q - j)
                if not (isinstance(cij, float) and cij == 1.0):
                    t = t * cij
                expr = t if expr is None else expr + t
        Mtop[(p, q)] = m.primitive("M%d%d" % (p, q), M00 * expr)

    # --- flux: order shift F_x[M_pq] = M_{p+1,q}, F_y[M_pq] = M_{p,q+1} ---
    def raw(pq):
        return M[pq] if pq in M else Mtop[pq]

    waves = None
    if exact_speeds:
        # Closed order-2 Gaussian bound. It is emitted as typed wave metadata on the Module; the
        # runtime still executes the CFL / Riemann logic in C++.
        ax = _sqrt(3.0 * C20)
        ay = _sqrt(3.0 * C02)
        waves = {
            "x": [u - ax, u, u + ax],
            "y": [v - ay, v, v + ay],
        }
    else:
        # Conservative bound used by high-order provided moment models when no exact moment
        # eigenstructure is exposed. It keeps the model installable through the compiled Rusanov/HLL
        # route; the per-cell CFL and flux dissipation still run entirely in C++.
        ax = _sqrt(float(order + 1) * C20)
        ay = _sqrt(float(order + 1) * C02)
        waves = {
            "x": [u - ax, u + ax],
            "y": [v - ay, v + ay],
        }

    flux = m.flux(
        "F",
        on=state,
        x=[raw((p + 1, q)) for (p, q) in idx],
        y=[raw((p, q + 1)) for (p, q) in idx],
        waves=waves,
    )

    source_handle = None
    if sources is not None:
        source_handle = m.source("source_default", on=state, value=sources(m, M))
    rhs = -_math.div(flux)
    if source_handle is not None:
        rhs = rhs + source_handle
    m.rate("explicit_rate", _math.ddt(state) == rhs)
    m.module.capabilities(
        moment_model=True,
        moment_order=order,
        exact_speeds=bool(exact_speeds),
        roe=False,
    )
    return m
