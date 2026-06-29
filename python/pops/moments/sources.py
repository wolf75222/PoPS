"""Moment-hierarchy source terms and typed source descriptors.

The free functions build symbolic expressions. The descriptor classes choose source routes for
``MomentModel.add_source(...)``; they never run per-cell Python work.
"""
from math import comb

from pops.descriptors import Descriptor

from .model_builder import moment_indices, _pow


def lorentz_sources(M, ex, ey, q_over_m, omega_c):
    """Sources of the moment hierarchy under the Lorentz force (Vlasov), generic in the
    order and INDEPENDENT of the closure (no higher-order moment referenced: the electric
    term LOWERS the order, the magnetic term CONSERVES it):

        S[M_pq] = q_over_m (p ex M_{p-1,q} + q ey M_{p,q-1}) + omega_c (p M_{p-1,q+1} - q M_{p+1,q-1})

    @p M: dict (p, q) -> Expr/value of the transported moments (keys = moment_indices).
    @p ex, ey: electric field (aux Expr or values). @p q_over_m, omega_c: param Expr or
    values. @return list aligned with moment_indices(order). Accepts plain numbers
    everywhere (usable as a numeric oracle)."""
    order = max(p + q for (p, q) in M)
    out = []
    for (p, q) in moment_indices(order):
        expr = None
        if p >= 1:
            t = q_over_m * (float(p) * ex * M[(p - 1, q)])
            expr = t if expr is None else expr + t
            t = omega_c * (float(p) * M[(p - 1, q + 1)])
            expr = expr + t
        if q >= 1:
            t = q_over_m * (float(q) * ey * M[(p, q - 1)])
            expr = t if expr is None else expr + t
            t = omega_c * (-float(q) * M[(p + 1, q - 1)])
            expr = expr + t
        out.append(0.0 if expr is None else expr)
    return out


def maxwellian_moments(M):
    """Raw moments of the LOCAL Maxwellian (Gaussian in velocity) matching the lower moments
    of M: density M00, mean (u, v) = M10/M00, M01/M00, and covariance [[C20, C11], [C11, C02]]
    from the second central moments. The Maxwellian is its own closure, so this is INDEPENDENT
    of the model closure.

    All odd central moments of a Gaussian vanish; the even ones follow Isserlis (Wick):
    C40 = 3 C20^2, C22 = C20 C02 + 2 C11^2, C04 = 3 C02^2, C31 = 3 C20 C11, C13 = 3 C02 C11,
    and every order-3 and order-5 central moment is 0. The Gaussian central moments are
    tabulated up to order 4, so this supports moment hierarchies up to order 4 (6, 10 or 15
    variables); an order-6-and-higher even central moment is not tabulated.

    @p M: dict (p, q) -> Expr/value of the transported moments (keys = moment_indices(order));
       the order is inferred as max(p + q) and must be at most 4. Accepts plain numbers
       (usable as a numeric oracle).
    @return list aligned with moment_indices(order): the equilibrium raw moments M_eq[p, q].
    """
    order = max(p + q for (p, q) in M)
    M00 = M[(0, 0)]
    u = M[(1, 0)] / M00
    v = M[(0, 1)] / M00
    # second central moments of M -> covariance of the matched Gaussian.
    C20 = M[(2, 0)] / M00 - u * u
    C11 = M[(1, 1)] / M00 - u * v
    C02 = M[(0, 2)] / M00 - v * v
    # Gaussian central moments up to order 4 (Isserlis); everything else (odd, incl. order 5) = 0.
    cg = {(0, 0): 1.0, (1, 0): 0.0, (0, 1): 0.0,
          (2, 0): C20, (1, 1): C11, (0, 2): C02,
          (3, 0): 0.0, (2, 1): 0.0, (1, 2): 0.0, (0, 3): 0.0,
          (4, 0): 3.0 * C20 * C20, (3, 1): 3.0 * C20 * C11,
          (2, 2): C20 * C02 + 2.0 * C11 * C11,
          (1, 3): 3.0 * C02 * C11, (0, 4): 3.0 * C02 * C02}
    out = []
    for (p, q) in moment_indices(order):
        # de-standardization / reconstruction: M_eq[p, q] = M00 * sum_ij C(p,i) C(q,j)
        # u^(p-i) v^(q-j) Cg(i, j); a numeric-zero Cg term drops out of the generated flux.
        acc = None
        for i in range(p + 1):
            for j in range(q + 1):
                cij = cg.get((i, j), 0.0)
                if isinstance(cij, (int, float)) and cij == 0.0:
                    continue
                t = float(comb(p, i) * comb(q, j)) * _pow(u, p - i) * _pow(v, q - j)
                if not (isinstance(cij, float) and cij == 1.0):
                    t = t * cij
                acc = t if acc is None else acc + t
        out.append(M00 * acc)
    return out


def bgk_source(M, nu):
    """BGK relaxation source S[M_pq] = nu (M_eq[p, q] - M[p, q]) toward the local Maxwellian.

    @p M: dict (p, q) -> Expr/value of the transported (conservative) moments.
    @p nu: collision frequency (Expr or value).
    @return list aligned with moment_indices(order). The collisional invariants M00, M10, M01
       are exact equilibria (M_eq == M there), so those rows are identically 0 (no term emitted)
       and mass and momentum are conserved by construction. Accepts plain numbers everywhere
       (usable as a numeric oracle).
    """
    meq = maxwellian_moments(M)
    out = []
    for k, (p, q) in enumerate(moment_indices(max(p + q for (p, q) in M))):
        if (p, q) in ((0, 0), (1, 0), (0, 1)):
            out.append(0.0)  # collisional invariant: M_eq == M, exact, no term emitted.
        else:
            out.append(nu * (meq[k] - M[(p, q)]))
    return out


class MomentSource(Descriptor):
    """Typed source contribution for :class:`pops.moments.MomentModel`."""

    category = "moment_source"

    def __init__(self, name, rule, *, options=None, capabilities=None):
        if not name:
            raise ValueError("MomentSource requires a non-empty name")
        if not callable(rule):
            raise TypeError("MomentSource(%r): rule must be callable" % (name,))
        self.source_name = str(name)
        self._rule = rule
        self._options = dict(options or {})
        self._capabilities = dict(capabilities or {})

    @classmethod
    def from_rule(cls, name, rule):
        """Build a custom symbolic source rule.

        ``rule(m, M)`` is invoked while creating the model IR, never during runtime stepping.
        """
        return cls(name, rule, capabilities={"provides": "custom_moment_source"})

    def options(self):
        return dict(self._options)

    def capabilities(self):
        return dict(self._capabilities)

    def apply(self, m, M):
        return self._rule(m, M)

    def as_sources(self):
        return self.apply

    def __repr__(self):
        return "%s(name=%r)" % (type(self).__name__, self.source_name)


class VlasovElectricSource(MomentSource):
    """Vlasov electric source descriptor."""

    def __init__(self, electric_field=("grad_x", "grad_y"), charge_over_mass="q_over_m"):
        ex, ey = electric_field
        self.electric_field = (str(ex), str(ey))
        self.charge_over_mass = str(charge_over_mass)
        super().__init__(
            "vlasov_electric",
            self._apply,
            options={
                "electric_field": self.electric_field,
                "charge_over_mass": self.charge_over_mass,
            },
            capabilities={"provides": "vlasov_electric"},
        )

    def _apply(self, m, M):
        ex_name, ey_name = self.electric_field
        qom = m.param(self.charge_over_mass, 1.0)
        return lorentz_sources(M, m.aux(ex_name), m.aux(ey_name), qom, 0.0)


class MagneticRotationSource(MomentSource):
    """Magnetic rotation source descriptor."""

    def __init__(self, omega_c="omega_c", axis="z"):
        if axis != "z":
            raise ValueError("MagneticRotationSource currently supports axis='z' only")
        self.omega_c = str(omega_c)
        self.axis = axis
        super().__init__(
            "magnetic_rotation",
            self._apply,
            options={"omega_c": self.omega_c, "axis": self.axis},
            capabilities={"provides": "magnetic_rotation"},
        )

    def _apply(self, m, M):
        return lorentz_sources(M, 0.0, 0.0, 1.0, m.param(self.omega_c, 1.0))


class MagneticMomentSource(MomentSource):
    """Magnetic Lorentz source bound to a runtime ``q_over_m`` parameter and ``B_z`` aux field."""

    def __init__(self, q_over_m="q_over_m", b_field="B_z"):
        self.q_over_m = str(q_over_m)
        self.b_field = str(b_field)
        super().__init__(
            "magnetic_moment",
            self._apply,
            options={"q_over_m": self.q_over_m, "b_field": self.b_field},
            capabilities={"provides": "magnetic_lorentz"},
        )

    def _apply(self, m, M):
        qom = m.param(self.q_over_m, 1.0)
        b_z = m.aux(self.b_field)
        return lorentz_sources(M, 0.0, 0.0, qom, qom * b_z)

    def __repr__(self):
        return "MagneticMomentSource(q_over_m=%r, b_field=%r)" % (self.q_over_m, self.b_field)


class VlasovSources:
    """Namespace of low-level symbolic source builders."""

    @staticmethod
    def lorentz(M, ex, ey, q_over_m, omega_c):
        return lorentz_sources(M, ex, ey, q_over_m, omega_c)

    @staticmethod
    def maxwellian_eq(M):
        return maxwellian_moments(M)

    @staticmethod
    def bgk(M, nu):
        return bgk_source(M, nu)


__all__ = [
    "MomentSource",
    "VlasovElectricSource",
    "MagneticRotationSource",
    "MagneticMomentSource",
    "VlasovSources",
    "lorentz_sources",
    "maxwellian_moments",
    "bgk_source",
]
