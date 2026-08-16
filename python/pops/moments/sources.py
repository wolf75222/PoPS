"""Moment-hierarchy source terms: Lorentz forcing, Maxwellian matching, BGK collision.

Symbols are re-exported via python/pops/lib/moments/__init__.py.

These functions are pure Python / DSL-expression arithmetic and do NOT require
the DSL compiler; no lazy import is needed here.
"""
from __future__ import annotations

from math import comb
from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet

from .model_builder import moment_indices, _pow


MOMENT_VELOCITY_DIMENSION = 2


def _two_velocity_components(values: Any, *, where: str) -> tuple[Any, Any]:
    if isinstance(values, (str, bytes)):
        raise TypeError("%s must be an ordered two-component vector" % where)
    try:
        components = tuple(values)
    except TypeError as exc:
        raise TypeError("%s must be an ordered two-component vector" % where) from exc
    if len(components) != MOMENT_VELOCITY_DIMENSION:
        raise ValueError(
            "%s requires exactly %d components; got %d"
            % (where, MOMENT_VELOCITY_DIMENSION, len(components)))
    return components


def lorentz_sources(
    M: Any,
    *,
    electric_components: Any,
    q_over_m: Any,
    magnetic_rotation: Any,
) -> list:
    """Return the explicitly two-velocity-dimensional Vlasov-Lorentz hierarchy source.

    The ``(p, q)`` moment basis is a 2V physical specialization, not a spatial-Dimension
    fallback. ``electric_components`` therefore has exactly two ordered velocity components and
    ``magnetic_rotation`` is the signed scalar cyclotron frequency about the unique axial
    direction. The source is generic in moment order and independent of the closure (no
    higher-order moment is referenced: the electric term lowers the order and the magnetic term
    conserves it):

        S[M_pq] = q_over_m (p ex M_{p-1,q} + q ey M_{p,q-1}) + omega_c (p M_{p-1,q+1} - q M_{p+1,q-1})

    @p M: dict (p, q) -> Expr/value of the transported moments (keys = moment_indices).
    @p electric_components: exact ordered 2V electric vector. @p q_over_m: charge-to-mass
    coefficient. @p magnetic_rotation: signed axial cyclotron frequency. @return list aligned
    with moment_indices(order). Accepts plain numbers everywhere (usable as a numeric oracle).
    """
    ex, ey = _two_velocity_components(
        electric_components, where="lorentz_sources.electric_components")
    omega_c = magnetic_rotation
    order = max(p + q for (p, q) in M)
    out = []
    for (p, q) in moment_indices(order):
        expr: Any = None
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


def maxwellian_moments(M: Any) -> list:
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
        acc: Any = None
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


def bgk_source(M: Any, nu: Any) -> list:
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


# --- thin facades over the free functions ----------------------------------
# These carry no math: they forward to the closure-free free functions above so the
# arithmetic stays in one place and lowers to the same generated flux.
class VlasovSources:
    """Namespace of moment-hierarchy source builders (pure forwarders).

    Every method forwards to the corresponding closure-free free function in this module;
    no arithmetic lives here. ``M`` is the dict ``(p, q) -> conservative-moment Expr``.
    """

    @staticmethod
    def lorentz(
        M: Any,
        *,
        electric_components: Any,
        q_over_m: Any,
        magnetic_rotation: Any,
    ) -> list:
        """The Vlasov-Lorentz hierarchy source (forwards to :func:`lorentz_sources`)."""
        return lorentz_sources(
            M,
            electric_components=electric_components,
            q_over_m=q_over_m,
            magnetic_rotation=magnetic_rotation,
        )

    @staticmethod
    def maxwellian_eq(M: Any) -> list:
        """The local-Maxwellian raw moments (forwards to :func:`maxwellian_moments`)."""
        return maxwellian_moments(M)

    @staticmethod
    def bgk(M: Any, nu: Any) -> list:
        """The BGK relaxation source (forwards to :func:`bgk_source`)."""
        return bgk_source(M, nu)


class MagneticMomentSource(Descriptor):
    """Descriptor of a pure-magnetic source for the explicit 2V moment specialization.

    It chooses the route binding one named axial provider component and one ``q_over_m``
    parameter. The axial component is required explicitly: there is no reserved magnetic name or
    silent planar default. The descriptor is a typed
    :class:`pops.descriptors.Descriptor` (Spec 5 sec.6): it declares its options /
    capabilities and is inspectable. :meth:`as_sources` returns a sources callable
    forwarding to the magnetic branch of :func:`lorentz_sources` (zero electric field).
    It carries no arithmetic.
    """

    category = "moment_source"

    def __init__(self, *, axial_component: Any, q_over_m: Any = "q_over_m") -> None:
        for value, where in ((axial_component, "axial_component"),
                             (q_over_m, "q_over_m")):
            if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
                raise TypeError("MagneticMomentSource %s must be a non-empty exact name" % where)
        self.q_over_m = q_over_m
        self.axial_component = axial_component

    def options(self) -> dict:
        return {
            "q_over_m": self.q_over_m,
            "axial_component": self.axial_component,
            "velocity_dimension": MOMENT_VELOCITY_DIMENSION,
        }

    def capabilities(self) -> Any:
        return CapabilitySet({
            "provides": "magnetic_lorentz",
            "velocity_dimension": MOMENT_VELOCITY_DIMENSION,
            "axial_components": 1,
        })

    def as_sources(self, q_over_m_value: Any = 1.0) -> Any:
        """A ``(m, M) -> list`` sources callable: ``omega_c = q_over_m * B`` (electric field 0).

        @p q_over_m_value: the default value of the declared ``q_over_m`` param.
        """
        qom_name, axial_name = self.q_over_m, self.axial_component

        def sources(m: Any, M: Any) -> Any:
            from pops.params import ConstParam

            qom_handle = m.param(ConstParam(qom_name, value=q_over_m_value))
            qom = m.value(qom_handle)
            axial = m.aux(axial_name)
            omega_c = qom * axial
            return lorentz_sources(
                M,
                electric_components=(0.0, 0.0),
                q_over_m=qom,
                magnetic_rotation=omega_c,
            )

        return sources

    def __repr__(self) -> str:
        return "MagneticMomentSource(q_over_m=%r, axial_component=%r)" % (
            self.q_over_m, self.axial_component)
