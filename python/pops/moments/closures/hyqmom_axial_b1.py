"""The explicitly selected order-four axial marginal of HyQMOM Appendix B.1.

This is one five-moment, one-velocity closure. It does not infer a recurrence for
other orders from the incomplete supplied note, and it is not a Gaussian closure.
The returned expressions are ordinary scalar IR consumed by generic Model.flux.
"""
from __future__ import annotations

from pops._ir.ops import sqrt


def hyqmom_b1_axial_flux(raw):
    """Return ``(M1,M2,M3,M4,M5)`` for the B.1 order-four x marginal.

    Requires positive density and variance; the caller must separately enforce
    the realizable moment domain. The fifth standardized moment is exactly
    ``S50 = S30 * (5*S40 - 3*S30**2 - 1) / 2`` from Appendix B.1.
    """
    values = tuple(raw)
    if len(values) != 5:
        raise ValueError("HyQMOM B.1 axial order four requires exactly M0..M4")
    rho, m1, m2, m3, m4 = values
    u = m1 / rho
    c2 = m2 - 2 * u * m1 + u * u * rho
    c3 = m3 - 3 * u * m2 + 3 * u * u * m1 - u * u * u * rho
    c4 = m4 - 4 * u * m3 + 6 * u * u * m2 - 4 * u * u * u * m1 + u * u * u * u * rho
    sigma = sqrt(c2 / rho)
    s30 = c3 / (rho * sigma * sigma * sigma)
    s40 = c4 / (rho * sigma * sigma * sigma * sigma)
    s50 = 0.5 * s30 * (5 * s40 - 3 * s30 * s30 - 1)
    c5 = rho * sigma * sigma * sigma * sigma * sigma * s50
    m5 = c5 + 5 * u * c4 + 10 * u * u * c3 + 10 * u * u * u * c2 + rho * u * u * u * u * u
    return m1, m2, m3, m4, m5


__all__ = ["hyqmom_b1_axial_flux"]
