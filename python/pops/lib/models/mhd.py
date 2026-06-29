"""Provided magnetohydrodynamics model builders."""

from pops.math import ddt, div, sqrt
from pops.physics import Model


_IDEAL_MHD_ROLES = {
    "rho": "density",
    "mx": "momentum_x",
    "my": "momentum_y",
    "E": "energy",
    "Bx": "magnetic_x",
    "By": "magnetic_y",
}


class IdealMHD:
    """Provided two-dimensional ideal-MHD model.

    This is a conservative 2D in-plane MHD system over
    ``(rho, mx, my, E, Bx, By)``. It declares a generic Rusanov/HLL-compatible
    fast-wave upper bound; divergence control remains a numerics/layout concern,
    not Python runtime work.
    """

    @staticmethod
    def model(name="ideal_mhd", *, gamma=1.4, parameter_name="gamma"):
        """Return a 6-variable ideal-MHD model ready for ``to_module()``."""
        m = Model(name)
        U = m.state(
            "U",
            components=["rho", "mx", "my", "E", "Bx", "By"],
            roles=_IDEAL_MHD_ROLES,
        )
        rho, mx, my, E, Bx, By = U
        u = m.primitive("u", mx / rho)
        v = m.primitive("v", my / rho)
        gamma_param = m.param(parameter_name, gamma)
        b2 = m.scalar("B2", Bx * Bx + By * By)
        kinetic = 0.5 * (mx * mx + my * my) / rho
        p = m.scalar("p", (gamma_param - 1.0) * (E - kinetic - 0.5 * b2))
        p_total = m.scalar("p_total", p + 0.5 * b2)
        b_dot_u = Bx * u + By * v
        fast = m.scalar("c_fast", sqrt(gamma_param * p / rho + b2 / rho))

        flux = m.flux(
            "F",
            on=U,
            x=[
                mx,
                mx * u + p_total - Bx * Bx,
                my * u - Bx * By,
                (E + p_total) * u - Bx * b_dot_u,
                0.0 * Bx,
                By * u - Bx * v,
            ],
            y=[
                my,
                mx * v - By * Bx,
                my * v + p_total - By * By,
                (E + p_total) * v - By * b_dot_u,
                Bx * v - By * u,
                0.0 * By,
            ],
            waves={
                "x": [u - fast, u, u, u, u, u + fast],
                "y": [v - fast, v, v, v, v, v + fast],
            },
        )
        m.rate("explicit_rate", ddt(U) == -div(flux))
        m.module.capabilities(moment_model=False, fluid_model=True, mhd_model=True,
                              equation="ideal_mhd")
        return m

    safe_default = model


__all__ = ["IdealMHD"]
