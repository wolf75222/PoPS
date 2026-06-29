"""Provided fluid model builders.

These builders are pure compositions of :class:`pops.physics.Model`. They return
board-authored models ready for ``to_module()`` / ``compile_problem`` and perform no
runtime numerical work in Python.
"""

from pops.math import ddt, div, sqrt
from pops.physics import Model


_FLUID_ROLES_3 = {
    "rho": "density",
    "mx": "momentum_x",
    "my": "momentum_y",
}

_FLUID_ROLES_4 = {
    **_FLUID_ROLES_3,
    "E": "energy",
}


class Isothermal:
    """Provided two-dimensional isothermal Euler model."""

    @staticmethod
    def model(name="isothermal_euler", *, cs2=1.0, parameter_name="cs2"):
        """Return a 3-variable isothermal Euler model.

        The state is ``(rho, mx, my)``. The pressure is ``p = cs2 * rho`` and the
        declared wave speeds are ``u +/- sqrt(cs2)`` and ``v +/- sqrt(cs2)``.
        """
        m = Model(name)
        U = m.state("U", components=["rho", "mx", "my"], roles=_FLUID_ROLES_3)
        rho, mx, my = U
        u = m.primitive("u", mx / rho)
        v = m.primitive("v", my / rho)
        cs2_param = m.param(parameter_name, cs2)
        p = m.scalar("p", cs2_param * rho)
        c = m.scalar("c", sqrt(cs2_param))
        flux = m.flux(
            "F",
            on=U,
            x=[mx, mx * u + p, mx * v],
            y=[my, my * u, my * v + p],
            waves={"x": [u - c, u, u + c], "y": [v - c, v, v + c]},
        )
        m.rate("explicit_rate", ddt(U) == -div(flux))
        m.module.capabilities(moment_model=False, fluid_model=True, equation="isothermal_euler")
        return m

    safe_default = model


class Euler:
    """Provided two-dimensional compressible Euler model."""

    @staticmethod
    def model(name="euler", *, gamma=1.4, parameter_name="gamma"):
        """Return a 4-variable perfect-gas Euler model.

        The state is ``(rho, mx, my, E)``. The pressure is
        ``p = (gamma - 1) * (E - 0.5 * (mx^2 + my^2) / rho)`` and the declared
        wave speeds are the standard acoustic bounds in each Cartesian direction.
        """
        m = Model(name)
        U = m.state("U", components=["rho", "mx", "my", "E"], roles=_FLUID_ROLES_4)
        rho, mx, my, E = U
        u = m.primitive("u", mx / rho)
        v = m.primitive("v", my / rho)
        gamma_param = m.param(parameter_name, gamma)
        kinetic = 0.5 * (mx * mx + my * my) / rho
        p = m.scalar("p", (gamma_param - 1.0) * (E - kinetic))
        c = m.scalar("c", sqrt(gamma_param * p / rho))
        flux = m.flux(
            "F",
            on=U,
            x=[mx, mx * u + p, mx * v, (E + p) * u],
            y=[my, mx * v, my * v + p, (E + p) * v],
            waves={"x": [u - c, u, u, u + c], "y": [v - c, v, v, v + c]},
        )
        m.rate("explicit_rate", ddt(U) == -div(flux))
        m.module.capabilities(moment_model=False, fluid_model=True, equation="euler")
        return m

    safe_default = model


__all__ = ["Euler", "Isothermal"]
