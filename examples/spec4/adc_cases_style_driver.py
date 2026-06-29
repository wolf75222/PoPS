#!/usr/bin/env python3
"""Spec 4 driver skeleton using the final compiled-problem route.

This is the reusable half an external scenario repository can mirror:

    model -> module
    Program -> compile_problem
    System -> install(compiled, instances=...) -> step_cfl

The script is intentionally fail-loud. Missing toolchain/Kokkos/lowering support should raise
instead of being hidden behind a silent fallback.
"""

import numpy as np

import pops
from pops.codegen import Production
from pops.lib.time import forward_euler
from pops.math import ddt, div, grad, laplacian, sqrt
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
from pops.numerics.spatial import spatial
from pops.physics import Model
from pops.solvers.elliptic import GeometricMG
from pops.time import Program


def build_model():
    """Reusable physics: 2D isothermal Euler with an electrostatic Poisson source."""
    m = Model("euler_poisson")
    state = m.state(
        "U",
        components=["rho", "mx", "my"],
        roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"},
    )
    rho, mx, my = state
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    cs2 = m.param("cs2", 1.0)
    pressure = m.scalar("p", cs2 * rho)
    sound = m.scalar("c", sqrt(cs2))
    flux = m.flux(
        "F",
        on=state,
        x=[mx, mx * u + pressure, mx * v],
        y=[my, mx * v, my * v + pressure],
        waves={"x": [u - sound, u, u + sound], "y": [v - sound, v, v + sound]},
    )
    phi = m.field("phi")
    m.solve_field(
        "fields_from_state",
        equation=(-laplacian(phi) == rho),
        outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
        solver=GeometricMG(),
    )
    e_field = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
    electric = m.source("electric", on=state, value=[0.0 * rho, rho * e_field.x, rho * e_field.y])
    m.rate("explicit_rate", ddt(state) == -div(flux) + electric)
    m.check()
    return m


def build_program(module):
    program = Program("forward_euler_driver").bind_operators(module)
    ops = module.operator_registry()
    forward_euler(
        program,
        "plasma",
        rhs_operator=ops.get("explicit_rate"),
        fields_operator=ops.get("fields_from_state"),
    )
    return program


def initial_condition(n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.05 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
    return np.stack([rho, np.zeros_like(rho), np.zeros_like(rho)])


def main():
    n = 48
    mesh = CartesianMesh(n=n, L=1.0, periodic=True)
    layout = Uniform(mesh)
    model = build_model()
    module = model.to_module()
    program = build_program(module)
    compiled = pops.compile_problem(model=module, time=program, backend=Production(), layout=layout)

    sim = pops.System(n=n, L=1.0, periodic=True)
    sim.install(
        compiled,
        instances={
            "plasma": {
                "model": module,
                "initial": initial_condition(n),
                "spatial": spatial.FiniteVolume(reconstruction=FirstOrder(), riemann=Rusanov()),
            },
        },
        solvers={"phi": GeometricMG()},
    )
    sim.step_cfl(0.4)
    print("OK: t =", sim.time(), "mass =", sim.mass("plasma"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
