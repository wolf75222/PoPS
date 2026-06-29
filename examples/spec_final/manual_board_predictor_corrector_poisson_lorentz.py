#!/usr/bin/env python3
"""Manual operator-first predictor-corrector for Euler-Poisson-Lorentz.

The script exercises the final public route:

    typed physics model + typed Program
        -> compile_problem(...)
        -> System(layout=...)
        -> sim.install(...)
        -> sim.step_cfl(...)

All numerical work runs in generated/native C++.
"""

import os
import sys
from pathlib import Path

import numpy as np

from pops import System, compile_problem
from pops.codegen import KokkosOpenMP, Production
from pops.math import ddt, div, grad, laplacian, sqrt
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.spatial import FiniteVolume
from pops.physics import Model
from pops.runtime import Profile
from pops.solvers.elliptic import GeometricMG
from pops.time import Program


def _repo_include():
    include = Path(__file__).resolve().parents[2] / "include"
    return str(include) if include.is_dir() else None


def _configure_source_tree_include():
    include = _repo_include()
    if include is not None:
        os.environ["POPS_INCLUDE"] = include
    return include


def build_model():
    m = Model("euler_poisson_lorentz")

    U = m.state(
        "U",
        components=["rho", "mx", "my"],
        roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"},
    )
    rho, mx, my = U
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)

    cs2 = m.param("cs2", 1.0)
    pressure = m.scalar("p", cs2 * rho)
    sound = m.scalar("c", sqrt(cs2))

    flux = m.flux(
        "F",
        on=U,
        x=[mx, mx * u + pressure, mx * v],
        y=[my, my * u, my * v + pressure],
        waves={"x": [u - sound, u, u + sound], "y": [v - sound, v, v + sound]},
    )

    phi = m.field("phi")
    m.solve_field(
        "fields_from_state",
        equation=(-laplacian(phi) == rho),
        outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
        solver=GeometricMG(),
    )

    electric_field = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
    electric = m.source(
        "electric",
        on=U,
        value=[0.0 * rho, rho * electric_field.x, rho * electric_field.y],
    )

    bz = m.aux("B_z")
    lorentz = m.local_linear_operator(
        "lorentz",
        on=U,
        matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]],
    )

    m.rate("explicit_rate", ddt(U) == -div(flux) + electric)
    m.operator("implicit_operator", returns=lorentz, inputs=["fields"])
    m.check()
    return m.to_module()


def build_program(module):
    T = Program("manual_predictor_corrector_poisson_lorentz")
    T.bind_operators(module)

    dt = T.dt
    U = T.state("U", block="plasma", space=module.state_spaces()["U"])
    operators = module.operator_registry()
    fields_from_state = operators.get("fields_from_state")
    explicit_rate = operators.get("explicit_rate")
    implicit_operator = operators.get("implicit_operator")

    fields_n = T.call(fields_from_state, U.n, name="fields_n")
    R_n = T.call(explicit_rate, U.n, fields_n, name="R_n")
    L_n = T.call(implicit_operator, fields_n, name="L_n")
    rhs_star = T.define("U_star_rhs", U.n + dt * R_n)
    U_star = T.solve_local_linear(
        name="U_star",
        operator=T.I - dt * L_n,
        rhs=rhs_star,
        fields=fields_n,
    )

    fields_star = T.call(fields_from_state, U_star, name="fields_star")
    R_star = T.call(explicit_rate, U_star, fields_star, name="R_star")
    L_star = T.call(implicit_operator, fields_star, name="L_star")
    rhs_next = T.define("U_next_rhs", U.n + 0.5 * dt * (R_n + R_star))
    U_next_value = T.solve_local_linear(
        name="U_next",
        operator=T.I - 0.5 * dt * L_star,
        rhs=rhs_next,
        fields=fields_star,
    )
    T.define(U.next, U_next_value)

    fields_next = T.call(fields_from_state, U.next, name="fields_next")
    T.record_scalar("mass", T.sum_component(U.n, 0))
    T.record_scalar("rho_min", T.min_component(U.n, 0))
    T.record_scalar("rho_max", T.max_component(U.n, 0))
    T.commit(U.next, fields=fields_next)
    T.validate()
    return T


def initial_state(mesh):
    x = (np.arange(mesh.n, dtype=float) + 0.5) / mesh.n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.05 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
    mx = 0.1 * rho
    my = -0.05 * rho
    return np.stack([rho, mx, my])


def run_once(n=16, cfl=0.2):
    include = _configure_source_tree_include()
    mesh = CartesianMesh(n=n, L=1.0, periodic=True)
    layout = Uniform(mesh)
    module = build_model()
    program = build_program(module)
    spatial = FiniteVolume(
        riemann=Rusanov(),
        reconstruction=MUSCL(limiter=Minmod()),
    )

    compiled = compile_problem(
        model=module,
        time=program,
        layout=layout,
        backend=Production(platform=KokkosOpenMP()),
        include=include,
    )

    sim = System(layout=layout)
    sim.install(
        compiled,
        instances={
            "plasma": {
                "initial": initial_state(mesh),
                "spatial": spatial,
            }
        },
        aux={"B_z": np.full((mesh.n, mesh.n), 0.2)},
        solvers={"phi": GeometricMG()},
    )

    with sim.profile(Profile.Basic()) as prof:
        sim.step_cfl(cfl)

    state = sim.get_state("plasma")
    fields = sim.get_current_fields("plasma")
    diagnostics = sim.get_recorded_scalars()
    summary = prof.summary()

    if state.shape != (3, mesh.n, mesh.n):
        raise RuntimeError("unexpected state shape %r" % (state.shape,))
    for key in ("phi", "grad_x", "grad_y"):
        if key not in fields:
            raise RuntimeError("missing field output %r" % key)
    for key in ("mass", "rho_min", "rho_max"):
        if key not in diagnostics:
            raise RuntimeError("missing diagnostic %r" % key)

    return {
        "compiled": compiled,
        "state": state,
        "fields": fields,
        "diagnostics": diagnostics,
        "profile": summary,
    }


def main():
    result = run_once()
    compiled = result["compiled"]
    state = result["state"]
    diagnostics = result["diagnostics"]

    print("problem:", compiled.so_path)
    print("state:", state.shape)
    print("diagnostics:", diagnostics)
    print(result["profile"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
