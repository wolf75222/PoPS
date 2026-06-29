#!/usr/bin/env python3
"""AMR Euler-Poisson-Lorentz example through the final public route."""

import os
import sys
from pathlib import Path

import numpy as np

from pops import System, compile_problem
from pops.codegen import KokkosOpenMP, Production
from pops.fields import PoissonProblem
from pops.math import ddt, div, grad, laplacian, sqrt
from pops.mesh import CartesianMesh
from pops.mesh.amr import (
    AMROutput,
    CheckpointPolicy,
    PatchLayout,
    ProperNesting,
    Refine,
    RegridEvery,
    TagUnion,
)
from pops.mesh.layouts import AMR
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.spatial import FiniteVolume
from pops.physics import Model
from pops.solvers.elliptic import GeometricMG
from pops.time import Program


def _repo_include():
    include = Path(__file__).resolve().parents[2] / "include"
    return str(include) if include.is_dir() else None


def _configure_source_tree_include():
    include = _repo_include()
    if include is not None:
        os.environ["POPS_INCLUDE"] = include
    root = os.environ.get("CONDA_PREFIX") or sys.prefix
    os.environ.setdefault("POPS_KOKKOS_ROOT", root)
    os.environ.setdefault("Kokkos_ROOT", root)
    return include


def build_layout(n=32):
    mesh = CartesianMesh(n=n, L=1.0, periodic=True)
    return AMR(
        base=mesh,
        max_levels=2,
        ratio=2,
        regrid=RegridEvery(4),
        patches=PatchLayout(coarse_max_grid=32),
        refine=TagUnion(
            Refine.on("Density").above(1.02),
            Refine.on("phi").gradient_above(0.25),
        ),
        nesting=ProperNesting(buffer=1),
        checkpoint=CheckpointPolicy(restartable=False),
        output=AMROutput(fields=("rho", "phi"), include_patch_boxes=True),
    )


def build_model():
    m = Model("amr_euler_poisson_lorentz")
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
    poisson = PoissonProblem(
        name="phi",
        unknown=phi,
        equation=(-laplacian(phi) == rho),
        solver=GeometricMG(),
    )
    poisson.validate()
    m.field_problem(
        "phi",
        equation=poisson.equation,
        outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
        solver=GeometricMG(),
    )
    m.solve_field(
        "fields_from_state",
        equation=poisson.equation,
        outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
        solver=GeometricMG(),
    )

    electric = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
    source = m.source(
        "electric",
        on=U,
        value=[0.0 * rho, rho * electric.x, rho * electric.y],
    )
    m.rate("explicit_rate", ddt(U) == -div(flux) + source)
    return m.to_module()


def build_program(module):
    T = Program("amr_euler_poisson_lorentz")
    T.bind_operators(module)
    U = T.state("U", block="plasma", space=module.state_spaces()["U"])
    operators = module.operator_registry()
    fields = T.call(operators.get("fields_from_state"), U.n, name="fields_n")
    rate = T.call(operators.get("explicit_rate"), U.n, fields, name="R_n")
    T.define(U.next, U.n + T.dt * rate)
    fields_next = T.call(operators.get("fields_from_state"), U.next, name="fields_next")
    T.record_scalar("mass", T.sum_component(U.n, 0))
    T.commit(U.next, fields=fields_next)
    T.validate()
    return T


def initial_state(mesh):
    x = (np.arange(mesh.n, dtype=float) + 0.5) / mesh.n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.04 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
    return np.stack([rho, np.zeros_like(rho), np.zeros_like(rho)])


def compile_example(n=32):
    include = _configure_source_tree_include()
    layout = build_layout(n=n)
    module = build_model()
    program = build_program(module)
    compiled = compile_problem(
        model=module,
        program=program,
        layout=layout,
        backend=Production(platform=KokkosOpenMP()),
        include=include,
    )
    compiled.inspect_amr()
    return compiled, layout


def run_once(n=32, cfl=0.1):
    compiled, layout = compile_example(n=n)
    sim = System(layout=layout)
    sim.install(
        compiled,
        instances={
            "plasma": {
                "initial": initial_state(layout.base),
                "spatial": FiniteVolume(
                    riemann=Rusanov(),
                    reconstruction=MUSCL(limiter=Minmod()),
                ),
            }
        },
        solvers={"phi": GeometricMG()},
    )
    sim.step_cfl(cfl)
    return sim.get_state("plasma")


def main():
    compiled, _layout = compile_example()
    print("problem:", compiled.so_path)
    print(compiled.inspect_amr())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
