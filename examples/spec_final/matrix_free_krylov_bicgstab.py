#!/usr/bin/env python3
"""Final matrix-free Krylov example.

Public route:

    MatrixFreeOperator + LinearProblem
        -> Program.solve_linear(...)
        -> compile_problem(...)
        -> System.install(...)
        -> sim.step_cfl(...)

The ``@A.apply`` function records Program IR only. The Krylov loop and every stencil apply run
inside the generated C++ problem artifact.
"""

import os
import sys
from pathlib import Path

import numpy as np

from pops import System, compile_problem
from pops.codegen import KokkosOpenMP, Production
from pops.linalg import LinearProblem, MatrixFreeOperator
from pops.math import ddt, div
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.spatial import FiniteVolume
from pops.physics import Model
from pops.solvers.krylov import BiCGStab
from pops.time import Program


def _repo_include():
    include = Path(__file__).resolve().parents[2] / "include"
    return str(include) if include.is_dir() else None


def _configure_source_tree_include():
    root = Path(__file__).resolve().parents[2]
    os.environ["POPS_CACHE_DIR"] = str(root / ".pops_cache")
    include = _repo_include()
    if include is not None:
        os.environ["POPS_INCLUDE"] = include
    if not os.environ.get("POPS_KOKKOS_ROOT"):
        prefix = Path(os.environ.get("CONDA_PREFIX") or sys.prefix)
        if (prefix / "include" / "Kokkos_Core.hpp").is_file():
            os.environ["POPS_KOKKOS_ROOT"] = str(prefix)
    return include


def build_model():
    m = Model("matrix_free_passive_scalar")
    U = m.state("U", components=["rho"], roles={"rho": "density"})
    rho = U[0]
    zero = 0.0 * rho

    flux = m.flux(
        "F",
        on=U,
        x=[zero],
        y=[zero],
        waves={"x": [1.0 + zero], "y": [1.0 + zero]},
    )
    m.rate("transport_rate", ddt(U) == -div(flux))
    m.check()
    return m.to_module()


def build_program():
    T = Program("matrix_free_bicgstab")
    U = T.state("U", block="plasma").n

    A = MatrixFreeOperator("helmholtz")

    @A.apply
    def apply(P, out, x):
        lap = P.scalar_field("lap")
        P.laplacian(lap, x)
        return x - 0.05 * lap

    problem = LinearProblem(operator=A, unknown="phi", rhs=U, name="helmholtz")
    phi = T.solve_linear(problem, method=BiCGStab(tolerance=1e-10, max_iter=120))
    T.commit("plasma", phi)
    T.validate()
    return T


def initial_state(mesh):
    x = (np.arange(mesh.n, dtype=float) + 0.5) / mesh.n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.2 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
    return np.stack([rho])


def run_once(n=16, cfl=0.4):
    include = _configure_source_tree_include()
    mesh = CartesianMesh(n=n, L=1.0, periodic=True)
    layout = Uniform(mesh)
    module = build_model()
    program = build_program()
    spatial = FiniteVolume(reconstruction=FirstOrder(), riemann=Rusanov())

    compiled = compile_problem(
        model=module,
        program=program,
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
    )
    sim.step_cfl(cfl)

    state = sim.get_state("plasma")
    if state.shape != (1, mesh.n, mesh.n):
        raise RuntimeError("unexpected state shape %r" % (state.shape,))
    return {"compiled": compiled, "state": state}


def main():
    result = run_once()
    print("problem:", result["compiled"].so_path)
    print("state:", result["state"].shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
