#!/usr/bin/env python3
"""Provided HyQMOM15 moment model through the final compile route."""

import os
import sys
from pathlib import Path

import numpy as np

from pops import System, compile_problem
from pops.codegen import KokkosOpenMP, Production
from pops.lib.models.moments import HyQMOM15
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.spatial import FiniteVolume
from pops.solvers.elliptic import GeometricMG
from pops.time import Program


def _repo_include():
    include = Path(__file__).resolve().parents[2] / "include"
    return str(include) if include.is_dir() else None


def _configure_source_tree_include():
    include = _repo_include()
    if include is not None:
        os.environ["POPS_INCLUDE"] = include
    conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    if conda_prefix:
        os.environ.setdefault("POPS_KOKKOS_ROOT", conda_prefix)
        os.environ.setdefault("Kokkos_ROOT", conda_prefix)
    return include


def build_model():
    physics_model = HyQMOM15.vlasov_poisson_magnetic(
        robust=True,
        exact_speeds=False,
    )
    return physics_model.to_module()


def build_program(module):
    T = Program("hyqmom15_forward_euler")
    T.bind_operators(module)
    U = T.state("U", block="moments", space=module.state_spaces()["U"])
    fields_from_state = module.operator_registry().get("fields_from_state")
    explicit_rate = module.operator_registry().get("explicit_rate")
    fields_n = T.call(fields_from_state, U.n, name="fields_n")
    R_n = T.call(explicit_rate, U.n, fields_n, name="R_n")
    T.define(U.next, U.n + T.dt * R_n)
    T.commit(U.next, fields=fields_n)
    T.validate()
    return T


def initial_state(mesh):
    x = (np.arange(mesh.n, dtype=float) + 0.5) / mesh.n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.01 * np.sin(2.0 * np.pi * X)
    zeros = np.zeros_like(rho)
    state = [rho, zeros, rho * 0.5, zeros, rho * 0.75]
    state += [zeros, zeros, zeros, zeros]
    state += [rho * 0.4, zeros, rho * 0.2]
    state += [zeros, zeros, rho * 0.3]
    return np.stack(state)


def compile_example(n=8):
    include = _configure_source_tree_include()
    mesh = CartesianMesh(n=n, L=1.0, periodic=True)
    layout = Uniform(mesh)
    module = build_model()
    program = build_program(module)
    return compile_problem(
        model=module,
        program=program,
        layout=layout,
        backend=Production(platform=KokkosOpenMP()),
        include=include,
    )


def run_once(n=8, cfl=0.05):
    mesh = CartesianMesh(n=n, L=1.0, periodic=True)
    layout = Uniform(mesh)
    compiled = compile_example(n=n)
    sim = System(layout=layout)
    sim.install(
        compiled,
        instances={
            "moments": {
                "initial": initial_state(mesh),
                "spatial": FiniteVolume(
                    riemann=Rusanov(),
                    reconstruction=MUSCL(limiter=Minmod()),
                ),
            }
        },
        aux={"B_z": np.zeros((mesh.n, mesh.n))},
        solvers={"phi": GeometricMG()},
    )
    sim.step_cfl(cfl)
    return sim.get_state("moments")


def main():
    compiled = compile_example()
    print("problem:", compiled.so_path)
    print("operators:", sorted(compiled.model.list_operators()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
