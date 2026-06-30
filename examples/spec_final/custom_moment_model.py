#!/usr/bin/env python3
"""Custom moment model through the final operator-first route.

The example uses the generic ``pops.moments`` builders, then lowers to a
``pops.model.Module`` and composes a time ``Program`` only through typed operator calls.
"""

import os
import sys
from pathlib import Path

import numpy as np

from pops import System, compile_problem
from pops.codegen import KokkosOpenMP, Production
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.moments import (
    CartesianVelocityMoments,
    MomentSource,
    RealizabilityProjection,
    VlasovElectricSource,
    moment_indices,
)
from pops.moments.closures import gaussian_closure
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
    root = Path(__file__).resolve().parents[2]
    os.environ["POPS_CACHE_DIR"] = str(root / ".pops_cache")
    include = _repo_include()
    if include is not None:
        os.environ["POPS_INCLUDE"] = include
    conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    if conda_prefix:
        os.environ.setdefault("POPS_KOKKOS_ROOT", conda_prefix)
        os.environ.setdefault("Kokkos_ROOT", conda_prefix)
    return include


def _relax_high_moments(m, M):
    nu = m.param("nu", 0.05)
    out = []
    for p, q in moment_indices(2):
        if (p, q) in ((0, 0), (1, 0), (0, 1)):
            out.append(0.0)
        else:
            out.append(-nu * M[(p, q)])
    return out


def build_model():
    spec = (
        CartesianVelocityMoments(
            order=2,
            closure=gaussian_closure(2),
            robust=True,
            exact_speeds=True,
        )
        .add_transport()
        .add_poisson_coupling(eps=1.0)
        .add_source(VlasovElectricSource(electric_field=("grad_x", "grad_y")))
        .add_source(MomentSource.from_rule("relax_high_moments", _relax_high_moments))
        .set_realizability(RealizabilityProjection(eps_m00=1e-10, eps_cov=1e-10))
    )
    return spec.to_module("custom_moments")


def build_program(module):
    T = Program("custom_moments_forward_euler")
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
    rho = 1.0 + 0.02 * np.cos(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
    u = 0.05
    v = -0.03
    c20 = 0.7
    c02 = 0.6
    c11 = 0.0
    return np.stack(
        [
            rho,
            rho * u,
            rho * (u * u + c20),
            rho * v,
            rho * (u * v + c11),
            rho * (v * v + c02),
        ]
    )


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


def run_once(n=8, cfl=0.1):
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
