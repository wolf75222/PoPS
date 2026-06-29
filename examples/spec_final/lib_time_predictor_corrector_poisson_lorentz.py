#!/usr/bin/env python3
"""Library-time predictor-corrector for Euler-Poisson-Lorentz.

This is the same public execution route as the manual board example, but the
time Program is assembled by ``pops.lib.time.predictor_corrector_local_linear``:

    typed physics model + library time macro
        -> compile_problem(...)
        -> System(layout=...)
        -> sim.install(...)
        -> sim.step_cfl(...)

All numerical work runs in generated/native C++.
"""

import sys
from pathlib import Path

import numpy as np

from pops import System, compile_problem
from pops.codegen import KokkosOpenMP, Production
from pops.lib.time import predictor_corrector_local_linear
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.numerics.riemann import Rusanov
from pops.numerics.reconstruction import MUSCL
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.spatial import FiniteVolume
from pops.runtime import Profile
from pops.solvers.elliptic import GeometricMG
from pops.time import Program

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.spec_final.manual_board_predictor_corrector_poisson_lorentz import (
    _configure_source_tree_include,
    build_model,
    initial_state,
)


def build_program(module):
    T = Program("lib_time_predictor_corrector_poisson_lorentz")
    T.bind_operators(module)

    operators = module.operator_registry()
    fields_from_state = operators.get("fields_from_state")
    explicit_rate = operators.get("explicit_rate")
    implicit_operator = operators.get("implicit_operator")

    U_next = predictor_corrector_local_linear(
        T,
        "plasma",
        fields_operator=fields_from_state,
        explicit_rate_operator=explicit_rate,
        implicit_operator=implicit_operator,
        commit=False,
    )

    fields_next = T.call(fields_from_state, U_next, name="fields_next")
    T.record_scalar("mass", T.sum_component(U_next, 0))
    T.record_scalar("rho_min", T.min_component(U_next, 0))
    T.record_scalar("rho_max", T.max_component(U_next, 0))
    T.commit("plasma", U_next, fields=fields_next)
    T.validate()
    return T


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
