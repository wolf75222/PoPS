"""``System.step_cfl`` drives the installed compiled problem artifact.

The public route under test is exactly:

    compiled = pops.compile_problem(...)
    sim = pops.System(...)
    sim.install(compiled, ...)
    sim.step_cfl(...)

The test intentionally avoids old native-only install shortcuts and private readbacks.
"""

from pathlib import Path
import os
import sys

import numpy as np
import pytest

import pops
from pops import time as adctime
from pops.codegen import KokkosOpenMP, Production
from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform
from pops.runtime._compiled_cadence import CompiledProgramCadence
from pops.solvers.elliptic import GeometricMG

from _module_models import first_order_rusanov, isothermal_transport_module


REPO_ROOT = Path(__file__).resolve().parents[2]
N = 24
CFL = 0.4


def _initial_state():
    x = (np.arange(N, dtype=float) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
    return np.stack([rho, 0.4 * rho, -0.2 * rho])


def _model():
    return isothermal_transport_module("time_step_cfl_model")


def _program(module, name="fe_step_cfl"):
    program = adctime.Program(name).bind_operators(module)
    state = program.state("U", block="ions", space=module.state_spaces()["U"])
    operators = module.operator_registry()
    fields = program.call(operators.get("fields_from_state"), state.n, name="fields_n")
    rate = program.call(operators.get("explicit_rate"), state.n, fields, name="R_n")
    program.define(state.next, state.n + program.dt * rate)
    program.commit(state.next, fields=fields)
    program.validate()
    return program


def _compile():
    include = str(REPO_ROOT / "include")
    os.environ.setdefault("POPS_INCLUDE", include)
    conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    if conda_prefix:
        os.environ.setdefault("POPS_KOKKOS_ROOT", conda_prefix)
        os.environ.setdefault("Kokkos_ROOT", conda_prefix)
    module = _model()
    return pops.compile_problem(
        model=module,
        program=_program(module),
        layout=Uniform(CartesianMesh(n=N, L=1.0, periodic=True)),
        backend=Production(platform=KokkosOpenMP()),
        include=include,
    )


def _install(compiled, cadence=None):
    mesh = CartesianMesh(n=N, L=1.0, periodic=True)
    sim = pops.System(layout=Uniform(mesh))
    sim.install(
        compiled,
        instances={
            "ions": {
                "initial": _initial_state(),
                "spatial": first_order_rusanov(),
            }
        },
        solvers={"phi": GeometricMG()},
        cadence=cadence,
    )
    return sim


def _state(sim):
    return np.array(sim.get_state("ions"))


def test_step_cfl_api_present():
    assert hasattr(pops.System(n=8, L=1.0, periodic=True), "step_cfl")


@pytest.mark.requires_toolchain
def test_step_cfl_matches_fixed_step_at_chosen_dt():
    compiled = _compile()

    sim_cfl = _install(compiled)
    before = _state(sim_cfl)
    dt = sim_cfl.step_cfl(CFL)
    after_cfl = _state(sim_cfl)

    assert dt > 0.0 and np.isfinite(dt)
    assert float(np.abs(after_cfl - before).max()) > 1e-9

    sim_fixed = _install(compiled)
    sim_fixed.step(dt)
    after_fixed = _state(sim_fixed)

    assert np.array_equal(after_cfl, after_fixed)


@pytest.mark.requires_toolchain
def test_step_cfl_honors_compiled_cadence():
    compiled = _compile()

    sim_sub1 = _install(compiled, CompiledProgramCadence(substeps=1))
    dt_sub1 = sim_sub1.step_cfl(CFL)
    state_sub1 = _state(sim_sub1)

    sim_sub2 = _install(compiled, CompiledProgramCadence(substeps=2))
    dt_sub2 = sim_sub2.step_cfl(CFL)
    state_sub2 = _state(sim_sub2)

    assert abs(dt_sub2 - dt_sub1) < 1e-14

    sim_sub2_fixed = _install(compiled, CompiledProgramCadence(substeps=2))
    sim_sub2_fixed.step(dt_sub2)
    assert np.array_equal(state_sub2, _state(sim_sub2_fixed))
    assert float(np.abs(state_sub2 - state_sub1).max()) > 1e-9

    steps = 4
    sim_stride = _install(compiled, CompiledProgramCadence(stride=2))
    dts = [sim_stride.step_cfl(CFL) for _ in range(steps)]
    state_stride = _state(sim_stride)

    sim_stride_fixed = _install(compiled, CompiledProgramCadence(stride=2))
    for dt in dts:
        sim_stride_fixed.step(dt)
    assert np.array_equal(state_stride, _state(sim_stride_fixed))
    assert abs(float(sim_stride.time()) - float(sum(dts))) < 1e-12
    assert sim_stride.macro_step() == steps

    sim_stride1 = _install(compiled, CompiledProgramCadence(stride=1))
    for dt in dts:
        sim_stride1.step(dt)
    assert float(np.abs(state_stride - _state(sim_stride1)).max()) > 1e-9
