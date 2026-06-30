"""Compiled-problem cadence around ``sim.step``.

The cadence record is a runtime install detail around a compiled problem artifact. This test keeps
the public execution route clean and does not compare against old native time policies.
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
DT = 2e-3


def _initial_state():
    x = (np.arange(N, dtype=float) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
    return np.stack([rho, 0.4 * rho, -0.2 * rho])


def _model():
    return isothermal_transport_module("time_substeps_stride_model")


def _program(module, name="fe_cadence"):
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


def _install(compiled, cadence):
    sim = pops.System(layout=Uniform(CartesianMesh(n=N, L=1.0, periodic=True)))
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


def _run(compiled, cadence, n_steps, dt=DT):
    sim = _install(compiled, cadence)
    for _ in range(n_steps):
        sim.step(dt)
    return np.array(sim.get_state("ions")), float(sim.time()), sim.macro_step()


def test_compiled_program_cadence_validation():
    assert CompiledProgramCadence(substeps=2).substeps == 2
    assert CompiledProgramCadence(stride=2).stride == 2
    assert CompiledProgramCadence(substeps=3, stride=4).substeps == 3
    assert CompiledProgramCadence(substeps=3, stride=4).stride == 4
    assert CompiledProgramCadence(cfl="program").cfl == "program"

    with pytest.raises(ValueError):
        CompiledProgramCadence(substeps=0)
    with pytest.raises(ValueError):
        CompiledProgramCadence(stride=0)
    with pytest.raises(ValueError):
        CompiledProgramCadence(cfl="not-a-policy")


@pytest.mark.requires_toolchain
def test_compiled_substeps_match_equivalent_half_steps():
    compiled = _compile()

    sub2, _, _ = _run(compiled, CompiledProgramCadence(substeps=2), 1, DT)
    half_twice, _, _ = _run(compiled, CompiledProgramCadence(substeps=1), 2, DT / 2.0)
    sub1, _, _ = _run(compiled, CompiledProgramCadence(substeps=1), 1, DT)

    assert np.array_equal(sub2, half_twice)
    assert float(np.abs(sub2 - sub1).max()) > 1e-9


@pytest.mark.requires_toolchain
def test_compiled_stride_holds_then_catches_up():
    compiled = _compile()
    steps = 4

    stride2, t_stride2, macro_stride2 = _run(compiled, CompiledProgramCadence(stride=2), steps, DT)
    stride2_again, t_again, macro_again = _run(
        compiled, CompiledProgramCadence(stride=2), steps, DT)
    stride1, _, _ = _run(compiled, CompiledProgramCadence(stride=1), steps, DT)

    assert np.array_equal(stride2, stride2_again)
    assert abs(t_stride2 - steps * DT) < 1e-14
    assert abs(t_again - steps * DT) < 1e-14
    assert macro_stride2 == steps
    assert macro_again == steps
    assert float(np.abs(stride2 - stride1).max()) > 1e-9
