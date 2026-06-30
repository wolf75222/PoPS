"""Multi-stage Programs use operator-first calls and the final compiled route."""

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
    return isothermal_transport_module("time_multistage_model")


def _operators(module):
    registry = module.operator_registry()
    return registry.get("fields_from_state"), registry.get("explicit_rate")


def _rate(program, fields_operator, rate_operator, state, suffix):
    fields = program.call(fields_operator, state, name="fields_%s" % suffix)
    return program.call(rate_operator, state, fields, name="k_%s" % suffix), fields


def ssprk2_program(module):
    program = adctime.Program("ssprk2_operator_first").bind_operators(module)
    fields_operator, rate_operator = _operators(module)
    state = program.state("U", block="ions", space=module.state_spaces()["U"])

    k0, _ = _rate(program, fields_operator, rate_operator, state.n, "0")
    program.define(state.stage(1), state.n + program.dt * k0)
    k1, fields1 = _rate(program, fields_operator, rate_operator, state.stage(1), "1")
    program.define(state.next, 0.5 * state.n + 0.5 * (state.stage(1) + program.dt * k1))
    program.commit(state.next, fields=fields1)
    program.validate()
    return program


def rk4_program(module):
    program = adctime.Program("rk4_operator_first").bind_operators(module)
    fields_operator, rate_operator = _operators(module)
    state = program.state("U", block="ions", space=module.state_spaces()["U"])
    dt = program.dt

    k1, _ = _rate(program, fields_operator, rate_operator, state.n, "1")
    u1 = program.define("U1", state.n + 0.5 * dt * k1)
    k2, _ = _rate(program, fields_operator, rate_operator, u1, "2")
    u2 = program.define("U2", state.n + 0.5 * dt * k2)
    k3, _ = _rate(program, fields_operator, rate_operator, u2, "3")
    u3 = program.define("U3", state.n + dt * k3)
    k4, fields4 = _rate(program, fields_operator, rate_operator, u3, "4")

    program.define(
        state.next,
        state.n + dt / 6.0 * k1 + dt / 3.0 * k2 + dt / 3.0 * k3 + dt / 6.0 * k4,
    )
    program.commit(state.next, fields=fields4)
    program.validate()
    return program


def _compile(program_builder):
    include = str(REPO_ROOT / "include")
    os.environ.setdefault("POPS_INCLUDE", include)
    conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    if conda_prefix:
        os.environ.setdefault("POPS_KOKKOS_ROOT", conda_prefix)
        os.environ.setdefault("Kokkos_ROOT", conda_prefix)
    module = _model()
    return pops.compile_problem(
        model=module,
        program=program_builder(module),
        layout=Uniform(CartesianMesh(n=N, L=1.0, periodic=True)),
        backend=Production(platform=KokkosOpenMP()),
        include=include,
    )


def _install(compiled):
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
    )
    return sim


def _run(program_builder):
    compiled = _compile(program_builder)
    sim = _install(compiled)
    before = np.array(sim.get_state("ions"))
    sim.step(DT)
    after = np.array(sim.get_state("ions"))
    return compiled, before, after


def test_multistage_programs_are_operator_first_ir():
    module = _model()
    for builder in (ssprk2_program, rk4_program):
        program = builder(module)
        ops = [value.op for value in program._values]
        assert "call" in ops
        assert "rhs" not in ops
        assert "solve_fields" not in ops
        assert "linear_source" not in ops


@pytest.mark.requires_toolchain
def test_ssprk2_and_rk4_compile_and_advance():
    ssprk2, before_ssp, after_ssp = _run(ssprk2_program)
    rk4, before_rk4, after_rk4 = _run(rk4_program)

    assert ssprk2.problem_hash
    assert rk4.problem_hash
    assert np.array_equal(before_ssp, before_rk4)
    assert np.isfinite(after_ssp).all()
    assert np.isfinite(after_rk4).all()
    assert float(np.abs(after_ssp - before_ssp).max()) > 1e-9
    assert float(np.abs(after_rk4 - before_rk4).max()) > 1e-9
    assert float(np.abs(after_ssp - after_rk4).max()) > 0.0
