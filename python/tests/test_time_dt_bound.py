"""Optional Program dt bounds lower to the compiled-problem ABI and affect ``step_cfl``."""

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
CFL = 0.4


def _initial_state():
    x = (np.arange(N, dtype=float) + 0.5) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
    return np.stack([rho, 0.4 * rho, -0.2 * rho])


def _model():
    return isothermal_transport_module("time_dt_bound_model")


def _program(module, name="fe_dt_bound", factor=None):
    program = adctime.Program(name).bind_operators(module)
    state = program.state("U", block="ions", space=module.state_spaces()["U"])
    operators = module.operator_registry()
    fields = program.call(operators.get("fields_from_state"), state.n, name="fields_n")
    rate = program.call(operators.get("explicit_rate"), state.n, fields, name="R_n")
    program.define(state.next, state.n + program.dt * rate)
    program.commit(state.next, fields=fields)

    if factor is not None:
        @program.dt_bound
        def _bound(P, cfl):
            bound_state = P.state("U", block="ions", space=module.state_spaces()["U"])
            w = P.max_wave_speed(bound_state.n)
            return factor * cfl * P.hmin() / w

    program.validate()
    return program


def _configure_toolchain_env():
    include = str(REPO_ROOT / "include")
    os.environ.setdefault("POPS_INCLUDE", include)
    conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    if conda_prefix:
        os.environ.setdefault("POPS_KOKKOS_ROOT", conda_prefix)
        os.environ.setdefault("Kokkos_ROOT", conda_prefix)
    return include


def _compile(factor=None, name="fe_dt_bound"):
    include = _configure_toolchain_env()
    module = _model()
    return pops.compile_problem(
        model=module,
        program=_program(module, name=name, factor=factor),
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


def _state(sim):
    return np.array(sim.get_state("ions"))


def test_dt_bound_codegen_uses_problem_abi_and_operator_first_program():
    module = _model()

    no_bound = _program(module, "fe_no_bound")
    assert not no_bound.has_dt_bound()
    src_no = no_bound.emit_cpp_program(model=module)
    assert "bool pops_problem_has_dt_bound()" in src_no
    assert "pops::Real pops_problem_dt_bound(" in src_no
    assert "return false;" in src_no
    assert "std::numeric_limits<pops::Real>::infinity()" in src_no

    with_bound = _program(module, "fe_with_bound", factor=0.5)
    assert with_bound.has_dt_bound()
    src_bound = with_bound.emit_cpp_program(model=module)
    assert "return true;" in src_bound
    assert "ctx.hmin()" in src_bound
    assert "ctx.max_wave_speed(0, " in src_bound
    assert "cfl" in src_bound.split("pops_problem_dt_bound", 1)[1]

    ops = [value.op for value in with_bound._values]
    assert "call" in ops
    assert "rhs" not in ops
    assert "solve_fields" not in ops
    assert "linear_source" not in ops


def test_dt_bound_validation_and_ir_hash():
    module = _model()
    no_bound = _program(module, "fe_no_bound")
    with_bound = _program(module, "fe_hash_bound", factor=0.5)
    assert no_bound._ir_hash() != with_bound._ir_hash()

    bad = adctime.Program("bad_dt_bound").bind_operators(module)
    with pytest.raises(ValueError, match="Scalar"):
        bad.set_dt_bound(lambda P, cfl: P.state("U", block="ions", space=module.state_spaces()["U"]).n)

    twice = adctime.Program("twice_dt_bound")
    twice.set_dt_bound(twice.hmin())
    with pytest.raises(ValueError, match="already set"):
        twice.set_dt_bound(twice.hmin())

    with pytest.raises(TypeError):
        bool(with_bound.hmin())


@pytest.mark.requires_toolchain
def test_step_cfl_applies_compiled_problem_dt_bound():
    no_bound = _compile(name="fe_no_bound")
    tight = _compile(factor=0.5, name="fe_tight_bound")
    loose = _compile(factor=2.0, name="fe_loose_bound")

    dt_base = _install(no_bound).step_cfl(CFL)
    dt_tight = _install(tight).step_cfl(CFL)
    dt_loose = _install(loose).step_cfl(CFL)

    assert dt_base > 0.0 and np.isfinite(dt_base)
    assert abs(dt_tight - 0.5 * dt_base) < 1e-9 * dt_base
    assert abs(dt_loose - dt_base) < 1e-14

    sim_tight = _install(tight)
    before = _state(sim_tight)
    dt = sim_tight.step_cfl(CFL)
    after_cfl = _state(sim_tight)

    assert abs(dt - dt_tight) < 1e-12
    assert float(np.abs(after_cfl - before).max()) > 1e-9

    sim_fixed = _install(tight)
    sim_fixed.step(dt)
    assert np.array_equal(after_cfl, _state(sim_fixed))
