"""ADC-668: manual and preset methods execute through one native ProgramGraph pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pops.lib.time as libtime
from pops.codegen.program_graph_lowering import emit_program_graph
from pops.physics._facade import Model
from pops.problem import Case
from pops.solvers import DenseLU
from pops.time import (
    FailRun,
    LocalLinear,
    Program,
    ProgramGraph,
    StagePoint,
    TimePoint,
)
from pops.time._program.detach import detach_compiled_program
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)


_EXPLICIT_RATE = -0.75
_IMPLICIT_RATE = -1.25


def _require_native() -> None:
    missing = missing_native_compile_requirement(repo_include(), default_cxx())
    if missing:
        require_native_or_skip(missing, optional_skip=pytest.skip)
    try:
        import pops.runtime._engine_descriptors  # noqa: F401
        import pops.runtime._system  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        require_native_or_skip(
            "normalized Program execution runtime unavailable: %s" % exc,
            optional_skip=pytest.skip,
        )


def _authoring() -> tuple[Any, Any, Any, Any]:
    model = Model("normalized-execution-model")
    (rho,) = model.conservative_vars("rho")
    model.primitive_vars(rho)
    model.conservative_from([rho])
    model.flux(x=[0.0 * rho], y=[0.0 * rho])
    model.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    model.source([_EXPLICIT_RATE * rho])
    explicit = model.rate("explicit_decay", flux=False, sources=("default",))
    implicit = model.local_linear_map("implicit_decay", [[_IMPLICIT_RATE]])
    block = Case("normalized-execution-case").block("fluid", model)
    declaration = next(
        record for record in model.declaration_index().records() if record.kind == "state"
    )
    return model, block[declaration], explicit, implicit


def _manual_ssprk2(state: Any, rate: Any) -> Program:
    program = Program("SSPRK2")
    temporal = program.state(state)
    u0 = temporal.n
    point0 = StagePoint("ssprk2_stage_0", {"main": TimePoint(program.clock, 0)})
    k0 = program.value("ssprk2_k_0", rate(u0), at=point0)
    point1 = StagePoint("ssprk2_stage_1", {"main": TimePoint(program.clock, 1)})
    u1 = program.value("ssprk2_U1", u0 + program.dt * k0, at=point1)
    k1 = program.value("ssprk2_k_1", rate(u1), at=point1)
    half = Fraction(1, 2)
    out = program.value(
        "ssprk2_step",
        u0 + program.dt * half * k0 + program.dt * half * k1,
        at=temporal.next.point,
    )
    program.commit(temporal.next, out)
    return program


def _manual_imex_euler(state: Any, explicit: Any, implicit: Any) -> Program:
    program = Program("IMEX")
    temporal = program.state(state)
    u0 = temporal.n
    point = StagePoint(
        "imex-euler_stage_0",
        {
            "explicit": TimePoint(program.clock, 0),
            "implicit": TimePoint(program.clock, 1),
        },
    )
    predictor = program.value("imex-euler_predictor_0", 1 * u0, at=point)
    linear = program.value("imex-euler_L_0", program._call(implicit), at=point)
    stage = program.solve(
        LocalLinear(operator=program.I - program.dt * linear, rhs=predictor),
        solver=DenseLU(),
        name="imex-euler_stage_solve_0",
    ).consume(action=FailRun())
    stage = program.value("imex-euler_stage_0", stage, at=point)
    explicit_rate = program.value("imex-euler_k_exp_0", explicit(stage), at=point)
    implicit_rate = program.value("imex-euler_k_imp_0", program.apply(linear, stage), at=point)
    out = program.value(
        "imex-euler_step",
        u0 + program.dt * explicit_rate + program.dt * implicit_rate,
        at=temporal.next.point,
    )
    program.commit(temporal.next, out)
    return program


@dataclass(frozen=True)
class _NormalizedProgram:
    authored: Program
    detached: Program
    graph: ProgramGraph
    source: str


def _normalize_and_lower(program: Program, model: Any) -> _NormalizedProgram:
    detached = detach_compiled_program(program)
    graph = detached.to_graph()
    source = emit_program_graph(graph, lowering_program=detached, model=model)
    assert detached.to_graph().graph_hash == graph.graph_hash
    return _NormalizedProgram(
        authored=program,
        detached=detached,
        graph=graph,
        source=source,
    )


@dataclass(frozen=True)
class _NativeResult:
    state: np.ndarray
    installed_hash: str
    time: float
    macro_step: int


def _install_and_step(
    compiled: Any, compiled_model: Any, initial: np.ndarray, dt: float
) -> _NativeResult:
    import pops.runtime._engine_descriptors as engine
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._system import System

    simulation = System(n=4, L=1.0, periodicity=(True, True))
    simulation.add_equation(
        "fluid",
        compiled_model,
        spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
        time=engine.Explicit(method="euler"),
    )
    simulation.set_state("fluid", np.stack([initial]))
    simulation.install_program(compiled.so_path)
    assert simulation.installed_program_hash() == compiled.program_hash
    simulation.step(dt)
    return _NativeResult(
        state=np.asarray(simulation.get_state("fluid"))[0].copy(),
        installed_hash=simulation.installed_program_hash(),
        time=float(simulation.time()),
        macro_step=int(simulation.macro_step()),
    )


def test_manual_preset_ssprk2_and_imex_execute_through_same_program_graph_pipeline(
    tmp_path: Path,
) -> None:
    _require_native()
    from pops.codegen._compile_drivers import compile_problem

    model, state, explicit, implicit = _authoring()
    authored = {
        "ssprk2_manual": _manual_ssprk2(state, explicit),
        "ssprk2_preset": libtime.SSPRK2(state, rate=explicit),
        "imex_manual": _manual_imex_euler(state, explicit, implicit),
        "imex_preset": libtime.IMEX(
            state,
            explicit_operator=explicit,
            implicit_operator=implicit,
        ),
    }
    normalized = {name: _normalize_and_lower(program, model) for name, program in authored.items()}

    for value in normalized.values():
        assert value.detached._compiled_detached is True
        assert not value.detached._operator_registries
        assert type(value.graph) is ProgramGraph
        assert value.source.count("ctx.install([=](double dt)") == 1
        assert value.source.count("ctx.begin_step(dt)") == 1
        assert value.source.count("ctx.commit_many(") == 1
        assert (
            value.source.index("ctx.install([=](double dt)")
            < value.source.index("ctx.begin_step(dt)")
            < value.source.index("ctx.commit_many(")
        )

    for manual, preset in (
        ("ssprk2_manual", "ssprk2_preset"),
        ("imex_manual", "imex_preset"),
    ):
        assert normalized[manual].graph.to_data() == normalized[preset].graph.to_data()
        assert normalized[manual].graph.graph_hash == normalized[preset].graph.graph_hash
        assert normalized[manual].source == normalized[preset].source
    assert "pops::SolveOutcome" in normalized["imex_manual"].source
    assert normalized["imex_manual"].source.index("pops::SolveOutcome") < normalized[
        "imex_manual"
    ].source.index("ctx.commit_many(")

    compiled = {
        name: compile_problem(
            so_path=str(tmp_path / ("%s.so" % name)),
            model=model,
            time=value.authored,
            include=repo_include(),
            cxx=default_cxx(),
        )
        for name, value in normalized.items()
    }
    for name, value in compiled.items():
        assert Path(value.so_path).is_file()
        assert value.program.to_graph().graph_hash == normalized[name].graph.graph_hash
    assert compiled["ssprk2_manual"].program_hash == compiled["ssprk2_preset"].program_hash
    assert compiled["imex_manual"].program_hash == compiled["imex_preset"].program_hash

    compiled_model = model.compile(
        backend="production",
        include=repo_include(),
        cxx=default_cxx(),
    )
    axis = (np.arange(4, dtype=float) + 0.5) / 4.0
    x, y = np.meshgrid(axis, axis, indexing="ij")
    initial = 1.0 + 0.2 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    dt = 0.2
    results = {
        name: _install_and_step(value, compiled_model, initial, dt)
        for name, value in compiled.items()
    }

    z = dt * _EXPLICIT_RATE
    expected_ssprk2 = initial * (1.0 + z + 0.5 * z * z)
    implicit_stage = initial / (1.0 - dt * _IMPLICIT_RATE)
    expected_imex = initial + dt * (_EXPLICIT_RATE + _IMPLICIT_RATE) * implicit_stage
    for name in ("ssprk2_manual", "ssprk2_preset"):
        np.testing.assert_allclose(results[name].state, expected_ssprk2, rtol=0.0, atol=1.0e-13)
    for name in ("imex_manual", "imex_preset"):
        np.testing.assert_allclose(results[name].state, expected_imex, rtol=0.0, atol=1.0e-13)
    np.testing.assert_array_equal(
        results["ssprk2_manual"].state,
        results["ssprk2_preset"].state,
    )
    np.testing.assert_array_equal(
        results["imex_manual"].state,
        results["imex_preset"].state,
    )
    for name, result in results.items():
        assert result.installed_hash == compiled[name].program_hash
        assert result.time == pytest.approx(dt, rel=0.0, abs=1.0e-15)
        assert result.macro_step == 1
        assert np.isfinite(result.state).all()
