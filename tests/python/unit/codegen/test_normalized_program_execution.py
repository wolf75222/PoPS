"""ADC-668: manual and preset methods share one normalized execution pipeline.

This proof deliberately stops before native compilation.  It exercises the real immutable
``ProgramGraph`` boundary and C++ lowering, then drives that same graph with a small scalar reference
driver.  The native acceptance matrix can therefore reuse the exact programs without introducing a
second authoring or normalization route.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

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


_EXPLICIT_RATE = -0.75
_IMPLICIT_RATE = -1.25


def _authoring() -> tuple[Any, Any, Any, Any]:
    model = Model("normalized-execution-model")
    (u,) = model.conservative_vars("u")
    model.source([_EXPLICIT_RATE * u])
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


def _number(data: dict[str, Any]) -> Fraction | float:
    if "scalar" in data:
        return _number(data["scalar"])
    kind = data["kind"]
    if kind == "integer":
        return Fraction(int(data["value"]))
    if kind == "rational":
        return Fraction(int(data["numerator"]), int(data["denominator"]))
    if kind in {"decimal", "float"}:
        return float(data["value"])
    raise AssertionError("unsupported canonical scalar %r" % (data,))


def _dt_polynomial(terms: list[list[dict[str, Any]]], dt: float) -> float:
    return sum(float(_number(value)) * dt ** int(_number(power)) for power, value in terms)


class _ScalarProgramGraphDriver:
    """Execute the scalar subset used by both normalized method families."""

    def __init__(self, operators: dict[str, float]) -> None:
        self._operators = dict(operators)

    def step(self, graph: ProgramGraph, *, state: float, dt: float) -> float:
        if type(graph) is not ProgramGraph:
            raise TypeError("scalar reference driver requires an exact ProgramGraph")
        values: dict[int, Any] = {}
        committed: list[float] = []

        for node in graph.to_data()["nodes"]:
            node_id = int(node["node_id"])
            kind = node["kind"]
            if kind == "state_read":
                values[node_id] = float(state)
                continue
            if kind == "operator_call":
                operator = node["operator"]
                name = operator["handle"]["local_id"]
                coefficient = self._operators[name]
                operation = operator["lowering"]["op"]
                inputs = [values[int(ref["node_id"])] for ref in node["inputs"]]
                if operation == "linear_source":
                    assert not inputs
                    values[node_id] = coefficient
                elif operation in {"rhs", "apply"}:
                    assert len(inputs) == 1
                    values[node_id] = coefficient * inputs[0]
                else:
                    raise AssertionError("unsupported operator call %r" % operation)
                continue
            if kind == "program_value":
                operation = node["op"]
                attrs = node["attrs"]["attrs"]
                inputs = [values[int(ref["node_id"])] for ref in node["inputs"]]
                if operation == "linear_combine":
                    values[node_id] = sum(
                        _dt_polynomial(terms, dt) * value
                        for terms, value in zip(attrs["coeffs"], inputs, strict=True)
                    )
                elif operation == "solve_local_linear":
                    rhs, linear = inputs
                    a = _dt_polynomial(attrs["a_coeff"], dt)
                    values[node_id] = rhs / (1.0 - a * linear)
                elif operation == "solve_outcome":
                    assert len(inputs) == 1
                    values[node_id] = tuple(inputs)
                elif operation == "solve_outcome_component":
                    index = int(_number(attrs["index"]))
                    values[node_id] = inputs[0][index]
                else:
                    raise AssertionError("unsupported Program value %r" % operation)
                continue
            if kind == "commit":
                committed.append(float(values[int(node["value"]["node_id"])]))
                continue
            raise AssertionError("unsupported ProgramGraph node %r" % kind)

        assert len(committed) == 1
        return committed[0]


@dataclass(frozen=True)
class _Execution:
    detached: Program
    graph: ProgramGraph
    source: str
    value: float


def _normalize_lower_and_drive(
    program: Program,
    *,
    model: Any,
    driver: _ScalarProgramGraphDriver,
    initial: float,
    dt: float,
) -> _Execution:
    detached = detach_compiled_program(program)
    graph = detached.to_graph()
    source = emit_program_graph(graph, lowering_program=detached, model=model)
    value = driver.step(graph, state=initial, dt=dt)
    assert detached.to_graph().graph_hash == graph.graph_hash
    return _Execution(detached=detached, graph=graph, source=source, value=value)


def test_manual_preset_ssprk2_and_imex_share_normalize_lower_driver_pipeline():
    model, state, explicit, implicit = _authoring()
    programs = {
        "ssprk2_manual": _manual_ssprk2(state, explicit),
        "ssprk2_preset": libtime.SSPRK2(state, rate=explicit),
        "imex_manual": _manual_imex_euler(state, explicit, implicit),
        "imex_preset": libtime.IMEX(
            state,
            explicit_operator=explicit,
            implicit_operator=implicit,
        ),
    }
    driver = _ScalarProgramGraphDriver(
        {"explicit_decay": _EXPLICIT_RATE, "implicit_decay": _IMPLICIT_RATE}
    )
    initial = 1.5
    dt = 0.2
    executions = {
        name: _normalize_lower_and_drive(
            program, model=model, driver=driver, initial=initial, dt=dt
        )
        for name, program in programs.items()
    }

    for execution in executions.values():
        assert execution.detached._compiled_detached is True
        assert not execution.detached._operator_registries
        assert type(execution.graph) is ProgramGraph
        assert execution.source.count("ctx.install([=](double dt)") == 1
        assert execution.source.count("ctx.begin_step(dt)") == 1
        assert execution.source.count("ctx.commit_many(") == 1
        assert (
            execution.source.index("ctx.install([=](double dt)")
            < execution.source.index("ctx.begin_step(dt)")
            < execution.source.index("ctx.commit_many(")
        )

    ssprk2_manual = executions["ssprk2_manual"]
    ssprk2_preset = executions["ssprk2_preset"]
    assert ssprk2_manual.graph.to_data() == ssprk2_preset.graph.to_data()
    assert ssprk2_manual.graph.graph_hash == ssprk2_preset.graph.graph_hash
    assert ssprk2_manual.source == ssprk2_preset.source

    imex_manual = executions["imex_manual"]
    imex_preset = executions["imex_preset"]
    assert imex_manual.graph.to_data() == imex_preset.graph.to_data()
    assert imex_manual.graph.graph_hash == imex_preset.graph.graph_hash
    assert imex_manual.source == imex_preset.source
    assert "pops::SolveOutcome" in imex_manual.source
    assert imex_manual.source.index("pops::SolveOutcome") < imex_manual.source.index(
        "ctx.commit_many("
    )

    z = dt * _EXPLICIT_RATE
    expected_ssprk2 = initial * (1.0 + z + 0.5 * z * z)
    implicit_stage = initial / (1.0 - dt * _IMPLICIT_RATE)
    expected_imex = initial + dt * (_EXPLICIT_RATE + _IMPLICIT_RATE) * implicit_stage
    assert ssprk2_manual.value == pytest.approx(expected_ssprk2, rel=0.0, abs=1.0e-14)
    assert ssprk2_preset.value == ssprk2_manual.value
    assert imex_manual.value == pytest.approx(expected_imex, rel=0.0, abs=1.0e-14)
    assert imex_preset.value == imex_manual.value
    assert 0.0 < imex_manual.value < ssprk2_manual.value < initial
