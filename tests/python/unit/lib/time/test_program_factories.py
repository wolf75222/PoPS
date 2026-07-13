"""ADC-693: final pops.lib.time factories are ordinary canonical Programs."""
from __future__ import annotations

import inspect
from fractions import Fraction

import pytest

import pops.lib.time as libtime
from pops.identity.semantic import program_semantic_data, semantic_identity_of
from pops.physics.facade import Model
from pops.problem import Case
from pops.time import Program, StagePoint, TimePoint
from pops.time.method_properties import certify_program_graph
from pops.time.method_tableau import AdditiveRungeKuttaTableau, RungeKuttaTableau


def _authoring():
    model = Model("factory-model")
    model.conservative_vars("u")
    explicit = model.rate("explicit", flux=False, sources=())
    implicit = model.local_linear_map("implicit", [[-1]])
    case = Case("factory-case")
    block = case.block("fluid", model)
    declaration = next(
        item for item in model.declaration_index().records() if item.kind == "state")
    return block[declaration], declaration, explicit, implicit


def _manual_ssprk2(state, rate):
    program = Program("SSPRK2")
    temporal = program.state(state)
    u0 = temporal.n
    point0 = StagePoint("ssprk2_stage_0", {"main": TimePoint(program.clock, 0)})
    k0 = program.value("ssprk2_k_0", program.call(rate, u0), at=point0)
    point1 = StagePoint("ssprk2_stage_1", {"main": TimePoint(program.clock, 1)})
    u1 = program.value("ssprk2_U1", u0 + program.dt * k0, at=point1)
    k1 = program.value("ssprk2_k_1", program.call(rate, u1), at=point1)
    half = Fraction(1, 2)
    out = program.value(
        "ssprk2_step",
        u0 + (program.dt * half) * k0 + (program.dt * half) * k1,
        at=temporal.next.point,
    )
    program.commit(temporal.next, out)
    return program


def _manual_imex_euler(state, explicit, implicit):
    program = Program("IMEX")
    temporal = program.state(state)
    u0 = temporal.n
    point = StagePoint("imex-euler_stage_0", {
        "explicit": TimePoint(program.clock, 0),
        "implicit": TimePoint(program.clock, 1),
    })
    predictor = program.value("imex-euler_predictor_0", 1 * u0, at=point)
    linear = program.value("imex-euler_L_0", program.call(implicit), at=point)
    stage = program.solve_local_linear(
        "imex-euler_stage_solve_0",
        operator=program.I - program.dt * linear,
        rhs=predictor,
        fields=None,
    )
    stage = program.value("imex-euler_stage_0", stage, at=point)
    explicit_rate = program.value(
        "imex-euler_k_exp_0", program.call(explicit, stage), at=point)
    implicit_rate = program.value(
        "imex-euler_k_imp_0", program.apply(linear, stage, fields=None), at=point)
    out = program.value(
        "imex-euler_step",
        u0 + program.dt * explicit_rate + program.dt * implicit_rate,
        at=temporal.next.point,
    )
    program.commit(temporal.next, out)
    return program


def test_ssprk2_factory_matches_manual_graph_identity_and_certificate():
    state, _, rate, _ = _authoring()
    preset = libtime.SSPRK2(state, rate=rate)
    manual = _manual_ssprk2(state, rate)

    assert type(preset) is Program
    assert preset.validate() is True
    assert preset.to_graph().to_data() == manual.to_graph().to_data()
    assert preset.to_graph().graph_hash == manual.to_graph().graph_hash
    assert program_semantic_data(preset) == program_semantic_data(manual)
    assert semantic_identity_of(program=preset) == semantic_identity_of(program=manual)
    certificate = certify_program_graph(preset.to_graph())
    assert certificate.properties.order == 2
    assert certificate.properties.ssp.coefficient == 1


def test_imex_factory_matches_the_same_manual_program_graph():
    state, _, explicit, implicit = _authoring()
    preset = libtime.IMEX(
        state, explicit_operator=explicit, implicit_operator=implicit)
    manual = _manual_imex_euler(state, explicit, implicit)

    assert type(preset) is Program
    assert preset.validate() is True
    assert preset.to_graph().to_data() == manual.to_graph().to_data()
    assert preset.to_graph().graph_hash == manual.to_graph().graph_hash
    assert program_semantic_data(preset) == program_semantic_data(manual)
    assert semantic_identity_of(program=preset) == semantic_identity_of(program=manual)


def test_imex_accepts_an_exact_generic_additive_tableau():
    state, _, explicit, implicit = _authoring()
    heun = RungeKuttaTableau(
        A=[[], [1]], b=[Fraction(1, 2), Fraction(1, 2)], c=[0, 1], name="heun")
    tableau = AdditiveRungeKuttaTableau(
        heun,
        implicit_A=[[Fraction(1, 2)], [0, Fraction(1, 2)]],
        implicit_b=[Fraction(1, 2), Fraction(1, 2)],
        name="generic-two-stage",
    )
    program = libtime.IMEX(
        state,
        explicit_operator=explicit,
        implicit_operator=implicit,
        tableau=tableau,
    )
    assert program.validate() is True
    assert [node["op"] for node in program.ir_nodes()].count("solve_local_linear") == 2
    assert [node["op"] for node in program.ir_nodes()].count("rhs") == 2


def test_factories_reject_legacy_shapes_and_free_operator_names():
    state, declaration, rate, implicit = _authoring()
    with pytest.raises(TypeError, match=r"block\[state\]"):
        libtime.SSPRK2(declaration, rate=rate)
    with pytest.raises(TypeError, match="OperatorHandle"):
        libtime.SSPRK2(state, rate="explicit")
    with pytest.raises(TypeError, match="OperatorHandle"):
        libtime.IMEX(
            state, explicit_operator=rate, implicit_operator="implicit")
    with pytest.raises(TypeError, match="positional"):
        libtime.SSPRK2(state, declaration, rate=rate)
    with pytest.raises(TypeError, match="unexpected keyword"):
        libtime.SSPRK2(state, rate=rate, flux=True)
    with pytest.raises(TypeError, match="unexpected keyword"):
        libtime.IMEX(
            state,
            explicit_operator=rate,
            implicit_operator=implicit,
            theta=1,
        )

    assert "block" not in inspect.signature(libtime.SSPRK2).parameters
    assert "state" in inspect.signature(libtime.SSPRK2).parameters
    assert "state" in inspect.signature(libtime.IMEX).parameters
    assert not hasattr(libtime, "imex_local")
    assert not hasattr(libtime, "imex_local_linear")
    assert not hasattr(libtime, "ark_local_linear")
