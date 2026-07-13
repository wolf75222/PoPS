"""ADC-693: final pops.lib.time factories are ordinary canonical Programs."""
from __future__ import annotations

import inspect
from fractions import Fraction

import pytest

import pops.lib.time as libtime
from pops.identity.semantic import program_semantic_data, semantic_identity_of
from pops.physics.facade import Model
from pops.problem import Case
from pops.solvers import DenseLU
from pops.time import FailRun, LocalLinear, Program, StagePoint, TimePoint
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
    k0 = program.value("ssprk2_k_0", rate(u0), at=point0)
    point1 = StagePoint("ssprk2_stage_1", {"main": TimePoint(program.clock, 1)})
    u1 = program.value("ssprk2_U1", u0 + program.dt * k0, at=point1)
    k1 = program.value("ssprk2_k_1", rate(u1), at=point1)
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
    linear = program.value("imex-euler_L_0", program._call(implicit), at=point)
    stage = program.solve(
        LocalLinear(operator=program.I - program.dt * linear, rhs=predictor),
        solver=DenseLU(), name="imex-euler_stage_solve_0",
    ).consume(action=FailRun())
    stage = program.value("imex-euler_stage_0", stage, at=point)
    explicit_rate = program.value(
        "imex-euler_k_exp_0", explicit(stage), at=point)
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


@pytest.mark.parametrize(
    ("factory", "expected_order"),
    [
        (libtime.ForwardEuler, 1),
        (libtime.SSPRK2, 2),
        (libtime.SSPRK3, 3),
        (libtime.RK4, 4),
    ],
)
def test_explicit_factories_are_valid_ordinary_programs(factory, expected_order):
    state, _, rate, _ = _authoring()
    program = factory(state, rate=rate)

    assert type(program) is Program
    assert program.validate() is True
    assert certify_program_graph(program.to_graph()).properties.order == expected_order


def test_generic_runge_kutta_and_multistep_factories_take_complete_typed_choices():
    state, _, rate, implicit = _authoring()
    heun = RungeKuttaTableau(
        A=[[], [1]],
        b=[Fraction(1, 2), Fraction(1, 2)],
        c=[0, 1],
        name="heun",
    )

    assert libtime.RungeKutta(state, rate=rate, tableau=heun).validate() is True
    assert libtime.AdamsBashforth(state, rate=rate, order=2).validate() is True
    assert libtime.BDF(state, implicit=implicit, explicit=rate, order=1).validate() is True


def _identity_subflow(program, state, fraction, *, at):
    return program.value(getattr(at, "name", "endpoint") + "_value", 1 * state, at=at)


def test_splitting_factories_retain_exact_partition_coordinates():
    state, _, _, _ = _authoring()
    strang = libtime.Strang(
        state, first=_identity_subflow, second=_identity_subflow)
    lie = libtime.Lie(
        state, first=_identity_subflow, second=_identity_subflow)

    assert strang.validate() is True
    assert lie.validate() is True
    stage_points = {
        value.point.name: value.point
        for value in strang._values
        if isinstance(value.point, StagePoint)
    }
    first = stage_points["strang_first_half"]
    second = stage_points["strang_second"]
    assert first.time_for("first") == TimePoint(strang.clock, Fraction(1, 2))
    assert first.time_for("second") == TimePoint(strang.clock, 0)
    assert second.time_for("first") == TimePoint(strang.clock, Fraction(1, 2))
    assert second.time_for("second") == TimePoint(strang.clock, 1)


def test_splitting_rejects_a_subflow_that_lies_about_its_endpoint():
    state, _, _, _ = _authoring()

    def wrong_endpoint(program, current, fraction, *, at):
        return program.value("wrong", 1 * current)

    with pytest.raises(ValueError, match="instead of"):
        libtime.Strang(
            state, first=wrong_endpoint, second=_identity_subflow)


def test_final_time_namespace_has_no_legacy_aliases_or_specialized_runtime_preset():
    for name in (
        "forward_euler",
        "ssprk3",
        "rk4",
        "rk",
        "explicit_rk",
        "strang",
        "lie",
        "adams_bashforth",
        "adams_bashforth2",
        "bdf",
        "predictor_corrector_local_linear",
        "CondensedSchur",
    ):
        assert not hasattr(libtime, name)

    program = Program("final-surface")
    for name in ("bind_operators", "linear_combine", "define", "fields", "op"):
        assert not hasattr(program, name)
    assert callable(program.solve)
