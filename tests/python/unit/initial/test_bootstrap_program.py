from __future__ import annotations

import copy

import pytest

import pops
from pops.analytic import constant, coordinates, input as analytic_input, param
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.lib.initial import Analytic, BootstrapProgram, FieldSolve, ProviderDerivative, StateMap
from pops.model.provider_pack import ComponentKey
from pops.params import Positive, RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.runtime._initial_program_execution import execute_uniform_initial_program, resolved_initial_field_slot
from pops.runtime._initial_source_lowering import validate_initial_source
from pops.spaces import CellState


def _profile(*, seed_parameter=None):
    frame = Rectangle("square", (-16.0, -16.0), (16.0, 16.0)).frame(Cartesian2D())
    x, _ = coordinates(frame)
    density = x * x + 1.0 if seed_parameter is None else param(seed_parameter)
    phi = ComponentKey("case/electrons", "field", "electric", "phi")
    rho = analytic_input(0, "rho")
    phi_y = analytic_input(1, "phi_y")
    phi_x = analytic_input(2, "phi_x")
    mapping = StateMap(components=(rho, -rho * phi_y / 1.0e12, rho * phi_x / 1.0e12),
                       inputs={0: ("state", 0), 1: ProviderDerivative(phi, axis=1),
                               2: ProviderDerivative(phi, axis=0)},
                       consumer_qid="case/initial/drift")
    profile = BootstrapProgram(seed=Analytic(frame=frame, components=(density, constant(0), constant(0))),
                stages=(FieldSolve(), mapping, FieldSolve()), consumer_qid="case/initial/seed")
    return frame, profile


def _source(profile):
    return {**profile.initial_source_options(),
            **ConservativeCellAverage().initial_projection_options()}


def _qualified_field_plans():
    """A real resolved Case plan; this fixture exercises dispatch, not a numerical solve."""
    from tests.python.unit.time.test_solved_field_sampling import field_sample_case
    from pops.time import FixedDt
    case, program, temporal, fields, _potential, _space, _storage, authority = field_sample_case()
    program.commit(temporal.next, program.value("unchanged", temporal.n, at=temporal.next.point))
    program.step_strategy(FixedDt(.01))
    case.program(program)
    builder, providers = authority
    layout = builder.resolve(states=(case.resolve(temporal.n.state_ref),),
                             fields=(case.resolve(fields.field_context.field),),
                             blocks=(case.resolve(temporal.n.block),))
    resolved = pops.resolve(pops.validate(case), layout=layout, layout_providers=providers)
    return resolved.field_plans


def test_native_program_preserves_two_solves_and_qualified_derivative_inputs():
    _frame, profile = _profile()
    source = _source(profile)
    assert validate_initial_source(source, where="test bootstrap") == "bootstrap_program"
    assert source["seed_sampling"] == "cell_center"
    assert [stage["operation"] for stage in source["stages"]] == [
        "solve_fields", "map_state", "solve_fields"]
    inputs = source["stages"][1]["inputs"]
    assert inputs[1]["source"] == "provider_derivative"
    assert inputs[1]["axis"] == 1 and inputs[1]["edge_order"] == 2
    assert inputs[1]["key"]["component"] == "phi"
    assert profile.reprojectable is False


def test_uniform_execution_is_native_seed_solve_map_solve_without_step():
    _frame, profile = _profile()
    source = _source(profile)
    plans = _qualified_field_plans()
    expected_slot = next(iter(plans.values())).native_options["provider_slot"]

    class Native:
        def __init__(self):
            self.calls = []

        def _set_analytic_mapped_state(self, name, opcodes, literals, inputs, consumer):
            self.calls.append(("map", name, inputs, consumer, len(opcodes), len(literals)))

        def solve_fields(self):
            raise AssertionError("bootstrap must never select a legacy default solver")

        def _solve_initial_field(self, slot):
            self.calls.append(("solve", slot))
            from types import SimpleNamespace
            return SimpleNamespace(iters=7, evaluations=8, residual_norm=3.,
                reference_residual_norm=40., rel_residual=.075, status="solved",
                action="continue", reason="native test evidence", residual_evaluation=None)

    native = Native()
    reports = execute_uniform_initial_program(native, "carrier", source, params={}, field_plans=plans)
    assert [call[0] for call in native.calls] == ["map", "solve", "map", "solve"]
    assert native.calls[0][2] == []
    assert native.calls[2][2][1] == {key: value for key, value in source["stages"][1]["inputs"][1].items()
                                      if key != "value_id"}
    assert native.calls[0][4:] == (3, 3)
    assert [call[1] for call in native.calls if call[0] == "solve"] == [expected_slot, expected_slot]
    assert [row["ordinal"] for row in reports] == [1,2]
    assert all(row["residual_norm"] == 3. and row["reason"] == "native test evidence"
               and row["residual_evaluation"] is None for row in reports)


def test_initial_solve_report_preserves_roundoff_evidence_and_returns_independent_snapshot():
    from types import SimpleNamespace
    from pops.runtime._initial_program_execution import (
        initial_solve_report_data, initial_program_reports_snapshot)
    evidence = SimpleNamespace(initial_residual_norm=4., rhs_norm=100.,
        initial_evaluation_bound=.2, evaluation_bound=.3, requested_threshold=.04,
        applied_threshold=.6, roundoff_limited=True, relative_to_rhs_upper=.008)
    report = SimpleNamespace(iters=9, evaluations=10, residual_norm=.5,
        reference_residual_norm=4., rel_residual=.125, status="solved", action="continue",
        reason="roundoff bound", residual_evaluation=evidence)
    row = initial_solve_report_data(report,provider_slot="real-slot",ordinal=2)
    engine = SimpleNamespace(_initial_program_reports={"carrier": (row,)})
    snapshot = initial_program_reports_snapshot(engine)
    assert snapshot["carrier"][0]["residual_evaluation"]["relative_to_rhs_upper"] == .008
    snapshot["carrier"][0]["residual_evaluation"]["requested_threshold"] = 99.
    assert row["residual_evaluation"]["requested_threshold"] == .04


def test_initial_field_dispatch_refuses_missing_ambiguous_and_foreign_plans_before_seed():
    _frame, profile = _profile()
    plans = _qualified_field_plans()
    real = next(iter(plans.values()))

    class Native:
        def _set_analytic_mapped_state(self, *args):
            raise AssertionError("invalid field authority must fail before native seed publication")

    for supplied in (None, {}, {"one":real,"two":real}):
        with pytest.raises(ValueError, match="resolved Case|exactly one"):
            execute_uniform_initial_program(Native(), "carrier", _source(profile),
                                             params={}, field_plans=supplied)
    with pytest.raises(ValueError, match="exactly one"):
        resolved_initial_field_slot("foreign_block", plans)
    with pytest.raises(TypeError, match="exact Uniform"):
        resolved_initial_field_slot("carrier", {"forged": object()})


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(seed_sampling="cell_average"),
    lambda s: s["stages"][0].update(unknown=True),
    lambda s: s["stages"][1]["inputs"][1].update(edge_order=1),
    lambda s: s["stages"][1]["inputs"][1].update(axis=True),
    lambda s: s["stages"][1]["inputs"][1].update(raw_slot=2),
    lambda s: s["stages"][1]["inputs"][1]["key"].update(owner_qid=""),
    lambda s: s["stages"].pop(),
    lambda s: s["stages"][1]["components"].pop(),
])
def test_program_schema_refuses_ambiguous_or_changed_authority(mutation):
    _frame, profile = _profile()
    source = copy.deepcopy(_source(profile))
    mutation(source)
    with pytest.raises((TypeError, ValueError)):
        validate_initial_source(source, where="test bootstrap")


def test_derivative_refuses_raw_slots_invalid_axis_or_changed_order():
    key = ComponentKey("case/electrons", "field", "electric", "phi")
    with pytest.raises(TypeError, match="ComponentKey"):
        ProviderDerivative(3, axis=0)
    for axis, order in ((True, 2), (-1, 2), (3, 2), (0, 1), (0, True)):
        with pytest.raises(ValueError):
            ProviderDerivative(key, axis=axis, edge_order=order)
    frame, profile = _profile()
    wrong = StateMap(components=profile.stages[1].components,
        inputs={0: ("state", 0), 1: ProviderDerivative(key, axis=2),
                2: ProviderDerivative(key, axis=0)}, consumer_qid="case/initial/bad-axis")
    with pytest.raises(ValueError, match="physical rank"):
        BootstrapProgram(seed=profile.seed, stages=(FieldSolve(), wrong, FieldSolve()),
                         consumer_qid="case/initial/seed")
    assert len(frame.axes) == 2


def test_profile_captures_detached_parameter_authorities():
    case = pops.Case("initial-parameter")
    amplitude = case.param(RuntimeParam("amplitude", default=1.0, domain=Positive()))
    frame, profile = _profile(seed_parameter=amplitude)
    model = pops.Model("moments", frame=frame)
    state = model.state("U", components=("rho", "mx", "my"),
                        representation=Conservative(), space=CellState(frame=frame))
    block = case.block("electrons", model)
    initial = InitialCondition(state=block[state], value=profile,
                               projection=ConservativeCellAverage())
    assert initial._value_reference_handles == (amplitude,)
    assert profile.captured_reference_handles() == (amplitude,)
    resolved = initial.resolve_references(case.resolve)
    assert resolved.state.is_resolved
    options = resolved.value.initial_source_options()
    assert options["seed_components"][0]["root"]["reference"] == case.resolve(amplitude).canonical_identity()
    assert validate_initial_source({**options,
        **ConservativeCellAverage().initial_projection_options()}, where="resolved") == "bootstrap_program"
    source = profile.initial_source_options()
    altered = copy.deepcopy(source)
    altered["seed_sampling"] = "cell_average"
    with pytest.raises(ValueError, match="identity and source"):
        BootstrapProgram.resolve_captured_references(value_identity=profile.to_data(),
            source_options=altered, references=(amplitude,), resolver=lambda value: value)
