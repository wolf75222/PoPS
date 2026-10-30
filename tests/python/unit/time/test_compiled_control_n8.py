"""N8 lot 2: the compiled control of an elaborated Advance, source side (spec 0.3 5.2, T28, T38).

An elaborated Advance lowers its default acceptance ``finite()`` as an assessment of the candidate
before its commit, and records an ``AdvanceControl``: its assessments with their origin, its retry
budget and its first proposal (SU9). The record enters the Program IR (version 30) and its
semantic identity, the generated C++ exports the attempt budget and raises the rejection of the
assessment in the phase that ``System::run_advance`` repeats, and the artifact manifest publishes
the record. ``tests/python/integration/runtime/test_compiled_control_n8_native.py`` runs it.
"""
import copy

import pytest

import pops
import examples.api_v02.models_M01_M14 as early
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.external.artifact_manifest import CompiledArtifactManifest, _advance_control_data
from pops.identity.semantic import program_semantic_data
from pops.runtime._simulation import (
    DEFAULT_MAX_STEPS, BudgetExhausted, Failure, RunFailed, Simulation,
)
from pops.time._program.control import (
    ADVANCE_ASSESSMENT_CONTRACT, ADVANCE_CONTROL_SCHEMA, AdvanceControl,
)
from pops.time._program.elaboration import Advance, Proposal, Retry, elaborate
from pops.time.solve_outcome import FailRun
from pops.time.values import _freeze_attr

BUDGET_ENTRY = 'extern "C" int pops_program_advance_max_attempts() { return %d; }'
ASSESSMENT_THROW = ('throw pops::runtime::program::StepAttemptRejected(pops::SolveStatus::'
                    'kInvalidEvaluation, "assessment", "Advance assessment \'finite\' rejected the '
                    'candidate");')


def _m01(monkeypatch, route):
    """The M01 spec built with ``route`` in place of its traced SSPRK2."""
    with monkeypatch.context() as patch:
        patch.setattr(early, "_traced_ssprk2", route)
        return early.build_m01()


def _advance(**advance):
    elaborations = []

    def route(case, block, state, flow, dt, *, name):
        elaborations.append(elaborate(Advance(early.shu_osher_ssprk2(flow=flow), **advance),
                                      case=case, state=block[state], proposal=Proposal(dt),
                                      name=name))
    return route, elaborations


def _elaborated(monkeypatch, **advance):
    route, elaborations = _advance(**advance)
    spec = _m01(monkeypatch, route)
    (elaboration,) = elaborations
    return spec, elaboration


def _source(spec):
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    return emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(
        resolved.blocks))


def _b6(case, block, state, flow, dt, *, name):
    return early._ssprk2(case, block, state, lambda p, u, point, label: flow(u), dt, name=name)


def _control(**overrides):
    values = {"assessments": (("finite", "default"),), "max_attempts": 3, "max_transforms": 0,
              "first_dt": 0.1}
    values.update(overrides)
    return AdvanceControl(**values)


# --- the control record ------------------------------------------------------------------------

def test_the_control_record_is_canonical_data():
    control = _control()
    data = control.to_data()
    assert data == {"schema": ADVANCE_CONTROL_SCHEMA,
                    "accept": [{"assess": "finite", "origin": "default"}],
                    "retry": {"max_attempts": 3, "max_transforms": 0},
                    "proposal": {"dt": (0.1).hex()}}
    again = AdvanceControl.from_data(copy.deepcopy(data))
    assert again == control and hash(again) == hash(control) and again is not control
    assert _control(max_attempts=2) != control
    with pytest.raises(AttributeError, match="immutable"):
        control.max_attempts = 4
    for broken, match in [
        (dict(data, schema="pops.advance-control@0"), "another schema"),
        (dict(data, extra=1), "another schema"),
        (dict(data, accept=[{"assess": "finite"}]), "malformed assessments"),
        (dict(data, retry={"max_attempts": 3}), "malformed retry"),
        (dict(data, proposal={"dt": 0.1}), "malformed proposal"),
        # float.fromhex reads both, but only the spelling of float.hex is canonical.
        (dict(data, proposal={"dt": "0x1.999999999999Ap-4"}), "not canonical"),
    ]:
        with pytest.raises(ValueError, match=match):
            AdvanceControl.from_data(broken)


@pytest.mark.parametrize("overrides, error", [
    ({"assessments": ()}, TypeError),
    ({"assessments": (("finite",),)}, TypeError),
    ({"assessments": (("finite", ""),)}, TypeError),
    ({"max_attempts": 0}, ValueError),
    ({"max_attempts": 2**31}, ValueError),  # the artifact exports a C int
    ({"max_attempts": True}, ValueError),
    ({"max_transforms": 1}, ValueError),
    ({"first_dt": 0.0}, ValueError),
    ({"first_dt": -0.1}, ValueError),
    ({"first_dt": float("inf")}, ValueError),
    ({"first_dt": float("nan")}, ValueError),
    ({"first_dt": 1}, ValueError),
], ids=["no-assessment", "pair", "empty-origin", "no-attempt", "c-int", "bool", "transforms",
        "zero-dt", "negative-dt", "infinite-dt", "nan-dt", "int-dt"])
def test_the_control_record_refuses_what_it_cannot_run(overrides, error):
    with pytest.raises(error):
        _control(**overrides)


# --- elaboration, IR and identity --------------------------------------------------------------

def test_the_retry_budget_reaches_the_record_the_ir_and_the_identity(monkeypatch):
    spec, elaboration = _elaborated(monkeypatch, retry=Retry(max_attempts=3))
    program = elaboration.program
    assert spec.case._time is program and program._advance_control is elaboration.control
    assert elaboration.control == AdvanceControl(
        assessments=(("finite", "default"),), max_attempts=3, max_transforms=0,
        first_dt=spec.dt)
    ir = program._serialize(include_provenance=False)
    assert ir["version"] == 30 and ir["advance_control"] == elaboration.control.to_data()
    assert program_semantic_data(program)["advance_control"] == elaboration.control.to_data()
    _, default = _elaborated(monkeypatch)
    assert default.control.max_attempts == 1
    left, right = program_semantic_data(program), program_semantic_data(default.program)
    assert left != right and dict(left, advance_control=None) == dict(right, advance_control=None)
    # The emitted Program hash, hence the artifact identity, digests the record; the node graph
    # holds no control, as it holds no step strategy.
    assert program._ir_hash() != default.program._ir_hash()
    assert program.to_graph().graph_hash == default.program.to_graph().graph_hash
    # A b6 Program has no compiled control and keeps its IR version.
    b6 = _m01(monkeypatch, _b6).case._time
    assert "advance_control" not in b6._serialize(include_provenance=False)
    assert b6._advance_control is None


def test_a_budget_beyond_a_c_int_is_refused_before_the_case_gets_a_program(monkeypatch):
    route, _ = _advance(retry=Retry(max_attempts=2**31))
    captured = {}

    def capture(case, block, state, flow, dt, *, name):
        captured["case"] = case
        route(case, block, state, flow, dt, name=name)
    with pytest.raises(ValueError, match="2\\*\\*31 - 1"):
        _m01(monkeypatch, capture)
    assert captured["case"]._time is None


def test_norm_inf_all_reads_a_field_with_an_owner(monkeypatch):
    _, elaboration = _elaborated(monkeypatch)
    program = elaboration.program
    (norm,) = [v for v in program._values if v.op == "reduce"]
    assert norm.attrs == {"kind": "norm_inf_all",
                          "component_contract": "pops.program.norm-inf-all@1"}
    with pytest.raises(ValueError, match="State/RHS"):
        program.norm_inf_all(norm)


# --- generated C++ -----------------------------------------------------------------------------

def test_the_generated_program_exports_the_budget_and_rejects_in_the_assessment_phase(
        monkeypatch):
    spec, _ = _elaborated(monkeypatch, retry=Retry(max_attempts=3))
    lines = [line.strip() for line in _source(spec).splitlines()]
    assert lines.count(BUDGET_ENTRY % 3) == 1
    assert lines.count(ASSESSMENT_THROW) == 1
    assert sum("ctx.norm_inf_all(0, " in line for line in lines) == 1
    b6 = _source(_m01(monkeypatch, _b6))
    assert "pops_program_advance_max_attempts" not in b6 and '"assessment"' not in b6
    assert "norm_inf_all" not in b6


def test_the_amr_route_refuses_the_assessment_by_name(monkeypatch):
    spec, _ = _elaborated(monkeypatch)
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    with pytest.raises(NotImplementedError, match="AMR route of the compiled control"):
        emit_cpp_program(resolved.time, target="amr_system",
                         model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))


def _forge(value, **changes):
    """Replace attributes of an authored value, as a forged IR would; None removes one."""
    attrs = dict(value.attrs)
    for key, item in changes.items():
        if item is None:
            del attrs[key]
        else:
            attrs[key] = item
    object.__setattr__(value, "attrs", _freeze_attr(attrs))


@pytest.mark.parametrize("forge, match", [
    (lambda guard, norm: _forge(guard, assessment="pops.advance-assessment@0"),
     "accepted guard|rejection contract"),
    (lambda guard, norm: _forge(guard, action=FailRun()), "accepted guard|rejection contract"),
    (lambda guard, norm: _forge(norm, component_contract=None), "all-component contract"),
], ids=["contract", "action", "norm-contract"])
def test_a_forged_assessment_is_not_emitted(monkeypatch, forge, match):
    spec, elaboration = _elaborated(monkeypatch)
    values = elaboration.program._values
    (guard,) = [v for v in values if v.op == "acceptance_guard"]
    (norm,) = [v for v in values if v.op == "reduce"]
    assert guard.attrs["assessment"] == ADVANCE_ASSESSMENT_CONTRACT
    forge(guard, norm)
    with pytest.raises(ValueError, match=match):
        _source(spec)


# --- the artifact manifest ---------------------------------------------------------------------

def test_the_manifest_publishes_the_control_additively():
    data = _control().to_data()
    manifest = CompiledArtifactManifest(model_name="transport", control=data)
    encoded = manifest.to_dict()
    assert encoded["payload"]["control"] == data
    again = CompiledArtifactManifest.from_dict(copy.deepcopy(encoded))
    assert again.to_dict() == encoded
    assert "control      : accept=[finite() (default)] max_attempts=3" in str(manifest)
    # Every other artifact keeps the wire form it had.
    plain = CompiledArtifactManifest(model_name="transport").to_dict()
    assert "control" not in plain["payload"]
    assert CompiledArtifactManifest.from_dict(plain).to_dict() == plain
    broken = copy.deepcopy(encoded)
    broken["payload"]["control"]["retry"]["max_attempts"] = 0
    with pytest.raises(ValueError):
        CompiledArtifactManifest.from_dict(broken)


def test_the_compiled_manifest_reads_the_control_of_its_program(monkeypatch):
    _, elaboration = _elaborated(monkeypatch, retry=Retry(max_attempts=2))

    class Compiled:
        def __init__(self, program):
            self.program = program

    class Wrapped:
        def __init__(self, program):
            self.program = program
    expected = elaboration.control.to_data()
    assert _advance_control_data(Compiled(elaboration.program)) == expected
    assert _advance_control_data(Compiled(Wrapped(elaboration.program))) == expected
    assert _advance_control_data(Compiled(_m01(monkeypatch, _b6).case._time)) is None
    assert _advance_control_data(Compiled(None)) is None


# --- the run surface ---------------------------------------------------------------------------

def test_the_run_surface_types():
    assert DEFAULT_MAX_STEPS == 1_000_000
    assert issubclass(BudgetExhausted, RunFailed) and issubclass(RunFailed, RuntimeError)
    with pytest.raises(TypeError, match="RuntimeInstance"):
        Simulation(object())
    with pytest.raises(TypeError, match="subclass of RunFailed"):
        Failure(RunFailed, None, {"budget": "max_steps"})
    with pytest.raises(ValueError, match="origin"):
        Failure(BudgetExhausted, "elsewhere", {"budget": "max_steps"})
    with pytest.raises(TypeError, match="RunResult"):
        BudgetExhausted(object())
