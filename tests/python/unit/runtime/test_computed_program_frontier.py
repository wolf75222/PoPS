"""Source and host acceptance for the native Program Scalar frontier contract."""

import copy
import json
import math

import pytest

from pops.runtime._step_strategy import prepare_program_run
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import ComputedDt
from tests.python.unit.runtime.test_step_strategy import _Engine, _Native
from tests.python.integration.runtime.test_computed_program_frontier_runtime import rotation_case


def _source_fixture():
    from pops import Case, Model
    from pops.frames import Cartesian2D
    from pops.time import Program
    model = Model("frontier_regions", frame=Cartesian2D())
    state = model.state("U", components=("u",))
    block = Case("frontier_regions").block("fluid", model)
    program = Program("frontier_regions")
    return program, program.state(block[state])


def test_rotation_authors_and_emits_a_runtime_scalar_frontier_without_host_gamma():
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program

    _, _, model, program = rotation_case()
    assert program.validate()
    lowered, _ = lower_and_validate(model, facade=model)
    source = emit_cpp_program(program, model=lowered)
    assert "ctx.reached_duration(static_cast<double>(" in source
    assert "pops::Real requested_dt_" in source
    assert "ctx.dot(" in source
    assert "0.8" not in source
    assert source.index("ctx.reached_duration") < source.index("ctx.commit_many")
    assert ComputedDt.from_data(program._step_strategy.to_data()) == program._step_strategy
    detached = program.eliminate_dead_nodes().eliminate_common_subexpressions()
    assert any(value.op == "reached_duration" for value in detached._values)
    assert detached.validate()


def test_scalar_frontier_refuses_amr_instead_of_relabelling_its_interval_outputs():
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program

    _, _, model, program = rotation_case()
    lowered, _ = lower_and_validate(model, facade=model)
    with pytest.raises(NotImplementedError, match="ComputedDt.*AMR interval"):
        emit_cpp_program(program, model=lowered, target="amr_system")


@pytest.mark.parametrize("nested", (False, True))
def test_scalar_frontier_refuses_flux_rhs_inside_public_lazy_branch(nested):
    from pops.codegen.program_computed_frontier import require_computed_frontier_support
    from pops.numerics.terms import Flux
    program, q = _source_fixture()
    requested = program.requested_dt()
    initial = q.n

    def rate(P):
        return P.rhs(state=initial, terms=[Flux()])

    def arm(P):
        return P.branch(requested > 0., rate, rate) if nested else rate(P)

    selected = program.branch(requested > 0., arm, arm)
    candidate = program.value("candidate", q.n + program.dt * selected, at=q.next.point)
    program.reached_duration(.8 * requested)
    program.commit(q.next, candidate)
    program.step_strategy(ComputedDt(1.))
    assert program.validate()
    assert not any(value.op == "rhs" for value in program._values)
    with pytest.raises(NotImplementedError, match="spatial interval exchanges"):
        require_computed_frontier_support(program, target="system")


def test_frontier_requires_exactly_one_owned_scalar_and_authenticated_strategy():
    _, _, _, program = rotation_case()
    with pytest.raises(ValueError, match="only once"):
        program.reached_duration(program.requested_dt())
    other, q = _source_fixture()
    with pytest.raises((TypeError, ValueError), match="Scalar"):
        other.reached_duration(q.n)
    duration = other.requested_dt()
    with pytest.raises(ValueError, match="different Program"):
        program.reached_duration(duration)


def _host(duration=.8):
    engine = _Engine(ComputedDt(1.))
    native = _Native()
    temporal = TemporalRestartState()
    engine._temporal_restart_state = temporal
    native.step = lambda dt: (setattr(native, "t", native.t + duration),
                              setattr(native, "cursor", native.cursor + 1))
    engine.program_diagnostics = lambda: {"pops.frontier.duration": duration}
    prepared = prepare_program_run(engine)
    prepared.begin(temporal, time=0., macro_step=0)
    return engine, native, temporal, prepared


def test_program_frontier_strict_restart_keeps_requested_and_effective_duration():
    _, native, temporal, prepared = _host()
    prepared.run_step(native, t_end=.8)
    receipt = temporal.controller_state["program_frontier"]
    assert receipt["requested_duration"] == 1.0.hex()
    assert receipt["duration"] == .8.hex()
    payload = temporal.checkpoint_json(time=.8, macro_step=1)
    restored = TemporalRestartState.from_json(payload, time=.8, macro_step=1)
    assert restored.controller_state == temporal.controller_state
    for field in ("duration", "requested_duration", "reached", "rejections"):
        altered = json.loads(payload)
        row = altered["controller_state"]["program_frontier"]
        row[field] = 4 if field == "rejections" else .5.hex()
        with pytest.raises(ValueError, match="Program|program"):
            TemporalRestartState.from_json(json.dumps(altered), time=.8, macro_step=1)


def test_program_frontier_refuses_overshoot_without_accepting_temporal_state():
    _, native, temporal, prepared = _host(duration=math.nextafter(.8, math.inf))
    with pytest.raises(RuntimeError, match="crosses.*frontier"):
        prepared.run_step(native, t_end=.8)
    assert temporal.time_hex == 0.0.hex() and temporal.macro_step == 0
    assert "program_frontier" not in temporal.controller_state
    # The direct host seam owns no numerical rollback; RuntimeInstance/native tests qualify that.


@pytest.mark.parametrize("field", range(5))
def test_computed_dt_preflight_compares_policy_entry_step_and_frontier(monkeypatch, field):
    from pops import _native_collectives
    from pops.runtime import _step_strategy
    from types import SimpleNamespace
    _, native, _, prepared = _host()
    before = copy.deepcopy(native.__dict__)

    def gather(_world, row):
        changed = list(row["contract"])
        changed[field] = "foreign"
        return row, {"contract": tuple(changed), "error": None}

    monkeypatch.setattr(_step_strategy, "_attempt_world", lambda *_a, **_k: SimpleNamespace(size=2))
    monkeypatch.setattr(_native_collectives, "allgather_value", gather)
    with pytest.raises(RuntimeError, match="ComputedDt preparation differs"):
        prepared.run_step(native, t_end=.8)
    assert native.t == before["t"] and native.cursor == before["cursor"]


@pytest.mark.parametrize("field", ("missing", "duration", "requested_duration", "reached", "rejections"))
def test_program_frontier_refuses_forged_live_receipt_before_native_step(field):
    _, native, temporal, prepared = _host()
    prepared.run_step(native, t_end=.8)
    if field == "missing":
        del temporal.controller_state["program_frontier"]
    else:
        row = temporal.controller_state["program_frontier"]
        row[field] = 4 if field == "rejections" else .5.hex()
    before = (native.t, native.cursor)
    with pytest.raises(ValueError, match="Program|program"):
        prepared.run_step(native, t_end=1.6)
    assert (native.t, native.cursor) == before
