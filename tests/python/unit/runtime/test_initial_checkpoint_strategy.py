"""Initial accepted temporal checkpoints follow installed Program authority.

Source-only install probes exercise the real Uniform/AMR tails; they perform no
native numerical evolution or JIT. Root replays the public native receptions.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pops
import pytest

from pops.runtime._amr_system_program import _AmrSystemProgram
from pops.runtime._multi_layout_executor import _CompositeTemporalRestartState
from pops.runtime._system_unified_install import _SystemUnifiedInstall
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import (
    AdaptiveCFL, ErrorControlledDt, ExternalTimeGrid, FixedDt,
    GuardRole, RejectAttempt,
)
from pops.time._program.detach import detach_compiled_program


POLICIES = (
    FixedDt(.125), AdaptiveCFL(.4, max_dt=.25),
    ErrorControlledDt(dt_init=.125, rtol=.001, atol=.0001,
                      dt_min=.01, dt_max=.5, max_rejections=3),
)


def _program(policy):
    program = pops.Program("initial_checkpoint")
    if isinstance(policy, ErrorControlledDt):
        estimate = program.scalar_field("embedded_error")
        program.guard("embedded_error", estimate,
                      program.norm_inf(estimate) <= policy.atol,
                      action=RejectAttempt(), role=GuardRole.ERROR_ESTIMATE)
    program.step_strategy(policy)
    return detach_compiled_program(program)


class _InstallProbe(_SystemUnifiedInstall, _AmrSystemProgram):
    def __init__(self):
        self._temporal_restart_state = TemporalRestartState()
        self._s = SimpleNamespace(time=lambda: 0., macro_step=lambda: 0)
        self._pending_native_packages = 0
        self.frozen = False
        self.events = []
    def time(self): return 0.
    def macro_step(self): return 0
    def set_program_cadence(self, substeps, stride):
        self._cadence = (substeps, stride)
    def program_substeps(self): return self._cadence[0]
    def program_stride(self): return self._cadence[1]
    def install_program(self, path): self.events.append("program")
    def _install_program_params(self, *args): self.events.append("params")
    def _validate_install_arguments(self, *args, **kwargs): pass
    def _finalize_bind(self, snapshot):
        assert self._step_strategy is None or \
            self._temporal_restart_state._initial_strategy_declaration is not None
        self.frozen = True
        self.events.append("freeze")


@pytest.mark.parametrize("route", ("uniform", "amr"))
@pytest.mark.parametrize("policy", POLICIES)
def test_real_install_tail_prepares_initial_checkpoint_before_any_run(monkeypatch, route, policy):
    import inspect
    package = Path(pops.__file__).resolve().parent
    assert Path(inspect.getfile(_SystemUnifiedInstall)).resolve() == package / "runtime/_system_unified_install.py"
    assert Path(inspect.getfile(_AmrSystemProgram)).resolve() == package / "runtime/_amr_system_program.py"
    program = _program(policy)
    compiled = SimpleNamespace(so_path="source-install-probe.so", program=program)
    owner = _InstallProbe()
    if route == "uniform":
        from pops.runtime import _bound_snapshot
        monkeypatch.setattr(_bound_snapshot, "build_uniform_snapshot", lambda *args, **kwargs: object())
        owner._install_compiled(compiled)
        assert owner.frozen and owner.events[-1] == "freeze"
    else:
        owner._finish_program_install(compiled, compiled.so_path, None, None)
    state = owner._temporal_restart_state
    before = deepcopy(state.__dict__)
    payload = state.checkpoint_json(time=0., macro_step=0)
    assert state.__dict__ == before  # checkpoint is strictly read-only
    manifest = json.loads(payload)
    assert manifest["strategy"] == {"strategy": policy.to_data(), "controls": {}}
    assert manifest["clock"] == {"time": (0.).hex(), "macro_step": 0}
    assert manifest["transaction_stats"] == {"accepted": 0, "failed": 0, "rejected": 0}
    assert manifest["controller_state"] == {"last_accepted_dt": None}
    assert (len(manifest["event_queue"]) == 1) == isinstance(policy, ErrorControlledDt)
    restored = TemporalRestartState.from_json(payload, time=0., macro_step=0)
    queued = deepcopy(restored.event_queue)
    restored.configure_program(program.temporal_manifest(), time=0., macro_step=0, strategy=policy)
    assert restored._restored_pending and restored.event_queue == queued
    assert restored.checkpoint_json(time=0., macro_step=0) == payload


@pytest.mark.parametrize("route", ("uniform", "amr"))
def test_undeclared_program_can_bind_without_inventing_a_controller(monkeypatch, route):
    program = detach_compiled_program(pops.Program("undeclared"))
    compiled = SimpleNamespace(so_path="source-install-probe.so", program=program)
    owner = _InstallProbe()
    if route == "uniform":
        from pops.runtime import _bound_snapshot
        monkeypatch.setattr(_bound_snapshot, "build_uniform_snapshot", lambda *args, **kwargs: object())
        owner._install_compiled(compiled)
    else:
        owner._finish_program_install(compiled, compiled.so_path, None, None)
    assert owner._temporal_restart_state.strategy is None
    with pytest.raises(RuntimeError, match="requires a declared step strategy"):
        owner._temporal_restart_state.checkpoint_json(time=0., macro_step=0)


@pytest.mark.parametrize("policy", POLICIES)
def test_initial_composite_leaves_have_one_exact_controller(policy):
    program = _program(policy)
    leaves = {}
    for key in ("coarse", "fine"):
        state = TemporalRestartState()
        state.configure_program(program.temporal_manifest(), time=0., macro_step=0, strategy=policy)
        leaves[key] = state
    composite = _CompositeTemporalRestartState(leaves)
    before = deepcopy(composite.to_data())
    assert before["strategy"] == {"strategy": policy.to_data(), "controls": {}}
    for state in composite.states:
        state.checkpoint_json(time=0., macro_step=0)
    assert composite.to_data() == before


def test_pending_external_grid_has_declared_authority_but_no_invented_controls():
    policy = ExternalTimeGrid("physical_times")
    program = _program(policy)
    state = TemporalRestartState()
    state.configure_program(program.temporal_manifest(), time=0., macro_step=0, strategy=policy)
    before = deepcopy(state.__dict__)
    assert state.strategy is None and state.event_queue == []
    with pytest.raises(RuntimeError, match="requires runtime controls.*external_time_grid"):
        state.checkpoint_json(time=0., macro_step=0)
    assert state.__dict__ == before
    controls = policy.runtime_controls_data({"physical_times": (0., .125, .5)})
    state.begin_run({"strategy": policy.to_data(), "controls": controls}, time=0., macro_step=0)
    restored = TemporalRestartState.from_json(state.checkpoint_json(time=0., macro_step=0),
                                               time=0., macro_step=0)
    restored.configure_program(program.temporal_manifest(), time=0., macro_step=0, strategy=policy)
    assert restored._restored_pending
    assert restored.strategy["controls"] == controls
    wrong = policy.runtime_controls_data({"physical_times": (0., .25, .5)})
    with pytest.raises(RuntimeError, match="exact next attempt"):
        restored.begin_run({"strategy": policy.to_data(), "controls": wrong}, time=0., macro_step=0)


def test_restored_controls_are_preserved_and_bind_does_not_consume_next_attempt():
    policy = AdaptiveCFL(.4)
    program = _program(policy)
    state = TemporalRestartState()
    state.configure_program(program.temporal_manifest(), time=0., macro_step=0, strategy=policy)
    controls = policy.runtime_controls_data({"dt_min": .01, "dt_max": .1})
    state.begin_run({"strategy": policy.to_data(), "controls": controls}, time=0., macro_step=0)
    payload = state.checkpoint_json(time=0., macro_step=0)
    restored = TemporalRestartState.from_json(payload, time=0., macro_step=0)
    restored.configure_program(program.temporal_manifest(), time=0., macro_step=0, strategy=policy)
    assert restored.strategy["controls"] == controls and restored._restored_pending
    before = deepcopy(restored.__dict__)
    with pytest.raises(RuntimeError, match="exact next attempt"):
        restored.begin_run({"strategy": policy.to_data(), "controls": {}}, time=0., macro_step=0)
    assert restored.__dict__ == before
    with pytest.raises(RuntimeError, match="omits the accepted step strategy"):
        restored.configure_program(program.temporal_manifest(), time=0., macro_step=0, strategy=None)
    assert restored.__dict__ == before


def test_wrong_installed_descriptor_and_invalid_initial_controls_fail_atomically(monkeypatch):
    policy = FixedDt(.125)
    program = _program(policy)
    state = TemporalRestartState()
    state.configure_program(program.temporal_manifest(), time=0., macro_step=0, strategy=policy)
    restored = TemporalRestartState.from_json(state.checkpoint_json(time=0., macro_step=0),
                                               time=0., macro_step=0)
    before = deepcopy(restored.__dict__)
    with pytest.raises(RuntimeError, match="installed step strategy differs"):
        restored.configure_program(program.temporal_manifest(), time=0., macro_step=0,
                                   strategy=FixedDt(.25))
    assert restored.__dict__ == before
    fresh = TemporalRestartState()
    before = deepcopy(fresh.__dict__)
    monkeypatch.setattr(FixedDt, "initial_runtime_controls", lambda self: {"invented": 1.})
    with pytest.raises(ValueError, match="does not accept runtime control"):
        fresh.configure_program(program.temporal_manifest(), time=0., macro_step=0, strategy=policy)
    assert fresh.__dict__ == before


def test_unknown_initial_controls_contract_fails_before_schedule_publication(monkeypatch):
    policy = FixedDt(.125)
    program = _program(policy)
    state = TemporalRestartState()
    before = deepcopy(state.__dict__)
    monkeypatch.setattr(FixedDt, "initial_controls_contract", "unknown@2")
    with pytest.raises(TypeError, match="unsupported initial step-strategy controls contract"):
        state.configure_program(program.temporal_manifest(), time=0., macro_step=0, strategy=policy)
    assert state.__dict__ == before


def test_pending_composite_cannot_hide_different_declarations():
    states = {}
    for key in ("first", "second"):
        policy = ExternalTimeGrid(key)
        program = _program(policy)
        state = TemporalRestartState()
        state.configure_program(program.temporal_manifest(), time=0., macro_step=0, strategy=policy)
        states[key] = state
    with pytest.raises(RuntimeError, match="_initial_strategy_declaration"):
        _CompositeTemporalRestartState(states)
