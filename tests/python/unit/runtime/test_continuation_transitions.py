"""Continuation choices are derived once; readers cannot publish provisional receipts."""
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace

import numpy as np
import pytest

import pops
from pops.codegen._compiled_artifact import CompiledPlanRecord
from pops.runtime._continuation_transitions import (
    ContinuationTransitionPlan, completed_restart_receipt, prepare_bind_continuation,
    require_resolved_continuation, committed_continuation_report,
)
from pops.runtime._checkpoint_exchanges import (
    capture_checkpoint_continuation, prepare_checkpoint_continuation,
)
from tests.python.unit.codegen.test_layout_plan_pipeline import _case
from tests.python.support.layout_plan import cartesian_grid


def _plan():
    case, _, _ = _case("continuation-obligations")
    return pops.resolve(case, layout=pops.layouts.Uniform(cartesian_grid(n=8)))


def test_resolved_and_detached_continuation_obligations_authenticate_every_event():
    plan = _plan()
    actual = require_resolved_continuation(plan)
    detached = CompiledPlanRecord.from_resolved(plan)
    assert require_resolved_continuation(detached) == actual
    assert {row["name"] for row in actual.require("restart") if row["kind"] == "state"} == {"first", "second"}
    with pytest.raises(FrozenInstanceError):
        actual._json = "{}"
    data = actual.to_data()
    del data["objects"][0]["transitions"]["restart"]
    with pytest.raises(ValueError, match="omitted"):
        ContinuationTransitionPlan(json.dumps(data))
    data = actual.to_data()
    data["objects"][0]["transitions"]["restart"]["action"] = "invent_past_values"
    with pytest.raises(ValueError, match="unsupported"):
        ContinuationTransitionPlan(json.dumps(data))
    object.__setattr__(plan, "continuation_transitions", None)
    with pytest.raises(ValueError, match="omits required"):
        plan.verify()


def test_child_scope_cannot_require_another_layouts_retained_state():
    plan = _plan()
    owner = SimpleNamespace()
    prepare_bind_continuation(owner, SimpleNamespace(artifact=SimpleNamespace(plan=plan)),
                              program=plan.time, block_names=("first",), field_names=())
    assert {row["name"] for row in owner._continuation_transition_plan.require("restart")
            if row["kind"] == "state"} == {"first"}


class _NativeMailbox:
    def __init__(self):
        self.image = b"POPSEX01" + bytes(8)
        self.validated = []

    def _checkpoint_program_exchanges(self):
        return self.image

    def _validate_checkpoint_program_exchanges(self, image):
        self.validated.append(image)
        if image != self.image:
            raise ValueError("invalid native exchange record")


def _owner():
    return SimpleNamespace(
        _continuation_transition_plan=_plan().continuation_transitions,
        _checkpoint_exchange_byte_capacity=4096,
        _s=_NativeMailbox(),
        _execution_context=SimpleNamespace(communicator=SimpleNamespace(identity="serial", handle=None)),
    )


def test_checkpoint_mailbox_preflight_is_exact_and_does_not_modify_the_owner():
    owner = _owner()
    payload = {}
    capture_checkpoint_continuation(owner, payload)
    assert prepare_checkpoint_continuation(owner, payload) == owner._s.image
    original = owner._s.image
    payload["program_exchange_state"][-1] = 1
    with pytest.raises(ValueError, match="invalid native exchange record"):
        prepare_checkpoint_continuation(owner, payload)
    assert owner._s.image == original
    payload["program_exchange_state"][-1] = 0
    payload["program_exchange_offsets"] = np.asarray([0, 17], dtype=np.int64)
    with pytest.raises(ValueError, match="offsets"):
        prepare_checkpoint_continuation(owner, payload)
    payload["program_exchange_offsets"] = np.asarray([0, 16], dtype=np.int64)
    payload["continuation_transition_plan"] = np.asarray("{}")
    with pytest.raises(ValueError, match="policies differ"):
        prepare_checkpoint_continuation(owner, payload)


def test_restart_receipt_requires_history_evidence_and_remains_unpublished():
    owner = _owner()
    data = owner._continuation_transition_plan.to_data()
    row = dict(data["objects"][0])
    row.update(kind="history", identity="history:ring", name="ring")
    data["objects"].append(row)
    owner._continuation_transition_plan = ContinuationTransitionPlan(json.dumps(data))
    owner._last_restart_identity = SimpleNamespace(token="restart:1")
    owner._s.time = lambda: .25
    owner._s.macro_step = lambda: 4
    owner._last_continuation_transition_report = {"transition": "initialization"}
    with pytest.raises(RuntimeError, match="lacks required history evidence"):
        completed_restart_receipt(owner)
    owner._last_restart_report = SimpleNamespace(histories=[{
        "name": "ring", "stored_slots": 2, "recomputed_slots": 2}])
    receipt = completed_restart_receipt(owner)
    assert receipt["objects"][-1]["action"] == "reconstruct"
    assert receipt["objects"][-1]["restored_slots"] == 4
    assert owner._last_continuation_transition_report == {"transition": "initialization"}


def test_public_receipt_refuses_provisional_restart_and_attempt():
    owner = _owner()
    owner._last_continuation_transition_report = {"transition": "initialization"}
    owner._checkpoint_restart_python_snapshot = ()
    with pytest.raises(RuntimeError, match="provisional restart"):
        committed_continuation_report(owner)
    del owner._checkpoint_restart_python_snapshot
    owner._s._step_transaction_depth = lambda: 1
    with pytest.raises(RuntimeError, match="provisional attempt"):
        committed_continuation_report(owner)
    owner._s._step_transaction_depth = lambda: 0
    assert committed_continuation_report(owner) == owner._last_continuation_transition_report


def test_historical_checkpoint_with_missing_mailbox_is_not_guessed_empty():
    owner = _owner()
    payload = {}
    capture_checkpoint_continuation(owner, payload)
    del payload["program_exchange_state"]
    with pytest.raises(ValueError, match="omits required"):
        prepare_checkpoint_continuation(owner, payload)
    assert owner._s.validated == []


def test_capture_post_gather_allocation_failure_reaches_serialization_consensus(monkeypatch):
    from pops.output import _checkpoint_collective as collective
    owner = _owner()
    stages = []
    original = collective.consensus
    def observe(topology, stage, **kwargs):
        stages.append((stage, kwargs.get("error")))
        return original(topology, stage, **kwargs)
    monkeypatch.setattr(collective, "consensus", observe)
    owner._checkpoint_exchange_byte_capacity = 0
    with pytest.raises(RuntimeError, match="resolved byte capacity"):
        capture_checkpoint_continuation(owner, {})
    assert len(stages) == 2
    assert stages[-1][0] == "accepted exchange checkpoint serialization"
    assert isinstance(stages[-1][1], RuntimeError)
