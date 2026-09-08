"""Continuation choices are derived once; readers cannot publish provisional receipts."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace

import numpy as np
import pytest

import pops
from pops.codegen._compiled_artifact import CompiledPlanRecord
from pops.runtime._continuation_transitions import (
    ContinuationTransitionPlan, completed_restart_receipt, prepare_bind_continuation,
    require_resolved_continuation, committed_continuation_report, derive_continuation_transitions,
)
from pops.runtime._checkpoint_exchanges import (
    capture_checkpoint_continuation, prepare_checkpoint_continuation,
    exchange_checkpoint_byte_capacity,
)
from tests.python.unit.codegen.test_layout_plan_pipeline import _case
from tests.python.support.layout_plan import cartesian_grid


def _plan():
    case, _, _ = _case("continuation-obligations")
    return pops.resolve(case, layout=pops.layouts.Uniform(cartesian_grid(n=8)))


@pytest.mark.parametrize("method", ("euler", "ssprk2"))
def test_exchange_capacity_consumes_the_real_resolved_program(method):
    from tests.python.integration.runtime.test_public_diffusion_matrix import _build

    case, layout, _ = _build("constant", 16, .001, method=method, transport=(.7, -.4))
    plan = pops.resolve(pops.validate(case), layout=layout)
    program = plan.time
    assert program._values and plan.blocks[0].resolved_operations.evaluations
    identity = program._ir_hash()
    capacity = exchange_checkpoint_byte_capacity(
        program, cells=(16**2,), dimension=2, rank_capacity=2, resolved_plan=plan)
    assert 16 < capacity < (1 << 63)
    program.freeze()
    assert exchange_checkpoint_byte_capacity(
        program, cells=(16**2,), dimension=2, rank_capacity=2, resolved_plan=plan) == capacity
    assert program._ir_hash() == identity
    with pytest.raises(OverflowError, match="exceeds int64"):
        exchange_checkpoint_byte_capacity(
            program, cells=(1 << 63,), dimension=2, rank_capacity=2, resolved_plan=plan)


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


@pytest.fixture(scope="module")
def joint_field_plan():
    from tests.python.unit.codegen.test_uniform_field_roles import uniform_multiphysics

    return uniform_multiphysics.__wrapped__()


def _field_projection(plan, options, *, blocks=None, bootstrap=None):
    return SimpleNamespace(
        target=plan.target, time=plan.time, blocks=plan.blocks if blocks is None else blocks,
        field_plans={name: SimpleNamespace(native_install_data=lambda value=value: deepcopy(value))
                     for name, value in options.items()},
        bootstrap_plan=plan.bootstrap_plan if bootstrap is None else bootstrap,
    )


def test_joint_field_consumers_share_only_the_exact_solved_storage(joint_field_plan):
    plan = joint_field_plan
    rows = require_resolved_continuation(plan).to_data()["objects"]
    options, = (field.native_install_data() for field in plan.field_plans.values())
    assert len(options["provider_pack"]) == 2
    assert len(options["output_route"]["component_keys"]) == 3
    assert not [row for row in rows if row["kind"] == "auxiliary"]
    assert {row["name"] for row in rows if row["kind"] == "state"} == {"electrons", "ions"}
    assert {row["identity"] for row in rows if row["kind"].startswith("field_")} == {
        kind + ":" + options["provider_slot"] for kind in ("field_value", "field_observation")}
    assert require_resolved_continuation(CompiledPlanRecord.from_resolved(plan)).to_data() == \
        require_resolved_continuation(plan).to_data()


@pytest.mark.parametrize("change", ("missing", "owner", "space", "component"))
def test_field_retention_requires_exact_output_component_coverage(joint_field_plan, change):
    options = {name: field.native_install_data() for name, field in joint_field_plan.field_plans.items()}
    if change == "missing":
        options.clear()
    else:
        output, = options.values()
        key = output["output_route"]["component_keys"][0]
        key[{"owner": "owner_qid", "space": "space_name", "component": "component"}[change]] += "_foreign"
    with pytest.raises(ValueError, match="no exact resolved output owner"):
        derive_continuation_transitions(_field_projection(joint_field_plan, options))


@pytest.mark.parametrize("change", ("contract", "producer", "slot"))
def test_shared_field_consumers_cannot_disagree_on_storage_contract(joint_field_plan, change):
    options = {name: field.native_install_data() for name, field in joint_field_plan.field_plans.items()}
    blocks = list(joint_field_plan.blocks)
    original = blocks[1]
    evidence = original.resolved_operations.to_data()
    component = evidence["provider_evidence"]["auxiliary"]["entries"][0]
    if change == "contract":
        component["contract"]["representation"] = "foreign"
    elif change == "producer":
        component["provider"]["producer"] += "_foreign"
    else:
        component["provider"]["slot"] += 1
    blocks[1] = SimpleNamespace(name=original.name, state_identities=original.state_identities,
                               resolved_operations=SimpleNamespace(to_data=lambda: deepcopy(evidence)))
    with pytest.raises(ValueError, match="conflicting component contracts"):
        derive_continuation_transitions(_field_projection(joint_field_plan, options, blocks=blocks))


def test_same_field_slot_aliases_preserve_bootstrap_and_reject_policy_conflicts(joint_field_plan):
    options, = (field.native_install_data() for field in joint_field_plan.field_plans.values())
    aliases = {"first_evaluation": options, "later_evaluation": deepcopy(options)}
    bootstrap = SimpleNamespace(actions=(SimpleNamespace(
        operation="recompute", evidence={"field_name": "later_evaluation"}),))
    projected = _field_projection(joint_field_plan, aliases, bootstrap=bootstrap)
    rows = derive_continuation_transitions(projected).to_data()["objects"]
    retained = [row for row in rows if row["kind"] in {"field_value", "field_observation", "solver_cache"}]
    assert len(retained) == 3
    assert all(row["transitions"]["initialization"]["action"] == "solve"
               for row in retained if row["kind"].startswith("field_"))
    aliases["later_evaluation"]["reaction"] = 17
    with pytest.raises(ValueError, match="conflicting resolved provider policies"):
        derive_continuation_transitions(_field_projection(joint_field_plan, aliases, bootstrap=bootstrap))


def test_distinct_slots_cannot_claim_one_retained_field_component(joint_field_plan):
    options, = (field.native_install_data() for field in joint_field_plan.field_plans.values())
    second = deepcopy(options)
    second["provider_slot"] += "_foreign"
    with pytest.raises(ValueError, match="conflicting output owners"):
        derive_continuation_transitions(_field_projection(joint_field_plan, {"first": options, "second": second}))


def test_runtime_input_fieldspace_remains_an_auxiliary_obligation(joint_field_plan):
    block = joint_field_plan.blocks[0]
    evidence = block.resolved_operations.to_data()
    for component in evidence["provider_evidence"]["auxiliary"]["entries"]:
        component["provider"]["producer"] = "runtime_input"
    projected_block = SimpleNamespace(name=block.name, state_identities=block.state_identities,
                                     resolved_operations=SimpleNamespace(to_data=lambda: deepcopy(evidence)))
    plan = derive_continuation_transitions(_field_projection(joint_field_plan, {}, blocks=(projected_block,)))
    rows = [row for row in plan.to_data()["objects"] if row["kind"] == "auxiliary"]
    assert len(rows) == 3
    assert all(row["transitions"]["initialization"]["action"] == "transfer" for row in rows)
    assert {row["validity"]["component"] for row in rows} == {"potential", "electric_x", "electric_y"}
    # Arbitrary duplicate obligations are still rejected, not globally deduplicated.
    duplicate = plan.to_data()
    duplicate["objects"].append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match="duplicate retained object"):
        ContinuationTransitionPlan(json.dumps(duplicate))


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
