"""Source orchestration through RuntimeInstance restart; Native boundary metadata-only."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from pops.identity import make_identity
from pops.runtime._runtime_instance import RuntimeInstance
from pops.runtime._run_manifest import begin_run
from pops.runtime._consumer_transaction import ConsumerCursorAuthority
from pops.output._consumer_contracts import ConsumerCursorSet

def fixture(monkeypatch, *, initial=False):
    from pops.output import _checkpoint_collective as codec
    source=None if initial else make_identity("run",{"saved":1});bind=make_identity("bind",{"owner":1})
    native=SimpleNamespace(_last_run_manifest=None,_last_run_identity=source,_restart_lineage_identity=source,bound_snapshot=SimpleNamespace(bind_identity=bind),time=lambda:0.0004,macro_step=lambda:4)
    def restore(identity):
        native._last_run_manifest=None;native._last_run_identity=identity;native._restart_lineage_identity=identity
    native._restore_checkpoint_run_identity=restore
    cursors=ConsumerCursorSet();authority=ConsumerCursorAuthority(cursors)
    publisher=SimpleNamespace(validate_diagnostic_restart_state=lambda d:d,diagnostic_restart_state=lambda:{"retained":1},restore_diagnostic_restart_state=lambda d:None)
    cache={"old":1};builder=SimpleNamespace(_geometry_cache=cache,invalidate_geometry_cache=cache.clear)
    owner=SimpleNamespace(_executor=native,_publisher=publisher,_snapshot_builder=builder,_consumer_cursor_authority=authority,_consumer_cursors=cursors)
    from pops.runtime._checkpoint_resource_budget import _producer_checkpoint_resource_budget
    owner._checkpoint_resource_budget=_producer_checkpoint_resource_budget({"checkpoint":np.frombuffer(b"x",dtype=np.uint8)},runtime_kind="uniform",authority="test-source-only")
    monkeypatch.setattr(codec,"decode_checkpoint_bytes",lambda p,b:{"runtime_consumer_diagnostics":np.asarray(json.dumps({"saved":1}))})
    monkeypatch.setattr("pops.runtime._checkpoint_manifest.checkpoint_run_identity",lambda d:source)
    fault={"after":False}
    def apply(*args,**kwargs):
        kwargs["prepare_outer_state"]()
        try:
            kwargs["after_native_apply"]("restored")
            if fault["after"]:
                fault["attempt_epoch"]=native._restart_lineage_identity
                raise ValueError("continuation publication fault")
        except BaseException:
            kwargs["rollback_after_native_apply"]();raise
        return "restored"
    monkeypatch.setattr(codec,"restore_checkpoint_payload",apply)
    owner._restart_operation=lambda:SimpleNamespace(reopen=lambda owner,path:None,restore=lambda owner,reopened:RuntimeInstance._restore_checkpoint(owner,b"x",cursors,bit_identical=True))
    return owner,native,source,fault

def run(native):return begin_run(native,t_end=.0005,step_transaction={},max_steps=1,output_dir=None)

def test_public_restart_distinguishes_replay_without_forgetting_source(monkeypatch):
    owner,native,source,fault=fixture(monkeypatch);first=run(native)
    assert RuntimeInstance.restart(owner,"metadata-only-checkpoint")=="restored"
    assert native._last_run_identity==source
    second=run(native);assert second.run_identity!=first.run_identity
    RuntimeInstance.restart(owner,"metadata-only-checkpoint");third=run(native)
    assert third.run_identity not in (first.run_identity,second.run_identity)

def test_restart_fault_restores_owner_run_and_lineage(monkeypatch):
    owner,native,source,fault=fixture(monkeypatch);first=run(native)
    before=(native._last_run_manifest,native._last_run_identity,native._restart_lineage_identity,owner._consumer_cursor_authority.snapshot(),dict(owner._snapshot_builder._geometry_cache))
    fault["after"]=True
    with pytest.raises(ValueError,match="continuation publication fault"):RuntimeInstance.restart(owner,"metadata-only-checkpoint")
    after=(native._last_run_manifest,native._last_run_identity,native._restart_lineage_identity,owner._consumer_cursor_authority.cursors,dict(owner._snapshot_builder._geometry_cache))
    assert after[:3]==before[:3] and after[3] is before[3][0] and after[4]==before[4]

def test_closed_run_remains_refused_without_restart(monkeypatch):
    from pops.runtime._runtime_consumers import RuntimeConsumerPublisher
    identity=make_identity("run",{"closed":1});publisher=object.__new__(RuntimeConsumerPublisher);publisher._closed_observer_runs={identity.token}
    monkeypatch.setattr(RuntimeConsumerPublisher,"_refuse_lost_observer_world",lambda self:None)
    monkeypatch.setattr(RuntimeConsumerPublisher,"_observer_key",lambda self,*a:None)
    with pytest.raises(RuntimeError,match="already closed run"):publisher.begin_post_commit_consumers(identity)
    assert publisher._closed_observer_runs=={identity.token}

@pytest.mark.parametrize("initial", [False, True])
def test_repeated_restart_without_run_advances_owned_epoch(monkeypatch, initial):
    owner,native,source,_fault=fixture(monkeypatch, initial=initial)
    RuntimeInstance.restart(owner,"metadata-only-checkpoint")
    first=native._restart_lineage_identity
    RuntimeInstance.restart(owner,"metadata-only-checkpoint")
    assert native._restart_lineage_identity != first
    assert native._last_run_identity == source


def test_child_authority_rollback_and_retry_are_deterministic(monkeypatch):
    owner,native,source,fault=fixture(monkeypatch)
    child=SimpleNamespace(_last_run_manifest=None,_last_run_identity=source,
                          _restart_lineage_identity=source)
    native._engines={"second-layout":child}
    old=(child._last_run_manifest,child._last_run_identity,child._restart_lineage_identity)
    fault["after"]=True
    with pytest.raises(ValueError,match="publication fault"):
        RuntimeInstance.restart(owner,"metadata-only-checkpoint")
    assert (child._last_run_manifest,child._last_run_identity,child._restart_lineage_identity)==old
    fault["after"]=False
    RuntimeInstance.restart(owner,"metadata-only-checkpoint")
    assert child._restart_lineage_identity==native._restart_lineage_identity==fault["attempt_epoch"]


def test_peer_continuation_disagreement_precedes_native_begin(monkeypatch):
    from pops.output import _checkpoint_collective as codec
    calls=[]
    executor=SimpleNamespace(**{name:(lambda *a, **k: calls.append("native")) for name in (
        "_prepare_checkpoint_restart","_begin_checkpoint_restart","_apply_checkpoint_restart",
        "_commit_checkpoint_restart","_finalize_checkpoint_restart","_rollback_checkpoint_restart")})
    owner=SimpleNamespace(_execution_context=SimpleNamespace(
        communicator=SimpleNamespace(identity="serial",handle=None)))
    original=codec.consensus
    def consensus(topology, phase, **kwargs):
        rows=original(topology,phase,**kwargs)
        if phase.endswith("outer-state preparation"):
            return rows+({**rows[0],"value":"foreign-epoch"},)
        return rows
    monkeypatch.setattr(codec,"consensus",consensus)
    with pytest.raises(ValueError,match="authority differs"):
        codec.restore_checkpoint_payload(owner,executor,b"checkpoint",bit_identical=True,
            prepare_outer_state=lambda:"owned-epoch",after_native_apply=lambda result:None,
            rollback_after_native_apply=lambda:None)
    assert calls==["native"] # preflight only; no transaction begin


def test_owner_paths_are_canonical_not_mapping_insertion_order(monkeypatch):
    epochs=[]
    for keys in (("z", "a"), ("a", "z")):
        owner,parent,source,_=fixture(monkeypatch)
        parent._engines={key:SimpleNamespace(_last_run_manifest=None,
            _last_run_identity=make_identity("run",{"layout":key}),
            _restart_lineage_identity=source) for key in keys}
        RuntimeInstance.restart(owner,"metadata-only-checkpoint")
        epochs.append(parent._restart_lineage_identity)
    assert epochs[0]==epochs[1]


@pytest.mark.parametrize("field", ["_last_run_identity", "_restart_lineage_identity"])
@pytest.mark.parametrize("invalid", ["token", make_identity("bind",{"foreign":1})])
def test_child_authority_is_typed_before_mutation(monkeypatch,field,invalid):
    owner,parent,source,_=fixture(monkeypatch)
    child=SimpleNamespace(_last_run_manifest=None,_last_run_identity=source,
                          _restart_lineage_identity=source)
    setattr(child,field,invalid)
    parent._engines={"selected":child}
    with pytest.raises(TypeError,match="exact domain"):
        RuntimeInstance.restart(owner,"metadata-only-checkpoint")
    assert parent._last_run_identity==source and getattr(child,field)==invalid


def test_nonstring_layout_and_cyclic_owner_fail_before_mutation(monkeypatch):
    owner,parent,source,_=fixture(monkeypatch)
    parent._engines={1:parent}
    with pytest.raises(TypeError,match="exact strings"):
        RuntimeInstance.restart(owner,"metadata-only-checkpoint")
    parent._engines={"cycle":parent}
    with pytest.raises(TypeError,match="cycle"):
        RuntimeInstance.restart(owner,"metadata-only-checkpoint")
    assert parent._last_run_identity==source
