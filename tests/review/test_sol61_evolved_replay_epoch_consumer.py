"""Source-only replay consumer; metadata inputs are not Native evidence."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from pops.identity import make_identity
from pops.runtime._run_manifest import RunManifest
from tests.python.integration.runtime.test_public_evolved_original_stage import (
    authenticate_replay_continuation, checkpoint_restart_authority)


def data():
    bind=make_identity("bind",{"fixture":1})
    controls={"t_end":.01,"max_steps":1,"step_transaction":{},"output_mode":"current-directory"}
    accepted=RunManifest(bind_identity=bind,start_time=0.,start_macro_step=0,controls=controls)
    rows=[{"owner_path":[],"previous_owner_run":None,"previous_owner_lineage":None}]
    authority={"contract":"pops.evolved-stage-restart-authority@1","owners":rows,
               "restored_source_run":accepted.run_identity.token}
    epoch=make_identity("run",{"continuation":"checkpoint_restart_epoch@1",
        "source_run_identity":accepted.run_identity.to_data(),
        "owner_authorities":[{"owner_path":(),"previous_owner_run":None,"previous_owner_lineage":None}]})
    replay=RunManifest(bind_identity=bind,start_time=.01,start_macro_step=1,
                       controls=controls,continuation_identity=epoch)
    return accepted,replay,authority


def test_real_epoch_positive_old_guard_is_discriminated():
    accepted,replay,authority=data()
    assert replay.continuation_identity != accepted.run_identity # old equality is RED
    assert authenticate_replay_continuation(accepted,replay,authority)==replay.continuation_identity


@pytest.mark.parametrize("attack",["source","epoch","legacy","owner","foreign","duplicate","path"])
def test_forged_source_or_continuation_is_refused(attack):
    accepted,replay,authority=data()
    authority=deepcopy(authority)
    if attack=="source":authority["restored_source_run"]=make_identity("run",{"forged":1}).token
    if attack in ("epoch","legacy"):
        replay=RunManifest(bind_identity=replay.bind_identity,start_time=replay.start_time,
            start_macro_step=replay.start_macro_step,controls=replay.controls,
            continuation_identity=accepted.run_identity if attack=="legacy" else make_identity("run",{"false":1}))
    if attack=="owner":authority["owners"][0]["previous_owner_run"]=make_identity("run",{"prior":1}).token
    if attack=="foreign":authority["owners"][0]["previous_owner_lineage"]=make_identity("bind",{"foreign":1}).token
    if attack=="duplicate":authority["owners"]*=2
    if attack=="path":authority["owners"][0]["owner_path"]=["not-root"]
    with pytest.raises((AssertionError,ValueError)):
        authenticate_replay_continuation(accepted,replay,authority)


def test_capture_live_owners_does_not_guess_or_mutate():
    run=make_identity("run",{"owned":1})
    child=lambda:SimpleNamespace(_last_run_identity=run,_restart_lineage_identity=None)
    owner=SimpleNamespace(_last_run_identity=None,_restart_lineage_identity=None,
                          _engines={"z":child(),"a":child()})
    before=checkpoint_restart_authority(SimpleNamespace(_executor=owner))
    assert [row["owner_path"] for row in before["owners"]]==[[],["a"],["z"]]
    assert owner._last_run_identity is None and owner._engines["a"]._last_run_identity==run
