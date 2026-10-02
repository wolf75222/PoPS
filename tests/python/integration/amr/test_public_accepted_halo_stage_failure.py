"""Actual Native test-hook@1; ROOT execution required, no synthetic runtime."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pops
import pytest
from pops._native_collectives import allgather_value
from pops.runtime._checkpoint_manifest import MANIFEST_KEY,IDENTITY_KEY
from tests.python.integration.amr.test_public_accepted_halo_substep_growth import build_growth,DT
from tests.python.integration.amr.test_public_evolving_accepted_halo import observe,publish,same
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.collective_checks import collective_call,collective_attempt,collective_check

def read_payload(path):
    with np.load(path,allow_pickle=False) as saved:return {k:saved[k].copy() for k in saved.files}

def same_accepted_payload(before,after):
    # Only lifecycle seals may differ after a new run was prepared. Every physical,
    # temporal, Field/Aux/history/diagnostic/member contract remains byte-exact.
    assert before.keys()==after.keys()
    for key in before:
        if key in (MANIFEST_KEY,IDENTITY_KEY):continue
        a,b=before[key],after[key]
        assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),key

def checkpoint_pair_proof(retry_path,control_path):
    retry,control=read_payload(retry_path),read_payload(control_path)
    same_accepted_payload(retry,control)
    assert int(retry['pops_amr_checkpoint_version'])==12
    assert json.loads(str(retry['amr_accepted_contract']))['schema_version']==9
    return {'schema':'pops.accepted-halo-test-failure.retry-control@1',
        'checkpoints':{label:{'path':str(Path(path).resolve()),
            'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            for label,path in (('retry',retry_path),('continuous_control',control_path))},
        'excluded_lifecycle_seals':[MANIFEST_KEY,IDENTITY_KEY],
        'lifecycle_seal_differences':[key for key in (MANIFEST_KEY,IDENTITY_KEY)
            if retry[key].tobytes()!=control[key].tobytes()]}

def receipt(engine):
    r=engine._accepted_halo_test_failure_receipt()
    return dict(version=r.request.version,phase=int(r.request.phase),block=r.request.block,level=r.request.level,
        rank=r.request.rank,requested=r.requested,reached=r.reached,consumed=r.consumed,
        before_publication=r.before_publication,local_error=r.local_error,tick=r.tick,
        armed_tick=r.armed_tick,topology_epoch=r.topology_epoch,time=r.physical_time,dt=r.dt)

@pytest.mark.compiler
@pytest.mark.native_loader
def test_public_accepted_halo_rank_local_failure_restores_before_publication(
        isolated_native_cache,tmp_path,record_property):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native=select_native_dimension(2);world=native.mpi_world()
    with collective_check(world):
        assert native.__abi_version__==8
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case,layout=collective_call(world,build_growth)
    resolved=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout))
    artifact=compile_resolved_plan_once(world,resolved,route='accepted-halo-test-failure@1',compile_artifact=pops.compile)
    context=collective_call(world,lambda:artifact_execution_context(artifact))
    def bind():return pops.bind(artifact,resources={'execution_context':context})
    runtime=collective_call(world,bind);control=collective_call(world,bind)
    directory=collective_directory(world,tmp_path/'halo-stage-failure')
    record_property('accepted_halo_stage_failure_receipt',str(directory))
    for owner in (runtime,control):
        collective_call(world,lambda owner=owner:pops.run(owner,t_end=DT,max_steps=1,console=False))
    before=observe(world,runtime);publish(world,directory,'before',before)
    original=collective_call(world,lambda:runtime.checkpoint(directory/'before-checkpoint'))
    engine=runtime._executor._s
    target=int(world.size)-1
    # Genuine Native preflight adversaries: all ranks enter the same collective
    # call and a rejected request leaves the diagnostic owner unarmed.
    invalid_requests=(dict(version=2,block=0,level=0,rank=target),
        dict(block=-1,level=0,rank=target),dict(block=1,level=0,rank=target),
        dict(block=0,level=-1,rank=target),dict(block=0,level=2,rank=target),
        dict(block=0,level=0,rank=-1),dict(block=0,level=0,rank=int(world.size)))
    preflight_failures=[]
    for arguments in invalid_requests:
        invalid=collective_call(world,lambda arguments=arguments:native._AcceptedHaloTestFailureRequest(**arguments))
        _,refusals=collective_attempt(world,lambda:engine._arm_accepted_halo_test_failure(invalid))
        preflight_failures.append(refusals)
        with collective_check(world):
            assert all(refusals)
            assert not receipt(engine)['requested']
    divergent_failures=[]
    if world.size>1:
        divergent=collective_call(world,lambda:native._AcceptedHaloTestFailureRequest(block=0,level=0,rank=int(world.rank)))
        _,divergent_failures=collective_attempt(world,lambda:engine._arm_accepted_halo_test_failure(divergent))
        with collective_check(world):
            assert all(divergent_failures)
            assert all('request differs between ranks' in row[1] for row in divergent_failures)
            assert not receipt(engine)['requested']
    request=collective_call(world,lambda:native._AcceptedHaloTestFailureRequest(block=0,level=0,rank=target))
    collective_call(world,lambda:engine._arm_accepted_halo_test_failure(request))
    armed=collective_call(world,lambda:receipt(engine))
    _,duplicate=collective_attempt(world,lambda:engine._arm_accepted_halo_test_failure(request))
    with collective_check(world):
        assert all(duplicate)
        assert receipt(engine)==armed
    _,failures=collective_attempt(world,lambda:pops.run(runtime,t_end=2*DT,max_steps=1,console=False))
    local_proof=collective_call(world,lambda:receipt(engine))
    proofs=allgather_value(world,local_proof)
    after=observe(world,runtime);publish(world,directory,'after-failure',after)
    restored=collective_call(world,lambda:runtime.checkpoint(directory/'after-failure-checkpoint'))
    with collective_check(world):
        assert len(failures)==world.size and all(failures)
        assert all(row[0] in ('RuntimeError','ValueError') for row in failures),failures
        boundary='accepted halo test failure after block-level preparation fence' if world.size==1 else 'accepted halo test failure after block-level preparation fence failed collectively'
        assert all(boundary in row[1] for row in failures),failures
        for rank,r in enumerate(proofs):
            assert r['version']==1 and r['phase']==1 and r['requested'] and r['reached'] and r['consumed'] and r['before_publication']
            assert (r['block'],r['level'],r['rank'])==(0,0,target)
            assert r['local_error']==(rank==target)
            assert r['armed_tick']==r['tick']==1 and r['time']==2*DT and r['dt']==DT
        same(before,after)
        if world.rank==0:
            a,b=read_payload(original),read_payload(restored)
            assert int(a['pops_amr_checkpoint_version'])==12
            assert json.loads(str(a['amr_accepted_contract']))['schema_version']==9
            same_accepted_payload(a,b)
            (directory/'failure-proof.json').write_text(json.dumps({'failures':failures,'receipts':proofs,
                'invalid_request_refusals':preflight_failures,'duplicate_arm_refusals':duplicate,
                'divergent_request_applicable':bool(world.size>1),'divergent_request_refusals':divergent_failures,
                'lifecycle_seal_differences':[key for key in (MANIFEST_KEY,IDENTITY_KEY) if a[key].tobytes()!=b[key].tobytes()]},indent=2)+'\n')
    # No rearming: successful retry proves consumption survived the physical rollback.
    collective_call(world,lambda:pops.run(runtime,t_end=2*DT,max_steps=1,console=False))
    collective_call(world,lambda:pops.run(control,t_end=2*DT,max_steps=1,console=False))
    retry=observe(world,runtime);reference=observe(world,control)
    publish(world,directory,'retry',retry);publish(world,directory,'continuous-control',reference)
    retry_checkpoint=collective_call(world,lambda:runtime.checkpoint(directory/'retry-checkpoint'))
    control_checkpoint=collective_call(world,lambda:control.checkpoint(directory/'continuous-control-checkpoint'))
    with collective_check(world):
        same(retry,reference)
        if world.rank==0:
            proof=checkpoint_pair_proof(retry_checkpoint,control_checkpoint)
            (directory/'retry-control-checkpoint-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    record_property('phase','after actual block-level preparation/fence; before Q publication')
