"""Public initial Field/Ghost@1: genuine Native execution required; Source is preparation only."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pops
import pytest
from tests.python.support.initial_field_ghost_native_case import build,DT
from tests.python.support.initial_field_ghost_native_oracle import check,CONTRACT
from tests.python.integration.amr.test_public_evolving_accepted_halo import observe,publish,same
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.collective_checks import collective_call,collective_check

@pytest.mark.compiler
@pytest.mark.native_loader
def test_public_initial_field_fresh_before_ghost_and_positive_point(isolated_native_cache,tmp_path,record_property,monkeypatch):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native=select_native_dimension(2);world=native.mpi_world()
    with collective_check(world):
        assert native.__abi_version__==8
        for file in (pops.__file__,native.__file__):
            assert Path(file).resolve().is_relative_to(Path(sys.prefix).resolve())
    case,layout=collective_call(world,build)
    plan=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout))
    with collective_check(world):
        component,=plan.component_inputs
        evidence=component.component_manifest.signature['inferred_boundary_expression']
        assert {i['name'] for i in evidence['interfaces']}=={'ghost_boundary','accepted_initial_ghost'}
    artifact=compile_resolved_plan_once(world,plan,route=CONTRACT,compile_artifact=pops.compile)
    context=collective_call(world,lambda:artifact_execution_context(artifact))
    # Private, explicit producer observation installed before the genuine Native finalizer.
    # This wrapper neither solves nor replaces preparation; old binaries fail capability lookup.
    from pops.runtime._amr_bootstrap_execution import NativeAMRBootstrapConsumer
    original_finalize=NativeAMRBootstrapConsumer.finalize_bootstrap
    def observing_finalize(owner):
        collective_call(world,lambda:owner._engine._s._enable_field_candidate_observation(1))
        return original_finalize(owner)
    monkeypatch.setattr(NativeAMRBootstrapConsumer,'finalize_bootstrap',observing_finalize)
    def bind():return pops.bind(artifact,resources={'execution_context':context})
    runtime=collective_call(world,bind)
    directory=collective_directory(world,tmp_path/'initial-field-ghost')
    record_property('initial_field_ghost_receipt',str(directory))
    record_property('artifact_identity',artifact.artifact_identity.token)
    record_property('native_sha256',hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())
    images={};field_images={}
    def capture(phase):
        image=observe(world,runtime);publish(world,directory,phase,image)
        if phase in ('initial','accepted'):
            # Freeze the actual full producer image before every potential getter observation.
            witness=collective_call(world,runtime._executor._s._field_candidate_observations)
            from tests.python.support.field_candidate_observation import save_rank_observations
            collective_call(world,lambda:save_rank_observations(directory,phase,world.rank,witness))

        # Observe the const native manifest before the potential accessor can materialize a Field.
        before_manifest=collective_call(world,runtime._executor._s.field_provider_checkpoint_manifest)
        slots=collective_call(world,runtime.field_provider_slots)
        with collective_check(world):assert len(slots)==1
        fields=[]
        for level in range(len(image[1])):
            values=collective_call(world,lambda level=level:runtime.field_potential_level_global(slots[0],level))
            fields.append((np.asarray(values).reshape(image[1][level][1].shape).copy(),image[1][level][1].copy()))
        inspection=collective_call(world,lambda:runtime.inspect().to_dict())
        with collective_check(world):
            if world.rank==0:
                np.savez(directory/(phase+'-fields.npz'),**{f'{level}-{part}':a for level,row in enumerate(fields) for part,a in enumerate(row)})
                (directory/(phase+'-pre-field-manifest.json')).write_text(json.dumps(before_manifest,indent=2)+'\n')
                (directory/(phase+'-inspect.json')).write_text(json.dumps(inspection,indent=2,default=str)+'\n')
        images[phase]=image;field_images[phase]=fields
    capture('initial')
    report=collective_call(world,lambda:pops.run(runtime,t_end=DT,max_steps=1,console=False))
    capture('accepted')
    checkpoint=collective_call(world,lambda:runtime.checkpoint(directory/'accepted-checkpoint'))
    # Capture all raw evidence before numerical assertions, including failed oracles.
    with collective_check(world):
        if world.rank==0:
            with np.load(checkpoint,allow_pickle=False) as archive:
                accepted=json.loads(str(archive['amr_accepted_contract']))
                assert int(archive['pops_amr_checkpoint_version'])==12 and accepted['schema_version']==9
            proof={'contract':CONTRACT,'component_signature':dict(component.component_manifest.signature),
                   'checkpoint':{'path':str(checkpoint),'sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()},
                   'potential_accessor_may_materialize':True,'native_path':native.__file__,'python_path':pops.__file__,'proof_scope':'uniform screened Field, physical xmin Ghost, synchronous two levels'}
            (directory/'provenance.json').write_text(json.dumps(proof,indent=2,default=str)+'\n')
        assert report.accepted_steps==1 and report.rejected_steps==0
        assert images['initial'][2][-1][:2]==(0.,0)
        assert images['accepted'][2][-1][:2]==(DT,1)
        proofs={phase:check(images[phase][0],field_images[phase],steps) for phase,steps in (('initial',0),('accepted',1))}
        if world.rank==0:(directory/'math-proof.json').write_text(json.dumps(proofs,indent=2)+'\n')
    restarted=collective_call(world,bind)
    collective_call(world,lambda:restarted.restart(checkpoint))
    runtime=restarted;capture('reloaded')
    reload_witness=collective_call(world,runtime._executor._s._field_candidate_observations)
    with collective_check(world):
        assert reload_witness==[], 'restart must not recreate or inherit a producer witness'
        same(images['accepted'],images['reloaded'])
        for old,new in zip(field_images['accepted'],field_images['reloaded'],strict=True):
            for a,b in zip(old,new,strict=True):
                assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
