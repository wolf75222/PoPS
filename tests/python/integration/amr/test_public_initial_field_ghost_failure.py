"""Genuine initial DSO fault; Source collection is not Native/MPI reception."""
import hashlib
import json
import math
import numpy as np
from pathlib import Path
import sys
import pops
import pytest
from tests.python.support.initial_field_ghost_native_case import build
from tests.python.support.initial_ghost_failure_component import InitialFailureBoundary,load_component
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.collective_checks import collective_call,collective_attempt,collective_check
from tests.python.support.integral_state_receipts import collective_directory

@pytest.mark.compiler
@pytest.mark.native_loader
def test_public_initial_ghost_rank_fault_keeps_prepublication_owner(isolated_native_cache,tmp_path,record_property,monkeypatch):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native=select_native_dimension(2);world=native.mpi_world()
    with collective_check(world):
        assert native.__abi_version__==8
        for file in (pops.__file__,native.__file__):assert Path(file).resolve().is_relative_to(Path(sys.prefix).resolve())
    directory=collective_directory(world,tmp_path/'initial-ghost-failure')
    record_property('initial_ghost_failure_receipt',str(directory))
    # Source package is shared; every rank materializes the same authenticated component.
    component_dir=directory/'component'
    collective_call(world,lambda:load_component(component_dir) if world.rank==0 else None)
    from pops.external import load
    from pops import interfaces
    component=collective_call(world,lambda:load(component_dir/'initial-failure.pops.json').require('ghost',interface=interfaces.GhostBoundary)())
    # All ranks must enter component loading collectives consistently.
    case,layout=collective_call(world,lambda:build(boundary_composer=lambda base:InitialFailureBoundary(base,component)))
    plan=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout,components=(component,)))
    artifact=compile_resolved_plan_once(world,plan,route='initial-ghost-failure-owned-face@2',compile_artifact=pops.compile)
    context=collective_call(world,lambda:artifact_execution_context(artifact))
    monkeypatch.setenv('POPS_TEST_INITIAL_GHOST_LOG',str(directory/'callback'))
    from pops.runtime._amr_bootstrap_execution import NativeAMRBootstrapConsumer
    original=NativeAMRBootstrapConsumer.finalize_bootstrap
    images=[];owners=[];targets=[]
    def image(owner):
        engine=owner._engine;native_owner=engine._s
        # Raw checkpoint carriers/registries only: no Field accessor, solve, refresh or publication.
        return {'blob':collective_call(world,lambda:bytes(native_owner.checkpoint_state_carriers())),
                'registry':collective_call(world,native_owner.checkpoint_rank_local_carrier_manifest),
                'field_manifest':collective_call(world,native_owner.field_provider_checkpoint_manifest),
                'time':collective_call(world,native_owner.time),'tick':collective_call(world,native_owner.macro_step)}
    def observed(owner):
        owners.append(owner._engine)
        before=image(owner)
        from tests.python.support.initial_ghost_failure_selection import select_xmin_owner,require_selection_agreement
        from pops._native_collectives import allgather_value
        # Local parsing failures vote before any next collective Native phase.
        selected=collective_call(world,lambda:select_xmin_owner(before['blob'],world.size))
        rows=allgather_value(world,selected)
        agreed=collective_call(world,lambda:require_selection_agreement(rows,selected))
        targets.append(agreed)
        old_target=__import__('os').environ.get('POPS_TEST_INITIAL_GHOST_TARGET_RANK')
        try:
            with collective_check(world):monkeypatch.setenv('POPS_TEST_INITIAL_GHOST_TARGET_RANK',str(selected['target']))
            return original(owner)
        except Exception:
            after=image(owner);images.append((before,after));raise
        finally:
            with collective_check(world):
                if old_target is None:monkeypatch.delenv('POPS_TEST_INITIAL_GHOST_TARGET_RANK',raising=False)
                else:monkeypatch.setenv('POPS_TEST_INITIAL_GHOST_TARGET_RANK',old_target)
    monkeypatch.setattr(NativeAMRBootstrapConsumer,'finalize_bootstrap',observed)
    _,failures=collective_attempt(world,lambda:pops.bind(artifact,resources={'execution_context':context}))
    # Persist the real collective refusal even when bind never reaches our finalizer.
    # This receipt claims no snapshot/rollback proof and cannot satisfy the assertions below.
    from tests.python.support.initial_ghost_failure_receipt import save_early_bind_receipt
    with collective_check(world):
        save_early_bind_receipt(directory,rank=world.rank,world_size=world.size,
            failures=failures,owner_count=len(owners),image_count=len(images),target_count=len(targets),
            native={'path':native.__file__,'sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()},
            artifact_identity=artifact.artifact_identity.token,component_manifest=component.component_manifest.to_data())
    with collective_check(world):
        assert len(images)==1 and len(owners)==1, 'actual initial owner/failure boundary not observed'
        before,after=images[0]
        (directory/f'before-rank{world.rank}.bin').write_bytes(before['blob'])
        (directory/f'after-rank{world.rank}.bin').write_bytes(after['blob'])
        metadata={phase:{k:v for k,v in value.items() if k!='blob'} for phase,value in [('before',before),('after',after)]}
        proof={'schema':'sol61.initial-ghost-failure-owned-face@2','rank':world.rank,'world_size':world.size,'target_selection':targets,'failures':failures,'metadata':metadata,'native':{'path':native.__file__,'sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()},'component_manifest':component.component_manifest.to_data(),'artifact_identity':artifact.artifact_identity.token,'observer_only':True,'outer_bootstrap_abort_not_certified':True}
        (directory/f'proof-rank{world.rank}.json').write_text(json.dumps(proof,indent=2,default=str)+'\n')
        assert all(failures) and len(failures)==world.size
        assert all('independent initial Ghost rank-local failure' in failure[1] for failure in failures)
        assert before==after, 'initial candidate failure changed same-owner prepublication state'
        log=directory/f'callback-rank{world.rank}.log'
        events=log.read_text() if log.exists() else ''
        reads=[line for line in events.splitlines() if line.startswith('field-before-write ')]
        if world.rank==targets[0]['agreed']['target']:assert reads and 'tentative-write' in events
        from tests.python.support.initial_field_ghost_native_oracle import FIELD_BOUND
        for line in reads:
            row=dict(item.split('=',1) for item in line.split()[1:])
            value=float.fromhex(row['field']);time=float.fromhex(row['time'])
            assert math.isfinite(value) and abs(value-2.)<=FIELD_BOUND
            assert time==0.
            assert int(row['rank'])==world.rank and int(row['size'])==world.size
        assert ('injected-failure' in events)==(world.rank==targets[0]['agreed']['target'])
