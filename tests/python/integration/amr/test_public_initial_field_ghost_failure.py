"""Genuine initial DSO fault; Source collection is not Native/MPI reception."""
import hashlib
import json
import math
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
    artifact=compile_resolved_plan_once(world,plan,route='initial-ghost-failure@1',compile_artifact=pops.compile)
    context=collective_call(world,lambda:artifact_execution_context(artifact))
    monkeypatch.setenv('POPS_TEST_INITIAL_GHOST_LOG',str(directory/'callback'))
    from pops.runtime._amr_bootstrap_execution import NativeAMRBootstrapConsumer
    original=NativeAMRBootstrapConsumer.finalize_bootstrap
    images=[];owners=[]
    def image(owner):
        engine=owner._engine;native_owner=engine._s
        # Raw checkpoint carriers/registries only: no Field accessor, solve, refresh or publication.
        return {'blob':bytes(native_owner.checkpoint_state_carriers()),
                'registry':native_owner.checkpoint_rank_local_carrier_manifest(),
                'field_manifest':native_owner.field_provider_checkpoint_manifest(),
                'time':native_owner.time(),'tick':native_owner.macro_step()}
    def observed(owner):
        owners.append(owner._engine)
        before=image(owner)
        try:return original(owner)
        except Exception:
            after=image(owner);images.append((before,after));raise
    monkeypatch.setattr(NativeAMRBootstrapConsumer,'finalize_bootstrap',observed)
    _,failures=collective_attempt(world,lambda:pops.bind(artifact,resources={'execution_context':context}))
    with collective_check(world):
        assert len(images)==1 and len(owners)==1, 'actual initial owner/failure boundary not observed'
        before,after=images[0]
        (directory/f'before-rank{world.rank}.bin').write_bytes(before['blob'])
        (directory/f'after-rank{world.rank}.bin').write_bytes(after['blob'])
        metadata={phase:{k:v for k,v in value.items() if k!='blob'} for phase,value in [('before',before),('after',after)]}
        proof={'schema':'sol61.initial-ghost-failure@1','rank':world.rank,'world_size':world.size,'failures':failures,'metadata':metadata,'native':{'path':native.__file__,'sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()},'component_manifest':component.component_manifest.to_data(),'artifact_identity':artifact.artifact_identity.token,'observer_only':True,'outer_bootstrap_abort_not_certified':True}
        (directory/f'proof-rank{world.rank}.json').write_text(json.dumps(proof,indent=2,default=str)+'\n')
        assert all(failures) and len(failures)==world.size
        assert all('independent initial Ghost rank-local failure' in failure[1] for failure in failures)
        assert before==after, 'initial candidate failure changed same-owner prepublication state'
        events=(directory/f'callback-rank{world.rank}.log').read_text()
        reads=[line for line in events.splitlines() if line.startswith('field-before-write ')]
        assert reads and 'tentative-write' in events
        from tests.python.support.initial_field_ghost_native_oracle import FIELD_BOUND
        for line in reads:
            row=dict(item.split('=',1) for item in line.split()[1:])
            value=float.fromhex(row['field']);time=float.fromhex(row['time'])
            assert math.isfinite(value) and abs(value-2.)<=FIELD_BOUND
            assert time==0.
            assert int(row['rank'])==world.rank and int(row['size'])==world.size
        assert ('injected-failure' in events)==(world.rank==world.size-1)
