"""Engineering Native gather-v2 witness: no evolution/scientific qualification.

Reuse the public evolved AMR fixture, then write accepted valid storage through
its existing engineering setter. No solver, topology or getter implementation is
substituted. Exact topology assertions fail closed when a requested case is absent.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pops,pytest
from pops.amr import PatchLayout
from pops.layouts import AMR
from tests.python.support.evolved_stage_amr import build
from tests.python.support.amr_gather_bitwise_oracle import decode,reconstruct_level
from tests.python.support.collective_checks import collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.m16_explicit_native_capture import retain_provenance
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once


def build_gather_case(policy):
    case,base=build(8,2)
    patches=PatchLayout(distribute_coarse=policy!='replicated',coarse_max_grid=4 if policy=='partitioned' else 8)
    layout=AMR(**{k:getattr(base,k) for k in ('grid','hierarchy','tagging','regrid','transfer','execution','embedded_boundary','load_balance','tagger','clustering','reflux')},patch_layout=patches)
    return case,layout

def validate_fixture_archive(archive,world_size):
    assert archive['dim']==2 and archive['real']==64 and archive['shard']==-1
    assert archive['levels']==2 and archive['ranks']==world_size
    assert archive['blocks']==['Q0','Q1','forcing'], 'exact fixture block partition required'

@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize('policy',('replicated','partitioned','empty-owner'))
def test_installed_amr_gather_preserves_object_bytes(policy,tmp_path,record_property,isolated_native_cache):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    from pops._native_collectives import allgather_value
    native=select_native_dimension(2);world=native.mpi_world()
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert world.size in (1,2)
    case,layout=collective_call(world,lambda:build_gather_case(policy))
    plan=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout,compile_options={"model_source_policy":"require"}))
    artifact=compile_resolved_plan_once(world,plan,route='AMR gather object bytes/'+policy,compile_artifact=pops.compile)
    directory=collective_directory(world,tmp_path/('amr-gather-'+policy));record_property('amr_gather_directory',str(directory))
    collective_call(world,lambda:retain_provenance(artifact,plan,native,directory,blocks=('Q0','Q1','forcing'),require_model_sources=True) if world.rank==0 else None)
    runtime=collective_call(world,lambda:pops.bind(artifact,resources={'execution_context':artifact_execution_context(artifact)}))
    engine=runtime._executor._s
    clock_before=collective_call(world,lambda:(runtime.time(),runtime.macro_step(),engine.checkpoint_temporal_relations()))
    original=collective_call(world,engine.checkpoint_state_carriers)
    collective_call(world,lambda:(directory/('initial-rank%d.carriers'%world.rank)).write_bytes(original))
    archive=decode(np.frombuffer(original,dtype=np.uint8))
    with collective_check(world):validate_fixture_archive(archive,world.size)
    # All destinations receive the same domain-sized array; existing write_field
    # writes only this rank's real valid Fabs. Ghost storage is not refreshed.
    for b,name in enumerate(archive['blocks']):
        for level in range(2):
            n=8*2**level;old,coverage=reconstruct_level(archive,b,level,n)
            values=(np.arange(old.size,dtype=np.float64).reshape(old.shape)%31-15)/8
            values[:,::2,::2]=-0.
            collective_call(world,lambda name=name,level=level,values=values:engine.set_block_level_state(name,level,values))
    raw=collective_call(world,engine.checkpoint_state_carriers)
    collective_call(world,lambda:(directory/('written-rank%d.carriers'%world.rank)).write_bytes(raw))
    written=decode(np.frombuffer(raw,dtype=np.uint8))
    with collective_check(world):validate_fixture_archive(written,world.size)
    files={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ('initial-rank%d.carriers'%world.rank,'written-rank%d.carriers'%world.rank)};topologies=[]
    for b,name in enumerate(written['blocks']):
        for level in range(2):
            n=8*2**level;expected,covered=reconstruct_level(written,b,level,n)
            actual=np.asarray(collective_call(world,lambda name=name,level=level:engine.block_level_state_global(name,level))).reshape(expected.shape)
            path=directory/('rank%d-block%d-level%d.npy'%(world.rank,b,level))
            collective_call(world,lambda path=path,actual=actual:np.save(path,actual,allow_pickle=False))
            with collective_check(world):
                assert actual.dtype==np.float64 and actual.shape==expected.shape
                assert actual.tobytes()==expected.tobytes(),'getter must match native valid bits; holes remain +0'
                assert np.any(expected.view(np.uint64)==np.uint64(1<<63)),'real -0 witness required'
                if level==1:assert np.any(covered==0) and np.any(covered==1),'partial fine topology required'
            files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            topologies.append({'block':name,'level':level,'holes':int(np.sum(covered==0))})
    owners=collective_call(world,lambda:allgather_value(world,engine.coarse_local_boxes()))
    with collective_check(world):
        coarse=[p for p in written['patches'] if p['key'][1]==0]
        if policy=='replicated':assert all(p['owner']==-1 for p in coarse)
        else:assert all(p['owner']>=0 for p in coarse)
        if world.size==2 and policy=='empty-owner':assert 0 in owners and max(owners)>0
        if world.size==2 and policy=='partitioned':assert all(x>0 for x in owners)
    clock_after=collective_call(world,lambda:(runtime.time(),runtime.macro_step(),engine.checkpoint_temporal_relations()))
    with collective_check(world):assert clock_before==clock_after
    receipt={'clock_before':clock_before,'clock_after':clock_after,'schema':'pops.amr-gather-object-bytes-engineering@1','policy':policy,'rank':world.rank,'size':world.size,'files':files,'coarse_local_boxes_by_rank':owners,'topology':topologies,'root_seal':False,'scope':'engineering valid storage setter/getter; no evolution, rollback or Ghost readiness'}
    collective_call(world,lambda:(directory/('receipt-rank%d.json'%world.rank)).write_text(json.dumps(receipt,sort_keys=True,allow_nan=False)+'\n'))
