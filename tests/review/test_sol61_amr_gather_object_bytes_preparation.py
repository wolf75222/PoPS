import numpy as np
import pytest
from tests.python.support.amr_gather_bitwise_oracle import reconstruct_level
from tests.python.integration.runtime.test_amr_gather_object_bytes_runtime import build_gather_case

def archive():
    return {'patches':[{'key':(0,1,0),'components':2,'axes':[(1,1,0,2),(0,0,-1,1)],'bits':[0,0,0,0,1<<63,0,0,0,0]+[0]*9}]}

def test_signed_zero_and_uncovered_cells_have_distinct_object_bytes():
    values,coverage=reconstruct_level(archive(),0,1,3)
    assert values[0,0,1].view(np.uint64)==np.uint64(1<<63)
    assert values[0,0,0].view(np.uint64)==0
    assert coverage.sum()==1
    collapsed=values.copy();collapsed[0,0,1]=0.
    assert np.array_equal(collapsed,values) and collapsed.tobytes()!=values.tobytes()

@pytest.mark.parametrize('mutation',('overlap','outside','width'))
def test_malformed_topology_refused(mutation):
    a=archive()
    if mutation=='overlap':a['patches']*=2
    if mutation=='outside':a['patches'][0]['axes'][0]=(3,3,2,4)
    if mutation=='width':a['patches'].append(dict(a['patches'][0],components=3))
    with pytest.raises((ValueError,TypeError)):reconstruct_level(a,0,1,3)

@pytest.mark.parametrize('policy',('replicated','partitioned','empty-owner'))
def test_actual_public_layout_can_resolve_requested_policy(policy):
    import pops
    case,layout=build_gather_case(policy)
    assert layout.patch_layout.distribute_coarse==(policy!='replicated')
    plan=pops.resolve(pops.validate(case),layout=layout)
    assert plan is not None

@pytest.mark.parametrize('key,value', [('dim',1),('real',32),('shard',0),('levels',1),('ranks',1),('blocks',['Q0','Q1'])])
def test_fixture_partition_and_wire_authority_are_exact(key,value):
    from tests.python.integration.runtime.test_amr_gather_object_bytes_runtime import validate_fixture_archive
    a=dict(dim=2,real=64,shard=-1,levels=2,ranks=2,blocks=['Q0','Q1','forcing'])
    validate_fixture_archive(a,2);a[key]=value
    with pytest.raises(AssertionError):validate_fixture_archive(a,2)
