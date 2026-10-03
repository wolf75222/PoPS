"""Actual DTO plus codec; synthetic Source images are not Native receipts."""
import struct,json
from types import SimpleNamespace
import numpy as np
import pytest
from pops.runtime._state_storage_observation import AcceptedStateStorageObservation
from tests.python.support.uniform_checkpoint9_capture import persist_phase,validate_phase


def observation():
    values=np.arange(4,dtype=np.float64).reshape(1,2,2)
    grown=np.pad(values,((0,0),(1,1),(1,1)));grown[:,0,:]=-0.
    u=lambda x:struct.pack('<Q',x);i=lambda x:struct.pack('<q',x)
    def wire(shard):
        return (b'POPSCAR1'+u(2)+u(64)+u(1)+i(shard)+u(1)+u(1)+u(1)+b'q'+u(1)+
                u(0)+u(0)+u(0)+u(1)+i(0)+b''.join(i(x) for x in (0,1,-1,2,0,1,-1,2))+u(grown.size)+grown.view(np.uint64).tobytes())
    return AcceptedStateStorageObservation(2,0.0,0,wire(0),wire(-1)),{'q':values}


def test_actual_dto_roundtrip_with_signedzero_and_immediate_evidence(tmp_path):
    image,values=observation()
    persist_phase(tmp_path,'initial',0,1,image,(0.0,0),values)
    validate_phase(image,(0.0,0),values)
    meta=json.loads((tmp_path/'initial.rank0.phase.json').read_text())
    assert meta['contract']==image.contract and meta['capture_complete'] is True
    assert (tmp_path/'initial.rank0.carriers').read_bytes()==image.rank_local
    assert (tmp_path/'initial.carriers').read_bytes()==image.complete
    assert np.load(tmp_path/'initial.rank0.block0.npy').tobytes()==values['q'].tobytes()


@pytest.mark.parametrize('route',['local-grown','complete-valid','clock'])
def test_bad_phase_is_saved_before_guard_and_no_next_step(tmp_path,route):
    image,values=observation()
    local,whole=image.rank_local,image.complete
    if route=='local-grown':local=local[:-8]+struct.pack('<d',123.)
    if route=='complete-valid':values['q'][0,0,0]=99.
    bad=AcceptedStateStorageObservation(2,0.0,0,local,whole)
    clock=(1.0,0) if route=='clock' else (0.0,0)
    future=[]
    persist_phase(tmp_path,'bad',0,1,bad,clock,values)
    with pytest.raises(ValueError):
        validate_phase(bad,clock,values);future.append('next-step')
    assert not future
    assert (tmp_path/'bad.rank0.carriers').read_bytes()==local
    assert (tmp_path/'bad.rank0.complete.carriers').read_bytes()==whole
    assert (tmp_path/'bad.rank0.phase.json').exists()


def test_observation_preserved_when_later_valid_getter_fails(tmp_path):
    image,_=observation();persist_phase(tmp_path,'failed',0,1,image,(0.0,0),{})
    meta=json.loads((tmp_path/'failed.rank0.phase.json').read_text())
    assert meta['capture_complete'] is False and meta['blocks']==[]
    assert (tmp_path/'failed.rank0.carriers').read_bytes()==image.rank_local


def test_pure_reader_refuses_unpinned_and_duplicate_json(tmp_path):
    from tests.review.sol61_uniform_checkpoint9_offline import receive,strict_json
    (tmp_path/'receipt.json').write_text('{"schema":1,"schema":2}')
    with pytest.raises(ValueError,match='duplicate'):strict_json(tmp_path/'receipt.json')
    with pytest.raises(ValueError,match='external pins'):receive(tmp_path,{})
    with pytest.raises(ValueError,match='external pin'):receive(tmp_path,{'receipt.json':'0'*64})
