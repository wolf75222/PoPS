"""Synthetic Source codec/DTO with independently declared State arities2/3."""
import struct
import numpy as np
import pytest
from pops.runtime._state_storage_observation import AcceptedStateStorageObservation
from tests.python.support.uniform_checkpoint9_capture import validate_phase,persist_phase
from tests.review.sol61_uniform_checkpoint9_offline_v3 import valid_block_join

def source_image(reverse=True):
    values={'early_transport':np.full((2,2,2),17.,dtype=np.float64),'late_transport':np.full((3,2,2),-9.,dtype=np.float64)}
    names=list(reversed(values)) if reverse else list(values)
    u=lambda n:struct.pack('<Q',n);i=lambda n:struct.pack('<q',n)
    def wire(shard):
        raw=b'POPSCAR1'+u(2)+u(64)+u(1)+i(shard)+u(1)+u(len(names))
        for name in names:raw+=u(len(name))+name.encode()
        raw+=u(len(names))
        for index,name in enumerate(names):
            v=values[name];grown=np.pad(v,((0,0),(1,1),(1,1)));grown[:,0,:]=-0.
            raw+=u(index)+u(0)+u(0)+u(v.shape[0])+i(0)+b''.join(i(x) for x in (0,1,-1,2,0,1,-1,2))+u(grown.size)+grown.view(np.uint64).tobytes()
        return raw
    return AcceptedStateStorageObservation(2,0.,0,wire(0),wire(-1)),values

@pytest.mark.parametrize('reverse',[False,True])
def test_exact_named_join_preserves_both_orders_and_raw_bytes(tmp_path,reverse):
    image,values=source_image(reverse)
    persist_phase(tmp_path,'initial',0,1,image,(0.,0),values,expected_blocks=tuple(values))
    validate_phase(image,(0.,0),values,rank=0,ranks=1)
    assert (tmp_path/'initial.carriers').read_bytes()==image.complete
    for index,value in enumerate(values.values()):assert np.load(tmp_path/('initial.rank0.block%d.npy'%index)).tobytes()==value.tobytes()

@pytest.mark.parametrize('attack',['foreign','missing','swapped-values','signedzero'])
def test_wrong_named_projection_refuses(attack):
    image,values=source_image()
    if attack=='foreign':values['foreign']=values.pop('early_transport')
    elif attack=='missing':values.pop('early_transport')
    elif attack=='swapped-values':values=dict(zip(values,reversed(tuple(values.values()))))
    else:values['early_transport'][0,0,0]=-0.
    with pytest.raises(ValueError):validate_phase(image,(0.,0),values,rank=0,ranks=1)

def test_reader_joins_npy_capture_index_to_actual_native_index():
    expected={'early_transport':2,'late_transport':3}
    meta={'blocks':list(expected),'expected_blocks':list(expected)}
    assert valid_block_join(meta,{'blocks':list(reversed(expected))},expected)==((0,1,'early_transport'),(1,0,'late_transport'))
    for names in (['early_transport'],['early_transport','foreign'],['early_transport','early_transport']):
        with pytest.raises(ValueError):valid_block_join(meta,{'blocks':names},expected)
    with pytest.raises(ValueError):valid_block_join({'blocks':list(reversed(expected)),'expected_blocks':list(expected)},{'blocks':list(expected)},expected)

@pytest.mark.parametrize('expected',[{}, {'':2}, {True:2}, {'q':True}, {'q':0}, {'q':2.0}])
def test_reader_requires_exact_nonempty_named_component_authority(expected):
    names=list(expected)
    with pytest.raises(ValueError):valid_block_join({'blocks':names,'expected_blocks':names},{'blocks':names},expected)
