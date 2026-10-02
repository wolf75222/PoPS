"""Synthetic Source codec/DTO adversaries, not Native evidence."""
import dataclasses,struct
from pathlib import Path
import numpy as np
import pytest
from pops.runtime._state_storage_observation import AcceptedStateStorageObservation
from tests.python.support.fan_li15_storage_capture import validate_storage


def wire(values,shard,*,owner=0):
    # Independent minimal POPSCAR1 encoder: one complete grown patch.
    word=lambda x:struct.pack('<Q',x)
    signed=lambda x:struct.pack('<q',x)
    bits=values.view(np.uint64).ravel()
    header=b'POPSCAR1'+word(2)+word(64)+word(1)+signed(shard)+word(1)+word(1)
    header+=word(3)+b'gas'+word(1)
    row=word(0)+word(0)+word(0)+word(15)+signed(owner)
    row+=b''.join(signed(x) for x in (0,15,0,15,0,15,0,15))
    return header+row+word(bits.size)+bits.astype('<u8').tobytes()


def example():
    values=np.arange(15*16*16,dtype=np.float64).reshape(15,16,16)
    values[0,0,0]=-0.0
    image=AcceptedStateStorageObservation(2,0.0,0,wire(values,0),wire(values,-1))
    return image,values


def test_genuine_immutable_dto_complete_wire_valid_bits():
    image,values=example()
    validate_storage(image,values,(0.0,0),rank=0,ranks=1)
    with pytest.raises(dataclasses.FrozenInstanceError):image.time=1.0


@pytest.mark.parametrize('change',('valid-bit','local-bit','trailing','truncated','clock','world','dimension'))
def test_storage_integrity_mutations_fail(change):
    image,values=example();clock=(0.0,0);ranks=1
    if change=='valid-bit':values=values.copy();values[0,0,0]=0.0
    elif change=='local-bit':image=dataclasses.replace(image,rank_local=image.rank_local[:-8]+struct.pack('<d',-7.0))
    elif change=='trailing':image=dataclasses.replace(image,complete=image.complete+b'x')
    elif change=='truncated':image=dataclasses.replace(image,complete=image.complete[:56])
    elif change=='clock':clock=(0.0001,1)
    elif change=='world':ranks=2
    elif change=='dimension':
        with pytest.raises(ValueError):dataclasses.replace(image,dimension=True)
        return
    with pytest.raises(ValueError):validate_storage(image,values,clock,rank=0,ranks=ranks)


def test_historical_v2_has_no_observation_dependency():
    from tests.python.integration.runtime import test_fan_li15_full_eight_step_public_runtime as old
    assert old.SCHEMA.endswith('@2') and old.CARRIER_CAPTURE['status']=='unavailable'
    assert 'observe_accepted_state_storage' not in Path(old.__file__).read_text()


def test_old_sdk_missing_public_observer_fails_without_private_fallback():
    from tests.python.support.fan_li15_storage_capture import observe_storage
    class OldSDK:
        calls=[]
        def checkpoint_state_carriers(self):
            self.calls.append('private')
            return b'not authority'
    old=OldSDK()
    with pytest.raises(TypeError,match='no fallback'):observe_storage(old)
    assert old.calls==[]
