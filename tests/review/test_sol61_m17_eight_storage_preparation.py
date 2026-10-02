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


def grown_example():
    image,values=example()
    grown=np.full((15,18,18),77.0,dtype=np.float64);grown[:,1:17,1:17]=values
    def encode(shard,rows):
        u=lambda x:struct.pack('<Q',x);i=lambda x:struct.pack('<q',x)
        header=b'POPSCAR1'+u(2)+u(64)+u(1)+i(shard)+u(1)+u(1)+u(3)+b'gas'+u(1)
        row=u(0)+u(0)+u(0)+u(15)+i(0)+b''.join(i(x) for x in (0,15,-1,16,0,15,-1,16))
        return header+row+u(rows.size)+rows.astype('<f8').tobytes()
    return AcceptedStateStorageObservation(2,0.0,0,encode(0,grown),encode(-1,grown)),values


@pytest.mark.parametrize('mutation',('local-grown-bit','complete-valid-bit','world-envelope'))
def test_raw_evidence_persisted_before_refusal_and_no_next_run(tmp_path,mutation):
    import hashlib,json
    from tests.python.support.fan_li15_storage_capture import persist_storage_phase,guard_persisted_storage
    from tests.python.support.m16_explicit_native_capture import persisted_capture
    from tests.python.integration.runtime import test_fan_li15_full_eight_storage_runtime as fixture
    image,values=grown_example()
    validate_storage(image,values,(0.0,0),rank=0,ranks=1)
    if mutation=='local-grown-bit':image=dataclasses.replace(image,rank_local=image.rank_local[:-8]+struct.pack('<d',-912.0))
    elif mutation=='complete-valid-bit':
        offset=struct.calcsize('<8sQQQqQQ')+8+3+8+5*8+8*8+8
        data=bytearray(image.complete);data[offset+(18+1)*8:offset+(18+2)*8]=struct.pack('<d',0.0)
        image=dataclasses.replace(image,complete=bytes(data))
    else:
        def ranks2(raw):return raw[:24]+struct.pack('<Q',2)+raw[32:]
        image=dataclasses.replace(image,rank_local=ranks2(image.rank_local),complete=ranks2(image.complete))
    events=[];failures=[];bulk=(values,(0.0,0),image)
    def receipt(status):
        pins={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir() if p.is_file() and p.name!='receipt.json'}
        (tmp_path/'receipt.json').write_text(json.dumps({'status':status,'files':pins,'failures':failures},allow_nan=False))
    def save(captured):
        persist_storage_phase(None,tmp_path,'initial',captured,lambda p,i:fixture.save_phase(tmp_path,p,i))
        events.append('persisted');receipt('partial-captures')
    def refused(rows):
        failures.extend(rows);events.append('refused');receipt('capture-failed')
    with pytest.raises(ValueError):
        captured=persisted_capture(None,lambda:bulk,save,refused)
        guard_persisted_storage(None,captured,refused)
        events.append('future-run')
    assert events==['persisted','refused'] and failures
    assert (tmp_path/'initial.rank0.carriers').read_bytes()==image.rank_local
    assert (tmp_path/'initial.complete.carriers').read_bytes()==image.complete
    assert np.load(tmp_path/'initial.npy',allow_pickle=False).tobytes()==values.tobytes()
    assert json.loads((tmp_path/'initial.clock.json').read_text())==[0.0,0]
    report=json.loads((tmp_path/'receipt.json').read_text());assert report['status']=='capture-failed'
    assert len(report['files'])==6
    assert all(hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()==digest for name,digest in report['files'].items())
