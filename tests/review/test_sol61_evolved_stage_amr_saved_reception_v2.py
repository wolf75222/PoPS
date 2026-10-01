"""Synthetic @2 wire frontier only; never real native output or ROOT seals."""
import importlib.util
from pathlib import Path
import struct
import sys
from copy import deepcopy
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("carrier_offline_test", Path(__file__).with_name("sol61_amr_full_carrier_offline.py"))
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


def synthetic():
    words = [2,64,2,-1,1,1]
    def w(value):
        return struct.pack("<Q", value % 2**64)
    bits = np.arange(16, dtype=np.float64).view(np.uint64)
    # One valid 2x2 patch, grown 4x4, component-major with x fastest.
    blob = b"POPSCAR1" + b"".join(w(v) for v in words) + w(2)+b"Q0" + w(1)
    blob += b"".join(w(v) for v in [0,0,0,1,0,0,1,-1,2,0,1,-1,2,16,*bits.tolist()])
    arrays = dict(state_carriers_checkpoint=np.frombuffer(blob,dtype=np.uint8).copy(), n_ranks=np.array(2), n_levels=np.array(1),
        blocks=np.array(["Q0"]), patch_boxes=np.array([[0,0,0,1,1]],dtype=np.int64), dmap_0=np.array([0],dtype=np.int64),
        distribution_mode_0=np.array("partitioned"), state_Q0_0=np.arange(16,dtype=np.float64).reshape(4,4)[1:3,1:3].ravel())
    archive=c.decode(arrays["state_carriers_checkpoint"]); patch=archive["patches"][0]
    row=["pops.amr.rank-local-carrier-manifest@1","state","Q0","0","0","0","1","0","1","0","1","-1","2","-1","2",c.carrier_hash(archive,patch,0)]
    return arrays,[[row],[]]


def test_synthetic_full_storage_maps_valid_bits_and_rank_ownership():
    arrays,rows=synthetic()
    archive=c.receive_carriers(arrays,rows,{"Q0":1})
    assert len(archive["patches"][0]["bits"]) == 16
    assert "pops" not in sys.modules


@pytest.mark.parametrize("attack",("magic","truncated","trailing","shard","owner","geometry","duplicate","width","ghostbit","validbit","hash","missingrank","wrongrank","components"))
def test_synthetic_carrier_frontier_refuses(attack):
    arrays,rows=synthetic(); data=bytearray(arrays["state_carriers_checkpoint"].tobytes())
    def setword(offset,value):
        data[offset:offset+8]=struct.pack("<Q",value % 2**64)
    # header 56 bytes, name length/name 10, patch count 8, row starts74.
    if attack=="magic": data[7]=ord("2")
    elif attack=="truncated": data=data[:-1]
    elif attack=="trailing": data+=b"x"
    elif attack=="shard": setword(32,0)
    elif attack=="owner": setword(74+32,2)
    elif attack=="geometry": setword(74+5*8,1)
    elif attack=="duplicate": setword(66,2); data+=data[74:]
    elif attack=="width": setword(16,32)
    elif attack=="ghostbit": data[74+14*8]^=1
    elif attack=="validbit": data[74+(14+5)*8]^=1
    elif attack=="hash": rows[0][0][-1]="foreign"
    elif attack=="missingrank": rows.pop()
    elif attack=="wrongrank": rows.reverse()
    else: setword(74+24,2)
    arrays["state_carriers_checkpoint"]=np.frombuffer(data,dtype=np.uint8).copy()
    with pytest.raises(ValueError): c.receive_carriers(arrays,rows,{"Q0":1})


def test_float32_high_bits_rejected_and_signed_zero_preserved_in_hash():
    arrays,_=synthetic(); archive=c.decode(arrays["state_carriers_checkpoint"])
    patch=archive["patches"][0]; before=c.carrier_hash(archive,patch,0)
    patch["bits"]=(0x8000000000000000,*patch["bits"][1:])
    assert c.carrier_hash(archive,patch,0)!=before


def test_versioned_reader_does_not_upcast_historical_contract():
    directory=Path(__file__).parent
    old=(directory/"sol61_evolved_stage_amr_saved_reception.py").read_text()
    new=(directory/"sol61_evolved_stage_amr_saved_reception_v2.py").read_text()
    assert '== 11' in old and '== 12' in new
    assert 'owner-pins@2' in new and 'root-approval@2' in new and 'full-carriers@2' in new
    assert 'receipt["carrier_registry"]' in new and 'state_carriers_checkpoint' in new



def test_v2_checkpoint_requires_full_image_and_preserves_old_history_negatives():
    oldspec=importlib.util.spec_from_file_location("historical_checkpoint_source_tests",Path(__file__).with_name("test_sol61_evolved_stage_amr_saved_reception.py"))
    old=importlib.util.module_from_spec(oldspec); oldspec.loader.exec_module(old)
    newspec=importlib.util.spec_from_file_location("new_checkpoint_reader",Path(__file__).with_name("sol61_evolved_stage_amr_saved_reception_v2.py"))
    new=importlib.util.module_from_spec(newspec); newspec.loader.exec_module(new)
    arrays,images,masks,identities = old.source_only_checkpoint()
    arrays["pops_amr_checkpoint_version"]=np.array(12)
    with pytest.raises(ValueError,match="full carrier image missing"):
        new.checkpoint(old.source_only_envelope(arrays),"accepted",images,8,2,2,identities,"SOURCE_ONLY")

    def word(value): return struct.pack("<Q",int(value)%2**64)
    blob=b"POPSCAR1"+b"".join(word(v) for v in (2,64,2,-1,2,3))
    for name in arrays["blocks"]:
        blob+=word(len(name))+name.encode()
    blob+=word(6)
    rows=[[],[]]
    for block,name in enumerate(arrays["blocks"]):
        components=3 if name=="forcing" else 1
        for level in (0,1):
            box=arrays["patch_boxes"][level]
            _,xlo,ylo,xhi,yhi=map(int,box)
            size=8*2**level
            valid=arrays[f"state_{name}_{level}"].reshape(components,size,size)[:,ylo:yhi+1,xlo:xhi+1]
            full=np.pad(valid,((0,0),(1,1),(1,1)),constant_values=17.)
            words=[block,level,0,components,0,xlo,xhi,xlo-1,xhi+1,ylo,yhi,ylo-1,yhi+1,full.size,*full.ravel().view(np.uint64).tolist()]
            blob+=b"".join(word(v) for v in words)
    arrays["state_carriers_checkpoint"]=np.frombuffer(blob,dtype=np.uint8).copy()
    archive=c.decode(arrays["state_carriers_checkpoint"])
    for patch in archive["patches"]:
        block,level,index=patch["key"]
        row=["pops.amr.rank-local-carrier-manifest@1","state",archive["blocks"][block],str(level),"0","0",str(patch["components"])]
        row += [str(v) for axis in patch["axes"] for v in axis[:2]]
        row += [str(v) for axis in patch["axes"] for v in axis[2:]]
        row += [c.carrier_hash(archive,patch,0)]
        rows[0].append(row)
    new.checkpoint(old.source_only_envelope(arrays),"accepted",images,8,2,2,identities,"SOURCE_ONLY")
    c.receive_carriers(arrays,rows,{"Q0":1,"Q1":1,"forcing":3})
    arrays["history_T0_level_0_0"]=arrays["history_T0_level_0_0"].copy()
    arrays["history_T0_level_0_0"][0]+=1e-3
    with pytest.raises(ValueError,match="history slot"):
        new.checkpoint(old.source_only_envelope(arrays),"accepted",images,8,2,2,identities,"SOURCE_ONLY")
