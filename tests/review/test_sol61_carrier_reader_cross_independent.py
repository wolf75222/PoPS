"""Independent pure host cross-oracle; synthetic bytes are never native acceptance evidence."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

DIRECTORY=Path(__file__).parent
spec=importlib.util.spec_from_file_location("independent_reader_cross",DIRECTORY/"sol61_amr_full_carrier_offline.py")
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

@pytest.fixture(scope="module")
def cross_binary(tmp_path_factory):
    directory=tmp_path_factory.mktemp("carrier-host-cross")
    binary=directory/"cross"
    # Source-only includes from the current reader checkout. ROOT supplies the integrated codec;
    # no installed PoPS module, DSO, SDK or Kokkos headers are involved.
    subprocess.run(["c++","-std=c++20","-Wall","-Wextra","-Werror","-I"+str(DIRECTORY.parents[1]/"include"),str(DIRECTORY/"sol61_carrier_hash_cross_host.cpp"),"-o",str(binary)],check=True)
    return binary,directory

def materialize(cross_binary,mode):
    binary,directory=cross_binary;prefix=directory/mode
    subprocess.run([str(binary),str(prefix),mode],check=True)
    raw=np.frombuffer(prefix.with_suffix(".bin").read_bytes(),dtype=np.uint8).copy()
    a=c.decode(raw)
    arrays=dict(state_carriers_checkpoint=raw,n_ranks=np.array(2),n_levels=np.array(1),blocks=np.array(a["blocks"]),
      patch_boxes=np.array([[0,0,0,1,3],[0,2,0,3,3]],dtype=np.int64),dmap_0=np.array([0,1],dtype=np.int64),distribution_mode_0=np.array(mode))
    rows=[[],[]];hashes={}
    for line in prefix.with_suffix(".hashes").read_text().splitlines():
        rank,block,index,local,value=line.split();hashes[(int(rank),int(block),int(index))]=(int(local),value)
    for block,name in enumerate(a["blocks"]):
        state=np.empty((block+1,4,4),dtype=np.uint64)
        for p in a["patches"]:
            if p["key"][0]!=block:continue
            index=p["key"][2];full=np.array(p["bits"],dtype=np.uint64).reshape(block+1,6,4)
            state[:,:,index*2:index*2+2]=full[:,1:5,1:3]
            for rank in (range(2) if mode=="replicated" else (index,)):
                local,value=hashes[(rank,block,index)]
                assert c.carrier_hash(a,p,local)==value
                row=["pops.amr.rank-local-carrier-manifest@1","state",name,"0",str(local),str(index),str(block+1)]
                row += [str(v) for axis in p["axes"] for v in axis[:2]]
                row += [str(v) for axis in p["axes"] for v in axis[2:]]
                row += [value];rows[rank].append(row)
        arrays[f"state_{name}_0"]=state.view(np.float64).ravel()
    return arrays,rows

@pytest.mark.parametrize("mode",["partitioned","replicated"])
def test_actual_cpp_codec_and_exact_builder_bind_every_rank(cross_binary,mode):
    arrays,rows=materialize(cross_binary,mode)
    c.receive_carriers(arrays,rows,{"opaque-A":1,"opaque-B":2})
    assert "pops" not in sys.modules

@pytest.mark.parametrize("attack",["ghost","valid","owner","grown","dtype","rank","truncate","float32-high-bits"])
def test_inner_resealed_archive_refused_against_external_cpp_hashes(cross_binary,attack):
    arrays,rows=materialize(cross_binary,"partitioned")
    raw=arrays["state_carriers_checkpoint"].copy()
    # Locate first row independently from the emitted fixed header and name lengths.
    data=raw.tobytes();at=56
    for _ in range(2):size=int.from_bytes(data[at:at+8],"little");at+=8+size
    at+=8
    def setword(offset,value):raw[offset:offset+8]=np.frombuffer(int(value).to_bytes(8,"little",signed=value<0),dtype=np.uint8)
    if attack=="ghost":raw[at+14*8]^=1
    elif attack=="valid":raw[at+(14+5)*8]^=1
    elif attack=="owner":setword(at+4*8,1)
    elif attack=="grown":setword(at+7*8,-2);setword(at+8*8,1) # same extent, shifted ghost bounds
    elif attack=="dtype":raw=raw.astype(np.int16)
    elif attack=="rank":rows.reverse()
    elif attack=="truncate":raw=raw[:-1]
    else:setword(16,32)
    # The inner file can be resealed; independently obtained CPP registry hashes stay immutable.
    arrays["state_carriers_checkpoint"]=raw
    with pytest.raises(ValueError):c.receive_carriers(arrays,rows,{"opaque-A":1,"opaque-B":2})

@pytest.mark.parametrize("mode",["partitioned","replicated"])
def test_rank_registry_omission_and_component_hash_mismatch_refused(cross_binary,mode):
    arrays,rows=materialize(cross_binary,mode)
    rows[1].pop()
    with pytest.raises(ValueError):c.receive_carriers(arrays,rows,{"opaque-A":1,"opaque-B":2})
