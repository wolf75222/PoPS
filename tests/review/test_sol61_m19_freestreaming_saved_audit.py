"""Labelled synthetic algebra/corruption checks, never Native evidence."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

def load(name):
    s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+".py"))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load("sol61_m19_freestreaming_saved_audit")
o=load("sol61_m19_freestreaming_oracle")

@pytest.mark.parametrize("nx,nv",((32,8),(64,12)))
@pytest.mark.parametrize("steps_factor",(.5,1))
def test_source_spectral_and_original_continuum(nx,nv,steps_factor):
    steps=int(nx*steps_factor);seed=o.initial(nx,nv)
    r.science(o.discrete_fourier(nx,nv,1/(4*nx),steps),seed,nx,nv,steps)

@pytest.mark.parametrize("attack",("sign","euler","stationary","transpose","float32","nan"))
def test_synthetic_wrong_physics_rejected(attack):
    nx,nv=32,8;seed=o.initial(nx,nv)
    value=o.discrete_fourier(nx,nv,1/(4*nx),nx//2,
        sign=-1 if attack=="sign" else 1,temporal_order=1 if attack=="euler" else 2)
    if attack=="stationary":value=seed.copy()
    if attack=="transpose":value=value.swapaxes(1,2)
    if attack=="float32":value=value.astype(np.float32)
    if attack=="nan":value[0,0,0]=np.nan
    with pytest.raises(ValueError):r.science(value,seed,nx,nv,nx//2)

def test_payload_signed_zero_cache_bits():
    a={"cache":np.array([0.])};b={"cache":np.array([-0.])}
    with pytest.raises(ValueError,match="bits"):r.equal(a,b)

@pytest.mark.parametrize("text",('{"initial":1,"initial":2}','{"phases":{"accepted":1,"accepted":2}}','{"x":NaN}'))
def test_duplicate_phase_and_nonfinite_json_refused(text):
    with pytest.raises(ValueError):r.strict_json(text)

@pytest.mark.parametrize("attack",("missing-phase","bool-axis","float-dt","bool-final"))
def test_exact_receipt_parameter_types(attack):
    data=dict(nx=32,nv=8,native_axes={"velocity":0,"position":1},dt=[1,128],final_time=[1,8],phases={p:{} for p in r.PHASES})
    if attack=="missing-phase":del data["phases"]["replay"]
    elif attack=="bool-axis":data["native_axes"]["velocity"]=False
    elif attack=="float-dt":data["dt"][0]=1.
    elif attack=="bool-final":data["final_time"][0]=True
    with pytest.raises(ValueError):r.metadata(data)

def test_empty_checkpoint_pair_cannot_claim_physical_payload_bits():
    with pytest.raises(ValueError,match="incomplete"):
        r.checkpoint_projection({}, {"population":o.initial(32,8)},0,32,8)

def test_npz_truncation_and_duplicate_members_refused():
    from io import BytesIO
    import zipfile
    buf=BytesIO();np.savez(buf,a=np.ones(2));raw=buf.getvalue()
    with pytest.raises(zipfile.BadZipFile):r.arrays(raw[:-30])
    with pytest.warns(UserWarning,match="Duplicate name"):
        with zipfile.ZipFile(buf,"a") as z:z.writestr("a.npy",b"malformed duplicate")
    with pytest.raises(ValueError,match="duplicate"):r.arrays(buf.getvalue())

@pytest.mark.parametrize("nx,nv",((32,8),(64,12)))
@pytest.mark.parametrize("attack",("positive","state","clock-type","version","aux-truncated","manifest-missing"))
def test_labelled_source_checkpoint_projection(nx,nv,attack):
    # Minimal labelled Source algebra fixture, not a Native receipt/manifest seal.
    import json
    population=o.initial(nx,nv)
    cp={"pops_checkpoint_version":np.array(8,dtype=np.int64),"t":np.array(0.,dtype=np.float64),
        "macro_step":np.array(0,dtype=np.int64),"state_kinetic":population.reshape(-1).copy(),
        "auxiliary_checkpoint":np.frombuffer(b"POPSAUX2"+b"Source-only",dtype=np.uint8),
        "pops_restart_identity":np.array("Source-only-unqualified")}
    cp["pops_checkpoint_manifest"]=np.array(json.dumps({"runtime_kind":"uniform",
        "arrays":{k:{} for k in cp if k!="pops_restart_identity"}}))
    if attack=="state":cp["state_kinetic"][0]+=.1
    elif attack=="clock-type":cp["macro_step"]=np.array(False)
    elif attack=="version":cp["pops_checkpoint_version"]=np.array(7)
    elif attack=="aux-truncated":cp["auxiliary_checkpoint"]=np.array([],dtype=np.uint8)
    elif attack=="manifest-missing":cp["pops_checkpoint_manifest"]=np.array('{"runtime_kind":"uniform","arrays":{}}')
    if attack=="positive":r.checkpoint_projection(cp,{"population":population},0,nx,nv)
    else:
        with pytest.raises(ValueError):r.checkpoint_projection(cp,{"population":population},0,nx,nv)
