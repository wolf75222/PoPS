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
