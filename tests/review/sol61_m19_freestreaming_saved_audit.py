"""Offline captured transport audit @1; no PoPS import or Native approval."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import numpy as np

CONTRACT = "sol61.m19-freestreaming-saved-audit@1"
PHASES = ("initial", "accepted", "continuous", "restored", "replay")

def need(ok, message):
    if not ok:
        raise ValueError(message)

def pinned(pin):
    need(type(pin) is dict and set(pin) == {"path", "sha256"}, "file pin shape")
    raw = Path(pin["path"]).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == pin["sha256"], "file digest differs")
    return raw

def arrays(raw):
    with np.load(BytesIO(raw), allow_pickle=False) as z:
        need(len(z.files) == len(set(z.files)), "duplicate NPZ entries")
        return {k: z[k].copy() for k in z.files}

def equal(a, b, excluded=()):
    need(a.keys() == b.keys(), "payload keys differ")
    for k in a.keys() - set(excluded):
        need(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
             and a[k].tobytes() == b[k].tobytes(), "payload bits differ: " + k)

def science(value, seed, nx, nv, steps):
    need(value.dtype == np.dtype("float64") and value.shape == (1,nx,nv)
         and np.isfinite(value).all(), "state type/shape/finitude")
    x=(np.arange(nx)+.5)/nx; v=-1+2*(np.arange(nv)+.5)/nv
    dt=1/(4*nx); theta=2*np.pi/nx
    # Independent spectral polynomial applied to the retained initial wave.
    expected=seed.copy()
    for j, speed in enumerate(v):
        lam=abs(speed)*nx*(complex(np.cos(theta),-np.sign(speed)*np.sin(theta))-1)
        growth=(1+dt*lam+(dt*lam)**2/2)**steps
        expected[0,:,j]=(1+.25*speed)*(1+.1*np.sinc(1/nx)*np.real(np.exp(2j*np.pi*x)*growth))
    need(np.max(abs(value-expected)) <= 3e-12, "signed SSPRK2 Fourier equation")
    t=steps*dt; b=2*np.pi*t; h=1/nv; z=b*h
    sinc=np.sin(z)/z; first=h*(np.sin(z)-z*np.cos(z))/z**2
    phase=2*np.pi*x[:,None]-b*v[None,:]
    continuum=((1+.25*v)+.1*np.sinc(1/nx)*((1+.25*v)*sinc*np.cos(phase)+.25*first*np.sin(phase)))[None]
    need(np.mean(abs(value-continuum)) <= .65*t/nx and np.max(abs(value-seed)) >= .005,
         "continuum/evolution guard")
    for k in range(3):
        need(abs(np.sum((value-seed)*v[None,None,:]**k)*2/(nx*nv)) <= 3e-12, "moment conservation")

def audit(path, sha):
    r=json.loads(pinned({"path":str(path),"sha256":sha}))
    need(r["schema"] == "pops.m19-freestreaming-native-fixture@1", "fixture version")
    nx,nv=r["nx"],r["nv"]
    need(type(nx) is int and type(nv) is int and (nx,nv) in ((32,8),(64,12)), "case")
    need(type(r["dimension"]) is int and r["dimension"] == 2 and r["native_axes"] == {"velocity":0,"position":1}
         and r["dt"] == [1,4*nx] and r["final_time"] == [1,8]
         and r["temporal"] == "SSPRK2"
         and r["physics"] == "ddt(f)=-div((0,v*f)); periodic position; velocity flux zero"
         and r["discretization"] == "ConservativeCellAverage/FirstOrder/HLLExplicitPair(v,v)", "axes/time/physics")
    need(set(r["phases"]) == set(PHASES), "five phases required")
    states,cps={},{}
    for phase in PHASES:
        row=json.loads(pinned(r["phases"][phase])); need(row["phase"] == phase,"phase")
        states[phase]=arrays(pinned(row["state"])); cps[phase]=arrays(pinned(row["checkpoint"]))
        steps={"initial":0,"accepted":nx//2,"continuous":nx,"restored":nx//2,"replay":nx}[phase]
        need(states[phase]["time"].dtype == np.dtype("float64") and states[phase]["time"].shape == ()
             and float(states[phase]["time"]) == steps/(4*nx)
             and states[phase]["step"].dtype == np.dtype("int64") and states[phase]["step"].shape == ()
             and int(states[phase]["step"]) == steps and row["macro_step"] == steps
             and row["time_hex"] == (steps/(4*nx)).hex(), "phase clock")
        if phase != "initial":
            aux=states[phase]["velocity_coordinate_native"]
            need(aux.dtype == np.dtype("float64") and aux.shape == (nx,nv)
                 and np.max(abs(aux-(-1+2*(np.arange(nv)+.5)/nv))) <= 1e-14,"native signed Aux")
    x=(np.arange(nx)+.5)/nx; v=-1+2*(np.arange(nv)+.5)/nv
    seed=states["initial"]["population"]
    need(seed.dtype == np.dtype("float64") and seed.shape == (1,nx,nv)
         and np.max(abs(seed-((1+.1*np.sinc(1/nx)*np.cos(2*np.pi*x[:,None]))*(1+.25*v))[None])) <= 3e-12,"initial cell averages")
    for phase in PHASES[1:]:
        science(states[phase]["population"],seed,nx,nv,nx//2 if phase in ("accepted","restored") else nx)
    equal(states["accepted"],states["restored"])
    equal(states["continuous"],states["replay"])
    for a,b in (("accepted","restored"),("continuous","replay")):
        equal(cps[a],cps[b],("pops_checkpoint_manifest","pops_restart_identity"))
    return {"contract":CONTRACT,"captured_science_and_payload_bits":True,
            "native_authority_or_root_approval_verified":False}

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("receipt");p.add_argument("sha256");a=p.parse_args()
    print(json.dumps(audit(a.receipt,a.sha256),sort_keys=True))
