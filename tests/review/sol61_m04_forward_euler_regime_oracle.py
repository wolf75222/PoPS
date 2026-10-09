"""Independent M04 stencil/mode audit; no PoPS import or native execution."""
from __future__ import annotations

import argparse
import cmath
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

# Owner-published historical receipt pin from evidence/83b2b12/manifest.json.
# This is not a seal calculated from a new positive campaign by this oracle.
HISTORICAL_RECEIPT_SHA = "8aab308a08c2a6c6c600db234fc89f3ef37421d9269e1355aea4f12e9b3b1355"
HISTORICAL_NATIVE_SHA = "220b48d3264f413aa5ccd8bba58a8b0d1a6b31bf5bade6faac904b1b7ef8cc87"
GRIDS = (32, 64, 128)
END, A, D, AMPLITUDE, SAFETY, COURANT = .1, 1., .01, .2, .9, 1./8.
GUARD = .7
CAPS = {32:.009, 64:.0048, 128:.0024}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def step(n, policy):
    require(type(n) is int and n>0, "positive integer mesh required")
    combined = SAFETY/(abs(A)*n+2*D*n*n)
    if policy == "combined_bound":
        return combined
    require(policy == "fixed_courant", "unknown declared step policy")
    return min(combined, COURANT/(abs(A)*n))


def schedule(dt):
    require(math.isfinite(dt) and dt>0, "positive finite duration required")
    durations=[]
    time=0.
    while time<END:
        h=min(dt, END-time)
        durations.append(h)
        time+=h
    return durations


def cell_means(n, time, *, velocity=A, diffusivity=D):
    # Integrate the continuous cosine over each cell; no imported author oracle.
    k=2*math.pi
    decay=math.exp(-k*k*diffusivity*time)
    return np.array([1+AMPLITUDE*n*decay/k*(math.sin(k*((i+1)/n-velocity*time))
                     -math.sin(k*(i/n-velocity*time))) for i in range(n)])


def stencil(initial, dt, *, velocity=A, diffusivity=D, temporal="forward_euler"):
    q=np.array(initial,dtype=float,copy=True)
    n=len(q)
    def derivative(value):
        upstream=np.roll(value,1 if velocity>=0 else -1)
        return (-abs(velocity)*n*(value-upstream)
                +diffusivity*n*n*(np.roll(value,1)-2*value+np.roll(value,-1)))
    require(temporal in {"forward_euler","ssprk2"}, "unknown temporal counter-equation")
    for h in schedule(dt):
        first=q+h*derivative(q)
        q=first if temporal=="forward_euler" else .5*q+.5*(first+h*derivative(first))
    return q


def mode(n, dt, *, velocity=A, diffusivity=D, temporal="forward_euler"):
    # Construct the amplification from the three actual stencil weights.
    # The incoming direction is encoded explicitly, independent of the author's
    # eigenvalue expression. A mode product is a mathematical reference only.
    theta=2*math.pi/n
    factor=1.+0.j
    for h in schedule(dt):
        c,r=abs(velocity)*n*h,diffusivity*n*n*h
        upstream=cmath.exp(-1j*math.copysign(theta,velocity))
        g=(c+r)*upstream+(1-c-2*r)+r/upstream
        require(temporal in {"forward_euler","ssprk2"}, "unknown temporal counter-equation")
        factor*=g if temporal=="forward_euler" else .5+.5*g*g
    mean_factor=math.sin(math.pi/n)/(math.pi/n)
    return np.array([1+AMPLITUDE*mean_factor*(cmath.exp(1j*theta*(i+.5))*factor).real
                     for i in range(n)])


def prediction(policy, grids=GRIDS):
    rows=[]
    for n in grids:
        dt=step(n,policy)
        exact=cell_means(n,END)
        numerical=mode(n,dt)
        iterative=stencil(cell_means(n,0.),dt)
        mismatch=float(np.max(abs(numerical-iterative)))
        require(mismatch<=3.e-12,"independent stencil and mode disagree")
        error=math.fsum(float(v) for v in abs(numerical-exact))/n
        leading_space=A/(2*n)
        leading_time=-A*A*math.fsum(h*h for h in schedule(dt))/(2*END)
        rows.append(dict(n=n,dt=dt,courant=abs(A)*n*dt,
                         combined_frequency_product=dt*(abs(A)*n+2*D*n*n),
                         steps=len(schedule(dt)),l1=error,stencil_mode_linf=mismatch,
                         leading_added_diffusivity_space=leading_space,
                         leading_added_diffusivity_time=leading_time,
                         leading_added_diffusivity_net=leading_space+leading_time))
    orders=[math.log2(a["l1"]/b["l1"]) for a,b in zip(rows[:-1],rows[1:],strict=True)]
    return dict(status="mathematical_prediction_only",native=False,step_policy=policy,rows=rows,
                observed_orders=orders,order_guard=GUARD,order_guard_met=all(p>=GUARD for p in orders),
                formula_limitation="leading modified-equation terms explain cancellation; exact stencil decides errors")


def archive(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as packed:
        names=[row.filename for row in packed.infolist()]
        require(len(names)==len(set(names)) and sum(row.file_size for row in packed.infolist())<=8*1024*1024,
                "duplicate/oversized historical arrays")
    with np.load(io.BytesIO(raw),allow_pickle=False) as packed:
        return {name:packed[name].copy() for name in packed.files}


def historical(directory):
    raw=(directory/"receipt.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest()==HISTORICAL_RECEIPT_SHA,"historical external receipt pin mismatch")
    receipt=json.loads(raw)
    require(receipt["status"]=="failed" and receipt["resolutions"]==list(GRIDS)
            and receipt["native_sha256"]==HISTORICAL_NATIVE_SHA
            and receipt["diffusion_tensor"]==[[D,0.],[0.,0.]]
            and receipt["method"]=="combined first-order Rusanov+two-point diffusion, ForwardEuler"
            and receipt["criteria"]["minimum_observed_order"]==GUARD,
            "historical method/physics/guard changed")
    rows=[]
    for n,record in zip(GRIDS,receipt["records"],strict=True):
        require(record["cells"]==[n,n] and record["fixed_dt"]==step(n,"combined_bound"),"historical step changed")
        require(record["saved_state"]==f"state_{n}.npz","historical state path changed")
        path=directory/record["saved_state"]
        require(path.is_file() and not path.is_symlink() and path.stat().st_size<=8*1024*1024,
                "missing/invalid actual historical state")
        data=path.read_bytes()
        require(hashlib.sha256(data).hexdigest()==record["saved_state_sha256"],"historical state external SHA mismatch")
        saved=archive(data)
        initial,actual=saved["initial"],saved["final"]
        require(initial.shape==actual.shape==(1,n,n) and initial.dtype==actual.dtype==np.dtype("float64")
                and np.isfinite(initial).all() and np.isfinite(actual).all(),"historical physical state invalid")
        require(saved["time"].item()==END and saved["dt"].item()==step(n,"combined_bound")
                and saved["velocity"].item()==A and saved["diffusivity"].item()==D
                and saved["transverse_diffusivity"].item()==0.,"historical physical parameters changed")
        require(float(np.max(abs(initial-cell_means(n,0.))))<=1.e-12,"initial physical means changed")
        iterative=stencil(initial[0,0],record["fixed_dt"])
        fourier=mode(n,record["fixed_dt"])
        errors={"stencil":float(np.max(abs(actual-iterative))),"mode":float(np.max(abs(actual-fourier)))}
        require(max(errors.values())<=3.e-12,"actual historical states do not realize ForwardEuler")
        exact=cell_means(n,END)
        l1=math.fsum(float(value) for value in abs(actual-exact).flat)/(n*n)
        require(abs(l1-record["density_l1"])<=1.e-14 and l1<=CAPS[n],"historical continuous L1 differs")
        mass=abs(math.fsum(float(value) for value in (actual-initial).flat)/(n*n))
        y_defect=float(np.max(abs(actual-actual[:,:1,:])))
        require(mass<=2.e-11 and y_defect<=2.e-11 and actual.min()>=.7999999999
                and actual.max()<=1.2000000001 and record["accepted_steps"]==len(schedule(record["fixed_dt"])),
                "historical conservation/support/maximum-principle/step-count guard differs")
        wrong={name:float(np.max(abs(actual-mode(n,record["fixed_dt"],**kwargs))))
               for name,kwargs in dict(ssprk2=dict(temporal="ssprk2"),reversed_advection=dict(velocity=-A),
                                       omitted_diffusion=dict(diffusivity=0.)).items()}
        require(all(value>1.e-5 for value in wrong.values()),"counter-equation fails to distinguish actual temporal/physical law")
        rows.append(dict(n=n,state_sha256=record["saved_state_sha256"],l1=l1,actual_errors=errors,
                         mass_defect=mass,y_invariance=y_defect,
                         wrong_equation_linf=wrong,steps=record["accepted_steps"]))
    orders=[math.log2(a["l1"]/b["l1"]) for a,b in zip(rows[:-1],rows[1:],strict=True)]
    require(orders[0]<GUARD and max(abs(a-b) for a,b in zip(orders,receipt["observed_orders"],strict=True))<1.e-12,
            "failed historical guard/result changed")
    return dict(status="historical_failure_received",receipt_sha256=HISTORICAL_RECEIPT_SHA,
                native_sha256=HISTORICAL_NATIVE_SHA,native_execution_here=False,rows=rows,observed_orders=orders,
                historical_guard_met=False,qualification="old failed CPU1 Dim2 receipt only; no new native qualification")


def receive_control(directory, receipt_sha, native_sha):
    require(type(receipt_sha) is str and len(receipt_sha)==64 and type(native_sha) is str and len(native_sha)==64,
            "external owner receipt/native pins are required")
    raw=(directory/"receipt.json").read_bytes()
    require(hashlib.sha256(raw).hexdigest()==receipt_sha,"external control receipt SHA mismatch")
    receipt=json.loads(raw)
    require(receipt["schema_version"]==3 and receipt["status"]=="passed"
            and receipt["resolutions"]==list(GRIDS) and receipt["native_sha256"]==native_sha
            and receipt["temporal_method"]=="forward_euler"
            and receipt["spatial_method"]=="combined first-order Rusanov+two-point diffusion"
            and receipt["diffusion_tensor"]==[[D,0.],[0.,0.]] and receipt["step_policy"]=="fixed_courant"
            and receipt["regime_control"] is True and receipt["fixed_advective_courant_cap"]==COURANT,
            "control method/physics/policy/native mismatch")
    criteria=receipt["criteria"]
    require(criteria["minimum_observed_order"]==GUARD
            and criteria["density_l1_max"]=={str(n):cap for n,cap in CAPS.items()}
            and criteria["mass_defect_max"]==2.e-11 and criteria["initial_max_error"]==1.e-12
            and criteria["time_error_max"]==1.e-12 and criteria["y_invariance_max"]==2.e-11
            and criteria["minimum_state"]==.7999999999 and criteria["maximum_state"]==1.2000000001
            and criteria["discrete_fourier_max_error"]==3.e-12,"control guards changed")
    rows=[]
    for n,record in zip(GRIDS,receipt["records"],strict=True):
        dt=step(n,"fixed_courant")
        require(record["cells"]==[n,n] and record["fixed_dt"]==dt and record["step_policy"]=="fixed_courant"
                and record["advective_courant"]==COURANT and record["saved_state"]==f"state_{n}.npz",
                "control duration/mesh policy mismatch")
        path=directory/record["saved_state"]
        require(path.is_file() and not path.is_symlink() and path.stat().st_size<=8*1024*1024,"invalid control file")
        data=path.read_bytes()
        require(hashlib.sha256(data).hexdigest()==record["saved_state_sha256"],"external control state SHA mismatch")
        saved=archive(data)
        initial,actual=saved["initial"],saved["final"]
        require(initial.shape==actual.shape==(1,n,n) and initial.dtype==actual.dtype==np.dtype("float64")
                and np.isfinite(initial).all() and np.isfinite(actual).all(),"control physical state invalid")
        require(type(saved["time"].item()) is float and saved["time"].item().hex()==END.hex()
                and saved["dt"].item()==dt and saved["velocity"].item()==A and saved["diffusivity"].item()==D
                and saved["transverse_diffusivity"].item()==0. and saved["step_policy"].item()=="fixed_courant"
                and saved["advective_courant"].item()==COURANT,"control physical parameters mismatch")
        initial_error=float(np.max(abs(initial-cell_means(n,0.))))
        numerical_error=float(np.max(abs(actual-mode(n,dt))))
        stencil_error=float(np.max(abs(actual-stencil(initial[0,0],dt))))
        require(initial_error<=1.e-12 and max(numerical_error,stencil_error)<=3.e-12,"actual control state does not realize ForwardEuler")
        exact=cell_means(n,END)
        l1=math.fsum(float(value) for value in abs(actual-exact).flat)/(n*n)
        mass=abs(math.fsum(float(value) for value in (actual-initial).flat)/(n*n))
        y_defect=float(np.max(abs(actual-actual[:,:1,:])))
        require(l1<=CAPS[n] and mass<=2.e-11 and y_defect<=2.e-11 and actual.min()>=.7999999999
                and actual.max()<=1.2000000001 and record["accepted_steps"]==len(schedule(dt)),"control scientific guards failed")
        require(abs(l1-record["density_l1"])<=1.e-14,"control reported L1 differs from actual state")
        wrong=float(np.max(abs(actual-mode(n,dt,temporal="ssprk2"))))
        require(wrong>1.e-5,"control does not distinguish the selected temporal method")
        rows.append(dict(n=n,state_sha256=record["saved_state_sha256"],l1=l1,mode_linf=numerical_error,
                         stencil_linf=stencil_error,mass_defect=mass,y_invariance=y_defect,
                         wrong_ssprk2_linf=wrong,steps=record["accepted_steps"]))
    orders=[math.log2(a["l1"]/b["l1"]) for a,b in zip(rows[:-1],rows[1:],strict=True)]
    require(all(p>=GUARD for p in orders) and max(abs(a-b) for a,b in zip(orders,receipt["observed_orders"],strict=True))<=1.e-12,
            "control observed order differs/fails")
    return dict(status="fixed_courant_control_received",receipt_sha256=receipt_sha,native_sha256=native_sha,
                rows=rows,observed_orders=orders,native_execution_here=False,
                historical_m04_received=False,scope="separate regime control only; no closure of old failed M04")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical",type=Path)
    parser.add_argument("--control",type=Path)
    parser.add_argument("--control-owner-sha256")
    parser.add_argument("--control-native-sha256")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=dict(schema="sol61.m04-forward-euler-regime-audit@1",
                combined_bound=prediction("combined_bound"),fixed_courant=prediction("fixed_courant"),
                new_native_reception="pending")
    if args.historical:
        result["historical"]=historical(args.historical)
    if args.control:
        result["control"]=receive_control(args.control,args.control_owner_sha256,args.control_native_sha256)
        result["new_native_reception"]="separate fixed-courant regime control received"
    raw=json.dumps(result,indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(raw)
    print(raw)


if __name__=="__main__":
    main()
