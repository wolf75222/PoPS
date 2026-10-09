"""Offline STD-only observation of actual Serial variants; does not seal science."""
import ast, hashlib, json, math, struct, sys, xml.etree.ElementTree as ET, zipfile
from pathlib import Path

def npy(raw):
    start=10 if raw[6]==1 else 12
    length=int.from_bytes(raw[8:start],"little")
    h=ast.literal_eval(raw[start:start+length].decode())
    return h,raw[start+length:]

def floats(z,name):
    h,raw=npy(z.read(name+".npy")); assert h["descr"]=="<f8"
    return struct.unpack("<"+str(len(raw)//8)+"d",raw)

def q(a,b): return a+a*a+.1*b*b,b+b*b+.2*a*b

def conserved_b(a,total):
    linear=1+.2*a; rhs=total-a-a*a
    return 2*rhs/(linear+math.sqrt(linear*linear+4*1.1*rhs))

def oracle(width,step):
    if width==1:
        value=.15+.15**2+step*((.16+.16**2)-(.15+.15**2))
        return [2*value/(1+math.sqrt(1+4*value))], [value]
    initial=q(.15,.25); total=sum(initial)
    first=q(.16,conserved_b(.16,total))
    target=[initial[i]+step*(first[i]-initial[i]) for i in range(2)]
    lo,hi=.15,.20
    for _ in range(90):
        mid=(lo+hi)/2
        if q(mid,conserved_b(mid,total))[0]<target[0]: lo=mid
        else: hi=mid
    a=(lo+hi)/2
    return [a,conserved_b(a,total)],target

def main(base):
    for name in ["identity.json","source-files.json"]:
        assert (base/name).read_bytes()==(base/"after"/name).read_bytes()
    result=json.loads((base/"result.json").read_text())
    assert result["status"]=="passed" and result["counts"]==dict(tests=4,failures=0,errors=0,skipped=0)
    for name,key in [("identity.json","identity_sha256"),("pytest.log","log_sha256")]:
        assert hashlib.sha256((base/name).read_bytes()).hexdigest()==result[key]
    paths=[]; out=[]
    for case in ET.parse(base/"pytest.xml").getroot().iter("testcase"):
        props={x.attrib["name"]:x.attrib["value"] for x in case.iter("property")}
        path=Path(props["evolved_stage_amr_receipt"]); assert path not in paths; paths.append(path)
        r=json.loads(path.read_text()); pins=[]
        def walk(v):
            if isinstance(v,dict):
                if "path" in v and "sha256" in v:
                    assert hashlib.sha256(Path(v["path"]).read_bytes()).hexdigest()==v["sha256"]; pins.append(v["path"])
                for x in v.values(): walk(x)
            elif isinstance(v,list):
                for x in v:walk(x)
        walk(r)
        for a,c in [("accepted","reloaded"),("continuous","replay")]:
            for level in [0,1]:
                with zipfile.ZipFile(path.parent/f"{a}-level{level}.npz") as x,zipfile.ZipFile(path.parent/f"{c}-level{level}.npz") as y:
                    assert set(x.namelist())==set(y.namelist())
                    assert all(x.read(n)==y.read(n) for n in x.namelist())
        phases=[]
        for phase,step in [("accepted",1),("continuous",2),("replay",2)]:
            expect,target=oracle(r["width"],step); max_t=0.; constraint=0.; amounts=[0.]*r["width"]; volume=0.; active_counts=[]
            for level in [0,1]:
                with zipfile.ZipFile(path.parent/f"{phase}-level{level}.npz") as z:
                    h,mask=npy(z.read("active.npy")); assert h["descr"]=="|b1" and all(x in [0,1] for x in mask)
                    active_counts.append(sum(mask)); ts=[floats(z,f"T{i}") for i in range(r["width"])]
                    cell_volume=1/(r["cells"]*2**level)**2
                    for i,active in enumerate(mask):
                        if not active:continue
                        volume+=cell_volume
                        vals=[t[i] for t in ts]
                        max_t=max(max_t,max(abs(a-b) for a,b in zip(vals,expect)))
                        qs=[vals[0]+vals[0]**2] if r["width"]==1 else q(*vals)
                        for k,v in enumerate(qs):amounts[k]+=v*cell_volume
                    if r["width"]==2:
                        zs=floats(z,"z")
                        constraint=max(constraint,max(abs(zs[i]-.25*ts[0][i]-.5*ts[1][i]) for i in range(len(mask)) if mask[i]))
            assert 0<active_counts[0]<r["cells"]**2
            assert abs(volume-1)<1e-14 and max_t<r["acceptance"] and constraint<r["acceptance"]
            assert max(abs(a-b) for a,b in zip(amounts,target))<r["acceptance"]
            recorded=r["phases"][phase]
            assert recorded["metadata"][-1][:2]==[step*r["dt"],step]
            assert max(abs(a-b) for a,b in zip(amounts,recorded["checks"]["Q_amounts"]))<1e-13
            with zipfile.ZipFile(path.parent/f"{phase}-checkpoint.npz") as z:
                h,raw=npy(z.read("pops_amr_checkpoint_version.npy")); assert struct.unpack("<q",raw)[0]==12
                h,raw=npy(z.read("state_carriers_checkpoint.npy")); assert raw.startswith(b"POPSCAR1")
                h,raw=npy(z.read("pops_checkpoint_manifest.npy")); manifest=json.loads(raw.decode("utf-32-le")); assert isinstance(manifest,dict)
            phases.append(dict(phase=phase,active_counts=active_counts,volume=volume,T_oracle_linf=max_t,constraint_linf=constraint,Q_amounts=amounts,Q_oracle_linf=max(abs(a-b) for a,b in zip(amounts,target)),recorded_checks=recorded["checks"]))
        out.append(dict(case=case.attrib["name"],receipt=str(path),receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pins=len(pins),phases=phases))
    assert len(paths)==4
    print(json.dumps(dict(scope="offline observations; no science seal",cases=out),indent=2))
if __name__=="__main__":main(Path(sys.argv[1]))
