"""Independent offline capture audit; never executes Native or grants ROOT approval."""
import ast,hashlib,json
from pathlib import Path
import numpy as np

def need(ok,msg):
    if not ok:raise ValueError(msg)
def load(path):
    def pairs(rows):
        out={}
        for k,v in rows:
            need(k not in out,"duplicate JSON key");out[k]=v
        return out
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs)
def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        need(len(z.files)==len(set(z.files)),"duplicate NPZ member")
        return {k:z[k].copy() for k in z.files}
def exact(a,b):
    need(a.keys()==b.keys(),"payload inventory")
    for k in a:need(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes(),"payload bits: "+k)
def divergent(proof,ranks):
    need(type(proof["divergent_request_applicable"]) is bool and proof["divergent_request_applicable"]==(ranks>1),"divergent applicability")
    rows=proof["divergent_request_refusals"]
    need(type(rows) is list,"divergent inventory type")
    if ranks==1:need(rows==[],"Serial divergent rows")
    else:need(len(rows)==ranks and all(type(r) is list and len(r)>=2 and type(r[1]) is str and "request differs between ranks" in r[1] for r in rows),"MPI divergent refusal")
def checkpoint_pair(directory,images):
    proof=load(Path(directory)/"retry-aligned-control-checkpoint-proof.json")
    need(proof["schema"]=="pops.accepted-halo-test-failure.retry-aligned-control@2","retry proof schema")
    need(proof["excluded_lifecycle_seals"]==["pops_checkpoint_manifest","pops_restart_identity"],"only lifecycle seals excluded")
    pins=proof["checkpoints"];need(set(pins)=={"retry","attempt_aligned_control"},"retry CP inventory")
    payloads={};paths=[]
    for key,phase in (("retry","retry"),("attempt_aligned_control","aligned-retry")):
        pin=pins[key];need(set(pin)=={"path","sha256"},"CP pin inventory")
        raw=Path(pin["path"]).read_bytes();need(hashlib.sha256(raw).hexdigest()==pin["sha256"],"retry CP pin")
        paths.append(Path(pin["path"]).resolve());cp=arrays(pin["path"])
        need(cp["pops_amr_checkpoint_version"].shape==() and cp["pops_amr_checkpoint_version"].dtype==np.dtype("int64") and cp["pops_amr_checkpoint_version"].item()==12,"retry CP12")
        need(json.loads(str(cp["amr_accepted_contract"]))["schema_version"]==9,"retry accepted9")
        need(cp["state_carriers_checkpoint"].tobytes()==images[phase][0],"retry CP carrier projection")
        need(cp["t"].item()==2/64 and cp["macro_step"].item()==2,"retry CP clock")
        payloads[key]=cp
    need(paths[0]!=paths[1],"distinct retry/control CP paths")
    a,b=payloads.values();need(a.keys()==b.keys(),"retry CP inventory differs")
    excluded=set(proof["excluded_lifecycle_seals"]);exact({k:v for k,v in a.items() if k not in excluded},{k:v for k,v in b.items() if k not in excluded})
    actual=[k for k in proof["excluded_lifecycle_seals"] if a[k].tobytes()!=b[k].tobytes()]
    need(proof["lifecycle_seal_differences"]==actual,"declared seal differences")
def audit(directory,ranks):
    d=Path(directory);images={}
    for phase,clock in (("before",[1/64,1]),("after-failure",[1/64,1]),("retry",[2/64,2]),("continuous-control",[2/64,2]),("aligned-before",[1/64,1]),("aligned-after-failure",[1/64,1]),("aligned-retry",[2/64,2])):
        m=load(d/(phase+"-metadata.json"));blob=(d/(phase+"-carriers.bin")).read_bytes()
        need(m["phase"]==phase and m["accepted_clock"]==clock,"phase clock")
        need(hashlib.sha256(blob).hexdigest()==m["carrier_sha256"],"carrier pin")
        need(m["accepted_halo_contract"]==[["pops.amr.accepted-halo-preparation@1","candidate_accepted_clock","all_state_components","1","1"]],"Halo contract")
        images[phase]=(blob,arrays(d/(phase+"-valid.npz")),ast.literal_eval(m["native_metadata_repr"]))
    for l,r in (("before","after-failure"),("retry","continuous-control"),("aligned-before","aligned-after-failure"),("retry","aligned-retry")):
        need(images[l][0]==images[r][0] and images[l][2]==images[r][2],"carrier/history/diagnostic/geometry/clock exact");exact(images[l][1],images[r][1])
    a=arrays(d/"before-checkpoint.npz");b=arrays(d/"after-failure-checkpoint.npz")
    need(a["pops_amr_checkpoint_version"].shape==() and a["pops_amr_checkpoint_version"].item()==12,"CP12")
    need(json.loads(str(a["amr_accepted_contract"]))["schema_version"]==9,"accepted9")
    need(a["state_carriers_checkpoint"].tobytes()==images["before"][0],"CP carrier projection")
    excluded={"pops_checkpoint_manifest","pops_restart_identity"};need(a.keys()==b.keys(),"CP inventory");exact({k:v for k,v in a.items() if k not in excluded},{k:v for k,v in b.items() if k not in excluded})
    proof=load(d/"failure-proof.json");need(len(proof["receipts"])==ranks and len(proof["failures"])==ranks,"rank inventory")
    for rank,row in enumerate(proof["receipts"]):
        need(all(row[k] is True for k in ("requested","reached","consumed","before_publication")),"one-shot boundary")
        need((row["version"],row["phase"],row["block"],row["level"],row["rank"],row["tick"],row["armed_tick"] )==(1,1,0,0,ranks-1,1,1),"request/tick")
        need(row["local_error"] is (rank==ranks-1) and row["time"]==2/64 and row["dt"]==1/64,"target/point")
    boundary="accepted halo test failure after block-level preparation fence"
    need(all(x and boundary in x[1] for x in proof["failures"]),"actual fenced failure")
    need(len(proof["invalid_request_refusals"])==7 and all(len(x)==ranks and all(x) for x in proof["invalid_request_refusals"]),"invalid requests")
    need(len(proof["duplicate_arm_refusals"])==ranks and all(proof["duplicate_arm_refusals"]),"duplicate arm")
    divergent(proof,ranks);checkpoint_pair(d,images)
    aligned=load(d/"aligned-failure-proof.json")
    need(aligned["receipts"]==proof["receipts"],"aligned receipt")
    need(len(aligned["failures"])==ranks and all(x and boundary in x[1] for x in aligned["failures"]),"aligned fenced error")
    first=arrays(d/"aligned-before-checkpoint.npz");after=arrays(d/"aligned-after-failure-checkpoint.npz")
    need(first.keys()==after.keys(),"aligned inventory");exact({k:v for k,v in first.items() if k not in excluded},{k:v for k,v in after.items() if k not in excluded})
    continuous_difference(d)
    return {"contract":"independent.acceptedHaloStageFailure@2","capture_rollback_retry_exact":True,"root_approval":False,"divergent_request_proof_persisted":True,"retry_control_full_checkpoint_claim":True,"wholegrown_formula_claim":False}

def continuous_difference(d):
    import importlib.util
    spec=importlib.util.spec_from_file_location("attempt_wire",Path(__file__).with_name("sol61_popsand9_attempt_spans.py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);spans=module.spans
    proof=load(Path(d)/"retry-continuous-checkpoint-differences.json")
    need(proof["schema"]=="pops.accepted-halo-test-failure.continuous-comparison@1","continuous schema")
    need(proof["accepted_attempts"]=={"retry":3,"continuous_control":2,"attempt_aligned_control":3},"causal attempts")
    pins=proof["checkpoints"];need(set(pins)=={"retry","continuous_control"},"continuous pins")
    payloads=[]
    for key in ("retry","continuous_control"):
        pin=pins[key];need(hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest()==pin["sha256"],"continuous pin");payloads.append(arrays(pin["path"]))
    a,b=payloads;need(a.keys()==b.keys(),"continuous inventory")
    differences={k for k in a if a[k].dtype!=b[k].dtype or a[k].shape!=b[k].shape or a[k].tobytes()!=b[k].tobytes()}
    need(differences==set(proof["different_members"])=={"program_accepted_state","program_accepted_state_source_authority","pops_checkpoint_manifest","pops_restart_identity"},"four differences")
    x,y=(p["program_accepted_state"].tobytes() for p in payloads);sx,sy=spans(x),spans(y)
    need(len(sx)==len(sy)==113,"attempt count")
    need(all(a[:3]==b[:3] and (a[3],b[3])==(3,2) for a,b in zip(sx,sy)),"causal values")
    covered={i for _,lo,hi,_ in sx for i in range(lo,hi)}
    need(len(x)==len(y) and all(x[i]==y[i] for i in range(len(x)) if i not in covered),"nonattempt bytes")
