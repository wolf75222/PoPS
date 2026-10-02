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
def audit(directory,ranks):
    d=Path(directory);images={}
    for phase,clock in (("before",[1/64,1]),("after-failure",[1/64,1]),("retry",[2/64,2]),("continuous-control",[2/64,2])):
        m=load(d/(phase+"-metadata.json"));blob=(d/(phase+"-carriers.bin")).read_bytes()
        need(m["phase"]==phase and m["accepted_clock"]==clock,"phase clock")
        need(hashlib.sha256(blob).hexdigest()==m["carrier_sha256"],"carrier pin")
        need(m["accepted_halo_contract"]==[["pops.amr.accepted-halo-preparation@1","candidate_accepted_clock","all_state_components","1","1"]],"Halo contract")
        images[phase]=(blob,arrays(d/(phase+"-valid.npz")),ast.literal_eval(m["native_metadata_repr"]))
    for l,r in (("before","after-failure"),("retry","continuous-control")):
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
    return {"capture_rollback_retry_exact":True,"root_approval":False,"divergent_request_proof_persisted": "divergent_request_refusals" in proof,"retry_control_full_checkpoint_claim":False,"wholegrown_formula_claim":False}
