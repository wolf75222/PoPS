"""Independent metadata-only attacks; no Native receipt or runtime."""
import importlib.util
from pathlib import Path
import pytest
s=importlib.util.spec_from_file_location("reader",Path(__file__).with_name("sol61_halo_stage_failure_saved_audit_v2.py"));r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
@pytest.mark.parametrize("ranks",[1,2])
def test_divergent_positive(ranks):
    r.divergent({"divergent_request_applicable":ranks>1,"divergent_request_refusals":[] if ranks==1 else [["ValueError","request differs between ranks",False]]*ranks},ranks)
@pytest.mark.parametrize("proof,ranks",[({"divergent_request_applicable":0,"divergent_request_refusals":[]},1),({"divergent_request_applicable":False,"divergent_request_refusals":[["ValueError","request differs between ranks",False]]},1),({"divergent_request_applicable":True,"divergent_request_refusals":[]},2),({"divergent_request_applicable":True,"divergent_request_refusals":[["ValueError","early failure",False]]*2},2)])
def test_divergent_negative(proof,ranks):
    with pytest.raises(ValueError):r.divergent(proof,ranks)

@pytest.mark.parametrize("attack",[None,"physical","pin","samepath","extraexclude"])
def test_checkpoint_pair_pins_and_physical_bits(tmp_path,attack):
    import numpy as np,json,hashlib
    blobs={"retry":b"carrier","aligned-retry":b"carrier"};pins={}
    for key in ("retry","attempt_aligned_control"):
        cp={"pops_amr_checkpoint_version":np.array(12,dtype="int64"),"amr_accepted_contract":np.array(json.dumps({"schema_version":9})),"state_carriers_checkpoint":np.frombuffer(b"carrier",dtype="uint8"),"t":np.array(2/64),"macro_step":np.array(2,dtype="int64"),"state":np.array([-0.0]),"pops_checkpoint_manifest":np.array(key),"pops_restart_identity":np.array(key)}
        if attack=="physical" and key=="attempt_aligned_control":cp["state"]=np.array([0.0])
        path=tmp_path/(key+".npz");np.savez_compressed(path,**cp);pins[key]={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    if attack=="pin":pins["retry"]["sha256"]="0"*64
    if attack=="samepath":pins["attempt_aligned_control"]=pins["retry"]
    proof={"schema":"pops.accepted-halo-test-failure.retry-aligned-control@2","checkpoints":pins,"excluded_lifecycle_seals":["pops_checkpoint_manifest","pops_restart_identity"],"lifecycle_seal_differences":["pops_checkpoint_manifest","pops_restart_identity"]}
    if attack=="extraexclude":proof["excluded_lifecycle_seals"].append("state")
    (tmp_path/"retry-aligned-control-checkpoint-proof.json").write_text(json.dumps(proof))
    images={k:(v,) for k,v in blobs.items()}
    if attack is None:r.checkpoint_pair(tmp_path,images)
    else:
        with pytest.raises(ValueError):r.checkpoint_pair(tmp_path,images)
