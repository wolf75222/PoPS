"""ROOT Native reception: evolving accepted halos, exact restart and preflight refusal.

No failure is claimed to occur inside halo staging: malformed archive rejection is
restart preflight. No production test hook is introduced.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pops
import pytest
from tests.python.support.evolving_accepted_halo_case import build,DT
from tests.python.support.collective_checks import collective_call,collective_check,collective_attempt
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.amr_snapshots import composite_active_mask
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.native_execution_context import artifact_execution_context
from tests.review.sol61_amr_full_carrier_offline import decode
from pops._generated_release_contract import AMR_CHECKPOINT_PAYLOAD_VERSION
from tests.python.support.evolving_accepted_halo_oracle import full_carrier,halo_rows
from tests.review.sol61_tag_selection_oracle import receive

def observe(world,runtime):
    blob=collective_call(world,lambda:bytes(runtime._executor.checkpoint_state_carriers()))
    levels=collective_call(world,runtime.n_levels)
    arrays=[]
    geometry=[]
    for level in range(levels):
        state=collective_call(world,lambda level=level:runtime.block_level_state_global("marker",level))
        active=collective_call(world,lambda level=level:composite_active_mask(runtime,level,refinement_ratio=2))
        shape=8*2**level
        snapshot=collective_call(world,lambda level=level,shape=shape:runtime._executor._s._output_geometry_snapshot(
            level,(0.,0.),(1/shape,1/shape),(shape,shape),
            (2,2) if level+1<levels else (0,0),"pops://cell-measures/cartesian-area@1"))
        with collective_check(world):
            assert snapshot["dimension"]==2 and tuple(snapshot["cell_shape"])==(shape,shape)
            np.testing.assert_array_equal(active,np.asarray(snapshot["valid_cells"]) & ~np.asarray(snapshot["coverage"]))
            geometry.append(tuple(tuple(row) for row in snapshot["boxes"]))
            arrays.append((np.asarray(state).copy(),np.asarray(active).copy()))
    report=collective_call(world,runtime.program_report)
    halo=collective_call(world,runtime._executor.checkpoint_accepted_halo_contract)
    clock=collective_call(world,lambda:(runtime.time(),runtime.macro_step(),tuple(runtime.patch_boxes())))
    return blob,arrays,(report.histories,report.diagnostics,halo,tuple(geometry),clock)

def same(left,right):
    assert left[0]==right[0] and left[2]==right[2]
    assert len(left[1])==len(right[1])
    for lrow,rrow in zip(left[1],right[1],strict=True):
        for a,b in zip(lrow,rrow,strict=True):
            assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def persist(directory,phase,image):
    (directory/(phase+"-carriers.bin")).write_bytes(image[0])
    np.savez(directory/(phase+"-valid.npz"),**{f"{i}-{j}":a for i,row in enumerate(image[1]) for j,a in enumerate(row)})
    (directory/(phase+"-metadata.json")).write_text(json.dumps({"phase":phase,"native_metadata_repr":repr(image[2]),"accepted_halo_contract":image[2][2],"accepted_clock":image[2][-1][:2],"writer_geometry_boxes":image[2][3],"fine_patch_boxes":image[2][-1][2],"carrier_sha256":hashlib.sha256(image[0]).hexdigest()},indent=2)+"\n")

def publish(world,directory,phase,image):
    with collective_check(world):
        if world.rank==0:persist(directory,phase,image)

@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("subcycled",(False,True))
def test_public_evolving_accepted_halo_restart_and_refusal(isolated_native_cache,tmp_path,record_property,subcycled):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native=select_native_dimension(2);world=native.mpi_world()
    with collective_check(world):
        assert native.__abi_version__==8
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case,layout=collective_call(world,lambda:build(subcycled))
    resolved=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout))
    artifact=compile_resolved_plan_once(world,resolved,route="evolving accepted halo public reception",compile_artifact=pops.compile)
    context=collective_call(world,lambda:artifact_execution_context(artifact))
    def bind():return pops.bind(artifact,resources={"execution_context":context})
    runtime=collective_call(world,bind)
    directory=collective_directory(world,tmp_path/"evolving-halo")
    images={"initial":observe(world,runtime)}
    publish(world,directory,"initial",images["initial"])
    for step,phase in ((1,"accepted"),(2,"continuous")):
        report=collective_call(world,lambda step=step:pops.run(runtime,t_end=step*DT,max_steps=1,console=False))
        images[phase]=observe(world,runtime)
        publish(world,directory,phase,images[phase])
        if step==1:path=collective_call(world,lambda:runtime.checkpoint(directory/"accepted-checkpoint"))
        with collective_check(world):assert report.accepted_steps==1 and report.rejected_steps==0
    restarted=collective_call(world,bind)
    collective_call(world,lambda:restarted.restart(path))
    images["reloaded"]=observe(world,restarted)
    publish(world,directory,"reloaded",images["reloaded"])
    collective_call(world,lambda:pops.run(restarted,t_end=2*DT,max_steps=1,console=False))
    images["replay"]=observe(world,restarted)
    publish(world,directory,"replay",images["replay"])
    with collective_check(world):
        if world.rank==0:
            with np.load(path,allow_pickle=False) as saved:
                payload={k:saved[k].copy() for k in saved.files}
            contract=json.loads(str(payload["amr_accepted_contract"]))
            (directory/"accepted-checkpoint-contract.json").write_text(json.dumps({
                "payload_version":int(payload["pops_amr_checkpoint_version"]),
                "accepted_contract":contract,
                "payload_members":{k:{"dtype":str(v.dtype),"shape":list(v.shape),"sha256":hashlib.sha256(v.tobytes()).hexdigest()} for k,v in payload.items()}},indent=2)+"\n")
    # Preserve all captures before the scientific/exact assertions.
    with collective_check(world):
        same(images["accepted"],images["reloaded"]);same(images["continuous"],images["replay"])
        initial_archive=decode(np.frombuffer(images["initial"][0],dtype=np.uint8))
        fine_boxes=images["initial"][2][-1][2]
        geometry=images["initial"][2][3]
        coarse,fine=receive(fine_boxes,(8,8),0)
        # patch_boxes uses inclusive native(x,y), Writer uses half-open numpy(y,x).
        assert all(row[0]==1 for row in fine_boxes)
        assert tuple((lo[1],lo[0],hi[1]+1,hi[0]+1) for level,lo,hi in fine_boxes)==geometry[1]
        assert len(images["initial"][1])==2
        np.testing.assert_array_equal(images["initial"][1][0][1],coarse)
        np.testing.assert_array_equal(images["initial"][1][1][1],fine)
        for phase,step in (("initial",0),("accepted",1),("continuous",2),("reloaded",1),("replay",2)):
            image=images[phase];assert image[2][-1][:2]==(step*DT,step)
            halo_rows(image[2][2])
            archive=decode(np.frombuffer(image[0],dtype=np.uint8))
            assert image[2][3]==geometry
            full_carrier(initial_archive,archive,geometry,step*DT,step,1 if subcycled else 0)
            for (current,mask),(initial,oldmask) in zip(image[1],images["initial"][1],strict=True):
                np.testing.assert_array_equal(mask,oldmask)
                current=current.reshape(2,*mask.shape);initial=initial.reshape(2,*mask.shape)
                np.testing.assert_array_equal(current[1 if subcycled else 0][mask],np.ones(int(mask.sum())))
                # Active cells exclude restricted covered coarse values: at most six
                # Euler additions, six rational-clock arithmetic ops and two
                # comparison subtractions across two macrosteps give gamma14.
                eps=np.finfo(np.float64).eps;k=14;gamma=k*eps/(1-k*eps)
                assert np.max(np.abs(current[0 if subcycled else 1][mask]-initial[0 if subcycled else 1][mask]-step*DT))<=gamma*(np.max(np.abs(initial[0 if subcycled else 1][mask]))+step*DT+1)
        if world.rank==0:
            assert int(payload["pops_amr_checkpoint_version"])==AMR_CHECKPOINT_PAYLOAD_VERSION==12
            assert type(contract["schema_version"]) is int and contract["schema_version"]==9
            halo_rows(contract["accepted_halo"])
            assert contract["accepted_halo"]==images["accepted"][2][2]
            assert payload["state_carriers_checkpoint"].dtype==np.uint8
            payload["state_carriers_checkpoint"]=payload["state_carriers_checkpoint"][:-1]
            bad=directory/"truncated-carrier-checkpoint.npz";np.savez(bad,**payload)
    bad=directory/"truncated-carrier-checkpoint.npz"
    _,failures=collective_attempt(world,lambda:restarted.restart(bad))
    refused=observe(world,restarted)
    with collective_check(world):
        if world.rank==0:persist(directory,"refused-preflight",refused)
        assert all(failures),failures
        assert all("carrier" in row[1].lower() for row in failures),failures
        same(images["replay"],refused)
    record_property("evolving_halo_observations",str(directory))
    record_property("refusal_phase","restart carrier preflight; NOT halo-stage rollback")
    record_property("artifact_identity",artifact.artifact_identity.token)
    record_property("native_sha256",hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())
