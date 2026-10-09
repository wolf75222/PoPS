"""Future true installed Native Serial/MPI receipt; no Source science claim."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pops,pytest
from tests.python.support.atomic_cubature_path_case import make_case
from tests.python.support.atomic_cubature_fv_oracle import initial_averages,authenticate_carrier_values
from tests.python.support.collective_checks import collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]

def test_installed_uniform_accepted_storage_observation_v2(tmp_path,record_property,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2)
    world=native.mpi_world() if native_mpi_communicator(native)=="MPI_COMM_WORLD" else None
    rank=0 if world is None else world.rank
    with collective_check(world):assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case,layout=collective_call(world,lambda:make_case(nonconservative=True))
    plan=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout,compile_options={"model_source_policy":"require"}))
    artifact=(collective_call(world,lambda:pops.compile(plan)) if world is None else compile_resolved_plan_once(world,plan,route="uniform-storage-observation-v2",compile_artifact=pops.compile))
    directory=collective_directory(world,tmp_path/"uniform-storage-observation-v2")
    record_property("uniform_storage_observation_v2",str(directory))
    from tests.python.support.uniform_storage_observation_v2 import retain_uniform_provenance,save_observation,authenticate_owner_shard
    collective_call(world,lambda:retain_uniform_provenance(artifact,native,directory) if rank==0 else None)
    subject=plan.initial_condition_plan.bindings[0].subject
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values={subject:initial_averages()},resources={"execution_context":artifact_execution_context(artifact)}))
    images=[];receipts=[]
    for phase in ('first','second'):
        image=collective_call(world,runtime.observe_accepted_state_storage)
        values=collective_call(world,lambda:np.array(runtime.state_global("population"),copy=True))
        clock=collective_call(world,lambda:{"time":runtime.time(),"macro_step":runtime.macro_step()})
        receipt=collective_call(world,lambda:save_observation(directory,phase,rank,image,values,clock))
        images.append(image);receipts.append(receipt)
        with collective_check(world):
            authenticate_owner_shard(image,rank,1 if world is None else world.size)
            authenticate_carrier_values(image.complete,values)
    with collective_check(world):assert images[0]==images[1]
    collective_call(world,lambda:(directory/("rank"+str(rank)+".receipt.json")).write_text(json.dumps({
        "schema":"pops.uniform-storage-observation-native@2","observations":receipts,
        "scope":"two successive readonly initial observations; no step or rollback",
        "root_scientific_approval":False},sort_keys=True,allow_nan=False)+'\n'))
