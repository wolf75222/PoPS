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

def test_installed_uniform_accepted_storage_observation(tmp_path,record_property,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2)
    world=native.mpi_world() if native_mpi_communicator(native)=="MPI_COMM_WORLD" else None
    rank=0 if world is None else world.rank
    with collective_check(world):assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case,layout=collective_call(world,lambda:make_case(nonconservative=True))
    plan=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout))
    artifact=(collective_call(world,lambda:pops.compile(plan)) if world is None else compile_resolved_plan_once(world,plan,route="uniform-storage-observation",compile_artifact=pops.compile))
    subject=plan.initial_condition_plan.bindings[0].subject
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values={subject:initial_averages()},resources={"execution_context":artifact_execution_context(artifact)}))
    directory=collective_directory(world,tmp_path/"uniform-storage-observation")
    record_property("uniform_storage_observation",str(directory))
    before=collective_call(world,runtime.observe_accepted_state_storage)
    again=collective_call(world,runtime.observe_accepted_state_storage)
    values=collective_call(world,lambda:np.asarray(runtime.state_global("population")))
    with collective_check(world):
        assert before==again
        assert (before.time,before.macro_step)==(runtime.time(),runtime.macro_step())
        authenticate_carrier_values(before.complete,values)
    # Preserve each actual owner shard, including empty ranks; only ROOT writes merged bytes.
    collective_call(world,lambda:(directory/("rank"+str(rank)+".bin")).write_bytes(before.rank_local))
    if rank==0:
        (directory/"complete.bin").write_bytes(before.complete)
        (directory/"receipt.json").write_text(json.dumps({"contract":before.contract,"time":before.time,"macro_step":before.macro_step,"scope":"readonly accepted Uniform full-grown storage; no transport or rollback qualification","sha256":hashlib.sha256(before.complete).hexdigest()},sort_keys=True,allow_nan=False)+"\n")
