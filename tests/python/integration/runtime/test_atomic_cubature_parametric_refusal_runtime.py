"""Future installed Native attempt; early bind refusal is not a stage proof."""
import json
from pathlib import Path
import sys
import pops
import pytest
from tests.python.support.atomic_cubature_path_case import make_case
from tests.python.support.atomic_cubature_fv_oracle import DT,initial_averages,authenticate_carrier_values
from tests.python.support.atomic_native_capture import select_layout_program,save_phase,execute_captured_step,layout_program_json_identity,execute_captured_bind,compile_with_model_tus
from tests.python.support.collective_checks import collective_call,collective_check,collective_attempt
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.integration.runtime.test_atomic_cubature_public_path_runtime import capture,digest

pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]


def test_installed_atomic_cubature_finite_parameter_attempt_refusal(tmp_path,record_property,
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2)
    world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case,layout,parameter=collective_call(world,lambda:make_case(nonconservative=True,
        amr=True,fixed_dt=DT,scale_parameter=True))
    resolved=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout))
    directory=collective_directory(world,tmp_path/'atomic-parameter-refusal')
    def compile_artifact(plan):
        if world is None or world.rank==0:return compile_with_model_tus(plan,directory/'actual-compiles',pops.compile)
        return pops.compile(plan)  # authenticated peer cache load only
    artifact=(collective_call(world,lambda:compile_artifact(resolved)) if world is None else
        compile_resolved_plan_once(world,resolved,route='atomic-raw-finite-parameter-refusal',compile_artifact=compile_artifact))
    row=collective_call(world,lambda:select_layout_program(artifact,resolved))
    program=row.program
    proof={'schema':'pops.atomic-cubature-parametric-refusal@2','parameter':1.e308,
        'phase':'before-bind','root_scientific_approval':False,'native_stage_refusal_proved':False}
    def persist():
        if world is None or world.rank==0:
            proof['files']={str(p.relative_to(directory)):digest(p) for p in sorted(directory.rglob('*'))
                if p.is_file() and p.name!='receipt.json'}
            (directory/'receipt.json').write_text(json.dumps(proof,sort_keys=True,allow_nan=False)+'\n')
    def provenance():
        if world is None or world.rank==0:
            if type(program._generated_cpp) is not str:raise ValueError('retained Program C++ required')
            (directory/'program.cpp').write_text(program._generated_cpp)
            program.dump_ir(directory/'program.ir.json')
            (directory/'compiled-manifest.json').write_text(json.dumps(artifact.manifest().to_dict(),sort_keys=True,allow_nan=False)+'\n')
            proof['compiled']={'layout':layout_program_json_identity(row),'path':str(program.so_path),
                'sha256':digest(program.so_path),'artifact_identity_token':artifact.artifact_identity.token}
            proof['native']={'path':str(native.__file__),'sha256':digest(native.__file__)}
            persist()
    collective_call(world,provenance)
    record_property('atomic_parametric_refusal_receipt',str(directory/'receipt.json'))
    subject=resolved.initial_condition_plan.bindings[0].subject
    def bind_failure(local_error,failures):
        rank=0 if world is None else int(world.rank)
        evidence={'schema':'pops.atomic-native-bind-failure@1','phase':'bind-failed',
            'rank':rank,'local_exception_chain':local_error,'collective_failures':failures,
            'native_stage_refusal_proved':False}
        (directory/('bind-failure-rank'+str(rank)+'.json')).write_text(json.dumps(evidence,sort_keys=True,allow_nan=False)+'\n')
        proof['phase']='bind-failed';proof['bind_failures']=failures
    runtime=execute_captured_bind(world,lambda:pops.bind(artifact,params={parameter:1.e308},
        initial_values={subject:initial_averages()},resources={'execution_context':artifact_execution_context(artifact)}),bind_failure,persist)
    before=capture(world,runtime)
    collective_call(world,lambda:save_phase(directory,'initial',before) if world is None or world.rank==0 else None)
    proof['phase']='initial-captured';collective_call(world,persist)
    failed_images=[]
    def on_attempt(failures):
        proof['attempt_failures']=failures;proof['phase']='attempt-returned';persist()
    def on_capture(image,failed):
        failed_images.append(image)
        if world is None or world.rank==0:save_phase(directory,'refused' if failed else 'unexpected-accepted',image)
        proof['phase']='refused-captured' if failed else 'unexpected-accepted';persist()
    _,failures=collective_attempt(world,lambda:execute_captured_step(world,
        lambda:pops.run(runtime,t_end=DT,max_steps=1,console=False),lambda:capture(world,runtime),on_attempt,on_capture))
    with collective_check(world):
        assert all(f and f[2] and 'prepared Cartesian path face tuple refused publication' in f[1] for f in failures),failures
        assert len(failed_images)==1
        after=failed_images[0]
        assert before[0].dtype==after[0].dtype and before[0].shape==after[0].shape
        assert before[0].tobytes()==after[0].tobytes() and before[1:]==after[1:]
        authenticate_carrier_values(before[1],before[0])
        authenticate_carrier_values(after[1],after[0])
        assert before[2]==(0.,0,1)
    proof['phase']='rollback-guards-passed';proof['native_stage_refusal_proved']=True
    collective_call(world,persist)
