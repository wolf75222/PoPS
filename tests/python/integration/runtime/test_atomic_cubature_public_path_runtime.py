"""Installed-package cubature Raw path; no full Fan–Li/AMR qualification."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pops
import pytest
from tests.python.support.atomic_cubature_path_case import make_case
from tests.python.support.atomic_cubature_fv_oracle import DT,NX,NY,initial_averages,forward_euler,authenticate_carrier_values
from tests.python.support.collective_checks import collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.atomic_native_capture import select_layout_program,save_phase,execute_captured_step,layout_program_json_identity,compile_with_model_tus

pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]
SCHEMA='pops.atomic-cubature-raw-native-fixture@2'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(world,runtime):
    values=collective_call(world,lambda:np.array(runtime.block_level_state_global('population',0),copy=True))
    carriers=collective_call(world,lambda:bytes(runtime._executor.checkpoint_state_carriers()))
    clock=collective_call(world,lambda:(runtime.time(),runtime.macro_step(),runtime.n_levels()))
    return values,carriers,clock


@pytest.mark.parametrize('nonconservative',[False,True],ids=['conservative-atoms','raw-density-product'])
def test_installed_atomic_cubature_raw_path(nonconservative,tmp_path,record_property,
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2)
    world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),'installed package required'
    case,layout=collective_call(world,lambda:make_case(nonconservative=nonconservative,amr=True,fixed_dt=DT))
    resolved=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout))
    directory=collective_directory(world,tmp_path/'atomic-raw')
    def compile_artifact(plan):
        if world is None or world.rank==0:return compile_with_model_tus(plan,directory/'actual-compiles',pops.compile)
        return pops.compile(plan)  # authenticated peer cache load, not a TU witness
    artifact=(collective_call(world,lambda:compile_artifact(resolved)) if world is None else
        compile_resolved_plan_once(world,resolved,route='atomic-raw-'+str(nonconservative),compile_artifact=compile_artifact))
    row=collective_call(world,lambda:select_layout_program(artifact,resolved))
    program=row.program
    def retain_provenance():
        if world is None or world.rank==0:
            if type(program._generated_cpp) is not str: raise ValueError('retained Program C++ required')
            (directory/'program.cpp').write_text(program._generated_cpp)
            program.dump_ir(directory/'program.ir.json')
            (directory/'compiled-manifest.json').write_text(json.dumps(artifact.manifest().to_dict(),sort_keys=True,allow_nan=False)+'\n')
    collective_call(world,retain_provenance)
    subject=resolved.initial_condition_plan.bindings[0].subject
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values={subject:initial_averages()},
        resources={'execution_context':artifact_execution_context(artifact)}))
    images=[]; reports=[]; phases=[]; attempts=[]
    def persist_receipt(status):
        if world is None or world.rank==0:
            receipt={'schema':SCHEMA,'nonconservative':nonconservative,'shape':[6,NY,NX],'dt':DT,
                'ranks':1 if world is None else int(world.size),'phases':phases,'clocks':[i[2] for i in images],
                'status':status,'science_assertions':'passed' if status=='fixture-guards-passed' else 'not-yet-run','attempts':attempts,
                'package':str(Path(pops.__file__).resolve()),'native':{'path':str(Path(native.__file__).resolve()),'sha256':digest(native.__file__)},
                'compiled':{'path':str(Path(program.so_path).resolve()),'sha256':digest(program.so_path),
                            'abi_key':program.abi_key,'problem_hash':program.problem_hash,'cache_key':program.cache_key,
                            'layout_program':layout_program_json_identity(row),
                            'artifact_identity_token':artifact.artifact_identity.token},
                'files':{str(p.relative_to(directory)):digest(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='receipt.json'},
                'root_scientific_approval':False,'full_m17_qualification':False}
            (directory/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,allow_nan=False)+'\n')
    def persist_capture(phase,image):
        if world is None or world.rank==0:save_phase(directory,phase,image)
    record_property('atomic_cubature_receipt',str(directory/'receipt.json'))
    images.append(capture(world,runtime));phases.append('initial')
    collective_call(world,lambda:persist_capture('initial',images[-1]))
    collective_call(world,lambda:persist_receipt('partial-captures'))
    for step,phase in ((1,'accepted'),(2,'continuous')):
        def on_attempt(failures):
            attempts.append({'step':step,'failures':failures})
            persist_receipt('native-run-failed' if any(failures) else 'partial-captures')
        def on_capture(image,failed):
            captured_phase='failed-step-'+str(step) if failed else phase
            images.append(image);phases.append(captured_phase)
            persist_capture(captured_phase,image)
            persist_receipt('native-run-failed' if failed else
                            'captures-complete' if step==2 else 'partial-captures')
        report,_=execute_captured_step(world,
            lambda:pops.run(runtime,t_end=step*DT,max_steps=1,console=False),
            lambda:capture(world,runtime),on_attempt,on_capture)
        reports.append(report)
    with collective_check(world):
        expected=initial_averages()
        for step,(image,blob,clock) in enumerate(images):
            saved=np.load(directory/(('initial','accepted','continuous')[step]+'.npy'),allow_pickle=False)
            assert saved.dtype==image.dtype and saved.shape==image.shape and saved.tobytes()==image.tobytes()
            authenticate_carrier_values(blob,saved)
            np.testing.assert_allclose(saved.reshape(expected.shape),expected,rtol=2e-12,atol=2e-13)
            assert blob and clock==(step*DT,step,1)
            expected=forward_euler(expected,nonconservative=nonconservative)
        assert all(report.accepted_steps==1 for report in reports)

    collective_call(world,lambda:persist_receipt('fixture-guards-passed'))
