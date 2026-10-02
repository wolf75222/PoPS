"""Installed Native reception of explicit degree-five update on the second State."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pytest
import pops
from tests.python.support import m16_second_state_native_case as case
from tests.python.integration.runtime import test_program_affine_moment_explicit_basis_runtime as a
from tests.python.support.m16_explicit_native_capture import retain_provenance,persisted_capture
from tests.python.support.atomic_native_capture import save_phase,execute_captured_step
from tests.python.support.collective_checks import collective_call,collective_attempt,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]
SCHEMA='pops.m16-second-state-native-fixture@2'


def test_installed_affine_degree_five_selected_second_state(tmp_path,record_property,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2)
    world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    root=lambda:world is None or world.rank==0
    with collective_check(world):assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    declaration,layout,subjects,_,_=collective_call(world,case.build)
    validated=collective_call(world,lambda:pops.validate(declaration))
    resolved=collective_call(world,lambda:pops.resolve(validated,layout=layout,compile_options={"model_source_policy":"require"}))
    artifact=(collective_call(world,lambda:pops.compile(resolved)) if world is None else
        compile_resolved_plan_once(world,resolved,route='m16-selected-second-state',compile_artifact=pops.compile))
    directory=collective_directory(world,tmp_path/'second-state-affine')
    record_property('m16_second_state_provenance',str(directory/'provenance.json'))
    collective_call(world,lambda:retain_provenance(artifact,resolved,native,directory,blocks=case.PARTITION,require_model_sources=True) if root() else None)
    initial=case.initials()
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values={validated.resolve(subjects[label]):initial[label] for label in case.PARTITION},resources={'execution_context':artifact_execution_context(artifact)}))
    images={};phases=[];attempts=[];capture_errors=[]
    def receipt(status):
        if not root():return
        data={'schema':SCHEMA,'status':status,'partition':case.PARTITION,'components':{'spectator':case.DUMMY,'population':a.NAMES},
            'basis':a.BASIS.to_data(),'binding':[[list(i),a.BINDING[i]] for i in a.INDICES],
            'phases':phases,'clocks':{phase:images[phase][2] for phase in phases},'attempts':attempts,'capture_errors':capture_errors,
            'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file() and p.name!='receipt.json'},'root_scientific_approval':False}
        (directory/'receipt.json').write_text(json.dumps(data,sort_keys=True,allow_nan=False)+'\n')
    def capture():
        rows=[]
        for label,names in (('spectator',case.DUMMY),('population',a.NAMES)):
            rows.append(collective_call(world,lambda:np.array(runtime.block_level_state_global(label,0),copy=True)).reshape(len(names),a.N,a.N))
        blob=collective_call(world,lambda:bytes(runtime._executor.checkpoint_state_carriers()))
        clock=collective_call(world,lambda:(runtime.time(),runtime.macro_step(),runtime.n_levels()))
        return np.concatenate(rows,axis=0),blob,clock
    def save(phase,image):
        images[phase]=image;phases.append(phase)
        if root():save_phase(directory,phase,image)
        receipt('partial-captures')
    def failed(phase,failures):
        capture_errors.append({'phase':phase,'failures':failures});receipt('capture-failed')
    record_property('m16_second_state_receipt',str(directory/'receipt.json'))
    before=persisted_capture(world,capture,lambda image:save('before',image),lambda failures:failed('before',failures))
    def attempt(failures):attempts.append(failures);receipt('native-run-failed' if any(failures) else 'native-run-returned')
    def after():return persisted_capture(world,capture,lambda image:save('after',image),lambda failures:failed('after',failures))
    outcome,failures=collective_attempt(world,lambda:execute_captured_step(world,lambda:pops.run(runtime,t_end=a.DT,max_steps=1,console=False),after,attempt,lambda image,fail:receipt('run-captured')))
    with collective_check(world):
        assert not any(failures) and outcome is not None,failures
        report,_=outcome;assert report.accepted_steps==1 and not capture_errors
        post=images['after'];offset=len(case.DUMMY)
        expected_before=np.concatenate([initial[label] for label in case.PARTITION],axis=0)
        assert before[0].dtype==expected_before.dtype and before[0].shape==expected_before.shape and before[0].tobytes()==expected_before.tobytes()
        dummy_before=before[0][:offset];dummy_after=post[0][:offset]
        assert dummy_after.dtype==dummy_before.dtype and dummy_after.shape==dummy_before.shape and dummy_after.tobytes()==dummy_before.tobytes()
        expected=np.broadcast_to(a.atom_moments(1)[:,None,None],(len(a.NAMES),a.N,a.N))
        np.testing.assert_allclose(post[0][offset:],expected,rtol=0,atol=3e-13)
        assert post[2]==(a.DT,1,1)
        assert before[1].startswith(b'POPSCAR1') and post[1].startswith(b'POPSCAR1')
    collective_call(world,lambda:receipt('fixture-guards-passed'))
