"""True installed Native witness; Source preparation is not Native reception."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pytest
import pops
from tests.python.support import scoped_wave_transport as s
from tests.python.support.atomic_native_capture import execute_captured_step,execute_captured_bind,save_phase
from tests.python.support.m16_explicit_native_capture import retain_provenance,persisted_capture
from tests.python.support.collective_checks import collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]


def test_installed_independent_state_scoped_wave_transports(tmp_path,record_property,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2);world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    root=lambda:world is None or world.rank==0
    with collective_check(world):assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case,layout,subjects=collective_call(world,s.build)
    validated=collective_call(world,lambda:pops.validate(case))
    # Public C25@2 required-source admission: no regeneration or private source fallback.
    resolved=collective_call(world,lambda:pops.resolve(validated,layout=layout,compile_options={'model_source_policy':'require'}))
    artifact=(collective_call(world,lambda:pops.compile(resolved)) if world is None else
        compile_resolved_plan_once(world,resolved,route='scoped-flux-waves',compile_artifact=pops.compile))
    directory=collective_directory(world,tmp_path/'scoped-wave-transport')
    collective_call(world,lambda:retain_provenance(artifact,resolved,native,directory,blocks=s.PARTITION,require_model_sources=True) if root() else None)
    record_property('scoped_wave_provenance',str(directory/'provenance.json'))
    record_property('scoped_wave_receipt',str(directory/'receipt.json'))
    initial=s.initial()
    def bind_failure(error,failures):
        path=directory/('bind-failure-rank'+str(0 if world is None else world.rank)+'.json')
        path.write_text(json.dumps({'schema':'pops.scoped-wave-bind-failure@1','exception':error,'collective_failures':failures},sort_keys=True,allow_nan=False)+'\n')
    runtime=execute_captured_bind(world,lambda:pops.bind(artifact,initial_values={validated.resolve(subjects[label]):initial[label] for label in s.PARTITION},resources={'execution_context':artifact_execution_context(artifact)}),bind_failure)
    images={};attempts=[];capture_errors=[]
    def receipt(status):
        if root():
            data={'schema':'pops.scoped-flux-wave-native@1','status':status,'partition':s.PARTITION,'components':s.NAMES,'velocities':s.VELOCITIES,'cells':s.N,'dt':s.DT,'attempts':attempts,'capture_errors':capture_errors,'phases':list(images),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file() and p.name!='receipt.json'},'root_scientific_approval':False}
            (directory/'receipt.json').write_text(json.dumps(data,sort_keys=True,allow_nan=False)+'\n')
    def capture():
        arrays=[collective_call(world,lambda label=label,names=names:np.array(runtime.block_level_state_global(label,0),copy=True).reshape(len(names),s.N,s.N)) for label,names in zip(s.PARTITION,s.NAMES)]
        blob=collective_call(world,lambda:bytes(runtime._executor.checkpoint_state_carriers()))
        clock=collective_call(world,lambda:(runtime.time(),runtime.macro_step(),runtime.n_levels()))
        return np.concatenate(arrays),blob,clock
    def save(phase,image):
        images[phase]=image
        if root():save_phase(directory,phase,image)
        receipt('captured')
    def failed(phase,failures):capture_errors.append({'phase':phase,'failures':failures});receipt('capture-failed')
    def retained(phase):return persisted_capture(world,capture,lambda image:save(phase,image),lambda failures:failed(phase,failures))
    before=retained('initial')
    expected=[initial[label] for label in s.PARTITION]
    for step in (1,2):
        phase='accepted'+str(step)
        def attempt(failures):attempts.append({'step':step,'failures':failures});receipt('run-returned')
        report,image=execute_captured_step(world,lambda:pops.run(runtime,t_end=step*s.DT,max_steps=1,console=False),lambda:retained(phase),attempt,lambda image,fail:receipt('run-captured'))
        expected=[s.step(q,k) for k,q in enumerate(expected)]
        with collective_check(world):
            assert report.accepted_steps==1 and not capture_errors
            assert image[2]==(step*s.DT,step,1)
            assert image[1].startswith(b'POPSCAR1')
            np.testing.assert_allclose(image[0],np.concatenate(expected),rtol=0,atol=3e-13)
            np.testing.assert_allclose(image[0].sum(axis=(1,2)),before[0].sum(axis=(1,2)),rtol=0,atol=3e-12)
            assert np.min(image[0][-1])>0
    with collective_check(world):
        assert before[0].tobytes()==np.concatenate([initial[label] for label in s.PARTITION]).tobytes()
        assert before[2]==(0.,0,1) and before[1].startswith(b'POPSCAR1')
    collective_call(world,lambda:receipt('fixture-guards-passed'))
