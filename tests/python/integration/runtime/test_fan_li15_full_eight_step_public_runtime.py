"""Original Uniform Fan–Li15 eight-step public composition, installed SDK only."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pops
import pytest
from tests.python.support import m17_fan_li_public_native_case as support
from tests.python.integration.runtime.test_fan_li15_public_composition_runtime import select_program,layout_program_json_identity
from tests.python.support.atomic_native_capture import execute_captured_step,execute_captured_bind
from tests.python.support.m16_explicit_native_capture import persisted_capture
from tests.python.support.collective_checks import collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]
SCHEMA='pops.fan-li15-public-composition-native-fixture@2'
STEPS=8
CARRIER_CAPTURE={'status':'unavailable','target':'system','scope':'Uniform valid arrays and clocks only; no grown/checkpoint/restart qualification'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retain_sources(artifact,resolved,native,directory):
    row=select_program(artifact,resolved)
    if row.target!='system' or len(artifact.blocks)!=1 or {block.name for block in artifact.blocks}!={'gas'}:
        raise ValueError('exact full Uniform gas model partition required')
    models=[]
    for block in artifact.blocks:
        evidence=block.model.source_provenance(require_complete=True)
        filename=block.name+'.model.cpp';block.model.dump_cpp(directory/filename)
        models.append({'block':block.name,'state_spaces':list(block.state_spaces),
            'binary':{'path':str(Path(block.model.so_path).resolve()),'sha256':digest(block.model.so_path)},
            'actual_source':evidence,'source_file':filename,'source_sha256':digest(directory/filename)})
    # This is the actual verified compile wrapper, not an advanced regeneration handle.
    row.program.dump_cpp(directory/'program.cpp');row.program.dump_ir(directory/'program.ir.json')
    (directory/'compiled-manifest.json').write_text(json.dumps(artifact.manifest().to_dict(),sort_keys=True,allow_nan=False)+'\n')
    proof={'schema':'pops.fan-li15-full-uniform-provenance@1','package':str(Path(pops.__file__).resolve()),
        'native':{'path':str(Path(native.__file__).resolve()),'sha256':digest(native.__file__)},
        'artifact_identity':artifact.artifact_identity.token,'layout_program':layout_program_json_identity(row),
        'program':{'path':str(Path(row.program.so_path).resolve()),'sha256':digest(row.program.so_path),
            'abi_key':row.program.abi_key,'problem_hash':row.program.problem_hash,'cache_key':row.program.cache_key},
        'models':models,'files':{name:digest(directory/name) for name in ('program.cpp','program.ir.json','compiled-manifest.json',*[m['source_file'] for m in models])},
        'root_scientific_approval':False}
    (directory/'provenance.json').write_text(json.dumps(proof,sort_keys=True,allow_nan=False)+'\n')
    return row


def save_phase(directory,phase,image):
    values,clock=image
    np.save(directory/(phase+'.npy'),values,allow_pickle=False)
    (directory/(phase+'.clock.json')).write_text(json.dumps(clock,allow_nan=False)+'\n')
    (directory/(phase+'.capture.json')).write_text(json.dumps({'schema':SCHEMA,'phase':phase,'status':'captured',
        'carrier_capture':CARRIER_CAPTURE,'shape':values.shape,'dtype':str(values.dtype)},sort_keys=True,allow_nan=False)+'\n')


@pytest.mark.parametrize('permuted',(False,True),ids=('canonical','reverse'))
def test_installed_original_fan_li_eight_steps(permuted,tmp_path,record_property,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2);world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    root=lambda:world is None or world.rank==0
    with collective_check(world):assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    module=support.original()
    with collective_check(world):assert (module.N,module.DT,module.STEPS)==(16,1e-4,STEPS)
    order=tuple(reversed(module.INDICES)) if permuted else module.INDICES
    case,layout=collective_call(world,lambda:support.make_case(order))
    validated=collective_call(world,lambda:pops.validate(case))
    resolved=collective_call(world,lambda:pops.resolve(validated,layout=layout,compile_options={'model_source_policy':'require'}))
    directory=collective_directory(world,tmp_path/'fan-li15-full-eight')
    record_property('fan_li15_full_uniform_receipt',str(directory/'receipt.json'))
    record_property('fan_li15_full_uniform_provenance',str(directory/'provenance.json'))
    def authoring():
        if root():
            data={'schema':SCHEMA,'order':order,'cells':[support.N,support.N],'dt':support.DT,'steps':STEPS,
                'layout':'Uniform','path':'normalized-analytic','original_Gauss4_default_preserved':True,
                'criteria':module.CRITERIA,'carrier_capture':CARRIER_CAPTURE,
                'script_sha256':digest(module.__file__),'support_sha256':digest(support.__file__),'fixture_sha256':digest(__file__),
                'oracle_sha256':digest(sys.modules['api040_m17_oracle'].__file__),'root_scientific_approval':False}
            (directory/'authoring.json').write_text(json.dumps(data,sort_keys=True,allow_nan=False)+'\n')
    collective_call(world,authoring)
    artifact=(collective_call(world,lambda:pops.compile(resolved)) if world is None else
        compile_resolved_plan_once(world,resolved,route='fan-li15-full-eight-'+str(permuted),compile_artifact=pops.compile))
    collective_call(world,lambda:retain_sources(artifact,resolved,native,directory) if root() else None)
    bindings=resolved.initial_condition_plan.bindings
    with collective_check(world):
        assert len(bindings)==1 and bindings[0].subject.kind=='state' and bindings[0].subject.local_id=='raw_moments'
    subject,=tuple(b.subject for b in bindings)
    def bind_failed(error,failures):
        (directory/('bind-failure-rank'+str(0 if world is None else world.rank)+'.json')).write_text(json.dumps({'schema':SCHEMA,'exception':error,'failures':failures},sort_keys=True,allow_nan=False)+'\n')
    runtime=execute_captured_bind(world,lambda:pops.bind(artifact,initial_values={subject:support.initial(order)},resources={'execution_context':artifact_execution_context(artifact)}),bind_failed)
    images={};attempts=[];capture_errors=[];math=None
    def receipt(status):
        if root():
            data={'schema':SCHEMA,'status':status,'order':order,'steps':STEPS,'ranks':1 if world is None else world.size,
                'phases':list(images),'clocks':{phase:image[1] for phase,image in images.items()},'attempts':attempts,
                'capture_errors':capture_errors,'math':math,'carrier_capture':CARRIER_CAPTURE,
                'files':{p.name:digest(p) for p in directory.iterdir() if p.is_file() and p.name!='receipt.json'},
                'full_M17_qualification':False,'root_scientific_approval':False}
            (directory/'receipt.json').write_text(json.dumps(data,sort_keys=True,allow_nan=False)+'\n')
    def capture():
        values=collective_call(world,lambda:np.array(runtime.state_global('gas'),copy=True).reshape(15,support.N,support.N))
        clock=collective_call(world,lambda:(runtime.time(),runtime.macro_step()))
        return values,clock
    def save(phase,image):
        if root():save_phase(directory,phase,image)
        images[phase]=image;receipt('partial-captures')
    def failed(phase,failures):capture_errors.append({'phase':phase,'failures':failures});receipt('capture-failed')
    def retained(phase):return persisted_capture(world,capture,lambda image:save(phase,image),lambda failures:failed(phase,failures))
    retained('initial')
    for step in range(1,STEPS+1):
        def attempt(failures):attempts.append({'step':step,'failures':failures});receipt('run-failed' if any(failures) else 'run-returned')
        report,_=execute_captured_step(world,lambda:pops.run(runtime,t_end=step*support.DT,max_steps=1,console=False),lambda:retained(('rejected-attempt' if any(attempts[-1]['failures']) else 'accepted')+str(step)),attempt,lambda image,fail:receipt('native-failure-captured' if fail else 'run-captured'))
        with collective_check(world):assert report.accepted_steps==1
    with collective_check(world):
        assert tuple(images)==('initial',)+tuple('accepted'+str(i) for i in range(1,STEPS+1)) and not capture_errors
        reference=support.canonical(images['initial'][0],order)[:,0,:]
        math=support.reference_proof(reference)
        assert math['h3_max']>1e-8 and math['h4_max']>1e-8
        assert math['regularized_face_max']>module.CRITERIA['nonconservative_max_min']
        assert math['active_vs_distinct_B0_step_gap']>1e-8
        assert math['quadrature24_48_step_gap']<module.CRITERIA['path_quadrature_gap']
        initial=support.canonical(images['initial'][0],order)
        for step,(phase,(values,clock)) in enumerate(images.items()):
            raw=support.canonical(values,order)
            assert abs(clock[0]-step*support.DT)<module.CRITERIA['time_error'] and clock[1]==step
            np.testing.assert_allclose(raw,np.broadcast_to(reference[:,None,:],raw.shape),rtol=0,atol=module.CRITERIA['oracle_max_error'])
            assert np.max(np.abs(raw-raw[:,0:1,:]))<module.CRITERIA['y_invariance']
            for k,index in enumerate(module.INDICES):
                if sum(index)<4:assert abs(raw[k].mean()-initial[k].mean())<module.CRITERIA['conserved_inventory']
            rho=raw[0];u=raw[1]/rho;v=raw[5]/rho;a=raw[2]/rho-u*u;b=raw[6]/rho-u*v;c=raw[9]/rho-v*v
            assert np.all(rho>0) and np.all(a>0) and np.all(c>0) and np.all(a*c-b*b>0)
            if step<STEPS:reference=support.ssprk2(reference)
        assert np.max(np.abs(images['initial'][0]-support.initial(order)))<module.CRITERIA['initial_max_error']
        dop=sys.modules['api040_m17_oracle'].solve_reference(initial[:,0,:],STEPS*support.DT)
        np.testing.assert_allclose(support.canonical(images['accepted8'][0],order),np.broadcast_to(dop[:,None,:],initial.shape),rtol=0,atol=module.CRITERIA['oracle_max_error'])
    collective_call(world,lambda:receipt('fixture-guards-passed'))
