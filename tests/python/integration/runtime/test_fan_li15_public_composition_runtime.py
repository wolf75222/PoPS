"""Future installed-SDK Fan--Li15 composition receipt; ROOT runs Native."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pops
import pytest
from tests.python.support.m17_fan_li_public_native_case import (
    N,DT,STEPS,original,make_case,initial,canonical,ssprk2,reference_proof)
from tests.python.support.collective_checks import collective_call,collective_check,collective_attempt
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once

pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]
SCHEMA='pops.fan-li15-public-composition-native-fixture@1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_program(artifact,resolved):
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact
    if type(artifact) is not CompiledSimulationArtifact:raise TypeError('exact compiled simulation artifact required')
    artifact.verify()
    ids={a.layout.qualified_id for a in resolved.layout_plan.assignments if a.subject_kind=='block' and a.subject.local_id=='gas'}
    if len(ids)!=1 or len(resolved.layout_plan.layouts)!=1:raise ValueError('one exact gas layout required')
    rows=tuple(row for row in artifact.layout_programs if row.layout_id in ids)
    if len(rows)!=1 or len(artifact.layout_programs)!=1:raise ValueError('exact gas layout program required')
    row,=rows;row.verify()
    if row.block_names!=('gas',) or artifact.program is not row.program:raise ValueError('compiled partition/program differs')
    return row


@pytest.mark.parametrize('permuted',[False,True],ids=['canonical','reverse'])
def test_installed_fan_li15_public_composition(permuted,tmp_path,record_property,
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2)
    world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),'installed package required'
    module=original();order=tuple(reversed(module.INDICES)) if permuted else module.INDICES
    case,layout=collective_call(world,lambda:make_case(order))
    resolved=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout))
    directory=collective_directory(world,tmp_path/'fan-li15-composition')
    def persist_source():
        if world is None or world.rank==0:
            (directory/'authoring.json').write_text(json.dumps({'schema':SCHEMA,'order':order,'cells':[N,N],
                'dt':DT,'steps':STEPS,'path':'normalized-analytic','original_Gauss4_default_preserved':True,
                'script_sha256':digest(module.__file__),'oracle_sha256':digest(sys.modules['api040_m17_oracle'].__file__),
                'case_support_sha256':digest(sys.modules['tests.python.support.m17_fan_li_public_native_case'].__file__),
                'fixture_sha256':digest(__file__),'hermite_reference_sha256':digest(Path(__file__).resolve().parents[2]/'support/fan_li15_oracle.py')},sort_keys=True,allow_nan=False)+'\n')
    collective_call(world,persist_source)
    artifact=(collective_call(world,lambda:pops.compile(resolved)) if world is None else
        compile_resolved_plan_once(world,resolved,route='fan-li15-public-'+str(permuted),compile_artifact=pops.compile))
    row=collective_call(world,lambda:select_program(artifact,resolved));program=row.program
    def persist_program():
        if world is None or world.rank==0:
            if type(program._generated_cpp) is not str:raise ValueError('retained actual C++ required')
            (directory/'program.cpp').write_text(program._generated_cpp);program.dump_ir(directory/'program.ir.json')
            (directory/'compiled-manifest.json').write_text(json.dumps(artifact.manifest().to_dict(),sort_keys=True,allow_nan=False)+'\n')
    collective_call(world,persist_program)
    phases=[];attempts=[];images=[];clocks=[]
    def receipt(status,math=None):
        if world is None or world.rank==0:
            data={'schema':SCHEMA,'status':status,'phases':phases,'clocks':clocks,'attempts':attempts,'math':math,
                'ranks':1 if world is None else int(world.size),'package':str(Path(pops.__file__).resolve()),
                'native':{'path':str(Path(native.__file__).resolve()),'sha256':digest(native.__file__)},
                'program':{'path':str(Path(program.so_path).resolve()),'sha256':digest(program.so_path),
                    'abi_key':program.abi_key,'problem_hash':program.problem_hash,'cache_key':program.cache_key,'layout_program':row.to_data()},
                'files':{p.name:digest(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name!='receipt.json'},
                'full_M17_qualification':False,'root_scientific_approval':False}
            (directory/'receipt.json').write_text(json.dumps(data,sort_keys=True,allow_nan=False)+'\n')
    record_property('fan_li15_composition_receipt',str(directory/'receipt.json'))
    bindings=resolved.initial_condition_plan.bindings
    with collective_check(world):
        assert len(bindings)==1 and bindings[0].subject.kind=='state' and bindings[0].subject.local_id=='raw_moments'
    subject,=tuple(binding.subject for binding in bindings)
    bind_error=None
    def bind_attempt():
        nonlocal bind_error
        try:return pops.bind(artifact,initial_values={subject:initial(order)},
            resources={'execution_context':artifact_execution_context(artifact)})
        except Exception as error:bind_error=error;raise
    runtime,bind_failures=collective_attempt(world,bind_attempt)
    attempts.append({'phase':'bind','failures':bind_failures})
    _,bind_save_failures=collective_attempt(world,lambda:receipt('native-bind-failed' if any(bind_failures) else 'bound'))
    if any(bind_failures):
        if bind_error is not None:bind_error.add_note(repr((bind_failures,bind_save_failures)));raise bind_error
        raise RuntimeError('peer Native bind failed: '+repr(bind_failures))
    assert not any(bind_save_failures),bind_save_failures
    def capture(phase):
        values=collective_call(world,lambda:np.array(runtime.state_global('gas'),copy=True).reshape(15,N,N))
        clock=collective_call(world,lambda:(runtime.time(),runtime.macro_step()))
        images.append(values);phases.append(phase);clocks.append(clock)
        def save():
            if world is None or world.rank==0:
                np.save(directory/(phase+'.npy'),values,allow_pickle=False)
                (directory/(phase+'.clock.json')).write_text(json.dumps(clock,allow_nan=False)+'\n')
        collective_call(world,save);collective_call(world,lambda:receipt('partial-captures'))
    capture('initial')
    for step in range(1,STEPS+1):
        original_error=None
        def run_attempt():
            nonlocal original_error
            try:return pops.run(runtime,t_end=step*DT,max_steps=1,console=False)
            except Exception as error:original_error=error;raise
        report,failures=collective_attempt(world,run_attempt);attempts.append({'step':step,'failures':failures})
        saved,save_failures=collective_attempt(world,lambda:receipt('native-run-failed' if any(failures) else 'partial-captures'))
        if any(failures):
            if original_error is not None:original_error.add_note(repr((failures,save_failures)));raise original_error
            raise RuntimeError('peer Native run failed: '+repr(failures))
        assert not any(save_failures),save_failures
        capture('accepted'+str(step))
        with collective_check(world):assert report.accepted_steps==1
    math=None
    with collective_check(world):
        reference=canonical(images[0],order)[:,0,:];math=reference_proof(reference)
        assert math['h3_max']>1e-8 and math['h4_max']>1e-8
        assert math['regularized_face_max']>module.CRITERIA['nonconservative_max_min']
        assert math['active_vs_distinct_B0_step_gap']>1e-8
        assert math['quadrature24_48_step_gap']<module.CRITERIA['path_quadrature_gap']
        for step,image in enumerate(images):
            raw=canonical(image,order);saved=np.load(directory/(phases[step]+'.npy'),allow_pickle=False)
            assert saved.shape==image.shape and saved.dtype==image.dtype and saved.tobytes()==image.tobytes()
            assert abs(clocks[step][0]-step*DT)<module.CRITERIA['time_error'] and clocks[step][1]==step
            np.testing.assert_allclose(raw,np.broadcast_to(reference[:,None,:],raw.shape),rtol=0,atol=module.CRITERIA['oracle_max_error'])
            assert np.max(np.abs(raw-raw[:,0:1,:]))<module.CRITERIA['y_invariance']
            for k,index in enumerate(module.INDICES):
                if sum(index)<4:assert abs(raw[k].mean()-canonical(images[0],order)[k].mean())<module.CRITERIA['conserved_inventory']
            rho=raw[0];u=raw[1]/rho;v=raw[5]/rho;a=raw[2]/rho-u*u;b=raw[6]/rho-u*v;c=raw[9]/rho-v*v
            assert np.all(rho>0) and np.all(a>0) and np.all(c>0) and np.all(a*c-b*b>0)
            reference=ssprk2(reference)
        assert np.max(np.abs(images[0]-initial(order)))<module.CRITERIA['initial_max_error']
    collective_call(world,lambda:receipt('fixture-guards-passed',math))
