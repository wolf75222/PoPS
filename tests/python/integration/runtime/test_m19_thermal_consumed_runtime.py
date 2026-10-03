"""Future genuine installed-package witness; never executed as Source qualification."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pops
import pytest
from tests.python.support.m19_thermal_consumed_case import build, reference, NAMES
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.evolved_stage_v_capture import retain_v_provenance
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once

def save_json(path,value):
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False,indent=2)+'\n')

def pin(path):
    return {'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize('gradient',(False,True))
@pytest.mark.parametrize('reverse',(False,True))
def test_installed_thermal_consumed_field_two_destinations(tmp_path,record_property,
        isolated_native_cache,native_cxx,kokkos_root,gradient,reverse):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2)
    world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    rank=0 if world is None else int(world.rank)
    directory=collective_directory(world,tmp_path/'thermal-consumed-field')
    collective_call(world,lambda:directory.mkdir(exist_ok=True))
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(__import__('sys').prefix).resolve())
        assert Path(native.__file__).resolve().is_file()
        assert native.module_capabilities('production')['abi_version']==9
        assert native.module_capabilities('production')['mapped_consumed_field_output'] is True
    provider_directory=directory/'providers'
    collective_call(world,lambda:provider_directory.mkdir(exist_ok=True))
    # Peers load the exact published source package, never overwrite it.
    if world is not None:
        from tests.python.integration.runtime.test_m19_product_support_runtime import _load_published_provider
        factory=None if rank==0 else _load_published_provider
    else: factory=None
    kwargs={} if factory is None else {'provider_factory':factory}
    resolved=collective_call(world,lambda:build(provider_directory,gradient=gradient,reverse=reverse) if rank==0 else None)
    if world is not None:
        resolved=collective_call(world,lambda:resolved if rank==0 else build(provider_directory,gradient=gradient,reverse=reverse,**kwargs))
    artifact=pops.compile(resolved) if world is None else compile_resolved_plan_once(world,resolved,
        route='thermal consumed Field',compile_artifact=pops.compile)
    collective_call(world,lambda:retain_v_provenance(artifact,native,directory,rank))
    collective_call(world,lambda:save_json(directory/('prepared-rank%d.json'%rank),{
        'schema':'sol61.nonkinetic-thermal-consumed-field@1','rank':rank,
        'ranks':1 if world is None else int(world.size),'gradient':gradient,'reverse':reverse,
        'artifact':artifact.artifact_identity.token,'package':str(Path(pops.__file__).resolve()),
        'native':pin(Path(native.__file__).resolve()),'C25_before_bind':True,'root_received':False}))
    initial,expected,phi,load=reference(gradient=gradient)
    collective_call(world,lambda:np.save(directory/('reference-phi-rank%d.npy'%rank),phi,allow_pickle=False))
    collective_call(world,lambda:np.save(directory/('reference-load-rank%d.npy'%rank),load,allow_pickle=False))
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_state=initial,
        resources={'execution_context':artifact_execution_context(artifact)}))
    def capture(label):
        # Each actual getter result is durable before the next getter or guard.
        files={}
        clock=collective_call(world,lambda:(runtime.time(),runtime.macro_step()))
        owners={}
        for name in NAMES:
            owners[name]=collective_call(world,lambda name=name:runtime.local_boxes(name))
        cursors=collective_call(world,lambda:runtime.consumer_cursors.to_data())
        def metadata(complete):
            save_json(directory/('%s-rank%d.json'%(label,rank)),{
                'schema':'sol61.thermal-state@1','phase':label,'rank':rank,
                'ranks':1 if world is None else int(world.size),'clock':list(clock),
                'consumer_cursors':cursors,'local_boxes':owners,'files':files,
                'capture_complete':complete,
                'storage_route':'public composite checkpoint; no private observer',
                'ghost_formula_received':False})
        collective_call(world,lambda:metadata(False))
        for name in NAMES:
            value=collective_call(world,lambda name=name:runtime.state_global(name))
            path=directory/('%s-rank%d-%s.npy'%(label,rank,name))
            collective_call(world,lambda value=value,path=path:np.save(path,value,allow_pickle=False))
            files[name]=pin(path)
            collective_call(world,lambda:metadata(False))
        # Public collective checkpoint carries each child full-storage contract.
        checkpoint=collective_call(world,lambda:runtime.checkpoint(directory/(label+'-checkpoint')))
        collective_call(world,lambda:save_json(directory/('%s-cp-rank%d.json'%(label,rank)),pin(Path(checkpoint))))
        collective_call(world,lambda:metadata(True))
        return files
    capture('initial')
    for step in (1,2):
        error=None
        def advance():
            nonlocal error
            try:return pops.run(runtime,t_end=step/64,max_steps=1,console=False)
            except Exception as caught:error=caught;raise
        _,failures=collective_attempt(world,advance)
        collective_call(world,lambda:save_json(directory/('step%d-rank%d-attempt.json'%(step,rank)),{'failures':failures}))
        if any(failures):
            try:capture('failed-step%d'%step)
            except Exception as capture_error:
                if error is not None:
                    error.add_note('failed-state capture error: '+repr(capture_error))
                    raise error
                raise
            if error is not None:raise error
            raise RuntimeError('peer Native thermal step failed: '+repr(failures))
        files=capture('accepted%d'%step)
        _,step_expected,_,_=reference(gradient=gradient,steps=step)
        with collective_check(world):
            for name in NAMES:
                actual=np.load(files[name]['file'],allow_pickle=False)
                assert np.all(np.isfinite(actual)) and np.all(actual!=0)
                np.testing.assert_allclose(actual,step_expected[name],rtol=0,atol=2e-11)
            assert runtime.macro_step()==step and runtime.time()==step/64
    record_property('thermal_receipt',str(directory/('prepared-rank%d.json'%rank)))
