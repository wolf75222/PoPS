"""Prospective installed VP witness. Source collection is not Native qualification."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pops
from pops._generated_release_contract import NATIVE_ABI_VERSION
import pytest
from tests.python.support.m19_vlasov_poisson_case import build, DT
from tests.python.support.m19_vlasov_poisson_oracle import initial, step
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.evolved_stage_v_capture import retain_v_provenance
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
NAMES=('a_spectator','m_number','z_phase')


def save(path,value):
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False,indent=2)+'\n')


def pin(path):
    return dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.compiler
@pytest.mark.native_loader
def test_installed_two_stage_vlasov_poisson(tmp_path,record_property,
        isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2)
    world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    rank=0 if world is None else int(world.rank)
    ranks=1 if world is None else int(world.size)
    directory=collective_directory(world,tmp_path/'vlasov-poisson')
    collective_call(world,lambda:directory.mkdir(exist_ok=True))
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert Path(native.__file__).resolve().is_file()
        capability=native.module_capabilities('production')
        assert capability['abi_version']==NATIVE_ABI_VERSION
        assert capability['mapped_consumed_field_output'] is True
    providers=directory/'providers'
    collective_call(world,lambda:providers.mkdir(exist_ok=True))
    if world is None:
        resolved=collective_call(world,lambda:build(providers))
    else:
        from tests.python.integration.runtime.test_m19_product_support_runtime import _load_published_provider
        resolved=collective_call(world,lambda:build(providers) if rank==0 else None)
        resolved=collective_call(world,lambda:resolved if rank==0 else build(providers,provider_factory=_load_published_provider))
    artifact=pops.compile(resolved) if world is None else compile_resolved_plan_once(
        world,resolved,route='finite velocity Vlasov Poisson',compile_artifact=pops.compile)
    collective_call(world,lambda:retain_v_provenance(artifact,native,directory,rank))
    collective_call(world,lambda:save(directory/('prepared-rank%d.json'%rank),dict(
        schema='sol61.finite-velocity-vlasov-poisson@1',rank=rank,ranks=ranks,
        dt=[DT.numerator,DT.denominator],steps=2,cells=[4,8],axes=['velocity','position'],
        velocity_boundary='zero flux',position_boundary='periodic',epsilon=1,
        charge=1,background_charge=-1,gauge='mean zero',temporal_method='SSPRK2',
        field_at_each_stage=True,actual_model_sources_before_bind=True,
        artifact=artifact.artifact_identity.token,native=pin(Path(native.__file__).resolve()),
        full_field_image_received=False,ghost_formula_received=False,root_received=False)))
    f=initial()
    initial_state={'z_phase':f[None,:,:], 'm_number':(.5*f.sum(axis=0))[None,:,None],
        'a_spectator':np.stack((np.full((4,8),7.),np.full((4,8),-13.)))}
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_state=initial_state,
        resources={'execution_context':artifact_execution_context(artifact)}))
    def capture(label):
        files={}
        clock=collective_call(world,lambda:(runtime.time(),runtime.macro_step()))
        cursors=collective_call(world,lambda:runtime.consumer_cursors.to_data())
        owners={name:collective_call(world,lambda name=name:runtime.local_boxes(name)) for name in NAMES}
        def receipt(complete):
            save(directory/('%s-rank%d.json'%(label,rank)),dict(schema='sol61.vp-state@1',
                phase=label,rank=rank,ranks=ranks,clock=clock,consumer_cursors=cursors,
                local_boxes=owners,files=files,capture_complete=complete,
                storage_route='public composite checkpoint',ghost_formula_received=False))
        collective_call(world,lambda:receipt(False))
        for name in NAMES:
            value=collective_call(world,lambda name=name:runtime.state_global(name))
            path=directory/('%s-rank%d-%s.npy'%(label,rank,name))
            collective_call(world,lambda:np.save(path,value,allow_pickle=False))
            files[name]=pin(path)
            collective_call(world,lambda:receipt(False))
        checkpoint=collective_call(world,lambda:runtime.checkpoint(directory/(label+'-checkpoint')))
        collective_call(world,lambda:save(directory/('%s-cp-rank%d.json'%(label,rank)),pin(Path(checkpoint))))
        collective_call(world,lambda:receipt(True))
        return files
    capture('initial')
    expected=f.copy()
    for index in (1,2):
        original=None
        def advance():
            nonlocal original
            try:return pops.run(runtime,t_end=index*float(DT),max_steps=1,console=False)
            except Exception as error:original=error;raise
        _,failures=collective_attempt(world,advance)
        collective_call(world,lambda:save(directory/('attempt%d-rank%d.json'%(index,rank)),dict(failures=failures)))
        if any(failures):
            try:capture('failed%d'%index)
            except Exception as capture_error:
                if original is not None:
                    original.add_note('capture failure: '+repr(capture_error));raise original
                raise
            if original is not None:raise original
            raise RuntimeError('peer VP step failed: '+repr(failures))
        actual=capture('accepted%d'%index)
        expected,stages=step(expected)
        with collective_check(world):
            result=np.load(actual['z_phase']['file'],allow_pickle=False)
            assert np.isfinite(result).all() and np.min(result)>0
            np.testing.assert_allclose(result,expected[None,:,:],rtol=0,atol=2e-11)
            # This State stores the last predictor moment, not an endpoint Field cache.
            np.testing.assert_allclose(np.load(actual['m_number']['file'],allow_pickle=False),
                stages['predictor_rho'][None,:,None],rtol=0,atol=2e-11)
            np.testing.assert_array_equal(np.load(actual['a_spectator']['file'],allow_pickle=False),initial_state['a_spectator'])
            assert runtime.time()==index*float(DT) and runtime.macro_step()==index
    record_property('vp_receipt',str(directory/('prepared-rank%d.json'%rank)))
