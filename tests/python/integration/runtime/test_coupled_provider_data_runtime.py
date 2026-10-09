"""Prospective installed non-BGK provider consumption; no Source/Native conflation."""
from pathlib import Path
import sys
import numpy as np
import pops
import pytest
from pops._generated_release_contract import NATIVE_ABI_VERSION
from tests.python.support.coupled_field_data_case import build,DT,FRACTIONS,NAMES
from tests.python.support.coupled_provider_data_oracle import data,step
from tests.python.support.collective_checks import collective_attempt,collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.evolved_stage_v_capture import retain_v_provenance,pin,save_json
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once

@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize('cells,reverse,permuted',(((7,3),False,False),((3,7),True,True)))
def test_installed_two_fractional_coupled_provider_reads(cells,reverse,permuted,tmp_path,
        record_property,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2)
    world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    rank=0 if world is None else int(world.rank);ranks=1 if world is None else int(world.size)
    directory=collective_directory(world,tmp_path/'coupled-provider-data')
    collective_call(world,lambda:directory.mkdir(exist_ok=True))
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert native.module_capabilities('production')['abi_version']==NATIVE_ABI_VERSION
    resolved=collective_call(world,lambda:build(cells=cells,reverse=reverse,permuted=permuted))
    artifact=collective_call(world,lambda:pops.compile(resolved)) if world is None else compile_resolved_plan_once(
        world,resolved,route='coupled-provider-data-%s-%s-%s'%(cells,reverse,permuted),compile_artifact=pops.compile)
    collective_call(world,lambda:retain_v_provenance(artifact,native,directory,rank))
    collective_call(world,lambda:save_json(directory/('prepared-rank%d.json'%rank),dict(
        schema='sol61.coupled-provider-data@1',rank=rank,ranks=ranks,cells=cells,
        reversed_declarations=reverse,permuted_inputs=permuted,readonly_catalyst=True,
        fractions=[[x.numerator,x.denominator] for x in FRACTIONS],dt=[DT.numerator,DT.denominator],
        formula='q=(2+(1+x+2*y)^2)*c*(b-a), response_rate=2+(1+x+2*y)^2',
        provider_time_dependent=False,actual_complete_C25_before_bind=True,
        artifact=artifact.artifact_identity.token,native=pin(native.__file__),
        capabilities=dict(native.module_capabilities('production')),root_received=False)))
    (a,b,c),gain=collective_call(world,lambda:data(cells))
    seed=dict(zip(NAMES,(a,b,c),strict=True))
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_state=seed,
        resources={'execution_context':artifact_execution_context(artifact)}))
    def capture(label):
        clock=collective_call(world,lambda:(runtime.time(),runtime.macro_step()))
        cursors=collective_call(world,lambda:runtime.consumer_cursors.to_data())
        owners={name:collective_call(world,lambda name=name:runtime.local_boxes(name)) for name in NAMES}
        files={};receipt=directory/('%s-rank%d.json'%(label,rank))
        def write(complete):return save_json(receipt,dict(schema='sol61.coupled-data-state@1',phase=label,
            rank=rank,ranks=ranks,clock=clock,consumer_cursors=cursors,local_boxes=owners,
            files=files,capture_complete=complete,storage_route='public composite CP9 checkpoint'))
        collective_call(world,lambda:write(False))
        for name in NAMES:
            array=collective_call(world,lambda name=name:runtime.state_global(name))
            path=directory/('%s-rank%d-%s.npy'%(label,rank,name))
            collective_call(world,lambda:np.save(path,array,allow_pickle=False))
            files[name]=collective_call(world,lambda:pin(path));collective_call(world,lambda:write(False))
        checkpoint=collective_call(world,lambda:runtime.checkpoint(directory/(label+'-checkpoint')))
        collective_call(world,lambda:save_json(directory/('%s-cp-rank%d.json'%(label,rank)),pin(checkpoint)))
        collective_call(world,lambda:write(True))
        return files
    capture('initial');expected=(a.copy(),b.copy(),c.copy())
    for index in (1,2):
        original=None
        def advance():
            nonlocal original
            try:return pops.run(runtime,t_end=index*float(DT),max_steps=1,console=False)
            except Exception as error:original=error;raise
        _,failures=collective_attempt(world,advance)
        collective_call(world,lambda:save_json(directory/('attempt%d-rank%d.json'%(index,rank)),dict(failures=failures)))
        if any(failures):
            try:capture('failed%d'%index)
            except Exception as capture_error:
                if original is not None:
                    original.add_note('capture failure: '+repr(capture_error));raise original
                raise
            if original is not None:raise original
            raise RuntimeError('peer coupled-data step failed: '+repr(failures))
        actual=capture('accepted%d'%index);expected=step(expected,gain)
        with collective_check(world):
            values=tuple(np.load(actual[name]['path'],allow_pickle=False) for name in NAMES)
            assert all(np.isfinite(value).all() for value in values)
            for value,reference in zip(values,expected,strict=True):
                np.testing.assert_allclose(value,reference,rtol=0,atol=2e-12)
            np.testing.assert_allclose(values[0][0]+values[1][0],a[0]+b[0],rtol=0,atol=2e-12)
            np.testing.assert_array_equal(values[2],c)
            assert values[2].dtype==c.dtype and values[2].tobytes()==c.tobytes()
            assert runtime.time()==index*float(DT) and runtime.macro_step()==index
    record_property('coupled_data_receipt',str(directory/('prepared-rank%d.json'%rank)))
