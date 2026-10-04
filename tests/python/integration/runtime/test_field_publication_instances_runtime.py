"""Prospective installed per-instance solved-field publication; requires a fresh Native build."""
from pathlib import Path
import sys
import numpy as np
import pops
import pytest
from pops._generated_release_contract import NATIVE_ABI_VERSION
from tests.python.support.field_publication_instance_case import build,DT,FRACTIONS,NAMES
from tests.python.support.field_publication_instance_oracle import data,step
from tests.python.support.collective_checks import collective_attempt,collective_call,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.evolved_stage_v_capture import retain_v_provenance,pin,save_json
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once

@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize('cells,reverse,different',(((7,3),False,False),((11,2),True,True)))
def test_installed_three_instance_solved_provider_reads(cells,reverse,different,tmp_path,
        record_property,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2)
    world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    rank=0 if world is None else int(world.rank);ranks=1 if world is None else int(world.size)
    directory=collective_directory(world,tmp_path/'field-publication-instance')
    collective_call(world,lambda:directory.mkdir(exist_ok=True))
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        assert native.module_capabilities('production')['abi_version']==NATIVE_ABI_VERSION
    resolved=collective_call(world,lambda:build(cells=cells,reverse=reverse,permuted=False,different_values=different,first_beta=different))
    artifact=collective_call(world,lambda:pops.compile(resolved)) if world is None else compile_resolved_plan_once(
        world,resolved,route='field-publication-instance-%s-%s-%s'%(cells,reverse,different),compile_artifact=pops.compile)
    collective_call(world,lambda:retain_v_provenance(artifact,native,directory,rank))
    collective_call(world,lambda:save_json(directory/('prepared-rank%d.json'%rank),dict(
        schema='sol61.field-publication-instance@1',rank=rank,ranks=ranks,cells=cells,
        reversed_declarations=reverse,first_beta=different,different_publications=different,readonly_catalyst=True,
        fractions=[[x.numerator,x.denominator] for x in FRACTIONS],dt=[DT.numerator,DT.denominator],
        formula='q=(2+phi_stage^2)*c*(b-a), response_rate=2+phi_stage^2',
        provider_time_dependent=True,stage_time_source='declared FieldProblem solved from candidate forcing carrier',actual_complete_C25_before_bind=True,
        artifact=artifact.artifact_identity.token,native=pin(native.__file__),
        capabilities=dict(native.module_capabilities('production')),root_received=False)))
    a,b,c,forcing=collective_call(world,lambda:data(cells))
    seed=dict(zip(NAMES,(a,b,c,forcing),strict=True))
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_state=seed,
        resources={'execution_context':artifact_execution_context(artifact)}))
    def capture(label):
        clock=collective_call(world,lambda:(runtime.time(),runtime.macro_step()))
        cursors=collective_call(world,lambda:runtime.consumer_cursors.to_data())
        owners={name:collective_call(world,lambda name=name:runtime.local_boxes(name)) for name in NAMES}
        files={};receipt=directory/('%s-rank%d.json'%(label,rank))
        def write(complete):return save_json(receipt,dict(schema='sol61.field-instance-state@1',phase=label,
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
    capture('initial');expected=(a.copy(),b.copy(),c.copy(),forcing.copy())
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
        actual=capture('accepted%d'%index);expected=step(expected,cells,(index-1)*float(DT),different=different)
        with collective_check(world):
            values=tuple(np.load(actual[name]['path'],allow_pickle=False) for name in NAMES)
            assert all(np.isfinite(value).all() for value in values)
            for value,reference in zip(values,expected,strict=True):
                np.testing.assert_allclose(value,reference,rtol=0,atol=2e-12)
            np.testing.assert_allclose(values[0][0]+values[1][0],a[0]+b[0],rtol=0,atol=2e-12)
            np.testing.assert_array_equal(values[2],c)
            assert values[2].dtype==c.dtype and values[2].tobytes()==c.tobytes()
            assert runtime.time()==index*float(DT) and runtime.macro_step()==index
    record_property('field_instance_receipt',str(directory/('prepared-rank%d.json'%rank)))
