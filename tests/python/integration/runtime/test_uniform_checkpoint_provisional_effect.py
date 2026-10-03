"""Prospective installed CP9 publication compensation; ROOT owns Native execution."""
import json,sys
from pathlib import Path
import numpy as np
import pytest
import pops
from tests.python.support.collective_checks import collective_call,collective_attempt,collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.evolved_stage_v_capture import save_json,pin
pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]
FAULT='CP9 test fault after genuine checkpoint publication'


def build():
    from tests.python.support import scoped_wave_transport as source
    from pops.layouts import Uniform
    from pops.output import Checkpoint,ConsumerGraph
    from pops.time import every
    case,amr,_=source.build()
    case.consumers(ConsumerGraph.from_consumers((Checkpoint(
        schedule=every(1,clock=amr.regrid.schedule.clock),target='transactional-cp9',bit_identical=True),)))
    return case,Uniform(amr.grid),source.initial(),source.DT


def test_installed_checkpoint_effect_failure_restores_full_uniform_state(tmp_path,record_property,monkeypatch,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    from pops.output._restart_provider import _RestartSnapshot
    from tests.python.support.uniform_checkpoint9_capture import retain_provenance,capture_valid_incrementally,validate_phase
    from tests.python.integration.runtime.test_uniform_state_carrier_checkpoint_runtime import initial_values_for_bindings
    native=select_native_dimension(2);world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    rank=0 if world is None else world.rank;ranks=1 if world is None else world.size
    with collective_check(world):assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case,layout,initial,dt=collective_call(world,build)
    resolved=collective_call(world,lambda:pops.resolve(pops.validate(case),layout=layout,compile_options={'model_source_policy':'require'}))
    artifact=collective_call(world,lambda:pops.compile(resolved)) if world is None else compile_resolved_plan_once(world,resolved,route='uniform-cp9-provisional',compile_artifact=pops.compile)
    directory=collective_directory(world,tmp_path/'cp9-provisional-effect')
    collective_call(world,lambda:retain_provenance(artifact,directory) if rank==0 else None)
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values=initial_values_for_bindings(resolved.initial_condition_plan.bindings,initial),resources={'execution_context':artifact_execution_context(artifact)}))
    def capture(label):
        image=collective_call(world,runtime.observe_accepted_state_storage);clock=(runtime.time,runtime.macro_step)
        cursors=runtime.consumer_cursors.to_data()
        collective_call(world,lambda:save_json(directory/(label+'.rank%d.cursors.json'%rank),cursors))
        values=capture_valid_incrementally(world,runtime,directory,label,rank,ranks,image,clock,tuple(initial))
        collective_call(world,lambda:validate_phase(image,clock,values,rank=rank,ranks=ranks))
        return image.complete,clock,cursors
    before=capture('before');events=[];targets=[]
    original=_RestartSnapshot.publish
    def failed_publication(snapshot,target):
        published=original(snapshot,target)  # genuine CP9 capture and owned final publication
        targets.append(Path(published))
        provisional_cursors=runtime.consumer_cursors.to_data()
        collective_call(world,lambda:save_json(directory/('provisional.rank%d.cursors.json'%rank),provisional_cursors))
        _,refusals=collective_attempt(world,runtime.observe_accepted_state_storage)
        with collective_check(world):
            assert all(row is not None and 'accepted idle state' in row[1] for row in refusals)
        def retained():
            if rank==0:
                payload=Path(published).read_bytes();copy=directory/'actually-published-before-fault.npz';copy.write_bytes(payload)
                with np.load(copy,allow_pickle=False) as archive:
                    assert int(archive['pops_checkpoint_version'])==9
                    assert archive['state_carriers_checkpoint'].dtype==np.uint8
                    assert archive['state_carriers_checkpoint'].tobytes()!=before[0]
                    assert float(archive['t'])==dt and int(archive['macro_step'])==1
                save_json(directory/'published-proof.json',{'schema':'pops.cp9-provisional-effect-proof@1','actual_published_path':str(published),'retained':pin(copy),'idle_observation_refusals':refusals,'accepted':False,'root_approval':False})
        collective_call(world,retained)
        events.append('genuine-publication-before-fault')
        raise RuntimeError(FAULT)  # same explicit test fault on all participants, no model branch
    monkeypatch.setattr(_RestartSnapshot,'publish',failed_publication)
    _,failures=collective_attempt(world,lambda:pops.run(runtime,t_end=dt,max_steps=1,console=False,output_dir=directory/'output'))
    collective_call(world,lambda:save_json(directory/('failure-rank%d.json'%rank),{'failures':failures,'events':events}))
    after=capture('after')
    with collective_check(world):
        assert all(row is not None and FAULT in row[1] for row in failures)
        assert events==['genuine-publication-before-fault'] and len(targets)==1
        assert before==after
        if rank==0:assert not targets[0].exists()  # published artifact compensated, not accepted
    if rank==0:
        receipt=save_json(directory/'receipt.json',{'schema':'pops.cp9-provisional-effect-native-fixture@1','ranks':ranks,'full_state_clock_cursors_exact':True,'actual_publication_then_failure':True,'field_history_qualified':False,'root_approval':False,'files':{str(p.relative_to(directory)):pin(p)['sha256'] for p in directory.rglob('*') if p.is_file()}})
        record_property('cp9_provisional_effect_receipt',receipt['path'])
