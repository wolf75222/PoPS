"""Future installed Uniform CP9 state-storage witness; ROOT owns actual execution."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pytest
import pops
from tests.python.support.collective_checks import collective_call,collective_check,collective_attempt
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.m16_explicit_native_capture import dump_retained_program
from tests.python.support.atomic_native_capture import execute_captured_step
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
pytestmark=[pytest.mark.compiler,pytest.mark.native_loader]


def build(profile):
    if profile=='fanli15':
        from tests.python.support import m17_fan_li_public_native_case as source
        case,layout=source.make_case(source.original().INDICES)
        return case,layout,{'gas':source.initial(source.original().INDICES)},source.DT
    from tests.python.support import scoped_wave_transport as source
    from pops.layouts import Uniform
    case,amr,_=source.build()
    return case,Uniform(amr.grid),source.initial(),source.DT


def initial_values_for_bindings(bindings, initial):
    from pops.model.ownership import OwnerKind
    values={};blocks=[]
    for binding in bindings:
        subject=binding.subject
        owners=tuple(node.name for node in subject.owner_path.nodes if node.kind is OwnerKind.BLOCK)
        if subject.kind != 'state' or len(owners) != 1 or owners[0] not in initial:
            raise ValueError('initial subject must have its exact declared block owner')
        if owners[0] in blocks:
            raise ValueError('witness requires one initial State per declared block')
        blocks.append(owners[0]);values[subject]=initial[owners[0]]
    if set(blocks) != set(initial):raise ValueError('initial block coverage differs')
    return values


@pytest.mark.parametrize('profile',('fanli15','two-transports'))
def test_installed_uniform_full_state_checkpoint_restart(profile,tmp_path,record_property,isolated_native_cache,native_cxx,kokkos_root):
    del isolated_native_cache,native_cxx,kokkos_root
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator
    native=select_native_dimension(2);world=native.mpi_world() if native_mpi_communicator(native)=='MPI_COMM_WORLD' else None
    root=lambda:world is None or world.rank==0
    with collective_check(world):assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    case,layout,initial,dt=collective_call(world,lambda:build(profile))
    validated=collective_call(world,lambda:pops.validate(case))
    resolved=collective_call(world,lambda:pops.resolve(validated,layout=layout,compile_options={'model_source_policy':'require'}))
    artifact=(collective_call(world,lambda:pops.compile(resolved)) if world is None else compile_resolved_plan_once(world,resolved,route='uniform-state-cp9-'+profile,compile_artifact=pops.compile))
    directory=collective_directory(world,tmp_path/('uniform-state-cp9-'+profile))
    from tests.python.support.uniform_checkpoint9_capture import retain_provenance,capture_valid_incrementally,validate_phase
    def provenance():
        if root():
            retain_provenance(artifact,directory)
            (directory/'preparation.json').write_text(json.dumps({'schema':'pops.uniform-state-carrier-checkpoint-preparation@2','profile':profile,'before_bind':True,'root_approval':False},allow_nan=False))
    collective_call(world,provenance)
    bindings=resolved.initial_condition_plan.bindings
    values=initial_values_for_bindings(bindings,initial)
    # Each subject is resolved by its block owner, never by component order/first species.
    runtime=collective_call(world,lambda:pops.bind(artifact,initial_values=values,resources={'execution_context':artifact_execution_context(artifact)}))
    phases={}
    def capture(label):
        observation=collective_call(world,runtime.observe_accepted_state_storage)
        clock=(runtime.time,runtime.macro_step)
        rank=0 if world is None else world.rank
        ranks=1 if world is None else world.size
        valid=capture_valid_incrementally(world,runtime,directory,label,rank,ranks,observation,clock,tuple(initial))
        # Every raw shard/complete/valid/envelope is durable before a rejecting guard.
        collective_call(world,lambda:validate_phase(observation,clock,valid,rank=rank,ranks=ranks))
        phases[label]=(observation.complete,clock)
        return phases[label]
    def run_to(label,end):
        def attempted(failures):
            if root():(directory/(label+'.attempt.json')).write_text(json.dumps(failures,allow_nan=False))
        _,image=execute_captured_step(world,lambda:pops.run(runtime,t_end=end,max_steps=1,console=False),lambda:capture(label),attempted,lambda image,failed:None)
        return image
    capture('initial')
    checkpoint_step = 4 if profile == 'fanli15' else 1
    final_step = 8 if profile == 'fanli15' else 2
    for step in range(1,checkpoint_step+1):
        accepted=run_to('accepted'+str(step),step*dt)
    checkpoint=directory/'accepted.npz';collective_call(world,lambda:runtime.checkpoint(checkpoint))
    with collective_check(world):
        if root():
            with np.load(checkpoint,allow_pickle=False) as payload:
                assert int(payload['pops_checkpoint_version'])==9
                assert payload['state_carriers_checkpoint'].tobytes()==accepted[0]
    refusals=[]
    for attack in ('truncated-carrier','projection-contradiction','legacy8'):
        bad=directory/(attack+'.npz')
        def mutate():
            if not root():return
            from pops.runtime._checkpoint_manifest import MANIFEST_KEY,IDENTITY_KEY,seal_checkpoint_payload
            with np.load(checkpoint,allow_pickle=False) as archive:
                payload={name:archive[name].copy() for name in archive.files if name not in (MANIFEST_KEY,IDENTITY_KEY)}
            if attack=='truncated-carrier':payload['state_carriers_checkpoint']=payload['state_carriers_checkpoint'][:-1].copy()
            elif attack=='projection-contradiction':
                name='state_'+next(iter(initial));payload[name].flat[0]+=1.
            else:
                payload['pops_checkpoint_version']=np.asarray(8)
                del payload['state_carriers_checkpoint']
            seal_checkpoint_payload(runtime,payload,runtime_kind='uniform')
            with bad.open('wb') as stream:np.savez_compressed(stream,**payload)
        collective_call(world,mutate)
        before=capture('before-'+attack)
        _,failures=collective_attempt(world,lambda:runtime.restart(bad))
        after=capture('after-'+attack)
        if root():(directory/(attack+'.failure.json')).write_text(json.dumps(failures,sort_keys=True,allow_nan=False))
        with collective_check(world):
            assert all(failure is not None for failure in failures)
            assert before==after
            if attack=='projection-contradiction':assert all('contradict' in failure[1] for failure in failures)
        refusals.append(attack)
        if attack=='legacy8':
            collective_call(world,lambda:runtime.restart(bad,state_storage='valid_only_legacy8'))
            # Explicit historical compatibility authenticates valid arrays/clock only.
            # It cannot qualify ghosts; restore the original full CP9 before comparisons.
            capture('legacy-valid-only-restored')
            collective_call(world,lambda:runtime.restart(checkpoint))
            restored=capture('full-restored-after-legacy')
            with collective_check(world):assert restored==accepted

    continuous={}
    for step in range(checkpoint_step+1,final_step+1):
        continuous[step]=run_to('continuous'+str(step),step*dt)
    collective_call(world,lambda:runtime.restart(checkpoint));reloaded=capture('reloaded')
    with collective_check(world):assert reloaded==accepted
    for step in range(checkpoint_step+1,final_step+1):
        replay=run_to('replay'+str(step),step*dt)
        with collective_check(world):assert replay==continuous[step]
    def receipt():
        if root():
            files={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob('*') if p.is_file()}
            (directory/'receipt.json').write_text(json.dumps({'schema':'pops.uniform-state-carrier-checkpoint-native-fixture@2','profile':profile,'checkpoint_step':checkpoint_step,'final_step':final_step,'ranks':1 if world is None else world.size,'phases':list(phases),'refusals':refusals,'files':files,'state_valid_grown_and_clock_exact':True,'field_history_fullgrown_qualified':False,'ghost_formula_qualified':False,'root_approval':False},sort_keys=True,allow_nan=False))
    collective_call(world,receipt);record_property('uniform_state_checkpoint_receipt',str(directory/'receipt.json'))
