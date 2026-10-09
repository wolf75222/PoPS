"""Actual consumer method and test wrapper with explicit Source-only lifecycle spies."""
import ast,os
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('diagnostic_kind',('available','canonical-refused','foreign'))
def test_actual_transition_wrapper_arms_before_real_consumer_dispatch(monkeypatch,diagnostic_kind):
    from pops.runtime._amr_bootstrap_execution import NativeAMRBootstrapConsumer
    import tests.python.support.initial_ghost_failure_selection as selection
    import pops._native_collectives as collectives
    source=ast.parse((ROOT/'tests/python/integration/amr/test_public_initial_field_ghost_failure.py').read_text())
    wrapper=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='observed')
    native_error=RuntimeError('Source-only callback refusal')
    events=[];images=[];owners=[];targets=[];actions=[];world=SimpleNamespace(size=1,rank=0)
    class State:
        def _begin_bootstrap_plan(self):events.append('begin')
    class Engine:
        _s=State()
        def _bootstrap_next_level(self):
            assert os.environ['POPS_TEST_INITIAL_GHOST_TARGET_RANK']=='0'
            events.extend(['actual-create-dispatch','Field-read','tentative-write','local-rollback'])
            raise native_error
    plan=SimpleNamespace(identity=SimpleNamespace(to_data=lambda:{'scope':'SourceOnly'}))
    owner=NativeAMRBootstrapConsumer(Engine(),plan,[])
    owner._tagged_level=0;owner._clustered=True
    def snapshot(owner,phase,captured_blob=None):events.append('snapshot');return {'blob':b'SourceOnly-snapshot'}
    monkeypatch.setattr(selection,'select_xmin_owner',lambda blob,size:{'target':0})
    monkeypatch.setattr(selection,'require_selection_agreement',lambda rows,selected:{'agreed':selected})
    import tests.python.support.initial_ghost_parent_capture as parent_capture
    def diagnostic(world,owner,blob):
        if diagnostic_kind=='foreign':raise ValueError('Source-only unexpected diagnostic error')
        return {'disposition':'available','blob':blob} if diagnostic_kind=='available' else {'disposition':'canonical-refused','scope':'SourceOnly-no-image'}
    monkeypatch.setattr(parent_capture,'capture_parent_rejected',diagnostic)
    monkeypatch.setattr(collectives,'allgather_value',lambda world,row:[row])
    scope={'original':NativeAMRBootstrapConsumer.consume_bootstrap_action,'image':snapshot,'transition_images':images,'diagnostic_errors':[],'owners':owners,'targets':targets,'observed_actions':actions,'world':world,'collective_call':lambda world,fn:fn(),'collective_check':lambda world:__import__('contextlib').nullcontext(),'monkeypatch':monkeypatch}
    exec(compile(ast.Module(body=[wrapper],type_ignores=[]),'<actual fixture transition wrapper>','exec'),scope)
    monkeypatch.delenv('POPS_TEST_INITIAL_GHOST_TARGET_RANK',raising=False)
    action=SimpleNamespace(operation='create_level',level=1,identity=SimpleNamespace(token='SourceOnly-action'))
    with pytest.raises(RuntimeError,match='Source-only callback refusal') as caught:scope['observed'](owner,action)
    assert caught.value is native_error
    assert events==['begin','snapshot','actual-create-dispatch','Field-read','tentative-write','local-rollback']+(['snapshot'] if diagnostic_kind=='available' else [])
    assert bool(scope['diagnostic_errors'])==(diagnostic_kind=='foreign')
    assert ('blob' in images[0][1])==(diagnostic_kind=='available')
    assert len(images)==len(targets)==len(actions)==1 and not owners
    assert 'POPS_TEST_INITIAL_GHOST_TARGET_RANK' not in os.environ
    assert owner._active # Outer abort is later; not certified by this bracket.

def test_source_scope_and_guarded_component_unchanged():
    source=(ROOT/'tests/python/integration/amr/test_public_initial_field_ghost_failure.py').read_text()
    assert "consume_bootstrap_action',observed" in source and 'finalize_bootstrap' not in source
    assert 'not successful construction or nested parent rollback' in source and 'assert before==after' in source
    assert "assert not diagnostic_errors" in source
    assert "assert time==0." in source and "actual initial owner/failure boundary not observed" in source


def test_real_bootstrap_abort_is_observed_after_actual_restore(monkeypatch):
    from pops.runtime._amr_bootstrap_execution import NativeAMRBootstrapConsumer
    tree=ast.parse((ROOT/'tests/python/integration/amr/test_public_initial_field_ghost_failure.py').read_text())
    wrappers={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ('observed_init','observed_abort')}
    events=[];images=[];owners=[];baseline=[]
    state={'value':0}
    class Native:
        def _begin_bootstrap_plan(self):events.append('begin')
    class Engine:
        _s=Native()
        def _rollback_bootstrap_level(self):events.append('genuine-restore');state['value']=0
    def snapshot(owner,phase,captured_blob=None):events.append('capture');return {'blob':bytes([state['value']])}
    scope={'original_init':NativeAMRBootstrapConsumer.__init__,'original_abort':NativeAMRBootstrapConsumer.abort_bootstrap,'owners':owners,'bootstrap_baselines':baseline,'images':images,'image':snapshot}
    exec(compile(ast.Module(body=list(wrappers.values()),type_ignores=[]),'<actual bootstrap observers>','exec'),scope)
    monkeypatch.setattr(NativeAMRBootstrapConsumer,'__init__',scope['observed_init'])
    monkeypatch.setattr(NativeAMRBootstrapConsumer,'abort_bootstrap',scope['observed_abort'])
    plan=SimpleNamespace(identity=SimpleNamespace(to_data=lambda:{'scope':'SourceOnly'}))
    owner=NativeAMRBootstrapConsumer(Engine(),plan,[])
    state['value']=42 # Explicit Source stand-in for intermediate preparation, not saved Native proof.
    owner.abort_bootstrap()
    assert events==['capture','begin','genuine-restore','capture']
    assert images==[(baseline[0],baseline[0])] and len(owners)==1 and not owner._active


def test_abort_failure_never_synthesizes_postrestore_capture(monkeypatch):
    from pops.runtime._amr_bootstrap_execution import NativeAMRBootstrapConsumer
    tree=ast.parse((ROOT/'tests/python/integration/amr/test_public_initial_field_ghost_failure.py').read_text())
    wrapper=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='observed_abort')
    images=[];calls=[]
    class Engine:
        def _rollback_bootstrap_level(self):raise RuntimeError('Source-only genuine abort failed')
    owner=object.__new__(NativeAMRBootstrapConsumer);owner._engine=Engine();owner._active=True
    scope={'original_abort':NativeAMRBootstrapConsumer.abort_bootstrap,'image':lambda owner,phase:calls.append('capture'),'bootstrap_baselines':[{'blob':b'SourceOnly'}],'images':images}
    exec(compile(ast.Module(body=[wrapper],type_ignores=[]),'<actual abort wrapper>','exec'),scope)
    with pytest.raises(RuntimeError,match='genuine abort failed'):scope['observed_abort'](owner)
    assert not images and not calls and owner._active

def test_full_proof_serialization_is_strict():
    source=(ROOT/'tests/python/integration/amr/test_public_initial_field_ghost_failure.py').read_text()
    assert 'json.dumps(proof,indent=2,allow_nan=False)' in source and 'default=str' not in source
    assert 'nested_parent_rollback_not_certified' in source and 'actual_outer_bootstrap_abort_observed' in source
