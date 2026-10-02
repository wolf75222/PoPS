"""Actual consumer method and test wrapper with explicit Source-only lifecycle spies."""
import ast,os
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[2]

def test_actual_transition_wrapper_arms_before_real_consumer_dispatch(monkeypatch):
    from pops.runtime._amr_bootstrap_execution import NativeAMRBootstrapConsumer
    import tests.python.support.initial_ghost_failure_selection as selection
    import pops._native_collectives as collectives
    source=ast.parse((ROOT/'tests/python/integration/amr/test_public_initial_field_ghost_failure.py').read_text())
    wrapper=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='observed')
    events=[];images=[];owners=[];targets=[];actions=[];world=SimpleNamespace(size=1,rank=0)
    class State:
        def _begin_bootstrap_plan(self):events.append('begin')
    class Engine:
        _s=State()
        def _bootstrap_next_level(self):
            assert os.environ['POPS_TEST_INITIAL_GHOST_TARGET_RANK']=='0'
            events.extend(['actual-create-dispatch','Field-read','tentative-write','local-rollback'])
            raise RuntimeError('Source-only callback refusal')
    plan=SimpleNamespace(identity=SimpleNamespace(to_data=lambda:{'scope':'SourceOnly'}))
    owner=NativeAMRBootstrapConsumer(Engine(),plan,[])
    owner._tagged_level=0;owner._clustered=True
    def snapshot(owner):events.append('snapshot');return {'blob':b'SourceOnly-snapshot'}
    monkeypatch.setattr(selection,'select_xmin_owner',lambda blob,size:{'target':0})
    monkeypatch.setattr(selection,'require_selection_agreement',lambda rows,selected:{'agreed':selected})
    monkeypatch.setattr(collectives,'allgather_value',lambda world,row:[row])
    scope={'original':NativeAMRBootstrapConsumer.consume_bootstrap_action,'image':snapshot,'images':images,'owners':owners,'targets':targets,'observed_actions':actions,'world':world,'collective_call':lambda world,fn:fn(),'collective_check':lambda world:__import__('contextlib').nullcontext(),'monkeypatch':monkeypatch}
    exec(compile(ast.Module(body=[wrapper],type_ignores=[]),'<actual fixture transition wrapper>','exec'),scope)
    monkeypatch.delenv('POPS_TEST_INITIAL_GHOST_TARGET_RANK',raising=False)
    action=SimpleNamespace(operation='create_level',level=1,identity=SimpleNamespace(token='SourceOnly-action'))
    with pytest.raises(RuntimeError,match='Source-only callback refusal'):scope['observed'](owner,action)
    assert events==['begin','snapshot','actual-create-dispatch','Field-read','tentative-write','local-rollback','snapshot']
    assert len(images)==len(owners)==len(targets)==len(actions)==1
    assert 'POPS_TEST_INITIAL_GHOST_TARGET_RANK' not in os.environ
    assert owner._active # Outer abort is later; not certified by this bracket.

def test_source_scope_and_guarded_component_unchanged():
    source=(ROOT/'tests/python/integration/amr/test_public_initial_field_ghost_failure.py').read_text()
    assert "consume_bootstrap_action',observed" in source and 'finalize_bootstrap' not in source
    assert 'not full bootstrap construction' in source and 'assert before==after' in source
    assert "assert time==0." in source and "actual initial owner/failure boundary not observed" in source
