"""Actual wrapper admission counterexamples, Source-only; no simulated Native proof."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

@pytest.mark.parametrize('failure_at',['selection','agreement'])
def test_failed_target_admission_cannot_enter_original_transition(monkeypatch,failure_at):
    import tests.python.support.initial_ghost_failure_selection as selection
    import pops._native_collectives as collectives
    root=Path(__file__).resolve().parents[2]
    tree=ast.parse((root/'tests/python/integration/amr/test_public_initial_field_ghost_failure.py').read_text())
    wrapper=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='observed')
    calls=[]
    def refuse(*args):raise ValueError('independent forged selection')
    monkeypatch.setattr(selection,'select_xmin_owner',refuse if failure_at=='selection' else lambda *args:{'target':0})
    monkeypatch.setattr(selection,'require_selection_agreement',refuse if failure_at=='agreement' else lambda *args:{})
    monkeypatch.setattr(collectives,'allgather_value',lambda world,row:[row])
    scope=dict(original=lambda *args:calls.append('original'),image=lambda owner:{'blob':b'SourceOnly'},
        images=[],owners=[],targets=[],observed_actions=[],world=SimpleNamespace(size=1),
        collective_call=lambda world,fn:fn(),collective_check=lambda world:__import__('contextlib').nullcontext(),monkeypatch=monkeypatch)
    exec(compile(ast.Module(body=[wrapper],type_ignores=[]),'<actual wrapper>','exec'),scope)
    owner=SimpleNamespace(_engine=object())
    action=SimpleNamespace(operation='create_level',level=1,identity=SimpleNamespace(token='SourceOnly'))
    monkeypatch.setenv('POPS_TEST_INITIAL_GHOST_TARGET_RANK','prior-test-value')
    with pytest.raises(ValueError,match='forged selection'):scope['observed'](owner,action)
    import os
    assert os.environ['POPS_TEST_INITIAL_GHOST_TARGET_RANK']=='prior-test-value'
    assert calls==[] and scope['images']==[]
    # Other actions remain exact original dispatch with no observer snapshot/admission.
    scope['observed'](owner,SimpleNamespace(operation='recompute'))
    assert calls==['original']
