"""Execute the fixture's exact injection body with explicit Source stand-ins."""
import ast
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pytest
from tests.python.support.collective_checks import collective_attempt,collective_check,collective_call
from tests.python.support.evolved_stage_v_capture import save_json,pin
ROOT=Path(__file__).resolve().parents[2]


def injection(env):
    tree=ast.parse((ROOT/'tests/python/integration/runtime/test_uniform_checkpoint_provisional_effect.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='failed_publication')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'exact_fixture_injection','exec'),env)
    return env['failed_publication']


def scope(tmp_path,observer_reason='bound accepted idle state'):
    calls=[];target=tmp_path/'owned.npz'
    def original(snapshot,path):
        calls.append('genuine-first')
        with target.open('wb') as f:np.savez(f,pops_checkpoint_version=np.asarray(9),state_carriers_checkpoint=np.frombuffer(b'POPSCAR1-source-test-only',dtype=np.uint8),t=np.asarray(.125),macro_step=np.asarray(1))
        return target
    class Runtime:
        def observe_accepted_state_storage(self):
            calls.append('idle-refused');raise RuntimeError(observer_reason)
    env=dict(original=original,targets=[],events=[],runtime=Runtime(),world=None,rank=0,np=np,Path=Path,
             directory=tmp_path,before=(b'baseline-source-only',None,None),dt=.125,save_json=save_json,pin=pin,
             collective_attempt=collective_attempt,collective_check=collective_check,collective_call=collective_call,
             FAULT='CP9 test fault after genuine checkpoint publication')
    return env,calls,target


def test_injection_after_publication_retains_proof_before_fault(tmp_path):
    env,calls,target=scope(tmp_path);hook=injection(env)
    with pytest.raises(RuntimeError,match=env['FAULT']):hook(object(),target)
    assert calls==['genuine-first','idle-refused']
    assert target.read_bytes()==(tmp_path/'actually-published-before-fault.npz').read_bytes()
    assert (tmp_path/'published-proof.json').exists()
    assert env['events']==['genuine-publication-before-fault']
    assert env['targets']==[target]


def test_foreign_observer_failure_does_not_mint_intended_fault(tmp_path):
    env,calls,target=scope(tmp_path,'foreign failure')
    with pytest.raises(AssertionError):injection(env)(object(),target)
    assert calls==['genuine-first','idle-refused'] and env['events']==[]
    assert not (tmp_path/'published-proof.json').exists()


def test_failed_real_publication_is_not_claimed_injected(tmp_path):
    env,calls,target=scope(tmp_path)
    error=RuntimeError('genuine publisher failed')
    def failed(snapshot,path):calls.append('publisher-refused');raise error
    env['original']=failed
    with pytest.raises(RuntimeError) as caught:injection(env)(object(),target)
    assert caught.value is error
    assert env['targets']==[] and env['events']==[] and calls==['publisher-refused']
