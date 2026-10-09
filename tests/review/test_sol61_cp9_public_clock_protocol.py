"""Genuine public RuntimeInstance methods; Source executor, no Native run."""
import ast,json,subprocess
from pathlib import Path
import numpy as np
import pytest
from pops.runtime._runtime_instance import RuntimeInstance
from tests.python.support.collective_checks import collective_call
from tests.python.support.uniform_checkpoint9_capture import capture_valid_incrementally,validate_phase
from tests.review.test_sol61_checkpoint9_capture_independent import observation
ROOT=Path(__file__).resolve().parents[2]
BASE='7bbeb375df24d22d08f91e9f009ffe05812ec9ef'


def runtime_source():
    image,values=observation()
    class Executor:
        def time(self):return 0.0
        def macro_step(self):return 0
        def observe_accepted_state_storage(self):return image
        def state_global(self,name):return values[name]
    from pops.output._consumer_contracts import ConsumerCursorSet
    runtime=object.__new__(RuntimeInstance);runtime._executor=Executor();runtime._consumer_cursors=ConsumerCursorSet()
    return runtime


def old_capture(tmp_path,runtime):
    source=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':tests/python/integration/runtime/test_uniform_state_carrier_checkpoint_runtime.py'],text=True)
    node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='capture')
    env=dict(runtime=runtime,world=None,directory=tmp_path,initial={'q':None},phases={},collective_call=collective_call,
             capture_valid_incrementally=capture_valid_incrementally,validate_phase=validate_phase)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'historical_actual_capture','exec'),env)
    return env['capture']


def test_historical_capture_refuses_method_objects_on_genuine_public_class(tmp_path):
    runtime=runtime_source()
    with pytest.raises(TypeError,match='method.*JSON serializable'):
        json.dumps((runtime.time,runtime.macro_step))
    with pytest.raises(AssertionError,match='method.*JSON serializable'):
        old_capture(tmp_path,runtime)('initial')
    assert (tmp_path/'initial.rank0.carriers').exists()


def test_public_methods_produce_exact_durable_clock(tmp_path):
    runtime=runtime_source()
    clock=(runtime.time(),runtime.macro_step())
    assert clock==(0.0,0) and type(clock[0]) is float and type(clock[1]) is int
    (tmp_path/'clock.json').write_text(json.dumps(clock,allow_nan=False))
    assert json.loads((tmp_path/'clock.json').read_text())==[0.0,0]


@pytest.mark.parametrize('filename',('test_uniform_state_carrier_checkpoint_runtime.py','test_uniform_checkpoint_provisional_effect.py'))
def test_both_actual_capture_bodies_use_genuine_public_method_protocol(tmp_path,filename):
    from tests.python.support.evolved_stage_v_capture import save_json
    source=(ROOT/'tests/python/integration/runtime'/filename).read_text()
    node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='capture')
    env=dict(runtime=runtime_source(),world=None,directory=tmp_path,rank=0,ranks=1,initial={'q':None},phases={},
             collective_call=collective_call,capture_valid_incrementally=capture_valid_incrementally,
             validate_phase=validate_phase,save_json=save_json)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'current_actual_capture','exec'),env)
    result=env['capture']('initial')
    assert result[1]==(0.0,0)
    assert json.loads((tmp_path/'initial.clock.json').read_text())==[0.0,0]
    assert (tmp_path/'initial.rank0.block0.npy').exists()
