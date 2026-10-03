"""Independent evidence-order checks against the actual fixture capture body.

The runtime seam is explicitly Source-only. No native execution is asserted.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tests.review.test_sol61_checkpoint9_capture_independent import observation
from tests.python.support.uniform_checkpoint9_capture import persist_phase, validate_phase


def capture_body(directory,runtime,world=None):
    source=Path(__file__).resolve().parents[2]/'tests/python/integration/runtime/test_uniform_state_carrier_checkpoint_runtime.py'
    tree=ast.parse(source.read_text())
    outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_installed_uniform_full_state_checkpoint_restart')
    node=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='capture')
    space={'runtime':runtime,'world':world,'directory':directory,'phases':{},
           'initial':{'q':None},'collective_call':lambda world,call:call(),
           'persist_phase':persist_phase,'validate_phase':validate_phase}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),space)
    return space['capture'],space['phases']


def test_actual_capture_saves_full_observation_before_real_getter_exception(tmp_path):
    image,_=observation();calls=[]
    def bad(name):
        calls.append(name)
        assert (tmp_path/'initial.rank0.carriers').read_bytes()==image.rank_local
        assert (tmp_path/'initial.rank0.complete.carriers').read_bytes()==image.complete
        raise RuntimeError('Source getter failure')
    runtime=SimpleNamespace(observe_accepted_state_storage=lambda:image,time=0.0,macro_step=0,state_global=bad)
    capture,phases=capture_body(tmp_path,runtime)
    with pytest.raises(RuntimeError,match='Source getter failure'):capture('initial')
    assert calls==['q'] and not phases
    assert json.loads((tmp_path/'initial.rank0.phase.json').read_text())['capture_complete'] is False


def test_actual_capture_saves_corrupt_shard_before_guard_and_never_marks_phase(tmp_path):
    from pops.runtime._state_storage_observation import AcceptedStateStorageObservation
    image,values=observation()
    bad=AcceptedStateStorageObservation(2,0.0,0,image.rank_local[:-8]+b'badbytes',image.complete)
    runtime=SimpleNamespace(observe_accepted_state_storage=lambda:bad,time=0.0,macro_step=0,state_global=lambda name:values[name])
    capture,phases=capture_body(tmp_path,runtime)
    with pytest.raises(ValueError,match='grown bits'):capture('corrupt')
    assert not phases
    assert (tmp_path/'corrupt.rank0.carriers').read_bytes()==bad.rank_local
    assert (tmp_path/'corrupt.rank0.block0.npy').is_file()


def test_detached_runtime_guard_does_not_claim_world_authority():
    from dataclasses import replace
    import struct
    image,values=observation()
    def wire(raw,shard):
        data=bytearray(raw)
        data[24:32]=struct.pack('<Q',2)
        data[32:40]=struct.pack('<q',shard)
        data[105:113]=struct.pack('<q',-1)
        return bytes(data)
    foreign=replace(image,rank_local=wire(image.rank_local,1),complete=wire(image.complete,-1))
    # Historical scope: this guard checks join/projection, not world0 authority.
    # The archive receiver separately requires external world/rank equality.
    validate_phase(foreign,(0.0,0),values)
