"""Source preparation only; installed Native CP/replay remains ROOT work."""
from pathlib import Path
import ast
import json
import subprocess
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]


def test_checkpoint8_valid_only_counterexample_and_9_carrier_authority():
    old=subprocess.check_output(['git','-C',str(ROOT),'show','3302c4a19d94a0b08dbba9c05c4de3ae2debb232:python/pops/runtime/_system_io.py'],text=True)
    capture=old[old.index('    def _capture_checkpoint'):old.index('    def checkpoint(',old.index('    def _capture_checkpoint'))]
    assert 'state_global(block)' in capture and 'state_carriers_checkpoint' not in capture
    # Independent concrete storage images have identical valid projection, different ghost bits.
    one=np.zeros((2,4,5),dtype=np.float64);two=one.copy();two[:,0,:]=-0.
    assert one[:,1:-1,1:-1].tobytes()==two[:,1:-1,1:-1].tobytes()
    assert one.tobytes()!=two.tobytes()
    from pops._generated_release_contract import UNIFORM_CHECKPOINT_PAYLOAD_VERSION,NATIVE_ABI_VERSION
    assert UNIFORM_CHECKPOINT_PAYLOAD_VERSION==9 and NATIVE_ABI_VERSION==8
    authority=json.loads((ROOT/'tests/review/sol61_uniform_checkpoint9_release_authority.json').read_text())
    assert authority['baseline_payload_version']==8 and authority['native_abi_change_claimed'] is False


def test_actual_source_restart_carriers_preflight_before_write_and_final_before_cursor_publish():
    module=ast.parse((ROOT/'python/pops/runtime/_system_io.py').read_text())
    functions={f.name:f for cls in module.body if isinstance(cls,ast.ClassDef) and cls.name=='_SystemIO' for f in cls.body if isinstance(f,ast.FunctionDef)}
    calls=lambda f:[n for n in ast.walk(f) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    preflight=calls(functions['_prepare_checkpoint_restart'])
    assert any(n.func.attr=='validate_checkpoint_state_carriers' for n in preflight)
    assert not any(n.func.attr in ('set_state','restore_checkpoint_state_carriers') for n in preflight)
    apply=functions['_apply_checkpoint_restart'];text=ast.unparse(apply)
    assert text.index('restore_checkpoint_state_carriers')>text.index('_restore_checkpoint_program_diagnostics')
    assert text.index('restore_checkpoint_state_carriers')<text.index('self._temporal_restart_state =')


@pytest.mark.parametrize('profile',('fanli15','two-transports'))
def test_public_native_fixture_authors_distinct_uniform_states(profile):
    from tests.python.integration.runtime.test_uniform_state_carrier_checkpoint_runtime import build
    from pops.layouts import Uniform
    case,layout,initial,dt=build(profile)
    assert isinstance(layout,Uniform) and dt>0
    widths=tuple(v.shape[0] for v in initial.values())
    assert widths==((15,) if profile=='fanli15' else (2,3))
    assert all(np.isfinite(values).all() for values in initial.values())
    import pops
    pops.validate(case)


def test_legacy8_compatibility_is_explicit_and_never_promoted_to_full9():
    from pops.runtime._checkpoint_manifest import require_exact_payload_version
    from pops.codegen._checkpoint_migration_uniform_v2 import UNIFORM_V2_TARGET_VERSION
    assert UNIFORM_V2_TARGET_VERSION == 8
    old = {'pops_checkpoint_version': np.asarray(8, dtype=np.int64)}
    assert require_exact_payload_version(old, key='pops_checkpoint_version', expected=8, runtime_kind='Uniform') == 8
    with pytest.raises(ValueError):
        require_exact_payload_version(old, key='pops_checkpoint_version', expected=9, runtime_kind='Uniform')
    for value in (True, 8., '8', np.asarray([8])):
        with pytest.raises(TypeError):
            require_exact_payload_version({'pops_checkpoint_version':value}, key='pops_checkpoint_version', expected=8, runtime_kind='Uniform')
    from pops.runtime._runtime_instance import RuntimeInstance
    # Invalid modes must fail before touching any runtime object or archive.
    for mode in (True, None, 'legacy', 'full9'):
        with pytest.raises(ValueError, match='state_storage'):
            RuntimeInstance.restart(object(), 'never-opened.npz', state_storage=mode)
    text=(ROOT/'python/pops/runtime/_system_io.py').read_text()
    assert 'expected=UNIFORM_CHECKPOINT_PAYLOAD_VERSION if state_storage == "full" else 8' in text
    assert 'elif "state_carriers_checkpoint" in d:' in text
    assert 'if prepared.state_carriers is not None:' in text


def test_v2_migration_drops_metadata_authority_ghosts_explicitly():
    # Actual output-building code, not a substitute migration implementation.
    module=ast.parse((ROOT/'python/pops/codegen/_checkpoint_migration_uniform_v2.py').read_text())
    functions={node.name:node for node in module.body if isinstance(node,ast.FunctionDef)}
    builders=[node for node in functions.values() if 'output[\'pops_checkpoint_version\']' in ast.unparse(node)]
    assert len(builders)==1
    text=ast.unparse(builders[0])
    assert "'state_carriers_checkpoint'" in text
    assert 'UNIFORM_V2_TARGET_VERSION' in text
    authority=ast.unparse(functions['_current_authority'])
    assert 'authority_version not in (8, 9)' in authority


def test_native_witness_initial_binding_uses_block_not_last_model_definition():
    from types import SimpleNamespace
    from pops.model.ownership import OwnerKind,OwnerPath,OwnerSegment
    from tests.python.integration.runtime.test_uniform_state_carrier_checkpoint_runtime import initial_values_for_bindings
    class Subject:
        kind='state'
        def __init__(self, block):
            self.owner_path=OwnerPath(OwnerSegment(OwnerKind.CASE,'case'), OwnerSegment(OwnerKind.BLOCK,block), OwnerSegment(OwnerKind.MODEL_DEFINITION,'same-model-definition'))
    first,second=Subject('first'),Subject('second')
    bindings=[SimpleNamespace(subject=first),SimpleNamespace(subject=second)]
    initial={'first':np.ones((2,3,3)), 'second':np.ones((3,3,3))*2}
    values=initial_values_for_bindings(bindings,initial)
    assert values[first] is initial['first'] and values[second] is initial['second']
    for rows in (bindings[:1], bindings+[bindings[0]], [SimpleNamespace(subject=Subject('foreign'))]):
        with pytest.raises(ValueError):initial_values_for_bindings(rows,initial)
