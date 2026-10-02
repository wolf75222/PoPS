"""Actual Source wrapper contract and synthetic file persistence; no Native."""
import ast
import json
from pathlib import Path
import numpy as np
import pytest
from pops.codegen._compiled_artifact import CompiledSimulationArtifact
from pops.codegen.loader import CompiledProblem
from tests.python.unit.codegen._typed_artifact_fixture import artifact_fixture,CanonicalValue
from tests.python.support.atomic_native_capture import select_layout_program,save_phase,execute_captured_step,layout_program_json_identity


def test_actual_wrapper_selects_exact_amr_partition_without_facade_dump(monkeypatch):
    # Genuine wrapper class with the repository's explicitly metadata-only components.
    # No shared library is loaded, no runtime or scientific receipt is fabricated.
    # Isolate only platform metadata lookup; there is deliberately no selected SDK.
    platform=CanonicalValue('source-metadata-only-no-native')
    monkeypatch.setattr('pops.codegen._compiled_artifact._common_platform_manifest',
                        lambda **kwargs:platform)
    artifact=artifact_fixture(target='amr_system',block_names=('population',))
    row=select_layout_program(artifact,artifact.plan)
    assert row.program is artifact.program
    # Exact actual identity bytes require an explicit token representation in JSON.
    with pytest.raises(TypeError):json.dumps(row.to_data())
    identity=layout_program_json_identity(row)
    assert json.loads(json.dumps(identity))['identity_token']==row.identity.token
    assert not hasattr(CompiledSimulationArtifact,'dump_ir')
    assert not hasattr(CompiledSimulationArtifact,'_generated_cpp')
    assert hasattr(CompiledProblem,'dump_ir')
    assert row.layout_id==next(iter({a.layout.qualified_id for a in artifact.plan.layout_plan.assignments
                                   if a.subject_kind=='block' and a.subject.local_id=='population'}))
    with pytest.raises(ValueError):
        foreign=artifact_fixture(target='amr_system',block_names=('other',))
        select_layout_program(foreign,foreign.plan)
    with pytest.raises(ValueError):
        uniform=artifact_fixture(target='system',block_names=('population',))
        select_layout_program(uniform,uniform.plan)
    with pytest.raises(TypeError):select_layout_program(object(),artifact.plan)


def test_capture_saved_immediately_survives_later_source_error(tmp_path):
    values=np.arange(12,dtype=np.float64).reshape(2,2,3)
    save_phase(tmp_path,'initial',(values,b'synthetic-not-native',(0.,0,1)))
    save_phase(tmp_path,'accepted',(values+1,b'synthetic-not-native',(1.,1,1)))
    original=ArithmeticError('synthetic second-step failure')
    events=[]
    def fail():raise original
    def attempt(failures):
        events.append(('failure',failures))
        (tmp_path/'failure.json').write_text(json.dumps(failures))
    def captured(image,failed):
        assert (tmp_path/'failure.json').exists()
        events.append(('captured',failed))
        save_phase(tmp_path,'failed-step-2',image)
    with pytest.raises(ArithmeticError) as error:
        execute_captured_step(None,fail,lambda:(values+1,b'synthetic-not-native',(1.,1,1)),attempt,captured)
    assert error.value is original
    assert events[0][0]=='failure' and events[1]==('captured',True)
    assert np.load(tmp_path/'accepted.npy',allow_pickle=False).tobytes()==(values+1).tobytes()
    assert json.loads((tmp_path/'accepted.capture.json').read_text())['science_assertions']=='not-yet-run'
    assert not (tmp_path/'continuous.npy').exists()


def test_real_fixture_preserves_original_exception_and_incremental_capture_order():
    path=Path(__file__).resolve().parents[1]/'python/integration/runtime/test_atomic_cubature_public_path_runtime.py'
    module=ast.parse(path.read_text())
    function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='test_installed_atomic_cubature_raw_path')
    source=ast.get_source_segment(path.read_text(),function)
    assert source.index("persist_capture('initial'")<source.index('for step,phase in')
    assert 'execute_captured_step(world,' in source
    import inspect
    helper=inspect.getsource(execute_captured_step)
    assert 'raise original' in helper
    assert helper.index('lambda:on_attempt(failures)')<helper.index('collective_attempt(world,capture)')
    assert 'artifact._generated_cpp' not in source and 'artifact.dump_ir' not in source
    assert 'artifact.problem_hash' not in source
