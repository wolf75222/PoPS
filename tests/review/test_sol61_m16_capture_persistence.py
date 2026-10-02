"""Source synthetic callbacks and genuine wrapper/dump classes; no Native evidence."""
import json
from pathlib import Path
import numpy as np
import pytest
import pops
from pops.codegen.loader import CompiledProblem
from tests.python.support.atomic_native_capture import execute_captured_step,save_phase,select_layout_program
from tests.python.support.m16_explicit_native_capture import dump_retained_program,persisted_capture


def test_original_native_shaped_error_survives_capture_error_and_before_file(tmp_path):
    values=np.arange(6,dtype=np.float64)
    image=(values,b'SOURCE-synthetic-not-native',(0.,0,1))
    persisted_capture(None,lambda:image,lambda row:save_phase(tmp_path,'before',row),lambda failures:pytest.fail(str(failures)))
    original=RuntimeError('SOURCE synthetic step rejected')
    capture_error=ValueError('SOURCE synthetic after capture refused')
    def operation():raise original
    def refused():raise capture_error
    def failure(rows):
        (tmp_path/'after-failed.json').write_text(json.dumps(rows))
    def attempt(rows):
        (tmp_path/'attempt.json').write_text(json.dumps(rows))
    def after():return persisted_capture(None,refused,lambda row:save_phase(tmp_path,'after',row),failure)
    with pytest.raises(RuntimeError) as caught:
        execute_captured_step(None,operation,after,attempt,lambda row,failed:pytest.fail('invented image'))
    assert caught.value is original
    assert np.load(tmp_path/'before.npy',allow_pickle=False).tobytes()==values.tobytes()
    assert not (tmp_path/'after.npy').exists()
    assert json.loads((tmp_path/'after-failed.json').read_text())==[['ValueError',str(capture_error),False]]
    assert json.loads((tmp_path/'attempt.json').read_text())==[['RuntimeError',str(original),True]]


def test_before_capture_error_preserved_without_fake_payload(tmp_path):
    original=RuntimeError('SOURCE synthetic before capture failed')
    def fail():raise original
    with pytest.raises(RuntimeError) as caught:
        persisted_capture(None,fail,lambda row:pytest.fail('invented image'),lambda rows:(tmp_path/'failure.json').write_text(json.dumps(rows)))
    assert caught.value is original and not (tmp_path/'before.npy').exists()


def test_genuine_program_dump_retains_bytes_and_refuses_regeneration(tmp_path,monkeypatch):
    # Actual compiled class with explicitly Source-only non-DSO bytes; never loaded.
    binary=tmp_path/'source-not-dso';binary.write_bytes(b'SOURCE metadata only')
    program=pops.Program('source_only_dump')
    compiled=CompiledProblem(str(binary),program,None,None,None,'c++20',generated_cpp='// retained Source witness',native_dimension=2)
    dump_retained_program(compiled,tmp_path)
    assert (tmp_path/'program.cpp').read_text()=='// retained Source witness'
    assert json.loads((tmp_path/'program.ir.json').read_text())==program._serialize()
    compiled._generated_cpp=None
    monkeypatch.setattr(compiled,'dump_cpp',lambda *args:pytest.fail('fallback entered'))
    with pytest.raises(ValueError,match='regeneration forbidden'):dump_retained_program(compiled,tmp_path)


def test_actual_wrapper_authenticates_unique_population_partition(monkeypatch):
    from tests.python.unit.codegen._typed_artifact_fixture import artifact_fixture,CanonicalValue
    platform=CanonicalValue('SOURCE-no-platform')
    monkeypatch.setattr('pops.codegen._compiled_artifact._common_platform_manifest',lambda **kwargs:platform)
    artifact=artifact_fixture(target='amr_system',block_names=('population',))
    selected=select_layout_program(artifact,artifact.plan)
    assert selected.program is artifact.program
    assert type(artifact.artifact_identity.token) is str
    # Raw identity payloads deliberately contain bytes, tokens serialize losslessly.
    assert type(selected.to_data()['identity']['digest']) is bytes
    assert json.loads(json.dumps({'identity':selected.identity.token}))['identity']==selected.identity.token
    with pytest.raises(ValueError):select_layout_program(artifact_fixture(target='amr_system',block_names=('foreign',)),artifact.plan)
