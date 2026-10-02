"""Actual tiny host process and poisoned provenance, never PoPS Native."""
import subprocess
import shutil
from types import SimpleNamespace
import pytest
from tests.python.support.actual_compile_capture import capture_actual_compiles


def test_retained_tu_poison_and_ambiguous_outputs_refuse_and_restore_driver(tmp_path):
    cpp=tmp_path/'input.cpp';cpp.write_text('extern "C" int v(){return 3;}\n')
    out=tmp_path/'stage.so'
    command=[shutil.which('clang++') or 'c++','-shared','-fPIC','-DPOPS_HEADER_SIG="host-only"',str(cpp),'-o',str(out)]
    def compile(cmd,purpose):return subprocess.run(cmd,check=True,capture_output=True)
    driver=SimpleNamespace(_run_compile=compile)
    with capture_actual_compiles(tmp_path/'capture',driver=driver) as capture:
        driver._run_compile(command,'real tiny host')
        retained=tmp_path/'capture/tu-0.cpp'
        original=retained.read_bytes();retained.write_bytes(b'poison')
        with pytest.raises(ValueError,match='changed'):capture.require_binary(out)
        retained.write_bytes(original)
        capture.records.append(dict(capture.records[0]))
        with pytest.raises(ValueError,match='unique'):capture.require_binary(out)
    assert driver._run_compile is compile


def test_original_compile_failure_identity_retained_input_and_driver_restoration(tmp_path):
    cpp=tmp_path/'input.cpp';cpp.write_text('actual bad translation unit')
    original=RuntimeError('original compiler failure')
    def fail(cmd,purpose):raise original
    driver=SimpleNamespace(_run_compile=fail)
    with pytest.raises(RuntimeError) as caught:
        with capture_actual_compiles(tmp_path/'capture',driver=driver) as capture:
            driver._run_compile([shutil.which('clang++') or 'c++','-DPOPS_HEADER_SIG="host-only"',str(cpp),'-o',str(tmp_path/'out')],'failure')
    assert caught.value is original and driver._run_compile is fail
    assert (tmp_path/'capture/tu-0.cpp').read_bytes()==cpp.read_bytes()
    assert capture.records==[]
