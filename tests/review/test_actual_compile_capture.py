"""Real tiny C++ compiler process; no PoPS Native execution."""
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import pytest
from tests.python.support.actual_compile_capture import capture_actual_compiles

def test_exact_compiler_bytes_and_published_binary(tmp_path):
    cpp=tmp_path/'input.cpp';cpp.write_text('extern "C" int answer(){return 42;}\n')
    output=tmp_path/'staged.so'
    compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
    command=[compiler,'-shared','-fPIC','-DPOPS_HEADER_SIG="source-host-only"',str(cpp),'-o',str(output)]
    def compile(command,purpose):return subprocess.run(command,check=True)
    driver=SimpleNamespace(_run_compile=compile)
    with capture_actual_compiles(tmp_path/'capture',driver=driver) as capture:
        driver._run_compile(command,'tiny host compiler witness')
        published=tmp_path/'published.so';output.rename(published)
        row=capture.require_binary(published)
        assert (tmp_path/'capture'/row['retained_file']).read_bytes()==cpp.read_bytes()
        assert row['status']=='compiler-succeeded'
        foreign=tmp_path/'foreign';foreign.write_bytes(b'foreign')
        with pytest.raises(ValueError,match='no unique'):capture.require_binary(foreign)
    assert driver._run_compile is compile

def test_cache_hit_and_failed_compiler_never_mint_success(tmp_path):
    binary=tmp_path/'cached';binary.write_bytes(b'cached')
    def compile(command,purpose):raise RuntimeError('original compiler refusal')
    driver=SimpleNamespace(_run_compile=compile)
    with capture_actual_compiles(tmp_path/'capture',driver=driver) as capture:
        with pytest.raises(ValueError,match='cache hit'):capture.require_binary(binary)
        cpp=tmp_path/'input.cpp';cpp.write_text('bad source')
        command=[shutil.which('clang++') or 'c++','-DPOPS_HEADER_SIG="host"',str(cpp),'-o',str(binary)]
        with pytest.raises(RuntimeError,match='original compiler refusal'):driver._run_compile(command,'negative')
        assert not capture.records and not list((tmp_path/'capture').glob('*.json'))
        assert (tmp_path/'capture/tu-0.cpp').read_text()=='bad source'
    assert driver._run_compile is compile

def test_malformed_signature_is_refused_before_original_and_retained_poison_refused(tmp_path):
    calls=[]
    def compile(command,purpose):calls.append(command)
    driver=SimpleNamespace(_run_compile=compile)
    cpp=tmp_path/'input.cpp';cpp.write_text('int x;')
    with capture_actual_compiles(tmp_path/'capture',driver=driver) as capture:
        command=[shutil.which('clang++') or 'c++',str(cpp),'-o',str(tmp_path/'out')]
        with pytest.raises(ValueError,match='signature'):driver._run_compile(command,'negative')
        assert calls==[]
