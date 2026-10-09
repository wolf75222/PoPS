from pathlib import Path
import pytest
from tests.python.unit.codegen.test_facade_compile_cache import _model

def test_review_imports_the_exact_checkout_without_native():
    import sys
    from pops.codegen import model_compile_evidence as evidence
    root=Path(__file__).resolve().parents[2]
    assert Path(evidence.__file__).resolve()==root/'python/pops/codegen/model_compile_evidence.py'
    assert not any(name=='_pops' or name.endswith('._pops') for name in sys.modules)

def test_explicit_destination_recompile_refuses_prior_publication(tmp_path, monkeypatch):
    calls=[]
    def compiler(path,*args,**kwargs):
        calls.append(path)
        Path(path).write_bytes(b'metadata-only, not Native')
        return path
    model=_model(tmp_path/'cache',monkeypatch,compiler)
    first=model.compile(include='test-headers')
    with pytest.raises(ValueError,match='fresh process'):
        model.compile(include='test-headers',so_path=first.so_path,model_source_policy='recompile')
    assert len(calls)==1

def test_compiled_handle_refuses_coherent_foreign_republication(tmp_path):
    import shutil,subprocess
    from pops.codegen import model_compile_evidence as evidence
    from pops.codegen.compile_provenance import publish_staged_artifact
    from pops.codegen.loader import CompiledModel
    from pops.identity import make_identity
    semantic=make_identity('semantic',{'state':'original'})
    spec=make_identity('artifact-spec',{'state':'original'})
    dest=tmp_path/'shared.so'
    def compile_and_publish(value,sem,specification):
        cpp=tmp_path/'input.cpp';stage=tmp_path/'staging.so'
        cpp.write_text('extern "C" int result(){return %d;}\n'%value)
        cmd=[shutil.which('clang++'),'-shared','-fPIC',str(cpp),'-o',str(stage)]
        evidence.retain(stage,cpp,cmd,'host-only',lambda cmd,purpose:subprocess.run(cmd,check=True,capture_output=True))
        return publish_staged_artifact(stage,dest,semantic_identity=sem,spec_identity=specification)
    binary,artifact=compile_and_publish(1,semantic,spec)
    model=object.__new__(CompiledModel)
    model.so_path=dest;model.semantic_identity=semantic;model.artifact_spec_identity=spec
    model.binary_identity=binary;model.artifact_identity=artifact
    compile_and_publish(2,make_identity('semantic',{'state':'foreign'}),make_identity('artifact-spec',{'state':'foreign'}))
    with pytest.raises(ValueError):model.source_provenance(require_complete=True)
    with pytest.raises(ValueError):model.dump_cpp(tmp_path/'misattributed.cpp')
    assert not (tmp_path/'misattributed.cpp').exists()

def test_export_never_returns_bytes_changed_after_authentication(tmp_path,monkeypatch):
    import shutil,subprocess,hashlib
    from pops.codegen import model_compile_evidence as evidence
    from pops.codegen.compile_provenance import publish_staged_artifact
    from pops.identity import make_identity
    cpp=tmp_path/'input.cpp';stage=tmp_path/'staging.so';dest=tmp_path/'published.so'
    cpp.write_text('extern "C" int result(){return 1;}\n')
    command=[shutil.which('clang++'),'-shared','-fPIC',str(cpp),'-o',str(stage)]
    evidence.retain(stage,cpp,command,'host-only',lambda cmd,purpose:subprocess.run(cmd,check=True,capture_output=True))
    publish_staged_artifact(stage,dest,semantic_identity=make_identity('semantic',{}),spec_identity=make_identity('artifact-spec',{}))
    pinned=evidence.read(dest)['source_sha256']
    source=evidence.paths(dest)[0]
    original=evidence.read
    calls=0
    def concurrent_change(binary,**kwargs):
        nonlocal calls
        calls+=1
        record=original(binary,**kwargs)
        if calls==2:source.write_bytes(b'foreign replacement after verified read\n')
        return record
    monkeypatch.setattr(evidence,'read',concurrent_change)
    try: exported=evidence.source(dest)
    except ValueError: return
    assert hashlib.sha256(exported.encode()).hexdigest()==pinned

def test_codegen_source_authority_detects_real_emitter_return(tmp_path,monkeypatch):
    from pops.codegen import model_compile_evidence as evidence
    marker=tmp_path/'model_compile_evidence.py';marker.write_text('')
    emitter=tmp_path/'moment_path_kernel.py'
    emitter.write_text('def emit():\n    integral = 1\n')
    monkeypatch.setattr(evidence,'__file__',str(marker))
    old=evidence.codegen_source_authority()
    emitter.write_text('def emit():\n    integral = 1\n    return integral\n')
    assert evidence.codegen_source_authority()!=old

def test_actual_tu_association_refuses_foreign_compiler_input(tmp_path):
    import shutil,subprocess
    from pops.codegen import model_compile_evidence as evidence
    a=tmp_path/'a.cpp';b=tmp_path/'b.cpp';binary=tmp_path/'out.so'
    a.write_text('extern "C" int result(){return 1;}\n')
    b.write_text('extern "C" int result(){return 2;}\n')
    command=[shutil.which('clang++'),'-shared','-fPIC',str(b),'-o',str(binary)]
    calls=[]
    def original(cmd,purpose):
        calls.append(cmd)
        subprocess.run(cmd,check=True,capture_output=True)
    with pytest.raises(ValueError,match='(input|source|command|TU)'):
        evidence.retain(binary,a,command,'host-only',original)
    assert calls==[]

def test_backend_explicit_recompile_refuses_prior_publication(tmp_path,monkeypatch):
    from pops.codegen import _compile_drivers as driver
    _model(tmp_path/'cache',monkeypatch,lambda *a,**k: None)
    import pops
    from tests.python.support.atomic_cubature_path_case import make_case
    from pops.codegen.module_lowering import lower_and_validate
    case,layout=make_case(nonconservative=True)
    resolved=pops.resolve(pops.validate(case),layout=layout)
    model,_=lower_and_validate(resolved.blocks[0].model, state_space=resolved.blocks[0].state_spaces[0],
                             resolved_operations=resolved.blocks[0].resolved_operations,numerics=resolved.blocks[0].numerics)
    model=model._m
    monkeypatch.setattr(driver,'_native_kokkos_compiler',lambda _: 'host-source-only')
    monkeypatch.setattr(driver,'_abi_key_python',lambda *args:'test-abi')
    calls=[]
    def compiler(model,path,*args,**kwargs):
        calls.append(path);Path(path).write_bytes(b'not a Native result');return path
    monkeypatch.setattr(driver,'compile_native',compiler)
    first=driver.compile_model(model,include='test-headers',std='c++23')
    with pytest.raises(ValueError,match='fresh process'):
        driver.compile_model(model,include='test-headers',std='c++23',so_path=first,
                             model_source_policy='recompile')
    assert len(calls)==1
