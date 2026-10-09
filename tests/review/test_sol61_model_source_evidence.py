"""Real tiny compiler and actual inspection methods; no PoPS Native/JIT."""
import json
import subprocess
import shutil
from pathlib import Path
import pytest
from pops.codegen import model_compile_evidence as evidence
from pops.codegen.compile_provenance import publish_staged_artifact,read_artifact_sidecar,StaleArtifactError
from pops.identity import make_identity
from pops.codegen.loader import CompiledModel

def tiny(tmp_path):
    cpp=tmp_path/'actual.cpp';cpp.write_bytes(b'extern "C" int actual(){return 19;}\r\n')
    binary=tmp_path/'staged.so'
    command=[shutil.which('clang++') or 'c++','-shared','-fPIC',str(cpp),'-o',str(binary)]
    evidence.retain(binary,cpp,command,'source-host-only',lambda command,purpose:subprocess.run(command,check=True,capture_output=True))
    return cpp,binary

def publish(tmp_path):
    cpp,stage=tiny(tmp_path);binary=tmp_path/'published.so'
    publish_staged_artifact(stage,binary,semantic_identity=make_identity('semantic',{'actual':'tiny'}),
        spec_identity=make_identity('artifact-spec',{'actual':'tiny'}))
    return cpp,binary

def test_actual_tu_cache_inspection_dump_and_committed_v2(tmp_path):
    cpp,binary=publish(tmp_path)
    model=object.__new__(CompiledModel);model.so_path=binary
    record=model.source_provenance(require_complete=True)
    assert record['complete'] and record['compiler_to_binary_graph_proof'] is False
    assert read_artifact_sidecar(binary)['protocol']=='pops.artifact-sidecar.v2'
    out=tmp_path/'dump.cpp';model.dump_cpp(out)
    assert out.read_bytes()==cpp.read_bytes() and model._generated_cpp.encode()==cpp.read_bytes()
    assert model.source_provenance()==record  # actual published cache hit; no compiler rerun

def test_cache_inspection_uses_recorded_compiler_hash_not_live_path(tmp_path,monkeypatch):
    _,binary=publish(tmp_path)
    record=evidence.read(binary);original=evidence.sha
    def no_live_compiler(path):
        assert str(path)!=record['compiler_file'],'inspection must not remint compiler evidence'
        return original(path)
    monkeypatch.setattr(evidence,'sha',no_live_compiler)
    assert evidence.source(binary)

@pytest.mark.parametrize('mutation',['source','proof','binary','symlink','truncated'])
def test_actual_companions_poison_fail_closed(tmp_path,mutation):
    cpp,binary=publish(tmp_path);source,proof=evidence.paths(binary)
    if mutation=='source':source.write_bytes(b'foreign source')
    elif mutation=='proof':
        data=json.loads(proof.read_text());data['header_signature']='forged';proof.write_text(json.dumps(data))
    elif mutation=='binary':binary.write_bytes(b'foreign binary')
    elif mutation=='symlink':source.unlink();source.symlink_to(cpp)
    else:proof.write_text('{')
    with pytest.raises((StaleArtifactError,ValueError)):
        evidence.source(binary)

def test_legacy_unavailable_explicit_no_regeneration_and_policy_types(tmp_path):
    model=object.__new__(CompiledModel);model.so_path=tmp_path/'legacy.so';model.so_path.write_bytes(b'legacy')
    assert model.source_provenance()=={'contract':evidence.CONTRACT,'status':'unavailable-legacy','complete':False}
    assert model._generated_cpp is None
    with pytest.raises(ValueError):model.dump_cpp(tmp_path/'not-minted.cpp')
    with pytest.raises(ValueError):model.source_provenance(require_complete=True)
    assert not (tmp_path/'not-minted.cpp').exists()
    for value in (None,True,'guess'):
        with pytest.raises(ValueError):evidence.policy(value)

def test_public_policy_projected_to_all_models_before_compile():
    root=Path(__file__).resolve().parents[2]
    phases=(root/'python/pops/codegen/_phases.py').read_text()
    orchestra=(root/'python/pops/codegen/_orchestration_compile.py').read_text()
    facade=(root/'python/pops/physics/_facade_compile.py').read_text()
    assert '"model_source_policy"' in phases
    assert '("include", "cxx", "std", "model_source_policy")' in orchestra
    assert 'model_source_policy != "recompile"' in facade
    assert 'read(so_path, require=model_source_policy == "require")' in facade

def test_public_model_cache_policy_legacy_require_and_explicit_recompile(tmp_path,monkeypatch):
    from tests.python.unit.codegen.test_facade_compile_cache import _model
    calls=[]
    def compiler(path,*args,**kwargs):
        calls.append(path);Path(path).write_bytes(b'metadata-only compiler fixture');return path
    model=_model(tmp_path/'cache',monkeypatch,compiler)
    first=model.compile(include='test-headers')
    assert first.source_provenance()['complete'] is False
    model.compile(include='test-headers',model_source_policy='allow_missing')
    assert len(calls)==1
    with pytest.raises(ValueError,match='unavailable'):
        model.compile(include='test-headers',model_source_policy='require')
    assert len(calls)==1
    with pytest.raises(ValueError,match='fresh process'):
        model.compile(include='test-headers',model_source_policy='recompile')
    assert len(calls)==1  # cannot replace a previously published loader path in this process
    fresh_calls=[]
    def fresh_compiler(path,*args,**kwargs):
        fresh_calls.append(path);Path(path).write_bytes(b'metadata-only fixture');return path
    fresh=_model(tmp_path/'fresh-cache',monkeypatch,fresh_compiler)
    with pytest.raises(ValueError,match='unavailable'):
        fresh.compile(include='test-headers',model_source_policy='recompile')
    assert len(fresh_calls)==1  # genuinely enters compiler route; cannot certify a fake TU

def test_public_resolution_accepts_typed_policy_and_refuses_invalid_before_compile():
    import pops
    from tests.python.support.atomic_cubature_path_case import make_case
    case,layout=make_case(nonconservative=True)
    resolved=pops.resolve(pops.validate(case),layout=layout,compile_options={'model_source_policy':'require'})
    assert resolved.compile_options['model_source_policy']=='require'
    with pytest.raises(ValueError,match='model_source_policy'):
        pops.resolve(pops.validate(case),layout=layout,compile_options={'model_source_policy':True})

def test_cache_lock_reentrant_and_companion_publish_commit_last(tmp_path,monkeypatch):
    from pops.codegen.cache import _artifact_cache_lock
    import pops.codegen.compile_provenance as provenance
    cpp,stage=tiny(tmp_path);dest=tmp_path/'published.so'
    original=provenance.os.replace;events=[]
    def replace(a,b):events.append(str(b));return original(a,b)
    monkeypatch.setattr(provenance.os,'replace',replace)
    with _artifact_cache_lock(dest):
        with _artifact_cache_lock(dest):
            publish_staged_artifact(stage,dest,semantic_identity=make_identity('semantic',{}),spec_identity=make_identity('artifact-spec',{}))
    assert events[-1]==str(dest)+'.pops-artifact.json'
    assert events.index(str(dest)+'.pops-model.cpp')<len(events)-1

def test_foreign_semantic_state_authority_refused_despite_valid_tu(tmp_path):
    from pops.codegen.compile_provenance import verify_cached_artifact
    _,binary=publish(tmp_path)
    with pytest.raises(StaleArtifactError):
        verify_cached_artifact(str(binary),semantic_identity=make_identity('semantic',{'actual':'foreign-state'}),
            spec_identity=make_identity('artifact-spec',{'actual':'tiny'}))

def test_codegen_source_authority_changes_without_header_or_native_change(tmp_path, monkeypatch):
    from pops.codegen import model_compile_evidence as evidence
    root = tmp_path / 'codegen'
    root.mkdir()
    marker = root / 'model_compile_evidence.py'
    marker.write_text('# marker\n')
    emitter = root / 'emitter.py'
    emitter.write_text('def emit():\n    pass\n')
    monkeypatch.setattr(evidence, '__file__', str(marker))
    old = evidence.codegen_source_authority()
    emitter.write_text('def emit():\n    return result\n')
    new = evidence.codegen_source_authority()
    assert old['contract'] == new['contract'] == 'pops.codegen-source@1'
    assert old['sha256'] != new['sha256']
    relocated = tmp_path / 'relocated'
    import shutil
    shutil.copytree(root, relocated)
    monkeypatch.setattr(evidence, '__file__', str(relocated / marker.name))
    assert evidence.codegen_source_authority() == new
    (relocated / 'ignored.pyc').write_bytes(b'foreign bytecode')
    assert evidence.codegen_source_authority() == new
