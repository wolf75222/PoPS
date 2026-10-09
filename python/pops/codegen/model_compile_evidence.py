"""Durable actual model TU evidence; local provenance, not a compiler graph proof."""
import hashlib
import json
import os
from pathlib import Path
import shutil

SUFFIXES=('.pops-model.cpp','.pops-model-compile.json')
CONTRACT='pops.model.actual-compile@1'

def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for data in iter(lambda:stream.read(1048576),b''):digest.update(data)
    return digest.hexdigest()

def paths(binary):return tuple(Path(str(binary)+suffix) for suffix in SUFFIXES)

def _write(path,data):
    import tempfile
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name,delete=False) as stream:
            temporary=stream.name
            stream.write(data if type(data) is bytes else data.encode('utf-8'))
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None and os.path.exists(temporary):os.unlink(temporary)

def retain(binary,source,command,header_signature,original):
    """Retain input before the unchanged compiler call; seal only its successful output."""
    if type(command) not in (list, tuple) or not command or any(type(v) is not str for v in command):
        raise ValueError('actual TU command must be explicit strings')
    if str(source) not in command or command.count(str(source)) != 1:
        raise ValueError('actual TU source is not the unique compiler input')
    outputs = [command[i+1] for i,v in enumerate(command[:-1]) if v == '-o']
    outputs += [v[4:] for v in command if v.startswith('/Fe:')]
    if outputs != [str(binary)]:
        raise ValueError('actual TU command output differs')
    if type(header_signature) is not str or not header_signature:
        raise ValueError('actual TU header signature unavailable')
    compiler=shutil.which(command[0])
    if compiler is None:raise ValueError('actual compiler executable unavailable')
    compiler=Path(compiler).resolve();compiler_hash=sha(compiler)
    raw=Path(source).read_bytes();raw.decode('utf-8')
    cpp,proof=paths(binary)
    if any(p.is_symlink() for p in (cpp,proof)):raise ValueError('model evidence companion is a symlink')
    _write(cpp,raw)
    result=original(command,'backend production, compile_native')
    if Path(source).read_bytes()!=raw or sha(compiler)!=compiler_hash:
        raise ValueError('actual TU or compiler changed during compilation')
    data={'contract':CONTRACT,'source_sha256':sha(cpp),'binary_sha256':sha(binary),
        'compiler_sha256':compiler_hash,'header_signature':header_signature,
        'command':list(command),'compiler_file':str(compiler),'source_path':str(source),
        'binary_path':str(binary),'compiler_hash_observation':'before-and-after-compile',
        'compiler_to_binary_graph_proof':False}
    _write(proof,json.dumps(data,sort_keys=True,separators=(',',':'))+'\n')
    return result

def read(binary,*,require=False):
    cpp,proof=paths(binary)
    if any(p.is_symlink() for p in (cpp,proof)):
        raise ValueError('model evidence companion is a symlink')
    if not cpp.exists() and not proof.exists():
        if require:raise ValueError('actual model TU provenance unavailable in legacy cache; explicit recompile required')
        return {'contract':CONTRACT,'status':'unavailable-legacy','complete':False}
    if any(p.is_symlink() for p in (cpp,proof)) or not all(p.is_file() for p in (cpp,proof)):
        raise ValueError('model evidence companions must be complete regular files without symlinks')
    def object_pairs(pairs):
        obj={}
        for key,value in pairs:
            if key in obj:raise ValueError('duplicate model provenance key')
            obj[key]=value
        return obj
    data=json.loads(proof.read_text(),object_pairs_hook=object_pairs)
    keys={'contract','source_sha256','binary_sha256','compiler_sha256','header_signature','command',
        'compiler_file','source_path','binary_path','compiler_hash_observation','compiler_to_binary_graph_proof'}
    if type(data) is not dict or set(data)!=keys or data['contract']!=CONTRACT:
        raise ValueError('model provenance schema differs')
    if any(type(data[k]) is not str or not data[k] for k in keys-{'command','compiler_to_binary_graph_proof'}):
        raise ValueError('model provenance scalar type differs')
    if type(data['command']) is not list or not data['command'] or any(type(v) is not str for v in data['command']):
        raise ValueError('model provenance command differs')
    for key in ('source_sha256','binary_sha256','compiler_sha256'):
        if len(data[key])!=64 or any(c not in '0123456789abcdef' for c in data[key]):raise ValueError('model provenance digest differs')
    if data['compiler_to_binary_graph_proof'] is not False or data['compiler_hash_observation']!='before-and-after-compile':
        raise ValueError('unsupported model provenance authority')
    if sha(cpp)!=data['source_sha256'] or sha(binary)!=data['binary_sha256']:
        raise ValueError('model TU or binary provenance bytes differ')
    cpp.read_bytes().decode('utf-8')
    return {**data,'status':'available','complete':True,'provenance_sha256':sha(proof)}

def source(binary):
    from .compile_provenance import read_artifact_sidecar
    sidecar=read_artifact_sidecar(binary)
    if sidecar is None or sidecar['protocol']!='pops.artifact-sidecar.v2':
        raise ValueError('actual model source lacks committed provenance authority')
    evidence=read(binary,require=True)
    raw=paths(binary)[0].read_bytes()
    if hashlib.sha256(raw).hexdigest()!=evidence['source_sha256']:
        raise ValueError('actual TU changed before export')
    return raw.decode('utf-8')

def publish(staging,destination):
    for old,new in zip(paths(staging),paths(destination)):
        if old.exists():
            if old.is_symlink() or new.is_symlink():raise ValueError('model evidence companion symlink')
            os.replace(old,new)

def policy(value):
    if type(value) is not str or value not in ('allow_missing','require','recompile'):
        raise ValueError('model_source_policy must be allow_missing, require or recompile')
    return value

def guard_recompile(binary):
    """Never replace a path already published in this interpreter's loader registry."""
    from .cache import _process_so_identity, _process_so_identity_lock
    with _process_so_identity_lock:
        if os.path.abspath(os.fspath(binary)) in _process_so_identity:
            raise ValueError('explicit model recompile requires a fresh process or fresh cache path; native loader path was already published')


def codegen_source_authority():
    """Versioned emitter implementation authority, independent of SDK headers.

    Logical relative names and source bytes define identity. Paths, timestamps,
    bytecode and compiler outputs do not. This deliberately invalidates model
    caches on any codegen implementation change, rather than claiming a TU hash.
    """
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256(b"pops.codegen-source@1\0")
    files = sorted(root.rglob("*.py"), key=lambda item: item.relative_to(root).as_posix())
    if not files:
        raise ValueError("codegen source authority unavailable")
    for path in files:
        if path.is_symlink():
            raise ValueError("codegen source authority refuses symlink")
        name = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        for part in (name, data):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return {"contract": "pops.codegen-source@1", "sha256": digest.hexdigest()}
