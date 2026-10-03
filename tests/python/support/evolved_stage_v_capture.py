"""V@3 test-only actual source and initial CP12 codec capture; no regeneration."""
import hashlib,json
from pathlib import Path
import numpy as np
from tests.python.support.collective_checks import collective_attempt,collective_call
from tests.python.support.m16_explicit_native_capture import dump_retained_program

def pin(path):
    p=Path(path)
    return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def save_json(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,allow_nan=False,indent=2)+'\n')
    return pin(path)

def retain_model_sources(model,root,index,name):
    from pops.codegen.compile_provenance import artifact_sidecar_path
    proof=model.source_provenance(require_complete=True)
    if proof.get('complete') is not True:raise ValueError('complete actual Model source required')
    cpp=root/('model-%d.cpp'%index);model.dump_cpp(cpp)
    if pin(cpp)['sha256']!=proof['source_sha256']:raise ValueError('actual Model dump hash differs from C25 proof')
    # CompiledModel carries the immutable compiler-owned Module IR manifest;
    # it has no dump_ir API. Never re-serialize a live authoring model here.
    ir=save_json(root/('model-%d.module-ir.json'%index),model.module_manifest.to_dict())
    binary=Path(model.so_path);sidecar=Path(artifact_sidecar_path(str(binary)))
    companions={}
    companion_dir=root/('model-%d-companions'%index);companion_dir.mkdir()
    for source in binary.parent.glob(binary.name+'.*'):
        if source.is_file() and not source.name.endswith('.lock'):
            retained=companion_dir/source.name;retained.write_bytes(source.read_bytes())
            companions[source.name]=pin(retained)
    retained_binary=root/('model-%d.so'%index);retained_binary.write_bytes(binary.read_bytes())
    retained_sidecar=root/('model-%d.artifact.json'%index);retained_sidecar.write_bytes(sidecar.read_bytes())
    if pin(retained_binary)['sha256']!=proof['binary_sha256']:raise ValueError('actual Model binary differs from C25 proof')
    return {'component':'block-'+name,'DSO':pin(retained_binary),'sidecar':pin(retained_sidecar),
                    'cpp':pin(cpp),'module_ir':ir,'actual_source':proof,'companions':companions}

def retain_v_provenance(artifact,native,directory,rank):
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact
    from pops.codegen.compile_provenance import artifact_sidecar_path
    if type(artifact) is not CompiledSimulationArtifact:raise TypeError('exact compiled artifact required')
    artifact.verify()
    block_names=tuple(block.name for block in artifact.blocks)
    if len(set(block_names))!=len(block_names):raise ValueError('duplicate artifact block')
    covered=tuple(name for row in artifact.layout_programs for name in row.block_names)
    if len(set(covered))!=len(covered) or set(covered)!=set(block_names):raise ValueError('exact full Program/model partition required')
    root=directory/('provenance-rank%d'%rank);root.mkdir(exist_ok=False)
    compiled_manifest=save_json(root/'compiled-manifest.json',artifact.manifest().to_dict())
    entries=[]
    for index,block in enumerate(artifact.blocks):
        entries.append(retain_model_sources(block.model,root,index,block.name))
    for index,row in enumerate(artifact.layout_programs):
        row.verify();path=root/('layout-%d'%index);path.mkdir();program=row.program
        dump_retained_program(program,path)
        binary=path/'program.so';binary.write_bytes(Path(program.so_path).read_bytes())
        sidecar=path/'program.artifact.json';sidecar.write_bytes(Path(artifact_sidecar_path(str(program.so_path))).read_bytes())
        entries.append({'component':'program-'+row.layout_id,'layout_identity':row.identity.token,
                        'block_names':list(row.block_names),'target':row.target,
                        'DSO':pin(binary),'sidecar':pin(sidecar),
                        'cpp':pin(path/'program.cpp'),'ir.json':pin(path/'program.ir.json'),
                        'compiled_manifest':compiled_manifest,'program_hash':program.program_hash,'command':program.compile_command})
    save_json(root/'receipt.json',{'schema':'pops.evolved-stage-v-retained-provenance@1','artifact':artifact.artifact_identity.token,
             'rank':rank,'native':pin(native.__file__),'compilation':entries,'before_bind':True})
    return entries

def capture_initial_carriers(world,runtime,directory):
    # Existing native codec used by CP12: no publication, solve or Field getter.
    method,failures=collective_attempt(world,lambda:required_initial_method(runtime))
    raw=None
    if not any(failures):raw,failures=collective_attempt(world,method)
    def save():
        evidence={'schema':'pops.evolved-stage-v-initial-carriers@1','rank':world.rank,
                  'ranks':world.size,'failures':failures,'available':raw is not None}
        if raw is not None:
            if type(raw) is not bytes:raise TypeError('native CP12 codec must return exact bytes')
            path=directory/('initial-carriers-rank%d.bin'%world.rank);path.write_bytes(raw);evidence['file']=pin(path)
        receipt=directory/('initial-carriers-rank%d.json'%world.rank)
        save_json(receipt,evidence)
        return {**evidence,'receipt':pin(receipt)}
    evidence=collective_call(world,save)
    # Real refusal persisted before admission; no NPY/carrier substitute.
    if any(failures):raise AssertionError(failures)
    from tests.review.sol61_amr_full_carrier_offline import decode
    def validate():
        image=decode(np.frombuffer(raw,dtype=np.uint8))
        validate_initial_envelope(image,world.size)
    collective_call(world,validate)
    return evidence

def required_initial_method(runtime):
    method=getattr(runtime._executor,'checkpoint_state_carriers',None)
    if not callable(method):raise RuntimeError('existing native CP12 carrier capture unavailable')
    return method

def validate_initial_envelope(image,ranks):
    if image['dim']!=2 or image['real']!=64 or image['shard']!=-1 or image['ranks']!=ranks or image['levels']!=2:
        raise ValueError('initial full global CP12 carrier envelope differs')
    if image['blocks'] not in (['Q0','forcing'],['Q0','Q1','forcing']):raise ValueError('initial artifact block registry differs')
