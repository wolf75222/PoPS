"""Source preparation; stand-ins are not Native captures or admission evidence."""
import json,struct,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest,pops
from tests.python.support.evolved_stage_v_capture import capture_initial_carriers,save_json,validate_initial_envelope,retain_v_provenance

def codec():
    word=lambda n:struct.pack('<q',n)
    data=b'POPSCAR1'+b''.join(word(n) for n in (2,64,1,-1,2,2))
    for name in (b'Q0',b'forcing'):data+=word(len(name))+name
    return data+word(0)

def test_source_origin_and_public_require_resolved():
    from tests.python.support.evolved_stage_amr import build
    assert Path(pops.__file__).resolve()==Path(__file__).resolve().parents[2]/'python/pops/__init__.py'
    assert not any(name.rsplit('.',1)[-1]=='_pops' for name in sys.modules)
    case,layout=build(8,1)
    resolved=pops.resolve(pops.validate(case),layout=layout,compile_options={'model_source_policy':'require'})
    assert resolved is not None

@pytest.mark.parametrize('fault',('none','runtime','missing','malformed'))
def test_initial_call_persisted_before_refusal(tmp_path,fault):
    calls=[]
    def capture():
        calls.append('native-codec-call')
        if fault=='runtime':raise RuntimeError('actual capture refusal')
        return b'badcodec' if fault=='malformed' else codec()
    executor=SimpleNamespace() if fault=='missing' else SimpleNamespace(checkpoint_state_carriers=capture)
    owner=SimpleNamespace(_executor=executor);world=SimpleNamespace(rank=0,size=1)
    import tests.python.support.evolved_stage_v_capture as module
    from tests.python.support.collective_checks import collective_attempt,collective_call
    with patch.object(module,'collective_attempt',lambda world,operation:collective_attempt(None,operation)),patch.object(module,'collective_call',lambda world,operation:collective_call(None,operation)):
        if fault!='none':
            with pytest.raises(AssertionError):capture_initial_carriers(world,owner,tmp_path)
        else:capture_initial_carriers(world,owner,tmp_path)
    evidence=json.loads((tmp_path/'initial-carriers-rank0.json').read_text())
    assert calls==([] if fault=='missing' else ['native-codec-call'])
    assert evidence['available'] is (fault in ('none','malformed'))
    if fault in ('runtime','missing'):assert not list(tmp_path.glob('*.bin'))
    else:assert (tmp_path/'initial-carriers-rank0.bin').read_bytes()==(b'badcodec' if fault=='malformed' else codec())

@pytest.mark.parametrize('key,value',[('dim',1),('real',32),('shard',0),('ranks',2),('levels',1),('blocks',['forcing'])])
def test_initial_envelope_refuses_foreign(key,value):
    image=dict(dim=2,real=64,shard=-1,ranks=1,levels=2,blocks=['Q0','forcing']);image[key]=value
    with pytest.raises(ValueError):validate_initial_envelope(image,1)

def test_exact_artifact_and_json_no_coercion(tmp_path):
    with pytest.raises(TypeError,match='exact compiled artifact'):retain_v_provenance(SimpleNamespace(),None,tmp_path,0)
    with pytest.raises(TypeError):save_json(tmp_path/'bytes.json',{'digest':b'rawbytes'})
    with pytest.raises(ValueError):save_json(tmp_path/'nan.json',{'value':float('nan')})

def test_original_fixture_and_physics_unchanged():
    import ast,subprocess
    root=Path(__file__).resolve().parents[2]
    old='tests/python/integration/runtime/test_public_evolved_stage_amr.py'
    assert (root/old).read_bytes()==subprocess.check_output(['git','show','3302c4a1:'+old],cwd=root)
    source=(root/'tests/python/integration/runtime/test_public_evolved_stage_amr_v3.py').read_text()
    tree=ast.parse(source);function=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    original=ast.parse((root/old).read_text());prior=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='test_public_evolved_stage_amr_checkpoint_and_composite_Q')
    assert [ast.dump(n) for n in function.decorator_list]==[ast.dump(n) for n in prior.decorator_list]
    assert 'model_source_policy' in source and '@3' in source
    assert source.index('retain_v_provenance(')<source.index('runtime = collective_call(world, bind)')

def test_genuine_module_manifest_json(tmp_path):
    from pops.model.module import Module
    module=Module('capture-manifest-authority')
    manifest=module.manifest()
    pin=save_json(tmp_path/'module.json',manifest.to_dict())
    assert json.loads((tmp_path/'module.json').read_text())==manifest.to_dict()
    assert len(pin['sha256'])==64


def test_peer_preflight_refusal_never_enters_native(tmp_path):
    import tests.python.support.evolved_stage_v_capture as module
    from tests.python.support.collective_checks import collective_call
    calls=[]
    owner=SimpleNamespace(_executor=SimpleNamespace(checkpoint_state_carriers=lambda:calls.append('forbidden')))
    world=SimpleNamespace(rank=0,size=2)
    with patch.object(module,'collective_attempt',return_value=(None,(None,('RuntimeError','peer missing codec',True)))),patch.object(module,'collective_call',lambda world,operation:collective_call(None,operation)):
        with pytest.raises(AssertionError,match='peer missing codec'):capture_initial_carriers(world,owner,tmp_path)
    assert calls==[]
    receipt=json.loads((tmp_path/'initial-carriers-rank0.json').read_text())
    assert receipt['available'] is False and receipt['failures'][1][1]=='peer missing codec'

def test_genuine_compiled_model_source_retention(tmp_path):
    # Genuine tiny host C25 proof and real CompiledModel API; not a PoPS Native model.
    from tests.review.test_sol61_model_source_evidence import publish
    from pops.codegen.loader import CompiledModel
    from pops.model.module import Module
    from tests.python.support.evolved_stage_v_capture import retain_model_sources
    cpp,binary=publish(tmp_path)
    model=CompiledModel(str(binary),'production',('u',),('scalar',),('u',),1,None,0,{},
                        {'cpu':True,'amr':True,'mpi':False,'gpu':False},'source-host-only','0'*64,
                        'clang++','c++20',2,target='amr_system',
                        module_manifest=Module('source-only-manifest').manifest())
    # Current low-level manifest() is not the aggregate public artifact API.
    with pytest.raises(TypeError,match='requires a CompiledSimulationArtifact'):model.manifest()
    root=tmp_path/'retained';root.mkdir()
    proof=retain_model_sources(model,root,0,'renamed-block')
    assert Path(proof['cpp']['path']).read_bytes()==cpp.read_bytes()
    assert Path(proof['DSO']['path']).read_bytes()==binary.read_bytes()
    assert json.loads(Path(proof['module_ir']['path']).read_text())==model.module_manifest.to_dict()
    assert proof['actual_source']['complete'] is True
    assert proof['companions'] and all(Path(row['path']).is_relative_to(root) for row in proof['companions'].values())
    # An advanced or legacy cache may not fabricate an actual TU.
    legacy=object.__new__(CompiledModel);legacy.so_path=tmp_path/'legacy.so'
    legacy.so_path.write_bytes(b'no actual source evidence')
    with pytest.raises(ValueError,match='provenance unavailable'):retain_model_sources(legacy,root,1,'foreign')
