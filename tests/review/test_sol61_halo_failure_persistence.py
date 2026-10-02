"""SOURCE-only persistence/orchestration checks; synthetic arrays are not Native evidence."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from pops.runtime._checkpoint_manifest import MANIFEST_KEY, IDENTITY_KEY
from tests.python.integration.amr import test_public_accepted_halo_stage_failure as fixture


def payload():
    return {'pops_amr_checkpoint_version':np.array(12),
            'amr_accepted_contract':np.array(json.dumps({'schema_version':9})),
            'state':np.array([1.]),'clock':np.array([2.]),'history':np.array([3.]),
            'field':np.array([4.]),'diagnostics':np.array([5.]),
            MANIFEST_KEY:np.array('retry seal'),IDENTITY_KEY:np.array('retry identity')}


def test_checkpoint_pair_records_actual_file_hashes_and_only_two_seals(tmp_path):
    a=payload();b={k:v.copy() for k,v in a.items()}
    b[MANIFEST_KEY]=np.array('control seal');b[IDENTITY_KEY]=np.array('control identity')
    retry=tmp_path/'retry.npz';control=tmp_path/'control.npz'
    np.savez_compressed(retry,**a);np.savez_compressed(control,**b)
    proof=fixture.checkpoint_pair_proof(retry,control)
    assert proof['excluded_lifecycle_seals']==[MANIFEST_KEY,IDENTITY_KEY]
    assert proof['lifecycle_seal_differences']==[MANIFEST_KEY,IDENTITY_KEY]
    for name,path in [('retry',retry),('continuous_control',control)]:
        assert proof['checkpoints'][name]=={'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize('key',['state','clock','history','field','diagnostics'])
def test_checkpoint_pair_refuses_changed_physical_member(tmp_path,key):
    a=payload();b={k:v.copy() for k,v in a.items()};b[key][0]=np.nextafter(b[key][0],np.inf)
    retry=tmp_path/'retry.npz';control=tmp_path/'control.npz'
    np.savez_compressed(retry,**a);np.savez_compressed(control,**b)
    with pytest.raises(AssertionError,match=key):fixture.checkpoint_pair_proof(retry,control)


def test_fixture_persists_divergence_and_two_genuine_collective_checkpoints():
    source=Path(fixture.__file__).read_text();tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_public_'))
    assignments={n.targets[0].id:n for n in ast.walk(fn) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    assert isinstance(assignments['divergent_failures'].value,ast.List)
    dictionaries=[n for n in ast.walk(fn) if isinstance(n,ast.Dict)]
    assert any('divergent_request_applicable' in [k.value for k in n.keys if isinstance(k,ast.Constant)] and 'divergent_request_refusals' in [k.value for k in n.keys if isinstance(k,ast.Constant)] for n in dictionaries)
    for variable,owner,stem in [('retry_checkpoint','runtime','retry-checkpoint'),('control_checkpoint','control','continuous-control-checkpoint')]:
        call=assignments[variable].value
        assert isinstance(call,ast.Call) and call.func.id=='collective_call'
        inner=call.args[1].body
        assert inner.func.attr=='checkpoint' and inner.func.value.id==owner
        assert inner.args[0].right.value==stem
    assert "checkpoint_pair_proof(retry_checkpoint,control_checkpoint)" in source
