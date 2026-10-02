"""Source diagnostic persistence, no Native or synthetic image proof."""
import json
from pathlib import Path
import pytest
from tests.python.support.initial_ghost_failure_receipt import save_early_bind_receipt

def args():
    return dict(rank=0,world_size=2,failures=[('ValueError','actual early bind error',False),('RuntimeError','peer refusal',True)],owner_count=0,image_count=0,target_count=0,native={'path':'source-only-dso-path','sha256':'source-only-pin'},artifact_identity='SourceOnly',component_manifest={'scope':'SourceOnly'})

def test_early_failure_without_owner_is_saved_without_image(tmp_path):
    (tmp_path/'callback-rank0.log').write_text('prepare field=0x0p+0\ndestroy field=0x0p+0\n')
    path=save_early_bind_receipt(tmp_path,**args());r=json.loads(path.read_text())
    assert r['failures']==[list(v) for v in args()['failures']]
    assert r['observed_counts']=={'owners':0,'images':0,'targets':0}
    assert r['callback_log'].startswith('prepare') and 'rollback or image certification' in r['scope']
    assert set(p.name for p in tmp_path.iterdir())=={'callback-rank0.log','early-bind-rank0.json'}

@pytest.mark.parametrize('changes',[{'world_size':True},{'failures':[None]},{'owner_count':True},{'failures':[('E','msg',0),None]}])
def test_malformed_receipt_refused_before_write(tmp_path,changes):
    a=args();a.update(changes)
    with pytest.raises(ValueError):save_early_bind_receipt(tmp_path,**a)
    assert not list(tmp_path.iterdir())

def test_fixture_persists_before_existing_snapshot_assertions():
    source=(Path(__file__).resolve().parents[2]/'tests/python/integration/amr/test_public_initial_field_ghost_failure.py').read_text()
    assert source.index('_,failures=collective_attempt')<source.index('save_early_bind_receipt(directory')<source.index('assert len(images)==1 and len(owners)==1')
    assert "assert before==after" in source and "assert all(failures)" in source
