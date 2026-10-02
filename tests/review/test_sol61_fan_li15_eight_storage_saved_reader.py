"""Synthetic offline storage probes, never receipts of Native experiments."""
import hashlib,json,struct
import numpy as np
import pytest
from tests.review import sol61_fan_li15_eight_storage_saved_reader as r

def wire(values,shard,*,partitioned=False):
    u=lambda x:struct.pack('<Q',x);i=lambda x:struct.pack('<q',x)
    grown=np.pad(values,((0,0),(1,1),(1,1)),constant_values=-7.)
    head=b'POPSCAR1'+u(2)+u(64)+u(2)+i(shard)+u(1)+u(1)+u(3)+b'gas'+u(0 if partitioned and shard==1 else 1)
    if partitioned and shard==1:return head
    row=u(0)+u(0)+u(0)+u(15)+i(0 if partitioned else -1)+b''.join(i(x) for x in (0,15,-1,16,0,15,-1,16))
    return head+row+u(grown.size)+grown.view(np.uint64).astype('<u8').tobytes()

def captures(path,*,partitioned=False):
    values=np.arange(15*16*16,dtype=np.float64).reshape(15,16,16);values[0,0,0]=-0.
    for step,p in enumerate(r.PHASES):
        np.save(path/(p+'.npy'),values)
        (path/(p+'.clock.json')).write_text(json.dumps([float(step*1e-4),step]))
        rows=[]
        for shard,label in ((-1,'complete'),(0,'rank0'),(1,'rank1')):
            name=p+'.'+label+'.carriers';(path/name).write_bytes(wire(values,shard,partitioned=partitioned))
            rows.append({'file':name,'sha256':hashlib.sha256((path/name).read_bytes()).hexdigest()})
        meta={'contract':'accepted-state-storage-observation@1','dimension':2,'time':float(step*1e-4),'macro_step':step,'ranks':2,'complete':rows[0],'rank_local':rows[1:]}
        (path/(p+'.storage.json')).write_text(json.dumps(meta))
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}

def test_replicated_full_grown_nine_phases_with_explicit_world(tmp_path):
    report=r.audit_storage(tmp_path,captures(tmp_path))
    assert report['ranks']==2 and report['patch_counts']==[1]*9
    assert report['native_authority'] is report['ghost_formula_qualified'] is False

@pytest.mark.parametrize('mutation',('missing','clock','bool_world','foreign_file','duplicate_rank','grown_bit','valid_bit','truncated'))
def test_resealed_storage_mutants_refused(tmp_path,mutation):
    pins=captures(tmp_path);name='accepted8.storage.json';meta=json.loads((tmp_path/name).read_text())
    if mutation=='missing':pins.pop('accepted3.rank1.carriers')
    elif mutation=='clock':meta['macro_step']=7
    elif mutation=='bool_world':meta['ranks']=True
    elif mutation=='foreign_file':meta['complete']['file']='../foreign'
    elif mutation=='duplicate_rank':meta['rank_local'][1]=meta['rank_local'][0]
    elif mutation=='valid_bit':
        values=np.load(tmp_path/'accepted8.npy');values[0,0,0]=0.;np.save(tmp_path/'accepted8.npy',values)
        pins['accepted8.npy']=hashlib.sha256((tmp_path/'accepted8.npy').read_bytes()).hexdigest()
    else:
        key='accepted8.rank1.carriers';data=(tmp_path/key).read_bytes()
        data=data[:-1] if mutation=='truncated' else data[:-8]+struct.pack('<d',99.)
        (tmp_path/key).write_bytes(data);pins[key]=hashlib.sha256(data).hexdigest();meta['rank_local'][1]['sha256']=pins[key]
    (tmp_path/name).write_text(json.dumps(meta));pins[name]=hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
    with pytest.raises(ValueError):r.audit_storage(tmp_path,pins)


def test_partitioned_empty_rank_preserves_full_inventory(tmp_path):
    report=r.audit_storage(tmp_path,captures(tmp_path,partitioned=True))
    assert report['ranks']==2 and report['patch_counts']==[1]*9


def test_coherently_resealed_replicated_nan_grown_is_refused(tmp_path):
    pins=captures(tmp_path);phase='accepted8';name=phase+'.storage.json'
    meta=json.loads((tmp_path/name).read_text())
    for row in [meta['complete'],*meta['rank_local']]:
        path=tmp_path/row['file'];path.write_bytes(path.read_bytes()[:-8]+struct.pack('<d',float('nan')))
        row['sha256']=pins[row['file']]=hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/name).write_text(json.dumps(meta));pins[name]=hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='nonfinite grown'):r.audit_storage(tmp_path,pins)
