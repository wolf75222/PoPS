"""Synthetic pure engineering receiver probes, never Native receipts."""
import hashlib,json,struct
import numpy as np
import pytest
from tests.review import sol61_amr_gather_engineering_saved_reader as r

def archive(ranks,policy,*,initial=False):
    u=lambda x:struct.pack('<Q',x);i=lambda x:struct.pack('<q',x)
    rows=[]
    for b,width in enumerate(r.WIDTHS):
        for level in (0,1):
            n=8*2**level
            values=(np.arange(width*n*n,dtype=np.float64).reshape(width,n,n)%31-15)/8;values[:,::2,::2]=-0.
            if initial:values=np.zeros_like(values)
            boxes=[(0,n-1,0,n-1)] if level==0 else [(2,5,0,n-1)]
            if level==0 and policy=='partitioned':boxes=[(0,3,0,7),(4,7,0,7)]
            for patch,(xl,xh,yl,yh) in enumerate(boxes):
                owner=-1 if level==0 and policy=='replicated' else patch%ranks
                grown=np.pad(values[:,yl:yh+1,xl:xh+1],((0,0),(1,1),(1,1)),constant_values=-17.)
                row=u(b)+u(level)+u(patch)+u(width)+i(owner)
                row+=b''.join(i(v) for v in (xl,xh,xl-1,xh+1,yl,yh,yl-1,yh+1))
                rows.append(row+u(grown.size)+grown.view(np.uint64).astype('<u8').tobytes())
    head=b'POPSCAR1'+u(2)+u(64)+u(ranks)+i(-1)+u(2)+u(3)
    for name in r.BLOCKS:head+=u(len(name))+name.encode()
    return head+u(len(rows))+b''.join(rows)

def captures(path,policy='empty-owner',ranks=2):
    for rank in range(ranks):
        files={};top=[]
        for phase in ('initial','written'):
            name=phase+'-rank%d.carriers'%rank;(path/name).write_bytes(archive(ranks,policy,initial=phase=='initial'))
            files[name]=hashlib.sha256((path/name).read_bytes()).hexdigest()
        data=r.decode(np.frombuffer((path/('written-rank%d.carriers'%rank)).read_bytes(),dtype=np.uint8))
        for b,name in enumerate(r.BLOCKS):
            for level in (0,1):
                values,covered,_=r.reconstruct(data,b,level);filename='rank%d-block%d-level%d.npy'%(rank,b,level)
                np.save(path/filename,values);files[filename]=hashlib.sha256((path/filename).read_bytes()).hexdigest()
                top.append({'block':name,'level':level,'holes':int(np.sum(covered==0))})
        owners=[p['owner'] for p in data['patches'] if p['key'][:2]==(0,0)]
        counts=[sum(owner in (-1,j) for owner in owners) for j in range(ranks)]
        receipt={'schema':'pops.amr-gather-object-bytes-engineering@1','policy':policy,'rank':rank,'size':ranks,'files':files,'clock_before':[0.,0,[]],'clock_after':[0.,0,[]],'coarse_local_boxes_by_rank':counts,'topology':top,'root_seal':False}
        (path/('receipt-rank%d.json'%rank)).write_text(json.dumps(receipt))
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}

@pytest.mark.parametrize('policy',('replicated','partitioned','empty-owner'))
def test_exact_three_block_partial_fine_geometry_and_owners(tmp_path,policy):
    result=r.receive(tmp_path,captures(tmp_path,policy),policy,2)
    assert result['native_authority'] is False and result['provenance_join_required'] is True

@pytest.mark.parametrize('mutation',('zero-sign','hole-sign','clock','bool_rank','missing','owner_count','wrong_partition'))
def test_resealed_adversaries_refused(tmp_path,mutation):
    pins=captures(tmp_path);name='receipt-rank1.json';record=json.loads((tmp_path/name).read_text())
    if mutation in ('zero-sign','hole-sign'):
        key='rank1-block2-level1.npy';values=np.load(tmp_path/key)
        if mutation=='zero-sign':values[0,0,2]=0.
        else:values[0,0,0]=-0.
        np.save(tmp_path/key,values);record['files'][key]=pins[key]=hashlib.sha256((tmp_path/key).read_bytes()).hexdigest()
    elif mutation=='clock':record['clock_after'][1]=1
    elif mutation=='bool_rank':record['rank']=True
    elif mutation=='missing':pins.pop('rank1-block0-level0.npy')
    elif mutation=='owner_count':record['coarse_local_boxes_by_rank']=[1,1]
    else:record['policy']='partitioned'
    (tmp_path/name).write_text(json.dumps(record));pins[name]=hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
    with pytest.raises(ValueError):r.receive(tmp_path,pins,'empty-owner',2)


@pytest.mark.parametrize('mutation',(None,'contract','header','foreign-binary','missing-source'))
def test_explicit_provenance_join_synthetic_correspondence(tmp_path,mutation):
    # Metadata and arbitrary bytes only: no compiler/Native authority claimed.
    files={};exports={};rows=[]
    for name in (*r.BLOCKS,'program'):
        binary=tmp_path/(name+'.so');binary.write_bytes(('synthetic '+name).encode())
        digest=hashlib.sha256(binary.read_bytes()).hexdigest();exports[name]=(binary,digest)
        if name=='program':continue
        source=name+'.model.cpp';(tmp_path/source).write_text('// synthetic '+name)
        sha=hashlib.sha256((tmp_path/source).read_bytes()).hexdigest();files[source]=sha
        rows.append({'block':name,'sha256':digest,'retained_source':{'file':source,'sha256':sha},'actual_source':{'contract':'pops.model.actual-compile@1','complete':True,'status':'available','source_sha256':sha,'binary_sha256':digest,'header_signature':'H'}})
    proof={'schema':'pops.m16-explicit-retained-provenance@2','native':{'sha256':'N'},'root_scientific_approval':False,'layout_program':{'target':'amr_system','blocks':list(r.BLOCKS)},'model_binaries':rows,'files':files,'program':{'sha256':exports['program'][1],'abi_key':'headers=H;'}}
    if mutation=='contract':rows[0]['actual_source']['contract']='foreign@1'
    elif mutation=='header':rows[0]['actual_source']['header_signature']='foreign'
    elif mutation=='foreign-binary':exports['Q0']=(exports['Q1'][0],exports['Q1'][1])
    elif mutation=='missing-source':files.pop(rows[0]['retained_source']['file'])
    (tmp_path/'provenance.json').write_text(json.dumps(proof))
    pins={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    if mutation is None:
        result=r.join_provenance(tmp_path,pins,exports,native_sha256='N',header_signature='H')
        assert result['compiler_to_dso_graph_proof'] is False
    else:
        with pytest.raises(ValueError):r.join_provenance(tmp_path,pins,exports,native_sha256='N',header_signature='H')
