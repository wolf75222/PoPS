"""Offline independent uniform Helmholtz/FE/Ghost reference, no PoPS import."""
import numpy as np
from tests.review.sol61_amr_full_carrier_offline import decode

CONTRACT='accepted-initial-field-ghost-public@4'
DT=1/64
ORIGINAL_F_BOUND=1e-10
FIELD_BOUND=1e-10  # fixed pre-execution uniform Helmholtz absolute error guard

def check(blob,fields,steps):
    image=decode(np.frombuffer(blob,dtype=np.uint8))
    assert image['dim']==2 and image['real']==64 and image['levels']==2
    expected=2*(1+DT)**steps
    physical=[]
    state=[np.full(np.asarray(row[0]).shape,np.nan) for row in fields]
    for patch in image['patches']:
        assert patch['components']==2
        (xlo,xhi,gxlo,gxhi),(ylo,yhi,gylo,gyhi)=patch['axes']
        values=np.asarray(patch['bits'],dtype=np.uint64).view(np.float64).reshape(2,gyhi-gylo+1,gxhi-gxlo+1)
        valid=values[1,ylo-gylo:yhi-gylo+1,xlo-gxlo:xhi-gxlo+1]
        level=patch['key'][1]
        state[level][ylo:yhi+1,xlo:xhi+1]=valid
        # FE multiplication/addition and independent factor construction: 4ops per step.
        eps=np.finfo(np.float64).eps;k=4*steps+2;bound=k*eps/(1-k*eps)*max(1,expected)
        assert np.max(np.abs(valid-expected))<=bound
        if xlo==0:
            ghost=values[1,ylo-gylo:yhi-gylo+1,-1-gxlo]
            target=expected+1+steps*DT
            error=float(np.max(np.abs(ghost-target)))
            assert error<=FIELD_BOUND+bound
            # Explicitly reject unchanged state / stale-initial Field / omitted time.
            assert np.min(np.abs(ghost-expected))>.5
            if steps:
                assert np.min(np.abs(ghost-(3+steps*DT)))>DT
                assert np.min(np.abs(ghost-(expected+1)))>DT/2
            physical.append({'key':patch['key'],'count':ghost.size,'max_error':error})
    assert physical
    assert len(fields)==2
    for field,mask in fields:
        a=np.asarray(field);mask=np.asarray(mask,dtype=bool)
        assert a.shape==mask.shape and mask.any()
        assert np.isfinite(a[mask]).all() and np.max(np.abs(a[mask]-expected))<=FIELD_BOUND
    residual=original_residual(fields,state)
    return {'original_F_linf_by_level':residual,'original_F_guard':ORIGINAL_F_BOUND,'contract':CONTRACT,'steps':steps,'expected_m_phi':expected,
            'expected_xmin_m_ghost':expected+1+steps*DT,'field_absolute_guard':FIELD_BOUND,
            'physical_faces':physical}


def original_residual(fields,state):
    """Original -Delta(phi)+8phi-8m on ratio2 composite Cartesian support.

    Neumann physical face flux is zero; covered coarse potential is restricted.
    Tensor quadratic parent interpolation fills missing fine samples; coarse
    interface flux is the mean of its two fine transverse faces (reflux).
    """
    coarse,fine=(np.asarray(row[0],dtype=float).copy() for row in fields)
    ca,fa=(np.asarray(row[1],dtype=bool) for row in fields)
    n=coarse.shape[0]
    assert coarse.shape==ca.shape==(n,n) and fine.shape==fa.shape==(2*n,2*n)
    covered=~ca
    assert np.array_equal(fa,np.repeat(np.repeat(covered,2,axis=0),2,axis=1))
    coarse[covered]=fine.reshape(n,2,n,2).mean(axis=(1,3))[covered]
    def cv(y,x):return coarse[min(n-1,max(0,y)),min(n-1,max(0,x))]
    def fv(y,x):
        y=min(2*n-1,max(0,y));x=min(2*n-1,max(0,x))
        if fa[y,x]:return fine[y,x]
        py,px=y//2,x//2
        sy,sx=(-.25 if y%2==0 else .25),(-.25 if x%2==0 else .25)
        wy=(sy*(sy-1)/2,1-sy*sy,sy*(sy+1)/2)
        wx=(sx*(sx-1)/2,1-sx*sx,sx*(sx+1)/2)
        return sum(wy[j]*wx[i]*cv(py+j-1,px+i-1) for j in range(3) for i in range(3))
    def fluxes(values,size,get):
        fx=np.zeros((size,size+1));fy=np.zeros((size+1,size))
        for y in range(size):
            for x in range(1,size):fx[y,x]=(get(y,x)-get(y,x-1))*size
        for y in range(1,size):
            for x in range(size):fy[y,x]=(get(y,x)-get(y-1,x))*size
        return fx,fy
    cfx,cfy=fluxes(coarse,n,cv);ffx,ffy=fluxes(fine,2*n,fv)
    for y in range(n):
        for x in range(1,n):
            if covered[y,x]!=covered[y,x-1]:cfx[y,x]=ffx[2*y:2*y+2,2*x].mean()
    for y in range(1,n):
        for x in range(n):
            if covered[y,x]!=covered[y-1,x]:cfy[y,x]=ffy[2*y,2*x:2*x+2].mean()
    out=[]
    for phi,m,mask,fx,fy,size in zip((coarse,fine),state,(ca,fa),(cfx,ffx),(cfy,ffy),(n,2*n),strict=True):
        lap=(fx[:,1:]-fx[:,:-1]+fy[1:,:]-fy[:-1,:])*size
        residual=-lap+8*phi-8*np.asarray(m)
        assert np.isfinite(residual[mask]).all()
        out.append(float(np.max(np.abs(residual[mask]))))
    assert max(out)<=ORIGINAL_F_BOUND, out
    return out


def check_observed(blob, masks, rows_by_rank, steps, *, expected_clock=None):
    """Full producer image consumed at Ghost point; cache getters are not inputs.

    Rows are real observer output in Native tests, synthetic only in offline unit tests.
    Every consumer invocation is checked independently across all source ranks.
    """
    if expected_clock is not None: assert type(expected_clock) is str and expected_clock
    state_image=decode(np.frombuffer(blob,dtype=np.uint8))
    ranks=state_image['ranks']
    assert type(rows_by_rank) is list and len(rows_by_rank)==ranks
    keyed=[]
    for rank,rows in enumerate(rows_by_rank):
        assert type(rows) is list and len(rows)==2
        bykey={}
        for row in rows:
            assert type(row) is dict and row['schema']=='pops.amr.field-candidate-observation@1'
            assert row['status']=='producer-completed-consumer-preparation-completed' and row['accepted_publication'] is False
            for key in ('consumer_level','owner_macro_step','topology_epoch','materialization_generation'):
                assert type(row[key]) is int and row[key]>=0
            assert type(row['consumer_block']) is str and row['consumer_block']==state_image['blocks'][0] and row['consumer_level'] in (0,1)
            assert type(row['owner_time']) is float and row['owner_time']==steps*DT
            assert row['owner_macro_step']==steps
            for key in ('provider_slot','configuration_identity','provider_identity','plan_identity','output_owner_identity','output_block','output_key'):
                assert type(row[key]) is str and row[key]
            point=row['point']
            assert type(point) is dict and type(point['clock']) is str and point['clock']
            if expected_clock is not None: assert point['clock']==expected_clock
            for key in ('tick','level','substep','stage','fraction_numerator','fraction_denominator'):
                assert type(point[key]) is int and point[key]>=0
            assert point['level']==row['consumer_level'] and point['fraction_denominator']>0
            assert point['fraction_numerator']<=point['fraction_denominator']
            assert type(point['dt']) is float and point['dt']==(DT if steps else 0.) and not np.signbit(point['dt'])
            assert type(point['physical_time']) is float and point['physical_time']==steps*DT
            assert point['tick']==0 and point['substep']==0 and point['stage']==0
            assert point['fraction_numerator']==steps and point['fraction_denominator']==1
            for key in ('graph_identity','rate_identity','application_identity'):
                assert type(point[key]) is str and point[key]==''
            key=(row['provider_slot'],row['consumer_block'],row['consumer_level'])
            assert key not in bykey
            image=decode(np.frombuffer(row['carrier_bytes'],dtype=np.uint8))
            assert image['dim']==2 and image['real']==64 and image['levels']==2 and image['ranks']==ranks and image['shard']==rank and image['blocks']==[row['provider_slot']]
            bykey[key]=(row,image)
        keyed.append(bykey)
    assert all(set(row)==set(keyed[0]) for row in keyed)
    results=[]
    for key in sorted(keyed[0]):
        patches={}
        reference={k:v for k,v in keyed[0][key][0].items() if k!='carrier_bytes'}
        for rank in range(ranks):
            row,image=keyed[rank][key]
            assert {k:v for k,v in row.items() if k!='carrier_bytes'}==reference
            for patch in image['patches']:
                assert patch['components']==1 and patch['owner'] in (-1,rank)
                if patch['key'] in patches: assert patches[patch['key']]==patch
                patches[patch['key']]=patch
        expected={p['key']:p for p in state_image['patches']}
        assert set(patches)==set(expected)
        fields=[(np.full(mask.shape,np.nan),mask.copy()) for mask in masks]
        for pkey,patch in patches.items():
            assert [a[:2] for a in patch['axes']]==[a[:2] for a in expected[pkey]['axes']] and patch['owner']==expected[pkey]['owner']
            (xl,xh,gxl,gxh),(yl,yh,gyl,gyh)=patch['axes']
            values=np.asarray(patch['bits'],dtype=np.uint64).view(np.float64).reshape(gyh-gyl+1,gxh-gxl+1)
            fields[pkey[1]][0][yl:yh+1,xl:xh+1]=values[yl-gyl:yh-gyl+1,xl-gxl:xh-gxl+1]
            if xl==0:
                # Actual dependency packing reads x=-1 and tangential-valid y;
                # no claim for corners or unconsumed Field halos.
                assert gxl<=-1
                producer=values[yl-gyl:yh-gyl+1,-1-gxl]
                state_patch=expected[pkey]
                (_,_,sgxl,sgxh),(_,_,sgyl,sgyh)=state_patch['axes']
                assert sgxl<=-1
                state_values=np.asarray(state_patch['bits'],dtype=np.uint64).view(np.float64).reshape(2,sgyh-sgyl+1,sgxh-sgxl+1)
                consumed=state_values[1,yl-sgyl:yh-sgyl+1,-1-sgxl]
                reconstructed=producer+1+steps*DT
                assert np.isfinite(producer).all() and np.isfinite(consumed).all()
                # Two physical additions plus three comparison/scale operations.
                eps=np.finfo(np.float64).eps;gamma=5*eps/(1-5*eps)
                scale=np.maximum(1,np.abs(producer)+1+steps*DT)
                assert np.all(np.abs(consumed-reconstructed)<=gamma*scale)
        results.append({'consumer_level':key[2],'point':reference['point'],'math':check(blob,fields,steps)})
    return {'scope':'consumed-full-candidate; retained-cache excluded','invocations':results}


def load_observed(directory, phase, ranks):
    """Read persisted rank-local evidence; immutable exact hash check, no Native call."""
    import hashlib
    import json
    from pathlib import Path
    directory=Path(directory).resolve()
    result=[]
    assert type(ranks) is int and ranks>0 and phase in ('initial','accepted')
    for rank in range(ranks):
        document=json.loads((directory/f'{phase}-field-candidate-rank{rank}.json').read_text())
        assert document['schema']=='pops.amr.field-candidate-observation@1' and type(document['rank']) is int and document['rank']==rank
        rows=[]
        for stored in document['observations']:
            row=dict(stored);carrier=row.pop('carrier');path=Path(carrier['path']).resolve()
            assert path.parent==directory and path.name.startswith(f'{phase}-field-candidate-rank{rank}-') and path.suffix=='.bin'
            blob=path.read_bytes()
            assert type(carrier['size_bytes']) is int and len(blob)==carrier['size_bytes']
            assert hashlib.sha256(blob).hexdigest()==carrier['sha256']
            row['carrier_bytes']=blob;rows.append(row)
        result.append(rows)
    return result


def checkpoint_primary_clock(path):
    """Primary clock from retained CP temporal schedule, never observation self-pin."""
    import json
    with np.load(path,allow_pickle=False) as archive:
        temporal=json.loads(str(archive['temporal_restart_state']))
    clock=temporal['program_schedule']['primary_clock']
    assert type(clock) is str and clock
    return clock
