"""Offline independent uniform Helmholtz/FE/Ghost reference, no PoPS import."""
import numpy as np
from tests.review.sol61_amr_full_carrier_offline import decode

CONTRACT='accepted-initial-field-ghost-public@1'
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
