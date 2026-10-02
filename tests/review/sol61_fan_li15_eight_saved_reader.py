"""Independent Uniform Fan--Li15 eight-step math reader@1, never imports PoPS.

No Native/ROOT scientific authority is issued. Whole-grown storage is outside
this Uniform valid-image profile until an authentic public carrier API exists.
"""
import hashlib,json
from math import comb,factorial,sqrt
from pathlib import Path
import numpy as np
from tests.python.support import fan_li15_oracle as primary

CONTRACT='pops.fan-li15.uniform-eight-saved-math@1'
INDICES=primary.INDICES
N=16;DT=1e-4;STEPS=8
TOP=primary.TOP
LOW=tuple(k for k in range(15) if k not in TOP)

def require(ok,message):
    if not ok:raise ValueError(message)

def domain(raw):
    raw=np.asarray(raw)
    require(raw.shape==(15,) and np.all(np.isfinite(raw)),'finite canonical fifteen moments required')
    rho=raw[0];require(rho>0,'density is not positive')
    u,v=raw[1]/rho,raw[5]/rho
    theta=np.array([[raw[2]/rho-u*u,raw[6]/rho-u*v],[raw[6]/rho-u*v,raw[9]/rho-v*v]])
    require(np.all(np.isfinite(theta)) and theta[0,0]>0 and theta[1,1]>0 and np.linalg.det(theta)>0,'Theta SPD violated')
    return rho,(u,v),theta

def wick(theta,p,q,memo=None):
    """Isserlis pair contraction in centered Gaussian coordinates."""
    memo={} if memo is None else memo
    if p<0 or q<0:return 0.
    if p+q==0:return 1.
    if (p+q)%2:return 0.
    if (p,q) not in memo:
        memo[p,q]=((p-1)*theta[0,0]*wick(theta,p-2,q,memo)+q*theta[0,1]*wick(theta,p-1,q-1,memo)
                   if p else (q-1)*theta[1,1]*wick(theta,0,q-2,memo))
    return memo[p,q]

def gram_flux(raw,direction=(1.,0.)):
    """Reconstruct degree-four Gaussian polynomial through a full Gram solve.

    This route does not call production Hermite/closure builders, and does not
    use primary.generating_hermite to derive the flux.
    """
    rho,(u,v),theta=domain(raw);memo={}
    gram=np.array([[wick(theta,p+r,q+s,memo) for r,s in INDICES] for p,q in INDICES])
    centered=np.array([sum(comb(p,i)*comb(q,j)*(-u)**(p-i)*(-v)**(q-j)*raw[INDICES.index((i,j))]
                           for i in range(p+1) for j in range(q+1)) for p,q in INDICES])
    coeff=np.linalg.solve(gram,centered)
    def moment(p,q):
        return sum(comb(p,i)*comb(q,j)*u**(p-i)*v**(q-j)*
                   sum(c*wick(theta,i+r,j+s,memo) for c,(r,s) in zip(coeff,INDICES))
                   for i in range(p+1) for j in range(q+1))
    return np.array([direction[0]*moment(p+1,q)+direction[1]*moment(p,q+1) for p,q in INDICES])

def path(left,right,points=48):
    """Independent GL quadrature of the explicit straight RAW path."""
    nodes,weights=np.polynomial.legendre.leggauss(points)
    jump=right-left;result=np.zeros(15)
    for node,weight in zip((nodes+1)/2,weights/2):
        value=left+node*jump;domain(value)
        result+=weight*np.asarray(primary.primary_product(value,jump,(1.,0.)),dtype=float)
    require(np.all(result[list(LOW)]==0),'terminal-only nonconservative support differs')
    return result

def rhs(line,points=48,active=True):
    require(line.shape==(15,N),'reference line shape differs')
    for cell in line.T:domain(cell)
    flux=np.stack([gram_flux(cell) for cell in line.T],axis=1)
    speed=sqrt(6+sqrt(10))*np.sqrt(line[2]/line[0])
    right=np.roll(line,-1,axis=1)
    face=.5*(flux+np.roll(flux,-1,axis=1)-np.maximum(speed,np.roll(speed,-1))[None,:]*(right-line))
    integral=np.stack([path(line[:,k],right[:,k],points) for k in range(N)],axis=1) if active else np.zeros_like(line)
    return -N*(face-np.roll(face,1,axis=1))-.5*N*(integral+np.roll(integral,1,axis=1))

def step(line,points=48,active=True):
    k0=rhs(line,points,active);stage=line+DT*k0
    for cell in stage.T:domain(cell)
    result=line+.5*DT*(k0+rhs(stage,points,active))
    for cell in result.T:domain(cell)
    return result

def initial():
    x=(np.arange(N)+.5)/N;w=.55+.03*np.sinc(1/N)*np.cos(2*np.pi*x)
    a=np.array(primary.gaussian_mixture(((1.,(.15,-.1),(.8,0.,1.1)),)))
    b=np.array(primary.gaussian_mixture(((1.,(-.2,.15),(1.2,0.,.9)),)))
    return a[:,None]*w+b[:,None]*(1-w)

def audit_states(states,clocks,order):
    require(type(order) is tuple and len(order)==15 and all(type(a) is tuple and len(a)==2 and all(type(i) is int for i in a) for a in order) and set(order)==set(INDICES),'complete degree-four ordered basis required')
    require(len(states)==len(clocks)==9,'exact initial and eight accepted states required')
    canonical=[]
    for k,(array,clock) in enumerate(zip(states,clocks)):
        require(type(array) is np.ndarray and array.dtype==np.float64 and array.shape==(15,N,N) and np.all(np.isfinite(array)),'saved State representation differs')
        require(type(clock) is dict and set(clock)=={'time','tick'} and type(clock['tick']) is int and clock['tick']==k and type(clock['time']) is float and np.isfinite(clock['time']) and abs(clock['time']-k*DT)<1e-14,'saved time/tick differs')
        value=array[[order.index(index) for index in INDICES]]
        require(np.max(abs(value-value[:,0:1,:]))<1e-12,'y extrusion differs')
        line=value[:,0,:]
        for cell in line.T:domain(cell)
        canonical.append(line)
    seed=initial();require(np.max(abs(canonical[0]-seed))<=1e-13,'analytic cell averages differ')
    h=[primary.generating_hermite(cell)[0] for cell in seed.T]
    h3=max(abs(v) for row in h for index,v in row.items() if sum(index)==3)
    h4=max(abs(v) for row in h for index,v in row.items() if sum(index)==4)
    require(h3>1e-8 and h4>1e-8,'non-Gaussian degree-three/four witness inactive')
    inventory=canonical[0][list(LOW)].mean(axis=1);metrics=[]
    for k in range(1,9):
        previous=canonical[k-1];expected=step(previous,48);coarse=step(previous,24)
        gap=np.max(abs(expected-coarse));error=np.max(abs(canonical[k]-expected))
        require(gap<3e-8 and error<3e-8,'independent quadrature/SSPRK2 guard violated')
        require(np.max(abs(canonical[k][list(LOW)].mean(axis=1)-inventory))<3e-13,'ten conservative inventories differ')
        product=np.stack([path(previous[:,i],previous[:,(i+1)%N]) for i in range(N)],axis=1)
        product24=np.stack([path(previous[:,i],previous[:,(i+1)%N],24) for i in range(N)],axis=1)
        product4=np.stack([path(previous[:,i],previous[:,(i+1)%N],4) for i in range(N)],axis=1)
        require(np.max(abs(product-product24))<3e-8 and np.max(abs(product-product4))<3e-8,'saved-face quadrature guards violated')
        require(np.max(abs(product))>1e-8,'regularized five-row contribution inactive')
        metrics.append({'step':k,'ssprk2_error':float(error),'quadrature24_48_gap':float(gap),'regularized_face_max':float(np.max(abs(product)))})
    final=canonical[-1];right=np.roll(final,-1,axis=1)
    final48=np.stack([path(final[:,i],right[:,i],48) for i in range(N)],axis=1)
    final4=np.stack([path(final[:,i],right[:,i],4) for i in range(N)],axis=1)
    require(np.max(abs(final48-final4))<3e-8,'final saved-face Gauss4/48 guard violated')
    from scipy.integrate import solve_ivp
    reference=solve_ivp(lambda t,u:rhs(u.reshape(15,N),24).ravel(),(0.,STEPS*DT),seed.ravel(),method='DOP853',rtol=2e-12,atol=2e-14)
    require(reference.success and np.all(np.isfinite(reference.y[:,-1])),'independent DOP853 failed')
    continuous_error=float(np.max(abs(canonical[-1]-reference.y[:,-1].reshape(15,N))))
    require(continuous_error<3e-8,'original semidiscrete time-reference guard violated')
    return {'contract':CONTRACT,'dop853_final_error':continuous_error,'metrics':metrics,'h3_max':float(h3),'h4_max':float(h4),'native_authority':False,'root_scientific_approval':False,'whole_carrier_qualification':False}

def receive(directory,pins,order,phase_files):
    """External pins authenticate originals; paths are explicit, not inferred."""
    directory=Path(directory)
    require(type(pins) is dict and pins and type(phase_files) is tuple and len(phase_files)==9,'external pins/nine phase inventory required')
    for name,pin in pins.items():
        p=directory/name
        require(not p.is_symlink() and p.resolve().is_relative_to(directory.resolve()),'foreign export path')
        require(hashlib.sha256(p.read_bytes()).hexdigest()==pin,'external export pin differs')
    states=[];clocks=[];seen=set()
    for state_name,clock_name in phase_files:
        require(state_name in pins and clock_name in pins and state_name not in seen and clock_name not in seen,'omitted or duplicate phase file')
        seen.update((state_name,clock_name));states.append(np.load(directory/state_name,allow_pickle=False))
        def pairs(items):
            d={}
            for key,value in items:
                require(key not in d,'duplicate clock key');d[key]=value
            return d
        clock=json.loads((directory/clock_name).read_text(),object_pairs_hook=pairs,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite clock')))
        require(type(clock) is list and len(clock)==2 and type(clock[0]) is float and type(clock[1]) is int,'Uniform clock wire must be exact [time,tick]')
        clocks.append({'time':clock[0],'tick':clock[1]})
    return audit_states(states,clocks,order)


def compare_permutation(left,left_order,right,right_order):
    """Compare already independently audited and externally pinned sequences."""
    require(len(left)==len(right)==9,'permutation requires nine states on both sides')
    require(set(left_order)==set(right_order)==set(INDICES) and len(left_order)==len(right_order)==15,'permutation basis differs')
    gaps=[]
    for a,b in zip(left,right):
        require(a.shape==b.shape==(15,N,N),'permutation shapes differ')
        gap=float(np.max(abs(a[[left_order.index(i) for i in INDICES]]-b[[right_order.index(i) for i in INDICES]])))
        require(np.isfinite(gap) and gap<1e-12,'original permutation guard violated')
        gaps.append(gap)
    return gaps

def audit_rejected_valid_rollback(before,after,before_clock,after_clock,failure):
    """Distinct failure-domain@1: valid-state rollback, not full storage rollback."""
    require(type(failure) is dict and failure.get('phase')=='attempt' and
            type(failure.get('exception_type')) is str and failure['exception_type'] and
            type(failure.get('message')) is str and failure['message'],
            'genuine attempted failure metadata required')
    require(type(before) is np.ndarray and type(after) is np.ndarray and
            before.dtype==after.dtype==np.float64 and before.shape==after.shape==(15,N,N),
            'rollback State representation differs')
    require(before.tobytes()==after.tobytes() and before_clock==after_clock,
            'rejected-attempt valid State/clock rollback differs')
    return {'contract':'pops.fan-li15.uniform-rejected-valid-rollback@1',
            'native_authority':False,'full_storage_rollback_qualified':False}
