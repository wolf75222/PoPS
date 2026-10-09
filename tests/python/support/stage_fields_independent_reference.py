"""Independent mathematical reader for one frozen composition; no PoPS/author imports."""
from dataclasses import dataclass, replace
from fractions import Fraction
import numpy as np

@dataclass(frozen=True)
class Physics:
    cells: tuple=(16,12)
    lengths: tuple=(2.,3.)
    dt: float=1e-4
    diffusion: float=.15
    decay: float=.8
    phi_weight: float=.2
    psi_weight: float=.4
    donor_psi: float=-.75
    screenings: tuple=(3.,5.)
    q_mean: float=2.
    d_mean: float=.6
    q_amplitude: float=.125
    d_amplitude: float=.07

CG_MAX_ITER=2000
CG_RTOL=1e-12
CG_ATOL=1e-14
# Frozen before Native: scope includes density/integral and common normalized/unscaled residual norms.
RHS_ABSOLUTE_CEILING=4.

def cell_means(p=Physics()):
    nx,ny=p.cells
    x=2*np.pi*np.arange(nx+1)/nx;y=2*np.pi*np.arange(ny+1)/ny
    mx=np.diff(np.sin(x))/(2*np.pi/nx);my=np.diff(np.sin(y))/(2*np.pi/ny)
    mode=my[:,None]*mx[None,:]
    return (p.q_mean+p.q_amplitude*mode)[None].copy(),(p.d_mean+p.d_amplitude*mode)[None].copy()

def eigenvalues(p=Physics()):
    nx,ny=p.cells;hx,hy=p.lengths[0]/nx,p.lengths[1]/ny
    return 4*np.sin(np.pi*np.arange(nx)/nx)**2/hx**2+4*np.sin(np.pi*np.arange(ny)/ny)[:,None]**2/hy**2

def laplacian_scalar(u,p=Physics()):
    nx,ny=p.cells;hx,hy=p.lengths[0]/nx,p.lengths[1]/ny
    out=np.empty_like(u)
    for y in range(ny):
        for x in range(nx):
            out[0,y,x]=(u[0,y,(x-1)%nx]-2*u[0,y,x]+u[0,y,(x+1)%nx])/hx**2+(u[0,(y-1)%ny,x]-2*u[0,y,x]+u[0,(y+1)%ny,x])/hy**2
    return out

def screened(rhs,sigma,p=Physics()):
    return np.fft.ifftn(np.fft.fftn(rhs,axes=(-2,-1))/(sigma+eigenvalues(p)),axes=(-2,-1)).real

def fields_and_rate(q,d,p=Physics()):
    rhs0=q+d;rhs1=q+p.donor_psi*d
    assert max(float(np.max(np.abs(rhs0))),float(np.max(np.abs(rhs1))))<RHS_ABSOLUTE_CEILING
    phi=screened(rhs0,p.screenings[0],p);psi=screened(rhs1,p.screenings[1],p)
    rate=p.diffusion*laplacian_scalar(q,p)-p.decay*q+p.phi_weight*phi+p.psi_weight*psi
    return phi,psi,rate

def reference(p=Physics(),frozen_fields=False):
    q0,d=cell_means(p);f0=fields_and_rate(q0,d,p)
    def evaluate(q):
        if not frozen_fields:return fields_and_rate(q,d,p)
        return f0[0],f0[1],p.diffusion*laplacian_scalar(q,p)-p.decay*q+p.phi_weight*f0[0]+p.psi_weight*f0[1]
    q1=q0+p.dt*f0[2];f1=evaluate(q1)
    q2=.75*q0+.25*q1+.25*p.dt*f1[2];f2=evaluate(q2)
    qfinal=q0/3+(2/3)*q2+(2/3)*p.dt*f2[2]
    return dict(q0=q0,q1=q1,q2=q2,qfinal=qfinal,d_initial=d,d_final=d.copy(),phi0=f0[0],psi0=f0[1],phi1=f1[0],psi1=f1[1],phi2=f2[0],psi2=f2[1],resident_phi=f2[0].copy(),resident_psi=f2[1].copy())

def modal_reference(p=Physics()):
    # Independently evolve the mean and single product-cosine coefficients with a scalar affine RHS.
    nx,ny=p.cells;hx,hy=p.lengths[0]/nx,p.lengths[1]/ny
    k=4*np.sin(np.pi/nx)**2/hx**2+4*np.sin(np.pi/ny)**2/hy**2
    x=(np.arange(nx)+.5)/nx;y=(np.arange(ny)+.5)/ny
    mode=(np.sin(np.pi/nx)/(np.pi/nx))*(np.sin(np.pi/ny)/(np.pi/ny))*np.cos(2*np.pi*y)[:,None]*np.cos(2*np.pi*x)[None,:]
    evolved=[]
    for lam,q,d in [(0.,p.q_mean,p.d_mean),(k,p.q_amplitude,p.d_amplitude)]:
        a=-p.diffusion*lam-p.decay+p.phi_weight/(lam+p.screenings[0])+p.psi_weight/(lam+p.screenings[1]);b=p.phi_weight/(lam+p.screenings[0])+p.psi_weight*p.donor_psi/(lam+p.screenings[1])
        f=lambda value:a*value+b*d
        q1=q+p.dt*f(q);q2=.75*q+.25*q1+.25*p.dt*f(q1);qend=q/3+(2/3)*q2+(2/3)*p.dt*f(q2)
        evolved.append([q,q1,q2,qend,d])
    result={name:(evolved[0][i]+evolved[1][i]*mode)[None]for i,name in enumerate(['q0','q1','q2','qfinal','d_initial'])};result['d_final']=result['d_initial'].copy()
    for stage in range(3):
        for role,sigma,donor in [('phi',p.screenings[0],1.),('psi',p.screenings[1],p.donor_psi)]:
            result[role+str(stage)]=((evolved[0][stage]+donor*evolved[0][4])/sigma+(evolved[1][stage]+donor*evolved[1][4])*mode/(k+sigma))[None]
    result['resident_phi']=result['phi2'].copy();result['resident_psi']=result['psi2'].copy()
    return result

def tolerances(p=Physics()):
    nx,ny=p.cells;hx,hy=p.lengths[0]/nx,p.lengths[1]/ny;volume=hx*hy;unknowns=2*nx*ny;eps=np.finfo(float).eps
    # N*R*rtol covers the supported norm normalization; abs floor covers both density/integral rows.
    residual=unknowns*RHS_ABSOLUTE_CEILING*CG_RTOL+CG_ATOL/min(1.,volume)
    rounding=128*eps*RHS_ABSOLUTE_CEILING;seed=16*eps*RHS_ABSOLUTE_CEILING
    base_phi=residual/p.screenings[0]+rounding;base_psi=residual/p.screenings[1]+rounding
    lipschitz=p.diffusion*4*(1/hx**2+1/hy**2)+abs(p.decay)+abs(p.phi_weight)/p.screenings[0]+abs(p.psi_weight)/p.screenings[1]
    cross=abs(p.phi_weight)/p.screenings[0]+abs(p.psi_weight*p.donor_psi)/p.screenings[1]
    field_force=abs(p.phi_weight)*base_phi+abs(p.psi_weight)*base_psi+cross*seed
    e0=seed;e1=(1+p.dt*lipschitz)*e0+p.dt*field_force+rounding
    e2=.75*e0+.25*(1+p.dt*lipschitz)*e1+.25*p.dt*field_force+rounding
    ef=e0/3+(2/3)*(1+p.dt*lipschitz)*e2+(2/3)*p.dt*field_force+rounding
    bounds=dict(q0=e0,q1=e1,q2=e2,qfinal=ef,d_initial=seed)
    for j,e in enumerate([e0,e1,e2]):
        bounds['phi'+str(j)]=base_phi+(e+seed)/p.screenings[0];bounds['psi'+str(j)]=base_psi+(e+abs(p.donor_psi)*seed)/p.screenings[1]
    bounds['resident_phi']=bounds['phi2'];bounds['resident_psi']=bounds['psi2']
    residual_rounding=256*eps*(4/hx**2+4/hy**2+max(p.screenings))*RHS_ABSOLUTE_CEILING
    return dict(schema='stage-fields-pre-Native-budget@1',CG=dict(max_iter=CG_MAX_ITER,rtol=CG_RTOL,atol=CG_ATOL),joint_unknowns=unknowns,rhs_ceiling=RHS_ABSOLUTE_CEILING,physical_cell_volume=volume,joint_residual_bound=residual,independent_residual_bound=residual+residual_rounding,point_roundoff_allowance=rounding,absolute_bounds=bounds,donor_final='byte-exact equality to actual saved initial donor, not approximate oraclebits',premises=['Existing screened operator is periodic positive SPD/M-matrix with sigma_min>=3.','CG success/residual controls and max2000 unchanged; bounded RHS<=4, zero/valid prior-field initial guess and supported density/integral/norm normalization.','If actual solver/residual metric contradicts these premises, report mismatch; never increase these budgets from observed errors.'])

def check_arrays(arrays,p=Physics()):
    expected=reference(p);budget=tolerances(p);comparisons={}
    for key,bound in budget['absolute_bounds'].items():
        if key not in arrays:raise KeyError('actual saved output missing: '+key)
        value=np.asarray(arrays[key]);assert value.dtype==np.float64 and value.shape==expected[key].shape and np.isfinite(value).all(),key
        defect=float(np.max(np.abs(value-expected[key])));assert defect<=bound,(key,defect,bound);comparisons[key]=dict(defect=defect,bound=bound)
    assert arrays['d_final'].dtype==arrays['d_initial'].dtype and arrays['d_final'].shape==arrays['d_initial'].shape and arrays['d_final'].tobytes()==arrays['d_initial'].tobytes()
    for j in range(3):
        q=arrays['q'+str(j)];d=arrays['d_initial']
        for role,sigma,rhs in [('phi',p.screenings[0],q+d),('psi',p.screenings[1],q+p.donor_psi*d)]:
            value=arrays[role+str(j)];res=float(np.max(np.abs(-laplacian_scalar(value,p)+sigma*value-rhs)));assert res<=budget['independent_residual_bound'],(j,role,res)
    return comparisons
