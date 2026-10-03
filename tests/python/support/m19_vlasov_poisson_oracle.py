"""Independent FV/SSPRK2 reference, periodic x and finite no-flux v; no PoPS.

Axes (v,x). epsilon=1, background=-q, mean-zero gauge. No mean subtraction.
"""
import numpy as np


def initial():
    v=-1+(np.arange(4)+.5)*.5
    # Exact cell averages of 3/4*(1-v²), normalized quadrature mass one.
    g=.75-.75*v*v-.25/16
    return g[:,None]*np.array([1.125,1.0625,1.,.9375,.875,.9375,1.,1.0625])[None,:]


def electrostatics(f,charge=1.):
    f=np.asarray(f,dtype=np.float64)
    if f.ndim!=2 or not np.isfinite(f).all():
        raise ValueError('finite ranked distribution required')
    nv,nx=f.shape
    rho=(2/nv)*f.sum(axis=0); source=charge*(rho-1)
    if abs(source.sum())>64*np.finfo(float).eps*nx*max(1.,abs(source).max()):
        raise ValueError('periodic Poisson source is not neutral')
    a=np.zeros((nx+1,nx+1))
    for j in range(nx):
        a[j,j]+=2*nx*nx
        a[j,(j-1)%nx]-=nx*nx
        a[j,(j+1)%nx]-=nx*nx
    a[:nx,nx]=1; a[nx,:nx]=1
    phi=np.linalg.solve(a,np.r_[source,0.])[:nx]
    residual=nx*nx*(2*phi-np.roll(phi,1)-np.roll(phi,-1))-source
    electric=-.5*nx*(np.roll(phi,-1)-np.roll(phi,1))
    return rho,phi,electric,residual


def rhs(f,electric,charge=1.):
    nv,nx=f.shape; v=-1+(np.arange(nv)+.5)*(2/nv)
    fx=v[:,None]*np.where(v[:,None]>=0,f,np.roll(f,-1,axis=1))
    fv=np.zeros((nv+1,nx)); speed=charge*electric
    fv[1:nv]=speed*np.where(speed>=0,f[:-1],f[1:])
    return -nx*(fx-np.roll(fx,1,axis=1))-(nv/2)*(fv[1:]-fv[:-1])


def step(f,dt=1/128,charge=1.,*,force_sign=1.,stale_predictor=False):
    rho,phi,electric,residual=electrostatics(f,charge)
    predictor=f+dt*rhs(f,force_sign*electric,charge)
    r1,p1,e1,res1=electrostatics(predictor,charge)
    used=electric if stale_predictor else e1
    accepted=.5*f+.5*(predictor+dt*rhs(predictor,force_sign*used,charge))
    return accepted,dict(initial_rho=rho,initial_phi=phi,initial_electric=electric,
        predictor_rho=r1,predictor_phi=p1,predictor_electric=e1,
        initial_residual=residual,predictor_residual=res1)
