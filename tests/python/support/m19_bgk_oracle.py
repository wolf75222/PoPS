"""Independent NumPy self-BGK reference, arrays (velocity, position).

No PoPS/provider/compiled model, equilibrium fit or moment renormalization.
"""
import numpy as np

def velocity(nv):
    return -8+(np.arange(nv,dtype=np.float64)+.5)*(16/nv)

def initial(nx=32,nv=32):
    v=velocity(nv)[:,None];x=(np.arange(nx)+.5)/nx
    n=1+.1*np.cos(2*np.pi*x)
    return n*(np.exp(-.5*((v-.7)/.75)**2)+np.exp(-.5*((v+.7)/.75)**2))/(2*.75*np.sqrt(2*np.pi))

def moments(f):
    f=np.asarray(f,dtype=np.float64)
    if f.ndim!=2 or not np.isfinite(f).all():raise ValueError('finite ranked distribution required')
    v=velocity(f.shape[0])
    return tuple(np.einsum('vx,v->x',f,(16/f.shape[0])*v**k) for k in range(3))

def equilibrium(f):
    n,p,e=moments(f)
    if np.any(n<=0):raise ValueError('BGK requires positive number and thermal variance')
    u=p/n;variance=e/n-u*u
    if not np.isfinite(variance).all() or np.any(n<=0) or np.any(variance<=0):
        raise ValueError('BGK requires positive number and thermal variance')
    return n/np.sqrt(2*np.pi*variance)*np.exp(-(velocity(f.shape[0])[:,None]-u)**2/(2*variance))

def step(f,dt=1/128,nu=1.5):
    predictor=f+dt*nu*(equilibrium(f)-f)
    accepted=.5*f+.5*(predictor+dt*nu*(equilibrium(predictor)-predictor))
    return accepted,dict(predictor=predictor,moments=moments(predictor))
