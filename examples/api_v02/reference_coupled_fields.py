"""Independent finite-grid NumPy reception oracles, NEVER the PoPS execution path.

These use FFT inverses and dense constrained linear solves to check results from
native C++/MPI kernels using CG and source-authored face elimination respectively.
"""
import numpy as np


def elliptic(rhs,lengths,*,reaction=0.,diffusion=1.):
    ny,nx=rhs.shape
    symbol=4*((np.sin(np.pi*np.arange(nx)/nx)/(lengths[0]/nx))[None,:]**2
              +(np.sin(np.pi*np.arange(ny)/ny)/(lengths[1]/ny))[:,None]**2)
    denominator=reaction+diffusion*symbol
    transformed=np.fft.fft2(rhs)
    if reaction==0:
        if abs(float(np.mean(rhs)))>1.e-12:
            raise ValueError("periodic Poisson reference requires compatible charge")
        denominator=denominator.copy();denominator[0,0]=1;transformed[0,0]=0
    return np.fft.ifft2(transformed/denominator).real


def ssprk2(initial,rhs,dt,steps=2):
    value=initial.copy()
    for _ in range(steps):
        first=value+dt*rhs(value)
        value=.5*(value+first+dt*rhs(first))
    return value


def maxwell_stefan_rhs(value,lengths):
    """4x4 constrained face solves, independent of the native 2x2 elimination."""
    masses=np.array((1.,2.,4.));charges=np.array((1.,-1.,0.))
    binary=np.array(((1.,1.,.8),(1.,1.,1.2),(.8,1.2,1.)))
    phi=elliptic(value[0]-value[1]/2,lengths)
    result=np.zeros_like(value)
    for axis in range(2):
        array_axis=-1-axis;h=lengths[axis]/value.shape[array_axis]
        left=np.roll(value,1,axis=array_axis);face=.5*(value+left)
        concentration=face/masses[:,None,None];ct=np.sum(concentration,axis=0)
        x=concentration/ct
        xl=left/masses[:,None,None];xl/=xl.sum(axis=0)
        xr=value/masses[:,None,None];xr=xr/xr.sum(axis=0)
        gradient=(phi-np.roll(phi,1,axis=array_axis))/h
        drive=(xr-xl)/h+(charges[:,None,None]*x-np.sum(charges[:,None,None]*x,axis=0)*face)*gradient
        matrices=np.zeros((*phi.shape,4,4))
        for i in range(3):
            for j in range(3):
                if i==j:continue
                k=1/(ct**3*masses[i]*masses[j]*binary[i,j])
                matrices[...,i,j]=-k*face[i]
                matrices[...,i,i]+=k*face[j]
        matrices[...,:3,3]=1;matrices[...,3,:3]=1
        right=np.zeros((*phi.shape,4));right[...,:3]=-np.moveaxis(drive,0,-1)
        solved=np.linalg.solve(matrices,right[...,None])[...,0]
        flux=np.moveaxis(solved[...,:3],-1,0)
        result+=(flux-np.roll(flux,-1,axis=array_axis))/h
    return result


def streamer_rhs(value,lengths,*,mu=.2,diffusion=.02,background=.3,
                 impact=.08,photo=.03,absorption=.4):
    phi=elliptic(value[1]-value[0],lengths)
    radiation=elliptic(photo*value[0],lengths,reaction=1.,diffusion=absorption**2)
    production=impact*value[0]+radiation
    result=np.stack((production,production,-production))
    for axis in range(2):
        array_axis=-1-axis;h=lengths[axis]/value.shape[array_axis]
        gradient=(phi-np.roll(phi,1,axis=array_axis))/h
        velocity=mu*(gradient-(background if axis==0 else 0.))
        left=np.roll(value[0],1,axis=array_axis)
        flux=velocity*np.where(velocity>=0,left,value[0])-diffusion*(value[0]-left)/h
        result[0]+=(flux-np.roll(flux,-1,axis=array_axis))/h
    return result
