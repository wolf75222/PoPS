"""Independent discrete Fourier and stage exchange reference; NumPy only."""
import numpy as np


def data(cells):
    nx,ny=cells
    x=(np.arange(nx)+.5)/nx;y=(np.arange(ny)+.5)/ny
    xx,yy=np.meshgrid(x,y,indexing='xy')
    a=np.stack((1+.2*np.sin(2*np.pi*xx),np.zeros_like(xx)))
    b=np.stack((2+.1*np.cos(2*np.pi*yy),np.full_like(xx,-3)))
    c=(1+.1*np.cos(2*np.pi*yy))[None,:,:]
    return a,b,c,np.cos(2*np.pi*xx)[None,:,:]


def stage_potential(forcing,cells,fraction,dt,*,different=False):
    nx,ny=cells
    mode=np.broadcast_to(np.cos(2*np.pi*(np.arange(nx)+.5)/nx),(ny,nx))
    eigenvalue=4*nx*nx*np.sin(np.pi/nx)**2
    # Joint equations use symmetric Reaction(2) exchange. The first-beta
    # consumer reads the second unknown, never its sibling's output.
    coefficient=(2*eigenvalue+6)/(eigenvalue*(eigenvalue+4)) if different else 1/eigenvalue
    return coefficient*(forcing[0]+fraction*dt*mode)


def step(states,cells,time,*,different=False,dt=1/64):
    del time # Stage source comes from the actual accepted forcing, not a time fit.
    a,b,c,forcing=states
    gain0=2+stage_potential(forcing,cells,1/4,dt,different=different)**2
    gain1=2+stage_potential(forcing,cells,3/4,dt,different=different)**2
    q0=gain0*c[0]*(b[0]-a[0])
    q1=gain1*c[0]*((b[0]-.5*dt*q0)-(a[0]+.5*dt*q0))
    nx,ny=cells
    mode=np.broadcast_to(np.cos(2*np.pi*(np.arange(nx)+.5)/nx),(ny,nx))
    return (np.stack((a[0]+dt*q1,a[1]+dt*gain1)),
            np.stack((b[0]-dt*q1,b[1]+dt*gain1)),c.copy(),forcing+dt*mode[None,:,:])
