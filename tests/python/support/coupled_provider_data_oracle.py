"""Independent static-provider exchange reference; no PoPS/native imports."""
import numpy as np

def data(cells=(7,3)):
    nx,ny=cells;x=(np.arange(nx)+.5)/nx;y=(np.arange(ny)+.5)/ny
    xx,yy=np.meshgrid(x,y,indexing='xy')
    gain=2+(1+xx+2*yy)**2
    a=np.stack((1+.2*np.sin(2*np.pi*xx),np.zeros_like(xx)))
    b=np.stack((2+.1*np.cos(2*np.pi*yy),np.full_like(xx,-3)))
    c=(1+.1*np.cos(2*np.pi*yy))[None,:,:]
    return (a,b,c),gain

def step(states,gain,dt=1/64):
    a,b,c=states
    q0=gain*c[0]*(b[0]-a[0])
    predictor_a=a[0]+.5*dt*q0;predictor_b=b[0]-.5*dt*q0
    q1=gain*c[0]*(predictor_b-predictor_a)
    return (np.stack((a[0]+dt*q1,a[1]+dt*gain)),
            np.stack((b[0]-dt*q1,b[1]+dt*gain)),c.copy())
