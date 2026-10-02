"""Independent offline finite-volume oracle, not production cell callbacks.

Storage is (component,y,x); cell averages use the analytic sinus integral.
Flux is derived from the six explicit atoms by independent fixed equations,
not AtomicCubature/Vandermonde or the emitted arithmetic graph.
"""
import numpy as np

NX,NY,LX,LY,DT=8,4,2.,1.,1.e-4
INDICES=((1,1),(0,0),(0,2),(1,0),(2,0),(0,1))
ATOMS=np.array(((0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1)),dtype=np.float64)


def initial_averages():
    x=(np.arange(NX)+.5)/NX; y=(np.arange(NY)+.5)/NY
    weights=np.empty((6,NY,NX),dtype=np.float64)
    for k in range(6):
        weights[k]=(.6+.05*k + .03*np.sinc(1/NX)*np.sin(2*np.pi*x+k/7)[None,:]
                    + .02*np.sinc(1/NY)*np.cos(2*np.pi*y-k/5)[:,None])
    assert np.min(weights)>0
    return np.ascontiguousarray(np.stack([sum(weights[k]*ATOMS[k,0]**p*ATOMS[k,1]**q
                                                for k in range(6)) for p,q in INDICES]))


def directional_flux(values,axis):
    cross,rho,yy,x,xx,y=values
    if axis==0:
        return np.stack((cross,x,cross,xx,x,cross))
    if axis==1:
        return np.stack((cross,y,y,cross,cross,yy))
    raise ValueError('invalid axis')


def face_tuple(left,right,axis,nonconservative):
    factor=1. if axis==0 else -.5
    strength=float(nonconservative)
    # |g_x|+|g_y|=1 at each positive Cartesian face.
    speed=1.+strength*abs(factor)*np.maximum(left[1],right[1])
    flux=(directional_flux(left,axis)+directional_flux(right,axis))/2-speed[None,:]*(right-left)/2
    integral=strength*factor*(left[1]+right[1])[None,:]*(right-left)/2
    return flux,-integral/2


def forward_euler(values,*,nonconservative):
    values=np.asarray(values,dtype=np.float64)
    if values.shape!=(6,NY,NX) or not np.isfinite(values).all() or not np.all(values[1]>0):
        raise ValueError('invalid FV image')
    residual=np.zeros_like(values)
    for physical_axis,array_axis,spacing in ((0,2,LX/NX),(1,1,LY/NY)):
        right=np.roll(values,-1,axis=array_axis)
        flux,ncp=face_tuple(values,right,physical_axis,nonconservative)
        residual+=(-flux+np.roll(flux,1,axis=array_axis)+ncp+np.roll(ncp,1,axis=array_axis))/spacing
    return values+DT*residual


def authenticate_carrier_values(blob,values):
    """Cross-check complete native POPSCAR1 valid storage; ghosts are retained."""
    from tests.review.sol61_amr_full_carrier_offline import decode
    image=decode(np.frombuffer(blob,dtype=np.uint8))
    if image['dim']!=2 or image['real']!=64 or image['levels']!=1 or image['shard']!=-1 or len(image['blocks'])!=1:
        raise ValueError('carrier authority differs from single-level fixture')
    values=np.asarray(values)
    if values.dtype!=np.float64 or values.size!=6*NY*NX:
        raise ValueError('carrier gather width/precision differs')
    values=values.reshape(6,NY,NX)
    covered=np.zeros((NY,NX),dtype=bool)
    for patch in image['patches']:
        if patch['key'][:2]!=(0,0) or patch['components']!=6:
            raise ValueError('foreign carrier row')
        (xl,xh,gxl,gxh),(yl,yh,gyl,gyh)=patch['axes']
        if not (0<=xl<=xh<NX and 0<=yl<=yh<NY) or covered[yl:yh+1,xl:xh+1].any():
            raise ValueError('carrier geometry/overlap differs')
        full=np.asarray(patch['bits'],dtype=np.uint64).view(np.float64).reshape(6,gyh-gyl+1,gxh-gxl+1)
        valid=full[:,yl-gyl:yh-gyl+1,xl-gxl:xh-gxl+1]
        expected=values[:,yl:yh+1,xl:xh+1]
        if valid.tobytes()!=expected.tobytes(): raise ValueError('carrier valid bits differ from saved gather')
        covered[yl:yh+1,xl:xh+1]=True
    if not covered.all():raise ValueError('carrier coverage incomplete')
