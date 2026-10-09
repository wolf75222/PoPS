"""Independent saved-state stencil audit; no PoPS import or solver execution."""
import numpy as np

def residuals(fields,state):
    coarse=fields[0][0].copy();fine=fields[1][0].copy()
    ca,fa=(np.asarray(row[1],dtype=bool) for row in fields);n=coarse.shape[0]
    covered=~ca
    assert np.array_equal(fa,np.repeat(np.repeat(covered,2,0),2,1))
    for y,x in zip(*np.where(covered)):coarse[y,x]=np.sum(fine[2*y:2*y+2,2*x:2*x+2])/4
    def fine_value(y,x):
        if fa[y,x]:return fine[y,x]
        cy,cx=y//2,x//2
        dy,dx=(y+.5)/2-(cy+.5),(x+.5)/2-(cx+.5)
        weights=lambda t:(t*(t-1)/2,1-t*t,t*(t+1)/2)
        wy,wx=weights(dy),weights(dx)
        return sum(wy[j]*wx[i]*coarse[np.clip(cy+j-1,0,n-1),np.clip(cx+i-1,0,n-1)] for j in range(3) for i in range(3))
    output=[]
    for level,(phi,mask,m) in enumerate(zip((coarse,fine),(ca,fa),state)):
        size=n*2**level;r=np.full(phi.shape,np.nan)
        for y,x in zip(*np.where(mask)):
            lap=0.
            for axis in (0,1):
                for sign in (-1,1):
                    coordinate=[y,x];coordinate[axis]+=sign
                    if not 0<=coordinate[axis]<size:continue # exact zero Neumann flux
                    yy,xx=coordinate
                    if level==1:lap+=(fine_value(yy,xx)-phi[y,x])*size**2
                    elif not covered[yy,xx]:lap+=(coarse[yy,xx]-phi[y,x])*n**2
                    else:
                        # Coarse outgoing flux = average two actual fine face gradients.
                        total=0.
                        for tangent in (0,1):
                            near=[2*y+tangent,2*x+tangent]
                            near[axis]=2*(y if axis==0 else x)+(1 if sign>0 else 0)
                            far=near.copy();far[axis]+=sign
                            total+=(fine_value(*far)-fine_value(*near))*2*n
                        lap+=n*total/2
            r[y,x]=8*phi[y,x]-8*m[y,x]-lap
        output.append(r)
    return output

def saved(directory,phase):
    with np.load(directory/(phase+'-fields.npz')) as a:fields=[(a[f'{level}-0'].copy(),a[f'{level}-1'].copy()) for level in range(2)]
    with np.load(directory/(phase+'-valid.npz')) as a:state=[a[f'{level}-0'].reshape(2,*fields[level][0].shape)[1].copy() for level in range(2)]
    r=residuals(fields,state)
    return {'phase':phase,'phi_absolute': [float(np.max(np.abs(phi[mask]-m[mask]))) for (phi,mask),m in zip(fields,state)],'original_F_linf':[float(np.max(np.abs(v[fields[level][1]]))) for level,v in enumerate(r)],'forcing_linf':[float(np.max(np.abs(8*m[mask]))) for m,(_,mask) in zip(state,fields)]}
