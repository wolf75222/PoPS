"""M17 saved-state physics @1; independent combinatorial oracle, no PoPS import."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
I=tuple((p,q) for q in range(5) for p in range(5-q)); SLOT={a:k for k,a in enumerate(I)}
DT=1e-4; N=16

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    def pairs(rows):
        out={}
        for k,v in rows:
            require(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def primitives(raw):
    r=raw[0]; u=raw[SLOT[1,0]]/r;v=raw[SLOT[0,1]]/r
    a=raw[SLOT[2,0]]/r-u*u;b=raw[SLOT[1,1]]/r-u*v;c=raw[SLOT[0,2]]/r-v*v
    require(np.isfinite(raw).all() and r>0 and a>0 and c>0 and a*c-b*b>0,'rho/SPD admission failed')
    return r,u,v,a,b,c

def exp_coefficient(p,q,u,v,a,b,c):
    # Closed coefficient of exp(-u*z-v*w-a*z²/2-b*z*w-c*w²/2).
    total=0.
    for zz in range(p//2+1):
        for ww in range(q//2+1):
            for zw in range(min(p-2*zz,q-2*ww)+1):
                z=p-2*zz-zw;w=q-2*ww-zw
                total+=(-u)**z*(-v)**w*(-a/2)**zz*(-b)**zw*(-c/2)**ww/(math.factorial(z)*math.factorial(w)*math.factorial(zz)*math.factorial(zw)*math.factorial(ww))
    return total

def coefficients(raw):
    r,u,v,a,b,c=primitives(raw)
    h={}
    for p,q in I:
        h[p,q]=sum(raw[SLOT[i,j]]*exp_coefficient(p-i,q-j,u,v,a,b,c)/(math.factorial(i)*math.factorial(j)) for j in range(q+1) for i in range(p+1))
    return h,(r,u,v,a,b,c)

def gaussian(p,q,u,v,a,b,c):
    # Direct Wick pair counting of shifted correlated Gaussian moments.
    return math.factorial(p)*math.factorial(q)*sum(u**(p-2*i-k)*v**(q-2*j-k)*(a/2)**i*b**k*(c/2)**j/(math.factorial(p-2*i-k)*math.factorial(q-2*j-k)*math.factorial(i)*math.factorial(j)*math.factorial(k)) for i in range(p//2+1) for j in range(q//2+1) for k in range(min(p-2*i,q-2*j)+1))

def flux(raw):
    h,(_,u,v,a,b,c)=coefficients(raw)
    out=[]
    for p,q in I:
        p+=1
        out.append(sum(h[i,j]*math.factorial(p)/math.factorial(p-i)*math.factorial(q)/math.factorial(q-j)*gaussian(p-i,q-j,u,v,a,b,c) for i,j in I if i<=p and j<=q))
    return np.array(out)

def product(raw,jump):
    h,(r,u,v,a,b,c)=coefficients(raw);dr=jump[0]
    du=((jump[SLOT[1,0]]-u*dr)/r,(jump[SLOT[0,1]]-v*dr)/r)
    da=(jump[SLOT[2,0]]-2*u*jump[SLOT[1,0]]+(u*u-a)*dr)/r
    db=(jump[SLOT[1,1]]-v*jump[SLOT[1,0]]-u*jump[SLOT[0,1]]+(u*v-b)*dr)/r
    dc=(jump[SLOT[0,2]]-2*v*jump[SLOT[0,1]]+(v*v-c)*dr)/r
    d=((da,db),(db,dc));out=np.zeros(15)
    for k,alpha in enumerate(I):
        if sum(alpha)!=4:continue
        for i in range(2):
            beta=(alpha[0]+1-(i==0),alpha[1]-(i==1))
            term=h.get(beta,0.)*du[i]
            for j in range(2):
                gamma=(beta[0]-(j==0),beta[1]-(j==1))
                term+=.5*h.get(gamma,0.)*d[i][j]
            out[k]-=math.factorial(alpha[0])*math.factorial(alpha[1])*(alpha[0]+1)*term
    return out

def rhs(line,points,regularized=True):
    right=np.roll(line,-1,axis=1);jump=right-line
    f=np.stack([flux(line[:,k]) for k in range(N)],axis=1)
    speed=np.sqrt(6+np.sqrt(10))*np.sqrt(line[SLOT[2,0]]/line[0])
    face=.5*(f+np.roll(f,-1,axis=1)-np.maximum(speed,np.roll(speed,-1))*jump)
    z,w=np.polynomial.legendre.leggauss(points);integral=np.zeros_like(line)
    for node,weight in zip((z+1)/2,w/2):
        for k in range(N):integral[:,k]+=weight*product(line[:,k]+node*jump[:,k],jump[:,k])
    return -N*(face-np.roll(face,1,axis=1))-(.5*N*(integral+np.roll(integral,1,axis=1)) if regularized else 0)
def step(line,points,regularized=True):
    first=rhs(line,points,regularized);return line+.5*DT*(first+rhs(line+DT*first,points,regularized))

def receive(directory):
    directory=Path(directory);receipt=load(directory/'receipt.json');author=load(directory/'authoring.json')
    require(receipt['schema']=='pops.fan-li15-public-composition-native-fixture@1' and author['schema']==receipt['schema'],'fixture schema differs')
    require(receipt['status']=='fixture-guards-passed','fixture did not close guards')
    require(type(receipt['ranks']) is int and receipt['ranks']==1,'reader@1 requires actual Serial')
    require(len(receipt['attempts'])==3 and all(row['failures']==[None] for row in receipt['attempts']),'Native attempt failed')
    require(receipt['program']['layout_program']['target']=='system' and receipt['program']['layout_program']['block_names']==['gas'],'layout authority differs')
    require(author['cells']==[N,N] and type(author['dt']) is float and author['dt']==DT and type(author['steps']) is int and author['steps']==2,'representative controls differ')
    require(author['path']=='normalized-analytic' and author['original_Gauss4_default_preserved'] is True,'authored method differs')
    for name,digest in receipt['files'].items():
        require(Path(name).name==name and sha(directory/name)==digest,'capture pin differs: '+name)
    order=tuple(tuple(x) for x in author['order']);require(len(order)==15 and all(len(x)==2 and all(type(i) is int for i in x) for x in order) and set(order)==set(I),'basis differs')
    for name in ('program.cpp','program.ir.json','compiled-manifest.json','initial.npy','accepted1.npy','accepted2.npy'):
        require(name in receipt['files'],'mandatory capture absent: '+name)
    phases=['initial','accepted1','accepted2'];require(receipt['phases']==phases,'phase list differs')
    images=[]
    for n,phase in enumerate(phases):
        image=np.load(directory/(phase+'.npy'),allow_pickle=False)
        require(image.dtype==np.dtype('float64') and image.shape==(15,N,N),'array type/shape differs')
        raw=image[[order.index(a) for a in I]];images.append(raw)
        require(np.max(np.abs(raw-raw[:,0:1,:]))<1e-12,'transverse invariance failed')
        clock=load(directory/(phase+'.clock.json'));require(type(clock[0]) is float and type(clock[1]) is int and abs(clock[0]-n*DT)<1e-14 and clock[1]==n,'clock differs')
        for k in range(N):primitives(raw[:,0,k])
    # Integrate the original cosine mixture over each cell, independently of np.sinc.
    left=np.arange(N)/N;right=(np.arange(N)+1)/N
    weight=.55+.03*N*(np.sin(2*np.pi*right)-np.sin(2*np.pi*left))/(2*np.pi)
    g1=np.array([gaussian(p,q,.15,-.1,.8,0.,1.1) for p,q in I])
    g2=np.array([gaussian(p,q,-.2,.15,1.2,0.,.9) for p,q in I])
    initial=g1[:,None]*weight+g2[:,None]*(1-weight)
    initial_error=float(np.max(np.abs(images[0]-initial[:,None,:])))
    require(initial_error<1e-13,'original cell-average mixture differs')
    b_gap=float(np.max(np.abs(step(images[0][:,0,:],48)-step(images[0][:,0,:],48,False))))
    require(b_gap>1e-8,'regularization is vacuous')
    errors=[];gaps=[];conservation=[]
    for n in (1,2):
        prior=images[n-1][:,0,:];ref48=step(prior,48);ref24=step(prior,24)
        error=float(np.max(np.abs(images[n]-ref48[:,None,:])));gap=float(np.max(np.abs(ref48-ref24)))
        require(error<3e-8 and gap<3e-8,'independent SSPRK2/path error')
        balance=max(abs(images[n][k].mean()-images[0][k].mean()) for k,a in enumerate(I) if sum(a)<4)
        require(balance<3e-13,'conservative inventory changed');errors.append(error);gaps.append(gap);conservation.append(float(balance))
    h=[coefficients(images[0][:,0,k])[0] for k in range(N)]
    require(max(abs(row[a]) for row in h for a in I if sum(a)==3)>1e-8 and max(abs(row[a]) for row in h for a in I if sum(a)==4)>1e-8,'higher moments vacuous')
    return {'schema':'sol61.m17.saved-physics@1','errors':errors,'initial_error':initial_error,'distinct_B0_step_gap':b_gap,'GL24_48_gaps':gaps,'conservation':conservation,'receipt_sha256':sha(directory/'receipt.json'),'full_carrier_files':[p.name for p in directory.glob('*.carriers')],'root_scientific_approval':False,'full_M17_qualification':False}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args();print(json.dumps(receive(args.directory),indent=2))
