import json,hashlib,struct
from pathlib import Path
import numpy as np
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(Path(p).read_bytes())
def read(p):return json.loads(Path(p).read_text())
def pin(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def physical(row):assert pin(row['path'])==row,row
def head(m,n):
 if n<24:return bytes([m*32+n])
 for lim,size,tag in [(255,1,24),(65535,2,25),(4294967295,4,26),(18446744073709551615,8,27)]:
  if n<=lim:return bytes([m*32+tag])+n.to_bytes(size,'big')
 raise ValueError(n)
def cbor(x):
 if x is None:return b'\xf6'
 if x is False:return b'\xf4'
 if x is True:return b'\xf5'
 if isinstance(x,int):return head(0,x)if x>=0 else head(1,-1-x)
 if isinstance(x,str):b=x.encode();return head(3,len(b))+b
 if isinstance(x,bytes):return head(2,len(x))+x
 if isinstance(x,(list,tuple)):return head(4,len(x))+b''.join(cbor(v)for v in x)
 if isinstance(x,dict):
  rows=sorted(((cbor(k),cbor(v))for k,v in x.items()),key=lambda r:(len(r[0]),r[0]));return head(5,len(rows))+b''.join(k+v for k,v in rows)
 raise TypeError(type(x))
def identity(domain,payload):return digest(cbor(dict(protocol='pops.identity',domain=domain,schema_version=1,payload=payload)))
def cpcheck(p):
 with np.load(p,allow_pickle=False)as z:
  m=json.loads(str(z['pops_checkpoint_manifest'].item()))
  assert set(z.files)==set(m['arrays'])|{'pops_checkpoint_manifest','pops_restart_identity'}
  for k,row in m['arrays'].items():
   a=np.asarray(z[k],order='C');assert not a.dtype.hasobject
   actual=dict(dtype=a.dtype.str,shape=list(a.shape),content_sha256=digest(cbor(dict(protocol='pops.array-evidence.v1',dtype=a.dtype.str,shape=list(a.shape)))+a.tobytes()))
   assert actual==row,k
  restart=identity('restart',{k:v for k,v in m.items()if k!='restart_identity'})
  assert restart==m['restart_identity']['hexdigest']
  assert str(z['pops_restart_identity'].item())=='pops.restart.v1:sha256:'+restart
  assert m['clock']==dict(time=float(z['t']).hex(),macro_step=int(z['macro_step']))
  assert float(z['t'])==1e-4 and int(z['macro_step'])==1
  return dict(**pin(p),version=int(z['pops_checkpoint_version']),arrays=len(m['arrays']),clock=m['clock'],restart_identity=restart)
def macho(p):
 b=Path(p).read_bytes();assert struct.unpack_from('<I',b)[0]==0xfeedfacf
 n=struct.unpack_from('<I',b,16)[0];off=32;sections={};deps=[];rpaths=[];uuid=None
 for _ in range(n):
  cmd,size=struct.unpack_from('<II',b,off)
  if cmd==0x19:
   ns=struct.unpack_from('<I',b,off+64)[0]
   for j in range(ns):
    s=off+72+j*80;name=b[s:s+16].split(b'\0')[0].decode();seg=b[s+16:s+32].split(b'\0')[0].decode();length,where=struct.unpack_from('<QI',b,s+40);flags=struct.unpack_from('<I',b,s+64)[0]
    if flags&255 not in (1,12,18):sections[seg+'/'+name]=dict(bytes=length,sha256=digest(b[where:where+length]))
  if cmd in (0xc,0x80000018,0x8000001f,0x80000023):
   start=struct.unpack_from('<I',b,off+8)[0];deps.append(b[off+start:off+size].split(b'\0')[0].decode())
  if cmd==0x8000001c:
   start=struct.unpack_from('<I',b,off+8)[0];rpaths.append(b[off+start:off+size].split(b'\0')[0].decode())
  if cmd==0x1b:uuid=b[off+8:off+24].hex()
  off+=size
 return dict(sections=sections,dependencies=deps,rpaths=rpaths,uuid=uuid)

