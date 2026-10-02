"""Exact offline grown/valid bit diff; does not qualify Native rollback."""
import hashlib
import struct
import numpy as np
from tests.review.sol61_amr_full_carrier_offline import decode

def inspect(before,after):
    a,b=(decode(np.frombuffer(data,dtype=np.uint8)) for data in (before,after))
    if {k:v for k,v in a.items() if k!='patches'}!={k:v for k,v in b.items() if k!='patches'}:raise ValueError('carrier envelope differs')
    if len(a['patches'])!=len(b['patches']):raise ValueError('patch count differs')
    rows=[]
    for x,y in zip(a['patches'],b['patches']):
        if {k:v for k,v in x.items() if k!='bits'}!={k:v for k,v in y.items() if k!='bits'}:raise ValueError('patch authority differs')
        widths=[axis[3]-axis[2]+1 for axis in x['axes']]
        points=int(np.prod(widths))
        for index,(old,new) in enumerate(zip(x['bits'],y['bits'])):
            if old==new:continue
            component=index//points;offset=index%points;coordinate=[]
            for axis,width in zip(x['axes'],widths):coordinate.append(axis[2]+offset%width);offset//=width
            valid=all(axis[0]<=q<=axis[1] for axis,q in zip(x['axes'],coordinate))
            rows.append(dict(key=x['key'],component=component,coordinate=coordinate,valid=valid,before_bits=hex(old),after_bits=hex(new)))
    poison=int.from_bytes(struct.pack('<d',-1234.),'little')
    return dict(schema='sol61.initial-parent-carrier-delta@1',scope='offline diagnostic; no acceptance',
        before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest(),
        changed_valid=sum(row['valid'] for row in rows),changed_ghost=sum(not row['valid'] for row in rows),
        poison_present_after=any(poison in patch['bits'] for patch in b['patches']),rows=rows)
