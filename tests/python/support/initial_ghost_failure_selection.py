"""Pure strict owned-face selection; caller must vote errors and agree before Native."""
import hashlib
import json
import numpy as np
from tests.review.sol61_amr_full_carrier_offline import decode

def select_xmin_owner(blob,world_size):
    if type(blob) is not bytes or type(world_size) is not int or world_size<1:raise ValueError('selection requires canonical bytes and exact positive world size')
    geometry=decode(np.frombuffer(blob,dtype=np.uint8))
    if geometry['dim']!=2 or geometry['ranks']!=world_size or geometry['shard']!=-1:raise ValueError('selection requires complete global Dim2 carrier authority')
    eligible=set()
    for patch in geometry['patches']:
        if patch['axes'][0][0]==0:eligible.update(range(world_size) if patch['owner']==-1 else (patch['owner'],))
    if not eligible:raise ValueError('selection has no owned xmin physical face')
    return {'schema':'sol61.initial-ghost-owned-face-selection@1','dimension':2,'world_size':world_size,'carrier_sha256':hashlib.sha256(blob).hexdigest(),'eligible':sorted(eligible),'target':max(eligible)}

def require_selection_agreement(rows,expected):
    keys={'schema','dimension','world_size','carrier_sha256','eligible','target'}
    if type(expected) is not dict or set(expected)!=keys or expected['schema']!='sol61.initial-ghost-owned-face-selection@1':raise ValueError('selection identity shape differs')
    if type(expected['dimension']) is not int or expected['dimension']!=2 or type(expected['world_size']) is not int or expected['world_size']<1:raise ValueError('selection identity authority differs')
    eligible=expected['eligible'];target=expected['target'];digest=expected['carrier_sha256']
    if type(eligible) is not list or not eligible or any(type(rank) is not int or not 0<=rank<expected['world_size'] for rank in eligible) or eligible!=sorted(set(eligible)) or type(target) is not int or target!=max(eligible):raise ValueError('selection identity owner differs')
    if type(digest) is not str or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('selection identity digest differs')
    if type(rows) not in (list,tuple) or len(rows)!=expected['world_size'] or any(json.dumps(row,sort_keys=True,allow_nan=False)!=json.dumps(expected,sort_keys=True,allow_nan=False) for row in rows):raise ValueError('initial Ghost selection identity differs between ranks')
    return {'schema':'sol61.initial-ghost-selection-agreement@1','rows':list(rows),'agreed':expected}
