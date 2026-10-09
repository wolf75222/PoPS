"""Strict intermediate capture disposition, not accepted-state persistence."""
import numpy as np
from tests.review.sol61_amr_full_carrier_offline import decode
CANONICAL_REFUSAL='state carrier capture canonicalization; rank 0: state carrier duplicate or divergent replicated source patch'

def classify_parent_capture(payload, failures, before_blob, world_size, phase):
    if phase!='parent-rejected' or type(world_size) is not int or world_size<1:
        raise ValueError('intermediate capture phase/rank authority differs')
    if type(failures) not in (list,tuple) or len(failures)!=world_size:
        raise ValueError('intermediate capture rank outcomes missing')
    if type(before_blob) is not bytes:raise ValueError('full parent-before carrier is mandatory')
    before=decode(np.frombuffer(before_blob,dtype=np.uint8))
    if before['ranks']!=world_size or before['shard']!=-1:raise ValueError('parent-before full carrier authority differs')
    if not any(f is not None for f in failures):
        if type(payload) is not bytes:raise ValueError('successful intermediate capture lacks bytes')
        after=decode(np.frombuffer(payload,dtype=np.uint8))
        if after['ranks']!=world_size or after['shard']!=-1:raise ValueError('intermediate full carrier authority differs')
        return {'schema':'sol61.initial-ghost-parent-capture-disposition@1','disposition':'available','blob':payload}
    exact=('RuntimeError',CANONICAL_REFUSAL,True)
    if world_size<2 or not any(p['owner']==-1 for p in before['patches']) or payload is not None or any(type(f) not in (tuple,list) or len(f)!=3 or type(f[0]) is not str or type(f[1]) is not str or type(f[2]) is not bool or tuple(f)!=exact for f in failures):
        raise ValueError('intermediate capture returned a foreign refusal')
    return {'schema':'sol61.initial-ghost-parent-capture-disposition@1','disposition':'canonical-refused','exception':'RuntimeError','message':CANONICAL_REFUSAL,'rank_count':world_size,'scope':'unaccepted divergent replicated halos; no image or registry payload received'}

def capture_parent_rejected(world, native_owner, before_blob):
    from tests.python.support.collective_checks import collective_attempt,collective_call
    payload,failures=collective_attempt(world,lambda:bytes(native_owner.checkpoint_state_carriers()))
    size=world.size if world is not None else 1
    return collective_call(world,lambda:classify_parent_capture(payload,failures,before_blob,size,'parent-rejected'))
