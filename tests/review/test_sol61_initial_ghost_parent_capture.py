"""Synthetic codec inputs explicitly Source-only; strict intermediate refusal adversaries."""
import pytest
from tests.review.test_sol61_initial_ghost_selection import wire
from tests.python.support.initial_ghost_parent_capture import classify_parent_capture,CANONICAL_REFUSAL

def args():return dict(payload=None,failures=(('RuntimeError',CANONICAL_REFUSAL,True),)*2,before_blob=wire(owners=(-1,)),world_size=2,phase='parent-rejected')

def test_exact_replicated_intermediate_refusal_has_no_image_or_registry():
    r=classify_parent_capture(**args());assert r['disposition']=='canonical-refused' and 'blob' not in r and 'rows' not in r

def test_available_intermediate_keeps_real_bytes():
    a=args();a.update(payload=a['before_blob'],failures=(None,None));assert classify_parent_capture(**a)['blob']==a['payload']

@pytest.mark.parametrize('change',[{'phase':'bootstrap-baseline'},{'phase':'after-bootstrap-abort'},{'phase':'parent-before'},{'failures':(('RuntimeError','other error',True),)*2},{'failures':(('AssertionError',CANONICAL_REFUSAL,False),)*2},{'failures':(('RuntimeError',CANONICAL_REFUSAL,True),None)},{'failures':(('RuntimeError',CANONICAL_REFUSAL,True),)},{'before_blob':wire(owners=(0,))},{'payload':b''}])
def test_foreign_refusal_missing_rank_nonreplicated_or_foreign_phase_fails(change):
    a=args();a.update(change)
    with pytest.raises(ValueError):classify_parent_capture(**a)


def test_real_collective_attempt_preserves_exact_canonical_refusal(monkeypatch):
    from types import SimpleNamespace
    import pops._native_collectives as collectives
    from tests.python.support.initial_ghost_parent_capture import capture_parent_rejected
    monkeypatch.setattr(collectives,'allgather_value',lambda world,value:[value]*world.size)
    class SourceOnlyOwner:
        def checkpoint_state_carriers(self):raise RuntimeError(CANONICAL_REFUSAL)
    world=SimpleNamespace(size=2)
    result=capture_parent_rejected(world,SourceOnlyOwner(),args()['before_blob'])
    assert result['disposition']=='canonical-refused' and result['message']==CANONICAL_REFUSAL
    class Foreign(SourceOnlyOwner):
        def checkpoint_state_carriers(self):raise RuntimeError('foreign capture error')
    with pytest.raises(AssertionError):capture_parent_rejected(world,Foreign(),args()['before_blob'])
