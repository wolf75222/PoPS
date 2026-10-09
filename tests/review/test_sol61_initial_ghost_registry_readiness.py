"""Pure Source dispositions using exact native refusal contract; no runtime payload invention."""
from copy import deepcopy
import pytest
from tests.python.support.initial_ghost_registry_readiness import classify_registry_disposition,EXPECTED_REFUSAL

def args(ready=False):
    return dict(world_size=1,slots=['qualified::field'],flags=[ready],manifest=[['pops.amr.field-provider-checkpoint-manifest@1','qualified::field','1','p','plan','cfg','1','1','materialized' if ready else 'unmaterialized']],failures=(None,) if ready else (('RuntimeError',EXPECTED_REFUSAL,True),),payload=[['SourceOnly-row']] if ready else None,phase='parent-before' if ready else 'bootstrap-baseline')

def test_exact_refusal_is_disposition_without_registry_payload():
    a=args();r=classify_registry_disposition(**a);a['phase']='after-bootstrap-abort';assert r==classify_registry_disposition(**a)
    assert r['disposition']=='readiness-refused' and 'rows' not in r and r['refusal']['message']==EXPECTED_REFUSAL

def test_ready_rows_remain_real_payload():
    a=args(True);r=classify_registry_disposition(**a);assert r['rows'] is a['payload'] and r['disposition']=='available'

@pytest.mark.parametrize('change',[{'failures':(('RuntimeError','different error',True),)},{'failures':(None,)},{'flags':[True]},{'flags':[0]},{'phase':'parent-before'},{'payload':[]},{'phase':'unknown'},{'failures':(('RuntimeError',EXPECTED_REFUSAL,False),)}])
def test_other_refusal_phase_or_unexpected_materialization_fails(change):
    a=args();a.update(change)
    with pytest.raises(ValueError):classify_registry_disposition(**a)

def test_readiness_manifest_disagreement_fails():
    a=args();a['manifest'][0][8]='materialized'
    with pytest.raises(ValueError):classify_registry_disposition(**a)


def test_actual_collective_attempt_readiness_route(monkeypatch):
    from tests.python.support.initial_ghost_registry_readiness import capture_registry_disposition
    import pops._native_collectives as collectives
    monkeypatch.setattr(collectives,'allgather_value',lambda world,value:[value])
    class SourceOnlyOwner:
        def field_provider_slots(self):return args()['slots']
        def field_provider_materialized(self,slot):assert slot=='qualified::field';return False
        def field_provider_checkpoint_manifest(self):return args()['manifest']
        def checkpoint_rank_local_carrier_manifest(self):raise RuntimeError(EXPECTED_REFUSAL)
    result=capture_registry_disposition(None,SourceOnlyOwner(),'bootstrap-baseline')
    assert result==classify_registry_disposition(**args())
    class WrongError(SourceOnlyOwner):
        def checkpoint_rank_local_carrier_manifest(self):raise RuntimeError('different runtime error')
    with pytest.raises(AssertionError):capture_registry_disposition(None,WrongError(),'bootstrap-baseline')

def test_missing_rank_never_becomes_readiness_disposition():
    a=args();a['world_size']=2
    with pytest.raises(ValueError):classify_registry_disposition(**a)
