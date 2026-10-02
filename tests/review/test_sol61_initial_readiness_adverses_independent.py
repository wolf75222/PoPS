"""Independent Source readiness refusals; recording APIs are not Native evidence."""
import pytest
from tests.python.support.initial_ghost_registry_readiness import capture_registry_disposition,classify_registry_disposition,EXPECTED_REFUSAL

def test_slot_consensus_refusal_precedes_any_per_slot_or_registry_call(monkeypatch):
    import pops._native_collectives as collectives
    monkeypatch.setattr(collectives,'allgather_value',lambda world,value:[['foreign-slot']])
    calls=[]
    class Owner:
        def field_provider_slots(self):return ['real-slot']
        def field_provider_materialized(self,slot):calls.append('readiness');return False
        def field_provider_checkpoint_manifest(self):calls.append('manifest');return []
        def checkpoint_rank_local_carrier_manifest(self):calls.append('registry');return []
    with pytest.raises(AssertionError):capture_registry_disposition(None,Owner(),'bootstrap-baseline')
    assert calls==[]

@pytest.mark.parametrize('exception',['LogicError','ValueError','RuntimeError'])
def test_foreign_or_ready_phase_error_is_not_unavailable_authority(exception):
    common=dict(world_size=1,slots=['slot'],flags=[False],
        manifest=[['pops.amr.field-provider-checkpoint-manifest@1','slot','1','p','plan','cfg','1','1','unmaterialized']],
        payload=None,phase='bootstrap-baseline')
    failures=((exception,EXPECTED_REFUSAL if exception!='RuntimeError' else EXPECTED_REFUSAL+' drift',True),)
    with pytest.raises(ValueError):classify_registry_disposition(failures=failures,**common)
    common.update(flags=[True],phase='parent-before');common['manifest'][0][8]='materialized'
    with pytest.raises(ValueError):classify_registry_disposition(failures=(('RuntimeError',EXPECTED_REFUSAL,True),),**common)
