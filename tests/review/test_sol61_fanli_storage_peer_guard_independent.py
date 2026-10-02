"""Source-only peer voting branch; no Native/MPI execution."""
from types import SimpleNamespace
import pytest
from tests.python.support import fan_li15_storage_capture as support
from tests.python.support import collective_checks

def test_peer_rejection_enters_ledger_and_prevents_next_operation(monkeypatch):
    events=[];world=SimpleNamespace(rank=0,size=2)
    failure=('ValueError','peer local/complete mismatch',False)
    # Explicit test transport: local checker succeeds, peer contributes failure.
    def attempt(w,operation):
        assert w is world
        operation();return None,(None,failure)
    monkeypatch.setattr(collective_checks,'collective_attempt',attempt)
    monkeypatch.setattr(collective_checks,'collective_call',lambda w,operation:operation())
    monkeypatch.setattr(support,'validate_storage',lambda *a,**kw:events.append(('guard',kw)))
    with pytest.raises(RuntimeError,match='peer storage integrity guard failed'):
        support.guard_persisted_storage(world,(None,None,None),lambda failures:events.append(('ledger',failures)))
        events.append('future-run')
    assert events==[('guard',{'rank':0,'ranks':2}),('ledger',(None,failure))]
