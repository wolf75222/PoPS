"""Source-only persistence adversaries, not a Native execution."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from tests.python.support.uniform_storage_observation_v2 import save_observation,digest

def test_two_phases_are_independent_and_immediately_pinned(tmp_path):
    image=SimpleNamespace(time=0.25,macro_step=0,rank_local=b'local',complete=b'complete',contract='accepted-state-storage-observation@1')
    clock={'time':0.25,'macro_step':0};values=np.arange(6.).reshape(2,3)
    first=save_observation(tmp_path,'first',0,image,values,clock)
    second=save_observation(tmp_path,'second',0,image,values,clock)
    for receipt in (first,second):
        assert digest(tmp_path/receipt['receipt'])==receipt['sha256']
        proof=json.loads((tmp_path/receipt['receipt']).read_text())
        assert all(digest(tmp_path/name)==sha for name,sha in proof['files'].items())
    assert first['receipt']!=second['receipt']
    with pytest.raises(ValueError,match='already persisted'):save_observation(tmp_path,'first',0,image,values,clock)
    (tmp_path/'second.rank0.complete.bin').write_bytes(b'poison')
    proof=json.loads((tmp_path/second['receipt']).read_text())
    assert digest(tmp_path/'second.rank0.complete.bin')!=proof['files']['second.rank0.complete.bin']
    assert digest(tmp_path/first['receipt'])==first['sha256']

def test_clock_mismatch_refused_before_any_write(tmp_path):
    image=SimpleNamespace(time=0.25,macro_step=0)
    with pytest.raises(ValueError,match='clock differ'):
        save_observation(tmp_path,'first',0,image,np.zeros(1),{'time':0.5,'macro_step':0})
    assert list(tmp_path.iterdir())==[]
