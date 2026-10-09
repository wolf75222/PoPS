"""Independent strict preflight probes using the authentic retained Native NPZ.

No Native execution, forged archive seal or runtime stand-in is used here.
"""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import pytest
from pops.runtime._uniform_restart_preflight import preflight_uniform_restart
from tests.review.test_sol61_cp9_archive_member_admission import payload,ARCHIVE
ROOT=Path(__file__).resolve().parents[2]


def test_original_archive_pin_and_historical_unknown_member_refusal():
    metadata=json.loads((ROOT/'tests/review/sol61_cp9_archive_member_admission_fix.json').read_text())
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==metadata['fixture_sha256']
    assert ARCHIVE.stat().st_size==metadata['fixture_bytes']==18300
    previous=subprocess.check_output(['git','-C',str(ROOT),'show',metadata['base']+':python/pops/runtime/_uniform_restart_preflight.py'],text=True)
    namespace={'__name__':'source_historical_uniform_preflight'}
    exec(compile(previous,'historical_uniform_restart_preflight','exec'),namespace)
    with pytest.raises(ValueError,match='unknown.*state_carriers_checkpoint'):
        namespace['preflight_uniform_restart'](payload())
    preflight_uniform_restart(payload())


@pytest.mark.parametrize('version',[np.asarray([9],dtype=np.int64),np.asarray('9'),np.asarray(7,dtype=np.int64),np.asarray(2**64-1,dtype=np.uint64),np.asarray(9,dtype=object)])
def test_nonexact_or_unsupported_version_does_not_admit_cp9_member(version):
    data=payload();data['pops_checkpoint_version']=version
    with pytest.raises((TypeError,ValueError)):
        preflight_uniform_restart(data)


@pytest.mark.parametrize('name',['state_carriers_checkpoint_extra','state_carriers','state_carriers_checkpoint/extra'])
def test_carrier_name_scope_does_not_admit_unknown_siblings(name):
    data=payload();data[name]=data['state_carriers_checkpoint'].copy()
    with pytest.raises(ValueError,match='unknown'):
        preflight_uniform_restart(data)
