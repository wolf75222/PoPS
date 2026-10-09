"""Real saved Native archive through the actual Python preflight; no Native execution."""
from pathlib import Path
import numpy as np
import pytest
from pops.runtime._uniform_restart_preflight import preflight_uniform_restart
ARCHIVE=Path(__file__).with_name("fixtures")/"cp9-native732268-accepted.npz"
def payload():
    with np.load(ARCHIVE,allow_pickle=False) as archive:return {k:archive[k].copy() for k in archive.files}
def test_actual_cp9_archive_passes_member_admission_without_native():
    preflight_uniform_restart(payload())
@pytest.mark.parametrize("fault",("extra","legacy8-carrier","missing-version","float-version","bool-version","missing-carrier9","unsupported-version"))
def test_version_scoped_member_admission_refuses(fault):
    d=payload()
    if fault=="extra":d["invented_native_member"]=np.asarray(1)
    elif fault=="legacy8-carrier":d["pops_checkpoint_version"]=np.asarray(8,dtype=np.int64)
    elif fault=="missing-version":del d["pops_checkpoint_version"]
    elif fault=="float-version":d["pops_checkpoint_version"]=np.asarray(9.)
    elif fault=="bool-version":d["pops_checkpoint_version"]=np.asarray(True)
    elif fault=="missing-carrier9":del d["state_carriers_checkpoint"]
    else:d["pops_checkpoint_version"]=np.asarray(10,dtype=np.int64)
    with pytest.raises((ValueError,TypeError,KeyError)):preflight_uniform_restart(d)
def test_legacy8_without_carrier_remains_explicit_valid_only_schema():
    d=payload();d["pops_checkpoint_version"]=np.asarray(8,dtype=np.int64);del d["state_carriers_checkpoint"]
    preflight_uniform_restart(d)
