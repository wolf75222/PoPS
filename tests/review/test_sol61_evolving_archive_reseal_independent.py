"""Independent actual archive copy checks; never invokes Native restart."""
import hashlib
import zipfile
import numpy as np
import pytest
from tests.review.test_sol61_evolving_halo_archive_source import ARCHIVE,reviewed_budget
from tests.python.support.evolving_accepted_halo_archive import truncated_carrier_archive
from pops.output._checkpoint_collective import decode_checkpoint_bytes
from pops.runtime._checkpoint_manifest import MANIFEST_KEY,IDENTITY_KEY,inspect_checkpoint_payload_integrity

def test_only_carrier_and_its_seal_change_original_lifecycle_survives(tmp_path):
    original=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    old=decode_checkpoint_bytes(ARCHIVE.read_bytes(),reviewed_budget(ARCHIVE))
    destination=tmp_path/'resealed.npz';truncated_carrier_archive(ARCHIVE,destination)
    new=decode_checkpoint_bytes(destination.read_bytes(),reviewed_budget(destination))
    before,_=inspect_checkpoint_payload_integrity(old,runtime_kind='amr')
    after,_=inspect_checkpoint_payload_integrity(new,runtime_kind='amr')
    assert set(old)==set(new)
    for key in old:
        if key in (MANIFEST_KEY,IDENTITY_KEY,'state_carriers_checkpoint'):continue
        assert old[key].dtype==new[key].dtype and old[key].shape==new[key].shape
        assert old[key].tobytes()==new[key].tobytes(),key
    for key in ('semantic_identity','artifact_identity','bind_identity','run_identity','clock','origin'):
        assert before.get(key)==after.get(key),key
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==original
    assert new['state_carriers_checkpoint'].tobytes()==old['state_carriers_checkpoint'].tobytes()[:-1]

def test_uncompressed_early_zip_refusal_is_distinct_from_carrier_phase(tmp_path):
    with np.load(ARCHIVE,allow_pickle=False) as saved:payload={k:saved[k].copy() for k in saved.files}
    bad=tmp_path/'uncompressed.npz';np.savez(bad,**payload)
    with zipfile.ZipFile(bad) as archive:assert all(i.compress_type==zipfile.ZIP_STORED for i in archive.infolist())
    with pytest.raises(ValueError,match='invalid member contract'):
        decode_checkpoint_bytes(bad.read_bytes(),reviewed_budget(bad))
