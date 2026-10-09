"""Authentic archived CP copy, Source/offline only; no runtime or Native binding."""
from pathlib import Path
import json
import hashlib
import zipfile
import numpy as np
import pytest
from pops.output._checkpoint_contract import CheckpointResourceBudget
from pops.output._checkpoint_collective import decode_checkpoint_bytes
from pops.runtime._checkpoint_manifest import inspect_checkpoint_payload_integrity
from tests.python.support.evolving_accepted_halo_archive import truncated_carrier_archive
from tests.review.sol61_amr_full_carrier_offline import decode

BASE=Path("/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001")
ARCHIVE=BASE/"installed-sdk3d8481-src5af964-public-evolving-halo-sync-serial-dim2/pytest-tmp/test_public_evolving_accepted_0/evolving-halo/accepted-checkpoint.npz"

def reviewed_budget(path):
    with zipfile.ZipFile(path) as archive:infos=archive.infolist()
    with np.load(path,allow_pickle=False) as image:
        manifest,_=inspect_checkpoint_payload_integrity(image,runtime_kind="amr")
    # Offline envelope capacity only, derived from actual retained member bytes.
    # This is not an installed live runtime resource proof.
    members=len(infos);total=sum(i.file_size for i in infos)
    chars=sum(len(i.filename) for i in infos)+len(json.dumps(manifest))*2
    return CheckpointResourceBudget("amr",members,chars,max(i.file_size for i in infos),total,path.stat().st_size,"Source-offline-actual-sealed-archive-copy")

def test_actual_checkpoint_rewrite_reaches_carrier_decoder(tmp_path):
    assert ARCHIVE.is_file(),"authentic ROOT archive required, no synthetic replacement"
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()=="f85e1cb586f2ead68784c49d3342d279d51ec07fa821364757acce6ae96ba81f"
    old=decode_checkpoint_bytes(ARCHIVE.read_bytes(),reviewed_budget(ARCHIVE))
    inspect_checkpoint_payload_integrity(old,runtime_kind="amr")
    decode(old["state_carriers_checkpoint"])
    bad=tmp_path/"carrier-truncated.npz";payload=truncated_carrier_archive(ARCHIVE,bad)
    new=decode_checkpoint_bytes(bad.read_bytes(),reviewed_budget(bad))
    inspect_checkpoint_payload_integrity(new,runtime_kind="amr")
    with zipfile.ZipFile(bad) as archive:assert all(i.compress_type==zipfile.ZIP_DEFLATED for i in archive.infolist())
    assert new["state_carriers_checkpoint"].size==old["state_carriers_checkpoint"].size-1
    assert bytes(new["state_carriers_checkpoint"])==bytes(old["state_carriers_checkpoint"])[:-1]
    with pytest.raises(ValueError,match="carrier payload shape differs"):decode(new["state_carriers_checkpoint"])
