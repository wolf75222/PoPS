"""Independent Source receipt: archived geometry is not a Native evolving run."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from tests.review.sol61_amr_full_carrier_offline import decode
from tests.python.support.evolving_accepted_halo_oracle import full_carrier

ARCHIVE=Path('/Users/romaindespoulain/dev/tmp/pops-api040-native-reception-evidence-20261001/installed-sdk3d8481-public-tag-representative-serial-dim2/pytest-tmp/test_public_tag_buffer_preserv0/tag-phase-capture')

def test_actual_archived_fine_only_geometry_cannot_satisfy_full_inventory():
    meta=json.loads((ARCHIVE/'bound-metadata.json').read_text())
    image=decode(np.frombuffer((ARCHIVE/'bound-state-carriers.bin').read_bytes(),dtype=np.uint8))
    assert all(row[0]==1 for row in meta['patch_boxes'])
    assert any(p['key'][1]==0 for p in image['patches'])
    with pytest.raises(AssertionError):
        full_carrier(image,image,meta['patch_boxes'],0.,0,0)

def test_native_structured_boxes_are_not_the_oracles_flat_boxes():
    meta=json.loads((ARCHIVE/'bound-metadata.json').read_text())
    assert all(len(row)==3 and len(row[1])==len(row[2])==2 for row in meta['patch_boxes'])
    image=decode(np.frombuffer((ARCHIVE/'bound-state-carriers.bin').read_bytes(),dtype=np.uint8))
    # Solely expose the tuple-shape seam, without fabricating a Native receipt.
    structured=[(0,[0,0],[7,7]),*meta['patch_boxes']]
    with pytest.raises(IndexError):full_carrier(image,image,structured,0.,0,0)
