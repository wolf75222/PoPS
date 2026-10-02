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

def test_archived_complete_halfopen_geometry_and_mutated_ghost():
    image=decode(np.frombuffer((ARCHIVE/'bound-state-carriers.bin').read_bytes(),dtype=np.uint8))
    geometry=tuple(tuple((p['axes'][1][0],p['axes'][0][0],p['axes'][1][1]+1,p['axes'][0][1]+1) for p in image['patches'] if p['key'][1]==level) for level in range(2))
    full_carrier(image,image,geometry,0.,0,0)
    bad=copy.deepcopy(image)
    bits=np.asarray(bad['patches'][1]['bits'],dtype=np.uint64).copy().view(np.float64).reshape(2,-1)
    bits[1,0]=12345.
    bad['patches'][1]['bits']=tuple(bits.view(np.uint64).ravel())
    with pytest.raises(AssertionError):full_carrier(image,bad,geometry,0.,0,0)

@pytest.mark.parametrize('constant',(0,1))
def test_affine_common_shift_at_unobserved_elapsed_and_order(constant):
    image=decode(np.frombuffer((ARCHIVE/'bound-state-carriers.bin').read_bytes(),dtype=np.uint8))
    if constant:
        for patch in image['patches']:
            bits=np.asarray(patch['bits'],dtype=np.uint64).reshape(2,-1)[::-1].copy()
            patch['bits']=tuple(bits.ravel())
    geometry=tuple(tuple((p['axes'][1][0],p['axes'][0][0],p['axes'][1][1]+1,p['axes'][0][1]+1) for p in image['patches'] if p['key'][1]==level) for level in range(2))
    advanced=copy.deepcopy(image);elapsed=3/128
    for patch in advanced['patches']:
        values=np.asarray(patch['bits'],dtype=np.uint64).copy().view(np.float64).reshape(2,-1)
        values[1-constant]+=elapsed;patch['bits']=tuple(values.view(np.uint64).ravel())
    full_carrier(image,advanced,geometry,elapsed,2,constant)
