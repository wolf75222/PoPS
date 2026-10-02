"""Explicit external-evidence test; invoke with SOL61_EXPORT_PINS, never Native."""
import json
import os
import hashlib
import numpy as np
import pytest
from tests.review.sol61_amr_full_carrier_offline import decode
from pathlib import Path
from tests.review.sol61_initial_field_ghost_saved_reader_v3 import receive


def authority():
    return json.loads(Path(os.environ['SOL61_EXPORT_PINS']).read_text())


def test_old_sdk14_lost_typed_signature_is_explicitly_refused():
    pinned = authority()
    assert len(pinned['pins']) == 28
    with pytest.raises(ValueError, match='signature'):
        receive(Path(pinned['case_directory']), pinned['pins'])


def test_actual_flat_component_major_images_match_full_carrier_bits():
    pinned = authority()
    directory = Path(pinned['case_directory'])
    assert set(pinned['pins']) == {p.name for p in directory.iterdir() if p.is_file()}
    for name, digest in pinned['pins'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    comparisons = 0
    for phase in ('initial','accepted','reloaded'):
        image = decode(np.frombuffer((directory/(phase+'-carriers.bin')).read_bytes(),dtype=np.uint8))
        with np.load(directory/(phase+'-valid.npz'),allow_pickle=False) as archive:
            for patch in image['patches']:
                level=patch['key'][1];size=(8,16)[level]
                state=archive[f'{level}-0']
                assert state.dtype==np.float64 and state.shape==(2*size*size,)
                (xl,xh,gxl,gxh),(yl,yh,gyl,gyh)=patch['axes']
                values=np.asarray(patch['bits'],dtype=np.uint64).reshape(2,gyh-gyl+1,gxh-gxl+1)
                expected=state.view(np.uint64).reshape(2,size,size)[:,yl:yh+1,xl:xh+1]
                assert values[:,yl-gyl:yh-gyl+1,xl-gxl:xh-gxl+1].tobytes()==expected.tobytes()
                comparisons+=1
    assert comparisons==9


@pytest.mark.parametrize('mutation',('missing','wrong_digest','extra'))
def test_external_inventory_and_pins_fail_before_signature_or_math(mutation):
    pinned=authority();pins=dict(pinned['pins'])
    if mutation=='missing':pins.pop(next(iter(pins)))
    if mutation=='wrong_digest':pins[next(iter(pins))]='0'*64
    if mutation=='extra':pins['invented-proof.json']='0'*64
    with pytest.raises(ValueError,match='export inventory|external export pin'):
        receive(Path(pinned['case_directory']),pins)
