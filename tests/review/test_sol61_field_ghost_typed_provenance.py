"""True public Source serializer and negative data grammar; no Native receipt."""
import json
import pytest
from tests.review.sol61_initial_field_ghost_saved_reader_v4 import decode_signature,strict_json_loads


def test_public_manifest_serializer_preserves_real_nested_signature_json():
    import pops
    from tests.python.support.initial_field_ghost_native_case import build
    case,layout=build();plan=pops.resolve(pops.validate(case),layout=layout)
    component,=plan.component_inputs
    data=component.component_manifest.to_data()
    wire=json.dumps(data['signature'],allow_nan=False)
    roundtrip=strict_json_loads(wire)
    signature=decode_signature(roundtrip['inferred_boundary_expression'])
    assert len(signature['expressions'])==2
    assert signature['expressions'][0]['value']['qualified_leaves'][0]['support']=='clamped-primary-state-interior-trace@1'
    assert 'physical_time' in signature['expressions'][1]['value']['ordered_cpp']
    # Original shallow/default=str export loses typing and is deliberately refused.
    lossy=json.loads(json.dumps(dict(component.component_manifest.signature),default=str))
    with pytest.raises(ValueError,match='lossy'):decode_signature(lossy['inferred_boundary_expression'])

@pytest.mark.parametrize('raw',("{'schema': 'inferred-boundary-expression-component@1'}",'{"schema":"inferred-boundary-expression-component@1"}',[],None,{'schema':'wrong'},{'schema':'inferred-boundary-expression-component@1','value':(1,2)}))
def test_wrong_representation_and_non_json_types_refused(raw):
    with pytest.raises(ValueError):decode_signature(raw)

@pytest.mark.parametrize('raw',('{"x":1,"x":2}','{"nested":{"x":1,"x":2}}','{"x":NaN}','not JSON'))
def test_duplicate_nonfinite_and_malformed_json_refused(raw):
    with pytest.raises(ValueError):strict_json_loads(raw)


def test_flat_component_major_authority_refuses_reinterpretations():
    import numpy as np
    from tests.review.sol61_initial_field_ghost_saved_reader_v4 import canonical_state
    raw=np.arange(128,dtype=np.float64)
    image=canonical_state(raw,8)
    assert image[0,7,7]==63 and image[1,0,0]==64
    for wrong in (raw.reshape(2,8,8),raw.astype(np.float32),raw[:-1],raw.view(np.uint64)):
        with pytest.raises(ValueError):canonical_state(wrong,8)


def test_rectangular_writer_box_authenticates_reversed_half_open_axes():
    from pathlib import Path
    from tests.review.sol61_initial_field_ghost_saved_reader_v4 import writer_box
    patch={'axes':[(2,4,1,5),(7,12,6,13)]}
    assert writer_box(patch)==[7,2,13,5]
    assert writer_box(patch)!=[2,7,5,13]
    root=Path(__file__).resolve().parents[2]
    native=(root/'python/bindings/core/init/output_geometry_binding.hpp').read_text()
    assert 'const int native_axis = Dim - 1 - array_axis;' in native
    assert 'box.lo[native_axis]' in native and 'box.hi[native_axis] + 1' in native


def test_rectangular_patch_slice_preserves_nonconstant_coordinate_bits():
    import numpy as np
    from tests.review.sol61_initial_field_ghost_saved_reader_v4 import writer_box,canonical_state
    y,x=np.indices((8,8));state=np.stack((100*y+x,1000+100*y+x)).astype(np.float64)
    flat=state.ravel(order='C')
    restored=canonical_state(flat,8)
    patch={'axes':[(1,2,0,3),(2,4,1,5)]}
    yl,xl,yh,xh=writer_box(patch)
    actual=restored[:,yl:yh,xl:xh]
    expected=state[:,2:5,1:3]
    assert actual.shape==(2,3,2) and actual.tobytes()==expected.tobytes()
    assert restored[:,1:3,2:5].shape!=actual.shape
