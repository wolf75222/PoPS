"""Independent typed-provenance refusals; no native execution or authority."""
import pytest
from types import MappingProxyType
from tests.review.sol61_initial_field_ghost_saved_reader_v3 import decode_signature, strict_json_loads

SCHEMA='inferred-boundary-expression-component@1'

@pytest.mark.parametrize('value',(
    {'nested':[float('nan')]},
    {'nested':{'number':float('inf')}},
    {'nested':{False:'bool key'}},
    {'nested':MappingProxyType({'x':1})},
    {'nested':{'tuple':(1,2)}},
))
def test_nested_non_json_signature_data_is_refused(value):
    with pytest.raises(ValueError):decode_signature({'schema':SCHEMA,**value})

@pytest.mark.parametrize('text',(
    '{"signature":{"schema":"a","schema":"b"}}',
    '{"signature":{"x":NaN}}',
    '{"signature":{"x":Infinity}}',
    '{"signature":{"x":-Infinity}}',
))
def test_nested_json_ambiguity_is_refused(text):
    with pytest.raises(ValueError):strict_json_loads(text)


def test_json_string_cannot_replace_the_typed_signature():
    with pytest.raises(ValueError,match='typed'):
        decode_signature('{"schema":"'+SCHEMA+'"}')


@pytest.mark.parametrize('mutation',('ranked','short','wrong_dtype','unknown_size','bool_size'))
def test_flat_gather_contract_rejects_shape_guessing(mutation):
    import numpy as np
    from tests.review.sol61_initial_field_ghost_saved_reader_v3 import canonical_state
    raw=np.arange(128,dtype=np.float64);size=8
    if mutation=='ranked':raw=raw.reshape(2,8,8)
    if mutation=='short':raw=raw[:-1]
    if mutation=='wrong_dtype':raw=raw.astype(np.float32)
    if mutation=='unknown_size':size=4
    if mutation=='bool_size':size=True
    with pytest.raises(ValueError):canonical_state(raw,size)
