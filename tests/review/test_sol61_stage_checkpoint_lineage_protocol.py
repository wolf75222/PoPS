"""Explicit synthetic codec tests, not native evidence or a physical oracle."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).with_name("sol61_stage_checkpoint_lineage.py")
SPEC = importlib.util.spec_from_file_location("independent_stage_lineage", PATH)
READER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READER)


@pytest.mark.parametrize("value,expected", [
    (None, "f6"), (False, "f4"), (True, "f5"), (0, "00"), (23, "17"),
    (24, "1818"), (256, "190100"), (-1, "20"), (-25, "3818"),
    ("x", "6178"), (b"x", "4178"), ([1, 2], "820102"),
    ({"b": 2, "a": 1}, "a2616101616202"),
    ({"long": 1, "b": 2}, "a2616202646c6f6e6701"),
])
def test_declared_cbor_wire_vectors(value, expected):
    assert READER.cbor(value).hex() == expected


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_json_ambiguity_refused(text):
    with pytest.raises(ValueError):
        READER.strict_json(text)


@pytest.mark.parametrize("value", [1.0, object(), {1: "bad"}])
def test_opaque_identity_values_refused(value):
    with pytest.raises(ValueError):
        READER.cbor(value)
