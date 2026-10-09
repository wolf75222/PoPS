"""Pure diagnostic byte-oracle checks; no Native result or execution."""
import ast
from pathlib import Path
import struct
import numpy as np
from tests.python.support.tag_phase_capture import ghost_diagnostics


def image(ghost):
    # Independent minimal POPSCAR1 Source byte fixture: 1 valid center, 8 ghosts.
    words=[2,64,1,-1,1,1]
    def word(v):return int(v).to_bytes(8,"little",signed=v<0)
    raw=b"POPSCAR1"+b"".join(word(v) for v in words)+word(6)+b"marker"+word(1)
    raw+=b"".join(word(v) for v in [0,0,0,2,-1,0,0,-1,1,0,0,-1,1,18])
    values=[ghost]*9;values[4]=1.
    raw+=b"".join(struct.pack("<d",v) for v in values+[.25]*9)
    return raw


def test_zero_ghost_origin_is_distinguished_from_exact_valid_constant():
    before=image(0.);row=ghost_diagnostics(before)[0]
    assert row["valid_cells"]==1 and row["ghost_cells"]==8
    assert row["constant_valid_nonone"]==0
    assert row["constant_ghost_nonone"]==row["constant_ghost_zero"]==8
    assert row["constant_nonone_offsets"]==[0,1,2,3,5,6,7,8]
    filled=ghost_diagnostics(image(1.))[0]
    assert filled["constant_ghost_nonone"]==0
    assert filled["valid_bits_sha256"]==row["valid_bits_sha256"]
    assert filled["ghost_bits_sha256"]!=row["ghost_bits_sha256"]
    assert before==image(0.)


def test_capture_precedes_strict_guard_and_does_not_prepare_storage():
    root=Path(__file__).resolve().parents[2]
    fixture=root/"tests/python/integration/amr/test_public_tag_selection_native.py"
    tree=ast.parse(fixture.read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    captures=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=="capture"]
    assert [n.args[5].value for n in sorted(captures,key=lambda n:n.lineno)]==["bound","accepted"]
    guard=next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=="assert_array_equal" and n.args and isinstance(n.args[0],ast.Subscript) and getattr(n.args[0].value,"id",None)=="values")
    assert max(n.lineno for n in captures)<guard.lineno
    helper=ast.parse((root/"tests/python/support/tag_phase_capture.py").read_text())
    attrs={n.func.attr for n in ast.walk(helper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not any("fill" in name or "prepare" in name or "restore" in name for name in attrs)
