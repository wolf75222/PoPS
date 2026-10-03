"""Public frame authority; Source only, no installed backend claim."""
import copy
import pytest
from pops.model import Module
from pops.model.manifest import ModuleManifest
from pops.domain import Rectangle
from pops.frames import Cartesian2D

def frame():
    return Rectangle("declared chart", (0, 0), (1, 2)).frame(Cartesian2D())

def test_legacy_manifest_stays_ten():
    a=Module("legacy"); a.state_space("state", ("q",))
    row=a.manifest().to_dict()
    assert row["schema_version"] == 10 and "physical_frame" not in row
    assert ModuleManifest.from_dict(row).to_dict() == row

def test_frame_all_states_roundtrip_identity():
    chart=frame(); a=Module("physical", frame=chart)
    for name, components in (("other", ("u","v")), ("selected", ("z",))):
        assert a.state_space(name, components).frame == chart.canonical_id
    row=a.manifest().to_dict()
    assert row["schema_version"] == 12 and row["physical_frame"] == chart.to_dict()
    assert ModuleManifest.from_dict(row).to_dict() == row
    assert ModuleManifest.from_dict(row).hash == a.manifest().hash

def test_foreign_state_frame_refused():
    a=Module("physical", frame=frame())
    with pytest.raises(ValueError, match="frame"):
        a.state_space("bad", ("q",), frame="foreign")

def test_manifest_frame_cannot_be_detached_from_states():
    a=Module("physical", frame=frame()); a.state_space("state", ("q",))
    row=copy.deepcopy(a.manifest().to_dict()); row["state_spaces"]["state"]["frame"]="foreign"
    with pytest.raises(ValueError, match="frame"):
        ModuleManifest.from_dict(row)

def test_invalid_frame_type_refused():
    with pytest.raises(TypeError): Module("bad", frame={})
