"""Physical qualification survives the authenticated serialization boundary."""
import copy
import json
from fractions import Fraction

import pytest

from pops._ir.quantity import PhysicalDimension, PhysicalSupport
from pops.model import Module, ModuleManifest
from pops.model.provider_pack import ProviderPack


def _module():
    module = Module("qualified_snapshot")
    module.state_space(
        "density", ("rho",), units=(PhysicalDimension((("length", Fraction(-3)),)),),
        support=PhysicalSupport((("x", "periodic_interval"),)), sampling="cell_average",
        value_shape=(), domain="positive",
    )
    module.field_space("temperature", ("rho",), units=(PhysicalDimension(),),
                       support=PhysicalSupport((("v", "velocity_space"),)))
    return module


def test_manifest_and_provider_roundtrip_preserve_physical_qualification():
    module = _module()
    before = module.module_hash()
    payload = module.manifest().to_dict()
    restored = ModuleManifest.from_json(json.dumps(payload))
    assert restored.to_dict() == payload
    assert payload["state_spaces"]["density"]["support"] != payload["field_spaces"]["temperature"]["support"]
    assert payload["state_spaces"]["density"]["value_shape"] == []
    assert payload["state_spaces"]["density"]["units"][0]["powers"] == [["length", -3, 1]]
    assert payload["field_spaces"]["temperature"]["units"][0]["powers"] == []
    pack = ProviderPack.from_data(payload["provider_pack"])
    assert pack.to_data() == payload["provider_pack"]
    module.freeze()
    assert module.module_hash() == before
    assert module.manifest().to_dict() == payload


@pytest.mark.parametrize("key,value", [("value_shape", [2]), ("domain", ""),
                                     ("support", {"kind": "physical_support", "coordinates": []})])
def test_manifest_rejects_malformed_physical_type(key, value):
    payload = copy.deepcopy(_module().manifest().to_dict())
    payload["state_spaces"]["density"][key] = value
    with pytest.raises((TypeError, ValueError)):
        ModuleManifest.from_json(json.dumps(payload))


def test_unknown_dimension_is_not_dimensionless():
    unknown = Module("unit_identity")
    unknown.state_space("U", ("u",))
    known = Module("unit_identity")
    known.state_space("U", ("u",), units=(PhysicalDimension(),))
    assert unknown.module_hash() != known.module_hash()
    assert unknown.manifest().state_spaces["U"]["units"] == (None,)
