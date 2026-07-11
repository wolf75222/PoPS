"""ADC-658: exact immutable typed component/provider lowering metadata."""
import pytest

pytest.importorskip("pops")

from pops.codegen.module_lowering import _module_to_model
from pops.model import Module
from pops.model.provider_pack import (
    ComponentContract,
    ComponentKey,
    MissingInputProvider,
    ProviderEntry,
    ProviderPack,
)


def _row(component="rho", slot=0, *, producer="initial", available=True):
    key = ComponentKey("owner", "state", "U", component)
    contract = ComponentContract("conservative", "cell", "kg/m3", "cell")
    return key, contract, ProviderEntry(producer, available, slot)


def test_provider_pack_exact_lookup_and_contract():
    key, contract, entry = _row()
    pack = ProviderPack([(key, contract, entry)], capacity=1)
    assert pack[key] == entry
    assert pack.lookup(key, contract) == entry
    assert pack.contract(key) == contract
    with pytest.raises(MissingInputProvider):
        pack.lookup(ComponentKey("owner", "state", "other", "rho"))
    with pytest.raises(MissingInputProvider, match="contract mismatch"):
        pack.lookup(key, ComponentContract("primitive", "cell", "kg/m3", "cell"))


@pytest.mark.parametrize("entry", [
    ProviderEntry(None, True, 0),
    ProviderEntry("producer", False, 0),
    ProviderEntry("producer", True, None),
])
def test_provider_pack_unset_or_unavailable_is_missing(entry):
    key, contract, _ = _row()
    pack = ProviderPack([(key, contract, entry)])
    with pytest.raises(MissingInputProvider):
        pack[key]


def test_provider_pack_accepts_exact_capacity_and_refuses_capacity_plus_one_atomically():
    first = _row("rho", 0)
    second = _row("mx", 1)
    exact = ProviderPack([first, second], capacity=2)
    assert len(exact) == 2
    assert ProviderPack.from_data(exact.to_data()).to_data() == exact.to_data()
    with pytest.raises(ValueError, match="capacity"):
        ProviderPack([first, second, _row("my", 2)], capacity=2)
    assert len(exact) == 2
    with pytest.raises(AttributeError):
        exact._capacity = 3


def test_module_lowering_retains_typed_contract_producer_and_availability():
    module = Module("typed")
    state = module.state_space("U", ("rho",), units=("kg/m3",))
    fields = module.field_space("electric", ("phi", "grad_x", "grad_y"),
                                units=("V", "V/m", "V/m"))
    module.operator("solve_electric", state >> fields, "field_operator", expr=1.0)
    lowered = _module_to_model(module)
    pack = lowered._component_provider_pack
    rows = pack.to_data()["entries"]
    phi = next(row for row in rows if row["key"]["component"] == "phi")
    assert phi["key"]["space_kind"] == "field"
    assert phi["key"]["space_name"] == "electric"
    assert phi["contract"]["unit"] == "V"
    assert "solve_electric" in phi["provider"]["producer"]
    assert phi["provider"]["availability"] is True


def test_same_component_spelling_in_distinct_typed_spaces_never_merges():
    module = Module("collision")
    module.state_space("U", ("rho",))
    module.field_space("left", ("phi",))
    module.field_space("right", ("phi",))
    with pytest.raises(ValueError, match="cannot be merged silently"):
        _module_to_model(module)


@pytest.mark.parametrize("count", [2, 4])
def test_module_field_operator_rejects_unsupported_output_arity(count):
    module = Module("bad_arity_%d" % count)
    state = module.state_space("U", ("rho",))
    fields = module.field_space("fields", tuple("f%d" % i for i in range(count)))
    module.operator("solve", state >> fields, "field_operator", expr=1.0)
    with pytest.raises(ValueError, match="length 1 or 3"):
        _module_to_model(module)
