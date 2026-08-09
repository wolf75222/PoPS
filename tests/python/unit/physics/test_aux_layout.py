"""Generic compact auxiliary-provider layout tests."""
from __future__ import annotations

import pytest

from pops._aux_layout import AuxLayout, aux_component_index, aux_layout, aux_total_n_aux
from pops.codegen.component_provider_packs import (
    auxiliary_component_slot,
    compact_auxiliary_provider_pack,
    consumer_provider_plan,
)
from pops.model.provider_pack import ComponentContract, ComponentKey, ProviderEntry, ProviderPack


def _contract() -> ComponentContract:
    return ComponentContract("auxiliary", "cell", None, "cell", "cell_scalar")


def test_aux_layout_is_declaration_ordered_and_compact() -> None:
    layout = aux_layout(("potential", "electric_x", "temperature"))

    assert isinstance(layout, AuxLayout)
    assert dict(layout.components) == {
        "potential": 0,
        "electric_x": 1,
        "temperature": 2,
    }
    assert aux_component_index("temperature", layout.names) == 2
    assert aux_total_n_aux(layout.names) == 3


def test_aux_layout_accepts_zero_components_and_rejects_invalid_or_duplicate_names() -> None:
    assert aux_layout(()).n_components == 0
    with pytest.raises(ValueError, match="valid identifier"):
        aux_layout(("not-valid",))
    with pytest.raises(ValueError, match="unique"):
        aux_layout(("coefficient", "coefficient"))


def test_provider_pack_is_the_only_auxiliary_slot_authority() -> None:
    owner = "model/demo"
    field = ComponentKey(owner, "field", "electrostatic", "potential")
    aux = ComponentKey(owner, "aux", "material", "temperature")
    complete = ProviderPack((
        (field, _contract(), ProviderEntry("field_solver", True, 17)),
        (aux, _contract(), ProviderEntry("runtime_input", True, 9)),
    ))

    compact = compact_auxiliary_provider_pack(complete)
    assert auxiliary_component_slot(compact, owner_qid=owner, name="potential") == 0
    assert auxiliary_component_slot(compact, owner_qid=owner, name="temperature") == 1

    plan = consumer_provider_plan(complete)
    assert [row["consumer_slot"] for row in plan] == [0, 1]
    assert [row["provider"]["slot"] for row in plan] == [17, 9]
    assert [row["key"]["component"] for row in plan] == ["potential", "temperature"]


def test_provider_pack_refuses_duplicate_unqualified_auxiliary_component_routes() -> None:
    owner = "model/demo"
    duplicate = ProviderPack((
        (ComponentKey(owner, "field", "a", "temperature"), _contract(),
         ProviderEntry("runtime_input", True, 0)),
        (ComponentKey(owner, "aux", "b", "temperature"), _contract(),
         ProviderEntry("runtime_input", True, 1)),
    ))

    with pytest.raises(ValueError, match="one owner-qualified storage route"):
        compact_auxiliary_provider_pack(duplicate)
