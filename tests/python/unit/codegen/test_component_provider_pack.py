"""ADC-658: exact immutable typed component/provider lowering metadata."""
from types import SimpleNamespace

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
    build_provider_pack,
    build_operator_provider_pack,
)


def _row(component="rho", slot=0, *, producer="initial", available=True):
    key = ComponentKey("owner", "state", "U", component)
    contract = ComponentContract("conservative", "cell", "kg/m3", "cell")
    return key, contract, ProviderEntry(producer, available, slot)


def _board_bound_field_module(*, legacy_input=False):
    """Build a pure-Python Board-shaped field binding and its legacy carrier."""
    from pops.domain import CartesianDomain
    from pops.fields import FieldOutput, GradientOutput
    from pops.math import laplacian
    from pops.physics import Model

    frame = CartesianDomain("provider-pack", (0.0, 0.0), (1.0, 1.0)).frame()
    model = Model("provider_pack_board", frame=frame)
    (charge,) = model.state("U", components=("charge",))
    if legacy_input:
        model.aux("external_input")
    potential = model.field("potential")
    model.field_operator(
        "electrostatic",
        unknown=potential,
        equation=(-laplacian(potential) == charge),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("electric", potential),
        ),
    )
    return model.module, potential


def _board_two_bound_field_module():
    """Build two Board field bindings sharing the one legacy aggregate carrier."""
    from pops.domain import CartesianDomain
    from pops.fields import FieldOutput, GradientOutput
    from pops.math import laplacian
    from pops.physics import Model

    frame = CartesianDomain("provider-pack-two", (0.0, 0.0), (1.0, 1.0)).frame()
    model = Model("provider_pack_board_two", frame=frame)
    (charge,) = model.state("U", components=("charge",))
    model.aux("external_input")
    potential = model.field("potential")
    magnetic_potential = model.field("magnetic_potential")
    model.field_operator(
        "electrostatic",
        unknown=potential,
        equation=(-laplacian(potential) == charge),
        outputs=(FieldOutput("potential", potential), GradientOutput("electric", potential)),
    )
    model.field_operator(
        "magnetostatic",
        unknown=magnetic_potential,
        equation=(-laplacian(magnetic_potential) == charge),
        outputs=(
            FieldOutput("magnetic_potential", magnetic_potential),
            GradientOutput("magnetic", magnetic_potential),
        ),
    )
    return model.module, potential, magnetic_potential


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


def test_minimal_selection_preserves_qualified_identity_and_refuses_missing_provider():
    contract = ComponentContract("field", "cell", "V/m", "cell")
    left = ComponentKey("case/left", "field", "electric", "grad_x")
    right = ComponentKey("case/right", "field", "electric", "grad_x")
    pack = ProviderPack([
        (left, contract, ProviderEntry("left_solver", True, 0)),
        (right, contract, ProviderEntry("right_solver", True, 0)),
    ])
    selected = pack.select([(left, contract)])
    assert list(selected) == [left]
    assert selected[left].producer == "left_solver"
    with pytest.raises(MissingInputProvider):
        selected[right]
    with pytest.raises(MissingInputProvider):
        pack.select([ComponentKey("case/missing", "field", "electric", "grad_x")])


def test_component_selection_is_space_qualified_and_refuses_homonym_ambiguity():
    contract = ComponentContract("field", "cell", "V/m", "cell")
    left = ComponentKey("owner", "field", "left", "grad_x")
    right = ComponentKey("owner", "field", "right", "grad_x")
    pack = ProviderPack([
        (left, contract, ProviderEntry("left_solver", True, 0)),
        (right, contract, ProviderEntry("right_solver", True, 0)),
    ])

    selected = pack.select_components(
        owner_qid="owner",
        spaces=(("field", "left"),),
        components=("grad_x",),
    )
    assert tuple(selected) == (left,)
    with pytest.raises(MissingInputProvider, match="ambiguous component"):
        pack.select_components(
            owner_qid="owner",
            spaces=(("field", "left"), ("field", "right")),
            components=("grad_x",),
        )


def test_operator_provider_pack_contains_fields_but_not_explicit_state_trace():
    module = Module("operator_pack")
    state = module.state_space("U", ("rho",))
    fields = module.field_space("electric", ("phi", "grad_x", "grad_y"))
    module.operator("solve", state >> fields, "field_operator", expr=1.0)
    operator = SimpleNamespace(signature=SimpleNamespace(inputs=(state, fields)))
    pack = build_operator_provider_pack(module, operator)
    assert {(key.space_kind, key.space_name, key.component) for key in pack} == {
        ("field", "electric", "phi"),
        ("field", "electric", "grad_x"),
        ("field", "electric", "grad_y"),
    }


def test_operator_requirements_select_only_declared_components():
    module = Module("operator_component_pack")
    state = module.state_space("U", ("rho",))
    fields = module.field_space("electric", ("phi", "grad_x", "grad_y"))
    module.operator("solve", state >> fields, "field_operator", expr=1.0)
    operator = SimpleNamespace(
        signature=SimpleNamespace(inputs=(state, fields)),
        requirements={"aux": ("grad_x",)},
    )

    pack = build_operator_provider_pack(module, operator)

    assert tuple(key.component for key in pack) == ("grad_x",)


def test_board_field_binding_projects_exact_operator_output_once():
    module, potential = _board_bound_field_module()
    pack = build_provider_pack(module)
    target = module.operator_binding(potential)._resolved(module.owner_path.canonical()).qualified_id
    claimed = [
        key for key in pack
        if key.space_kind == "field" and key.space_name == "potential"
    ]

    assert tuple(key.component for key in claimed) == ("potential", "electric_x", "electric_y")
    assert {pack[key].producer for key in claimed} == {target}
    assert {pack[key].slot for key in claimed} == {0, 1, 2}
    assert not any(
        key.space_kind == "field"
        and key.space_name == "fields"
        and key.component in {"potential", "electric_x", "electric_y"}
        for key in pack
    )
    assert len([key for key in pack if key.component == "potential"]) == 1
    assert len([key for key in pack if key.component == "electric_x"]) == 1
    assert len([key for key in pack if key.component == "electric_y"]) == 1


def test_board_field_binding_preserves_unclaimed_legacy_input():
    module, _ = _board_bound_field_module(legacy_input=True)
    pack = build_provider_pack(module)
    key = ComponentKey(
        str(module.owner_path.canonical()), "field", "fields", "external_input"
    )

    assert pack[key] == ProviderEntry("runtime_input", True, 0)
    assert [
        (field_key.space_name, field_key.component, pack[field_key].slot)
        for field_key in pack
        if field_key.space_kind == "field"
    ] == [
        ("fields", "external_input", 0),
        ("potential", "potential", 1),
        ("potential", "electric_x", 2),
        ("potential", "electric_y", 3),
    ]


def test_two_board_field_bindings_share_carrier_without_alias_or_legacy_duplicates():
    module, potential, magnetic_potential = _board_two_bound_field_module()
    pack = build_provider_pack(module)
    bindings = module.operator_bindings()
    expected_producers = {
        subject.local_id: target._resolved(module.owner_path.canonical()).qualified_id
        for subject, target in bindings.items()
        if subject in {potential, magnetic_potential}
    }

    assert module.operator_registry().aliases() == {}
    for subject, components in {
        "potential": ("potential", "electric_x", "electric_y"),
        "magnetic_potential": ("magnetic_potential", "magnetic_x", "magnetic_y"),
    }.items():
        keys = [
            key for key in pack
            if key.space_kind == "field" and key.space_name == subject
        ]
        assert tuple(key.component for key in keys) == components
        assert {pack[key].producer for key in keys} == {expected_producers[subject]}
        assert not any(
            key.space_kind == "field"
            and key.space_name == "fields"
            and key.component in set(components)
            for key in pack
        )


def test_board_field_binding_keeps_operator_requirements_component_resolvable():
    module, _ = _board_bound_field_module()
    fields = module.field_spaces()["fields"]
    operator = SimpleNamespace(
        name="consumer",
        signature=SimpleNamespace(inputs=(fields,)),
        requirements={"aux": ("electric_x",)},
    )

    pack = build_operator_provider_pack(module, operator)

    assert tuple((key.space_name, key.component) for key in pack) == (("potential", "electric_x"),)


def test_direct_module_field_space_without_binding_is_unchanged():
    module = Module("direct_provider_pack")
    state = module.state_space("U", ("rho",))
    fields = module.field_space("electric", ("phi",))
    module.operator("solve", state >> fields, "field_operator", expr=1.0)

    pack = build_provider_pack(module)
    key = ComponentKey(str(module.owner_path.canonical()), "field", "electric", "phi")

    assert pack[key].producer == module.operator_handle("solve")._resolved(
        module.owner_path.canonical()
    ).qualified_id


def test_board_field_binding_refuses_invalid_ambiguous_and_duplicate_claims():
    module, potential = _board_bound_field_module()
    from pops.model import Rate

    state = module.state_spaces()["U"]
    module.operator(
        "not_a_field_provider",
        (state,) >> Rate(state),
        "local_source",
        expr="invalid binding target",
    )
    module._operator_bindings[potential] = module.operator_handle("not_a_field_provider")
    with pytest.raises(ValueError, match="invalid field operator binding"):
        build_provider_pack(module)

    module, _ = _board_bound_field_module()
    from pops.model import FieldSpace

    carrier = module.field_spaces()["fields"]
    module._field_spaces["duplicate"] = FieldSpace("duplicate", carrier.components)
    with pytest.raises(ValueError, match="ambiguous storage carriers"):
        build_provider_pack(module)

    module, potential = _board_bound_field_module()
    module._operator_bindings[module.field_handle(module.field_spaces()["fields"])] = (
        module.operator_binding(potential)
    )
    with pytest.raises(ValueError, match="claim legacy carrier component .* more than once"):
        build_provider_pack(module)


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
    state = module.state_space("U", ("rho",))
    # A scalar output is rank-independent.  Gradient outputs require the
    # resolved mesh/frame; a raw Module declaration must not silently assume
    # two Cartesian axes from legacy phi/grad spellings.
    fields = module.field_space("electric", ("electric_potential",))
    module.operator("solve_electric", state >> fields, "field_operator", expr=1.0)
    lowered = _module_to_model(module)
    pack = lowered._component_provider_pack
    rows = pack.to_data()["entries"]
    potential = next(row for row in rows if row["key"]["component"] == "electric_potential")
    assert potential["key"]["space_kind"] == "field"
    assert potential["key"]["space_name"] == "electric"
    assert potential["contract"]["unit"] is None
    assert "solve_electric" in potential["provider"]["producer"]
    assert potential["provider"]["availability"] is True


@pytest.mark.parametrize(
    "declare",
    (
        lambda module: module.state_space("U", ("rho",), units=("kg/m3",)),
        lambda module: module.field_space("electric", ("phi",), units=("V",)),
    ),
)
def test_module_spaces_refuse_opaque_units_until_a_typed_protocol_exists(declare):
    with pytest.raises(TypeError, match="Space units are unsupported"):
        declare(Module("opaque-units"))


def test_same_component_spelling_in_distinct_typed_spaces_never_merges():
    module = Module("collision")
    module.state_space("U", ("rho",))
    module.field_space("left", ("phi",))
    module.field_space("right", ("phi",))
    with pytest.raises(
        ValueError,
        match=(
            "typed components field/left/phi and field/right/phi both lower to legacy aux name "
            "'phi'; the spaces are distinct and cannot be merged silently"
        ),
    ):
        _module_to_model(module)


@pytest.mark.parametrize("count", [2, 4])
def test_module_field_operator_rejects_unsupported_output_arity(count):
    module = Module("bad_arity_%d" % count)
    state = module.state_space("U", ("rho",))
    fields = module.field_space("fields", tuple("f%d" % i for i in range(count)))
    module.operator("solve", state >> fields, "field_operator", expr=1.0)
    with pytest.raises(ValueError, match="gradient outputs require an exact 1D/2D/3D frame"):
        _module_to_model(module)
