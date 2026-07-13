"""ADC-652/653: stable Handle values plus authenticated owner qualification."""
from __future__ import annotations

import pytest

from pops.model import (
    AmbiguousReferenceError,
    DeclarationIndex,
    DoubleOwnershipError,
    Handle,
    IdentityCollisionError,
    MissingOwnershipError,
    Module,
    OperatorHandle,
    OperatorRegistry,
    OwnerKind,
    OwnerPath,
    OwnershipError,
    Signature,
    StateSpace,
    UnresolvedOwnershipError,
)
from pops.problem import Case
from pops.problem.handles import BlockHandle
from pops.problem.registries import BlockRegistry, FieldRegistry


class _DeclaredModel:
    def __init__(self, name: str = "transport") -> None:
        self.name = name
        self.owner_path = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, name)
        self.u = Handle("u", kind="state", owner=self.owner_path)

    def declaration_index(self) -> DeclarationIndex:
        return DeclarationIndex(owner=self.owner_path, handles=(self.u,))


def test_same_canonical_identity_is_boolean_equal_and_hash_compatible():
    owner = OwnerPath.model("transport")
    left = Handle("u", kind="state", owner=owner)
    right = Handle("u", kind="state", owner=OwnerPath.model("transport"))

    assert (left == right) is True
    assert hash(left) == hash(right)
    assert {left: "state"}[right] == "state"


def test_same_named_authoring_owners_are_distinct_until_resolution():
    first = _DeclaredModel("transport")
    second = _DeclaredModel("transport")

    assert first.owner_path != second.owner_path
    with pytest.raises(UnresolvedOwnershipError, match="definition fingerprint"):
        first.owner_path.canonical()
    first.declaration_index()
    second.declaration_index()
    assert first.owner_path.canonical() == second.owner_path.canonical()
    assert first.u != second.u
    assert len({first.u, second.u}) == 2
    assert first.u.inspect()["ownership_phase"] == "authoring"
    assert "#authoring=" not in first.u.inspect()["qualified_id"]
    with pytest.raises(UnresolvedOwnershipError, match="resolve"):
        first.u.canonical_identity()


def _state_module(name: str, components: tuple[str, ...]) -> tuple[Module, Handle]:
    module = Module(name)
    state = module.state_space("U", components=components)
    return module, module.state_handle(state)


def test_module_definition_fingerprint_is_reproducible_and_content_addressed():
    first, first_state = _state_module("transport", ("rho", "momentum"))
    repeated, repeated_state = _state_module("transport", ("rho", "momentum"))
    different, different_state = _state_module("transport", ("rho", "energy"))

    first_owner = first.owner_path.canonical()
    repeated_owner = repeated.owner_path.canonical()
    different_owner = different.owner_path.canonical()
    assert first.module_hash() == repeated.module_hash()
    assert first_owner == repeated_owner
    assert first_owner.definition_fingerprint.startswith("pops.module:sha256:")
    assert first_owner != different_owner
    assert first.module_hash() != different.module_hash()

    first_canonical = first.declaration_index().authenticate(first_state)._resolved()
    repeated_canonical = repeated.declaration_index().authenticate(repeated_state)._resolved()
    assert first_canonical == repeated_canonical
    assert first_canonical.qualified_id == repeated_canonical.qualified_id
    with pytest.raises(MissingOwnershipError, match="owned by"):
        different.declaration_index().authenticate(first_canonical)
    assert different.declaration_index().authenticate(different_state) is different_state


def test_same_local_name_in_different_canonical_owners_or_kinds_is_distinct():
    model_a = OwnerPath.model("a")
    model_b = OwnerPath.model("b")
    state_a = Handle("u", kind="state", owner=model_a)
    state_b = Handle("u", kind="state", owner=model_b)
    field_a = Handle("u", kind="field", owner=model_a)

    assert state_a != state_b
    assert state_a != field_a
    assert len({state_a, state_b, field_a}) == 3


def test_same_model_instantiated_twice_has_distinct_authenticated_handles():
    model = _DeclaredModel()
    problem = Case(name="transport")
    block_a = problem.block("a", model)
    block_b = problem.block("b", model)

    a_u = block_a[model.u]
    b_u = block_b[model.u]

    assert a_u != b_u
    assert a_u.declaration_ref is model.u
    assert b_u.declaration_ref is model.u
    assert a_u.block_ref is block_a
    assert problem.qualify(model.u, block=block_a) is a_u
    assert problem.qualify(a_u) is a_u


def test_qualification_rejects_ghost_foreign_double_and_ambiguous_references():
    model = _DeclaredModel()
    foreign = _DeclaredModel("foreign")
    problem = Case(name="transport")
    block_a = problem.block("a", model)
    block_b = problem.block("b", model)
    ghost = Handle("ghost", kind="state", owner=model.owner_path)

    with pytest.raises(MissingOwnershipError, match="not registered"):
        block_a[ghost]
    with pytest.raises(MissingOwnershipError, match="owned by"):
        block_a[foreign.u]
    with pytest.raises(DoubleOwnershipError, match="already block-qualified"):
        block_b[block_a[model.u]]
    with pytest.raises(AmbiguousReferenceError) as error:
        problem.qualify(model.u)
    assert str(block_a.instance_owner_path) in str(error.value)
    assert str(block_b.instance_owner_path) in str(error.value)


def test_consumer_validation_reports_ambiguous_unqualified_reference_candidates():
    from pops.output import OutputPolicy

    model = _DeclaredModel()
    problem = Case(name="transport")
    block_a = problem.block("a", model)
    block_b = problem.block("b", model)
    problem.output(OutputPolicy(fields=[model.u]))

    report = problem.validate_report()
    issue = next(
        item for item in report.issues
        if item.code == "runtime.ambiguous_declaration_reference")
    assert str(block_a.instance_owner_path) in issue.message
    assert str(block_b.instance_owner_path) in issue.message

    resolved_problem = Case(name="transport-resolved")
    resolved_block = resolved_problem.block("a", model)
    resolved_problem.output(OutputPolicy(fields=[resolved_block[model.u]]))
    assert resolved_problem.validate_report().ok


def test_same_named_model_authorities_collide_before_lowering():
    first = _DeclaredModel("transport")
    second = _DeclaredModel("transport")
    problem = Case(name="case")
    problem.block("a", first)
    with pytest.raises(IdentityCollisionError, match="same owner"):
        problem.block("b", second)


def test_same_named_different_model_definitions_are_distinct_block_owners():
    first, first_state = _state_module("transport", ("rho", "momentum"))
    second, second_state = _state_module("transport", ("rho", "energy"))
    problem = Case(name="case")

    first_block = problem.block("first", first)
    second_block = problem.block("second", second)
    first_instance = problem.resolve(first_block[first_state])
    second_instance = problem.resolve(second_block[second_state])

    assert first_block.model_owner_path != second_block.model_owner_path
    assert first_block.model_owner_path.canonical() != second_block.model_owner_path.canonical()
    assert first_instance.owner_path != second_instance.owner_path
    with pytest.raises(MissingOwnershipError, match="owned by"):
        second_block[first_state]


def test_resolved_instance_identity_round_trips_without_losing_origin():
    model = _DeclaredModel("m:/unicode-é")
    problem = Case(name="case/with:reserved")
    block = problem.block("block/β", model)
    resolved = problem.resolve(block[model.u])
    identity = resolved.canonical_identity()
    decoded = Handle.from_canonical_identity(identity)

    assert resolved.is_resolved
    assert decoded == resolved
    assert decoded.canonical_identity() == identity
    assert decoded.declaration_ref.local_id == "u"
    assert decoded.block_ref.local_id == "block/β"
    assert "#authoring=" not in str(resolved.owner_path)


def test_problem_reauthenticates_canonical_roundtrips_and_rejects_foreign_data():
    from pops.fields import FieldProblem

    model = _DeclaredModel("transport")
    problem = Case(name="case")
    block = problem.block("fluid", model)

    canonical_block = problem.resolve(block)
    decoded_block = Handle.from_canonical_identity(canonical_block.canonical_identity())
    assert isinstance(decoded_block, BlockHandle)
    assert decoded_block.model_owner_path == canonical_block.model_owner_path
    assert decoded_block.canonical_identity()["handle_type"] == "block"
    assert decoded_block._instance_registry is None
    assert problem.resolve(decoded_block).canonical_identity() == canonical_block.canonical_identity()

    case_field = problem.field(FieldProblem(name="phi"))
    canonical_field = problem.resolve(case_field)
    decoded_field = Handle.from_canonical_identity(canonical_field.canonical_identity())
    assert problem.resolve(decoded_field).canonical_identity() == canonical_field.canonical_identity()

    canonical_state = problem.resolve(block[model.u])
    decoded_state = Handle.from_canonical_identity(canonical_state.canonical_identity())
    assert problem.resolve(decoded_state).canonical_identity() == canonical_state.canonical_identity()

    decoded_declaration = Handle.from_canonical_identity(
        canonical_state.canonical_identity()["declaration_ref"])
    assert model.declaration_index().authenticate(decoded_declaration) is model.u
    assert problem.resolve(decoded_declaration).canonical_identity() == canonical_state.canonical_identity()

    wrong_schema = Handle(
        "u", kind="state", owner=model.owner_path.canonical(), schema_version=2)
    with pytest.raises(MissingOwnershipError, match="registry-authenticated identity"):
        problem.resolve(wrong_schema)

    foreign = Handle(
        "u", kind="state", owner=OwnerPath.model("foreign-model"))
    with pytest.raises(MissingOwnershipError, match="no block in this case instantiates"):
        problem.resolve(foreign)


def test_canonical_block_roundtrip_rejects_forged_or_erased_model_provenance():
    model, _ = _state_module("transport", ("rho", "momentum"))
    foreign, _ = _state_module("transport", ("rho", "energy"))
    problem = Case(name="case")
    block = problem.block("fluid", model)
    canonical = problem.resolve(block)

    identity = canonical.canonical_identity()
    assert identity["model_owner_path"] == model.owner_path.canonical().to_data()
    forged_identity = dict(identity)
    forged_identity["model_owner_path"] = foreign.owner_path.canonical().to_data()
    forged = Handle.from_canonical_identity(forged_identity)
    assert isinstance(forged, BlockHandle)
    with pytest.raises(MissingOwnershipError, match="not registered by this case"):
        problem.resolve(forged)

    erased_identity = {
        key: value
        for key, value in identity.items()
        if key not in {"handle_type", "model_owner_path"}
    }
    erased = Handle.from_canonical_identity(erased_identity)
    assert not isinstance(erased, BlockHandle)
    with pytest.raises(TypeError, match="BlockHandle"):
        problem.resolve(erased)


def test_explicit_shared_owner_is_not_reowned_by_a_block():
    model = _DeclaredModel()
    problem = Case(name="case")
    block = problem.block("a", model)
    shared = Handle("gravity", kind="param", owner=OwnerPath.shared("environment"))

    assert block[shared] is shared
    assert problem.resolve(shared) == shared


def test_owner_path_canonical_roundtrip_and_strict_schema():
    owner = (OwnerPath.case("case/:é")
             .child(OwnerKind.BLOCK, "b/β")
             .instance_of(OwnerPath.model("model:λ")))
    assert OwnerPath.from_data(owner.to_data()) == owner

    bad = owner.to_data()
    bad["schema_version"] = True
    with pytest.raises(TypeError, match="integer"):
        OwnerPath.from_data(bad)
    bad["schema_version"] = 1.0
    with pytest.raises(TypeError, match="integer"):
        OwnerPath.from_data(bad)


def test_owner_path_rejects_invalid_topology():
    with pytest.raises(Exception, match="transition"):
        OwnerPath.case("c").child(OwnerKind.MODEL_DEFINITION, "m")


def test_handle_schema_version_is_part_of_identity():
    owner = OwnerPath.model("transport")
    one = Handle("u", kind="state", owner=owner, schema_version=1)
    two = Handle("u", kind="state", owner=owner, schema_version=2)
    assert one != two
    assert one.qualified_id.startswith("pops.handle.v1::")
    assert two.qualified_id.startswith("pops.handle.v2::")


def test_handle_identity_and_operator_metadata_are_immutable():
    handle = OperatorHandle(
        "advect", kind="local_rate", owner=OwnerPath.model("transport"), category="rate")
    for attribute, replacement in (
        ("local_id", "other"),
        ("owner_path", OwnerPath.model("other")),
        ("kind", "local_source"),
        ("category", "source"),
    ):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(handle, attribute, replacement)


def test_operator_metadata_does_not_change_value_identity():
    owner = OwnerPath.model("transport")
    u = StateSpace("U", ("rho",))
    v = StateSpace("V", ("tracer",))
    first = OperatorHandle(
        "advance", kind="local_rate", owner=owner,
        signature=Signature((u,), u), category="rate")
    second = OperatorHandle(
        "advance", kind="local_rate", owner=owner,
        signature=Signature((v,), v), category="custom")
    assert first == second
    assert hash(first) == hash(second)


def test_owner_is_required_and_never_stringified():
    with pytest.raises(TypeError):
        Handle("u", kind="state")
    with pytest.raises(TypeError, match="OwnerPath"):
        Handle("u", kind="state", owner="bare")
    with pytest.raises(TypeError, match="OwnerPath"):
        BlockRegistry("bare")


def test_canonical_owner_identifies_data_but_cannot_authorize_mutable_registries():
    model_owner = OwnerPath.model("transport")
    case_owner = OwnerPath.case("case")

    # Immutable resolved references are intentionally reconstructible from canonical data.
    assert Handle("u", kind="state", owner=model_owner).is_resolved

    with pytest.raises(UnresolvedOwnershipError, match="canonical owner"):
        Module("transport", owner=model_owner)
    with pytest.raises(UnresolvedOwnershipError, match="canonical owner"):
        OperatorRegistry(owner=model_owner)
    with pytest.raises(UnresolvedOwnershipError, match="canonical owner"):
        BlockRegistry(case_owner)
    with pytest.raises(UnresolvedOwnershipError, match="canonical owner"):
        FieldRegistry(case_owner)


def test_mutable_registry_authorities_require_exact_root_kind_and_matching_name():
    case_owner = OwnerPath.fresh(OwnerKind.CASE, "case")
    model_owner = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, "transport")
    nested_case_owner = case_owner.child(OwnerKind.DESCRIPTOR, "nested")
    nested_model_owner = model_owner.child(OwnerKind.DESCRIPTOR, "nested")

    with pytest.raises(OwnershipError, match="root model_definition"):
        Module("case", owner=case_owner)
    with pytest.raises(OwnershipError, match="root model_definition"):
        Module("transport", owner=nested_model_owner)
    with pytest.raises(OwnershipError, match="does not match declaration name"):
        Module("other", owner=model_owner)
    with pytest.raises(OwnershipError, match="root model_definition"):
        OperatorRegistry(owner=case_owner)
    with pytest.raises(OwnershipError, match="root model_definition"):
        OperatorRegistry(owner=nested_model_owner)
    with pytest.raises(OwnershipError, match="root case"):
        BlockRegistry(model_owner)
    with pytest.raises(OwnershipError, match="root case"):
        BlockRegistry(nested_case_owner)
    with pytest.raises(OwnershipError, match="root case"):
        FieldRegistry(nested_case_owner)


@pytest.mark.parametrize(
    ("owner", "error", "message"),
    [
        (OwnerPath.model("transport"), UnresolvedOwnershipError, "canonical owner"),
        (OwnerPath.fresh(OwnerKind.CASE, "transport"), OwnershipError,
         "root model_definition"),
        (OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, "transport").child(
            OwnerKind.DESCRIPTOR, "nested"), OwnershipError, "root model_definition"),
        (OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, "other"), OwnershipError,
         "does not match declaration name"),
    ],
)
def test_block_registry_rejects_non_authoritative_model_owner(owner, error, message):
    class ModelWithOwner:
        name = "transport"
        owner_path = owner

    registry = BlockRegistry(OwnerPath.fresh(OwnerKind.CASE, "case"))
    with pytest.raises(error, match=message):
        registry.add("fluid", ModelWithOwner())
    assert registry.names() == []


def test_problem_and_registries_share_one_authoring_authority():
    problem = Case(name="transport")
    for value in (problem, problem._block_registry, problem._field_registry):
        assert value.owner_path == problem.owner_path
        with pytest.raises(AttributeError):
            value.owner_path = OwnerPath.case("other")


@pytest.mark.parametrize("registry_type", [BlockRegistry, FieldRegistry])
def test_problem_registry_owner_is_structural(registry_type):
    with pytest.raises((TypeError, ValueError)):
        registry_type(object())


@pytest.mark.parametrize("name", ["", 3, object()])
def test_problem_rejects_invalid_names_before_owner_allocation(name):
    with pytest.raises(TypeError, match="non-empty string"):
        Case(name=name)
