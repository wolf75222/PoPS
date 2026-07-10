"""ADC-652: declaration handles have stable, owner-qualified value identity."""
from __future__ import annotations

import pytest

from pops.model import Handle, OperatorHandle, OwnerPath, Signature, StateSpace
from pops.problem import Problem
from pops.problem.registries import BlockRegistry, FieldRegistry


def test_same_qualified_identity_is_boolean_equal_and_hash_compatible():
    owner = OwnerPath("case", "transport", "model")
    left = Handle("u", kind="state", owner=owner)
    right = Handle("u", kind="state", owner=OwnerPath("case", "transport", "model"))

    comparison = left == right

    assert comparison is True
    assert hash(left) == hash(right)
    assert {left: "state"}[right] == "state"


def test_same_local_name_in_different_owners_or_kinds_is_distinct():
    model_a = OwnerPath("case", "a", "model")
    model_b = OwnerPath("case", "b", "model")

    state_a = Handle("u", kind="state", owner=model_a)
    state_b = Handle("u", kind="state", owner=model_b)
    field_a = Handle("u", kind="field", owner=model_a)

    assert state_a != state_b
    assert state_a != field_a
    assert len({state_a, state_b, field_a}) == 3
    assert state_a.qualified_id != state_b.qualified_id


def test_handle_schema_version_is_part_of_the_qualified_identity():
    owner = OwnerPath("case", "transport")
    version_one = Handle("u", kind="state", owner=owner, schema_version=1)
    version_two = Handle("u", kind="state", owner=owner, schema_version=2)

    assert version_one != version_two
    assert version_one.qualified_id.startswith("pops.handle.v1::")
    assert version_two.qualified_id.startswith("pops.handle.v2::")


def test_owner_path_is_structural_not_a_flat_string_key():
    split_after_first = Handle("u", kind="state", owner=OwnerPath("a", "b/c"))
    split_after_second = Handle("u", kind="state", owner=OwnerPath("a/b", "c"))

    assert split_after_first != split_after_second
    assert split_after_first.owner_path.segments == ("a", "b/c")
    assert split_after_second.owner_path.segments == ("a/b", "c")


def test_legitimate_authoring_named_segment_never_collides_with_fresh_owner_token():
    user_owner = OwnerPath("case", "authoring-prod")
    fresh_owner = OwnerPath.fresh("case", "authoring-prod")

    assert user_owner != fresh_owner
    assert user_owner.canonical_declaration_path() == fresh_owner.canonical_declaration_path()
    assert "#authoring=" not in str(user_owner)
    assert "#authoring=" in str(fresh_owner)


def test_handle_identity_and_metadata_are_immutable():
    handle = OperatorHandle(
        "advect",
        kind="local_rate",
        owner=OwnerPath("model", "transport"),
        category="rate",
    )

    for attribute, replacement in (
        ("local_id", "other"),
        ("owner_path", OwnerPath("model", "other")),
        ("kind", "local_source"),
        ("category", "source"),
    ):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(handle, attribute, replacement)

    with pytest.raises(AttributeError, match="immutable"):
        del handle.local_id


def test_operator_metadata_does_not_change_qualified_identity():
    owner = OwnerPath("model", "transport")
    u = StateSpace("U", ("rho",))
    v = StateSpace("V", ("tracer",))
    first = OperatorHandle(
        "advance", kind="local_rate", owner=owner, signature=Signature((u,), u), category="rate")
    second = OperatorHandle(
        "advance", kind="local_rate", owner=owner, signature=Signature((v,), v), category="custom")

    assert first == second
    assert hash(first) == hash(second)
    assert first.inspect()["qualified_id"] == second.inspect()["qualified_id"]

    with pytest.raises(TypeError, match="Signature"):
        OperatorHandle("bad", kind="local_rate", owner=owner, signature=[])


def test_owner_is_required_instead_of_silently_falling_back_to_a_global_namespace():
    with pytest.raises(TypeError):
        Handle("u", kind="state")
    with pytest.raises(TypeError):
        OperatorHandle("advance", kind="local_rate")
    with pytest.raises(TypeError, match="OwnerPath"):
        Handle("u", kind="state", owner="bare")
    with pytest.raises(TypeError, match="OwnerPath"):
        BlockRegistry("bare")


@pytest.mark.parametrize("segments", [(object(),), ("model", 3), ("",)])
def test_owner_path_rejects_unstable_non_string_segments(segments):
    with pytest.raises(ValueError, match="string segments"):
        OwnerPath(*segments)


@pytest.mark.parametrize(("local_id", "kind"), [(object(), "state"), ("u", object())])
def test_handle_rejects_implicit_stringification(local_id, kind):
    with pytest.raises(ValueError, match="non-empty string"):
        Handle(local_id, kind=kind, owner=OwnerPath("model", "transport"))


def test_problem_and_its_handle_registries_expose_read_only_owner_anchors():
    problem = Problem(name="transport")
    registries = (problem._block_registry, problem._field_registry)

    for value in (problem, *registries):
        assert value.owner_path == problem.owner_path
        with pytest.raises(AttributeError):
            value.owner_path = OwnerPath("problem", "other")


@pytest.mark.parametrize("registry_type", [BlockRegistry, FieldRegistry])
def test_problem_registry_owner_is_structural_and_never_stringified(registry_type):
    with pytest.raises((TypeError, ValueError)):
        registry_type(object())


@pytest.mark.parametrize("name", ["", 3, object()])
def test_problem_rejects_invalid_names_before_allocating_an_owner(name):
    with pytest.raises(TypeError, match="non-empty string"):
        Problem(name=name)
