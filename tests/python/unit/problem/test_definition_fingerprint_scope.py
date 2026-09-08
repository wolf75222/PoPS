"""Canonical projections reuse identities without pinning mutable scientific sources."""
import pytest

from pops.model import DeclarationIndex, Module, OwnerKind
from pops.model.handles import Handle, OwnerPath
from pops.model.ownership import UnresolvedOwnershipError, _definition_fingerprint_scope
from pops.problem import Case
from pops.time import Program


def _owner(name="same"):
    owner = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, name)
    state = {"value": "a", "calls": 0}

    def fingerprint():
        state["calls"] += 1
        return "test:sha256:" + state["value"] * 64

    owner._bind_definition_fingerprint_provider(fingerprint)
    return owner, state


def test_nested_scope_computes_once_and_reauthenticates_on_exit():
    owner, state = _owner()
    with _definition_fingerprint_scope():
        first = owner.canonical()
        for _ in range(20):
            with _definition_fingerprint_scope():
                assert owner.canonical() == first
        assert state["calls"] == 1
    assert state["calls"] == 2
    assert owner.canonical() == first
    assert state["calls"] == 3


@pytest.mark.parametrize("exceptional", [False, True])
def test_mutation_is_detected_even_on_exceptional_exit_and_cache_is_discarded(exceptional):
    owner, state = _owner()
    with pytest.raises(UnresolvedOwnershipError, match="changed during canonical projection"):
        with _definition_fingerprint_scope():
            previous = owner.canonical()
            state["value"] = "b"
            if exceptional:
                raise RuntimeError("projection failed")
    assert state["calls"] == 2
    assert owner.canonical() != previous
    with _definition_fingerprint_scope():
        assert owner.canonical().definition_fingerprint == "test:sha256:" + "b" * 64


def test_unchanged_exception_is_preserved_and_still_reauthenticates():
    owner, state = _owner()
    failure = RuntimeError("projection failed")
    with pytest.raises(RuntimeError) as caught:
        with _definition_fingerprint_scope():
            owner.canonical()
            raise failure
    assert caught.value is failure
    assert state["calls"] == 2
    owner.canonical()
    assert state["calls"] == 3


def test_recursive_fingerprint_is_never_satisfied_from_a_partial_cache():
    owner, _ = _owner()
    owner._bind_definition_fingerprint_provider(lambda: owner.definition_fingerprint)
    with pytest.raises(UnresolvedOwnershipError, match="recursively depends"):
        with _definition_fingerprint_scope():
            owner.canonical()


def test_equal_names_keep_distinct_authority_and_foreign_handles_are_refused():
    owner, _ = _owner()
    foreign, state = _owner()
    state["value"] = "b"
    own_handle = Handle("U", kind="state", owner=owner)
    foreign_handle = Handle("U", kind="state", owner=foreign)
    index = DeclarationIndex(owner=owner, handles=(own_handle,))
    with _definition_fingerprint_scope():
        assert owner.canonical() != foreign.canonical()
        assert index.authenticate(own_handle) is own_handle
        with pytest.raises(ValueError):
            index.authenticate(foreign_handle)


def test_real_mutable_module_changes_are_visible_after_scope():
    module = Module("scope-mutation")
    module.state_space("U", ("u",))
    with pytest.raises(UnresolvedOwnershipError, match="changed during canonical projection"):
        with _definition_fingerprint_scope():
            before = module.owner_path.canonical()
            module.state_space("V", ("v",))
    assert module.owner_path.canonical() != before
    assert "V" in module.state_spaces()


def test_program_serialization_has_identical_bytes_with_bounded_hash_calls(monkeypatch):
    program = Program("fingerprint-reuse")
    module = Module("fluid")
    space = module.state_space("U", ("u",))
    case = Case("fingerprint-reuse")
    block = case.block("fluid", module)
    state = program.state(block[module.state_handle(space)])
    value = state.n
    for number in range(12):
        value = program.value("copy_%d" % number, value, at=state.next.point)
    program.commit(state.next, value)
    calls = []
    original = Module.module_hash

    def counted(module):
        calls.append(module)
        return original(module)

    monkeypatch.setattr(Module, "module_hash", counted)
    expected = program._serialize.__wrapped__(program)
    unscoped_calls = len(calls)
    calls.clear()
    actual = program._serialize()
    assert actual == expected
    assert len(calls) == 2  # One captured identity, one fresh exit authentication.
    assert unscoped_calls > len(calls)
    assert program._serialize() == expected
