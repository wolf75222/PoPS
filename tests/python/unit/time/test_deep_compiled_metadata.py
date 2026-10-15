"""The compiler's detached boundary has no extra Python container-depth limit."""
from types import MappingProxyType

import pytest

from pops import Program
from pops._frozen_data import freeze_containers
from pops.time._program.detach import detach_compiled_program


def test_detach_deep_metadata_preserves_identity_and_immutable_containers():
    program = Program("deep_compiler_metadata")
    leaf = program.int64(7)
    payload = 7
    for index in range(1200):
        payload = {"branch": (index, payload)}
    program._new("int64", "int64_constant", (),
                 {**leaf.attrs, "value": 9, "metadata_probe": payload}, None, None)
    expected = program._ir_hash()
    detached = detach_compiled_program(program)
    assert detached._ir_hash() == expected
    assert detached._frozen and detached._compiled_detached
    frozen = detached._values[-1].attrs["metadata_probe"]
    for index in reversed(range(1200)):
        assert isinstance(frozen, MappingProxyType)
        assert frozen["branch"][0] == index
        frozen = frozen["branch"][1]
    assert frozen == 7
    assert all(value.prog is detached for value in detached._values)
    assert detached._values[0] is not leaf


@pytest.mark.parametrize("boundary", ["containers", "compiled_plan"])
def test_freeze_containers_keeps_leaf_identity_without_invoking_hooks(boundary):
    from pops.codegen._plans import _deep_freeze

    class Leaf:
        def freeze(self):
            raise AssertionError("container freezer must not invoke leaf protocols")

    leaf = Leaf()
    value = leaf
    for _ in range(2000):
        value = [value]
    frozen = (freeze_containers if boundary == "containers" else _deep_freeze)(value)
    for _ in range(2000):
        assert type(frozen) is tuple
        frozen = frozen[0]
    assert frozen is leaf


def test_container_cycles_reject_deterministically():
    source = []
    source.append(source)
    with pytest.raises(ValueError, match="cycle"):
        freeze_containers(source)


def test_freezing_does_not_alias_mutable_source_containers():
    shared = [1]
    source = {"a": shared, "b": shared}
    frozen = freeze_containers(source)
    assert frozen["a"] == frozen["b"] == (1,)
    assert frozen["a"] is not frozen["b"]
    shared.append(2)
    assert frozen["a"] == (1,)


def test_interaction_quadrature_walks_deep_shared_field_ancestry_without_recursion():
    from pops.codegen.program_interaction_exchanges import accepted_interaction_quadrature

    program = Program("deep_inventory_ancestry")
    value = program.int64(7)
    for _index in range(1600):
        value = program._new("int64", "structural_projection", (value, value), {}, None, None)
    # The empty inventory proof never dereferences the endpoint. The real graph
    # objects exercise canonicalization and shared DAG ancestry, not copied math.
    program._commits[object()] = value
    assert accepted_interaction_quadrature(program) == ()
