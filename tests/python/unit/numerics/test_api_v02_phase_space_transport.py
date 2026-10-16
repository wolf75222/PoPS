"""Products, independent kinetic conservation, and source graph closure."""

import numpy as np
import pops
import pytest

from examples.api_v02.models_kinetic_nd import build_kinetic_nd
from examples.api_v02.phase_space_transport import TensorGrid, partition
from tests.python.support.kinetic_nd_oracle import kinetic_reference, coordinates


def test_six_dimensional_product_is_a_bijection_with_axis_neighbours():
    grid = TensorGrid((2, 3, 4, 3, 4, 5), ((0.0, 1.0),) * 6, ("periodic",) * 6)
    for gid in range(grid.size):
        coordinates0 = tuple(grid.ordinal(gid, a) for a in range(6))
        assert sum(i * s for i, s in zip(coordinates0, grid.strides, strict=True)) == gid
        for axis in range(6):
            shifted = grid.periodic_neighbour(gid, axis, 1)
            assert grid.periodic_neighbour(shifted, axis, -1) == gid
            assert all(grid.ordinal(shifted, b) == coordinates0[b] for b in range(6) if b != axis)
    assert partition(4, 7) == (0, 0, 1, 1, 2, 2, 3, 4)


@pytest.mark.parametrize(
    "model,masses,charges", [("M19", (1.0,), (-1.0,)), ("M20", (1.0, 1.5, 2.0), (-1.0, 1.0, 1.0))]
)
def test_independent_3x3v_reference_preserves_invariants(model, masses, charges):
    shape = (3, 3, 3)
    velocity = (4, 4, 4)
    initial, result, diagnostics = kinetic_reference(
        model=model,
        configuration_shape=shape,
        velocity_shape=velocity,
        masses=masses,
        charges=charges,
    )
    assert result.min() > 0
    assert np.max(abs(result - initial)) > 1e-8
    change = (result - initial).reshape(len(masses), 27, 64)
    np.testing.assert_allclose(change.sum(axis=(1, 2)), 0, atol=1e-14, rtol=0)
    if model == "M19":
        assert all(amplitude > 1e-5 for amplitude in diagnostics[0]["field_amplitudes"])
    else:
        nodes = coordinates(velocity, [(-4.0, 4.0)] * 3).reshape(64, 3)
        momentum = sum(
            m * np.einsum("xv,vd->d", row, nodes) for m, row in zip(masses, change, strict=True)
        )
        energy = sum(
            0.5 * m * np.einsum("xv,v->", row, (nodes * nodes).sum(axis=1))
            for m, row in zip(masses, change, strict=True)
        )
        np.testing.assert_allclose(momentum, 0, atol=1e-14, rtol=0)
        assert abs(energy) < 1e-14


def test_polynomial_collision_projection_does_not_claim_global_positivity():
    """The real candidate assessment must reject this finite-method domain exit."""
    initial, result, _ = kinetic_reference(
        model="M20",
        configuration_shape=(3, 3, 3),
        velocity_shape=(4, 4, 4),
        masses=(1.0, 2.0, 3.0),
        charges=(-1.0, 1.0, 1.0),
        dt=1e-5,
        steps=2,
    )
    assert initial.min() > 0
    assert result.min() < 0


def test_collision_projection_has_a_native_scope_before_transport():
    """A full three-species step must not retain the collision workspace to publication."""
    spec = build_kinetic_nd(
        model="M20", configuration_shape=(3, 3, 3), velocity_shape=(4, 4, 4),
        masses=(1.0, 1.5, 2.0), charges=(-1.0, 1.0, 1.0),
    )
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert not plan.blocks
    encoded = spec.case._time._serialize(include_provenance=False)
    scopes = [node for node in encoded["nodes"] if node["op"] == "structured_fold"]
    assert len(scopes) == 1
    scope = scopes[0]
    assert [name for name, _ in scope["attrs"]["signature"]["members"]] == ["f0", "f1", "f2"]
    body = scope["attrs"]["body_block"]
    assert sum(node["op"] == "structured_fold" for node in body) == 3
    assert any(node["op"] == "distributed_scatter_reduce" for node in body)
    # The existing native receiver compares all cells to the independent oracle
    # with the original budget and tolerances; this check receives lexical scope.
    from pops.codegen.program_codegen import emit_cpp_program

    cpp = emit_cpp_program(spec.case._time)
    assert "for (; iteration_%d" % scope["id"] in cpp
    assert "ctx.continuation_publish(" in cpp


def test_kinetic_resource_capacity_reaches_schema_and_every_tensor_operation():
    capacity = 1 << 30
    spec = build_kinetic_nd(configuration_shape=(3,), velocity_shape=(4,),
                            byte_capacity=capacity, storage_bytes=2 * capacity,
                            storage_versions=131072)
    data = spec.case._time._serialize(include_provenance=False)
    budgets, storage_bytes, storage_versions = [], [], []
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "byte_capacity" in node:
                budgets.append(node["byte_capacity"])
            if "storage_bytes" in node:
                storage_bytes.append(node["storage_bytes"])
            if "storage_versions" in node:
                storage_versions.append(node["storage_versions"])
            if "distributed_storage_bytes" in node:
                storage_bytes.append(node["distributed_storage_bytes"])
            if "distributed_storage_versions" in node:
                storage_versions.append(node["distributed_storage_versions"])
            stack.extend(node.values())
        elif isinstance(node, (tuple, list)):
            stack.extend(node)
    assert budgets and set(budgets) == {capacity}
    assert storage_bytes and set(storage_bytes) == {2 * capacity}
    assert storage_versions and set(storage_versions) == {131072}
    with pytest.raises((TypeError, ValueError), match="capacity"):
        build_kinetic_nd(configuration_shape=(3,), velocity_shape=(4,), byte_capacity=True)


@pytest.mark.parametrize("resource", ("storage_bytes", "storage_versions"))
def test_kinetic_storage_policy_rejects_non_integer_resources(resource):
    with pytest.raises((TypeError, ValueError), match="capacity"):
        build_kinetic_nd(configuration_shape=(3,), velocity_shape=(4,), **{resource: True})
