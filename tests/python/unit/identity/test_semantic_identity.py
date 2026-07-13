from __future__ import annotations

from pops.identity import Identity, canonical_sha256
from pops.identity.semantic import semantic_identity, semantic_identity_of
from pops.descriptors import Descriptor
from pops.mesh import CartesianMesh
from pops.layouts import Uniform
from pops.model import Module
from pops.problem import Case
from pops.problem._snapshot import AuthoringSnapshot, build_authoring_snapshot


def _snapshot(*, representation="conservative", centering="cell", units=None,
              frame="model", clock="simulation", roles=None, layout=None):
    module = Module("transport")
    module.state_space(
        "U", ("rho",), roles=roles or {"density": "rho"},
        representation=representation, centering=centering, units=units,
        frame=frame, clock=clock,
    )
    problem = Case(name="case")
    problem.block("fluid", module)
    selected_layout = layout or Uniform(CartesianMesh(n=16, L=1.0))
    return build_authoring_snapshot(problem, layout=selected_layout, time=None)


class _ForeignLayout(Descriptor):
    """Test-only extension: accepted without adding a branch to the snapshotter."""

    category = "layout"

    def __init__(self, mesh):
        self.mesh = mesh

    def options(self):
        return {"mesh": self.mesh, "partition": "third-party"}


def test_authoring_snapshot_exposes_exact_typed_semantic_identity():
    snapshot = _snapshot()
    assert isinstance(snapshot.semantic_identity, Identity)
    assert snapshot.semantic_identity.domain == "semantic"
    assert semantic_identity(snapshot.semantic_to_dict()) == snapshot.semantic_identity
    assert semantic_identity_of(snapshot=snapshot) == snapshot.semantic_identity
    assert snapshot.hash == canonical_sha256(snapshot.to_dict())


def test_space_physics_and_owner_are_semantic():
    baseline = _snapshot()
    changes = [
        _snapshot(representation="primitive"),
        _snapshot(centering="face"),
        _snapshot(frame="laboratory"),
        _snapshot(clock="material"),
        _snapshot(roles={"mass": "rho"}),
    ]
    assert all(item.semantic_identity != baseline.semantic_identity for item in changes)
    assert _snapshot().semantic_identity == baseline.semantic_identity


def test_opaque_space_units_are_rejected_before_semantic_identity():
    import pytest

    with pytest.raises(TypeError, match="Space units are unsupported"):
        _snapshot(units=("kg/m3",))


def test_mapping_insertion_order_is_not_semantic():
    left = semantic_identity({"outer": {"a": 1, "b": 2}})
    right = semantic_identity({"outer": {"b": 2, "a": 1}})
    assert left == right


def test_third_party_layout_uses_the_descriptor_semantic_protocol() -> None:
    baseline = _snapshot(layout=_ForeignLayout(CartesianMesh(n=8, L=1.0)))
    equivalent = _snapshot(layout=_ForeignLayout(CartesianMesh(n=8, L=1.0)))
    changed = _snapshot(layout=_ForeignLayout(CartesianMesh(n=16, L=1.0)))

    assert baseline.semantic_identity == equivalent.semantic_identity
    assert baseline.semantic_identity != changed.semantic_identity


def test_artifact_hash_is_not_the_semantic_identity_alias():
    snapshot = AuthoringSnapshot(
        {"provenance": "exact"},
        artifact_payload={"lowering": "legacy"},
        semantic_payload={"science": "transport"},
    )
    assert snapshot.semantic_identity.hexdigest != snapshot.artifact_hash
    assert snapshot.semantic_identity == semantic_identity({"science": "transport"})
