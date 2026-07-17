"""Generic two-block interface endpoint ownership preflight."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.mesh.boundaries import BlockInterfaceSide, ConservativeInterface
from pops.mesh.boundaries.composition import compose_shared_interfaces
from pops.model import OwnerPath
from pops.numerics import DiscretizationPlan
from pops.problem.handles import BlockHandle, StateHandle


class _Authority:
    def __init__(self, name, claims):
        self.name = name
        self._claims = claims

    def to_data(self):
        return {"schema_version": 1, "name": self.name}

    def interface_endpoint_claims(self):
        return self._claims

    def compose_resolved_blocks(self, blocks, layout_plan):  # pragma: no cover - preflight fails
        raise AssertionError("endpoint preflight must precede composition")


class _OtherAuthority(_Authority):
    """Distinct extension class proving collisions are protocol-wide."""


def _claim(block, boundary, **extra):
    return {
        "schema_version": 1,
        "block": block,
        "boundary": boundary.canonical_identity(),
        "level": 0,
        **extra,
    }


def _block(name, identity, interfaces):
    numerics = SimpleNamespace(
        block=SimpleNamespace(qualified_id=identity), interfaces=tuple(interfaces))
    return SimpleNamespace(name=name, numerics=numerics)


def _boundaries():
    frame = Rectangle(
        "claims", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    return frame.boundaries


def test_two_extension_classes_cannot_claim_the_same_block_face():
    boundaries = _boundaries()
    first = _Authority("first", (
        _claim("case::block::a", boundaries.x_max),
        _claim("case::block::b", boundaries.x_min),
    ))
    second = _OtherAuthority("second", (
        _claim("case::block::a", boundaries.x_max),
        _claim("case::block::c", boundaries.x_min),
    ))
    blocks = (
        _block("a", "case::block::a", (first, second)),
        _block("b", "case::block::b", (first,)),
        _block("c", "case::block::c", (second,)),
    )
    with pytest.raises(ValueError, match="same block boundary endpoint"):
        compose_shared_interfaces(blocks, layout_plan=object())


def test_endpoint_claim_schema_is_exact_and_rejects_extra_keys():
    boundaries = _boundaries()
    malformed = _Authority("malformed", (
        _claim("case::block::a", boundaries.x_max, alias="bypass"),
        _claim("case::block::b", boundaries.x_min),
    ))
    blocks = (
        _block("a", "case::block::a", (malformed,)),
        _block("b", "case::block::b", (malformed,)),
    )
    with pytest.raises(TypeError, match="exact canonical v1"):
        compose_shared_interfaces(blocks, layout_plan=object())


def test_claimed_blocks_must_equal_the_two_registration_owners():
    boundaries = _boundaries()
    authority = _Authority("foreign", (
        _claim("case::block::a", boundaries.x_max),
        _claim("case::block::c", boundaries.x_min),
    ))
    blocks = (
        _block("a", "case::block::a", (authority,)),
        _block("b", "case::block::b", (authority,)),
    )
    with pytest.raises(ValueError, match="registered numerical blocks"):
        compose_shared_interfaces(blocks, layout_plan=object())


def test_two_plan_attach_preflights_both_destinations_before_mutation():
    authority = _Authority("attach", ())
    left = DiscretizationPlan()
    right = DiscretizationPlan()
    right.freeze()
    with pytest.raises(RuntimeError, match="interfaces is frozen"):
        ConservativeInterface.attach(authority, left, right)
    assert left.interfaces.values() == ()
    assert right.interfaces.values() == ()


def _resolved_state(block_name):
    model_owner = OwnerPath.model("claim-model")
    block = BlockHandle(
        block_name, owner=OwnerPath.case("claim-case"), model_owner=model_owner)._resolved()
    declaration = StateHandle("U", owner=model_owner)
    state = declaration._with_owner(
        block.instance_owner_path,
        declaration_ref=declaration._resolved(),
        block_ref=block,
    )
    return block, state


def test_endpoint_boundary_must_belong_to_the_owning_block_frame():
    owned_frame = Rectangle(
        "owned", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    foreign_frame = Rectangle(
        "foreign", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    left_block, left_state = _resolved_state("left")
    _right_block, right_state = _resolved_state("right")

    class InterfaceLike:
        left = BlockInterfaceSide(left_state, foreign_frame.boundaries.x_max)
        right = BlockInterfaceSide(right_state, owned_frame.boundaries.x_min)

        def resolve_references(self, resolver):
            return self

    context = SimpleNamespace(
        resolve=lambda handle: handle, frame=owned_frame, block=left_block)
    with pytest.raises(ValueError, match="does not belong"):
        ConservativeInterface.resolve_for_numerics(InterfaceLike(), context)
