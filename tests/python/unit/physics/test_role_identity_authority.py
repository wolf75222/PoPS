"""The role vocabulary has one neutral identity authority and one compatible physics facade."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pops.model import identity
from pops.physics.aux import roles_for
from pops.physics import roles
from pops.runtime._bricks_time_imex import _norm_implicit


@dataclass(frozen=True)
class _Axis:
    index: int


def test_physics_roles_reexport_the_exact_model_identity_objects() -> None:
    for name in identity.__all__:
        assert getattr(roles, name) is getattr(identity, name)
    assert roles.__all__ == identity.__all__


def test_role_key_is_itself_a_typed_component_role() -> None:
    role = identity.parse_role("q1")

    assert isinstance(role, identity.ComponentRole)
    assert identity.native_role_token(role) == "q1"


def test_custom_roles_cross_the_physics_runtime_seam_without_reparsing_types() -> None:
    custom = roles.Custom("electron_entropy")
    momentum = roles.Momentum(_Axis(2))

    schema = identity.StateSchema.resolve(
        (roles.Density(), momentum, custom),
        dimension=3,
        where="identity authority test",
    )

    assert tuple(role.token for role in schema.roles) == (
        "density",
        "momentum:2",
        "electron_entropy",
    )
    assert schema.index(custom) == 2
    assert _norm_implicit("IMEX", None, (momentum, custom)) == (
        [],
        ["momentum:2", "electron_entropy"],
    )


def test_exact_dimension_validation_remains_on_the_neutral_authority() -> None:
    with pytest.raises(ValueError, match="outside dimension 2"):
        identity.StateSchema.resolve(
            (roles.Momentum(_Axis(2)),),
            dimension=2,
            where="identity authority test",
        )


@pytest.mark.parametrize(
    "value",
    (
        lambda: identity.RoleKey("custom"),
        lambda: identity.parse_role("custom"),
        lambda: identity.RoleKey("custom", label="density"),
        lambda: identity.RoleKey("custom", label="momentum:0"),
        lambda: identity.RoleKey("custom", label="momentum:01"),
    ),
)
def test_neutral_authority_rejects_anonymous_or_colliding_custom_roles(value) -> None:
    with pytest.raises(ValueError):
        value()


def test_canonical_lowering_rejects_duplicate_roles_and_out_of_rank_axes() -> None:
    with pytest.raises(ValueError, match="duplicate token 'density'"):
        roles_for(("rho", "density"))
    with pytest.raises(ValueError, match="duplicate token 'q1'"):
        roles_for(("first", "second"), (roles.Custom("q1"), roles.Custom("q1")))
    with pytest.raises(ValueError, match="outside dimension 2"):
        roles_for(("mz",), (roles.Momentum(_Axis(2)),), dimension=2)


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("representation", ["typed", "token", "parsed"])
def test_axial_schema_retains_all_physical_embedding_axes(dimension, representation):
    values = {
        "typed": tuple(roles.Axial(_Axis(axis)) for axis in range(3)),
        "token": tuple("axial:%d" % axis for axis in range(3)),
        "parsed": tuple(identity.RoleKey("axial", axis) for axis in range(3)),
    }[representation]
    schema = identity.StateSchema.resolve(values, dimension=dimension)
    assert schema.dimension == dimension
    assert schema.axes("axial") == (0, 1, 2)
    assert tuple(role.token for role in schema.roles) == ("axial:0", "axial:1", "axial:2")
    assert schema.index(roles.Axial(_Axis(2))) == 2
    assert identity.StateSchema.resolve(tuple(role.token for role in schema.roles),
                                        dimension=dimension) == schema


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_canonical_role_tuple_retains_out_of_plane_axial_metadata(dimension):
    tokens = ("density",) + tuple("momentum:%d" % axis for axis in range(dimension)) + ("axial:2",)
    schema = identity.StateSchema.resolve(tokens, dimension=dimension)
    assert tuple(role.token for role in schema.roles) == tokens
    assert schema.index("axial:2") == dimension + 1
    assert schema.axes("momentum") == tuple(range(dimension))


@pytest.mark.parametrize("dimension", [None, 1, 2, 3])
def test_axial_role_refuses_axes_outside_physical_embedding(dimension):
    with pytest.raises(ValueError, match="physical x/y/z embedding"):
        identity.parse_role("axial:3", dimension=dimension)
    with pytest.raises(ValueError, match="physical x/y/z embedding"):
        identity.native_role_token(roles.Axial(_Axis(3)), dimension=dimension)
    if dimension is not None:
        with pytest.raises(ValueError, match="physical x/y/z embedding"):
            identity.StateSchema.resolve((identity.RoleKey("axial", 3),), dimension=dimension)


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("family", ["momentum", "velocity"])
def test_polar_role_axes_remain_limited_to_mesh_dimension(dimension, family):
    with pytest.raises(ValueError, match="outside dimension"):
        identity.StateSchema.resolve(("%s:%d" % (family, dimension),), dimension=dimension)


@pytest.mark.parametrize("dimension", [True, 0, 4, "2"])
def test_axial_embedding_does_not_bypass_invalid_mesh_dimension(dimension):
    with pytest.raises(ValueError, match="must be one of 1, 2, or 3"):
        identity.parse_role("axial:2", dimension=dimension)
