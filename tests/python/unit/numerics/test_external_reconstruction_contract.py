"""The reconstruction catalog exposes only routes the native Kokkos lowering can execute."""

from __future__ import annotations

import importlib

import pytest


def test_symbolic_user_reconstruction_selector_is_published() -> None:
    module = importlib.import_module("pops.numerics.reconstruction")

    assert "User" in module.__all__
    assert callable(module.User)
    assert module.reconstruction.User is module.User


@pytest.mark.parametrize("catalog", [
    importlib.import_module("pops.numerics.reconstruction"),
    importlib.import_module("pops.numerics.reconstruction").reconstruction,
])
def test_user_reconstruction_requires_an_authored_body_and_explicit_order(catalog) -> None:
    from pops.numerics.reconstruction.user import authenticated_user_reconstruction
    policy = catalog.User(lambda sample: sample(0), formal_order=1)
    authenticated_user_reconstruction(policy)
    assert policy.options["formal_order"] == 1
    assert policy.options["ghost_depth"] == 1
    with pytest.raises(TypeError, match="formal_order"):
        catalog.User(lambda sample: sample(0))
    with pytest.raises(TypeError, match="callable symbolic body"):
        catalog.User("acme.reconstruction", formal_order=1)


@pytest.mark.parametrize("scheme,native_id", [
    ("user", "acme_reconstruct"),
    ("minmod", "pops::Minmod"),
])
def test_external_descriptor_cannot_masquerade_as_a_native_reconstruction(
    scheme: str, native_id: str
) -> None:
    from pops.descriptors import BrickDescriptor
    from pops.runtime._bricks_scheme import Spatial

    descriptor = BrickDescriptor(
        "acme.reconstruction",
        "external_cpp",
        category="reconstruction",
        native_id=native_id,
        scheme=scheme,
        requirements={"capabilities": ["device_callable"]},
        capabilities={"formal_order": 3, "ghost_depth": 2},
        options={
            "formal_order": 2,
            "ghost_depth": 2,
            "muscl_compatible": True,
        },
    )

    expected = ("not a known limiter scheme" if scheme == "user"
                else "authenticated native descriptor")
    with pytest.raises(ValueError, match=expected):
        Spatial(reconstruction=descriptor)


def test_builtin_reconstruction_contract_is_derived_from_generated_routes() -> None:
    from pops.numerics.reconstruction import (
        FirstOrder,
        MUSCL,
        WENO5,
        WENO5Z,
        authenticated_reconstruction_route,
    )
    from pops.numerics.reconstruction.limiters import MC, Minmod, Superbee, VanLeer

    for descriptor, token, native_id, order, depth in (
        (FirstOrder(), "none", "pops::NoSlope", 1, 1),
        (Minmod(), "minmod", "pops::Minmod", 2, 2),
        (VanLeer(), "vanleer", "pops::VanLeer", 2, 2),
        (MC(), "mc", "pops::MC", 2, 2),
        (Superbee(), "superbee", "pops::Superbee", 2, 2),
        (MUSCL(VanLeer()), "vanleer", "pops::VanLeer", 2, 2),
        (MUSCL(MC()), "mc", "pops::MC", 2, 2),
        (MUSCL(Superbee()), "superbee", "pops::Superbee", 2, 2),
        (WENO5(), "weno5", "pops::Weno5", 5, 3),
        (WENO5Z(), "weno5", "pops::Weno5", 5, 3),
    ):
        route = authenticated_reconstruction_route(descriptor)
        assert descriptor.scheme == route.token == token
        assert descriptor.native_id == route.native_entry == native_id
        assert descriptor.options["formal_order"] == route.metadata["formal_order"] == order
        assert descriptor.options["ghost_depth"] == route.metadata["n_ghost"] == depth


def test_native_token_with_wrong_entry_or_structural_claim_is_rejected() -> None:
    from pops.descriptors import BrickDescriptor
    from pops.runtime._bricks_scheme import Spatial

    wrong_entry = BrickDescriptor(
        "spoofed-minmod", "native", category="limiter",
        native_id="pops::VanLeer", scheme="minmod",
        options={"formal_order": 2, "ghost_depth": 2, "muscl_compatible": True},
    )
    with pytest.raises(ValueError, match="native_id=.*generated catalogue requires"):
        Spatial(limiter=wrong_entry)

    wrong_order = BrickDescriptor(
        "spoofed-minmod", "native", category="limiter",
        native_id="pops::Minmod", scheme="minmod",
        options={"formal_order": 7, "ghost_depth": 2, "muscl_compatible": True},
    )
    with pytest.raises(ValueError, match="formal_order=7.*expected 2"):
        Spatial(limiter=wrong_order)
