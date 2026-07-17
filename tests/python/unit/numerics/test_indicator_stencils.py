"""Exact, serializable AMR indicator stencil contract."""
from __future__ import annotations

import json

import pytest

from pops.descriptors import _native
from pops.model import Handle, OwnerPath
from pops.numerics import FiniteVolume
from pops.numerics.indicator_stencils import (
    DiscreteGradientStencil,
    LinearAxisStencil,
)
from pops.numerics.reconstruction import MUSCL
from pops.numerics.riemann import ScalarUpwind
from pops.numerics.variables import Conservative


def _method(*, reconstruction=None):
    owner = OwnerPath.model("transport")
    state = Handle("U", kind="state", owner=owner)
    return FiniteVolume(
        flux=Handle("F", kind="flux", owner=owner),
        variables=Conservative(state),
        reconstruction=MUSCL() if reconstruction is None else reconstruction,
        riemann=ScalarUpwind(velocity=Handle("a", kind="vector", owner=owner)),
    )


def test_finite_volume_projects_stencil_to_json_and_round_trips_identity():
    method = _method()
    lowering = method.amr_indicator_stencil(dimension=2)
    reopened = DiscreteGradientStencil.from_data(lowering.to_data())

    assert reopened == lowering
    assert reopened.identity == lowering.identity
    assert json.loads(json.dumps(lowering.to_data())) == lowering.to_data()
    projected = method.to_data()["reconstruction"]["options"][
        "amr_gradient_stencil"]
    assert projected == lowering.axes[0].to_data()
    assert json.loads(json.dumps(method.to_data())) == method.to_data()


def test_missing_typed_stencil_is_refused_instead_of_selecting_a_fallback():
    reconstruction = _native(
        "custom", "external::Custom", "custom", category="reconstruction",
        formal_order=2, ghost_depth=2)
    with pytest.raises(NotImplementedError, match="typed AMR gradient stencil"):
        _method(reconstruction=reconstruction).amr_indicator_stencil(dimension=2)


def test_claimed_formal_order_must_be_authenticated_by_every_moment():
    with pytest.raises(ValueError, match="authenticate formal_order=4.*moment 3"):
        LinearAxisStencil(
            (-2, -1, 1, 2), (0.0, -0.5, 0.5, 0.0), formal_order=4)


def test_serialized_halo_and_identity_cannot_be_forged():
    lowering = _method().amr_indicator_stencil(dimension=2)
    bad_halo = lowering.to_data()
    bad_halo["axes"][0]["ghost_lower"] = 7
    with pytest.raises(ValueError, match="inconsistent halo"):
        DiscreteGradientStencil.from_data(bad_halo)

    bad_identity = lowering.to_data()
    bad_identity["identity"] = "forged"
    with pytest.raises(ValueError, match="identity is not authentic"):
        DiscreteGradientStencil.from_data(bad_identity)
