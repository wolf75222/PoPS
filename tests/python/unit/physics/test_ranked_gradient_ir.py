"""Cartesian derivatives are one ranked symbolic operation, not separate dimensional paths."""
from __future__ import annotations

import pytest

from pops import math
from pops.frames import X_AXIS, Y_AXIS, Z_AXIS


@pytest.mark.parametrize(
    ("axis", "ordinal", "constructor"),
    ((X_AXIS, 0, math.dx), (Y_AXIS, 1, math.dy), (Z_AXIS, 2, math.dz)),
)
def test_gradient_component_uses_the_typed_axis_ordinal(axis, ordinal, constructor) -> None:
    field = "phi"

    selected = math.grad(field)[axis]
    named = constructor(field)

    assert selected.axis == ordinal
    assert named.axis == ordinal
    assert getattr(math.grad(field), axis.name).axis == ordinal


@pytest.mark.parametrize("invalid", (-1, 3, True, "x"))
def test_partial_rejects_non_cartesian_ordinals(invalid) -> None:
    with pytest.raises((TypeError, ValueError)):
        math.Partial("phi", invalid)
