"""The physics graph carries the domain-inferred axis rank without a 2D selector."""
from __future__ import annotations

import pytest

from pops.domain import CartesianDomain
from pops.physics import Model


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_board_flux_and_stability_maps_follow_the_typed_frame(dimension: int) -> None:
    frame = CartesianDomain(
        "ranked-flux-%d" % dimension,
        (0.0,) * dimension,
        (1.0,) * dimension,
    ).frame()
    model = Model("ranked_flux_%d" % dimension, frame=frame)
    state = model.state("U", components=("u",))
    (u,) = state
    components = {
        axis: (u * (axis.index + 1),) for axis in frame.axes
    }
    waves = {
        axis: (0 * u + axis.index + 1,) for axis in frame.axes
    }

    model.flux("F", frame=frame, state=state, components=components, waves=waves)

    names = tuple(axis.name for axis in frame.axes)
    assert tuple(model._dsl._m._flux) == names
    assert tuple(model._dsl._m._eig) == names
    for axis in frame.axes:
        assert model.flux_value([2.0], None, axis).tolist() == [
            2.0 * (axis.index + 1)
        ]


def test_ranked_flux_rejects_partial_axis_maps() -> None:
    frame = CartesianDomain("partial-flux", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)).frame()
    model = Model("partial_flux", frame=frame)
    state = model.state("U", components=("u",))

    with pytest.raises(ValueError, match="every typed frame axis"):
        model.flux(
            "F",
            frame=frame,
            state=state,
            components={frame.x: (state[0],), frame.y: (state[0],)},
        )
