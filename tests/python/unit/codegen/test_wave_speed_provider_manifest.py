"""Wave-speed source provenance survives the public physics-to-Module projection."""
from __future__ import annotations

import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.physics import Model


def _transport_model(name: str):
    frame = Rectangle(name + "-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    model = Model(name, frame=frame)
    state = model.state("U", components=("q1", "q2"))
    q1, q2 = state
    x_axis, y_axis = frame.axes
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (q1, -q2), y_axis: (-q1, q2)},
    )
    return model, state, flux, frame


def test_jacobian_setter_invalidates_cached_module_manifest():
    model, _, _, _ = _transport_model("jacobian-provider-manifest")
    assert model.module.manifest().wave_speed_provider is None

    model.wave_speeds_from_jacobian(eig="fd", blocks={"x": [[0], [1]], "y": [[1], [0]]})

    assert model.module.manifest().wave_speed_provider == "jacobian"


def test_explicit_pair_setter_invalidates_cached_module_manifest():
    model, state, flux, frame = _transport_model("explicit-provider-manifest")
    assert model.module.manifest().wave_speed_provider is None
    q1, _ = state
    x_axis, y_axis = frame.axes

    model.wave_speeds(
        flux,
        frame=frame,
        values={x_axis: (-q1, q1), y_axis: (-q1, q1)},
    )

    assert model.module.manifest().wave_speed_provider == "explicit_pair"


@pytest.mark.parametrize("declaration", ("primitive", "scalar"))
def test_pressure_primitive_invalidates_cached_module_manifest(declaration):
    model, state, _, _ = _transport_model("pressure-provider-" + declaration)
    assert model.module.manifest().wave_speed_provider is None

    getattr(model, declaration)("p", state[0])

    assert model.module.manifest().wave_speed_provider == "pressure_derived"
