"""Formula and installed-model checks remain reachable from the final Model authority."""
from __future__ import annotations

import numpy as np
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.math import sqrt
from pops.mesh import NativeSpatialLayout
from pops.physics import Model
from pops.physics.roles import Density, Momentum


def _isothermal_model(*, broken_roundtrip: bool = False, nan_flux: bool = False):
    frame = Rectangle(
        "check-model-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("check-model", frame=frame)
    state = model.state(
        "U",
        components=("rho", "mx", "my"),
        roles={
            "rho": Density(),
            "mx": Momentum(x_axis),
            "my": Momentum(y_axis),
        },
    )
    rho, mx, my = state
    sound_speed_squared = 0.5
    u = model.primitive("u", mx / rho)
    v = model.primitive("v", my / rho)
    pressure = model.scalar("p", sound_speed_squared * rho)
    sound_speed = sqrt(sound_speed_squared)
    mass_flux_x = sqrt(rho - 10.0) if nan_flux else mx
    model.flux(
        "isothermal",
        frame=frame,
        state=state,
        components={
            x_axis: (mass_flux_x, mx * u + pressure, mx * v),
            y_axis: (my, my * u, my * v + pressure),
        },
        waves={
            x_axis: (u - sound_speed, u, u + sound_speed),
            y_axis: (v - sound_speed, v, v + sound_speed),
        },
    )

    lowering = model.__pops_compiler_lowering__()
    assert lowering.facade is model
    assert lowering.source_module is model.module
    emitter = lowering.emit_model
    emitter.primitive_vars(rho, u, v)
    emitter.conservative_from(
        [rho, rho * u, (2.0 if broken_roundtrip else 1.0) * rho * v]
    )
    emitter.elliptic_rhs(0.0 * rho)
    return model, emitter


def _runtime_layout(cells: int = 8) -> NativeSpatialLayout:
    from pops._geometry_contracts import cartesian_geometry_contract

    coordinates, measure = cartesian_geometry_contract(2)
    return NativeSpatialLayout(
        layout_id="layout://tests/check-runtime",
        coordinate_system=coordinates,
        cell_measure=measure,
        axis_names=("x", "y"),
        shape=(cells, cells),
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
        periodicity=(True, True),
        centering="cell",
        decomposition={
            "schema_version": 1,
            "kind": "single_box",
            "boxes": [{"lower": [0, 0], "upper_exclusive": [cells, cells]}],
        },
    )


def test_authenticated_formula_oracle_detects_roundtrip_and_nonfinite_flux() -> None:
    _, healthy = _isothermal_model()
    report = healthy.check_model()
    assert report["ok"] is True
    assert report["n_samples"] == 64

    _, broken = _isothermal_model(broken_roundtrip=True)
    with pytest.raises(ValueError, match="round-trip"):
        broken.check_model()

    _, nonfinite = _isothermal_model(nan_flux=True)
    with pytest.raises(ValueError, match="flux"):
        nonfinite.check_model()
    report = nonfinite.check_model(raise_on_error=False)
    assert report["ok"] is False
    assert any("flux" in failure for failure in report["failures"])


@pytest.mark.compiler
@pytest.mark.native_loader
def test_compiled_model_rechecks_the_installed_native_block(
    isolated_native_cache, native_cxx, kokkos_root,
) -> None:
    del isolated_native_cache, native_cxx, kokkos_root
    _, emitter = _isothermal_model()
    compiled = emitter.compile(backend="production", target="system")

    with pytest.raises(ValueError, match="authenticated NativeSpatialLayout"):
        compiled.check_runtime()

    layout = _runtime_layout()
    report = compiled.check_runtime(layout)
    assert report["ok"] is True
    assert report["failures"] == []

    zeros = np.zeros((8, 8), dtype=np.float64)
    report = compiled.check_runtime(
        layout,
        state={"rho": zeros, "mx": zeros, "my": zeros},
        raise_on_error=False,
    )
    assert report["ok"] is False
    assert any(
        "residual -div F + S evaluation failed" in failure
        and "numerical flux evaluation reject" in failure
        and "reason_code=0x53544201" in failure
        for failure in report["failures"]
    )
    assert any("Density" in failure for failure in report["failures"])
