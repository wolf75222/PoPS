"""Native HLL oracles for Jacobian eigenvalues, partitions and finite differences.

All authoring uses the final blackboard ``Model``.  Three independently compiled models exercise
the full numeric Jacobian, a direction-specific block partition and the finite-difference Jacobian;
the native HLL residual is checked against an independent NumPy reference in both directions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pops.runtime._engine_descriptors as engine
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.math import ddt, div
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import HLL
from pops.numerics.riemann.waves import FromJacobian, provider_of
from pops.physics import Model
from pops.runtime._system import System
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
)


INCLUDE = repo_include()

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _frame(name: str):
    return Rectangle(name, (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())


def _jacobian_model(name: str, *, eig: str = "numeric", blocks=None) -> Model:
    frame = _frame(name)
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state("U", components=("q1", "q2"))
    q1, q2 = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={
            # Symmetric, genuinely coupled Jacobians.  Singleton partitions therefore produce
            # different spectra from the full eigensolve and cannot pass by being ignored.
            x_axis: (0.5 * q1 * q1 + 0.2 * q2, 0.5 * q2 * q2 + 0.2 * q1),
            # Ay is deliberately asymmetric with Ax to catch a direction swap.
            y_axis: (-0.25 * q1 * q1 + 0.15 * q2,
                     -0.75 * q2 * q2 + 0.15 * q1),
        },
    )
    model.wave_speeds_from_jacobian(eig=eig, blocks=blocks)
    model.rate("transport", equation=ddt(state) == -div(flux))
    return model


def _compile_native(model: Model, path: Path):
    lowering = model.__pops_compiler_lowering__()
    assert lowering.facade is model
    assert lowering.source_module is model.module
    return lowering.emit_model.compile(str(path), INCLUDE, backend="production")


def _state(n: int) -> np.ndarray:
    points = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(points, points, indexing="xy")
    q1 = 1.0 + 0.25 * np.sin(2.0 * np.pi * x) * np.cos(4.0 * np.pi * y)
    q2 = -0.55 + 0.15 * np.cos(4.0 * np.pi * x) + 0.1 * np.sin(2.0 * np.pi * y)
    return np.stack((q1, q2))


def _physical_flux(state: np.ndarray, direction: int) -> np.ndarray:
    q1, q2 = state
    if direction == 0:
        return np.stack((0.5 * q1 * q1 + 0.2 * q2,
                         0.5 * q2 * q2 + 0.2 * q1))
    return np.stack((-0.25 * q1 * q1 + 0.15 * q2,
                     -0.75 * q2 * q2 + 0.15 * q1))


def _wave_speeds(
    state: np.ndarray,
    direction: int,
    *,
    partitioned: bool,
) -> tuple[np.ndarray, np.ndarray]:
    q1, q2 = state
    if direction == 0:
        diagonal_left, diagonal_right, coupling = q1, q2, 0.2
    else:
        diagonal_left, diagonal_right, coupling = -0.5 * q1, -1.5 * q2, 0.15
    if partitioned:
        return (np.minimum(diagonal_left, diagonal_right),
                np.maximum(diagonal_left, diagonal_right))
    center = 0.5 * (diagonal_left + diagonal_right)
    radius = np.sqrt((0.5 * (diagonal_left - diagonal_right)) ** 2 + coupling**2)
    return center - radius, center + radius


def _expected_hll_rhs(state: np.ndarray, n: int, *, partitioned: bool) -> np.ndarray:
    rhs = np.zeros_like(state)
    spacing = 1.0 / n
    for direction, array_axis in ((0, 2), (1, 1)):
        left = state
        right = np.roll(state, -1, axis=array_axis)
        flux_left = _physical_flux(left, direction)
        flux_right = _physical_flux(right, direction)
        lo_left, hi_left = _wave_speeds(left, direction, partitioned=partitioned)
        lo_right, hi_right = _wave_speeds(right, direction, partitioned=partitioned)
        speed_left = np.minimum(lo_left, lo_right)
        speed_right = np.maximum(hi_left, hi_right)
        span = speed_right - speed_left
        hll = (
            speed_right * flux_left
            - speed_left * flux_right
            + speed_left * speed_right * (right - left)
        ) / span
        interface_flux = np.where(
            speed_left >= 0.0,
            flux_left,
            np.where(speed_right <= 0.0, flux_right, hll),
        )
        rhs -= (interface_flux - np.roll(interface_flux, 1, axis=array_axis)) / spacing
    return rhs


def _native_rhs(compiled, state: np.ndarray) -> np.ndarray:
    n = state.shape[-1]
    simulation = System(n=n, L=1.0, periodic=True)
    simulation.add_equation(
        "toy",
        model=compiled,
        spatial=engine.Spatial(limiter=FirstOrder(), flux=HLL()),
        time=engine.Explicit(),
    )
    simulation.set_state("toy", state)
    return np.asarray(simulation.eval_rhs("toy"))


def test_compiled_jacobian_speeds_cover_eigensolve_blocks_fd_and_directions(
    tmp_path: Path,
) -> None:
    reason = missing_native_compile_requirement(INCLUDE, default_cxx())
    if reason:
        pytest.skip(reason)

    partitions = {"x": [[0], [1]], "y": [[1], [0]]}
    numeric_model = _jacobian_model("jacobian_numeric")
    partitioned_model = _jacobian_model("jacobian_partitioned", blocks=partitions)
    fd_model = _jacobian_model("jacobian_fd", eig="fd")

    numeric_provider = provider_of(numeric_model)
    partitioned_provider = provider_of(partitioned_model)
    fd_provider = provider_of(fd_model)
    assert numeric_provider.kind == FromJacobian(eig="numeric").kind
    assert numeric_provider.options()["eig"] == "numeric"
    assert partitioned_provider.kind == FromJacobian(
        eig="numeric", blocks=partitions).kind
    assert partitioned_provider.options()["eig"] == "numeric"
    assert fd_provider.kind == FromJacobian(eig="fd").kind
    assert fd_provider.options()["eig"] == "fd"

    numeric = _compile_native(numeric_model, tmp_path / "jacobian_numeric.so")
    partitioned = _compile_native(
        partitioned_model, tmp_path / "jacobian_partitioned.so")
    finite_difference = _compile_native(fd_model, tmp_path / "jacobian_fd.so")
    assert numeric.has_wave_speeds
    assert partitioned.has_wave_speeds
    assert finite_difference.has_wave_speeds

    state = _state(24)
    expected = _expected_hll_rhs(state, 24, partitioned=False)
    expected_partitioned = _expected_hll_rhs(state, 24, partitioned=True)
    assert np.max(np.abs(expected_partitioned - expected)) > 1.0e-3
    numeric_rhs = _native_rhs(numeric, state)
    partitioned_rhs = _native_rhs(partitioned, state)
    fd_rhs = _native_rhs(finite_difference, state)

    # The independent reference uses different signed spectra in x and y, so a direction swap or a
    # scalar-radius fallback cannot satisfy this equality.
    np.testing.assert_allclose(numeric_rhs, expected, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        partitioned_rhs, expected_partitioned, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(fd_rhs, numeric_rhs, rtol=1.0e-5, atol=1.0e-8)
