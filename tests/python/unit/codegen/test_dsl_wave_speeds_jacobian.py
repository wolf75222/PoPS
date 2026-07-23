"""Public HLL oracles for Jacobian eigenvalues, partitions and finite differences."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pops
from pops._dense_spectral import DenseSpectralCapacityError
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.riemann import FromJacobian, provider_of
from pops.numerics.spatial import FiniteVolume
from pops.physics import Model
from pops.time import FixedDt


ROOT = Path(__file__).resolve().parents[4]
N = 24
# One explicit stage with dt=1 returns exactly U + RHS, retaining the original RHS oracle tolerance.
DT = 1.0

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


def _nonhyperbolic_jacobian_model(name: str) -> Model:
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
            # Constant Jacobian [[1, -1e-6], [1e-6, 1]], eigenvalues 1 +- 1e-6 i.
            # The tiny but genuine complex pair catches any implicit positive tolerance.
            x_axis: (q1 - 1.0e-6 * q2, 1.0e-6 * q1 + q2),
            y_axis: (q1 - 1.0e-6 * q2, 1.0e-6 * q1 + q2),
        },
    )
    model.wave_speeds_from_jacobian(eig="numeric")
    model.rate("transport", equation=ddt(state) == -div(flux))
    return model


def _diagonal_jacobian_model(name: str, components: int) -> Model:
    frame = _frame(name)
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state(
        "U", components=tuple("q%d" % index for index in range(components)))
    model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: tuple(state), y_axis: tuple(state)},
    )
    return model


def _compile_public(
    model: Model,
    *,
    case_name: str,
    waves: FromJacobian,
    cxx: str,
):
    """Compile one HLL model through the final public lifecycle."""
    state = model.states["U"]
    flux = model.fluxes["transport"]
    rate = model.operators["transport"]
    case = pops.Case(case_name)
    block = case.block("toy", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.HLL(waves=waves),
        ),
    )
    case.numerics(numerics, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    layout = Uniform(
        CartesianGrid(
            frame=model.frame,
            cells=(N, N),
            periodic=PeriodicAxes(model.frame.axes),
        )
    )
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": cxx},
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    assert len(artifact.blocks) == 1
    assert artifact.blocks[0].model.has_wave_speeds
    return artifact


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


def _public_rhs(artifact, state: np.ndarray) -> np.ndarray:
    initial = np.ascontiguousarray(state)
    simulation = pops.bind(artifact, initial_state={"toy": initial})
    report = pops.run(simulation, t_end=DT, max_steps=1)
    assert report.accepted_steps == 1
    final = np.asarray(simulation.get_state("toy"), dtype=np.float64).reshape(initial.shape)
    return (final - initial) / DT


def test_compiled_jacobian_speeds_cover_eigensolve_blocks_fd_and_directions(
    isolated_native_cache: Path,
    native_cxx: str,
    kokkos_root: Path,
) -> None:
    del isolated_native_cache, kokkos_root

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

    numeric = _compile_public(
        numeric_model,
        case_name="jacobian_numeric_case",
        waves=FromJacobian(eig="numeric"),
        cxx=native_cxx,
    )
    partitioned = _compile_public(
        partitioned_model,
        case_name="jacobian_partitioned_case",
        waves=FromJacobian(eig="numeric", blocks=partitions),
        cxx=native_cxx,
    )
    finite_difference = _compile_public(
        fd_model,
        case_name="jacobian_fd_case",
        waves=FromJacobian(eig="fd"),
        cxx=native_cxx,
    )

    state = _state(N)
    expected = _expected_hll_rhs(state, N, partitioned=False)
    expected_partitioned = _expected_hll_rhs(state, N, partitioned=True)
    assert np.max(np.abs(expected_partitioned - expected)) > 1.0e-3
    numeric_rhs = _public_rhs(numeric, state)
    partitioned_rhs = _public_rhs(partitioned, state)
    fd_rhs = _public_rhs(finite_difference, state)

    # The independent reference uses different signed spectra in x and y, so a direction swap or a
    # scalar-radius fallback cannot satisfy this equality.
    np.testing.assert_allclose(numeric_rhs, expected, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        partitioned_rhs, expected_partitioned, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(fd_rhs, numeric_rhs, rtol=1.0e-5, atol=1.0e-8)


def test_hll_from_dense_jacobian_rejects_non_real_spectrum(
    isolated_native_cache: Path,
    native_cxx: str,
    kokkos_root: Path,
) -> None:
    del isolated_native_cache, kokkos_root

    model = _nonhyperbolic_jacobian_model("jacobian_nonhyperbolic")
    artifact = _compile_public(
        model,
        case_name="jacobian_nonhyperbolic_case",
        waves=FromJacobian(eig="numeric"),
        cxx=native_cxx,
    )
    state = np.zeros((2, N, N), dtype=np.float64)
    state[0, :, :] = 1.0
    state[1, :, :] = 0.25
    simulation = pops.bind(artifact, initial_state={"toy": state})
    with pytest.raises(
        RuntimeError,
        match=r"solve status=invalid_evaluation.*numerical flux evaluation reject",
    ):
        pops.run(simulation, t_end=DT, max_steps=1)


def test_hll_dense_spectral_capacity_is_checked_per_declared_block() -> None:
    boundary = _diagonal_jacobian_model("dense_hll_boundary", 16)
    boundary.wave_speeds_from_jacobian()
    assert boundary._dsl._m._ws_jacobian["blocks"]["x"] == [list(range(16))]

    too_large = _diagonal_jacobian_model("dense_hll_too_large", 17)
    with pytest.raises(DenseSpectralCapacityError) as caught:
        too_large.wave_speeds_from_jacobian()
    assert caught.value.components == 17
    assert caught.value.max_components == 16
    assert "blocks=" in str(caught.value)
    assert "model.wave_speeds" in str(caught.value)
    assert too_large._dsl._m._ws_jacobian is None

    partitioned = _diagonal_jacobian_model("dense_hll_partitioned", 17)
    partitioned.wave_speeds_from_jacobian(
        blocks=(tuple(range(16)), (16,)))
    assert tuple(
        len(block) for block in partitioned._dsl._m._ws_jacobian["blocks"]["x"]
    ) == (16, 1)
