"""Final lifecycle coverage for the generic condensed implicit Program route."""

from __future__ import annotations

import numpy as np
import pops
import pytest
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.linalg import LinearProblem
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.physics import Density, Momentum
from pops.representations import Conservative
from pops.solvers import BiCGStab
from pops.spaces import CellState
from pops.time import FailRun, FixedDt


CELLS = 8
DT = 0.4
ROTATION_RATE = 12.0
RHO = 2.0

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _build_case():
    frame = Rectangle(
        "condensed_unit_square", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("renamed_condensed_rotation", frame=frame)
    state = model.state(
        "U",
        components=("inventory", "east_memory", "north_memory"),
        representation=Conservative(),
        space=CellState(frame=frame),
        roles={
            "inventory": Density(),
            "east_memory": Momentum(axis=x_axis),
            "north_memory": Momentum(axis=y_axis),
        },
    )
    inventory, east_memory, north_memory = state
    zero_flux = (
        0.0 * inventory,
        0.0 * east_memory,
        0.0 * north_memory,
    )
    flux = model.flux(
        "inert_transport",
        frame=frame,
        state=state,
        components={x_axis: zero_flux, y_axis: zero_flux},
        waves={x_axis: (0.0, 0.0, 0.0), y_axis: (0.0, 0.0, 0.0)},
    )
    inert_rate = model.rate("inert_rate", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(
        inert_rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    rotation = model.operator(
        "implicit_rotation",
        returns=model.local_linear_operator(
            "implicit_rotation",
            on=state,
            matrix=(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, ROTATION_RATE),
                (0.0, -ROTATION_RATE, 0.0),
            ),
        ),
    )

    case = pops.Case("uniform_condensed_rotation")
    block = case.block("packet", model=model)
    case.numerics(numerics, block=block)
    block_state = block[state]

    program = pops.Program("uniform_condensed_implicit")
    temporal = program.state(block_state)
    current = temporal.n
    coefficients = program.condensed_coeffs(
        "condensed_tensor",
        state=current,
        linear_operator=rotation,
        subset=(1, 2),
        c=program.dt * program.dt,
        th_dt=program.dt,
        c_rho=0,
    )
    phi_previous = program.scalar_field("condensed_phi_zero")
    rhs_storage = program.scalar_field("condensed_rhs_storage")
    rhs = program.condensed_rhs(
        rhs_storage,
        phi_previous,
        current,
        linear_operator=rotation,
        subset=(1, 2),
        th_dt=program.dt,
        g=program.dt,
    )
    operator = program.matrix_free_operator("condensed_elliptic_operator")

    def apply(builder, _out, value):
        laplacian = builder.scalar_field("condensed_laplacian")
        return -1 * builder.apply_laplacian_coeff(
            laplacian, value, coefficients)

    program.set_apply(operator, apply)
    phi = program.solve(
        LinearProblem(operator, rhs, initial_guess=phi_previous),
        solver=BiCGStab(max_iter=80, rel_tol=1.0e-12),
        name="condensed_potential",
    ).consume(action=FailRun())
    reconstructed = program.condensed_reconstruct(
        "implicit_rotated_state",
        state=current,
        phi=phi,
        linear_operator=rotation,
        subset=(1, 2),
        th_dt=program.dt,
        c_rho=0,
    )
    endpoint = program.value(
        "accepted_condensed_state", reconstructed, at=temporal.next.point)
    program.commit(temporal.next, endpoint)
    program.step_strategy(FixedDt(DT))
    case.program(program)

    layout = Uniform(CartesianGrid(
        frame=frame,
        cells=(CELLS, CELLS),
        periodic=PeriodicAxes(frame.axes),
    ))
    return case, layout, state


def _centered_difference(values, *, axis):
    spacing = 1.0 / CELLS
    return (
        np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)
    ) / (2.0 * spacing)


def _periodic_condensed_oracle(density, east, north):
    """Independent Fourier solve of the emitted constant-coefficient discrete system."""
    rotation = DT * ROTATION_RATE
    denominator = 1.0 + rotation * rotation
    flux_east = (east + rotation * north) / denominator
    flux_north = (-rotation * east + north) / denominator
    rhs = -DT * (
        _centered_difference(flux_east, axis=1)
        + _centered_difference(flux_north, axis=0)
    )
    assert np.max(np.abs(rhs)) > 1.0e-3

    # For the authored skew rotation, the two constant cross coefficients are exact opposites, so
    # their discrete mixed derivatives cancel.  The matrix-free operator is therefore
    # ``-a * Laplacian_5`` with this positive scalar coefficient.
    coefficient = 1.0 + DT * DT * float(density[0, 0]) / denominator
    modes = np.fft.fftfreq(CELLS) * CELLS
    kx, ky = np.meshgrid(modes, modes, indexing="xy")
    eigenvalue = coefficient * 4.0 * CELLS * CELLS * (
        np.sin(np.pi * kx / CELLS) ** 2
        + np.sin(np.pi * ky / CELLS) ** 2
    )
    rhs_hat = np.fft.fft2(rhs)
    phi_hat = np.zeros_like(rhs_hat)
    nonzero = eigenvalue > 0.0
    phi_hat[nonzero] = rhs_hat[nonzero] / eigenvalue[nonzero]
    potential = np.fft.ifft2(phi_hat).real
    gradient_east = _centered_difference(potential, axis=1)
    gradient_north = _centered_difference(potential, axis=0)
    velocity_east = east / density - DT * gradient_east
    velocity_north = north / density - DT * gradient_north
    expected_east = density * (
        velocity_east + rotation * velocity_north
    ) / denominator
    expected_north = density * (
        -rotation * velocity_east + velocity_north
    ) / denominator
    return np.stack((density, expected_east, expected_north)), potential


def test_uniform_condensed_implicit_matches_discrete_fourier_oracle(
    isolated_native_cache, native_cxx, kokkos_root,
):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, state = _build_case()
    assert state.components == ("inventory", "east_memory", "north_memory")
    resolved = pops.resolve(pops.validate(case), layout=layout)
    artifact = pops.compile(resolved)

    coordinate = (np.arange(CELLS, dtype=np.float64) + 0.5) / CELLS
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    density = np.full((CELLS, CELLS), RHO)
    east = 1.5 + 0.25 * np.sin(2.0 * np.pi * x) + 0.10 * np.cos(2.0 * np.pi * y)
    north = -0.5 + 0.20 * np.cos(2.0 * np.pi * x) - 0.15 * np.sin(2.0 * np.pi * y)
    initial = np.stack((density, east, north))
    runtime = pops.bind(artifact, initial_state={"packet": initial})
    report = pops.run(runtime, t_end=DT, max_steps=1)
    assert report.accepted_steps == 1

    actual = np.asarray(runtime.state_global("packet"), dtype=np.float64).reshape(
        3, CELLS, CELLS)
    expected, potential = _periodic_condensed_oracle(density, east, north)

    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=5.0e-10, atol=5.0e-10)
    assert np.array_equal(actual[0], density)
    assert float(actual[0].sum()) == float(density.sum())
    cell_measure = 1.0 / (CELLS * CELLS)
    expected_inventory_integral = float(actual[0].sum()) * cell_measure
    assert runtime.integral("packet") == expected_inventory_integral
    assert runtime.integral("packet", component=0, levels=(0,)) == expected_inventory_integral
    with pytest.raises(TypeError, match="non-negative integer"):
        runtime.integral("packet", component=True)
    with pytest.raises(ValueError, match="strictly increasing and unique"):
        runtime.integral("packet", levels=(0, 0))
    with pytest.raises(ValueError, match="only level 0"):
        runtime.integral("packet", levels=(1,))
    assert np.max(np.abs(potential)) > 1.0e-6

    # The elliptic correction is observable: a broken/omitted RHS, coefficient assembly or matvec
    # would reduce to the local backward-Euler rotation and fail this separation.
    rotation = DT * ROTATION_RATE
    denominator = 1.0 + rotation * rotation
    local_only = np.stack((
        (east + rotation * north) / denominator,
        (-rotation * east + north) / denominator,
    ))
    assert np.max(np.abs(actual[1:] - local_only)) > 1.0e-4
    assert rotation > 1.0
