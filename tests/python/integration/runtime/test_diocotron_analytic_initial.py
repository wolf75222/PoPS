"""A diocotron annulus is materialized by the generic native analytic profile."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pops
import pytest
from pops.analytic import angle, between, radius, sin, where
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import Analytic
from pops.lib.time import ForwardEuler
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]

ROOT = Path(__file__).resolve().parents[4]
CELLS = 48
BACKGROUND = 1.0e-4


def test_diocotron_profile_is_materialized_natively_without_embedded_boundary(
    isolated_native_cache, native_cxx, kokkos_root,
) -> None:
    del isolated_native_cache, kokkos_root

    frame = Rectangle(
        "diocotron-box", lower=(-0.5, -0.5), upper=(0.5, 0.5)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("diocotron-density", frame=frame)
    state = model.state("U", components=("density",))
    (rho,) = state
    flux = model.flux(
        "stationary-density",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    rate = model.rate("stationary-rate", equation=ddt(state) == -div(flux))

    case = pops.Case("diocotron-analytic-initial")
    block = case.block("plasma", model)
    block_state = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    program = ForwardEuler(block_state, rate=rate)
    program.step_strategy(FixedDt(1.0e-3))
    case.program(program)

    density = where(
        between(radius(frame), 0.35, 0.40),
        0.9 + 0.1 * sin(4.0 * angle(frame)),
        BACKGROUND,
    )
    case.initials.add(
        InitialCondition(
            state=block_state,
            value=Analytic(frame=frame, components=(density,)),
            projection=ConservativeCellAverage(),
        )
    )
    layout = Uniform(
        CartesianGrid(
            frame=frame,
            cells=(CELLS, CELLS),
            periodic=PeriodicAxes(frame.axes),
        )
    )

    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": native_cxx},
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    simulation = pops.bind(artifact)
    materialized = np.asarray(
        simulation.state_global("plasma"), dtype=np.float64
    ).reshape((1, CELLS, CELLS))[0]

    centers = -0.5 + (np.arange(CELLS, dtype=np.float64) + 0.5) / CELLS
    x, y = np.meshgrid(centers, centers, indexing="xy")
    radial_coordinate = np.hypot(x, y)
    angular_modulation = np.sin(4.0 * np.arctan2(y, x))

    assert np.isfinite(materialized).all()
    assert np.min(materialized) > 0.0

    # These cells are farther than a full cell diagonal from either annulus interface.
    outside = (radial_coordinate < 0.28) | (radial_coordinate > 0.47)
    assert np.count_nonzero(outside) > 1000
    np.testing.assert_allclose(
        materialized[outside], BACKGROUND, rtol=0.0, atol=5.0e-16
    )

    # The selected cells are wholly inside the annulus, so their cell averages retain the
    # positive and negative lobes of the four-fold angular perturbation.
    annulus_core = (radial_coordinate > 0.365) & (radial_coordinate < 0.385)
    positive_lobe = annulus_core & (angular_modulation > 0.8)
    negative_lobe = annulus_core & (angular_modulation < -0.8)
    assert np.count_nonzero(positive_lobe) >= 16
    assert np.count_nonzero(negative_lobe) >= 16
    assert np.mean(materialized[positive_lobe]) > 0.96
    assert np.mean(materialized[negative_lobe]) < 0.84
    assert (
        np.mean(materialized[positive_lobe])
        - np.mean(materialized[negative_lobe])
        > 0.15
    )
