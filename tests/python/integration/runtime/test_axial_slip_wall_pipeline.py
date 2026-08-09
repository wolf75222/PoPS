"""An axial component survives the public compile-to-bind boundary pipeline."""

from __future__ import annotations

import numpy as np
import pops
import pops.lib.time as libtime
import pytest
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import SlipWall
from pops.domain import Rectangle
from pops.frames import Cartesian2D, Z_AXIS
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.physics import Axial, Density, Momentum
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt
from tests.python.support.native_execution_context import artifact_execution_context


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _axial_wall_case() -> tuple[pops.Case, Uniform]:
    frame = Rectangle(
        "axial-wall-square", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("axial-wall-model", frame=frame)
    state = model.state(
        "U",
        components=("rho", "mx", "my", "bz"),
        representation=Conservative(),
        space=CellState(frame=frame),
        roles={
            "rho": Density(),
            "mx": Momentum(axis=x_axis),
            "my": Momentum(axis=y_axis),
            "bz": Axial(axis=Z_AXIS),
        },
    )
    rho, mx, my, bz = state
    flux = model.flux(
        "identity-flux",
        frame=frame,
        state=state,
        components={
            x_axis: (rho, mx, my, bz),
            y_axis: (rho, mx, my, bz),
        },
        waves={
            x_axis: (1.0, 1.0, 1.0, 1.0),
            y_axis: (1.0, 1.0, 1.0, 1.0),
        },
    )
    rate = model.rate("transport", equation=ddt(state) == -div(flux))
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

    case = pops.Case("axial-slip-wall-pipeline")
    block = case.block("fluid", model=model)
    numerics.boundaries.add(
        TransportBoundarySet(
            {
                boundary: SlipWall(state=block[state])
                for boundary in frame.boundaries.all
            }
        )
    )
    case.numerics(numerics, block=block)
    program = libtime.ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(0.01))
    case.program(program)
    return case, Uniform(CartesianGrid(frame=frame, cells=(4, 4)))


def test_axial_role_compiles_binds_and_round_trips_native_metadata(
    isolated_native_cache, native_cxx, kokkos_root,
) -> None:
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout = _axial_wall_case()
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial = np.ones((4, 4, 4), dtype=np.float64)
    runtime = pops.bind(
        artifact,
        initial_state={"fluid": initial},
        resources={"execution_context": artifact_execution_context(artifact)},
    )

    assert list(runtime._executor._s.variable_roles("fluid", "conservative")) == [
        "density",
        "momentum_x",
        "momentum_y",
        "axial_z",
    ]
    installed = runtime._executor._boundary_authorities["fluid"]
    assert [face["type"] for face in installed["faces"]] == ["slip_wall"] * 4
