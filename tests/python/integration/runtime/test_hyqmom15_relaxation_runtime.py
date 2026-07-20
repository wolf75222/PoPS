"""Native numerical parity for the public HyQMOM15 relaxation transform."""

from __future__ import annotations

import numpy as np
import pops
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import HyQMOM15Relaxation, moment_names
from pops.moments import _relaxation_reference as relaxation_reference
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt


CELLS = 4
DT = 0.125
SOURCE = np.array(
    [
        1.0, 0.0, 1.0, 6.0, 46.0,
        0.0, 0.99, 0.1, 0.0,
        1.0, 0.0, 3.0,
        1.0, 0.0,
        4.0,
    ],
    dtype=np.float64,
)
MATLAB_RELAXED = np.array(
    [
        1.0, 0.0, 1.0, 1.5, 8.52985,
        0.0, 0.99, 1.485, 0.0,
        1.0, 1.5, 8.52985,
        0.325, 0.0,
        8.52985,
    ],
    dtype=np.float64,
)

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _case_and_layout():
    frame = Rectangle(
        "hyqmom15-relaxation-square", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes

    model = pops.Model("native-hyqmom15-relaxation", frame=frame)
    state = model.state(
        "U",
        components=tuple(moment_names(4)),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    zero_flux = tuple(0.0 * component for component in state)
    zero_waves = tuple(0.0 for _component in state)
    flux = model.flux(
        "zero-flux",
        frame=frame,
        state=state,
        components={x_axis: zero_flux, y_axis: zero_flux},
        waves={x_axis: zero_waves, y_axis: zero_waves},
    )
    rate = model.rate("zero-rate", equation=ddt(state) == -div(flux))
    relaxation = HyQMOM15Relaxation(spectral_tolerance=1.0e-14)
    transform = relaxation.declare(model, state)

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

    case = pops.Case("native-hyqmom15-relaxation-case")
    block = case.block("plasma", model=model)
    case.numerics(numerics, block=block)

    program = pops.Program("native-hyqmom15-relaxation-program")
    moments = program.state(block[state])
    rhs = rate(moments.n)
    candidate = program.value(
        "candidate", moments.n + program.dt * rhs, at=moments.next.point
    )
    relaxed = program.transform(
        candidate, transform=transform, name="relaxed-candidate"
    )
    program.commit(moments.next, relaxed)
    program.step_strategy(FixedDt(DT))
    case.program(program)

    layout = Uniform(
        CartesianGrid(
            frame=frame,
            cells=(CELLS, CELLS),
            periodic=PeriodicAxes(frame.axes),
        )
    )
    return case, layout, relaxation


def test_hyqmom15_relaxation_matches_matlab_through_native_program(
    isolated_native_cache, native_cxx, kokkos_root, monkeypatch,
) -> None:
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, relaxation = _case_and_layout()

    density_scale = np.linspace(0.75, 1.50, CELLS * CELLS, dtype=np.float64).reshape(
        (1, CELLS, CELLS)
    )
    initial = SOURCE[:, None, None] * density_scale
    expected = relaxation_reference._apply_hyqmom15_relaxation_array(
        initial,
        cutoff=relaxation.eigenvalue_cutoff,
        mach=relaxation.mach,
        small=relaxation.small,
        spectral_tolerance=relaxation.spectral_tolerance,
    )
    np.testing.assert_allclose(
        expected[:, 0, 0], MATLAB_RELAXED * density_scale[0, 0, 0],
        rtol=2.0e-12, atol=2.0e-12,
    )

    def forbidden_python_oracle(*_args, **_kwargs):
        raise AssertionError("native relaxation called the Python reference oracle")

    monkeypatch.setattr(
        relaxation_reference,
        "_apply_hyqmom15_relaxation_array",
        forbidden_python_oracle,
    )

    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    artifact.verify()
    simulation = pops.bind(artifact, initial_state={"plasma": initial})
    report = pops.run(simulation, t_end=DT, max_steps=1)

    assert report.accepted_steps == 1
    actual = np.asarray(
        simulation.state_global("plasma"), dtype=np.float64
    ).reshape(initial.shape)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-11, atol=3.0e-11)

    invalid = initial.copy()
    invalid[0, :, :] = 0.0
    rejected = pops.bind(artifact, initial_state={"plasma": invalid})
    with pytest.raises(RuntimeError, match="relaxation15|local_transform"):
        pops.run(rejected, t_end=DT, max_steps=1)
    unchanged = np.asarray(
        rejected.state_global("plasma"), dtype=np.float64
    ).reshape(invalid.shape)
    np.testing.assert_array_equal(unchanged, invalid)
