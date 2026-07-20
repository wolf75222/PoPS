"""Public AMR patch geometry after a real, clocked regrid.

The only runtime object used here is the :func:`pops.bind` result.  In particular, patch geometry
is read from the public ``RuntimeInstance`` after an AMR execution whose 2:1 level-clock relation
was explicitly authored.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pops
import pytest
from pops.analytic import coordinates
from pops.amr import (
    AMRClockRelation,
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Analytic, Gaussian
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics import Model
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, StagePoint, TimePoint, every


ROOT = Path(__file__).resolve().parents[4]
N = 32
DT = 1.0e-4

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _scalar_model(name, frame):
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.25 * rho,), y_axis: (-0.1 * rho,)},
        waves={x_axis: (0.25,), y_axis: (-0.1,)},
    )
    rate = model.rate("transport_rate", equation=ddt(state) == -div(flux))
    return model, state, flux, rate


def _program(rows):
    program = pops.Program("amr_patch_boxes_forward_euler_%d_block" % len(rows))
    endpoints = []
    for name, state, rate in rows:
        temporal = program.state(state)
        stage = StagePoint(name + "_stage", {"main": TimePoint(program.clock, 0)})
        rhs = program.value(name + "_rhs", rate(temporal.n), at=stage)
        next_value = program.value(
            name + "_next", temporal.n + program.dt * rhs, at=temporal.next.point)
        endpoints.append((temporal.next, next_value))
    for endpoint, value in endpoints:
        program.commit(endpoint, value)
    program.step_strategy(FixedDt(DT))
    return program


def _resolved(native_cxx, block_count):
    frame = Rectangle(
        "amr-patch-box-domain",
        (0.0, 0.0),
        (1.0, 1.0),
    ).frame(Cartesian2D())
    case = pops.Case("amr-patch-box-case-%d-block" % block_count)
    block_specs = []
    definitions = (
        ("tracer", (0.35, 0.55)),
        ("reference", (0.65, 0.45)),
    )[:block_count]
    for name, center in definitions:
        model, state, flux, rate = _scalar_model("patch-%s" % name, frame)
        block = case.block(name, model, states=(state,))
        block_specs.append((name, block, block[state], state, flux, rate, center))
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
    program = _program(tuple(
        (name, instance, rate)
        for name, _block, instance, _state, _flux, rate, _center in block_specs
    ))
    case.program(program)
    x_axis, y_axis = frame.axes
    x_coord, y_coord = coordinates(frame)
    for name, _block, instance, _state, _flux, _rate, center in block_specs:
        profile = Gaussian(
            frame=frame, center={x_axis: center[0], y_axis: center[1]},
            background=0.0, amplitude=1.0, inverse_width=120.0,
        )
        if name == "reference":
            profile = Analytic(
                frame=frame,
                components=(
                    0.25 + x_coord * x_coord + 2.0 * y_coord + x_coord * y_coord,
                ),
            )
        case.initials.add(InitialCondition(
            state=instance,
            value=profile,
            projection=ConservativeCellAverage(),
        ))
    threshold = case.param(RuntimeParam("patch_refine_threshold", default=0.2))
    transfer = AMRTransfer()
    for _name, _block, instance, _state, _flux, _rate, _center in block_specs:
        transfer.state(instance, StateTransfer())
    tracer_instance = block_specs[0][2]
    layout = AMR(
        grid=CartesianGrid(frame=frame, cells=(N, N), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(tracer_instance) > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(1, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": native_cxx},
    )


def _assert_public_patch_geometry(simulation):
    boxes = simulation.patch_boxes()
    rectangles = simulation.patch_rectangles()
    assert isinstance(boxes, list)
    assert len(boxes) == len(rectangles) > 0
    for level, ilo, jlo, ihi, jhi in boxes:
        limit = N * (2 ** level)
        assert level >= 1
        assert 0 <= ilo <= ihi < limit
        assert 0 <= jlo <= jhi < limit
    for x0, y0, width, height in rectangles:
        assert width > 0.0 and height > 0.0
        assert 0.0 <= x0 <= x0 + width <= 1.0
        assert 0.0 <= y0 <= y0 + height <= 1.0
    return boxes, rectangles


def test_public_amr_patch_boxes_are_parallel_to_rectangles_and_read_only(
    native_cxx, isolated_native_cache, kokkos_root,
):
    del isolated_native_cache, kokkos_root
    for block_count in (1, 2):
        simulation = pops.bind(pops.compile(_resolved(native_cxx, block_count)))
        if block_count == 2:
            centers = (np.arange(N, dtype=np.float64) + 0.5) / N
            x_coord, y_coord = np.meshgrid(centers, centers, indexing="xy")
            dx = 1.0 / N
            expected_reference = (
                0.25 + x_coord * x_coord + dx * dx / 12.0
                + 2.0 * y_coord + x_coord * y_coord
            )
            initial_reference = np.asarray(
                simulation.block_level_state_global("reference", 0), dtype=np.float64,
            ).reshape((1, N, N))[0]
            np.testing.assert_allclose(
                initial_reference, expected_reference, rtol=0.0, atol=2.0e-14,
            )
        report = pops.run(simulation, t_end=2.0 * DT, max_steps=2)
        expected_names = ("tracer",) if block_count == 1 else ("tracer", "reference")
        assert simulation.block_names() == expected_names
        before = {
            name: np.asarray(
                simulation.block_level_state_global(name, 0), dtype=np.float64,
            ).copy()
            for name in simulation.block_names()
        }
        first_boxes, first_rectangles = _assert_public_patch_geometry(simulation)
        second_boxes, second_rectangles = _assert_public_patch_geometry(simulation)
        _ = simulation.patch_boxes()
        assert report.accepted_steps == simulation.macro_step() == 2
        assert simulation.n_levels() == 2
        assert second_boxes == first_boxes
        assert second_rectangles == first_rectangles
        for name, expected in before.items():
            actual = np.asarray(
                simulation.block_level_state_global(name, 0), dtype=np.float64,
            )
            np.testing.assert_array_equal(actual, expected)
