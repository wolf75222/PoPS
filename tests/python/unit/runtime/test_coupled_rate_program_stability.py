"""Public coupled-rate stability from authoring through native AMR execution.

The pure resolution check proves the public AMR contract.  The native check separately exercises
the complete root lifecycle and its numerical result; neither test recreates the retired
``CoupledSource.frequency(expr)`` registration seam or reaches a private runtime engine.
"""

from __future__ import annotations

import numpy as np
import pytest

import pops
from pops.amr import (
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
from pops.lib.initial import Constant
from pops.math import Const, ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import AdaptiveCFL, every
from tests.python.support.requirements import repo_include


GRID_CELLS = 8
COLLISION_FREQUENCY = 4.0
PROGRAM_CFL = 0.25
PROGRAM_DT_BOUND = PROGRAM_CFL / COLLISION_FREQUENCY
INCLUDE = repo_include()


def _coupled_rate_case() -> tuple[pops.Case, AMR]:
    frame = Rectangle("coupled_rate_domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    model = pops.Model("two_species_relaxation", frame=frame)
    electrons = model.species("electrons", state=("ne",))
    ions = model.species("ions", state=("ni",))
    x_axis, y_axis = frame.axes
    electron_flux = model.flux(
        "stationary_electrons",
        frame=frame,
        state=electrons,
        components={x_axis: (0.0 * electrons["ne"],), y_axis: (0.0 * electrons["ne"],)},
        waves={x_axis: (Const(0.0),), y_axis: (Const(0.0),)},
    )
    ion_flux = model.flux(
        "stationary_ions",
        frame=frame,
        state=ions,
        components={x_axis: (0.0 * ions["ni"],), y_axis: (0.0 * ions["ni"],)},
        waves={x_axis: (Const(0.0),), y_axis: (Const(0.0),)},
    )
    electron_transport = model.rate(
        "electron_transport", equation=ddt(electrons) == -div(electron_flux)
    )
    ion_transport = model.rate("ion_transport", equation=ddt(ions) == -div(ion_flux))
    collision = model.coupled_rate(
        "density_exchange",
        inputs=(electrons, ions),
        outputs={
            electrons: (Const(COLLISION_FREQUENCY) * (ions["ni"] - electrons["ne"]),),
            ions: (Const(COLLISION_FREQUENCY) * (electrons["ne"] - ions["ni"]),),
        },
    )

    case = pops.Case("coupled_rate_amr")
    electron_block = case.block("electrons", model, states=(electrons,))
    ion_block = case.block("ions", model, states=(ions,))
    electron_state = electron_block[electrons]
    ion_state = ion_block[ions]
    refine_threshold = case.param(RuntimeParam("refine_threshold", default=0.75))
    electron_numerics = DiscretizationPlan()
    electron_numerics.rates.add(
        electron_transport,
        FiniteVolume(
            flux=electron_flux,
            variables=variables.Conservative(electrons),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    ion_numerics = DiscretizationPlan()
    ion_numerics.rates.add(
        ion_transport,
        FiniteVolume(
            flux=ion_flux,
            variables=variables.Conservative(ions),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(electron_numerics, block=electron_block)
    case.numerics(ion_numerics, block=ion_block)

    program = pops.Program("explicit_coupled_rate")
    electron_time = program.state(electron_state)
    ion_time = program.state(ion_state)
    exchange = collision(electron_time.n, ion_time.n)
    program.commit_many(
        {
            electron_time.next: program.value(
                "electrons_next",
                electron_time.n + program.dt * exchange[electron_block],
                at=electron_time.next.point,
            ),
            ion_time.next: program.value(
                "ions_next",
                ion_time.n + program.dt * exchange[ion_block],
                at=ion_time.next.point,
            ),
        }
    )
    program.set_dt_bound(lambda _program, cfl: cfl / COLLISION_FREQUENCY)
    program.step_strategy(AdaptiveCFL(PROGRAM_CFL))
    case.program(program)

    case.initials.add(
        InitialCondition(
            state=electron_state,
            value=Constant((1.0,)),
            projection=ConservativeCellAverage(),
        )
    )
    case.initials.add(
        InitialCondition(
            state=ion_state,
            value=Constant((0.5,)),
            projection=ConservativeCellAverage(),
        )
    )

    transfer = AMRTransfer()
    transfer.state(electron_state, StateTransfer())
    transfer.state(ion_state, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(GRID_CELLS, GRID_CELLS),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(electron_state) > case.value(refine_threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(100, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
    )
    return case, layout


def _resolve_coupled_rate_case(*, cxx: str | None = None):
    case, layout = _coupled_rate_case()
    compile_options = None if cxx is None else {"include": INCLUDE, "cxx": cxx}
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options=compile_options,
    )


def test_coupled_rate_program_resolves_with_explicit_amr_stability_bound() -> None:
    resolved = _resolve_coupled_rate_case()

    assert resolved.target == "amr_system"
    assert resolved.time.has_dt_bound()
    assert [node["op"] for node in resolved.time.ir_nodes()].count("coupled_rate") == 1
    assert resolved.capabilities["resolution"]["amr_program"]["status"] == "proven"


@pytest.mark.compiler
@pytest.mark.native_loader
def test_coupled_rate_bound_drives_a_conservative_native_amr_step(
    isolated_native_cache, native_cxx, kokkos_root
) -> None:
    del isolated_native_cache, kokkos_root
    artifact = pops.compile(_resolve_coupled_rate_case(cxx=native_cxx))
    simulation = pops.bind(artifact)
    level_count = simulation.n_levels()
    assert level_count == 2
    before = {
        block: tuple(
            np.asarray(simulation.block_level_state_global(block, level), dtype=np.float64).copy()
            for level in range(level_count)
        )
        for block in ("electrons", "ions")
    }

    requested_end = 2.0 * PROGRAM_DT_BOUND
    assert requested_end > PROGRAM_DT_BOUND
    report = pops.run(simulation, t_end=requested_end, max_steps=1)

    assert report.accepted_steps == 1
    step = float(report.final_time)
    assert 0.0 < step <= PROGRAM_DT_BOUND
    assert simulation.time() == step
    assert simulation.n_levels() == level_count
    after = {
        block: tuple(
            np.asarray(simulation.block_level_state_global(block, level), dtype=np.float64).copy()
            for level in range(level_count)
        )
        for block in ("electrons", "ions")
    }

    for level in range(level_count):
        electron_before = before["electrons"][level]
        ion_before = before["ions"][level]
        electron_after = after["electrons"][level]
        ion_after = after["ions"][level]
        electron_expected = electron_before + step * COLLISION_FREQUENCY * (
            ion_before - electron_before
        )
        ion_expected = ion_before + step * COLLISION_FREQUENCY * (electron_before - ion_before)

        assert np.all(electron_after < electron_before)
        assert np.all(ion_after > ion_before)
        np.testing.assert_array_equal(electron_after, electron_expected)
        np.testing.assert_array_equal(ion_after, ion_expected)
        np.testing.assert_array_equal(electron_after + ion_after, electron_before + ion_before)
