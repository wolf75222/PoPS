"""Full two-level subcycled AMR transport-diffusion through the generated Program."""

from __future__ import annotations

from pathlib import Path

import pops
import pytest
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
from pops.lib.initial import Gaussian
from pops.lib.time import ForwardEuler
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every
from tests.python.support.amr_snapshots import composite_active_block_state
from tests.python.support.native_execution_context import artifact_execution_context


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
ROOT = Path(__file__).resolve().parents[4]
DT = 2.0e-4
CELLS = 12


def _case_and_layout():
    frame = Rectangle("amr-transport-diffusion-domain", lower=(0.0, 0.0),
                      upper=(1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("amr-transport-diffusion-model", frame=frame)
    state = model.state("U", components=("u",))
    (u,) = state
    velocity = (0.35, -0.2)
    transport = model.flux(
        "transport", frame=frame, state=state,
        components={axis: (speed * u,) for axis, speed in zip(frame.axes, velocity, strict=True)},
        waves={axis: (speed,) for axis, speed in zip(frame.axes, velocity, strict=True)},
    )
    diffusion = model.diffusive_flux(
        "diffusion", state=state, value=0.05 * pops.math.grad(u))
    rate = model.rate(
        "transport-diffusion", equation=ddt(state) == -div(transport) + div(diffusion))

    case = pops.Case("amr-transport-diffusion-case")
    block = case.block("heat", model)
    block_state = block[state]
    plan = DiscretizationPlan()
    plan.rates.add(
        rate,
        Diffusion(
            flux=diffusion,
            transport=FiniteVolume(
                flux=transport,
                variables=variables.Conservative(state),
                reconstruction=reconstruction.FirstOrder(),
                riemann=riemann.Rusanov(),
            ),
        ),
    )
    case.numerics(plan, block=block)
    program = ForwardEuler(block_state, rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    case.initials.add(InitialCondition(
        state=block_state,
        value=Gaussian(
            frame=frame,
            center={frame.x: 0.37, frame.y: 0.43},
            background=1.0,
            amplitude=0.3,
            inverse_width=70.0,
        ),
        projection=ConservativeCellAverage(),
    ))
    threshold = case.param(RuntimeParam("refine-threshold", default=1.04))
    transfer = AMRTransfer()
    transfer.state(block_state, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(frame=frame, cells=(CELLS, CELLS),
                           periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(block_state) > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(1, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )
    return case, layout


def _composite_mass(simulation):
    mass = 0.0
    for level in range(simulation.n_levels()):
        active = composite_active_block_state(
            simulation, "heat", level, refinement_ratio=2)
        mass += float(active.sum()) / (CELLS * (2**level)) ** 2
    return mass


def test_generated_amr_transport_diffusion_subcycles_regrids_and_reconciles_fluxes(
    isolated_native_cache, native_cxx, kokkos_root,
):
    del isolated_native_cache, kokkos_root
    case, layout = _case_and_layout()
    artifact = pops.compile(pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": native_cxx},
    ))
    simulation = pops.bind(
        artifact,
        resources={"execution_context": artifact_execution_context(artifact)},
    )
    assert simulation.n_levels() == 2
    assert any(level == 1 for level, _lower, _upper in simulation.patch_boxes())
    mass_before = _composite_mass(simulation)

    report = pops.run(simulation, t_end=2 * DT, max_steps=2, console=False)
    assert report.accepted_steps == 2 and report.rejected_steps == 0
    assert simulation.n_levels() == 2
    assert abs(_composite_mass(simulation) - mass_before) < 5.0e-11

    manifest = tuple(tuple(map(str, row)) for row in
                     simulation._executor.program_flux_ledger_manifest())
    assert manifest
    flattened = "\n".join("/".join(row) for row in manifest)
    assert "provider/1" in flattened
    assert "provider/4" in flattened
