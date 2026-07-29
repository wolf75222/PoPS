"""Operational RegridOnRestart over one real artifact-backed AMR hierarchy.

The source run deliberately advances a compact scalar-advection profile while the ordinary regrid
cadence is held beyond the test window. Its checkpoint therefore contains an accepted field whose
recorded fine boxes are stale relative to the installed tagger. RegridOnRestart must first restore
that exact accepted image, then execute one real native tag/cluster/regrid pass at the same physical
time and macro-step.

The test also injects a failure after the native topology mutation. The restart transaction must
restore the fresh runtime bit-for-bit before a second, successful restart can publish its weaker
continuation identity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pops
import pytest
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
    PatchLayout,
    Tag,
)
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import BergerRigoutsos, StateTransfer
from pops.lib.initial import Gaussian
from pops.lib.time import SSPRK2
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.output import Checkpoint, ConsumerGraph, RegridOnRestart
from pops.params import RuntimeParam
from pops.physics import Model
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every
from tests.python.support.native_execution_context import artifact_execution_context


ROOT = Path(__file__).resolve().parents[4]
N = 16
DT = 1.0e-2
NSTEPS = 14

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _resolved(native_cxx):
    frame = Rectangle(
        "regrid-on-restart-domain",
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("regrid-on-restart-advection", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={
            x_axis: (rho,),
            y_axis: (0.0 * rho,),
        },
        waves={
            x_axis: (1.0 + 0.0 * rho,),
            y_axis: (0.0 * rho,),
        },
    )
    rate = model.rate("transport_rate", equation=ddt(state) == -div(flux))

    case = pops.Case("regrid-on-restart-case")
    block = case.block("tracer", model)
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
    program = SSPRK2(block_state, rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=block_state,
            value=Gaussian(
                frame=frame,
                center={x_axis: 0.25, y_axis: 0.5},
                background=1.0,
                amplitude=0.5,
                inverse_width=140.0,
            ),
            projection=ConservativeCellAverage(),
        )
    )

    threshold = case.param(RuntimeParam("regrid_on_restart_threshold", default=1.08))
    transfer = AMRTransfer()
    transfer.state(block_state, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(N, N),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(block_state) > case.value(threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        # Bootstrap creates the initial fine layout, but no ordinary regrid fires in this test.
        regrid=AMRRegrid(schedule=every(1_000, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
        patch_layout=PatchLayout(distribute_coarse=True, coarse_max_grid=8),
        clustering=BergerRigoutsos(maximum_box_size=8),
    )
    case.consumers(
        ConsumerGraph.from_consumers(
            (
                Checkpoint(
                    schedule=every(10_000, clock=program.clock),
                    target="unused/regrid-on-restart",
                    hierarchy=RegridOnRestart(),
                ),
            )
        )
    )
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": native_cxx},
    )


def _bind(artifact):
    return pops.bind(
        artifact,
        resources={"execution_context": artifact_execution_context(artifact)},
    )


def _accepted_image(runtime):
    native = runtime._executor._s
    levels = int(runtime.n_levels())
    return {
        "time": float(runtime.time()),
        "step": int(runtime.macro_step()),
        "boxes": tuple(tuple(int(value) for value in box) for box in runtime.patch_boxes()),
        "regrid_count": int(native.checkpoint_regrid_count()),
        "topology_epoch": int(native.checkpoint_topology_epoch()),
        "states": tuple(
            np.asarray(
                runtime.block_level_state_global("tracer", level),
                dtype=np.float64,
            ).copy()
            for level in range(levels)
        ),
        "program_state": bytes(native.program_accepted_state()),
        "run_identity": runtime.last_run_identity,
    }


def _assert_same_accepted_image(runtime, expected):
    actual = _accepted_image(runtime)
    assert {
        key: value for key, value in actual.items() if key != "states"
    } == {
        key: value for key, value in expected.items() if key != "states"
    }
    assert len(actual["states"]) == len(expected["states"])
    for current, recorded in zip(actual["states"], expected["states"], strict=True):
        np.testing.assert_array_equal(current, recorded)


def test_regrid_on_restart_changes_real_boxes_and_rolls_back_post_regrid_fault(
    native_cxx,
    isolated_native_cache,
    kokkos_root,
    monkeypatch,
    tmp_path,
):
    del isolated_native_cache, kokkos_root
    artifact = pops.compile(_resolved(native_cxx))
    source = _bind(artifact)
    bootstrap_boxes = tuple(source.patch_boxes())
    bootstrap_regrids = source._executor._s.checkpoint_regrid_count()
    report = pops.run(
        source,
        t_end=NSTEPS * DT,
        max_steps=NSTEPS,
        console=False,
    )
    assert report.accepted_steps == NSTEPS
    assert tuple(source.patch_boxes()) == bootstrap_boxes
    assert source._executor._s.checkpoint_regrid_count() == bootstrap_regrids
    source_time = float(source.time())
    source_step = int(source.macro_step())
    source_run_identity = source.last_run_identity
    checkpoint = source.checkpoint(tmp_path / "moving-profile")

    restarted = _bind(artifact)
    rollback_image = _accepted_image(restarted)
    from pops.runtime import _amr_checkpoint_v3 as checkpoint_codec

    original_conservation_check = checkpoint_codec._require_restart_conservation
    transformed_boxes = []

    def fail_after_native_regrid(before, after):
        del before, after
        transformed_boxes.append(tuple(restarted.patch_boxes()))
        raise RuntimeError("injected post-regrid validation failure")

    monkeypatch.setattr(
        checkpoint_codec,
        "_require_restart_conservation",
        fail_after_native_regrid,
    )
    with pytest.raises(RuntimeError, match="injected post-regrid validation failure"):
        restarted.restart(checkpoint)
    assert transformed_boxes and transformed_boxes[0] != rollback_image["boxes"]
    _assert_same_accepted_image(restarted, rollback_image)
    assert restarted._executor.last_restart_regrid_receipt() is None

    monkeypatch.setattr(
        checkpoint_codec,
        "_require_restart_conservation",
        original_conservation_check,
    )
    restart_identity = restarted.restart(checkpoint)
    receipt = restarted._executor.last_restart_regrid_receipt()
    assert receipt["changed"] is True
    assert receipt["before"]["topology_identity"] != receipt["after"]["topology_identity"]
    assert tuple(restarted.patch_boxes()) == transformed_boxes[0]
    assert tuple(restarted.patch_boxes()) != bootstrap_boxes
    assert float(restarted.time()) == receipt["accepted_time"] == source_time
    assert int(restarted.macro_step()) == receipt["accepted_macro_step"] == source_step
    assert restart_identity == restarted.last_restart_identity
    assert restarted.last_run_identity.domain == "run"
    assert restarted.last_run_identity != source_run_identity

    before = receipt["composite_integrals_before"]
    after = receipt["composite_integrals_after"]
    assert [(row["block"], row["component"]) for row in before] == [
        (row["block"], row["component"]) for row in after
    ]
    np.testing.assert_allclose(
        [row["value"] for row in after],
        [row["value"] for row in before],
        rtol=2.0e-12,
        atol=2.0e-13,
    )
