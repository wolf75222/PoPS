#!/usr/bin/env python3
"""Process probe for the public AMR checkpoint restart rank-topology contract.

The serial pytest driver starts this file in separate MPI jobs.  ``capture`` runs a genuinely
transporting scalar through coarse/fine interfaces on two ranks and publishes one accepted-state
checkpoint; ``restart`` runs with one rank and must rematerialize the recorded hierarchy ownership
and non-zero flux history without changing its geometry, global state, accepted clock, or dense
Program histories.  A second strict-policy checkpoint proves that ``bit_identical=True`` continues
to reject a two-to-one rank change before mutating the fresh runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    # Keep the assembled/installed package first so this probe never hides its native extension
    # with a source-tree Python overlay.  The checkout is needed only for shared test support.
    sys.path.append(str(ROOT))

from _compile_once import compile_resolved_plan_once  # noqa: E402
from tests.python.support.requirements import require_mpi_or_skip  # noqa: E402

try:
    import numpy as np

    import pops
    from pops import _pops
    from pops._native_collectives import barrier
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
    from pops.analytic import coordinates, exp, maximum
    from pops.codegen import Production
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.initial import InitialCondition
    from pops.layouts import AMR
    from pops.lib.amr import BergerRigoutsos, StateTransfer
    from pops.lib.initial import Analytic
    import pops.lib.time as libtime
    from pops.math import ValueExpr, ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.output import Checkpoint, ConsumerGraph
    from pops.params import RuntimeParam
    from pops.physics import Model
    from pops.projection import ConservativeCellAverage
    from pops.time import FixedDt, every
except (Exception, SystemExit) as exc:
    require_mpi_or_skip("AMR rank-change probe imports are unavailable: %s" % exc)


N = 16
DT = 2.0e-3
CHECKPOINT_STEPS = 3
CONTINUATION_STEPS = 3
if getattr(_pops, "__has_mpi__", False) is not True:
    require_mpi_or_skip("AMR rank-change probe requires an MPI-enabled native module")
_COMM = _pops.mpi_world()


def _resolved(*, bit_identical: bool) -> Any:
    """Build one light two-island AMR case with a dense AB2 Program history."""
    policy = "strict" if bit_identical else "rematerialized"
    frame = Rectangle(
        "amr-rank-change-%s-domain" % policy,
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("amr-rank-change-%s-model" % policy, frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "nonzero-transport",
        frame=frame,
        state=state,
        components={x_axis: (rho,), y_axis: (0.25 * rho,)},
        waves={
            x_axis: (1.0 + 0.0 * rho,),
            y_axis: (0.25 + 0.0 * rho,),
        },
    )
    rate = model.rate(
        "transport-rate",
        equation=ddt(state) == -div(flux),
    )

    case = pops.Case("amr-rank-change-%s-case" % policy)
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
    program = libtime.AdamsBashforth(block_state, rate=rate, order=2)
    program.step_strategy(FixedDt(DT))
    case.program(program)

    x_coord, y_coord = coordinates(frame)
    left_radius_squared = (
        (x_coord - 0.25) * (x_coord - 0.25)
        + (y_coord - 0.25) * (y_coord - 0.25)
    )
    right_radius_squared = (
        (x_coord - 0.75) * (x_coord - 0.75)
        + (y_coord - 0.75) * (y_coord - 0.75)
    )
    two_islands = 1.0 + 0.5 * maximum(
        exp(-220.0 * left_radius_squared),
        exp(-220.0 * right_radius_squared),
    )
    case.initials.add(
        InitialCondition(
            state=block_state,
            value=Analytic(frame=frame, components=(two_islands,)),
            projection=ConservativeCellAverage(),
        )
    )

    threshold = case.param(RuntimeParam("rank_change_refine_threshold", default=1.12))
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
        regrid=AMRRegrid(schedule=every(2, clock=program.clock)),
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
                    target="unused/rank-change-checkpoint",
                    bit_identical=bit_identical,
                ),
            )
        )
    )
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include")},
    )


def _runtime(*, bit_identical: bool) -> Any:
    route = "amr-rank-change/%s" % ("strict" if bit_identical else "rematerialized")
    artifact = compile_resolved_plan_once(
        _COMM,
        _resolved(bit_identical=bit_identical),
        route=route,
        compile_artifact=pops.compile,
    )
    context = pops.ExecutionContext.mpi_world(artifact)
    return pops.bind(artifact, resources={"execution_context": context})


def _advance(runtime: Any, steps: int) -> None:
    report = pops.run(
        runtime,
        t_end=float(runtime.time()) + steps * DT,
        max_steps=steps,
    )
    if report.accepted_steps != steps:
        raise AssertionError(
            "rank-change probe accepted %d/%d requested steps"
            % (report.accepted_steps, steps)
        )


def _capture_arrays(runtime: Any, *, prefix: str) -> tuple[dict[str, Any], dict[str, Any]]:
    levels = int(runtime.n_levels())
    names = tuple(runtime.history_names())
    arrays: dict[str, Any] = {}
    for level in range(levels):
        arrays["%s_state_level_%d" % (prefix, level)] = np.asarray(
            runtime.block_level_state_global("tracer", level),
            dtype=np.float64,
        ).copy()
    histories = []
    for history_index, name in enumerate(names):
        depth = int(runtime.history_depth(name))
        ncomp = int(runtime.history_ncomp(name))
        histories.append({"name": name, "depth": depth, "ncomp": ncomp})
        for slot in range(depth):
            arrays["%s_history_%d_slot_%d" % (prefix, history_index, slot)] = np.asarray(
                runtime.history_global(name, slot),
                dtype=np.float64,
            ).copy()
    report = runtime.amr.explain_regrid()
    program = runtime.program_report()
    if not program.flux_ledger or not any(
        int(entry["level"]) == 1 for entry in program.flux_ledger
    ):
        raise AssertionError(
            "rank-change proof requires an accepted non-zero-transport flux ledger "
            "on the refined level"
        )
    synchronization = {str(event["phase"]) for event in program.synchronization}
    if not {"reflux", "average_down"} <= synchronization:
        raise AssertionError(
            "rank-change proof requires accepted reflux and average-down events"
        )
    metadata = {
        "time_hex": float(runtime.time()).hex(),
        "macro_step": int(runtime.macro_step()),
        "n_levels": levels,
        "patch_boxes": [list(row) for row in runtime.patch_boxes()],
        "histories": histories,
        "regrid_count": int(report.regrid_count),
        "topology_epoch": int(report.topology_epoch),
        "flux_ledger": program.flux_ledger,
        "synchronization": program.synchronization,
    }
    return metadata, arrays


def _write_evidence(
    path: Path,
    *,
    checkpoint_metadata: dict[str, Any],
    checkpoint_arrays: dict[str, Any],
    final_metadata: dict[str, Any],
    final_arrays: dict[str, Any],
    source_owners: tuple[int, ...],
    initial_mass: float,
) -> None:
    metadata = {
        "schema_version": 1,
        "checkpoint": checkpoint_metadata,
        "final": final_metadata,
        "source_fine_owners": list(source_owners),
        "initial_mass_hex": initial_mass.hex(),
    }
    payload = dict(checkpoint_arrays)
    payload.update(final_arrays)
    payload["metadata"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        np.savez_compressed(stream, **payload)


def _load_evidence(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"]))
        arrays = {
            name: np.asarray(stored[name]).copy()
            for name in stored.files
            if name != "metadata"
        }
    if metadata.get("schema_version") != 1:
        raise ValueError("rank-change evidence has an unsupported schema")
    return metadata, arrays


def _assert_snapshot(
    runtime: Any,
    *,
    expected_metadata: dict[str, Any],
    expected_arrays: dict[str, np.ndarray],
    prefix: str,
) -> None:
    actual_metadata, actual_arrays = _capture_arrays(runtime, prefix=prefix)
    if actual_metadata != expected_metadata:
        raise AssertionError(
            "restored AMR metadata differs:\nexpected=%r\nactual=%r"
            % (expected_metadata, actual_metadata)
        )
    for name, actual in actual_arrays.items():
        expected = expected_arrays[name]
        if not np.array_equal(actual, expected):
            raise AssertionError(
                "%s differs after rank rematerialization (max|d|=%.17g)"
                % (name, float(np.max(np.abs(actual - expected))))
            )


def _checkpoint_source_owners(checkpoint: Path) -> tuple[int, ...]:
    with np.load(checkpoint, allow_pickle=False) as payload:
        if int(payload["n_ranks"]) != 2 or int(payload["n_levels"]) != 2:
            raise AssertionError("capture checkpoint must record exactly two ranks and two levels")
        boxes = np.asarray(payload["patch_boxes"], dtype=np.int64)
        owners = tuple(
            int(value)
            for value in np.asarray(payload["dmap_rank_0_level_1"], dtype=np.int64)
        )
        if boxes.ndim != 2 or boxes.shape[1] != 5:
            raise AssertionError("capture checkpoint has malformed AMR patch geometry")
        fine_boxes = sum(int(row[0]) == 1 for row in boxes)
        if fine_boxes < 2 or len(owners) != fine_boxes:
            raise AssertionError(
                "rank-change proof requires at least two aligned fine patches "
                "(boxes=%d, owners=%d)" % (fine_boxes, len(owners))
            )
        if set(owners) != {0, 1}:
            raise AssertionError(
                "fine patches were not distributed across both source ranks: %r" % (owners,)
            )
        for rank in range(2):
            for level in range(2):
                peer = tuple(
                    int(value)
                    for value in np.asarray(
                        payload["dmap_rank_%d_level_%d" % (rank, level)],
                        dtype=np.int64,
                    )
                )
                root = tuple(
                    int(value)
                    for value in np.asarray(
                        payload["dmap_rank_0_level_%d" % level],
                        dtype=np.int64,
                    )
                )
                if peer != root:
                    raise AssertionError(
                        "source rank %d records a divergent level-%d ownership map"
                        % (rank, level)
                    )
    return owners


def _assert_single_rank_checkpoint(checkpoint: Path) -> None:
    with np.load(checkpoint, allow_pickle=False) as payload:
        if int(payload["n_ranks"]) != 1 or int(payload["n_levels"]) != 2:
            raise AssertionError(
                "post-restart checkpoint must record one rank and the same two levels"
            )
        for level in range(2):
            owners = tuple(
                int(value)
                for value in np.asarray(
                    payload["dmap_rank_0_level_%d" % level],
                    dtype=np.int64,
                )
            )
            if not owners or set(owners) != {0}:
                raise AssertionError(
                    "level-%d ownership was not rematerialized entirely onto rank 0: %r"
                    % (level, owners)
                )


def _capture(checkpoint: Path, evidence: Path | None, *, bit_identical: bool) -> None:
    if int(_COMM.size) != 2:
        require_mpi_or_skip(
            "AMR rank-change capture requires exactly two MPI ranks (observed %d)"
            % int(_COMM.size)
        )
    runtime = _runtime(bit_identical=bit_identical)
    initial_state = np.asarray(
        runtime.block_level_state_global("tracer", 0), dtype=np.float64
    ).copy()
    initial_mass = float(runtime.integral("tracer", levels=(0,)))
    _advance(runtime, CHECKPOINT_STEPS)
    checkpoint_metadata, checkpoint_arrays = _capture_arrays(
        runtime, prefix="checkpoint"
    )
    if checkpoint_metadata["n_levels"] != 2:
        raise AssertionError("rank-change capture did not build its second AMR level")
    checkpoint_state = checkpoint_arrays["checkpoint_state_level_0"]
    transport_change = float(np.max(np.abs(checkpoint_state - initial_state)))
    if not np.isfinite(transport_change) or transport_change <= 1.0e-12:
        raise AssertionError(
            "rank-change proof did not observe a non-zero transport update "
            "(max|U-U0|=%.17g)" % transport_change
        )
    checkpoint_mass = float(runtime.integral("tracer", levels=(0,)))
    if abs(checkpoint_mass - initial_mass) >= 1.0e-8:
        raise AssertionError(
            "rank-change capture lost mass before checkpoint "
            "(|M-M0|=%.17g)" % abs(checkpoint_mass - initial_mass)
        )
    published = Path(runtime.checkpoint(checkpoint))
    barrier(_COMM)
    source_owners = _checkpoint_source_owners(published) if int(_COMM.rank) == 0 else ()

    if not bit_identical:
        _advance(runtime, CONTINUATION_STEPS)
        final_metadata, final_arrays = _capture_arrays(runtime, prefix="final")
        final_mass = float(runtime.integral("tracer", levels=(0,)))
        if abs(final_mass - initial_mass) >= 1.0e-8:
            raise AssertionError(
                "uninterrupted two-rank continuation lost mass "
                "(|M-M0|=%.17g)" % abs(final_mass - initial_mass)
            )
        if evidence is None:
            raise ValueError("relaxed rank-change capture requires an evidence path")
        if int(_COMM.rank) == 0:
            _write_evidence(
                evidence,
                checkpoint_metadata=checkpoint_metadata,
                checkpoint_arrays=checkpoint_arrays,
                final_metadata=final_metadata,
                final_arrays=final_arrays,
                source_owners=source_owners,
                initial_mass=initial_mass,
            )
    barrier(_COMM)
    if int(_COMM.rank) == 0:
        print(
            "PASS capture %s AMR checkpoint on two ranks"
            % ("strict" if bit_identical else "rematerialized"),
            flush=True,
        )


def _restart_relaxed(checkpoint: Path, evidence: Path, rematerialized: Path) -> None:
    if int(_COMM.size) != 1:
        require_mpi_or_skip(
            "AMR rank-change restart requires exactly one MPI rank (observed %d)"
            % int(_COMM.size)
        )
    metadata, arrays = _load_evidence(evidence)
    runtime = _runtime(bit_identical=False)
    runtime.restart(checkpoint)
    _assert_snapshot(
        runtime,
        expected_metadata=metadata["checkpoint"],
        expected_arrays=arrays,
        prefix="checkpoint",
    )
    initial_mass = float.fromhex(str(metadata["initial_mass_hex"]))
    restored_mass = float(runtime.integral("tracer", levels=(0,)))
    if abs(restored_mass - initial_mass) >= 1.0e-8:
        raise AssertionError(
            "one-rank restart changed the conserved mass "
            "(|M-M0|=%.17g)" % abs(restored_mass - initial_mass)
        )

    # This second public checkpoint is the observable ownership witness: every recorded
    # DistributionMapping must now contain only rank zero, without reaching into the native engine.
    post_restart = Path(runtime.checkpoint(rematerialized))
    _assert_single_rank_checkpoint(post_restart)
    _assert_snapshot(
        runtime,
        expected_metadata=metadata["checkpoint"],
        expected_arrays=arrays,
        prefix="checkpoint",
    )

    _advance(runtime, CONTINUATION_STEPS)
    _assert_snapshot(
        runtime,
        expected_metadata=metadata["final"],
        expected_arrays=arrays,
        prefix="final",
    )
    continued_mass = float(runtime.integral("tracer", levels=(0,)))
    if abs(continued_mass - initial_mass) >= 1.0e-8:
        raise AssertionError(
            "one-rank continuation after flux rematerialization lost mass "
            "(|M-M0|=%.17g)" % abs(continued_mass - initial_mass)
        )
    print(
        "PASS restart AMR checkpoint from two ranks onto one rank and continue",
        flush=True,
    )


def _restart_strict(checkpoint: Path) -> None:
    if int(_COMM.size) != 1:
        require_mpi_or_skip(
            "strict AMR rank-change refusal requires exactly one MPI rank (observed %d)"
            % int(_COMM.size)
        )
    runtime = _runtime(bit_identical=True)
    state_before = np.asarray(
        runtime.block_level_state_global("tracer", 0), dtype=np.float64
    ).copy()
    boxes_before = tuple(tuple(int(value) for value in row) for row in runtime.patch_boxes())
    try:
        runtime.restart(checkpoint)
    except Exception as exc:  # noqa: BLE001 -- exact public refusal text is asserted below
        message = str(exc)
        if (
            "bit_identical=True" not in message
            or "recorded MPI rank topology" not in message
            or "checkpoint=2, current=1" not in message
        ):
            raise AssertionError(
                "strict rank-change restart failed for the wrong reason: " + message
            ) from exc
    else:
        raise AssertionError("bit_identical=True accepted a two-to-one MPI rank change")
    state_after = np.asarray(
        runtime.block_level_state_global("tracer", 0), dtype=np.float64
    )
    boxes_after = tuple(tuple(int(value) for value in row) for row in runtime.patch_boxes())
    if (
        runtime.macro_step() != 0
        or runtime.time() != 0.0
        or boxes_after != boxes_before
        or not np.array_equal(state_after, state_before)
    ):
        raise AssertionError("strict rank-topology refusal mutated the fresh runtime")
    print("PASS bit_identical=True refuses AMR two-to-one restart atomically", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("capture-relaxed", "capture-strict", "restart-relaxed", "restart-strict"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--rematerialized-checkpoint", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "capture-relaxed":
        _capture(args.checkpoint, args.evidence, bit_identical=False)
    elif args.mode == "capture-strict":
        _capture(args.checkpoint, None, bit_identical=True)
    elif args.mode == "restart-relaxed":
        if args.evidence is None or args.rematerialized_checkpoint is None:
            raise ValueError(
                "restart-relaxed requires --evidence and --rematerialized-checkpoint"
            )
        _restart_relaxed(
            args.checkpoint,
            args.evidence,
            args.rematerialized_checkpoint,
        )
    else:
        _restart_strict(args.checkpoint)


if __name__ == "__main__":
    main()
