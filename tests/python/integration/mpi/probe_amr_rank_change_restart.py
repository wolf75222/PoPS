#!/usr/bin/env python3
"""Process probe for the public AMR checkpoint restart rank-topology contract.

The serial pytest driver starts this file in separate MPI jobs.  ``capture`` runs a genuinely
transporting scalar through coarse/fine interfaces on two ranks and publishes one accepted-state
checkpoint; ``restart`` runs with one rank and must rematerialize the recorded hierarchy ownership
and non-zero flux history plus persistent tagging hysteresis without changing its geometry, global
state, accepted clock, or dense Program histories.  A second strict-policy checkpoint proves that
``bit_identical=True`` continues to reject a two-to-one rank change before mutating the fresh
runtime.
"""

from __future__ import annotations

import argparse
import hashlib
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
    from pops._native_collectives import allgather_value, barrier
    from pops.amr import (
        AMRExecution,
        AMRHierarchy,
        AMRRegrid,
        AMRTagging,
        AMRTransfer,
        Buffer,
        Coarsen,
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
HYSTERESIS_CYCLES = 4
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
    left_radius_squared = (x_coord - 0.25) * (x_coord - 0.25) + (y_coord - 0.25) * (y_coord - 0.25)
    right_radius_squared = (x_coord - 0.75) * (x_coord - 0.75) + (y_coord - 0.75) * (y_coord - 0.75)
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
    coarsen_threshold = case.param(RuntimeParam("rank_change_coarsen_threshold", default=1.10))
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
                Coarsen(ValueExpr(block_state) < case.value(coarsen_threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(HYSTERESIS_CYCLES, EqualityPolicy.HOLD),
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
            "rank-change probe accepted %d/%d requested steps" % (report.accepted_steps, steps)
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
        history_levels = [int(level) for level in runtime.history_levels(name)]
        histories.append({"name": name, "depth": depth, "ncomp": ncomp, "levels": history_levels})
        for history_level in history_levels:
            for slot in range(depth):
                arrays[
                    "%s_history_%d_level_%d_slot_%d" % (prefix, history_index, history_level, slot)
                ] = np.asarray(
                    runtime.history_global(name, history_level, slot),
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
        raise AssertionError("rank-change proof requires accepted reflux and average-down events")
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
    tagging_hysteresis: bytes,
    initial_mass: float,
) -> None:
    metadata = {
        "schema_version": 1,
        "checkpoint": checkpoint_metadata,
        "final": final_metadata,
        "source_fine_owners": list(source_owners),
        "tagging_hysteresis_sha256": hashlib.sha256(tagging_hysteresis).hexdigest(),
        "initial_mass_hex": initial_mass.hex(),
    }
    payload = dict(checkpoint_arrays)
    payload.update(final_arrays)
    payload["checkpoint_tagging_hysteresis"] = np.frombuffer(
        tagging_hysteresis, dtype=np.uint8
    ).copy()
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        np.savez_compressed(stream, **payload)


def _load_evidence(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"]))
        arrays = {
            name: np.asarray(stored[name]).copy() for name in stored.files if name != "metadata"
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


def _accepted_tagging_hysteresis_span(payload: Any) -> tuple[bytes, int]:
    """Extract the opaque persistent-tagging bytes and their authenticated offset."""
    encoded = (
        bytes(payload)
        if isinstance(payload, (bytes, bytearray, memoryview))
        else np.asarray(payload, dtype=np.uint8).reshape(-1).tobytes()
    )
    cursor = 0

    def read_size() -> int:
        nonlocal cursor
        if cursor + 8 > len(encoded):
            raise AssertionError("accepted-state payload is truncated before a size field")
        value = int.from_bytes(encoded[cursor : cursor + 8], "little")
        cursor += 8
        return value

    def skip_string() -> None:
        nonlocal cursor
        cursor += read_size()
        if cursor > len(encoded):
            raise AssertionError("accepted-state string is truncated")

    if encoded[:8] != b"POPSAND3":
        raise AssertionError("checkpoint does not contain exact-ranked accepted-state v3")
    cursor = 8
    cursor += 8  # native dimension
    skip_string()  # exact spatial contract
    cursor += 2 * 8  # topology epoch, materialization generation
    level_count = read_size()
    clock_bytes = level_count * 40
    if cursor + clock_bytes > len(encoded):
        raise AssertionError("accepted-state level clocks are truncated")
    cursor += clock_bytes
    logical_clock_count = read_size()
    for _ in range(logical_clock_count):
        name_size = read_size()
        if cursor + name_size + 8 > len(encoded):
            raise AssertionError("accepted-state logical-clock map is truncated")
        cursor += name_size + 8
    history_count = read_size()
    for _ in range(history_count):
        skip_string()
        cursor += 8  # Program owner
        for _identity in range(4):
            skip_string()
        cursor += 2 * 8  # depth, component count
    history_slot_count = read_size()
    for _ in range(history_slot_count):
        skip_string()
        cursor += 5 * 8  # level, slot, outgoing dt, initialized, fill count
    cursor += 8  # CellTemporalPartitionKind
    provider_size = read_size()
    cursor += provider_size
    cursor += 3 * 8  # topology epoch, synchronization tick, tick denominator
    cell_count = read_size()
    cursor += cell_count * 32  # level, cell id, rung, accepted tick (four i64 words)
    if cursor > len(encoded):
        raise AssertionError("accepted-state temporal partition is truncated")
    tagging_size = read_size()
    if cursor + tagging_size > len(encoded):
        raise AssertionError("accepted-state persistent-tagging payload is truncated")
    return encoded[cursor : cursor + tagging_size], cursor


def _accepted_tagging_hysteresis(payload: Any) -> bytes:
    """Extract the opaque persistent-tagging bytes from exact-ranked accepted-state v3."""
    tagging, _ = _accepted_tagging_hysteresis_span(payload)
    return tagging


def _assert_active_tagging_hysteresis(encoded: bytes) -> None:
    """Require a real accepted transition window, not only an empty schema envelope."""
    if encoded[:8] != b"POPSHYS2" or len(encoded) < 40:
        raise AssertionError("checkpoint persistent-tagging payload is malformed")
    if int.from_bytes(encoded[8:12], "little") != 2:
        raise AssertionError("checkpoint persistent-tagging dimension differs")
    minimum_cycles = int.from_bytes(encoded[12:16], "little")
    cycle = int.from_bytes(encoded[16:24], "little")
    identity_size = int.from_bytes(encoded[24:32], "little")
    count_offset = 32 + identity_size
    if count_offset + 8 > len(encoded):
        raise AssertionError("checkpoint persistent-tagging identity is truncated")
    active_entries = int.from_bytes(encoded[count_offset : count_offset + 8], "little")
    if minimum_cycles != HYSTERESIS_CYCLES or cycle == 0 or active_entries == 0:
        raise AssertionError(
            "rank-change proof requires an active persistent-tagging window "
            "(min_cycles=%d, cycle=%d, entries=%d)" % (minimum_cycles, cycle, active_entries)
        )


def _checkpoint_source_authorities(checkpoint: Path) -> tuple[tuple[int, ...], bytes]:
    with np.load(checkpoint, allow_pickle=False) as payload:
        if int(payload["n_ranks"]) != 2 or int(payload["n_levels"]) != 2:
            raise AssertionError("capture checkpoint must record exactly two ranks and two levels")
        boxes = np.asarray(payload["patch_boxes"], dtype=np.int64)
        owners = tuple(int(value) for value in np.asarray(payload["dmap_1"], dtype=np.int64))
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
        tagging = _accepted_tagging_hysteresis(payload["program_accepted_state"])
        if not tagging:
            raise AssertionError("rank-change proof requires non-empty persistent tagging state")
        _assert_active_tagging_hysteresis(tagging)
    return owners, tagging


def _assert_single_rank_checkpoint(checkpoint: Path, *, expected_tagging_hysteresis: bytes) -> None:
    with np.load(checkpoint, allow_pickle=False) as payload:
        if int(payload["n_ranks"]) != 1 or int(payload["n_levels"]) != 2:
            raise AssertionError(
                "post-restart checkpoint must record one rank and the same two levels"
            )
        for level in range(2):
            owners = tuple(
                int(value)
                for value in np.asarray(
                    payload["dmap_%d" % level],
                    dtype=np.int64,
                )
            )
            if not owners or set(owners) != {0}:
                raise AssertionError(
                    "level-%d ownership was not rematerialized entirely onto rank 0: %r"
                    % (level, owners)
                )
        actual_tagging = _accepted_tagging_hysteresis(payload["program_accepted_state"])
        if actual_tagging != expected_tagging_hysteresis:
            raise AssertionError(
                "persistent tagging state changed during two-to-one rematerialization"
            )


def _capture(checkpoint: Path, evidence: Path | None, *, bit_identical: bool) -> None:
    if int(_COMM.size) != 2:
        require_mpi_or_skip(
            "AMR rank-change capture requires exactly two MPI ranks (observed %d)" % int(_COMM.size)
        )
    runtime = _runtime(bit_identical=bit_identical)
    initial_state = np.asarray(
        runtime.block_level_state_global("tracer", 0), dtype=np.float64
    ).copy()
    initial_mass = float(runtime.integral("tracer", levels=(0,)))
    _advance(runtime, CHECKPOINT_STEPS)
    checkpoint_metadata, checkpoint_arrays = _capture_arrays(runtime, prefix="checkpoint")
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
    if int(_COMM.rank) == 0:
        try:
            source_owners, tagging_hysteresis = _checkpoint_source_authorities(published)
            authority_row = {
                "ok": True,
                "owners": list(source_owners),
                "tagging_hex": tagging_hysteresis.hex(),
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001 -- publish the root refusal to every rank
            authority_row = {
                "ok": False,
                "owners": [],
                "tagging_hex": "",
                "error": "%s: %s" % (type(exc).__name__, exc),
            }
    else:
        authority_row = {"ok": None, "owners": [], "tagging_hex": "", "error": ""}
    authority_rows = allgather_value(_COMM, authority_row)
    root_authority = authority_rows[0]
    if root_authority.get("ok") is not True:
        raise RuntimeError(
            "rank-change checkpoint authority inspection failed collectively: %s"
            % root_authority.get("error", "missing rank-0 status")
        )
    source_owners = tuple(int(owner) for owner in root_authority["owners"])
    tagging_hysteresis = bytes.fromhex(root_authority["tagging_hex"])

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
        evidence_error = ""
        if int(_COMM.rank) == 0:
            try:
                _write_evidence(
                    evidence,
                    checkpoint_metadata=checkpoint_metadata,
                    checkpoint_arrays=checkpoint_arrays,
                    final_metadata=final_metadata,
                    final_arrays=final_arrays,
                    source_owners=source_owners,
                    tagging_hysteresis=tagging_hysteresis,
                    initial_mass=initial_mass,
                )
            except Exception as exc:  # noqa: BLE001 -- propagate root-only I/O failure
                evidence_error = "%s: %s" % (type(exc).__name__, exc)
        evidence_errors = allgather_value(_COMM, evidence_error)
        root_evidence_error = str(evidence_errors[0])
        if root_evidence_error:
            raise RuntimeError(
                "rank-change evidence publication failed collectively: %s" % root_evidence_error
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
            "AMR rank-change restart requires exactly one MPI rank (observed %d)" % int(_COMM.size)
        )
    metadata, arrays = _load_evidence(evidence)
    expected_tagging_hysteresis = (
        np.asarray(arrays["checkpoint_tagging_hysteresis"], dtype=np.uint8).reshape(-1).tobytes()
    )
    if (
        hashlib.sha256(expected_tagging_hysteresis).hexdigest()
        != metadata["tagging_hysteresis_sha256"]
    ):
        raise AssertionError("rank-change evidence persistent-tagging digest is corrupt")
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
    _assert_single_rank_checkpoint(
        post_restart,
        expected_tagging_hysteresis=expected_tagging_hysteresis,
    )
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
    state_after = np.asarray(runtime.block_level_state_global("tracer", 0), dtype=np.float64)
    boxes_after = tuple(tuple(int(value) for value in row) for row in runtime.patch_boxes())
    if (
        runtime.macro_step() != 0
        or runtime.time() != 0.0
        or boxes_after != boxes_before
        or not np.array_equal(state_after, state_before)
    ):
        raise AssertionError("strict rank-topology refusal mutated the fresh runtime")
    print("PASS bit_identical=True refuses AMR two-to-one restart atomically", flush=True)


def _capture_divergent(checkpoint: Path) -> None:
    """Prove source-rank tagging disagreement aborts collectively before publication."""
    if int(_COMM.size) != 2:
        require_mpi_or_skip(
            "AMR divergent capture requires exactly two MPI ranks (observed %d)" % int(_COMM.size)
        )
    runtime = _runtime(bit_identical=False)
    _advance(runtime, CHECKPOINT_STEPS)
    executor = getattr(runtime, "_executor", None)
    native = getattr(executor, "_s", None)
    if native is None:
        raise AssertionError("rank-change probe cannot reach its bound native AMR engine")
    prepare_error = ""
    try:
        original = bytes(native.program_accepted_state())
        tagging, tagging_offset = _accepted_tagging_hysteresis_span(original)
        _assert_active_tagging_hysteresis(tagging)
        if int(_COMM.rank) == 1:
            divergent = bytearray(original)
            divergent[tagging_offset + len(tagging) - 1] ^= 1
            native.restore_program_accepted_state(bytes(divergent))
    except Exception as exc:  # noqa: BLE001 -- coordinate local preparation failures
        prepare_error = "%s: %s" % (type(exc).__name__, exc)
    prepare_rows = allgather_value(
        _COMM,
        {"rank": int(_COMM.rank), "error": prepare_error},
    )
    prepare_failures = tuple(row for row in prepare_rows if str(row.get("error", "")))
    if prepare_failures:
        raise RuntimeError(
            "divergent accepted-state preparation failed collectively: %r" % (prepare_failures,)
        )

    caught = False
    message = ""
    try:
        runtime.checkpoint(checkpoint)
    except Exception as exc:  # noqa: BLE001 -- exact collective refusal is asserted below
        caught = True
        message = str(exc)
    rows = allgather_value(
        _COMM,
        {"caught": caught, "message": message},
    )
    if len(rows) != 2 or any(
        row.get("caught") is not True
        or "collective checkpoint AMR accepted-state capture sealed payload failed"
        not in row.get("message", "")
        or "tagging hysteresis" not in row.get("message", "")
        for row in rows
    ):
        raise AssertionError(
            "persistent-tagging producer disagreement did not fail identically "
            "on every rank: %r" % (rows,)
        )
    residue = tuple(sorted(path.name for path in checkpoint.parent.iterdir()))
    if residue:
        raise AssertionError(
            "collectively rejected checkpoint left partial publication: %r" % (residue,)
        )
    print(
        "PASS divergent source tagging payload refuses collective publication atomically",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "capture-relaxed",
            "capture-strict",
            "capture-divergent",
            "restart-relaxed",
            "restart-strict",
        ),
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
    elif args.mode == "capture-divergent":
        _capture_divergent(args.checkpoint)
    elif args.mode == "restart-relaxed":
        if args.evidence is None or args.rematerialized_checkpoint is None:
            raise ValueError("restart-relaxed requires --evidence and --rematerialized-checkpoint")
        _restart_relaxed(
            args.checkpoint,
            args.evidence,
            args.rematerialized_checkpoint,
        )
    else:
        _restart_strict(args.checkpoint)


if __name__ == "__main__":
    main()
