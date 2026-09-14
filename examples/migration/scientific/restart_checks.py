"""Exact AMR restart comparisons, shared measurement without physics."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import json
import numpy as np


@dataclass(frozen=True, slots=True)
class ScalarRuntimeSnapshot:
    """Restart-sensitive evidence retained without exposing native implementation objects."""

    time: float
    macro_step: int
    states: tuple[np.ndarray, ...]
    patch_boxes: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]
    regrid_count: int
    topology_epoch: int
    program_hash: str
    program_transaction_state: str
    consumer_graph_identity: str
    consumer_cursors: dict[str, Any]


def _program_transaction_state(simulation: Any) -> str:
    """Canonicalize every restart-sensitive Program registry without native objects."""
    report = simulation.program_report().to_dict()
    return json.dumps(
        {
            "cache": report["cache"],
            "clocks": report["clocks"],
            "diagnostics": report["diagnostics"],
            "flux_ledger": report["flux_ledger"],
            "histories": report["histories"],
            "level_relations": report["level_relations"],
            "synchronization": report["synchronization"],
            "temporal": report["temporal"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _snapshot(simulation: Any) -> ScalarRuntimeSnapshot:
    """Capture every state item required for strict AMR continuation parity."""
    blocks = tuple(simulation.block_names())
    if blocks != ("tracer",):
        raise RuntimeError("scalar acceptance expected exactly the qualified tracer block")
    level_count = int(simulation.n_levels())
    if level_count <= 0:
        raise RuntimeError("scalar acceptance installed no AMR hierarchy levels")
    regrid = simulation.amr.explain_regrid()
    return ScalarRuntimeSnapshot(
        time=float(simulation.time()),
        macro_step=int(simulation.macro_step()),
        states=tuple(
            (
                np.asarray(
                    simulation.block_level_state_global("tracer", level), dtype=np.float64
                ).copy()
                for level in range(level_count)
            )
        ),
        patch_boxes=tuple(
            (
                (
                    int(level),
                    tuple((int(value) for value in lower)),
                    tuple((int(value) for value in upper)),
                )
                for level, lower, upper in simulation.patch_boxes()
            )
        ),
        regrid_count=int(regrid.regrid_count),
        topology_epoch=int(regrid.topology_epoch),
        program_hash=str(simulation.installed_program_hash()),
        program_transaction_state=_program_transaction_state(simulation),
        consumer_graph_identity=simulation.consumer_graph.identity.token,
        consumer_cursors=simulation.consumer_cursors.to_data(),
    )


def _require_same_snapshot(
    left: ScalarRuntimeSnapshot, right: ScalarRuntimeSnapshot, *, where: str
) -> None:
    """Reject any hidden state, topology, identity, clock or schedule drift."""
    exact = {
        "time": (left.time, right.time),
        "macro_step": (left.macro_step, right.macro_step),
        "patch_boxes": (left.patch_boxes, right.patch_boxes),
        "regrid_count": (left.regrid_count, right.regrid_count),
        "topology_epoch": (left.topology_epoch, right.topology_epoch),
        "program_hash": (left.program_hash, right.program_hash),
        "program_transaction_state": (
            left.program_transaction_state,
            right.program_transaction_state,
        ),
        "consumer_graph_identity": (left.consumer_graph_identity, right.consumer_graph_identity),
        "consumer_cursors": (left.consumer_cursors, right.consumer_cursors),
    }
    for name, (expected, actual) in exact.items():
        if expected != actual:
            raise RuntimeError("%s changed %s" % (where, name))
    if len(left.states) != len(right.states) or any(
        (
            not np.array_equal(expected, actual)
            for expected, actual in zip(left.states, right.states, strict=True)
        )
    ):
        raise RuntimeError("%s changed the conservative tracer AMR hierarchy" % where)


def _require_refined_hierarchy(snapshot: ScalarRuntimeSnapshot, *, where: str) -> None:
    """Require the AMR acceptance target to execute at least one genuinely refined level."""
    expected_levels = tuple(range(1, len(snapshot.states)))
    actual_levels = tuple(sorted({row[0] for row in snapshot.patch_boxes}))
    if not expected_levels or actual_levels != expected_levels:
        raise RuntimeError(
            "%s did not execute the requested refined AMR hierarchy: expected=%r, actual=%r"
            % (where, expected_levels, actual_levels)
        )
    if snapshot.regrid_count <= 0 or snapshot.topology_epoch <= 0:
        raise RuntimeError(
            "%s exposes refined patches but no completed dynamic topology replacement: regrid_count=%d, topology_epoch=%d"
            % (where, snapshot.regrid_count, snapshot.topology_epoch)
        )


def _require_regrid_progress(
    before: ScalarRuntimeSnapshot, after: ScalarRuntimeSnapshot, *, where: str
) -> None:
    """Require continuation to cross a completed topology-changing regrid window."""
    if after.macro_step <= before.macro_step:
        raise RuntimeError(
            "%s did not advance the accepted macro-step (%d -> %d)"
            % (where, before.macro_step, after.macro_step)
        )
    if after.regrid_count <= before.regrid_count:
        raise RuntimeError(
            "%s did not complete a dynamic regrid (%d -> %d)"
            % (where, before.regrid_count, after.regrid_count)
        )
    if after.topology_epoch <= before.topology_epoch:
        raise RuntimeError(
            "%s did not replace the accepted topology (%d -> %d)"
            % (where, before.topology_epoch, after.topology_epoch)
        )
