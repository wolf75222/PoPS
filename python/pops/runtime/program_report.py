"""Structured compiled-Program runtime report (ADC-594).

``System.program_report()`` and ``AmrSystem.program_report()`` return a :class:`ProgramRuntimeReport`
value object built from this module. It aggregates the ALREADY-bound C++ accessors of the extracted
Program subsystem (``pops::runtime::program::ProgramRuntimeState``) into ONE inspectable, JSON-ready
structure: the installed step / hash, typed step transaction, name-based block map, the per-block
runtime-param counts, the recorded diagnostics, the multistep histories, the scheduler cache slots and
the profiler state.

Deliberately metadata-only (the ADC-591 inspection house rule): it reads owned facts, never field
arrays, never recompiles, never installs a program. It is the SINGLE SOURCE the
:class:`~pops.runtime.inspection.RuntimeInspectionReport` ``program`` section is now built from, so the
two reports never drift.

The native report remains metadata-only; transaction identity is read from the authenticated Python
Program contract installed alongside the native closure.
"""

from __future__ import annotations

import json

from typing import Any


# An aggregate report/clock read is optimistic: a child publication is allowed to race the
# metadata reads, but it must be detected and retried.  Keeping the bound here makes the
# fail-closed behaviour explicit and testable without introducing a lock or thread-local state.
_MAX_OPTIMISTIC_SNAPSHOT_ATTEMPTS = 3


def _generation_vector(children: Any, *, where: str) -> tuple[int, ...]:
    """Read the authenticated accepted-transaction generation of every child in order."""
    vector = []
    for index, child in enumerate(tuple(children)):
        witness = getattr(child, "_accepted_transaction_generation_", None)
        if not callable(witness):
            raise RuntimeError(
                "%s child %d lacks _accepted_transaction_generation_()" % (where, index)
            )
        try:
            generation = witness()
        except Exception as error:
            raise RuntimeError(
                "%s child %d generation witness failed" % (where, index)
            ) from error
        if type(generation) is not int or generation < 0:
            raise RuntimeError(
                "%s child %d generation witness must return a non-negative integer" % (
                    where,
                    index,
                )
            )
        vector.append(generation)
    if not vector:
        raise RuntimeError("%s requires at least one child generation witness" % where)
    return tuple(vector)


def _optimistic_multi_layout_read(children: Any, compose: Any, *, where: str) -> Any:
    """Compose one all-child value only from a stable monotone generation vector.

    ``compose`` is deliberately rerun from scratch after an interleaved publication.  Generation
    values are authenticated by the child engines and are monotone, so an equal before/after
    vector cannot hide an ABA publication.  A missing/invalid witness and generation regression
    fail closed instead of returning a mixed result.
    """
    ordered_children = tuple(children)
    previous_after = None
    for _attempt in range(_MAX_OPTIMISTIC_SNAPSHOT_ATTEMPTS):
        before = _generation_vector(ordered_children, where=where)
        if previous_after is not None and any(
            current < previous
            for current, previous in zip(before, previous_after, strict=True)
        ):
            raise RuntimeError("%s generation vector regressed" % where)
        compose_error = None
        try:
            result = compose()
        except BaseException as error:
            # A mixed child image can fail an otherwise valid aggregate invariant (for example,
            # equal clocks or one common temporal partition).  Authenticate the generation vector
            # before deciding whether that exception belongs to a stable accepted image.  Only a
            # witnessed publication authorizes a retry; a stable-vector exception is the real
            # result and must propagate unchanged.
            compose_error = error
            result = None
        after = _generation_vector(ordered_children, where=where)
        if any(
            current < previous for current, previous in zip(after, before, strict=True)
        ):
            raise RuntimeError("%s generation vector regressed during read" % where)
        if before == after:
            if compose_error is not None:
                raise compose_error
            return result
        previous_after = after
    raise RuntimeError(
        "%s changed during optimistic snapshot after %d attempts"
        % (where, _MAX_OPTIMISTIC_SNAPSHOT_ATTEMPTS)
    )


def _call(obj: Any, name: Any, default: Any = None, *args: Any) -> Any:
    """Call ``obj.name(*args)`` if present + callable, else return @p default (never raises)."""
    fn = getattr(obj, name, None)
    if not callable(fn):
        return default
    try:
        return fn(*args)
    except Exception:
        return default


class ProgramRuntimeReport:
    """Structured, printable snapshot of the compiled-Program runtime subsystem (ADC-594).

    Inert, JSON-ready (``to_dict`` / ``to_json`` array-free), and stable: it holds plain scalars,
    dicts and lists of dicts, no field arrays. ``installed`` is False on a fresh runtime (empty
    sections); a bound program fills the sections from the C++ Program subsystem accessors.
    """

    schema_version = 5
    report_type = "program_runtime"

    def __init__(
        self,
        *,
        installed: Any,
        program_hash: Any,
        step_transaction: Any,
        block_map: Any,
        params: Any,
        diagnostics: Any,
        histories: Any,
        cache: Any,
        profiler: Any,
        clocks: Any,
        level_relations: Any,
        flux_ledger: Any,
        synchronization: Any,
        temporal_partition: Any,
        temporal: Any,
    ) -> None:
        self.installed = bool(installed)
        self.program_hash = program_hash or ""
        self.step_transaction = dict(step_transaction)
        self.block_map = list(block_map)
        self.params = [dict(row) for row in params]
        self.diagnostics = dict(diagnostics)
        self.histories = [dict(row) for row in histories]
        self.cache = [dict(row) for row in cache]
        self.profiler = dict(profiler)
        self.clocks = [dict(row) for row in clocks]
        self.level_relations = [dict(row) for row in level_relations]
        self.flux_ledger = [dict(row) for row in flux_ledger]
        self.synchronization = [dict(row) for row in synchronization]
        self.temporal_partition = dict(temporal_partition)
        self.temporal = dict(temporal)

    def to_dict(self) -> Any:
        return {
            "schema_version": self.schema_version,
            "report_type": self.report_type,
            "installed": self.installed,
            "program_hash": self.program_hash,
            "step_transaction": dict(self.step_transaction),
            "block_map": list(self.block_map),
            "params": [dict(row) for row in self.params],
            "diagnostics": dict(self.diagnostics),
            "histories": [dict(row) for row in self.histories],
            "cache": [dict(row) for row in self.cache],
            "profiler": dict(self.profiler),
            "clocks": [dict(row) for row in self.clocks],
            "level_relations": [dict(row) for row in self.level_relations],
            "flux_ledger": [dict(row) for row in self.flux_ledger],
            "synchronization": [dict(row) for row in self.synchronization],
            "temporal_partition": dict(self.temporal_partition),
            "temporal": dict(self.temporal),
        }

    def to_json(self, path: Any = None, *, indent: int = 2) -> Any:
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def __repr__(self) -> Any:
        return "ProgramRuntimeReport(installed=%r, hash=%r, histories=%d, cache=%d)" % (
            self.installed,
            self.program_hash or "(none)",
            len(self.histories),
            len(self.cache),
        )

    def __str__(self) -> Any:
        strategy = self.step_transaction.get("strategy", {})
        lines = ["program runtime report (schema=%d)" % self.schema_version]
        lines.append("  installed   : %s" % self.installed)
        lines.append("  hash        : %s" % (self.program_hash or "(none)"))
        lines.append("  strategy    : %s" % (strategy.get("kind") or "(none)"))
        lines.append("  block_map   : %s" % (self.block_map or "(identity)"))
        lines.append("  params      : %d block(s)" % len(self.params))
        lines.append("  diagnostics : %d scalar(s)" % len(self.diagnostics))
        lines.append("  histories   : %d ring(s)" % len(self.histories))
        lines.append("  cache       : %d slot(s)" % len(self.cache))
        lines.append("  profiler    : enabled=%s" % self.profiler.get("enabled"))
        lines.append("  clocks      : %d cursor(s)" % len(self.clocks))
        lines.append("  flux ledger : %d accepted contribution(s)" % len(self.flux_ledger))
        lines.append("  sync        : %d phase event(s)" % len(self.synchronization))
        lines.append("  partition   : %s" % (self.temporal_partition.get("kind") or "(none)"))
        return "\n".join(lines)


def _params(sim: Any) -> Any:
    """Per-program-block runtime-param COUNT + the kMaxRuntimeParams limit (never the values -- inert
    metadata). Derived from the block map (or, absent it, block 0): a block with no runtime param reports
    count 0. The limit (ADC-610) surfaces the previously-hidden fixed-array capacity so a block's headroom
    is introspectable."""
    from pops.physics.aux import max_runtime_params  # lazy: keep the report import-light

    limit = max_runtime_params()
    rows = []
    block_map = list(_call(sim, "program_block_map", []) or [])
    prog_blocks = list(range(len(block_map))) if block_map else [0]
    for prog_block in prog_blocks:
        count = _call(sim, "program_param_count", None, prog_block)
        if count is None:
            # Compatibility for report-only authorities used by downstream integrations.  Native
            # System and AmrSystem expose program_param_count directly, without publishing values.
            rp = _call(sim, "program_params", None, prog_block)
            count = getattr(rp, "count", None) if rp is not None else None
        rows.append({"program_block": prog_block, "count": count, "limit": limit})
    return rows


def _histories(sim: Any) -> Any:
    rows = []
    for name in _call(sim, "history_names", []) or []:
        depth = _call(sim, "history_depth", None, name)
        row = {
            "name": name,
            "depth": depth,
            "ncomp": _call(sim, "history_ncomp", None, name),
        }
        levels = _call(sim, "history_levels", None, name)
        if levels is None:
            row["initialized"] = _call(sim, "history_initialized", None, name)
            row["fill_count"] = _call(sim, "history_fill_count", None, name)
            row["slot_dt"] = [
                _call(sim, "history_slot_dt", None, name, slot)
                for slot in range(int(depth or 0))
            ]
        else:
            row["levels"] = [
                {
                    "level": int(level),
                    "initialized": _call(sim, "history_initialized", None, name, int(level)),
                    "fill_count": _call(sim, "history_fill_count", None, name, int(level)),
                    "slot_dt": [
                        _call(sim, "history_slot_dt", None, name, int(level), slot)
                        for slot in range(int(depth or 0))
                    ],
                }
                for level in levels
            ]
        rows.append(row)
    return rows


def _cache(sim: Any) -> Any:
    rows = []
    # The native cache is a bind-sealed dense table.  Reports enumerate every
    # slot, including invalid/cold slots whose accumulated window must survive
    # a restart; no graph node id or path lookup belongs in this metadata path.
    for slot in _call(sim, "program_cache_slots", []) or []:
        slot = int(slot)
        rows.append(
            {
                "slot": slot,
                "valid": bool(_call(sim, "program_cache_valid", False, slot)),
                "cold": bool(_call(sim, "program_cache_cold", False, slot)),
                "name": _call(sim, "program_cache_name", "", slot),
                "last_update_step": _call(sim, "program_cache_last_update_step", None, slot),
                "accumulated_dt": _call(sim, "program_cache_accumulated_dt", None, slot),
            }
        )
    return rows


def _amr_temporal_report(sim: Any) -> tuple[Any, Any, Any, Any, Any]:
    clocks = []
    for row in _call(sim, "program_clock_manifest", []) or []:
        if row[0] == "level" and len(row) == 6:
            clocks.append(
                {
                    "kind": "level",
                    "level": int(row[1]),
                    "macro_step": int(row[2]),
                    "phase": {"numerator": int(row[3]), "denominator": int(row[4])},
                    "physical_time": float(row[5]),
                }
            )
        elif row[0] == "logical" and len(row) == 3:
            clocks.append({"kind": "logical", "clock": row[1], "tick": int(row[2])})
        else:
            raise ValueError("native AMR Program clock report has an invalid row")
    relations = []
    for row in _call(sim, "checkpoint_temporal_relations", []) or []:
        if len(row) != 5:
            raise ValueError("native AMR temporal relation report has an invalid row")
        relations.append(
            {
                "parent_level": int(row[0]),
                "child_level": int(row[1]),
                "temporal_ratio": {"numerator": int(row[2]), "denominator": int(row[3])},
                "remainder_policy": row[4],
            }
        )
    ledger = []
    for row in _call(sim, "program_flux_ledger_manifest", []) or []:
        if len(row) != 13:
            raise ValueError("native AMR Program flux-ledger report has an invalid row")
        ledger.append(
            {
                "owner": row[0],
                "state": row[1],
                "rate": row[2],
                "flux": row[3],
                "level": int(row[4]),
                "macro_step": int(row[5]),
                "phase": {"numerator": int(row[6]), "denominator": int(row[7])},
                "stage_weight": {"numerator": int(row[8]), "denominator": int(row[9])},
                "orientation": row[10],
                "face_measure": float(row[11]),
                "substep_duration": float(row[12]),
            }
        )
    synchronization = []
    for row in _call(sim, "program_sync_manifest", []) or []:
        if len(row) != 7:
            raise ValueError("native AMR Program synchronization report has an invalid row")
        synchronization.append(
            {
                "parent_level": int(row[0]),
                "child_level": int(row[1]),
                "block": int(row[2]),
                "phase": row[3],
                "macro_step": int(row[4]),
                "clock_phase": {"numerator": int(row[5]), "denominator": int(row[6])},
            }
        )
    temporal_partition = {}
    for row in _call(sim, "program_temporal_partition_manifest", []) or []:
        if row[0] == "summary" and len(row) == 7:
            if temporal_partition:
                raise ValueError("native temporal-partition report has duplicate summary rows")
            temporal_partition = {
                "kind": row[1],
                "provider_identity": row[2],
                "topology_epoch": int(row[3]),
                "synchronization_tick": int(row[4]),
                "tick_denominator": int(row[5]),
                "cell_count": int(row[6]),
                "rungs": [],
            }
        elif row[0] == "rung" and len(row) == 3 and temporal_partition:
            temporal_partition["rungs"].append({"rung": int(row[1]), "cells": int(row[2])})
        else:
            raise ValueError("native temporal-partition report has an invalid row")
    return clocks, relations, ledger, synchronization, temporal_partition


def build_program_report(sim: Any) -> Any:
    """Aggregate the bound Program-subsystem accessors of @p sim into a :class:`ProgramRuntimeReport`.

    @p sim is the engine (or a delegating view that forwards to it). Every field is read gracefully
    (never raises): a fresh runtime yields ``installed=False`` and empty sections; an older ``.so``
    missing an accessor yields ``None`` for that field.
    """
    program_hash = _call(sim, "installed_program_hash", "") or ""
    clocks, relations, ledger, synchronization, temporal_partition = _amr_temporal_report(sim)
    temporal_state = getattr(sim, "_temporal_restart_state", None)
    temporal = temporal_state.to_data() if temporal_state is not None else {}
    return ProgramRuntimeReport(
        installed=bool(program_hash),
        program_hash=program_hash,
        step_transaction=(
            sim._step_transaction_plan.to_data()
            if getattr(sim, "_step_transaction_plan", None) is not None
            else {}
        ),
        block_map=list(_call(sim, "program_block_map", []) or []),
        params=_params(sim),
        diagnostics=dict(_call(sim, "program_diagnostics", {}) or {}),
        histories=_histories(sim),
        cache=_cache(sim),
        profiler={"enabled": _call(sim, "is_profiling", None)},
        clocks=clocks,
        level_relations=relations,
        flux_ledger=ledger,
        synchronization=synchronization,
        temporal_partition=temporal_partition,
        temporal=temporal,
    )
