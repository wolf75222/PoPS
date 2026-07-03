"""Structured compiled-Program runtime report (ADC-594).

``System.program_report()`` and ``AmrSystem.program_report()`` return a :class:`ProgramRuntimeReport`
value object built from this module. It aggregates the ALREADY-bound C++ accessors of the extracted
Program subsystem (``pops::runtime::program::ProgramRuntimeState``) into ONE inspectable, JSON-ready
structure: the installed step / hash, the global cadence, the name-based block map, the per-block
runtime-param counts, the recorded diagnostics, the multistep histories, the scheduler cache slots and
the profiler state.

Deliberately metadata-only (the ADC-591 inspection house rule): it reads owned facts, never field
arrays, never recompiles, never installs a program. It is the SINGLE SOURCE the
:class:`~pops.runtime.inspection.RuntimeInspectionReport` ``program`` section is now built from, so the
two reports never drift.

The builder is graceful against an older prebuilt ``.so`` (the ADC-592 hasattr-gating pattern): an
engine that predates a given accessor -- notably the ADC-594 ``program_substeps`` / ``program_stride``
getters -- yields ``None`` for that field rather than raising, so the report stays describable on a
stale extension (CI proves the fresh accessors).
"""

from __future__ import annotations

import json

from typing import Any


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

    schema_version = 1
    report_type = "program_runtime"

    def __init__(self, *, installed: Any, program_hash: Any, cadence: Any, block_map: Any,
                 params: Any, diagnostics: Any, histories: Any, cache: Any,
                 profiler: Any) -> None:
        self.installed = bool(installed)
        self.program_hash = program_hash or ""
        self.cadence = dict(cadence)
        self.block_map = list(block_map)
        self.params = [dict(row) for row in params]
        self.diagnostics = dict(diagnostics)
        self.histories = [dict(row) for row in histories]
        self.cache = [dict(row) for row in cache]
        self.profiler = dict(profiler)

    def to_dict(self) -> Any:
        return {
            "schema_version": self.schema_version,
            "report_type": self.report_type,
            "installed": self.installed,
            "program_hash": self.program_hash,
            "cadence": dict(self.cadence),
            "block_map": list(self.block_map),
            "params": [dict(row) for row in self.params],
            "diagnostics": dict(self.diagnostics),
            "histories": [dict(row) for row in self.histories],
            "cache": [dict(row) for row in self.cache],
            "profiler": dict(self.profiler),
        }

    def to_json(self, path: Any = None, *, indent: int = 2) -> Any:
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def __repr__(self) -> Any:
        return ("ProgramRuntimeReport(installed=%r, hash=%r, histories=%d, cache=%d)"
                % (self.installed, self.program_hash or "(none)", len(self.histories),
                   len(self.cache)))

    def __str__(self) -> Any:
        cad = self.cadence
        lines = ["program runtime report (schema=%d)" % self.schema_version]
        lines.append("  installed   : %s" % self.installed)
        lines.append("  hash        : %s" % (self.program_hash or "(none)"))
        lines.append("  cadence     : substeps=%s stride=%s"
                     % (cad.get("substeps"), cad.get("stride")))
        lines.append("  block_map   : %s" % (self.block_map or "(identity)"))
        lines.append("  params      : %d block(s)" % len(self.params))
        lines.append("  diagnostics : %d scalar(s)" % len(self.diagnostics))
        lines.append("  histories   : %d ring(s)" % len(self.histories))
        lines.append("  cache       : %d slot(s)" % len(self.cache))
        lines.append("  profiler    : enabled=%s" % self.profiler.get("enabled"))
        return "\n".join(lines)


def _cadence(sim: Any) -> Any:
    """The GLOBAL macro-step cadence (ADC-594). ``program_substeps`` / ``program_stride`` are the
    ADC-594 getters; an older ``.so`` lacks them -> None (graceful, CI proves the fresh accessors)."""
    return {
        "substeps": _call(sim, "program_substeps", None),
        "stride": _call(sim, "program_stride", None),
    }


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
        rp = _call(sim, "program_params", None, prog_block)
        count = getattr(rp, "count", None) if rp is not None else None
        rows.append({"program_block": prog_block, "count": count, "limit": limit})
    return rows


def _histories(sim: Any) -> Any:
    rows = []
    for name in _call(sim, "history_names", []) or []:
        rows.append({
            "name": name,
            "depth": _call(sim, "history_depth", None, name),
            "ncomp": _call(sim, "history_ncomp", None, name),
            "initialized": _call(sim, "history_initialized", None, name),
        })
    return rows


def _cache(sim: Any) -> Any:
    rows = []
    for node_id in _call(sim, "program_cache_nodes", []) or []:
        rows.append({
            "node_id": int(node_id),
            "name": _call(sim, "program_cache_name", "", node_id),
            "last_update_step": _call(sim, "program_cache_last_update_step", None, node_id),
            "accumulated_dt": _call(sim, "program_cache_accumulated_dt", None, node_id),
        })
    return rows


def build_program_report(sim: Any) -> Any:
    """Aggregate the bound Program-subsystem accessors of @p sim into a :class:`ProgramRuntimeReport`.

    @p sim is the engine (or a delegating view that forwards to it). Every field is read gracefully
    (never raises): a fresh runtime yields ``installed=False`` and empty sections; an older ``.so``
    missing an accessor yields ``None`` for that field.
    """
    program_hash = _call(sim, "installed_program_hash", "") or ""
    return ProgramRuntimeReport(
        installed=bool(program_hash),
        program_hash=program_hash,
        cadence=_cadence(sim),
        block_map=list(_call(sim, "program_block_map", []) or []),
        params=_params(sim),
        diagnostics=dict(_call(sim, "program_diagnostics", {}) or {}),
        histories=_histories(sim),
        cache=_cache(sim),
        profiler={"enabled": _call(sim, "is_profiling", None)},
    )
