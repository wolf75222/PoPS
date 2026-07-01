"""Structured runtime inspection reports (ADC-591).

``System.inspect()`` and ``AmrSystem.inspect()`` return a value object from this module. The
report is deliberately metadata-only: it reads already-owned C++ facts and runtime registries, never
field arrays, never recompiles, never installs a program.
"""

import json

from pops._capabilities import native_capability_report
from pops.runtime.profile import PerformanceSummary
from pops.runtime_environment import runtime_environment_report


class RuntimeInspectionReport:
    """Structured, printable snapshot of a live runtime facade."""

    schema_version = 1
    report_type = "runtime_inspection"

    def __init__(self, *, runtime, blocks, clock, runtime_environment, capabilities, program,
                 profile, history, cache, diagnostics, amr=None, limitations=None):
        self.runtime = runtime
        self.blocks = list(blocks)
        self.clock = dict(clock)
        self.runtime_environment = dict(runtime_environment)
        self.capabilities = dict(capabilities)
        self.program = dict(program)
        self.profile = dict(profile)
        self.history = list(history)
        self.cache = list(cache)
        self.diagnostics = dict(diagnostics)
        self.amr = dict(amr) if amr is not None else None
        self.limitations = list(limitations or [])

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "report_type": self.report_type,
            "runtime": self.runtime,
            "blocks": list(self.blocks),
            "clock": dict(self.clock),
            "runtime_environment": dict(self.runtime_environment),
            "capabilities": dict(self.capabilities),
            "program": dict(self.program),
            "profile": dict(self.profile),
            "history": [dict(row) for row in self.history],
            "cache": [dict(row) for row in self.cache],
            "diagnostics": dict(self.diagnostics),
            "amr": dict(self.amr) if self.amr is not None else None,
            "limitations": [dict(row) for row in self.limitations],
        }

    def to_json(self, path=None, *, indent=2):
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def __repr__(self):
        return ("RuntimeInspectionReport(runtime=%r, blocks=%d, history=%d, cache=%d)"
                % (self.runtime, len(self.blocks), len(self.history), len(self.cache)))

    def __str__(self):
        rt = self.runtime_environment
        prof = self.profile
        lines = ["%s report (schema=%d)" % (self.runtime, self.schema_version)]
        lines.append("  blocks      : %s" % (", ".join(self.blocks) or "(none)"))
        lines.append("  clock       : t=%s macro_step=%s"
                     % (self.clock.get("time"), self.clock.get("macro_step")))
        lines.append("  runtime     : dimension=%s ratio=%s precision=%s communicator=%s"
                     % (rt.get("dimension"), rt.get("amr_refinement_ratio"),
                        rt.get("precision"), rt.get("communicator")))
        lines.append("  program     : installed=%s hash=%s"
                     % (self.program.get("installed"), self.program.get("hash") or "(none)"))
        lines.append("  profile     : source=%s scopes=%d counters=%d"
                     % (prof.get("source"), len(prof.get("scopes", {})),
                        len(prof.get("counters", {}))))
        lines.append("  history     : %d ring(s)" % len(self.history))
        lines.append("  cache       : %d slot(s)" % len(self.cache))
        lines.append("  diagnostics : %d scalar(s)" % len(self.diagnostics))
        if self.amr is not None:
            lines.append("  amr         : levels=%s patches=%s"
                         % (self.amr.get("max_levels"), _amr_patch_count(self.amr)))
        if self.limitations:
            partial = sum(1 for row in self.limitations if row.get("status") == "partial")
            unavailable = sum(1 for row in self.limitations if row.get("status") == "unavailable")
            lines.append("  limitations : %d partial, %d unavailable route(s)"
                         % (partial, unavailable))
        return "\n".join(lines)


def build_runtime_inspection(sim, *, runtime):
    cap_report = native_capability_report()
    cap_dict = cap_report.to_dict()
    limitations = [
        {"feature": row.feature, "status": row.status, "reason": row.limitation}
        for row in cap_report.routes
        if row.status != "available"
    ]
    return RuntimeInspectionReport(
        runtime=runtime,
        blocks=_block_names(sim),
        clock=_clock(sim),
        runtime_environment=runtime_environment_report(),
        capabilities=cap_dict,
        program=_program(sim),
        profile=PerformanceSummary(_profile_payload(sim)).to_dict(),
        history=_history(sim),
        cache=_cache(sim),
        diagnostics=_diagnostics(sim),
        amr=_amr(sim) if runtime == "amr_system" else None,
        limitations=limitations)


def _call(obj, name, default=None, *args):
    fn = getattr(obj, name, None)
    if not callable(fn):
        return default
    try:
        return fn(*args)
    except Exception:
        return default


def _block_names(sim):
    return list(_call(sim, "block_names", []) or [])


def _clock(sim):
    return {
        "time": _call(sim, "time", None),
        "macro_step": _call(sim, "macro_step", None),
    }


def _profile_payload(sim):
    snapshot = getattr(sim, "profile_snapshot", None)
    if callable(snapshot):
        return snapshot()
    return _call(sim, "profile_report", "") or ""


def _program(sim):
    h = _call(sim, "installed_program_hash", "") or ""
    return {"installed": bool(h), "hash": h}


def _history(sim):
    rows = []
    for name in _call(sim, "history_names", []) or []:
        rows.append({
            "name": name,
            "depth": _call(sim, "history_depth", None, name),
            "ncomp": _call(sim, "history_ncomp", None, name),
            "initialized": _call(sim, "history_initialized", None, name),
        })
    return rows


def _cache(sim):
    rows = []
    for node_id in _call(sim, "program_cache_nodes", []) or []:
        rows.append({
            "node_id": int(node_id),
            "name": _call(sim, "program_cache_name", "", node_id),
            "last_update_step": _call(sim, "program_cache_last_update_step", None, node_id),
            "accumulated_dt": _call(sim, "program_cache_accumulated_dt", None, node_id),
            "ncomp": _call(sim, "program_cache_ncomp", None, node_id),
            "ngrow": _call(sim, "program_cache_ngrow", None, node_id),
        })
    return rows


def _diagnostics(sim):
    return dict(_call(sim, "program_diagnostics", {}) or {})


def _amr(sim):
    try:
        return sim.amr.hierarchy_snapshot().to_dict()
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _amr_patch_count(amr):
    patch_table = amr.get("patch_table") or {}
    return patch_table.get("n_patches")


__all__ = ["RuntimeInspectionReport", "build_runtime_inspection"]
