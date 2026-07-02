"""Structured runtime inspection reports (ADC-591).

``System.inspect()`` and ``AmrSystem.inspect()`` return a value object from this module. The
report is deliberately metadata-only: it reads already-owned C++ facts and runtime registries, never
field arrays, never recompiles, never installs a program.
"""

import json

from pops._capabilities import native_capability_report
from pops.runtime.defaults import numerical_defaults_report
from pops.runtime.fallbacks import fallback_diagnostics_report
from pops.runtime.profile import PerformanceSummary
from pops.runtime_environment import runtime_environment_report


class RuntimeInspectionReport:
    """Structured, printable snapshot of a live runtime facade."""

    schema_version = 1
    report_type = "runtime_inspection"

    def __init__(self, *, runtime, blocks, clock, runtime_environment, capabilities, program,
                 profile, history, cache, diagnostics, options=None, amr=None, limitations=None,
                 routes=None, lifecycle=None, bound_snapshot=None):
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
        self.options = dict(options) if options is not None else {}
        self.amr = dict(amr) if amr is not None else None
        self.limitations = list(limitations or [])
        self.routes = dict(routes) if routes is not None else {}
        # RUNTIME FREEZE LIFECYCLE (ADC-592): the lifecycle state ("assembling"/"bound"/"running") and,
        # once bound, the BoundSnapshot manifest of WHAT was bound (as a plain dict + its stable hash).
        # An engine never bound reports "assembling" and no snapshot (bound_snapshot is None).
        self.lifecycle = lifecycle if lifecycle is not None else "assembling"
        self.bound_snapshot = dict(bound_snapshot) if bound_snapshot is not None else None

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
            "options": dict(self.options),
            "amr": dict(self.amr) if self.amr is not None else None,
            "limitations": [dict(row) for row in self.limitations],
            "routes": dict(self.routes),
            "lifecycle": self.lifecycle,
            "bound_snapshot": dict(self.bound_snapshot) if self.bound_snapshot is not None else None,
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
        # RUNTIME FREEZE LIFECYCLE (ADC-592): the state, and once bound the snapshot hash + a one-line
        # block / solver summary of what was frozen.
        lines.append("  lifecycle   : %s" % self.lifecycle)
        snap = self.bound_snapshot
        if snap is not None:
            snap_blocks = ", ".join(str(b.get("name")) for b in snap.get("blocks", [])) or "(none)"
            snap_solvers = ", ".join(sorted(snap.get("solvers", {}))) or "(none)"
            lines.append("  bound       : snapshot=%s blocks=[%s] solvers=[%s]"
                         % (snap.get("snapshot_hash", "(none)"), snap_blocks, snap_solvers))
        lines.append("  profile     : source=%s scopes=%d counters=%d"
                     % (prof.get("source"), len(prof.get("scopes", {})),
                        len(prof.get("counters", {}))))
        lines.append("  history     : %d ring(s)" % len(self.history))
        lines.append("  cache       : %d slot(s)" % len(self.cache))
        fallbacks = self.diagnostics.get("fallbacks", {})
        lines.append("  diagnostics : %d scalar(s), fallbacks=%s"
                     % (len(self.diagnostics), fallbacks.get("total_count", 0)))
        opts = self.options
        lines.append("  options     : blocks=%d source_stages=%d"
                     % (len(opts.get("blocks", [])), len(opts.get("source_stages", []))))
        if self.routes:
            lines.append("  routes      : %d block(s), poisson=%s"
                         % (len(self.routes.get("blocks", [])),
                            (self.routes.get("poisson") or {}).get("solver", {}).get("id",
                                                                                     "(none)")))
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
    options = _options(sim, runtime)
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
        diagnostics=_diagnostics(sim, options),
        options=options,
        amr=_amr(sim) if runtime == "amr_system" else None,
        limitations=limitations,
        routes=_routes(options),
        lifecycle=_lifecycle(sim),
        bound_snapshot=_bound_snapshot(sim))


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
    """The compiled-Program section, built FROM the structured ProgramRuntimeReport (ADC-594) so the
    two reports share a SINGLE source. Kept back-compatible: the historical inspection keys
    ("installed"/"hash") are preserved, with the richer cadence/block_map/param/history/cache summary
    folded in from the same report."""
    from pops.runtime.program_report import build_program_report
    report = build_program_report(sim)
    return {
        "installed": report.installed,
        "hash": report.program_hash,
        "cadence": dict(report.cadence),
        "block_map": list(report.block_map),
        "params": [dict(row) for row in report.params],
        "histories": [dict(row) for row in report.histories],
        "cache": [dict(row) for row in report.cache],
        "profiler": dict(report.profiler),
    }


def _lifecycle(sim):
    """The runtime lifecycle state (ADC-592): "assembling" for an engine never bound, else the
    engine's own ``lifecycle_state()`` ("bound"/"running"). Graceful default keeps a pre-bind or
    low-level engine describable rather than raising."""
    state = _call(sim, "lifecycle_state", None)
    return str(state) if state is not None else "assembling"


def _bound_snapshot(sim):
    """The BoundSnapshot manifest of what pops.bind froze, as a plain dict + its hash (ADC-592).

    Reads the engine's ``bound_snapshot`` (None before bind); serialises it via ``to_dict()`` and
    folds in the stable ``snapshot_hash`` so inspection carries the frozen identity. Returns None when
    the engine was never bound (an engine driven by the low-level seam without pops.bind)."""
    snap = getattr(sim, "bound_snapshot", None)
    if snap is None:
        return None
    to_dict = getattr(snap, "to_dict", None)
    payload = dict(to_dict()) if callable(to_dict) else {}
    snapshot_hash = getattr(snap, "snapshot_hash", None)
    if snapshot_hash is not None:
        payload["snapshot_hash"] = snapshot_hash
    return payload


def _lifecycle(sim):
    """The runtime lifecycle state (ADC-592): "assembling" for an engine never bound, else the
    engine's own ``lifecycle_state()`` ("bound"/"running"). Graceful default keeps a pre-bind or
    low-level engine describable rather than raising."""
    state = _call(sim, "lifecycle_state", None)
    return str(state) if state is not None else "assembling"


def _bound_snapshot(sim):
    """The BoundSnapshot manifest of what pops.bind froze, as a plain dict + its hash (ADC-592).

    Reads the engine's ``bound_snapshot`` (None before bind); serialises it via ``to_dict()`` and
    folds in the stable ``snapshot_hash`` so inspection carries the frozen identity. Returns None when
    the engine was never bound (an engine driven by the low-level seam without pops.bind)."""
    snap = getattr(sim, "bound_snapshot", None)
    if snap is None:
        return None
    to_dict = getattr(snap, "to_dict", None)
    payload = dict(to_dict()) if callable(to_dict) else {}
    snapshot_hash = getattr(snap, "snapshot_hash", None)
    if snapshot_hash is not None:
        payload["snapshot_hash"] = snapshot_hash
    return payload


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


def _diagnostics(sim, options):
    diagnostics = dict(_call(sim, "program_diagnostics", {}) or {})
    diagnostics["solver_events"] = list(_call(sim, "solver_diagnostics", []) or [])
    diagnostics["fallbacks"] = fallback_diagnostics_report(options)
    return diagnostics


def _options(sim, runtime):
    report = _call(sim, "effective_options_report", None)
    if report:
        try:
            return dict(report)
        except Exception:
            pass
    return {
        "schema_version": 1,
        "runtime": runtime,
        "defaults": numerical_defaults_report(),
        "blocks": [],
        "poisson": {},
        "source_stages": [],
        "time": {"scheme": None, "gauss_policy": None},
        "amr": None,
    }


def _try_route(family, token):
    """Route manifest of @p token in @p family, or a minimal unregistered row (ADC-584).

    The effective options carry the wire tokens; MOST map to a typed native route. A token
    outside the registry is NOT an error here: a compiled DSL block reports its generated
    transport (not a builtin brick), and inspection must describe it rather than refuse it.
    """
    if not token:
        return None
    from pops.runtime.routes import resolve
    try:
        return resolve(family, str(token)).manifest()
    except ValueError:
        return {"family": family, "id": None, "token": str(token),
                "native_entry": "unregistered (compiled/DSL or external route)",
                "requirements": [], "limitations": []}


def _routes(options):
    """The typed native routes USED by the live runtime (ADC-584 inspection).

    Derived from the effective options report (which already carries the per-block wire
    tokens): each block's scheme/time/model tokens and the Poisson rhs/solver/bc/wall are
    mapped to their route manifests (family, id, native entry point, requirements,
    limitations).
    """
    blocks = []
    for blk in options.get("blocks", []) or []:
        row = {"name": blk.get("name")}
        for family, key in (("limiter", "limiter"), ("riemann", "riemann"), ("recon", "recon"),
                            ("time", "time"), ("transport", "transport"), ("source", "source"),
                            ("elliptic", "elliptic")):
            manifest = _try_route(family, blk.get(key))
            if manifest is not None:
                row[family] = manifest
        blocks.append(row)
    poisson = {}
    pois = options.get("poisson", {}) or {}
    for family, key in (("poisson_rhs", "rhs"), ("field_solver", "solver"),
                        ("poisson_bc", "bc"), ("wall", "wall")):
        manifest = _try_route(family, pois.get(key))
        if manifest is not None:
            poisson[family if family != "field_solver" else "solver"] = manifest
    return {"blocks": blocks, "poisson": poisson}


def _amr(sim):
    try:
        return sim.amr.hierarchy_snapshot().to_dict()
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _amr_patch_count(amr):
    patch_table = amr.get("patch_table") or {}
    return patch_table.get("n_patches")


__all__ = ["RuntimeInspectionReport", "build_runtime_inspection"]
