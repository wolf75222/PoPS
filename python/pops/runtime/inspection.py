"""Structured runtime inspection reports (ADC-591).

``System.inspect()`` and ``AmrSystem.inspect()`` return a value object from this module. The
report is deliberately metadata-only: it reads already-owned C++ facts and runtime registries, never
field arrays, never recompiles, never installs a program.
"""
from __future__ import annotations

from typing import Any

from pops._report import Report
from pops._capabilities import native_capability_report
from pops.runtime.defaults import numerical_defaults_report
from pops.runtime.fallbacks import fallback_diagnostics_report
from pops.runtime._profile import PerformanceSummary
from pops.runtime_environment import runtime_environment_report


class RuntimeInspectionReport(Report):
    """Structured, printable snapshot of a live runtime facade using the internal report base."""

    schema_version = 1
    report_type = "runtime_inspection"

    def __init__(self, *, runtime: Any, blocks: Any, clock: Any, runtime_environment: Any,
                 capabilities: Any, program: Any, profile: Any, history: Any, cache: Any,
                 diagnostics: Any, options: Any = None, amr: Any = None, limitations: Any = None,
                 routes: Any = None, lifecycle: Any = None, bound_snapshot: Any = None,
                 instance: Any = None) -> None:
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
        self.instance = dict(instance) if instance is not None else None

    def to_dict(self) -> Any:
        payload = {
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
        if self.instance is not None:
            payload["instance"] = dict(self.instance)
        return payload

    def __repr__(self) -> Any:
        return ("RuntimeInspectionReport(runtime=%r, blocks=%d, history=%d, cache=%d)"
                % (self.runtime, len(self.blocks), len(self.history), len(self.cache)))

    def __str__(self) -> Any:
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
        # RUNTIME FREEZE LIFECYCLE: the state, canonical bind identity and frozen composition.
        # block / solver summary of what was frozen.
        lines.append("  lifecycle   : %s" % self.lifecycle)
        snap = self.bound_snapshot
        if snap is not None:
            snap_blocks = ", ".join(str(b.get("name")) for b in snap.get("blocks", [])) or "(none)"
            snap_solvers = ", ".join(sorted(snap.get("solvers", {}))) or "(none)"
            identity = snap.get("bind_identity", {})
            lines.append("  bound       : identity=%s blocks=[%s] solvers=[%s]"
                         % (identity.get("hexdigest", "(none)"), snap_blocks, snap_solvers))
        lines.append("  profile     : source=%s scopes=%d counters=%d"
                     % (prof.get("source"), len(prof.get("scopes", {})),
                        len(prof.get("counters", {}))))
        lines.append("  history     : %d ring(s)" % len(self.history))
        lines.append("  cache       : %d slot(s)" % len(self.cache))
        fallbacks = self.diagnostics.get("fallbacks", {})
        lines.append("  diagnostics : %d scalar(s), fallbacks=%s"
                     % (len(self.diagnostics), fallbacks.get("total_count", 0)))
        opts = self.options
        lines.append("  options     : blocks=%d" % len(opts.get("blocks", [])))
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
        if self.instance is not None:
            lines.append("  instance    : bind=%s consumers=%d attempts=%s"
                         % (self.instance.get("bind_identity", {}).get("digest", "(none)"),
                            len(self.instance.get("consumer_graph", {}).get("nodes", [])),
                            self.instance.get("attempt")))
        return "\n".join(lines)


def build_runtime_inspection(
    sim: Any,
    *,
    runtime: Any,
    adaptive: bool | None = None,
    instance: Any = None,
) -> Any:
    """Build the :class:`RuntimeInspectionReport` of a bound simulation (inert, no numerics).

    Reads the carried metadata of @p sim (blocks, clock, capabilities, program, profile,
    history, cache, diagnostics) plus the native capability report; @p runtime names the
    runtime kind ("System" / "AmrSystem"). It runs no step and touches no field data."""
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
        amr=_amr(sim) if (runtime == "amr_system" if adaptive is None else adaptive) else None,
        limitations=limitations,
        routes=_routes(options),
        lifecycle=_lifecycle(sim),
        bound_snapshot=_bound_snapshot(sim), instance=instance)


def _call(obj: Any, name: Any, default: Any = None, *args: Any) -> Any:
    fn = getattr(obj, name, None)
    if not callable(fn):
        return default
    try:
        return fn(*args)
    except Exception:
        return default


def _block_names(sim: Any) -> Any:
    return list(_call(sim, "block_names", []) or [])


def _clock(sim: Any) -> Any:
    return {
        "time": _call(sim, "time", None),
        "macro_step": _call(sim, "macro_step", None),
    }


def _profile_payload(sim: Any) -> Any:
    snapshot = getattr(sim, "profile_snapshot", None)
    if callable(snapshot):
        return snapshot()
    return _call(sim, "profile_report", "") or ""


def _program(sim: Any) -> Any:
    """The compiled-Program section, built FROM the structured ProgramRuntimeReport (ADC-594) so the
    two reports share a SINGLE source. Kept back-compatible: the historical inspection keys
    ("installed"/"hash") are preserved, with the richer transaction/block-map/parameter/history/cache
    summary
    folded in from the same report."""
    from pops.runtime.program_report import build_program_report
    report = build_program_report(sim)
    return {
        "installed": report.installed,
        "hash": report.program_hash,
        "step_transaction": dict(report.step_transaction),
        "block_map": list(report.block_map),
        "params": [dict(row) for row in report.params],
        "histories": [dict(row) for row in report.histories],
        "cache": [dict(row) for row in report.cache],
        "profiler": dict(report.profiler),
    }


def _lifecycle(sim: Any) -> Any:
    """The runtime lifecycle state (ADC-592): "assembling" for an engine never bound, else the
    engine's own ``lifecycle_state()`` ("bound"/"running"). Graceful default keeps a pre-bind or
    low-level engine describable rather than raising."""
    state = _call(sim, "lifecycle_state", None)
    return str(state) if state is not None else "assembling"


def _bound_snapshot(sim: Any) -> Any:
    """Canonical BindManifest of what ``pops.bind`` froze, as a detached plain dict."""
    snap = getattr(sim, "bound_snapshot", None)
    if snap is None:
        return None
    to_dict: Any = getattr(snap, "to_dict", None)
    raw: Any = to_dict() if callable(to_dict) else {}
    return dict(raw)


def _history(sim: Any) -> Any:
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
            "ncomp": _call(sim, "program_cache_ncomp", None, node_id),
            "ngrow": _call(sim, "program_cache_ngrow", None, node_id),
        })
    return rows


def _diagnostics(sim: Any, options: Any) -> Any:
    diagnostics: Any = dict(_call(sim, "program_diagnostics", {}) or {})
    diagnostics["solver_events"] = list(_call(sim, "solver_diagnostics", []) or [])
    diagnostics["fallbacks"] = fallback_diagnostics_report(options)
    return diagnostics


def _options(sim: Any, runtime: Any) -> Any:
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
        "amr": None,
    }


def _try_route(family: Any, token: Any) -> Any:
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


def _routes(options: Any) -> Any:
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


def _amr(sim: Any) -> Any:
    try:
        return sim.amr.hierarchy_snapshot().to_dict()
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _amr_patch_count(amr: Any) -> Any:
    patch_table = amr.get("patch_table") or {}
    return patch_table.get("n_patches")


__all__ = ["RuntimeInspectionReport", "build_runtime_inspection"]
