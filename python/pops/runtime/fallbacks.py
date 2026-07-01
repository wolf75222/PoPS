"""Structured fallback/degraded-route diagnostics."""


def _static_report():
    return {
        "schema_version": 1,
        "source": "pops.runtime.fallbacks.static_fallback",
        "entries": [
            {
                "key": "elliptic.fft.direct_dft",
                "route": "PoissonFFT::fft1d",
                "cause": "FFT extent is not a power of two",
                "policy": "allowed_with_counter",
                "default_action": "allow",
                "impact": "correct O(n^2) transform replaces the radix-2 FFT",
                "frequency": "per 1D transform",
                "count": 0,
                "explicit_opt_in": False,
                "performance_degraded": True,
                "semantics_changed": False,
            },
            {
                "key": "spatial.positivity.order1_face",
                "route": "Zhang-Shu positivity limiter",
                "cause": "reconstructed face density falls below positivity_floor",
                "policy": "explicit_opt_in",
                "default_action": "disabled_until_positivity_floor_positive",
                "impact": "offending face is replaced by the source-cell average",
                "frequency": "per offending reconstructed face",
                "count": 0,
                "explicit_opt_in": True,
                "performance_degraded": False,
                "semantics_changed": True,
            },
        ],
        "total_count": 0,
    }


def _native_report():
    try:
        from pops import _pops  # noqa: PLC0415

        fn = getattr(_pops, "fallback_diagnostics_report", None)
        if callable(fn):
            return dict(fn())
    except Exception:
        pass
    return _static_report()


def _configured_routes(options):
    configured = []
    for block in (options or {}).get("blocks", []) or []:
        name = block.get("name")
        if float(block.get("positivity_floor") or 0.0) > 0.0:
            configured.append({
                "key": "spatial.positivity.order1_face",
                "block": name,
                "policy": "explicit_opt_in",
                "configured_by": "positivity_floor",
                "value": float(block.get("positivity_floor")),
            })
        cons = list(block.get("conservative_vars") or [])
        route = block.get("route", "")
        if route in ("dynamic_loader", "aot_loader", "native_loader") and cons:
            legacy_names = all(str(v) == "u%d" % i for i, v in enumerate(cons))
            if legacy_names:
                configured.append({
                    "key": "runtime.native_loader.legacy_metadata",
                    "block": name,
                    "policy": "report_and_compat",
                    "configured_by": "legacy_or_missing_metadata",
                    "value": {"conservative_vars": cons},
                })
    return configured


def fallback_diagnostics_report(options=None):
    """Return structured fallback/degraded-route diagnostics and explicit policies."""
    report = _native_report()
    entries = [dict(row) for row in report.get("entries", [])]
    total = sum(int(row.get("count") or 0) for row in entries)
    out = dict(report)
    out["entries"] = entries
    out["total_count"] = total
    out["configured"] = _configured_routes(options or {})
    return out


def reset_fallback_diagnostics():
    """Reset process-local fallback/degraded-route counters when the native module supports it."""
    try:
        from pops import _pops  # noqa: PLC0415

        fn = getattr(_pops, "reset_fallback_diagnostics", None)
        if callable(fn):
            fn()
    except Exception:
        pass


__all__ = ["fallback_diagnostics_report", "reset_fallback_diagnostics"]
