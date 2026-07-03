"""pops.runtime.profile -- typed profiling surface (Spec 5 sec.12.5, criteria 41-44).

A typed replacement for the stringly ``sim.enable_profiling()`` / ``sim.profile_report()``
dance. Two pieces:

* :class:`Profile` -- a typed profiling *level* (``Profile.Basic()`` / ``Profile.Advanced()``),
  NOT ``profile="advanced"``. It is an inert descriptor: it carries no timers and computes
  nothing in Python; it only declares WHICH native counters a level wants surfaced.
* :class:`PerformanceSummary` -- a printable wrapper around the native profile report. It parses
  the report the C++ :class:`pops::runtime::program::Profiler` produces (a per-scope timing table
  plus integer counters) into a structured dict and exposes typed views:
  :meth:`~PerformanceSummary.by_program_node` / :meth:`~PerformanceSummary.by_native_brick` /
  :meth:`~PerformanceSummary.by_solver` / :meth:`~PerformanceSummary.by_elliptic` /
  :meth:`~PerformanceSummary.by_amr_mpi` / :meth:`~PerformanceSummary.by_memory`. When a measure is not
  available on the current build (the heavy per-brick / scheduler counters are Kokkos-gated and only
  move under a compiled ``.so`` step; the AMR / MPI phase scopes only exist under a distributed AMR
  run), the view DECLARES it unavailable honestly rather than fabricating a zero.

The off-by-default contract (criterion 44): profiling adds no heavy timers unless explicitly
enabled. The native ``enable_profiling`` already gates this -- a plain run leaves the profiler
disabled. :meth:`System.profile` (the context manager in :mod:`pops.runtime.system`) is the typed
front door: it enables on ``__enter__`` and disables on ``__exit__``, and exposes
``prof.summary()`` -> :class:`PerformanceSummary`.

This module is a pure typed/parsing wrapper: it imports neither ``_pops`` nor numpy. The native
extension is reached only through the :class:`System` instance the context manager is bound to.
"""
from __future__ import annotations

import os
from typing import Any


# Native scope-name conventions the C++ Profiler emits (program_context.hpp / system.cpp):
# coarse System phases, per-Program-node scopes ("node:<name>"), and the integer counters.
_COARSE_PHASES = ("step", "field_solve")
_NODE_PREFIX = "node:"
# A field-solve Program node ("node:solve_fields...") is the solver-attributable scope on the
# native path; the coarse "field_solve" phase is the System-level elliptic solve.
_SOLVER_SCOPES = ("field_solve",)
_SOLVER_NODE_HINT = "solve_fields"
# Memory counters (program_context.hpp count_scratch): allocation count + the largest single
# scratch buffer in bytes. A live-bytes total is deliberately NOT tracked by the native runtime.
_MEMORY_COUNTERS = ("scratch_allocs", "scratch_peak_bytes")
# Scheduler / cache counters that only move under a compiled .so step body (Kokkos/ROMEO); absent
# (the honest zero) on the native host path.
_ADVANCED_COUNTERS = ("cache_hits", "cache_misses", "nodes_due", "nodes_skipped")
# Elliptic-solver counters (Spec 5 sec.13.11.1, ADC-479 criteria 42/43): the System reads these back
# at the field_solve seam (system.cpp). The elliptic solve is 96-99.9% of step cost, so these break the
# opaque "field_solve" scope into actionable numbers. mg_cycles / krylov_iters accumulate; mg_levels is
# the (constant) hierarchy depth via count_max; elliptic_bottom is a TIMING SCOPE (the coarsest-grid
# self-time). A direct FFT solver reports honest zeros (no cycles / levels / iters / bottom solve).
_ELLIPTIC_COUNTERS = ("mg_cycles", "krylov_iters", "mg_levels")
_ELLIPTIC_TIME_SCOPE = "elliptic_bottom"

# AMR / MPI phase timings + counters (Spec 5 sec.12.5, ADC-479 criterion 43): regrid, halo exchange
# (fill_boundary / fill_ghosts), reflux, average_down, plus MPI reduction/message counts -- TIMING
# SCOPES (chrono self-time, like elliptic_bottom) and integer COUNTERS. Names are matched flexibly:
# a scope/counter whose name CONTAINS a token lands in that bucket ("fill_boundary" and
# "halo_exchange" both count as halo). On a host / non-AMR build none of these scopes or counters
# exist, so the view declares itself unavailable (never a faked 0).
_AMR_MPI_TIME_TOKENS = ("regrid", "fill_boundary", "halo_exchange", "reflux", "average_down")
_AMR_MPI_COUNTER_TOKENS = (
    "regrid", "fill_boundary", "halo_exchange", "halo_exchanges", "reflux", "average_down",
    "mpi_reductions", "mpi_messages",
    # ADC-607 data-structure counters: tag_density = tagged/total permille of the dense TagBox
    # (dense-vs-sparse measured); the others = copy-schedule cache engagement.
    "tag_density", "box_hash_rebuilds", "copy_cache_hits", "copy_cache_misses")

# POPS_PROFILE: map sim.profile() called with NO argument to a default level.
_ENV_VAR = "POPS_PROFILE"
_ENV_OFF = ("", "0", "off", "false", "no", "none")
_ENV_ADVANCED = ("advanced", "2", "full")


class Profile:
    """A typed profiling level (Spec 5 sec.12.5). Inert: it carries no timers.

    Use the named constructors rather than a string flag::

        with sim.profile(pops.Profile.Basic()) as prof:
            sim.run(0.1)
        print(prof.summary())

    ``Basic`` surfaces the coarse phase timings + the kernel/step counters; ``Advanced`` also asks
    for the per-program-node timings and the scheduler/memory counters (which only populate under a
    compiled step on a Kokkos build -- declared unavailable, never faked, otherwise).
    """

    __slots__ = ("level",)

    #: The two recognised levels.
    _LEVELS = ("basic", "advanced")

    def __init__(self, level: str = "basic") -> None:
        if level not in self._LEVELS:
            raise ValueError(
                "Profile level must be one of %s (got %r)" % (self._LEVELS, level))
        self.level = level

    @classmethod
    def Basic(cls) -> Any:
        """Coarse phase timings + step / kernel counters."""
        return cls("basic")

    @classmethod
    def Advanced(cls) -> Any:
        """Per-program-node timings + scheduler / memory counters (Kokkos-gated; honest about gaps)."""
        return cls("advanced")

    @property
    def advanced(self) -> Any:
        """True for the Advanced level (asks for the per-node / scheduler / memory views)."""
        return self.level == "advanced"

    @classmethod
    def from_env(cls, default: Any = None) -> Any:
        """Resolve the level from ``POPS_PROFILE`` (sim.profile() with no argument).

        Unset / ``0`` / ``off`` -> @p default (a Basic() when @p default is None); ``advanced`` /
        ``full`` -> Advanced(); anything else -> Basic(). Returns None when the env asks for OFF and
        no @p default is given, so the caller can leave profiling disabled.
        """
        raw = os.environ.get(_ENV_VAR)
        if raw is None or raw.strip().lower() in _ENV_OFF:
            return default
        if raw.strip().lower() in _ENV_ADVANCED:
            return cls.Advanced()
        return cls.Basic()

    def __eq__(self, other: Any) -> Any:
        return isinstance(other, Profile) and other.level == self.level

    def __hash__(self) -> Any:
        return hash(("Profile", self.level))

    def __repr__(self) -> Any:
        return "Profile.%s()" % self.level.capitalize()


def _parse_report(report: Any) -> Any:
    """Parse the native ``profile_report()`` string into a structured dict.

    The C++ Profiler renders (profiler.hpp ``report()``)::

        Profiler report (total 0.010849 s, 2 scopes)
          step  count=2  total=0.007229s  mean=0.003614s  min=...s  max=...s
          field_solve  count=1  total=...s  ...
        counters:  steps=2  kernels=3

    Returns ``{"scopes": {name: {count,total_s,mean_s,min_s,max_s}}, "counters": {name: int},
    "total_s": float}``. An empty / unrecognised report yields empty tables (never raises).
    """
    scopes = {}
    counters = {}
    total_s = 0.0
    for line in (report or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Profiler report"):
            total_s = _extract_float(stripped, "total ")
            continue
        if stripped.startswith("counters:"):
            for tok in stripped[len("counters:"):].split():
                if "=" in tok:
                    key, val = tok.split("=", 1)
                    counters[key] = _to_int(val)
            continue
        # A scope line: "<name>  count=..  total=..s  mean=..s  min=..s  max=..s".
        if "count=" in stripped:
            name = stripped.split("  ", 1)[0].strip()
            fields = {}
            for tok in stripped.split():
                if "=" in tok:
                    key, val = tok.split("=", 1)
                    if key in ("count",):
                        fields["count"] = _to_int(val)
                    elif key in ("total", "mean", "min", "max"):
                        fields["%s_s" % key] = _to_float(val.rstrip("s"))
            if name:
                scopes[name] = fields
    return {"scopes": scopes, "counters": counters, "total_s": total_s}


def _parse_snapshot(snapshot: Any) -> Any:
    """Normalize the C++ ``profile_snapshot()`` dict into the PerformanceSummary internal shape."""
    scopes = {}
    counters = {}
    for row in snapshot.get("scopes", []) or []:
        name = row.get("name")
        if not name:
            continue
        scopes[str(name)] = {
            "count": _to_int(row.get("count")),
            "total_s": _to_float(row.get("total_s")),
            "mean_s": _to_float(row.get("mean_s")),
            "min_s": _to_float(row.get("min_s")),
            "max_s": _to_float(row.get("max_s")),
        }
    for row in snapshot.get("counters", []) or []:
        name = row.get("name")
        if name:
            counters[str(name)] = _to_int(row.get("value"))
    total_s = _to_float(snapshot.get("total_s"))
    if total_s == 0.0 and scopes:
        total_s = sum(v.get("total_s", 0.0) for v in scopes.values())
    return {
        "schema_version": _to_int(snapshot.get("schema_version")),
        "enabled": bool(snapshot.get("enabled", False)),
        "scopes": scopes,
        "counters": counters,
        "total_s": total_s,
    }


def _extract_float(text: Any, after: Any) -> Any:
    """Best-effort: the float token that follows @p after in @p text (else 0.0)."""
    idx = text.find(after)
    if idx < 0:
        return 0.0
    return _to_float(text[idx + len(after):].split()[0]) if text[idx + len(after):].split() else 0.0


def _to_float(token: Any) -> Any:
    try:
        return float(token)
    except (TypeError, ValueError):
        return 0.0


def _to_int(token: Any) -> Any:
    try:
        return int(token)
    except (TypeError, ValueError):
        return 0


class _Unavailable:
    """A sentinel view: a measure the current build does not surface (declared, not faked)."""

    __slots__ = ("measure", "reason")

    def __init__(self, measure: Any, reason: Any) -> None:
        self.measure = measure
        self.reason = reason

    @property
    def available(self) -> Any:
        return False

    def to_dict(self) -> Any:
        return {"available": False, "measure": self.measure, "reason": self.reason, "entries": {}}

    def __bool__(self) -> Any:
        return False

    def __repr__(self) -> Any:
        return "<unavailable %s: %s>" % (self.measure, self.reason)


# PerformanceSummary (the printable wrapper over the parsed report) is split into
# ``_profile_summary`` for the 500-line cap (ADC-550) and re-exported here so
# ``from pops.runtime.profile import PerformanceSummary`` is unchanged. It imports the parsing
# helpers, the Profile level and the _Unavailable sentinel back from this module.
from pops.runtime._profile_summary import PerformanceSummary  # noqa: E402,F401


__all__ = ["Profile", "PerformanceSummary"]
