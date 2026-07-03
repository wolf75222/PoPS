"""pops.runtime._profile_summary -- the PerformanceSummary view over a native profile report.

Split out of :mod:`pops.runtime.profile` for the 500-line cap (ADC-550): the parsing helpers,
the :class:`Profile` level and the :class:`_Unavailable` sentinel stay in ``profile``; the
printable :class:`PerformanceSummary` wrapper and its :func:`_view_to_dict` serialiser live here.
``pops.runtime.profile`` re-exports :class:`PerformanceSummary`, so
``from pops.runtime.profile import PerformanceSummary`` is unchanged.

Like ``profile``, this is a pure typed/parsing wrapper: it imports neither ``_pops`` nor numpy.
The native extension is reached only through the :class:`System` instance the context manager the
summary is built by is bound to.
"""

import json

from pops.runtime.profile import (
    _AMR_MPI_COUNTER_TOKENS,
    _AMR_MPI_TIME_TOKENS,
    _ELLIPTIC_COUNTERS,
    _ELLIPTIC_TIME_SCOPE,
    _MEMORY_COUNTERS,
    _NODE_PREFIX,
    _SOLVER_NODE_HINT,
    _SOLVER_SCOPES,
    Profile,
    _parse_report,
    _parse_snapshot,
    _Unavailable,
)


class PerformanceSummary:
    """A printable, typed wrapper around the native profile report (Spec 5 criteria 41-43).

    Built from the structured ``profile_snapshot()`` dict when available, or from the legacy string
    :meth:`System.profile_report` returns. It exposes the report as a structured dict
    (:meth:`to_dict` / :meth:`to_json`)
    and typed views: :meth:`by_program_node`, :meth:`by_native_brick`, :meth:`by_solver`,
    :meth:`by_elliptic`, :meth:`by_amr_mpi`, :meth:`by_memory`. Views read the parsed native tables; a
    view the build does not surface returns an :class:`_Unavailable` sentinel (``bool(view) is False``)
    rather than a faked zero.
    """

    def __init__(self, report, profile=None):
        self._snapshot = dict(report) if isinstance(report, dict) else None
        self._report_text = "" if self._snapshot is not None else (report or "")
        self._profile = profile if profile is not None else Profile.Basic()
        self._parsed = (_parse_snapshot(self._snapshot)
                        if self._snapshot is not None else _parse_report(self._report_text))

    # ---- raw access -------------------------------------------------------------------------
    @property
    def profile(self):
        """The :class:`Profile` level the run requested."""
        return self._profile

    @property
    def raw_report(self):
        """The exact legacy string the native profiler returned, or ``""`` for snapshot input."""
        return self._report_text

    @property
    def source(self):
        """``"snapshot"`` for structured C++ input, ``"text"`` for the legacy parser path."""
        return "snapshot" if self._snapshot is not None else "text"

    def scopes(self):
        """All timed scopes: ``{name: {count, total_s, mean_s, min_s, max_s}}``."""
        return dict(self._parsed["scopes"])

    def counters(self):
        """All integer counters: ``{name: int}``."""
        return dict(self._parsed["counters"])

    def total_s(self):
        """Sum of every scope's total wall-clock time (seconds)."""
        return self._parsed["total_s"]

    # ---- typed views ------------------------------------------------------------------------
    def by_program_node(self):
        """Per-program-node timings (the ``node:<name>`` scopes the compiled step emits).

        Keys are the bare node names (``rhs2``, ``solve_fields1``, ...). Empty on a native step
        (no compiled Program); populated under a compiled ``.so`` step.
        """
        nodes = {name[len(_NODE_PREFIX):]: dict(fields)
                 for name, fields in self._parsed["scopes"].items()
                 if name.startswith(_NODE_PREFIX)}
        return nodes

    def by_native_brick(self):
        """Per-native-brick timings.

        The native runtime times Program nodes and coarse phases, not individual bricks: there is no
        per-brick scope to read. Declared unavailable rather than faked (the per-brick granularity is
        a documented follow-up wired through the compiled ProgramContext).
        """
        return _Unavailable(
            "by_native_brick",
            "native runtime times program nodes / phases, not individual bricks")

    def by_solver(self):
        """Solver-attributable timings: the elliptic field-solve phase + any solve_fields node.

        Reads the coarse ``field_solve`` phase and the ``node:solve_fields*`` program nodes. Empty
        when no field solve ran under profiling.
        """
        out = {}
        for name, fields in self._parsed["scopes"].items():
            if name in _SOLVER_SCOPES:
                out[name] = dict(fields)
            elif name.startswith(_NODE_PREFIX) and _SOLVER_NODE_HINT in name:
                out[name[len(_NODE_PREFIX):]] = dict(fields)
        return out

    def by_elliptic(self):
        """Elliptic-solver counters: the most actionable view given the elliptic-solve dominance.

        The elliptic field solve is 96-99.9% of step cost (Spec 5 sec.13.11.1), yet ``by_solver`` only
        exposes the opaque ``field_solve`` phase. This view breaks it down with the native counters the
        System reads back at the field_solve seam (ADC-479 criteria 42/43):

        * ``mg_cycles`` -- total geometric-multigrid V-cycles over the run;
        * ``krylov_iters`` -- total Krylov iterations (0 on the default Poisson path: it uses multigrid
          / a direct FFT, never a Krylov elliptic solver);
        * ``mg_levels`` -- multigrid hierarchy depth (a structural constant, reported as the peak);
        * ``elliptic_bottom`` -- coarsest-grid (bottom) solve self-time, as a timing entry
          ``{count, total_s, mean_s, min_s, max_s}``.

        Returns ``{name: int | timing-dict}`` with only the counters / scope the run actually produced.
        A direct FFT solver yields zeros (no cycles / levels / bottom solve), and a build whose ``_pops``
        predates these counters (no ``mg_cycles`` etc.) declares the view unavailable rather than faking.
        """
        out = {name: self._parsed["counters"][name]
               for name in _ELLIPTIC_COUNTERS if name in self._parsed["counters"]}
        bottom = self._parsed["scopes"].get(_ELLIPTIC_TIME_SCOPE)
        if bottom is not None:
            out[_ELLIPTIC_TIME_SCOPE] = dict(bottom)
        if not out:
            return _Unavailable(
                "by_elliptic",
                "elliptic-solver counters need a field-solve under profiling on a _pops build "
                "that emits them (mg_cycles / krylov_iters / mg_levels / elliptic_bottom)")
        return out

    def by_amr_mpi(self):
        """AMR / MPI phase timings + counters: the distributed-runtime dimension (criterion 43).

        Spec 5 sec.12.5 requires time attributable to AMR / MPI alongside the program-node, native-brick,
        solver, and memory views. The distributed AMR runtime spends its non-numeric time in named
        phases the C++ profiler can scope and count:

        * ``regrid`` -- rebuilding the patch hierarchy (timing scope + a per-run count);
        * ``fill_boundary`` / ``halo_exchange`` -- the cross-rank ghost-cell halo exchange;
        * ``reflux`` -- the coarse-fine flux correction at refinement boundaries;
        * ``average_down`` -- restricting fine-level data onto the coarse level;
        * ``mpi_reductions`` / ``mpi_messages`` -- MPI collective / point-to-point counts.

        A scope whose name contains one of the timing tokens is surfaced as a timing entry
        (``{count, total_s, mean_s, min_s, max_s}``); a counter whose name contains one of the counter
        tokens is surfaced as an int. Returns ``{name: int | timing-dict}`` with only the phases the run
        actually produced.

        On the common host / serial / non-AMR build NONE of these scopes or counters exist -- no C++
        path emits them yet (the native regrid / halo / reduction timers are a documented follow-up in
        ``include/pops/runtime/program`` and the AMR runtime). When the report carries none of them this
        view declares itself :class:`_Unavailable` honestly rather than fabricating a zero, exactly like
        :meth:`by_native_brick` / :meth:`by_memory` on a host build. It surfaces real numbers as soon as
        a distributed AMR run under profiling emits the scopes.
        """
        out = {}
        for name, fields in self._parsed["scopes"].items():
            if any(token in name for token in _AMR_MPI_TIME_TOKENS):
                out[name] = dict(fields)
        for name, value in self._parsed["counters"].items():
            # A phase emitted as a TIMING scope (regrid / fill_boundary / average_down) already
            # carries its call count in the timing dict; its bare same-named counter is redundant, so
            # never clobber the richer timing entry with the int. Genuine counter-only names
            # (mpi_reductions / mpi_messages) are still surfaced as ints.
            if name in out:
                continue
            if any(token in name for token in _AMR_MPI_COUNTER_TOKENS):
                out[name] = value
        if not out:
            return _Unavailable(
                "by_amr_mpi",
                "AMR / MPI phase timings and counters (regrid / fill_boundary / halo_exchange / "
                "reflux / average_down / mpi_reductions) populate only under a distributed AMR run; "
                "no scope is emitted on a host / non-AMR build")
        return out

    def by_memory(self):
        """Scratch-memory counters: allocation count + the largest single scratch buffer (bytes).

        Reads ``scratch_allocs`` / ``scratch_peak_bytes`` (program_context.hpp ``count_scratch``).
        These move only under a compiled step on a Kokkos build; on a native host step neither
        counter is created, so this view declares itself unavailable rather than faking a 0.
        """
        present = {name: self._parsed["counters"][name]
                   for name in _MEMORY_COUNTERS if name in self._parsed["counters"]}
        if not present:
            return _Unavailable(
                "by_memory",
                "scratch memory counters populate only under a compiled Kokkos step")
        return present

    # ---- serialisation ----------------------------------------------------------------------
    def to_dict(self):
        """The full structured report: level + scopes + counters + total, plus the typed views.

        ``by_native_brick`` / ``by_amr_mpi`` / ``by_memory`` serialise their availability honestly (an
        unavailable view records ``{"available": False, "reason": ...}``).
        """
        return {
            "profile": self._profile.level,
            "source": self.source,
            "schema_version": self._parsed.get("schema_version", 0),
            "enabled": self._parsed.get("enabled"),
            "total_s": self.total_s(),
            "scopes": self.scopes(),
            "counters": self.counters(),
            "views": {
                "by_program_node": self.by_program_node(),
                "by_native_brick": _view_to_dict(self.by_native_brick()),
                "by_solver": self.by_solver(),
                "by_elliptic": _view_to_dict(self.by_elliptic()),
                "by_amr_mpi": _view_to_dict(self.by_amr_mpi()),
                "by_memory": _view_to_dict(self.by_memory()),
            },
        }

    def to_json(self, path=None):
        """Serialise :meth:`to_dict` to JSON. Writes to @p path when given; returns the JSON string."""
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path is not None:
            with open(path, "w", encoding="ascii") as handle:
                handle.write(text)
        return text

    # ---- printable --------------------------------------------------------------------------
    def __str__(self):
        if not self._parsed["scopes"] and not self._parsed["counters"]:
            return "PerformanceSummary(%s): no profiling data recorded" % self._profile.level
        lines = ["PerformanceSummary (%s, total %.6f s, %d scopes)"
                 % (self._profile.level, self.total_s(), len(self._parsed["scopes"]))]
        for name, fields in self._parsed["scopes"].items():
            lines.append("  %-24s count=%d total=%.6fs mean=%.6fs"
                         % (name, fields.get("count", 0),
                            fields.get("total_s", 0.0), fields.get("mean_s", 0.0)))
        if self._parsed["counters"]:
            counters = "  ".join("%s=%d" % (k, v) for k, v in self._parsed["counters"].items())
            lines.append("counters: %s" % counters)
        return "\n".join(lines)

    def print(self):
        """Print the human-readable summary (``print(summary)`` sugar)."""
        print(str(self))

    def __repr__(self):
        return "PerformanceSummary(profile=%r, scopes=%d, counters=%d)" % (
            self._profile.level, len(self._parsed["scopes"]), len(self._parsed["counters"]))


def _view_to_dict(view):
    """Serialise a typed view: a dict passes through; an _Unavailable records its availability."""
    if isinstance(view, _Unavailable):
        return view.to_dict()
    return {"available": True, "entries": view}


__all__ = ["PerformanceSummary"]
