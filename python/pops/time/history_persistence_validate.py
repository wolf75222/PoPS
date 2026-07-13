"""Compile-time validation of the history-persistence policies (ADC-626).

A non-Dense policy (:class:`pops.time.Interval` / :class:`~pops.time.Revolve`) recomputes the
non-stored ring slots at restart by DETERMINISTIC replay of the installed Program. That is
bit-identical ONLY if the Program's macro-step is a deterministic function of ``(state, dt,
params)``. This module runs the compile-time gate the plan's sec.5 describes:

  1. per ring, ``policy.validate_for(depth)`` -- coherence (k / snapshots vs depth), loud;
  2. if the policy is non-Dense, scan the whole Program op graph; if it reaches a
     NON-DETERMINISTIC op / brick, refuse loudly (never a silent degrade to Dense).

FAIL-CLOSED (plan R3): the built-in time-DSL op vocabulary is entirely deterministic (pure
numerical operators: arithmetic combines, flux divergence, gradient, field / linear solves,
Schur stages, history, control flow -- no RNG, no wall-clock, no stochastic source). An op OUTSIDE
that vetted allow-list, or one referencing an external brick that DECLARES itself
non-deterministic, is treated as non-deterministic-unless-proven and refused -- rather than risking
    a silent replay drift. The refusal composes into the caller's ``ReportTree`` (verbatim
message tested).
"""

from pops._report import ReportTree

#: The complete built-in time-DSL op vocabulary (every op ``Program._new`` emits). Each is a pure,
#: deterministic function of ``(state, dt, params)`` -- there is no stochastic / RNG / wall-clock op
#: in the language. An op OUTSIDE this set is treated as non-deterministic-unless-proven (fail-closed):
#: the pass refuses a non-Dense policy whose replay reaches it, rather than risk a silent drift.
_KNOWN_DETERMINISTIC_OPS = frozenset({
    "state", "history", "store_history", "linear_combine", "project",
    "solve_fields", "solve_fields_from_blocks", "solve_coupled_implicit",
    "rhs", "apply", "source", "linear_source", "coupled_rate", "coupled_rate_out",
    "divergence", "gradient", "laplacian", "apply_laplacian_coeff",
    "apply_in", "apply_out", "cell_compare", "scalar_field", "rhs_jacvec",
    "solve_linear", "solve_local_linear", "matrix_free_operator",
    "solve_outcome", "solve_outcome_component",
    "schur_coeffs", "schur_energy", "schur_reconstruct", "schur_explicit_flux", "schur_rhs",
    "cfl", "hmin", "max_wave_speed", "record_scalar", "reduce", "scalar_op", "compare",
    "while", "range", "subcycle", "branch", "synchronize", "solve_local_nonlinear",
})

#: The capability tag an external brick clears to declare it is NOT a deterministic function of its
#: inputs (an RNG / stochastic source). A brick that supports it under a non-Dense policy is refused.
_NONDETERMINISM_TAG = "nondeterministic"


def _walk_ops(values):
    """Yield every op node of a Program, descending into control-flow sub-blocks (which carry their
    own nested node lists in attrs). Order is definition order; sub-block ops follow their owner."""
    for v in values:
        yield v
        attrs = getattr(v, "attrs", {}) or {}
        for key in ("cond_block", "body_block", "true_block", "false_block",
                    "apply_block", "residual_block"):
            block = attrs.get(key)
            if block:
                yield from _walk_ops(block)


def _op_is_nondeterministic(op):
    """Whether op node @p op is (or may be) non-deterministic under replay.

    True when the op is OUTSIDE the vetted built-in allow-list (fail-closed: an unknown op the pass
    has not proven pure), OR it references an external brick whose descriptor declares the
    ``supports_nondeterministic`` capability. Returns the reason string, or ``None`` when the op is
    provably deterministic."""
    op_name = getattr(op, "op", None)
    if op_name not in _KNOWN_DETERMINISTIC_OPS:
        return ("op %r is not on the vetted deterministic-op allow-list" % op_name)
    # An op may carry an external brick / operator descriptor in its attrs (a bound operator, a
    # custom source). If that descriptor DECLARES non-determinism, refuse (fail-closed on the
    # declaration; a brick that never declares it is trusted as deterministic by default).
    attrs = getattr(op, "attrs", {}) or {}
    for value in attrs.values():
        caps = getattr(value, "capabilities", None)
        if callable(caps):
            try:
                supports = getattr(caps(), "supports", None)
                if callable(supports) and supports(_NONDETERMINISM_TAG):
                    return ("op %r references a brick declaring it is non-deterministic" % op_name)
            except Exception:  # noqa: BLE001 -- a descriptor whose capabilities() raises is not our gate
                continue
    return None


def validate_history_persistence(program, report: ReportTree) -> ReportTree:
    """Accumulate the ADC-626 compile-time refusals into @p report for @p program.

    For each ``keep_history`` ring (recorded on ``program._history_persistence`` as ``(depth,
    policy)``): validate the policy coherence against depth (loud), then -- for a non-Dense policy --
    scan the Program op graph and refuse if any op is (or may be) non-deterministic (the replay would
    silently drift). @p report is an immutable :class:`pops.ReportTree`; every issue carries the
    source ``"history_persistence"`` and the ring name for a precise, verbatim message.
    Returns @p report (chains)."""
    persistence = getattr(program, "_history_persistence", None) or {}
    if not persistence:
        return report
    # Scan once: the first non-deterministic op (if any) is shared by every non-Dense ring's replay.
    nondet_reason = None
    for op in _walk_ops(getattr(program, "_values", []) or []):
        reason = _op_is_nondeterministic(op)
        if reason is not None:
            nondet_reason = reason
            break
    for name, (depth, policy) in sorted(persistence.items()):
        try:
            policy.validate_for(depth)
        except (ValueError, TypeError) as exc:
            report = report.error(
                "history_persistence", "incoherent_policy",
                "history %r: %s" % (name, exc), context={"history": name})
            continue
        if policy.degenerate_to_dense(depth):
            continue  # Dense (or a budget-covers-all ring): no replay, never refused
        if nondet_reason is not None:
            report = report.error(
                "history_persistence", "nondeterministic_replay",
                "history %r uses %s which recomputes ring slots by deterministic replay, but the "
                "Program step reaches a non-deterministic op (%s) -- use Dense() (store every slot) "
                "or make the op deterministic" % (name, policy.name, nondet_reason),
                context={"history": name})
    return report


def check_program(program):
    """Run the ADC-626 compile-time gate over @p program and RAISE (loud) on any refusal.

    The single entry point ``Program.validate`` calls at ``pops.compile``: builds a report, runs
    :func:`validate_history_persistence`, and raises via ``raise_if_error``. A Program with no
    non-Dense ring adds no cost (the scan short-circuits on an empty persistence map)."""
    report = ReportTree(
        phase="validation", severity="info", code="validation.history_persistence.report",
        source="history_persistence", owner=program,
    )
    report = validate_history_persistence(program, report)
    report.raise_if_error()


__all__ = ["validate_history_persistence", "check_program"]
