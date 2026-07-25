"""Compile-time validation of the history-persistence policies (ADC-626).

A non-Dense policy (:class:`pops.time.Interval` / :class:`~pops.time.Revolve`) recomputes the
non-stored ring slots at restart by replaying the installed Program from one exact owner-state
anchor. That is bit-identical only for a primary-clock, single-owner transition whose committed
state is a strictly affine function of ``(owner_state, dt)``. This module runs the compile-time
gate the plan's sec.5 describes:

  1. every declared ring has exactly one typed policy at the same compiled depth;
  2. per ring, ``policy.validate_for(depth)`` -- coherence (k / snapshots vs depth), loud;
  3. if the policy is non-Dense, prove the committed owner transition and the complete Program use
     only the affine replay vocabulary; separately refuse any non-deterministic op / brick.

FAIL-CLOSED (plan R3): the built-in time-DSL op vocabulary is entirely deterministic (pure
numerical operators: arithmetic combines, flux divergence, gradient, field / linear solves,
Schur stages, history, control flow -- no RNG, no wall-clock, no stochastic source). An op OUTSIDE
that vetted allow-list, or one referencing an external brick that DECLARES itself
non-deterministic, is treated as non-deterministic-unless-proven and refused -- rather than risking
a silent replay drift. Deterministic does not imply replay-safe: selective replay is restricted
further to the owner-affine subset below. Every refusal composes into the caller's ``ReportTree``.
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

# Native selective replay restores one qualified owner state and re-executes one full primary-clock
# Program step. The only transition currently proven bit-exact is an affine graph made exclusively
# of the restored state and linear combinations whose coefficients are exact polynomials in dt.
# An RHS/source/operator/field solve may be deterministic in isolation, but its native execution can
# read ghost, auxiliary, warm-start, boundary, topology, field or time context that a single state
# anchor does not restore. Keep the gate deliberately smaller than the general deterministic DSL.
_OWNER_AFFINE_OPS = frozenset({"state", "linear_combine"})

# ``history`` is accepted here only as bookkeeping for an exact-zero ``prev`` depth declaration;
# `_effective_inputs` below proves that it is not load-bearing in the committed transition.
_AFFINE_REPLAY_PROGRAM_OPS = frozenset({
    "state", "history", "store_history", "linear_combine",
})


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


def _effective_inputs(value):
    """Return inputs which can numerically influence @p value.

    Affine lowering preserves exact zero coefficients as an empty CoeffPolynomial. Ignoring only
    those canonical zeros keeps ``0 * U.prev(k)`` usable as a depth-declaration edge while refusing
    every load-bearing lag dependency.
    """
    inputs = tuple(getattr(value, "inputs", ()) or ())
    if getattr(value, "op", None) != "linear_combine":
        return inputs
    coeffs = tuple((getattr(value, "attrs", {}) or {}).get("coeffs", ()))
    if len(coeffs) != len(inputs):
        return inputs  # malformed/unknown metadata is handled fail-closed by the caller
    return tuple(item for item, coeff in zip(inputs, coeffs, strict=True) if bool(coeff))


def _owner_affine_refusal(program, state):
    """Return why @p state's committed transition is not strictly owner-affine, or ``None``."""
    committed = (getattr(program, "_commits", {}) or {}).get(state.state)
    if committed is None:
        return "the owner state has no committed primary-step transition"
    owner_block = state.block
    owner_state = state.state
    seen = set()

    def visit(value):
        identity = id(value)
        if identity in seen:
            return None
        seen.add(identity)
        op = getattr(value, "op", None)
        if op == "history":
            attrs = getattr(value, "attrs", {}) or {}
            return "the committed transition depends on lagged history %r" % attrs.get("history")
        if op not in _OWNER_AFFINE_OPS:
            return "the committed transition reaches unproved replay op %r" % op
        block = getattr(value, "block", None)
        if block is not None and block != owner_block:
            return "the committed transition depends on another block"
        if op == "state":
            state_ref = getattr(value, "state_ref", None)
            if state_ref is None:
                state_ref = (getattr(value, "attrs", {}) or {}).get("state")
            if state_ref != owner_state:
                return "the committed transition depends on another state"
        for input_value in _effective_inputs(value):
            reason = visit(input_value)
            if reason is not None:
                return reason
        return None

    return visit(committed)


def _program_context_refusal(program):
    """Reject any Program work outside the proven affine replay vocabulary."""
    for value in _walk_ops(getattr(program, "_values", ()) or ()):
        op = getattr(value, "op", None)
        if op not in _AFFINE_REPLAY_PROGRAM_OPS:
            return "the Program executes non-affine/context-dependent op %r" % op
    return None


def validate_history_persistence(program, report: ReportTree) -> ReportTree:
    """Accumulate the ADC-626 compile-time refusals into @p report for @p program.

    Require an exact one-to-one mapping between declared manual/temporal rings and compiled policy
    records, validate every policy against the physical ``max_lag + 1`` slot count, then -- for a
    non-Dense policy -- scan the Program op graph and refuse if any op is (or may be)
    non-deterministic (the replay would silently drift). @p report is an immutable validation tree;
    every issue carries the
    source ``"history_persistence"`` and the ring name for a precise, verbatim message.
    Returns @p report (chains)."""
    histories = dict(getattr(program, "_histories", None) or {})
    persistence = dict(getattr(program, "_history_persistence", None) or {})
    history_names = set(histories)
    persistence_names = set(persistence)
    for name in sorted(history_names - persistence_names):
        report = report.error(
            "history_persistence", "missing_policy",
            "history %r has no compiled persistence policy" % name,
            context={"history": name})
    for name in sorted(persistence_names - history_names):
        report = report.error(
            "history_persistence", "orphan_policy",
            "history persistence policy %r has no declared ring" % name,
            context={"history": name})
    if not persistence:
        return report
    # Scan once: the first non-deterministic op (if any) is shared by every non-Dense ring's replay.
    nondet_reason = None
    for op in _walk_ops(getattr(program, "_values", []) or []):
        reason = _op_is_nondeterministic(op)
        if reason is not None:
            nondet_reason = reason
            break
    context_reason = _program_context_refusal(program)
    from pops.time._history.persistence import HistoryPersistence
    # Selective native replay seeds the owner block with an exact older state sample and re-executes
    # the Program forward. Only keep_history establishes both facts: its stored value is U.n and its
    # store is ordered before the tail commit/rotate. A manual store_history may hold an RHS or a
    # post-commit field and has no such phase provenance; accepting a selective policy for it would
    # make the runtime guess which dt/value relation the ring represents.
    replay_provenance = {
        getattr(node, "attrs", {}).get("history"): state
        for state, node in (getattr(program, "_time_history_stores", {}) or {}).items()
    }

    for name, configured in sorted(persistence.items()):
        if not isinstance(configured, tuple) or len(configured) != 2:
            report = report.error(
                "history_persistence", "invalid_policy_record",
                "history %r has an invalid compiled persistence record" % name,
                context={"history": name})
            continue
        ring_slots, policy = configured
        declared_lag = histories.get(name)
        expected_slots = None if declared_lag is None else declared_lag + 1
        if expected_slots is not None and ring_slots != expected_slots:
            report = report.error(
                "history_persistence", "depth_mismatch",
                "history %r persistence slot count %r differs from declared max lag %r "
                "(expected %r slots)"
                % (name, ring_slots, declared_lag, expected_slots),
                context={"history": name})
            continue
        if not isinstance(policy, HistoryPersistence):
            report = report.error(
                "history_persistence", "invalid_policy",
                "history %r persistence policy is not a typed HistoryPersistence" % name,
                context={"history": name})
            continue
        try:
            policy.validate_for(ring_slots)
        except (ValueError, TypeError) as exc:
            report = report.error(
                "history_persistence", "incoherent_policy",
                "history %r: %s" % (name, exc), context={"history": name})
            continue
        if policy.degenerate_to_dense(ring_slots):
            continue  # Dense (or a budget-covers-all ring): no replay, never refused
        if name not in replay_provenance:
            report = report.error(
                "history_persistence", "unqualified_replay_provenance",
                "history %r uses %s but was not declared by keep_history; selective replay "
                "requires a pre-commit owner-state sample with an authenticated outgoing-dt "
                "ledger -- use keep_history for a state ring, or Dense() for a manual store"
                % (name, policy.name),
                context={"history": name})
        else:
            state = replay_provenance[name]
            if state.clock != getattr(program, "clock", None):
                report = report.error(
                    "history_persistence", "non_primary_clock_replay",
                    "history %r uses %s on child clock %r, but selective replay executes one full "
                    "primary-clock Program step; use Dense() until a child-clock replay provider "
                    "exists"
                    % (name, policy.name, getattr(state.clock, "name", state.clock)),
                    context={"history": name})
            reason = _owner_affine_refusal(program, state)
            if reason is not None:
                report = report.error(
                    "history_persistence", "non_affine_replay",
                    "history %r uses %s but native selective replay restores only one exact "
                    "owner-state anchor and proves only a strictly affine owner transition: %s; "
                    "use Dense() until a complete historical-context provider exists"
                    % (name, policy.name, reason),
                    context={"history": name})
            if context_reason is not None:
                report = report.error(
                    "history_persistence", "unrestored_replay_context",
                    "history %r uses %s but single-anchor replay cannot reproduce the complete "
                    "historical execution context: %s; use Dense() until that context has a typed "
                    "checkpoint/replay provider"
                    % (name, policy.name, context_reason),
                    context={"history": name})
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
