"""History-ring checkpoint serialization + selective-persistence restart (ADC-626).

Split out of :mod:`pops.runtime._system_io` (the 500-line cap): the checkpoint WRITER emits, per
history ring, the authoring policy and one resolved physical storage plan. Selective slots are used
only after every logical slot is authentic and on a stable hierarchy window. Cold/partially filled
rings are promoted to ``dense_cold_start_safety``; a window containing a scheduled regrid is promoted
to ``dense_regrid_safety``. Restart replays only gaps proven exact by both guards.
"""
from dataclasses import dataclass
import json


@dataclass(frozen=True, slots=True)
class HistoryRingCapture:
    """Collective-free metadata fixing one ring's subsequent gather order."""

    name: str
    depth: int
    ncomp: int
    initialized: bool
    fill_count: int
    policy_json: str
    requested_stored_slots: tuple[int, ...]
    stored_slots: tuple[int, ...]
    storage_mode: str
    regrid_steps: tuple[int, ...] | None
    slot_dt: tuple[float, ...] | None

    def to_data(self):
        return {
            "name": self.name,
            "depth": self.depth,
            "ncomp": self.ncomp,
            "initialized": self.initialized,
            "fill_count": self.fill_count,
            "policy_json": self.policy_json,
            "requested_stored_slots": list(self.requested_stored_slots),
            "stored_slots": list(self.stored_slots),
            "storage_mode": self.storage_mode,
            "regrid_steps": (
                None if self.regrid_steps is None else list(self.regrid_steps)
            ),
            # This projection participates in the checkpoint capture identity.  Canonical CBOR
            # intentionally refuses Python floats, so preserve each binary64 value exactly.
            "slot_dt": (
                None if self.slot_dt is None
                else [float(value).hex() for value in self.slot_dt]
            ),
        }


@dataclass(frozen=True, slots=True)
class HistoryCapturePlan:
    """Validated all-rank plan; no native collective has run while building it."""

    rings: tuple[HistoryRingCapture, ...]

    def to_data(self):
        return [ring.to_data() for ring in self.rings]


def resolve_ring_policy(policy, depth):
    """Validate the explicitly installed history-persistence policy for one ring."""
    from pops.time._history.persistence import HistoryPersistence
    if not isinstance(policy, HistoryPersistence):
        raise RuntimeError(
            "checkpoint history ring has no explicit compiled persistence policy")
    policy.validate_for(depth)
    return policy


def history_fill_count_from_payload(payload, name, depth, initialized):
    """Return one authenticated fill age, conservatively deriving legacy payloads.

    Older payloads recorded only ``initialized``.  They cannot prove that every cold-start copy has
    since been replaced, so an initialized legacy ring derives fill count one (zero otherwise).
    This may retain dense storage longer after restart, but can never authorize replay from a
    synthetic slot.
    """
    key = "history_fill_count_" + name
    fill_count = int(payload[key]) if key in payload else (1 if initialized else 0)
    if fill_count < 0 or fill_count > depth:
        raise ValueError(
            "history '%s' fill count %d is outside [0, %d]"
            % (name, fill_count, depth)
        )
    if initialized != (fill_count > 0):
        raise ValueError(
            "history '%s' has inconsistent initialized/fill-count metadata" % name
        )
    return fill_count


def replay_regrid_steps(depth, m, regrid_every):
    """Scheduled regrid cursors inside a selective ring's prospective replay window.

    A pure function of the ring depth, the checkpoint macro-step @p m and the cadence @p regrid_every,
    mirroring ``detail::AmrHistoryOps::scheduled_regrid_steps`` bit-for-bit. Replay restarts from each
    exact older stored anchor and sweeps that anchor gap toward its newer anchor. Across those gaps,
    the re-step producing slot j runs at cursor ``m-1-j`` (the original step that landed the ring on
    macro-step m-j ran with ``ctx.macro_step()==m-j-1``, pre-increment), and a regrid is due when that
    cursor is > 0 and divisible by @p regrid_every. A non-empty result forces
    ``dense_regrid_safety`` at capture and is persisted as its authenticated explanation. Empty when
    @p regrid_every is 0 (uniform or a frozen hierarchy) or the ring has depth < 2."""
    steps = set()
    if regrid_every and regrid_every > 0 and depth >= 2:
        for j in range(depth - 2, -1, -1):
            cursor = m - 1 - j
            if cursor > 0 and cursor % regrid_every == 0:
                steps.add(cursor)
    return sorted(steps)


def resolve_history_storage(policy, depth, *, fill_count, macro_step, regrid_every):
    """Resolve authoring intent to one exact, restart-safe physical storage plan.

    Selective replay is exact only after every logical ring slot contains an authentic accepted
    store, and while the replay window retains one hierarchy.  The first store broadcasts its value
    into cold slots for multistep startup; those copies are valid for evaluation but are not replay
    anchors.  A partially filled ring is therefore promoted to ``dense_cold_start_safety``.  A warm
    ring whose window contains a regrid is promoted to ``dense_regrid_safety``.  The requested slots,
    effective slots, mode and schedule fingerprint are persisted, so each promotion is observable and
    authenticated rather than a silent fallback.
    """
    policy = resolve_ring_policy(policy, depth)
    if isinstance(fill_count, bool) or not isinstance(fill_count, int):
        raise TypeError("history fill count must be a Python int")
    if fill_count < 0 or fill_count > depth:
        raise ValueError(
            "history fill count %d is outside [0, %d]" % (fill_count, depth))
    requested = tuple(int(slot) for slot in policy.stored_slots(depth))
    regrid_steps = None
    if len(requested) < depth:
        regrid_steps = tuple(replay_regrid_steps(
            depth, int(macro_step), int(regrid_every)))
    if len(requested) < depth and fill_count < depth:
        return (
            requested,
            tuple(range(depth)),
            "dense_cold_start_safety",
            regrid_steps,
        )
    if regrid_steps:
        return requested, tuple(range(depth)), "dense_regrid_safety", regrid_steps
    return requested, requested, "policy", regrid_steps


def prepare_history_capture(system, persistence, *, macro_step=0, regrid_every=0):
    """Validate every ring and freeze its exact gather order without a collective call."""
    names = tuple(str(name) for name in system.history_names())
    if len(names) != len(set(names)):
        raise ValueError("checkpoint history names must be unique")
    rings = []
    for hname in names:
        depth = int(system.history_depth(hname))
        ncomp = int(system.history_ncomp(hname))
        initialized = bool(system.history_initialized(hname))
        fill_count = int(system.history_fill_count(hname))
        if initialized != (fill_count > 0):
            raise RuntimeError(
                "checkpoint history '%s' has inconsistent initialized/fill-count metadata"
                % hname
            )
        policy = resolve_ring_policy(persistence.get(hname), depth)
        requested, stored, storage_mode, regrid_steps = resolve_history_storage(
            policy,
            depth,
            fill_count=fill_count,
            macro_step=int(macro_step),
            regrid_every=int(regrid_every),
        )
        slot_dt = None
        if hasattr(system, "history_slot_dt"):
            slot_dt = tuple(float(system.history_slot_dt(hname, k)) for k in range(depth))
        rings.append(HistoryRingCapture(
            name=hname,
            depth=depth,
            ncomp=ncomp,
            initialized=initialized,
            fill_count=fill_count,
            policy_json=json.dumps(
                policy.to_manifest(), sort_keys=True, separators=(",", ":")),
            requested_stored_slots=requested,
            stored_slots=stored,
            storage_mode=storage_mode,
            regrid_steps=regrid_steps,
            slot_dt=slot_dt,
        ))
    return HistoryCapturePlan(tuple(rings))


def capture_histories(system, plan, out):
    """Execute only the history gathers fixed by an agreed :class:`HistoryCapturePlan`.

    Per ring: depth / ncomp /
    initialized / authentic fill count / policy manifest / requested and effective slot arrays /
    storage mode / per-slot dt, then the resolved effective slots' global buffers. The gather is
    collective (all ranks call), like ``state_global``."""
    import numpy as np
    if not isinstance(plan, HistoryCapturePlan):
        raise TypeError("history capture requires its exact prepared plan")
    out["history_names"] = np.array([ring.name for ring in plan.rings])
    for ring in plan.rings:
        hname = ring.name
        depth = ring.depth
        out["history_depth_" + hname] = depth
        out["history_ncomp_" + hname] = ring.ncomp
        out["history_init_" + hname] = ring.initialized
        out["history_fill_count_" + hname] = ring.fill_count
        out["history_policy_" + hname] = np.array(ring.policy_json)
        out["history_requested_stored_slots_" + hname] = np.asarray(
            ring.requested_stored_slots, dtype=np.int64)
        out["history_stored_slots_" + hname] = np.asarray(
            ring.stored_slots, dtype=np.int64)
        out["history_storage_mode_" + hname] = np.array(ring.storage_mode)
        # Authenticated schedule fingerprint. A non-empty schedule resolves the effective storage mode
        # to dense_regrid_safety, because replaying historical remaps from the checkpoint layout is not
        # exact; an empty schedule permits selective replay.
        if ring.regrid_steps is not None:
            out["history_regrid_steps_" + hname] = np.asarray(
                ring.regrid_steps, dtype=np.int64)
        if ring.slot_dt is not None:
            out["history_slot_dt_" + hname] = np.asarray(
                ring.slot_dt, dtype=np.float64)
        for k in ring.stored_slots:
            out["history_%s_%d" % (hname, k)] = np.asarray(
                system.history_global(hname, k), dtype=np.float64)


def serialize_histories(system, persistence, out):
    """Compatibility helper for serial callers; collective codecs use the split protocol."""
    plan = prepare_history_capture(
        system,
        persistence,
        macro_step=int(out.get("macro_step", 0)),
        regrid_every=int(out.get("regrid_every", 0)),
    )
    capture_histories(system, plan, out)


def restore_histories(system, d, fired_out=None):
    """Restore every checkpointed ring, replaying omitted slots only on a stable hierarchy.

    The current payload is validated in full before native mutation.  All rings then restore their
    stored anchors, per-slot dt and initialized flag before any ring is replayed: a compiled Program
    may read another ring while reconstructing one gap, so replaying during the first ring's restore
    would make checkpoint key order observable.  ``rebuild_history_slots`` subsequently reconstructs
    only the omitted slots by deterministic replay. Requested/effective storage or policy mismatches
    are refused verbatim. When @p fired_out is a dict it records native replay guard evidence; every
    valid value is empty because regrid-window captures use dense safety storage. Returns the typed
    :class:`~pops.time._history.report.HistoryReplayReport`."""
    import numpy as np
    from pops.time._history.persistence import HistoryPersistence
    from pops.time._history.report import HistoryReplayReport

    names = tuple(str(h) for h in d["history_names"])
    if len(names) != len(set(names)):
        raise ValueError("restart : checkpoint history names must be unique")
    required_restore_seams = (
        "restore_history",
        "set_history_initialized",
        "restore_history_fill_count",
    )
    missing_restore_seams = tuple(
        seam for seam in required_restore_seams if not hasattr(system, seam)
    )
    if names and missing_restore_seams:
        raise RuntimeError(
            "restart : runtime lacks required history restore seam(s): %s"
            % ", ".join(missing_restore_seams)
        )

    # Phase 1a -- validate and materialize the whole payload without mutating the runtime.  Keeping
    # the staged values alive lets an np.load-backed caller retain the archive until native scatter
    # has consumed each binary64 buffer.
    prepared = []
    for hname in names:
        depth = int(d["history_depth_" + hname])
        initialized = bool(d["history_init_" + hname])
        fill_count = history_fill_count_from_payload(
            d, hname, depth, initialized
        )
        policy_key = "history_policy_" + hname
        if policy_key not in d:
            raise RuntimeError(
                "restart : history '%s' lacks its required persistence manifest" % hname)
        policy = HistoryPersistence.from_json(str(d[policy_key]))
        requested_key = "history_requested_stored_slots_" + hname
        stored_key = "history_stored_slots_" + hname
        mode_key = "history_storage_mode_" + hname
        if requested_key not in d or stored_key not in d or mode_key not in d:
            raise RuntimeError(
                "restart : history '%s' lacks its resolved storage plan" % hname)
        requested = tuple(sorted(int(s) for s in d[requested_key]))
        stored = tuple(sorted(int(s) for s in d[stored_key]))
        expected_requested, expected_stored, expected_mode, _steps = resolve_history_storage(
            policy,
            depth,
            fill_count=fill_count,
            macro_step=int(d.get("macro_step", 0)),
            regrid_every=int(d.get("regrid_every", 0)),
        )
        if list(requested) != list(expected_requested):
            raise RuntimeError(
                "restart : history '%s' checkpoint requested slots %r != policy %s expects %r"
                % (hname, list(requested), policy.name, list(expected_requested)))
        if list(stored) != list(expected_stored) or str(d[mode_key]) != expected_mode:
            raise RuntimeError(
                "restart : history '%s' resolved storage plan (%r, %s) != expected (%r, %s)"
                % (hname, list(stored), str(d[mode_key]), list(expected_stored), expected_mode))
        anchors = tuple(
            (
                k,
                np.asarray(
                    d["history_%s_%d" % (hname, k)],
                    dtype=np.float64,
                ),
            )
            for k in stored
        )
        slot_dt_key = "history_slot_dt_" + hname
        slot_dt = (
            None
            if slot_dt_key not in d
            else tuple(float(dt) for dt in np.asarray(d[slot_dt_key], dtype=np.float64))
        )
        if slot_dt is not None and len(slot_dt) != depth:
            raise ValueError(
                "restart : history '%s' slot-dt count %d != depth %d"
                % (hname, len(slot_dt), depth)
            )
        if len(stored) < depth and slot_dt is None:
            raise RuntimeError(
                "restart : history '%s' requires selective replay but lacks per-slot dt"
                % hname
            )
        if slot_dt is not None and not hasattr(system, "restore_history_slot_dt"):
            raise RuntimeError(
                "restart : history '%s' carries per-slot dt but the runtime "
                "does not expose restore_history_slot_dt" % hname
            )
        if len(stored) < depth and not hasattr(system, "rebuild_history_slots"):
            raise RuntimeError(
                "restart : history '%s' requires selective replay but the runtime "
                "does not expose rebuild_history_slots" % hname
            )
        prepared.append(
            (
                hname,
                depth,
                fill_count,
                policy,
                requested,
                stored,
                expected_mode,
                anchors,
                slot_dt,
                initialized,
            )
        )

    # Phase 1b -- publish every exact anchor and its metadata before executing the first Program
    # replay.  This is deliberately a separate pass from both validation and replay.
    for (
        hname,
        _depth,
        fill_count,
        _policy,
        _requested,
        _stored,
        _expected_mode,
        anchors,
        slot_dt,
        initialized,
    ) in prepared:
        for k, values in anchors:
            system.restore_history(hname, k, values)
        if slot_dt is not None:
            for k, dt in enumerate(slot_dt):
                system.restore_history_slot_dt(hname, k, dt)
        system.set_history_initialized(hname, initialized)
        system.restore_history_fill_count(hname, fill_count)

    # Phase 2 -- all ring dependencies now expose the checkpoint image.  Native replay restores its
    # own save bracket after each ring and returns one count per Program step, which is also the
    # number of omitted slots now that stored anchors are never stepped into.
    report = HistoryReplayReport()
    for (
        hname,
        depth,
        _fill_count,
        policy,
        requested,
        stored,
        expected_mode,
        _anchors,
        _slot_dt,
        _initialized,
    ) in prepared:
        recomputed = 0
        replay_steps = 0
        if len(stored) < depth:
            replay_steps = sum(
                stored[index + 1] - stored[index] - 1
                for index in range(len(stored) - 1)
            )
            recomputed = int(system.rebuild_history_slots(hname, list(stored)))
            if recomputed != replay_steps:
                raise RuntimeError(
                    "restart : history '%s' native replay rebuilt %d slot(s), "
                    "but its stored anchors require %d Program step(s)"
                    % (hname, recomputed, replay_steps)
                )
            if fired_out is not None and hasattr(system, "last_replay_regrid_steps"):
                fired_out[hname] = [int(s) for s in system.last_replay_regrid_steps()]
        report.add(
            name=hname,
            depth=depth,
            policy_kind=policy.kind,
            storage_mode=expected_mode,
            requested_slots=len(requested),
            stored_slots=len(stored),
            recomputed_slots=recomputed,
            replay_steps=replay_steps,
        )
    return report


__all__ = [
    "HistoryCapturePlan",
    "capture_histories",
    "history_fill_count_from_payload",
    "prepare_history_capture",
    "replay_regrid_steps",
    "resolve_history_storage",
    "resolve_ring_policy",
    "restore_histories",
    "serialize_histories",
]
