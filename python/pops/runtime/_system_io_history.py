"""History-ring checkpoint serialization + selective-persistence restart (ADC-626).

Split out of :mod:`pops.runtime._system_io` (the 500-line cap): the checkpoint WRITER emits, per
history ring, the authoring policy and one resolved physical storage plan. Selective slots are used
only on a stable hierarchy window; a window containing a scheduled regrid is explicitly promoted to
``dense_regrid_safety``. Restart replays only gaps that are exact on the stable hierarchy.
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


def resolve_history_storage(policy, depth, *, macro_step, regrid_every):
    """Resolve authoring intent to one exact, restart-safe physical storage plan.

    Selective replay is exact only while the replay window retains one hierarchy.  If a regrid is
    scheduled inside that window, the resolved plan explicitly promotes this checkpoint instance to
    dense storage.  The requested slots, effective slots, mode and schedule fingerprint are all
    persisted, so this safety promotion is observable and authenticated rather than a silent fallback.
    """
    policy = resolve_ring_policy(policy, depth)
    requested = tuple(int(slot) for slot in policy.stored_slots(depth))
    regrid_steps = None
    if len(requested) < depth:
        regrid_steps = tuple(replay_regrid_steps(
            depth, int(macro_step), int(regrid_every)))
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
        policy = resolve_ring_policy(persistence.get(hname), depth)
        requested, stored, storage_mode, regrid_steps = resolve_history_storage(
            policy,
            depth,
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
    initialized / policy manifest / requested and effective slot arrays / storage mode / per-slot dt,
    then the resolved effective slots' global buffers. The gather is collective (all ranks call), like
    ``state_global``."""
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

    The current payload restores the stored slots + per-slot dt, then
    ``rebuild_history_slots`` reconstructs any gaps by deterministic
    replay. Requested/effective storage or policy mismatches are refused verbatim. When @p fired_out is
    a dict it records native replay guard evidence; every valid value is empty because regrid-window
    captures use dense safety storage. Returns the typed
    :class:`~pops.time._history.report.HistoryReplayReport`."""
    import numpy as np
    from pops.time._history.persistence import HistoryPersistence
    from pops.time._history.report import HistoryReplayReport
    report = HistoryReplayReport()
    for hname in (str(h) for h in d["history_names"]):
        depth = int(d["history_depth_" + hname])
        policy_key = "history_policy_" + hname
        if policy_key not in d:
            raise RuntimeError(
                "restart : history '%s' lacks its required persistence manifest" % hname)
        policy = HistoryPersistence.from_json(str(d[policy_key]))
        requested_key = "history_requested_stored_slots_" + hname
        mode_key = "history_storage_mode_" + hname
        if requested_key not in d or mode_key not in d:
            raise RuntimeError(
                "restart : history '%s' lacks its resolved storage plan" % hname)
        requested = [int(s) for s in d[requested_key]]
        stored = [int(s) for s in d["history_stored_slots_" + hname]]
        expected_requested, expected_stored, expected_mode, _steps = resolve_history_storage(
            policy,
            depth,
            macro_step=int(d.get("macro_step", 0)),
            regrid_every=int(d.get("regrid_every", 0)),
        )
        if sorted(requested) != list(expected_requested):
            raise RuntimeError(
                "restart : history '%s' checkpoint requested slots %r != policy %s expects %r"
                % (hname, sorted(requested), policy.name, list(expected_requested)))
        if sorted(stored) != list(expected_stored) or str(d[mode_key]) != expected_mode:
            raise RuntimeError(
                "restart : history '%s' resolved storage plan (%r, %s) != expected (%r, %s)"
                % (hname, sorted(stored), str(d[mode_key]), list(expected_stored), expected_mode))
        for k in stored:
            system.restore_history(
                hname, k, np.asarray(d["history_%s_%d" % (hname, k)], dtype=np.float64))
        if hasattr(system, "restore_history_slot_dt") and ("history_slot_dt_" + hname) in d:
            for k, dt in enumerate(np.asarray(d["history_slot_dt_" + hname], dtype=np.float64)):
                system.restore_history_slot_dt(hname, int(k), float(dt))
        system.set_history_initialized(hname, bool(d["history_init_" + hname]))
        recomputed = 0
        if len(stored) < depth and hasattr(system, "rebuild_history_slots"):
            recomputed = int(system.rebuild_history_slots(hname, sorted(stored)))
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
            replay_steps=recomputed,
        )
    return report


__all__ = [
    "HistoryCapturePlan",
    "capture_histories",
    "prepare_history_capture",
    "replay_regrid_steps",
    "resolve_history_storage",
    "resolve_ring_policy",
    "restore_histories",
    "serialize_histories",
]
