"""History-ring checkpoint serialization + selective-persistence restart (ADC-626).

Split out of :mod:`pops.runtime._system_io` (the 500-line cap): the checkpoint WRITER emits, per
history ring, only the policy-selected slots plus the policy manifest and the per-slot dt; the
RESTART reader restores the stored slots and REPLAYS the recomputed gaps via the native
``rebuild_history_slots`` seam. A Dense policy stores every slot and never replays.
"""
import json


def resolve_ring_policy(policy, depth):
    """Validate the explicitly installed history-persistence policy for one ring."""
    from pops.time._history.persistence import HistoryPersistence
    if not isinstance(policy, HistoryPersistence):
        raise RuntimeError(
            "checkpoint history ring has no explicit compiled persistence policy")
    policy.validate_for(depth)
    return policy


def replay_regrid_steps(depth, m, regrid_every):
    """The macro-step cursors at which the ADC-635 replay of a depth-@p depth ring fires an in-window
    regrid.

    A pure function of the ring depth, the checkpoint macro-step @p m and the cadence @p regrid_every,
    mirroring ``detail::AmrHistoryOps::expected_regrid_steps`` bit-for-bit. The replay is ONE continuous
    forward sweep from the oldest anchor (slot depth-1) to slot 0; the re-step producing slot j runs at
    cursor ``m-1-j`` (the original step that landed the ring on macro-step m-j ran with
    ``ctx.macro_step()==m-j-1``, pre-increment), and a regrid is due when that cursor is > 0 and divisible
    by @p regrid_every. The WRITE-time fingerprint the v3 reader asserts against what the replay actually
    fires. Empty when @p regrid_every is 0 (Uniform / a frozen hierarchy) or the ring has depth < 2."""
    steps = set()
    if regrid_every and regrid_every > 0 and depth >= 2:
        for j in range(depth - 2, -1, -1):
            cursor = m - 1 - j
            if cursor > 0 and cursor % regrid_every == 0:
                steps.add(cursor)
    return sorted(steps)


def serialize_histories(system, persistence, out):
    """Write every registered ring of @p system into the checkpoint dict @p out (ADC-626).

    @p persistence is the complete compiled ``name -> policy`` map. Per ring: depth / ncomp /
    initialized / the policy manifest / the stored-slot index array / the per-slot dt, then ONLY the
    policy-selected slots' global buffers (a recomputed slot is replayed at restart, not stored). The
    gather is collective (all ranks call), like state_global."""
    import numpy as np
    # ADC-635: the checkpoint macro-step + regrid cadence drive the replay's in-window regrid fingerprint.
    # write_v3 has already put both in @p out; a Uniform checkpoint has no regrid_every (0 -> no fingerprint).
    m = int(out.get("macro_step", 0))
    regrid_every = int(out.get("regrid_every", 0))
    names = list(system.history_names())
    out["history_names"] = np.array(names)
    for hname in names:
        depth = int(system.history_depth(hname))
        out["history_depth_" + hname] = depth
        out["history_ncomp_" + hname] = int(system.history_ncomp(hname))
        out["history_init_" + hname] = bool(system.history_initialized(hname))
        policy = resolve_ring_policy(persistence.get(hname), depth)
        stored = list(policy.stored_slots(depth))
        out["history_policy_" + hname] = np.array(json.dumps(policy.to_manifest()))
        out["history_stored_slots_" + hname] = np.asarray(stored, dtype=np.int64)
        # ADC-635: the in-window regrid schedule the restart replay must reproduce (a schedule
        # fingerprint, NOT a stored layout -- the layouts are reproduced by determinism). Written per
        # NON-Dense ring under active regridding; the v3 reader asserts the replay fired exactly it.
        if len(stored) < depth:
            out["history_regrid_steps_" + hname] = np.asarray(
                replay_regrid_steps(depth, m, regrid_every), dtype=np.int64)
        if hasattr(system, "history_slot_dt"):
            out["history_slot_dt_" + hname] = np.asarray(
                [float(system.history_slot_dt(hname, k)) for k in range(depth)], dtype=np.float64)
        for k in stored:
            out["history_%s_%d" % (hname, k)] = np.asarray(
                system.history_global(hname, k), dtype=np.float64)


def restore_histories(system, d, fired_out=None):
    """Restore every checkpointed ring of @p system, replaying the recomputed slots (ADC-626).

    The current payload restores the stored slots + per-slot dt, then
    ``rebuild_history_slots`` reconstructs any gaps by deterministic
    replay. A stored-slots / policy mismatch is refused verbatim. When @p fired_out is a dict it is
    populated ``name -> the in-window regrid steps the replay fired`` (ADC-635; the AMR reader asserts it
    against the checkpoint fingerprint). Returns the typed
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
        stored = [int(s) for s in d["history_stored_slots_" + hname]]
        expected = list(policy.stored_slots(depth))
        if sorted(stored) != expected:
            raise RuntimeError(
                "restart : history '%s' checkpoint stored slots %r != policy %s expects %r"
                % (hname, sorted(stored), policy.name, expected))
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
        report.add(name=hname, depth=depth, policy_kind=policy.kind,
                   stored_slots=len(stored), recomputed_slots=recomputed, replay_steps=recomputed)
    return report


__all__ = ["resolve_ring_policy", "replay_regrid_steps", "serialize_histories", "restore_histories"]
