"""AMR checkpoint format v3 -- restartable under ACTIVE regridding (ADC-542 addendum B).

The v1/v2 AMR checkpoint restarts a FROZEN hierarchy only (regrid_every == 0): a post-restart regrid
would re-diverge because the restart cannot impose the mid-run hierarchy. v3 designs that away. By the
determinism theorem (addendum B.2) the regrid is a PURE function of (state, composition, macro_step):
restore the hierarchy (BoxArrays + owner-rank DistributionMappings), the full per-level per-block
state (covered cells included), the shared phi warm-start, and the clock EXACTLY, and every
post-restart regrid reproduces the uninterrupted layout sequence.

v3 is ADDITIVE (reader accepts {1, 2, 3}); v1/v2 checkpoints restore on the existing frozen path with
zero behaviour change. New keys (all additive):
  - ``pops_amr_checkpoint_version = 3``
  - ``dmap_<k>``: owner rank per box of level k, aligned with the level-k rows of ``patch_boxes``
    (bit-identity requires the box->rank map: it fixes the local-fab aggregation order).
  - ``aux_<k>``: the FULL shared aux of level k, ALL components (phi comp 0, gradients, named aux),
    flat c*nf*nf+j*nf+i. Absent on the single-block coupler path (its aux is derived +
    static-reapplied; the reader falls back to phi-only, the documented fallback semantics).
  - ``regrid_count``, ``regrid_every``: regrid metadata (report parity + the cadence guard).
  - ``program_hash``: the installed compiled-Program hash (a compiled AMR Program must restart under
    the SAME program; absent for a native composition -- the guard is skipped, like the uniform writer).
  - ``n_ranks`` + ``history_*`` (ADC-631): the rank count and the multistep history-ring payload
    (per ring: depth / ncomp / init / policy manifest / stored slots / per-slot dt + the policy-
    selected slots' flat buffers). Replaying a non-Dense ring across a rank-count change is refused.

Restore order (addendum B.6): guards -> rebuild_hierarchy (impose the mid-run hierarchy) -> per-level
state -> aux -> phi -> clock. The clock is LAST so the next step's regrid_if_due sees the
uninterrupted clock.
"""

_V3 = 3


def write_v3(sim, path, L, regrid_every, persistence=None):
    """Write a v3 AMR checkpoint (restartable under active regridding). Returns the path.

    @p sim is the C++ AmrSystem engine (``self._s``); @p L the domain length; @p regrid_every the live
    regrid cadence; @p persistence the ``name -> policy`` history-persistence map (ADC-631). COLLECTIVE
    under np>1 (the _global gather accessors run on every rank; only rank 0 writes). Multi-block keys the
    state per block; mono-block keys it by the single block name.
    """
    import os
    import numpy as np
    from pops import _pops
    from pops.runtime.bricks import abi_key

    # Controlled refusal (ADC-597 matrix standard): a block-less engine cannot describe a
    # hierarchy; refuse HERE with the route-specific reason instead of surfacing the raw engine
    # error from n_levels().
    if int(sim.n_blocks()) == 0:
        raise ValueError(
            "AmrSystem.checkpoint: no blocks installed (nothing to checkpoint); bind a compiled "
            "problem with pops.bind(...) before checkpointing")

    gather = _pops.n_ranks() != 1
    # Route state I/O on the ENGINE, not the block count: a compiled Program forces the multi-block
    # AmrRuntime engine even for ONE block, where n_vars / level_state throw and only the per-block
    # accessors work (ADC-631). uses_runtime_engine() is the exact discriminator (older _pops without it
    # -> fall back to the block-count heuristic, correct for the pre-631 coupler / >=2-block cases).
    multi = (sim.uses_runtime_engine() if hasattr(sim, "uses_runtime_engine")
             else sim.n_blocks() != 1)
    nlev = int(sim.n_levels())
    names = list(sim.block_names())
    pb = sim.patch_boxes()
    out = {"pops_amr_checkpoint_version": _V3,
           "t": sim.time(), "macro_step": sim.macro_step(),
           "n": sim.nx(), "L": L, "regrid_every": int(regrid_every),
           "abi_key": abi_key(), "blocks": np.array(names), "n_levels": nlev,
           # ADC-631: the rank count the checkpoint was written under. Replaying a NON-Dense history
           # ring re-issues collective regrids; a different n_ranks at restart would desync the
           # deterministic regrid, so restart_v3 refuses that one case (Dense rings need no replay).
           "n_ranks": int(_pops.n_ranks()),
           "patch_boxes": (np.asarray(pb, dtype=np.int64) if pb
                           else np.zeros((0, 5), dtype=np.int64))}
    # program-hash guard (m5): a compiled AMR Program must restart under the SAME program. Absent for a
    # native composition (the guard is skipped, like the uniform writer).
    phash = sim.installed_program_hash() if hasattr(sim, "installed_program_hash") else ""
    if phash:
        out["program_hash"] = str(phash)
    # per-level owner-rank DistributionMapping (m1): the box->rank map fixing the local-fab order.
    for k in range(nlev):
        ranks = list(sim.level_owner_ranks(k)) if hasattr(sim, "level_owner_ranks") else []
        out["dmap_%d" % k] = np.asarray(ranks, dtype=np.int64)
    # FULL per-level per-block conservative state (covered cells included, as-is) + shared phi.
    if multi:
        for b in names:
            out["n_vars_%s" % b] = int(sim.block_n_vars(b))
            for k in range(nlev):
                out["state_%s_%d" % (b, k)] = np.asarray(
                    sim.block_level_state_global(b, k) if gather else sim.block_level_state(b, k),
                    dtype=np.float64)
    else:
        b = names[0] if names else ""
        out["n_vars_%s" % b] = int(sim.n_vars())
        for k in range(nlev):
            out["state_%s_%d" % (b, k)] = np.asarray(
                sim.level_state_global(k) if gather else sim.level_state(k), dtype=np.float64)
    for k in range(nlev):
        out["phi_%d" % k] = np.asarray(
            sim.level_potential_global(k) if gather else sim.level_potential(k), dtype=np.float64)
    # FULL shared aux per level (m2): ALL aux components (phi comp 0 + gradients + named aux), flat
    # c*nf*nf+j*nf+i. EMPTY on the single-block coupler path (its aux is derived + static-reapplied;
    # phi_<k> suffices there) -- the key is then skipped and the reader falls back to phi-only.
    for k in range(nlev):
        aux = np.asarray(sim.level_aux_flat_global(k) if gather else sim.level_aux_flat(k),
                         dtype=np.float64)
        if aux.size:
            out["aux_%d" % k] = aux
    # ADC-631: multistep history rings (keep_history / T.prev). serialize_histories stores only the
    # policy-selected slots + the per-slot dt (a recomputed slot is replayed at restart); the per-level
    # slices are hidden inside AmrSystem.history_global's flat concat, so the SHARED writer is reused
    # verbatim. No rings (no keep_history / an engine-less coupler) -> history_names() is empty (no-op).
    if sim.history_names():
        from pops.runtime._system_io_history import serialize_histories
        serialize_histories(sim, persistence or {}, out)

    target = path if path.endswith(".npz") else path + ".npz"
    if _pops.my_rank() != 0:
        return target  # only rank 0 writes (the gather is already done on every rank)
    tmp = target + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **out)
    os.replace(tmp, target)
    return target


def restart_v3(sim, d, L):
    """Restore a v3 AMR checkpoint into @p sim (the C++ AmrSystem engine). @p d is the loaded npz.

    The restore order (addendum B.6, the realized subset): guards (grid / blocks / components / regrid
    metadata / program-hash) -> rebuild_hierarchy (impose the mid-run hierarchy from the manifest) ->
    per-level per-block state (as-saved, no re-prolongation) -> shared phi warm-start -> multistep
    history rings (ADC-631: restore stored slots + replay recomputed gaps) -> clock LAST (so the next
    step's regrid_if_due sees the uninterrupted clock). Returns the HistoryReplayReport (or None).
    """
    import numpy as np

    # (2) GUARDS.
    if int(d["n"]) != sim.nx():
        raise ValueError("restart : checkpoint grid (n=%d) != system (n=%d)"
                         % (int(d["n"]), sim.nx()))
    if float(d["L"]) != L:
        raise ValueError("restart : checkpoint domain (L=%r) != system (L=%r) -- different dx"
                         % (float(d["L"]), L))
    chk_blocks = [str(b) for b in d["blocks"]]
    cur_blocks = list(sim.block_names())
    if chk_blocks != cur_blocks:
        raise ValueError("restart : checkpoint blocks %r != current composition %r "
                         "(replay the SAME composition before restart)" % (chk_blocks, cur_blocks))
    nlev = int(d["n_levels"])
    if nlev != int(sim.n_levels()):
        raise ValueError("restart : %d levels in the checkpoint, %d here (composition / refinement "
                         "differ?)" % (nlev, int(sim.n_levels())))
    # program-hash guard (m5): a v3 checkpoint of a compiled AMR Program refuses a DIFFERENT program.
    chk_hash = str(d["program_hash"]) if "program_hash" in d else ""
    cur_hash = sim.installed_program_hash() if hasattr(sim, "installed_program_hash") else ""
    if chk_hash and cur_hash and chk_hash != cur_hash:
        raise ValueError(
            "restart : checkpoint program hash %r != installed program hash %r (a different compiled "
            "AMR Program cannot restart this checkpoint)" % (chk_hash, cur_hash))
    # Route on the ENGINE (see write_v3): a compiled Program forces the runtime engine for ONE block too,
    # where only the per-block accessors + rebuild_hierarchy work.
    multi = (sim.uses_runtime_engine() if hasattr(sim, "uses_runtime_engine")
             else sim.n_blocks() != 1)

    # (3) HIERARCHY REBUILD: impose the mid-run hierarchy from the manifest (multi-block runtime path).
    # The boxes carry their level; the dmaps give the owner rank per box (aligned by level order).
    boxes = [tuple(int(x) for x in row) for row in np.asarray(d["patch_boxes"], dtype=np.int64)]
    if multi:
        owner_ranks = _owner_ranks_for_boxes(d, boxes, nlev)
        sim.rebuild_hierarchy(boxes, owner_ranks)
    elif nlev >= 2:
        # single-block coupler path: impose the fine hierarchy (level 1) as v1/v2 does.
        fine = [b for b in boxes if b[0] == 1]
        if not fine:
            raise ValueError("restart : %d-level hierarchy but no fine patch (level 1) in the "
                             "checkpoint (inconsistent)." % nlev)
        sim.set_hierarchy(boxes)

    # (4) PER-LEVEL PER-BLOCK STATE (as-saved, no re-prolongation).
    for b in cur_blocks:
        cur_nv = int(sim.block_n_vars(b)) if multi else int(sim.n_vars())
        chk_nv = int(d["n_vars_%s" % b])
        if chk_nv != cur_nv:
            raise ValueError("restart : block '%s' has %d components in the checkpoint, %d here"
                             % (b, chk_nv, cur_nv))
        for k in range(nlev):
            st = np.asarray(d["state_%s_%d" % (b, k)], dtype=np.float64)
            if multi:
                sim.set_block_level_state(b, k, st)
            else:
                sim.set_level_state(k, st)

    # (5) FULL SHARED AUX per level when the checkpoint carries it (m2; the reader PREFERS aux_<k> and
    # falls back to phi-only when absent -- the single-block coupler path), then the phi warm-start
    # (separate storage: the level-0 phi is the multigrid warm start mg_.phi(), not aux comp 0).
    for k in range(nlev):
        key = "aux_%d" % k
        if key in d:
            sim.set_level_aux_flat(k, np.asarray(d[key], dtype=np.float64).ravel())
    for k in range(nlev):
        sim.set_level_potential(k, np.asarray(d["phi_%d" % k], dtype=np.float64).ravel())

    # (6) MULTISTEP HISTORY RINGS (ADC-631): restore the policy-stored slots + per-slot dt, then replay
    # the recomputed gaps (restore_histories drives rebuild_history_slots, which re-steps the installed
    # Program with regrid frozen). Runs AFTER state/aux/phi (the replay seeds from a ring slot and is
    # SAVE/RESTORE bracketed) and BEFORE the clock. The SHARED restore_histories is reused verbatim.
    report = _restore_histories_v3(sim, d)

    # (8) CLOCK LAST (macro_step advances the regrid cadence phase).
    sim.set_clock(float(d["t"]), int(d["macro_step"]))
    return report


def _restore_histories_v3(sim, d):
    """Restore + replay the v3 history rings, refusing the two genuinely new AMR impossible cases.

    (1) RANK-COUNT CHANGE: replaying a NON-Dense ring re-steps the installed Program whose regrids are
    collective, so a different rank count from the checkpoint would desync the deterministic regrid --
    refuse LOUD (Dense rings need no replay -> they restart across any np).
    (2) REGRID BETWEEN THE SEED ANCHOR'S ERA AND THE CHECKPOINT: ring slots are stored REMAPPED onto
    the checkpoint hierarchy. A head-of-step regrid due at any original macro-step from the seed
    anchor's re-step on breaks bit-exactness two ways: a regrid INSIDE the replayed steps is one the
    frozen re-step skips (and its remap destroyed the pre-regrid fine data an exact re-step needs), and
    a regrid AFTER them remapped the stored targets while the replay steps the REMAPPED seed (step and
    remap do not commute). Both are unreconstructable from the checkpoint -- refuse LOUD. Outside these
    two cases the replay re-executes the ORIGINAL regrid-free step sequence on the checkpoint hierarchy
    (the engine freezes the regrid cadence for the internal re-step; the original steps of a clean
    window saw the SAME empty regrid schedule), so the reconstruction is bit-exact.
    Returns the typed HistoryReplayReport, or ``None`` when the checkpoint has no rings.
    """
    if "history_names" not in d or not len(list(d["history_names"])):
        return None
    from pops import _pops
    from pops.runtime._system_io_history import restore_histories
    chk_ranks = int(d["n_ranks"]) if "n_ranks" in d else 1
    cur_ranks = int(_pops.n_ranks())
    m = int(d["macro_step"])
    regrid_every = int(d["regrid_every"]) if "regrid_every" in d else 0
    for hname in (str(h) for h in d["history_names"]):
        depth = int(d["history_depth_" + hname])
        key = "history_stored_slots_" + hname
        stored = sorted(int(s) for s in d[key]) if key in d else list(range(depth))
        if len(stored) >= depth:
            continue  # Dense (every slot stored): no replay -> neither refusal applies.
        if chk_ranks != cur_ranks:
            raise ValueError(
                "restart : history '%s' uses a non-Dense persistence policy that must REPLAY the "
                "installed Program to reconstruct its slots, but the checkpoint was written under "
                "%d rank(s) and this restart uses %d; the deterministic regrid would desync across "
                "the rank-count change. Restart under %d rank(s), or checkpoint the ring with "
                "Dense() (Dense needs no replay and restarts across any np)."
                % (hname, chk_ranks, cur_ranks, chk_ranks))
        if regrid_every > 0:
            # The replay of gap (newer=a, older=b) re-executes the original macro-steps from the seed
            # anchor's era (step m-b) forward; ANY head-of-step regrid due at s in [m-b, m-1] breaks
            # bit-exactness (skipped-regrid re-step, or a stored-value remap the re-step cannot
            # commute with). Due when s > 0 and s % regrid_every == 0.
            for a, b in zip(stored, stored[1:]):
                if b - a < 2:
                    continue  # adjacent anchors: no missing slot in this gap
                for s in range(m - b, m):
                    if s > 0 and s % regrid_every == 0:
                        raise ValueError(
                            "restart : history '%s' cannot be replayed: a regrid was due at "
                            "macro-step %d, between the era of stored slot %d and the checkpoint "
                            "(macro-step %d, regrid_every=%d); the regrid remap makes the "
                            "recomputed slots of gap %d..%d unreconstructable bit-exactly. "
                            "Checkpoint the ring with Dense(), or checkpoint at a step whose "
                            "replay windows cross no regrid boundary."
                            % (hname, s, b, m, regrid_every, a, b))
    return restore_histories(sim, d, ckpt_version=3)


def _owner_ranks_for_boxes(d, boxes, nlev):
    """The per-box owner rank aligned with @p boxes, read from the per-level ``dmap_<k>`` arrays.

    ``patch_boxes`` lists boxes grouped by level (level 1, then 2, ...); ``dmap_<k>`` is the owner-rank
    array of level k aligned with that level's box order. Level 0 is implicit in patch_boxes (only fine
    levels appear), so the mapping walks the fine-level boxes in order.
    """
    import numpy as np
    per_level_ranks = {k: list(np.asarray(d["dmap_%d" % k], dtype=np.int64)) for k in range(nlev)
                       if ("dmap_%d" % k) in d}
    cursor = {k: 0 for k in range(nlev)}
    owners = []
    for (lvl, _ilo, _jlo, _ihi, _jhi) in boxes:
        ranks = per_level_ranks.get(lvl, [])
        idx = cursor.get(lvl, 0)
        owners.append(int(ranks[idx]) if idx < len(ranks) else 0)
        cursor[lvl] = idx + 1
    return owners


__all__ = ["write_v3", "restart_v3"]
