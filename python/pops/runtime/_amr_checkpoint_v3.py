"""Strict v3 AMR checkpoint: exact topology, fields, histories and accepted Program state.

The sealed payload preserves owner-rank mappings, all block/level state, aux and elliptic warm starts,
regrid counters, qualified history rings, rational clocks, lagged flux publications, level relations
and transfer-plan provenance. Restore is transactional and the public route has one guarantee only:
bit-identical continuation under the same bound composition. Historical or weaker fallback formats
are refused.
"""

from pops._generated_release_contract import AMR_CHECKPOINT_PAYLOAD_VERSION as _V3


def write_v3(owner, sim, path, L, regrid_every, persistence=None):
    """Write a v3 AMR checkpoint (restartable under active regridding). Returns the path.

    @p sim is the C++ AmrSystem engine (``self._s``); @p L the domain length; @p regrid_every the live
    regrid cadence; @p persistence the ``name -> policy`` history-persistence map (ADC-631). COLLECTIVE
    under np>1 (the _global gather accessors run on every rank; only rank 0 writes). Multi-block keys the
    state per block; mono-block keys it by the single block name.
    """
    import os
    import numpy as np
    from pops import _pops
    from pops.runtime._engine_descriptors import abi_key

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
    temporal = getattr(owner, "_temporal_restart_state", None)
    if temporal is None:
        raise RuntimeError("checkpoint requires the AMR temporal restart state")
    out["temporal_restart_state"] = np.array(temporal.checkpoint_json(
        time=sim.time(), macro_step=sim.macro_step()))
    from pops.runtime._amr_checkpoint_contract import encode_contract
    out["regrid_count"] = int(sim.checkpoint_regrid_count())
    out["topology_epoch"] = int(sim.checkpoint_topology_epoch())
    out["program_accepted_state"] = np.frombuffer(
        sim.program_accepted_state(), dtype=np.uint8).copy()
    out["amr_accepted_contract"] = encode_contract(sim)
    # program-hash guard (m5): a compiled AMR Program must restart under the SAME program. Absent for a
    # native composition (the guard is skipped, like the uniform writer).
    phash = sim.installed_program_hash() if hasattr(sim, "installed_program_hash") else ""
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
    # Qualified field-provider warm starts.  Slots are ordered native registry keys; each carries its
    # complete canonical identity in the sealed BoundSnapshot, so restart validates composition before
    # mutating a solver.  Composite fields retain every native level, level-local fields retain coarse.
    field_slots = list(sim.field_provider_slots()) if hasattr(sim, "field_provider_slots") else []
    out["field_provider_slots"] = np.asarray(field_slots)
    for index, slot in enumerate(field_slots):
        levels = int(sim.field_provider_levels(slot))
        out["field_provider_levels_%d" % index] = levels
        for k in range(levels):
            out["field_provider_phi_%d_%d" % (index, k)] = np.asarray(
                sim.field_potential_level_global(slot, k), dtype=np.float64)
    # FULL shared aux per level (m2): ALL aux components (phi comp 0 + gradients + named aux), flat
    # c*nf*nf+j*nf+i. An engine with no shared aux records an explicit empty array.
    for k in range(nlev):
        aux = np.asarray(sim.level_aux_flat_global(k) if gather else sim.level_aux_flat(k),
                         dtype=np.float64)
        out["aux_%d" % k] = aux
    # ADC-631: multistep history rings (keep_history / T.prev). serialize_histories stores only the
    # policy-selected slots + the per-slot dt (a recomputed slot is replayed at restart); the per-level
    # slices are hidden inside AmrSystem.history_global's flat concat, so the SHARED writer is reused
    # verbatim. No rings (no keep_history / an engine-less coupler) -> history_names() is empty (no-op).
    from pops.runtime._system_io_history import serialize_histories
    serialize_histories(sim, persistence or {}, out)

    from pops.runtime._checkpoint_manifest import seal_checkpoint_payload
    seal_checkpoint_payload(owner, out, runtime_kind="amr")
    target = path if path.endswith(".npz") else path + ".npz"
    if _pops.my_rank() != 0:
        return target  # only rank 0 writes (the gather is already done on every rank)
    tmp = target + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **out)
    os.replace(tmp, target)
    return target


def restart_v3(owner, sim, d, L):
    """Restore a v3 AMR checkpoint into @p sim (the C++ AmrSystem engine). @p d is the loaded npz.

    The restore order (addendum B.6, the realized subset): guards (grid / blocks / components / regrid
    metadata / program-hash) -> rebuild_hierarchy (impose the mid-run hierarchy from the manifest) ->
    per-level per-block state (as-saved, no re-prolongation) -> shared phi warm-start -> multistep
    history rings (ADC-631: restore stored slots + replay recomputed gaps) -> clock LAST (so the next
    step's regrid_if_due sees the uninterrupted clock). Returns the HistoryReplayReport (or None).
    """
    import numpy as np
    from pops import _pops
    from pops.runtime._amr_checkpoint_contract import preflight_contract
    from pops.runtime._temporal_restart import TemporalRestartState

    program_state, regrid_count, topology_epoch = preflight_contract(sim, d)
    if "temporal_restart_state" not in d:
        raise ValueError("restart: AMR checkpoint lacks its strict temporal state")
    installed_schedule = getattr(
        getattr(owner, "_temporal_restart_state", None), "program_schedule", None)
    restored_temporal = TemporalRestartState.from_json(
        d["temporal_restart_state"], time=d["t"], macro_step=d["macro_step"],
        program_schedule=installed_schedule)

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
    chk_hash = str(d["program_hash"])
    cur_hash = sim.installed_program_hash() if hasattr(sim, "installed_program_hash") else ""
    if chk_hash != cur_hash:
        raise ValueError(
            "restart : checkpoint program hash %r != installed program hash %r (a different compiled "
            "AMR Program cannot restart this checkpoint)" % (chk_hash, cur_hash))
    # Route on the ENGINE (see write_v3): a compiled Program forces the runtime engine for ONE block too,
    # where only the per-block accessors + rebuild_hierarchy work.
    multi = (sim.uses_runtime_engine() if hasattr(sim, "uses_runtime_engine")
             else sim.n_blocks() != 1)

    # Validate the complete qualified provider registry and every persisted level before hierarchy or
    # state mutation.  The checkpoint manifest has already authenticated the payload; this check binds
    # it to the exact live native registry and prevents partial field restoration.
    checkpoint_slots = ([str(slot) for slot in d["field_provider_slots"]]
                        if "field_provider_slots" in d else [])
    current_slots = (list(sim.field_provider_slots())
                     if hasattr(sim, "field_provider_slots") else [])
    if checkpoint_slots != current_slots:
        raise ValueError(
            "restart : checkpoint qualified field providers %r != installed providers %r"
            % (checkpoint_slots, current_slots))
    field_payload = []
    for index, slot in enumerate(checkpoint_slots):
        levels_key = "field_provider_levels_%d" % index
        if levels_key not in d:
            raise ValueError("restart: checkpoint lacks field provider level count for %s" % slot)
        checkpoint_levels = int(d[levels_key])
        current_levels = int(sim.field_provider_levels(slot))
        if checkpoint_levels != current_levels:
            raise ValueError(
                "restart: field provider %s has %d checkpoint levels, %d installed levels"
                % (slot, checkpoint_levels, current_levels))
        values = []
        for k in range(checkpoint_levels):
            key = "field_provider_phi_%d_%d" % (index, k)
            if key not in d:
                raise ValueError(
                    "restart: checkpoint lacks level %d potential for field provider %s"
                    % (k, slot))
            width = int(sim.nx()) << k
            value = np.asarray(d[key], dtype=np.float64).ravel()
            if value.size != width * width:
                raise ValueError(
                    "restart: field provider %s level %d potential has size %d, expected %d"
                    % (slot, k, value.size, width * width))
            values.append(value)
        field_payload.append((slot, values))

    # Preflight the complete topology and every dense native payload before the transaction starts.
    # The manifest seal authenticates bytes; these guards prove that all writes are shape-compatible
    # with the live composition, so malformed state/aux/history cannot fail only after a hierarchy
    # mutation.  The native transaction remains the final exception-safety boundary.
    raw_boxes = np.asarray(d["patch_boxes"], dtype=np.int64)
    if raw_boxes.ndim != 2 or raw_boxes.shape[1] != 5:
        raise ValueError("restart: patch_boxes must have shape (npatches, 5)")
    boxes = [tuple(int(x) for x in row) for row in raw_boxes]
    per_level_boxes = {k: [] for k in range(nlev)}
    for box in boxes:
        level, ilo, jlo, ihi, jhi = box
        if level <= 0 or level >= nlev:
            raise ValueError(
                "restart: fine patch level %d is outside [1, %d]" % (level, nlev - 1))
        width = int(sim.nx()) << level
        if ilo < 0 or jlo < 0 or ihi < ilo or jhi < jlo or ihi >= width or jhi >= width:
            raise ValueError("restart: invalid level-%d patch box %r for width %d"
                             % (level, box[1:], width))
        for other in per_level_boxes[level]:
            if not (ihi < other[0] or other[2] < ilo or
                    jhi < other[1] or other[3] < jlo):
                raise ValueError(
                    "restart: overlapping level-%d patch boxes %r and %r"
                    % (level, other, box[1:]))
        per_level_boxes[level].append((ilo, jlo, ihi, jhi))
    if nlev > 1:
        for level in range(1, nlev):
            if not per_level_boxes[level]:
                raise ValueError(
                    "restart: %d-level hierarchy has no patch at fine level %d" % (nlev, level))

    owner_ranks = []
    if multi:
        from pops.runtime._amr_checkpoint_topology import owner_ranks_for_boxes
        owner_ranks = owner_ranks_for_boxes(d, boxes, nlev)
        nranks = int(_pops.n_ranks())
        for level in range(1, nlev):
            key = "dmap_%d" % level
            if key not in d:
                raise ValueError("restart: checkpoint lacks owner-rank map for AMR level %d" % level)
            ranks = np.asarray(d[key], dtype=np.int64).ravel()
            if ranks.size != len(per_level_boxes[level]):
                raise ValueError(
                    "restart: owner-rank map for level %d has %d entries, expected %d"
                    % (level, ranks.size, len(per_level_boxes[level])))
            if any(int(rank) < 0 or int(rank) >= nranks for rank in ranks):
                raise ValueError(
                    "restart: owner-rank map for level %d contains a rank outside [0, %d)"
                    % (level, nranks))

    state_payload = []
    for block in cur_blocks:
        nvars_key = "n_vars_%s" % block
        if nvars_key not in d:
            raise ValueError("restart: checkpoint lacks component count for block '%s'" % block)
        current_nvars = int(sim.block_n_vars(block)) if multi else int(sim.n_vars())
        checkpoint_nvars = int(d[nvars_key])
        if checkpoint_nvars != current_nvars:
            raise ValueError("restart : block '%s' has %d components in the checkpoint, %d here"
                             % (block, checkpoint_nvars, current_nvars))
        levels = []
        for level in range(nlev):
            key = "state_%s_%d" % (block, level)
            if key not in d:
                raise ValueError("restart: checkpoint lacks state for block '%s' level %d"
                                 % (block, level))
            width = int(sim.nx()) << level
            state = np.asarray(d[key], dtype=np.float64)
            expected = current_nvars * width * width
            if state.size != expected:
                raise ValueError(
                    "restart: block '%s' level %d state has size %d, expected %d"
                    % (block, level, state.size, expected))
            levels.append(state)
        state_payload.append((block, levels))

    aux_payload = []
    phi_payload = []
    for level in range(nlev):
        aux_key = "aux_%d" % level
        phi_key = "phi_%d" % level
        if aux_key not in d or phi_key not in d:
            raise ValueError("restart: checkpoint lacks aux or potential payload for level %d" % level)
        aux = np.asarray(d[aux_key], dtype=np.float64).ravel()
        expected_aux = len(sim.level_aux_flat(level))
        if aux.size != expected_aux:
            raise ValueError("restart: level %d aux has size %d, expected %d"
                             % (level, aux.size, expected_aux))
        width = int(sim.nx()) << level
        phi = np.asarray(d[phi_key], dtype=np.float64).ravel()
        if phi.size != width * width:
            raise ValueError("restart: level %d potential has size %d, expected %d"
                             % (level, phi.size, width * width))
        aux_payload.append(aux)
        phi_payload.append(phi)

    _preflight_histories_v3(sim, d)

    sim.begin_restart_transaction()
    try:
        # (3) Impose the exact recorded hierarchy.
        if multi:
            sim.rebuild_hierarchy(boxes, owner_ranks)
        elif nlev >= 2:
            sim.set_hierarchy(boxes)

        # (4) Restore every block/level state as saved, without re-prolongation.
        for block, levels in state_payload:
            for level, state in enumerate(levels):
                if multi:
                    sim.set_block_level_state(block, level, state)
                else:
                    sim.set_level_state(level, state)

        # (5) Restore shared aux only on the runtime route (the coupler deliberately persists an
        # explicit empty aux payload), then all elliptic warm starts.
        for level, aux in enumerate(aux_payload):
            if aux.size:
                sim.set_level_aux_flat(level, aux)
        for level, phi in enumerate(phi_payload):
            sim.set_level_potential(level, phi)
        for slot, levels in field_payload:
            for level, value in enumerate(levels):
                sim.set_field_potential_level(slot, level, value)

        # (6) Histories may replay the Program and regrid; they are inside the same native transaction.
        report = _restore_histories_v3(sim, d)

        # (7) Replay is allowed to mutate Program clocks/ring publications and regrid counters while it
        # reconstructs policy-omitted dense values. Replace those temporary values with the checkpoint's
        # exact accepted semantic state before exposing the runtime again.
        sim.restore_program_accepted_state(program_state)
        sim.restore_checkpoint_counters(regrid_count, topology_epoch)

        # (8) Clock last: the next cadence decision is identical to the uninterrupted run.
        sim.set_clock(float(d["t"]), int(d["macro_step"]))
    except BaseException as original:
        try:
            sim.rollback_restart_transaction()
        except BaseException as rollback_error:
            raise RuntimeError(
                "restart failed and native accepted-state rollback also failed: %s" % original
            ) from rollback_error
        raise
    else:
        sim.commit_restart_transaction()
    owner._temporal_restart_state = restored_temporal
    owner._step_controller = None
    return report


def _preflight_histories_v3(sim, d):
    """Validate the entire ring registry and persisted buffers without mutating native state."""
    import numpy as np
    from pops import _pops
    from pops.time.history_persistence import HistoryPersistence

    checkpoint_names = ([str(name) for name in d["history_names"]]
                        if "history_names" in d else [])
    current_names = list(sim.history_names())
    if checkpoint_names != current_names:
        raise ValueError("restart: checkpoint history rings %r != installed rings %r"
                         % (checkpoint_names, current_names))
    checkpoint_ranks = int(d["n_ranks"]) if "n_ranks" in d else 1
    current_ranks = int(_pops.n_ranks())
    for name in checkpoint_names:
        required = ["history_depth_" + name, "history_ncomp_" + name,
                    "history_init_" + name, "history_policy_" + name,
                    "history_stored_slots_" + name]
        missing = [key for key in required if key not in d]
        if missing:
            raise ValueError("restart: history '%s' lacks keys %r" % (name, missing))
        depth = int(d["history_depth_" + name])
        ncomp = int(d["history_ncomp_" + name])
        if depth != int(sim.history_depth(name)) or ncomp != int(sim.history_ncomp(name)):
            raise ValueError(
                "restart: history '%s' shape (%d, %d) != installed shape (%d, %d)"
                % (name, depth, ncomp, int(sim.history_depth(name)),
                   int(sim.history_ncomp(name))))
        policy = HistoryPersistence.from_json(str(d["history_policy_" + name]))
        stored = sorted(int(slot) for slot in d["history_stored_slots_" + name])
        expected = list(policy.stored_slots(depth))
        if stored != expected:
            raise ValueError("restart: history '%s' stored slots %r != policy %s expects %r"
                             % (name, stored, policy.name, expected))
        if len(stored) < depth:
            if checkpoint_ranks != current_ranks:
                raise ValueError(
                    "restart: non-Dense history '%s' was written under %d rank(s), current run has %d"
                    % (name, checkpoint_ranks, current_ranks))
            if "history_regrid_steps_" + name not in d:
                raise ValueError("restart: history '%s' lacks its regrid replay fingerprint" % name)
        expected_values = None
        for slot in stored:
            key = "history_%s_%d" % (name, slot)
            if key not in d:
                raise ValueError("restart: history '%s' lacks stored slot %d" % (name, slot))
            values = np.asarray(d[key], dtype=np.float64).ravel()
            if expected_values is None:
                expected_values = len(sim.history_global(name, slot))
            if values.size != expected_values:
                raise ValueError("restart: history '%s' slot %d has size %d, expected %d"
                                 % (name, slot, values.size, expected_values))
        dt_key = "history_slot_dt_" + name
        if dt_key in d and np.asarray(d[dt_key]).size != depth:
            raise ValueError("restart: history '%s' dt vector has wrong length" % name)


def _restore_histories_v3(sim, d):
    """Restore + replay the v3 history rings THROUGH in-window regrids (ADC-635), refusing the one
    genuinely impossible AMR case.

    RANK-COUNT CHANGE (kept): replaying a NON-Dense ring re-steps the installed Program whose regrids
    are collective, so a different rank count from the checkpoint would desync the deterministic regrid
    -- refuse LOUD (Dense rings need no replay -> they restart across any np).

    The ADC-631 straddle refusal is LIFTED: the replay now re-steps with regrid ACTIVE, driving the
    facade cursor so the ORIGINAL in-window regrid schedule fires and each recomputed slot rides the
    same incremental remap chain the stored anchors rode (rebuild_history_slots). The facade cursor is
    primed to the checkpoint macro-step m BEFORE the replay (the replay reads it as the anchor for the
    per-re-step cursor m-1-j and restores it afterwards; the final set_clock re-imposes m regardless).

    COHERENCE GUARD (replaces the refusal): the engine refuses a regrid completed OFF the due schedule
    derived from (depth, m, regrid_every); here we additionally assert the checkpoint's recorded
    fingerprint history_regrid_steps_<name> matches the schedule re-derived from the manifest's own
    scalars, and that every completed replay regrid sits on it. A mismatch fails LOUD; a due step whose regrid
    no-ops deterministically (single-level hierarchy, empty tags) is legitimate on both runs.
    Returns the typed HistoryReplayReport, or ``None`` when the checkpoint has no rings.
    """
    if "history_names" not in d or not len(list(d["history_names"])):
        return None
    from pops import _pops
    from pops.runtime._system_io_history import replay_regrid_steps, restore_histories
    chk_ranks = int(d["n_ranks"])
    cur_ranks = int(_pops.n_ranks())
    m = int(d["macro_step"])
    regrid_every = int(d["regrid_every"])
    for hname in (str(h) for h in d["history_names"]):
        depth = int(d["history_depth_" + hname])
        key = "history_stored_slots_" + hname
        if key not in d:
            raise ValueError("restart: history '%s' lacks its stored-slot index" % hname)
        stored = sorted(int(s) for s in d[key])
        if len(stored) >= depth:
            continue  # Dense (every slot stored): no replay -> the refusal does not apply.
        if chk_ranks != cur_ranks:
            raise ValueError(
                "restart : history '%s' uses a non-Dense persistence policy that must REPLAY the "
                "installed Program to reconstruct its slots, but the checkpoint was written under "
                "%d rank(s) and this restart uses %d; the deterministic regrid would desync across "
                "the rank-count change. Restart under %d rank(s), or checkpoint the ring with "
                "Dense() (Dense needs no replay and restarts across any np)."
                % (hname, chk_ranks, cur_ranks, chk_ranks))

    # Prime the facade cursor to the checkpoint macro-step so the replay's per-re-step cursor (m-1-j)
    # reproduces the ORIGINAL in-window regrid schedule. The engine bracket saves/restores it; the
    # final set_clock in restart_v3 re-imposes m (idempotent, the uninterrupted-clock invariant holds).
    sim.set_clock(float(d["t"]), m)

    fired = {}
    report = restore_histories(sim, d, fired_out=fired)

    # Fingerprint assertion (ADC-635). Two sound checks per replayed ring:
    # (1) the recorded fingerprint history_regrid_steps_<name> must equal the schedule re-derived from
    #     the manifest's own (depth, macro_step, regrid_every) -- pure arithmetic on the same scalars,
    #     so any corruption of the fingerprint, the cadence or the clock fails LOUD;
    # (2) every regrid the replay actually COMPLETED must sit on that schedule (an off-schedule firing
    #     means broken cursor driving / a divergent restart composition). A due step that completed
    #     NOTHING is legitimate: regrid() no-ops deterministically (single-level hierarchy, no wired
    #     predicate, empty tags) and by determinism the original run no-oped there identically.
    for hname, got in fired.items():
        derived = replay_regrid_steps(int(d["history_depth_" + hname]), m, regrid_every)
        key = "history_regrid_steps_" + hname
        if key not in d:
            raise ValueError("restart: history '%s' lacks its regrid replay fingerprint" % hname)
        recorded = sorted(int(s) for s in d[key])
        if recorded != derived:
            raise ValueError(
                "restart : history '%s' checkpoint records the in-window regrid schedule %r but "
                "its own macro_step=%d / regrid_every=%d derive %r; the manifest is corrupted or "
                "inconsistent with the recorded in-window regrid schedule."
                % (hname, recorded, m, regrid_every, derived))
        off = sorted(set(int(s) for s in got) - set(recorded))
        if off:
            raise ValueError(
                "restart : history '%s' replay completed regrids at macro-steps %r which are OFF the "
                "recorded in-window regrid schedule %r (checkpoint macro-step %d, regrid_every=%d); "
                "the restart composition is inconsistent with the recorded in-window regrid schedule."
                % (hname, off, recorded, m, regrid_every))
    return report


__all__ = ["write_v3", "restart_v3"]
