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

Restore order (addendum B.6): guards -> rebuild_hierarchy (impose the mid-run hierarchy) -> per-level
state -> aux -> phi -> clock. The clock is LAST so the next step's regrid_if_due sees the
uninterrupted clock.
"""

_V3 = 3


def write_v3(sim, path, L, regrid_every):
    """Write a v3 AMR checkpoint (restartable under active regridding). Returns the path.

    @p sim is the C++ AmrSystem engine (``self._s``); @p L the domain length; @p regrid_every the live
    regrid cadence. COLLECTIVE under np>1 (the _global gather accessors run on every rank; only rank 0
    writes). Multi-block keys the state per block; mono-block keys it by the single block name.
    """
    import os
    import numpy as np
    from pops import _pops
    from pops.runtime.bricks import abi_key

    gather = _pops.n_ranks() != 1
    multi = sim.n_blocks() != 1
    nlev = int(sim.n_levels())
    names = list(sim.block_names())
    pb = sim.patch_boxes()
    out = {"pops_amr_checkpoint_version": _V3,
           "t": sim.time(), "macro_step": sim.macro_step(),
           "n": sim.nx(), "L": L, "regrid_every": int(regrid_every),
           "abi_key": abi_key(), "blocks": np.array(names), "n_levels": nlev,
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

    The 8-step restore order (addendum B.6, the realized subset): guards (grid / blocks / components /
    regrid metadata / program-hash) -> rebuild_hierarchy (impose the mid-run hierarchy from the
    manifest) -> per-level per-block state (as-saved, no re-prolongation) -> shared phi warm-start ->
    clock LAST (so the next step's regrid_if_due sees the uninterrupted clock).
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
    multi = sim.n_blocks() != 1

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

    # (8) CLOCK LAST (macro_step advances the regrid cadence phase).
    sim.set_clock(float(d["t"]), int(d["macro_step"]))


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
