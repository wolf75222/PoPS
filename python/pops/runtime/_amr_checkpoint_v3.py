"""Strict v3 AMR checkpoint: exact topology, fields, histories and accepted Program state.

The sealed payload preserves owner-rank mappings, all block/level state, aux and elliptic warm starts,
regrid counters, qualified history rings, rational clocks, lagged flux publications, level relations
and transfer-plan provenance. Restore is transactional and the public route has one guarantee only:
bit-identical continuation under the same bound composition. Historical or weaker fallback formats
are refused.
"""

from dataclasses import dataclass
from typing import Any

from pops._generated_release_contract import AMR_CHECKPOINT_PAYLOAD_VERSION as _V3


@dataclass(frozen=True, slots=True)
class _PreparedAMRRestart:
    payload: Any
    temporal_state: Any
    program_state: Any
    regrid_count: int
    topology_epoch: int
    levels: int
    boxes: tuple[Any, ...]
    owner_ranks: tuple[int, ...]
    multi: bool
    state_payload: tuple[Any, ...]
    aux_payload: tuple[Any, ...]
    potential_payload: tuple[Any, ...]
    field_payload: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _PreparedAMRCapture:
    target: Any
    payload: dict[str, Any]
    multi: bool
    names: tuple[str, ...]
    levels: int
    field_slots: tuple[str, ...]
    field_levels: tuple[int, ...]
    history_plan: Any
    topology: Any
    local_dmaps: tuple[tuple[int, ...], ...]
    local_program_state: bytes
    capture_identity: str


def _prepare_capture_v3(owner, sim, path, lengths, lower, regrid_every, persistence):
    """Freeze the complete AMR gather plan without invoking a native collective."""
    import numpy as np
    from pops.identity import make_identity
    from pops.output._checkpoint_collective import canonical_checkpoint_path, checkpoint_topology
    from pops.runtime._amr_checkpoint_contract import encode_contract
    from pops.runtime._engine_descriptors import abi_key
    from pops.runtime._system_io_history import prepare_history_capture

    if int(sim.n_blocks()) == 0:
        raise ValueError(
            "AmrSystem.checkpoint: no blocks installed (nothing to checkpoint); bind a compiled "
            "problem with pops.bind(...) before checkpointing")
    topology = checkpoint_topology(owner)
    target = canonical_checkpoint_path(path)
    multi = bool(sim.uses_runtime_engine()) if hasattr(sim, "uses_runtime_engine") \
        else int(sim.n_blocks()) != 1
    levels = int(sim.n_levels())
    names = tuple(str(name) for name in sim.block_names())
    if levels <= 0 or not names or len(names) != len(set(names)):
        raise ValueError("checkpoint requires non-empty unique AMR blocks and levels")
    patch_boxes = tuple(tuple(int(value) for value in row) for row in sim.patch_boxes())
    if any(len(row) != 5 for row in patch_boxes):
        raise ValueError("checkpoint AMR patch-box rows must contain exactly five integers")
    time = float(sim.time())
    macro_step = int(sim.macro_step())
    temporal = getattr(owner, "_temporal_restart_state", None)
    if temporal is None:
        raise RuntimeError("checkpoint requires the AMR temporal restart state")
    temporal_json = temporal.checkpoint_json(time=time, macro_step=macro_step)
    program_state = bytes(sim.program_accepted_state())
    accepted_contract = encode_contract(sim)
    dmaps = tuple(
        tuple(int(rank) for rank in sim.level_owner_ranks(level))
        if hasattr(sim, "level_owner_ranks") else ()
        for level in range(levels)
    )
    nvars = tuple(
        int(sim.block_n_vars(name)) if multi else int(sim.n_vars()) for name in names)
    if any(value <= 0 for value in nvars):
        raise ValueError("checkpoint AMR blocks must have a positive conservative size")
    field_slots = tuple(str(slot) for slot in sim.field_provider_slots()) \
        if hasattr(sim, "field_provider_slots") else ()
    if len(field_slots) != len(set(field_slots)):
        raise ValueError("checkpoint AMR field-provider slots must be unique")
    field_levels = tuple(int(sim.field_provider_levels(slot)) for slot in field_slots)
    if any(value <= 0 or value > levels for value in field_levels):
        raise ValueError("checkpoint AMR field-provider level counts are invalid")
    history_plan = prepare_history_capture(
        sim,
        persistence or {},
        macro_step=macro_step,
        regrid_every=int(regrid_every),
    )
    required_collectives = []
    if topology.distributed:
        required_collectives.extend(
            ["block_level_state_global" if multi else "level_state_global",
             "level_potential_global", "level_aux_flat_global"])
    else:
        required_collectives.extend(
            ["block_level_state" if multi else "level_state",
             "level_potential", "level_aux_flat"])
    if field_slots:
        required_collectives.append("field_potential_level_global")
    if any(ring.stored_slots for ring in history_plan.rings):
        required_collectives.append("history_global")
    missing_collectives = sorted({
        name for name in required_collectives if not callable(getattr(sim, name, None))
    })
    if missing_collectives:
        raise TypeError(
            "checkpoint AMR engine lacks capture accessors %r" % missing_collectives)
    nx, ny = int(sim.nx()), int(sim.ny())
    Lx, Ly = (float(value) for value in lengths)
    xlo, ylo = (float(value) for value in lower)
    out = {
        "pops_amr_checkpoint_version": _V3,
        "t": time,
        "macro_step": macro_step,
        # n/L retain their historical meaning as the x-axis values.  The y-axis and origin are
        # persisted independently so rectangular and shifted Cartesian domains authenticate exactly.
        "n": nx,
        "ny": ny,
        "L": Lx,
        "Ly": Ly,
        "xlo": xlo,
        "ylo": ylo,
        "regrid_every": int(regrid_every),
        "abi_key": abi_key(),
        "blocks": np.array(names),
        "n_levels": levels,
        "n_ranks": topology.size,
        "patch_boxes": (np.asarray(patch_boxes, dtype=np.int64)
                        if patch_boxes else np.zeros((0, 5), dtype=np.int64)),
        "temporal_restart_state": np.array(temporal_json),
        "regrid_count": int(sim.checkpoint_regrid_count()),
        "topology_epoch": int(sim.checkpoint_topology_epoch()),
        "amr_accepted_contract": accepted_contract,
        "program_hash": str(sim.installed_program_hash())
        if hasattr(sim, "installed_program_hash") else "",
        "field_provider_slots": np.asarray(field_slots),
    }
    for name, value in zip(names, nvars, strict=True):
        out["n_vars_%s" % name] = value
    for index, value in enumerate(field_levels):
        out["field_provider_levels_%d" % index] = value
    capture_identity = make_identity("checkpoint-capture-plan", {
        "runtime_kind": "amr",
        "target": str(target),
        "clock": {"time": time.hex(), "macro_step": macro_step},
        "cells": [nx, ny],
        "lower": [xlo.hex(), ylo.hex()],
        "upper": [(xlo + Lx).hex(), (ylo + Ly).hex()],
        "regrid_every": int(regrid_every),
        "abi_key": str(out["abi_key"]),
        "blocks": [{"name": name, "nvars": value}
                   for name, value in zip(names, nvars, strict=True)],
        "levels": levels,
        "patch_boxes": [list(row) for row in patch_boxes],
        # Distribution mappings and the opaque accepted Program image contain rank-local state.
        # Only their schema participates in the collective plan; capture stores every exact rank
        # image in the single authenticated artifact below.
        "dmap_sizes": [len(row) for row in dmaps],
        "field_slots": [
            {"name": slot, "levels": count}
            for slot, count in zip(field_slots, field_levels, strict=True)
        ],
        "program_hash": str(out["program_hash"]),
        "program_state_present": bool(program_state),
        "accepted_contract": accepted_contract,
        "histories": history_plan.to_data(),
        "runtime_identities": [value.to_data() for value in owner._checkpoint_identities()],
        "run_identity": owner.last_run_identity.to_data(),
    }).token
    return _PreparedAMRCapture(
        target, out, multi, names, levels, field_slots, field_levels,
        history_plan, topology, dmaps, program_state, capture_identity)


def _capture_v3(owner, sim, prepared):
    """Execute the agreed AMR gather order and seal the in-memory payload."""
    if not isinstance(prepared, _PreparedAMRCapture):
        raise TypeError("AMR checkpoint capture requires its exact prepared plan")
    import numpy as np
    from pops._native_collectives import allgather_bytes
    from pops.output._checkpoint_collective import consensus
    from pops.runtime._checkpoint_manifest import seal_checkpoint_payload
    from pops.runtime._system_io_history import capture_histories

    out = dict(prepared.payload)
    gather = prepared.topology.distributed
    program_states = (
        allgather_bytes(prepared.topology.communicator, prepared.local_program_state)
        if prepared.topology.distributed else (prepared.local_program_state,)
    )
    rank_rows = consensus(
        prepared.topology,
        "AMR rank-local owner maps",
        value={"dmaps": [list(row) for row in prepared.local_dmaps]},
    )
    for row in rank_rows:
        rank = int(row["rank"])
        metadata = row["value"]
        if not isinstance(metadata, dict) or set(metadata) != {"dmaps"}:
            raise RuntimeError("checkpoint AMR rank-local metadata has an invalid schema")
        dmaps = metadata["dmaps"]
        if not isinstance(dmaps, list) or len(dmaps) != prepared.levels:
            raise ValueError("checkpoint AMR rank-local owner maps have an invalid level count")
        out["program_accepted_state_rank_%d" % rank] = np.frombuffer(
            program_states[rank], dtype=np.uint8).copy()
        for level, ranks in enumerate(dmaps):
            if not isinstance(ranks, list) or any(
                isinstance(value, bool) or not isinstance(value, int) for value in ranks
            ):
                raise TypeError("checkpoint AMR rank-local owner map must contain integers")
            out["dmap_rank_%d_level_%d" % (rank, level)] = np.asarray(
                ranks, dtype=np.int64)
    if prepared.multi:
        for name in prepared.names:
            for level in range(prepared.levels):
                out["state_%s_%d" % (name, level)] = np.asarray(
                    sim.block_level_state_global(name, level)
                    if gather else sim.block_level_state(name, level),
                    dtype=np.float64,
                )
    else:
        name = prepared.names[0]
        for level in range(prepared.levels):
            out["state_%s_%d" % (name, level)] = np.asarray(
                sim.level_state_global(level) if gather else sim.level_state(level),
                dtype=np.float64,
            )
    for level in range(prepared.levels):
        out["phi_%d" % level] = np.asarray(
            sim.level_potential_global(level) if gather else sim.level_potential(level),
            dtype=np.float64,
        )
    for index, (slot, count) in enumerate(
        zip(prepared.field_slots, prepared.field_levels, strict=True)
    ):
        for level in range(count):
            out["field_provider_phi_%d_%d" % (index, level)] = np.asarray(
                sim.field_potential_level_global(slot, level), dtype=np.float64)
    for level in range(prepared.levels):
        out["aux_%d" % level] = np.asarray(
            sim.level_aux_flat_global(level) if gather else sim.level_aux_flat(level),
            dtype=np.float64,
        )
    capture_histories(sim, prepared.history_plan, out)
    identity = seal_checkpoint_payload(owner, out, runtime_kind="amr")
    return out, identity.token


def write_v3(owner, sim, path, lengths, lower, regrid_every, persistence=None):
    """Capture exact AMR accepted state with preflight consensus before native gathers."""
    import os
    import numpy as np
    from pops.output._checkpoint_collective import collective_checkpoint_capture

    prepared_holder = {}

    def prepare():
        prepared = _prepare_capture_v3(
            owner, sim, path, lengths, lower, regrid_every, persistence or {})
        prepared_holder["plan"] = prepared
        return prepared, prepared.capture_identity

    def capture(prepared):
        return _capture_v3(owner, sim, prepared)

    def publish(payload):
        prepared = prepared_holder["plan"]
        temporary = prepared.target.with_name(prepared.target.name + ".tmp")
        try:
            with open(temporary, "wb") as stream:
                np.savez_compressed(stream, **payload)
            os.replace(temporary, prepared.target)
        finally:
            temporary.unlink(missing_ok=True)
        return str(prepared.target)

    return collective_checkpoint_capture(
        owner, "AMR accepted-state capture", prepare, capture, publish)


def prepare_v3(owner, sim, d, lengths, lower):
    """Validate a v3 AMR payload completely without mutating the native engine.

    This is the all-rank preflight boundary used before ``begin_restart_transaction``.
    """
    import numpy as np
    from pops.output._checkpoint_collective import checkpoint_topology
    from pops.runtime._amr_checkpoint_contract import preflight_contract
    from pops.runtime._temporal_restart import TemporalRestartState

    topology = checkpoint_topology(owner)
    current_ranks = topology.size
    if "n_ranks" not in d:
        raise ValueError("restart: AMR checkpoint lacks its exact native rank count")
    checkpoint_ranks = int(d["n_ranks"])
    if checkpoint_ranks != current_ranks:
        raise ValueError(
            "restart: AMR accepted state was captured under %d rank(s), current run has %d; "
            "rank-local DistributionMappings and compiled Program publications require the exact "
            "native rank topology"
            % (checkpoint_ranks, current_ranks)
        )
    if "n_levels" not in d:
        raise ValueError("restart: AMR checkpoint lacks its hierarchy level count")
    checkpoint_levels = int(d["n_levels"])
    selected = dict(d)
    for rank in range(checkpoint_ranks):
        state_key = "program_accepted_state_rank_%d" % rank
        if state_key not in d:
            raise ValueError(
                "restart: AMR checkpoint lacks accepted Program state for rank %d" % rank)
        state = np.asarray(d[state_key])
        if state.dtype != np.dtype("uint8") or state.ndim != 1:
            raise TypeError(
                "restart: AMR accepted Program state for rank %d must be a uint8 vector" % rank)
        for level in range(checkpoint_levels):
            dmap_key = "dmap_rank_%d_level_%d" % (rank, level)
            if dmap_key not in d:
                raise ValueError(
                    "restart: AMR checkpoint lacks owner map for rank %d level %d"
                    % (rank, level)
                )
            owner_map = np.asarray(d[dmap_key])
            if owner_map.dtype.kind not in "iu" or owner_map.ndim != 1:
                raise TypeError(
                    "restart: AMR owner map for rank %d level %d must be an integer vector"
                    % (rank, level)
                )
    selected["program_accepted_state"] = np.asarray(
        d["program_accepted_state_rank_%d" % topology.rank], dtype=np.uint8)
    for level in range(checkpoint_levels):
        selected["dmap_%d" % level] = np.asarray(
            d["dmap_rank_%d_level_%d" % (topology.rank, level)], dtype=np.int64)
    d = selected
    program_state, regrid_count, topology_epoch = preflight_contract(sim, d)
    if "temporal_restart_state" not in d:
        raise ValueError("restart: AMR checkpoint lacks its strict temporal state")
    installed_schedule = getattr(
        getattr(owner, "_temporal_restart_state", None), "program_schedule", None)
    restored_temporal = TemporalRestartState.from_json(
        d["temporal_restart_state"], time=d["t"], macro_step=d["macro_step"],
        program_schedule=installed_schedule)

    # (2) GUARDS.
    geometry_keys = ("n", "ny", "L", "Ly", "xlo", "ylo")
    missing_geometry = [key for key in geometry_keys if key not in d]
    if missing_geometry:
        raise ValueError(
            "restart: AMR checkpoint lacks exact Cartesian geometry keys %r"
            % missing_geometry)
    current_cells = (int(sim.nx()), int(sim.ny()))
    checkpoint_cells = (int(d["n"]), int(d["ny"]))
    if checkpoint_cells != current_cells:
        raise ValueError("restart : checkpoint grid %r != system grid %r"
                         % (checkpoint_cells, current_cells))
    current_lengths = tuple(float(value) for value in lengths)
    checkpoint_lengths = (float(d["L"]), float(d["Ly"]))
    if checkpoint_lengths != current_lengths:
        raise ValueError(
            "restart : checkpoint domain lengths %r != system lengths %r -- different spacing"
            % (checkpoint_lengths, current_lengths))
    current_lower = tuple(float(value) for value in lower)
    checkpoint_lower = (float(d["xlo"]), float(d["ylo"]))
    if checkpoint_lower != current_lower:
        raise ValueError("restart : checkpoint lower bounds %r != system lower bounds %r"
                         % (checkpoint_lower, current_lower))
    chk_blocks = [str(b) for b in d["blocks"]]
    cur_blocks = list(sim.block_names())
    if chk_blocks != cur_blocks:
        raise ValueError("restart : checkpoint blocks %r != current composition %r "
                         "(replay the SAME composition before restart)" % (chk_blocks, cur_blocks))
    nlev = int(d["n_levels"])
    if nlev < 1:
        raise ValueError("restart: an AMR checkpoint must contain at least the coarse level")
    maximum_levels = int(sim.max_levels()) if hasattr(sim, "max_levels") else int(sim.n_levels())
    if nlev > maximum_levels:
        raise ValueError(
            "restart: checkpoint active depth %d exceeds the resolved maximum depth %d"
            % (nlev, maximum_levels))
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
    if not multi and nlev != int(sim.n_levels()):
        raise ValueError(
            "restart: the legacy fixed-hierarchy route cannot change active depth "
            "(%d checkpoint levels, %d materialized levels)"
            % (nlev, int(sim.n_levels())))

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
        provider_levels = int(d[levels_key])
        if provider_levels != nlev:
            raise ValueError(
                "restart: field provider %s persists %d levels for a %d-level checkpoint"
                % (slot, provider_levels, nlev))
        values = []
        for k in range(provider_levels):
            key = "field_provider_phi_%d_%d" % (index, k)
            if key not in d:
                raise ValueError(
                    "restart: checkpoint lacks level %d potential for field provider %s"
                    % (k, slot))
            width = int(sim.nx()) << k
            height = int(sim.ny()) << k
            value = np.asarray(d[key], dtype=np.float64).ravel()
            if value.size != width * height:
                raise ValueError(
                    "restart: field provider %s level %d potential has size %d, expected %d"
                    % (slot, k, value.size, width * height))
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
        height = int(sim.ny()) << level
        if (ilo < 0 or jlo < 0 or ihi < ilo or jhi < jlo
                or ihi >= width or jhi >= height):
            raise ValueError("restart: invalid level-%d patch box %r for shape (%d, %d)"
                             % (level, box[1:], height, width))
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
        nranks = current_ranks
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
            height = int(sim.ny()) << level
            state = np.asarray(d[key], dtype=np.float64)
            expected = current_nvars * width * height
            if state.size != expected:
                raise ValueError(
                    "restart: block '%s' level %d state has size %d, expected %d"
                    % (block, level, state.size, expected))
            levels.append(state)
        state_payload.append((block, levels))

    aux_payload = []
    phi_payload = []
    coarse_width = int(sim.nx()) * int(sim.ny())
    coarse_aux_size = len(sim.level_aux_flat(0))
    if coarse_width < 1 or coarse_aux_size % coarse_width:
        raise ValueError("restart: installed coarse auxiliary storage has an invalid shape")
    aux_components = coarse_aux_size // coarse_width
    for level in range(nlev):
        aux_key = "aux_%d" % level
        phi_key = "phi_%d" % level
        if aux_key not in d or phi_key not in d:
            raise ValueError("restart: checkpoint lacks aux or potential payload for level %d" % level)
        aux = np.asarray(d[aux_key], dtype=np.float64).ravel()
        width = int(sim.nx()) << level
        height = int(sim.ny()) << level
        expected_aux = aux_components * width * height
        if aux.size != expected_aux:
            raise ValueError("restart: level %d aux has size %d, expected %d"
                             % (level, aux.size, expected_aux))
        phi = np.asarray(d[phi_key], dtype=np.float64).ravel()
        if phi.size != width * height:
            raise ValueError("restart: level %d potential has size %d, expected %d"
                             % (level, phi.size, width * height))
        aux_payload.append(aux)
        phi_payload.append(phi)

    _preflight_histories_v3(sim, d, current_ranks)

    return _PreparedAMRRestart(
        payload=d,
        temporal_state=restored_temporal,
        program_state=program_state,
        regrid_count=int(regrid_count),
        topology_epoch=int(topology_epoch),
        levels=nlev,
        boxes=tuple(boxes),
        owner_ranks=tuple(int(rank) for rank in owner_ranks),
        multi=bool(multi),
        state_payload=tuple((block, tuple(levels)) for block, levels in state_payload),
        aux_payload=tuple(aux_payload),
        potential_payload=tuple(phi_payload),
        field_payload=tuple((slot, tuple(levels)) for slot, levels in field_payload),
    )


def apply_v3(owner, sim, prepared):
    """Apply a fully prepared AMR payload inside an already active native transaction."""
    if type(prepared) is not _PreparedAMRRestart:
        raise TypeError("AMR restart requires its exact prepared payload")
    d = prepared.payload

    # (3) Impose the exact recorded hierarchy.
    if prepared.multi:
        sim.rebuild_hierarchy(prepared.boxes, prepared.owner_ranks)
    elif prepared.levels >= 2:
        sim.set_hierarchy(prepared.boxes)

    # A freshly bound compiled Program has not executed its prelude yet, so its native history
    # rings do not exist.  Materialize the exact qualified registry from the authenticated accepted
    # image on the already rebuilt hierarchy; never advance physics merely to trigger allocation.
    history_names = ([str(name) for name in d["history_names"]]
                     if "history_names" in d else [])
    if history_names and prepared.program_state:
        sim.materialize_program_restart_histories(
            prepared.program_state,
            history_names,
            [int(d["history_depth_" + name]) for name in history_names],
            [int(d["history_ncomp_" + name]) for name in history_names],
        )
    installed_names = list(sim.history_names())
    if installed_names != history_names:
        raise ValueError(
            "restart: checkpoint history rings %r != installed rings %r"
            % (history_names, installed_names))
    for name in history_names:
        depth = int(d["history_depth_" + name])
        ncomp = int(d["history_ncomp_" + name])
        if depth != int(sim.history_depth(name)) or ncomp != int(sim.history_ncomp(name)):
            raise ValueError(
                "restart: history '%s' shape (%d, %d) != installed shape (%d, %d)"
                % (name, depth, ncomp, int(sim.history_depth(name)),
                   int(sim.history_ncomp(name))))

    # (4) Restore every block/level state as saved, without re-prolongation.
    for block, levels in prepared.state_payload:
        for level, state in enumerate(levels):
            if prepared.multi:
                sim.set_block_level_state(block, level, state)
            else:
                sim.set_level_state(level, state)

    # (5) Restore shared aux only on the runtime route (the coupler deliberately persists an
    # explicit empty aux payload), then all elliptic warm starts.
    for level, aux in enumerate(prepared.aux_payload):
        if aux.size:
            sim.set_level_aux_flat(level, aux)
    for level, phi in enumerate(prepared.potential_payload):
        sim.set_level_potential(level, phi)
    for slot, levels in prepared.field_payload:
        for level, value in enumerate(levels):
            sim.set_field_potential_level(slot, level, value)

    # (6) Selectively stored clean-window histories may replay the Program. A window containing a
    # scheduled regrid was explicitly promoted to dense storage during capture.
    from pops.output._checkpoint_collective import checkpoint_topology
    report = _restore_histories_v3(sim, d, checkpoint_topology(owner).size)

    # (7) Replay is allowed to mutate Program clocks/ring publications and regrid counters while it
    # reconstructs policy-omitted dense values. Replace those temporary values with the checkpoint's
    # exact accepted semantic state before exposing the runtime again.
    sim.restore_program_accepted_state(prepared.program_state)
    from pops.runtime._amr_checkpoint_contract import validate_restored_contract
    validate_restored_contract(sim, d)
    sim.restore_checkpoint_counters(prepared.regrid_count, prepared.topology_epoch)

    # (8) Clock last: the next cadence decision is identical to the uninterrupted run.
    sim.set_clock(float(d["t"]), int(d["macro_step"]))
    owner._temporal_restart_state = prepared.temporal_state
    owner._step_controller = None
    return report


def _preflight_histories_v3(sim, d, current_ranks):
    """Validate the entire ring registry and persisted buffers without mutating native state."""
    import numpy as np
    from pops.time._history.persistence import HistoryPersistence

    checkpoint_names = ([str(name) for name in d["history_names"]]
                        if "history_names" in d else [])
    if len(checkpoint_names) != len(set(checkpoint_names)):
        raise ValueError("restart: checkpoint history ring names must be unique")
    checkpoint_ranks = int(d["n_ranks"]) if "n_ranks" in d else 1
    for name in checkpoint_names:
        required = ["history_depth_" + name, "history_ncomp_" + name,
                    "history_init_" + name, "history_policy_" + name,
                    "history_requested_stored_slots_" + name,
                    "history_stored_slots_" + name, "history_storage_mode_" + name,
                    "history_slot_dt_" + name]
        missing = [key for key in required if key not in d]
        if missing:
            raise ValueError("restart: history '%s' lacks keys %r" % (name, missing))
        depth = int(d["history_depth_" + name])
        ncomp = int(d["history_ncomp_" + name])
        if depth < 2 or ncomp < 1:
            raise ValueError(
                "restart: history '%s' requires depth >= 2 and component count >= 1" % name)
        policy = HistoryPersistence.from_json(str(d["history_policy_" + name]))
        from pops.runtime._system_io_history import resolve_history_storage
        expected_requested, expected_stored, expected_mode, expected_steps = \
            resolve_history_storage(
                policy,
                depth,
                macro_step=int(d["macro_step"]),
                regrid_every=int(d["regrid_every"]),
            )
        requested = sorted(
            int(slot) for slot in d["history_requested_stored_slots_" + name])
        stored = sorted(int(slot) for slot in d["history_stored_slots_" + name])
        mode = str(d["history_storage_mode_" + name])
        if requested != list(expected_requested):
            raise ValueError(
                "restart: history '%s' requested slots %r != policy %s expects %r"
                % (name, requested, policy.name, list(expected_requested)))
        if stored != list(expected_stored) or mode != expected_mode:
            raise ValueError(
                "restart: history '%s' resolved storage plan (%r, %s) != expected (%r, %s)"
                % (name, stored, mode, list(expected_stored), expected_mode))
        if len(stored) < depth:
            if checkpoint_ranks != current_ranks:
                raise ValueError(
                    "restart: non-Dense history '%s' was written under %d rank(s), current run has %d"
                    % (name, checkpoint_ranks, current_ranks))
            if "history_regrid_steps_" + name not in d:
                raise ValueError("restart: history '%s' lacks its regrid replay fingerprint" % name)
            recorded = sorted(int(step) for step in d["history_regrid_steps_" + name])
            if recorded != list(expected_steps or ()):
                raise ValueError(
                    "restart: history '%s' regrid fingerprint %r differs from manifest cadence %r"
                    % (name, recorded, list(expected_steps or ())))
        elif expected_steps is not None:
            key = "history_regrid_steps_" + name
            if key not in d:
                raise ValueError("restart: history '%s' lacks its regrid schedule fingerprint" % name)
            recorded = sorted(int(step) for step in d[key])
            if recorded != list(expected_steps):
                raise ValueError(
                    "restart: history '%s' dense safety regrid fingerprint %r is inconsistent "
                    "with manifest cadence %r"
                    % (name, recorded, list(expected_steps)))
        expected_values = ncomp * sum(
            (int(sim.nx()) << level) * (int(sim.ny()) << level)
            for level in range(int(d["n_levels"])))
        for slot in stored:
            key = "history_%s_%d" % (name, slot)
            if key not in d:
                raise ValueError("restart: history '%s' lacks stored slot %d" % (name, slot))
            values = np.asarray(d[key], dtype=np.float64).ravel()
            if values.size != expected_values:
                raise ValueError("restart: history '%s' slot %d has size %d, expected %d"
                                 % (name, slot, values.size, expected_values))
        dt_key = "history_slot_dt_" + name
        if np.asarray(d[dt_key], dtype=np.float64).ravel().size != depth:
            raise ValueError("restart: history '%s' dt vector has wrong length" % name)


def _restore_histories_v3(sim, d, cur_ranks):
    """Restore v3 rings and replay only policy-omitted slots on a stable hierarchy.

    Capture resolves any selective ring whose replay window contains a scheduled regrid to explicit
    ``dense_regrid_safety`` storage. Therefore this function replays only clean windows; the native
    seam independently refuses an unsafe handcrafted payload. Returns the typed
    :class:`HistoryReplayReport`, or ``None`` when the checkpoint has no rings.
    """
    if "history_names" not in d or not len(list(d["history_names"])):
        return None
    from pops.runtime._system_io_history import replay_regrid_steps, restore_histories
    chk_ranks = int(d["n_ranks"])
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
                "the rank-count change. Restart under %d rank(s)."
                % (hname, chk_ranks, cur_ranks, chk_ranks))

    # Prime the facade cursor to the checkpoint macro-step so clean-window re-steps use their original
    # logical cursors. The final set_clock in apply_v3 re-imposes m.
    sim.set_clock(float(d["t"]), m)

    fired = {}
    report = restore_histories(sim, d, fired_out=fired)

    # A clean-window replay must neither record nor complete a regrid. Preflight already authenticated
    # the schedule; this guards the native execution seam as well.
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
        completed = sorted(set(int(s) for s in got))
        if recorded or completed:
            raise ValueError(
                "restart : history '%s' selective replay was not a stable-hierarchy window "
                "(recorded=%r, completed=%r); capture must resolve it as dense_regrid_safety"
                % (hname, recorded, completed))
    return report


__all__ = ["apply_v3", "prepare_v3", "write_v3"]
