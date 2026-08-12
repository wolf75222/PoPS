"""Strict accepted-state checkpoint/restart mixin for the Uniform System engine.

Scientific output belongs exclusively to ConsumerGraph providers.  This private engine adapter only
owns the restart payload codec and its native transaction; it has no format or MPI policy surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pops.runtime._engine_descriptors import abi_key

if TYPE_CHECKING:
    from pops.runtime._system_contract import _System
else:
    _System = object


@dataclass(frozen=True, slots=True)
class _PreparedUniformRestart:
    payload: Any
    restart_identity: Any
    temporal_state: Any
    cadence_state: Any


@dataclass(frozen=True, slots=True)
class _PreparedUniformCapture:
    target: Path
    payload: dict[str, Any]
    blocks: tuple[str, ...]
    field_slots: tuple[str, ...]
    history_plan: Any
    cache_nodes: tuple[int, ...]
    capture_identity: str


class _SystemIO(_System):
    """Private accepted-state codec and transactional restore adapter for ``System``."""

    def set_history_persistence(self, mapping):
        """Attach the per-history persistence policies (ADC-626): @p mapping is ``name -> policy`` (a
        :class:`pops.time._history.persistence.HistoryPersistence`). The checkpoint reads it to store only
        the policy-selected slots; a ring absent from the map (or an empty map) persists Dense (the whole
        ring. Idempotent; ``None`` clears it."""
        self._history_persistence = dict(mapping or {})
        return self

    def last_restart_report(self):
        """The typed :class:`~pops.time._history.report.HistoryReplayReport` of the last
        restart (stored-vs-recomputed ring slots + replay steps), or ``None`` if no restart has run.
        Metadata-only (ADC-591); populated by the accepted-state restore transaction."""
        return getattr(self, "_last_restart_report", None)

    def _prepare_checkpoint_capture(self, path: Any) -> _PreparedUniformCapture:
        """Freeze and identify all local metadata before the first native collective."""
        import math

        import numpy as np
        from pops._generated_release_contract import UNIFORM_CHECKPOINT_PAYLOAD_VERSION
        from pops.identity import make_identity
        from pops.output._checkpoint_collective import canonical_checkpoint_path
        from pops.runtime._checkpoint_embedded_boundary import (
            add_checkpoint_embedded_boundary_contract,
            current_checkpoint_embedded_boundary_contract,
        )
        from pops.runtime._checkpoint_spatial import (
            add_checkpoint_spatial_contract,
            require_checkpoint_spatial_contract,
        )
        from pops.runtime._program_cadence_checkpoint import capture_program_cadence
        from pops.runtime._system_io_history import prepare_history_capture

        target = canonical_checkpoint_path(path)
        temporal = getattr(self, "_temporal_restart_state", None)
        if temporal is None:
            raise RuntimeError("checkpoint requires the Uniform temporal restart state")
        time = float(self._s.time())
        macro_step = int(self._s.macro_step())
        temporal_json = temporal.checkpoint_json(time=time, macro_step=macro_step)
        cadence = capture_program_cadence(self._s, macro_step=macro_step)
        blocks = tuple(str(block) for block in self._s.block_names())
        if not blocks or len(blocks) != len(set(blocks)):
            raise ValueError("checkpoint requires a non-empty unique Uniform block order")
        required_collectives = ["state_global", "potential_global"]
        if not callable(getattr(self._s, "capture_auxiliary_checkpoint_accepted_state", None)):
            raise TypeError("checkpoint Uniform engine lacks exact auxiliary checkpoint capture")
        out = {
            "pops_checkpoint_version": UNIFORM_CHECKPOINT_PAYLOAD_VERSION,
            "t": time,
            "macro_step": macro_step,
            "abi_key": abi_key(),
            "blocks": np.array(blocks),
            "temporal_restart_state": np.array(temporal_json),
        }
        spatial = require_checkpoint_spatial_contract(self)
        add_checkpoint_spatial_contract(out, spatial)
        embedded_boundary = current_checkpoint_embedded_boundary_contract(self)
        add_checkpoint_embedded_boundary_contract(out, embedded_boundary)
        out.update(cadence.to_payload())
        block_evidence = []
        for b in blocks:
            nv = int(self._s.n_vars(b))
            names = tuple(str(name) for name in self._s.variable_names(b, "conservative"))
            if nv <= 0 or len(names) != nv or len(names) != len(set(names)):
                raise ValueError("checkpoint block %r has an invalid conservative schema" % b)
            out["ncomp_" + b] = nv
            out["names_" + b] = np.array(names)
            block_evidence.append({"name": b, "ncomp": nv, "variables": list(names)})
        field_slots = tuple(str(slot) for slot in self._s.field_provider_slots())
        if len(field_slots) != len(set(field_slots)):
            raise ValueError("checkpoint field-provider slots must be unique")
        out["field_provider_slots"] = np.array(field_slots)
        if field_slots:
            required_collectives.append("field_potential_global")
        prog_hash = (
            str(self._s.installed_program_hash())
            if hasattr(self._s, "installed_program_hash")
            else ""
        )
        if not prog_hash:
            raise RuntimeError("checkpoint requires the installed compiled Program hash")
        out["program_hash"] = prog_hash
        history_plan = prepare_history_capture(
            self._s,
            getattr(self, "_history_persistence", None) or {},
            macro_step=macro_step,
        )
        if any(ring.stored_slots for ring in history_plan.rings):
            required_collectives.append("history_global")
        cache_nodes = (
            tuple(int(node) for node in self._s.program_cache_nodes())
            if hasattr(self._s, "program_cache_nodes")
            else ()
        )
        if cache_nodes != tuple(sorted(set(cache_nodes))) or any(node < 0 for node in cache_nodes):
            raise ValueError(
                "checkpoint scheduled-cache node ids must be unique non-negative canonical ids"
            )
        if cache_nodes and macro_step == 0:
            raise ValueError("checkpoint cannot carry a valid scheduled cache at step zero")
        if cache_nodes:
            required_collectives.append("program_cache_global")
        missing_collectives = sorted(
            {name for name in required_collectives if not callable(getattr(self._s, name, None))}
        )
        if missing_collectives:
            raise TypeError(
                "checkpoint Uniform engine lacks collective accessors %r" % missing_collectives
            )
        out["cache_nodes"] = np.array(cache_nodes, dtype=np.int64)
        cache_evidence = []
        cache_names = []
        for nid in cache_nodes:
            name = str(self._s.program_cache_name(nid))
            ncomp = int(self._s.program_cache_ncomp(nid))
            ngrow = int(self._s.program_cache_ngrow(nid))
            last_update = int(self._s.program_cache_last_update_step(nid))
            accum_dt = float(self._s.program_cache_accumulated_dt(nid))
            if (
                not name
                or ncomp <= 0
                or ngrow < 0
                or last_update < 0
                or last_update >= macro_step
                or not math.isfinite(accum_dt)
                or accum_dt < 0.0
            ):
                raise ValueError("checkpoint scheduled-cache metadata is invalid for node %d" % nid)
            cache_names.append(name)
            out["cache_ncomp_%d" % nid] = ncomp
            out["cache_ngrow_%d" % nid] = ngrow
            out["cache_last_update_%d" % nid] = last_update
            out["cache_accum_dt_%d" % nid] = accum_dt
            cache_evidence.append(
                {
                    "node": nid,
                    "name": name,
                    "ncomp": ncomp,
                    "ngrow": ngrow,
                    "last_update": last_update,
                    "accumulated_dt": accum_dt.hex(),
                }
            )
        out["cache_names"] = np.asarray(cache_names, dtype=str)
        runtime_identities = [value.to_data() for value in self._checkpoint_identities()]
        run_identity = self.last_run_identity.to_data()
        capture_identity = make_identity(
            "checkpoint-capture-plan",
            {
                "runtime_kind": "uniform",
                "target": str(target),
                "clock": {"time": time.hex(), "macro_step": macro_step},
                "spatial_contract": spatial.to_data(),
                "embedded_boundary_contract": embedded_boundary.to_data(),
                "abi_key": str(out["abi_key"]),
                "blocks": block_evidence,
                "field_slots": list(field_slots),
                "program_hash": prog_hash,
                "program_cadence": cadence.to_data(),
                "histories": history_plan.to_data(),
                "cache": cache_evidence,
                "runtime_identities": runtime_identities,
                "run_identity": run_identity,
            },
        ).token
        return _PreparedUniformCapture(
            target, out, blocks, field_slots, history_plan, cache_nodes, capture_identity
        )

    def _capture_checkpoint(self, prepared: _PreparedUniformCapture) -> tuple[dict[str, Any], str]:
        """Run the agreed native gather sequence and seal the in-memory payload."""
        if not isinstance(prepared, _PreparedUniformCapture):
            raise TypeError("Uniform checkpoint capture requires its exact prepared plan")
        import numpy as np
        from pops.runtime._checkpoint_manifest import seal_checkpoint_payload
        from pops.runtime._system_io_history import capture_histories

        out = dict(prepared.payload)
        for block in prepared.blocks:
            out["state_" + block] = np.asarray(self._s.state_global(block), dtype=np.float64)
        out["phi"] = np.asarray(self._s.potential_global(), dtype=np.float64)
        for index, slot in enumerate(prepared.field_slots):
            out["field_potential_%d" % index] = np.asarray(
                self._s.field_potential_global(slot), dtype=np.float64
            )
        auxiliary_checkpoint = self._s.capture_auxiliary_checkpoint_accepted_state()
        if type(auxiliary_checkpoint) is not bytes or not auxiliary_checkpoint.startswith(
            b"POPSAUX2"
        ):
            raise RuntimeError(
                "native Uniform exact auxiliary checkpoint is not a POPSAUX2 bytes image"
            )
        out["auxiliary_checkpoint"] = np.frombuffer(auxiliary_checkpoint, dtype=np.uint8).copy()
        capture_histories(self._s, prepared.history_plan, out)
        for node in prepared.cache_nodes:
            out["cache_value_%d" % node] = np.asarray(
                self._s.program_cache_global(node), dtype=np.float64
            )
        identity = seal_checkpoint_payload(self, out, runtime_kind="uniform")
        return out, identity.token

    def checkpoint(self, path: Any) -> Any:
        """Capture the exact accepted state through the collective checkpoint protocol."""
        import os
        import numpy as np
        from pops.output._checkpoint_collective import collective_checkpoint_capture

        prepared_holder = {}

        def prepare():
            prepared = self._prepare_checkpoint_capture(path)
            prepared_holder["plan"] = prepared
            return prepared, prepared.capture_identity

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
            self, "Uniform accepted-state capture", prepare, self._capture_checkpoint, publish
        )

    def _prepare_checkpoint_restart(
        self,
        payload: bytes,
        *,
        bit_identical: bool,
        hierarchy_mode: str = "restore_recorded_hierarchy",
        hierarchy_identity: str | None = None,
    ) -> _PreparedUniformRestart:
        """Authenticate and validate every byte before the first native state write."""
        if hierarchy_mode != "restore_recorded_hierarchy":
            raise NotImplementedError(
                "Uniform restart does not support RegridOnRestart; use an AMR layout"
            )
        if hierarchy_identity is not None:
            raise ValueError(
                "Uniform restart hierarchy identity is only valid with RegridOnRestart"
            )
        import numpy as np
        from pops._generated_release_contract import UNIFORM_CHECKPOINT_PAYLOAD_VERSION
        from pops.output._checkpoint_collective import (
            decode_checkpoint_bytes,
            require_restart_bit_identical,
        )
        from pops.runtime._checkpoint_manifest import (
            authenticate_checkpoint_payload,
            require_exact_payload_version,
        )
        from pops.runtime._checkpoint_embedded_boundary import (
            authenticate_checkpoint_embedded_boundary_contract,
        )
        from pops.runtime._checkpoint_spatial import authenticate_checkpoint_spatial_contract
        from pops.runtime._program_cadence_checkpoint import prepare_program_cadence
        from pops.runtime._temporal_restart import TemporalRestartState
        from pops.runtime._uniform_restart_preflight import preflight_uniform_restart
        from pops.time._history.persistence import HistoryPersistence
        from pops.runtime._system_io_history import (
            history_fill_count_from_payload,
            resolve_history_storage,
        )

        require_restart_bit_identical(bit_identical, where="Uniform restart")
        d = decode_checkpoint_bytes(payload)
        identity = authenticate_checkpoint_payload(self, d, runtime_kind="uniform")
        require_exact_payload_version(
            d,
            key="pops_checkpoint_version",
            expected=UNIFORM_CHECKPOINT_PAYLOAD_VERSION,
            runtime_kind="Uniform",
        )
        spatial = authenticate_checkpoint_spatial_contract(self, d)
        authenticate_checkpoint_embedded_boundary_contract(self, d)
        preflight_uniform_restart(d)
        if not callable(getattr(self._s, "restore_auxiliary_checkpoint_accepted_state", None)):
            raise TypeError("restart: Uniform engine lacks exact auxiliary checkpoint restore")
        if "auxiliary_checkpoint" not in d:
            raise ValueError("restart: checkpoint lacks its exact auxiliary payload")
        auxiliary_checkpoint = np.asarray(d["auxiliary_checkpoint"], dtype=np.uint8).ravel()
        if auxiliary_checkpoint.size < 8 or auxiliary_checkpoint[:8].tobytes() != b"POPSAUX2":
            raise ValueError("restart: exact auxiliary checkpoint is not POPSAUX2")
        cadence = prepare_program_cadence(
            self._s,
            d,
            macro_step=int(d["macro_step"]),
            accepted_time=float(d["t"]),
        )
        installed_schedule = getattr(
            getattr(self, "_temporal_restart_state", None), "program_schedule", None
        )
        temporal = TemporalRestartState.from_json(
            d["temporal_restart_state"],
            time=d["t"],
            macro_step=d["macro_step"],
            program_schedule=installed_schedule,
        )
        cells = spatial.cells_at_level(0)
        blocks = [str(block) for block in d["blocks"]]
        current_blocks = list(self._s.block_names())
        if blocks != current_blocks:
            raise ValueError(
                "restart : checkpoint blocks %r != current composition %r "
                "(replay the SAME composition before restart)" % (blocks, current_blocks)
            )
        for block in blocks:
            ncomp = int(d["ncomp_" + block])
            if ncomp != int(self._s.n_vars(block)):
                raise ValueError(
                    "restart : block '%s' has %d components in the checkpoint, %d here"
                    % (block, ncomp, self._s.n_vars(block))
                )
            if np.asarray(d["state_" + block]).size != ncomp * cells:
                raise ValueError("restart : block '%s' state payload has the wrong size" % block)
        if np.asarray(d["phi"]).size != cells:
            raise ValueError("restart : potential payload has the wrong size")
        slots = [str(slot) for slot in d["field_provider_slots"]]
        current_slots = list(self._s.field_provider_slots())
        if slots != current_slots:
            raise RuntimeError(
                "checkpoint qualified field providers %r != installed providers %r"
                % (slots, current_slots)
            )
        for index, slot in enumerate(slots):
            key = "field_potential_%d" % index
            if key not in d or np.asarray(d[key]).size != cells:
                raise RuntimeError(
                    "checkpoint potential for qualified field provider %s is missing or malformed"
                    % slot
                )
        checkpoint_hash = str(d["program_hash"])
        current_hash = (
            self._s.installed_program_hash() if hasattr(self._s, "installed_program_hash") else ""
        )
        if current_hash != checkpoint_hash:
            raise RuntimeError("checkpoint was created with a different compiled Program hash")

        history_names = [str(name) for name in d["history_names"]]
        current_histories = (
            list(self._s.history_names()) if hasattr(self._s, "history_names") else []
        )
        # The generated Program registers its native rings lazily inside the first step closure.
        # A fresh bind is nevertheless a valid restart target, so the authenticated installed
        # temporal schedule is the pre-mutation authority until that native registry exists.
        # Once any ring has materialized, require the whole registry to agree exactly.
        if current_histories and history_names != current_histories:
            raise RuntimeError(
                "checkpoint Program histories %r != installed qualified histories %r"
                % (history_names, current_histories)
            )
        schedule = temporal.program_schedule
        if (history_names or len(d["cache_nodes"])) and schedule is None:
            raise RuntimeError(
                "checkpoint history/cache state requires an installed temporal Program schedule"
            )
        scheduled_histories = {
            str(row["name"]): row for row in (schedule or {}).get("histories", ())
        }
        if history_names != list(scheduled_histories):
            raise RuntimeError(
                "checkpoint Program histories differ from the installed temporal schedule"
            )
        for name in history_names:
            depth = int(d["history_depth_" + name])
            ncomp = int(d["history_ncomp_" + name])
            scheduled = scheduled_histories[name]
            scheduled_ncomp = scheduled.get("ncomp")
            if scheduled_ncomp is None:
                owner = scheduled.get("owner")
                owner_name = owner.get("local_id") if isinstance(owner, dict) else None
                if not isinstance(owner_name, str) or owner_name not in current_blocks:
                    raise RuntimeError(
                        "installed full-state history '%s' has no current block owner" % name
                    )
                scheduled_ncomp = int(self._s.n_vars(owner_name))
            if depth != int(scheduled["ring_slots"]) or ncomp != int(scheduled_ncomp):
                raise ValueError(
                    "restart : history '%s' shape differs from the installed ring" % name
                )
            if current_histories and (
                depth != int(self._s.history_depth(name))
                or ncomp != int(self._s.history_ncomp(name))
            ):
                raise ValueError(
                    "restart : history '%s' shape differs from the materialized native ring" % name
                )
            policy = HistoryPersistence.from_json(str(d["history_policy_" + name]))
            initialized = bool(d["history_init_" + name])
            fill_count = history_fill_count_from_payload(d, name, depth, initialized)
            requested = sorted(int(slot) for slot in d["history_requested_stored_slots_" + name])
            stored = sorted(int(slot) for slot in d["history_stored_slots_" + name])
            (
                expected_requested,
                expected_stored,
                expected_mode,
                expected_regrid_steps,
            ) = resolve_history_storage(
                policy,
                depth,
                fill_count=fill_count,
                macro_step=int(d["macro_step"]),
                regrid_every=0,
                program_substeps=int(d["program_cadence_substeps"]),
                program_stride=int(d["program_cadence_stride"]),
            )
            if requested != list(expected_requested):
                raise ValueError(
                    "restart : history '%s' requested slots differ from its policy" % name
                )
            if (
                stored != list(expected_stored)
                or str(d["history_storage_mode_" + name]) != expected_mode
            ):
                raise ValueError(
                    "restart : history '%s' stored slots/mode differ from its resolved plan" % name
                )
            regrid_key = "history_regrid_steps_" + name
            if expected_regrid_steps is None:
                if regrid_key in d:
                    raise ValueError(
                        "restart : dense history '%s' carries a spurious replay schedule" % name
                    )
            else:
                if regrid_key not in d:
                    raise ValueError(
                        "restart : selective history '%s' lacks its replay schedule" % name
                    )
                recorded_regrid_steps = tuple(int(step) for step in d[regrid_key])
                if recorded_regrid_steps != tuple(expected_regrid_steps):
                    raise ValueError(
                        "restart : history '%s' replay schedule differs from its policy" % name
                    )
            if len(stored) < depth and not hasattr(self._s, "rebuild_history_slots"):
                raise RuntimeError(
                    "runtime cannot rebuild selectively persisted history '%s'" % name
                )
            for slot in stored:
                if np.asarray(d["history_%s_%d" % (name, slot)]).size != ncomp * cells:
                    raise ValueError(
                        "restart : history '%s' slot %d payload has the wrong size" % (name, slot)
                    )

        cache_nodes = [int(node) for node in d["cache_nodes"]]
        if cache_nodes and not hasattr(self._s, "restore_program_cache"):
            raise RuntimeError("runtime cannot restore the checkpoint's scheduled value cache")
        scheduled_cache_nodes = {
            int(row["node_id"])
            for row in (schedule or {}).get("schedules", ())
            if bool(row["cache_required"])
        }
        unknown_cache_nodes = sorted(set(cache_nodes) - scheduled_cache_nodes)
        if unknown_cache_nodes:
            raise ValueError(
                "restart : scheduled cache node(s) %r do not belong to the installed Program"
                % unknown_cache_nodes
            )
        for node in cache_nodes:
            ncomp = int(d["cache_ncomp_%d" % node])
            ngrow = int(d["cache_ngrow_%d" % node])
            if ncomp <= 0 or ngrow < 0:
                raise ValueError("restart : scheduled cache node %d has invalid metadata" % node)
            if np.asarray(d["cache_value_%d" % node]).size != ncomp * cells:
                raise ValueError(
                    "restart : scheduled cache node %d has the wrong value size" % node
                )
        return _PreparedUniformRestart(d, identity, temporal, cadence)

    def _begin_checkpoint_restart(self) -> None:
        if "_checkpoint_restart_python_snapshot" in self.__dict__:
            raise RuntimeError("Uniform checkpoint restart transaction is already active")
        self._checkpoint_restart_python_snapshot = (
            getattr(self, "_last_restart_identity", None),
            getattr(self, "_last_restart_report", None),
            getattr(self, "_temporal_restart_state", None),
            getattr(self, "_step_controller", None),
        )
        try:
            self._s._begin_restart_transaction()
        except BaseException:
            del self._checkpoint_restart_python_snapshot
            raise

    def _apply_checkpoint_restart(self, prepared: _PreparedUniformRestart) -> Any:
        if type(prepared) is not _PreparedUniformRestart:
            raise TypeError("Uniform restart requires its exact prepared payload")
        import numpy as np
        from pops.runtime._program_cadence_checkpoint import restore_program_cadence
        from pops.runtime._system_io_history import restore_histories

        d = prepared.payload
        for block in (str(value) for value in d["blocks"]):
            self._s.set_state(block, np.asarray(d["state_" + block], dtype=np.float64))
        self._s.set_potential(np.asarray(d["phi"], dtype=np.float64).ravel())
        for index, slot in enumerate(str(value) for value in d["field_provider_slots"]):
            self._s.set_field_potential(
                slot, np.asarray(d["field_potential_%d" % index], dtype=np.float64).ravel()
            )
        self._s.restore_auxiliary_checkpoint_accepted_state(
            np.asarray(d["auxiliary_checkpoint"], dtype=np.uint8).ravel().tobytes()
        )
        macro_step = int(d["macro_step"])
        restore_program_cadence(
            self._s,
            prepared.cadence_state,
            macro_step=macro_step,
            accepted_time=float(d["t"]),
        )
        # Selective history replay executes the installed Program. Restore its authoritative
        # accepted cursor first so time-, schedule- and cursor-dependent nodes see the exact
        # checkpoint context, matching the AMR restart route.
        self._s.set_clock(float(d["t"]), macro_step)
        histories = [str(name) for name in d["history_names"]]
        self._last_restart_report = restore_histories(self._s, d) if histories else None
        cache_names = [str(name) for name in d["cache_names"]]
        for index, node in enumerate(int(value) for value in d["cache_nodes"]):
            self._s.restore_program_cache(
                node,
                int(d["cache_ncomp_%d" % node]),
                int(d["cache_ngrow_%d" % node]),
                int(d["cache_last_update_%d" % node]),
                float(d["cache_accum_dt_%d" % node]),
                cache_names[index],
                np.asarray(d["cache_value_%d" % node], dtype=np.float64),
            )
        self._temporal_restart_state = prepared.temporal_state
        self._step_controller = None
        self._last_restart_identity = prepared.restart_identity
        return prepared.restart_identity

    def _commit_checkpoint_restart(self) -> None:
        # The fallible seal keeps the accepted native snapshot live until collective agreement.
        self._s._commit_restart_transaction()

    def _finalize_checkpoint_restart(self) -> None:
        # Native finalization is noexcept and only releases the already-committed snapshot.
        self._s._finalize_restart_transaction()
        del self._checkpoint_restart_python_snapshot

    def _rollback_checkpoint_restart(self) -> None:
        snapshot = self._checkpoint_restart_python_snapshot
        try:
            self._s._rollback_restart_transaction()
        finally:
            (
                self._last_restart_identity,
                self._last_restart_report,
                self._temporal_restart_state,
                self._step_controller,
            ) = snapshot
            del self._checkpoint_restart_python_snapshot

    def restart(self, path: Any, *, bit_identical: bool = False) -> Any:
        """Restore the direct engine through the native collective transaction protocol."""
        from pops.output._checkpoint_collective import restore_checkpoint_path

        return restore_checkpoint_path(
            self,
            self,
            path,
            bit_identical=bit_identical,
            phase_prefix="Uniform direct-engine restart",
        )
