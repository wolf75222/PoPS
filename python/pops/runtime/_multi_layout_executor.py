"""Exact multi-layout Uniform runtime coordination and transactional persistence."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pops.runtime._component_execution_context import component_execution_data


@dataclass(frozen=True, slots=True)
class _PreparedMultiLayoutRestart:
    restart_identity: Any
    mapping: dict[str, int]
    children: tuple[Any, ...]


def _common_exact(values: Any, *, where: str) -> Any:
    rows = tuple(values)
    if not rows:
        raise ValueError("%s requires at least one value" % where)
    first = rows[0]
    if any(row != first for row in rows[1:]):
        raise ValueError("%s differs across materialized layouts" % where)
    return first


class _LayoutCompiledView:
    """Per-layout compiled view carrying the aggregate identities and exact sliced binary."""

    def __init__(self, artifact: Any, layout_program: Any) -> None:
        self._artifact = artifact
        self._layout_program = layout_program
        self.program = layout_program.program.program
        self.bind_schema = artifact.bind_schema
        self.semantic_identity = artifact.semantic_identity
        self.artifact_identity = artifact.artifact_identity
        self.so_path = layout_program.program.so_path
        self.target = layout_program.target

    def arguments(self) -> Any:
        from pops.codegen.inspect_compiled import build_layout_arguments

        return build_layout_arguments(self._artifact, self._layout_program.layout_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._layout_program.program, name)


def _block_layouts(plan: Any) -> dict[str, str]:
    return {
        row.subject.local_id: row.layout.qualified_id
        for row in plan.artifact.layout_plan.assignments
        if row.subject_kind == "block"
    }


def _mapping_block(subject: Any) -> str:
    block = getattr(subject, "block_ref", None)
    name = getattr(block, "local_id", None)
    if not isinstance(name, str) or not name:
        raise NotImplementedError(
            "multi-layout native transfer requires block-qualified state ports"
        )
    return name


def _require_unique_transfer_targets(transfers: Any) -> None:
    """Defend install against order-dependent overwrite transfers in a forged runtime plan."""
    writers: dict[tuple[str, str, str], str] = {}
    for transfer in transfers:
        if transfer.operation_abi != 1:
            continue
        key = (transfer.target_layout_id, transfer.target_subject_id, transfer.synchronization_uri)
        previous = writers.get(key)
        if previous is not None:
            raise ValueError(
                "runtime Transfer plan has concurrent overwrite mappings %s and %s for one "
                "target/synchronization; an explicit merge protocol is required"
                % (previous, transfer.mapping_id)
            )
        writers[key] = transfer.mapping_id


def _require_conservative_cell_average_geometry(source: Any, target: Any) -> None:
    """Authenticate the geometric domain represented by the v1 averaging operation.

    ``CONSERVATIVE_CELL_AVERAGE_V1`` has refinement ratios and field extents, but no coordinate
    transform.  It therefore represents nested resolutions of one exact physical Cartesian domain,
    never interpolation between unrelated domains.
    """
    if float(source.L) != float(target.L):
        raise ValueError(
            "CONSERVATIVE_CELL_AVERAGE_V1 requires identical physical extents; select a mapped "
            "Transfer operation/provider for distinct geometries"
        )
    if bool(source.periodic) != bool(target.periodic):
        raise ValueError(
            "CONSERVATIVE_CELL_AVERAGE_V1 requires identical boundary topology; select a mapped "
            "Transfer operation/provider for distinct topologies"
        )


def _require_runtime_plan_bundle(plan: Any, runtime_plan: Any) -> None:
    """Authenticate the exact bundle and its Transfer projection against one InstallPlan."""
    from pops.identity import make_identity
    from pops.runtime._runtime_plan_contracts import LayoutTransfer, RuntimePlanBundle

    if type(runtime_plan) is not RuntimePlanBundle:
        raise TypeError("multi-layout install requires an exact RuntimePlanBundle")
    if runtime_plan.identity != make_identity("runtime-plan-bundle", runtime_plan._payload()):
        raise ValueError("RuntimePlanBundle identity does not authenticate its payload")
    expected_identities = (
        (runtime_plan.install_identity, plan.bind_identity, "bind"),
        (runtime_plan.platform_identity, plan.artifact.platform_manifest.identity, "platform"),
        (
            runtime_plan.execution_context_identity,
            plan.execution_context.identity,
            "execution context",
        ),
    )
    for actual, expected, label in expected_identities:
        if actual != expected:
            raise ValueError("RuntimePlanBundle %s identity differs from InstallPlan" % label)
    layout_plan = plan.artifact.layout_plan
    if runtime_plan.layout_plan_id != layout_plan.qualified_id:
        raise ValueError("RuntimePlanBundle layout identity differs from compiled LayoutPlan")
    transfers = tuple(runtime_plan.communication.transfers)
    mapping_ids = tuple(row.mapping_id for row in transfers)
    if len(mapping_ids) != len(set(mapping_ids)):
        raise ValueError("RuntimePlanBundle contains duplicate Transfer mapping identities")
    expected_transfers = tuple(
        LayoutTransfer(
            row.requirement.qualified_id,
            row.provider_id,
            row.provider_identity["component_id"],
            row.requirement.source_layout.qualified_id,
            row.requirement.target_layout.qualified_id,
            row.requirement.source_port.subject.qualified_id,
            row.requirement.target_port.subject.qualified_id,
            row.requirement.source_port.representation.value,
            row.requirement.target_port.representation.value,
            int(row.requirement.operation),
            row.requirement.synchronization.value,
        )
        for row in layout_plan.mappings
    )
    if transfers != expected_transfers:
        raise ValueError(
            "RuntimePlanBundle Transfers differ from the authenticated compiled LayoutPlan"
        )
    _require_unique_transfer_targets(transfers)


def _transfer_descriptor(values: Any, *, layout_id: str, block: str) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("native layout transfer values must have (components, axis0, axis1) shape")
    return {
        "values": array,
        "dimension": 2,
        "extents": (int(array.shape[1]), int(array.shape[2]), 1),
        "layout_identity": layout_id,
        "patch_identity": block,
        "centering": 1,
        "centering_axes": 0,
        "ghost_lower": (0, 0, 0),
        "ghost_upper": (0, 0, 0),
        "scalar_type": 2,
        "memory_space": 1,
        "ownership": 1,
    }


class _CompositeTemporalRestartState:
    """Broadcast temporal mutations and prove every layout clock stays identical."""

    def __init__(self, states: Any) -> None:
        self.states = tuple(states)
        if not self.states:
            raise ValueError("composite temporal state requires one state per layout")

    def _same_attribute(self, name: str) -> Any:
        values = tuple(getattr(state, name) for state in self.states)
        if any(value != values[0] for value in values[1:]):
            raise RuntimeError("per-layout temporal state diverged at %s" % name)
        return values[0]

    @property
    def _restored_pending(self) -> Any:
        return self._same_attribute("_restored_pending")

    @property
    def controller_state(self) -> Any:
        return self._same_attribute("controller_state")

    @property
    def program_schedule(self) -> Any:
        return self._same_attribute("program_schedule")

    def to_data(self) -> dict[str, Any]:
        """Project one temporal report only after proving every layout is identical."""
        rows = tuple(state.to_data() for state in self.states)
        return _common_exact(rows, where="multi-layout temporal report")

    def _broadcast(self, name: str, **kwargs: Any) -> None:
        for state in self.states:
            getattr(state, name)(**kwargs)

    def begin_run(self, strategy: Any, *, time: Any, macro_step: Any) -> None:
        for state in self.states:
            state.begin_run(strategy, time=time, macro_step=macro_step)

    def before_attempt(self, *, time: Any, macro_step: Any) -> None:
        self._broadcast("before_attempt", time=time, macro_step=macro_step)

    def accept(self, **kwargs: Any) -> None:
        self._broadcast("accept", **kwargs)

    def reject(self, **kwargs: Any) -> None:
        self._broadcast("reject", **kwargs)

    def fail(self, **kwargs: Any) -> None:
        self._broadcast("fail", **kwargs)

    def cursor_for_clock(self, clock: Any) -> Any:
        values = tuple(state.cursor_for_clock(clock) for state in self.states)
        if any(value != values[0] for value in values[1:]):
            raise RuntimeError("per-layout temporal cursors diverged")
        return values[0]


class _MultiLayoutUniformExecutor:
    """Atomic coordinator for independently compiled Uniform layout Systems."""

    def __init__(
        self, plan: Any, runtime_plan: Any, engines: dict[str, Any], blocks: dict[str, str]
    ) -> None:
        self._plan = plan
        self._execution_context = plan.execution_context
        self._runtime_plan = runtime_plan
        self._engines = dict(engines)
        self._block_layouts = dict(blocks)
        self._mapping_evaluations = {
            row.mapping_id: 0 for row in runtime_plan.communication.transfers
        }
        self._mapping_snapshot = None
        self._last_run_manifest = None
        self._last_run_identity = None
        self._last_restart_identity = None
        self._step_strategy = _common_exact(
            (engine._step_strategy for engine in self._engines.values()),
            where="multi-layout step strategy",
        )
        self._step_transaction_plan = _common_exact(
            (engine._step_transaction_plan for engine in self._engines.values()),
            where="multi-layout transaction plan",
        )
        self._temporal_restart_state = _CompositeTemporalRestartState(
            engine._temporal_restart_state for engine in self._engines.values()
        )
        self._step_controller = None
        self._last_step_transaction_report = None
        from pops.runtime._bound_snapshot import MultiLayoutBoundSnapshot

        snapshot = MultiLayoutBoundSnapshot(
            plan, tuple(engine.bound_snapshot for engine in self._engines.values())
        )
        self._bound_snapshot = snapshot
        for engine in self._engines.values():
            engine._bound_snapshot = snapshot

    @property
    def bound_snapshot(self) -> Any:
        return self._bound_snapshot

    def _checkpoint_identities(self) -> tuple[Any, Any, Any]:
        return (
            self._bound_snapshot.semantic_identity,
            self._bound_snapshot.artifact_identity,
            self._bound_snapshot.bind_identity,
        )

    @property
    def last_run_identity(self) -> Any:
        return self._last_run_identity

    @property
    def last_restart_identity(self) -> Any:
        return self._last_restart_identity

    def executor_for_layout(self, layout_id: str) -> Any:
        try:
            return self._engines[layout_id]
        except KeyError:
            raise KeyError("unknown RuntimeInstance layout %s" % layout_id) from None

    def executor_for_block(self, block: str) -> Any:
        try:
            return self._engines[self._block_layouts[block]]
        except KeyError:
            raise KeyError("unknown RuntimeInstance block %s" % block) from None

    def block_names(self) -> tuple[str, ...]:
        return tuple(self._block_layouts)

    def state_global(self, block: str) -> Any:
        return self.executor_for_block(block).state_global(block)

    def get_state(self, block: str) -> Any:
        return self.executor_for_block(block).get_state(block)

    def set_state(self, block: str, values: Any) -> Any:
        return self.executor_for_block(block).set_state(block, values)

    def nx(self) -> int:
        raise ValueError("multi-layout geometry requires executor_for_layout(layout_id).nx()")

    def ny(self) -> int:
        raise ValueError("multi-layout geometry requires executor_for_layout(layout_id).ny()")

    def _common_clock(self, method: str) -> Any:
        values = tuple(getattr(engine, method)() for engine in self._engines.values())
        if any(value != values[0] for value in values[1:]):
            raise RuntimeError("multi-layout native clocks diverged at %s" % method)
        return values[0]

    def time(self) -> float:
        return float(self._common_clock("time"))

    def macro_step(self) -> int:
        return int(self._common_clock("macro_step"))

    def _native_step_target(self) -> Any:
        """The coordinator itself is the raw target for one composite attempt."""
        return self

    def _mapping_blocks(self, transfer: Any) -> tuple[str, str]:
        matches = tuple(
            row
            for row in self._plan.artifact.layout_plan.mappings
            if row.requirement.qualified_id == transfer.mapping_id
        )
        if len(matches) != 1:
            raise RuntimeError(
                "runtime Transfer must resolve to exactly one authenticated layout mapping"
            )
        requirement = matches[0].requirement
        return (
            _mapping_block(requirement.source_port.subject),
            _mapping_block(requirement.target_port.subject),
        )

    def _capture_mapping_source(self, transfer: Any) -> dict[str, Any]:
        source_block, _target_block = self._mapping_blocks(transfer)
        source_engine = self.executor_for_layout(transfer.source_layout_id)
        import numpy as np

        # Force an owned copy.  Every before-step Transfer reads the same pre-transfer state even
        # when another mapping writes this layout first (A->B->C and explicit cycles included).
        source_flat = np.array(
            source_engine.state_global(source_block), dtype=np.float64, copy=True, order="C"
        )
        source_shape = (
            source_flat.size // (source_engine.nx() * source_engine.ny()),
            source_engine.ny(),
            source_engine.nx(),
        )
        return _transfer_descriptor(
            source_flat.reshape(source_shape),
            layout_id=transfer.source_layout_id,
            block=source_block,
        )

    def _apply_mapping(self, transfer: Any, source: dict[str, Any] | None = None) -> dict[str, Any]:
        source_block, target_block = self._mapping_blocks(transfer)
        target_engine = self.executor_for_layout(transfer.target_layout_id)
        import numpy as np

        if source is None:
            source = self._capture_mapping_source(transfer)
        target_flat = np.asarray(target_engine.state_global(target_block), dtype=np.float64)
        target_shape = (
            target_flat.size // (target_engine.nx() * target_engine.ny()),
            target_engine.ny(),
            target_engine.nx(),
        )
        target = _transfer_descriptor(
            target_flat.reshape(target_shape),
            layout_id=transfer.target_layout_id,
            block=target_block,
        )
        ratios = []
        for source_extent, target_extent in zip(
            source["extents"][:2], target["extents"][:2], strict=True
        ):
            if source_extent % target_extent:
                raise ValueError("layout transfer grids are not integer-aligned fine-to-coarse")
            ratios.append(source_extent // target_extent)
        component = self._plan.components[transfer.component_id]
        apply = getattr(component.native_handle, "_transfer_apply", None)
        if not callable(apply):
            raise TypeError("installed Transfer component exposes no native _transfer_apply seam")
        receipt = apply(
            source,
            target,
            tuple(ratios),
            transfer.operation_abi,
            component_execution_data(self._plan.execution_context),
        )
        if (
            not isinstance(receipt, dict)
            or receipt.get("applied") is not True
            or receipt.get("provider_component_id") != transfer.component_id
        ):
            raise RuntimeError("native Transfer component returned an unauthenticated receipt")
        target_engine.set_state(target_block, target["values"].reshape(-1))
        self._mapping_evaluations[transfer.mapping_id] += 1
        return receipt

    def step(self, dt: float) -> None:
        from pops.runtime._native_step_target import native_step_target

        transfers = tuple(self._runtime_plan.communication.transfers)
        sources = tuple(self._capture_mapping_source(transfer) for transfer in transfers)
        for transfer, source in zip(transfers, sources, strict=True):
            self._apply_mapping(transfer, source)
        for engine in self._engines.values():
            native_step_target(engine).step(dt)
        self._common_clock("time")
        self._common_clock("macro_step")

    def step_cfl(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise NotImplementedError(
            "multi-layout CFL requires a qualified global reduction and is not inferred"
        )

    def mapping_report(self) -> dict[str, int]:
        return dict(self._mapping_evaluations)

    def _begin_step_transaction(self) -> None:
        self._synchronize_child_temporal_states()
        self._mapping_snapshot = dict(self._mapping_evaluations)
        begun = []
        try:
            for engine in self._engines.values():
                engine._begin_step_transaction()
                begun.append(engine)
        except BaseException:
            for engine in reversed(begun):
                engine._rollback_step_transaction()
            self._mapping_snapshot = None
            raise

    def _commit_step_transaction(self) -> None:
        for engine in self._engines.values():
            engine._commit_step_transaction()

    def _finalize_step_transaction(self) -> None:
        # Native System::finalize_step_transaction only checks the already-proved committed
        # precondition and resets a unique_ptr snapshot; after every commit above succeeded it has
        # no fallible numerical/resource operation.  Keeping finalize separate preserves the native
        # two-phase transaction while making the no-fail boundary explicit.
        for engine in self._engines.values():
            engine._finalize_step_transaction()
        self._mapping_snapshot = None

    def _rollback_step_transaction(self) -> None:
        error = None
        for engine in reversed(tuple(self._engines.values())):
            try:
                engine._rollback_step_transaction()
            except BaseException as caught:
                error = error or caught
        if self._mapping_snapshot is not None:
            self._mapping_evaluations = self._mapping_snapshot
        self._mapping_snapshot = None
        if error is not None:
            raise error

    def checkpoint_topology_epoch(self) -> int:
        return 0

    def _synchronize_child_temporal_states(self) -> None:
        states = tuple(self._temporal_restart_state.states)
        if len(states) != len(self._engines):
            raise RuntimeError("composite temporal state count differs from native layouts")
        for engine, state in zip(self._engines.values(), states, strict=True):
            engine._temporal_restart_state = state

    def _restore_temporal_restart_state(self, state: Any) -> None:
        """Restore the coordinator envelope and every child authority atomically."""
        if not isinstance(state, _CompositeTemporalRestartState):
            raise TypeError("multi-layout temporal restore requires a composite state")
        self._temporal_restart_state = state
        self._synchronize_child_temporal_states()

    def _rebuild_composite_temporal_state(self) -> None:
        self._temporal_restart_state = _CompositeTemporalRestartState(
            engine._temporal_restart_state for engine in self._engines.values()
        )
        self._common_clock("time")
        self._common_clock("macro_step")

    @staticmethod
    def _result_evidence(result: Any) -> Any:
        to_data = getattr(result, "to_data", None)
        return to_data() if callable(to_data) else result

    def _checkpoint_children(
        self, root: str, prefix: str, topology: Any, *, retain_payloads: bool,
    ) -> tuple[tuple[str, ...], tuple[bytes, ...]]:
        """Capture every child collectively; only rank zero may read the resulting files."""
        from pops.output._checkpoint_collective import (
            canonical_checkpoint_path,
            consensus,
            root_effect,
        )
        from pops.runtime._checkpoint_manifest import authenticate_checkpoint_payload
        import numpy as np

        sync_error = None
        try:
            self._synchronize_child_temporal_states()
        except BaseException as error:
            sync_error = error
        consensus(topology, "%s temporal synchronization" % prefix, error=sync_error)
        paths = []
        payloads = []
        for index, engine in enumerate(self._engines.values()):
            engine._last_run_manifest = self._last_run_manifest
            engine._last_run_identity = self._last_run_identity
            expected = canonical_checkpoint_path(
                os.path.join(root, "%s-%d" % (prefix, index)))
            produced = None
            capture_error = None
            try:
                produced = canonical_checkpoint_path(engine.checkpoint(str(expected)))
                if produced != expected:
                    raise RuntimeError(
                        "layout %d checkpoint returned %s, expected %s"
                        % (index, produced, expected)
                    )
            except BaseException as error:
                capture_error = error
            rows = consensus(
                topology,
                "%s layout %d capture" % (prefix, index),
                error=capture_error,
                value=None if produced is None else str(produced),
            )
            if any(row["value"] != str(expected) for row in rows):
                raise RuntimeError(
                    "%s layout %d ranks returned different checkpoint paths" % (prefix, index)
                )

            def authenticate_root(
                child_engine: Any = engine, child_path: Path = expected,
            ) -> bytes | None:
                with np.load(child_path, allow_pickle=False) as stored:
                    authenticate_checkpoint_payload(
                        child_engine, stored, runtime_kind="uniform")
                return child_path.read_bytes() if retain_payloads else None

            payload = root_effect(
                topology, "%s layout %d authentication" % (prefix, index),
                authenticate_root,
            )
            paths.append(str(expected))
            if topology.rank == 0 and retain_payloads:
                if not isinstance(payload, bytes):
                    raise RuntimeError("rank zero lost an authenticated child checkpoint payload")
                payloads.append(payload)
        return tuple(paths), tuple(payloads)

    @staticmethod
    def _shared_checkpoint_root(target: Any, topology: Any, purpose: str) -> str:
        from pops.output._checkpoint_collective import root_value

        def create_root() -> str:
            target.parent.mkdir(parents=True, exist_ok=True)
            return tempfile.mkdtemp(
                prefix=".%s.%s." % (target.name, purpose), dir=target.parent)

        return str(root_value(topology, "%s workspace selection" % purpose, create_root))

    @staticmethod
    def _cleanup_checkpoint_root(root: str, topology: Any, purpose: str) -> None:
        from pops.output._checkpoint_collective import root_effect

        root_effect(
            topology, "%s workspace cleanup" % purpose,
            lambda: shutil.rmtree(root, ignore_errors=False),
        )

    def checkpoint(self, path: Any) -> str:
        import numpy as np
        from pops.runtime._engine_descriptors import abi_key
        from pops.runtime._checkpoint_manifest import (
            authenticate_checkpoint_payload,
            seal_checkpoint_payload,
        )
        from pops.output._checkpoint_collective import (
            canonical_checkpoint_path,
            checkpoint_topology,
            consensus,
            root_effect,
        )

        topology = checkpoint_topology(self)
        target = canonical_checkpoint_path(path)
        rows = consensus(topology, "multi-layout target", value=str(target))
        if any(row["value"] != str(target) for row in rows):
            raise ValueError("multi-layout checkpoint target differs across ranks")
        root = self._shared_checkpoint_root(target, topology, "capture")
        try:
            _paths, children = self._checkpoint_children(
                root, "child", topology, retain_payloads=True)

            def write_root() -> None:
                if len(children) != len(self._engines):
                    raise RuntimeError("rank zero did not retain every child checkpoint")
                payload = {
                    "t": np.asarray(self.time()),
                    "macro_step": np.asarray(self.macro_step()),
                    "abi_key": np.asarray(abi_key()),
                    "layout_ids": np.asarray(tuple(self._engines), dtype=np.str_),
                    "mapping_evaluations": np.asarray(json.dumps(
                        self._mapping_evaluations,
                        sort_keys=True,
                        separators=(",", ":"),
                    )),
                }
                for index, child in enumerate(children):
                    payload["layout_checkpoint_%d" % index] = np.frombuffer(
                        child, dtype=np.uint8).copy()
                seal_checkpoint_payload(self, payload, runtime_kind="multi_layout_uniform")
                fd, temporary_name = tempfile.mkstemp(
                    prefix=".%s." % target.name, suffix=".tmp", dir=target.parent)
                os.close(fd)
                temporary = os.fspath(temporary_name)
                try:
                    with open(temporary, "wb") as stream:
                        np.savez_compressed(stream, **payload)
                    os.replace(temporary, target)
                finally:
                    Path(temporary).unlink(missing_ok=True)
                with np.load(target, allow_pickle=False) as stored:
                    authenticate_checkpoint_payload(
                        self, stored, runtime_kind="multi_layout_uniform")

            root_effect(topology, "multi-layout container sealing", write_root)
        finally:
            self._cleanup_checkpoint_root(root, topology, "capture")
        return str(target)

    def _prepare_checkpoint_restart(self, payload: bytes) -> _PreparedMultiLayoutRestart:
        import numpy as np
        from pops.output._checkpoint_collective import decode_checkpoint_bytes
        from pops.runtime._checkpoint_manifest import authenticate_checkpoint_payload

        stored = decode_checkpoint_bytes(payload)
        identity = authenticate_checkpoint_payload(
            self, stored, runtime_kind="multi_layout_uniform")
        layout_ids = tuple(str(value) for value in stored["layout_ids"])
        if layout_ids != tuple(self._engines):
            raise ValueError("checkpoint layout identities differ from RuntimeInstance")
        child_names = tuple(
            "layout_checkpoint_%d" % index for index in range(len(layout_ids)))
        if any(name not in stored.files for name in child_names):
            raise ValueError("checkpoint lacks a per-layout native payload")
        mapping = json.loads(str(stored["mapping_evaluations"]))
        if set(mapping) != set(self._mapping_evaluations) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in mapping.values()
        ):
            raise ValueError("checkpoint mapping counters differ from RuntimeInstance plan")
        prepared_children = []
        for index, (engine, name) in enumerate(zip(
            self._engines.values(), child_names, strict=True
        )):
            prepare = getattr(engine, "_prepare_checkpoint_restart", None)
            if not callable(prepare):
                raise TypeError(
                    "layout %d engine lacks the in-memory restart preflight protocol" % index)
            child_bytes = np.asarray(stored[name], dtype=np.uint8).tobytes()
            prepared_children.append(prepare(child_bytes))
        return _PreparedMultiLayoutRestart(identity, dict(mapping), tuple(prepared_children))

    def _begin_checkpoint_restart(self) -> None:
        if "_checkpoint_restart_snapshot" in self.__dict__:
            raise RuntimeError("multi-layout checkpoint restart transaction is already active")
        self._checkpoint_restart_snapshot = (
            dict(self._mapping_evaluations), self._last_restart_identity,
            self._temporal_restart_state, getattr(self, "_step_controller", None),
        )
        begun = []
        try:
            for engine in self._engines.values():
                begun.append(engine)
                engine._begin_checkpoint_restart()
        except BaseException as begin_error:
            rollback_errors = []
            for engine in reversed(begun):
                try:
                    engine._rollback_checkpoint_restart()
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            mapping, identity, temporal, controller = self.__dict__.pop(
                "_checkpoint_restart_snapshot")
            self._mapping_evaluations = mapping
            self._last_restart_identity = identity
            self._temporal_restart_state = temporal
            self._step_controller = controller
            if rollback_errors:
                raise RuntimeError(
                    "multi-layout child begin rollback failed after %s: %s"
                    % (begin_error, "; ".join(map(str, rollback_errors)))
                ) from begin_error
            raise

    def _apply_checkpoint_restart(self, prepared: _PreparedMultiLayoutRestart) -> Any:
        if type(prepared) is not _PreparedMultiLayoutRestart:
            raise TypeError("multi-layout restart requires its exact prepared payload")
        if len(prepared.children) != len(self._engines):
            raise RuntimeError("multi-layout prepared child count is incomplete")
        for engine, child in zip(self._engines.values(), prepared.children, strict=True):
            engine._apply_checkpoint_restart(child)
        self._mapping_evaluations = dict(prepared.mapping)
        self._rebuild_composite_temporal_state()
        self._last_restart_identity = prepared.restart_identity
        return prepared.restart_identity

    def _commit_checkpoint_restart(self) -> None:
        for engine in self._engines.values():
            engine._commit_checkpoint_restart()

    def _finalize_checkpoint_restart(self) -> None:
        for engine in self._engines.values():
            engine._finalize_checkpoint_restart()
        del self._checkpoint_restart_snapshot

    def _rollback_checkpoint_restart(self) -> None:
        errors = []
        for engine in reversed(tuple(self._engines.values())):
            try:
                engine._rollback_checkpoint_restart()
            except BaseException as error:
                errors.append(error)
        mapping, identity, temporal, controller = self._checkpoint_restart_snapshot
        self._mapping_evaluations = mapping
        self._last_restart_identity = identity
        self._temporal_restart_state = temporal
        self._step_controller = controller
        del self._checkpoint_restart_snapshot
        if errors:
            raise RuntimeError(
                "multi-layout child rollback failed: %s" % "; ".join(map(str, errors)))

    def restart(self, path: Any) -> str:
        from pops.output._checkpoint_collective import (
            canonical_checkpoint_path,
            checkpoint_topology,
            consensus,
            restore_checkpoint_payload,
            root_bytes,
        )

        topology = checkpoint_topology(self)
        target = canonical_checkpoint_path(path)
        rows = consensus(topology, "multi-layout restart target", value=str(target))
        if any(row["value"] != str(target) for row in rows):
            raise ValueError("multi-layout restart target differs across ranks")
        payload = root_bytes(
            topology, "multi-layout restart read", target.read_bytes)
        restore_checkpoint_payload(
            self, self, payload, phase_prefix="multi-layout restart")
        return str(target)


def install_multi_layout_uniform(plan: Any, runtime_plan: Any) -> Any:
    from pops.codegen._layout_resolution import ResolvedRuntimeLayouts
    from pops.runtime._runtime_mesh_lowering import system_config_from_layout
    from pops.runtime._system import System
    from pops.time._step.strategy import FixedDt

    _require_runtime_plan_bundle(plan, runtime_plan)
    layouts = plan.layout
    if type(layouts) is not ResolvedRuntimeLayouts:
        raise TypeError("multi-layout InstallPlan lost its ResolvedRuntimeLayouts authority")
    if plan.aux:
        raise NotImplementedError(
            "multi-layout aux storage requires an explicit layout assignment and transfer plan"
        )
    if plan.artifact.plan.field_plans:
        raise NotImplementedError("multi-layout FieldOperator plans are not executable")
    if any(block.boundaries for block in plan.artifact.plan.blocks):
        raise NotImplementedError(
            "multi-layout boundary components require per-layout boundary install authorities"
        )
    blocks = _block_layouts(plan)
    programs = {row.layout_id: row for row in plan.artifact.layout_programs}
    if set(programs) != {row.handle.qualified_id for row in layouts.plan.layouts}:
        raise ValueError("compiled layout Program set is not exact")
    strategies = []
    transaction_plans = []
    configs = {}
    for row in layouts.rows:
        layout_id = row.handle.qualified_id
        authored = programs[layout_id].program.program
        strategy = getattr(authored, "_step_strategy", None)
        if type(strategy) is not FixedDt:
            raise NotImplementedError(
                "multi-layout execution requires exact FixedDt Program strategy"
            )
        strategies.append(strategy)
        transaction_plans.append(authored.transaction_plan())
        configs[layout_id] = system_config_from_layout(row.descriptor)
    if any(value != strategies[0] for value in strategies[1:]) or any(
        value != transaction_plans[0] for value in transaction_plans[1:]
    ):
        raise ValueError("per-layout Programs do not share one exact transaction strategy")
    transfer_rows = {row.mapping_id: row for row in runtime_plan.communication.transfers}
    if set(transfer_rows) != {
        row.requirement.qualified_id for row in plan.artifact.layout_plan.mappings
    }:
        raise ValueError("runtime transfer plan differs from the resolved LayoutPlan")
    for transfer in transfer_rows.values():
        if (
            transfer.operation_abi != 1
            or transfer.synchronization_uri != "pops://synchronization/before-step@1"
        ):
            raise NotImplementedError("native multi-layout transfer operation is unsupported")
        component = plan.components.get(transfer.component_id)
        apply = getattr(getattr(component, "native_handle", None), "_transfer_apply", None)
        if not callable(apply):
            raise TypeError("mapping Transfer component is not loaded with _transfer_apply")
        source = configs[transfer.source_layout_id]
        target = configs[transfer.target_layout_id]
        _require_conservative_cell_average_geometry(source, target)
        if source.n < target.n or source.n % target.n:
            raise ValueError(
                "CONSERVATIVE_CELL_AVERAGE_V1 requires aligned fine-to-coarse layouts"
            )

    engines = {}
    for row in layouts.rows:
        layout_id = row.handle.qualified_id
        engine = System(configs[layout_id])
        cast(Any, engine)._execution_context = plan.execution_context
        selected = {
            name: spec for name, spec in plan.instances.items() if blocks[name] == layout_id
        }
        view = _LayoutCompiledView(plan.artifact, programs[layout_id])
        engine._install_compiled(
            view, instances=selected, params=plan.params, aux={}, field_plans={}
        )
        engines[layout_id] = engine
    return _MultiLayoutUniformExecutor(plan, runtime_plan, engines, blocks)


__all__ = ["install_multi_layout_uniform"]
