"""Private implementation of the single runtime value returned by :func:`pops.bind`."""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterable, Mapping
from typing import Any, cast

from pops.codegen._plans import require_install_plan
from pops.fields import LayoutBinding
from pops.identity import Identity, make_identity
from pops.time import TimePoint

from pops.output._consumer_contracts import (
    ConsumerCursorSet,
    ConsumerGraph,
    ConsumerKind,
    ConsumerMoment,
    ScheduleCursor,
    SkipSampleReported,
)
from ._consumer_planning import plan_accepted_side_effects
from ._consumer_transaction import ConsumerTransaction
from ._runtime_component_manifests import component_manifests_for_install
from ._runtime_consumers import RuntimeConsumerPublisher, RuntimeOutputSnapshot, _layout_identity
from ._runtime_executor import install_runtime_executor
from ._runtime_planning import build_runtime_plans
from ._runtime_plan_io import thaw_data
from .run_report import RunReport


def _identity_data(value: Any) -> Any:
    """Detach an optional identity without retaining a live report/runtime object."""
    if value is None:
        return None
    to_data = getattr(value, "to_data", None)
    if callable(to_data):
        return to_data()
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("runtime identity evidence must implement to_data() or be canonical scalar data")


_FIELD_TOPOLOGY_ROW_KEYS = frozenset({
    "patch_identity",
    "topology_digest",
    "provenance",
    "material_points",
    "connected_components",
})
_FIELD_TOPOLOGY_OPTIONAL_ROW_KEYS = frozenset({
    "source_layout_identity",
    "materialized_layout_identity",
})


def _field_topology_rows(executor: Any, slot: str) -> tuple[dict[str, Any], ...]:
    """Read one native topology report through its private inspection seam.

    Multi-layout executors retain one native engine per layout.  Slot ownership is discovered from
    the already-installed native registries; no backend is built and no field array is touched.
    """
    engines = getattr(executor, "_engines", None)
    candidates = tuple(engines.values()) if isinstance(engines, Mapping) else (executor,)
    reports: list[Any] = []
    for candidate in candidates:
        slots = getattr(candidate, "field_provider_slots", None)
        if callable(slots) and slot not in tuple(cast(Iterable[Any], slots())):
            continue
        native = getattr(candidate, "_s", candidate)
        inspect_topology = getattr(native, "_field_topology_report", None)
        if not callable(inspect_topology):
            continue
        reports.append(inspect_topology(slot))
    if len(reports) > 1:
        raise RuntimeError(
            "one qualified field-provider slot is materialized by multiple native executors")
    if not reports:
        return ()
    normalized = []
    for raw in reports[0]:
        if not isinstance(raw, Mapping):
            raise TypeError("native field topology report rows must be mappings")
        row = dict(raw)
        keys = frozenset(row)
        if not _FIELD_TOPOLOGY_ROW_KEYS <= keys or keys - (
                _FIELD_TOPOLOGY_ROW_KEYS | _FIELD_TOPOLOGY_OPTIONAL_ROW_KEYS):
            raise TypeError("native field topology report row has an invalid schema")
        for name in ("patch_identity", "topology_digest", "provenance"):
            if not isinstance(row[name], str) or not row[name]:
                raise TypeError("native field topology report %s must be non-empty" % name)
        for name in ("material_points", "connected_components"):
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError("native field topology report %s must be non-negative" % name)
        for name in _FIELD_TOPOLOGY_OPTIONAL_ROW_KEYS:
            value = row.get(name)
            if value == "":
                value = None
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError("native field topology report %s must be qualified" % name)
            row[name] = value
        normalized.append(row)
    normalized.sort(key=lambda row: row["patch_identity"])
    if len({row["patch_identity"] for row in normalized}) != len(normalized):
        raise RuntimeError("native field topology report contains duplicate patch identities")
    return tuple(normalized)


def _field_solver_configuration(executor: Any, slot: str) -> dict[str, Any] | None:
    """Read the exact installed AMR MG/FAC PODs without exposing the private executor."""
    engines = getattr(executor, "_engines", None)
    candidates = tuple(engines.values()) if isinstance(engines, Mapping) else (executor,)
    reports: list[Any] = []
    for candidate in candidates:
        slots = getattr(candidate, "field_provider_slots", None)
        if callable(slots) and slot not in tuple(cast(Iterable[Any], slots())):
            continue
        native = getattr(candidate, "_s", candidate)
        getter = getattr(native, "field_solver_configuration", None)
        if callable(getter):
            reports.append(getter(slot))
    if len(reports) > 1:
        raise RuntimeError(
            "one qualified field-provider slot has multiple native solver configurations")
    if not reports:
        return None
    row = reports[0]
    if not isinstance(row, Mapping):
        raise TypeError("native field solver configuration must be a mapping")
    result = copy.deepcopy(dict(row))
    expected = {
        "schema_version", "provider_slot", "plan_identity", "solver", "hierarchy", "mg", "fac",
    }
    if set(result) != expected or result["schema_version"] != 1 \
            or result["provider_slot"] != slot \
            or not isinstance(result["plan_identity"], str) \
            or not result["plan_identity"]:
        raise TypeError("native field solver configuration has an invalid schema")
    if not isinstance(result["solver"], str) or not isinstance(result["hierarchy"], str):
        raise TypeError("native field solver configuration requires typed solver identities")
    nested = {
        "mg": {
            "rel_tol", "abs_tol", "max_cycles", "min_coarse", "pre_smooth",
            "post_smooth", "bottom_sweeps", "coarse_threshold",
        },
        "fac": {
            "max_iters", "fine_sweeps", "rel_tol", "abs_tol", "coarse_rel_tol",
            "coarse_abs_tol", "coarse_cycles", "verbose",
        },
    }
    for name, keys in nested.items():
        if not isinstance(result[name], Mapping) or set(result[name]) != keys:
            raise TypeError("native field solver configuration %s has an invalid schema" % name)
        result[name] = dict(result[name])
    # RunReport payloads are semantic evidence and therefore never retain platform binary floats.
    # Preserve the native POD values exactly as canonical binary64 spellings before freeze_data().
    from pops.identity.semantic import semantic_value

    return semantic_value(result, where="native field solver configuration")


def _field_provider_evidence(
    install_plan: Any, layout_plan: Any, executor: Any,
) -> tuple[dict[str, Any], ...]:
    """Return one common, honest report schema for builtin and external field providers."""
    result = []
    for name, field_plan in sorted(install_plan.artifact.plan.field_plans.items()):
        options = field_plan.native_options
        provider = options["solver_provider"]
        provider_kind = provider["provider_kind"]
        if provider_kind not in {"builtin_v1", "external_component_v1"}:
            raise RuntimeError("resolved field plan carries an unknown provider kind")
        slot = options["provider_slot"]
        patches = _field_topology_rows(executor, slot)
        digests = {row["topology_digest"] for row in patches}
        provenances = {row["provenance"] for row in patches}
        materialized_layouts = {
            row["materialized_layout_identity"] for row in patches
            if row["materialized_layout_identity"] is not None
        }
        if len(digests) > 1 or len(provenances) > 1 or len(materialized_layouts) > 1:
            raise RuntimeError(
                "one field-provider materialization returned inconsistent global topology facts")
        topology = provider["topology"]
        if provider_kind == "builtin_v1":
            authority = {
                "kind": "builtin",
                "provider_kind": topology["provider_kind"],
                "declared_provenance": topology["provenance"],
                "declared_topology_digest": topology["topology_digest"],
            }
        else:
            interface = topology["native_interface"]
            authority = {
                "kind": "component",
                "component_id": topology["component_id"],
                "component_manifest_identity": topology["component_manifest_identity"],
                "source_package_identity": topology["source_package_identity"],
                "interface_uri": interface["uri"],
                "interface_version": topology["interface_version"],
            }
        result.append({
            "field": name,
            "provider_slot": slot,
            "provider_kind": provider_kind,
            "source_layout_identity": layout_plan.qualified_id,
            "topology_recipe_identity": provider["topology_recipe_identity"],
            "authority": authority,
            "materialized": bool(patches),
            "materialized_layout_identity": (
                next(iter(materialized_layouts)) if materialized_layouts else None
            ),
            "topology_digest": next(iter(digests)) if digests else None,
            "provenance": next(iter(provenances)) if provenances else None,
            "solver_configuration": _field_solver_configuration(executor, slot),
            "patches": list(patches),
        })
    return tuple(result)


class RuntimeInstance:
    """Authenticated InstallPlan plus its sole native executor and transactional consumers."""

    __slots__ = (
        "_install_plan",
        "_layout_plan",
        "_execution_context",
        "_installed_components",
        "_component_manifests",
        "_runtime_plan",
        "_consumer_graph",
        "_executor",
        "_consumer_cursors",
        "_consumer_reports",
        "_output_root",
        "_attempt",
        "_checkpoint_cursor_override",
        "_snapshot_builder",
        "_publisher",
    )

    def __init__(
        self,
        install_plan: Any,
        *,
        executor: Any = None,
        component_manifests: Any = None,
        publisher: Any = None,
    ) -> None:
        plan = require_install_plan(install_plan)
        manifests = component_manifests_for_install(plan) \
            if component_manifests is None else component_manifests
        runtime_plan = build_runtime_plans(plan, manifests)
        graph = plan.artifact.plan.consumer_graph
        if graph is None:
            graph = ConsumerGraph(())
        if type(graph) is not ConsumerGraph:
            raise TypeError("RuntimeInstance requires an exact ConsumerGraph")
        native = install_runtime_executor(plan, runtime_plan) if executor is None else executor
        if native is None:
            raise TypeError("RuntimeInstance executor cannot be None")
        self._install_plan = plan
        self._layout_plan = plan.artifact.layout_plan
        self._execution_context = plan.execution_context
        self._installed_components = plan.components
        self._component_manifests = manifests
        self._runtime_plan = runtime_plan
        self._consumer_graph = graph
        self._executor = native
        self._consumer_cursors = ConsumerCursorSet()
        self._consumer_reports = ()
        self._output_root = None
        self._attempt = 0
        self._checkpoint_cursor_override = None
        self._snapshot_builder = RuntimeOutputSnapshot(self)
        self._publisher = RuntimeConsumerPublisher(self) if publisher is None else publisher

    @property
    def bind_identity(self) -> Identity:
        """Identity of the authenticated bind transaction."""
        return self._install_plan.bind_identity

    @property
    def consumer_graph(self) -> ConsumerGraph:
        """Resolved public authority for accepted runtime effects."""
        return self._consumer_graph

    @property
    def consumer_cursors(self) -> ConsumerCursorSet:
        """Immutable accepted cursors for the resolved consumer graph."""
        return self._consumer_cursors

    @property
    def last_run_identity(self) -> Any:
        return getattr(self._executor, "last_run_identity", None)

    @property
    def last_restart_identity(self) -> Any:
        return getattr(self._executor, "last_restart_identity", None)

    def layout_identity(self, layout_id: str) -> Identity:
        rows = [row for row in self._layout_plan.layouts
                if row.handle.qualified_id == layout_id]
        if len(rows) != 1:
            raise KeyError("unknown RuntimeInstance layout %s" % layout_id)
        return _layout_identity(rows[0])

    def _executor_for_layout(self, layout_id: str) -> Any:
        selector = getattr(self._executor, "executor_for_layout", None)
        if callable(selector):
            return selector(layout_id)
        rows = [row for row in self._layout_plan.layouts
                if row.handle.qualified_id == layout_id]
        if len(rows) != 1 or len(self._layout_plan.layouts) != 1:
            raise KeyError("unknown RuntimeInstance layout %s" % layout_id)
        return self._executor

    def _executor_for_block(self, block: str) -> Any:
        selector = getattr(self._executor, "executor_for_block", None)
        if callable(selector):
            return selector(block)
        matches = [row for row in self._layout_plan.assignments
                   if row.subject_kind == "block" and row.subject.local_id == block]
        if len(matches) != 1:
            raise KeyError("unknown RuntimeInstance block %s" % block)
        return self._executor_for_layout(matches[0].layout.qualified_id)

    def _output_snapshot(self, manifest: Any, diagnostics: Any = ()) -> Any:
        return self._snapshot_builder.build(manifest, tuple(diagnostics))

    def _checkpoint_identities(self) -> tuple[Identity, Identity, Identity]:
        artifact = self._install_plan.artifact
        return (
            artifact.semantic_identity,
            artifact.artifact_identity,
            self._install_plan.bind_identity,
        )

    # Public read-only runtime surface.  Each route is intentionally named here: there is no
    # generic delegation to the private System/AmrSystem engine.
    def time(self) -> float:
        return float(self._executor.time())

    def macro_step(self) -> int:
        return int(self._executor.macro_step())

    def block_names(self) -> tuple[str, ...]:
        return tuple(self._executor.block_names())

    def get_state(self, block: str) -> Any:
        return self._executor.get_state(block)

    def state_global(self, block: str) -> Any:
        provider = getattr(self._executor, "state_global", None)
        if not callable(provider):
            raise NotImplementedError(
                "this runtime has no single-resolution global state; AMR callers must choose an "
                "explicit level with block_level_state_global(block, level)"
            )
        return provider(block)

    def local_boxes(self, block: str) -> tuple[tuple[int, int, int, int], ...]:
        """Return this rank's exact native uniform boxes in global index coordinates."""
        provider = getattr(self._executor, "local_boxes", None)
        if not callable(provider):
            raise NotImplementedError(
                "this runtime provider does not expose rank-owned local boxes"
            )
        result: list[tuple[int, int, int, int]] = []
        for raw_box in cast(Iterable[Iterable[Any]], provider(block)):
            box = tuple(int(value) for value in raw_box)
            if len(box) != 4:
                raise TypeError("native local box rows must contain exactly four indices")
            result.append(cast(tuple[int, int, int, int], box))
        return tuple(result)

    def local_state(self, block: str, box_index: int) -> Any:
        """Return the native state owned by one box from :meth:`local_boxes`."""
        if isinstance(box_index, bool) or not isinstance(box_index, int) or box_index < 0:
            raise TypeError("local_state box_index must be a non-negative integer")
        provider = getattr(self._executor, "local_state", None)
        if not callable(provider):
            raise NotImplementedError(
                "this runtime provider does not expose rank-owned local state"
            )
        return provider(block, box_index)

    def nx(self) -> int:
        return int(self._executor.nx())

    def ny(self) -> int:
        return int(self._executor.ny())

    def n_levels(self) -> int:
        return int(self._executor.n_levels())

    def patch_boxes(self) -> Any:
        return self._executor.patch_boxes()

    def patch_rectangles(self) -> Any:
        return self._executor.patch_rectangles()

    def block_level_state(self, block: str, level: int) -> Any:
        return self._executor.block_level_state(block, level)

    def block_level_state_global(self, block: str, level: int) -> Any:
        return self._executor.block_level_state_global(block, level)

    def field_provider_slots(self) -> tuple[str, ...]:
        return tuple(self._executor.field_provider_slots())

    def field_provider_levels(self, slot: str) -> int:
        return int(self._executor.field_provider_levels(slot))

    def field_potential_global(self, slot: str) -> Any:
        return self._executor.field_potential_global(slot)

    def field_potential_level_global(self, slot: str, level: int) -> Any:
        return self._executor.field_potential_level_global(slot, level)

    def history_names(self) -> tuple[str, ...]:
        return tuple(self._executor.history_names())

    def history_depth(self, name: str) -> int:
        return int(self._executor.history_depth(name))

    def history_ncomp(self, name: str) -> int:
        return int(self._executor.history_ncomp(name))

    def history_global(self, name: str, slot: int) -> Any:
        return self._executor.history_global(name, slot)

    def installed_program_hash(self) -> str:
        return str(self._executor.installed_program_hash())

    def program_report(self) -> Any:
        return self._executor.program_report()

    @property
    def amr(self) -> Any:
        """Read-only AMR hierarchy/report view supplied by an adaptive executor."""
        return self._executor.amr

    def inspect(self) -> Any:
        """Return one array-free report spanning the accepted runtime and its install contract."""
        from pops.runtime.inspection import build_runtime_inspection

        layouts = tuple(self._layout_plan.layouts)
        adaptive = any(row.adaptive for row in layouts)
        return build_runtime_inspection(
            self._executor,
            runtime="adaptive" if adaptive else "uniform",
            adaptive=adaptive,
            instance={
                "bind_identity": self.bind_identity.to_data(),
                "artifact_identity": self._install_plan.artifact.artifact_identity.to_data(),
                "plan_identity": self._install_plan.artifact.plan.plan_identity.to_data(),
                "layout_plan": self._layout_plan.inspect(),
                "execution_context": self._execution_context.to_data(),
                "installed_components": [
                    component.to_data()
                    for component in self._installed_components.values()
                ],
                "consumer_graph": self._consumer_graph.to_data(),
                "consumer_cursors": self._consumer_cursors.to_data(),
                "consumer_reports": [report.to_data() for report in self._consumer_reports],
                "attempt": self._attempt,
                "output_root": None if self._output_root is None else str(self._output_root),
                "last_run_identity": _identity_data(self.last_run_identity),
                "last_restart_identity": _identity_data(self.last_restart_identity),
                "field_providers": list(_field_provider_evidence(
                    self._install_plan, self._layout_plan, self._executor)),
            },
        )

    def _layout_bindings(self) -> tuple[LayoutBinding, ...]:
        generation = 0
        counter = getattr(self._executor, "checkpoint_topology_epoch", None)
        if callable(counter):
            generation = int(cast(Any, counter()))
        return tuple(LayoutBinding(row.handle, generation) for row in self._layout_plan.layouts)

    def _moments(self, *, at_start: bool = False, at_end: bool = False) -> tuple[ConsumerMoment, ...]:
        clocks = {row.schedule.domain.clock for row in self._consumer_graph.nodes}
        native = self._executor
        temporal = getattr(native, "_temporal_restart_state", None)
        if clocks and temporal is None:
            raise RuntimeError(
                "RuntimeInstance consumers require accepted qualified temporal clock state")
        temporal_state = cast(Any, temporal)
        accepted_step = int(native.macro_step())
        moments = []
        for clock in sorted(clocks, key=lambda value: value.qualified_id):
            cursor = temporal_state.cursor_for_clock(clock)
            moments.append(ConsumerMoment(
                TimePoint(clock, step=int(cursor["tick"])),
                accepted_step=accepted_step,
                attempt=self._attempt,
                clock_tick=int(cursor["tick"]),
                wall_tick=accepted_step,
                layouts=self._layout_bindings(),
                at_start=at_start,
                at_end=at_end,
            ))
        return tuple(moments)

    def _stage_consumers(
        self, *, at_start: bool = False, at_end: bool = False,
    ) -> tuple[ConsumerTransaction, ...]:
        plans = tuple(
            plan_accepted_side_effects(
                self._runtime_plan, self._consumer_graph, moment, self._consumer_cursors)
            for moment in self._moments(at_start=at_start, at_end=at_end)
        )
        plans = tuple(plan for plan in plans if plan.effects)
        all_effects = tuple(effect for plan in plans for effect in plan.effects)
        checkpoint_ids = {
            row.qualified_id for row in self._consumer_graph.nodes
            if row.kind is ConsumerKind.CHECKPOINT
        }
        checkpoint_effects = tuple(
            effect for effect in all_effects if effect.consumer_id in checkpoint_ids)
        if checkpoint_effects:
            if len(checkpoint_effects) != 1 or all_effects[-1] is not checkpoint_effects[0]:
                raise ValueError(
                    "an accepted transaction may publish exactly one checkpoint and it must be "
                    "the final ConsumerGraph effect"
                )
            if any(type(effect.failure_action) is SkipSampleReported for effect in all_effects):
                raise ValueError(
                    "a checkpoint transaction cannot predict restart cursors when another "
                    "effect may skip its sample"
                )
            predicted = self._consumer_cursors
            for effect in all_effects:
                predicted = predicted.replace(effect.cursor_after)
            self._checkpoint_cursor_override = predicted

        staged = []
        try:
            for plan in plans:
                staged.append(ConsumerTransaction(
                    plan, self._consumer_cursors, self._publisher))
        except BaseException as error:
            cleanup_error = self._abort_consumers(tuple(staged))
            if cleanup_error is not None:
                raise cleanup_error from error
            raise
        finally:
            self._checkpoint_cursor_override = None
        return tuple(staged)

    @staticmethod
    def _abort_consumers(transactions: tuple[ConsumerTransaction, ...]) -> BaseException | None:
        failure = None
        for transaction in reversed(transactions):
            try:
                transaction.abort()
            except BaseException as error:
                if failure is None:
                    failure = error
        return failure

    def _accept_consumers(
        self, transactions: tuple[ConsumerTransaction, ...],
    ) -> tuple[tuple[Any, ...], ConsumerCursorSet, tuple[Any, ...]]:
        reports = tuple(transaction.accept() for transaction in transactions)
        cursors = self._consumer_cursors
        for transaction in transactions:
            for cursor in transaction.cursor_updates:
                cursors = cursors.replace(cursor)
        return reports, cursors, self._consumer_reports + reports

    def _fire_consumers(self, *, at_start: bool = False, at_end: bool = False) -> tuple[Any, ...]:
        transactions = self._stage_consumers(at_start=at_start, at_end=at_end)
        try:
            reports, cursors, all_reports = self._accept_consumers(transactions)
        except BaseException as error:
            cleanup_error = self._abort_consumers(transactions)
            if cleanup_error is not None:
                raise cleanup_error from error
            raise
        self._consumer_cursors = cursors
        self._consumer_reports = all_reports
        for transaction in transactions:
            transaction.seal()
        return reports

    def _step_transaction_methods(self) -> tuple[Any, Any, Any, Any]:
        native = self._executor
        methods = tuple(getattr(native, name, None) for name in (
            "_begin_step_transaction", "_commit_step_transaction",
            "_finalize_step_transaction", "_rollback_step_transaction",
        ))
        if any(not callable(method) for method in methods):
            raise TypeError(
                "RuntimeInstance executor must implement the native step-transaction protocol"
            )
        return cast(tuple[Any, Any, Any, Any], methods)

    def _step_envelope_snapshot(self) -> dict[str, Any]:
        native = self._executor
        return {
            "attempt": self._attempt,
            "consumer_cursors": self._consumer_cursors,
            "consumer_reports": self._consumer_reports,
            "checkpoint_cursor_override": self._checkpoint_cursor_override,
            "temporal_restart_state": copy.deepcopy(
                getattr(native, "_temporal_restart_state", None)),
            "step_controller": copy.deepcopy(getattr(native, "_step_controller", None)),
            "last_step_transaction_report": getattr(
                native, "_last_step_transaction_report", None),
        }

    def _restore_step_envelope(self, snapshot: dict[str, Any]) -> None:
        native = self._executor
        self._attempt = snapshot["attempt"]
        self._consumer_cursors = snapshot["consumer_cursors"]
        self._consumer_reports = snapshot["consumer_reports"]
        self._checkpoint_cursor_override = snapshot["checkpoint_cursor_override"]
        if hasattr(native, "_temporal_restart_state"):
            native._temporal_restart_state = snapshot["temporal_restart_state"]
        if hasattr(native, "_step_controller"):
            native._step_controller = snapshot["step_controller"]
        if hasattr(native, "_last_step_transaction_report"):
            native._last_step_transaction_report = snapshot["last_step_transaction_report"]

    def _accepted_step_transaction(
        self,
        advance: Any,
        *,
        at_end: Any = False,
    ) -> Any:
        """Advance once and publish its due effects as one rollback boundary."""
        from pops.time import StepTransactionReport

        native = self._executor
        begin, commit, finalize, rollback = self._step_transaction_methods()
        snapshot = self._step_envelope_snapshot()
        phase = "begin"
        attempts = 1
        failure_report = None
        transactions = ()
        native_active = False
        begin_entered = False
        try:
            # ``begin`` is part of the failure boundary: a native backend may
            # have captured or mutated provisional stores before reporting a
            # failure.  Once entered, rollback must therefore be attempted even
            # when ``begin`` itself does not return.
            begin_entered = True
            begin()
            native_active = True
            phase = "solve"
            result, attempts = advance()
            if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
                raise RuntimeError("step controller returned an invalid native-attempt count")
            phase = "effect"
            self._attempt += attempts
            transactions = self._stage_consumers(
                at_end=bool(at_end() if callable(at_end) else at_end))
            phase = "commit"
            commit()
            phase = "effect"
            reports, cursors, all_reports = self._accept_consumers(transactions)
            phase = "commit"
            finalize()
            native_active = False
            self._consumer_cursors = cursors
            self._consumer_reports = all_reports
            for transaction in transactions:
                transaction.seal()
            return result
        except BaseException as error:
            failure_report = getattr(native, "_last_step_transaction_report", None)
            cleanup_error = self._abort_consumers(transactions)
            try:
                if native_active or begin_entered:
                    try:
                        rollback()
                    except BaseException as rollback_error:
                        # Preserve the initiating error.  Backends are required
                        # to make rollback safe after entering begin; surfacing
                        # that secondary contract violation as a note avoids
                        # replacing the actionable root cause.
                        add_note = getattr(error, "add_note", None)
                        if callable(add_note):
                            add_note(
                                "step-transaction rollback also failed: "
                                f"{rollback_error}")
            finally:
                self._restore_step_envelope(snapshot)
            if phase in {"effect", "commit"}:
                transaction_plan = getattr(native, "_step_transaction_plan", None)
                stores = tuple(
                    store.value
                    for store in cast(
                        Iterable[Any], getattr(transaction_plan, "stores", ()))
                ) if transaction_plan is not None else ()
                failure_report = StepTransactionReport(
                    status="failed",
                    phase=phase,
                    action="fail_run",
                    attempts=attempts,
                    staged_effects=stores,
                    rolled_back_effects=stores,
                    diagnostics=(str(error),),
                )
            if failure_report is not None and hasattr(native, "_last_step_transaction_report"):
                native._last_step_transaction_report = failure_report
            if cleanup_error is not None:
                raise cleanup_error from error
            raise

    def _run(self, t_end: Any, *, max_steps: int = 1_000_000,
             output_dir: Any = None, **controller_controls: Any) -> RunReport:
        if "strategy" in controller_controls or "cfl" in controller_controls:
            raise TypeError(
                "RuntimeInstance._run does not accept strategy= or cfl=; declare the controller "
                "with Program.step_strategy(...)"
            )
        from pops.runtime._step_strategy import (
            prepare_step_controller, resolve_run_strategy, run_control_payload, run_step_attempt)
        from pops.runtime.run_report import RunStopReason

        native = self._executor
        selected = resolve_run_strategy(native)
        control = run_control_payload(selected, controller_controls)
        prepare_step_controller(native, selected, controller_controls)
        self._step_transaction_methods()
        temporal = getattr(native, "_temporal_restart_state", None)
        if temporal is not None:
            temporal.begin_run(control, time=native.time(), macro_step=native.macro_step())
        from pops.runtime._run_manifest import begin_run

        manifest = begin_run(
            native,
            t_end=t_end,
            step_transaction=control,
            max_steps=max_steps,
            output_dir=output_dir,
        )
        previous_root, self._output_root = self._output_root, output_dir
        steps = 0
        rejected_steps = 0
        try:
            self._fire_consumers(at_start=True)
            while native.time() < t_end and steps < max_steps:
                def advance() -> tuple[Any, int]:
                    report = run_step_attempt(
                        native, native, selected, t_end=float(t_end),
                        controls=controller_controls)
                    return report, report.attempts

                step_report = self._accepted_step_transaction(
                    advance,
                    at_end=lambda: not (native.time() < t_end),
                )
                rejected_steps += int(step_report.attempts) - 1
                steps += 1
            if native.time() < t_end:
                raise RuntimeError(
                    "max_steps exhausted before t_end: "
                    f"accepted {steps} step(s), reached t={native.time()!r}, "
                    f"requested t_end={t_end!r}")
            if steps == 0:
                self._fire_consumers(at_end=True)
        finally:
            self._output_root = previous_root
        return RunReport(
            accepted_steps=steps,
            rejected_steps=rejected_steps,
            final_time=native.time(),
            final_macro_step=native.macro_step(),
            stop_reason=RunStopReason.TARGET_TIME_REACHED,
            run_identity=manifest.run_identity,
            bind_identity=self.bind_identity,
            execution_identity=self._execution_context.identity,
            artifact_identity=self._install_plan.artifact.artifact_identity,
            field_providers=_field_provider_evidence(
                self._install_plan, self._layout_plan, self._executor),
        )

    def _checkpoint_payload(self, path: Any) -> str:
        target = self._executor.checkpoint(str(path))
        import numpy as np
        from ._checkpoint_manifest import IDENTITY_KEY, MANIFEST_KEY, seal_checkpoint_payload

        with np.load(target, allow_pickle=False) as stored:
            old_manifest = json.loads(str(stored[MANIFEST_KEY]))
            payload = {
                name: np.asarray(stored[name]).copy()
                for name in stored.files if name not in {MANIFEST_KEY, IDENTITY_KEY}
            }
        payload["runtime_consumer_graph"] = np.asarray(self._consumer_graph.identity.token)
        cursors = self._checkpoint_cursor_override or self._consumer_cursors
        payload["runtime_consumer_cursors"] = np.asarray(json.dumps(
            cursors.to_data(), sort_keys=True, separators=(",", ":")))
        seal_checkpoint_payload(self, payload, runtime_kind=old_manifest["runtime_kind"])
        temporary = str(target) + ".runtime-instance.tmp"
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, target)
        return target

    def _restart_operation(self) -> Any:
        manifests = tuple(
            row for row in self._consumer_graph.nodes if row.kind is ConsumerKind.CHECKPOINT)
        if not manifests:
            from pops.output._restart_provider import RestartV3

            return RestartV3()
        evidence = {
            make_identity("restart-provider", thaw_data(row.operation_data)).token
            for row in manifests
        }
        if len(evidence) != 1:
            raise ValueError("RuntimeInstance has multiple incompatible restart providers")
        return manifests[0].operation

    def checkpoint(self, path: Any) -> str:
        target = str(path)
        operation = self._restart_operation()
        extension = operation.consumer_data()["extension"]
        if not target.endswith(extension):
            target += extension
        snapshot = operation.snapshot(self, os.path.dirname(target) or ".")
        return str(operation.write(snapshot, target))

    def _reopen_checkpoint(self, path: Any) -> tuple[str, ConsumerCursorSet]:
        import numpy as np

        target = str(path) if str(path).endswith(".npz") else str(path) + ".npz"
        with np.load(target, allow_pickle=False) as stored:
            required = {"runtime_consumer_graph", "runtime_consumer_cursors"}
            if not required <= set(stored.files):
                raise ValueError("restart checkpoint lacks RuntimeInstance consumer state")
            if str(stored["runtime_consumer_graph"]) != self._consumer_graph.identity.token:
                raise ValueError("restart ConsumerGraph identity differs from the installed graph")
            cursor_data = json.loads(str(stored["runtime_consumer_cursors"]))
        if set(cursor_data) != {"schema_version", "rows"} or cursor_data["schema_version"] != 1:
            raise ValueError("restart consumer cursor schema is unsupported")
        cursors = ConsumerCursorSet(tuple(ScheduleCursor(
            row["consumer_id"], row["last_occurrence"], row["committed_samples"])
            for row in cursor_data["rows"]))
        return target, cursors

    def _restore_checkpoint(self, target: Any, cursors: ConsumerCursorSet) -> Any:
        result = self._executor.restart(str(target))
        self._consumer_cursors = cursors
        return result

    def restart(self, path: Any) -> Any:
        operation = self._restart_operation()
        reopened = operation.reopen(self, path)
        return operation.restore(self, reopened)

    def __str__(self) -> str:
        return "RuntimeInstance(layouts=%d, blocks=%d, consumers=%d)" % (
            len(self._layout_plan.layouts), len(self._install_plan.instances),
            len(self._consumer_graph.nodes))


__all__ = ["RuntimeInstance"]
