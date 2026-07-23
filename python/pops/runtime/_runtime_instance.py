"""Private implementation of the single runtime value returned by :func:`pops.bind`."""
from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from pops.codegen._plans import require_install_plan
from pops.fields import LayoutBinding
from pops.identity import Identity, make_identity
from pops.output._writers.common import _QuarantineRecovery
from pops.time import TimePoint

from pops.output._consumer_contracts import (
    ConsumerCursorSet,
    ConsumerGraph,
    ConsumerKind,
    ConsumerMoment,
    ScheduleCursor,
    SkipSampleReported,
)
from ._consumer_planning import next_consumer_deadline, plan_accepted_side_effects
from ._consumer_transaction import ConsumerTransaction, ConsumerTransactionReport
from ._output_publisher import preflight_consumer_publication
from ._runtime_component_manifests import component_manifests_for_install
from ._runtime_consumers import RuntimeConsumerPublisher, RuntimeOutputSnapshot, _layout_identity
from ._runtime_executor import install_runtime_executor
from ._runtime_planning import build_runtime_plans
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


def _same_physical_time(left: float, right: float) -> bool:
    tolerance = 4.0 * max(math.ulp(left), math.ulp(right))
    return abs(left - right) <= tolerance


def _validate_external_grid_deadline(
    strategy: Any,
    controls: Mapping[str, Any],
    deadline: float | None,
    run_end: float,
) -> None:
    """Require an exact external grid to contain every active consumer hard boundary."""
    from pops.time import ExternalTimeGrid

    if type(strategy) is not ExternalTimeGrid or deadline is None:
        return
    if deadline > run_end:
        return
    grid = tuple(float(value) for value in controls[strategy.grid_id])
    if not any(
            value >= deadline and _same_physical_time(value, deadline)
            for value in grid):
        raise ValueError(
            "every_dt deadline %s is absent from ExternalTimeGrid %r; add every physical-output "
            "deadline to the declared external grid"
            % (deadline.hex(), strategy.grid_id)
        )


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
    """Read one provider-owned native option schema without exposing the executor."""
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
        "schema_version", "provider_slot", "plan_identity", "provider_identity", "solver",
        "hierarchy_policy", "option_schema_identity", "options",
    }
    if set(result) != expected or result["schema_version"] != 1 \
            or result["provider_slot"] != slot \
            or not isinstance(result["plan_identity"], str) \
            or not result["plan_identity"]:
        raise TypeError("native field solver configuration has an invalid schema")
    for name in ("provider_identity", "solver", "option_schema_identity"):
        if not isinstance(result[name], str) or not result[name]:
            raise TypeError(
                "native field solver configuration requires an exact %s" % name
            )
    hierarchy_policy = result["hierarchy_policy"]
    if not isinstance(hierarchy_policy, Mapping) or set(hierarchy_policy) != {
        "policy_id", "interface_version", "option_schema", "options",
    }:
        raise TypeError("native field solver hierarchy policy has an invalid schema")
    hierarchy_policy = dict(hierarchy_policy)
    for name in ("policy_id", "option_schema"):
        if not isinstance(hierarchy_policy[name], str) or not hierarchy_policy[name]:
            raise TypeError(
                "native field solver hierarchy policy requires an exact %s" % name
            )
    version = hierarchy_policy["interface_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise TypeError(
            "native field solver hierarchy policy requires a positive interface_version"
        )
    if not isinstance(hierarchy_policy["options"], Mapping) or any(
        type(key) is not str or not key for key in hierarchy_policy["options"]
    ):
        raise TypeError("native field solver hierarchy policy options have an invalid schema")
    hierarchy_policy["options"] = dict(hierarchy_policy["options"])
    result["hierarchy_policy"] = hierarchy_policy
    if not isinstance(result["options"], Mapping) or any(
        type(key) is not str or not key for key in result["options"]
    ):
        raise TypeError("native field solver configuration options have an invalid schema")
    result["options"] = dict(result["options"])
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
        from pops.fields._prepared_field_solver_registry import (
            prepared_field_solver_binding_from_data,
        )

        binding = prepared_field_solver_binding_from_data(options["solver_provider"])
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
        result.append({
            "field": name,
            "provider_slot": slot,
            "provider": binding.to_data()["provider"],
            "source_layout_identity": layout_plan.qualified_id,
            "topology_recipe_identity": binding.facts.layout["topology_identity"],
            "topology_contract": binding.resolution.to_data()["topology_contract"],
            "component_bindings": binding.resolution.to_data()["component_bindings"],
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


@dataclass(frozen=True, slots=True)
class ConsumerRecoveryRecord:
    """Narrow inspection value for one retained output-quarantine authority."""

    recovery_id: str
    public_path: Path
    quarantine_path: Path
    owner: tuple[int, int]
    state: str
    consumer_report_index: int | None


@dataclass(frozen=True, slots=True)
class _ConsumerRecoveryOwner:
    record: ConsumerRecoveryRecord
    authority: _QuarantineRecovery


@dataclass(frozen=True, slots=True)
class _ConsumerRecoveryBatch:
    recoveries: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _PendingConsumerFinalization:
    transaction: Any
    consumer_report_index: int
    base_diagnostics: tuple[str, ...]


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
        "_consumer_finalize_pending",
        "_consumer_recoveries",
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
        preflight_consumer_publication(graph, plan.execution_context)
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
        self._consumer_finalize_pending: tuple[_PendingConsumerFinalization, ...] = ()
        self._consumer_recoveries: dict[str, _ConsumerRecoveryOwner] = {}
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
    def bound_snapshot(self) -> Any:
        """Immutable evidence of the exact artifact, layouts and inputs bound to this runtime."""
        snapshot = getattr(self._executor, "bound_snapshot", None)
        if snapshot is None:
            raise RuntimeError("RuntimeInstance executor lost its authenticated bound snapshot")
        return snapshot

    @property
    def consumer_graph(self) -> ConsumerGraph:
        """Resolved public authority for accepted runtime effects."""
        return self._consumer_graph

    @property
    def consumer_cursors(self) -> ConsumerCursorSet:
        """Immutable accepted cursors for the resolved consumer graph."""
        return self._consumer_cursors

    @property
    def consumer_recoveries(self) -> tuple[ConsumerRecoveryRecord, ...]:
        """Typed output recoveries retained after a cleanup race, ordered by identity."""
        registry = getattr(self, "_consumer_recoveries", {})
        return tuple(registry[key].record for key in sorted(registry))

    @property
    def post_commit_reports(self) -> tuple[Any, ...]:
        """Post-commit delivery reports retained across completed runs."""
        rows = getattr(self._publisher, "post_commit_reports", ())
        if not isinstance(rows, tuple):
            raise TypeError("consumer publisher post_commit_reports must be a tuple")
        return rows

    @property
    def post_commit_diagnostics(self) -> tuple[str, ...]:
        """Operational delivery failures that never rewrote numerical acceptance."""
        rows = getattr(self._publisher, "post_commit_diagnostics", ())
        if not isinstance(rows, tuple) or any(not isinstance(row, str) for row in rows):
            raise TypeError("consumer publisher post_commit_diagnostics must be tuple[str, ...]")
        return rows

    @property
    def live_visualization_reports(self) -> tuple[Any, ...]:
        """Compatibility alias for :attr:`post_commit_reports`."""
        return self.post_commit_reports

    @property
    def live_visualization_diagnostics(self) -> tuple[str, ...]:
        """Compatibility alias for :attr:`post_commit_diagnostics`."""
        return self.post_commit_diagnostics

    def flush_post_commit_consumers(self) -> tuple[Any, ...]:
        """Block until the current run's queued post-commit frames have terminal reports.

        Every MPI rank must call this method, as for the rest of the collective RuntimeInstance
        surface.  SERIAL/ROOT workers do not call MPI; PER_RANK/COLLECTIVE workers use their
        authenticated duplicated observer lanes and require ``MPI_THREAD_MULTIPLE``.
        """
        run_identity = self.last_run_identity
        if type(run_identity) is not Identity or run_identity.domain != "run":
            raise RuntimeError("no authenticated run is available to flush")
        flush = getattr(self._publisher, "flush_post_commit_consumers", None)
        if not callable(flush):
            raise NotImplementedError(
                "the installed consumer publisher has no post-commit flush route")
        reports = flush(run_identity)
        if not isinstance(reports, tuple):
            raise TypeError("post-commit flush must return a tuple of reports")
        return reports

    def flush_live_visualizations(self) -> tuple[Any, ...]:
        """Compatibility alias for :meth:`flush_post_commit_consumers`."""
        return self.flush_post_commit_consumers()

    def retry_consumer_finalizers(self) -> tuple[str, ...]:
        """Retry release-only finalizers without reopening accepted transactions."""
        return self._retry_consumer_finalizers()

    def restore_consumer_recovery(self, recovery_id: str) -> ConsumerRecoveryRecord:
        """Restore one retained inode without overwriting a newer public path."""
        registry = getattr(self, "_consumer_recoveries", {})
        owner = registry.get(recovery_id)
        if owner is None:
            raise KeyError("unknown consumer recovery %r" % recovery_id)
        owner.authority.restore()
        updated = replace(owner.record, state="restored")
        registry[recovery_id] = replace(owner, record=updated)
        return updated

    def cleanup_consumer_recovery(self, recovery_id: str) -> None:
        """Release one explicitly restored quarantine hardlink and forget its authority."""
        registry = getattr(self, "_consumer_recoveries", {})
        owner = registry.get(recovery_id)
        if owner is None:
            raise KeyError("unknown consumer recovery %r" % recovery_id)
        if owner.record.state != "restored":
            raise RuntimeError(
                "consumer recovery must be restored before its retained authority is cleaned")
        owner.authority.cleanup_restored()
        del registry[recovery_id]

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

    def integral(
        self, block: str, component: int = 0, *, levels: tuple[int, ...] | None = None,
    ) -> float:
        """Return the native volume integral of one conservative component.

        Uniform execution uses its component reduction plus the resolved Cartesian cell measure;
        adaptive execution uses the native composite reduction on exactly ``levels`` (all levels
        when omitted), masking covered coarse cells between selected levels.  No state array is
        copied into Python for this diagnostic.
        """
        if isinstance(component, bool) or not isinstance(component, int) or component < 0:
            raise TypeError("integral component must be a non-negative integer")
        selected_levels = () if levels is None else tuple(levels)
        if any(isinstance(level, bool) or not isinstance(level, int) or level < 0
               for level in selected_levels):
            raise TypeError("integral levels must contain non-negative integers")
        if tuple(sorted(set(selected_levels))) != selected_levels:
            raise ValueError("integral levels must be strictly increasing and unique")
        assignments = [
            row for row in self._layout_plan.assignments
            if row.subject_kind == "block" and row.subject.local_id == block
        ]
        if len(assignments) != 1:
            raise KeyError("unknown RuntimeInstance block %s" % block)
        layouts = [
            row for row in self._layout_plan.layouts
            if row.handle == assignments[0].layout
        ]
        if len(layouts) != 1:
            raise KeyError("block %s has no exact runtime layout" % block)
        layout = layouts[0]
        engine = self._executor_for_block(block)
        if layout.adaptive:
            provider = getattr(engine, "composite_reduce", None)
            if not callable(provider):
                raise NotImplementedError(
                    "adaptive runtime provider does not expose composite_reduce"
                )
            return float(cast(float, provider(block, "sum", component, list(selected_levels))))

        provider = getattr(engine, "reduce_component", None)
        if not callable(provider):
            raise NotImplementedError(
                "uniform runtime provider does not expose reduce_component")
        if selected_levels not in {(), (0,)}:
            raise ValueError("uniform integral accepts only level 0")
        from pops.mesh._layout_plan_contracts import (
            CARTESIAN_CELL_AREA,
            NormalizedGeometry,
        )

        geometry = layout.geometry
        if type(geometry) is not NormalizedGeometry \
                or geometry.cell_measure != CARTESIAN_CELL_AREA:
            raise NotImplementedError(
                "uniform integral requires the native Cartesian cell measure")
        measure = 1.0
        for length, cells in zip(geometry.lengths, geometry.cells, strict=True):
            measure *= float(length) / int(cells)
        return measure * float(cast(float, provider(block, "sum", component)))

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
        provider: Any = getattr(self._executor, "n_levels", None)
        if callable(provider):
            return int(cast(Any, provider()))
        if not any(row.adaptive for row in self._layout_plan.layouts):
            return 1
        raise NotImplementedError(
            "adaptive runtime provider does not expose its native hierarchy level count"
        )

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
                "restart_authority": (
                    self._install_plan.artifact.plan.restart_authority.to_data()
                ),
                "consumer_cursors": self._consumer_cursors.to_data(),
                "consumer_reports": [report.to_data() for report in self._consumer_reports],
                "accepted_diagnostics": [
                    payload.to_data() for payload in self._publisher.accepted_diagnostics
                ],
                "post_commit_reports": [
                    report.to_data() for report in self.post_commit_reports
                ],
                "post_commit_diagnostics": list(self.post_commit_diagnostics),
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
                physical_time_hex=cursor["time"],
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

    def _retain_consumer_recoveries(
        self,
        transaction: Any,
        *,
        consumer_report_index: int | None,
    ) -> tuple[str, ...]:
        """Transfer typed quarantine authorities before a transaction owner is released."""
        failures = []
        try:
            recoveries = getattr(transaction, "recoveries", ())
            if type(recoveries) is not tuple:
                raise TypeError("ConsumerTransaction.recoveries must return a tuple")
        except BaseException as error:
            return (
                "consumer recovery registry failed: %s: %s"
                % (type(error).__name__, error),
            )
        registry = getattr(self, "_consumer_recoveries", None)
        if registry is None:
            registry = {}
            self._consumer_recoveries = registry
        for recovery in recoveries:
            if type(recovery) is not _QuarantineRecovery:
                failures.append(
                    "consumer recovery registry refused a non-canonical authority")
                continue
            recovery_id = make_identity("consumer-output-recovery", {
                "public_path": recovery.public_path.as_posix(),
                "quarantine_path": recovery.quarantine_path.as_posix(),
                "owner": list(recovery.owner),
                "directory_owner": list(recovery.directory_owner),
            }).token
            existing = registry.get(recovery_id)
            if existing is not None:
                if existing.authority is not recovery and existing.authority != recovery:
                    failures.append(
                        "consumer recovery identity collides with another authority")
                continue
            record = ConsumerRecoveryRecord(
                recovery_id,
                recovery.public_path,
                recovery.quarantine_path,
                recovery.owner,
                "retained",
                consumer_report_index,
            )
            registry[recovery_id] = _ConsumerRecoveryOwner(record, recovery)
        return tuple(failures)

    def _retain_output_recoveries(self, recoveries: tuple[Any, ...]) -> tuple[str, ...]:
        """Typed callback used when writer staging fails before a transaction is returned."""
        if type(recoveries) is not tuple:
            return ("output recovery transfer requires a tuple",)
        return self._retain_consumer_recoveries(
            _ConsumerRecoveryBatch(recoveries), consumer_report_index=None)

    def _abort_consumers(self, transactions: tuple[ConsumerTransaction, ...]) -> BaseException | None:
        failure = None
        for transaction in reversed(transactions):
            try:
                transaction.abort()
            except BaseException as error:
                if failure is None:
                    failure = error
            recovery_failures = self._retain_consumer_recoveries(
                transaction, consumer_report_index=None)
            if recovery_failures and failure is None:
                failure = RuntimeError("; ".join(recovery_failures))
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

    @staticmethod
    def _consumer_seal_diagnostics(transaction: Any) -> tuple[str, ...]:
        try:
            result = transaction.seal()
            if type(result) is not tuple or any(
                    not isinstance(item, str) or not item for item in result):
                return (
                    "consumer seal contract violation: seal() must return tuple[str, ...]",
                )
            return result
        except BaseException as error:
            # A release failure is evidence only after native finalization.  Catch BaseException
            # so a rank-local cancellation cannot reopen compensation or strand its owner.
            return (
                "consumer seal failed post-commit: %s: %s"
                % (type(error).__name__, error),
            )

    @staticmethod
    def _report_with_finalize_diagnostics(
        report: Any,
        base_diagnostics: tuple[str, ...],
        diagnostics: tuple[str, ...],
    ) -> Any:
        if type(report) is not ConsumerTransactionReport:
            return report
        try:
            return replace(report, diagnostics=base_diagnostics + diagnostics)
        except BaseException:
            return report

    def _seal_consumer_reports(
        self,
        transactions: tuple[ConsumerTransaction, ...],
        reports: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """Seal, report, and retain every release owner until its finalizer succeeds."""
        sealed = []
        pending: list[_PendingConsumerFinalization] = list(
            getattr(self, "_consumer_finalize_pending", ())
        )
        report_offset = len(self._consumer_reports)
        for index, transaction in enumerate(transactions):
            report = reports[index]
            base_diagnostics = (
                report.diagnostics if type(report) is ConsumerTransactionReport else ()
            )
            report_index = report_offset + index
            diagnostics = self._consumer_seal_diagnostics(transaction)
            diagnostics += self._retain_consumer_recoveries(
                transaction, consumer_report_index=report_index)
            report = self._report_with_finalize_diagnostics(
                report, base_diagnostics, diagnostics)
            sealed.append(report)
            if diagnostics or bool(getattr(transaction, "finalize_pending", False)):
                pending.append(
                    _PendingConsumerFinalization(transaction, report_index, base_diagnostics)
                )
        self._consumer_finalize_pending = tuple(pending)
        return tuple(sealed)

    def _retry_consumer_finalizers(self) -> tuple[str, ...]:
        """Retry pending releases and replace, rather than accumulate, operational diagnostics."""
        pending: list[_PendingConsumerFinalization] = list(
            getattr(self, "_consumer_finalize_pending", ())
        )
        if not pending:
            return ()
        reports = list(self._consumer_reports)
        remaining: list[_PendingConsumerFinalization] = []
        current_diagnostics: list[str] = []
        for owner in pending:
            diagnostics = self._consumer_seal_diagnostics(owner.transaction)
            diagnostics += self._retain_consumer_recoveries(
                owner.transaction,
                consumer_report_index=owner.consumer_report_index,
            )
            current_diagnostics.extend(diagnostics)
            if 0 <= owner.consumer_report_index < len(reports):
                reports[owner.consumer_report_index] = self._report_with_finalize_diagnostics(
                    reports[owner.consumer_report_index],
                    owner.base_diagnostics,
                    diagnostics,
                )
            else:
                diagnostics += ("consumer finalizer lost its accepted report slot",)
            if diagnostics or bool(getattr(owner.transaction, "finalize_pending", False)):
                remaining.append(owner)
        self._consumer_reports = tuple(reports)
        self._consumer_finalize_pending = tuple(remaining)
        return tuple(current_diagnostics)

    def _fire_consumers(self, *, at_start: bool = False, at_end: bool = False) -> tuple[Any, ...]:
        self._retry_consumer_finalizers()
        transactions = self._stage_consumers(at_start=at_start, at_end=at_end)
        try:
            reports, cursors, _all_reports = self._accept_consumers(transactions)
        except BaseException as error:
            cleanup_error = self._abort_consumers(transactions)
            if cleanup_error is not None:
                raise cleanup_error from error
            raise
        sealed_reports = self._seal_consumer_reports(transactions, reports)
        self._consumer_cursors = cursors
        report_offset = len(self._consumer_reports)
        self._consumer_reports = self._consumer_reports + sealed_reports
        self._retry_consumer_finalizers()
        return self._consumer_reports[report_offset:]

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
            restored_temporal = snapshot["temporal_restart_state"]
            restore_temporal = getattr(native, "_restore_temporal_restart_state", None)
            if callable(restore_temporal):
                restore_temporal(restored_temporal)
            else:
                native._temporal_restart_state = restored_temporal
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

        self._retry_consumer_finalizers()
        native = self._executor
        begin, commit, finalize, rollback = self._step_transaction_methods()
        snapshot = self._step_envelope_snapshot()
        phase = "begin"
        attempts = 1
        failure_report = None
        transactions = ()
        result = None
        reports: tuple[Any, ...] = ()
        cursors = self._consumer_cursors
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
            reports, cursors, _all_reports = self._accept_consumers(transactions)
            phase = "commit"
            finalize()
            native_active = False
            phase = "native_finalized"
            sealed_reports = self._seal_consumer_reports(transactions, reports)
            self._consumer_cursors = cursors
            self._consumer_reports = self._consumer_reports + sealed_reports
            self._retry_consumer_finalizers()
            return result
        except BaseException as error:
            if phase == "native_finalized":
                # The engine and published receipts are already irrevocably accepted.  A release
                # bug is operational evidence only: never compensate artifacts, call native
                # rollback, or restore the pre-step Python envelope from this boundary.
                accepted_reports = list(reports)
                diagnostic = (
                    "consumer finalization failed post-commit: %s: %s"
                    % (type(error).__name__, error)
                )
                if accepted_reports and type(accepted_reports[0]) is ConsumerTransactionReport:
                    try:
                        report = accepted_reports[0]
                        accepted_reports[0] = replace(
                            report, diagnostics=report.diagnostics + (diagnostic,))
                    except BaseException:
                        pass
                self._consumer_cursors = cursors
                self._consumer_reports = (
                    snapshot["consumer_reports"] + tuple(accepted_reports))
                return result
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
             output_dir: Any = None, console: bool = True,
             **controller_controls: Any) -> RunReport:
        if type(console) is not bool:
            raise TypeError("pops.run console= must be an exact bool")
        if "progress" in controller_controls:
            raise TypeError(
                "pops.run progress= was removed; declare a scheduled "
                "pops.output.ConsoleMonitor instead")
        if "strategy" in controller_controls or "cfl" in controller_controls:
            raise TypeError(
                "RuntimeInstance._run does not accept strategy= or cfl=; declare the controller "
                "with Program.step_strategy(...)"
            )
        from pops.runtime._step_strategy import (
            prepare_step_controller, resolve_run_strategy, run_control_payload, run_step_attempt)
        from pops.runtime._native_step_target import native_step_target
        from pops.runtime.run_report import RunStopReason

        native = self._executor
        step_target = native_step_target(native)
        selected = resolve_run_strategy(native)
        control = run_control_payload(selected, controller_controls)
        self._step_transaction_methods()
        entry_temporal = copy.deepcopy(getattr(native, "_temporal_restart_state", None))
        entry_controller = copy.deepcopy(getattr(native, "_step_controller", None))
        previous_root, self._output_root = self._output_root, output_dir
        steps = 0
        rejected_steps = 0
        console_session = None
        manifest = None
        try:
            prepare_step_controller(native, selected, controller_controls)
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
            if console:
                from pops.runtime._console_run import safe_begin_console_run

                console_session = safe_begin_console_run(self, manifest, selected)
            begin_post_commit = getattr(
                self._publisher, "begin_post_commit_consumers", None)
            if callable(begin_post_commit):
                begin_post_commit(manifest.run_identity)
            self._fire_consumers(at_start=True)
            while native.time() < t_end and steps < max_steps:
                deadline = next_consumer_deadline(self._consumer_graph, self._moments())
                run_end = float(t_end)
                _validate_external_grid_deadline(
                    selected, controller_controls, deadline, run_end)
                # A tolerance can validate a controller landing, but it must never extend the
                # requested run.  In particular, a threshold one ULP above t_end is a future
                # occurrence, not an end-of-run sample.
                if deadline is not None and deadline <= run_end:
                    deadline_is_active = True
                    step_end = (
                        run_end if _same_physical_time(deadline, run_end) else deadline)
                else:
                    deadline_is_active = False
                    step_end = run_end

                def advance(
                    *,
                    accepted_deadline: float | None = deadline if deadline_is_active else None,
                    accepted_step_end: float = step_end,
                ) -> tuple[Any, int]:
                    report = run_step_attempt(
                        native, step_target, selected, t_end=accepted_step_end,
                        controls=controller_controls)
                    reached = float(native.time())
                    if accepted_deadline is not None \
                            and reached > accepted_deadline \
                            and not _same_physical_time(reached, accepted_deadline):
                        raise RuntimeError(
                            "step controller crossed every_dt hard deadline %s and reached %s"
                            % (accepted_deadline.hex(), reached.hex())
                        )
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
            close_live = getattr(self._publisher, "close_live_visualizations", None)
            if callable(close_live):
                close_live(manifest.run_identity)
        except BaseException as error:
            if manifest is not None:
                close_live = getattr(self._publisher, "close_live_visualizations", None)
                if callable(close_live):
                    before = len(self.post_commit_diagnostics)
                    try:
                        close_live(manifest.run_identity, raise_on_failure=False)
                    except BaseException as close_error:
                        add_note = getattr(error, "add_note", None)
                        if callable(add_note):
                            add_note(
                                "post-commit consumer close also failed: %s" % close_error)
                    after = self.post_commit_diagnostics
                    if len(after) > before:
                        add_note = getattr(error, "add_note", None)
                        if callable(add_note):
                            add_note(
                                "post-commit consumer delivery diagnostics: %s"
                                % "; ".join(after[before:]))
            # ``begin_run`` binds controller/strategy state before the first native transaction.
            # If no macro-step commits, the complete failed call leaves the temporal authority at
            # its entry boundary.  After one or more accepted steps, each later failed transaction
            # already restores the last accepted boundary and that progress must be retained.
            if steps == 0:
                restore_error = None
                try:
                    restore_temporal = getattr(native, "_restore_temporal_restart_state", None)
                    if callable(restore_temporal):
                        restore_temporal(entry_temporal)
                    elif hasattr(native, "_temporal_restart_state"):
                        native._temporal_restart_state = entry_temporal
                    if hasattr(native, "_step_controller"):
                        native._step_controller = entry_controller
                except BaseException as caught:
                    restore_error = caught
                add_note = getattr(error, "add_note", None)
                if restore_error is not None and callable(add_note):
                    add_note("run-entry temporal rollback also failed: %s" % restore_error)
            if console_session is not None:
                from pops.runtime._console_run import safe_console_failed

                safe_console_failed(
                    console_session,
                    error,
                    accepted_steps=steps,
                    final_time=float(native.time()),
                )
            raise
        finally:
            self._output_root = previous_root
        report = RunReport(
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
        if console_session is not None:
            from pops.runtime._console_run import safe_console_completed

            safe_console_completed(console_session, report)
        return report

    def _checkpoint_payload(self, path: Any) -> str:
        from pops.output._checkpoint_collective import (
            canonical_checkpoint_path,
            checkpoint_topology,
            consensus,
            root_value,
        )

        topology = checkpoint_topology(self)
        expected = canonical_checkpoint_path(path)
        target = None
        capture_error = None
        try:
            target = canonical_checkpoint_path(self._executor.checkpoint(str(expected)))
            if target != expected:
                raise RuntimeError(
                    "native checkpoint returned %s for shared staging target %s"
                    % (target, expected)
                )
        except BaseException as error:
            capture_error = error
        rows = consensus(
            topology,
            "native capture",
            error=capture_error,
            value=None if target is None else str(target),
        )
        if any(row["value"] != str(expected) for row in rows):
            raise RuntimeError("native checkpoint ranks returned different staged paths")

        import numpy as np
        from ._checkpoint_manifest import (
            IDENTITY_KEY,
            MANIFEST_KEY,
            authenticate_checkpoint_payload,
            seal_checkpoint_payload,
        )

        def seal_root() -> str:
            if not expected.is_file():
                raise RuntimeError("native checkpoint did not create the shared staged file")
            with np.load(expected, allow_pickle=False) as stored:
                old_manifest = json.loads(str(stored[MANIFEST_KEY]))
                runtime_kind = old_manifest.get("runtime_kind")
                if not isinstance(runtime_kind, str) or not runtime_kind:
                    raise ValueError("native checkpoint manifest lacks its runtime kind")
                # Authenticate every native byte before replacing its envelope with the
                # RuntimeInstance consumer/cursor authority.
                authenticate_checkpoint_payload(self, stored, runtime_kind=runtime_kind)
                payload = {
                    name: np.asarray(stored[name]).copy()
                    for name in stored.files if name not in {MANIFEST_KEY, IDENTITY_KEY}
                }
            payload["runtime_consumer_graph"] = np.asarray(self._consumer_graph.identity.token)
            cursors = self._checkpoint_cursor_override or self._consumer_cursors
            payload["runtime_consumer_cursors"] = np.asarray(json.dumps(
                cursors.to_data(), sort_keys=True, separators=(",", ":")))
            payload["runtime_consumer_diagnostics"] = np.asarray(json.dumps(
                self._publisher.diagnostic_restart_state(),
                sort_keys=True, separators=(",", ":")))
            seal_checkpoint_payload(self, payload, runtime_kind=runtime_kind)
            temporary = expected.with_name(expected.name + ".runtime-instance.tmp")
            try:
                with open(temporary, "wb") as stream:
                    np.savez_compressed(stream, **payload)
                os.replace(temporary, expected)
            finally:
                temporary.unlink(missing_ok=True)
            # A staged checkpoint is not publishable until its final envelope has been read back
            # and authenticated by the same strict path used during restart.
            self._inspect_checkpoint_file(expected)
            return str(expected)

        sealed = Path(root_value(topology, "runtime envelope sealing", seal_root))
        if sealed != expected:
            raise RuntimeError("rank zero sealed a different checkpoint staging path")
        return str(expected)

    def _restart_operation(self) -> Any:
        from pops.output._restart_provider import RestartAuthority

        authority = self._install_plan.artifact.plan.restart_authority
        if type(authority) is not RestartAuthority:
            raise TypeError("installed plan has no exact restart authority")
        return authority.operation

    def checkpoint(self, path: Any) -> str:
        self._retry_consumer_finalizers()
        try:
            operation = self._restart_operation()
            extension = operation.consumer_data()["extension"]
            from pops.output._checkpoint_collective import canonical_checkpoint_path

            target = canonical_checkpoint_path(path, extension=extension)
            snapshot = operation.snapshot(self, target.parent)
            operation.validate_snapshot(snapshot)
            try:
                return str(operation.write(snapshot, target))
            except BaseException as error:
                discard = getattr(snapshot, "discard", None)
                if callable(discard):
                    try:
                        discard()
                    except BaseException as cleanup_error:
                        add_note = getattr(error, "add_note", None)
                        if callable(add_note):
                            add_note("checkpoint staging cleanup also failed: %s" % cleanup_error)
                raise
        finally:
            self._retry_consumer_finalizers()

    @staticmethod
    def _checkpoint_cursors_from_data(cursor_data: Any) -> ConsumerCursorSet:
        if not isinstance(cursor_data, Mapping) \
                or set(cursor_data) != {"schema_version", "rows"} \
                or cursor_data["schema_version"] != 1 \
                or not isinstance(cursor_data["rows"], list):
            raise ValueError("restart consumer cursor schema is unsupported")
        cursors = ConsumerCursorSet(tuple(ScheduleCursor(
            row["consumer_id"], row["last_occurrence"], row["committed_samples"])
            for row in cursor_data["rows"]))
        # Reject duplicate/non-canonical input instead of accepting a decoder normalization.
        if cursors.to_data() != dict(cursor_data):
            raise ValueError("restart consumer cursor rows are not canonical")
        return cursors

    def _inspect_checkpoint_file(self, path: Any) -> ConsumerCursorSet:
        """Rank-zero-only complete authentication; performs no native mutation."""
        return self._inspect_checkpoint_payload(Path(path).read_bytes())

    def _inspect_checkpoint_payload(self, payload: bytes) -> ConsumerCursorSet:
        """Authenticate exact in-memory bytes on rank zero without native mutation."""
        from pops.output._checkpoint_collective import decode_checkpoint_bytes
        from ._checkpoint_manifest import MANIFEST_KEY, authenticate_checkpoint_payload

        stored = decode_checkpoint_bytes(payload)
        if MANIFEST_KEY not in stored.files:
            raise ValueError("checkpoint has no canonical manifest")
        manifest = json.loads(str(stored[MANIFEST_KEY]))
        runtime_kind = manifest.get("runtime_kind")
        if not isinstance(runtime_kind, str) or not runtime_kind:
            raise ValueError("checkpoint manifest lacks its runtime kind")
        authenticate_checkpoint_payload(self, stored, runtime_kind=runtime_kind)
        required = {
            "runtime_consumer_graph",
            "runtime_consumer_cursors",
            "runtime_consumer_diagnostics",
        }
        if not required <= set(stored.files):
            raise ValueError("restart checkpoint lacks RuntimeInstance consumer state")
        if str(stored["runtime_consumer_graph"]) != self._consumer_graph.identity.token:
            raise ValueError("restart ConsumerGraph identity differs from the installed graph")
        cursor_data = json.loads(str(stored["runtime_consumer_cursors"]))
        diagnostic_data = json.loads(str(stored["runtime_consumer_diagnostics"]))
        self._publisher.validate_diagnostic_restart_state(diagnostic_data)
        return self._checkpoint_cursors_from_data(cursor_data)

    def _restore_checkpoint(self, payload: bytes, cursors: ConsumerCursorSet) -> Any:
        from pops.output._checkpoint_collective import (
            decode_checkpoint_bytes,
            restore_checkpoint_payload,
        )

        stored = decode_checkpoint_bytes(payload)
        diagnostic_data = json.loads(str(stored["runtime_consumer_diagnostics"]))
        canonical_diagnostics = self._publisher.validate_diagnostic_restart_state(
            diagnostic_data)

        result = restore_checkpoint_payload(
            self, self._executor, payload, phase_prefix="native restart")
        self._consumer_cursors = cursors
        self._publisher.restore_diagnostic_restart_state(canonical_diagnostics)
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
