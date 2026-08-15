"""Exact multi-layout Uniform runtime coordination and transactional persistence."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from pops.runtime._component_execution_context import component_execution_data


@dataclass(frozen=True, slots=True)
class _PreparedMultiLayoutRestart:
    restart_identity: Any
    mapping: dict[str, int]
    children: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _NativeTransferRoute:
    transfer: Any
    source_block: str
    target_block: str
    session: Any
    source_element_count: int
    destination_element_count: int


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


def _mapping_blocks(plan: Any, transfer: Any) -> tuple[str, str]:
    matches = tuple(
        row
        for row in plan.artifact.layout_plan.mappings
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
    source_shape = tuple(source.shape)
    target_shape = tuple(target.shape)
    source_lower = tuple(source.lower)
    target_lower = tuple(target.lower)
    source_upper = tuple(source.upper)
    target_upper = tuple(target.upper)
    if (
        len(source_shape) not in (1, 2, 3)
        or len(source_shape) != len(target_shape)
        or not (
            len(source_lower)
            == len(target_lower)
            == len(source_upper)
            == len(target_upper)
            == len(source_shape)
        )
    ):
        raise ValueError(
            "CONSERVATIVE_CELL_AVERAGE_V1 requires one identical compile-time spatial rank"
        )
    if source.coordinate_system != target.coordinate_system:
        raise ValueError(
            "CONSERVATIVE_CELL_AVERAGE_V1 requires identical coordinate systems; select a mapped "
            "Transfer operation/provider for distinct geometries"
        )
    if source_upper != target_upper:
        raise ValueError(
            "CONSERVATIVE_CELL_AVERAGE_V1 requires identical physical upper bounds; select a mapped "
            "Transfer operation/provider for distinct geometries"
        )
    if source_lower != target_lower:
        raise ValueError(
            "CONSERVATIVE_CELL_AVERAGE_V1 requires identical physical lower bounds; select a mapped "
            "Transfer operation/provider for translated geometries"
        )
    source_periodicity = source.periodicity
    target_periodicity = target.periodicity
    for label, periodicity in (
        ("source", source_periodicity),
        ("target", target_periodicity),
    ):
        if (
            type(periodicity) is not tuple
            or len(periodicity) != len(source_shape)
            or any(type(axis) is not bool for axis in periodicity)
        ):
            raise TypeError(
                "CONSERVATIVE_CELL_AVERAGE_V1 requires an exact ranked periodicity tuple "
                f"on the {label} layout"
            )
    if source_periodicity != target_periodicity:
        raise ValueError(
            "CONSERVATIVE_CELL_AVERAGE_V1 requires identical boundary topology; select a mapped "
            "Transfer operation/provider for distinct topologies"
        )


def _require_runtime_plan_projection(
    plan: Any, runtime_plan: Any, transfers: tuple[Any, ...]
) -> None:
    """Require every multi-layout route the provider claims before engine construction."""
    layout_plan = plan.artifact.layout_plan
    assignments = {
        row.subject.local_id: (row.subject_id, row.layout.qualified_id)
        for row in layout_plan.assignments
        if row.subject_kind == "block"
    }
    expected_calls = tuple(assignments[block.name] for block in plan.artifact.blocks)
    actual_calls = tuple((row.block_id, row.layout_id) for row in runtime_plan.calls)
    if actual_calls != expected_calls:
        raise ValueError(
            "RuntimePlanBundle calls differ from the multi-layout InstallPlan projection"
        )
    if runtime_plan.communication.halos:
        raise NotImplementedError(
            "multi-layout RuntimePlan halos require an explicit per-layout halo scheduler"
        )
    expected_providers = tuple(sorted({row.provider_id for row in transfers}))
    if runtime_plan.resources.mapping_provider_ids != expected_providers:
        raise ValueError("RuntimePlanBundle mapping providers differ from the consumed Transfers")


def _layout_runtime_authority_plan(plan: Any, selected_names: Any) -> Any:
    """Project one InstallPlan onto the exact block set owned by a child layout."""
    names = tuple(selected_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("layout authority projection requires unique selected block names")
    selected = set(names)
    compiled = tuple(row for row in plan.artifact.blocks if row.name in selected)
    plan_blocks = tuple(row for row in plan.artifact.plan.blocks if row.name in selected)
    if (
        len(compiled) != len(names)
        or len(plan_blocks) != len(names)
        or {row.name for row in compiled} != selected
        or {row.name for row in plan_blocks} != selected
    ):
        raise ValueError(
            "layout authority projection must match the selected compiled/plan block set exactly"
        )
    artifact = plan.artifact
    return SimpleNamespace(
        artifact=SimpleNamespace(
            resolved_dimension=artifact.resolved_dimension,
            blocks=compiled,
            plan=SimpleNamespace(blocks=plan_blocks, field_plans=artifact.plan.field_plans),
            layout_plan=artifact.layout_plan,
        ),
        params=plan.params,
        components=plan.components,
        execution_context=plan.execution_context,
    )


def _release_layout_engines(engines: list[Any]) -> None:
    """Destroy already-materialized child Systems in reverse install order."""
    errors: list[BaseException] = []
    while engines:
        engine = engines.pop()
        try:
            destroy = getattr(engine, "destroy", None)
            if callable(destroy):
                destroy()
            elif getattr(engine, "_s", None) is not None:
                engine._s = None
        except BaseException as error:
            errors.append(error)
    if errors:
        raise RuntimeError(
            "multi-layout child release failed: %s" % "; ".join(map(str, errors))
        ) from errors[0]


def _require_runtime_plan_bundle(plan: Any, runtime_plan: Any) -> None:
    """Authenticate the exact bundle and its Transfer projection against one InstallPlan."""
    from pops.runtime._runtime_plan_contracts import LayoutTransfer
    from pops.runtime._runtime_planning import require_runtime_plan_bundle

    runtime_plan = require_runtime_plan_bundle(plan, runtime_plan)
    layout_plan = plan.artifact.layout_plan
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
    _require_runtime_plan_projection(plan, runtime_plan, transfers)
    _require_unique_transfer_targets(transfers)


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
    def event_queue(self) -> Any:
        return self._same_attribute("event_queue")

    @property
    def time_hex(self) -> Any:
        return self._same_attribute("time_hex")

    @property
    def macro_step(self) -> Any:
        return self._same_attribute("macro_step")

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

    def before_queued_attempt(
        self,
        event: Any,
        *,
        time: Any,
        macro_step: Any,
    ) -> None:
        for state in self.states:
            state.before_queued_attempt(event, time=time, macro_step=macro_step)

    def queue_error_controlled_proposal(self, **kwargs: Any) -> Any:
        return _common_exact(
            (state.queue_error_controlled_proposal(**kwargs) for state in self.states),
            where="multi-layout temporal proposal",
        )

    def queued_error_controlled_proposal(self, **kwargs: Any) -> Any:
        return _common_exact(
            (state.queued_error_controlled_proposal(**kwargs) for state in self.states),
            where="multi-layout queued temporal proposal",
        )

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
        self,
        plan: Any,
        runtime_plan: Any,
        engines: dict[str, Any],
        blocks: dict[str, str],
        transfer_routes: tuple[_NativeTransferRoute, ...],
    ) -> None:
        self._plan = plan
        self._execution_context = plan.execution_context
        self._runtime_plan = runtime_plan
        self._engines = dict(engines)
        self._block_layouts = dict(blocks)
        self._transfer_routes = tuple(transfer_routes)
        self._mapping_evaluations = {
            row.mapping_id: 0 for row in runtime_plan.communication.transfers
        }
        self._mapping_snapshot = None
        self._transfer_generation = 0
        self._active_transfer_generation = None
        self._transfer_attempt = 0
        self._last_mapping_receipts = ()
        self._last_run_manifest = None
        self._last_run_identity = None
        self._restart_lineage_identity = None
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
        from pops.identity import make_identity
        from pops.runtime._checkpoint_resource_budget import (
            aggregate_checkpoint_resource_budgets,
            require_checkpoint_resource_budget,
        )

        child_budgets = tuple(
            require_checkpoint_resource_budget(engine) for engine in self._engines.values()
        )
        budget_authority = make_identity(
            "multi-layout-checkpoint-resource-budget",
            {
                "artifact": plan.artifact.artifact_identity.token,
                "bind": plan.bind_identity.token,
                "layouts": [
                    {"layout": layout_id, "budget": budget.authority}
                    for layout_id, budget in zip(self._engines, child_budgets, strict=True)
                ],
                "mappings": tuple(self._mapping_evaluations),
            },
        ).token
        self._checkpoint_resource_budget = aggregate_checkpoint_resource_budgets(
            child_budgets,
            authority=budget_authority,
            install_plan=plan,
            layout_ids=tuple(self._engines),
            mapping_ids=tuple(self._mapping_evaluations),
        )
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

    def _restore_checkpoint_run_identity(self, identity: Any) -> None:
        from pops.identity import Identity

        if type(identity) is not Identity or identity.domain != "run":
            raise TypeError("multi-layout restart requires an authenticated run identity")
        restored = Identity.from_data(identity.to_data())
        self._last_run_manifest = None
        self._last_run_identity = restored
        self._restart_lineage_identity = restored
        for engine in self._engines.values():
            engine._restore_checkpoint_run_identity(restored)

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

    def _ordered_program_reports(self) -> tuple[tuple[Any, tuple[str, ...], Any], ...]:
        """Return one authenticated report for every independently installed Program."""
        from pops.runtime.program_report import ProgramRuntimeReport

        layout_programs = tuple(self._plan.artifact.layout_programs)
        layout_ids = tuple(row.layout_id for row in layout_programs)
        if layout_ids != tuple(self._engines):
            raise RuntimeError(
                "multi-layout Program reports differ from the installed layout order"
            )
        rows = []
        for layout_program in layout_programs:
            engine = self._engines[layout_program.layout_id]
            report = engine.program_report()
            if type(report) is not ProgramRuntimeReport:
                raise TypeError("multi-layout child returned a non-canonical ProgramRuntimeReport")
            if (
                not report.installed
                or not isinstance(report.program_hash, str)
                or not (report.program_hash)
            ):
                raise RuntimeError("multi-layout child has no authenticated installed Program")
            engine_blocks = tuple(engine.block_names())
            if len(engine_blocks) != len(set(engine_blocks)) or set(engine_blocks) != set(
                layout_program.block_names
            ):
                raise RuntimeError(
                    "multi-layout child block registry differs from its compiled partition"
                )
            local_map = tuple(report.block_map)
            if (
                len(local_map) != len(engine_blocks)
                or any(
                    isinstance(index, bool) or not isinstance(index, int) or index < 0
                    for index in local_map
                )
                or tuple(sorted(local_map)) != tuple(range(len(engine_blocks)))
            ):
                raise RuntimeError(
                    "multi-layout child Program block map is not an exact local bijection"
                )
            parameter_blocks = tuple(row.get("program_block") for row in report.params)
            if len(parameter_blocks) != len(local_map) or any(
                isinstance(index, bool) or not isinstance(index, int) for index in parameter_blocks
            ):
                raise RuntimeError("multi-layout child Program parameter report is not exact")
            exact_parameter_blocks = cast(tuple[int, ...], parameter_blocks)
            if tuple(sorted(exact_parameter_blocks)) != tuple(range(len(local_map))):
                raise RuntimeError("multi-layout child Program parameter report is not exact")
            rows.append((layout_program, engine_blocks, report))
        return tuple(rows)

    def program_report(self) -> Any:
        """Aggregate every real child Program without inventing a single native engine."""
        from pops.identity import make_identity
        from pops.runtime.program_report import ProgramRuntimeReport

        children = self._ordered_program_reports()
        global_blocks = self.block_names()
        global_block_indices = {name: index for index, name in enumerate(global_blocks)}
        if len(global_block_indices) != len(global_blocks):
            raise RuntimeError("multi-layout global block registry contains a duplicate")

        block_map = []
        params = []
        diagnostics = {}
        histories = []
        cache = []
        clocks = []
        level_relations = []
        flux_ledger = []
        synchronization = []
        program_offset = 0

        qualified_row_sets = (
            ("history", histories, "histories"),
            ("clock", clocks, "clocks"),
            ("level relation", level_relations, "level_relations"),
            ("flux ledger", flux_ledger, "flux_ledger"),
            ("synchronization", synchronization, "synchronization"),
        )
        for layout_program, engine_blocks, report in children:
            layout_id = layout_program.layout_id
            local_map = tuple(report.block_map)
            local_program_blocks = tuple(
                engine_blocks[local_system_index] for local_system_index in local_map
            )
            block_map.extend(global_block_indices[name] for name in local_program_blocks)

            for raw in report.params:
                row = dict(raw)
                local_program_block = row["program_block"]
                if "layout_id" in row or "block" in row:
                    raise RuntimeError(
                        "multi-layout child parameter report contains reserved qualifiers"
                    )
                row["program_block"] = program_offset + local_program_block
                row["layout_id"] = layout_id
                row["block"] = local_program_blocks[local_program_block]
                params.append(row)

            for name, value in report.diagnostics.items():
                if not isinstance(name, str) or not name:
                    raise RuntimeError("multi-layout child diagnostic name must be non-empty")
                diagnostics["%s::%s" % (layout_id, name)] = value

            for label, destination, attribute in qualified_row_sets:
                for raw in getattr(report, attribute):
                    row = dict(raw)
                    if "layout_id" in row:
                        raise RuntimeError(
                            "multi-layout child %s report contains a reserved qualifier" % label
                        )
                    row["layout_id"] = layout_id
                    destination.append(row)

            for raw in report.cache:
                row = dict(raw)
                if "layout_id" in row or "layout_node_id" in row:
                    raise RuntimeError(
                        "multi-layout child cache report contains reserved qualifiers"
                    )
                local_node_id = row.get("node_id")
                if (
                    isinstance(local_node_id, bool)
                    or not isinstance(local_node_id, int)
                    or local_node_id < 0
                ):
                    raise RuntimeError(
                        "multi-layout child cache report has an invalid node identity"
                    )
                row["layout_id"] = layout_id
                row["layout_node_id"] = local_node_id
                row["node_id"] = len(cache)
                cache.append(row)
            program_offset += len(local_map)

        program_hash = make_identity(
            "multi-layout-program",
            [
                {
                    "layout_id": layout_program.layout_id,
                    "layout_program_identity": layout_program.identity.token,
                    "installed_program_hash": report.program_hash,
                }
                for layout_program, _engine_blocks, report in children
            ],
        ).hexdigest
        return ProgramRuntimeReport(
            installed=True,
            program_hash=program_hash,
            step_transaction=_common_exact(
                (report.step_transaction for _row, _blocks, report in children),
                where="multi-layout Program transaction report",
            ),
            block_map=block_map,
            params=params,
            diagnostics=diagnostics,
            histories=histories,
            cache=cache,
            profiler=_common_exact(
                (report.profiler for _row, _blocks, report in children),
                where="multi-layout Program profiler report",
            ),
            clocks=clocks,
            level_relations=level_relations,
            flux_ledger=flux_ledger,
            synchronization=synchronization,
            temporal_partition=_common_exact(
                (report.temporal_partition for _row, _blocks, report in children),
                where="multi-layout Program temporal-partition report",
            ),
            temporal=_common_exact(
                (report.temporal for _row, _blocks, report in children),
                where="multi-layout Program temporal report",
            ),
        )

    def installed_program_hash(self) -> str:
        """Return the domain-separated identity of the exact installed Program set."""
        return self.program_report().program_hash

    def state_global(self, block: str) -> Any:
        return self.executor_for_block(block).state_global(block)

    def get_state(self, block: str) -> Any:
        return self.executor_for_block(block).get_state(block)

    def set_state(self, block: str, values: Any) -> Any:
        return self.executor_for_block(block).set_state(block, values)

    def spatial_shape(self) -> tuple[int, ...]:
        raise ValueError(
            "multi-layout geometry requires executor_for_layout(layout_id).spatial_shape()"
        )

    def _common_clock(self, method: str) -> Any:
        values = tuple(getattr(engine, method)() for engine in self._engines.values())
        if any(value != values[0] for value in values[1:]):
            raise RuntimeError("multi-layout native clocks diverged at %s" % method)
        return values[0]

    def time(self) -> float:
        return float(self._common_clock("time"))

    def macro_step(self) -> int:
        return int(self._common_clock("macro_step"))

    def _consume_step_projections(self) -> tuple[str, ...]:
        """Consume the union of projection identities executed by child layouts."""
        result = []
        for engine in self._engines.values():
            consume = getattr(engine, "_consume_step_projections", None)
            if not callable(consume):
                raise TypeError("multi-layout child lacks the step-projection report protocol")
            rows = consume()
            if not isinstance(rows, (tuple, list)) or any(
                not isinstance(name, str) or not name for name in rows
            ):
                raise TypeError("multi-layout child returned an invalid step-projection report")
            for name in rows:
                if name not in result:
                    result.append(name)
        return tuple(result)

    def _native_step_target(self) -> Any:
        """The coordinator itself is the raw target for one composite attempt."""
        return self

    def _mapping_blocks(self, transfer: Any) -> tuple[str, str]:
        return _mapping_blocks(self._plan, transfer)

    def _authenticate_mapping_receipt(
        self,
        route: _NativeTransferRoute,
        receipt: Any,
        *,
        generation: int,
        attempt: int,
    ) -> None:
        transfer = route.transfer
        component = self._plan.components[transfer.component_id]
        expected = {
            "applied": True,
            "mapping_identity": transfer.mapping_id,
            "provider_identity": transfer.provider_id,
            "provider_component_identity": transfer.component_id,
            "provider_manifest_identity": component.component_manifest.token,
            "source_layout_identity": transfer.source_layout_id,
            "target_layout_identity": transfer.target_layout_id,
            "source_block": route.source_block,
            "target_block": route.target_block,
            "execution_identity": self._plan.execution_context.identity.token,
            "operation": transfer.operation_abi,
            "generation": generation,
            "attempt": attempt,
            "source_element_count": route.source_element_count,
            "destination_element_count": route.destination_element_count,
        }
        for name, value in expected.items():
            if getattr(receipt, name, object()) != value:
                raise RuntimeError(
                    "native Transfer receipt does not authenticate %s for mapping %s"
                    % (name, transfer.mapping_id)
                )

    def _restore_rejected_native_attempt(self, generation: int, attempt: int) -> None:
        """Restore every child to the outer accepted snapshot before a controller retries."""
        for route in self._transfer_routes:
            route.session.reject_attempt(generation, attempt)
        rollback_errors = []
        for engine in reversed(tuple(self._engines.values())):
            try:
                engine._rollback_step_transaction()
            except BaseException as error:
                rollback_errors.append(error)
        if rollback_errors:
            raise RuntimeError(
                "multi-layout rejected-attempt rollback failed: %s"
                % "; ".join(map(str, rollback_errors))
            )
        begun = []
        try:
            for engine in self._engines.values():
                engine._begin_step_transaction()
                begun.append(engine)
        except BaseException as begin_error:
            for engine in reversed(begun):
                try:
                    engine._rollback_step_transaction()
                except BaseException:
                    pass
            raise RuntimeError(
                "multi-layout rejected-attempt transaction restart failed"
            ) from begin_error

    def step(self, dt: float) -> None:
        from pops.runtime._native_step_target import native_step_target
        from pops._bootstrap import StepAttemptRejected

        generation = self._active_transfer_generation
        if not isinstance(generation, int) or generation <= 0:
            raise RuntimeError("multi-layout native step requires an active transfer transaction")
        self._transfer_attempt += 1
        attempt = self._transfer_attempt
        # Snapshot every source before any destination changes.  Cycles therefore observe one
        # common pre-transfer state and never depend on mapping declaration order.
        for route in self._transfer_routes:
            route.session.capture(generation, attempt)
        receipts = []
        try:
            for route in self._transfer_routes:
                receipt = route.session.apply(generation, attempt)
                self._authenticate_mapping_receipt(
                    route, receipt, generation=generation, attempt=attempt
                )
                receipts.append(receipt)
            for engine in self._engines.values():
                native_step_target(engine).step(dt)
        except StepAttemptRejected:
            self._restore_rejected_native_attempt(generation, attempt)
            raise
        for route in self._transfer_routes:
            self._mapping_evaluations[route.transfer.mapping_id] += 1
        self._last_mapping_receipts = tuple(receipts)
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
        begun_routes = []
        try:
            for engine in self._engines.values():
                engine._begin_step_transaction()
                begun.append(engine)
            self._transfer_generation += 1
            generation = self._transfer_generation
            for route in self._transfer_routes:
                route.session.begin_transaction(generation)
                begun_routes.append(route)
            self._active_transfer_generation = generation
            self._transfer_attempt = 0
        except BaseException:
            for route in reversed(begun_routes):
                route.session.rollback_transaction(self._transfer_generation)
            for engine in reversed(begun):
                engine._rollback_step_transaction()
            self._active_transfer_generation = None
            self._transfer_attempt = 0
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
        generation = self._active_transfer_generation
        if isinstance(generation, int):
            for route in self._transfer_routes:
                route.session.finalize_transaction(generation)
        self._active_transfer_generation = None
        self._transfer_attempt = 0
        self._mapping_snapshot = None

    def _rollback_step_transaction(self) -> None:
        error = None
        for engine in reversed(tuple(self._engines.values())):
            try:
                engine._rollback_step_transaction()
            except BaseException as caught:
                error = error or caught
        generation = self._active_transfer_generation
        if isinstance(generation, int):
            for route in reversed(self._transfer_routes):
                route.session.rollback_transaction(generation)
        if self._mapping_snapshot is not None:
            self._mapping_evaluations = self._mapping_snapshot
        self._mapping_snapshot = None
        self._active_transfer_generation = None
        self._transfer_attempt = 0
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
        self,
        root: str,
        prefix: str,
        topology: Any,
        *,
        retain_payloads: bool,
    ) -> tuple[tuple[str, ...], tuple[bytes, ...]]:
        """Capture every child collectively; only rank zero may read the resulting files."""
        from pops.output._checkpoint_collective import (
            canonical_checkpoint_path,
            consensus,
            root_effect,
        )
        from pops.runtime._checkpoint_manifest import authenticate_checkpoint_payload

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
            expected = canonical_checkpoint_path(os.path.join(root, "%s-%d" % (prefix, index)))
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
                child_engine: Any = engine,
                child_path: Path = expected,
            ) -> bytes | None:
                from pops.output._checkpoint_collective import (
                    _bounded_checkpoint_path_bytes,
                    decode_checkpoint_bytes,
                )
                from pops.runtime._checkpoint_resource_budget import (
                    require_checkpoint_resource_budget,
                )

                child_budget = require_checkpoint_resource_budget(child_engine)
                child_bytes = _bounded_checkpoint_path_bytes(
                    child_path, child_budget.max_archive_bytes
                )
                stored = decode_checkpoint_bytes(child_bytes, child_budget)
                authenticate_checkpoint_payload(child_engine, stored, runtime_kind="uniform")
                return child_bytes if retain_payloads else None

            payload = root_effect(
                topology,
                "%s layout %d authentication" % (prefix, index),
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
            return tempfile.mkdtemp(prefix=".%s.%s." % (target.name, purpose), dir=target.parent)

        return str(root_value(topology, "%s workspace selection" % purpose, create_root))

    @staticmethod
    def _cleanup_checkpoint_root(root: str, topology: Any, purpose: str) -> None:
        from pops.output._checkpoint_collective import root_effect

        root_effect(
            topology,
            "%s workspace cleanup" % purpose,
            lambda: shutil.rmtree(root, ignore_errors=False),
        )

    def checkpoint(self, path: Any) -> str:
        """Capture a public multi-layout checkpoint with atomic final publication."""
        return self._checkpoint(path)

    def _checkpoint_precreated_inode(self, path: Any, *, precreated_descriptor: int | None) -> str:
        """Internal RuntimeInstance seam retaining its precreated output inode."""
        return self._checkpoint(
            path,
            precreated_inode=True,
            precreated_descriptor=precreated_descriptor,
        )

    def _checkpoint(
        self,
        path: Any,
        *,
        precreated_inode: bool = False,
        precreated_descriptor: int | None = None,
    ) -> str:
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
            write_precreated_checkpoint_payload,
        )

        topology = checkpoint_topology(self)
        target = canonical_checkpoint_path(path)
        rows = consensus(topology, "multi-layout target", value=str(target))
        if any(row["value"] != str(target) for row in rows):
            raise ValueError("multi-layout checkpoint target differs across ranks")
        root = self._shared_checkpoint_root(target, topology, "capture")
        try:
            _paths, children = self._checkpoint_children(
                root, "child", topology, retain_payloads=True
            )

            def write_root() -> None:
                if len(children) != len(self._engines):
                    raise RuntimeError("rank zero did not retain every child checkpoint")
                payload = {
                    "t": np.asarray(self.time()),
                    "macro_step": np.asarray(self.macro_step()),
                    "abi_key": np.asarray(abi_key()),
                    "layout_ids": np.asarray(tuple(self._engines), dtype=np.str_),
                    "mapping_evaluations": np.asarray(
                        json.dumps(
                            self._mapping_evaluations,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                }
                for index, child in enumerate(children):
                    payload["layout_checkpoint_%d" % index] = np.frombuffer(
                        child, dtype=np.uint8
                    ).copy()
                seal_checkpoint_payload(self, payload, runtime_kind="multi_layout_uniform")
                if precreated_inode:
                    if type(precreated_descriptor) is not int:
                        raise RuntimeError(
                            "precreated multi-layout checkpoint publication requires the root "
                            "descriptor"
                        )
                    write_precreated_checkpoint_payload(precreated_descriptor, payload)
                else:
                    fd, temporary_name = tempfile.mkstemp(
                        prefix=".%s." % target.name, suffix=".tmp", dir=target.parent
                    )
                    os.close(fd)
                    temporary = os.fspath(temporary_name)
                    try:
                        with open(temporary, "wb") as stream:
                            np.savez_compressed(stream, **payload)
                        os.replace(temporary, target)
                    finally:
                        Path(temporary).unlink(missing_ok=True)
                from pops.output._checkpoint_collective import (
                    _bounded_checkpoint_path_bytes,
                    decode_checkpoint_bytes,
                )
                from pops.runtime._checkpoint_resource_budget import (
                    require_checkpoint_resource_budget,
                )

                container_budget = require_checkpoint_resource_budget(self)
                stored = decode_checkpoint_bytes(
                    _bounded_checkpoint_path_bytes(target, container_budget.max_archive_bytes),
                    container_budget,
                )
                authenticate_checkpoint_payload(self, stored, runtime_kind="multi_layout_uniform")

            root_effect(topology, "multi-layout container sealing", write_root)
        finally:
            self._cleanup_checkpoint_root(root, topology, "capture")
        return str(target)

    def _prepare_checkpoint_restart(
        self,
        payload: bytes,
        *,
        bit_identical: bool,
        hierarchy_mode: str = "restore_recorded_hierarchy",
        hierarchy_identity: str | None = None,
    ) -> _PreparedMultiLayoutRestart:
        if hierarchy_mode != "restore_recorded_hierarchy":
            raise NotImplementedError("multi-layout restart does not yet support RegridOnRestart")
        if hierarchy_identity is not None:
            raise ValueError(
                "multi-layout restart hierarchy identity is only valid with RegridOnRestart"
            )
        import numpy as np
        from pops.output._checkpoint_collective import (
            decode_checkpoint_bytes,
            require_restart_bit_identical,
        )
        from pops.runtime._checkpoint_manifest import authenticate_checkpoint_payload
        from pops.runtime._checkpoint_resource_budget import require_checkpoint_resource_budget

        policy = require_restart_bit_identical(bit_identical, where="multi-layout restart")
        stored = decode_checkpoint_bytes(payload, require_checkpoint_resource_budget(self))
        identity = authenticate_checkpoint_payload(
            self, stored, runtime_kind="multi_layout_uniform"
        )
        layout_ids = tuple(str(value) for value in stored["layout_ids"])
        if layout_ids != tuple(self._engines):
            raise ValueError("checkpoint layout identities differ from RuntimeInstance")
        child_names = tuple("layout_checkpoint_%d" % index for index in range(len(layout_ids)))
        if any(name not in stored.files for name in child_names):
            raise ValueError("checkpoint lacks a per-layout native payload")
        mapping = json.loads(str(stored["mapping_evaluations"]))
        if set(mapping) != set(self._mapping_evaluations) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in mapping.values()
        ):
            raise ValueError("checkpoint mapping counters differ from RuntimeInstance plan")
        prepared_children = []
        for index, (engine, name) in enumerate(
            zip(self._engines.values(), child_names, strict=True)
        ):
            prepare = getattr(engine, "_prepare_checkpoint_restart", None)
            if not callable(prepare):
                raise TypeError(
                    "layout %d engine lacks the in-memory restart preflight protocol" % index
                )
            child_bytes = np.asarray(stored[name], dtype=np.uint8).tobytes()
            prepared_children.append(prepare(child_bytes, bit_identical=policy))
        return _PreparedMultiLayoutRestart(identity, dict(mapping), tuple(prepared_children))

    def _begin_checkpoint_restart(self) -> None:
        if "_checkpoint_restart_snapshot" in self.__dict__:
            raise RuntimeError("multi-layout checkpoint restart transaction is already active")
        self._checkpoint_restart_snapshot = (
            dict(self._mapping_evaluations),
            self._last_restart_identity,
            self._temporal_restart_state,
            getattr(self, "_step_controller", None),
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
                "_checkpoint_restart_snapshot"
            )
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
        # Prepare every child release while all accepted snapshots remain rollback-capable.  A
        # failure in a later child can therefore roll every earlier committed child back.
        for engine in self._engines.values():
            engine._commit_checkpoint_restart()

    def _finalize_checkpoint_restart(self) -> None:
        # No child releases until every child preparation above has succeeded and the enclosing
        # collective commit consensus has completed.
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
                "multi-layout child rollback failed: %s" % "; ".join(map(str, errors))
            )

    def restart(self, path: Any, *, bit_identical: bool = False) -> str:
        from pops.output._checkpoint_collective import (
            _bounded_checkpoint_path_bytes,
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
        from pops.runtime._checkpoint_resource_budget import require_checkpoint_resource_budget

        archive_budget = require_checkpoint_resource_budget(self).max_archive_bytes
        payload = root_bytes(
            topology,
            "multi-layout restart read",
            lambda: _bounded_checkpoint_path_bytes(target, archive_budget),
            max_bytes=archive_budget,
        )
        restore_checkpoint_payload(
            self,
            self,
            payload,
            bit_identical=bit_identical,
            phase_prefix="multi-layout restart",
        )
        return str(target)


def install_multi_layout_uniform(plan: Any, runtime_plan: Any) -> Any:
    from pops.codegen._layout_resolution import ResolvedRuntimeLayouts
    from pops.runtime._runtime_mesh_lowering import (
        install_embedded_boundary,
        system_config_from_layout,
    )
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
        configs[layout_id] = system_config_from_layout(plan.artifact.native_layouts[layout_id])
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
        if getattr(component, "native_handle", None) is None:
            raise TypeError("mapping Transfer component has no authenticated native handle")
        source = configs[transfer.source_layout_id]
        target = configs[transfer.target_layout_id]
        _require_conservative_cell_average_geometry(source, target)
        source_shape = tuple(source.shape)
        target_shape = tuple(target.shape)
        if any(
            source_extent < target_extent or source_extent % target_extent
            for source_extent, target_extent in zip(source_shape, target_shape, strict=True)
        ):
            raise ValueError("CONSERVATIVE_CELL_AVERAGE_V1 requires aligned fine-to-coarse layouts")

    from pops.runtime._runtime_authorities import install_runtime_authorities
    from pops.runtime._runtime_executor import _uniform_initial_sources

    initial_sources = _uniform_initial_sources(plan)
    engines = {}
    materialized: list[Any] = []
    try:
        for row in layouts.rows:
            layout_id = row.handle.qualified_id
            engine = System(configs[layout_id])
            materialized.append(engine)
            from pops.runtime._checkpoint_spatial import install_checkpoint_spatial_contract

            normalized_layout = layouts.plan.normalized(row.handle)
            install_checkpoint_spatial_contract(
                engine,
                normalized_layout.native_spatial_layout,
                transition_ratios=normalized_layout.transition_ratios,
            )
            cast(Any, engine)._execution_context = plan.execution_context
            install_embedded_boundary(engine, normalized_layout)
            selected = {
                name: spec for name, spec in plan.instances.items() if blocks[name] == layout_id
            }
            selected_initials = {
                name: source for name, source in initial_sources.items() if name in selected
            }
            view = _LayoutCompiledView(plan.artifact, programs[layout_id])
            install_runtime_authorities(
                engine, _layout_runtime_authority_plan(plan, tuple(selected))
            )
            engine._install_compiled(
                view,
                instances=selected,
                params=plan.params,
                aux={},
                field_plans={},
                initial_sources=selected_initials,
                _layout_checkpoint_install=(
                    programs[layout_id].program.program,
                    tuple(programs[layout_id].block_names),
                    plan.artifact.artifact_identity,
                    plan.bind_identity,
                    plan,
                ),
            )
            engines[layout_id] = engine

        execution = component_execution_data(plan.execution_context)
        transfer_routes = []
        for transfer in runtime_plan.communication.transfers:
            source_block, target_block = _mapping_blocks(plan, transfer)
            source_engine = engines[transfer.source_layout_id]
            target_engine = engines[transfer.target_layout_id]
            source_shape = tuple(int(value) for value in source_engine.spatial_shape())
            target_shape = tuple(int(value) for value in target_engine.spatial_shape())
            if len(source_shape) != len(target_shape) or any(
                source_extent < target_extent or source_extent % target_extent
                for source_extent, target_extent in zip(source_shape, target_shape, strict=True)
            ):
                raise ValueError(
                    "CONSERVATIVE_CELL_AVERAGE_V1 requires aligned fine-to-coarse layouts"
                )
            ratio = tuple(
                source_extent // target_extent
                for source_extent, target_extent in zip(source_shape, target_shape, strict=True)
            )
            component = plan.components[transfer.component_id]
            source_native = source_engine._native_step_target()
            target_native = target_engine._native_step_target()
            session = source_native._prepare_layout_transfer(
                target_native,
                component.native_handle,
                {
                    "mapping_identity": transfer.mapping_id,
                    "provider_identity": transfer.provider_id,
                    "provider_component_identity": transfer.component_id,
                    "provider_manifest_identity": component.component_manifest.token,
                    "source_layout_identity": transfer.source_layout_id,
                    "target_layout_identity": transfer.target_layout_id,
                    "source_block": source_block,
                    "target_block": target_block,
                    "source_representation": transfer.source_representation_uri,
                    "target_representation": transfer.target_representation_uri,
                    "synchronization_identity": transfer.synchronization_uri,
                    "refinement_ratio": ratio,
                    "operation": transfer.operation_abi,
                },
                execution,
            )
            source_components = int(source_engine.n_vars(source_block))
            target_components = int(target_engine.n_vars(target_block))
            if source_components != target_components or source_components <= 0:
                raise ValueError("layout transfer source/target component counts differ")
            transfer_routes.append(
                _NativeTransferRoute(
                    transfer=transfer,
                    source_block=source_block,
                    target_block=target_block,
                    session=session,
                    source_element_count=source_components * math.prod(source_shape),
                    destination_element_count=target_components * math.prod(target_shape),
                )
            )
        return _MultiLayoutUniformExecutor(
            plan, runtime_plan, engines, blocks, tuple(transfer_routes)
        )
    except BaseException as error:
        engines.clear()
        try:
            _release_layout_engines(materialized)
        except BaseException as release_error:
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note("multi-layout child release failed: %s" % release_error)
        raise


__all__ = ["install_multi_layout_uniform"]
