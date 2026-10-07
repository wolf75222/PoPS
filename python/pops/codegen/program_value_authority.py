"""Private lowering of produced SSA values consumed by auxiliary-aware Field RHSs.

The installed plan contains only the actual transitive state inputs of V2 Field
sites.  It describes storage membership; runtime tickets supply successful-write
authority.  Neither a node label nor a borrowed MultiFab is a produced value.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class ValueProducer:
    ssa: int
    block: int
    storage: str
    storage_id: int
    subslot: int
    history: str
    lag: int
    inputs: tuple[int, ...]
    input_branches: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class FieldValueEdge:
    field_node: int
    block: int
    source: int
    accepted_only: bool = False


@dataclass(frozen=True, slots=True)
class ProgramValueAuthorityPlan:
    program_identity: str
    producers: tuple[ValueProducer, ...]
    fields: tuple[FieldValueEdge, ...]
    block_names: tuple[str, ...] = ()

    def producer(self, ssa: int) -> ValueProducer | None:
        return next((row for row in self.producers if row.ssa == ssa), None)

    def cpp_install(self) -> list[str]:
        if not self.fields:
            return []
        prefix = "pops::runtime::program::"
        producers = ["{%d, %d, %sProgramValueStorage::%s, %d, %d, %s, %d, {%s}, {%s}}" % (
            row.ssa, row.block, prefix, row.storage, row.storage_id, row.subslot,
            json.dumps(row.history), row.lag, ", ".join(map(str, row.inputs)),
            ", ".join("{%s}" % ", ".join(map(str, branch)) for branch in row.input_branches))
            for row in self.producers]
        fields = ["{%d, %d, %d, %s}" % (row.field_node, row.block, row.source,
                  "true" if row.accepted_only else "false")
                  for row in self.fields]
        return ["ctx.install_program_value_plan(%sProgramValuePlan{%s, {%s}, {%s}});" % (
            prefix, json.dumps(self.program_identity), ", ".join(producers), ", ".join(fields))]


_BUFFER_TYPES = frozenset(("state", "rhs", "scalar_field", "vector_field", "mask"))
_RHS_STORAGE = frozenset(("rhs", "source", "implicit_source", "apply"))
_STATE_STORAGE = frozenset(("linear_combine", "solve_local_linear", "solve_local_nonlinear",
                            "solve_implicit_source", "where", "pointwise_expression"))
_PURE_ALIASES = frozenset(("solve_outcome", "solve_outcome_component",
                           "acceptance_guard", "store_history"))


def prepare_program_value_authority(program: Any, authority: Any,
                                    field_plans: Any) -> ProgramValueAuthorityPlan:
    """Select exact Auxiliary packs, then seal their readable SSA dependency closure."""
    from pops.codegen.program_emit_field_routes import _walk_program_nodes, resolved_field_route
    from pops.codegen.program_models import ProgramModelGraph
    from pops.time.references import block_name

    blocks = program._block_indices()
    owner_names = [block_name(block) for block in sorted(blocks, key=blocks.get)]
    nodes = tuple(_walk_program_nodes(program._values))
    by_id = {node.id: node for node in nodes}
    edges = []
    for node in nodes:
        if node.op not in ("solve_fields", "solve_fields_from_blocks"):
            continue
        _, field = resolved_field_route(node.attrs["field"], field_plans)
        routes = field.native_options.get("provider_pack", ())
        providers = getattr(field, "rhs_providers", ())
        if routes and len(providers) != len(routes):
            raise ValueError("Field value authority requires the complete authenticated RHS route")
        v2_owners = set()
        for provider, route in zip(providers, routes, strict=True):
            owner = route["owner_block"]
            if provider.block_ref is None or block_name(provider.block_ref) != owner:
                raise ValueError("Field value authority changes its authenticated provider owner")
            emitter = (authority.model_for_block(owner)
                       if type(authority) is ProgramModelGraph else authority)
            if emitter is None:
                continue
            implementation = getattr(emitter, "_m", emitter)
            packs = getattr(implementation, "_component_operator_provider_packs", None)
            key = route["key"]
            if packs is None or key not in packs:
                raise ValueError("Field value authority lacks its exact RHS ProviderPack")
            # Empty packs retain V1, regardless of operator names or expressions.
            if len(packs[key]):
                v2_owners.add(owner)
        sources = {block_name(source.block): source for source in node.inputs}
        for owner in sorted(v2_owners):
            if owner not in owner_names:
                owner_names.append(owner)
            source = sources.get(owner)
            if source is not None and by_id.get(source.id) is not source:
                raise ValueError("Field value authority has a detached SSA source")
            edges.append(FieldValueEdge(node.id, owner_names.index(owner),
                                        -1 if source is None else source.id, source is None))

    producers = {}

    def visit(source: Any, owner: int) -> ValueProducer:
        if source.id not in by_id or by_id[source.id] is not source:
            raise ValueError("Field value authority has a detached SSA source")
        if source.id in producers:
            existing = producers[source.id]
            if existing.block != owner:
                raise ValueError("Field value authority changes SSA block ownership")
            return existing
        source_owner = blocks.get(source.block, owner)
        if source_owner != owner:
            raise ValueError("Field value authority mixes distinct state owners")
        readable = tuple(value for value in source.inputs
                         if value.vtype in _BUFFER_TYPES or value.op == "solve_outcome")
        inputs = tuple(visit(value, owner).ssa for value in readable)
        storage, storage_id, subslot, history, lag = "", source.id, 0, "", 0
        if source.op == "state":
            storage, storage_id = "State", -1
        elif source.op == "history":
            storage, storage_id = "History", -1
            history, lag = source.attrs["history"], int(source.attrs["lag"])
        elif source.op in _PURE_ALIASES:
            storage = "Alias"
        elif source.op == "synchronize":
            if source.attrs["relation"].get("kind") == "history_interpolation":
                storage = "StateScratch"
            else:
                storage = "Alias"
        elif source.op in ("fill_boundary", "project"):
            if len(inputs) != 1:
                raise ValueError("in-place SSA producer requires one authenticated value")
            storage = "Alias"
        elif source.op in _RHS_STORAGE:
            if source.op == "rhs" and source.attrs.get("path_conservative", False):
                raise NotImplementedError("Field RHS path producer requires its actual delegated storage hook")
            storage = "RhsScratch"
        elif source.op in _STATE_STORAGE:
            storage = ("ScalarScratch" if source.op == "pointwise_expression"
                       and "field_product_seed" in source.attrs else "StateScratch")
        elif source.op == "cell_compare":
            storage = "ScalarScratch"
        else:
            raise NotImplementedError("Field RHS SSA producer %r has no issued storage lowering" % source.op)
        if storage == "Alias" and len(inputs) != 1:
            raise NotImplementedError("Field RHS SSA alias requires one readable native producer")
        branches = ()
        if source.attrs.get("schedule") is not None:
            from pops.codegen.program_emit_schedule import _lower_schedule_ir
            from pops.time._schedule.api import ScheduleAction
            policy = _lower_schedule_ir(source, source.attrs["schedule"]).off
            if any(action in (ScheduleAction.RESTORE, ScheduleAction.ZERO)
                   for action in policy.off_cadence):
                branches = ((),)
        row = ValueProducer(source.id, owner, storage, storage_id, subslot, history, lag, inputs, branches)
        producers[source.id] = row
        return row

    for edge in edges:
        if not edge.accepted_only:
            visit(by_id[edge.source], edge.block)
    return ProgramValueAuthorityPlan(program._ir_hash(), tuple(producers.values()), tuple(edges),
                                     tuple(owner_names))


def _plan(var: Any) -> ProgramValueAuthorityPlan | None:
    return var.get(("program_value_authority",))


def field_uses_value_authority(value: Any, var: Any) -> bool:
    plan = _plan(var)
    return plan is not None and any(edge.field_node == value.id for edge in plan.fields)


def field_value_overrides_cpp(value: Any, var: Any, block_indices: Any) -> str:
    plan = _plan(var)
    rows = []
    for source in value.inputs:
        owner = block_indices[source.block]
        edge = next((item for item in plan.fields
                     if item.field_node == value.id and item.block == owner), None)
        if edge is not None and (edge.accepted_only or edge.source != source.id):
            raise ValueError("active Field source differs from its installed SSA edge")
        rows.append("{%d, &%s, %d}" % (
            owner, var[source.id], -1 if edge is None else source.id))
    return ", ".join(rows)


def _dependencies(row: ValueProducer, var: Any) -> str:
    plan = _plan(var)
    return ", ".join("ctx.program_value(%d, %d, %s)" % (
        item, plan.producer(item).block, var[item]) for item in row.inputs)


def begin_value_write_cpp(value: Any, var: Any, *, alternate: bool = False) -> list[str]:
    plan = _plan(var)
    row = plan.producer(value.id) if plan is not None else None
    if row is None or value.op in ("state", "history") or (
            row.storage == "Alias" and (value.op == "synchronize" or value.op in _PURE_ALIASES)):
        return []
    if alternate and () not in row.input_branches:
        raise ValueError("scheduled producer has no declared off-cadence input branch")
    return ["auto value_write_%d = ctx.begin_program_value_write(%d, %d, %s, {%s});" % (
        value.id, value.id, row.block, var[value.id], "" if alternate else _dependencies(row, var))]


def complete_value_cpp(value: Any, var: Any) -> list[str]:
    plan = _plan(var)
    row = plan.producer(value.id) if plan is not None else None
    if row is None:
        return []
    if value.op in ("state", "history"):
        return ["(void)ctx.capture_program_value(%d, %d, %s);" % (
            value.id, row.block, var[value.id])]
    if row.storage == "Alias" and (value.op == "synchronize" or value.op in _PURE_ALIASES):
        if len(row.inputs) != 1:
            raise ValueError("SSA alias requires one exact produced input")
        return ["(void)ctx.alias_program_value(%d, %s);" % (
            value.id, _dependencies(row, var))]
    return ["(void)ctx.complete_program_value_write(std::move(value_write_%d));" % value.id]


def instrument_value_cpp(value: Any, var: Any, lines: list[str], before_write: int) -> None:
    """Place issue before evaluation and completion after every successful guard."""
    before = begin_value_write_cpp(value, var)
    lines.extend(complete_value_cpp(value, var))
    if before:
        lines[before_write:before_write] = before
