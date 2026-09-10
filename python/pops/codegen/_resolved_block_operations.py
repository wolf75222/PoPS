"""Bind physical operations to the exact block numerics and temporal evaluations."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .resolved_operations import EvaluationRequest, ExchangeRecord, TermOccurrence, build_resolved_operations
from ._resolved_operation_inputs import _reference


def _native_rhs_method(operation, value):
    """A named generated divergence and a configured FV flux are distinct methods."""
    if value.op != "rhs" or not value.attrs.get("fluxes") or not operation.exchanges:
        return operation
    selected = operation.guarantees.get("numerical_method")
    if selected is not None and selected.get("method") != "native_named_centered_divergence":
        from .resolved_operations import _reject

        _reject(operation.identity, "numerical_method_realization_mismatch",
                "selected method is not realized by the named centered-divergence native route")
    return replace(operation, stencil_radius=1, sampling="cell_centered_divergence",
        inputs=tuple(replace(read, sampling="cell") if "face_trace" in read.sampling else read
                     for read in operation.inputs),
        guarantees={**operation.guarantees, "stencil_known": True,
                    "numerical_method": {"method": "native_named_centered_divergence",
                                         "physical_fluxes": list(value.attrs["fluxes"])}})


def _legacy_rhs_selections(module, value):
    definitions = tuple(module.operator_registry())
    grids = tuple(op for op in definitions if op.kind == "grid_operator")
    names = []
    if value.attrs.get("flux", True):
        fluxes = value.attrs.get("fluxes")
        if fluxes is None:
            fluxes = tuple(op.name for op in grids if op.name in {"flux", "flux_default"})
            if not fluxes:
                defaults = tuple(dict.fromkeys(op.lowering["default_flux"] for op in definitions
                    if op.kind == "local_rate" and op.lowering.get("default_flux") is not None))
                if len(defaults) == 1:
                    fluxes = defaults
        names.extend((name, -1) for name in (fluxes or ()))
    sources = value.attrs.get("sources")
    if sources is None:
        sources = tuple(op.name for op in definitions
                        if op.kind == "local_source" and op.name == "default")
    names.extend((name, 1) for name in sources if name != "default" or any(
        op.name == "default" and op.kind == "local_source" for op in definitions))
    return tuple(names)


def _projection(value: Any) -> Any:
    if value is None:
        return None
    from pops.numerics.plan import _callable_projection

    return _callable_projection(value, "resolved numerical context")


def _program_values(program: Any):
    """Read exact SSA values, preserving control-region identity and lexical guards."""
    def walk(values, path, guards):
        for value in values:
            location = "%s/value:%d" % (path, value.id)
            yield value, location, guards
            for key in ("cond_block", "body_block", "apply_block", "residual_block",
                        "true_block", "false_block"):
                nested = value.attrs.get(key)
                if isinstance(nested, (tuple, list)):
                    guard = "%s/%s" % (location, key)
                    yield from walk(nested, guard, (*guards, guard))
    yield from walk(program._values, "program", ())


def build_block_resolved_operations(block: Any, program: Any):
    """Resolve one block using its actual numerical and Program authorities.

    Declaration records remain reusable emitter selections. Every concrete tagged
    SSA evaluation has separate occurrence coverage; no timestep-wide counting or
    sharing of an evaluation across different stages is implied.
    """
    from pops.codegen._compiler_lowering import require_compiler_lowering
    from pops.model import OperatorHandle

    module = require_compiler_lowering(block.model).source_module
    boundaries = () if block.numerics is None else block.numerics.boundaries
    boundary_data = tuple(_projection(row) for row in boundaries)
    # Numerical boundary providers retain typed authored data as well as their
    # immutable serialization. Both enter the dependency collector.
    base = build_resolved_operations(module, boundary_data=boundaries)
    selected = {}
    if block.numerics is not None:
        for row in block.numerics.rates:
            selected[row.rate.registered_operator_name] = row.method
    declarations = {"operation:%s" % _reference(module.operator_handle(op.name)): op
                    for op in module.operator_registry()}
    operations, requests = [], list(base.evaluations)
    for operation in base.operations:
        definition = declarations[operation.identity]
        method = selected.get(definition.name)
        if method is None and definition.kind in {"grid_operator", "local_rate"}:
            method = block.spatial
        numerical = None if method is None else _projection(method)
        radius = getattr(method, "ghost_depth", None)
        # Generic numerical providers cannot obtain an invented zero-neighborhood
        # optimization: keep an explicit barrier until they provide their stencil.
        effects = operation.effects
        if method is not None and radius is None:
            effects = tuple(dict.fromkeys((*effects, "unknown_stencil")))
        if radius is not None and (type(radius) is not int or radius < 0):
            raise TypeError("selected numerical ghost_depth must be a non-negative integer")
        guarantees = {**operation.guarantees,
                      "declaration_operation": operation.identity,
                      "block_instance": block.instance_owner_qid,
                      "numerical_method": numerical,
                      "stencil_known": radius is not None or definition.kind not in {
                          "grid_operator", "local_rate"},
                      "boundary_authorities": boundary_data}
        balance = definition.lowering.get("physical_balance")
        if balance is not None:
            guarantees["accumulation"] = _projection(balance.accumulation.resolve_references(
                lambda handle: handle._resolved()))
        operations.append(replace(operation, stencil_radius=radius or 0,
                                  effects=effects, guarantees=guarantees))
    by_identity = {item.identity: item for item in operations}
    evaluations = {item.identity: item for item in requests}
    values_to_operations = {}
    for value, location, guards in _program_values(program):
        from pops.problem.handles import BlockHandle

        owner = getattr(value, "block", None)
        instance = owner if isinstance(owner, BlockHandle) else getattr(owner, "block_ref", None)
        if instance is not None and str(instance.instance_owner_path.canonical()) != block.instance_owner_qid:
            continue
        handle = value.attrs.get("operator_handle")
        if not isinstance(handle, OperatorHandle):
            if value.op == "rhs":
                selection = _legacy_rhs_selections(module, value)
                evaluation = "evaluation:%s/%s" % (block.instance_owner_qid, location)
                terms, clones = [], []
                for ordinal, (name, sign) in enumerate(selection):
                    source_id = "operation:%s" % _reference(module.operator_handle(name))
                    original = _native_rhs_method(by_identity[source_id], value)
                    prototype = evaluations[original.evaluation].occurrences[0]
                    term = TermOccurrence("%s/term:%d" % (evaluation, ordinal),
                        prototype.operator, prototype.target,
                        "transport" if sign < 0 else "source", sign)
                    terms.append(term)
                    identity = "%s/%s/term:%d" % (source_id, location, ordinal)
                    context = {"node_id": value.id, "operation": value.op,
                        "clock": _projection(value.clock), "point": _projection(value.point),
                        "input_values": [item.id for item in value.inputs],
                        "schedule": _projection(value.attrs.get("schedule")),
                        "selection": {"name": name, "coefficient": sign}}
                    clones.append(replace(original, identity=identity, evaluation=evaluation,
                        consumes=(term.identity,), guards=tuple(guards),
                        exchanges=((ExchangeRecord(term.identity, term.target),) if sign < 0 else ()),
                        guarantees={**original.guarantees, "program_evaluation": context}))
                requests.append(EvaluationRequest(evaluation, tuple(terms), location))
                operations.extend(clones)
                values_to_operations[id(value)] = tuple(op.identity for op in clones)
            continue
        instance = handle.block_ref
        if instance is not None and str(handle.owner_path.canonical()) != block.instance_owner_qid:
            continue
        declaration = handle.declaration_ref or handle
        source_id = "operation:%s" % _reference(declaration)
        if source_id not in by_identity:
            continue  # Another block's independently authored Module.
        original = _native_rhs_method(by_identity[source_id], value)
        identity = "%s/%s" % (source_id, location)
        evaluation = "evaluation:%s" % identity
        requests.append(EvaluationRequest(evaluation,
            evaluations[original.evaluation].occurrences, location))
        dependencies = tuple(dict.fromkeys(identity
                                           for item in value.inputs
                                           for identity in values_to_operations.get(id(item), ())))
        # SSA outputs are materialized by the current Program engine. Stencil
        # consumers exchange those values; their original input halo is not reused.
        materialized = dependencies if original.stencil_radius else ()
        communication = tuple("halo_exchange:%s" % item for item in materialized)
        context = {"node_id": value.id, "operation": value.op,
                   "clock": _projection(value.clock), "point": _projection(value.point),
                   "input_values": [item.id for item in value.inputs],
                   "schedule": _projection(value.attrs.get("schedule"))}
        operation = replace(original, identity=identity, evaluation=evaluation,
            dependencies=dependencies, materialized_inputs=materialized,
            communication=communication, guards=tuple(guards),
            guarantees={**original.guarantees, "program_evaluation": context})
        operations.append(operation)
        values_to_operations[id(value)] = (identity,)
    plan = build_resolved_operations(module, constructions=operations,
                                    evaluation_requests=requests, boundary_data=boundaries)
    from .program_field_publication import attach_publication_claims, publication_claims

    return attach_publication_claims(plan, module, publication_claims(block, program))


def resolved_operation_mapping(blocks: Any):
    from types import MappingProxyType
    return MappingProxyType({block.name: block.resolved_operations for block in blocks})


def operation_coverage(blocks: Any):
    """Use the existing lowering report with collision-free block qualification."""
    from .lowering_coverage import LoweringCoverageReport, LoweringCoverageRow
    return LoweringCoverageReport(tuple(LoweringCoverageRow(
        source="block:%s/resolved:%s" % (block.name, row.source),
        disposition=row.disposition,
        targets=tuple("block:%s/resolved:%s" % (block.name, target) for target in row.targets),
        rule=row.rule, gate=row.gate)
        for block in blocks for row in block.resolved_operations.coverage.rows))
