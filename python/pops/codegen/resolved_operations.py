"""One immutable numerical plan between scientific Module IR and native consumers.

Resolution records a realization or refusal. It never claims that compilation,
execution, numerical verification or performance characterization has occurred.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pops.identity.digest import Identity, make_identity

from ._resolved_operation_records import (
    EvaluationRequest, ExchangeRecord, NumericalConstruction, ResolvedAccess,
    ResolvedOperation, TermOccurrence, data, freeze, identifier, identifiers,
    require_semantic_rewrite,
)
from .lowering_coverage import LoweringCoverageReport, LoweringCoverageRow, LoweringRejection


def _reject(source: str, gate: str, message: str, rows: Any = ()) -> None:
    report = LoweringCoverageReport((*rows, LoweringCoverageRow(source, "rejected", gate=gate)))
    raise LoweringRejection(message, coverage_report=report, source=source, gate=gate)


def _coverage(evaluations: tuple[EvaluationRequest, ...],
              operations: tuple[NumericalConstruction, ...]) -> LoweringCoverageReport:
    requests = {item.identity: item for item in evaluations}
    rows = []
    for operation in operations:
        if operation.evaluation not in requests:
            _reject(operation.identity, "unknown_evaluation", "construction has no requested evaluation")
    for request in evaluations:
        expected = {item.identity: item for item in request.occurrences}
        claimed: dict[str, str] = {}
        for operation in operations:
            if operation.evaluation != request.identity:
                continue
            for occurrence in operation.consumes:
                source = "evaluation:%s/occurrence:%s" % (request.identity, occurrence)
                if occurrence not in expected:
                    _reject(source, "unrequested_term", "construction consumes an unrequested occurrence")
                if occurrence in claimed:
                    _reject(source, "duplicate_term_coverage", "numerical constructions double-count a term")
                claimed[occurrence] = operation.identity
            for exchange in operation.exchanges:
                term = expected.get(exchange.occurrence)
                if term is None or exchange.occurrence not in operation.consumes \
                        or exchange.target != term.target:
                    _reject(operation.identity, "exchange_occurrence_mismatch",
                            "exchange must refer to the consumed occurrence and its physical target")
        missing = set(expected) - set(claimed)
        if missing:
            _reject("evaluation:%s" % request.identity, "incomplete_term_coverage",
                    "requested evaluation has uncovered terms: %s" % sorted(missing))
        for occurrence in request.occurrences:
            rows.append(LoweringCoverageRow(
                "evaluation:%s/occurrence:%s" % (request.identity, occurrence.identity),
                "derived", (claimed[occurrence.identity],),
                rule="signed occurrence preserved by resolved numerical construction"))
    for operation in operations:
        if operation.native_route is None:
            rows.append(LoweringCoverageRow(operation.identity, "rejected", gate=operation.refusal))
        else:
            rows.append(LoweringCoverageRow(
                operation.identity, "derived", ("native-route:%s" % operation.native_route,),
                rule="selected emitter route; native execution is not established by resolution"))
    return LoweringCoverageReport(rows)


def _order(operations: tuple[NumericalConstruction, ...]) -> tuple[str, ...]:
    known = {operation.identity: operation for operation in operations}
    active: set[str] = set()
    done: set[str] = set()
    ordered: list[str] = []

    def visit(identity: str) -> None:
        if identity not in known:
            _reject(identity, "missing_operation_dependency", "operation dependency is not in the plan")
        if identity in active:
            _reject(identity, "explicit_operation_cycle",
                    "explicit operation cycle needs a declared solve realization")
        if identity in done:
            return
        active.add(identity)
        for dependency in known[identity].dependencies:
            visit(dependency)
        active.remove(identity)
        done.add(identity)
        ordered.append(identity)

    for operation in operations:
        visit(operation.identity)
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class ResolvedOperationPlan:
    source_module_hash: str
    evaluations: tuple[EvaluationRequest, ...]
    operations: tuple[NumericalConstruction, ...]
    provider_evidence: Mapping[str, Any] = field(default_factory=dict)
    identity: Identity = field(init=False)
    coverage: LoweringCoverageReport = field(init=False)
    execution_order: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        identifier(self.source_module_hash, "source Module hash")
        for name, cls in (("evaluations", EvaluationRequest), ("operations", NumericalConstruction)):
            values = tuple(getattr(self, name))
            if any(type(item) is not cls for item in values):
                raise TypeError("plan.%s must contain exact %s records" % (name, cls.__name__))
            identifiers((item.identity for item in values), "plan.%s" % name)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "provider_evidence", freeze(self.provider_evidence))
        object.__setattr__(self, "execution_order", _order(self.operations))
        object.__setattr__(self, "coverage", _coverage(self.evaluations, self.operations))
        from ._plans import _evidence

        object.__setattr__(self, "identity", make_identity("resolved_operations",
            _evidence(self._payload(), where="resolved operations")))

    def _payload(self) -> dict[str, Any]:
        return {"schema_version": 1, "source_module_hash": self.source_module_hash,
                "evaluations": [data(item) for item in self.evaluations],
                "operations": [data(item) for item in self.operations],
                "provider_evidence": data(self.provider_evidence),
                "coverage": self.coverage.to_data(), "execution_order": list(self.execution_order)}

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.token}

    @classmethod
    def from_data(cls, value: Any) -> ResolvedOperationPlan:
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version", "source_module_hash", "evaluations", "operations",
            "provider_evidence", "coverage", "execution_order", "identity",
        }:
            raise TypeError("ResolvedOperationPlan data has an invalid schema")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported resolved operation schema")
        result = cls(value["source_module_hash"],
                     tuple(EvaluationRequest.from_data(item) for item in value["evaluations"]),
                     tuple(NumericalConstruction.from_data(item) for item in value["operations"]),
                     value["provider_evidence"])
        if result.to_data() != dict(value):
            raise ValueError("resolved operation identity/evidence is not authentic or canonical")
        return result

    def operation(self, identity: str) -> NumericalConstruction:
        matches = tuple(item for item in self.operations if item.identity == identity)
        if len(matches) != 1:
            raise KeyError("unknown resolved operation %r" % identity)
        return matches[0]

    def require_module(self, module: Any) -> None:
        from pops.model import Module
        if not isinstance(module, Module) or module.module_hash() != self.source_module_hash:
            _reject("resolved-plan", "source_module_drift",
                    "resolved operations do not belong to the supplied current Module")

    def require_provider_packs(self, module: Any) -> Any:
        """Reauthenticate the existing typed carrier; wire data cannot become an emitter."""
        from ._resolved_operation_inputs import provider_evidence, resolve_plan_provider_packs

        self.require_module(module)
        packs = resolve_plan_provider_packs(module, self.operations)
        if data(self.provider_evidence) != provider_evidence(packs):
            _reject("resolved-plan", "provider_plan_drift", "resolved provider access plan changed")
        return packs

    def require_native(self, identity: str, *, module: Any,
                       native_realizations: Mapping[str, str] | None = None) -> str:
        """Guard the actual emitter call against stale source and unavailable current routes."""
        from ._resolved_operation_inputs import native_route, _reference

        self.require_provider_packs(module)
        operation = self.operation(identity)
        if operation.native_route is None:
            raise LoweringRejection(
                "operation has no native realization: %s" % operation.refusal,
                coverage_report=self.coverage, source=identity, gate=operation.refusal)
        from ._resolved_operation_authority import require_operation_authority

        require_operation_authority(self, operation, module)
        current = dict(native_realizations or {})
        if native_realizations is None:
            for definition in module.operator_registry():
                key = "operation:%s" % _reference(module.operator_handle(definition.name))
                route, _ = native_route(module, definition)
                if route is not None:
                    current[key] = route
        declaration = operation.guarantees.get("declaration_operation", identity)
        if current.get(identity, current.get(declaration)) != operation.native_route:
            _reject(identity, "native_realization_unavailable",
                    "current compiler realization does not match the resolved operation")
        return operation.native_route

    def halo_requirements(self) -> Mapping[str, int]:
        """Sequential stencils compose along paths; independent paths take their maximum."""
        depth: dict[str, int] = {}
        for identity in self.execution_order:
            operation = self.operation(identity)
            upstream = max((depth[item] for item in operation.dependencies
                            if item not in operation.materialized_inputs), default=0)
            depth[identity] = operation.stencil_radius + upstream
        return MappingProxyType(depth)

    def can_fuse(self, producer: str, consumer: str) -> bool:
        first, second = self.operation(producer), self.operation(consumer)
        begin, end = self.execution_order.index(producer), self.execution_order.index(consumer)
        if begin >= end:
            return False
        intervening = tuple(self.operation(item) for item in self.execution_order[begin + 1:end])
        if any(item.effects or item.communication or item.reductions or item.stencil_radius
               or item.guards != first.guards for item in intervening):
            return False
        # This conservative gate grants no cross-stencil fusion. Such a realization must
        # explicitly construct a new operation with its enlarged access neighborhood.
        return bool(first.native_route and second.native_route
                    and first.evaluation == second.evaluation
                    and first.sampling == second.sampling and first.guards == second.guards
                    and not first.effects and not second.effects
                    and not first.communication and not second.communication
                    and not first.reductions and not second.reductions
                    and not first.stencil_radius and not second.stencil_radius
                    and producer not in second.materialized_inputs)

    def explain(self, result: str | None = None) -> dict[str, Any]:
        selected = self.operations if result is None else (self.operation(result),)
        depths = self.halo_requirements()
        requests = {item.identity: item for item in self.evaluations}
        return {"identity": self.identity.token, "source_module_hash": self.source_module_hash,
                "operations": [{**data(item), "disposition": item.disposition,
                                "context": requests[item.evaluation].context,
                                "occurrences": [data(term) for term in requests[item.evaluation].occurrences
                                                if term.identity in item.consumes],
                                "effective_input_halo": depths[item.identity],
                                "evidence": {"represented": True, "resolved": True,
                                             "native_route_selected": item.native_route is not None,
                                             "emitted": False, "executed": False,
                                             "numerically_checked": False,
                                             "performance_characterized": False}}
                               for item in selected]}


def build_resolved_operations(module: Any, *, constructions: Any = None,
                              evaluation_requests: Any = None,
                              native_realizations: Mapping[str, str] | None = None,
                              boundary_data: Any = ()) -> ResolvedOperationPlan:
    """Derive the baseline Module route, or validate selected numerical constructions.

    Explicit constructions are compiler choices over the supplied requests, never
    another authored model. Boundary data belongs to the instantiated problem.
    """
    from pops.model import Module
    from .component_provider_packs import resolve_component_provider_packs
    from ._resolved_operation_inputs import (
        derive_module_operations, provider_evidence, resolve_plan_provider_packs,
    )

    if not isinstance(module, Module):
        raise TypeError("resolved operations require a canonical Module")
    packs = resolve_component_provider_packs(module)
    inferred_requests, inferred_operations = derive_module_operations(
        module, packs, boundary_data=boundary_data)
    requests = inferred_requests if evaluation_requests is None else tuple(evaluation_requests)
    operations = inferred_operations if constructions is None else tuple(constructions)
    if native_realizations:
        from dataclasses import replace
        unknown = set(native_realizations) - {item.identity for item in operations}
        if unknown:
            raise ValueError("native realizations name unknown operations: %s" % sorted(unknown))
        operations = tuple(replace(item, native_route=native_realizations[item.identity], refusal=None)
                           if item.identity in native_realizations else item for item in operations)
    packs = resolve_plan_provider_packs(module, operations)
    return ResolvedOperationPlan(module.module_hash(), requests, operations, provider_evidence(packs))


__all__ = ["ResolvedOperationPlan", "ResolvedOperation", "NumericalConstruction",
           "ResolvedAccess", "ExchangeRecord", "EvaluationRequest", "TermOccurrence",
           "build_resolved_operations", "require_semantic_rewrite"]
