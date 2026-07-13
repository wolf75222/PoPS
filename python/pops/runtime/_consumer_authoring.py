"""Callback-free policy-to-ConsumerGraph authoring contracts."""
from __future__ import annotations

import re
from copy import copy
from dataclasses import dataclass
from typing import Any

from pops.identity import make_identity
from pops.model import Handle, OwnerKind, OwnerPath
from pops.time import Schedule

from ._consumer_contracts import (
    ConsumerFailureAction,
    ConsumerKind,
    ConsumerManifest,
    ConsumerOperation,
    ConsumerQuantity,
    FailRun,
    ParallelMode,
    _FAILURE_ACTIONS,
)


_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")
_QUANTITY_KINDS = frozenset({"state", "field", "aux", "block"})


def _protocol(value: Any, method: str, *, where: str) -> Any:
    member = getattr(value, method, None)
    if not callable(member):
        raise TypeError("%s must implement %s()" % (where, method))
    return member


def _references(value: Any, *, where: str) -> tuple[Handle, ...]:
    supplied = _protocol(value, "declaration_references", where=where)()
    if not isinstance(supplied, tuple):
        raise TypeError("%s.declaration_references() must return a tuple" % where)
    if any(not isinstance(reference, Handle) for reference in supplied):
        raise TypeError("%s.declaration_references() must contain only Handles" % where)
    if len(set(supplied)) != len(supplied):
        raise ValueError("%s.declaration_references() contains duplicates" % where)
    return supplied


def _frozen_descriptor(value: Any, *, where: str) -> Any:
    if isinstance(value, (str, bytes)):
        raise TypeError("%s must be a typed descriptor, not text" % where)
    clone = copy(value)
    freeze = _protocol(clone, "freeze", where=where)
    if freeze() is not clone:
        raise TypeError("%s.freeze() must return self" % where)
    return clone


@dataclass(frozen=True, slots=True, init=False)
class ConsumerOperationAuthoring:
    """One open operation name, inert configuration and typed diagnostic descriptors."""

    name: str
    configuration: dict[str, Any]
    diagnostics: tuple[Any, ...]

    def __init__(self, name: Any, configuration: Any, diagnostics: Any = ()) -> None:
        if not isinstance(name, str) or not name or name.strip() != name:
            raise TypeError("consumer operation name must be canonical text")
        if not isinstance(configuration, dict):
            raise TypeError("consumer operation configuration must be a dict")
        rows = tuple(
            _frozen_descriptor(value, where="consumer diagnostic") for value in diagnostics)
        for index, value in enumerate(rows):
            where = "consumer diagnostic %d" % index
            _references(value, where=where)
            _protocol(value, "resolve_references", where=where)
            _protocol(value, "consumer_data", where=where)
        # ConsumerOperation owns the strict canonical value validation and deep freeze.
        frozen = ConsumerOperation(name, configuration)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "configuration", frozen.to_data()["data"])
        object.__setattr__(self, "diagnostics", rows)

    def declaration_references(self) -> tuple[Handle, ...]:
        result = []
        for index, diagnostic in enumerate(self.diagnostics):
            for reference in _references(
                    diagnostic, where="consumer diagnostic %d" % index):
                if reference not in result:
                    result.append(reference)
        return tuple(result)

    def resolve(self, resolver: Any) -> ConsumerOperation:
        if not callable(resolver):
            raise TypeError("consumer operation resolver must be callable")
        diagnostics = []
        for index, diagnostic in enumerate(self.diagnostics):
            where = "consumer diagnostic %d" % index
            resolved = _protocol(
                diagnostic, "resolve_references", where=where)(resolver)
            references = tuple(resolver(reference) for reference in _references(
                resolved, where=where))
            if any(not isinstance(reference, Handle) or not reference.is_resolved
                   for reference in references):
                raise TypeError("resolved diagnostic references must be canonical Handles")
            diagnostics.append({
                "descriptor": _protocol(resolved, "consumer_data", where=where)(),
                "references": [reference.canonical_identity() for reference in references],
            })
        data = dict(self.configuration)
        data["diagnostics"] = diagnostics
        return ConsumerOperation(self.name, data)

    def inspect(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "configuration": dict(self.configuration),
            "diagnostics": [diagnostic.inspect() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class ConsumerAuthoringNode:
    """One immutable direct consumer declaration before Case/layout resolution."""

    label: str
    kind: ConsumerKind
    references: tuple[Handle, ...]
    schedule: Schedule
    target_uri: str
    output_format: str
    parallel_mode: ParallelMode
    levels: Any
    operation: ConsumerOperationAuthoring
    failure_action: ConsumerFailureAction = FailRun()

    def __post_init__(self) -> None:
        for name in ("label", "target_uri", "output_format"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise TypeError("ConsumerAuthoringNode.%s must be canonical text" % name)
        if type(self.kind) is not ConsumerKind:
            raise TypeError("ConsumerAuthoringNode.kind must be an exact ConsumerKind")
        if not isinstance(self.references, tuple) or any(
                not isinstance(reference, Handle) for reference in self.references):
            raise TypeError("ConsumerAuthoringNode.references must contain Handles")
        if any(reference.kind not in _QUANTITY_KINDS for reference in self.references):
            raise TypeError("consumer quantities must be state, field, aux, or block Handles")
        if len(set(self.references)) != len(self.references):
            raise ValueError("ConsumerAuthoringNode contains duplicate quantity references")
        if type(self.schedule) is not Schedule:
            raise TypeError("ConsumerAuthoringNode.schedule must be an exact Schedule")
        if type(self.parallel_mode) is not ParallelMode:
            raise TypeError("ConsumerAuthoringNode.parallel_mode must be an exact ParallelMode")
        _protocol(self.levels, "select_levels", where="consumer level selection")
        _protocol(self.levels, "to_data", where="consumer level selection")
        if type(self.operation) is not ConsumerOperationAuthoring:
            raise TypeError(
                "ConsumerAuthoringNode.operation must be an exact ConsumerOperationAuthoring")
        if type(self.failure_action) not in _FAILURE_ACTIONS:
            raise TypeError("ConsumerAuthoringNode.failure_action has an unsupported type")

    def declaration_references(self) -> tuple[Handle, ...]:
        result = list(self.references)
        for reference in self.operation.declaration_references():
            if reference not in result:
                result.append(reference)
        return tuple(result)

    def canonical_data(self, resolver: Any) -> dict[str, Any]:
        if not callable(resolver):
            raise TypeError("ConsumerAuthoringNode resolver must be callable")
        references = tuple(resolver(reference) for reference in self.references)
        if any(not isinstance(reference, Handle) or not reference.is_resolved
               for reference in references):
            raise TypeError("consumer resolver must return canonical Handles")
        operation = self.operation.resolve(resolver)
        return {
            "label": self.label,
            "kind": self.kind.value,
            "references": [reference.canonical_identity() for reference in references],
            "schedule": self.schedule.to_data(),
            "target_uri": self.target_uri,
            "output_format": self.output_format,
            "parallel_mode": self.parallel_mode.value,
            "levels": self.levels.to_data(),
            "operation": operation.to_data(),
            "failure_action": self.failure_action.to_data(),
        }

    @staticmethod
    def _layout_subject(reference: Handle) -> Handle:
        if reference.kind in {"state", "field", "block"}:
            return reference
        if reference.kind == "aux" and reference.block_ref is not None:
            return reference.block_ref
        raise TypeError(
            "consumer quantity %s has no materialized layout subject" % reference.qualified_id)

    def resolve(self, resolver: Any, layout_plan: Any, *, owner: Any) -> ConsumerManifest:
        case_owner = OwnerPath.coerce(owner)
        references = tuple(resolver(reference) for reference in self.references)
        quantities = []
        layout_rows = []
        for reference in references:
            layout = layout_plan.layout_for(self._layout_subject(reference))
            normalized = layout_plan.normalized(layout)
            levels = self.levels.select_levels(normalized)
            quantities.append(ConsumerQuantity(
                reference,
                "declaration:%s" % reference.qualified_id,
                layout.qualified_id,
                levels,
            ))
            layout_rows.append({
                "reference": reference.canonical_identity(),
                "layout": layout.canonical_identity(),
                "levels": list(levels),
            })
        seed = self.canonical_data(resolver)
        seed["resolved_layouts"] = layout_rows
        digest = make_identity("consumer-authoring-node", seed).hexdigest[:16]
        label = _SAFE_NAME.sub("-", self.label).strip("-.").lower() or "consumer"
        handle = Handle(
            "%s-%s" % (label, digest),
            kind="consumer",
            owner=case_owner.child(OwnerKind.CONSUMER, "graph"),
        )
        return ConsumerManifest(
            handle=handle,
            kind=self.kind,
            quantities=tuple(quantities),
            schedule=self.schedule,
            target_uri=self.target_uri,
            output_format=self.output_format,
            parallel_mode=self.parallel_mode,
            failure_action=self.failure_action,
            operation=self.operation.resolve(resolver),
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind.value,
            "references": [reference.inspect() for reference in self.references],
            "schedule": self.schedule.to_data(),
            "target_uri": self.target_uri,
            "output_format": self.output_format,
            "parallel_mode": self.parallel_mode.value,
            "levels": self.levels.to_data(),
            "operation": self.operation.inspect(),
            "failure_action": self.failure_action.to_data(),
        }


__all__ = ["ConsumerAuthoringNode", "ConsumerOperationAuthoring"]
