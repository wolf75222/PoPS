"""Output-owned callback-free consumer-to-ConsumerGraph authoring contracts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pops.identity import make_identity
from pops.model import Handle, OwnerKind, OwnerPath
from pops.time import Schedule

from ._consumer_contracts import (
    ConsumerFailureAction,
    ConsumerKind,
    ConsumerManifest,
    ConsumerQuantity,
    DiagnosticQuantity,
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
    from copy import copy as shallow_copy

    clone = shallow_copy(value)
    freeze = _protocol(clone, "freeze", where=where)
    if freeze() is not clone:
        raise TypeError("%s.freeze() must return self" % where)
    return clone


@dataclass(frozen=True, slots=True)
class ConsumerAuthoringNode:
    """One immutable direct consumer declaration before Case/layout resolution."""

    label: str
    kind: ConsumerKind
    references: tuple[Handle, ...]
    schedule: Schedule
    target_uri: str
    output_format: Any
    parallel_mode: ParallelMode
    levels: Any
    operation: Any
    diagnostics: tuple[Any, ...] = ()
    failure_action: ConsumerFailureAction = FailRun()

    def __post_init__(self) -> None:
        for name in ("label", "target_uri"):
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
        rows = tuple(
            _frozen_descriptor(value, where="consumer diagnostic")
            for value in self.diagnostics)
        for index, value in enumerate(rows):
            where = "consumer diagnostic %d" % index
            _references(value, where=where)
            _protocol(value, "resolve_references", where=where)
            _protocol(value, "consumer_data", where=where)
            _protocol(value, "diagnostic_execution", where=where)
        object.__setattr__(self, "diagnostics", rows)
        if self.kind is ConsumerKind.SCIENTIFIC_OUTPUT:
            from pops.output.provider import consumer_format_data
            consumer_format_data(self.output_format, where="ConsumerAuthoringNode.output_format")
            if self.operation is not None:
                raise ValueError("ScientificOutput carries no competing operation provider")
        elif self.kind is ConsumerKind.CHECKPOINT:
            if self.output_format is not None or self.operation is None:
                raise ValueError("Checkpoint requires only its restart operation provider")
        elif self.output_format is not None or self.operation is not None:
            raise ValueError("Diagnostic/Monitor authoring carries no publication provider")
        if type(self.failure_action) not in _FAILURE_ACTIONS:
            raise TypeError("ConsumerAuthoringNode.failure_action has an unsupported type")

    def declaration_references(self) -> tuple[Handle, ...]:
        result = list(self.references)
        for index, diagnostic in enumerate(self.diagnostics):
            for reference in _references(
                    diagnostic, where="consumer diagnostic %d" % index):
                if reference not in result:
                    result.append(reference)
        return tuple(result)

    def canonical_data(self, resolver: Any) -> dict[str, Any]:
        if not callable(resolver):
            raise TypeError("ConsumerAuthoringNode resolver must be callable")
        references: list[Handle] = []
        for reference in self.references:
            resolved_reference = resolver(reference)
            if not isinstance(resolved_reference, Handle) or not resolved_reference.is_resolved:
                raise TypeError("consumer resolver must return canonical Handles")
            references.append(resolved_reference)
        diagnostics = []
        for index, diagnostic in enumerate(self.diagnostics):
            where = "consumer diagnostic %d" % index
            resolved = _protocol(diagnostic, "resolve_references", where=where)(resolver)
            diagnostic_references = []
            for reference in _references(resolved, where=where):
                resolved_reference = resolver(reference)
                if not isinstance(resolved_reference, Handle) \
                        or not resolved_reference.is_resolved:
                    raise TypeError("consumer resolver must return canonical Handles")
                diagnostic_references.append(resolved_reference.canonical_identity())
            diagnostics.append({
                "descriptor": _protocol(resolved, "consumer_data", where=where)(),
                "references": diagnostic_references,
            })
        output_data = None if self.output_format is None else self.output_format.consumer_data()
        operation_data = None if self.operation is None else self.operation.consumer_data()
        return {
            "label": self.label,
            "kind": self.kind.value,
            "references": [reference.canonical_identity() for reference in references],
            "schedule": self.schedule.to_data(),
            "target_uri": self.target_uri,
            "output_format": output_data,
            "parallel_mode": self.parallel_mode.value,
            "levels": self.levels.to_data(),
            "operation": operation_data,
            "diagnostics": diagnostics,
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

    @staticmethod
    def _diagnostic_state(
        diagnostic: Any,
        references: tuple[Handle, ...],
        layout_plan: Any,
        *,
        where: str,
    ) -> Handle:
        """Resolve one diagnostic to the sole exact state it reduces."""
        declared = _references(diagnostic, where=where)
        blocks = tuple(reference for reference in declared if reference.kind == "block")
        states = tuple(reference for reference in declared if reference.kind == "state")
        if any(reference.kind not in {"block", "state"} for reference in declared):
            raise TypeError("%s may reference only one block or one state" % where)
        assigned_states = tuple(
            assignment.subject
            for assignment in layout_plan.assignments
            if assignment.subject_kind == "state"
        )
        if states:
            candidates = states
        elif blocks:
            if len(blocks) != 1:
                raise ValueError("%s must select exactly one block" % where)
            candidates = tuple(
                state for state in assigned_states if state.block_ref == blocks[0])
        else:
            referenced_states = tuple(
                reference for reference in references if reference.kind == "state")
            referenced_blocks = {
                reference.block_ref for reference in referenced_states
                if reference.block_ref is not None
            }
            if len(referenced_blocks) == 1:
                block = next(iter(referenced_blocks))
                candidates = tuple(
                    state for state in assigned_states if state.block_ref == block)
            else:
                candidates = assigned_states
        unique = {state.canonical_identity()["qualified_id"]: state for state in candidates}
        if len(unique) != 1:
            raise ValueError(
                "%s does not resolve to exactly one conservative state; "
                "qualify the diagnostic with block=<BlockHandle>" % where)
        state = next(iter(unique.values()))
        if state.kind != "state" or not state.is_resolved:
            raise TypeError("%s resolved to a non-canonical state" % where)
        return state

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
        format_data = None
        if self.kind is ConsumerKind.SCIENTIFIC_OUTPUT:
            from pops.output.provider import consumer_format_data

            format_data = consumer_format_data(
                self.output_format,
                where="ConsumerAuthoringNode.output_format",
            )
        resolved_diagnostics = tuple(
            _protocol(value, "resolve_references", where="consumer diagnostic")(
                resolver)
            for value in self.diagnostics)
        seed = self.canonical_data(resolver)
        seed["resolved_layouts"] = layout_rows
        digest = make_identity("consumer-authoring-node", seed).hexdigest[:16]
        label = _SAFE_NAME.sub("-", self.label).strip("-.").lower() or "consumer"
        handle = Handle(
            "%s-%s" % (label, digest),
            kind="consumer",
            owner=case_owner.child(OwnerKind.CONSUMER, "graph"),
        )
        diagnostic_quantities = []
        diagnostic_owner = handle.owner_path.child(
            OwnerKind.DESCRIPTOR, handle.local_id,
        ).child(OwnerKind.DESCRIPTOR, "diagnostics")
        for index, diagnostic in enumerate(resolved_diagnostics):
            where = "consumer diagnostic %d" % index
            state = self._diagnostic_state(
                diagnostic, references, layout_plan, where=where)
            layout = layout_plan.layout_for(state)
            normalized = layout_plan.normalized(layout)
            levels = self.levels.select_levels(normalized)
            execution = _protocol(
                diagnostic, "diagnostic_execution", where=where)()
            diagnostic_seed = {
                "descriptor": _protocol(
                    diagnostic, "consumer_data", where=where)(),
                "state": state.canonical_identity(),
                "layout": layout.canonical_identity(),
                "levels": list(levels),
                "execution": execution,
            }
            diagnostic_digest = make_identity(
                "consumer-diagnostic-authoring", diagnostic_seed).hexdigest[:16]
            diagnostic_quantities.append(DiagnosticQuantity(
                Handle(
                    "diagnostic-%s" % diagnostic_digest,
                    kind="diagnostic",
                    owner=diagnostic_owner,
                ),
                state,
                "declaration:%s" % state.qualified_id,
                layout.qualified_id,
                levels,
                execution,
            ))
        if format_data is not None:
            selection_contract = format_data.get("selection_contract")
            selected_layouts = {
                quantity.layout_id
                for quantity in (*quantities, *diagnostic_quantities)
            }
            if selection_contract is not None \
                    and selection_contract["layout_cardinality"] == "single" \
                    and len(selected_layouts) > 1:
                raise ValueError(
                    "scientific output format %s accepts one exact layout per consumer; "
                    "declare one ScientificOutput per layout"
                    % format_data["provider_id"])
        return ConsumerManifest(
            handle=handle,
            kind=self.kind,
            quantities=tuple(quantities),
            schedule=self.schedule,
            target_uri=self.target_uri,
            output_format=self.output_format,
            parallel_mode=self.parallel_mode,
            failure_action=self.failure_action,
            operation=self.operation,
            diagnostics=resolved_diagnostics,
            diagnostic_quantities=tuple(diagnostic_quantities),
        )

    def inspect(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind.value,
            "references": [reference.inspect() for reference in self.references],
            "schedule": self.schedule.to_data(),
            "target_uri": self.target_uri,
            "output_format": None if self.output_format is None
            else self.output_format.consumer_data(),
            "parallel_mode": self.parallel_mode.value,
            "levels": self.levels.to_data(),
            "operation": None if self.operation is None else self.operation.consumer_data(),
            "diagnostics": [diagnostic.inspect() for diagnostic in self.diagnostics],
            "failure_action": self.failure_action.to_data(),
        }


__all__ = ["ConsumerAuthoringNode"]
