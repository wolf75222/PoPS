"""Typed FieldContext for a callable Case field operator.

Calling ``field(state, ...)`` creates an internal field-solve IR value. This module attaches a real,
inert descriptor to that value so a stage's field solve is
identifiable and a cross-stage / cross-block read fails loud instead of silently reading a stale
solve. It mirrors the C++ ``pops::FieldContext`` (include/pops/runtime/context/field_context.hpp):
a validity token, not a container -- it holds no field data and changes no numerics.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True, init=False)
class FieldContext:
    """Provenance and validity token for one callable field-operator evaluation.

    Attributes:
        field: exact owner-qualified field identity retained by the Program. Strings, ``None`` and
            default sentinels are rejected: every solve route is explicit and symmetric.
        stage_sources: the ordered, immutable ``((block, state_id), ...)`` provenance of every
            stage state this solve consumed. A single-block solve has one entry; a simultaneous
            coupled solve has one exact entry per participating block. There is deliberately no
            singular ``block`` / ``stage_source`` projection: choosing the first coupled input
            would discard provenance and could validate a stale read from another block.
        outputs: the ordered output handle names this solve produces (``("phi", "grad_x",
            "grad_y")`` for the default), for reports / structured-output lookup.
    """

    field: Any
    stage_sources: tuple[tuple[Any, Any], ...]
    outputs: tuple[Any, ...]
    __pops_ir_immutable__ = True

    def __init__(self, field: Any, stage_sources: Any, outputs: Any = ()) -> None:
        from pops.model import Handle
        if isinstance(field, str) or not isinstance(field, Handle) or field.kind not in (
                "field", "field_operator"):
            raise TypeError(
                "FieldContext field must be an exact owner-qualified field Handle")
        if isinstance(stage_sources, Mapping):
            entries = tuple(stage_sources.items())
        else:
            try:
                entries = tuple(tuple(item) for item in stage_sources)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "FieldContext stage_sources must be a mapping or iterable of (block, state_id) "
                    "pairs") from exc
        if not entries:
            raise ValueError("FieldContext stage_sources must contain at least one block/state pair")
        seen = set()
        for entry in entries:
            if len(entry) != 2:
                raise ValueError(
                    "FieldContext stage_sources entries must be (block, state_id) pairs")
            block, state_id = entry
            from pops.problem.handles import BlockHandle
            if not isinstance(block, BlockHandle):
                raise TypeError("FieldContext blocks must be BlockHandle values")
            if isinstance(state_id, bool) or not isinstance(state_id, int) or state_id < 0:
                raise TypeError("FieldContext state ids must be non-negative Python ints")
            if block in seen:
                raise ValueError("FieldContext stage_sources contains block %r more than once" % block)
            seen.add(block)
        output_names = tuple(outputs)
        if any(not isinstance(name, str) or not name for name in output_names):
            raise TypeError("FieldContext outputs must be non-empty strings")
        if len(set(output_names)) != len(output_names):
            raise ValueError("FieldContext outputs must be unique")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "stage_sources", entries)
        object.__setattr__(self, "outputs", output_names)

    def matches(self, field: Any, block: Any, stage_source: Any) -> Any:
        """True when this context was produced by exactly the requested triple."""
        return ((self.field == field)
                and any(candidate_block == block and candidate_source == stage_source
                        for candidate_block, candidate_source in self.stage_sources))

    def require_read(self, field: Any, block: Any, stage_source: Any) -> Any:
        """Assert a downstream read targets THIS solve, else raise a structured error naming the
        field problem, the block and the stage that mismatched (the ADC-588 incompatible-context
        contract). Returns ``self`` so it composes in an expression.
        """
        if not self.matches(field, block, stage_source):
            raise ValueError(
                "incompatible field context: output of field problem %r solved from stage sources "
                "%r cannot be read as problem %r / block %r / stage source %r"
                % (self.field, self.stage_sources, field, block, stage_source))
        return self

    def output(self, handle: Any) -> Any:
        """Resolve an output handle, raising a structured error listing the known outputs when the
        handle is unknown (never a silent miss). The default problem exposes ``phi`` /
        ``grad_x`` / ``grad_y``; a named field exposes the handles its problem declared.
        """
        if handle in self.outputs:
            return handle
        raise KeyError(
            "unknown field output %r of problem %r; known outputs: %s"
            % (handle, self.field, list(self.outputs)))

    def __repr__(self) -> str:
        return ("FieldContext(field=%r, stage_sources=%r, outputs=%r)"
                % (self.field, self.stage_sources, self.outputs))


@dataclass(frozen=True, slots=True, init=False)
class FieldReadProvenance:
    """Exact set of field solves read by a derived materialized value.

    Multi-stage formulas legitimately combine rates evaluated with different stage solves. Such a
    value has no single FieldContext, but dropping that fact would make later validation opaque.
    This immutable collection retains every distinct token without selecting one by input order.
    """

    contexts: tuple[FieldContext, ...]
    __pops_ir_immutable__ = True

    def __init__(self, contexts: Any) -> None:
        flattened = []
        for item in contexts:
            candidates = item.contexts if isinstance(item, FieldReadProvenance) else (item,)
            for context in candidates:
                if not isinstance(context, FieldContext):
                    raise TypeError(
                        "FieldReadProvenance accepts only FieldContext values (got %r)" % context)
                if context not in flattened:
                    flattened.append(context)
        if len(flattened) < 2:
            raise ValueError(
                "FieldReadProvenance represents two or more distinct field contexts")
        object.__setattr__(self, "contexts", tuple(flattened))

    def contains(self, context: FieldContext) -> bool:
        return context in self.contexts


def merge_field_provenance(*items: Any) -> Any:
    """Return ``None``, one FieldContext, or an immutable multi-context provenance."""
    contexts = []
    for item in items:
        if item is None:
            continue
        candidates = item.contexts if isinstance(item, FieldReadProvenance) else (item,)
        for context in candidates:
            if not isinstance(context, FieldContext):
                raise TypeError("invalid field provenance %r" % (item,))
            if context not in contexts:
                contexts.append(context)
    if not contexts:
        return None
    if len(contexts) == 1:
        return contexts[0]
    return FieldReadProvenance(contexts)


def field_provenance_contains(provenance: Any, context: FieldContext) -> bool:
    if isinstance(provenance, FieldContext):
        return provenance == context
    if isinstance(provenance, FieldReadProvenance):
        return provenance.contains(context)
    return False


def remap_field_provenance(
    provenance: Any,
    remap_source: Any,
    remap_reference: Any = None,
) -> Any:
    """Rebuild provenance after an SSA/reference rewrite.

    ``remap_source(old_id)`` reowns SSA ids.  ``remap_reference(handle)`` is optional and lets the
    compiled-program detachment boundary replace live block/field handles by canonical inert values;
    ordinary optimization passes retain their existing reference identity.
    """
    if remap_reference is None:
        def _identity(value: Any) -> Any:
            return value
        remap_reference = _identity
    if isinstance(provenance, FieldReadProvenance):
        return merge_field_provenance(
            *(
                remap_field_provenance(item, remap_source, remap_reference)
                for item in provenance.contexts
            )
        )
    return FieldContext(
        remap_reference(provenance.field),
        tuple(
            (remap_reference(block), remap_source(source))
            for block, source in provenance.stage_sources
        ),
        provenance.outputs,
    )


def require_field_read(fields: Any, state: Any, where: str, *, allow_derived: bool = False) -> Any:
    """Validate one explicit field-context read.

    ``rhs`` / ``source`` / ``apply`` evaluate physics at a stage State and therefore require the
    context solved from that exact ``(block, state.id)``. ``solve_local_linear`` may instead consume
    a materialized RHS derived from that stage; in that one case ``allow_derived`` accepts the exact
    context propagated on the RHS graph. No block/name heuristic or first-input fallback exists.
    """
    context = getattr(fields, "field_context", None)
    if context is None:
        raise ValueError(
            "%s: fields value does not carry field-solve provenance" % where)
    if allow_derived and field_provenance_contains(
            getattr(state, "field_context", None), context):
        return context
    return context.require_read(context.field, state.block, state.id)


def merge_field_contexts(values: Any, where: str) -> Any:
    """Propagate the one exact field context used to build a materialized State.

    Rate-like inputs describe the current stage evaluation and supersede provenance carried by the
    already-materialized base State. If there is no Rate input, provenance from State inputs is
    retained. Multiple distinct contexts remain explicit in a :class:`FieldReadProvenance` rather
    than choosing one implicitly.
    """
    current = [value.field_context for value in values
               if getattr(value, "vtype", None) == "rhs"
               and getattr(value, "field_context", None) is not None]
    if not current:
        current = [value.field_context for value in values
                   if getattr(value, "vtype", None) == "state"
                   and getattr(value, "field_context", None) is not None]
    return merge_field_provenance(*current)


__all__ = [
    "FieldContext", "FieldReadProvenance",
    "field_provenance_contains", "merge_field_contexts", "merge_field_provenance",
    "remap_field_provenance", "require_field_read",
]
