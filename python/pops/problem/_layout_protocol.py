"""Small protocol adapters between Case assembly and descriptor/LayoutPlan authorities."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def layout_name(layout: Any) -> Any:
    if layout is None:
        return None
    return getattr(layout, "name", getattr(layout, "qualified_id", None))


def layout_requirements(layout: Any) -> dict[str, Any]:
    if layout is None:
        return {}
    resources = getattr(layout, "resource_requirements", None)
    if callable(resources):
        rows = resources()
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Iterable):
            raise TypeError("layout resource_requirements() must return an iterable")
        return {"layout_resources": list(rows)}
    return layout.requirements().to_dict()


def layout_capabilities(layout: Any) -> dict[str, Any]:
    if layout is None:
        return {}
    evidence = getattr(layout, "capability_evidence", None)
    if callable(evidence):
        return {"layout_plan": evidence()}
    return layout.capabilities().to_dict()


def layout_available(problem: Any, layout: Any, context: Any) -> Any:
    if layout is None:
        return None
    validate_subjects = getattr(layout, "validate_subjects", None)
    if callable(validate_subjects):
        from pops.descriptors import Availability
        try:
            validate_subjects(**materialized_layout_subjects(problem))
        except (TypeError, ValueError) as exc:
            return Availability.no(str(exc), missing=["layout_assignment"])
        return None
    status = layout.available(context)
    return None if status.ok else status


def validate_layout_report(problem: Any, report: Any, layout: Any, context: Any) -> Any:
    if layout is None:
        return report
    validate_subjects = getattr(layout, "validate_subjects", None)
    if callable(validate_subjects):
        try:
            validate_subjects(**materialized_layout_subjects(problem))
        except Exception as exc:  # noqa: BLE001 -- aggregate typed validation evidence
            return report.error("layout", "layout_plan_invalid", str(exc))
        return report
    try:
        layout.validate(context)
    except Exception as exc:  # noqa: BLE001 -- surface descriptor validation evidence
        report = report.error("layout", "layout_invalid", str(exc))
    return report


def field_validation_layout(layout: Any) -> Any:
    return None if callable(getattr(layout, "validate_subjects", None)) else layout


def materialized_layout_subjects(problem: Any) -> dict[str, tuple[Any, ...]]:
    blocks, states = [], []
    for _name, block in sorted(problem.blocks().items()):
        blocks.append(problem.resolve(block))
        for declaration in problem._block_registry.spec(block.local_id)["states"]:
            states.append(problem.resolve(declaration, block=block))
    fields = [problem.resolve(field) for _name, field in sorted(problem.fields().items())]
    def key(value: Any) -> str:
        return value.qualified_id
    return {"blocks": tuple(sorted(blocks, key=key)),
            "states": tuple(sorted(states, key=key)),
            "fields": tuple(sorted(fields, key=key))}


__all__ = [
    "field_validation_layout", "layout_available", "layout_capabilities", "layout_name",
    "layout_requirements", "materialized_layout_subjects", "validate_layout_report",
]
