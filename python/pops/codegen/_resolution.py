"""Pure resolve-time requirement/capability evidence for ADC-660.

This module does not compile, emit, cache, or install anything.  It consumes an already-frozen
``Case`` plus resolved layout/library values and returns one strict JSON-ready proof document.
Unknown evidence is a refusal: compile may consume this result, but may not repair or recompute it.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any


CAPABILITY_EVIDENCE_SCHEMA_VERSION = 2


class CapabilityResolutionError(ValueError):
    """A requirement could not be proven from closed resolve-time evidence."""


def resolve_capability_evidence(
    problem: Any,
    *,
    layout: Any,
    libraries: Any = (),
    time: Any = None,
    module_abi_key: Any = None,
) -> dict[str, Any]:
    """Join requirements and providers before artifact creation.

    ``problem`` must already have crossed validation/freeze.  ``layout`` and ``libraries`` are the
    resolved values selected for this compilation.  The returned mapping contains only strict JSON
    data with deterministic ordering; :func:`canonical_capability_evidence_json` is its canonical
    byte-level representation.
    """
    if not bool(getattr(problem, "frozen", False)):
        raise TypeError(
            "resolve capabilities requires a frozen pops.Case; validate/freeze it before resolve")
    layout_name = _layout_name(layout)
    provider_sources: dict[str, set[str]] = {}
    required: set[str] = set()

    _collect_problem_evidence(problem, provider_sources, required)
    library_rows, external_refs = _library_rows(libraries)
    for source, row in library_rows:
        _add_tokens(provider_sources, source, _tokens(row.get("capabilities")))
        required.update(_tokens(row.get("requirements")))

    provided = set(provider_sources)
    external_evidence = []
    for source, row in library_rows:
        if _is_external_row(row):
            # A component cannot prove its own prerequisites.  Capabilities from other resolved
            # providers remain eligible, including another brick in the same library.
            external_providers = {
                capability for capability, sources in provider_sources.items()
                if sources.difference({source})
            }
            external_evidence.append(
                _resolve_external_row(source, row, layout_name, external_providers))
    for ref in external_refs:
        external_evidence.append(_resolve_external_ref(
            ref, layout_name=layout_name, provided=provided, module_abi_key=module_abi_key))

    amr_evidence = _resolve_amr_program(layout_name, time)
    evidence = {
        "schema_version": CAPABILITY_EVIDENCE_SCHEMA_VERSION,
        "layout": layout_name,
        "providers": [
            {"capability": name, "sources": sorted(provider_sources[name])}
            for name in sorted(provider_sources)
        ],
        "requirements": sorted(required),
        "external_bricks": sorted(external_evidence, key=lambda row: row["id"]),
        "amr_program": amr_evidence,
        "layout_plan": _layout_plan_evidence(layout),
        "layout_resources": _layout_resources(layout),
    }
    # Round-trip through the strict encoder now.  This rejects opaque values/NaN here rather than
    # allowing a later compile/cache layer to invent a representation.
    return json.loads(canonical_capability_evidence_json(evidence))


def canonical_capability_evidence_json(evidence: Any) -> str:
    """Return the canonical JSON encoding of a resolution evidence mapping."""
    if not isinstance(evidence, Mapping):
        raise TypeError("capability evidence must be a mapping")
    try:
        return json.dumps(
            dict(evidence), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("capability evidence must contain strict JSON data: %s" % exc) from exc


def _collect_problem_evidence(
    problem: Any, provider_sources: dict[str, set[str]], required: set[str],
) -> None:
    problem_caps = _projection(getattr(problem, "capabilities", None))
    _add_tokens(provider_sources, "problem", _tokens(problem_caps))
    required.update(_tokens(_projection(getattr(problem, "requirements", None))))
    blocks = getattr(problem, "_blocks", None)
    if blocks is None or not callable(getattr(blocks, "items", None)):
        raise TypeError("frozen Case has no typed block registry")
    for name, spec in blocks.items():
        if not isinstance(spec, Mapping) or spec.get("model") is None:
            raise CapabilityResolutionError("block %r has no resolved model evidence" % name)
        model = spec["model"]
        caps = getattr(model, "provided_capabilities", None)
        if callable(caps):
            caps = caps()
        if caps is not None:
            _add_tokens(provider_sources, "block:%s" % name, _tokens(caps))
        module = getattr(model, "module", None)
        registry = getattr(module if module is not None else model, "operator_registry", None)
        if callable(registry):
            registry = registry()
            names = registry.names()
            for operator_name in names:
                operator = registry.get(operator_name)
                _add_tokens(
                    provider_sources,
                    "block:%s/operator:%s" % (name, operator_name),
                    _tokens(getattr(operator, "capabilities", None)),
                )
                required.update(_tokens(getattr(operator, "requirements", None)))


def _library_rows(libraries: Any) -> tuple[list[tuple[str, Mapping]], list[Any]]:
    try:
        values = tuple(libraries or ())
    except TypeError:
        raise TypeError("resolved libraries must be an iterable") from None
    rows: list[tuple[str, Mapping]] = []
    refs = []
    for index, library in enumerate(values):
        from pops.external.bricks import CompiledBrickRef

        if isinstance(library, CompiledBrickRef):
            refs.append(library)
            continue
        bricks = getattr(library, "bricks", None)
        if bricks is None and isinstance(library, Mapping):
            bricks = library.get("bricks")
        if bricks is None:
            raise TypeError(
                "resolved libraries[%d] must be a LibraryManifest or CompiledBrickRef" % index)
        for row in bricks:
            if not isinstance(row, Mapping):
                raise TypeError("resolved library brick evidence must be a mapping")
            brick_id = row.get("id")
            if not isinstance(brick_id, str) or not brick_id:
                raise CapabilityResolutionError("resolved library brick has no canonical id")
            if row.get("available") is not True:
                raise CapabilityResolutionError(
                    "library brick %r is not proven available during resolution" % brick_id)
            rows.append(("library:%d/brick:%s" % (index, brick_id), row))
    return rows, refs


def _resolve_external_ref(ref: Any, *, layout_name: str, provided: set[str],
                          module_abi_key: Any) -> dict[str, Any]:
    from pops.external._brick_gates import validate_ref

    record = ref.manifest_record()
    if record is None:
        raise CapabilityResolutionError(
            "external brick %r is absent from its manifest" % getattr(ref, "native_id", "<unknown>"))
    context = {
        "canonical_resolution": True,
        "capabilities": sorted(provided),
        "layout": layout_name,
        "module_abi_key": module_abi_key,
    }
    try:
        validate_ref(
            record,
            manifest_abi_key=getattr(ref, "_manifest_abi_key", None),
            context=context,
            handle=getattr(ref, "_handle", None),
            module_abi_key=module_abi_key,
        )
    except (RuntimeError, ValueError) as exc:
        raise CapabilityResolutionError(str(exc)) from exc
    return _external_evidence(record, layout_name, provided)


def _resolve_external_row(
    source: str, row: Mapping, layout_name: str, provided: set[str],
) -> dict[str, Any]:
    required_fields = ("requirements", "capabilities", "options")
    missing = [name for name in required_fields if row.get(name) is None]
    if missing:
        raise CapabilityResolutionError(
            "%s has unknown external brick evidence %s" % (source, sorted(missing)))
    options = row.get("options")
    supported = options.get("supported_layouts") if isinstance(options, Mapping) else None
    if not supported:
        raise CapabilityResolutionError(
            "%s has unknown supported_layouts evidence" % source)
    projected = dict(row)
    projected["supported_layouts"] = supported
    return _external_evidence(projected, layout_name, provided)


def _external_evidence(record: Mapping, layout_name: str, provided: set[str]) -> dict[str, Any]:
    brick_id = record.get("native_id") or record.get("id")
    requirements = sorted(_tokens(record.get("requirements")))
    missing = sorted(set(requirements) - provided)
    if missing:
        raise CapabilityResolutionError(
            "external brick %r requires missing capability %r; providers are %s"
            % (brick_id, missing[0], sorted(provided) or "(none)"))
    supported = sorted(str(value).lower() for value in record.get("supported_layouts", ()))
    if not supported:
        raise CapabilityResolutionError(
            "external brick %r has unknown supported_layouts evidence" % brick_id)
    if layout_name not in supported:
        raise CapabilityResolutionError(
            "external brick %r does not support layout=%s; supported layouts are %s"
            % (brick_id, layout_name, supported))
    return {
        "id": str(brick_id),
        "requirements": requirements,
        "capabilities": sorted(_tokens(record.get("capabilities"))),
        "supported_layouts": supported,
        "status": "proven",
    }


def _resolve_amr_program(layout_name: str, time: Any) -> dict[str, Any]:
    if layout_name != "amr" or time is None:
        return {"groups": [], "status": "not_applicable"}
    from pops.runtime.amr_program_support import amr_program_op_support

    support = amr_program_op_support(time)
    pending = {name: status for name, status in support.items() if status != "green"}
    if pending:
        details = ", ".join("%s=%s" % item for item in sorted(pending.items()))
        raise CapabilityResolutionError(
            "AMR Program uses unsupported capability group(s) %s; resolution refuses before "
            "artifact creation" % details)
    return {
        "groups": [{"name": name, "status": support[name]} for name in sorted(support)],
        "status": "proven",
    }


def _layout_name(layout: Any) -> str:
    if layout is None:
        raise CapabilityResolutionError("resolved layout evidence is missing")
    from pops.mesh import LayoutPlan

    if isinstance(layout, LayoutPlan):
        if len(layout.layouts) != 1:
            raise CapabilityResolutionError(
                "runtime capability resolution requires exactly one normalized layout")
        caps = layout.layouts[0].capabilities
    else:
        caps = _projection(getattr(layout, "capabilities", None))
    token = caps.get("layout") if isinstance(caps, Mapping) else None
    if not isinstance(token, str) or not token or token.strip() != token:
        raise CapabilityResolutionError(
            "resolved layout must declare a canonical capabilities()['layout'] token")
    return token.lower()


def _layout_plan_evidence(layout: Any) -> Any:
    from pops.mesh import LayoutPlan

    return layout.capability_evidence() if isinstance(layout, LayoutPlan) else None


def _layout_resources(layout: Any) -> list[dict[str, Any]]:
    from pops.mesh import LayoutPlan

    return list(layout.resource_requirements()) if isinstance(layout, LayoutPlan) else []


def _projection(value: Any) -> Mapping:
    if callable(value):
        value = value()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    return value if isinstance(value, Mapping) else {}


def _tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(key) for key, enabled in value.items() if bool(enabled)}
    if isinstance(value, str):
        return {value}
    try:
        return {str(item) for item in value}
    except TypeError:
        raise TypeError("capability/requirement evidence must be a mapping or iterable") from None


def _add_tokens(sources: dict[str, set[str]], source: str, tokens: set[str]) -> None:
    for token in tokens:
        sources.setdefault(token, set()).add(source)


def _is_external_row(row: Mapping) -> bool:
    return str(row.get("brick_type", "")).lower() in ("external", "external_cpp")


__all__ = [
    "CAPABILITY_EVIDENCE_SCHEMA_VERSION",
    "CapabilityResolutionError",
    "canonical_capability_evidence_json",
    "resolve_capability_evidence",
]
