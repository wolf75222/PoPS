"""Pure resolve-time layout planning, qualification, and runtime-provider gates."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any


class LayoutCapabilityError(ValueError):
    """A resolved LayoutPlan cannot be served by the currently selected runtime route."""

    def __init__(self, message: str, *, evidence: dict[str, Any], coverage_report: Any) -> None:
        super().__init__(message)
        self.evidence = evidence
        self.coverage_report = coverage_report

    @property
    def report(self) -> Any:
        return self.coverage_report


@dataclass(frozen=True, slots=True)
class ResolvedLayoutAuthority:
    """Canonical plan plus an orchestration-only provider, never hidden inside plan identity."""

    plan: Any
    runtime_descriptor: Any = None

    def __post_init__(self) -> None:
        from pops.mesh import LayoutPlan

        if type(self.plan) is not LayoutPlan:
            raise TypeError("ResolvedLayoutAuthority.plan must be an exact LayoutPlan")
        if self.runtime_descriptor is not None:
            if len(self.plan.layouts) != 1:
                raise ValueError(
                    "one runtime descriptor cannot serve a heterogeneous LayoutPlan")
            _authenticate_runtime_descriptor(self.plan, self.runtime_descriptor)

    def require_runtime(self) -> Any:
        if len(self.plan.layouts) != 1:
            _refuse_runtime(
                self.plan,
                gate="multi_layout_runtime_unavailable",
                message=(
                    "pops.resolve proved the heterogeneous LayoutPlan, but this runtime supports "
                    "only one materialized layout; refusing before artifact creation"
                ),
            )
        if self.runtime_descriptor is None:
            _refuse_runtime(
                self.plan,
                gate="layout_runtime_provider_unavailable",
                message=(
                    "pops.resolve received a canonical LayoutPlan without a separately "
                    "authenticated runtime provider; refusing before artifact creation"
                ),
            )
        return self.runtime_descriptor


def materialized_layout_subjects(problem: Any) -> dict[str, tuple[Any, ...]]:
    """Enumerate exact canonical blocks, state instances, and case fields without representatives."""
    protocol = getattr(problem, "layout_subjects", None)
    if not callable(protocol):
        raise TypeError("layout planning requires the Case materialized-subject protocol")
    snapshot = protocol()
    project = getattr(snapshot, "to_dict", None)
    if not callable(project):
        raise TypeError("Case.layout_subjects() must return a typed LayoutSubjects snapshot")
    return project()


def validate_layout(problem: Any, layout: Any) -> None:
    """Validate the one runtime provider selected after LayoutPlan resolution."""
    capabilities = layout.capabilities()
    to_dict = getattr(capabilities, "to_dict", None)
    capabilities = to_dict() if callable(to_dict) else capabilities
    if not isinstance(capabilities, Mapping):
        raise TypeError("layout capabilities() must project to a mapping")
    supports_amr = capabilities.get("supports_amr", False)
    if type(supports_amr) is not bool:
        raise TypeError("layout capability supports_amr must be an exact bool")
    if getattr(layout, "refine", None) is not None and not supports_amr \
            and getattr(layout, "ignore_amr", None) is None:
        raise ValueError(
            "pops.resolve: selected layout cannot consume active AMR refinement criteria")
    context = {"layout": layout}
    layout.validate(context)
    problem._field_registry.validate(context).raise_if_error()


def validate_program_layout_reads(problem: Any, plan: Any, *, time: Any = None) -> None:
    """Prove every materialized Program read belongs to one resolved layout."""
    from pops.model import Handle

    def assigned(reference: Any) -> Any:
        if not isinstance(reference, Handle) or reference.kind not in ("state", "field", "block"):
            return None
        canonical = problem.resolve(reference)
        return plan.layout_for(canonical)

    if time is None:
        return
    for value in getattr(time, "_values", ()):
        targets = {layout for layout in (
            assigned(getattr(value, "state_ref", None)),
            assigned(getattr(value, "block", None)),
        ) if layout is not None}
        if len(targets) > 1:
            raise ValueError(
                "Program value %r has state/block references assigned to different layouts"
                % getattr(value, "name", "<unnamed>"))
        target = next(iter(targets), None)
        if target is None:
            continue
        source_layouts = set()
        for source in getattr(value, "inputs", ()):
            for reference in (getattr(source, "state_ref", None), getattr(source, "block", None)):
                source_layout = assigned(reference)
                if source_layout is not None:
                    source_layouts.add(source_layout)
        for source in source_layouts - {target}:
            matches = [mapping for mapping in plan.mappings
                       if mapping.requirement.source == source
                       and mapping.requirement.target == target]
            if not matches:
                raise ValueError(
                    "Program read requires an explicit layout mapping %s -> %s"
                    % (source.qualified_id, target.qualified_id))


def resolve_layout(problem: Any, layout: Any, *, providers: Any = None) \
        -> ResolvedLayoutAuthority:
    """Resolve one LayoutPlan authority and keep opaque runtime data outside its identity."""
    from pops.mesh import LayoutPlan, normalize_layout_plan
    from pops.problem._detached import detached_frozen

    if layout is None:
        raise ValueError("pops.resolve requires one layout descriptor or LayoutPlan")
    selected = layout

    subjects = materialized_layout_subjects(problem)
    if isinstance(selected, LayoutPlan):
        if selected.owner != problem.owner_path.canonical():
            raise ValueError("LayoutPlan owner does not match the frozen Case authority")
        selected.validate_subjects(**subjects)
        runtime_descriptor = _select_runtime_provider(selected, providers)
        return ResolvedLayoutAuthority(selected, runtime_descriptor)

    if providers:
        raise ValueError(
            "layout_providers is only valid when layout= is an explicit LayoutPlan")

    descriptor = _resolve_descriptor(problem, selected)
    validate_layout(problem, descriptor)
    runtime_descriptor = detached_frozen(descriptor)
    plan = normalize_layout_plan(
        runtime_descriptor,
        owner=problem.owner_path.canonical(),
        states=subjects["states"],
        fields=subjects["fields"],
        blocks=subjects["blocks"],
        handle_resolver=lambda value: (
            value if getattr(value, "is_resolved", False) else problem.resolve(value)
        ),
    )
    plan.validate_subjects(**subjects)
    return ResolvedLayoutAuthority(plan, runtime_descriptor)


def _select_runtime_provider(plan: Any, providers: Any) -> Any:
    if providers is None:
        return None
    if not isinstance(providers, Mapping):
        raise TypeError("layout_providers must be a {LayoutHandle: descriptor} mapping")
    from pops.mesh import LayoutHandle

    if any(not isinstance(handle, LayoutHandle) for handle in providers):
        raise TypeError("layout_providers keys must be canonical LayoutHandle values")
    from pops.problem._detached import detached_frozen

    declared = {row.handle for row in plan.layouts}
    unknown = [handle.qualified_id for handle in providers if handle not in declared]
    if unknown:
        raise ValueError("layout_providers contains undeclared layouts: %s" % sorted(unknown))
    detached = {}
    for handle, descriptor in providers.items():
        row = plan.normalized(handle)
        runtime_descriptor = detached_frozen(descriptor)
        candidate = _normalized_runtime_descriptor(handle, runtime_descriptor)
        if candidate.to_data() != row.to_data():
            raise ValueError(
                "runtime layout provider %s does not authenticate LayoutPlan snapshot"
                % handle.qualified_id)
        detached[handle] = runtime_descriptor
    if len(plan.layouts) != 1:
        return None
    return detached.get(plan.layouts[0].handle)


def _resolve_descriptor(problem: Any, selected: Any) -> Any:
    """Authenticate any layout implementation through one fail-closed extension protocol."""
    protocol = getattr(selected, "resolve_for_case", None)
    if not callable(protocol):
        raise TypeError(
            "pops.resolve layout descriptors must implement resolve_for_case(resolver); "
            "concrete layout classes and names are never dispatched centrally"
        )

    def resolver(value: Any) -> Any:
        return value if getattr(value, "is_resolved", False) else problem.resolve(value)

    descriptor = protocol(resolver)
    if descriptor is None or not all(callable(getattr(descriptor, name, None)) for name in (
            "validate", "capabilities", "requirements", "options")):
        raise TypeError(
            "layout resolve_for_case() must return a typed descriptor implementing the "
            "layout metadata protocol"
        )
    return descriptor


def _authenticate_runtime_descriptor(plan: Any, descriptor: Any) -> None:
    row = plan.layouts[0]
    candidate = _normalized_runtime_descriptor(row.handle, descriptor)
    if candidate.to_data() != row.to_data():
        raise ValueError("runtime layout provider does not authenticate LayoutPlan snapshot")


def _normalized_runtime_descriptor(handle: Any, descriptor: Any) -> Any:
    from pops.mesh.layout_plan import normalize_layout

    return normalize_layout(handle, descriptor)


def layout_lowering_coverage(plan: Any, *, rejected_gate: str | None = None) -> Any:
    from pops.codegen.lowering_coverage import LoweringCoverageReport, LoweringCoverageRow

    rows = [
        LoweringCoverageRow(
            source="layout:%s" % row.handle.qualified_id,
            disposition="lowered",
            targets=("normalized-layout:%s" % row.handle.qualified_id,),
        )
        for row in plan.layouts
    ]
    rows.extend(
        LoweringCoverageRow(
            source="layout-assignment:%s" % assignment.subject_id,
            disposition="derived",
            rule="exact-layout-assignment",
        )
        for assignment in plan.assignments
    )
    rows.extend(
        LoweringCoverageRow(
            source="layout-mapping:%s" % mapping.requirement.qualified_id,
            disposition="lowered",
            targets=("layout-provider:%s" % mapping.provider_id,),
        )
        for mapping in plan.mappings
    )
    if rejected_gate is not None:
        rows.append(LoweringCoverageRow(
            source="layout-runtime:%s" % plan.qualified_id,
            disposition="rejected", gate=rejected_gate))
    else:
        rows.append(LoweringCoverageRow(
            source="layout-runtime:%s" % plan.qualified_id,
            disposition="lowered", targets=("runtime-layout-provider",)))
    return LoweringCoverageReport(rows)


def _refuse_runtime(plan: Any, *, gate: str, message: str) -> None:
    coverage = layout_lowering_coverage(plan, rejected_gate=gate)
    evidence = {
        "gate": gate,
        "layout_plan": plan.inspect(),
        "capabilities": plan.capability_evidence(),
        "resources": list(plan.resource_requirements()),
        "lowering_coverage": coverage.to_data(),
    }
    raise LayoutCapabilityError(message, evidence=evidence, coverage_report=coverage)


__all__ = [
    "LayoutCapabilityError", "ResolvedLayoutAuthority", "layout_lowering_coverage",
    "materialized_layout_subjects", "resolve_layout", "validate_layout",
    "validate_program_layout_reads",
]
