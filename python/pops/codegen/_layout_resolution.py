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
class ResolvedRuntimeLayout:
    handle: Any
    descriptor: Any


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeLayouts:
    """Exact authenticated runtime descriptors for every normalized layout."""

    plan: Any
    rows: tuple[ResolvedRuntimeLayout, ...]

    def __post_init__(self) -> None:
        from pops.mesh import LayoutPlan

        if type(self.plan) is not LayoutPlan:
            raise TypeError("ResolvedRuntimeLayouts.plan must be an exact LayoutPlan")
        rows = tuple(self.rows)
        if any(type(row) is not ResolvedRuntimeLayout for row in rows):
            raise TypeError("ResolvedRuntimeLayouts rows must be exact values")
        expected = tuple(row.handle for row in self.plan.layouts)
        actual = tuple(row.handle for row in rows)
        if actual != expected:
            raise ValueError("runtime layout providers must match normalized layout order exactly")
        for row in rows:
            _authenticate_runtime_descriptor(self.plan, row.handle, row.descriptor)
        object.__setattr__(self, "rows", rows)

    def descriptor(self, handle: Any) -> Any:
        matches = [row.descriptor for row in self.rows if row.handle == handle]
        if len(matches) != 1:
            identity = getattr(handle, "qualified_id", repr(handle))
            raise KeyError("unknown runtime layout provider %s" % identity)
        return matches[0]

    def single(self) -> Any:
        if len(self.rows) != 1:
            raise ValueError("operation requires exactly one materialized runtime layout")
        return self.rows[0].descriptor

    def to_data(self) -> dict[str, Any]:
        return {
            "layout_plan": self.plan.qualified_id,
            "providers": [
                {
                    "layout": row.handle.canonical_identity(),
                    "normalized": self.plan.normalized(row.handle).to_data(),
                }
                for row in self.rows
            ],
        }


@dataclass(frozen=True, slots=True)
class ResolvedLayoutAuthority:
    """Canonical plan plus an orchestration-only provider, never hidden inside plan identity."""

    plan: Any
    runtime_layouts: ResolvedRuntimeLayouts | None = None

    def __post_init__(self) -> None:
        from pops.mesh import LayoutPlan

        if type(self.plan) is not LayoutPlan:
            raise TypeError("ResolvedLayoutAuthority.plan must be an exact LayoutPlan")
        if self.runtime_layouts is not None:
            if type(self.runtime_layouts) is not ResolvedRuntimeLayouts:
                raise TypeError("runtime_layouts must be an exact ResolvedRuntimeLayouts")
            if self.runtime_layouts.plan != self.plan:
                raise ValueError("runtime layouts authenticate a different LayoutPlan")

    def require_runtime(self) -> Any:
        if self.runtime_layouts is None:
            _refuse_runtime(
                self.plan,
                gate="layout_runtime_provider_unavailable",
                message=(
                    "pops.resolve received a canonical LayoutPlan without a separately "
                    "authenticated runtime provider; refusing before artifact creation"
                ),
            )
        return self.runtime_layouts


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

    def canonical(reference: Any) -> Any:
        if not isinstance(reference, Handle) or reference.kind not in ("state", "field", "block"):
            return None
        return problem.resolve(reference)

    def assigned(reference: Any) -> Any:
        value = canonical(reference)
        return None if value is None else plan.layout_for(value)

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
        target_subject = canonical(getattr(value, "state_ref", None))
        for source in source_layouts - {target}:
            source_subjects = {
                canonical(reference)
                for candidate in getattr(value, "inputs", ())
                for reference in (
                    getattr(candidate, "state_ref", None), getattr(candidate, "block", None))
                if assigned(reference) == source and canonical(reference) is not None
            }
            matches = [mapping for mapping in plan.mappings
                       if mapping.requirement.source_layout == source
                       and mapping.requirement.target_layout == target
                       and mapping.requirement.source_port.subject in source_subjects
                       and (target_subject is None
                            or mapping.requirement.target_port.subject == target_subject)]
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
        runtime_layouts = _select_runtime_providers(selected, providers)
        return ResolvedLayoutAuthority(selected, runtime_layouts)

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
    return ResolvedLayoutAuthority(
        plan, ResolvedRuntimeLayouts(
            plan, (ResolvedRuntimeLayout(plan.layouts[0].handle, runtime_descriptor),)))


def _select_runtime_providers(plan: Any, providers: Any) -> Any:
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
    missing = [handle.qualified_id for handle in declared if handle not in providers]
    if missing:
        raise ValueError("layout_providers is missing declared layouts: %s" % sorted(missing))
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
    return ResolvedRuntimeLayouts(
        plan,
        tuple(ResolvedRuntimeLayout(row.handle, detached[row.handle]) for row in plan.layouts),
    )


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


def _authenticate_runtime_descriptor(plan: Any, handle: Any, descriptor: Any) -> None:
    row = plan.normalized(handle)
    candidate = _normalized_runtime_descriptor(handle, descriptor)
    if candidate.to_data() != row.to_data():
        raise ValueError("runtime layout provider does not authenticate LayoutPlan snapshot")


def validate_layout_mapping_components(plan: Any, components: Any) -> None:
    """Authenticate mapping-provider component claims against exact resolve inputs."""
    from pops.external import ExternalComponent

    values = tuple(components)
    component_rows = {}
    for component in values:
        if type(component) is not ExternalComponent:
            continue
        component_id = component.component_manifest.component_id
        if component_id in component_rows:
            raise ValueError("resolve components contain duplicate component_id %r" % component_id)
        component_rows[component_id] = component.to_data()
    for mapping in plan.mappings:
        identity = mapping.provider_identity
        if identity.get("provider_type") != "native_transfer_component":
            _refuse_runtime(
                plan,
                gate="layout_mapping_provider_not_executable",
                message=("mapping provider %s has no authenticated native Transfer component"
                         % mapping.provider_id),
            )
        component_id = identity.get("component_id")
        if component_rows.get(component_id) != identity.get("component"):
            _refuse_runtime(
                plan,
                gate="layout_mapping_component_unavailable",
                message=("mapping provider %s is not backed by the exact component passed to "
                         "pops.resolve(..., components=...)" % mapping.provider_id),
            )


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
    "LayoutCapabilityError", "ResolvedLayoutAuthority", "ResolvedRuntimeLayout",
    "ResolvedRuntimeLayouts", "layout_lowering_coverage",
    "materialized_layout_subjects", "resolve_layout", "validate_layout",
    "validate_layout_mapping_components", "validate_program_layout_reads",
]
