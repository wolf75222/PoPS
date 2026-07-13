"""Resolution of the final object-level AMR authoring surface.

This module is the only join between continuous-looking tagging expressions, resolved numerical
stencils, layout ownership, hierarchy providers, transfers and Case initial conditions.  Every
extension is invoked through a narrow protocol; there is no class-name or string-selector switch.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pops.identity import make_identity
from pops.identity.semantic import semantic_value
from pops.model import Handle, OwnerPath, ParamHandle

from .authoring import (
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    ResolvedAMRAuthorities,
)


def _canonical_owner(value: Any, *, where: str) -> OwnerPath:
    owner = OwnerPath.coerce(value)
    if not owner.is_canonical:
        raise TypeError("%s must be a canonical OwnerPath" % where)
    return owner


def _dimension(layout_plan: Any) -> int:
    adaptive = tuple(row for row in layout_plan.layouts if row.adaptive)
    if len(adaptive) != 1:
        raise ValueError(
            "object-level AMR resolution requires exactly one adaptive layout authority"
        )
    dimension = adaptive[0].capabilities.get("dim")
    if isinstance(dimension, bool) or dimension not in (1, 2, 3):
        raise ValueError("adaptive layout must authenticate dimension 1, 2, or 3")
    return dimension


def _protocol(value: Any, name: str, *, where: str) -> Callable[..., Any]:
    method = getattr(value, name, None)
    if not callable(method):
        raise TypeError("%s must implement %s(...)" % (where, name))
    return method


def _handle_token(domain: str, payload: Any) -> str:
    return make_identity(domain, semantic_value(payload, where=domain)).token


@dataclass(frozen=True, slots=True)
class ResolvedTaggingAuthority:
    """Resolved tag graph plus the explicit spatial dilation requested by authoring."""

    graph: Any
    buffer_cells: int

    def __post_init__(self) -> None:
        from pops.mesh.amr import ResolvedTaggingGraph

        if type(self.graph) is not ResolvedTaggingGraph:
            raise TypeError("ResolvedTaggingAuthority.graph must be a ResolvedTaggingGraph")
        if isinstance(self.buffer_cells, bool) or not isinstance(self.buffer_cells, int) \
                or self.buffer_cells < 0:
            raise ValueError("ResolvedTaggingAuthority.buffer_cells must be an integer >= 0")

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "authority_type": "resolved_amr_tagging",
            "graph": self.graph.canonical_identity(),
            "buffer_cells": self.buffer_cells,
        }

    inspect = canonical_identity


@dataclass(frozen=True, slots=True)
class AMRTaggingResolutionContext:
    """Small context offered to semantic AMR indicator implementations."""

    owner: OwnerPath
    layout_plan: Any
    numerics: tuple[Any, ...]
    resolve: Callable[[Handle], Handle]

    def __post_init__(self) -> None:
        from pops.mesh import LayoutPlan

        object.__setattr__(self, "owner", _canonical_owner(self.owner, where="AMR tagging owner"))
        if type(self.layout_plan) is not LayoutPlan:
            raise TypeError("AMR tagging requires an exact LayoutPlan")
        rows = tuple(self.numerics)
        if not rows:
            raise ValueError("AMR tagging requires at least one resolved numerical plan")
        if any(not hasattr(row, "rates") or not hasattr(row, "identity") for row in rows):
            raise TypeError("AMR tagging numerical plans do not implement the resolved protocol")
        if not callable(self.resolve):
            raise TypeError("AMR tagging resolve must be callable")
        object.__setattr__(self, "numerics", rows)

    def _discrete_context(self, state: Handle) -> Any:
        from pops.mesh.amr import DiscreteIndicatorContext

        if not isinstance(state, Handle) or state.kind != "state" or not state.is_resolved:
            raise TypeError(
                "AMR discrete indicators require an owner-qualified block-state Handle"
            )
        matches = []
        for plan in self.numerics:
            for rate in plan.rates:
                subject = rate.method.variables.options.get("state")
                if isinstance(subject, Handle) and subject.qualified_id == state.qualified_id:
                    matches.append((plan, rate.method))
        if not matches:
            raise ValueError(
                "AMR indicator state %s has no resolved spatial discretization"
                % state.qualified_id
            )
        plan_ids = {plan.identity.token for plan, _ in matches}
        methods = {_handle_token("amr-indicator-method", method.to_data())
                   for _, method in matches}
        if len(plan_ids) != 1 or len(methods) != 1:
            raise ValueError(
                "AMR indicator state %s has ambiguous discrete methods; select one exact "
                "state-to-discretization authority" % state.qualified_id
            )
        plan, method = matches[0]
        layout = self.layout_plan.layout_for(state)
        discretization = Handle(
            "discretization_%s" % plan.identity.token,
            kind="discretization",
            owner=self.owner,
        )
        stencil_payload = {
            "method": method.to_data(),
            "state": state.canonical_identity(),
            "layout": layout.canonical_identity(),
        }
        stencil = Handle(
            "stencil_%s" % _handle_token("amr-indicator-stencil", stencil_payload),
            kind="stencil",
            owner=self.owner,
        )
        return DiscreteIndicatorContext(layout, discretization, stencil)

    def resolve_gradient_magnitude(
        self,
        *,
        field: Any,
        scale: Any,
        action: str,
        comparison: str,
        threshold: ParamHandle,
    ) -> Any:
        """Bind ``norm(grad(state))`` to the exact selected FV layout and stencil."""
        from pops.mesh.amr import GradientAbove, GradientBelow

        if isinstance(scale, bool) or scale != 1:
            raise NotImplementedError(
                "the installed discrete-gradient provider requires unit gradient scale; "
                "a scaled indicator needs an explicit provider that preserves that scale"
            )
        references = getattr(field, "declaration_references", None)
        refs = references() if callable(references) else ()
        state = getattr(field, "handle", None)
        if len(refs) != 1 or state is not refs[0]:
            raise TypeError(
                "norm(grad(...)) AMR tagging requires exactly ValueExpr(block[state]); "
                "compound indicators need their own resolve_for_amr_tagging protocol"
            )
        context = self._discrete_context(state)
        if action == "refine" and comparison == "gt":
            return GradientAbove(state, threshold, context)
        if action == "coarsen" and comparison == "lt":
            return GradientBelow(state, threshold, context)
        expected = "strict >" if action == "refine" else "strict <"
        raise ValueError("AMR %s gradient rule requires %s threshold" % (action, expected))


def _threshold(value: Any) -> ParamHandle:
    references = getattr(value, "declaration_references", None)
    refs = references() if callable(references) else ()
    handle = getattr(value, "handle", None)
    if len(refs) != 1 or handle is not refs[0] or not isinstance(handle, ParamHandle):
        raise TypeError(
            "AMR tagging thresholds must be explicit ValueExpr(RuntimeParam Handle) values"
        )
    if not handle.is_resolved or handle.param_kind != "runtime":
        raise TypeError("AMR tagging thresholds must be canonical runtime ParamHandles")
    return handle


def _resolve_predicate(predicate: Any, *, action: str, context: Any) -> Any:
    comparison = getattr(predicate, "comparison", None)
    left = getattr(predicate, "a", None)
    right = getattr(predicate, "b", None)
    if comparison not in {"lt", "gt"}:
        raise ValueError("AMR tagging supports only strict < or > comparisons")
    left_protocol = getattr(left, "resolve_for_amr_tagging", None)
    right_protocol = getattr(right, "resolve_for_amr_tagging", None)
    if callable(left_protocol) == callable(right_protocol):
        raise TypeError(
            "AMR comparison must contain exactly one indicator implementing "
            "resolve_for_amr_tagging(...)"
        )
    if callable(right_protocol):
        comparison = {"lt": "gt", "gt": "lt"}[comparison]
        indicator, threshold_expr = right, left
    else:
        indicator, threshold_expr = left, right
    return indicator.resolve_for_amr_tagging(
        context,
        action=action,
        comparison=comparison,
        threshold=_threshold(threshold_expr),
    )


def _union(values: list[Any]) -> Any:
    from pops.mesh.amr import AnyOf

    if not values:
        return None
    return values[0] if len(values) == 1 else AnyOf(*values)


def resolve_tagging(
    authoring: AMRTagging,
    context: AMRTaggingResolutionContext,
) -> ResolvedTaggingAuthority:
    """Resolve an object-level priority list into one authenticated inert tag graph."""
    from pops.mesh.amr import TaggingGraph

    if type(authoring) is not AMRTagging:
        raise TypeError("resolve_tagging requires an exact AMRTagging value")
    if type(context) is not AMRTaggingResolutionContext:
        raise TypeError("resolve_tagging requires an AMRTaggingResolutionContext")
    resolved = authoring.resolve_references(context.resolve)
    roots: dict[str, list[Any]] = {"refine": [], "coarsen": []}
    buffer_cells = None
    for rule in resolved.rules:
        action = getattr(rule, "action", None)
        if action in roots:
            roots[action].append(
                _resolve_predicate(rule.predicate, action=action, context=context)
            )
        elif action == "buffer":
            buffer_cells = rule.cells
        else:
            raise TypeError(
                "AMR tagging rule must expose action refine, coarsen, or buffer"
            )
    if buffer_cells is None:
        raise ValueError("AMR tagging resolution lost its explicit Buffer authority")
    graph = TaggingGraph(
        refine=_union(roots["refine"]),
        coarsen=_union(roots["coarsen"]),
        hysteresis=resolved.hysteresis,
        conflict_policy=resolved.conflict_policy,
    ).resolve()
    return ResolvedTaggingAuthority(graph, buffer_cells)


@dataclass(frozen=True, slots=True)
class AMRResolutionContext:
    """All Case-owned authorities needed by an adaptive-layout descriptor."""

    owner: OwnerPath
    layout_plan: Any
    numerics: tuple[Any, ...]
    initials: Any
    program: Any
    resolve: Callable[[Handle], Handle]

    def __post_init__(self) -> None:
        from pops.mesh import LayoutPlan
        from pops.time import Program

        object.__setattr__(self, "owner", _canonical_owner(self.owner, where="AMR owner"))
        if type(self.layout_plan) is not LayoutPlan:
            raise TypeError("AMR resolution requires an exact LayoutPlan")
        rows = tuple(self.numerics)
        if not rows:
            raise ValueError("AMR resolution requires resolved numerics")
        object.__setattr__(self, "numerics", rows)
        if not callable(getattr(self.initials, "resolve_amr", None)):
            raise TypeError("AMR initials authority must implement resolve_amr(...)")
        if type(self.program) is not Program:
            raise TypeError("adaptive regridding requires one explicit Program")
        if not callable(self.resolve):
            raise TypeError("AMR resolution resolver must be callable")


def _combined_requirement(
    *,
    kind: str,
    sources: tuple[Any, ...],
    owner: OwnerPath,
    dimension: int,
) -> Any:
    from pops.mesh.amr import NestingRequirementSource

    expected = "amr_%s_requirement" % kind
    for source in sources:
        if type(source) is not NestingRequirementSource or source.provider.kind != expected:
            raise TypeError("AMR %s protocol returned an invalid nesting source" % kind)
        if len(source.minimum_buffer) != dimension:
            raise ValueError("AMR %s nesting source has a different dimension" % kind)
    minimum_buffer = tuple(
        max((row.minimum_buffer[axis] for row in sources), default=0)
        for axis in range(dimension)
    )
    minimum_lookahead = max((row.minimum_lookahead for row in sources), default=0)
    payload = {
        "kind": kind,
        "sources": [row.canonical_identity() for row in sources],
        "minimum_buffer": list(minimum_buffer),
        "minimum_lookahead": minimum_lookahead,
    }
    provider = Handle(
        "%s_%s" % (kind, _handle_token("amr-%s-requirement" % kind, payload)),
        kind=expected,
        owner=owner,
    )
    return NestingRequirementSource(
        provider,
        minimum_buffer,
        minimum_lookahead,
    )


def _hierarchy(
    authoring: AMRHierarchy,
    regrid: AMRRegrid,
    transfer: Any,
    tagging: ResolvedTaggingAuthority,
    context: AMRResolutionContext,
) -> Any:
    from pops.mesh.amr import (
        CanonicalOptions,
        ClusteringPolicy,
        DerivedNestingRequirements,
        HierarchyPlan,
        HierarchyProviderCapabilities,
        HierarchyResolutionContext,
        LevelTransition,
        LoadBalancePolicy,
        NestingRequirementSource,
        PatchGenerationPolicy,
        RegridSchedule,
        resolve_hierarchy,
    )
    from pops.time import EventHandle

    dimension = _dimension(context.layout_plan)
    stencil_sources = tuple(
        _protocol(row, "amr_stencil_requirement", where="resolved numerics")(
            owner=context.owner, dimension=dimension
        )
        for row in context.numerics
    )
    reflux_sources = tuple(
        _protocol(row, "amr_reflux_requirement", where="resolved numerics")(
            owner=context.owner, dimension=dimension
        )
        for row in context.numerics
    )
    boundary_sources = tuple(
        _protocol(boundary, "amr_boundary_requirement", where="resolved boundary")(
            owner=context.owner, dimension=dimension
        )
        for row in context.numerics
        for boundary in row.boundaries
    )
    nesting = DerivedNestingRequirements(
        stencil=_combined_requirement(
            kind="stencil",
            sources=stencil_sources,
            owner=context.owner,
            dimension=dimension,
        ),
        transfer=transfer.nesting_requirement,
        reflux=_combined_requirement(
            kind="reflux",
            sources=reflux_sources,
            owner=context.owner,
            dimension=dimension,
        ),
        boundary=(
            _combined_requirement(
                kind="boundary",
                sources=boundary_sources,
                owner=context.owner,
                dimension=dimension,
            )
            if boundary_sources
            else NestingRequirementSource(
                Handle(
                    "boundary_none_%s" % context.layout_plan.canonical_id,
                    kind="amr_boundary_requirement",
                    owner=context.owner,
                ),
                (0,) * dimension,
                0,
            )
        ),
    )
    minimum_buffer = tuple(
        max(tagging.buffer_cells, value) for value in nesting.minimum_buffer
    )
    transitions = tuple(
        LevelTransition(
            coarse_level=index,
            fine_level=index + 1,
            ratio=(ratio,) * dimension,
            buffer=minimum_buffer,
            lookahead=nesting.minimum_lookahead,
        )
        for index, ratio in enumerate(authoring.ratios)
    )
    def provider(local_id: str, kind: str) -> Handle:
        return Handle(local_id, kind=kind, owner=context.owner)

    due_id = _handle_token(
        "amr-regrid-event",
        {
            "layout_plan": context.layout_plan.qualified_id,
            "schedule": regrid.schedule.to_data(),
        },
    )
    plan = HierarchyPlan(
        transitions=transitions,
        nesting=nesting,
        clustering=ClusteringPolicy(
            provider("berger_rigoutsos", "amr_clustering_provider"),
            CanonicalOptions({"native_route": "berger_rigoutsos"}),
        ),
        patch_generation=PatchGenerationPolicy(
            provider("box_array", "amr_patch_generation_provider"),
            CanonicalOptions({"native_route": "box_array"}),
        ),
        load_balance=LoadBalancePolicy(
            provider("round_robin", "amr_load_balance_provider"),
            CanonicalOptions({"native_route": "round_robin"}),
        ),
        regrid=RegridSchedule(
            regrid.schedule,
            EventHandle(regrid.schedule.clock.owner, "amr.regrid.due.%s" % due_id),
        ),
    )
    capabilities = HierarchyProviderCapabilities(
        provider("shared_n_level", "amr_hierarchy_provider"),
        supported_dimensions=(2,),
        supports_anisotropic_ratio=False,
        max_materialized_level_count=2_147_483_647,
        supports_transactional_regrid=True,
        supports_lifecycle_events=True,
        options=CanonicalOptions({"native_route": "shared_n_level"}),
    )
    return resolve_hierarchy(
        plan,
        capabilities,
        HierarchyResolutionContext(context.program.clock),
    )


def resolve_amr_authorities(
    *,
    hierarchy: AMRHierarchy,
    tagging: AMRTagging,
    regrid: AMRRegrid,
    transfer: Any,
    execution: AMRExecution,
    context: AMRResolutionContext,
) -> ResolvedAMRAuthorities:
    """Resolve every adaptive-layout concern exactly once from its owning declaration."""
    from pops.mesh.amr import AMRTransfer

    if type(hierarchy) is not AMRHierarchy or type(tagging) is not AMRTagging \
            or type(regrid) is not AMRRegrid or type(execution) is not AMRExecution:
        raise TypeError("AMR layout carries an unsupported object-level authority")
    if type(transfer) is not AMRTransfer:
        raise TypeError("AMR layout transfer must be an exact AMRTransfer")
    if type(context) is not AMRResolutionContext:
        raise TypeError("AMR resolution requires an AMRResolutionContext")
    resolved_transfer = transfer.resolve_references(context.resolve).resolve(context.layout_plan)
    tagging_context = AMRTaggingResolutionContext(
        context.owner,
        context.layout_plan,
        context.numerics,
        context.resolve,
    )
    resolved_tagging = resolve_tagging(tagging, tagging_context)
    resolved_hierarchy = _hierarchy(
        hierarchy,
        regrid,
        resolved_transfer,
        resolved_tagging,
        context,
    )
    initial = context.initials.resolve_amr(
        layout_plan=context.layout_plan,
        transfers=resolved_transfer,
        hierarchy=resolved_hierarchy,
        tagging=resolved_tagging.graph,
        constraints=(),
    )
    return ResolvedAMRAuthorities(
        hierarchy=resolved_hierarchy,
        transfer=resolved_transfer,
        tagging=resolved_tagging,
        initial_conditions=initial.initial_condition_plan,
        bootstrap=initial.bootstrap_plan,
        execution=execution,
    )


__all__ = [
    "AMRResolutionContext",
    "AMRTaggingResolutionContext",
    "ResolvedTaggingAuthority",
    "resolve_amr_authorities",
    "resolve_tagging",
]
