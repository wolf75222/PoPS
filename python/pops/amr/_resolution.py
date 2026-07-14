"""Internal resolution of the final object-level AMR authoring surface.

This module is the only join between continuous-looking tagging expressions, resolved numerical
stencils, layout ownership, hierarchy providers, transfers and Case initial conditions.  Every
extension is invoked through a narrow protocol; there is no class-name or string-selector switch.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pops.identity import make_identity
from pops.identity.semantic import semantic_value
from pops.model import Handle, OwnerPath, ParamHandle

from .authoring import (
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
)


@dataclass(frozen=True, slots=True)
class ResolvedAMRAuthorities:
    """Internal, fully resolved adaptive authorities consumed by code generation."""

    hierarchy: Any
    transfer: Any
    tagging: Any
    initial_conditions: Any
    bootstrap: Any
    execution: Any
    providers: Any

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "authority_type": "resolved_amr_authorities",
            "hierarchy": self.hierarchy.canonical_identity(),
            "transfer": self.transfer.canonical_identity(),
            "tagging": self.tagging.canonical_identity(),
            "initial_conditions": self.initial_conditions.canonical_identity(),
            "bootstrap": self.bootstrap.canonical_identity(),
            "execution": self.execution.to_data(),
            "providers": dict(self.providers),
        }


@runtime_checkable
class AMRLayoutResolver(Protocol):
    """Small extension interface implemented by an adaptive layout authority."""

    def resolve_amr_authorities(
        self, context: "AMRResolutionContext",
    ) -> ResolvedAMRAuthorities: ...


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
        from pops.mesh._amr import ResolvedTaggingGraph

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
        from pops.mesh._amr import DiscreteIndicatorContext

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
        lower_stencil = getattr(method, "amr_indicator_stencil", None)
        if not callable(lower_stencil):
            raise NotImplementedError(
                "AMR indicator discretization does not provide amr_indicator_stencil(...)"
            )
        lowering = lower_stencil(dimension=_dimension(self.layout_plan))
        from pops.numerics.indicator_stencils import DiscreteGradientStencil

        if type(lowering) is not DiscreteGradientStencil:
            raise TypeError(
                "amr_indicator_stencil(...) must return a DiscreteGradientStencil")
        discretization = Handle(
            "discretization_%s" % plan.identity.token,
            kind="discretization",
            owner=self.owner,
        )
        stencil_payload = {
            "method": method.to_data(),
            "state": state.canonical_identity(),
            "layout": layout.canonical_identity(),
            "lowering": lowering.to_data(),
        }
        stencil = Handle(
            "stencil_%s" % _handle_token("amr-indicator-stencil", stencil_payload),
            kind="stencil",
            owner=self.owner,
        )
        return DiscreteIndicatorContext(layout, discretization, stencil, lowering)

    def resolve_value_indicator(
        self,
        *,
        handle: Handle,
        action: str,
        comparison: str,
        threshold: ParamHandle,
    ) -> Any:
        """Bind a direct block-state value to strict Above/Below tagging leaves."""
        from pops.mesh._amr import Above, Below

        if not isinstance(handle, Handle) or handle.kind != "state" or not handle.is_resolved:
            raise TypeError(
                "AMR value indicators require an owner-qualified block-state Handle")
        if handle.owner_path.nodes[0] != self.owner.nodes[0]:
            raise ValueError(
                "AMR value indicator %s belongs to a different Case owner"
                % handle.qualified_id)
        self.layout_plan.layout_for(handle)
        components = tuple(getattr(getattr(handle, "space", None), "components", ()))
        if len(components) != 1:
            raise ValueError(
                "AMR direct state indicators require a scalar state; state %s has components %s. "
                "Select a typed component indicator explicitly."
                % (handle.qualified_id, components))
        if action == "refine" and comparison == "gt":
            return Above(handle, threshold)
        if action == "coarsen" and comparison == "lt":
            return Below(handle, threshold)
        expected = "strict >" if action == "refine" else "strict <"
        raise ValueError("AMR %s value rule requires %s threshold" % (action, expected))

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
        from pops.mesh._amr import GradientAbove, GradientBelow

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
        components = tuple(getattr(getattr(state, "space", None), "components", ()))
        if len(components) != 1:
            raise ValueError(
                "AMR gradient indicators require a scalar state; state %s has components %s. "
                "Select a typed component indicator explicitly."
                % (state.qualified_id, components))
        if action == "refine" and comparison == "gt":
            return GradientAbove(state, threshold, context)
        if action == "coarsen" and comparison == "lt":
            return GradientBelow(state, threshold, context)
        expected = "strict >" if action == "refine" else "strict <"
        raise ValueError("AMR %s gradient rule requires %s threshold" % (action, expected))

    def resolve_comparison(self, predicate: Any, *, action: str) -> Any:
        """Lower one comparison leaf; Boolean composition remains owned by Expr node protocols."""
        comparison = getattr(predicate, "comparison", None)
        left = getattr(predicate, "a", None)
        right = getattr(predicate, "b", None)
        if comparison not in {"lt", "gt"}:
            raise ValueError("AMR tagging supports only strict < or > comparison leaves")
        left_threshold = _threshold_candidate(left)
        right_threshold = _threshold_candidate(right)
        if (left_threshold is None) == (right_threshold is None):
            raise TypeError(
                "AMR comparison must contain exactly one runtime parameter threshold"
            )
        if left_threshold is not None:
            comparison = {"lt": "gt", "gt": "lt"}[comparison]
            indicator, threshold = right, left_threshold
        else:
            indicator, threshold = left, right_threshold
        protocol = getattr(indicator, "resolve_for_amr_tagging", None)
        if not callable(protocol):
            raise TypeError("AMR indicator must implement resolve_for_amr_tagging(...)")
        return protocol(
            self, action=action, comparison=comparison, threshold=threshold)


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


def _threshold_candidate(value: Any) -> ParamHandle | None:
    references = getattr(value, "declaration_references", None)
    refs = references() if callable(references) else ()
    handle = getattr(value, "handle", None)
    if len(refs) == 1 and handle is refs[0] and isinstance(handle, ParamHandle):
        return _threshold(value)
    return None


def _resolve_predicate(predicate: Any, *, action: str, context: Any) -> Any:
    protocol = getattr(predicate, "resolve_for_amr_predicate", None)
    if not callable(protocol):
        raise TypeError("AMR predicate must implement resolve_for_amr_predicate(...)")
    return protocol(context, action=action)


def _union(values: list[Any]) -> Any:
    from pops.mesh._amr import AnyOf

    if not values:
        return None
    return values[0] if len(values) == 1 else AnyOf(*values)


def resolve_tagging(
    authoring: Any,
    context: AMRTaggingResolutionContext,
) -> ResolvedTaggingAuthority:
    """Resolve an object-level priority list into one authenticated inert tag graph."""
    from pops.mesh._amr import TaggingGraph

    for method in ("resolve_references", "inspect"):
        if not callable(getattr(authoring, method, None)):
            raise TypeError("resolve_tagging authoring must implement %s()" % method)
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
    components: tuple[Any, ...] = ()

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
        object.__setattr__(self, "components", tuple(self.components))


def _combined_requirement(
    *,
    kind: str,
    sources: tuple[Any, ...],
    owner: OwnerPath,
    dimension: int,
) -> Any:
    from pops.mesh._amr import NestingRequirementSource

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
    clustering: Any,
) -> Any:
    from pops.mesh._amr import (
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
            provider(
                "clustering_%s" % clustering.runtime_binding_data()["provider_identity"],
                "amr_clustering_provider",
            ),
            CanonicalOptions({
                "provider": {
                    **clustering.runtime_binding_data(),
                    "layout_identity": context.layout_plan.qualified_id,
                }
            }),
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
    hierarchy: Any,
    tagging: Any,
    regrid: Any,
    transfer: Any,
    execution: Any,
    tagger: Any,
    clustering: Any,
    context: AMRResolutionContext,
) -> ResolvedAMRAuthorities:
    """Resolve every adaptive-layout concern exactly once from its owning declaration."""
    protocols = {
        "hierarchy": (hierarchy, ("to_data",)),
        "tagging": (tagging, ("resolve_references", "resolve", "inspect")),
        "regrid": (regrid, ("to_data",)),
        "transfer": (transfer, ("resolve_references", "resolve", "inspect")),
        "execution": (execution, ("to_data", "runtime_execution_data")),
    }
    for slot, (authority, methods) in protocols.items():
        for method in methods:
            if not callable(getattr(authority, method, None)):
                raise TypeError("AMR %s authority must implement %s()" % (slot, method))
    if type(context) is not AMRResolutionContext:
        raise TypeError("AMR resolution requires an AMRResolutionContext")
    providers = {"tagger": tagger, "clustering": clustering}
    for slot, value in providers.items():
        for method in (
                "inspect", "resolve_references", "require_component_inputs",
                "require_tagging_graph", "runtime_binding_data"):
            if slot == "clustering" and method == "require_tagging_graph":
                continue
            if not callable(getattr(value, method, None)):
                raise TypeError("AMR %s provider must implement %s()" % (slot, method))
    resolved_providers = {
        slot: value.resolve_references(context.resolve) for slot, value in providers.items()
    }
    for slot, value in resolved_providers.items():
        value.require_component_inputs(context.components)
    resolved_transfer = transfer.resolve_references(context.resolve).resolve(context.layout_plan)
    tagging_context = AMRTaggingResolutionContext(
        context.owner,
        context.layout_plan,
        context.numerics,
        context.resolve,
    )
    resolved_tagging = resolve_tagging(tagging, tagging_context)
    resolved_providers["tagger"].require_tagging_graph(resolved_tagging.graph)
    resolved_hierarchy = _hierarchy(
        hierarchy,
        regrid,
        resolved_transfer,
        resolved_tagging,
        context,
        resolved_providers["clustering"],
    )
    initial = context.initials.resolve_amr(
        layout_plan=context.layout_plan,
        transfers=resolved_transfer,
        hierarchy=resolved_hierarchy,
        tagging=resolved_tagging.graph,
        constraints=(),
    )
    provider_bindings = {
        slot: {
            **value.runtime_binding_data(),
            "layout_identity": context.layout_plan.qualified_id,
        }
        for slot in ("clustering", "tagger")
        for value in (resolved_providers[slot],)
    }
    provider_bindings["tagger"] = {
        **provider_bindings["tagger"],
        "clock_identity": context.program.clock.qualified_id,
        "tagging_graph_identity": resolved_tagging.graph.qualified_id,
    }
    identity_payload = {
        key: value for key, value in provider_bindings["tagger"].items()
        if key != "provider_identity"
    }
    provider_bindings["tagger"]["provider_identity"] = make_identity(
        "resolved-amr-tagger-provider",
        semantic_value(identity_payload, where="resolved AMR Tagger provider"),
    ).token
    return ResolvedAMRAuthorities(
        hierarchy=resolved_hierarchy,
        transfer=resolved_transfer,
        tagging=resolved_tagging,
        initial_conditions=initial.initial_condition_plan,
        bootstrap=initial.bootstrap_plan,
        execution=execution,
        providers=provider_bindings,
    )


__all__ = [
    "AMRLayoutResolver",
    "AMRResolutionContext",
    "AMRTaggingResolutionContext",
    "ResolvedAMRAuthorities",
    "ResolvedTaggingAuthority",
    "resolve_amr_authorities",
    "resolve_tagging",
]
