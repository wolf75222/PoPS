"""Field-complete identity proof for the resolved AMR tagging program.

The graph identity is not merely descriptive metadata: it authenticates the
data-only native program installed by ``flow_bootstrap_tagging``.  Each case
below changes one relevant authored field and proves that the change survives
both graph resolution and authenticated native lowering.

The layout, discretization, and stencil Handles are derived provenance
authorities, not numerical hot-loop operands.  They therefore change only the
bound-program identity.  The separate ``stencil_lowering`` value owns the
actual coefficients copied into the native program and is checked separately.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pops
from pops.mesh._amr import (
    Above,
    AllOf,
    AnyOf,
    Below,
    ConflictPolicy,
    DiscreteIndicatorContext,
    EqualityPolicy,
    GradientAbove,
    Hysteresis,
    MagnitudeAbove,
    Not,
    TaggingGraph,
)
from pops.model import Handle, OwnerPath
from pops.numerics.indicator_stencils import (
    FOURTH_ORDER_AXIS,
    SECOND_ORDER_AXIS,
    gradient_stencil,
)
from pops.params import RuntimeParam
from pops.runtime._runtime_mesh_lowering import flow_bootstrap_tagging


class _NativeTaggingProbe:
    def __init__(self) -> None:
        self.call = None

    def _set_bootstrap_tagging(self, *args) -> None:
        self.call = args


def _resolved_handles():
    left_model = pops.Model("tagging-identity-left")
    left_state = left_model.state("U", components=("n",))
    right_model = pops.Model("tagging-identity-right")
    right_state = right_model.state("U", components=("m",))
    case = pops.Case("tagging-identity")
    left = case.block("left", left_model, states=(left_state,))
    right = case.block("right", right_model, states=(right_state,))
    magnitude = case.param(RuntimeParam("magnitude", default=0.5))
    gradient = case.param(RuntimeParam("gradient", default=0.25))
    coarsen = case.param(RuntimeParam("coarsen", default=0.1))
    alternate = case.param(RuntimeParam("alternate", default=0.75))
    validated = pops.validate(case)
    return (
        validated.resolve(left[left_state]),
        validated.resolve(right[right_state]),
        validated.resolve(magnitude),
        validated.resolve(gradient),
        validated.resolve(coarsen),
        validated.resolve(alternate),
    )


def _context(
    *,
    layout: str = "adaptive",
    discretization: str = "finite_volume",
    stencil: str = "centered",
    fourth_order: bool = False,
) -> DiscreteIndicatorContext:
    owner = OwnerPath.case("tagging-identity")
    axis = FOURTH_ORDER_AXIS if fourth_order else SECOND_ORDER_AXIS
    return DiscreteIndicatorContext(
        layout=Handle(layout, kind="layout", owner=owner),
        discretization=Handle(
            discretization, kind="discretization", owner=owner),
        stencil=Handle(stencil, kind="stencil", owner=owner),
        lowering=gradient_stencil(axis, dimension=2),
    )


def _lower(graph: TaggingGraph, params):
    resolved = graph.resolve()
    probe = _NativeTaggingProbe()
    flow_bootstrap_tagging(
        probe,
        SimpleNamespace(tagging=resolved),
        params,
        clock_identity="case::tagging-clock",
    )
    assert probe.call is not None
    return resolved, probe.call


def _changed_positions(left, right) -> set[int]:
    assert len(left) == len(right) == 18
    return {index for index, (a, b) in enumerate(zip(left, right, strict=True)) if a != b}


def test_every_tagging_field_survives_resolution_and_native_lowering():
    (left, right, magnitude_threshold, gradient_threshold,
     coarsen_threshold, alternate_threshold) = _resolved_handles()
    context = _context()
    magnitude = MagnitudeAbove(left, magnitude_threshold)
    gradient = GradientAbove(left, gradient_threshold, context)
    negated_gradient = Not(gradient)
    refine = AnyOf(magnitude, negated_gradient)
    coarsen = Below(left, coarsen_threshold)
    graph = TaggingGraph(
        refine=refine,
        coarsen=coarsen,
        hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.REFINE_WINS,
    )
    params = {
        magnitude_threshold: 0.5,
        gradient_threshold: 0.25,
        coarsen_threshold: 0.1,
        alternate_threshold: 0.75,
    }
    resolved, lowered = _lower(graph, params)

    changed_magnitude_indicator = replace(magnitude, indicator=right)
    changed_magnitude_threshold = replace(
        magnitude, threshold=alternate_threshold)
    changed_context_layout = replace(
        gradient, context=_context(layout="uniform"))
    changed_context_discretization = replace(
        gradient, context=_context(discretization="discontinuous_galerkin"))
    changed_context_stencil = replace(
        gradient, context=_context(stencil="centered-alternate"))
    changed_context_lowering = replace(
        gradient, context=_context(stencil="fourth-order", fourth_order=True))

    cases = {
        # Threshold-leaf fields and concrete leaf semantics.
        "indicator": (
            replace(graph, refine=AnyOf(changed_magnitude_indicator, negated_gradient)),
            {1, 2, 3, 17},
        ),
        "threshold": (
            replace(graph, refine=AnyOf(changed_magnitude_threshold, negated_gradient)),
            {6, 17},
        ),
        "leaf_type": (
            replace(
                graph,
                refine=AnyOf(
                    Above(left, magnitude_threshold), negated_gradient)),
            {5, 9, 17},
        ),
        # Every discrete-indicator context field remains authenticated.  The
        # first three are derived provenance authorities and intentionally
        # alter only the bound-program identity (position 14).  They are not
        # misrepresented as numerical operands.  Only stencil_lowering owns
        # executable coefficients, projected in position 5.
        "context_layout": (
            replace(graph, refine=AnyOf(
                magnitude, Not(changed_context_layout))),
            {17},
        ),
        "context_discretization": (
            replace(graph, refine=AnyOf(
                magnitude, Not(changed_context_discretization))),
            {17},
        ),
        "context_stencil": (
            replace(graph, refine=AnyOf(
                magnitude, Not(changed_context_stencil))),
            {17},
        ),
        "context_lowering": (
            replace(graph, refine=AnyOf(
                magnitude, Not(changed_context_lowering))),
            {8, 17},
        ),
        # Boolean-node kind, child order, and Not.child all reach bytecode.
        "children_order": (
            replace(graph, refine=AnyOf(negated_gradient, magnitude)),
            {5, 6, 7, 9, 17},
        ),
        "logical_node_type": (
            replace(graph, refine=AllOf(magnitude, negated_gradient)),
            {9, 17},
        ),
        "not_child": (
            replace(
                graph,
                refine=AnyOf(
                    magnitude, Not(Above(left, gradient_threshold)))),
            {5, 7, 8, 9, 17},
        ),
        # All four TaggingGraph fields and both Hysteresis fields are projected.
        "coarsen": (
            replace(graph, coarsen=None),
            {0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 17},
        ),
        "coarsen_threshold": (
            replace(graph, coarsen=Below(left, alternate_threshold)),
            {6, 17},
        ),
        "minimum_cycles": (
            replace(graph, hysteresis=Hysteresis(1, EqualityPolicy.HOLD)),
            {13, 17},
        ),
        "equality_policy": (
            replace(graph, hysteresis=Hysteresis(0, EqualityPolicy.REFINE)),
            {14, 17},
        ),
        "conflict_policy": (
            replace(graph, conflict_policy=ConflictPolicy.HOLD),
            {15, 17},
        ),
    }

    identities = {graph.canonical_id}
    resolved_identities = {resolved.canonical_id}
    program_identities = {lowered[17]}
    for name, (candidate, expected_positions) in cases.items():
        candidate_resolved, candidate_lowered = _lower(candidate, params)
        assert candidate.canonical_identity() != graph.canonical_identity(), name
        assert candidate.canonical_id not in identities, name
        assert candidate_resolved.canonical_identity() != resolved.canonical_identity(), name
        assert candidate_resolved.canonical_id not in resolved_identities, name
        assert candidate_lowered[17] not in program_identities, name
        assert _changed_positions(lowered, candidate_lowered) == expected_positions, name
        identities.add(candidate.canonical_id)
        resolved_identities.add(candidate_resolved.canonical_id)
        program_identities.add(candidate_lowered[17])
