from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pops


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"


def _example():
    spec = importlib.util.spec_from_file_location("pops_public_amr_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolved_target():
    from pops.amr import AMRResolutionContext
    from pops.mesh import normalize_layout_plan

    target = _example().build_final_case()
    case = pops.validate(target.authoring.case)
    layout = target.layout.resolve_for_case(case.resolve)
    subjects = case._materialized_layout_subjects()
    layout_plan = normalize_layout_plan(
        layout,
        owner=case.owner_path.canonical(),
        states=subjects["states"],
        fields=subjects["fields"],
        blocks=subjects["blocks"],
        handle_resolver=lambda value: value if value.is_resolved else case.resolve(value),
    )
    numerics = tuple(
        case._resolved_numerics_for(name) for name in sorted(case._numerics_assignments)
    )
    context = AMRResolutionContext(
        owner=case.owner_path.canonical(),
        layout_plan=layout_plan,
        numerics=numerics,
        initials=case.initials,
        program=case._time,
        resolve=lambda value: value if value.is_resolved else case.resolve(value),
    )
    return target, layout, layout_plan, layout.resolve_amr_authorities(context)


def test_final_amr_authorities_derive_discrete_context_and_nesting():
    from pops.mesh.amr import GradientAbove, GradientBelow

    target, layout, layout_plan, authorities = _resolved_target()
    assert layout.capabilities().get("transition_ratios") == [2, 2]
    assert authorities.hierarchy.plan.level_count == 3
    assert [row.ratio for row in authorities.hierarchy.plan.transitions] == [
        (2, 2),
        (2, 2),
    ]
    assert all(row.buffer == (2, 2) for row in authorities.hierarchy.plan.transitions)
    assert authorities.transfer.layout_plan_id == layout_plan.qualified_id

    graph = authorities.tagging.graph.graph
    assert type(graph.refine) is GradientAbove
    assert type(graph.coarsen) is GradientBelow
    assert graph.refine.indicator == target.authoring.case.resolve(
        target.authoring.tracer_state
    )
    assert graph.refine.context == graph.coarsen.context
    assert graph.refine.context.layout == layout_plan.layout_for(graph.refine.indicator)
    assert graph.refine.context.discretization.kind == "discretization"
    assert graph.refine.context.stencil.kind == "stencil"
    assert authorities.initial_conditions.layout_plan_id == layout_plan.qualified_id
    assert authorities.bootstrap.tagging == authorities.tagging.graph


def test_layout_refuses_unrepresentable_transition_ratios_without_substitution():
    from pops.amr import AMRHierarchy
    from pops.layouts import AMR

    target = _example().build_final_case()
    authored = target.layout
    heterogeneous = AMR(
        grid=authored.grid,
        hierarchy=AMRHierarchy(max_levels=3, ratios=(2, 4)),
        tagging=authored.tagging,
        regrid=authored.regrid,
        transfer=authored.transfer,
        execution=authored.execution,
    )
    status = heterogeneous.available()
    assert not status.ok
    assert "heterogeneous transition ratios" in status.reason


def test_symbolic_gradient_indicator_cannot_escape_discrete_resolution():
    from pops.ir import SymbolicTruthValueError, ValueExpr
    from pops.math import grad, norm

    core = _example().build_authoring()
    indicator = norm(grad(ValueExpr(core.tracer_state)))
    try:
        bool(indicator > core.case.value(core.refine_threshold))
    except SymbolicTruthValueError:
        pass
    else:
        raise AssertionError("symbolic AMR comparison became a Python bool")
    try:
        indicator.to_cpp()
    except TypeError as exc:
        assert "discrete consumer context" in str(exc)
    else:
        raise AssertionError("continuous-looking indicator bypassed discrete resolution")
