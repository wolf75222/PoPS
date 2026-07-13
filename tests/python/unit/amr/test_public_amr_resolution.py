from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pops
import pytest


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"


def _example():
    spec = importlib.util.spec_from_file_location("pops_public_amr_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolved_target(*, hysteresis=None, conflict_policy=None):
    from pops.amr import AMRResolutionContext
    from pops.mesh import normalize_layout_plan

    target = _example().build_final_case()
    authored_layout = target.layout
    if hysteresis is not None or conflict_policy is not None:
        if hysteresis is None or conflict_policy is None:
            raise ValueError("custom tagging requires both hysteresis and conflict_policy")
        from pops.amr import AMRTagging

        authored_layout = type(authored_layout)(
            grid=authored_layout.grid,
            hierarchy=authored_layout.hierarchy,
            tagging=AMRTagging(
                rules=authored_layout.tagging.rules,
                hysteresis=hysteresis,
                conflict_policy=conflict_policy,
            ),
            regrid=authored_layout.regrid,
            transfer=authored_layout.transfer,
            execution=authored_layout.execution,
        )
    case = pops.validate(target.authoring.case)
    layout = authored_layout.resolve_for_case(case.resolve)
    subjects = case.layout_subjects()
    layout_plan = normalize_layout_plan(
        layout,
        owner=case.owner_path.canonical(),
        states=subjects.states,
        fields=subjects.fields,
        blocks=subjects.blocks,
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


def test_tagging_resolution_preserves_explicit_hysteresis_equality_and_conflict():
    from pops.amr import ConflictPolicy, EqualityPolicy, Hysteresis

    authored = Hysteresis(min_cycles=3, equality=EqualityPolicy.COARSEN)
    _, layout, _, authorities = _resolved_target(
        hysteresis=authored,
        conflict_policy=ConflictPolicy.ERROR,
    )
    graph = authorities.tagging.graph.graph
    assert layout.tagging.hysteresis == authored
    assert graph.hysteresis == authored
    assert graph.hysteresis.equality is EqualityPolicy.COARSEN
    assert graph.conflict_policy is ConflictPolicy.ERROR


def test_tagging_authority_requires_exact_explicit_policy_types():
    from pops.amr import AMRTagging, ConflictPolicy, EqualityPolicy, Hysteresis

    rules = _example().build_final_case().layout.tagging.rules
    with pytest.raises(TypeError, match="exact Hysteresis"):
        AMRTagging(
            rules=rules,
            hysteresis=object(),
            conflict_policy=ConflictPolicy.ERROR,
        )
    with pytest.raises(TypeError, match="exact ConflictPolicy"):
        AMRTagging(
            rules=rules,
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy="error",
        )


def test_public_resolve_derives_every_amr_authority_without_manual_injection():
    from pops.codegen import Production

    target = _example().build_final_case()
    resolved = pops.resolve(
        pops.validate(target.authoring.case),
        layout=target.layout,
        backend=Production(),
    )

    assert resolved.resolved_hierarchy.plan.level_count == 3
    assert resolved.bootstrap_plan.hierarchy_identity == resolved.resolved_hierarchy.identity
    assert resolved.bootstrap_plan.transfer_identity == resolved.amr_transfer.identity
    assert resolved.bootstrap_plan.initial_identity == resolved.initial_condition_plan.identity
    assert resolved.amr_execution.mode == "subcycled"


def test_runtime_tagging_compiles_refine_and_coarsen_to_data_only_vm():
    from pops.runtime._runtime_mesh_lowering import flow_bootstrap_tagging

    target, _, _, authorities = _resolved_target()
    params = _example().build_bind_params(target.authoring)

    class NativeProbe:
        call = None

        def _set_bootstrap_tagging(self, *args):
            self.call = args

    native = NativeProbe()
    flow_bootstrap_tagging(native, authorities.bootstrap, params)
    assert native.call is not None
    (blocks, variables, leaf_ops, thresholds, refine_ops, refine_args,
     coarsen_ops, coarsen_args, min_cycles, equality, conflict, provider) = native.call
    assert blocks == ["tracer", "tracer"]
    assert variables == ["U", "U"]
    assert leaf_ops == [4, 5]
    assert thresholds == [0.10, 0.04]
    assert (refine_ops, refine_args) == ([4], [0])
    assert (coarsen_ops, coarsen_args) == ([5], [1])
    assert (min_cycles, equality, conflict) == (0, "hold", "refine_wins")
    assert provider == authorities.tagging.graph.qualified_id


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
