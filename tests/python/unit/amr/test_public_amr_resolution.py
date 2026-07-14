from __future__ import annotations

import importlib.util
from fractions import Fraction
from pathlib import Path
import sys
from types import SimpleNamespace

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
    from pops.amr._resolution import AMRResolutionContext
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
    from pops.mesh._amr import GradientAbove, GradientBelow

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
    assert graph.refine.context.lowering.route == "linear_axis_stencil_l2_v1"
    assert graph.refine.context.lowering.dimension == 2
    assert [axis.offsets for axis in graph.refine.context.lowering.axes] == [
        (-1, 1), (-1, 1)]
    assert authorities.initial_conditions.layout_plan_id == layout_plan.qualified_id
    assert authorities.bootstrap.tagging == authorities.tagging.graph


def test_temporal_relations_are_exact_explicit_and_independent_from_spatial_ratios():
    from pops.amr import (
        AMRClockRelation,
        AMRExecution,
        AMRRemainderPolicy,
    )

    relation = AMRClockRelation(0, 1, 3)
    execution = AMRExecution.subcycled((relation,))
    assert relation.temporal_ratio == Fraction(3, 1)
    assert execution.to_data()["relations"] == [{
        "parent_level": 0,
        "child_level": 1,
        "temporal_ratio": {"numerator": 3, "denominator": 1},
        "remainder_policy": "integral_only",
    }]
    with pytest.raises(ValueError, match="EXPLICIT_FINAL_SUBSTEP"):
        AMRClockRelation(0, 1, Fraction(3, 2))
    remainder = AMRClockRelation(
        0, 1, Fraction(3, 2), AMRRemainderPolicy.EXPLICIT_FINAL_SUBSTEP)
    assert remainder.temporal_ratio == Fraction(3, 2)
    with pytest.raises(ValueError, match="synchronous"):
        AMRExecution("synchronous", (relation,))
    with pytest.raises(OverflowError, match="native exact-clock range"):
        AMRClockRelation(0, 1, Fraction(1 << 63, 1))


def test_runtime_authority_installs_exact_temporal_relation_without_spatial_inference():
    from pops import interfaces
    from pops._generated_component_interfaces import NATIVE_TAGGING_PROGRAM_ABI
    from pops.amr import AMRClockRelation, AMRExecution
    from pops._platform_contracts import (
        ExecutionContext,
        ExecutionResource,
        proven_serial_manifest,
    )
    from pops.runtime._runtime_authorities import install_runtime_authorities

    tagging_abi = NATIVE_TAGGING_PROGRAM_ABI

    class Engine:
        _s = None
        installed = None

        def set_temporal_relations(self, numerators, denominators, policies):
            self.installed = (numerators, denominators, policies)

    layout_identity = "test::adaptive-layout"
    artifact = SimpleNamespace(
        blocks=(),
        plan=SimpleNamespace(blocks=(), field_plans={}),
        layout_plan=SimpleNamespace(
            qualified_id=layout_identity,
            layouts=(SimpleNamespace(adaptive=True),),
        ),
    )
    execution_context = ExecutionContext(
        backend=proven_serial_manifest(
            backend="production",
            target="amr_system",
            abi="test|python-runtime-authorities|v1",
            runtime=True,
        ),
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", "float64"),
        device=ExecutionResource("device", "host"),
    )
    install_plan = SimpleNamespace(
        artifact=artifact,
        amr_execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 3),)),
        resolved_hierarchy=SimpleNamespace(plan=SimpleNamespace(transitions=(object(),))),
        bootstrap_plan=None,
        params={},
        execution_context=execution_context,
        components={},
        amr_providers={
            "clustering": {
                "schema_version": 1,
                "provider_type": "builtin_amr_clustering",
                "provider_id": "pops.lib.amr::berger_rigoutsos",
                "provider_identity": "test::clustering-provider",
                "native_interface": interfaces.Clustering.to_data(),
                "minimum_efficiency": 0.7,
                "minimum_box_size": 1,
                "maximum_box_size": 32,
                "layout_identity": layout_identity,
            },
            "tagger": {
                "schema_version": 1,
                "provider_type": "builtin_amr_tagger",
                "provider_id": "pops.lib.amr::symbolic_tagger",
                "provider_identity": "test::tagger-provider",
                "native_interface": interfaces.Tagger.to_data(),
                "layout_identity": layout_identity,
                "clock_identity": "test::clock",
                "tagging_graph_identity": "test::tagging-graph",
                "tagging_capability": {
                    "schema_version": 1,
                    "capability_type": "amr_tagging_program",
                    "leaf_opcodes": list(tagging_abi["leaf_opcodes"]),
                    "leaf_opcode_ids": list(tagging_abi["leaf_opcodes"].values()),
                    "logical_opcodes": list(tagging_abi["logical_opcodes"]),
                    "logical_opcode_ids": list(tagging_abi["logical_opcodes"].values()),
                    "candidate_outputs": list(tagging_abi["candidate_outputs"]),
                    "indicator_stencil_routes": list(
                        tagging_abi["indicator_stencil_routes"]),
                    "maximum_stencil_terms": tagging_abi[
                        "maximum_stencil_terms"],
                    "maximum_instruction_count": tagging_abi[
                        "maximum_instruction_count"],
                    "non_finite_policy": tagging_abi["non_finite_policy"],
                    "persistent_hysteresis": tagging_abi["persistent_hysteresis"],
                },
            },
        },
    )
    engine = Engine()
    install_runtime_authorities(engine, install_plan)
    assert engine.installed == ([3], [1], ["integral_only"])


def test_tagging_resolution_refuses_unimplemented_persistent_hysteresis():
    from pops.amr import ConflictPolicy, EqualityPolicy, Hysteresis

    authored = Hysteresis(min_cycles=3, equality=EqualityPolicy.COARSEN)
    with pytest.raises(
            NotImplementedError, match="persistent tagging state; it is never accepted"):
        _resolved_target(
            hysteresis=authored,
            conflict_policy=ConflictPolicy.ERROR,
        )


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
    flow_bootstrap_tagging(
        native, authorities.bootstrap, params, clock_identity="case::clock")
    assert native.call is not None
    (blocks, variables, leaf_ops, thresholds, stencil_indices, stencils,
     refine_ops, refine_args, coarsen_ops, coarsen_args, min_cycles,
     equality, conflict, clock, provider) = native.call
    assert blocks == ["tracer", "tracer"]
    # The runtime VM consumes the scalar component token, not the aggregate state handle.
    assert variables == ["u", "u"]
    assert leaf_ops == [4, 5]
    assert thresholds == [0.10, 0.04]
    assert stencil_indices == [0, 0]
    assert len(stencils) == 1
    assert stencils[0]["route"] == "linear_axis_stencil_l2_v1"
    assert (refine_ops, refine_args) == ([4], [0])
    assert (coarsen_ops, coarsen_args) == ([5], [1])
    assert (min_cycles, equality, conflict) == (0, "hold", "refine_wins")
    assert clock == "case::clock"
    assert provider.startswith("pops.bound-amr-tagging-program.v1:sha256:")


def test_layout_preserves_heterogeneous_transitions_before_provider_refusal():
    from pops.amr import AMRHierarchy
    from pops.layouts import AMR
    from pops.mesh.layout_plan import LayoutHandle, normalize_layout
    from pops.model import OwnerPath

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
    assert status.ok
    normalized = normalize_layout(
        LayoutHandle("heterogeneous", owner=OwnerPath.case("ratio-proof")),
        heterogeneous,
        handle_resolver=pops.validate(target.authoring.case).resolve,
    )
    assert normalized.transition_ratios == (2, 4)
    assert tuple(level.refinement for level in normalized.levels) == (1, 2, 8)
    with pytest.raises((ValueError, NotImplementedError), match="transition|provider|ratio"):
        pops.resolve(pops.validate(target.authoring.case), layout=heterogeneous)


def test_symbolic_gradient_indicator_cannot_escape_discrete_resolution():
    from pops.math import SymbolicTruthValueError, ValueExpr, grad, norm

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
