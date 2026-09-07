from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest

from pops import model
from pops._ir.expr import Const, Var
from pops._ir.values import StateRef
from pops.codegen.lowering_coverage import LoweringRejection
from pops.codegen.resolved_operations import (
    EvaluationRequest, ExchangeRecord, NumericalConstruction, ResolvedAccess,
    ResolvedOperationPlan, TermOccurrence, build_resolved_operations,
    require_semantic_rewrite,
)


def source_module():
    module = model.Module("resolved_source")
    state = module.state_space("U", ("rho", "momentum"))
    rho, _ = module.state_symbols(state)
    module.operator("source", (state,) >> model.Rate(state), "local_source",
                    expr=(rho, Const(0)))
    return module


def occurrence(identity="term", coefficient=1):
    return TermOccurrence(identity, "physical:operator", "quantity:U", "source", Fraction(coefficient))


def construction(identity="numerical", consumes=("term",), evaluation="evaluation", **kwargs):
    return NumericalConstruction(identity, evaluation, consumes, **kwargs)


def plan(operations, terms=(occurrence(),)):
    return ResolvedOperationPlan("source-hash", (EvaluationRequest("evaluation", terms),), operations)


def test_baseline_source_plan_uses_current_native_route_and_existing_typed_carrier():
    module = source_module()
    resolved = build_resolved_operations(module)
    operation = resolved.operations[0]
    assert resolved.require_native(operation.identity, module=module) == "legacy:source_term"
    assert len(resolved.require_provider_packs(module).physical_flux) == 0
    state_reads = [item for item in operation.inputs if item.kind == "state"]
    assert [(item.components, item.complete) for item in state_reads] == [(('rho',), False)]
    explanation = resolved.explain(operation.identity)["operations"][0]
    assert explanation["evidence"]["native_route_selected"] is True
    assert explanation["evidence"]["executed"] is False
    assert explanation["occurrences"][0]["coefficient"] == [1, 1]


def test_different_physical_supports_require_an_explicit_map_even_at_equal_rank():
    access = ResolvedAccess("phase", "state", {"kind": "state", "support": ["x", "v"]})
    output = {"kind": "field", "support": ["x", "y"]}
    with pytest.raises(ValueError, match="explicit physical map"):
        construction(inputs=(access,), outputs=(output,))
    construction(inputs=(replace(access, physical_map="moment:v"),), outputs=(output,))


def test_fusion_cannot_cross_an_intervening_effect_or_reverse_evaluation_order():
    terms = (occurrence("a"), occurrence("barrier"), occurrence("b"))
    resolved = plan((construction("a", ("a",), native_route="pointwise"),
        construction("barrier", ("barrier",), effects=("collective",)),
        construction("b", ("b",), native_route="pointwise")), terms)
    assert not resolved.can_fuse("a", "b")
    assert not resolved.can_fuse("b", "a")


def test_domain_restricted_expression_is_a_fallible_operation_barrier():
    module = model.Module("fallible")
    state = module.state_space("U", ("rho",))
    (rho,) = module.state_symbols(state)
    module.operator("inverse", (state,) >> model.Rate(state), "local_source", expr=(1 / rho,))
    assert "fallible" in build_resolved_operations(module).operations[0].effects


def test_resolved_plan_wire_identity_and_nested_immutability():
    resolved = build_resolved_operations(source_module())
    wire = resolved.to_data()
    restored = ResolvedOperationPlan.from_data(wire)
    assert restored.identity == resolved.identity
    assert restored.to_data() == wire
    with pytest.raises(FrozenInstanceError):
        resolved.operations = ()
    with pytest.raises(TypeError):
        resolved.operations[0].outputs[0]["representation"] = "integrated"
    wire["operations"][0]["sampling"] = "right_face_trace"
    with pytest.raises(ValueError, match="identity/evidence"):
        ResolvedOperationPlan.from_data(wire)


def test_fresh_authoring_objects_have_the_same_canonical_operation_identity():
    first = build_resolved_operations(source_module())
    second = build_resolved_operations(source_module())
    assert first.to_data() == second.to_data()
    assert "authoring=" not in first.operations[0].identity
    assert all("authoring=" not in read.reference for read in first.operations[0].inputs)


def test_joint_signed_coverage_retains_occurrence_multiplicity_and_refuses_duplicate_or_gap():
    terms = (occurrence("transport", -1), occurrence("diffusion", 1))
    joint = construction(consumes=("transport", "diffusion"), native_route="test:joint")
    resolved = plan((joint,), terms)
    assert [row["coefficient"] for row in resolved.explain()["operations"][0]["occurrences"]] == [[-1, 1], [1, 1]]
    with pytest.raises(LoweringRejection) as caught:
        plan((joint, construction("duplicate", ("diffusion",))), terms)
    assert caught.value.gate == "duplicate_term_coverage"
    with pytest.raises(LoweringRejection) as caught:
        plan((construction(consumes=("transport",)),), terms)
    assert caught.value.gate == "incomplete_term_coverage"


def test_repeated_stages_can_evaluate_the_same_physical_occurrence_again():
    term = occurrence()
    resolved = ResolvedOperationPlan("source-hash", (
        EvaluationRequest("stage0", (term,), context="t=0"),
        EvaluationRequest("stage1", (term,), context="t=0"),
    ), (construction("a", evaluation="stage0"), construction("b", evaluation="stage1")))
    assert len(resolved.coverage.target_to_sources["a"]) == 1
    assert len(resolved.coverage.target_to_sources["b"]) == 1


def test_joint_only_construction_refuses_an_invented_imex_decomposition():
    joint = construction(consumes=("transport", "diffusion"))
    joint.require_partition(("transport", "diffusion"))
    with pytest.raises(ValueError, match="no declared realization"):
        joint.require_partition(("diffusion",))
    decomposable = replace(joint, decomposition=(("transport",), ("diffusion",)))
    decomposable.require_partition(("diffusion",))
    with pytest.raises(ValueError, match="exactly partition"):
        replace(joint, decomposition=(("transport",), ("transport",)))


def test_stencil_chain_adds_path_depth_but_parallel_inputs_take_maximum():
    terms = tuple(occurrence(key) for key in ("a", "b", "c"))
    resolved = plan((
        construction("a", ("a",), stencil_radius=2),
        construction("b", ("b",), stencil_radius=3),
        construction("c", ("c",), dependencies=("a", "b"), stencil_radius=1),
    ), terms)
    assert dict(resolved.halo_requirements()) == {"a": 2, "b": 3, "c": 4}
    exchanged = replace(resolved.operations[2], materialized_inputs=("a", "b"),
                        communication=("halo_exchange:a", "halo_exchange:b"))
    assert plan((*resolved.operations[:2], exchanged), terms).halo_requirements()["c"] == 1
    with pytest.raises(ValueError, match="halo exchange"):
        replace(exchanged, communication=())


def test_planner_refuses_explicit_cycles_and_unknown_dependencies():
    with pytest.raises(LoweringRejection) as caught:
        plan((construction(dependencies=("numerical",)),))
    assert caught.value.gate == "explicit_operation_cycle"
    with pytest.raises(LoweringRejection) as caught:
        plan((construction(dependencies=("missing",)),))
    assert caught.value.gate == "missing_operation_dependency"


@pytest.mark.parametrize("changes", [
    {"effects": ("fallible",)}, {"effects": ("solve", "collective")},
    {"guards": ("positive_density",)}, {"sampling": "left_face_trace"},
    {"stencil_radius": 1}, {"communication": ("exchange",)},
])
def test_fusion_respects_effect_guard_sampling_and_stencil_boundaries(changes):
    terms = (occurrence("a"), occurrence("b"))
    first = construction("a", ("a",), native_route="test:a")
    second = construction("b", ("b",), dependencies=("a",), native_route="test:b")
    assert plan((first, second), terms).can_fuse("a", "b")
    assert not plan((first, replace(second, **changes)), terms).can_fuse("a", "b")


def test_exchange_must_match_the_consumed_occurrence_and_physical_target():
    with pytest.raises(LoweringRejection) as caught:
        plan((construction(exchanges=(ExchangeRecord("term", "another_quantity"),)),))
    assert caught.value.gate == "exchange_occurrence_mismatch"
    with pytest.raises(ValueError, match="orientation"):
        ExchangeRecord("term", "quantity:U", orientation=True)


def test_unsupported_native_kind_is_structured_and_source_drift_is_not_accepted():
    module = model.Module("matrix")
    state = module.state_space("U", ("rho",))
    module.operator("matrix", (state,) >> model.MatrixFreeOperator(state, state),
                    "matrix_free_operator", expr=Const(0))
    resolved = build_resolved_operations(module)
    with pytest.raises(LoweringRejection) as caught:
        resolved.require_native(resolved.operations[0].identity, module=module)
    assert caught.value.gate == "operator_kind_not_lowerable"
    source = source_module()
    resolved = build_resolved_operations(source)
    source.operator("extra", (source.state_spaces()["U"],) >> model.Rate(source.state_spaces()["U"]),
                    "local_source", expr=(Const(0), Const(0)))
    with pytest.raises(LoweringRejection) as caught:
        resolved.require_native(resolved.operations[0].identity, module=source)
    assert caught.value.gate == "source_module_drift"


def test_homonymous_state_components_and_left_right_samples_stay_distinct():
    module = model.Module("two_states")
    a = module.state_space("a", ("rho",))
    b = module.state_space("b", ("rho",))
    qa, = module.state_symbols(a)
    qb, = module.state_symbols(b)
    field = module.field_space("field", ("potential",))
    module.operator("interaction", (a, b) >> field, "field_operator",
                    expr=(StateRef("L", qa) + StateRef("R", qb),))
    resolved = build_resolved_operations(module)
    assert resolved.operations[0].refusal == "multi_state_field_provider_unsupported"
    reads = resolved.operations[0].inputs
    assert {(item.reference, item.sampling) for item in reads} == {
        (module.state_handle(a)._resolved().qualified_id, "left_face_trace"),
        (module.state_handle(b)._resolved().qualified_id, "right_face_trace"),
    }


def test_opaque_body_reads_complete_inputs_without_a_fake_native_success():
    module = model.Module("opaque")
    state = module.state_space("U", ("rho", "momentum"))
    module.operator("closure", (state,) >> model.Rate(state), "local_source", expr=lambda: None)
    resolved = build_resolved_operations(module)
    read, = resolved.operations[0].inputs
    assert read.complete and read.components == ("rho", "momentum")
    assert resolved.operations[0].refusal == "expression_body_required"


def test_native_footprint_and_boundary_dependencies_join_typed_input_reads():
    module = model.Module("footprint")
    state = module.state_space("U", ("rho", "momentum"))
    rho, momentum = module.state_symbols(state)

    class NativeClosure:
        def __call__(self):
            raise AssertionError("resolution must never execute a native closure")

        def __pops_native_read_footprint__(self):
            return (rho,)

    module.operator("closure", state >> model.Rate(state), "local_source", expr=NativeClosure())
    resolved = build_resolved_operations(module, boundary_data=(momentum,))
    reads = resolved.operations[0].inputs
    assert {(read.components, read.sampling) for read in reads} == {
        (("rho",), "cell"), (("momentum",), "boundary")}
    assert {origin for read in reads for origin in read.origins} == {
        "native_footprint", "physical_boundary"}


def test_wave_speed_dependencies_are_retained_even_when_flux_body_is_constant():
    module = model.Module("waves")
    state = module.state_space("U", ("rho",))
    (rho,) = module.state_symbols(state)
    module.operator("flux", state >> model.Rate(state), "grid_operator", expr=(Const(0),))
    module.eigenvalues(x=(rho,))
    reads = build_resolved_operations(module).operations[0].inputs
    assert len(reads) == 1 and reads[0].components == ("rho",)
    assert reads[0].origins == ("wave_speed",)


def test_transparent_field_component_reads_narrow_the_existing_native_provider_plan():
    module = model.Module("field_projection")
    state = module.state_space("U", ("rho",))
    field = module.field_space("electric", ("phi", "Ex", "Ey"))
    module.operator("solve", state >> field, "field_operator", expr=Const(1))
    module.operator("force", (state, field) >> model.Rate(state), "local_source",
                    expr=(Var("Ex", "aux"),))
    module.operator("unused", (state, field) >> model.Rate(state), "local_source",
                    expr=(Const(0),))
    resolved = build_resolved_operations(module)
    packs = resolved.require_provider_packs(module)
    assert tuple(key.component for key in packs.by_operator["force"]) == ("Ex",)
    assert len(packs.by_operator["unused"]) == 0
    assert {key.component for key in packs.complete if key.space_kind == "field"} == {"phi", "Ex", "Ey"}


def test_field_result_projections_keep_distinct_operator_qualified_targets():
    def make_module(*, pure_component="pure_phi"):
        module = model.Module("two_field_results")
        state = module.state_space("U", ("pure_forcing", "screened_forcing"))
        pure, screened = module.state_symbols(state)
        module.field_space("fields", ("pure_phi", "screened_phi"))
        for name, component, body in (("pure_poisson", pure_component, pure),
                                      ("screened_poisson", "screened_phi", screened)):
            result = model.FieldSpace(name, (component,))
            module.operator(name, state >> result, "field_operator", expr=body)
        return module

    module = make_module()
    resolved = build_resolved_operations(module)
    wire = resolved.to_data()
    targets = tuple(request.occurrences[0].target for request in resolved.evaluations)
    assert targets == tuple("output:%s" % operation.identity for operation in resolved.operations)
    assert len(set(targets)) == 2
    carrier = module.field_handle(module.field_spaces()["fields"])._resolved().qualified_id
    assert all(target != carrier and "authoring=" not in target for target in targets)
    assert [operation["outputs"] for operation in wire["operations"]] == [
        [operator.signature.output.to_data()] for operator in module.operator_registry()]
    assert {key.component for key in resolved.require_provider_packs(module).complete
            if key.space_kind == "field"} \
        == {"pure_phi", "screened_phi"}
    assert ResolvedOperationPlan.from_data(wire).to_data() == wire
    assert build_resolved_operations(make_module()).to_data() == wire
    changed = build_resolved_operations(make_module(pure_component="screened_phi"))
    assert changed.source_module_hash != resolved.source_module_hash
    assert changed.identity != resolved.identity


def test_unregistered_field_output_does_not_authorize_an_unregistered_input():
    module = model.Module("invalid_field_input")
    module.state_space("U", ("rho",))
    unregistered = model.StateSpace("foreign", ("rho",))
    result = model.FieldSpace("pure_poisson", ("pure_phi",))
    module.operator("solve", unregistered >> result, "field_operator", expr=Const(0))
    with pytest.raises(ValueError, match="input representation differs from its Module registry"):
        build_resolved_operations(module)


def test_registered_field_output_still_requires_its_exact_carrier_descriptor():
    module = model.Module("invalid_registered_field_output")
    state = module.state_space("U", ("rho",))
    module.field_space("fields", ("pure_phi", "screened_phi"))
    partial = model.FieldSpace("fields", ("pure_phi",))
    module.operator("solve", state >> partial, "field_operator", expr=Const(0))
    with pytest.raises(ValueError, match="representation differs from its Module registry"):
        build_resolved_operations(module)


def test_explicit_field_requirement_adds_a_component_not_read_by_the_body():
    module = model.Module("field_requirement_union")
    state = module.state_space("U", ("rho",))
    field = module.field_space("coefficients", ("first", "second"))
    module.operator("provide", state >> field, "field_operator", expr=Const(1))
    module.operator("consume", (state, field) >> model.Rate(state), "local_source",
                    expr=(Var("first", "aux"),), requirements={"aux": ("second",)})
    resolved = build_resolved_operations(module)
    consume = next(op for op in resolved.operations if op.identity.endswith("::target::consume"))
    assert {(read.components, read.sampling) for read in consume.inputs} == {
        (("first",), "cell"), (("second",), "cell")}
    second = next(read for read in consume.inputs if read.components == ("second",))
    assert second.origins == ("declared_provider_requirement",)
    packs = resolved.require_provider_packs(module)
    assert {key.component for key in packs.by_operator["consume"]} == {"first", "second"}


def test_explicit_requirement_needs_its_sampling_context_even_for_a_read_component():
    module = model.Module("field_requirement_sampling")
    state = module.state_space("U", ("rho",))
    field = module.field_space("coefficients", ("first", "second"))
    module.operator("provide", state >> field, "field_operator", expr=Const(1))
    module.operator("consume", (state, field) >> model.Rate(state), "local_source",
                    expr=(StateRef("R", Var("second", "aux")),),
                    requirements={"aux": ("second",)})
    resolved = build_resolved_operations(module)
    consume = next(op for op in resolved.operations if op.identity.endswith("::target::consume"))
    assert {(read.components, read.sampling) for read in consume.inputs} == {
        (("second",), "right_face_trace"), (("second",), "cell")}
    assert [key.component for key in resolved.require_provider_packs(module).by_operator["consume"]] \
        == ["second"]


def test_imposed_auxiliary_uses_its_registry_handle_and_exact_representation():
    module = model.Module("auxiliary")
    state = module.state_space("U", ("rho",))
    imposed = module.aux_field("temperature", unit="K")
    module.operator("heating", state >> model.Rate(state), "local_source",
                    expr=(Var("temperature", "aux"),), requirements={"aux": ("temperature",)})
    resolved = build_resolved_operations(module)
    read, = resolved.operations[0].inputs
    assert read.reference == module.aux_handle(imposed)._resolved().qualified_id
    assert read.representation["unit"] == "K"
    assert read.components == ("temperature",)


def test_transitive_primitive_recipes_keep_precise_reads_and_sampling_contexts():
    module = model.Module("recipes")
    state = module.state_space("U", ("rho", "momentum"))
    field = module.field_space("electric", ("phi", "Ex", "Ey"))
    rho, _ = module.state_symbols(state)
    module.set_primitive_recipes({"density": rho, "pressure": 2 * Var("density", "prim")})
    module.operator("pressure_force", (state, field) >> model.Rate(state), "local_source",
                    expr=(Var("pressure", "prim") + Var("pressure", "prim"), Const(0)))
    resolved = build_resolved_operations(module, boundary_data=(
        StateRef("R", Var("pressure", "prim")),))
    reads = resolved.operations[0].inputs
    assert {(read.components, read.sampling, read.complete) for read in reads} == {
        (("rho",), "cell", False), (("rho",), "right_face_trace", False)}
    assert len(resolved.require_provider_packs(module).by_operator["pressure_force"]) == 0


def test_domain_effects_in_a_primitive_recipe_reach_the_numerical_operation():
    module = source_module()
    (rho, _) = module.state_symbols(module.state_spaces()["U"])
    module.set_primitive_recipes({"inverse_density": 1 / rho})
    state = module.state_spaces()["U"]
    module.operator("inverse", state >> model.Rate(state), "local_source",
                    expr=(Var("inverse_density", "prim"), Const(0)))
    inverse = build_resolved_operations(module).operations[-1]
    assert "fallible" in inverse.effects


def test_m1_balance_views_preserve_signed_occurrences_and_real_input_bodies():
    from tests.python.unit.numerics.test_discretization_plan import _declarations
    from pops.math import ddt, div

    _, physical, state, flux, _, _ = _declarations()
    source = physical.source("forcing", on=state, value=[state[0]])
    rate = physical.rate("signed", equation=ddt(state) == -div(flux) + source + Fraction(2, 3) * source)
    selected = rate.select(rate.occurrences[2])
    module = physical.module
    resolved = build_resolved_operations(module)
    operation = next(op for op in resolved.operations
                     if op.identity.endswith("::target::" + rate.local_id))
    explanation = resolved.explain(operation.identity)["operations"][0]
    assert [term["coefficient"] for term in explanation["occurrences"]] == [[-1, 1], [1, 1], [2, 3]]
    assert len({term["identity"] for term in explanation["occurrences"]}) == 3
    assert len(operation.exchanges) == 1
    assert any(read.components == ("u",) for read in operation.inputs)
    view = next(op for op in resolved.operations
                if op.identity.endswith("::target::" + selected.local_id))
    assert view.consumes == (operation.consumes[2],)
    assert view.refusal == "unsupported_physical_balance"


def test_m1_joint_projection_retains_shared_context_and_explicit_native_refusal():
    from tests.python.unit.physics.test_joint_rate_projection_balance import _joint_model
    from pops.math import ddt

    physical, left, _, application, captures = _joint_model()
    physical.rate("balance", equation=ddt(left) == application[left] + application[left])
    resolved = build_resolved_operations(physical.module)
    operation = resolved.operations[-1]
    assert operation.refusal == "unsupported_physical_balance"
    terms = resolved.explain(operation.identity)["operations"][0]["occurrences"]
    assert terms[0]["operator"] == terms[1]["operator"]
    assert terms[0]["identity"] != terms[1]["identity"]
    assert {(read.kind, read.components) for read in operation.inputs} == {
        ("state", ("rho",)), ("state", ("energy",))}
    assert captures == [1]

    other, other_left, _, other_application, _ = _joint_model()
    other.rate("balance", equation=ddt(other_left) ==
               other_application[other_left] + other_application[other_left])
    assert build_resolved_operations(other.module).to_data() == resolved.to_data()


def test_spatial_coefficient_and_accumulation_rewrites_fail_closed():
    for rewrite in ("move_coefficient_through_divergence", "discrete_accumulation_chain_rule",
                    "infer_inverse_physical_map"):
        with pytest.raises(ValueError, match="normalization refuses"):
            require_semantic_rewrite(rewrite)
    require_semantic_rewrite("move_coefficient_through_divergence", spatially_constant_coefficient=True)


def test_noncanonical_exact_coefficient_and_stray_wire_fields_are_rejected():
    with pytest.raises(ValueError, match="not canonical"):
        TermOccurrence.from_data({"identity": "t", "operator": "o", "target": "u",
                                  "kind": "source", "coefficient": [2, 2]})
    wire = build_resolved_operations(source_module()).to_data()
    wire["diagnostic_line"] = 12
    with pytest.raises(TypeError, match="invalid schema"):
        ResolvedOperationPlan.from_data(wire)
