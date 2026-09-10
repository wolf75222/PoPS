"""Source contracts: selected numerics flow into immutable compiler operations."""
from dataclasses import replace

import pytest

import pops
from pops.codegen._compiled_artifact import CompiledPlanRecord
from pops.codegen.lowering_coverage import LoweringRejection
from pops.codegen.resolved_operations import ResolvedOperationPlan
from pops.layouts import Uniform
from pops.numerics import DiscretizationPlan
from tests.python.support.layout_plan import cartesian_grid
from tests.python.unit.numerics.test_discretization_plan import _declarations


def resolved_transport():
    _, model, state, _, rate, method = _declarations()
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, method)
    case = pops.Case("resolved-operations")
    block = case.block("tracer", model)
    case.numerics(numerics, block=block)
    from pops.lib.time import SSPRK2

    case.program(SSPRK2(block[state], rate=rate))
    pops.validate(case)
    return pops.resolve(case, layout=Uniform(cartesian_grid(n=8, periodic=True)))


def test_resolve_keeps_exact_method_and_separate_ssprk_evaluation_coverage():
    resolved = resolved_transport()
    plan = resolved.resolved_operations["tracer"]
    selected = tuple(op for op in plan.operations if "program_evaluation" in op.guarantees)
    assert len(selected) == 2
    assert all(op.stencil_radius == 2 for op in selected)
    assert all(op.guarantees["numerical_method"]["method"] == "finite_volume" for op in selected)
    assert selected[0].evaluation != selected[1].evaluation
    assert selected[0].consumes == selected[1].consumes
    assert selected[0].guarantees["program_evaluation"]["input_values"] != \
        selected[1].guarantees["program_evaluation"]["input_values"]
    assert any("resolved:" in row.source for row in resolved.lowering_coverage.rows)
    assert all(not row["evidence"]["executed"]
               for row in resolved.explain("tracer")["tracer"]["operations"])


def test_compiled_record_detaches_plan_and_preserves_authenticated_explanation():
    resolved = resolved_transport()
    record = CompiledPlanRecord.from_resolved(resolved)
    source = resolved.resolved_operations["tracer"]
    detached = record.resolved_operations["tracer"]
    assert detached is not source
    assert detached.to_data() == source.to_data()
    assert record.explain() == resolved.explain()
    wire = detached.to_data()
    wire["operations"][-1]["stencil_radius"] += 1
    with pytest.raises(ValueError, match="identity/evidence"):
        ResolvedOperationPlan.from_data(wire)


def test_selected_evaluation_requires_current_realization_not_historical_type():
    resolved = resolved_transport()
    plan = resolved.resolved_operations["tracer"]
    operation = next(op for op in plan.operations if "program_evaluation" in op.guarantees)
    from pops.codegen._compiler_lowering import require_compiler_lowering

    module = require_compiler_lowering(resolved.blocks[0].model).source_module
    with pytest.raises(LoweringRejection) as caught:
        plan.require_native(operation.identity, module=module, native_realizations={})
    assert caught.value.gate == "native_realization_unavailable"
    current = {operation.guarantees["declaration_operation"]: operation.native_route}
    assert plan.require_native(operation.identity, module=module, native_realizations=current) \
        == operation.native_route


def test_changing_operation_choices_changes_resolved_plan_identity():
    resolved = resolved_transport()
    original = resolved.blocks[0].resolved_operations
    changed = replace(original, operations=tuple(
        replace(op, guarantees={**op.guarantees, "diagnostic_choice": "changed"})
        for op in original.operations))
    block = replace(resolved.blocks[0], resolved_operations=changed)
    other = replace(resolved, blocks=(block,))
    assert other.plan_identity != resolved.plan_identity


@pytest.mark.parametrize("assign_fv", (False, True))
def test_legacy_named_rhs_has_actual_centered_method_and_refuses_assigned_fv(assign_fv):
    _, model, state, _, rate, method = _declarations()
    case = pops.Case("named-rhs")
    block = case.block("tracer", model)
    if assign_fv:
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, method)
        case.numerics(numerics, block=block)
    program = pops.Program("named-rhs-time")
    temporal = program.state(block[state])
    from pops.numerics.terms import Flux

    module = model.module
    grid = next(op for op in module.operator_registry() if op.kind == "grid_operator")
    rhs = program.rhs(state=temporal.n, terms=(Flux(module.operator_handle(grid.name)),))
    program.commit(temporal.next, program.value(
        "next", temporal.n + program.dt * rhs, at=temporal.next.point))
    case.program(program)
    if assign_fv:
        pops.validate(case)
        with pytest.raises(LoweringRejection) as caught:
            pops.resolve(case, layout=Uniform(cartesian_grid(n=8, periodic=True)))
        assert caught.value.gate == "numerical_method_realization_mismatch"
    else:
        # Public scientific authoring requires a numerical plan. Exercise only
        # the retained low-level Program compatibility path without assigning FV.
        from types import SimpleNamespace
        from pops.codegen._resolved_block_operations import build_block_resolved_operations

        plan = build_block_resolved_operations(SimpleNamespace(
            model=model, numerics=None, spatial=None,
            instance_owner_qid=str(block.instance_owner_path.canonical())), program)
        operation, = [op for op in plan.operations if "program_evaluation" in op.guarantees]
        assert operation.stencil_radius == 1
        assert operation.guarantees["numerical_method"]["method"] == "native_named_centered_divergence"
        assert plan.explain(operation.identity)["operations"][0]["occurrences"][0]["coefficient"] == [-1, 1]
        from pops.codegen._compiler_lowering import require_compiler_lowering

        assert plan.require_native(operation.identity, module=require_compiler_lowering(
            model).source_module) == "legacy:flux"


def test_native_adapter_retains_resolved_plan_across_repeated_compile_entry():
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.component_provider_packs import emitter_carrier_snapshot

    resolved = resolved_transport()
    block = resolved.blocks[0]
    first, module = lower_and_validate(
        block.model, state_space=block.state_spaces[0],
        resolved_operations=block.resolved_operations)
    first_carrier = emitter_carrier_snapshot(first._m)
    second, repeated_module = lower_and_validate(first)
    assert second._resolved_operations is block.resolved_operations
    assert repeated_module.module_hash() == module.module_hash()
    assert emitter_carrier_snapshot(second._m) == first_carrier


@pytest.mark.parametrize("change", ("coefficient", "operator", "target", "kind", "omission"))
def test_native_adapter_reauthenticates_selected_scientific_occurrences(change):
    from fractions import Fraction
    from pops.codegen._compiler_lowering import require_compiler_lowering
    from pops.codegen.module_lowering import lower_and_validate

    resolved = resolved_transport()
    block = resolved.blocks[0]
    plan = block.resolved_operations
    operation = next(item for item in plan.operations if "program_evaluation" in item.guarantees)
    request = next(item for item in plan.evaluations if item.identity == operation.evaluation)
    changes = {"coefficient": {"coefficient": Fraction(42)}, "operator": {"operator": "other"},
               "target": {"target": "other"}, "kind": {"kind": "other"}}
    terms = () if change == "omission" else tuple(replace(term, **changes[change])
                                                 for term in request.occurrences)
    changed_operation = replace(operation, consumes=(), exchanges=()) if change == "omission" else operation
    if change == "target":
        changed_operation = replace(operation, exchanges=tuple(
            replace(exchange, target="other") for exchange in operation.exchanges))
    forged = replace(plan,
        evaluations=tuple(replace(item, occurrences=terms) if item is request else item
                          for item in plan.evaluations),
        operations=tuple(changed_operation if item is operation else item for item in plan.operations))
    module = require_compiler_lowering(block.model).source_module
    with pytest.raises(LoweringRejection) as caught:
        forged.require_native(operation.identity, module=module)
    assert caught.value.gate == "resolved_operation_authority_mismatch"
    with pytest.raises(LoweringRejection) as caught:
        lower_and_validate(block.model, state_space=block.state_spaces[0], resolved_operations=forged)
    assert caught.value.gate == "resolved_operation_authority_mismatch"


@pytest.mark.parametrize("change", ("effects", "inputs", "outputs", "sampling", "declaration"))
def test_native_adapter_preserves_scientific_input_output_and_effect_obligations(change):
    from pops import model
    from pops._ir.expr import Const
    from pops.codegen.resolved_operations import build_resolved_operations

    module = model.Module("fallible-source")
    state = module.state_space("U", ("rho",))
    rho, = module.state_symbols(state)
    module.operator("source", state >> model.Rate(state), "local_source", expr=(Const(1) / rho,))
    plan = build_resolved_operations(module)
    operation, = plan.operations
    changes = {"effects": {"effects": ()}, "inputs": {"inputs": ()}, "outputs": {"outputs": ()},
               "sampling": {"sampling": "face_trace"},
               "declaration": {"guarantees": {"declaration_operation": "unregistered"}}}
    forged = replace(plan, operations=(replace(operation, **changes[change]),))
    with pytest.raises(LoweringRejection) as caught:
        forged.require_native(operation.identity, module=module)
    assert caught.value.gate == "resolved_operation_authority_mismatch"
