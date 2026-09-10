"""Selected compiler views of the full two-fluid example keep separate authority."""
from dataclasses import replace
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import pops
from pops.codegen import Production
from pops.codegen._compiler_lowering import require_compiler_lowering
from pops.codegen.component_provider_packs import (
    emitter_carrier_snapshot,
    resolve_component_provider_packs,
)
from pops.codegen.lowering_coverage import LoweringRejection
from pops.codegen.module_lowering import lower_and_validate


@pytest.fixture(scope="module")
def full_multiphysics():
    path = Path(__file__).resolve().parents[4] / (
        "examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py")
    spec = importlib.util.spec_from_file_location("_multistate_selected_example", path)
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    target = example.build_final_case()
    resolved = pops.resolve(
        target.authoring.case,
        layout=target.layout_plan,
        layout_providers={target.layout_handle: target.layout_provider},
        backend=Production(),
    )
    assert {block.name for block in resolved.blocks} == {"electrons", "ions"}
    return target, resolved


def _pack_snapshot(packs):
    target = SimpleNamespace()
    packs.attach(target)
    return emitter_carrier_snapshot(target)


def _source_snapshot(model):
    return (
        model.module.module_hash(),
        id(model._dsl),
        id(model._dsl._m),
        tuple(model._dsl.cons_names),
        emitter_carrier_snapshot(model._dsl),
        emitter_carrier_snapshot(model._dsl._m),
    )


def test_full_multiphysics_selected_packs_and_repeated_views_are_independent(full_multiphysics):
    target, resolved = full_multiphysics
    model = target.authoring.model
    module = require_compiler_lowering(model).source_module
    before = _source_snapshot(model)
    default_packs = _pack_snapshot(resolve_component_provider_packs(module))
    states = module.state_spaces()
    assert len(states) == 2
    emitted = []

    # Interleave both species, then repeat the first species. In particular, no
    # default pack may be attached before a selected Program evaluation pack.
    for block in (*resolved.blocks, resolved.blocks[0]):
        state_name, = block.state_spaces
        plan = block.resolved_operations
        expected = _pack_snapshot(plan.require_provider_packs(module))
        assert expected != default_packs
        emitter, source = lower_and_validate(
            block.model, state_space=state_name, resolved_operations=plan)
        assert source is module
        assert emitter._resolved_operations is plan
        assert tuple(emitter.cons_names) == tuple(states[state_name].components)
        assert emitter_carrier_snapshot(emitter) == expected
        assert emitter_carrier_snapshot(emitter._m) == expected

        # A subsequent compiler entry must inherit and reauthenticate the exact
        # plan while still requiring the explicit route through this Module.
        repeated, repeated_source = lower_and_validate(emitter, state_space=state_name)
        assert repeated_source is module
        assert repeated._resolved_operations is plan
        assert tuple(repeated.cons_names) == tuple(states[state_name].components)
        assert emitter_carrier_snapshot(repeated) == expected
        assert emitter_carrier_snapshot(repeated._m) == expected
        emitted.extend(((emitter, expected), (repeated, expected)))

    assert len({id(emitter) for emitter, _ in emitted}) == len(emitted)
    assert len({id(emitter._m) for emitter, _ in emitted}) == len(emitted)
    for emitter, expected in emitted:
        assert emitter_carrier_snapshot(emitter) == expected
        assert emitter_carrier_snapshot(emitter._m) == expected
    assert _source_snapshot(model) == before


@pytest.mark.parametrize("change, gate", (
    ("source", "source_module_drift"),
    ("providers", "provider_plan_drift"),
    ("coefficient", "resolved_operation_authority_mismatch"),
))
def test_full_multiphysics_rejects_inauthentic_selected_plan_without_source_mutation(
    full_multiphysics, change, gate,
):
    target, resolved = full_multiphysics
    model = target.authoring.model
    before = _source_snapshot(model)
    block = resolved.blocks[0]
    plan = block.resolved_operations
    if change == "source":
        forged = replace(plan, source_module_hash="foreign-source")
    elif change == "providers":
        forged = replace(plan, provider_evidence={})
    else:
        operation = next(item for item in plan.operations
                         if "program_evaluation" in item.guarantees
                         and any(request.identity == item.evaluation and request.occurrences
                                 for request in plan.evaluations))
        request = next(item for item in plan.evaluations if item.identity == operation.evaluation)
        changed = replace(request, occurrences=tuple(
            replace(term, coefficient=term.coefficient + Fraction(1))
            for term in request.occurrences))
        forged = replace(plan, evaluations=tuple(
            changed if item is request else item for item in plan.evaluations))
    with pytest.raises(LoweringRejection) as caught:
        lower_and_validate(
            block.model, state_space=block.state_spaces[0], resolved_operations=forged)
    assert caught.value.gate == gate
    assert _source_snapshot(model) == before


def test_full_multiphysics_detached_program_emits_exact_qualified_collision_inputs(
    full_multiphysics,
):
    from pops._balance_due_contract import BalanceDueContract
    from pops._ir.quantity import QuantityRef
    from pops.codegen._shared_interface_evidence import _issue_shared_interface_codegen_evidence
    from pops.codegen.program_emit_control import _walk_expr
    from pops.codegen.program_graph_lowering import _emit_resolved_program_graph
    from pops.codegen.program_models import ProgramModelGraph
    from pops.time._program.detach import detach_compiled_program

    target, resolved = full_multiphysics
    model = target.authoring.model
    before = _source_snapshot(model)
    source_program_hash = resolved.time._ir_hash()
    program = detach_compiled_program(resolved.time)
    graph = program.to_graph()
    authority = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    collision = next(op for op in model.module.operator_registry() if op.kind == "coupled_rate")
    quantities = tuple(q for formulas in collision.body.values() for formula in formulas
                       for q in _walk_expr(formula) if isinstance(q, QuantityRef))
    assert quantities
    assert all(q.handle.owner_path.is_authoring for q in quantities)
    call = next(value for value in program._values if value.op == "solve_coupled_implicit")
    assert all(value.state_ref.is_resolved for value in call.inputs)
    assert not program._operator_registries
    kwargs = dict(
        lowering_program=program,
        model_graph=authority,
        field_plans=resolved.field_plans,
        balance_due_contract=BalanceDueContract.from_consumer_graph(resolved.consumer_graph),
        shared_interface_codegen_evidence=_issue_shared_interface_codegen_evidence(resolved),
    )
    cpp = _emit_resolved_program_graph(graph, **kwargs)
    assert "const pops::Real pops_input_0_component_1 = Ueval[1];" in cpp
    assert "const pops::Real pops_input_1_component_1 = Ueval[4];" in cpp
    assert "const pops::Real pops_input_0_component_1 = Ueval[4];" not in cpp
    assert "const pops::Real pops_input_1_component_1 = Ueval[1];" not in cpp
    assert _emit_resolved_program_graph(graph, **kwargs) == cpp
    assert program.to_graph().graph_hash == graph.graph_hash
    assert resolved.time._ir_hash() == source_program_hash
    assert _source_snapshot(model) == before
    assert all(q.handle.owner_path.is_authoring for q in quantities)


def test_detached_collision_binding_does_not_collapse_equal_authored_modules():
    from pops.codegen.program_emit_control import _detached_coupled_quantity_identity
    from pops.model import Module

    source = Module("identical_display")
    source_state = source.state_space("electrons", ("rho",))
    own_quantity, = source.state_symbols(source_state)
    foreign = Module("identical_display")
    foreign_state = foreign.state_space("electrons", ("rho",))
    foreign_quantity, = foreign.state_symbols(foreign_state)
    # Even identical canonical content cannot grant a foreign live declaration
    # the capability owned by the retained source Module.
    assert source.module_hash() == foreign.module_hash()
    assert _detached_coupled_quantity_identity(own_quantity, source) is not None
    assert _detached_coupled_quantity_identity(foreign_quantity, source) is None
