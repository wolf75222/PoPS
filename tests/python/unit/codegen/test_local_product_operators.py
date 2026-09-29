"""Physical local operators inside an original residual product."""
import pytest
from pops import time
from pops.physics._facade import Model
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.solvers.nonlinear import LocalNewton
from typed_program_support import typed_state, solve_field, codegen_field_plans


def operator_product(*, nonlocal_body=False, auxiliary=False, fields=False,
                     field_on_candidate=False, source_on_capture=False, duplicate_capture=False):
    model = Model("product_physics")
    x,y = model.conservative_vars("x", "y")
    model.primitive_vars(x,y)
    model.conservative_from([x,y])
    model.flux(x=[0*x,0*y], y=[0*x,0*y])
    model.eigenvalues(x=[0*x,0*x],y=[0*y,0*y])
    from pops.params import RuntimeParam
    gain = model.value(model.param(RuntimeParam("gain", default=.5)))
    coefficient = model.aux("frozen_coefficient") if auxiliary else 1
    source = model.source_term("quadratic", [gain*x*x*coefficient,gain*y*y*coefficient])
    linear = model.linear_source("linear", [[1,0],[0,2]])
    program = time.Program("product_physical_calls")
    a,b = (typed_state(program,key,model=model) for key in ("left","right"))
    field = solve_field(program, a) if fields else None
    before = program._ir_hash()

    def residual(P,z,*,old_a,old_b,frozen=None,alias=None):
        if nonlocal_body:
            P.gradient(old_a)
        state = old_a if source_on_capture else z["a"]
        if field_on_candidate:
            state = z["a"]
        if alias is not None:
            assert alias is old_a
        sa = P.source(source,state=state,fields=frozen)
        lb = P.apply(linear,state=z["b"])
        return {"a":tuple((z["a"][i]-sa[i] if source_on_capture else sa[i]-old_a[i])
                          for i in range(2)),
                "b":tuple(lb[i]+z["a"][i]-old_b[i] for i in range(2))}

    try:
        outcome = program.solve(time.LocalResidual(residual, {"b":b,"a":a},
            captures={"old_b":b,"old_a":a, **({"frozen":field} if fields else {}),
                      **({"alias":a} if duplicate_capture else {})}),
            solver=LocalNewton(tolerance=1e-11))
    except Exception:
        assert program._ir_hash() == before
        raise
    solved = outcome.consume(action=time.FailRun())
    for value in (a,b):
        end = typed_state(program,value.block.local_id,state_name="U",model=model).next
        program.commit(end, program.value("next_"+value.block.local_id,solved[value.block],at=end.point))
    return program, lower_and_validate(model)[0]


def test_local_operator_product_has_closed_subgraph_and_per_block_ports():
    program, model = operator_product()
    assert program.validate()
    token = next(v for v in program._values if v.op=="solve_coupled_implicit")
    assert token.attrs["product_version"] == 2
    assert {v.op for v in token.attrs["residual_block"]} == {"state","source","apply"}
    source = emit_cpp_program(program,model=model)
    assert "prepare_local_nonlinear_problem<4>" in source
    assert "product_params_" in source
    assert "ctx.program_params(0)" in source
    assert "residual_value_" in source


def test_local_operator_product_does_not_freeze_candidate_dependent_auxiliary():
    program, model = operator_product(auxiliary=True)
    with pytest.raises(NotImplementedError, match="provider evaluation at a Newton candidate"):
        emit_cpp_program(program,model=model)


def test_nonlocal_operation_cannot_be_hidden_in_local_product():
    with pytest.raises((ValueError,TypeError),match="LOCAL|scalar_field|gradient"):
        operator_product(nonlocal_body=True)


def test_field_capture_retains_frozen_state_authority_and_preparation_action():
    program, model = operator_product(auxiliary=True, fields=True, source_on_capture=True)
    assert program.validate()
    token = next(v for v in program._values if v.op == "solve_coupled_implicit")
    source_call = next(v for v in token.attrs["residual_block"] if v.op == "source")
    assert source_call.field_context.stage_sources == ((source_call.block, source_call.inputs[0].id),)
    assert source_call.inputs[0].id != token.attrs["residual_block"][0].id
    rendered = emit_cpp_program(program, model=model, field_plans=codegen_field_plans(program))
    assert "ctx.prepare_provider_values_for_solve(" in rendered
    assert "auxiliary_nonfinite_candidate" in rendered
    # Residual metadata survives the serialised IR, including captured fields.
    assert "product_argument_positions" in str(program._serialize())
    normalized = program.optimize()
    assert normalized.validate()
    assert "prepare_local_nonlinear_problem<4>" in emit_cpp_program(
        normalized, model=model, field_plans=codegen_field_plans(normalized))


def test_field_capture_cannot_be_relabelled_as_a_newton_candidate():
    with pytest.raises(ValueError, match="state|State|context|Context"):
        operator_product(fields=True, field_on_candidate=True)


def test_duplicate_frozen_capture_alias_does_not_change_field_authority():
    program, model = operator_product(fields=True, source_on_capture=True, duplicate_capture=True)
    assert program.validate()
    for copy in (program, program.optimize()):
        assert "prepare_local_nonlinear_problem<4>" in emit_cpp_program(
            copy, model=model, field_plans=codegen_field_plans(copy))


@pytest.mark.parametrize("widths", ((1,2), (2,3), (1,2,4)))
@pytest.mark.parametrize("reverse", (False, True))
def test_public_case_detaches_exact_local_operator_products_and_full_loaders(widths, reverse):
    import pops
    from pops.codegen.program_models import ProgramModelGraph
    from pops.time._program.detach import detach_compiled_program
    from tests.python.support.local_product_operator_case import make_case
    case, layout, _ = make_case(widths, reverse=reverse)
    plan = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    detached = detach_compiled_program(plan.time)
    assert detached._ir_hash() == plan.time._ir_hash()
    assert emit_cpp_program(detached, model=graph) == emit_cpp_program(plan.time, model=graph)
    assert not detached._operator_registries
    for program in (plan.time, detached, plan.time.optimize()):
        rendered = emit_cpp_program(program, model=graph)
        assert "prepare_local_nonlinear_problem<%d>" % sum(widths) in rendered
        assert rendered.count("auto residual_eval =") == 1
        for block in plan.blocks:
            emitter, _ = lower_and_validate(block.model, state_space=block.state_spaces[0],
                resolved_operations=block.resolved_operations, numerics=block.numerics)
            loader = emitter._m.emit_cpp_native_loader(name="LocalOps"+block.name,
                target="system", consumer_owner_qid=block.instance_owner_qid)
            assert "program_only_storage = true" in loader
            assert "State flux(" not in loader
