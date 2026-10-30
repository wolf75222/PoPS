import pops
import pytest
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_codegen import emit_cpp_program
from tests.python.support.finite_m09_case import case_module, oracle


@pytest.mark.parametrize("condensed", (False, True))
@pytest.mark.parametrize("permuted", (False, True))
def test_true_finite_w06_routes_to_existing_native_provider(condensed, permuted):
    case, layout, _, _ = case_module.build_case(oracle.witness(), condensed=condensed, permuted=permuted)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    source = emit_cpp_program(resolved.time, model_graph=graph)
    assert "prepare_local_nonlinear_problem<%d>" % (4 if condensed else 12) in source
    assert "finite_linear.hpp" in source
    assert "finite_linear_apply<8, 4>" in source
    assert "finite_linear_apply<4, 8>" in source
    assert ("finite_linear_solve<8, 8>" in source) == condensed
    assert "original_velocity" in source and "original_potential" in source


def test_finite_materialization_requires_exact_output_and_owned_captures():
    from pops.linalg import FiniteSupport
    case,_,_,_=case_module.build_case(oracle.witness())
    other,_,_,_=case_module.build_case(oracle.witness())
    program=case._time
    def state(p):
        return next(v for v in p._values if v.op=="state" and len(v.space.components)==8)
    own,foreign=state(program),state(other._time)
    support=FiniteSupport("velocity",tuple(own.space.components))
    before=program._ir_hash()
    with pytest.raises(ValueError,match="components"):
        FiniteSupport("velocity",tuple(reversed(support.dofs))).bind(own)
    with pytest.raises(ValueError,match="template"):
        support.bind(own).materialize(program,"bad",template=next(
            v for v in program._values if v.op=="state" and len(v.space.components)==4))
    with pytest.raises((TypeError,ValueError),match="different|owned|owner|program|Program"):
        support.bind(foreign).materialize(program,"foreign",template=own)
    assert program._ir_hash()==before


def test_native_materialization_votes_before_any_fab_access():
    from pops.codegen.program_emit_expressions import emit_pointwise_kernel
    case,_,_,_=case_module.build_case(oracle.witness(),condensed=True)
    value=next(v for v in case._time._values if v.name=="original_potential_residual")
    assert len(value.inputs[0].space.components)==8
    assert len(value.space.components)==4
    variables={item.id:"input%d"%i for i,item in enumerate(value.inputs)}
    source="\n".join(emit_pointwise_kernel(value,variables,"output",block_index=0,status="status"))
    assert source.index("all_reduce_max(finite_layout_error_") < source.index(".fab(li)")
    assert "long finite_layout_error_" in source
    template=variables[value.inputs[value.attrs["finite_template_index"]].id]
    assert "outA(index, 3) = %sA(index, 3);"%template in source
