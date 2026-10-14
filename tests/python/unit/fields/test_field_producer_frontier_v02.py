"""An observation owns its current solve, not every solve in its State ancestry."""
import pytest
import pops
from examples.api_v02.models_M13 import build_m13
from pops.fields._observation_contract import (validate_field_observation,
                                               direct_native_field_operations)


class Changed:
    """A deliberately corrupted loaded-IR view for validation countertests."""
    def __init__(self, original, **changes):
        self.original=original
        self.__dict__.update(changes)
    def __getattr__(self,name):
        return getattr(self.original,name)


@pytest.fixture(scope="module")
def coupled_program():
    spec=build_m13(cells=(8,8))
    return pops.resolve(pops.validate(spec.case),layout=spec.layout)


def test_two_field_problems_compose_across_both_runge_kutta_stages(coupled_program):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    observations=[v for v in coupled_program.time._values if v.op=="field_component"]
    assert len(observations)==4
    assert len({v.attrs["field_problem_identity"] for v in observations})==2
    for value in observations:
        validate_field_observation(value)
    cpp=emit_cpp_program(coupled_program.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(coupled_program.blocks))
    assert cpp.count("PreparedStencilInput<")==2
    assert cpp.count("stage_exchange_batch")==2


@pytest.mark.parametrize("producer", ("load","apply","coefficients"))
def test_current_producer_cannot_borrow_an_ancestor_field_identity(coupled_program,producer):
    observations=[v for v in coupled_program.time._values if v.op=="field_component"]
    observation=observations[2]
    foreign=observations[1].attrs["field_problem_identity"]
    _width,_selected,solve=validate_field_observation(observation)
    load,apply,coefficients=direct_native_field_operations(solve)
    target={"load":load,"apply":apply,"coefficients":coefficients}[producer]
    altered=Changed(target,attrs={**target.attrs,"field_problem_identity":foreign})
    if producer=="load":load=altered
    elif producer=="coefficients":
        apply=Changed(apply,inputs=(*apply.inputs[:2],altered,*apply.inputs[3:]))
    else:apply=altered
    operator=solve.inputs[0]
    body=tuple(apply if node.op=="field_problem_apply" else node for node in operator.attrs["apply_block"])
    operator=Changed(operator,attrs={**operator.attrs,"apply_block":body})
    solve=Changed(solve,inputs=(operator,load,*solve.inputs[2:]))
    consumed=observation.inputs[0]
    outcome=consumed.inputs[0]
    outcome=Changed(outcome,inputs=(solve,*outcome.inputs[1:]))
    consumed=Changed(consumed,inputs=(outcome,*consumed.inputs[1:]))
    changed=Changed(observation,inputs=(consumed,))
    with pytest.raises(ValueError,match="different native field problem"):
        validate_field_observation(changed)
