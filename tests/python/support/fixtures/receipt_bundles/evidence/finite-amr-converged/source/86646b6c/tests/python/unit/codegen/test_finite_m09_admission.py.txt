"""Public source admission, explicitly not a native co-location-vote test."""
import pops
import pytest
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes, LayoutPlanBuilder
from tests.python.support.finite_m09_case import case_module, oracle


@pytest.mark.parametrize("condensed",(False,True))
def test_two_seeds_preserve_the_original_equation_and_frozen_captures(condensed):
    tokens=[]
    seeds=[]
    for scale in (0,.375):
        case,_,_,_=case_module.build_case(oracle.witness(),condensed=condensed,seed_scale=scale)
        token=next(v for v in case._time._values if v.op=="solve_coupled_implicit")
        tokens.append(token)
        seeds.append(next(v for v in case._time._values if v.name=="p_seed"))
    assert seeds[0].attrs!=seeds[1].attrs
    assert tokens[0].attrs["expression_nodes"]==tokens[1].attrs["expression_nodes"]
    assert tokens[0].attrs["expressions"]==tokens[1].attrs["expressions"]
    assert tokens[0].attrs["product_reads"]==tokens[1].attrs["product_reads"]


def test_distinct_spatial_batches_cannot_silently_supply_finite_map_operands():
    case,layout,_,_=case_module.build_case(oracle.witness())
    validated=pops.validate(case)
    subjects=validated.layout_subjects()
    builder=LayoutPlanBuilder(validated.owner_path.canonical())
    one=builder.layout("one",layout)
    other=Uniform(CartesianGrid(frame=layout.mesh.frame,cells=(2,1),
                               periodic=PeriodicAxes(layout.mesh.frame.axes)))
    two=builder.layout("two",other)
    for block in subjects.blocks:
        builder.assign_block(block,one if block.local_id=="velocity" else two)
    for state in subjects.states:
        builder.assign_state(state,one if state.block_ref.local_id=="velocity" else two)
    plan=builder.resolve(**subjects.to_dict())
    with pytest.raises(ValueError,match="Program read requires an explicit layout mapping"):
        pops.resolve(validated,layout=plan,layout_providers={one:layout,two:other})
