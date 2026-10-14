"""Array implicit derivatives retain complete equations and native column receipts."""
import pytest
import pops
from examples.api_v02.implicit_array_v14 import build_implicit_array_v14

@pytest.mark.parametrize('transpose', (False, True))
def test_dynamic_array_implicit_graph_resolves(transpose):
    spec = build_implicit_array_v14(3, transpose=transpose)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert plan.time is not None
    assert not plan.time._block_indices() and not spec.initial_values


@pytest.mark.parametrize('transpose',(False,True))
def test_distributed_array_implicit_graph_resolves(transpose):
    spec=build_implicit_array_v14(3,transpose=transpose,distributed_ranks=3)
    plan=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    assert not plan.time._block_indices() and not spec.initial_values


@pytest.mark.parametrize('transpose',(False,True))
def test_overdetermined_redundant_array_relation_preserves_every_equation(transpose):
    spec=build_implicit_array_v14(3,transpose=transpose,redundant=True)
    plan=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    assert not plan.time._block_indices()
    assert 'redundant' in str(plan.time.to_graph().to_data())
