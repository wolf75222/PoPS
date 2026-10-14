"""Source closure of overdetermined implicit data-domain counterexamples."""
import pops
import pytest
from examples.api_v02.implicit_domain_v14 import build_implicit_domain_v14

@pytest.mark.parametrize('mode',('compatible','incompatible','transpose'))
def test_overdetermined_domain_counterexample_resolves_without_ambient_extension(mode):
    spec=build_implicit_domain_v14(mode)
    plan=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    assert not plan.time._block_indices()
    assert 'second' in str(plan.time.to_graph().to_data())
