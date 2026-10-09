"""Inherited finite-volume contracts survive real symbolic-path resolution."""
import pops
import pytest

from tests.python.support.symbolic_path_case import make_case


@pytest.mark.parametrize("reverse", (False, True))
def test_symbolic_path_validation_preserves_its_authenticated_empty_sampling(reverse):
    case, layout = make_case(reverse=reverse)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    method = resolved.blocks[0].numerics.rates[0].method
    assert method.sampling == ()
    assert method.options()["sampling"] == ()
    assert method.to_data()["sampling"] == []
    # This is a physical single-state path, not a principal multi-state group.
    assert resolved.blocks[0].numerics.principal_groups == ()
