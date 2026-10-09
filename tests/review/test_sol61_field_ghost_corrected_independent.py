"""Independent composite support probe; synthetic data, no Native receipt."""
import numpy as np
import pytest
from tests.review.test_sol61_initial_field_ghost_native_preparation import fields
from tests.python.support.initial_field_ghost_native_oracle import original_residual

def test_covered_parent_values_are_restricted_but_interface_fine_defect_rejected():
    values=fields(0);coarse,ca=values[0];fine,fa=values[1]
    coarse[~ca]=12345.0
    state=[np.full(a.shape,2.) for a,_ in values]
    assert original_residual(values,state)==[0.,0.]
    # Perturb an active fine interface cell, well inside the phi absolute bound;
    # an original composite operator must see the interface/neighbor gradients.
    fine[1,1]+=9e-12
    assert abs(fine[1,1]-2)<1e-10
    with pytest.raises(AssertionError):original_residual(values,state)
