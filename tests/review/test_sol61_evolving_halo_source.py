"""Pure public Source authoring; no Native readiness inference."""
import sys
import pytest
import pops
from tests.python.support.evolving_accepted_halo_case import build

@pytest.mark.parametrize("subcycled",(False,True))
def test_public_evolving_halo_authoring(subcycled):
    case,layout=build(subcycled)
    resolved=pops.resolve(pops.validate(case),layout=layout)
    assert resolved is not None
    data=layout.runtime_layout_data()["execution"]
    assert data["accepted_halo"]["cells"] == (1,1)
    if subcycled:
        assert data["relations"][0]["temporal_ratio"] == {"numerator":5,"denominator":2}
        assert data["relations"][0]["remainder_policy"] == "explicit_final_substep"
    assert not any(n=="_pops" or n.endswith("._pops") for n in sys.modules)
