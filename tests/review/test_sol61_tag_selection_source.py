"""Source/math preparation; no native test is executed here."""
from pathlib import Path
import sys
import numpy as np
import pops
import pytest
from tests.python.support.tag_selection_case import build
from tests.review.sol61_tag_selection_oracle import physical_tags, periodic_dilation, receive


@pytest.mark.parametrize("shape", ((8, 8), (8, 12)))
@pytest.mark.parametrize("buffer", (0, 1))
@pytest.mark.parametrize("transfer", ("linear", "injection"))
def test_actual_public_case_source_admits(shape, buffer, transfer):
    case, layout = build(shape, buffer, transfer)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    assert resolved is not None
    assert Path(pops.__file__).resolve() == Path(__file__).resolve().parents[2]/"python/pops/__init__.py"
    assert "pops._pops" not in sys.modules
    assert not any(name.startswith("pops._native.dim") for name in sys.modules)


@pytest.mark.parametrize("shape", ((8, 8), (8, 12)))
def test_independent_wrap_dilation_and_full_refinement_counterexample(shape):
    tags = physical_tags(shape)
    assert np.flatnonzero(tags[0]).tolist() == [0, 7]
    for radius, columns in ((0,[0,7]), (1,[0,1,6,7])):
        assert np.flatnonzero(periodic_dilation(tags,radius)[0]).tolist() == columns
        boxes = [(1,(2*x,0),(2*x+1,2*shape[1]-1)) for x in columns]
        coarse,fine = receive(boxes,shape,radius)
        assert coarse.any() and fine.any()
        with pytest.raises(ValueError,match="authored periodic"):
            receive([(1,(0,0),(15,2*shape[1]-1))],shape,radius)
        # Buffer1 must not be silently treated as Buffer0.
        if radius:
            with pytest.raises(ValueError):
                receive([(1,(0,0),(1,2*shape[1]-1)),(1,(14,0),(15,2*shape[1]-1))],shape,radius)
