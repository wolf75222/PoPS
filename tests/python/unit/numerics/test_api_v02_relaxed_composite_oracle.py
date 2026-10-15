import numpy as np
from tests.python.support import relaxed_composite_oracle as reference


def test_composite_rrk_recovers_rotation_and_requires_shared_factor_with_subcycles():
    for ratio in (1, 2):
        old, provisional, boxes = reference.synthetic_rotation(ratio)
        gamma = reference.relaxation(old, provisional, boxes, cells=(4, 4))
        relaxed = [a+gamma*(b-a) for a, b in zip(old, provisional, strict=True)]
        assert abs(reference.pair(relaxed, relaxed, boxes, cells=(4, 4))-1) < 5.e-16
        if ratio == 1:
            assert gamma == .8
            assert np.max(abs(relaxed[0][0]-.6)) < 1.e-15
            assert np.max(abs(relaxed[0][1]-.8)) < 1.e-15
        else:
            coarse = -.5; fine = .515625-1
            expected = -(coarse+fine)/(.5*(coarse*coarse+1+fine*fine+.875**2))
            assert gamma == expected
            assert .8 < gamma < .9
            # A raw sum over cells overweights the refined half by four.
            wrong = -2*(coarse+4*fine)/(coarse*coarse+1+4*(fine*fine+.875**2))
            assert abs(wrong-gamma) > .01
