"""Independent rational partition and saved-active-array Source oracle probes."""
from fractions import Fraction
import numpy as np
import pytest
from tests.python.support.accepted_halo_substep_oracle import check,bound,DT

def partition(ratio):
    nominal=1/ratio
    durations=[nominal for _ in range(ratio.numerator//ratio.denominator)]
    remainder=1-sum(durations,Fraction())
    if remainder:durations.append(remainder)
    return durations

@pytest.mark.parametrize('level',(0,1))
@pytest.mark.parametrize('steps',(1,2))
def test_fraction_reference_signed_active_cells_and_floating_associations(level,steps):
    durations=[Fraction(1)] if level==0 else partition(Fraction(5,2))
    assert sum(durations)==1
    if level:assert durations==[Fraction(2,5),Fraction(2,5),Fraction(1,5)]
    exact=Fraction(1)
    for _ in range(steps):
        for fraction in durations:exact*=1+Fraction(1,64)*fraction
    initial=np.array([[[1.,1.],[1.,1.]],[[.9,-.9],[.125,-.125]]])
    mask=np.array([[True,True],[False,False]])
    current=initial.copy()
    for _ in range(steps):
        for fraction in durations:current[1]+=float(Fraction(1,64)*fraction)*current[1]
    # The exact rational result is independent of the author's FP factor expression.
    assert np.all(np.abs(current[1][mask]-initial[1][mask]*float(exact))<=bound(initial[1][mask],steps,level))
    check(initial,current,mask,steps,level)

@pytest.mark.parametrize('second_schedule',((DT,),(DT/2,DT/2),(2*DT/5,2*DT/5)))
def test_correct_first_macro_does_not_hide_wrong_second_macro(second_schedule):
    initial=np.array([[[1.]],[[.9]]]);current=initial.copy();mask=np.ones((1,1),bool)
    for duration in [float(Fraction(1,64)*p) for p in partition(Fraction(5,2))]:current[1]+=duration*current[1]
    for duration in second_schedule:current[1]+=duration*current[1]
    with pytest.raises(AssertionError):check(initial,current,mask,2,1)

def test_covered_coarse_cells_are_deliberately_outside_temporal_oracle():
    initial=np.array([[[1.,1.]],[[.9,.9]]]);current=initial.copy();mask=np.array([[True,False]])
    current[1,0,0]+=DT*current[1,0,0];current[1,0,1]=12345.
    check(initial,current,mask,1,0)
