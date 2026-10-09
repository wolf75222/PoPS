from fractions import Fraction
import pytest
from pops.moments.atomic_cubature import AtomicCubature

NODES = ((0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1))
INDICES = ((1,1),(0,0),(0,2),(1,0),(2,0),(0,1))


def test_non_gaussian_six_atom_flux_and_raw_straight_path():
    declaration = AtomicCubature(indices=INDICES, nodes=NODES)
    weights = (1.,2.,3.,4.,5.,6.)
    moments = tuple(sum(w*x**i*y**j for w,(x,y) in zip(weights,NODES)) for i,j in INDICES)
    assert declaration.weights(moments) == weights
    for i,j in INDICES:
        for axis in range(2):
            requested = (i+(axis == 0), j+(axis == 1))
            expected = sum(w*x**requested[0]*y**requested[1] for w,(x,y) in zip(weights,NODES))
            assert declaration.moment(moments, requested) == expected
    # Independent exact integration of rho(s)*dU/ds on the stated RAW straight path.
    left, right, direction = Fraction(3), Fraction(5), Fraction(-7,2)
    integral = direction * (left+right)/2
    simpson = direction * (left + 4*(left+right)/2 + right)/6
    assert integral == simpson
    velocities = tuple(Fraction(2)*x-Fraction(3)*y for x,y in NODES)
    coupling = Fraction(2)-Fraction(-3,2)
    bound = max(abs(v) for v in velocities)+abs(coupling)*max(left,right)
    for s in (Fraction(0), Fraction(1,3), Fraction(1)):
        rho = left+s*(right-left)
        assert all(abs(v+coupling*rho) <= bound for v in velocities)


def test_other_unisolvent_nodes_and_singular_admission():
    c = AtomicCubature(indices=((2,),(0,),(1,)), nodes=((-2,),(0,),(3,)))
    m = (Fraction(31),Fraction(6),Fraction(7))
    assert c.moment(m,(3,)) == 73
    with pytest.raises(ValueError, match='unisolvent'):
        AtomicCubature(indices=((0,),(1,)), nodes=((1,),(1,)))
    with pytest.raises(ValueError):
        AtomicCubature(indices=((False,),), nodes=((0,),))
