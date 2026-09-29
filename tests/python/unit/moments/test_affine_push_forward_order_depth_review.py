"""Independent authoring-order counterexample above Python's recursion depth."""

from pops.moments import affine_push_forward


def test_complete_order_1000_basis_is_permutation_invariant_without_stack_ceiling():
    order = 1000
    ascending = tuple((degree,) for degree in range(order + 1))
    descending = ascending[::-1]
    moments = (1.,) + (0.,) * order
    forward = affine_push_forward(moments, indices=ascending,
                                  matrix=((1.,),), offset=(0.,))
    reverse = affine_push_forward(moments[::-1], indices=descending,
                                  matrix=((1.,),), offset=(0.,))
    assert len(forward) == len(reverse) == order + 1
    assert forward[0] is moments[0] and reverse[-1] is moments[0]
    assert forward == reverse[::-1] == moments
