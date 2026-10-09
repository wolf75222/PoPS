"""Public authoring validation on the retained PoPS API; no JIT in these tests."""

from fractions import Fraction

import pytest
import pops

from pops_witnesses.author import author_case, literal_inputs


@pytest.mark.parametrize(
    "epsilon,beta", ((0, 0), (Fraction(1, 20), 0), (Fraction(1, 20), Fraction(1, 20)))
)
def test_public_composition_validates_and_resolves(epsilon, beta):
    authored = author_case(epsilon=epsilon, beta=beta)
    resolved = pops.resolve(pops.validate(authored.case), layout=authored.layout)
    assert resolved.resolved_dimension == 2
    assert len(authored.rhs) == 3
    assert len(authored.observed_fields) == 3
    assert literal_inputs()["receiver"].shape == (1, 12, 16)


@pytest.mark.parametrize("kwargs", ({"epsilon": -1}, {"beta": float("nan")}, {"beta": 1}))
def test_witness_parameters_are_bounded(kwargs):
    with pytest.raises(ValueError, match="must be finite"):
        author_case(**kwargs)
