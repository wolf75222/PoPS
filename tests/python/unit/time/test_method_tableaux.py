"""ADC-663: exact RK/ARK authorities and proof-carrying method properties."""
from fractions import Fraction

from pops.lib.time.rk import RK4_TABLEAU, SSPRK2_TABLEAU
from pops.lib.time.ssprk import SSPRK3_TABLEAU
from pops.time._methods.tableau import AdditiveRungeKuttaTableau, RungeKuttaTableau


def test_classical_tableaux_have_exact_certified_properties():
    assert RK4_TABLEAU.properties.order == 4
    assert RK4_TABLEAU.properties.stability_polynomial == (
        1, 1, Fraction(1, 2), Fraction(1, 6), Fraction(1, 24))
    assert RK4_TABLEAU.properties.flux_weights == (
        Fraction(1, 6), Fraction(1, 3), Fraction(1, 3), Fraction(1, 6))
    assert RK4_TABLEAU.properties.ssp is None

    assert SSPRK2_TABLEAU.properties.order == 2
    assert SSPRK2_TABLEAU.properties.ssp.coefficient == 1
    assert SSPRK3_TABLEAU.properties.order == 3
    assert SSPRK3_TABLEAU.properties.abscissae == (0, 1, Fraction(1, 2))
    assert SSPRK3_TABLEAU.properties.ssp.coefficient == 1

    renamed = RungeKuttaTableau(
        SSPRK2_TABLEAU.A, SSPRK2_TABLEAU.b, SSPRK2_TABLEAU.c, name="manual-heun")
    impostor = RungeKuttaTableau(A=[[]], b=[1], c=[0], name="ssprk2")
    assert renamed.properties == SSPRK2_TABLEAU.properties
    assert renamed.certificate == SSPRK2_TABLEAU.certificate
    assert impostor.properties.ssp is None


def test_lower_order_is_proved_from_exact_conditions():
    method = RungeKuttaTableau(A=[[], [Fraction(1, 3)]], b=[0, 1], c=[0, Fraction(1, 3)])
    assert method.properties.order == 1


def test_ark_partition_abscissae_are_exact_and_immutable():
    explicit = RungeKuttaTableau(A=[[], [1]], b=[Fraction(1, 2)] * 2, name="heun")
    ark = AdditiveRungeKuttaTableau(
        explicit,
        implicit_A=[[Fraction(1, 2)], [0, Fraction(1, 2)]],
        implicit_b=[Fraction(1, 2)] * 2,
        name="partitioned",
    )
    assert ark.abscissae == ((0, Fraction(1, 2)), (1, Fraction(1, 2)))
    assert ark.properties.order == 2
    assert dict(ark.properties.flux_weights) == {
        "explicit": (Fraction(1, 2), Fraction(1, 2)),
        "implicit": (Fraction(1, 2), Fraction(1, 2)),
    }
    try:
        ark.name = "changed"
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("ARK tableau must be immutable")
