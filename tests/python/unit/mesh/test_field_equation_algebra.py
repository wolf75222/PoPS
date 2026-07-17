"""Spec 5 (sec.9.2, ADC-491): the board-node elliptic-operator algebra.

The field-equation forms Spec 5 itself uses must be AUTHORABLE as inert, inspectable IR:

  -laplacian(phi) + k*phi == rhs            (screened Poisson)
  -div(eps*grad(phi)) + kappa*phi == rhs    (anisotropic / variable coefficient, sec.9.2)

These build elliptic operator terms (Reaction / CoeffGradient / DivCoeffGrad / EllipticSum);
nothing computes in Python -- the C++ elliptic solver executes. This test covers the IR
construction and the principal-kind inspection. Physical field validation belongs to
``FieldOperator`` contract tests; the retired Poisson-problem wrappers are not recreated here.
"""

import pytest

pytest.importorskip("pops")

from pops.math import (  # noqa: E402
    laplacian, grad, div, unknown, principal_kinds,
    Reaction, CoeffGradient, DivCoeffGrad, EllipticSum, Laplacian, Divergence, Equation)
from pops.math import RateTerm, Var  # noqa: E402
from pops.fields.coefficients import ScalarCoefficient, ReactionCoefficient  # noqa: E402
from pops.model import Handle, OwnerPath  # noqa: E402


def _coefficient_field(name):
    return Handle(name, kind="field", owner=OwnerPath.shared("mesh.field_equation"))


def test_reaction_term_constructs():
    phi = unknown("phi")
    react = 0.5 * phi
    assert isinstance(react, Reaction)
    assert principal_kinds(react) == {"reaction"}
    # coefficient (not just a scalar) is accepted
    react2 = ReactionCoefficient(_coefficient_field("kappa")) * phi
    assert isinstance(react2, Reaction)


def test_coeff_gradient_and_div():
    phi = unknown("phi")
    eps = ScalarCoefficient(_coefficient_field("eps"))
    cg = eps * grad(phi)
    assert isinstance(cg, CoeffGradient)
    op = div(cg)
    assert isinstance(op, DivCoeffGrad)
    assert principal_kinds(op) == {"div_coeff_grad"}


def test_div_grad_is_laplacian():
    phi = unknown("phi")
    op = div(grad(phi))  # no coefficient -> the constant-coefficient Laplacian
    assert isinstance(op, Laplacian)
    assert principal_kinds(op) == {"laplacian"}


def test_div_of_a_flux_stays_hyperbolic():
    # Regression: div of a model flux handle must still build a hyperbolic flux term.
    op = div("default_flux")
    assert isinstance(op, Divergence)
    assert isinstance(op, RateTerm)  # composes into a rate equation, not an elliptic sum


def test_screened_form_constructs_and_inspects():
    phi = unknown("phi")
    rhs = Var("charge", "cons")
    lhs = -laplacian(phi) + 0.5 * phi
    assert isinstance(lhs, EllipticSum)
    assert principal_kinds(lhs) == {"laplacian", "reaction"}
    eq = (lhs == rhs)
    assert isinstance(eq, Equation)
    assert "EllipticSum" in repr(lhs) and "Reaction" in repr(lhs)


def test_spec_9_2_headline_form_constructs():
    # -div(eps*grad(phi)) + kappa*phi == charge  (Spec 5 sec.9.2 headline)
    phi = unknown("phi")
    eps = ScalarCoefficient(_coefficient_field("eps"))
    kappa = ReactionCoefficient(_coefficient_field("kappa"))
    charge = Var("charge", "cons")
    eq = (-div(eps * grad(phi)) + kappa * phi == charge)
    assert isinstance(eq, Equation)
    assert principal_kinds(eq.lhs) == {"div_coeff_grad", "reaction"}


def test_terms_are_inert_no_runtime_data():
    # The elliptic nodes carry references/coefficients, not arrays; they compute nothing.
    phi = unknown("phi")
    react = ScalarCoefficient(_coefficient_field("eps")) * phi
    with pytest.raises(NotImplementedError):
        react.eval({})                           # generic Expr protocol, no host implementation
    assert react.field is phi                   # a reference, not a value
    assert react.coeff.name == "eps"            # the coefficient descriptor, not an array
    assert "Reaction" in repr(react)            # inspectable
