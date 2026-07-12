import dataclasses
from fractions import Fraction

import pytest

from pops.time import (
    AlgebraicTerm, ApproximateLinearization, AutomaticJVP, ConsistentInitialization, Dt, EquationSpace,
    ExactJacobian, IdentityTerm, Index1DAE, PostStepValidation, PreconditionerContract,
    PreconditionerDomain, ResidualOperator, SupportReport, SupportStatus, UnknownSpace,
    linearization_fidelity,
)


def _dae(**changes):
    unknown_space = UnknownSpace("Y", ("case::u", "case::lambda"))
    equation_space = EquationSpace("F", ("case::evolution", "case::constraint"))
    args = dict(
        name="semi_explicit",
        unknown_space=unknown_space,
        equation_space=equation_space,
        terms=(IdentityTerm("case::evolution", "case::u"),
               AlgebraicTerm("case::constraint", "constraint", ("case::u", "case::lambda"))),
        dt=Dt(), linearization=ExactJacobian("jacobian", unknown_space, equation_space),
        dae=Index1DAE(("case::u",), ("case::lambda",)),
        consistent_initialization=ConsistentInitialization("newton", 1e-10, 20),
        post_step=PostStepValidation(1e-9),
    )
    args.update(changes)
    return ResidualOperator(**args)


def test_descriptor_is_immutable_canonical_and_valid():
    residual = _dae()
    assert residual.validate()
    assert residual.report().facts["dae_index"] == 1
    assert residual.report().facts["n_identity_terms"] == 1
    assert residual.report().facts["n_algebraic_terms"] == 1
    assert residual.report().facts["n_dt"] == 1
    assert residual.report().facts["n_fields"] == 0
    assert residual.to_data()["unknown_space"]["kind"] == "unknown_space"
    assert residual.to_data()["linearization"]["kind"] == "exact_jacobian"
    with pytest.raises(dataclasses.FrozenInstanceError):
        residual.name = "changed"


def test_spaces_require_qualified_unique_components():
    with pytest.raises(ValueError, match="qualified ids"):
        UnknownSpace("Y", ("u",))
    with pytest.raises(ValueError, match="duplicate"):
        EquationSpace("F", ("case::f", "case::f"))


def test_validation_is_fail_closed_for_bad_references_and_dae_policies():
    residual = _dae(terms=(IdentityTerm("case::missing", "case::u"),),
                    consistent_initialization=None)
    report = residual.report()
    assert not report.valid
    assert any("outside equation_space" in error for error in report.errors)
    assert any("consistent-initialization" in error for error in report.errors)
    with pytest.raises(ValueError, match="invalid residual operator"):
        residual.validate()


def test_validation_requires_terms_to_cover_whole_product_space():
    residual = _dae(terms=(IdentityTerm("case::evolution", "case::u"),),
                    dae=None, consistent_initialization=None,
                    post_step=PostStepValidation(1e-9))
    report = residual.report()
    assert not report.valid
    assert any("cover every equation component" in error for error in report.errors)
    assert any("cover every unknown component" in error for error in report.errors)


def test_linearization_fidelity_is_not_conflated():
    unknowns = UnknownSpace("Y", ("case::u",))
    equations = EquationSpace("F", ("case::f",))
    assert linearization_fidelity(ExactJacobian("J", unknowns, equations)).value == "exact"
    assert linearization_fidelity(AutomaticJVP("autodiff")).value == "automatic"
    with pytest.raises(ValueError, match="cannot claim exact fidelity"):
        ApproximateLinearization("J0", "exact", unknowns, equations)


def test_partial_jacobian_domain_cannot_validate():
    partial = UnknownSpace("partial", ("case::u",))
    equations = EquationSpace("F", ("case::evolution", "case::constraint"))
    residual = _dae(linearization=ExactJacobian("J", partial, equations))
    assert not residual.report().valid
    with pytest.raises(ValueError, match="cover unknown_space exactly"):
        residual.validate()


def test_preconditioner_requires_exact_direction_and_block_order():
    residual = _dae()
    reversed_equations = EquationSpace(
        "F-reversed", tuple(reversed(residual.equation_space.components)))
    contract = PreconditionerContract(
        "P", PreconditionerDomain(reversed_equations, residual.unknown_space))
    incompatible = _dae(preconditioner=contract)
    with pytest.raises(ValueError, match="equation_space exactly and in block order"):
        incompatible.validate()

    good = PreconditionerContract(
        "P", PreconditionerDomain(residual.equation_space, residual.unknown_space))
    assert good.validate_for(residual)


def test_exact_scalar_domains_are_preserved():
    term = IdentityTerm("case::f", "case::u", Fraction(1, 3))
    assert term.coefficient.to_python() == Fraction(1, 3)
    init = ConsistentInitialization("newton", Fraction(1, 1000), 4)
    assert init.tolerance.to_python() == Fraction(1, 1000)


def test_support_never_fabricates_backend_coverage():
    residual = _dae()
    report = residual.support()
    assert not report.supported
    assert report.status is SupportStatus.UNKNOWN
    assert report.missing == ("residual_backend",)

    class BadBackend:
        name = "bad"

        def residual_support(self, operator):
            return True

    report = residual.support(BadBackend())
    assert not report.supported
    assert "structured SupportReport" in report.reasons[0]

    class GoodBackend:
        name = "test-only"

        def residual_support(self, operator):
            return SupportReport(SupportStatus.AVAILABLE, "test-only")

    assert residual.support(GoodBackend()).supported
