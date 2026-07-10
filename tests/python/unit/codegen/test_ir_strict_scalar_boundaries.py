"""ADC-652: IR scalar and identity boundaries never stringify or round implicitly."""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from pops.ir import Const, Divergence, EigWitness, RateExpr, RuntimeParamRef, ScalarLiteral, Var
from pops.ir.lowering import diff
from pops.ir.values import set_runtime_param_indices
from pops.model import Handle, OwnerPath


class _Stringable:
    def __str__(self):
        return "coerced"


class _LiteralHook:
    def __init__(self, kind):
        self.kind = kind

    def __pops_scalar_literal__(self):
        return {"kind": self.kind, "payload": 1, "cpp": "pops::Real(1)"}


def test_scalar_literal_hook_rejects_stringable_kind_objects():
    with pytest.raises(TypeError, match="kind must be a non-empty string"):
        ScalarLiteral.from_value(_LiteralHook(_Stringable()))


def test_scalar_literal_target_and_algebraic_spellings_are_strict_strings():
    with pytest.raises(TypeError, match="scalar target"):
        ScalarLiteral.from_value(1, target=_Stringable())
    with pytest.raises(TypeError, match="string symbolic"):
        ScalarLiteral.algebraic(_Stringable(), cpp="pops::Real(1)")
    with pytest.raises(TypeError, match="string symbolic"):
        ScalarLiteral.algebraic("sqrt(2)", cpp=_Stringable())


def test_diff_rejects_implicit_variable_stringification():
    with pytest.raises(TypeError, match="Var, declaration Handle"):
        diff(Var("u", "cons"), _Stringable())


@pytest.mark.parametrize("tol", [Fraction(1, 3), Decimal("1e-30")])
def test_eig_witness_retains_exact_tolerance_until_cpp_lowering(tol):
    witness = EigWitness([[Const(1)]], "all_real", im_tol=tol)

    assert witness.im_tol == tol
    assert type(witness.im_tol) is type(tol)
    emitted = witness._extra_args_cpp()[0]
    if isinstance(tol, Fraction):
        assert emitted == "(pops::Real(1) / pops::Real(3))"
    else:
        assert emitted == "pops::Real(1E-30)"


@pytest.mark.parametrize("tol", [True, float("nan"), float("inf"), 0, -1])
def test_eig_witness_rejects_invalid_tolerance(tol):
    with pytest.raises((TypeError, ValueError)):
        EigWitness([[Const(1)]], "all_real", im_tol=tol)


def test_runtime_parameter_identity_and_index_are_never_coerced():
    with pytest.raises(TypeError, match="non-empty string"):
        RuntimeParamRef(_Stringable(), 1)
    with pytest.raises(TypeError, match="parameter names"):
        set_runtime_param_indices({_Stringable(): 0})
    with pytest.raises(TypeError, match="non-negative Python ints"):
        set_runtime_param_indices({"alpha": True})

    param = RuntimeParamRef("alpha", Fraction(1, 3))
    set_runtime_param_indices({"alpha": 2})
    assert param.to_cpp() == "params.get(2)"
    assert param.value == Fraction(1, 3)


def test_rate_expr_requires_matching_handle_payload_and_finite_exact_sign():
    owner = OwnerPath("model", "transport")
    flux = Handle("F", kind="flux", owner=owner)
    source = Handle("S", kind="source", owner=owner)

    exact = RateExpr([("flux", flux, Fraction(1, 3)),
                      ("source", source, Decimal("0.125"))])
    assert exact.terms[0][2] == Fraction(1, 3)
    assert exact.terms[1][2] == Decimal("0.125")
    with pytest.raises(TypeError, match="matching declaration Handle"):
        RateExpr([("flux", object(), 1)])
    for sign in (True, float("nan"), float("inf")):
        with pytest.raises((TypeError, ValueError)):
            RateExpr([("flux", flux, sign)])


def test_exact_board_repr_never_formats_rationals_through_float():
    flux = Handle("F", kind="flux", owner=OwnerPath("model", "transport"))
    assert "1/3" in repr(Divergence(flux, Fraction(1, 3)))
