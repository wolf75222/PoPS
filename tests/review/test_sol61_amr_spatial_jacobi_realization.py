"""Explicit actual-operator Jacobi source contract and independent affine algebra."""
from fractions import Fraction
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("permutation", ((0, 1, 2), (2, 0, 1)))
def test_affine_zero_response_is_removed_without_changing_original_operator(permutation):
    matrix = [[Fraction(-3), Fraction(1, 3), Fraction(2, 7)],
              [Fraction(1, 5), Fraction(2), Fraction(-1, 8)],
              [Fraction(1, 9), Fraction(-2, 5), Fraction(1, 2)]]
    offset = [Fraction(7, 10), Fraction(-1), Fraction(1, 5)]
    matrix = [[matrix[i][j] for j in permutation] for i in permutation]
    offset = [offset[i] for i in permutation]
    def response(vector):
        return [sum(a*x for a, x in zip(row, vector))+b for row, b in zip(matrix, offset)]
    zero = response([Fraction(0)]*3)
    diagonal = []
    for i in range(3):
        basis = [Fraction(j == i) for j in range(3)]
        diagonal.append(response(basis)[i]-zero[i])
    assert diagonal == [matrix[i][i] for i in range(3)]
    assert diagonal != [response([Fraction(j == i) for j in range(3)])[i] for i in range(3)]
    # Signed finite inverse is legitimate for nonsymmetric right-GMRES;
    # no SPD or reaction/body identity is borrowed by this realization.
    residual = [Fraction(3, 10), Fraction(7, 10), Fraction(-1, 5)]
    direction = [r/d for r, d in zip(residual, diagonal)]
    assert [d*v for d, v in zip(diagonal, direction)] == residual


def test_realization_is_explicit_and_legacy_authority_bytes_are_unchanged():
    source = (ROOT / "include/pops/runtime/program/prepared_amr_field_residual.hpp").read_text()
    assert "AmrFieldRightPreconditioner preconditioner = AmrFieldRightPreconditioner::kIdentity" in source
    assert "unknown original AMR field right-preconditioner realization" in source
    assert "preconditioner_ == AmrFieldRightPreconditioner::kIdentity" in source
    assert '? identity : preconditioned_identity' in source
    assert '"pops.prepared-amr-original-field-residual@2"' in source
    assert "std::string(invocation_identity()), contract" in source
    assert re.search(r"if \(preconditioner_ != AmrFieldRightPreconditioner::kIdentity\)\s*"
                     r'exact.text\("pops.amr.original-spatial-jacobi.basis-response@1"\)', source)
    prepare = source.split("void prepare_spatial_jacobi_", 1)[1].split("template <class Operation>", 1)[0]
    assert "op_->apply_original_field_operator(candidate_, minus_);" in prepare
    assert "op_->apply_original_field_operator(candidate_, plus_);" in prepare
    assert "value(cell, component) - zero(cell, component)" in prepare
    assert "diagonal == Real(0)" in prepare and "!std::isfinite(reciprocal)" in prepare
    assert "active(cell, 0) < Real(.5)" in prepare
    assert "contains_local(global)" in prepare
    assert "require_authority(authority_, lane);" in prepare
    assert "local_phase_(lane, [&]" in prepare
    # Both numerical snapshots are produced by actual apply, never a guessed
    # spacing formula, recognized model name, R matrix or local source callback.
    assert "spacing(" not in prepare and "add_local" not in prepare


def test_legacy_identity_exact_contract_builder_sequence_is_preserved():
    path = "include/pops/runtime/program/prepared_amr_field_residual.hpp"
    old = subprocess.check_output(["git", "show", "619fec88665983843b00e33f52c136658e99befb:"+path],
                                  cwd=ROOT, text=True)
    current = (ROOT / path).read_text()
    def builder(text):
        return text.split("ExactContractBuilder exact;", 1)[1].split("contract = ", 1)[0]
    folded = builder(current).replace("invocation_identity()", "identity")
    folded = re.sub(r"if \(preconditioner_ != AmrFieldRightPreconditioner::kIdentity\)\s*"
                    r'exact.text\("pops.amr.original-spatial-jacobi.basis-response@1"\);', "", folded)
    assert re.sub(r"\s+", "", folded) == re.sub(r"\s+", "", builder(old))


def test_identity_and_preconditioned_profiles_share_the_same_original_equation():
    source = (ROOT / "tests/cpp/unit/elliptic/amr_original_field_residual.inc").read_text()
    assert "original_nonlinear_native_profile(AmrFieldRightPreconditioner::kIdentity, true);" in source
    assert "original_nonlinear_native_profile(AmrFieldRightPreconditioner::kSpatialBasisJacobi);" in source
    body = source.split("void original_nonlinear_native_profile", 1)[1].split(
        "TEST(CompositeGeneralField, OriginalAmrNonlinearResidual", 1)[0]
    assert "for (int cells : {16, 32})" in body
    assert "std::array<int, 3>{2, 0, 1}" in body
    assert "auto body = original_local_body(permutation);" in body
    assert "original_controls()," in body and "Real(1e-5), lane, realization" in body
    assert "original_controls().tolerance *" in body
    assert "original_field_dot(defect, defect)" in body
    assert "iteration_limit:newton=0:columns=240:" in body
    assert "EXPECT_THROW(prepared->candidate(authority, lane)" in body
    assert "realization" not in source.split("auto original_local_body", 1)[-1].split(
        "void original_nonlinear_native_profile", 1)[0]
