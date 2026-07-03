// ADC-595: host-side validation of the typed CouplingOperator conservation contract.
//
// include/pops/coupling/source/coupling_operator.hpp wraps the flat CoupledSourceProgram with a
// DECLARED conservation contract (which roles a coupling conserves versus creates). validate_coupling
// _contract raises (host, fail-loud) when the declaration is inconsistent with the actual output terms,
// so a coupling with a false "conserved" declaration cannot register. This test exercises that guard on
// the header directly (no runtime System needed): a role is conserved iff its output terms come in
// opposite-signed leg pairs (the shape add_pair / the named C++ couplings emit), CsOp::Neg == 5.

#include <gtest/gtest.h>

#include <pops/coupling/source/coupling_operator.hpp>

#include <string>
#include <vector>

using namespace pops;

namespace {

// Opcode values mirror pops::CsOp (coupled_source_program.hpp), frozen ABI.
constexpr int kPushReg = 0;
constexpr int kMul = 3;
constexpr int kNeg = 5;

// A conservative momentum exchange: two terms on "momentum_x", the second the exact Neg of the first
// (gain leg [PushReg] and loss leg [PushReg, Neg]) -- the add_pair shape.
CoupledSourceProgram conservative_exchange() {
  CoupledSourceProgram p;
  p.in_blocks = {"a", "b"};
  p.in_roles = {"momentum_x", "momentum_x"};
  p.out_blocks = {"a", "b"};
  p.out_roles = {"momentum_x", "momentum_x"};
  p.prog_ops = {kPushReg, /*loss leg:*/ kPushReg, kNeg};
  p.prog_args = {0, 0, 0};
  p.prog_lens = {1, 2};
  return p;
}

}  // namespace

TEST(CouplingOperatorContract, ConservedRolePairsAreAccepted) {
  CouplingOperator op;
  op.label = "collision";
  op.program = conservative_exchange();
  op.conservation.conserved_roles = {"momentum_x"};
  EXPECT_TRUE(coupling_role_terms_cancel(op.program, "momentum_x"));
  EXPECT_NO_THROW(validate_coupling_contract(op, "test"));
}

TEST(CouplingOperatorContract, ConservedRoleThatDoesNotCancelRaises) {
  CoupledSourceProgram p = conservative_exchange();
  // Break the pairing: make both legs identical (no Neg) so they no longer cancel.
  p.prog_ops = {kPushReg, kPushReg};
  p.prog_lens = {1, 1};
  CouplingOperator op;
  op.label = "broken";
  op.program = p;
  op.conservation.conserved_roles = {"momentum_x"};
  EXPECT_FALSE(coupling_role_terms_cancel(p, "momentum_x"));
  EXPECT_THROW(validate_coupling_contract(op, "test"), std::runtime_error);
}

TEST(CouplingOperatorContract, CreatedRoleMayNetSource) {
  // Ionization-like: three density terms that do NOT cancel (net source), declared CREATED -> legal.
  CoupledSourceProgram p;
  p.in_blocks = {"e", "g"};
  p.in_roles = {"density", "density"};
  p.consts = {1.7};
  p.out_blocks = {"g", "i", "e"};
  p.out_roles = {"density", "density", "density"};
  // Each term = k * ne * ng ; the neutral leg is Neg of that. Non-cancelling as a set (an e/i pair is
  // created), which is exactly why the role is declared CREATED, not conserved.
  p.prog_ops = {kPushReg, kPushReg, kMul, kPushReg, kMul, kNeg,  // g: -(k*ne*ng)
                kPushReg, kPushReg, kMul, kPushReg, kMul,        // i: +(k*ne*ng)
                kPushReg, kPushReg, kMul, kPushReg, kMul};       // e: +(k*ne*ng)
  p.prog_args = {2, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 2, 0, 0, 1, 0};
  p.prog_lens = {6, 5, 5};
  CouplingOperator op;
  op.label = "ionization";
  op.program = p;
  op.conservation.created_roles = {"density"};
  EXPECT_NO_THROW(validate_coupling_contract(op, "test"));
}

TEST(CouplingOperatorContract, RoleDeclaredBothConservedAndCreatedRaises) {
  CouplingOperator op;
  op.label = "contradiction";
  op.program = conservative_exchange();
  op.conservation.conserved_roles = {"momentum_x"};
  op.conservation.created_roles = {"momentum_x"};
  EXPECT_THROW(validate_coupling_contract(op, "test"), std::runtime_error);
}

TEST(CouplingOperatorContract, DeclaredRoleWithNoTermRaises) {
  CouplingOperator op;
  op.label = "stale";
  op.program = conservative_exchange();  // targets only momentum_x
  op.conservation.conserved_roles = {"energy"};  // no term targets energy
  EXPECT_THROW(validate_coupling_contract(op, "test"), std::runtime_error);
}

TEST(CouplingOperatorContract, UncheckedContractIsANoOp) {
  // A raw coupling declaring nothing (empty contract) keeps its historical unchecked behavior: even a
  // non-cancelling program passes validation because nothing was declared.
  CoupledSourceProgram p = conservative_exchange();
  p.prog_ops = {kPushReg, kPushReg};  // no cancellation
  p.prog_lens = {1, 1};
  CouplingOperator op;
  op.label = "raw";
  op.program = p;
  EXPECT_TRUE(op.conservation.unchecked());
  EXPECT_NO_THROW(validate_coupling_contract(op, "test"));
}
