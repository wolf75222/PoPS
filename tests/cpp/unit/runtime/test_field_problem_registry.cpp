// ADC-596: the native FieldProblemRegistry unifying the default and named field problems. The
// registry is a host-only DESCRIPTOR layer (it owns no solver, changes no numerics); this suite
// pins register/find/at, the named-field-from-components adapter, and the early
// solver x layout x boundary x output validation that must refuse a bad combination BEFORE bind.
#include <gtest/gtest.h>

#include <string>

#include <pops/runtime/system/field_problem_registry.hpp>

namespace {

TEST(FieldProblemRegistry, RegisterFindAtAndStableIds) {
  pops::FieldProblemRegistry reg;
  const int phi = reg.register_problem(pops::default_poisson_entry());
  const int psi = reg.register_problem(
      pops::named_field_entry("psi", /*phi=*/pops::kAuxNamedBase, /*gx=*/-1, /*gy=*/-1));
  EXPECT_EQ(phi, 0);
  EXPECT_EQ(psi, 1);
  EXPECT_EQ(reg.size(), 2);
  EXPECT_EQ(reg.find("phi"), 0);
  EXPECT_EQ(reg.find("psi"), 1);
  EXPECT_EQ(reg.find("absent"), -1);
  EXPECT_EQ(reg.at(0).id, "phi");
  EXPECT_EQ(reg.at(1).equation, pops::EquationKind::Poisson);
  EXPECT_THROW(reg.at(2), std::out_of_range);
}

TEST(FieldProblemRegistry, ReRegisterKeepsIdAndReplaces) {
  pops::FieldProblemRegistry reg;
  const int a = reg.register_problem(pops::default_poisson_entry());
  pops::FieldProblemEntry replaced = pops::default_poisson_entry();
  replaced.solver = pops::EllipticSolverKind::FFT;
  const int b = reg.register_problem(replaced);
  EXPECT_EQ(a, b) << "same id string keeps the same integer id";
  EXPECT_EQ(reg.size(), 1) << "re-register replaces in place, no duplicate";
  EXPECT_EQ(reg.at(a).solver, pops::EllipticSolverKind::FFT);
}

TEST(FieldProblemRegistry, EmptyIdRejected) {
  pops::FieldProblemRegistry reg;
  pops::FieldProblemEntry bad;
  bad.id = "";
  EXPECT_THROW(reg.register_problem(bad), std::invalid_argument);
}

TEST(FieldProblemRegistry, NamedFieldEntryDerivesLayoutFromComponents) {
  const pops::FieldProblemEntry e = pops::named_field_entry(
      "psi", /*phi=*/pops::kAuxNamedBase, /*gx=*/pops::kAuxNamedBase + 1,
      /*gy=*/pops::kAuxNamedBase + 2);
  EXPECT_EQ(e.layout.component_of("psi"), pops::kAuxNamedBase);
  EXPECT_EQ(e.layout.component_of("psi_grad_x"), pops::kAuxNamedBase + 1);
  EXPECT_EQ(e.layout.component_of("psi_grad_y"), pops::kAuxNamedBase + 2);
  // A phi-only named field (no gradient slots) declares just the potential channel.
  const pops::FieldProblemEntry phi_only =
      pops::named_field_entry("chi", pops::kAuxNamedBase, -1, -1);
  EXPECT_EQ(phi_only.layout.channels().size(), 1u);
}

TEST(FieldProblemRegistry, ValidateAcceptsDefaultOnBothRoutes) {
  pops::FieldProblemRegistry reg;
  reg.register_problem(pops::default_poisson_entry());  // GeometricMG, periodic
  EXPECT_NO_THROW(reg.validate(0, pops::LayoutRoute::Uniform));
  EXPECT_NO_THROW(reg.validate(0, pops::LayoutRoute::Amr));
}

TEST(FieldProblemRegistry, ValidateRefusesFftOnAmr) {
  pops::FieldProblemRegistry reg;
  pops::FieldProblemEntry fft = pops::default_poisson_entry();
  fft.solver = pops::EllipticSolverKind::FFT;
  const int id = reg.register_problem(fft);
  EXPECT_NO_THROW(reg.validate(id, pops::LayoutRoute::Uniform));
  try {
    reg.validate(id, pops::LayoutRoute::Amr);
    FAIL() << "FFT on AMR must be refused";
  } catch (const std::invalid_argument& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("FFT"), std::string::npos);
    EXPECT_NE(msg.find("AMR"), std::string::npos);
    EXPECT_NE(msg.find("GeometricMG"), std::string::npos) << "names the alternative";
  }
}

TEST(FieldProblemRegistry, ValidateRefusesFftWithDirichlet) {
  pops::FieldProblemRegistry reg;
  pops::FieldProblemEntry fft = pops::default_poisson_entry();
  fft.solver = pops::EllipticSolverKind::FFT;
  fft.boundary = pops::FieldBoundaryKind::Dirichlet;
  const int id = reg.register_problem(fft);
  EXPECT_THROW(reg.validate(id, pops::LayoutRoute::Uniform), std::invalid_argument);
}

TEST(FieldProblemRegistry, ValidateRefusesRouteNotSupported) {
  pops::FieldProblemRegistry reg;
  pops::FieldProblemEntry uniform_only = pops::default_poisson_entry();
  uniform_only.supports_amr = false;
  const int id = reg.register_problem(uniform_only);
  EXPECT_NO_THROW(reg.validate(id, pops::LayoutRoute::Uniform));
  try {
    reg.validate(id, pops::LayoutRoute::Amr);
    FAIL() << "an entry that does not support AMR must be refused there";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("AMR"), std::string::npos);
  }
}

TEST(FieldProblemRegistry, ValidateRefusesEmptyOutputLayout) {
  pops::FieldProblemRegistry reg;
  pops::FieldProblemEntry empty;
  empty.id = "hollow";  // no channels added at all
  const int id = reg.register_problem(empty);
  try {
    reg.validate(id, pops::LayoutRoute::Uniform);
    FAIL() << "a field problem producing no output must be refused";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("hollow"), std::string::npos);
  }
}

TEST(FieldProblemRegistry, ValidateAllWalksEveryEntry) {
  pops::FieldProblemRegistry reg;
  reg.register_problem(pops::default_poisson_entry());
  pops::FieldProblemEntry fft = pops::default_poisson_entry();
  fft.id = "phi_fft";
  fft.solver = pops::EllipticSolverKind::FFT;
  reg.register_problem(fft);
  EXPECT_NO_THROW(reg.validate_all(pops::LayoutRoute::Uniform));
  EXPECT_THROW(reg.validate_all(pops::LayoutRoute::Amr), std::invalid_argument)
      << "the FFT entry fails the AMR pass";
}

}  // namespace
