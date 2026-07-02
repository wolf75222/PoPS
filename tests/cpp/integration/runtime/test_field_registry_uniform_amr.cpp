// ADC-596: the SAME FieldProblemRegistry abstraction drives the Uniform and the AMR routes, and
// its entries surface as report rows. This integration test builds a registry the way both
// register_named_field paths do (default "phi" + a named GeometricMG field), then:
//   - validates it for Uniform AND for AMR (parity: a plain named GeometricMG field is legal on
//     both routes, so both passes accept it);
//   - shows the FFT-on-AMR asymmetry is caught by the shared validate();
//   - checks the field-problem report rows carry id / equation / solver / outputs for each route.
#include <gtest/gtest.h>

#include <string>

#include <pops/runtime/module_capabilities.hpp>          // field_problem_routes
#include <pops/runtime/system/field_problem_registry.hpp>

namespace {

// Build a registry exactly like the Uniform/AMR register_named_field seams do: seed "phi", then a
// named GeometricMG field with a full phi+grad layout in the named-aux band.
pops::FieldProblemRegistry make_default_plus_named() {
  pops::FieldProblemRegistry reg;
  reg.register_problem(pops::default_poisson_entry());
  reg.register_problem(pops::named_field_entry(
      "psi", pops::kAuxNamedBase, pops::kAuxNamedBase + 1, pops::kAuxNamedBase + 2,
      pops::EllipticSolverKind::GeometricMG));
  return reg;
}

TEST(FieldRegistryUniformAmr, SameRegistryValidatesOnBothRoutes) {
  pops::FieldProblemRegistry reg = make_default_plus_named();
  // The identical entries pass on Uniform AND AMR -- the shared abstraction, one description.
  EXPECT_NO_THROW(reg.validate_all(pops::LayoutRoute::Uniform));
  EXPECT_NO_THROW(reg.validate_all(pops::LayoutRoute::Amr));
  EXPECT_EQ(reg.find("phi"), 0);
  EXPECT_EQ(reg.find("psi"), 1);
}

TEST(FieldRegistryUniformAmr, FftEntryIsUniformOnlyUnderTheSharedValidate) {
  pops::FieldProblemRegistry reg;
  pops::FieldProblemEntry fft = pops::default_poisson_entry();
  fft.solver = pops::EllipticSolverKind::FFT;
  reg.register_problem(fft);
  EXPECT_NO_THROW(reg.validate_all(pops::LayoutRoute::Uniform));
  EXPECT_THROW(reg.validate_all(pops::LayoutRoute::Amr), std::invalid_argument)
      << "the ONE shared validate refuses FFT on AMR -- Uniform and AMR agree on the rule";
}

TEST(FieldRegistryUniformAmr, ReportRowsCarryIdSolverAndOutputsPerRoute) {
  pops::FieldProblemRegistry reg = make_default_plus_named();

  const std::vector<pops::CapabilityRouteReport> uni =
      pops::field_problem_routes(reg, pops::LayoutRoute::Uniform);
  ASSERT_EQ(uni.size(), 2u);
  EXPECT_EQ(uni[0].feature, "field_problem:phi");
  EXPECT_EQ(uni[0].layout, "uniform");
  EXPECT_NE(uni[0].reason.find("GeometricMG"), std::string::npos);
  EXPECT_NE(uni[0].reason.find("phi"), std::string::npos) << "the phi output is listed";
  EXPECT_EQ(uni[1].feature, "field_problem:psi");
  EXPECT_NE(uni[1].reason.find("psi"), std::string::npos);

  const std::vector<pops::CapabilityRouteReport> amr =
      pops::field_problem_routes(reg, pops::LayoutRoute::Amr);
  ASSERT_EQ(amr.size(), 2u);
  EXPECT_EQ(amr[0].layout, "amr") << "same entries, route-accurate layout column";
  EXPECT_EQ(amr[1].feature, "field_problem:psi");
}

TEST(FieldRegistryUniformAmr, NamedFieldComponentsAreTheLowLevelTruthBehindTheLayout) {
  pops::FieldProblemRegistry reg = make_default_plus_named();
  // The named field's AuxLayout is the named VIEW of the phi/grad components the runtime NamedField
  // keeps -- component_of round-trips back to those raw indices.
  const pops::FieldProblemEntry& psi = reg.at(reg.find("psi"));
  EXPECT_EQ(psi.layout.component_of("psi"), pops::kAuxNamedBase);
  EXPECT_EQ(psi.layout.component_of("psi_grad_x"), pops::kAuxNamedBase + 1);
  EXPECT_EQ(psi.layout.component_of("psi_grad_y"), pops::kAuxNamedBase + 2);
}

}  // namespace
