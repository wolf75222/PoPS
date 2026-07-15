#include <gtest/gtest.h>

#include <limits>
#include <type_traits>

#include <pops/runtime/amr/amr_tensor_elliptic.hpp>
#include <pops/runtime/numerical_defaults.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

namespace {

using pops::Real;
using pops::runtime::program::AmrProgramContext;
using pops::runtime::program::AmrTensorElliptic;

using ConfigureSolver = void (AmrProgramContext::*)(int, int, int, Real, Real, int, int) const;
static_assert(std::is_same_v<
              decltype(&AmrProgramContext::configure_composite_tensor_fac), ConfigureSolver>);

TEST(AmrProgramContextContract, AnonymousRateIdentityIsRejectedBeforeTopologyLookup) {
  AmrProgramContext context(nullptr, nullptr);
  EXPECT_THROW((void)context.boundary_evaluation_point(-1), std::invalid_argument);
}

TEST(AmrTensorFacSolver, OmittedFacControlsResolveFromNativeOptionsOnly) {
  AmrTensorElliptic driver(nullptr, 0, 1);
  EXPECT_THROW(driver.composite_fac_options(Real(1.0e-8), Real(0), 23), std::logic_error);

  driver.configure_composite_tensor_fac(0, Real(0), Real(0), 0, -1);
  const pops::CompositeFacOptions options =
      driver.composite_fac_options(Real(3.0e-8), Real(2.0e-12), 23);

  EXPECT_EQ(options.max_iters, 23);
  EXPECT_EQ(options.rel_tol, Real(3.0e-8));
  EXPECT_EQ(options.abs_tol, Real(2.0e-12));
  EXPECT_EQ(options.fine_sweeps, pops::kFACDefaultFineSweeps);
  EXPECT_EQ(options.coarse_rel_tol, pops::kFACInitialCoarseRelTol);
  EXPECT_EQ(options.coarse_abs_tol, pops::kFACInitialCoarseAbsTol);
  EXPECT_EQ(options.coarse_cycles, pops::kFACInitialCoarseMaxCycles);
  EXPECT_FALSE(options.verbose);
}

TEST(AmrTensorFacSolver, ExplicitFacControlsJoinDirectSolverControls) {
  AmrTensorElliptic driver(nullptr, 0, 1);
  driver.configure_composite_tensor_fac(7, Real(2.0e-7), Real(4.0e-14), 9, 1);
  pops::CompositeFacOptions options =
      driver.composite_fac_options(Real(4.0e-8), Real(3.0e-13), 17);

  EXPECT_EQ(options.max_iters, 17);
  EXPECT_EQ(options.rel_tol, Real(4.0e-8));
  EXPECT_EQ(options.abs_tol, Real(3.0e-13));
  EXPECT_EQ(options.fine_sweeps, 7);
  EXPECT_EQ(options.coarse_rel_tol, Real(2.0e-7));
  EXPECT_EQ(options.coarse_abs_tol, Real(4.0e-14));
  EXPECT_EQ(options.coarse_cycles, 9);
  EXPECT_TRUE(options.verbose);

  driver.configure_composite_tensor_fac(8, Real(3.0e-7), Real(5.0e-14), 10, 0);
  options = driver.composite_fac_options(Real(5.0e-8), Real(4.0e-13), 19);
  EXPECT_EQ(options.max_iters, 19);
  EXPECT_EQ(options.rel_tol, Real(5.0e-8));
  EXPECT_EQ(options.abs_tol, Real(4.0e-13));
  EXPECT_EQ(options.fine_sweeps, 8);
  EXPECT_EQ(options.coarse_rel_tol, Real(3.0e-7));
  EXPECT_EQ(options.coarse_abs_tol, Real(5.0e-14));
  EXPECT_EQ(options.coarse_cycles, 10);
  EXPECT_FALSE(options.verbose);
}

TEST(AmrTensorFacSolver, WireAndDirectSolverControlsAreStrictlyValidated) {
  AmrTensorElliptic driver(nullptr, 0, 1);
  EXPECT_THROW(driver.configure_composite_tensor_fac(-1, Real(0), Real(0), 0, -1),
               std::invalid_argument);
  EXPECT_THROW(driver.configure_composite_tensor_fac(0, Real(-1.0e-7), Real(0), 0, -1),
               std::invalid_argument);
  EXPECT_THROW(driver.configure_composite_tensor_fac(0, Real(1), Real(0), 0, -1),
               std::invalid_argument);
  EXPECT_THROW(
      driver.configure_composite_tensor_fac(
          0, std::numeric_limits<Real>::quiet_NaN(), Real(0), 0, -1),
      std::invalid_argument);
  EXPECT_THROW(driver.configure_composite_tensor_fac(0, Real(0), Real(0), -1, -1),
               std::invalid_argument);
  EXPECT_THROW(driver.configure_composite_tensor_fac(0, Real(0), Real(0), 0, -2),
               std::invalid_argument);
  EXPECT_THROW(driver.configure_composite_tensor_fac(0, Real(0), Real(0), 0, 2),
               std::invalid_argument);

  EXPECT_THROW(driver.configure_composite_tensor_fac(0, Real(0), Real(-1), 0, -1),
               std::invalid_argument);
  driver.configure_composite_tensor_fac(0, Real(0), Real(0), 0, -1);
  EXPECT_THROW(driver.composite_fac_options(Real(0), Real(0), 1), std::invalid_argument);
  EXPECT_THROW(
      driver.composite_fac_options(std::numeric_limits<Real>::quiet_NaN(), Real(0), 1),
      std::invalid_argument);
  EXPECT_THROW(driver.composite_fac_options(Real(1.0e-8), Real(-1), 1),
               std::invalid_argument);
  EXPECT_THROW(
      driver.composite_fac_options(Real(1.0e-8),
                                   std::numeric_limits<Real>::quiet_NaN(), 1),
      std::invalid_argument);
  EXPECT_THROW(driver.composite_fac_options(Real(1.0e-8), Real(0), 0),
               std::invalid_argument);
}

TEST(AmrTensorFacSolver, NonScalarOperatorIsRejectedAtTheNativeBoundary) {
  EXPECT_THROW((AmrTensorElliptic(nullptr, 0, 2)), std::invalid_argument);
}

}  // namespace
