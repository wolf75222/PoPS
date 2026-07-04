#include <gtest/gtest.h>

#include <pops/numerics/elliptic/linear/krylov_solver.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <string>

using namespace pops;

TEST(test_numerical_defaults, newton_options_defaults_are_centralized) {
  NewtonOptions n;
  EXPECT_EQ(n.max_iters, kNewtonDefaultMaxIters) << "Newton max_iters default is centralized";
  EXPECT_EQ(n.rel_tol, kNewtonDefaultRelTol) << "Newton rel_tol default is centralized";
  EXPECT_EQ(n.abs_tol, kNewtonDefaultAbsTol) << "Newton abs_tol default is centralized";
  EXPECT_EQ(n.fd_eps, kNewtonDefaultFdEps) << "Newton fd_eps default is centralized";
  EXPECT_EQ(n.damping, kNewtonDefaultDamping) << "Newton damping default is centralized";
  EXPECT_EQ(n.fail_policy, kNewtonDefaultFailPolicy) << "Newton fail_policy default is centralized";
  EXPECT_EQ(std::string(newton_fail_policy_name(n.fail_policy)), "none")
      << "Newton fail policy report name is stable";
}

TEST(test_numerical_defaults, model_spec_defaults_are_centralized) {
  ModelSpec spec;
  EXPECT_EQ(spec.gamma, static_cast<double>(kPhysicalDefaultGamma)) << "ModelSpec gamma default";
  EXPECT_EQ(spec.B0, static_cast<double>(kPhysicalDefaultB0)) << "ModelSpec B0 default";
  EXPECT_EQ(spec.cs2, static_cast<double>(kPhysicalDefaultFluidStateCs2)) << "ModelSpec cs2 default";
  EXPECT_EQ(spec.q, static_cast<double>(kPhysicalDefaultChargeQ)) << "ModelSpec charge default";
}

TEST(test_numerical_defaults, mg_krylov_fac_amr_named_constants) {
  EXPECT_EQ(kMGDefaultRelTol, Real(1e-8)) << "MG rel_tol default is reported";
  EXPECT_EQ(kMGDefaultMaxCycles, 50) << "MG max_cycles default is reported";
  EXPECT_EQ(kMGDefaultCoarseThreshold, 0) << "MG coarse_threshold sentinel (0 = disabled) is named";
  EXPECT_EQ(kTensorKrylovDefaultMaxIters, 200) << "Tensor Krylov budget is reported";
  EXPECT_EQ(kSchurKrylovCartesianMaxIters, 400) << "Cartesian Schur Krylov budget is reported";
  EXPECT_EQ(kSchurKrylovPolarMaxIters, 600) << "Polar Schur Krylov budget is reported";
  EXPECT_EQ(kFACDefaultMaxIters, 30) << "FAC max_iters default is reported";
  EXPECT_EQ(kFACInitialCoarseRelTol, Real(1e-12)) << "FAC initial coarse tolerance is reported";
  EXPECT_EQ(kFACInitialCoarseMaxCycles, 100) << "FAC initial coarse cycle budget is reported";
  EXPECT_EQ(kAmrRefinementDisabledThreshold, Real(1e30))
      << "AMR disabled refinement threshold is named";
  EXPECT_EQ(kWenoEpsilon, Real(1e-40)) << "WENO epsilon is named";
  EXPECT_EQ(kEbCutFractionFloor, Real(1e-3)) << "EB cut fraction floor is named";
}
