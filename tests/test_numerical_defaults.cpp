#include <pops/numerics/elliptic/linear/krylov_solver.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <cmath>
#include <iostream>
#include <string>

namespace {
void chk(bool cond, const std::string& label) {
  std::cout << "  [" << (cond ? "OK " : "XX ") << "] " << label << "\n";
  if (!cond)
    std::exit(1);
}
}  // namespace

int main() {
  using namespace pops;

  NewtonOptions n;
  chk(n.max_iters == kNewtonDefaultMaxIters, "Newton max_iters default is centralized");
  chk(n.rel_tol == kNewtonDefaultRelTol, "Newton rel_tol default is centralized");
  chk(n.abs_tol == kNewtonDefaultAbsTol, "Newton abs_tol default is centralized");
  chk(n.fd_eps == kNewtonDefaultFdEps, "Newton fd_eps default is centralized");
  chk(n.damping == kNewtonDefaultDamping, "Newton damping default is centralized");
  chk(n.fail_policy == kNewtonDefaultFailPolicy, "Newton fail_policy default is centralized");
  chk(std::string(newton_fail_policy_name(n.fail_policy)) == "none",
      "Newton fail policy report name is stable");

  ModelSpec spec;
  chk(spec.gamma == static_cast<double>(kPhysicalDefaultGamma), "ModelSpec gamma default");
  chk(spec.B0 == static_cast<double>(kPhysicalDefaultB0), "ModelSpec B0 default");
  chk(spec.cs2 == static_cast<double>(kPhysicalDefaultFluidStateCs2), "ModelSpec cs2 default");
  chk(spec.q == static_cast<double>(kPhysicalDefaultChargeQ), "ModelSpec charge default");

  chk(kMGDefaultRelTol == Real(1e-8), "MG rel_tol default is reported");
  chk(kMGDefaultMaxCycles == 50, "MG max_cycles default is reported");
  chk(kTensorKrylovDefaultMaxIters == 200, "Tensor Krylov budget is reported");
  chk(kSchurKrylovCartesianMaxIters == 400, "Cartesian Schur Krylov budget is reported");
  chk(kSchurKrylovPolarMaxIters == 600, "Polar Schur Krylov budget is reported");
  chk(kFACDefaultMaxIters == 30, "FAC max_iters default is reported");
  chk(kFACInitialCoarseRelTol == Real(1e-12), "FAC initial coarse tolerance is reported");
  chk(kFACInitialCoarseMaxCycles == 100, "FAC initial coarse cycle budget is reported");
  chk(kAmrRefinementDisabledThreshold == Real(1e30),
      "AMR disabled refinement threshold is named");
  chk(kWenoEpsilon == Real(1e-40), "WENO epsilon is named");
  chk(kEbCutFractionFloor == Real(1e-3), "EB cut fraction floor is named");

  return 0;
}
