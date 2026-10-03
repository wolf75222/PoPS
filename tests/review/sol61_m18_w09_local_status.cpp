#include <cmath>
#include <iostream>
#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>

int main() {
  // SourceOnly declared quadrature: the W09 interval is exact; this concrete
  // three-node positive quadrature is a witness, not an undocumented original.
  for (double target : {0.0, 0.49, 0.5, 0.9}) {
    auto residual = [target](const pops::Real (&a)[2], pops::Real (&r)[2]) {
      r[0] = -1; r[1] = -target;
      for (double v : {-0.5, 0.0, 0.5}) {
        const double p = std::exp(a[0] + v*a[1])/3;
        r[0] += p; r[1] += v*p;
      }
      return pops::LocalNonlinearEvaluationResult::ok();
    };
    pops::PreparedLocalNonlinearControls controls;
    controls.max_iterations = 40;
    controls.max_backtracks = 16;
    controls.absolute_tolerance = 2e-11;
    controls.minimum_step = std::ldexp(1.0,-16);
    controls.safeguard = pops::LocalSafeguardKind::kBacktrackingLineSearch;
    const auto problem = pops::prepare_local_nonlinear_problem<2>(residual,
      pops::FiniteDifferenceLocalJacobian<2>{}, pops::AcceptAllLocalCandidates<2>{}, controls);
    const pops::Real seed[2] = {0,0};
    const auto result = pops::solve_prepared_local_nonlinear(problem,seed);
    std::cout.precision(17);
    std::cout << target << ' ' << static_cast<int>(result.status) << ' '
      << result.iterations << ' ' << result.residual_norm << '\n';
    if (target == .9 && result.solved()) return 2;
  }
}
