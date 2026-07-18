#pragma once

#include <pops/core/identity/prepared_provider_options.hpp>
#include <pops/runtime/numerical_defaults.hpp>
#include <pops/runtime/system/system_poisson_options.hpp>

#include <cstdint>
#include <string>

namespace pops {

using AmrFieldSolverOptionValue = PreparedProviderOptionValue;
using AmrFieldSolverOptions = PreparedProviderOptions;

inline AmrFieldSolverOptions geometric_mg_amr_field_solver_options(const GeometricMgOptions& mg,
                                                                   const CompositeFacOptions& fac) {
  AmrFieldSolverOptions options;
  options.schema_identity = "pops.amr.field-solver-options.geometric-mg@1";
  options.values = {
      {"mg.abs_tol", static_cast<double>(mg.abs_tol)},
      {"mg.rel_tol", static_cast<double>(mg.rel_tol)},
      {"mg.max_cycles", static_cast<std::int64_t>(mg.max_cycles)},
      {"mg.min_coarse", static_cast<std::int64_t>(mg.min_coarse)},
      {"mg.pre_smooth", static_cast<std::int64_t>(mg.nu1)},
      {"mg.post_smooth", static_cast<std::int64_t>(mg.nu2)},
      {"mg.bottom_sweeps", static_cast<std::int64_t>(mg.nbottom)},
      {"mg.coarse_threshold", static_cast<std::int64_t>(mg.coarse_threshold)},
      {"fac.max_iters", static_cast<std::int64_t>(fac.max_iters)},
      {"fac.fine_sweeps", static_cast<std::int64_t>(fac.fine_sweeps)},
      {"fac.rel_tol", static_cast<double>(fac.rel_tol)},
      {"fac.abs_tol", static_cast<double>(fac.abs_tol)},
      {"fac.coarse_rel_tol", static_cast<double>(fac.coarse_rel_tol)},
      {"fac.coarse_abs_tol", static_cast<double>(fac.coarse_abs_tol)},
      {"fac.coarse_cycles", static_cast<std::int64_t>(fac.coarse_cycles)},
      {"fac.verbose", fac.verbose},
  };
  return options;
}

}  // namespace pops
