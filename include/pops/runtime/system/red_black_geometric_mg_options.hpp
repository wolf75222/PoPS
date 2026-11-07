#pragma once
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/core/identity/prepared_provider_options.hpp>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>

namespace pops::runtime::system {

template <int Dim>
elliptic::mg::GeometricMultigridOptions decode_red_black_geometric_mg_options(
    const PreparedProviderOptions& source) {
  const bool roundoff_contract =
      source.schema_identity == "pops.system.red-black-geometric-mg-options@2";
  if ((!roundoff_contract &&
       source.schema_identity != "pops.system.red-black-geometric-mg-options@1") ||
      source.values.size() != (roundoff_contract ? 18u : 16u))
    throw std::invalid_argument("staircase field MG requires its exact versioned option schema");
  const auto real = [&](const char* name) {
    const auto found = source.values.find(name);
    if (found == source.values.end() || !std::holds_alternative<double>(found->second))
      throw std::invalid_argument(std::string("staircase field MG requires exact real option: ") +
                                  name);
    return static_cast<Real>(std::get<double>(found->second));
  };
  const auto integer = [&](const char* name) {
    const auto found = source.values.find(name);
    if (found == source.values.end() || !std::holds_alternative<std::int64_t>(found->second))
      throw std::invalid_argument(
          std::string("staircase field MG requires exact integer option: ") + name);
    const std::int64_t value = std::get<std::int64_t>(found->second);
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
      throw std::invalid_argument("staircase field MG integer option overflows its native type");
    return static_cast<int>(value);
  };
  elliptic::mg::GeometricMultigridOptions options;
  options.relative_tolerance = real("rel_tol");
  options.absolute_tolerance = real("abs_tol");
  options.maximum_cycles = integer("max_cycles");
  options.minimum_coarse_extent = integer("min_coarse");
  options.pre_sweeps = integer("pre_smooth");
  options.post_sweeps = integer("post_smooth");
  options.bottom_sweeps = integer("bottom_sweeps");
  options.coarse_cell_threshold = integer("coarse_threshold");
  if (integer("smoother") != 1 || (integer("prolongation") != 0 && integer("prolongation") != 1))
    throw std::invalid_argument("staircase field MG has an unknown exact V-cycle algorithm");
  options.smoother = elliptic::mg::GeometricSmoother::red_black_gauss_seidel;
  options.prolongation = integer("prolongation") == 1
                             ? elliptic::mg::GeometricProlongation::constant_injection
                             : elliptic::mg::GeometricProlongation::multilinear;
  const auto bottom = source.values.find("pre_smooth_bottom");
  if (bottom == source.values.end() || !std::holds_alternative<bool>(bottom->second))
    throw std::invalid_argument(
        "staircase field MG requires its exact bottom pre-smoothing option");
  options.pre_smooth_bottom = std::get<bool>(bottom->second);
  if (roundoff_contract) {
    if (integer("roundoff_policy") != 1 || integer("roundoff_policy_version") != 1)
      throw std::invalid_argument(
          "red-black MG requires the exact supported residual evaluation policy");
    options.roundoff_aware_residual = true;
  }
  const elliptic::mg::CellCentreBallDomain domain{
      integer("domain_dimension"),
      {real("domain_centre_0"), real("domain_centre_1"), real("domain_centre_2")},
      real("domain_radius")};
  if (domain.dimension != 0)
    options.field_domain = domain;
  else if (domain.radius != Real(0) || domain.centre != std::array<Real, 3>{})
    throw std::invalid_argument("full-layout geometric MG cannot carry dormant region coordinates");
  elliptic::mg::detail::validate_options<Dim>(options);
  return options;
}

}  // namespace pops::runtime::system
