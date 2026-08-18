#include <pops_bench/cases.hpp>
#include <pops_bench/ranked_setup.hpp>

#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops::bench {
namespace {

constexpr int kDim = kNativeDimension;
constexpr Real kPi = Real(3.141592653589793238462643383279502884L);

using Solver = elliptic::mg::GeometricMG<kDim>;
using BenchmarkBox = Box<kDim>;
using BenchmarkLayout = mesh::BoxArray<kDim>;

POPS_HD Real exact_solution(const Geometry<kDim>& geometry, const Index<kDim>& index) {
  Real value = Real(1);
  for (int axis = 0; axis < kDim; ++axis)
    value *= Kokkos::sin(kPi * geometry.cell_coordinate(axis, index[axis]));
  return value;
}

struct ManufacturedRhs {
  FieldView<Real, kDim> rhs{};
  Geometry<kDim> geometry;

  POPS_HD void operator()(const Index<kDim>& index) const {
    rhs(index, 0) = Real(kDim) * kPi * kPi * exact_solution(geometry, index);
  }
};

EllipticBuildRequest<kDim> build_request(const BenchmarkConfig& config) {
  const BenchmarkBox domain = BenchmarkBox::from_extents(filled_ranked<Extent<kDim>>(config.mg_n));
  const Geometry<kDim> geometry = Geometry<kDim>::from_bounds(
      domain, RealVector<kDim>{}, filled_ranked<RealVector<kDim>>(Real(1)));
  const BenchmarkLayout layout =
      BenchmarkLayout::from_domain(domain, filled_ranked<Extent<kDim>>(config.mg_tile));
  const mesh::RankSpace<kDim> ranks = benchmark_rank_space<kDim>();
  const auto ownership = parallel::LoadBalanceProvider<kDim>::space_filling_curve().prepare(
      layout, ranks, load_balance_budget<kDim>(layout, domain));

  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * kDim)> faces{};
  faces.fill(PhysicalBoundaryFace{PhysicalBoundaryKind::dirichlet, Real(0)});
  RealVector<kDim> spacing{};
  for (int axis = 0; axis < kDim; ++axis)
    spacing[axis] = geometry.spacing(axis);

  return {geometry,
          layout,
          ownership.distribution(),
          benchmark_local_rank(ranks),
          PhysicalBoundaryConditions<kDim>{BoundaryTopology<kDim>::physical(), faces, spacing},
          Extent<kDim>{},
          filled_ranked<Extent<kDim>>(1),
          layout_validation_budget(layout)};
}

void fill_manufactured_rhs(Solver& solver) {
  for (std::size_t local = 0; local < solver.rhs().local_size(); ++local)
    for_each_cell(solver.rhs().box(local),
                  ManufacturedRhs{solver.rhs().fab(local).view(), solver.geom()});
  device_fence();
}

double maximum_solution_error(const Solver& solver) {
  double local_error = 0;
  long local_nonfinite = 0;
  for (std::size_t local = 0; local < solver.phi().local_size(); ++local) {
    const auto& fab = solver.phi().fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::int64_t ordinal = 0; ordinal < fab.box().numPts(); ++ordinal) {
      const Index<kDim> index = index_from_ordinal(fab.box(), ordinal);
      const double computed = static_cast<double>(host(host_offset(fab.grown_box(), index)));
      const double exact = static_cast<double>(exact_solution(solver.geom(), index));
      if (!std::isfinite(computed) || !std::isfinite(exact)) {
        local_nonfinite = 1;
        continue;
      }
      local_error = std::max(local_error, std::abs(computed - exact));
    }
  }
  if (all_reduce_max(local_nonfinite) != 0)
    return std::numeric_limits<double>::infinity();
  return all_reduce_max(local_error);
}

std::string parameters_json(const BenchmarkConfig& config, const Solver& solver) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"dimension\":" << kDim << ",\"uniform_extent\":" << config.mg_n
      << ",\"tile\":" << config.mg_tile << ",\"boxes\":" << solver.phi().layout().size()
      << ",\"global_valid_cells\":" << solver.geom().domain().numPts()
      << ",\"operator\":\"-laplacian\",\"boundary\":\"homogeneous_dirichlet\""
      << ",\"solver\":\"geometric_multigrid\",\"levels\":" << solver.num_levels()
      << ",\"rel_tol\":" << config.mg_rel_tol << ",\"abs_tol\":" << config.mg_abs_tol
      << ",\"max_cycles\":" << config.mg_max_cycles << '}';
  return out.str();
}

}  // namespace

void run_scalar_mg_case(const BenchmarkConfig& config, const RuntimeMetadata& metadata,
                        JsonlWriter& writer) {
  elliptic::mg::GeometricMultigridOptions options;
  options.relative_tolerance = static_cast<Real>(config.mg_rel_tol);
  options.absolute_tolerance = static_cast<Real>(config.mg_abs_tol);
  options.maximum_cycles = config.mg_max_cycles;
  options.bottom_sweeps = 60;
  Solver solver(build_request(config), options);
  solver.install_nullspace(FieldNullspacePlan<kDim>{},
                           PreparedVectorDistribution<kDim>::distributed());
  fill_manufactured_rhs(solver);

  SolveReport report;
  std::vector<int> measured_iterations;
  auto prepare = [&] { solver.phi().set_val(Real(0)); };
  auto run = [&] { report = solver.solve(); };
  auto observe = [&](bool measured) {
    if (!report.solved())
      throw std::runtime_error(std::string("scalar multigrid solve failed: ") + report.reason);
    if (measured)
      measured_iterations.push_back(report.iters);
  };

  // validate_before_timing: manufactured residual/error must pass before samples are recorded.
  prepare();
  run();
  device_fence();
  barrier();
  const double max_error = maximum_solution_error(solver);
  const double forcing_norm = static_cast<double>(report.reference_residual_norm);
  const double residual = static_cast<double>(report.residual_norm);
  const double requested_stop = std::max(config.mg_rel_tol * forcing_norm, config.mg_abs_tol);
  const double residual_limit =
      4.0 * requested_stop + 512.0 * static_cast<double>(std::numeric_limits<Real>::epsilon()) *
                                 std::max(1.0, forcing_norm);
  const double dx = 1.0 / static_cast<double>(config.mg_n);
  const double discretization_limit = 64.0 * static_cast<double>(kDim) * dx * dx;
  const bool passed = report.solved() && std::isfinite(residual) && residual <= residual_limit &&
                      std::isfinite(max_error) && max_error <= discretization_limit;
  if (!passed)
    throw std::runtime_error("scalar multigrid numerical validation failed");

  const std::vector<double> samples =
      run_repeated(config.warmups, config.repetitions, prepare, run, observe);

  std::vector<double> iteration_values;
  iteration_values.reserve(measured_iterations.size());
  for (const int value : measured_iterations)
    iteration_values.push_back(static_cast<double>(value));
  const RobustStats iteration_stats = summarize(iteration_values);

  std::ostringstream validation;
  validation << "{\"passed\":" << (passed ? "true" : "false")
             << ",\"timed\":false,\"solve_status\":" << json_escape(report.status_name())
             << ",\"residual_l2\":" << json_number(residual)
             << ",\"forcing_l2\":" << json_number(forcing_norm)
             << ",\"residual_limit\":" << json_number(residual_limit)
             << ",\"manufactured_max_error\":" << json_number(max_error)
             << ",\"resolution_scaled_error_limit\":" << json_number(discretization_limit) << '}';

  std::ostringstream iterations;
  iterations << std::setprecision(17) << "{\"samples\":" << json_integer_array(measured_iterations)
             << ",\"min\":" << iteration_stats.minimum << ",\"median\":" << iteration_stats.median
             << ",\"max\":" << iteration_stats.maximum << '}';

  const std::string timing =
      "{\"unit\":\"seconds\",\"clock\":\"steady_clock\","
      "\"rank_aggregation\":\"max\",\"device_fence\":\"before_and_after\","
      "\"mpi_barrier\":\"before_and_after\",\"performance_threshold\":null,\"warmups\":" +
      std::to_string(config.warmups) + ",\"repetitions\":" + std::to_string(config.repetitions) +
      ",\"statistics\":" + stats_json(samples) + '}';

  writer.write(record_prefix(metadata, "scalar_mg", "geometric_multigrid", "cold_repeated") +
               ",\"parameters\":" + parameters_json(config, solver) + ",\"timing\":" + timing +
               ",\"iterations\":" + iterations.str() + ",\"validation\":" + validation.str() + '}');
}

}  // namespace pops::bench
