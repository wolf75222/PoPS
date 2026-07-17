#include <pops_bench/cases.hpp>

#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>
#include <pops/runtime/program/coeff_elliptic_ops.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops::bench {
namespace {

constexpr Real kAxx = Real(1);
constexpr Real kAyy = Real(1);
constexpr Real kAxy = Real(0.2);
constexpr Real kAyx = Real(-0.1);
constexpr double kPi = 3.141592653589793238462643383279502884;

POPS_HD Real exact_solution(Real x, Real y) {
  return static_cast<Real>(std::sin(kPi * static_cast<double>(x)) *
                           std::sin(kPi * static_cast<double>(y)));
}

struct ManufacturedRhs {
  Array4 rhs;
  Geometry geometry;

  POPS_HD void operator()(int i, int j) const {
    const Real x = static_cast<Real>(geometry.x_cell(i));
    const Real y = static_cast<Real>(geometry.y_cell(j));
    const Real sin_sin = exact_solution(x, y);
    const Real cos_cos = static_cast<Real>(
        std::cos(kPi * static_cast<double>(x)) * std::cos(kPi * static_cast<double>(y)));
    rhs(i, j, 0) = static_cast<Real>(-kPi * kPi) * (kAxx + kAyy) * sin_sin +
                   static_cast<Real>(kPi * kPi) * (kAxy + kAyx) * cos_cos;
  }
};

struct PreparedBenchmarkContext {
  GridContext grid;

  bool is_polar_geometry() const { return false; }
  GridContext grid_context() const { return grid; }
  void count_kernel() const {}
};

OperatorEvaluationSnapshot benchmark_snapshot(const MultiFab& prototype) {
  return {{UINT64_C(1), UINT64_C(2), UINT64_C(3), UINT64_C(4)},
          1,
          0,
          0,
          1,
          std::bit_cast<std::uint64_t>(1.0),
          std::bit_cast<std::uint64_t>(0.0),
          1,
          detail::layout_fingerprint(prototype),
          {UINT64_C(5), UINT64_C(6), UINT64_C(7), UINT64_C(8)}};
}

std::string parameters_json(const BenchmarkConfig& config, const BoxArray& boxes) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"nx\":" << config.krylov_n
      << ",\"ny\":" << config.krylov_n << ",\"tile\":" << config.krylov_tile
      << ",\"boxes\":" << boxes.size() << ",\"global_valid_cells\":"
      << static_cast<long long>(config.krylov_n) * static_cast<long long>(config.krylov_n)
      << ",\"operator\":\"div(A grad phi)\",\"axx\":" << kAxx
      << ",\"ayy\":" << kAyy << ",\"axy\":" << kAxy << ",\"ayx\":" << kAyx
      << ",\"boundary\":\"homogeneous_dirichlet\","
      << "\"preconditioner\":\"geometric_mg_one_vcycle_diagonal_tensor\","
      << "\"rel_tol\":" << config.krylov_rel_tol << ",\"abs_tol\":"
      << config.krylov_abs_tol << ",\"max_iters\":" << config.krylov_max_iters << '}';
  return out.str();
}

}  // namespace

void run_tensor_krylov_case(const BenchmarkConfig& config, const RuntimeMetadata& metadata,
                            JsonlWriter& writer) {
  const Box2D domain = Box2D::from_extents(config.krylov_n, config.krylov_n);
  const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
  const BoxArray boxes = BoxArray::from_domain(domain, config.krylov_tile);
  BCRec boundary;
  boundary.xlo = boundary.xhi = boundary.ylo = boundary.yhi = BCType::Dirichlet;

  auto op = std::make_shared<GeometricMG>(geometry, boxes, boundary);
  op->set_epsilon_anisotropic([](Real, Real) { return kAxx; },
                              [](Real, Real) { return kAyy; });
  op->set_cross_terms([](Real, Real) { return kAxy; }, [](Real, Real) { return kAyx; });
  MultiFab iterate(boxes, op->dmap(), 1, 1);
  MultiFab rhs(boxes, op->dmap(), 1, 0);
  for (int local = 0; local < rhs.local_size(); ++local)
    for_each_cell(rhs.box(local), ManufacturedRhs{rhs.fab(local).array(), geometry});

  ApplyFn apply = [op](MultiFab& out, const MultiFab& in) {
    MultiFab& mutable_input = const_cast<MultiFab&>(in);
    device_fence();
    fill_ghosts(mutable_input, op->geom().domain, op->bc());
    apply_laplacian(mutable_input, op->geom(), out, op->op_coef(), op->op_eps(),
                    op->op_kappa(), op->op_eps_y(), op->op_a_xy(), op->op_a_yx());
  };

  auto context = std::make_shared<PreparedBenchmarkContext>();
  context->grid.dom = domain;
  context->grid.geom = geometry;
  context->grid.bc = boundary;
  auto mg = std::make_shared<runtime::program::GeometricMgPreconditioner>();
  PreparedLinearPreconditioner preconditioner(
      iterate,
      [context, mg](MultiFab& out, const MultiFab& in) { mg->apply(*context, out, in); },
      [context, mg, &iterate]() { mg->prepare(*context, iterate); });
  const KrylovFootprint footprint{1, 1, 0, true};
  OperatorEvaluationSnapshot snapshot = benchmark_snapshot(iterate);
  PreparedAffineLinearProblem problem(
      iterate, std::move(apply), std::move(preconditioner), LinearOperatorProperties::general(),
      footprint, PreparedNullspacePolicy::nonsingular(), [&snapshot]() { return snapshot; });
  KrylovWorkspace workspace(iterate, KrylovMethod::kBicgstab, footprint);
  problem.prepare(snapshot);
  workspace.bind(problem);
  const KrylovControls controls{KrylovMethod::kBicgstab,
                                static_cast<Real>(config.krylov_rel_tol),
                                static_cast<Real>(config.krylov_abs_tol),
                                config.krylov_max_iters,
                                0,
                                Real(1)};

  SolveReport report;
  std::vector<int> measured_iterations;
  auto prepare = [&] { PureFieldAlgebra::zero_valid(iterate); };
  auto run = [&] { report = solve_prepared_affine(problem, workspace, iterate, rhs, controls); };
  auto observe = [&](bool measured) {
    if (!report.solved())
      throw std::runtime_error(std::string("tensor_krylov solve failed: ") + report.status_name());
    if (measured)
      measured_iterations.push_back(report.iters);
  };
  const std::vector<double> samples =
      run_repeated(config.warmups, config.repetitions, prepare, run, observe);

  // Validation is outside the timed interval: independently recompute the true residual and inspect
  // the manufactured-solution error over valid cells.
  MultiFab residual_field(boxes, op->dmap(), 1, 1);
  problem.true_residual(residual_field, rhs, iterate);
  const double residual = static_cast<double>(PureFieldAlgebra::norm(residual_field));
  const double forcing_norm = static_cast<double>(report.reference_residual_norm);
  const double relative_residual = forcing_norm > 0.0 ? residual / forcing_norm : residual;
  device_fence();
  barrier();
  iterate.sync_host();
  double local_error = 0.0;
  long local_nonfinite = 0;
  for (int local = 0; local < iterate.local_size(); ++local) {
    const ConstArray4 phi = iterate.fab(local).const_array();
    const Box2D valid = iterate.box(local);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        const double exact = static_cast<double>(
            exact_solution(static_cast<Real>(geometry.x_cell(i)),
                           static_cast<Real>(geometry.y_cell(j))));
        const double computed = static_cast<double>(phi(i, j, 0));
        if (!std::isfinite(exact) || !std::isfinite(computed)) {
          local_nonfinite = 1;
          continue;
        }
        local_error = std::max(local_error, std::fabs(computed - exact));
      }
  }
  const double max_error = all_reduce_max(local_error);
  const bool nonfinite_solution = all_reduce_max(local_nonfinite) != 0;
  const double requested_stop =
      std::max(config.krylov_rel_tol * forcing_norm, config.krylov_abs_tol);
  const double residual_limit =
      4.0 * requested_stop +
      512.0 * static_cast<double>(std::numeric_limits<Real>::epsilon()) *
          std::max(1.0, forcing_norm);
  const double dx = 1.0 / static_cast<double>(config.krylov_n);
  const double discretization_limit = 64.0 * dx * dx;
  const bool passed = report.solved() && !nonfinite_solution && std::isfinite(residual) &&
                      std::isfinite(relative_residual) && residual <= residual_limit &&
                      std::isfinite(max_error) && max_error <= discretization_limit;

  std::vector<double> iteration_values;
  iteration_values.reserve(measured_iterations.size());
  for (const int value : measured_iterations)
    iteration_values.push_back(static_cast<double>(value));
  const RobustStats iteration_stats = summarize(iteration_values);

  std::ostringstream validation;
  validation << "{\"passed\":" << (passed ? "true" : "false")
             << ",\"nonfinite_solution\":" << (nonfinite_solution ? "true" : "false")
             << ",\"timed\":false,\"solve_status\":" << json_escape(report.status_name())
             << ",\"true_residual_l2\":" << json_number(residual)
             << ",\"forcing_l2\":" << json_number(forcing_norm)
             << ",\"true_relative_residual\":" << json_number(relative_residual)
             << ",\"residual_limit\":" << json_number(residual_limit)
             << ",\"manufactured_max_error\":" << json_number(max_error)
             << ",\"resolution_scaled_error_limit\":" << json_number(discretization_limit)
             << '}';

  std::ostringstream iterations;
  iterations << std::setprecision(17) << "{\"samples\":"
             << json_integer_array(measured_iterations) << ",\"min\":" << iteration_stats.minimum
             << ",\"median\":" << iteration_stats.median << ",\"max\":"
             << iteration_stats.maximum << '}';

  const std::string timing =
      "{\"unit\":\"seconds\",\"clock\":\"steady_clock\","
      "\"rank_aggregation\":\"max\",\"device_fence\":\"before_and_after\","
      "\"mpi_barrier\":\"before_and_after\",\"performance_threshold\":null,\"warmups\":" +
      std::to_string(config.warmups) + ",\"repetitions\":" +
      std::to_string(config.repetitions) + ",\"statistics\":" + stats_json(samples) + '}';

  writer.write(record_prefix(metadata, "tensor_krylov", "bicgstab_geometric_mg", "cold_repeated") +
               ",\"parameters\":" + parameters_json(config, boxes) + ",\"timing\":" + timing +
               ",\"iterations\":" + iterations.str() + ",\"validation\":" +
               validation.str() + '}');

  if (!passed)
    throw std::runtime_error("tensor_krylov numerical validation failed");
}

}  // namespace pops::bench
