#include <gtest/gtest.h>

#include <pops/runtime/amr/amr_tensor_elliptic.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr pops::Real kPi = pops::Real(3.14159265358979323846);

template <int Dim>
pops::PhysicalBoundaryConditions<Dim> dirichlet(const pops::Geometry<Dim>& geometry) {
  std::array<pops::PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  faces.fill({pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)});
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  return {pops::BoundaryTopology<Dim>::physical(), faces, spacing};
}

pops::runtime::program::HierarchyTensorSolverBuildRequest<2> request(int cells) {
  using namespace pops;
  using namespace pops::runtime::program;
  const Box<2> domain{Index<2>{0, 0}, Index<2>{cells - 1, cells - 1}};
  const Geometry<2> coarse =
      Geometry<2>::from_bounds(domain, RealVector<2>{0, 0}, RealVector<2>{1, 1});
  const mesh::BoxArray<2> coarse_layout(std::vector<Box<2>>{domain});
  const mesh::RankSpace<2> ranks{Index<2>{0, 0}, Extent<2>{1, 1}};
  const auto coarse_distribution = mesh::Distribution<2>::replicated(coarse_layout, ranks);
  const Geometry<2> fine = coarse.refine(Extent<2>{2, 2});
  const Box<2> patch{Index<2>{cells / 2, cells / 2},
                     Index<2>{3 * cells / 2 - 1, 3 * cells / 2 - 1}};
  const mesh::BoxArray<2> fine_layout(std::vector<Box<2>>{patch});
  const auto fine_distribution =
      mesh::Distribution<2>::partitioned(fine_layout, ranks, std::vector<Index<2>>{Index<2>{0, 0}});
  HierarchyTensorSolverBuildRequest<2> result;
  result.block = 2;
  result.components = 1;
  result.levels = {{coarse, dirichlet(coarse), coarse_layout, coarse_distribution, Index<2>{0, 0}}};
  result.levels.push_back({fine, dirichlet(fine), fine_layout, fine_distribution, Index<2>{0, 0}});
  result.ratios = {amr::RefinementRatio<2>{std::array<int, 2>{2, 2}}};
  result.plan_identity = "tests.variable-epsilon-composite";
  result.operator_contract_identity =
      std::string(tensor_elliptic_detail::kScalarTensorEllipticContract);
  result.assembly_field_slots = tensor_elliptic_detail::assembly_slots<2>();
  result.solution_field_slot = "pops.tensor-elliptic.solution";
  result.options = tensor_elliptic_detail::default_options();
  result.options.values.emplace("fac.fine_sweeps", std::int64_t{64});
  result.options.values.emplace("fac.coarse_cycles", std::int64_t{120});
  result.options.values.emplace("fac.coarse_rel_tol", 1.0e-10);
  return result;
}

std::size_t ordinal(const pops::Box<2>& box, const pops::Index<2>& cell) {
  return static_cast<std::size_t>(cell[0] - box.lo[0]) +
         static_cast<std::size_t>(cell[1] - box.lo[1]) * static_cast<std::size_t>(box.length(0));
}

pops::Real exact(pops::Real x, pops::Real y) {
  return std::sin(kPi * x) * std::sin(kPi * y);
}

pops::Real epsilon(pops::Real x, pops::Real y) {
  return pops::Real(1) +
         pops::Real(0.25) * std::sin(pops::Real(2) * kPi * x) * std::sin(pops::Real(2) * kPi * y);
}

pops::Real rhs_value(pops::Real x, pops::Real y) {
  const pops::Real u = exact(x, y);
  const pops::Real ux = kPi * std::cos(kPi * x) * std::sin(kPi * y);
  const pops::Real uy = kPi * std::sin(kPi * x) * std::cos(kPi * y);
  const pops::Real ex =
      pops::Real(0.5) * kPi * std::cos(pops::Real(2) * kPi * x) * std::sin(pops::Real(2) * kPi * y);
  const pops::Real ey =
      pops::Real(0.5) * kPi * std::sin(pops::Real(2) * kPi * x) * std::cos(pops::Real(2) * kPi * y);
  return pops::Real(2) * kPi * kPi * epsilon(x, y) * u - ex * ux - ey * uy;
}

double solve_error(int cells) {
  using namespace pops;
  using namespace pops::runtime::program;
  auto build = request(cells);
  std::vector<Geometry<2>> geometries;
  for (const auto& level : build.levels)
    geometries.push_back(level.geometry);
  const ExecutionLane lane = ExecutionLane::world("tests.variable-epsilon-composite");
  const auto registry = make_default_hierarchy_tensor_solver_provider_registry<2>(lane);
  auto solver = prepare_hierarchy_tensor_solver_collectively(
      *registry, tensor_elliptic_detail::kCompositeTensorProvider, std::move(build), lane);
  for (int level = 0; level < solver->level_count(); ++level) {
    auto& xx = solver->assembly_target(tensor_elliptic_detail::coefficient_slot(0, 0), level);
    auto& yy = solver->assembly_target(tensor_elliptic_detail::coefficient_slot(1, 1), level);
    auto& rhs = solver->assembly_target("pops.tensor-elliptic.rhs", level);
    auto xx_host = xx.fab(0).create_host_mirror();
    auto yy_host = yy.fab(0).create_host_mirror();
    auto rhs_host = rhs.fab(0).create_host_mirror();
    const auto box = rhs.box(0);
    for (int j = box.lo[1]; j <= box.hi[1]; ++j)
      for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
        const Real x = geometries[static_cast<std::size_t>(level)].cell_coordinate(0, i);
        const Real y = geometries[static_cast<std::size_t>(level)].cell_coordinate(1, j);
        const std::size_t rhs_index = ordinal(rhs.fab(0).grown_box(), Index<2>{i, j});
        const std::size_t coefficient_index = ordinal(xx.fab(0).grown_box(), Index<2>{i, j});
        xx_host(coefficient_index) = epsilon(x, y);
        yy_host(coefficient_index) = epsilon(x, y);
        rhs_host(rhs_index) = rhs_value(x, y);
      }
    xx.fab(0).copy_from_host(xx_host);
    yy.fab(0).copy_from_host(yy_host);
    solver->assembly_target(tensor_elliptic_detail::coefficient_slot(0, 1), level).set_val(Real(0));
    solver->assembly_target(tensor_elliptic_detail::coefficient_slot(1, 0), level).set_val(Real(0));
    rhs.fab(0).copy_from_host(rhs_host);
    solver->stage_initial_guess(level, nullptr);
  }
  SolveOutcome outcome =
      solve_prepared_hierarchy_tensor_collectively(*solver, {Real(1e-6), Real(1e-12), 100}, lane);
  const SolveReport report = consume_solve_outcome(std::move(outcome));
  if (!report.solved())
    throw std::runtime_error(report.reason);
  constexpr int measured_level = 1;
  const auto& solution = solver->solution(measured_level);
  const auto& fab = solution.fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  const Box<2> region = fab.box().grow(-std::max(2, cells / 4));
  double error = 0;
  for (int j = region.lo[1]; j <= region.hi[1]; ++j)
    for (int i = region.lo[0]; i <= region.hi[0]; ++i) {
      const Real x = geometries[static_cast<std::size_t>(measured_level)].cell_coordinate(0, i);
      const Real y = geometries[static_cast<std::size_t>(measured_level)].cell_coordinate(1, j);
      error = std::max(error, std::abs(static_cast<double>(
                                  host(ordinal(fab.grown_box(), Index<2>{i, j})) - exact(x, y))));
    }
  return error;
}

}  // namespace

TEST(test_composite_fac_variable_eps, variable_diagonal_composite_retains_refinement_accuracy) {
  constexpr int coarse_cells = 24;
  const double coarse_refined = solve_error(coarse_cells);
  const double fine_refined = solve_error(2 * coarse_cells);
  EXPECT_GT(coarse_refined, 0.0);
  EXPECT_LT(fine_refined, 0.75 * coarse_refined)
      << "genuinely refined variable-epsilon FAC must converge under hierarchy refinement";
}
