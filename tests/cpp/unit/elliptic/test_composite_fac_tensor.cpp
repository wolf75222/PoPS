#include <gtest/gtest.h>

#include <pops/runtime/amr/amr_tensor_elliptic.hpp>
#include <pops/runtime/program/prepared_tensor_boundary_session.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

template <class Ranked, int Dim, class Value>
Ranked filled(Value value) {
  Ranked result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::PhysicalBoundaryConditions<Dim> dirichlet(const pops::Geometry<Dim>& geometry) {
  std::array<pops::PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  faces.fill({pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)});
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  return {pops::BoundaryTopology<Dim>::physical(), faces, spacing};
}

pops::runtime::program::HierarchyTensorSolverBuildRequest<2> request(int coarse_cells) {
  using namespace pops;
  using namespace pops::runtime::program;
  const Box<2> coarse_domain{Index<2>{0, 0}, Index<2>{coarse_cells - 1, coarse_cells - 1}};
  const Geometry<2> coarse_geometry =
      Geometry<2>::from_bounds(coarse_domain, RealVector<2>{0, 0}, RealVector<2>{1, 1});
  const mesh::BoxArray<2> coarse_layout(std::vector<Box<2>>{coarse_domain});
  const mesh::RankSpace<2> rank_space{Index<2>{0, 0}, Extent<2>{1, 1}};
  const auto coarse_distribution = mesh::Distribution<2>::replicated(coarse_layout, rank_space);
  const Geometry<2> fine_geometry = coarse_geometry.refine(Extent<2>{2, 2});
  const int fine_cells = 2 * coarse_cells;
  const Box<2> patch{Index<2>{fine_cells / 4, fine_cells / 4},
                     Index<2>{3 * fine_cells / 4 - 1, 3 * fine_cells / 4 - 1}};
  const mesh::BoxArray<2> fine_layout(std::vector<Box<2>>{patch});
  const auto fine_distribution = mesh::Distribution<2>::partitioned(
      fine_layout, rank_space, std::vector<Index<2>>{Index<2>{0, 0}});

  HierarchyTensorSolverBuildRequest<2> result;
  result.block = 1;
  result.components = 1;
  result.levels = {{coarse_geometry, dirichlet(coarse_geometry), coarse_layout, coarse_distribution,
                    Index<2>{0, 0}}};
  result.levels.push_back(
      {fine_geometry, dirichlet(fine_geometry), fine_layout, fine_distribution, Index<2>{0, 0}});
  result.ratios = {amr::RefinementRatio<2>{std::array<int, 2>{2, 2}}};
  result.plan_identity = "tests.full-tensor-composite-mms";
  result.operator_contract_identity =
      std::string(tensor_elliptic_detail::kScalarTensorEllipticContract);
  result.assembly_field_slots = tensor_elliptic_detail::assembly_slots<2>();
  result.solution_field_slot = "pops.tensor-elliptic.solution";
  result.options = tensor_elliptic_detail::default_options();
  result.options.values.emplace("fac.fine_sweeps", std::int64_t{48});
  result.options.values.emplace("fac.coarse_cycles", std::int64_t{96});
  result.options.values.emplace("fac.coarse_rel_tol", 1.0e-10);
  return result;
}

std::size_t ordinal(const pops::Box<2>& box, const pops::Index<2>& cell) {
  return static_cast<std::size_t>(cell[0] - box.lo[0]) +
         static_cast<std::size_t>(cell[1] - box.lo[1]) * static_cast<std::size_t>(box.length(0));
}

pops::Real exact(pops::Real x, pops::Real y) {
  return x * (pops::Real(1) - x) * y * (pops::Real(1) - y);
}

double solve_error(int cells) {
  using namespace pops;
  using namespace pops::runtime::program;
  auto build = request(cells);
  std::vector<Geometry<2>> geometries;
  for (const auto& level : build.levels)
    geometries.push_back(level.geometry);
  const ExecutionLane lane = ExecutionLane::world("tests.full-tensor-composite-mms");
  const auto registry = make_default_hierarchy_tensor_solver_provider_registry<2>(lane);
  auto solver = prepare_hierarchy_tensor_solver_collectively(
      *registry, tensor_elliptic_detail::kCompositeTensorProvider, std::move(build), lane);
  constexpr Real axx = Real(2);
  constexpr Real axy = Real(0.3);
  constexpr Real ayx = Real(-0.2);
  constexpr Real ayy = Real(1.5);
  for (int level = 0; level < solver->level_count(); ++level) {
    solver->assembly_target(tensor_elliptic_detail::coefficient_slot(0, 0), level).set_val(axx);
    solver->assembly_target(tensor_elliptic_detail::coefficient_slot(0, 1), level).set_val(axy);
    solver->assembly_target(tensor_elliptic_detail::coefficient_slot(1, 0), level).set_val(ayx);
    solver->assembly_target(tensor_elliptic_detail::coefficient_slot(1, 1), level).set_val(ayy);
    auto& rhs = solver->assembly_target("pops.tensor-elliptic.rhs", level);
    auto& fab = rhs.fab(0);
    auto host = fab.create_host_mirror();
    const auto box = fab.box();
    for (int j = box.lo[1]; j <= box.hi[1]; ++j)
      for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
        const Real x = geometries[static_cast<std::size_t>(level)].cell_coordinate(0, i);
        const Real y = geometries[static_cast<std::size_t>(level)].cell_coordinate(1, j);
        const Real rhs_value = Real(2) * axx * y * (Real(1) - y) +
                               Real(2) * ayy * x * (Real(1) - x) -
                               (axy + ayx) * (Real(1) - Real(2) * x) * (Real(1) - Real(2) * y);
        host(ordinal(fab.grown_box(), Index<2>{i, j})) = rhs_value;
      }
    fab.copy_from_host(host);
    solver->stage_initial_guess(level, nullptr);
  }
  const SolveReport report =
      solve_prepared_hierarchy_tensor_collectively(*solver, {Real(8e-7), Real(1e-12), 80}, lane)
          .consume(SolveConsumption::kAccept);
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

TEST(test_composite_fac_tensor, full_tensor_composite_retains_refinement_accuracy) {
  constexpr int coarse_cells = 24;
  const double coarse_refined = solve_error(coarse_cells);
  const double fine_refined = solve_error(2 * coarse_cells);
  EXPECT_GT(coarse_refined, 0.0);
  EXPECT_LT(fine_refined, 0.4 * coarse_refined)
      << "genuinely refined full-tensor FAC must converge under hierarchy refinement";
}

TEST(test_composite_fac_tensor, tensor_boundary_point_refresh_is_collective_and_transactional) {
  using namespace pops;
  using namespace pops::runtime::program;
  constexpr int patch_cells = 4;
  const ExecutionLane lane = ExecutionLane::world("tests.tensor-boundary-point-refresh");
  const int rank_count = lane.size();
  const int local_rank = lane.rank();
  const Box<2> domain{Index<2>{0, 0}, Index<2>{patch_cells * rank_count - 1, patch_cells - 1}};
  const Geometry<2> geometry =
      Geometry<2>::from_bounds(domain, RealVector<2>{0, 0}, RealVector<2>{1, 1});
  std::vector<Box<2>> patches;
  std::vector<Index<2>> owners;
  patches.reserve(static_cast<std::size_t>(rank_count));
  owners.reserve(static_cast<std::size_t>(rank_count));
  for (int rank = 0; rank < rank_count; ++rank) {
    patches.push_back(
        {Index<2>{rank * patch_cells, 0}, Index<2>{(rank + 1) * patch_cells - 1, patch_cells - 1}});
    owners.push_back(Index<2>{rank, 0});
  }
  const mesh::BoxArray<2> layout(std::move(patches));
  const mesh::RankSpace<2> rank_space{Index<2>{0, 0}, Extent<2>{rank_count, 1}};
  const auto distribution =
      mesh::Distribution<2>::partitioned(layout, rank_space, std::move(owners));
  MultiFab<2> prototype(layout, distribution, Index<2>{local_rank, 0}, 1, Extent<2>{1, 1});
  int program_owner = 0;
  int runtime_owner = 0;
  const runtime::multiblock::BoundaryEvaluationPoint initial{"main", 0,    0,  0,  1, {1, 2},
                                                             0.1,    0.05, "", "", ""};
  auto session = PreparedTensorBoundarySession<2>::prepare(
      geometry, dirichlet(geometry), prototype, lane, 1,
      PreparedTensorBoundaryAuthority{&program_owner, &runtime_owner,
                                      reinterpret_cast<std::uintptr_t>(&prototype),
                                      std::string(lane.identity()), 0, 0, 0, 11, 17},
      initial);

  const auto previous = session->point();
  auto rejected = previous;
  if (rank_count > 1) {
    // Every rank supplies a locally valid point, but the full exact contract diverges by tick.
    rejected.tick = static_cast<std::int64_t>(local_rank + 1);
    rejected.physical_time = 0.1 * static_cast<double>(local_rank + 1);
  } else {
    // The serial fixture still proves that local failure does not partially publish the candidate.
    rejected.clock = "foreign";
  }
  EXPECT_THROW(session->refresh_point(rejected), std::exception);
  EXPECT_EQ(session->point(), previous);

  auto retry = previous;
  retry.tick = 1;
  retry.physical_time = 0.15;
  EXPECT_NO_THROW(session->refresh_point(retry));
  EXPECT_EQ(session->point(), retry);
}
