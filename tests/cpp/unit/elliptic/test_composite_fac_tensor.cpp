#include <gtest/gtest.h>

#include <pops/core/foundation/allocator.hpp>
#include <pops/runtime/amr/amr_tensor_elliptic.hpp>
#include <pops/runtime/program/prepared_tensor_boundary_session.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
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

pops::runtime::program::HierarchyTensorSolverBuildRequest<2> fragmented_request(int coarse_cells) {
  using namespace pops;
  using namespace pops::runtime::program;
  if (coarse_cells < 4 || coarse_cells % 2 != 0)
    throw std::invalid_argument("fragmented tensor hierarchy requires an even coarse extent");

  auto result = request(coarse_cells);
  const mesh::RankSpace<2> rank_space{Index<2>{0, 0}, Extent<2>{1, 1}};
  const int coarse_midpoint = coarse_cells / 2;
  const mesh::BoxArray<2> coarse_layout(std::vector<Box<2>>{
      {Index<2>{0, 0}, Index<2>{coarse_midpoint - 1, coarse_cells - 1}},
      {Index<2>{coarse_midpoint, 0}, Index<2>{coarse_cells - 1, coarse_cells - 1}},
  });
  result.levels[0].layout = coarse_layout;
  // Keep the serial fixture rank-local, but make the parent uniquely owned.  This materializes
  // the RegionTransferPlan contracts and resident transport arenas that a replicated parent does
  // not own, while the same test remains valid under an MPI world lane.
  result.levels[0].distribution = mesh::Distribution<2>::partitioned(
      coarse_layout, rank_space, std::vector<Index<2>>{Index<2>{0, 0}, Index<2>{0, 0}});

  const int fine_cells = 2 * coarse_cells;
  const int fine_begin = fine_cells / 4;
  const int fine_midpoint = fine_cells / 2;
  const int fine_end = 3 * fine_cells / 4 - 1;
  const mesh::BoxArray<2> fine_layout(std::vector<Box<2>>{
      {Index<2>{fine_begin, fine_begin}, Index<2>{fine_midpoint - 1, fine_end}},
      {Index<2>{fine_midpoint, fine_begin}, Index<2>{fine_end, fine_end}},
  });
  result.levels[1].layout = fine_layout;
  result.levels[1].distribution = mesh::Distribution<2>::partitioned(
      fine_layout, rank_space, std::vector<Index<2>>{Index<2>{0, 0}, Index<2>{0, 0}});
  return result;
}

template <int Dim>
pops::runtime::program::HierarchyTensorConfiguredStorageRequest<Dim> configured_storage_request(
    const pops::runtime::program::HierarchyTensorSolverBuildRequest<Dim>& build,
    const pops::ExecutionLane& lane) {
  using namespace pops;
  using namespace pops::runtime::program;
  HierarchyTensorConfiguredStorageRequest<Dim> result;
  result.rank_bound = static_cast<std::uint64_t>(lane.size());
  result.components = build.components;
  result.provider_identity = std::string(tensor_elliptic_detail::kCompositeTensorProvider);
  result.provider_interface_version = CompositeTensorHierarchyProvider<Dim>{}.interface_version();
  result.execution_lane_identity = std::string(lane.identity());
  result.plan_identity = build.plan_identity;
  result.operator_contract_identity = build.operator_contract_identity;
  result.assembly_field_slots = build.assembly_field_slots;
  result.solution_field_slot = build.solution_field_slot;
  result.options = build.options;
  result.level_cell_bounds.reserve(build.levels.size());
  result.patch_bounds.reserve(build.levels.size());
  for (const auto& level : build.levels) {
    const std::int64_t cells = level.geometry.domain().numPts();
    if (cells <= 0)
      throw std::logic_error("test hierarchy has an empty configured level");
    result.level_cell_bounds.push_back(static_cast<std::uint64_t>(cells));
    result.patch_bounds.push_back(static_cast<std::uint64_t>(level.layout.size()));
  }
  result.parent_child_pair_bounds.reserve(build.ratios.size());
  for (std::size_t parent = 0; parent + 1U < build.levels.size(); ++parent) {
    const std::uint64_t left = static_cast<std::uint64_t>(build.levels[parent].layout.size());
    const std::uint64_t right = static_cast<std::uint64_t>(build.levels[parent + 1U].layout.size());
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
      throw std::overflow_error("test configured pair bound overflows uint64");
    result.parent_child_pair_bounds.push_back(left * right);
  }
  return result;
}

template <int Dim>
pops::runtime::program::HierarchyTensorConfiguredStorageRequest<Dim>
synthetic_configured_storage_request() {
  using namespace pops::runtime::program;
  HierarchyTensorConfiguredStorageRequest<Dim> result;
  result.level_cell_bounds = {64, 128};
  result.patch_bounds = {1, 2};
  result.parent_child_pair_bounds = {2};
  result.rank_bound = 2;
  result.components = 1;
  result.provider_identity = std::string(tensor_elliptic_detail::kCompositeTensorProvider);
  result.provider_interface_version = CompositeTensorHierarchyProvider<Dim>{}.interface_version();
  result.execution_lane_identity = "tests.tensor-configured-storage-synthetic";
  result.plan_identity = "tests.tensor-configured-storage-plan";
  result.operator_contract_identity =
      std::string(tensor_elliptic_detail::kScalarTensorEllipticContract);
  result.assembly_field_slots = tensor_elliptic_detail::assembly_slots<Dim>();
  result.solution_field_slot = "pops.tensor-elliptic.solution";
  result.options = tensor_elliptic_detail::default_options();
  return result;
}

class UnboundedTensorProvider final
    : public pops::runtime::program::HierarchyTensorSolverProvider<2> {
 public:
  using request_type = pops::runtime::program::HierarchyTensorSolverBuildRequest<2>;
  using solver_type = pops::runtime::program::PreparedHierarchyTensorSolver<2>;

  std::string_view identity() const noexcept override { return "tests.tensor-provider.unknown"; }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return "tests.tensor-provider.unknown.contract";
  }
  std::vector<std::string> capability_contracts() const override { return {}; }
  pops::PreparedProviderOptions default_options() const override {
    return pops::runtime::program::tensor_elliptic_detail::default_options();
  }
  pops::PreparedProviderSupport accepts_options(
      const pops::PreparedProviderOptions&) const noexcept override {
    return pops::PreparedProviderSupport::accept();
  }
  pops::PreparedProviderSupport supports(const request_type&) const noexcept override {
    return pops::PreparedProviderSupport::accept();
  }
  pops::PreparedProviderSupport accepts_execution(
      const request_type&,
      pops::runtime::program::HierarchyTensorSolverExecutionPath) const noexcept override {
    return pops::PreparedProviderSupport::accept();
  }
  std::string expected_prepared_contract(const request_type&) const override { return "unused"; }
  std::unique_ptr<solver_type> prepare(const request_type&,
                                       const pops::ExecutionLane&) const override {
    return {};
  }
};

std::size_t ordinal(const pops::Box<2>& box, const pops::Index<2>& cell) {
  return static_cast<std::size_t>(cell[0] - box.lo[0]) +
         static_cast<std::size_t>(cell[1] - box.lo[1]) * static_cast<std::size_t>(box.length(0));
}

pops::Real exact(pops::Real x, pops::Real y) {
  return x * (pops::Real(1) - x) * y * (pops::Real(1) - y);
}

struct SolveMeasurement {
  double error = 0;
  pops::SolveReport report{};
};

SolveMeasurement solve_error(int cells) {
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
  auto outcome =
      solve_prepared_hierarchy_tensor_collectively(*solver, {Real(8e-7), Real(1e-12), 80}, lane);
  const SolveReport report =
      outcome.consume(outcome.report().solved_value_available()
                          ? SolveConsumption::kAccept
                          : (outcome.report().action == SolveAction::kRejectAttempt
                                 ? SolveConsumption::kRejectAttempt
                                 : SolveConsumption::kFailRun));
  if (!report.solved())
    return {0, report};

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
  return {error, report};
}

}  // namespace

TEST(test_composite_fac_tensor, full_tensor_composite_retains_refinement_accuracy) {
  constexpr int coarse_cells = 24;
  const SolveMeasurement coarse = solve_error(coarse_cells);
  const SolveMeasurement fine = solve_error(2 * coarse_cells);
  ASSERT_TRUE(coarse.report.solved()) << coarse.report.reason;
  ASSERT_TRUE(fine.report.solved()) << fine.report.reason;
  const double coarse_refined = coarse.error;
  const double fine_refined = fine.error;
  EXPECT_GT(coarse_refined, 0.0);
  EXPECT_LT(fine_refined, 0.4 * coarse_refined)
      << "genuinely refined full-tensor FAC must converge under hierarchy refinement";
}

TEST(test_composite_fac_tensor,
     configured_provider_storage_ceiling_is_finite_monotone_and_dominates_preparation) {
  using namespace pops;
  using namespace pops::runtime::program;
  const ExecutionLane lane = ExecutionLane::world("tests.tensor-configured-storage");
  const auto build = request(16);
  const auto configured = configured_storage_request(build, lane);
  ASSERT_TRUE(hierarchy_tensor_detail::request_fits_configured_storage(build, configured));
  const CompositeTensorHierarchyProvider<2> provider;
  const HierarchyTensorConfiguredStorageLimit limit = provider.configured_storage_limit(configured);
  ASSERT_TRUE(limit.is_exact());
  ASSERT_GT(limit.maximum_bytes, 0U);

  auto registry = std::make_shared<HierarchyTensorSolverProviderRegistry<2>>();
  registry->add(std::make_shared<CompositeTensorHierarchyProvider<2>>(), lane);
  auto solver = prepare_hierarchy_tensor_solver_collectively(
      *registry, tensor_elliptic_detail::kCompositeTensorProvider, build, lane);
  const PreparedResidentStorage actual = solver->resident_storage();
  ASSERT_TRUE(actual.is_exact());
  EXPECT_LE(actual.bytes, limit.maximum_bytes)
      << "the sealed configured tensor ceiling must dominate the prepared resident receipt";

  auto larger = configured;
  ++larger.patch_bounds.back();
  larger.level_cell_bounds.back() *= 2U;
  ++larger.parent_child_pair_bounds.back();
  ++larger.rank_bound;
  const HierarchyTensorConfiguredStorageLimit larger_limit =
      provider.configured_storage_limit(larger);
  ASSERT_TRUE(larger_limit.is_exact());
  EXPECT_GE(larger_limit.maximum_bytes, limit.maximum_bytes)
      << "increasing a configured hierarchy envelope may not reduce its storage ceiling";

  auto foreign_provider = configured;
  foreign_provider.provider_identity = "tests.tensor-provider.foreign";
  EXPECT_FALSE(provider.configured_storage_limit(foreign_provider).is_exact());

  auto foreign_version = configured;
  ++foreign_version.provider_interface_version;
  EXPECT_FALSE(provider.configured_storage_limit(foreign_version).is_exact());

  auto overflowing = configured;
  overflowing.level_cell_bounds.front() = std::numeric_limits<std::uint64_t>::max();
  overflowing.patch_bounds.front() = 2;
  EXPECT_THROW((void)provider.configured_storage_limit(overflowing), std::overflow_error);
}

TEST(test_composite_fac_tensor,
     configured_provider_storage_ceiling_dominates_each_topology_within_patch_envelope) {
  using namespace pops;
  using namespace pops::runtime::program;
  const ExecutionLane lane = ExecutionLane::world("tests.tensor-configured-storage-fragmented");
  const auto compact = request(16);
  const auto fragmented = fragmented_request(16);
  const auto configured = configured_storage_request(fragmented, lane);
  ASSERT_TRUE(hierarchy_tensor_detail::request_fits_configured_storage(compact, configured));
  ASSERT_TRUE(hierarchy_tensor_detail::request_fits_configured_storage(fragmented, configured));

  const CompositeTensorHierarchyProvider<2> provider;
  const HierarchyTensorConfiguredStorageLimit limit = provider.configured_storage_limit(configured);
  ASSERT_TRUE(limit.is_exact());

  auto registry = std::make_shared<HierarchyTensorSolverProviderRegistry<2>>();
  registry->add(std::make_shared<CompositeTensorHierarchyProvider<2>>(), lane);
  auto compact_solver = prepare_hierarchy_tensor_solver_collectively(
      *registry, tensor_elliptic_detail::kCompositeTensorProvider, compact, lane);
  auto fragmented_solver = prepare_hierarchy_tensor_solver_collectively(
      *registry, tensor_elliptic_detail::kCompositeTensorProvider, fragmented, lane);
  const PreparedResidentStorage compact_storage = compact_solver->resident_storage();
  const PreparedResidentStorage fragmented_storage = fragmented_solver->resident_storage();
  ASSERT_TRUE(compact_storage.is_exact());
  ASSERT_TRUE(fragmented_storage.is_exact());
  EXPECT_LE(compact_storage.bytes, limit.maximum_bytes);
  EXPECT_LE(fragmented_storage.bytes, limit.maximum_bytes);

  auto too_small = configured;
  --too_small.patch_bounds.front();
  EXPECT_FALSE(hierarchy_tensor_detail::request_fits_configured_storage(fragmented, too_small));
}

TEST(test_composite_fac_tensor, configured_provider_storage_ceiling_is_native_rank_generic) {
  using namespace pops::runtime::program;
  const auto limit_1 = CompositeTensorHierarchyProvider<1>{}.configured_storage_limit(
      synthetic_configured_storage_request<1>());
  const auto limit_2 = CompositeTensorHierarchyProvider<2>{}.configured_storage_limit(
      synthetic_configured_storage_request<2>());
  const auto limit_3 = CompositeTensorHierarchyProvider<3>{}.configured_storage_limit(
      synthetic_configured_storage_request<3>());
  EXPECT_TRUE(limit_1.is_exact());
  EXPECT_TRUE(limit_2.is_exact());
  EXPECT_TRUE(limit_3.is_exact());
  EXPECT_GT(limit_1.maximum_bytes, 0U);
  EXPECT_GT(limit_2.maximum_bytes, 0U);
  EXPECT_GT(limit_3.maximum_bytes, 0U);
}

TEST(test_composite_fac_tensor, unbounded_provider_is_rejected_by_default_configured_contract) {
  using namespace pops;
  const ExecutionLane lane = ExecutionLane::world("tests.tensor-configured-storage-unknown");
  const auto build = request(8);
  const auto configured = configured_storage_request(build, lane);
  const UnboundedTensorProvider provider;
  EXPECT_FALSE(provider.configured_storage_limit(configured).is_exact())
      << "third-party providers must opt in to a finite configured ceiling";
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
  const AllocationEventStats allocation_before_rejection = allocation_event_stats();
  EXPECT_THROW(session->refresh_point(rejected), std::exception);
  EXPECT_EQ(allocation_event_stats(), allocation_before_rejection)
      << "a rejected prepared tensor boundary refresh must not allocate a fallback transport";
  EXPECT_EQ(session->point(), previous);

  auto retry = previous;
  retry.tick = 1;
  retry.physical_time = 0.15;
  const AllocationEventStats allocation_before_retry = allocation_event_stats();
  EXPECT_NO_THROW(session->refresh_point(retry));
  EXPECT_EQ(allocation_event_stats(), allocation_before_retry)
      << "a prepared tensor boundary refresh must reuse its cold-bound transport";
  EXPECT_EQ(session->point(), retry);
}

pops::PhysicalBoundaryConditions<2> all_periodic(const pops::Geometry<2>& geometry) {
  std::array<bool, 2> periodic{};
  periodic.fill(true);
  pops::RealVector<2> spacing{};
  spacing[0] = geometry.spacing(0);
  spacing[1] = geometry.spacing(1);
  return {pops::BoundaryTopology<2>::axis_periodic(periodic), {}, spacing};
}

pops::runtime::program::HierarchyTensorSolverBuildRequest<2> periodic_request(int coarse_cells) {
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
  result.block = 4;
  result.components = 1;
  result.levels = {{coarse_geometry, all_periodic(coarse_geometry), coarse_layout,
                    coarse_distribution, Index<2>{0, 0}}};
  result.levels.push_back(
      {fine_geometry, all_periodic(fine_geometry), fine_layout, fine_distribution, Index<2>{0, 0}});
  result.ratios = {amr::RefinementRatio<2>{std::array<int, 2>{2, 2}}};
  result.plan_identity = "tests.full-tensor-composite-periodic-nullspace";
  result.operator_contract_identity =
      std::string(tensor_elliptic_detail::kScalarTensorEllipticContract);
  result.assembly_field_slots = tensor_elliptic_detail::assembly_slots<2>();
  result.solution_field_slot = "pops.tensor-elliptic.solution";
  result.options = tensor_elliptic_detail::default_options();
  result.options.values.emplace("fac.fine_sweeps", std::int64_t{32});
  result.options.values.emplace("fac.coarse_cycles", std::int64_t{80});
  result.options.values.emplace("fac.coarse_rel_tol", 1.0e-10);
  return result;
}

void fill_periodic_mode(pops::MultiFab<2>& rhs, const pops::Geometry<2>& geometry) {
  constexpr pops::Real kTwoPi = pops::Real{6.283185307179586476925286766559005768L};
  constexpr pops::Real kPi = pops::Real{3.141592653589793238462643383279502884L};
  pops::Real eigenvalue = pops::Real(0);
  for (int axis = 0; axis < 2; ++axis) {
    const pops::Real angle = kPi / static_cast<pops::Real>(geometry.domain().length(axis));
    const pops::Real inverse = pops::Real(1) / geometry.spacing(axis);
    eigenvalue += pops::Real(4) * std::sin(angle) * std::sin(angle) * inverse * inverse;
  }
  auto& fab = rhs.fab(0);
  auto host = fab.create_host_mirror();
  const auto box = fab.box();
  for (int j = box.lo[1]; j <= box.hi[1]; ++j)
    for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
      const pops::Real x = geometry.cell_coordinate(0, i);
      const pops::Real y = geometry.cell_coordinate(1, j);
      host(ordinal(fab.grown_box(), pops::Index<2>{i, j})) =
          eigenvalue * std::sin(kTwoPi * x) * std::sin(kTwoPi * y);
    }
  fab.copy_from_host(host);
}

double solution_mean(const pops::MultiFab<2>& field, const pops::Geometry<2>& geometry) {
  double sum = 0;
  double measure = 0;
  const auto& fab = field.fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  const auto box = fab.box();
  const double cell = static_cast<double>(geometry.spacing(0) * geometry.spacing(1));
  for (int j = box.lo[1]; j <= box.hi[1]; ++j)
    for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
      sum += static_cast<double>(host(ordinal(fab.grown_box(), pops::Index<2>{i, j}))) * cell;
      measure += cell;
    }
  return measure > 0 ? sum / measure : 0;
}

TEST(test_composite_fac_tensor, periodic_tensor_fac_applies_mean_zero_gauge) {
  using namespace pops;
  using namespace pops::runtime::program;
  auto build = periodic_request(16);
  std::vector<Geometry<2>> geometries;
  for (const auto& level : build.levels)
    geometries.push_back(level.geometry);
  const ExecutionLane lane = ExecutionLane::world("tests.full-tensor-composite-periodic-nullspace");
  const auto configured = configured_storage_request(build, lane);
  const auto configured_limit =
      CompositeTensorHierarchyProvider<2>{}.configured_storage_limit(configured);
  ASSERT_TRUE(configured_limit.is_exact());
  const auto registry = make_default_hierarchy_tensor_solver_provider_registry<2>(lane);
  {
    auto incompatible = prepare_hierarchy_tensor_solver_collectively(
        *registry, tensor_elliptic_detail::kCompositeTensorProvider, periodic_request(16), lane);
    for (int level = 0; level < incompatible->level_count(); ++level) {
      incompatible->assembly_target(tensor_elliptic_detail::coefficient_slot(0, 0), level)
          .set_val(Real(1));
      incompatible->assembly_target(tensor_elliptic_detail::coefficient_slot(1, 1), level)
          .set_val(Real(1));
      incompatible->assembly_target(tensor_elliptic_detail::coefficient_slot(0, 1), level)
          .set_val(Real(0));
      incompatible->assembly_target(tensor_elliptic_detail::coefficient_slot(1, 0), level)
          .set_val(Real(0));
      incompatible->assembly_target("pops.tensor-elliptic.rhs", level).set_val(Real(1));
      incompatible->stage_initial_guess(level, nullptr);
    }
    auto outcome = solve_prepared_hierarchy_tensor_collectively(
        *incompatible, {Real(1e-4), Real(1e-12), 40}, lane);
    const SolveReport report = outcome.consume(SolveConsumption::kFailRun);
    EXPECT_EQ(report.status, SolveStatus::kIncompatibleRhs) << report.reason;
  }
  auto solver = prepare_hierarchy_tensor_solver_collectively(
      *registry, tensor_elliptic_detail::kCompositeTensorProvider, std::move(build), lane);
  const PreparedResidentStorage storage_before_solve = solver->resident_storage();
  ASSERT_TRUE(storage_before_solve.is_exact())
      << "a periodic tensor FAC workspace must be fully materialized before its first solve";
  EXPECT_LE(storage_before_solve.bytes, configured_limit.maximum_bytes)
      << "the detached configured ceiling must include the possible singular nullspace image";
  const AllocationEventStats allocation_before_solve = allocation_event_stats();
  for (int level = 0; level < solver->level_count(); ++level) {
    solver->assembly_target(tensor_elliptic_detail::coefficient_slot(0, 0), level).set_val(Real(1));
    solver->assembly_target(tensor_elliptic_detail::coefficient_slot(1, 1), level).set_val(Real(1));
    solver->assembly_target(tensor_elliptic_detail::coefficient_slot(0, 1), level).set_val(Real(0));
    solver->assembly_target(tensor_elliptic_detail::coefficient_slot(1, 0), level).set_val(Real(0));
    fill_periodic_mode(solver->assembly_target("pops.tensor-elliptic.rhs", level),
                       geometries[static_cast<std::size_t>(level)]);
    solver->stage_initial_guess(level, nullptr);
  }
  auto outcome =
      solve_prepared_hierarchy_tensor_collectively(*solver, {Real(1e-4), Real(1e-12), 60}, lane);
  const SolveReport report =
      outcome.consume(outcome.report().solved_value_available() ? SolveConsumption::kAccept
                                                                : SolveConsumption::kFailRun);
  EXPECT_EQ(allocation_event_stats(), allocation_before_solve)
      << "a prepared tensor nullspace must not allocate during solve";
  const PreparedResidentStorage storage_after_solve = solver->resident_storage();
  ASSERT_TRUE(storage_after_solve.is_exact())
      << "a prepared tensor nullspace must remain resident-storage exact after solve";
  EXPECT_EQ(storage_after_solve.bytes, storage_before_solve.bytes)
      << "tensor nullspace solve must not change the prepared resident footprint";
  EXPECT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm;
  EXPECT_NEAR(solution_mean(solver->solution(0), geometries[0]), 0.0, 1.0e-7);
  EXPECT_NEAR(solution_mean(solver->solution(1), geometries[1]), 0.0, 1.0e-6);
}
