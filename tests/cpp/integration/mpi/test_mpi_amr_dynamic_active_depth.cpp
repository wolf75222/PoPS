#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "test_harness.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace hierarchy = pops::amr::hierarchy;
namespace tagging = pops::amr::tagging;

constexpr pops::mesh::BoxArrayValidationBudget kLayoutBudget{64, 2016};
constexpr hierarchy::HierarchyValidationBudget kHierarchyBudget{4, 4096};
constexpr pops::parallel::LoadBalancePreparationBudget kLoadBalanceBudget{
    64, 16, std::numeric_limits<std::int64_t>::max()};

template <int Dim>
pops::Extent<Dim> filled_extent(int value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::Index<Dim> rank_coordinate(int rank) {
  pops::Index<Dim> result{};
  result[0] = rank;
  return result;
}

template <int Dim>
std::shared_ptr<const pops::PreparedLoadBalanceAuthority<Dim>> load_balance() {
  return std::make_shared<const pops::PreparedLoadBalanceAuthority<Dim>>(
      pops::prepare_load_balance_authority<Dim>(
          "space_filling_curve", "test.mpi-active-depth.sfc",
          pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
}

template <int Dim>
pops::runtime::amr::AmrRuntime<Dim> make_runtime(int n, int ranks, int rank) {
  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = n - 1;
  const pops::Box<Dim> domain{lower, upper};
  pops::Extent<Dim> tile = filled_extent<Dim>(n);
  tile[0] = n / ranks;
  const pops::mesh::BoxArray<Dim> patches = pops::mesh::BoxArray<Dim>::from_domain(domain, tile);
  pops::Extent<Dim> rank_extent = filled_extent<Dim>(1);
  rank_extent[0] = ranks;
  const pops::mesh::RankSpace<Dim> rank_space(pops::Index<Dim>{}, rank_extent);
  std::vector<pops::Index<Dim>> owners;
  owners.reserve(patches.size());
  for (std::size_t patch = 0; patch < patches.size(); ++patch)
    owners.push_back(
        rank_coordinate<Dim>(static_cast<int>(patch % static_cast<std::size_t>(ranks))));
  const auto distribution =
      pops::mesh::Distribution<Dim>::partitioned(patches, rank_space, std::move(owners));
  hierarchy::LevelLayout<Dim> coarse(0, domain, patches, distribution,
                                     pops::amr::RefinementRatio<Dim>{}, kLayoutBudget);
  pops::MultiFab<Dim> state(patches, distribution, rank_coordinate<Dim>(rank), 1,
                            filled_extent<Dim>(1));
  state.set_val(pops::Real(1));
  return pops::runtime::amr::AmrRuntime<Dim>(
      hierarchy::AmrHierarchy<Dim>::from_coarse(coarse, std::move(state), kHierarchyBudget),
      load_balance<Dim>(), "test.mpi-active-depth.spatial");
}

template <int Dim>
pops::Box<Dim> inset(const pops::Box<Dim>& box, int amount) {
  pops::Index<Dim> lower = box.lo;
  pops::Index<Dim> upper = box.hi;
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] += amount;
    upper[axis] -= amount;
  }
  return {lower, upper};
}

template <int Dim>
tagging::ClusterResult<Dim> cluster_result(const hierarchy::LevelLayout<Dim>& parent,
                                           std::vector<pops::Box<Dim>> boxes,
                                           std::string identity) {
  const pops::mesh::BoxArray<Dim> clustered(std::move(boxes));
  tagging::ClusterOptions<Dim> options;
  options.min_efficiency = 0.7;
  for (int axis = 0; axis < Dim; ++axis) {
    options.min_box_size[static_cast<std::size_t>(axis)] = 1;
    options.max_box_size[static_cast<std::size_t>(axis)] = 64;
  }
  options.budget = {64, 4096, 1U << 20, 128, 1U << 22};
  tagging::ClusterResultIdentity<Dim> exact_identity{
      std::move(identity), parent.exact_identity(), options, {}, clustered.boxes()};
  return {clustered, std::move(exact_identity)};
}

template <int Dim>
pops::amr::RefinementRatio<Dim> ratio_two() {
  std::array<int, Dim> components{};
  components.fill(2);
  return pops::amr::RefinementRatio<Dim>(components);
}

template <int Dim>
pops::amr::regridding::RegridPreparationBudget regrid_budget() {
  return {.clustered_parent_layout = kLayoutBudget,
          .fine_layout = kLayoutBudget,
          .load_balance = kLoadBalanceBudget};
}

template <int Dim>
void publish_child(pops::runtime::amr::AmrRuntime<Dim>& engine, std::size_t parent_level,
                   pops::Box<Dim> parent_patch, const std::string& identity, pops::Real value) {
  auto prepared = engine.prepare_regrid(
      parent_level, ratio_two<Dim>(),
      cluster_result(engine.hierarchy().layout(parent_level), {std::move(parent_patch)}, identity),
      regrid_budget<Dim>());
  if (!prepared.fine_layout().has_value())
    throw std::runtime_error("active-depth fixture failed to prepare a child layout");
  pops::MultiFab<Dim> child(prepared.fine_layout()->patches(),
                            prepared.fine_layout()->distribution(),
                            engine.hierarchy().state(parent_level).local_rank(),
                            engine.hierarchy().state(parent_level).ncomp(),
                            engine.hierarchy().state(parent_level).ghosts());
  child.set_val(value);
  engine.publish_regrid(parent_level, std::move(prepared), std::move(child));
}

template <int Dim>
void prove_dynamic_active_depth(int n, int ranks, int rank) {
  auto engine = make_runtime<Dim>(n, ranks, rank);
  const pops::Box<Dim> first_parent = inset(engine.hierarchy().layout(0).domain(), 2);
  publish_child(engine, 0, first_parent, "test.mpi-active-depth.level-1", pops::Real(2));
  ASSERT_EQ(engine.hierarchy().num_levels(), 2U);
  const pops::Box<Dim> second_parent =
      inset(engine.hierarchy().layout(1).patches().boxes().front(), 2);
  publish_child(engine, 1, second_parent, "test.mpi-active-depth.level-2", pops::Real(3));
  ASSERT_EQ(engine.hierarchy().num_levels(), 3U);
  const std::uint64_t populated_epoch = engine.topology_epoch();

  auto empty = engine.prepare_regrid(
      0, ratio_two<Dim>(),
      cluster_result<Dim>(engine.hierarchy().layout(0), {}, "test.mpi-active-depth.remove"),
      regrid_budget<Dim>());
  ASSERT_TRUE(empty.removes_fine_level());
  engine.publish_regrid(0, std::move(empty), std::nullopt);
  ASSERT_EQ(engine.hierarchy().num_levels(), 1U);
  EXPECT_EQ(engine.topology_epoch(), populated_epoch + 1);

  const pops::Box<Dim> regrown_parent = inset(engine.hierarchy().layout(0).domain(), 2);
  publish_child(engine, 0, regrown_parent, "test.mpi-active-depth.regrow-1", pops::Real(4));
  const pops::Box<Dim> regrown_second_parent =
      inset(engine.hierarchy().layout(1).patches().boxes().front(), 2);
  publish_child(engine, 1, regrown_second_parent, "test.mpi-active-depth.regrow-2", pops::Real(5));
  ASSERT_EQ(engine.hierarchy().num_levels(), 3U);
  EXPECT_EQ(pops::reduce_min_local(engine.hierarchy().state(2)), pops::Real(5));
  EXPECT_EQ(pops::reduce_max_local(engine.hierarchy().state(2)), pops::Real(5));

  const double depth = static_cast<double>(engine.hierarchy().num_levels());
  const double epoch = static_cast<double>(engine.topology_epoch());
  EXPECT_EQ(pops::all_reduce_max(depth) - (-pops::all_reduce_max(-depth)), 0.0);
  EXPECT_EQ(pops::all_reduce_max(epoch) - (-pops::all_reduce_max(-epoch)), 0.0);
}

int run_dynamic_active_depth(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int failure = 0;
  {
    Kokkos::ScopeGuard guard(argc, argv);
    try {
      if (16 % pops::n_ranks() != 0)
        throw std::runtime_error("active-depth fixture requires a rank count dividing 16");
      prove_dynamic_active_depth<pops::kNativeDimension>(16, pops::n_ranks(), pops::my_rank());
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d active-depth proof failed: %s\n", pops::my_rank(),
                   error.what());
      failure = 1;
    }
    failure = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(failure || ::testing::Test::HasFailure())));
    if (pops::my_rank() == 0 && failure == 0)
      std::printf("OK test_mpi_amr_dynamic_active_depth np=%d dim=%d exact-depth\n",
                  pops::n_ranks(), pops::kNativeDimension);
  }
  pops::comm_finalize();
  return failure;
}

}  // namespace

TEST(test_mpi_amr_dynamic_active_depth, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&run_dynamic_active_depth, "test_mpi_amr_dynamic_active_depth"),
            0);
}
