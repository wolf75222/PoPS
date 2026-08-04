#include <gtest/gtest.h>

#include <pops/runtime/amr/amr_runtime.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace hierarchy = pops::amr::hierarchy;
namespace regridding = pops::amr::regridding;
namespace tagging = pops::amr::tagging;
namespace mesh = pops::mesh;
namespace runtime = pops::runtime::amr;

using pops::Box;
using pops::Extent;
using pops::Index;
using pops::MultiFab;

constexpr mesh::BoxArrayValidationBudget kLayoutBudget{32, 496};
constexpr hierarchy::HierarchyValidationBudget kHierarchyBudget{4, 1024};
constexpr pops::parallel::LoadBalancePreparationBudget kLoadBalanceBudget{
    32, 8, std::numeric_limits<std::int64_t>::max()};

template <int Dim>
mesh::RankSpace<Dim> serial_rank_space() {
  Index<Dim> origin{};
  Extent<Dim> extent{};
  for (int axis = 0; axis < Dim; ++axis) {
    origin[axis] = axis - 3;
    extent[axis] = 1;
  }
  return {origin, extent};
}

template <int Dim>
std::shared_ptr<const pops::PreparedLoadBalanceAuthority<Dim>> load_balance() {
  return std::make_shared<const pops::PreparedLoadBalanceAuthority<Dim>>(
      pops::prepare_load_balance_authority<Dim>(
          "space_filling_curve", "test.amr-runtime.sfc",
          pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
}

template <int Dim>
hierarchy::LevelLayout<Dim> coarse_layout(
    const Box<Dim>& domain, const pops::PreparedLoadBalanceAuthority<Dim>& authority) {
  Extent<Dim> tile{};
  for (int axis = 0; axis < Dim; ++axis)
    tile[axis] = 4;
  const mesh::BoxArray<Dim> patches = mesh::BoxArray<Dim>::from_domain(domain, tile);
  const auto ownership = authority.prepare(patches, serial_rank_space<Dim>(), kLoadBalanceBudget);
  return hierarchy::LevelLayout<Dim>(0, domain, patches, ownership.plan().distribution(),
                                     pops::amr::RefinementRatio<Dim>{}, kLayoutBudget);
}

template <int Dim>
tagging::ClusterResult<Dim> cluster_result(const hierarchy::LevelLayout<Dim>& parent,
                                           std::vector<Box<Dim>> boxes) {
  const mesh::BoxArray<Dim> clustered(std::move(boxes));
  tagging::ClusterOptions<Dim> options;
  options.min_efficiency = 0.7;
  for (int axis = 0; axis < Dim; ++axis) {
    options.min_box_size[static_cast<std::size_t>(axis)] = 1;
    options.max_box_size[static_cast<std::size_t>(axis)] = 8;
  }
  options.budget = tagging::ClusterWorkBudget{8, 128, 4096, 32, 1U << 20};
  tagging::ClusterResultIdentity<Dim> identity{
      "test.cluster.exact", parent.exact_identity(), options, {}, clustered.boxes()};
  return {clustered, std::move(identity)};
}

}  // namespace

TEST(test_nd_amr_runtime, anisotropic_regrid_publishes_one_authenticated_spatial_contract) {
  const auto authority = load_balance<3>();
  const Box<3> domain{Index<3>{-2, 4, 7}, Index<3>{1, 7, 8}};
  auto coarse = coarse_layout(domain, *authority);
  const Index<3> local_rank = serial_rank_space<3>().origin();
  MultiFab<3> coarse_state(coarse.patches(), coarse.distribution(), local_rank, 2,
                           Extent<3>{1, 1, 1});
  auto hierarchy_state =
      hierarchy::AmrHierarchy<3>::from_coarse(coarse, std::move(coarse_state), kHierarchyBudget);
  runtime::AmrRuntime<3> engine(std::move(hierarchy_state), authority, "test.amr-runtime.spatial");
  const std::string initial_contract(engine.spatial_contract());

  const pops::amr::RefinementRatio<3> ratio{2, 3, 1};
  const Box<3> parent_patch{Index<3>{-2, 4, 7}, Index<3>{-1, 5, 8}};
  const regridding::RegridPreparationBudget budget{
      .clustered_parent_layout = kLayoutBudget,
      .fine_layout = kLayoutBudget,
      .load_balance = kLoadBalanceBudget,
  };
  auto prepared = engine.prepare_regrid(
      0, ratio, cluster_result(engine.hierarchy().layout(0), {parent_patch}), budget);
  ASSERT_FALSE(prepared.removes_fine_level());
  ASSERT_TRUE(prepared.fine_layout().has_value());
  MultiFab<3> child_state(prepared.fine_layout()->patches(), prepared.fine_layout()->distribution(),
                          local_rank, 2, Extent<3>{1, 1, 1});

  engine.publish_regrid(0, std::move(prepared), std::move(child_state));

  ASSERT_EQ(engine.hierarchy().num_levels(), 2U);
  EXPECT_EQ(engine.hierarchy().layout(1).domain(), hierarchy::refine_box(domain, ratio));
  EXPECT_EQ(engine.hierarchy().layout(1).ratio_from_parent(), ratio);
  EXPECT_EQ(engine.topology_epoch(), 1U);
  EXPECT_EQ(engine.materialization_generation(), 1U);
  EXPECT_NE(engine.spatial_contract(), initial_contract);
  EXPECT_EQ(engine.hierarchy().state(1).distribution(),
            engine.hierarchy().layout(1).distribution());
}

TEST(test_nd_amr_runtime, empty_regrid_truncates_finer_state_transactionally) {
  const auto authority = load_balance<1>();
  const Box<1> domain{Index<1>{-4}, Index<1>{3}};
  auto coarse = coarse_layout(domain, *authority);
  const Index<1> local_rank = serial_rank_space<1>().origin();
  MultiFab<1> coarse_state(coarse.patches(), coarse.distribution(), local_rank, 1, Extent<1>{0});
  runtime::AmrRuntime<1> engine(
      hierarchy::AmrHierarchy<1>::from_coarse(coarse, std::move(coarse_state), kHierarchyBudget),
      authority, "test.amr-runtime.truncate");
  const regridding::RegridPreparationBudget budget{
      .clustered_parent_layout = kLayoutBudget,
      .fine_layout = kLayoutBudget,
      .load_balance = kLoadBalanceBudget,
  };

  auto populated = engine.prepare_regrid(
      0, pops::amr::RefinementRatio<1>{2},
      cluster_result(engine.hierarchy().layout(0), {Box<1>{Index<1>{-4}, Index<1>{-1}}}), budget);
  MultiFab<1> child(populated.fine_layout()->patches(), populated.fine_layout()->distribution(),
                    local_rank, 1, Extent<1>{0});
  engine.publish_regrid(0, std::move(populated), std::move(child));
  ASSERT_EQ(engine.hierarchy().num_levels(), 2U);
  const std::uint64_t populated_epoch = engine.topology_epoch();

  auto empty = engine.prepare_regrid(0, pops::amr::RefinementRatio<1>{2},
                                     cluster_result<1>(engine.hierarchy().layout(0), {}), budget);
  ASSERT_TRUE(empty.removes_fine_level());
  engine.publish_regrid(0, std::move(empty), std::nullopt);

  EXPECT_EQ(engine.hierarchy().num_levels(), 1U);
  EXPECT_EQ(engine.topology_epoch(), populated_epoch + 1);
}

TEST(test_nd_amr_runtime, level_state_rejects_storage_from_another_patch_layout) {
  const auto authority = load_balance<1>();
  const Box<1> domain{Index<1>{0}, Index<1>{3}};
  auto layout = coarse_layout(domain, *authority);
  const mesh::BoxArray<1> other_patches(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{1}}});
  const auto other_ownership =
      authority->prepare(other_patches, serial_rank_space<1>(), kLoadBalanceBudget);
  MultiFab<1> other_state(other_patches, other_ownership.plan().distribution(),
                          serial_rank_space<1>().origin(), 1, Extent<1>{0});

  EXPECT_THROW((void)hierarchy::AmrLevelState<1>(layout, std::move(other_state)),
               std::invalid_argument);
}
