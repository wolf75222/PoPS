#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace hierarchy = pops::amr::hierarchy;
namespace tagging = pops::amr::tagging;

constexpr pops::mesh::BoxArrayValidationBudget kLayoutBudget{32, 496};
constexpr hierarchy::HierarchyValidationBudget kHierarchyBudget{4, 1024};
constexpr pops::parallel::LoadBalancePreparationBudget kLoadBalanceBudget{
    32, 8, std::numeric_limits<std::int64_t>::max()};

template <int Dim>
pops::mesh::RankSpace<Dim> serial_rank_space() {
  pops::Index<Dim> origin{};
  pops::Extent<Dim> extent{};
  for (int axis = 0; axis < Dim; ++axis)
    extent[axis] = 1;
  return {origin, extent};
}

template <int Dim>
std::shared_ptr<const pops::PreparedLoadBalanceAuthority<Dim>> load_balance() {
  return std::make_shared<const pops::PreparedLoadBalanceAuthority<Dim>>(
      pops::prepare_load_balance_authority<Dim>(
          "space_filling_curve", "test.regrid-union.sfc",
          pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
}

template <int Dim>
hierarchy::LevelLayout<Dim> coarse_layout(
    const pops::Box<Dim>& domain, const pops::PreparedLoadBalanceAuthority<Dim>& authority) {
  pops::Extent<Dim> tile{};
  for (int axis = 0; axis < Dim; ++axis)
    tile[axis] = 8;
  const pops::mesh::BoxArray<Dim> patches = pops::mesh::BoxArray<Dim>::from_domain(domain, tile);
  const auto ownership = authority.prepare(patches, serial_rank_space<Dim>(), kLoadBalanceBudget);
  return hierarchy::LevelLayout<Dim>(0, domain, patches, ownership.plan().distribution(),
                                     pops::amr::RefinementRatio<Dim>{}, kLayoutBudget);
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
    options.max_box_size[static_cast<std::size_t>(axis)] = 8;
  }
  options.budget = {16, 256, 8192, 64, 1U << 20};
  tagging::ClusterResultIdentity<Dim> exact_identity{
      std::move(identity), parent.exact_identity(), options, {}, clustered.boxes()};
  return {clustered, std::move(exact_identity)};
}

template <int Dim>
pops::Box<Dim> uniform_box(int lower, int upper) {
  pops::Index<Dim> lo{};
  pops::Index<Dim> hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    lo[axis] = lower;
    hi[axis] = upper;
  }
  return {lo, hi};
}

template <int Dim>
struct RuntimeFixture {
  std::shared_ptr<const pops::PreparedLoadBalanceAuthority<Dim>> authority = load_balance<Dim>();
  pops::Index<Dim> local_rank = serial_rank_space<Dim>().origin();
  pops::runtime::amr::AmrRuntime<Dim> engine;

  RuntimeFixture() : engine(make_hierarchy(), authority, "test.regrid-union.spatial") {}

  hierarchy::AmrHierarchy<Dim> make_hierarchy() const {
    const auto coarse = coarse_layout(uniform_box<Dim>(0, 15), *authority);
    pops::MultiFab<Dim> state(coarse.patches(), coarse.distribution(), local_rank, 1,
                              pops::Extent<Dim>{});
    state.set_val(pops::Real(1));
    return hierarchy::AmrHierarchy<Dim>::from_coarse(coarse, std::move(state), kHierarchyBudget);
  }

  static pops::amr::regridding::RegridPreparationBudget budget() {
    return {.clustered_parent_layout = kLayoutBudget,
            .fine_layout = kLayoutBudget,
            .load_balance = kLoadBalanceBudget};
  }

  pops::MultiFab<Dim> child_for(const pops::amr::regridding::PreparedRegrid<Dim>& prepared) const {
    return pops::MultiFab<Dim>(prepared.fine_layout()->patches(),
                               prepared.fine_layout()->distribution(), local_rank, 1,
                               pops::Extent<Dim>{});
  }
};

template <int Dim>
bool contains_box(const pops::mesh::BoxArray<Dim>& boxes, const pops::Box<Dim>& expected) {
  return std::find(boxes.boxes().begin(), boxes.boxes().end(), expected) != boxes.boxes().end();
}

}  // namespace

TEST(test_amr_multiblock_regrid_union, PublishesUnionOfDisjointTaggedSubjects) {
  constexpr int Dim = pops::kNativeDimension;
  RuntimeFixture<Dim> fixture;
  const pops::Box<Dim> subject_a = uniform_box<Dim>(2, 5);
  const pops::Box<Dim> subject_b = uniform_box<Dim>(10, 13);
  std::array<int, Dim> ratio_components{};
  ratio_components.fill(2);
  const pops::amr::RefinementRatio<Dim> ratio(ratio_components);

  auto prepared = fixture.engine.prepare_regrid(
      0, ratio,
      cluster_result(fixture.engine.hierarchy().layout(0), {subject_a, subject_b},
                     "test.regrid-union.subjects-a-b"),
      RuntimeFixture<Dim>::budget());
  ASSERT_TRUE(prepared.fine_layout().has_value());
  ASSERT_FALSE(prepared.removes_fine_level());
  pops::MultiFab<Dim> child = fixture.child_for(prepared);
  child.set_val(pops::Real(7));

  fixture.engine.publish_regrid(0, std::move(prepared), std::move(child));

  ASSERT_EQ(fixture.engine.hierarchy().num_levels(), 2U);
  const auto& fine = fixture.engine.hierarchy().layout(1).patches();
  EXPECT_TRUE(contains_box(fine, hierarchy::refine_box(subject_a, ratio)));
  EXPECT_TRUE(contains_box(fine, hierarchy::refine_box(subject_b, ratio)));
  EXPECT_EQ(pops::reduce_min_local(fixture.engine.hierarchy().state(1)), pops::Real(7));
  EXPECT_EQ(pops::reduce_max_local(fixture.engine.hierarchy().state(1)), pops::Real(7));
  EXPECT_EQ(fixture.engine.topology_epoch(), 1U);
}

TEST(test_amr_multiblock_regrid_union, RefusesStaleUnionWithoutChangingPublishedHierarchy) {
  constexpr int Dim = pops::kNativeDimension;
  RuntimeFixture<Dim> fixture;
  std::array<int, Dim> ratio_components{};
  ratio_components.fill(2);
  const pops::amr::RefinementRatio<Dim> ratio(ratio_components);

  auto accepted = fixture.engine.prepare_regrid(
      0, ratio,
      cluster_result(fixture.engine.hierarchy().layout(0), {uniform_box<Dim>(2, 5)},
                     "test.regrid-union.accepted"),
      RuntimeFixture<Dim>::budget());
  pops::MultiFab<Dim> accepted_child = fixture.child_for(accepted);
  accepted_child.set_val(pops::Real(3));
  fixture.engine.publish_regrid(0, std::move(accepted), std::move(accepted_child));

  auto stale = fixture.engine.prepare_regrid(
      1, ratio,
      cluster_result(fixture.engine.hierarchy().layout(1), {uniform_box<Dim>(6, 9)},
                     "test.regrid-union.stale-child"),
      RuntimeFixture<Dim>::budget());
  pops::MultiFab<Dim> stale_child = fixture.child_for(stale);
  stale_child.set_val(pops::Real(5));

  auto replacement = fixture.engine.prepare_regrid(
      0, ratio,
      cluster_result(fixture.engine.hierarchy().layout(0), {uniform_box<Dim>(10, 13)},
                     "test.regrid-union.replacement"),
      RuntimeFixture<Dim>::budget());
  pops::MultiFab<Dim> replacement_child = fixture.child_for(replacement);
  replacement_child.set_val(pops::Real(9));
  fixture.engine.publish_regrid(0, std::move(replacement), std::move(replacement_child));
  const std::string accepted_contract(fixture.engine.spatial_contract());
  const std::uint64_t accepted_epoch = fixture.engine.topology_epoch();

  EXPECT_THROW(fixture.engine.publish_regrid(1, std::move(stale), std::move(stale_child)),
               std::invalid_argument);
  EXPECT_EQ(fixture.engine.spatial_contract(), accepted_contract);
  EXPECT_EQ(fixture.engine.topology_epoch(), accepted_epoch);
  EXPECT_EQ(pops::reduce_min_local(fixture.engine.hierarchy().state(1)), pops::Real(9));
  EXPECT_EQ(pops::reduce_max_local(fixture.engine.hierarchy().state(1)), pops::Real(9));
}
