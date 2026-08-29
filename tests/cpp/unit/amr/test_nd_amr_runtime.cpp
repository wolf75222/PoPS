#include <gtest/gtest.h>

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr/prepared_multiblock_hierarchy.hpp>

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
mesh::RankSpace<Dim> execution_rank_space(const pops::ExecutionLane& lane) {
  Index<Dim> origin{};
  Extent<Dim> extent{};
  for (int axis = 0; axis < Dim; ++axis)
    extent[axis] = 1;
  extent[0] = lane.size();
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
    const Box<Dim>& domain, const pops::PreparedLoadBalanceAuthority<Dim>& authority,
    const mesh::RankSpace<Dim>& rank_space = serial_rank_space<Dim>()) {
  Extent<Dim> tile{};
  for (int axis = 0; axis < Dim; ++axis)
    tile[axis] = 4;
  const mesh::BoxArray<Dim> patches = mesh::BoxArray<Dim>::from_domain(domain, tile);
  const auto ownership = authority.prepare(patches, rank_space, kLoadBalanceBudget);
  return hierarchy::LevelLayout<Dim>(0, domain, patches, ownership.plan().distribution(),
                                     pops::amr::RefinementRatio<Dim>{}, kLayoutBudget);
}

template <int Dim>
hierarchy::LevelLayout<Dim> coarse_layout(const Box<Dim>& domain,
                                          const pops::PreparedLoadBalanceAuthority<Dim>& authority,
                                          const mesh::RankSpace<Dim>& rank_space,
                                          const pops::ExecutionLane& lane) {
  Extent<Dim> tile{};
  for (int axis = 0; axis < Dim; ++axis)
    tile[axis] = 4;
  const mesh::BoxArray<Dim> patches = mesh::BoxArray<Dim>::from_domain(domain, tile);
  const auto ownership = authority.prepare(patches, rank_space, kLoadBalanceBudget, {}, lane);
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

void consume_restore_publication_twice() {
  const auto authority = load_balance<1>();
  const Box<1> domain{Index<1>{0}, Index<1>{3}};
  auto coarse = coarse_layout(domain, *authority);
  MultiFab<1> state(coarse.patches(), coarse.distribution(), serial_rank_space<1>().origin(), 1,
                    Extent<1>{0});
  runtime::AmrRuntime<1> engine(
      hierarchy::AmrHierarchy<1>::from_coarse(coarse, std::move(state), kHierarchyBudget),
      authority, "test.amr-runtime.restore-consume");
  auto publication = engine.prepare_restore_publication(engine.snapshot());
  engine.authenticate_prepared_restore_publication(publication);
  engine.publish_authenticated_restore_noexcept(std::move(publication));
  engine.publish_authenticated_restore_noexcept(std::move(publication));
}

void consume_regrid_publication_twice() {
  const auto authority = load_balance<1>();
  const Box<1> domain{Index<1>{0}, Index<1>{3}};
  auto coarse = coarse_layout(domain, *authority);
  const Index<1> local_rank = serial_rank_space<1>().origin();
  MultiFab<1> state(coarse.patches(), coarse.distribution(), local_rank, 1, Extent<1>{0});
  runtime::AmrRuntime<1> engine(
      hierarchy::AmrHierarchy<1>::from_coarse(coarse, std::move(state), kHierarchyBudget),
      authority, "test.amr-runtime.regrid-consume");
  const auto prepared = engine.prepare_regrid(
      0, pops::amr::RefinementRatio<1>{2},
      cluster_result(engine.hierarchy().layout(0), {Box<1>{Index<1>{0}, Index<1>{1}}}),
      regridding::RegridPreparationBudget{.clustered_parent_layout = kLayoutBudget,
                                          .fine_layout = kLayoutBudget,
                                          .load_balance = kLoadBalanceBudget});
  MultiFab<1> child(prepared.fine_layout()->patches(), prepared.fine_layout()->distribution(),
                    local_rank, 1, Extent<1>{0});
  auto publication = engine.prepare_regrid_publication(0, prepared, std::move(child));
  engine.authenticate_prepared_regrid_publication(publication);
  engine.publish_authenticated_regrid_noexcept(std::move(publication));
  engine.publish_authenticated_regrid_noexcept(std::move(publication));
}

template <int Dim>
struct ConservativeExchange {
  pops::FieldView<pops::Real, Dim> first{};
  pops::FieldView<pops::Real, Dim> second{};
  pops::Real dt = pops::Real(0);

  POPS_HD void operator()(const Index<Dim>& cell) const {
    const pops::Real amount = dt * pops::Real(0.25) * first(cell);
    first(cell) -= amount;
    second(cell) += amount;
  }
};

template <int Dim>
void prove_prepared_multiblock_hierarchy() {
  using MultiBlock = pops::runtime::amr::PreparedMultiBlockAmrHierarchy<Dim>;
  pops::ExecutionLane parent = pops::ExecutionLane::duplicate_world_collectively(
      "test.prepared-multiblock-amr.parent." + std::to_string(Dim));
  const auto authority = load_balance<Dim>();
  Index<Dim> lower{};
  Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = 3;
  const auto ranks = execution_rank_space<Dim>(parent);
  upper[0] = 4 * static_cast<int>(ranks.size()) - 1;
  const Box<Dim> domain{lower, upper};
  auto coarse = coarse_layout(domain, *authority, ranks, parent);
  const Index<Dim> local_rank = ranks.coordinate(static_cast<std::size_t>(parent.rank()));
  MultiFab<Dim> primary_state(coarse.patches(), coarse.distribution(), local_rank, 1,
                              Extent<Dim>{});
  primary_state.set_val(pops::Real(1));
  pops::runtime::amr::AmrRuntime<Dim> engine(
      hierarchy::AmrHierarchy<Dim>::from_coarse(coarse, std::move(primary_state), kHierarchyBudget),
      authority, "test.prepared-multiblock-amr.spatial." + std::to_string(Dim));

  MultiFab<Dim> secondary_state(engine.hierarchy().state(0).layout(),
                                engine.hierarchy().state(0).distribution(), local_rank, 1,
                                Extent<Dim>{});
  secondary_state.set_val(pops::Real(3));
  std::vector<typename MultiBlock::AdditionalBlock> additional;
  additional.push_back({"neutral", {std::move(secondary_state)}});
  MultiBlock prepared =
      MultiBlock::prepare_collectively(parent, std::move(engine), "ion", std::move(additional),
                                       "test.prepared-multiblock-amr.lane." + std::to_string(Dim));

  pops::CouplingOperatorView view;
  view.label = "conservative-exchange";
  view.conservation.conserved_roles = {"density"};
  prepared.install_prepared_coupling_operator(
      "test.prepared-multiblock-amr.exchange.v1", view,
      [](pops::Real dt, const std::vector<MultiFab<Dim>*>& states) {
        MultiFab<Dim>& first = *states[0];
        MultiFab<Dim>& second = *states[1];
        for (std::size_t local = 0; local < first.local_size(); ++local)
          pops::for_each_cell(
              first.box(local),
              ConservativeExchange<Dim>{first.fab(local).view(), second.fab(local).view(), dt});
        if (dt > pops::Real(0.5))
          throw std::runtime_error("deliberate post-kernel provider failure");
      });

  MultiFab<Dim> unsealed_primary(prepared.state(0, 0));
  MultiFab<Dim> unsealed_secondary(prepared.state(1, 0));
  std::vector<MultiFab<Dim>*> unsealed_candidates{&unsealed_primary, &unsealed_secondary};
  EXPECT_THROW(prepared.apply_coupling_operators_at_level(0, pops::Real(0.1), unsealed_candidates),
               std::exception);
  prepared.seal_couplings();
  EXPECT_FALSE(prepared.coupling_registry_contract().empty());

  EXPECT_EQ(prepared.apply_and_publish_level(0, pops::Real(0.4)), 1U);
  EXPECT_EQ(pops::reduce_min_local(prepared.state(0, 0)), pops::Real(0.9));
  EXPECT_EQ(pops::reduce_max_local(prepared.state(0, 0)), pops::Real(0.9));
  EXPECT_EQ(pops::reduce_min_local(prepared.state(1, 0)), pops::Real(3.1));
  EXPECT_EQ(pops::reduce_max_local(prepared.state(1, 0)), pops::Real(3.1));
  EXPECT_EQ(prepared.accepted_revision(), 1U);

  const std::vector<std::string> reverse_order{"neutral", "ion"};
  const auto reverse_map = prepared.prepare_program_block_map(reverse_order);
  MultiFab<Dim> neutral_candidate(prepared.state(1, 0));
  MultiFab<Dim> ion_candidate(prepared.state(0, 0));
  std::vector<MultiFab<Dim>*> reverse_candidates{&neutral_candidate, &ion_candidate};
  const pops::runtime::multiblock::BoundaryEvaluationPoint point{
      "test.prepared-multiblock-amr.direct", 0, 0, 0, 0, {0, 1}, 0.1, 0.0, {}, {}, {}};
  EXPECT_EQ(prepared.apply_program_candidates(reverse_map, 0, pops::Real(0.1), reverse_candidates,
                                              point, nullptr),
            1U);
  EXPECT_NEAR(pops::reduce_min_local(ion_candidate), pops::Real(0.8775), 1e-14);
  EXPECT_NEAR(pops::reduce_min_local(neutral_candidate), pops::Real(3.1225), 1e-14);
  EXPECT_EQ(pops::reduce_min_local(prepared.state(0, 0)), pops::Real(0.9))
      << "Program candidates stay private until group publication";
  auto forged_map = reverse_map;
  forged_map.exact_contract += "forged";
  EXPECT_THROW(prepared.apply_program_candidates(forged_map, 0, pops::Real(0.1), reverse_candidates,
                                                 point, nullptr),
               std::exception);

  EXPECT_THROW(prepared.apply_and_publish_level(0, pops::Real(0.8)), std::runtime_error);
  EXPECT_EQ(pops::reduce_min_local(prepared.state(0, 0)), pops::Real(0.9));
  EXPECT_EQ(pops::reduce_max_local(prepared.state(1, 0)), pops::Real(3.1));
  EXPECT_EQ(prepared.accepted_revision(), 1U)
      << "a failed device provider never publishes a partial block pack";

  const std::string coarse_coupling_contract(prepared.coupling_registry_contract());

  std::array<int, Dim> ratio_components{};
  ratio_components.fill(2);
  const pops::amr::RefinementRatio<Dim> ratio(ratio_components);
  auto regrid = prepared.topology_runtime().prepare_regrid(
      0, ratio,
      cluster_result(prepared.topology_runtime().hierarchy().layout(0),
                     prepared.topology_runtime().hierarchy().layout(0).patches().boxes()),
      regridding::RegridPreparationBudget{.clustered_parent_layout = kLayoutBudget,
                                          .fine_layout = kLayoutBudget,
                                          .load_balance = kLoadBalanceBudget},
      prepared.lane());
  ASSERT_TRUE(regrid.fine_layout());
  MultiFab<Dim> primary_child(regrid.fine_layout()->patches(), regrid.fine_layout()->distribution(),
                              local_rank, 1, Extent<Dim>{});
  MultiFab<Dim> secondary_child(regrid.fine_layout()->patches(),
                                regrid.fine_layout()->distribution(), local_rank, 1, Extent<Dim>{});
  primary_child.set_val(pops::Real(5));
  secondary_child.set_val(pops::Real(7));
  std::vector<std::optional<MultiFab<Dim>>> children;
  children.emplace_back(std::move(primary_child));
  children.emplace_back(std::move(secondary_child));
  typename MultiBlock::PreparedRegridTransactionStack regrid_stack(2);
  regrid_stack.execute_and_publish(
      prepared.prepare_regrid_transaction(0, std::move(regrid), std::move(children)));
  ASSERT_EQ(regrid_stack.published_transitions(), 1U);
  ASSERT_TRUE(regrid_stack.transaction(0).changes_topology());
  ASSERT_TRUE(regrid_stack.transaction(0).candidate_published());
  ASSERT_FALSE(regrid_stack.transaction(0).inverse_consumed());

  ASSERT_EQ(prepared.level_count(), 2U);
  EXPECT_EQ(pops::reduce_min_local(prepared.state(0, 1)), pops::Real(5));
  EXPECT_EQ(pops::reduce_min_local(prepared.state(1, 1)), pops::Real(7));
  EXPECT_EQ(prepared.state(0, 1).layout(), prepared.state(1, 1).layout());
  EXPECT_EQ(prepared.state(0, 1).distribution(), prepared.state(1, 1).distribution());
  EXPECT_NE(prepared.coupling_registry_contract(), coarse_coupling_contract)
      << "the sealed provider registry is rebound to the new exact topology";
  const std::string first_candidate_contract(prepared.collective_contract());

  // A finite stack rejects its next transition before authenticating or publishing it.  The
  // rejected over-budget authority therefore cannot clobber the first candidate topology.
  typename MultiBlock::PreparedRegridTransactionStack budget_stack(1);
  auto budget_regrid = prepared.topology_runtime().prepare_regrid(
      1, ratio,
      cluster_result(prepared.topology_runtime().hierarchy().layout(1),
                     prepared.topology_runtime().hierarchy().layout(1).patches().boxes()),
      regridding::RegridPreparationBudget{.clustered_parent_layout = kLayoutBudget,
                                          .fine_layout = kLayoutBudget,
                                          .load_balance = kLoadBalanceBudget},
      prepared.lane());
  MultiFab<Dim> budget_primary(budget_regrid.fine_layout()->patches(),
                               budget_regrid.fine_layout()->distribution(), local_rank, 1,
                               Extent<Dim>{});
  MultiFab<Dim> budget_secondary(budget_regrid.fine_layout()->patches(),
                                 budget_regrid.fine_layout()->distribution(), local_rank, 1,
                                 Extent<Dim>{});
  std::vector<std::optional<MultiFab<Dim>>> budget_children;
  budget_children.emplace_back(std::move(budget_primary));
  budget_children.emplace_back(std::move(budget_secondary));
  budget_stack.execute_and_publish(
      prepared.prepare_regrid_transaction(1, std::move(budget_regrid), std::move(budget_children)));
  const std::string budget_candidate_contract(prepared.collective_contract());
  const std::size_t budget_candidate_levels = prepared.level_count();
  auto over_budget_regrid = prepared.topology_runtime().prepare_regrid(
      2, ratio,
      cluster_result(prepared.topology_runtime().hierarchy().layout(2),
                     prepared.topology_runtime().hierarchy().layout(2).patches().boxes()),
      regridding::RegridPreparationBudget{.clustered_parent_layout = kLayoutBudget,
                                          .fine_layout = kLayoutBudget,
                                          .load_balance = kLoadBalanceBudget},
      prepared.lane());
  MultiFab<Dim> over_budget_primary(over_budget_regrid.fine_layout()->patches(),
                                    over_budget_regrid.fine_layout()->distribution(), local_rank, 1,
                                    Extent<Dim>{});
  MultiFab<Dim> over_budget_secondary(over_budget_regrid.fine_layout()->patches(),
                                      over_budget_regrid.fine_layout()->distribution(), local_rank,
                                      1, Extent<Dim>{});
  std::vector<std::optional<MultiFab<Dim>>> over_budget_children;
  over_budget_children.emplace_back(std::move(over_budget_primary));
  over_budget_children.emplace_back(std::move(over_budget_secondary));
  EXPECT_THROW(budget_stack.execute_and_publish(prepared.prepare_regrid_transaction(
                   2, std::move(over_budget_regrid), std::move(over_budget_children))),
               std::logic_error);
  EXPECT_EQ(prepared.level_count(), budget_candidate_levels);
  EXPECT_EQ(prepared.collective_contract(), budget_candidate_contract);
  budget_stack.publish_inverse_lifo_noexcept();
  budget_stack.reset_for_next_candidate_noexcept();
  EXPECT_EQ(prepared.collective_contract(), first_candidate_contract);

  // A three-level hierarchy publishes two distinct prepared transitions in one candidate and
  // then restores them LIFO.  No transition has a global singleton budget: the cold-bound stack
  // owns exactly the finite maximum supplied by the enclosing step authority.
  auto second_regrid = prepared.topology_runtime().prepare_regrid(
      1, ratio,
      cluster_result(prepared.topology_runtime().hierarchy().layout(1),
                     prepared.topology_runtime().hierarchy().layout(1).patches().boxes()),
      regridding::RegridPreparationBudget{.clustered_parent_layout = kLayoutBudget,
                                          .fine_layout = kLayoutBudget,
                                          .load_balance = kLoadBalanceBudget},
      prepared.lane());
  ASSERT_TRUE(second_regrid.fine_layout());
  MultiFab<Dim> second_primary_child(second_regrid.fine_layout()->patches(),
                                     second_regrid.fine_layout()->distribution(), local_rank, 1,
                                     Extent<Dim>{});
  MultiFab<Dim> second_secondary_child(second_regrid.fine_layout()->patches(),
                                       second_regrid.fine_layout()->distribution(), local_rank, 1,
                                       Extent<Dim>{});
  second_primary_child.set_val(pops::Real(11));
  second_secondary_child.set_val(pops::Real(13));
  std::vector<std::optional<MultiFab<Dim>>> second_children;
  second_children.emplace_back(std::move(second_primary_child));
  second_children.emplace_back(std::move(second_secondary_child));
  regrid_stack.execute_and_publish(
      prepared.prepare_regrid_transaction(1, std::move(second_regrid), std::move(second_children)));
  ASSERT_EQ(regrid_stack.published_transitions(), 2U);
  ASSERT_EQ(prepared.level_count(), 3U);
  EXPECT_EQ(pops::reduce_min_local(prepared.state(0, 2)), pops::Real(11));
  EXPECT_EQ(pops::reduce_min_local(prepared.state(1, 2)), pops::Real(13));

  auto invalid_snapshot = prepared.snapshot();
  invalid_snapshot.additional[0].identity = "another-block";
  EXPECT_THROW(prepared.restore(invalid_snapshot), std::exception);
  EXPECT_EQ(prepared.level_count(), 3U)
      << "snapshot validation finishes before the first live topology mutation";

  auto invalid_shape_snapshot = prepared.snapshot();
  MultiFab<Dim> wrong_shape(prepared.state(1, 0).layout(), prepared.state(1, 0).distribution(),
                            local_rank, 2, prepared.state(1, 0).ghosts());
  invalid_shape_snapshot.additional[0].levels[0] = std::move(wrong_shape);
  EXPECT_THROW(prepared.restore(invalid_shape_snapshot), std::exception);
  EXPECT_EQ(prepared.level_count(), 3U);

  // Both prebuilt inverses are consumed in reverse order.  The stack itself owns no unbounded
  // registration path; a third transition would be refused before its forward publication.
  regrid_stack.publish_inverse_lifo_noexcept();
  EXPECT_TRUE(regrid_stack.transaction(0).inverse_consumed());
  EXPECT_TRUE(regrid_stack.transaction(1).inverse_consumed());

  ASSERT_DEATH(regrid_stack.publish_inverse_lifo_noexcept(), "");
  ASSERT_DEATH(regrid_stack.discard_after_accept_noexcept(), "");

  EXPECT_EQ(prepared.level_count(), 1U);
  EXPECT_EQ(prepared.accepted_revision(), 1U);
  EXPECT_EQ(pops::reduce_min_local(prepared.state(0, 0)), pops::Real(0.9));
  EXPECT_EQ(pops::reduce_min_local(prepared.state(1, 0)), pops::Real(3.1));
  EXPECT_EQ(prepared.coupling_registry_contract(), coarse_coupling_contract);

  // Rollback keeps the cold capacity, so an identical retry may be accepted and retired. A later
  // transition proves that acceptance clears consumed authorities without exhausting the next
  // candidate's finite budget.
  regrid_stack.reset_for_next_candidate_noexcept();
  EXPECT_EQ(regrid_stack.published_transitions(), 0U);
  auto retry_regrid = prepared.topology_runtime().prepare_regrid(
      0, ratio,
      cluster_result(prepared.topology_runtime().hierarchy().layout(0),
                     prepared.topology_runtime().hierarchy().layout(0).patches().boxes()),
      regridding::RegridPreparationBudget{.clustered_parent_layout = kLayoutBudget,
                                          .fine_layout = kLayoutBudget,
                                          .load_balance = kLoadBalanceBudget},
      prepared.lane());
  MultiFab<Dim> retry_primary(retry_regrid.fine_layout()->patches(),
                              retry_regrid.fine_layout()->distribution(), local_rank, 1,
                              Extent<Dim>{});
  MultiFab<Dim> retry_secondary(retry_regrid.fine_layout()->patches(),
                                retry_regrid.fine_layout()->distribution(), local_rank, 1,
                                Extent<Dim>{});
  retry_primary.set_val(pops::Real(5));
  retry_secondary.set_val(pops::Real(7));
  std::vector<std::optional<MultiFab<Dim>>> retry_children;
  retry_children.emplace_back(std::move(retry_primary));
  retry_children.emplace_back(std::move(retry_secondary));
  regrid_stack.execute_and_publish(
      prepared.prepare_regrid_transaction(0, std::move(retry_regrid), std::move(retry_children)));
  EXPECT_EQ(prepared.collective_contract(), first_candidate_contract);
  regrid_stack.discard_after_accept_noexcept();
  EXPECT_EQ(regrid_stack.published_transitions(), 0U);

  auto post_accept_regrid = prepared.topology_runtime().prepare_regrid(
      1, ratio,
      cluster_result(prepared.topology_runtime().hierarchy().layout(1),
                     prepared.topology_runtime().hierarchy().layout(1).patches().boxes()),
      regridding::RegridPreparationBudget{.clustered_parent_layout = kLayoutBudget,
                                          .fine_layout = kLayoutBudget,
                                          .load_balance = kLoadBalanceBudget},
      prepared.lane());
  MultiFab<Dim> post_accept_primary(post_accept_regrid.fine_layout()->patches(),
                                    post_accept_regrid.fine_layout()->distribution(), local_rank, 1,
                                    Extent<Dim>{});
  MultiFab<Dim> post_accept_secondary(post_accept_regrid.fine_layout()->patches(),
                                      post_accept_regrid.fine_layout()->distribution(), local_rank,
                                      1, Extent<Dim>{});
  post_accept_primary.set_val(pops::Real(17));
  post_accept_secondary.set_val(pops::Real(19));
  std::vector<std::optional<MultiFab<Dim>>> post_accept_children;
  post_accept_children.emplace_back(std::move(post_accept_primary));
  post_accept_children.emplace_back(std::move(post_accept_secondary));
  regrid_stack.execute_and_publish(prepared.prepare_regrid_transaction(
      1, std::move(post_accept_regrid), std::move(post_accept_children)));
  EXPECT_EQ(prepared.level_count(), 3U);
  regrid_stack.publish_inverse_lifo_noexcept();
  regrid_stack.reset_for_next_candidate_noexcept();
  EXPECT_EQ(prepared.level_count(), 2U);
}

}  // namespace

TEST(test_nd_amr_runtime, prepared_multiblock_hierarchy_is_exact_ranked_in_1d_2d_3d) {
  prove_prepared_multiblock_hierarchy<1>();
  prove_prepared_multiblock_hierarchy<2>();
  prove_prepared_multiblock_hierarchy<3>();
}

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
  const auto accepted_snapshot = engine.snapshot();
  auto stale_restore = engine.prepare_restore_publication(accepted_snapshot);
  engine.authenticate_prepared_restore_publication(stale_restore);
  auto stale_regrid = engine.prepare_regrid(
      0, ratio, cluster_result(engine.hierarchy().layout(0), {parent_patch}), budget);
  MultiFab<3> stale_child(stale_regrid.fine_layout()->patches(),
                          stale_regrid.fine_layout()->distribution(), local_rank, 2,
                          Extent<3>{1, 1, 1});
  auto stale_publication =
      engine.prepare_regrid_publication(0, stale_regrid, std::move(stale_child));
  engine.authenticate_prepared_regrid_publication(stale_publication);
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

  const std::string accepted_candidate_contract(engine.spatial_contract());
  const std::size_t accepted_candidate_levels = engine.hierarchy().num_levels();
  EXPECT_THROW(engine.authenticate_prepared_restore_publication(stale_restore),
               std::invalid_argument);
  EXPECT_THROW(engine.authenticate_prepared_regrid_publication(stale_publication),
               std::invalid_argument);
  EXPECT_EQ(engine.hierarchy().num_levels(), accepted_candidate_levels);
  EXPECT_EQ(engine.spatial_contract(), accepted_candidate_contract)
      << "stale publications are rejected before the first live swap";
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

TEST(test_nd_amr_runtime, prepared_runtime_publications_are_one_shot_fail_stop_authorities) {
  ASSERT_DEATH(consume_restore_publication_twice(), "");
  ASSERT_DEATH(consume_regrid_publication_twice(), "");
}
