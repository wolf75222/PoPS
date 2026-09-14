#include <gtest/gtest.h>

#include <pops/amr/reflux/metric_reflux.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace hierarchy = pops::amr::hierarchy;
namespace mesh = pops::mesh;
namespace reflux = pops::amr::reflux;
namespace runtime = pops::runtime::amr;
namespace time_amr = pops::numerics::time::amr;

constexpr mesh::BoxArrayValidationBudget kLayoutBudget{32, 496};
constexpr hierarchy::HierarchyValidationBudget kHierarchyBudget{3, 256};
constexpr pops::parallel::LoadBalancePreparationBudget kLoadBalanceBudget{
    32, 8, std::numeric_limits<std::int64_t>::max()};

template <int Dim>
using MultiBlock = runtime::PreparedMultiBlockAmrHierarchy<Dim>;

template <int Dim>
using Engine = time_amr::PreparedMultiBlockAmrSubcyclingEngine<Dim, double>;

template <int Dim>
std::shared_ptr<const pops::PreparedLoadBalanceAuthority<Dim>> load_balance() {
  return std::make_shared<const pops::PreparedLoadBalanceAuthority<Dim>>(
      pops::prepare_load_balance_authority<Dim>(
          "space_filling_curve", "test.synchronized-continuation.sfc",
          pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
}

template <int Dim>
mesh::RankSpace<Dim> execution_rank_space(const pops::ExecutionLane& lane) {
  pops::Index<Dim> origin{};
  pops::Extent<Dim> extent{};
  for (int axis = 0; axis < Dim; ++axis)
    extent[axis] = 1;
  extent[0] = lane.size();
  return {origin, extent};
}

template <int Dim>
pops::amr::RefinementRatio<Dim> spatial_ratio(int transition) {
  std::array<int, Dim> ratio{};
  ratio.fill(2);
  if constexpr (Dim > 1)
    ratio[static_cast<std::size_t>(transition % Dim)] = 3;
  return pops::amr::RefinementRatio<Dim>(ratio);
}

template <int Dim>
hierarchy::LevelLayout<Dim> make_coarse_layout(
    const pops::Box<Dim>& domain, const pops::PreparedLoadBalanceAuthority<Dim>& authority,
    const mesh::RankSpace<Dim>& ranks, const pops::ExecutionLane& lane) {
  pops::Extent<Dim> tile{};
  for (int axis = 0; axis < Dim; ++axis)
    tile[axis] = 4;
  const mesh::BoxArray<Dim> patches = mesh::BoxArray<Dim>::from_domain(domain, tile);
  const auto ownership = authority.prepare(patches, ranks, kLoadBalanceBudget, {}, lane);
  return {0,
          domain,
          patches,
          ownership.plan().distribution(),
          pops::amr::RefinementRatio<Dim>{},
          kLayoutBudget};
}

template <int Dim>
hierarchy::LevelLayout<Dim> make_partial_child_layout(
    const hierarchy::LevelLayout<Dim>& parent, int level,
    const pops::amr::RefinementRatio<Dim>& ratio,
    const pops::PreparedLoadBalanceAuthority<Dim>& authority, const pops::ExecutionLane& lane) {
  pops::Box<Dim> covered = parent.patches().boxes().front();
  covered.hi[0] = covered.lo[0] + (covered.length(0) / 2) - 1;
  const pops::Box<Dim> child_patch = hierarchy::refine_box(covered, ratio);
  const pops::Box<Dim> child_domain = hierarchy::refine_box(parent.domain(), ratio);
  const mesh::BoxArray<Dim> patches(std::vector<pops::Box<Dim>>{child_patch});
  const auto ownership =
      authority.prepare(patches, parent.distribution().rank_space(), kLoadBalanceBudget, {}, lane);
  return {level, child_domain, patches, ownership.plan().distribution(), ratio, kLayoutBudget};
}

template <int Dim>
MultiBlock<Dim> make_hierarchy() {
  pops::ExecutionLane parent = pops::ExecutionLane::duplicate_world_collectively(
      "test.synchronized-continuation.parent." + std::to_string(Dim));
  const auto authority = load_balance<Dim>();
  const auto ranks = execution_rank_space<Dim>(parent);
  pops::Index<Dim> lo{};
  pops::Index<Dim> hi{};
  for (int axis = 0; axis < Dim; ++axis)
    hi[axis] = 3;
  hi[0] = 4 * static_cast<int>(std::max<std::size_t>(2, ranks.size())) - 1;
  const pops::Box<Dim> domain{lo, hi};
  std::vector<hierarchy::LevelLayout<Dim>> layouts;
  layouts.push_back(make_coarse_layout(domain, *authority, ranks, parent));
  layouts.push_back(
      make_partial_child_layout(layouts.back(), 1, spatial_ratio<Dim>(0), *authority, parent));
  layouts.push_back(
      make_partial_child_layout(layouts.back(), 2, spatial_ratio<Dim>(1), *authority, parent));

  const pops::Index<Dim> local_rank = ranks.coordinate(static_cast<std::size_t>(parent.rank()));
  std::vector<hierarchy::AmrLevelState<Dim>> primary_levels;
  std::vector<pops::MultiFab<Dim>> secondary_levels;
  for (const auto& layout : layouts) {
    pops::MultiFab<Dim> primary(layout.patches(), layout.distribution(), local_rank, 1,
                                pops::Extent<Dim>{});
    pops::MultiFab<Dim> secondary(layout.patches(), layout.distribution(), local_rank, 1,
                                  pops::Extent<Dim>{});
    primary.set_val(pops::Real(1));
    secondary.set_val(pops::Real(4));
    primary_levels.emplace_back(layout, std::move(primary));
    secondary_levels.push_back(std::move(secondary));
  }
  runtime::AmrRuntime<Dim> topology(
      hierarchy::AmrHierarchy<Dim>(std::move(primary_levels), kHierarchyBudget), authority,
      "test.synchronized-continuation.spatial." + std::to_string(Dim));
  std::vector<typename MultiBlock<Dim>::AdditionalBlock> additional;
  additional.push_back({"electrons", std::move(secondary_levels)});
  return MultiBlock<Dim>::prepare_collectively(
      parent, std::move(topology), "ions", std::move(additional),
      "test.synchronized-continuation.lane." + std::to_string(Dim));
}

template <int Dim>
auto prepare_engine(MultiBlock<Dim>& hierarchy) {
  const std::vector<pops::amr::ParentChildClockRelation> relations{
      {0, 1, {1, 1}, pops::amr::RemainderPolicy::IntegralOnly},
      {1, 2, {1, 1}, pops::amr::RemainderPolicy::IntegralOnly}};
  return Engine<Dim>::prepare(hierarchy, relations,
                              {{2, {32, 496}}, reflux::FaceFluxLedgerBudget{256, 256, 1}});
}

template <int Dim>
void expect_value(const pops::MultiFab<Dim>& field, pops::Real expected) {
  if (field.local_size() != 0) {
    EXPECT_EQ(pops::reduce_min_local(field), expected);
    EXPECT_EQ(pops::reduce_max_local(field), expected);
  }
}

template <int Dim>
void prove_persistent_candidates() {
  auto hierarchy = make_hierarchy<Dim>();
  auto engine = prepare_engine(hierarchy);
  const pops::amr::ClockWindow window{{0, 0, {0, 1}, 0.0}, {0, 0, {1, 1}, 0.25}};
  const auto revision = hierarchy.accepted_revision();
  const auto generation = hierarchy.topology_runtime().materialization_generation();
  const auto attempt = engine.begin_synchronized(window);
  EXPECT_EQ(attempt, 1U);
  EXPECT_EQ(engine.synchronized_attempt(), attempt);
  EXPECT_TRUE(engine.has_attempt_candidates());
  EXPECT_FALSE(engine.has_synchronized_groups());
  EXPECT_THROW(engine.synchronized_level_group(0), std::logic_error);
  std::array<typename Engine<Dim>::LevelAdvanceContext*, 3> group_addresses{};
  int callbacks = 0;
  engine.resume_synchronized([&](auto root) {
    ++callbacks;
    EXPECT_EQ(root.size(), 2U);
    for (std::size_t level = 0; level < 3; ++level) {
      const auto group = engine.synchronized_level_group(level);
      group_addresses[level] = group.data();
      for (auto& context : group) {
        EXPECT_EQ(&context.candidate, &engine.attempt_state(context.block, level));
        EXPECT_EQ(context.attempt, attempt);
        context.candidate.set_val(pops::Real(2 + context.block));
      }
    }
  });
  EXPECT_FALSE(engine.has_synchronized_groups());
  // A native map can modify the same detached stage between callbacks.
  for (std::size_t block = 0; block < 2; ++block)
    for (std::size_t level = 0; level < 3; ++level) {
      engine.attempt_state(block, level).set_val(pops::Real(5 + block));
      expect_value(hierarchy.state(block, level), block == 0 ? pops::Real(1) : pops::Real(4));
      EXPECT_FALSE(engine.accepted_clock(block, level));
      EXPECT_FALSE(engine.accepted_history(block, level));
    }
  EXPECT_EQ(hierarchy.accepted_revision(), revision);
  EXPECT_EQ(hierarchy.topology_runtime().materialization_generation(), generation);
  engine.resume_synchronized([&](auto) {
    ++callbacks;
    for (std::size_t level = 0; level < 3; ++level) {
      const auto group = engine.synchronized_level_group(level);
      EXPECT_EQ(group.data(), group_addresses[level]);
      for (auto& context : group) {
        expect_value(context.candidate, pops::Real(5 + context.block));
        if (level != 0) {
          expect_value(*context.staged_parent, context.block == 0 ? pops::Real(1) : pops::Real(4));
          EXPECT_EQ(context.incoming_flux,
                    engine.synchronized_level_group(level - 1)[context.block].outgoing_flux);
        }
        context.candidate.set_val(pops::Real(6 + context.block));
      }
    }
  });
  EXPECT_EQ(callbacks, 2);
  EXPECT_EQ(engine.last_accepted_attempt(), 0U);
  EXPECT_EQ(hierarchy.accepted_revision(), revision);
  std::vector<std::size_t> reflux_order;
  int validations = 0;
  int stages = 0;
  engine.finish_synchronized([&](auto& context) { reflux_order.push_back(context.parent_level); },
                             [&](std::size_t block, std::size_t, const auto& state) {
                               ++validations;
                               expect_value(state, pops::Real(6 + block));
                             },
                             [&](std::size_t, auto pack) {
                               ++stages;
                               EXPECT_EQ(pack.size(), 2U);
                               EXPECT_EQ(hierarchy.accepted_revision(), revision);
                             });
  EXPECT_EQ(validations, 6);
  EXPECT_EQ(stages, 3);
  EXPECT_EQ(reflux_order, (std::vector<std::size_t>{1, 1, 0, 0}));
  EXPECT_FALSE(engine.has_attempt_candidates());
  EXPECT_FALSE(engine.has_synchronized_attempt());
  EXPECT_EQ(engine.synchronized_attempt(), 0U);
  EXPECT_EQ(engine.last_accepted_attempt(), attempt);
  for (std::size_t block = 0; block < 2; ++block)
    for (std::size_t level = 0; level < 3; ++level) {
      expect_value(hierarchy.state(block, level), pops::Real(6 + block));
      ASSERT_TRUE(engine.accepted_clock(block, level));
      EXPECT_EQ(engine.accepted_clock(block, level)->physical_time, 0.25);
      ASSERT_TRUE(engine.accepted_history(block, level));
      expect_value(engine.accepted_history(block, level)->older,
                   block == 0 ? pops::Real(1) : pops::Real(4));
      expect_value(engine.accepted_history(block, level)->newer, pops::Real(6 + block));
    }
}

template <int Dim>
void prove_failure_abort_and_retry() {
  auto hierarchy = make_hierarchy<Dim>();
  auto engine = prepare_engine(hierarchy);
  const pops::amr::ClockWindow window{{0, 0, {0, 1}, 0.0}, {0, 0, {1, 1}, 0.25}};
  const auto revision = hierarchy.accepted_revision();
  const auto mutate = [&](auto) {
    for (std::size_t block = 0; block < 2; ++block)
      for (std::size_t level = 0; level < 3; ++level)
        engine.attempt_state(block, level).set_val(pops::Real(9));
  };
  const auto unchanged = [&] {
    EXPECT_EQ(hierarchy.accepted_revision(), revision);
    EXPECT_EQ(engine.last_accepted_attempt(), 0U);
    for (std::size_t block = 0; block < 2; ++block)
      for (std::size_t level = 0; level < 3; ++level) {
        expect_value(hierarchy.state(block, level), block == 0 ? pops::Real(1) : pops::Real(4));
        EXPECT_FALSE(engine.accepted_clock(block, level));
      }
  };
  auto reconcile = [](auto&) {};
  auto validate = [](std::size_t, std::size_t, const auto&) {};
  engine.begin_synchronized(window);
  EXPECT_THROW(engine.resume_synchronized([&](auto root) {
    mutate(root);
    if (hierarchy.lane().rank() == 0)
      throw std::runtime_error("rank-local continuation rejection");
  }),
               std::runtime_error);
  EXPECT_FALSE(engine.has_synchronized_attempt());
  unchanged();
  engine.begin_synchronized(window);
  engine.resume_synchronized(mutate);
  engine.abort_synchronized();
  engine.abort_synchronized();
  EXPECT_FALSE(engine.has_attempt_candidates());
  unchanged();
  engine.begin_synchronized(window);
  EXPECT_THROW(engine.resume_synchronized([&](auto root) {
    if (hierarchy.lane().rank() == 0)
      engine.abort_synchronized();
    // Cancellation cannot release a context while this callback still uses its references.
    root.front().candidate.set_val(pops::Real(12));
  }),
               std::runtime_error);
  unchanged();
  engine.begin_synchronized(window);
  engine.resume_synchronized(mutate);
  EXPECT_THROW(engine.finish_synchronized(reconcile,
                                          [&](std::size_t, std::size_t, const auto&) {
                                            if (hierarchy.lane().rank() == 0)
                                              throw std::runtime_error(
                                                  "rank-local final validation rejection");
                                          }),
               std::runtime_error);
  unchanged();
  EXPECT_FALSE(engine.has_synchronized_attempt());
  // Corrupt a later level during staging: the first level publishes, then the hierarchy
  // rejects the second level's component contract. The engine restores the complete snapshot.
  engine.begin_synchronized(window);
  engine.resume_synchronized(mutate);
  auto corrupt_later_level = [&](std::size_t level, auto pack) {
    if (level != 1)
      return;
    const auto& layout = hierarchy.topology_runtime().hierarchy().layout(level);
    const auto local_rank = layout.distribution().rank_space().coordinate(
        static_cast<std::size_t>(hierarchy.lane().rank()));
    *pack.front() = pops::MultiFab<Dim>(layout.patches(), layout.distribution(), local_rank, 2,
                                        pops::Extent<Dim>{});
  };
  EXPECT_THROW(engine.finish_synchronized(reconcile, validate, corrupt_later_level),
               std::exception);
  unchanged();
  EXPECT_FALSE(engine.has_synchronized_attempt());
  const auto final_attempt = engine.begin_synchronized(window);
  EXPECT_EQ(final_attempt, 6U);
  engine.resume_synchronized(mutate);
  engine.finish_synchronized(reconcile, validate);
  EXPECT_EQ(engine.last_accepted_attempt(), final_attempt);
  for (std::size_t block = 0; block < 2; ++block)
    for (std::size_t level = 0; level < 3; ++level)
      expect_value(hierarchy.state(block, level), pops::Real(9));
}

void prove_collective_identity_and_sequence() {
  auto hierarchy = make_hierarchy<1>();
  auto engine = prepare_engine(hierarchy);
  const pops::amr::ClockWindow window{{0, 0, {0, 1}, 0.0}, {0, 0, {1, 1}, 0.25}};
  auto reconcile = [](auto&) {};
  auto validate = [](std::size_t, std::size_t, const auto&) {};
  if (hierarchy.lane().size() > 1) {
    auto divergent = window;
    if (hierarchy.lane().rank() == 0)
      divergent.end.physical_time = 0.5;
    EXPECT_THROW(engine.begin_synchronized(divergent), std::invalid_argument);
    EXPECT_FALSE(engine.has_synchronized_attempt());
    engine.begin_synchronized(window);
    engine.resume_synchronized([](auto) {});
    if (hierarchy.lane().rank() == 0)
      EXPECT_THROW(engine.resume_synchronized([](auto) {}), std::invalid_argument);
    else
      EXPECT_THROW(engine.finish_synchronized(reconcile, validate), std::invalid_argument);
    engine.abort_synchronized();
  }
  engine.begin_synchronized(window);
  EXPECT_THROW(engine.begin_synchronized(window), std::exception);
  EXPECT_THROW(engine.finish_synchronized(reconcile, validate), std::exception);
  engine.resume_synchronized([](auto) {});
  engine.abort_synchronized();
  // Publishing through a different accepted-state authority invalidates a paused attempt,
  // even if it leaves the topology and every scalar value unchanged.
  engine.begin_synchronized(window);
  engine.resume_synchronized([](auto) {});
  std::vector<std::string> identities;
  std::vector<pops::MultiFab<1>> republished;
  republished.reserve(hierarchy.block_count());
  std::vector<pops::MultiFab<1>*> candidates;
  for (std::size_t block = 0; block < hierarchy.block_count(); ++block) {
    identities.push_back(hierarchy.block_identity(block));
    republished.emplace_back(hierarchy.state(block, 0));
    candidates.push_back(&republished.back());
  }
  const auto map = hierarchy.prepare_program_block_map(identities);
  hierarchy.publish_program_candidates(map, 0, candidates);
  EXPECT_THROW(engine.resume_synchronized([](auto) {}), std::exception);
  engine.abort_synchronized();
  const std::vector<pops::amr::ParentChildClockRelation> asynchronous{
      {0, 1, {2, 1}, pops::amr::RemainderPolicy::IntegralOnly},
      {1, 2, {1, 1}, pops::amr::RemainderPolicy::IntegralOnly}};
  auto other = Engine<1>::prepare(hierarchy, asynchronous,
                                  {{2, {32, 496}}, reflux::FaceFluxLedgerBudget{256, 256, 1}});
  EXPECT_THROW(other.begin_synchronized(window), std::exception);
  EXPECT_FALSE(other.has_synchronized_attempt());
}

}  // namespace

TEST(test_amr_synchronized_continuation, stable_candidates_across_two_resumes) {
  prove_persistent_candidates<1>();
  prove_persistent_candidates<2>();
  prove_persistent_candidates<3>();
}

TEST(test_amr_synchronized_continuation, collective_failure_abort_publication_rollback_and_retry) {
  prove_failure_abort_and_retry<1>();
  prove_failure_abort_and_retry<2>();
  prove_failure_abort_and_retry<3>();
}

TEST(test_amr_synchronized_continuation, exact_clock_and_collective_operation_authority) {
  prove_collective_identity_and_sequence();
}

TEST(test_amr_synchronized_continuation,
     rank_local_callback_reentry_is_rejected_before_collectives) {
  auto hierarchy = make_hierarchy<1>();
  auto engine = prepare_engine(hierarchy);
  const pops::amr::ClockWindow window{{0, 0, {0, 1}, 0.0}, {0, 0, {1, 1}, 0.25}};
  const auto revision = hierarchy.accepted_revision();
  auto reconcile = [](auto&) {};
  auto validate = [](std::size_t, std::size_t, const auto&) {};
  for (int operation = 0; operation < 3; ++operation) {
    engine.begin_synchronized(window);
    EXPECT_THROW(engine.resume_synchronized([&](auto root) {
      root.front().candidate.set_val(pops::Real(9));
      if (hierarchy.lane().rank() != 0)
        return;
      if (operation == 0)
        engine.begin_synchronized(window);
      else if (operation == 1)
        engine.resume_synchronized([](auto) {});
      else
        engine.finish_synchronized(reconcile, validate);
    }),
                 std::exception);
    EXPECT_FALSE(engine.has_synchronized_attempt());
    EXPECT_EQ(engine.last_accepted_attempt(), 0U);
    EXPECT_EQ(hierarchy.accepted_revision(), revision);
    for (std::size_t block = 0; block < 2; ++block)
      for (std::size_t level = 0; level < 3; ++level)
        expect_value(hierarchy.state(block, level), block == 0 ? pops::Real(1) : pops::Real(4));
  }
  const auto accepted = engine.begin_synchronized(window);
  engine.resume_synchronized([](auto) {});
  engine.finish_synchronized(reconcile, validate);
  EXPECT_EQ(engine.last_accepted_attempt(), accepted);
}
