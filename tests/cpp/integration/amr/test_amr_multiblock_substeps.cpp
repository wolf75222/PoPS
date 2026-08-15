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
          "space_filling_curve", "test.multiblock-subcycling.sfc",
          pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
}

template <int Dim>
mesh::RankSpace<Dim> execution_rank_space() {
  pops::Index<Dim> origin{};
  pops::Extent<Dim> extent{};
  for (int axis = 0; axis < Dim; ++axis)
    extent[axis] = 1;
  extent[0] = pops::ExecutionLane::world().size();
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
    const mesh::RankSpace<Dim>& ranks) {
  pops::Extent<Dim> tile{};
  for (int axis = 0; axis < Dim; ++axis)
    tile[axis] = 4;
  const mesh::BoxArray<Dim> patches = mesh::BoxArray<Dim>::from_domain(domain, tile);
  const auto ownership = authority.prepare(patches, ranks, kLoadBalanceBudget);
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
    const pops::PreparedLoadBalanceAuthority<Dim>& authority) {
  pops::Box<Dim> covered = parent.patches().boxes().front();
  covered.hi[0] = covered.lo[0] + (covered.length(0) / 2) - 1;
  const pops::Box<Dim> child_patch = hierarchy::refine_box(covered, ratio);
  const pops::Box<Dim> child_domain = hierarchy::refine_box(parent.domain(), ratio);
  const mesh::BoxArray<Dim> patches(std::vector<pops::Box<Dim>>{child_patch});
  const auto ownership =
      authority.prepare(patches, parent.distribution().rank_space(), kLoadBalanceBudget);
  return {level, child_domain, patches, ownership.plan().distribution(), ratio, kLayoutBudget};
}

template <int Dim>
MultiBlock<Dim> make_hierarchy() {
  const auto authority = load_balance<Dim>();
  const auto ranks = execution_rank_space<Dim>();
  pops::Index<Dim> lo{};
  pops::Index<Dim> hi{};
  for (int axis = 0; axis < Dim; ++axis)
    hi[axis] = 3;
  hi[0] = 4 * static_cast<int>(std::max<std::size_t>(2, ranks.size())) - 1;
  const pops::Box<Dim> domain{lo, hi};
  std::vector<hierarchy::LevelLayout<Dim>> layouts;
  layouts.push_back(make_coarse_layout(domain, *authority, ranks));
  layouts.push_back(
      make_partial_child_layout(layouts.back(), 1, spatial_ratio<Dim>(0), *authority));
  layouts.push_back(
      make_partial_child_layout(layouts.back(), 2, spatial_ratio<Dim>(1), *authority));

  const pops::Index<Dim> local_rank =
      ranks.coordinate(static_cast<std::size_t>(pops::ExecutionLane::world().rank()));
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
      "test.multiblock-subcycling.spatial." + std::to_string(Dim));
  std::vector<typename MultiBlock<Dim>::AdditionalBlock> additional;
  additional.push_back({"electrons", std::move(secondary_levels)});
  return MultiBlock<Dim>::prepare_collectively(
      std::move(topology), "ions", std::move(additional),
      "test.multiblock-subcycling.lane." + std::to_string(Dim));
}

template <int Dim>
void add_conservative_pair(pops::MultiFab<Dim>& field, pops::Real amount) {
  if (field.local_size() == 0)
    return;
  const pops::Box<Dim> box = field.box(0);
  pops::Index<Dim> first = box.lo;
  pops::Index<Dim> second = first;
  ++second[0];
  const auto values = field.fab(0).view();
  pops::for_each_cell(pops::Box<Dim>{first, first},
                      [=] POPS_HD(const pops::Index<Dim>& cell) { values(cell) += amount; });
  pops::for_each_cell(pops::Box<Dim>{second, second},
                      [=] POPS_HD(const pops::Index<Dim>& cell) { values(cell) -= amount; });
  Kokkos::fence();
}

template <int Dim>
void exchange_coupled_pack(pops::MultiFab<Dim>& first, pops::MultiFab<Dim>& second,
                           pops::Real fraction) {
  if (first.layout() != second.layout() || first.distribution() != second.distribution() ||
      first.local_global_indices() != second.local_global_indices())
    throw std::invalid_argument("coupled test pack lost its common block/level topology");
  for (std::size_t local = 0; local < first.local_size(); ++local) {
    const auto first_values = first.fab(local).view();
    const auto second_values = second.fab(local).view();
    pops::for_each_cell(first.box(local), [=] POPS_HD(const pops::Index<Dim>& cell) {
      const pops::Real amount = fraction * first_values(cell);
      first_values(cell) -= amount;
      second_values(cell) += amount;
    });
  }
  Kokkos::fence();
}

template <int Dim>
reflux::CoarseFaceRefluxKey<Dim> reflux_key(std::string owner, std::size_t parent_level,
                                            std::uint64_t attempt,
                                            const pops::amr::ClockWindow& window) {
  reflux::CoarseFaceRefluxKey<Dim> query;
  query.owner = std::move(owner);
  query.state = "density";
  query.levels = {static_cast<int>(parent_level), static_cast<int>(parent_level + 1)};
  query.centering = reflux::FaceLedgerCentering::Face;
  query.axis = 0;
  query.attempt = attempt;
  query.macro_step = window.begin.macro_step;
  query.window_begin = window.begin.phase;
  query.window_end = window.end.phase;
  return query;
}

template <int Dim>
reflux::FaceFluxFragmentKey<Dim> fragment_key(const reflux::CoarseFaceRefluxKey<Dim>& query,
                                              reflux::FaceLedgerRole role,
                                              const pops::Index<Dim>& face,
                                              const pops::amr::ClockWindow& window) {
  reflux::FaceFluxFragmentKey<Dim> key;
  key.owner = query.owner;
  key.state = query.state;
  key.levels = query.levels;
  key.centering = query.centering;
  key.axis = query.axis;
  key.face = face;
  key.coarse_face = query.coarse_face;
  key.clock = window.begin;
  key.stage = "forward-euler";
  key.attempt = query.attempt;
  key.role = role;
  key.contribution = reflux::FaceLedgerContribution::NumericalFlux;
  return key;
}

template <int Dim>
void prove_multiblock_subcycling() {
  MultiBlock<Dim> hierarchy = make_hierarchy<Dim>();
  const std::vector<pops::amr::ParentChildClockRelation> relations{
      {0, 1, {5, 2}, pops::amr::RemainderPolicy::ExplicitFinalSubstep},
      {1, 2, {2, 1}, pops::amr::RemainderPolicy::IntegralOnly}};
  Engine<Dim> engine = Engine<Dim>::prepare(
      hierarchy, relations, {{2, {32, 496}}, reflux::FaceFluxLedgerBudget{256, 256, 1}});

  std::array<std::array<pops::Real, 3>, 2> initial_mass{};
  for (std::size_t block = 0; block < 2; ++block)
    for (std::size_t level = 0; level < 3; ++level)
      initial_mass[block][level] = pops::reduce_sum(hierarchy.state(block, level));
  std::vector<std::vector<pops::MultiFab<Dim>>> initial(2);
  for (std::size_t block = 0; block < 2; ++block)
    for (std::size_t level = 0; level < 3; ++level)
      initial[block].emplace_back(hierarchy.state(block, level));

  bool fail_finest = false;
  std::vector<std::string> callback_order;
  std::vector<std::size_t> reflux_order;
  const reflux::MetricRefluxBudget metric_budget{64, 256, 64};

  auto record_flux = [&](auto& context, typename Engine<Dim>::ledger_type& ledger,
                         std::size_t parent_level, reflux::FaceLedgerRole role) {
    const auto parent_window =
        role == reflux::FaceLedgerRole::Coarse
            ? context.window
            : pops::amr::ClockWindow{
                  {static_cast<int>(parent_level), context.window.begin.macro_step, {0, 1}, 0.0},
                  {static_cast<int>(parent_level), context.window.end.macro_step, {1, 1}, 1.0}};
    auto query = reflux_key<Dim>(std::string(context.block_identity), parent_level, context.attempt,
                                 parent_window);
    const auto& ratio =
        hierarchy.topology_runtime().hierarchy().layout(parent_level + 1).ratio_from_parent();
    const reflux::FaceRefinementMapping<Dim> mapping{
        hierarchy.topology_runtime().hierarchy().layout(parent_level).domain().lo,
        hierarchy.topology_runtime().hierarchy().layout(parent_level + 1).domain().lo};
    const auto fine_faces =
        reflux::fine_faces_for_coarse_face(query, ratio, mapping, metric_budget);
    const double duration = context.window.end.physical_time - context.window.begin.physical_time;
    const reflux::FaceFluxFragmentMeasure measure{
        {1, 1},
        context.window.begin.phase,
        context.window.end.phase,
        duration,
        role == reflux::FaceLedgerRole::Coarse ? static_cast<double>(fine_faces.size()) : 1.0};
    if (role == reflux::FaceLedgerRole::Coarse) {
      ledger.accumulate(fragment_key(query, role, query.coarse_face, context.window), measure, 1.0);
    } else {
      for (const auto& face : fine_faces)
        ledger.accumulate(fragment_key(query, role, face, context.window), measure, 1.25);
    }
  };

  auto advance = [&](std::span<typename Engine<Dim>::LevelAdvanceContext> group) {
    if (group.size() != 2 || group[0].block != 0 || group[1].block != 1 ||
        group[0].level != group[1].level || group[0].substep != group[1].substep ||
        group[0].window.begin != group[1].window.begin ||
        group[0].window.end != group[1].window.end)
      throw std::runtime_error("level-group callback lost its simultaneous block pack");
    for (auto& context : group) {
      callback_order.push_back(std::string(context.block_identity) + ":L" +
                               std::to_string(context.level) + ":S" +
                               std::to_string(context.substep));
      if (context.level > 0 && context.staged_parent == nullptr)
        throw std::runtime_error("child callback lost its staged parent-time image");
      add_conservative_pair(
          context.candidate,
          pops::Real(0.01) * pops::Real(context.block + 1) * pops::Real(context.level + 1) *
              pops::Real(context.window.end.physical_time - context.window.begin.physical_time));
    }
    const pops::Real coupled_fraction =
        pops::Real(0.05) *
        pops::Real(group[0].window.end.physical_time - group[0].window.begin.physical_time);
    exchange_coupled_pack(group[0].candidate, group[1].candidate, coupled_fraction);
    for (auto& context : group) {
      if (context.outgoing_flux != nullptr)
        record_flux(context, *context.outgoing_flux, context.level, reflux::FaceLedgerRole::Coarse);
      if (context.incoming_flux != nullptr)
        record_flux(context, *context.incoming_flux, context.level - 1,
                    reflux::FaceLedgerRole::Fine);
    }
    if (fail_finest && group[0].level == 2 && group[0].substep == 1)
      throw std::runtime_error("injected finest-level failure");
  };

  auto reconcile = [&](typename Engine<Dim>::RefluxContext& context) {
    reflux_order.push_back(context.parent_level);
    auto query = reflux_key<Dim>(std::string(context.block_identity), context.parent_level,
                                 context.attempt, context.parent_window);
    const auto result = hierarchy.topology_runtime().reconcile_reflux_for_owner(
        context.flux, query, context.block_identity, "density", context.face_mapping, metric_budget,
        [](double& destination, double factor, const double& source) {
          destination += factor * source;
        });
    if (!(result.mismatch > 0.0))
      throw std::runtime_error("test reflux did not reconcile a positive fine/coarse mismatch");
    // Reflux is applied as a conservative pair outside the restricted child footprint.
    add_conservative_pair(context.parent, static_cast<pops::Real>(result.mismatch));
  };
  auto validate = [](std::size_t, std::size_t, const pops::MultiFab<Dim>& candidate) {
    if (candidate.local_size() == 0)
      return;
    if (!std::isfinite(static_cast<double>(pops::reduce_min_local(candidate))) ||
        !std::isfinite(static_cast<double>(pops::reduce_max_local(candidate))))
      throw std::runtime_error("non-finite block/level candidate");
  };

  const pops::amr::ClockWindow first{{0, 0, {0, 1}, 0.0}, {0, 0, {1, 1}, 0.2}};
  engine.advance(first, advance, reconcile, validate);
  EXPECT_EQ(engine.last_accepted_attempt(), 1U);
  EXPECT_EQ(callback_order.size(), 20U);
  EXPECT_EQ(reflux_order, (std::vector<std::size_t>{1, 1, 1, 1, 1, 1, 0, 0}));
  for (std::size_t block = 0; block < 2; ++block) {
    EXPECT_EQ(engine.ledgers(block, 0).size(), 1U);
    EXPECT_EQ(engine.ledgers(block, 1).size(), 3U);
    for (std::size_t level = 0; level < 3; ++level) {
      EXPECT_TRUE(engine.accepted_clock(block, level).has_value());
      EXPECT_TRUE(engine.accepted_history(block, level).has_value());
    }
    EXPECT_GT(pops::difference_sum_sq_all(hierarchy.state(block, 2), initial[block][2]),
              pops::Real(0));
  }
  for (std::size_t level = 0; level < 3; ++level)
    EXPECT_LT(pops::reduce_sum(hierarchy.state(0, level)), initial_mass[0][level]);
  for (std::size_t level = 0; level < 3; ++level)
    EXPECT_GT(pops::reduce_sum(hierarchy.state(1, level)), initial_mass[1][level]);
  for (std::size_t level = 0; level < 3; ++level) {
    const pops::Real expected_total = initial_mass[0][level] + initial_mass[1][level];
    EXPECT_NEAR(
        pops::reduce_sum(hierarchy.state(0, level)) + pops::reduce_sum(hierarchy.state(1, level)),
        expected_total, pops::Real(1e-12) * std::abs(expected_total) + pops::Real(1e-12));
  }

  std::vector<std::vector<pops::MultiFab<Dim>>> accepted(2);
  for (std::size_t block = 0; block < 2; ++block)
    for (std::size_t level = 0; level < 3; ++level)
      accepted[block].emplace_back(hierarchy.state(block, level));
  const std::uint64_t accepted_revision = hierarchy.accepted_revision();
  const auto accepted_clock = engine.accepted_clock(1, 2);
  const auto accepted_history_window = engine.accepted_history(1, 2)->window;
  const std::size_t accepted_finest_ledgers = engine.ledgers(1, 1).size();

  callback_order.clear();
  reflux_order.clear();
  fail_finest = true;
  const pops::amr::ClockWindow retry{{0, 1, {0, 1}, 0.2}, {0, 1, {1, 1}, 0.4}};
  EXPECT_THROW(engine.advance(retry, advance, reconcile, validate), std::runtime_error);
  EXPECT_EQ(engine.last_accepted_attempt(), 1U);
  EXPECT_EQ(hierarchy.accepted_revision(), accepted_revision);
  EXPECT_EQ(engine.accepted_clock(1, 2), accepted_clock);
  EXPECT_EQ(engine.accepted_history(1, 2)->window.begin, accepted_history_window.begin);
  EXPECT_EQ(engine.ledgers(1, 1).size(), accepted_finest_ledgers);
  for (std::size_t block = 0; block < 2; ++block)
    for (std::size_t level = 0; level < 3; ++level)
      EXPECT_EQ(pops::difference_sum_sq_all(hierarchy.state(block, level), accepted[block][level]),
                pops::Real(0));

  fail_finest = false;
  callback_order.clear();
  reflux_order.clear();
  engine.advance(retry, advance, reconcile, validate);
  EXPECT_EQ(engine.last_accepted_attempt(), 3U);
  EXPECT_EQ(callback_order.size(), 20U);
  EXPECT_EQ(reflux_order, (std::vector<std::size_t>{1, 1, 1, 1, 1, 1, 0, 0}));

  std::vector<std::vector<pops::MultiFab<Dim>>> before_typed_rejection(2);
  for (std::size_t block = 0; block < 2; ++block)
    for (std::size_t level = 0; level < 3; ++level)
      before_typed_rejection[block].emplace_back(hierarchy.state(block, level));
  auto reject_typed = [&](std::span<typename Engine<Dim>::LevelAdvanceContext> group) {
    advance(group);
    if (group[0].level == 2 && group[0].substep == 1 && hierarchy.lane().rank() == 0)
      throw pops::runtime::program::StepAttemptRejected(
          pops::SolveStatus::kIterationLimit,
          pops::runtime::program::StepAttemptDisposition::kRetry, 0x494d4558u, "implicit-source",
          "implicit_source_iteration_limit_index_4_0");
  };
  const pops::amr::ClockWindow typed_window{{0, 2, {0, 1}, 0.4}, {0, 2, {1, 1}, 0.6}};
  try {
    engine.advance(typed_window, reject_typed, reconcile, validate);
    FAIL() << "typed level-group rejection must leave the attempt";
  } catch (const pops::runtime::program::StepAttemptRejected& rejected) {
    EXPECT_EQ(rejected.status(), pops::SolveStatus::kIterationLimit);
    EXPECT_EQ(rejected.disposition(), pops::runtime::program::StepAttemptDisposition::kRetry);
    EXPECT_EQ(rejected.reason_code(), 0x494d4558u);
    EXPECT_EQ(rejected.phase(), "implicit-source");
    EXPECT_EQ(rejected.detail(), "implicit_source_iteration_limit_index_4_0");
  }
  EXPECT_EQ(engine.last_accepted_attempt(), 3U);
  for (std::size_t block = 0; block < 2; ++block)
    for (std::size_t level = 0; level < 3; ++level)
      EXPECT_EQ(pops::difference_sum_sq_all(hierarchy.state(block, level),
                                            before_typed_rejection[block][level]),
                pops::Real(0));

  engine.advance(typed_window, advance, reconcile, validate);
  EXPECT_EQ(engine.last_accepted_attempt(), 5U);
}

}  // namespace

TEST(test_amr_multiblock_substeps, three_levels_two_blocks_are_atomic_and_conservative) {
  prove_multiblock_subcycling<1>();
  prove_multiblock_subcycling<2>();
  prove_multiblock_subcycling<3>();
}
