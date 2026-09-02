/// @file
/// @brief Exact eight-rank 3-D corner exchange and transactional AMR authority proof.
///
/// This is deliberately a native MPI proof, not a capability inventory.  It requires the
/// compiled artifact to be POPS_NATIVE_DIM=3 and the launch to contain exactly the eight ranks
/// of RankSpace extent {2, 2, 2}; a different launch is a test failure, never a skip.  The first
/// proof exercises an actual (+x,+y,+z) ghost transfer.  The second proof targets the stable
/// prepared multi-block subcycling authority directly.  It does not claim that every public
/// Program frontend accepts a non-integral 5/2 relation: frontends that do not expose that
/// relation must refuse it before mutation rather than silently lowering it to a positive case.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/amr/reflux/face_flux_ledger.hpp>
#include <pops/amr/reflux/metric_reflux.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/boundary/halo_exchange.hpp>
#include <pops/mesh/boundary/halo_schedule.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace hierarchy = pops::amr::hierarchy;
namespace mesh = pops::mesh;
namespace reflux = pops::amr::reflux;
namespace time_amr = pops::numerics::time::amr;

constexpr int kRequiredRanks = 8;
constexpr int kBlockExtent = 2;
constexpr int kBlocksPerAxis = 2;
constexpr int kCoarseTileExtent = 4;
constexpr mesh::BoxArrayValidationBudget kLayoutBudget{32, 496};
constexpr hierarchy::HierarchyValidationBudget kHierarchyBudget{3, 256};
constexpr pops::parallel::LoadBalancePreparationBudget kLoadBalanceBudget{
    32, 8, std::numeric_limits<std::int64_t>::max()};

template <int Dim>
using MultiBlock = pops::runtime::amr::PreparedMultiBlockAmrHierarchy<Dim>;

template <int Dim>
using Engine = time_amr::PreparedMultiBlockAmrSubcyclingEngine<Dim, double>;

template <int Dim>
mesh::RankSpace<Dim> cube_rank_space() {
  pops::Index<Dim> origin{};
  pops::Extent<Dim> extent{};
  for (int axis = 0; axis < Dim; ++axis)
    extent[axis] = kBlocksPerAxis;
  return {origin, extent};
}

template <int Dim>
pops::Index<Dim> rank_coordinate(const mesh::RankSpace<Dim>& ranks, int rank) {
  return ranks.coordinate(static_cast<std::size_t>(rank));
}

template <int Dim>
pops::Index<Dim> rank_coordinate(int rank) {
  return rank_coordinate(cube_rank_space<Dim>(), rank);
}

template <int Dim>
std::vector<pops::Box<Dim>> cube_boxes() {
  static_assert(Dim == 3, "the corner authority is specifically a 3-D proof");
  std::vector<pops::Box<Dim>> boxes;
  boxes.reserve(kRequiredRanks);
  // RankSpace linear order is x + 2*y + 4*z, so this order authenticates owners exactly.
  for (int z = 0; z < kBlocksPerAxis; ++z)
    for (int y = 0; y < kBlocksPerAxis; ++y)
      for (int x = 0; x < kBlocksPerAxis; ++x) {
        pops::Index<Dim> lower{};
        pops::Index<Dim> upper{};
        lower[0] = kBlockExtent * x;
        lower[1] = kBlockExtent * y;
        lower[2] = kBlockExtent * z;
        for (int axis = 0; axis < Dim; ++axis)
          upper[axis] = lower[axis] + kBlockExtent - 1;
        boxes.push_back({lower, upper});
      }
  return boxes;
}

template <int Dim>
std::vector<pops::Index<Dim>> cube_owners(const mesh::RankSpace<Dim>& ranks) {
  std::vector<pops::Index<Dim>> owners;
  owners.reserve(kRequiredRanks);
  for (std::size_t linear = 0; linear < ranks.size(); ++linear)
    owners.push_back(ranks.coordinate(linear));
  return owners;
}

template <int Dim>
pops::Index<Dim> index_from_cell(const pops::Box<Dim>& box, std::size_t cell) {
  pops::Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t extent = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(cell % extent);
    cell /= extent;
  }
  return index;
}

template <int Dim>
pops::Real corner_value(const pops::Index<Dim>& index, std::size_t owner) {
  pops::Real value = static_cast<pops::Real>(1000000.0 * static_cast<double>(owner) + 17.0);
  pops::Real scale = pops::Real{1};
  for (int axis = 0; axis < Dim; ++axis) {
    value += scale * static_cast<pops::Real>(index[axis]);
    scale *= pops::Real{97};
  }
  return value;
}

template <int Dim>
void fill_corner_valid(pops::MultiFab<Dim>& fields) {
  const auto& ranks = fields.distribution().rank_space();
  for (const std::size_t global_box : fields.local_global_indices()) {
    auto& fab = fields.fab_global(global_box);
    auto host = fab.create_host_mirror();
    const pops::Box<Dim>& grown = fab.grown_box();
    const std::size_t cells = static_cast<std::size_t>(grown.numPts());
    const std::size_t owner = ranks.linear_rank(fields.distribution().owner(global_box));
    for (std::size_t cell = 0; cell < cells; ++cell) {
      const pops::Index<Dim> index = index_from_cell(grown, cell);
      host(cell) = fab.box().contains(index) ? corner_value(index, owner) : pops::Real{-777};
    }
    fab.copy_from_host(host);
  }
  Kokkos::fence();
}

template <int Dim>
pops::Real value_at(const pops::MultiFab<Dim>& fields, std::size_t global_box,
                    const pops::Index<Dim>& index) {
  const auto& fab = fields.fab_global(global_box);
  const pops::Box<Dim>& grown = fab.grown_box();
  std::size_t stride = 1;
  std::size_t cell = 0;
  for (int axis = 0; axis < Dim; ++axis) {
    cell += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(grown.length(axis));
  }
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  return host(cell);
}

template <int Dim>
void prove_exact_three_dimensional_corner_exchange() {
  static_assert(Dim == 3, "the corner authority is specifically a 3-D proof");
  const mesh::RankSpace<Dim> ranks = cube_rank_space<Dim>();
  const pops::Index<Dim> local_rank = rank_coordinate(ranks, pops::my_rank());
  const mesh::BoxArray<Dim> layout(cube_boxes<Dim>());
  const mesh::Distribution<Dim> distribution =
      mesh::Distribution<Dim>::partitioned(layout, ranks, cube_owners(ranks));

  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = kBlocksPerAxis * kBlockExtent - 1;
  const pops::Box<Dim> domain{lower, upper};
  pops::Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    ghosts[axis] = 1;
  std::array<bool, Dim> periodic{};
  periodic.fill(false);
  const pops::BoundaryTopology<Dim> topology = pops::BoundaryTopology<Dim>::axis_periodic(periodic);
  pops::MultiFab<Dim> fields(layout, distribution, local_rank, 1, ghosts);
  fill_corner_valid(fields);

  const pops::HaloScheduleBudget budget{{kRequiredRanks, 128},  // layout and overlap work
                                        4096,                   // box-image pairs
                                        4096,                   // jobs
                                        1,                      // no periodic images
                                        64,                     // peer plans
                                        4096,                   // local elements
                                        4096,                   // sends
                                        4096};                  // receives
  const pops::HaloSchedule<Dim> schedule =
      pops::prepare_halo_schedule(fields, domain, topology, budget);
  const pops::ExecutionLane lane =
      pops::ExecutionLane::duplicate_world_collectively("test.mpi-amr-3d-corner-authority");

  EXPECT_EQ(ranks.origin(), (pops::Index<Dim>{0, 0, 0}));
  EXPECT_EQ(ranks.extent(), (pops::Extent<Dim>{2, 2, 2}));
  EXPECT_EQ(ranks.size(), static_cast<std::size_t>(kRequiredRanks));
  EXPECT_EQ(ranks.linear_rank(local_rank), static_cast<std::size_t>(pops::my_rank()));
  for (std::size_t box = 0; box < layout.size(); ++box) {
    EXPECT_EQ(distribution.owner(box), ranks.coordinate(box));
    EXPECT_EQ(ranks.linear_rank(distribution.owner(box)), box);
  }

  const pops::Index<Dim> destination_corner{2, 2, 2};
  std::size_t canonical_corner_jobs = 0;
  for (const auto& job : schedule.canonical_jobs()) {
    if (job.destination_box == 0 && job.destination_region.contains(destination_corner)) {
      ++canonical_corner_jobs;
      EXPECT_EQ(job.source_box, static_cast<std::size_t>(7));
      EXPECT_EQ(job.source_from_destination, (pops::Index<Dim>{0, 0, 0}));
      pops::Index<Dim> corner_lower{};
      corner_lower[0] = 2;
      corner_lower[1] = 2;
      corner_lower[2] = 2;
      EXPECT_EQ(job.destination_region, (pops::Box<Dim>{corner_lower, corner_lower}));
    }
  }
  EXPECT_EQ(canonical_corner_jobs, static_cast<std::size_t>(1));

  const pops::Index<Dim> diagonal_peer{1, 1, 1};
  long local_corner_receives = 0;
  bool diagonal_corner_found = false;
  for (const auto& plan : schedule.receive_plans()) {
    if (plan.peer != diagonal_peer)
      continue;
    for (const auto& job : plan.jobs)
      if (job.destination_box == 0 && job.destination_region.contains(destination_corner)) {
        diagonal_corner_found = true;
        ++local_corner_receives;
        EXPECT_EQ(job.source_box, static_cast<std::size_t>(7));
        EXPECT_EQ(job.destination_region.numPts(), static_cast<std::int64_t>(1));
      }
  }
  EXPECT_EQ(pops::all_reduce_sum(local_corner_receives, lane.communicator()), 1L);
  if (pops::my_rank() == 0)
    EXPECT_TRUE(diagonal_corner_found);
  else
    EXPECT_FALSE(diagonal_corner_found);

  if (pops::my_rank() == 0)
    EXPECT_EQ(value_at(fields, 0, destination_corner), pops::Real{-777});
  pops::HaloExchange<Dim> exchange(schedule, lane, pops::HaloExchangeContext{101, 103});
  exchange.begin(fields, lane);
  // begin() stages and posts only; no destination ghost may be published before complete().
  if (pops::my_rank() == 0)
    EXPECT_EQ(value_at(fields, 0, destination_corner), pops::Real{-777});
  exchange.complete(fields, lane);
  const pops::Real expected = corner_value(destination_corner, 7);
  if (pops::my_rank() == 0)
    EXPECT_EQ(value_at(fields, 0, destination_corner), expected);
  EXPECT_EQ(exchange.live_request_count(), static_cast<std::size_t>(0));
  EXPECT_FALSE(exchange.sealed());

  // A second exact replay must assign the same authenticated corner, not accumulate it twice.
  exchange.execute(fields, lane);
  if (pops::my_rank() == 0)
    EXPECT_EQ(value_at(fields, 0, destination_corner), expected);
}

template <int Dim>
std::shared_ptr<const pops::PreparedLoadBalanceAuthority<Dim>> load_balance() {
  return std::make_shared<const pops::PreparedLoadBalanceAuthority<Dim>>(
      pops::prepare_load_balance_authority<Dim>(
          "space_filling_curve", "test.mpi-amr-3d-corner-authority.sfc",
          pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
}

template <int Dim>
hierarchy::LevelLayout<Dim> make_coarse_layout(
    const pops::Box<Dim>& domain, const pops::PreparedLoadBalanceAuthority<Dim>& authority,
    const mesh::RankSpace<Dim>& ranks, const pops::ExecutionLane& lane) {
  pops::Extent<Dim> tile{};
  for (int axis = 0; axis < Dim; ++axis)
    tile[axis] = kCoarseTileExtent;
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
pops::amr::RefinementRatio<Dim> spatial_ratio(int transition) {
  std::array<int, Dim> ratio{};
  ratio.fill(2);
  if constexpr (Dim > 1)
    ratio[static_cast<std::size_t>(transition % Dim)] = 3;
  return pops::amr::RefinementRatio<Dim>(ratio);
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
MultiBlock<Dim> make_amr_hierarchy() {
  const pops::ExecutionLane parent =
      pops::ExecutionLane::duplicate_world_collectively("test.mpi-amr-3d-corner-authority.parent");
  const auto authority = load_balance<Dim>();
  const mesh::RankSpace<Dim> ranks = cube_rank_space<Dim>();
  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = kCoarseTileExtent * kBlocksPerAxis - 1;
  const pops::Box<Dim> domain{lower, upper};

  std::vector<hierarchy::LevelLayout<Dim>> layouts;
  layouts.push_back(make_coarse_layout(domain, *authority, ranks, parent));
  layouts.push_back(
      make_partial_child_layout(layouts.back(), 1, spatial_ratio<Dim>(0), *authority, parent));
  layouts.push_back(
      make_partial_child_layout(layouts.back(), 2, spatial_ratio<Dim>(1), *authority, parent));

  const pops::Index<Dim> local_rank = rank_coordinate(ranks, parent.rank());
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
  pops::runtime::amr::AmrRuntime<Dim> topology(
      hierarchy::AmrHierarchy<Dim>(std::move(primary_levels), kHierarchyBudget), authority,
      "test.mpi-amr-3d-corner-authority.spatial");
  std::vector<typename MultiBlock<Dim>::AdditionalBlock> additional;
  additional.push_back({"electrons", std::move(secondary_levels)});
  return MultiBlock<Dim>::prepare_collectively(parent, std::move(topology), "ions",
                                               std::move(additional),
                                               "test.mpi-amr-3d-corner-authority.lane");
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
    throw std::invalid_argument("3-D corner authority lost its common block/level topology");
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
std::vector<pops::RealBits> snapshot_bits(const pops::MultiFab<Dim>& field) {
  std::vector<pops::RealBits> result;
  for (const std::size_t global_box : field.local_global_indices()) {
    const auto& fab = field.fab_global(global_box);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    result.reserve(result.size() + host.size());
    for (std::size_t element = 0; element < host.size(); ++element)
      result.push_back(std::bit_cast<pops::RealBits>(host(element)));
  }
  return result;
}

template <int Dim>
void expect_all_ledgers_attempt(const Engine<Dim>& engine, std::uint64_t attempt) {
  for (std::size_t block = 0; block < 2; ++block)
    for (std::size_t parent = 0; parent < 2; ++parent) {
      const auto& ledgers = engine.ledgers(block, parent);
      ASSERT_FALSE(ledgers.empty());
      for (const auto& ledger : ledgers) {
        ASSERT_GT(ledger.published_entries(0).size(), static_cast<std::size_t>(0));
        for (const auto& entry : ledger.published_entries(0))
          EXPECT_EQ(entry.key.attempt, attempt);
      }
    }
}

template <int Dim>
void prove_three_level_two_block_subcycling() {
  static_assert(Dim == 3, "the AMR authority proof is specifically a 3-D proof");
  MultiBlock<Dim> hierarchy = make_amr_hierarchy<Dim>();
  const std::vector<pops::amr::ParentChildClockRelation> relations{
      {0, 1, {5, 2}, pops::amr::RemainderPolicy::ExplicitFinalSubstep},
      {1, 2, {2, 1}, pops::amr::RemainderPolicy::IntegralOnly}};
  Engine<Dim> engine = Engine<Dim>::prepare(
      hierarchy, relations, {{2, {32, 496}}, reflux::FaceFluxLedgerBudget{256, 256, 1}});

  std::array<std::array<pops::Real, 3>, 2> initial_mass{};
  for (std::size_t block = 0; block < 2; ++block)
    for (std::size_t level = 0; level < 3; ++level)
      initial_mass[block][level] = pops::reduce_sum(hierarchy.state(block, level));

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
    const auto query = reflux_key<Dim>(std::string(context.block_identity), parent_level,
                                       context.attempt, parent_window);
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
      throw std::runtime_error("3-D AMR level-group callback lost its simultaneous block pack");
    for (auto& context : group) {
      callback_order.push_back(std::string(context.block_identity) + ":L" +
                               std::to_string(context.level) + ":S" +
                               std::to_string(context.substep));
      if (context.level > 0 && context.staged_parent == nullptr)
        throw std::runtime_error("3-D AMR child callback lost its staged parent image");
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
      throw std::runtime_error("injected 3-D AMR finest-level failure");
  };

  auto reconcile = [&](typename Engine<Dim>::RefluxContext& context) {
    reflux_order.push_back(context.parent_level);
    const auto query = reflux_key<Dim>(std::string(context.block_identity), context.parent_level,
                                       context.attempt, context.parent_window);
    const auto result = hierarchy.topology_runtime().reconcile_reflux_for_owner(
        context.flux, query, context.block_identity, "density", context.face_mapping, metric_budget,
        [](double& destination, double factor, const double& source) {
          destination += factor * source;
        });
    if (!(result.mismatch > 0.0))
      throw std::runtime_error("3-D AMR reflux did not produce a positive mismatch");
    add_conservative_pair(context.parent, static_cast<pops::Real>(result.mismatch));
  };
  auto validate = [](std::size_t, std::size_t, const pops::MultiFab<Dim>& candidate) {
    if (candidate.local_size() == 0)
      return;
    if (!std::isfinite(static_cast<double>(pops::reduce_min_local(candidate))) ||
        !std::isfinite(static_cast<double>(pops::reduce_max_local(candidate))))
      throw std::runtime_error("3-D AMR candidate contains a non-finite value");
  };

  const pops::amr::ClockWindow first{{0, 0, {0, 1}, 0.0}, {0, 0, {1, 1}, 0.2}};
  engine.advance(first, advance, reconcile, validate);
  EXPECT_EQ(engine.last_accepted_attempt(), static_cast<std::uint64_t>(1));
  EXPECT_EQ(callback_order.size(), static_cast<std::size_t>(20));
  EXPECT_EQ(reflux_order, (std::vector<std::size_t>{1, 1, 1, 1, 1, 1, 0, 0}));
  expect_all_ledgers_attempt(engine, 1);

  for (std::size_t level = 0; level < 3; ++level) {
    const pops::Real expected_total = initial_mass[0][level] + initial_mass[1][level];
    EXPECT_NEAR(
        pops::reduce_sum(hierarchy.state(0, level)) + pops::reduce_sum(hierarchy.state(1, level)),
        expected_total, pops::Real(1e-5) * std::abs(expected_total) + pops::Real(1e-5));
  }

  std::vector<std::vector<std::vector<pops::RealBits>>> accepted_bits(2);
  for (std::size_t block = 0; block < 2; ++block) {
    accepted_bits[block].reserve(3);
    for (std::size_t level = 0; level < 3; ++level)
      accepted_bits[block].push_back(snapshot_bits(hierarchy.state(block, level)));
  }
  const std::uint64_t accepted_revision = hierarchy.accepted_revision();
  const std::vector<std::size_t> accepted_ledger_sizes{
      engine.ledgers(0, 0).size(), engine.ledgers(0, 1).size(), engine.ledgers(1, 0).size(),
      engine.ledgers(1, 1).size()};

  callback_order.clear();
  reflux_order.clear();
  fail_finest = true;
  const pops::amr::ClockWindow retry{{0, 1, {0, 1}, 0.2}, {0, 1, {1, 1}, 0.4}};
  EXPECT_THROW(engine.advance(retry, advance, reconcile, validate), std::runtime_error);
  EXPECT_EQ(engine.last_accepted_attempt(), static_cast<std::uint64_t>(1));
  EXPECT_EQ(hierarchy.accepted_revision(), accepted_revision);
  EXPECT_EQ(accepted_ledger_sizes,
            (std::vector<std::size_t>{engine.ledgers(0, 0).size(), engine.ledgers(0, 1).size(),
                                      engine.ledgers(1, 0).size(), engine.ledgers(1, 1).size()}));
  for (std::size_t block = 0; block < 2; ++block)
    for (std::size_t level = 0; level < 3; ++level)
      EXPECT_EQ(snapshot_bits(hierarchy.state(block, level)), accepted_bits[block][level]);
  expect_all_ledgers_attempt(engine, 1);

  fail_finest = false;
  callback_order.clear();
  reflux_order.clear();
  engine.advance(retry, advance, reconcile, validate);
  EXPECT_EQ(engine.last_accepted_attempt(), static_cast<std::uint64_t>(3));
  EXPECT_EQ(callback_order.size(), static_cast<std::size_t>(20));
  EXPECT_EQ(reflux_order, (std::vector<std::size_t>{1, 1, 1, 1, 1, 1, 0, 0}));
  expect_all_ledgers_attempt(engine, 3);

  // A typed rejection is collective even when only rank zero raises it.  It must leave the
  // accepted image untouched; the next exact retry then consumes one fresh attempt identity.
  std::vector<std::vector<std::vector<pops::RealBits>>> before_typed_rejection(2);
  for (std::size_t block = 0; block < 2; ++block) {
    before_typed_rejection[block].reserve(3);
    for (std::size_t level = 0; level < 3; ++level)
      before_typed_rejection[block].push_back(snapshot_bits(hierarchy.state(block, level)));
  }
  const std::uint64_t revision_before_typed_rejection = hierarchy.accepted_revision();
  const auto reject_typed = [&](std::span<typename Engine<Dim>::LevelAdvanceContext> group) {
    advance(group);
    if (group[0].level == 2 && group[0].substep == 1 && hierarchy.lane().rank() == 0)
      throw pops::runtime::program::StepAttemptRejected(
          pops::SolveStatus::kIterationLimit,
          pops::runtime::program::StepAttemptDisposition::kRetry, 0x33434452u, "3d-corner-subcycle",
          "3d_corner_authority_iteration_limit");
  };
  const pops::amr::ClockWindow typed_window{{0, 2, {0, 1}, 0.4}, {0, 2, {1, 1}, 0.6}};
  bool typed_rejection_seen = false;
  try {
    engine.advance(typed_window, reject_typed, reconcile, validate);
  } catch (const pops::runtime::program::StepAttemptRejected& rejected) {
    typed_rejection_seen = true;
    EXPECT_EQ(rejected.status(), pops::SolveStatus::kIterationLimit);
    EXPECT_EQ(rejected.disposition(), pops::runtime::program::StepAttemptDisposition::kRetry);
    EXPECT_EQ(rejected.reason_code(), 0x33434452u);
    EXPECT_EQ(rejected.phase(), "3d-corner-subcycle");
    EXPECT_EQ(rejected.detail(), "3d_corner_authority_iteration_limit");
  }
  EXPECT_EQ(pops::all_reduce_sum(typed_rejection_seen ? 1L : 0L),
            static_cast<long>(pops::n_ranks()));
  EXPECT_EQ(engine.last_accepted_attempt(), static_cast<std::uint64_t>(3));
  EXPECT_EQ(hierarchy.accepted_revision(), revision_before_typed_rejection);
  for (std::size_t block = 0; block < 2; ++block)
    for (std::size_t level = 0; level < 3; ++level)
      EXPECT_EQ(snapshot_bits(hierarchy.state(block, level)), before_typed_rejection[block][level]);
  expect_all_ledgers_attempt(engine, 3);

  callback_order.clear();
  reflux_order.clear();
  engine.advance(typed_window, advance, reconcile, validate);
  EXPECT_EQ(engine.last_accepted_attempt(), static_cast<std::uint64_t>(5));
  EXPECT_EQ(callback_order.size(), static_cast<std::size_t>(20));
  expect_all_ledgers_attempt(engine, 5);
}

int run_mpi_amr_program_3d_corner_authority(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int failure = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
      if (pops::n_ranks() != kRequiredRanks)
        ADD_FAILURE() << "test_mpi_amr_program_3d_corner_authority requires exactly "
                      << kRequiredRanks << " MPI ranks, got " << pops::n_ranks();
      if constexpr (pops::kNativeDimension != 3)
        ADD_FAILURE() << "test_mpi_amr_program_3d_corner_authority requires POPS_NATIVE_DIM=3, got "
                      << pops::kNativeDimension;
      if (pops::n_ranks() == kRequiredRanks && pops::kNativeDimension == 3) {
        prove_exact_three_dimensional_corner_exchange<3>();
        prove_three_level_two_block_subcycling<3>();
      }
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d exact 3-D corner/AMR authority proof failed: %s\n",
                   pops::my_rank(), error.what());
      failure = 1;
    }
    failure = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(failure || ::testing::Test::HasFailure())));
    if (pops::my_rank() == 0 && failure == 0)
      std::printf("OK test_mpi_amr_program_3d_corner_authority np=%d dim=%d corner+subcycling\n",
                  pops::n_ranks(), pops::kNativeDimension);
  }
  pops::comm_finalize();
  return failure;
}

}  // namespace

TEST(test_mpi_amr_program_3d_corner_authority, ExactEightRankCornerAndTransactionalSubcycling) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_amr_program_3d_corner_authority,
                                    "test_mpi_amr_program_3d_corner_authority"),
            0);
}
