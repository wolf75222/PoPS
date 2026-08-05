#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/parallel/comm.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace pops;
using namespace pops::mesh;

namespace {

constexpr Real kGhost = Real{-777};

template <int Dim>
struct HaloCase {
  Box<Dim> domain{};
  BoxArray<Dim> layout{};
  Distribution<Dim> distribution{};
  Extent<Dim> ghosts{};
  BoundaryTopology<Dim> topology{};
};

template <int Dim>
Index<Dim> rank_coordinate(int rank) {
  Index<Dim> coordinate{};
  coordinate[0] = rank;
  return coordinate;
}

template <int Dim>
HaloCase<Dim> make_case(int ranks, bool replicated) {
  constexpr int boxes_per_rank = 2;
  Index<Dim> lower{};
  Index<Dim> upper{};
  upper[0] = ranks * boxes_per_rank * 2 - 1;
  for (int axis = 1; axis < Dim; ++axis)
    upper[axis] = 2;
  const Box<Dim> domain{lower, upper};

  std::vector<Box<Dim>> boxes;
  std::vector<Index<Dim>> owners;
  boxes.reserve(static_cast<std::size_t>(ranks * boxes_per_rank));
  owners.reserve(static_cast<std::size_t>(ranks * boxes_per_rank));
  for (int box = 0; box < ranks * boxes_per_rank; ++box) {
    Index<Dim> box_lower = lower;
    Index<Dim> box_upper = upper;
    box_lower[0] = 2 * box;
    box_upper[0] = 2 * box + 1;
    boxes.push_back(Box<Dim>{box_lower, box_upper});
    owners.push_back(rank_coordinate<Dim>(box / boxes_per_rank));
  }
  const BoxArray<Dim> layout(std::move(boxes));

  Extent<Dim> rank_extent{};
  rank_extent[0] = ranks;
  for (int axis = 1; axis < Dim; ++axis)
    rank_extent[axis] = 1;
  const RankSpace<Dim> rank_space{Index<Dim>{}, rank_extent};
  const Distribution<Dim> distribution =
      replicated ? Distribution<Dim>::replicated(layout, rank_space)
                 : Distribution<Dim>::partitioned(layout, rank_space, std::move(owners));

  Extent<Dim> ghosts{};
  ghosts[0] = 3;
  for (int axis = 1; axis < Dim; ++axis)
    ghosts[axis] = 1;
  std::array<bool, Dim> periodic{};
  periodic.fill(true);
  return HaloCase<Dim>{domain, layout, distribution, ghosts,
                       BoundaryTopology<Dim>::axis_periodic(periodic)};
}

template <int Dim>
HaloScheduleBudget budget(std::size_t boxes) {
  constexpr std::size_t images = 64;
  return HaloScheduleBudget{{boxes, boxes * (boxes - 1) / 2},
                            boxes * boxes * images,
                            boxes * boxes * images * static_cast<std::size_t>(2 * Dim),
                            images,
                            boxes,
                            1'000'000,
                            1'000'000,
                            1'000'000};
}

template <int Dim>
Index<Dim> index_from_cell(const Box<Dim>& box, std::size_t cell) {
  Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t extent = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(cell % extent);
    cell /= extent;
  }
  return index;
}

template <int Dim>
Real encoded_value(const Index<Dim>& index, int component, Real bias) {
  Real value = bias + static_cast<Real>(component * 10'000);
  Real scale = Real{1};
  for (int axis = 0; axis < Dim; ++axis) {
    value += scale * static_cast<Real>(index[axis]);
    scale *= Real{97};
  }
  return value;
}

template <int Dim>
void fill_valid(MultiFab<Dim>& fields, Real bias) {
  for (const std::size_t global_box : fields.local_global_indices()) {
    auto& fab = fields.fab_global(global_box);
    auto host = fab.create_host_mirror();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t cells = static_cast<std::size_t>(grown.numPts());
    for (int component = 0; component < fab.ncomp(); ++component)
      for (std::size_t cell = 0; cell < cells; ++cell) {
        const Index<Dim> index = index_from_cell(grown, cell);
        host(static_cast<std::size_t>(component) * cells + cell) =
            fab.box().contains(index) ? encoded_value(index, component, bias) : kGhost;
      }
    fab.copy_from_host(host);
  }
}

template <int Dim>
void overwrite_valid(MultiFab<Dim>& fields, Real bias) {
  for (const std::size_t global_box : fields.local_global_indices()) {
    auto& fab = fields.fab_global(global_box);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t cells = static_cast<std::size_t>(grown.numPts());
    for (int component = 0; component < fab.ncomp(); ++component)
      for (std::size_t cell = 0; cell < cells; ++cell) {
        const Index<Dim> index = index_from_cell(grown, cell);
        if (fab.box().contains(index))
          host(static_cast<std::size_t>(component) * cells + cell) =
              encoded_value(index, component, bias);
      }
    fab.copy_from_host(host);
  }
}

template <int Dim>
std::vector<Real> snapshot(const MultiFab<Dim>& fields) {
  std::vector<Real> result;
  for (const std::size_t global_box : fields.local_global_indices()) {
    const auto& fab = fields.fab_global(global_box);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t element = 0; element < host.size(); ++element)
      result.push_back(host(element));
  }
  return result;
}

template <int Dim>
Real value_at(const MultiFab<Dim>& fields, std::size_t global_box, const Index<Dim>& index,
              int component) {
  const auto& fab = fields.fab_global(global_box);
  const Box<Dim>& grown = fab.grown_box();
  std::size_t stride = 1;
  std::size_t cell = 0;
  for (int axis = 0; axis < Dim; ++axis) {
    cell += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(grown.length(axis));
  }
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  return host(static_cast<std::size_t>(component) * stride + cell);
}

template <int Dim>
void expect_job_published(const HaloSchedule<Dim>& schedule, const MultiFab<Dim>& fields,
                          const HaloJob<Dim>& job, Real bias) {
  for (int component = 0; component < schedule.ncomp(); ++component)
    for (std::size_t cell = 0; cell < static_cast<std::size_t>(job.destination_region.numPts());
         ++cell) {
      const Index<Dim> destination = index_from_cell(job.destination_region, cell);
      Index<Dim> source = destination;
      for (int axis = 0; axis < Dim; ++axis)
        source[axis] += job.source_from_destination[axis];
      EXPECT_EQ(value_at(fields, job.destination_box, destination, component),
                encoded_value(source, component, bias));
    }
}

template <int Dim>
void expect_schedule_published(const HaloSchedule<Dim>& schedule, const MultiFab<Dim>& fields,
                               Real bias) {
  for (const HaloJob<Dim>& job : schedule.local_jobs())
    expect_job_published(schedule, fields, job, bias);
  for (const HaloPeerPlan<Dim>& plan : schedule.receive_plans())
    for (const HaloJob<Dim>& job : plan.jobs)
      expect_job_published(schedule, fields, job, bias);
}

template <int Dim>
void expect_plan_witnesses(const HaloSchedule<Dim>& schedule, const ExecutionLane& lane) {
  bool local = !schedule.local_jobs().empty();
  bool multi_job_receive = false;
  bool nonzero_offset = false;
  bool periodic_remote = false;
  for (const HaloPeerPlan<Dim>& plan : schedule.receive_plans()) {
    multi_job_receive = multi_job_receive || plan.jobs.size() > 1;
    for (const HaloJob<Dim>& job : plan.jobs) {
      nonzero_offset = nonzero_offset || job.offset != 0;
      for (int axis = 0; axis < Dim; ++axis)
        periodic_remote = periodic_remote || job.source_from_destination[axis] != 0;
    }
  }
  EXPECT_EQ(all_reduce_max(local ? 1L : 0L, lane.communicator()), 1L);
  EXPECT_EQ(all_reduce_max(multi_job_receive ? 1L : 0L, lane.communicator()), 1L);
  EXPECT_EQ(all_reduce_max(nonzero_offset ? 1L : 0L, lane.communicator()), 1L);
  EXPECT_EQ(all_reduce_max(periodic_remote ? 1L : 0L, lane.communicator()), 1L);
}

template <int Dim>
void expect_success(int ranks, int rank, bool replicated, const ExecutionLane& lane,
                    std::uint64_t generation) {
  const HaloCase<Dim> definition = make_case<Dim>(ranks, replicated);
  MultiFab<Dim> fields(definition.layout, definition.distribution, rank_coordinate<Dim>(rank), 3,
                       definition.ghosts);
  fill_valid(fields, Real{0});
  const HaloSchedule<Dim> schedule = prepare_halo_schedule(
      fields, definition.domain, definition.topology, budget<Dim>(definition.layout.size()));
  HaloExchangeContext context{};
  context.context_generation = generation;
  context.schedule_generation = generation + 1;
  HaloExchange<Dim> exchange(schedule, lane, context);

  if (!replicated && ranks >= 2) {
    EXPECT_TRUE(schedule.has_remote_jobs());
    expect_plan_witnesses(schedule, lane);
  }
  const std::vector<Real> before = snapshot(fields);
  exchange.begin(fields, lane);
  EXPECT_TRUE(exchange.in_flight());
  EXPECT_EQ(snapshot(fields), before);
  overwrite_valid(fields, Real{500'000});
  EXPECT_EQ(all_reduce_max(0L, lane.communicator()), 0L);
  exchange.complete(fields, lane);
  EXPECT_FALSE(exchange.in_flight());
  EXPECT_FALSE(exchange.sealed());
  EXPECT_EQ(exchange.live_request_count(), 0U);
  expect_schedule_published(schedule, fields, Real{0});

  fill_valid(fields, Real{1'000'000});
  const std::vector<Real> second_before = snapshot(fields);
  fill_boundary(fields, exchange, lane);
  EXPECT_NE(snapshot(fields), second_before);
  expect_schedule_published(schedule, fields, Real{1'000'000});
}

template <int Dim>
void expect_one_shot(int ranks, int rank, const ExecutionLane& lane, std::uint64_t generation) {
  const HaloCase<Dim> definition = make_case<Dim>(ranks, false);
  MultiFab<Dim> fields(definition.layout, definition.distribution, rank_coordinate<Dim>(rank), 2,
                       definition.ghosts);
  fill_valid(fields, Real{2'000'000});
  const HaloSchedule<Dim> schedule = prepare_halo_schedule(
      fields, definition.domain, definition.topology, budget<Dim>(definition.layout.size()));
  HaloExchangeContext context{};
  context.context_generation = generation;
  context.schedule_generation = generation + 1;
  fill_boundary(fields, definition.domain, definition.topology,
                budget<Dim>(definition.layout.size()), lane, context);
  expect_schedule_published(schedule, fields, Real{2'000'000});
}

void expect_collective_constructor_failures(int ranks, int rank, const ExecutionLane& lane) {
  const HaloCase<1> definition = make_case<1>(ranks, false);
  MultiFab<1> fields(definition.layout, definition.distribution, rank_coordinate<1>(rank), 2,
                     definition.ghosts);
  const HaloSchedule<1> schedule = prepare_halo_schedule(
      fields, definition.domain, definition.topology, budget<1>(definition.layout.size()));

  HaloExchangeContext mismatch{};
  mismatch.context_generation = static_cast<std::uint64_t>(rank == 0 ? 701 : 703);
  mismatch.schedule_generation = 709;
  bool mismatch_threw = false;
  try {
    HaloExchange<1> exchange(schedule, lane, mismatch);
    (void)exchange;
  } catch (const std::exception&) {
    mismatch_threw = true;
  }
  EXPECT_EQ(all_reduce_max(mismatch_threw ? 0L : 1L, lane.communicator()), 0L);

  HaloExchangeContext allocation{};
  allocation.context_generation = 719;
  allocation.schedule_generation = 727;
  allocation.fail_allocation_rank = 0;
  bool allocation_threw = false;
  try {
    HaloExchange<1> exchange(schedule, lane, allocation);
    (void)exchange;
  } catch (const std::exception&) {
    allocation_threw = true;
  }
  EXPECT_EQ(all_reduce_max(allocation_threw ? 0L : 1L, lane.communicator()), 0L);
}

void expect_sealed_failure(int ranks, int rank, const ExecutionLane& lane,
                           HaloExchangeDiagnosticStage stage, std::uint64_t generation) {
  const HaloCase<1> definition = make_case<1>(ranks, false);
  MultiFab<1> fields(definition.layout, definition.distribution, rank_coordinate<1>(rank), 2,
                     definition.ghosts);
  fill_valid(fields, Real{0});
  const HaloSchedule<1> schedule = prepare_halo_schedule(
      fields, definition.domain, definition.topology, budget<1>(definition.layout.size()));
  HaloExchangeContext context{};
  context.context_generation = generation;
  context.schedule_generation = generation + 1;
  if (stage == HaloExchangeDiagnosticStage::receive_post)
    context.fail_receive_post_rank = 0;
  else if (stage == HaloExchangeDiagnosticStage::send_post)
    context.fail_send_post_rank = 0;
  else if (stage == HaloExchangeDiagnosticStage::wait)
    context.fail_wait_rank = 0;
  else if (stage == HaloExchangeDiagnosticStage::staging)
    context.fail_staging_rank = 0;
  else
    throw std::invalid_argument("unsupported injected halo failure stage");

  HaloExchange<1> exchange(schedule, lane, context);
  const std::vector<Real> before = snapshot(fields);
  bool threw = false;
  try {
    exchange.begin(fields, lane);
    EXPECT_EQ(snapshot(fields), before);
    exchange.complete(fields, lane);
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_EQ(all_reduce_max(threw ? 0L : 1L, lane.communicator()), 0L);
  EXPECT_TRUE(exchange.sealed());
  EXPECT_EQ(exchange.diagnostic_stage(), stage);
  EXPECT_EQ(exchange.live_request_count(), 0U);
  EXPECT_EQ(snapshot(fields), before);

  bool reuse_threw = false;
  try {
    exchange.begin(fields, lane);
  } catch (const std::exception&) {
    reuse_threw = true;
  }
  EXPECT_EQ(all_reduce_max(reuse_threw ? 0L : 1L, lane.communicator()), 0L);
}

void expect_complete_requires_exact_field(int ranks, int rank, const ExecutionLane& lane) {
  const HaloCase<1> definition = make_case<1>(ranks, false);
  MultiFab<1> packed(definition.layout, definition.distribution, rank_coordinate<1>(rank), 2,
                     definition.ghosts);
  MultiFab<1> impostor(definition.layout, definition.distribution, rank_coordinate<1>(rank), 2,
                       definition.ghosts);
  fill_valid(packed, Real{0});
  fill_valid(impostor, Real{10'000'000});
  const HaloSchedule<1> schedule = prepare_halo_schedule(
      packed, definition.domain, definition.topology, budget<1>(definition.layout.size()));
  HaloExchangeContext context{};
  context.context_generation = 809;
  context.schedule_generation = 811;
  HaloExchange<1> exchange(schedule, lane, context);
  const std::vector<Real> packed_before = snapshot(packed);
  const std::vector<Real> impostor_before = snapshot(impostor);
  exchange.begin(packed, lane);
  bool threw = false;
  try {
    exchange.complete(impostor, lane);
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_EQ(all_reduce_max(threw ? 0L : 1L, lane.communicator()), 0L);
  EXPECT_TRUE(exchange.sealed());
  EXPECT_EQ(exchange.diagnostic_stage(), HaloExchangeDiagnosticStage::binding);
  EXPECT_EQ(exchange.live_request_count(), 0U);
  EXPECT_EQ(snapshot(packed), packed_before);
  EXPECT_EQ(snapshot(impostor), impostor_before);
}

void expect_complete_rejects_rebound_storage(int ranks, int rank, const ExecutionLane& lane) {
  const HaloCase<1> definition = make_case<1>(ranks, false);
  MultiFab<1> fields(definition.layout, definition.distribution, rank_coordinate<1>(rank), 2,
                     definition.ghosts);
  MultiFab<1> replacement(definition.layout, definition.distribution, rank_coordinate<1>(rank), 2,
                          definition.ghosts);
  fill_valid(fields, Real{0});
  fill_valid(replacement, Real{20'000'000});
  const HaloSchedule<1> schedule = prepare_halo_schedule(
      fields, definition.domain, definition.topology, budget<1>(definition.layout.size()));
  HaloExchangeContext context{};
  context.context_generation = 821;
  context.schedule_generation = 823;
  HaloExchange<1> exchange(schedule, lane, context);
  exchange.begin(fields, lane);
  fields = std::move(replacement);
  const std::vector<Real> rebound_before = snapshot(fields);
  bool threw = false;
  try {
    exchange.complete(fields, lane);
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_EQ(all_reduce_max(threw ? 0L : 1L, lane.communicator()), 0L);
  EXPECT_TRUE(exchange.sealed());
  EXPECT_EQ(exchange.diagnostic_stage(), HaloExchangeDiagnosticStage::binding);
  EXPECT_EQ(exchange.live_request_count(), 0U);
  EXPECT_EQ(snapshot(fields), rebound_before);
}

int run_mpi_halo_exchange(int argc, char** argv) {
  comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    const int rank = my_rank();
    const int ranks = n_ranks();
    auto lane = ExecutionLane::duplicate_world_collectively("production-halo-exchange");

    expect_success<1>(ranks, rank, false, lane, 101);
    expect_success<2>(ranks, rank, false, lane, 211);
    expect_success<3>(ranks, rank, false, lane, 307);
    expect_success<1>(ranks, rank, true, lane, 401);
    expect_success<2>(ranks, rank, true, lane, 503);
    expect_success<3>(ranks, rank, true, lane, 601);
    expect_one_shot<2>(ranks, rank, lane, 659);

    if (ranks >= 2) {
      expect_collective_constructor_failures(ranks, rank, lane);
      expect_sealed_failure(ranks, rank, lane, HaloExchangeDiagnosticStage::receive_post, 907);
      expect_sealed_failure(ranks, rank, lane, HaloExchangeDiagnosticStage::send_post, 919);
      expect_sealed_failure(ranks, rank, lane, HaloExchangeDiagnosticStage::wait, 929);
      expect_sealed_failure(ranks, rank, lane, HaloExchangeDiagnosticStage::staging, 937);
      expect_complete_requires_exact_field(ranks, rank, lane);
      expect_complete_rejects_rebound_storage(ranks, rank, lane);
    }
    result = ::testing::Test::HasFailure() ? 1 : 0;
  }
  comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_halo_exchange, RunsExactRankedProductionMatrix) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_halo_exchange, "test_mpi_halo_exchange"), 0);
}
