#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/nd_proof/translation_exchange.hpp>
#include <pops/parallel/comm.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <latch>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace pops;
using namespace pops::mesh::nd_proof;
using pops::mesh::Distribution;
using pops::mesh::RankSpace;

template <int Dim>
using ProductionBoxArray = pops::mesh::BoxArray<Dim>;

static_assert(std::is_nothrow_move_assignable_v<ExecutionLane>);

namespace {

constexpr Real kGhost = Real{-777};

template <int Dim>
TranslationScheduleBudget budget() {
  return TranslationScheduleBudget{
      4096,  128,   65536,
      65536, 65536, LocalNeighborWorkBudget{4096, 4096, {4096, 1'000'000}, {1'000'000, 1'000'000}}};
}

template <int Dim>
Index<Dim> rank_coordinate(int rank) {
  Index<Dim> coordinate{};
  coordinate[0] = rank;
  return coordinate;
}

template <int Dim>
Real value_for(const Index<Dim>& index, int component, Real bias) {
  Real value = bias + static_cast<Real>(component * 10'000);
  Real scale = Real{1};
  for (int axis = 0; axis < Dim; ++axis) {
    value += scale * static_cast<Real>(index[axis]);
    scale *= Real{97};
  }
  return value;
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
TranslationSchedule<Dim> make_schedule(int ranks, int rank, bool replicated, int boxes_per_rank,
                                       int ghost, int first_component = 1,
                                       int component_count = 2) {
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
    owners.push_back(rank_coordinate<Dim>(box % ranks));
  }
  const ProductionBoxArray<Dim> layout(std::move(boxes));
  Extent<Dim> rank_extent{};
  rank_extent[0] = ranks;
  for (int axis = 1; axis < Dim; ++axis)
    rank_extent[axis] = 1;
  const RankSpace<Dim> rank_space{Index<Dim>{}, rank_extent};
  const Distribution<Dim> distribution =
      replicated ? Distribution<Dim>::replicated(layout, rank_space)
                 : Distribution<Dim>::partitioned(layout, rank_space, std::move(owners));
  Extent<Dim> ghosts{};
  ghosts[0] = ghost;
  for (int axis = 1; axis < Dim; ++axis)
    ghosts[axis] = 1;
  std::array<int, Dim> hash_bins{};
  hash_bins.fill(2);
  std::array<bool, Dim> periodic{};
  periodic[0] = true;
  return TranslationSchedule<Dim>{layout,
                                  distribution,
                                  domain,
                                  PeriodicTopology<Dim>::axis_translations(periodic),
                                  ghosts,
                                  3,
                                  first_component,
                                  component_count,
                                  rank_coordinate<Dim>(rank),
                                  hash_bins,
                                  BoxHashBudget{4096, 4096, 4096},
                                  budget<Dim>()};
}

template <int Dim>
void fill_valid(MultiFab<Dim>& fields, Real bias) {
  for (std::size_t global_box : fields.local_global_indices()) {
    auto& fab = fields.fab_global(global_box);
    auto host = fab.create_host_mirror();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t cells = static_cast<std::size_t>(grown.numPts());
    for (int component = 0; component < fab.ncomp(); ++component)
      for (std::size_t cell = 0; cell < cells; ++cell) {
        const Index<Dim> index = index_from_cell(grown, cell);
        host(static_cast<std::size_t>(component) * cells + cell) =
            fab.box().contains(index) ? value_for(index, component, bias) : kGhost;
      }
    fab.copy_from_host(host);
  }
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
void expect_replayed(const TranslationSchedule<Dim>& schedule, const MultiFab<Dim>& fields,
                     Real bias, bool check_untouched) {
  const auto expect_job = [&](const typename TranslationSchedule<Dim>::Job& job) {
    for (int component = schedule.first_component();
         component < schedule.first_component() + schedule.component_count(); ++component)
      for (std::size_t cell = 0; cell < static_cast<std::size_t>(job.destination_region.numPts());
           ++cell) {
        const Index<Dim> destination = index_from_cell(job.destination_region, cell);
        Index<Dim> source{};
        for (int axis = 0; axis < Dim; ++axis)
          source[axis] = static_cast<int>(static_cast<std::int64_t>(destination[axis]) +
                                          job.source_from_destination[axis]);
        EXPECT_EQ(value_at(fields, job.destination_box, destination, component),
                  value_for(source, component, bias));
        if (check_untouched)
          EXPECT_EQ(value_at(fields, job.destination_box, destination, 0), kGhost);
      }
  };
  for (const auto& job : schedule.local_jobs())
    expect_job(job);
  for (const auto& plan : schedule.receive_plans())
    for (const auto& job : plan.jobs)
      expect_job(job);
}

template <int Dim>
MultiFab<Dim> make_fields(const TranslationSchedule<Dim>& schedule, int rank, Real bias) {
  MultiFab<Dim> fields(schedule.layout(), schedule.distribution(), rank_coordinate<Dim>(rank), 3,
                       schedule.ghosts());
  fill_valid(fields, bias);
  return fields;
}

template <int Dim>
void expect_cross_rank_plan_structure(const TranslationSchedule<Dim>& schedule,
                                      const ExecutionLane& lane) {
  bool multi_job_receive = false;
  bool later_job_offset = false;
  bool periodic_remote_job = false;
  for (const auto& plan : schedule.receive_plans()) {
    multi_job_receive = multi_job_receive || plan.jobs.size() > 1;
    for (const auto& job : plan.jobs) {
      later_job_offset = later_job_offset || job.offset > 0;
      for (int axis = 0; axis < Dim; ++axis)
        periodic_remote_job = periodic_remote_job || job.source_from_destination[axis] != 0;
    }
  }

  // The alternating two-box-per-rank layout gives every rank a multi-job receive plan with a
  // checked later offset. Only the two end ranks own periodic-wrap receives, so that witness is
  // intentionally collective-global rather than per-rank.
  EXPECT_TRUE(multi_job_receive);
  EXPECT_TRUE(later_job_offset);
  EXPECT_EQ(all_reduce_max(multi_job_receive ? 0L : 1L, lane.communicator()), 0L);
  EXPECT_EQ(all_reduce_max(later_job_offset ? 0L : 1L, lane.communicator()), 0L);
  EXPECT_EQ(all_reduce_max(periodic_remote_job ? 1L : 0L, lane.communicator()), 1L);
}

template <int Dim>
void expect_two_replays(int ranks, int rank, bool replicated) {
  auto schedule = make_schedule<Dim>(ranks, rank, replicated, 2, 1);
  auto lane = ExecutionLane::duplicate_world_collectively("nd-exchange-replay");
  TranslationExchange<Dim> exchange(schedule, lane, TranslationExchangeContext{17, 23});
  auto fields = make_fields(schedule, rank, Real{0});
  EXPECT_TRUE(lane.owns_communicator());
  EXPECT_EQ(exchange.diagnostic_stage(), TranslationExchangeDiagnosticStage::none);
  exchange.execute(fields, lane);
  expect_replayed(schedule, fields, Real{0}, true);
  EXPECT_EQ(exchange.live_request_count(), 0U);
  exchange.execute(fields, lane);
  expect_replayed(schedule, fields, Real{0}, true);
  EXPECT_FALSE(exchange.sealed());
  EXPECT_EQ(exchange.diagnostic_stage(), TranslationExchangeDiagnosticStage::none);
  EXPECT_EQ(exchange.live_request_count(), 0U);
}

void expect_unborrowed_lane_move_assignment() {
  auto destination = ExecutionLane::duplicate_world_collectively("nd-exchange-move-destination");
  auto source = ExecutionLane::duplicate_world_collectively("nd-exchange-move-source");
  const std::string source_identity(source.identity());
  destination = std::move(source);
  EXPECT_EQ(destination.identity(), source_identity);
  EXPECT_TRUE(destination.active());
  EXPECT_FALSE(source.active());
  destination = std::move(destination);
  EXPECT_EQ(destination.identity(), source_identity);
  EXPECT_TRUE(destination.active());
}

template <int Dim>
void expect_collective_constructor_failure(const TranslationSchedule<Dim>& schedule,
                                           const ExecutionLane& lane,
                                           TranslationExchangeContext context) {
  bool threw = false;
  try {
    TranslationExchange<Dim> exchange(schedule, lane, context);
    (void)exchange;
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_EQ(all_reduce_max(threw ? 0L : 1L, lane.communicator()), 0L);
}

template <int Dim>
void expect_sealed_failure(const TranslationSchedule<Dim>& schedule, const ExecutionLane& lane,
                           TranslationExchangeContext context,
                           TranslationExchangeDiagnosticStage expected_stage) {
  TranslationExchange<Dim> exchange(schedule, lane, context);
  auto fields = make_fields(schedule, lane.rank(), Real{0});
  bool threw = false;
  try {
    exchange.execute(fields, lane);
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_EQ(all_reduce_max(threw ? 0L : 1L, lane.communicator()), 0L);
  EXPECT_TRUE(exchange.sealed());
  EXPECT_EQ(exchange.diagnostic_stage(), expected_stage);
  EXPECT_EQ(exchange.live_request_count(), 0U);
  EXPECT_THROW(exchange.execute(fields, lane), std::runtime_error);
}

int run_mpi_nd_translation_exchange(int argc, char** argv) {
  comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    const int rank = my_rank();
    const int ranks = n_ranks();
    EXPECT_GE(mpi_thread_level(), MPI_THREAD_MULTIPLE);
    expect_unborrowed_lane_move_assignment();

    if (ranks == 1) {
      expect_two_replays<1>(ranks, rank, false);
      expect_two_replays<2>(ranks, rank, false);
      expect_two_replays<3>(ranks, rank, false);
      expect_two_replays<1>(ranks, rank, true);
      expect_two_replays<2>(ranks, rank, true);
      expect_two_replays<3>(ranks, rank, true);
    }

    if (ranks >= 2) {
      auto schedule_1d = make_schedule<1>(ranks, rank, false, 2, ranks == 2 ? 1 : 3);
      auto lane = ExecutionLane::duplicate_world_collectively("nd-exchange-traffic");
      TranslationExchange<1> exchange(schedule_1d, lane, TranslationExchangeContext{31, 37});
      auto fields = make_fields(schedule_1d, rank, Real{0});
      EXPECT_GT(schedule_1d.send_plan_count(), 0U);
      EXPECT_GT(schedule_1d.receive_plan_count(), 0U);
      EXPECT_GT(exchange.peer_count(), 0U);
      expect_cross_rank_plan_structure(schedule_1d, lane);
      exchange.execute(fields, lane);
      expect_replayed(schedule_1d, fields, Real{0}, true);
      EXPECT_EQ(exchange.live_request_count(), 0U);

      auto schedule_2d = make_schedule<2>(ranks, rank, false, 2, ranks == 2 ? 1 : 3);
      auto fields_2d = make_fields(schedule_2d, rank, Real{0});
      TranslationExchange<2> exchange_2d(schedule_2d, lane, TranslationExchangeContext{41, 43});
      if (ranks >= 4)
        EXPECT_GE(exchange_2d.peer_count(), 2U);
      expect_cross_rank_plan_structure(schedule_2d, lane);
      exchange_2d.execute(fields_2d, lane);
      expect_replayed(schedule_2d, fields_2d, Real{0}, true);

      auto schedule_3d = make_schedule<3>(ranks, rank, false, 2, ranks == 2 ? 1 : 3);
      auto fields_3d = make_fields(schedule_3d, rank, Real{0});
      TranslationExchange<3> exchange_3d(schedule_3d, lane, TranslationExchangeContext{47, 53});
      if (ranks >= 4)
        EXPECT_GE(exchange_3d.peer_count(), 2U);
      expect_cross_rank_plan_structure(schedule_3d, lane);
      exchange_3d.execute(fields_3d, lane);
      expect_replayed(schedule_3d, fields_3d, Real{0}, true);

      expect_collective_constructor_failure(
          schedule_1d, lane,
          TranslationExchangeContext{static_cast<std::uint64_t>(rank == 0 ? 59 : 61), 67});
      expect_collective_constructor_failure(
          schedule_1d, lane,
          TranslationExchangeContext{71, static_cast<std::uint64_t>(rank == 0 ? 73 : 79)});
      expect_collective_constructor_failure(
          schedule_1d, lane, TranslationExchangeContext{83, 89, 2, rank == 0 ? 0 : -1});

      expect_sealed_failure(schedule_1d, lane,
                            TranslationExchangeContext{79, 83, 2, -1, rank == 0 ? 0 : -1},
                            TranslationExchangeDiagnosticStage::receive_post);
      expect_sealed_failure(schedule_1d, lane,
                            TranslationExchangeContext{89, 97, 2, -1, -1, rank == 0 ? 0 : -1},
                            TranslationExchangeDiagnosticStage::send_post);
      expect_sealed_failure(schedule_1d, lane,
                            TranslationExchangeContext{101, 103, 2, -1, -1, -1, rank == 0 ? 0 : -1},
                            TranslationExchangeDiagnosticStage::wait);
    }

    if (ranks >= 2) {
      auto lane_a = ExecutionLane::duplicate_world_collectively("nd-exchange-concurrent-a");
      auto lane_b = ExecutionLane::duplicate_world_collectively("nd-exchange-concurrent-b");
      auto schedule_a = make_schedule<1>(ranks, rank, false, 2, 1);
      auto schedule_b = make_schedule<1>(ranks, rank, false, 2, 1);
      auto fields_a = make_fields(schedule_a, rank, Real{0});
      auto fields_b = make_fields(schedule_b, rank, Real{1'000'000});
      TranslationExchangeContext context_a{107, 109};
      TranslationExchangeContext context_b{113, 127};
      EXPECT_NE(lane_a.identity(), lane_b.identity());
      EXPECT_EQ(context_a.tag, context_b.tag);
      EXPECT_EQ(context_a.tag, ExecutionLane::translation_message_tag);
      EXPECT_NE(context_a.context_generation, 0U);
      EXPECT_NE(context_b.context_generation, 0U);
      EXPECT_NE(context_a.schedule_generation, 0U);
      EXPECT_NE(context_b.schedule_generation, 0U);
      EXPECT_NE(context_a.context_generation, context_b.context_generation);
      EXPECT_NE(context_a.schedule_generation, context_b.schedule_generation);
      int lane_relation = MPI_UNEQUAL;
      EXPECT_EQ(MPI_Comm_compare(lane_a.native_handle(), lane_b.native_handle(), &lane_relation),
                MPI_SUCCESS);
      EXPECT_EQ(lane_relation, MPI_CONGRUENT);
      TranslationExchange<1> exchange_a(schedule_a, lane_a, context_a);
      TranslationExchange<1> exchange_b(schedule_b, lane_b, context_b);
      std::exception_ptr failure_a;
      std::exception_ptr failure_b;
      std::latch workers_ready{2};
      std::latch release_workers{1};
      std::jthread first([&] {
        try {
          workers_ready.count_down();
          release_workers.wait();
          exchange_a.execute(fields_a, lane_a);
        } catch (...) {
          failure_a = std::current_exception();
        }
      });
      std::jthread second([&] {
        try {
          workers_ready.count_down();
          release_workers.wait();
          exchange_b.execute(fields_b, lane_b);
        } catch (...) {
          failure_b = std::current_exception();
        }
      });
      workers_ready.wait();
      release_workers.count_down();
      first.join();
      second.join();
      EXPECT_EQ(all_reduce_max((failure_a || failure_b) ? 1L : 0L), 0L);
      expect_replayed(schedule_a, fields_a, Real{0}, true);
      expect_replayed(schedule_b, fields_b, Real{1'000'000}, true);
      EXPECT_EQ(exchange_a.live_request_count(), 0U);
      EXPECT_EQ(exchange_b.live_request_count(), 0U);
    }
    result = ::testing::Test::HasFailure() ? 1 : 0;
  }
  comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_nd_translation_exchange, RunsProofMatrix) {
  EXPECT_EQ(
      pops::test::RunTestBody(&run_mpi_nd_translation_exchange, "test_mpi_nd_translation_exchange"),
      0);
}
