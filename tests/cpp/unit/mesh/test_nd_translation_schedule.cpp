#include <gtest/gtest.h>

#include <pops/mesh/nd_proof/translation_schedule.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace pops;
using namespace pops::mesh::nd_proof;

namespace {

template <int Dim>
TranslationScheduleBudget schedule_budget(std::size_t jobs = 512, std::size_t peers = 64,
                                          std::size_t local = 4096, std::size_t send = 4096,
                                          std::size_t receive = 4096) {
  return TranslationScheduleBudget{
      jobs, peers,   local,
      send, receive, LocalNeighborWorkBudget{512, jobs, {512, 200000}, {200000, 200000}}};
}

constexpr BoxHashBudget kHashBudget{4096, 4096, 4096};

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
Real value_for(const Index<Dim>& index, int component) {
  Real value = static_cast<Real>(component * 10000);
  Real scale = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    value += scale * static_cast<Real>(index[axis]);
    scale *= 97;
  }
  return value;
}

template <int Dim, class MemorySpace>
void fill_valid(MultiFab<Dim, MemorySpace>& fields, Real ghost_value = Real{-777}) {
  for (const std::size_t global_box : fields.local_global_indices()) {
    auto& fab = fields.fab(global_box);
    auto host = fab.create_host_mirror();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t cells = static_cast<std::size_t>(grown.numPts());
    for (int component = 0; component < fab.ncomp(); ++component)
      for (std::size_t cell = 0; cell < cells; ++cell) {
        const Index<Dim> index = index_from_cell(grown, cell);
        host(static_cast<std::size_t>(component) * cells + cell) =
            fab.box().contains(index) ? value_for(index, component) : ghost_value;
      }
    fab.copy_from_host(host);
  }
}

template <int Dim, class MemorySpace>
Real value_at(const MultiFab<Dim, MemorySpace>& fields, std::size_t global_box,
              const Index<Dim>& index, int component) {
  const auto& fab = fields.fab(global_box);
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

template <int Dim, class MemorySpace>
std::vector<Real> snapshot(const MultiFab<Dim, MemorySpace>& fields) {
  std::vector<Real> result;
  for (const std::size_t global_box : fields.local_global_indices()) {
    const auto& fab = fields.fab(global_box);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t element = 0; element < host.size(); ++element)
      result.push_back(host(element));
  }
  return result;
}

template <class Buffer>
std::vector<Real> snapshot_buffer(const Buffer& buffer) {
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, buffer);
  std::vector<Real> result;
  result.reserve(host.extent(0));
  for (std::size_t element = 0; element < host.extent(0); ++element)
    result.push_back(host(element));
  return result;
}

template <int Dim>
std::vector<Real> expected_payload(const typename TranslationSchedule<Dim>::Job& job,
                                   int first_component, int component_count) {
  std::vector<Real> result;
  const std::size_t cells = static_cast<std::size_t>(job.destination_region.numPts());
  result.reserve(job.elements);
  for (int component = first_component; component < first_component + component_count; ++component)
    for (std::size_t cell = 0; cell < cells; ++cell) {
      const Index<Dim> destination = index_from_cell(job.destination_region, cell);
      Index<Dim> source{};
      for (int axis = 0; axis < Dim; ++axis)
        source[axis] = static_cast<int>(static_cast<std::int64_t>(destination[axis]) +
                                        job.source_from_destination[axis]);
      result.push_back(value_for(source, component));
    }
  return result;
}

template <int Dim>
void expect_partitioned_two_rank_multi_job_transfer() {
  Index<Dim> lower{};
  Index<Dim> upper{};
  lower.values[0] = 0;
  upper.values[0] = 5;
  for (int axis = 1; axis < Dim; ++axis) {
    lower.values[axis] = 0;
    upper.values[axis] = 1;
  }
  const Box<Dim> domain{lower, upper};
  std::vector<Box<Dim>> boxes;
  for (int slab = 0; slab < 3; ++slab) {
    Index<Dim> slab_lower = lower;
    Index<Dim> slab_upper = upper;
    slab_lower.values[0] = 2 * slab;
    slab_upper.values[0] = 2 * slab + 1;
    boxes.push_back(Box<Dim>{slab_lower, slab_upper});
  }
  const BoxArray<Dim> layout(std::move(boxes));
  Extent<Dim> rank_extent{};
  rank_extent.values[0] = 2;
  for (int axis = 1; axis < Dim; ++axis)
    rank_extent.values[axis] = 1;
  const RankSpace<Dim> ranks{Index<Dim>{}, rank_extent};
  const Index<Dim> rank0{};
  Index<Dim> rank1{};
  rank1.values[0] = 1;
  const auto distribution = Distribution<Dim>::partitioned(layout, ranks, {rank0, rank1, rank0});
  std::array<int, Dim> hash_bins{};
  hash_bins.fill(2);
  Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    ghosts.values[axis] = 1;
  TranslationSchedule<Dim> sender(layout, distribution, domain, PeriodicTopology<Dim>{}, ghosts, 3,
                                  1, 2, rank0, hash_bins, kHashBudget, schedule_budget<Dim>());
  TranslationSchedule<Dim> receiver(layout, distribution, domain, PeriodicTopology<Dim>{}, ghosts,
                                    3, 1, 2, rank1, hash_bins, kHashBudget, schedule_budget<Dim>());
  const auto& send = sender.send_plan(rank1);
  const auto& receive = receiver.receive_plan(rank0);
  ASSERT_EQ(send.jobs.size(), 2U);
  EXPECT_EQ(send.jobs, receive.jobs);
  EXPECT_EQ(send.elements, receive.elements);
  EXPECT_EQ(send.jobs[0].offset, 0U);
  EXPECT_EQ(send.jobs[1].offset, send.jobs[0].elements);
  EXPECT_GT(send.jobs[1].offset, 0U);

  const auto& reverse_send = receiver.send_plan(rank0);
  const auto& reverse_receive = sender.receive_plan(rank1);
  EXPECT_EQ(reverse_send.jobs, reverse_receive.jobs);
  EXPECT_EQ(reverse_send.elements, reverse_receive.elements);
  EXPECT_EQ(reverse_send.jobs.size(), 2U);

  MultiFab<Dim> source(layout, distribution, rank0, 3, ghosts);
  MultiFab<Dim> destination(layout, distribution, rank1, 3, ghosts);
  fill_valid(source);
  fill_valid(destination);
  typename TranslationSchedule<Dim>::buffer_type buffer("translation_multi_job", send.elements);
  sender.pack(source, rank1, buffer);
  std::vector<Real> expected;
  for (const auto& job : send.jobs) {
    const std::vector<Real> job_payload = expected_payload<Dim>(job, 1, 2);
    expected.insert(expected.end(), job_payload.begin(), job_payload.end());
  }
  EXPECT_EQ(snapshot_buffer(buffer), expected);
  destination.fab(1).set_val(Real{-113});
  receiver.unpack(destination, rank0, buffer);
  for (const auto& job : receive.jobs) {
    const std::size_t cells = static_cast<std::size_t>(job.destination_region.numPts());
    for (int component = 1; component <= 2; ++component)
      for (std::size_t cell = 0; cell < cells; ++cell) {
        const Index<Dim> destination_index = index_from_cell(job.destination_region, cell);
        Index<Dim> source_index{};
        for (int axis = 0; axis < Dim; ++axis)
          source_index.values[axis] =
              static_cast<int>(static_cast<std::int64_t>(destination_index.values[axis]) +
                               job.source_from_destination[axis]);
        EXPECT_EQ(value_at(destination, job.destination_box, destination_index, component),
                  value_for(source_index, component));
      }
  }
}

}  // namespace

TEST(test_nd_translation_schedule,
     partitioned_two_rank_multi_job_payloads_are_identical_in_dim1_dim2_and_dim3) {
  expect_partitioned_two_rank_multi_job_transfer<1>();
  expect_partitioned_two_rank_multi_job_transfer<2>();
  expect_partitioned_two_rank_multi_job_transfer<3>();
}

TEST(test_nd_translation_schedule,
     partitioned_2d_pack_unpack_has_shared_ordinals_and_component_axis_zero_order) {
  const Box<2> domain{Index<2>{0, 0}, Index<2>{2, 1}};
  const BoxArray<2> layout(std::vector<Box<2>>{Box<2>{Index<2>{0, 0}, Index<2>{2, 0}},
                                               Box<2>{Index<2>{0, 1}, Index<2>{2, 1}}});
  const RankSpace<2> ranks{Index<2>{4, -2}, Extent<2>{2, 1}};
  const Index<2> sender_rank{4, -2};
  const Index<2> receiver_rank{5, -2};
  const auto distribution =
      Distribution<2>::partitioned(layout, ranks, {sender_rank, receiver_rank});
  const auto topology = PeriodicTopology<2>{};
  const auto budget = schedule_budget<2>();
  TranslationSchedule<2> sender(layout, distribution, domain, topology, Extent<2>{1, 1}, 3, 1, 2,
                                sender_rank, {3, 1}, kHashBudget, budget);
  TranslationSchedule<2> receiver(layout, distribution, domain, topology, Extent<2>{1, 1}, 3, 1, 2,
                                  receiver_rank, {3, 1}, kHashBudget, budget);

  ASSERT_EQ(sender.send_plan_count(), 1U);
  ASSERT_EQ(receiver.receive_plan_count(), 1U);
  const auto& send = sender.send_plan(receiver_rank);
  const auto& receive = receiver.receive_plan(sender_rank);
  ASSERT_EQ(send.jobs.size(), 1U);
  EXPECT_EQ(send.jobs, receive.jobs);
  EXPECT_EQ(send.elements, receive.elements);
  EXPECT_EQ(send.jobs[0].ordinal, receive.jobs[0].ordinal);
  EXPECT_EQ(send.jobs[0].destination_region, (Box<2>{Index<2>{0, 0}, Index<2>{2, 0}}));
  EXPECT_EQ(send.elements, 6U);
  EXPECT_EQ(send.jobs[0].offset, 0U);

  MultiFab<2> source(layout, distribution, sender_rank, 3, Extent<2>{1, 1});
  MultiFab<2> destination(layout, distribution, receiver_rank, 3, Extent<2>{1, 1});
  fill_valid(source);
  fill_valid(destination);
  typename TranslationSchedule<2>::buffer_type buffer("translation_payload", send.elements);
  Kokkos::deep_copy(buffer, Real{-31});
  sender.pack(source, receiver_rank, buffer);
  const std::vector<Real> expected = expected_payload<2>(send.jobs[0], 1, 2);
  EXPECT_EQ(snapshot_buffer(buffer), expected);
  EXPECT_EQ(expected, (std::vector<Real>{10000, 10001, 10002, 20000, 20001, 20002}));

  destination.fab(1).set_val(Real{-19});
  receiver.unpack(destination, sender_rank, buffer);
  for (int component = 1; component <= 2; ++component)
    for (int x = 0; x <= 2; ++x)
      EXPECT_EQ(value_at(destination, 1, Index<2>{x, 0}, component),
                value_for(Index<2>{x, 0}, component));
}

TEST(test_nd_translation_schedule, replicated_dim1_and_deep_dim3_periodic_replay_are_local_only) {
  const Box<1> line_domain{Index<1>{0}, Index<1>{2}};
  const BoxArray<1> line_layout(std::vector<Box<1>>{line_domain});
  const RankSpace<1> line_ranks{Index<1>{-3}, Extent<1>{1}};
  const auto line_distribution = Distribution<1>::replicated(line_layout, line_ranks);
  MultiFab<1> line(line_layout, line_distribution, Index<1>{-3}, 2, Extent<1>{1});
  fill_valid(line);
  TranslationSchedule<1> line_schedule(
      line_layout, line_distribution, line_domain, PeriodicTopology<1>::axis_translations({true}),
      Extent<1>{1}, 2, 1, 1, Index<1>{-3}, {3}, kHashBudget, schedule_budget<1>());
  EXPECT_FALSE(line_schedule.local_jobs().empty());
  EXPECT_EQ(line_schedule.send_plan_count(), 0U);
  EXPECT_EQ(line_schedule.receive_plan_count(), 0U);
  line_schedule.replay(line);
  EXPECT_EQ(value_at(line, 0, Index<1>{-1}, 1), value_for(Index<1>{2}, 1));
  EXPECT_EQ(value_at(line, 0, Index<1>{3}, 1), value_for(Index<1>{0}, 1));

  const Box<3> point{Index<3>{0, 0, 0}, Index<3>{0, 0, 0}};
  const BoxArray<3> volume_layout(std::vector<Box<3>>{point});
  const RankSpace<3> volume_ranks{Index<3>{1, -2, 7}, Extent<3>{1, 1, 1}};
  const auto volume_distribution = Distribution<3>::replicated(volume_layout, volume_ranks);
  TranslationSchedule<3> volume_schedule(volume_layout, volume_distribution, point,
                                         PeriodicTopology<3>::axis_translations({true, true, true}),
                                         Extent<3>{2, 2, 2}, 1, 0, 1, Index<3>{1, -2, 7}, {1, 1, 1},
                                         kHashBudget, schedule_budget<3>(256));
  ASSERT_EQ(volume_schedule.global_job_count(), 124U);
  ASSERT_EQ(volume_schedule.local_job_count(), 124U);
  EXPECT_EQ(volume_schedule.send_plan_count(), 0U);
  EXPECT_EQ(volume_schedule.receive_plan_count(), 0U);
  for (std::size_t job = 0; job < volume_schedule.local_jobs().size(); ++job)
    EXPECT_EQ(volume_schedule.local_jobs()[job].ordinal, job);
  MultiFab<3> volume(volume_layout, volume_distribution, Index<3>{1, -2, 7}, 1, Extent<3>{2, 2, 2});
  fill_valid(volume);
  volume_schedule.replay(volume);
  EXPECT_EQ(value_at(volume, 0, Index<3>{-2, 2, -1}, 0), value_for(Index<3>{0, 0, 0}, 0));
}

TEST(test_nd_translation_schedule, peer_plans_sort_in_rank_space_order_and_budgets_are_cumulative) {
  const Box<1> domain{Index<1>{0}, Index<1>{2}};
  const BoxArray<1> layout(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{0}},
                                               Box<1>{Index<1>{1}, Index<1>{1}},
                                               Box<1>{Index<1>{2}, Index<1>{2}}});
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{3}};
  const Index<1> local{1};
  const auto distribution =
      Distribution<1>::partitioned(layout, ranks, {Index<1>{2}, local, Index<1>{0}});
  TranslationSchedule<1> schedule(layout, distribution, domain, PeriodicTopology<1>{}, Extent<1>{1},
                                  1, 0, 1, local, {1}, kHashBudget, schedule_budget<1>());
  ASSERT_EQ(schedule.send_plan_count(), 2U);
  ASSERT_EQ(schedule.receive_plan_count(), 2U);
  EXPECT_EQ(schedule.send_plans()[0].peer, (Index<1>{0}));
  EXPECT_EQ(schedule.send_plans()[1].peer, (Index<1>{2}));
  EXPECT_EQ(schedule.receive_plans()[0].peer, (Index<1>{0}));
  EXPECT_EQ(schedule.receive_plans()[1].peer, (Index<1>{2}));
  EXPECT_THROW((void)TranslationSchedule<1>(layout, distribution, domain, PeriodicTopology<1>{},
                                            Extent<1>{1}, 1, 0, 1, local, {1}, kHashBudget,
                                            schedule_budget<1>(32, 3)),
               std::length_error);
  EXPECT_THROW((void)TranslationSchedule<1>(layout, distribution, domain, PeriodicTopology<1>{},
                                            Extent<1>{1}, 1, 0, 1, local, {1}, kHashBudget,
                                            schedule_budget<1>(32, 8, 8, 1, 8)),
               std::length_error);
  EXPECT_THROW((void)TranslationSchedule<1>(layout, distribution, domain, PeriodicTopology<1>{},
                                            Extent<1>{1}, 1, 0, 1, local, {1}, kHashBudget,
                                            schedule_budget<1>(32, 8, 8, 8, 1)),
               std::length_error);
  const auto replicated = Distribution<1>::replicated(layout, ranks);
  EXPECT_THROW(
      (void)TranslationSchedule<1>(layout, replicated, domain, PeriodicTopology<1>{}, Extent<1>{1},
                                   1, 0, 1, local, {1}, kHashBudget, schedule_budget<1>(32, 0, 1)),
      std::length_error);
  EXPECT_THROW((void)TranslationSchedule<1>(layout, distribution, domain, PeriodicTopology<1>{},
                                            Extent<1>{1}, 1, 0, 1, local, {1}, kHashBudget,
                                            schedule_budget<1>(0)),
               std::length_error);
}

TEST(test_nd_translation_schedule, identity_and_buffer_refusals_leave_caller_storage_unchanged) {
  const Box<1> domain{Index<1>{0}, Index<1>{3}};
  const BoxArray<1> layout(
      std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{1}}, Box<1>{Index<1>{2}, Index<1>{3}}});
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{2}};
  const auto distribution = Distribution<1>::partitioned(layout, ranks, {Index<1>{0}, Index<1>{1}});
  TranslationSchedule<1> sender(layout, distribution, domain, PeriodicTopology<1>{}, Extent<1>{1},
                                2, 1, 1, Index<1>{0}, {1}, kHashBudget, schedule_budget<1>());
  TranslationSchedule<1> receiver(layout, distribution, domain, PeriodicTopology<1>{}, Extent<1>{1},
                                  2, 1, 1, Index<1>{1}, {1}, kHashBudget, schedule_budget<1>());
  MultiFab<1> source(layout, distribution, Index<1>{0}, 2, Extent<1>{1});
  MultiFab<1> destination(layout, distribution, Index<1>{1}, 2, Extent<1>{1});
  fill_valid(source);
  fill_valid(destination);
  const std::size_t elements = sender.send_plan(Index<1>{1}).elements;
  TranslationSchedule<1>::buffer_type buffer("refusal_buffer", elements);
  Kokkos::deep_copy(buffer, Real{42});
  const std::vector<Real> original_buffer = snapshot_buffer(buffer);
  const std::vector<Real> original_destination = snapshot(destination);

  TranslationSchedule<1>::buffer_type wrong("wrong_buffer", elements + 1);
  Kokkos::deep_copy(wrong, Real{17});
  const std::vector<Real> original_wrong = snapshot_buffer(wrong);
  EXPECT_THROW(sender.pack(source, Index<1>{1}, wrong), std::invalid_argument);
  EXPECT_EQ(snapshot_buffer(wrong), original_wrong);
  EXPECT_THROW(sender.pack(source, Index<1>{0}, buffer), std::invalid_argument);
  EXPECT_EQ(snapshot_buffer(buffer), original_buffer);
  EXPECT_THROW(receiver.unpack(destination, Index<1>{0}, wrong), std::invalid_argument);
  EXPECT_EQ(snapshot(destination), original_destination);
  EXPECT_EQ(snapshot_buffer(wrong), original_wrong);

  const BoxArray<1> regridded(
      std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{0}}, Box<1>{Index<1>{1}, Index<1>{3}}});
  const auto regridded_distribution =
      Distribution<1>::partitioned(regridded, ranks, {Index<1>{0}, Index<1>{1}});
  MultiFab<1> layout_stale(regridded, regridded_distribution, Index<1>{0}, 2, Extent<1>{1});
  fill_valid(layout_stale);
  EXPECT_THROW(sender.pack(layout_stale, Index<1>{1}, buffer), std::invalid_argument);
  EXPECT_EQ(snapshot_buffer(buffer), original_buffer);
  const BoxArray<1> reordered(std::vector<Box<1>>{layout[1], layout[0]});
  const auto reordered_distribution =
      Distribution<1>::partitioned(reordered, ranks, {Index<1>{0}, Index<1>{1}});
  MultiFab<1> reordered_stale(reordered, reordered_distribution, Index<1>{0}, 2, Extent<1>{1});
  fill_valid(reordered_stale);
  EXPECT_THROW(sender.pack(reordered_stale, Index<1>{1}, buffer), std::invalid_argument);
  EXPECT_EQ(snapshot_buffer(buffer), original_buffer);
  const auto changed_owners =
      Distribution<1>::partitioned(layout, ranks, {Index<1>{1}, Index<1>{0}});
  MultiFab<1> owner_stale(layout, changed_owners, Index<1>{0}, 2, Extent<1>{1});
  fill_valid(owner_stale);
  EXPECT_THROW(sender.pack(owner_stale, Index<1>{1}, buffer), std::invalid_argument);
  EXPECT_EQ(snapshot_buffer(buffer), original_buffer);
  MultiFab<1> rank_stale(layout, distribution, Index<1>{1}, 2, Extent<1>{1});
  fill_valid(rank_stale);
  EXPECT_THROW(sender.pack(rank_stale, Index<1>{1}, buffer), std::invalid_argument);
  EXPECT_EQ(snapshot_buffer(buffer), original_buffer);
  MultiFab<1> ghosts_stale(layout, distribution, Index<1>{0}, 2, Extent<1>{2});
  fill_valid(ghosts_stale);
  EXPECT_THROW(sender.pack(ghosts_stale, Index<1>{1}, buffer), std::invalid_argument);
  EXPECT_EQ(snapshot_buffer(buffer), original_buffer);
  MultiFab<1> ncomp_stale(layout, distribution, Index<1>{0}, 3, Extent<1>{1});
  fill_valid(ncomp_stale);
  EXPECT_THROW(sender.pack(ncomp_stale, Index<1>{1}, buffer), std::invalid_argument);
  EXPECT_EQ(snapshot_buffer(buffer), original_buffer);
  const auto replicated = Distribution<1>::replicated(layout, ranks);
  MultiFab<1> mode_stale(layout, replicated, Index<1>{0}, 2, Extent<1>{1});
  fill_valid(mode_stale);
  EXPECT_THROW(sender.pack(mode_stale, Index<1>{1}, buffer), std::invalid_argument);
  EXPECT_EQ(snapshot_buffer(buffer), original_buffer);

  destination.fab(1).set_val(Real{-5});
  const std::vector<Real> before_unpack = snapshot(destination);
  EXPECT_THROW(receiver.unpack(destination, Index<1>{1}, buffer), std::invalid_argument);
  EXPECT_EQ(snapshot(destination), before_unpack);
  const std::vector<Real> before_replay = snapshot(destination);
  EXPECT_THROW(sender.replay(destination), std::invalid_argument);
  EXPECT_EQ(snapshot(destination), before_replay);
}

TEST(test_nd_translation_schedule, metadata_and_large_3d_element_overflow_fail_before_storage) {
  const Box<1> domain{Index<1>{0}, Index<1>{1}};
  const BoxArray<1> layout(
      std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{0}}, Box<1>{Index<1>{1}, Index<1>{1}}});
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{2}};
  const auto distribution = Distribution<1>::partitioned(layout, ranks, {Index<1>{0}, Index<1>{1}});
  const auto good = schedule_budget<1>();
  EXPECT_THROW(
      (void)TranslationSchedule<1>(layout, distribution, Box<1>{}, PeriodicTopology<1>{},
                                   Extent<1>{1}, 1, 0, 1, Index<1>{0}, {1}, kHashBudget, good),
      std::invalid_argument);
  EXPECT_THROW(
      (void)TranslationSchedule<1>(layout, distribution, domain, PeriodicTopology<1>{},
                                   Extent<1>{-1}, 1, 0, 1, Index<1>{0}, {1}, kHashBudget, good),
      std::invalid_argument);
  EXPECT_THROW(
      (void)TranslationSchedule<1>(layout, distribution, domain, PeriodicTopology<1>{},
                                   Extent<1>{1}, 1, 1, 1, Index<1>{0}, {1}, kHashBudget, good),
      std::invalid_argument);
  EXPECT_THROW(
      (void)TranslationSchedule<1>(layout, distribution, domain, PeriodicTopology<1>{},
                                   Extent<1>{1}, 1, 0, 1, Index<1>{2}, {1}, kHashBudget, good),
      std::invalid_argument);
  const Box<2> plane{Index<2>{0, 0}, Index<2>{1, 1}};
  const BoxArray<2> plane_layout(std::vector<Box<2>>{plane});
  const RankSpace<2> plane_ranks{Index<2>{0, 0}, Extent<2>{1, 1}};
  const auto plane_distribution = Distribution<2>::replicated(plane_layout, plane_ranks);
  const PeriodicTopology<2> mapped{std::vector<PeriodicIdentification<2>>{PeriodicIdentification<2>{
      Face<2>{0, Side::lower}, Face<2>{1, Side::upper}, SignedPermutation<2>{{1, 0}, {1, -1}}}}};
  EXPECT_THROW((void)TranslationSchedule<2>(plane_layout, plane_distribution, plane, mapped,
                                            Extent<2>{1, 1}, 1, 0, 1, Index<2>{0, 0}, {2, 2},
                                            kHashBudget, schedule_budget<2>()),
               std::invalid_argument);

  constexpr int minimum = std::numeric_limits<int>::min();
  constexpr int maximum = std::numeric_limits<int>::max();
  const Box<3> huge_domain{Index<3>{0, minimum, minimum}, Index<3>{1, maximum, maximum}};
  const BoxArray<3> huge_layout(
      std::vector<Box<3>>{Box<3>{Index<3>{0, minimum, minimum}, Index<3>{0, maximum, maximum}},
                          Box<3>{Index<3>{1, minimum, minimum}, Index<3>{1, maximum, maximum}}});
  const RankSpace<3> huge_ranks{Index<3>{0, 0, 0}, Extent<3>{1, 1, 1}};
  const auto huge_distribution = Distribution<3>::replicated(huge_layout, huge_ranks);
  const Box<3> execution_domain{Index<3>{0, minimum, 0}, Index<3>{1, maximum, 1073741823}};
  const BoxArray<3> execution_layout(
      std::vector<Box<3>>{Box<3>{Index<3>{0, minimum, 0}, Index<3>{0, maximum, 1073741823}},
                          Box<3>{Index<3>{1, minimum, 0}, Index<3>{1, maximum, 1073741823}}});
  const auto execution_distribution = Distribution<3>::replicated(execution_layout, huge_ranks);
  EXPECT_THROW((void)TranslationSchedule<3>(
                   execution_layout, execution_distribution, execution_domain,
                   PeriodicTopology<3>{}, Extent<3>{1, 0, 0}, 3, 0, 3, Index<3>{0, 0, 0},
                   {maximum, maximum, maximum}, BoxHashBudget{64, 64, 64}, schedule_budget<3>(32)),
               std::overflow_error);
  EXPECT_THROW((void)TranslationSchedule<3>(huge_layout, huge_distribution, huge_domain,
                                            PeriodicTopology<3>{}, Extent<3>{1, 0, 0}, 2, 0, 2,
                                            Index<3>{0, 0, 0}, {maximum, maximum, maximum},
                                            BoxHashBudget{64, 64, 64}, schedule_budget<3>(32)),
               std::overflow_error);
}
