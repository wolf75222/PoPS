#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/parallel/copy_transport.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/parallel/load_balance.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace mesh_parallel = pops::mesh::parallel;

template <int Dim>
pops::Extent<Dim> filled_extent(std::int64_t value) {
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
pops::Index<Dim> index_from_ordinal(const pops::Box<Dim>& box, std::size_t ordinal) {
  pops::Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t extent = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(ordinal % extent);
    ordinal /= extent;
  }
  return index;
}

template <int Dim>
std::size_t field_offset(const pops::Box<Dim>& grown, const pops::Index<Dim>& index,
                         int component) {
  std::size_t cell = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    cell += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(grown.length(axis));
  }
  return static_cast<std::size_t>(component) * stride + cell;
}

template <int Dim>
pops::Real encoded_value(const pops::Index<Dim>& index, int component) {
  pops::Real result = static_cast<pops::Real>(component * 100'000);
  pops::Real scale = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result += scale * static_cast<pops::Real>(index[axis]);
    scale *= pops::Real{101};
  }
  return result;
}

template <int Dim>
void fill_valid(pops::MultiFab<Dim>& field) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const auto& box = fab.box();
    for (int component = 0; component < field.ncomp(); ++component)
      for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(box.numPts()); ++ordinal) {
        const auto index = index_from_ordinal(box, ordinal);
        host(field_offset(fab.grown_box(), index, component)) = encoded_value(index, component);
      }
    fab.copy_from_host(host);
  }
}

template <int Dim>
long count_invalid_values(const pops::MultiFab<Dim>& field) {
  long failures = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const auto& box = fab.box();
    for (int component = 0; component < field.ncomp(); ++component)
      for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(box.numPts()); ++ordinal) {
        const auto index = index_from_ordinal(box, ordinal);
        if (host(field_offset(fab.grown_box(), index, component)) !=
            encoded_value(index, component))
          ++failures;
      }
  }
  return failures;
}

template <int Dim>
std::vector<pops::Real> snapshot_local_bytes(const pops::MultiFab<Dim>& field) {
  std::vector<pops::Real> snapshot;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    snapshot.reserve(snapshot.size() + fab.size());
    for (std::size_t element = 0; element < fab.size(); ++element)
      snapshot.push_back(host(element));
  }
  return snapshot;
}

template <int Dim>
pops::CopyScheduleBudget exact_copy_budget(const pops::mesh::BoxArray<Dim>& destination,
                                           const pops::mesh::BoxArray<Dim>& source) {
  const auto overlap_pairs = [](std::size_t count) {
    return count < 2 ? std::size_t{0} : count * (count - 1) / 2;
  };
  return pops::CopyScheduleBudget{destination.size() * source.size(),
                                  destination.size() * source.size(),
                                  overlap_pairs(destination.size()), overlap_pairs(source.size())};
}

template <int Dim>
mesh_parallel::RegionTransferBudget exact_region_budget(const pops::CopySchedule<Dim>& schedule,
                                                        int components) {
  std::size_t elements = 0;
  for (const auto& job : schedule.canonical_jobs())
    elements +=
        static_cast<std::size_t>(job.region.numPts()) * static_cast<std::size_t>(components);
  return mesh_parallel::RegionTransferBudget{schedule.canonical_jobs().size(),
                                             schedule.source_distribution().rank_space().size(),
                                             elements, elements, elements};
}

template <int Dim>
std::vector<mesh_parallel::RegionTransferJob<Dim>> canonical_region_jobs(
    const pops::CopySchedule<Dim>& schedule) {
  std::vector<mesh_parallel::RegionTransferJob<Dim>> result;
  result.reserve(schedule.canonical_jobs().size());
  for (const auto& job : schedule.canonical_jobs())
    result.push_back(
        {job.source_box, job.destination_box, schedule.source_distribution().owner(job.source_box),
         schedule.destination_distribution().owner(job.destination_box), job.region, job.region});
  return result;
}

template <int Dim>
void execute_prepared_copy(const pops::CopySchedule<Dim>& schedule,
                           const pops::MultiFab<Dim>& source, pops::MultiFab<Dim>& destination,
                           const pops::ExecutionLane& lane) {
  mesh_parallel::PreparedCopyTransport<Dim> transport(
      schedule, source.ncomp(), exact_region_budget(schedule, source.ncomp()));
  transport.attach_lane(lane);
  transport.execute(destination, source);
}

template <int Dim>
void execute_direct_region_copy(const pops::CopySchedule<Dim>& schedule,
                                const pops::MultiFab<Dim>& source,
                                pops::MultiFab<Dim>& destination,
                                const pops::ExecutionLane& lane) {
  mesh_parallel::RegionTransport<Dim> transport(mesh_parallel::RegionTransferPlan<Dim>{
      schedule.source_distribution().rank_space(), schedule.local_rank(), source.ncomp(),
      canonical_region_jobs(schedule), exact_region_budget(schedule, source.ncomp())});
  transport.attach_lane(lane);
  auto source_view = [&source](const mesh_parallel::RegionTransferJob<Dim>& job) {
    return pops::FieldView<const pops::Real, Dim>(source.fab_global(job.source_patch).view());
  };
  auto destination_view = [&destination](const mesh_parallel::RegionTransferJob<Dim>& job) {
    return pops::FieldView<pops::Real, Dim>(destination.fab_global(job.destination_patch).view());
  };
  transport.execute(source_view, destination_view);
}

template <int Dim>
void prove_remote_redistribution(int rank_count, int rank) {
  static_assert(pops::MultiFab<Dim>::dimension == Dim);

  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = (axis == 0 ? rank_count * 8 : 4) - 1;
  const pops::Box<Dim> domain{pops::Index<Dim>{}, upper};
  const auto source_layout = pops::mesh::BoxArray<Dim>::from_domain(domain, filled_extent<Dim>(4));

  auto rank_extent = filled_extent<Dim>(1);
  rank_extent[0] = rank_count;
  const pops::mesh::RankSpace<Dim> rank_space{pops::Index<Dim>{}, rank_extent};
  const pops::parallel::LoadBalancePreparationBudget load_budget{
      source_layout.size(), static_cast<std::size_t>(rank_count), domain.numPts()};
  const auto source_plan = pops::parallel::LoadBalanceProvider<Dim>::space_filling_curve().prepare(
      source_layout, rank_space, load_budget);

  std::vector<pops::Box<Dim>> slabs;
  std::vector<pops::Index<Dim>> slab_owners;
  slabs.reserve(static_cast<std::size_t>(rank_count));
  slab_owners.reserve(static_cast<std::size_t>(rank_count));
  for (int slab = 0; slab < rank_count; ++slab) {
    pops::Index<Dim> lower{};
    pops::Index<Dim> slab_upper = upper;
    lower[0] = slab * 8;
    slab_upper[0] = lower[0] + 7;
    slabs.push_back(pops::Box<Dim>{lower, slab_upper});
    slab_owners.push_back(rank_coordinate<Dim>((slab + 1) % rank_count));
  }
  const pops::mesh::BoxArray<Dim> destination_layout(std::move(slabs));
  const auto destination_distribution = pops::mesh::Distribution<Dim>::partitioned(
      destination_layout, rank_space, std::move(slab_owners));
  const auto local_rank = rank_coordinate<Dim>(rank);

  pops::MultiFab<Dim> source(source_layout, source_plan.distribution(), local_rank, 2,
                             pops::Extent<Dim>{});
  pops::MultiFab<Dim> destination(destination_layout, destination_distribution, local_rank, 2,
                                  pops::Extent<Dim>{});
  fill_valid(source);
  destination.set_val(pops::Real{-1});

  const auto forward = pops::prepare_copy_schedule(
      destination, source, exact_copy_budget(destination.layout(), source.layout()));
  EXPECT_TRUE(forward.has_remote_jobs());
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test-mpi-redistribute-dim-" +
                                                                std::to_string(Dim));
  execute_prepared_copy(forward, source, destination, lane);
  EXPECT_EQ(pops::all_reduce_sum(count_invalid_values(destination)), 0L);

  pops::MultiFab<Dim> round_trip(source_layout, source_plan.distribution(), local_rank, 2,
                                 pops::Extent<Dim>{});
  round_trip.set_val(pops::Real{-2});
  const auto backward = pops::prepare_copy_schedule(
      round_trip, destination, exact_copy_budget(round_trip.layout(), destination.layout()));
  EXPECT_TRUE(backward.has_remote_jobs());
  execute_prepared_copy(backward, destination, round_trip, lane);
  EXPECT_EQ(pops::all_reduce_sum(count_invalid_values(round_trip)), 0L);

  const auto replicated_distribution =
      pops::mesh::Distribution<Dim>::replicated(source_layout, rank_space);
  pops::MultiFab<Dim> replicated_source(source_layout, replicated_distribution, local_rank, 2,
                                        pops::Extent<Dim>{});
  pops::MultiFab<Dim> replicated_destination(source_layout, replicated_distribution, local_rank, 2,
                                             pops::Extent<Dim>{});
  fill_valid(replicated_source);
  replicated_destination.set_val(pops::Real{-3});
  const auto replicated_schedule = pops::prepare_copy_schedule(
      replicated_destination, replicated_source,
      exact_copy_budget(replicated_destination.layout(), replicated_source.layout()));
  EXPECT_FALSE(replicated_schedule.has_remote_jobs());
  execute_prepared_copy(replicated_schedule, replicated_source, replicated_destination, lane);
  EXPECT_EQ(count_invalid_values(replicated_destination), 0L);

  // A rank-local disagreement about replicated versus partitioned storage must be rejected before
  // either branch attaches its RegionTransport or enters point-to-point communication.
  const auto mode_distribution = rank == 0 ? replicated_distribution : source_plan.distribution();
  pops::MultiFab<Dim> mode_source(source_layout, mode_distribution, local_rank, 2,
                                  pops::Extent<Dim>{});
  pops::MultiFab<Dim> mode_destination(source_layout, mode_distribution, local_rank, 2,
                                       pops::Extent<Dim>{});
  const auto mode_schedule = pops::prepare_copy_schedule(
      mode_destination, mode_source,
      exact_copy_budget(mode_destination.layout(), mode_source.layout()));
  mesh_parallel::PreparedCopyTransport<Dim> mode_transport(
      mode_schedule, mode_source.ncomp(), exact_region_budget(mode_schedule, mode_source.ncomp()));
  bool mode_divergence_rejected = false;
  try {
    mode_transport.attach_lane(lane);
  } catch (const std::invalid_argument&) {
    mode_divergence_rejected = true;
  }
  EXPECT_TRUE(mode_divergence_rejected);

  // Ownership is part of the canonical plan even when every rank selected the partitioned mode.
  // Mutating it on one rank cannot reach MPI_Isend/Irecv with an incompatible peer graph.
  std::vector<pops::Index<Dim>> divergent_owners = source_plan.distribution().owners();
  if (divergent_owners.size() < 2)
    throw std::logic_error("MPI redistribution divergence proof requires at least two patches");
  if (rank == 0)
    for (auto& owner : divergent_owners) {
      const std::size_t shifted = (rank_space.linear_rank(owner) + 1u) % rank_space.size();
      owner = rank_space.coordinate(shifted);
    }
  const auto divergent_distribution = pops::mesh::Distribution<Dim>::partitioned(
      source_layout, rank_space, std::move(divergent_owners));
  pops::MultiFab<Dim> divergent_destination(source_layout, divergent_distribution, local_rank, 2,
                                            pops::Extent<Dim>{});
  const auto divergent_schedule = pops::prepare_copy_schedule(
      divergent_destination, source,
      exact_copy_budget(divergent_destination.layout(), source.layout()));
  mesh_parallel::PreparedCopyTransport<Dim> divergent_transport(
      divergent_schedule, source.ncomp(), exact_region_budget(divergent_schedule, source.ncomp()));
  bool owner_divergence_rejected = false;
  try {
    divergent_transport.attach_lane(lane);
  } catch (const std::invalid_argument&) {
    owner_divergence_rejected = true;
  }
  EXPECT_TRUE(owner_divergence_rejected);

  // RegionTransport is also an SDK substrate used directly by FAC. Ordered jobs and owners must
  // agree before its rank-local peer graph or MPI request storage is prepared.
  destination.set_val(pops::Real{-4});
  const auto owner_divergence_candidate = snapshot_local_bytes(destination);
  auto owner_divergent_jobs = canonical_region_jobs(forward);
  if (owner_divergent_jobs.size() < 2)
    throw std::logic_error("direct region divergence proof requires at least two jobs");
  if (rank == 0) {
    std::swap(owner_divergent_jobs[0], owner_divergent_jobs[1]);
    auto& owner = owner_divergent_jobs[0].destination_rank;
    owner = rank_space.coordinate((rank_space.linear_rank(owner) + 1u) % rank_space.size());
  }
  mesh_parallel::RegionTransport<Dim> owner_divergent_direct(
      mesh_parallel::RegionTransferPlan<Dim>{
          rank_space, local_rank, source.ncomp(), std::move(owner_divergent_jobs),
          exact_region_budget(forward, source.ncomp())});
  bool direct_owner_divergence_rejected = false;
  try {
    owner_divergent_direct.attach_lane(lane);
  } catch (const std::invalid_argument&) {
    direct_owner_divergence_rejected = true;
  }
  EXPECT_TRUE(direct_owner_divergence_rejected);
  EXPECT_EQ(snapshot_local_bytes(destination), owner_divergence_candidate);
  execute_direct_region_copy(forward, source, destination, lane);
  EXPECT_EQ(pops::all_reduce_sum(count_invalid_values(destination)), 0L);

  // An empty direct plan models the replicated fast path's absent point-to-point transport. A
  // one-rank replicated/partitioned presence split must also fail before candidate publication.
  destination.set_val(pops::Real{-5});
  const auto presence_divergence_candidate = snapshot_local_bytes(destination);
  auto presence_divergent_jobs = canonical_region_jobs(forward);
  if (rank == 0)
    presence_divergent_jobs.clear();
  mesh_parallel::RegionTransport<Dim> presence_divergent_direct(
      mesh_parallel::RegionTransferPlan<Dim>{
          rank_space, local_rank, source.ncomp(), std::move(presence_divergent_jobs),
          exact_region_budget(forward, source.ncomp())});
  bool direct_presence_divergence_rejected = false;
  try {
    presence_divergent_direct.attach_lane(lane);
  } catch (const std::invalid_argument&) {
    direct_presence_divergence_rejected = true;
  }
  EXPECT_TRUE(direct_presence_divergence_rejected);
  EXPECT_EQ(snapshot_local_bytes(destination), presence_divergence_candidate);
  execute_direct_region_copy(forward, source, destination, lane);
  EXPECT_EQ(pops::all_reduce_sum(count_invalid_values(destination)), 0L);
}

int run_mpi_redistribute(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
      prove_remote_redistribution<pops::kNativeDimension>(pops::n_ranks(), pops::my_rank());
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d exact-rank MPI redistribution failed: %s\n", pops::my_rank(),
                   error.what());
      result = 1;
    }
    result = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(result || ::testing::Test::HasFailure())));
    if (pops::my_rank() == 0 && result == 0)
      std::printf("OK test_mpi_redistribute (np=%d dim=%d tiles<->slabs exact-rank)\n",
                  pops::n_ranks(), pops::kNativeDimension);
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_redistribute, NativeDimensionUsesPreparedExactRankMpiTransport) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_redistribute, "test_mpi_redistribute"), 0);
}
