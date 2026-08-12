#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/parallel/region_transfer.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/parallel/load_balance.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace mesh_parallel = pops::mesh::parallel;

template <int Dim>
pops::Extent<Dim> uniform_extent(std::int64_t value) {
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
    const std::size_t length = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(ordinal % length);
    ordinal /= length;
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
    for (int component = 0; component < field.ncomp(); ++component)
      for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
           ++ordinal) {
        const auto index = index_from_ordinal(fab.box(), ordinal);
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
    for (int component = 0; component < field.ncomp(); ++component)
      for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
           ++ordinal) {
        const auto index = index_from_ordinal(fab.box(), ordinal);
        if (host(field_offset(fab.grown_box(), index, component)) !=
            encoded_value(index, component))
          ++failures;
      }
  }
  return failures;
}

template <int Dim>
std::vector<mesh_parallel::RegionTransferJob<Dim>> canonical_jobs(
    const pops::MultiFab<Dim>& destination, const pops::MultiFab<Dim>& source) {
  std::vector<mesh_parallel::RegionTransferJob<Dim>> jobs;
  for (std::size_t destination_patch = 0; destination_patch < destination.layout().size();
       ++destination_patch)
    for (std::size_t source_patch = 0; source_patch < source.layout().size(); ++source_patch) {
      const pops::Box<Dim> region =
          destination.layout()[destination_patch].intersect(source.layout()[source_patch]);
      if (!region.empty())
        jobs.push_back({source_patch, destination_patch, source.distribution().owner(source_patch),
                        destination.distribution().owner(destination_patch), region, region});
    }
  return jobs;
}

template <int Dim>
mesh_parallel::RegionTransferBudget region_budget(
    const std::vector<mesh_parallel::RegionTransferJob<Dim>>& jobs, int components,
    std::size_t ranks) {
  std::size_t elements = 0;
  for (const auto& job : jobs) {
    const auto cells = static_cast<std::size_t>(job.source_region.numPts());
    if (cells > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(components) ||
        elements >
            std::numeric_limits<std::size_t>::max() - cells * static_cast<std::size_t>(components))
      throw std::overflow_error("MPI redistribution region budget exceeds size_t");
    elements += cells * static_cast<std::size_t>(components);
  }
  return {jobs.size(), ranks, elements, elements, elements};
}

template <int Dim>
void redistribute(const pops::MultiFab<Dim>& source, pops::MultiFab<Dim>& destination,
                  const pops::ExecutionLane& lane) {
  auto jobs = canonical_jobs(destination, source);
  const auto budget = region_budget(jobs, source.ncomp(), source.rank_space().size());
  mesh_parallel::RegionTransport<Dim> transport(mesh_parallel::RegionTransferPlan<Dim>{
      source.rank_space(), source.local_rank(), source.ncomp(), std::move(jobs), budget});
  transport.prepare_collectively(lane);
  transport.execute(
      [&source](const auto& job) {
        return pops::FieldView<const pops::Real, Dim>(source.fab_global(job.source_patch).view());
      },
      [&destination](const auto& job) {
        return pops::FieldView<pops::Real, Dim>(
            destination.fab_global(job.destination_patch).view());
      });
}

template <int Dim>
void prove_conservative_redistribution(int rank_count, int rank) {
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = (axis == 0 ? rank_count * 8 : 8) - 1;
  const pops::Box<Dim> domain{pops::Index<Dim>{}, upper};
  const auto tiles = pops::mesh::BoxArray<Dim>::from_domain(domain, uniform_extent<Dim>(4));

  auto rank_extent = uniform_extent<Dim>(1);
  rank_extent[0] = rank_count;
  const pops::mesh::RankSpace<Dim> rank_space{pops::Index<Dim>{}, rank_extent};
  const pops::parallel::LoadBalancePreparationBudget balance_budget{
      tiles.size(), static_cast<std::size_t>(rank_count), domain.numPts()};
  const auto tile_plan = pops::parallel::LoadBalanceProvider<Dim>::space_filling_curve().prepare(
      tiles, rank_space, balance_budget);

  std::vector<pops::Box<Dim>> slabs;
  std::vector<pops::Index<Dim>> slab_owners;
  for (int slab = 0; slab < rank_count; ++slab) {
    pops::Index<Dim> lower{};
    pops::Index<Dim> slab_upper = upper;
    lower[0] = slab * 8;
    slab_upper[0] = lower[0] + 7;
    slabs.emplace_back(lower, slab_upper);
    slab_owners.push_back(rank_coordinate<Dim>((slab + 1) % rank_count));
  }
  const pops::mesh::BoxArray<Dim> bands(std::move(slabs));
  const auto band_distribution =
      pops::mesh::Distribution<Dim>::partitioned(bands, rank_space, std::move(slab_owners));
  const auto local_rank = rank_coordinate<Dim>(rank);

  pops::MultiFab<Dim> tiled(tiles, tile_plan.distribution(), local_rank, 2, pops::Extent<Dim>{});
  pops::MultiFab<Dim> banded(bands, band_distribution, local_rank, 2, pops::Extent<Dim>{});
  fill_valid(tiled);
  banded.set_val(pops::Real{-1});
  const pops::Real source_sum = pops::reduce_sum(tiled, 0);

  auto lane = pops::ExecutionLane::duplicate_world_collectively("pops.test.mpi-redistribute.dim-" +
                                                                std::to_string(Dim));
  redistribute(tiled, banded, lane);
  EXPECT_EQ(pops::all_reduce_sum(count_invalid_values(banded)), 0L);
  EXPECT_DOUBLE_EQ(pops::reduce_sum(banded, 0), source_sum);

  pops::MultiFab<Dim> round_trip(tiles, tile_plan.distribution(), local_rank, 2,
                                 pops::Extent<Dim>{});
  round_trip.set_val(pops::Real{-2});
  redistribute(banded, round_trip, lane);
  EXPECT_EQ(pops::all_reduce_sum(count_invalid_values(round_trip)), 0L);
  EXPECT_DOUBLE_EQ(pops::reduce_sum(round_trip, 0), source_sum);

  const auto replicated_distribution = pops::mesh::Distribution<Dim>::replicated(tiles, rank_space);
  pops::MultiFab<Dim> replicated_source(tiles, replicated_distribution, local_rank, 2,
                                        pops::Extent<Dim>{});
  pops::MultiFab<Dim> replicated_destination(tiles, replicated_distribution, local_rank, 2,
                                             pops::Extent<Dim>{});
  fill_valid(replicated_source);
  replicated_destination.set_val(pops::Real{-3});
  const pops::CopyScheduleBudget copy_budget{tiles.size() * tiles.size(), tiles.size(),
                                             tiles.size() * (tiles.size() - 1) / 2,
                                             tiles.size() * (tiles.size() - 1) / 2};
  const auto local_schedule =
      pops::prepare_copy_schedule(replicated_destination, replicated_source, copy_budget);
  EXPECT_FALSE(local_schedule.has_remote_jobs());
  pops::parallel_copy(replicated_destination, replicated_source, local_schedule);
  EXPECT_EQ(count_invalid_values(replicated_destination), 0L);

  if (rank_count > 1) {
    const auto forward_jobs = canonical_jobs(banded, tiled);
    const bool local_remote =
        std::any_of(forward_jobs.begin(), forward_jobs.end(),
                    [](const auto& job) { return job.source_rank != job.destination_rank; });
    EXPECT_TRUE(local_remote);
  }
}

int run_mpi_redistribute(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
      prove_conservative_redistribution<pops::kNativeDimension>(pops::n_ranks(), pops::my_rank());
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d exact-rank MPI redistribution failed: %s\n", pops::my_rank(),
                   error.what());
      result = 1;
    }
    result = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(result || ::testing::Test::HasFailure())));
    if (pops::my_rank() == 0 && result == 0)
      std::printf("OK test_mpi_redistribute (np=%d dim=%d conservative tiles<->bands)\n",
                  pops::n_ranks(), pops::kNativeDimension);
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_redistribute, NativeDimensionUsesPreparedRegionTransport) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_redistribute, "test_mpi_redistribute"), 0);
}
