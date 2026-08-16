#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/amr/prepared_amr_ghost_fill.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace {

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
pops::Real parent_value(const pops::Index<Dim>& index, int component) {
  pops::Real result = static_cast<pops::Real>(component * 10'000);
  pops::Real scale = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result += scale * static_cast<pops::Real>(index[axis]);
    scale *= pops::Real{97};
  }
  return result;
}

template <int Dim>
pops::Real fine_affine_value(const pops::Index<Dim>& index, int component) {
  pops::Real result = static_cast<pops::Real>(component * 10'000);
  pops::Real scale = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    const int parent = index[axis] / 2;
    const int child = index[axis] % 2;
    result += scale * (static_cast<pops::Real>(parent) +
                       (child == 0 ? pops::Real{-0.25} : pops::Real{0.25}));
    scale *= pops::Real{97};
  }
  return result;
}

template <int Dim>
void fill_parent(pops::MultiFab<Dim>& field) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (int component = 0; component < field.ncomp(); ++component)
      for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
           ++ordinal) {
        const auto index = index_from_ordinal(fab.box(), ordinal);
        host(field_offset(fab.grown_box(), index, component)) = parent_value(index, component);
      }
    fab.copy_from_host(host);
  }
}

template <int Dim>
void seed_child_valid(pops::MultiFab<Dim>& field, pops::Real sentinel) {
  field.set_val(sentinel);
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (int component = 0; component < field.ncomp(); ++component)
      for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
           ++ordinal) {
        const auto index = index_from_ordinal(fab.box(), ordinal);
        host(field_offset(fab.grown_box(), index, component)) = fine_affine_value(index, component);
      }
    fab.copy_from_host(host);
  }
}

template <int Dim>
std::pair<long, long> count_child_failures(const pops::MultiFab<Dim>& field, pops::Real sentinel) {
  long failures = 0;
  long populated_ghosts = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (int component = 0; component < field.ncomp(); ++component)
      for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.grown_box().numPts());
           ++ordinal) {
        const auto index = index_from_ordinal(fab.grown_box(), ordinal);
        const pops::Real value = host(field_offset(fab.grown_box(), index, component));
        if (value != fine_affine_value(index, component))
          ++failures;
        if (!fab.box().contains(index) && value != sentinel)
          ++populated_ghosts;
      }
  }
  return {failures, populated_ghosts};
}

template <int Dim>
pops::runtime::amr::AmrGhostFillBudget ghost_budget(std::size_t coarse_boxes,
                                                    std::size_t fine_boxes) {
  const std::size_t pairs =
      coarse_boxes * fine_boxes + coarse_boxes * coarse_boxes + fine_boxes * fine_boxes + 16;
  return {pops::runtime::amr::CoarseFineGhostScheduleBudget{fine_boxes, 32 * fine_boxes, pairs,
                                                            128 * fine_boxes, 32, 2'000'000,
                                                            2'000'000, 2'000'000},
          pops::HaloScheduleBudget{{fine_boxes, pairs},
                                   fine_boxes * fine_boxes * 64,
                                   fine_boxes * fine_boxes * 64 * static_cast<std::size_t>(2 * Dim),
                                   64,
                                   32,
                                   2'000'000,
                                   2'000'000,
                                   2'000'000}};
}

template <int Dim>
void prove_distributed_parent_injection(int rank_count, int rank) {
  pops::Index<Dim> coarse_upper{};
  for (int axis = 0; axis < Dim; ++axis)
    coarse_upper[axis] = (axis == 0 ? rank_count * 16 : 32) - 1;
  const pops::Box<Dim> coarse_domain{pops::Index<Dim>{}, coarse_upper};
  const pops::Box<Dim> fine_domain = pops::refine(coarse_domain, 2);

  auto coarse_grid = uniform_extent<Dim>(16);
  coarse_grid[0] = 8;
  const auto coarse_layout = pops::mesh::BoxArray<Dim>::from_domain(coarse_domain, coarse_grid);

  pops::Index<Dim> fine_lower{};
  pops::Index<Dim> fine_upper = fine_domain.hi;
  fine_lower[0] = rank_count * 8;
  fine_upper[0] = rank_count * 24 - 1;
  for (int axis = 1; axis < Dim; ++axis) {
    fine_lower[axis] = 16;
    fine_upper[axis] = 47;
  }
  const pops::Box<Dim> covered_fine{fine_lower, fine_upper};
  auto fine_grid = uniform_extent<Dim>(32);
  fine_grid[0] = 8;
  const auto fine_layout = pops::mesh::BoxArray<Dim>::from_domain(covered_fine, fine_grid);

  auto rank_extent = uniform_extent<Dim>(1);
  rank_extent[0] = rank_count;
  const pops::mesh::RankSpace<Dim> rank_space{pops::Index<Dim>{}, rank_extent};
  std::vector<pops::Index<Dim>> coarse_owners;
  std::vector<pops::Index<Dim>> fine_owners;
  for (std::size_t patch = 0; patch < coarse_layout.size(); ++patch)
    coarse_owners.push_back(rank_coordinate<Dim>(static_cast<int>(patch % rank_count)));
  for (std::size_t patch = 0; patch < fine_layout.size(); ++patch)
    fine_owners.push_back(rank_coordinate<Dim>(static_cast<int>((patch + 1) % rank_count)));
  const auto coarse_distribution = pops::mesh::Distribution<Dim>::partitioned(
      coarse_layout, rank_space, std::move(coarse_owners));
  const auto fine_distribution =
      pops::mesh::Distribution<Dim>::partitioned(fine_layout, rank_space, std::move(fine_owners));
  const auto local_rank = rank_coordinate<Dim>(rank);

  pops::MultiFab<Dim> parent(coarse_layout, coarse_distribution, local_rank, 3,
                             pops::Extent<Dim>{});
  pops::MultiFab<Dim> child(fine_layout, fine_distribution, local_rank, 3, uniform_extent<Dim>(1));
  fill_parent(parent);
  constexpr pops::Real sentinel = pops::Real{-12345};
  seed_child_valid(child, sentinel);

  std::array<int, Dim> ratio_values{};
  ratio_values.fill(2);
  pops::runtime::amr::AmrGhostFillPreparation<Dim> request{};
  request.fine_level = 1;
  request.coarse_domain = coarse_domain;
  request.fine_domain = fine_domain;
  request.ratio = pops::amr::RefinementRatio<Dim>{ratio_values};
  request.topology = pops::BoundaryTopology<Dim>::physical();
  request.topology_generation = 17;
  request.materialization_generation = 23;
  request.field_identity = "pops.test.mpi-coupler-inject.aux";
  request.budget = ghost_budget<Dim>(coarse_layout.size(), fine_layout.size());

  auto lane = pops::ExecutionLane::duplicate_world_collectively(
      "pops.test.mpi-coupler-inject.dim-" + std::to_string(Dim));
  const auto injection = pops::runtime::amr::prepare_amr_ghost_fill(parent, child, request, lane);
  if (rank_count > 1)
    EXPECT_TRUE(injection.has_remote_parent_jobs());
  pops::runtime::multiblock::BoundaryEvaluationPoint point{};
  point.level = 1;
  injection(child, point);

  const auto [local_failures, local_populated_ghosts] = count_child_failures(child, sentinel);
  EXPECT_EQ(pops::all_reduce_sum(local_failures, lane), 0L);
  EXPECT_GT(pops::all_reduce_sum(local_populated_ghosts, lane), 0L);
}

int run_mpi_coupler_inject(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
      prove_distributed_parent_injection<pops::kNativeDimension>(pops::n_ranks(), pops::my_rank());
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d exact-rank MPI parent injection failed: %s\n", pops::my_rank(),
                   error.what());
      result = 1;
    }
    result = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(result || ::testing::Test::HasFailure())));
    if (pops::my_rank() == 0 && result == 0)
      std::printf("OK test_mpi_coupler_inject (np=%d dim=%d prepared parent transport)\n",
                  pops::n_ranks(), pops::kNativeDimension);
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_coupler_inject, NativeDimensionUsesPreparedMultiblockParentTransport) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_coupler_inject, "test_mpi_coupler_inject"), 0);
}
