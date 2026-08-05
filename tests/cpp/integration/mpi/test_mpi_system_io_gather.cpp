/// @file
/// @brief MPI proof for exact-ranked System field marshaling in 1D, 2D, and 3D.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"

#include <pops/parallel/comm.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#ifdef POPS_HAS_MPI
#include <mpi.h>
#endif

namespace {

using namespace pops;
using pops::mesh::BoxArray;
using pops::mesh::Distribution;
using pops::mesh::RankSpace;

template <int Dim>
Extent<Dim> uniform_extent(std::int64_t value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
struct ExactFixture {
  Box<Dim> domain;
  BoxArray<Dim> layout;
  RankSpace<Dim> ranks;
  Distribution<Dim> distribution;
  Index<Dim> local_rank;

  ExactFixture(int rank, int processes)
      : domain(make_domain_(processes)),
        layout(make_layout_(domain, processes)),
        ranks(make_rank_space_(processes)),
        distribution(Distribution<Dim>::partitioned(layout, ranks, make_owners_(ranks))),
        local_rank(ranks.coordinate(static_cast<std::size_t>(rank))) {}

 private:
  static Box<Dim> make_domain_(int processes) {
    Index<Dim> upper{};
    upper[0] = 3 * processes - 1;
    for (int axis = 1; axis < Dim; ++axis)
      upper[axis] = axis + 2;
    return Box<Dim>{Index<Dim>{}, upper};
  }

  static BoxArray<Dim> make_layout_(const Box<Dim>& domain, int processes) {
    std::vector<Box<Dim>> patches;
    patches.reserve(static_cast<std::size_t>(processes));
    for (int rank = 0; rank < processes; ++rank) {
      Box<Dim> patch = domain;
      patch.lo[0] = 3 * rank;
      patch.hi[0] = patch.lo[0] + 2;
      patches.push_back(patch);
    }
    return BoxArray<Dim>{std::move(patches)};
  }

  static RankSpace<Dim> make_rank_space_(int processes) {
    Extent<Dim> process_shape = uniform_extent<Dim>(1);
    process_shape[0] = processes;
    return RankSpace<Dim>{Index<Dim>{}, process_shape};
  }

  static std::vector<Index<Dim>> make_owners_(const RankSpace<Dim>& ranks) {
    std::vector<Index<Dim>> owners;
    owners.reserve(ranks.size());
    for (std::size_t rank = 0; rank < ranks.size(); ++rank)
      owners.push_back(ranks.coordinate(rank));
    return owners;
  }
};

template <int Dim, class Check>
void prove_exact_marshaling(int rank, int processes, Check&& check) {
  ExactFixture<Dim> fixture(rank, processes);
  MultiFab<Dim> field(fixture.layout, fixture.distribution, fixture.local_rank, 2,
                      uniform_extent<Dim>(1));
  field.set_val(Real{-17});

  const std::size_t cells = runtime::system::marshaling::checked_cell_count(fixture.domain);
  std::vector<double> payload(2 * cells);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    payload[cell] = static_cast<double>(cell) + 0.25;
    payload[cells + cell] = -static_cast<double>(cell) - 2.5;
  }

  runtime::system::marshaling::write_global(field, fixture.domain, payload, 2);
  check(runtime::system::marshaling::gather_global(field, fixture.domain, 2) == payload,
        "partitioned round-trip differs");

  if (processes > 1) {
    // A different checkpoint image on one rank must be rejected by every rank before resident data
    // changes. The following gather also proves that all ranks remained collectively aligned.
    std::vector<double> divergent = payload;
    if (rank == 0)
      divergent.front() += 1.0;
    bool rejected = false;
    try {
      runtime::system::marshaling::write_global(field, fixture.domain, divergent, 2);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check(rejected, "rank-divergent restore payload was accepted");
    check(runtime::system::marshaling::gather_global(field, fixture.domain, 2) == payload,
          "rejected restore mutated resident data");
  }

  // A replicated decomposition has one canonical collective contributor but every resident replica
  // is restored. This prevents rank-count multiplication without a special dimension route.
  const auto replicated = Distribution<Dim>::replicated(fixture.layout, fixture.ranks);
  MultiFab<Dim> replica(fixture.layout, replicated, fixture.local_rank, 2, uniform_extent<Dim>(1));
  replica.set_val(Real{-23});
  runtime::system::marshaling::write_global(replica, fixture.domain, payload, 2);
  check(runtime::system::marshaling::gather_global(replica, fixture.domain, 2) == payload,
        "replicated round-trip double-counted the payload");

  if (processes > 1) {
    bool component_request_rejected = false;
    try {
      (void)runtime::system::marshaling::gather_global(field, fixture.domain, rank == 0 ? 1 : 2);
    } catch (const std::invalid_argument&) {
      component_request_rejected = true;
    }
    check(component_request_rejected, "rank-divergent gather request was accepted");
  }
}

int run_mpi_system_io_gather(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int rank = my_rank();
  const int processes = n_ranks();
  long failures = 0;
  const auto check = [&](bool condition, const char* message) {
    if (!condition) {
      std::fprintf(stderr, "[rank %d/%d] Dim-generic System I/O failure: %s\n", rank, processes,
                   message);
      ++failures;
    }
  };

  prove_exact_marshaling<1>(rank, processes, check);
  prove_exact_marshaling<2>(rank, processes, check);
  prove_exact_marshaling<3>(rank, processes, check);

#ifdef POPS_HAS_MPI
  if (processes > 1) {
    long collective_failures = 0;
    MPI_Allreduce(&failures, &collective_failures, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    failures = collective_failures;
  }
#endif
  if (rank == 0 && failures == 0)
    std::printf("OK exact-ranked MPI System I/O (np=%d, Dim=1/2/3)\n", processes);
  comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_system_io_gather, exact_ranked_collective_round_trip_and_fail_close) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_system_io_gather, "test_mpi_system_io_gather"), 0);
}
