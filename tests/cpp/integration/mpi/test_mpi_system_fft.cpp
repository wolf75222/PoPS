// Exact distributed FFT provider gate.  The historical executable name is retained so CI keeps
// exercising np={1,2,4}, but this test no longer constructs the retired System single-box remap.
// It proves that the only concrete provider is PoissonFFTSolver<2> on canonical ordered slabs and
// that a multi-rank single-box request is rejected instead of being silently remapped.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

constexpr int kDim = 2;
constexpr int kCells = 16;
constexpr pops::Real kPi = pops::Real(3.141592653589793238462643383279502884L);

using Request = pops::EllipticBuildRequest<kDim>;
using Layout = pops::mesh::BoxArray<kDim>;
using RankSpace = pops::mesh::RankSpace<kDim>;
using Distribution = pops::mesh::Distribution<kDim>;

static_assert(!pops::PoissonFFTCapabilities<1>::available);
static_assert(pops::PoissonFFTCapabilities<2>::available);
static_assert(!pops::PoissonFFTCapabilities<3>::available);

std::size_t storage_ordinal(const pops::Box<kDim>& storage, const pops::Index<kDim>& index) {
  return static_cast<std::size_t>(index[0] - storage.lo[0]) +
         static_cast<std::size_t>(index[1] - storage.lo[1]) *
             static_cast<std::size_t>(storage.length(0));
}

Request fft_request(bool canonical_slabs) {
  const int ranks = pops::n_ranks();
  if (ranks < 1 || kCells % ranks != 0)
    throw std::invalid_argument("FFT provider test extent must divide the communicator size");

  const pops::Box<kDim> domain{pops::Index<kDim>{0, 0},
                               pops::Index<kDim>{kCells - 1, kCells - 1}};
  const pops::Geometry<kDim> geometry = pops::Geometry<kDim>::from_bounds(
      domain, pops::RealVector<kDim>{0, 0}, pops::RealVector<kDim>{1, 1});
  const RankSpace rank_space{pops::Index<kDim>{0, 0}, pops::Extent<kDim>{ranks, 1}};

  std::vector<pops::Box<kDim>> boxes;
  std::vector<pops::Index<kDim>> owners;
  if (canonical_slabs) {
    const int local_y = kCells / ranks;
    boxes.reserve(static_cast<std::size_t>(ranks));
    owners.reserve(static_cast<std::size_t>(ranks));
    for (int rank = 0; rank < ranks; ++rank) {
      boxes.emplace_back(pops::Index<kDim>{0, rank * local_y},
                         pops::Index<kDim>{kCells - 1, (rank + 1) * local_y - 1});
      owners.push_back(rank_space.coordinate(static_cast<std::size_t>(rank)));
    }
  } else {
    boxes.push_back(domain);
    owners.push_back(rank_space.coordinate(0));
  }

  Layout layout(std::move(boxes));
  const Distribution distribution =
      Distribution::partitioned(layout, rank_space, std::move(owners));
  const std::array<pops::PhysicalBoundaryFace, 2 * kDim> faces{};
  const pops::RealVector<kDim> spacing{geometry.spacing(0), geometry.spacing(1)};
  const std::size_t pairs = layout.size() * (layout.size() - 1) / 2;
  return {geometry,
          std::move(layout),
          distribution,
          rank_space.coordinate(static_cast<std::size_t>(pops::my_rank())),
          pops::PhysicalBoundaryConditions<kDim>{
              pops::BoundaryTopology<kDim>::axis_periodic({true, true}), faces, spacing},
          pops::Extent<kDim>{0, 0},
          pops::Extent<kDim>{1, 1},
          {static_cast<std::size_t>(ranks), pairs}};
}

void fill_manufactured_rhs(pops::MultiFab<kDim>& rhs, const pops::Geometry<kDim>& geometry) {
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    auto& fab = rhs.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const pops::Box<kDim>& valid = fab.box();
    const pops::Box<kDim>& storage = fab.grown_box();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        host(storage_ordinal(storage, pops::Index<kDim>{i, j})) =
            std::sin(pops::Real(2) * kPi * geometry.cell_coordinate(0, i)) *
            std::sin(pops::Real(2) * kPi * geometry.cell_coordinate(1, j));
    fab.copy_from_host(host);
  }
}

pops::Real manufactured_error(const pops::PoissonFFTSolver<kDim>& solver) {
  const pops::Real dx = solver.geom().spacing(0);
  const pops::Real dy = solver.geom().spacing(1);
  const pops::Real theta = pops::Real(2) * kPi / pops::Real(kCells);
  const pops::Real eigenvalue = pops::Real(2) * (pops::Real(1) - std::cos(theta)) / (dx * dx) +
                                pops::Real(2) * (pops::Real(1) - std::cos(theta)) / (dy * dy);
  pops::Real local_error = 0;
  const auto& field = solver.phi();
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const pops::Box<kDim>& valid = fab.box();
    const pops::Box<kDim>& storage = fab.grown_box();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        const pops::Real rhs =
            std::sin(pops::Real(2) * kPi * solver.geom().cell_coordinate(0, i)) *
            std::sin(pops::Real(2) * kPi * solver.geom().cell_coordinate(1, j));
        const pops::Real value =
            host(storage_ordinal(storage, pops::Index<kDim>{i, j}));
        local_error = std::max(local_error, std::abs(value - rhs / eigenvalue));
      }
  }
  return pops::all_reduce_max(local_error);
}

int run_exact_mpi_fft_provider(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  long local_failures = 0;
  auto check = [&](bool condition, const char* label) {
    if (!condition) {
      std::printf("[rank %d/%d] FAIL %s\n", pops::my_rank(), pops::n_ranks(), label);
      ++local_failures;
    }
  };

  {
    Request request = fft_request(true);
    const auto expected = pops::PoissonFFTSolver<kDim>::expected_operator_contract(request);
    const pops::Real cell_measure = request.geometry.spacing(0) * request.geometry.spacing(1);
    pops::PoissonFFTSolver<kDim> solver =
        pops::make_elliptic_solver<pops::PoissonFFTSolver<kDim>>(
            std::move(request), pops::PoissonFFTFactory<kDim>{});
    solver.install_nullspace(
        pops::constant_mean_zero_nullspace<kDim>(
            "periodic-mpi-fft", "test-mpi-system-fft", cell_measure),
        pops::PreparedVectorDistribution<kDim>::distributed());
    check(solver.prepared_operator_contract().exact_fingerprint() ==
              expected.exact_fingerprint(),
          "exact_operator_contract");

    fill_manufactured_rhs(solver.rhs(), solver.geom());
    const pops::SolveReport first = solver.solve();
    const pops::Real first_error = manufactured_error(solver);
    const pops::SolveReport second = solver.solve();
    const pops::Real second_error = manufactured_error(solver);
    check(first.solved() && second.solved(), "exact_fft_solved_twice");
    check(first.residual_norm < pops::Real(1e-9) &&
              second.residual_norm < pops::Real(1e-9),
          "exact_fft_roundoff_residual");
    check(first_error < pops::Real(1e-12) && second_error < pops::Real(1e-12),
          "exact_fft_manufactured_solution");
    if (pops::my_rank() == 0)
      std::printf(
          "PoissonFFTSolver<2> np=%d residual=%.3e manufactured_error=%.3e\n",
          pops::n_ranks(), static_cast<double>(second.residual_norm),
          static_cast<double>(second_error));
  }

  if (pops::n_ranks() > 1) {
    bool rejected = false;
    try {
      Request remap_request = fft_request(false);
      pops::PoissonFFTSolver<kDim> forbidden =
          pops::make_elliptic_solver<pops::PoissonFFTSolver<kDim>>(
              std::move(remap_request), pops::PoissonFFTFactory<kDim>{});
      (void)forbidden;
    } catch (const std::exception&) {
      rejected = true;
    }
    check(rejected, "multi_rank_single_box_remap_is_rejected");
  }

  const long failures = pops::all_reduce_sum(local_failures);
  if (failures == 0 && pops::my_rank() == 0)
    std::printf("OK test_mpi_system_fft exact provider (np=%d)\n", pops::n_ranks());
  pops::comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_system_fft, exact_rank_two_provider_has_no_single_box_remap) {
  EXPECT_EQ(pops::test::RunTestBody(&run_exact_mpi_fft_provider, "test_mpi_system_fft"), 0);
}
