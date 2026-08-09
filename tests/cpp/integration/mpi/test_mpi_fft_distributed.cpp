#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>
#include <pops/parallel/comm.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr int kDim = 2;
constexpr int kCells = 64;
constexpr pops::Real kPi = pops::Real(3.141592653589793238462643383279502884L);

std::size_t storage_ordinal(const pops::Box<kDim>& storage, const pops::Index<kDim>& index) {
  return static_cast<std::size_t>(index[0] - storage.lo[0]) +
         static_cast<std::size_t>(index[1] - storage.lo[1]) *
             static_cast<std::size_t>(storage.length(0));
}

pops::EllipticBuildRequest<kDim> distributed_request() {
  const int ranks = pops::n_ranks();
  if (ranks < 1 || kCells % ranks != 0)
    throw std::invalid_argument("MPI FFT test extent must be divisible by communicator size");
  const pops::Box<kDim> domain{pops::Index<kDim>{0, 0}, pops::Index<kDim>{kCells - 1, kCells - 1}};
  const pops::Geometry<kDim> geometry = pops::Geometry<kDim>::from_bounds(
      domain, pops::RealVector<kDim>{0, 0}, pops::RealVector<kDim>{1, 1});
  const int local_y = kCells / ranks;
  std::vector<pops::Box<kDim>> slabs;
  slabs.reserve(static_cast<std::size_t>(ranks));
  for (int rank = 0; rank < ranks; ++rank)
    slabs.emplace_back(pops::Index<kDim>{0, rank * local_y},
                       pops::Index<kDim>{kCells - 1, (rank + 1) * local_y - 1});
  pops::mesh::BoxArray<kDim> layout(std::move(slabs));
  const pops::mesh::RankSpace<kDim> rank_space{pops::Index<kDim>{0, 0},
                                               pops::Extent<kDim>{ranks, 1}};
  std::vector<pops::Index<kDim>> owners;
  owners.reserve(static_cast<std::size_t>(ranks));
  for (int rank = 0; rank < ranks; ++rank)
    owners.push_back(rank_space.coordinate(static_cast<std::size_t>(rank)));
  const pops::mesh::Distribution<kDim> distribution =
      pops::mesh::Distribution<kDim>::partitioned(layout, rank_space, std::move(owners));
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

void fill_mean_zero_mode(pops::MultiFab<kDim>& field, const pops::Geometry<kDim>& geometry) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const pops::Box<kDim>& valid = fab.box();
    const pops::Box<kDim>& storage = fab.grown_box();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j) {
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        host(storage_ordinal(storage, pops::Index<kDim>{i, j})) =
            std::sin(pops::Real(2) * kPi * geometry.cell_coordinate(0, i)) *
            std::sin(pops::Real(2) * kPi * geometry.cell_coordinate(1, j));
      }
    }
    fab.copy_from_host(host);
  }

  const pops::Real mean = pops::reduce_sum(field) / static_cast<pops::Real>(kCells * kCells);
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const pops::Box<kDim>& valid = fab.box();
    const pops::Box<kDim>& storage = fab.grown_box();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        host(storage_ordinal(storage, pops::Index<kDim>{i, j})) -= mean;
    fab.copy_from_host(host);
  }
}

int run_mpi_fft_distributed(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  long local_failures = 0;
  {
    auto request = distributed_request();
    const pops::Real measure = request.geometry.spacing(0) * request.geometry.spacing(1);
    pops::PoissonFFTSolver<kDim> solver = pops::make_elliptic_solver<pops::PoissonFFTSolver<kDim>>(
        std::move(request), pops::PoissonFFTFactory<kDim>{});
    solver.install_nullspace(
        pops::constant_mean_zero_nullspace<kDim>("periodic-mpi-fft", "mpi-test", measure),
        pops::PreparedVectorDistribution<kDim>::distributed());
    fill_mean_zero_mode(solver.rhs(), solver.geom());
    const pops::SolveReport report = solver.solve();
    if (!report.solved() || report.residual_norm > pops::Real(1e-9))
      ++local_failures;
    if (pops::my_rank() == 0)
      std::printf("PoissonFFTSolver<2> np=%d residual(-lap(phi)-rhs)=%.3e status=%s\n",
                  pops::n_ranks(), static_cast<double>(report.residual_norm), report.status_name());
  }

  const long failures = pops::all_reduce_sum(local_failures);
  if (failures == 0 && pops::my_rank() == 0)
    std::printf("OK test_mpi_fft_distributed\n");
  pops::comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_fft_distributed, exact_rank_two_slabs_are_invariant_across_mpi_sizes) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_fft_distributed, "test_mpi_fft_distributed"), 0);
}
