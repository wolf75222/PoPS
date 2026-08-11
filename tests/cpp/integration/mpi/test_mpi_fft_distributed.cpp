#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>
#include <pops/parallel/comm.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kCells = 64;
constexpr pops::Real kPi = pops::Real(3.141592653589793238462643383279502884L);

template <int Dim>
pops::EllipticBuildRequest<Dim> distributed_request() {
  const int ranks = pops::n_ranks();
  if (ranks < 1 || kCells % ranks != 0)
    throw std::invalid_argument("MPI FFT extent must divide communicator size");
  pops::Index<Dim> lo{};
  pops::Index<Dim> hi{};
  pops::RealVector<Dim> lower{};
  pops::RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    hi[axis] = kCells - 1;
    upper[axis] = pops::Real(1);
  }
  const pops::Box<Dim> domain{lo, hi};
  const pops::Geometry<Dim> geometry = pops::Geometry<Dim>::from_bounds(domain, lower, upper);
  const int local_last = kCells / ranks;
  std::vector<pops::Box<Dim>> slabs;
  slabs.reserve(static_cast<std::size_t>(ranks));
  for (int rank = 0; rank < ranks; ++rank) {
    auto slab_lo = lo;
    auto slab_hi = hi;
    slab_lo[Dim - 1] = rank * local_last;
    slab_hi[Dim - 1] = (rank + 1) * local_last - 1;
    slabs.emplace_back(slab_lo, slab_hi);
  }
  pops::mesh::BoxArray<Dim> layout(std::move(slabs));
  pops::Extent<Dim> rank_extent{};
  for (int axis = 0; axis < Dim; ++axis)
    rank_extent[axis] = axis == Dim - 1 ? ranks : 1;
  const pops::mesh::RankSpace<Dim> rank_space{pops::Index<Dim>{}, rank_extent};
  std::vector<pops::Index<Dim>> owners;
  owners.reserve(static_cast<std::size_t>(ranks));
  for (int rank = 0; rank < ranks; ++rank)
    owners.push_back(rank_space.coordinate(static_cast<std::size_t>(rank)));
  const pops::mesh::Distribution<Dim> distribution =
      pops::mesh::Distribution<Dim>::partitioned(layout, rank_space, std::move(owners));
  std::array<pops::PhysicalBoundaryFace, 2 * Dim> faces{};
  std::array<bool, Dim> periodic{};
  pops::RealVector<Dim> spacing{};
  pops::Extent<Dim> rhs_ghosts{};
  pops::Extent<Dim> phi_ghosts{};
  periodic.fill(true);
  for (int axis = 0; axis < Dim; ++axis) {
    spacing[axis] = geometry.spacing(axis);
    phi_ghosts[axis] = 1;
  }
  const std::size_t pairs = layout.size() * (layout.size() - 1) / 2;
  return {geometry,
          std::move(layout),
          distribution,
          rank_space.coordinate(static_cast<std::size_t>(pops::my_rank())),
          pops::PhysicalBoundaryConditions<Dim>{
              pops::BoundaryTopology<Dim>::axis_periodic(periodic), faces, spacing},
          rhs_ghosts,
          phi_ghosts,
          {static_cast<std::size_t>(ranks), pairs}};
}

template <int Dim>
bool verify_solver_dimension(const pops::ExecutionLane& lane) {
  auto request = distributed_request<Dim>();
  pops::Real measure = pops::Real(1);
  for (int axis = 0; axis < Dim; ++axis)
    measure *= request.geometry.spacing(axis);
  pops::PoissonFFTSolver<Dim> solver = pops::make_elliptic_solver<pops::PoissonFFTSolver<Dim>>(
      std::move(request), pops::PoissonFFTFactory<Dim>{lane}, lane);
  solver.install_nullspace(
      pops::constant_mean_zero_nullspace<Dim>("periodic-mpi-fft", "mpi-nd-test", measure),
      pops::PreparedVectorDistribution<Dim>::distributed());
  auto rhs = solver.rhs().fab(0).view();
  const auto geometry = solver.geom();
  pops::for_each_cell(solver.rhs().box(0), [=](const pops::CellIndex<Dim>& cell) {
    rhs(cell, 0) = Kokkos::sin(pops::Real(2) * kPi * geometry.cell_coordinate(0, cell[0]));
  });
  const pops::SolveReport report = solver.solve();
  return report.solved() && report.residual_norm < pops::Real(1e-9);
}

bool verify_last_axis_fallback_contract(const pops::ExecutionLane& lane) {
  const int extent = 3 * pops::n_ranks();
  pops::PoissonFFT<1> fft({extent}, {1.0}, lane, "pops.mpi.last-axis-fallback");
  pops::PoissonFFT<1>::device_view rhs("mpi_last_fallback_rhs", fft.local_cell_count());
  pops::PoissonFFT<1>::device_view phi("mpi_last_fallback_phi", fft.local_cell_count());
  Kokkos::deep_copy(rhs, pops::PoissonFFT<1>::complex_type(0.0, 0.0));
  fft.solve(rhs, phi);
  return pops::poisson_fft_direct_dft_fallback_count() > 0;
}

bool verify_divergent_valid_request_is_rejected_collectively(const pops::ExecutionLane& lane) {
  if (pops::n_ranks() == 1)
    return true;
  const int extent = pops::n_ranks() * (pops::my_rank() == 0 ? 4 : 8);
  bool rejected = false;
  try {
    pops::PoissonFFT<1> fft({extent}, {1.0}, lane, "pops.mpi.divergent-valid-request");
    (void)fft;
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  return pops::all_reduce_min(rejected ? 1L : 0L, lane) == 1L;
}

bool verify_rank_local_invalid_request_is_rejected_collectively(const pops::ExecutionLane& lane) {
  const int extent = pops::my_rank() == 0 ? 0 : pops::n_ranks() * 4;
  bool rejected = false;
  try {
    pops::PoissonFFT<1> fft({extent}, {1.0}, lane, "pops.mpi.rank-local-invalid-request");
    (void)fft;
  } catch (const std::exception&) {
    rejected = true;
  }
  return pops::all_reduce_min(rejected ? 1L : 0L, lane) == 1L;
}

bool verify_rank_local_diagnostic_failure_is_collective(pops::PoissonFFTDiagnosticStage stage,
                                                        int extent, const char* identity,
                                                        const pops::ExecutionLane& lane) {
  bool rejected = false;
  {
    try {
      pops::PoissonFFT<1> fft({extent}, {1.0}, lane, identity,
                              pops::PoissonFFTDiagnosticContext{stage, 0});
      pops::PoissonFFT<1>::device_view rhs("mpi_fft_diagnostic_rhs", fft.local_cell_count());
      pops::PoissonFFT<1>::device_view phi("mpi_fft_diagnostic_phi", fft.local_cell_count());
      Kokkos::deep_copy(rhs, pops::PoissonFFT<1>::complex_type(0.0, 0.0));
      fft.solve(rhs, phi);
    } catch (const std::exception&) {
      rejected = true;
    }
  }
  if (pops::all_reduce_min(rejected ? 1L : 0L, lane) != 1L)
    return false;

  pops::PoissonFFT<1> recovery({extent}, {1.0}, lane, std::string(identity) + "/recovery");
  pops::PoissonFFT<1>::device_view rhs("mpi_fft_recovery_rhs", recovery.local_cell_count());
  pops::PoissonFFT<1>::device_view phi("mpi_fft_recovery_phi", recovery.local_cell_count());
  Kokkos::deep_copy(rhs, pops::PoissonFFT<1>::complex_type(0.0, 0.0));
  recovery.solve(rhs, phi);
  return true;
}

bool verify_failure_boundaries_and_lane_unwind(const pops::ExecutionLane& lane) {
  const int ranks = pops::n_ranks();
  bool valid = verify_rank_local_diagnostic_failure_is_collective(
      pops::PoissonFFTDiagnosticStage::workspace_allocation, 4 * ranks,
      "pops.mpi.fft-workspace-allocation-failure", lane);
  valid = verify_rank_local_diagnostic_failure_is_collective(
              pops::PoissonFFTDiagnosticStage::peer_dft_launch, 3 * ranks,
              "pops.mpi.fft-peer-launch-failure", lane) &&
          valid;
  valid = verify_rank_local_diagnostic_failure_is_collective(
              pops::PoissonFFTDiagnosticStage::peer_accumulation, 3 * ranks,
              "pops.mpi.fft-peer-accumulation-failure", lane) &&
          valid;
  if (ranks > 1 && pops::is_pow2(ranks))
    valid = verify_rank_local_diagnostic_failure_is_collective(
                pops::PoissonFFTDiagnosticStage::distributed_radix, 4 * ranks,
                "pops.mpi.fft-distributed-radix-failure", lane) &&
            valid;
  return valid;
}

int run_mpi_fft_distributed(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  const pops::ExecutionLane lane = pops::ExecutionLane::world("pops.test.mpi-fft-distributed");
  pops::reset_poisson_fft_direct_dft_fallback_count();
  long failures = 0;
  failures += verify_solver_dimension<1>(lane) ? 0 : 1;
  failures += verify_solver_dimension<2>(lane) ? 0 : 1;
  failures += verify_solver_dimension<3>(lane) ? 0 : 1;
  failures += pops::poisson_fft_direct_dft_fallback_count() == 0 ? 0 : 1;
  failures += verify_last_axis_fallback_contract(lane) ? 0 : 1;
  failures += verify_divergent_valid_request_is_rejected_collectively(lane) ? 0 : 1;
  failures += verify_rank_local_invalid_request_is_rejected_collectively(lane) ? 0 : 1;
  failures += verify_failure_boundaries_and_lane_unwind(lane) ? 0 : 1;
  failures = pops::all_reduce_sum(failures, lane);
  if (failures == 0 && pops::my_rank() == 0)
    std::printf("OK test_mpi_fft_distributed 1D/2D/3D np=%d\n", pops::n_ranks());
  pops::comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_fft_distributed, cartesian_solver_is_exact_in_every_native_dimension) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_fft_distributed, "test_mpi_fft_distributed"), 0);
}
