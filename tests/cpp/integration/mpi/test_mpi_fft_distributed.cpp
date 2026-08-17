#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
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
bool verify_device_product_of_sines(const pops::ExecutionLane& lane) {
  constexpr int kExtent = 8;
  if (kExtent % lane.size() != 0)
    return false;
  std::array<int, Dim> cells{};
  std::array<double, Dim> lengths{};
  cells.fill(kExtent);
  lengths.fill(1.0);
  pops::PoissonFFT<Dim> fft(cells, lengths, lane, "pops.mpi.nd-device-fft");
  const auto& extents = fft.cells();
  if (fft.local_last_extent() * lane.size() != extents[Dim - 1] ||
      fft.local_last_begin() != lane.rank() * fft.local_last_extent())
    return false;
  std::size_t transverse = 1;
  for (int axis = 0; axis < Dim - 1; ++axis)
    transverse *= static_cast<std::size_t>(extents[axis]);
  if (fft.local_cell_count() != transverse * static_cast<std::size_t>(fft.local_last_extent()))
    return false;
  typename pops::PoissonFFT<Dim>::device_view rhs("mpi_fft_nd_rhs", fft.local_cell_count());
  typename pops::PoissonFFT<Dim>::device_view phi("mpi_fft_nd_phi", fft.local_cell_count());
  auto host_rhs = Kokkos::create_mirror_view(rhs);
  for (std::size_t ordinal = 0; ordinal < fft.local_cell_count(); ++ordinal) {
    pops::Real value = pops::Real(1);
    std::size_t cursor = ordinal;
    for (int axis = 0; axis < Dim - 1; ++axis) {
      const int coordinate = static_cast<int>(cursor % static_cast<std::size_t>(extents[axis]));
      cursor /= static_cast<std::size_t>(extents[axis]);
      value *= std::sin(pops::Real(2) * kPi * (coordinate + pops::Real(0.5)) / extents[axis]);
    }
    const int last = fft.local_last_begin() + static_cast<int>(cursor);
    value *= std::sin(pops::Real(2) * kPi * (last + pops::Real(0.5)) / extents[Dim - 1]);
    host_rhs[ordinal] = typename pops::PoissonFFT<Dim>::complex_type(value, 0.0);
  }
  Kokkos::deep_copy(rhs, host_rhs);
  fft.solve(rhs, phi);
  auto host_phi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, phi);
  pops::Real lambda = pops::Real(0);
  for (int axis = 0; axis < Dim; ++axis) {
    const pops::Real h = static_cast<pops::Real>(lengths[axis]) / extents[axis];
    lambda += (pops::Real(2) * std::cos(pops::Real(2) * kPi / extents[axis]) - pops::Real(2)) /
              (h * h);
  }
  pops::Real local_error = 0;
  for (std::size_t ordinal = 0; ordinal < fft.local_cell_count(); ++ordinal)
    local_error = std::max(local_error, std::abs(host_phi[ordinal].real() - host_rhs[ordinal].real() / lambda));
  return pops::all_reduce_max(static_cast<double>(local_error), lane) < 1e-11;
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
    pops::Real value = pops::Real(1);
    for (int axis = 0; axis < Dim; ++axis)
      value *= Kokkos::sin(pops::Real(2) * kPi * geometry.cell_coordinate(axis, cell[axis]));
    rhs(cell, 0) = value;
  });
  const pops::SolveReport report = solver.solve();
  return report.solved() && report.residual_norm < pops::Real(1e-9);
}

bool verify_undivided_final_axis_is_refused_collectively(const pops::ExecutionLane& lane) {
  const int ranks = lane.size();
  if (ranks < 2)
    return true;
  bool rejected = false;
  try {
    pops::PoissonFFT<3> fft({8, 8, ranks + 1}, {1.0, 1.0, 1.0}, lane,
                            "pops.mpi.undivided-final-axis");
    (void)fft;
  } catch (const std::exception&) {
    rejected = true;
  }
  return pops::all_reduce_min(rejected ? 1L : 0L, lane) == 1L;
}

bool verify_replicated_distribution_is_refused_collectively(const pops::ExecutionLane& lane) {
  if (lane.size() < 2)
    return true;
  bool rejected = false;
  try {
    auto request = distributed_request<3>();
    request.distribution =
        pops::mesh::Distribution<3>::replicated(request.boxes, request.distribution.rank_space());
    pops::PoissonFFTSolver<3> solver = pops::make_elliptic_solver<pops::PoissonFFTSolver<3>>(
        std::move(request), pops::PoissonFFTFactory<3>{lane}, lane);
    (void)solver;
  } catch (const std::exception&) {
    rejected = true;
  }
  return pops::all_reduce_min(rejected ? 1L : 0L, lane) == 1L;
}

bool verify_full_final_axis_transverse_partition_is_refused(const pops::ExecutionLane& lane) {
  const int ranks = lane.size();
  if (ranks < 2 || kCells % ranks != 0)
    return true;
  bool rejected = false;
  try {
    pops::Index<3> lo{};
    pops::Index<3> hi{kCells - 1, kCells - 1, kCells - 1};
    pops::RealVector<3> lower{};
    pops::RealVector<3> upper{1, 1, 1};
    const pops::Box<3> domain{lo, hi};
    const pops::Geometry<3> geometry = pops::Geometry<3>::from_bounds(domain, lower, upper);
    const int local_x = kCells / ranks;
    std::vector<pops::Box<3>> boxes;
    boxes.reserve(static_cast<std::size_t>(ranks));
    for (int rank = 0; rank < ranks; ++rank) {
      auto box_lo = lo;
      auto box_hi = hi;
      box_lo[0] = rank * local_x;
      box_hi[0] = (rank + 1) * local_x - 1;
      boxes.emplace_back(box_lo, box_hi);
    }
    pops::mesh::BoxArray<3> layout(std::move(boxes));
    const pops::mesh::RankSpace<3> rank_space{pops::Index<3>{}, pops::Extent<3>{ranks, 1, 1}};
    std::vector<pops::Index<3>> owners;
    owners.reserve(static_cast<std::size_t>(ranks));
    for (int rank = 0; rank < ranks; ++rank)
      owners.push_back(rank_space.coordinate(static_cast<std::size_t>(rank)));
    const pops::mesh::Distribution<3> distribution =
        pops::mesh::Distribution<3>::partitioned(layout, rank_space, std::move(owners));
    std::array<pops::PhysicalBoundaryFace, 6> faces{};
    pops::RealVector<3> spacing{};
    for (int axis = 0; axis < 3; ++axis)
      spacing[axis] = geometry.spacing(axis);
    const std::size_t pairs = layout.size() * (layout.size() - 1) / 2;
    pops::EllipticBuildRequest<3> request{
        geometry,
        std::move(layout),
        distribution,
        rank_space.coordinate(static_cast<std::size_t>(pops::my_rank())),
        pops::PhysicalBoundaryConditions<3>{
            pops::BoundaryTopology<3>::axis_periodic({true, true, true}), faces, spacing},
        pops::Extent<3>{},
        pops::Extent<3>{1, 1, 1},
        {static_cast<std::size_t>(ranks), pairs}};
    pops::PoissonFFTSolver<3> solver = pops::make_elliptic_solver<pops::PoissonFFTSolver<3>>(
        std::move(request), pops::PoissonFFTFactory<3>{lane}, lane);
    (void)solver;
  } catch (const std::exception&) {
    rejected = true;
  }
  return pops::all_reduce_min(rejected ? 1L : 0L, lane) == 1L;
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
  int result = 1;
  {
    const pops::ExecutionLane lane =
        pops::ExecutionLane::duplicate_world_collectively("pops.test.mpi-fft-distributed@1");
    pops::reset_poisson_fft_direct_dft_fallback_count();
    long failures = 0;
    failures += verify_device_product_of_sines<1>(lane) ? 0 : 1;
    failures += verify_device_product_of_sines<2>(lane) ? 0 : 1;
    failures += verify_device_product_of_sines<3>(lane) ? 0 : 1;
    failures += verify_solver_dimension<1>(lane) ? 0 : 1;
    failures += verify_solver_dimension<2>(lane) ? 0 : 1;
    failures += verify_solver_dimension<3>(lane) ? 0 : 1;
    failures += verify_undivided_final_axis_is_refused_collectively(lane) ? 0 : 1;
    failures += verify_replicated_distribution_is_refused_collectively(lane) ? 0 : 1;
    failures += verify_full_final_axis_transverse_partition_is_refused(lane) ? 0 : 1;
    failures += pops::poisson_fft_direct_dft_fallback_count() == 0 ? 0 : 1;
    failures += verify_last_axis_fallback_contract(lane) ? 0 : 1;
    failures += verify_divergent_valid_request_is_rejected_collectively(lane) ? 0 : 1;
    failures += verify_rank_local_invalid_request_is_rejected_collectively(lane) ? 0 : 1;
    failures += verify_failure_boundaries_and_lane_unwind(lane) ? 0 : 1;
    failures = pops::all_reduce_sum(failures, lane);
    if (failures == 0 && pops::my_rank() == 0)
      std::printf("OK test_mpi_fft_distributed 1D/2D/3D np=%d\n", pops::n_ranks());
    result = failures == 0 ? 0 : 1;
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_fft_distributed, cartesian_solver_is_exact_in_every_native_dimension) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_fft_distributed, "test_mpi_fft_distributed"), 0);
}
