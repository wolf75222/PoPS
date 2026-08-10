// Solveur de Poisson periodique spectral distribue (lance via mpirun -np N).
// On choisit un RHS de moyenne nulle rho(i,j), on resout lap_h phi = rho en
// reparti par bandes et on verifie localement la solution manufacturée: aucun
// rang ne reconstruit le champ global pendant cette qualification MPI.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/numerics/elliptic/poisson/poisson_fft.hpp>
#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace pops;

static int pops_run_test_mpi_poisson(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  constexpr double kPi = 3.14159265358979323846;

  const int Nx = 64, Ny = 64;  // puissances de 2, divisibles par np <= 64
  const double Lx = 1.0, Ly = 1.0;
  PoissonFFT<2> solver({Nx, Ny}, {Lx, Ly}, "pops.mpi.raw-fft");
  const int nyl = solver.local_last_extent(), y0 = solver.local_last_begin();

  // RHS de moyenne nulle : produit de cosinus (somme nulle sur les periodes).
  auto rho = [&](int i, int j) {
    return std::cos(2 * kPi * 2 * i / Nx) * std::cos(2 * kPi * 3 * j / Ny) +
           0.5 * std::cos(2 * kPi * 5 * i / Nx) * std::cos(2 * kPi * 1 * j / Ny);
  };

  std::vector<double> rho_local(static_cast<std::size_t>(nyl) * Nx), phi_local;
  for (int jl = 0; jl < nyl; ++jl)
    for (int i = 0; i < Nx; ++i)
      rho_local[jl * Nx + i] = rho(i, y0 + jl);

  PoissonFFT<2>::device_view rhs("mpi_fft_rhs", rho_local.size());
  PoissonFFT<2>::device_view phi("mpi_fft_phi", rho_local.size());
  auto host_rhs = Kokkos::create_mirror_view(rhs);
  for (std::size_t ordinal = 0; ordinal < rho_local.size(); ++ordinal)
    host_rhs[ordinal] = PoissonFFT<2>::complex_type(rho_local[ordinal], 0.0);
  Kokkos::deep_copy(rhs, host_rhs);
  solver.solve(rhs, phi);
  auto host_phi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, phi);
  phi_local.resize(rho_local.size());
  for (std::size_t ordinal = 0; ordinal < phi_local.size(); ++ordinal)
    phi_local[ordinal] = host_phi[ordinal].real();

  const double dx = Lx / Nx, dy = Ly / Ny;
  const double first_lambda = (2.0 * std::cos(2.0 * kPi * 2 / Nx) - 2.0) / (dx * dx) +
                              (2.0 * std::cos(2.0 * kPi * 3 / Ny) - 2.0) / (dy * dy);
  const double second_lambda = (2.0 * std::cos(2.0 * kPi * 5 / Nx) - 2.0) / (dx * dx) +
                               (2.0 * std::cos(2.0 * kPi * 1 / Ny) - 2.0) / (dy * dy);
  double local_error = 0.0;
  for (int jl = 0; jl < nyl; ++jl)
    for (int i = 0; i < Nx; ++i) {
      const int j = y0 + jl;
      const double expected =
          std::cos(2 * kPi * 2 * i / Nx) * std::cos(2 * kPi * 3 * j / Ny) / first_lambda +
          0.5 * std::cos(2 * kPi * 5 * i / Nx) * std::cos(2 * kPi * 1 * j / Ny) / second_lambda;
      local_error = std::max(
          local_error, std::fabs(phi_local[static_cast<std::size_t>(jl) * Nx + i] - expected));
    }
  const double max_error = all_reduce_max(local_error);
  const long fails = max_error > 1e-10 ? 1 : 0;
  if (me == 0)
    std::printf("np=%d max|phi-phi_exact|=%.3e\n", np, max_error);
  if (me == 0 && fails == 0)
    std::printf("OK test_mpi_poisson (np=%d)\n", np);
  comm_finalize();
  return fails == 0 ? 0 : 1;
}

TEST(test_mpi_poisson, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_poisson, "test_mpi_poisson"), 0);
}
