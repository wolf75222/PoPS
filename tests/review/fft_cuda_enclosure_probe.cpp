// Bounded Source FFT engine probe; this is not an installed Native qualification.
#include <pops/numerics/elliptic/poisson/poisson_fft.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef POPS_FFT_PROBE_REQUIRE_CUDA
static_assert(std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>,
              "CUDA probe must use the actual CUDA default execution space");
#endif

// Instantiate every method, including the distributed branches not taken at world1.
template class pops::PoissonFFT<1>;
template class pops::PoissonFFT<2>;
template class pops::PoissonFFT<3>;

namespace {
void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <int Dim>
double check_shape(const std::array<int, Dim>& cells, std::ofstream& output) {
  using Engine = pops::PoissonFFT<Dim>;
  std::array<double, Dim> lengths{};
  for (int axis = 0; axis < Dim; ++axis)
    lengths[axis] = 1. + .25 * axis;
  auto lane = pops::ExecutionLane::world("tests.fft-enclosure");
  require(lane.size() == 1, "bounded FFT probe requires world1");
  Engine fft(cells, lengths, lane, "tests.fft-enclosure.product-mode");
  typename Engine::device_view rhs("probe RHS", fft.local_cell_count());
  typename Engine::device_view phi("probe phi", fft.local_cell_count());
  auto host_rhs = Kokkos::create_mirror_view(rhs);
  double eigenvalue = 0.;
  for (int axis = 0; axis < Dim; ++axis) {
    const int mode = cells[axis] > 3 ? 1 + axis % 2 : 1;
    const double spacing = lengths[axis] / cells[axis];
    eigenvalue +=
        (2 * std::cos(2 * std::numbers::pi * mode / cells[axis]) - 2) / (spacing * spacing);
  }
  for (std::size_t ordinal = 0; ordinal < fft.local_cell_count(); ++ordinal) {
    std::size_t cursor = ordinal;
    double real = 1., imaginary = .3;
    for (int axis = 0; axis < Dim; ++axis) {
      const int coordinate = static_cast<int>(cursor % cells[axis]);
      cursor /= cells[axis];
      const int mode = cells[axis] > 3 ? 1 + axis % 2 : 1;
      const double angle = 2 * std::numbers::pi * mode * (coordinate + .5) / cells[axis];
      real *= std::sin(angle);
      imaginary *= std::cos(angle);
    }
    host_rhs[ordinal] = typename Engine::complex_type(real, imaginary);
  }
  Kokkos::deep_copy(rhs, host_rhs);
  std::vector<double> first;
  double maximum = 0.;
  for (int attempt = 0; attempt < 2; ++attempt) {
    fft.solve(rhs, phi);
    Kokkos::fence();
    auto host_phi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, phi);
    std::vector<double> values;
    for (std::size_t ordinal = 0; ordinal < fft.local_cell_count(); ++ordinal) {
      require(std::isfinite(host_phi[ordinal].real()) && std::isfinite(host_phi[ordinal].imag()),
              "nonfinite FFT result cannot pass the independent oracle");
      maximum = std::max(
          maximum, std::abs(host_phi[ordinal].real() - host_rhs[ordinal].real() / eigenvalue));
      maximum = std::max(
          maximum, std::abs(host_phi[ordinal].imag() - host_rhs[ordinal].imag() / eigenvalue));
      values.push_back(host_phi[ordinal].real());
      values.push_back(host_phi[ordinal].imag());
    }
    if (attempt == 0)
      first = values;
    else
      require(first == values, "repeat solve did not reset the same FFT workspace");
  }
  require(maximum < 1e-11, "independent discrete Fourier mode oracle failed");
  output.write(reinterpret_cast<const char*>(cells.data()), sizeof(int) * Dim);
  output.write(reinterpret_cast<const char*>(first.data()), sizeof(double) * first.size());
  Kokkos::deep_copy(rhs, typename Engine::complex_type(7., -.5));
  fft.solve(rhs, phi);
  auto zero = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, phi);
  for (std::size_t ordinal = 0; ordinal < fft.local_cell_count(); ++ordinal)
    require(std::abs(zero[ordinal].real()) < 1e-11 && std::abs(zero[ordinal].imag()) < 1e-11,
            "zero Fourier symbol handling changed");
  typename Engine::device_view short_rhs("invalid probe RHS", fft.local_cell_count() - 1);
  bool refused = false;
  try {
    fft.solve(short_rhs, phi);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "view extent mismatch was accepted");
  return maximum;
}
}  // namespace

int main(int argc, char** argv) {
  require(argc == 2, "full output byte stream filename is required");
  std::ofstream output(argv[1], std::ios::binary);
  output.exceptions(std::ios::badbit | std::ios::failbit);
  Kokkos::initialize(argc, argv);
  double maximum = 0.;
  {
    maximum = std::max(maximum, check_shape<1>({8}, output));
    maximum = std::max(maximum, check_shape<1>({5}, output));
    maximum = std::max(maximum, check_shape<2>({8, 8}, output));
    maximum = std::max(maximum, check_shape<2>({3, 5}, output));
    maximum = std::max(maximum, check_shape<3>({8, 8, 8}, output));
    maximum = std::max(maximum, check_shape<3>({3, 5, 7}, output));
    auto lane = pops::ExecutionLane::world("tests.fft-enclosure.allocation-refusal");
    bool refused = false;
    try {
      pops::PoissonFFT<2> fail({8, 8}, {1., 1.}, lane, "tests.fft-enclosure.allocation",
                               {pops::PoissonFFTDiagnosticStage::workspace_allocation, 0});
    } catch (const std::bad_alloc&) {
      refused = true;
    }
    require(refused, "workspace allocation injection was accepted");
  }
  std::cout << "{\"default_execution\":\"" << Kokkos::DefaultExecutionSpace::name()
            << "\",\"default_memory\":\"" << Kokkos::DefaultExecutionSpace::memory_space::name()
            << "\",\"fftw\":" << (pops::poisson_fft_fftw_configured() ? "true" : "false")
            << ",\"shapes\":6,\"max_independent_mode_error\":" << maximum << "}\n";
  Kokkos::finalize();
}
