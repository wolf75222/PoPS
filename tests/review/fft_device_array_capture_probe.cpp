// Independent parameter-capture controls, not an Engine internal trace.
#include <Kokkos_Array.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <type_traits>

#ifdef POPS_FFT_PROBE_REQUIRE_CUDA
static_assert(std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>);
#endif

static void number(double value) {
  if (std::isfinite(value))
    std::cout << value;
  else if (std::isnan(value))
    std::cout << "\"NaN\"";
  else
    std::cout << (value > 0 ? "\"+Infinity\"" : "\"-Infinity\"");
}

template <int Dim>
struct CaptureControls {
  using View = Kokkos::View<Kokkos::complex<double>*>;
  static_assert(std::is_trivially_copyable_v<Kokkos::Array<int, Dim>>);
  static_assert(std::is_trivially_copyable_v<Kokkos::Array<double, Dim>>);

  static void standard(const std::array<int, Dim>& cells, const std::array<double, Dim>& spacing,
                       const View& out) {
    Kokkos::parallel_for(
        "std_array_capture_negative_control", Kokkos::RangePolicy<>(0, Dim),
        KOKKOS_LAMBDA(int d) { out[d] = Kokkos::complex<double>(cells[d], spacing[d]); });
  }
  static void device_array(const std::array<int, Dim>& cells,
                           const std::array<double, Dim>& spacing, const View& out) {
    Kokkos::Array<int, Dim> device_cells{};
    Kokkos::Array<double, Dim> device_spacing{};
    for (int d = 0; d < Dim; ++d) {
      device_cells[d] = cells[d];
      device_spacing[d] = spacing[d];
    }
    Kokkos::parallel_for(
        "kokkos_array_capture_positive_control", Kokkos::RangePolicy<>(0, Dim),
        KOKKOS_LAMBDA(int d) {
          out[d] = Kokkos::complex<double>(device_cells[d], device_spacing[d]);
        });
  }
  static void scalar(const std::array<int, Dim>& cells, const std::array<double, Dim>& spacing,
                     const View& out) {
    for (int d = 0; d < Dim; ++d) {
      const int n = cells[d];
      const double dx = spacing[d];
      Kokkos::parallel_for(
          "scalar_capture_positive_control", Kokkos::RangePolicy<>(0, 1),
          KOKKOS_LAMBDA(int) { out[d] = Kokkos::complex<double>(n, dx); });
    }
  }
  static int run() {
    // Nonuniform inputs distinguish axes; these are parameter-copy controls only.
    std::array<int, Dim> cells{};
    std::array<double, Dim> spacing{};
    for (int d = 0; d < Dim; ++d) {
      cells[d] = 3 + 2 * d;
      spacing[d] = (1. + .25 * d) / cells[d];
    }
    View standard_values("standard", Dim), device_values("device", Dim),
        scalar_values("scalar", Dim);
    standard(cells, spacing, standard_values);
    device_array(cells, spacing, device_values);
    scalar(cells, spacing, scalar_values);
    Kokkos::fence();
    auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, standard_values);
    auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_values);
    auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, scalar_values);
    int failed = 0;
    for (int d = 0; d < Dim; ++d) {
      bool device_ok = b[d].real() == cells[d] && b[d].imag() == spacing[d];
      bool scalar_ok = c[d].real() == cells[d] && c[d].imag() == spacing[d];
      std::cout << std::setprecision(17) << "{\"dim\":" << Dim << ",\"axis\":" << d
                << ",\"expected\":[" << cells[d] << ',' << spacing[d] << "],\"standard\":[";
      number(a[d].real());
      std::cout << ',';
      number(a[d].imag());
      std::cout << "],\"device_array\":[";
      number(b[d].real());
      std::cout << ',';
      number(b[d].imag());
      std::cout << "],\"scalar\":[";
      number(c[d].real());
      std::cout << ',';
      number(c[d].imag());
      std::cout << "],\"device_accepted\":" << (device_ok ? "true" : "false")
                << ",\"scalar_accepted\":" << (scalar_ok ? "true" : "false") << "}\n";
      std::cout.flush();
      failed += !device_ok || !scalar_ok;
    }
    return failed;
  }
};

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int failed = CaptureControls<1>::run() + CaptureControls<2>::run() + CaptureControls<3>::run();
  Kokkos::finalize();
  return failed ? 1 : 0;
}
