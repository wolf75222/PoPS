// Test-only CUDA/Host FFT stage diagnostic. Every oracle stays 1e-11.
#include <pops/numerics/elliptic/poisson/poisson_fft.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_device_kernels.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <limits>
#include <string>
#include <vector>
#include <type_traits>
#ifdef POPS_FFT_PROBE_REQUIRE_CUDA
static_assert(std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>);
#endif
template class pops::PoissonFFT<1>;
template class pops::PoissonFFT<2>;
template class pops::PoissonFFT<3>;
namespace {
using Z = std::complex<double>;
struct Evidence {
  std::ofstream rows, raw;
  int failed = 0, stages = 0, shapes = 0;
  double maximum = 0.;
  explicit Evidence(const std::string& root)
      : rows(root + ".jsonl"), raw(root + ".bin", std::ios::binary) {
    rows.exceptions(std::ios::badbit | std::ios::failbit);
    raw.exceptions(std::ios::badbit | std::ios::failbit);
    rows << std::setprecision(17);
  }
  static void number(std::ostream& out, double value) {
    if (std::isfinite(value))
      out << value;
    else if (std::isnan(value))
      out << "\"NaN\"";
    else
      out << (value > 0 ? "\"+Infinity\"" : "\"-Infinity\"");
  }
  template <int Dim, class View>
  void compare(const char* label, const std::array<int, Dim>& cells, const View& actual,
               const std::vector<Z>& expected, int attempt = 0) {
    Kokkos::fence();
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, actual);
    bool finite = true;
    double maxerror = 0.;
    std::size_t at = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      double a[2] = {host[i].real(), host[i].imag()},
             r[2] = {expected[i].real(), expected[i].imag()};
      bool ok =
          std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(r[0]) && std::isfinite(r[1]);
      double e = ok ? std::max(std::abs(a[0] - r[0]), std::abs(a[1] - r[1]))
                    : std::numeric_limits<double>::infinity();
      finite = finite && ok;
      if (e > maxerror) {
        maxerror = e;
        at = i;
      }
      // IEEE actual and references always precede any failure decision.
      raw.write(reinterpret_cast<const char*>(a), sizeof(a));
      raw.write(reinterpret_cast<const char*>(r), sizeof(r));
      rows << "{\"stage\":\"" << label << "\",\"dim\":" << Dim << ",\"cells\":[";
      for (int d = 0; d < Dim; ++d) {
        if (d)
          rows << ',';
        rows << cells[d];
      }
      rows << "],\"attempt\":" << attempt << ",\"ordinal\":" << i << ",\"actual\":[";
      number(rows, a[0]);
      rows << ',';
      number(rows, a[1]);
      rows << "],\"reference\":[";
      number(rows, r[0]);
      rows << ',';
      number(rows, r[1]);
      rows << "],\"finite\":" << (ok ? "true" : "false") << ",\"error\":";
      number(rows, e);
      rows << "}\n";
    }
    bool accepted = finite && maxerror < 1e-11;
    rows << "{\"stage_summary\":\"" << label << "\",\"dim\":" << Dim << ",\"attempt\":" << attempt
         << ",\"max_ordinal\":" << at << ",\"finite\":" << (finite ? "true" : "false")
         << ",\"max_error\":";
    number(rows, maxerror);
    rows << ",\"oracle\":1e-11,\"accepted\":" << (accepted ? "true" : "false") << "}\n";
    rows.flush();
    raw.flush();
    ++stages;
    if (!accepted)
      ++failed;
    maximum = std::max(maximum, maxerror);
  }
};
template <int Dim>
std::vector<Z> host_axis(const std::vector<Z>& input, const std::array<int, Dim>& cells, int axis,
                         bool inverse) {
  std::size_t stride = 1;
  for (int d = 0; d < axis; ++d)
    stride *= cells[d];
  const int n = cells[axis];
  std::vector<Z> output(input.size());
  for (std::size_t ordinal = 0; ordinal < input.size(); ++ordinal) {
    const int k = static_cast<int>((ordinal / stride) % n);
    const std::size_t base = ordinal - static_cast<std::size_t>(k) * stride;
    Z sum{};
    for (int j = 0; j < n; ++j) {
      double angle = (inverse ? 1. : -1.) * 2. * std::numbers::pi * k * j / n;
      sum +=
          input[base + stride * static_cast<std::size_t>(j)] * Z(std::cos(angle), std::sin(angle));
    }
    output[ordinal] = inverse ? sum / static_cast<double>(n) : sum;
  }
  return output;
}
template <int Dim>
struct Diagnostic {
  using Engine = pops::PoissonFFT<Dim>;
  using View = typename Engine::device_view;
  using Kernels =
      pops::detail::PoissonFFTDeviceKernels<Dim,
                                            typename Kokkos::DefaultExecutionSpace::memory_space>;
  static void snapshot_arrays(const std::array<int, Dim>& cells,
                              const std::array<double, Dim>& spacing, const View& actual) {
    Kokkos::parallel_for(
        "fft_diagnostic_array_snapshots", Kokkos::RangePolicy<>(0, Dim),
        KOKKOS_LAMBDA(int d) { actual[d] = typename Engine::complex_type(cells[d], spacing[d]); });
  }
  static void axis(View& values, View& scratch, const std::array<int, Dim>& cells, int d,
                   bool inverse) {
    std::size_t stride = 1;
    for (int i = 0; i < d; ++i)
      stride *= cells[i];
    const int n = cells[d];
    const auto count = values.extent(0);
    double sign = inverse ? 1. : -1.;
    if (n > 0 && (n & (n - 1)) == 0) {
      Kernels::bit_reverse(values, scratch, stride, n, count);
      std::swap(values, scratch);
      for (int length = 2; length <= n; length *= 2) {
        Kernels::local_radix_stage(values, scratch, stride, n, length, length / 2, sign, count);
        std::swap(values, scratch);
      }
      if (inverse)
        Kernels::local_radix_normalize(values, n, count);
    } else {
      Kernels::local_dft(values, scratch, stride, n, inverse, sign,
                         count / static_cast<std::size_t>(n));
      std::swap(values, scratch);
    }
  }
  static void shape(const std::array<int, Dim>& cells, Evidence& evidence) {
    ++evidence.shapes;
    std::array<double, Dim> lengths{}, spacing{};
    std::size_t count = 1, transverse = 1;
    double eigenvalue = 0.;
    for (int d = 0; d < Dim; ++d) {
      lengths[d] = 1. + .25 * d;
      spacing[d] = lengths[d] / cells[d];
      count *= cells[d];
      if (d < Dim - 1)
        transverse *= cells[d];
      int mode = cells[d] > 3 ? 1 + d % 2 : 1;
      eigenvalue +=
          (2 * std::cos(2 * std::numbers::pi * mode / cells[d]) - 2) / (spacing[d] * spacing[d]);
    }
    View rhs("diagnostic RHS", count), phi("diagnostic phi", count),
        values("diagnostic values", count), scratch("diagnostic scratch", count),
        arrays("diagnostic arrays", Dim);
    auto host = Kokkos::create_mirror_view(rhs);
    std::vector<Z> input(count), expected(count), array_expected(Dim);
    for (std::size_t i = 0; i < count; ++i) {
      std::size_t cursor = i;
      double real = 1., imag = .3;
      for (int d = 0; d < Dim; ++d) {
        int c = cursor % cells[d];
        cursor /= cells[d];
        int mode = cells[d] > 3 ? 1 + d % 2 : 1;
        double angle = 2 * std::numbers::pi * mode * (c + .5) / cells[d];
        real *= std::sin(angle);
        imag *= std::cos(angle);
      }
      input[i] = Z(real, imag);
      expected[i] = input[i] / eigenvalue;
      host[i] = typename Engine::complex_type(real, imag);
    }
    Kokkos::deep_copy(rhs, host);
    evidence.compare<Dim>("RHS_copy", cells, rhs, input);
    auto lane = pops::ExecutionLane::world("fft.stage-diagnostic");
    if (lane.size() != 1)
      throw std::runtime_error("diagnostic requires world1");
    Engine engine(cells, lengths, lane, "fft.stage-diagnostic.product-mode");
    for (int attempt = 0; attempt < 2; ++attempt) {
      engine.solve(rhs, phi);
      evidence.compare<Dim>("engine_phi", cells, phi, expected, attempt);
    }
    snapshot_arrays(cells, spacing, arrays);
    for (int d = 0; d < Dim; ++d)
      array_expected[d] = Z(cells[d], spacing[d]);
    evidence.compare<Dim>("device_array_snapshot", cells, arrays, array_expected);
    Kokkos::deep_copy(values, rhs);
    auto reference = input;
    for (int d = 0; d < Dim; ++d) {
      axis(values, scratch, cells, d, false);
      reference = host_axis<Dim>(reference, cells, d, false);
      evidence.compare<Dim>("forward_axis", cells, values, reference, d);
    }
    Kernels::inverse_symbol(values, cells, spacing, transverse, cells[Dim - 1], 0, false, count);
    for (std::size_t i = 0; i < count; ++i) {
      std::size_t cursor = i;
      double lambda = 0.;
      for (int d = 0; d < Dim; ++d) {
        int k = cursor % cells[d];
        cursor /= cells[d];
        lambda +=
            (2 * std::cos(2 * std::numbers::pi * k / cells[d]) - 2) / (spacing[d] * spacing[d]);
      }
      reference[i] = std::abs(lambda) < 1e-14 ? Z{} : reference[i] / lambda;
    }
    evidence.compare<Dim>("inverse_symbol", cells, values, reference);
    for (int d = Dim - 1; d >= 0; --d) {
      axis(values, scratch, cells, d, true);
      reference = host_axis<Dim>(reference, cells, d, true);
      evidence.compare<Dim>("inverse_axis", cells, values, reference, d);
    }
    evidence.compare<Dim>("direct_pipeline_phi", cells, values, expected);
    // The original zero-symbol criterion is retained and its actuals are saved.
    Kokkos::deep_copy(rhs, typename Engine::complex_type(7., -.5));
    engine.solve(rhs, phi);
    evidence.compare<Dim>("engine_zero_symbol", cells, phi, std::vector<Z>(count));
  }
};
}  // namespace
int main(int argc, char** argv) {
  if (argc != 2)
    throw std::runtime_error("diagnostic evidence filename prefix required");
  Evidence evidence(argv[1]);
  Kokkos::initialize(argc, argv);
  try {
    Diagnostic<1>::shape({8}, evidence);
    Diagnostic<1>::shape({5}, evidence);
    Diagnostic<2>::shape({8, 8}, evidence);
    Diagnostic<2>::shape({3, 5}, evidence);
    Diagnostic<3>::shape({8, 8, 8}, evidence);
    Diagnostic<3>::shape({3, 5, 7}, evidence);
  } catch (const std::exception& e) {
    std::cerr << "diagnostic exception: " << e.what() << "\n";
    ++evidence.failed;
  }
  std::cout << std::setprecision(17) << "{\"default_execution\":\""
            << Kokkos::DefaultExecutionSpace::name() << "\",\"default_memory\":\""
            << Kokkos::DefaultExecutionSpace::memory_space::name()
            << "\",\"fftw\":" << (pops::poisson_fft_fftw_configured() ? "true" : "false")
            << ",\"shapes\":" << evidence.shapes << ",\"stages\":" << evidence.stages
            << ",\"failed_stages\":" << evidence.failed << ",\"max_independent_mode_error\":";
  Evidence::number(std::cout, evidence.maximum);
  std::cout << "}\n";
  Kokkos::finalize();
  return evidence.failed ? 1 : 0;
}
