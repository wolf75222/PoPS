#include <gtest/gtest.h>

#include <pops/diagnostics/fallback_diagnostics.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft.hpp>
#include <pops/numerics/linalg/dense_eig.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>

namespace {
const pops::FallbackDiagnosticEntry* find_entry(const pops::FallbackDiagnosticsReport& report,
                                                const std::string& key) {
  const auto it =
      std::find_if(report.entries.begin(), report.entries.end(),
                   [&](const pops::FallbackDiagnosticEntry& row) { return row.key == key; });
  return it == report.entries.end() ? nullptr : &*it;
}
}  // namespace

// Pipeline stateful : les compteurs de fallback sont des globales partagees, incrementees section
// par section puis relues -- l'ordre des sections est LOAD-BEARING (reset au debut, report final
// apres tous les declenchements).
TEST(test_fallback_diagnostics, counters_and_report_track_triggered_fallbacks) {
  using namespace pops;

  reset_fallback_diagnostics_counters();
  FallbackDiagnosticsReport report = fallback_diagnostics_report();
  EXPECT_NE(find_entry(report, "elliptic.fft.direct_dft"), nullptr)
      << "FFT fallback policy is reported";
  EXPECT_NE(find_entry(report, "linalg.dense_eig.gershgorin"), nullptr)
      << "Gershgorin fallback policy is reported";
  EXPECT_NE(find_entry(report, "spatial.positivity.order1_face"), nullptr)
      << "positivity fallback policy is reported";

  {
    const ExecutionLane lane = ExecutionLane::world("tests.fallback-diagnostics.poisson-fft");
    PoissonFFT<2> solver({6, 6}, {1.0, 1.0}, lane, "tests.fallback-diagnostics.poisson-fft");
    PoissonFFT<2>::device_view rhs("fallback_fft_rhs", solver.local_cell_count());
    PoissonFFT<2>::device_view phi("fallback_fft_phi", solver.local_cell_count());
    Kokkos::deep_copy(rhs, PoissonFFT<2>::complex_type(0.0, 0.0));
    solver.solve(rhs, phi);
    if (poisson_fft_fftw_configured())
      EXPECT_EQ(poisson_fft_direct_dft_fallback_count(), 0u)
          << "FFTW covers non-power-of-two local extents without the O(n^2) fallback";
    else
      EXPECT_GT(poisson_fft_direct_dft_fallback_count(), 0u)
          << "direct DFT fallback increments counter";
  }

  {
    Real A[3][3] = {
        {Real(0), Real(0), Real(6)}, {Real(1), Real(0), Real(-11)}, {Real(0), Real(1), Real(6)}};
    bool fallback = false;
    const EigBounds bounds = real_eig_minmax(A, /*max_iter_per_eig=*/0, &fallback);
    EXPECT_TRUE(!bounds.converged && fallback) << "dense eig reports forced Gershgorin fallback";
    EXPECT_GT(fallback_count(FallbackCounter::kDenseEigGershgorin), 0u)
        << "Gershgorin fallback increments counter";
  }

  if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>) {
    const std::size_t before = fallback_count(FallbackCounter::kForeachSerialSmallBox);
    if (detail::foreach_serial_threshold() > 1) {
      Extent<kNativeDimension> extent{};
      for (int axis = 0; axis < kNativeDimension; ++axis)
        extent[axis] = 1;
      for_each_cell(Box<kNativeDimension>::from_extents(extent),
                    [] POPS_HD(const CellIndex<kNativeDimension>&) {});
      EXPECT_GT(fallback_count(FallbackCounter::kForeachSerialSmallBox), before)
          << "small host for_each serial fallback increments counter";
    }
  }

  report = fallback_diagnostics_report();
  const FallbackDiagnosticEntry* fft = find_entry(report, "elliptic.fft.direct_dft");
  const FallbackDiagnosticEntry* eig = find_entry(report, "linalg.dense_eig.gershgorin");
  ASSERT_NE(fft, nullptr);
  ASSERT_NE(eig, nullptr);
  EXPECT_TRUE(fft->policy == "allowed_with_counter")
      << "FFT fallback report carries policy";
  if (!poisson_fft_fftw_configured())
    EXPECT_GT(fft->count, 0u) << "FFT fallback report carries count";
  EXPECT_TRUE(eig->count > 0 && eig->semantics_changed)
      << "Gershgorin report carries count and semantic impact";

  reset_fallback_diagnostics_counters();
  EXPECT_EQ(fallback_count(FallbackCounter::kFftDirectDft), 0u)
      << "reset clears FFT fallback count";
  EXPECT_EQ(fallback_count(FallbackCounter::kDenseEigGershgorin), 0u)
      << "reset clears Gershgorin fallback count";
}
