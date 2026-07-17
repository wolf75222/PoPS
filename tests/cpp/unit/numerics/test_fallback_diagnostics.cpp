#include <gtest/gtest.h>

#include <pops/diagnostics/fallback_diagnostics.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/index/box2d.hpp>
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
    PoissonFFT solver(6, 6, 1.0, 1.0);
    std::vector<double> rho(36, 0.0), phi;
    rho[0] = 1.0;
    solver.solve(rho, phi);
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
      for_each_cell(Box2D::from_extents(1, 1), [] POPS_HD(int, int) {});
      EXPECT_GT(fallback_count(FallbackCounter::kForeachSerialSmallBox), before)
          << "small host for_each serial fallback increments counter";
    }
  }

  report = fallback_diagnostics_report();
  const FallbackDiagnosticEntry* fft = find_entry(report, "elliptic.fft.direct_dft");
  const FallbackDiagnosticEntry* eig = find_entry(report, "linalg.dense_eig.gershgorin");
  ASSERT_NE(fft, nullptr);
  ASSERT_NE(eig, nullptr);
  EXPECT_TRUE(fft->count > 0 && fft->policy == "allowed_with_counter")
      << "FFT fallback report carries count and policy";
  EXPECT_TRUE(eig->count > 0 && eig->semantics_changed)
      << "Gershgorin report carries count and semantic impact";

  reset_fallback_diagnostics_counters();
  EXPECT_EQ(fallback_count(FallbackCounter::kFftDirectDft), 0u)
      << "reset clears FFT fallback count";
  EXPECT_EQ(fallback_count(FallbackCounter::kDenseEigGershgorin), 0u)
      << "reset clears Gershgorin fallback count";
}
