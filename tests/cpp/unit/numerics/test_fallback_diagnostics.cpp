#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/diagnostics/fallback_diagnostics.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft.hpp>
#include <pops/numerics/linalg/dense_eig.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace {
void chk(bool cond, const std::string& label) {
  std::cout << "  [" << (cond ? "OK " : "XX ") << "] " << label << "\n";
  if (!cond)
    std::exit(1);
}

const pops::FallbackDiagnosticEntry* find_entry(const pops::FallbackDiagnosticsReport& report,
                                                const std::string& key) {
  const auto it = std::find_if(report.entries.begin(), report.entries.end(),
                               [&](const pops::FallbackDiagnosticEntry& row) {
                                 return row.key == key;
                               });
  return it == report.entries.end() ? nullptr : &*it;
}
}  // namespace

static int pops_run_test_fallback_diagnostics() {
  using namespace pops;

  reset_fallback_diagnostics_counters();
  FallbackDiagnosticsReport report = fallback_diagnostics_report();
  chk(find_entry(report, "elliptic.fft.direct_dft") != nullptr, "FFT fallback policy is reported");
  chk(find_entry(report, "linalg.dense_eig.gershgorin") != nullptr,
      "Gershgorin fallback policy is reported");
  chk(find_entry(report, "spatial.positivity.order1_face") != nullptr,
      "positivity fallback policy is reported");

  {
    PoissonFFT solver(6, 6, 1.0, 1.0);
    std::vector<double> rho(36, 0.0), phi;
    rho[0] = 1.0;
    solver.solve(rho, phi);
    chk(poisson_fft_direct_dft_fallback_count() > 0, "direct DFT fallback increments counter");
  }

  {
    Real A[3][3] = {{Real(0), Real(0), Real(6)},
                    {Real(1), Real(0), Real(-11)},
                    {Real(0), Real(1), Real(6)}};
    bool fallback = false;
    const EigBounds bounds = real_eig_minmax(A, /*max_iter_per_eig=*/0, &fallback);
    chk(!bounds.converged && fallback, "dense eig reports forced Gershgorin fallback");
    chk(fallback_count(FallbackCounter::kDenseEigGershgorin) > 0,
        "Gershgorin fallback increments counter");
  }

  if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>) {
    const std::size_t before = fallback_count(FallbackCounter::kForeachSerialSmallBox);
    if (detail::foreach_serial_threshold() > 1) {
      for_each_cell(Box2D::from_extents(1, 1), [] POPS_HD(int, int) {});
      chk(fallback_count(FallbackCounter::kForeachSerialSmallBox) > before,
          "small host for_each serial fallback increments counter");
    }
  }

  report = fallback_diagnostics_report();
  const FallbackDiagnosticEntry* fft = find_entry(report, "elliptic.fft.direct_dft");
  const FallbackDiagnosticEntry* eig = find_entry(report, "linalg.dense_eig.gershgorin");
  chk(fft != nullptr && fft->count > 0 && fft->policy == "allowed_with_counter",
      "FFT fallback report carries count and policy");
  chk(eig != nullptr && eig->count > 0 && eig->semantics_changed,
      "Gershgorin report carries count and semantic impact");

  reset_fallback_diagnostics_counters();
  chk(fallback_count(FallbackCounter::kFftDirectDft) == 0, "reset clears FFT fallback count");
  chk(fallback_count(FallbackCounter::kDenseEigGershgorin) == 0,
      "reset clears Gershgorin fallback count");
  return 0;
}

TEST(test_fallback_diagnostics, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_fallback_diagnostics, "test_fallback_diagnostics"), 0);
}
