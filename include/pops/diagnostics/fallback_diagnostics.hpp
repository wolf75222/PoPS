#pragma once

/// @file
/// @brief Process-local fallback/degraded-route diagnostics.
///
/// The counters here are intentionally lightweight and global: they cover fallback routes that are
/// otherwise hidden inside low-level helpers. Runtime inspection layers can combine these event
/// counters with per-system configuration (for example positivity floors) to explain both what has
/// happened and which degraded routes are enabled by policy.

#include <atomic>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace pops {

enum class FallbackCounter {
  kFftDirectDft,
  kDenseEigGershgorin,
  kForeachSerialSmallBox,
  kNativeLoaderLegacyMetadata,
  kRolelessComponentIndex,
};

inline std::atomic<std::size_t> g_fallback_fft_direct_dft{0};
inline std::atomic<std::size_t> g_fallback_dense_eig_gershgorin{0};
inline std::atomic<std::size_t> g_fallback_foreach_serial_small_box{0};
inline std::atomic<std::size_t> g_fallback_native_loader_legacy_metadata{0};
inline std::atomic<std::size_t> g_fallback_roleless_component_index{0};

inline std::atomic<std::size_t>& fallback_counter(FallbackCounter counter) {
  switch (counter) {
    case FallbackCounter::kFftDirectDft:
      return g_fallback_fft_direct_dft;
    case FallbackCounter::kDenseEigGershgorin:
      return g_fallback_dense_eig_gershgorin;
    case FallbackCounter::kForeachSerialSmallBox:
      return g_fallback_foreach_serial_small_box;
    case FallbackCounter::kNativeLoaderLegacyMetadata:
      return g_fallback_native_loader_legacy_metadata;
    case FallbackCounter::kRolelessComponentIndex:
      return g_fallback_roleless_component_index;
  }
  return g_fallback_fft_direct_dft;
}

inline void record_fallback(FallbackCounter counter, std::size_t n = 1) {
  fallback_counter(counter).fetch_add(n, std::memory_order_relaxed);
}

inline std::size_t fallback_count(FallbackCounter counter) {
  return fallback_counter(counter).load(std::memory_order_relaxed);
}

inline void reset_fallback_diagnostics_counters() {
  fallback_counter(FallbackCounter::kFftDirectDft).store(0, std::memory_order_relaxed);
  fallback_counter(FallbackCounter::kDenseEigGershgorin).store(0, std::memory_order_relaxed);
  fallback_counter(FallbackCounter::kForeachSerialSmallBox).store(0, std::memory_order_relaxed);
  fallback_counter(FallbackCounter::kNativeLoaderLegacyMetadata).store(0, std::memory_order_relaxed);
  fallback_counter(FallbackCounter::kRolelessComponentIndex).store(0, std::memory_order_relaxed);
}

struct FallbackDiagnosticEntry {
  std::string key;
  std::string route;
  std::string cause;
  std::string policy;
  std::string default_action;
  std::string impact;
  std::string frequency;
  std::size_t count = 0;
  bool explicit_opt_in = false;
  bool performance_degraded = false;
  bool semantics_changed = false;
};

struct FallbackDiagnosticsReport {
  int schema_version = 1;
  std::string source = "pops.diagnostics.fallback_diagnostics";
  std::vector<FallbackDiagnosticEntry> entries;
};

inline FallbackDiagnosticEntry fallback_entry(std::string key, std::string route,
                                              std::string cause, std::string policy,
                                              std::string default_action, std::string impact,
                                              std::string frequency, std::size_t count,
                                              bool explicit_opt_in,
                                              bool performance_degraded,
                                              bool semantics_changed) {
  FallbackDiagnosticEntry entry;
  entry.key = std::move(key);
  entry.route = std::move(route);
  entry.cause = std::move(cause);
  entry.policy = std::move(policy);
  entry.default_action = std::move(default_action);
  entry.impact = std::move(impact);
  entry.frequency = std::move(frequency);
  entry.count = count;
  entry.explicit_opt_in = explicit_opt_in;
  entry.performance_degraded = performance_degraded;
  entry.semantics_changed = semantics_changed;
  return entry;
}

inline FallbackDiagnosticsReport fallback_diagnostics_report() {
  FallbackDiagnosticsReport report;
  report.entries.push_back(fallback_entry(
      "elliptic.fft.direct_dft", "PoissonFFT::fft1d",
      "FFT extent is not a power of two", "allowed_with_counter", "allow",
      "correct O(n^2) transform replaces the radix-2 FFT", "per 1D transform",
      fallback_count(FallbackCounter::kFftDirectDft), false, true, false));
  report.entries.push_back(fallback_entry(
      "linalg.dense_eig.gershgorin", "real_eig_minmax",
      "QR iteration cap reached before convergence", "allowed_with_counter",
      "return_bound_not_spectrum",
      "returns an enclosing Gershgorin bound; callers must not treat it as eigenvalues",
      "per dense block solve", fallback_count(FallbackCounter::kDenseEigGershgorin), false,
      false, true));
  report.entries.push_back(fallback_entry(
      "mesh.for_each.serial_small_box", "for_each_cell",
      "host execution space and cell count below POPS_FOREACH_SERIAL_THRESHOLD",
      "allowed_with_counter", "allow",
      "bit-identical sequential host loop avoids Kokkos launch overhead", "per cell loop launch",
      fallback_count(FallbackCounter::kForeachSerialSmallBox), false, false, false));
  report.entries.push_back(fallback_entry(
      "runtime.native_loader.legacy_metadata", "native_loader",
      "compiled artifact lacks names, roles or gamma metadata", "report_and_compat",
      "allow_legacy_abi",
      "uses u0.. names, empty roles or legacy default gamma unless explicit metadata is present",
      "per loaded legacy block", fallback_count(FallbackCounter::kNativeLoaderLegacyMetadata),
      false, false, true));
  report.entries.push_back(fallback_entry(
      "runtime.roleless_component_index", "System coupled-source component resolver",
      "legacy block declares no roles", "allow_legacy_roleless_only",
      "allow_only_when_roles_empty",
      "canonical component indices are used; role-bearing blocks missing a role are rejected",
      "per required role lookup", fallback_count(FallbackCounter::kRolelessComponentIndex),
      false, false, true));
  report.entries.push_back(fallback_entry(
      "spatial.positivity.order1_face", "Zhang-Shu positivity limiter",
      "reconstructed face density falls below positivity_floor", "explicit_opt_in",
      "disabled_until_positivity_floor_positive",
      "offending face is replaced by the source-cell average, locally reducing order",
      "per offending reconstructed face", 0, true, false, true));
  report.entries.push_back(fallback_entry(
      "elliptic.polar.jacobi_preconditioner", "PolarTensorOperator",
      "radial-line preconditioner layout constraints are not requested or not suitable",
      "explicit_opt_in", "radial_line_by_default",
      "Jacobi is a weaker preconditioner but has no radial-column layout constraint",
      "per polar tensor solve configured with Jacobi", 0, true, true, false));
  report.entries.push_back(fallback_entry(
      "runtime.limiter_unknown_muscl_ghost", "limiter_n_ghost",
      "unknown limiter reaches the halo-width helper", "refuse_final_route",
      "validate_then_throw",
      "temporary MUSCL-width allocation compatibility only; final dispatch rejects the limiter",
      "per invalid limiter validation attempt", 0, false, false, false));
  return report;
}

}  // namespace pops
