#pragma once

/// @file
/// @brief Small structured diagnostic reports owned by a runtime/solver instance.
///
/// This is intentionally not a global logger. Objects that need diagnostics own a report and decide
/// when to record events. The report is then inspectable by C++ tests or exposed through Python
/// bindings without scraping stdout/stderr.

#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

struct RuntimeDiagnosticEvent {
  std::string code;
  std::string component;
  std::string severity;
  std::string message;
  int iteration = -1;
  double value = 0.0;
};

struct RuntimeDiagnosticsReport {
  int schema_version = 1;
  std::string source = "pops.runtime.diagnostics";
  std::vector<RuntimeDiagnosticEvent> events;
  std::size_t dropped_events = 0;

  void clear() noexcept {
    events.clear();
    dropped_events = 0;
  }

  void record(std::string code, std::string component, std::string severity, std::string message,
              int iteration = -1, double value = 0.0) {
    events.push_back(RuntimeDiagnosticEvent{std::move(code), std::move(component),
                                            std::move(severity), std::move(message), iteration,
                                            value});
  }

  /// Numerical hot paths must never abandon an MPI trace because diagnostic storage allocation
  /// failed on one rank. This non-throwing variant owns all string construction inside its catch
  /// boundary and publishes a saturating dropped-event witness instead of altering the solve.
  [[nodiscard]] bool try_record(std::string_view code, std::string_view component,
                                std::string_view severity, std::string_view message,
                                int iteration = -1, double value = 0.0) noexcept {
    try {
      events.push_back(RuntimeDiagnosticEvent{std::string(code), std::string(component),
                                              std::string(severity), std::string(message),
                                              iteration, value});
      return true;
    } catch (...) {
      if (dropped_events != std::numeric_limits<std::size_t>::max())
        ++dropped_events;
      return false;
    }
  }

  std::size_t count(const std::string& code) const {
    std::size_t n = 0;
    for (const RuntimeDiagnosticEvent& event : events)
      if (event.code == code)
        ++n;
    return n;
  }
};

inline RuntimeDiagnosticsReport make_runtime_diagnostics_report(std::string source) {
  RuntimeDiagnosticsReport report;
  report.source = std::move(source);
  return report;
}

}  // namespace pops
