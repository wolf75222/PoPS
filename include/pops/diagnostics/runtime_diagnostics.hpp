#pragma once

/// @file
/// @brief Small structured diagnostic reports owned by a runtime/solver instance.
///
/// This is intentionally not a global logger. Objects that need diagnostics own a report and decide
/// when to record events. The report is then inspectable by C++ tests or exposed through Python
/// bindings without scraping stdout/stderr.

#include <cstddef>
#include <string>
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

  void clear() { events.clear(); }

  void record(std::string code, std::string component, std::string severity, std::string message,
              int iteration = -1, double value = 0.0) {
    events.push_back(RuntimeDiagnosticEvent{std::move(code), std::move(component),
                                            std::move(severity), std::move(message), iteration,
                                            value});
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
