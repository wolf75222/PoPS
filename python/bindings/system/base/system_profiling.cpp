// ADC-632: profiling seam of the System facade -- enable/disable/is/reset_profiling, profile_report
// and solver_diagnostics. This small TU is a subdivision of system.cpp isolating the per-node/per-
// brick timing surface (ADC-459 Profiler) from the hot install/fields paths.
// Pure body move from system.cpp, no logic changed -> production trajectories bit-identical.
#include "system_impl.hpp"  // ADC-632: shared System::Impl + facade helpers (binding-private)

namespace pops {

// --- profiling (ADC-459) -------------------------------------------------------------------------
// enable_profiling / profile_report drive the System-owned Profiler. Today the System wraps its
// coarse phases (step, field_solve); the per-Program-node / per-native-brick granularity is wired
// through the compiled-program ProgramContext as a follow-up.
void System::enable_profiling() {
  p_->program_.profiler_.enable();
}
void System::disable_profiling() {
  p_->program_.profiler_.disable();
}
bool System::is_profiling() const {
  return p_->program_.profiler_.enabled();
}
void System::reset_profiling() {
  p_->program_.profiler_.reset();
}
std::string System::profile_report() const {
  return p_->program_.profiler_.report();
}
std::vector<RuntimeDiagnosticEvent> System::solver_diagnostics() const {
  return p_->fields_.combined_diagnostics_report().events;
}
// The System-owned Profiler reference (ADC-459): the compiled-program ProgramContext::profile_node
// times each Program node into it, so per-node scopes accumulate in the SAME table as the coarse
// step / field_solve phases. POPS_EXPORT: resolved by a generated problem.so across the dlopen boundary.
POPS_EXPORT pops::runtime::program::Profiler& System::profiler() {
  return p_->program_.profiler_;
}


}  // namespace pops
