/// @file
/// @brief Compile-time-ranked profiling and structured-diagnostic System facade.

#include "system_impl.hpp"

#include <pops/core/foundation/native_dimension.hpp>

#include <string>
#include <vector>

namespace pops {

template <int Dim>
void System<Dim>::enable_profiling() {
  p_->program_.profiler_.enable();
}

template <int Dim>
void System<Dim>::disable_profiling() {
  p_->program_.profiler_.disable();
}

template <int Dim>
bool System<Dim>::is_profiling() const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->program_.profiler_.enabled();
}

template <int Dim>
void System<Dim>::reset_profiling() {
  p_->program_.profiler_.reset();
}

template <int Dim>
std::string System<Dim>::profile_report() const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->program_.profiler_.report();
}

template <int Dim>
std::vector<RuntimeDiagnosticEvent> System<Dim>::solver_diagnostics() const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  // Exact-ranked Cartesian field solves return their typed SolveReport through SolveOutcome. They
  // do not own a second persistent trace log, so this legacy inspection projection is empty until a
  // provider explicitly publishes structured events.
  return {};
}

template <int Dim>
POPS_EXPORT pops::runtime::program::Profiler System<Dim>::profiler() const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->program_.profiler_;
}

template <int Dim>
POPS_EXPORT pops::runtime::program::Profiler& System<Dim>::program_profiler_() {
  return p_->program_.profiler_;
}

template void System<kNativeDimension>::enable_profiling();
template void System<kNativeDimension>::disable_profiling();
template bool System<kNativeDimension>::is_profiling() const;
template void System<kNativeDimension>::reset_profiling();
template std::string System<kNativeDimension>::profile_report() const;
template std::vector<RuntimeDiagnosticEvent> System<kNativeDimension>::solver_diagnostics() const;
template runtime::program::Profiler System<kNativeDimension>::profiler() const;
template runtime::program::Profiler& System<kNativeDimension>::program_profiler_();

}  // namespace pops
