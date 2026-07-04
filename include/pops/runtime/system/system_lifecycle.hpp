#pragma once

#include <stdexcept>
#include <string>

/// @file
/// @brief The runtime freeze LIFECYCLE state machine of a System (ADC-578 / ADC-592).
///
/// Replaces the single `bool bound_` that was the whole lifecycle. It is the authoritative NATIVE
/// source of the runtime lifecycle the Python freeze gates query (derive_lifecycle_state prefers
/// `_s.lifecycle_state()`); the Python `_lifecycle` flag stays only as the documented fallback for a
/// prebuilt .so with no native symbols.
///
/// STATES: Assembling -> Bound -> (Running is DERIVED, see below) -> Checkpointed / Finalized.
///  - Assembling: the composition is mutable (structural setters allowed).
///  - Bound: pops.bind completed; structural setters are refused.
///  - Running: NOT a stored phase -- it is DERIVED at query time from the macro-step counter
///    (bound AND macro_step > 0). The stepper must never set it, so lifecycle stays stepper-invisible
///    (MockImpl never had bound_) and the observable "running" edge is unchanged.
///  - Checkpointed: an explicit, informational mark (a checkpointed sim resumes: it refuses nothing).
///  - Finalized: terminal; a superset of Bound for refusals (structural setters refused) plus
///    double-finalize / re-bind refused.
///
/// OBSERVABLE CONTRACT (bit-identity): for every call sequence that existed before this type,
/// lifecycle_state() returns the SAME three strings ("assembling" / "bound" / "running") and
/// require_assembling() throws the SAME message. Checkpointed / Finalized are reachable ONLY through
/// the NEW transitions (to_checkpointed / to_finalized), which have no caller in the current
/// codebase, so no existing test observes them.

namespace pops {
namespace runtime {
namespace system {

/// The stored lifecycle phase. `Running` is intentionally absent -- it is derived from the macro-step.
enum class LifecyclePhase { Assembling, Bound, Checkpointed, Finalized };

/// Typed runtime freeze lifecycle. Data-only + guarded transitions; stepper-invisible.
struct SystemLifecycle {
  LifecyclePhase phase = LifecyclePhase::Assembling;

  /// True once the composition is frozen (pops.bind completed). A structural setter refuses when this
  /// holds -- the same predicate the old `bound_` bool expressed (Bound / Checkpointed / Finalized).
  bool frozen() const { return phase != LifecyclePhase::Assembling; }

  /// The ONE transition into the frozen state (System::mark_bound). Assembling -> Bound; a second call
  /// throws (a composition binds exactly once), with the SAME message the old bool guard raised.
  void to_bound() {
    if (phase != LifecyclePhase::Assembling)
      throw std::runtime_error(
          "System::mark_bound: the composition is already bound (pops.bind binds a compiled Case "
          "exactly once; a fresh run needs a fresh pops.bind)");
    phase = LifecyclePhase::Bound;
  }

  /// NEW (ADC-578): mark a bound/running simulation as Checkpointed (informational). Refuses nothing;
  /// a checkpointed sim resumes. Rejected before bind and after finalize. No current caller.
  void to_checkpointed() {
    if (phase == LifecyclePhase::Assembling)
      throw std::runtime_error(
          "System::to_checkpointed: cannot checkpoint an unbound composition (bind it first)");
    if (phase == LifecyclePhase::Finalized)
      throw std::runtime_error(
          "System::to_checkpointed: the simulation is finalized (terminal); no further transition");
    phase = LifecyclePhase::Checkpointed;
  }

  /// NEW (ADC-578): finalize a bound simulation (terminal). Rejected before bind and refuses a second
  /// finalize. After this, structural setters stay refused (Finalized is a superset of Bound). No
  /// current caller.
  void to_finalized() {
    if (phase == LifecyclePhase::Assembling)
      throw std::runtime_error(
          "System::to_finalized: cannot finalize an unbound composition (bind it first)");
    if (phase == LifecyclePhase::Finalized)
      throw std::runtime_error(
          "System::to_finalized: the simulation is already finalized (terminal)");
    phase = LifecyclePhase::Finalized;
  }

  /// The observable lifecycle string. @p macro_step is the System macro-step counter: "running" is
  /// derived from it so the stepper never touches lifecycle. Preserves the historical three strings
  /// for the pre-existing states; surfaces the new states only when explicitly transitioned.
  std::string state(int macro_step) const {
    switch (phase) {
      case LifecyclePhase::Assembling:
        return "assembling";
      case LifecyclePhase::Finalized:
        return "finalized";
      case LifecyclePhase::Checkpointed:
        return "checkpointed";
      case LifecyclePhase::Bound:
      default:
        return macro_step > 0 ? "running" : "bound";
    }
  }
};

}  // namespace system
}  // namespace runtime
}  // namespace pops
