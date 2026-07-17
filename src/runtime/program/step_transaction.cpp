#if !defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI) || !defined(POPS_EXPORT_BUILDING_MODULE)
#error "step_transaction.cpp requires the shared runtime exception ABI producer contract"
#endif

#include <pops/runtime/program/step_transaction.hpp>

namespace pops::runtime::program {

// Key function for one canonical cross-DSO vtable/typeinfo. Generated Program loaders throw this
// type while the already-loaded _pops module catches and translates it to the registered Python
// exception; a header-only destructor would permit hidden, DSO-local RTTI identities.
StepAttemptRejected::~StepAttemptRejected() noexcept = default;

}  // namespace pops::runtime::program
