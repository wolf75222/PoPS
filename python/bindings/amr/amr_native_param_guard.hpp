// ADC-514 native runtime-param loader guard for the AmrSystem binding TU.
//
// Binding-private include (not a public header): amr_system.cpp pulls it in near the top so
// add_native_block can call the guard. Needs only kMaxRuntimeParams + the dynlib dlsym/dlclose
// layer (both already included above the include point). Split out to keep amr_system.cpp on its
// frozen architecture budget (tests/python/architecture/test_no_legacy_runtime_routes.py).
#ifndef POPS_PYTHON_BINDINGS_AMR_AMR_NATIVE_PARAM_GUARD_HPP_
#define POPS_PYTHON_BINDINGS_AMR_AMR_NATIVE_PARAM_GUARD_HPP_

namespace pops {
namespace detail {

// DEFENSE IN DEPTH (ADC-514): reject a hand-built .so whose pops_compiled_nparams exceeds the fixed-size
// device RuntimeParams (mirror of native_loader.hpp), BEFORE add_compiled_model seeds the value block.
// The symbol is OPTIONAL -> an older AMR loader omits it -> 0, bit-identical. Closes @p h on overflow.
inline int reject_excessive_amr_runtime_params(void* h) {
  auto nparams_fn = reinterpret_cast<int (*)()>(dlsym(h, "pops_compiled_nparams"));
  const int nparams = nparams_fn ? nparams_fn() : 0;
  if (nparams > kMaxRuntimeParams) {
    dlclose(h);
    throw std::runtime_error(
        "AmrSystem::add_native_block : the .so declares " + std::to_string(nparams) +
        " runtime parameters > kMaxRuntimeParams=" + std::to_string(kMaxRuntimeParams) +
        " (include/pops/runtime/config/runtime_params.hpp); regenerate the compiled module with the "
        "current headers (the codegen enforces the same bound).");
  }
  return nparams;
}

}  // namespace detail
}  // namespace pops

#endif  // POPS_PYTHON_BINDINGS_AMR_AMR_NATIVE_PARAM_GUARD_HPP_
