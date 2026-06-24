#ifndef ADC_RUNTIME_PROGRAM_MODULE_METADATA_HPP
#define ADC_RUNTIME_PROGRAM_MODULE_METADATA_HPP

// GeneratedModule metadata (Spec 2 / ADC-442). A combined model+program ``problem.so`` carries,
// alongside ``GeneratedProgram`` (the installed step), a ``GeneratedModule`` descriptor: the typed
// operator registry the Python codegen emits (adc.time.Program._emit_module_metadata) as a set of
// ``extern "C"`` accessors. This header reads that descriptor from an already-dlopen'd handle, for
// INTROSPECTION and install-time requirement validation. It is read ONCE at install; the step body
// never touches it, so operators stay inlined and there is NO string lookup in any hot kernel.
//
// Backward compatible: a .so generated before Spec 2 exports no ``adc_module_*`` symbols, so
// read_module_metadata returns ``present == false`` and the caller simply skips module introspection.
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <dlfcn.h>

namespace adc {
namespace runtime {
namespace program {

/// Integer id of an operator within a module: its registration index. The generated .so addresses
/// operators by this id; the name/kind/signature strings are metadata only (debug, introspection,
/// validation), never a hot-path lookup.
using OperatorId = std::uint32_t;

/// Integer id of a state or field space within a module.
using SpaceId = std::uint32_t;

/// One operator's metadata, as exported by the .so.
struct OperatorMetadata {
  OperatorId id = 0;
  std::string name;
  std::string kind;          ///< one of the Spec-2 operator kinds (local_rate, field_operator, ...)
  std::string signature;     ///< human-readable typed signature
  std::string requirements;  ///< JSON, e.g. {"kind":"local_source","aux":["grad_x","grad_y"]}
};

/// The GeneratedModule descriptor read from a problem.so. ``present`` is false when the .so exports
/// no module descriptor (a pre-Spec-2 .so) -- callers then skip module introspection / validation.
struct ModuleMetadata {
  bool present = false;
  std::vector<OperatorMetadata> operators;
  std::vector<std::string> state_spaces;
  std::vector<std::string> field_spaces;

  /// The operator with this name, or nullptr if none.
  const OperatorMetadata* find(const std::string& name) const {
    for (const auto& op : operators) {
      if (op.name == name) {
        return &op;
      }
    }
    return nullptr;
  }
};

namespace detail {

/// Call a ``const char* (int)`` accessor at index @p i; empty string if the symbol is absent.
inline std::string module_str(void* handle, const char* symbol, int i) {
  using Fn = const char* (*)(int);
  // dlsym yields a void*; the cast to a function pointer is the standard (and only) idiom.
  auto* fn = reinterpret_cast<Fn>(dlsym(handle, symbol));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
  if (fn == nullptr) {
    return std::string();
  }
  const char* s = fn(i);
  return s != nullptr ? std::string(s) : std::string();
}

/// Read a ``(count, name)`` string table (state/field spaces) from the handle.
inline std::vector<std::string> module_names(void* handle, const char* count_symbol,
                                             const char* name_symbol) {
  std::vector<std::string> out;
  using CountFn = int (*)();
  auto* count = reinterpret_cast<CountFn>(dlsym(handle, count_symbol));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
  if (count == nullptr) {
    return out;
  }
  const int n = count();
  for (int i = 0; i < n; ++i) {
    out.push_back(module_str(handle, name_symbol, i));
  }
  return out;
}

}  // namespace detail

/// Read the GeneratedModule metadata from an already-dlopen'd problem.so @p dl_handle. Returns a
/// descriptor with ``present == false`` (and empty vectors) when the handle is null or exports no
/// ``adc_module_operator_count`` symbol (a pre-Spec-2 .so).
inline ModuleMetadata read_module_metadata(void* dl_handle) {
  ModuleMetadata meta;
  if (dl_handle == nullptr) {
    return meta;
  }
  using CountFn = int (*)();
  auto* count = reinterpret_cast<CountFn>(  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
      dlsym(dl_handle, "adc_module_operator_count"));
  if (count == nullptr) {
    return meta;  // pre-Spec-2 .so: no GeneratedModule descriptor
  }
  meta.present = true;
  const int n = count();
  if (n > 0) {
    meta.operators.reserve(static_cast<std::size_t>(n));
  }
  for (int i = 0; i < n; ++i) {
    OperatorMetadata op;
    op.id = static_cast<OperatorId>(i);
    op.name = detail::module_str(dl_handle, "adc_module_operator_name", i);
    op.kind = detail::module_str(dl_handle, "adc_module_operator_kind", i);
    op.signature = detail::module_str(dl_handle, "adc_module_operator_signature", i);
    op.requirements = detail::module_str(dl_handle, "adc_module_operator_requirements", i);
    meta.operators.push_back(std::move(op));
  }
  meta.state_spaces =
      detail::module_names(dl_handle, "adc_module_state_space_count", "adc_module_state_space_name");
  meta.field_spaces =
      detail::module_names(dl_handle, "adc_module_field_space_count", "adc_module_field_space_name");
  return meta;
}

}  // namespace program
}  // namespace runtime
}  // namespace adc

#endif  // ADC_RUNTIME_PROGRAM_MODULE_METADATA_HPP
