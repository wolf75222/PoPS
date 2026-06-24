// Locks the GeneratedModule metadata reader (include/adc/runtime/program/module_metadata.hpp,
// Spec 2 / ADC-442): the typed operator registry a problem.so exports for introspection and
// install-time validation. OperatorId is the registration index; the reader degrades gracefully on
// a pre-Spec-2 .so (no adc_module_* symbols) by returning present=false. Deliberately light: it does
// NOT build a .so (the end-to-end read of a real generated .so is validated on the Kokkos/AOT path,
// ROMEO); it pins the struct semantics + the absence handling the install path relies on.
#include <adc/runtime/program/module_metadata.hpp>

#include <dlfcn.h>

#include <cstdio>
#include <string>

using namespace adc::runtime::program;

namespace {

int failures = 0;

void check(bool cond, const char* what) {
  if (cond) {
    std::printf("  [OK ] %s\n", what);
  } else {
    std::printf("  [FAIL] %s\n", what);
    ++failures;
  }
}

}  // namespace

int main() {
  // (1) Default-constructed descriptor: not present, empty, find() == nullptr.
  ModuleMetadata empty;
  check(!empty.present, "default ModuleMetadata is not present");
  check(empty.operators.empty(), "default ModuleMetadata has no operators");
  check(empty.find("anything") == nullptr, "find on an empty descriptor returns nullptr");

  // (2) A hand-built descriptor: OperatorId is the index; find() resolves by name and carries kind.
  ModuleMetadata m;
  m.present = true;
  m.operators.push_back({0U, "fields_from_state", "field_operator", "(U) -> Fields", "{}"});
  m.operators.push_back({1U, "explicit_rhs", "local_rate", "(U, Fields) -> Rate(U)",
                         "{\"kind\":\"local_rate\"}"});
  m.state_spaces.push_back("U");
  m.field_spaces.push_back("fields");
  check(m.find("explicit_rhs") != nullptr, "find resolves a known operator");
  check(m.find("explicit_rhs")->id == 1U, "OperatorId is the registration index");
  check(m.find("explicit_rhs")->kind == "local_rate", "operator kind is carried");
  check(m.find("nope") == nullptr, "find on an unknown operator returns nullptr");

  // (3) Reading a null handle, or a handle that exports no adc_module_* symbols (the running program
  // itself), yields a not-present descriptor -- the backward-compatible / graceful path the install
  // routine takes for a pre-Spec-2 .so.
  check(!read_module_metadata(nullptr).present, "read_module_metadata(nullptr) is not present");
  void* self = dlopen(nullptr, RTLD_NOW | RTLD_LOCAL);
  check(!read_module_metadata(self).present,
        "reading a handle without adc_module_* symbols is not present");
  if (self != nullptr) {
    dlclose(self);
  }

  std::printf(failures == 0 ? "OK  test_module_metadata\n" : "FAILED test_module_metadata\n");
  return failures == 0 ? 0 : 1;
}
