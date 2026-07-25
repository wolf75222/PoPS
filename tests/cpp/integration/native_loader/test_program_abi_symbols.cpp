// ADC-538: ABI SYMBOL-PRESENCE fence for the compiled time-Program loader path (epic ADC-399 /
// ADC-401 Phase 2c). System::install_program dlopens a generated problem.so and resolves a contract of
// extern "C" symbols across the ABI boundary; test_program_loader.cpp proves the end-to-end numeric
// step, while this test isolates the SYMBOL PRESENCE half: it compiles a stub problem.so exporting the
// full Program + module-metadata symbol family and asserts every contract symbol RESOLVES via dlsym,
// from a runtime-compiled DSO. This closes the module-side ABI symbol-presence proof without relying
// on a fake process-handle probe.
//
// Contract symbols asserted present on the stub .so:
//  - pops_program_abi_key  (REQUIRED: install_program fails loud if missing);
//  - pops_install_program  (REQUIRED: the macro-step installer);
//  - pops_program_block_count + pops_program_block_name (REQUIRED: explicit block identities;
//    positional binding is forbidden);
//  - pops_program_route_manifest (required route-registry identity);
//  - the complete owner-qualified pops_module operator/state-space/field-space metadata family.
// pops_program_hash / pops_program_name remain diagnostic identity symbols.
// The stub is compiled with the exact compiler/Kokkos contract injected by CMake. A missing compiler
// or compilation failure is a hard failure because otherwise no ABI symbol was actually proven.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"
#include <pops/runtime/dynamic/dynlib.hpp>
#include <pops/runtime/program/module_metadata.hpp>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

// The generated problem.so surface: the extern "C" ABI a real codegen emits. Hand-written here for an
// autonomous symbol-presence test (no numeric body needed -- pops_install_program is a no-op stub, it
// only has to EXIST and be resolvable). The ABI key is the preprocessor LITERAL, like test_program_loader.
std::string stub_source() {
  // clang-format off
  return R"CPP(
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <cstdint>
extern "C" const char* pops_program_abi_key() { return POPS_ABI_KEY_LITERAL; }
extern "C" const char* pops_program_route_manifest() { return pops::kRouteRegistrySignature; }
extern "C" const char* pops_program_name() { return "abi_symbol_stub"; }
extern "C" const char* pops_program_hash() { return "deadbeef"; }
extern "C" int pops_program_operator_authority_count() { return 0; }
extern "C" std::uint64_t pops_program_operator_authority_word(int, int) { return 0; }
extern "C" int pops_program_block_count() { return 1; }
extern "C" const char* pops_program_block_name(int i) { return i == 0 ? "gas" : ""; }
extern "C" void pops_install_program(void* /*sys*/) { /* no-op: symbol presence only */ }
extern "C" int  pops_module_operator_count() { return 1; }
extern "C" int  pops_module_state_space_count() { return 0; }
extern "C" int  pops_module_field_space_count() { return 0; }
extern "C" const char* pops_module_operator_owner(int) { return "model/a"; }
extern "C" const char* pops_module_operator_name(int) { return "rhs"; }
extern "C" const char* pops_module_operator_kind(int) { return "hyperbolic"; }
extern "C" const char* pops_module_operator_signature(int) { return "rhs_into"; }
extern "C" const char* pops_module_operator_requirements(int) { return "{\"kind\":\"hyperbolic\"}"; }
extern "C" const char* pops_module_state_space_name(int) { return ""; }
extern "C" const char* pops_module_state_space_owner(int) { return ""; }
extern "C" const char* pops_module_field_space_name(int) { return ""; }
extern "C" const char* pops_module_field_space_owner(int) { return ""; }
)CPP";
  // clang-format on
}

}  // namespace

static int pops_run_test_program_abi_symbols(int argc, char** argv) {
  (void)argc;
  (void)argv;

  const std::string tmp = std::string(POPS_TEST_TMPDIR) + "/program_abi_" +
                          std::to_string(static_cast<long>(std::clock()));
  const std::string src = tmp + ".cpp";
  const std::string so = tmp + ".so";
  {
    std::ofstream f(src);
    f << stub_source();
  }
  const auto package = pops::test::native_dso::compile_shared(src, so);
  if (!package.ok) {
    pops::test::native_dso::report_compile_failure("test_program_abi_symbols", package);
    return 1;
  }

  pops::dynlib::handle h = pops::dynlib::open(so);
  if (!pops::dynlib::valid(h)) {
    std::printf("FAIL dlopen('%s'): %s\n", so.c_str(), pops::dynlib::last_error().c_str());
    return 1;
  }

  int fails = 0;
  // REQUIRED symbols: install_program hard-fails without these.
  const char* required[] = {"pops_program_abi_key",
                            "pops_program_route_manifest",
                            "pops_install_program",
                            "pops_program_block_count",
                            "pops_program_block_name",
                            "pops_program_operator_authority_count",
                            "pops_program_operator_authority_word",
                            "pops_module_operator_count",
                            "pops_module_state_space_count",
                            "pops_module_field_space_count",
                            "pops_module_operator_owner",
                            "pops_module_operator_name",
                            "pops_module_operator_kind",
                            "pops_module_operator_signature",
                            "pops_module_operator_requirements",
                            "pops_module_state_space_name",
                            "pops_module_state_space_owner",
                            "pops_module_field_space_name",
                            "pops_module_field_space_owner"};
  for (const char* name : required) {
    if (!pops::dynlib::sym(h, name)) {
      std::printf("FAIL required ABI symbol '%s' absent from the stub .so\n", name);
      ++fails;
    }
  }
  // Diagnostic identity symbols are present in every current generated artifact but are not part of
  // the execution metadata table itself.
  const char* optional_family[] = {"pops_program_hash", "pops_program_name"};
  for (const char* name : optional_family) {
    if (!pops::dynlib::sym(h, name)) {
      std::printf("FAIL module-metadata ABI symbol '%s' absent from the stub .so\n", name);
      ++fails;
    }
  }

  try {
    const auto metadata = pops::runtime::program::read_module_metadata(h);
    if (metadata.operators.size() != 1 || metadata.operators.front().owner != "model/a" ||
        metadata.operators.front().name != "rhs") {
      std::printf("FAIL strict module metadata reader returned the wrong operator identity\n");
      ++fails;
    }
  } catch (const std::exception& error) {
    std::printf("FAIL strict module metadata reader rejected the complete contract: %s\n",
                error.what());
    ++fails;
  }

  // The ABI key the stub exports equals the module key literal it was compiled against (the guard
  // install_program enforces): resolve and call it, confirm it is non-empty.
  auto key_fn = reinterpret_cast<const char* (*)()>(pops::dynlib::sym(h, "pops_program_abi_key"));
  if (key_fn) {
    const char* k = key_fn();
    if (!k || k[0] == '\0') {
      std::printf("FAIL pops_program_abi_key() returned an empty key\n");
      ++fails;
    }
  }

  pops::dynlib::close(h);

  if (fails == 0)
    std::printf(
        "OK test_program_abi_symbols (all Program + module ABI symbols resolve; key non-empty)\n");
  return fails ? 1 : 0;
}

TEST(test_program_abi_symbols, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_program_abi_symbols, "test_program_abi_symbols"),
            0);
}
