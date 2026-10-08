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
// pops_program_hash owns the installed frozen clock manifest; pops_program_name is diagnostic.
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
namespace pops { class System; }
extern "C" const char* pops_program_abi_key() { return POPS_ABI_KEY_LITERAL; }
extern "C" const char* pops_program_route_manifest() { return pops::kRouteRegistrySignature; }
extern "C" const char* pops_program_name() { return "abi_symbol_stub"; }
extern "C" const char* pops_program_hash() { return "deadbeef"; }
extern "C" const char* pops_program_checkpoint_clock_manifest_contract() { return "pops.program.owned-clock-manifest@1"; }
extern "C" int pops_program_checkpoint_logical_clock_count() { return 1; }
extern "C" const char* pops_program_checkpoint_logical_clock_identity(int i) { return i == 0 ? "symbol-presence-clock" : ""; }
extern "C" const char* pops_program_checkpoint_primary_clock_identity() { return "symbol-presence-clock"; }
extern "C" int pops_program_operator_authority_count() { return 0; }
extern "C" std::uint64_t pops_program_operator_authority_word(int, int) { return 0; }
extern "C" int pops_program_block_count() { return 1; }
extern "C" const char* pops_program_block_name(int i) { return i == 0 ? "gas" : ""; }
extern "C" void pops_install_program(pops::System* /*sys*/) { /* no-op: symbol presence only */ }
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

// Every rejected image is a real isolated DSO; no System or native provider is installed.
int reject_incomplete_clock_manifests(const std::string& tmp) {
  struct Case {
    const char* name;
    const char* contract;
    const char* omitted;
    const char* expected;
    bool contract_rejected;
  };
  const Case cases[] = {
      {"missing-contract", "", "contract", "pops_program_checkpoint_clock_manifest_contract", true},
      {"null-contract", "nullptr", "", "unsupported Program owned clock manifest contract", true},
      {"empty-contract", "\"\"", "", "unsupported Program owned clock manifest contract", true},
      {"future-version", "\"pops.program.owned-clock-manifest@2\"", "",
       "unsupported Program owned clock manifest contract", true},
      {"different-schema", "\"pops.program.owned-clock-manifest@1.extra\"", "",
       "unsupported Program owned clock manifest contract", true},
      {"missing-count", "\"pops.program.owned-clock-manifest@1\"", "count",
       "pops_program_checkpoint_logical_clock_count", false},
      {"missing-identity", "\"pops.program.owned-clock-manifest@1\"", "identity",
       "pops_program_checkpoint_logical_clock_identity", false},
      {"missing-primary", "\"pops.program.owned-clock-manifest@1\"", "primary",
       "pops_program_checkpoint_primary_clock_identity", false},
  };
  int fails = 0;
  for (const auto& test : cases) {
    const std::string omitted = test.omitted;
    std::string source =
        "static int calls = 0;\n"
        "extern \"C\" int pops_test_clock_callbacks() { return calls; }\n";
    if (omitted != "contract")
      source +=
          "extern \"C\" const char* pops_program_checkpoint_clock_manifest_contract() { return " +
          std::string(test.contract) + "; }\n";
    if (omitted != "count")
      source +=
          "extern \"C\" int pops_program_checkpoint_logical_clock_count() { ++calls; return 1; }\n";
    if (omitted != "identity")
      source +=
          "extern \"C\" const char* pops_program_checkpoint_logical_clock_identity(int) { ++calls; "
          "return \"clock\"; }\n";
    if (omitted != "primary")
      source +=
          "extern \"C\" const char* pops_program_checkpoint_primary_clock_identity() { ++calls; "
          "return \"clock\"; }\n";
    const std::string path = tmp + "-" + test.name;
    {
      std::ofstream file(path + ".cpp");
      file << source;
    }
    const auto package = pops::test::native_dso::compile_shared(path + ".cpp", path + ".so");
    if (!package.ok) {
      pops::test::native_dso::report_compile_failure(test.name, package);
      ++fails;
      continue;
    }
    auto handle = pops::dynlib::open(path + ".so");
    if (!pops::dynlib::valid(handle)) {
      std::printf("FAIL clock manifest negative '%s' cannot load: %s\n", test.name,
                  pops::dynlib::last_error().c_str());
      ++fails;
      continue;
    }
    bool rejected = false;
    try {
      (void)pops::runtime::program::read_program_owned_clock_manifest(handle, "negative-owner");
    } catch (const std::runtime_error& error) {
      rejected = std::string(error.what()).find(test.expected) != std::string::npos;
    }
    if (!rejected) {
      std::printf("FAIL clock manifest negative '%s' did not reject its exact defect\n", test.name);
      ++fails;
    }
    using CallsFn = int (*)();
    const auto calls =
        reinterpret_cast<CallsFn>(pops::dynlib::sym(handle, "pops_test_clock_callbacks"));
    if (!calls || (test.contract_rejected && calls() != 0)) {
      std::printf("FAIL clock manifest negative '%s' invoked tables before schema authentication\n",
                  test.name);
      ++fails;
    }
    pops::dynlib::close(handle);
  }
  return fails;
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
                            "pops_program_hash",
                            "pops_program_checkpoint_clock_manifest_contract",
                            "pops_program_checkpoint_logical_clock_count",
                            "pops_program_checkpoint_logical_clock_identity",
                            "pops_program_checkpoint_primary_clock_identity",
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
  // The human-readable name remains diagnostic rather than a numeric execution selector.
  const char* optional_family[] = {"pops_program_name"};
  for (const char* name : optional_family) {
    if (!pops::dynlib::sym(h, name)) {
      std::printf("FAIL module-metadata ABI symbol '%s' absent from the stub .so\n", name);
      ++fails;
    }
  }

  try {
    using HashFn = const char* (*)();
    const auto hash = reinterpret_cast<HashFn>(pops::dynlib::sym(h, "pops_program_hash"));
    if (!hash) throw std::runtime_error("owned Program hash symbol is absent");
    const auto clocks = pops::runtime::program::read_program_owned_clock_manifest(h, hash());
    if (clocks.primary_clock_identity != "symbol-presence-clock" ||
        clocks.logical_clock_identities != std::vector<std::string>{"symbol-presence-clock"})
      throw std::runtime_error("owned clock metadata reader returned wrong fixture data");
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
  fails += reject_incomplete_clock_manifests(tmp);

  if (fails == 0)
    std::printf(
        "OK test_program_abi_symbols (all Program + module ABI symbols resolve; key non-empty)\n");
  return fails ? 1 : 0;
}

TEST(test_program_abi_symbols, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_program_abi_symbols, "test_program_abi_symbols"),
            0);
}
