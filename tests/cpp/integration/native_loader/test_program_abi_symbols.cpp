// ADC-538: ABI SYMBOL-PRESENCE fence for the compiled time-Program loader path (epic ADC-399 /
// ADC-401 Phase 2c). System::install_program dlopens a generated problem.so and resolves a contract of
// extern "C" symbols across the ABI boundary; test_program_loader.cpp proves the end-to-end numeric
// step, while this test isolates the SYMBOL PRESENCE half: it compiles a stub problem.so exporting the
// sole v5 candidate install entry and asserts it resolves from a runtime-compiled DSO. This closes
// the ABI symbol-presence proof without relying on a fake process-handle probe.
// The stub is compiled with the exact compiler/Kokkos contract injected by CMake. A missing compiler
// or compilation failure is a hard failure because otherwise no ABI symbol was actually proven.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"
#include <pops/runtime/dynamic/dynlib.hpp>
#include <pops/runtime/program/program_abi.hpp>
#include <pops/runtime/program/module_metadata.hpp>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <type_traits>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

// The generated problem.so surface: the sole v5 entry a real codegen emits. Hand-written here for an
// autonomous symbol-presence test (no numeric body needed -- the install entry is a no-op stub, it
// only has to EXIST and be resolvable). The ABI key is the preprocessor LITERAL, like test_program_loader.
std::string stub_source() {
  // clang-format off
  return R"CPP(
#include <pops/runtime/program/program_abi.hpp>
#include <type_traits>
extern "C" bool pops_install_program(
    const pops::runtime::program::ProgramHostDescriptor*,
    pops::runtime::program::ProgramCandidateDescriptor*,
    pops::runtime::program::ProgramInstallDiagnostic*) noexcept { return false; }
static_assert(std::is_same_v<decltype(&pops_install_program),
                             pops::runtime::program::ProgramInstallFn>);
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
  // ABI v5 has one entry point.  All identity and metadata travel in its fixed-layout candidate
  // tables; no legacy accessor family is part of the dynamic surface.
  const char* required[] = {"pops_install_program"};
  for (const char* name : required) {
    if (!pops::dynlib::sym(h, name)) {
      std::printf("FAIL required ABI symbol '%s' absent from the stub .so\n", name);
      ++fails;
    }
  }
  auto install = reinterpret_cast<pops::runtime::program::ProgramInstallFn>(
      pops::dynlib::sym(h, "pops_install_program"));
  pops::runtime::program::ProgramInstallDiagnostic diagnostic{};
  if (!install || install(nullptr, nullptr, &diagnostic)) {
    std::printf("FAIL pops_install_program does not satisfy the v5 bool candidate ABI\n");
    ++fails;
  }

  pops::dynlib::close(h);

  if (fails == 0)
    std::printf(
        "OK test_program_abi_symbols (sole v5 Program install entry resolves)\n");
  return fails ? 1 : 0;
}

TEST(test_program_abi_symbols, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_program_abi_symbols, "test_program_abi_symbols"),
            0);
}
