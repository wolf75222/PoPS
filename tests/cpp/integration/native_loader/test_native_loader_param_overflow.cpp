// Audit of the runtime-parameter capacity at the native_loader boundary. include/pops/runtime/config/
// runtime_params.hpp declares kMaxRuntimeParams=32 and documents (runtime_params.hpp:24-25) that "a model
// exceeding this bound is REJECTED ON THE PYTHON SIDE (codegen), never here" -- i.e. the C++ loader was
// never meant to see an oversized nparams, because the DSL codegen (python/pops/physics/_authoring_params.py
// assign_runtime_indices) raises a ValueError before emitting a .so whose pops_compiled_nparams() could
// exceed 32. That upstream check does NOT run when a .so is built OUTSIDE the DSL codegen path (a stub
// compiled directly against compiled_block_abi.hpp, exactly the recipe test_amr_native_loader.cpp /
// test_program_loader.cpp use to probe the loader in isolation): nothing stops a hand-written .so from
// declaring pops_compiled_nparams() = 33.
//
// We build such a stub and load it through System::add_compiled_block (include/pops/runtime/builders/
// compiled/native_loader.hpp add_compiled_block<ImplT>) and REQUIRE the ADC-610 defence-in-depth guard
// (#432): add_compiled_block must throw std::runtime_error EARLY, at the host boundary, when the .so
// declares nparams > kMaxRuntimeParams, with a message naming the count and the limit. Before that
// guard landed, the overflow was accepted silently (values beyond index 31 dropped by the clamp in
// compiled_block::make_model_with_params -- never a memory-safety bug, the fixed array is never indexed
// out of bounds, but a silent data loss). This test locks the fail-loud contract so the gap can never
// reopen unnoticed.
//
// Skips (exit 0) under Kokkos or without a known C++ compiler, same policy as the sibling native_loader
// tests (a nu CPU loader is ABI-incompatible with the device module).
#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "test_harness.hpp"  // pops::test::Checker
#include <pops/runtime/config/runtime_params.hpp>  // kMaxRuntimeParams
#include <pops/runtime/system.hpp>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

using namespace pops;

namespace {

// Minimal hand-written stub exposing the FULL extern "C" ABI add_compiled_block requires (mandatory:
// pops_model_nvars / pops_compiled_residual(_p) / pops_compiled_advance(_p) / pops_compiled_max_speed(_p)
// / pops_compiled_poisson_rhs(_p); optional but needed here to engage the `_p` params path:
// pops_compiled_naux / pops_compiled_nparams / pops_compiled_param_defaults). A 1-variable scalar model,
// zero flux / zero source / zero elliptic (values never read in this test: we only probe what happens at
// add_compiled_block TIME, not during a step). pops_compiled_nparams() LIES: it reports
// kMaxRuntimeParams + 1, exactly the shape a codegen bug or a hand-written .so (bypassing the Python
// authoring-time check) could produce.
std::string stub_source() {
  // clang-format off
  return R"CPP(
extern "C" int pops_model_nvars() { return 1; }
extern "C" int pops_compiled_naux() { return 3; }
extern "C" int pops_compiled_nparams() { return )CPP" +
         std::to_string(kMaxRuntimeParams + 1) + R"CPP(; }
extern "C" void pops_compiled_param_defaults(double* out) {
  for (int k = 0; k < )CPP" +
         std::to_string(kMaxRuntimeParams + 1) + R"CPP(; ++k) out[k] = 0.0;
}
extern "C" void pops_compiled_residual(const double*, double* R, const double*, int n, double, double,
                                      int, const char*, const char*, int) {
  for (int k = 0; k < n * n; ++k) R[k] = 0.0;
}
extern "C" void pops_compiled_residual_p(const double*, double* R, const double*, int n, double, double,
                                        int, const char*, const char*, int, const double*, int, double) {
  for (int k = 0; k < n * n; ++k) R[k] = 0.0;
}
extern "C" void pops_compiled_advance(double*, const double*, int, double, double, int, const char*,
                                     const char*, int, int, double, int) {}
extern "C" void pops_compiled_advance_p(double*, const double*, int, double, double, int, const char*,
                                       const char*, int, int, double, int, const double*, int, double) {}
extern "C" double pops_compiled_max_speed(const double*, const double*, int, double, double, int) {
  return 0.0;
}
extern "C" double pops_compiled_max_speed_p(const double*, const double*, int, double, double, int,
                                           const double*, int) {
  return 0.0;
}
extern "C" void pops_compiled_poisson_rhs(const double*, double* rhs, int n) {
  for (int k = 0; k < n * n; ++k) rhs[k] = 0.0;
}
extern "C" void pops_compiled_poisson_rhs_p(const double*, double* rhs, int n, const double*, int) {
  for (int k = 0; k < n * n; ++k) rhs[k] = 0.0;
}
)CPP";
  // clang-format on
}

bool compile_stub(const std::string& src_path, const std::string& so_path) {
#if defined(__APPLE__)
  const std::string cc = "/usr/bin/c++";  // xcrun wrapper: resolves the SDK sysroot (same clang family)
#else
  const std::string cc = POPS_TEST_CXX;
#endif
  std::string cmd = cc + " -shared -fPIC -std=" + POPS_TEST_CXX_STD + " -O2 -I " + POPS_TEST_INCLUDE +
                    " " + src_path + " -o " + so_path;
#if defined(__APPLE__)
  cmd += " -undefined dynamic_lookup";
#endif
  cmd += " 2> /dev/null";
  return std::system(cmd.c_str()) == 0;
}

}  // namespace

static int pops_run_test_native_loader_param_overflow(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  (void)argc;
  (void)argv;
  std::printf("skip test_native_loader_param_overflow (backend Kokkos: nu CPU stub incompatible)\n");
  return 0;
#else
  (void)argc;
  (void)argv;
  const char* cxx = POPS_TEST_CXX;
  if (!cxx || cxx[0] == '\0') {
    std::printf("skip test_native_loader_param_overflow (no C++ compiler known to the build)\n");
    return 0;
  }

  pops::test::Checker chk;

  const std::string tmp = std::string(POPS_TEST_TMPDIR) + "/native_loader_param_overflow_" +
                          std::to_string(static_cast<long>(std::clock()));
  const std::string src = tmp + ".cpp";
  const std::string so = tmp + ".so";
  {
    std::ofstream f(src);
    f << stub_source();
  }
  if (!compile_stub(src, so)) {
    std::printf(
        "skip test_native_loader_param_overflow (stub .so compilation failed -- headers/std?)\n");
    return 0;
  }

  const int n = 8;
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  System sys(cfg);

  // ADC-610 (#432) defence-in-depth contract: the loader must REFUSE the oversized declaration EARLY,
  // at add_compiled_block time, with a message naming the offending count and the kMaxRuntimeParams
  // limit -- never a silent clamp, never a partially-registered block.
  bool threw = false;
  std::string what;
  try {
    sys.add_compiled_block("gas", so, "none", "rusanov", "conservative", "explicit");
  } catch (const std::exception& e) {
    threw = true;
    what = e.what();
  }
  chk(threw,
      "add_compiled_block throws on nparams=kMaxRuntimeParams+1 (the #432 host-boundary guard)");
  if (threw) {
    chk(what.find(std::to_string(kMaxRuntimeParams)) != std::string::npos,
        "the overflow error names the kMaxRuntimeParams limit");
    chk(what.find(std::to_string(kMaxRuntimeParams + 1)) != std::string::npos,
        "the overflow error names the offending declared count");
  } else {
    std::printf("FAIL: add_compiled_block accepted nparams=%d without throwing; the ADC-610 guard "
                "(native_loader.hpp) is missing or bypassed\n",
                kMaxRuntimeParams + 1);
  }
  // The refused block must not have been registered: the System stays empty and usable.
  chk(sys.n_blocks() == 0, "the refused compiled block is not partially registered");

  if (chk.fails() == 0)
    std::printf(
        "OK test_native_loader_param_overflow (documents the current native_loader nparams "
        "capacity gap; D-phase full build should re-run this after any guard is added)\n");
  return chk.failed();
#endif  // POPS_HAS_KOKKOS
}

TEST(test_native_loader_param_overflow, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_native_loader_param_overflow,
                                    "test_native_loader_param_overflow"),
            0);
}
