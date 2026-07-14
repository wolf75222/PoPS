// Final production-package ABI guard: a hand-written package cannot exceed RuntimeParams capacity.
#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "test_harness.hpp"

#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/config/runtime_params.hpp>
#include <pops/runtime/system.hpp>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

using namespace pops;

namespace {

std::string parameter_names(int count) {
  std::string result;
  for (int index = 0; index < count; ++index) {
    if (!result.empty())
      result += ',';
    result += "p" + std::to_string(index);
  }
  return result;
}

std::string stub_source() {
  const int count = kMaxRuntimeParams + 1;
  return
      "extern \"C\" const char* pops_native_abi_key() { return \"" + System::abi_key() +
      "\"; }\n"
      "extern \"C\" const char* pops_compiled_route_manifest() { return \"" +
      route_registry_signature() +
      "\"; }\n"
      "extern \"C\" int pops_compiled_nparams() { return " + std::to_string(count) +
      "; }\n"
      "extern \"C\" const char* pops_compiled_param_names() { return \"" +
      parameter_names(count) + "\"; }\n";
}

bool compile_stub(const std::string& source, const std::string& library) {
#if defined(__APPLE__)
  const std::string compiler = "/usr/bin/c++";
#else
  const std::string compiler = POPS_TEST_CXX;
#endif
  std::string command = compiler + " -shared -fPIC -std=" + POPS_TEST_CXX_STD +
                        " -O2 " + source + " -o " + library;
#if defined(__APPLE__)
  command += " -undefined dynamic_lookup";
#endif
  command += " 2> /dev/null";
  return std::system(command.c_str()) == 0;
}

}  // namespace

static int pops_run_test_native_loader_param_overflow(int argc, char** argv) {
  (void)argc;
  (void)argv;
  if (std::string(POPS_TEST_CXX).empty()) {
    std::printf("skip test_native_loader_param_overflow (no C++ compiler)\n");
    return 0;
  }

  const std::string stem = std::string(POPS_TEST_TMPDIR) + "/native_package_param_overflow_" +
                           std::to_string(static_cast<long>(std::clock()));
  const std::string source = stem + ".cpp";
  const std::string library = stem + ".so";
  {
    std::ofstream output(source);
    output << stub_source();
  }
  if (!compile_stub(source, library)) {
    std::printf("skip test_native_loader_param_overflow (stub compilation failed)\n");
    return 0;
  }

  SystemConfig config;
  config.n = 8;
  config.L = 1.0;
  config.periodic = true;
  System system(config);
  std::vector<double> params(static_cast<std::size_t>(kMaxRuntimeParams + 1), 0.0);

  bool threw = false;
  std::string message;
  try {
    system.add_native_block("gas", library, "none", "rusanov", "conservative", "explicit",
                            1.4, 1, true, 1, params, 0.0);
  } catch (const std::exception& error) {
    threw = true;
    message = error.what();
  }

  pops::test::Checker checker;
  checker(threw, "production package rejects parameter vectors above exact capacity");
  checker(message.find(std::to_string(kMaxRuntimeParams + 1)) != std::string::npos,
          "error names the declared count");
  checker(message.find(std::to_string(kMaxRuntimeParams)) != std::string::npos,
          "error names the supported capacity");
  checker(system.n_blocks() == 0, "refused package does not partially install a block");
  return checker.failed();
}

TEST(test_native_loader_param_overflow, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_native_loader_param_overflow,
                                    "test_native_loader_param_overflow"),
            0);
}
