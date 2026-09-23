#include "native_dso_compiler.hpp"
#include "synthetic_loader_fixture.hpp"

#include <fstream>
#include <iostream>

// Build-time only: use the existing exact host compiler/ABI/backend contract.
// The transaction cases consume fresh copies of these binaries and never compile.
int main(int argc, char** argv) {
  try {
    if (argc != 4)
      throw std::invalid_argument("expected variant, source path, and shared-object path");
    std::size_t consumed = 0;
    const auto value = std::stoul(argv[1], &consumed);
    if (consumed != std::string(argv[1]).size() || value > 31)
      throw std::invalid_argument("invalid synthetic loader variant encoding");
    const bool interface_blocks = value & 1, histories = value & 2, attempt_cursor = value & 4,
               with_flux = value & 8, mapping = value & 16;
    (void)pops::test::synthetic_loader::variant(interface_blocks, histories, attempt_cursor,
                                                with_flux, mapping);
    {
      std::ofstream source(argv[2]);
      source.exceptions(std::ios::badbit | std::ios::failbit);
      source << pops::test::synthetic_loader::source(interface_blocks, histories, attempt_cursor,
                                                     with_flux, mapping);
    }
    const auto result = pops::test::native_dso::compile_shared(
        argv[2], argv[3], "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI");
    if (!result.ok) {
      pops::test::native_dso::report_compile_failure("synthetic loader fixture", result);
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
