#include "field_rhs_value_program_fixture.hpp"
#include "native_dso_compiler.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument("expected source path and shared-object path");
    {
      std::ofstream output(argv[1]);
      output.exceptions(std::ios::badbit | std::ios::failbit);
      output << pops::test::field_rhs_value_program::source();
    }
    const auto result = pops::test::native_dso::compile_shared(
      argv[1], argv[2], "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI");
    if (!result.ok) {
      pops::test::native_dso::report_compile_failure("Field RHS value Program fixture", result);
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
