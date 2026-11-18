#include "field_rhs_value_program_fixture.hpp"
#ifdef POPS_TEST_GENERATED_FIXTURE_CRITERION
#include "field_rhs_fixture_criterion.hpp"
#endif
#include "native_dso_compiler.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
#ifdef POPS_TEST_GENERATED_FIXTURE_CRITERION
    if (argc != 4)
      throw std::invalid_argument(
          "expected source path, shared-object path and criterion fragment");
#else
    if (argc != 3)
      throw std::invalid_argument("expected source path and shared-object path");
#endif
    {
      std::ofstream output(argv[1]);
      output.exceptions(std::ios::badbit | std::ios::failbit);
#ifdef POPS_TEST_GENERATED_FIXTURE_CRITERION
      output << pops::test::qualify_fixture_program_source(
          pops::test::field_rhs_value_program::source(),
          pops::test::field_rhs_value_program::identity, pops::test::field_rhs_value);
      pops::test::append_fixture_criterion(output, argv[3]);
#else
      output << pops::test::field_rhs_value_program::source();
#endif
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
