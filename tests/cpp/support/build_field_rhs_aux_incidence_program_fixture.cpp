#include <pops/runtime/nd_proof/field_rhs_aux_incidence_program_fixture.hpp>
#include "amr_fixture_criteria.hpp"
#include "native_dso_compiler.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc != 4)
      throw std::invalid_argument(
          "expected source path, shared-object path and criterion fragment");
    {
      std::ofstream output(argv[1]);
      output.exceptions(std::ios::badbit | std::ios::failbit);
      output << "#define pops_program_hash pops_test_untagged_program_hash\n";
      output << pops::test::aux_incidence_value_program::source();
      const auto hash = pops::test::field_rhs_incidence.program_hash(
          pops::test::aux_incidence_value_program::identity);
      output << "\n#undef pops_program_hash\n"
                "extern \"C\" const char* pops_program_hash() { return \""
             << hash << "\"; }\n";
      pops::test::append_fixture_criterion(output, argv[3]);
    }
    const auto result = pops::test::native_dso::compile_shared(
        argv[1], argv[2], "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI");
    if (!result.ok) {
      pops::test::native_dso::report_compile_failure("AMR two-field Aux incidence fixture", result);
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
