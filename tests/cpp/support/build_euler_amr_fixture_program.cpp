#include "euler_amr_fixture_criteria.hpp"
#include "native_dso_compiler.hpp"

#include <fstream>
#include <initializer_list>
#include <iostream>

namespace {
template <std::size_t Size>
void write_definitions(std::ostream& output, const pops::test::FixtureCriterionGroup<Size>& group,
                       std::string_view program = "test-forward-euler-amr",
                       std::initializer_list<std::string_view> blocks = {}) {
  output << "#define POPS_TEST_EULER_BLOCKS {";
  if (blocks.size() == 0) {
    for (const auto& field : group.fields)
      output << '"' << field.block << "\",";
  } else {
    for (const auto block : blocks)
      output << '"' << block << "\",";
  }
  output << "}\n#define POPS_TEST_EULER_HASH \"" << program << '/' << group.identity << "\"\n";
}

void append(std::ostream& output, const char* path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("cannot read Forward Euler Program fixture input");
  output << '\n' << input.rdbuf();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7)
      throw std::invalid_argument(
          "expected fixture variant, body, criterion, include, source and DSO");
    const std::string_view variant(argv[1]);
    {
      std::ofstream source(argv[5]);
      source.exceptions(std::ios::badbit | std::ios::failbit);
      if (variant == "compiled_parity")
        write_definitions(source, pops::test::compiled_parity);
      else if (variant == "regrid_parity")
        write_definitions(source, pops::test::regrid_parity);
      else if (variant == "gravity")
        write_definitions(source, pops::test::gravity);
      else if (variant == "riemann")
        write_definitions(source, pops::test::riemann);
      else if (variant == "magnetic")
        write_definitions(source, pops::test::magnetic);
      else if (variant == "magnetic_union")
        write_definitions(source, pops::test::magnetic_union);
      else if (variant == "fickian")
        write_definitions(source, pops::test::fickian);
      else if (variant == "prepared_constitutive") {
        write_definitions(source, pops::test::fickian, "test-prepared-constitutive-amr");
        source << "#define POPS_TEST_PREPARED_CONSTITUTIVE_AMR\n";
      } else if (variant == "empty_cadence") {
        write_definitions(source, pops::test::empty_cadence, "test-empty-amr-cadence");
        source << "#define POPS_TEST_EMPTY_AMR_CADENCE\n";
      } else if (variant == "noop_history") {
        write_definitions(source, pops::test::noop_history, "test-noop-history-amr");
        source << "#define POPS_TEST_NOOP_HISTORY\n"
               << "#define POPS_TEST_EULER_FINE_CLOCK \"clock.level.1\"\n";
      } else if (variant == "positivity_floor") {
        write_definitions(source, pops::test::positivity_floor, "test-positivity-floor-amr",
                          {"density", "density_peer"});
        source << "#define POPS_TEST_POSITIVITY_FLOOR\n";
      } else if (variant == "declared_subcycling") {
        write_definitions(source, pops::test::declared_subcycling, "test-declared-subcycling-amr");
        source << "#define POPS_TEST_DECLARED_SUBCYCLING\n"
               << "#define POPS_TEST_EULER_FINE_CLOCK \"clock.level.1\"\n"
               << "#define POPS_TEST_EULER_SECOND_FINE_CLOCK \"clock.level.2\"\n";
      } else if (variant == "mixed_cfl") {
        write_definitions(source, pops::test::mixed_cfl, "test-mixed-cfl-amr",
                          {"tracer", "storage"});
        source << "#define POPS_TEST_MIXED_CFL\n"
               << "#define POPS_TEST_EULER_FINE_CLOCK \"clock.level.1\"\n";
      } else if (variant == "mixed_norm" || variant == "mixed_norm_wrong") {
        write_definitions(
            source, variant == "mixed_norm" ? pops::test::mixed_norm : pops::test::mixed_norm_wrong,
            std::string("test-") + argv[1]);
        source
            << "#define POPS_TEST_MIXED_NORM\n"
            << "#define POPS_TEST_EULER_FINE_CLOCK \"tests.norm2.mixed-level/clock/level-1@1\"\n";
      } else if (variant == "scalar_history" || variant == "scalar_history_ghost" ||
                 variant == "scalar_history_output_2" || variant == "scalar_history_dense_1" ||
                 variant == "scalar_history_dense_2") {
        write_definitions(source, pops::test::scalar_history, std::string("test-") + argv[1]);
        source << "#define POPS_TEST_SCALAR_HISTORY "
               << (variant == "scalar_history_ghost" ? 1 : -1) << '\n'
               << "#define POPS_TEST_EULER_FINE_CLOCK \"clock.level.1\"\n"
               << "#define POPS_TEST_SCALAR_HISTORY_RATIO " << (variant.ends_with("_2") ? 2 : 1)
               << '\n'
               << "#define POPS_TEST_SCALAR_HISTORY_DENSE "
               << (variant == "scalar_history_dense_1" || variant == "scalar_history_dense_2")
               << '\n';
      } else
        throw std::invalid_argument("unknown AMR Program fixture");
      source << "#define POPS_TEST_EULER_SOLVE_DEFAULT "
             << (variant == "gravity" || variant == "riemann") << '\n';
      if (variant == "riemann")
        source << "#define POPS_TEST_EULER_FINE_CLOCK \"tests.amr-riemann-native/clock/level-1\"\n";
      append(source, argv[2]);
      append(source, argv[3]);
    }
    const auto flags =
        "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI -I" + pops::test::native_dso::shell_quote(argv[4]);
    const auto result = pops::test::native_dso::compile_shared(argv[5], argv[6], flags);
    if (!result.ok) {
      pops::test::native_dso::report_compile_failure("Forward Euler Program fixture", result);
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
