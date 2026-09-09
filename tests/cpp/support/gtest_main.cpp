#include <gtest/gtest.h>
#include "gtest_rank_output.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  std::optional<pops::test::RankedGtestOutput> ranked;
  std::vector<std::string> storage;
  std::vector<char*> arguments;
  try {
    ranked = pops::test::prepare_ranked_gtest_output(argc, argv, std::getenv);
    if (ranked && ranked->argument) {
      storage.reserve(static_cast<std::size_t>(argc) + 1);
      for (int index = 0; index < argc; ++index)
        storage.emplace_back(argv[index]);
      if (ranked->replace_argument >= 0) {
        storage[static_cast<std::size_t>(ranked->replace_argument)] = *ranked->argument;
      } else {
        storage.push_back(
            *ranked->argument);  // CLI overrides GTEST_OUTPUT before listener creation.
      }
      arguments.reserve(storage.size() + 1);
      for (auto& value : storage)
        arguments.push_back(value.data());
      arguments.push_back(nullptr);
      argc = static_cast<int>(storage.size());
      argv = arguments.data();
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "PoPS GoogleTest evidence configuration error: %s\n", error.what());
    return 2;
  }
  ::testing::InitGoogleTest(&argc, argv);
  if (ranked) {
    // Global RecordProperty calls become direct <testsuites>/<properties>/<property> entries.
    ::testing::Test::RecordProperty("pops_mpi_rank", ranked->rank);
    ::testing::Test::RecordProperty("pops_mpi_expected_size", ranked->expected_size);
    ::testing::Test::RecordProperty("pops_mpi_rank_source", ranked->rank_source);
  }
  return RUN_ALL_TESTS();
}
