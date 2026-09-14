#pragma once

#include <charconv>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pops::test {

struct RankedGtestOutput {
  int rank{};
  int expected_size{};
  std::string rank_source;
  std::optional<std::string> argument;
  int replace_argument{-1};
};

inline int gtest_launcher_integer(std::string_view text, const char* name, bool positive) {
  int value{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || (text.size() > 1 && text.front() == '0') || text.front() < '0' ||
      text.front() > '9' || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      value < (positive ? 1 : 0)) {
    throw std::runtime_error(std::string("invalid canonical launcher integer: ") + name);
  }
  return value;
}

// Pure argument projection: neither MPI nor GoogleTest is initialized here. The caller keeps
// the rewritten argument storage alive through InitGoogleTest and RUN_ALL_TESTS.
template <class Environment>
std::optional<RankedGtestOutput> prepare_ranked_gtest_output(int argc, char* const* argv,
                                                             Environment environment) {
  const char* expected = environment("POPS_TEST_EXPECT_RANKS");
  if (expected == nullptr)
    return std::nullopt;  // Ordinary serial invocations are unchanged.

  RankedGtestOutput result;
  result.expected_size = gtest_launcher_integer(expected, "POPS_TEST_EXPECT_RANKS", true);
  bool found = false;
  const auto accept_family = [&](const char* rank_name, const char* size_name, const char* source) {
    if (found)
      return;  // Lower-priority metadata may belong to an enclosing MPI job.
    const char* rank_text = environment(rank_name);
    const char* size_text = environment(size_name);
    if (rank_text == nullptr && size_text == nullptr)
      return;
    if (rank_text == nullptr || size_text == nullptr) {
      throw std::runtime_error(std::string("incomplete launcher rank/size family: ") + source);
    }
    const int rank = gtest_launcher_integer(rank_text, rank_name, false);
    const int size = gtest_launcher_integer(size_text, size_name, true);
    if (size != result.expected_size || rank >= size) {
      throw std::runtime_error(std::string("launcher rank/size differs from expected ranks: ") +
                               source);
    }
    result.rank = rank;
    result.rank_source = source;
    found = true;
  };
  // Prefer the MPI launcher's world identity. SLURM_PROCID/SIZE may describe an enclosing job;
  // they are deliberately not evidence of this mpiexec invocation. PMIX_RANK alone lacks size.
  accept_family("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD");
  accept_family("PMI_RANK", "PMI_SIZE", "PMI");
  if (!found) {
    throw std::runtime_error("unsupported launcher: expected OMPI or PMI world rank/size pair");
  }

  std::optional<std::string> output;
  constexpr std::string_view prefix = "--gtest_output=";
  constexpr std::string_view flagfile_prefix = "--gtest_flagfile=";
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg(argv[index]);
    if (arg == "--gtest_output" || arg == "--gtest_flagfile" ||
        arg.substr(0, flagfile_prefix.size()) == flagfile_prefix) {
      throw std::runtime_error("ranked GoogleTest output requires direct --gtest_output=value");
    }
    if (arg.substr(0, prefix.size()) != prefix)
      continue;
    if (output)
      throw std::runtime_error("duplicate --gtest_output arguments");
    output = std::string(arg.substr(prefix.size()));
    result.replace_argument = index;
  }
  if (!output) {
    if (const char* value = environment("GTEST_OUTPUT"))
      output = value;
  }
  if (!output || output->empty())
    return result;  // No XML was requested.

  const auto colon = output->find(':');
  const std::string format = output->substr(0, colon);
  if (format != "xml" && format != "json") {
    throw std::runtime_error("ranked GoogleTest output supports XML or JSON");
  }
  const std::string suffix = "." + format;
  std::string path =
      colon == std::string::npos ? "test_detail" + suffix : output->substr(colon + 1);
  // As in GoogleTest, a trailing separator (or empty explicit path) denotes a directory.
  // Rank-qualified executable names replace its shared-file uniqueness probe.
  const auto native_path = std::filesystem::path(path);
  if (path.empty() || !native_path.has_filename()) {
    auto executable = std::filesystem::path(argv[0]).filename();
#ifdef _WIN32
    if (executable.extension() == ".exe")
      executable.replace_extension();
#endif
    if (executable.empty())
      throw std::runtime_error("ranked output requires an executable name");
    path = (native_path / executable).string() + suffix;
  }
  if (path.size() >= suffix.size() && path.substr(path.size() - suffix.size()) == suffix) {
    path.resize(path.size() - suffix.size());
  }
  result.argument =
      "--gtest_output=" + format + ":" + path + ".rank" + std::to_string(result.rank) + suffix;
  return result;
}

}  // namespace pops::test
