#pragma once

#include <pops/core/identity/sha256.hpp>
#include <pops/parallel/world_communicator.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if !defined(POPS_NATIVE_DIM)
#error "native DSO tests require the host target's exact POPS_NATIVE_DIM"
#endif

namespace pops::test::native_dso {

inline std::string shell_quote(const std::string& value) {
  std::string quoted = "'";
  for (const char ch : value) {
    if (ch == '\'')
      quoted += "'\\''";
    else
      quoted.push_back(ch);
  }
  quoted.push_back('\'');
  return quoted;
}

// POPS_NATIVE_KOKKOS_* and POPS_NATIVE_MPI_* are serialized with `|`: this preserves paths,
// definitions and options containing spaces. Each record becomes exactly one compiler argv item.
// CMake's explicit SHELL: records are already a trusted toolchain fragment and retain their
// intended word splitting.
inline void append_serialized_flags(std::string& command, const char* value,
                                    const std::string& prefix = {}) {
  const std::string records = value != nullptr ? value : "";
  std::size_t begin = 0;
  while (begin <= records.size()) {
    const std::size_t end = records.find('|', begin);
    const std::string token =
        records.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (!token.empty()) {
      if (prefix.empty() && token.rfind("SHELL:", 0) == 0)
        command += " " + token.substr(6);
      else
        command += " " + shell_quote(prefix + token);
    }
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
}

struct CompileResult {
  bool ok = false;
  int status = -1;
  std::string compiler;
  std::string log_path;
};

struct CollectiveCompileResult {
  bool ok = false;
  std::string library_path;
};

namespace detail {

inline std::string sha256_of_bytes(const std::string& bytes) {
  return identity::sha256_hex(std::vector<std::uint8_t>(bytes.begin(), bytes.end()));
}

inline std::string rank_local_path(const std::string& requested_path, int rank) {
  const std::filesystem::path requested(requested_path);
  const std::filesystem::path parent = requested.parent_path();
  const std::string extension = requested.extension().string();
  const std::string stem = requested.stem().string();
  return (parent / (stem + ".rank" + std::to_string(rank) + extension)).string();
}

inline std::string rank_zero_source_path(const std::string& requested_path) {
  const std::filesystem::path requested(requested_path);
  return (requested.parent_path() /
          (requested.stem().string() + ".rank0" + requested.extension().string()))
      .string();
}

inline std::string source_consensus_record(std::string_view identity, std::string_view status,
                                           std::size_t bytes, std::string_view sha256) {
  return std::string(identity) + "\n" + std::string(status) + "\n" + std::to_string(bytes) + "\n" +
         std::string(sha256);
}

inline std::string exception_text() {
  try {
    throw;
  } catch (const std::exception& error) {
    return error.what();
  } catch (...) {
    return "unknown source-generation exception";
  }
}

}  // namespace detail

// Compile a runtime-generated native package with the exact compiler, header signature, Kokkos and
// MPI development contracts of the host test target. A native-loader proof is invalid if its DSO
// silently drops any native backend carried by the executable that will dlopen it.
inline CompileResult compile_shared(const std::string& source_path, const std::string& library_path,
                                    const std::string& extra_flags = {}) {
  CompileResult result;
  result.compiler = POPS_TEST_CXX;
  result.log_path = library_path + ".log";
  if (result.compiler.empty())
    return result;

  std::string command = shell_quote(result.compiler) +
                        " -shared -fPIC -std=" + std::string(POPS_TEST_CXX_STD) + " -O2 -I" +
                        shell_quote(POPS_TEST_INCLUDE);
  command += " -DPOPS_NATIVE_DIM=" + std::to_string(POPS_NATIVE_DIM);
  command += " -DPOPS_RUNTIME_SHARED_EXCEPTION_ABI";
#if defined(POPS_TEST_HEADER_SIG)
  command += " -D" + shell_quote(std::string("POPS_HEADER_SIG=\"") + POPS_TEST_HEADER_SIG + "\"");
#endif
#if defined(POPS_HAS_KOKKOS)
  append_serialized_flags(command, POPS_TEST_KOKKOS_INC, "-I");
  append_serialized_flags(command, POPS_TEST_KOKKOS_DEFS, "-D");
  append_serialized_flags(command, POPS_TEST_KOKKOS_OPTS);
  command += " -DPOPS_HAS_KOKKOS";
#endif
#if defined(POPS_HAS_MPI)
  append_serialized_flags(command, POPS_TEST_MPI_INCLUDE, "-I");
  append_serialized_flags(command, POPS_TEST_MPI_COMPILE_DEFINITIONS, "-D");
  append_serialized_flags(command, POPS_TEST_MPI_COMPILE_OPTIONS);
  command +=
      " -DPOPS_HAS_MPI -D" + shell_quote(std::string("POPS_MPI_ABI=\"") + POPS_TEST_MPI_ABI + "\"");
#endif
  if (!extra_flags.empty())
    command += " " + extra_flags;
  command += " " + shell_quote(source_path) + " -o " + shell_quote(library_path);
#if defined(POPS_HAS_MPI)
  append_serialized_flags(command, POPS_TEST_MPI_LINK_OPTIONS);
  append_serialized_flags(command, POPS_TEST_MPI_LINK_LIBRARIES);
#endif
#if defined(__APPLE__)
  command += " -undefined dynamic_lookup";
#endif
  command += " >" + shell_quote(result.log_path) + " 2>&1";
  result.status = std::system(command.c_str());
  result.ok = result.status == 0;
  return result;
}

inline void report_compile_failure(const char* test_name, const CompileResult& result);

/// Build one generated test package collectively when an MPI test has several ranks.  Every rank
/// first generates and authenticates the exact source bytes; only rank zero invokes the compiler.
/// The resulting DSO is then broadcast, authenticated again, and written to a rank-local path so
/// no rank races another rank's dlopen input.  Serial and one-rank MPI callers retain the existing
/// direct source/compile path.
template <class SourceFactory>
inline CollectiveCompileResult compile_shared_collectively(
    std::string_view source_identity, SourceFactory&& make_source, const std::string& source_path,
    const std::string& library_path, const char* test_name, const std::string& extra_flags = {}) {
  auto compile_direct = [&]() -> CollectiveCompileResult {
    std::string source;
    try {
      source = std::forward<SourceFactory>(make_source)();
    } catch (...) {
      throw std::runtime_error("native package source generation failed: " +
                               detail::exception_text());
    }
    std::ofstream output(source_path, std::ios::binary);
    if (!output)
      throw std::runtime_error("cannot create native package source");
    output.write(source.data(), static_cast<std::streamsize>(source.size()));
    output.close();
    if (!output)
      throw std::runtime_error("cannot write native package source");
    const CompileResult compiled = compile_shared(source_path, library_path, extra_flags);
    if (!compiled.ok) {
      report_compile_failure(test_name, compiled);
      return {};
    }
    return {true, library_path};
  };

#if !defined(POPS_HAS_MPI)
  return compile_direct();
#else
  // A serial test may be linked against an MPI-capable runtime without owning an active world.
  // Do not initialize MPI merely to compile a local fixture: the collective path is only an
  // authority once the test harness has established a real multi-rank process world.
  if (!comm_active())
    return compile_direct();
  WorldCommunicator& world = WorldCommunicator::world();
  if (world.size() <= 1)
    return compile_direct();

  std::string source;
  std::string source_error;
  try {
    source = std::forward<SourceFactory>(make_source)();
  } catch (...) {
    source_error = detail::exception_text();
  }
  const std::string source_digest = source_error.empty() ? detail::sha256_of_bytes(source) : "";
  const std::string local_source_record = detail::source_consensus_record(
      source_identity, source_error.empty() ? "ok" : "error", source.size(), source_digest);
  const std::vector<std::string> source_records = world.allgather_bytes(local_source_record);
  if (source_records.empty() || !std::all_of(source_records.begin(), source_records.end(),
                                             [&local_source_record](const std::string& record) {
                                               return record == local_source_record;
                                             })) {
    throw std::runtime_error(
        "MPI ranks generated non-identical native package source identities, lengths, or SHA-256");
  }
  if (!source_error.empty())
    throw std::runtime_error("collective native package source generation failed: " + source_error);

  const int rank = world.rank();
  const std::string root_source_path = detail::rank_zero_source_path(source_path);
  const std::string root_library_path = detail::rank_local_path(library_path, 0);
  std::string status;
  std::string package_bytes;
  if (rank == 0) {
    try {
      std::ofstream output(root_source_path, std::ios::binary);
      if (!output)
        throw std::runtime_error("cannot create rank-zero native package source");
      output.write(source.data(), static_cast<std::streamsize>(source.size()));
      output.close();
      if (!output)
        throw std::runtime_error("cannot write rank-zero native package source");
      const CompileResult compiled =
          compile_shared(root_source_path, root_library_path, extra_flags);
      if (!compiled.ok) {
        report_compile_failure(test_name, compiled);
        status = "compile-failed:" + std::to_string(compiled.status);
      } else {
        std::ifstream binary(root_library_path, std::ios::binary);
        if (!binary)
          throw std::runtime_error("cannot read rank-zero native package DSO");
        package_bytes.assign(std::istreambuf_iterator<char>(binary),
                             std::istreambuf_iterator<char>());
        if (!binary.good() && !binary.eof())
          throw std::runtime_error("cannot read complete rank-zero native package DSO");
        status = "ok\n" + std::to_string(package_bytes.size()) + "\n" +
                 detail::sha256_of_bytes(package_bytes);
      }
    } catch (const std::exception& error) {
      status = "prepare-failed:" + std::string(error.what());
    } catch (...) {
      status = "prepare-failed:unknown exception";
    }
  }
  status = world.broadcast_bytes(std::move(status), 0);
  if (status.rfind("ok\n", 0) != 0)
    throw std::runtime_error("rank-zero collective native package build failed: " + status);
  const std::size_t first_separator = status.find('\n');
  const std::size_t second_separator = status.find('\n', first_separator + 1);
  if (first_separator == std::string::npos || second_separator == std::string::npos ||
      second_separator + 1 >= status.size())
    throw std::runtime_error("rank-zero collective native package status is malformed");
  std::size_t expected_bytes = 0;
  try {
    expected_bytes =
        std::stoull(status.substr(first_separator + 1, second_separator - first_separator - 1));
  } catch (...) {
    throw std::runtime_error("rank-zero collective native package size is malformed");
  }
  const std::string expected_digest = status.substr(second_separator + 1);
  if (expected_digest.size() != 64)
    throw std::runtime_error("rank-zero collective native package SHA-256 is malformed");

  package_bytes = world.broadcast_bytes(std::move(package_bytes), 0);
  if (package_bytes.size() != expected_bytes ||
      detail::sha256_of_bytes(package_bytes) != expected_digest)
    throw std::runtime_error(
        "broadcast native package DSO does not match rank-zero size or SHA-256");

  const std::string local_library_path = detail::rank_local_path(library_path, rank);
  std::string local_write_error;
  try {
    std::ofstream output(local_library_path, std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error("cannot create rank-local native package DSO");
    output.write(package_bytes.data(), static_cast<std::streamsize>(package_bytes.size()));
    output.close();
    if (!output)
      throw std::runtime_error("cannot write rank-local native package DSO");
    std::ifstream binary(local_library_path, std::ios::binary);
    if (!binary)
      throw std::runtime_error("cannot reopen rank-local native package DSO");
    const std::string local_bytes((std::istreambuf_iterator<char>(binary)),
                                  std::istreambuf_iterator<char>());
    if (local_bytes.size() != expected_bytes ||
        detail::sha256_of_bytes(local_bytes) != expected_digest)
      throw std::runtime_error("rank-local native package DSO failed SHA-256 verification");
  } catch (const std::exception& error) {
    local_write_error = error.what();
  } catch (...) {
    local_write_error = "unknown rank-local native package write failure";
  }
  const std::vector<std::string> write_errors = world.allgather_bytes(local_write_error);
  const auto write_failure = std::find_if(write_errors.begin(), write_errors.end(),
                                          [](const std::string& error) { return !error.empty(); });
  if (write_failure != write_errors.end())
    throw std::runtime_error("collective rank-local native package write failed: " +
                             *write_failure);
  world.barrier();
  return {true, local_library_path};
#endif
}

inline void report_compile_failure(const char* test_name, const CompileResult& result) {
  if (result.compiler.empty()) {
    std::fprintf(stderr, "%s: POPS_TEST_CXX is empty; native package was not compiled\n",
                 test_name);
    return;
  }
  std::fprintf(stderr, "%s: native package compilation failed with %s (status %d); log: %s\n",
               test_name, result.compiler.c_str(), result.status, result.log_path.c_str());
}

}  // namespace pops::test::native_dso
