#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

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

inline void append_include_flags(std::string& command, const char* value) {
  std::string token;
  const std::string flags = value != nullptr ? value : "";
  for (std::size_t index = 0; index <= flags.size(); ++index) {
    if (index == flags.size() || flags[index] == ' ') {
      if (!token.empty())
        command += " -I" + shell_quote(token);
      token.clear();
    } else {
      token.push_back(flags[index]);
    }
  }
}

inline void append_definition_flags(std::string& command, const char* value) {
  std::string token;
  const std::string flags = value != nullptr ? value : "";
  for (std::size_t index = 0; index <= flags.size(); ++index) {
    if (index == flags.size() || flags[index] == ' ') {
      if (!token.empty())
        command += " -D" + shell_quote(token);
      token.clear();
    } else {
      token.push_back(flags[index]);
    }
  }
}

// POPS_NATIVE_MPI_* is serialized by PopsMpiContract.cmake with `|`: unlike the historical
// space-splitting Kokkos test seam, this preserves paths and definitions containing spaces. Each
// record becomes exactly one compiler argv item. CMake's explicit SHELL: records are already a
// trusted toolchain fragment and retain their intended word splitting.
inline void append_serialized_flags(std::string& command, const char* value,
                                    const std::string& prefix = {}) {
  const std::string records = value != nullptr ? value : "";
  std::size_t begin = 0;
  while (begin <= records.size()) {
    const std::size_t end = records.find('|', begin);
    const std::string token = records.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
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

  std::string command = shell_quote(result.compiler) + " -shared -fPIC -std=" +
                        std::string(POPS_TEST_CXX_STD) + " -O2 -I" +
                        shell_quote(POPS_TEST_INCLUDE);
#if defined(POPS_TEST_HEADER_SIG)
  command += " -D" + shell_quote(std::string("POPS_HEADER_SIG=\"") + POPS_TEST_HEADER_SIG + "\"");
#endif
#if defined(POPS_HAS_KOKKOS)
  append_include_flags(command, POPS_TEST_KOKKOS_INC);
  std::string options = POPS_TEST_KOKKOS_OPTS;
  for (std::size_t position = options.find("SHELL:"); position != std::string::npos;
       position = options.find("SHELL:"))
    options.erase(position, 6);
  if (!options.empty())
    command += " " + options;
  append_definition_flags(command, POPS_TEST_KOKKOS_DEFS);
  command += " -DPOPS_HAS_KOKKOS";
#endif
#if defined(POPS_HAS_MPI)
  append_serialized_flags(command, POPS_TEST_MPI_INCLUDE, "-I");
  append_serialized_flags(command, POPS_TEST_MPI_COMPILE_DEFINITIONS, "-D");
  append_serialized_flags(command, POPS_TEST_MPI_COMPILE_OPTIONS);
  command += " -DPOPS_HAS_MPI -D" +
             shell_quote(std::string("POPS_MPI_ABI=\"") + POPS_TEST_MPI_ABI + "\"");
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

inline void report_compile_failure(const char* test_name, const CompileResult& result) {
  if (result.compiler.empty()) {
    std::fprintf(stderr, "%s: POPS_TEST_CXX is empty; native package was not compiled\n", test_name);
    return;
  }
  std::fprintf(stderr, "%s: native package compilation failed with %s (status %d); log: %s\n",
               test_name, result.compiler.c_str(), result.status, result.log_path.c_str());
}

}  // namespace pops::test::native_dso
