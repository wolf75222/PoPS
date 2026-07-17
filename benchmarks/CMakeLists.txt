cmake_minimum_required(VERSION 3.21)

project(PoPSBenchmarks LANGUAGES C CXX)

get_filename_component(_pops_bench_default_source "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)
set(POPS_BENCH_POPS_SOURCE_DIR "${_pops_bench_default_source}" CACHE PATH
    "PoPS source tree used by the standalone benchmark harness")
option(POPS_BENCH_ENABLE_MPI "Build the benchmark harness with MPI" OFF)

if(NOT EXISTS "${POPS_BENCH_POPS_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
          "POPS_BENCH_POPS_SOURCE_DIR is not a PoPS source tree: ${POPS_BENCH_POPS_SOURCE_DIR}")
endif()

# Consume the real PoPS targets while keeping the standalone harness small and deterministic.
set(POPS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(POPS_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(POPS_INSTALL OFF CACHE BOOL "" FORCE)
set(POPS_USE_KOKKOS ON CACHE BOOL "" FORCE)
set(POPS_USE_MPI ${POPS_BENCH_ENABLE_MPI} CACHE BOOL "" FORCE)
set(POPS_USE_HDF5 OFF CACHE BOOL "" FORCE)
add_subdirectory("${POPS_BENCH_POPS_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/pops-core"
                 EXCLUDE_FROM_ALL)

find_package(Git QUIET)
set(POPS_BENCH_GIT_SHA "unknown")
set(POPS_BENCH_GIT_DIRTY 0)
if(Git_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${POPS_BENCH_POPS_SOURCE_DIR}" rev-parse HEAD
    RESULT_VARIABLE _pops_bench_git_sha_result
    OUTPUT_VARIABLE _pops_bench_git_sha
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_pops_bench_git_sha_result EQUAL 0)
    set(POPS_BENCH_GIT_SHA "${_pops_bench_git_sha}")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${POPS_BENCH_POPS_SOURCE_DIR}" status --porcelain
            --untracked-files=normal
    RESULT_VARIABLE _pops_bench_git_status_result
    OUTPUT_VARIABLE _pops_bench_git_status
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_pops_bench_git_status_result EQUAL 0 AND NOT _pops_bench_git_status STREQUAL "")
    set(POPS_BENCH_GIT_DIRTY 1)
  endif()
endif()

if(CMAKE_BUILD_TYPE)
  set(POPS_BENCH_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
else()
  set(POPS_BENCH_BUILD_TYPE "multi-config-or-unspecified")
endif()
set(POPS_BENCH_COMPILER "${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")

add_library(pops_benchmark_support STATIC
  src/benchmark_support.cpp)
target_compile_features(pops_benchmark_support PUBLIC cxx_std_20)
target_include_directories(pops_benchmark_support PUBLIC
  "${CMAKE_CURRENT_LIST_DIR}/include")
target_link_libraries(pops_benchmark_support PUBLIC pops::pops)
target_compile_definitions(pops_benchmark_support PRIVATE
  POPS_BENCH_GIT_SHA="${POPS_BENCH_GIT_SHA}"
  POPS_BENCH_GIT_DIRTY=${POPS_BENCH_GIT_DIRTY}
  POPS_BENCH_COMPILER="${POPS_BENCH_COMPILER}"
  POPS_BENCH_BUILD_TYPE="${POPS_BENCH_BUILD_TYPE}")

add_executable(pops_benchmark
  src/main.cpp
  src/arith_halo_case.cpp
  src/tensor_krylov_case.cpp)
target_compile_features(pops_benchmark PRIVATE cxx_std_20)
target_link_libraries(pops_benchmark PRIVATE pops_benchmark_support)

set_target_properties(pops_benchmark PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

message(STATUS "PoPS benchmark source: ${POPS_BENCH_POPS_SOURCE_DIR}")
message(STATUS "PoPS benchmark MPI: ${POPS_BENCH_ENABLE_MPI}")
message(STATUS "PoPS benchmark executable: ${CMAKE_BINARY_DIR}/bin/pops_benchmark")
