cmake_minimum_required(VERSION 3.21)

foreach(_required POPS_SOURCE_DIR POPS_FIXTURE_SOURCE_DIR POPS_KOKKOS_FIXTURE_DIR
                  POPS_TEST_BUILD_DIR POPS_NATIVE_DIM)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "Kokkos nested-replay rejection is missing -D${_required}=...")
  endif()
endforeach()

file(REMOVE_RECURSE "${POPS_TEST_BUILD_DIR}")
set(_command
  "${CMAKE_COMMAND}"
  -S "${POPS_FIXTURE_SOURCE_DIR}"
  -B "${POPS_TEST_BUILD_DIR}"
  "-DPOPS_SOURCE_DIR=${POPS_SOURCE_DIR}"
  "-DPOPS_KOKKOS_FIXTURE_DIR=${POPS_KOKKOS_FIXTURE_DIR}"
  "-DPOPS_KOKKOS_NESTED_OPTION_FIXTURE=ON"
  "-DPOPS_NATIVE_DIM=${POPS_NATIVE_DIM}")
if(DEFINED POPS_GENERATOR AND NOT POPS_GENERATOR STREQUAL "")
  list(APPEND _command -G "${POPS_GENERATOR}")
endif()
execute_process(
  COMMAND ${_command}
  RESULT_VARIABLE _result
  OUTPUT_VARIABLE _stdout
  ERROR_VARIABLE _stderr)
if(_result EQUAL 0)
  message(FATAL_ERROR "PoPS accepted a nested Kokkos CXX replay expression")
endif()
set(_diagnostic "${_stdout}${_stderr}")
if(NOT _diagnostic MATCHES "nested or malformed CXX expression")
  message(FATAL_ERROR
    "PoPS rejected the nested Kokkos expression for the wrong reason (${_result}):\n${_diagnostic}")
endif()
