cmake_minimum_required(VERSION 3.21)

foreach(_required POPS_BINARY_DIR POPS_CONSUMER_SOURCE_DIR POPS_FAKE_MPI_MODULE_DIR
                  POPS_SMOKE_ROOT POPS_DEPENDENCY_PREFIX_PATH)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "installed MPI mismatch smoke is missing -D${_required}=...")
  endif()
endforeach()

set(_prefix "${POPS_SMOKE_ROOT}/prefix")
set(_consumer_build "${POPS_SMOKE_ROOT}/consumer-build")
file(REMOVE_RECURSE "${POPS_SMOKE_ROOT}")
file(MAKE_DIRECTORY "${POPS_SMOKE_ROOT}")

set(_install_command "${CMAKE_COMMAND}" --install "${POPS_BINARY_DIR}" --prefix "${_prefix}")
if(DEFINED POPS_BUILD_CONFIG AND NOT POPS_BUILD_CONFIG STREQUAL "")
  list(APPEND _install_command --config "${POPS_BUILD_CONFIG}")
endif()
execute_process(
  COMMAND ${_install_command}
  RESULT_VARIABLE _install_result
  OUTPUT_VARIABLE _install_stdout
  ERROR_VARIABLE _install_stderr)
if(NOT _install_result EQUAL 0)
  message(FATAL_ERROR
    "PoPS mismatch-smoke installation failed (${_install_result}):\n"
    "${_install_stdout}${_install_stderr}")
endif()

set(_configure_command
  "${CMAKE_COMMAND}"
  -S "${POPS_CONSUMER_SOURCE_DIR}"
  -B "${_consumer_build}"
  "-DCMAKE_PREFIX_PATH=${_prefix}\;${POPS_DEPENDENCY_PREFIX_PATH}"
  "-DCMAKE_MODULE_PATH=${POPS_FAKE_MPI_MODULE_DIR}"
  -DPOPS_EXPECT_MPI=ON
  -DPOPS_EXPECT_HDF5=OFF
  -DPOPS_EXPECT_PARALLEL_HDF5=OFF
  -DPOPS_EXPECT_C_LOADED=OFF)
if(DEFINED POPS_GENERATOR AND NOT POPS_GENERATOR STREQUAL "")
  list(APPEND _configure_command -G "${POPS_GENERATOR}")
endif()
foreach(_hint CMAKE_CXX_COMPILER CMAKE_C_COMPILER)
  if(DEFINED POPS_${_hint} AND NOT POPS_${_hint} STREQUAL "")
    list(APPEND _configure_command "-D${_hint}=${POPS_${_hint}}")
  endif()
endforeach()
list(APPEND _configure_command "-DKokkos_DIR=${POPS_FAKE_MPI_MODULE_DIR}")

execute_process(
  COMMAND ${_configure_command}
  RESULT_VARIABLE _configure_result
  OUTPUT_VARIABLE _configure_stdout
  ERROR_VARIABLE _configure_stderr)
set(_configure_log "${_configure_stdout}${_configure_stderr}")
if(_configure_result EQUAL 0)
  message(FATAL_ERROR "installed PoPS accepted a deliberately incompatible MPI consumer")
endif()
if(NOT _configure_log MATCHES "PoPS MPI ABI mismatch")
  message(FATAL_ERROR
    "installed PoPS failed for the wrong reason; expected MPI ABI mismatch:\n${_configure_log}")
endif()
