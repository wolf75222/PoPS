cmake_minimum_required(VERSION 3.21)

foreach(_required POPS_BINARY_DIR POPS_CONSUMER_SOURCE_DIR POPS_SMOKE_ROOT
                  POPS_EXPECT_MPI POPS_EXPECT_HDF5 POPS_EXPECT_PARALLEL_HDF5
                  POPS_EXPECT_C_LOADED)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "installed-package smoke is missing -D${_required}=...")
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
    "PoPS package installation failed (${_install_result}):\n"
    "${_install_stdout}${_install_stderr}")
endif()

set(_configure_command
  "${CMAKE_COMMAND}"
  -S "${POPS_CONSUMER_SOURCE_DIR}"
  -B "${_consumer_build}"
  "-DCMAKE_PREFIX_PATH=${_prefix}\;${POPS_DEPENDENCY_PREFIX_PATH}"
  "-DPOPS_EXPECT_MPI=${POPS_EXPECT_MPI}"
  "-DPOPS_EXPECT_HDF5=${POPS_EXPECT_HDF5}"
  "-DPOPS_EXPECT_PARALLEL_HDF5=${POPS_EXPECT_PARALLEL_HDF5}"
  "-DPOPS_EXPECT_C_LOADED=${POPS_EXPECT_C_LOADED}")

if(DEFINED POPS_GENERATOR AND NOT POPS_GENERATOR STREQUAL "")
  list(APPEND _configure_command -G "${POPS_GENERATOR}")
endif()
foreach(_hint
    CMAKE_CXX_COMPILER CMAKE_C_COMPILER Kokkos_DIR MPI_CXX_COMPILER MPI_C_COMPILER HDF5_ROOT
    OpenMP_CXX_FLAGS OpenMP_omp_LIBRARY)
  if(DEFINED POPS_${_hint} AND NOT POPS_${_hint} STREQUAL "")
    list(APPEND _configure_command "-D${_hint}=${POPS_${_hint}}")
  endif()
endforeach()

execute_process(
  COMMAND ${_configure_command}
  RESULT_VARIABLE _configure_result
  OUTPUT_VARIABLE _configure_stdout
  ERROR_VARIABLE _configure_stderr)
if(NOT _configure_result EQUAL 0)
  message(FATAL_ERROR
    "installed PoPS consumer configure failed (${_configure_result}):\n"
    "${_configure_stdout}${_configure_stderr}")
endif()

set(_build_command "${CMAKE_COMMAND}" --build "${_consumer_build}")
if(DEFINED POPS_BUILD_CONFIG AND NOT POPS_BUILD_CONFIG STREQUAL "")
  list(APPEND _build_command --config "${POPS_BUILD_CONFIG}")
endif()
execute_process(
  COMMAND ${_build_command}
  RESULT_VARIABLE _build_result
  OUTPUT_VARIABLE _build_stdout
  ERROR_VARIABLE _build_stderr)
if(NOT _build_result EQUAL 0)
  message(FATAL_ERROR
    "installed PoPS consumer build failed (${_build_result}):\n"
    "${_build_stdout}${_build_stderr}")
endif()

set(_executable "${_consumer_build}/pops_installed_package_consumer${CMAKE_EXECUTABLE_SUFFIX}")
if(NOT EXISTS "${_executable}" AND DEFINED POPS_BUILD_CONFIG)
  set(_executable
    "${_consumer_build}/${POPS_BUILD_CONFIG}/pops_installed_package_consumer${CMAKE_EXECUTABLE_SUFFIX}")
endif()
if(NOT EXISTS "${_executable}")
  message(FATAL_ERROR "installed PoPS consumer executable was not produced")
endif()
execute_process(
  COMMAND "${_executable}"
  RESULT_VARIABLE _run_result
  OUTPUT_VARIABLE _run_stdout
  ERROR_VARIABLE _run_stderr)
if(NOT _run_result EQUAL 0)
  message(FATAL_ERROR
    "installed PoPS consumer failed at runtime (${_run_result}):\n"
    "${_run_stdout}${_run_stderr}")
endif()
