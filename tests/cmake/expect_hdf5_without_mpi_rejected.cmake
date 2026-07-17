cmake_minimum_required(VERSION 3.21)

foreach(_required POPS_SOURCE_DIR POPS_REJECT_BUILD_DIR)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "HDF5 fail-closed smoke is missing -D${_required}=...")
  endif()
endforeach()

file(REMOVE_RECURSE "${POPS_REJECT_BUILD_DIR}")
set(_configure_command
  "${CMAKE_COMMAND}"
  -S "${POPS_SOURCE_DIR}"
  -B "${POPS_REJECT_BUILD_DIR}"
  -DPOPS_BUILD_TESTS=OFF
  -DPOPS_BUILD_PYTHON=OFF
  -DPOPS_USE_MPI=OFF
  -DPOPS_USE_HDF5=ON)
if(DEFINED POPS_GENERATOR AND NOT POPS_GENERATOR STREQUAL "")
  list(APPEND _configure_command -G "${POPS_GENERATOR}")
endif()

execute_process(
  COMMAND ${_configure_command}
  RESULT_VARIABLE _configure_result
  OUTPUT_VARIABLE _configure_stdout
  ERROR_VARIABLE _configure_stderr)
if(_configure_result EQUAL 0)
  message(FATAL_ERROR "POPS_USE_HDF5=ON without MPI configured successfully")
endif()
set(_configure_log "${_configure_stdout}${_configure_stderr}")
if(NOT _configure_log MATCHES "POPS_USE_HDF5=ON enables the native collective HDF5 writer"
   OR NOT _configure_log MATCHES "requires POPS_USE_MPI=ON")
  message(FATAL_ERROR
    "HDF5-without-MPI failed for an unexpected reason:\n${_configure_log}")
endif()
