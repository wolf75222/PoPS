foreach(_pops_required
    POPS_FIXTURE_SOURCE_DIR
    POPS_TEST_BUILD_DIR
    POPS_RUNTIME_SCAN_SCRIPT
    POPS_GENERATOR)
  if(NOT DEFINED ${_pops_required})
    message(FATAL_ERROR "Missing required test input ${_pops_required}")
  endif()
endforeach()

file(REMOVE_RECURSE "${POPS_TEST_BUILD_DIR}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${POPS_FIXTURE_SOURCE_DIR}" -B "${POPS_TEST_BUILD_DIR}"
    -G "${POPS_GENERATOR}"
  RESULT_VARIABLE _pops_configure_result
  OUTPUT_VARIABLE _pops_configure_stdout
  ERROR_VARIABLE _pops_configure_stderr)
if(NOT _pops_configure_result EQUAL 0)
  message(FATAL_ERROR
    "HDF5/MPI binary fixture configure failed:\n${_pops_configure_stdout}${_pops_configure_stderr}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${POPS_TEST_BUILD_DIR}"
  RESULT_VARIABLE _pops_build_result
  OUTPUT_VARIABLE _pops_build_stdout
  ERROR_VARIABLE _pops_build_stderr)
if(NOT _pops_build_result EQUAL 0)
  message(FATAL_ERROR
    "HDF5/MPI binary fixture build failed:\n${_pops_build_stdout}${_pops_build_stderr}")
endif()

include("${POPS_TEST_BUILD_DIR}/artifacts.cmake")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DPOPS_HDF5_RUNTIME_FILES=${POPS_FIXTURE_HDF5}"
    "-DPOPS_EXPECTED_MPI_RUNTIME_FILES=${POPS_FIXTURE_MPI_OTHER}"
    "-DPOPS_RUNTIME_SCAN_OUTPUT=${POPS_TEST_BUILD_DIR}/matching.txt"
    -P "${POPS_RUNTIME_SCAN_SCRIPT}"
  RESULT_VARIABLE _pops_match_result
  OUTPUT_VARIABLE _pops_match_stdout
  ERROR_VARIABLE _pops_match_stderr)
if(NOT _pops_match_result EQUAL 0)
  message(FATAL_ERROR
    "Matching HDF5/MPI binary pair was rejected:\n${_pops_match_stdout}${_pops_match_stderr}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DPOPS_HDF5_RUNTIME_FILES=${POPS_FIXTURE_HDF5}"
    "-DPOPS_EXPECTED_MPI_RUNTIME_FILES=${POPS_FIXTURE_MPI_EXPECTED}"
    "-DPOPS_RUNTIME_SCAN_OUTPUT=${POPS_TEST_BUILD_DIR}/mismatching.txt"
    -P "${POPS_RUNTIME_SCAN_SCRIPT}"
  RESULT_VARIABLE _pops_mismatch_result
  OUTPUT_VARIABLE _pops_mismatch_stdout
  ERROR_VARIABLE _pops_mismatch_stderr)
if(_pops_mismatch_result EQUAL 0)
  message(FATAL_ERROR "Mismatching HDF5/MPI binary pair was silently accepted")
endif()
set(_pops_mismatch_log "${_pops_mismatch_stdout}${_pops_mismatch_stderr}")
if(NOT _pops_mismatch_log MATCHES
   "(not bound to the exact MPI runtime|outside the exact PoPS contract)")
  message(FATAL_ERROR "Mismatch failed for the wrong reason:\n${_pops_mismatch_log}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DPOPS_HDF5_RUNTIME_FILES=${POPS_FIXTURE_HDF5_DUAL}"
    "-DPOPS_EXPECTED_MPI_RUNTIME_FILES=${POPS_FIXTURE_MPI_OTHER}"
    "-DPOPS_RUNTIME_SCAN_OUTPUT=${POPS_TEST_BUILD_DIR}/dual.txt"
    -P "${POPS_RUNTIME_SCAN_SCRIPT}"
  RESULT_VARIABLE _pops_dual_result
  OUTPUT_VARIABLE _pops_dual_stdout
  ERROR_VARIABLE _pops_dual_stderr)
if(_pops_dual_result EQUAL 0)
  message(FATAL_ERROR "HDF5 linked to two MPI-family runtimes was silently accepted")
endif()
set(_pops_dual_log "${_pops_dual_stdout}${_pops_dual_stderr}")
if(NOT _pops_dual_log MATCHES "outside the exact PoPS contract")
  message(FATAL_ERROR "Dual-MPI fixture failed for the wrong reason:\n${_pops_dual_log}")
endif()
