cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS
    POPS_MPIEXEC_EXECUTABLE
    POPS_MPIEXEC_NUMPROC_FLAG
    POPS_MPI_TEST_EXECUTABLE
    POPS_MPI_RANKS
    POPS_MPI_SIGNATURE_PREFIX)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "run_mpi_rank_parity.cmake requires ${_required}")
  endif()
endforeach()

if(NOT POPS_MPI_SIGNATURE_PREFIX MATCHES "^[A-Za-z0-9_.:-]+$")
  message(FATAL_ERROR "POPS_MPI_SIGNATURE_PREFIX contains unsupported characters")
endif()
list(LENGTH POPS_MPI_RANKS _rank_count)
if(_rank_count LESS 2)
  message(FATAL_ERROR "POPS_MPI_RANKS must contain at least two ranks")
endif()

set(_reference_signature)
set(_reference_rank)
set(_previous_rank 0)
foreach(_rank IN LISTS POPS_MPI_RANKS)
  if(NOT _rank MATCHES "^[1-9][0-9]*$" OR _rank LESS_EQUAL _previous_rank)
    message(FATAL_ERROR "POPS_MPI_RANKS must be unique positive integers in ascending order")
  endif()
  set(_previous_rank ${_rank})

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "POPS_TEST_EXPECT_RANKS=${_rank}"
      "OMP_NUM_THREADS=1"
      "${POPS_MPIEXEC_EXECUTABLE}" "${POPS_MPIEXEC_NUMPROC_FLAG}" "${_rank}"
      ${POPS_MPIEXEC_PREFLAGS}
      "${POPS_MPI_TEST_EXECUTABLE}"
      ${POPS_MPIEXEC_POSTFLAGS}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
    TIMEOUT 180)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "MPI rank-parity launch failed at np=${_rank} (exit=${_result})\n"
      "stdout:\n${_stdout}\nstderr:\n${_stderr}")
  endif()

  set(_output "${_stdout}\n${_stderr}")
  string(REPLACE "\r\n" "\n" _output "${_output}")
  string(REPLACE "\r" "\n" _output "${_output}")
  string(REGEX MATCHALL "${POPS_MPI_SIGNATURE_PREFIX}[^\n]*" _signatures "${_output}")
  list(LENGTH _signatures _signature_count)
  if(NOT _signature_count EQUAL 1)
    message(FATAL_ERROR
      "MPI rank-parity launch at np=${_rank} emitted ${_signature_count} canonical signatures; "
      "expected exactly one ${POPS_MPI_SIGNATURE_PREFIX} line\n"
      "stdout:\n${_stdout}\nstderr:\n${_stderr}")
  endif()
  list(GET _signatures 0 _signature)

  if(NOT DEFINED _reference_signature OR "${_reference_signature}" STREQUAL "")
    set(_reference_signature "${_signature}")
    set(_reference_rank ${_rank})
  elseif(NOT _signature STREQUAL _reference_signature)
    message(FATAL_ERROR
      "MPI rank-parity mismatch between np=${_reference_rank} and np=${_rank}\n"
      "np=${_reference_rank}: ${_reference_signature}\n"
      "np=${_rank}: ${_signature}")
  endif()
endforeach()

message(STATUS
  "MPI rank parity verified for ${POPS_MPI_TEST_EXECUTABLE} at ranks ${POPS_MPI_RANKS}: "
  "${_reference_signature}")
