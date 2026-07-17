# Deliberately incompatible MPI development contract for the installed-package rejection smoke.
# It is a CMake test fixture only: no executable or scientific code is linked against these files.
set(_pops_fake_mpi_root "${CMAKE_BINARY_DIR}/pops-fake-mpi")
file(MAKE_DIRECTORY "${_pops_fake_mpi_root}/include" "${_pops_fake_mpi_root}/lib")
file(WRITE "${_pops_fake_mpi_root}/include/mpi.h"
  "#pragma once\n#define POPS_FAKE_MPI_VENDOR 1\n")
file(WRITE "${_pops_fake_mpi_root}/lib/libmpi-mismatch.so" "not-a-real-mpi-library\n")

set(MPI_FOUND TRUE)
set(MPI_VERSION "99.0-mismatch")
set(MPIEXEC_EXECUTABLE "${CMAKE_COMMAND}")
foreach(_language C CXX)
  set(MPI_${_language}_FOUND TRUE)
  set(MPI_${_language}_COMPILER "${CMAKE_${_language}_COMPILER}")
  set(MPI_${_language}_VERSION "99.0-mismatch")
  set(MPI_${_language}_INCLUDE_DIRS "${_pops_fake_mpi_root}/include")
  set(MPI_${_language}_LIBRARIES "${_pops_fake_mpi_root}/lib/libmpi-mismatch.so")
  if(NOT TARGET MPI::MPI_${_language})
    add_library(MPI::MPI_${_language} UNKNOWN IMPORTED)
    set_target_properties(MPI::MPI_${_language} PROPERTIES
      IMPORTED_LOCATION "${_pops_fake_mpi_root}/lib/libmpi-mismatch.so"
      INTERFACE_INCLUDE_DIRECTORIES "${_pops_fake_mpi_root}/include")
  endif()
endforeach()
