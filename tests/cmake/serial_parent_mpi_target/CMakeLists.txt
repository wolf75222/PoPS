cmake_minimum_required(VERSION 3.21)
project(pops_serial_parent_mpi_target LANGUAGES CXX)

foreach(_required POPS_SOURCE_DIR POPS_KOKKOS_FIXTURE_DIR)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "serial parent-MPI fixture is missing -D${_required}=...")
  endif()
endforeach()

# Model a parent application which already configured an unrelated static MPI. PoPS is added as a
# serial subproject and must neither inspect this target nor inherit/export its ABI identity.
set(_fake_mpi_root "${CMAKE_BINARY_DIR}/parent-static-mpi")
file(MAKE_DIRECTORY "${_fake_mpi_root}/include" "${_fake_mpi_root}/lib")
file(WRITE "${_fake_mpi_root}/include/mpi.h" "#pragma once\n#define PARENT_MPI_ONLY 1\n")
file(WRITE "${_fake_mpi_root}/lib/libmpi-parent.a" "!<arch>\n")
add_library(MPI::MPI_CXX UNKNOWN IMPORTED GLOBAL)
set_target_properties(MPI::MPI_CXX PROPERTIES
  IMPORTED_LOCATION "${_fake_mpi_root}/lib/libmpi-parent.a"
  INTERFACE_INCLUDE_DIRECTORIES "${_fake_mpi_root}/include")
set(MPI_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
set(MPI_CXX_VERSION "parent-static")
set(MPI_CXX_INCLUDE_DIRS "${_fake_mpi_root}/include")
set(MPI_CXX_LIBRARIES "${_fake_mpi_root}/lib/libmpi-parent.a")

set(POPS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(POPS_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(POPS_USE_MPI OFF CACHE BOOL "" FORCE)
set(POPS_USE_HDF5 OFF CACHE BOOL "" FORCE)
set(POPS_INSTALL OFF CACHE BOOL "" FORCE)
set(Kokkos_DIR "${POPS_KOKKOS_FIXTURE_DIR}" CACHE PATH "" FORCE)
add_subdirectory("${POPS_SOURCE_DIR}" pops-subbuild)

get_target_property(_definitions pops INTERFACE_COMPILE_DEFINITIONS)
foreach(_definition IN LISTS _definitions)
  if(_definition STREQUAL "POPS_HAS_MPI" OR _definition MATCHES "^POPS_MPI_ABI=")
    message(FATAL_ERROR
      "serial add_subdirectory PoPS leaked parent MPI definition: ${_definition}")
  endif()
endforeach()
