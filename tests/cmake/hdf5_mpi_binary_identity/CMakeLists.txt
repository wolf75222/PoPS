cmake_minimum_required(VERSION 3.21)
project(pops_hdf5_mpi_binary_identity_fixture LANGUAGES C)

add_library(mpi_expected SHARED mpi_expected.c)
add_library(mpi_other SHARED mpi_other.c)
add_library(hdf5_fixture SHARED hdf5_fixture.c)
target_link_libraries(hdf5_fixture PRIVATE mpi_other)
add_library(hdf5_dual_fixture SHARED hdf5_dual_fixture.c)
target_link_libraries(hdf5_dual_fixture PRIVATE mpi_other mpi_expected)

string(CONCAT _pops_artifacts
  "set(POPS_FIXTURE_HDF5 \"$<TARGET_FILE:hdf5_fixture>\")\n"
  "set(POPS_FIXTURE_HDF5_DUAL \"$<TARGET_FILE:hdf5_dual_fixture>\")\n"
  "set(POPS_FIXTURE_MPI_EXPECTED \"$<TARGET_FILE:mpi_expected>\")\n"
  "set(POPS_FIXTURE_MPI_OTHER \"$<TARGET_FILE:mpi_other>\")\n")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/artifacts.cmake" CONTENT "${_pops_artifacts}")
