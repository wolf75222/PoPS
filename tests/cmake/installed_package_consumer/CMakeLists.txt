cmake_minimum_required(VERSION 3.21)

# Deliberately start as a CXX-only consumer.  A collective-HDF5 PoPS package must enable C through
# its exported config, while serial and MPI-only packages must leave this project CXX-only.
project(pops_installed_package_consumer LANGUAGES CXX)

foreach(_required POPS_EXPECT_MPI POPS_EXPECT_HDF5 POPS_EXPECT_PARALLEL_HDF5
                  POPS_EXPECT_C_LOADED)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "installed-package smoke requires -D${_required}=ON/OFF")
  endif()
endforeach()

find_package(pops CONFIG REQUIRED)

if(POPS_EXPECT_C_LOADED AND NOT CMAKE_C_COMPILER_LOADED)
  message(FATAL_ERROR "the collective-HDF5 package did not rehydrate its C toolchain")
elseif(NOT POPS_EXPECT_C_LOADED AND CMAKE_C_COMPILER_LOADED)
  message(FATAL_ERROR "a serial or MPI-only PoPS package unexpectedly enabled the C language")
endif()

get_target_property(_pops_definitions pops::pops INTERFACE_COMPILE_DEFINITIONS)
get_target_property(_pops_links pops::pops INTERFACE_LINK_LIBRARIES)

function(_pops_expect_list_member list_name member expected)
  list(FIND ${list_name} "${member}" _index)
  if(expected AND _index EQUAL -1)
    message(FATAL_ERROR "pops::pops is missing expected ${list_name} member ${member}")
  elseif(NOT expected AND NOT _index EQUAL -1)
    message(FATAL_ERROR "pops::pops unexpectedly exports ${list_name} member ${member}")
  endif()
endfunction()

_pops_expect_list_member(_pops_definitions POPS_HAS_MPI ${POPS_EXPECT_MPI})
_pops_expect_list_member(_pops_definitions POPS_HAS_HDF5 ${POPS_EXPECT_HDF5})
_pops_expect_list_member(
  _pops_definitions POPS_HAS_PARALLEL_HDF5 ${POPS_EXPECT_PARALLEL_HDF5})
_pops_expect_list_member(_pops_links MPI::MPI_CXX ${POPS_EXPECT_MPI})
_pops_expect_list_member(_pops_links HDF5::HDF5 ${POPS_EXPECT_HDF5})

add_executable(pops_installed_package_consumer main.cpp)
target_link_libraries(pops_installed_package_consumer PRIVATE pops::pops)
target_compile_features(pops_installed_package_consumer PRIVATE cxx_std_20)
