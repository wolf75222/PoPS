cmake_minimum_required(VERSION 3.21)

project(PoPSAdc757Campaign LANGUAGES C CXX)

set(POPS_ADC757_SOURCE_ROOT "" CACHE PATH "PoPS revision exercised by the ADC-757 campaign")
set(POPS_ADC757_REVISION "unknown" CACHE STRING "Resolved source revision recorded in evidence")

if(NOT EXISTS "${POPS_ADC757_SOURCE_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR
          "POPS_ADC757_SOURCE_ROOT is not a complete PoPS source tree: "
          "${POPS_ADC757_SOURCE_ROOT}")
endif()

set(POPS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(POPS_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(POPS_INSTALL OFF CACHE BOOL "" FORCE)
set(POPS_USE_KOKKOS ON CACHE BOOL "" FORCE)
set(POPS_USE_MPI ON CACHE BOOL "" FORCE)
set(POPS_USE_HDF5 OFF CACHE BOOL "" FORCE)
add_subdirectory("${POPS_ADC757_SOURCE_ROOT}" "${CMAKE_BINARY_DIR}/pops-core"
                 EXCLUDE_FROM_ALL)

add_executable(adc757_heterogeneous_numerics heterogeneous_numerics.cpp)
target_compile_features(adc757_heterogeneous_numerics PRIVATE cxx_std_20)
target_link_libraries(adc757_heterogeneous_numerics PRIVATE pops::pops)
target_compile_definitions(adc757_heterogeneous_numerics PRIVATE
  POPS_ADC757_REVISION="${POPS_ADC757_REVISION}"
  POPS_ADC757_BUILD_ID="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}-${CMAKE_BUILD_TYPE}")

set_target_properties(adc757_heterogeneous_numerics PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
