cmake_minimum_required(VERSION 3.21)

project(PoPSAdc757Campaign LANGUAGES C CXX)

set(POPS_ADC757_REVISION "unknown" CACHE STRING "Resolved source revision recorded in evidence")
set(POPS_ADC757_WHEEL_SHA256 "" CACHE STRING "Exact installed wheel digest")
set(POPS_ADC757_MODULE_ABI_SHA256 "" CACHE STRING "Exact installed module ABI digest")
set(POPS_ADC757_INCLUDE_ROOT "" CACHE PATH "Authenticated include root from the installed wheel")

foreach(_digest POPS_ADC757_WHEEL_SHA256 POPS_ADC757_MODULE_ABI_SHA256)
  string(LENGTH "${${_digest}}" _digest_length)
  if(NOT _digest_length EQUAL 64 OR NOT "${${_digest}}" MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "${_digest} must be one lowercase sha256 digest")
  endif()
endforeach()

# Python wheels intentionally do not install popsConfig.cmake.  Consume their authenticated header
# payload directly instead of building a second PoPS copy from the archived source.  runtime_probe.py
# has already proved that this include tree has the signature baked into the installed _pops module.
if(NOT EXISTS "${POPS_ADC757_INCLUDE_ROOT}/pops/core/foundation/types.hpp")
  message(FATAL_ERROR
          "POPS_ADC757_INCLUDE_ROOT is not the authenticated installed PoPS header tree: "
          "${POPS_ADC757_INCLUDE_ROOT}")
endif()
find_package(Kokkos CONFIG REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
add_library(pops_adc757_installed INTERFACE)
target_include_directories(pops_adc757_installed INTERFACE "${POPS_ADC757_INCLUDE_ROOT}")
target_compile_features(pops_adc757_installed INTERFACE cxx_std_20)
target_compile_definitions(pops_adc757_installed INTERFACE POPS_HAS_KOKKOS POPS_HAS_MPI)
target_link_libraries(pops_adc757_installed INTERFACE Kokkos::kokkos MPI::MPI_CXX)

add_executable(adc757_heterogeneous_numerics heterogeneous_numerics.cpp)
target_compile_features(adc757_heterogeneous_numerics PRIVATE cxx_std_20)
target_link_libraries(adc757_heterogeneous_numerics PRIVATE pops_adc757_installed)
target_compile_definitions(adc757_heterogeneous_numerics PRIVATE
  POPS_ADC757_REVISION="${POPS_ADC757_REVISION}"
  POPS_ADC757_WHEEL_SHA256="${POPS_ADC757_WHEEL_SHA256}"
  POPS_ADC757_MODULE_ABI_SHA256="${POPS_ADC757_MODULE_ABI_SHA256}"
  POPS_ADC757_BUILD_ID="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}-${CMAKE_BUILD_TYPE}")

set_target_properties(adc757_heterogeneous_numerics PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
