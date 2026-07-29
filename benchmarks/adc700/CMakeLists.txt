cmake_minimum_required(VERSION 3.21)

project(PoPSAdc700Campaign LANGUAGES C CXX)

set(POPS_ADC700_SOURCE_ROOT "" CACHE PATH "PoPS revision exercised by the ADC-700 campaign")
set(POPS_ADC700_ROUTE "" CACHE STRING "Temporal route: pre_cutover_native or program_only")
set(POPS_ADC700_REVISION "unknown" CACHE STRING "Resolved source revision recorded in evidence")

if(NOT EXISTS "${POPS_ADC700_SOURCE_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR
          "POPS_ADC700_SOURCE_ROOT is not a complete PoPS source tree: "
          "${POPS_ADC700_SOURCE_ROOT}")
endif()
if(NOT POPS_ADC700_ROUTE STREQUAL "pre_cutover_native"
   AND NOT POPS_ADC700_ROUTE STREQUAL "program_only")
  message(FATAL_ERROR
          "POPS_ADC700_ROUTE must be pre_cutover_native or program_only, got "
          "'${POPS_ADC700_ROUTE}'")
endif()

set(POPS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(POPS_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(POPS_INSTALL OFF CACHE BOOL "" FORCE)
set(POPS_USE_KOKKOS ON CACHE BOOL "" FORCE)
set(POPS_USE_MPI ON CACHE BOOL "" FORCE)
set(POPS_USE_HDF5 OFF CACHE BOOL "" FORCE)
add_subdirectory("${POPS_ADC700_SOURCE_ROOT}" "${CMAKE_BINARY_DIR}/pops-core"
                 EXCLUDE_FROM_ALL)

# The root intentionally omits native runtime objects when neither tests nor Python are enabled.
# Add the revision's canonical source manifest, never a campaign-owned copy of runtime sources.
set(POPS_CXX_STD 20)
add_subdirectory("${POPS_ADC700_SOURCE_ROOT}/src" "${CMAKE_BINARY_DIR}/pops-runtime"
                 EXCLUDE_FROM_ALL)

add_executable(adc700_program_cutover program_cutover.cpp)
target_compile_features(adc700_program_cutover PRIVATE cxx_std_20)
target_include_directories(adc700_program_cutover PRIVATE
  "${POPS_ADC700_SOURCE_ROOT}/tests/cpp/support")
target_link_libraries(adc700_program_cutover PRIVATE pops_runtime_amr)
target_compile_definitions(adc700_program_cutover PRIVATE
  POPS_ADC700_REVISION="${POPS_ADC700_REVISION}")

if(POPS_ADC700_ROUTE STREQUAL "pre_cutover_native")
  target_compile_definitions(adc700_program_cutover PRIVATE
    POPS_ADC700_PRE_CUTOVER=1
    POPS_ADC700_ROUTE_NAME="pre_cutover_native")
else()
  target_compile_definitions(adc700_program_cutover PRIVATE
    POPS_ADC700_PRE_CUTOVER=0
    POPS_ADC700_ROUTE_NAME="program_only")
endif()

set_target_properties(adc700_program_cutover PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
