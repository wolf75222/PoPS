# Compiled native runtime implementation.
#
# This directory is the single source authority for the System/AmrSystem object libraries used by
# both the Python extension and the C++ conformance tests.  Consumer CMake files link these targets;
# they never repeat the source manifest or regenerate a second seam product.

include(${CMAKE_CURRENT_SOURCE_DIR}/runtime/builders/seam_combinations.cmake)

set(POPS_GENERATED_SEAMS_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_seams)
set(POPS_RUNTIME_SYSTEM_GENERATED_SEAMS "")
set(POPS_RUNTIME_AMR_GENERATED_SEAMS "")
foreach(_seam_row IN LISTS POPS_SEAM_COMBINATIONS)
  string(REPLACE "|" ";" _seam_columns "${_seam_row}")
  list(GET _seam_columns 1 _seam_side)
  if(_seam_side STREQUAL "system")
    pops_generate_seam(
      "${_seam_row}" "${POPS_GENERATED_SEAMS_DIR}" POPS_RUNTIME_SYSTEM_GENERATED_SEAMS)
  else()
    pops_generate_seam(
      "${_seam_row}" "${POPS_GENERATED_SEAMS_DIR}" POPS_RUNTIME_AMR_GENERATED_SEAMS)
  endif()
endforeach()

set(POPS_RUNTIME_SYSTEM_SOURCES
    runtime/system/system.cpp
    runtime/system/system_install.cpp
    runtime/system/system_fields.cpp
    runtime/system/system_io.cpp
    runtime/system/system_profiling.cpp
    runtime/system/system_program.cpp
    runtime/system/system_polar.cpp
    ${POPS_RUNTIME_SYSTEM_GENERATED_SEAMS})

set(POPS_RUNTIME_AMR_SOURCES
    runtime/amr/amr_system.cpp
    runtime/builders/amr/block/compressible/amr_block_compressible.cpp
    runtime/builders/amr/compiled/compressible/amr_compiled_compressible.cpp
    ${POPS_RUNTIME_AMR_GENERATED_SEAMS})

set(POPS_RUNTIME_OUTPUT_SOURCES
    runtime/output/hdf5_collective.cpp)

add_library(pops_runtime_system OBJECT ${POPS_RUNTIME_SYSTEM_SOURCES})
add_library(pops_runtime_amr OBJECT ${POPS_RUNTIME_AMR_SOURCES})
add_library(pops_runtime_output OBJECT ${POPS_RUNTIME_OUTPUT_SOURCES})
set(POPS_RUNTIME_TARGETS pops_runtime_system pops_runtime_amr pops_runtime_output)

foreach(_runtime_target IN LISTS POPS_RUNTIME_TARGETS)
  target_link_libraries(${_runtime_target} PUBLIC pops::pops ${CMAKE_DL_LIBS})
  target_compile_features(${_runtime_target} PRIVATE cxx_std_${POPS_CXX_STD})
endforeach()

if(POPS_BUILD_TESTS)
  # Preserve the historical test-runtime compilation contract: the heavy implementation objects and
  # their consumers use -O0 in optimized configurations, and only heavy objects occupy this Ninja
  # pool. This test-only pool is intentionally independent from the production module's -O3 pool.
  set_property(GLOBAL APPEND PROPERTY
    JOB_POOLS pops_heavy_test_tu=${POPS_HEAVY_TEST_TU_POOL})
  foreach(_runtime_target IN LISTS POPS_RUNTIME_TARGETS)
    target_compile_options(${_runtime_target} PUBLIC
      $<$<AND:$<CONFIG:Release,RelWithDebInfo,MinSizeRel>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O0>)
    set_target_properties(${_runtime_target} PROPERTIES JOB_POOL_COMPILE pops_heavy_test_tu)
  endforeach()
elseif(POPS_BUILD_PYTHON)
  # A Python-only build compiles these production objects at -O3. When tests are enabled they stay
  # on the independent -O0 test pool above, even if the same objects also feed the Python module.
  foreach(_runtime_target IN LISTS POPS_RUNTIME_TARGETS)
    set_target_properties(${_runtime_target} PROPERTIES JOB_POOL_COMPILE pops_heavy_module_tu)
  endforeach()
endif()

if(POPS_BUILD_PYTHON)
  # These objects used to be direct sources of pybind11_add_module.  Preserve every compilation
  # property they inherited there; POPS_EXPORT remains the only public native symbol seam.
  foreach(_runtime_target IN LISTS POPS_RUNTIME_TARGETS)
    target_link_libraries(${_runtime_target} PRIVATE pops_dev_options)
    set_target_properties(${_runtime_target} PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CXX_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN ON)
  endforeach()

  pops_compute_header_signature(POPS_NATIVE_HEADER_SIGNATURE)

  foreach(_runtime_target IN LISTS POPS_RUNTIME_TARGETS)
    target_compile_definitions(${_runtime_target} PRIVATE
      # Mirrors the DEFINE_SYMBOL CMake adds when these files are direct _pops MODULE sources.
      _pops_EXPORTS
      POPS_CXX_STD=${POPS_CXX_STD}
      POPS_VERSION="${PROJECT_VERSION}"
      POPS_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
      POPS_HEADER_SIG="${POPS_NATIVE_HEADER_SIGNATURE}")
  endforeach()

  # Make the exact native facts available to the pybind-only adapter TUs in python/CMakeLists.txt.
  set(POPS_NATIVE_HEADER_SIGNATURE "${POPS_NATIVE_HEADER_SIGNATURE}" PARENT_SCOPE)
  set(POPS_NATIVE_MPI_INCLUDE "${POPS_NATIVE_MPI_INCLUDE}" PARENT_SCOPE)
  set(POPS_NATIVE_MPI_ABI "${POPS_NATIVE_MPI_ABI}" PARENT_SCOPE)
  set(POPS_NATIVE_MPI_COMPILER "${POPS_NATIVE_MPI_COMPILER}" PARENT_SCOPE)
  set(POPS_NATIVE_MPI_STANDARD "${POPS_NATIVE_MPI_STANDARD}" PARENT_SCOPE)
  set(POPS_NATIVE_MPI_COMPILE_OPTIONS "${POPS_NATIVE_MPI_COMPILE_OPTIONS}" PARENT_SCOPE)
  set(POPS_NATIVE_MPI_COMPILE_DEFINITIONS
      "${POPS_NATIVE_MPI_COMPILE_DEFINITIONS}" PARENT_SCOPE)
  set(POPS_NATIVE_MPI_LINK_OPTIONS "${POPS_NATIVE_MPI_LINK_OPTIONS}" PARENT_SCOPE)
  set(POPS_NATIVE_MPI_LINK_LIBRARIES "${POPS_NATIVE_MPI_LINK_LIBRARIES}" PARENT_SCOPE)
  set(POPS_NATIVE_MPI_HEADER_PATHS "${POPS_NATIVE_MPI_HEADER_PATHS}" PARENT_SCOPE)
  set(POPS_NATIVE_MPI_HEADER_HASHES "${POPS_NATIVE_MPI_HEADER_HASHES}" PARENT_SCOPE)
  set(POPS_NATIVE_MPI_LIBRARY_PATHS "${POPS_NATIVE_MPI_LIBRARY_PATHS}" PARENT_SCOPE)
  set(POPS_NATIVE_MPI_LIBRARY_HASHES "${POPS_NATIVE_MPI_LIBRARY_HASHES}" PARENT_SCOPE)
endif()

# Called after pybind11 has been discovered.  Keeping this operation here ensures that enabling
# POPS_PY_LTO applies the same ThinLTO compilation interface to the central runtime objects and the
# module adapter objects, just as when all sources belonged directly to pybind11_add_module.
function(pops_runtime_enable_python_thin_lto)
  if(NOT TARGET pybind11::thin_lto)
    message(FATAL_ERROR "POPS_PY_LTO requires the pybind11::thin_lto target")
  endif()
  foreach(_runtime_target pops_runtime_system pops_runtime_amr pops_runtime_output)
    target_link_libraries(${_runtime_target} PRIVATE pybind11::thin_lto)
  endforeach()
endfunction()
