# Deterministic identity of the native implementation shared by every Dim/MPI leaf.
#
# The public ABI key deliberately authenticates headers and per-leaf toolchain facts.  It does not
# change for an implementation-only edit in src/runtime or python/bindings, so it cannot decide
# whether an installed sibling dimension may survive a pip replacement.  This fingerprint closes
# that separate build-coherence boundary while leaving dimension, MPI and HDF5 as per-variant facts.

function(pops_compute_native_build_fingerprint output)
  get_filename_component(
    _pops_native_source_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
  foreach(_required
      POPS_NATIVE_HEADER_SIGNATURE POPS_NATIVE_KOKKOS_ABI POPS_REAL_TYPE POPS_CXX_STD
      POPS_HEADER_MANIFEST POPS_HEADER_ROOT POPS_INSTALLED_HEADERS)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
      message(FATAL_ERROR "native build fingerprint requires ${_required}")
    endif()
  endforeach()

  file(GLOB_RECURSE _pops_native_implementation_inputs
    CONFIGURE_DEPENDS
    RELATIVE "${_pops_native_source_root}"
    "${_pops_native_source_root}/src/runtime/*.c"
    "${_pops_native_source_root}/src/runtime/*.cc"
    "${_pops_native_source_root}/src/runtime/*.cpp"
    "${_pops_native_source_root}/src/runtime/*.cxx"
    "${_pops_native_source_root}/src/runtime/*.h"
    "${_pops_native_source_root}/src/runtime/*.hpp"
    "${_pops_native_source_root}/src/runtime/*.inc"
    "${_pops_native_source_root}/python/bindings/*.c"
    "${_pops_native_source_root}/python/bindings/*.cc"
    "${_pops_native_source_root}/python/bindings/*.cpp"
    "${_pops_native_source_root}/python/bindings/*.cxx"
    "${_pops_native_source_root}/python/bindings/*.h"
    "${_pops_native_source_root}/python/bindings/*.hpp"
    "${_pops_native_source_root}/python/bindings/*.inc")
  list(APPEND _pops_native_implementation_inputs
    "CMakeLists.txt"
    "src/CMakeLists.txt"
    "python/CMakeLists.txt"
    "cmake/PopsNativeBuildFingerprint.cmake"
    "schemas/component_catalog.v2.json")
  list(REMOVE_DUPLICATES _pops_native_implementation_inputs)
  list(SORT _pops_native_implementation_inputs)

  if(NOT _pops_native_implementation_inputs)
    message(FATAL_ERROR "native build fingerprint has no implementation inputs")
  endif()

  set(_pops_native_build_material "")
  set(_pops_native_absolute_inputs "")
  foreach(_relative IN LISTS _pops_native_implementation_inputs)
    set(_absolute "${_pops_native_source_root}/${_relative}")
    if(IS_SYMLINK "${_absolute}" OR NOT EXISTS "${_absolute}" OR IS_DIRECTORY "${_absolute}")
      message(FATAL_ERROR "native build fingerprint input is absent or not a regular file: ${_relative}")
    endif()
    file(SHA256 "${_absolute}" _digest)
    string(APPEND _pops_native_build_material "file=${_relative}\nsha256=${_digest}\n")
    list(APPEND _pops_native_absolute_inputs "${_absolute}")
  endforeach()

  # These are common specialization facts.  POPS_NATIVE_DIM, MPI/MPI_ABI and HDF5 are intentionally
  # absent so Dim1/2/3 serial leaves can coexist with one rebuilt MPI + parallel-HDF5 leaf.
  string(APPEND _pops_native_build_material
    "header_sig=${POPS_NATIVE_HEADER_SIGNATURE}\n"
    "kokkos_abi=${POPS_NATIVE_KOKKOS_ABI}\n"
    "real=${POPS_REAL_TYPE}\n"
    "cxx_std=${POPS_CXX_STD}\n")
  string(SHA256 _pops_native_build_fingerprint "${_pops_native_build_material}")

  # Reading file contents during configure is not itself a build-system dependency.  Register every
  # input explicitly so an incremental .cpp/.inc/header edit forces CMake to refresh the compiled
  # literal.  Public and Kokkos headers enter through their aggregate signatures above, but their
  # concrete files must still wake the configure step that recomputes those signatures.
  foreach(_header IN LISTS POPS_INSTALLED_HEADERS)
    list(APPEND _pops_native_absolute_inputs "${POPS_HEADER_ROOT}/${_header}")
  endforeach()
  list(APPEND _pops_native_absolute_inputs "${POPS_HEADER_MANIFEST}")
  list(APPEND _pops_native_absolute_inputs ${_pops_kokkos_header_paths})
  list(REMOVE_DUPLICATES _pops_native_absolute_inputs)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${_pops_native_absolute_inputs})
  set(${output} "${_pops_native_build_fingerprint}" PARENT_SCOPE)
endfunction()
