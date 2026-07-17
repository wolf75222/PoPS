# Script-mode helper used by PopsMpiContract.cmake.  file(GET_RUNTIME_DEPENDENCIES)
# is deliberately executed outside project mode so CMake can inspect the native
# loader closure without loading either HDF5 or MPI into the configure process.

foreach(_pops_required
    POPS_HDF5_RUNTIME_FILES
    POPS_EXPECTED_MPI_RUNTIME_FILES
    POPS_RUNTIME_SCAN_OUTPUT)
  if(NOT DEFINED ${_pops_required} OR "${${_pops_required}}" STREQUAL "")
    message(FATAL_ERROR "Missing required runtime-scan input ${_pops_required}")
  endif()
endforeach()

string(REPLACE "|" ";" _pops_hdf5_files "${POPS_HDF5_RUNTIME_FILES}")
string(REPLACE "|" ";" _pops_expected_mpi_files "${POPS_EXPECTED_MPI_RUNTIME_FILES}")

set(_pops_search_directories "")
foreach(_pops_file IN LISTS _pops_hdf5_files _pops_expected_mpi_files)
  if(NOT IS_ABSOLUTE "${_pops_file}" OR NOT EXISTS "${_pops_file}")
    message(FATAL_ERROR "Runtime-scan input is not an existing absolute file: ${_pops_file}")
  endif()
  cmake_path(GET _pops_file PARENT_PATH _pops_parent)
  list(APPEND _pops_search_directories "${_pops_parent}")
endforeach()
list(REMOVE_DUPLICATES _pops_search_directories)

# Inspect only the concrete MPI runtime names selected by FindMPI.  Walking the
# complete HDF5 dependency closure is both unnecessary for this contract and
# unreliable on platforms whose system libraries live in a loader cache (for
# example macOS dyld cache stubs that ``otool`` cannot reopen).  A permissive
# basename regex is safe here because every resolved candidate is subsequently
# authenticated by canonical path and SHA-256.
set(_pops_expected_mpi_regexes "")
foreach(_pops_expected IN LISTS _pops_expected_mpi_files)
  get_filename_component(_pops_expected_name "${_pops_expected}" NAME)
  list(APPEND _pops_expected_mpi_regexes ".*${_pops_expected_name}$")
endforeach()
# Also inspect any additional library whose loader name advertises another MPI
# runtime.  Otherwise an HDF5 binary linked to both the expected MPI and a
# second vendor could satisfy the subset check below while still loading two
# incompatible runtimes into one process.
list(APPEND _pops_expected_mpi_regexes ".*(mpi|mpich|msmpi).*")
# Also collect any second MPI-family runtime.  Merely finding the expected one is insufficient if
# HDF5 has accidentally been linked to two MPI implementations in the same process.
list(APPEND _pops_expected_mpi_regexes
  "(^|[/\\\\])(lib)?(mpi|mpich|msmpi)[^/\\\\]*$")

file(GET_RUNTIME_DEPENDENCIES
  LIBRARIES ${_pops_hdf5_files}
  DIRECTORIES ${_pops_search_directories}
  PRE_INCLUDE_REGEXES ${_pops_expected_mpi_regexes}
  PRE_EXCLUDE_REGEXES ".*"
  RESOLVED_DEPENDENCIES_VAR _pops_resolved
  UNRESOLVED_DEPENDENCIES_VAR _pops_unresolved
  CONFLICTING_DEPENDENCIES_PREFIX _pops_conflicts)

if(_pops_unresolved)
  list(JOIN _pops_unresolved ", " _pops_unresolved_text)
  message(FATAL_ERROR
    "The parallel HDF5 runtime has unresolved native dependencies: ${_pops_unresolved_text}")
endif()
if(_pops_conflicts_FILENAMES)
  list(JOIN _pops_conflicts_FILENAMES ", " _pops_conflicts_text)
  message(FATAL_ERROR
    "The parallel HDF5 runtime has conflicting native dependencies: ${_pops_conflicts_text}")
endif()

set(_pops_resolved_identities "")
foreach(_pops_dependency IN LISTS _pops_resolved)
  file(REAL_PATH "${_pops_dependency}" _pops_dependency_real)
  file(SHA256 "${_pops_dependency_real}" _pops_dependency_sha256)
  list(APPEND _pops_resolved_identities
    "${_pops_dependency_real}::${_pops_dependency_sha256}")
endforeach()

set(_pops_expected_identities "")
foreach(_pops_expected IN LISTS _pops_expected_mpi_files)
  file(REAL_PATH "${_pops_expected}" _pops_expected_real)
  file(SHA256 "${_pops_expected_real}" _pops_expected_sha256)
  list(APPEND _pops_expected_identities
    "${_pops_expected_real}::${_pops_expected_sha256}")
endforeach()

set(_pops_unexpected_mpi "")
foreach(_pops_dependency IN LISTS _pops_resolved)
  file(REAL_PATH "${_pops_dependency}" _pops_dependency_real)
  file(SHA256 "${_pops_dependency_real}" _pops_dependency_sha256)
  if(NOT "${_pops_dependency_real}::${_pops_dependency_sha256}" IN_LIST
      _pops_expected_identities)
    list(APPEND _pops_unexpected_mpi "${_pops_dependency_real}")
  endif()
endforeach()
if(_pops_unexpected_mpi)
  list(JOIN _pops_unexpected_mpi ", " _pops_unexpected_mpi_text)
  message(FATAL_ERROR
    "Parallel HDF5 loads MPI-family runtimes outside the exact PoPS contract: "
    "${_pops_unexpected_mpi_text}")
endif()

set(_pops_missing_mpi "")
set(_pops_expected_identities "")
foreach(_pops_expected IN LISTS _pops_expected_mpi_files)
  file(REAL_PATH "${_pops_expected}" _pops_expected_real)
  file(SHA256 "${_pops_expected_real}" _pops_expected_sha256)
  set(_pops_expected_identity "${_pops_expected_real}::${_pops_expected_sha256}")
  list(APPEND _pops_expected_identities "${_pops_expected_identity}")
  if(NOT _pops_expected_identity IN_LIST _pops_resolved_identities)
    list(APPEND _pops_missing_mpi "${_pops_expected_real}")
  endif()
endforeach()

if(_pops_missing_mpi)
  list(JOIN _pops_missing_mpi ", " _pops_missing_mpi_text)
  message(FATAL_ERROR
    "Parallel HDF5 is not bound to the exact MPI runtime selected by PoPS: "
    "missing ${_pops_missing_mpi_text} from its binary dependency closure")
endif()

set(_pops_unexpected_mpi "")
foreach(_pops_dependency IN LISTS _pops_resolved)
  file(REAL_PATH "${_pops_dependency}" _pops_dependency_real)
  file(SHA256 "${_pops_dependency_real}" _pops_dependency_sha256)
  if(NOT "${_pops_dependency_real}::${_pops_dependency_sha256}" IN_LIST
      _pops_expected_identities)
    list(APPEND _pops_unexpected_mpi "${_pops_dependency_real}")
  endif()
endforeach()
if(_pops_unexpected_mpi)
  list(JOIN _pops_unexpected_mpi ", " _pops_unexpected_mpi_text)
  message(FATAL_ERROR
    "Parallel HDF5 loads an additional MPI runtime outside the authenticated PoPS contract: "
    "${_pops_unexpected_mpi_text}")
endif()

file(WRITE "${POPS_RUNTIME_SCAN_OUTPUT}" "ok\n")
