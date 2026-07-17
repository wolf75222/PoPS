# Compute the concrete MPI development ABI selected by FindMPI.
#
# PoPS exports this identity to every C++ consumer through pops::pops and replays the exact same
# authenticated contract when Python asks the native compiler to build a loader/component.  Keeping
# the computation at the top-level CMake boundary avoids a Python-build-only ABI guard.
function(pops_configure_mpi_contract target)
  set(_pops_outputs
    POPS_NATIVE_MPI_INCLUDE
    POPS_NATIVE_MPI_ABI
    POPS_NATIVE_MPI_COMPILER
    POPS_NATIVE_MPI_STANDARD
    POPS_NATIVE_MPI_COMPILE_OPTIONS
    POPS_NATIVE_MPI_COMPILE_DEFINITIONS
    POPS_NATIVE_MPI_LINK_OPTIONS
    POPS_NATIVE_MPI_LINK_LIBRARIES
    POPS_NATIVE_MPI_HEADER_PATHS
    POPS_NATIVE_MPI_HEADER_HASHES
    POPS_NATIVE_MPI_LIBRARY_PATHS
    POPS_NATIVE_MPI_LIBRARY_HASHES)
  foreach(_pops_output IN LISTS _pops_outputs)
    set(${_pops_output} "")
  endforeach()

  if(TARGET MPI::MPI_CXX)
    set(_pops_mpi_include_dirs ${MPI_CXX_INCLUDE_DIRS})
    if(NOT _pops_mpi_include_dirs)
      get_target_property(_pops_mpi_include_dirs MPI::MPI_CXX INTERFACE_INCLUDE_DIRECTORIES)
    endif()
    if(NOT _pops_mpi_include_dirs AND MPI_CXX_HEADER_DIR)
      set(_pops_mpi_include_dirs ${MPI_CXX_HEADER_DIR})
    endif()
    list(REMOVE_DUPLICATES _pops_mpi_include_dirs)
    if(NOT _pops_mpi_include_dirs)
      message(FATAL_ERROR
        "MPI::MPI_CXX exposes no include directory. Runtime-compiled PoPS components cannot "
        "reproduce the host MPI ABI without the directory containing mpi.h.")
    endif()

    set(_pops_mpi_compile_options ${MPI_CXX_COMPILE_OPTIONS})
    set(_pops_mpi_compile_definitions ${MPI_CXX_COMPILE_DEFINITIONS})
    set(_pops_mpi_link_options "")
    if(MPI_CXX_LINK_FLAGS)
      separate_arguments(_pops_mpi_link_options NATIVE_COMMAND "${MPI_CXX_LINK_FLAGS}")
    endif()
    set(_pops_mpi_libraries ${MPI_CXX_LIBRARIES})
    if(NOT _pops_mpi_libraries)
      get_target_property(_pops_mpi_libraries MPI::MPI_CXX INTERFACE_LINK_LIBRARIES)
    endif()
    if(NOT _pops_mpi_libraries)
      message(FATAL_ERROR
        "MPI::MPI_CXX exposes no link-library identity. PoPS cannot authenticate the MPI ABI "
        "used by runtime-compiled native components.")
    endif()

    set(_pops_mpi_abi_material
      "compiler=${MPI_CXX_COMPILER}\nstandard=${MPI_CXX_VERSION}\n")
    foreach(_pops_mpi_option IN LISTS _pops_mpi_compile_options)
      string(APPEND _pops_mpi_abi_material "compile_option=${_pops_mpi_option}\n")
    endforeach()
    foreach(_pops_mpi_definition IN LISTS _pops_mpi_compile_definitions)
      string(APPEND _pops_mpi_abi_material "compile_definition=${_pops_mpi_definition}\n")
    endforeach()
    foreach(_pops_mpi_option IN LISTS _pops_mpi_link_options)
      string(APPEND _pops_mpi_abi_material "link_option=${_pops_mpi_option}\n")
    endforeach()

    set(_pops_mpi_normalized_includes "")
    set(_pops_mpi_header_paths "")
    set(_pops_mpi_header_hashes "")
    set(_pops_mpi_header_count 0)
    foreach(_pops_mpi_include IN LISTS _pops_mpi_include_dirs)
      if(_pops_mpi_include MATCHES "^\\$<")
        message(FATAL_ERROR
          "MPI::MPI_CXX exposes generator-expression include directories (${_pops_mpi_include}); "
          "PoPS requires concrete mpi.h paths for runtime compilation.")
      endif()
      cmake_path(NORMAL_PATH _pops_mpi_include OUTPUT_VARIABLE _pops_mpi_include_normalized)
      if(NOT IS_ABSOLUTE "${_pops_mpi_include_normalized}" OR
         NOT IS_DIRECTORY "${_pops_mpi_include_normalized}")
        message(FATAL_ERROR
          "MPI::MPI_CXX include is not an existing absolute directory "
          "(${_pops_mpi_include_normalized}); runtime compilation cannot replay it.")
      endif()
      list(APPEND _pops_mpi_normalized_includes "${_pops_mpi_include_normalized}")
      string(APPEND _pops_mpi_abi_material "include=${_pops_mpi_include_normalized}\n")
      set(_pops_mpi_header "${_pops_mpi_include_normalized}/mpi.h")
      if(EXISTS "${_pops_mpi_header}")
        file(SHA256 "${_pops_mpi_header}" _pops_mpi_header_sha256)
        list(APPEND _pops_mpi_header_paths "${_pops_mpi_header}")
        list(APPEND _pops_mpi_header_hashes "${_pops_mpi_header_sha256}")
        string(APPEND _pops_mpi_abi_material
          "header=${_pops_mpi_header};sha256=${_pops_mpi_header_sha256}\n")
        math(EXPR _pops_mpi_header_count "${_pops_mpi_header_count} + 1")
      endif()
    endforeach()
    if(_pops_mpi_header_count EQUAL 0)
      message(FATAL_ERROR
        "MPI::MPI_CXX include directories contain no mpi.h. Runtime-compiled PoPS components "
        "would not be able to reproduce the host MPI ABI.")
    endif()

    set(_pops_mpi_normalized_libraries "")
    set(_pops_mpi_library_hashes "")
    foreach(_pops_mpi_library IN LISTS _pops_mpi_libraries)
      if(_pops_mpi_library MATCHES "^\\$<")
        message(FATAL_ERROR
          "MPI::MPI_CXX exposes a non-replayable generator-expression library "
          "(${_pops_mpi_library}); use a concrete FindMPI development package.")
      endif()
      cmake_path(NORMAL_PATH _pops_mpi_library OUTPUT_VARIABLE _pops_mpi_library_normalized)
      if(NOT IS_ABSOLUTE "${_pops_mpi_library_normalized}" OR
         NOT EXISTS "${_pops_mpi_library_normalized}")
        message(FATAL_ERROR
          "MPI::MPI_CXX library is not an existing absolute file "
          "(${_pops_mpi_library_normalized}); runtime compilation cannot replay/authenticate it.")
      endif()
      if(POPS_BUILD_PYTHON AND _pops_mpi_library_normalized MATCHES "\\.a$")
        message(FATAL_ERROR
          "MPI::MPI_CXX exposes static archive ${_pops_mpi_library_normalized}. The Python "
          "runtime compiles dynamic plugins, and relinking a static MPI archive into each plugin "
          "would create multiple MPI runtimes. Use the shared MPI development libraries.")
      endif()
      file(SHA256 "${_pops_mpi_library_normalized}" _pops_mpi_library_sha256)
      list(APPEND _pops_mpi_normalized_libraries "${_pops_mpi_library_normalized}")
      list(APPEND _pops_mpi_library_hashes "${_pops_mpi_library_sha256}")
      string(APPEND _pops_mpi_abi_material
        "library=${_pops_mpi_library_normalized};sha256=${_pops_mpi_library_sha256}\n")
    endforeach()
    string(SHA256 POPS_NATIVE_MPI_ABI "${_pops_mpi_abi_material}")

    foreach(_pops_mpi_list_name
        _pops_mpi_normalized_includes _pops_mpi_compile_options
        _pops_mpi_compile_definitions _pops_mpi_link_options
        _pops_mpi_normalized_libraries _pops_mpi_header_paths
        _pops_mpi_header_hashes _pops_mpi_library_hashes)
      foreach(_pops_mpi_item IN LISTS ${_pops_mpi_list_name})
        if(_pops_mpi_item MATCHES "\\$<")
          message(FATAL_ERROR
            "MPI::MPI_CXX contract item contains a generator expression that cannot be replayed "
            "by the runtime compiler: ${_pops_mpi_item}")
        endif()
        if(_pops_mpi_item MATCHES "[|;\r\n]")
          message(FATAL_ERROR
            "MPI::MPI_CXX contract item cannot be serialized exactly: ${_pops_mpi_item}")
        endif()
      endforeach()
    endforeach()
    foreach(_pops_mpi_item "${MPI_CXX_COMPILER}" "${MPI_CXX_VERSION}")
      if(_pops_mpi_item MATCHES "[|\r\n]")
        message(FATAL_ERROR
          "MPI compiler/version fact cannot be serialized exactly: ${_pops_mpi_item}")
      endif()
    endforeach()

    list(JOIN _pops_mpi_normalized_includes "|" POPS_NATIVE_MPI_INCLUDE)
    list(JOIN _pops_mpi_compile_options "|" POPS_NATIVE_MPI_COMPILE_OPTIONS)
    list(JOIN _pops_mpi_compile_definitions "|" POPS_NATIVE_MPI_COMPILE_DEFINITIONS)
    list(JOIN _pops_mpi_link_options "|" POPS_NATIVE_MPI_LINK_OPTIONS)
    list(JOIN _pops_mpi_normalized_libraries "|" POPS_NATIVE_MPI_LINK_LIBRARIES)
    list(JOIN _pops_mpi_header_paths "|" POPS_NATIVE_MPI_HEADER_PATHS)
    list(JOIN _pops_mpi_header_hashes "|" POPS_NATIVE_MPI_HEADER_HASHES)
    list(JOIN _pops_mpi_normalized_libraries "|" POPS_NATIVE_MPI_LIBRARY_PATHS)
    list(JOIN _pops_mpi_library_hashes "|" POPS_NATIVE_MPI_LIBRARY_HASHES)
    set(POPS_NATIVE_MPI_COMPILER "${MPI_CXX_COMPILER}")
    set(POPS_NATIVE_MPI_STANDARD "${MPI_CXX_VERSION}")
    foreach(_pops_mpi_serialized_name
        POPS_NATIVE_MPI_COMPILER POPS_NATIVE_MPI_STANDARD POPS_NATIVE_MPI_INCLUDE
        POPS_NATIVE_MPI_COMPILE_OPTIONS POPS_NATIVE_MPI_COMPILE_DEFINITIONS
        POPS_NATIVE_MPI_LINK_OPTIONS POPS_NATIVE_MPI_LINK_LIBRARIES
        POPS_NATIVE_MPI_HEADER_PATHS POPS_NATIVE_MPI_HEADER_HASHES
        POPS_NATIVE_MPI_LIBRARY_PATHS POPS_NATIVE_MPI_LIBRARY_HASHES)
      string(REPLACE "\\" "\\\\" ${_pops_mpi_serialized_name}
        "${${_pops_mpi_serialized_name}}")
      string(REPLACE "\"" "\\\"" ${_pops_mpi_serialized_name}
        "${${_pops_mpi_serialized_name}}")
    endforeach()

    if(NOT "${target}" STREQUAL "")
      if(NOT TARGET ${target})
        message(FATAL_ERROR "pops_configure_mpi_contract target does not exist: ${target}")
      endif()
      target_compile_definitions(${target} INTERFACE POPS_MPI_ABI="${POPS_NATIVE_MPI_ABI}")
    endif()
  endif()

  foreach(_pops_output IN LISTS _pops_outputs)
    set(${_pops_output} "${${_pops_output}}" PARENT_SCOPE)
  endforeach()
endfunction()

# Resolve concrete files from FindPackage variables and imported targets without evaluating
# arbitrary generator expressions.  The resulting contract is intentionally stricter than CMake's
# normal link-interface semantics: an ABI identity must be inspectable at configure time.
function(_pops_collect_native_link_files output)
  set(_pops_queue ${ARGN})
  set(_pops_seen_targets "")
  set(_pops_files "")
  while(_pops_queue)
    list(POP_FRONT _pops_queue _pops_item)
    if(_pops_item STREQUAL "" OR
       _pops_item STREQUAL "optimized" OR
       _pops_item STREQUAL "debug" OR
       _pops_item STREQUAL "general")
      continue()
    endif()
    if(_pops_item MATCHES "^\\$<LINK_ONLY:([^>]+)>$")
      set(_pops_item "${CMAKE_MATCH_1}")
    elseif(_pops_item MATCHES "\\$<")
      message(FATAL_ERROR
        "Native dependency contract contains a non-inspectable generator expression: ${_pops_item}")
    endif()

    if(TARGET "${_pops_item}")
      if(_pops_item IN_LIST _pops_seen_targets)
        continue()
      endif()
      list(APPEND _pops_seen_targets "${_pops_item}")
      get_target_property(_pops_imported "${_pops_item}" IMPORTED)
      if(_pops_imported)
        get_target_property(_pops_location "${_pops_item}" IMPORTED_LOCATION)
        if(_pops_location AND NOT _pops_location MATCHES "-NOTFOUND$")
          list(APPEND _pops_queue "${_pops_location}")
        endif()
        get_target_property(_pops_configs "${_pops_item}" IMPORTED_CONFIGURATIONS)
        foreach(_pops_config IN LISTS _pops_configs)
          string(TOUPPER "${_pops_config}" _pops_config_upper)
          get_target_property(_pops_location "${_pops_item}"
            "IMPORTED_LOCATION_${_pops_config_upper}")
          if(_pops_location AND NOT _pops_location MATCHES "-NOTFOUND$")
            list(APPEND _pops_queue "${_pops_location}")
          endif()
        endforeach()
      endif()
      get_target_property(_pops_links "${_pops_item}" INTERFACE_LINK_LIBRARIES)
      if(_pops_links AND NOT _pops_links MATCHES "-NOTFOUND$")
        list(APPEND _pops_queue ${_pops_links})
      endif()
    elseif(IS_ABSOLUTE "${_pops_item}" AND EXISTS "${_pops_item}")
      list(APPEND _pops_files "${_pops_item}")
    endif()
  endwhile()
  list(REMOVE_DUPLICATES _pops_files)
  set(${output} "${_pops_files}" PARENT_SCOPE)
endfunction()

function(pops_configure_parallel_hdf5_mpi_contract output)
  if(NOT POPS_NATIVE_MPI_ABI)
    message(FATAL_ERROR
      "Parallel HDF5 identity requires the authenticated PoPS MPI C++ contract first")
  endif()
  if(NOT TARGET MPI::MPI_C OR NOT TARGET MPI::MPI_CXX)
    message(FATAL_ERROR "Parallel HDF5 requires both MPI::MPI_C and MPI::MPI_CXX")
  endif()
  if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR
      "PoPS cannot prove the native HDF5/MPI runtime pair while cross-compiling. "
      "Provide a natively built package; an unexecuted or guessed MPI contract is not accepted.")
  endif()

  # MPI_C and MPI_CXX must share the same concrete mpi.h and at least one identical core runtime.
  set(_pops_mpi_c_includes ${MPI_C_INCLUDE_DIRS})
  if(NOT _pops_mpi_c_includes)
    get_target_property(_pops_mpi_c_includes MPI::MPI_C INTERFACE_INCLUDE_DIRECTORIES)
  endif()
  set(_pops_mpi_c_header_hashes "")
  foreach(_pops_include IN LISTS _pops_mpi_c_includes)
    if(_pops_include MATCHES "\\$<" OR NOT IS_ABSOLUTE "${_pops_include}")
      message(FATAL_ERROR "MPI_C exposes a non-concrete include directory: ${_pops_include}")
    endif()
    if(EXISTS "${_pops_include}/mpi.h")
      file(SHA256 "${_pops_include}/mpi.h" _pops_hash)
      list(APPEND _pops_mpi_c_header_hashes "${_pops_hash}")
    endif()
  endforeach()
  string(REPLACE "|" ";" _pops_mpi_cxx_header_hashes "${POPS_NATIVE_MPI_HEADER_HASHES}")
  set(_pops_common_header FALSE)
  foreach(_pops_hash IN LISTS _pops_mpi_c_header_hashes)
    if(_pops_hash IN_LIST _pops_mpi_cxx_header_hashes)
      set(_pops_common_header TRUE)
    endif()
  endforeach()
  if(NOT _pops_common_header)
    message(FATAL_ERROR
      "MPI_C and MPI_CXX do not expose the same mpi.h bytes; mixing their ABIs is unsupported")
  endif()

  _pops_collect_native_link_files(_pops_mpi_c_files ${MPI_C_LIBRARIES} MPI::MPI_C)
  string(REPLACE "|" ";" _pops_mpi_cxx_files "${POPS_NATIVE_MPI_LIBRARY_PATHS}")
  set(_pops_mpi_cxx_identities "")
  foreach(_pops_file IN LISTS _pops_mpi_cxx_files)
    file(REAL_PATH "${_pops_file}" _pops_real)
    file(SHA256 "${_pops_real}" _pops_hash)
    list(APPEND _pops_mpi_cxx_identities "${_pops_real}::${_pops_hash}")
  endforeach()
  set(_pops_common_mpi_runtimes "")
  foreach(_pops_file IN LISTS _pops_mpi_c_files)
    get_filename_component(_pops_name "${_pops_file}" NAME)
    string(TOLOWER "${_pops_name}" _pops_name_lower)
    if(NOT _pops_name_lower MATCHES "(mpi|mpich|msmpi)")
      continue()
    endif()
    if(_pops_name_lower MATCHES "\\.a$")
      message(FATAL_ERROR
        "Static MPI runtime ${_pops_file} cannot provide one process-wide HDF5/MPI identity")
    endif()
    file(REAL_PATH "${_pops_file}" _pops_real)
    file(SHA256 "${_pops_real}" _pops_hash)
    if("${_pops_real}::${_pops_hash}" IN_LIST _pops_mpi_cxx_identities)
      list(APPEND _pops_common_mpi_runtimes "${_pops_real}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _pops_common_mpi_runtimes)
  if(NOT _pops_common_mpi_runtimes)
    message(FATAL_ERROR
      "MPI_C and MPI_CXX do not resolve to an identical native MPI runtime (path and SHA256)")
  endif()

  set(_pops_hdf5_candidates ${HDF5_C_LIBRARIES})
  if(TARGET HDF5::HDF5)
    list(APPEND _pops_hdf5_candidates HDF5::HDF5)
  endif()
  if(TARGET hdf5::hdf5)
    list(APPEND _pops_hdf5_candidates hdf5::hdf5)
  endif()
  _pops_collect_native_link_files(_pops_hdf5_link_files ${_pops_hdf5_candidates})
  set(_pops_hdf5_runtime_files "")
  foreach(_pops_file IN LISTS _pops_hdf5_link_files)
    get_filename_component(_pops_name "${_pops_file}" NAME)
    string(TOLOWER "${_pops_name}" _pops_name_lower)
    if(NOT _pops_name_lower MATCHES "hdf5")
      continue()
    endif()
    if(_pops_name_lower MATCHES "\\.a$" OR _pops_name_lower MATCHES "\\.lib$")
      message(FATAL_ERROR
        "PoPS cannot prove the process-wide MPI identity of static/import HDF5 artifact "
        "${_pops_file}. Use the shared parallel HDF5 C runtime for this package.")
    endif()
    file(REAL_PATH "${_pops_file}" _pops_real)
    list(APPEND _pops_hdf5_runtime_files "${_pops_real}")
  endforeach()
  list(REMOVE_DUPLICATES _pops_hdf5_runtime_files)
  if(NOT _pops_hdf5_runtime_files)
    message(FATAL_ERROR
      "Parallel HDF5 exposes no concrete shared C runtime that PoPS can authenticate")
  endif()

  list(JOIN _pops_hdf5_runtime_files "|" _pops_hdf5_joined)
  list(JOIN _pops_common_mpi_runtimes "|" _pops_mpi_joined)
  set(_pops_scan_output "${CMAKE_BINARY_DIR}/CMakeFiles/pops-hdf5-mpi-runtime-scan.txt")
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DPOPS_HDF5_RUNTIME_FILES=${_pops_hdf5_joined}"
      "-DPOPS_EXPECTED_MPI_RUNTIME_FILES=${_pops_mpi_joined}"
      "-DPOPS_RUNTIME_SCAN_OUTPUT=${_pops_scan_output}"
      -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/PopsRuntimeDependencyScan.cmake"
    RESULT_VARIABLE _pops_scan_result
    OUTPUT_VARIABLE _pops_scan_stdout
    ERROR_VARIABLE _pops_scan_stderr)
  if(NOT _pops_scan_result EQUAL 0 OR NOT EXISTS "${_pops_scan_output}")
    message(FATAL_ERROR
      "Parallel HDF5/MPI binary identity proof failed.\n${_pops_scan_stdout}${_pops_scan_stderr}")
  endif()

  set(_pops_pair_material "mpi=${POPS_NATIVE_MPI_ABI}\n")
  foreach(_pops_file IN LISTS _pops_hdf5_runtime_files)
    file(SHA256 "${_pops_file}" _pops_hash)
    string(APPEND _pops_pair_material "hdf5_library=${_pops_file};sha256=${_pops_hash}\n")
  endforeach()
  set(_pops_hdf5_includes ${HDF5_C_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
  if(TARGET HDF5::HDF5)
    get_target_property(_pops_target_includes HDF5::HDF5 INTERFACE_INCLUDE_DIRECTORIES)
    if(_pops_target_includes AND NOT _pops_target_includes MATCHES "-NOTFOUND$")
      list(APPEND _pops_hdf5_includes ${_pops_target_includes})
    endif()
  endif()
  list(REMOVE_DUPLICATES _pops_hdf5_includes)
  set(_pops_hdf5_header_count 0)
  foreach(_pops_include IN LISTS _pops_hdf5_includes)
    if(_pops_include MATCHES "\\$<" OR NOT IS_ABSOLUTE "${_pops_include}")
      message(FATAL_ERROR "HDF5 exposes a non-concrete include directory: ${_pops_include}")
    endif()
    foreach(_pops_header hdf5.h H5pubconf.h)
      if(EXISTS "${_pops_include}/${_pops_header}")
        file(REAL_PATH "${_pops_include}/${_pops_header}" _pops_header_real)
        file(SHA256 "${_pops_header_real}" _pops_hash)
        string(APPEND _pops_pair_material
          "hdf5_header=${_pops_header_real};sha256=${_pops_hash}\n")
        math(EXPR _pops_hdf5_header_count "${_pops_hdf5_header_count} + 1")
      endif()
    endforeach()
  endforeach()
  if(_pops_hdf5_header_count EQUAL 0)
    message(FATAL_ERROR "Parallel HDF5 exposes no concrete C headers to authenticate")
  endif()
  string(SHA256 _pops_hdf5_mpi_abi "${_pops_pair_material}")
  set(${output} "${_pops_hdf5_mpi_abi}" PARENT_SCOPE)
endfunction()
