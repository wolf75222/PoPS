# One authoritative classification drives source installs, wheel installs, and the native ABI
# signature. Keep this parser language-neutral: the Python packaging preflight and runtime
# toolchain consume the same manifest and enforce the same closed category set.
set(POPS_HEADER_MANIFEST "${CMAKE_CURRENT_LIST_DIR}/../include/pops_headers.manifest")
set(POPS_HEADER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../include")
file(STRINGS "${POPS_HEADER_MANIFEST}" _pops_header_manifest_rows)

set(POPS_API_HEADERS "")
set(POPS_ABI_HEADERS "")
set(POPS_SDK_ROOT_HEADERS "")
set(POPS_SDK_SUPPORT_HEADERS "")
set(POPS_TEST_ONLY_HEADERS "")
set(POPS_INSTALLED_HEADERS "")
set(POPS_INSTALLED_HEADER_ROWS "")
set(POPS_ALL_CLASSIFIED_HEADERS "")

foreach(_pops_header_row IN LISTS _pops_header_manifest_rows)
  if(_pops_header_row MATCHES "^[ \t]*(#|$)")
    continue()
  endif()
  if(NOT _pops_header_row MATCHES
      "^(api|abi|sdk-root|sdk-support|test-only) (pops/.+\\.(h|hpp|inc))$")
    message(FATAL_ERROR
      "invalid row in ${POPS_HEADER_MANIFEST}: ${_pops_header_row}")
  endif()
  set(_pops_header_category "${CMAKE_MATCH_1}")
  set(_pops_header "${CMAKE_MATCH_2}")

  list(FIND POPS_ALL_CLASSIFIED_HEADERS "${_pops_header}" _pops_duplicate_index)
  if(NOT _pops_duplicate_index EQUAL -1)
    message(FATAL_ERROR
      "duplicate header in ${POPS_HEADER_MANIFEST}: ${_pops_header}")
  endif()
  list(APPEND POPS_ALL_CLASSIFIED_HEADERS "${_pops_header}")

  if(_pops_header_category STREQUAL "api")
    list(APPEND POPS_API_HEADERS "${_pops_header}")
  elseif(_pops_header_category STREQUAL "abi")
    list(APPEND POPS_ABI_HEADERS "${_pops_header}")
  elseif(_pops_header_category STREQUAL "sdk-root")
    list(APPEND POPS_SDK_ROOT_HEADERS "${_pops_header}")
  elseif(_pops_header_category STREQUAL "sdk-support")
    list(APPEND POPS_SDK_SUPPORT_HEADERS "${_pops_header}")
  else()
    list(APPEND POPS_TEST_ONLY_HEADERS "${_pops_header}")
  endif()

  if(NOT _pops_header_category STREQUAL "test-only")
    list(APPEND POPS_INSTALLED_HEADERS "${_pops_header}")
    # Signing includes the normalized category as well as the path/content. Reclassifying a header
    # therefore changes the ABI identity even when the bytes happen to be unchanged.
    list(APPEND POPS_INSTALLED_HEADER_ROWS
         "${_pops_header_category}|${_pops_header}")
  endif()
endforeach()

foreach(_pops_header_list
    POPS_API_HEADERS POPS_ABI_HEADERS POPS_SDK_ROOT_HEADERS POPS_SDK_SUPPORT_HEADERS
    POPS_TEST_ONLY_HEADERS POPS_INSTALLED_HEADERS POPS_INSTALLED_HEADER_ROWS
    POPS_ALL_CLASSIFIED_HEADERS)
  list(SORT ${_pops_header_list})
endforeach()

foreach(_pops_required_category API ABI SDK_ROOT SDK_SUPPORT)
  if(NOT POPS_${_pops_required_category}_HEADERS)
    message(FATAL_ERROR
      "${POPS_HEADER_MANIFEST} declares no ${_pops_required_category} headers")
  endif()
endforeach()

function(pops_compute_header_signature output)
  set(_pops_header_blob "")
  foreach(_pops_header_row IN LISTS POPS_INSTALLED_HEADER_ROWS)
    string(REPLACE "|" ";" _pops_header_columns "${_pops_header_row}")
    list(GET _pops_header_columns 0 _pops_header_category)
    list(GET _pops_header_columns 1 _pops_header)
    file(SHA256 "${POPS_HEADER_ROOT}/${_pops_header}" _pops_header_digest)
    string(APPEND _pops_header_blob
           "${_pops_header_category} ${_pops_header}\n${_pops_header_digest}\n")
  endforeach()
  string(SHA256 _pops_header_signature "${_pops_header_blob}")
  set(${output} "${_pops_header_signature}" PARENT_SCOPE)
endfunction()

function(pops_install_headers destination)
  # sdk-support is deliberately not a standalone surface, but it is installed because the api,
  # abi and sdk-root closures require it. Every installed file also enters POPS_HEADER_SIG.
  foreach(_pops_header IN LISTS POPS_INSTALLED_HEADERS)
    get_filename_component(_pops_header_dir "${_pops_header}" DIRECTORY)
    install(FILES "${POPS_HEADER_ROOT}/${_pops_header}"
            DESTINATION "${destination}/${_pops_header_dir}")
  endforeach()
  install(FILES "${POPS_HEADER_MANIFEST}" DESTINATION "${destination}")
endfunction()
