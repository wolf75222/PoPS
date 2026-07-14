# One authoritative classification drives source installs, wheel installs, and the native ABI
# signature.  Keep the data language-neutral: the Python packaging preflight reads the same file.
set(POPS_PUBLIC_HEADER_MANIFEST
    "${CMAKE_CURRENT_LIST_DIR}/../include/pops_public_headers.manifest")
set(POPS_PUBLIC_HEADER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../include")
file(STRINGS "${POPS_PUBLIC_HEADER_MANIFEST}" _pops_header_manifest_rows)

set(POPS_PUBLIC_HEADERS "")
set(POPS_TEST_ONLY_HEADERS "")
foreach(_pops_header_row IN LISTS _pops_header_manifest_rows)
  if(_pops_header_row MATCHES "^[ \t]*(#|$)")
    continue()
  elseif(_pops_header_row MATCHES "^public (pops/.+\\.(h|hpp))$")
    list(APPEND POPS_PUBLIC_HEADERS "${CMAKE_MATCH_1}")
  elseif(_pops_header_row MATCHES "^test-only (pops/.+\\.(h|hpp))$")
    list(APPEND POPS_TEST_ONLY_HEADERS "${CMAKE_MATCH_1}")
  else()
    message(FATAL_ERROR
      "invalid row in ${POPS_PUBLIC_HEADER_MANIFEST}: ${_pops_header_row}")
  endif()
endforeach()
list(SORT POPS_PUBLIC_HEADERS)
list(SORT POPS_TEST_ONLY_HEADERS)

if(NOT POPS_PUBLIC_HEADERS)
  message(FATAL_ERROR "${POPS_PUBLIC_HEADER_MANIFEST} declares no public headers")
endif()

function(pops_install_public_headers destination)
  foreach(_pops_header IN LISTS POPS_PUBLIC_HEADERS)
    get_filename_component(_pops_header_dir "${_pops_header}" DIRECTORY)
    install(FILES "${POPS_PUBLIC_HEADER_ROOT}/${_pops_header}"
            DESTINATION "${destination}/${_pops_header_dir}")
  endforeach()
  # The runtime ABI check recomputes the signature from these exact rows.
  install(FILES "${POPS_PUBLIC_HEADER_MANIFEST}" DESTINATION "${destination}")
endfunction()
