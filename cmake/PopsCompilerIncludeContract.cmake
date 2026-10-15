# Capture compiler-declared implicit headers before native compilation. The
# generated module publishes content digests, never a permissive path prefix.
set(_pops_compiler_contract_header
    "${CMAKE_CURRENT_BINARY_DIR}/compiler-contract/pops_compiler_include_contract.hpp")
set(_pops_compiler_contract_roots "")
foreach(_pops_compiler_include IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
  list(APPEND _pops_compiler_contract_roots --root "${_pops_compiler_include}")
endforeach()
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "${CMAKE_SOURCE_DIR}/scripts/write_compiler_include_contract.py"
          --compiler "${CMAKE_CXX_COMPILER}"
          --output "${_pops_compiler_contract_header}"
          ${_pops_compiler_contract_roots}
  RESULT_VARIABLE _pops_compiler_contract_status
  OUTPUT_VARIABLE _pops_compiler_contract_output
  ERROR_VARIABLE _pops_compiler_contract_error)
if(NOT _pops_compiler_contract_status EQUAL 0)
  message(FATAL_ERROR
    "Cannot authenticate the native compiler implicit headers:\n"
    "${_pops_compiler_contract_output}${_pops_compiler_contract_error}")
endif()
string(STRIP "${_pops_compiler_contract_output}" _pops_compiler_contract_output)
message(STATUS "Native compiler include contract: ${_pops_compiler_contract_output}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/scripts/write_compiler_include_contract.py"
  "${CMAKE_SOURCE_DIR}/python/pops/codegen/_compiler_include_tree.py")
target_include_directories(_pops PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/compiler-contract")
