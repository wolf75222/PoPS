# Preserve authenticated native contract bytes across compiler wrappers. A -D
# argument is not a lossless transport for strings containing whitespace:
# nvcc_wrapper can erase spaces even when CMake quotes the macro argument.
# Write one hexadecimal escape per UTF-8 byte directly into a generated header.
# Adjacent escapes are unambiguous because each starts with its own backslash.
function(_pops_append_native_contract_string _header _name _value)
  if(NOT _name MATCHES "^[A-Za-z_][A-Za-z_0-9]*$")
    message(FATAL_ERROR "Invalid native contract macro name: ${_name}")
  endif()
  string(HEX "${_value}" _hex)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "\\\\x\\1" _literal "${_hex}")
  file(APPEND "${_header}" "#define ${_name} \"${_literal}\"\n")
endfunction()
