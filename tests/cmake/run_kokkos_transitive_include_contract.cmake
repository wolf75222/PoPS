cmake_minimum_required(VERSION 3.21)

foreach(_required POPS_SOURCE_DIR POPS_KOKKOS_FIXTURE_ROOT POPS_TEST_BUILD_DIR POPS_NATIVE_DIM)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "Kokkos transitive-include contract test is missing -D${_required}=...")
  endif()
endforeach()

function(_pops_configure_kokkos_fixture _name _expect_success _expected_text)
  set(_build_dir "${POPS_TEST_BUILD_DIR}/${_name}")
  file(REMOVE_RECURSE "${_build_dir}")
  set(_command
    "${CMAKE_COMMAND}"
    -S "${POPS_SOURCE_DIR}"
    -B "${_build_dir}"
    -DPOPS_BUILD_TESTS=OFF
    -DPOPS_BUILD_PYTHON=OFF
    -DPOPS_USE_MPI=OFF
    "-DPOPS_NATIVE_DIM=${POPS_NATIVE_DIM}"
    "-DKokkos_DIR=${POPS_KOKKOS_FIXTURE_ROOT}/${_name}")
  if(DEFINED POPS_GENERATOR AND NOT POPS_GENERATOR STREQUAL "")
    list(APPEND _command -G "${POPS_GENERATOR}")
  endif()
  execute_process(
    COMMAND ${_command}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(_log "${_stdout}${_stderr}")
  if(_expect_success)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR
        "Kokkos transitive include fixture ${_name} failed to configure (${_result}):\n${_log}")
    endif()
  elseif(_result EQUAL 0)
    message(FATAL_ERROR "Kokkos transitive include fixture ${_name} configured successfully")
  endif()
  if(NOT _log MATCHES "${_expected_text}")
    message(FATAL_ERROR
      "Kokkos transitive include fixture ${_name} did not emit its required contract evidence "
      "(${_expected_text}):\n${_log}")
  endif()
endfunction()

_pops_configure_kokkos_fixture(
  valid TRUE "Kokkos runtime include authority: Kokkos::kokkoscore")
_pops_configure_kokkos_fixture(
  missing_core_includes FALSE
  "Kokkos::kokkoscore exposes none")
