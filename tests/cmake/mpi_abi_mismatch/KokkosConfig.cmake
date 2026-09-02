# Minimal dependency fixture: the mismatch smoke must reach PoPS' MPI authentication without
# depending on the host OpenMP toolchain. No target from this file is ever compiled or linked.
set(Kokkos_FOUND TRUE)
set(Kokkos_VERSION "99.0-test-fixture")
if(NOT TARGET Kokkos::kokkos)
  add_library(Kokkos::kokkos INTERFACE IMPORTED)
  set_target_properties(Kokkos::kokkos PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include"
    # Match the installed CUDA Kokkos 4.4.1 target shape used on ROMEO.  CMake
    # exposes the semicolon-delimited payload as separate property records;
    # PoPS must preserve every option while accepting only this exact CXX
    # language wrapper.
    INTERFACE_COMPILE_OPTIONS
      "$<$<COMPILE_LANGUAGE:CXX>:>;$<$<COMPILE_LANGUAGE:CXX>:-extended-lambda;-Wext-lambda-captures-this;-expt-relaxed-constexpr;-arch=sm_90>;$<$<COMPILE_LANGUAGE:CXX>:>")
endif()
