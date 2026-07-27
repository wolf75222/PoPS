# Minimal dependency fixture: the mismatch smoke must reach PoPS' MPI authentication without
# depending on the host OpenMP toolchain. No target from this file is ever compiled or linked.
set(Kokkos_FOUND TRUE)
set(Kokkos_VERSION "99.0-test-fixture")
if(NOT TARGET Kokkos::kokkos)
  add_library(Kokkos::kokkos INTERFACE IMPORTED)
  set_target_properties(Kokkos::kokkos PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include")
endif()
