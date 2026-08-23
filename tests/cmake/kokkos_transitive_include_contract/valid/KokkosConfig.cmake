# Hermetic configuration fixture: external Kokkos build trees can expose the aggregate target as
# link-only while its declared canonical core target owns the include authority.
set(Kokkos_FOUND TRUE)
set(Kokkos_VERSION "99.0-test-fixture")
if(NOT TARGET Kokkos::kokkoscore)
  add_library(Kokkos::kokkoscore INTERFACE IMPORTED)
  set_target_properties(Kokkos::kokkoscore PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include")
endif()
if(NOT TARGET Kokkos::kokkos)
  add_library(Kokkos::kokkos INTERFACE IMPORTED)
  set_target_properties(Kokkos::kokkos PROPERTIES
    INTERFACE_LINK_LIBRARIES "Kokkos::kokkoscore")
endif()
