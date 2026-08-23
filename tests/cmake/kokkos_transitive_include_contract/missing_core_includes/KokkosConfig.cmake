# Negative fixture: the aggregate declares the standard core dependency, but neither target offers
# a concrete development include authority.  PoPS must reject this before any build is attempted.
set(Kokkos_FOUND TRUE)
set(Kokkos_VERSION "99.0-test-fixture")
if(NOT TARGET Kokkos::kokkoscore)
  add_library(Kokkos::kokkoscore INTERFACE IMPORTED)
endif()
if(NOT TARGET Kokkos::kokkos)
  add_library(Kokkos::kokkos INTERFACE IMPORTED)
  set_target_properties(Kokkos::kokkos PROPERTIES
    INTERFACE_LINK_LIBRARIES "Kokkos::kokkoscore")
endif()
