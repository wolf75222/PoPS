# Version companion for the hermetic Kokkos fixture.
#
# PoPS requests ``find_package(Kokkos 4.2 CONFIG)``.  A Config file without a
# ConfigVersion companion is rejected by CMake for that versioned request,
# which silently selected the network FetchContent fallback in isolated
# add_subdirectory tests.  Keep this fixture deliberately newer than every
# supported production Kokkos while retaining the normal compatible-version
# semantics.
set(PACKAGE_VERSION "99.0.0")

if(PACKAGE_FIND_VERSION VERSION_GREATER PACKAGE_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
endif()

if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
  set(PACKAGE_VERSION_EXACT TRUE)
endif()
