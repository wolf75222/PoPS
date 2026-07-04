// Fixture .so B for the external-brick registry isolation test (ADC-622). Sibling of fixture A with a
// DISJOINT set of brick ids ("iso_b_riemann" only). Header-light (only external_brick.hpp): no Kokkos,
// no _pops. Its manifest MUST list only "iso_b_riemann" -- never fixture A's ids -- even when both .so
// are dlopen'd in the SAME process (the STB_GNU_UNIQUE unification ADC-622 fixes would otherwise leak
// A's ids into B's manifest on Linux/glibc).
#include <pops/runtime/program/external_brick.hpp>

#include <string>

POPS_REGISTER_BRICK("iso_b_riemann", "riemann", "temperature");
POPS_DEFINE_BRICK_MANIFEST();
