// Fixture .so A for the external-brick registry isolation test (ADC-622). Registers TWO brick ids
// unique to THIS image and exports the manifest reader. Header-light (only external_brick.hpp): no
// Kokkos, no _pops -- CMake compiles it into a standalone .so the test dlopens. Its manifest MUST list
// only "iso_a_riemann" / "iso_a_precond", never the ids fixture B registers (that is the isolation
// property under test). Sibling of external_brick_fixture_b.cpp with distinct ids.
#include <pops/runtime/program/external_brick.hpp>

#include <string>

POPS_REGISTER_BRICK("iso_a_riemann", "riemann", "pressure,wave_speeds");
POPS_REGISTER_BRICK("iso_a_precond", "preconditioner", "");
POPS_DEFINE_BRICK_MANIFEST();
