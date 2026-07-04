// External-brick registry PER-.so ISOLATION (ADC-622). Two independently built fixture .so, each
// registering DISJOINT brick ids via POPS_REGISTER_BRICK + POPS_DEFINE_BRICK_MANIFEST, are dlopen'd in
// ONE process; each .so's pops_brick_manifest() must describe ONLY that .so's bricks.
//
// WHY THIS EXISTS. BrickRegistry::instance() is a header-only Meyers singleton. On Linux/GCC the
// function-local static was emitted with default visibility + vague linkage -> STB_GNU_UNIQUE, which
// glibc's loader UNIFIES across every dlopen'd image EVEN UNDER RTLD_LOCAL. Loading fixture A then B
// would make B's pops_brick_manifest() list A's ids too (the whole process's bricks). ADC-622 gives
// BrickRegistry hidden visibility (POPS_BRICK_LOCAL) so the symbol is per-image; the brick .so also
// build with -fno-gnu-unique (belt-and-suspenders).
//
// PLATFORM NOTE. On macOS (two-level namespace) and Windows (per-.dll symbols) the isolation ALREADY
// holds, so this test passes trivially there -- it is written to be MEANINGFUL on all platforms, but
// the REAL regression proof is the Linux/GCC CI lane where GNU_UNIQUE used to unify the registry. The
// two fixture .so paths arrive as compile definitions (POPS_ISO_FIXTURE_A_SO / _B_SO) set by CMake,
// which builds them from external_brick_fixture_{a,b}.cpp with -fno-gnu-unique on GCC.

#include <gtest/gtest.h>

#include <pops/runtime/dynamic/dynlib.hpp>  // portable dlopen<->LoadLibraryW (ADC-99)

#include <string>

namespace {

using ManifestFn = const char* (*)();

// dlopen @p so_path (RTLD_LOCAL, as CPython loads a brick .so) and return its pops_brick_manifest()
// string. Fails the test with an actionable message if the .so cannot open or does not export the
// reader. The RTLD_LOCAL default matters: it is the scope under which GNU_UNIQUE (before ADC-622)
// still unified the registry across images -- so loading under it is exactly the regression condition.
std::string load_manifest(const char* so_path) {
  pops::dynlib::handle h = pops::dynlib::open(so_path);
  EXPECT_TRUE(pops::dynlib::valid(h)) << "dlopen('" << so_path << "'): " << pops::dynlib::last_error();
  if (!pops::dynlib::valid(h))
    return "";
  auto fn = reinterpret_cast<ManifestFn>(pops::dynlib::sym(h, "pops_brick_manifest"));
  EXPECT_TRUE(fn != nullptr) << "'" << so_path << "' does not export pops_brick_manifest()";
  return fn ? std::string(fn()) : std::string();
}

bool contains(const std::string& hay, const char* needle) {
  return hay.find(needle) != std::string::npos;
}

}  // namespace

// Each fixture .so's manifest lists ONLY its own bricks, even with BOTH loaded in one process. This is
// the ADC-622 property: without hidden visibility, on Linux/GCC the second manifest would also carry
// the first .so's ids (STB_GNU_UNIQUE unification of the header-only registry singleton).
TEST(ExternalBrickIsolation, EachSoManifestListsOnlyItsOwnBricks) {
  // Load A first, then B: the ORDER matters for the (fixed) bug -- B loaded after A is where the
  // unified registry would have leaked A's ids into B's manifest.
  const std::string a = load_manifest(POPS_ISO_FIXTURE_A_SO);
  const std::string b = load_manifest(POPS_ISO_FIXTURE_B_SO);

  // A's manifest carries A's two ids and NEITHER of B's.
  EXPECT_TRUE(contains(a, "iso_a_riemann")) << "A manifest lists its own riemann id";
  EXPECT_TRUE(contains(a, "iso_a_precond")) << "A manifest lists its own preconditioner id";
  EXPECT_FALSE(contains(a, "iso_b_riemann")) << "A manifest must NOT leak B's id (registry isolation)";

  // B's manifest carries B's id and NEITHER of A's -- the regression direction (B loaded after A).
  EXPECT_TRUE(contains(b, "iso_b_riemann")) << "B manifest lists its own riemann id";
  EXPECT_FALSE(contains(b, "iso_a_riemann")) << "B manifest must NOT leak A's id (registry isolation)";
  EXPECT_FALSE(contains(b, "iso_a_precond")) << "B manifest must NOT leak A's preconditioner id";
}
