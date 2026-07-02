// ADC-291: the C++ canonical aux name<->component table (pops/core/aux_names.hpp) is the mirror of
// AUX_CANONICAL (python/pops/dsl.py), generated from the SAME single source as pops::Aux (the base
// contract phi/grad_x/grad_y + the POPS_AUX_FIELDS X-macro B_z/T_e). It lets a C++ caller resolve a
// canonical aux field by name WITHOUT the Python facade. This test pins the C++ side; the
// C++<->Python coherence is pinned by tests/python/unit/runtime/test_capabilities.py.

#include <gtest/gtest.h>

#include <pops/core/state/aux_names.hpp>
#include <pops/core/state/state.hpp>

#include <string_view>

using namespace pops;

// Compile-time coherence: the table is constexpr, so the canonical indices are pinned at build time
// (a drift from POPS_AUX_FIELDS / the base contract is a hard compile error, not a runtime surprise).
static_assert(aux_canonical_index("phi") == 0, "phi must be aux component 0");
static_assert(aux_canonical_index("B_z") == 3, "B_z must be aux component 3 (POPS_AUX_FIELDS)");
static_assert(aux_canonical_index("T_e") == 4, "T_e must be aux component 4 (POPS_AUX_FIELDS)");
static_assert(aux_canonical_index("kappa") == -1, "a model-named field is not canonical");
static_assert(kAuxMaxComps == kAuxNamedBase + kAuxMaxExtra, "kAuxMaxComps = base + max extras");

// canonical name -> component (mirror of AUX_CANONICAL on the DSL side)
TEST(AuxNames, CanonicalIndexByName) {
  EXPECT_EQ(aux_canonical_index("phi"), 0) << "phi=0";
  EXPECT_EQ(aux_canonical_index("grad_x"), 1) << "grad_x=1";
  EXPECT_EQ(aux_canonical_index("grad_y"), 2) << "grad_y=2";
  EXPECT_EQ(aux_canonical_index("B_z"), 3) << "B_z=3";
  EXPECT_EQ(aux_canonical_index("T_e"), 4) << "T_e=4";
  // a model-NAMED field (resolved per block by the facade) is NOT a canonical aux field
  EXPECT_EQ(aux_canonical_index("kappa"), -1) << "named field not canonical";
  EXPECT_EQ(aux_canonical_index(""), -1) << "empty not canonical";
}

// inverse coherence: component -> canonical name
TEST(AuxNames, CanonicalNameByIndex) {
  EXPECT_EQ(aux_canonical_name(0), "phi") << "name(0)=phi";
  EXPECT_EQ(aux_canonical_name(4), "T_e") << "name(4)=T_e";
  EXPECT_EQ(aux_canonical_name(kAuxNamedBase), std::string_view{}) << "name(kAuxNamedBase) empty";
}

// the canonical extras live STRICTLY below the named base (B_z/T_e never collide with extra[k])
TEST(AuxNames, CanonicalExtrasBelowNamedBase) {
  EXPECT_TRUE(aux_canonical_index("B_z") < kAuxNamedBase) << "B_z below named base";
  EXPECT_TRUE(aux_canonical_index("T_e") < kAuxNamedBase) << "T_e below named base";
}
