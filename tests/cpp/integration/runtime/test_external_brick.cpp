// BrickRegistry: the process-global external-C++ brick registry backing Spec 3 section 21-22
// (criterion 20, ADC-463). Exercises macro registration, lookup of a registered/unknown id, the
// id listing, and the manifest fields a registered brick carries -- no Program/codegen needed.

#include <gtest/gtest.h>

#include <pops/runtime/program/external_brick.hpp>

#include <string>
#include <vector>

using pops::runtime::program::BrickManifestEntry;
using pops::runtime::program::BrickRegistry;

// Register two bricks at static-init time via the macro. The third argument is the
// requirements/capabilities CSV the manifest surfaces (a host-only string).
POPS_REGISTER_BRICK("test_hllc", "riemann", "pressure,wave_speeds");
POPS_REGISTER_BRICK("test_precond", "preconditioner", "");

// Sequential sections below share the process-global BrickRegistry populated by the macros
// above (and mutated by the reregistration section), so they run as ordered phases of a single
// TEST rather than independent tests that could interleave with global registry state.
TEST(ExternalBrick, RegistryLookupListingAndReregistration) {
  const BrickRegistry& reg = BrickRegistry::instance();

  // --- lookup of a registered brick returns its manifest entry ---
  {
    const BrickManifestEntry* e = reg.lookup("test_hllc");
    ASSERT_TRUE(e != nullptr) << "lookup_registered_hllc";
    EXPECT_TRUE(e->id == "test_hllc") << "hllc_id";
    EXPECT_TRUE(e->category == "riemann") << "hllc_category";
    EXPECT_TRUE(e->requirements == "pressure,wave_speeds") << "hllc_requirements";

    const BrickManifestEntry* p = reg.lookup("test_precond");
    ASSERT_TRUE(p != nullptr) << "lookup_registered_precond";
    EXPECT_TRUE(p->category == "preconditioner") << "precond_category";
    EXPECT_TRUE(p->requirements.empty()) << "precond_no_requirements";
  }

  // --- lookup of an unknown id returns null (never throws) ---
  EXPECT_TRUE(reg.lookup("not_registered") == nullptr) << "lookup_unknown_is_null";

  // --- ids() lists every registered brick ---
  {
    const std::vector<std::string>& ids = reg.ids();
    bool has_hllc = false;
    bool has_precond = false;
    for (const std::string& id : ids) {
      if (id == "test_hllc")
        has_hllc = true;
      if (id == "test_precond")
        has_precond = true;
    }
    EXPECT_TRUE(has_hllc) << "ids_lists_hllc";
    EXPECT_TRUE(has_precond) << "ids_lists_precond";
    EXPECT_TRUE(ids.size() >= 2) << "ids_at_least_two";
  }

  // --- exact re-registration is idempotent; a conflicting row is a schema error ---
  {
    const std::size_t before = reg.ids().size();
    BrickRegistry::instance().register_brick(
        {"test_hllc", "riemann", "pressure,wave_speeds", "", "test_hllc"});
    EXPECT_TRUE(reg.ids().size() == before) << "reregister_does_not_duplicate";
    EXPECT_THROW(
        BrickRegistry::instance().register_brick(
            {"test_hllc", "riemann", "pressure,wave_speeds,contact_speed", "", "test_hllc"}),
        std::runtime_error)
        << "conflicting_reregistration_is_rejected";
    const BrickManifestEntry* e = reg.lookup("test_hllc");
    EXPECT_TRUE(e != nullptr && e->requirements == "pressure,wave_speeds")
        << "conflicting_reregistration_does_not_mutate_original";
  }
}
