#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/runtime/module_capabilities.hpp>

#include "test_harness.hpp"

#include <string>

using namespace pops;

static int pops_run_test_capability_report() {
  pops::test::Checker chk;
  const NativeCapabilityReport report = native_capability_report();

  chk(report.schema_version == kCapabilityReportSchemaVersion, "schema_version");
  chk(report.abi_version == kAbiVersion, "abi_version");
  chk(report.target == "module", "target_module");
  chk(!report.abi_key.empty(), "abi_key_present");
  chk(report.runtime.dimension == 2, "runtime_dimension");
  chk(report.runtime.amr_refinement_ratio == 2, "runtime_amr_ratio");
  chk(!report.routes.empty(), "routes_present");

  bool saw_amr_ratio = false;
  bool saw_precision = false;
  bool saw_custom_comm = false;
  bool saw_kokkos_lifecycle = false;
  for (const auto& row : report.routes) {
    chk(!row.route_id.empty(), "route_id_nonempty");
    chk(row.status == "available" || row.status == "partial" || row.status == "unavailable",
        "route_status_vocab");
    if (row.route_id == "amr:refinement_ratio") {
      saw_amr_ratio = true;
      chk(row.status == "partial", "amr_ratio_partial");
      chk(row.reason.find("ratio=2") != std::string::npos, "amr_ratio_reason");
    } else if (row.route_id == "precision:single_or_mixed") {
      saw_precision = true;
      chk(row.status == "unavailable", "precision_unavailable");
      chk(row.available_route == "precision=double", "precision_available_route");
    } else if (row.route_id == "parallel:custom_communicator") {
      saw_custom_comm = true;
      chk(row.status == "unavailable", "custom_comm_unavailable");
    } else if (row.route_id == "runtime:kokkos_lifecycle") {
      saw_kokkos_lifecycle = true;
      chk(row.status == "partial", "kokkos_lifecycle_partial");
    }
  }
  chk(saw_amr_ratio, "saw_amr_ratio");
  chk(saw_precision, "saw_precision");
  chk(saw_custom_comm, "saw_custom_comm");
  chk(saw_kokkos_lifecycle, "saw_kokkos_lifecycle");

  if (chk.fails() == 0)
    std::printf("OK test_capability_report\n");
  return chk.failed();
}

TEST(test_capability_report, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_capability_report, "test_capability_report"), 0);
}
