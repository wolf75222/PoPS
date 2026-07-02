#include <gtest/gtest.h>

#include <pops/runtime/module_capabilities.hpp>

#include <string>

using namespace pops;

TEST(CapabilityReport, ReportsSchemaAbiAndRouteVocabulary) {
  const NativeCapabilityReport report = native_capability_report();

  EXPECT_TRUE(report.schema_version == kCapabilityReportSchemaVersion) << "schema_version";
  EXPECT_TRUE(report.abi_version == kAbiVersion) << "abi_version";
  EXPECT_TRUE(report.target == "module") << "target_module";
  EXPECT_TRUE(!report.abi_key.empty()) << "abi_key_present";
  EXPECT_TRUE(report.runtime.dimension == 2) << "runtime_dimension";
  EXPECT_TRUE(report.runtime.amr_refinement_ratio == 2) << "runtime_amr_ratio";
  EXPECT_TRUE(!report.routes.empty()) << "routes_present";

  bool saw_amr_ratio = false;
  bool saw_precision = false;
  bool saw_custom_comm = false;
  bool saw_kokkos_lifecycle = false;
  for (const auto& row : report.routes) {
    EXPECT_TRUE(!row.route_id.empty()) << "route_id_nonempty";
    EXPECT_TRUE(row.status == "available" || row.status == "partial" ||
                row.status == "unavailable")
        << "route_status_vocab";
    if (row.route_id == "amr:refinement_ratio") {
      saw_amr_ratio = true;
      EXPECT_TRUE(row.status == "partial") << "amr_ratio_partial";
      EXPECT_TRUE(row.reason.find("ratio=2") != std::string::npos) << "amr_ratio_reason";
    } else if (row.route_id == "precision:single_or_mixed") {
      saw_precision = true;
      EXPECT_TRUE(row.status == "unavailable") << "precision_unavailable";
      EXPECT_TRUE(row.available_route == "precision=double") << "precision_available_route";
    } else if (row.route_id == "parallel:custom_communicator") {
      saw_custom_comm = true;
      EXPECT_TRUE(row.status == "unavailable") << "custom_comm_unavailable";
    } else if (row.route_id == "runtime:kokkos_lifecycle") {
      saw_kokkos_lifecycle = true;
      EXPECT_TRUE(row.status == "partial") << "kokkos_lifecycle_partial";
    }
  }
  EXPECT_TRUE(saw_amr_ratio) << "saw_amr_ratio";
  EXPECT_TRUE(saw_precision) << "saw_precision";
  EXPECT_TRUE(saw_custom_comm) << "saw_custom_comm";
  EXPECT_TRUE(saw_kokkos_lifecycle) << "saw_kokkos_lifecycle";
}
