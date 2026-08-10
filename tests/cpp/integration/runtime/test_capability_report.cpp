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
  EXPECT_TRUE(report.runtime.dimension == kNativeDimension) << "runtime_dimension";
  EXPECT_TRUE(report.runtime.amr_refinement_ratio_selection == "hierarchy_exact_rank")
      << "runtime_amr_ratio_authority";
  EXPECT_TRUE(report.runtime.amr_refinement_ratio_rank == kNativeDimension)
      << "runtime_amr_ratio_rank";
  EXPECT_TRUE(!report.routes.empty()) << "routes_present";

  bool saw_amr_ratio = false;
  bool saw_precision = false;
  bool saw_custom_comm = false;
  bool saw_kokkos_lifecycle = false;
  bool saw_uniform_checkpoint_v5 = false;
  bool saw_amr_checkpoint_v7 = false;
  bool saw_dynamic_regrid_checkpoint = false;
  bool saw_mpi_world = false;
  bool saw_weno5 = false;
  for (const auto& row : report.routes) {
    EXPECT_TRUE(!row.route_id.empty()) << "route_id_nonempty";
    EXPECT_TRUE(row.status == "available" || row.status == "partial" || row.status == "unavailable")
        << "route_status_vocab";
    if (row.route_id == "amr:refinement_ratio") {
      saw_amr_ratio = true;
      EXPECT_TRUE(row.status == "partial") << "amr_ratio_partial";
      EXPECT_TRUE(row.reason.find("process-global ratio invariant") != std::string::npos)
          << "amr_ratio_reason";
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
    } else if (row.route_id == "checkpoint:uniform_accepted_state_v5") {
      saw_uniform_checkpoint_v5 = true;
      EXPECT_TRUE(row.status == "available") << "uniform_checkpoint_v5_available";
      EXPECT_TRUE(row.layout == "uniform") << "uniform_checkpoint_v5_layout";
    } else if (row.route_id == "checkpoint:amr_accepted_state_v7") {
      saw_amr_checkpoint_v7 = true;
      EXPECT_TRUE(row.status == "available") << "amr_checkpoint_v7_available";
      EXPECT_TRUE(row.layout == "amr") << "amr_checkpoint_v7_layout";
    } else if (row.route_id == "checkpoint:amr_dynamic_regrid") {
      saw_dynamic_regrid_checkpoint = true;
      EXPECT_TRUE(row.status == "available") << "dynamic_regrid_checkpoint_available";
    } else if (row.route_id == "parallel:mpi_world_communicator") {
      saw_mpi_world = true;
      EXPECT_TRUE((row.status == "available") == report.capabilities.supports_mpi)
          << "mpi_world_status_matches_build";
      EXPECT_TRUE(row.reason.find("ParallelContext") == std::string::npos)
          << "mpi_world_uses_final_execution_context";
    } else if (row.route_id == "reconstruction:weno5") {
      saw_weno5 = true;
      EXPECT_TRUE(row.status == "available") << "weno5_available";
      EXPECT_TRUE(row.layout == "uniform|amr") << "weno5_public_layout";
      EXPECT_TRUE(row.reason.find("order-5 coarse/fine provider") != std::string::npos)
          << "weno5_amr_requirement";
    }
  }
  EXPECT_TRUE(saw_amr_ratio) << "saw_amr_ratio";
  EXPECT_TRUE(saw_precision) << "saw_precision";
  EXPECT_TRUE(saw_custom_comm) << "saw_custom_comm";
  EXPECT_TRUE(saw_kokkos_lifecycle) << "saw_kokkos_lifecycle";
  EXPECT_TRUE(saw_uniform_checkpoint_v5) << "saw_uniform_checkpoint_v5";
  EXPECT_TRUE(saw_amr_checkpoint_v7) << "saw_amr_checkpoint_v7";
  EXPECT_TRUE(saw_dynamic_regrid_checkpoint) << "saw_dynamic_regrid_checkpoint";
  EXPECT_TRUE(saw_mpi_world) << "saw_mpi_world";
  EXPECT_TRUE(saw_weno5) << "saw_weno5";
}
