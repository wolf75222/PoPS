#pragma once

#include <pops/runtime/amr/amr_layout_transfer_bridge.hpp>

namespace amr_layout_transfer_binding {
using Spec = pops::SystemLayoutTransferSpec<pops::kNativeDimension>;
using PhysicalSpec = pops::AmrPhysicalTransferSpec<pops::kNativeDimension>;
using Session = pops::PreparedAmrSystemLayoutTransfer<pops::kNativeDimension>;
using Receipt = pops::AmrLayoutTransferReceipt;

inline void exact_keys(const py::dict& row, std::initializer_list<const char*> expected,
                       const char* context) {
  if (row.size() != expected.size())
    throw py::value_error(std::string(context) + " has unexpected keys");
  for (const auto* key : expected)
    if (!row.contains(py::str(key)))
      throw py::value_error(std::string(context) + " is missing " + key);
}

inline Spec specification(const py::dict& row) {
  exact_keys(row,
             {"mapping_identity", "provider_identity", "provider_component_identity",
              "provider_manifest_identity", "source_layout_identity", "target_layout_identity",
              "source_block", "target_block", "source_representation", "target_representation",
              "synchronization_identity", "refinement_ratio", "operation", "physical_contract",
              "physical_source_to_target", "physical_source_active", "physical_target_active",
              "program_invocation"},
             "AMR transfer spec");
  Spec spec;
  spec.mapping_identity = py::cast<std::string>(row["mapping_identity"]);
  spec.provider_identity = py::cast<std::string>(row["provider_identity"]);
  spec.provider_component_identity = py::cast<std::string>(row["provider_component_identity"]);
  spec.provider_manifest_identity = py::cast<std::string>(row["provider_manifest_identity"]);
  spec.source_layout_identity = py::cast<std::string>(row["source_layout_identity"]);
  spec.target_layout_identity = py::cast<std::string>(row["target_layout_identity"]);
  spec.source_block = py::cast<std::string>(row["source_block"]);
  spec.target_block = py::cast<std::string>(row["target_block"]);
  spec.source_representation = py::cast<std::string>(row["source_representation"]);
  spec.target_representation = py::cast<std::string>(row["target_representation"]);
  spec.synchronization_identity = py::cast<std::string>(row["synchronization_identity"]);
  spec.refinement_ratio = py::cast<decltype(spec.refinement_ratio)>(row["refinement_ratio"]);
  spec.operation = py::cast<std::int32_t>(row["operation"]);
  spec.physical_contract = py::cast<bool>(row["physical_contract"]);
  spec.physical_source_to_target =
      py::cast<decltype(spec.physical_source_to_target)>(row["physical_source_to_target"]);
  spec.physical_source_active =
      py::cast<decltype(spec.physical_source_active)>(row["physical_source_active"]);
  spec.physical_target_active =
      py::cast<decltype(spec.physical_target_active)>(row["physical_target_active"]);
  spec.program_invocation = py::cast<std::string>(row["program_invocation"]);
  return spec;
}

inline pops::SystemLayoutTransferExecution execution(const py::dict& row) {
  exact_keys(row,
             {"execution_identity", "context_version", "memory_space", "backend_identity",
              "device_identity", "scalar_type", "storage_precision", "compute_precision",
              "accumulation_precision", "reduction_precision", "stream_handle", "stream_identity",
              "communicator_f_handle", "communicator_datatype_f_handle", "communicator_identity",
              "communicator_datatype_identity"},
             "AMR transfer execution");
  return {py::cast<std::uint32_t>(row["context_version"]),
          py::cast<std::string>(row["execution_identity"]),
          py::cast<std::int32_t>(row["memory_space"]),
          py::cast<std::string>(row["backend_identity"]),
          py::cast<std::string>(row["device_identity"]),
          py::cast<std::int32_t>(row["scalar_type"]),
          py::cast<std::int32_t>(row["storage_precision"]),
          py::cast<std::int32_t>(row["compute_precision"]),
          py::cast<std::int32_t>(row["accumulation_precision"]),
          py::cast<std::int32_t>(row["reduction_precision"]),
          py::cast<std::uint64_t>(row["stream_handle"]),
          py::cast<std::string>(row["stream_identity"]),
          py::cast<std::int64_t>(row["communicator_f_handle"]),
          py::cast<std::int64_t>(row["communicator_datatype_f_handle"]),
          py::cast<std::string>(row["communicator_identity"]),
          py::cast<std::string>(row["communicator_datatype_identity"])};
}

inline PhysicalSpec physical_specification(const py::dict& specification_row, const py::dict& row) {
  exact_keys(row,
             {"physical_contract_identity", "base_bin_weights", "base_bin_lower", "base_bin_upper",
              "quadrature_identity", "budget"},
             "AMR physical measure");
  PhysicalSpec spec;
  spec.authentication = specification(specification_row);
  spec.physical_contract_identity = py::cast<std::string>(row["physical_contract_identity"]);
  spec.base_bin_weights = py::cast<decltype(spec.base_bin_weights)>(row["base_bin_weights"]);
  spec.base_bin_lower = py::cast<decltype(spec.base_bin_lower)>(row["base_bin_lower"]);
  spec.base_bin_upper = py::cast<decltype(spec.base_bin_upper)>(row["base_bin_upper"]);
  spec.quadrature_identity = py::cast<std::string>(row["quadrature_identity"]);
  const auto budget = py::cast<py::dict>(row["budget"]);
  exact_keys(budget,
             {"destination_cells", "intersection_probes", "canonical_jobs", "transported_elements",
              "prepared_bytes"},
             "AMR transfer budget");
  spec.budget = {py::cast<std::size_t>(budget["destination_cells"]),
                 py::cast<std::size_t>(budget["intersection_probes"]),
                 py::cast<std::size_t>(budget["canonical_jobs"]),
                 py::cast<std::size_t>(budget["transported_elements"]),
                 py::cast<std::size_t>(budget["prepared_bytes"])};
  return spec;
}

inline void bind(py::module_& module, py::class_<pops::AmrSystem<pops::kNativeDimension>>& cls) {
  py::class_<Receipt>(module, "_AmrLayoutTransferReceipt")
      .def_property_readonly("program_invocation",
                             [](const Receipt& r) { return r.transfer.program_invocation; })
      .def_property_readonly("applied", [](const Receipt& r) { return r.transfer.applied; })
      .def_property_readonly("mapping_identity",
                             [](const Receipt& r) { return r.transfer.mapping_identity; })
      .def_property_readonly("provider_identity",
                             [](const Receipt& r) { return r.transfer.provider_identity; })
      .def_property_readonly(
          "provider_component_identity",
          [](const Receipt& r) { return r.transfer.provider_component_identity; })
      .def_property_readonly("provider_manifest_identity",
                             [](const Receipt& r) { return r.transfer.provider_manifest_identity; })
      .def_property_readonly("source_layout_identity",
                             [](const Receipt& r) { return r.transfer.source_layout_identity; })
      .def_property_readonly("target_layout_identity",
                             [](const Receipt& r) { return r.transfer.target_layout_identity; })
      .def_property_readonly("source_block",
                             [](const Receipt& r) { return r.transfer.source_block; })
      .def_property_readonly("target_block",
                             [](const Receipt& r) { return r.transfer.target_block; })
      .def_property_readonly("execution_identity",
                             [](const Receipt& r) { return r.transfer.execution_identity; })
      .def_property_readonly("operation", [](const Receipt& r) { return r.transfer.operation; })
      .def_property_readonly("generation", [](const Receipt& r) { return r.transfer.generation; })
      .def_property_readonly("attempt", [](const Receipt& r) { return r.transfer.attempt; })
      .def_property_readonly("source_element_count",
                             [](const Receipt& r) { return r.transfer.source_element_count; })
      .def_property_readonly("destination_element_count",
                             [](const Receipt& r) { return r.transfer.destination_element_count; })
      .def_readonly("physical_contract_identity", &Receipt::physical_contract_identity)
      .def_readonly("source_hierarchy_identity", &Receipt::source_hierarchy_identity)
      .def_readonly("target_hierarchy_identity", &Receipt::target_hierarchy_identity)
      .def_readonly("source_hierarchy_generation", &Receipt::source_hierarchy_generation)
      .def_readonly("target_hierarchy_generation", &Receipt::target_hierarchy_generation)
      .def_readonly("source_stage_identity", &Receipt::source_stage_identity)
      .def_readonly("source_stage_generation", &Receipt::source_stage_generation)
      .def_readonly("target_stage_identity", &Receipt::target_stage_identity)
      .def_readonly("target_stage_generation", &Receipt::target_stage_generation)
      .def_readonly("source_active_elements", &Receipt::source_active_elements)
      .def_readonly("destination_active_elements", &Receipt::destination_active_elements)
      .def_readonly("canonical_jobs", &Receipt::canonical_jobs)
      .def_readonly("transported_elements", &Receipt::transported_elements)
      .def_readonly("prepared_bytes", &Receipt::prepared_bytes);
  py::class_<Session, std::shared_ptr<Session>>(module, "_PreparedAmrSystemLayoutTransfer")
      .def("expected_receipt_contract", &Session::expected_receipt_contract)
      .def("begin_transaction", &Session::begin_transaction, py::arg("generation"))
      .def("capture", &Session::capture, py::arg("generation"), py::arg("attempt"))
      .def("apply", &Session::apply, py::arg("generation"), py::arg("attempt"))
      .def("reject_attempt", &Session::reject_attempt, py::arg("generation"), py::arg("attempt"))
      .def("finalize_transaction", &Session::finalize_transaction, py::arg("generation"))
      .def("rollback_transaction", &Session::rollback_transaction, py::arg("generation"));
  cls.def("_layout_transfer_capacity_budget",
          [](pops::AmrSystem<pops::kNativeDimension>& source,
             pops::AmrSystem<pops::kNativeDimension>& target, const std::string& source_block,
             const std::string& target_block, std::size_t source_cells, std::size_t target_cells) {
            const auto budget = Session::capacity_budget(source, target, source_block, target_block,
                                                         source_cells, target_cells);
            py::dict result;
            result["destination_cells"] = budget.destination_cells;
            result["intersection_probes"] = budget.intersection_probes;
            result["canonical_jobs"] = budget.canonical_jobs;
            result["transported_elements"] = budget.transported_elements;
            result["prepared_bytes"] = budget.prepared_bytes;
            return result;
          });
  cls.def(
      "_prepare_layout_transfer",
      [](pops::AmrSystem<pops::kNativeDimension>& source,
         pops::AmrSystem<pops::kNativeDimension>& target,
         std::shared_ptr<pops::component::LoadedComponent> component, const py::dict& spec,
         const py::dict& context, const py::dict& physical) {
        return Session::prepare(source, target, std::move(component),
                                physical_specification(spec, physical), execution(context));
      },
      py::arg("target"), py::arg("component"), py::arg("spec"), py::arg("execution"),
      py::arg("physical"), py::keep_alive<0, 1>(), py::keep_alive<0, 2>());
}
}  // namespace amr_layout_transfer_binding
