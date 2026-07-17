#include "../bindings_detail.hpp"

#include <pops/runtime/dynamic/component_loader.hpp>

#include "generated_component_invokers.inc"

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace {

py::dict report(const pops::component::LoadedComponent& loaded) {
  const auto& api = loaded.api();
  py::list interfaces;
  for (std::size_t index = 0; index < api.interface_count; ++index) {
    const auto& row = api.interfaces[index];
    py::dict entry;
    entry["id"] = static_cast<int>(row.interface_id);
    entry["version"] = row.interface_version;
    entry["table"] = pops::component::generated_native_interface_table_name(row.interface_id);
    entry["table_size"] = row.table_size;
    interfaces.append(std::move(entry));
  }
  py::dict result;
  result["component_id"] = api.component_id;
  result["semantic_identity"] = api.semantic_identity;
  result["manifest_identity"] = api.manifest_identity;
  result["abi_key"] = api.abi_key;
  result["catalog_sha256"] = api.catalog_sha256;
  result["protocol_abi"] = api.protocol_abi;
  result["interfaces"] = std::move(interfaces);
  return result;
}

}  // namespace

void init_component_loader(py::module_& m) {
  using pops::component::ExpectedNativeComponent;
  using pops::component::LoadedComponent;
  using pops::component::RequiredNativeInterface;

  auto handle =
      py::class_<LoadedComponent, std::shared_ptr<LoadedComponent>>(m, "_NativeComponentHandle");
  handle.def("report", &report);
  pops::component::generated_pybind::register_component_invokers(handle);

  m.def(
      "_load_component",
      [](const std::string& path, const std::string& component_id,
         const std::string& semantic_identity, const std::string& manifest_identity,
         const std::string& catalog_sha256, const std::string& abi_key,
         const std::vector<std::tuple<int, std::uint32_t, std::string>>& interfaces,
         const std::string& prepare_parameters_json, const std::string& prepare_target_json) {
        ExpectedNativeComponent expected{
            component_id, semantic_identity,       manifest_identity,  catalog_sha256, abi_key,
            {},           prepare_parameters_json, prepare_target_json};
        for (const auto& [raw_id, version, declared_table] : interfaces) {
          if (raw_id < 0)
            throw py::value_error("native component interface id must be non-negative");
          const auto id = static_cast<PopsNativeInterfaceIdV1>(raw_id);
          const auto* generated_table = pops::component::generated_native_interface_table_name(id);
          if (generated_table == nullptr || declared_table != generated_table)
            throw py::value_error("native component interface C table name mismatch");
          expected.interfaces.push_back(RequiredNativeInterface{
              id, version, pops::component::native_interface_table_size(id)});
        }
        return std::make_shared<LoadedComponent>(LoadedComponent::load(path, expected));
      },
      py::arg("path"), py::arg("component_id"), py::arg("semantic_identity"),
      py::arg("manifest_identity"), py::arg("catalog_sha256"), py::arg("abi_key"),
      py::arg("interfaces"), py::arg("prepare_parameters_json"), py::arg("prepare_target_json"),
      "Load and authenticate one generated native component table exactly once.");
}
