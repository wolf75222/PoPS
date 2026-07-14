#pragma once

#include <pops/runtime/config/generated_component_abi.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/dynlib.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pops::component {

struct RequiredNativeInterface {
  PopsNativeInterfaceIdV1 id{};
  std::uint32_t version = 1;
  std::size_t minimum_table_size = sizeof(PopsComponentTableHeaderV1);
};

struct ExpectedNativeComponent {
  std::string component_id;
  std::string semantic_identity;
  std::string manifest_identity;
  std::string catalog_sha256;
  std::vector<RequiredNativeInterface> interfaces;
  std::string prepare_parameters_json = "{}";
  std::string prepare_target_json = "{}";
};

inline std::size_t native_interface_table_size(PopsNativeInterfaceIdV1 id) {
  const auto result = generated_native_interface_table_size(id);
  if (result == 0) throw std::invalid_argument("unknown native component interface id");
  return result;
}

class LoadedComponent final {
 public:
  LoadedComponent() = default;
  LoadedComponent(const LoadedComponent&) = delete;
  LoadedComponent& operator=(const LoadedComponent&) = delete;

  LoadedComponent(LoadedComponent&& other) noexcept
      : handle_(std::exchange(other.handle_, {})), api_(std::exchange(other.api_, nullptr)),
        prepared_(std::move(other.prepared_)),
        execution_(std::move(other.execution_)),
        prepare_parameters_json_(std::move(other.prepare_parameters_json_)),
        prepare_target_json_(std::move(other.prepare_target_json_)) {
    other.prepared_.clear();
    other.execution_.reset();
  }

  LoadedComponent& operator=(LoadedComponent&& other) noexcept {
    if (this != &other) {
      reset();
      handle_ = std::exchange(other.handle_, {});
      api_ = std::exchange(other.api_, nullptr);
      prepared_ = std::move(other.prepared_);
      execution_ = std::move(other.execution_);
      prepare_parameters_json_ = std::move(other.prepare_parameters_json_);
      prepare_target_json_ = std::move(other.prepare_target_json_);
      other.prepared_.clear();
      other.execution_.reset();
    }
    return *this;
  }

  ~LoadedComponent() { reset(); }

  static LoadedComponent load(const std::string& path, const ExpectedNativeComponent& expected) {
    if (expected.component_id.empty() || expected.semantic_identity.empty() ||
        expected.manifest_identity.empty() || expected.catalog_sha256.empty() ||
        expected.interfaces.empty() || expected.prepare_parameters_json.empty() ||
        expected.prepare_target_json.empty())
      throw std::invalid_argument("native component expectation is incomplete");
    dynlib::handle handle = dynlib::open(path);
    if (!dynlib::valid(handle))
      throw std::runtime_error("cannot load native component '" + path + "': " +
                               dynlib::last_error());
    try {
      auto* raw = dynlib::sym(handle, POPS_COMPONENT_API_SYMBOL_V1);
      if (raw == nullptr)
        throw std::runtime_error("native component misses "
                                 POPS_COMPONENT_API_SYMBOL_V1);
      const auto getter = reinterpret_cast<PopsComponentApiGetterV1>(raw);
      const PopsComponentApiV1* api = getter();
      validate(api, expected);
      return LoadedComponent(handle, api, expected.prepare_parameters_json,
                             expected.prepare_target_json);
    } catch (...) {
      dynlib::close(handle);
      throw;
    }
  }

  [[nodiscard]] const PopsComponentApiV1& api() const {
    if (api_ == nullptr)
      throw std::logic_error("native component handle is empty");
    return *api_;
  }

  [[nodiscard]] const PopsComponentInterfaceEntryV1& interface(
      PopsNativeInterfaceIdV1 id, std::uint32_t version = 1) const {
    const auto& value = api();
    for (std::size_t index = 0; index < value.interface_count; ++index) {
      const auto& row = value.interfaces[index];
      if (row.interface_id == id && row.interface_version == version)
        return row;
    }
    throw std::invalid_argument("loaded component does not implement requested native interface");
  }

  template <class Table>
  [[nodiscard]] const Table& table(PopsNativeInterfaceIdV1 id,
                                   std::uint32_t version = 1) const {
    const auto& row = interface(id, version);
    if (row.table == nullptr || row.table_size < sizeof(PopsComponentTableHeaderV1))
      throw std::runtime_error("native component interface table is truncated");
    const auto* header = static_cast<const PopsComponentTableHeaderV1*>(row.table);
    // Both sizes cross the untrusted component boundary.  The entry describes the allocation
    // exposed to the loader, while the embedded header describes the initialized table prefix.
    // Trusting only row.table_size lets a header-only object advertise sizeof(Table), pass load,
    // and then be reinterpreted below past its initialized object representation.
    if (row.table_size < sizeof(Table) || header->struct_size < sizeof(Table) ||
        header->struct_size > row.table_size)
      throw std::runtime_error("native component interface table is truncated");
    return *static_cast<const Table*>(row.table);
  }

  [[nodiscard]] void* prepared_state(
      PopsNativeInterfaceIdV1 id, std::uint32_t version,
      const PopsExecutionContextV1& execution,
      std::string parameters_json = {}, std::string target_json = {}) {
    validate_execution_context(execution);
    if (parameters_json.empty()) parameters_json = prepare_parameters_json_;
    if (target_json.empty()) target_json = prepare_target_json_;
    const auto& row = interface(id, version);
    const auto* header = static_cast<const PopsComponentTableHeaderV1*>(row.table);
    if (execution_.has_value()) {
      if (!execution_->matches(execution))
        throw std::invalid_argument(
            "native component invocation context differs from its prepared resources");
    } else {
      execution_.emplace(execution);
    }
    for (auto& prepared : prepared_) {
      if (prepared.id == id && prepared.version == version &&
          prepared.parameters_json == parameters_json && prepared.target_json == target_json)
        return prepared.state;
    }
    void* state = nullptr;
    if (header->prepare != nullptr) {
      PopsComponentStatusV1 status{
          sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
      const PopsComponentPrepareRequestV1 request{
          sizeof(PopsComponentPrepareRequestV1), parameters_json.c_str(),
          target_json.c_str(), execution};
      const int code = header->prepare(&request, &state, &status);
      if (code != 0 || status.code != 0 || status.action != POPS_COMPONENT_CONTINUE_V1) {
        if (state != nullptr && header->destroy != nullptr) header->destroy(state);
        throw std::runtime_error(
            status.reason == nullptr ? "native component preparation failed" : status.reason);
      }
    }
    try {
      prepared_.push_back(PreparedInterfaceState{
          id, version, state, header->destroy, std::move(parameters_json),
          std::move(target_json)});
    } catch (...) {
      if (state != nullptr && header->destroy != nullptr) header->destroy(state);
      throw;
    }
    return state;
  }

 private:
  struct ExecutionIdentity {
    std::uint32_t context_version = 0;
    std::string execution_identity;
    PopsMemorySpaceV1 memory_space{};
    std::string backend_identity;
    std::string device_identity;
    PopsScalarTypeV1 scalar_type{};
    PopsPrecisionV1 storage_precision{};
    PopsPrecisionV1 compute_precision{};
    PopsPrecisionV1 accumulation_precision{};
    PopsPrecisionV1 reduction_precision{};
    std::uint64_t stream_handle = 0;
    std::string stream_identity;
    std::int64_t communicator_f_handle = 0;
    std::int64_t communicator_datatype_f_handle = 0;
    std::string communicator_identity;
    std::string communicator_datatype_identity;

    explicit ExecutionIdentity(const PopsExecutionContextV1& value)
        : context_version(value.context_version),
          execution_identity(
              value.execution_identity == nullptr ? "" : value.execution_identity),
          memory_space(value.memory_space),
          backend_identity(value.backend_identity == nullptr ? "" : value.backend_identity),
          device_identity(value.device_identity == nullptr ? "" : value.device_identity),
          scalar_type(value.scalar_type), storage_precision(value.storage_precision),
          compute_precision(value.compute_precision),
          accumulation_precision(value.accumulation_precision),
          reduction_precision(value.reduction_precision), stream_handle(value.stream_handle),
          stream_identity(value.stream_identity == nullptr ? "" : value.stream_identity),
          communicator_f_handle(value.communicator_f_handle),
          communicator_datatype_f_handle(value.communicator_datatype_f_handle),
          communicator_identity(
              value.communicator_identity == nullptr ? "" : value.communicator_identity),
          communicator_datatype_identity(
              value.communicator_datatype_identity == nullptr
                  ? "" : value.communicator_datatype_identity) {}

    [[nodiscard]] bool matches(const PopsExecutionContextV1& value) const {
      return value.context_version == context_version &&
             value.execution_identity != nullptr &&
             execution_identity == value.execution_identity &&
             value.memory_space == memory_space &&
             value.backend_identity != nullptr && backend_identity == value.backend_identity &&
             value.device_identity != nullptr && device_identity == value.device_identity &&
             value.scalar_type == scalar_type && value.storage_precision == storage_precision &&
             value.compute_precision == compute_precision &&
             value.accumulation_precision == accumulation_precision &&
             value.reduction_precision == reduction_precision &&
             value.stream_handle == stream_handle && value.stream_identity != nullptr &&
             stream_identity == value.stream_identity &&
             value.communicator_f_handle == communicator_f_handle &&
             value.communicator_datatype_f_handle == communicator_datatype_f_handle &&
             value.communicator_identity != nullptr &&
             communicator_identity == value.communicator_identity &&
             value.communicator_datatype_identity != nullptr &&
             communicator_datatype_identity == value.communicator_datatype_identity;
    }
  };

  struct PreparedInterfaceState {
    PopsNativeInterfaceIdV1 id{};
    std::uint32_t version = 0;
    void* state = nullptr;
    PopsComponentDestroyFnV1 destroy = nullptr;
    std::string parameters_json;
    std::string target_json;
  };

  LoadedComponent(dynlib::handle handle, const PopsComponentApiV1* api,
                  std::string prepare_parameters_json,
                  std::string prepare_target_json)
      : handle_(handle), api_(api),
        prepare_parameters_json_(std::move(prepare_parameters_json)),
        prepare_target_json_(std::move(prepare_target_json)) {}

  static std::string require_text(const char* value, std::string_view field) {
    if (value == nullptr || *value == '\0')
      throw std::runtime_error("native component table has empty " + std::string(field));
    return value;
  }

  static void validate(const PopsComponentApiV1* api,
                       const ExpectedNativeComponent& expected) {
    if (api == nullptr || api->struct_size < sizeof(PopsComponentApiV1))
      throw std::runtime_error("native component API table is null or truncated");
    if (api->protocol_abi != POPS_COMPONENT_PROTOCOL_ABI_V1)
      throw std::runtime_error("native component protocol ABI mismatch");
    if (require_text(api->catalog_sha256, "catalog_sha256") != expected.catalog_sha256 ||
        require_text(api->component_id, "component_id") != expected.component_id ||
        require_text(api->semantic_identity, "semantic_identity") !=
            expected.semantic_identity ||
        require_text(api->manifest_identity, "manifest_identity") !=
            expected.manifest_identity)
      throw std::runtime_error("native component API identity mismatch");
    if (api->interface_count == 0 || api->interfaces == nullptr)
      throw std::runtime_error("native component exports no interface table");
    std::unordered_set<std::uint64_t> expected_identities;
    for (const auto& required : expected.interfaces) {
      const std::uint64_t key =
          (static_cast<std::uint64_t>(static_cast<std::uint32_t>(required.id)) << 32u) |
          static_cast<std::uint64_t>(required.version);
      if (!expected_identities.insert(key).second)
        throw std::runtime_error(
            "native component expectation declares a duplicate interface table");
    }
    if (api->interface_count != expected_identities.size())
      throw std::runtime_error(
          "native component exported interface set differs from its manifest");
    std::unordered_set<std::uint64_t> identities;
    for (std::size_t index = 0; index < api->interface_count; ++index) {
      const auto& row = api->interfaces[index];
      const std::uint64_t key =
          (static_cast<std::uint64_t>(static_cast<std::uint32_t>(row.interface_id)) << 32u) |
          static_cast<std::uint64_t>(row.interface_version);
      if (!identities.insert(key).second)
        throw std::runtime_error("native component exports a duplicate interface table");
      if (!expected_identities.contains(key))
        throw std::runtime_error(
            "native component exports an interface table absent from its manifest");
      if (row.table == nullptr || row.table_size < sizeof(PopsComponentTableHeaderV1))
        throw std::runtime_error("native component interface table is null or truncated");
      const auto* header = static_cast<const PopsComponentTableHeaderV1*>(row.table);
      if (header->struct_size > row.table_size ||
          header->struct_size < sizeof(PopsComponentTableHeaderV1) ||
          header->abi_version != POPS_COMPONENT_PROTOCOL_ABI_V1 ||
          header->interface_id != row.interface_id ||
          header->interface_version != row.interface_version)
        throw std::runtime_error("native component interface table header mismatch");
      if ((header->prepare == nullptr) != (header->destroy == nullptr))
        throw std::runtime_error(
            "native component interface prepare/destroy callbacks must be paired");
    }
    for (const auto& required : expected.interfaces) {
      bool found = false;
      for (std::size_t index = 0; index < api->interface_count; ++index) {
        const auto& row = api->interfaces[index];
        if (row.interface_id == required.id && row.interface_version == required.version) {
          const auto* header = static_cast<const PopsComponentTableHeaderV1*>(row.table);
          if (row.table_size < required.minimum_table_size ||
              header->struct_size < required.minimum_table_size)
            throw std::runtime_error("native component required interface table is truncated");
          found = true;
          break;
        }
      }
      if (!found)
        throw std::runtime_error("native component misses a required interface table");
    }
  }

  void reset() noexcept {
    for (auto row = prepared_.rbegin(); row != prepared_.rend(); ++row)
      if (row->destroy != nullptr) row->destroy(row->state);
    prepared_.clear();
    execution_.reset();
    if (dynlib::valid(handle_))
      dynlib::close(handle_);
    handle_ = {};
    api_ = nullptr;
  }

  dynlib::handle handle_{};
  const PopsComponentApiV1* api_ = nullptr;
  std::vector<PreparedInterfaceState> prepared_;
  std::optional<ExecutionIdentity> execution_;
  std::string prepare_parameters_json_ = "{}";
  std::string prepare_target_json_ = "{}";
};

}  // namespace pops::component
