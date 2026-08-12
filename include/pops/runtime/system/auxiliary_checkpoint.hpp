#pragma once

/// @file
/// @brief Durable, exact accepted-state image for auxiliary provider groups.
///
/// Version two keeps metadata and the exact accepted values in one sealed image.  Values remain
/// grouped by the storage identity resolved from ComponentKey; they are never flattened into an
/// owner-erased auxiliary array.  The enclosing AMR checkpoint contributes the block/level axis and
/// the live spatial contract contributes the valid-cell extent.

#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/system/exact_aux_registry.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::system {

template <int Dim>
struct AuxiliaryCheckpointStorageGroup final {
  std::string identity;
  AuxiliaryComponentContract contract;
  AuxiliaryStorageShape<Dim> shape;
  std::size_t component_count = 0;
  /// Component-major valid-cell payload.  The AMR facade gathers it over the prepared hierarchy
  /// lane and validates its exact size against the live level domain before publication.
  std::vector<double> payload;

  friend bool operator==(const AuxiliaryCheckpointStorageGroup&,
                         const AuxiliaryCheckpointStorageGroup&) = default;
};

template <int Dim>
struct AuxiliaryCheckpointComponent final {
  std::string provider_identity;
  AuxiliaryProviderKind provider_kind = AuxiliaryProviderKind::input;
  AuxiliaryComponentKey key;
  AuxiliaryComponentContract contract;
  AuxiliaryStorageShape<Dim> shape;
  AuxiliaryStorageAddress<Dim> address;

  friend bool operator==(const AuxiliaryCheckpointComponent& left,
                         const AuxiliaryCheckpointComponent& right) {
    return left.provider_identity == right.provider_identity &&
           left.provider_kind == right.provider_kind && left.key == right.key &&
           left.contract == right.contract && left.shape == right.shape &&
           left.address.group == right.address.group &&
           left.address.component == right.address.component;
  }
};

/// Accepted provider provenance is kept separately from each output because one provider can own
/// several ComponentKeys while it publishes at one exact evaluation point.
struct AuxiliaryCheckpointProviderPublication final {
  std::string identity;
  AuxiliaryProviderKind kind = AuxiliaryProviderKind::input;
  std::optional<AuxiliaryEvaluationPoint> accepted_point;

  friend bool operator==(const AuxiliaryCheckpointProviderPublication&,
                         const AuxiliaryCheckpointProviderPublication&) = default;
};

template <int Dim>
struct AuxiliaryCheckpointAcceptedState final {
  static_assert(Dim >= 1 && Dim <= 3, "auxiliary checkpoints support dimensions 1..3");

  std::string registry_contract;
  std::uint64_t accepted_generation = 0;
  std::vector<AuxiliaryCheckpointStorageGroup<Dim>> groups;
  std::vector<AuxiliaryCheckpointComponent<Dim>> components;
  std::vector<AuxiliaryCheckpointProviderPublication> providers;

  friend bool operator==(const AuxiliaryCheckpointAcceptedState&,
                         const AuxiliaryCheckpointAcceptedState&) = default;
};

namespace auxiliary_checkpoint_detail {

inline constexpr std::array<std::uint8_t, 8> kMagic{'P', 'O', 'P', 'S', 'A', 'U', 'X', '2'};

class Writer final {
 public:
  void raw(std::span<const std::uint8_t> bytes) {
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }
  void u64(std::uint64_t value) {
    for (int shift = 0; shift != 64; shift += 8)
      bytes_.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
  void i32(int value) { u64(static_cast<std::uint64_t>(static_cast<std::int64_t>(value))); }
  void real(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u64(bits);
  }
  void size(std::size_t value) { u64(static_cast<std::uint64_t>(value)); }
  void string(std::string_view value) {
    size(value.size());
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  [[nodiscard]] std::vector<std::uint8_t> take() && { return std::move(bytes_); }

 private:
  std::vector<std::uint8_t> bytes_;
};

class Reader final {
 public:
  explicit Reader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  void expect_raw(std::span<const std::uint8_t> expected) {
    require_(expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
      if (bytes_[cursor_ + index] != expected[index])
        fail_("unsupported magic/version");
    cursor_ += expected.size();
  }
  [[nodiscard]] std::uint64_t u64() {
    require_(sizeof(std::uint64_t));
    std::uint64_t value = 0;
    for (int shift = 0; shift != 64; shift += 8)
      value |= static_cast<std::uint64_t>(bytes_[cursor_++]) << shift;
    return value;
  }
  [[nodiscard]] int i32() {
    const std::int64_t value = static_cast<std::int64_t>(u64());
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
      fail_("integer is outside native int range");
    return static_cast<int>(value);
  }
  [[nodiscard]] double real() {
    const std::uint64_t bits = u64();
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  [[nodiscard]] std::size_t size(std::size_t element_bytes = 1) {
    const std::uint64_t value = u64();
    constexpr std::uint64_t kMaxElements = std::uint64_t{1} << 30;
    if (element_bytes == 0 || value > kMaxElements ||
        value > static_cast<std::uint64_t>((bytes_.size() - cursor_) / element_bytes))
      fail_("container length is not credible");
    return static_cast<std::size_t>(value);
  }
  [[nodiscard]] std::string string() {
    const std::size_t count = size();
    require_(count);
    std::string value(reinterpret_cast<const char*>(bytes_.data() + cursor_), count);
    cursor_ += count;
    return value;
  }
  void finish() const {
    if (cursor_ != bytes_.size())
      fail_("trailing bytes");
  }

 private:
  [[noreturn]] static void fail_(std::string_view reason) {
    throw std::runtime_error("invalid exact auxiliary checkpoint: " + std::string(reason));
  }
  void require_(std::size_t count) const {
    if (count > bytes_.size() - cursor_)
      fail_("truncated payload");
  }

  std::span<const std::uint8_t> bytes_;
  std::size_t cursor_ = 0;
};

inline void write_optional_string(Writer& out, const std::optional<std::string>& value) {
  out.u64(value ? 1U : 0U);
  if (value)
    out.string(*value);
}

inline std::optional<std::string> read_optional_string(Reader& in) {
  const std::uint64_t present = in.u64();
  if (present > 1U)
    throw std::runtime_error("invalid exact auxiliary checkpoint: invalid optional-string tag");
  return present ? std::optional<std::string>{in.string()} : std::nullopt;
}

inline void write_key(Writer& out, const AuxiliaryComponentKey& key) {
  key.validate();
  out.string(key.owner_qid);
  out.string(key.space_kind);
  out.string(key.space_name);
  out.string(key.component);
}

inline AuxiliaryComponentKey read_key(Reader& in) {
  return {in.string(), in.string(), in.string(), in.string()};
}

inline void write_contract(Writer& out, const AuxiliaryComponentContract& contract) {
  contract.validate();
  out.string(contract.representation);
  out.string(contract.centering);
  write_optional_string(out, contract.unit);
  out.string(contract.layout);
  write_optional_string(out, contract.value_kind);
}

inline AuxiliaryComponentContract read_contract(Reader& in) {
  AuxiliaryComponentContract result;
  result.representation = in.string();
  result.centering = in.string();
  result.unit = read_optional_string(in);
  result.layout = in.string();
  result.value_kind = read_optional_string(in);
  return result;
}

template <int Dim>
void write_shape(Writer& out, const AuxiliaryStorageShape<Dim>& shape) {
  shape.validate();
  out.i32(shape.spatial_rank);
  out.i32(shape.value_components);
  for (int axis = 0; axis < Dim; ++axis)
    out.i32(shape.halo[axis]);
}

template <int Dim>
AuxiliaryStorageShape<Dim> read_shape(Reader& in) {
  AuxiliaryStorageShape<Dim> result;
  result.spatial_rank = in.i32();
  result.value_components = in.i32();
  for (int axis = 0; axis < Dim; ++axis)
    result.halo[axis] = in.i32();
  return result;
}

inline void write_point(Writer& out, const AuxiliaryEvaluationPoint& point) {
  point.validate();
  out.string(point.clock);
  out.u64(point.accepted_step);
  out.u64(point.layout_generation);
  out.i32(point.level);
  out.i32(point.substep);
  out.i32(point.stage);
  out.i32(point.nonlinear_iteration);
  out.u64(static_cast<std::uint64_t>(point.event));
}

inline AuxiliaryEvaluationPoint read_point(Reader& in) {
  AuxiliaryEvaluationPoint result;
  result.clock = in.string();
  result.accepted_step = in.u64();
  result.layout_generation = in.u64();
  result.level = in.i32();
  result.substep = in.i32();
  result.stage = in.i32();
  result.nonlinear_iteration = in.i32();
  const std::uint64_t event = in.u64();
  if (event > static_cast<std::uint64_t>(AuxiliaryEvaluationEvent::output))
    throw std::runtime_error("invalid exact auxiliary checkpoint: invalid evaluation event");
  result.event = static_cast<AuxiliaryEvaluationEvent>(event);
  return result;
}

inline void write_kind(Writer& out, AuxiliaryProviderKind kind) {
  if (kind != AuxiliaryProviderKind::input && kind != AuxiliaryProviderKind::derived &&
      kind != AuxiliaryProviderKind::field_output)
    throw std::invalid_argument("auxiliary checkpoint provider kind is invalid");
  out.u64(static_cast<std::uint64_t>(kind));
}

inline AuxiliaryProviderKind read_kind(Reader& in) {
  const std::uint64_t kind = in.u64();
  if (kind > static_cast<std::uint64_t>(AuxiliaryProviderKind::field_output))
    throw std::runtime_error("invalid exact auxiliary checkpoint: invalid provider kind");
  return static_cast<AuxiliaryProviderKind>(kind);
}

inline void require_valid_kind(AuxiliaryProviderKind kind) {
  if (kind != AuxiliaryProviderKind::input && kind != AuxiliaryProviderKind::derived &&
      kind != AuxiliaryProviderKind::field_output)
    throw std::invalid_argument("auxiliary checkpoint provider kind is invalid");
}

template <int Dim>
void validate_state(const AuxiliaryCheckpointAcceptedState<Dim>& state) {
  if (state.registry_contract.empty())
    throw std::invalid_argument("auxiliary checkpoint requires a sealed registry contract");
  std::map<std::string, const AuxiliaryCheckpointStorageGroup<Dim>*> groups;
  for (const auto& group : state.groups) {
    group.contract.validate();
    group.shape.validate();
    if (group.identity.empty() || group.component_count == 0 ||
        group.component_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("auxiliary checkpoint group is incomplete");
    const AuxiliaryStorageGroupKey<Dim> expected{group.contract.representation,
                                                 group.contract.centering, group.contract.layout,
                                                 group.shape};
    if (group.identity != expected.exact_key())
      throw std::invalid_argument("auxiliary checkpoint group identity differs from its shape");
    if (!groups.emplace(group.identity, &group).second)
      throw std::invalid_argument("auxiliary checkpoint has duplicate group identity");
    for (const double value : group.payload)
      if (!std::isfinite(value))
        throw std::invalid_argument("auxiliary checkpoint group payload must be finite");
  }

  std::map<std::string, const AuxiliaryCheckpointProviderPublication*> providers;
  for (const auto& provider : state.providers) {
    if (provider.identity.empty())
      throw std::invalid_argument("auxiliary checkpoint provider identity is empty");
    require_valid_kind(provider.kind);
    if (provider.accepted_point)
      provider.accepted_point->validate();
    if (!providers.emplace(provider.identity, &provider).second)
      throw std::invalid_argument("auxiliary checkpoint has duplicate provider identity");
  }

  std::map<std::string, bool> keys;
  std::map<std::pair<std::string, std::size_t>, bool> group_components;
  for (const auto& component : state.components) {
    component.key.validate();
    component.contract.validate();
    component.shape.validate();
    const auto group = groups.find(component.address.group);
    if (component.provider_identity.empty() || group == groups.end() ||
        component.address.component >= group->second->component_count)
      throw std::invalid_argument("auxiliary checkpoint component has an invalid group address");
    if (!(component.contract == group->second->contract) ||
        !(component.shape == group->second->shape))
      throw std::invalid_argument("auxiliary checkpoint component differs from its storage group");
    const auto provider = providers.find(component.provider_identity);
    if (provider == providers.end() || provider->second->kind != component.provider_kind)
      throw std::invalid_argument("auxiliary checkpoint component has invalid provider ownership");
    if (!keys.emplace(component.key.exact_key(), true).second ||
        !group_components
             .emplace(std::make_pair(component.address.group, component.address.component), true)
             .second)
      throw std::invalid_argument("auxiliary checkpoint has duplicate component ownership");
  }
  for (const auto& [identity, group] : groups) {
    for (std::size_t component = 0; component < group->component_count; ++component)
      if (!group_components.contains({identity, component}))
        throw std::invalid_argument("auxiliary checkpoint group has an unowned component");
  }
}

}  // namespace auxiliary_checkpoint_detail

/// Export the accepted registry image.  Group payloads are deliberately staged by the caller on
/// its rank-local checkpoint path after this exact metadata image has been captured.
template <int Dim>
[[nodiscard]] AuxiliaryCheckpointAcceptedState<Dim> capture_auxiliary_checkpoint_state(
    const ExactAuxiliaryRegistry<Dim>& registry) {
  AuxiliaryCheckpointAcceptedState<Dim> state;
  state.registry_contract = std::string(registry.collective_contract());
  state.accepted_generation = registry.accepted_generation();
  for (const auto& group : registry.storage_groups())
    state.groups.push_back(
        {group.identity, group.contract, group.shape, group.component_count, {}});
  const auto& accepted_points = registry.accepted_points();
  for (std::size_t index = 0; index < registry.provider_count(); ++index) {
    const auto& provider = registry.provider(index);
    state.providers.push_back({provider.identity(), provider.kind(), accepted_points[index]});
    for (const auto& output : provider.outputs())
      state.components.push_back({provider.identity(), provider.kind(), output.key, output.contract,
                                  output.shape, registry.address_of(output.key)});
  }
  auxiliary_checkpoint_detail::validate_state(state);
  return state;
}

template <int Dim>
[[nodiscard]] std::vector<std::uint8_t> serialize_auxiliary_checkpoint_state(
    const AuxiliaryCheckpointAcceptedState<Dim>& state) {
  namespace detail = auxiliary_checkpoint_detail;
  detail::validate_state(state);
  detail::Writer out;
  out.raw(detail::kMagic);
  out.i32(Dim);
  out.string(state.registry_contract);
  out.u64(state.accepted_generation);
  out.size(state.groups.size());
  for (const auto& group : state.groups) {
    out.string(group.identity);
    detail::write_contract(out, group.contract);
    detail::write_shape(out, group.shape);
    out.size(group.component_count);
    out.size(group.payload.size());
    for (const double value : group.payload)
      out.real(value);
  }
  out.size(state.components.size());
  for (const auto& component : state.components) {
    out.string(component.provider_identity);
    detail::write_kind(out, component.provider_kind);
    detail::write_key(out, component.key);
    detail::write_contract(out, component.contract);
    detail::write_shape(out, component.shape);
    out.string(component.address.group);
    out.size(component.address.component);
  }
  out.size(state.providers.size());
  for (const auto& provider : state.providers) {
    out.string(provider.identity);
    detail::write_kind(out, provider.kind);
    out.u64(provider.accepted_point ? 1U : 0U);
    if (provider.accepted_point)
      detail::write_point(out, *provider.accepted_point);
  }
  return std::move(out).take();
}

template <int Dim>
[[nodiscard]] AuxiliaryCheckpointAcceptedState<Dim> deserialize_auxiliary_checkpoint_state(
    std::span<const std::uint8_t> bytes) {
  namespace detail = auxiliary_checkpoint_detail;
  detail::Reader in(bytes);
  in.expect_raw(detail::kMagic);
  if (in.i32() != Dim)
    throw std::runtime_error("invalid exact auxiliary checkpoint: native dimension differs");
  AuxiliaryCheckpointAcceptedState<Dim> state;
  state.registry_contract = in.string();
  state.accepted_generation = in.u64();
  state.groups.resize(in.size());
  for (auto& group : state.groups) {
    group.identity = in.string();
    group.contract = detail::read_contract(in);
    group.shape = detail::read_shape<Dim>(in);
    group.component_count = in.size();
    group.payload.resize(in.size(sizeof(double)));
    for (double& value : group.payload)
      value = in.real();
  }
  state.components.resize(in.size());
  for (auto& component : state.components) {
    component.provider_identity = in.string();
    component.provider_kind = detail::read_kind(in);
    component.key = detail::read_key(in);
    component.contract = detail::read_contract(in);
    component.shape = detail::read_shape<Dim>(in);
    component.address.group = in.string();
    component.address.component = in.size();
  }
  state.providers.resize(in.size());
  for (auto& provider : state.providers) {
    provider.identity = in.string();
    provider.kind = detail::read_kind(in);
    const std::uint64_t present = in.u64();
    if (present > 1U)
      throw std::runtime_error("invalid exact auxiliary checkpoint: invalid point tag");
    if (present)
      provider.accepted_point = detail::read_point(in);
  }
  in.finish();
  detail::validate_state(state);
  return state;
}

/// Verify a rank-local carrier allocation before a backend copies its checkpoint payload into it.
/// There is no flat auxiliary array: every physical component remains qualified by its exact group.
template <int Dim>
void require_auxiliary_checkpoint_storage(const AuxiliaryCheckpointAcceptedState<Dim>& state,
                                          const AuxiliaryStorageGroups<Dim>& storage) {
  auxiliary_checkpoint_detail::validate_state(state);
  if (storage.groups.size() != state.groups.size())
    throw std::invalid_argument("auxiliary checkpoint storage group count differs from its image");
  for (const auto& descriptor : state.groups) {
    const MultiFab<Dim>* const group = storage.find(descriptor.identity);
    if (group == nullptr || group->ncomp() != static_cast<int>(descriptor.component_count))
      throw std::invalid_argument(
          "auxiliary checkpoint storage group identity/component count differs");
    for (int axis = 0; axis < Dim; ++axis)
      if (group->ghosts()[axis] != descriptor.shape.halo[axis])
        throw std::invalid_argument(
            "auxiliary checkpoint storage group halo differs from its image");
  }
}

/// Reject a checkpoint before mutating registry provenance.  The caller must have restored the
/// rank-local group payload only into a private candidate; after this returns it may atomically
/// publish that candidate alongside the registry's accepted generation.
template <int Dim>
void restore_auxiliary_checkpoint_state(const AuxiliaryCheckpointAcceptedState<Dim>& state,
                                        ExactAuxiliaryRegistry<Dim>& registry,
                                        const ExecutionLane& lane = ExecutionLane::world()) {
  long local_preflight_failure = 0;
  std::vector<std::uint8_t> bytes;
  AuxiliaryCheckpointAcceptedState<Dim> expected;
  try {
    auxiliary_checkpoint_detail::validate_state(state);
    bytes = serialize_auxiliary_checkpoint_state(state);
    expected = capture_auxiliary_checkpoint_state(registry);
  } catch (...) {
    local_preflight_failure = 1;
  }
  if (all_reduce_max(local_preflight_failure, lane) != 0)
    throw std::invalid_argument(
        "exact auxiliary checkpoint could not be preflighted on every communicator rank");
  const std::string_view payload(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("pops.exact-auxiliary-checkpoint"), payload}}, lane))
    throw std::runtime_error("exact auxiliary checkpoint differs between communicator ranks");

  bool compatible = state.registry_contract == expected.registry_contract &&
                    state.groups.size() == expected.groups.size() &&
                    state.components == expected.components &&
                    state.providers.size() == expected.providers.size();
  for (std::size_t index = 0; index < state.groups.size() && compatible; ++index) {
    const auto& restored = state.groups[index];
    const auto& live = expected.groups[index];
    compatible = restored.identity == live.identity && restored.contract == live.contract &&
                 restored.shape == live.shape && restored.component_count == live.component_count;
  }
  std::vector<std::optional<AuxiliaryEvaluationPoint>> accepted_points;
  try {
    accepted_points.reserve(state.providers.size());
    for (const auto& provider : state.providers)
      accepted_points.push_back(provider.accepted_point);
  } catch (...) {
    local_preflight_failure = 1;
  }
  for (std::size_t index = 0; index < state.providers.size() && compatible; ++index)
    if (state.providers[index].identity != expected.providers[index].identity ||
        state.providers[index].kind != expected.providers[index].kind)
      local_preflight_failure = 1;
  if (!compatible)
    local_preflight_failure = 1;
  if (all_reduce_max(local_preflight_failure, lane) != 0)
    throw std::invalid_argument("auxiliary checkpoint differs from the sealed provider registry");
  registry.restore_accepted_publication(state.accepted_generation, std::move(accepted_points));
}

}  // namespace pops::runtime::system
