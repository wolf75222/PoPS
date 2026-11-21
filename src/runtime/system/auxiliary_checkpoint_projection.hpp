#pragma once

#include <pops/runtime/system/auxiliary_checkpoint.hpp>

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace pops::runtime::system {

/// A detached view of the final sealed Uniform registry. No provider executes in this view, and
/// its storage addresses never escape the checkpoint codec.
template <int Dim>
class AuxiliaryCheckpointProjection final {
 public:
  using Registry = ExactAuxiliaryRegistry<Dim>;
  using State = AuxiliaryCheckpointAcceptedState<Dim>;

  [[nodiscard]] static bool available(const Registry& live) {
    if (!live.sealed() || live.provider_count() == 0)
      return false;
    for (std::size_t index = 0; index < live.provider_count(); ++index)
      if (!live.provider(index).checkpoint_role())
        return false;
    return std::ranges::all_of(live.consumer_plans(),
                               [](const auto& plan) { return !plan.checkpoint_role.empty(); });
  }

  explicit AuxiliaryCheckpointProjection(const Registry& live) : live_(live) {
    if (!available(live))
      throw std::invalid_argument(
          "Uniform checkpoint has no installed positional auxiliary authority");
    for (std::size_t index = 0; index < live.provider_count(); ++index) {
      const auto& provider = live.provider(index);
      const auto& role = provider.checkpoint_role().value();
      if (!provider_to_role_.emplace(provider.identity(), role.identity).second ||
          !role_to_provider_.emplace(role.identity, provider.identity()).second)
        throw std::invalid_argument(
            "Uniform auxiliary checkpoint provider roles are not bijective");
      for (std::size_t slot = 0; slot < provider.outputs().size(); ++slot) {
        const auto& key = provider.outputs()[slot].key;
        auto projected = key;
        projected.owner_qid = role.output_owners[slot];
        if (!key_to_role_.emplace(key.exact_key(), projected).second ||
            !role_to_key_.emplace(projected.exact_key(), key).second)
          throw std::invalid_argument(
              "Uniform auxiliary checkpoint component roles are not bijective");
      }
    }
    const auto project_key = [this](const AuxiliaryComponentKey& key) {
      return key_to_role_.at(key.exact_key());
    };
    for (std::size_t index = 0; index < live.provider_count(); ++index)
      durable_.add(live.provider(index).checkpoint_view(project_key));
    for (auto plan : live.consumer_plans()) {
      plan.consumer_qid = std::move(plan.checkpoint_role);
      for (auto& value : plan.values)
        value.dependency.key = project_key(value.dependency.key);
      durable_.add_consumer_plan(std::move(plan));
    }
    durable_.seal();
  }

  [[nodiscard]] const Registry& durable_registry() const noexcept { return durable_; }

  void validate_durable(const State& state, std::size_t cells) const {
    require_structure_(state, durable_, cells);
  }

  [[nodiscard]] State to_durable(const State& live, std::size_t cells) const {
    return reindex_(live, live_, durable_, role_to_key_, role_to_provider_, provider_to_role_,
                    cells, true);
  }

  [[nodiscard]] State to_live(const State& durable, std::size_t cells) const {
    return reindex_(durable, durable_, live_, key_to_role_, provider_to_role_, role_to_provider_,
                    cells, false);
  }

 private:
  using KeyMap = std::map<std::string, AuxiliaryComponentKey>;
  using NameMap = std::map<std::string, std::string>;

  static void require_structure_(const State& state, const Registry& registry, std::size_t cells) {
    if (!auxiliary_checkpoint_structure_matches(state,
                                                capture_auxiliary_checkpoint_state(registry)))
      throw std::invalid_argument("auxiliary checkpoint differs from the sealed provider registry");
    for (const auto& group : state.groups)
      if (cells > std::numeric_limits<std::size_t>::max() / group.component_count ||
          group.payload.size() != cells * group.component_count)
        throw std::invalid_argument("Uniform auxiliary checkpoint group payload shape differs");
  }

  [[nodiscard]] static State reindex_(const State& source, const Registry& source_registry,
                                      const Registry& target_registry,
                                      const KeyMap& target_to_source_key,
                                      const NameMap& target_to_source_provider,
                                      const NameMap& source_to_target_provider, std::size_t cells,
                                      bool positional) {
    require_structure_(source, source_registry, cells);

    auto result = capture_auxiliary_checkpoint_state(target_registry);
    result.accepted_generation = source.accepted_generation;
    result.positional = positional;
    std::map<std::string, const AuxiliaryCheckpointStorageGroup<Dim>*> source_groups;
    std::map<std::string, AuxiliaryCheckpointStorageGroup<Dim>*> target_groups;
    for (const auto& group : source.groups)
      source_groups.emplace(group.identity, &group);
    for (auto& group : result.groups) {
      if (cells > std::numeric_limits<std::size_t>::max() / group.component_count)
        throw std::length_error("Uniform auxiliary checkpoint group payload exceeds size_t");
      group.payload.resize(cells * group.component_count);
      target_groups.emplace(group.identity, &group);
    }
    for (const auto& target : result.components) {
      const auto& key = target_to_source_key.at(target.key.exact_key());
      const auto address = source_registry.address_of(key);
      const auto& values = source_groups.at(address.group)->payload;
      auto& reordered = target_groups.at(target.address.group)->payload;
      if (cells != 0)
        std::copy_n(values.data() + address.component * cells, cells,
                    reordered.data() + target.address.component * cells);
    }

    std::map<std::string, const AuxiliaryCheckpointProviderPublication*> source_providers;
    for (const auto& provider : source.providers)
      source_providers.emplace(provider.identity, &provider);
    for (auto& target : result.providers)
      target.accepted_point =
          source_providers.at(target_to_source_provider.at(target.identity))->accepted_point;
    for (const auto& identity : source.invalidated_providers)
      result.invalidated_providers.push_back(source_to_target_provider.at(identity));
    std::ranges::sort(result.invalidated_providers);
    return result;
  }

  const Registry& live_;
  Registry durable_;
  KeyMap key_to_role_;
  KeyMap role_to_key_;
  NameMap provider_to_role_;
  NameMap role_to_provider_;
};

}  // namespace pops::runtime::system
