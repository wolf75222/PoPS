#pragma once

/// @file
/// @brief Fail-closed exact registry and publication transaction for auxiliary providers.

#include <pops/runtime/system/derived_aux_provider.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pops::runtime::system {

/// Immutable resolved graph plus its accepted publication history.  The registry intentionally
/// owns only contracts, slots, scheduling, and transaction state.  Storage allocation, AMR
/// transfer, MPI collectives, and Kokkos kernels remain the responsibilities of the integrating
/// runtime; that boundary prevents a metadata registry from becoming a second solver.
template <int Dim>
class ExactAuxiliaryRegistry final {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "ExactAuxiliaryRegistry supports dimensions 1, 2, and 3");

  using provider_type = PreparedAuxiliaryProvider<Dim>;

  /// One non-copyable candidate publication.  Destruction rejects an unfinished candidate, so an
  /// exception cannot make a partially produced auxiliary state visible.  A provider's launcher
  /// may write its runtime-owned candidate buffers, but those buffers are published only after
  /// every due graph node is recorded ready and @ref accept succeeds.
  class PublicationTransaction final {
   public:
    PublicationTransaction(const PublicationTransaction&) = delete;
    PublicationTransaction& operator=(const PublicationTransaction&) = delete;
    PublicationTransaction(PublicationTransaction&&) = delete;
    PublicationTransaction& operator=(PublicationTransaction&&) = delete;

    ~PublicationTransaction() {
      if (active_)
        reject();
    }

    [[nodiscard]] std::uint64_t candidate_generation() const noexcept {
      return candidate_generation_;
    }
    [[nodiscard]] const AuxiliaryEvaluationPoint& point() const noexcept { return point_; }

    /// Stage one externally prepared input/field route.  Derived routes cannot bypass their typed
    /// native launcher, and routes not due at this exact point cannot be smuggled into a candidate.
    void stage_external(std::string_view provider_identity) {
      ensure_active_();
      const std::size_t index = registry_->provider_index_(provider_identity);
      const provider_type& provider = registry_->providers_[index];
      if (provider.has_launcher())
        throw std::invalid_argument(
            "a host-launched auxiliary provider cannot be staged as an external candidate");
      stage_(index);
    }

    /// Launch every presently dependency-ready native provider in topological order.  Repeating
    /// this call after staging an external dependency is intentional and allocation-free; already
    /// staged nodes are skipped.  The registry never iterates cells or accepts a Python callback.
    void launch_ready_native() {
      ensure_active_();
      bool progressed = true;
      while (progressed) {
        progressed = false;
        for (const std::size_t index : registry_->topological_order_) {
          if (!due_[index] || staged_[index] || !registry_->providers_[index].has_launcher() ||
              !dependencies_ready_(index))
            continue;
          registry_->providers_[index].launch(
              registry_->launch_context_(index, point_, candidate_generation_));
          staged_[index] = true;
          progressed = true;
        }
      }
    }

    /// Atomically make the complete candidate generation visible in registry metadata.  The
    /// integrating owner must pair this with its own storage publication after this preflight; an
    /// incomplete graph, a stale point, or any earlier exception leaves accepted state unchanged.
    void accept() {
      ensure_active_();
      for (std::size_t index = 0; index < due_.size(); ++index)
        if (due_[index] && !staged_[index])
          throw std::logic_error(
              "auxiliary candidate is incomplete: a due provider has not produced a candidate");
      registry_->commit_candidate_(point_, candidate_generation_, due_);
      active_ = false;
    }

    /// Explicit rollback.  Candidate metadata disappears and accepted generations remain exact.
    void reject() noexcept {
      if (!active_)
        return;
      registry_->reject_candidate_();
      active_ = false;
    }

   private:
    friend class ExactAuxiliaryRegistry;

    PublicationTransaction(ExactAuxiliaryRegistry& registry, AuxiliaryEvaluationPoint point,
                           std::uint64_t candidate_generation, std::vector<bool> due)
        : registry_(&registry),
          point_(std::move(point)),
          candidate_generation_(candidate_generation),
          due_(std::move(due)),
          staged_(due_.size(), false) {}

    void ensure_active_() const {
      if (!active_)
        throw std::logic_error("auxiliary publication transaction is no longer active");
    }

    void stage_(std::size_t index) {
      if (!due_[index])
        throw std::logic_error(
            "auxiliary provider is not due at this exact evaluation point and cannot be staged");
      if (!dependencies_ready_(index))
        throw std::logic_error(
            "auxiliary provider dependency is not accepted or staged in this candidate generation");
      staged_[index] = true;
    }

    [[nodiscard]] bool dependencies_ready_(std::size_t consumer) const {
      for (const std::size_t producer : registry_->dependency_providers_[consumer])
        if (due_[producer] && !staged_[producer])
          return false;
      return true;
    }

    ExactAuxiliaryRegistry* registry_ = nullptr;
    AuxiliaryEvaluationPoint point_;
    std::uint64_t candidate_generation_ = 0;
    std::vector<bool> due_;
    std::vector<bool> staged_;
    bool active_ = true;
  };

  ExactAuxiliaryRegistry() = default;
  ExactAuxiliaryRegistry(const ExactAuxiliaryRegistry&) = delete;
  ExactAuxiliaryRegistry& operator=(const ExactAuxiliaryRegistry&) = delete;

  /// Structural registration only.  It is forbidden after @ref seal, so accepted executions see
  /// one immutable topology and one exact communicator-comparable contract.
  void add(provider_type provider) {
    if (sealed_)
      throw std::logic_error("cannot add an auxiliary provider after the registry is sealed");
    providers_.push_back(std::move(provider));
  }

  /// Validate all owner/space/component keys, contracts, dense slots, producer uniqueness, and the
  /// dependency DAG.  No accepted or candidate generation exists until this method succeeds.
  void seal() {
    if (sealed_)
      throw std::logic_error("auxiliary registry is already sealed");

    std::unordered_map<std::string, std::size_t> identity_to_provider;
    std::unordered_map<std::string, OutputLocation> output_by_key;
    std::unordered_map<std::size_t, std::string> key_by_slot;
    for (std::size_t provider_index = 0; provider_index < providers_.size(); ++provider_index) {
      const provider_type& provider = providers_[provider_index];
      if (!identity_to_provider.emplace(provider.identity(), provider_index).second)
        throw std::invalid_argument("auxiliary registry contains a duplicate provider identity");
      for (const AuxiliaryOutput<Dim>& output : provider.outputs()) {
        output.validate();
        const std::string key = output.key.exact_key();
        if (!output_by_key.emplace(key, OutputLocation{provider_index, output}).second)
          throw std::invalid_argument(
              "auxiliary registry contains duplicate producer ownership for one component key");
        if (!key_by_slot.emplace(output.slot, key).second)
          throw std::invalid_argument("auxiliary registry contains duplicate compact output slot");
      }
    }
    for (std::size_t slot = 0; slot < key_by_slot.size(); ++slot)
      if (!key_by_slot.contains(slot))
        throw std::invalid_argument("auxiliary registry output slots must be compact [0, N)");

    dependency_providers_.assign(providers_.size(), {});
    resolved_outputs_.assign(providers_.size(), {});
    resolved_dependencies_.assign(providers_.size(), {});
    for (std::size_t provider_index = 0; provider_index < providers_.size(); ++provider_index) {
      const provider_type& provider = providers_[provider_index];
      for (const AuxiliaryOutput<Dim>& output : provider.outputs())
        resolved_outputs_[provider_index].push_back({output.slot, output.key, output.contract});
      for (const AuxiliaryDependency<Dim>& dependency : provider.dependencies()) {
        dependency.validate();
        const auto producer = output_by_key.find(dependency.key.exact_key());
        if (producer == output_by_key.end())
          throw std::invalid_argument(
              "auxiliary provider dependency has no registered producer for its component key");
        if (!(producer->second.output.contract == dependency.contract))
          throw std::invalid_argument(
              "auxiliary provider dependency contract differs from the registered producer");
        const std::size_t producer_index = producer->second.provider_index;
        if (std::find(dependency_providers_[provider_index].begin(),
                      dependency_providers_[provider_index].end(),
                      producer_index) != dependency_providers_[provider_index].end())
          throw std::invalid_argument(
              "auxiliary provider declares the same producer dependency more than once");
        dependency_providers_[provider_index].push_back(producer_index);
        resolved_dependencies_[provider_index].push_back(
            {producer->second.output.slot, dependency.key, dependency.contract});
      }
    }

    topological_order_ = topological_order_or_throw_(dependency_providers_);
    ExactContractBuilder exact;
    exact.text("pops.exact-auxiliary-registry").scalar(std::uint32_t{1}).scalar(Dim);
    exact.sequence(providers_, [](ExactContractBuilder& item, const provider_type& provider) {
      item.bytes(provider.collective_contract());
    });
    exact.sequence(topological_order_);
    collective_contract_ = std::move(exact).release();
    accepted_points_.assign(providers_.size(), std::nullopt);
    sealed_ = true;
  }

  [[nodiscard]] bool sealed() const noexcept { return sealed_; }
  [[nodiscard]] std::size_t provider_count() const noexcept { return providers_.size(); }
  [[nodiscard]] std::size_t slot_count() const noexcept { return slot_count_(); }
  [[nodiscard]] std::uint64_t accepted_generation() const noexcept { return accepted_generation_; }
  [[nodiscard]] std::string_view collective_contract() const {
    require_sealed_();
    return collective_contract_;
  }
  [[nodiscard]] const std::vector<std::size_t>& topological_order() const {
    require_sealed_();
    return topological_order_;
  }
  [[nodiscard]] const provider_type& provider(std::size_t index) const {
    require_sealed_();
    if (index >= providers_.size())
      throw std::out_of_range("auxiliary provider index is outside the sealed registry");
    return providers_[index];
  }
  [[nodiscard]] std::size_t slot_of(const AuxiliaryComponentKey& key) const {
    require_sealed_();
    const std::string encoded = key.exact_key();
    for (const auto& outputs : resolved_outputs_)
      for (const auto& output : outputs)
        if (output.key.exact_key() == encoded)
          return output.slot;
    throw std::out_of_range("auxiliary component key is not produced by the sealed registry");
  }
  [[nodiscard]] const std::optional<AuxiliaryEvaluationPoint>& last_accepted_point(
      std::string_view provider_identity) const {
    require_sealed_();
    return accepted_points_[provider_index_(provider_identity)];
  }

  /// MPI integrations compare this byte string collectively before allocating candidate storage.
  /// Keeping the comparison here as an exact API makes that requirement testable without making
  /// this header own a particular communicator implementation.
  void require_collective_contract(std::string_view peer_contract) const {
    require_sealed_();
    if (peer_contract != collective_contract_)
      throw std::runtime_error("auxiliary registry differs across MPI ranks");
  }

  [[nodiscard]] PublicationTransaction begin_publication(AuxiliaryEvaluationPoint point) {
    require_sealed_();
    point.validate();
    if (candidate_open_)
      throw std::logic_error("auxiliary registry already has an unconsumed candidate generation");
    std::vector<bool> due;
    due.reserve(providers_.size());
    for (std::size_t index = 0; index < providers_.size(); ++index)
      due.push_back(providers_[index].policy().requires_evaluation(accepted_points_[index], point));
    candidate_open_ = true;
    return PublicationTransaction(*this, std::move(point), accepted_generation_ + 1,
                                  std::move(due));
  }

 private:
  struct OutputLocation {
    std::size_t provider_index = 0;
    AuxiliaryOutput<Dim> output;
  };

  static std::vector<std::size_t> topological_order_or_throw_(
      const std::vector<std::vector<std::size_t>>& dependencies) {
    std::vector<std::size_t> remaining(dependencies.size(), 0);
    std::vector<std::vector<std::size_t>> consumers(dependencies.size());
    for (std::size_t consumer = 0; consumer < dependencies.size(); ++consumer) {
      remaining[consumer] = dependencies[consumer].size();
      for (const std::size_t producer : dependencies[consumer])
        consumers[producer].push_back(consumer);
    }
    std::vector<std::size_t> ready;
    for (std::size_t index = 0; index < remaining.size(); ++index)
      if (remaining[index] == 0)
        ready.push_back(index);
    std::vector<std::size_t> ordered;
    ordered.reserve(dependencies.size());
    for (std::size_t cursor = 0; cursor < ready.size(); ++cursor) {
      const std::size_t producer = ready[cursor];
      ordered.push_back(producer);
      for (const std::size_t consumer : consumers[producer])
        if (--remaining[consumer] == 0)
          ready.push_back(consumer);
    }
    if (ordered.size() != dependencies.size())
      throw std::invalid_argument("auxiliary provider dependency graph contains a cycle");
    return ordered;
  }

  [[nodiscard]] std::size_t provider_index_(std::string_view identity) const {
    for (std::size_t index = 0; index < providers_.size(); ++index)
      if (providers_[index].identity() == identity)
        return index;
    throw std::out_of_range("auxiliary provider identity is not registered");
  }

  [[nodiscard]] AuxiliaryKernelLaunchContext<Dim> launch_context_(
      std::size_t index, const AuxiliaryEvaluationPoint& point,
      std::uint64_t candidate_generation) const {
    return {point, candidate_generation, resolved_outputs_[index], resolved_dependencies_[index]};
  }

  void commit_candidate_(const AuxiliaryEvaluationPoint& point, std::uint64_t generation,
                         const std::vector<bool>& due) {
    if (!candidate_open_ || generation != accepted_generation_ + 1)
      throw std::logic_error("auxiliary registry candidate generation is stale");
    for (std::size_t index = 0; index < due.size(); ++index)
      if (due[index])
        accepted_points_[index] = point;
    accepted_generation_ = generation;
    candidate_open_ = false;
  }

  void reject_candidate_() noexcept { candidate_open_ = false; }

  void require_sealed_() const {
    if (!sealed_)
      throw std::logic_error("auxiliary registry must be sealed before use");
  }

  [[nodiscard]] std::size_t slot_count_() const noexcept {
    std::size_t result = 0;
    for (const auto& provider : providers_)
      result += provider.outputs().size();
    return result;
  }

  std::vector<provider_type> providers_;
  std::vector<std::vector<std::size_t>> dependency_providers_;
  std::vector<std::vector<AuxiliaryResolvedSlot<Dim>>> resolved_outputs_;
  std::vector<std::vector<AuxiliaryResolvedSlot<Dim>>> resolved_dependencies_;
  std::vector<std::size_t> topological_order_;
  std::vector<std::optional<AuxiliaryEvaluationPoint>> accepted_points_;
  std::string collective_contract_;
  std::uint64_t accepted_generation_ = 0;
  bool sealed_ = false;
  bool candidate_open_ = false;
};

}  // namespace pops::runtime::system
