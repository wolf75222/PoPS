#pragma once

/// @file
/// @brief Fail-closed exact registry and publication transaction for auxiliary providers.

#include <pops/runtime/system/derived_aux_provider.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
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
    PublicationTransaction(PublicationTransaction&& other) noexcept
        : registry_(other.registry_),
          point_(std::move(other.point_)),
          candidate_generation_(other.candidate_generation_),
          due_(std::move(other.due_)),
          staged_(std::move(other.staged_)),
          candidate_points_(std::move(other.candidate_points_)),
          active_(other.active_) {
      other.registry_ = nullptr;
      other.active_ = false;
    }
    PublicationTransaction& operator=(PublicationTransaction&&) = delete;

    ~PublicationTransaction() {
      if (active_)
        reject();
    }

    [[nodiscard]] std::uint64_t candidate_generation() const noexcept {
      return candidate_generation_;
    }
    [[nodiscard]] const AuxiliaryEvaluationPoint& point() const noexcept { return point_; }

    /// Whether this exact transaction requires @p provider_identity to publish a candidate.
    /// Runtimes use this authority for external InputAux/field routes so an explicitly dirtied
    /// ``once`` provider is not accidentally skipped by re-evaluating freshness policy alone.
    [[nodiscard]] bool requires_staging(std::string_view provider_identity) const {
      ensure_active_();
      return due_[registry_->provider_index_(provider_identity)];
    }

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
    /// @p storage is supplied by the integrating runtime and remains immutable in shape for the
    /// complete transaction.  The registry preserves an empty binding for contract-only tests;
    /// production runtime integration validates the binding before entering this method.
    void launch_ready_native(AuxiliaryCarrierStorage<Dim> storage = {}) {
      launch_ready_native_(storage, [](const provider_type&, std::exception_ptr error) {
        if (error)
          std::rethrow_exception(error);
      });
    }

    /// Variant for runtime-owned carrier transports. The completion hook runs on every rank after
    /// each native launch attempt, carrying any rank-local exception, and before a dependent
    /// provider can launch. The storage owner must collectively reject a local error before it
    /// enters a halo exchange; this keeps a throwing rank from stranding its peers in MPI.
    template <class Completion>
    void launch_ready_native(AuxiliaryCarrierStorage<Dim> storage, Completion&& completion) {
      launch_ready_native_(storage, std::forward<Completion>(completion));
    }

   private:
    template <class Completion>
    void launch_ready_native_(AuxiliaryCarrierStorage<Dim> storage, Completion&& completion) {
      ensure_active_();
      bool progressed = true;
      while (progressed) {
        progressed = false;
        for (const std::size_t index : registry_->topological_order_) {
          if (!due_[index] || staged_[index] || !registry_->providers_[index].has_launcher() ||
              !dependencies_ready_(index))
            continue;
          std::exception_ptr local_error;
          try {
            registry_->providers_[index].launch(
                registry_->launch_context_(index, point_, candidate_generation_, storage));
          } catch (...) {
            local_error = std::current_exception();
          }
          completion(registry_->providers_[index], local_error);
          if (local_error)
            std::rethrow_exception(local_error);
          staged_[index] = true;
          progressed = true;
        }
      }
    }

   public:
    /// Atomically make the complete candidate generation visible in registry metadata.  The
    /// integrating owner must pair this with its own storage publication after this preflight; an
    /// incomplete graph, a stale point, or any earlier exception leaves accepted state unchanged.
    void validate_complete() const {
      ensure_active_();
      for (std::size_t index = 0; index < due_.size(); ++index)
        if (due_[index] && !staged_[index])
          throw std::logic_error(
              "auxiliary candidate is incomplete: a due provider has not produced a candidate");
    }

    void accept() {
      validate_complete();
      registry_->commit_candidate_(candidate_generation_, std::move(candidate_points_));
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
          staged_(due_.size(), false),
          candidate_points_(registry.accepted_points_) {
      for (std::size_t index = 0; index < due_.size(); ++index)
        if (due_[index])
          candidate_points_[index] = point_;
    }

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
        if ((due_[producer] && !staged_[producer]) ||
            (!due_[producer] && !registry_->accepted_points_[producer]))
          return false;
      return true;
    }

    ExactAuxiliaryRegistry* registry_ = nullptr;
    AuxiliaryEvaluationPoint point_;
    std::uint64_t candidate_generation_ = 0;
    std::vector<bool> due_;
    std::vector<bool> staged_;
    std::vector<std::optional<AuxiliaryEvaluationPoint>> candidate_points_;
    bool active_ = true;
  };

  ExactAuxiliaryRegistry() = default;
  /// The registry is an immutable-value snapshot outside an active publication transaction.  Value
  /// copies are used by the System's seal/install rollback journal; an open candidate is never
  /// copied by that journal.
  ExactAuxiliaryRegistry(const ExactAuxiliaryRegistry&) = default;
  ExactAuxiliaryRegistry& operator=(const ExactAuxiliaryRegistry&) = default;

  /// Structural registration only.  It is forbidden after @ref seal, so accepted executions see
  /// one immutable topology and one exact communicator-comparable contract.
  void add(provider_type provider) {
    if (sealed_)
      throw std::logic_error("cannot add an auxiliary provider after the registry is sealed");
    providers_.push_back(std::move(provider));
  }

  /// Register one consumer-local view of globally owned provider values.  Its local slots are
  /// resolved at seal and cannot be inferred from a physics convention or from global storage order.
  void add_consumer_plan(AuxiliaryConsumerProviderPlan<Dim> plan) {
    plan.validate();
    if (candidate_open_)
      throw std::logic_error("cannot change auxiliary consumer plans during a publication");
    for (const auto& existing : consumer_plans_)
      if (existing.consumer_qid == plan.consumer_qid)
        throw std::invalid_argument("auxiliary registry contains duplicate consumer identity");
    consumer_plans_.push_back(std::move(plan));
    if (sealed_) {
      std::sort(consumer_plans_.begin(), consumer_plans_.end(),
                [](const auto& left, const auto& right) {
                  return left.consumer_qid < right.consumer_qid;
                });
      resolved_consumer_plans_.clear();
      resolved_consumer_plans_.reserve(consumer_plans_.size());
      for (const auto& candidate : consumer_plans_)
        resolved_consumer_plans_.push_back(resolve_consumer_plan_(candidate));
      rebuild_collective_contract_();
    }
  }

  /// Validate all owner/space/component keys, contracts, dense slots, producer uniqueness, and the
  /// dependency DAG.  No accepted or candidate generation exists until this method succeeds.
  void seal() {
    if (sealed_)
      throw std::logic_error("auxiliary registry is already sealed");

    std::sort(providers_.begin(), providers_.end(),
              [](const provider_type& left, const provider_type& right) {
                return left.identity() < right.identity();
              });
    std::sort(
        consumer_plans_.begin(), consumer_plans_.end(),
        [](const auto& left, const auto& right) { return left.consumer_qid < right.consumer_qid; });
    std::unordered_map<std::string, std::size_t> identity_to_provider;
    std::unordered_map<std::string, OutputLocation> output_by_key;
    for (std::size_t provider_index = 0; provider_index < providers_.size(); ++provider_index) {
      const provider_type& provider = providers_[provider_index];
      if (!identity_to_provider.emplace(provider.identity(), provider_index).second)
        throw std::invalid_argument("auxiliary registry contains a duplicate provider identity");
      for (const AuxiliaryOutput<Dim>& output : provider.outputs()) {
        output.validate();
        const std::string key = output.key.exact_key();
        if (!output_by_key.emplace(key, OutputLocation{provider_index, output, {}}).second)
          throw std::invalid_argument(
              "auxiliary registry contains duplicate producer ownership for one component key");
      }
    }
    std::map<std::string, std::vector<std::string>> keys_by_group;
    for (const auto& [key, location] : output_by_key) {
      const AuxiliaryStorageGroupKey<Dim> group{
          location.output.contract.representation, location.output.contract.centering,
          location.output.contract.layout, location.output.shape};
      keys_by_group[group.exact_key()].push_back(key);
    }
    resolved_storage_groups_.clear();
    for (auto& [group, keys] : keys_by_group) {
      std::sort(keys.begin(), keys.end());
      for (std::size_t component = 0; component < keys.size(); ++component)
        output_by_key.at(keys[component]).address = {group, component};
      const auto& sample = output_by_key.at(keys.front()).output;
      resolved_storage_groups_.push_back({group, sample.contract, sample.shape, keys.size()});
    }

    dependency_providers_.assign(providers_.size(), {});
    resolved_outputs_.assign(providers_.size(), {});
    resolved_dependencies_.assign(providers_.size(), {});
    for (std::size_t provider_index = 0; provider_index < providers_.size(); ++provider_index) {
      const provider_type& provider = providers_[provider_index];
      for (const AuxiliaryOutput<Dim>& output : provider.outputs())
        resolved_outputs_[provider_index].push_back(
            {output_by_key.at(output.key.exact_key()).address, output.key, output.contract,
             output.shape});
      for (const AuxiliaryDependency<Dim>& dependency : provider.dependencies()) {
        dependency.validate();
        const auto producer = output_by_key.find(dependency.key.exact_key());
        if (producer == output_by_key.end())
          throw std::invalid_argument(
              "auxiliary provider dependency has no registered producer for its component key");
        if (!(producer->second.output.contract == dependency.contract) ||
            !(producer->second.output.shape == dependency.shape))
          throw std::invalid_argument(
              "auxiliary provider dependency contract differs from the registered producer");
        const std::size_t producer_index = producer->second.provider_index;
        if (std::find(dependency_providers_[provider_index].begin(),
                      dependency_providers_[provider_index].end(),
                      producer_index) == dependency_providers_[provider_index].end())
          dependency_providers_[provider_index].push_back(producer_index);
        for (const auto& prior : resolved_dependencies_[provider_index])
          if (prior.key.exact_key() == dependency.key.exact_key())
            throw std::invalid_argument(
                "auxiliary provider declares the same component dependency more than once");
        resolved_dependencies_[provider_index].push_back(
            {producer->second.address, dependency.key, dependency.contract, dependency.shape});
      }
    }

    resolved_consumer_plans_.clear();
    resolved_consumer_plans_.reserve(consumer_plans_.size());
    for (const AuxiliaryConsumerProviderPlan<Dim>& plan : consumer_plans_) {
      resolved_consumer_plans_.push_back(resolve_consumer_plan_(plan));
    }

    topological_order_ = topological_order_or_throw_(dependency_providers_);
    rebuild_collective_contract_();
    accepted_points_.assign(providers_.size(), std::nullopt);
    sealed_ = true;
  }

  [[nodiscard]] bool sealed() const noexcept { return sealed_; }
  [[nodiscard]] std::size_t provider_count() const noexcept { return providers_.size(); }
  [[nodiscard]] std::size_t slot_count() const noexcept { return slot_count_(); }
  [[nodiscard]] const std::vector<ResolvedAuxiliaryStorageGroup<Dim>>& storage_groups() const {
    require_sealed_();
    return resolved_storage_groups_;
  }
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
  [[nodiscard]] const provider_type& provider_for_key(const AuxiliaryComponentKey& key) const {
    require_sealed_();
    return providers_[provider_for_key_(key)];
  }
  [[nodiscard]] std::vector<std::string> dependent_provider_identities(
      const std::vector<std::string>& provider_ids) const {
    require_sealed_();
    std::vector<bool> selected(providers_.size(), false);
    for (const std::string& identity : provider_ids)
      selected[provider_index_(identity)] = true;
    for (const std::size_t producer : topological_order_)
      if (selected[producer])
        for (std::size_t consumer = 0; consumer < dependency_providers_.size(); ++consumer)
          if (std::find(dependency_providers_[consumer].begin(),
                        dependency_providers_[consumer].end(),
                        producer) != dependency_providers_[consumer].end())
            selected[consumer] = true;
    std::vector<std::string> result;
    for (const std::size_t provider : topological_order_)
      if (selected[provider] &&
          !provider_ids_contain_(provider_ids, providers_[provider].identity()))
        result.push_back(providers_[provider].identity());
    return result;
  }
  [[nodiscard]] AuxiliaryStorageAddress<Dim> address_of(const AuxiliaryComponentKey& key) const {
    require_sealed_();
    const std::string encoded = key.exact_key();
    for (const auto& outputs : resolved_outputs_)
      for (const auto& output : outputs)
        if (output.key.exact_key() == encoded)
          return output.address;
    throw std::out_of_range("auxiliary component key is not produced by the sealed registry");
  }
  [[nodiscard]] const std::optional<AuxiliaryEvaluationPoint>& last_accepted_point(
      std::string_view provider_identity) const {
    require_sealed_();
    return accepted_points_[provider_index_(provider_identity)];
  }
  /// Immutable accepted publication provenance in canonical provider order.  Checkpoint owners
  /// persist these exact integer points together with @ref accepted_generation rather than trying
  /// to infer freshness from physical time or a carrier component number.
  [[nodiscard]] const std::vector<std::optional<AuxiliaryEvaluationPoint>>& accepted_points()
      const {
    require_sealed_();
    return accepted_points_;
  }

  /// Replace the accepted publication provenance after a checkpoint owner has restored its
  /// carrier groups.  This is deliberately unavailable while a candidate exists: a failed
  /// restart must leave the prior accepted generation and every provider point untouched.
  void restore_accepted_publication(
      std::uint64_t generation,
      std::vector<std::optional<AuxiliaryEvaluationPoint>> accepted_points) {
    require_sealed_();
    if (candidate_open_)
      throw std::logic_error(
          "cannot restore auxiliary accepted publication during a candidate generation");
    if (accepted_points.size() != providers_.size())
      throw std::invalid_argument(
          "auxiliary checkpoint accepted-point count differs from the sealed provider graph");
    for (const auto& point : accepted_points)
      if (point)
        point->validate();

    // Validate before either assignment: callers can pair this with their storage transaction and
    // retain an exact rollback image if any rank rejects a checkpoint preflight.
    accepted_points_ = std::move(accepted_points);
    accepted_generation_ = generation;
  }

  /// Allocation-free publication seam for a fully validated private registry candidate.  The
  /// caller must already have authenticated both sealed graphs and closed candidate state through
  /// its collective preflight; this operation deliberately swaps only accepted provenance.
  void swap_accepted_publication(ExactAuxiliaryRegistry& candidate) noexcept {
    static_assert(
        std::is_nothrow_swappable_v<std::vector<std::optional<AuxiliaryEvaluationPoint>>>);
    accepted_points_.swap(candidate.accepted_points_);
    std::swap(accepted_generation_, candidate.accepted_generation_);
  }

  /// Allocation-free exchange of one complete, already materialized registry image.  Native
  /// package finalization builds its candidate directly in the live System only after retaining a
  /// full snapshot; rollback uses this seam before releasing any DSO that may own provider
  /// launchers.  Every member exchange is statically required to be non-throwing.
  void swap_complete(ExactAuxiliaryRegistry& other) noexcept {
    static_assert(noexcept(providers_.swap(other.providers_)));
    static_assert(noexcept(consumer_plans_.swap(other.consumer_plans_)));
    static_assert(noexcept(dependency_providers_.swap(other.dependency_providers_)));
    static_assert(noexcept(resolved_outputs_.swap(other.resolved_outputs_)));
    static_assert(noexcept(resolved_dependencies_.swap(other.resolved_dependencies_)));
    static_assert(noexcept(resolved_storage_groups_.swap(other.resolved_storage_groups_)));
    static_assert(noexcept(resolved_consumer_plans_.swap(other.resolved_consumer_plans_)));
    static_assert(noexcept(topological_order_.swap(other.topological_order_)));
    static_assert(noexcept(accepted_points_.swap(other.accepted_points_)));
    static_assert(noexcept(collective_contract_.swap(other.collective_contract_)));
    static_assert(std::is_nothrow_swappable_v<std::uint64_t>);
    static_assert(std::is_nothrow_swappable_v<bool>);
    providers_.swap(other.providers_);
    consumer_plans_.swap(other.consumer_plans_);
    dependency_providers_.swap(other.dependency_providers_);
    resolved_outputs_.swap(other.resolved_outputs_);
    resolved_dependencies_.swap(other.resolved_dependencies_);
    resolved_storage_groups_.swap(other.resolved_storage_groups_);
    resolved_consumer_plans_.swap(other.resolved_consumer_plans_);
    topological_order_.swap(other.topological_order_);
    accepted_points_.swap(other.accepted_points_);
    collective_contract_.swap(other.collective_contract_);
    std::swap(accepted_generation_, other.accepted_generation_);
    std::swap(sealed_, other.sealed_);
    std::swap(candidate_open_, other.candidate_open_);
  }
  [[nodiscard]] const ResolvedAuxiliaryConsumerPlan<Dim>& consumer_plan(
      std::string_view consumer_qid) const {
    require_sealed_();
    for (const auto& plan : resolved_consumer_plans_)
      if (plan.consumer_qid == consumer_qid)
        return plan;
    throw std::out_of_range("auxiliary consumer plan is not registered");
  }

  /// MPI integrations compare this byte string collectively before allocating candidate storage.
  /// Keeping the comparison here as an exact API makes that requirement testable without making
  /// this header own a particular communicator implementation.
  void require_collective_contract(std::string_view peer_contract) const {
    require_sealed_();
    if (peer_contract != collective_contract_)
      throw std::runtime_error("auxiliary registry differs across MPI ranks");
  }

  [[nodiscard]] PublicationTransaction begin_publication(
      AuxiliaryEvaluationPoint point, const std::vector<std::string>& forced_provider_ids = {},
      const std::vector<std::string>& consumer_qids = {}) {
    require_sealed_();
    point.validate();
    if (candidate_open_)
      throw std::logic_error("auxiliary registry already has an unconsumed candidate generation");
    std::vector<bool> required(providers_.size(), consumer_qids.empty());
    for (const std::string& qid : consumer_qids) {
      const auto plan =
          std::find_if(resolved_consumer_plans_.begin(), resolved_consumer_plans_.end(),
                       [&](const auto& row) { return row.consumer_qid == qid; });
      if (plan == resolved_consumer_plans_.end())
        throw std::out_of_range("auxiliary consumer is not registered");
      for (const auto& value : plan->values)
        required[provider_for_key_(value.key)] = true;
    }
    std::vector<bool> forced(providers_.size(), false);
    for (const std::string& identity : forced_provider_ids) {
      const std::size_t provider = provider_index_(identity);
      forced[provider] = true;
      required[provider] = true;
    }
    for (std::size_t reverse = topological_order_.size(); reverse-- > 0;) {
      const std::size_t consumer = topological_order_[reverse];
      if (required[consumer])
        for (const std::size_t producer : dependency_providers_[consumer])
          required[producer] = true;
    }
    std::vector<bool> due;
    due.reserve(providers_.size());
    for (std::size_t index = 0; index < providers_.size(); ++index)
      due.push_back(required[index] &&
                    (forced[index] || providers_[index].policy().requires_evaluation(
                                          accepted_points_[index], point)));
    for (const std::size_t consumer : topological_order_) {
      if (!required[consumer] || providers_[consumer].kind() == AuxiliaryProviderKind::input)
        continue;
      for (const std::size_t producer : dependency_providers_[consumer])
        if (due[producer]) {
          due[consumer] = true;
          break;
        }
    }
    PublicationTransaction transaction(*this, std::move(point), accepted_generation_ + 1,
                                       std::move(due));
    candidate_open_ = true;
    return transaction;
  }

  /// Begin a publication rooted at one or more externally materialized providers.  Only those
  /// roots and their prerequisites participate. Downstream providers are marked stale separately
  /// by the integrating runtime and refresh only when an exact consumer requests them; this permits
  /// several coupled fields to publish sequentially without exposing a partially recomputed DAG.
  [[nodiscard]] PublicationTransaction begin_external_publication(
      AuxiliaryEvaluationPoint point, const std::vector<std::string>& provider_ids) {
    require_sealed_();
    point.validate();
    if (candidate_open_)
      throw std::logic_error("auxiliary registry already has an unconsumed candidate generation");
    if (provider_ids.empty())
      throw std::invalid_argument("external auxiliary publication requires a provider identity");

    std::vector<bool> forced(providers_.size(), false);
    std::vector<bool> required(providers_.size(), false);
    for (const std::string& identity : provider_ids) {
      const std::size_t provider = provider_index_(identity);
      forced[provider] = true;
      required[provider] = true;
    }
    // Pull in prerequisites of the external roots.
    for (std::size_t reverse = topological_order_.size(); reverse-- > 0;) {
      const std::size_t consumer = topological_order_[reverse];
      if (required[consumer])
        for (const std::size_t producer : dependency_providers_[consumer])
          required[producer] = true;
    }

    std::vector<bool> due(providers_.size(), false);
    for (std::size_t index = 0; index < providers_.size(); ++index)
      due[index] = required[index] &&
                   (forced[index] ||
                    providers_[index].policy().requires_evaluation(accepted_points_[index], point));
    for (const std::size_t consumer : topological_order_)
      if (required[consumer] && providers_[consumer].kind() != AuxiliaryProviderKind::input)
        for (const std::size_t producer : dependency_providers_[consumer])
          if (due[producer]) {
            due[consumer] = true;
            break;
          }

    PublicationTransaction transaction(*this, std::move(point), accepted_generation_ + 1,
                                       std::move(due));
    candidate_open_ = true;
    return transaction;
  }

 private:
  struct OutputLocation {
    std::size_t provider_index = 0;
    AuxiliaryOutput<Dim> output;
    AuxiliaryStorageAddress<Dim> address;
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

  [[nodiscard]] std::size_t provider_for_key_(const AuxiliaryComponentKey& key) const {
    const std::string encoded = key.exact_key();
    for (std::size_t provider = 0; provider < resolved_outputs_.size(); ++provider)
      for (const auto& output : resolved_outputs_[provider])
        if (output.key.exact_key() == encoded)
          return provider;
    throw std::logic_error("sealed auxiliary registry has no provider for component key");
  }

  static bool provider_ids_contain_(const std::vector<std::string>& identities,
                                    std::string_view candidate) {
    return std::find(identities.begin(), identities.end(), candidate) != identities.end();
  }

  [[nodiscard]] AuxiliaryKernelLaunchContext<Dim> launch_context_(
      std::size_t index, const AuxiliaryEvaluationPoint& point, std::uint64_t candidate_generation,
      AuxiliaryCarrierStorage<Dim> storage) const {
    return {point, candidate_generation, resolved_outputs_[index], resolved_dependencies_[index],
            storage};
  }

  [[nodiscard]] ResolvedAuxiliaryConsumerPlan<Dim> resolve_consumer_plan_(
      const AuxiliaryConsumerProviderPlan<Dim>& plan) const {
    plan.validate();
    ResolvedAuxiliaryConsumerPlan<Dim> resolved;
    resolved.consumer_qid = plan.consumer_qid;
    std::unordered_map<std::size_t, bool> slots;
    resolved.values.reserve(plan.values.size());
    for (const AuxiliaryConsumerValue<Dim>& value : plan.values) {
      const AuxiliaryDependency<Dim>& dependency = value.dependency;
      std::optional<AuxiliaryResolvedSlot<Dim>> producer;
      for (const auto& outputs : resolved_outputs_)
        for (const auto& output : outputs)
          if (output.key.exact_key() == dependency.key.exact_key()) {
            if (producer)
              throw std::logic_error("sealed auxiliary registry has duplicate output ownership");
            producer = output;
          }
      if (!producer)
        throw std::invalid_argument(
            "auxiliary consumer plan dependency has no registered producer");
      if (!(producer->contract == dependency.contract) || !(producer->shape == dependency.shape))
        throw std::invalid_argument(
            "auxiliary consumer plan dependency contract differs from its producer");
      if (!slots.emplace(value.consumer_slot, true).second)
        throw std::invalid_argument("auxiliary consumer plan has duplicate local value slot");
      resolved.values.push_back({dependency.key, dependency.contract, dependency.shape,
                                 producer->address, value.consumer_slot});
    }
    for (std::size_t slot = 0; slot < slots.size(); ++slot)
      if (!slots.contains(slot))
        throw std::invalid_argument("auxiliary consumer value slots must be compact [0, N)");
    std::sort(resolved.values.begin(), resolved.values.end(),
              [](const ResolvedAuxiliaryConsumerValue<Dim>& left,
                 const ResolvedAuxiliaryConsumerValue<Dim>& right) {
                return left.consumer_slot < right.consumer_slot;
              });
    return resolved;
  }

  void rebuild_collective_contract_() {
    ExactContractBuilder exact;
    exact.text("pops.exact-auxiliary-registry").scalar(std::uint32_t{1}).scalar(Dim);
    exact.sequence(providers_, [](ExactContractBuilder& item, const provider_type& provider) {
      item.bytes(provider.collective_contract());
    });
    exact.sequence(consumer_plans_,
                   [](ExactContractBuilder& item, const AuxiliaryConsumerProviderPlan<Dim>& plan) {
                     plan.serialize_exact(item);
                   });
    exact.sequence(topological_order_);
    collective_contract_ = std::move(exact).release();
  }

  void commit_candidate_(
      std::uint64_t generation,
      std::vector<std::optional<AuxiliaryEvaluationPoint>> candidate_points) noexcept {
    if (!candidate_open_ || generation != accepted_generation_ + 1)
      std::terminate();
    accepted_points_.swap(candidate_points);
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
  std::vector<AuxiliaryConsumerProviderPlan<Dim>> consumer_plans_;
  std::vector<std::vector<std::size_t>> dependency_providers_;
  std::vector<std::vector<AuxiliaryResolvedSlot<Dim>>> resolved_outputs_;
  std::vector<std::vector<AuxiliaryResolvedSlot<Dim>>> resolved_dependencies_;
  std::vector<ResolvedAuxiliaryStorageGroup<Dim>> resolved_storage_groups_;
  std::vector<ResolvedAuxiliaryConsumerPlan<Dim>> resolved_consumer_plans_;
  std::vector<std::size_t> topological_order_;
  std::vector<std::optional<AuxiliaryEvaluationPoint>> accepted_points_;
  std::string collective_contract_;
  std::uint64_t accepted_generation_ = 0;
  bool sealed_ = false;
  bool candidate_open_ = false;
};

}  // namespace pops::runtime::system
