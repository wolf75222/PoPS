/// @file
/// @brief Multi-block ranked AMR subcycling execution authority.

#pragma once

#include <pops/numerics/time/amr/levels/amr_subcycling_plan.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_execution.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>
#include <pops/runtime/amr/prepared_multiblock_hierarchy.hpp>
#include <pops/runtime/program/program_preparation_image.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::numerics::time::amr {

/// Budgets for one exact multi-block, multi-level conservative attempt.
struct MultiBlockAmrSubcyclingBudget {
  AmrSubcyclePreparationBudget transitions{};
  ::pops::amr::reflux::FaceFluxLedgerBudget flux_ledger{};
};

/// Transactional conservative hierarchy driver over block-qualified state carriers.
///
/// The driver never exposes accepted storage as a workspace. It advances a private
/// `[block][level]` candidate matrix, stages an exact-clock parent interpolation for every child
/// interval, reconciles each block/transition ledger in finest-first order, restricts every child
/// carrier, validates the complete matrix, and only then publishes all levels.  A local callback
/// exception is reduced before another callback or collective is entered.
template <int Dim, class Payload,
          class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedMultiBlockAmrSubcyclingEngine {
 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "multi-block AMR subcycling only supports dimensions 1, 2, and 3");

  using hierarchy_type = ::pops::runtime::amr::PreparedMultiBlockAmrHierarchy<Dim, MemorySpace>;
  using runtime_type = typename hierarchy_type::engine_type;
  using field_type = typename hierarchy_type::field_type;
  using relation_type = ::pops::amr::ParentChildClockRelation;
  using ledger_type = ::pops::amr::reflux::TransactionalFaceFluxLedger<Dim, Payload>;
  using average_down_type = PreparedAverageDown<Dim, MemorySpace>;

  struct LevelAdvanceContext {
    std::size_t block = 0;
    std::string_view block_identity{};
    std::size_t level = 0;
    int substep = 0;
    std::uint64_t attempt = 0;
    ::pops::amr::ClockWindow window{};
    field_type& candidate;
    /// Parent state interpolated at `window.begin`; null only on the root level.
    const field_type* staged_parent = nullptr;
    /// Ledger for this level as a fine contributor; null on the root level.
    ledger_type* incoming_flux = nullptr;
    /// Ledger for this level as a coarse contributor; null on the finest level.
    ledger_type* outgoing_flux = nullptr;
  };

  /// Simultaneous candidate pack for exactly one level/substep callback.
  using LevelAdvanceGroup = std::span<LevelAdvanceContext>;

  struct RefluxContext {
    std::size_t block = 0;
    std::string_view block_identity{};
    std::size_t parent_level = 0;
    std::uint64_t attempt = 0;
    ::pops::amr::ClockWindow parent_window{};
    field_type& parent;
    const field_type& child;
    const ledger_type& flux;
    ::pops::amr::RefinementRatio<Dim> spatial_ratio;
    ::pops::amr::reflux::FaceRefinementMapping<Dim> face_mapping{};
  };

  struct AcceptedHistory {
    field_type older;
    field_type newer;
    ::pops::amr::ClockWindow window{};
  };

  /// Bind-resident image of the state which changes after an accepted advance.  Topology and
  /// prepared workspaces remain in the engine; rollback only copies this finite accepted image.
  struct MutableStateImage {
    std::string exact_contract;
    std::vector<std::vector<std::optional<AcceptedHistory>>> accepted_histories;
    std::vector<std::vector<std::optional<::pops::amr::ClockStamp>>> accepted_clocks;
    std::vector<std::vector<std::vector<ledger_type>>> accepted_ledgers;
    std::uint64_t next_attempt = 0;
    std::uint64_t last_accepted_attempt = 0;
  };

  PreparedMultiBlockAmrSubcyclingEngine(const PreparedMultiBlockAmrSubcyclingEngine&) = delete;
  PreparedMultiBlockAmrSubcyclingEngine& operator=(const PreparedMultiBlockAmrSubcyclingEngine&) =
      delete;
  PreparedMultiBlockAmrSubcyclingEngine(PreparedMultiBlockAmrSubcyclingEngine&&) noexcept = default;
  PreparedMultiBlockAmrSubcyclingEngine& operator=(
      PreparedMultiBlockAmrSubcyclingEngine&&) noexcept = default;

  static PreparedMultiBlockAmrSubcyclingEngine prepare(hierarchy_type& hierarchy,
                                                       std::span<const relation_type> relations,
                                                       MultiBlockAmrSubcyclingBudget budget) {
    std::exception_ptr local_error;
    std::vector<relation_type> prepared_relations;
    std::optional<PreparedAmrSubcyclePlan<Dim, MemorySpace>> spatial_plan;
    typename hierarchy_type::ProgramBlockMap map;
    std::string exact_contract;
    try {
      if (hierarchy.level_count() == 0 || relations.size() + 1 != hierarchy.level_count())
        throw std::invalid_argument(
            "multi-block AMR subcycling requires one temporal relation per transition");
      std::vector<int> temporal_counts;
      temporal_counts.reserve(relations.size());
      prepared_relations.reserve(relations.size());
      for (std::size_t transition = 0; transition < relations.size(); ++transition) {
        const relation_type& relation = relations[transition];
        if (relation.parent_level() != static_cast<int>(transition) ||
            relation.child_level() != static_cast<int>(transition + 1))
          throw std::invalid_argument(
              "multi-block AMR temporal relations are not the exact hierarchy chain");
        const auto ratio = relation.temporal_ratio();
        const std::int64_t quotient = ratio.numerator / ratio.denominator;
        const std::int64_t remainder = ratio.numerator % ratio.denominator;
        if (quotient > std::numeric_limits<int>::max() ||
            (quotient == std::numeric_limits<int>::max() && remainder != 0))
          throw std::overflow_error("multi-block AMR temporal partition exceeds int");
        temporal_counts.push_back(static_cast<int>(quotient + (remainder == 0 ? 0 : 1)));
        prepared_relations.push_back(relation);
      }
      spatial_plan.emplace(PreparedAmrSubcyclePlan<Dim, MemorySpace>::prepare(
          hierarchy.topology_runtime(), temporal_counts, budget.transitions));
      spatial_plan->require_live(hierarchy.topology_runtime());

      (void)ledger_type(budget.flux_ledger);
      std::vector<std::string> identities;
      identities.reserve(hierarchy.block_count());
      for (std::size_t block = 0; block < hierarchy.block_count(); ++block)
        identities.push_back(hierarchy.block_identity(block));
      map = hierarchy.prepare_program_block_map(identities);

      ExactContractBuilder contract;
      contract.text("pops.prepared-multiblock-amr-subcycling")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(hierarchy.collective_contract())
          .scalar(static_cast<std::uint64_t>(relations.size()));
      for (const relation_type& relation : relations)
        contract.scalar(std::int32_t{relation.parent_level()})
            .scalar(std::int32_t{relation.child_level()})
            .scalar(relation.temporal_ratio().numerator)
            .scalar(relation.temporal_ratio().denominator)
            .scalar(static_cast<std::uint8_t>(relation.remainder_policy()));
      exact_contract = std::move(contract).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(hierarchy, local_error,
                          "multi-block AMR subcycling preparation failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr-subcycling"), exact_contract}},
            hierarchy.lane()))
      throw std::invalid_argument("multi-block AMR subcycling contract differs between ranks");
    struct LiveAuthority final {
      hierarchy_type& value;
      const auto& hierarchy() const { return value.topology_runtime().hierarchy(); }
      const field_type& state(std::size_t block, std::size_t level) const {
        return value.state(block, level);
      }
      std::size_t block_count() const { return value.block_count(); }
      std::string_view block_identity(std::size_t block) const {
        return value.block_identity(block);
      }
      std::string_view collective_contract() const { return value.collective_contract(); }
      const ExecutionLane& lane() const { return value.lane(); }
      std::string_view spatial_contract() const {
        return value.topology_runtime().spatial_contract();
      }
      std::uint64_t topology_epoch() const { return value.topology_runtime().topology_epoch(); }
      std::uint64_t materialization_generation() const {
        return value.topology_runtime().materialization_generation();
      }
      hierarchy_type& eventual_owner() const { return value; }
      runtime_type& eventual_runtime() const { return value.topology_runtime(); }
      typename hierarchy_type::ProgramBlockMap prepare_program_block_map(
          std::span<const std::string> ids) const {
        return value.prepare_program_block_map(ids);
      }
    } live{hierarchy};
    return PreparedMultiBlockAmrSubcyclingEngine(live, hierarchy, std::move(prepared_relations),
                                                 std::move(*spatial_plan), budget.flux_ledger,
                                                 std::move(map), std::move(exact_contract));
  }

  template <class ForwardView>
  static PreparedMultiBlockAmrSubcyclingEngine prepare_forward(
      const ForwardView& forward, std::span<const relation_type> relations,
      MultiBlockAmrSubcyclingBudget budget) {
    std::exception_ptr local_error;
    hierarchy_type* eventual = nullptr;
    std::vector<relation_type> prepared_relations;
    std::optional<PreparedAmrSubcyclePlan<Dim, MemorySpace>> plan;
    typename hierarchy_type::ProgramBlockMap map;
    std::string exact_contract;
    try {
      eventual = std::addressof(forward.eventual_owner());
      if (eventual == nullptr || !forward.lane().active())
        throw std::invalid_argument("multi-block AMR forward topology has no eventual owner");
      if (forward.block_count() == 0 || forward.hierarchy().num_levels() == 0 ||
          relations.size() + 1 != forward.hierarchy().num_levels())
        throw std::invalid_argument("multi-block AMR forward topology has an invalid level shape");
      std::vector<int> temporal_counts;
      temporal_counts.reserve(relations.size());
      prepared_relations.reserve(relations.size());
      for (std::size_t transition = 0; transition < relations.size(); ++transition) {
        const auto& relation = relations[transition];
        if (relation.parent_level() != static_cast<int>(transition) ||
            relation.child_level() != static_cast<int>(transition + 1))
          throw std::invalid_argument("multi-block AMR forward temporal relations are not exact");
        const auto ratio = relation.temporal_ratio();
        const auto quotient = ratio.numerator / ratio.denominator;
        const auto remainder = ratio.numerator % ratio.denominator;
        if (quotient > std::numeric_limits<int>::max() ||
            (quotient == std::numeric_limits<int>::max() && remainder != 0))
          throw std::overflow_error("multi-block AMR forward temporal partition exceeds int");
        temporal_counts.push_back(static_cast<int>(quotient + (remainder == 0 ? 0 : 1)));
        prepared_relations.push_back(relation);
      }
      plan.emplace(PreparedAmrSubcyclePlan<Dim, MemorySpace>::prepare_from_hierarchy(
          forward.hierarchy(), std::string(forward.spatial_contract()), forward.topology_epoch(),
          forward.materialization_generation(), temporal_counts, budget.transitions));
      (void)ledger_type(budget.flux_ledger);
      std::vector<std::string> identities;
      identities.reserve(forward.block_count());
      for (std::size_t block = 0; block < forward.block_count(); ++block)
        identities.emplace_back(forward.block_identity(block));
      map = forward.prepare_program_block_map(identities);
      ExactContractBuilder contract;
      contract.text("pops.prepared-multiblock-amr-subcycling")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(forward.collective_contract())
          .scalar(static_cast<std::uint64_t>(relations.size()));
      for (const auto& relation : relations)
        contract.scalar(std::int32_t{relation.parent_level()})
            .scalar(std::int32_t{relation.child_level()})
            .scalar(relation.temporal_ratio().numerator)
            .scalar(relation.temporal_ratio().denominator)
            .scalar(static_cast<std::uint8_t>(relation.remainder_policy()));
      exact_contract = std::move(contract).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, forward.lane()) != 0) {
      if (forward.lane().size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("multi-block AMR forward preparation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr-subcycling"), exact_contract}},
            forward.lane()))
      throw std::invalid_argument(
          "multi-block AMR forward subcycling contract differs between ranks");
    return PreparedMultiBlockAmrSubcyclingEngine(forward, *eventual, std::move(prepared_relations),
                                                 std::move(*plan), budget.flux_ledger,
                                                 std::move(map), std::move(exact_contract));
  }

  std::string_view exact_contract() const noexcept { return exact_contract_; }
  std::uint64_t last_accepted_attempt() const noexcept { return last_accepted_attempt_; }

  const std::optional<::pops::amr::ClockStamp>& accepted_clock(std::size_t block,
                                                               std::size_t level) const {
    return last_accepted_attempt_ == 0 ? empty_clock_ : accepted_clocks_.at(block).at(level);
  }

  const std::optional<AcceptedHistory>& accepted_history(std::size_t block,
                                                         std::size_t level) const {
    return last_accepted_attempt_ == 0 ? empty_history_ : accepted_histories_.at(block).at(level);
  }

  const std::vector<ledger_type>& ledgers(std::size_t block, std::size_t parent_level) const {
    return last_accepted_attempt_ == 0 ? empty_ledgers_
                                       : accepted_ledgers_.at(block).at(parent_level);
  }

  /// Cold-only normalized parent window for one deterministic candidate-ledger invocation.
  /// Invocation is the DFS mixed-radix index accumulated by the recursive advance driver; use
  /// the same ParentChildClockRelation::partition implementation here so non-integral final
  /// children retain their exact parent endpoint before resident ledger slots are bound.
  [[nodiscard]] ::pops::amr::ClockWindow candidate_ledger_window(std::size_t parent_level,
                                                                 std::size_t invocation) const {
    if (parent_level >= relations_.size() || parent_level >= ledger_invocations_.size() ||
        invocation >= ledger_invocations_[parent_level])
      throw std::out_of_range("candidate AMR ledger window is outside its prepared invocation");

    ::pops::amr::ClockWindow window{{0, 0, {0, 1}, 0.0}, {0, 0, {1, 1}, 1.0}};
    for (std::size_t ancestor = 0; ancestor < parent_level; ++ancestor) {
      if (ancestor + 1 >= ledger_invocations_.size() || ledger_invocations_[ancestor + 1] == 0 ||
          ledger_invocations_[parent_level] % ledger_invocations_[ancestor + 1] != 0)
        throw std::logic_error("candidate AMR ledger has an invalid mixed-radix invocation plan");
      const std::size_t stride =
          ledger_invocations_[parent_level] / ledger_invocations_[ancestor + 1];
      const auto children = relations_[ancestor].partition(window);
      if (children.empty() ||
          children.size() != ledger_invocations_[ancestor + 1] / ledger_invocations_[ancestor])
        throw std::logic_error("candidate AMR ledger relation differs from its prepared plan");
      const std::size_t digit = (invocation / stride) % children.size();
      window = children[digit].window;
    }
    if (window.begin.level != static_cast<int>(parent_level) ||
        window.end.level != static_cast<int>(parent_level) ||
        !(window.begin.phase < window.end.phase) ||
        !(window.begin.physical_time < window.end.physical_time))
      throw std::logic_error("candidate AMR ledger window is not a valid parent interval");
    return window;
  }

  /// Cold-only access to the complete candidate-ledger topology.  The engine owns these ledgers
  /// for their whole prepared lifetime; consumers may prime resident slots here, but no callback
  /// can run once an attempt has begun.  Invocation is the deterministic mixed-radix subcycle
  /// slot selected by the recursive driver.
  template <class Binder>
  void bind_candidate_ledger_slots(Binder&& binder) {
    if (attempt_candidates_ != nullptr)
      throw std::logic_error("candidate ledger slots cannot bind during an active AMR attempt");
    for (std::size_t block = 0; block < candidate_ledgers_.size(); ++block)
      for (std::size_t parent = 0; parent < candidate_ledgers_[block].size(); ++parent)
        for (std::size_t invocation = 0; invocation < candidate_ledgers_[block][parent].size();
             ++invocation)
          std::forward<Binder>(binder)(block, parent, invocation,
                                       candidate_ledgers_[block][parent][invocation]);
  }

  /// Cold-only mirror for the accepted ledger image.  Static Program routes always target the
  /// candidate ledger addresses, so accepted publication copies values into this image instead
  /// of swapping the two ledger matrices after every attempt.
  void mirror_candidate_ledger_slots_into_accepted_at_bind() {
    if (attempt_candidates_ != nullptr)
      throw std::logic_error("accepted ledger slots cannot bind during an active AMR attempt");
    for (std::size_t block = 0; block < candidate_ledgers_.size(); ++block)
      for (std::size_t parent = 0; parent < candidate_ledgers_[block].size(); ++parent) {
        if (accepted_ledgers_[block][parent].size() != candidate_ledgers_[block][parent].size())
          throw std::logic_error("accepted AMR ledger image changed its prepared shape");
        for (std::size_t invocation = 0; invocation < candidate_ledgers_[block][parent].size();
             ++invocation) {
          const auto& candidate = candidate_ledgers_[block][parent][invocation];
          if (!candidate.resident_slots_bound())
            throw std::logic_error("accepted AMR ledger mirror requires resident candidate slots");
          accepted_ledgers_[block][parent][invocation] = candidate;
        }
      }
  }

  /// Cold-bind a full mutable rollback image.  Copies here are permitted only before the first
  /// accepted attempt; hot snapshot/restore uses the capacity-checked methods below.
  [[nodiscard]] MutableStateImage capture_mutable_state_at_bind() const {
    if (attempt_candidates_ != nullptr)
      throw std::logic_error("cannot capture an active AMR subcycling mutable state");
    return {exact_contract_,   accepted_histories_, accepted_clocks_,
            accepted_ledgers_, next_attempt_,       last_accepted_attempt_};
  }

  /// Dynamic storage retained by the bind-resident mutable rollback image.  The image object is
  /// embedded in its owning optional; this method therefore charges only its external string,
  /// vector layers, field payloads, and ledger arenas.  Unlike the live engine footprint, every
  /// ledger is included because the rollback image owns a distinct complete copy.
  [[nodiscard]] std::uint64_t mutable_state_image_resident_storage_bytes() const {
    const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
      if (value > std::numeric_limits<std::uint64_t>::max() - total)
        throw std::overflow_error("AMR subcycling mutable-state storage overflows uint64");
      total += value;
    };
    const auto vector_bytes = [](const auto& values) -> std::uint64_t {
      using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
      if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
        throw std::overflow_error("AMR subcycling mutable-state vector overflows uint64");
      return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
    };
    const auto external_string_bytes = [](const std::string& value) -> std::uint64_t {
      const auto begin = reinterpret_cast<std::uintptr_t>(&value);
      const auto end = begin + sizeof(value);
      const auto data = reinterpret_cast<std::uintptr_t>(value.data());
      return data >= begin && data < end ? 0 : static_cast<std::uint64_t>(value.capacity()) + 1U;
    };

    std::uint64_t total = external_string_bytes(exact_contract_);
    checked_add(total, vector_bytes(accepted_histories_));
    for (const auto& row : accepted_histories_) {
      checked_add(total, vector_bytes(row));
      for (const auto& history : row)
        if (history) {
          checked_add(total, history->older.resident_storage_bytes());
          checked_add(total, history->newer.resident_storage_bytes());
        }
    }
    checked_add(total, vector_bytes(accepted_clocks_));
    for (const auto& row : accepted_clocks_)
      checked_add(total, vector_bytes(row));
    checked_add(total, vector_bytes(accepted_ledgers_));
    for (const auto& block : accepted_ledgers_) {
      checked_add(total, vector_bytes(block));
      for (const auto& parent : block) {
        checked_add(total, vector_bytes(parent));
        for (const ledger_type& ledger : parent)
          checked_add(total, ledger.retained_storage_bytes());
      }
    }
    return total;
  }

  void require_mutable_state_image(const MutableStateImage& image) const {
    require_mutable_state_image_(image);
  }

  void copy_mutable_state_into_preallocated(MutableStateImage& destination) const {
    if (attempt_candidates_ != nullptr)
      throw std::logic_error("cannot snapshot an active AMR subcycling attempt");
    require_mutable_state_image_(destination);
    copy_history_matrix_preallocated_(destination.accepted_histories, accepted_histories_);
    destination.accepted_clocks = accepted_clocks_;
    copy_ledger_matrix_preallocated_(destination.accepted_ledgers, accepted_ledgers_);
    destination.next_attempt = next_attempt_;
    destination.last_accepted_attempt = last_accepted_attempt_;
  }

  /// Restore the same-generation mutable image in place; this never rebuilds the engine.
  void restore_mutable_state_from_preallocated(const MutableStateImage& source) {
    if (attempt_candidates_ != nullptr)
      throw std::logic_error("cannot restore an active AMR subcycling attempt");
    require_mutable_state_image_(source);
    copy_history_matrix_preallocated_(accepted_histories_, source.accepted_histories);
    accepted_clocks_ = source.accepted_clocks;
    copy_ledger_matrix_preallocated_(accepted_ledgers_, source.accepted_ledgers);
    next_attempt_ = source.next_attempt;
    last_accepted_attempt_ = source.last_accepted_attempt;
  }

  /// The private `[block][level]` candidate tower is complete before the first level-group
  /// callback. Hierarchy-scoped Program gathers must read this tower by `active_level_`, not the
  /// current group's candidate pack: that pack is one level, while the gather walks every level.
  bool has_attempt_candidates() const noexcept { return attempt_candidates_ != nullptr; }

  field_type& attempt_state(std::size_t block, std::size_t level) const {
    if (attempt_candidates_ == nullptr)
      throw std::logic_error(
          "multi-block AMR attempt state is only available during an active advance");
    if (block >= attempt_candidates_->size() || level >= (*attempt_candidates_)[block].size())
      throw std::out_of_range("multi-block AMR attempt state is outside the candidate tower");
    return (*attempt_candidates_)[block][level];
  }

  /// Four-argument advance keeps a no-op staging path so generic callers retain their
  /// existing collective sequence.
  template <class Advance, class Reflux, class Validate>
  void advance(const ::pops::amr::ClockWindow& root, Advance&& advance_level, Reflux&& reflux,
               Validate&& validate) {
    advance(root, std::forward<Advance>(advance_level), std::forward<Reflux>(reflux),
            std::forward<Validate>(validate), DefaultPublicationStage{});
  }

  /// `stage` runs after every candidate is validated and before any live hierarchy publication.
  /// The callback is generic; embedded-boundary policy belongs to the caller.
  template <class Advance, class Reflux, class Validate, class Stage>
  void advance(const ::pops::amr::ClockWindow& root, Advance&& advance_level, Reflux&& reflux,
               Validate&& validate, Stage&& stage) {
    require_live_();
    if (root.begin.level != 0 || root.end.level != 0 ||
        root.begin.macro_step != root.end.macro_step || !(root.begin.phase < root.end.phase) ||
        !(root.begin.physical_time < root.end.physical_time))
      throw std::invalid_argument("multi-block AMR root clock window is invalid");
    if (next_attempt_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("multi-block AMR subcycling attempt identity overflow");
    const std::uint64_t attempt = ++next_attempt_;

    std::exception_ptr preparation_error;
    try {
      reset_attempt_workspace_();
      for (std::size_t block = 0; block < hierarchy_->block_count(); ++block)
        for (std::size_t level = 0; level < hierarchy_->level_count(); ++level)
          copy_field_(hierarchy_->state(block, level), candidates_[block][level]);
      ::pops::device_fence(hot_fence_label_);
    } catch (...) {
      preparation_error = std::current_exception();
    }
    try {
      collectively_rethrow_(*hierarchy_, preparation_error,
                            "multi-block AMR candidate preparation failed collectively");
    } catch (...) {
      throw;
    }

    bool publication_started = false;
    struct AttemptTowerBind {
      std::vector<std::vector<field_type>>*& slot;
      explicit AttemptTowerBind(std::vector<std::vector<field_type>>*& slot,
                                std::vector<std::vector<field_type>>& tower)
          : slot(slot) {
        slot = &tower;
      }
      ~AttemptTowerBind() { slot = nullptr; }
      AttemptTowerBind(const AttemptTowerBind&) = delete;
      AttemptTowerBind& operator=(const AttemptTowerBind&) = delete;
    };
    hierarchy_->capture_resident_rollback();
    const AttemptTowerBind attempt_tower(attempt_candidates_, candidates_);
    try {
      advance_level_recursive_(0, root, 0, empty_field_pointers_, empty_ledger_pointers_, attempt,
                               advance_level, reflux);

      for (std::size_t block = 0; block < hierarchy_->block_count(); ++block)
        for (std::size_t level = 0; level < hierarchy_->level_count(); ++level)
          invoke_collectively_(
              [&] { validate(block, level, std::as_const(candidates_[block][level])); },
              "multi-block AMR candidate validation failed collectively");

      for (std::size_t level = 0; level < hierarchy_->level_count(); ++level) {
        if constexpr (!std::is_same_v<std::decay_t<Stage>, DefaultPublicationStage>) {
          invoke_collectively_([&] { stage(level, std::span<field_type*>(level_packs_[level])); },
                               "multi-block AMR candidate staging failed collectively");
        }
      }

      for (std::size_t level = 0; level < hierarchy_->level_count(); ++level) {
        publication_started = true;
        hierarchy_->publish_resident_program_candidates(program_map_, level, level_packs_[level]);
      }
      accepted_histories_.swap(candidate_histories_);
      accepted_clocks_.swap(candidate_clocks_);
      if (uses_resident_ledger_slots_()) {
        require_ledger_matrix_copy_(accepted_ledgers_, candidate_ledgers_);
        copy_ledger_matrix_preallocated_(accepted_ledgers_, candidate_ledgers_);
      } else {
        accepted_ledgers_.swap(candidate_ledgers_);
      }
      last_accepted_attempt_ = attempt;
    } catch (...) {
      const std::exception_ptr attempt_error = std::current_exception();
      if (publication_started)
        hierarchy_->restore_resident_rollback();
      std::rethrow_exception(attempt_error);
    }
  }

  /// Exact retained payload of the generic subcycling carrier, excluding all face-flux ledgers.
  /// Ledger storage is accounted by the Program flux family because one ledger can be routed by
  /// several generated expressions; charging it here would make the host receipt double-count it.
  [[nodiscard]] std::uint64_t resident_storage_bytes_excluding_flux() const {
    const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
      if (value > std::numeric_limits<std::uint64_t>::max() - total)
        throw std::overflow_error("prepared AMR subcycling resident storage overflows uint64");
      total += value;
    };
    const auto external_string_bytes = [](const std::string& value) -> std::uint64_t {
      const auto begin = reinterpret_cast<std::uintptr_t>(&value);
      const auto end = begin + sizeof(value);
      const auto data = reinterpret_cast<std::uintptr_t>(value.data());
      return data >= begin && data < end ? 0 : static_cast<std::uint64_t>(value.capacity()) + 1U;
    };
    const auto vector_bytes = [](const auto& values) -> std::uint64_t {
      using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
      if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
        throw std::overflow_error("prepared AMR subcycling vector storage overflows uint64");
      return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
    };
    const auto add_field_matrix = [&](const CandidateMatrix& matrix, std::uint64_t& total) {
      checked_add(total, vector_bytes(matrix));
      for (const auto& row : matrix) {
        checked_add(total, vector_bytes(row));
        for (const field_type& field : row)
          checked_add(total, field.resident_storage_bytes());
      }
    };
    const auto add_history_matrix = [&](const HistoryMatrix& matrix, std::uint64_t& total) {
      checked_add(total, vector_bytes(matrix));
      for (const auto& row : matrix) {
        checked_add(total, vector_bytes(row));
        for (const auto& history : row)
          if (history) {
            checked_add(total, history->older.resident_storage_bytes());
            checked_add(total, history->newer.resident_storage_bytes());
          }
      }
    };
    const auto add_pointer_matrix = [&](const auto& matrix, std::uint64_t& total) {
      checked_add(total, vector_bytes(matrix));
      for (const auto& row : matrix)
        checked_add(total, vector_bytes(row));
    };
    const auto add_ledger_container_matrix = [&](const LedgerMatrix& matrix,
                                                 bool include_unrouted_ledger_storage,
                                                 std::uint64_t& total) {
      // Flux families own every ledger's separately allocated resident slot/payload arena.  The
      // generic engine still owns these three vector layers and the ledger objects themselves,
      // including the all-cancel case where no final term retains a route to a ledger.
      checked_add(total, vector_bytes(matrix));
      for (const auto& block : matrix) {
        checked_add(total, vector_bytes(block));
        for (const auto& parent : block) {
          checked_add(total, vector_bytes(parent));
          if (include_unrouted_ledger_storage)
            for (const ledger_type& ledger : parent)
              checked_add(total, ledger.retained_storage_bytes());
        }
      }
    };
    std::uint64_t total = 0;
    checked_add(total, external_string_bytes(hierarchy_contract_));
    checked_add(total, external_string_bytes(exact_contract_));
    checked_add(total, external_string_bytes(hot_fence_label_));
    checked_add(total, vector_bytes(relations_));
    add_history_matrix(accepted_histories_, total);
    // At seal time the accepted image retains constructor retry buffers while the candidate
    // image's cold-bound slots are charged by the flux family.  They swap after acceptance, so
    // this fixed split continues to charge exactly one prepared and one unprepared image.
    add_ledger_container_matrix(accepted_ledgers_, true, total);
    add_field_matrix(candidates_, total);
    add_field_matrix(older_, total);
    add_field_matrix(staged_, total);
    add_history_matrix(candidate_histories_, total);
    add_ledger_container_matrix(candidate_ledgers_, false, total);
    checked_add(total, vector_bytes(accepted_clocks_));
    for (const auto& row : accepted_clocks_)
      checked_add(total, vector_bytes(row));
    checked_add(total, vector_bytes(candidate_clocks_));
    for (const auto& row : candidate_clocks_)
      checked_add(total, vector_bytes(row));
    checked_add(total, vector_bytes(temporal_states_));
    for (const auto& row : temporal_states_) {
      checked_add(total, vector_bytes(row));
      for (const auto& workspace : row)
        for (const auto* qualified : {&workspace.older, &workspace.newer, &workspace.target}) {
          checked_add(total, external_string_bytes(qualified->state_identity));
          checked_add(total, external_string_bytes(qualified->spatial_contract));
        }
    }
    checked_add(total, vector_bytes(block_identities_));
    for (const auto& identity : block_identities_)
      checked_add(total, external_string_bytes(identity));
    add_pointer_matrix(level_packs_, total);
    add_pointer_matrix(level_groups_, total);
    add_pointer_matrix(staged_packs_, total);
    add_pointer_matrix(outgoing_packs_, total);
    checked_add(total, vector_bytes(average_down_));
    for (const auto& row : average_down_) {
      checked_add(total, vector_bytes(row));
      for (const auto& average_down : row)
        if (average_down) {
          checked_add(total, sizeof(*average_down));
          checked_add(total, average_down->resident_storage_bytes());
        }
    }
    checked_add(total, vector_bytes(child_substeps_));
    for (const auto& row : child_substeps_)
      checked_add(total, vector_bytes(row));
    checked_add(total, vector_bytes(ledger_cursors_));
    checked_add(total, vector_bytes(ledger_invocations_));
    return total;
  }

  /// Complete retained payload of this engine when it is owned as an independent forward
  /// bundle.  Unlike the host receipt, a detached bundle has no generated flux family sharing
  /// the candidate-ledger arenas, so those arenas and the program-map authority are charged here.
  [[nodiscard]] std::uint64_t resident_storage_bytes() const {
    const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
      if (value > std::numeric_limits<std::uint64_t>::max() - total)
        throw std::overflow_error("prepared AMR subcycling resident storage overflows uint64");
      total += value;
    };
    const auto vector_bytes = [](const auto& values) -> std::uint64_t {
      using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
      if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
        throw std::overflow_error("prepared AMR subcycling vector storage overflows uint64");
      return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
    };
    const auto external_string_bytes = [](const std::string& value) -> std::uint64_t {
      const auto begin = reinterpret_cast<std::uintptr_t>(&value);
      const auto end = begin + sizeof(value);
      const auto data = reinterpret_cast<std::uintptr_t>(value.data());
      return data >= begin && data < end ? 0 : static_cast<std::uint64_t>(value.capacity()) + 1U;
    };

    std::uint64_t total = resident_storage_bytes_excluding_flux();
    // `candidate_ledgers_` is deliberately excluded from the accepted host receipt, where the
    // generated flux family owns it.  A detached PreparedSubcyclingBundle owns no such shared
    // receipt and must retain every arena itself.
    for (const auto& block : candidate_ledgers_)
      for (const auto& parent : block)
        for (const ledger_type& ledger : parent)
          checked_add(total, ledger.retained_storage_bytes());

    checked_add(total, vector_bytes(program_map_.canonical_indices));
    checked_add(total, external_string_bytes(program_map_.hierarchy_contract));
    checked_add(total, external_string_bytes(program_map_.exact_contract));
    if (program_map_.authority) {
      checked_add(total, sizeof(*program_map_.authority));
      checked_add(total, vector_bytes(program_map_.authority->canonical_indices));
      checked_add(total, external_string_bytes(program_map_.authority->hierarchy_contract));
      checked_add(total, external_string_bytes(program_map_.authority->exact_contract));
    }

    checked_add(total, spatial_plan_.resident_storage_bytes());
    return total;
  }

 private:
  struct DefaultPublicationStage {
    void operator()(std::size_t, std::span<field_type*>) const noexcept {}
  };

  using CandidateMatrix = std::vector<std::vector<field_type>>;
  using HistoryMatrix = std::vector<std::vector<std::optional<AcceptedHistory>>>;
  using ClockMatrix = std::vector<std::vector<std::optional<::pops::amr::ClockStamp>>>;
  using LedgerMatrix = std::vector<std::vector<std::vector<ledger_type>>>;

  struct TemporalStateWorkspace {
    ::pops::amr::transfer::QualifiedTemporalState older;
    ::pops::amr::transfer::QualifiedTemporalState newer;
    ::pops::amr::transfer::QualifiedTemporalState target;
  };

  using TemporalStateMatrix = std::vector<std::vector<TemporalStateWorkspace>>;

  template <class PreparationAuthority>
  PreparedMultiBlockAmrSubcyclingEngine(const PreparationAuthority& authority,
                                        hierarchy_type& hierarchy,
                                        std::vector<relation_type> relations,
                                        PreparedAmrSubcyclePlan<Dim, MemorySpace> spatial_plan,
                                        ::pops::amr::reflux::FaceFluxLedgerBudget flux_budget,
                                        typename hierarchy_type::ProgramBlockMap program_map,
                                        std::string exact_contract)
      : hierarchy_(&hierarchy),
        // The eventual runtime is an anchor used only after publication by `require_live_` and
        // execution.  Forward preparation must not query it for topology or state.
        runtime_(std::addressof(authority.eventual_runtime())),
        topology_epoch_(authority.topology_epoch()),
        materialization_generation_(authority.materialization_generation()),
        hierarchy_contract_(authority.collective_contract()),
        relations_(std::move(relations)),
        spatial_plan_(std::move(spatial_plan)),
        flux_budget_(flux_budget),
        program_map_(std::move(program_map)),
        exact_contract_(std::move(exact_contract)),
        accepted_histories_(authority.block_count()),
        accepted_clocks_(authority.block_count()),
        accepted_ledgers_(authority.block_count()),
        candidates_(authority.block_count()),
        older_(authority.block_count()),
        staged_(authority.block_count()),
        candidate_histories_(authority.block_count()),
        candidate_clocks_(authority.block_count()),
        candidate_ledgers_(authority.block_count()),
        temporal_states_(authority.block_count()),
        block_identities_(authority.block_count()),
        level_packs_(authority.hierarchy().num_levels()),
        level_groups_(authority.hierarchy().num_levels()),
        staged_packs_(relations_.size()),
        outgoing_packs_(relations_.size()),
        average_down_(relations_.size()),
        child_substeps_(relations_.size()),
        ledger_cursors_(relations_.size(), 0),
        ledger_invocations_(relations_.size(), 1) {
    const std::size_t blocks = authority.block_count();
    const std::size_t levels = authority.hierarchy().num_levels();
    for (std::size_t block = 0; block < blocks; ++block)
      block_identities_[block] = authority.block_identity(block);
    for (std::size_t parent = 0; parent < relations_.size(); ++parent) {
      const int substeps = spatial_plan_.transition(parent).temporal_substeps();
      if (substeps < 1 || ledger_invocations_[parent] > std::numeric_limits<std::size_t>::max() /
                                                            static_cast<std::size_t>(substeps))
        throw std::overflow_error("multi-block AMR resident ledger workspace exceeds size_t");
      if (parent + 1 < ledger_invocations_.size())
        ledger_invocations_[parent + 1] =
            ledger_invocations_[parent] * static_cast<std::size_t>(substeps);
      child_substeps_[parent].resize(static_cast<std::size_t>(substeps));
    }

    for (std::size_t block = 0; block < blocks; ++block) {
      accepted_histories_[block].reserve(levels);
      candidate_histories_[block].reserve(levels);
      candidates_[block].reserve(levels);
      older_[block].reserve(levels);
      staged_[block].reserve(levels);
      temporal_states_[block].resize(levels);
      accepted_clocks_[block].resize(levels);
      candidate_clocks_[block].resize(levels);
      accepted_ledgers_[block].resize(relations_.size());
      candidate_ledgers_[block].resize(relations_.size());
      for (std::size_t level = 0; level < levels; ++level) {
        const field_type& state = authority.state(block, level);
        candidates_[block].emplace_back(state);
        older_[block].emplace_back(state);
        staged_[block].emplace_back(state);
        accepted_histories_[block].emplace_back(AcceptedHistory{state, state, {}});
        candidate_histories_[block].emplace_back(AcceptedHistory{state, state, {}});
        // Optional presence is part of the bind-sealed snapshot shape.  Keep both accepted and
        // candidate clocks engaged from cold construction; `last_accepted_attempt_` remains the
        // public validity gate until a real accepted group has populated their values.
        accepted_clocks_[block][level] = {static_cast<int>(level), 0, {0, 1}, 0.0};
        candidate_clocks_[block][level] = {static_cast<int>(level), 0, {0, 1}, 0.0};
        const std::string identity = block_identities_[block] + "/level/" + std::to_string(level);
        const std::string spatial_contract(authority.spatial_contract());
        for (auto* qualified :
             {&temporal_states_[block][level].older, &temporal_states_[block][level].newer,
              &temporal_states_[block][level].target}) {
          qualified->state_identity = identity;
          qualified->spatial_contract = spatial_contract;
          qualified->topology_generation = authority.topology_epoch();
          qualified->materialization_generation = authority.materialization_generation();
        }
      }
      for (std::size_t parent = 0; parent < relations_.size(); ++parent) {
        accepted_ledgers_[block][parent].reserve(ledger_invocations_[parent]);
        candidate_ledgers_[block][parent].reserve(ledger_invocations_[parent]);
        for (std::size_t invocation = 0; invocation < ledger_invocations_[parent]; ++invocation) {
          accepted_ledgers_[block][parent].emplace_back(flux_budget_);
          candidate_ledgers_[block][parent].emplace_back(flux_budget_);
        }
      }
    }
    for (std::size_t level = 0; level < levels; ++level) {
      level_packs_[level].reserve(blocks);
      level_groups_[level].reserve(blocks);
      for (std::size_t block = 0; block < blocks; ++block) {
        level_packs_[level].push_back(&candidates_[block][level]);
        level_groups_[level].emplace_back(LevelAdvanceContext{block,
                                                              block_identities_[block],
                                                              level,
                                                              0,
                                                              0,
                                                              {},
                                                              candidates_[block][level],
                                                              nullptr,
                                                              nullptr,
                                                              nullptr});
      }
    }
    for (std::size_t parent = 0; parent < relations_.size(); ++parent) {
      staged_packs_[parent].reserve(blocks);
      outgoing_packs_[parent].resize(blocks);
      for (std::size_t block = 0; block < blocks; ++block)
        staged_packs_[parent].push_back(&staged_[block][parent]);
    }

    std::exception_ptr average_down_storage_error;
    try {
      for (std::size_t parent = 0; parent < relations_.size(); ++parent) {
        average_down_[parent].reserve(blocks);
        for (std::size_t block = 0; block < blocks; ++block)
          average_down_[parent].push_back(std::make_unique<average_down_type>());
      }
    } catch (...) {
      average_down_storage_error = std::current_exception();
    }
    collectively_rethrow_(authority.lane(), average_down_storage_error,
                          "multi-block AMR average-down storage preparation failed collectively");
    std::exception_ptr average_down_prepare_error;
    try {
      for (std::size_t parent = 0; parent < relations_.size(); ++parent)
        for (std::size_t block = 0; block < blocks; ++block)
          average_down_[parent][block]->prepare_forward(
              authority.hierarchy(), authority.eventual_runtime(), authority.topology_epoch(),
              authority.materialization_generation(), parent + 1,
              std::as_const(candidates_[block][parent + 1]), candidates_[block][parent],
              authority.lane());
    } catch (...) {
      average_down_prepare_error = std::current_exception();
    }
    collectively_rethrow_(authority.lane(), average_down_prepare_error,
                          "multi-block AMR average-down preparation failed collectively");
  }

  static void collectively_rethrow_(const hierarchy_type& hierarchy,
                                    const std::exception_ptr& local_error,
                                    std::string_view message) {
    collectively_rethrow_(hierarchy.lane(), local_error, message);
  }

  static void collectively_rethrow_(const ExecutionLane& lane,
                                    const std::exception_ptr& local_error,
                                    std::string_view message) {
    if (all_reduce_max(local_error ? 1L : 0L, lane.communicator()) == 0)
      return;
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(std::string(message));
  }

  // clang-format off
#include <pops/numerics/time/amr/levels/amr_subcycling_engine_execution.hpp>
  // clang-format on
};

}  // namespace pops::numerics::time::amr
