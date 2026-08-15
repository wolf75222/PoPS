/// @file
/// @brief Multi-block ranked AMR subcycling execution authority.

#pragma once

#include <pops/numerics/time/amr/levels/amr_subcycling_plan.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>
#include <pops/runtime/amr/prepared_multiblock_hierarchy.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
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
    return PreparedMultiBlockAmrSubcyclingEngine(hierarchy, std::move(prepared_relations),
                                                 std::move(*spatial_plan), budget.flux_ledger,
                                                 std::move(map), std::move(exact_contract));
  }

  std::string_view exact_contract() const noexcept { return exact_contract_; }
  std::uint64_t last_accepted_attempt() const noexcept { return last_accepted_attempt_; }

  const std::optional<::pops::amr::ClockStamp>& accepted_clock(std::size_t block,
                                                               std::size_t level) const {
    return accepted_clocks_.at(block).at(level);
  }

  const std::optional<AcceptedHistory>& accepted_history(std::size_t block,
                                                         std::size_t level) const {
    return accepted_histories_.at(block).at(level);
  }

  const std::vector<ledger_type>& ledgers(std::size_t block, std::size_t parent_level) const {
    return accepted_ledgers_.at(block).at(parent_level);
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

    const auto accepted_snapshot = hierarchy_->snapshot();
    CandidateMatrix candidates;
    HistoryMatrix candidate_histories;
    ClockMatrix candidate_clocks;
    LedgerMatrix candidate_ledgers(hierarchy_->block_count());
    std::exception_ptr preparation_error;
    try {
      candidates.resize(hierarchy_->block_count());
      candidate_histories.resize(hierarchy_->block_count());
      candidate_clocks = accepted_clocks_;
      for (std::size_t block = 0; block < hierarchy_->block_count(); ++block) {
        candidate_ledgers[block].resize(relations_.size());
        candidates[block].reserve(hierarchy_->level_count());
        candidate_histories[block] = accepted_histories_[block];
        for (std::size_t level = 0; level < hierarchy_->level_count(); ++level)
          candidates[block].emplace_back(hierarchy_->state(block, level));
      }
      Kokkos::fence();
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
    try {
      std::vector<const field_type*> no_parent;
      std::vector<ledger_type*> no_incoming_flux;
      advance_level_recursive_(0, root, 0, no_parent, no_incoming_flux, candidates,
                               candidate_histories, candidate_clocks, candidate_ledgers, attempt,
                               advance_level, reflux);

      for (std::size_t block = 0; block < hierarchy_->block_count(); ++block)
        for (std::size_t level = 0; level < hierarchy_->level_count(); ++level)
          invoke_collectively_(
              [&] { validate(block, level, std::as_const(candidates[block][level])); },
              "multi-block AMR candidate validation failed collectively");

      std::vector<std::vector<field_type*>> packs(hierarchy_->level_count());
      for (std::size_t level = 0; level < hierarchy_->level_count(); ++level) {
        packs[level].reserve(hierarchy_->block_count());
        for (std::size_t block = 0; block < hierarchy_->block_count(); ++block)
          packs[level].push_back(&candidates[block][level]);
        if constexpr (!std::is_same_v<std::decay_t<Stage>, DefaultPublicationStage>) {
          invoke_collectively_([&] { stage(level, std::span<field_type*>(packs[level])); },
                               "multi-block AMR candidate staging failed collectively");
        }
      }

      for (std::size_t level = 0; level < hierarchy_->level_count(); ++level) {
        publication_started = true;
        hierarchy_->publish_program_candidates(program_map_, level, packs[level]);
      }
      accepted_histories_.swap(candidate_histories);
      accepted_clocks_.swap(candidate_clocks);
      accepted_ledgers_.swap(candidate_ledgers);
      last_accepted_attempt_ = attempt;
    } catch (...) {
      const std::exception_ptr attempt_error = std::current_exception();
      if (publication_started)
        hierarchy_->restore(accepted_snapshot);
      std::rethrow_exception(attempt_error);
    }
  }

 private:
  struct DefaultPublicationStage {
    void operator()(std::size_t, std::span<field_type*>) const noexcept {}
  };

  using CandidateMatrix = std::vector<std::vector<field_type>>;
  using HistoryMatrix = std::vector<std::vector<std::optional<AcceptedHistory>>>;
  using ClockMatrix = std::vector<std::vector<std::optional<::pops::amr::ClockStamp>>>;
  using LedgerMatrix = std::vector<std::vector<std::vector<ledger_type>>>;

  PreparedMultiBlockAmrSubcyclingEngine(hierarchy_type& hierarchy,
                                        std::vector<relation_type> relations,
                                        PreparedAmrSubcyclePlan<Dim, MemorySpace> spatial_plan,
                                        ::pops::amr::reflux::FaceFluxLedgerBudget flux_budget,
                                        typename hierarchy_type::ProgramBlockMap program_map,
                                        std::string exact_contract)
      : hierarchy_(&hierarchy),
        hierarchy_contract_(hierarchy.collective_contract()),
        relations_(std::move(relations)),
        spatial_plan_(std::move(spatial_plan)),
        flux_budget_(flux_budget),
        program_map_(std::move(program_map)),
        exact_contract_(std::move(exact_contract)),
        accepted_histories_(hierarchy.block_count()),
        accepted_clocks_(hierarchy.block_count()),
        accepted_ledgers_(hierarchy.block_count()) {
    for (std::size_t block = 0; block < hierarchy.block_count(); ++block) {
      accepted_histories_[block].resize(hierarchy.level_count());
      accepted_clocks_[block].resize(hierarchy.level_count());
      accepted_ledgers_[block].resize(relations_.size());
    }
  }

  static void collectively_rethrow_(const hierarchy_type& hierarchy,
                                    const std::exception_ptr& local_error,
                                    std::string_view message) {
    if (all_reduce_max(local_error ? 1L : 0L, hierarchy.lane().communicator()) == 0)
      return;
    if (hierarchy.lane().size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(std::string(message));
  }

  template <class Callback>
  void invoke_collectively_(Callback&& callback, std::string_view message) const {
    enum class ExceptionKind : long { None = 0, StepRejected = 1, Ordinary = 2 };
    ExceptionKind kind = ExceptionKind::None;
    std::string rejection_payload;
    std::exception_ptr local_error;
    try {
      callback();
    } catch (const ::pops::runtime::program::StepAttemptRejected& rejected) {
      try {
        rejection_payload = encode_step_rejection_(rejected);
        kind = ExceptionKind::StepRejected;
      } catch (...) {
        kind = ExceptionKind::Ordinary;
        local_error = std::current_exception();
      }
    } catch (...) {
      kind = ExceptionKind::Ordinary;
      local_error = std::current_exception();
    }
    try {
      Kokkos::fence();
    } catch (...) {
      kind = ExceptionKind::Ordinary;
      local_error = std::current_exception();
    }

    const auto communicator = hierarchy_->lane().communicator();
    const long ordinary = kind == ExceptionKind::Ordinary ? 1L : 0L;
    const long rejected = kind == ExceptionKind::StepRejected ? 1L : 0L;
    if (all_reduce_max(ordinary, communicator) != 0) {
      if (hierarchy_->lane().size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error(std::string(message));
    }
    if (all_reduce_max(rejected, communicator) == 0)
      return;

    // When every rank rejected, authenticate the complete typed envelope exactly. When only a
    // subset rejected, the first rejecting rank is the deterministic control authority. Its byte
    // envelope is broadcast with reductions on the lane communicator so every participant follows
    // the same collective sequence without requiring ownership of the communicator observer.
    std::string selected_payload;
    if (all_reduce_min(rejected, communicator) != 0) {
      if (!all_ranks_agree_exact_ordered_byte_pairs({{"step-rejection", rejection_payload}},
                                                    communicator))
        throw std::runtime_error("collective step rejection fields differ between ranks");
      selected_payload = std::move(rejection_payload);
    } else {
      const long local_root = rejected != 0 ? static_cast<long>(hierarchy_->lane().rank())
                                            : static_cast<long>(hierarchy_->lane().size());
      const long root = all_reduce_min(local_root, communicator);
      if (root < 0 || root >= static_cast<long>(hierarchy_->lane().size()))
        throw std::runtime_error("collective step rejection lost its typed envelope");

      const bool authoritative = hierarchy_->lane().rank() == root;
      const long invalid_length =
          authoritative && rejection_payload.size() >
                               static_cast<std::size_t>(std::numeric_limits<long>::max())
              ? 1L
              : 0L;
      if (all_reduce_max(invalid_length, communicator) != 0)
        throw std::length_error("collective step rejection envelope exceeds long capacity");
      const long encoded_length = all_reduce_max(
          authoritative ? static_cast<long>(rejection_payload.size()) : 0L, communicator);
      if (encoded_length <= 0)
        throw std::runtime_error("collective step rejection envelope is empty");

      long allocation_failed = 0;
      try {
        if (authoritative)
          selected_payload = rejection_payload;
        selected_payload.resize(static_cast<std::size_t>(encoded_length));
      } catch (...) {
        allocation_failed = 1;
      }
      if (all_reduce_max(allocation_failed, communicator) != 0)
        throw std::bad_alloc();
      broadcast_bytes_inplace(selected_payload.data(), selected_payload.size(),
                              static_cast<int>(root), communicator);
      const long typed_envelope_mismatch =
          rejected != 0 && rejection_payload != selected_payload ? 1L : 0L;
      if (all_reduce_max(typed_envelope_mismatch, communicator) != 0)
        throw std::runtime_error("collective step rejection fields differ between rejecting ranks");
    }
    const StepRejectionEnvelope envelope = decode_step_rejection_(selected_payload);
    throw ::pops::runtime::program::StepAttemptRejected(envelope.status, envelope.disposition,
                                                        envelope.reason_code, envelope.phase,
                                                        envelope.detail);
  }

  struct StepRejectionEnvelope {
    SolveStatus status = SolveStatus::kInvalidInput;
    ::pops::runtime::program::StepAttemptDisposition disposition =
        ::pops::runtime::program::StepAttemptDisposition::kReject;
    std::uint32_t reason_code = 0;
    std::string phase;
    std::string detail;
  };

  static void append_u64_(std::string& bytes, std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
      bytes.push_back(static_cast<char>((value >> shift) & 0xffu));
  }

  static std::uint64_t read_u64_(std::string_view bytes, std::size_t& cursor) {
    if (cursor > bytes.size() || bytes.size() - cursor < 8)
      throw std::runtime_error("collective step rejection envelope is truncated");
    std::uint64_t value = 0;
    for (int byte = 0; byte < 8; ++byte)
      value = (value << 8u) | static_cast<unsigned char>(bytes[cursor++]);
    return value;
  }

  static void append_text_(std::string& bytes, std::string_view value) {
    append_u64_(bytes, static_cast<std::uint64_t>(value.size()));
    bytes.append(value.data(), value.size());
  }

  static std::string read_text_(std::string_view bytes, std::size_t& cursor) {
    const std::uint64_t encoded_size = read_u64_(bytes, cursor);
    if (encoded_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      throw std::overflow_error("collective step rejection text exceeds size_t");
    const std::size_t size = static_cast<std::size_t>(encoded_size);
    if (cursor > bytes.size() || size > bytes.size() - cursor)
      throw std::runtime_error("collective step rejection text is truncated");
    std::string value(bytes.substr(cursor, size));
    cursor += size;
    return value;
  }

  static std::string encode_step_rejection_(
      const ::pops::runtime::program::StepAttemptRejected& rejected) {
    std::string bytes("pops.step-rejection.v1");
    append_u64_(bytes, static_cast<std::uint64_t>(rejected.status()));
    append_u64_(bytes, static_cast<std::uint64_t>(rejected.disposition()));
    append_u64_(bytes, rejected.reason_code());
    append_text_(bytes, rejected.phase());
    append_text_(bytes, rejected.detail());
    return bytes;
  }

  static StepRejectionEnvelope decode_step_rejection_(std::string_view bytes) {
    constexpr std::string_view prefix = "pops.step-rejection.v1";
    if (!bytes.starts_with(prefix))
      throw std::runtime_error("collective step rejection envelope has another schema");
    std::size_t cursor = prefix.size();
    const std::uint64_t status = read_u64_(bytes, cursor);
    const std::uint64_t disposition = read_u64_(bytes, cursor);
    const std::uint64_t reason_code = read_u64_(bytes, cursor);
    if (status > static_cast<std::uint64_t>(SolveStatus::kSafeguardFailure) ||
        disposition >
            static_cast<std::uint64_t>(::pops::runtime::program::StepAttemptDisposition::kReject) ||
        reason_code > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("collective step rejection envelope has invalid enum fields");
    StepRejectionEnvelope result;
    result.status = static_cast<SolveStatus>(status);
    result.disposition = static_cast<::pops::runtime::program::StepAttemptDisposition>(disposition);
    result.reason_code = static_cast<std::uint32_t>(reason_code);
    result.phase = read_text_(bytes, cursor);
    result.detail = read_text_(bytes, cursor);
    if (cursor != bytes.size())
      throw std::runtime_error("collective step rejection envelope has trailing bytes");
    return result;
  }

  void require_live_() const {
    std::exception_ptr local_error;
    try {
      if (hierarchy_ == nullptr || hierarchy_->collective_contract() != hierarchy_contract_ ||
          hierarchy_->block_count() != accepted_ledgers_.size() ||
          hierarchy_->level_count() != relations_.size() + 1)
        throw std::invalid_argument("prepared multi-block AMR subcycling engine is stale");
      spatial_plan_.require_live(hierarchy_->topology_runtime());
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(*hierarchy_, local_error,
                          "prepared multi-block AMR subcycling liveness failed collectively");
  }

  std::vector<field_type> stage_parent_(std::size_t parent_level,
                                        const ::pops::amr::ClockWindow& parent_window,
                                        const ::pops::amr::ClockStamp& target,
                                        const std::vector<field_type>& older,
                                        const CandidateMatrix& candidates) const {
    std::vector<field_type> staged;
    std::exception_ptr local_error;
    try {
      staged.reserve(hierarchy_->block_count());
      for (std::size_t block = 0; block < hierarchy_->block_count(); ++block) {
        staged.emplace_back(older[block]);
        const std::string state_identity =
            hierarchy_->block_identity(block) + "/level/" + std::to_string(parent_level);
        const auto qualify = [&](const ::pops::amr::ClockStamp& clock) {
          return ::pops::amr::transfer::QualifiedTemporalState{
              state_identity, std::string(hierarchy_->topology_runtime().spatial_contract()),
              hierarchy_->topology_runtime().topology_epoch(),
              hierarchy_->topology_runtime().materialization_generation(), clock};
        };
        ::pops::amr::ClockStamp parent_target = target;
        parent_target.level = static_cast<int>(parent_level);
        for (std::size_t local = 0; local < staged.back().local_size(); ++local) {
          const auto prepared = prepare_linear_time_interpolation(
              hierarchy_->topology_runtime(), parent_level,
              std::as_const(older[block].fab(local)).view(),
              std::as_const(candidates[block][parent_level].fab(local)).view(),
              staged.back().fab(local).view(), staged.back().box(local),
              qualify(parent_window.begin), qualify(parent_window.end), qualify(parent_target),
              {0, 0, 0, staged.back().ncomp()});
          execute_prepared_transfer(prepared);
        }
      }
      Kokkos::fence();
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(*hierarchy_, local_error,
                          "multi-block AMR parent-time interpolation failed collectively");
    return staged;
  }

  template <class Advance, class Reflux>
  void advance_level_recursive_(std::size_t level, const ::pops::amr::ClockWindow& window,
                                int substep, const std::vector<const field_type*>& staged_parent,
                                const std::vector<ledger_type*>& incoming_flux,
                                CandidateMatrix& candidates, HistoryMatrix& histories,
                                ClockMatrix& clocks, LedgerMatrix& candidate_ledgers,
                                std::uint64_t attempt, Advance& advance_level, Reflux& reflux) {
    std::vector<field_type> older;
    older.reserve(hierarchy_->block_count());
    for (std::size_t block = 0; block < hierarchy_->block_count(); ++block)
      older.emplace_back(candidates[block][level]);

    std::vector<ledger_type> outgoing_flux;
    if (level < relations_.size()) {
      outgoing_flux.reserve(hierarchy_->block_count());
      for (std::size_t block = 0; block < hierarchy_->block_count(); ++block) {
        outgoing_flux.emplace_back(flux_budget_);
        outgoing_flux.back().begin(attempt);
      }
    }
    std::vector<ledger_type*> outgoing_views;
    outgoing_views.reserve(outgoing_flux.size());
    for (ledger_type& current : outgoing_flux)
      outgoing_views.push_back(&current);

    std::vector<LevelAdvanceContext> group;
    group.reserve(hierarchy_->block_count());
    for (std::size_t block = 0; block < hierarchy_->block_count(); ++block)
      group.push_back(LevelAdvanceContext{
          block, hierarchy_->block_identity(block), level, substep, attempt, window,
          candidates[block][level], level == 0 ? nullptr : staged_parent.at(block),
          level == 0 ? nullptr : incoming_flux.at(block),
          level == relations_.size() ? nullptr : &outgoing_flux[block]});

    invoke_collectively_([&] { advance_level(LevelAdvanceGroup(group)); },
                         "multi-block AMR level-group callback failed collectively");
    for (std::size_t block = 0; block < hierarchy_->block_count(); ++block) {
      histories[block][level].emplace(
          AcceptedHistory{older[block], field_type(candidates[block][level]), window});
      clocks[block][level] = window.end;
    }

    if (level == relations_.size())
      return;
    std::vector<::pops::amr::ChildSubstep> children;
    std::exception_ptr partition_error;
    try {
      children = relations_[level].partition(window);
    } catch (...) {
      partition_error = std::current_exception();
    }
    collectively_rethrow_(*hierarchy_, partition_error,
                          "multi-block AMR temporal partition failed collectively");

    for (std::size_t child = 0; child < children.size(); ++child) {
      std::vector<field_type> staged =
          stage_parent_(level, window, children[child].window.begin, older, candidates);
      std::vector<const field_type*> staged_views;
      staged_views.reserve(staged.size());
      for (const field_type& field : staged)
        staged_views.push_back(&field);
      advance_level_recursive_(level + 1, children[child].window, static_cast<int>(child),
                               staged_views, outgoing_views, candidates, histories, clocks,
                               candidate_ledgers, attempt, advance_level, reflux);
    }

    // The recursive child has already synchronized its own descendants, so this is finest-first.
    for (std::size_t block = 0; block < hierarchy_->block_count(); ++block)
      invoke_collectively_([&] { outgoing_flux[block].commit(); },
                           "multi-block AMR flux-ledger commit failed collectively");
    const auto ratio =
        hierarchy_->topology_runtime().hierarchy().layout(level + 1).ratio_from_parent();
    const ::pops::amr::reflux::FaceRefinementMapping<Dim> mapping{
        hierarchy_->topology_runtime().hierarchy().layout(level).domain().lo,
        hierarchy_->topology_runtime().hierarchy().layout(level + 1).domain().lo};
    for (std::size_t block = 0; block < hierarchy_->block_count(); ++block) {
      RefluxContext context{block,
                            hierarchy_->block_identity(block),
                            level,
                            attempt,
                            window,
                            candidates[block][level],
                            candidates[block][level + 1],
                            outgoing_flux[block],
                            ratio,
                            mapping};
      invoke_collectively_([&] { reflux(context); },
                           "multi-block AMR reflux callback failed collectively");
      execute_average_down_collectively(hierarchy_->topology_runtime(), level + 1,
                                        std::as_const(candidates[block][level + 1]),
                                        candidates[block][level], hierarchy_->lane());
      histories[block][level]->newer = field_type(candidates[block][level]);
      candidate_ledgers[block][level].push_back(std::move(outgoing_flux[block]));
    }
  }

  hierarchy_type* hierarchy_ = nullptr;
  std::string hierarchy_contract_;
  std::vector<relation_type> relations_;
  PreparedAmrSubcyclePlan<Dim, MemorySpace> spatial_plan_;
  ::pops::amr::reflux::FaceFluxLedgerBudget flux_budget_{};
  typename hierarchy_type::ProgramBlockMap program_map_;
  std::string exact_contract_;
  HistoryMatrix accepted_histories_;
  ClockMatrix accepted_clocks_;
  LedgerMatrix accepted_ledgers_;
  std::uint64_t next_attempt_ = 0;
  std::uint64_t last_accepted_attempt_ = 0;
};

}  // namespace pops::numerics::time::amr
