/// @file
/// @brief Compile-time-ranked AMR runtime topology and conservative-operation authority.

#pragma once

#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/amr/reflux/metric_reflux.hpp>
#include <pops/amr/regridding/regrid.hpp>
#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/parallel/prepared_load_balance.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace pops::runtime::amr {

namespace detail {

template <int Dim>
void append_level_spatial_contract(
    ExactContractBuilder& contract,
    const ::pops::amr::hierarchy::LevelStateSpatialContract<Dim>& level) {
  contract.scalar(level.layout.level);
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(level.layout.domain.lo[axis])
        .scalar(level.layout.domain.hi[axis])
        .scalar(level.layout.ratio_from_parent[axis])
        .scalar(level.layout.rank_space.origin()[axis])
        .scalar(level.layout.rank_space.extent()[axis])
        // The collective spatial identity describes one global hierarchy. The process-local
        // coordinate is authenticated separately by each field/provider binding.
        .scalar(level.ghosts[axis]);
  contract.scalar(static_cast<std::uint8_t>(level.layout.distribution_mode))
      .scalar(level.components)
      .scalar(static_cast<std::uint64_t>(level.layout.validation_budget.boxes))
      .scalar(static_cast<std::uint64_t>(level.layout.validation_budget.overlap_pairs))
      .sequence(level.layout.patches,
                [](ExactContractBuilder& item, const Box<Dim>& patch) {
                  for (int axis = 0; axis < Dim; ++axis)
                    item.scalar(patch.lo[axis]).scalar(patch.hi[axis]);
                })
      .sequence(level.layout.owners, [](ExactContractBuilder& item, const Index<Dim>& owner) {
        for (int axis = 0; axis < Dim; ++axis)
          item.scalar(owner[axis]);
      });
}

template <int Dim, class MemorySpace>
std::string exact_runtime_spatial_contract(
    std::string_view spatial_identity,
    const ::pops::amr::hierarchy::AmrHierarchy<Dim, MemorySpace>& hierarchy,
    std::uint64_t topology_epoch, std::uint64_t materialization_generation) {
  if (spatial_identity.empty())
    throw std::invalid_argument("AMR runtime spatial identity must be non-empty");
  ExactContractBuilder contract;
  contract.text("pops.amr-runtime-spatial-contract")
      .scalar(std::uint32_t{2})
      .scalar(static_cast<std::uint32_t>(Dim))
      .text(spatial_identity)
      .scalar(topology_epoch)
      .scalar(materialization_generation)
      .scalar(static_cast<std::uint64_t>(hierarchy.validation_budget().levels))
      .scalar(static_cast<std::uint64_t>(hierarchy.validation_budget().parent_child_patch_pairs));
  const auto levels = hierarchy.spatial_contract();
  contract.scalar(static_cast<std::uint64_t>(levels.size()));
  for (const auto& level : levels)
    append_level_spatial_contract(contract, level);
  return std::move(contract).release();
}

inline std::uint64_t next_generation(std::uint64_t current, std::string_view name) {
  if (current == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error(std::string(name) + " exceeds uint64_t");
  return current + 1;
}

}  // namespace detail

/// Spatial AMR engine selected once from the Python-authored dimension before native allocation.
///
/// The runtime owns no dynamic dimension tag and contains no rank branch.  Every topology mutation
/// prepares a complete hierarchy value and its exact spatial contract before the no-throw publish.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class AmrRuntime {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "AmrRuntime only supports dimensions 1, 2, and 3");

  using hierarchy_type = ::pops::amr::hierarchy::AmrHierarchy<Dim, MemorySpace>;
  using level_type = typename hierarchy_type::level_type;
  using field_type = typename hierarchy_type::field_type;

  AmrRuntime(hierarchy_type hierarchy,
             std::shared_ptr<const PreparedLoadBalanceAuthority<Dim>> load_balance,
             std::string spatial_identity)
      : hierarchy_(std::move(hierarchy)),
        load_balance_(std::move(load_balance)),
        spatial_identity_(std::move(spatial_identity)) {
    if (!load_balance_)
      throw std::invalid_argument("AMR runtime requires one prepared load-balance authority");
    exact_spatial_contract_ = detail::exact_runtime_spatial_contract(
        spatial_identity_, hierarchy_, topology_epoch_, materialization_generation_);
  }

  AmrRuntime(const AmrRuntime&) = delete;
  AmrRuntime& operator=(const AmrRuntime&) = delete;
  AmrRuntime(AmrRuntime&&) noexcept = default;
  AmrRuntime& operator=(AmrRuntime&&) noexcept = default;

  static constexpr int dimension = Dim;

  hierarchy_type& hierarchy() noexcept { return hierarchy_; }
  const hierarchy_type& hierarchy() const noexcept { return hierarchy_; }
  const std::string& spatial_identity() const noexcept { return spatial_identity_; }
  std::string_view spatial_contract() const noexcept { return exact_spatial_contract_; }
  std::uint64_t topology_epoch() const noexcept { return topology_epoch_; }
  std::uint64_t materialization_generation() const noexcept { return materialization_generation_; }
  const PreparedLoadBalanceAuthority<Dim>& load_balance_authority() const noexcept {
    return *load_balance_;
  }

  /// Complete rollback image for an outer accepted-step transaction. The spatial contract is
  /// re-authenticated against the copied hierarchy and saved generations before publication.
  struct Snapshot {
    hierarchy_type hierarchy;
    std::uint64_t topology_epoch = 0;
    std::uint64_t materialization_generation = 0;
    std::string exact_spatial_contract;
  };

  Snapshot snapshot() const {
    return {hierarchy_, topology_epoch_, materialization_generation_, exact_spatial_contract_};
  }

  /// Fully materialized rollback publication.  Snapshot authentication and every potentially
  /// fallible hierarchy/string copy happen during preparation; publication only checks that the
  /// live source is unchanged and moves the prepared value into place.
  class PreparedRestorePublication {
   public:
    PreparedRestorePublication(const PreparedRestorePublication&) = delete;
    PreparedRestorePublication& operator=(const PreparedRestorePublication&) = delete;
    PreparedRestorePublication(PreparedRestorePublication&& other) noexcept
        : hierarchy_(std::move(other.hierarchy_)),
          source_topology_epoch_(other.source_topology_epoch_),
          source_materialization_generation_(other.source_materialization_generation_),
          source_spatial_contract_(std::move(other.source_spatial_contract_)),
          topology_epoch_(other.topology_epoch_),
          materialization_generation_(other.materialization_generation_),
          exact_spatial_contract_(std::move(other.exact_spatial_contract_)),
          collectively_authenticated_(std::exchange(other.collectively_authenticated_, false)),
          consumed_(std::exchange(other.consumed_, true)) {}

    PreparedRestorePublication& operator=(PreparedRestorePublication&& other) noexcept {
      if (this == &other)
        return *this;
      hierarchy_ = std::move(other.hierarchy_);
      source_topology_epoch_ = other.source_topology_epoch_;
      source_materialization_generation_ = other.source_materialization_generation_;
      source_spatial_contract_ = std::move(other.source_spatial_contract_);
      topology_epoch_ = other.topology_epoch_;
      materialization_generation_ = other.materialization_generation_;
      exact_spatial_contract_ = std::move(other.exact_spatial_contract_);
      collectively_authenticated_ = std::exchange(other.collectively_authenticated_, false);
      consumed_ = std::exchange(other.consumed_, true);
      return *this;
    }

    const hierarchy_type& hierarchy() const noexcept { return hierarchy_; }
    hierarchy_type& mutable_hierarchy_for_preparation() noexcept {
      if (consumed_)
        std::terminate();
      return hierarchy_;
    }
    std::string_view spatial_contract() const noexcept { return exact_spatial_contract_; }
    std::uint64_t topology_epoch() const noexcept { return topology_epoch_; }
    std::uint64_t materialization_generation() const noexcept {
      return materialization_generation_;
    }

   private:
    friend class AmrRuntime;

    PreparedRestorePublication(hierarchy_type hierarchy, std::uint64_t source_topology_epoch,
                               std::uint64_t source_materialization_generation,
                               std::string source_spatial_contract, std::uint64_t topology_epoch,
                               std::uint64_t materialization_generation,
                               std::string exact_spatial_contract)
        : hierarchy_(std::move(hierarchy)),
          source_topology_epoch_(source_topology_epoch),
          source_materialization_generation_(source_materialization_generation),
          source_spatial_contract_(std::move(source_spatial_contract)),
          topology_epoch_(topology_epoch),
          materialization_generation_(materialization_generation),
          exact_spatial_contract_(std::move(exact_spatial_contract)) {}

    hierarchy_type hierarchy_;
    std::uint64_t source_topology_epoch_ = 0;
    std::uint64_t source_materialization_generation_ = 0;
    std::string source_spatial_contract_;
    std::uint64_t topology_epoch_ = 0;
    std::uint64_t materialization_generation_ = 0;
    std::string exact_spatial_contract_;
    bool collectively_authenticated_ = false;
    bool consumed_ = false;
  };

  PreparedRestorePublication prepare_restore_publication(const Snapshot& snapshot) const {
    const std::string expected = detail::exact_runtime_spatial_contract(
        spatial_identity_, snapshot.hierarchy, snapshot.topology_epoch,
        snapshot.materialization_generation);
    if (expected != snapshot.exact_spatial_contract)
      throw std::invalid_argument("AMR runtime rollback snapshot is not authentic");
    return PreparedRestorePublication(hierarchy_type(snapshot.hierarchy), topology_epoch_,
                                      materialization_generation_, exact_spatial_contract_,
                                      snapshot.topology_epoch, snapshot.materialization_generation,
                                      snapshot.exact_spatial_contract);
  }

  void authenticate_prepared_restore_publication(PreparedRestorePublication& publication) const {
    if (publication.consumed_ || publication.source_topology_epoch_ != topology_epoch_ ||
        publication.source_materialization_generation_ != materialization_generation_ ||
        publication.source_spatial_contract_ != exact_spatial_contract_)
      throw std::invalid_argument("prepared AMR restore publication is stale");
    const std::string contract = detail::exact_runtime_spatial_contract(
        spatial_identity_, publication.hierarchy_, publication.topology_epoch_,
        publication.materialization_generation_);
    if (contract != publication.exact_spatial_contract_)
      throw std::invalid_argument("prepared AMR restore publication is not authentic");
    publication.collectively_authenticated_ = true;
  }

  void publish_authenticated_restore_noexcept(PreparedRestorePublication&& publication) noexcept {
    // Collective authentication proves the materialized target.  Recheck the live source at the
    // final no-throw boundary: a stale or reused authority must never swap a later generation.
    if (!publication.collectively_authenticated_ || publication.consumed_ ||
        publication.source_topology_epoch_ != topology_epoch_ ||
        publication.source_materialization_generation_ != materialization_generation_ ||
        publication.source_spatial_contract_ != exact_spatial_contract_)
      std::terminate();
    static_assert(std::is_nothrow_move_assignable_v<hierarchy_type>);
    publication.consumed_ = true;
    hierarchy_ = std::move(publication.hierarchy_);
    topology_epoch_ = publication.topology_epoch_;
    materialization_generation_ = publication.materialization_generation_;
    exact_spatial_contract_.swap(publication.exact_spatial_contract_);
  }

  void publish_prepared_restore(PreparedRestorePublication publication) {
    authenticate_prepared_restore_publication(publication);
    publish_authenticated_restore_noexcept(std::move(publication));
  }

  void restore(const Snapshot& snapshot) {
    publish_prepared_restore(prepare_restore_publication(snapshot));
  }

  ::pops::amr::regridding::PreparedRegrid<Dim> prepare_regrid(
      std::size_t parent_level, ::pops::amr::RefinementRatio<Dim> ratio,
      ::pops::amr::tagging::ClusterResult<Dim> clustered,
      ::pops::amr::regridding::RegridPreparationBudget budget,
      const ExecutionLane& lane = ExecutionLane::world()) const {
    if (parent_level >= hierarchy_.num_levels())
      throw std::out_of_range("AMR runtime regrid parent lies outside the hierarchy");
    return ::pops::amr::regridding::prepare_regrid(
        hierarchy_.layout(parent_level), ratio, std::move(clustered), *load_balance_, budget, lane);
  }

  /// Prepare a regrid from an authenticated unpublished authority.  This is deliberately an
  /// internal composition seam: an accepted-step transaction may stage more than one topology
  /// transition, and transition N+1 must use the hierarchy produced by transition N rather than
  /// rediscovering the still-live hierarchy.  It performs no publication and does not retain the
  /// snapshot.
  ::pops::amr::regridding::PreparedRegrid<Dim> prepare_regrid_from_snapshot(
      const Snapshot& source, std::size_t parent_level, ::pops::amr::RefinementRatio<Dim> ratio,
      ::pops::amr::tagging::ClusterResult<Dim> clustered,
      ::pops::amr::regridding::RegridPreparationBudget budget,
      const ExecutionLane& lane = ExecutionLane::world()) const {
    require_authentic_snapshot_(source, "AMR runtime forward regrid source");
    if (parent_level >= source.hierarchy.num_levels())
      throw std::out_of_range("AMR runtime forward regrid parent lies outside the hierarchy");
    return ::pops::amr::regridding::prepare_regrid(source.hierarchy.layout(parent_level), ratio,
                                                   std::move(clustered), *load_balance_, budget,
                                                   lane);
  }

  /// Fully materialized topology publication.  All hierarchy copies, generation checks and exact
  /// contract allocation happen in `prepare_regrid_publication`; publishing this value is a
  /// no-allocation move/swap cutover suitable for an outer multi-block collective transaction.
  class PreparedRegridPublication {
   public:
    PreparedRegridPublication(const PreparedRegridPublication&) = delete;
    PreparedRegridPublication& operator=(const PreparedRegridPublication&) = delete;
    PreparedRegridPublication(PreparedRegridPublication&& other) noexcept
        : hierarchy_(std::move(other.hierarchy_)),
          source_topology_epoch_(other.source_topology_epoch_),
          source_materialization_generation_(other.source_materialization_generation_),
          source_spatial_contract_(std::move(other.source_spatial_contract_)),
          topology_epoch_(other.topology_epoch_),
          materialization_generation_(other.materialization_generation_),
          exact_spatial_contract_(std::move(other.exact_spatial_contract_)),
          changes_topology_(other.changes_topology_),
          collectively_authenticated_(std::exchange(other.collectively_authenticated_, false)),
          consumed_(std::exchange(other.consumed_, true)) {}

    PreparedRegridPublication& operator=(PreparedRegridPublication&& other) noexcept {
      if (this == &other)
        return *this;
      hierarchy_ = std::move(other.hierarchy_);
      source_topology_epoch_ = other.source_topology_epoch_;
      source_materialization_generation_ = other.source_materialization_generation_;
      source_spatial_contract_ = std::move(other.source_spatial_contract_);
      topology_epoch_ = other.topology_epoch_;
      materialization_generation_ = other.materialization_generation_;
      exact_spatial_contract_ = std::move(other.exact_spatial_contract_);
      changes_topology_ = other.changes_topology_;
      collectively_authenticated_ = std::exchange(other.collectively_authenticated_, false);
      consumed_ = std::exchange(other.consumed_, true);
      return *this;
    }

    bool changes_topology() const noexcept { return changes_topology_; }
    const hierarchy_type& hierarchy() const noexcept { return hierarchy_; }
    /// Internal cold-preparation seam for a not-yet-published regrid target.  It never exposes an
    /// accepted hierarchy: publication consumes this object, and any later access fail-stops.
    hierarchy_type& mutable_hierarchy_for_preparation() noexcept {
      if (consumed_)
        std::terminate();
      return hierarchy_;
    }
    std::string_view spatial_contract() const noexcept { return exact_spatial_contract_; }

   private:
    friend class AmrRuntime;

    PreparedRegridPublication(hierarchy_type hierarchy, std::uint64_t source_topology_epoch,
                              std::uint64_t source_materialization_generation,
                              std::string source_spatial_contract, std::uint64_t topology_epoch,
                              std::uint64_t materialization_generation,
                              std::string exact_spatial_contract, bool changes_topology)
        : hierarchy_(std::move(hierarchy)),
          source_topology_epoch_(source_topology_epoch),
          source_materialization_generation_(source_materialization_generation),
          source_spatial_contract_(std::move(source_spatial_contract)),
          topology_epoch_(topology_epoch),
          materialization_generation_(materialization_generation),
          exact_spatial_contract_(std::move(exact_spatial_contract)),
          changes_topology_(changes_topology) {}

    hierarchy_type hierarchy_;
    std::uint64_t source_topology_epoch_ = 0;
    std::uint64_t source_materialization_generation_ = 0;
    std::string source_spatial_contract_;
    std::uint64_t topology_epoch_ = 0;
    std::uint64_t materialization_generation_ = 0;
    std::string exact_spatial_contract_;
    bool changes_topology_ = false;
    bool collectively_authenticated_ = false;
    bool consumed_ = false;
  };

  /// Prepare the inverse of one already prepared regrid before either direction is published.
  ///
  /// This is intentionally the only way to bind a restore publication to a future topology
  /// source.  The candidate is private state produced by `prepare_regrid_publication`; callers
  /// cannot manufacture its source epoch, materialization generation, or exact contract.  An
  /// outer accepted-step transaction uses this to retain a one-shot, no-allocation rollback
  /// authority while its candidate topology is visible only to the writer.
  PreparedRestorePublication prepare_inverse_restore_publication(
      const Snapshot& accepted, const PreparedRegridPublication& candidate) const {
    const std::string accepted_contract = detail::exact_runtime_spatial_contract(
        spatial_identity_, accepted.hierarchy, accepted.topology_epoch,
        accepted.materialization_generation);
    if (accepted_contract != accepted.exact_spatial_contract)
      throw std::invalid_argument("AMR runtime inverse rollback snapshot is not authentic");
    if (candidate.consumed_ || candidate.source_topology_epoch_ != topology_epoch_ ||
        candidate.source_materialization_generation_ != materialization_generation_ ||
        candidate.source_spatial_contract_ != exact_spatial_contract_)
      throw std::invalid_argument("AMR runtime inverse rollback candidate is stale");
    const std::string candidate_contract = detail::exact_runtime_spatial_contract(
        spatial_identity_, candidate.hierarchy_, candidate.topology_epoch_,
        candidate.materialization_generation_);
    if (candidate_contract != candidate.exact_spatial_contract_)
      throw std::invalid_argument("AMR runtime inverse rollback candidate is not authentic");
    return PreparedRestorePublication(hierarchy_type(accepted.hierarchy), candidate.topology_epoch_,
                                      candidate.materialization_generation_,
                                      candidate.exact_spatial_contract_, accepted.topology_epoch,
                                      accepted.materialization_generation,
                                      accepted.exact_spatial_contract);
  }

  /// Prepare the inverse relative to an unpublished source.  Unlike the public/live overload,
  /// this never consults `hierarchy_`; it is used only while composing a bounded stack of forward
  /// transactions before HiddenPublish.
  PreparedRestorePublication prepare_inverse_restore_publication_from_snapshot(
      const Snapshot& source, const PreparedRegridPublication& candidate) const {
    require_authentic_snapshot_(source, "AMR runtime forward inverse source");
    if (candidate.consumed_ || candidate.source_topology_epoch_ != source.topology_epoch ||
        candidate.source_materialization_generation_ != source.materialization_generation ||
        candidate.source_spatial_contract_ != source.exact_spatial_contract)
      throw std::invalid_argument("AMR runtime forward inverse candidate is stale");
    const std::string candidate_contract = detail::exact_runtime_spatial_contract(
        spatial_identity_, candidate.hierarchy_, candidate.topology_epoch_,
        candidate.materialization_generation_);
    if (candidate_contract != candidate.exact_spatial_contract_)
      throw std::invalid_argument("AMR runtime forward inverse candidate is not authentic");
    return PreparedRestorePublication(
        hierarchy_type(source.hierarchy), candidate.topology_epoch_,
        candidate.materialization_generation_, candidate.exact_spatial_contract_,
        source.topology_epoch, source.materialization_generation, source.exact_spatial_contract);
  }

  /// Authenticate an inverse prepared before the forward publication.  This leaves both values
  /// move-only and unpublished; it merely proves that the inverse source is precisely the
  /// forward candidate rather than an independently reconstructed topology.
  void authenticate_inverse_restore_publication(PreparedRestorePublication& inverse,
                                                const PreparedRegridPublication& candidate) const {
    if (inverse.consumed_ || candidate.consumed_ ||
        candidate.source_topology_epoch_ != topology_epoch_ ||
        candidate.source_materialization_generation_ != materialization_generation_ ||
        candidate.source_spatial_contract_ != exact_spatial_contract_ ||
        inverse.source_topology_epoch_ != candidate.topology_epoch_ ||
        inverse.source_materialization_generation_ != candidate.materialization_generation_ ||
        inverse.source_spatial_contract_ != candidate.exact_spatial_contract_)
      throw std::invalid_argument("AMR runtime inverse restore authority is stale");
    const std::string inverse_contract = detail::exact_runtime_spatial_contract(
        spatial_identity_, inverse.hierarchy_, inverse.topology_epoch_,
        inverse.materialization_generation_);
    if (inverse_contract != inverse.exact_spatial_contract_)
      throw std::invalid_argument("AMR runtime inverse restore authority is not authentic");
    inverse.collectively_authenticated_ = true;
  }

  void authenticate_prepared_regrid_publication_from_snapshot(
      const Snapshot& source, PreparedRegridPublication& publication) const {
    require_authentic_snapshot_(source, "AMR runtime forward publication source");
    if (publication.consumed_ || publication.source_topology_epoch_ != source.topology_epoch ||
        publication.source_materialization_generation_ != source.materialization_generation ||
        publication.source_spatial_contract_ != source.exact_spatial_contract)
      throw std::invalid_argument("prepared AMR forward regrid publication is stale");
    const std::string contract = detail::exact_runtime_spatial_contract(
        spatial_identity_, publication.hierarchy_, publication.topology_epoch_,
        publication.materialization_generation_);
    if (contract != publication.exact_spatial_contract_)
      throw std::invalid_argument("prepared AMR forward regrid publication is not authentic");
    publication.collectively_authenticated_ = true;
  }

  void authenticate_inverse_restore_publication_from_snapshot(
      const Snapshot& source, PreparedRestorePublication& inverse,
      const PreparedRegridPublication& candidate) const {
    require_authentic_snapshot_(source, "AMR runtime forward inverse source");
    if (inverse.consumed_ || candidate.consumed_ ||
        candidate.source_topology_epoch_ != source.topology_epoch ||
        candidate.source_materialization_generation_ != source.materialization_generation ||
        candidate.source_spatial_contract_ != source.exact_spatial_contract ||
        inverse.source_topology_epoch_ != candidate.topology_epoch_ ||
        inverse.source_materialization_generation_ != candidate.materialization_generation_ ||
        inverse.source_spatial_contract_ != candidate.exact_spatial_contract_)
      throw std::invalid_argument("AMR runtime forward inverse restore authority is stale");
    const std::string inverse_contract = detail::exact_runtime_spatial_contract(
        spatial_identity_, inverse.hierarchy_, inverse.topology_epoch_,
        inverse.materialization_generation_);
    if (inverse_contract != inverse.exact_spatial_contract_)
      throw std::invalid_argument("AMR runtime forward inverse restore authority is not authentic");
    inverse.collectively_authenticated_ = true;
  }

  PreparedRegridPublication prepare_regrid_publication(
      std::size_t parent_level, const ::pops::amr::regridding::PreparedRegrid<Dim>& prepared,
      std::optional<field_type> child_state) const {
    if (parent_level >= hierarchy_.num_levels() ||
        prepared.source_level() != hierarchy_.layout(parent_level).exact_identity())
      throw std::invalid_argument("AMR runtime regrid source is stale for the live hierarchy");

    bool changes = true;
    hierarchy_type candidate(hierarchy_);
    if (prepared.removes_fine_level()) {
      if (child_state)
        throw std::invalid_argument("an empty prepared regrid cannot publish child storage");
      if (parent_level + 1 == hierarchy_.num_levels()) {
        changes = false;
      } else {
        candidate = hierarchy_.truncated(parent_level);
      }
    } else {
      if (!child_state || !prepared.fine_layout() || !prepared.ownership())
        throw std::invalid_argument(
            "a non-empty prepared regrid requires its authenticated ownership and child storage");
      if (prepared.fine_layout()->level() != static_cast<int>(parent_level + 1) ||
          prepared.fine_layout()->distribution() != prepared.ownership()->plan().distribution())
        throw std::invalid_argument("prepared regrid child layout lost its ownership provenance");
      level_type child(*prepared.fine_layout(), std::move(*child_state));
      candidate = hierarchy_.with_level(std::move(child));
    }

    const std::uint64_t next_topology =
        changes ? detail::next_generation(topology_epoch_, "AMR runtime topology epoch")
                : topology_epoch_;
    const std::uint64_t next_materialization =
        changes ? detail::next_generation(materialization_generation_,
                                          "AMR runtime materialization generation")
                : materialization_generation_;
    std::string next_contract =
        changes ? detail::exact_runtime_spatial_contract(spatial_identity_, candidate,
                                                         next_topology, next_materialization)
                : exact_spatial_contract_;
    return PreparedRegridPublication(
        std::move(candidate), topology_epoch_, materialization_generation_, exact_spatial_contract_,
        next_topology, next_materialization, std::move(next_contract), changes);
  }

  /// Prepare a complete, externally qualified hierarchy replacement without exposing it through
  /// the accepted runtime.  Direct hierarchy rebuild uses this same two-direction publication
  /// object as ordinary regridding: all target bytes and the inverse are therefore available
  /// before the enclosing transaction reaches HiddenPublish.
  PreparedRegridPublication prepare_full_rebuild_publication(const Snapshot& target) const {
    require_authentic_snapshot_(target, "AMR runtime full rebuild target");
    const std::uint64_t next_topology =
        detail::next_generation(topology_epoch_, "AMR runtime topology epoch");
    const std::uint64_t next_materialization = detail::next_generation(
        materialization_generation_, "AMR runtime materialization generation");
    if (target.topology_epoch != next_topology ||
        target.materialization_generation != next_materialization)
      throw std::invalid_argument(
          "AMR runtime full rebuild target does not carry the next exact generation");
    return PreparedRegridPublication(hierarchy_type(target.hierarchy), topology_epoch_,
                                     materialization_generation_, exact_spatial_contract_,
                                     target.topology_epoch, target.materialization_generation,
                                     target.exact_spatial_contract, true);
  }

  PreparedRegridPublication prepare_regrid_publication_from_snapshot(
      const Snapshot& source, std::size_t parent_level,
      const ::pops::amr::regridding::PreparedRegrid<Dim>& prepared,
      std::optional<field_type> child_state) const {
    require_authentic_snapshot_(source, "AMR runtime forward publication source");
    if (parent_level >= source.hierarchy.num_levels() ||
        prepared.source_level() != source.hierarchy.layout(parent_level).exact_identity())
      throw std::invalid_argument("AMR runtime forward regrid source is stale");

    bool changes = true;
    hierarchy_type candidate(source.hierarchy);
    if (prepared.removes_fine_level()) {
      if (child_state)
        throw std::invalid_argument("an empty prepared regrid cannot publish child storage");
      if (parent_level + 1 == source.hierarchy.num_levels())
        changes = false;
      else
        candidate = source.hierarchy.truncated(parent_level);
    } else {
      if (!child_state || !prepared.fine_layout() || !prepared.ownership())
        throw std::invalid_argument(
            "a non-empty prepared regrid requires authenticated child storage");
      if (prepared.fine_layout()->level() != static_cast<int>(parent_level + 1) ||
          prepared.fine_layout()->distribution() != prepared.ownership()->plan().distribution())
        throw std::invalid_argument("prepared regrid child layout lost its ownership provenance");
      candidate =
          source.hierarchy.with_level(level_type(*prepared.fine_layout(), std::move(*child_state)));
    }
    const std::uint64_t next_topology =
        changes ? detail::next_generation(source.topology_epoch, "AMR runtime topology epoch")
                : source.topology_epoch;
    const std::uint64_t next_materialization =
        changes ? detail::next_generation(source.materialization_generation,
                                          "AMR runtime materialization generation")
                : source.materialization_generation;
    std::string next_contract =
        changes ? detail::exact_runtime_spatial_contract(spatial_identity_, candidate,
                                                         next_topology, next_materialization)
                : source.exact_spatial_contract;
    return PreparedRegridPublication(std::move(candidate), source.topology_epoch,
                                     source.materialization_generation,
                                     source.exact_spatial_contract, next_topology,
                                     next_materialization, std::move(next_contract), changes);
  }

  /// Authenticate a materialized candidate before it reaches the no-throw publication phase.
  void authenticate_prepared_regrid_publication(PreparedRegridPublication& publication) const {
    if (publication.consumed_ || publication.source_topology_epoch_ != topology_epoch_ ||
        publication.source_materialization_generation_ != materialization_generation_ ||
        publication.source_spatial_contract_ != exact_spatial_contract_)
      throw std::invalid_argument("prepared AMR regrid publication is stale");
    const std::string contract = detail::exact_runtime_spatial_contract(
        spatial_identity_, publication.hierarchy_, publication.topology_epoch_,
        publication.materialization_generation_);
    if (contract != publication.exact_spatial_contract_)
      throw std::invalid_argument("prepared AMR regrid publication is not authentic");
    publication.collectively_authenticated_ = true;
  }

  /// Publish an already authenticated candidate using only no-throw moves/swaps.
  void publish_authenticated_regrid_noexcept(PreparedRegridPublication&& publication) noexcept {
    // The publication can be delayed after collective authentication.  Its source must still be
    // the live hierarchy immediately before the first swap, and a consumed authority is fatal.
    if (!publication.collectively_authenticated_ || publication.consumed_ ||
        publication.source_topology_epoch_ != topology_epoch_ ||
        publication.source_materialization_generation_ != materialization_generation_ ||
        publication.source_spatial_contract_ != exact_spatial_contract_)
      std::terminate();
    publication.consumed_ = true;
    if (!publication.changes_topology_)
      return;
    // Prepared graph contexts retain references to the exact forward Fabs.  Move assignment may
    // replace those storage objects; a no-throw swap makes the already-prepared hierarchy become
    // the accepted owner without invalidating its Fab addresses.
    static_assert(std::is_nothrow_swappable_v<hierarchy_type>);
    using std::swap;
    swap(hierarchy_, publication.hierarchy_);
    topology_epoch_ = publication.topology_epoch_;
    materialization_generation_ = publication.materialization_generation_;
    exact_spatial_contract_.swap(publication.exact_spatial_contract_);
  }

  /// Compatibility convenience. New transaction paths authenticate collectively first and call
  /// `publish_authenticated_regrid_noexcept` directly.
  void publish_prepared_regrid(PreparedRegridPublication publication) {
    authenticate_prepared_regrid_publication(publication);
    publish_authenticated_regrid_noexcept(std::move(publication));
  }

  /// Publish a prepared child layout and transferred state, or remove every level above its parent.
  void publish_regrid(std::size_t parent_level,
                      ::pops::amr::regridding::PreparedRegrid<Dim> prepared,
                      std::optional<field_type> child_state) {
    publish_prepared_regrid(
        prepare_regrid_publication(parent_level, prepared, std::move(child_state)));
  }

  PreparedRebalanceDecision<Dim> prepare_rebalance(
      std::size_t level, ResourceEstimates estimates,
      parallel::LoadBalancePreparationBudget preparation_budget, const RebalancePolicy& policy,
      const ExecutionLane& lane = ExecutionLane::world()) const {
    if (level >= hierarchy_.num_levels())
      throw std::out_of_range("AMR runtime rebalance level lies outside the hierarchy");
    const auto& layout = hierarchy_.layout(level);
    return load_balance_->decide_rebalance(
        static_cast<int>(level), layout.patches(), layout.distribution(), topology_epoch_,
        materialization_generation_, estimates, preparation_budget, policy, lane);
  }

  PreparedRebalanceDecision<Dim> prepare_rebalance(
      std::size_t level, ResourceEstimates estimates,
      parallel::LoadBalancePreparationBudget preparation_budget,
      const ExecutionLane& lane = ExecutionLane::world()) const {
    if (level >= hierarchy_.num_levels())
      throw std::out_of_range("AMR runtime rebalance level lies outside the hierarchy");
    const auto& layout = hierarchy_.layout(level);
    return load_balance_->decide_rebalance(
        static_cast<int>(level), layout.patches(), layout.distribution(), topology_epoch_,
        materialization_generation_, estimates, preparation_budget, lane);
  }

  /// Publish remapped storage only if the prepared decision still names the live source exactly.
  ///
  /// Finer levels are invalidated because their ownership and transfer histories were prepared from
  /// the prior parent materialization.
  void apply_rebalance(std::size_t level, PreparedRebalanceDecision<Dim> decision,
                       field_type remapped_state) {
    if (level >= hierarchy_.num_levels() || !decision.accepted ||
        decision.reason != RebalanceReason::NetBenefit)
      throw std::invalid_argument("AMR runtime requires an accepted prepared rebalance decision");
    if (decision.topology_epoch != topology_epoch_ ||
        decision.materialization_generation != materialization_generation_)
      throw std::invalid_argument("AMR runtime rebalance decision is stale");

    const auto& current = hierarchy_.layout(level);
    const std::string live_source =
        load_balance_->rebalance_source_contract(static_cast<int>(level), current.distribution(),
                                                 topology_epoch_, materialization_generation_);
    if (decision.source_contract != live_source ||
        decision.exact_contract != ::pops::detail::exact_rebalance_decision(decision))
      throw std::invalid_argument("AMR runtime rebalance decision contract is not authentic");
    const mesh::Distribution<Dim>& proposed = decision.proposed.plan().distribution();
    if (!proposed.matches_layout(current.patches()) ||
        proposed.rank_space() != current.distribution().rank_space())
      throw std::invalid_argument("AMR runtime rebalance proposal targets another level layout");

    ::pops::amr::hierarchy::LevelLayout<Dim> rebalanced(
        current.level(), current.domain(), current.patches(), proposed, current.ratio_from_parent(),
        current.validation_budget());
    level_type replacement(std::move(rebalanced), std::move(remapped_state));
    commit_hierarchy_(hierarchy_.with_level(std::move(replacement)));
  }

  template <::pops::amr::transfer::Centering Center>
  ::pops::amr::transfer::PreparedTransfer<Dim> prepare_transfer(
      std::size_t source_level, std::size_t destination_level,
      const ::pops::amr::hierarchy::LevelStateSpatialContract<Dim>& source_contract,
      const ::pops::amr::hierarchy::LevelStateSpatialContract<Dim>& destination_contract,
      ::pops::amr::transfer::TransferKind kind, FieldView<const Real, Dim> source,
      FieldView<Real, Dim> destination, const Box<Dim>& destination_region,
      ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
      ::pops::amr::transfer::ComponentRange components = {}) const {
    if (source_level >= hierarchy_.num_levels() || destination_level >= hierarchy_.num_levels() ||
        source_contract != hierarchy_.level(source_level).spatial_contract() ||
        destination_contract != hierarchy_.level(destination_level).spatial_contract())
      throw std::invalid_argument(
          "AMR runtime transfer requires adjacent live level spatial contracts");
    std::size_t fine_level = 0;
    switch (kind) {
      case ::pops::amr::transfer::TransferKind::ConservativeRestriction:
        if (destination_level + 1 != source_level)
          throw std::invalid_argument(
              "AMR restriction must transfer from one fine level to its parent");
        fine_level = source_level;
        break;
      case ::pops::amr::transfer::TransferKind::LinearProlongation:
      case ::pops::amr::transfer::TransferKind::CoarseFineGhostInterpolation:
      case ::pops::amr::transfer::TransferKind::FifthOrderCoarseFineGhostInterpolation:
      case ::pops::amr::transfer::TransferKind::ConstantInjection:
      case ::pops::amr::transfer::TransferKind::NodeMultilinearProlongation:
        if (source_level + 1 != destination_level)
          throw std::invalid_argument(
              "AMR interpolation must transfer from one parent to its child");
        fine_level = destination_level;
        break;
      default:
        throw std::invalid_argument("AMR runtime transfer kind is not registered");
    }
    const auto& ratio = hierarchy_.layout(fine_level).ratio_from_parent();
    return ::pops::amr::transfer::TransferProvider<Dim, Center>(kind).prepare(
        source, destination, destination_region, ratio, mapping, components);
  }

  template <class Payload, class Axpy>
  ::pops::amr::reflux::MetricFaceReflux<Payload> reconcile_reflux(
      const ::pops::amr::reflux::TransactionalFaceFluxLedger<Dim, Payload>& ledger,
      const ::pops::amr::reflux::CoarseFaceRefluxKey<Dim>& key, std::string_view state_identity,
      const ::pops::amr::reflux::FaceRefinementMapping<Dim>& mapping,
      const ::pops::amr::reflux::MetricRefluxBudget& budget, Axpy&& axpy) const {
    return reconcile_reflux_for_owner(ledger, key, spatial_identity_, state_identity, mapping,
                                      budget, std::forward<Axpy>(axpy));
  }

  /// Reconcile one block-qualified carrier on this runtime's canonical spatial hierarchy.
  /// The ordinary overload retains the topology identity as owner; multi-block transactions name
  /// the physical block explicitly while still authenticating the same adjacent live levels.
  template <class Payload, class Axpy>
  ::pops::amr::reflux::MetricFaceReflux<Payload> reconcile_reflux_for_owner(
      const ::pops::amr::reflux::TransactionalFaceFluxLedger<Dim, Payload>& ledger,
      const ::pops::amr::reflux::CoarseFaceRefluxKey<Dim>& key, std::string_view owner_identity,
      std::string_view state_identity,
      const ::pops::amr::reflux::FaceRefinementMapping<Dim>& mapping,
      const ::pops::amr::reflux::MetricRefluxBudget& budget, Axpy&& axpy) const {
    if (owner_identity.empty() || state_identity.empty() || key.owner != owner_identity ||
        key.state != state_identity || key.levels.coarse < 0 ||
        static_cast<std::size_t>(key.levels.fine) >= hierarchy_.num_levels() ||
        key.levels.fine != key.levels.coarse + 1)
      throw std::invalid_argument(
          "AMR runtime reflux key does not authenticate an adjacent live state transition");
    const auto& ratio =
        hierarchy_.layout(static_cast<std::size_t>(key.levels.fine)).ratio_from_parent();
    return ::pops::amr::reflux::metric_reflux(ledger, key, ratio, mapping, budget,
                                              std::forward<Axpy>(axpy));
  }

 private:
  void require_authentic_snapshot_(const Snapshot& snapshot, std::string_view authority) const {
    const std::string expected = detail::exact_runtime_spatial_contract(
        spatial_identity_, snapshot.hierarchy, snapshot.topology_epoch,
        snapshot.materialization_generation);
    if (expected != snapshot.exact_spatial_contract)
      throw std::invalid_argument(std::string(authority) + " is not authentic");
  }

  void commit_hierarchy_(hierarchy_type candidate) {
    static_assert(std::is_nothrow_move_assignable_v<hierarchy_type>);
    const std::uint64_t next_topology =
        detail::next_generation(topology_epoch_, "AMR runtime topology epoch");
    const std::uint64_t next_materialization = detail::next_generation(
        materialization_generation_, "AMR runtime materialization generation");
    std::string next_contract = detail::exact_runtime_spatial_contract(
        spatial_identity_, candidate, next_topology, next_materialization);
    hierarchy_ = std::move(candidate);
    topology_epoch_ = next_topology;
    materialization_generation_ = next_materialization;
    exact_spatial_contract_.swap(next_contract);
  }

  hierarchy_type hierarchy_;
  std::shared_ptr<const PreparedLoadBalanceAuthority<Dim>> load_balance_;
  std::string spatial_identity_;
  std::uint64_t topology_epoch_ = 0;
  std::uint64_t materialization_generation_ = 0;
  std::string exact_spatial_contract_;
};

}  // namespace pops::runtime::amr
