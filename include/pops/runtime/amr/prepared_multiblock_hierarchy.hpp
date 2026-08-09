/// @file
/// @brief One exact-ranked AMR topology with transactional state carriers for every Program block.

#pragma once

#include <pops/coupling/source/coupling_operator.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/system/system_coupling_registry.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::amr {

/// Prepared multi-block ownership around one canonical AMR topology.
///
/// `AmrRuntime<Dim>` remains the sole topology/regrid authority and owns the primary block state.
/// Every other block owns one independent state carrier per canonical level.  No block duplicates a
/// `LevelLayout`, chooses another distribution, or advances a private topology generation.  Program
/// mappings, coupling application, publication and regrid are consequently validated against one
/// ordered block registry and one spatial contract.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedMultiBlockAmrHierarchy {
 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedMultiBlockAmrHierarchy only supports dimensions 1, 2, and 3");

  using engine_type = AmrRuntime<Dim, MemorySpace>;
  using hierarchy_type = typename engine_type::hierarchy_type;
  using field_type = typename engine_type::field_type;
  using coupling_registry_type = runtime::system::SystemCouplingRegistry<Dim>;
  using coupling_operation_type = runtime::system::PreparedCouplingOperator<Dim>;

  struct AdditionalBlock {
    std::string identity;
    std::vector<field_type> levels;
  };

  struct ProgramBlockMap {
    std::vector<std::size_t> canonical_indices;
    std::string hierarchy_contract;
    std::string exact_contract;
  };

  struct Snapshot {
    typename engine_type::Snapshot primary;
    std::vector<AdditionalBlock> additional;
    std::uint64_t accepted_revision = 0;
    std::string exact_collective_contract;
  };

  PreparedMultiBlockAmrHierarchy(const PreparedMultiBlockAmrHierarchy&) = delete;
  PreparedMultiBlockAmrHierarchy& operator=(const PreparedMultiBlockAmrHierarchy&) = delete;
  PreparedMultiBlockAmrHierarchy(PreparedMultiBlockAmrHierarchy&&) noexcept = default;
  PreparedMultiBlockAmrHierarchy& operator=(PreparedMultiBlockAmrHierarchy&&) noexcept = default;

  /// Prepare all carriers first, reach rank consensus, and only then duplicate the owning lane.
  /// The caller may build the canonical engine through the normal AmrRuntime collective preflight;
  /// this cutover never creates a second communicator or topology while local validation can throw.
  static PreparedMultiBlockAmrHierarchy prepare_collectively(
      engine_type primary, std::string primary_identity, std::vector<AdditionalBlock> additional,
      std::string lane_identity) {
    const ExecutionCommunicator parent = ExecutionCommunicator::world();
    std::exception_ptr local_error;
    std::string contract;
    std::string qualified_lane_identity;
    try {
      validate_carriers_(primary, primary_identity, additional);
      if (lane_identity.empty())
        throw std::invalid_argument("prepared multi-block AMR lane identity must be non-empty");
      if (lane_identity.size() >
          std::numeric_limits<std::size_t>::max() - parent.identity().size() - 1U)
        throw std::length_error("prepared multi-block AMR lane identity is too large");
      qualified_lane_identity.reserve(parent.identity().size() + 1U + lane_identity.size());
      qualified_lane_identity.assign(parent.identity());
      qualified_lane_identity.push_back('/');
      qualified_lane_identity.append(lane_identity);
      contract =
          exact_hierarchy_contract_(primary, primary_identity, additional, qualified_lane_identity);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L) != 0) {
      if (n_ranks() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("prepared multi-block AMR carrier preflight failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr"), contract}}))
      throw std::invalid_argument(
          "prepared multi-block AMR carrier contract differs between MPI ranks");

    ExecutionLane lane = ExecutionLane::duplicate_collectively(parent, lane_identity);
    return PreparedMultiBlockAmrHierarchy(std::move(primary), std::move(primary_identity),
                                          std::move(additional), std::move(lane),
                                          std::move(contract));
  }

  static constexpr int dimension = Dim;

  std::size_t block_count() const noexcept { return additional_.size() + 1; }
  std::size_t level_count() const noexcept { return primary_.hierarchy().num_levels(); }
  std::uint64_t accepted_revision() const noexcept { return accepted_revision_; }
  std::string_view collective_contract() const noexcept { return collective_contract_; }
  const ExecutionLane& lane() const noexcept { return lane_; }
  engine_type& topology_runtime() noexcept { return primary_; }
  const engine_type& topology_runtime() const noexcept { return primary_; }

  const std::string& block_identity(std::size_t block) const {
    require_block_(block);
    return block == 0 ? primary_identity_ : additional_[block - 1].identity;
  }

  std::size_t block_index(std::string_view identity) const {
    std::size_t match = block_count();
    for (std::size_t block = 0; block < block_count(); ++block) {
      if (block_identity(block) != identity)
        continue;
      if (match != block_count())
        throw std::logic_error("prepared multi-block AMR registry contains duplicate identities");
      match = block;
    }
    if (match == block_count())
      throw std::out_of_range("unknown prepared multi-block AMR block '" + std::string(identity) +
                              "'");
    return match;
  }

  field_type& state(std::size_t block, std::size_t level) {
    require_level_(level);
    require_block_(block);
    return block == 0 ? primary_.hierarchy().state(level) : additional_[block - 1].levels[level];
  }

  const field_type& state(std::size_t block, std::size_t level) const {
    require_level_(level);
    require_block_(block);
    return block == 0 ? primary_.hierarchy().state(level) : additional_[block - 1].levels[level];
  }

  /// Resolve a Program order once.  It must be an exact permutation: every accepted carrier has one
  /// owner and a later candidate pack therefore cannot omit, duplicate, or alias a block silently.
  ProgramBlockMap prepare_program_block_map(std::span<const std::string> ordered_blocks) const {
    std::exception_ptr local_error;
    ProgramBlockMap result;
    try {
      if (ordered_blocks.size() != block_count())
        throw std::invalid_argument(
            "AMR Program block map must name every prepared carrier exactly once");
      result.canonical_indices.reserve(ordered_blocks.size());
      std::vector<bool> seen(block_count(), false);
      for (const std::string& identity : ordered_blocks) {
        const std::size_t canonical = block_index(identity);
        if (seen[canonical])
          throw std::invalid_argument("AMR Program block map contains a duplicate block");
        seen[canonical] = true;
        result.canonical_indices.push_back(canonical);
      }
      result.hierarchy_contract = collective_contract_;
      result.exact_contract = exact_program_contract_(result.canonical_indices);
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error, "AMR Program block-map preflight failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr-program-map"), result.exact_contract}},
            lane_))
      throw std::invalid_argument("AMR Program block map differs between MPI ranks");
    return result;
  }

  /// Install one exact prepared coupling provider.  The executable is copied into a candidate
  /// registry only after its inspect/provider contract agrees byte-for-byte on the owning lane.
  void install_prepared_coupling_operator(std::string provider_contract, CouplingOperatorView view,
                                          coupling_operation_type operation) {
    std::exception_ptr local_error;
    std::string exact;
    coupling_registry_type candidate;
    try {
      if (couplings_sealed_)
        throw std::logic_error("prepared multi-block AMR couplings are already sealed");
      if (provider_contract.empty() || view.label.empty() || !operation ||
          !std::isfinite(view.frequency.constant_mu) || view.frequency.constant_mu < 0.0)
        throw std::invalid_argument(
            "prepared AMR coupling requires exact identity, executable, and finite frequency");
      ExactContractBuilder contract;
      contract.text("pops.prepared-multiblock-amr.coupling")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .text(provider_contract)
          .text(view.label)
          .scalar(view.frequency.constant_mu)
          .scalar(static_cast<std::uint8_t>(view.frequency.per_cell ? 1 : 0))
          .sequence(view.conservation.conserved_roles,
                    [](ExactContractBuilder& item, const std::string& role) { item.text(role); })
          .sequence(view.conservation.created_roles,
                    [](ExactContractBuilder& item, const std::string& role) { item.text(role); });
      exact = std::move(contract).release();
      candidate = couplings_;
      candidate.operators.push_back(std::move(operation));
      candidate.coupled_operators.push_back(std::move(view));
      candidate.operator_contracts.push_back(exact);
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error, "prepared AMR coupling installation failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr-coupling"), exact}}, lane_))
      throw std::invalid_argument("prepared AMR coupling provider differs between MPI ranks");
    couplings_ = std::move(candidate);
  }

  /// Seal the ordered provider registry only after every rank agrees on the exact provider list.
  /// Application is forbidden until this collective cutover succeeds.
  void seal_couplings() {
    std::exception_ptr local_error;
    std::string exact;
    try {
      if (couplings_sealed_)
        throw std::logic_error("prepared multi-block AMR couplings are already sealed");
      if (couplings_.operator_contracts.size() != couplings_.operators.size() ||
          couplings_.operators.size() != couplings_.coupled_operators.size())
        throw std::logic_error("prepared multi-block AMR coupling registry is inconsistent");
      exact = exact_coupling_registry_contract_(collective_contract_);
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error, "prepared AMR coupling seal failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr-coupling-registry"), exact}}, lane_))
      throw std::invalid_argument("prepared AMR coupling registry differs between MPI ranks");
    coupling_registry_contract_.swap(exact);
    couplings_sealed_ = true;
  }
  std::size_t coupling_count() const noexcept { return couplings_.operators.size(); }
  std::string_view coupling_registry_contract() const noexcept {
    return coupling_registry_contract_;
  }
  const std::vector<CouplingOperatorView>& coupled_operators() const noexcept {
    return couplings_.coupled_operators;
  }

  /// Apply couplings to a complete Program-owned candidate pack.  Candidates are restored in their
  /// Kokkos memory space on any local or remote execution failure; accepted hierarchy storage is
  /// never a hidden workspace.
  std::size_t apply_program_candidates(const ProgramBlockMap& map, std::size_t level, Real dt,
                                       std::span<field_type* const> program_candidates) const {
    std::vector<field_type*> canonical =
        validate_program_candidates_(map, level, dt, program_candidates);
    std::vector<field_type> rollback = copy_pack_collectively_(canonical, "candidate rollback");

    std::exception_ptr local_error;
    try {
      couplings_.apply(dt, canonical);
      Kokkos::fence();
    } catch (...) {
      local_error = std::current_exception();
    }
    const bool failed = all_reduce_max(local_error ? 1L : 0L, lane_.communicator()) != 0;
    if (!failed)
      return couplings_.operators.size();

    restore_pack_collectively_(rollback, canonical, "candidate coupling rollback");
    if (lane_.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("prepared AMR coupling failed and rolled back collectively");
  }

  /// Canonical-order convenience used by a provider that already owns the Program map.
  std::size_t apply_coupling_operators_at_level(std::size_t level, Real dt,
                                                std::span<field_type* const> candidates) const {
    ProgramBlockMap map;
    map.hierarchy_contract = collective_contract_;
    map.canonical_indices.resize(block_count());
    for (std::size_t block = 0; block < block_count(); ++block)
      map.canonical_indices[block] = block;
    map.exact_contract = canonical_program_contract_;
    return apply_program_candidates(map, level, dt, candidates);
  }

  /// Publish a complete Program candidate group to accepted carriers.  Every rollback allocation
  /// and every candidate validation precedes the first live write.  Publication failure restores
  /// all blocks, on every rank, before reporting the collective error.
  void publish_program_candidates(const ProgramBlockMap& map, std::size_t level,
                                  std::span<field_type* const> program_candidates) {
    std::vector<field_type*> canonical =
        validate_program_candidates_(map, level, Real(0), program_candidates,
                                     /*require_dt_consensus=*/false,
                                     /*require_sealed_couplings=*/false);
    std::vector<field_type*> accepted;
    accepted.reserve(block_count());
    for (std::size_t block = 0; block < block_count(); ++block)
      accepted.push_back(&state(block, level));
    std::vector<field_type> rollback = copy_pack_collectively_(accepted, "publication rollback");

    std::exception_ptr local_error;
    try {
      copy_pack_(canonical, accepted);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane_.communicator()) != 0) {
      restore_pack_collectively_(rollback, accepted, "accepted publication rollback");
      if (lane_.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("prepared AMR block publication rolled back collectively");
    }
    increment_revision_collectively_();
  }

  /// Complete source-only transaction used by direct runtimes and tests: clone all live blocks,
  /// apply the same prepared provider to the private pack, then publish the group atomically.
  std::size_t apply_and_publish_level(std::size_t level, Real dt) {
    preflight_application_level_(level, dt);
    std::vector<field_type*> accepted;
    accepted.reserve(block_count());
    for (std::size_t block = 0; block < block_count(); ++block)
      accepted.push_back(&state(block, level));
    std::vector<field_type> candidates = copy_pack_collectively_(accepted, "coupling candidates");
    std::vector<field_type*> candidate_pack;
    candidate_pack.reserve(candidates.size());
    for (field_type& candidate : candidates)
      candidate_pack.push_back(&candidate);
    const std::size_t applied = apply_coupling_operators_at_level(level, dt, candidate_pack);

    ProgramBlockMap map;
    map.hierarchy_contract = collective_contract_;
    map.canonical_indices.resize(block_count());
    for (std::size_t block = 0; block < block_count(); ++block)
      map.canonical_indices[block] = block;
    map.exact_contract = canonical_program_contract_;
    publish_program_candidates(map, level, candidate_pack);
    return applied;
  }

  /// Prepare one transfer against the canonical spatial contracts while selecting the state carrier
  /// by block identity.  Thus every block uses the same PreparedTransfer implementation and level
  /// relationship; there is no block-specific transfer engine.
  template <::pops::amr::transfer::Centering Center>
  ::pops::amr::transfer::PreparedTransfer<Dim> prepare_transfer(
      std::size_t block, std::size_t source_level, std::size_t source_local,
      std::size_t destination_level, std::size_t destination_local,
      ::pops::amr::transfer::TransferKind kind, const Box<Dim>& destination_region,
      ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
      ::pops::amr::transfer::ComponentRange components = {}) {
    const field_type& source = state(block, source_level);
    field_type& destination = state(block, destination_level);
    return primary_.template prepare_transfer<Center>(
        source_level, destination_level,
        primary_.hierarchy().level(source_level).spatial_contract(),
        primary_.hierarchy().level(destination_level).spatial_contract(), kind,
        source.fab(source_local).view(), destination.fab(destination_local).view(),
        destination_region, mapping, components);
  }

  /// Publish one prepared topology mutation with one transferred child carrier per block.  Secondary
  /// carrier vectors are fully materialized first; after the canonical engine commits, their swap is
  /// no-throw.  Removal likewise truncates every block to the same depth in one transaction.
  void publish_regrid(std::size_t parent_level,
                      ::pops::amr::regridding::PreparedRegrid<Dim> prepared,
                      std::vector<std::optional<field_type>> child_states) {
    std::exception_ptr local_error;
    std::vector<AdditionalBlock> candidate_additional;
    std::optional<typename engine_type::PreparedRegridPublication> primary_publication;
    std::string next_collective_contract;
    std::string next_program_contract;
    std::string next_coupling_registry_contract;
    try {
      if (parent_level >= level_count() ||
          prepared.source_level() != primary_.hierarchy().layout(parent_level).exact_identity())
        throw std::invalid_argument("multi-block AMR regrid source is stale");
      if (child_states.size() != block_count())
        throw std::invalid_argument("multi-block AMR regrid requires one child slot per block");
      if (accepted_revision_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("multi-block AMR accepted revision overflow");
      if (prepared.removes_fine_level()) {
        if (std::any_of(child_states.begin(), child_states.end(),
                        [](const auto& child) { return child.has_value(); }))
          throw std::invalid_argument("removing an AMR level cannot publish child carriers");
      } else {
        if (!prepared.fine_layout() || !prepared.ownership())
          throw std::invalid_argument("prepared multi-block AMR regrid lost its child layout");
        for (std::size_t block = 0; block < block_count(); ++block) {
          if (!child_states[block])
            throw std::invalid_argument("multi-block AMR regrid is missing one child carrier");
          require_child_contract_(block, parent_level, *prepared.fine_layout(),
                                  *child_states[block]);
        }
      }

      candidate_additional.reserve(additional_.size());
      for (std::size_t index = 0; index < additional_.size(); ++index) {
        AdditionalBlock candidate;
        candidate.identity = additional_[index].identity;
        const std::size_t retained = std::min(parent_level + 1, additional_[index].levels.size());
        candidate.levels.reserve(parent_level + 2);
        for (std::size_t level = 0; level < retained; ++level)
          candidate.levels.push_back(additional_[index].levels[level]);
        if (!prepared.removes_fine_level())
          candidate.levels.push_back(std::move(*child_states[index + 1]));
        candidate_additional.push_back(std::move(candidate));
      }
      primary_publication.emplace(
          primary_.prepare_regrid_publication(parent_level, prepared, std::move(child_states[0])));
      next_collective_contract = exact_hierarchy_contract_(
          primary_publication->hierarchy(), primary_publication->spatial_contract(),
          primary_identity_, candidate_additional, lane_.identity());
      next_program_contract = exact_canonical_program_contract_(
          next_collective_contract, primary_identity_, candidate_additional);
      if (couplings_sealed_)
        next_coupling_registry_contract =
            exact_coupling_registry_contract_(next_collective_contract);
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error, "multi-block AMR regrid preflight failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr-regrid"), prepared.exact_contract()},
             {std::string_view("prepared-multiblock-amr-next"), next_collective_contract},
             {std::string_view("prepared-multiblock-amr-next-coupling"),
              next_coupling_registry_contract}},
            lane_))
      throw std::invalid_argument("prepared multi-block AMR regrid differs between MPI ranks");

    const bool changes = primary_publication->changes_topology();
    primary_.publish_prepared_regrid(std::move(*primary_publication));
    if (!changes)
      return;
    additional_.swap(candidate_additional);
    ++accepted_revision_;
    collective_contract_.swap(next_collective_contract);
    canonical_program_contract_.swap(next_program_contract);
    if (couplings_sealed_)
      coupling_registry_contract_.swap(next_coupling_registry_contract);
  }

  Snapshot snapshot() const {
    return {primary_.snapshot(), additional_, accepted_revision_, collective_contract_};
  }

  /// Restore both the canonical topology and every secondary carrier.  A candidate copy is prepared
  /// and authenticated collectively before topology publication, so every post-consensus state
  /// transition is a no-allocation move/swap.
  void restore(const Snapshot& snapshot) {
    std::exception_ptr local_error;
    std::vector<AdditionalBlock> candidate;
    std::optional<typename engine_type::PreparedRestorePublication> primary_publication;
    std::string next_collective_contract;
    std::string next_program_contract;
    std::string next_coupling_registry_contract;
    std::string restore_contract;
    try {
      candidate = snapshot.additional;
      validate_snapshot_carriers_(snapshot.primary.hierarchy, candidate, additional_);
      primary_publication.emplace(primary_.prepare_restore_publication(snapshot.primary));
      next_collective_contract = exact_hierarchy_contract_(
          primary_publication->hierarchy(), primary_publication->spatial_contract(),
          primary_identity_, candidate, lane_.identity());
      if (snapshot.exact_collective_contract != next_collective_contract) {
        const auto mismatch = std::mismatch(
            snapshot.exact_collective_contract.begin(), snapshot.exact_collective_contract.end(),
            next_collective_contract.begin(), next_collective_contract.end());
        throw std::invalid_argument(
            "prepared multi-block AMR snapshot collective contract is not authentic at byte " +
            std::to_string(static_cast<std::size_t>(mismatch.first -
                                                    snapshot.exact_collective_contract.begin())) +
            " (snapshot bytes=" + std::to_string(snapshot.exact_collective_contract.size()) +
            ", restored bytes=" + std::to_string(next_collective_contract.size()) + ")");
      }
      next_program_contract =
          exact_canonical_program_contract_(next_collective_contract, primary_identity_, candidate);
      if (couplings_sealed_)
        next_coupling_registry_contract =
            exact_coupling_registry_contract_(next_collective_contract);
      ExactContractBuilder contract;
      contract.text("pops.prepared-multiblock-amr.restore")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(collective_contract_)
          .bytes(next_collective_contract)
          .scalar(snapshot.accepted_revision);
      restore_contract = std::move(contract).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error, "prepared multi-block AMR restore failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr-restore"), restore_contract}}, lane_))
      throw std::invalid_argument("prepared multi-block AMR restore differs between MPI ranks");

    primary_.publish_prepared_restore(std::move(*primary_publication));
    additional_.swap(candidate);
    accepted_revision_ = snapshot.accepted_revision;
    collective_contract_.swap(next_collective_contract);
    canonical_program_contract_.swap(next_program_contract);
    if (couplings_sealed_)
      coupling_registry_contract_.swap(next_coupling_registry_contract);
  }

 private:
  PreparedMultiBlockAmrHierarchy(engine_type primary, std::string primary_identity,
                                 std::vector<AdditionalBlock> additional, ExecutionLane lane,
                                 std::string collective_contract)
      : primary_(std::move(primary)),
        primary_identity_(std::move(primary_identity)),
        additional_(std::move(additional)),
        lane_(std::move(lane)),
        collective_contract_(std::move(collective_contract)) {
    canonical_program_contract_ = exact_canonical_program_contract_();
  }

  static void validate_carriers_(const engine_type& primary, std::string_view primary_identity,
                                 const std::vector<AdditionalBlock>& additional) {
    if (primary_identity.empty())
      throw std::invalid_argument("prepared multi-block AMR primary identity must be non-empty");
    if (additional.empty())
      throw std::invalid_argument(
          "PreparedMultiBlockAmrHierarchy requires at least two physical block carriers");
    std::vector<std::string_view> identities{primary_identity};
    for (const AdditionalBlock& block : additional) {
      if (block.identity.empty())
        throw std::invalid_argument("prepared multi-block AMR block identity must be non-empty");
      if (std::find(identities.begin(), identities.end(), block.identity) != identities.end())
        throw std::invalid_argument("prepared multi-block AMR block identities must be unique");
      identities.push_back(block.identity);
      if (block.levels.size() != primary.hierarchy().num_levels())
        throw std::invalid_argument("prepared multi-block AMR block has another hierarchy depth");
      for (std::size_t level = 0; level < block.levels.size(); ++level)
        require_shared_level_contract_(primary.hierarchy().state(level), block.levels[level]);
    }
  }

  static void validate_snapshot_carriers_(const hierarchy_type& primary,
                                          const std::vector<AdditionalBlock>& additional,
                                          const std::vector<AdditionalBlock>& live_additional) {
    if (additional.empty() || additional.size() != live_additional.size())
      throw std::invalid_argument("multi-block AMR snapshot lost its secondary carriers");
    for (std::size_t index = 0; index < additional.size(); ++index) {
      const AdditionalBlock& block = additional[index];
      if (block.identity.empty() || block.identity != live_additional[index].identity ||
          block.levels.size() != primary.num_levels())
        throw std::invalid_argument("multi-block AMR snapshot carrier is incomplete");
      for (std::size_t level = 0; level < block.levels.size(); ++level)
        require_shared_level_contract_(primary.state(level), block.levels[level]);
    }
  }

  static void require_shared_level_contract_(const field_type& topology_state,
                                             const field_type& carrier) {
    if (carrier.layout() != topology_state.layout() ||
        carrier.distribution() != topology_state.distribution() ||
        carrier.local_rank() != topology_state.local_rank())
      throw std::invalid_argument(
          "prepared multi-block AMR carrier does not share the canonical level topology");
  }

  static std::string exact_hierarchy_contract_(const engine_type& primary,
                                               std::string_view primary_identity,
                                               const std::vector<AdditionalBlock>& additional,
                                               std::string_view lane_identity) {
    return exact_hierarchy_contract_(primary.hierarchy(), primary.spatial_contract(),
                                     primary_identity, additional, lane_identity);
  }

  static std::string exact_hierarchy_contract_(const hierarchy_type& primary,
                                               std::string_view spatial_contract,
                                               std::string_view primary_identity,
                                               const std::vector<AdditionalBlock>& additional,
                                               std::string_view lane_identity) {
    ExactContractBuilder contract;
    contract.text("pops.prepared-multiblock-amr")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(lane_identity)
        .bytes(spatial_contract)
        .scalar(static_cast<std::uint64_t>(additional.size() + 1))
        .text(primary_identity);
    const auto append_shape = [&](const field_type& field) {
      contract.scalar(field.ncomp());
      for (int axis = 0; axis < Dim; ++axis)
        contract.scalar(field.ghosts()[axis]);
    };
    for (std::size_t level = 0; level < primary.num_levels(); ++level)
      append_shape(primary.state(level));
    for (const AdditionalBlock& block : additional) {
      contract.text(block.identity);
      for (const field_type& level : block.levels)
        append_shape(level);
    }
    return std::move(contract).release();
  }

  std::string exact_canonical_program_contract_() const {
    return exact_canonical_program_contract_(collective_contract_, primary_identity_, additional_);
  }

  static std::string exact_canonical_program_contract_(
      std::string_view hierarchy_contract, std::string_view primary_identity,
      const std::vector<AdditionalBlock>& additional) {
    ExactContractBuilder contract;
    contract.text("pops.prepared-multiblock-amr.program-map")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(hierarchy_contract)
        .scalar(static_cast<std::uint64_t>(additional.size() + 1))
        .text(primary_identity)
        .scalar(std::uint64_t{0});
    for (std::size_t block = 0; block < additional.size(); ++block)
      contract.text(additional[block].identity).scalar(static_cast<std::uint64_t>(block + 1));
    return std::move(contract).release();
  }

  std::string exact_program_contract_(std::span<const std::size_t> canonical_indices) const {
    ExactContractBuilder contract;
    contract.text("pops.prepared-multiblock-amr.program-map")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(collective_contract_)
        .scalar(static_cast<std::uint64_t>(canonical_indices.size()));
    for (std::size_t canonical : canonical_indices)
      contract.text(block_identity(canonical)).scalar(static_cast<std::uint64_t>(canonical));
    return std::move(contract).release();
  }

  std::string exact_coupling_registry_contract_(std::string_view hierarchy_contract) const {
    ExactContractBuilder contract;
    contract.text("pops.prepared-multiblock-amr.coupling-registry")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(hierarchy_contract)
        .sequence(
            couplings_.operator_contracts,
            [](ExactContractBuilder& item, const std::string& provider) { item.bytes(provider); });
    return std::move(contract).release();
  }

  void require_block_(std::size_t block) const {
    if (block >= block_count())
      throw std::out_of_range("prepared multi-block AMR block is out of range");
  }

  void require_level_(std::size_t level) const {
    if (level >= level_count())
      throw std::out_of_range("prepared multi-block AMR level is out of range");
  }

  void collectively_rethrow_(const std::exception_ptr& local_error,
                             std::string_view collective_message) const {
    if (all_reduce_max(local_error ? 1L : 0L, lane_.communicator()) == 0)
      return;
    if (lane_.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(std::string(collective_message));
  }

  void require_map_(const ProgramBlockMap& map) const {
    if (map.hierarchy_contract != collective_contract_ ||
        map.canonical_indices.size() != block_count() ||
        map.exact_contract != exact_program_contract_(map.canonical_indices))
      throw std::invalid_argument("AMR Program block map is stale for this hierarchy");
    std::vector<bool> seen(block_count(), false);
    for (std::size_t canonical : map.canonical_indices) {
      if (canonical >= block_count() || seen[canonical])
        throw std::invalid_argument("AMR Program block map is not an exact permutation");
      seen[canonical] = true;
    }
  }

  static bool same_field_contract_(const field_type& left, const field_type& right) noexcept {
    return left.layout() == right.layout() && left.distribution() == right.distribution() &&
           left.local_rank() == right.local_rank() && left.ncomp() == right.ncomp() &&
           left.ghosts() == right.ghosts() &&
           left.local_global_indices() == right.local_global_indices();
  }

  std::vector<field_type*> validate_program_candidates_(
      const ProgramBlockMap& map, std::size_t level, Real dt,
      std::span<field_type* const> program_candidates, bool require_dt_consensus = true,
      bool require_sealed_couplings = true) const {
    std::exception_ptr local_error;
    std::vector<field_type*> canonical(block_count(), nullptr);
    std::string invocation_contract;
    try {
      require_map_(map);
      require_level_(level);
      if (require_sealed_couplings && !couplings_sealed_)
        throw std::logic_error("prepared multi-block AMR couplings must be sealed before use");
      if (program_candidates.size() != map.canonical_indices.size())
        throw std::invalid_argument("AMR Program candidate pack does not match its block map");
      if (require_dt_consensus && (!std::isfinite(static_cast<double>(dt)) || dt < Real(0)))
        throw std::invalid_argument("AMR coupling dt must be finite and non-negative");
      for (std::size_t program = 0; program < program_candidates.size(); ++program) {
        field_type* candidate = program_candidates[program];
        const std::size_t block = map.canonical_indices[program];
        if (candidate == nullptr || !same_field_contract_(*candidate, state(block, level)))
          throw std::invalid_argument(
              "AMR Program candidate differs from its exact block/level carrier");
        for (std::size_t accepted_block = 0; accepted_block < block_count(); ++accepted_block)
          if (candidate == &state(accepted_block, level))
            throw std::invalid_argument("accepted AMR state cannot be a Program workspace");
        if (std::find(canonical.begin(), canonical.end(), candidate) != canonical.end())
          throw std::invalid_argument("AMR Program candidate pack aliases two blocks");
        canonical[block] = candidate;
      }
      ExactContractBuilder invocation;
      invocation.text("pops.prepared-multiblock-amr.application")
          .scalar(std::uint32_t{2})
          .scalar(std::int32_t{Dim})
          .bytes(map.exact_contract)
          .scalar(static_cast<std::uint64_t>(level))
          .scalar(static_cast<std::uint8_t>(require_dt_consensus ? 1 : 0))
          .scalar(static_cast<std::uint8_t>(require_sealed_couplings ? 1 : 0));
      if (require_dt_consensus)
        invocation.scalar(dt);
      if (require_sealed_couplings)
        invocation.bytes(coupling_registry_contract_)
            .scalar(static_cast<std::uint64_t>(couplings_.operators.size()));
      invocation_contract = std::move(invocation).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error, "AMR Program candidate preflight failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr-application"), invocation_contract}},
            lane_))
      throw std::invalid_argument("AMR Program invocation differs between MPI ranks");
    return canonical;
  }

  void preflight_application_level_(std::size_t level, Real dt) const {
    std::exception_ptr local_error;
    std::string invocation_contract;
    try {
      require_level_(level);
      if (!couplings_sealed_)
        throw std::logic_error("prepared multi-block AMR couplings must be sealed before use");
      if (!std::isfinite(static_cast<double>(dt)) || dt < Real(0))
        throw std::invalid_argument("AMR coupling dt must be finite and non-negative");
      ExactContractBuilder invocation;
      invocation.text("pops.prepared-multiblock-amr.level-application")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(collective_contract_)
          .bytes(coupling_registry_contract_)
          .scalar(static_cast<std::uint64_t>(couplings_.operators.size()))
          .scalar(static_cast<std::uint64_t>(level))
          .scalar(dt);
      invocation_contract = std::move(invocation).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error,
                          "prepared multi-block AMR level application failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr-level-application"), invocation_contract}},
            lane_))
      throw std::invalid_argument(
          "prepared multi-block AMR level application differs between MPI ranks");
  }

  std::vector<field_type> copy_pack_collectively_(std::span<field_type* const> source,
                                                  std::string_view purpose) const {
    std::exception_ptr local_error;
    std::vector<field_type> result;
    try {
      result.reserve(source.size());
      for (const field_type* field : source)
        result.emplace_back(*field);
      Kokkos::fence();
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error, "prepared multi-block AMR " + std::string(purpose) +
                                           " allocation failed collectively");
    return result;
  }

  static void copy_pack_(std::span<field_type* const> source,
                         std::span<field_type* const> destination) {
    if (source.size() != destination.size())
      throw std::invalid_argument("AMR field-pack copy requires equal pack sizes");
    for (std::size_t block = 0; block < source.size(); ++block) {
      if (!same_field_contract_(*source[block], *destination[block]))
        throw std::invalid_argument("AMR field-pack copy lost an exact carrier contract");
      for (std::size_t local = 0; local < source[block]->local_size(); ++local)
        Kokkos::deep_copy(destination[block]->fab(local).storage(),
                          source[block]->fab(local).storage());
    }
    Kokkos::fence();
  }

  void restore_pack_collectively_(std::vector<field_type>& rollback,
                                  std::span<field_type* const> destination,
                                  std::string_view purpose) const {
    std::vector<field_type*> source;
    source.reserve(rollback.size());
    for (field_type& field : rollback)
      source.push_back(&field);
    std::exception_ptr local_error;
    try {
      copy_pack_(source, destination);
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(
        local_error, "prepared multi-block AMR " + std::string(purpose) + " failed collectively");
  }

  void increment_revision_collectively_() {
    const long exhausted =
        accepted_revision_ == std::numeric_limits<std::uint64_t>::max() ? 1L : 0L;
    if (all_reduce_max(exhausted, lane_.communicator()) != 0)
      throw std::overflow_error("multi-block AMR accepted revision overflow");
    ++accepted_revision_;
  }

  void require_child_contract_(std::size_t block, std::size_t parent_level,
                               const ::pops::amr::hierarchy::LevelLayout<Dim>& layout,
                               const field_type& child) const {
    const field_type& parent = state(block, parent_level);
    if (child.layout() != layout.patches() || child.distribution() != layout.distribution() ||
        child.local_rank() != parent.local_rank() || child.ncomp() != parent.ncomp() ||
        child.ghosts() != parent.ghosts())
      throw std::invalid_argument(
          "multi-block AMR child carrier differs from its block and prepared topology");
  }

  engine_type primary_;
  std::string primary_identity_;
  std::vector<AdditionalBlock> additional_;
  // Declared before all providers/registries so reverse member destruction releases their callbacks
  // first and frees the communicator last.
  ExecutionLane lane_;
  coupling_registry_type couplings_;
  std::string coupling_registry_contract_;
  bool couplings_sealed_ = false;
  std::uint64_t accepted_revision_ = 0;
  std::string collective_contract_;
  std::string canonical_program_contract_;
};

}  // namespace pops::runtime::amr
