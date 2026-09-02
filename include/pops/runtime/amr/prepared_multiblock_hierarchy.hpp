/// @file
/// @brief One exact-ranked AMR topology with transactional state carriers for every Program block.

#pragma once

#include <pops/coupling/source/coupling_operator.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/multiblock/interface_flux_scheduler.hpp>
#include <pops/runtime/system/system_coupling_registry.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
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

namespace pops {
template <int Dim>
class AmrSystem;
}

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
  using interface_scheduler_type = runtime::multiblock::InterfaceFluxScheduler<Dim>;
  using interface_publication_type = runtime::multiblock::InterfaceFluxFragmentPublication;
  using interface_installer_type = std::function<void(interface_scheduler_type&)>;

 private:
  // Only the owning AmrSystem may turn a decoded accepted Program state into a carrier
  // requalification request.  Keep the archived bytes owned here: the requalification can cross
  // collective preparation and must never borrow storage from a transient decode image.
  struct RestartAuthority {
    std::string spatial_contract;
    std::uint64_t topology_epoch = 0;
    std::uint64_t materialization_generation = 0;
  };

  friend class ::pops::AmrSystem<Dim>;

 public:
  struct AdditionalBlock {
    std::string identity;
    std::vector<field_type> levels;
  };

  struct ProgramBlockMapAuthority {
    const PreparedMultiBlockAmrHierarchy* owner = nullptr;
    std::vector<std::size_t> canonical_indices;
    std::string hierarchy_contract;
    std::string exact_contract;
  };

  struct ProgramBlockMap {
    std::vector<std::size_t> canonical_indices;
    std::string hierarchy_contract;
    std::string exact_contract;
    // Cold-authorized immutable copy.  Program execution may inspect this witness but must
    // never reconstruct its exact contract or allocate a duplicate-permutation bitmap.
    std::shared_ptr<const ProgramBlockMapAuthority> authority;
  };

  struct Snapshot {
    typename engine_type::Snapshot primary;
    std::vector<AdditionalBlock> additional;
    std::uint64_t accepted_revision = 0;
    std::string exact_collective_contract;
  };

  /// Fully materialized cold image for one fixed topology.  It owns rollback fields and the
  /// fixed pointer packs used by resident publication; pointers are rebound only after the
  /// corresponding no-throw topology swap has completed.
  struct ResidentWorkspace {
    std::vector<std::vector<field_type>> rollback;
    std::vector<std::vector<field_type*>> accepted_packs;
    std::vector<std::vector<field_type*>> publication_candidates;
    std::uint64_t rollback_revision = 0;
    bool rollback_captured = false;
  };

  /// One reusable arena per hierarchy level. The generic coupling workspace owns canonical
  /// pointers, rollback/conservation images and raw inspection buffers; AMR adds only the
  /// topology-specific interface residual fields/views that the scheduler requires.
  struct CouplingLevelWorkspace {
    std::unique_ptr<runtime::system::PreparedCouplingWorkspace<Dim>> workspace;
    std::vector<field_type> interface_rhs;
    std::vector<field_type*> interface_rhs_views;
    typename interface_scheduler_type::InterfaceInvocationWorkspace interface_invocation;
    std::string level_contract;

    struct ResidentStorageFootprint {
      std::uint64_t generic_workspace = 0;
      std::uint64_t interface_rhs_images = 0;
      std::uint64_t interface_rhs_views = 0;
      std::uint64_t interface_invocation = 0;
      std::uint64_t level_contract_characters = 0;
    };

    [[nodiscard]] static std::uint64_t resident_storage_bytes(
        const ResidentStorageFootprint& footprint) {
      const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
        if (value > std::numeric_limits<std::uint64_t>::max() - total)
          throw std::overflow_error("prepared AMR coupling level storage overflows uint64");
        total += value;
      };
      std::uint64_t total = 0;
      for (const std::uint64_t value : {
               footprint.generic_workspace,
               footprint.interface_rhs_images,
               footprint.interface_rhs_views,
               footprint.interface_invocation,
               footprint.level_contract_characters,
           })
        checked_add(total, value);
      return total;
    }

    [[nodiscard]] static std::size_t contract_payload_bytes(
        std::size_t hierarchy_contract_characters) {
      constexpr std::size_t frame = 1U + sizeof(std::uint64_t);
      constexpr std::size_t scalar_i32 = frame + 2U + sizeof(std::int32_t);
      constexpr std::size_t scalar_u32 = frame + 2U + sizeof(std::uint32_t);
      constexpr std::size_t scalar_u64 = frame + 2U + sizeof(std::uint64_t);
      std::size_t total =
          frame + std::string_view("pops.prepared-multiblock-amr.coupling-level").size();
      const auto add = [&total](std::size_t value) {
        if (value > std::numeric_limits<std::size_t>::max() - total)
          throw std::overflow_error("prepared AMR coupling level contract size overflows size_t");
        total += value;
      };
      add(scalar_u32);
      add(scalar_i32);
      add(frame + hierarchy_contract_characters);
      add(scalar_u64);
      return total;
    }

    [[nodiscard]] std::uint64_t resident_storage_bytes() const {
      const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
        if (value > std::numeric_limits<std::uint64_t>::max() - total)
          throw std::overflow_error("prepared AMR coupling level storage overflows uint64");
        total += value;
      };
      const auto vector_bytes = [](const auto& values) -> std::uint64_t {
        using value_type = typename std::remove_reference_t<decltype(values)>::value_type;
        if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
          throw std::overflow_error("prepared AMR coupling level storage overflows uint64");
        return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
      };
      const auto external_string_bytes = [](const std::string& value) -> std::uint64_t {
        const auto begin = reinterpret_cast<std::uintptr_t>(std::addressof(value));
        const auto end = begin + sizeof(value);
        const auto data = reinterpret_cast<std::uintptr_t>(value.data());
        if (data >= begin && data < end)
          return 0;
        if (value.capacity() == std::numeric_limits<std::uint64_t>::max())
          throw std::overflow_error("prepared AMR coupling level string storage overflows uint64");
        return static_cast<std::uint64_t>(value.capacity()) + 1U;
      };

      ResidentStorageFootprint footprint;
      if (workspace != nullptr) {
        footprint.generic_workspace = sizeof(*workspace);
        checked_add(footprint.generic_workspace, workspace->resident_storage_bytes());
      }
      footprint.interface_rhs_images = vector_bytes(interface_rhs);
      for (const field_type& rhs : interface_rhs)
        checked_add(footprint.interface_rhs_images, rhs.resident_storage_bytes());
      footprint.interface_rhs_views = vector_bytes(interface_rhs_views);
      footprint.interface_invocation = interface_invocation.resident_storage_bytes();
      footprint.level_contract_characters = external_string_bytes(level_contract);
      return resident_storage_bytes(footprint);
    }
  };

  /// Sealed Program authority for the scheduler's temporary consensus bytes.  These values are
  /// copied through every forward/inverse workspace image; a scheduler may never infer them from
  /// a candidate stage point while an accepted-step transaction is active.
  struct ProgramInterfaceInvocationCapacity {
    std::size_t clock_characters = 0;
    std::size_t graph_rate_application_characters = 0;
    std::size_t stage_characters = 0;
    std::string exact_contract;

    [[nodiscard]] bool active() const noexcept {
      return clock_characters != 0 || graph_rate_application_characters != 0 ||
             stage_characters != 0;
    }
  };

  class PreparedProgramInterfaceInvocationBinding {
   public:
    PreparedProgramInterfaceInvocationBinding(const PreparedProgramInterfaceInvocationBinding&) =
        delete;
    PreparedProgramInterfaceInvocationBinding& operator=(
        const PreparedProgramInterfaceInvocationBinding&) = delete;
    PreparedProgramInterfaceInvocationBinding(
        PreparedProgramInterfaceInvocationBinding&&) noexcept = default;
    PreparedProgramInterfaceInvocationBinding& operator=(
        PreparedProgramInterfaceInvocationBinding&&) noexcept = default;

    [[nodiscard]] std::string_view exact_contract() const noexcept {
      return capacity ? std::string_view(capacity->exact_contract) : std::string_view{};
    }

    [[nodiscard]] const std::vector<std::string>& coupling_operator_contracts() const {
      if (owner == nullptr || !capacity)
        throw std::logic_error(
            "prepared AMR Program interface invocation binding has no coupling authority");
      return owner->couplings_.operator_contracts;
    }

    [[nodiscard]] bool coupling_requires_conservation() const {
      if (owner == nullptr || !capacity)
        throw std::logic_error(
            "prepared AMR Program interface invocation binding has no coupling authority");
      return std::any_of(owner->couplings_.operators.begin(), owner->couplings_.operators.end(),
                         [](const coupling_operation_type& operation) {
                           return !operation.conservation_groups().empty();
                         });
    }

    [[nodiscard]] bool requires_interface_rhs() const {
      if (owner == nullptr || !capacity)
        throw std::logic_error(
            "prepared AMR Program interface invocation binding has no coupling authority");
      return static_cast<bool>(owner->interface_scheduler_);
    }

    [[nodiscard]] const ProgramInterfaceInvocationCapacity& invocation_capacity() const {
      if (!capacity)
        throw std::logic_error(
            "prepared AMR Program interface invocation binding has no capacity authority");
      return *capacity;
    }

    [[nodiscard]] std::uint64_t configured_interface_invocation_storage_bytes() const {
      if (owner == nullptr || !capacity)
        throw std::logic_error(
            "prepared AMR Program interface invocation binding has no capacity authority");
      if (!owner->interface_scheduler_ || !capacity->active())
        return 0;
      typename interface_scheduler_type::InterfaceInvocationWorkspace workspace;
      owner->interface_scheduler_->bind_invocation_workspace(
          workspace, capacity->clock_characters, capacity->graph_rate_application_characters,
          capacity->stage_characters);
      return workspace.resident_storage_bytes();
    }

    [[nodiscard]] std::size_t configured_hierarchy_contract_payload_bytes(
        std::size_t spatial_contract_characters, std::size_t level_count) const {
      if (owner == nullptr || !capacity)
        throw std::logic_error(
            "prepared AMR Program interface invocation binding has no coupling authority");
      if (level_count == 0)
        throw std::invalid_argument(
            "prepared AMR Program coupling level contract requires a positive depth");
      constexpr std::size_t frame = 1U + sizeof(std::uint64_t);
      constexpr std::size_t scalar_i32 = frame + 2U + sizeof(std::int32_t);
      constexpr std::size_t scalar_u32 = frame + 2U + sizeof(std::uint32_t);
      constexpr std::size_t scalar_u64 = frame + 2U + sizeof(std::uint64_t);
      const auto add = [](std::size_t& total, std::size_t value) {
        if (value > std::numeric_limits<std::size_t>::max() - total)
          throw std::overflow_error(
              "prepared AMR configured hierarchy contract size overflows size_t");
        total += value;
      };
      std::size_t hierarchy_contract =
          frame + std::string_view("pops.prepared-multiblock-amr").size();
      add(hierarchy_contract, scalar_u32);
      add(hierarchy_contract, scalar_i32);
      add(hierarchy_contract, frame + owner->lane_contract_identity_.size());
      add(hierarchy_contract, frame + spatial_contract_characters);
      add(hierarchy_contract, scalar_u64);
      add(hierarchy_contract, frame + owner->primary_identity_.size());
      const auto append_shape = [&](const field_type& field) {
        (void)
            field;  // ExactContractBuilder scalar framing is fixed-width; values are authenticated elsewhere.
        add(hierarchy_contract, scalar_i32);
        for (int axis = 0; axis < Dim; ++axis)
          add(hierarchy_contract, scalar_i32);
      };
      for (std::size_t level = 0; level < level_count; ++level)
        append_shape(owner->primary_->hierarchy().state(0));
      for (const AdditionalBlock& block : owner->additional_) {
        add(hierarchy_contract, frame + block.identity.size());
        for (std::size_t level = 0; level < level_count; ++level)
          append_shape(block.levels.at(0));
      }
      return hierarchy_contract;
    }

    [[nodiscard]] std::uint64_t configured_level_contract_storage_bytes(
        std::size_t hierarchy_contract_characters) const {
      const auto level_contract =
          CouplingLevelWorkspace::contract_payload_bytes(hierarchy_contract_characters);
      return runtime::system::PreparedCouplingWorkspace<Dim>::reserved_string_storage_bytes(
          level_contract);
    }

    /// One exact configured level image.  This scalar shape is the only forward-capacity input:
    /// it names the same vector, MultiFab and contract families walked by the resident owner.
    struct ConfiguredCouplingLevelShape {
      std::size_t block_count = 0;
      std::uint64_t global_patch_count = 0;
      std::uint64_t local_patch_count = 0;
      std::uint64_t owner_count = 0;
      std::uint64_t field_value_count = 0;
      std::size_t hierarchy_contract_characters = 0;
    };

    /// C(n): exact storage retained by the complete coupling-workspace owner at every prefix of
    /// `levels`.  The enclosing vector is charged at its actual reserved prefix capacity, so the
    /// caller can fold accepted/forward/inverse owners without an opaque multiplier.
    [[nodiscard]] std::vector<std::uint64_t> configured_workspace_storage_bytes_by_level(
        std::span<const ConfiguredCouplingLevelShape> levels) const {
      if (owner == nullptr || !capacity || levels.empty())
        throw std::logic_error(
            "prepared AMR Program coupling workspace capacity has no complete authority");
      using workspace_type = runtime::system::PreparedCouplingWorkspace<Dim>;
      const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
        if (value > std::numeric_limits<std::uint64_t>::max() - total)
          throw std::overflow_error("prepared AMR configured coupling workspace overflows uint64");
        total += value;
      };
      const auto checked_multiply = [](std::uint64_t left, std::uint64_t right) {
        if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
          throw std::overflow_error("prepared AMR configured coupling workspace overflows uint64");
        return left * right;
      };
      const auto& operator_contracts = coupling_operator_contracts();
      const bool requires_interface_rhs = this->requires_interface_rhs();
      const bool requires_rollback_image = !operator_contracts.empty() || requires_interface_rhs;
      const bool requires_conservation = coupling_requires_conservation();
      const std::uint64_t interface_invocation = configured_interface_invocation_storage_bytes();
      std::vector<std::uint64_t> result;
      result.reserve(levels.size());
      std::uint64_t level_owners = 0;
      for (const ConfiguredCouplingLevelShape& level : levels) {
        if (level.block_count == 0)
          throw std::invalid_argument(
              "prepared AMR configured coupling workspace requires one Program block");
        if (level.local_patch_count > level.global_patch_count ||
            (level.owner_count != 0 && level.owner_count != level.global_patch_count))
          throw std::invalid_argument(
              "prepared AMR configured coupling workspace has an invalid ownership image");
        const auto block_count = static_cast<std::uint64_t>(level.block_count);
        const auto global_metadata_per_patch = static_cast<std::uint64_t>(
            2U * sizeof(Box<Dim>) + sizeof(Index<Dim>) + sizeof(std::size_t));
        const auto local_metadata_per_patch =
            static_cast<std::uint64_t>(sizeof(std::size_t) + sizeof(typename field_type::fab_type));
        std::uint64_t field_image =
            workspace_type::template reserved_vector_storage_bytes<field_type>(level.block_count);
        std::uint64_t metadata =
            checked_multiply(level.global_patch_count, global_metadata_per_patch);
        checked_add(metadata, checked_multiply(level.local_patch_count, local_metadata_per_patch));
        // Replicated distributions intentionally retain no owner vector.  A partitioned image
        // retains exactly one rank coordinate per global patch; the validation above excludes a
        // partial owner table.
        if (level.owner_count == 0)
          metadata -= checked_multiply(level.global_patch_count, sizeof(Index<Dim>));
        checked_add(field_image, checked_multiply(block_count, metadata));
        checked_add(field_image, checked_multiply(level.field_value_count, sizeof(Real)));

        typename workspace_type::ResidentStorageFootprint generic;
        generic.prototype_state_pointers =
            workspace_type::template reserved_vector_storage_bytes<const field_type*>(
                level.block_count);
        generic.canonical_state_pointers =
            workspace_type::template reserved_vector_storage_bytes<field_type*>(level.block_count);
        generic.host_patch_offsets =
            workspace_type::template reserved_vector_storage_bytes<std::size_t>(level.block_count);
        generic.host_patch_counts =
            workspace_type::template reserved_vector_storage_bytes<std::size_t>(level.block_count);
        if (requires_rollback_image) {
          generic.rollback_images = field_image;
          const auto patch_slots = checked_multiply(block_count, level.local_patch_count);
          if (patch_slots > std::numeric_limits<std::size_t>::max())
            throw std::overflow_error(
                "prepared AMR configured coupling host-patch count overflows size_t");
          generic.host_patch_descriptors = workspace_type::template reserved_vector_storage_bytes<
              typename workspace_type::HostPatch>(static_cast<std::size_t>(patch_slots));
          generic.host_candidate_values = checked_multiply(level.field_value_count, sizeof(Real));
        }
        if (requires_conservation) {
          generic.conservation_images = field_image;
          generic.host_before_values = checked_multiply(level.field_value_count, sizeof(Real));
        }
        generic.operator_contract_slots =
            workspace_type::template reserved_vector_storage_bytes<std::string>(
                operator_contracts.size());
        for (const std::string& contract : operator_contracts)
          checked_add(generic.operator_contract_characters,
                      workspace_type::copied_string_storage_bytes(contract));
        generic.invocation_characters = workspace_type::invocation_static_storage_bytes(
            level.block_count, operator_contracts, requires_interface_rhs);

        typename CouplingLevelWorkspace::ResidentStorageFootprint level_footprint;
        level_footprint.generic_workspace = sizeof(workspace_type);
        checked_add(level_footprint.generic_workspace,
                    workspace_type::resident_storage_bytes(generic));
        if (requires_interface_rhs) {
          level_footprint.interface_rhs_images = field_image;
          level_footprint.interface_rhs_views =
              workspace_type::template reserved_vector_storage_bytes<field_type*>(
                  level.block_count);
          level_footprint.interface_invocation = interface_invocation;
        }
        level_footprint.level_contract_characters =
            configured_level_contract_storage_bytes(level.hierarchy_contract_characters);
        checked_add(level_owners, CouplingLevelWorkspace::resident_storage_bytes(level_footprint));
        std::uint64_t cumulative =
            workspace_type::template reserved_vector_storage_bytes<CouplingLevelWorkspace>(
                result.size() + 1U);
        checked_add(cumulative, level_owners);
        if (!result.empty() && cumulative < result.back())
          throw std::logic_error(
              "prepared AMR configured coupling workspace capacity is not monotone by level");
        result.push_back(cumulative);
      }
      return result;
    }

    [[nodiscard]] std::uint64_t resident_storage_bytes() const {
      if (!capacity)
        throw std::logic_error(
            "prepared AMR Program interface invocation binding has no capacity authority");
      return workspace_storage_bytes(workspaces ? std::span<const CouplingLevelWorkspace>(
                                                      workspaces->data(), workspaces->size())
                                                : std::span<const CouplingLevelWorkspace>{},
                                     workspaces ? workspaces->capacity() : 0U);
    }

    /// Exact materialized C(n) prefixes.  This is the concrete counterpart of the configured
    /// walker: every prefix owns a vector reserved for exactly that hierarchy depth.
    [[nodiscard]] std::vector<std::uint64_t> materialized_workspace_storage_bytes_by_level() const {
      if (!workspaces || workspaces->empty())
        throw std::logic_error(
            "prepared AMR Program interface invocation binding has no materialized workspaces");
      std::vector<std::uint64_t> result;
      result.reserve(workspaces->size());
      for (std::size_t level_count = 1; level_count <= workspaces->size(); ++level_count)
        result.push_back(workspace_storage_bytes(
            std::span<const CouplingLevelWorkspace>(workspaces->data(), level_count), level_count));
      return result;
    }

    [[nodiscard]] static std::uint64_t workspace_storage_bytes(
        std::span<const CouplingLevelWorkspace> workspaces, std::size_t vector_capacity) {
      if (vector_capacity >
          std::numeric_limits<std::uint64_t>::max() / sizeof(CouplingLevelWorkspace))
        throw std::overflow_error(
            "prepared AMR Program interface invocation vector storage overflows uint64");
      std::uint64_t total =
          static_cast<std::uint64_t>(vector_capacity) * sizeof(CouplingLevelWorkspace);
      for (const CouplingLevelWorkspace& workspace : workspaces) {
        const auto bytes = workspace.resident_storage_bytes();
        if (bytes > std::numeric_limits<std::uint64_t>::max() - total)
          throw std::overflow_error(
              "prepared AMR Program coupling workspace storage overflows uint64");
        total += bytes;
      }
      return total;
    }

   private:
    friend class PreparedMultiBlockAmrHierarchy;
    PreparedProgramInterfaceInvocationBinding() = default;
    PreparedMultiBlockAmrHierarchy* owner = nullptr;
    std::optional<ProgramInterfaceInvocationCapacity> capacity;
    std::optional<std::vector<CouplingLevelWorkspace>> workspaces;
    bool collectively_authenticated = false;
  };

  class PreparedRestore {
   public:
    PreparedRestore(const PreparedRestore&) = delete;
    PreparedRestore& operator=(const PreparedRestore&) = delete;
    PreparedRestore(PreparedRestore&&) noexcept = default;
    PreparedRestore& operator=(PreparedRestore&&) noexcept = default;

    /// Read/write view of the completely prepared, still-unpublished restore carrier.  Cold
    /// rollback uses it to build every graph and Program execution object before the accepted
    /// hierarchy is exchanged; no caller can observe one of these states through the live owner.
    class TopologyView {
     public:
      [[nodiscard]] const auto& hierarchy() const {
        return require_().primary_publication->hierarchy();
      }
      [[nodiscard]] std::string_view spatial_contract() const {
        return require_().primary_publication->spatial_contract();
      }
      [[nodiscard]] std::uint64_t topology_epoch() const {
        return require_().primary_publication->topology_epoch();
      }
      [[nodiscard]] std::uint64_t materialization_generation() const {
        return require_().primary_publication->materialization_generation();
      }
      [[nodiscard]] const ExecutionLane& lane() const { return require_owner_().lane_; }
      [[nodiscard]] std::size_t block_count() const { return require_().additional.size() + 1U; }
      [[nodiscard]] PreparedMultiBlockAmrHierarchy& eventual_owner() const {
        return require_owner_();
      }
      [[nodiscard]] std::string_view block_identity(std::size_t block) const {
        if (block >= block_count())
          throw std::out_of_range("prepared AMR restore block identity is outside the topology");
        return require_owner_().block_identity(block);
      }
      [[nodiscard]] const field_type& state(std::size_t block, std::size_t level) const {
        const auto& restore = require_();
        if (block == 0)
          return restore.primary_publication->hierarchy().state(level);
        return restore.additional.at(block - 1U).levels.at(level);
      }
      [[nodiscard]] field_type& mutable_state(std::size_t block, std::size_t level) const {
        auto& restore = require_mutable_();
        if (block == 0)
          return restore.primary_publication->mutable_hierarchy_for_preparation().state(level);
        return restore.additional.at(block - 1U).levels.at(level);
      }
      [[nodiscard]] std::string_view collective_contract() const {
        return require_().collective_contract;
      }
      [[nodiscard]] std::string_view interface_flux_provider_contract() const {
        return require_owner_().interface_flux_provider_contract();
      }
      [[nodiscard]] std::size_t coupling_count() const noexcept {
        return restore_ != nullptr && restore_->owner != nullptr ? restore_->owner->coupling_count()
                                                                 : 0U;
      }
      [[nodiscard]] const interface_scheduler_type* interface_scheduler() const {
        const auto& restore = require_();
        if (restore.interface_scheduler)
          return std::addressof(*restore.interface_scheduler);
        return require_owner_().interface_scheduler_.get();
      }
      [[nodiscard]] ProgramBlockMap prepare_program_block_map(
          std::span<const std::string> ordered_blocks) const {
        auto& owner = require_owner_();
        if (ordered_blocks.size() != block_count())
          throw std::invalid_argument(
              "AMR Program restore block map must name every staged carrier exactly once");
        ProgramBlockMap result;
        result.canonical_indices.reserve(ordered_blocks.size());
        std::vector<bool> seen(block_count(), false);
        for (const std::string& identity : ordered_blocks) {
          const std::size_t canonical = owner.block_index(identity);
          if (seen.at(canonical))
            throw std::invalid_argument("AMR Program restore block map contains a duplicate block");
          seen[canonical] = true;
          result.canonical_indices.push_back(canonical);
        }
        result.hierarchy_contract = std::string(collective_contract());
        ExactContractBuilder exact;
        exact.text("pops.prepared-multiblock-amr.program-map")
            .scalar(std::uint32_t{1})
            .scalar(std::int32_t{Dim})
            .bytes(result.hierarchy_contract)
            .scalar(static_cast<std::uint64_t>(result.canonical_indices.size()));
        for (const std::size_t canonical : result.canonical_indices)
          exact.text(owner.block_identity(canonical)).scalar(static_cast<std::uint64_t>(canonical));
        result.exact_contract = std::move(exact).release();
        if (!all_ranks_agree_exact_ordered_byte_pairs(
                {{std::string_view("prepared-multiblock-amr-program-map"), result.exact_contract}},
                owner.lane_))
          throw std::invalid_argument("AMR restored Program block map differs between MPI ranks");
        owner.authorize_program_map_(result);
        return result;
      }

     private:
      friend class PreparedRestore;
      explicit TopologyView(PreparedRestore& restore) noexcept : restore_(&restore) {}
      [[nodiscard]] const PreparedRestore& require_() const {
        if (restore_ == nullptr || restore_->owner == nullptr || !restore_->primary_publication ||
            restore_->collectively_authenticated)
          throw std::logic_error("prepared AMR restore topology view is not live");
        return *restore_;
      }
      [[nodiscard]] PreparedRestore& require_mutable_() const {
        (void)require_();
        return *restore_;
      }
      [[nodiscard]] PreparedMultiBlockAmrHierarchy& require_owner_() const {
        auto& restore = require_mutable_();
        return *restore.owner;
      }
      PreparedRestore* restore_ = nullptr;
    };

    [[nodiscard]] TopologyView topology_view() { return TopologyView(*this); }

   private:
    friend class PreparedMultiBlockAmrHierarchy;
    friend class PreparedRegridTransactionStack;

    PreparedRestore() = default;

    PreparedMultiBlockAmrHierarchy* owner = nullptr;
    std::vector<AdditionalBlock> additional;
    std::optional<typename engine_type::PreparedRestorePublication> primary_publication;
    std::uint64_t accepted_revision = 0;
    std::uint64_t source_accepted_revision = 0;
    std::string source_collective_contract;
    std::string collective_contract;
    std::string canonical_program_contract;
    std::string coupling_registry_contract;
    std::optional<interface_scheduler_type> interface_scheduler;
    std::string restore_contract;
    bool collectively_authenticated = false;
  };

  /// Cold repair image for a topology that was already authenticated and published directly by
  /// the canonical AmrRuntime.  Unlike a regrid transaction this owns no second primary carrier:
  /// the runtime is already at the target topology.  It nevertheless keeps every multiblock
  /// resident arena, contract and optional interface image detached until the complete target is
  /// collectively authenticated.
  class PreparedExternalTopologyRefresh {
   public:
    PreparedExternalTopologyRefresh(const PreparedExternalTopologyRefresh&) = delete;
    PreparedExternalTopologyRefresh& operator=(const PreparedExternalTopologyRefresh&) = delete;
    PreparedExternalTopologyRefresh(PreparedExternalTopologyRefresh&&) noexcept = default;
    PreparedExternalTopologyRefresh& operator=(PreparedExternalTopologyRefresh&&) noexcept =
        default;

    /// Exact target topology exposed after the carrier image has been authenticated but before
    /// its no-throw publication.  Graph, Program-map and budget builders must use this view rather
    /// than borrowing the stale live multiblock contract.
    class TopologyView {
     public:
      [[nodiscard]] const hierarchy_type& hierarchy() const {
        return require_().owner_->primary_->hierarchy();
      }
      [[nodiscard]] std::string_view spatial_contract() const {
        return require_().owner_->primary_->spatial_contract();
      }
      [[nodiscard]] std::uint64_t topology_epoch() const {
        return require_().target_topology_epoch_;
      }
      [[nodiscard]] std::uint64_t materialization_generation() const {
        return require_().target_materialization_generation_;
      }
      [[nodiscard]] const ExecutionLane& lane() const { return require_().owner_->lane_; }
      [[nodiscard]] std::size_t block_count() const {
        return require_().owner_->additional_.size() + 1U;
      }
      [[nodiscard]] std::string_view block_identity(std::size_t block) const {
        const auto& refresh = require_();
        if (block >= block_count())
          throw std::out_of_range(
              "prepared AMR external-refresh block identity is outside the topology");
        return refresh.owner_->block_identity(block);
      }
      [[nodiscard]] const field_type& state(std::size_t block, std::size_t level) const {
        const auto& refresh = require_();
        return refresh.owner_->state(block, level);
      }
      [[nodiscard]] field_type& mutable_state(std::size_t block, std::size_t level) const {
        auto& refresh = require_mutable_();
        return refresh.owner_->state(block, level);
      }
      [[nodiscard]] std::string_view collective_contract() const {
        return require_().target_collective_contract_;
      }
      [[nodiscard]] PreparedMultiBlockAmrHierarchy& eventual_owner() const {
        return *require_().owner_;
      }
      [[nodiscard]] const interface_scheduler_type* interface_scheduler() const {
        const auto& refresh = require_();
        return refresh.interface_scheduler_ ? std::addressof(*refresh.interface_scheduler_)
                                            : refresh.owner_->interface_scheduler_.get();
      }
      [[nodiscard]] std::size_t coupling_count() const {
        return require_().owner_->coupling_count();
      }
      [[nodiscard]] std::string_view interface_flux_provider_contract() const {
        return require_().owner_->interface_flux_provider_contract();
      }
      [[nodiscard]] ProgramBlockMap prepare_program_block_map(
          std::span<const std::string> ordered_blocks) const {
        const auto& refresh = require_();
        auto& owner = *refresh.owner_;
        if (ordered_blocks.size() != block_count())
          throw std::invalid_argument(
              "AMR Program external-refresh block map must name every carrier exactly once");
        ProgramBlockMap result;
        result.canonical_indices.reserve(ordered_blocks.size());
        std::vector<bool> seen(block_count(), false);
        for (const std::string& identity : ordered_blocks) {
          const std::size_t canonical = owner.block_index(identity);
          if (seen.at(canonical))
            throw std::invalid_argument(
                "AMR Program external-refresh block map contains a duplicate block");
          seen[canonical] = true;
          result.canonical_indices.push_back(canonical);
        }
        result.hierarchy_contract = refresh.target_collective_contract_;
        ExactContractBuilder exact;
        exact.text("pops.prepared-multiblock-amr.program-map")
            .scalar(std::uint32_t{1})
            .scalar(std::int32_t{Dim})
            .bytes(result.hierarchy_contract)
            .scalar(static_cast<std::uint64_t>(result.canonical_indices.size()));
        for (const std::size_t canonical : result.canonical_indices)
          exact.text(owner.block_identity(canonical)).scalar(static_cast<std::uint64_t>(canonical));
        result.exact_contract = std::move(exact).release();
        if (!all_ranks_agree_exact_ordered_byte_pairs(
                {{std::string_view("prepared-multiblock-amr-program-map"), result.exact_contract}},
                owner.lane_))
          throw std::invalid_argument(
              "AMR external-refresh Program block map differs between MPI ranks");
        owner.authorize_program_map_(result);
        return result;
      }

     private:
      friend class PreparedExternalTopologyRefresh;
      explicit TopologyView(PreparedExternalTopologyRefresh& refresh) noexcept
          : refresh_(&refresh) {}
      [[nodiscard]] const PreparedExternalTopologyRefresh& require_() const {
        if (refresh_ == nullptr || refresh_->owner_ == nullptr ||
            !refresh_->collectively_authenticated_ || refresh_->published_)
          throw std::logic_error("prepared AMR external-refresh topology view is not live");
        return *refresh_;
      }
      [[nodiscard]] PreparedExternalTopologyRefresh& require_mutable_() const {
        (void)require_();
        return *refresh_;
      }
      PreparedExternalTopologyRefresh* refresh_ = nullptr;
    };

    [[nodiscard]] TopologyView topology_view() { return TopologyView(*this); }
    void execute();
    void publish_noexcept() noexcept;

   private:
    friend class PreparedMultiBlockAmrHierarchy;
    PreparedExternalTopologyRefresh() = default;

    PreparedMultiBlockAmrHierarchy* owner_ = nullptr;
    std::uint64_t source_accepted_revision_ = 0;
    std::uint64_t target_accepted_revision_ = 0;
    std::uint64_t target_topology_epoch_ = 0;
    std::uint64_t target_materialization_generation_ = 0;
    std::string source_collective_contract_;
    std::string target_collective_contract_;
    std::string target_canonical_program_contract_;
    std::string target_coupling_registry_contract_;
    std::optional<ResidentWorkspace> resident_workspace_;
    std::optional<std::vector<CouplingLevelWorkspace>> coupling_workspaces_;
    std::optional<interface_scheduler_type> interface_scheduler_;
    std::string exact_refresh_contract_;
    bool collectively_authenticated_ = false;
    bool published_ = false;
  };

  /// One-shot two-direction topology authority for an accepted-step candidate.
  ///
  /// Both directions are materialized and collectively authenticated before the forward swap.
  /// In particular, `publish_inverse_noexcept()` never reconstructs a hierarchy, a secondary
  /// carrier, an interface scheduler, or an exact contract after a failed candidate.
  class PreparedRegridTransaction {
   public:
    PreparedRegridTransaction(const PreparedRegridTransaction&) = delete;
    PreparedRegridTransaction& operator=(const PreparedRegridTransaction&) = delete;
    PreparedRegridTransaction(PreparedRegridTransaction&&) noexcept = default;
    PreparedRegridTransaction& operator=(PreparedRegridTransaction&&) noexcept = default;

    bool changes_topology() const noexcept { return changes_topology_; }
    bool candidate_published() const noexcept { return candidate_published_; }
    bool inverse_consumed() const noexcept { return inverse_consumed_; }

    /// The topology transaction owns two independent coupling images while its forward authority
    /// is staged: one for publication and one for LIFO inverse rollback.  Keep this accounting
    /// local to that ownership boundary; it is not a multiplier of the accepted workspace.
    [[nodiscard]] std::uint64_t resident_coupling_workspace_bytes() const {
      const auto add = [](std::uint64_t& total, std::uint64_t value) {
        if (value > std::numeric_limits<std::uint64_t>::max() - total)
          throw std::overflow_error(
              "prepared AMR regrid coupling workspace storage overflows uint64");
        total += value;
      };
      const auto workspace_bytes = [](const auto& workspaces) {
        return PreparedProgramInterfaceInvocationBinding::workspace_storage_bytes(
            std::span<const CouplingLevelWorkspace>(workspaces.data(), workspaces.size()),
            workspaces.capacity());
      };
      std::uint64_t total = 0;
      if (forward_coupling_workspaces_)
        add(total, workspace_bytes(*forward_coupling_workspaces_));
      if (inverse_coupling_workspaces_)
        add(total, workspace_bytes(*inverse_coupling_workspaces_));
      return total;
    }

    /// Read-only topology projected by this transaction before its forward publication.  This is
    /// deliberately a view, rather than a second hierarchy owner: callers may cold-prepare
    /// provider/history/field carriers against the exact forward layout without exposing a live
    /// topology mutation to a candidate transaction.
    class ForwardTopologyView {
     public:
      [[nodiscard]] const auto& hierarchy() const {
        return require_().forward_primary_->hierarchy();
      }
      [[nodiscard]] std::string_view spatial_contract() const {
        return require_().forward_primary_->spatial_contract();
      }
      [[nodiscard]] std::uint64_t topology_epoch() const {
        return require_().forward_topology_epoch_;
      }
      [[nodiscard]] std::uint64_t materialization_generation() const {
        return require_().forward_materialization_generation_;
      }
      [[nodiscard]] const ExecutionLane& lane() const { return require_().owner_->lane_; }
      [[nodiscard]] std::size_t block_count() const {
        return require_().forward_additional_.size() + 1U;
      }
      /// The forward engine may bind this eventual owner while Candidate is still detached, but
      /// it must obtain every layout/state/contract from this view until the transaction's
      /// no-throw publication makes the owner live.
      [[nodiscard]] PreparedMultiBlockAmrHierarchy& eventual_owner() const {
        auto& transaction = require_mutable_();
        if (transaction.owner_ == nullptr)
          throw std::logic_error("prepared AMR forward topology has no eventual owner");
        return *transaction.owner_;
      }
      [[nodiscard]] std::string_view block_identity(std::size_t block) const {
        const auto& transaction = require_();
        if (transaction.owner_ == nullptr || block >= block_count())
          throw std::out_of_range("prepared AMR forward block identity is outside the topology");
        return transaction.owner_->block_identity(block);
      }
      [[nodiscard]] const field_type& state(std::size_t block, std::size_t level) const {
        const auto& transaction = require_();
        if (block == 0)
          return transaction.forward_primary_->hierarchy().state(level);
        return transaction.forward_additional_.at(block - 1).levels.at(level);
      }
      /// Mutable access is intentionally limited to the cold forward-builder scope.  The target
      /// Fabs remain owned by the staged transaction and are never reachable by accepted readers
      /// before `publish_candidate_noexcept()`.
      [[nodiscard]] field_type& mutable_state(std::size_t block, std::size_t level) const {
        auto& transaction = require_mutable_();
        if (block == 0)
          return transaction.forward_primary_->mutable_hierarchy_for_preparation().state(level);
        return transaction.forward_additional_.at(block - 1).levels.at(level);
      }
      [[nodiscard]] std::string_view collective_contract() const {
        return require_().forward_collective_contract_;
      }
      /// The interface provider is structural and immutable across the forward transaction.
      /// Exposing its frozen declaration here keeps candidate-only consumers from reaching back
      /// into the accepted multiblock carrier after staging has begun.
      [[nodiscard]] std::string_view interface_flux_provider_contract() const {
        return require_().owner_->interface_flux_provider_contract();
      }
      [[nodiscard]] std::size_t coupling_count() const noexcept {
        return require_().owner_->coupling_count();
      }
      /// Detached rollback image of the exact forward carrier.  This is intentionally built from
      /// the transaction-owned primary/additional publications, never from the accepted owner;
      /// AcceptedSnapshot::from_forward uses it while Candidate is still private.
      [[nodiscard]] Snapshot snapshot() const {
        const auto& transaction = require_();
        return Snapshot{
            {transaction.forward_primary_->hierarchy(), transaction.forward_topology_epoch_,
             transaction.forward_materialization_generation_,
             std::string(transaction.forward_primary_->spatial_contract())},
            transaction.forward_additional_,
            transaction.target_accepted_revision_,
            transaction.forward_collective_contract_};
      }
      [[nodiscard]] const interface_scheduler_type* interface_scheduler() const {
        const auto& transaction = require_();
        // A topology-neutral transition retains the immutable scheduler declaration from its
        // source.  A topology-changing one owns the rematerialized scheduler that must seed the
        // next forward transition.
        if (transaction.forward_interface_scheduler_)
          return &*transaction.forward_interface_scheduler_;
        return transaction.owner_->interface_scheduler_.get();
      }
      /// Rebuild the finite Program block map against the forward hierarchy contract.  Block
      /// identities are structural and therefore borrowed from the accepted registry; the
      /// topology witness is deliberately the staged forward contract.
      [[nodiscard]] ProgramBlockMap prepare_program_block_map(
          std::span<const std::string> ordered_blocks) const {
        const auto& transaction = require_();
        const auto& owner = *transaction.owner_;
        if (ordered_blocks.size() != block_count())
          throw std::invalid_argument(
              "AMR Program block map must name every staged carrier exactly once");
        ProgramBlockMap result;
        result.canonical_indices.reserve(ordered_blocks.size());
        std::vector<bool> seen(block_count(), false);
        for (const std::string& identity : ordered_blocks) {
          const std::size_t canonical = owner.block_index(identity);
          if (seen[canonical])
            throw std::invalid_argument("AMR Program block map contains a duplicate block");
          seen[canonical] = true;
          result.canonical_indices.push_back(canonical);
        }
        result.hierarchy_contract = transaction.forward_collective_contract_;
        ExactContractBuilder exact;
        exact.text("pops.prepared-multiblock-amr.program-map")
            .scalar(std::uint32_t{1})
            .scalar(std::int32_t{Dim})
            .bytes(result.hierarchy_contract)
            .scalar(static_cast<std::uint64_t>(result.canonical_indices.size()));
        for (const std::size_t canonical : result.canonical_indices)
          exact.text(owner.block_identity(canonical)).scalar(static_cast<std::uint64_t>(canonical));
        result.exact_contract = std::move(exact).release();
        if (!all_ranks_agree_exact_ordered_byte_pairs(
                {{std::string_view("prepared-multiblock-amr-program-map"), result.exact_contract}},
                owner.lane_))
          throw std::invalid_argument("AMR staged Program block map differs between MPI ranks");
        owner.authorize_program_map_(result);
        return result;
      }

     private:
      friend class PreparedRegridTransaction;
      explicit ForwardTopologyView(PreparedRegridTransaction& transaction)
          : transaction_(&transaction) {}
      [[nodiscard]] const PreparedRegridTransaction& require_() const {
        if (transaction_ == nullptr || transaction_->owner_ == nullptr ||
            !transaction_->forward_primary_ || !transaction_->collectively_authenticated_ ||
            transaction_->candidate_published_ || transaction_->inverse_consumed_)
          throw std::logic_error("prepared AMR forward topology view is not live");
        return *transaction_;
      }
      [[nodiscard]] PreparedRegridTransaction& require_mutable_() const {
        (void)require_();
        return *transaction_;
      }
      PreparedRegridTransaction* transaction_ = nullptr;
    };

    [[nodiscard]] ForwardTopologyView forward_topology() { return ForwardTopologyView(*this); }

    /// Reach collective agreement for both the forward and inverse publication.
    void execute();
    /// Publish the candidate through only no-throw moves/swaps. Misuse is fail-stop.
    void publish_candidate_noexcept() noexcept;
    /// Restore the accepted topology through the prebuilt inverse. One-shot and fail-stop.
    void publish_inverse_noexcept() noexcept;

   private:
    friend class PreparedMultiBlockAmrHierarchy;

    PreparedRegridTransaction() = default;

    PreparedMultiBlockAmrHierarchy* owner_ = nullptr;
    std::optional<typename engine_type::PreparedRegridPublication> forward_primary_;
    std::vector<AdditionalBlock> forward_additional_;
    std::uint64_t source_accepted_revision_ = 0;
    std::uint64_t target_accepted_revision_ = 0;
    std::uint64_t forward_topology_epoch_ = 0;
    std::uint64_t forward_materialization_generation_ = 0;
    std::string source_collective_contract_;
    std::string candidate_collective_contract_;
    std::string forward_collective_contract_;
    std::string forward_canonical_program_contract_;
    std::string forward_coupling_registry_contract_;
    std::optional<interface_scheduler_type> forward_interface_scheduler_;
    std::optional<PreparedRestore> inverse_;
    std::optional<ResidentWorkspace> forward_resident_workspace_;
    std::optional<ResidentWorkspace> inverse_resident_workspace_;
    // A topology-changing candidate must carry the coupling arenas for both directions.  Binding
    // these fields after the no-throw publication would reintroduce cold MultiFab/mirror
    // allocation into the next Program coupling invocation.
    std::optional<std::vector<CouplingLevelWorkspace>> forward_coupling_workspaces_;
    std::optional<std::vector<CouplingLevelWorkspace>> inverse_coupling_workspaces_;
    /// Present only for a successor prepared from an earlier unpublished forward authority.
    /// `execute()` authenticates against this exact source; HiddenPublish still checks the live
    /// owner, which by then has received every preceding stack transition.
    std::optional<Snapshot> forward_source_snapshot_;
    std::string exact_transaction_contract_;
    bool changes_topology_ = false;
    bool collectively_authenticated_ = false;
    bool candidate_published_ = false;
    bool inverse_consumed_ = false;
  };

  /// Fixed-capacity LIFO authority for the finite set of topology transitions in one AMR step.
  ///
  /// The vector capacity is reserved during cold transaction binding.  Adding a transition after
  /// that boundary cannot grow the registry; an over-budget candidate is rejected before its
  /// forward publication.  Accepted-step rollback consumes inverses in reverse publication order.
  class PreparedRegridTransactionStack {
   public:
    explicit PreparedRegridTransactionStack(std::size_t maximum_transitions)
        : maximum_transitions_(maximum_transitions) {
      if (maximum_transitions_ == 0)
        throw std::invalid_argument("AMR regrid transaction stack requires a positive budget");
      transitions_.reserve(maximum_transitions_);
    }

    PreparedRegridTransactionStack(const PreparedRegridTransactionStack&) = delete;
    PreparedRegridTransactionStack& operator=(const PreparedRegridTransactionStack&) = delete;
    PreparedRegridTransactionStack(PreparedRegridTransactionStack&&) noexcept = default;
    PreparedRegridTransactionStack& operator=(PreparedRegridTransactionStack&&) noexcept = default;

    std::size_t maximum_transitions() const noexcept { return maximum_transitions_; }
    std::size_t published_transitions() const noexcept { return transitions_.size(); }
    /// Sum only the coupling-family arenas carried by the retained transition sequence.  The
    /// fixed transaction vector shell belongs to the transaction stack owner and is accounted by
    /// that family separately.
    [[nodiscard]] std::uint64_t resident_coupling_workspace_bytes() const {
      std::uint64_t total = 0;
      for (const PreparedRegridTransaction& transaction : transitions_) {
        const auto bytes = transaction.resident_coupling_workspace_bytes();
        if (bytes > std::numeric_limits<std::uint64_t>::max() - total)
          throw std::overflow_error(
              "prepared AMR regrid stack coupling workspace storage overflows uint64");
        total += bytes;
      }
      return total;
    }
    [[nodiscard]] bool has_published_transitions() const noexcept {
      return std::any_of(transitions_.begin(), transitions_.end(),
                         [](const PreparedRegridTransaction& transaction) {
                           return transaction.candidate_published();
                         });
    }
    const PreparedRegridTransaction& transaction(std::size_t index) const {
      return transitions_.at(index);
    }
    PreparedRegridTransaction& transaction(std::size_t index) { return transitions_.at(index); }

    /// Return the exact, unpublished forward topology of the most recently staged transition.
    /// The returned view is intentionally short-lived: callers must finish all fallible graph
    /// preparation before staging another transition, because the fixed-capacity vector may move
    /// its entries while it remains within its cold-reserved budget.
    [[nodiscard]] typename PreparedRegridTransaction::ForwardTopologyView
    latest_forward_topology() {
      if (transitions_.empty())
        throw std::logic_error("AMR regrid stack has no staged forward topology");
      return transitions_.back().forward_topology();
    }

    /// Authenticate and retain one transition without publishing its forward topology.  The
    /// transaction writer can now cold-prepare the remaining candidate authority against the
    /// exact forward view while accepted readers still see the old hierarchy.
    void execute_and_stage(PreparedRegridTransaction transaction) {
      if (transitions_.size() == maximum_transitions_)
        throw std::logic_error("AMR regrid transaction budget is exhausted");
      transaction.execute();
      transitions_.push_back(std::move(transaction));
    }

    /// Materialize transition N+1 from the exact forward view of transition N.  This is the only
    /// stack entry point for a cumulative AMR sweep; callers cannot accidentally fall back to the
    /// still-accepted multi-block carrier between two staged parents.
    void prepare_and_stage_successor(std::size_t parent_level,
                                     ::pops::amr::regridding::PreparedRegrid<Dim> prepared,
                                     std::vector<std::optional<field_type>> child_states) {
      if (transitions_.empty())
        throw std::logic_error("AMR regrid successor requires one staged forward transition");
      if (transitions_.size() == maximum_transitions_)
        throw std::logic_error("AMR regrid transaction budget is exhausted");
      PreparedRegridTransaction& prior = transitions_.back();
      if (prior.owner_ == nullptr || prior.candidate_published_ || prior.inverse_consumed_)
        throw std::logic_error("AMR regrid successor has no unpublished forward authority");
      auto forward = prior.forward_topology();
      PreparedRegridTransaction successor = prior.owner_->prepare_regrid_successor_transaction(
          forward, parent_level, std::move(prepared), std::move(child_states));
      execute_and_stage(std::move(successor));
    }

    /// Perform the no-throw forward swaps only after hidden memory publication.  All allocations
    /// and collective authentication happened in `execute_and_stage`.
    void publish_staged_noexcept() noexcept {
      for (PreparedRegridTransaction& transaction : transitions_) {
        if (transaction.candidate_published() || transaction.inverse_consumed())
          std::terminate();
        transaction.publish_candidate_noexcept();
      }
    }

    /// Bootstrap/restart convenience: accepted-step Program execution must use the split path.
    void execute_and_publish(PreparedRegridTransaction transaction) {
      execute_and_stage(std::move(transaction));
      publish_staged_noexcept();
    }

    /// Restore every topology candidate in reverse publication order.  This is deliberately
    /// no-throw: a stale or double-consumed inverse is an authority violation and fail-stops.
    void publish_inverse_lifo_noexcept() noexcept {
      bool has_topology_authority = false;
      for (const PreparedRegridTransaction& transaction : transitions_) {
        if (!transaction.candidate_published())
          std::terminate();
        if (!transaction.changes_topology())
          continue;
        has_topology_authority = true;
        if (transaction.inverse_consumed())
          std::terminate();
      }
      if (!has_topology_authority)
        return;
      for (auto cursor = transitions_.rbegin(); cursor != transitions_.rend(); ++cursor) {
        if (cursor->changes_topology())
          cursor->publish_inverse_noexcept();
      }
    }

    /// Seal a successful candidate generation without returning retained rollback authority to a
    /// later step. `clear()` keeps the cold-reserved capacity intact.
    void discard_after_accept_noexcept() noexcept {
      for (const PreparedRegridTransaction& transaction : transitions_) {
        if (!transaction.candidate_published() || transaction.inverse_consumed())
          std::terminate();
      }
      transitions_.clear();
    }

    /// Reset after a rejected candidate. It is valid only once every retained inverse has been
    /// consumed, and retains the exact cold binding capacity for the retry.
    void reset_for_next_candidate_noexcept() noexcept {
      for (const PreparedRegridTransaction& transaction : transitions_) {
        if (transaction.candidate_published_ && transaction.changes_topology_ &&
            !transaction.inverse_consumed_)
          std::terminate();
      }
      transitions_.clear();
    }

   private:
    std::size_t maximum_transitions_ = 0;
    std::vector<PreparedRegridTransaction> transitions_;
  };

  PreparedMultiBlockAmrHierarchy(const PreparedMultiBlockAmrHierarchy&) = delete;
  PreparedMultiBlockAmrHierarchy& operator=(const PreparedMultiBlockAmrHierarchy&) = delete;
  PreparedMultiBlockAmrHierarchy(PreparedMultiBlockAmrHierarchy&&) noexcept = default;
  PreparedMultiBlockAmrHierarchy& operator=(PreparedMultiBlockAmrHierarchy&&) noexcept = default;

  /// Prepare all carriers first, reach rank consensus, and only then duplicate the owning lane.
  /// The caller may build the canonical engine through the normal AmrRuntime collective preflight;
  /// this cutover never creates a second communicator or topology while local validation can throw.
  static PreparedMultiBlockAmrHierarchy prepare_collectively(
      const ExecutionLane& parent, engine_type primary, std::string primary_identity,
      std::vector<AdditionalBlock> additional, std::string lane_identity) {
    if (!parent.active())
      throw std::invalid_argument(
          "prepared multi-block AMR requires an active authenticated parent lane");
    std::shared_ptr<engine_type> primary_candidate;
    std::exception_ptr local_error;
    try {
      primary_candidate = std::make_shared<engine_type>(std::move(primary));
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, parent) != 0) {
      if (parent.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("prepared multi-block AMR topology allocation failed collectively");
    }
    return prepare_collectively(parent, std::move(primary_candidate), std::move(primary_identity),
                                std::move(additional), std::move(lane_identity));
  }

  static PreparedMultiBlockAmrHierarchy prepare_collectively(
      const ExecutionLane& parent, std::shared_ptr<engine_type> primary,
      std::string primary_identity, std::vector<AdditionalBlock> additional,
      std::string lane_identity) {
    if (!parent.active())
      throw std::invalid_argument(
          "prepared multi-block AMR requires an active authenticated parent lane");
    std::exception_ptr local_error;
    std::string contract;
    std::string canonical_program_contract;
    std::string qualified_lane_identity;
    try {
      if (!primary)
        throw std::invalid_argument("prepared multi-block AMR requires a topology runtime");
      validate_carriers_(*primary, primary_identity, additional);
      if (lane_identity.empty())
        throw std::invalid_argument("prepared multi-block AMR lane identity must be non-empty");
      if (lane_identity.size() >
          std::numeric_limits<std::size_t>::max() - parent.identity().size() - 1U)
        throw std::length_error("prepared multi-block AMR lane identity is too large");
      qualified_lane_identity.reserve(parent.identity().size() + 1U + lane_identity.size());
      qualified_lane_identity.assign(parent.identity());
      qualified_lane_identity.push_back('/');
      qualified_lane_identity.append(lane_identity);
      contract = exact_hierarchy_contract_(*primary, primary_identity, additional,
                                           qualified_lane_identity);
      canonical_program_contract =
          exact_canonical_program_contract_(contract, primary_identity, additional);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, parent) != 0) {
      if (parent.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("prepared multi-block AMR carrier preflight failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr"), contract}}, parent))
      throw std::invalid_argument(
          "prepared multi-block AMR carrier contract differs between MPI ranks");

    ExecutionLane lane = ExecutionLane::duplicate_collectively(parent, lane_identity);
    return PreparedMultiBlockAmrHierarchy(std::move(primary), std::move(primary_identity),
                                          std::move(additional), std::move(lane),
                                          std::move(qualified_lane_identity), std::move(contract),
                                          std::move(canonical_program_contract));
  }

  static constexpr int dimension = Dim;

  std::size_t block_count() const noexcept { return additional_.size() + 1; }
  std::size_t level_count() const noexcept { return primary_->hierarchy().num_levels(); }
  std::uint64_t accepted_revision() const noexcept { return accepted_revision_; }
  std::string_view collective_contract() const noexcept { return collective_contract_; }

  /// The accepted-step carrier copies state Fabs in place.  Its rollback image also retains the
  /// scalar publication revision, which is not encoded in a Fab and must be restored with the
  /// same topology contract.  This is intentionally a no-throw final-half operation: all
  /// topology checks occurred before the candidate acquired its writer lease.
  void restore_accepted_revision_noexcept(std::uint64_t revision,
                                          std::string_view collective_contract) noexcept {
    if (collective_contract != collective_contract_)
      std::terminate();
    accepted_revision_ = revision;
  }
  const ExecutionLane& lane() const noexcept { return lane_; }
  engine_type& topology_runtime() noexcept { return *primary_; }
  const engine_type& topology_runtime() const noexcept { return *primary_; }

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
    return block == 0 ? primary_->hierarchy().state(level) : additional_[block - 1].levels[level];
  }

  const field_type& state(std::size_t block, std::size_t level) const {
    require_level_(level);
    require_block_(block);
    return block == 0 ? primary_->hierarchy().state(level) : additional_[block - 1].levels[level];
  }

  /// Capture every accepted carrier into bind-sized rollback storage.  This path is used by the
  /// prepared subcycling engine after its topology contract has already been authenticated.
  void capture_resident_rollback() {
    std::exception_ptr local_error;
    try {
      require_resident_workspace_();
      for (std::size_t block = 0; block < block_count(); ++block)
        for (std::size_t level = 0; level < level_count(); ++level)
          copy_field_in_place_(state(block, level), resident_workspace_.rollback[block][level]);
      ::pops::device_fence(hot_fence_label_);
      resident_workspace_.rollback_revision = accepted_revision_;
      resident_workspace_.rollback_captured = true;
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error,
                          "prepared multi-block AMR resident rollback capture failed collectively");
  }

  /// Restore the latest resident capture.  A stale topology or an absent capture is an authority
  /// violation: this never reallocates or attempts a best-effort recovery.
  void restore_resident_rollback() {
    std::exception_ptr local_error;
    try {
      if (!resident_workspace_.rollback_captured)
        throw std::logic_error("prepared multi-block AMR resident rollback was not captured");
      require_resident_workspace_();
      for (std::size_t block = 0; block < block_count(); ++block)
        for (std::size_t level = 0; level < level_count(); ++level)
          copy_field_in_place_(resident_workspace_.rollback[block][level], state(block, level));
      ::pops::device_fence(hot_fence_label_);
      accepted_revision_ = resident_workspace_.rollback_revision;
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error,
                          "prepared multi-block AMR resident rollback restore failed collectively");
  }

  /// Allocation-free publication for a canonical Program map and its bind-sized candidate pack.
  /// The enclosing prepared attempt owns the resident all-level rollback image, so any rejected
  /// publication is restored by `restore_resident_rollback()` before it reaches the caller.
  void publish_resident_program_candidates(const ProgramBlockMap& map, std::size_t level,
                                           std::span<field_type* const> program_candidates) {
    std::exception_ptr local_error;
    try {
      require_resident_workspace_();
      if (!resident_workspace_.rollback_captured ||
          map.hierarchy_contract != collective_contract_ ||
          map.exact_contract != canonical_program_contract_ || level >= level_count() ||
          program_candidates.size() != block_count())
        throw std::invalid_argument("prepared multi-block AMR resident publication is stale");
      for (std::size_t block = 0; block < block_count(); ++block) {
        if (map.canonical_indices.at(block) != block || program_candidates[block] == nullptr ||
            !same_field_contract_(*program_candidates[block], state(block, level)))
          throw std::invalid_argument(
              "prepared multi-block AMR resident candidate pack is invalid");
        for (std::size_t other = 0; other < block; ++other)
          if (program_candidates[block] == program_candidates[other])
            throw std::invalid_argument(
                "prepared multi-block AMR resident candidate pack aliases blocks");
        resident_workspace_.publication_candidates[level][block] = program_candidates[block];
      }
      copy_pack_(resident_workspace_.publication_candidates[level],
                 resident_workspace_.accepted_packs[level]);
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane_.communicator()) != 0) {
      if (lane_.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("prepared multi-block AMR resident publication failed collectively");
    }
    increment_revision_collectively_();
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
    authorize_program_map_(result);
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
      for (const auto& conservation : operation.conservation_groups()) {
        if (conservation.members.empty())
          throw std::invalid_argument(
              "prepared AMR coupling conservation group has no exact state role");
        const std::string_view exact_state_role = conservation.members.front().state_role;
        for (const auto& role : conservation.members) {
          if (role.canonical_block >= block_count())
            throw std::out_of_range(
                "prepared AMR coupling conservation block is outside the carrier registry");
          if (role.owner != block_identity(role.canonical_block))
            throw std::invalid_argument(
                "prepared AMR coupling conservation owner differs from its canonical block");
          if (role.component < 0 ||
              role.component >= state(role.canonical_block, std::size_t{0}).ncomp())
            throw std::out_of_range(
                "prepared AMR coupling conservation component is outside its state carrier");
          if (role.state_role.empty() || role.state_role != exact_state_role)
            throw std::invalid_argument(
                "prepared AMR coupling conservation group mixes non-exact state roles");
        }
      }
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
      contract.sequence(operation.conservation_groups(),
                        [](ExactContractBuilder& group,
                           const runtime::system::PreparedCouplingConservationGroup& conservation) {
                          group.text(conservation.identity)
                              .scalar(conservation.absolute_tolerance)
                              .scalar(conservation.relative_tolerance)
                              .sequence(
                                  conservation.members,
                                  [](ExactContractBuilder& member,
                                     const runtime::system::PreparedCouplingStateRole& role) {
                                    member.text(role.owner)
                                        .scalar(static_cast<std::uint64_t>(role.canonical_block))
                                        .scalar(std::int32_t{role.component})
                                        .text(role.state_role);
                                  });
                        });
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
    local_error = nullptr;
    try {
      bind_coupling_workspaces_();
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error,
                          "prepared AMR resident coupling workspace bind failed collectively");
    coupling_registry_contract_.swap(exact);
    couplings_sealed_ = true;
  }

  PreparedProgramInterfaceInvocationBinding prepare_program_interface_invocation_binding(
      ProgramInterfaceInvocationCapacity capacity) const {
    PreparedProgramInterfaceInvocationBinding prepared;
    std::exception_ptr local_error;
    std::string exact;
    try {
      const bool active = capacity.active();
      if (active &&
          (capacity.clock_characters == 0 || capacity.graph_rate_application_characters == 0 ||
           capacity.stage_characters == 0))
        throw std::invalid_argument(
            "prepared AMR Program interface invocation capacity is incomplete");
      ExactContractBuilder builder;
      builder.text("pops.prepared-multiblock-amr.program-interface-invocation-capacity")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .presence(active)
          .scalar(static_cast<std::uint64_t>(capacity.clock_characters))
          .scalar(static_cast<std::uint64_t>(capacity.graph_rate_application_characters))
          .scalar(static_cast<std::uint64_t>(capacity.stage_characters));
      exact = std::move(builder).release();
      if (!capacity.exact_contract.empty() && capacity.exact_contract != exact)
        throw std::invalid_argument(
            "prepared AMR Program interface invocation capacity has a stale exact contract");
      capacity.exact_contract = exact;
      prepared.owner = const_cast<PreparedMultiBlockAmrHierarchy*>(this);
      prepared.capacity.emplace(std::move(capacity));
      // A Program binding owns the complete coupling image, not merely the optional interface
      // scheduler arena.  A generic prepared coupling is hot just as an interface coupling is,
      // so retain its workspace even when no interface scheduler was installed.
      if (couplings_sealed_) {
        prepared.workspaces.emplace(prepare_coupling_workspaces_(
            primary_->hierarchy(), additional_, static_cast<bool>(interface_scheduler_),
            collective_contract_, interface_scheduler_.get(), std::addressof(*prepared.capacity)));
      }
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error,
                          "prepared AMR Program interface invocation bind failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-amr-program-interface-invocation-capacity"), exact}},
            lane_))
      throw std::invalid_argument(
          "prepared AMR Program interface invocation capacities differ between MPI ranks");
    prepared.collectively_authenticated = true;
    return std::move(prepared);
  }

  void publish_program_interface_invocation_binding_noexcept(
      PreparedProgramInterfaceInvocationBinding&& prepared) noexcept {
    if (prepared.owner != this || !prepared.collectively_authenticated || !prepared.capacity ||
        (couplings_sealed_ && !prepared.workspaces))
      std::terminate();
    static_assert(std::is_nothrow_swappable_v<decltype(coupling_workspaces_)>);
    static_assert(std::is_nothrow_swappable_v<decltype(program_interface_invocation_capacity_)>);
    if (prepared.workspaces)
      coupling_workspaces_.swap(*prepared.workspaces);
    program_interface_invocation_capacity_.swap(prepared.capacity);
    prepared.collectively_authenticated = false;
  }
  std::size_t coupling_count() const noexcept { return couplings_.operators.size(); }
  [[nodiscard]] std::uint64_t resident_coupling_workspace_bytes() const {
    return PreparedProgramInterfaceInvocationBinding::workspace_storage_bytes(
        std::span<const CouplingLevelWorkspace>(coupling_workspaces_.data(),
                                                coupling_workspaces_.size()),
        coupling_workspaces_.capacity());
  }
  bool resident_coupling_workspace_bound() const noexcept {
    if (!couplings_sealed_ || coupling_workspaces_.size() != level_count())
      return false;
    for (std::size_t level = 0; level < level_count(); ++level)
      if (!coupling_workspace_bound_for_level_(level))
        return false;
    return true;
  }
  std::string_view coupling_registry_contract() const noexcept {
    return coupling_registry_contract_;
  }
  const std::vector<CouplingOperatorView>& coupled_operators() const noexcept {
    return couplings_.coupled_operators;
  }

  void install_interface_flux_provider(std::string provider_contract, const Geometry<Dim>& geometry,
                                       interface_installer_type installer) {
    std::exception_ptr local_error;
    std::string next_contract;
    try {
      if (provider_contract.empty() || !installer)
        throw std::invalid_argument(
            "prepared AMR interface provider requires an exact contract and installer");
      ExactContractBuilder exact;
      exact.text("pops.prepared-multiblock-amr.interface-provider")
          .scalar(std::uint32_t{1})
          .bytes(interface_provider_contract_)
          .bytes(provider_contract);
      next_contract = std::move(exact).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error,
                          "prepared AMR interface provider preflight failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr-interface-provider"), next_contract}},
            lane_))
      throw std::invalid_argument(
          "prepared AMR interface provider contracts differ between MPI ranks");

    local_error = nullptr;
    std::shared_ptr<interface_scheduler_type> candidate_scheduler;
    std::optional<std::vector<CouplingLevelWorkspace>> candidate_workspaces;
    std::string installation_witness;
    try {
      candidate_scheduler = interface_scheduler_
                                ? std::make_shared<interface_scheduler_type>(*interface_scheduler_)
                                : std::make_shared<interface_scheduler_type>();
      installer(*candidate_scheduler);
      candidate_scheduler->require_runtime_rematerialization_ready(static_cast<int>(level_count()));
      if (couplings_sealed_) {
        candidate_workspaces.emplace(
            prepare_coupling_workspaces_(primary_->hierarchy(), additional_, true,
                                         collective_contract_, candidate_scheduler.get()));
        if (candidate_workspaces->size() != level_count())
          throw std::logic_error(
              "prepared AMR interface workspace candidate has an invalid level count");
        for (std::size_t level = 0; level < level_count(); ++level) {
          const CouplingLevelWorkspace& workspace = candidate_workspaces->at(level);
          if (!workspace.workspace ||
              workspace.workspace->canonical_states().size() != block_count() ||
              workspace.interface_rhs.size() != block_count() ||
              workspace.interface_rhs_views.size() != block_count() ||
              workspace.level_contract.empty())
            throw std::logic_error(
                "prepared AMR interface workspace candidate has an invalid bound shape");
          for (std::size_t block = 0; block < block_count(); ++block)
            if (workspace.interface_rhs_views[block] != &workspace.interface_rhs[block] ||
                !same_field_contract_(workspace.interface_rhs[block], state(block, level)))
              throw std::logic_error(
                  "prepared AMR interface workspace candidate lost its RHS authority");
        }
      }
      ExactContractBuilder exact;
      exact.text("pops.prepared-multiblock-amr.interface-provider-installation")
          .scalar(std::uint32_t{1})
          .bytes(coupling_registry_contract_)
          .bytes(next_contract)
          .scalar(static_cast<std::uint8_t>(couplings_sealed_))
          .scalar(static_cast<std::uint64_t>(candidate_scheduler->size()))
          .scalar(
              static_cast<std::uint64_t>(candidate_workspaces ? candidate_workspaces->size() : 0U));
      installation_witness = std::move(exact).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane_) != 0) {
      if (lane_.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("prepared AMR interface provider installation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr-interface-provider-installation"),
              installation_witness}},
            lane_))
      throw std::invalid_argument(
          "prepared AMR interface provider installation differs between MPI ranks");
    // The final swaps are no-throw.  On any preceding local or collective refusal, the active
    // scheduler and its sealed RHS arenas stay untouched.
    interface_scheduler_.swap(candidate_scheduler);
    if (candidate_workspaces)
      coupling_workspaces_.swap(*candidate_workspaces);
    interface_lower_ = geometry.lower();
    interface_upper_ = geometry.upper();
    interface_provider_contract_.swap(next_contract);
  }

  bool has_interface_flux_provider() const noexcept {
    return interface_scheduler_ && interface_scheduler_->size() != 0;
  }

  std::string_view interface_flux_provider_contract() const noexcept {
    return interface_provider_contract_;
  }

  runtime::multiblock::InterfaceFluxProductionBudget interface_flux_production_budget() const {
    return interface_flux_production_budget(level_count());
  }

  runtime::multiblock::InterfaceFluxProductionBudget interface_flux_production_budget(
      std::size_t configured_level_count) const {
    if (configured_level_count == 0 ||
        configured_level_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument(
          "AMR interface production budget requires a finite configured hierarchy depth");
    if (!interface_scheduler_)
      return {std::vector<runtime::multiblock::InterfaceFluxProductionBudget::Level>(
                  configured_level_count),
              0, "pops.multiblock.interface-flux-production-budget/none"};
    if (configured_level_count != level_count())
      throw std::logic_error(
          "configured AMR interface production requires explicit hierarchy face capacities");
    return interface_scheduler_->production_budget(static_cast<int>(configured_level_count));
  }

  runtime::multiblock::InterfaceFluxProductionBudget interface_flux_production_budget(
      std::size_t configured_level_count,
      std::span<const std::size_t> configured_face_capacities) const {
    if (configured_level_count == 0 ||
        configured_level_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        configured_face_capacities.size() != configured_level_count)
      throw std::invalid_argument(
          "AMR configured interface production budget has an invalid hierarchy shape");
    if (!interface_scheduler_) {
      if (std::any_of(configured_face_capacities.begin(), configured_face_capacities.end(),
                      [](std::size_t value) { return value == 0; }))
        throw std::invalid_argument(
            "AMR configured interface production budget has an empty face capacity");
      return {std::vector<runtime::multiblock::InterfaceFluxProductionBudget::Level>(
                  configured_level_count),
              0, "pops.multiblock.interface-flux-configured-production-budget/none"};
    }
    return interface_scheduler_->production_budget(static_cast<int>(configured_level_count),
                                                   configured_face_capacities);
  }

  /// Apply couplings to a complete Program-owned candidate pack.  Candidates are restored in their
  /// Kokkos memory space on any local or remote execution failure; accepted hierarchy storage is
  /// never a hidden workspace.
  std::size_t apply_program_candidates(const ProgramBlockMap& map, std::size_t level, Real dt,
                                       std::span<field_type* const> program_candidates,
                                       const runtime::multiblock::BoundaryEvaluationPoint& point,
                                       interface_publication_type* publication) {
    bool local_preflight_failed = false;
    try {
      if (!couplings_sealed_ || level >= level_count() || level >= coupling_workspaces_.size() ||
          !coupling_workspace_bound_for_level_(level) || !program_map_authorized_(map) ||
          map.hierarchy_contract != collective_contract_ ||
          !std::isfinite(static_cast<double>(dt)) || dt < Real(0) ||
          program_candidates.size() != block_count() ||
          map.canonical_indices.size() != block_count() ||
          (has_interface_flux_provider() &&
           (publication == nullptr || publication->ledger == nullptr)))
        local_preflight_failed = true;
      for (std::size_t program = 0; !local_preflight_failed && program < block_count(); ++program) {
        if (program_candidates[program] == nullptr ||
            map.canonical_indices[program] >= block_count()) {
          local_preflight_failed = true;
          break;
        }
        for (std::size_t earlier = 0; earlier < program; ++earlier)
          if (map.canonical_indices[earlier] == map.canonical_indices[program])
            local_preflight_failed = true;
      }
    } catch (...) {
      local_preflight_failed = true;
    }
    if (all_reduce_max(local_preflight_failed ? 1L : 0L, lane_.communicator()) != 0)
      throw std::runtime_error(
          "prepared AMR resident coupling candidate preflight failed collectively");

    // The level and map fields are authenticated before this first workspace access.  Their
    // consensus joins the resident dt witness without a hot ExactContractBuilder allocation.
    CouplingLevelWorkspace& bound = require_coupling_workspace_(level);
    bound.workspace->patch_invocation_dt(dt);
    const auto workspace_pairs = bound.workspace->invocation_pairs();
    const std::array<ExactOrderedBytePair, 4> invocation_pairs{{
        {"prepared-multiblock-amr-coupling-level", bound.level_contract},
        {"prepared-multiblock-amr-coupling-program-map", map.exact_contract},
        workspace_pairs[0],
        workspace_pairs[1],
    }};
    if (!all_ranks_agree_exact_ordered_byte_pairs(invocation_pairs, lane_))
      throw std::invalid_argument(
          "prepared AMR resident coupling invocation differs between MPI ranks");

    auto& canonical = bound.workspace->canonical_states();
    local_preflight_failed = false;
    try {
      std::fill(canonical.begin(), canonical.end(), nullptr);
      for (std::size_t program = 0; !local_preflight_failed && program < program_candidates.size();
           ++program) {
        field_type* candidate = program_candidates[program];
        const std::size_t block = map.canonical_indices[program];
        if (!same_field_contract_(*candidate, state(block, level))) {
          local_preflight_failed = true;
          break;
        }
        for (std::size_t accepted = 0; accepted < block_count(); ++accepted)
          if (candidate == &state(accepted, level))
            local_preflight_failed = true;
        canonical[block] = candidate;
      }
      if (!local_preflight_failed)
        bound.workspace->bind_candidates(
            std::span<field_type* const>(canonical.data(), canonical.size()));
    } catch (...) {
      local_preflight_failed = true;
    }
    if (all_reduce_max(local_preflight_failed ? 1L : 0L, lane_.communicator()) != 0)
      throw std::runtime_error(
          "prepared AMR resident coupling candidate binding failed collectively");
    std::exception_ptr snapshot_error;
    try {
      bound.workspace->capture_rollback();
    } catch (...) {
      snapshot_error = std::current_exception();
    }
    collectively_rethrow_(snapshot_error,
                          "prepared AMR resident coupling rollback snapshot failed collectively");
    std::exception_ptr local_error;
    bool local_numeric_failure = false;
    try {
      couplings_.apply(dt, *bound.workspace);
      local_numeric_failure = bound.workspace->numeric_failure();
      if (has_interface_flux_provider()) {
        if (!program_interface_invocation_capacity_ ||
            !program_interface_invocation_capacity_->active())
          throw std::logic_error(
              "prepared multi-block AMR Program interface invocation is not bound");
        for (field_type& rhs : bound.interface_rhs)
          rhs.set_val(Real(0));
        interface_scheduler_->apply(point, canonical, bound.interface_rhs_views,
                                    bound.interface_invocation, publication);
        for (std::size_t block = 0; block < canonical.size(); ++block)
          pops::saxpy(*canonical[block], dt, bound.interface_rhs[block]);
      }
    } catch (...) {
      local_error = std::current_exception();
    }
    const bool failed =
        all_reduce_max(local_error || local_numeric_failure ? 1L : 0L, lane_.communicator()) != 0;
    if (!failed)
      return couplings_.operators.size() +
             (interface_scheduler_ ? interface_scheduler_->size() : 0);
    std::exception_ptr restore_error;
    try {
      bound.workspace->restore_rollback();
    } catch (...) {
      restore_error = std::current_exception();
    }
    collectively_rethrow_(restore_error,
                          "prepared AMR resident coupling rollback failed collectively");
    if (local_numeric_failure)
      throw std::runtime_error(
          "prepared AMR coupling validation failed and rolled back collectively");
    if (lane_.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("prepared AMR coupling failed and rolled back collectively");
  }

  /// Canonical-order convenience used by a provider that already owns the Program map.
  std::size_t apply_coupling_operators_at_level(std::size_t level, Real dt,
                                                std::span<field_type* const> candidates) {
    if (couplings_sealed_)
      throw std::logic_error(
          "direct AMR coupling application is unavailable after Program coupling bind; "
          "use the resident Program attempt route");
    ProgramBlockMap map;
    map.hierarchy_contract = collective_contract_;
    map.canonical_indices.resize(block_count());
    for (std::size_t block = 0; block < block_count(); ++block)
      map.canonical_indices[block] = block;
    map.exact_contract = canonical_program_contract_;
    runtime::multiblock::BoundaryEvaluationPoint point;
    point.clock = "pops.prepared-multiblock.direct";
    point.tick = 0;
    point.level = static_cast<int>(level);
    point.substep = 0;
    point.stage = 0;
    point.stage_fraction = {0, 1};
    point.dt = dt;
    point.physical_time = 0.0;
    if (interface_scheduler_)
      throw std::logic_error(
          "direct AMR coupling application cannot bypass interface accepted-state provenance");
    return apply_program_candidates(map, level, dt, candidates, point, nullptr);
  }

  /// Publish a complete Program candidate group to accepted carriers.  Every rollback allocation
  /// and every candidate validation precedes the first live write.  Publication failure restores
  /// all blocks, on every rank, before reporting the collective error.
  void publish_program_candidates(const ProgramBlockMap& map, std::size_t level,
                                  std::span<field_type* const> program_candidates) {
    if (couplings_sealed_)
      throw std::logic_error(
          "direct AMR candidate publication is unavailable after Program coupling bind; "
          "the resident attempt publication authority is required");
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

  /// Legacy direct source-only transaction.  It deliberately fails closed once couplings are
  /// sealed: the resident Program attempt owns the only canonical candidate and publication
  /// authority.  Before seal the coupling workspace is intentionally absent, so this API is not
  /// a cold-allocation bypass for the bound route either.
  std::size_t apply_and_publish_level(std::size_t level, Real dt) {
    if (couplings_sealed_)
      throw std::logic_error(
          "direct AMR coupling publication is unavailable after Program coupling bind; "
          "use the resident Program attempt route");
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
    return primary_->template prepare_transfer<Center>(
        source_level, destination_level,
        primary_->hierarchy().level(source_level).spatial_contract(),
        primary_->hierarchy().level(destination_level).spatial_contract(), kind,
        source.fab(source_local).view(), destination.fab(destination_local).view(),
        destination_region, mapping, components);
  }

  /// Materialize both directions of one topology mutation without writing the accepted hierarchy.
  ///
  /// The caller must execute the returned authority collectively before either no-throw publish.
  /// This intentionally keeps a complete old carrier/topology aggregate alive while the candidate
  /// is visible to the transaction writer, so rollback cannot require a fresh hierarchy rebuild.
  PreparedRegridTransaction prepare_regrid_transaction(
      std::size_t parent_level, ::pops::amr::regridding::PreparedRegrid<Dim> prepared,
      std::vector<std::optional<field_type>> child_states) {
    std::exception_ptr local_error;
    std::optional<Snapshot> accepted_snapshot;
    std::vector<AdditionalBlock> candidate_additional;
    std::optional<typename engine_type::PreparedRegridPublication> primary_publication;
    std::string next_collective_contract;
    std::string next_program_contract;
    std::string next_coupling_registry_contract;
    std::optional<interface_scheduler_type> next_interface_scheduler;
    std::optional<PreparedRestore> inverse;
    try {
      if (parent_level >= level_count() ||
          prepared.source_level() != primary_->hierarchy().layout(parent_level).exact_identity())
        throw std::invalid_argument("multi-block AMR regrid source is stale");
      if (child_states.size() != block_count())
        throw std::invalid_argument("multi-block AMR regrid requires one child slot per block");
      if (accepted_revision_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("multi-block AMR accepted revision overflow");
      accepted_snapshot.emplace(snapshot());
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
          primary_->prepare_regrid_publication(parent_level, prepared, std::move(child_states[0])));
      if (interface_scheduler_ && primary_publication->changes_topology()) {
        const auto state_provider = [&](std::size_t block, int level) -> field_type& {
          if (block == 0)
            return primary_publication->mutable_hierarchy_for_preparation().state(
                static_cast<std::size_t>(level));
          return candidate_additional.at(block - 1).levels.at(static_cast<std::size_t>(level));
        };
        const auto geometry_provider = [&](int level) {
          return Geometry<Dim>::from_bounds(
              primary_publication->hierarchy().layout(static_cast<std::size_t>(level)).domain(),
              interface_lower_, interface_upper_);
        };
        next_interface_scheduler.emplace(interface_scheduler_->rematerialized(
            static_cast<int>(primary_publication->hierarchy().num_levels()), state_provider,
            geometry_provider,
            runtime::multiblock::InterfaceRematerializationAuthority::BindBootstrap));
      }
      next_collective_contract = exact_hierarchy_contract_(
          primary_publication->hierarchy(), primary_publication->spatial_contract(),
          primary_identity_, candidate_additional, lane_contract_identity_);
      next_program_contract = exact_canonical_program_contract_(
          next_collective_contract, primary_identity_, candidate_additional);
      if (couplings_sealed_)
        next_coupling_registry_contract =
            exact_coupling_registry_contract_(next_collective_contract);
      if (primary_publication->changes_topology()) {
        inverse.emplace(prepare_restore(*accepted_snapshot));
        inverse->primary_publication.emplace(primary_->prepare_inverse_restore_publication(
            accepted_snapshot->primary, *primary_publication));
        inverse->source_accepted_revision = accepted_revision_ + 1U;
        inverse->source_collective_contract = next_collective_contract;
        ExactContractBuilder inverse_contract;
        inverse_contract.text("pops.prepared-multiblock-amr.regrid-inverse")
            .scalar(std::uint32_t{1})
            .scalar(std::int32_t{Dim})
            .bytes(next_collective_contract)
            .bytes(accepted_snapshot->exact_collective_contract)
            .scalar(accepted_revision_)
            .scalar(accepted_revision_ + 1U);
        inverse->restore_contract = std::move(inverse_contract).release();
      }
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error, "multi-block AMR regrid preflight failed collectively");
    // No allocation may follow the final collective preflight.  Building the transaction itself
    // owns vectors and exact-contract strings, so aggregate it under a second collective guard.
    PreparedRegridTransaction transaction;
    std::exception_ptr assembly_error;
    try {
      transaction.owner_ = this;
      transaction.forward_primary_ = std::move(primary_publication);
      if (!transaction.forward_primary_)
        throw std::logic_error("multi-block AMR regrid lost its primary publication");
      transaction.forward_additional_ = std::move(candidate_additional);
      transaction.source_accepted_revision_ = accepted_revision_;
      transaction.target_accepted_revision_ = transaction.forward_primary_->changes_topology()
                                                  ? accepted_revision_ + 1U
                                                  : accepted_revision_;
      if (transaction.forward_primary_->changes_topology()) {
        if (primary_->topology_epoch() == std::numeric_limits<std::uint64_t>::max() ||
            primary_->materialization_generation() == std::numeric_limits<std::uint64_t>::max())
          throw std::overflow_error("prepared multi-block AMR forward generation overflows");
        transaction.forward_topology_epoch_ = primary_->topology_epoch() + 1U;
        transaction.forward_materialization_generation_ =
            primary_->materialization_generation() + 1U;
      } else {
        transaction.forward_topology_epoch_ = primary_->topology_epoch();
        transaction.forward_materialization_generation_ = primary_->materialization_generation();
      }
      transaction.source_collective_contract_ = collective_contract_;
      transaction.candidate_collective_contract_ = next_collective_contract;
      transaction.forward_collective_contract_ = std::move(next_collective_contract);
      transaction.forward_canonical_program_contract_ = std::move(next_program_contract);
      transaction.forward_coupling_registry_contract_ = std::move(next_coupling_registry_contract);
      transaction.forward_interface_scheduler_ = std::move(next_interface_scheduler);
      transaction.inverse_ = std::move(inverse);
      transaction.changes_topology_ = transaction.forward_primary_->changes_topology();
      if (transaction.changes_topology_) {
        transaction.forward_resident_workspace_.emplace(prepare_resident_workspace_(
            transaction.forward_primary_->hierarchy(), transaction.forward_additional_));
        if (!transaction.inverse_ || !transaction.inverse_->primary_publication)
          throw std::logic_error("multi-block AMR regrid lost its resident inverse workspace");
        transaction.inverse_resident_workspace_.emplace(
            prepare_resident_workspace_(transaction.inverse_->primary_publication->hierarchy(),
                                        transaction.inverse_->additional));
        if (couplings_sealed_) {
          transaction.forward_coupling_workspaces_.emplace(prepare_coupling_workspaces_(
              transaction.forward_primary_->hierarchy(), transaction.forward_additional_,
              static_cast<bool>(interface_scheduler_), transaction.forward_collective_contract_,
              transaction.forward_interface_scheduler_
                  ? std::addressof(*transaction.forward_interface_scheduler_)
                  : interface_scheduler_.get()));
          transaction.inverse_coupling_workspaces_.emplace(prepare_coupling_workspaces_(
              transaction.inverse_->primary_publication->hierarchy(),
              transaction.inverse_->additional, static_cast<bool>(interface_scheduler_),
              transaction.inverse_->collective_contract,
              transaction.inverse_->interface_scheduler
                  ? std::addressof(*transaction.inverse_->interface_scheduler)
                  : interface_scheduler_.get()));
        }
      }
      ExactContractBuilder contract;
      contract.text("pops.prepared-multiblock-amr.regrid-transaction")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(prepared.exact_contract())
          .bytes(transaction.source_collective_contract_)
          .bytes(transaction.forward_collective_contract_)
          .bytes(transaction.forward_coupling_registry_contract_)
          .scalar(transaction.source_accepted_revision_)
          .scalar(transaction.target_accepted_revision_)
          .scalar(transaction.forward_topology_epoch_)
          .scalar(transaction.forward_materialization_generation_)
          .scalar(static_cast<std::uint8_t>(transaction.changes_topology_));
      if (transaction.inverse_)
        contract.bytes(transaction.inverse_->restore_contract);
      transaction.exact_transaction_contract_ = std::move(contract).release();
    } catch (...) {
      assembly_error = std::current_exception();
    }
    collectively_rethrow_(assembly_error,
                          "multi-block AMR regrid transaction assembly failed collectively");
    static_assert(std::is_nothrow_move_constructible_v<PreparedRegridTransaction>);
    return std::move(transaction);
  }

  /// Prepare a full hierarchy replacement as the same two-direction authority used by a normal
  /// accepted-step regrid.  This is deliberately not a direct publication API: callers must stage
  /// the returned transaction, build every dependent Program/field image from its forward view,
  /// and let the outer transaction own HiddenPublish and inverse rollback.
  PreparedRegridTransaction prepare_full_rebuild_transaction(Snapshot target) {
    std::optional<Snapshot> source;
    std::optional<typename engine_type::PreparedRegridPublication> primary_publication;
    std::string next_collective_contract;
    std::string next_program_contract;
    std::string next_coupling_registry_contract;
    std::optional<interface_scheduler_type> next_interface_scheduler;
    std::optional<PreparedRestore> inverse;
    std::exception_ptr local_error;
    try {
      source.emplace(snapshot());
      validate_snapshot_carriers_(target.primary.hierarchy, target.additional, additional_);
      if (source->accepted_revision == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("prepared multi-block AMR full rebuild revision overflows");
      primary_publication.emplace(primary_->prepare_full_rebuild_publication(target.primary));
      if (interface_scheduler_) {
        const auto state_provider = [&](std::size_t block, int level) -> field_type& {
          if (block == 0)
            return primary_publication->mutable_hierarchy_for_preparation().state(
                static_cast<std::size_t>(level));
          return target.additional.at(block - 1).levels.at(static_cast<std::size_t>(level));
        };
        const auto geometry_provider = [&](int level) {
          return Geometry<Dim>::from_bounds(
              primary_publication->hierarchy().layout(static_cast<std::size_t>(level)).domain(),
              interface_lower_, interface_upper_);
        };
        next_interface_scheduler.emplace(interface_scheduler_->rematerialized(
            static_cast<int>(primary_publication->hierarchy().num_levels()), state_provider,
            geometry_provider,
            runtime::multiblock::InterfaceRematerializationAuthority::BindBootstrap));
      }
      next_collective_contract = exact_hierarchy_contract_(
          primary_publication->hierarchy(), primary_publication->spatial_contract(),
          primary_identity_, target.additional, lane_contract_identity_);
      if (target.exact_collective_contract != next_collective_contract)
        throw std::invalid_argument(
            "prepared multi-block AMR full rebuild target contract is not authentic");
      next_program_contract = exact_canonical_program_contract_(
          next_collective_contract, primary_identity_, target.additional);
      if (couplings_sealed_)
        next_coupling_registry_contract =
            exact_coupling_registry_contract_(next_collective_contract);

      PreparedRestore restore;
      restore.owner = this;
      restore.additional = source->additional;
      restore.primary_publication.emplace(
          primary_->prepare_inverse_restore_publication(source->primary, *primary_publication));
      restore.accepted_revision = source->accepted_revision;
      restore.source_accepted_revision = source->accepted_revision + 1U;
      restore.source_collective_contract = next_collective_contract;
      restore.collective_contract = source->exact_collective_contract;
      restore.canonical_program_contract = canonical_program_contract_;
      restore.coupling_registry_contract = coupling_registry_contract_;
      if (interface_scheduler_)
        restore.interface_scheduler.emplace(*interface_scheduler_);
      ExactContractBuilder restore_contract;
      restore_contract.text("pops.prepared-multiblock-amr.full-rebuild-inverse")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(next_collective_contract)
          .bytes(source->exact_collective_contract)
          .scalar(source->accepted_revision)
          .scalar(source->accepted_revision + 1U);
      restore.restore_contract = std::move(restore_contract).release();
      inverse.emplace(std::move(restore));
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error,
                          "prepared multi-block AMR full rebuild preflight failed collectively");

    PreparedRegridTransaction transaction;
    std::exception_ptr assembly_error;
    try {
      if (!source || !primary_publication || !inverse)
        throw std::logic_error("prepared multi-block AMR full rebuild lost its authority");
      transaction.owner_ = this;
      transaction.forward_primary_ = std::move(primary_publication);
      transaction.forward_additional_ = std::move(target.additional);
      transaction.source_accepted_revision_ = source->accepted_revision;
      transaction.target_accepted_revision_ = source->accepted_revision + 1U;
      transaction.forward_topology_epoch_ = target.primary.topology_epoch;
      transaction.forward_materialization_generation_ = target.primary.materialization_generation;
      transaction.source_collective_contract_ = source->exact_collective_contract;
      transaction.candidate_collective_contract_ = next_collective_contract;
      transaction.forward_collective_contract_ = std::move(next_collective_contract);
      transaction.forward_canonical_program_contract_ = std::move(next_program_contract);
      transaction.forward_coupling_registry_contract_ = std::move(next_coupling_registry_contract);
      transaction.forward_interface_scheduler_ = std::move(next_interface_scheduler);
      transaction.inverse_ = std::move(inverse);
      transaction.changes_topology_ = true;
      transaction.forward_resident_workspace_.emplace(prepare_resident_workspace_(
          transaction.forward_primary_->hierarchy(), transaction.forward_additional_));
      if (!transaction.inverse_ || !transaction.inverse_->primary_publication)
        throw std::logic_error("multi-block AMR full rebuild lost its resident inverse workspace");
      transaction.inverse_resident_workspace_.emplace(
          prepare_resident_workspace_(transaction.inverse_->primary_publication->hierarchy(),
                                      transaction.inverse_->additional));
      if (couplings_sealed_) {
        transaction.forward_coupling_workspaces_.emplace(prepare_coupling_workspaces_(
            transaction.forward_primary_->hierarchy(), transaction.forward_additional_,
            static_cast<bool>(interface_scheduler_), transaction.forward_collective_contract_,
            transaction.forward_interface_scheduler_
                ? std::addressof(*transaction.forward_interface_scheduler_)
                : interface_scheduler_.get()));
        transaction.inverse_coupling_workspaces_.emplace(prepare_coupling_workspaces_(
            transaction.inverse_->primary_publication->hierarchy(),
            transaction.inverse_->additional, static_cast<bool>(interface_scheduler_),
            transaction.inverse_->collective_contract,
            transaction.inverse_->interface_scheduler
                ? std::addressof(*transaction.inverse_->interface_scheduler)
                : interface_scheduler_.get()));
      }
      ExactContractBuilder contract;
      contract.text("pops.prepared-multiblock-amr.full-rebuild-transaction")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(transaction.source_collective_contract_)
          .bytes(transaction.forward_collective_contract_)
          .bytes(transaction.forward_coupling_registry_contract_)
          .scalar(transaction.source_accepted_revision_)
          .scalar(transaction.target_accepted_revision_)
          .scalar(transaction.forward_topology_epoch_)
          .scalar(transaction.forward_materialization_generation_);
      contract.bytes(transaction.inverse_->restore_contract);
      transaction.exact_transaction_contract_ = std::move(contract).release();
    } catch (...) {
      assembly_error = std::current_exception();
    }
    collectively_rethrow_(
        assembly_error,
        "prepared multi-block AMR full rebuild transaction assembly failed collectively");
    return transaction;
  }

  /// Build a successor transaction from the last unpublished forward authority.  The source is
  /// retained in the transaction itself, so execution can authenticate a finite stack before the
  /// first HiddenPublish swap.  In particular this method must not read `primary_`, `additional_`
  /// or `accepted_revision_` as topology sources after `source` has been obtained.
  PreparedRegridTransaction prepare_regrid_successor_transaction(
      const typename PreparedRegridTransaction::ForwardTopologyView& source,
      std::size_t parent_level, ::pops::amr::regridding::PreparedRegrid<Dim> prepared,
      std::vector<std::optional<field_type>> child_states) {
    std::exception_ptr local_error;
    std::optional<Snapshot> source_snapshot;
    std::vector<AdditionalBlock> candidate_additional;
    std::optional<typename engine_type::PreparedRegridPublication> primary_publication;
    std::string next_collective_contract;
    std::string next_program_contract;
    std::string next_coupling_registry_contract;
    std::optional<interface_scheduler_type> next_interface_scheduler;
    std::optional<PreparedRestore> inverse;
    try {
      source_snapshot.emplace(source.snapshot());
      if (parent_level >= source_snapshot->primary.hierarchy.num_levels() ||
          prepared.source_level() !=
              source_snapshot->primary.hierarchy.layout(parent_level).exact_identity())
        throw std::invalid_argument("multi-block AMR forward regrid source is stale");
      if (child_states.size() != source_snapshot->additional.size() + 1U)
        throw std::invalid_argument("multi-block AMR forward regrid requires one child per block");
      if (source_snapshot->accepted_revision == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("multi-block AMR forward accepted revision overflow");
      if (prepared.removes_fine_level()) {
        if (std::any_of(child_states.begin(), child_states.end(),
                        [](const auto& child) { return child.has_value(); }))
          throw std::invalid_argument("removing an AMR level cannot publish child carriers");
      } else if (!prepared.fine_layout() || !prepared.ownership()) {
        throw std::invalid_argument("multi-block AMR forward regrid lost its child layout");
      }

      candidate_additional.reserve(source_snapshot->additional.size());
      for (std::size_t index = 0; index < source_snapshot->additional.size(); ++index) {
        const AdditionalBlock& prior = source_snapshot->additional[index];
        AdditionalBlock candidate;
        candidate.identity = prior.identity;
        const std::size_t retained = std::min(parent_level + 1U, prior.levels.size());
        candidate.levels.reserve(parent_level + 2U);
        for (std::size_t level = 0; level < retained; ++level)
          candidate.levels.push_back(prior.levels[level]);
        if (!prepared.removes_fine_level()) {
          if (!child_states[index + 1U])
            throw std::invalid_argument("multi-block AMR forward regrid is missing child storage");
          require_child_contract_from_parent_(prior.levels.at(parent_level),
                                              *prepared.fine_layout(), *child_states[index + 1U]);
          candidate.levels.push_back(std::move(*child_states[index + 1U]));
        }
        candidate_additional.push_back(std::move(candidate));
      }
      if (!prepared.removes_fine_level() && !child_states[0])
        throw std::invalid_argument("multi-block AMR forward regrid is missing primary child");
      primary_publication.emplace(primary_->prepare_regrid_publication_from_snapshot(
          source_snapshot->primary, parent_level, prepared, std::move(child_states[0])));
      if (source.interface_scheduler() && primary_publication->changes_topology()) {
        const auto state_provider = [&](std::size_t block, int level) -> field_type& {
          if (block == 0)
            return primary_publication->mutable_hierarchy_for_preparation().state(
                static_cast<std::size_t>(level));
          return candidate_additional.at(block - 1U).levels.at(static_cast<std::size_t>(level));
        };
        const auto geometry_provider = [&](int level) {
          return Geometry<Dim>::from_bounds(
              primary_publication->hierarchy().layout(static_cast<std::size_t>(level)).domain(),
              interface_lower_, interface_upper_);
        };
        next_interface_scheduler.emplace(source.interface_scheduler()->rematerialized(
            static_cast<int>(primary_publication->hierarchy().num_levels()), state_provider,
            geometry_provider,
            runtime::multiblock::InterfaceRematerializationAuthority::BindBootstrap));
      }
      next_collective_contract = exact_hierarchy_contract_(
          primary_publication->hierarchy(), primary_publication->spatial_contract(),
          primary_identity_, candidate_additional, lane_contract_identity_);
      next_program_contract = exact_canonical_program_contract_(
          next_collective_contract, primary_identity_, candidate_additional);
      if (couplings_sealed_)
        next_coupling_registry_contract =
            exact_coupling_registry_contract_(next_collective_contract);
      if (primary_publication->changes_topology()) {
        PreparedRestore restore;
        restore.owner = this;
        restore.additional = source_snapshot->additional;
        restore.primary_publication.emplace(
            primary_->prepare_inverse_restore_publication_from_snapshot(source_snapshot->primary,
                                                                        *primary_publication));
        restore.collective_contract = source_snapshot->exact_collective_contract;
        restore.canonical_program_contract = exact_canonical_program_contract_(
            restore.collective_contract, primary_identity_, restore.additional);
        if (couplings_sealed_)
          restore.coupling_registry_contract =
              exact_coupling_registry_contract_(restore.collective_contract);
        if (source.interface_scheduler())
          restore.interface_scheduler.emplace(*source.interface_scheduler());
        restore.accepted_revision = source_snapshot->accepted_revision;
        restore.source_accepted_revision = source_snapshot->accepted_revision + 1U;
        restore.source_collective_contract = next_collective_contract;
        ExactContractBuilder inverse_contract;
        inverse_contract.text("pops.prepared-multiblock-amr.regrid-inverse")
            .scalar(std::uint32_t{1})
            .scalar(std::int32_t{Dim})
            .bytes(next_collective_contract)
            .bytes(source_snapshot->exact_collective_contract)
            .scalar(source_snapshot->accepted_revision)
            .scalar(source_snapshot->accepted_revision + 1U);
        restore.restore_contract = std::move(inverse_contract).release();
        inverse.emplace(std::move(restore));
      }
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error,
                          "multi-block AMR forward regrid preflight failed collectively");

    PreparedRegridTransaction transaction;
    std::exception_ptr assembly_error;
    try {
      if (!source_snapshot || !primary_publication)
        throw std::logic_error("multi-block AMR forward regrid lost its source authority");
      transaction.owner_ = this;
      transaction.forward_source_snapshot_ = std::move(*source_snapshot);
      transaction.forward_primary_ = std::move(primary_publication);
      transaction.forward_additional_ = std::move(candidate_additional);
      transaction.source_accepted_revision_ =
          transaction.forward_source_snapshot_->accepted_revision;
      transaction.target_accepted_revision_ = transaction.forward_primary_->changes_topology()
                                                  ? transaction.source_accepted_revision_ + 1U
                                                  : transaction.source_accepted_revision_;
      transaction.forward_topology_epoch_ =
          transaction.forward_primary_->changes_topology()
              ? transaction.forward_source_snapshot_->primary.topology_epoch + 1U
              : transaction.forward_source_snapshot_->primary.topology_epoch;
      transaction.forward_materialization_generation_ =
          transaction.forward_primary_->changes_topology()
              ? transaction.forward_source_snapshot_->primary.materialization_generation + 1U
              : transaction.forward_source_snapshot_->primary.materialization_generation;
      transaction.source_collective_contract_ =
          transaction.forward_source_snapshot_->exact_collective_contract;
      transaction.candidate_collective_contract_ = next_collective_contract;
      transaction.forward_collective_contract_ = std::move(next_collective_contract);
      transaction.forward_canonical_program_contract_ = std::move(next_program_contract);
      transaction.forward_coupling_registry_contract_ = std::move(next_coupling_registry_contract);
      transaction.forward_interface_scheduler_ = std::move(next_interface_scheduler);
      transaction.inverse_ = std::move(inverse);
      transaction.changes_topology_ = transaction.forward_primary_->changes_topology();
      if (transaction.changes_topology_) {
        transaction.forward_resident_workspace_.emplace(prepare_resident_workspace_(
            transaction.forward_primary_->hierarchy(), transaction.forward_additional_));
        if (!transaction.inverse_ || !transaction.inverse_->primary_publication)
          throw std::logic_error(
              "multi-block AMR forward regrid lost its resident inverse workspace");
        transaction.inverse_resident_workspace_.emplace(
            prepare_resident_workspace_(transaction.inverse_->primary_publication->hierarchy(),
                                        transaction.inverse_->additional));
        if (couplings_sealed_) {
          transaction.forward_coupling_workspaces_.emplace(prepare_coupling_workspaces_(
              transaction.forward_primary_->hierarchy(), transaction.forward_additional_,
              static_cast<bool>(interface_scheduler_), transaction.forward_collective_contract_,
              transaction.forward_interface_scheduler_
                  ? std::addressof(*transaction.forward_interface_scheduler_)
                  : interface_scheduler_.get()));
          transaction.inverse_coupling_workspaces_.emplace(prepare_coupling_workspaces_(
              transaction.inverse_->primary_publication->hierarchy(),
              transaction.inverse_->additional, static_cast<bool>(interface_scheduler_),
              transaction.inverse_->collective_contract,
              transaction.inverse_->interface_scheduler
                  ? std::addressof(*transaction.inverse_->interface_scheduler)
                  : interface_scheduler_.get()));
        }
      }
      ExactContractBuilder contract;
      contract.text("pops.prepared-multiblock-amr.regrid-transaction")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(prepared.exact_contract())
          .bytes(transaction.source_collective_contract_)
          .bytes(transaction.forward_collective_contract_)
          .bytes(transaction.forward_coupling_registry_contract_)
          .scalar(transaction.source_accepted_revision_)
          .scalar(transaction.target_accepted_revision_)
          .scalar(transaction.forward_topology_epoch_)
          .scalar(transaction.forward_materialization_generation_)
          .scalar(static_cast<std::uint8_t>(transaction.changes_topology_));
      if (transaction.inverse_)
        contract.bytes(transaction.inverse_->restore_contract);
      transaction.exact_transaction_contract_ = std::move(contract).release();
    } catch (...) {
      assembly_error = std::current_exception();
    }
    collectively_rethrow_(
        assembly_error, "multi-block AMR forward regrid transaction assembly failed collectively");
    return transaction;
  }

  /// Legacy convenience for a caller that does not need the inverse after successful publication.
  /// Candidate/inverse materialization and collective authentication remain identical to the
  /// transaction path above.
  void publish_regrid(std::size_t parent_level,
                      ::pops::amr::regridding::PreparedRegrid<Dim> prepared,
                      std::vector<std::optional<field_type>> child_states) {
    PreparedRegridTransaction transaction =
        prepare_regrid_transaction(parent_level, std::move(prepared), std::move(child_states));
    transaction.execute();
    transaction.publish_candidate_noexcept();
  }

  /// Authenticate the two move-only publications before the first accepted write.
  void execute_prepared_regrid_transaction(PreparedRegridTransaction& transaction) const {
    std::exception_ptr local_error;
    try {
      const bool successor = transaction.forward_source_snapshot_.has_value();
      const std::uint64_t expected_revision =
          successor ? transaction.forward_source_snapshot_->accepted_revision : accepted_revision_;
      const std::string_view expected_contract =
          successor
              ? std::string_view(transaction.forward_source_snapshot_->exact_collective_contract)
              : std::string_view(collective_contract_);
      if (transaction.owner_ != this || !transaction.forward_primary_ ||
          transaction.collectively_authenticated_ || transaction.candidate_published_ ||
          transaction.inverse_consumed_ ||
          transaction.source_accepted_revision_ != expected_revision ||
          transaction.source_collective_contract_ != expected_contract)
        throw std::invalid_argument("prepared multi-block AMR regrid transaction is stale");
      if (successor) {
        primary_->authenticate_prepared_regrid_publication_from_snapshot(
            transaction.forward_source_snapshot_->primary, *transaction.forward_primary_);
      } else {
        primary_->authenticate_prepared_regrid_publication(*transaction.forward_primary_);
      }
      if (transaction.changes_topology_) {
        if (!transaction.inverse_ ||
            transaction.target_accepted_revision_ != expected_revision + 1U)
          throw std::invalid_argument(
              "prepared multi-block AMR topology transaction lost its inverse authority");
        if (successor) {
          primary_->authenticate_inverse_restore_publication_from_snapshot(
              transaction.forward_source_snapshot_->primary,
              *transaction.inverse_->primary_publication, *transaction.forward_primary_);
        } else {
          primary_->authenticate_inverse_restore_publication(
              *transaction.inverse_->primary_publication, *transaction.forward_primary_);
        }
        if (transaction.inverse_->source_accepted_revision !=
                transaction.target_accepted_revision_ ||
            transaction.inverse_->source_collective_contract !=
                transaction.forward_collective_contract_)
          throw std::invalid_argument(
              "prepared multi-block AMR inverse revision authority is not authentic");
      } else if (transaction.inverse_ ||
                 transaction.target_accepted_revision_ != accepted_revision_) {
        throw std::invalid_argument(
            "topology-neutral prepared multi-block regrid carries an inverse authority");
      }
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(
        local_error, "prepared multi-block AMR regrid transaction execution failed collectively");
    // The transaction's exact strings were assembled before the candidate writer entered this
    // path.  Keep the collective witness fixed-size as well: regrid execution is hot and must
    // not allocate merely to enumerate its two or three already-owned contracts.
    const std::array<std::pair<std::string_view, std::string_view>, 3> contracts{{
        {"prepared-multiblock-amr-regrid-transaction", transaction.exact_transaction_contract_},
        {"prepared-multiblock-amr-regrid-forward", transaction.forward_collective_contract_},
        {"prepared-multiblock-amr-regrid-inverse",
         transaction.inverse_ ? std::string_view(transaction.inverse_->restore_contract)
                              : std::string_view{}},
    }};
    const std::span<const std::pair<std::string_view, std::string_view>> active_contracts(
        contracts.data(), transaction.inverse_ ? 3U : 2U);
    if (!all_ranks_agree_exact_ordered_byte_pairs(active_contracts, lane_))
      throw std::invalid_argument(
          "prepared multi-block AMR regrid transaction differs between MPI ranks");
    transaction.collectively_authenticated_ = true;
  }

  void publish_prepared_regrid_candidate_noexcept(PreparedRegridTransaction& transaction) noexcept {
    if (transaction.owner_ != this || !transaction.forward_primary_ ||
        !transaction.collectively_authenticated_ || transaction.candidate_published_ ||
        transaction.inverse_consumed_ ||
        transaction.source_accepted_revision_ != accepted_revision_ ||
        transaction.source_collective_contract_ != collective_contract_)
      std::terminate();
    try {
      primary_->publish_authenticated_regrid_noexcept(std::move(*transaction.forward_primary_));
      if (transaction.changes_topology_) {
        if (!transaction.forward_resident_workspace_ ||
            (couplings_sealed_ && !transaction.forward_coupling_workspaces_))
          std::terminate();
        additional_.swap(transaction.forward_additional_);
        accepted_revision_ = transaction.target_accepted_revision_;
        collective_contract_.swap(transaction.forward_collective_contract_);
        canonical_program_contract_.swap(transaction.forward_canonical_program_contract_);
        if (couplings_sealed_)
          coupling_registry_contract_.swap(transaction.forward_coupling_registry_contract_);
        if (transaction.forward_interface_scheduler_)
          interface_scheduler_->swap(*transaction.forward_interface_scheduler_);
        resident_workspace_ = std::move(*transaction.forward_resident_workspace_);
        bind_resident_workspace_pointers_();
        if (couplings_sealed_)
          coupling_workspaces_ = std::move(*transaction.forward_coupling_workspaces_);
      }
      transaction.candidate_published_ = true;
    } catch (...) {
      std::terminate();
    }
  }

  void publish_prepared_regrid_inverse_noexcept(PreparedRegridTransaction& transaction) noexcept {
    if (transaction.owner_ != this || !transaction.collectively_authenticated_ ||
        !transaction.candidate_published_ || transaction.inverse_consumed_ ||
        !transaction.changes_topology_ || !transaction.inverse_ ||
        transaction.target_accepted_revision_ != accepted_revision_ ||
        transaction.candidate_collective_contract_ != collective_contract_)
      std::terminate();
    try {
      primary_->publish_authenticated_restore_noexcept(
          std::move(*transaction.inverse_->primary_publication));
      if (!transaction.inverse_resident_workspace_ ||
          (couplings_sealed_ && !transaction.inverse_coupling_workspaces_))
        std::terminate();
      additional_.swap(transaction.inverse_->additional);
      accepted_revision_ = transaction.inverse_->accepted_revision;
      collective_contract_.swap(transaction.inverse_->collective_contract);
      canonical_program_contract_.swap(transaction.inverse_->canonical_program_contract);
      if (couplings_sealed_)
        coupling_registry_contract_.swap(transaction.inverse_->coupling_registry_contract);
      if (transaction.inverse_->interface_scheduler)
        interface_scheduler_->swap(*transaction.inverse_->interface_scheduler);
      resident_workspace_ = std::move(*transaction.inverse_resident_workspace_);
      bind_resident_workspace_pointers_();
      if (couplings_sealed_)
        coupling_workspaces_ = std::move(*transaction.inverse_coupling_workspaces_);
      transaction.inverse_consumed_ = true;
    } catch (...) {
      std::terminate();
    }
  }

  /// True only while the resident multiblock carrier still describes the already-published raw
  /// runtime topology.  This deliberately compares the structural hierarchy witness, rather than
  /// the generated Program graph, so ordinary package refreshes do not enter the external repair
  /// protocol.
  [[nodiscard]] bool current_topology_contract_matches_runtime() const noexcept {
    try {
      return collective_contract_ == exact_hierarchy_contract_(*primary_, primary_identity_,
                                                               additional_,
                                                               lane_contract_identity_) &&
             canonical_program_contract_ == exact_canonical_program_contract_(collective_contract_,
                                                                              primary_identity_,
                                                                              additional_) &&
             (!couplings_sealed_ || coupling_registry_contract_ ==
                                        exact_coupling_registry_contract_(collective_contract_));
    } catch (...) {
      return false;
    }
  }

  /// Materialize every resident carrier that is structurally derived from a raw-engine topology.
  /// The primary runtime has already authenticated its mutation; this method only prepares the
  /// independent multi-block image required to make that accepted primary observable to Program
  /// execution again.
  PreparedExternalTopologyRefresh prepare_external_topology_refresh() {
    PreparedExternalTopologyRefresh prepared;
    std::exception_ptr local_error;
    try {
      if (current_topology_contract_matches_runtime())
        throw std::logic_error(
            "prepared multi-block AMR external topology refresh has no stale carrier");
      if (accepted_revision_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("multi-block AMR external accepted revision overflow");
      validate_carriers_(*primary_, primary_identity_, additional_);
      prepared.owner_ = this;
      prepared.source_accepted_revision_ = accepted_revision_;
      prepared.target_accepted_revision_ = prepared.source_accepted_revision_ + 1U;
      prepared.source_collective_contract_ = collective_contract_;
      prepared.target_topology_epoch_ = primary_->topology_epoch();
      prepared.target_materialization_generation_ = primary_->materialization_generation();
      prepared.target_collective_contract_ = exact_hierarchy_contract_(
          *primary_, primary_identity_, additional_, lane_contract_identity_);
      prepared.target_canonical_program_contract_ = exact_canonical_program_contract_(
          prepared.target_collective_contract_, primary_identity_, additional_);
      if (couplings_sealed_)
        prepared.target_coupling_registry_contract_ =
            exact_coupling_registry_contract_(prepared.target_collective_contract_);

      prepared.resident_workspace_.emplace(
          prepare_resident_workspace_(primary_->hierarchy(), additional_));
      bind_resident_workspace_pointers_for_(*prepared.resident_workspace_, primary_->hierarchy(),
                                            additional_);

      if (interface_scheduler_) {
        const auto state_provider = [&](std::size_t block, int level) -> field_type& {
          return state(block, static_cast<std::size_t>(level));
        };
        const auto geometry_provider = [&](int level) {
          return Geometry<Dim>::from_bounds(
              primary_->hierarchy().layout(static_cast<std::size_t>(level)).domain(),
              interface_lower_, interface_upper_);
        };
        prepared.interface_scheduler_.emplace(interface_scheduler_->rematerialized(
            static_cast<int>(primary_->hierarchy().num_levels()), state_provider, geometry_provider,
            runtime::multiblock::InterfaceRematerializationAuthority::BindBootstrap));
      }
      if (couplings_sealed_) {
        const ProgramInterfaceInvocationCapacity* invocation_capacity =
            program_interface_invocation_capacity_
                ? std::addressof(*program_interface_invocation_capacity_)
                : nullptr;
        prepared.coupling_workspaces_.emplace(prepare_coupling_workspaces_(
            primary_->hierarchy(), additional_, static_cast<bool>(interface_scheduler_),
            prepared.target_collective_contract_,
            prepared.interface_scheduler_ ? std::addressof(*prepared.interface_scheduler_)
                                          : interface_scheduler_.get(),
            invocation_capacity));
      }

      ExactContractBuilder exact;
      exact.text("pops.prepared-multiblock-amr.external-topology-refresh")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(prepared.source_collective_contract_)
          .bytes(prepared.target_collective_contract_)
          .bytes(prepared.target_canonical_program_contract_)
          .bytes(prepared.target_coupling_registry_contract_)
          .scalar(prepared.source_accepted_revision_)
          .scalar(prepared.target_accepted_revision_)
          .scalar(prepared.target_topology_epoch_)
          .scalar(prepared.target_materialization_generation_)
          .presence(prepared.interface_scheduler_.has_value())
          .presence(prepared.coupling_workspaces_.has_value());
      prepared.exact_refresh_contract_ = std::move(exact).release();
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(
        local_error,
        "prepared multi-block AMR external topology refresh preparation failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr-external-topology-refresh"),
              prepared.exact_refresh_contract_}},
            lane_))
      throw std::invalid_argument(
          "prepared multi-block AMR external topology refresh differs between MPI ranks");
    return std::move(prepared);
  }

  void execute_prepared_external_topology_refresh(PreparedExternalTopologyRefresh& prepared) {
    std::exception_ptr local_error;
    try {
      if (prepared.owner_ != this || !prepared.resident_workspace_ ||
          prepared.collectively_authenticated_ || prepared.published_ ||
          prepared.source_accepted_revision_ != accepted_revision_ ||
          prepared.source_accepted_revision_ == std::numeric_limits<std::uint64_t>::max() ||
          prepared.target_accepted_revision_ != prepared.source_accepted_revision_ + 1U ||
          prepared.source_collective_contract_ != collective_contract_ ||
          prepared.target_topology_epoch_ != primary_->topology_epoch() ||
          prepared.target_materialization_generation_ != primary_->materialization_generation())
        throw std::invalid_argument("prepared multi-block AMR external topology refresh is stale");
      validate_carriers_(*primary_, primary_identity_, additional_);
      const std::string target_collective = exact_hierarchy_contract_(
          *primary_, primary_identity_, additional_, lane_contract_identity_);
      if (target_collective != prepared.target_collective_contract_ ||
          exact_canonical_program_contract_(target_collective, primary_identity_, additional_) !=
              prepared.target_canonical_program_contract_ ||
          (couplings_sealed_ && exact_coupling_registry_contract_(target_collective) !=
                                    prepared.target_coupling_registry_contract_))
        throw std::invalid_argument(
            "prepared multi-block AMR external topology refresh contract is not authentic");

      const ResidentWorkspace& resident = *prepared.resident_workspace_;
      const std::size_t levels = primary_->hierarchy().num_levels();
      const std::size_t blocks = additional_.size() + 1U;
      if (resident.rollback.size() != blocks || resident.accepted_packs.size() != levels ||
          resident.publication_candidates.size() != levels)
        throw std::invalid_argument(
            "prepared multi-block AMR external refresh resident workspace has an invalid shape");
      for (std::size_t block = 0; block < blocks; ++block) {
        if (resident.rollback[block].size() != levels)
          throw std::invalid_argument(
              "prepared multi-block AMR external refresh rollback workspace is stale");
        for (std::size_t level = 0; level < levels; ++level) {
          const field_type& accepted = state(block, level);
          if (!same_field_contract_(accepted, resident.rollback[block][level]) ||
              resident.accepted_packs[level].size() != blocks ||
              resident.publication_candidates[level].size() != blocks ||
              resident.accepted_packs[level][block] != std::addressof(accepted))
            throw std::invalid_argument(
                "prepared multi-block AMR external refresh resident workspace lost its carrier");
        }
      }

      if (interface_scheduler_) {
        if (!prepared.interface_scheduler_)
          throw std::invalid_argument(
              "prepared multi-block AMR external refresh lost its interface scheduler");
        prepared.interface_scheduler_->require_runtime_rematerialization_ready(
            static_cast<int>(levels));
      } else if (prepared.interface_scheduler_) {
        throw std::invalid_argument(
            "prepared multi-block AMR external refresh introduced an interface scheduler");
      }
      if (couplings_sealed_) {
        if (!prepared.coupling_workspaces_ || prepared.coupling_workspaces_->size() != levels)
          throw std::invalid_argument(
              "prepared multi-block AMR external refresh lost its coupling workspaces");
        for (std::size_t level = 0; level < levels; ++level) {
          const CouplingLevelWorkspace& workspace = prepared.coupling_workspaces_->at(level);
          if (!workspace.workspace || workspace.workspace->canonical_states().size() != blocks ||
              workspace.level_contract.empty())
            throw std::invalid_argument(
                "prepared multi-block AMR external refresh coupling workspace is incomplete");
          if (interface_scheduler_) {
            if (workspace.interface_rhs.size() != blocks ||
                workspace.interface_rhs_views.size() != blocks)
              throw std::invalid_argument(
                  "prepared multi-block AMR external refresh interface workspace is incomplete");
            for (std::size_t block = 0; block < blocks; ++block)
              if (workspace.interface_rhs_views[block] != &workspace.interface_rhs[block] ||
                  !same_field_contract_(workspace.interface_rhs[block], state(block, level)))
                throw std::invalid_argument(
                    "prepared multi-block AMR external refresh interface workspace is stale");
          } else if (!workspace.interface_rhs.empty() || !workspace.interface_rhs_views.empty()) {
            throw std::invalid_argument(
                "prepared multi-block AMR external refresh has an unexpected interface workspace");
          }
        }
      } else if (prepared.coupling_workspaces_) {
        throw std::invalid_argument(
            "prepared multi-block AMR external refresh introduced coupling workspaces");
      }
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(
        local_error,
        "prepared multi-block AMR external topology refresh execution failed collectively");
    const std::array<std::pair<std::string_view, std::string_view>, 4> contracts{{
        {"prepared-multiblock-amr-external-topology-refresh", prepared.exact_refresh_contract_},
        {"prepared-multiblock-amr-external-topology-refresh-collective",
         prepared.target_collective_contract_},
        {"prepared-multiblock-amr-external-topology-refresh-program",
         prepared.target_canonical_program_contract_},
        {"prepared-multiblock-amr-external-topology-refresh-coupling",
         prepared.target_coupling_registry_contract_},
    }};
    const std::span<const std::pair<std::string_view, std::string_view>> active_contracts(
        contracts.data(), couplings_sealed_ ? 4U : 3U);
    if (!all_ranks_agree_exact_ordered_byte_pairs(active_contracts, lane_))
      throw std::invalid_argument(
          "prepared multi-block AMR external topology refresh contracts differ between MPI ranks");
    prepared.collectively_authenticated_ = true;
  }

  void publish_prepared_external_topology_refresh_noexcept(
      PreparedExternalTopologyRefresh& prepared) noexcept {
    if (prepared.owner_ != this || !prepared.collectively_authenticated_ || prepared.published_ ||
        !prepared.resident_workspace_ || prepared.source_accepted_revision_ != accepted_revision_ ||
        prepared.source_accepted_revision_ == std::numeric_limits<std::uint64_t>::max() ||
        prepared.target_accepted_revision_ != prepared.source_accepted_revision_ + 1U ||
        prepared.source_collective_contract_ != collective_contract_ ||
        prepared.target_topology_epoch_ != primary_->topology_epoch() ||
        prepared.target_materialization_generation_ != primary_->materialization_generation() ||
        (couplings_sealed_ && !prepared.coupling_workspaces_))
      std::terminate();
    static_assert(std::is_nothrow_swappable_v<ResidentWorkspace>);
    static_assert(std::is_nothrow_swappable_v<std::vector<CouplingLevelWorkspace>>);
    static_assert(std::is_nothrow_swappable_v<interface_scheduler_type>);
    static_assert(std::is_nothrow_swappable_v<std::string>);
    try {
      // The staged pointer packs were bound to the raw target before this point.  Publication
      // merely exchanges owners; it never allocates or performs a post-swap rebind.
      std::swap(resident_workspace_, *prepared.resident_workspace_);
      if (prepared.interface_scheduler_)
        std::swap(*interface_scheduler_, *prepared.interface_scheduler_);
      if (couplings_sealed_)
        std::swap(coupling_workspaces_, *prepared.coupling_workspaces_);
      std::swap(collective_contract_, prepared.target_collective_contract_);
      std::swap(canonical_program_contract_, prepared.target_canonical_program_contract_);
      if (couplings_sealed_)
        std::swap(coupling_registry_contract_, prepared.target_coupling_registry_contract_);
      accepted_revision_ = prepared.target_accepted_revision_;
      prepared.published_ = true;
    } catch (...) {
      std::terminate();
    }
  }

  Snapshot snapshot() const {
    return {primary_->snapshot(), additional_, accepted_revision_, collective_contract_};
  }

  /// Build the entire topology/carrier rollback image without entering the owning lane.
  PreparedRestore prepare_restore(const Snapshot& snapshot) {
    PreparedRestore prepared;
    prepared.owner = this;
    prepared.additional = snapshot.additional;
    validate_snapshot_carriers_(snapshot.primary.hierarchy, prepared.additional, additional_);
    prepared.primary_publication.emplace(primary_->prepare_restore_publication(snapshot.primary));
    if (interface_scheduler_) {
      const auto state_provider = [&](std::size_t block, int level) -> field_type& {
        if (block == 0)
          return prepared.primary_publication->mutable_hierarchy_for_preparation().state(
              static_cast<std::size_t>(level));
        return prepared.additional.at(block - 1).levels.at(static_cast<std::size_t>(level));
      };
      const auto geometry_provider = [&](int level) {
        return Geometry<Dim>::from_bounds(prepared.primary_publication->hierarchy()
                                              .layout(static_cast<std::size_t>(level))
                                              .domain(),
                                          interface_lower_, interface_upper_);
      };
      prepared.interface_scheduler.emplace(interface_scheduler_->rematerialized(
          static_cast<int>(prepared.primary_publication->hierarchy().num_levels()), state_provider,
          geometry_provider));
    }
    prepared.collective_contract = exact_hierarchy_contract_(
        prepared.primary_publication->hierarchy(), prepared.primary_publication->spatial_contract(),
        primary_identity_, prepared.additional, lane_contract_identity_);
    if (snapshot.exact_collective_contract != prepared.collective_contract) {
      const auto mismatch = std::mismatch(
          snapshot.exact_collective_contract.begin(), snapshot.exact_collective_contract.end(),
          prepared.collective_contract.begin(), prepared.collective_contract.end());
      throw std::invalid_argument(
          "prepared multi-block AMR snapshot collective contract is not authentic at byte " +
          std::to_string(static_cast<std::size_t>(mismatch.first -
                                                  snapshot.exact_collective_contract.begin())) +
          " (snapshot bytes=" + std::to_string(snapshot.exact_collective_contract.size()) +
          ", restored bytes=" + std::to_string(prepared.collective_contract.size()) + ")");
    }
    prepared.canonical_program_contract = exact_canonical_program_contract_(
        prepared.collective_contract, primary_identity_, prepared.additional);
    if (couplings_sealed_)
      prepared.coupling_registry_contract =
          exact_coupling_registry_contract_(prepared.collective_contract);
    ExactContractBuilder contract;
    contract.text("pops.prepared-multiblock-amr.restore")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(collective_contract_)
        .bytes(prepared.collective_contract)
        .scalar(snapshot.accepted_revision);
    prepared.restore_contract = std::move(contract).release();
    prepared.accepted_revision = snapshot.accepted_revision;
    prepared.source_accepted_revision = accepted_revision_;
    prepared.source_collective_contract = collective_contract_;
    return prepared;
  }

  /// Authenticate one fully prepared restore on the owning lane without publishing it.
  void execute_prepared_restore(PreparedRestore& prepared) const {
    std::exception_ptr local_error;
    try {
      if (prepared.owner != this || !prepared.primary_publication ||
          prepared.source_accepted_revision != accepted_revision_ ||
          prepared.source_collective_contract != collective_contract_)
        throw std::invalid_argument("prepared multi-block AMR restore publication is stale");
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error,
                          "prepared multi-block AMR restore execution failed collectively");
    try {
      primary_->authenticate_prepared_restore_publication(*prepared.primary_publication);
    } catch (...) {
      collectively_rethrow_(std::current_exception(),
                            "prepared multi-block AMR restore authentication failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr-restore"), prepared.restore_contract}},
            lane_))
      throw std::invalid_argument("prepared multi-block AMR restore differs between MPI ranks");
    prepared.collectively_authenticated = true;
  }

  /// Publish an authenticated restore using only no-allocation moves/swaps.
  void publish_prepared_restore(PreparedRestore&& prepared) noexcept {
    if (prepared.owner != this || !prepared.collectively_authenticated ||
        !prepared.primary_publication || prepared.source_accepted_revision != accepted_revision_ ||
        prepared.source_collective_contract != collective_contract_)
      std::terminate();
    primary_->publish_authenticated_restore_noexcept(std::move(*prepared.primary_publication));
    additional_.swap(prepared.additional);
    accepted_revision_ = prepared.accepted_revision;
    collective_contract_.swap(prepared.collective_contract);
    canonical_program_contract_.swap(prepared.canonical_program_contract);
    if (couplings_sealed_)
      coupling_registry_contract_.swap(prepared.coupling_registry_contract);
    if (prepared.interface_scheduler)
      interface_scheduler_->swap(*prepared.interface_scheduler);
  }

  /// Restore both the canonical topology and every secondary carrier through explicit phases.
  void restore(const Snapshot& snapshot) {
    std::optional<PreparedRestore> prepared;
    std::exception_ptr local_error;
    try {
      prepared.emplace(prepare_restore(snapshot));
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error, "prepared multi-block AMR restore failed collectively");
    execute_prepared_restore(*prepared);
    publish_prepared_restore(std::move(*prepared));
  }

  /// Requalify this retained physical carrier for one authenticated restart authority.
  ///
  /// The caller supplies bytes decoded from an accepted Program image, but those bytes are never
  /// trusted as a description of topology.  We derive the spatial contract from the retained live
  /// hierarchy and spatial identity under the requested generations, compare it byte-for-byte to
  /// the accepted authority, then drive the normal PreparedRestore publication protocol.
  void requalify_restart_authority_collectively(const RestartAuthority& authority) {
    std::optional<PreparedRestore> prepared;
    std::exception_ptr local_error;
    try {
      Snapshot candidate = snapshot();
      candidate.primary.topology_epoch = authority.topology_epoch;
      candidate.primary.materialization_generation = authority.materialization_generation;
      candidate.primary.exact_spatial_contract = detail::exact_runtime_spatial_contract(
          primary_->spatial_identity(), candidate.primary.hierarchy,
          candidate.primary.topology_epoch, candidate.primary.materialization_generation);
      if (candidate.primary.exact_spatial_contract != authority.spatial_contract)
        throw std::invalid_argument(
            "prepared multi-block AMR restart authority does not authenticate the rebuilt "
            "hierarchy");
      candidate.exact_collective_contract = exact_hierarchy_contract_(
          candidate.primary.hierarchy, candidate.primary.exact_spatial_contract, primary_identity_,
          candidate.additional, lane_contract_identity_);
      prepared.emplace(prepare_restore(candidate));
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(
        local_error, "prepared multi-block AMR restart authority preparation failed collectively");
    execute_prepared_restore(*prepared);
    publish_prepared_restore(std::move(*prepared));
  }

 private:
  PreparedMultiBlockAmrHierarchy(std::shared_ptr<engine_type> primary, std::string primary_identity,
                                 std::vector<AdditionalBlock> additional, ExecutionLane lane,
                                 std::string lane_contract_identity,
                                 std::string collective_contract,
                                 std::string canonical_program_contract)
      : primary_(std::move(primary)),
        primary_identity_(std::move(primary_identity)),
        additional_(std::move(additional)),
        lane_(std::move(lane)),
        lane_contract_identity_(std::move(lane_contract_identity)),
        collective_contract_(std::move(collective_contract)),
        canonical_program_contract_(std::move(canonical_program_contract)) {
    resident_workspace_ = prepare_resident_workspace_(primary_->hierarchy(), additional_);
    bind_resident_workspace_pointers_();
  }

  static void validate_carriers_(const engine_type& primary, std::string_view primary_identity,
                                 const std::vector<AdditionalBlock>& additional) {
    if (primary_identity.empty())
      throw std::invalid_argument("prepared multi-block AMR primary identity must be non-empty");
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
    if (additional.size() != live_additional.size())
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

  static void copy_field_in_place_(const field_type& source, field_type& destination) {
    if (!same_field_contract_(source, destination) ||
        source.local_size() != destination.local_size())
      throw std::invalid_argument("prepared multi-block AMR resident carrier shape changed");
    if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
      for (std::size_t local = 0; local < source.local_size(); ++local) {
        const auto& source_storage = source.fab(local).storage();
        const auto& destination_storage = destination.fab(local).storage();
        std::copy_n(source_storage.data(), source_storage.extent(0), destination_storage.data());
      }
    } else {
      for (std::size_t local = 0; local < source.local_size(); ++local)
        Kokkos::deep_copy(destination.fab(local).storage(), source.fab(local).storage());
    }
  }

  static ResidentWorkspace prepare_resident_workspace_(
      const hierarchy_type& primary, const std::vector<AdditionalBlock>& additional) {
    const std::size_t blocks = additional.size() + 1U;
    const std::size_t levels = primary.num_levels();
    ResidentWorkspace workspace;
    workspace.rollback.resize(blocks);
    workspace.accepted_packs.resize(levels);
    workspace.publication_candidates.resize(levels);
    for (std::size_t block = 0; block < blocks; ++block) {
      workspace.rollback[block].reserve(levels);
      for (std::size_t level = 0; level < levels; ++level) {
        const field_type& state =
            block == 0 ? primary.state(level) : additional[block - 1].levels.at(level);
        workspace.rollback[block].emplace_back(state);
      }
    }
    for (std::size_t level = 0; level < levels; ++level) {
      workspace.accepted_packs[level].resize(blocks, nullptr);
      workspace.publication_candidates[level].resize(blocks, nullptr);
    }
    return workspace;
  }

  static void bind_resident_workspace_pointers_for_(
      ResidentWorkspace& workspace, hierarchy_type& primary,
      std::vector<AdditionalBlock>& additional) noexcept {
    const std::size_t blocks = additional.size() + 1U;
    const std::size_t levels = primary.num_levels();
    if (workspace.accepted_packs.size() != levels ||
        workspace.publication_candidates.size() != levels)
      std::terminate();
    for (std::size_t level = 0; level < levels; ++level) {
      if (workspace.accepted_packs[level].size() != blocks ||
          workspace.publication_candidates[level].size() != blocks)
        std::terminate();
      for (std::size_t block = 0; block < blocks; ++block)
        workspace.accepted_packs[level][block] =
            block == 0 ? std::addressof(primary.state(level))
                       : std::addressof(additional[block - 1].levels.at(level));
    }
  }

  void bind_resident_workspace_pointers_() noexcept {
    bind_resident_workspace_pointers_for_(resident_workspace_, primary_->hierarchy(), additional_);
  }

  void require_resident_workspace_() const {
    if (resident_workspace_.rollback.size() != block_count() ||
        resident_workspace_.accepted_packs.size() != level_count() ||
        resident_workspace_.publication_candidates.size() != level_count())
      throw std::invalid_argument("prepared multi-block AMR resident workspace is stale");
    for (std::size_t block = 0; block < block_count(); ++block) {
      if (resident_workspace_.rollback[block].size() != level_count())
        throw std::invalid_argument("prepared multi-block AMR resident rollback shape changed");
      for (std::size_t level = 0; level < level_count(); ++level)
        if (!same_field_contract_(state(block, level), resident_workspace_.rollback[block][level]))
          throw std::invalid_argument("prepared multi-block AMR resident rollback carrier changed");
    }
    for (std::size_t level = 0; level < level_count(); ++level) {
      if (resident_workspace_.accepted_packs[level].size() != block_count() ||
          resident_workspace_.publication_candidates[level].size() != block_count())
        throw std::invalid_argument("prepared multi-block AMR resident publication shape changed");
      for (std::size_t block = 0; block < block_count(); ++block)
        if (resident_workspace_.accepted_packs[level][block] != &state(block, level))
          throw std::invalid_argument("prepared multi-block AMR resident accepted carrier changed");
    }
  }

  void bind_coupling_workspaces_() {
    if (!coupling_workspaces_.empty()) {
      if (coupling_workspaces_.size() != level_count())
        throw std::logic_error("prepared AMR resident coupling workspace level count drifted");
      return;
    }
    coupling_workspaces_ =
        prepare_coupling_workspaces_(primary_->hierarchy(), additional_,
                                     static_cast<bool>(interface_scheduler_), collective_contract_);
  }

  std::vector<CouplingLevelWorkspace> prepare_coupling_workspaces_(
      const hierarchy_type& primary, const std::vector<AdditionalBlock>& additional,
      bool requires_interface_rhs, std::string_view hierarchy_contract,
      interface_scheduler_type* scheduler = nullptr,
      const ProgramInterfaceInvocationCapacity* invocation_capacity = nullptr) const {
    std::vector<CouplingLevelWorkspace> workspaces;
    const std::size_t blocks = additional.size() + 1U;
    const std::size_t levels = primary.num_levels();
    workspaces.reserve(levels);
    for (std::size_t level = 0; level < levels; ++level) {
      CouplingLevelWorkspace level_workspace;
      ExactContractBuilder level_contract;
      level_contract.text("pops.prepared-multiblock-amr.coupling-level")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(hierarchy_contract)
          .scalar(static_cast<std::uint64_t>(level));
      const std::string exact_level_contract = std::move(level_contract).release();
      level_workspace.level_contract.reserve(exact_level_contract.size());
      level_workspace.level_contract.assign(exact_level_contract);
      level_workspace.workspace =
          std::make_unique<runtime::system::PreparedCouplingWorkspace<Dim>>();
      std::vector<const field_type*> prototypes;
      prototypes.reserve(blocks);
      for (std::size_t block = 0; block < blocks; ++block)
        prototypes.push_back(block == 0 ? &primary.state(level)
                                        : &additional[block - 1].levels.at(level));
      couplings_.bind_workspace(*level_workspace.workspace, prototypes, requires_interface_rhs);
      if (requires_interface_rhs) {
        level_workspace.interface_rhs.reserve(blocks);
        level_workspace.interface_rhs_views.reserve(blocks);
        for (const field_type* prototype : prototypes) {
          level_workspace.interface_rhs.emplace_back(*prototype);
          level_workspace.interface_rhs.back().set_val(Real(0));
        }
        for (field_type& rhs : level_workspace.interface_rhs)
          level_workspace.interface_rhs_views.push_back(&rhs);
        const ProgramInterfaceInvocationCapacity* capacity =
            invocation_capacity != nullptr
                ? invocation_capacity
                : (program_interface_invocation_capacity_
                       ? std::addressof(*program_interface_invocation_capacity_)
                       : nullptr);
        interface_scheduler_type* prepared_scheduler =
            scheduler != nullptr ? scheduler : interface_scheduler_.get();
        if (capacity != nullptr && capacity->active() && prepared_scheduler != nullptr)
          prepared_scheduler->bind_invocation_workspace(
              level_workspace.interface_invocation, capacity->clock_characters,
              capacity->graph_rate_application_characters, capacity->stage_characters);
      }
      workspaces.push_back(std::move(level_workspace));
    }
    return workspaces;
  }

  void authorize_program_map_(ProgramBlockMap& map) const {
    auto authority = std::make_shared<ProgramBlockMapAuthority>();
    authority->owner = this;
    authority->canonical_indices = map.canonical_indices;
    authority->hierarchy_contract = map.hierarchy_contract;
    authority->exact_contract = map.exact_contract;
    map.authority = std::move(authority);
  }

  bool program_map_authorized_(const ProgramBlockMap& map) const noexcept {
    const auto& authority = map.authority;
    return authority != nullptr && authority->owner == this &&
           authority->canonical_indices == map.canonical_indices &&
           authority->hierarchy_contract == map.hierarchy_contract &&
           authority->exact_contract == map.exact_contract;
  }

  bool coupling_workspace_bound_for_level_(std::size_t level) const noexcept {
    if (!couplings_sealed_ || level >= level_count() || level >= coupling_workspaces_.size())
      return false;
    const CouplingLevelWorkspace& bound = coupling_workspaces_[level];
    if (!bound.workspace || bound.workspace->canonical_states().size() != block_count() ||
        bound.level_contract.empty())
      return false;
    if (!interface_scheduler_)
      return bound.interface_rhs.empty() && bound.interface_rhs_views.empty();
    if (bound.interface_rhs.size() != block_count() ||
        bound.interface_rhs_views.size() != block_count())
      return false;
    for (std::size_t block = 0; block < block_count(); ++block)
      if (bound.interface_rhs_views[block] != &bound.interface_rhs[block] ||
          !same_field_contract_(bound.interface_rhs[block], state(block, level)))
        return false;
    return true;
  }

  CouplingLevelWorkspace& require_coupling_workspace_(std::size_t level) {
    if (!coupling_workspace_bound_for_level_(level))
      throw std::logic_error("prepared AMR resident coupling workspace is not bound");
    return coupling_workspaces_[level];
  }

  std::vector<field_type*> validate_program_candidates_(
      const ProgramBlockMap& map, std::size_t level, Real dt,
      std::span<field_type* const> program_candidates, bool require_dt_consensus = true,
      bool require_sealed_couplings = true) const {
    // Do not inspect a candidate field until the complete pointer pack has passed a collective
    // preflight.  In particular, a null pointer on one rank must prevent other ranks from
    // dereferencing their local candidates before the call has failed everywhere.
    std::exception_ptr presence_error;
    std::string presence_contract;
    try {
      require_map_(map);
      require_level_(level);
      if (require_sealed_couplings && !couplings_sealed_)
        throw std::logic_error("prepared multi-block AMR couplings must be sealed before use");
      if (program_candidates.size() != map.canonical_indices.size())
        throw std::invalid_argument("AMR Program candidate pack does not match its block map");
      if (require_dt_consensus && (!std::isfinite(static_cast<double>(dt)) || dt < Real(0)))
        throw std::invalid_argument("AMR coupling dt must be finite and non-negative");
      ExactContractBuilder presence;
      presence.text("pops.prepared-multiblock-amr.candidate-pointer-pack")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(map.exact_contract)
          .scalar(static_cast<std::uint64_t>(level))
          .scalar(static_cast<std::uint64_t>(program_candidates.size()));
      for (field_type* candidate : program_candidates) {
        if (candidate == nullptr)
          throw std::invalid_argument("AMR Program candidate pack contains a null state");
        presence.presence(true);
      }
      presence_contract = std::move(presence).release();
    } catch (...) {
      presence_error = std::current_exception();
    }
    collectively_rethrow_(presence_error,
                          "AMR Program candidate pointer preflight failed collectively");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-multiblock-amr-candidate-pointer-pack"),
              presence_contract}},
            lane_))
      throw std::invalid_argument("AMR Program candidate pointer packs differ between MPI ranks");

    std::exception_ptr local_error;
    std::vector<field_type*> canonical(block_count(), nullptr);
    std::string invocation_contract;
    try {
      for (std::size_t program = 0; program < program_candidates.size(); ++program) {
        field_type* candidate = program_candidates[program];
        const std::size_t block = map.canonical_indices[program];
        if (!same_field_contract_(*candidate, state(block, level)))
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
      ::pops::device_fence(hot_fence_label_);
    } catch (...) {
      local_error = std::current_exception();
    }
    collectively_rethrow_(local_error, "prepared multi-block AMR " + std::string(purpose) +
                                           " allocation failed collectively");
    return result;
  }

  void copy_pack_(std::span<field_type* const> source,
                  std::span<field_type* const> destination) const {
    if (source.size() != destination.size())
      throw std::invalid_argument("AMR field-pack copy requires equal pack sizes");
    for (std::size_t block = 0; block < source.size(); ++block) {
      if (!same_field_contract_(*source[block], *destination[block]))
        throw std::invalid_argument("AMR field-pack copy lost an exact carrier contract");
      copy_field_in_place_(*source[block], *destination[block]);
    }
    ::pops::device_fence(hot_fence_label_);
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
    require_child_contract_from_parent_(state(block, parent_level), layout, child);
  }

  static void require_child_contract_from_parent_(
      const field_type& parent, const ::pops::amr::hierarchy::LevelLayout<Dim>& layout,
      const field_type& child) {
    if (child.layout() != layout.patches() || child.distribution() != layout.distribution() ||
        child.local_rank() != parent.local_rank() || child.ncomp() != parent.ncomp() ||
        child.ghosts() != parent.ghosts())
      throw std::invalid_argument(
          "multi-block AMR child carrier differs from its block and prepared topology");
  }

  std::shared_ptr<engine_type> primary_;
  std::string primary_identity_;
  std::vector<AdditionalBlock> additional_;
  // Declared before all providers/registries so reverse member destruction releases their callbacks
  // first and frees the communicator last.
  ExecutionLane lane_;
  std::string lane_contract_identity_;
  coupling_registry_type couplings_;
  std::vector<CouplingLevelWorkspace> coupling_workspaces_;
  std::string coupling_registry_contract_;
  bool couplings_sealed_ = false;
  std::shared_ptr<interface_scheduler_type> interface_scheduler_;
  std::optional<ProgramInterfaceInvocationCapacity> program_interface_invocation_capacity_;
  std::string interface_provider_contract_;
  RealVector<Dim> interface_lower_{};
  RealVector<Dim> interface_upper_{};
  std::uint64_t accepted_revision_ = 0;
  std::string collective_contract_;
  std::string canonical_program_contract_;
  std::string hot_fence_label_ = "pops.prepared-multiblock-amr.hot-fence";
  ResidentWorkspace resident_workspace_;
};

template <int Dim, class MemorySpace>
void PreparedMultiBlockAmrHierarchy<Dim, MemorySpace>::PreparedExternalTopologyRefresh::execute() {
  if (owner_ == nullptr)
    throw std::logic_error("prepared multi-block AMR external topology refresh has no owner");
  owner_->execute_prepared_external_topology_refresh(*this);
}

template <int Dim, class MemorySpace>
void PreparedMultiBlockAmrHierarchy<
    Dim, MemorySpace>::PreparedExternalTopologyRefresh::publish_noexcept() noexcept {
  if (owner_ == nullptr)
    std::terminate();
  owner_->publish_prepared_external_topology_refresh_noexcept(*this);
}

template <int Dim, class MemorySpace>
void PreparedMultiBlockAmrHierarchy<Dim, MemorySpace>::PreparedRegridTransaction::execute() {
  if (owner_ == nullptr)
    throw std::logic_error("prepared multi-block AMR regrid transaction has no owner");
  owner_->execute_prepared_regrid_transaction(*this);
}

template <int Dim, class MemorySpace>
void PreparedMultiBlockAmrHierarchy<
    Dim, MemorySpace>::PreparedRegridTransaction::publish_candidate_noexcept() noexcept {
  if (owner_ == nullptr)
    std::terminate();
  owner_->publish_prepared_regrid_candidate_noexcept(*this);
}

template <int Dim, class MemorySpace>
void PreparedMultiBlockAmrHierarchy<
    Dim, MemorySpace>::PreparedRegridTransaction::publish_inverse_noexcept() noexcept {
  if (owner_ == nullptr)
    std::terminate();
  owner_->publish_prepared_regrid_inverse_noexcept(*this);
}

}  // namespace pops::runtime::amr
