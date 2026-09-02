/// @file
/// @brief Resident storage for a prepared coupling application.
///
/// This is deliberately independent of System and AMR ownership.  A binding authority supplies
/// the canonical field prototypes once, then a hot caller only patches the pointer pack and
/// copies into already allocated images/buffers.  The same object is therefore usable by the
/// Uniform facade and by a future hierarchy-level coupling adapter without giving either one a
/// second transaction authority.

#pragma once

#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::system {

/// Bind-time arena for exactly one canonical simultaneous coupling-state pack.
///
/// `rollback_states` restores an entire multi-operator application.  `conservation_before_states`
/// is refreshed before every individual operator, so a valid first operator cannot hide a bad
/// second one.  The raw host buffers are intentionally independent of Fab::HostMirror: they are
/// allocated once and may be refilled from a detached candidate of the same exact layout.
template <int Dim>
class PreparedCouplingWorkspace {
 public:
  using field_type = MultiFab<Dim>;
  using fab_type = typename field_type::fab_type;
  using raw_host_buffer_type = typename fab_type::raw_host_mirror_type;

  struct HostPatch {
    std::size_t block = 0;
    std::size_t local = 0;
    std::size_t values = 0;
    raw_host_buffer_type before{};
    raw_host_buffer_type candidate{};
  };

  /// Byte components retained by one bound workspace.  This POD is shared by the concrete
  /// resident walker and the configured AMR ceiling: neither side may fold one of these owners
  /// into an unnamed multiplier.
  struct ResidentStorageFootprint {
    std::uint64_t prototype_state_pointers = 0;
    std::uint64_t canonical_state_pointers = 0;
    std::uint64_t rollback_images = 0;
    std::uint64_t conservation_images = 0;
    std::uint64_t host_patch_descriptors = 0;
    std::uint64_t host_patch_offsets = 0;
    std::uint64_t host_patch_counts = 0;
    std::uint64_t operator_contract_slots = 0;
    std::uint64_t operator_contract_characters = 0;
    std::uint64_t invocation_characters = 0;
    std::uint64_t host_before_values = 0;
    std::uint64_t host_candidate_values = 0;
  };

  PreparedCouplingWorkspace() = default;
  PreparedCouplingWorkspace(const PreparedCouplingWorkspace&) = delete;
  PreparedCouplingWorkspace& operator=(const PreparedCouplingWorkspace&) = delete;
  PreparedCouplingWorkspace(PreparedCouplingWorkspace&&) = delete;
  PreparedCouplingWorkspace& operator=(PreparedCouplingWorkspace&&) = delete;

  /// Fixed pair protocol used by the hot exact consensus.  The static field is an
  /// ExactContractBuilder payload produced during bind; the dt field is the canonical framed
  /// scalar encoding (s|length|d/f|IEEE-big-endian bits) patched in place for each invocation.
  static constexpr std::string_view invocation_schema() noexcept {
    return "pops.system.prepared-coupling-invocation.v1";
  }

  bool bound() const noexcept { return bound_; }
  std::size_t block_count() const noexcept { return canonical_states_.size(); }
  const std::vector<field_type*>& canonical_states() const noexcept { return canonical_states_; }
  std::vector<field_type*>& canonical_states() noexcept { return canonical_states_; }
  std::string_view static_invocation_bytes() const noexcept { return invocation_static_; }

  /// Adds the named dynamic owners retained by one binding.  It is intentionally public so a
  /// configured topology can use the same checked arithmetic without fabricating a live facade.
  [[nodiscard]] static std::uint64_t resident_storage_bytes(
      const ResidentStorageFootprint& footprint) {
    const auto checked_add = [](std::uint64_t& total, std::uint64_t value) {
      if (value > std::numeric_limits<std::uint64_t>::max() - total)
        throw std::overflow_error("prepared coupling workspace storage overflows uint64");
      total += value;
    };
    std::uint64_t total = 0;
    for (const std::uint64_t value : {
             footprint.prototype_state_pointers,
             footprint.canonical_state_pointers,
             footprint.rollback_images,
             footprint.conservation_images,
             footprint.host_patch_descriptors,
             footprint.host_patch_offsets,
             footprint.host_patch_counts,
             footprint.operator_contract_slots,
             footprint.operator_contract_characters,
             footprint.invocation_characters,
             footprint.host_before_values,
             footprint.host_candidate_values,
         })
      checked_add(total, value);
    return total;
  }

  template <class Value>
  [[nodiscard]] static std::uint64_t reserved_vector_storage_bytes(std::size_t count) {
    std::vector<Value> values;
    values.reserve(count);
    return vector_storage_bytes_(values);
  }

  template <class Value>
  [[nodiscard]] static std::uint64_t grown_vector_storage_bytes(std::size_t count) {
    std::vector<Value> values;
    for (std::size_t index = 0; index < count; ++index)
      values.emplace_back();
    return vector_storage_bytes_(values);
  }

  [[nodiscard]] static std::uint64_t copied_string_storage_bytes(std::string_view value) {
    return external_string_bytes_(std::string(value));
  }

  [[nodiscard]] static std::uint64_t reserved_string_storage_bytes(std::size_t characters) {
    std::string value;
    value.reserve(characters);
    return external_string_bytes_(value);
  }

  [[nodiscard]] static std::uint64_t invocation_static_storage_bytes(
      std::size_t block_count, const std::vector<std::string>& operator_contracts,
      bool requires_mutation_rollback) {
    return external_string_bytes_(make_invocation_static_(block_count, operator_contracts,
                                                           requires_mutation_rollback));
  }

  /// Exact storage retained by this one bind-owned coupling image.  This deliberately includes
  /// every object reached from the workspace (rather than merely its candidate pointer pack),
  /// because accepted, forward and inverse AMR hierarchy images own independent instances.
  [[nodiscard]] std::uint64_t resident_storage_bytes() const {
    // The unique_ptr owner charges this object shell.  This method reports only the dynamic
    // storage retained beneath it so callers cannot accidentally count the shell twice.
    ResidentStorageFootprint footprint;
    footprint.prototype_state_pointers = vector_storage_bytes_(prototype_states_);
    footprint.canonical_state_pointers = vector_storage_bytes_(canonical_states_);
    footprint.rollback_images = field_vector_storage_bytes_(rollback_states_);
    footprint.conservation_images = field_vector_storage_bytes_(conservation_before_states_);
    footprint.host_patch_descriptors = vector_storage_bytes_(host_patches_);
    footprint.host_patch_offsets = vector_storage_bytes_(host_patch_offsets_);
    footprint.host_patch_counts = vector_storage_bytes_(host_patch_counts_);
    footprint.operator_contract_slots = vector_storage_bytes_(operator_contracts_);
    for (const std::string& contract : operator_contracts_)
      checked_add_(footprint.operator_contract_characters, external_string_bytes_(contract));
    footprint.invocation_characters = external_string_bytes_(invocation_static_);
    for (const HostPatch& patch : host_patches_) {
      checked_add_(footprint.host_before_values,
                   storage_values_bytes_(patch.before.extent(0)));
      checked_add_(footprint.host_candidate_values,
                   storage_values_bytes_(patch.candidate.extent(0)));
    }
    return resident_storage_bytes(footprint);
  }

  /// Cold, collective binding is performed by the owning facade after its coupling registry has
  /// frozen.  A second bind is accepted only when every prepared shape and provider contract is
  /// exactly the same; it never silently reallocates an accepted authority.
  void bind(const std::vector<const field_type*>& prototypes,
            const std::vector<std::string>& operator_contracts, bool requires_conservation,
            bool requires_mutation_rollback = false) {
    if (prototypes.empty())
      throw std::invalid_argument("prepared coupling workspace requires at least one block");
    validate_prototypes_(prototypes);
    if (bound_) {
      require_same_prototypes_(prototypes);
      if (operator_contracts_ != operator_contracts)
        throw std::logic_error("prepared coupling workspace provider contracts changed after bind");
      if (requires_conservation_ != requires_conservation ||
          requires_mutation_rollback_ != requires_mutation_rollback)
        throw std::logic_error(
            "prepared coupling workspace conservation contract changed after bind");
      return;
    }

    canonical_states_.assign(prototypes.size(), nullptr);
    prototype_states_.reserve(prototypes.size());
    host_patch_offsets_.resize(prototypes.size());
    host_patch_counts_.resize(prototypes.size());
    for (const field_type* prototype : prototypes)
      prototype_states_.push_back(prototype);

    // An empty registry never executes an operator.  Avoid allocating any field image or host
    // arena for it.  Non-empty registries always need rollback + candidate inspection storage;
    // the second image and before buffers exist only for declared conservation ledgers.
    if (!operator_contracts.empty() || requires_mutation_rollback) {
      rollback_states_.reserve(prototypes.size());
      for (const field_type* prototype : prototypes)
        rollback_states_.emplace_back(*prototype);
    }
    requires_conservation_ = requires_conservation;
    requires_mutation_rollback_ = requires_mutation_rollback;
    if (requires_conservation_) {
      conservation_before_states_.reserve(prototypes.size());
      for (const field_type* prototype : prototypes)
        conservation_before_states_.emplace_back(*prototype);
    }

    if (!operator_contracts.empty() || requires_mutation_rollback_)
      {
        std::size_t host_patch_count = 0;
        for (const field_type* prototype : prototypes) {
          if (prototype->local_size() > std::numeric_limits<std::size_t>::max() - host_patch_count)
            throw std::overflow_error("prepared coupling host patch count overflows size_t");
          host_patch_count += prototype->local_size();
        }
        host_patches_.reserve(host_patch_count);
      }
    if (!operator_contracts.empty() || requires_mutation_rollback_)
      for (std::size_t block = 0; block < prototypes.size(); ++block) {
        const field_type& field = *prototypes[block];
        host_patch_offsets_[block] = host_patches_.size();
        host_patch_counts_[block] = field.local_size();
        for (std::size_t local = 0; local < field.local_size(); ++local) {
          const fab_type& fab = field.fab(local);
          HostPatch patch{};
          patch.block = block;
          patch.local = local;
          patch.values = fab.size();
          if (requires_conservation_)
            patch.before = fab_type::create_raw_host_buffer(fab.size(), "pops_coupling_before");
          patch.candidate = fab_type::create_raw_host_buffer(fab.size(), "pops_coupling_candidate");
          host_patches_.push_back(std::move(patch));
        }
      }

    operator_contracts_ = operator_contracts;
    invocation_static_ =
        make_invocation_static_(prototypes.size(), operator_contracts_, requires_mutation_rollback_);
    initialize_dt_witness_();
    invocation_pairs_[0] = {"system-coupling-application-static-v1", invocation_static_};
    invocation_pairs_[1] = {"system-coupling-application-dt-v1",
                            std::string_view(dt_witness_.data(), dt_witness_.size())};
    bound_ = true;
  }

  /// Patches only resident pointer slots.  Any shape/alias drift is refused before an operator
  /// can mutate a detached candidate.
  void bind_candidates(std::span<field_type* const> candidates) {
    require_bound_();
    if (candidates.size() != prototype_states_.size())
      throw std::invalid_argument("prepared coupling candidate count changed after bind");
    for (std::size_t block = 0; block < candidates.size(); ++block) {
      field_type* candidate = candidates[block];
      if (candidate == nullptr)
        throw std::invalid_argument("prepared coupling candidate pack contains a null state");
      require_same_field_(*prototype_states_[block], *candidate);
      for (std::size_t previous = 0; previous < block; ++previous)
        if (candidates[previous] == candidate)
          throw std::invalid_argument("prepared coupling candidate pack aliases two blocks");
      canonical_states_[block] = candidate;
    }
  }

  void bind_candidates(const std::vector<field_type*>& candidates) {
    bind_candidates(std::span<field_type* const>(candidates.data(), candidates.size()));
  }

  void capture_rollback() {
    // An operator may have returned from an asynchronous execution space.  A rollback image must
    // never race that producer, including on host-accessible OpenMP configurations.
    synchronize_host_access_();
    copy_fields_(canonical_states_, rollback_states_);
  }
  void restore_rollback() {
    // An operator may signal a fault before the normal candidate-to-host inspection runs.  Fence
    // here as well, otherwise a HostSpace rollback could race that unfinished producer.
    synchronize_host_access_();
    copy_fields_from_images_(rollback_states_, canonical_states_);
  }
  void capture_conservation_before() {
    if (!requires_conservation_)
      throw std::logic_error("prepared coupling conservation image was not bound");
    synchronize_host_access_();
    copy_fields_(canonical_states_, conservation_before_states_);
    copy_images_to_host_(conservation_before_states_, true);
  }
  void copy_candidates_to_host() {
    // `Fab::copy_to_host()` fences before its HostSpace read.  Keep the same guarantee for the
    // preallocated raw-buffer route: host accessibility does not imply completion.
    synchronize_host_access_();
    copy_fields_to_host_(canonical_states_, false);
  }

  void patch_invocation_dt(Real dt) { patch_invocation_dt_(dt); }

  template <class Value>
  void patch_invocation_dt_(Value dt) {
    if constexpr (sizeof(Value) == sizeof(std::uint32_t)) {
      std::uint32_t bits = std::bit_cast<std::uint32_t>(dt);
      for (std::size_t byte = 0; byte < sizeof(bits); ++byte) {
        const std::size_t shift = (sizeof(bits) - 1 - byte) * 8;
        dt_witness_[kDtPayloadOffset + byte] = static_cast<char>((bits >> shift) & 0xffu);
      }
    } else {
      static_assert(sizeof(Value) == sizeof(std::uint64_t));
      std::uint64_t bits = std::bit_cast<std::uint64_t>(dt);
      for (std::size_t byte = 0; byte < sizeof(bits); ++byte) {
        const std::size_t shift = (sizeof(bits) - 1 - byte) * 8;
        dt_witness_[kDtPayloadOffset + byte] = static_cast<char>((bits >> shift) & 0xffu);
      }
    }
  }
  std::span<const ExactOrderedBytePair> invocation_pairs() const noexcept {
    return std::span<const ExactOrderedBytePair>(invocation_pairs_.data(),
                                                 invocation_pairs_.size());
  }

  const HostPatch& host_patch(std::size_t block, std::size_t local) const {
    if (block >= host_patch_offsets_.size() || local >= host_patch_counts_[block])
      throw std::logic_error("prepared coupling host patch is not bound");
    const HostPatch& patch = host_patches_[host_patch_offsets_[block] + local];
    if (patch.block != block || patch.local != local)
      throw std::logic_error("prepared coupling host patch descriptor is invalid");
    return patch;
  }

  const field_type& conservation_before(std::size_t block) const {
    if (block >= conservation_before_states_.size())
      throw std::out_of_range("prepared coupling conservation image block is out of range");
    return conservation_before_states_[block];
  }
  const field_type& prototype(std::size_t block) const {
    if (block >= prototype_states_.size())
      throw std::out_of_range("prepared coupling prototype block is out of range");
    return *prototype_states_[block];
  }

  void clear_numeric_failure() noexcept {
    numeric_failure_ = false;
    failing_operator_ = 0;
    failing_group_ = 0;
  }
  void mark_numeric_failure(std::size_t operator_index, std::size_t group_index = 0) noexcept {
    numeric_failure_ = true;
    failing_operator_ = operator_index;
    failing_group_ = group_index;
  }
  bool numeric_failure() const noexcept { return numeric_failure_; }
  std::size_t failing_operator() const noexcept { return failing_operator_; }
  std::size_t failing_group() const noexcept { return failing_group_; }

 private:
  static void checked_add_(std::uint64_t& total, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - total)
      throw std::overflow_error("prepared coupling workspace storage overflows uint64");
    total += value;
  }

  static std::uint64_t storage_values_bytes_(std::size_t values) {
    if (values != 0 && values > std::numeric_limits<std::uint64_t>::max() / sizeof(Real))
      throw std::overflow_error("prepared coupling workspace storage overflows uint64");
    return static_cast<std::uint64_t>(values) * sizeof(Real);
  }

  template <class Values>
  static std::uint64_t vector_storage_bytes_(const Values& values) {
    using value_type = typename std::remove_reference_t<Values>::value_type;
    if (values.capacity() > std::numeric_limits<std::uint64_t>::max() / sizeof(value_type))
      throw std::overflow_error("prepared coupling workspace storage overflows uint64");
    return static_cast<std::uint64_t>(values.capacity()) * sizeof(value_type);
  }

  static std::uint64_t field_vector_storage_bytes_(const std::vector<field_type>& fields) {
    std::uint64_t total = vector_storage_bytes_(fields);
    for (const field_type& field : fields)
      checked_add_(total, field.resident_storage_bytes());
    return total;
  }

  static std::uint64_t external_string_bytes_(const std::string& value) {
    const auto begin = reinterpret_cast<std::uintptr_t>(std::addressof(value));
    const auto end = begin + sizeof(value);
    const auto data = reinterpret_cast<std::uintptr_t>(value.data());
    if (data >= begin && data < end)
      return 0;
    if (value.capacity() == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("prepared coupling workspace string storage overflows uint64");
    return static_cast<std::uint64_t>(value.capacity()) + 1U;
  }

  static std::string make_invocation_static_(std::size_t block_count,
                                              const std::vector<std::string>& operator_contracts,
                                              bool requires_mutation_rollback) {
    ExactContractBuilder static_contract;
    static_contract.text(invocation_schema())
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(static_cast<std::uint64_t>(block_count))
        .scalar(static_cast<std::uint8_t>(requires_mutation_rollback ? 1 : 0))
        .sequence(operator_contracts, [](ExactContractBuilder& item, const std::string& provider) {
          item.bytes(provider);
        });
    return std::move(static_contract).release();
  }

  static void require_same_field_(const field_type& expected, const field_type& actual) {
    if (expected.layout() != actual.layout() || expected.distribution() != actual.distribution() ||
        expected.local_rank() != actual.local_rank() || expected.ncomp() != actual.ncomp() ||
        expected.ghosts() != actual.ghosts() || expected.local_size() != actual.local_size())
      throw std::invalid_argument("prepared coupling candidate layout changed after bind");
    for (std::size_t local = 0; local < expected.local_size(); ++local)
      if (expected.global_index(local) != actual.global_index(local) ||
          expected.fab(local).size() != actual.fab(local).size())
        throw std::invalid_argument(
            "prepared coupling candidate patch ownership changed after bind");
  }

  static void validate_prototypes_(const std::vector<const field_type*>& prototypes) {
    for (const field_type* prototype : prototypes) {
      if (prototype == nullptr || prototype->ncomp() < 1)
        throw std::invalid_argument("prepared coupling workspace has an invalid prototype");
      for (std::size_t local = 0; local < prototype->local_size(); ++local)
        if (prototype->fab(local).size() == 0)
          throw std::invalid_argument("prepared coupling workspace has an empty local patch");
    }
  }

  void require_same_prototypes_(const std::vector<const field_type*>& prototypes) const {
    if (prototypes.size() != prototype_states_.size())
      throw std::logic_error("prepared coupling workspace block count changed after bind");
    for (std::size_t block = 0; block < prototypes.size(); ++block) {
      if (prototypes[block] == nullptr)
        throw std::invalid_argument("prepared coupling workspace has an invalid prototype");
      require_same_field_(*prototype_states_[block], *prototypes[block]);
    }
  }

  static void copy_field_(const field_type& source, field_type& destination) {
    require_same_field_(source, destination);
    for (std::size_t local = 0; local < source.local_size(); ++local) {
      if constexpr (std::is_same_v<typename field_type::memory_space, Kokkos::HostSpace>)
        std::copy_n(source.fab(local).storage().data(), source.fab(local).size(),
                    destination.fab(local).storage().data());
      else
        Kokkos::deep_copy(destination.fab(local).storage(), source.fab(local).storage());
    }
  }

  static void copy_fields_(const std::vector<field_type*>& sources,
                           std::vector<field_type>& destinations) {
    if (sources.size() != destinations.size())
      throw std::logic_error("prepared coupling snapshot image count changed after bind");
    for (std::size_t block = 0; block < sources.size(); ++block) {
      if (sources[block] == nullptr)
        throw std::logic_error("prepared coupling candidate slot is unbound");
      copy_field_(*sources[block], destinations[block]);
    }
    // HostSpace copies are ordinary host stores after a synchronous HostSpace kernel return
    // (Serial/OpenMP); device-resident copies require the one explicit completion fence.
    if constexpr (!std::is_same_v<typename field_type::memory_space, Kokkos::HostSpace>)
      ::pops::device_fence();
  }

  static void copy_fields_from_images_(const std::vector<field_type>& sources,
                                       const std::vector<field_type*>& destinations) {
    if (sources.size() != destinations.size())
      throw std::logic_error("prepared coupling restore image count changed after bind");
    for (std::size_t block = 0; block < sources.size(); ++block) {
      if (destinations[block] == nullptr)
        throw std::logic_error("prepared coupling candidate slot is unbound");
      copy_field_(sources[block], *destinations[block]);
    }
    if constexpr (!std::is_same_v<typename field_type::memory_space, Kokkos::HostSpace>)
      ::pops::device_fence();
  }

  static void synchronize_host_access_() {
    // This is deliberately unconditional.  Kokkos permits asynchronous execution even when the
    // allocation is host-accessible; the resident workspace has already been warm-primed, so this
    // completion operation must not allocate.
    ::pops::device_fence();
  }

  template <class Fields>
  void copy_fields_to_host_(const Fields& fields, bool before) {
    for (const HostPatch& patch : host_patches_) {
      const auto& field = fields[patch.block];
      if constexpr (std::is_pointer_v<std::remove_reference_t<decltype(field)>>) {
        if (field == nullptr)
          throw std::logic_error("prepared coupling candidate slot is unbound");
        field->fab(patch.local)
            .copy_to_host_buffer(before ? patch.before : patch.candidate,
                                 ::pops::detail::default_execution_space());
      } else {
        field.fab(patch.local)
            .copy_to_host_buffer(before ? patch.before : patch.candidate,
                                 ::pops::detail::default_execution_space());
      }
    }
    if constexpr (!std::is_same_v<typename field_type::memory_space, Kokkos::HostSpace>)
      ::pops::device_fence();
  }

  void copy_images_to_host_(const std::vector<field_type>& fields, bool before) {
    copy_fields_to_host_(fields, before);
  }

  void require_bound_() const {
    if (!bound_)
      throw std::logic_error("prepared coupling workspace was not bound during installation");
  }

  static constexpr std::size_t kDtPayloadOffset = 10;
  static constexpr std::size_t kDtWitnessBytes = kDtPayloadOffset + sizeof(Real);

  void initialize_dt_witness_() noexcept {
    // ExactContractBuilder::scalar(Real): frame 's' + uint64 payload bytes + scalar tag + bits.
    dt_witness_[0] = 's';
    const std::size_t payload_bytes = 1 + sizeof(Real);
    for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte) {
      const std::size_t shift = (sizeof(std::uint64_t) - 1 - byte) * 8;
      dt_witness_[1 + byte] = static_cast<char>((payload_bytes >> shift) & 0xffu);
    }
    dt_witness_[9] = sizeof(Real) == sizeof(float) ? 'f' : 'd';
    patch_invocation_dt(Real(0));
  }

  bool bound_ = false;
  std::vector<const field_type*> prototype_states_;
  std::vector<field_type*> canonical_states_;
  std::vector<field_type> rollback_states_;
  std::vector<field_type> conservation_before_states_;
  std::vector<HostPatch> host_patches_;
  /// Fixed `(canonical block, local fab) -> host patch` descriptors; hot checks never search or
  /// build an associative lookup from a candidate pointer.
  std::vector<std::size_t> host_patch_offsets_;
  std::vector<std::size_t> host_patch_counts_;
  std::vector<std::string> operator_contracts_;
  std::string invocation_static_;
  std::array<char, kDtWitnessBytes> dt_witness_{};
  std::array<ExactOrderedBytePair, 2> invocation_pairs_{};
  bool numeric_failure_ = false;
  std::size_t failing_operator_ = 0;
  std::size_t failing_group_ = 0;
  bool requires_conservation_ = false;
  bool requires_mutation_rollback_ = false;
};

}  // namespace pops::runtime::system
