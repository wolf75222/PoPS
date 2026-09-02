#pragma once

/// @file
/// @brief Dense, bind-sealed scheduler cache for Program resource slots.
///
/// The lowering phase assigns every scheduled Program value a finite slot in its
/// ``ProgramResourcePlan``. The runtime keeps only that dense slot index on the
/// hot path. Complete value/path/owner/space/clock/level identities are retained
/// in the bind-time plan and in host-owned checkpoint images, never searched while
/// executing a schedule.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/program/program_persistent_value_store.hpp>

namespace pops::runtime::program {

namespace detail {

template <int Dim>
bool same_cache_value_layout(const MultiFab<Dim>& left, const MultiFab<Dim>& right) noexcept {
  return left.layout() == right.layout() && left.distribution() == right.distribution() &&
         left.local_rank() == right.local_rank() && left.ncomp() == right.ncomp() &&
         left.ghosts() == right.ghosts();
}

/// Copy a complete resident value. The one possible allocation is confined to
/// preparation or the first explicit prime_slot call; repeated stores reuse the
/// exact resident layout.
template <int Dim>
void copy_cache_value_into(MultiFab<Dim>& destination, const MultiFab<Dim>& source) {
  if (!same_cache_value_layout(destination, source))
    destination = MultiFab<Dim>(source.layout(), source.distribution(), source.local_rank(),
                                source.ncomp(), source.ghosts());
  for (std::size_t local = 0; local < destination.local_size(); ++local) {
    const std::size_t global = destination.global_index(local);
    const std::size_t source_local = source.local_index_of(global);
    if (source_local == MultiFab<Dim>::not_local)
      throw std::logic_error("cache value copy found inconsistent local ownership");
    Kokkos::deep_copy(destination.fab(local).storage(), source.fab(source_local).storage());
  }
}

}  // namespace detail

using ProgramCacheSlot = std::size_t;

/// One scheduled value and its accepted temporal bookkeeping.
template <int Dim>
struct CacheSlot final {
  MultiFab<Dim> value;
  int last_update_step = -1;
  Real accumulated_dt = Real(0);
  bool valid = false;
  bool cold = true;
  bool resident = false;
  std::string name;
};

/// Host-owned checkpoint image for one dense scheduler slot. Invalid/cold slots
/// carry no fabricated value, but their accumulated window and complete static
/// key remain durable across restart.
template <int Dim>
struct CacheSlotSnapshot final {
  ProgramCacheSlot slot = 0;
  std::string plan_schema;
  std::string plan_digest;
  ProgramPersistentValueKey key{};
  std::string identity;
  std::string occurrence_path;
  std::string owner_identity;
  std::string space_identity;
  std::string clock_identity;
  std::int32_t amr_level = -1;
  int last_update_step = -1;
  Real accumulated_dt = Real(0);
  bool valid = false;
  bool cold = true;
  std::string name;
  std::optional<MultiFab<Dim>> value;
};

/// Dense cache bound to the immutable Program resource plan.
template <int Dim>
class CacheManager final {
 public:
  CacheManager() = default;
  CacheManager(const CacheManager&) = default;
  CacheManager& operator=(const CacheManager&) = default;
  CacheManager(CacheManager&&) noexcept = default;
  CacheManager& operator=(CacheManager&&) noexcept = default;

  /// Bind the finite slot table before Program execution. All slot identities
  /// and diagnostic strings are copied here; later operations use only a dense
  /// index and a vector lookup.
  void bind(const ProgramResourcePlan& plan) {
    if (bound_)
      throw std::logic_error("Program scheduler cache is already bind-sealed");
    CacheManager prepared;
    prepared.bound_ = true;
    prepared.plan_schema_ = std::string(plan.schema());
    prepared.plan_digest_ = std::string(plan.digest());
    prepared.plan_entries_ = plan.entries();
    prepared.slots_.resize(prepared.plan_entries_.size());
    for (std::size_t slot = 0; slot != prepared.plan_entries_.size(); ++slot) {
      if (prepared.plan_entries_[slot].slot != slot)
        throw std::logic_error("Program scheduler cache plan slots are not dense");
      prepared.slots_[slot].name = prepared.plan_entries_[slot].identity;
      // Explicit schedule labels are optional metadata. Reserve a bounded
      // diagnostic envelope (at least the complete plan identity) while the
      // plan is prepared so naming a slot never grows a string on the hot path.
      prepared.slots_[slot].name.reserve(
          std::max<std::size_t>(128, prepared.plan_entries_[slot].identity.size()));
    }
    swap(prepared);
  }

  [[nodiscard]] bool bound() const noexcept { return bound_; }

  /// Compare only the immutable bind authority.  This deliberately excludes accepted values so a
  /// prepared artifact publication may retain its newly bound dense image unless a staged image
  /// was prepared from precisely the same resource plan.
  [[nodiscard]] bool has_same_bound_plan(const CacheManager& other) const noexcept {
    return bound_ && other.bound_ && plan_schema_ == other.plan_schema_ &&
           plan_digest_ == other.plan_digest_ && plan_entries_ == other.plan_entries_ &&
           slots_.size() == other.slots_.size();
  }
  [[nodiscard]] std::size_t size() const noexcept { return slots_.size(); }
  [[nodiscard]] std::string_view plan_schema() const noexcept { return plan_schema_; }
  [[nodiscard]] std::string_view plan_digest() const noexcept { return plan_digest_; }
  [[nodiscard]] const ProgramResourcePlanEntry& plan_entry(ProgramCacheSlot slot) const {
    (void)checked_slot_(slot);
    return plan_entries_[slot];
  }

  bool is_due(ProgramCacheSlot slot, int macro_step, int every_n) const {
    const CacheSlot<Dim>& state = checked_slot_(slot);
    if (!state.valid || every_n <= 1)
      return true;
    return (macro_step % every_n) == 0;
  }

  /// Prime a slot's MultiFab storage during preparation. This is the explicit
  /// boundary for field allocation; stores after priming only deep-copy.
  void prime_slot(ProgramCacheSlot slot, const MultiFab<Dim>& prototype) {
    CacheSlot<Dim>& state = checked_slot_(slot);
    detail::copy_cache_value_into(state.value, prototype);
    state.resident = true;
  }

  void store(ProgramCacheSlot slot, const MultiFab<Dim>& value, int macro_step) {
    CacheSlot<Dim>& state = checked_slot_(slot);
    if (!state.resident)
      throw std::logic_error("Program scheduler cache slot was not primed during preparation");
    if (!detail::same_cache_value_layout(state.value, value))
      throw std::logic_error("Program scheduler cache store changed the prepared value layout");
    detail::copy_cache_value_into(state.value, value);
    state.resident = true;
    state.last_update_step = macro_step;
    state.accumulated_dt = Real(0);
    state.valid = true;
    state.cold = false;
  }

  void store(ProgramCacheSlot slot, const MultiFab<Dim>& value, int macro_step,
             std::string_view name) {
    store(slot, value, macro_step);
    CacheSlot<Dim>& state = checked_slot_(slot);
    assign_name_(state, name);
  }

  const MultiFab<Dim>& retrieve(ProgramCacheSlot slot) const { return checked_slot_(slot).value; }

  void restore_into(ProgramCacheSlot slot, MultiFab<Dim>& destination) const {
    const CacheSlot<Dim>& state = checked_slot_(slot);
    if (!state.valid)
      throw std::logic_error("cannot restore an invalid Program scheduler cache slot");
    if (!detail::same_cache_value_layout(destination, state.value))
      throw std::logic_error(
          "Program scheduler cache restore changed the prepared destination layout");
    detail::copy_cache_value_into(destination, state.value);
  }

  void accumulate_dt(ProgramCacheSlot slot, Real dt) {
    if (!std::isfinite(static_cast<double>(dt)) || dt < Real(0))
      throw std::invalid_argument("Program cache accumulated dt must be finite and non-negative");
    CacheSlot<Dim>& state = checked_slot_(slot);
    const Real next = state.accumulated_dt + dt;
    if (!std::isfinite(static_cast<double>(next)))
      throw std::overflow_error("Program cache accumulated dt overflows its scalar range");
    state.accumulated_dt = next;
  }

  [[nodiscard]] Real accumulated_dt(ProgramCacheSlot slot) const {
    return checked_slot_(slot).accumulated_dt;
  }

  Real effective_dt(ProgramCacheSlot slot, Real dt_now) {
    if (!std::isfinite(static_cast<double>(dt_now)) || dt_now < Real(0))
      throw std::invalid_argument("Program cache current dt must be finite and non-negative");
    CacheSlot<Dim>& state = checked_slot_(slot);
    const Real effective = dt_now + state.accumulated_dt;
    if (!std::isfinite(static_cast<double>(effective)))
      throw std::overflow_error("Program cache effective dt overflows its scalar range");
    state.accumulated_dt = Real(0);
    return effective;
  }

  [[nodiscard]] bool has(ProgramCacheSlot slot) const { return checked_slot_(slot).valid; }
  [[nodiscard]] bool valid(ProgramCacheSlot slot) const { return has(slot); }
  [[nodiscard]] bool cold(ProgramCacheSlot slot) const { return checked_slot_(slot).cold; }
  [[nodiscard]] int last_update_step(ProgramCacheSlot slot) const {
    return checked_slot_(slot).last_update_step;
  }

  /// Valid slot indices. This host observer is not part of generated execution.
  [[nodiscard]] std::vector<ProgramCacheSlot> slot_indices() const {
    std::vector<ProgramCacheSlot> result;
    result.reserve(slots_.size());
    for (ProgramCacheSlot slot = 0; slot != slots_.size(); ++slot)
      if (slots_[slot].valid)
        result.push_back(slot);
    return result;
  }

  /// Every bind-declared slot, including invalid/cold slots with pending dt.
  [[nodiscard]] std::vector<ProgramCacheSlot> checkpoint_slot_indices() const {
    std::vector<ProgramCacheSlot> result(slots_.size());
    for (ProgramCacheSlot slot = 0; slot != slots_.size(); ++slot)
      result[slot] = slot;
    return result;
  }

  [[nodiscard]] std::string name_of(ProgramCacheSlot slot) const {
    const CacheSlot<Dim>& state = checked_slot_(slot);
    return state.name.empty() ? plan_entries_[slot].identity : state.name;
  }

  [[nodiscard]] int ncomp_of(ProgramCacheSlot slot) const {
    const CacheSlot<Dim>& state = checked_slot_(slot);
    if (!state.valid)
      throw std::out_of_range("invalid Program scheduler cache slot has no value layout");
    return state.value.ncomp();
  }

  [[nodiscard]] Extent<Dim> ghosts_of(ProgramCacheSlot slot) const {
    const CacheSlot<Dim>& state = checked_slot_(slot);
    if (!state.valid)
      throw std::out_of_range("invalid Program scheduler cache slot has no value layout");
    return state.value.ghosts();
  }

  [[nodiscard]] const MultiFab<Dim>& value_of(ProgramCacheSlot slot) const {
    const CacheSlot<Dim>& state = checked_slot_(slot);
    if (!state.valid)
      throw std::out_of_range("invalid Program scheduler cache slot has no value");
    return state.value;
  }

  [[nodiscard]] std::vector<CacheSlotSnapshot<Dim>> checkpoint_slots() const {
    std::vector<CacheSlotSnapshot<Dim>> result;
    result.reserve(slots_.size());
    for (ProgramCacheSlot slot = 0; slot != slots_.size(); ++slot) {
      const auto& state = slots_[slot];
      const auto& row = plan_entries_[slot];
      CacheSlotSnapshot<Dim> image;
      image.slot = slot;
      image.plan_schema = plan_schema_;
      image.plan_digest = plan_digest_;
      image.key = row.key;
      image.identity = row.identity;
      image.occurrence_path = row.occurrence_path;
      image.owner_identity = row.owner_identity;
      image.space_identity = row.space_identity;
      image.clock_identity = row.clock_identity;
      image.amr_level = row.key.amr_level;
      image.last_update_step = state.last_update_step;
      image.accumulated_dt = state.accumulated_dt;
      image.valid = state.valid;
      image.cold = state.cold;
      image.name = state.name;
      if (state.valid)
        image.value = state.value;
      result.push_back(std::move(image));
    }
    return result;
  }

  /// Restore one already-prepared valid value. This does not create a slot.
  void restore_slot(ProgramCacheSlot slot, MultiFab<Dim> value, int last_update_step,
                    Real accumulated_dt, std::string_view name) {
    if (!std::isfinite(static_cast<double>(accumulated_dt)) || accumulated_dt < Real(0) ||
        last_update_step < 0)
      throw std::invalid_argument("invalid Program scheduler cache slot metadata");
    CacheSlot<Dim>& state = checked_slot_(slot);
    state.value = std::move(value);
    state.resident = true;
    state.last_update_step = last_update_step;
    state.accumulated_dt = accumulated_dt;
    state.valid = true;
    state.cold = false;
    assign_name_(state, name);
  }

  /// Restore a pending cold slot without constructing a dummy field value.
  void restore_pending_slot(ProgramCacheSlot slot, Real accumulated_dt, std::string_view name) {
    if (!std::isfinite(static_cast<double>(accumulated_dt)) || accumulated_dt < Real(0))
      throw std::invalid_argument("invalid pending Program scheduler cache slot");
    CacheSlot<Dim>& state = checked_slot_(slot);
    state.last_update_step = -1;
    state.accumulated_dt = accumulated_dt;
    state.valid = false;
    state.cold = true;
    assign_name_(state, name);
  }

  /// Refresh mutable accepted state from an identically bound cache. The table
  /// and identities remain untouched and cannot allocate.
  void copy_from_preallocated(const CacheManager& source) {
    require_same_bind_(source);
    for (ProgramCacheSlot slot = 0; slot != slots_.size(); ++slot) {
      const CacheSlot<Dim>& input = source.slots_[slot];
      CacheSlot<Dim>& output = slots_[slot];
      if (input.valid) {
        if (!output.resident || !detail::same_cache_value_layout(output.value, input.value))
          throw std::logic_error("Program scheduler cache value storage was not primed");
        detail::copy_cache_value_into(output.value, input.value);
      }
      output.last_update_step = input.last_update_step;
      output.accumulated_dt = input.accumulated_dt;
      output.valid = input.valid;
      output.cold = input.cold;
      output.resident = output.resident || input.resident;
      if (input.name.size() > output.name.capacity())
        throw std::logic_error("Program scheduler cache label capacity was not primed");
      output.name.assign(input.name);
    }
  }

  class PreparedCheckpointRestore final {
   public:
    PreparedCheckpointRestore() = default;
    PreparedCheckpointRestore(const PreparedCheckpointRestore&) = delete;
    PreparedCheckpointRestore& operator=(const PreparedCheckpointRestore&) = delete;
    PreparedCheckpointRestore(PreparedCheckpointRestore&&) noexcept = default;
    PreparedCheckpointRestore& operator=(PreparedCheckpointRestore&&) noexcept = default;

   private:
    friend class CacheManager;
    bool bound = false;
    std::string plan_schema, plan_digest;
    std::vector<ProgramResourcePlanEntry> plan_entries;
    std::vector<CacheSlot<Dim>> slots;
  };

  [[nodiscard]] PreparedCheckpointRestore prepare_checkpoint_restore(
      const std::vector<CacheSlotSnapshot<Dim>>& images) const {
    if (!bound_ || images.size() != slots_.size())
      throw std::invalid_argument("Program scheduler cache checkpoint has the wrong bound plan");
    PreparedCheckpointRestore prepared;
    prepared.bound = true;
    prepared.plan_schema = plan_schema_;
    prepared.plan_digest = plan_digest_;
    prepared.plan_entries = plan_entries_;
    prepared.slots.resize(slots_.size());
    for (ProgramCacheSlot slot = 0; slot != images.size(); ++slot) {
      const auto& image = images[slot];
      const auto& row = plan_entries_[slot];
      if (image.slot != slot || image.plan_schema != plan_schema_ ||
          image.plan_digest != plan_digest_ || image.key != row.key ||
          image.identity != row.identity || image.occurrence_path != row.occurrence_path ||
          image.owner_identity != row.owner_identity ||
          image.space_identity != row.space_identity ||
          image.clock_identity != row.clock_identity || image.amr_level != row.key.amr_level ||
          !std::isfinite(static_cast<double>(image.accumulated_dt)) ||
          image.accumulated_dt < Real(0))
        throw std::invalid_argument("Program scheduler cache checkpoint identity mismatch");
      if (image.valid != image.value.has_value() || (image.valid && image.cold) ||
          (!image.valid && !image.cold) || (image.valid && image.last_update_step < 0) ||
          (!image.valid && image.last_update_step != -1))
        throw std::invalid_argument("Program scheduler cache checkpoint state mismatch");
      CacheSlot<Dim>& state = prepared.slots[slot];
      state.last_update_step = image.last_update_step;
      state.accumulated_dt = image.accumulated_dt;
      state.valid = image.valid;
      state.cold = image.cold;
      state.name.reserve(std::max<std::size_t>(128, row.identity.size()));
      state.name = image.name;
      if (image.value) {
        state.value = *image.value;
        state.resident = true;
      }
    }
    return prepared;
  }

  void publish_checkpoint_restore(PreparedCheckpointRestore&& prepared) noexcept {
    using std::swap;
    swap(bound_, prepared.bound);
    plan_schema_.swap(prepared.plan_schema);
    plan_digest_.swap(prepared.plan_digest);
    plan_entries_.swap(prepared.plan_entries);
    slots_.swap(prepared.slots);
  }

  void restore_checkpoint_slots(const std::vector<CacheSlotSnapshot<Dim>>& images) {
    auto prepared = prepare_checkpoint_restore(images);
    publish_checkpoint_restore(std::move(prepared));
  }

  /// Lifecycle reset. A replacement must bind a fresh plan before execution.
  void clear() noexcept { CacheManager{}.swap(*this); }

  void swap(CacheManager& other) noexcept {
    using std::swap;
    swap(bound_, other.bound_);
    plan_schema_.swap(other.plan_schema_);
    plan_digest_.swap(other.plan_digest_);
    plan_entries_.swap(other.plan_entries_);
    slots_.swap(other.slots_);
  }

 private:
  [[nodiscard]] const CacheSlot<Dim>& checked_slot_(ProgramCacheSlot slot) const {
    if (!bound_)
      throw std::logic_error("Program scheduler cache is not bind-sealed");
    if (slot >= slots_.size())
      throw std::out_of_range("Program scheduler cache slot is outside the prepared plan");
    return slots_[slot];
  }

  [[nodiscard]] CacheSlot<Dim>& checked_slot_(ProgramCacheSlot slot) {
    return const_cast<CacheSlot<Dim>&>(std::as_const(*this).checked_slot_(slot));
  }

  void assign_name_(CacheSlot<Dim>& state, std::string_view name) {
    if (name.empty())
      return;
    if (name.size() > state.name.capacity())
      throw std::logic_error("Program scheduler cache label was not primed");
    state.name.assign(name.data(), name.size());
  }

  void require_same_bind_(const CacheManager& source) const {
    if (!bound_ || !source.bound_ || plan_schema_ != source.plan_schema_ ||
        plan_digest_ != source.plan_digest_ || plan_entries_ != source.plan_entries_ ||
        slots_.size() != source.slots_.size())
      throw std::logic_error("Program scheduler cache bind image changed after preparation");
  }

  bool bound_ = false;
  std::string plan_schema_, plan_digest_;
  std::vector<ProgramResourcePlanEntry> plan_entries_;
  std::vector<CacheSlot<Dim>> slots_;
};

template <int Dim>
inline void swap(CacheManager<Dim>& left, CacheManager<Dim>& right) noexcept {
  left.swap(right);
}

}  // namespace pops::runtime::program
