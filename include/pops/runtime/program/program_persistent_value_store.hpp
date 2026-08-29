#pragma once

/// @file
/// @brief Bind-sealed, slot-addressed persistent Program resource inventory.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace pops::runtime::program {

enum class ProgramValueLifetime : std::uint8_t { transient = 1, persistent_schedule = 2 };
enum class ProgramValueCentering : std::uint8_t { cell = 1, face = 2, node = 3 };
enum class ProgramScheduleOffPolicy : std::uint8_t {
  none = 0,
  hold = 1,
  accumulate_dt = 2,
  zero = 3,
  error = 4
};
enum class ProgramSpatialTransferPolicy : std::uint8_t {
  redistribute_exact = 1,
  qualified_regrid_provider = 2,
  refuse = 3,
};

/// Compact lookup identity retained solely for diagnostics and cold lookup. Generated execution
/// carries a dense slot and never calls slot_for.
struct ProgramPersistentValueKey final {
  std::uint64_t value_id = 0;
  std::uint64_t occurrence_path_id = 0;
  std::uint32_t owner = 0;
  std::uint32_t space = 0;
  std::uint32_t clock = 0;
  std::int32_t amr_level = -1;

  [[nodiscard]] bool operator==(const ProgramPersistentValueKey&) const noexcept = default;
  [[nodiscard]] bool operator<(const ProgramPersistentValueKey& other) const noexcept {
    return std::tie(value_id, occurrence_path_id, owner, space, clock, amr_level) <
           std::tie(other.value_id, other.occurrence_path_id, other.owner, other.space, other.clock,
                    other.amr_level);
  }
};

/// Host-owned lossless row. Human-readable fields are retained for checkpoint/diagnostic capture;
/// generated execution addresses the bound value by its preassigned vector slot only.
struct ProgramResourcePlanEntry final {
  std::uint32_t slot = 0;
  ProgramPersistentValueKey key{};
  std::string identity, occurrence_path, owner_identity, space_identity, clock_identity;
  ProgramValueLifetime lifetime = ProgramValueLifetime::transient;
  ProgramValueCentering centering = ProgramValueCentering::cell;
  ProgramScheduleOffPolicy off_policy = ProgramScheduleOffPolicy::none;
  ProgramSpatialTransferPolicy spatial_transfer = ProgramSpatialTransferPolicy::refuse;
  std::uint32_t components = 0;
  std::uint32_t ghosts = 0;
  std::uint64_t bytes = 0;
  std::uint64_t maximum_bytes = 0;
  bool communicates = false;
  bool restart_required = false;
  std::string communication, transfer_identity, restart_identity, component_names, shape;
  std::optional<std::uint64_t> cells, itemsize;

  friend bool operator==(const ProgramResourcePlanEntry&,
                         const ProgramResourcePlanEntry&) = default;
};

class ProgramResourcePlan final {
 public:
  ProgramResourcePlan() = default;
  explicit ProgramResourcePlan(std::vector<ProgramResourcePlanEntry> entries,
                               std::uint64_t maximum_bytes,
                               std::string schema = "program-resource-plan:v1",
                               std::string digest = {})
      : entries_(std::move(entries)),
        maximum_bytes_(maximum_bytes),
        schema_(std::move(schema)),
        digest_(std::move(digest)) {
    validate_();
  }

  [[nodiscard]] const std::vector<ProgramResourcePlanEntry>& entries() const noexcept {
    return entries_;
  }
  [[nodiscard]] std::size_t slot_count() const noexcept { return entries_.size(); }
  [[nodiscard]] const ProgramResourcePlanEntry& entry(std::size_t slot) const {
    if (slot >= entries_.size())
      throw std::out_of_range("Program resource plan slot is outside the sealed plan");
    return entries_[slot];
  }
  [[nodiscard]] std::uint64_t maximum_bytes() const noexcept { return maximum_bytes_; }
  [[nodiscard]] std::string_view schema() const noexcept { return schema_; }
  [[nodiscard]] std::string_view digest() const noexcept { return digest_; }

 private:
  void validate_() {
    if (schema_ != "program-resource-plan:v1")
      throw std::invalid_argument("Program resource plan has an unsupported schema");
    if (entries_.empty()) {
      if (maximum_bytes_ != 0 || digest_.size() != 64 ||
          !std::all_of(digest_.begin(), digest_.end(), [](unsigned char value) {
            return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
          }))
        throw std::invalid_argument(
            "empty Program resource plan has an invalid memory ceiling or digest");
      return;
    }
    if (digest_.size() != 64 ||
        !std::all_of(digest_.begin(), digest_.end(), [](unsigned char value) {
          return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
        }))
      throw std::invalid_argument("Program resource plan has no SHA-256 digest");
    std::vector<const ProgramResourcePlanEntry*> ordered;
    ordered.reserve(entries_.size());
    std::uint64_t total = 0;
    for (const auto& entry : entries_) {
      const auto lifetime = static_cast<std::uint8_t>(entry.lifetime);
      const auto centering = static_cast<std::uint8_t>(entry.centering);
      const auto off_policy = static_cast<std::uint8_t>(entry.off_policy);
      const auto spatial_transfer = static_cast<std::uint8_t>(entry.spatial_transfer);
      if (lifetime < static_cast<std::uint8_t>(ProgramValueLifetime::transient) ||
          lifetime > static_cast<std::uint8_t>(ProgramValueLifetime::persistent_schedule) ||
          centering < static_cast<std::uint8_t>(ProgramValueCentering::cell) ||
          centering > static_cast<std::uint8_t>(ProgramValueCentering::node) ||
          off_policy > static_cast<std::uint8_t>(ProgramScheduleOffPolicy::error) ||
          spatial_transfer <
              static_cast<std::uint8_t>(ProgramSpatialTransferPolicy::redistribute_exact) ||
          spatial_transfer > static_cast<std::uint8_t>(ProgramSpatialTransferPolicy::refuse))
        throw std::invalid_argument("Program resource plan has an unknown enum value");
      if (entry.slot >= entries_.size() || entry.components == 0 || entry.bytes == 0 ||
          entry.key.amr_level < -1 || entry.maximum_bytes < entry.bytes || entry.identity.empty() ||
          entry.occurrence_path.empty() || entry.owner_identity.empty() ||
          entry.space_identity.empty() || entry.clock_identity.empty() ||
          entry.communication.empty() || entry.component_names.empty() || entry.shape.empty())
        throw std::invalid_argument("Program resource plan has an incomplete lossless entry");
      if (entry.lifetime == ProgramValueLifetime::transient &&
          entry.off_policy != ProgramScheduleOffPolicy::none)
        throw std::invalid_argument(
            "transient Program scratch cannot carry an off-schedule policy");
      if (entry.spatial_transfer == ProgramSpatialTransferPolicy::qualified_regrid_provider &&
          entry.transfer_identity.empty())
        throw std::invalid_argument("qualified regrid value requires a transfer provider");
      if (entry.restart_required && entry.restart_identity.empty())
        throw std::invalid_argument("restart-required value requires a restart provider");
      if ((entry.cells && *entry.cells == 0) || (entry.itemsize && *entry.itemsize == 0))
        throw std::invalid_argument("Program resource plan has a zero optional extent");
      if (entry.maximum_bytes > std::numeric_limits<std::uint64_t>::max() - total)
        throw std::overflow_error("Program resource plan byte bound overflows uint64");
      total += entry.maximum_bytes;
      ordered.push_back(&entry);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto* left, const auto* right) { return left->slot < right->slot; });
    for (std::size_t slot = 0; slot != ordered.size(); ++slot) {
      if (ordered[slot]->slot != slot)
        throw std::invalid_argument("Program resource plan slots are not dense");
      for (std::size_t prior = 0; prior != slot; ++prior) {
        const auto* const value = ordered[slot];
        const auto* const other = ordered[prior];
        // The strings are the authoritative complete identity.  Compact owner/space/clock/path
        // ids are acceleration fields only: a forged row with a different compact id must still be
        // rejected when its complete value/path/owner/space/clock/level key is duplicated.
        if (value->key.value_id == other->key.value_id &&
            value->key.amr_level == other->key.amr_level &&
            value->occurrence_path == other->occurrence_path &&
            value->owner_identity == other->owner_identity &&
            value->space_identity == other->space_identity &&
            value->clock_identity == other->clock_identity)
          throw std::invalid_argument("Program resource plan has a duplicate complete key");
        if (value->identity == other->identity)
          throw std::invalid_argument("Program resource plan has a duplicate resource identity");
        if (value->key.occurrence_path_id == other->key.occurrence_path_id &&
            value->occurrence_path != other->occurrence_path)
          throw std::invalid_argument("Program resource plan occurrence digest collision");
      }
    }
    if (total > maximum_bytes_)
      throw std::invalid_argument("Program resource plan exceeds its bind-sealed byte budget");
  }

  std::vector<ProgramResourcePlanEntry> entries_;
  std::uint64_t maximum_bytes_ = 0;
  std::string schema_ = "program-resource-plan:v1";
  std::string digest_;
};

/// Metadata remains present even for an invalid/cold slot: accumulated temporal windows are part
/// of accepted state and therefore survive reject, restart seam, and cold recomputation.
struct ProgramPersistentValueMetadata final {
  std::uint64_t accepted_coordinate = 0;
  std::uint64_t cursor = 0;
  double accumulated_dt = 0.0;
  std::uint64_t topology_epoch = 0;
  std::uint64_t layout_generation = 0;
  bool valid = false;
  bool cold = true;

  friend bool operator==(const ProgramPersistentValueMetadata&,
                         const ProgramPersistentValueMetadata&) = default;
};

class ProgramPersistentValueStore final {
 public:
  struct Snapshot final {
    bool bound = false;
    std::uint64_t maximum_bytes = 0;
    std::string schema, digest;
    std::vector<ProgramResourcePlanEntry> plan_entries;
    std::vector<ProgramPersistentValueKey> keys;
    std::vector<ProgramPersistentValueMetadata> metadata;
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> value_bytes;
    std::vector<std::byte> storage;
  };

  ProgramPersistentValueStore() = default;
  ProgramPersistentValueStore(const ProgramPersistentValueStore&) = default;
  ProgramPersistentValueStore& operator=(const ProgramPersistentValueStore&) = default;
  ProgramPersistentValueStore(ProgramPersistentValueStore&&) noexcept = default;
  ProgramPersistentValueStore& operator=(ProgramPersistentValueStore&&) noexcept = default;

  /// Prepare all vector capacity in a temporary image, then publish with a non-throwing swap.
  /// A failed bind leaves both the previous accepted image and its metadata untouched.
  void bind(const ProgramResourcePlan& plan) {
    if (bound_)
      throw std::logic_error("Program persistent value store is already bind-sealed");
    ProgramPersistentValueStore prepared;
    prepared.bind_unpublished_(plan);
    swap(prepared);
  }

  void reset() noexcept { ProgramPersistentValueStore{}.swap(*this); }
  [[nodiscard]] ProgramPersistentValueStore clone() const { return *this; }
  [[nodiscard]] Snapshot snapshot() const {
    Snapshot result{bound_, maximum_bytes_, schema_,  digest_, plan_entries_,
                    keys_,  slots_,         offsets_, {},      storage_};
    result.value_bytes.resize(value_bytes_.size());
    for (std::size_t slot = 0; slot != result.value_bytes.size(); ++slot)
      result.value_bytes[slot] = slots_[slot].valid ? plan_entries_[slot].bytes : 0;
    return result;
  }
  void restore(const Snapshot& snapshot) {
    ProgramPersistentValueStore prepared;
    prepared.bound_ = snapshot.bound;
    prepared.maximum_bytes_ = snapshot.maximum_bytes;
    prepared.schema_ = snapshot.schema;
    prepared.digest_ = snapshot.digest;
    prepared.plan_entries_ = snapshot.plan_entries;
    prepared.keys_ = snapshot.keys;
    prepared.slots_ = snapshot.metadata;
    prepared.offsets_ = snapshot.offsets;
    prepared.value_bytes_ = snapshot.value_bytes;
    prepared.storage_ = snapshot.storage;
    prepared.validate_bound_image_();
    swap(prepared);
  }

  /// Build a fully allocated detached image.  No state of this store is changed if validation or
  /// allocation fails; callers may publish the returned image with publish_prepared_restore().
  [[nodiscard]] ProgramPersistentValueStore prepare_restore(const Snapshot& snapshot) const {
    ProgramPersistentValueStore prepared;
    prepared.restore(snapshot);
    return prepared;
  }

  /// Publish an image prepared by prepare_restore() without allocation or throwing operations.
  void publish_prepared_restore(ProgramPersistentValueStore&& prepared) noexcept { swap(prepared); }

  /// Refresh only the mutable accepted slot image from an identically bind-sealed store. The
  /// resource plan, keys, offsets, schema, and digest are immutable after bind; copying those
  /// vectors again would both waste work and allocate on every accepted-step snapshot. Callers use
  /// this after checking the exact bind contract, so the operation has no capacity-changing path.
  void copy_from_preallocated(const ProgramPersistentValueStore& source) {
    if (bound_ != source.bound_ || maximum_bytes_ != source.maximum_bytes_ ||
        schema_ != source.schema_ || digest_ != source.digest_ ||
        plan_entries_ != source.plan_entries_ || keys_ != source.keys_ ||
        offsets_ != source.offsets_ || slots_.size() != source.slots_.size() ||
        storage_.size() != source.storage_.size())
      throw std::logic_error("Program persistent value bind image changed after preparation");
    std::copy(source.slots_.begin(), source.slots_.end(), slots_.begin());
    std::copy(source.storage_.begin(), source.storage_.end(), storage_.begin());
    for (std::size_t slot = 0; slot != value_bytes_.size(); ++slot)
      value_bytes_[slot] = source.slots_[slot].valid ? plan_entries_[slot].bytes : 0;
  }
  void swap(ProgramPersistentValueStore& other) noexcept {
    using std::swap;
    swap(bound_, other.bound_);
    swap(maximum_bytes_, other.maximum_bytes_);
    schema_.swap(other.schema_);
    digest_.swap(other.digest_);
    plan_entries_.swap(other.plan_entries_);
    keys_.swap(other.keys_);
    slots_.swap(other.slots_);
    offsets_.swap(other.offsets_);
    value_bytes_.swap(other.value_bytes_);
    storage_.swap(other.storage_);
  }

  [[nodiscard]] bool bound() const noexcept { return bound_; }
  [[nodiscard]] std::size_t size() const noexcept { return slots_.size(); }
  [[nodiscard]] std::uint64_t maximum_bytes() const noexcept { return maximum_bytes_; }
  [[nodiscard]] std::string_view schema() const noexcept { return schema_; }
  [[nodiscard]] std::string_view digest() const noexcept { return digest_; }
  /// Host-only static rows retained for lossless checkpoint capture. Generated execution addresses
  /// values by dense slot and never traverses these strings.
  [[nodiscard]] const std::vector<ProgramResourcePlanEntry>& resource_plan_entries()
      const noexcept {
    return plan_entries_;
  }
  [[nodiscard]] std::uint64_t value_bytes(std::size_t slot) const noexcept {
    return slots_[slot].valid ? plan_entries_[slot].bytes : 0;
  }

  /// Return the exact logical payload span. ``value(slot)`` intentionally
  /// exposes the reserved capacity for bind-time storage, while this accessor
  /// is used by checkpoint/restart code that must not serialize invalid bytes.
  [[nodiscard]] std::span<std::byte> logical_value(std::size_t slot) noexcept {
    return value(slot).first(static_cast<std::size_t>(value_bytes(slot)));
  }
  [[nodiscard]] std::span<const std::byte> logical_value(std::size_t slot) const noexcept {
    return value(slot).first(static_cast<std::size_t>(value_bytes(slot)));
  }

  [[nodiscard]] const ProgramPersistentValueKey& key(std::size_t slot) const noexcept {
    return keys_[slot];
  }

  [[nodiscard]] std::size_t slot_for(const ProgramPersistentValueKey& key) const {
    if (!bound_)
      throw std::logic_error("Program persistent value store is not bind-sealed");
    for (std::size_t slot = 0; slot != keys_.size(); ++slot)
      if (keys_[slot] == key)
        return slot;
    throw std::out_of_range("Program persistent value key is absent from the sealed manifest");
  }
  [[nodiscard]] ProgramPersistentValueMetadata& metadata(std::size_t slot) noexcept {
    return slots_[slot];
  }
  [[nodiscard]] const ProgramPersistentValueMetadata& metadata(std::size_t slot) const noexcept {
    return slots_[slot];
  }
  [[nodiscard]] std::span<std::byte> value(std::size_t slot) noexcept {
    const auto begin = static_cast<std::size_t>(offsets_[slot]);
    const auto end = static_cast<std::size_t>(offsets_[slot + 1]);
    return std::span<std::byte>(storage_).subspan(begin, end - begin);
  }
  [[nodiscard]] std::span<const std::byte> value(std::size_t slot) const noexcept {
    const auto begin = static_cast<std::size_t>(offsets_[slot]);
    const auto end = static_cast<std::size_t>(offsets_[slot + 1]);
    return std::span<const std::byte>(storage_).subspan(begin, end - begin);
  }

 private:
  void bind_unpublished_(const ProgramResourcePlan& plan) {
    const auto count = plan.entries().size();
    plan_entries_ = plan.entries();
    keys_.resize(count);
    slots_.resize(count);
    offsets_.resize(count + 1, 0);
    value_bytes_.resize(count, 0);
    std::uint64_t bytes = 0;
    for (const auto& entry : plan.entries()) {
      const auto slot = static_cast<std::size_t>(entry.slot);
      keys_[slot] = entry.key;
      offsets_[slot] = bytes;
      value_bytes_[slot] = 0;
      if (entry.maximum_bytes > std::numeric_limits<std::uint64_t>::max() - bytes)
        throw std::overflow_error("Program persistent value buffer bound overflows uint64");
      bytes += entry.maximum_bytes;
    }
    offsets_.back() = bytes;
    if (bytes > maximum_size_t_())
      throw std::overflow_error("Program persistent value buffer bound exceeds size_t");
    storage_.resize(static_cast<std::size_t>(bytes));
    maximum_bytes_ = plan.maximum_bytes();
    schema_ = std::string(plan.schema());
    digest_ = std::string(plan.digest());
    bound_ = true;
    validate_bound_image_();
  }
  void validate_bound_image_() const {
    if (!bound_) {
      if (maximum_bytes_ != 0 || !schema_.empty() || !digest_.empty() || !plan_entries_.empty() ||
          !keys_.empty() || !slots_.empty() || !offsets_.empty() || !value_bytes_.empty() ||
          !storage_.empty())
        throw std::invalid_argument("unbound Program persistent value snapshot owns storage");
      return;
    }
    if (schema_ != "program-resource-plan:v1" || digest_.size() != 64 ||
        !std::all_of(digest_.begin(), digest_.end(),
                     [](unsigned char value) {
                       return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
                     }) ||
        plan_entries_.size() != keys_.size() || keys_.size() != slots_.size() ||
        value_bytes_.size() != keys_.size() || offsets_.size() != keys_.size() + 1 ||
        offsets_.empty() || offsets_.front() != 0 || offsets_.back() != storage_.size() ||
        offsets_.back() > maximum_bytes_ || offsets_.back() > maximum_size_t_())
      throw std::invalid_argument("Program persistent value snapshot is malformed");
    // Re-run the complete static plan validator.  This also authenticates dense slots, duplicate
    // complete keys, digest collisions, providers, dimensions and exact byte bounds before a swap.
    ProgramResourcePlan plan(plan_entries_, maximum_bytes_, schema_, digest_);
    (void)plan;
    for (std::size_t slot = 1; slot != offsets_.size(); ++slot)
      if (offsets_[slot] < offsets_[slot - 1])
        throw std::invalid_argument("Program persistent value snapshot has descending offsets");
    for (std::size_t slot = 0; slot != keys_.size(); ++slot) {
      const auto& entry = plan_entries_[slot];
      const auto& metadata = slots_[slot];
      const std::uint64_t extent = offsets_[slot + 1] - offsets_[slot];
      const std::uint64_t expected_value_bytes = metadata.valid ? entry.bytes : 0;
      if (entry.slot != slot || entry.key != keys_[slot] ||
          value_bytes_[slot] != expected_value_bytes || extent != entry.maximum_bytes ||
          value_bytes_[slot] > extent || metadata.valid == metadata.cold)
        throw std::invalid_argument("Program persistent value snapshot row/storage mismatch");
      if (!std::isfinite(metadata.accumulated_dt) || metadata.accumulated_dt < 0.0)
        throw std::invalid_argument("Program persistent value snapshot has invalid accumulated dt");
    }
  }
  [[nodiscard]] static constexpr std::uint64_t maximum_size_t_() noexcept {
    return static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
  }
  bool bound_ = false;
  std::uint64_t maximum_bytes_ = 0;
  std::string schema_, digest_;
  std::vector<ProgramResourcePlanEntry> plan_entries_;
  std::vector<ProgramPersistentValueKey> keys_;
  std::vector<ProgramPersistentValueMetadata> slots_;
  std::vector<std::uint64_t> offsets_;
  std::vector<std::uint64_t> value_bytes_;
  std::vector<std::byte> storage_;
};

inline void swap(ProgramPersistentValueStore& left, ProgramPersistentValueStore& right) noexcept {
  left.swap(right);
}

}  // namespace pops::runtime::program
