#pragma once

/// @file
/// @brief Lossless, versioned checkpoint image for persistent Program values.
///
/// The checkpoint carrier is deliberately host-owned.  Generated execution never consumes this
/// type: it addresses a bound value by its dense slot and therefore has no map/string lookup on the
/// hot path.  Checkpoint preparation validates the complete static plan and materializes every
/// potentially-fallible allocation before a caller publishes the detached store image.

#include <pops/core/identity/sha256.hpp>
#include <pops/runtime/program/program_persistent_value_store.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::program {

inline constexpr std::string_view kProgramPersistentValueCheckpointSchema =
    "program-persistent-value-checkpoint:v1";
inline constexpr std::string_view kProgramPersistentValueCheckpointPlanSchema =
    "program-resource-plan:v1";

/// One complete accepted image.  ``rows`` is ordered by dense ``slot`` and is never reduced to a
/// digest-only identity.  ``offsets`` address the reserved per-slot capacity (maximum_bytes), while
/// ``value_bytes`` records the exact logical payload size from the static plan for valid rows and
/// is zero for invalid/cold rows whose capacity and raw storage remain authenticated.
struct ProgramPersistentValueCheckpoint final {
  bool bound = false;
  std::string schema = std::string(kProgramPersistentValueCheckpointSchema);
  std::string plan_schema;
  std::string plan_digest;
  std::uint64_t maximum_bytes = 0;
  std::uint32_t slot_count = 0;
  std::vector<ProgramResourcePlanEntry> rows;
  std::vector<ProgramPersistentValueMetadata> metadata;
  std::vector<std::uint64_t> offsets;
  std::vector<std::uint64_t> value_bytes;
  std::vector<std::byte> storage;

  friend bool operator==(const ProgramPersistentValueCheckpoint&,
                         const ProgramPersistentValueCheckpoint&) = default;
};

/// Host-owned detached restore image.  Decoding, plan authentication and allocation happen before
/// this value is constructed.  Publication validates only the immutable installation generation,
/// then exchanges the preallocated store without allocation; a handle is consumable exactly once.
class PreparedProgramPersistentValueRestore final {
 public:
  PreparedProgramPersistentValueRestore() = delete;
  PreparedProgramPersistentValueRestore(ProgramPersistentValueStore prepared,
                                        std::uint64_t install_generation) noexcept
      : prepared_(std::move(prepared)), install_generation_(install_generation) {}
  PreparedProgramPersistentValueRestore(const PreparedProgramPersistentValueRestore&) = delete;
  PreparedProgramPersistentValueRestore& operator=(const PreparedProgramPersistentValueRestore&) =
      delete;
  PreparedProgramPersistentValueRestore(PreparedProgramPersistentValueRestore&& other) noexcept
      : prepared_(std::move(other.prepared_)),
        install_generation_(std::exchange(other.install_generation_, 0)),
        validated_(std::exchange(other.validated_, false)),
        consumed_(std::exchange(other.consumed_, true)) {}
  PreparedProgramPersistentValueRestore& operator=(PreparedProgramPersistentValueRestore&&) =
      delete;

  [[nodiscard]] std::uint64_t install_generation() const noexcept { return install_generation_; }
  [[nodiscard]] bool consumed() const noexcept { return consumed_; }

  /// Throwing identity/consumption check for the collective pre-publication phase.
  void validate_publication(std::uint64_t current_install_generation) {
    validated_ = false;
    if (consumed_)
      throw std::logic_error("Program persistent value restore image was already consumed");
    if (install_generation_ == 0 || install_generation_ != current_install_generation)
      throw std::logic_error(
          "Program persistent value restore image targets a different installed Program");
    validated_ = true;
  }

  /// Final half of publication.  Call only after ``validate_publication`` succeeded on every
  /// rank while the enclosing restart transaction owns its accepted writer; this performs no
  /// allocation, validation, or collective operation.
  void publish_validated_into(ProgramPersistentValueStore& accepted) noexcept {
    if (!validated_ || consumed_)
      std::terminate();
    static_assert(noexcept(accepted.swap(prepared_)));
    accepted.swap(prepared_);
    validated_ = false;
    consumed_ = true;
  }

 private:
  ProgramPersistentValueStore prepared_;
  std::uint64_t install_generation_ = 0;
  bool validated_ = false;
  bool consumed_ = false;
};

namespace persistent_value_checkpoint_detail {

inline constexpr std::array<std::uint8_t, 8> kMagic{'P', 'O', 'P', 'S', 'P', 'V', 'S', '1'};

[[noreturn]] inline void fail(std::string_view reason) {
  throw std::invalid_argument("invalid Program persistent value checkpoint: " +
                              std::string(reason));
}

inline bool valid_digest(std::string_view digest) noexcept {
  if (digest.size() != 64)
    return false;
  for (const unsigned char value : digest)
    if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f')))
      return false;
  return true;
}

inline void require_enum_values(const ProgramResourcePlanEntry& row) {
  const auto lifetime = static_cast<std::uint8_t>(row.lifetime);
  const auto centering = static_cast<std::uint8_t>(row.centering);
  const auto off_policy = static_cast<std::uint8_t>(row.off_policy);
  const auto transfer = static_cast<std::uint8_t>(row.spatial_transfer);
  if (lifetime < static_cast<std::uint8_t>(ProgramValueLifetime::transient) ||
      lifetime > static_cast<std::uint8_t>(ProgramValueLifetime::persistent_schedule) ||
      centering < static_cast<std::uint8_t>(ProgramValueCentering::cell) ||
      centering > static_cast<std::uint8_t>(ProgramValueCentering::node) ||
      off_policy > static_cast<std::uint8_t>(ProgramScheduleOffPolicy::error) ||
      transfer < static_cast<std::uint8_t>(ProgramSpatialTransferPolicy::redistribute_exact) ||
      transfer > static_cast<std::uint8_t>(ProgramSpatialTransferPolicy::refuse))
    fail("resource row has an unknown enum value");
}

inline void validate(const ProgramPersistentValueCheckpoint& image) {
  if (image.schema != kProgramPersistentValueCheckpointSchema)
    fail("unsupported checkpoint schema");
  if (!image.bound) {
    if (!image.plan_schema.empty() || !image.plan_digest.empty() || image.maximum_bytes != 0 ||
        image.slot_count != 0 || !image.rows.empty() || !image.metadata.empty() ||
        !image.offsets.empty() || !image.value_bytes.empty() || !image.storage.empty())
      fail("unbound checkpoint owns a resource image");
    return;
  }
  if (image.plan_schema != kProgramPersistentValueCheckpointPlanSchema)
    fail("unsupported resource-plan schema");
  if (!valid_digest(image.plan_digest))
    fail("resource-plan digest is not lowercase SHA-256");
  if (image.rows.size() > std::numeric_limits<std::uint32_t>::max() ||
      image.slot_count != image.rows.size())
    throw std::overflow_error("Program persistent value checkpoint slot count overflows uint32");
  if (image.rows.size() != image.metadata.size() || image.rows.size() != image.value_bytes.size() ||
      image.offsets.size() != image.rows.size() + 1 || image.offsets.empty())
    fail("row, metadata, value-size and offset counts disagree");
  if (image.offsets.front() != 0 || image.offsets.back() != image.storage.size() ||
      image.offsets.back() > image.maximum_bytes ||
      image.offsets.back() > std::numeric_limits<std::size_t>::max())
    fail("storage offsets exceed the authenticated memory bound");
  for (std::size_t slot = 1; slot != image.offsets.size(); ++slot)
    if (image.offsets[slot] < image.offsets[slot - 1])
      fail("storage offsets descend");

  std::uint64_t maximum_total = 0;
  for (std::size_t slot = 0; slot != image.rows.size(); ++slot) {
    const auto& row = image.rows[slot];
    require_enum_values(row);
    const auto& metadata = image.metadata[slot];
    const std::uint64_t extent = image.offsets[slot + 1] - image.offsets[slot];
    const std::uint64_t expected_value_bytes = metadata.valid ? row.bytes : 0;
    if (row.slot != slot || image.value_bytes[slot] != expected_value_bytes ||
        extent != row.maximum_bytes || image.value_bytes[slot] > extent ||
        metadata.valid == metadata.cold)
      fail("dense row and storage metadata disagree");
    if (maximum_total > std::numeric_limits<std::uint64_t>::max() - row.maximum_bytes)
      throw std::overflow_error(
          "Program persistent value checkpoint memory bound overflows uint64");
    maximum_total += row.maximum_bytes;
    if (!std::isfinite(metadata.accumulated_dt) || metadata.accumulated_dt < 0.0)
      fail("slot accumulated_dt is not finite and non-negative");
  }
  if (maximum_total > image.maximum_bytes)
    fail("resource rows exceed the checkpoint memory ceiling");

  // Re-run all static validation, including duplicate complete keys, digest collisions, provider
  // requirements, exact bytes and dense slot numbering.  The constructor is intentionally used on
  // the host side only; no generated execution path takes this lock-free metadata route.
  ProgramResourcePlan plan(image.rows, image.maximum_bytes, image.plan_schema, image.plan_digest);
  (void)plan;
}

class Writer final {
 public:
  void raw(std::span<const std::uint8_t> bytes) {
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }
  void u8(std::uint8_t value) { bytes_.push_back(value); }
  void u32(std::uint32_t value) {
    for (int shift = 0; shift != 32; shift += 8)
      bytes_.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
  void u64(std::uint64_t value) {
    for (int shift = 0; shift != 64; shift += 8)
      bytes_.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
  void i32(std::int32_t value) {
    u64(static_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
  }
  void real(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u64(bits);
  }
  void string(std::string_view value) {
    u64(static_cast<std::uint64_t>(value.size()));
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }
  void bytes(std::span<const std::byte> value) {
    u64(static_cast<std::uint64_t>(value.size()));
    for (const std::byte byte : value)
      bytes_.push_back(std::to_integer<std::uint8_t>(byte));
  }
  [[nodiscard]] std::vector<std::uint8_t> take() && { return std::move(bytes_); }

 private:
  std::vector<std::uint8_t> bytes_;
};

class Reader final {
 public:
  explicit Reader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  void expect_raw(std::span<const std::uint8_t> expected) {
    require_(expected.size());
    for (std::size_t index = 0; index != expected.size(); ++index)
      if (bytes_[cursor_ + index] != expected[index])
        fail("unsupported checkpoint magic");
    cursor_ += expected.size();
  }
  [[nodiscard]] std::uint8_t u8() {
    require_(1);
    return bytes_[cursor_++];
  }
  [[nodiscard]] std::uint32_t u32() {
    require_(sizeof(std::uint32_t));
    std::uint32_t value = 0;
    for (int shift = 0; shift != 32; shift += 8)
      value |= static_cast<std::uint32_t>(bytes_[cursor_++]) << shift;
    return value;
  }
  [[nodiscard]] std::uint64_t u64() {
    require_(sizeof(std::uint64_t));
    std::uint64_t value = 0;
    for (int shift = 0; shift != 64; shift += 8)
      value |= static_cast<std::uint64_t>(bytes_[cursor_++]) << shift;
    return value;
  }
  [[nodiscard]] std::int32_t i32() {
    const std::int64_t value = static_cast<std::int64_t>(u64());
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max())
      fail("signed integer is outside int32 range");
    return static_cast<std::int32_t>(value);
  }
  [[nodiscard]] double real() {
    const std::uint64_t bits = u64();
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  [[nodiscard]] std::size_t count(std::size_t element_bytes = 1) {
    const std::uint64_t value = u64();
    if (element_bytes == 0 || value > std::numeric_limits<std::size_t>::max() ||
        value > static_cast<std::uint64_t>((bytes_.size() - cursor_) / element_bytes))
      fail("container length is not credible");
    return static_cast<std::size_t>(value);
  }
  [[nodiscard]] std::string string() {
    const auto size = count();
    require_(size);
    std::string value(reinterpret_cast<const char*>(bytes_.data() + cursor_), size);
    cursor_ += size;
    return value;
  }
  [[nodiscard]] std::vector<std::byte> bytes() {
    const auto size = count();
    require_(size);
    std::vector<std::byte> value(size);
    for (std::size_t index = 0; index != size; ++index)
      value[index] = static_cast<std::byte>(bytes_[cursor_ + index]);
    cursor_ += size;
    return value;
  }
  void finish() const {
    if (cursor_ != bytes_.size())
      fail("trailing bytes");
  }

 private:
  void require_(std::size_t count) const {
    if (count > bytes_.size() - cursor_)
      fail("truncated payload");
  }

  std::span<const std::uint8_t> bytes_;
  std::size_t cursor_ = 0;
};

inline void write_optional_u64(Writer& out, const std::optional<std::uint64_t>& value) {
  out.u8(value ? 1U : 0U);
  if (value)
    out.u64(*value);
}

inline std::optional<std::uint64_t> read_optional_u64(Reader& in) {
  const auto present = in.u8();
  if (present > 1)
    fail("optional integer marker is invalid");
  return present == 0 ? std::nullopt : std::optional<std::uint64_t>(in.u64());
}

inline void write_entry(Writer& out, const ProgramResourcePlanEntry& row) {
  out.u32(row.slot);
  out.u64(row.key.value_id);
  out.u64(row.key.occurrence_path_id);
  out.u32(row.key.owner);
  out.u32(row.key.space);
  out.u32(row.key.clock);
  out.i32(row.key.amr_level);
  out.string(row.identity);
  out.string(row.occurrence_path);
  out.string(row.owner_identity);
  out.string(row.space_identity);
  out.string(row.clock_identity);
  out.u8(static_cast<std::uint8_t>(row.lifetime));
  out.u8(static_cast<std::uint8_t>(row.centering));
  out.u8(static_cast<std::uint8_t>(row.off_policy));
  out.u8(static_cast<std::uint8_t>(row.spatial_transfer));
  out.u32(row.components);
  out.u32(row.ghosts);
  out.u64(row.bytes);
  out.u64(row.maximum_bytes);
  out.u8(row.communicates ? 1U : 0U);
  out.u8(row.restart_required ? 1U : 0U);
  out.string(row.communication);
  out.string(row.transfer_identity);
  out.string(row.restart_identity);
  out.string(row.component_names);
  out.string(row.shape);
  write_optional_u64(out, row.cells);
  write_optional_u64(out, row.itemsize);
}

inline ProgramResourcePlanEntry read_entry(Reader& in) {
  ProgramResourcePlanEntry row;
  row.slot = in.u32();
  row.key.value_id = in.u64();
  row.key.occurrence_path_id = in.u64();
  row.key.owner = in.u32();
  row.key.space = in.u32();
  row.key.clock = in.u32();
  row.key.amr_level = in.i32();
  row.identity = in.string();
  row.occurrence_path = in.string();
  row.owner_identity = in.string();
  row.space_identity = in.string();
  row.clock_identity = in.string();
  row.lifetime = static_cast<ProgramValueLifetime>(in.u8());
  row.centering = static_cast<ProgramValueCentering>(in.u8());
  row.off_policy = static_cast<ProgramScheduleOffPolicy>(in.u8());
  row.spatial_transfer = static_cast<ProgramSpatialTransferPolicy>(in.u8());
  row.components = in.u32();
  row.ghosts = in.u32();
  row.bytes = in.u64();
  row.maximum_bytes = in.u64();
  const auto communicates = in.u8();
  const auto restart_required = in.u8();
  if (communicates > 1 || restart_required > 1)
    fail("resource boolean is not 0 or 1");
  row.communicates = communicates != 0;
  row.restart_required = restart_required != 0;
  row.communication = in.string();
  row.transfer_identity = in.string();
  row.restart_identity = in.string();
  row.component_names = in.string();
  row.shape = in.string();
  row.cells = read_optional_u64(in);
  row.itemsize = read_optional_u64(in);
  return row;
}

inline void write_metadata(Writer& out, const ProgramPersistentValueMetadata& metadata) {
  out.u64(metadata.accepted_coordinate);
  out.u64(metadata.cursor);
  out.real(metadata.accumulated_dt);
  out.u64(metadata.topology_epoch);
  out.u64(metadata.layout_generation);
  out.u8(metadata.valid ? 1U : 0U);
  out.u8(metadata.cold ? 1U : 0U);
}

inline ProgramPersistentValueMetadata read_metadata(Reader& in) {
  ProgramPersistentValueMetadata metadata;
  metadata.accepted_coordinate = in.u64();
  metadata.cursor = in.u64();
  metadata.accumulated_dt = in.real();
  metadata.topology_epoch = in.u64();
  metadata.layout_generation = in.u64();
  const auto valid = in.u8();
  const auto cold = in.u8();
  if (valid > 1 || cold > 1)
    fail("slot boolean is not 0 or 1");
  metadata.valid = valid != 0;
  metadata.cold = cold != 0;
  return metadata;
}

inline void write_body(Writer& out, const ProgramPersistentValueCheckpoint& image) {
  out.raw(kMagic);
  out.u8(image.bound ? 1U : 0U);
  out.string(image.schema);
  out.string(image.plan_schema);
  out.string(image.plan_digest);
  out.u64(image.maximum_bytes);
  out.u32(image.slot_count);
  out.u64(static_cast<std::uint64_t>(image.rows.size()));
  for (const auto& row : image.rows)
    write_entry(out, row);
  out.u64(static_cast<std::uint64_t>(image.metadata.size()));
  for (const auto& metadata : image.metadata)
    write_metadata(out, metadata);
  out.u64(static_cast<std::uint64_t>(image.offsets.size()));
  for (const auto offset : image.offsets)
    out.u64(offset);
  out.u64(static_cast<std::uint64_t>(image.value_bytes.size()));
  for (const auto value_bytes : image.value_bytes)
    out.u64(value_bytes);
  out.bytes(image.storage);
}

inline ProgramPersistentValueCheckpoint read_body(Reader& in) {
  in.expect_raw(kMagic);
  const auto bound = in.u8();
  if (bound > 1)
    fail("bound marker is not 0 or 1");
  ProgramPersistentValueCheckpoint image;
  image.bound = bound != 0;
  image.schema = in.string();
  image.plan_schema = in.string();
  image.plan_digest = in.string();
  image.maximum_bytes = in.u64();
  image.slot_count = in.u32();
  const auto rows = in.count();
  image.rows.reserve(rows);
  for (std::size_t index = 0; index != rows; ++index)
    image.rows.push_back(read_entry(in));
  const auto metadata = in.count();
  image.metadata.reserve(metadata);
  for (std::size_t index = 0; index != metadata; ++index)
    image.metadata.push_back(read_metadata(in));
  const auto offsets = in.count(sizeof(std::uint64_t));
  image.offsets.reserve(offsets);
  for (std::size_t index = 0; index != offsets; ++index)
    image.offsets.push_back(in.u64());
  const auto value_bytes = in.count(sizeof(std::uint64_t));
  image.value_bytes.reserve(value_bytes);
  for (std::size_t index = 0; index != value_bytes; ++index)
    image.value_bytes.push_back(in.u64());
  image.storage = in.bytes();
  in.finish();
  return image;
}

inline ProgramPersistentValueStore::Snapshot snapshot_from_checkpoint(
    const ProgramPersistentValueCheckpoint& image) {
  std::vector<ProgramPersistentValueKey> keys;
  keys.reserve(image.rows.size());
  for (const auto& row : image.rows)
    keys.push_back(row.key);
  return {image.bound,       image.maximum_bytes, image.plan_schema, image.plan_digest,
          image.rows,        std::move(keys),     image.metadata,    image.offsets,
          image.value_bytes, image.storage};
}

inline void require_target_plan(const ProgramPersistentValueCheckpoint& image,
                                const ProgramResourcePlan& target) {
  if (!image.bound || target.schema() != image.plan_schema ||
      target.digest() != image.plan_digest || target.maximum_bytes() != image.maximum_bytes ||
      target.entries() != image.rows)
    fail("checkpoint resource plan differs from the prepared target plan");
}

}  // namespace persistent_value_checkpoint_detail

/// Validate a host-owned image without touching any live store.
inline void validate_program_persistent_value_checkpoint(
    const ProgramPersistentValueCheckpoint& image) {
  persistent_value_checkpoint_detail::validate(image);
}

/// Capture the complete host-owned accepted image, including invalid/cold slot metadata.
[[nodiscard]] inline ProgramPersistentValueCheckpoint capture_program_persistent_value_checkpoint(
    const ProgramPersistentValueStore& store) {
  const auto snapshot = store.snapshot();
  ProgramPersistentValueCheckpoint image;
  image.bound = snapshot.bound;
  if (snapshot.bound) {
    image.plan_schema = snapshot.schema;
    image.plan_digest = snapshot.digest;
    image.maximum_bytes = snapshot.maximum_bytes;
    image.slot_count = static_cast<std::uint32_t>(snapshot.plan_entries.size());
    image.rows = snapshot.plan_entries;
    image.metadata = snapshot.metadata;
    image.offsets = snapshot.offsets;
    image.value_bytes = snapshot.value_bytes;
    image.storage = snapshot.storage;
  }
  persistent_value_checkpoint_detail::validate(image);
  return image;
}

/// Serialize the checkpoint with a trailing SHA-256 over every preceding byte.  This catches both
/// metadata and raw-value corruption before a caller can prepare a detached restore image.
[[nodiscard]] inline std::vector<std::uint8_t> serialize_program_persistent_value_checkpoint(
    const ProgramPersistentValueCheckpoint& image) {
  persistent_value_checkpoint_detail::validate(image);
  persistent_value_checkpoint_detail::Writer body;
  persistent_value_checkpoint_detail::write_body(body, image);
  auto encoded = std::move(body).take();
  const auto digest = identity::sha256_hex(encoded);
  persistent_value_checkpoint_detail::Writer out;
  out.raw(encoded);
  out.string(digest);
  return std::move(out).take();
}

/// Decode and authenticate a version-one carrier.  No runtime store is touched.
[[nodiscard]] inline ProgramPersistentValueCheckpoint
deserialize_program_persistent_value_checkpoint(std::span<const std::uint8_t> bytes) {
  constexpr std::size_t kTrailerBytes = sizeof(std::uint64_t) + 64;
  if (bytes.size() < kTrailerBytes)
    persistent_value_checkpoint_detail::fail("truncated checksum trailer");
  const auto body = bytes.first(bytes.size() - kTrailerBytes);
  const auto trailer = bytes.subspan(body.size());
  persistent_value_checkpoint_detail::Reader trailer_reader(trailer);
  const auto expected_digest = trailer_reader.string();
  trailer_reader.finish();
  const auto actual_digest =
      identity::sha256_hex(std::vector<std::uint8_t>(body.begin(), body.end()));
  if (expected_digest != actual_digest)
    persistent_value_checkpoint_detail::fail("checkpoint SHA-256 digest mismatch");
  persistent_value_checkpoint_detail::Reader reader(body);
  auto image = persistent_value_checkpoint_detail::read_body(reader);
  persistent_value_checkpoint_detail::validate(image);
  return image;
}

/// Prepare a detached image against an exact installed plan.  Allocation and all throwing checks
/// occur before the caller publishes the returned store.
[[nodiscard]] inline ProgramPersistentValueStore prepare_program_persistent_value_restore(
    const ProgramPersistentValueCheckpoint& image, const ProgramResourcePlan& target_plan) {
  persistent_value_checkpoint_detail::validate(image);
  persistent_value_checkpoint_detail::require_target_plan(image, target_plan);
  ProgramPersistentValueStore prepared;
  prepared.restore(persistent_value_checkpoint_detail::snapshot_from_checkpoint(image));
  return prepared;
}

/// Prepare a detached restore without a separately retained target plan.  The checkpoint's own
/// complete rows are reconstructed and validated before allocation.
[[nodiscard]] inline ProgramPersistentValueStore prepare_program_persistent_value_restore(
    const ProgramPersistentValueCheckpoint& image) {
  persistent_value_checkpoint_detail::validate(image);
  if (!image.bound)
    return {};
  const ProgramResourcePlan target(image.rows, image.maximum_bytes, image.plan_schema,
                                   image.plan_digest);
  return prepare_program_persistent_value_restore(image, target);
}

/// Publish a detached restore with no allocation and no throwing operation.
inline void publish_program_persistent_value_restore(
    ProgramPersistentValueStore& store, ProgramPersistentValueStore&& prepared) noexcept {
  store.publish_prepared_restore(std::move(prepared));
}

/// Exact redistribution is valid only for rows whose static policy explicitly permits it.  The
/// optional transform is a host-side rank/layout materializer; it must return a complete image for
/// the target plan.  With no transform, only an unchanged exact image is accepted.
using ProgramPersistentValueRedistributor = std::function<ProgramPersistentValueCheckpoint(
    const ProgramPersistentValueCheckpoint&, const ProgramResourcePlan&)>;

[[nodiscard]] inline ProgramPersistentValueStore prepare_program_persistent_value_redistribution(
    const ProgramPersistentValueCheckpoint& source, const ProgramResourcePlan& target_plan,
    ProgramPersistentValueRedistributor redistributor = {}) {
  persistent_value_checkpoint_detail::validate(source);
  for (const auto& row : source.rows) {
    if (row.spatial_transfer != ProgramSpatialTransferPolicy::redistribute_exact)
      persistent_value_checkpoint_detail::fail(
          "exact redistribution is not permitted by every resource row");
  }
  for (const auto& row : target_plan.entries()) {
    if (row.spatial_transfer != ProgramSpatialTransferPolicy::redistribute_exact)
      persistent_value_checkpoint_detail::fail(
          "exact redistribution target is not permitted by every resource row");
  }
  ProgramPersistentValueCheckpoint target = source;
  if (redistributor) {
    target = redistributor(source, target_plan);
  } else {
    persistent_value_checkpoint_detail::require_target_plan(source, target_plan);
  }
  return prepare_program_persistent_value_restore(target, target_plan);
}

/// Qualified regrid provider seam.  The provider is called only after every row has passed static
/// validation and only when a non-empty provider is supplied.  An absent provider therefore fails
/// before the live store can be mutated; the returned image is validated against the target plan.
using ProgramPersistentValueQualifiedRegridProvider =
    std::function<ProgramPersistentValueCheckpoint(const ProgramPersistentValueCheckpoint&,
                                                   const ProgramResourcePlan&)>;

[[nodiscard]] inline ProgramPersistentValueStore prepare_program_persistent_value_regrid(
    const ProgramPersistentValueCheckpoint& source, const ProgramResourcePlan& target_plan,
    ProgramPersistentValueQualifiedRegridProvider provider = {}) {
  persistent_value_checkpoint_detail::validate(source);
  bool requires_provider = false;
  for (const auto& row : source.rows) {
    if (row.spatial_transfer == ProgramSpatialTransferPolicy::refuse)
      persistent_value_checkpoint_detail::fail("resource row refuses qualified regrid transfer");
    requires_provider |=
        row.spatial_transfer == ProgramSpatialTransferPolicy::qualified_regrid_provider;
  }
  for (const auto& row : target_plan.entries()) {
    if (row.spatial_transfer == ProgramSpatialTransferPolicy::refuse)
      persistent_value_checkpoint_detail::fail(
          "target resource row refuses qualified regrid transfer");
  }
  if (requires_provider && !provider)
    persistent_value_checkpoint_detail::fail("qualified regrid resource row has no provider");
  if (!provider)
    return prepare_program_persistent_value_restore(source, target_plan);
  const auto target = provider(source, target_plan);
  return prepare_program_persistent_value_restore(target, target_plan);
}

}  // namespace pops::runtime::program
