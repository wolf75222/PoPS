#pragma once

/// @file
/// @brief Stable, typed host image used while a native Program is prepared.
///
/// The v5 ABI deliberately keeps ``ProgramPreparationHostRef::image`` opaque.  This host-side
/// object gives that pointer a concrete, checked lifetime: a candidate may construct its execution
/// services from the image, but it cannot recover a System/AmrSystem facade from the raw service
/// registry.  The typed specialisation lives with ProgramExecutionServices so this lightweight ABI
/// header remains independent of the two runtime facades.

#include <pops/runtime/program/program_abi.hpp>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace pops::runtime::program {

class ProgramPreparationImage {
 public:
  static constexpr std::uint64_t kMagic = UINT64_C(0x504F505350524550);  // "POPSPREP"
  static constexpr std::uint32_t kSchemaVersion = 1;

  ProgramPreparationImage(const ProgramPreparationImage&) = delete;
  ProgramPreparationImage& operator=(const ProgramPreparationImage&) = delete;
  ~ProgramPreparationImage() = default;

  [[nodiscard]] std::uint32_t native_dimension() const noexcept { return native_dimension_; }
  [[nodiscard]] ProgramRuntimeKind runtime_kind() const noexcept { return runtime_kind_; }
  [[nodiscard]] const ProgramExecutionServicesRef& services() const noexcept { return services_; }
  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
  [[nodiscard]] std::uint64_t service_witness() const noexcept { return service_witness_; }
  /// Inspection images carry a complete POD service table solely so a v5 descriptor remains
  /// structurally self-consistent.  They are deliberately not execution images: a candidate may
  /// describe itself through `pops_install_program`, but cannot recover a provider until the host
  /// has captured the detached topology and bound the final preparation image.
  [[nodiscard]] bool execution_ready() const noexcept { return execution_ready_; }

  [[nodiscard]] bool matches_services(const ProgramExecutionServicesRef& services) const noexcept {
    return services_.state_store == services.state_store &&
           services_.field_store == services.field_store &&
           services_.spatial_executor == services.spatial_executor &&
           services_.hierarchy_executor == services.hierarchy_executor &&
           services_.history_store == services.history_store &&
           services_.clock_service == services.clock_service &&
           services_.reduction_service == services.reduction_service &&
           services_.transaction_service == services.transaction_service &&
           services_.persistent_value_store == services.persistent_value_store &&
           service_witness_ == service_witness_for_(services);
  }

  [[nodiscard]] bool matches(const ProgramHostDescriptor& host) const noexcept {
    return magic_ == kMagic && schema_version_ == kSchemaVersion &&
           native_dimension_ == host.native_dimension && runtime_kind_ == host.runtime_kind;
  }

 protected:
  ProgramPreparationImage(std::uint32_t native_dimension, ProgramRuntimeKind runtime_kind,
                          ProgramExecutionServicesRef services, std::uint64_t generation,
                          bool execution_ready = true)
      : native_dimension_(native_dimension),
        runtime_kind_(runtime_kind),
        generation_(generation),
        source_services_(services),
        source_service_witness_(service_witness_for_(services)),
        execution_ready_(execution_ready) {
    if (generation_ == 0)
      throw std::invalid_argument("Program preparation image generation must be non-zero");
  }

  /// The ABI POD given to a DSO names only image-owned adapter identities.  The original facade
  /// service table is retained solely as a host-side witness used by bind_program_preparation_image.
  void bind_image_services(ProgramExecutionServicesRef services) {
    if (services.state_store == nullptr || services.field_store == nullptr ||
        services.spatial_executor == nullptr || services.hierarchy_executor == nullptr ||
        services.history_store == nullptr || services.clock_service == nullptr ||
        services.reduction_service == nullptr || services.transaction_service == nullptr ||
        services.persistent_value_store == nullptr)
      throw std::invalid_argument(
          "Program preparation image has an incomplete adapter service table");
    services_ = services;
    service_witness_ = service_witness_for_(services_);
  }

 private:
  [[nodiscard]] bool source_services_equal_(
      const ProgramExecutionServicesRef& services) const noexcept {
    return source_services_.state_store == services.state_store &&
           source_services_.field_store == services.field_store &&
           source_services_.spatial_executor == services.spatial_executor &&
           source_services_.hierarchy_executor == services.hierarchy_executor &&
           source_services_.history_store == services.history_store &&
           source_services_.clock_service == services.clock_service &&
           source_services_.reduction_service == services.reduction_service &&
           source_services_.transaction_service == services.transaction_service &&
           source_services_.persistent_value_store == services.persistent_value_store &&
           source_service_witness_ == service_witness_for_(services);
  }
  static std::uint64_t service_witness_for_(const ProgramExecutionServicesRef& services) noexcept {
    const auto word = [](const void* pointer) noexcept {
      return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
    };
    return word(services.state_store) ^ (word(services.field_store) << 1U) ^
           (word(services.spatial_executor) << 7U) ^ (word(services.hierarchy_executor) << 11U) ^
           (word(services.history_store) << 17U) ^ (word(services.clock_service) << 23U) ^
           (word(services.reduction_service) << 31U) ^ (word(services.transaction_service) << 41U) ^
           (word(services.persistent_value_store) << 53U);
  }

  std::uint64_t magic_ = kMagic;
  std::uint32_t schema_version_ = kSchemaVersion;
  std::uint32_t native_dimension_ = 0;
  ProgramRuntimeKind runtime_kind_ = ProgramRuntimeKind::uniform;
  std::uint64_t generation_ = 0;
  ProgramExecutionServicesRef services_{};
  ProgramExecutionServicesRef source_services_{};
  std::uint64_t service_witness_ = service_witness_for_(services_);
  std::uint64_t source_service_witness_ = 0;
  bool execution_ready_ = false;
};

static_assert(std::is_standard_layout_v<ProgramPreparationImage>);

inline void bind_program_preparation_image(ProgramHostDescriptor& host,
                                           const std::shared_ptr<ProgramPreparationImage>& image) {
  if (!image || !image->matches(host))
    throw std::invalid_argument("Program preparation image does not match the host descriptor");
  host.preparation.struct_size = sizeof(ProgramPreparationHostRef);
  host.preparation.abi_version = kProgramInstallAbiVersion;
  // `pops_install_program` receives the outer descriptor before it receives a prepare callback.
  // Substitute both views now: an image-only `preparation.services` is insufficient if the DSO
  // can retain the outer `host.services` facade table during inspection.
  host.services = image->services();
  host.preparation.services = image->services();
  host.preparation.image = image.get();
}

[[nodiscard]] inline const ProgramPreparationImage& require_program_preparation_image(
    const ProgramPreparationHostRef& preparation, std::uint32_t native_dimension) {
  if (preparation.struct_size != sizeof(ProgramPreparationHostRef) ||
      preparation.abi_version != kProgramInstallAbiVersion || preparation.image == nullptr)
    throw std::invalid_argument("Program preparation image is not sealed");
  std::uint64_t magic = 0;
  std::memcpy(&magic, preparation.image, sizeof(magic));
  if (magic != ProgramPreparationImage::kMagic)
    throw std::invalid_argument("Program preparation image has no host image tag");
  const auto& image = *static_cast<const ProgramPreparationImage*>(preparation.image);
  if (image.native_dimension() != native_dimension ||
      (image.runtime_kind() != ProgramRuntimeKind::uniform &&
       image.runtime_kind() != ProgramRuntimeKind::amr) ||
      image.services().state_store == nullptr || !image.matches_services(preparation.services))
    throw std::invalid_argument("Program preparation image has the wrong native authority");
  return image;
}

/// `inspect_program_installation` accepts an image that is intentionally not ready for provider
/// construction, while `OwnedProgramInstallation::prepare` requires the detached execution image.
/// Keeping this distinction in the host tag prevents a DSO from treating an inspection descriptor
/// as a borrowed runtime facade.
[[nodiscard]] inline const ProgramPreparationImage& require_program_execution_preparation_image(
    const ProgramPreparationHostRef& preparation, std::uint32_t native_dimension,
    ProgramRuntimeKind runtime_kind) {
  const auto& image = require_program_preparation_image(preparation, native_dimension);
  if (image.runtime_kind() != runtime_kind)
    throw std::invalid_argument("Program preparation image has the wrong runtime authority");
  if (!image.execution_ready())
    throw std::invalid_argument("Program preparation image has no detached execution authority");
  return image;
}

[[nodiscard]] inline const ProgramPreparationImage& require_program_preparation_image(
    const ProgramPreparationHostRef& preparation, std::uint32_t native_dimension,
    ProgramRuntimeKind runtime_kind) {
  const auto& image = require_program_preparation_image(preparation, native_dimension);
  if (image.runtime_kind() != runtime_kind)
    throw std::invalid_argument("Program preparation image has the wrong runtime authority");
  return image;
}

}  // namespace pops::runtime::program
