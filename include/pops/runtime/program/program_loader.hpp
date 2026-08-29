#pragma once

#include <pops/runtime/dynamic/dynlib.hpp>
#include <pops/runtime/program/owned_program_installation.hpp>

#include <memory>
#include <stdexcept>
#include <string>

namespace pops::runtime::program {

inline OwnedProgramInstallation inspect_program_installation(pops::dynlib::UniqueHandle image,
                                                             const ProgramHostDescriptor& host) {
  if (!valid_program_host_descriptor(host))
    throw std::invalid_argument("Program loader received an invalid v5 host descriptor");
  // `valid_program_host_descriptor` stays POD-only so a malformed foreign pointer cannot be
  // dereferenced while validating an ABI record.  The loader is the first host-only boundary and
  // therefore authenticates the opaque image tag and the exact substituted service table before
  // calling into the DSO.
  const auto& preparation_image = require_program_preparation_image(
      host.preparation, host.native_dimension, host.runtime_kind);
  if (!preparation_image.matches_services(host.services))
    throw std::invalid_argument("Program loader received an incoherent preparation image");
  if (!pops::dynlib::valid(image.get()))
    throw std::runtime_error("Program loader received an invalid program image");
  const auto install = reinterpret_cast<ProgramInstallFn>(pops::dynlib::sym(image.get(), "pops_install_program"));
  if (!install)
    throw std::runtime_error("Program loader: pops_install_program v5 entry is missing");
  ProgramCandidateDescriptor candidate{};
  struct CandidateCleanup final {
    ProgramCandidateDescriptor* candidate = nullptr;
    ~CandidateCleanup() {
      if (candidate != nullptr && candidate->destroy != nullptr && candidate->context != nullptr)
        candidate->destroy(candidate->context);
    }
    void release() noexcept { candidate = nullptr; }
  } cleanup{&candidate};
  ProgramInstallDiagnostic diagnostic{};
  if (!install(&host, &candidate, &diagnostic)) {
    std::size_t size = 0;
    while (size != sizeof(diagnostic.message) && diagnostic.message[size] != '\0')
      ++size;
    throw std::runtime_error("Program loader: candidate refused: " +
                             std::string(diagnostic.message, size));
  }
  if (!valid_program_candidate_descriptor(candidate)) {
    throw std::runtime_error("Program loader: candidate descriptor is malformed");
  }
  if (candidate.native_dimension != host.native_dimension || candidate.runtime_kind != host.runtime_kind ||
      (candidate.required_capability_bits & ~host.capability_bits) != 0) {
    throw std::runtime_error("Program loader: candidate runtime authority differs from host");
  }
  const auto require_service = [&](std::uint64_t bit, const void* service) {
    return (candidate.required_service_bits & bit) == 0 || service != nullptr;
  };
  if (!require_service(kProgramServiceState, host.services.state_store) ||
      !require_service(kProgramServiceFields, host.services.field_store) ||
      !require_service(kProgramServiceSpatial, host.services.spatial_executor) ||
      !require_service(kProgramServiceHierarchy, host.services.hierarchy_executor) ||
      !require_service(kProgramServiceHistory, host.services.history_store) ||
      !require_service(kProgramServiceClock, host.services.clock_service) ||
      !require_service(kProgramServiceReduction, host.services.reduction_service) ||
      !require_service(kProgramServiceTransaction, host.services.transaction_service) ||
      !require_service(kProgramServicePersistentValues, host.services.persistent_value_store)) {
    throw std::runtime_error("Program loader: candidate requires an unavailable host service");
  }
  // Copy and validate every DSO-backed view before ownership leaves this scope.  In particular,
  // `PreparedProgramInstallation` must not retain a record whose string points into the image.
  std::size_t aggregate_metadata_bytes = 0;
  auto metadata = ProgramInstallationMetadata::materialize(candidate, aggregate_metadata_bytes);
  auto tables = ProgramInstallationTables::materialize(candidate, aggregate_metadata_bytes);
  // This is deliberately before ``OwnedProgramInstallation`` escapes and, more importantly,
  // before the generated candidate can receive its prepare callback.  A forged digest or a
  // temporal manifest whose embedded resource plan differs by one slot is a refusal, never a
  // rollback path through a live System/AMR facade.
  tables.validate_resource_authority(metadata, candidate.maximum_bytes);
  OwnedProgramInstallation result(std::move(image), candidate, std::move(metadata), std::move(tables));
  cleanup.release();
  return result;
}

inline PreparedProgramInstallation prepare_program_installation(
    pops::dynlib::UniqueHandle image, const ProgramHostDescriptor& host,
    std::shared_ptr<ProgramPreparationImage> preparation_image) {
  ProgramHostDescriptor preparation_host = host;
  bind_program_preparation_image(preparation_host, preparation_image);
  auto owner = inspect_program_installation(std::move(image), preparation_host);
  owner.set_preparation_image(std::move(preparation_image));
  owner.prepare(preparation_host);
  return PreparedProgramInstallation(std::move(owner));
}

inline PreparedProgramInstallation prepare_program_installation(
    const std::string& path, const ProgramHostDescriptor& host,
    std::shared_ptr<ProgramPreparationImage> preparation_image) {
  pops::dynlib::UniqueHandle image(pops::dynlib::open_private_image(path));
  if (!pops::dynlib::valid(image.get()))
    throw std::runtime_error("Program loader cannot open '" + path + "': " + pops::dynlib::last_error());
  return prepare_program_installation(std::move(image), host, std::move(preparation_image));
}

inline OwnedProgramInstallation inspect_program_installation(const std::string& path,
                                                             const ProgramHostDescriptor& host) {
  pops::dynlib::UniqueHandle image(pops::dynlib::open_private_image(path));
  if (!pops::dynlib::valid(image.get()))
    throw std::runtime_error("Program loader cannot open '" + path + "': " + pops::dynlib::last_error());
  return inspect_program_installation(std::move(image), host);
}

}  // namespace pops::runtime::program
