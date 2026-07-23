#pragma once

#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/config/generated_component_abi.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace pops::component {

/// Owning native projection of the sole RuntimeInstance ExecutionContext authority.
///
/// The ABI struct contains borrowed C strings, so retaining the POD received from a temporary
/// Python marshaller would dangle.  This value owns every identity and regenerates one stable view
/// used by prepare and every scientific invocation.  It is never synthesized from process globals.
class PreparedExecutionContextV1 final {
 public:
  PreparedExecutionContextV1(std::string execution_identity, std::uint32_t context_version,
                             PopsMemorySpaceV1 memory_space, std::string backend_identity,
                             std::string device_identity, PopsScalarTypeV1 scalar_type,
                             PopsPrecisionV1 storage_precision, PopsPrecisionV1 compute_precision,
                             PopsPrecisionV1 accumulation_precision,
                             PopsPrecisionV1 reduction_precision, std::uint64_t stream_handle,
                             std::string stream_identity, std::int64_t communicator_f_handle,
                             std::int64_t communicator_datatype_f_handle,
                             std::string communicator_identity,
                             std::string communicator_datatype_identity)
      : execution_identity_(std::move(execution_identity)),
        context_version_(context_version),
        memory_space_(memory_space),
        backend_identity_(std::move(backend_identity)),
        device_identity_(std::move(device_identity)),
        scalar_type_(scalar_type),
        storage_precision_(storage_precision),
        compute_precision_(compute_precision),
        accumulation_precision_(accumulation_precision),
        reduction_precision_(reduction_precision),
        stream_handle_(stream_handle),
        stream_identity_(std::move(stream_identity)),
        communicator_f_handle_(communicator_f_handle),
        communicator_datatype_f_handle_(communicator_datatype_f_handle),
        communicator_identity_(std::move(communicator_identity)),
        communicator_datatype_identity_(std::move(communicator_datatype_identity)) {
    if (execution_identity_.empty())
      throw std::invalid_argument(
          "component execution context requires the exact RuntimeInstance identity");
    component::validate_execution_context(view());
  }

  [[nodiscard]] const std::string& identity() const noexcept { return execution_identity_; }

  /// Remove every communicator and datatype handle while retaining the exact device/stream.
  ///
  /// This is not a serial communicator.  It is an explicitly noncollective execution authority
  /// for callbacks invoked independently per local patch; all MPI consensus remains in PoPS.
  [[nodiscard]] PreparedExecutionContextV1 without_collective_authority() const {
    return PreparedExecutionContextV1(
        execution_identity_ + "/noncollective", context_version_, memory_space_,
        backend_identity_, device_identity_, scalar_type_, storage_precision_, compute_precision_,
        accumulation_precision_, reduction_precision_, stream_handle_, stream_identity_, 0, 0,
        POPS_EXECUTION_NONCOLLECTIVE_IDENTITY_V1, "none");
  }

  /// Derive the exact ABI execution authority for one materialized native lane.
  ///
  /// The RuntimeInstance identity, precision policy, device and stream remain unchanged. Only the
  /// communicator authority is replaced, using the lane's real C/Fortran MPI handle pair rather
  /// than a guessed process-global identity. In a serial build the canonical all-zero/"serial"
  /// representation is retained.
  [[nodiscard]] PreparedExecutionContextV1 for_lane(const ExecutionLane& lane) const {
#ifdef POPS_HAS_MPI
    if (!lane.active() || lane.identity().empty() || lane.identity() == "serial")
      throw std::invalid_argument(
          "component execution lane requires an active non-serial MPI authority");
    return PreparedExecutionContextV1(
        execution_identity_, context_version_, memory_space_, backend_identity_, device_identity_,
        scalar_type_, storage_precision_, compute_precision_, accumulation_precision_,
        reduction_precision_, stream_handle_, stream_identity_,
        static_cast<std::int64_t>(MPI_Comm_c2f(lane.native_handle())),
        static_cast<std::int64_t>(MPI_Type_c2f(MPI_DOUBLE)), std::string(lane.identity()),
        "MPI_DOUBLE");
#else
    (void)lane;
    return PreparedExecutionContextV1(
        execution_identity_, context_version_, memory_space_, backend_identity_, device_identity_,
        scalar_type_, storage_precision_, compute_precision_, accumulation_precision_,
        reduction_precision_, stream_handle_, stream_identity_, 0, 0, "serial", "none");
#endif
  }

  /// Authenticate that this owned context was derived from the exact lane, not merely from a
  /// communicator with the same rank set or a colliding textual label.
  [[nodiscard]] bool matches_lane(const ExecutionLane& lane) const {
#ifdef POPS_HAS_MPI
    if (!lane.active() || communicator_identity_ != lane.identity() ||
        communicator_datatype_identity_ != "MPI_DOUBLE")
      return false;
    int relation = MPI_UNEQUAL;
    ::pops::detail::require_mpi_success(
        MPI_Comm_compare(MPI_Comm_f2c(static_cast<MPI_Fint>(communicator_f_handle_)),
                         lane.native_handle(), &relation),
        "MPI_Comm_compare(component execution lane)");
    return relation == MPI_IDENT &&
           MPI_Type_f2c(static_cast<MPI_Fint>(communicator_datatype_f_handle_)) == MPI_DOUBLE;
#else
    (void)lane;
    return communicator_f_handle_ == 0 && communicator_datatype_f_handle_ == 0 &&
           communicator_identity_ == "serial" && communicator_datatype_identity_ == "none";
#endif
  }

  [[nodiscard]] PopsExecutionContextV1 view() const noexcept {
    return {sizeof(PopsExecutionContextV1),
            context_version_,
            execution_identity_.c_str(),
            memory_space_,
            backend_identity_.c_str(),
            device_identity_.c_str(),
            scalar_type_,
            storage_precision_,
            compute_precision_,
            accumulation_precision_,
            reduction_precision_,
            stream_handle_,
            stream_identity_.c_str(),
            communicator_f_handle_,
            communicator_datatype_f_handle_,
            communicator_identity_.c_str(),
            communicator_datatype_identity_.c_str()};
  }

 private:
  std::string execution_identity_;
  std::uint32_t context_version_ = 0;
  PopsMemorySpaceV1 memory_space_{};
  std::string backend_identity_;
  std::string device_identity_;
  PopsScalarTypeV1 scalar_type_{};
  PopsPrecisionV1 storage_precision_{};
  PopsPrecisionV1 compute_precision_{};
  PopsPrecisionV1 accumulation_precision_{};
  PopsPrecisionV1 reduction_precision_{};
  std::uint64_t stream_handle_ = 0;
  std::string stream_identity_;
  std::int64_t communicator_f_handle_ = 0;
  std::int64_t communicator_datatype_f_handle_ = 0;
  std::string communicator_identity_;
  std::string communicator_datatype_identity_;
};

}  // namespace pops::component
