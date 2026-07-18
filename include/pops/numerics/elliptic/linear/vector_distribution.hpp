#pragma once

/// @file
/// @brief Authenticated provider protocol for one prepared linear-vector distribution.

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/layout/field_distribution.hpp>
#include <pops/mesh/storage/field_replica_consensus.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/parallel/field_distribution_reduction.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

/// Provider-owned outcome returned only after a distribution callback has completed its complete
/// collective trace. A source must return the same status (including success) on every rank and
/// keep `reason` backed by static storage. The callback itself owns that consensus so the handle can
/// throw uniformly without adding a second reduction to every Krylov dot/norm or permitting an
/// exception to split MPI ordering inside an extension callback.
struct PreparedVectorDistributionStatus {
  std::uint64_t code = 0;
  std::string_view reason{};

  [[nodiscard]] constexpr bool accepted() const noexcept { return code == 0; }
  [[nodiscard]] static constexpr PreparedVectorDistributionStatus success() noexcept { return {}; }
  [[nodiscard]] static constexpr PreparedVectorDistributionStatus failure(
      std::uint64_t code, std::string_view reason) noexcept {
    return {code, reason};
  }
};

/// Extension protocol for one physical vector-distribution strategy.  A provider owns layout
/// authentication, canonical layout identity, exact-value preflight, scientific reduction and all
/// scratch requirements. Krylov therefore never switches on a closed distribution enum. Source
/// objects are immutable and shared by concurrent workspace lanes; callbacks must be reentrant and
/// keep mutable state exclusively in the caller-provided scratch spans. Collective callbacks are
/// noexcept and must complete one identical trace and status on every lane rank.
template <class Source>
concept PreparedVectorDistributionSource =
    std::copy_constructible<std::remove_cvref_t<Source>> &&
    requires(const std::remove_cvref_t<Source>& source, ExactContractBuilder& contract,
             const MultiFab& field, std::span<double> values, std::span<double> reduction_scratch,
             std::span<char> validation_scratch, const char* where, const ExecutionLane& lane) {
      {
        std::remove_cvref_t<Source>::provider_identity()
      } noexcept -> std::same_as<PreparedProviderIdentity>;
      { source.serialize_exact_parameters(contract) } -> std::same_as<void>;
      { source.layout_matches(field) } -> std::same_as<bool>;
      { source.layout_contract(field) } -> std::same_as<std::string>;
      { source.reduction_scratch_value_count(values.size()) } noexcept -> std::same_as<std::size_t>;
      { source.validation_scratch_byte_count() } noexcept -> std::same_as<std::size_t>;
      {
        source.reduce_sum_values(values, reduction_scratch, where, lane)
      } noexcept -> std::same_as<PreparedVectorDistributionStatus>;
      {
        source.reduce_max_values(values, reduction_scratch, where, lane)
      } noexcept -> std::same_as<PreparedVectorDistributionStatus>;
      {
        source.require_exact_values(field, validation_scratch, where, lane)
      } noexcept -> std::same_as<PreparedVectorDistributionStatus>;
    };

namespace detail {

struct NativeFieldDistributionSource {
  FieldDistribution distribution = FieldDistribution::Distributed;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.vector-distribution.native-field", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(distribution);
  }

  [[nodiscard]] bool layout_matches(const MultiFab& field) const {
    return field_distribution_layout_matches(field, distribution);
  }

  [[nodiscard]] std::string layout_contract(const MultiFab& field) const {
    return field_distribution_layout_contract(field, distribution);
  }

  [[nodiscard]] std::size_t reduction_scratch_value_count(std::size_t value_count) const noexcept {
    return distribution == FieldDistribution::Replicated
               ? field_distribution_consensus_storage_size(value_count)
               : 0u;
  }

  [[nodiscard]] std::size_t validation_scratch_byte_count() const noexcept {
    return distribution == FieldDistribution::Replicated ? field_replica_consensus_storage_size()
                                                         : 0u;
  }

  PreparedVectorDistributionStatus reduce_sum_values(std::span<double> values,
                                                     std::span<double> scratch, const char*,
                                                     const ExecutionLane& lane) const noexcept {
    try {
      if (values.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return PreparedVectorDistributionStatus::failure(
            1, "native distribution reduction exceeds MPI count capacity");
      if (distribution == FieldDistribution::Distributed) {
        all_reduce_sum_inplace(values.data(), static_cast<int>(values.size()), lane);
        return PreparedVectorDistributionStatus::success();
      }
      const FieldDistributionReductionStatus status = reduce_replicated_field_values_inplace(
          values.data(), values.size(), scratch.data(), scratch.size(), lane.communicator());
      if (status == FieldDistributionReductionStatus::NonfiniteReplica) {
        std::fill(values.begin(), values.end(), std::numeric_limits<double>::quiet_NaN());
        return PreparedVectorDistributionStatus::success();
      }
      if (status == FieldDistributionReductionStatus::InconsistentReplica)
        return PreparedVectorDistributionStatus::failure(
            2, "replicated vector reduction received inconsistent rank values");
      return PreparedVectorDistributionStatus::success();
    } catch (...) {
      return PreparedVectorDistributionStatus::failure(
          3, "native vector sum reduction failed collectively");
    }
  }

  PreparedVectorDistributionStatus reduce_max_values(std::span<double> values,
                                                     std::span<double> scratch, const char* where,
                                                     const ExecutionLane& lane) const noexcept {
    if (distribution != FieldDistribution::Distributed)
      return reduce_sum_values(values, scratch, where, lane);
    try {
      if (values.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return PreparedVectorDistributionStatus::failure(
            1, "native distribution reduction exceeds MPI count capacity");
      all_reduce_max_inplace(values.data(), static_cast<int>(values.size()), lane);
      return PreparedVectorDistributionStatus::success();
    } catch (...) {
      return PreparedVectorDistributionStatus::failure(
          4, "native vector maximum reduction failed collectively");
    }
  }

  PreparedVectorDistributionStatus require_exact_values(const MultiFab& field,
                                                        std::span<char> scratch, const char* where,
                                                        const ExecutionLane& lane) const noexcept {
    if (distribution != FieldDistribution::Replicated)
      return PreparedVectorDistributionStatus::success();
    try {
      require_exact_replicated_field_values_prevalidated(field, scratch.data(), scratch.size(),
                                                         where, lane.communicator());
      return PreparedVectorDistributionStatus::success();
    } catch (...) {
      return PreparedVectorDistributionStatus::failure(
          5, "replicated vector values failed exact collective validation");
    }
  }
};

}  // namespace detail

/// Immutable, type-erased provider handle.  Builtin distributed/replicated values are presets of
/// the same public extension protocol and not cases known by Krylov.
class PreparedVectorDistribution {
 public:
  PreparedVectorDistribution(const PreparedVectorDistribution&) = default;
  PreparedVectorDistribution(PreparedVectorDistribution&&) noexcept = default;
  PreparedVectorDistribution& operator=(const PreparedVectorDistribution&) = default;
  PreparedVectorDistribution& operator=(PreparedVectorDistribution&&) noexcept = default;

  template <class Source>
    requires(!std::same_as<std::remove_cvref_t<Source>, PreparedVectorDistribution> &&
             PreparedVectorDistributionSource<Source>)
  explicit PreparedVectorDistribution(Source source)
      : implementation_(std::make_shared<Model<std::remove_cvref_t<Source>>>(std::move(source))) {}

  /// Bridge from the mesh storage descriptor into the prepared provider protocol.  Numerical
  /// consumers retain only this authenticated handle; extension providers are not limited to it.
  explicit PreparedVectorDistribution(FieldDistribution distribution)
      : PreparedVectorDistribution(detail::NativeFieldDistributionSource{distribution}) {
    if (!field_distribution_is_valid(distribution))
      throw std::invalid_argument("invalid native field distribution");
  }

  PreparedVectorDistribution()
      : PreparedVectorDistribution(
            detail::NativeFieldDistributionSource{FieldDistribution::Distributed}) {}

  [[nodiscard]] static PreparedVectorDistribution distributed() {
    return PreparedVectorDistribution(
        detail::NativeFieldDistributionSource{FieldDistribution::Distributed});
  }

  [[nodiscard]] static PreparedVectorDistribution replicated() {
    return PreparedVectorDistribution(
        detail::NativeFieldDistributionSource{FieldDistribution::Replicated});
  }

  [[nodiscard]] PreparedProviderIdentity provider_identity() const noexcept {
    return implementation_->provider_identity();
  }

  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return implementation_->collective_contract();
  }

  [[nodiscard]] bool layout_matches(const MultiFab& field) const {
    return implementation_->layout_matches(field);
  }

  [[nodiscard]] std::string layout_contract(const MultiFab& field) const {
    return implementation_->layout_contract(field);
  }

  [[nodiscard]] std::size_t reduction_scratch_value_count(std::size_t value_count) const noexcept {
    return implementation_->reduction_scratch_value_count(value_count);
  }

  [[nodiscard]] std::size_t validation_scratch_byte_count() const noexcept {
    return implementation_->validation_scratch_byte_count();
  }

  void reduce_sum_values(std::span<double> values, std::span<double> scratch, const char* where,
                         const ExecutionLane& lane) const {
    const PreparedVectorDistributionStatus status =
        implementation_->reduce_sum_values(values, scratch, where, lane);
    require_callback_success_(status, where);
  }
  void reduce_sum_values(std::span<double> values, std::span<double> scratch,
                         const char* where) const {
    const ExecutionLane lane = ExecutionLane::world();
    reduce_sum_values(values, scratch, where, lane);
  }

  void reduce_max_values(std::span<double> values, std::span<double> scratch, const char* where,
                         const ExecutionLane& lane) const {
    const PreparedVectorDistributionStatus status =
        implementation_->reduce_max_values(values, scratch, where, lane);
    require_callback_success_(status, where);
  }
  void reduce_max_values(std::span<double> values, std::span<double> scratch,
                         const char* where) const {
    const ExecutionLane lane = ExecutionLane::world();
    reduce_max_values(values, scratch, where, lane);
  }

  void require_collective_layout(const MultiFab& field, const char* where,
                                 const ExecutionLane& lane) const {
    const PreparedProviderIdentity identity = provider_identity();
    bool matches = false;
    long callback_failure_local = 0;
    try {
      matches = layout_matches(field);
    } catch (...) {
      callback_failure_local = 1;
    }
    const long callback_failure = all_reduce_max(callback_failure_local, lane);
    if (callback_failure != 0)
      throw std::invalid_argument(std::string(where) +
                                  ": vector-distribution layout predicate failed on at least "
                                  "one communicator rank for provider '" +
                                  std::string(identity.name) + "'");
    const long invalid = all_reduce_max(matches ? 0L : 1L, lane);
    if (invalid != 0)
      throw std::invalid_argument(std::string(where) + ": vector layout rejected by provider '" +
                                  std::string(identity.name) + "'");

    std::string local_layout_contract;
    callback_failure_local = 0;
    try {
      local_layout_contract = layout_contract(field);
    } catch (...) {
      callback_failure_local = 1;
    }
    const long contract_failure = all_reduce_max(callback_failure_local, lane);
    if (contract_failure != 0)
      throw std::invalid_argument(std::string(where) +
                                  ": vector-distribution layout contract failed on at least "
                                  "one communicator rank for provider '" +
                                  std::string(identity.name) + "'");
    const bool agrees = all_ranks_agree_exact_ordered_byte_pairs(
        {{std::string_view("prepared-vector-distribution"), collective_contract()},
         {std::string_view("prepared-vector-layout"), local_layout_contract}},
        lane);
    if (!agrees)
      throw std::invalid_argument(std::string(where) +
                                  ": vector distribution contract differs between ranks");
  }

  void require_collective_layout(const MultiFab& field, const char* where) const {
    const ExecutionLane lane = ExecutionLane::world();
    require_collective_layout(field, where, lane);
  }

  void require_exact_values(const MultiFab& field, std::span<char> scratch, const char* where,
                            const ExecutionLane& lane) const {
    const PreparedVectorDistributionStatus status =
        implementation_->require_exact_values(field, scratch, where, lane);
    require_callback_success_(status, where);
  }
  void require_exact_values(const MultiFab& field, std::span<char> scratch,
                            const char* where) const {
    const ExecutionLane lane = ExecutionLane::world();
    require_exact_values(field, scratch, where, lane);
  }

  friend bool operator==(const PreparedVectorDistribution& left,
                         const PreparedVectorDistribution& right) noexcept {
    return left.collective_contract() == right.collective_contract();
  }

  friend bool operator!=(const PreparedVectorDistribution& left,
                         const PreparedVectorDistribution& right) noexcept {
    return !(left == right);
  }

  static const PreparedVectorDistribution Distributed;
  static const PreparedVectorDistribution Replicated;

 private:
  struct Concept {
    virtual ~Concept() = default;
    [[nodiscard]] virtual PreparedProviderIdentity provider_identity() const noexcept = 0;
    [[nodiscard]] virtual std::string_view collective_contract() const noexcept = 0;
    [[nodiscard]] virtual bool layout_matches(const MultiFab&) const = 0;
    [[nodiscard]] virtual std::string layout_contract(const MultiFab&) const = 0;
    [[nodiscard]] virtual std::size_t reduction_scratch_value_count(std::size_t) const noexcept = 0;
    [[nodiscard]] virtual std::size_t validation_scratch_byte_count() const noexcept = 0;
    virtual PreparedVectorDistributionStatus reduce_sum_values(
        std::span<double>, std::span<double>, const char*, const ExecutionLane&) const noexcept = 0;
    virtual PreparedVectorDistributionStatus reduce_max_values(
        std::span<double>, std::span<double>, const char*, const ExecutionLane&) const noexcept = 0;
    virtual PreparedVectorDistributionStatus require_exact_values(
        const MultiFab&, std::span<char>, const char*, const ExecutionLane&) const noexcept = 0;
  };

  template <PreparedVectorDistributionSource Source>
  class Model final : public Concept {
   public:
    explicit Model(Source source) : source_(std::move(source)) {
      const PreparedProviderIdentity identity = Source::provider_identity();
      if (identity.name.empty() || identity.version == 0)
        throw std::invalid_argument("prepared vector distribution identity is invalid");
      ExactContractBuilder parameters;
      source_.serialize_exact_parameters(parameters);
      ExactContractBuilder contract;
      contract.text("pops.prepared-vector-distribution")
          .scalar(std::uint32_t{2})
          .text(identity.name)
          .scalar(identity.version)
          .bytes(parameters.view());
      collective_contract_ = std::move(contract).release();
    }

    PreparedProviderIdentity provider_identity() const noexcept override {
      return Source::provider_identity();
    }
    std::string_view collective_contract() const noexcept override { return collective_contract_; }
    bool layout_matches(const MultiFab& field) const override {
      return source_.layout_matches(field);
    }
    std::string layout_contract(const MultiFab& field) const override {
      return source_.layout_contract(field);
    }
    std::size_t reduction_scratch_value_count(std::size_t count) const noexcept override {
      return source_.reduction_scratch_value_count(count);
    }
    std::size_t validation_scratch_byte_count() const noexcept override {
      return source_.validation_scratch_byte_count();
    }
    PreparedVectorDistributionStatus reduce_sum_values(
        std::span<double> values, std::span<double> scratch, const char* where,
        const ExecutionLane& lane) const noexcept override {
      return source_.reduce_sum_values(values, scratch, where, lane);
    }
    PreparedVectorDistributionStatus reduce_max_values(
        std::span<double> values, std::span<double> scratch, const char* where,
        const ExecutionLane& lane) const noexcept override {
      return source_.reduce_max_values(values, scratch, where, lane);
    }
    PreparedVectorDistributionStatus require_exact_values(
        const MultiFab& field, std::span<char> scratch, const char* where,
        const ExecutionLane& lane) const noexcept override {
      return source_.require_exact_values(field, scratch, where, lane);
    }

   private:
    Source source_;
    std::string collective_contract_;
  };

  std::shared_ptr<const Concept> implementation_;

  static void require_callback_success_(const PreparedVectorDistributionStatus& status,
                                        const char* where) {
    // The source protocol requires status uniformity as part of the callback's own collective
    // trace. Re-checking it here would add a second MPI reduction to every Krylov dot/norm. The
    // handle's responsibility is to make the already-uniform provider failure observable rather
    // than silently accepting it.
    if (status.accepted())
      return;
    const std::string_view reason =
        status.reason.empty() ? std::string_view("provider callback failed") : status.reason;
    throw std::runtime_error(std::string(where) + ": " + std::string(reason) +
                             " (provider status " + std::to_string(status.code) + ")");
  }
};

inline const PreparedVectorDistribution PreparedVectorDistribution::Distributed =
    PreparedVectorDistribution::distributed();
inline const PreparedVectorDistribution PreparedVectorDistribution::Replicated =
    PreparedVectorDistribution::replicated();

inline bool field_distribution_is_valid(const PreparedVectorDistribution&) noexcept {
  return true;
}

namespace detail {

inline std::string field_distribution_layout_contract(
    const MultiFab& field, const PreparedVectorDistribution& distribution) {
  return distribution.layout_contract(field);
}

inline bool field_distribution_layout_matches(const MultiFab& field,
                                              const PreparedVectorDistribution& distribution) {
  return distribution.layout_matches(field);
}

inline void require_collective_field_distribution_layout(
    const MultiFab& field, const PreparedVectorDistribution& distribution, const char* where) {
  distribution.require_collective_layout(field, where);
}

inline void require_collective_field_distribution_layout(
    const MultiFab& field, const PreparedVectorDistribution& distribution, const char* where,
    const ExecutionLane& lane) {
  distribution.require_collective_layout(field, where, lane);
}

inline void reduce_prepared_vector_values_inplace(const PreparedVectorDistribution& distribution,
                                                  double* values, int count, double* scratch,
                                                  std::size_t scratch_count, const char* quantity,
                                                  const ExecutionLane& lane) {
  if (count < 0)
    throw std::invalid_argument(std::string(quantity) + " has a negative reduction count");
  distribution.reduce_sum_values(std::span<double>(values, static_cast<std::size_t>(count)),
                                 std::span<double>(scratch, scratch_count), quantity, lane);
}

}  // namespace detail
}  // namespace pops
