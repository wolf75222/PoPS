#pragma once

/// @file
/// @brief Exact, typed metric protocol for prepared linear vector spaces.

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/numerics/elliptic/linear/pure_field_algebra.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

namespace detail {

inline std::string prepared_vector_metric_layout_contract(
    const MultiFab& prototype, const PreparedVectorDistribution& distribution) {
  if (!distribution.layout_matches(prototype))
    throw std::invalid_argument("prepared vector metric received an incoherent field layout");

  ExactContractBuilder contract;
  contract.text("pops.prepared-vector-space")
      .scalar(std::uint32_t{1})
      .bytes(distribution.collective_contract())
      .bytes(distribution.layout_contract(prototype))
      .scalar(static_cast<std::int32_t>(prototype.ncomp()))
      .scalar(static_cast<std::int32_t>(prototype.n_grow()));
  return std::move(contract).release();
}

template <class Operation>
Real prepared_metric_value_noexcept(Operation&& operation) noexcept {
  try {
    return std::forward<Operation>(operation)();
  } catch (...) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
}

template <class Operation>
void prepared_metric_payload_noexcept(std::span<double> payload, Operation&& operation) noexcept {
  try {
    std::forward<Operation>(operation)();
  } catch (...) {
    std::fill(payload.begin(), payload.end(), std::numeric_limits<double>::quiet_NaN());
  }
}

}  // namespace detail

/// Typed extension protocol for one prepared vector metric.
///
/// A source owns a stable identity, serializes every resolved parameter, and implements the global
/// and local forms of the same inner product. The local robust payload is what lets restarted GMRES
/// retain one batched communicator reduction instead of branching on a concrete metric class. Its
/// width is part of the exact contract and may therefore differ for a future composite metric.
/// Every callback is noexcept. Global callbacks must complete one identical collective trace and
/// return the same value (including a non-finite failure witness) on every execution-lane rank;
/// local callbacks publish failure by returning/filling NaN before the common reduction. This is a
/// native provider trust boundary: throwing rank-locally could strand peers inside the next MPI
/// collective and is rejected structurally by the concept. The immutable source may be shared by
/// concurrent workspace lanes, so its callbacks must also be reentrant and keep all mutable scratch
/// in the spans supplied by the caller.
template <class Source>
concept PreparedVectorMetricSource =
    std::copy_constructible<std::remove_cvref_t<Source>> &&
    requires(const std::remove_cvref_t<Source>& source, ExactContractBuilder& contract,
             const MultiFab& left, const MultiFab& right,
             const PreparedVectorDistribution& distribution, std::span<double> local_payload,
             std::span<const double> global_payload, std::span<double> distribution_scratch,
             const ExecutionLane& lane) {
      {
        std::remove_cvref_t<Source>::provider_identity()
      } noexcept -> std::same_as<PreparedProviderIdentity>;
      { source.serialize_exact_parameters(contract) } -> std::same_as<void>;
      { source.robust_payload_width() } noexcept -> std::convertible_to<std::size_t>;
      {
        source.inner_product(left, right, distribution, distribution_scratch, lane)
      } noexcept -> std::same_as<Real>;
      {
        source.absolute_inner_product(left, right, distribution, distribution_scratch, lane)
      } noexcept -> std::same_as<Real>;
      {
        source.nullspace_inner_product(left, right, Real(1), distribution, distribution_scratch,
                                       lane)
      } noexcept -> std::same_as<Real>;
      {
        source.nullspace_absolute_inner_product(left, right, Real(1), distribution,
                                                distribution_scratch, lane)
      } noexcept -> std::same_as<Real>;
      {
        source.norm(left, distribution, distribution_scratch, lane)
      } noexcept -> std::same_as<Real>;
      { source.local_inner_product(left, right) } noexcept -> std::same_as<Real>;
      {
        source.local_robust_inner_product_payload(left, right, local_payload)
      } noexcept -> std::same_as<void>;
      {
        source.inner_product_from_global_robust_payload(global_payload)
      } noexcept -> std::same_as<Real>;
    };

/// Default physical metric. It preserves the existing all-component Euclidean arithmetic exactly,
/// including the scale-safe norm and robust-dot fallback used at extreme exponent ranges.
struct EuclideanPreparedVectorMetricSource {
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.vector-metric.euclidean", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder&) const {}

  [[nodiscard]] constexpr std::size_t robust_payload_width() const noexcept {
    return detail::PreparedFieldAlgebra::kRobustDotPayloadWidth;
  }

  Real inner_product(const MultiFab& left, const MultiFab& right,
                     const PreparedVectorDistribution& distribution,
                     std::span<double> distribution_scratch,
                     const ExecutionLane& lane) const noexcept {
    return detail::prepared_metric_value_noexcept([&] {
      return detail::PreparedFieldAlgebra::dot(left, right, distribution, distribution_scratch,
                                               lane);
    });
  }

  Real norm(const MultiFab& value, const PreparedVectorDistribution& distribution,
            std::span<double> distribution_scratch, const ExecutionLane& lane) const noexcept {
    return detail::prepared_metric_value_noexcept([&] {
      return detail::PreparedFieldAlgebra::norm(value, distribution, distribution_scratch, lane);
    });
  }

  Real absolute_inner_product(const MultiFab& left, const MultiFab& right,
                              const PreparedVectorDistribution& distribution,
                              std::span<double> distribution_scratch,
                              const ExecutionLane& lane) const noexcept {
    return detail::prepared_metric_value_noexcept([&] {
      return detail::PreparedFieldAlgebra::absolute_dot(left, right, distribution,
                                                        distribution_scratch, lane);
    });
  }

  Real nullspace_inner_product(const MultiFab& left, const MultiFab& right, Real cell_measure,
                               const PreparedVectorDistribution& distribution,
                               std::span<double> distribution_scratch,
                               const ExecutionLane& lane) const noexcept {
    return detail::prepared_metric_value_noexcept([&] {
      return detail::PreparedFieldAlgebra::nullspace_pairing(
          left, right, cell_measure, false, distribution, distribution_scratch, lane);
    });
  }

  Real nullspace_absolute_inner_product(const MultiFab& left, const MultiFab& right,
                                        Real cell_measure,
                                        const PreparedVectorDistribution& distribution,
                                        std::span<double> distribution_scratch,
                                        const ExecutionLane& lane) const noexcept {
    return detail::prepared_metric_value_noexcept([&] {
      return detail::PreparedFieldAlgebra::nullspace_pairing(
          left, right, cell_measure, true, distribution, distribution_scratch, lane);
    });
  }

  Real local_inner_product(const MultiFab& left, const MultiFab& right) const noexcept {
    return detail::prepared_metric_value_noexcept(
        [&] { return detail::PreparedFieldAlgebra::local_dot(left, right); });
  }

  void local_robust_inner_product_payload(const MultiFab& left, const MultiFab& right,
                                          std::span<double> payload) const noexcept {
    detail::prepared_metric_payload_noexcept(payload, [&] {
      if (payload.size() != robust_payload_width())
        throw std::invalid_argument("Euclidean vector metric received an invalid robust payload");
      detail::PreparedFieldAlgebra::local_robust_dot_payload(left, right, payload.data());
    });
  }

  Real inner_product_from_global_robust_payload(std::span<const double> payload) const noexcept {
    return detail::prepared_metric_value_noexcept([&] {
      if (payload.size() != robust_payload_width())
        throw std::invalid_argument("Euclidean vector metric received an invalid robust payload");
      return detail::PreparedFieldAlgebra::dot_from_global_robust_payload(payload.data());
    });
  }
};

/// Immutable type-erased metric bound to one exact MultiFab vector space and field distribution.
/// Consumers dispatch only through this protocol; no solver branches on implementation names.
class PreparedVectorMetric {
 public:
  PreparedVectorMetric() = default;

  template <PreparedVectorMetricSource Source>
  PreparedVectorMetric(const MultiFab& prototype, PreparedVectorDistribution distribution,
                       Source source) {
    initialize_(prototype, distribution, std::move(source));
  }

  [[nodiscard]] static PreparedVectorMetric euclidean(
      const MultiFab& prototype,
      PreparedVectorDistribution distribution = PreparedVectorDistribution::Distributed) {
    return PreparedVectorMetric(prototype, distribution, EuclideanPreparedVectorMetricSource{});
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(inner_product_);
  }

  [[nodiscard]] const PreparedVectorDistribution& distribution() const noexcept {
    return distribution_;
  }
  [[nodiscard]] std::size_t robust_payload_width() const noexcept { return robust_payload_width_; }
  [[nodiscard]] std::size_t reduction_scratch_value_count() const noexcept {
    return distribution_.reduction_scratch_value_count(
        std::max(robust_payload_width_, std::size_t{1}));
  }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }

  [[nodiscard]] bool compatible_with(const MultiFab& prototype,
                                     const PreparedVectorDistribution& distribution) const {
    return static_cast<bool>(*this) && distribution == distribution_ &&
           layout_contract_ ==
               detail::prepared_vector_metric_layout_contract(prototype, distribution);
  }

  Real inner_product(const MultiFab& left, const MultiFab& right,
                     std::span<double> distribution_scratch, const ExecutionLane& lane) const {
    require_initialized_();
    require_scratch_(distribution_scratch);
    return inner_product_(left, right, distribution_scratch, lane);
  }
  Real inner_product(const MultiFab& left, const MultiFab& right,
                     std::span<double> distribution_scratch) const {
    const ExecutionLane lane = ExecutionLane::world();
    return inner_product(left, right, distribution_scratch, lane);
  }

  Real norm(const MultiFab& value, std::span<double> distribution_scratch,
            const ExecutionLane& lane) const {
    require_initialized_();
    require_scratch_(distribution_scratch);
    return norm_(value, distribution_scratch, lane);
  }
  Real norm(const MultiFab& value, std::span<double> distribution_scratch) const {
    const ExecutionLane lane = ExecutionLane::world();
    return norm(value, distribution_scratch, lane);
  }

  Real absolute_inner_product(const MultiFab& left, const MultiFab& right,
                              std::span<double> distribution_scratch,
                              const ExecutionLane& lane) const {
    require_initialized_();
    require_scratch_(distribution_scratch);
    return absolute_inner_product_(left, right, distribution_scratch, lane);
  }
  Real absolute_inner_product(const MultiFab& left, const MultiFab& right,
                              std::span<double> distribution_scratch) const {
    const ExecutionLane lane = ExecutionLane::world();
    return absolute_inner_product(left, right, distribution_scratch, lane);
  }

  Real nullspace_inner_product(const MultiFab& left, const MultiFab& right, Real cell_measure,
                               std::span<double> distribution_scratch,
                               const ExecutionLane& lane) const {
    require_initialized_();
    require_scratch_(distribution_scratch);
    return nullspace_inner_product_(left, right, cell_measure, distribution_scratch, lane);
  }
  Real nullspace_inner_product(const MultiFab& left, const MultiFab& right, Real cell_measure,
                               std::span<double> distribution_scratch) const {
    const ExecutionLane lane = ExecutionLane::world();
    return nullspace_inner_product(left, right, cell_measure, distribution_scratch, lane);
  }

  Real nullspace_absolute_inner_product(const MultiFab& left, const MultiFab& right,
                                        Real cell_measure, std::span<double> distribution_scratch,
                                        const ExecutionLane& lane) const {
    require_initialized_();
    require_scratch_(distribution_scratch);
    return nullspace_absolute_inner_product_(left, right, cell_measure, distribution_scratch, lane);
  }
  Real nullspace_absolute_inner_product(const MultiFab& left, const MultiFab& right,
                                        Real cell_measure,
                                        std::span<double> distribution_scratch) const {
    const ExecutionLane lane = ExecutionLane::world();
    return nullspace_absolute_inner_product(left, right, cell_measure, distribution_scratch, lane);
  }

  Real local_inner_product(const MultiFab& left, const MultiFab& right) const {
    require_initialized_();
    return local_inner_product_(left, right);
  }

  void local_robust_inner_product_payload(const MultiFab& left, const MultiFab& right,
                                          std::span<double> payload) const {
    require_initialized_();
    if (payload.size() != robust_payload_width_)
      throw std::invalid_argument("prepared vector metric received an invalid robust payload");
    local_robust_inner_product_payload_(left, right, payload);
  }

  Real inner_product_from_global_robust_payload(std::span<const double> payload) const {
    require_initialized_();
    if (payload.size() != robust_payload_width_)
      throw std::invalid_argument("prepared vector metric received an invalid robust payload");
    return inner_product_from_global_robust_payload_(payload);
  }

 private:
  template <PreparedVectorMetricSource Source>
  void initialize_(const MultiFab& prototype, PreparedVectorDistribution distribution,
                   Source source) {
    using S = std::remove_cvref_t<Source>;
    const PreparedProviderIdentity identity = S::provider_identity();
    if (identity.name.empty())
      throw std::invalid_argument("prepared vector metric identity must not be empty");
    if (identity.version == 0)
      throw std::invalid_argument("prepared vector metric version must be positive");
    const std::size_t width = static_cast<std::size_t>(source.robust_payload_width());
    if (width == 0 || width > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("prepared vector metric robust payload width is invalid");

    ExactContractBuilder parameters;
    source.serialize_exact_parameters(parameters);
    layout_contract_ = detail::prepared_vector_metric_layout_contract(prototype, distribution);
    distribution_ = distribution;
    robust_payload_width_ = width;
    ExactContractBuilder collective;
    collective.text("pops.prepared-vector-metric")
        .scalar(std::uint32_t{2})
        .text(identity.name)
        .scalar(identity.version)
        .bytes(parameters.view())
        .bytes(layout_contract_)
        .scalar(static_cast<std::uint64_t>(robust_payload_width_));
    collective_contract_ = std::move(collective).release();

    inner_product_ = [source, distribution](const MultiFab& left, const MultiFab& right,
                                            std::span<double> scratch, const ExecutionLane& lane) {
      return source.inner_product(left, right, distribution, scratch, lane);
    };
    norm_ = [source, distribution](const MultiFab& value, std::span<double> scratch,
                                   const ExecutionLane& lane) {
      return source.norm(value, distribution, scratch, lane);
    };
    absolute_inner_product_ = [source, distribution](const MultiFab& left, const MultiFab& right,
                                                     std::span<double> scratch,
                                                     const ExecutionLane& lane) {
      return source.absolute_inner_product(left, right, distribution, scratch, lane);
    };
    nullspace_inner_product_ = [source, distribution](const MultiFab& left, const MultiFab& right,
                                                      Real cell_measure, std::span<double> scratch,
                                                      const ExecutionLane& lane) {
      return source.nullspace_inner_product(left, right, cell_measure, distribution, scratch, lane);
    };
    nullspace_absolute_inner_product_ =
        [source, distribution](const MultiFab& left, const MultiFab& right, Real cell_measure,
                               std::span<double> scratch, const ExecutionLane& lane) {
          return source.nullspace_absolute_inner_product(left, right, cell_measure, distribution,
                                                         scratch, lane);
        };
    local_inner_product_ = [source](const MultiFab& left, const MultiFab& right) {
      return source.local_inner_product(left, right);
    };
    local_robust_inner_product_payload_ = [source](const MultiFab& left, const MultiFab& right,
                                                   std::span<double> payload) {
      source.local_robust_inner_product_payload(left, right, payload);
    };
    inner_product_from_global_robust_payload_ = [source](std::span<const double> payload) {
      return source.inner_product_from_global_robust_payload(payload);
    };
  }

  void require_initialized_() const {
    if (!static_cast<bool>(*this))
      throw std::logic_error("prepared vector metric is not initialized");
  }

  void require_scratch_(std::span<double> scratch) const {
    if (scratch.size() < reduction_scratch_value_count())
      throw std::invalid_argument("prepared vector metric reduction scratch is too small");
  }

  using InnerProduct = std::function<Real(const MultiFab&, const MultiFab&, std::span<double>,
                                          const ExecutionLane&)>;
  using LocalInnerProduct = std::function<Real(const MultiFab&, const MultiFab&)>;
  using NullspaceInnerProduct = std::function<Real(const MultiFab&, const MultiFab&, Real,
                                                   std::span<double>, const ExecutionLane&)>;
  using Norm = std::function<Real(const MultiFab&, std::span<double>, const ExecutionLane&)>;
  using RobustPayload = std::function<void(const MultiFab&, const MultiFab&, std::span<double>)>;
  using RobustReconstruction = std::function<Real(std::span<const double>)>;

  PreparedVectorDistribution distribution_ = PreparedVectorDistribution::Distributed;
  std::size_t robust_payload_width_ = 0;
  std::string layout_contract_;
  std::string collective_contract_;
  InnerProduct inner_product_;
  InnerProduct absolute_inner_product_;
  NullspaceInnerProduct nullspace_inner_product_;
  NullspaceInnerProduct nullspace_absolute_inner_product_;
  Norm norm_;
  LocalInnerProduct local_inner_product_;
  RobustPayload local_robust_inner_product_payload_;
  RobustReconstruction inner_product_from_global_robust_payload_;
};

}  // namespace pops
