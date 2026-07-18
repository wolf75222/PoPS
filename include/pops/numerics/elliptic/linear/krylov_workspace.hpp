#pragma once

/// @file
/// @brief Persistent storage for prepared affine Krylov solves.

#include <pops/numerics/elliptic/linear/krylov_method_provider.hpp>
#include <pops/numerics/elliptic/linear/scaled_scalar.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace pops {

namespace detail {
struct KrylovWorkspaceAccess;
}

struct KrylovControls {
  PreparedKrylovMethod method{};
  Real rel_tol = Real(1e-8);
  Real abs_tol = Real(0);
  int max_iterations = 1;
};

class KrylovWorkspace {
 public:
  static int max_batched_basis_extent() {
    return max_krylov_batched_basis_extent(detail::PreparedFieldAlgebra::kRobustDotPayloadWidth);
  }

  static int max_batched_basis_extent(std::size_t robust_payload_width) {
    return max_krylov_batched_basis_extent(robust_payload_width);
  }

  KrylovWorkspace(
      const MultiFab& prototype, PreparedKrylovMethod method, KrylovFootprint footprint,
      PreparedVectorDistribution vector_distribution = PreparedVectorDistribution::Distributed,
      PreparedVectorMetric metric = {})
      : method_(std::move(method)),
        footprint_(footprint),
        vector_distribution_(std::move(vector_distribution)),
        metric_(std::move(metric)) {
    const long footprint_failure_local =
        footprint_.components != prototype.ncomp() ||
                footprint_.input_ghosts != prototype.n_grow()
            ? 1L
            : 0L;
    if (all_reduce_max(footprint_failure_local) != 0)
      throw std::invalid_argument("KrylovWorkspace footprint disagrees with prototype");

    // Layout authentication is a materialization-time provider callback.  Its exact contract and
    // every dynamic consensus allocation are completed here, never lazily inside solve.
    vector_distribution_.require_collective_layout(prototype, "KrylovWorkspace");
    long vector_space_failure_local = 0;
    try {
      layout_ = detail::layout_fingerprint(prototype, vector_distribution_);
      vector_distribution_layout_valid_ =
          detail::field_distribution_layout_matches(prototype, vector_distribution_);
      if (!metric_)
        metric_ = PreparedVectorMetric::euclidean(prototype, vector_distribution_);
      if (!metric_.compatible_with(prototype, vector_distribution_))
        vector_space_failure_local = 1;
    } catch (...) {
      vector_space_failure_local = 1;
    }
    if (all_reduce_max(vector_space_failure_local) != 0)
      throw std::invalid_argument("KrylovWorkspace metric disagrees with the vector space");
    metric_fingerprint_ = detail::fingerprint_seed();
    detail::fingerprint_mix(metric_fingerprint_, metric_.collective_contract());
    distribution_fingerprint_ = detail::fingerprint_seed();
    detail::fingerprint_mix(distribution_fingerprint_,
                            vector_distribution_.collective_contract());
    long provider_failure_local = 0;
    try {
      requirements_ = method_.workspace_requirements(KrylovWorkspaceRequest{
          footprint_, vector_distribution_, metric_.robust_payload_width()});
    } catch (...) {
      provider_failure_local = 1;
    }
    if (all_reduce_max(provider_failure_local) != 0)
      throw std::invalid_argument(
          "prepared Krylov provider failed to produce workspace requirements on at least one "
          "communicator rank");

    ExactContractBuilder requirements_contract;
    append_requirements_contract_(requirements_contract, requirements_);
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("prepared-krylov-workspace"), requirements_contract.view()}}))
      throw std::invalid_argument(
          "prepared Krylov workspace requirements differ between communicator ranks");
    long materialization_failure_local = 0;
    try {
      distribution_validation_data_.assign(
          vector_distribution_.validation_scratch_byte_count(), char{0});
      fields_.reserve(requirements_.field_count);
      for (std::size_t index = 0; index < requirements_.field_count; ++index)
        fields_.emplace_back(prototype.box_array(), prototype.dmap(), prototype.ncomp(),
                             footprint_.input_ghosts);
      for (MultiFab& field : fields_)
        field.share_halo_cache_from(prototype);
      real_values_.assign(requirements_.real_count, Real(0));
      scaled_values_.assign(requirements_.scaled_scalar_count, detail::ScaledScalar::zero());
      collective_values_.assign(requirements_.collective_value_count, 0.0);
      distribution_reduction_data_.assign(
          vector_distribution_.reduction_scratch_value_count(
              std::max(requirements_.reduction_value_capacity, std::size_t{1})),
          0.0);
      state_words_.assign(requirements_.state_word_count, std::uint64_t{0});
    } catch (...) {
      materialization_failure_local = 1;
    }
    if (all_reduce_max(materialization_failure_local) != 0) {
      fields_.clear();
      real_values_.clear();
      scaled_values_.clear();
      collective_values_.clear();
      distribution_reduction_data_.clear();
      distribution_validation_data_.clear();
      state_words_.clear();
      throw std::runtime_error(
          "prepared Krylov persistent workspace materialization failed on at least one "
          "communicator rank");
    }
    allocation_count_ = requirements_.field_count;
  }
  static std::size_t required_fields(const PreparedKrylovMethod& method,
                                     const KrylovWorkspaceRequest& request) {
    return method.workspace_requirements(request).field_count;
  }

  void bind(const PreparedAffineLinearProblem& problem) {
    detail::KrylovCollectivePayload payload;
    long local_failure = detail::PreparedProblemAccess::append_collective_state(problem, payload);
    append_collective_state_(payload);
    if (local_failure == 0 && !vector_distribution_layout_valid_)
      local_failure = 8;
    if (local_failure == 0 &&
        (problem.layout_fingerprint() != layout_ || problem.footprint() != footprint_ ||
         problem.vector_distribution() != vector_distribution_))
      local_failure = 4;
    if (local_failure == 0 &&
        (detail::PreparedProblemAccess::metric_fingerprint(problem) != metric_fingerprint_ ||
         problem.metric().collective_contract() != metric_.collective_contract()))
      local_failure = 9;
    if (local_failure == 0 && problem.has_preconditioner() != footprint_.preconditioned)
      local_failure = 5;
    const bool agrees = detail::collective_payload_agrees(payload);
    const long collective_failure = all_reduce_max(local_failure);
    throw_collective_failure_(collective_failure);
    if (!agrees)
      throw std::logic_error("KrylovWorkspace bind contract differs across communicator ranks");
    snapshot_ = detail::PreparedProblemAccess::stored_snapshot(problem);
  }

  void require_bound(const PreparedAffineLinearProblem& problem,
                     const KrylovControls& controls) const {
    detail::KrylovCollectivePayload payload;
    long local_failure = detail::PreparedProblemAccess::append_collective_state(problem, payload);
    append_collective_state_(payload);
    append_controls_(payload, controls);
    const auto& problem_snapshot = detail::PreparedProblemAccess::stored_snapshot(problem);
    if (local_failure == 0 && (!snapshot_ || !problem_snapshot || *snapshot_ != *problem_snapshot))
      local_failure = 6;
    if (local_failure == 0 && !(controls.method == method_))
      local_failure = 7;
    const bool agrees = detail::collective_payload_agrees(payload);
    const long collective_failure = all_reduce_max(local_failure);
    throw_collective_failure_(collective_failure);
    if (!agrees)
      throw std::logic_error("KrylovWorkspace bound contract differs across communicator ranks");
  }

  const PreparedKrylovMethod& method() const { return method_; }
  const PreparedVectorDistribution& vector_distribution() const { return vector_distribution_; }
  const PreparedVectorMetric& metric() const { return metric_; }
  const KrylovFootprint& footprint() const { return footprint_; }
  /// Number of persistent MultiFab work vectors (not heap-allocation events).
  std::size_t allocation_count() const { return allocation_count_; }
  std::size_t scalar_value_count() const {
    return real_values_.size() + scaled_values_.size();
  }
  std::size_t collective_value_count() const {
    return collective_values_.size() + distribution_reduction_data_.size();
  }
  std::size_t distribution_validation_byte_count() const {
    return distribution_validation_data_.size();
  }

 private:
  friend struct detail::KrylovWorkspaceAccess;
  friend class PreparedKrylovSolveContext;

  static void append_footprint_(detail::KrylovCollectivePayload& payload,
                                const KrylovFootprint& footprint) noexcept {
    payload.append(footprint.components);
    payload.append(footprint.input_ghosts);
    payload.append(static_cast<std::uint8_t>(footprint.preconditioned));
  }

  static void append_controls_(detail::KrylovCollectivePayload& payload,
                               const KrylovControls& controls) noexcept {
    payload.append(controls.method.fingerprint());
    payload.append(std::bit_cast<std::uint64_t>(controls.rel_tol));
    payload.append(std::bit_cast<std::uint64_t>(controls.abs_tol));
    payload.append(controls.max_iterations);
  }

  void append_collective_state_(detail::KrylovCollectivePayload& payload) const noexcept {
    payload.append(method_.fingerprint());
    append_footprint_(payload, footprint_);
    append_requirements_(payload, requirements_);
    payload.append(layout_);
    payload.append(distribution_fingerprint_);
    payload.append(static_cast<std::uint8_t>(vector_distribution_layout_valid_));
    payload.append(static_cast<std::uint8_t>(snapshot_.has_value()));
    payload.append(snapshot_.value_or(OperatorEvaluationSnapshot{}));
    payload.append(metric_fingerprint_);
    payload.append(static_cast<std::uint64_t>(metric_.robust_payload_width()));
  }

  static void append_requirements_(detail::KrylovCollectivePayload& payload,
                                   const KrylovWorkspaceRequirements& requirements) noexcept {
    payload.append(static_cast<std::uint64_t>(requirements.field_count));
    payload.append(static_cast<std::uint64_t>(requirements.real_count));
    payload.append(static_cast<std::uint64_t>(requirements.scaled_scalar_count));
    payload.append(static_cast<std::uint64_t>(requirements.collective_value_count));
    payload.append(static_cast<std::uint64_t>(requirements.reduction_value_capacity));
    payload.append(static_cast<std::uint64_t>(requirements.state_word_count));
    payload.append(static_cast<std::uint64_t>(requirements.initial_residual_field));
  }

  static void append_requirements_contract_(
      ExactContractBuilder& contract,
      const KrylovWorkspaceRequirements& requirements) {
    contract.scalar(static_cast<std::uint64_t>(requirements.field_count))
        .scalar(static_cast<std::uint64_t>(requirements.real_count))
        .scalar(static_cast<std::uint64_t>(requirements.scaled_scalar_count))
        .scalar(static_cast<std::uint64_t>(requirements.collective_value_count))
        .scalar(static_cast<std::uint64_t>(requirements.reduction_value_capacity))
        .scalar(static_cast<std::uint64_t>(requirements.state_word_count))
        .scalar(static_cast<std::uint64_t>(requirements.initial_residual_field));
  }

  static void throw_collective_failure_(long failure) {
    if (failure == 0)
      return;
    if (failure == 1)
      throw std::logic_error(
          "operator snapshot mutated after preparation on at least one communicator rank");
    if (failure == 2)
      throw std::logic_error("operator snapshot probe failed on at least one communicator rank");
    if (failure == 3)
      throw std::logic_error(
          "PreparedAffineLinearProblem is not prepared on every communicator rank");
    if (failure == 4)
      throw std::invalid_argument("KrylovWorkspace is incompatible with prepared problem");
    if (failure == 5)
      throw std::invalid_argument("KrylovWorkspace preconditioner footprint mismatch");
    if (failure == 6)
      throw std::logic_error("KrylovWorkspace snapshot is not bound to prepared problem");
    if (failure == 8)
      throw std::invalid_argument(
          "KrylovWorkspace vector layout was rejected by its distribution provider");
    if (failure == 9)
      throw std::invalid_argument("KrylovWorkspace metric is incompatible with prepared problem");
    throw std::invalid_argument("KrylovWorkspace method/restart mismatch");
  }

  // Work storage is deliberately not a public extension seam. Replacing even one iso-layout field
  // could discard its warmed halo/MPI buffers and reintroduce allocation inside an iteration; all
  // algorithms reach these stable slots through the private detail access object instead.
  MultiFab& field(std::size_t index) {
    if (index >= fields_.size())
      throw std::out_of_range("KrylovWorkspace field index");
    return fields_[index];
  }
  const MultiFab& field(std::size_t index) const {
    if (index >= fields_.size())
      throw std::out_of_range("KrylovWorkspace field index");
    return fields_[index];
  }

  Real& real_value(std::size_t index) { return real_values_.at(index); }
  detail::ScaledScalar& scaled_value(std::size_t index) { return scaled_values_.at(index); }
  double* collective_data() { return collective_values_.data(); }
  std::size_t collective_data_size() const { return collective_values_.size(); }
  std::uint64_t& state_word(std::size_t index) { return state_words_.at(index); }
  double* distribution_reduction_data() { return distribution_reduction_data_.data(); }
  std::size_t distribution_reduction_size() const {
    return distribution_reduction_data_.size();
  }
  char* distribution_validation_data() { return distribution_validation_data_.data(); }
  std::size_t distribution_validation_size() const {
    return distribution_validation_data_.size();
  }
  const KrylovWorkspaceRequirements& requirements() const { return requirements_; }
  std::size_t metric_robust_payload_width() const { return metric_.robust_payload_width(); }

  bool provider_report_reason_agrees_(std::string_view reason) {
    constexpr std::size_t kLengthBytes = sizeof(std::uint64_t);
    constexpr std::size_t kOverflowBytes = 1;
    constexpr std::size_t kPayloadCapacity =
        kProviderReportReasonConsensusCapacity - kLengthBytes - kOverflowBytes;
    const bool overflow = reason.size() > kPayloadCapacity;
    const std::size_t copied = std::min(reason.size(), kPayloadCapacity);

    provider_report_reason_min_.fill(char{0});
    const std::uint64_t length = static_cast<std::uint64_t>(reason.size());
    for (std::size_t byte = 0; byte < kLengthBytes; ++byte)
      provider_report_reason_min_[byte] =
          static_cast<char>((length >> (8u * byte)) & std::uint64_t{0xff});
    if (copied != 0)
      std::copy_n(reason.data(), copied,
                  provider_report_reason_min_.data() + kLengthBytes);
    provider_report_reason_min_.back() = overflow ? char{1} : char{0};
    provider_report_reason_max_ = provider_report_reason_min_;
    all_reduce_min_inplace(provider_report_reason_min_.data(),
                           provider_report_reason_min_.size());
    all_reduce_max_inplace(provider_report_reason_max_.data(),
                           provider_report_reason_max_.size());
    return provider_report_reason_min_.back() == char{0} &&
           provider_report_reason_min_ == provider_report_reason_max_;
  }

  PreparedKrylovMethod method_;
  KrylovFootprint footprint_;
  KrylovWorkspaceRequirements requirements_{};
  OperatorFingerprint layout_{};
  PreparedVectorDistribution vector_distribution_ = PreparedVectorDistribution::Distributed;
  OperatorFingerprint distribution_fingerprint_{};
  PreparedVectorMetric metric_;
  OperatorFingerprint metric_fingerprint_{};
  bool vector_distribution_layout_valid_ = true;
  std::vector<MultiFab> fields_;
  std::size_t allocation_count_ = 0;
  std::optional<OperatorEvaluationSnapshot> snapshot_{};
  std::vector<Real> real_values_;
  std::vector<detail::ScaledScalar> scaled_values_;
  std::vector<double> collective_values_;
  std::vector<double> distribution_reduction_data_;
  std::vector<char, comm_allocator<char>> distribution_validation_data_;
  std::vector<std::uint64_t> state_words_;
  static constexpr std::size_t kProviderReportReasonConsensusCapacity = 4096;
  std::array<char, kProviderReportReasonConsensusCapacity> provider_report_reason_min_{};
  std::array<char, kProviderReportReasonConsensusCapacity> provider_report_reason_max_{};
};

}  // namespace pops
