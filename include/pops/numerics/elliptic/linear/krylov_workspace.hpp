#pragma once

/// @file
/// @brief Persistent storage for prepared affine Krylov solves.

#include <pops/numerics/elliptic/linear/prepared_affine_problem.hpp>
#include <pops/numerics/elliptic/linear/scaled_scalar.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace pops {

namespace detail {
struct KrylovWorkspaceAccess;
}

struct KrylovControls {
  KrylovMethod method = KrylovMethod::kCg;
  Real rel_tol = Real(1e-8);
  Real abs_tol = Real(0);
  int max_iterations = 1;
  int restart = 0;
  Real relaxation = Real(1);
};

class KrylovWorkspace {
 public:
  static constexpr int max_gmres_restart() {
    return std::numeric_limits<int>::max() /
               static_cast<int>(detail::PreparedFieldAlgebra::kRobustDotPayloadWidth) -
           1;
  }

  KrylovWorkspace(const MultiFab& prototype, KrylovMethod method, KrylovFootprint footprint)
      : method_(method), footprint_(footprint), layout_(detail::layout_fingerprint(prototype)) {
    if (footprint_.components != prototype.ncomp() || footprint_.input_ghosts != prototype.n_grow())
      throw std::invalid_argument("KrylovWorkspace footprint disagrees with prototype");
    if (method_ == KrylovMethod::kGmres) {
      if (footprint_.restart < 1)
        throw std::invalid_argument("KrylovWorkspace GMRES restart must be positive");
      if (footprint_.restart > max_gmres_restart())
        throw std::invalid_argument(
            "KrylovWorkspace GMRES restart exceeds the native batched robust-dot collective "
            "capacity");
      gmres_restart_ = static_cast<std::size_t>(footprint_.restart);
      if (gmres_restart_ > std::numeric_limits<std::size_t>::max() / (gmres_restart_ + 1))
        throw std::length_error("KrylovWorkspace GMRES Hessenberg size overflows size_t");
    } else if (footprint_.restart != 0) {
      throw std::invalid_argument("KrylovWorkspace restart belongs only to GMRES");
    }
    const std::size_t count = required_fields(method_, footprint_);
    fields_.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
      fields_.emplace_back(prototype.box_array(), prototype.dmap(), prototype.ncomp(),
                           footprint_.input_ghosts);
    for (MultiFab& field : fields_)
      field.share_halo_cache_from(prototype);
    allocation_count_ = count;
    if (method_ == KrylovMethod::kGmres) {
      h_.assign((gmres_restart_ + 1) * gmres_restart_, Real(0));
      cosine_.assign(gmres_restart_, Real(0));
      sine_.assign(gmres_restart_, Real(0));
      rotated_rhs_.assign(gmres_restart_ + 1, Real(0));
      solution_coefficient_.assign(gmres_restart_, Real(0));
      // These mirrors carry the least-squares right hand side and triangular coefficients through
      // exponent ranges which cannot be represented by a `Real`.  They are sized here, before the
      // solve boundary, so a GMRES iteration never allocates while avoiding an overflowing y/H*y.
      scaled_h_.assign((gmres_restart_ + 1) * gmres_restart_, detail::ScaledScalar::zero());
      scaled_rotated_rhs_.assign(gmres_restart_ + 1, detail::ScaledScalar::zero());
      scaled_solution_coefficient_.assign(gmres_restart_, detail::ScaledScalar::zero());
      // One persistent contiguous MPI payload for all Arnoldi projections of a column (plus its
      // local norm term). The normal pass and the selective DGKS reorthogonalization reuse it; no
      // temporary vector appears in the loop.
      const std::size_t ordinary_count = gmres_restart_ + 1;
      constexpr std::size_t kValuesPerProjection =
          detail::PreparedFieldAlgebra::kRobustDotPayloadWidth + 1;
      if (ordinary_count > std::numeric_limits<std::size_t>::max() / kValuesPerProjection)
        throw std::length_error("KrylovWorkspace GMRES reduction storage overflows size_t");
      gmres_reduction_data_.assign(ordinary_count * kValuesPerProjection, 0.0);
    }
  }

  static std::size_t required_fields(KrylovMethod method, const KrylovFootprint& footprint) {
    if (method != KrylovMethod::kGmres && footprint.restart != 0)
      throw std::invalid_argument("KrylovWorkspace restart belongs only to GMRES");
    switch (method) {
      case KrylovMethod::kRichardson:
        if (footprint.preconditioned)
          throw std::invalid_argument("Richardson has no prepared preconditioner slot");
        return 2;  // b_eff, true residual
      case KrylovMethod::kCg:
        if (footprint.preconditioned)
          throw std::invalid_argument("CG has no prepared preconditioner slot");
        return 4;  // b_eff, r, p, Ap/true residual
      case KrylovMethod::kBicgstab:
        return footprint.preconditioned ? 9 : 7;
      case KrylovMethod::kGmres:
        if (footprint.restart < 1 || footprint.restart > max_gmres_restart())
          throw std::invalid_argument(
              "GMRES restart exceeds the native batched robust-dot collective capacity");
        // b_eff, V[0..restart], w/raw residual, plus one preconditioned Arnoldi vector only when
        // M is non-identity. The identity route neither allocates nor copies through a fake M slot.
        return static_cast<std::size_t>(footprint.restart) + (footprint.preconditioned ? 4u : 3u);
    }
    throw std::invalid_argument("unknown Krylov method");
  }

  void bind(const PreparedAffineLinearProblem& problem) {
    detail::KrylovCollectivePayload payload;
    long local_failure = detail::PreparedProblemAccess::append_collective_state(problem, payload);
    append_collective_state_(payload);
    if (local_failure == 0 &&
        (problem.layout_fingerprint() != layout_ || problem.footprint() != footprint_))
      local_failure = 4;
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
    if (local_failure == 0 &&
        (controls.method != method_ || controls.restart != footprint_.restart))
      local_failure = 7;
    const bool agrees = detail::collective_payload_agrees(payload);
    const long collective_failure = all_reduce_max(local_failure);
    throw_collective_failure_(collective_failure);
    if (!agrees)
      throw std::logic_error("KrylovWorkspace bound contract differs across communicator ranks");
  }

  KrylovMethod method() const { return method_; }
  const KrylovFootprint& footprint() const { return footprint_; }
  /// Number of persistent MultiFab work vectors (not heap-allocation events).
  std::size_t allocation_count() const { return allocation_count_; }
  std::size_t scalar_value_count() const {
    return h_.size() + cosine_.size() + sine_.size() + rotated_rhs_.size() +
           solution_coefficient_.size() + scaled_h_.size() + scaled_rotated_rhs_.size() +
           scaled_solution_coefficient_.size();
  }
  std::size_t collective_value_count() const { return gmres_reduction_data_.size(); }

 private:
  friend struct detail::KrylovWorkspaceAccess;

  static void append_footprint_(detail::KrylovCollectivePayload& payload,
                                const KrylovFootprint& footprint) noexcept {
    payload.append(footprint.components);
    payload.append(footprint.input_ghosts);
    payload.append(footprint.restart);
    payload.append(static_cast<std::uint8_t>(footprint.preconditioned));
  }

  static void append_controls_(detail::KrylovCollectivePayload& payload,
                               const KrylovControls& controls) noexcept {
    payload.append(static_cast<std::uint8_t>(controls.method));
    payload.append(std::bit_cast<std::uint64_t>(controls.rel_tol));
    payload.append(std::bit_cast<std::uint64_t>(controls.abs_tol));
    payload.append(controls.max_iterations);
    payload.append(controls.restart);
    payload.append(std::bit_cast<std::uint64_t>(controls.relaxation));
  }

  void append_collective_state_(detail::KrylovCollectivePayload& payload) const noexcept {
    payload.append(static_cast<std::uint8_t>(method_));
    append_footprint_(payload, footprint_);
    payload.append(layout_);
    payload.append(static_cast<std::uint8_t>(snapshot_.has_value()));
    payload.append(snapshot_.value_or(OperatorEvaluationSnapshot{}));
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

  Real& h(int row, int column) {
    return h_[static_cast<std::size_t>(row) * gmres_restart_ + static_cast<std::size_t>(column)];
  }
  Real& cosine(int index) { return cosine_[static_cast<std::size_t>(index)]; }
  Real& sine(int index) { return sine_[static_cast<std::size_t>(index)]; }
  Real& rotated_rhs(int index) { return rotated_rhs_[static_cast<std::size_t>(index)]; }
  Real& solution_coefficient(int index) {
    return solution_coefficient_[static_cast<std::size_t>(index)];
  }
  detail::ScaledScalar& scaled_h(int row, int column) {
    return scaled_h_[static_cast<std::size_t>(row) * gmres_restart_ +
                     static_cast<std::size_t>(column)];
  }
  detail::ScaledScalar& scaled_rotated_rhs(int index) {
    return scaled_rotated_rhs_[static_cast<std::size_t>(index)];
  }
  detail::ScaledScalar& scaled_solution_coefficient(int index) {
    return scaled_solution_coefficient_[static_cast<std::size_t>(index)];
  }
  double* gmres_reduction_data() { return gmres_reduction_data_.data(); }
  double* gmres_robust_reduction_data() {
    return gmres_reduction_data_.data() + gmres_restart_ + 1;
  }
  std::size_t gmres_reduction_size() const { return gmres_restart_ + 1; }

  KrylovMethod method_;
  KrylovFootprint footprint_;
  OperatorFingerprint layout_{};
  std::vector<MultiFab> fields_;
  std::size_t allocation_count_ = 0;
  std::optional<OperatorEvaluationSnapshot> snapshot_{};
  std::size_t gmres_restart_ = 0;
  std::vector<Real> h_;
  std::vector<Real> cosine_;
  std::vector<Real> sine_;
  std::vector<Real> rotated_rhs_;
  std::vector<Real> solution_coefficient_;
  std::vector<detail::ScaledScalar> scaled_h_;
  std::vector<detail::ScaledScalar> scaled_rotated_rhs_;
  std::vector<detail::ScaledScalar> scaled_solution_coefficient_;
  std::vector<double> gmres_reduction_data_;
};

}  // namespace pops
