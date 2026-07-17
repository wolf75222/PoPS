#pragma once

/// @file
/// @brief Persistent storage for prepared affine Krylov solves.

#include <pops/numerics/elliptic/linear/prepared_affine_problem.hpp>

#include <cstddef>
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
  KrylovWorkspace(const MultiFab& prototype, KrylovMethod method, KrylovFootprint footprint)
      : method_(method), footprint_(footprint), layout_(detail::layout_fingerprint(prototype)) {
    if (footprint_.components != prototype.ncomp() || footprint_.input_ghosts != prototype.n_grow())
      throw std::invalid_argument("KrylovWorkspace footprint disagrees with prototype");
    if (method_ == KrylovMethod::kGmres) {
      if (footprint_.restart < 1)
        throw std::invalid_argument("KrylovWorkspace GMRES restart must be positive");
      gmres_restart_ = static_cast<std::size_t>(footprint_.restart);
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
      // One persistent contiguous MPI payload for all Arnoldi projections of a column (plus its
      // local norm term). The normal pass and the selective DGKS reorthogonalization reuse it; no
      // temporary vector appears in the loop.
      gmres_reduction_data_.assign(gmres_restart_ + 1, 0.0);
    }
  }

  static std::size_t required_fields(KrylovMethod method, const KrylovFootprint& footprint) {
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
        // b_eff, V[0..restart], w/raw residual, plus one preconditioned Arnoldi vector only when
        // M is non-identity. The identity route neither allocates nor copies through a fake M slot.
        return static_cast<std::size_t>(footprint.restart) + (footprint.preconditioned ? 4u : 3u);
    }
    throw std::invalid_argument("unknown Krylov method");
  }

  void bind(const PreparedAffineLinearProblem& problem) {
    problem.require_current();
    if (problem.layout_fingerprint() != layout_ || problem.footprint() != footprint_)
      throw std::invalid_argument("KrylovWorkspace is incompatible with prepared problem");
    if (problem.has_preconditioner() != footprint_.preconditioned)
      throw std::invalid_argument("KrylovWorkspace preconditioner footprint mismatch");
    snapshot_ = problem.snapshot();
  }

  void require_bound(const PreparedAffineLinearProblem& problem,
                     const KrylovControls& controls) const {
    problem.require_current();
    if (!snapshot_ || *snapshot_ != problem.snapshot())
      throw std::logic_error("KrylovWorkspace snapshot is not bound to prepared problem");
    if (controls.method != method_ || controls.restart != footprint_.restart)
      throw std::invalid_argument("KrylovWorkspace method/restart mismatch");
  }

  KrylovMethod method() const { return method_; }
  const KrylovFootprint& footprint() const { return footprint_; }
  /// Number of persistent MultiFab work vectors (not heap-allocation events).
  std::size_t allocation_count() const { return allocation_count_; }
  std::size_t scalar_value_count() const {
    return h_.size() + cosine_.size() + sine_.size() + rotated_rhs_.size() +
           solution_coefficient_.size();
  }
  std::size_t collective_value_count() const { return gmres_reduction_data_.size(); }

 private:
  friend struct detail::KrylovWorkspaceAccess;

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
  double* gmres_reduction_data() { return gmres_reduction_data_.data(); }
  std::size_t gmres_reduction_size() const { return gmres_reduction_data_.size(); }

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
  std::vector<double> gmres_reduction_data_;
};

}  // namespace pops
