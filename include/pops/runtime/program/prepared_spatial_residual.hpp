#pragma once

#include <pops/numerics/elliptic/interface/field_newton_krylov.hpp>

#include <cmath>
#include <limits>

namespace pops::runtime::program {

/// One ordered vector spatial residual in solver coordinates. The callable evaluates the complete
/// F(q)=Q(q)-U_n-tau R(Q(q)); U_n and tau are frozen captures, never inferred from the seed.
/// The existing field Newton workspace is only a numerical algorithm here: no physical field
/// registration, nullspace compatibility projection or gauge is introduced by this adapter.
template <int Dim>
class PreparedSpatialResidual final {
 public:
  using field_type = MultiFab<Dim>;

  PreparedSpatialResidual(const field_type& prototype, FieldNewtonOptions options,
                          Real difference_step, bool guard_local = false,
                          const ExecutionLane* authority_lane = nullptr)
      : newton_(prototype.layout(), prototype.distribution(), prototype.local_rank(), options,
                prototype.ncomp()),
        candidate_(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                   prototype.ncomp(), Extent<Dim>{}),
        perturbed_(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                   prototype.ncomp(), Extent<Dim>{}),
        plus_(prototype.layout(), prototype.distribution(), prototype.local_rank(),
              prototype.ncomp(), Extent<Dim>{}),
        minus_(prototype.layout(), prototype.distribution(), prototype.local_rank(),
               prototype.ncomp(), Extent<Dim>{}),
        difference_step_(difference_step), guard_local_(guard_local), authority_lane_(authority_lane) {
    if (guard_local_ && !authority_lane_)
      throw std::invalid_argument("guarded spatial residual requires its prepared execution lane");
    if (prototype.ncomp() <= 0 || !std::isfinite(difference_step_) || difference_step_ <= Real(0))
      throw std::invalid_argument(
          "spatial residual requires nonempty component storage and a finite FD step");
  }

  template <class Residual>
  SolveReport solve(const field_type* seed, Residual&& residual, const ExecutionLane& lane) {
    residual_evaluations_ = 0;
    derivative_evaluations_ = 0;
    const auto& prepared_lane = guard_local_ ? *authority_lane_ : lane;
    local_phase_(prepared_lane, [&] {
      if (guard_local_ && &lane != authority_lane_)
        throw std::invalid_argument("spatial residual received a foreign execution lane");
      if (seed) {
        authenticate_(*seed);
        lincomb(candidate_, Real(1), *seed, Real(0), *seed);
      } else candidate_.set_val(Real(0));
    });
    auto evaluate = [&](const field_type& q, field_type& result, int evaluation) {
      local_phase_(prepared_lane, [&] { increment_(residual_evaluations_); if (guard_local_) { authenticate_(q); authenticate_(result); } });
      residual(q, result, evaluation);
    };
    // FieldNewtonKrylovWorkspace solves J delta = b-A(q). Its residual convention is
    // the negative of the public equation F(q)=0; J remains the derivative of F.
    auto defect = [&](const field_type& q, field_type& result, int evaluation) {
      evaluate(q, result, evaluation);
      local_phase_(prepared_lane, [&] { scale(result, Real(-1)); });
    };
    auto derivative = [&](const field_type& q, const field_type& direction, field_type& result,
                          int evaluation) {
      local_phase_(prepared_lane, [&] { increment_(derivative_evaluations_); });
      Real norm_q = 0, norm_v = 0;
      if (guard_local_) {
        Real local_q = 0, local_v = 0;
        local_phase_(prepared_lane, [&] { local_q = dot_all_local(q, q); local_v = dot_all_local(direction, direction); });
        norm_q = std::sqrt(all_reduce_sum(local_q, prepared_lane));
        norm_v = std::sqrt(all_reduce_sum(local_v, prepared_lane));
      } else {
        norm_q = std::sqrt(all_reduce_sum(dot_all_local(q, q), lane));
        norm_v = std::sqrt(all_reduce_sum(dot_all_local(direction, direction), lane));
      }
      if (norm_v == Real(0)) {
        local_phase_(prepared_lane, [&] { result.set_val(Real(0)); });
        return;
      }
      const Real step = difference_step_ * std::max(Real(1), norm_q) / norm_v;
      if (!std::isfinite(step) || !(step > Real(0))) {
        local_phase_(prepared_lane, [&] { result.set_val(std::numeric_limits<Real>::quiet_NaN()); });
        return;
      }
      local_phase_(prepared_lane, [&] { lincomb(perturbed_, Real(1), q, step, direction); });
      evaluate(perturbed_, plus_, evaluation);
      local_phase_(prepared_lane, [&] { lincomb(perturbed_, Real(1), q, -step, direction); });
      evaluate(perturbed_, minus_, evaluation);
      local_phase_(prepared_lane, [&] { lincomb(result, Real(0.5) / step, plus_, -Real(0.5) / step, minus_); });
    };
    auto no_gauge = [](field_type&) {};
    auto report = newton_.solve(candidate_, defect, derivative, no_gauge, prepared_lane, guard_local_);
    // Report actual full-residual evaluations, including the two evaluations per JVP.
    report.evaluations = residual_evaluations_;
    return report;
  }

  const field_type& candidate() const noexcept { return candidate_; }
  int residual_evaluations() const noexcept { return residual_evaluations_; }
  int derivative_evaluations() const noexcept { return derivative_evaluations_; }

 private:
  static void increment_(int& count) {
    if (count == std::numeric_limits<int>::max())
      throw std::length_error("spatial residual evaluation counter exceeds its report capacity");
    ++count;
  }
  void authenticate_(const field_type& value) const {
    if (value.layout() != candidate_.layout() ||
        value.distribution() != candidate_.distribution() ||
        value.local_rank() != candidate_.local_rank() || value.ncomp() != candidate_.ncomp())
      throw std::invalid_argument(
          "spatial residual seed differs from its prepared component layout");
  }
  template <class Operation>
  void local_phase_(const ExecutionLane& lane, Operation&& operation) const {
    if (!guard_local_) { operation(); return; }
    std::exception_ptr error;
    try { operation(); } catch (...) { error = std::current_exception(); }
    try { Kokkos::fence(); } catch (...) { if (!error) error = std::current_exception(); }
    collectively_rethrow_exception(error, lane, "candidate spatial residual local phase");
  }
  FieldNewtonKrylovWorkspace<Dim> newton_;
  field_type candidate_, perturbed_, plus_, minus_;
  Real difference_step_;
  bool guard_local_ = false;
  const ExecutionLane* authority_lane_ = nullptr;
  int residual_evaluations_ = 0;
  int derivative_evaluations_ = 0;
};

}  // namespace pops::runtime::program
