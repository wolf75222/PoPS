#pragma once

#include <pops/numerics/elliptic/interface/field_newton_krylov.hpp>

#include <cmath>
#include <limits>

namespace pops::runtime::program {

/// One scalar spatial residual in solver coordinates. The callable evaluates the complete
/// F(q)=Q(q)-U_n-tau R(Q(q)); U_n and tau are frozen captures, never inferred from the seed.
/// The existing field Newton workspace is only a numerical algorithm here: no physical field
/// registration, nullspace compatibility projection or gauge is introduced by this adapter.
template <int Dim>
class PreparedSpatialResidual final {
 public:
  using field_type = MultiFab<Dim>;

  PreparedSpatialResidual(const field_type& prototype, FieldNewtonOptions options,
                          Real difference_step)
      : newton_(prototype.layout(), prototype.distribution(), prototype.local_rank(), options),
        candidate_(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
                   Extent<Dim>{}),
        perturbed_(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
                   Extent<Dim>{}),
        plus_(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
              Extent<Dim>{}),
        minus_(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
               Extent<Dim>{}),
        difference_step_(difference_step) {
    if (prototype.ncomp() != 1 || !std::isfinite(difference_step_) || difference_step_ <= Real(0))
      throw std::invalid_argument("spatial residual requires scalar storage and a finite FD step");
  }

  template <class Residual>
  SolveReport solve(const field_type* seed, Residual&& residual, const ExecutionLane& lane) {
    residual_evaluations_ = 0;
    derivative_evaluations_ = 0;
    if (seed) {
      authenticate_(*seed);
      lincomb(candidate_, Real(1), *seed, Real(0), *seed);
    } else {
      candidate_.set_val(Real(0));
    }
    auto evaluate = [&](const field_type& q, field_type& result, int evaluation) {
      increment_(residual_evaluations_);
      residual(q, result, evaluation);
    };
    // FieldNewtonKrylovWorkspace solves J delta = b-A(q). Its residual convention is
    // the negative of the public equation F(q)=0; J remains the derivative of F.
    auto defect = [&](const field_type& q, field_type& result, int evaluation) {
      evaluate(q, result, evaluation);
      scale(result, Real(-1));
    };
    auto derivative = [&](const field_type& q, const field_type& direction, field_type& result,
                          int evaluation) {
      increment_(derivative_evaluations_);
      const Real norm_q = std::sqrt(all_reduce_sum(dot_local(q, q), lane));
      const Real norm_v = std::sqrt(all_reduce_sum(dot_local(direction, direction), lane));
      if (norm_v == Real(0)) {
        result.set_val(Real(0));
        return;
      }
      const Real step = difference_step_ * std::max(Real(1), norm_q) / norm_v;
      if (!std::isfinite(step) || !(step > Real(0))) {
        result.set_val(std::numeric_limits<Real>::quiet_NaN());
        return;
      }
      lincomb(perturbed_, Real(1), q, step, direction);
      evaluate(perturbed_, plus_, evaluation);
      lincomb(perturbed_, Real(1), q, -step, direction);
      evaluate(perturbed_, minus_, evaluation);
      lincomb(result, Real(0.5) / step, plus_, -Real(0.5) / step, minus_);
    };
    auto no_gauge = [](field_type&) {};
    auto report = newton_.solve(candidate_, defect, derivative, no_gauge, lane);
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
        value.local_rank() != candidate_.local_rank() || value.ncomp() != 1)
      throw std::invalid_argument("spatial residual seed differs from its prepared scalar layout");
  }
  FieldNewtonKrylovWorkspace<Dim> newton_;
  field_type candidate_, perturbed_, plus_, minus_;
  Real difference_step_;
  int residual_evaluations_ = 0;
  int derivative_evaluations_ = 0;
};

}  // namespace pops::runtime::program
