/// @file
/// @brief Prepared conservative face diffusion; no transport or solve
/// authority.
#pragma once
#include <array>
#include <cmath>
#include <exception>
#include <memory>
#include <optional>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/diffusion/bernoulli.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>
#include <pops/runtime/program/accepted_exchange.hpp>
#include <pops/runtime/program/host_face_publication.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::program {
class DiffusiveEvaluationError final : public std::runtime_error {
  int status_;
  unsigned reason_;

 public:
  explicit DiffusiveEvaluationError(const std::string& message, int status = 2,
                                    unsigned reason = 501)
      : std::runtime_error(message), status_(status), reason_(reason) {}
  int status() const noexcept { return status_; }
  unsigned reason() const noexcept { return reason_; }
};
template <int Dim, int Components = 1, bool Tensor = false>
struct DiffusiveLawResult {
  std::array<Real, Components*(Tensor ? 1 + Dim* Dim + Components : Dim + 2)> values{};
  int evaluation_status = 0;
  unsigned reason_code = 0;
};
enum class DiffusiveBoundaryKind { periodic, value, conormal };
template <int Dim>
struct DiffusiveBoundary {
  DiffusiveBoundaryKind kind = DiffusiveBoundaryKind::periodic;
  Real value = Real(0);
  std::array<Real, Dim> slope{};
  POPS_HD Real trace(const Geometry<Dim>& geometry, const Index<Dim>& face, int axis) const {
    Real result = value;
    for (int d = 0; d < Dim; ++d)
      result += slope[d] * (d == axis ? geometry.face_coordinate(d, face[d])
                                      : geometry.cell_coordinate(d, face[d]));
    return result;
  }
};

// Namespace-scope functors preserve device compilation without extended-lambda
// access restrictions or conditional first captures. The numerical bodies are
// unchanged.
namespace prepared_diffusion_detail {
template <int Size>
POPS_HD bool positive_definite_(const std::array<Real, Size * Size>& matrix) {
  std::array<Real, Size * Size> factor{};
  for (int row = 0; row < Size; ++row)
    for (int column = 0; column <= row; ++column) {
      const Real value = matrix[row * Size + column];
      const Real transpose = matrix[column * Size + row];
      if (!Kokkos::isfinite(value) || !Kokkos::isfinite(transpose) ||
          Kokkos::abs(value - transpose) >
              64 * std::numeric_limits<Real>::epsilon() *
                  Kokkos::max(Real(1), Kokkos::max(Kokkos::abs(value), Kokkos::abs(transpose))))
        return false;
      Real remainder = value;
      for (int k = 0; k < column; ++k)
        remainder -= factor[row * Size + k] * factor[column * Size + k];
      if (row == column) {
        if (!(remainder > 0) || !Kokkos::isfinite(remainder))
          return false;
        factor[row * Size + column] = Kokkos::sqrt(remainder);
      } else {
        factor[row * Size + column] = remainder / factor[column * Size + column];
      }
    }
  return true;
}
template <int Dim, int Components, bool Tensor, bool Coupled, bool Fitted, class Law>
struct EvaluateLaw {
  static constexpr int LawStride = Tensor ? 1 + Dim * Dim + Components : Dim + 2;
  Law law;
  FieldView<Real, Dim> w, a, status, reason;
  POPS_HD void operator()(const Index<Dim>& cell) const {
    const auto evaluation = law(cell);
    std::array<Real, Components * LawStride> values;
    if constexpr (requires {
                    evaluation.evaluation_status;
                    evaluation.reason_code;
                    evaluation.values;
                  }) {
      int category = evaluation.evaluation_status;
      if (category < 0 || category > 3)
        category = 3;
      status(cell, 0) = category;
      reason(cell, 0) = evaluation.reason_code;
      if (category != 0)
        return;  // Failed native outputs remain unread.
      values = evaluation.values;
    } else {
      values = evaluation;
      status(cell, 0) = 0;
      reason(cell, 0) = 0;
    }
    bool valid = true;
    if constexpr (Tensor) {
      std::array<Real, Components * Components> jacobian{};
      for (int component = 0; component < Components; ++component) {
        const int offset = component * LawStride;
        w(cell, component) = values[offset];
        valid = valid && Kokkos::isfinite(values[offset]);
        std::array<Real, Dim * Dim> tensor{};
        for (int entry = 0; entry < Dim * Dim; ++entry) {
          tensor[entry] = values[offset + 1 + entry];
          a(cell, component * Dim * Dim + entry) = tensor[entry];
        }
        valid = valid && positive_definite_<Dim>(tensor);
        for (int column = 0; column < Components; ++column) {
          const Real derivative = values[offset + 1 + Dim * Dim + column];
          jacobian[component * Components + column] = derivative;
          w(cell, Components + component * Components + column) = derivative;
        }
      }
      if constexpr (Coupled) {
        for (const Real entry : jacobian)
          valid = valid && Kokkos::isfinite(entry);
      } else {
        valid = valid && positive_definite_<Components>(jacobian);
      }
    } else {
      for (int component = 0; component < Components; ++component) {
        const int offset = component * (Dim + 2);
        valid = valid && Kokkos::isfinite(values[offset]) &&
                Kokkos::isfinite(values[offset + Dim + 1]) && values[offset + Dim + 1] >= 0;
        w(cell, component) = values[offset];
        w(cell, Components + component) = values[offset + Dim + 1];
        for (int axis = 0; axis < Dim; ++axis) {
          a(cell, component * Dim + axis) = values[offset + axis + 1];
          valid = valid && Kokkos::isfinite(values[offset + axis + 1]) &&
                  (Fitted ? values[offset + axis + 1] > 0 : values[offset + axis + 1] >= 0);
        }
      }
    }
    if (!valid) {
      status(cell, 0) = 2;
      reason(cell, 0) = 501;
    }
  }
};

template <int Dim>
struct Status {
  FieldView<const Real, Dim> status;
  POPS_HD Real operator()(const Index<Dim>& cell) const { return status(cell, 0); }
};

template <int Dim>
struct Reason {
  FieldView<const Real, Dim> status, reason;
  int category;
  POPS_HD Real operator()(const Index<Dim>& cell) const {
    return status(cell, 0) == category ? reason(cell, 0) : Real(0);
  }
};

template <int Dim, int Components>
struct TensorGradient {
  FieldView<const Real, Dim> w, a;
  FieldView<Real, Dim> flux, norm_a, norm_j;
  Geometry<Dim> geometry;
  POPS_HD void operator()(const Index<Dim>& cell) const {
    Real maximum_a = 0, maximum_j = 0;
    for (int component = 0; component < Components; ++component) {
      Real row_j = 0;
      for (int column = 0; column < Components; ++column)
        row_j += Kokkos::abs(w(cell, Components + component * Components + column));
      maximum_j = Kokkos::max(maximum_j, row_j);
      for (int axis = 0; axis < Dim; ++axis) {
        Real value = 0, row_a = 0;
        for (int tangent = 0; tangent < Dim; ++tangent) {
          auto left = cell, right = cell;
          --left[tangent];
          ++right[tangent];
          const Real coefficient = a(cell, component * Dim * Dim + axis * Dim + tangent);
          value += coefficient * (w(right, component) - w(left, component)) /
                   (2 * geometry.spacing(tangent));
          row_a += Kokkos::abs(coefficient);
        }
        maximum_a = Kokkos::max(maximum_a, row_a);
        auto left = cell, right = cell;
        --left[axis];
        ++right[axis];
        flux(cell, component * Dim + axis) = value;
        flux(cell, Components * Dim + component * Dim + axis) =
            a(cell, component * Dim * Dim + axis * Dim + axis) *
            (w(right, component) - 2 * w(cell, component) + w(left, component)) /
            (2 * geometry.spacing(axis));
      }
    }
    norm_a(cell, 0) = maximum_a;
    norm_j(cell, 0) = maximum_j;
  }
};

template <int Dim, int Components, bool Tensor, bool Fitted>
struct FaceFlux {
  FieldView<const Real, Dim> q, w, a, tensor_flux;
  FieldView<Real, Dim> face_values;
  Geometry<Dim> geometry;
  std::array<DiffusiveBoundary<Dim>, 2 * Dim * Components> physical;
  std::array<DiffusiveBoundary<Dim>, 2 * Dim> potential_boundaries;
  Real drift_ratio;
  int axis;
  POPS_HD void operator()(const Index<Dim>& face) const {
    Index<Dim> left = face, right = face;
    --left[axis];
    const bool lower = face[axis] == geometry.domain().lo[axis];
    const bool upper = face[axis] == geometry.domain().hi[axis] + 1;
    const Real h = geometry.spacing(axis);
    for (int component = 0; component < Components; ++component) {
      const auto boundary = physical[component * 2 * Dim + 2 * axis + (upper ? 1 : 0)];
      Real flux = Real(0), conductance = Real(0), left_loss = Real(0), right_loss = Real(0);
      if constexpr (Tensor) {
        const int index = component * Dim + axis;
        flux = Real(.5) * (tensor_flux(left, index) + tensor_flux(right, index) +
                           tensor_flux(left, Components * Dim + index) -
                           tensor_flux(right, Components * Dim + index));
      } else if constexpr (Fitted) {
        Real left_state, right_state, left_potential, right_potential, coefficient, distance = h;
        if ((lower || upper) && boundary.kind != DiffusiveBoundaryKind::periodic) {
          const Index<Dim> center = lower ? right : left;
          Index<Dim> inside = center;
          inside[axis] += lower ? 1 : -1;
          coefficient = Real(1.5) * a(center, component * Dim + axis) -
                        Real(0.5) * a(inside, component * Dim + axis);
          distance = h / Real(2);
          const Real density_trace = boundary.trace(geometry, face, axis);
          const Real potential_trace =
              potential_boundaries[2 * axis + (upper ? 1 : 0)].trace(geometry, face, axis);
          left_state = lower ? density_trace : q(center, component);
          right_state = lower ? q(center, component) : density_trace;
          left_potential = lower ? potential_trace : w(center, component);
          right_potential = lower ? w(center, component) : potential_trace;
        } else {
          coefficient =
              Real(0.5) * (a(left, component * Dim + axis) + a(right, component * Dim + axis));
          left_state = q(left, component);
          right_state = q(right, component);
          left_potential = w(left, component);
          right_potential = w(right, component);
        }
        const Real jump = drift_ratio * (right_potential - left_potential);
        left_loss = coefficient * scharfetter_gummel_bernoulli(jump) / distance;
        right_loss = coefficient * scharfetter_gummel_bernoulli(-jump) / distance;
        flux = right_loss * right_state - left_loss * left_state;
        if (!(coefficient > 0))
          flux = std::numeric_limits<Real>::quiet_NaN();
      } else if ((lower || upper) && boundary.kind != DiffusiveBoundaryKind::periodic) {
        const Index<Dim> center = lower ? right : left;
        const Real orientation = lower ? Real(-1) : Real(1);
        Index<Dim> inside = center;
        inside[axis] += lower ? 1 : -1;
        const Real coefficient = Real(1.5) * a(center, component * Dim + axis) -
                                 Real(0.5) * a(inside, component * Dim + axis);
        if (boundary.kind == DiffusiveBoundaryKind::conormal) {
          const Real trace = boundary.trace(geometry, face, axis);
          flux = orientation * trace;
          if (!Kokkos::isfinite(coefficient) || coefficient < 0 || (coefficient == 0 && trace != 0))
            flux = std::numeric_limits<Real>::quiet_NaN();
        } else {
          if (coefficient == 0) {
            flux = conductance = Real(0);
          } else if (!(coefficient > 0)) {
            flux = std::numeric_limits<Real>::quiet_NaN();
          } else {
            flux = orientation * coefficient * Real(2) *
                   (boundary.trace(geometry, face, axis) - w(center, component)) / h;
            conductance = Real(2) * coefficient * w(center, Components + component) / h;
          }
        }
      } else {
        const Real coefficient =
            Real(0.5) * (a(left, component * Dim + axis) + a(right, component * Dim + axis));
        if (coefficient == 0) {
          flux = conductance = Real(0);
        } else {
          flux = coefficient * (w(right, component) - w(left, component)) / h;
          const Real difference = q(right, component) - q(left, component);
          const Real secant = difference != 0
                                  ? (w(right, component) - w(left, component)) / difference
                                  : Real(0.5) * (w(right, Components + component) +
                                                 w(left, Components + component));
          conductance = coefficient * secant / h;
        }
      }
      if constexpr (!Fitted)
        left_loss = right_loss = conductance;
      face_values(face, component) = flux;
      face_values(face, Components + component) = left_loss;
      face_values(face, 2 * Components + component) = right_loss;
    }
  }
};

template <int Dim, int Components>
struct Divergence {
  nd::FaceFieldView<Real, Dim> faces;
  FieldView<Real, Dim> result, status;
  Geometry<Dim> geometry;
  POPS_HD void operator()(const Index<Dim>& cell) const {
    Real maximum_frequency = 0;
    for (int component = 0; component < Components; ++component) {
      Real divergence = 0, frequency = 0;
      bool valid = true;
      for (int axis = 0; axis < Dim; ++axis) {
        Index<Dim> upper = cell;
        ++upper[axis];
        for (int side = 0; side < 2; ++side) {
          const auto face = side == 0 ? cell : upper;
          valid = valid && Kokkos::isfinite(faces.axes[axis](face, component)) &&
                  Kokkos::isfinite(faces.axes[axis](face, Components + component)) &&
                  faces.axes[axis](face, Components + component) >= 0 &&
                  Kokkos::isfinite(faces.axes[axis](face, 2 * Components + component)) &&
                  faces.axes[axis](face, 2 * Components + component) >= 0;
        }
        divergence += (faces.axes[axis](upper, component) - faces.axes[axis](cell, component)) /
                      geometry.spacing(axis);
        frequency += (faces.axes[axis](upper, Components + component) +
                      faces.axes[axis](cell, 2 * Components + component)) /
                     geometry.spacing(axis);
      }
      result(cell, component) = divergence;
      const Real component_frequency =
          valid && Kokkos::isfinite(divergence) && Kokkos::isfinite(frequency) && frequency >= 0
              ? frequency
              : std::numeric_limits<Real>::infinity();
      maximum_frequency = Kokkos::max(maximum_frequency, component_frequency);
    }
    status(cell, 0) = maximum_frequency;
  }
};

}  // namespace prepared_diffusion_detail

/// One preparation per Program evaluation/implicit operator, reused for every
/// trial. The law factory supplies W(U), nonnegative diagonal A(U,fields), and
/// dW/dU at each cell. Neighbor differences are taken AFTER evaluating W: no
/// discrete chain rule is substituted.
template <int Dim, int Components = 1, bool Tensor = false, bool Coupled = false>
class PreparedDiffusion {
  static_assert(Dim >= 1 && Dim <= 3, "Cartesian diffusion requires native dimensions 1 through 3");
  static_assert(Components > 0, "diffusion requires at least one accumulated component");
  static_assert(!Coupled || Tensor, "coupled gradients require the full component Jacobian");
  using Field = MultiFab<Dim>;
  using Boundary = PreparedScalarBoundarySession<Dim>;
  Geometry<Dim> geometry_;
  std::array<DiffusiveBoundary<Dim>, 2 * Dim * Components> physical_;
  static constexpr int LawStride = Tensor ? 1 + Dim * Dim + Components : Dim + 2;
  Field variable_, coefficients_, status_, reason_, tensor_flux_;
  std::shared_ptr<Boundary> tensor_flux_boundary_;
  std::shared_ptr<Boundary> state_boundary_, variable_boundary_, coefficient_boundary_;
  std::vector<nd::FaceField<Dim>> faces_;
  const ExecutionLane* lane_ = nullptr;
  bool prepared_amr_ghosts_ = false;
  Real frequency_ = Real(0);
  bool evaluated_ = false;
  bool matches_storage_(const Field& field) const {
    return field.layout() == variable_.layout() &&
           field.distribution() == variable_.distribution() &&
           field.local_rank() == variable_.local_rank() &&
           field.local_size() == variable_.local_size() && field.ghosts() == variable_.ghosts() &&
           field.ncomp() == Components;
  }

  Box<Dim> evaluation_box_(const Field& input, std::size_t local) const {
    if (!prepared_amr_ghosts_)
      return input.box(local);
    // The diagonal face formula reads adjacent cells; the tensor cell-gradient
    // formula reads one more layer. Extra State allocation halos are not law
    // inputs.
    auto box = input.box(local).grow(Tensor ? 2 : 1);
    for (int axis = 0; axis < Dim; ++axis) {
      if (physical_[2 * axis].kind != DiffusiveBoundaryKind::periodic)
        box.lo[axis] = std::max(box.lo[axis], geometry_.domain().lo[axis]);
      if (physical_[2 * axis + 1].kind != DiffusiveBoundaryKind::periodic)
        box.hi[axis] = std::min(box.hi[axis], geometry_.domain().hi[axis]);
    }
    // Periodic AMR ghosts may have only a coarse donor. Evaluate their prepared
    // storage values directly; coordinate-dependent laws retain their periodic
    // extension. Physical exterior cells instead use the authored face trace
    // below.
    return box;
  }

 public:
  template <class Context>
  PreparedDiffusion(Context& ctx, Field& prototype,
                    std::array<DiffusiveBoundary<Dim>, 2 * Dim * Components> physical,
                    bool prepared_amr_ghosts = false)
      : geometry_(ctx.geometry()), physical_(physical), prepared_amr_ghosts_(prepared_amr_ghosts) {
    lane_ = &ctx.prepared_execution_lane();
    if (all_reduce_max(prototype.ncomp() == Components ? 0L : 1L, *lane_) != 0)
      throw std::invalid_argument(
          "diffusion component count differs from the "
          "prepared constitutive signature "
          "collectively");
    std::exception_ptr allocation_error;
    try {
      variable_ = Field(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                        Components * (Tensor ? 1 + Components : 2), prototype.ghosts());
      coefficients_ = Field(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                            Dim * Components * (Tensor ? Dim : 1), prototype.ghosts());
      status_ = Field(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
                      prototype.ghosts());
      reason_ = Field(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
                      prototype.ghosts());
      if constexpr (Tensor)
        tensor_flux_ = Field(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                             2 * Dim * Components, prototype.ghosts());
      faces_.reserve(prototype.local_size());
      for (std::size_t local = 0; local < prototype.local_size(); ++local)
        faces_.emplace_back(prototype.box(local), 3 * Components);
    } catch (...) {
      allocation_error = std::current_exception();
    }
    if (all_reduce_max(allocation_error ? 1L : 0L, *lane_) != 0) {
      if (lane_->size() == 1 && allocation_error)
        std::rethrow_exception(allocation_error);
      throw std::runtime_error("diffusion preparation allocation failed collectively");
    }
    std::exception_ptr session_error;
    try {
      state_boundary_ = ctx.prepare_mesh_boundary_session(prototype, *lane_);
      variable_boundary_ = ctx.prepare_mesh_boundary_session(variable_, *lane_);
      coefficient_boundary_ = ctx.prepare_mesh_boundary_session(coefficients_, *lane_);
      if constexpr (Tensor)
        tensor_flux_boundary_ = ctx.prepare_mesh_boundary_session(tensor_flux_, *lane_);
      for (int axis = 0; axis < Dim; ++axis) {
        if (geometry_.domain().length(axis) < 2 || prototype.ghosts()[axis] < 1)
          throw std::invalid_argument(
              "diffusion requires at least two cells and one halo per axis");
        if constexpr (Tensor) {
          if (prototype.ghosts()[axis] < 2)
            throw std::invalid_argument("tensor diffusion requires two prepared halos per axis");
          for (int side = 0; side < 2; ++side)
            for (int component = 0; component < Components; ++component)
              if (physical_[component * 2 * Dim + 2 * axis + side].kind !=
                  DiffusiveBoundaryKind::periodic)
                throw std::invalid_argument(
                    "the adjoint tensor diffusion realization requires "
                    "periodic boundaries");
        }
        for (int side = 0; side < 2; ++side) {
          const bool periodic = state_boundary_->topology().is_periodic(
              Face<Dim>{axis, side == 0 ? BoundarySide::lower : BoundarySide::upper});
          for (int component = 0; component < Components; ++component)
            if (periodic != (physical_[component * 2 * Dim + 2 * axis + side].kind ==
                             DiffusiveBoundaryKind::periodic))
              throw std::invalid_argument(
                  "physical diffusive boundary differs "
                  "from bound mesh topology");
        }
      }
    } catch (...) {
      session_error = std::current_exception();
    }
    if (all_reduce_max(session_error ? 1L : 0L, *lane_) != 0) {
      if (lane_->size() == 1 && session_error)
        std::rethrow_exception(session_error);
      throw std::runtime_error("diffusion boundary preparation failed collectively");
    }
  }

  template <class LawFactory>
  void apply(Field& input, Field& output, LawFactory factory) {
    apply_impl_<false>(input, output, factory, Real(0), {});
  }

  /// The same gradient sampling, halo, oriented divergence and accepted ledger,
  /// with one exponentially fitted flux consuming the declared drift and
  /// diffusion together.
  template <class LawFactory>
  void apply_fitted(Field& input, Field& output, LawFactory factory, Real mobility_over_diffusion,
                    std::array<DiffusiveBoundary<Dim>, 2 * Dim> potential_boundaries) {
    static_assert(Components == 1 && !Tensor,
                  "fitted drift diffusion requires a scalar diagonal density law");
    if (!std::isfinite(mobility_over_diffusion) || mobility_over_diffusion < 0)
      throw DiffusiveEvaluationError("fitted drift ratio must be finite and nonnegative");
    for (int face = 0; face < 2 * Dim; ++face) {
      if (physical_[face].kind != potential_boundaries[face].kind ||
          physical_[face].kind == DiffusiveBoundaryKind::conormal)
        throw std::invalid_argument(
            "fitted drift diffusion requires paired periodic or value "
            "density/potential traces");
    }
    apply_impl_<true>(input, output, factory, mobility_over_diffusion, potential_boundaries);
  }

 private:
  template <bool Fitted, class LawFactory>
  void apply_impl_(Field& input, Field& output, LawFactory factory, Real drift_ratio,
                   std::array<DiffusiveBoundary<Dim>, 2 * Dim> potential_boundaries) {
    evaluated_ = false;
    const long storage_error =
        input.shares_storage_with(output) || !matches_storage_(input) || !matches_storage_(output)
            ? 1L
            : 0L;
    if (all_reduce_max(storage_error, *lane_) != 0)
      throw std::invalid_argument(
          "diffusion input/output do not match "
          "prepared constitutive storage collectively");
    std::exception_ptr preparation_error;
    try {
      for (std::size_t local = 0; local < input.local_size(); ++local) {
        const auto law = factory(local);
        const auto w = variable_.fab(local).view();
        const auto a = coefficients_.fab(local).view();
        const auto status = status_.fab(local).view();
        const auto reason = reason_.fab(local).view();
        const Box<Dim> evaluation_box = evaluation_box_(input, local);
        for_each_cell(
            evaluation_box,
            prepared_diffusion_detail::EvaluateLaw<Dim, Components, Tensor, Coupled, Fitted,
                                                   decltype(law)>{law, w, a, status, reason});
      }
      device_fence();
    } catch (...) {
      preparation_error = std::current_exception();
      try {
        device_fence();
      } catch (...) {
      }
    }
    if (all_reduce_max(preparation_error ? 1L : 0L, *lane_) != 0) {
      if (lane_->size() == 1 && preparation_error)
        std::rethrow_exception(preparation_error);
      throw DiffusiveEvaluationError("diffusive constitutive preparation failed collectively", 3,
                                     501);
    }
    Real local_category = 0;
    std::exception_ptr reduction_error;
    try {
      for (std::size_t local = 0; local < input.local_size(); ++local) {
        const auto status = std::as_const(status_).fab(local).view();
        const Box<Dim> evaluation_box = evaluation_box_(input, local);
        local_category = std::max(
            local_category, for_each_cell_reduce_max(
                                evaluation_box, prepared_diffusion_detail::Status<Dim>{status}));
      }
    } catch (...) {
      reduction_error = std::current_exception();
    }
    if (all_reduce_max(reduction_error ? 1L : 0L, *lane_) != 0) {
      if (lane_->size() == 1 && reduction_error)
        std::rethrow_exception(reduction_error);
      throw DiffusiveEvaluationError("diffusive status reduction failed collectively", 3, 501);
    }
    const int category = static_cast<int>(all_reduce_max(local_category, *lane_));
    if (category != 0) {
      Real selected_reason = 0;
      reduction_error = nullptr;
      try {
        for (std::size_t local = 0; local < input.local_size(); ++local) {
          const auto status = std::as_const(status_).fab(local).view();
          const auto reason = std::as_const(reason_).fab(local).view();
          selected_reason = std::max(
              selected_reason, for_each_cell_reduce_max(evaluation_box_(input, local),
                                                        prepared_diffusion_detail::Reason<Dim>{
                                                            status, reason, category}));
        }
      } catch (...) {
        reduction_error = std::current_exception();
      }
      if (all_reduce_max(reduction_error ? 1L : 0L, *lane_) != 0) {
        if (lane_->size() == 1 && reduction_error)
          std::rethrow_exception(reduction_error);
        throw DiffusiveEvaluationError("diffusive reason reduction failed collectively", 3, 501);
      }
      selected_reason = all_reduce_max(selected_reason, *lane_);
      throw DiffusiveEvaluationError("diffusive constitutive evaluation failed", category,
                                     static_cast<unsigned>(selected_reason));
    }
    std::exception_ptr boundary_error;
    try {
      if (!prepared_amr_ghosts_)
        state_boundary_->fill_halo(input);
      variable_boundary_->fill_halo(variable_);
      coefficient_boundary_->fill_halo(coefficients_);
    } catch (...) {
      boundary_error = std::current_exception();
    }
    if (all_reduce_max(boundary_error ? 1L : 0L, *lane_) != 0) {
      if (lane_->size() == 1 && boundary_error)
        std::rethrow_exception(boundary_error);
      throw DiffusiveEvaluationError("diffusive halo preparation failed collectively", 3, 501);
    }
    // G is the centered cell gradient and H its directional second difference
    // /2h. Face averaging below realizes -G^T A G - H^T diag(A) H exactly on
    // periodic meshes. The H term is O(h^2), removes the centered checkerboard
    // kernel, and recovers the ordinary two-point stencil for constant diagonal
    // coefficients.
    Real tensor_frequency = 0;
    if constexpr (Tensor) {
      std::exception_ptr tensor_error;
      const auto geometry = geometry_;
      try {
        for (std::size_t local = 0; local < input.local_size(); ++local) {
          const auto w = std::as_const(variable_).fab(local).view();
          const auto a = std::as_const(coefficients_).fab(local).view();
          const auto flux = tensor_flux_.fab(local).view();
          const auto norm_a = status_.fab(local).view();
          const auto norm_j = reason_.fab(local).view();
          auto box = input.box(local);
          if (prepared_amr_ghosts_) {
            box = evaluation_box_(input, local);
            for (int axis = 0; axis < Dim; ++axis) {
              ++box.lo[axis];
              --box.hi[axis];
            }
          }
          for_each_cell(box, prepared_diffusion_detail::TensorGradient<Dim, Components>{
                                 w, a, flux, norm_a, norm_j, geometry});
        }
        device_fence();
        tensor_flux_boundary_->fill_halo(tensor_flux_);
      } catch (...) {
        tensor_error = std::current_exception();
      }
      if (all_reduce_max(tensor_error ? 1L : 0L, *lane_) != 0) {
        if (lane_->size() == 1 && tensor_error)
          std::rethrow_exception(tensor_error);
        throw DiffusiveEvaluationError("tensor gradient preparation failed collectively", 3, 501);
      }
      Real maximum_a = 0, maximum_j = 0;
      tensor_error = nullptr;
      try {
        maximum_a = reduce_max_local(status_);
        maximum_j = reduce_max_local(reason_);
      } catch (...) {
        tensor_error = std::current_exception();
      }
      if (all_reduce_max(tensor_error ? 1L : 0L, *lane_) != 0)
        throw DiffusiveEvaluationError("tensor coefficient bound reduction failed collectively", 3,
                                       501);
      maximum_a = all_reduce_max(maximum_a, *lane_);
      maximum_j = all_reduce_max(maximum_j, *lane_);
      for (int axis = 0; axis < Dim; ++axis)
        tensor_frequency += (Coupled ? Real(4) : Real(2.5)) * maximum_a * maximum_j /
                            (geometry.spacing(axis) * geometry.spacing(axis));
    }
    const auto geometry = geometry_;
    const auto physical = physical_;
    std::exception_ptr face_error;
    try {
      for (std::size_t local = 0; local < input.local_size(); ++local) {
        const auto q = std::as_const(input).fab(local).view();
        const auto w = std::as_const(variable_).fab(local).view();
        const auto a = std::as_const(coefficients_).fab(local).view();
        const auto faces = faces_[local].view();
        const auto tensor_flux = [&] {
          if constexpr (Tensor)
            return std::as_const(tensor_flux_).fab(local).view();
          else
            return std::as_const(variable_).fab(local).view();
        }();
        for (int axis = 0; axis < Dim; ++axis) {
          const auto face_values = faces.axes[axis];
          for_each_cell(nd::face_box(input.box(local), axis),
                        prepared_diffusion_detail::FaceFlux<Dim, Components, Tensor, Fitted>{
                            q, w, a, tensor_flux, face_values, geometry, physical,
                            potential_boundaries, drift_ratio, axis});
        }
        const auto result = output.fab(local).view();
        const auto status = status_.fab(local).view();
        for_each_cell(output.box(local), prepared_diffusion_detail::Divergence<Dim, Components>{
                                             faces, result, status, geometry});
      }
      device_fence();
    } catch (...) {
      face_error = std::current_exception();
      try {
        device_fence();
      } catch (...) {
      }
    }
    if (all_reduce_max(face_error ? 1L : 0L, *lane_) != 0) {
      if (lane_->size() == 1 && face_error)
        std::rethrow_exception(face_error);
      throw DiffusiveEvaluationError("diffusive face preparation failed collectively", 3, 501);
    }
    Real local_frequency = Real(0);
    reduction_error = nullptr;
    try {
      local_frequency = reduce_max_local(status_);
    } catch (...) {
      reduction_error = std::current_exception();
    }
    if (all_reduce_max(reduction_error ? 1L : 0L, *lane_) != 0) {
      if (lane_->size() == 1 && reduction_error)
        std::rethrow_exception(reduction_error);
      throw DiffusiveEvaluationError("diffusive frequency reduction failed collectively", 3, 501);
    }
    frequency_ = Kokkos::max(all_reduce_max(local_frequency, *lane_), tensor_frequency);
    if (!std::isfinite(frequency_))
      throw DiffusiveEvaluationError("diffusive face evaluation is invalid");
    evaluated_ = true;
  }

 public:
  /// Storage reuse only. Laws and all field/parameter dependencies are
  /// reevaluated by apply; the owning context additionally keys
  /// topology/materialization epochs and the Program evaluation identity so
  /// live stage face ledgers never alias. Compare only the already acquired
  /// geometry/lane snapshot: a cache may call this on only some ranks, so
  /// context getters and collective preparation are forbidden here.
  bool matches_preparation(const Geometry<Dim>& geometry, const ExecutionLane& lane,
                           const Field& prototype,
                           const std::array<DiffusiveBoundary<Dim>, 2 * Dim * Components>& physical,
                           bool prepared_amr_ghosts = false) const {
    if (!matches_storage_(prototype) || lane_ != &lane ||
        prepared_amr_ghosts_ != prepared_amr_ghosts || geometry_.domain() != geometry.domain())
      return false;
    for (std::size_t local = 0; local < prototype.local_size(); ++local)
      if (prototype.box(local) != variable_.box(local))
        return false;
    for (int axis = 0; axis < Dim; ++axis)
      if (geometry_.spacing(axis) != geometry.spacing(axis) ||
          geometry_.face_coordinate(axis, 0) != geometry.face_coordinate(axis, 0))
        return false;
    for (int face = 0; face < 2 * Dim * Components; ++face)
      if (physical_[face].kind != physical[face].kind ||
          physical_[face].value != physical[face].value ||
          physical_[face].slope != physical[face].slope)
        return false;
    return true;
  }

  Real explicit_frequency() const {
    if (!evaluated_)
      throw std::logic_error("diffusive stability requires a completed face evaluation");
    return frequency_;
  }
  /// The accepted affine quadrature supplies this weight once, after all
  /// evaluations succeed. Two cell incidences of an interior face have opposite
  /// orientation. They are retained as distinct quadrature records, so
  /// cancellation and each boundary exchange remain auditable.
  template <class Context>
  void stage_accepted_exchanges(Context& ctx, int program_block, const std::string& operation,
                                const std::string& occurrence, const std::string& evaluation,
                                Real temporal_weight, bool physical_boundary_only = false) const {
    with_exchange_records_(ctx, program_block, operation, occurrence, evaluation, temporal_weight,
                           physical_boundary_only, [&](auto&& produce) {
                             ctx.stage_exchange_batch([&](auto&& stage_exchange) {
                               produce([&](ExchangeRecord record, const Index<Dim>&) {
                                 stage_exchange(std::move(record));
                               });
                             });
                           });
  }

  /// Retain the evaluated rate's face records without publishing a delivery.
  /// The coefficient belongs to this rate; its later affine consumer supplies
  /// the time increment through the ordinary field-expression algebra.
  template <class Context>
  void capture_exchanges(Context& ctx, const Field& rhs, int program_block,
                         const std::string& operation, const std::string& occurrence,
                         const std::string& evaluation, Real coefficient,
                         bool physical_boundary_only = false) const {
    if (all_reduce_max(matches_storage_(rhs) ? 0L : 1L, *lane_) != 0)
      throw std::invalid_argument("captured diffusive rate has foreign storage collectively");
    with_exchange_records_(ctx, program_block, operation, occurrence, evaluation, coefficient,
                           physical_boundary_only, [&](auto&& produce) {
                             ctx.capture_stencil_exchanges(
                                 rhs, program_block, evaluation,
                                 std::forward<decltype(produce)>(produce));
                           });
  }

 private:
  template <class Context, class Consumer>
  void with_exchange_records_(Context& ctx, int program_block, const std::string& operation,
                              const std::string& occurrence, const std::string& evaluation,
                              Real temporal_weight, bool physical_boundary_only,
                              Consumer&& consume) const {
    (void)explicit_frequency();
    const Field* active = nullptr;
    const Field* coverage = nullptr;
    std::exception_ptr active_error;
    long active_layout_error = 0;
    bool owner_contributes = false;
    std::optional<decltype(ctx.prepare_external_trace_face_predicate())> external_face;
    try {
      external_face.emplace(ctx.prepare_external_trace_face_predicate());
      sync_host();
      active = ctx.pointwise_active_mask(program_block, variable_);
      coverage = ctx.pointwise_exchange_coverage_mask(program_block, variable_);
      owner_contributes = accepted_exchange_contributes(variable_, *lane_);
      for (const Field* mask : {active, coverage}) {
        if (mask == nullptr)
          continue;
        sync_host();
        if (mask->local_size() != variable_.local_size())
          active_layout_error = 1;
        else
          for (std::size_t local = 0; local < variable_.local_size(); ++local)
            if (mask->box(local) != variable_.box(local))
              active_layout_error = 1;
      }
    } catch (...) {
      active_error = std::current_exception();
    }
    if (all_reduce_max(active_error ? 1L : 0L, *lane_) != 0) {
      if (lane_->size() == 1 && active_error)
        std::rethrow_exception(active_error);
      throw std::invalid_argument("accepted diffusive face mask preparation failed collectively");
    }
    if (all_reduce_max(active_layout_error, *lane_) != 0)
      throw std::invalid_argument(
          "accepted diffusive face mask differs from "
          "local patches collectively");
    std::forward<Consumer>(consume)([&](auto&& append) {
      if (!owner_contributes)
        return;
      for (std::size_t local = 0; local < variable_.local_size(); ++local) {
        const auto box = variable_.box(local);
        const auto extent = box.extent();
        // The ledger producer runs on the host. Publish completed device data
        // explicitly; a fence alone never makes CudaSpace pointers host-readable.
        // These snapshots stay alive until every record for this patch is staged.
        // Allocation/copy errors are caught by the enclosing collective batch
        // preparation before any records enter the accepted ledger.
        const HostFacePublication<Dim> face_publication(faces_[local]);
        const auto faces = face_publication.view();
        std::optional<HostFabPublication<Dim>> active_publication, coverage_publication;
        if (active != nullptr)
          active_publication.emplace(std::as_const(*active).fab(local));
        if (coverage != nullptr)
          coverage_publication.emplace(std::as_const(*coverage).fab(local));
        const auto active_values =
            active_publication ? active_publication->view() : FieldView<const Real, Dim>{};
        const auto coverage_values =
            coverage_publication ? coverage_publication->view() : FieldView<const Real, Dim>{};
        for (std::int64_t ordinal = 0; ordinal < box.numPts(); ++ordinal) {
          auto remainder = ordinal;
          Index<Dim> cell = box.lo;
          for (int axis = 0; axis < Dim; ++axis) {
            cell[axis] += static_cast<int>(remainder % extent[axis]);
            remainder /= extent[axis];
          }
          if (active != nullptr && active_values(cell, 0) < Real(0.5))
            continue;
          if (coverage != nullptr && coverage_values(cell, 0) < Real(0.5))
            continue;
          for (int axis = 0; axis < Dim; ++axis) {
            Real measure = 1;
            for (int tangent = 0; tangent < Dim; ++tangent)
              if (tangent != axis)
                measure *= geometry_.spacing(tangent);
            for (int side = 0; side < 2; ++side) {
              Index<Dim> face = cell;
              face[axis] += side;
              if (physical_boundary_only) {
                const bool domain_boundary = side == 0
                                                 ? face[axis] == geometry_.domain().lo[axis]
                                                 : face[axis] == geometry_.domain().hi[axis] + 1;
                if (!domain_boundary ||
                    physical_[2 * axis + side].kind == DiffusiveBoundaryKind::periodic)
                  continue;
              }
              std::string identity = "cell";
              for (int d = 0; d < Dim; ++d)
                identity += ":" + std::to_string(cell[d]);
              identity += "/axis:" + std::to_string(axis) + "/side:" + std::to_string(side);
              for (int component = 0; component < Components; ++component) {
                const auto component_identity =
                    Components == 1 ? identity
                                    : identity + "/component:" + std::to_string(component);
                append(ExchangeRecord{operation, occurrence, evaluation, component_identity,
                                      side == 0 ? -1 : 1, measure,
                                      faces.axes[axis](face, component), temporal_weight, 1, axis,
                                      side, component, (*external_face)(axis, side, cell)},
                       cell);
              }
            }
          }
        }
      }
    });
  }

 public:
  const auto& faces() const { return faces_; }
  const Field& prototype() const { return variable_; }
  const auto& geometry() const { return geometry_; }
};

/// Periodic signed component-gradient provider. Physical D and R are retained
/// separately by the authored law; W=(D+R)U and the identity spatial tensor
/// give a two-point flux on each Cartesian axis. Its reported frequency bounds
/// the infinity norm of the semidiscrete operator by
/// 4 ||D+R||_infinity sum_axis h_axis^-2; it is not a monotone CFL certificate
/// for the skew part.
template <int Dim, int Components>
class PreparedCoupledGradient final : public PreparedDiffusion<Dim, Components, true, true> {
  static_assert(Dim >= 1 && Dim <= 3, "coupled gradient requires one to three Cartesian axes");

 public:
  using PreparedDiffusion<Dim, Components, true, true>::PreparedDiffusion;
};
}  // namespace pops::runtime::program
