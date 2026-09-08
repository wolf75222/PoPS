/// @file
/// @brief Prepared conservative face diffusion; no transport or solve authority.
#pragma once
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>
#include <pops/runtime/program/accepted_exchange.hpp>
#include <pops/numerics/diffusion/bernoulli.hpp>
#include <array>
#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
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
template <int Dim>
struct DiffusiveLawResult {
  std::array<Real, Dim + 2> values{};
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

/// One preparation per Program evaluation/implicit operator, reused for every trial.
/// The law factory supplies W(U), positive diagonal A(U,fields), and dW/dU at each cell.
/// Neighbor differences are taken AFTER evaluating W: no discrete chain rule is substituted.
template <int Dim>
class PreparedDiffusion {
  static_assert(Dim == 1 || Dim == 2, "selected diffusion matrix is Cartesian Dim1/Dim2");
  using Field = MultiFab<Dim>;
  using Boundary = PreparedScalarBoundarySession<Dim>;
  Geometry<Dim> geometry_;
  std::array<DiffusiveBoundary<Dim>, 2 * Dim> physical_;
  Field variable_, coefficients_, status_, reason_;
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
           field.ncomp() == 1;
  }

 public:
  template <class Context>
  PreparedDiffusion(Context& ctx, Field& prototype,
                    std::array<DiffusiveBoundary<Dim>, 2 * Dim> physical,
                    bool prepared_amr_ghosts = false)
      : geometry_(ctx.geometry()), physical_(physical), prepared_amr_ghosts_(prepared_amr_ghosts) {
    if constexpr (!Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                              typename Field::memory_space>::accessible)
      throw std::invalid_argument(
          "selected diffusion face ledger requires host-accessible native storage");
    lane_ = &ctx.prepared_execution_lane();
    if (all_reduce_max(prototype.ncomp() == 1 ? 0L : 1L, *lane_) != 0)
      throw std::invalid_argument("diffusion requires one scalar evolved component collectively");
    std::exception_ptr allocation_error;
    try {
      variable_ = Field(prototype.layout(), prototype.distribution(), prototype.local_rank(), 2,
                        prototype.ghosts());
      coefficients_ = Field(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                            Dim, prototype.ghosts());
      status_ = Field(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
                      prototype.ghosts());
      reason_ = Field(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
                      prototype.ghosts());
      faces_.reserve(prototype.local_size());
      for (std::size_t local = 0; local < prototype.local_size(); ++local)
        faces_.emplace_back(prototype.box(local), 3);
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
      for (int axis = 0; axis < Dim; ++axis) {
        if (geometry_.domain().length(axis) < 2 || prototype.ghosts()[axis] < 1)
          throw std::invalid_argument(
              "diffusion requires at least two cells and one halo per axis");
        for (int side = 0; side < 2; ++side) {
          const bool periodic = state_boundary_->topology().is_periodic(
              Face<Dim>{axis, side == 0 ? BoundarySide::lower : BoundarySide::upper});
          if (periodic != (physical_[2 * axis + side].kind == DiffusiveBoundaryKind::periodic))
            throw std::invalid_argument(
                "physical diffusive boundary differs from bound mesh topology");
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

  /// The same gradient sampling, halo, oriented divergence and accepted ledger, with one
  /// exponentially fitted flux consuming the declared drift and diffusion together.
  template <class LawFactory>
  void apply_fitted(Field& input, Field& output, LawFactory factory, Real mobility_over_diffusion,
                    std::array<DiffusiveBoundary<Dim>, 2 * Dim> potential_boundaries) {
    static_assert(Dim == 1, "selected Scharfetter-Gummel realization requires native Dim1");
    if (!std::isfinite(mobility_over_diffusion) || mobility_over_diffusion < 0)
      throw DiffusiveEvaluationError("fitted drift ratio must be finite and nonnegative");
    for (int face = 0; face < 2 * Dim; ++face) {
      if (physical_[face].kind != potential_boundaries[face].kind ||
          physical_[face].kind == DiffusiveBoundaryKind::conormal)
        throw std::invalid_argument(
            "fitted drift diffusion requires paired periodic or value density/potential traces");
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
          "diffusion input/output do not match prepared scalar storage collectively");
    std::exception_ptr preparation_error;
    try {
      for (std::size_t local = 0; local < input.local_size(); ++local) {
        const auto law = factory(local);
        const auto w = variable_.fab(local).view();
        const auto a = coefficients_.fab(local).view();
        const auto status = status_.fab(local).view();
        const auto reason = reason_.fab(local).view();
        const Box<Dim> evaluation_box =
            prepared_amr_ghosts_ ? input.fab(local).grown_box().intersect(geometry_.domain())
                                 : input.box(local);
        for_each_cell(evaluation_box, [=] POPS_HD(const Index<Dim>& cell) {
          const auto evaluation = law(cell);
          std::array<Real, Dim + 2> values;
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
          bool valid = Kokkos::isfinite(values[0]) && Kokkos::isfinite(values[Dim + 1]) &&
                       values[Dim + 1] >= 0;
          w(cell, 0) = values[0];
          w(cell, 1) = values[Dim + 1];
          for (int axis = 0; axis < Dim; ++axis) {
            a(cell, axis) = values[axis + 1];
            valid = valid && Kokkos::isfinite(values[axis + 1]) && values[axis + 1] > 0;
          }
          if (!valid) {
            status(cell, 0) = 2;
            reason(cell, 0) = 501;
          }
        });
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
        const Box<Dim> evaluation_box =
            prepared_amr_ghosts_ ? input.fab(local).grown_box().intersect(geometry_.domain())
                                 : input.box(local);
        local_category = std::max(
            local_category,
            for_each_cell_reduce_max(
                evaluation_box, [=] POPS_HD(const Index<Dim>& cell) { return status(cell, 0); }));
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
              selected_reason,
              for_each_cell_reduce_max(
                  prepared_amr_ghosts_ ? input.fab(local).grown_box().intersect(geometry_.domain())
                                       : input.box(local),
                  [=] POPS_HD(const Index<Dim>& cell) {
                    return status(cell, 0) == category ? reason(cell, 0) : Real(0);
                  }));
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
    const auto geometry = geometry_;
    const auto physical = physical_;
    std::exception_ptr face_error;
    try {
      for (std::size_t local = 0; local < input.local_size(); ++local) {
        const auto q = std::as_const(input).fab(local).view();
        const auto w = std::as_const(variable_).fab(local).view();
        const auto a = std::as_const(coefficients_).fab(local).view();
        const auto faces = faces_[local].view();
        for (int axis = 0; axis < Dim; ++axis) {
          const auto face_values = faces.axes[axis];
          for_each_cell(nd::face_box(input.box(local), axis), [=] POPS_HD(const Index<Dim>& face) {
            Index<Dim> left = face, right = face;
            --left[axis];
            const bool lower = face[axis] == geometry.domain().lo[axis];
            const bool upper = face[axis] == geometry.domain().hi[axis] + 1;
            const auto boundary = physical[2 * axis + (upper ? 1 : 0)];
            const Real h = geometry.spacing(axis);
            Real flux = Real(0), conductance = Real(0), left_loss = Real(0), right_loss = Real(0);
            if constexpr (Fitted) {
              Real left_state, right_state, left_potential, right_potential, coefficient,
                  distance = h;
              if ((lower || upper) && boundary.kind != DiffusiveBoundaryKind::periodic) {
                const Index<Dim> center = lower ? right : left;
                Index<Dim> inside = center;
                inside[axis] += lower ? 1 : -1;
                coefficient = Real(1.5) * a(center, axis) - Real(0.5) * a(inside, axis);
                distance = h / Real(2);
                const Real density_trace = boundary.trace(geometry, face, axis);
                const Real potential_trace =
                    potential_boundaries[2 * axis + (upper ? 1 : 0)].trace(geometry, face, axis);
                left_state = lower ? density_trace : q(center, 0);
                right_state = lower ? q(center, 0) : density_trace;
                left_potential = lower ? potential_trace : w(center, 0);
                right_potential = lower ? w(center, 0) : potential_trace;
              } else {
                coefficient = Real(0.5) * (a(left, axis) + a(right, axis));
                left_state = q(left, 0);
                right_state = q(right, 0);
                left_potential = w(left, 0);
                right_potential = w(right, 0);
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
              if (boundary.kind == DiffusiveBoundaryKind::conormal)
                flux = orientation * boundary.trace(geometry, face, axis);
              else {
                Index<Dim> inside = center;
                inside[axis] += lower ? 1 : -1;
                const Real coefficient = Real(1.5) * a(center, axis) - Real(0.5) * a(inside, axis);
                flux = orientation * coefficient * Real(2) *
                       (boundary.trace(geometry, face, axis) - w(center, 0)) / h;
                conductance = Real(2) * coefficient * w(center, 1) / h;
                if (!(coefficient > 0))
                  flux = std::numeric_limits<Real>::quiet_NaN();
              }
            } else {
              const Real coefficient = Real(0.5) * (a(left, axis) + a(right, axis));
              flux = coefficient * (w(right, 0) - w(left, 0)) / h;
              const Real difference = q(right, 0) - q(left, 0);
              const Real secant = difference != 0 ? (w(right, 0) - w(left, 0)) / difference
                                                  : Real(0.5) * (w(right, 1) + w(left, 1));
              conductance = coefficient * secant / h;
            }
            if constexpr (!Fitted)
              left_loss = right_loss = conductance;
            face_values(face, 0) = flux;
            face_values(face, 1) = left_loss;
            face_values(face, 2) = right_loss;
          });
        }
        const auto result = output.fab(local).view();
        const auto status = status_.fab(local).view();
        for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
          Real divergence = 0, frequency = 0;
          bool valid = true;
          for (int axis = 0; axis < Dim; ++axis) {
            Index<Dim> upper = cell;
            ++upper[axis];
            for (int side = 0; side < 2; ++side) {
              const auto face = side == 0 ? cell : upper;
              valid = valid && Kokkos::isfinite(faces.axes[axis](face, 0)) &&
                      Kokkos::isfinite(faces.axes[axis](face, 1)) &&
                      faces.axes[axis](face, 1) >= 0 &&
                      Kokkos::isfinite(faces.axes[axis](face, 2)) && faces.axes[axis](face, 2) >= 0;
            }
            divergence +=
                (faces.axes[axis](upper, 0) - faces.axes[axis](cell, 0)) / geometry.spacing(axis);
            frequency +=
                (faces.axes[axis](upper, 1) + faces.axes[axis](cell, 2)) / geometry.spacing(axis);
          }
          result(cell, 0) = divergence;
          status(cell, 0) =
              valid && Kokkos::isfinite(divergence) && Kokkos::isfinite(frequency) && frequency >= 0
                  ? frequency
                  : std::numeric_limits<Real>::infinity();
        });
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
    frequency_ = all_reduce_max(local_frequency, *lane_);
    if (!std::isfinite(frequency_))
      throw DiffusiveEvaluationError("diffusive face evaluation is invalid");
    evaluated_ = true;
  }

 public:
  Real explicit_frequency() const {
    if (!evaluated_)
      throw std::logic_error("diffusive stability requires a completed face evaluation");
    return frequency_;
  }
  /// The accepted affine quadrature supplies this weight once, after all evaluations succeed.
  /// Two cell incidences of an interior face have opposite orientation. They are retained as
  /// distinct quadrature records, so cancellation and each boundary exchange remain auditable.
  template <class Context>
  void stage_accepted_exchanges(Context& ctx, int program_block, const std::string& operation,
                                const std::string& occurrence, const std::string& evaluation,
                                Real temporal_weight, bool physical_boundary_only = false) const {
    (void)explicit_frequency();
    sync_host();
    const Field* const active = ctx.pointwise_active_mask(program_block, variable_);
    if (active != nullptr)
      sync_host();
    long active_layout_error = 0;
    if (active != nullptr) {
      if (active->local_size() != variable_.local_size())
        active_layout_error = 1;
      else
        for (std::size_t local = 0; local < variable_.local_size(); ++local)
          if (active->box(local) != variable_.box(local))
            active_layout_error = 1;
    }
    if (all_reduce_max(active_layout_error, *lane_) != 0)
      throw std::invalid_argument(
          "accepted diffusive face mask differs from local patches collectively");
    for (std::size_t local = 0; local < variable_.local_size(); ++local) {
      const auto box = variable_.box(local);
      const auto extent = box.extent();
      const auto faces = faces_[local].view();
      const auto active_values = active == nullptr ? FieldView<const Real, Dim>{}
                                                   : std::as_const(*active).fab(local).view();
      for (std::int64_t ordinal = 0; ordinal < box.numPts(); ++ordinal) {
        auto remainder = ordinal;
        Index<Dim> cell = box.lo;
        for (int axis = 0; axis < Dim; ++axis) {
          cell[axis] += static_cast<int>(remainder % extent[axis]);
          remainder /= extent[axis];
        }
        if (active != nullptr && active_values(cell, 0) < Real(0.5))
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
            ctx.stage_exchange({operation, occurrence, evaluation, identity, side == 0 ? -1 : 1,
                                measure, faces.axes[axis](face, 0), temporal_weight, 1});
          }
        }
      }
    }
  }
  const auto& faces() const { return faces_; }
  const Field& prototype() const { return variable_; }
  const auto& geometry() const { return geometry_; }
};
}  // namespace pops::runtime::program
