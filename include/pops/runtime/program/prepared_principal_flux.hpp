/// Complete principal finite-volume evaluation on a shared native state carrier.
#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/prepared_metric_provider.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/finite_volume.hpp>
#include <pops/numerics/spatial/nd/reconstruction.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>
#include <exception>
#include <memory>
#include <vector>

namespace pops::runtime::program {

template <int Dim, int Components>
class PreparedPrincipalFlux {
  using Field = MultiFab<Dim>;
  using Metric = PreparedMappedMetricProvider<CartesianCoordinateMap<Dim>>;
  Geometry<Dim> geometry_;
  Metric metric_;
  const ExecutionLane* lane_ = nullptr;
  Field status_, candidate_;
  std::vector<nd::FaceField<Dim>> flux_, bound_;
  std::shared_ptr<PreparedScalarBoundarySession<Dim>> boundary_;
  Real frequency_ = Real(0);
  mutable bool evaluated_ = false;

  static Metric make_metric_(const Geometry<Dim>& geometry) {
    RealVector<Dim> lengths{};
    for (int axis = 0; axis < Dim; ++axis)
      lengths[axis] = geometry.upper()[axis] - geometry.lower()[axis];
    return Metric::prepare(geometry.domain(),
                           CartesianCoordinateMap<Dim>::make(geometry.lower(), lengths));
  }

  bool same_layout_(const Field& field) const {
    return field.layout() == candidate_.layout() &&
           field.distribution() == candidate_.distribution() &&
           field.local_rank() == candidate_.local_rank() &&
           field.local_size() == candidate_.local_size() &&
           field.ncomp() == Components && field.ghosts() == candidate_.ghosts();
  }

  void collective_error_(std::exception_ptr error, const char* message) const {
    if (all_reduce_max(error ? 1L : 0L, *lane_) == 0)
      return;
    if (lane_->size() == 1 && error)
      std::rethrow_exception(error);
    throw std::runtime_error(message);
  }

  template <int Axis, class Model, class Reconstruction, class Numerical>
  void faces_(const Field& input, Model model, Reconstruction reconstruction, Numerical numerical) {
    const auto metric = metric_;
    for (std::size_t local = 0; local < input.local_size(); ++local) {
      const auto state = input.fab(local).view();
      const auto flux = flux_[local].view().template axis<Axis>();
      const auto bound = bound_[local].view().template axis<Axis>();
      for_each_face<Axis>(input.box(local), [=] POPS_HD(const FaceIndex<Dim, Axis>& face) {
        const auto pair = nd::reconstruct_face_pair<Axis>(model, state, face, reconstruction);
        const Real invalid = std::numeric_limits<Real>::quiet_NaN();
        Real speed = invalid;
        typename Model::State integrated{};
        for (int component = 0; component < Components; ++component)
          integrated[component] = invalid;
        if (pair.succeeded()) {
          // Positive-axis face density is integrated once; both cell incidences read it.
          const auto evaluation = nd::evaluate_axis_flux<Axis>(numerical, model, pair.left, pair.right);
          if (evaluation.succeeded()) {
            Index<Dim> adjacent = face.coordinate;
            const Real measure = nd::metric_face_context<Axis, MetricFaceSide::Lower>(
                metric, adjacent).face_measure;
            const auto density = evaluation.checked_density().value;
            speed = evaluation.stability.value;
            for (int component = 0; component < Components; ++component)
              integrated[component] = density[component] * measure;
          }
        }
        for (int component = 0; component < Components; ++component)
          flux(face.coordinate, component) = integrated[component];
        bound(face.coordinate, 0) = speed;
      });
    }
    if constexpr (Axis + 1 < Dim)
      faces_<Axis + 1>(input, model, reconstruction, numerical);
  }

 public:
  template <std::size_t Count>
  void pack(const std::array<Field*, Count>& inputs, Field& packed,
            const std::array<int, Count>& component_counts) const {
    evaluated_ = false;
    long invalid = !same_layout_(packed);
    int total = 0;
    std::size_t slot = 0;
    for (const auto* input : inputs) {
      invalid |= input == nullptr;
      const int expected = component_counts[slot++];
      if (!input)
        continue;
      invalid |= input->ncomp() != expected;
      total += input->ncomp();
      invalid |= input->layout() != packed.layout() ||
                 input->distribution() != packed.distribution() ||
                 input->local_rank() != packed.local_rank();
    }
    invalid |= total != Components;
    if (all_reduce_max(invalid, *lane_) != 0)
      throw std::invalid_argument("principal inputs do not share their complete group layout");
    std::exception_ptr error;
    try {
      int offset = 0;
      for (const auto* input : inputs) {
        const int count = input->ncomp(), first = offset;
        for (std::size_t local = 0; local < packed.local_size(); ++local) {
          const auto source = input->fab(local).view();
          const auto target = packed.fab(local).view();
          for_each_cell(packed.box(local), [=] POPS_HD(const Index<Dim>& cell) {
            for (int component = 0; component < count; ++component)
              target(cell, first + component) = source(cell, component);
          });
        }
        offset += count;
      }
      device_fence();
    } catch (...) { error = std::current_exception(); }
    collective_error_(error, "principal input packing failed collectively");
  }

  template <std::size_t Count>
  void publish(const std::array<Field*, Count>& outputs,
               const std::array<int, Count>& component_counts) const {
    long invalid = !evaluated_;
    int total = 0;
    std::size_t slot = 0;
    for (auto* output : outputs) {
      invalid |= output == nullptr;
      const int expected = component_counts[slot++];
      if (!output)
        continue;
      invalid |= output->ncomp() != expected;
      total += output->ncomp();
      invalid |= output->layout() != candidate_.layout() ||
                 output->distribution() != candidate_.distribution() ||
                 output->local_rank() != candidate_.local_rank();
    }
    invalid |= total != Components;
    if (all_reduce_max(invalid, *lane_) != 0)
      throw std::invalid_argument("principal outputs do not share their complete group layout");
    std::exception_ptr error;
    try {
      int offset = 0;
      for (auto* output : outputs) {
        const int count = output->ncomp(), first = offset;
        for (std::size_t local = 0; local < candidate_.local_size(); ++local) {
          const auto source = std::as_const(candidate_).fab(local).view();
          const auto target = output->fab(local).view();
          for_each_cell(output->box(local), [=] POPS_HD(const Index<Dim>& cell) {
            for (int component = 0; component < count; ++component)
              target(cell, component) = source(cell, first + component);
          });
        }
        offset += count;
      }
      device_fence();
    } catch (...) { error = std::current_exception(); }
    collective_error_(error, "principal output publication failed collectively");
  }

  template <class Context>
  PreparedPrincipalFlux(Context& ctx, Field& prototype)
      : geometry_(ctx.geometry()), metric_(make_metric_(geometry_)),
        lane_(&ctx.prepared_execution_lane()) {
    std::exception_ptr error;
    try {
      if (prototype.ncomp() != Components)
        throw std::invalid_argument("principal carrier component count differs from its signature");
      candidate_ = Field(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                         Components, prototype.ghosts());
      status_ = Field(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                      1, prototype.ghosts());
      flux_.reserve(prototype.local_size());
      bound_.reserve(prototype.local_size());
      for (std::size_t local = 0; local < prototype.local_size(); ++local) {
        flux_.emplace_back(prototype.box(local), Components);
        bound_.emplace_back(prototype.box(local), 1);
      }
    } catch (...) {
      error = std::current_exception();
    }
    collective_error_(error, "principal preparation allocation failed collectively");
    boundary_ = ctx.prepare_mesh_boundary_session(prototype, *lane_);
    long unsupported_boundary = 0;
    for (int axis = 0; axis < Dim; ++axis)
      for (auto side : {BoundarySide::lower, BoundarySide::upper})
        unsupported_boundary |= !boundary_->topology().is_periodic(Face<Dim>{axis, side});
    if (all_reduce_max(unsupported_boundary, *lane_) != 0)
      throw std::invalid_argument(
          "principal group requires periodic boundaries until its joint physical boundary adapter is prepared");
  }

  bool matches_preparation(const Geometry<Dim>& geometry, const ExecutionLane& lane,
                           const Field& prototype) const {
    if (lane_ != &lane || geometry_.domain() != geometry.domain() || !same_layout_(prototype))
      return false;
    for (int axis = 0; axis < Dim; ++axis)
      if (geometry_.spacing(axis) != geometry.spacing(axis) ||
          geometry_.face_coordinate(axis, 0) != geometry.face_coordinate(axis, 0))
        return false;
    return true;
  }

  template <class Model, class Reconstruction, class Numerical>
  const Field& evaluate(Field& input, Model model, Reconstruction reconstruction, Numerical numerical) {
    evaluated_ = false;
    static_assert(Model::dimension == Dim && Model::n_vars == Components);
    long invalid = !same_layout_(input);
    for (int axis = 0; axis < Dim; ++axis)
      invalid |= input.ghosts()[axis] < Reconstruction::n_ghost;
    if (all_reduce_max(invalid, *lane_) != 0)
      throw std::invalid_argument("principal input layout or halo differs from its prepared contract");
    boundary_->fill_halo(input);
    std::exception_ptr error;
    try {
      faces_<0>(std::as_const(input), model, reconstruction, numerical);
      const auto metric = metric_;
      const auto geometry = geometry_;
      for (std::size_t local = 0; local < input.local_size(); ++local) {
        const auto faces = std::as_const(flux_[local]).view();
        const auto bounds = std::as_const(bound_[local]).view();
        const auto result = candidate_.fab(local).view();
        const auto status = status_.fab(local).view();
        for_each_cell(input.box(local), [=] POPS_HD(const Index<Dim>& cell) {
          const auto residual = nd::conservative_residual<Components>(metric, faces, cell);
          bool valid = residual.succeeded();
          Real frequency = Real(0);
          for (int axis = 0; axis < Dim; ++axis) {
            auto upper = cell;
            ++upper[axis];
            const Real lower_speed = bounds.axes[axis](cell, 0);
            const Real upper_speed = bounds.axes[axis](upper, 0);
            valid = valid && Kokkos::isfinite(lower_speed) && Kokkos::isfinite(upper_speed) &&
                    lower_speed >= Real(0) && upper_speed >= Real(0);
            frequency += Kokkos::max(lower_speed, upper_speed) / geometry.spacing(axis);
          }
          for (int component = 0; component < Components; ++component)
            result(cell, component) = residual.value[component];
          status(cell, 0) = valid && Kokkos::isfinite(frequency)
                               ? frequency : std::numeric_limits<Real>::infinity();
        });
      }
      device_fence();
    } catch (...) {
      error = std::current_exception();
      try { device_fence(); } catch (...) {}
    }
    collective_error_(error, "principal face evaluation failed collectively");
    Real local_frequency = 0;
    error = nullptr;
    try { local_frequency = reduce_max_local(status_); }
    catch (...) { error = std::current_exception(); }
    collective_error_(error, "principal stability reduction failed collectively");
    frequency_ = all_reduce_max(local_frequency, *lane_);
    if (!std::isfinite(frequency_))
      throw std::runtime_error("principal face evaluation has invalid state, flux or group bound");
    evaluated_ = true;
    return candidate_;
  }

  Real explicit_frequency() const {
    if (!evaluated_)
      throw std::logic_error("principal stability requires a completed group evaluation");
    return frequency_;
  }
};
}  // namespace pops::runtime::program
