/// Closed source-authored Cartesian face algebra on native MultiFab storage.
#pragma once
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/static_geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>
#include <pops/numerics/elliptic/nd/general_field_operator.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace pops::runtime::program {
template <int Components>
struct StencilFaceResult {
  std::array<Real, Components> flux{};
  Real frequency = Real(0);  // One incident contribution, units 1/time.
  bool valid = true;
};

/// Stored face INTEGRALS feed both divergence and the accepted flux ledger.
/// Geometry, halo exchange and failure consensus are independent of the model.
///
/// With @p StaticArrays the faces read the static geometry arrays of node positions (N17, C1):
/// the face of axis a is the face of a uniform box whose spacing along a is the distance across
/// it, which the generated body reads as its spacing (static_arrays()), and its integral and the
/// divergence take the face and cell measures of the arrays. The incident frequency f of a face
/// bounds |dF/du| / h on that box, so f d is a face conductance, and each incident cell K reads
/// f d |S| / |K|. A chart, whose faces need not be orthogonal to the line of the centroids, is
/// refused at construction.
template <int Dim, int Components, bool StaticArrays = false>
class PreparedStencilFlux {
  using Field = MultiFab<Dim>;
  Geometry<Dim> geometry_;
  const ExecutionLane* lane_;
  std::array<int, Dim> ghosts_;
  Field status_;
  std::shared_ptr<PreparedScalarBoundarySession<Dim>> boundary_;
  std::vector<nd::FaceField<Dim>> faces_, frequencies_;
  Real frequency_ = Real(0);
  bool evaluated_ = false;
  // The static geometry arrays that the faces read, with their owner (StaticArrays only).
  std::shared_ptr<const StaticGeometry<Dim>> static_geometry_;
  StaticGeometryView<Dim> arrays_{};

  void consensus_(std::exception_ptr error, const char* operation) const {
    if (all_reduce_max(error ? 1L : 0L, *lane_)) {
      if (lane_->size() == 1 && error)
        std::rethrow_exception(error);
      throw std::runtime_error(operation);
    }
  }

 public:
  template <class Context>
  PreparedStencilFlux(Context& ctx, Field& prototype, std::array<int, Dim> ghosts)
      : geometry_(ctx.geometry()), lane_(&ctx.prepared_execution_lane()), ghosts_(ghosts) {
    long invalid = prototype.ncomp() != Components;
    for (int axis = 0; axis < Dim; ++axis)
      invalid |= ghosts[axis] < 0 || prototype.ghosts()[axis] < ghosts[axis] ||
                 geometry_.domain().length(axis) < 2;
    if (all_reduce_max(invalid, *lane_))
      throw std::invalid_argument("stencil flux component or ghost extent mismatch collectively");
    std::exception_ptr error;
    try {
      status_ = Field(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
                      prototype.ghosts());
      for (std::size_t local = 0; local < prototype.local_size(); ++local) {
        faces_.emplace_back(prototype.box(local), Components);
        frequencies_.emplace_back(prototype.box(local), 1);
      }
    } catch (...) {
      error = std::current_exception();
    }
    consensus_(error, "stencil flux allocation failed collectively");
    error = nullptr;
    try {
      boundary_ = ctx.prepare_mesh_boundary_session(prototype, *lane_);
    } catch (...) {
      error = std::current_exception();
    }
    consensus_(error, "stencil flux boundary preparation failed collectively");
    if constexpr (StaticArrays) {
      error = nullptr;
      try {
        static_geometry_ = ctx.static_geometry();
        if (!static_geometry_ || static_geometry_->kind() != StaticGeometryKind::nodes ||
            static_geometry_->domain() != geometry_.domain())
          throw std::invalid_argument(
              "stencil flux on static geometry arrays requires the node positions of its layout");
        arrays_ = static_geometry_->view();
      } catch (...) {
        error = std::current_exception();
      }
      consensus_(error, "stencil flux static geometry preparation failed collectively");
    }
  }

  /// The arrays that a generated face of a mapped layout reads: the distance across the face of
  /// axis a is its spacing along a.
  const StaticGeometryView<Dim>& static_arrays() const {
    static_assert(StaticArrays, "a stencil flux of the logical box reads its uniform spacings");
    return arrays_;
  }

  bool matches_preparation(const Geometry<Dim>& geometry, const ExecutionLane& lane,
                           const Field& prototype, const std::array<int, Dim>& ghosts) const {
    if (lane_ != &lane || ghosts != ghosts_ || geometry.domain() != geometry_.domain() ||
        prototype.ncomp() != Components || prototype.layout() != status_.layout() ||
        prototype.distribution() != status_.distribution() ||
        prototype.local_rank() != status_.local_rank() || prototype.ghosts() != status_.ghosts())
      return false;
    for (int axis = 0; axis < Dim; ++axis)
      if (geometry.spacing(axis) != geometry_.spacing(axis) ||
          geometry.face_coordinate(axis, 0) != geometry_.face_coordinate(axis, 0))
        return false;
    return true;
  }

  template <class Factory>
  void apply(Field& input, Field& output, Factory factory, bool halo_prepared = false) {
    evaluated_ = false;
    long invalid = input.layout() != status_.layout() || output.layout() != status_.layout() ||
                   input.distribution() != status_.distribution() ||
                   output.distribution() != status_.distribution() ||
                   input.local_rank() != status_.local_rank() ||
                   output.local_rank() != status_.local_rank() || input.ncomp() != Components ||
                   output.ncomp() != Components || input.ghosts() != status_.ghosts() ||
                   input.shares_storage_with(output);
    if (all_reduce_max(invalid, *lane_))
      throw std::invalid_argument("stencil flux input/output authority mismatch collectively");
    std::exception_ptr error;
    try {
      if (!halo_prepared) {
        for (int axis = 0; axis < Dim; ++axis)
          for (int side = 0; side < 2; ++side)
            if (!boundary_->topology().is_periodic(
                    Face<Dim>{axis, side == 0 ? BoundarySide::lower : BoundarySide::upper}))
              throw std::invalid_argument(
                  "nonperiodic stencil requires the exact block boundary provider");
        boundary_->fill_halo(input);
      }
    } catch (...) {
      error = std::current_exception();
    }
    consensus_(error, "stencil flux halo preparation failed collectively");
    const auto geometry = geometry_;
    error = nullptr;
    try {
      for (std::size_t local = 0; local < input.local_size(); ++local) {
        const auto evaluate = factory(local);
        const auto faces = faces_[local].view();
        const auto frequencies = frequencies_[local].view();
        if constexpr (StaticArrays) {  // the measures of the arrays, below the uniform box
          apply_static_(input.box(local), output.fab(local).view(), status_.fab(local).view(),
                        evaluate, faces, frequencies);
          continue;
        }
        for (int axis = 0; axis < Dim; ++axis) {
          Real area = Real(1);
          for (int d = 0; d < Dim; ++d)
            if (d != axis)
              area *= geometry.spacing(d);
          const auto flux = faces.axes[axis];
          const auto bound = frequencies.axes[axis];
          for_each_cell(nd::face_box(input.box(local), axis), [=] POPS_HD(const Index<Dim>& face) {
            const auto evaluated = evaluate(face, axis);
            bool valid = evaluated.valid && Kokkos::isfinite(evaluated.frequency) &&
                         evaluated.frequency >= Real(0);
            for (int component = 0; component < Components; ++component) {
              const Real integral = area * evaluated.flux[component];
              valid = valid && Kokkos::isfinite(integral);
              flux(face, component) = integral;
            }
            bound(face, 0) = valid ? evaluated.frequency : std::numeric_limits<Real>::infinity();
          });
        }
        const auto result = output.fab(local).view();
        const auto status = status_.fab(local).view();
        Real volume = Real(1);
        for (int d = 0; d < Dim; ++d)
          volume *= geometry.spacing(d);
        for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
          Real frequency = 0;
          for (int component = 0; component < Components; ++component) {
            Real net = 0;
            for (int axis = 0; axis < Dim; ++axis) {
              Index<Dim> upper = cell;
              ++upper[axis];
              net += faces.axes[axis](cell, component) - faces.axes[axis](upper, component);
              if (component == 0)
                frequency += frequencies.axes[axis](cell, 0) + frequencies.axes[axis](upper, 0);
            }
            result(cell, component) = net / volume;
            if (!Kokkos::isfinite(net / volume))
              frequency = std::numeric_limits<Real>::infinity();
          }
          status(cell, 0) = frequency;
        });
      }
      device_fence();
    } catch (...) {
      error = std::current_exception();
      try {
        device_fence();
      } catch (...) {
      }
    }
    consensus_(error, "stencil flux evaluation failed collectively");
    Real local_frequency = 0;
    error = nullptr;
    try {
      local_frequency = reduce_max_local(status_);
    } catch (...) {
      error = std::current_exception();
    }
    consensus_(error, "stencil flux reduction failed collectively");
    frequency_ = all_reduce_max(local_frequency, *lane_);
    if (!std::isfinite(frequency_) || frequency_ < Real(0))
      throw std::runtime_error(
          "stencil flux has a nonfinite expression or invalid frequency bound");
    evaluated_ = true;
  }

  Real explicit_frequency() const {
    if (!evaluated_)
      throw std::logic_error("stencil frequency requires a completed evaluation");
    return frequency_;
  }
  const auto& faces() const { return faces_; }

 private:
  /// The faces and the balance of one patch on the arrays: the integral |S| F of each face, the
  /// face conductance f d |S| of its incident frequency, and in each cell the net integral and
  /// the sum of the conductances of its faces, both over |K|.
  template <class Evaluate, class FaceView, class ResultView, class StatusView>
  void apply_static_(const Box<Dim>& cells, ResultView result, StatusView status,
                     const Evaluate& evaluate, const FaceView& faces,
                     const FaceView& frequencies) const {
    const auto arrays = arrays_;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto flux = faces.axes[axis];
      const auto bound = frequencies.axes[axis];
      for_each_cell(nd::face_box(cells, axis), [=] POPS_HD(const Index<Dim>& face) {
        const auto evaluated = evaluate(face, axis);
        const Real area = arrays.face_measure_at(axis, face);
        const Real conductance = evaluated.frequency * arrays.distance_at(axis, face) * area;
        bool valid = evaluated.valid && Kokkos::isfinite(evaluated.frequency) &&
                     evaluated.frequency >= Real(0) && Kokkos::isfinite(conductance);
        for (int component = 0; component < Components; ++component) {
          const Real integral = area * evaluated.flux[component];
          valid = valid && Kokkos::isfinite(integral);
          flux(face, component) = integral;
        }
        bound(face, 0) = valid ? conductance : std::numeric_limits<Real>::infinity();
      });
    }
    for_each_cell(cells, [=] POPS_HD(const Index<Dim>& cell) {
      const Real volume = arrays.measure(cell);
      Real conductance = 0;
      for (int axis = 0; axis < Dim; ++axis) {
        Index<Dim> upper = cell;
        ++upper[axis];
        conductance += frequencies.axes[axis](cell, 0) + frequencies.axes[axis](upper, 0);
      }
      Real frequency = conductance / volume;
      for (int component = 0; component < Components; ++component) {
        Real net = 0;
        for (int axis = 0; axis < Dim; ++axis) {
          Index<Dim> upper = cell;
          ++upper[axis];
          net += faces.axes[axis](cell, component) - faces.axes[axis](upper, component);
        }
        result(cell, component) = net / volume;
        if (!Kokkos::isfinite(net / volume))
          frequency = std::numeric_limits<Real>::infinity();
      }
      status(cell, 0) = Kokkos::isfinite(frequency) && frequency >= Real(0)
                            ? frequency
                            : std::numeric_limits<Real>::infinity();
    });
  }
};

/// A typed field sample gets an explicitly prepared boundary extension. InputIndex
/// distinguishes resource-cache slots for several fields of one face operation.
template <int Dim, int InputIndex>
class PreparedStencilInput {
  using Field = MultiFab<Dim>;
  Geometry<Dim> geometry_;
  const ExecutionLane* lane_;
  std::array<int, Dim> ghosts_;
  Field sampled_;
  std::shared_ptr<PreparedScalarBoundarySession<Dim>> boundary_;
  void consensus_(std::exception_ptr error, const char* operation) const {
    if (all_reduce_max(error ? 1L : 0L, *lane_)) {
      if (lane_->size() == 1 && error)
        std::rethrow_exception(error);
      throw std::runtime_error(operation);
    }
  }

 public:
  template <class Context>
  PreparedStencilInput(Context& ctx, const Field& state, const Field& input,
                       std::array<int, Dim> ghosts)
      : geometry_(ctx.geometry()), lane_(&ctx.prepared_execution_lane()), ghosts_(ghosts) {
    long invalid = input.ncomp() < 1 || input.layout() != state.layout() ||
                   input.distribution() != state.distribution() ||
                   input.local_rank() != state.local_rank();
    for (int axis = 0; axis < Dim; ++axis)
      invalid |= ghosts[axis] < 0 || geometry_.domain().length(axis) < 2;
    if (all_reduce_max(invalid, *lane_))
      throw std::invalid_argument("stencil field co-location mismatch collectively");
    std::exception_ptr error;
    try {
      Extent<Dim> extent;
      for (int axis = 0; axis < Dim; ++axis)
        extent[axis] = ghosts[axis];
      sampled_ =
          Field(state.layout(), state.distribution(), state.local_rank(), input.ncomp(), extent);
    } catch (...) {
      error = std::current_exception();
    }
    consensus_(error, "stencil field allocation failed collectively");
    error = nullptr;
    try {
      boundary_ = ctx.prepare_mesh_boundary_session(sampled_, *lane_);
    } catch (...) {
      error = std::current_exception();
    }
    consensus_(error, "stencil field halo preparation failed collectively");
  }
  bool matches_preparation(const Geometry<Dim>& geometry, const ExecutionLane& lane,
                           const Field& state, const Field& input,
                           const std::array<int, Dim>& ghosts) const {
    if (lane_ != &lane || ghosts != ghosts_ || geometry.domain() != geometry_.domain() ||
        state.layout() != sampled_.layout() || state.distribution() != sampled_.distribution() ||
        state.local_rank() != sampled_.local_rank() || input.layout() != sampled_.layout() ||
        input.distribution() != sampled_.distribution() ||
        input.local_rank() != sampled_.local_rank() || input.ncomp() != sampled_.ncomp())
      return false;
    for (int axis = 0; axis < Dim; ++axis)
      if (geometry.spacing(axis) != geometry_.spacing(axis) ||
          geometry.face_coordinate(axis, 0) != geometry_.face_coordinate(axis, 0))
        return false;
    return true;
  }
  /// Uniform solved fields use the exact declared physical field boundary.
  /// Sparse AMR inputs use sample_extension after their provider reconstructs CF.
  const Field& sample_uniform(
      const Field& input,
      const std::array<elliptic::nd::PhysicalFieldBoundary, 2 * Dim>& physical) {
    long invalid = input.layout() != sampled_.layout() ||
                   input.distribution() != sampled_.distribution() ||
                   input.local_rank() != sampled_.local_rank() || input.ncomp() != sampled_.ncomp();
    if (all_reduce_max(invalid, *lane_))
      throw std::invalid_argument("stencil field source authority mismatch collectively");
    std::exception_ptr error;
    try {
      elliptic::nd::require_field_boundary(*boundary_, physical);
      const int width = sampled_.ncomp();
      for (std::size_t local = 0; local < sampled_.local_size(); ++local) {
        const auto source = input.fab(local).view();
        const auto target = sampled_.fab(local).view();
        for_each_cell(sampled_.box(local), [=] POPS_HD(const Index<Dim>& cell) {
          for (int component = 0; component < width; ++component)
            target(cell, component) = source(cell, component);
        });
      }
      device_fence();
    } catch (...) {
      error = std::current_exception();
      try {
        device_fence();
      } catch (...) {
      }
    }
    consensus_(error, "stencil field copy failed collectively");
    error = nullptr;
    try {
      boundary_->fill(sampled_);
    } catch (...) {
      error = std::current_exception();
    }
    consensus_(error, "stencil field halo fill failed collectively");
    return sampled_;
  }

  /// Extract only after the generated caller obtains its source-owned extension.
  /// This routine neither infers a CF law nor repairs an unauthenticated halo.
  const Field& sample_extension(const Field& input, int first_component = 0) {
    long invalid =
        input.layout() != sampled_.layout() || input.distribution() != sampled_.distribution() ||
        input.local_rank() != sampled_.local_rank() || first_component < 0 ||
        first_component > input.ncomp() || sampled_.ncomp() > input.ncomp() - first_component;
    for (int axis = 0; axis < Dim; ++axis)
      invalid |= input.ghosts()[axis] < ghosts_[axis];
    if (all_reduce_max(invalid, *lane_))
      throw std::invalid_argument("stencil source extension does not cover its declared samples");
    std::exception_ptr error;
    try {
      const int width = sampled_.ncomp();
      for (std::size_t local = 0; local < sampled_.local_size(); ++local) {
        const auto source = input.fab(local).view();
        const auto target = sampled_.fab(local).view();
        for_each_cell(sampled_.fab(local).grown_box(), [=] POPS_HD(const Index<Dim>& cell) {
          for (int component = 0; component < width; ++component)
            target(cell, component) = source(cell, first_component + component);
        });
      }
      device_fence();
    } catch (...) {
      error = std::current_exception();
      try {
        device_fence();
      } catch (...) {
      }
    }
    consensus_(error, "stencil authenticated extension extraction failed collectively");
    return sampled_;
  }
};
}  // namespace pops::runtime::program
