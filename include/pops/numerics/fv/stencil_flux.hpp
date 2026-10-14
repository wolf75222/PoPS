/// Closed source-authored Cartesian face algebra on native MultiFab storage.
#pragma once
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>
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

/// Periodic Uniform only. Stored face INTEGRALS feed both divergence and ledger.
/// Geometry, halo exchange and failure consensus are independent of the model.
template <int Dim, int Components>
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

  void consensus_(std::exception_ptr error, const char* operation) const {
    if (all_reduce_max(error ? 1L : 0L, *lane_)) {
      if (lane_->size() == 1 && error) std::rethrow_exception(error);
      throw std::runtime_error(operation);
    }
  }

 public:
  template <class Context>
  PreparedStencilFlux(Context& ctx, Field& prototype, std::array<int, Dim> ghosts)
      : geometry_(ctx.geometry()), lane_(&ctx.prepared_execution_lane()), ghosts_(ghosts) {
    long invalid = prototype.ncomp() != Components;
    // The existing accepted face ledger reads native face values on the host.
    // Refuse device-only memory until it has an explicit mirror publication.
    invalid |= !Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                          typename Field::memory_space>::accessible;
    for (int axis = 0; axis < Dim; ++axis)
      invalid |= ghosts[axis] < 0 || prototype.ghosts()[axis] < ghosts[axis]
                 || geometry_.domain().length(axis) < 2;
    if (all_reduce_max(invalid, *lane_))
      throw std::invalid_argument("stencil flux component or ghost extent mismatch collectively");
    std::exception_ptr error;
    try {
      status_ = Field(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                      1, prototype.ghosts());
      for (std::size_t local = 0; local < prototype.local_size(); ++local) {
        faces_.emplace_back(prototype.box(local), Components);
        frequencies_.emplace_back(prototype.box(local), 1);
      }
    } catch (...) { error = std::current_exception(); }
    consensus_(error, "stencil flux allocation failed collectively");
    error = nullptr;
    try {
      boundary_ = ctx.prepare_mesh_boundary_session(prototype, *lane_);
      for (int axis = 0; axis < Dim; ++axis)
        for (int side = 0; side < 2; ++side)
          if (!boundary_->topology().is_periodic(
                  Face<Dim>{axis, side == 0 ? BoundarySide::lower : BoundarySide::upper}))
            throw std::invalid_argument("stencil flux v1 requires periodic Cartesian boundaries");
    } catch (...) { error = std::current_exception(); }
    consensus_(error, "stencil flux boundary preparation failed collectively");
  }

  bool matches_preparation(const Geometry<Dim>& geometry, const ExecutionLane& lane,
                           const Field& prototype, const std::array<int, Dim>& ghosts) const {
    if (lane_ != &lane || ghosts != ghosts_ || geometry.domain() != geometry_.domain()
        || prototype.ncomp() != Components || prototype.layout() != status_.layout()
        || prototype.distribution() != status_.distribution()
        || prototype.local_rank() != status_.local_rank()
        || prototype.ghosts() != status_.ghosts()) return false;
    for (int axis = 0; axis < Dim; ++axis)
      if (geometry.spacing(axis) != geometry_.spacing(axis)
          || geometry.face_coordinate(axis, 0) != geometry_.face_coordinate(axis, 0)) return false;
    return true;
  }

  template <class Factory>
  void apply(Field& input, Field& output, Factory factory) {
    evaluated_ = false;
    long invalid = input.layout() != status_.layout() || output.layout() != status_.layout()
        || input.distribution() != status_.distribution()
        || output.distribution() != status_.distribution()
        || input.local_rank() != status_.local_rank() || output.local_rank() != status_.local_rank()
        || input.ncomp() != Components || output.ncomp() != Components
        || input.ghosts() != status_.ghosts() || input.shares_storage_with(output);
    if (all_reduce_max(invalid, *lane_))
      throw std::invalid_argument("stencil flux input/output authority mismatch collectively");
    std::exception_ptr error;
    try { boundary_->fill_halo(input); }
    catch (...) { error = std::current_exception(); }
    consensus_(error, "stencil flux halo preparation failed collectively");
    const auto geometry = geometry_;
    error = nullptr;
    try {
      for (std::size_t local = 0; local < input.local_size(); ++local) {
        const auto evaluate = factory(local);
        const auto faces = faces_[local].view();
        const auto frequencies = frequencies_[local].view();
        for (int axis = 0; axis < Dim; ++axis) {
          Real area = Real(1);
          for (int d = 0; d < Dim; ++d) if (d != axis) area *= geometry.spacing(d);
          const auto flux = faces.axes[axis];
          const auto bound = frequencies.axes[axis];
          for_each_cell(nd::face_box(input.box(local), axis), [=] POPS_HD(const Index<Dim>& face) {
            const auto evaluated = evaluate(face, axis);
            bool valid = evaluated.valid && Kokkos::isfinite(evaluated.frequency)
                         && evaluated.frequency >= Real(0);
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
        for (int d = 0; d < Dim; ++d) volume *= geometry.spacing(d);
        for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
          Real frequency = 0;
          for (int component = 0; component < Components; ++component) {
            Real net = 0;
            for (int axis = 0; axis < Dim; ++axis) {
              Index<Dim> upper = cell; ++upper[axis];
              net += faces.axes[axis](cell, component) - faces.axes[axis](upper, component);
              if (component == 0)
                frequency += frequencies.axes[axis](cell, 0) + frequencies.axes[axis](upper, 0);
            }
            result(cell, component) = net / volume;
            if (!Kokkos::isfinite(net / volume)) frequency = std::numeric_limits<Real>::infinity();
          }
          status(cell, 0) = frequency;
        });
      }
      device_fence();
    } catch (...) {
      error = std::current_exception();
      try { device_fence(); } catch (...) {}
    }
    consensus_(error, "stencil flux evaluation failed collectively");
    Real local_frequency = 0;
    error = nullptr;
    try { local_frequency = reduce_max_local(status_); }
    catch (...) { error = std::current_exception(); }
    consensus_(error, "stencil flux reduction failed collectively");
    frequency_ = all_reduce_max(local_frequency, *lane_);
    if (!std::isfinite(frequency_) || frequency_ < Real(0))
      throw std::runtime_error("stencil flux has a nonfinite expression or invalid frequency bound");
    evaluated_ = true;
  }

  Real explicit_frequency() const {
    if (!evaluated_) throw std::logic_error("stencil frequency requires a completed evaluation");
    return frequency_;
  }
  const auto& faces() const { return faces_; }
};

/// A typed solved-field sample gets its own bounded periodic halo. InputIndex
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
      if (lane_->size() == 1 && error) std::rethrow_exception(error);
      throw std::runtime_error(operation);
    }
  }
 public:
  template <class Context>
  PreparedStencilInput(Context& ctx, const Field& state, const Field& input,
                       std::array<int, Dim> ghosts)
      : geometry_(ctx.geometry()), lane_(&ctx.prepared_execution_lane()), ghosts_(ghosts) {
    long invalid = input.ncomp() < 1 || input.layout() != state.layout()
        || input.distribution() != state.distribution() || input.local_rank() != state.local_rank();
    for (int axis = 0; axis < Dim; ++axis)
      invalid |= ghosts[axis] < 0 || geometry_.domain().length(axis) < 2;
    if (all_reduce_max(invalid, *lane_))
      throw std::invalid_argument("stencil field co-location mismatch collectively");
    std::exception_ptr error;
    try {
      Extent<Dim> extent;
      for (int axis = 0; axis < Dim; ++axis) extent[axis] = ghosts[axis];
      sampled_ = Field(state.layout(), state.distribution(), state.local_rank(), input.ncomp(), extent);
    } catch (...) { error = std::current_exception(); }
    consensus_(error, "stencil field allocation failed collectively");
    error = nullptr;
    try {
      boundary_ = ctx.prepare_mesh_boundary_session(sampled_, *lane_);
      for (int axis = 0; axis < Dim; ++axis)
        for (int side = 0; side < 2; ++side)
          if (!boundary_->topology().is_periodic(
                  Face<Dim>{axis, side == 0 ? BoundarySide::lower : BoundarySide::upper}))
            throw std::invalid_argument("stencil field requires periodic boundaries");
    } catch (...) { error = std::current_exception(); }
    consensus_(error, "stencil field halo preparation failed collectively");
  }
  bool matches_preparation(const Geometry<Dim>& geometry, const ExecutionLane& lane,
                           const Field& state, const Field& input,
                           const std::array<int, Dim>& ghosts) const {
    if (lane_ != &lane || ghosts != ghosts_ || geometry.domain() != geometry_.domain()
        || state.layout() != sampled_.layout() || state.distribution() != sampled_.distribution()
        || state.local_rank() != sampled_.local_rank() || input.layout() != sampled_.layout()
        || input.distribution() != sampled_.distribution() || input.local_rank() != sampled_.local_rank()
        || input.ncomp() != sampled_.ncomp()) return false;
    for (int axis = 0; axis < Dim; ++axis)
      if (geometry.spacing(axis) != geometry_.spacing(axis)
          || geometry.face_coordinate(axis, 0) != geometry_.face_coordinate(axis, 0)) return false;
    return true;
  }
  const Field& sample(const Field& input) {
    long invalid = input.layout() != sampled_.layout() || input.distribution() != sampled_.distribution()
        || input.local_rank() != sampled_.local_rank() || input.ncomp() != sampled_.ncomp();
    if (all_reduce_max(invalid, *lane_))
      throw std::invalid_argument("stencil field source authority mismatch collectively");
    std::exception_ptr error;
    try {
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
      try { device_fence(); } catch (...) {}
    }
    consensus_(error, "stencil field copy failed collectively");
    error = nullptr;
    try { boundary_->fill_halo(sampled_); }
    catch (...) { error = std::current_exception(); }
    consensus_(error, "stencil field halo fill failed collectively");
    return sampled_;
  }
};
}  // namespace pops::runtime::program
