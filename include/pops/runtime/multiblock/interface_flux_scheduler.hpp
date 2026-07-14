#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/runtime/config/generated_component_abi.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::multiblock {

/// Exact identity of one residual evaluation.  Both sides of an interface are assembled by the same
/// scheduler call and therefore observe this exact same point; no side reconstructs time from a
/// rounded physical value.
struct BoundaryEvaluationPoint {
  std::string clock;
  std::int64_t tick = 0;
  int level = 0;
  int substep = 0;
  int stage = 0;
  ::pops::amr::Rational stage_fraction{0, 1};
  double dt = std::numeric_limits<double>::quiet_NaN();
  double physical_time = std::numeric_limits<double>::quiet_NaN();

  friend bool operator==(const BoundaryEvaluationPoint&, const BoundaryEvaluationPoint&) = default;
};

enum class InterfaceAxis { X, Y };
enum class InterfaceSide { Low, High };
enum class TangentialOrientation { Aligned, Reversed };

/// The deliberately narrow first production route: two opposite, axis-aligned faces with equal
/// normal/tangential discretisation.  right_component_for_left is an explicit bijection from the
/// canonical (left) flux component order to storage on the right block.  The numerical flux is
/// defined positive OUTWARD from the left block.
struct AxisAlignedInterface {
  std::string identity;
  std::size_t left_block = 0;
  std::size_t right_block = 0;
  int level = 0;
  InterfaceAxis left_axis = InterfaceAxis::X;
  InterfaceAxis right_axis = InterfaceAxis::X;
  InterfaceSide left_side = InterfaceSide::High;
  InterfaceSide right_side = InterfaceSide::Low;
  TangentialOrientation tangential_orientation = TangentialOrientation::Aligned;
  std::vector<int> right_component_for_left;
  // Optional authenticated affine map from right physical coordinates into the left frame.  Empty
  // identity means the faces must coincide directly and all three values must remain their identity
  // defaults.  A non-empty identity makes a translated/reversed topology explicit rather than
  // silently connecting two merely equal-sized but physically unrelated faces.
  std::string affine_mapping_identity;
  Real right_normal_translation = Real(0);
  Real right_tangential_scale = Real(1);
  Real right_tangential_offset = Real(0);
};

/// One complete prepared-interface batch.  The scheduler packs both boundary traces in canonical
/// left-component/tangential order, calls the evaluator ONCE, then scatters the returned shared flux.
/// This POD-shaped view is also the sole hook a future NumericalFlux component-ABI adapter must fill.
struct InterfaceFluxBatch {
  const Real* left_state = nullptr;
  const Real* right_state = nullptr;
  Real* shared_flux = nullptr;
  int face_count = 0;
  int component_count = 0;
};

using InterfaceFluxEvaluator =
    std::function<void(const BoundaryEvaluationPoint&, const InterfaceFluxBatch&)>;
using InterfaceFluxEvaluatorFactory = std::function<InterfaceFluxEvaluator()>;

class InterfaceFluxScheduler {
 public:
  /// Prepare and install one supported route.  Layout, component permutation, face orientation and
  /// equal discretisation are all proved here, before any residual evaluation can begin.
  void install(AxisAlignedInterface route, MultiFab& left_state, const Geometry& left_geometry,
               MultiFab& right_state, const Geometry& right_geometry,
               const PopsExecutionContextV1& execution,
               InterfaceFluxEvaluatorFactory evaluator_factory) {
    if (route.identity.empty() || route.left_block == route.right_block || route.level < 0)
      throw std::invalid_argument("multi-block interface identity/ownership is invalid");
    if (!evaluator_factory)
      throw std::invalid_argument(
          "multi-block interface has no numerical-flux evaluator factory");
    if (route.left_axis != route.right_axis)
      throw std::invalid_argument("multi-block interface mapping is not axis-aligned");
    if (route.left_side == route.right_side)
      throw std::invalid_argument("multi-block interface faces do not have opposite orientation");
    component::validate_execution_context(execution);
    if (execution.communicator_f_handle != 0 ||
        std::string(execution.communicator_identity) != "serial")
      throw std::invalid_argument(
          "multi-block interface scheduler requires a serial execution capability; "
          "distributed trace exchange is not installed");
    if (left_state.box_array().size() < 1 || right_state.box_array().size() < 1 ||
        left_state.local_size() != left_state.box_array().size() ||
        right_state.local_size() != right_state.box_array().size())
      throw std::invalid_argument(
          "serial multi-block interface requires every prepared box to be locally owned");
    const int component_count = left_state.ncomp();
    if (component_count < 1 || right_state.ncomp() != component_count ||
        route.right_component_for_left.size() != static_cast<std::size_t>(component_count))
      throw std::invalid_argument("multi-block interface component spaces are not equal");
    std::vector<char> seen(static_cast<std::size_t>(component_count), 0);
    for (const int right_component : route.right_component_for_left) {
      if (right_component < 0 || right_component >= component_count ||
          seen[static_cast<std::size_t>(right_component)] != 0)
        throw std::invalid_argument("multi-block interface component permutation is not bijective");
      seen[static_cast<std::size_t>(right_component)] = 1;
    }
    const auto claims_endpoint = [](const AxisAlignedInterface& candidate,
                                    std::size_t block, InterfaceAxis axis,
                                    InterfaceSide side) {
      return (candidate.left_block == block && candidate.left_axis == axis &&
              candidate.left_side == side) ||
             (candidate.right_block == block && candidate.right_axis == axis &&
              candidate.right_side == side);
    };
    for (const PreparedInterface& installed : interfaces_) {
      if (installed.route.identity == route.identity && installed.route.level == route.level)
        throw std::invalid_argument("multi-block interface identity is already installed on level");
      if (installed.route.level == route.level &&
          (claims_endpoint(installed.route, route.left_block, route.left_axis, route.left_side) ||
           claims_endpoint(installed.route, route.right_block, route.right_axis,
                           route.right_side)))
        throw std::invalid_argument(
            "multi-block interface endpoint face is already owned on level");
    }

    const Box2D left_box = left_state.box_array().bounding_box();
    const Box2D right_box = right_state.box_array().bounding_box();
    const int left_faces = tangential_count_(left_box, route.left_axis);
    const int right_faces = tangential_count_(right_box, route.right_axis);
    const Real left_normal = normal_spacing_(left_geometry, route.left_axis);
    const Real right_normal = normal_spacing_(right_geometry, route.right_axis);
    const Real left_tangential = tangential_spacing_(left_geometry, route.left_axis);
    const Real right_tangential = tangential_spacing_(right_geometry, route.right_axis);
    if (left_faces != right_faces || left_faces < 1 || !(left_normal > Real(0)) ||
        !(right_normal > Real(0)) || left_normal != right_normal ||
        left_tangential != right_tangential)
      throw std::invalid_argument(
          "multi-block interface discretisations are not exactly equal");
    const bool has_affine_map = !route.affine_mapping_identity.empty();
    if (!std::isfinite(static_cast<double>(route.right_normal_translation)) ||
        !std::isfinite(static_cast<double>(route.right_tangential_scale)) ||
        !std::isfinite(static_cast<double>(route.right_tangential_offset)) ||
        (!has_affine_map &&
         (route.right_normal_translation != Real(0) ||
          route.right_tangential_scale != Real(1) ||
          route.right_tangential_offset != Real(0))) ||
        route.right_tangential_scale !=
            (route.tangential_orientation == TangentialOrientation::Aligned ? Real(1)
                                                                            : Real(-1)))
      throw std::invalid_argument("multi-block interface affine mapping is not authenticated");
    const Real left_normal_coordinate =
        normal_coordinate_(left_geometry, route.left_axis, route.left_side);
    const Real mapped_right_normal =
        normal_coordinate_(right_geometry, route.right_axis, route.right_side) +
        route.right_normal_translation;
    const Real left_tangent_low = tangential_low_(left_geometry, route.left_axis);
    const Real left_tangent_high = tangential_high_(left_geometry, route.left_axis);
    const Real right_tangent_low = tangential_low_(right_geometry, route.right_axis);
    const Real right_tangent_high = tangential_high_(right_geometry, route.right_axis);
    const Real mapped_right_low =
        route.right_tangential_scale *
            (route.tangential_orientation == TangentialOrientation::Aligned
                 ? right_tangent_low
                 : right_tangent_high) +
        route.right_tangential_offset;
    const Real mapped_right_high =
        route.right_tangential_scale *
            (route.tangential_orientation == TangentialOrientation::Aligned
                 ? right_tangent_high
                 : right_tangent_low) +
        route.right_tangential_offset;
    if (left_normal_coordinate != mapped_right_normal || left_tangent_low != mapped_right_low ||
        left_tangent_high != mapped_right_high)
      throw std::invalid_argument(
          "multi-block interface faces do not coincide in physical space");

    std::vector<BoundaryCell> left_cells = boundary_cells_(
        left_state, route.left_axis, route.left_side, left_faces);
    std::vector<BoundaryCell> right_cells = boundary_cells_(
        right_state, route.right_axis, route.right_side, right_faces);
    // Component prepare may allocate resources or have observable external effects.  Invoke it only
    // after every route/layout/geometry capability has been proved, but before mutating the scheduler
    // registry.  A rejected route therefore never prepares/caches a component, and a failed prepare
    // never leaves a half-installed interface.
    InterfaceFluxEvaluator evaluator = evaluator_factory();
    if (!evaluator)
      throw std::invalid_argument(
          "multi-block interface evaluator factory returned an empty evaluator");
    interfaces_.push_back(PreparedInterface{
        std::move(route), left_state.box_array().boxes(), left_state.dmap().ranks(),
        right_state.box_array().boxes(), right_state.dmap().ranks(), std::move(left_cells),
        std::move(right_cells), left_normal, right_normal, left_faces, component_count,
        std::move(evaluator), 0});
  }

  /// Convenience for already prepared in-process evaluators (principally native unit tests).
  void install(AxisAlignedInterface route, MultiFab& left_state, const Geometry& left_geometry,
               MultiFab& right_state, const Geometry& right_geometry,
               const PopsExecutionContextV1& execution,
               InterfaceFluxEvaluator evaluator) {
    install(
        std::move(route), left_state, left_geometry, right_state, right_geometry, execution,
        InterfaceFluxEvaluatorFactory(
            [evaluator = std::move(evaluator)]() mutable { return std::move(evaluator); }));
  }

  /// Apply every interface installed for point.level.  states/rhs are the complete block vectors of
  /// the owning runtime executor.  Each route calls its evaluator exactly once and scatters one shared
  /// flux with -/+ signs into left/right RHS at the same BoundaryEvaluationPoint.
  void apply(const BoundaryEvaluationPoint& point, const std::vector<MultiFab*>& states,
             const std::vector<MultiFab*>& rhs) {
    validate_point_(point);
    for (PreparedInterface& prepared : interfaces_) {
      if (prepared.route.level != point.level)
        continue;
      if (prepared.route.left_block >= states.size() ||
          prepared.route.right_block >= states.size() ||
          prepared.route.left_block >= rhs.size() || prepared.route.right_block >= rhs.size())
        throw std::runtime_error("multi-block interface runtime block vector is incomplete");
      const bool left_active = states[prepared.route.left_block] != nullptr ||
                               rhs[prepared.route.left_block] != nullptr;
      const bool right_active = states[prepared.route.right_block] != nullptr ||
                                rhs[prepared.route.right_block] != nullptr;
      if (!left_active && !right_active)
        continue;  // sparse RHS group unrelated to this installed interface
      if (!left_active || !right_active || states[prepared.route.left_block] == nullptr ||
          states[prepared.route.right_block] == nullptr ||
          rhs[prepared.route.left_block] == nullptr || rhs[prepared.route.right_block] == nullptr)
        throw std::runtime_error(
            "multi-block interface rate group must contain both sides at one StagePoint");
      apply_one_(prepared, point, *states[prepared.route.left_block],
                 *states[prepared.route.right_block], *rhs[prepared.route.left_block],
                 *rhs[prepared.route.right_block]);
    }
  }

  std::size_t size() const { return interfaces_.size(); }

  /// Roll back a failed pre-bind installation transaction.  Prepared evaluator ownership is
  /// released together with every route; no partially installed interface remains executable.
  void clear() { interfaces_.clear(); }

  bool has_interfaces(int level) const {
    for (const PreparedInterface& prepared : interfaces_)
      if (prepared.route.level == level)
        return true;
    return false;
  }

  bool participates(std::size_t block, int level) const {
    for (const PreparedInterface& prepared : interfaces_)
      if (prepared.route.level == level &&
          (prepared.route.left_block == block || prepared.route.right_block == block))
        return true;
    return false;
  }

  std::size_t evaluation_count(const std::string& identity, int level) const {
    for (const PreparedInterface& prepared : interfaces_)
      if (prepared.route.identity == identity && prepared.route.level == level)
        return prepared.evaluation_count;
    throw std::out_of_range("multi-block interface identity is not installed on level");
  }

 private:
  struct BoundaryCell {
    int local_box = -1;
    int i = 0;
    int j = 0;
  };

  struct PreparedInterface {
    AxisAlignedInterface route;
    std::vector<Box2D> left_boxes;
    std::vector<int> left_ranks;
    std::vector<Box2D> right_boxes;
    std::vector<int> right_ranks;
    std::vector<BoundaryCell> left_cells;
    std::vector<BoundaryCell> right_cells;
    Real left_normal_spacing = 0;
    Real right_normal_spacing = 0;
    int face_count = 0;
    int component_count = 0;
    InterfaceFluxEvaluator evaluator;
    std::size_t evaluation_count = 0;
  };

  static int tangential_count_(const Box2D& box, InterfaceAxis axis) {
    return axis == InterfaceAxis::X ? box.ny() : box.nx();
  }
  static Real normal_spacing_(const Geometry& geometry, InterfaceAxis axis) {
    return axis == InterfaceAxis::X ? geometry.dx() : geometry.dy();
  }
  static Real tangential_spacing_(const Geometry& geometry, InterfaceAxis axis) {
    return axis == InterfaceAxis::X ? geometry.dy() : geometry.dx();
  }
  static Real normal_coordinate_(const Geometry& geometry, InterfaceAxis axis,
                                 InterfaceSide side) {
    if (axis == InterfaceAxis::X)
      return side == InterfaceSide::Low ? geometry.xlo : geometry.xhi;
    return side == InterfaceSide::Low ? geometry.ylo : geometry.yhi;
  }
  static Real tangential_low_(const Geometry& geometry, InterfaceAxis axis) {
    return axis == InterfaceAxis::X ? geometry.ylo : geometry.xlo;
  }
  static Real tangential_high_(const Geometry& geometry, InterfaceAxis axis) {
    return axis == InterfaceAxis::X ? geometry.yhi : geometry.xhi;
  }

  static std::vector<BoundaryCell> boundary_cells_(const MultiFab& field, InterfaceAxis axis,
                                                    InterfaceSide side, int face_count) {
    const Box2D domain = field.box_array().bounding_box();
    const int normal_axis = axis == InterfaceAxis::X ? 0 : 1;
    const int tangent_axis = 1 - normal_axis;
    const int normal = side == InterfaceSide::Low ? domain.lo[normal_axis]
                                                   : domain.hi[normal_axis];
    std::vector<BoundaryCell> cells;
    cells.reserve(static_cast<std::size_t>(face_count));
    for (int face = 0; face < face_count; ++face) {
      const int tangent = domain.lo[tangent_axis] + face;
      const int i = normal_axis == 0 ? normal : tangent;
      const int j = normal_axis == 0 ? tangent : normal;
      int owner = -1;
      for (int global = 0; global < field.box_array().size(); ++global) {
        if (!field.box_array()[global].contains(i, j))
          continue;
        if (owner != -1)
          throw std::invalid_argument(
              "multi-block interface boundary decomposition overlaps at one face cell");
        owner = field.local_index_of(global);
      }
      if (owner < 0)
        throw std::invalid_argument(
            "multi-block interface boundary decomposition has a gap or a remote face cell");
      cells.push_back(BoundaryCell{owner, i, j});
    }
    return cells;
  }

  static void validate_point_(const BoundaryEvaluationPoint& point) {
    if (point.clock.empty() || point.tick < 0 || point.level < 0 || point.substep < 0 ||
        point.stage < 0 || !(point.dt > 0.0) || !std::isfinite(point.dt) ||
        !std::isfinite(point.physical_time) ||
        point.stage_fraction < ::pops::amr::Rational(0, 1) ||
        ::pops::amr::Rational(1, 1) < point.stage_fraction)
      throw std::invalid_argument("multi-block interface evaluation point is not fully qualified");
  }

  static void validate_runtime_field_(const MultiFab& field,
                                      const std::vector<Box2D>& expected_boxes,
                                      const std::vector<int>& expected_ranks,
                                      int component_count, const char* role) {
    if (field.box_array().boxes() != expected_boxes || field.dmap().ranks() != expected_ranks ||
        field.local_size() != static_cast<int>(expected_boxes.size()) ||
        field.ncomp() != component_count)
      throw std::runtime_error(std::string("multi-block interface ") + role +
                               " differs from its prepared layout");
  }

  static void apply_one_(PreparedInterface& prepared, const BoundaryEvaluationPoint& point,
                         MultiFab& left_state, MultiFab& right_state, MultiFab& left_rhs,
                         MultiFab& right_rhs) {
    validate_runtime_field_(left_state, prepared.left_boxes, prepared.left_ranks,
                            prepared.component_count,
                            "left state");
    validate_runtime_field_(right_state, prepared.right_boxes, prepared.right_ranks,
                            prepared.component_count,
                            "right state");
    validate_runtime_field_(left_rhs, prepared.left_boxes, prepared.left_ranks,
                            prepared.component_count, "left RHS");
    validate_runtime_field_(right_rhs, prepared.right_boxes, prepared.right_ranks,
                            prepared.component_count, "right RHS");
    left_state.sync_host();
    right_state.sync_host();
    left_rhs.sync_host();
    right_rhs.sync_host();

    const std::size_t packed_size = static_cast<std::size_t>(prepared.face_count) *
                                    static_cast<std::size_t>(prepared.component_count);
    std::vector<Real> left(packed_size);
    std::vector<Real> right(packed_size);
    std::vector<Real> flux(packed_size, std::numeric_limits<Real>::quiet_NaN());
    for (int face = 0; face < prepared.face_count; ++face) {
      const int mapped_face = prepared.route.tangential_orientation ==
                                      TangentialOrientation::Aligned
                                  ? face
                                  : prepared.face_count - 1 - face;
      const BoundaryCell& left_cell = prepared.left_cells[static_cast<std::size_t>(face)];
      const BoundaryCell& right_cell =
          prepared.right_cells[static_cast<std::size_t>(mapped_face)];
      const ConstArray4 left_values = left_state.fab(left_cell.local_box).const_array();
      const ConstArray4 right_values = right_state.fab(right_cell.local_box).const_array();
      for (int component = 0; component < prepared.component_count; ++component) {
        const std::size_t offset = static_cast<std::size_t>(face) *
                                       static_cast<std::size_t>(prepared.component_count) +
                                   static_cast<std::size_t>(component);
        left[offset] = left_values(left_cell.i, left_cell.j, component);
        right[offset] = right_values(
            right_cell.i, right_cell.j,
            prepared.route.right_component_for_left[static_cast<std::size_t>(component)]);
      }
    }

    const InterfaceFluxBatch batch{left.data(), right.data(), flux.data(), prepared.face_count,
                                   prepared.component_count};
    prepared.evaluator(point, batch);  // exactly one invocation for the complete prepared pair
    for (const Real value : flux)
      if (!std::isfinite(static_cast<double>(value)))
        throw std::runtime_error("multi-block interface evaluator returned a non-finite flux");
    ++prepared.evaluation_count;

    for (int face = 0; face < prepared.face_count; ++face) {
      const int mapped_face = prepared.route.tangential_orientation ==
                                      TangentialOrientation::Aligned
                                  ? face
                                  : prepared.face_count - 1 - face;
      const BoundaryCell& left_cell = prepared.left_cells[static_cast<std::size_t>(face)];
      const BoundaryCell& right_cell =
          prepared.right_cells[static_cast<std::size_t>(mapped_face)];
      Array4 left_out = left_rhs.fab(left_cell.local_box).array();
      Array4 right_out = right_rhs.fab(right_cell.local_box).array();
      for (int component = 0; component < prepared.component_count; ++component) {
        const std::size_t offset = static_cast<std::size_t>(face) *
                                       static_cast<std::size_t>(prepared.component_count) +
                                   static_cast<std::size_t>(component);
        const Real shared = flux[offset];
        left_out(left_cell.i, left_cell.j, component) -= shared / prepared.left_normal_spacing;
        right_out(right_cell.i, right_cell.j,
                  prepared.route.right_component_for_left[static_cast<std::size_t>(component)]) +=
            shared / prepared.right_normal_spacing;
      }
    }
    left_rhs.sync_device();
    right_rhs.sync_device();
  }

  std::vector<PreparedInterface> interfaces_;
};

}  // namespace pops::runtime::multiblock
