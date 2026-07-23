#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/config/generated_component_abi.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
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
    const bool collective_world = comm_active() && n_ranks() > 1;
    bool distributed = false;
    int communicator_size = 1;
    int component_count = 0;
    int left_faces = 0;
    Real left_normal = Real(0);
    Real right_normal = Real(0);
    std::vector<BoundaryCell> left_cells;
    std::vector<BoundaryCell> right_cells;
    std::exception_ptr structural_failure;
    try {
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
      const std::string communicator_identity(execution.communicator_identity);
      if (communicator_identity == "MPI_COMM_WORLD") {
#ifdef POPS_HAS_MPI
        if (!comm_active())
          throw std::invalid_argument(
              "multi-block interface MPI_COMM_WORLD capability is not active");
        int communicator_relation = MPI_UNEQUAL;
        ::pops::detail::require_mpi_success(
            MPI_Comm_compare(MPI_Comm_f2c(static_cast<MPI_Fint>(execution.communicator_f_handle)),
                             MPI_COMM_WORLD, &communicator_relation),
            "MPI_Comm_compare(interface execution context)");
        if (communicator_relation != MPI_IDENT ||
            MPI_Type_f2c(static_cast<MPI_Fint>(execution.communicator_datatype_f_handle)) !=
                MPI_DOUBLE)
          throw std::invalid_argument(
              "multi-block interface execution handles do not identify exact "
              "MPI_COMM_WORLD/MPI_DOUBLE");
        communicator_size = n_ranks();
        distributed = communicator_size > 1;
#else
        throw std::invalid_argument(
            "multi-block interface scheduler received MPI_COMM_WORLD from a serial build");
#endif
#ifdef POPS_HAS_MPI
      } else if (comm_active() && n_ranks() > 1) {
        throw std::invalid_argument(
            "multi-block interface cannot use a serial execution identity in an active "
            "multi-rank MPI world");
#endif
      }
      if (left_state.box_array().size() < 1 || right_state.box_array().size() < 1)
        throw std::invalid_argument("multi-block interface layouts cannot be empty");
      if (!distributed && (left_state.local_size() != left_state.box_array().size() ||
                           right_state.local_size() != right_state.box_array().size()))
        throw std::invalid_argument(
            "local multi-block interface execution requires every prepared box to be locally "
            "owned");
      component_count = left_state.ncomp();
      if (component_count < 1 || right_state.ncomp() != component_count ||
          route.right_component_for_left.size() != static_cast<std::size_t>(component_count))
        throw std::invalid_argument("multi-block interface component spaces are not equal");
      std::vector<char> seen(static_cast<std::size_t>(component_count), 0);
      for (const int right_component : route.right_component_for_left) {
        if (right_component < 0 || right_component >= component_count ||
            seen[static_cast<std::size_t>(right_component)] != 0)
          throw std::invalid_argument(
              "multi-block interface component permutation is not bijective");
        seen[static_cast<std::size_t>(right_component)] = 1;
      }
      const auto claims_endpoint = [](const AxisAlignedInterface& candidate, std::size_t block,
                                      InterfaceAxis axis, InterfaceSide side) {
        return (candidate.left_block == block && candidate.left_axis == axis &&
                candidate.left_side == side) ||
               (candidate.right_block == block && candidate.right_axis == axis &&
                candidate.right_side == side);
      };
      for (const PreparedInterface& installed : interfaces_) {
        if (installed.route.identity == route.identity && installed.route.level == route.level)
          throw std::invalid_argument(
              "multi-block interface identity is already installed on level");
        if (installed.route.level == route.level &&
            (claims_endpoint(installed.route, route.left_block, route.left_axis, route.left_side) ||
             claims_endpoint(installed.route, route.right_block, route.right_axis,
                             route.right_side)))
          throw std::invalid_argument(
              "multi-block interface endpoint face is already owned on level");
      }

      const Box2D left_box = left_state.box_array().bounding_box();
      const Box2D right_box = right_state.box_array().bounding_box();
      left_faces = tangential_count_(left_box, route.left_axis);
      const int right_faces = tangential_count_(right_box, route.right_axis);
      left_normal = normal_spacing_(left_geometry, route.left_axis);
      right_normal = normal_spacing_(right_geometry, route.right_axis);
      const Real left_tangential = tangential_spacing_(left_geometry, route.left_axis);
      const Real right_tangential = tangential_spacing_(right_geometry, route.right_axis);
      if (left_faces != right_faces || left_faces < 1 || !(left_normal > Real(0)) ||
          !(right_normal > Real(0)) || left_normal != right_normal ||
          left_tangential != right_tangential)
        throw std::invalid_argument("multi-block interface discretisations are not exactly equal");
      const bool has_affine_map = !route.affine_mapping_identity.empty();
      if (!std::isfinite(static_cast<double>(route.right_normal_translation)) ||
          !std::isfinite(static_cast<double>(route.right_tangential_scale)) ||
          !std::isfinite(static_cast<double>(route.right_tangential_offset)) ||
          (!has_affine_map &&
           (route.right_normal_translation != Real(0) || route.right_tangential_scale != Real(1) ||
            route.right_tangential_offset != Real(0))) ||
          route.right_tangential_scale !=
              (route.tangential_orientation == TangentialOrientation::Aligned ? Real(1) : Real(-1)))
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
              (route.tangential_orientation == TangentialOrientation::Aligned ? right_tangent_high
                                                                              : right_tangent_low) +
          route.right_tangential_offset;
      if (left_normal_coordinate != mapped_right_normal || left_tangent_low != mapped_right_low ||
          left_tangent_high != mapped_right_high)
        throw std::invalid_argument(
            "multi-block interface faces do not coincide in physical space");

      left_cells = boundary_cells_(left_state, route.left_axis, route.left_side, left_faces);
      right_cells = boundary_cells_(right_state, route.right_axis, route.right_side, right_faces);
    } catch (...) {
      structural_failure = std::current_exception();
    }
    finish_collective_preflight_(collective_world, structural_failure,
                                 "route/layout/execution preflight");
    if (distributed && !registry_agrees_across_ranks_())
      throw std::runtime_error("multi-block interface prepared registry differs across MPI ranks");
    const std::string collective_identity = collective_plan_identity_(
        route, left_state, left_geometry, right_state, right_geometry, left_normal, right_normal,
        left_faces, component_count, communicator_size);
    if (distributed &&
        !all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view(route.identity), std::string_view(collective_identity)}}))
      throw std::runtime_error(
          "multi-block interface prepared route/layout differs across MPI ranks");
    PreparedInterface prepared;
    std::exception_ptr materialization_failure;
    try {
      interfaces_.reserve(interfaces_.size() + 1);
      prepared = PreparedInterface{std::move(route),
                                   left_state.box_array().boxes(),
                                   left_state.dmap().ranks(),
                                   right_state.box_array().boxes(),
                                   right_state.dmap().ranks(),
                                   std::move(left_cells),
                                   std::move(right_cells),
                                   left_normal,
                                   right_normal,
                                   left_faces,
                                   component_count,
                                   distributed,
                                   communicator_size,
                                   collective_identity,
                                   InterfaceFluxEvaluator{},
                                   0};
      const std::size_t packed_size = static_cast<std::size_t>(left_faces) *
                                      static_cast<std::size_t>(component_count);
      if (packed_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 2)
        throw std::overflow_error(
            "multi-block interface trace batch exceeds the native MPI count domain");
      prepared.traces.assign(2 * packed_size, Real(0));
      prepared.flux.assign(packed_size, std::numeric_limits<Real>::quiet_NaN());
      prepared.consensus.assign(packed_size, Real(0));
    } catch (...) {
      materialization_failure = std::current_exception();
    }
    finish_collective_preflight_(distributed, materialization_failure,
                                 "prepared-route materialization");
    // Component prepare may allocate resources or have observable external effects.  Invoke it only
    // after every route/layout/geometry capability has been proved, but before mutating the scheduler
    // registry.  A rejected route therefore never prepares/caches a component, and a failed prepare
    // never leaves a half-installed interface.
    InterfaceFluxEvaluator evaluator;
    std::exception_ptr evaluator_prepare_failure;
    try {
      evaluator = evaluator_factory();
      if (!evaluator)
        throw std::invalid_argument(
            "multi-block interface evaluator factory returned an empty evaluator");
    } catch (...) {
      evaluator_prepare_failure = std::current_exception();
    }
    finish_collective_preflight_(distributed, evaluator_prepare_failure, "evaluator preparation");
    prepared.evaluator = std::move(evaluator);
    interfaces_.push_back(std::move(prepared));
  }

  /// Convenience for already prepared in-process evaluators (principally native unit tests).
  void install(AxisAlignedInterface route, MultiFab& left_state, const Geometry& left_geometry,
               MultiFab& right_state, const Geometry& right_geometry,
               const PopsExecutionContextV1& execution, InterfaceFluxEvaluator evaluator) {
    install(std::move(route), left_state, left_geometry, right_state, right_geometry, execution,
            InterfaceFluxEvaluatorFactory(
                [evaluator = std::move(evaluator)]() mutable { return std::move(evaluator); }));
  }

  /// Apply every interface installed for point.level.  states/rhs are the complete block vectors of
  /// the owning runtime executor.  Each route calls its evaluator exactly once and scatters one shared
  /// flux with -/+ signs into left/right RHS at the same BoundaryEvaluationPoint.
  void apply(const BoundaryEvaluationPoint& point, const std::vector<MultiFab*>& states,
             const std::vector<MultiFab*>& rhs) {
    const bool collective_world = comm_active() && n_ranks() > 1;
    std::exception_ptr point_failure;
    try {
      validate_point_(point);
    } catch (...) {
      point_failure = std::current_exception();
    }
    finish_collective_preflight_(collective_world, point_failure, "evaluation-point preflight");
    if (collective_world && !registry_agrees_across_ranks_())
      throw std::runtime_error("multi-block interface prepared registry differs across MPI ranks");
    const std::string point_identity = collective_point_identity_(point);
    if (collective_world && !all_ranks_agree_exact_ordered_byte_pairs(
                                {{std::string_view("point"), std::string_view(point_identity)}}))
      throw std::runtime_error(
          "multi-block interface BoundaryEvaluationPoint differs across MPI ranks");

    for (PreparedInterface& prepared : interfaces_) {
      if (prepared.route.level != point.level)
        continue;
      MultiFab* left_state = nullptr;
      MultiFab* right_state = nullptr;
      MultiFab* left_rhs = nullptr;
      MultiFab* right_rhs = nullptr;
      bool active = false;
      std::exception_ptr active_mask_failure;
      try {
        if (prepared.route.left_block >= states.size() ||
            prepared.route.right_block >= states.size() ||
            prepared.route.left_block >= rhs.size() || prepared.route.right_block >= rhs.size())
          throw std::runtime_error("multi-block interface runtime block vector is incomplete");
        left_state = states[prepared.route.left_block];
        right_state = states[prepared.route.right_block];
        left_rhs = rhs[prepared.route.left_block];
        right_rhs = rhs[prepared.route.right_block];
        const bool left_active = left_state != nullptr || left_rhs != nullptr;
        const bool right_active = right_state != nullptr || right_rhs != nullptr;
        active = left_active || right_active;
        if (active && (!left_active || !right_active || left_state == nullptr ||
                       right_state == nullptr || left_rhs == nullptr || right_rhs == nullptr))
          throw std::runtime_error(
              "multi-block interface rate group must contain both sides at one StagePoint");
      } catch (...) {
        active_mask_failure = std::current_exception();
      }
      finish_collective_preflight_(prepared.distributed, active_mask_failure,
                                   "active-mask preflight");
      if (prepared.distributed) {
        const long minimum_active = all_reduce_min(active ? 1L : 0L);
        const long maximum_active = all_reduce_max(active ? 1L : 0L);
        if (minimum_active != maximum_active)
          throw std::runtime_error("multi-block interface active mask differs across MPI ranks");
      }
      if (!active)
        continue;  // sparse RHS group unrelated to this installed interface on every rank
      apply_one_(prepared, point, *left_state, *right_state, *left_rhs, *right_rhs);
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
    bool distributed = false;
    int communicator_size = 1;
    std::string collective_identity;
    InterfaceFluxEvaluator evaluator;
    std::size_t evaluation_count = 0;
    // Persistent host ABI scratch.  The current external NumericalFlux ABI is explicitly host
    // memory, but a fixed interface must not allocate and copy whole trace vectors on every stage.
    std::vector<Real> traces;
    std::vector<Real> flux;
    std::vector<Real> consensus;
  };
  static_assert(std::is_nothrow_move_constructible_v<PreparedInterface>);

  static void finish_collective_preflight_(bool collective, const std::exception_ptr& local_failure,
                                           const char* phase) {
    const long failure_count =
        collective ? all_reduce_sum(local_failure ? 1L : 0L) : (local_failure ? 1L : 0L);
    if (failure_count == 0)
      return;
    if (local_failure)
      std::rethrow_exception(local_failure);
    throw std::runtime_error(std::string("multi-block interface ") + phase +
                             " failed on another MPI rank");
  }

  template <class Value>
  static void append_identity_scalar_(std::string& bytes, const Value& value) {
    static_assert(std::is_trivially_copyable_v<Value>);
    bytes.append(reinterpret_cast<const char*>(&value), sizeof(Value));
  }

  static void append_identity_text_(std::string& bytes, std::string_view value) {
    append_identity_scalar_(bytes, static_cast<std::uint64_t>(value.size()));
    bytes.append(value.data(), value.size());
  }

  static void append_identity_box_(std::string& bytes, const Box2D& box) {
    append_identity_scalar_(bytes, box.lo[0]);
    append_identity_scalar_(bytes, box.lo[1]);
    append_identity_scalar_(bytes, box.hi[0]);
    append_identity_scalar_(bytes, box.hi[1]);
  }

  static void append_identity_layout_(std::string& bytes, const MultiFab& field) {
    const auto& boxes = field.box_array().boxes();
    const auto& ranks = field.dmap().ranks();
    append_identity_scalar_(bytes, static_cast<std::uint64_t>(boxes.size()));
    for (std::size_t index = 0; index < boxes.size(); ++index) {
      append_identity_box_(bytes, boxes[index]);
      append_identity_scalar_(bytes, ranks[index]);
    }
    append_identity_scalar_(bytes, field.ncomp());
    append_identity_scalar_(bytes, field.n_grow());
  }

  static void append_identity_geometry_(std::string& bytes, const Geometry& geometry) {
    append_identity_box_(bytes, geometry.domain);
    append_identity_scalar_(bytes, geometry.xlo);
    append_identity_scalar_(bytes, geometry.xhi);
    append_identity_scalar_(bytes, geometry.ylo);
    append_identity_scalar_(bytes, geometry.yhi);
  }

  static std::string collective_plan_identity_(
      const AxisAlignedInterface& route, const MultiFab& left_state, const Geometry& left_geometry,
      const MultiFab& right_state, const Geometry& right_geometry, Real left_normal,
      Real right_normal, int face_count, int component_count, int communicator_size) {
    std::string bytes;
    append_identity_text_(bytes, "pops.multiblock.interface-plan.v1");
    append_identity_text_(bytes, route.identity);
    append_identity_scalar_(bytes, static_cast<std::uint64_t>(route.left_block));
    append_identity_scalar_(bytes, static_cast<std::uint64_t>(route.right_block));
    append_identity_scalar_(bytes, route.level);
    append_identity_scalar_(bytes, route.left_axis);
    append_identity_scalar_(bytes, route.right_axis);
    append_identity_scalar_(bytes, route.left_side);
    append_identity_scalar_(bytes, route.right_side);
    append_identity_scalar_(bytes, route.tangential_orientation);
    append_identity_scalar_(bytes,
                            static_cast<std::uint64_t>(route.right_component_for_left.size()));
    for (const int component : route.right_component_for_left)
      append_identity_scalar_(bytes, component);
    append_identity_text_(bytes, route.affine_mapping_identity);
    append_identity_scalar_(bytes, route.right_normal_translation);
    append_identity_scalar_(bytes, route.right_tangential_scale);
    append_identity_scalar_(bytes, route.right_tangential_offset);
    append_identity_layout_(bytes, left_state);
    append_identity_layout_(bytes, right_state);
    append_identity_geometry_(bytes, left_geometry);
    append_identity_geometry_(bytes, right_geometry);
    append_identity_scalar_(bytes, left_normal);
    append_identity_scalar_(bytes, right_normal);
    append_identity_scalar_(bytes, face_count);
    append_identity_scalar_(bytes, component_count);
    append_identity_scalar_(bytes, communicator_size);
    return bytes;
  }

  static std::string collective_point_identity_(const BoundaryEvaluationPoint& point) {
    std::string bytes;
    append_identity_text_(bytes, "pops.multiblock.evaluation-point.v1");
    append_identity_text_(bytes, point.clock);
    append_identity_scalar_(bytes, point.tick);
    append_identity_scalar_(bytes, point.level);
    append_identity_scalar_(bytes, point.substep);
    append_identity_scalar_(bytes, point.stage);
    append_identity_scalar_(bytes, point.stage_fraction.numerator);
    append_identity_scalar_(bytes, point.stage_fraction.denominator);
    append_identity_scalar_(bytes, point.dt);
    append_identity_scalar_(bytes, point.physical_time);
    return bytes;
  }

  bool registry_agrees_across_ranks_() const {
    std::vector<std::pair<std::string_view, std::string_view>> identities;
    identities.reserve(interfaces_.size());
    for (const PreparedInterface& prepared : interfaces_)
      identities.emplace_back(prepared.route.identity, prepared.collective_identity);
    return all_ranks_agree_exact_ordered_byte_pairs(identities);
  }

  static int tangential_count_(const Box2D& box, InterfaceAxis axis) {
    return axis == InterfaceAxis::X ? box.ny() : box.nx();
  }
  static Real normal_spacing_(const Geometry& geometry, InterfaceAxis axis) {
    return axis == InterfaceAxis::X ? geometry.dx() : geometry.dy();
  }
  static Real tangential_spacing_(const Geometry& geometry, InterfaceAxis axis) {
    return axis == InterfaceAxis::X ? geometry.dy() : geometry.dx();
  }
  static Real normal_coordinate_(const Geometry& geometry, InterfaceAxis axis, InterfaceSide side) {
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
    const int normal = side == InterfaceSide::Low ? domain.lo[normal_axis] : domain.hi[normal_axis];
    std::vector<BoundaryCell> cells;
    cells.reserve(static_cast<std::size_t>(face_count));
    for (int face = 0; face < face_count; ++face) {
      const int tangent = domain.lo[tangent_axis] + face;
      const int i = normal_axis == 0 ? normal : tangent;
      const int j = normal_axis == 0 ? tangent : normal;
      int global_owner = -1;
      for (int global = 0; global < field.box_array().size(); ++global) {
        if (!field.box_array()[global].contains(i, j))
          continue;
        if (global_owner != -1)
          throw std::invalid_argument(
              "multi-block interface boundary decomposition overlaps at one face cell");
        global_owner = global;
      }
      if (global_owner < 0)
        throw std::invalid_argument(
            "multi-block interface boundary decomposition has a gap at one face cell");
      const int local_owner = field.local_index_of(global_owner);
      if ((field.dmap()[global_owner] == my_rank()) != (local_owner >= 0))
        throw std::logic_error(
            "multi-block interface local ownership differs from its DistributionMapping");
      cells.push_back(BoundaryCell{local_owner, i, j});
    }
    return cells;
  }

  static void validate_point_(const BoundaryEvaluationPoint& point) {
    if (point.clock.empty() || point.tick < 0 || point.level < 0 || point.substep < 0 ||
        point.stage < 0 || !(point.dt > 0.0) || !std::isfinite(point.dt) ||
        !std::isfinite(point.physical_time) || point.stage_fraction < ::pops::amr::Rational(0, 1) ||
        ::pops::amr::Rational(1, 1) < point.stage_fraction)
      throw std::invalid_argument("multi-block interface evaluation point is not fully qualified");
  }

  static bool runtime_field_matches_(const MultiFab& field,
                                     const std::vector<Box2D>& expected_boxes,
                                     const std::vector<int>& expected_ranks, int component_count) {
    int expected_local_size = 0;
    for (const int owner : expected_ranks)
      if (owner == my_rank())
        ++expected_local_size;
    return field.box_array().boxes() == expected_boxes && field.dmap().ranks() == expected_ranks &&
           field.local_size() == expected_local_size && field.ncomp() == component_count;
  }

  static void require_distributed_flux_consensus_(std::vector<Real>& flux,
                                                  std::vector<Real>& reference) {
#ifdef POPS_HAS_MPI
    if (reference.size() != flux.size())
      throw std::logic_error("multi-block interface consensus scratch changed size");
    std::copy(flux.begin(), flux.end(), reference.begin());
    ::pops::detail::require_mpi_success(
        MPI_Bcast(reference.data(), static_cast<int>(reference.size()), MPI_DOUBLE, 0,
                  MPI_COMM_WORLD),
        "MPI_Bcast(multi-block shared flux)");
    const bool equal = std::memcmp(reference.data(), flux.data(), flux.size() * sizeof(Real)) == 0;
    if (all_reduce_sum(equal ? 0L : 1L) != 0)
      throw std::runtime_error(
          "multi-block interface evaluator returned rank-dependent shared flux");
    std::copy(reference.begin(), reference.end(), flux.begin());
#else
    (void)flux;
    throw std::logic_error(
        "distributed multi-block flux consensus is unavailable in a serial build");
#endif
  }

  static void apply_one_(PreparedInterface& prepared, const BoundaryEvaluationPoint& point,
                         MultiFab& left_state, MultiFab& right_state, MultiFab& left_rhs,
                         MultiFab& right_rhs) {
    if (prepared.distributed && (!comm_active() || n_ranks() != prepared.communicator_size))
      throw std::runtime_error("multi-block interface MPI world changed after route preparation");
    const bool layouts_match =
        runtime_field_matches_(left_state, prepared.left_boxes, prepared.left_ranks,
                               prepared.component_count) &&
        runtime_field_matches_(right_state, prepared.right_boxes, prepared.right_ranks,
                               prepared.component_count) &&
        runtime_field_matches_(left_rhs, prepared.left_boxes, prepared.left_ranks,
                               prepared.component_count) &&
        runtime_field_matches_(right_rhs, prepared.right_boxes, prepared.right_ranks,
                               prepared.component_count);
    if (prepared.distributed) {
      if (all_reduce_sum(layouts_match ? 0L : 1L) != 0)
        throw std::runtime_error(
            "multi-block interface runtime fields differ from their prepared layouts on one "
            "or more MPI ranks");
    } else if (!layouts_match) {
      throw std::runtime_error(
          "multi-block interface runtime fields differ from their prepared layouts");
    }
    left_state.sync_host();
    right_state.sync_host();
    left_rhs.sync_host();
    right_rhs.sync_host();

    const std::size_t packed_size = static_cast<std::size_t>(prepared.face_count) *
                                    static_cast<std::size_t>(prepared.component_count);
    if (packed_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 2)
      throw std::overflow_error(
          "multi-block interface trace batch exceeds the native MPI count domain");
    if (prepared.traces.size() != 2 * packed_size || prepared.flux.size() != packed_size ||
        prepared.consensus.size() != packed_size)
      throw std::logic_error("multi-block interface prepared scratch changed size");
    std::fill(prepared.traces.begin(), prepared.traces.end(), Real(0));
    std::fill(prepared.flux.begin(), prepared.flux.end(),
              std::numeric_limits<Real>::quiet_NaN());
    Real* const left = prepared.traces.data();
    Real* const right = prepared.traces.data() + packed_size;
    for (int face = 0; face < prepared.face_count; ++face) {
      const int mapped_face =
          prepared.route.tangential_orientation == TangentialOrientation::Aligned
              ? face
              : prepared.face_count - 1 - face;
      const BoundaryCell& left_cell = prepared.left_cells[static_cast<std::size_t>(face)];
      const BoundaryCell& right_cell = prepared.right_cells[static_cast<std::size_t>(mapped_face)];
      if (left_cell.local_box >= 0) {
        const ConstArray4 left_values = left_state.fab(left_cell.local_box).const_array();
        for (int component = 0; component < prepared.component_count; ++component) {
          const std::size_t offset =
              static_cast<std::size_t>(face) * static_cast<std::size_t>(prepared.component_count) +
              static_cast<std::size_t>(component);
          left[offset] = left_values(left_cell.i, left_cell.j, component);
        }
      }
      if (right_cell.local_box >= 0) {
        const ConstArray4 right_values = right_state.fab(right_cell.local_box).const_array();
        for (int component = 0; component < prepared.component_count; ++component) {
          const std::size_t offset =
              static_cast<std::size_t>(face) * static_cast<std::size_t>(prepared.component_count) +
              static_cast<std::size_t>(component);
          right[offset] = right_values(
              right_cell.i, right_cell.j,
              prepared.route.right_component_for_left[static_cast<std::size_t>(component)]);
        }
      }
    }
    if (prepared.distributed)
      all_reduce_sum_inplace(prepared.traces.data(), prepared.traces.size());

    const InterfaceFluxBatch batch{left, right, prepared.flux.data(), prepared.face_count,
                                   prepared.component_count};
    std::exception_ptr evaluator_failure;
    try {
      prepared.evaluator(point, batch);  // once per rank, always with the complete prepared pair
    } catch (...) {
      evaluator_failure = std::current_exception();
    }
    if (prepared.distributed) {
      if (all_reduce_sum(evaluator_failure ? 1L : 0L) != 0)
        throw std::runtime_error("multi-block interface evaluator failed on one or more MPI ranks");
    } else if (evaluator_failure) {
      std::rethrow_exception(evaluator_failure);
    }
    bool finite_flux = true;
    for (const Real value : prepared.flux)
      finite_flux = finite_flux && std::isfinite(static_cast<double>(value));
    if (prepared.distributed) {
      if (all_reduce_sum(finite_flux ? 0L : 1L) != 0)
        throw std::runtime_error(
            "multi-block interface evaluator returned a non-finite flux on one or more MPI ranks");
      require_distributed_flux_consensus_(prepared.flux, prepared.consensus);
    } else if (!finite_flux) {
      throw std::runtime_error("multi-block interface evaluator returned a non-finite flux");
    }
    ++prepared.evaluation_count;

    for (int face = 0; face < prepared.face_count; ++face) {
      const int mapped_face =
          prepared.route.tangential_orientation == TangentialOrientation::Aligned
              ? face
              : prepared.face_count - 1 - face;
      const BoundaryCell& left_cell = prepared.left_cells[static_cast<std::size_t>(face)];
      const BoundaryCell& right_cell = prepared.right_cells[static_cast<std::size_t>(mapped_face)];
      if (left_cell.local_box >= 0) {
        Array4 left_out = left_rhs.fab(left_cell.local_box).array();
        for (int component = 0; component < prepared.component_count; ++component) {
          const std::size_t offset =
              static_cast<std::size_t>(face) * static_cast<std::size_t>(prepared.component_count) +
              static_cast<std::size_t>(component);
          left_out(left_cell.i, left_cell.j, component) -=
              prepared.flux[offset] / prepared.left_normal_spacing;
        }
      }
      if (right_cell.local_box >= 0) {
        Array4 right_out = right_rhs.fab(right_cell.local_box).array();
        for (int component = 0; component < prepared.component_count; ++component) {
          const std::size_t offset =
              static_cast<std::size_t>(face) * static_cast<std::size_t>(prepared.component_count) +
              static_cast<std::size_t>(component);
          right_out(right_cell.i, right_cell.j,
                    prepared.route.right_component_for_left[static_cast<std::size_t>(component)]) +=
              prepared.flux[offset] / prepared.right_normal_spacing;
        }
      }
    }
    left_rhs.sync_device();
    right_rhs.sync_device();
  }

  std::vector<PreparedInterface> interfaces_;
};

}  // namespace pops::runtime::multiblock
