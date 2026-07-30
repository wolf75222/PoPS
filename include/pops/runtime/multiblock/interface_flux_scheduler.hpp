#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/numerics/time/amr/reflux/amr_interface_flux_ledger.hpp>
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

using InterfaceFluxFragmentPayload = std::vector<Real>;
using InterfaceFluxFragmentLedger =
    ::pops::amr::TransactionalInterfaceFluxLedger<InterfaceFluxFragmentPayload>;

/// Exact Program-owned transaction context for publishing one scheduler evaluation as the
/// level-qualified contribution of a fixed two-level interface. The scheduler owns the canonical
/// flux batch; the Program owns the temporal/topology identity and enclosing attempt transaction.
struct InterfaceFluxFragmentPublication {
  InterfaceFluxFragmentLedger* ledger = nullptr;
  std::uint64_t topology_epoch = 0;
  int coarse_level = 0;
  int fine_level = 1;
  ::pops::amr::ClockStamp clock;
  std::string stage_identity;
  ::pops::amr::ClockWindow interval;
  ::pops::amr::Rational stage_weight{1, 1};
  bool stage_weight_resolved = true;
};

class InterfaceFluxScheduler {
 public:
  /// Prepare and install one supported route.  Layout, component permutation, face orientation and
  /// equal discretisation are all proved here, before any residual evaluation can begin.
  void install(AxisAlignedInterface route, MultiFab& left_state, const Geometry& left_geometry,
               MultiFab& right_state, const Geometry& right_geometry,
               const PopsExecutionContextV1& execution,
               InterfaceFluxEvaluatorFactory evaluator_factory) {
    // MultiFab/DistributionMapping still stores owners in the process-world rank space.  Retain
    // that storage authority under an explicit name for admission only: every numerical
    // collective below runs on the communicator carried by ExecutionContext.  Once field storage
    // owns a communicator-relative rank space, this single compatibility seam can disappear.
    const CommunicatorView field_rank_space =
        comm_active() ? world_communicator_view() : CommunicatorView{};
    const bool collective_world = field_rank_space.active() && field_rank_space.size() > 1;
    const CommunicatorView admission_communicator =
        collective_world ? field_rank_space : CommunicatorView{};
    bool distributed = false;
    CommunicatorView execution_communicator;
    int communicator_rank = 0;
    int communicator_size = 1;
    std::string communicator_identity = "serial";
    int component_count = 0;
    int left_faces = 0;
    Real left_normal = Real(0);
    Real right_normal = Real(0);
    Real left_tangential = Real(0);
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
      communicator_identity.assign(execution.communicator_identity);
      if (communicator_identity != "serial" &&
          communicator_identity != POPS_EXECUTION_NONCOLLECTIVE_IDENTITY_V1) {
#ifdef POPS_HAS_MPI
        if (!comm_active())
          throw std::invalid_argument(
              "multi-block interface communicator capability is not active");
        const MPI_Comm communicator =
            MPI_Comm_f2c(static_cast<MPI_Fint>(execution.communicator_f_handle));
        if (communicator == MPI_COMM_NULL ||
            MPI_Type_f2c(static_cast<MPI_Fint>(execution.communicator_datatype_f_handle)) !=
                MPI_DOUBLE)
          throw std::invalid_argument(
              "multi-block interface execution handles do not identify a live "
              "communicator/MPI_DOUBLE authority");
        int communicator_relation = MPI_UNEQUAL;
        ::pops::detail::require_mpi_success(
            MPI_Comm_compare(communicator, field_rank_space.native_handle(),
                             &communicator_relation),
            "MPI_Comm_compare(interface field rank space)");
        if (communicator_relation != MPI_IDENT && communicator_relation != MPI_CONGRUENT)
          throw std::invalid_argument(
              "multi-block interface communicator must preserve the field rank space");
        execution_communicator = CommunicatorView{communicator};
        communicator_rank = execution_communicator.rank();
        communicator_size = execution_communicator.size();
        distributed = communicator_size > 1;
#else
        throw std::invalid_argument(
            "multi-block interface scheduler received a distributed context from a serial build");
#endif
      } else if (communicator_identity == POPS_EXECUTION_NONCOLLECTIVE_IDENTITY_V1) {
        throw std::invalid_argument(
            "multi-block interface scheduler requires collective execution authority");
#ifdef POPS_HAS_MPI
      } else if (comm_active() && n_ranks() > 1) {
        throw std::invalid_argument(
            "multi-block interface cannot use a serial execution identity in an active "
            "multi-rank MPI world");
#endif
      }
      if (!interfaces_.empty()) {
        const PreparedInterface& existing = interfaces_.front();
        if (existing.communicator_identity != communicator_identity ||
            existing.communicator_size != communicator_size)
          throw std::invalid_argument(
              "multi-block interface routes require one exact execution communicator");
#ifdef POPS_HAS_MPI
        if (distributed) {
          int relation = MPI_UNEQUAL;
          ::pops::detail::require_mpi_success(
              MPI_Comm_compare(existing.communicator.native_handle(),
                               execution_communicator.native_handle(), &relation),
              "MPI_Comm_compare(installed interface communicators)");
          if (relation != MPI_IDENT)
            throw std::invalid_argument(
                "multi-block interface routes require the same communicator context");
        }
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
      if (!tiles_declared_physical_face_(left_box, left_geometry, route.left_axis,
                                         route.left_side) ||
          !tiles_declared_physical_face_(right_box, right_geometry, route.right_axis,
                                         route.right_side))
        throw std::invalid_argument(
            "multi-block interface level layout does not tile its declared physical face");
      left_faces = tangential_count_(left_box, route.left_axis);
      const int right_faces = tangential_count_(right_box, route.right_axis);
      left_normal = normal_spacing_(left_geometry, route.left_axis);
      right_normal = normal_spacing_(right_geometry, route.right_axis);
      left_tangential = tangential_spacing_(left_geometry, route.left_axis);
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

      left_cells = boundary_cells_(left_state, route.left_axis, route.left_side, left_faces,
                                   communicator_rank);
      right_cells = boundary_cells_(right_state, route.right_axis, route.right_side, right_faces,
                                    communicator_rank);
    } catch (...) {
      structural_failure = std::current_exception();
    }
    finish_collective_preflight_(admission_communicator, structural_failure,
                                 "route/layout/execution preflight");
    if (distributed && !registry_agrees_across_ranks_(execution_communicator))
      throw std::runtime_error("multi-block interface prepared registry differs across MPI ranks");
    const std::string collective_identity = collective_plan_identity_(
        route, left_state, left_geometry, right_state, right_geometry, left_normal, right_normal,
        left_faces, component_count, communicator_identity, communicator_size);
    if (distributed &&
        !all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view(route.identity), std::string_view(collective_identity)}},
            execution_communicator))
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
                                   left_tangential,
                                   left_faces,
                                   component_count,
                                   distributed,
                                   execution_communicator,
                                   communicator_rank,
                                   communicator_size,
                                   communicator_identity,
                                   collective_identity,
                                   InterfaceFluxEvaluator{},
                                   0};
      const std::size_t packed_size =
          static_cast<std::size_t>(left_faces) * static_cast<std::size_t>(component_count);
      if (packed_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 2)
        throw std::overflow_error(
            "multi-block interface trace batch exceeds the native MPI count domain");
      prepared.traces.assign(2 * packed_size, Real(0));
      prepared.flux.assign(packed_size, std::numeric_limits<Real>::quiet_NaN());
      prepared.consensus.assign(packed_size, Real(0));
    } catch (...) {
      materialization_failure = std::current_exception();
    }
    finish_collective_preflight_(execution_communicator, materialization_failure,
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
    finish_collective_preflight_(execution_communicator, evaluator_prepare_failure,
                                 "evaluator preparation");
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
             const std::vector<MultiFab*>& rhs,
             InterfaceFluxFragmentPublication* publication = nullptr) {
    if (interfaces_.empty()) {
      validate_point_(point);
      if (publication != nullptr)
        validate_fragment_publication_(point, *publication);
      return;
    }
    const CommunicatorView execution_communicator = interfaces_.front().communicator;
    const bool collective = execution_communicator.active() && execution_communicator.size() > 1;
    std::exception_ptr point_failure;
    try {
      validate_point_(point);
    } catch (...) {
      point_failure = std::current_exception();
    }
    finish_collective_preflight_(execution_communicator, point_failure,
                                 "evaluation-point preflight");
    std::exception_ptr publication_failure;
    try {
      if (publication != nullptr) {
        if (collective)
          throw std::runtime_error(
              "AMR interface-flux fragment publication does not yet support distributed MPI");
        validate_fragment_publication_(point, *publication);
      }
    } catch (...) {
      publication_failure = std::current_exception();
    }
    finish_collective_preflight_(execution_communicator, publication_failure,
                                 "interface-fragment publication preflight");
    if (collective && !registry_agrees_across_ranks_(execution_communicator))
      throw std::runtime_error("multi-block interface prepared registry differs across MPI ranks");
    const std::string point_identity = collective_point_identity_(point);
    if (collective && !all_ranks_agree_exact_ordered_byte_pairs(
                          {{std::string_view("point"), std::string_view(point_identity)}},
                          execution_communicator))
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
      finish_collective_preflight_(prepared.communicator, active_mask_failure,
                                   "active-mask preflight");
      if (prepared.distributed) {
        const long minimum_active = all_reduce_min(active ? 1L : 0L, prepared.communicator);
        const long maximum_active = all_reduce_max(active ? 1L : 0L, prepared.communicator);
        if (minimum_active != maximum_active)
          throw std::runtime_error("multi-block interface active mask differs across MPI ranks");
      }
      if (!active)
        continue;  // sparse RHS group unrelated to this installed interface on every rank
      if (publication != nullptr && prepared.distributed)
        throw std::runtime_error(
            "AMR interface-flux fragment publication does not yet support distributed MPI");
      apply_one_(prepared, point, *left_state, *right_state, *left_rhs, *right_rhs, publication);
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

  /// Boundary plans are shared across levels. A fixed two-level Program must therefore schedule the
  /// same interface on both levels instead of omitting a touching face on one level with no canonical
  /// flux to put back.
  void require_complete_fixed_two_level_registry() const {
    for (const PreparedInterface& prepared : interfaces_) {
      if (prepared.route.level != 0 && prepared.route.level != 1)
        throw std::runtime_error(
            "fixed two-level interface registry contains a route outside levels 0/1");
      const int peer_level = 1 - prepared.route.level;
      const PreparedInterface* peer = nullptr;
      for (const PreparedInterface& candidate : interfaces_)
        if (candidate.route.identity == prepared.route.identity &&
            candidate.route.level == peer_level) {
          peer = &candidate;
          break;
        }
      if (peer == nullptr || !same_route_across_levels_(prepared.route, peer->route) ||
          prepared.component_count != peer->component_count)
        throw std::runtime_error(
            "fixed two-level interface registry is missing an exact peer-level route");
    }
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
    Real face_measure = 0;
    int face_count = 0;
    int component_count = 0;
    bool distributed = false;
    CommunicatorView communicator;
    int communicator_rank = 0;
    int communicator_size = 1;
    std::string communicator_identity;
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

  static void finish_collective_preflight_(const CommunicatorView& communicator,
                                           const std::exception_ptr& local_failure,
                                           const char* phase) {
    const bool collective = communicator.active() && communicator.size() > 1;
    const long failure_count = collective ? all_reduce_sum(local_failure ? 1L : 0L, communicator)
                                          : (local_failure ? 1L : 0L);
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
      Real right_normal, int face_count, int component_count,
      std::string_view communicator_identity, int communicator_size) {
    std::string bytes;
    append_identity_text_(bytes, "pops.multiblock.interface-plan.v2");
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
    append_identity_text_(bytes, communicator_identity);
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

  bool registry_agrees_across_ranks_(const CommunicatorView& communicator) const {
    std::vector<std::pair<std::string_view, std::string_view>> identities;
    identities.reserve(interfaces_.size());
    for (const PreparedInterface& prepared : interfaces_)
      identities.emplace_back(prepared.route.identity, prepared.collective_identity);
    return all_ranks_agree_exact_ordered_byte_pairs(identities, communicator);
  }

  static int tangential_count_(const Box2D& box, InterfaceAxis axis) {
    return axis == InterfaceAxis::X ? box.ny() : box.nx();
  }
  static bool tiles_declared_physical_face_(const Box2D& box, const Geometry& geometry,
                                            InterfaceAxis axis, InterfaceSide side) {
    const int normal_axis = axis == InterfaceAxis::X ? 0 : 1;
    const int tangent_axis = 1 - normal_axis;
    const int normal = side == InterfaceSide::Low ? box.lo[normal_axis] : box.hi[normal_axis];
    const int expected_normal = side == InterfaceSide::Low ? geometry.domain.lo[normal_axis]
                                                           : geometry.domain.hi[normal_axis];
    return normal == expected_normal && box.lo[tangent_axis] == geometry.domain.lo[tangent_axis] &&
           box.hi[tangent_axis] == geometry.domain.hi[tangent_axis];
  }
  static bool same_route_across_levels_(const AxisAlignedInterface& lhs,
                                        const AxisAlignedInterface& rhs) {
    return lhs.identity == rhs.identity && lhs.left_block == rhs.left_block &&
           lhs.right_block == rhs.right_block && lhs.left_axis == rhs.left_axis &&
           lhs.right_axis == rhs.right_axis && lhs.left_side == rhs.left_side &&
           lhs.right_side == rhs.right_side &&
           lhs.tangential_orientation == rhs.tangential_orientation &&
           lhs.right_component_for_left == rhs.right_component_for_left &&
           lhs.affine_mapping_identity == rhs.affine_mapping_identity &&
           lhs.right_normal_translation == rhs.right_normal_translation &&
           lhs.right_tangential_scale == rhs.right_tangential_scale &&
           lhs.right_tangential_offset == rhs.right_tangential_offset;
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
                                                   InterfaceSide side, int face_count,
                                                   int communicator_rank) {
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
      if ((field.dmap()[global_owner] == communicator_rank) != (local_owner >= 0))
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

  static void validate_fragment_publication_(const BoundaryEvaluationPoint& point,
                                             const InterfaceFluxFragmentPublication& publication) {
    if (publication.ledger == nullptr || !publication.ledger->in_transaction())
      throw std::invalid_argument(
          "AMR interface-flux fragment publication requires an active ledger transaction");
    const ::pops::amr::Rational interval_span =
        publication.interval.end.phase - publication.interval.begin.phase;
    const ::pops::amr::Rational expected_phase =
        publication.interval.begin.phase + point.stage_fraction * interval_span;
    const double expected_physical_time =
        publication.interval.begin.physical_time +
        point.stage_fraction.value() *
            (publication.interval.end.physical_time - publication.interval.begin.physical_time);
    if (publication.ledger->topology_epoch() != publication.topology_epoch ||
        publication.coarse_level != 0 || publication.fine_level != 1 ||
        publication.clock.level != point.level ||
        (point.level != publication.coarse_level && point.level != publication.fine_level) ||
        publication.interval.begin.level != point.level ||
        publication.interval.end.level != point.level ||
        publication.interval.begin.macro_step != point.tick ||
        publication.interval.end.macro_step != point.tick ||
        publication.clock.macro_step != point.tick || publication.clock.phase != expected_phase ||
        publication.clock.physical_time != point.physical_time ||
        publication.clock.physical_time != expected_physical_time ||
        publication.stage_identity.empty())
      throw std::invalid_argument(
          "AMR interface-flux fragment publication identity differs from its scheduler point");
  }

  static void publish_fragment_(const PreparedInterface& prepared,
                                const BoundaryEvaluationPoint& point,
                                const InterfaceFluxFragmentPublication& publication) {
    const auto orientation = publication.clock.level == publication.coarse_level
                                 ? ::pops::amr::InterfaceFluxOrientation::CoarseOutward
                                 : ::pops::amr::InterfaceFluxOrientation::FineOutward;
    ::pops::amr::InterfaceFluxFragmentKey key{
        prepared.route.identity,   publication.topology_epoch,
        publication.coarse_level,  publication.fine_level,
        publication.clock,         publication.stage_identity,
        publication.interval,      orientation,
        prepared.route.left_block, prepared.route.right_block};
    const ::pops::amr::InterfaceFluxFragmentMeasure measure{publication.stage_weight,
                                                            prepared.face_measure, point.dt,
                                                            publication.stage_weight_resolved};
    InterfaceFluxFragmentPayload payload(prepared.flux.begin(), prepared.flux.end());
    publication.ledger->accumulate(std::move(key), measure, std::move(payload));
  }

  static bool runtime_field_matches_(const MultiFab& field,
                                     const std::vector<Box2D>& expected_boxes,
                                     const std::vector<int>& expected_ranks, int component_count,
                                     int communicator_rank) {
    int expected_local_size = 0;
    for (const int owner : expected_ranks)
      if (owner == communicator_rank)
        ++expected_local_size;
    return field.box_array().boxes() == expected_boxes && field.dmap().ranks() == expected_ranks &&
           field.local_size() == expected_local_size && field.ncomp() == component_count;
  }

  static void require_distributed_flux_consensus_(std::vector<Real>& flux,
                                                  std::vector<Real>& reference,
                                                  const CommunicatorView& communicator) {
#ifdef POPS_HAS_MPI
    if (reference.size() != flux.size())
      throw std::logic_error("multi-block interface consensus scratch changed size");
    std::copy(flux.begin(), flux.end(), reference.begin());
    broadcast_bytes_inplace(reinterpret_cast<char*>(reference.data()),
                            reference.size() * sizeof(Real), 0, communicator);
    const bool equal = std::memcmp(reference.data(), flux.data(), flux.size() * sizeof(Real)) == 0;
    if (all_reduce_sum(equal ? 0L : 1L, communicator) != 0)
      throw std::runtime_error(
          "multi-block interface evaluator returned rank-dependent shared flux");
    std::copy(reference.begin(), reference.end(), flux.begin());
#else
    (void)flux;
    (void)reference;
    (void)communicator;
    throw std::logic_error(
        "distributed multi-block flux consensus is unavailable in a serial build");
#endif
  }

  static void apply_one_(PreparedInterface& prepared, const BoundaryEvaluationPoint& point,
                         MultiFab& left_state, MultiFab& right_state, MultiFab& left_rhs,
                         MultiFab& right_rhs, InterfaceFluxFragmentPublication* publication) {
    if (prepared.distributed && (!prepared.communicator.active() ||
                                 prepared.communicator.size() != prepared.communicator_size ||
                                 prepared.communicator.rank() != prepared.communicator_rank))
      throw std::runtime_error(
          "multi-block interface execution communicator changed after route preparation");
    const bool layouts_match =
        runtime_field_matches_(left_state, prepared.left_boxes, prepared.left_ranks,
                               prepared.component_count, prepared.communicator_rank) &&
        runtime_field_matches_(right_state, prepared.right_boxes, prepared.right_ranks,
                               prepared.component_count, prepared.communicator_rank) &&
        runtime_field_matches_(left_rhs, prepared.left_boxes, prepared.left_ranks,
                               prepared.component_count, prepared.communicator_rank) &&
        runtime_field_matches_(right_rhs, prepared.right_boxes, prepared.right_ranks,
                               prepared.component_count, prepared.communicator_rank);
    if (prepared.distributed) {
      if (all_reduce_sum(layouts_match ? 0L : 1L, prepared.communicator) != 0)
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
    std::fill(prepared.flux.begin(), prepared.flux.end(), std::numeric_limits<Real>::quiet_NaN());
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
      all_reduce_sum_inplace(prepared.traces.data(), prepared.traces.size(), prepared.communicator);

    const InterfaceFluxBatch batch{left, right, prepared.flux.data(), prepared.face_count,
                                   prepared.component_count};
    std::exception_ptr evaluator_failure;
    try {
      prepared.evaluator(point, batch);  // once per rank, always with the complete prepared pair
    } catch (...) {
      evaluator_failure = std::current_exception();
    }
    if (prepared.distributed) {
      if (all_reduce_sum(evaluator_failure ? 1L : 0L, prepared.communicator) != 0)
        throw std::runtime_error("multi-block interface evaluator failed on one or more MPI ranks");
    } else if (evaluator_failure) {
      std::rethrow_exception(evaluator_failure);
    }
    bool finite_flux = true;
    for (const Real value : prepared.flux)
      finite_flux = finite_flux && std::isfinite(static_cast<double>(value));
    if (prepared.distributed) {
      if (all_reduce_sum(finite_flux ? 0L : 1L, prepared.communicator) != 0)
        throw std::runtime_error(
            "multi-block interface evaluator returned a non-finite flux on one or more MPI ranks");
      require_distributed_flux_consensus_(prepared.flux, prepared.consensus, prepared.communicator);
    } else if (!finite_flux) {
      throw std::runtime_error("multi-block interface evaluator returned a non-finite flux");
    }
    if (publication != nullptr)
      publish_fragment_(prepared, point, *publication);
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
