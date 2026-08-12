/// @file
/// @brief Exact compile-time-ranked conservative interface scheduling.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/numerics/time/amr/reflux/amr_interface_flux_ledger.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/config/generated_component_abi.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::multiblock {

enum class InterfaceSide : std::uint8_t { Low, High };
enum class InterfaceTraceOperation : std::uint8_t {
  Unspecified,
  CellAverage,
  ReconstructedFace,
};

/// Authenticated affine map from right tangential coordinates into canonical left order.
///
/// `right_tangent_for_left[t]` is an ordinal in the right face tangent basis. `sign[t]` is +1 or
/// -1 and `offset[t]` is expressed in physical coordinates.  The empty arrays of the 1-D
/// specialization are the unique zero-dimensional face map; no synthetic transverse coordinate is
/// introduced.
template <int Dim>
struct TangentialTransform {
  static_assert(Dim >= 1 && Dim <= 3, "TangentialTransform only supports dimensions 1, 2, and 3");
  static constexpr int tangent_dimension = Dim - 1;

  std::array<int, tangent_dimension> right_tangent_for_left = [] {
    std::array<int, tangent_dimension> result{};
    for (int tangent = 0; tangent < tangent_dimension; ++tangent)
      result[static_cast<std::size_t>(tangent)] = tangent;
    return result;
  }();
  std::array<int, tangent_dimension> sign = [] {
    std::array<int, tangent_dimension> result{};
    result.fill(1);
    return result;
  }();
  std::array<Real, tangent_dimension> offset{};

  bool operator==(const TangentialTransform&) const = default;
};

/// One exact-ranked connection between two complete, opposite Cartesian faces.
template <int Dim>
struct AxisAlignedInterface {
  static_assert(Dim >= 1 && Dim <= 3, "AxisAlignedInterface only supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;

  std::string identity;
  std::size_t left_block = 0;
  std::size_t right_block = 0;
  int level = 0;
  int left_axis = 0;
  int right_axis = 0;
  InterfaceSide left_side = InterfaceSide::High;
  InterfaceSide right_side = InterfaceSide::Low;
  TangentialTransform<Dim> tangential_transform{};
  std::vector<int> right_component_for_left;

  std::string left_trace_projection_identity;
  std::string right_trace_projection_identity;
  std::string left_trace_provider_identity;
  std::string right_trace_provider_identity;
  InterfaceTraceOperation left_trace_operation = InterfaceTraceOperation::Unspecified;
  InterfaceTraceOperation right_trace_operation = InterfaceTraceOperation::Unspecified;
  int left_trace_required_depth = 0;
  int right_trace_required_depth = 0;

  std::string affine_mapping_identity;
  Real right_normal_translation = Real(0);
};

struct InterfaceFluxBatch {
  const Real* left_state = nullptr;
  const Real* right_state = nullptr;
  const Real* outward_normals = nullptr;
  Real* shared_flux = nullptr;
  int face_count = 0;
  int component_count = 0;
  Real face_measure = Real(0);
  PopsMemorySpaceV1 memory_space = POPS_MEMORY_SPACE_HOST_V1;
};

using InterfaceFluxEvaluator =
    std::function<void(const BoundaryEvaluationPoint&, const InterfaceFluxBatch&)>;
using InterfaceFluxEvaluatorFactory = std::function<InterfaceFluxEvaluator()>;

enum class InterfaceRematerializationAuthority : std::uint8_t {
  RuntimeTopology,
  BindBootstrap,
};

using InterfaceFluxFragmentPayload = std::vector<Real>;
using InterfaceFluxFragmentLedger =
    ::pops::amr::TransactionalInterfaceFluxLedger<InterfaceFluxFragmentPayload>;

struct InterfaceFluxFragmentPublication {
  InterfaceFluxFragmentLedger* ledger = nullptr;
  std::uint64_t topology_epoch = 0;
  int active_level_count = 0;
  ::pops::amr::ClockStamp clock;
  std::string stage_identity;
  ::pops::amr::ClockWindow interval;
  ::pops::amr::Rational stage_weight{1, 1};
  bool stage_weight_resolved = true;
};

struct InterfaceFluxProductionBudget {
  struct Level {
    std::size_t fragment_count_per_application = 0;
    std::size_t payload_terms_per_application = 0;
  };

  std::vector<Level> levels;
  std::size_t maximum_interface_identity_characters = 0;
  std::string exact_contract;
};

/// Prepared conservative interface executor for one immutable compile-time spatial rank.
///
/// Faces are packed in canonical left tangent order.  The exact right permutation/reflection is
/// resolved once during installation. Device fields are packed/scattered in their native memory
/// space; only the compact face batch crosses to pinned host memory for the component ABI and MPI.
template <int Dim>
class InterfaceFluxScheduler {
  static_assert(Dim >= 1 && Dim <= 3,
                "InterfaceFluxScheduler only supports dimensions 1, 2, and 3");

 public:
  static constexpr int dimension = Dim;
  static constexpr int tangent_dimension = Dim - 1;
  using route_type = AxisAlignedInterface<Dim>;
  using field_type = MultiFab<Dim>;
  using geometry_type = Geometry<Dim>;
  using box_type = Box<Dim>;
  using index_type = Index<Dim>;
  using layout_type = typename field_type::layout_type;
  using distribution_type = typename field_type::distribution_type;
  using rank_type = typename field_type::rank_type;
  using ghost_type = typename field_type::ghost_type;
  using memory_space = typename field_type::memory_space;

  void install(route_type route, field_type& left_state, const geometry_type& left_geometry,
               field_type& right_state, const geometry_type& right_geometry,
               const PopsExecutionContextV1& execution,
               InterfaceFluxEvaluatorFactory evaluator_factory) {
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
    std::size_t face_count = 0;
    Real left_normal = Real(0);
    Real right_normal = Real(0);
    Real face_measure = Real(1);
    std::vector<BoundaryCell> left_cells;
    std::vector<BoundaryCell> right_cells;
    std::exception_ptr structural_failure;
    try {
      validate_route_structure_(route);
      if (!evaluator_factory)
        throw std::invalid_argument(
            "multi-block interface has no numerical-flux evaluator factory");
      component::validate_execution_context(execution);
      if (!execution_memory_matches_(execution.memory_space))
        throw std::invalid_argument(
            "multi-block interface execution memory differs from exact field storage");
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
              "multi-block interface execution handles do not identify MPI_DOUBLE authority");
        int relation = MPI_UNEQUAL;
        ::pops::detail::require_mpi_success(
            MPI_Comm_compare(communicator, field_rank_space.native_handle(), &relation),
            "MPI_Comm_compare(interface field rank space)");
        if (relation != MPI_IDENT && relation != MPI_CONGRUENT)
          throw std::invalid_argument(
              "multi-block interface communicator must preserve the field rank space");
        execution_communicator = CommunicatorView{communicator};
        communicator_rank = execution_communicator.rank();
        communicator_size = execution_communicator.size();
        distributed = communicator_size > 1;
#else
        throw std::invalid_argument(
            "multi-block interface received a distributed context from a serial build");
#endif
      } else if (communicator_identity == POPS_EXECUTION_NONCOLLECTIVE_IDENTITY_V1) {
        throw std::invalid_argument(
            "multi-block interface scheduler requires collective execution authority");
#ifdef POPS_HAS_MPI
      } else if (comm_active() && n_ranks() > 1) {
        throw std::invalid_argument(
            "multi-block interface cannot use serial execution in an active MPI world");
#endif
      }

      validate_registry_communicator_(communicator_identity, communicator_size, distributed,
                                      execution_communicator);
      validate_field_rank_space_(left_state, right_state, communicator_rank, communicator_size,
                                 distributed);
      if (left_state.layout().empty() || right_state.layout().empty())
        throw std::invalid_argument("multi-block interface layouts cannot be empty");
      component_count = left_state.ncomp();
      validate_component_permutation_(route, component_count, right_state.ncomp());
      validate_trace_contract_(route);
      validate_unique_route_(route);

      const box_type left_domain = left_state.layout().bounding_box();
      const box_type right_domain = right_state.layout().bounding_box();
      if (!tiles_declared_physical_face_(left_domain, left_geometry, route.left_axis,
                                         route.left_side) ||
          !tiles_declared_physical_face_(right_domain, right_geometry, route.right_axis,
                                         route.right_side))
        throw std::invalid_argument(
            "multi-block interface layout does not tile its declared physical face");

      const auto left_tangents = tangential_axes_(route.left_axis);
      const auto right_tangents = tangential_axes_(route.right_axis);
      face_count = face_count_(left_domain, left_tangents);
      if (face_count == 0 || face_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("multi-block interface face count exceeds the ABI domain");
      validate_tangential_discretization_(route, left_domain, right_domain, left_geometry,
                                          right_geometry, left_tangents, right_tangents);
      validate_physical_coincidence_(route, left_geometry, right_geometry, left_tangents,
                                     right_tangents);
      left_normal = left_geometry.spacing(route.left_axis);
      right_normal = right_geometry.spacing(route.right_axis);
      if (!(left_normal > Real(0)) || left_normal != right_normal)
        throw std::invalid_argument(
            "multi-block interface normal discretisations are not exactly equal");
      face_measure = Real(1);
      for (const int axis : left_tangents)
        face_measure *= left_geometry.spacing(axis);

      left_cells = left_boundary_cells_(left_state, route, left_domain, face_count);
      right_cells =
          right_boundary_cells_(right_state, route, left_domain, right_domain, face_count);
    } catch (...) {
      structural_failure = std::current_exception();
    }
    finish_collective_preflight_(admission_communicator, structural_failure,
                                 "route/layout/execution preflight");

    if (distributed && !registry_agrees_across_ranks_(execution_communicator))
      throw std::runtime_error("multi-block interface prepared registry differs across MPI ranks");
    const std::string collective_identity = collective_plan_identity_(
        route, left_state, left_geometry, right_state, right_geometry, left_normal, right_normal,
        face_count, component_count, communicator_identity, communicator_size);
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
      prepared.route = std::move(route);
      prepared.left_layout = left_state.layout();
      prepared.left_distribution = left_state.distribution();
      prepared.left_rank = left_state.local_rank();
      prepared.left_ghosts = left_state.ghosts();
      prepared.right_layout = right_state.layout();
      prepared.right_distribution = right_state.distribution();
      prepared.right_rank = right_state.local_rank();
      prepared.right_ghosts = right_state.ghosts();
      prepared.left_cells = std::move(left_cells);
      prepared.right_cells = std::move(right_cells);
      prepared.left_normal_spacing = left_normal;
      prepared.right_normal_spacing = right_normal;
      prepared.face_measure = face_measure;
      prepared.face_count = face_count;
      prepared.component_count = component_count;
      prepared.distributed = distributed;
      prepared.communicator = execution_communicator;
      prepared.communicator_rank = communicator_rank;
      prepared.communicator_size = communicator_size;
      prepared.communicator_identity = std::move(communicator_identity);
      prepared.memory_space = execution.memory_space;
      prepared.device_identity = execution.device_identity;
      prepared.collective_identity = collective_identity;
      materialize_storage_(prepared);
    } catch (...) {
      materialization_failure = std::current_exception();
    }
    finish_collective_preflight_(execution_communicator, materialization_failure,
                                 "prepared-route materialization");

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

  void install(route_type route, field_type& left_state, const geometry_type& left_geometry,
               field_type& right_state, const geometry_type& right_geometry,
               const PopsExecutionContextV1& execution, InterfaceFluxEvaluator evaluator) {
    install(std::move(route), left_state, left_geometry, right_state, right_geometry, execution,
            InterfaceFluxEvaluatorFactory(
                [evaluator = std::move(evaluator)]() mutable { return std::move(evaluator); }));
  }

  void apply(const BoundaryEvaluationPoint& point, const std::vector<field_type*>& states,
             const std::vector<field_type*>& rhs,
             InterfaceFluxFragmentPublication* publication = nullptr) {
    apply(point, std::span<field_type* const>(states.data(), states.size()),
          std::span<field_type* const>(rhs.data(), rhs.size()), publication);
  }

  void apply(const BoundaryEvaluationPoint& point, std::span<field_type* const> states,
             std::span<field_type* const> rhs,
             InterfaceFluxFragmentPublication* publication = nullptr) {
    if (interfaces_.empty()) {
      validate_point_(point);
      if (publication != nullptr)
        validate_fragment_publication_(point, *publication);
      return;
    }
    const CommunicatorView communicator = interfaces_.front().communicator;
    const bool collective = communicator.active() && communicator.size() > 1;
    std::exception_ptr point_failure;
    try {
      validate_point_(point);
      if (publication != nullptr)
        validate_fragment_publication_(point, *publication);
    } catch (...) {
      point_failure = std::current_exception();
    }
    finish_collective_preflight_(communicator, point_failure, "evaluation-point preflight");
    if (collective) {
      const long minimum = all_reduce_min(publication != nullptr ? 1L : 0L, communicator);
      const long maximum = all_reduce_max(publication != nullptr ? 1L : 0L, communicator);
      if (minimum != maximum)
        throw std::runtime_error(
            "multi-block interface fragment publication presence differs across MPI ranks");
      if (!registry_agrees_across_ranks_(communicator))
        throw std::runtime_error(
            "multi-block interface prepared registry differs across MPI ranks");
      const std::string point_identity = collective_point_identity_(point);
      if (!all_ranks_agree_exact_ordered_byte_pairs(
              {{std::string_view("point"), std::string_view(point_identity)}}, communicator))
        throw std::runtime_error(
            "multi-block interface BoundaryEvaluationPoint differs across MPI ranks");
      if (publication != nullptr) {
        const std::string identity = collective_fragment_publication_identity_(*publication);
        if (!all_ranks_agree_exact_ordered_byte_pairs(
                {{std::string_view("publication"), std::string_view(identity)}}, communicator))
          throw std::runtime_error(
              "multi-block interface fragment publication differs across MPI ranks");
      }
    }

    for (PreparedInterface& prepared : interfaces_) {
      if (prepared.route.level != point.level)
        continue;
      field_type* left_state = nullptr;
      field_type* right_state = nullptr;
      field_type* left_rhs = nullptr;
      field_type* right_rhs = nullptr;
      bool active = false;
      std::exception_ptr active_failure;
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
        active_failure = std::current_exception();
      }
      finish_collective_preflight_(prepared.communicator, active_failure, "active-mask preflight");
      if (prepared.distributed) {
        const long minimum = all_reduce_min(active ? 1L : 0L, prepared.communicator);
        const long maximum = all_reduce_max(active ? 1L : 0L, prepared.communicator);
        if (minimum != maximum)
          throw std::runtime_error("multi-block interface active mask differs across MPI ranks");
      }
      if (active)
        apply_one_(prepared, point, *left_state, *right_state, *left_rhs, *right_rhs, publication);
    }
  }

  std::size_t size() const noexcept { return interfaces_.size(); }

  InterfaceFluxProductionBudget production_budget(int active_level_count) const {
    if (active_level_count < 1)
      throw std::invalid_argument(
          "multi-block interface production budget requires a positive hierarchy depth");
    InterfaceFluxProductionBudget budget;
    budget.levels.resize(static_cast<std::size_t>(active_level_count));
    ExactContractBuilder exact;
    exact.text("pops.multiblock.interface-flux-production-budget")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(active_level_count)
        .scalar(static_cast<std::uint64_t>(interfaces_.size()));
    for (const PreparedInterface& prepared : interfaces_) {
      if (prepared.route.level < 0 || prepared.route.level >= active_level_count ||
          prepared.component_count < 1)
        throw std::logic_error(
            "multi-block interface production budget contains an invalid prepared route");
      const std::size_t level = static_cast<std::size_t>(prepared.route.level);
      const std::size_t orientations =
          (prepared.route.level > 0 ? std::size_t{1} : std::size_t{0}) +
          (prepared.route.level + 1 < active_level_count ? std::size_t{1} : std::size_t{0});
      if (prepared.face_count > std::numeric_limits<std::size_t>::max() /
                                    static_cast<std::size_t>(prepared.component_count))
        throw std::length_error("multi-block interface payload-term budget exceeds size_t");
      const std::size_t payload =
          prepared.face_count * static_cast<std::size_t>(prepared.component_count);
      if (orientations != 0 && payload > std::numeric_limits<std::size_t>::max() / orientations)
        throw std::length_error("multi-block interface oriented payload budget exceeds size_t");
      auto& row = budget.levels[level];
      if (row.fragment_count_per_application >
              std::numeric_limits<std::size_t>::max() - orientations ||
          row.payload_terms_per_application >
              std::numeric_limits<std::size_t>::max() - orientations * payload)
        throw std::length_error("multi-block interface production budget exceeds size_t");
      row.fragment_count_per_application += orientations;
      row.payload_terms_per_application += orientations * payload;
      budget.maximum_interface_identity_characters =
          std::max(budget.maximum_interface_identity_characters, prepared.route.identity.size());
      exact.text(prepared.route.identity)
          .scalar(prepared.route.level)
          .scalar(static_cast<std::uint64_t>(prepared.route.left_block))
          .scalar(static_cast<std::uint64_t>(prepared.route.right_block))
          .scalar(static_cast<std::uint64_t>(prepared.face_count))
          .scalar(prepared.component_count)
          .scalar(static_cast<std::uint64_t>(orientations));
    }
    for (const auto& level : budget.levels)
      exact.scalar(static_cast<std::uint64_t>(level.fragment_count_per_application))
          .scalar(static_cast<std::uint64_t>(level.payload_terms_per_application));
    exact.scalar(static_cast<std::uint64_t>(budget.maximum_interface_identity_characters));
    budget.exact_contract = std::move(exact).release();
    return budget;
  }

  /// Configured-depth ceiling authenticated by the installed logical interface registry and one
  /// hierarchy-owned full-face capacity per possible level.  Runtime routes are installed only for
  /// materialized levels; treating their current face counts as future-level authority would
  /// underbound a restart/regrid.  The level-zero rows are the immutable authored interface set,
  /// while the caller's capacities are derived from the sealed configured domains.
  InterfaceFluxProductionBudget production_budget(
      int configured_level_count, std::span<const std::size_t> configured_face_capacities) const {
    if (configured_level_count < 1 ||
        configured_face_capacities.size() != static_cast<std::size_t>(configured_level_count) ||
        std::any_of(
            configured_face_capacities.begin(), configured_face_capacities.end(),
            [](std::size_t value) { return value == 0; }))
      throw std::invalid_argument(
          "multi-block interface configured production budget has an invalid hierarchy shape");
    if (interfaces_.empty()) {
      InterfaceFluxProductionBudget budget;
      budget.levels.resize(static_cast<std::size_t>(configured_level_count));
      ExactContractBuilder exact;
      exact.text("pops.multiblock.interface-flux-configured-production-budget")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .scalar(configured_level_count)
          .scalar(std::uint64_t{0});
      for (const std::size_t capacity : configured_face_capacities)
        exact.scalar(static_cast<std::uint64_t>(capacity));
      budget.exact_contract = std::move(exact).release();
      return budget;
    }

    struct LogicalInterface {
      std::string identity;
      std::size_t components = 0;
    };
    std::vector<LogicalInterface> logical;
    logical.reserve(interfaces_.size());
    for (const PreparedInterface& prepared : interfaces_) {
      if (prepared.route.level != 0)
        continue;
      if (prepared.route.identity.empty() || prepared.component_count < 1 ||
          std::find_if(
              logical.begin(), logical.end(),
              [&](const LogicalInterface& row) {
                return row.identity == prepared.route.identity;
              }) != logical.end())
        throw std::logic_error(
            "multi-block interface configured budget has a malformed level-zero authority");
      logical.push_back(
          {prepared.route.identity, static_cast<std::size_t>(prepared.component_count)});
    }
    if (logical.empty())
      throw std::logic_error(
          "multi-block interface configured budget lacks its level-zero authored registry");
    std::sort(logical.begin(), logical.end(),
              [](const auto& left, const auto& right) { return left.identity < right.identity; });

    // Every already materialized route must be one exact image of a level-zero authored identity.
    // Missing configured levels are then bounded, not silently represented as zero production.
    std::vector<std::vector<std::string>> seen(static_cast<std::size_t>(configured_level_count));
    for (const PreparedInterface& prepared : interfaces_) {
      if (prepared.route.level < 0 || prepared.route.level >= configured_level_count ||
          prepared.component_count < 1)
        throw std::logic_error(
            "multi-block interface configured budget contains an out-of-range route");
      const auto authority =
          std::lower_bound(logical.begin(), logical.end(), prepared.route.identity,
                           [](const LogicalInterface& row, std::string_view identity) {
                             return row.identity < identity;
                           });
      if (authority == logical.end() || authority->identity != prepared.route.identity ||
          authority->components != static_cast<std::size_t>(prepared.component_count))
        throw std::logic_error(
            "multi-block interface configured budget route differs from level-zero authority");
      auto& identities = seen[static_cast<std::size_t>(prepared.route.level)];
      if (std::find(identities.begin(), identities.end(), prepared.route.identity) !=
          identities.end())
        throw std::logic_error(
            "multi-block interface configured budget contains a duplicate level route");
      identities.push_back(prepared.route.identity);
    }

    InterfaceFluxProductionBudget budget;
    budget.levels.resize(static_cast<std::size_t>(configured_level_count));
    ExactContractBuilder exact;
    exact.text("pops.multiblock.interface-flux-configured-production-budget")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(configured_level_count)
        .scalar(static_cast<std::uint64_t>(logical.size()));
    for (const LogicalInterface& interface : logical) {
      budget.maximum_interface_identity_characters =
          std::max(budget.maximum_interface_identity_characters, interface.identity.size());
      exact.text(interface.identity).scalar(static_cast<std::uint64_t>(interface.components));
    }
    for (int level = 0; level < configured_level_count; ++level) {
      const std::size_t orientations =
          (level > 0 ? std::size_t{1} : std::size_t{0}) +
          (level + 1 < configured_level_count ? std::size_t{1} : std::size_t{0});
      auto& row = budget.levels[static_cast<std::size_t>(level)];
      if (orientations != 0 &&
          logical.size() > std::numeric_limits<std::size_t>::max() / orientations)
        throw std::length_error("multi-block interface configured fragment budget exceeds size_t");
      row.fragment_count_per_application = orientations * logical.size();
      std::size_t terms = 0;
      for (const LogicalInterface& interface : logical) {
        if (configured_face_capacities[static_cast<std::size_t>(level)] >
            std::numeric_limits<std::size_t>::max() / interface.components)
          throw std::length_error("multi-block interface configured payload budget exceeds size_t");
        const std::size_t interface_terms =
            configured_face_capacities[static_cast<std::size_t>(level)] * interface.components;
        if (interface_terms > std::numeric_limits<std::size_t>::max() - terms)
          throw std::length_error("multi-block interface configured payload budget exceeds size_t");
        terms += interface_terms;
      }
      if (orientations != 0 && terms > std::numeric_limits<std::size_t>::max() / orientations)
        throw std::length_error(
            "multi-block interface configured oriented payload budget exceeds size_t");
      row.payload_terms_per_application = orientations * terms;
      exact.scalar(static_cast<std::uint64_t>(level))
          .scalar(static_cast<std::uint64_t>(
              configured_face_capacities[static_cast<std::size_t>(level)]))
          .scalar(static_cast<std::uint64_t>(row.fragment_count_per_application))
          .scalar(static_cast<std::uint64_t>(row.payload_terms_per_application));
    }
    budget.exact_contract = std::move(exact).release();
    return budget;
  }
  void rollback_installations(std::size_t accepted_size) {
    if (accepted_size > interfaces_.size())
      throw std::runtime_error(
          "multi-block interface rollback lost part of the accepted registry prefix");
    interfaces_.resize(accepted_size);
  }
  void clear() { interfaces_.clear(); }

  bool has_interfaces(int level) const noexcept {
    return std::any_of(interfaces_.begin(), interfaces_.end(),
                       [level](const PreparedInterface& p) { return p.route.level == level; });
  }

  void require_exact_jacvec_pair(int level, std::size_t first_block,
                                 std::size_t second_block) const {
    if (level < 0 || first_block == second_block)
      throw std::invalid_argument("multi-block implicit JVP pair is invalid");
    std::size_t level_routes = 0;
    const PreparedInterface* matched = nullptr;
    for (const PreparedInterface& prepared : interfaces_) {
      if (prepared.route.level != level)
        continue;
      ++level_routes;
      if ((prepared.route.left_block == first_block &&
           prepared.route.right_block == second_block) ||
          (prepared.route.left_block == second_block && prepared.route.right_block == first_block))
        matched = &prepared;
    }
    if (level_routes != 1 || matched == nullptr)
      throw std::runtime_error(
          "multi-block implicit JVP requires one exact prepared two-block interface route");
    if (matched->distributed || matched->communicator_size != 1 ||
        matched->communicator_identity != "serial")
      throw std::runtime_error("multi-block implicit JVP requires serial rank-one execution");
  }

  template <class StateProvider, class GeometryProvider>
  InterfaceFluxScheduler rematerialized(
      int active_level_count, StateProvider&& state_provider, GeometryProvider&& geometry_provider,
      InterfaceRematerializationAuthority authority =
          InterfaceRematerializationAuthority::RuntimeTopology) const {
    if (active_level_count < 1)
      throw std::invalid_argument(
          "multi-block interface rematerialization requires a positive active level count");
    if (authority != InterfaceRematerializationAuthority::RuntimeTopology &&
        authority != InterfaceRematerializationAuthority::BindBootstrap)
      throw std::invalid_argument(
          "multi-block interface rematerialization has an invalid lifecycle authority");
    const CommunicatorView communicator =
        interfaces_.empty() ? CommunicatorView{} : interfaces_.front().communicator;
    InterfaceFluxScheduler candidate;
    std::exception_ptr allocation_failure;
    try {
      candidate.interfaces_.reserve(interfaces_.size());
    } catch (...) {
      allocation_failure = std::current_exception();
    }
    finish_collective_preflight_(communicator, allocation_failure,
                                 "replacement registry allocation");
    for (const PreparedInterface& prepared : interfaces_) {
      PreparedInterface replacement;
      std::exception_ptr failure;
      try {
        if (prepared.route.level < 0 || prepared.route.level >= active_level_count)
          throw std::runtime_error(
              "multi-block interface replacement changed the active hierarchy depth");
        field_type& left =
            std::invoke(state_provider, prepared.route.left_block, prepared.route.level);
        field_type& right =
            std::invoke(state_provider, prepared.route.right_block, prepared.route.level);
        const geometry_type geometry = std::invoke(geometry_provider, prepared.route.level);
        replacement = rematerialize_prepared_(prepared, left, geometry, right, geometry);
      } catch (...) {
        failure = std::current_exception();
      }
      finish_collective_preflight_(communicator, failure, "replacement route/layout preflight");
      std::exception_ptr storage_failure;
      try {
        candidate.interfaces_.push_back(std::move(replacement));
      } catch (...) {
        storage_failure = std::current_exception();
      }
      finish_collective_preflight_(communicator, storage_failure,
                                   "replacement registry materialization");
    }
    std::exception_ptr completeness_failure;
    try {
      const bool incremental_bind_prefix =
          authority == InterfaceRematerializationAuthority::BindBootstrap &&
          !candidate.has_interfaces(active_level_count - 1);
      if (!incremental_bind_prefix)
        candidate.require_complete_active_level_registry_(active_level_count);
    } catch (...) {
      completeness_failure = std::current_exception();
    }
    finish_collective_preflight_(communicator, completeness_failure,
                                 "replacement registry structural completeness");
    if (communicator.active() && communicator.size() > 1 &&
        !candidate.registry_agrees_across_ranks_(communicator))
      throw std::runtime_error(
          "multi-block interface replacement registry differs across MPI ranks");
    return candidate;
  }

  void swap(InterfaceFluxScheduler& other) noexcept { interfaces_.swap(other.interfaces_); }

  void require_complete_active_level_registry(int active_level_count) const {
    if (active_level_count < 1)
      throw std::invalid_argument(
          "multi-block interface registry validation requires a positive active level count");
    require_complete_active_level_registry_(active_level_count);
  }

  void require_runtime_rematerialization_ready(int active_level_count) const {
    const CommunicatorView communicator =
        interfaces_.empty() ? CommunicatorView{} : interfaces_.front().communicator;
    std::exception_ptr failure;
    try {
      require_complete_active_level_registry(active_level_count);
    } catch (...) {
      failure = std::current_exception();
    }
    finish_collective_preflight_(communicator, failure,
                                 "accepted registry structural rematerialization preflight");
    if (communicator.active() && communicator.size() > 1 &&
        !registry_agrees_across_ranks_(communicator))
      throw std::runtime_error("multi-block interface accepted registry differs across MPI ranks");
  }

  bool participates(std::size_t block, int level) const noexcept {
    return std::any_of(
        interfaces_.begin(), interfaces_.end(), [block, level](const PreparedInterface& prepared) {
          return prepared.route.level == level &&
                 (prepared.route.left_block == block || prepared.route.right_block == block);
        });
  }

  bool owns_face(std::size_t block, int level, int axis, InterfaceSide side) const noexcept {
    for (const PreparedInterface& prepared : interfaces_) {
      const route_type& route = prepared.route;
      if (route.level == level &&
          ((route.left_block == block && route.left_axis == axis && route.left_side == side) ||
           (route.right_block == block && route.right_axis == axis && route.right_side == side)))
        return true;
    }
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
    std::size_t local_box = field_type::not_local;
    index_type index{};
    int face = 0;
  };

  struct DeviceFaceJob {
    index_type index{};
    int face = 0;
  };
  using device_job_view = Kokkos::View<DeviceFaceJob*, memory_space>;
  using device_real_view = Kokkos::View<Real*, memory_space>;
  using host_real_view = Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace>;

  struct LocalFacePlan {
    std::size_t local_box = field_type::not_local;
    device_job_view jobs{};
  };

  struct PreparedInterface {
    route_type route;
    layout_type left_layout;
    distribution_type left_distribution;
    rank_type left_rank{};
    ghost_type left_ghosts{};
    layout_type right_layout;
    distribution_type right_distribution;
    rank_type right_rank{};
    ghost_type right_ghosts{};
    std::vector<BoundaryCell> left_cells;
    std::vector<BoundaryCell> right_cells;
    std::vector<LocalFacePlan> left_plans;
    std::vector<LocalFacePlan> right_plans;
    device_real_view device_traces{};
    device_real_view device_flux{};
    device_real_view device_normals{};
    Kokkos::View<int*, memory_space> device_right_components{};
    host_real_view host_traces{};
    host_real_view host_flux{};
    host_real_view host_normals{};
    host_real_view host_consensus{};
    Real left_normal_spacing = Real(0);
    Real right_normal_spacing = Real(0);
    Real face_measure = Real(1);
    std::size_t face_count = 0;
    int component_count = 0;
    bool distributed = false;
    CommunicatorView communicator;
    int communicator_rank = 0;
    int communicator_size = 1;
    std::string communicator_identity;
    PopsMemorySpaceV1 memory_space = POPS_MEMORY_SPACE_HOST_V1;
    std::string device_identity;
    std::string collective_identity;
    InterfaceFluxEvaluator evaluator;
    std::size_t evaluation_count = 0;
  };

  struct PackKernel {
    FieldView<const Real, Dim> field{};
    device_job_view jobs{};
    device_real_view traces{};
    Kokkos::View<int*, memory_space> component_map{};
    std::size_t trace_base = 0;
    int component_count = 0;
    bool map_components = false;

    POPS_HD void operator()(std::int64_t element) const {
      const int component = static_cast<int>(element % component_count);
      const std::size_t job = static_cast<std::size_t>(element / component_count);
      const DeviceFaceJob selected = jobs(job);
      const int stored_component = map_components ? component_map(component) : component;
      traces(trace_base + static_cast<std::size_t>(selected.face) * component_count + component) =
          field(selected.index, stored_component);
    }
  };

  struct ScatterKernel {
    FieldView<Real, Dim> field{};
    device_job_view jobs{};
    device_real_view flux{};
    Kokkos::View<int*, memory_space> component_map{};
    Real scale = Real(0);
    int component_count = 0;
    bool map_components = false;

    POPS_HD void operator()(std::int64_t element) const {
      const int component = static_cast<int>(element % component_count);
      const std::size_t job = static_cast<std::size_t>(element / component_count);
      const DeviceFaceJob selected = jobs(job);
      const int stored_component = map_components ? component_map(component) : component;
      field(selected.index, stored_component) +=
          scale * flux(static_cast<std::size_t>(selected.face) * component_count + component);
    }
  };

  static void validate_route_structure_(const route_type& route) {
    if (route.identity.empty() || route.left_block == route.right_block || route.level < 0)
      throw std::invalid_argument("multi-block interface identity/ownership is invalid");
    validate_axis_(route.left_axis);
    validate_axis_(route.right_axis);
    if (route.left_side == route.right_side)
      throw std::invalid_argument("multi-block interface faces do not have opposite orientation");
    std::array<bool, tangent_dimension> seen{};
    for (int tangent = 0; tangent < tangent_dimension; ++tangent) {
      const int mapped = route.tangential_transform.right_tangent_for_left[tangent];
      const int sign = route.tangential_transform.sign[tangent];
      const Real offset = route.tangential_transform.offset[tangent];
      if (mapped < 0 || mapped >= tangent_dimension || seen[static_cast<std::size_t>(mapped)])
        throw std::invalid_argument(
            "multi-block interface tangential permutation is not bijective");
      if ((sign != -1 && sign != 1) || !std::isfinite(static_cast<double>(offset)))
        throw std::invalid_argument("multi-block interface tangential affine map is invalid");
      seen[static_cast<std::size_t>(mapped)] = true;
    }
    if (!std::isfinite(static_cast<double>(route.right_normal_translation)))
      throw std::invalid_argument("multi-block interface normal translation is invalid");
    if (route.affine_mapping_identity.empty()) {
      if (route.right_normal_translation != Real(0))
        throw std::invalid_argument("multi-block interface affine map is unauthenticated");
      for (int tangent = 0; tangent < tangent_dimension; ++tangent)
        if (route.tangential_transform.right_tangent_for_left[tangent] != tangent ||
            route.tangential_transform.sign[tangent] != 1 ||
            route.tangential_transform.offset[tangent] != Real(0))
          throw std::invalid_argument("multi-block interface affine map is unauthenticated");
    }
  }

  static void validate_axis_(int axis) {
    if (axis < 0 || axis >= Dim)
      throw std::invalid_argument(
          "multi-block interface axis is outside the compile-time spatial rank");
  }

  static constexpr bool execution_memory_matches_(PopsMemorySpaceV1 claimed) {
    if constexpr (std::is_same_v<memory_space, Kokkos::HostSpace>)
      return claimed == POPS_MEMORY_SPACE_HOST_V1;
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible)
      return claimed == POPS_MEMORY_SPACE_MANAGED_V1;
    return claimed == POPS_MEMORY_SPACE_DEVICE_V1;
  }

  void validate_registry_communicator_(std::string_view identity, int size, bool distributed,
                                       const CommunicatorView& communicator) const {
    if (interfaces_.empty())
      return;
    const PreparedInterface& existing = interfaces_.front();
    if (existing.communicator_identity != identity || existing.communicator_size != size)
      throw std::invalid_argument(
          "multi-block interface routes require one exact execution communicator");
#ifdef POPS_HAS_MPI
    if (distributed) {
      int relation = MPI_UNEQUAL;
      ::pops::detail::require_mpi_success(MPI_Comm_compare(existing.communicator.native_handle(),
                                                           communicator.native_handle(), &relation),
                                          "MPI_Comm_compare(installed interface communicators)");
      if (relation != MPI_IDENT)
        throw std::invalid_argument(
            "multi-block interface routes require the same communicator context");
    }
#else
    (void)distributed;
    (void)communicator;
#endif
  }

  static void validate_field_rank_space_(const field_type& left, const field_type& right,
                                         int communicator_rank, int communicator_size,
                                         bool distributed) {
    if (left.rank_space() != right.rank_space() || left.local_rank() != right.local_rank())
      throw std::invalid_argument(
          "multi-block interface endpoints use different exact process spaces");
    const auto& ranks = left.rank_space();
    if (ranks.size() != static_cast<std::size_t>(communicator_size) ||
        ranks.linear_rank(left.local_rank()) != static_cast<std::size_t>(communicator_rank))
      throw std::invalid_argument(
          "multi-block interface process coordinates do not match the communicator");
    if (distributed && (left.distribution().replicated() || right.distribution().replicated()))
      throw std::invalid_argument(
          "distributed multi-block interfaces require unique patch ownership");
  }

  static void validate_component_permutation_(const route_type& route, int left_components,
                                              int right_components) {
    if (left_components < 1 || right_components != left_components ||
        route.right_component_for_left.size() != static_cast<std::size_t>(left_components))
      throw std::invalid_argument("multi-block interface component spaces are not equal");
    std::vector<bool> seen(static_cast<std::size_t>(left_components), false);
    for (const int component : route.right_component_for_left) {
      if (component < 0 || component >= left_components || seen[component])
        throw std::invalid_argument("multi-block interface component permutation is not bijective");
      seen[component] = true;
    }
  }

  static void validate_trace_contract_(const route_type& route) {
    if (route.left_trace_projection_identity.empty() ||
        route.right_trace_projection_identity.empty() ||
        route.left_trace_provider_identity.empty() || route.right_trace_provider_identity.empty() ||
        route.left_trace_required_depth < 1 || route.right_trace_required_depth < 1)
      throw std::invalid_argument("multi-block interface trace projection contract is incomplete");
    if (route.left_trace_operation != InterfaceTraceOperation::CellAverage ||
        route.right_trace_operation != InterfaceTraceOperation::CellAverage)
      throw std::invalid_argument(
          "multi-block interface reconstructed face projection has no executable provider");
  }

  void validate_unique_route_(const route_type& route) const {
    const auto claims = [](const route_type& candidate, std::size_t block, int axis,
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
          (claims(installed.route, route.left_block, route.left_axis, route.left_side) ||
           claims(installed.route, route.right_block, route.right_axis, route.right_side)))
        throw std::invalid_argument(
            "multi-block interface endpoint face is already owned on level");
    }
  }

  static std::array<int, tangent_dimension> tangential_axes_(int normal_axis) {
    std::array<int, tangent_dimension> result{};
    int tangent = 0;
    for (int axis = 0; axis < Dim; ++axis)
      if (axis != normal_axis)
        result[static_cast<std::size_t>(tangent++)] = axis;
    return result;
  }

  static std::size_t face_count_(const box_type& domain,
                                 const std::array<int, tangent_dimension>& tangents) {
    std::size_t count = 1;
    for (const int axis : tangents) {
      const std::uint64_t extent = static_cast<std::uint64_t>(domain.length(axis));
      if (extent == 0 || extent > std::numeric_limits<std::size_t>::max() / count)
        throw std::overflow_error("multi-block interface face count overflows size_t");
      count *= static_cast<std::size_t>(extent);
    }
    return count;
  }

  static bool tiles_declared_physical_face_(const box_type& box, const geometry_type& geometry,
                                            int axis, InterfaceSide side) {
    const int normal = side == InterfaceSide::Low ? box.lo[axis] : box.hi[axis];
    const int expected =
        side == InterfaceSide::Low ? geometry.domain().lo[axis] : geometry.domain().hi[axis];
    if (normal != expected)
      return false;
    for (int tangent = 0; tangent < Dim; ++tangent)
      if (tangent != axis && (box.lo[tangent] != geometry.domain().lo[tangent] ||
                              box.hi[tangent] != geometry.domain().hi[tangent]))
        return false;
    return true;
  }

  static void validate_tangential_discretization_(
      const route_type& route, const box_type& left_domain, const box_type& right_domain,
      const geometry_type& left_geometry, const geometry_type& right_geometry,
      const std::array<int, tangent_dimension>& left_tangents,
      const std::array<int, tangent_dimension>& right_tangents) {
    for (int tangent = 0; tangent < tangent_dimension; ++tangent) {
      const int left_axis = left_tangents[tangent];
      const int mapped = route.tangential_transform.right_tangent_for_left[tangent];
      const int right_axis = right_tangents[mapped];
      if (left_domain.length(left_axis) != right_domain.length(right_axis) ||
          left_geometry.spacing(left_axis) != right_geometry.spacing(right_axis))
        throw std::invalid_argument(
            "multi-block interface tangential discretisations are not exactly equal");
    }
  }

  static void validate_physical_coincidence_(
      const route_type& route, const geometry_type& left_geometry,
      const geometry_type& right_geometry, const std::array<int, tangent_dimension>& left_tangents,
      const std::array<int, tangent_dimension>& right_tangents) {
    const Real left_normal = route.left_side == InterfaceSide::Low
                                 ? left_geometry.lower()[route.left_axis]
                                 : left_geometry.upper()[route.left_axis];
    const Real right_normal = route.right_side == InterfaceSide::Low
                                  ? right_geometry.lower()[route.right_axis]
                                  : right_geometry.upper()[route.right_axis];
    if (left_normal != right_normal + route.right_normal_translation)
      throw std::invalid_argument(
          "multi-block interface normal faces do not coincide in physical space");
    for (int tangent = 0; tangent < tangent_dimension; ++tangent) {
      const int left_axis = left_tangents[tangent];
      const int right_axis =
          right_tangents[route.tangential_transform.right_tangent_for_left[tangent]];
      const Real sign = static_cast<Real>(route.tangential_transform.sign[tangent]);
      const Real offset = route.tangential_transform.offset[tangent];
      const Real right_low =
          sign > Real(0) ? right_geometry.lower()[right_axis] : right_geometry.upper()[right_axis];
      const Real right_high =
          sign > Real(0) ? right_geometry.upper()[right_axis] : right_geometry.lower()[right_axis];
      if (left_geometry.lower()[left_axis] != sign * right_low + offset ||
          left_geometry.upper()[left_axis] != sign * right_high + offset)
        throw std::invalid_argument(
            "multi-block interface tangential faces do not coincide in physical space");
    }
  }

  static std::array<int, tangent_dimension> decode_face_(
      std::size_t face, const box_type& domain,
      const std::array<int, tangent_dimension>& tangents) {
    std::array<int, tangent_dimension> offsets{};
    for (int tangent = 0; tangent < tangent_dimension; ++tangent) {
      const std::size_t extent = static_cast<std::size_t>(domain.length(tangents[tangent]));
      offsets[tangent] = static_cast<int>(face % extent);
      face /= extent;
    }
    return offsets;
  }

  static index_type left_face_index_(const route_type& route, const box_type& left_domain,
                                     std::size_t face) {
    const auto tangents = tangential_axes_(route.left_axis);
    const auto offsets = decode_face_(face, left_domain, tangents);
    index_type index{};
    index[route.left_axis] = route.left_side == InterfaceSide::Low
                                 ? left_domain.lo[route.left_axis]
                                 : left_domain.hi[route.left_axis];
    for (int tangent = 0; tangent < tangent_dimension; ++tangent)
      index[tangents[tangent]] = left_domain.lo[tangents[tangent]] + offsets[tangent];
    return index;
  }

  static index_type right_face_index_(const route_type& route, const box_type& left_domain,
                                      const box_type& right_domain, std::size_t face) {
    const auto left_tangents = tangential_axes_(route.left_axis);
    const auto right_tangents = tangential_axes_(route.right_axis);
    const auto offsets = decode_face_(face, left_domain, left_tangents);
    index_type index{};
    index[route.right_axis] = route.right_side == InterfaceSide::Low
                                  ? right_domain.lo[route.right_axis]
                                  : right_domain.hi[route.right_axis];
    for (int tangent = 0; tangent < tangent_dimension; ++tangent) {
      const int mapped = route.tangential_transform.right_tangent_for_left[tangent];
      const int right_axis = right_tangents[mapped];
      index[right_axis] = route.tangential_transform.sign[tangent] > 0
                              ? right_domain.lo[right_axis] + offsets[tangent]
                              : right_domain.hi[right_axis] - offsets[tangent];
    }
    return index;
  }

  template <class IndexFactory>
  static std::vector<BoundaryCell> boundary_cells_(const field_type& field, std::size_t face_count,
                                                   IndexFactory&& make_index) {
    std::vector<BoundaryCell> cells;
    cells.reserve(face_count);
    for (std::size_t face = 0; face < face_count; ++face) {
      const index_type index = std::invoke(make_index, face);
      std::size_t global_owner = field_type::not_local;
      for (std::size_t global = 0; global < field.layout().size(); ++global) {
        if (!field.layout()[global].contains(index))
          continue;
        if (global_owner != field_type::not_local)
          throw std::invalid_argument(
              "multi-block interface boundary decomposition overlaps at one face cell");
        global_owner = global;
      }
      if (global_owner == field_type::not_local)
        throw std::invalid_argument(
            "multi-block interface boundary decomposition has a gap at one face cell");
      cells.push_back(
          BoundaryCell{field.local_index_of(global_owner), index, static_cast<int>(face)});
    }
    return cells;
  }

  static std::vector<BoundaryCell> left_boundary_cells_(const field_type& field,
                                                        const route_type& route,
                                                        const box_type& domain,
                                                        std::size_t face_count) {
    return boundary_cells_(field, face_count,
                           [&](std::size_t face) { return left_face_index_(route, domain, face); });
  }

  static std::vector<BoundaryCell> right_boundary_cells_(const field_type& field,
                                                         const route_type& route,
                                                         const box_type& left_domain,
                                                         const box_type& right_domain,
                                                         std::size_t face_count) {
    return boundary_cells_(field, face_count, [&](std::size_t face) {
      return right_face_index_(route, left_domain, right_domain, face);
    });
  }

  static std::vector<LocalFacePlan> local_plans_(const std::vector<BoundaryCell>& cells,
                                                 std::size_t local_size, std::string_view label) {
    std::vector<std::vector<DeviceFaceJob>> grouped(local_size);
    for (const BoundaryCell& cell : cells)
      if (cell.local_box != field_type::not_local)
        grouped.at(cell.local_box).push_back(DeviceFaceJob{cell.index, cell.face});
    std::vector<LocalFacePlan> plans;
    for (std::size_t local = 0; local < grouped.size(); ++local) {
      if (grouped[local].empty())
        continue;
      device_job_view jobs(std::string(label) + "_jobs", grouped[local].size());
      auto host = Kokkos::create_mirror_view(jobs);
      for (std::size_t job = 0; job < grouped[local].size(); ++job)
        host(job) = grouped[local][job];
      Kokkos::deep_copy(jobs, host);
      plans.push_back(LocalFacePlan{local, std::move(jobs)});
    }
    return plans;
  }

  static void materialize_storage_(PreparedInterface& prepared) {
    const std::size_t packed =
        prepared.face_count * static_cast<std::size_t>(prepared.component_count);
    if (packed > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 2)
      throw std::overflow_error(
          "multi-block interface trace batch exceeds the native MPI count domain");
    prepared.left_plans =
        local_plans_(prepared.left_cells,
                     prepared.left_distribution.local_box_indices(prepared.left_rank).size(),
                     "pops_interface_left");
    prepared.right_plans =
        local_plans_(prepared.right_cells,
                     prepared.right_distribution.local_box_indices(prepared.right_rank).size(),
                     "pops_interface_right");
    prepared.device_traces = device_real_view("pops_interface_traces", 2 * packed);
    prepared.device_flux = device_real_view("pops_interface_flux", packed);
    prepared.device_normals =
        device_real_view("pops_interface_outward_normals", prepared.face_count * Dim);
    prepared.device_right_components =
        Kokkos::View<int*, memory_space>("pops_interface_component_map", prepared.component_count);
    auto component_host = Kokkos::create_mirror_view(prepared.device_right_components);
    for (int component = 0; component < prepared.component_count; ++component)
      component_host(component) = prepared.route.right_component_for_left[component];
    Kokkos::deep_copy(prepared.device_right_components, component_host);
    prepared.host_traces = host_real_view("pops_interface_host_traces", 2 * packed);
    prepared.host_flux = host_real_view("pops_interface_host_flux", packed);
    prepared.host_normals =
        host_real_view("pops_interface_host_normals", prepared.face_count * Dim);
    prepared.host_consensus = host_real_view("pops_interface_host_consensus", packed);
    std::fill_n(prepared.host_normals.data(), prepared.host_normals.extent(0), Real(0));
    const Real outward_sign = prepared.route.left_side == InterfaceSide::Low ? Real(-1) : Real(1);
    for (std::size_t face = 0; face < prepared.face_count; ++face)
      prepared.host_normals(face * Dim + static_cast<std::size_t>(prepared.route.left_axis)) =
          outward_sign;
    Kokkos::deep_copy(prepared.device_normals, prepared.host_normals);
  }

  static PreparedInterface rematerialize_prepared_(const PreparedInterface& prepared,
                                                   field_type& left_state,
                                                   const geometry_type& left_geometry,
                                                   field_type& right_state,
                                                   const geometry_type& right_geometry) {
    validate_field_rank_space_(left_state, right_state, prepared.communicator_rank,
                               prepared.communicator_size, prepared.distributed);
    validate_component_permutation_(prepared.route, left_state.ncomp(), right_state.ncomp());
    const box_type left_domain = left_state.layout().bounding_box();
    const box_type right_domain = right_state.layout().bounding_box();
    if (!tiles_declared_physical_face_(left_domain, left_geometry, prepared.route.left_axis,
                                       prepared.route.left_side) ||
        !tiles_declared_physical_face_(right_domain, right_geometry, prepared.route.right_axis,
                                       prepared.route.right_side))
      throw std::invalid_argument(
          "multi-block interface replacement does not tile its declared physical face");
    const auto left_tangents = tangential_axes_(prepared.route.left_axis);
    const auto right_tangents = tangential_axes_(prepared.route.right_axis);
    const std::size_t faces = face_count_(left_domain, left_tangents);
    validate_tangential_discretization_(prepared.route, left_domain, right_domain, left_geometry,
                                        right_geometry, left_tangents, right_tangents);
    validate_physical_coincidence_(prepared.route, left_geometry, right_geometry, left_tangents,
                                   right_tangents);
    PreparedInterface replacement = prepared;
    replacement.left_layout = left_state.layout();
    replacement.left_distribution = left_state.distribution();
    replacement.left_rank = left_state.local_rank();
    replacement.left_ghosts = left_state.ghosts();
    replacement.right_layout = right_state.layout();
    replacement.right_distribution = right_state.distribution();
    replacement.right_rank = right_state.local_rank();
    replacement.right_ghosts = right_state.ghosts();
    replacement.left_cells = left_boundary_cells_(left_state, prepared.route, left_domain, faces);
    replacement.right_cells =
        right_boundary_cells_(right_state, prepared.route, left_domain, right_domain, faces);
    replacement.left_normal_spacing = left_geometry.spacing(prepared.route.left_axis);
    replacement.right_normal_spacing = right_geometry.spacing(prepared.route.right_axis);
    replacement.face_measure = Real(1);
    for (const int axis : left_tangents)
      replacement.face_measure *= left_geometry.spacing(axis);
    replacement.face_count = faces;
    replacement.collective_identity = collective_plan_identity_(
        prepared.route, left_state, left_geometry, right_state, right_geometry,
        replacement.left_normal_spacing, replacement.right_normal_spacing, faces,
        prepared.component_count, prepared.communicator_identity, prepared.communicator_size);
    materialize_storage_(replacement);
    return replacement;
  }

  static bool runtime_field_matches_(const field_type& field, const layout_type& layout,
                                     const distribution_type& distribution, const rank_type& rank,
                                     const ghost_type& ghosts, int components) {
    return field.layout() == layout && field.distribution() == distribution &&
           field.local_rank() == rank && field.ghosts() == ghosts && field.ncomp() == components;
  }

  static void pack_(PreparedInterface& prepared, const field_type& left, const field_type& right) {
    Kokkos::deep_copy(prepared.device_traces, Real(0));
    const std::size_t packed =
        prepared.face_count * static_cast<std::size_t>(prepared.component_count);
    for (const LocalFacePlan& plan : prepared.left_plans) {
      const std::int64_t elements =
          static_cast<std::int64_t>(plan.jobs.extent(0)) * prepared.component_count;
      Kokkos::parallel_for(
          "pops_interface_pack_left", Kokkos::RangePolicy<>(0, elements),
          PackKernel{left.fab(plan.local_box).view(), plan.jobs, prepared.device_traces,
                     prepared.device_right_components, 0, prepared.component_count, false});
    }
    for (const LocalFacePlan& plan : prepared.right_plans) {
      const std::int64_t elements =
          static_cast<std::int64_t>(plan.jobs.extent(0)) * prepared.component_count;
      Kokkos::parallel_for(
          "pops_interface_pack_right", Kokkos::RangePolicy<>(0, elements),
          PackKernel{right.fab(plan.local_box).view(), plan.jobs, prepared.device_traces,
                     prepared.device_right_components, packed, prepared.component_count, true});
    }
    Kokkos::deep_copy(prepared.host_traces, prepared.device_traces);
  }

  static void scatter_(PreparedInterface& prepared, field_type& left, field_type& right) {
    Kokkos::deep_copy(prepared.device_flux, prepared.host_flux);
    for (const LocalFacePlan& plan : prepared.left_plans) {
      const std::int64_t elements =
          static_cast<std::int64_t>(plan.jobs.extent(0)) * prepared.component_count;
      Kokkos::parallel_for(
          "pops_interface_scatter_left", Kokkos::RangePolicy<>(0, elements),
          ScatterKernel{left.fab(plan.local_box).view(), plan.jobs, prepared.device_flux,
                        prepared.device_right_components, Real(-1) / prepared.left_normal_spacing,
                        prepared.component_count, false});
    }
    for (const LocalFacePlan& plan : prepared.right_plans) {
      const std::int64_t elements =
          static_cast<std::int64_t>(plan.jobs.extent(0)) * prepared.component_count;
      Kokkos::parallel_for(
          "pops_interface_scatter_right", Kokkos::RangePolicy<>(0, elements),
          ScatterKernel{right.fab(plan.local_box).view(), plan.jobs, prepared.device_flux,
                        prepared.device_right_components, Real(1) / prepared.right_normal_spacing,
                        prepared.component_count, true});
    }
    Kokkos::fence();
  }

  static void apply_one_(PreparedInterface& prepared, const BoundaryEvaluationPoint& point,
                         field_type& left_state, field_type& right_state, field_type& left_rhs,
                         field_type& right_rhs, InterfaceFluxFragmentPublication* publication) {
    const bool layouts_match =
        runtime_field_matches_(left_state, prepared.left_layout, prepared.left_distribution,
                               prepared.left_rank, prepared.left_ghosts,
                               prepared.component_count) &&
        runtime_field_matches_(right_state, prepared.right_layout, prepared.right_distribution,
                               prepared.right_rank, prepared.right_ghosts,
                               prepared.component_count) &&
        runtime_field_matches_(left_rhs, prepared.left_layout, prepared.left_distribution,
                               prepared.left_rank, prepared.left_ghosts,
                               prepared.component_count) &&
        runtime_field_matches_(right_rhs, prepared.right_layout, prepared.right_distribution,
                               prepared.right_rank, prepared.right_ghosts,
                               prepared.component_count);
    if (prepared.distributed) {
      if (all_reduce_sum(layouts_match ? 0L : 1L, prepared.communicator) != 0)
        throw std::runtime_error(
            "multi-block interface runtime fields differ from prepared layouts on one rank");
    } else if (!layouts_match) {
      throw std::runtime_error(
          "multi-block interface runtime fields differ from their prepared layouts");
    }

    std::exception_ptr packing_failure;
    try {
      pack_(prepared, left_state, right_state);
    } catch (...) {
      packing_failure = std::current_exception();
    }
    finish_collective_preflight_(prepared.communicator, packing_failure, "device trace packing");
    const std::size_t packed =
        prepared.face_count * static_cast<std::size_t>(prepared.component_count);
    if (prepared.distributed)
      all_reduce_sum_inplace(prepared.host_traces.data(), 2 * packed, prepared.communicator);
    const bool native_memory_evaluation = prepared.memory_space != POPS_MEMORY_SPACE_HOST_V1;
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    std::fill_n(prepared.host_flux.data(), packed, nan);
    if (native_memory_evaluation) {
      // MPI always authenticates the compact batch in pinned host memory. Restore the collective
      // canonical traces to native memory before invoking a device/managed component.
      Kokkos::deep_copy(prepared.device_traces, prepared.host_traces);
      Kokkos::deep_copy(prepared.device_flux, nan);
    }
    const Real* const left_batch =
        native_memory_evaluation ? prepared.device_traces.data() : prepared.host_traces.data();
    const Real* const right_batch = native_memory_evaluation
                                        ? prepared.device_traces.data() + packed
                                        : prepared.host_traces.data() + packed;
    Real* const flux_batch =
        native_memory_evaluation ? prepared.device_flux.data() : prepared.host_flux.data();
    const Real* const normal_batch =
        native_memory_evaluation ? prepared.device_normals.data() : prepared.host_normals.data();
    const InterfaceFluxBatch batch{left_batch,
                                   right_batch,
                                   normal_batch,
                                   flux_batch,
                                   static_cast<int>(prepared.face_count),
                                   prepared.component_count,
                                   prepared.face_measure,
                                   prepared.memory_space};
    std::exception_ptr evaluator_failure;
    try {
      prepared.evaluator(point, batch);
    } catch (...) {
      evaluator_failure = std::current_exception();
    }
    if (prepared.distributed) {
      if (all_reduce_sum(evaluator_failure ? 1L : 0L, prepared.communicator) != 0)
        throw std::runtime_error("multi-block interface evaluator failed on one or more ranks");
    } else if (evaluator_failure) {
      std::rethrow_exception(evaluator_failure);
    }
    if (native_memory_evaluation)
      Kokkos::deep_copy(prepared.host_flux, prepared.device_flux);
    bool finite = true;
    for (std::size_t value = 0; value < packed; ++value)
      finite = finite && std::isfinite(static_cast<double>(prepared.host_flux(value)));
    if (prepared.distributed) {
      if (all_reduce_sum(finite ? 0L : 1L, prepared.communicator) != 0)
        throw std::runtime_error(
            "multi-block interface evaluator returned a non-finite flux on one rank");
      require_distributed_flux_consensus_(prepared, packed);
    } else if (!finite) {
      throw std::runtime_error("multi-block interface evaluator returned a non-finite flux");
    }
    if (publication != nullptr) {
      std::optional<typename InterfaceFluxFragmentLedger::PreparedAccumulation>
          prepared_publication;
      std::exception_ptr publication_failure;
      try {
        prepared_publication.emplace(prepare_fragment_publication_(prepared, point, *publication));
      } catch (...) {
        publication_failure = std::current_exception();
      }
      finish_collective_preflight_(prepared.communicator, publication_failure,
                                   "interface-fragment accumulation");
      if (!all_ranks_agree_exact_ordered_byte_pairs(
              {{std::string_view("interface-fragment-accumulation"),
                prepared_publication->exact_contract()}},
              prepared.communicator))
        throw std::runtime_error(
            "multi-block interface fragment candidates differ across MPI ranks");
      publication->ledger->publish_prepared_accumulation(*prepared_publication);
    }
    try {
      scatter_(prepared, left_rhs, right_rhs);
    } catch (...) {
      if (prepared.distributed)
        std::terminate();
      throw;
    }
    ++prepared.evaluation_count;
  }

  static void require_distributed_flux_consensus_(PreparedInterface& prepared, std::size_t packed) {
#ifdef POPS_HAS_MPI
    std::copy_n(prepared.host_flux.data(), packed, prepared.host_consensus.data());
    broadcast_bytes_inplace(reinterpret_cast<char*>(prepared.host_consensus.data()),
                            packed * sizeof(Real), 0, prepared.communicator);
    const bool equal = std::memcmp(prepared.host_consensus.data(), prepared.host_flux.data(),
                                   packed * sizeof(Real)) == 0;
    if (all_reduce_sum(equal ? 0L : 1L, prepared.communicator) != 0)
      throw std::runtime_error(
          "multi-block interface evaluator returned rank-dependent shared flux");
    std::copy_n(prepared.host_consensus.data(), packed, prepared.host_flux.data());
#else
    (void)prepared;
    (void)packed;
    throw std::logic_error(
        "distributed multi-block flux consensus is unavailable in a serial build");
#endif
  }

  static typename InterfaceFluxFragmentLedger::PreparedAccumulation prepare_fragment_publication_(
      const PreparedInterface& prepared, const BoundaryEvaluationPoint& point,
      const InterfaceFluxFragmentPublication& publication) {
    using entry_type = typename InterfaceFluxFragmentLedger::Entry;
    const ::pops::amr::InterfaceFluxFragmentMeasure measure{publication.stage_weight,
                                                            prepared.face_measure, point.dt,
                                                            publication.stage_weight_resolved};
    std::vector<entry_type> entries;
    entries.reserve(2);
    const auto accumulate = [&](int coarse_level, int fine_level,
                                ::pops::amr::InterfaceFluxOrientation orientation) {
      ::pops::amr::InterfaceFluxFragmentKey key{prepared.route.identity,
                                                publication.topology_epoch,
                                                coarse_level,
                                                fine_level,
                                                publication.clock,
                                                publication.stage_identity,
                                                point.graph_identity,
                                                point.rate_identity,
                                                point.application_identity,
                                                publication.interval,
                                                orientation,
                                                prepared.route.left_block,
                                                prepared.route.right_block};
      InterfaceFluxFragmentPayload payload(
          prepared.host_flux.data(), prepared.host_flux.data() + prepared.host_flux.extent(0));
      entries.push_back({std::move(key), measure, std::move(payload)});
    };
    if (point.level > 0)
      accumulate(point.level - 1, point.level, ::pops::amr::InterfaceFluxOrientation::FineOutward);
    if (point.level + 1 < publication.active_level_count)
      accumulate(point.level, point.level + 1,
                 ::pops::amr::InterfaceFluxOrientation::CoarseOutward);
    return publication.ledger->prepare_accumulation(std::move(entries));
  }

  void require_complete_active_level_registry_(int active_level_count) const {
    for (const PreparedInterface& prepared : interfaces_) {
      if (prepared.route.level < 0 || prepared.route.level >= active_level_count)
        throw std::runtime_error(
            "multi-block interface registry contains a route outside the active hierarchy");
      for (int level = 0; level < active_level_count; ++level) {
        const PreparedInterface* peer = nullptr;
        for (const PreparedInterface& candidate : interfaces_)
          if (candidate.route.identity == prepared.route.identity &&
              candidate.route.level == level) {
            peer = &candidate;
            break;
          }
        if (peer == nullptr || !same_route_across_levels_(prepared.route, peer->route) ||
            prepared.component_count != peer->component_count)
          throw std::runtime_error(
              "multi-block interface registry is incomplete on the active hierarchy");
      }
    }
  }

  static bool same_route_across_levels_(const route_type& left, const route_type& right) {
    route_type lhs = left;
    route_type rhs = right;
    lhs.level = 0;
    rhs.level = 0;
    return lhs.identity == rhs.identity && lhs.left_block == rhs.left_block &&
           lhs.right_block == rhs.right_block && lhs.left_axis == rhs.left_axis &&
           lhs.right_axis == rhs.right_axis && lhs.left_side == rhs.left_side &&
           lhs.right_side == rhs.right_side &&
           lhs.tangential_transform == rhs.tangential_transform &&
           lhs.right_component_for_left == rhs.right_component_for_left &&
           lhs.left_trace_projection_identity == rhs.left_trace_projection_identity &&
           lhs.right_trace_projection_identity == rhs.right_trace_projection_identity &&
           lhs.left_trace_provider_identity == rhs.left_trace_provider_identity &&
           lhs.right_trace_provider_identity == rhs.right_trace_provider_identity &&
           lhs.left_trace_operation == rhs.left_trace_operation &&
           lhs.right_trace_operation == rhs.right_trace_operation &&
           lhs.left_trace_required_depth == rhs.left_trace_required_depth &&
           lhs.right_trace_required_depth == rhs.right_trace_required_depth &&
           lhs.affine_mapping_identity == rhs.affine_mapping_identity &&
           lhs.right_normal_translation == rhs.right_normal_translation;
  }

  static void finish_collective_preflight_(const CommunicatorView& communicator,
                                           const std::exception_ptr& local_failure,
                                           const char* phase) {
    const bool collective = communicator.active() && communicator.size() > 1;
    const long failures = collective ? all_reduce_sum(local_failure ? 1L : 0L, communicator)
                                     : (local_failure ? 1L : 0L);
    if (failures == 0)
      return;
    if (local_failure)
      std::rethrow_exception(local_failure);
    throw std::runtime_error(std::string("multi-block interface ") + phase +
                             " failed on another MPI rank");
  }

  template <class Value>
  static void append_scalar_(std::string& bytes, const Value& value) {
    static_assert(std::is_trivially_copyable_v<Value>);
    bytes.append(reinterpret_cast<const char*>(&value), sizeof(Value));
  }
  static void append_text_(std::string& bytes, std::string_view value) {
    append_scalar_(bytes, static_cast<std::uint64_t>(value.size()));
    bytes.append(value.data(), value.size());
  }
  static void append_index_(std::string& bytes, const index_type& index) {
    for (int axis = 0; axis < Dim; ++axis)
      append_scalar_(bytes, index[axis]);
  }
  static void append_extent_(std::string& bytes, const ghost_type& extent) {
    for (int axis = 0; axis < Dim; ++axis)
      append_scalar_(bytes, extent[axis]);
  }
  static void append_box_(std::string& bytes, const box_type& box) {
    append_index_(bytes, box.lo);
    append_index_(bytes, box.hi);
  }
  static void append_layout_(std::string& bytes, const field_type& field) {
    append_scalar_(bytes, static_cast<std::uint64_t>(field.layout().size()));
    for (const box_type& box : field.layout().boxes())
      append_box_(bytes, box);
    append_scalar_(bytes, static_cast<int>(field.distribution().mode()));
    append_index_(bytes, field.rank_space().origin());
    append_extent_(bytes, field.rank_space().extent());
    if (!field.distribution().replicated())
      for (const rank_type& owner : field.distribution().owners())
        append_index_(bytes, owner);
    append_scalar_(bytes, field.ncomp());
    append_extent_(bytes, field.ghosts());
  }
  static void append_geometry_(std::string& bytes, const geometry_type& geometry) {
    append_box_(bytes, geometry.domain());
    for (int axis = 0; axis < Dim; ++axis) {
      append_scalar_(bytes, geometry.lower()[axis]);
      append_scalar_(bytes, geometry.upper()[axis]);
    }
  }

  static std::string collective_plan_identity_(
      const route_type& route, const field_type& left_state, const geometry_type& left_geometry,
      const field_type& right_state, const geometry_type& right_geometry, Real left_normal,
      Real right_normal, std::size_t face_count, int component_count,
      std::string_view communicator_identity, int communicator_size) {
    std::string bytes;
    append_text_(bytes, "pops.multiblock.interface-plan.nd.v1");
    append_scalar_(bytes, Dim);
    append_text_(bytes, route.identity);
    append_scalar_(bytes, static_cast<std::uint64_t>(route.left_block));
    append_scalar_(bytes, static_cast<std::uint64_t>(route.right_block));
    append_scalar_(bytes, route.level);
    append_scalar_(bytes, route.left_axis);
    append_scalar_(bytes, route.right_axis);
    append_scalar_(bytes, route.left_side);
    append_scalar_(bytes, route.right_side);
    for (int tangent = 0; tangent < tangent_dimension; ++tangent) {
      append_scalar_(bytes, route.tangential_transform.right_tangent_for_left[tangent]);
      append_scalar_(bytes, route.tangential_transform.sign[tangent]);
      append_scalar_(bytes, route.tangential_transform.offset[tangent]);
    }
    for (const int component : route.right_component_for_left)
      append_scalar_(bytes, component);
    append_text_(bytes, route.left_trace_projection_identity);
    append_text_(bytes, route.right_trace_projection_identity);
    append_text_(bytes, route.left_trace_provider_identity);
    append_text_(bytes, route.right_trace_provider_identity);
    append_scalar_(bytes, route.left_trace_operation);
    append_scalar_(bytes, route.right_trace_operation);
    append_scalar_(bytes, route.left_trace_required_depth);
    append_scalar_(bytes, route.right_trace_required_depth);
    append_text_(bytes, route.affine_mapping_identity);
    append_scalar_(bytes, route.right_normal_translation);
    append_layout_(bytes, left_state);
    append_layout_(bytes, right_state);
    append_geometry_(bytes, left_geometry);
    append_geometry_(bytes, right_geometry);
    append_scalar_(bytes, left_normal);
    append_scalar_(bytes, right_normal);
    append_scalar_(bytes, static_cast<std::uint64_t>(face_count));
    append_scalar_(bytes, component_count);
    append_text_(bytes, communicator_identity);
    append_scalar_(bytes, communicator_size);
    return bytes;
  }

  static std::string collective_point_identity_(const BoundaryEvaluationPoint& point) {
    std::string bytes;
    append_text_(bytes, "pops.multiblock.evaluation-point.v2");
    append_text_(bytes, point.clock);
    append_scalar_(bytes, point.tick);
    append_scalar_(bytes, point.level);
    append_scalar_(bytes, point.substep);
    append_scalar_(bytes, point.stage);
    append_scalar_(bytes, point.stage_fraction.numerator);
    append_scalar_(bytes, point.stage_fraction.denominator);
    append_scalar_(bytes, point.dt);
    append_scalar_(bytes, point.physical_time);
    append_text_(bytes, point.graph_identity);
    append_text_(bytes, point.rate_identity);
    append_text_(bytes, point.application_identity);
    return bytes;
  }

  static void append_clock_(std::string& bytes, const ::pops::amr::ClockStamp& clock) {
    append_scalar_(bytes, clock.level);
    append_scalar_(bytes, clock.macro_step);
    append_scalar_(bytes, clock.phase.numerator);
    append_scalar_(bytes, clock.phase.denominator);
    append_scalar_(bytes, clock.physical_time);
  }

  static std::string collective_fragment_publication_identity_(
      const InterfaceFluxFragmentPublication& publication) {
    std::string bytes;
    append_text_(bytes, "pops.multiblock.interface-fragment-publication.nd.v1");
    append_scalar_(bytes, publication.topology_epoch);
    append_scalar_(bytes, publication.active_level_count);
    append_clock_(bytes, publication.clock);
    append_text_(bytes, publication.stage_identity);
    append_clock_(bytes, publication.interval.begin);
    append_clock_(bytes, publication.interval.end);
    append_scalar_(bytes, publication.stage_weight.numerator);
    append_scalar_(bytes, publication.stage_weight.denominator);
    append_scalar_(bytes, static_cast<std::uint8_t>(publication.stage_weight_resolved));
    append_scalar_(bytes, publication.ledger->topology_epoch());
    append_scalar_(bytes, static_cast<std::uint64_t>(publication.ledger->transaction_depth()));
    append_scalar_(bytes, static_cast<std::uint64_t>(publication.ledger->pending_size()));
    append_scalar_(bytes, static_cast<std::uint64_t>(publication.ledger->published_size()));
    return bytes;
  }

  bool registry_agrees_across_ranks_(const CommunicatorView& communicator) const {
    std::vector<std::pair<std::string_view, std::string_view>> identities;
    std::exception_ptr allocation_failure;
    try {
      identities.reserve(interfaces_.size());
      for (const PreparedInterface& prepared : interfaces_)
        identities.emplace_back(prepared.route.identity, prepared.collective_identity);
    } catch (...) {
      allocation_failure = std::current_exception();
    }
    finish_collective_preflight_(communicator, allocation_failure, "registry identity allocation");
    return all_ranks_agree_exact_ordered_byte_pairs(identities, communicator);
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
          "AMR interface-flux fragment publication requires an active transaction");
    const ::pops::amr::Rational span =
        publication.interval.end.phase - publication.interval.begin.phase;
    const ::pops::amr::Rational expected_phase =
        publication.interval.begin.phase + point.stage_fraction * span;
    const double expected_time =
        publication.interval.begin.physical_time +
        point.stage_fraction.value() *
            (publication.interval.end.physical_time - publication.interval.begin.physical_time);
    if (point.graph_identity.empty() || point.rate_identity.empty() ||
        point.application_identity.empty() ||
        publication.ledger->topology_epoch() != publication.topology_epoch ||
        publication.active_level_count < 2 || point.level >= publication.active_level_count ||
        publication.clock.level != point.level || publication.interval.begin.level != point.level ||
        publication.interval.end.level != point.level ||
        publication.interval.begin.macro_step != point.tick ||
        publication.interval.end.macro_step != point.tick ||
        publication.clock.macro_step != point.tick || publication.clock.phase != expected_phase ||
        publication.clock.physical_time != point.physical_time ||
        publication.clock.physical_time != expected_time || publication.stage_identity.empty())
      throw std::invalid_argument(
          "AMR interface-flux fragment publication differs from its scheduler point");
  }

  std::vector<PreparedInterface> interfaces_;
};

}  // namespace pops::runtime::multiblock
