/// @file
/// @brief Final exact-ranked generated-package preparation for adaptive Cartesian blocks.

#pragma once

#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/numerics/spatial/embedded_boundary/cut_geometry.hpp>
#include <pops/numerics/spatial/embedded_boundary/operator.hpp>
#include <pops/runtime/amr/bootstrap_transfer_builtins.hpp>
#include <pops/numerics/spatial/operators/cartesian_operator.hpp>
#include <pops/numerics/spatial/operators/masked_operator.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr/prepared_amr_ghost_fill.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

/// Maximum qualified clock identity accepted by one prepared AMR evaluation workspace. Storage is
/// reserved during hierarchy materialization so stamping an evaluation point cannot allocate in
/// the residual/JVP hot path.
inline constexpr std::size_t kPreparedAmrClockIdentityCapacity = 4096;

/// Authenticated full-domain ghost population used only by the root level. Sparse fine levels use
/// runtime::amr::PreparedAmrGhostFill, which combines parent interpolation and same-level exchange.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
using PreparedRootAmrGhostFill = PreparedProvider<void(
    MultiFab<Dim, MemorySpace>&, const runtime::multiblock::BoundaryEvaluationPoint&)>;

/// Provider groups are independently allocated by resolved storage address, so AMR ghost filling
/// operates on the complete group carrier rather than pretending it is one component slab.
template <int Dim>
using PreparedProviderGroupsGhostFill =
    PreparedProvider<void(runtime::system::AuxiliaryStorageGroups<Dim>&,
                          const runtime::multiblock::BoundaryEvaluationPoint&)>;

/// Every value required to materialize one generated block on one live hierarchy level.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct GeneratedAmrLevelContext {
  static_assert(Dim >= 1 && Dim <= 3,
                "GeneratedAmrLevelContext only supports dimensions 1, 2, and 3");

  std::size_t level = 0;
  /// Borrowed exact execution authority for every implicit-source collective. The prepared
  /// hierarchy owns this lane for longer than every generated level block.
  const ExecutionLane* lane = nullptr;
  /// Exact block-owned state carrier for this level. The AMR runtime remains the topology and
  /// synchronization authority, but generated multi-block packages must not alias every block to
  /// the hierarchy's primary state storage.
  MultiFab<Dim, MemorySpace>* state = nullptr;
  Geometry<Dim> geometry;
  BoundaryTopology<Dim> topology;
  /// Global owner-qualified provider storage.  It is null exactly when the sealed registry
  /// contains no values; kernels never receive this field directly.
  runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage = nullptr;
  const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* provider_plan = nullptr;
  runtime::amr::PreparedAmrGhostFill<Dim, MemorySpace> state_ghost_fill;
  PreparedProviderGroupsGhostFill<Dim> provider_ghost_fill;
  PreparedRootAmrGhostFill<Dim, MemorySpace> root_state_ghost_fill;
  PreparedProviderGroupsGhostFill<Dim> root_provider_ghost_fill;
  /// Runtime-installed generic component executors.  The hierarchy owns every captured component
  /// and the execution lane for longer than this level block; generated code only invokes the
  /// already-authenticated typed closures.
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&,
                     MultiFab<Dim, MemorySpace>&, const Geometry<Dim>&, const ExecutionLane&)>
      external_ghost_boundary;
  std::function<void(
      const runtime::multiblock::BoundaryEvaluationPoint&, const MultiFab<Dim, MemorySpace>&,
      std::vector<nd::FaceField<Dim, MemorySpace>>&, const Geometry<Dim>&, const ExecutionLane&)>
      external_boundary_flux;
  std::vector<CompiledFieldBoundaryKernel<Dim>> external_field_boundaries;
  std::shared_ptr<const PreparedHyperbolicBoundary<Dim>> physical_boundary;
  std::shared_ptr<const runtime::system::PreparedEmbeddedBoundaryGeometry<Dim>> embedded_boundary;
  std::string state_identity;
  std::string provider_storage_identity;
  std::string boundary_identity;
  std::string embedded_boundary_provider_identity;
  std::size_t clock_identity_capacity = kPreparedAmrClockIdentityCapacity;
};

/// A transactionally materialized AMR residual and its integrated face fluxes.
///
/// The face fields are retained rather than reconstructed from the cell residual.  The Program
/// reflux primitive can therefore authenticate and accumulate exactly the fluxes that produced
/// this candidate update.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct PreparedAmrLevelEvaluation {
  runtime::multiblock::BoundaryEvaluationPoint point;
  std::string spatial_contract;
  std::uint64_t topology_epoch = 0;
  std::uint64_t materialization_generation = 0;
  MultiFab<Dim, MemorySpace> residual;
  std::vector<nd::FaceField<Dim, MemorySpace>> integrated_face_fluxes;
};

namespace generated_amr_detail {

template <class Model>
concept ExactGeneratedModelContract = requires(const Model& model, ExactContractBuilder& contract) {
  { Model::provider_identity() } -> std::same_as<PreparedProviderIdentity>;
  { model.serialize_exact_parameters(contract) } -> std::same_as<void>;
};

template <class Model>
std::string exact_model_contract(const Model& model) {
  static_assert(
      ExactGeneratedModelContract<Model>,
      "generated AMR models must publish provider_identity() and serialize_exact_parameters() so "
      "MPI ranks authenticate physical parameters rather than only their numerical routes");
  const PreparedProviderIdentity identity = Model::provider_identity();
  if (identity.name.empty() || identity.version == 0)
    throw std::invalid_argument("generated AMR model identity is incomplete");
  ExactContractBuilder parameters;
  model.serialize_exact_parameters(parameters);
  ExactContractBuilder contract;
  contract.text("pops.generated-amr-model")
      .scalar(std::uint32_t{1})
      .text(identity.name)
      .scalar(identity.version)
      .bytes(std::move(parameters).release());
  return std::move(contract).release();
}

template <int Dim>
bool has_physical_faces(const BoundaryTopology<Dim>& topology) {
  for (int axis = 0; axis < Dim; ++axis) {
    if (topology.is_physical(Face<Dim>{axis, BoundarySide::lower}) ||
        topology.is_physical(Face<Dim>{axis, BoundarySide::upper}))
      return true;
  }
  return false;
}

template <int Dim>
void append_geometry_contract(ExactContractBuilder& contract, const Geometry<Dim>& geometry,
                              const BoundaryTopology<Dim>& topology) {
  for (int axis = 0; axis < Dim; ++axis) {
    contract.scalar(std::int64_t{geometry.domain().lo[axis]})
        .scalar(std::int64_t{geometry.domain().hi[axis]})
        .scalar(static_cast<double>(geometry.lower()[axis]))
        .scalar(static_cast<double>(geometry.upper()[axis]))
        .scalar(topology.kind(Face<Dim>{axis, BoundarySide::lower}))
        .scalar(topology.kind(Face<Dim>{axis, BoundarySide::upper}));
  }
}

inline void append_variable_set_contract(ExactContractBuilder& contract,
                                         const VariableSet& variables) {
  contract.scalar(static_cast<std::int32_t>(variables.kind))
      .scalar(std::int32_t{variables.size})
      .sequence(variables.names,
                [](ExactContractBuilder& item, const std::string& name) { item.text(name); })
      .sequence(variables.roles,
                [](ExactContractBuilder& item, VariableSemantic role) {
                  item.scalar(static_cast<std::int32_t>(role.kind))
                      .presence(role.has_axis())
                      .scalar(std::int32_t{role.axis});
                })
      .sequence(variables.user_roles,
                [](ExactContractBuilder& item, const std::string& role) { item.text(role); });
}

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class RuntimeTopologyView {
 public:
  using runtime_type = runtime::amr::AmrRuntime<Dim, MemorySpace>;
  using hierarchy_type = typename runtime_type::hierarchy_type;
  using field_type = MultiFab<Dim, MemorySpace>;

  virtual ~RuntimeTopologyView() = default;
  [[nodiscard]] virtual const hierarchy_type& hierarchy() const = 0;
  [[nodiscard]] virtual std::string_view spatial_contract() const = 0;
  [[nodiscard]] virtual std::uint64_t topology_epoch() const = 0;
  [[nodiscard]] virtual std::uint64_t materialization_generation() const = 0;
  [[nodiscard]] virtual const ExecutionLane& lane() const = 0;
  [[nodiscard]] virtual std::size_t block_count() const = 0;
  [[nodiscard]] virtual const field_type& state(std::size_t block, std::size_t level) const = 0;
  /// Writable access is restricted to cold graph construction, where ghost-fill preparation binds
  /// the exact target Fab.  Dependency routes use state() and never gain a candidate write path.
  [[nodiscard]] virtual field_type& mutable_state(std::size_t block, std::size_t level) const = 0;
  [[nodiscard]] virtual std::string_view collective_contract() const = 0;
};

template <int Dim, class MemorySpace>
void require_level_context(const RuntimeTopologyView<Dim, MemorySpace>& runtime,
                           const GeneratedAmrLevelContext<Dim, MemorySpace>& context,
                           int state_components, int provider_components,
                           const Extent<Dim>& required_ghosts,
                           std::string_view staircase_provider_identity,
                           std::string_view cut_cell_provider_identity) {
  if (context.level >= runtime.hierarchy().num_levels())
    throw std::out_of_range("generated AMR block level lies outside the live hierarchy");
  if (context.state_identity.empty() || context.provider_storage_identity.empty())
    throw std::invalid_argument("generated AMR block requires exact state and provider identities");
  if (context.lane == nullptr || !context.lane->active())
    throw std::invalid_argument("generated AMR block requires one active prepared execution lane");
  if (context.geometry.domain() != runtime.hierarchy().layout(context.level).domain())
    throw std::invalid_argument(
        "generated AMR level geometry differs from the live hierarchy domain");
  if (provider_components > 0) {
    if (context.provider_storage == nullptr || context.provider_plan == nullptr ||
        context.provider_plan->value_count() != static_cast<std::size_t>(provider_components))
      throw std::invalid_argument("generated AMR block requires resolved provider storage");
  } else if (context.provider_storage != nullptr || context.provider_plan != nullptr) {
    throw std::invalid_argument("provider-free generated AMR block cannot retain provider state");
  }
  if (context.level == 0) {
    if (!context.root_state_ghost_fill || context.state_ghost_fill)
      throw std::invalid_argument(
          "generated AMR root level requires one prepared state ghost provider");
    if (provider_components > 0) {
      if (!context.root_provider_ghost_fill || context.provider_ghost_fill)
        throw std::invalid_argument(
            "generated AMR root providers require one prepared full-domain ghost provider");
    } else if (context.root_provider_ghost_fill || context.provider_ghost_fill) {
      throw std::invalid_argument(
          "provider-free generated AMR block cannot retain provider ghosts");
    }
  } else if (!context.state_ghost_fill || context.root_state_ghost_fill) {
    throw std::invalid_argument(
        "generated AMR fine level requires one prepared sparse state ghost provider");
  } else if (provider_components > 0) {
    if (!context.provider_ghost_fill || context.root_provider_ghost_fill)
      throw std::invalid_argument(
          "generated AMR fine providers require one prepared sparse coarse/fine ghost provider");
  } else if (context.provider_ghost_fill || context.root_provider_ghost_fill) {
    throw std::invalid_argument("provider-free generated AMR block cannot retain provider ghosts");
  }
  if (has_physical_faces(context.topology) &&
      (!context.physical_boundary || context.boundary_identity.empty()))
    throw std::invalid_argument(
        "generated AMR block requires an authenticated physical-boundary provider");

  if (context.embedded_boundary) {
    const auto mode = context.embedded_boundary->mode();
    const std::string_view expected_provider =
        mode == runtime::system::PreparedEmbeddedBoundaryMode::staircase
            ? staircase_provider_identity
        : mode == runtime::system::PreparedEmbeddedBoundaryMode::cut_cell
            ? cut_cell_provider_identity
            : std::string_view{};
    if (mode != runtime::system::PreparedEmbeddedBoundaryMode::inactive &&
        (expected_provider.empty() ||
         context.embedded_boundary_provider_identity != expected_provider))
      throw std::invalid_argument(
          "generated AMR embedded boundary lacks its authenticated numerical provider");
    if (mode == runtime::system::PreparedEmbeddedBoundaryMode::inactive &&
        !context.embedded_boundary_provider_identity.empty())
      throw std::invalid_argument(
          "inactive generated AMR embedded geometry cannot select a transport provider");
    if (mode != runtime::system::PreparedEmbeddedBoundaryMode::inactive &&
        context.physical_boundary)
      throw std::invalid_argument(
          "generated AMR embedded transport requires an EB-qualified physical-boundary provider");
    const auto& embedded = *context.embedded_boundary;
    if (embedded.geometry().domain() != context.geometry.domain() ||
        embedded.topology() != context.topology)
      throw std::invalid_argument(
          "generated AMR embedded geometry differs from its exact level topology");
  } else if (!context.embedded_boundary_provider_identity.empty()) {
    throw std::invalid_argument("generated AMR embedded provider has no prepared level geometry");
  }

  if (context.state == nullptr)
    throw std::invalid_argument("generated AMR block requires an exact bound state carrier");
  const auto& state = *context.state;
  const auto& topology_state = runtime.hierarchy().state(context.level);
  if (state.layout() != topology_state.layout() ||
      state.distribution() != topology_state.distribution() ||
      state.local_rank() != topology_state.local_rank() ||
      state.local_size() != topology_state.local_size())
    throw std::invalid_argument(
        "generated AMR block state carrier differs from the canonical hierarchy topology");
  if (state.ncomp() != state_components)
    throw std::invalid_argument("generated AMR state component count differs from its model");
  if (provider_components > 0) {
    for (const auto& value : context.provider_plan->values) {
      const auto* const group = context.provider_storage->find(value.address.group);
      if (group == nullptr || value.address.component >= static_cast<std::size_t>(group->ncomp()) ||
          group->layout() != state.layout() || group->distribution() != state.distribution() ||
          group->local_rank() != state.local_rank() || group->local_size() != state.local_size() ||
          value.contract.centering != "cell" || value.shape.value_components != 1)
        throw std::invalid_argument(
            "generated AMR provider group is not pointwise compatible with state");
    }
  }
  for (int axis = 0; axis < Dim; ++axis) {
    if (state.ghosts()[axis] < required_ghosts[axis])
      throw std::invalid_argument(
          "generated AMR state storage is narrower than its reconstruction stencil");
  }
  if (context.embedded_boundary) {
    const auto require_embedded_field = [&](const MultiFab<Dim, MemorySpace>& field, int components,
                                            const char* label) {
      if (field.ncomp() != components || field.layout() != state.layout() ||
          field.distribution() != state.distribution() ||
          field.local_rank() != state.local_rank() || field.local_size() != state.local_size())
        throw std::invalid_argument(std::string("generated AMR embedded ") + label +
                                    " differs from its exact level ownership");
    };
    require_embedded_field(context.embedded_boundary->phi(), 1, "level set");
    require_embedded_field(context.embedded_boundary->active_mask(), 1, "active mask");
    require_embedded_field(context.embedded_boundary->volume_fraction(), 1, "volume fraction");
    require_embedded_field(context.embedded_boundary->inverse_volume_fraction(), 1,
                           "inverse volume fraction");
    require_embedded_field(context.embedded_boundary->face_aperture_lower(), Dim,
                           "face aperture lower");
    require_embedded_field(context.embedded_boundary->face_aperture_upper(), Dim,
                           "face aperture upper");
  }
}

template <int Dim, class MemorySpace>
std::string level_contract(const RuntimeTopologyView<Dim, MemorySpace>& runtime,
                           const GeneratedAmrLevelContext<Dim, MemorySpace>& context,
                           std::string_view provider_identity,
                           std::optional<Real> parabolic_frequency) {
  ExactContractBuilder contract;
  contract.text("pops.generated-amr-level-block")
      .scalar(std::uint32_t{2})
      .scalar(std::int32_t{Dim})
      .scalar(static_cast<std::uint64_t>(context.level))
      .text(provider_identity)
      .text(context.state_identity)
      .text(context.provider_storage_identity)
      .text(context.boundary_identity)
      .text(context.embedded_boundary_provider_identity)
      .bytes(runtime.spatial_contract())
      .scalar(runtime.topology_epoch())
      .scalar(runtime.materialization_generation())
      .optional_collective_contract(context.state_ghost_fill)
      .optional_collective_contract(context.provider_ghost_fill)
      .optional_collective_contract(context.root_state_ghost_fill)
      .optional_collective_contract(context.root_provider_ghost_fill)
      .presence(parabolic_frequency.has_value())
      .scalar(parabolic_frequency.value_or(Real(0)));
  contract.presence(static_cast<bool>(context.embedded_boundary));
  if (context.embedded_boundary)
    contract
        .text(runtime::system::prepared_embedded_boundary_mode_name(
            context.embedded_boundary->mode()))
        .text(context.embedded_boundary->semantic_digest())
        .text(context.embedded_boundary->digest())
        .scalar(context.embedded_boundary->generation());
  append_geometry_contract(contract, context.geometry, context.topology);
  return std::move(contract).release();
}

template <class Operation>
void collective_phase(const ExecutionLane& lane, Operation&& operation,
                      const char* failure_message) {
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    std::forward<Operation>(operation)();
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_failure, lane) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(failure_message);
  }
}

template <int Dim, class Model>
MultiFab<Dim> materialize_source(
    const Model& model, const MultiFab<Dim>& state,
    const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
    const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* provider_plan) {
  return generated_system_detail::materialize_source<Dim>(model, state, provider_storage,
                                                          provider_plan);
}

template <int Dim, class Model>
MultiFab<Dim> materialize_masked_source(
    const Model& model, const MultiFab<Dim>& state,
    const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
    const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* provider_plan,
    const runtime::system::PreparedEmbeddedBoundaryGeometry<Dim>& embedded) {
  return generated_system_detail::materialize_masked_source<Dim>(model, state, provider_storage,
                                                                 provider_plan, embedded);
}

template <int Dim, class Model>
Real maximum_speed(const Model& model, const MultiFab<Dim>& state,
                   const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
                   const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* provider_plan,
                   const ExecutionLane& lane) {
  return generated_system_detail::maximum_speed<Dim>(model, state, provider_storage, provider_plan,
                                                     lane);
}

template <int Dim, class Model>
void add_poisson_rhs_locally(const Model& model, const MultiFab<Dim>& state, MultiFab<Dim>& rhs) {
  if (rhs.ncomp() != 1)
    throw std::invalid_argument("generated AMR Poisson RHS destination must have one component");
  generated_system_detail::require_same_layout(state, rhs, 1, "generated AMR Poisson RHS");
  MultiFab<Dim> candidate(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
  MultiFab<Dim> status(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
  MultiFab<Dim> updated(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
  candidate.set_val(Real(0));
  status.set_val(Real(0));
  for (std::size_t local = 0; local < state.local_size(); ++local)
    for_each_cell(state.box(local), generated_system_detail::MaterializePoissonRhs<Dim, Model>{
                                        model, state.fab(local).view(), candidate.fab(local).view(),
                                        status.fab(local).view()});
  device_fence();
  if (state.local_size() != 0 && reduce_max_local(status) != Real(0))
    throw std::runtime_error("generated AMR Poisson RHS produced a non-finite value");
  generated_system_detail::copy_valid(rhs, updated);
  saxpy(updated, Real(1), candidate);
  device_fence();
  static_assert(std::is_nothrow_move_assignable_v<MultiFab<Dim>>);
  rhs = std::move(updated);
}

template <int Dim, class Model>
Real source_frequency(const Model& model, const MultiFab<Dim>& state,
                      const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
                      const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* provider_plan,
                      const ExecutionLane& lane) {
  constexpr int provider_count = provider_count_for<Model, Dim>();
  std::unique_ptr<MultiFab<Dim>> values;
  collective_phase(
      lane,
      [&] {
        values = std::make_unique<MultiFab<Dim>>(state.layout(), state.distribution(),
                                                 state.local_rank(), 1, state.ghosts());
        for (std::size_t local = 0; local < state.local_size(); ++local)
          for_each_cell(state.box(local),
                        generated_system_detail::MaterializeSourceFrequency<Dim, Model>{
                            model, state.fab(local).view(),
                            runtime::system::bind_provider_storage_view<Dim, provider_count>(
                                provider_plan, provider_storage, local),
                            values->fab(local).view()});
        device_fence();
      },
      "generated AMR source-frequency preparation failed collectively");
  Real local_frequency = Real(0);
  collective_phase(
      lane,
      [&] {
        device_fence();
        if (state.local_size() != 0)
          local_frequency = reduce_max_local(*values);
      },
      "generated AMR source-frequency local reduction failed collectively");
  const Real frequency = static_cast<Real>(all_reduce_max(local_frequency, lane));
  if (!std::isfinite(frequency) || frequency < Real(0))
    throw std::runtime_error("generated AMR source frequency is invalid");
  return frequency;
}

template <int Dim, class Model>
Real stability_dt(const Model& model, const MultiFab<Dim>& state,
                  const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
                  const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* provider_plan,
                  const ExecutionLane& lane) {
  constexpr int provider_count = provider_count_for<Model, Dim>();
  std::unique_ptr<MultiFab<Dim>> values;
  collective_phase(
      lane,
      [&] {
        values = std::make_unique<MultiFab<Dim>>(state.layout(), state.distribution(),
                                                 state.local_rank(), 1, state.ghosts());
        for (std::size_t local = 0; local < state.local_size(); ++local)
          for_each_cell(state.box(local),
                        generated_system_detail::MaterializeStabilityDt<Dim, Model>{
                            model, state.fab(local).view(),
                            runtime::system::bind_provider_storage_view<Dim, provider_count>(
                                provider_plan, provider_storage, local),
                            values->fab(local).view()});
        device_fence();
      },
      "generated AMR stability-dt preparation failed collectively");
  Real local_dt = std::numeric_limits<Real>::infinity();
  collective_phase(
      lane,
      [&] {
        device_fence();
        if (state.local_size() != 0)
          local_dt = reduce_min_local(*values);
      },
      "generated AMR stability-dt local reduction failed collectively");
  const Real dt = static_cast<Real>(all_reduce_min(local_dt, lane));
  if (!std::isfinite(dt) || !(dt > Real(0)))
    throw std::runtime_error("generated AMR stability dt is invalid");
  return dt;
}

template <int Dim>
Real all_component_norm_inf(const MultiFab<Dim>& field) {
  Real result = Real(0);
  for (int component = 0; component < field.ncomp(); ++component)
    result = std::max(result, norm_inf(field, component));
  return result;
}

}  // namespace generated_amr_detail

/// One exact generated spatial specialization bound to an authenticated AMR topology view.  The
/// view is either accepted or forward-staged; generated blocks must not retain a raw runtime that
/// would silently switch a candidate graph back to the accepted hierarchy.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedGeneratedAmrLevelBlock {
 public:
  using runtime_type = runtime::amr::AmrRuntime<Dim, MemorySpace>;
  using topology_view_type = generated_amr_detail::RuntimeTopologyView<Dim, MemorySpace>;
  using field_type = MultiFab<Dim, MemorySpace>;
  using evaluation_type = PreparedAmrLevelEvaluation<Dim, MemorySpace>;
  using point_type = runtime::multiblock::BoundaryEvaluationPoint;
  using StatePreparation = std::function<void(const point_type&, field_type&)>;
  using PhysicalBoundaryPreparation = std::function<void(const point_type&, field_type&)>;
  using Evaluator = std::function<void(const point_type&, field_type&, evaluation_type&)>;
  using BoundaryJvp =
      std::function<void(const point_type&, field_type&, const field_type&, field_type&)>;
  using SourceEvaluator = std::function<void(const point_type&, field_type&, field_type&)>;
  using ImplicitSourceSolver =
      std::function<SolveOutcome(const point_type&, field_type&, Real, const NewtonOptions&)>;
  using implicit_workspace_type = PreparedImplicitSourceWorkspace<Dim, MemorySpace>;
  using PointwiseProjection = std::function<void(field_type&)>;
  using Speed = std::function<Real(const field_type&)>;
  using PoissonRhs = std::function<void(const field_type&, field_type&)>;

  PreparedGeneratedAmrLevelBlock(
      const topology_view_type& runtime, std::size_t level, field_type& state,
      std::string state_identity, std::string provider_identity, std::string collective_contract,
      const ExecutionLane& lane, StatePreparation prepare_state,
      PhysicalBoundaryPreparation prepare_physical, Evaluator evaluator, Evaluator flux_evaluator,
      Evaluator core_evaluator, Evaluator flux_core_evaluator, Evaluator boundary_evaluator,
      BoundaryJvp boundary_jvp, SourceEvaluator source_evaluator,
      std::shared_ptr<implicit_workspace_type> implicit_workspace,
      std::uint64_t implicit_workspace_generation, ImplicitSourceSolver implicit_source_solver,
      Speed maximum_speed, PoissonRhs poisson_rhs, PointwiseProjection pointwise_projection,
      Speed source_frequency, std::optional<Real> parabolic_frequency, Speed stability_dt)
      // RuntimeTopologyView is a cold-build adapter and may itself live only for the duration of
      // graph construction.  Retain the immutable generation witness by value; the owning
      // PreparedHierarchy publishes or discards this block together with the fields it addresses.
      : level_(level),
        prepared_level_count_(runtime.hierarchy().num_levels()),
        spatial_contract_(runtime.spatial_contract()),
        state_(&state),
        state_identity_(std::move(state_identity)),
        provider_identity_(std::move(provider_identity)),
        collective_contract_(std::move(collective_contract)),
        lane_(&lane),
        prepare_state_(std::move(prepare_state)),
        prepare_physical_(std::move(prepare_physical)),
        evaluator_(std::move(evaluator)),
        flux_evaluator_(std::move(flux_evaluator)),
        core_evaluator_(std::move(core_evaluator)),
        flux_core_evaluator_(std::move(flux_core_evaluator)),
        boundary_evaluator_(std::move(boundary_evaluator)),
        boundary_jvp_(std::move(boundary_jvp)),
        source_evaluator_(std::move(source_evaluator)),
        implicit_workspace_(std::move(implicit_workspace)),
        implicit_workspace_generation_(implicit_workspace_generation),
        implicit_source_solver_(std::move(implicit_source_solver)),
        maximum_speed_(std::move(maximum_speed)),
        poisson_rhs_(std::move(poisson_rhs)),
        pointwise_projection_(std::move(pointwise_projection)),
        source_frequency_(std::move(source_frequency)),
        parabolic_frequency_(std::move(parabolic_frequency)),
        stability_dt_(std::move(stability_dt)),
        topology_epoch_(runtime.topology_epoch()),
        materialization_generation_(runtime.materialization_generation()) {
    if (level_ >= prepared_level_count_ || state_ == nullptr || lane_ == nullptr ||
        state_identity_.empty() || provider_identity_.empty() || collective_contract_.empty() ||
        !prepare_state_ || !prepare_physical_ || !evaluator_ || !flux_evaluator_ ||
        !core_evaluator_ || !flux_core_evaluator_ || !boundary_evaluator_ || !boundary_jvp_ ||
        !source_evaluator_ || !implicit_workspace_ || !implicit_workspace_->prepared() ||
        implicit_workspace_->state_generation() != implicit_workspace_generation_ ||
        implicit_workspace_generation_ != materialization_generation_ || !implicit_source_solver_ ||
        !maximum_speed_ || !poisson_rhs_)
      throw std::invalid_argument("generated AMR level block preparation is incomplete");
  }

  static constexpr int dimension = Dim;
  std::size_t level() const noexcept { return level_; }
  std::string_view state_identity() const noexcept { return state_identity_; }
  std::string_view provider_identity() const noexcept { return provider_identity_; }
  std::string_view collective_contract() const noexcept { return collective_contract_; }

  void prepare(const point_type& point, field_type& state) const {
    generated_amr_detail::collective_phase(
        *lane_,
        [&] {
          require_live_();
          require_state_(point, state);
          require_bound_state_(state);
        },
        "generated AMR state-preparation preflight failed collectively");
    prepare_state_(point, state);
  }

  /// Apply the already-prepared source block's model-qualified physical and external Ghost
  /// authorities to a caller-owned detached image.  Same-level/coarse-fine transport is owned by
  /// the routed dependency graph and is deliberately not repeated here.
  void prepare_physical(const point_type& point, field_type& state) const {
    require_live_();
    require_state_(point, state);
    prepare_physical_(point, state);
  }

  void evaluate(const point_type& point, field_type& state, evaluation_type& evaluation) const {
    require_live_();
    require_state_(point, state);
    require_bound_state_(state);
    require_evaluation_contract_(evaluation);
    require_prepared_point_(point, evaluation.point);
    evaluator_(point, state, evaluation);
    stamp_point_(point, evaluation.point);
  }

  void evaluate(const point_type& point, evaluation_type& evaluation) const {
    evaluate(point, *state_, evaluation);
  }

  void evaluate_flux(const point_type& point, field_type& state,
                     evaluation_type& evaluation) const {
    require_live_();
    require_state_(point, state);
    require_bound_state_(state);
    require_evaluation_contract_(evaluation);
    require_prepared_point_(point, evaluation.point);
    flux_evaluator_(point, state, evaluation);
    stamp_point_(point, evaluation.point);
  }

  void evaluate_flux(const point_type& point, evaluation_type& evaluation) const {
    evaluate_flux(point, *state_, evaluation);
  }

  void evaluate_core(const point_type& point, field_type& state, bool flux_only,
                     evaluation_type& evaluation) const {
    require_live_();
    require_state_(point, state);
    require_bound_state_(state);
    require_evaluation_contract_(evaluation);
    require_prepared_point_(point, evaluation.point);
    (flux_only ? flux_core_evaluator_ : core_evaluator_)(point, state, evaluation);
    stamp_point_(point, evaluation.point);
  }

  void evaluate_boundary(const point_type& point, field_type& state,
                         evaluation_type& evaluation) const {
    require_live_();
    require_state_(point, state);
    require_bound_state_(state);
    require_evaluation_contract_(evaluation);
    require_prepared_point_(point, evaluation.point);
    boundary_evaluator_(point, state, evaluation);
    stamp_point_(point, evaluation.point);
  }

  void boundary_jvp(const point_type& point, field_type& state, const field_type& direction,
                    field_type& result) const {
    require_live_();
    require_state_(point, state);
    require_state_contract_(direction);
    require_state_contract_(result);
    boundary_jvp_(point, state, direction, result);
  }

  void source_into(const point_type& point, field_type& state, field_type& result) const {
    require_live_();
    require_state_(point, state);
    require_bound_state_(state);
    require_state_contract_(result);
    source_evaluator_(point, state, result);
  }

  void source_into(const point_type& point, field_type& result) const {
    source_into(point, *state_, result);
  }

  /// Solve directly on the caller's detached Program candidate. The facade authenticates the
  /// block/level route before entry; retaining the candidate as the inner SolveOutcome target lets
  /// that lane-qualified outcome remain the sole publication transaction.
  [[nodiscard]] SolveOutcome solve_implicit_source(const point_type& point, field_type& state,
                                                   Real dt, const NewtonOptions& options) const {
    require_live_();
    require_state_(point, state);
    if (!implicit_workspace_ || !implicit_workspace_->prepared() ||
        implicit_workspace_->state_generation() != implicit_workspace_generation_)
      return SolveOutcome::collective_lane(SolveReport::capability_failure(), *lane_);
    return implicit_source_solver_(point, state, dt, options);
  }

  [[nodiscard]] SolveOutcome solve_implicit_source(const point_type& point, Real dt,
                                                   const NewtonOptions& options) const {
    return solve_implicit_source(point, *state_, dt, options);
  }

  /// Mutate only an exact detached Program candidate.  The closure is omitted when the generated
  /// physical model has no pointwise projection capability.
  void project(field_type& detached_candidate) const {
    require_live_();
    require_state_contract_(detached_candidate);
    if (&detached_candidate == state_)
      throw std::invalid_argument(
          "generated AMR pointwise projection refuses the accepted live state carrier");
    if (!pointwise_projection_)
      throw std::runtime_error("generated AMR block has no pointwise projection provider");
    pointwise_projection_(detached_candidate);
  }

  [[nodiscard]] bool has_pointwise_projection() const noexcept {
    return static_cast<bool>(pointwise_projection_);
  }

  Real maximum_speed(const field_type& state) const {
    generated_amr_detail::collective_phase(
        *lane_,
        [&] {
          require_live_();
          require_state_contract_(state);
        },
        "generated AMR maximum-speed preflight failed collectively");
    return maximum_speed_(state);
  }

  Real maximum_speed() const { return maximum_speed(*state_); }

  void add_poisson_rhs(field_type& rhs) const { add_poisson_rhs(*state_, rhs); }

  void add_poisson_rhs(const field_type& state, field_type& rhs) const {
    std::optional<field_type> candidate;
    generated_amr_detail::collective_phase(
        *lane_,
        [&] {
          require_live_();
          require_state_contract_(state);
          candidate.emplace(rhs.layout(), rhs.distribution(), rhs.local_rank(), rhs.ncomp(),
                            rhs.ghosts());
          generated_system_detail::copy_valid(rhs, *candidate);
          poisson_rhs_(state, *candidate);
        },
        "generated AMR Poisson-RHS preparation failed collectively");
    static_assert(std::is_nothrow_move_assignable_v<field_type>);
    rhs = std::move(*candidate);
  }

  /// Rank-local contribution seam used only by the host's grouped all-block transaction. The host
  /// authenticates the complete schedule and wraps every call in one lane failure convergence.
  void add_poisson_rhs_locally(const field_type& state, field_type& rhs) const {
    require_live_();
    require_state_contract_(state);
    poisson_rhs_(state, rhs);
  }

  std::optional<Real> source_frequency() const {
    generated_amr_detail::collective_phase(
        *lane_, [&] { require_live_(); },
        "generated AMR source-frequency preflight failed collectively");
    const long available = source_frequency_ ? 1L : 0L;
    if (all_reduce_min(available, *lane_) != all_reduce_max(available, *lane_))
      throw std::runtime_error("generated AMR source-frequency availability differs between ranks");
    if (!source_frequency_)
      return std::nullopt;
    return source_frequency_(*state_);
  }

  std::optional<Real> stability_dt() const {
    generated_amr_detail::collective_phase(
        *lane_, [&] { require_live_(); },
        "generated AMR stability-dt preflight failed collectively");
    const long available = stability_dt_ ? 1L : 0L;
    if (all_reduce_min(available, *lane_) != all_reduce_max(available, *lane_))
      throw std::runtime_error("generated AMR stability-dt availability differs between ranks");
    if (!stability_dt_)
      return std::nullopt;
    return stability_dt_(*state_);
  }

  std::optional<Real> parabolic_frequency() const {
    require_live_();
    return parabolic_frequency_;
  }

 private:
  static void require_prepared_point_(const point_type& point, const point_type& prepared) {
    if (prepared.clock.capacity() < point.clock.size())
      throw std::length_error(
          "generated AMR evaluation point exceeds its prepared clock-identity capacity");
  }

  static void stamp_point_(const point_type& source, point_type& destination) noexcept {
    destination.clock.assign(source.clock.data(), source.clock.size());
    destination.tick = source.tick;
    destination.level = source.level;
    destination.substep = source.substep;
    destination.stage = source.stage;
    destination.stage_fraction = source.stage_fraction;
    destination.dt = source.dt;
    destination.physical_time = source.physical_time;
  }

  void require_state_(const point_type& point, const field_type& state) const {
    if (point.level != static_cast<int>(level_))
      throw std::invalid_argument("generated AMR residual point targets another hierarchy level");
    require_state_contract_(state);
  }

  void require_state_contract_(const field_type& state) const {
    const field_type& live = *state_;
    if (state.layout() != live.layout() || state.distribution() != live.distribution() ||
        state.local_rank() != live.local_rank() || state.local_size() != live.local_size() ||
        state.ncomp() != live.ncomp() || state.ghosts() != live.ghosts())
      throw std::invalid_argument(
          "generated AMR stage state differs from its exact live level contract");
  }

  void require_bound_state_(const field_type& state) const {
    if (&state != state_)
      throw std::invalid_argument(
          "generated AMR ghost providers require their exact bound live level; stage candidates "
          "must enter through AmrSystem's transactional evaluation route");
  }

  void require_evaluation_contract_(const evaluation_type& evaluation) const {
    require_state_contract_(evaluation.residual);
    if (evaluation.spatial_contract != spatial_contract_ ||
        evaluation.topology_epoch != topology_epoch_ ||
        evaluation.materialization_generation != materialization_generation_ ||
        evaluation.integrated_face_fluxes.size() != state_->local_size())
      throw std::invalid_argument(
          "generated AMR evaluation output differs from its prepared level generation");
    for (std::size_t local = 0; local < state_->local_size(); ++local)
      if (evaluation.integrated_face_fluxes[local].cell_box() != state_->box(local) ||
          evaluation.integrated_face_fluxes[local].ncomp() != state_->ncomp())
        throw std::invalid_argument(
            "generated AMR evaluation face output differs from its prepared patch layout");
  }

  void require_live_() const {
    if (state_ == nullptr || level_ >= prepared_level_count_)
      throw std::invalid_argument(
          "generated AMR level block is stale after a hierarchy topology mutation");
  }

  std::size_t level_ = 0;
  std::size_t prepared_level_count_ = 0;
  std::string spatial_contract_;
  field_type* state_ = nullptr;
  std::string state_identity_;
  std::string provider_identity_;
  std::string collective_contract_;
  const ExecutionLane* lane_ = nullptr;
  StatePreparation prepare_state_;
  PhysicalBoundaryPreparation prepare_physical_;
  Evaluator evaluator_;
  Evaluator flux_evaluator_;
  Evaluator core_evaluator_;
  Evaluator flux_core_evaluator_;
  Evaluator boundary_evaluator_;
  BoundaryJvp boundary_jvp_;
  SourceEvaluator source_evaluator_;
  /// One cold-primed source workspace for this exact (block, level, implicit-route) image.  Its
  /// shape prototype is the materialized level state; Program candidates are checked against that
  /// immutable contract before the workspace is reserved.
  std::shared_ptr<implicit_workspace_type> implicit_workspace_;
  std::uint64_t implicit_workspace_generation_ = 0;
  ImplicitSourceSolver implicit_source_solver_;
  Speed maximum_speed_;
  PoissonRhs poisson_rhs_;
  PointwiseProjection pointwise_projection_;
  Speed source_frequency_;
  std::optional<Real> parabolic_frequency_;
  Speed stability_dt_;
  std::uint64_t topology_epoch_ = 0;
  std::uint64_t materialization_generation_ = 0;
};

/// Complete package-owned image prepared before the AMR facade is mutated.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct PreparedAmrSystemBlock {
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedAmrSystemBlock only supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;

  using runtime_type = runtime::amr::AmrRuntime<Dim, MemorySpace>;
  using topology_view_type = generated_amr_detail::RuntimeTopologyView<Dim, MemorySpace>;
  using context_type = GeneratedAmrLevelContext<Dim, MemorySpace>;
  using level_block_type = PreparedGeneratedAmrLevelBlock<Dim, MemorySpace>;
  using LevelMaterializer =
      std::function<level_block_type(const topology_view_type&, context_type)>;

  std::string name;
  std::string provider_identity;
  std::string provider_consumer_qid;
  std::string staircase_provider_identity;
  std::string cut_cell_provider_identity;
  std::string collective_contract;
  int ncomp = 0;
  int provider_components = 0;
  bool has_pointwise_projection = false;
  VariableSet conservative_variables{};
  VariableSet primitive_variables{};
  double gamma = 1.0;
  Extent<Dim> ghosts{};
  int reconstruction_order = 1;
  int substeps = 1;
  int stride = 1;
  NewtonOptions newton{};
  bool newton_diagnostics = false;
  std::string time_route;
  LevelMaterializer materialize_level;
  std::function<void(const double*, double*)> primitive_to_conservative;
  std::function<RecoveryReport(const double*, double*)> conservative_to_primitive;
  UniformCellRecovery batch_conservative_to_primitive;

  level_block_type prepare_level(const topology_view_type& runtime, context_type context) const {
    if (!materialize_level)
      throw std::logic_error("prepared AMR system block has no level materializer");
    return materialize_level(runtime, std::move(context));
  }
};

/// One generated elliptic attachment retained by the same DSO as its prepared spatial block.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct PreparedNativeAmrEllipticAttachment {
  std::string field;
  std::string block_identity;
  std::string binding_identity;
  std::string rhs_provider_identity;
  std::string rhs_provider_key;
  std::size_t binding_ordinal = std::numeric_limits<std::size_t>::max();
  double coefficient = 0.0;
  std::vector<runtime::system::AuxiliaryComponentKey> output_keys;
  int gradient_sign = 1;
  std::function<void(const MultiFab<Dim, MemorySpace>&, MultiFab<Dim, MemorySpace>&)> rhs;
};

/// Complete inert candidate produced by one native AMR installer callback.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct PreparedNativeAmrPackage {
  PreparedAmrSystemBlock<Dim, MemorySpace> block;
  std::vector<PreparedNativeAmrEllipticAttachment<Dim, MemorySpace>> elliptic_attachments;
};

namespace generated_amr_detail {

template <int Dim, class Model>
struct MaterializePointwiseProjection {
  static constexpr int provider_count = provider_count_for<Model, Dim>();
  Model model;
  FieldView<const Real, Dim> source{};
  FieldView<Real, Dim> destination{};
  ProviderStorageView<Dim, provider_count> providers{};
  FieldView<const Real, Dim> active{};
  FieldView<Real, Dim> status{};
  bool has_active_mask = false;

  POPS_HD void operator()(const Index<Dim>& index) const {
    if (has_active_mask && active(index) < Real(0.5)) {
      status(index) = Real(0);
      return;
    }
    const auto projected = model.project(load_state<Model>(source, index),
                                         load_provider_values<provider_count>(providers, index));
    Real failure = Real(0);
    for (int component = 0; component < Model::n_vars; ++component) {
      destination(index, component) = projected[component];
      if (!Kokkos::isfinite(projected[component]))
        failure = Real(1);
    }
    status(index) = failure;
  }
};

template <int Dim, nd::ReconstructionVariables Variables, class Model, class Metric,
          class Reconstruction, class Numerical, class MemorySpace, int ProviderCount>
void materialize_masked_patch(
    const Model& model, const Metric& metric, const Reconstruction& reconstruction,
    const Numerical& numerical, Real positivity_floor, const Fab<Dim, MemorySpace>& state,
    const ProviderStorageView<Dim, ProviderCount>& providers, const Fab<Dim, MemorySpace>& active,
    nd::FaceField<Dim, MemorySpace>& faces, Fab<Dim, MemorySpace>& residual,
    nd::FaceField<Dim, MemorySpace>& face_candidate, nd::FaceField<Dim, MemorySpace>& face_statuses,
    Fab<Dim, MemorySpace>& residual_candidate, Fab<Dim, MemorySpace>& cell_statuses) {
  if constexpr (DiffusiveModel<Model>)
    throw std::invalid_argument(
        "generated AMR embedded transport does not support Fickian diffusion without EB face "
        "geometry");
  const int positivity_component =
      nd::cartesian_operator_detail::resolve_positivity_component<Model>(positivity_floor);
  nd::masked_operator_detail::materialize_axes<0, Variables>(
      model, metric, reconstruction, numerical, positivity_floor, positivity_component, state,
      active, providers, face_candidate, face_statuses, {});
  const Real face_failure = nd::cartesian_operator_detail::maximum_face_status<0>(face_statuses);
  if (face_failure != static_cast<Real>(nd::FiniteVolumeStatus::Success))
    throw std::runtime_error("generated AMR embedded face evaluation refused publication");

  for_each_cell(
      state.box(),
      nd::masked_operator_detail::MaterializeMaskedResidual<Dim, Metric, Model::n_vars>{
          metric, static_cast<const nd::FaceField<Dim, MemorySpace>&>(face_candidate).view(),
          active.view(), residual_candidate.view(), cell_statuses.view()});
  const Real cell_failure = for_each_cell_reduce_max(
      state.box(), nd::cartesian_operator_detail::FieldStatusMaximum<Dim>{
                       static_cast<const Fab<Dim, MemorySpace>&>(cell_statuses).view()});
  if (cell_failure != static_cast<Real>(nd::FiniteVolumeStatus::Success))
    throw std::runtime_error("generated AMR embedded residual refused publication");
  nd::cartesian_operator_detail::copy_face_axes<0>(face_candidate, faces, Model::n_vars);
  for_each_cell(state.box(),
                nd::cartesian_operator_detail::CopyCellField<Dim>{
                    static_cast<const Fab<Dim, MemorySpace>&>(residual_candidate).view(),
                    residual.view(), Model::n_vars});
  device_fence();
}

template <int Dim, nd::ReconstructionVariables Variables, class Model, class Spatial,
          class Reconstruction, class Numerical, class MemorySpace, int ProviderCount>
void materialize_cut_cell_patch(
    const Model& model, const Spatial& spatial, const Reconstruction& reconstruction,
    const Numerical& numerical, Real positivity_floor, const Fab<Dim, MemorySpace>& state,
    const ProviderStorageView<Dim, ProviderCount>& providers,
    const runtime::system::PreparedEmbeddedBoundaryGeometry<Dim>& embedded, std::size_t local,
    nd::FaceField<Dim, MemorySpace>& faces, Fab<Dim, MemorySpace>& residual,
    nd::FaceField<Dim, MemorySpace>& face_candidate, nd::FaceField<Dim, MemorySpace>& face_statuses,
    Fab<Dim, MemorySpace>& residual_candidate, Fab<Dim, MemorySpace>& cell_statuses) {
  const auto metric =
      nd::PreparedEmbeddedBoundaryMetric<Dim, std::remove_cvref_t<decltype(spatial.metric())>>::
          prepare(spatial.metric(), embedded.inverse_volume_fraction().fab(local),
                  embedded.face_aperture_lower().fab(local),
                  embedded.face_aperture_upper().fab(local), state.box());
  materialize_masked_patch<Dim, Variables>(
      model, metric, reconstruction, numerical, positivity_floor, state, providers,
      embedded.active_mask().fab(local), faces, residual, face_candidate, face_statuses,
      residual_candidate, cell_statuses);
}

/// Generated AMR seam for the one uniform-ratio CutCellFractions restrict/prolong/reflux path.
template <int Dim>
void apply_generated_amr_cut_cell_fraction_transfer(FieldView<const Real, Dim> fine_phi,
                                                    FieldView<Real, Dim> coarse_volume,
                                                    FieldView<Real, Dim> fine_volume,
                                                    FieldView<Real, Dim> coarse_aperture_residual,
                                                    const Box<Dim>& coarse_region,
                                                    const amr::RefinementRatio<Dim>& ratio,
                                                    amr::transfer::IndexMapping<Dim> mapping = {}) {
  nd::apply_cut_cell_fraction_amr_transfer(fine_phi, coarse_volume, fine_volume,
                                           coarse_aperture_residual, coarse_region, ratio, mapping);
}

template <int Dim, class Model, class Reconstruction, class Numerical,
          nd::ReconstructionVariables Variables, class Request>
PreparedAmrSystemBlock<Dim> materialize_system(Request request, Reconstruction reconstruction,
                                               Numerical numerical) {
  static_assert(Model::dimension == Dim);
  if (request.provider_consumer_qid.empty())
    throw std::invalid_argument("generated AMR block requires one explicit provider consumer qid");
  constexpr int provider_count = provider_count_for<Model, Dim>();
  const auto spatial_factory =
      [model = request.model, reconstruction, numerical,
       positivity_floor = request.routes.positivity_floor](const Geometry<Dim>& geometry) {
        return nd::prepare_cartesian_operator<Dim, Model, Reconstruction, Numerical, Variables>(
            geometry, model, reconstruction, numerical, positivity_floor);
      };

  Extent<Dim> required_ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    required_ghosts[axis] = Reconstruction::n_ghost;

  const Model model = request.model;
  const std::string model_contract = exact_model_contract(model);
  const std::string name = request.name;
  const std::string provider_identity =
      "pops.generated.amr.cartesian.nd/" + std::to_string(Dim) + "/" + request.routes.limiter +
      "/" + request.routes.riemann + "/" + request.routes.reconstruction;
  const std::string staircase_provider_identity =
      "pops.generated.amr.staircase.nd/" + std::to_string(Dim) + "/" + provider_identity;
  const std::string cut_cell_provider_identity =
      "pops.generated.amr.cut-cell.nd/" + std::to_string(Dim) + "/" + provider_identity;

  PreparedAmrSystemBlock<Dim> result;
  result.name = name;
  result.provider_identity = provider_identity;
  result.provider_consumer_qid = request.provider_consumer_qid;
  result.staircase_provider_identity = staircase_provider_identity;
  result.cut_cell_provider_identity = cut_cell_provider_identity;
  result.ncomp = Model::n_vars;
  result.provider_components = provider_count;
  result.has_pointwise_projection = HasPointwiseProjection<Model>;
  result.conservative_variables = Model::conservative_vars();
  result.primitive_variables = Model::primitive_vars();
  result.gamma = request.gamma;
  result.ghosts = required_ghosts;
  result.reconstruction_order = Reconstruction::formal_order;
  result.substeps = request.substeps;
  result.stride = request.stride;
  result.newton = request.newton;
  result.newton_diagnostics = request.newton_diagnostics;
  result.time_route = request.routes.time;

  ExactContractBuilder package_contract;
  package_contract.text("pops.prepared-generated-amr-system-block")
      .scalar(std::uint32_t{6})
      .scalar(std::int32_t{Dim})
      .text(name)
      .text(provider_identity)
      .text(request.provider_consumer_qid)
      .text(staircase_provider_identity)
      .text(cut_cell_provider_identity)
      .scalar(std::int32_t{Model::n_vars})
      .scalar(std::int32_t{provider_count})
      .presence(result.has_pointwise_projection)
      .scalar(std::int32_t{Reconstruction::formal_order})
      .scalar(request.gamma)
      .scalar(std::int32_t{request.substeps})
      .scalar(std::int32_t{request.stride})
      .scalar(std::int32_t{request.newton.max_iters})
      .scalar(static_cast<double>(request.newton.rel_tol))
      .scalar(static_cast<double>(request.newton.abs_tol))
      .scalar(static_cast<double>(request.newton.fd_eps))
      .scalar(static_cast<double>(request.newton.damping))
      .scalar(request.newton_diagnostics)
      .text(request.routes.time)
      .bytes(model_contract)
      .scalar(static_cast<double>(request.routes.positivity_floor))
      .scalar(static_cast<double>(request.routes.weno_epsilon));
  append_variable_set_contract(package_contract, result.conservative_variables);
  append_variable_set_contract(package_contract, result.primitive_variables);
  for (int axis = 0; axis < Dim; ++axis)
    package_contract.scalar(std::int64_t{required_ghosts[axis]});
  result.collective_contract = std::move(package_contract).release();

  result.materialize_level = [model, spatial_factory, reconstruction, numerical,
                              positivity_floor = request.routes.positivity_floor, required_ghosts,
                              provider_identity, staircase_provider_identity,
                              cut_cell_provider_identity](const RuntimeTopologyView<Dim>& runtime,
                                                          GeneratedAmrLevelContext<Dim> context) {
    require_level_context(runtime, context, Model::n_vars, provider_count, required_ghosts,
                          staircase_provider_identity, cut_cell_provider_identity);
    const auto spatial = spatial_factory(context.geometry);
    runtime::system::AuxiliaryStorageGroups<Dim>* const provider_storage = context.provider_storage;
    const auto* const provider_plan = context.provider_plan;
    const auto state_ghost_fill = context.state_ghost_fill;
    const auto provider_ghost_fill = context.provider_ghost_fill;
    const auto root_state_ghost_fill = context.root_state_ghost_fill;
    const auto root_provider_ghost_fill = context.root_provider_ghost_fill;
    const auto physical_boundary = context.physical_boundary;
    const auto external_ghost_boundary = context.external_ghost_boundary;
    const auto external_boundary_flux = context.external_boundary_flux;
    const auto external_field_boundaries = context.external_field_boundaries;
    const auto embedded_boundary = context.embedded_boundary;
    const Geometry<Dim> geometry = context.geometry;
    const std::size_t level = context.level;
    const ExecutionLane* const lane = context.lane;
    // This is deliberately constructed while the generated hierarchy image is cold.  The
    // workspace belongs to this exact block/level route and is never resized or rebound during a
    // Program attempt; hierarchy materialization publishes a replacement bundle after regrid.
    auto implicit_workspace = std::make_shared<PreparedImplicitSourceWorkspace<Dim>>();
    const std::uint64_t implicit_workspace_generation = runtime.materialization_generation();
    implicit_workspace->bind(*context.state, implicit_workspace_generation);

    struct EvaluationScratch {
      static PreparedAmrLevelEvaluation<Dim> make_evaluation(
          const MultiFab<Dim>& prototype, std::string_view spatial_contract,
          std::uint64_t topology_epoch, std::uint64_t materialization_generation,
          std::size_t clock_capacity) {
        PreparedAmrLevelEvaluation<Dim> evaluation{
            .spatial_contract = std::string(spatial_contract),
            .topology_epoch = topology_epoch,
            .materialization_generation = materialization_generation,
            .residual =
                MultiFab<Dim>(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                              prototype.ncomp(), prototype.ghosts()),
            .integrated_face_fluxes = nd::make_face_flux_workspace(prototype)};
        evaluation.point.clock.reserve(clock_capacity);
        return evaluation;
      }

      EvaluationScratch(const MultiFab<Dim>& prototype, std::string_view spatial_contract,
                        std::uint64_t topology_epoch, std::uint64_t materialization_generation,
                        std::size_t clock_capacity)
          : physical(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                     prototype.ncomp(), prototype.ghosts()),
            core(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                 prototype.ncomp(), prototype.ghosts()),
            physical_boundary_candidate(prototype.layout(), prototype.distribution(),
                                        prototype.local_rank(), prototype.ncomp(),
                                        prototype.ghosts()),
            physical_source(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                            prototype.ncomp(), prototype.ghosts()),
            core_source(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                        prototype.ncomp(), prototype.ghosts()),
            source_values(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                          prototype.ncomp(), prototype.ghosts()),
            source_status(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
                          prototype.ghosts()),
            perturbed(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                      prototype.ncomp(), prototype.ghosts()),
            spatial(prototype),
            core_evaluation(make_evaluation(prototype, spatial_contract, topology_epoch,
                                            materialization_generation, clock_capacity)),
            base_evaluation(make_evaluation(prototype, spatial_contract, topology_epoch,
                                            materialization_generation, clock_capacity)),
            shifted_evaluation(make_evaluation(prototype, spatial_contract, topology_epoch,
                                               materialization_generation, clock_capacity)) {}
      std::recursive_mutex mutex;
      MultiFab<Dim> physical;
      MultiFab<Dim> core;
      MultiFab<Dim> physical_boundary_candidate;
      MultiFab<Dim> physical_source;
      MultiFab<Dim> core_source;
      MultiFab<Dim> source_values;
      MultiFab<Dim> source_status;
      MultiFab<Dim> perturbed;
      nd::PreparedCartesianOperatorScratch<Dim> spatial;
      PreparedAmrLevelEvaluation<Dim> core_evaluation;
      PreparedAmrLevelEvaluation<Dim> base_evaluation;
      PreparedAmrLevelEvaluation<Dim> shifted_evaluation;
    };
    auto evaluation_scratch = std::make_shared<EvaluationScratch>(
        *context.state, runtime.spatial_contract(), runtime.topology_epoch(),
        runtime.materialization_generation(), context.clock_identity_capacity);
    evaluation_scratch->spatial.require_layout(*context.state);

    if (physical_boundary)
      physical_boundary->template require_model_qualified_characteristic_provider<Model>();

    auto prepare_state_with_physical =
        [provider_storage, state_ghost_fill, provider_ghost_fill, root_state_ghost_fill,
         root_provider_ghost_fill, physical_boundary, external_ghost_boundary, geometry, level,
         model, lane, evaluation_scratch](const runtime::multiblock::BoundaryEvaluationPoint& point,
                                          MultiFab<Dim>& state, bool physical) {
          std::lock_guard lock(evaluation_scratch->mutex);
          collective_phase(
              *lane,
              [&] {
                if (level == 0) {
                  root_state_ghost_fill(state, point);
                  if constexpr (provider_count > 0)
                    root_provider_ghost_fill(*provider_storage, point);
                } else {
                  state_ghost_fill(state, point);
                  if constexpr (provider_count > 0)
                    provider_ghost_fill(*provider_storage, point);
                }
                if (physical && physical_boundary)
                  physical_boundary->fill_physical_model_qualified(
                      state, geometry, model, *lane,
                      evaluation_scratch->physical_boundary_candidate);
                if (physical && external_ghost_boundary)
                  external_ghost_boundary(point, state, geometry, *lane);
              },
              "generated AMR ghost/boundary phase failed collectively");
        };
    auto prepare_state = [prepare_state_with_physical](const auto& point, auto& state) {
      prepare_state_with_physical(point, state, true);
    };
    auto prepare_physical = [physical_boundary, external_ghost_boundary, geometry, model, lane,
                             evaluation_scratch](
                                const runtime::multiblock::BoundaryEvaluationPoint& point,
                                MultiFab<Dim>& state) {
      std::lock_guard lock(evaluation_scratch->mutex);
      collective_phase(
          *lane,
          [&] {
            if (physical_boundary)
              physical_boundary->fill_physical_model_qualified(
                  state, geometry, model, *lane, evaluation_scratch->physical_boundary_candidate);
            if (external_ghost_boundary)
              external_ghost_boundary(point, state, geometry, *lane);
          },
          "generated AMR detached physical boundary phase failed collectively");
    };

    auto make_flux_evaluator = [model, spatial, reconstruction, numerical, positivity_floor,
                                provider_storage, provider_plan, prepare_state_with_physical,
                                physical_boundary, external_boundary_flux, embedded_boundary,
                                geometry, lane, evaluation_scratch](bool physical) {
      return [model, spatial, reconstruction, numerical, positivity_floor, provider_storage,
              provider_plan, prepare_state_with_physical, physical_boundary, external_boundary_flux,
              embedded_boundary, geometry, lane, evaluation_scratch,
              physical](const runtime::multiblock::BoundaryEvaluationPoint& point,
                        MultiFab<Dim>& state, PreparedAmrLevelEvaluation<Dim>& evaluation) {
        std::lock_guard lock(evaluation_scratch->mutex);
        MultiFab<Dim>& image = physical ? evaluation_scratch->physical : evaluation_scratch->core;
        collective_phase(
            *lane,
            [&] {
              image.set_val(Real(0));
              generated_system_detail::copy_valid(state, image);
              evaluation.residual.set_val(Real(0));
              for (auto& faces : evaluation.integrated_face_fluxes)
                faces.set_val(Real(0));
            },
            "generated AMR evaluation workspace reset failed collectively");
        prepare_state_with_physical(point, image, physical);
        auto& faces = evaluation.integrated_face_fluxes;
        MultiFab<Dim>& residual = evaluation.residual;
        collective_phase(
            *lane,
            [&] {
              if (embedded_boundary &&
                  embedded_boundary->mode() !=
                      runtime::system::PreparedEmbeddedBoundaryMode::inactive) {
                for (std::size_t local = 0; local < image.local_size(); ++local) {
                  auto& face_candidate = evaluation_scratch->spatial.face_candidate(local);
                  auto& face_status = evaluation_scratch->spatial.face_status(local);
                  auto& residual_candidate =
                      evaluation_scratch->spatial.residual_candidate().fab(local);
                  auto& residual_status = evaluation_scratch->spatial.residual_status().fab(local);
                  if (embedded_boundary->mode() ==
                      runtime::system::PreparedEmbeddedBoundaryMode::staircase) {
                    materialize_masked_patch<Dim, Variables>(
                        model, spatial.metric(), reconstruction, numerical, positivity_floor,
                        image.fab(local),
                        runtime::system::bind_provider_storage_view<Dim, provider_count>(
                            provider_plan, provider_storage, local),
                        embedded_boundary->active_mask().fab(local), faces[local],
                        residual.fab(local), face_candidate, face_status, residual_candidate,
                        residual_status);
                  } else {
                    materialize_cut_cell_patch<Dim, Variables>(
                        model, spatial, reconstruction, numerical, positivity_floor,
                        image.fab(local),
                        runtime::system::bind_provider_storage_view<Dim, provider_count>(
                            provider_plan, provider_storage, local),
                        *embedded_boundary, local, faces[local], residual.fab(local),
                        face_candidate, face_status, residual_candidate, residual_status);
                  }
                }
              } else {
                for (std::size_t local = 0; local < image.local_size(); ++local) {
                  if constexpr (provider_count == 0)
                    spatial.materialize_face_fluxes(
                        image.fab(local), faces[local],
                        evaluation_scratch->spatial.face_candidate(local),
                        evaluation_scratch->spatial.face_status(local));
                  else
                    spatial.materialize_face_fluxes(
                        image.fab(local),
                        runtime::system::bind_provider_storage_view<Dim, provider_count>(
                            provider_plan, provider_storage, local),
                        faces[local], evaluation_scratch->spatial.face_candidate(local),
                        evaluation_scratch->spatial.face_status(local));
                  if (physical && physical_boundary)
                    physical_boundary->apply_physical_flux_conditions(faces[local],
                                                                      geometry.domain());
                }
                if (physical && external_boundary_flux)
                  external_boundary_flux(point, image, faces, geometry, *lane);
                spatial.assemble_residual_from_face_fluxes(
                    faces, residual, evaluation_scratch->spatial.residual_candidate(),
                    evaluation_scratch->spatial.residual_status());
              }
            },
            "generated AMR flux/residual materialization failed collectively");
      };
    };
    auto flux_evaluator = make_flux_evaluator(true);
    auto flux_core_evaluator = make_flux_evaluator(false);
    auto source_evaluator = [model, provider_storage, provider_plan, prepare_state,
                             embedded_boundary, lane, evaluation_scratch](
                                const runtime::multiblock::BoundaryEvaluationPoint& point,
                                MultiFab<Dim>& state, MultiFab<Dim>& result) {
      std::lock_guard lock(evaluation_scratch->mutex);
      MultiFab<Dim>& image = evaluation_scratch->physical_source;
      MultiFab<Dim>& source = evaluation_scratch->source_values;
      MultiFab<Dim>& status = evaluation_scratch->source_status;
      collective_phase(
          *lane,
          [&] {
            image.set_val(Real(0));
            generated_system_detail::copy_valid(state, image);
            source.set_val(Real(0));
            status.set_val(Real(0));
          },
          "generated AMR source workspace reset failed collectively");
      prepare_state(point, image);
      Real local_status = Real(0);
      collective_phase(
          *lane,
          [&] {
            if constexpr (generated_system_detail::GeneratedSourceModel<Dim, Model>) {
              const bool has_active_mask =
                  embedded_boundary && embedded_boundary->mode() !=
                                           runtime::system::PreparedEmbeddedBoundaryMode::inactive;
              for (std::size_t local = 0; local < image.local_size(); ++local)
                for_each_cell(
                    image.box(local),
                    generated_system_detail::MaterializeSource<Dim, Model>{
                        model, std::as_const(image).fab(local).view(),
                        runtime::system::bind_provider_storage_view<Dim, provider_count>(
                            provider_plan, provider_storage, local),
                        source.fab(local).view(), status.fab(local).view(),
                        has_active_mask ? embedded_boundary->active_mask().fab(local).view()
                                        : FieldView<const Real, Dim>{},
                        has_active_mask});
              local_status = reduce_max_local(status);
            }
          },
          "generated AMR source materialization failed collectively");
      if (all_reduce_max(local_status, *lane) != Real(0))
        throw std::runtime_error("generated AMR source produced a non-finite component");
      collective_phase(
          *lane, [&] { generated_system_detail::copy_valid(source, result); },
          "generated AMR source publication failed collectively");
    };
    auto implicit_source_solver = [model, provider_storage, provider_plan, prepare_state,
                                   embedded_boundary, lane, implicit_workspace,
                                   implicit_workspace_generation](
                                      const runtime::multiblock::BoundaryEvaluationPoint& point,
                                      MultiFab<Dim>& state, Real dt, const NewtonOptions& options) {
      const auto provider_at = [provider_storage, provider_plan](std::size_t local) {
        if constexpr (provider_count == 0)
          return ProviderStorageView<Dim, 0>{};
        else
          return runtime::system::bind_provider_storage_view<Dim, provider_count>(
              provider_plan, provider_storage, local);
      };
      const MultiFab<Dim>* active_cells = nullptr;
      if (embedded_boundary &&
          embedded_boundary->mode() != runtime::system::PreparedEmbeddedBoundaryMode::inactive)
        active_cells = &embedded_boundary->active_mask();
      if constexpr (generated_system_detail::GeneratedSourceModel<Dim, Model>) {
        return detail::backward_euler_source_prepared(
            model, provider_at, state, *implicit_workspace, implicit_workspace_generation, dt,
            options, *lane, {}, active_cells, std::static_pointer_cast<void>(implicit_workspace),
            [&prepare_state, &point, &state] { prepare_state(point, state); });
      } else {
        return SolveOutcome::collective_lane(SolveReport::capability_failure(), *lane);
      }
    };
    auto make_evaluator = [model, provider_storage, provider_plan, embedded_boundary, lane,
                           evaluation_scratch](auto flux, bool physical) {
      return [flux = std::move(flux), model, provider_storage, provider_plan, embedded_boundary,
              lane, evaluation_scratch,
              physical](const runtime::multiblock::BoundaryEvaluationPoint& point,
                        MultiFab<Dim>& state, PreparedAmrLevelEvaluation<Dim>& evaluation) mutable {
        std::lock_guard lock(evaluation_scratch->mutex);
        flux(point, state, evaluation);
        MultiFab<Dim>& source_image =
            physical ? evaluation_scratch->physical_source : evaluation_scratch->core_source;
        auto& source = evaluation_scratch->source_values;
        Real local_status = Real(0);
        collective_phase(
            *lane,
            [&] {
              source_image.set_val(Real(0));
              generated_system_detail::copy_valid(state, source_image);
              source.set_val(Real(0));
              if constexpr (generated_system_detail::GeneratedSourceModel<Dim, Model>) {
                auto& status = evaluation_scratch->source_status;
                status.set_val(Real(0));
                const bool has_active_mask =
                    embedded_boundary &&
                    embedded_boundary->mode() !=
                        runtime::system::PreparedEmbeddedBoundaryMode::inactive;
                for (std::size_t local = 0; local < source_image.local_size(); ++local)
                  for_each_cell(
                      source_image.box(local),
                      generated_system_detail::MaterializeSource<Dim, Model>{
                          model, std::as_const(source_image).fab(local).view(),
                          runtime::system::bind_provider_storage_view<Dim, provider_count>(
                              provider_plan, provider_storage, local),
                          source.fab(local).view(), status.fab(local).view(),
                          has_active_mask ? embedded_boundary->active_mask().fab(local).view()
                                          : FieldView<const Real, Dim>{},
                          has_active_mask});
                local_status = reduce_max_local(status);
              }
            },
            "generated AMR source workspace materialization failed collectively");
        if (all_reduce_max(local_status, *lane) != Real(0))
          throw std::runtime_error("generated AMR source produced a non-finite component");
        collective_phase(
            *lane, [&] { saxpy(evaluation.residual, Real(1), source); },
            "generated AMR source residual accumulation failed collectively");
      };
    };
    auto evaluator = make_evaluator(flux_evaluator, true);
    auto core_evaluator = make_evaluator(flux_core_evaluator, false);
    auto field_boundary_context = [](const auto& point) {
      FieldBoundaryExecutionContext<Dim> context;
      context.point.time = static_cast<Real>(point.physical_time);
      context.point.dt = static_cast<Real>(point.dt);
      context.point.stage_slot = point.stage;
      context.point.level = point.level;
      context.point.step = point.tick;
      context.point.substep = point.substep;
      context.point.stage_fraction_numerator = point.stage_fraction.numerator;
      context.point.stage_fraction_denominator = point.stage_fraction.denominator;
      context.clock_identity = &point.clock;
      return context;
    };
    auto boundary_evaluator = [evaluator, core_evaluator, external_field_boundaries, geometry,
                               field_boundary_context,
                               evaluation_scratch](const auto& point, MultiFab<Dim>& state,
                                                   PreparedAmrLevelEvaluation<Dim>& full) mutable {
      std::lock_guard lock(evaluation_scratch->mutex);
      evaluator(point, state, full);
      core_evaluator(point, state, evaluation_scratch->core_evaluation);
      lincomb(full.residual, Real(1), full.residual, Real(-1),
              evaluation_scratch->core_evaluation.residual);
      const FieldBoundaryExecutionContext<Dim> context = field_boundary_context(point);
      for (const auto& closure : external_field_boundaries)
        for (int face = 0; face < 2 * Dim; ++face)
          closure.add_residual(face, state, full.residual, geometry, context);
    };
    auto built_in_boundary_evaluator = [evaluator, core_evaluator, evaluation_scratch](
                                           const auto& point, MultiFab<Dim>& state,
                                           PreparedAmrLevelEvaluation<Dim>& full) mutable {
      std::lock_guard lock(evaluation_scratch->mutex);
      evaluator(point, state, full);
      core_evaluator(point, state, evaluation_scratch->core_evaluation);
      lincomb(full.residual, Real(1), full.residual, Real(-1),
              evaluation_scratch->core_evaluation.residual);
    };
    auto boundary_jvp = [built_in_boundary_evaluator, external_field_boundaries,
                         field_boundary_context, geometry, lane, evaluation_scratch](
                            const auto& point, MultiFab<Dim>& state, const MultiFab<Dim>& direction,
                            MultiFab<Dim>& result) mutable {
      std::lock_guard lock(evaluation_scratch->mutex);
      Real local_state_norm = Real(0);
      Real local_direction_norm = Real(0);
      collective_phase(
          *lane,
          [&] {
            device_fence();
            local_state_norm = all_component_norm_inf(state);
            local_direction_norm = all_component_norm_inf(direction);
          },
          "generated AMR boundary JVP norm preparation failed collectively");
      const Real state_norm =
          static_cast<Real>(all_reduce_max(static_cast<double>(local_state_norm), *lane));
      const Real direction_norm =
          static_cast<Real>(all_reduce_max(static_cast<double>(local_direction_norm), *lane));
      const Real step = direction_norm > Real(0) ? std::sqrt(std::numeric_limits<Real>::epsilon()) *
                                                       (Real(1) + state_norm) / direction_norm
                                                 : std::sqrt(std::numeric_limits<Real>::epsilon());
      MultiFab<Dim>& perturbed = evaluation_scratch->perturbed;
      collective_phase(
          *lane,
          [&] {
            perturbed.set_val(Real(0));
            generated_system_detail::copy_valid(state, perturbed);
            lincomb(perturbed, Real(1), state, step, direction);
          },
          "generated AMR boundary JVP perturbation staging failed collectively");
      built_in_boundary_evaluator(point, state, evaluation_scratch->base_evaluation);
      built_in_boundary_evaluator(point, perturbed, evaluation_scratch->shifted_evaluation);
      lincomb(result, Real(1) / step, evaluation_scratch->shifted_evaluation.residual,
              Real(-1) / step, evaluation_scratch->base_evaluation.residual);
      const FieldBoundaryExecutionContext<Dim> context = field_boundary_context(point);
      for (const auto& closure : external_field_boundaries)
        for (int face = 0; face < 2 * Dim; ++face)
          closure.apply_jvp(face, state, direction, result, geometry, context);
    };
    auto speed = [model, provider_storage, provider_plan, lane](const MultiFab<Dim>& state) {
      return maximum_speed<Dim>(model, state, provider_storage, provider_plan, *lane);
    };
    auto poisson_rhs = [model](const MultiFab<Dim>& state, MultiFab<Dim>& rhs) {
      add_poisson_rhs_locally<Dim>(model, state, rhs);
    };
    typename PreparedGeneratedAmrLevelBlock<Dim>::PointwiseProjection pointwise_projection;
    if constexpr (HasPointwiseProjection<Model>) {
      pointwise_projection = [model, provider_storage, provider_plan, embedded_boundary, lane,
                              evaluation_scratch](MultiFab<Dim>& detached_candidate) {
        std::lock_guard lock(evaluation_scratch->mutex);
        MultiFab<Dim>& status = evaluation_scratch->source_status;
        Real local_status = Real(0);
        collective_phase(
            *lane,
            [&] {
              status.set_val(Real(0));
              const bool has_active_mask =
                  embedded_boundary && embedded_boundary->mode() !=
                                           runtime::system::PreparedEmbeddedBoundaryMode::inactive;
              for (std::size_t local = 0; local < detached_candidate.local_size(); ++local) {
                const FieldView<const Real, Dim> source =
                    std::as_const(detached_candidate).fab(local).view();
                const FieldView<const Real, Dim> active =
                    has_active_mask ? embedded_boundary->active_mask().fab(local).view()
                                    : FieldView<const Real, Dim>{};
                for_each_cell(detached_candidate.box(local),
                              MaterializePointwiseProjection<Dim, Model>{
                                  model, source, detached_candidate.fab(local).view(),
                                  runtime::system::bind_provider_storage_view<Dim, provider_count>(
                                      provider_plan, provider_storage, local),
                                  active, status.fab(local).view(), has_active_mask});
              }
              device_fence();
              local_status = reduce_max_local(status);
            },
            "generated AMR pointwise projection failed collectively");
        if (all_reduce_max(local_status, *lane) != Real(0))
          throw std::runtime_error(
              "generated AMR pointwise projection produced a non-finite value");
      };
    }
    typename PreparedGeneratedAmrLevelBlock<Dim>::Speed source_frequency_bound;
    if constexpr (requires(const Model& value, const typename Model::State& state,
                           const ProviderValues<provider_count>& providers) {
                    value.source_frequency(state, providers);
                  }) {
      source_frequency_bound = [model, provider_storage, provider_plan,
                                lane](const MultiFab<Dim>& state) {
        return source_frequency<Dim>(model, state, provider_storage, provider_plan, *lane);
      };
    }
    std::optional<Real> parabolic_frequency_bound;
    if constexpr (DiffusiveModel<Model>) {
      const Real q = generated_system_detail::parabolic_frequency<Dim>(model, spatial.metric());
      parabolic_frequency_bound = q;
    }
    typename PreparedGeneratedAmrLevelBlock<Dim>::Speed stability_dt_bound;
    if constexpr (requires(const Model& value, const typename Model::State& state,
                           const ProviderValues<provider_count>& providers) {
                    value.stability_dt(state, providers);
                  }) {
      auto model_stability_dt = [model, provider_storage, provider_plan,
                                 lane](const MultiFab<Dim>& state) {
        return stability_dt<Dim>(model, state, provider_storage, provider_plan, *lane);
      };
      stability_dt_bound = std::move(model_stability_dt);
    }
    std::string contract =
        level_contract(runtime, context, provider_identity, parabolic_frequency_bound);
    MultiFab<Dim>* const bound_state = context.state;
    return PreparedGeneratedAmrLevelBlock<Dim>(
        runtime, level, *bound_state, std::move(context.state_identity), provider_identity,
        std::move(contract), *lane, std::move(prepare_state), std::move(prepare_physical),
        std::move(evaluator), std::move(flux_evaluator), std::move(core_evaluator),
        std::move(flux_core_evaluator), std::move(boundary_evaluator), std::move(boundary_jvp),
        std::move(source_evaluator), std::move(implicit_workspace), implicit_workspace_generation,
        std::move(implicit_source_solver), std::move(speed), std::move(poisson_rhs),
        std::move(pointwise_projection), std::move(source_frequency_bound),
        std::move(parabolic_frequency_bound), std::move(stability_dt_bound));
  };

  result.primitive_to_conservative = [model](const double* primitive, double* conservative) {
    generated_system_detail::publish_conservative_state(model, primitive, conservative);
  };
  auto recovery = std::make_shared<PreparedModelVariableInversionRecovery<Model>>(model);
  result.conservative_to_primitive = [recovery](const double* conservative, double* primitive) {
    Real input[Model::n_vars]{};
    for (int component = 0; component < Model::n_vars; ++component)
      input[component] = static_cast<Real>(conservative[component]);
    const auto prepared = recovery->recover(input);
    const RecoveryOutcome<Model::n_vars>& outcome = prepared.outcome;
    if (outcome.publication_permitted())
      for (int component = 0; component < Model::n_vars; ++component)
        primitive[component] = static_cast<double>(outcome.value[component]);
    return recovery_report(outcome);
  };
  result.batch_conservative_to_primitive = make_uniform_variable_inversion_consumer(recovery);
  return result;
}

template <int Dim, nd::ReconstructionVariables Variables, class Request, class Reconstruction>
PreparedAmrSystemBlock<Dim> select_riemann(Request request, Reconstruction reconstruction) {
  using Model = std::remove_cvref_t<decltype(request.model)>;
  switch (parse_riemann_route(request.routes.riemann, "generated AMR block")) {
    case RiemannRouteId::kRusanov:
      return materialize_system<Dim, Model, Reconstruction, RusanovFlux, Variables>(
          std::move(request), reconstruction, RusanovFlux{});
    case RiemannRouteId::kHll:
      if constexpr (detail::wave_speeds_all_axes<Model>())
        return materialize_system<Dim, Model, Reconstruction, HLLFlux, Variables>(
            std::move(request), reconstruction, HLLFlux{});
      break;
    case RiemannRouteId::kHllc:
      if constexpr (HasHLLCStructure<Model>)
        return materialize_system<Dim, Model, Reconstruction, HLLCFlux, Variables>(
            std::move(request), reconstruction, HLLCFlux{});
      break;
    case RiemannRouteId::kRoe:
      if constexpr (HasRoeDissipation<Model>)
        return materialize_system<Dim, Model, Reconstruction, RoeFlux, Variables>(
            std::move(request), reconstruction, RoeFlux{});
      break;
    case RiemannRouteId::kRoeHllRusanovRecovery:
      if constexpr (HasRoeDissipation<Model> && detail::wave_speeds_all_axes<Model>())
        return materialize_system<Dim, Model, Reconstruction, RoeHllRusanovRecoveryPolicy,
                                  Variables>(std::move(request), reconstruction,
                                             RoeHllRusanovRecoveryPolicy{});
      break;
  }
  throw std::invalid_argument(
      "generated model does not satisfy the requested AMR Riemann capability");
}

template <int Dim, nd::ReconstructionVariables Variables, class Request>
PreparedAmrSystemBlock<Dim> select_reconstruction(Request request) {
  switch (parse_limiter_route(request.routes.limiter, "generated AMR block")) {
    case LimiterRouteId::kNone:
      return select_riemann<Dim, Variables>(std::move(request), NoSlope{});
    case LimiterRouteId::kMinmod:
      return select_riemann<Dim, Variables>(std::move(request), Minmod{});
    case LimiterRouteId::kVanLeer:
      return select_riemann<Dim, Variables>(std::move(request), VanLeer{});
    case LimiterRouteId::kWeno5: {
      const Real epsilon = request.routes.weno_epsilon;
      return select_riemann<Dim, Variables>(std::move(request),
                                            configured_reconstruction<Weno5>(epsilon));
    }
    case LimiterRouteId::kMc:
      return select_riemann<Dim, Variables>(std::move(request), MC{});
    case LimiterRouteId::kSuperbee:
      return select_riemann<Dim, Variables>(std::move(request), Superbee{});
  }
  throw std::logic_error("generated AMR limiter route escaped its exhaustive selector");
}

}  // namespace generated_amr_detail

/// Prepare one package-owned exact AMR block image without touching a runtime facade.
template <class Request>
auto prepare_generated_amr_system_block(Request request)
    -> PreparedAmrSystemBlock<Request::dimension> {
  constexpr int Dim = Request::dimension;
  using Model = std::remove_cvref_t<decltype(request.model)>;
  static_assert(Model::dimension == Dim,
                "generated AMR request and physical model have different ranks");
  switch (parse_recon_route(request.routes.reconstruction, "generated AMR block")) {
    case ReconRouteId::kConservative:
      return generated_amr_detail::select_reconstruction<Dim,
                                                         nd::ReconstructionVariables::Conservative>(
          std::move(request));
    case ReconRouteId::kPrimitive:
      return generated_amr_detail::select_reconstruction<Dim,
                                                         nd::ReconstructionVariables::Primitive>(
          std::move(request));
  }
  throw std::logic_error("generated AMR reconstruction route escaped its exhaustive selector");
}

/// Authored routes frozen into one exact-ranked generated AMR package image.
struct CompiledAmrSystemBlockRoutes {
  std::string limiter;
  std::string riemann;
  std::string reconstruction;
  std::string time;
  Real positivity_floor = Real(0);
  Real weno_epsilon = kWenoEpsilon;
  bool wave_speed_cache = false;
};

/// Immutable input to the exact generated AMR package factory.
template <int Dim, class Model>
struct CompiledAmrSystemBlockPreparation {
  static_assert(Dim >= 1 && Dim <= 3,
                "CompiledAmrSystemBlockPreparation only supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;

  std::string name;
  std::string provider_consumer_qid;
  Model model;
  CompiledAmrSystemBlockRoutes routes;
  double gamma = 1.0;
  int substeps = 1;
  int stride = 1;
  NewtonOptions newton{};
  bool newton_diagnostics = false;
};

namespace compiled_amr_detail {

inline void validate_routes(const CompiledAmrSystemBlockRoutes& routes) {
  if (routes.limiter.empty() || routes.riemann.empty())
    throw std::invalid_argument("compiled AMR block requires explicit limiter and Riemann routes");
  (void)parse_limiter_route(routes.limiter, "compiled AMR block");
  (void)parse_riemann_route(routes.riemann, "compiled AMR block");
  (void)parse_recon_route(routes.reconstruction, "compiled AMR block");
  if (routes.time != "explicit" && routes.time != "euler" && routes.time != "ssprk3" &&
      routes.time != "imex" && routes.time != "imexrk_ars222")
    throw std::invalid_argument("compiled AMR block has an unsupported Program time route");
  if (!std::isfinite(routes.positivity_floor) || routes.positivity_floor < Real(0))
    throw std::invalid_argument(
        "compiled AMR block positivity floor must be finite and non-negative");
  if (!std::isfinite(routes.weno_epsilon) || !(routes.weno_epsilon > Real(0)))
    throw std::invalid_argument("compiled AMR block WENO epsilon must be finite and positive");
  if (routes.limiter != "weno5" && routes.weno_epsilon != kWenoEpsilon)
    throw std::invalid_argument(
        "compiled AMR block WENO epsilon is only meaningful for limiter='weno5'");
  if (routes.wave_speed_cache)
    throw std::invalid_argument(
        "compiled exact-ranked AMR blocks have no prepared wave-speed cache provider");
}

}  // namespace compiled_amr_detail

/// Prepare a complete generated AMR block image without mutating the facade.
template <int Dim, class Model>
PreparedAmrSystemBlock<Dim> prepare_compiled_amr_system_block(
    const std::string& name, Model model, const std::string& limiter, const std::string& riemann,
    const std::string& reconstruction, const std::string& time, double gamma, int substeps,
    int stride, double positivity_floor = 0.0,
    double weno_epsilon = static_cast<double>(kWenoEpsilon), bool wave_speed_cache = false,
    const std::string& provider_consumer_qid = {}, NewtonOptions newton = {},
    bool newton_diagnostics = false) {
  static_assert(Dim >= 1 && Dim <= 3);
  static_assert(
      requires { Model::dimension; },
      "a generated AMR model must publish its exact spatial dimension");
  static_assert(Model::dimension == Dim,
                "generated model dimension differs from the target AmrSystem specialization");
  static_assert(requires {
    Model::n_vars;
    { Model::conservative_vars() } -> std::same_as<VariableSet>;
    { Model::primitive_vars() } -> std::same_as<VariableSet>;
  });

  if (name.empty())
    throw std::invalid_argument("compiled AMR block name must be non-empty");
  if (provider_consumer_qid.empty())
    throw std::invalid_argument("compiled AMR block requires one explicit provider consumer qid");
  if (!std::isfinite(gamma) || !(gamma > 0.0))
    throw std::invalid_argument("compiled AMR block gamma must be finite and positive");
  if (substeps < 1 || stride < 1)
    throw std::invalid_argument("compiled AMR block substeps and stride must be positive");
  validate_newton_options(newton, "compiled AMR block");
  if (positivity_floor > 0.0 && Model::conservative_vars().index_of(VariableRole::Density) < 0)
    throw std::invalid_argument("compiled AMR positivity requires a conservative Density variable");

  CompiledAmrSystemBlockRoutes routes{limiter,
                                      riemann,
                                      reconstruction,
                                      time,
                                      static_cast<Real>(positivity_floor),
                                      static_cast<Real>(weno_epsilon),
                                      wave_speed_cache};
  compiled_amr_detail::validate_routes(routes);
  return prepare_generated_amr_system_block(CompiledAmrSystemBlockPreparation<Dim, Model>{
      name, provider_consumer_qid, std::move(model), std::move(routes), gamma, substeps, stride,
      newton, newton_diagnostics});
}

/// Stage the same default block/state identity Python add_equation installs when pops.bind is
/// skipped. Callers that already installed a unique route keep that identity.
template <int Dim>
void ensure_compiled_amr_state_route(AmrSystem<Dim>& system, const std::string& name) {
  try {
    system.install_block_state_route(name, "pops.runtime.package." + name + "/state");
  } catch (const std::runtime_error& error) {
    const std::string_view message = error.what();
    if (message.find("unique") == std::string_view::npos &&
        message.find("duplicate") == std::string_view::npos)
      throw;
  } catch (const std::logic_error& error) {
    const std::string_view message = error.what();
    if (message.find("must be installed before") == std::string_view::npos)
      throw;
  }
}

/// Publish one complete generated package through the facade's atomic exact-ranked seam.
template <int Dim>
void install_prepared_amr_block(AmrSystem<Dim>& system, PreparedAmrSystemBlock<Dim> prepared) {
  ensure_compiled_amr_state_route(system, prepared.name);
  system.install_prepared_amr_block(std::move(prepared));
}

/// Convenience composition used by generated AMR native loaders.
template <int Dim, class Model>
void add_compiled_model(
    AmrSystem<Dim>& system, const std::string& name, Model model,
    const std::string& limiter = "minmod", const std::string& riemann = "rusanov",
    const std::string& reconstruction = "conservative", const std::string& time = "explicit",
    double gamma = static_cast<double>(kPhysicalDefaultGamma), int substeps = 1, int stride = 1,
    const std::vector<std::string>& implicit_vars = {},
    const std::vector<std::string>& implicit_roles = {}, double positivity_floor = 0.0,
    double weno_epsilon = static_cast<double>(kWenoEpsilon), bool wave_speed_cache = false,
    const std::string& provider_consumer_qid = {}) {
  if (!implicit_vars.empty() || !implicit_roles.empty())
    throw std::invalid_argument(
        "compiled AMR block has no prepared exact-ranked partial-implicit provider");
  install_prepared_amr_block(
      system, prepare_compiled_amr_system_block<Dim>(
                  name, std::move(model), limiter, riemann, reconstruction, time, gamma, substeps,
                  stride, positivity_floor, weno_epsilon, wave_speed_cache, provider_consumer_qid));
}

}  // namespace pops
