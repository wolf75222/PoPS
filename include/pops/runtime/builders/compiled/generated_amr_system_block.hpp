/// @file
/// @brief Final exact-ranked generated-package preparation for adaptive Cartesian blocks.

#pragma once

#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>
#include <pops/numerics/spatial/embedded_boundary/operator.hpp>
#include <pops/numerics/spatial/operators/cartesian_operator.hpp>
#include <pops/numerics/spatial/operators/masked_operator.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr/prepared_amr_ghost_fill.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

/// Authenticated full-domain ghost population used only by the root level. Sparse fine levels use
/// runtime::amr::PreparedAmrGhostFill, which combines parent interpolation and same-level exchange.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
using PreparedRootAmrGhostFill = PreparedProvider<void(
    MultiFab<Dim, MemorySpace>&, const runtime::multiblock::BoundaryEvaluationPoint&)>;

/// Provider groups are independently allocated by resolved storage address, so AMR ghost filling
/// operates on the complete group carrier rather than pretending it is one component slab.
template <int Dim>
using PreparedProviderGroupsGhostFill = PreparedProvider<void(
    runtime::system::AuxiliaryStorageGroups<Dim>&,
    const runtime::multiblock::BoundaryEvaluationPoint&)>;

/// Every value required to materialize one generated block on one live hierarchy level.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct GeneratedAmrLevelContext {
  static_assert(Dim >= 1 && Dim <= 3,
                "GeneratedAmrLevelContext only supports dimensions 1, 2, and 3");

  std::size_t level = 0;
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
  std::shared_ptr<const PreparedHyperbolicBoundary<Dim>> physical_boundary;
  std::shared_ptr<const runtime::system::PreparedEmbeddedBoundaryGeometry<Dim>> embedded_boundary;
  std::string state_identity;
  std::string provider_storage_identity;
  std::string boundary_identity;
  std::string embedded_boundary_provider_identity;
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

template <int Dim, class MemorySpace>
void require_level_context(const runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime,
                           const GeneratedAmrLevelContext<Dim, MemorySpace>& context,
                           int state_components, int provider_components,
                           const Extent<Dim>& required_ghosts,
                           std::string_view staircase_provider_identity,
                           std::string_view cut_cell_provider_identity) {
  if (context.level >= runtime.hierarchy().num_levels())
    throw std::out_of_range("generated AMR block level lies outside the live hierarchy");
  if (context.state_identity.empty() || context.provider_storage_identity.empty())
    throw std::invalid_argument("generated AMR block requires exact state and provider identities");
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
      throw std::invalid_argument("provider-free generated AMR block cannot retain provider ghosts");
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

  const auto& state = runtime.hierarchy().state(context.level);
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
    if (provider_components > 0)
    if (context.embedded_boundary) {
      const auto require_embedded_field = [&](const MultiFab<Dim, MemorySpace>& field,
                                              int components, const char* label) {
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
    }
  }
}

template <int Dim, class MemorySpace>
std::string level_contract(const runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime,
                           const GeneratedAmrLevelContext<Dim, MemorySpace>& context,
                           std::string_view provider_identity) {
  ExactContractBuilder contract;
  contract.text("pops.generated-amr-level-block")
      .scalar(std::uint32_t{1})
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
      .optional_collective_contract(context.root_provider_ghost_fill);
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
void collective_phase(Operation&& operation, const char* failure_message) {
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    std::forward<Operation>(operation)();
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_failure) != 0) {
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
  return generated_system_detail::materialize_masked_source<Dim>(
      model, state, provider_storage, provider_plan, embedded);
}

template <int Dim, class Model>
Real maximum_speed(const Model& model, const MultiFab<Dim>& state,
                   const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
                   const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* provider_plan) {
  return generated_system_detail::maximum_speed<Dim>(model, state, provider_storage,
                                                      provider_plan);
}

template <int Dim, class Model>
void add_poisson_rhs(const Model& model, const MultiFab<Dim>& state, MultiFab<Dim>& rhs) {
  collective_phase(
      [&] {
        if (rhs.ncomp() != 1)
          throw std::invalid_argument(
              "generated AMR Poisson RHS destination must have one component");
        generated_system_detail::require_same_layout(state, rhs, 1, "generated AMR Poisson RHS");
      },
      "generated AMR Poisson RHS preflight failed collectively");
  std::optional<MultiFab<Dim>> candidate;
  std::optional<MultiFab<Dim>> status;
  std::optional<MultiFab<Dim>> updated;
  collective_phase(
      [&] {
        candidate.emplace(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
        status.emplace(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
        updated.emplace(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
      },
      "generated AMR Poisson RHS workspace allocation failed collectively");
  Real local_status = Real(0);
  collective_phase(
      [&] {
        for (std::size_t local = 0; local < state.local_size(); ++local)
          for_each_cell(state.box(local),
                        generated_system_detail::MaterializePoissonRhs<Dim, Model>{
                            model, state.fab(local).view(), candidate->fab(local).view(),
                            status->fab(local).view()});
        local_status = reduce_max_local(*status);
      },
      "generated AMR Poisson RHS materialization failed collectively");
  if (all_reduce_max(local_status) != Real(0))
    throw std::runtime_error("generated AMR Poisson RHS produced a non-finite value");
  collective_phase(
      [&] {
        generated_system_detail::copy_valid(rhs, *updated);
        saxpy(*updated, Real(1), *candidate);
      },
      "generated AMR Poisson RHS candidate accumulation failed collectively");
  static_assert(std::is_nothrow_move_assignable_v<MultiFab<Dim>>);
  rhs = std::move(*updated);
}

template <int Dim, class Model>
Real source_frequency(const Model& model, const MultiFab<Dim>& state,
                      const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
                      const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* provider_plan) {
  constexpr int provider_count = provider_count_for<Model, Dim>();
  MultiFab<Dim> values(state.layout(), state.distribution(), state.local_rank(), 1, state.ghosts());
  for (std::size_t local = 0; local < state.local_size(); ++local)
    for_each_cell(state.box(local), generated_system_detail::MaterializeSourceFrequency<Dim, Model>{
                                        model, state.fab(local).view(),
                                        runtime::system::bind_provider_storage_view<Dim, provider_count>(
                                            provider_plan, provider_storage, local),
                                        values.fab(local).view()});
  const Real frequency = reduce_max(values);
  if (!std::isfinite(frequency) || frequency < Real(0))
    throw std::runtime_error("generated AMR source frequency is invalid");
  return frequency;
}

template <int Dim, class Model>
Real stability_dt(const Model& model, const MultiFab<Dim>& state,
                  const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
                  const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* provider_plan) {
  constexpr int provider_count = provider_count_for<Model, Dim>();
  MultiFab<Dim> values(state.layout(), state.distribution(), state.local_rank(), 1, state.ghosts());
  for (std::size_t local = 0; local < state.local_size(); ++local)
    for_each_cell(state.box(local), generated_system_detail::MaterializeStabilityDt<Dim, Model>{
                                        model, state.fab(local).view(),
                                        runtime::system::bind_provider_storage_view<Dim, provider_count>(
                                            provider_plan, provider_storage, local),
                                        values.fab(local).view()});
  const Real dt = reduce_min(values);
  if (!std::isfinite(dt) || !(dt > Real(0)))
    throw std::runtime_error("generated AMR stability dt is invalid");
  return dt;
}

}  // namespace generated_amr_detail

/// One exact generated spatial specialization bound to a live AMR level generation.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedGeneratedAmrLevelBlock {
 public:
  using runtime_type = runtime::amr::AmrRuntime<Dim, MemorySpace>;
  using field_type = MultiFab<Dim, MemorySpace>;
  using evaluation_type = PreparedAmrLevelEvaluation<Dim, MemorySpace>;
  using point_type = runtime::multiblock::BoundaryEvaluationPoint;
  using StatePreparation = std::function<void(const point_type&, field_type&)>;
  using Evaluator = std::function<evaluation_type(const point_type&, field_type&)>;
  using Speed = std::function<Real(const field_type&)>;
  using PoissonRhs = std::function<void(const field_type&, field_type&)>;

  PreparedGeneratedAmrLevelBlock(runtime_type& runtime, std::size_t level,
                                 std::string state_identity, std::string provider_identity,
                                 std::string collective_contract, StatePreparation prepare_state,
                                 Evaluator evaluator, Speed maximum_speed, PoissonRhs poisson_rhs,
                                 Speed source_frequency, Speed stability_dt)
      : runtime_(&runtime),
        level_(level),
        state_identity_(std::move(state_identity)),
        provider_identity_(std::move(provider_identity)),
        collective_contract_(std::move(collective_contract)),
        prepare_state_(std::move(prepare_state)),
        evaluator_(std::move(evaluator)),
        maximum_speed_(std::move(maximum_speed)),
        poisson_rhs_(std::move(poisson_rhs)),
        source_frequency_(std::move(source_frequency)),
        stability_dt_(std::move(stability_dt)),
        topology_epoch_(runtime.topology_epoch()),
        materialization_generation_(runtime.materialization_generation()) {
    if (level_ >= runtime.hierarchy().num_levels() || state_identity_.empty() ||
        provider_identity_.empty() || collective_contract_.empty() || !prepare_state_ ||
        !evaluator_ || !maximum_speed_ || !poisson_rhs_)
      throw std::invalid_argument("generated AMR level block preparation is incomplete");
  }

  static constexpr int dimension = Dim;
  std::size_t level() const noexcept { return level_; }
  std::string_view state_identity() const noexcept { return state_identity_; }
  std::string_view provider_identity() const noexcept { return provider_identity_; }
  std::string_view collective_contract() const noexcept { return collective_contract_; }

  void prepare(const point_type& point, field_type& state) const {
    require_live_();
    require_state_(point, state);
    require_bound_state_(state);
    prepare_state_(point, state);
  }

  evaluation_type evaluate(const point_type& point, field_type& state) const {
    require_live_();
    require_state_(point, state);
    require_bound_state_(state);
    evaluation_type evaluation = evaluator_(point, state);
    evaluation.point = point;
    evaluation.spatial_contract.assign(runtime_->spatial_contract());
    evaluation.topology_epoch = topology_epoch_;
    evaluation.materialization_generation = materialization_generation_;
    return evaluation;
  }

  evaluation_type evaluate(const point_type& point) const {
    return evaluate(point, runtime_->hierarchy().state(level_));
  }

  Real maximum_speed(const field_type& state) const {
    require_live_();
    require_state_contract_(state);
    return maximum_speed_(state);
  }

  Real maximum_speed() const { return maximum_speed(runtime_->hierarchy().state(level_)); }

  void add_poisson_rhs(field_type& rhs) const {
    require_live_();
    poisson_rhs_(runtime_->hierarchy().state(level_), rhs);
  }

  void add_poisson_rhs(const field_type& state, field_type& rhs) const {
    require_live_();
    require_state_contract_(state);
    poisson_rhs_(state, rhs);
  }

  std::optional<Real> source_frequency() const {
    require_live_();
    if (!source_frequency_)
      return std::nullopt;
    return source_frequency_(runtime_->hierarchy().state(level_));
  }

  std::optional<Real> stability_dt() const {
    require_live_();
    if (!stability_dt_)
      return std::nullopt;
    return stability_dt_(runtime_->hierarchy().state(level_));
  }

 private:
  void require_state_(const point_type& point, const field_type& state) const {
    if (point.level != static_cast<int>(level_))
      throw std::invalid_argument("generated AMR residual point targets another hierarchy level");
    require_state_contract_(state);
  }

  void require_state_contract_(const field_type& state) const {
    const field_type& live = runtime_->hierarchy().state(level_);
    if (state.layout() != live.layout() || state.distribution() != live.distribution() ||
        state.local_rank() != live.local_rank() || state.local_size() != live.local_size() ||
        state.ncomp() != live.ncomp() || state.ghosts() != live.ghosts())
      throw std::invalid_argument(
          "generated AMR stage state differs from its exact live level contract");
  }

  void require_bound_state_(const field_type& state) const {
    if (&state != &runtime_->hierarchy().state(level_))
      throw std::invalid_argument(
          "generated AMR ghost providers require their exact bound live level; stage candidates "
          "must enter through AmrSystem's transactional evaluation route");
  }

  void require_live_() const {
    if (runtime_ == nullptr || level_ >= runtime_->hierarchy().num_levels() ||
        topology_epoch_ != runtime_->topology_epoch() ||
        materialization_generation_ != runtime_->materialization_generation())
      throw std::invalid_argument(
          "generated AMR level block is stale after a hierarchy topology mutation");
  }

  runtime_type* runtime_ = nullptr;
  std::size_t level_ = 0;
  std::string state_identity_;
  std::string provider_identity_;
  std::string collective_contract_;
  StatePreparation prepare_state_;
  Evaluator evaluator_;
  Speed maximum_speed_;
  PoissonRhs poisson_rhs_;
  Speed source_frequency_;
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
  using context_type = GeneratedAmrLevelContext<Dim, MemorySpace>;
  using level_block_type = PreparedGeneratedAmrLevelBlock<Dim, MemorySpace>;
  using LevelMaterializer = std::function<level_block_type(runtime_type&, context_type)>;

  std::string name;
  std::string provider_identity;
  std::string provider_consumer_qid;
  std::string staircase_provider_identity;
  std::string cut_cell_provider_identity;
  std::string collective_contract;
  int ncomp = 0;
  int provider_components = 0;
  VariableSet conservative_variables{};
  VariableSet primitive_variables{};
  double gamma = 1.0;
  Extent<Dim> ghosts{};
  int substeps = 1;
  int stride = 1;
  std::string time_route;
  LevelMaterializer materialize_level;
  std::function<void(const double*, double*)> primitive_to_conservative;
  std::function<RecoveryReport(const double*, double*)> conservative_to_primitive;
  UniformCellRecovery batch_conservative_to_primitive;

  level_block_type prepare_level(runtime_type& runtime, context_type context) const {
    if (!materialize_level)
      throw std::logic_error("prepared AMR system block has no level materializer");
    return materialize_level(runtime, std::move(context));
  }
};

namespace generated_amr_detail {

template <int Dim, nd::ReconstructionVariables Variables, class Model, class Metric,
          class Reconstruction, class Numerical, class MemorySpace, int ProviderCount>
void materialize_masked_patch(const Model& model, const Metric& metric,
                              const Reconstruction& reconstruction, const Numerical& numerical,
                              Real positivity_floor, const Fab<Dim, MemorySpace>& state,
                              const ProviderStorageView<Dim, ProviderCount>& providers,
                              const Fab<Dim, MemorySpace>& active,
                              nd::FaceField<Dim, MemorySpace>& faces,
                              Fab<Dim, MemorySpace>& residual) {
  const int positivity_component =
      nd::cartesian_operator_detail::resolve_positivity_component<Model>(positivity_floor);
  nd::FaceField<Dim, MemorySpace> face_statuses(state.box(), 1);
  nd::masked_operator_detail::materialize_axes<0, Variables>(
      model, metric, reconstruction, numerical, positivity_floor, positivity_component, state,
      active, providers, faces, face_statuses, {});
  const Real face_failure = nd::cartesian_operator_detail::maximum_face_status<0>(face_statuses);
  if (face_failure != static_cast<Real>(nd::FiniteVolumeStatus::Success))
    throw std::runtime_error("generated AMR embedded face evaluation refused publication");

  Fab<Dim, MemorySpace> candidate(state.box(), Model::n_vars);
  Fab<Dim, MemorySpace> cell_statuses(state.box(), 1);
  for_each_cell(state.box(),
                nd::masked_operator_detail::MaterializeMaskedResidual<Dim, Metric, Model::n_vars>{
                    metric, static_cast<const nd::FaceField<Dim, MemorySpace>&>(faces).view(),
                    active.view(), candidate.view(), cell_statuses.view()});
  const Real cell_failure = for_each_cell_reduce_max(
      state.box(), nd::cartesian_operator_detail::FieldStatusMaximum<Dim>{
                       static_cast<const Fab<Dim, MemorySpace>&>(cell_statuses).view()});
  if (cell_failure != static_cast<Real>(nd::FiniteVolumeStatus::Success))
    throw std::runtime_error("generated AMR embedded residual refused publication");
  for_each_cell(state.box(), nd::cartesian_operator_detail::CopyCellField<Dim>{
                                 static_cast<const Fab<Dim, MemorySpace>&>(candidate).view(),
                                 residual.view(), Model::n_vars});
}

template <int Dim, nd::ReconstructionVariables Variables, class Model, class Spatial,
          class Reconstruction, class Numerical, class MemorySpace, int ProviderCount>
void materialize_cut_cell_patch(
    const Model& model, const Spatial& spatial, const Reconstruction& reconstruction,
    const Numerical& numerical, Real positivity_floor, const Fab<Dim, MemorySpace>& state,
    const ProviderStorageView<Dim, ProviderCount>& providers,
    const runtime::system::PreparedEmbeddedBoundaryGeometry<Dim>& embedded, std::size_t local,
    nd::FaceField<Dim, MemorySpace>& faces, Fab<Dim, MemorySpace>& residual) {
  const auto metric = nd::PreparedEmbeddedBoundaryMetric<
      Dim, std::remove_cvref_t<decltype(spatial.metric())>>::prepare(
      spatial.metric(), embedded.inverse_volume_fraction().fab(local), state.box());
  materialize_masked_patch<Dim, Variables>(model, metric, reconstruction, numerical,
                                           positivity_floor, state, providers,
                                           embedded.active_mask().fab(local), faces, residual);
}

template <int Dim, class Model, class Reconstruction, class Numerical,
          nd::ReconstructionVariables Variables, class Request>
PreparedAmrSystemBlock<Dim> materialize_system(Request request, Reconstruction reconstruction,
                                               Numerical numerical) {
  static_assert(Model::dimension == Dim);
  if (request.provider_consumer_qid.empty())
    throw std::invalid_argument(
        "generated AMR block requires one explicit provider consumer qid");
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
  result.conservative_variables = Model::conservative_vars();
  result.primitive_variables = Model::primitive_vars();
  result.gamma = request.gamma;
  result.ghosts = required_ghosts;
  result.substeps = request.substeps;
  result.stride = request.stride;
  result.time_route = request.routes.time;

  ExactContractBuilder package_contract;
  package_contract.text("pops.prepared-generated-amr-system-block")
      .scalar(std::uint32_t{3})
      .scalar(std::int32_t{Dim})
      .text(name)
      .text(provider_identity)
      .text(request.provider_consumer_qid)
      .text(staircase_provider_identity)
      .text(cut_cell_provider_identity)
      .scalar(std::int32_t{Model::n_vars})
      .scalar(std::int32_t{provider_count})
      .scalar(request.gamma)
      .scalar(std::int32_t{request.substeps})
      .scalar(std::int32_t{request.stride})
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
                              cut_cell_provider_identity](runtime::amr::AmrRuntime<Dim>& runtime,
                                                          GeneratedAmrLevelContext<Dim> context) {
    require_level_context(runtime, context, Model::n_vars, provider_count,
                          required_ghosts, staircase_provider_identity, cut_cell_provider_identity);
    const auto spatial = spatial_factory(context.geometry);
    runtime::system::AuxiliaryStorageGroups<Dim>* const provider_storage =
        context.provider_storage;
    const auto* const provider_plan = context.provider_plan;
    const auto state_ghost_fill = context.state_ghost_fill;
    const auto provider_ghost_fill = context.provider_ghost_fill;
    const auto root_state_ghost_fill = context.root_state_ghost_fill;
    const auto root_provider_ghost_fill = context.root_provider_ghost_fill;
    const auto physical_boundary = context.physical_boundary;
    const auto embedded_boundary = context.embedded_boundary;
    const Geometry<Dim> geometry = context.geometry;
    const std::size_t level = context.level;

    auto prepare_state = [provider_storage, state_ghost_fill, provider_ghost_fill,
                          root_state_ghost_fill, root_provider_ghost_fill, physical_boundary, geometry,
                          level](const runtime::multiblock::BoundaryEvaluationPoint& point,
                                 MultiFab<Dim>& state) {
      collective_phase(
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
            if (physical_boundary)
              physical_boundary->fill_physical(state, geometry);
          },
          "generated AMR ghost/boundary phase failed collectively");
    };

    auto evaluator = [model, spatial, reconstruction, numerical, positivity_floor, provider_storage,
                      provider_plan,
                      prepare_state, physical_boundary, embedded_boundary,
                      geometry](const runtime::multiblock::BoundaryEvaluationPoint& point,
                                MultiFab<Dim>& state) {
      prepare_state(point, state);
      std::optional<std::vector<nd::FaceField<Dim>>> faces;
      std::optional<MultiFab<Dim>> residual;
      collective_phase(
          [&] {
            faces.emplace(nd::make_face_flux_workspace(state));
            residual.emplace(state.layout(), state.distribution(), state.local_rank(),
                             Model::n_vars, state.ghosts());
          },
          "generated AMR residual workspace allocation failed collectively");
      collective_phase(
          [&] {
            if (embedded_boundary && embedded_boundary->mode() !=
                                         runtime::system::PreparedEmbeddedBoundaryMode::inactive) {
              for (std::size_t local = 0; local < state.local_size(); ++local) {
                if (embedded_boundary->mode() ==
                    runtime::system::PreparedEmbeddedBoundaryMode::staircase) {
                  materialize_masked_patch<Dim, Variables>(
                      model, spatial.metric(), reconstruction, numerical, positivity_floor,
                      state.fab(local),
                      runtime::system::bind_provider_storage_view<Dim, provider_count>(
                          provider_plan, provider_storage, local),
                      embedded_boundary->active_mask().fab(local),
                      (*faces)[local], residual->fab(local));
                } else {
                  materialize_cut_cell_patch<Dim, Variables>(
                      model, spatial, reconstruction, numerical, positivity_floor, state.fab(local),
                      runtime::system::bind_provider_storage_view<Dim, provider_count>(
                          provider_plan, provider_storage, local),
                      *embedded_boundary, local, (*faces)[local],
                      residual->fab(local));
                }
              }
            } else {
              for (std::size_t local = 0; local < state.local_size(); ++local) {
                if constexpr (provider_count == 0)
                  spatial.materialize_face_fluxes(state.fab(local), (*faces)[local]);
                else
                  spatial.materialize_face_fluxes(
                      state.fab(local),
                      runtime::system::bind_provider_storage_view<Dim, provider_count>(
                          provider_plan, provider_storage, local),
                      (*faces)[local]);
                if (physical_boundary)
                  physical_boundary->apply_physical_flux_conditions((*faces)[local],
                                                                    geometry.domain());
              }
              spatial.assemble_residual_from_face_fluxes(*faces, *residual);
            }
          },
          "generated AMR flux/residual materialization failed collectively");
      MultiFab<Dim> source =
          embedded_boundary && embedded_boundary->mode() !=
                                   runtime::system::PreparedEmbeddedBoundaryMode::inactive
              ? materialize_masked_source<Dim>(model, state, provider_storage, provider_plan,
                                               *embedded_boundary)
              : materialize_source<Dim>(model, state, provider_storage, provider_plan);
      saxpy(*residual, Real(1), source);
      return PreparedAmrLevelEvaluation<Dim>{.residual = std::move(*residual),
                                             .integrated_face_fluxes = std::move(*faces)};
    };
    auto speed = [model, provider_storage, provider_plan](const MultiFab<Dim>& state) {
      return maximum_speed<Dim>(model, state, provider_storage, provider_plan);
    };
    auto poisson_rhs = [model](const MultiFab<Dim>& state, MultiFab<Dim>& rhs) {
      add_poisson_rhs<Dim>(model, state, rhs);
    };
    typename PreparedGeneratedAmrLevelBlock<Dim>::Speed source_frequency_bound;
    if constexpr (requires(const Model& value, const typename Model::State& state,
                           const ProviderValues<provider_count>& providers) {
                    value.source_frequency(state, providers);
                  }) {
      source_frequency_bound = [model, provider_storage, provider_plan](const MultiFab<Dim>& state) {
        return source_frequency<Dim>(model, state, provider_storage, provider_plan);
      };
    }
    typename PreparedGeneratedAmrLevelBlock<Dim>::Speed stability_dt_bound;
    if constexpr (requires(const Model& value, const typename Model::State& state,
                           const ProviderValues<provider_count>& providers) {
                    value.stability_dt(state, providers);
                  }) {
      stability_dt_bound = [model, provider_storage, provider_plan](const MultiFab<Dim>& state) {
        return stability_dt<Dim>(model, state, provider_storage, provider_plan);
      };
    }
    std::string contract = level_contract(runtime, context, provider_identity);
    return PreparedGeneratedAmrLevelBlock<Dim>(
        runtime, level, std::move(context.state_identity), provider_identity, std::move(contract),
        std::move(prepare_state), std::move(evaluator), std::move(speed), std::move(poisson_rhs),
        std::move(source_frequency_bound), std::move(stability_dt_bound));
  };

  result.primitive_to_conservative = [model](const double* primitive, double* conservative) {
    generated_system_detail::publish_conservative_state(model, primitive, conservative);
  };
  const auto recovery_plan = prepare_model_variable_recovery(model);
  result.conservative_to_primitive = [recovery_plan](const double* conservative,
                                                     double* primitive) {
    Real input[Model::n_vars]{};
    Real initial[Model::n_vars]{};
    for (int component = 0; component < Model::n_vars; ++component)
      input[component] = static_cast<Real>(conservative[component]);
    const auto outcome = recover_prepared_variable(recovery_plan, input, initial);
    if (outcome.publication_permitted())
      for (int component = 0; component < Model::n_vars; ++component)
        primitive[component] = static_cast<double>(outcome.value[component]);
    return recovery_report(outcome);
  };
  result.batch_conservative_to_primitive = make_uniform_recovery_consumer(model);
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
    const std::string& provider_consumer_qid = {}) {
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
    throw std::invalid_argument(
        "compiled AMR block requires one explicit provider consumer qid");
  if (!std::isfinite(gamma) || !(gamma > 0.0))
    throw std::invalid_argument("compiled AMR block gamma must be finite and positive");
  if (substeps < 1 || stride < 1)
    throw std::invalid_argument("compiled AMR block substeps and stride must be positive");
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
      name, provider_consumer_qid, std::move(model), std::move(routes), gamma, substeps, stride});
}

/// Publish one complete generated package through the facade's atomic exact-ranked seam.
template <int Dim>
void install_prepared_amr_block(AmrSystem<Dim>& system, PreparedAmrSystemBlock<Dim> prepared) {
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
