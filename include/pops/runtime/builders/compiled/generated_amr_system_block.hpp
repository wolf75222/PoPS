/// @file
/// @brief Final exact-ranked generated-package preparation for adaptive Cartesian blocks.

#pragma once

#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>
#include <pops/numerics/spatial/operators/cartesian_operator.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

/// One authenticated ghost-population operation prepared by the hierarchy owner.
///
/// Sparse fine levels require both same-level exchange and coarse/fine interpolation.  A Uniform
/// HaloSchedule cannot prove that combined operation because a fine patch set deliberately does
/// not tile the full physical domain.  The generated numerical package therefore consumes this
/// exact provider instead of silently replaying a Uniform or local-only schedule.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
using PreparedAmrGhostFill = PreparedProvider<void(
    MultiFab<Dim, MemorySpace>&, const runtime::multiblock::BoundaryEvaluationPoint&)>;

/// Every value required to materialize one generated block on one live hierarchy level.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct GeneratedAmrLevelContext {
  static_assert(Dim >= 1 && Dim <= 3,
                "GeneratedAmrLevelContext only supports dimensions 1, 2, and 3");

  std::size_t level = 0;
  Geometry<Dim> geometry;
  BoundaryTopology<Dim> topology;
  MultiFab<Dim, MemorySpace>* auxiliary = nullptr;
  PreparedAmrGhostFill<Dim, MemorySpace> state_ghost_fill;
  PreparedAmrGhostFill<Dim, MemorySpace> auxiliary_ghost_fill;
  std::shared_ptr<const PreparedHyperbolicBoundary<Dim>> physical_boundary;
  std::string state_identity;
  std::string auxiliary_identity;
  std::string boundary_identity;
};

/// A transactionally materialized AMR residual and its integrated face fluxes.
///
/// The face fields are retained rather than reconstructed from the cell residual.  The Program
/// reflux primitive can therefore authenticate and accumulate exactly the fluxes that produced
/// this candidate update.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct PreparedAmrLevelEvaluation {
  MultiFab<Dim, MemorySpace> residual;
  std::vector<nd::FaceField<Dim, MemorySpace>> integrated_face_fluxes;
};

namespace generated_amr_detail {

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

template <int Dim, class MemorySpace>
void require_level_context(
    const runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime,
    const GeneratedAmrLevelContext<Dim, MemorySpace>& context, int state_components,
    int auxiliary_components, const Extent<Dim>& required_ghosts) {
  if (context.level >= runtime.hierarchy().num_levels())
    throw std::out_of_range("generated AMR block level lies outside the live hierarchy");
  if (context.state_identity.empty() || context.auxiliary_identity.empty())
    throw std::invalid_argument(
        "generated AMR block requires exact state and auxiliary identities");
  if (context.geometry.domain() != runtime.hierarchy().layout(context.level).domain())
    throw std::invalid_argument(
        "generated AMR level geometry differs from the live hierarchy domain");
  if (context.auxiliary == nullptr)
    throw std::invalid_argument("generated AMR block requires a level auxiliary owner");
  if (!context.state_ghost_fill || !context.auxiliary_ghost_fill)
    throw std::invalid_argument(
        "generated AMR block requires prepared state and auxiliary ghost providers");
  if (has_physical_faces(context.topology) &&
      (!context.physical_boundary || context.boundary_identity.empty()))
    throw std::invalid_argument(
        "generated AMR block requires an authenticated physical-boundary provider");

  const auto& state = runtime.hierarchy().state(context.level);
  const auto& auxiliary = *context.auxiliary;
  if (state.ncomp() != state_components || auxiliary.ncomp() < auxiliary_components ||
      state.layout() != auxiliary.layout() || state.distribution() != auxiliary.distribution() ||
      state.local_rank() != auxiliary.local_rank() || state.local_size() != auxiliary.local_size())
    throw std::invalid_argument(
        "generated AMR state and auxiliary storage do not share one exact level layout");
  for (int axis = 0; axis < Dim; ++axis) {
    if (state.ghosts()[axis] < required_ghosts[axis] ||
        auxiliary.ghosts()[axis] < required_ghosts[axis])
      throw std::invalid_argument(
          "generated AMR level storage is narrower than its reconstruction stencil");
  }
}

template <int Dim, class MemorySpace>
std::string level_contract(
    const runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime,
    const GeneratedAmrLevelContext<Dim, MemorySpace>& context,
    std::string_view provider_identity) {
  ExactContractBuilder contract;
  contract.text("pops.generated-amr-level-block")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .scalar(static_cast<std::uint64_t>(context.level))
      .text(provider_identity)
      .text(context.state_identity)
      .text(context.auxiliary_identity)
      .text(context.boundary_identity)
      .bytes(runtime.spatial_contract())
      .scalar(runtime.topology_epoch())
      .scalar(runtime.materialization_generation())
      .bytes(context.state_ghost_fill.collective_contract())
      .bytes(context.auxiliary_ghost_fill.collective_contract());
  append_geometry_contract(contract, context.geometry, context.topology);
  return std::move(contract).release();
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
  using Evaluator = std::function<evaluation_type(const point_type&, const field_type&)>;
  using Speed = std::function<Real(const field_type&)>;
  using PoissonRhs = std::function<void(const field_type&, field_type&)>;

  PreparedGeneratedAmrLevelBlock(runtime_type& runtime, std::size_t level,
                                 std::string state_identity, std::string provider_identity,
                                 std::string collective_contract, Evaluator evaluator,
                                 Speed maximum_speed, PoissonRhs poisson_rhs)
      : runtime_(&runtime),
        level_(level),
        state_identity_(std::move(state_identity)),
        provider_identity_(std::move(provider_identity)),
        collective_contract_(std::move(collective_contract)),
        evaluator_(std::move(evaluator)),
        maximum_speed_(std::move(maximum_speed)),
        poisson_rhs_(std::move(poisson_rhs)),
        topology_epoch_(runtime.topology_epoch()),
        materialization_generation_(runtime.materialization_generation()) {
    if (level_ >= runtime.hierarchy().num_levels() || state_identity_.empty() ||
        provider_identity_.empty() || collective_contract_.empty() || !evaluator_ ||
        !maximum_speed_ || !poisson_rhs_)
      throw std::invalid_argument("generated AMR level block preparation is incomplete");
  }

  static constexpr int dimension = Dim;
  std::size_t level() const noexcept { return level_; }
  std::string_view state_identity() const noexcept { return state_identity_; }
  std::string_view provider_identity() const noexcept { return provider_identity_; }
  std::string_view collective_contract() const noexcept { return collective_contract_; }

  evaluation_type evaluate(const point_type& point) const {
    require_live_();
    if (point.level != static_cast<int>(level_))
      throw std::invalid_argument(
          "generated AMR residual point targets another hierarchy level");
    return evaluator_(point, runtime_->hierarchy().state(level_));
  }

  Real maximum_speed() const {
    require_live_();
    return maximum_speed_(runtime_->hierarchy().state(level_));
  }

  void add_poisson_rhs(field_type& rhs) const {
    require_live_();
    poisson_rhs_(runtime_->hierarchy().state(level_), rhs);
  }

 private:
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
  Evaluator evaluator_;
  Speed maximum_speed_;
  PoissonRhs poisson_rhs_;
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
  std::string collective_contract;
  int ncomp = 0;
  int aux_components = 0;
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

template <int Dim, class Model, class Reconstruction, class Numerical,
          nd::ReconstructionVariables Variables, class Request>
PreparedAmrSystemBlock<Dim> materialize_system(Request request, Reconstruction reconstruction,
                                               Numerical numerical) {
  static_assert(Model::dimension == Dim);
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
  const std::string name = request.name;
  const std::string provider_identity =
      "pops.generated.amr.cartesian.nd/" + std::to_string(Dim) + "/" +
      request.routes.limiter + "/" + request.routes.riemann + "/" +
      request.routes.reconstruction;

  PreparedAmrSystemBlock<Dim> result;
  result.name = name;
  result.provider_identity = provider_identity;
  result.ncomp = Model::n_vars;
  result.aux_components = aux_comps_for<Model, Dim>();
  result.conservative_variables = Model::conservative_vars();
  result.primitive_variables = Model::primitive_vars();
  result.gamma = request.gamma;
  result.ghosts = required_ghosts;
  result.substeps = request.substeps;
  result.stride = request.stride;
  result.time_route = request.routes.time;

  ExactContractBuilder package_contract;
  package_contract.text("pops.prepared-generated-amr-system-block")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .text(name)
      .text(provider_identity)
      .scalar(std::int32_t{Model::n_vars})
      .scalar(std::int32_t{aux_comps_for<Model, Dim>()})
      .scalar(request.gamma)
      .scalar(std::int32_t{request.substeps})
      .scalar(std::int32_t{request.stride})
      .text(request.routes.time)
      .scalar(static_cast<double>(request.routes.positivity_floor))
      .scalar(static_cast<double>(request.routes.weno_epsilon));
  for (int axis = 0; axis < Dim; ++axis)
    package_contract.scalar(std::int64_t{required_ghosts[axis]});
  result.collective_contract = std::move(package_contract).release();

  result.materialize_level =
      [model, spatial_factory, required_ghosts, provider_identity](
          runtime::amr::AmrRuntime<Dim>& runtime, GeneratedAmrLevelContext<Dim> context) {
        require_level_context(runtime, context, Model::n_vars, aux_comps_for<Model, Dim>(),
                              required_ghosts);
        const auto spatial = spatial_factory(context.geometry);
        MultiFab<Dim>* const auxiliary = context.auxiliary;
        const auto state_ghost_fill = context.state_ghost_fill;
        const auto auxiliary_ghost_fill = context.auxiliary_ghost_fill;
        const auto physical_boundary = context.physical_boundary;
        const Geometry<Dim> geometry = context.geometry;
        const std::size_t level = context.level;

        auto evaluator = [model, spatial, auxiliary, state_ghost_fill, auxiliary_ghost_fill,
                          physical_boundary, geometry](
                             const runtime::multiblock::BoundaryEvaluationPoint& point,
                             const MultiFab<Dim>& live_state) {
          MultiFab<Dim> state(live_state);
          MultiFab<Dim> aux(*auxiliary);
          state_ghost_fill(state, point);
          auxiliary_ghost_fill(aux, point);
          if (physical_boundary)
            physical_boundary->fill_physical(state, geometry);

          auto faces = nd::make_face_flux_workspace(state);
          for (std::size_t local = 0; local < state.local_size(); ++local) {
            spatial.materialize_face_fluxes(state.fab(local), aux.fab(local), faces[local]);
            if (physical_boundary)
              physical_boundary->apply_physical_flux_conditions(faces[local], geometry.domain());
          }

          MultiFab<Dim> residual(state.layout(), state.distribution(), state.local_rank(),
                                 Model::n_vars, state.ghosts());
          spatial.assemble_residual_from_face_fluxes(faces, residual);
          MultiFab<Dim> source =
              generated_system_detail::materialize_source<Dim>(model, state, aux);
          saxpy(residual, Real(1), source);
          return PreparedAmrLevelEvaluation<Dim>{std::move(residual), std::move(faces)};
        };
        auto speed = [model, auxiliary](const MultiFab<Dim>& state) {
          return generated_system_detail::maximum_speed<Dim>(model, state, *auxiliary);
        };
        auto poisson_rhs = [model](const MultiFab<Dim>& state, MultiFab<Dim>& rhs) {
          generated_system_detail::add_poisson_rhs<Dim>(model, state, rhs);
        };
        std::string contract = level_contract(runtime, context, provider_identity);
        return PreparedGeneratedAmrLevelBlock<Dim>(
            runtime, level, std::move(context.state_identity), provider_identity,
            std::move(contract), std::move(evaluator), std::move(speed),
            std::move(poisson_rhs));
      };

  result.primitive_to_conservative = [model](const double* primitive, double* conservative) {
    typename Model::Primitive input{};
    for (int component = 0; component < Model::n_vars; ++component)
      input[component] = static_cast<Real>(primitive[component]);
    const auto converted = model.make_conservative(input);
    for (int component = 0; component < Model::n_vars; ++component)
      conservative[component] = converted.succeeded()
                                    ? static_cast<double>(converted.value[component])
                                    : std::numeric_limits<double>::quiet_NaN();
  };
  const auto recovery_plan = prepare_model_variable_recovery(model);
  result.conservative_to_primitive =
      [recovery_plan](const double* conservative, double* primitive) {
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
      return generated_amr_detail::select_reconstruction<
          Dim, nd::ReconstructionVariables::Conservative>(std::move(request));
    case ReconRouteId::kPrimitive:
      return generated_amr_detail::select_reconstruction<
          Dim, nd::ReconstructionVariables::Primitive>(std::move(request));
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
  Model model;
  CompiledAmrSystemBlockRoutes routes;
  double gamma = 1.0;
  int substeps = 1;
  int stride = 1;
};

namespace compiled_amr_detail {

inline void validate_routes(const CompiledAmrSystemBlockRoutes& routes) {
  if (routes.limiter.empty() || routes.riemann.empty())
    throw std::invalid_argument(
        "compiled AMR block requires explicit limiter and Riemann routes");
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
    const std::string& name, Model model, const std::string& limiter,
    const std::string& riemann, const std::string& reconstruction, const std::string& time,
    double gamma, int substeps, int stride, double positivity_floor = 0.0,
    double weno_epsilon = static_cast<double>(kWenoEpsilon),
    bool wave_speed_cache = false) {
  static_assert(Dim >= 1 && Dim <= 3);
  static_assert(requires { Model::dimension; },
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
  if (!std::isfinite(gamma) || !(gamma > 0.0))
    throw std::invalid_argument("compiled AMR block gamma must be finite and positive");
  if (substeps < 1 || stride < 1)
    throw std::invalid_argument("compiled AMR block substeps and stride must be positive");
  if (positivity_floor > 0.0 &&
      Model::conservative_vars().index_of(VariableRole::Density) < 0)
    throw std::invalid_argument(
        "compiled AMR positivity requires a conservative Density variable");

  CompiledAmrSystemBlockRoutes routes{
      limiter, riemann, reconstruction, time, static_cast<Real>(positivity_floor),
      static_cast<Real>(weno_epsilon), wave_speed_cache};
  compiled_amr_detail::validate_routes(routes);
  return prepare_generated_amr_system_block(
      CompiledAmrSystemBlockPreparation<Dim, Model>{name, std::move(model), std::move(routes),
                                                    gamma, substeps, stride});
}

/// Installation fence for the current facade.
///
/// AmrSystem::set_compiled_block accepts an obsolete deferred builder, discards it, and retains
/// only metadata. Publishing through it would report success without an executable exact block.
template <int Dim>
[[noreturn]] void install_prepared_amr_block(AmrSystem<Dim>&,
                                             PreparedAmrSystemBlock<Dim>) {
  throw std::runtime_error(
      "compiled AMR block preparation succeeded, but AmrSystem<Dim> lacks the atomic "
      "install_prepared_amr_block(PreparedAmrSystemBlock<Dim>) seam required to publish exact "
      "level operators and reflux face fluxes; installation was refused before mutation");
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
    double weno_epsilon = static_cast<double>(kWenoEpsilon), bool wave_speed_cache = false) {
  if (!implicit_vars.empty() || !implicit_roles.empty())
    throw std::invalid_argument(
        "compiled AMR block has no prepared exact-ranked partial-implicit provider");
  install_prepared_amr_block(
      system, prepare_compiled_amr_system_block<Dim>(
                  name, std::move(model), limiter, riemann, reconstruction, time, gamma,
                  substeps, stride, positivity_floor, weno_epsilon, wave_speed_cache));
}

}  // namespace pops
