// Exact-rank MPI proof for simultaneous multi-block AMR field solves.  The test intentionally
// exercises the generated Program route: nullptr entries borrow accepted block state, non-null
// entries are detached candidates, and publication remains private until SolveOutcome acceptance.

#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"
#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/coupling/base/elliptic_rhs.hpp>
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/solve_report_consensus.hpp>
#include <pops/physics/bricks/source.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/amr/exact_field_solver_provider.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/system.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int Dim = pops::kNativeDimension;
using Field = pops::MultiFab<Dim>;
using Context = pops::runtime::program::AmrProgramContext<Dim>;

enum class AmrProviderFault {
  none,
  divergent_report,
  throw_on_rank_one,
  candidate_validation_throw_once_on_rank_one,
};

struct BoundaryBindingAudit {
  inline static const Field* bound_state = nullptr;
  inline static pops::Real bound_state_min = std::numeric_limits<pops::Real>::quiet_NaN();
  inline static int binding_count = 0;

  static void reset() {
    bound_state = nullptr;
    bound_state_min = std::numeric_limits<pops::Real>::quiet_NaN();
    binding_count = 0;
  }

  static void bind(
      const std::shared_ptr<const pops::PreparedFieldBoundaryContextSet<Dim>>& contexts) {
    if (!contexts || contexts->size() == 0)
      throw std::invalid_argument("boundary binding audit requires a prepared context");
    const auto context = contexts->view(0, 0);
    if (context.state_count != 1 || context.states == nullptr || context.states[0] == nullptr)
      throw std::invalid_argument("boundary binding audit requires one state dependency");
    bound_state = context.states[0];
    bound_state_min = pops::reduce_min_local(*bound_state);
    ++binding_count;
  }

  static void add_residual(int, const Field&, Field&, const pops::Geometry<Dim>&,
                           const pops::FieldBoundaryExecutionContext<Dim>&) {}

  static void apply_jvp(int, const Field&, const Field&, Field&, const pops::Geometry<Dim>&,
                        const pops::FieldBoundaryExecutionContext<Dim>&) {}

  static pops::CompiledFieldBoundaryKernel<Dim> kernel() {
    return {"tests.mpi.multiblock-field.boundary-audit",
            "tests.mpi.multiblock-field.boundary-audit.residual",
            "tests.mpi.multiblock-field.boundary-audit.jvp",
            {},
            {},
            &add_residual,
            &apply_jvp,
            false};
  }
};

struct PreparedHierarchyLaneAudit {
  inline static bool observed = false;
  inline static bool owns_communicator = false;
  inline static bool has_qualified_identity = false;
  inline static bool is_distinct_world_duplicate = false;

  static void reset() {
    observed = false;
    owns_communicator = false;
    has_qualified_identity = false;
    is_distinct_world_duplicate = false;
  }

  static void observe(const pops::ExecutionLane& lane) {
    observed = true;
    owns_communicator = lane.owns_communicator();
    constexpr std::string_view hierarchy_lane_marker = "/pops.generated-amr-levels/";
    const std::size_t marker = lane.identity().find(hierarchy_lane_marker);
    has_qualified_identity = marker != std::string_view::npos && marker != 0 &&
                             marker + hierarchy_lane_marker.size() < lane.identity().size();
    int world_relation = MPI_UNEQUAL;
    is_distinct_world_duplicate =
        MPI_Comm_compare(lane.native_handle(), MPI_COMM_WORLD, &world_relation) == MPI_SUCCESS &&
        world_relation == MPI_CONGRUENT;
  }
};

class ConsensusAmrFieldSolver final : public pops::runtime::amr::ExactAmrFieldSolver<Dim> {
 public:
  using request_type = pops::runtime::amr::ExactAmrFieldSolverBuildRequest<Dim>;

  ConsensusAmrFieldSolver(const request_type& request, std::string identity, std::string contract,
                          AmrProviderFault fault)
      : identity_(std::move(identity)), contract_(std::move(contract)), fault_(fault) {
    rhs_.reserve(request.hierarchy.levels.size());
    candidates_.reserve(request.hierarchy.levels.size());
    for (const auto& level : request.hierarchy.levels) {
      rhs_.push_back(std::make_unique<Field>(level.boxes, level.distribution, level.local_rank, 1,
                                             level.rhs_ghosts));
      candidates_.push_back(std::make_unique<Field>(level.boxes, level.distribution,
                                                    level.local_rank, 1, level.phi_ghosts));
      rhs_.back()->set_val(pops::Real(0));
      candidates_.back()->set_val(pops::Real(0));
    }
  }

  std::string_view provider_identity() const noexcept override { return identity_; }
  std::string_view exact_prepared_contract() const noexcept override { return contract_; }
  bool couples_hierarchy_levels() const noexcept override { return false; }
  int level_count() const noexcept override { return static_cast<int>(rhs_.size()); }
  Field& rhs_level(int level) override { return *rhs_.at(static_cast<std::size_t>(level)); }
  Field& candidate_level(int level) override {
    validate_candidate_access_();
    return *candidates_.at(static_cast<std::size_t>(level));
  }
  const Field& candidate_level(int level) const override {
    validate_candidate_access_();
    return *candidates_.at(static_cast<std::size_t>(level));
  }
  void install_newton(pops::FieldNewtonOptions) override {}
  void install_boundary_kernel(pops::CompiledFieldBoundaryKernel<Dim>) override {}
  void set_boundary_contexts(
      std::shared_ptr<const pops::PreparedFieldBoundaryContextSet<Dim>> contexts) override {
    BoundaryBindingAudit::bind(contexts);
    boundary_contexts_ = std::move(contexts);
  }
  void install_nullspace(pops::PreparedFieldNullspace<Dim>,
                         std::vector<pops::PreparedVectorDistribution<Dim>>) override {}
  int maximum_iterations() const noexcept override { return 8; }
  pops::SolveReport solve(const pops::ExecutionLane& lane) override {
    PreparedHierarchyLaneAudit::observe(lane);
    if (fault_ == AmrProviderFault::throw_on_rank_one && pops::my_rank() == 1)
      throw std::runtime_error("intentional rank-local AMR provider solve failure");
    ++solve_generation_;
    candidate_accesses_after_solve_ = 0;
    pops::SolveReport report;
    report.iters = 1;
    report.evaluations = 1;
    report.rel_residual = pops::Real(0);
    report.reference_residual_norm = pops::Real(1);
    report.residual_norm = pops::Real(0);
    report.mark_solved(fault_ == AmrProviderFault::divergent_report && pops::my_rank() == 1
                           ? "rank-one-provider-report"
                           : "collective-provider-report");
    return report;
  }

 private:
  void validate_candidate_access_() const {
    if (solve_generation_ == 0 ||
        fault_ != AmrProviderFault::candidate_validation_throw_once_on_rank_one ||
        pops::my_rank() != 1)
      return;
    ++candidate_accesses_after_solve_;
    if (solve_generation_ == 1 && make_validation_fault_armed_) {
      make_validation_fault_armed_ = false;
      throw std::runtime_error("intentional rank-local AMR outcome validation failure");
    }
    if (solve_generation_ >= 2 && consumption_validation_fault_armed_ &&
        candidate_accesses_after_solve_ == static_cast<int>(candidates_.size()) + 1) {
      consumption_validation_fault_armed_ = false;
      throw std::runtime_error("intentional rank-local AMR consumption validation failure");
    }
  }

  std::string identity_;
  std::string contract_;
  AmrProviderFault fault_ = AmrProviderFault::none;
  mutable int solve_generation_ = 0;
  mutable int candidate_accesses_after_solve_ = 0;
  mutable bool make_validation_fault_armed_ = true;
  mutable bool consumption_validation_fault_armed_ = true;
  std::vector<std::unique_ptr<Field>> rhs_;
  std::vector<std::unique_ptr<Field>> candidates_;
  std::shared_ptr<const pops::PreparedFieldBoundaryContextSet<Dim>> boundary_contexts_;
};

class ConsensusAmrFieldProvider final
    : public pops::runtime::amr::ExactAmrFieldSolverProvider<Dim> {
 public:
  using request_type = pops::runtime::amr::ExactAmrFieldSolverBuildRequest<Dim>;
  using solver_type = pops::runtime::amr::ExactAmrFieldSolver<Dim>;

  ConsensusAmrFieldProvider(std::string identity, AmrProviderFault fault)
      : identity_(std::move(identity)),
        collective_contract_(identity_ + "/collective@1"),
        fault_(fault) {}

  std::string_view identity() const noexcept override { return identity_; }
  std::string_view collective_contract() const noexcept override { return collective_contract_; }
  pops::PreparedProviderSupport supports(const request_type&,
                                         const pops::ExecutionLane&) const noexcept override {
    return pops::PreparedProviderSupport::accept();
  }
  std::string expected_prepared_contract(const request_type& request,
                                         const pops::ExecutionLane& lane) const override {
    return pops::runtime::amr::make_exact_amr_field_solver_contract(identity_, request, lane);
  }
  std::unique_ptr<solver_type> build(const request_type& request,
                                     const pops::ExecutionLane& lane) const override {
    return std::make_unique<ConsensusAmrFieldSolver>(
        request, identity_, expected_prepared_contract(request, lane), fault_);
  }

 private:
  std::string identity_;
  std::string collective_contract_;
  AmrProviderFault fault_ = AmrProviderFault::none;
};

struct RankLocalPublication {
  bool layout_valid = true;
  bool accepted = false;

  static void validate(void* context) {
    if (!static_cast<RankLocalPublication*>(context)->layout_valid)
      throw std::logic_error("rank-local destination layout drift");
  }
  static void accept(void* context) noexcept {
    static_cast<RankLocalPublication*>(context)->accepted = true;
  }
};

struct RankLocalFallibleSource {
  using State = pops::StateVec<1>;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 1;

  pops::ImplicitEvaluationStatus rank_one_status = pops::ImplicitEvaluationStatus::kOk;
  std::uint32_t reason = 0;

  POPS_HD State source(const State& state, const pops::ProviderValues<1>&) const {
    return State{-state[0]};
  }
  POPS_HD pops::ImplicitEvaluationResult evaluate_source(const State& state,
                                                         const pops::ProviderValues<1>& providers,
                                                         State& output) const {
    output = State{-state[0]};
    if (providers[0] <= pops::Real(0.5))
      return pops::ImplicitEvaluationResult::ok();
    switch (rank_one_status) {
      case pops::ImplicitEvaluationStatus::kOk:
        return pops::ImplicitEvaluationResult::ok();
      case pops::ImplicitEvaluationStatus::kRetry:
        return pops::ImplicitEvaluationResult::retry(reason);
      case pops::ImplicitEvaluationStatus::kReject:
        return pops::ImplicitEvaluationResult::reject(reason);
      case pops::ImplicitEvaluationStatus::kFailed:
        return pops::ImplicitEvaluationResult::failed(reason);
      case pops::ImplicitEvaluationStatus::kInvalid:
        return pops::ImplicitEvaluationResult::invalid(reason);
    }
    return pops::ImplicitEvaluationResult::invalid(reason);
  }
};

enum class EllipticFactoryFault {
  throw_on_rank_one,
  null_on_rank_one,
  wrong_components_on_rank_one,
  aliased_fields_on_rank_one,
  wrong_ghosts_on_rank_one,
  wrong_operator_contract_on_rank_one,
  wrong_distribution_on_rank_one,
  inspection_throws_on_rank_one,
};

class ConsensusElliptic {
 public:
  static constexpr int dimension = Dim;
  using field_type = Field;
  using request_type = pops::EllipticBuildRequest<Dim>;

  ConsensusElliptic(request_type request, EllipticFactoryFault fault)
      : geometry_(request.geometry),
        rhs_(request.boxes, materialized_distribution(request, fault), request.local_rank,
             fault == EllipticFactoryFault::wrong_components_on_rank_one && pops::my_rank() == 1
                 ? 2
                 : 1,
             materialized_rhs_ghosts(request, fault)),
        phi_(request.boxes, materialized_distribution(request, fault), request.local_rank, 1,
             request.phi_ghosts),
        alias_fields_(fault == EllipticFactoryFault::aliased_fields_on_rank_one &&
                      pops::my_rank() == 1),
        inspection_throws_(fault == EllipticFactoryFault::inspection_throws_on_rank_one &&
                           pops::my_rank() == 1),
        operator_contract_(pops::make_materialized_elliptic_operator_contract(
            fault == EllipticFactoryFault::wrong_operator_contract_on_rank_one &&
                    pops::my_rank() == 1
                ? pops::EllipticOperatorIdentity{"tests.mpi.elliptic.wrong", 1}
                : operator_identity(),
            geometry_, request.boundary, rhs_, phi_)) {}

  static constexpr pops::EllipticOperatorIdentity operator_identity() noexcept {
    return {"tests.mpi.elliptic.consensus", 1};
  }
  static pops::EllipticOperatorContract expected_operator_contract(const request_type& request) {
    return pops::make_expected_elliptic_operator_contract(operator_identity(), request);
  }
  Field& rhs() {
    if (inspection_throws_)
      throw std::runtime_error("intentional rank-local elliptic inspection failure");
    return rhs_;
  }
  Field& phi() { return alias_fields_ ? rhs_ : phi_; }
  void solve() {}
  pops::Real residual() const { return pops::Real(0); }
  const pops::Geometry<Dim>& geom() const { return geometry_; }
  const pops::EllipticOperatorContract& prepared_operator_contract() const noexcept {
    return operator_contract_;
  }

 private:
  static pops::mesh::Distribution<Dim> materialized_distribution(const request_type& request,
                                                                 EllipticFactoryFault fault) {
    if (fault == EllipticFactoryFault::wrong_distribution_on_rank_one && pops::my_rank() == 1)
      return pops::mesh::Distribution<Dim>::replicated(request.boxes,
                                                       request.distribution.rank_space());
    return request.distribution;
  }
  static pops::Extent<Dim> materialized_rhs_ghosts(const request_type& request,
                                                   EllipticFactoryFault fault) {
    pops::Extent<Dim> ghosts = request.rhs_ghosts;
    if (fault == EllipticFactoryFault::wrong_ghosts_on_rank_one && pops::my_rank() == 1)
      ++ghosts[0];
    return ghosts;
  }

  pops::Geometry<Dim> geometry_;
  Field rhs_;
  Field phi_;
  bool alias_fields_ = false;
  bool inspection_throws_ = false;
  pops::EllipticOperatorContract operator_contract_;
};

struct ConsensusEllipticFactory {
  int* constructions = nullptr;
  EllipticFactoryFault fault = EllipticFactoryFault::throw_on_rank_one;

  std::string_view collective_contract() const noexcept {
    return "tests.mpi.elliptic.consensus-factory@1";
  }
  pops::EllipticOperatorContract expected_operator_contract(
      const ConsensusElliptic::request_type& request) const {
    return ConsensusElliptic::expected_operator_contract(request);
  }
  bool supports(const ConsensusElliptic::request_type&) const noexcept { return true; }
  pops::EllipticFactoryBuildResult<ConsensusElliptic> build(
      ConsensusElliptic::request_type request) const noexcept {
    if (fault == EllipticFactoryFault::null_on_rank_one && pops::my_rank() == 1) {
      ++*constructions;
      return {};
    }
    return pops::capture_local_elliptic_factory_build<ConsensusElliptic>(
        [this, request = std::move(request)]() mutable {
          ++*constructions;
          if (fault == EllipticFactoryFault::throw_on_rank_one && pops::my_rank() == 1)
            throw std::runtime_error("intentional rank-local elliptic factory failure");
          return ConsensusElliptic(std::move(request), fault);
        });
  }
};

static_assert(pops::EllipticFactory<ConsensusEllipticFactory, ConsensusElliptic>);

template <int Rank>
struct Model {
  using Law = pops::nd::ScalarAdvection<Rank>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Rank;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 1;
  Law law;
  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"tests.mpi.multiblock-field.scalar-advection", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Rank; ++axis)
      contract.scalar(law.velocity()[axis]);
  }
  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  POPS_HD pops::nd::StateConversion<Primitive> recover(const State& state) const {
    return law.recover(state);
  }
  POPS_HD pops::nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return law.make_conservative(primitive);
  }
  POPS_HD pops::nd::StateConversionStatus admissibility(const State& state) const {
    return law.admissibility(state);
  }
  template <int Axis>
  POPS_HD State flux(const State& state) const {
    return law.template flux<Axis>(state);
  }
  template <int Axis>
  POPS_HD pops::Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, pops::Real& lower, pops::Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State&, const pops::ProviderValues<1>& providers) const {
    return State{providers[0]};
  }
  POPS_HD pops::Real elliptic_rhs(const State& state) const { return state[0]; }
};

pops::AmrSystemConfig<Dim> exact_config() {
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 2;
  config.distribute_coarse = false;
  config.regrid_every = 0;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 8;
    config.coarse_max_grid[axis] = 4;
    config.lower[axis] = pops::Real(0);
    config.upper[axis] = pops::Real(1);
    config.periodicity[axis] = false;
  }
  return config;
}

std::vector<pops::AmrPatch<Dim>> fine_patches(const pops::AmrSystemConfig<Dim>& config) {
  pops::Index<Dim> first_lo{};
  pops::Index<Dim> first_hi{};
  pops::Index<Dim> second_lo{};
  pops::Index<Dim> second_hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    first_lo[axis] = 2;
    first_hi[axis] = 2 * config.shape[axis] - 3;
    second_lo[axis] = first_lo[axis];
    second_hi[axis] = first_hi[axis];
  }
  first_hi[0] = config.shape[0] - 1;
  second_lo[0] = config.shape[0];
  return {{1, {first_lo, first_hi}}, {1, {second_lo, second_hi}}};
}

std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t count = 1;
  for (int axis = 0; axis < Dim; ++axis)
    count *= static_cast<std::size_t>(shape[axis]);
  return count;
}

std::vector<double> initial_state(const pops::Extent<Dim>& shape, double offset) {
  std::vector<double> state(cell_count(shape));
  for (std::size_t ordinal = 0; ordinal < state.size(); ++ordinal) {
    std::size_t remainder = ordinal;
    double value = offset;
    for (int axis = 0; axis < Dim; ++axis) {
      const int coordinate = static_cast<int>(remainder % static_cast<std::size_t>(shape[axis]));
      remainder /= static_cast<std::size_t>(shape[axis]);
      value += (axis + 1) * (coordinate + 1.0) / static_cast<double>(shape[axis]);
    }
    state[ordinal] = value;
  }
  return state;
}

std::size_t linear_ordinal(const pops::Index<Dim>& index, const pops::Extent<Dim>& shape) {
  std::size_t ordinal = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    ordinal += static_cast<std::size_t>(index[axis]) * stride;
    stride *= static_cast<std::size_t>(shape[axis]);
  }
  return ordinal;
}

std::vector<double> discriminating_tag_state(const pops::Extent<Dim>& shape) {
  std::vector<double> state(cell_count(shape), -0.25);
  for (std::size_t ordinal = 0; ordinal < state.size(); ++ordinal)
    if (ordinal % 3 == 1)
      state[ordinal] = 1.25;
  return state;
}

void add_constant(Field& field, pops::Real value) {
  Field increment(field);
  increment.set_val(value);
  pops::saxpy(field, pops::Real(1), increment);
}

template <int RuntimeDim>
void install_system_runtime_authority(pops::System<RuntimeDim>& system, std::string_view identity) {
  auto lane = std::make_shared<pops::ExecutionLane>(
      pops::ExecutionLane::duplicate_world_collectively(identity));
  system.install_prepared_boundary_execution_lane(std::move(lane));
}

void install_block_boundary(pops::AmrSystem<Dim>& system, const std::string& name) {
  const std::string state_identity = "tests.mpi.multiblock-field/state/" + name;
  std::vector<std::string> face_types(static_cast<std::size_t>(2 * Dim), "foextrap");
  std::vector<std::string> face_identities;
  face_identities.reserve(static_cast<std::size_t>(2 * Dim));
  for (int face = 0; face < 2 * Dim; ++face)
    face_identities.push_back("tests.mpi.multiblock-field/" + name + "/face/" +
                              std::to_string(face));
  system.install_hyperbolic_boundary(name, "tests.mpi.multiblock-field/boundary/" + name + "@1", 1,
                                     face_types,
                                     std::vector<double>(static_cast<std::size_t>(2 * Dim), 0.0),
                                     face_identities, {"Scalar"}, state_identity);
}

void install_block(pops::AmrSystem<Dim>& system, const std::string& name) {
  pops::RealVector<Dim> velocity{};
  velocity[0] = pops::Real(0.25);
  Model<Dim> model{pops::nd::ScalarAdvection<Dim>::prepare(velocity)};
  pops::add_compiled_model<Dim>(system, name, model, "none", "rusanov", "conservative", "explicit",
                                1.4, 1, 1, {}, {}, 0.0, static_cast<double>(pops::kWenoEpsilon),
                                false, "tests.mpi.multiblock-field/physical-flux/" + name);
}

pops::runtime::system::AuxiliaryComponentKey install_field_output(pops::AmrSystem<Dim>& system) {
  using namespace pops::runtime::system;
  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 1;
  const AuxiliaryComponentKey key{"tests.mpi.multiblock-field", "field", "potential", "value"};
  const AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "amr-field",
                                            "scalar"};
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "tests.mpi.multiblock-field/output@1",
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      {{key, contract, shape}},
      {}});
  system.install_auxiliary_consumer_plan(
      AuxiliaryConsumerProviderPlan<Dim>{"a", {{{key, contract, shape}, 0}}});
  system.install_auxiliary_consumer_plan(
      AuxiliaryConsumerProviderPlan<Dim>{"b", {{{key, contract, shape}, 0}}});
  system.seal_auxiliary_providers();
  return key;
}

void install_field_plan(pops::AmrSystem<Dim>& system,
                        const pops::runtime::system::AuxiliaryComponentKey& output) {
  const pops::AmrFieldHierarchyPolicyAuthority hierarchy{
      "pops.field-hierarchy.level-local", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  system.set_field_solver_plan(
      "field/coupled", "tests.mpi.multiblock-field/plan@1", "tests.mpi.multiblock-field/provider@1",
      "tests.mpi.multiblock-field", "a", "potential", {output}, 1,
      {"tests.mpi.multiblock-field/a/rhs", "tests.mpi.multiblock-field/b/rhs"}, {"a", "b"},
      {"potential", "potential"}, {2.0, -0.5}, "geometric_mg", hierarchy,
      pops::geometric_mg_amr_field_solver_options(pops::GeometricMgOptions{},
                                                  pops::CompositeFacOptions{}));
  system.set_field_reaction("field/coupled", 2.0);
  system.register_elliptic_field("a", "potential", {output}, 1);
}

void install_rejecting_field_plan(pops::AmrSystem<Dim>& system) {
  const pops::AmrFieldHierarchyPolicyAuthority hierarchy{
      "pops.field-hierarchy.level-local", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  pops::GeometricMgOptions options;
  options.rel_tol = pops::Real(0);
  options.abs_tol = pops::Real(0);
  options.max_cycles = 1;
  options.nu1 = 1;
  options.nu2 = 1;
  options.nbottom = 1;
  system.set_field_solver_plan(
      "field/reject", "tests.mpi.multiblock-field/reject-plan@1",
      "tests.mpi.multiblock-field/reject-provider@1", "tests.mpi.multiblock-field", "a",
      "reject-potential", {{"tests.mpi.multiblock-field", "field", "reject-potential", "value"}}, 1,
      {"tests.mpi.multiblock-field/a/rhs", "tests.mpi.multiblock-field/b/rhs"}, {"a", "b"},
      {"potential", "potential"}, {2.0, -0.5}, "geometric_mg", hierarchy,
      pops::geometric_mg_amr_field_solver_options(options, pops::CompositeFacOptions{}));
  system.set_field_reaction("field/reject", 2.0);
}

void install_consensus_provider_plan(pops::AmrSystem<Dim>& system, const std::string& slot,
                                     AmrProviderFault fault, bool audit_boundary = false) {
  const std::string identity = "tests.mpi.multiblock-field/provider/" + slot;
  system.register_field_solver_provider(
      std::make_shared<ConsensusAmrFieldProvider>(identity, fault));
  const pops::AmrFieldHierarchyPolicyAuthority hierarchy{
      "pops.field-hierarchy.level-local", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  system.set_field_solver_plan(
      slot, "tests.mpi.multiblock-field/plan/" + slot + "@1", identity,
      "tests.mpi.multiblock-field", "a", "provider-potential",
      {{"tests.mpi.multiblock-field", "field", "provider-potential", slot}}, 1,
      {"tests.mpi.multiblock-field/a/rhs", "tests.mpi.multiblock-field/b/rhs"}, {"a", "b"},
      {"potential", "potential"}, {2.0, -0.5}, identity, hierarchy,
      pops::geometric_mg_amr_field_solver_options(pops::GeometricMgOptions{},
                                                  pops::CompositeFacOptions{}));
  system.set_field_reaction(slot, 2.0);
  if (audit_boundary) {
    system.set_field_boundary_dependencies(slot, {"a"}, {0}, {}, {}, {});
    system.set_field_boundary_kernel(slot, BoundaryBindingAudit::kernel());
  }
}

pops::PreparedProviderOptions system_cartesian_cg_options() {
  return {"pops.system.cartesian-cg-options@1",
          {{"abs_tol", 0.0}, {"max_iterations", std::int64_t{200}}, {"rel_tol", 1.0e-8}}};
}

bool system_registry_bind_rejected(std::string token) {
  pops::SystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 4;
  pops::System<Dim> system(config);
  install_system_runtime_authority(system, "tests.mpi.system-registry/runtime@1");
  system.register_configured_field_solver_provider("cartesian_cg", "field/registry",
                                                   system_cartesian_cg_options());
  system.set_field_solver_plan("field/registry", std::move(token), "provider/registry",
                               "output/registry", "a", "potential", {"rhs/registry"}, {"a"},
                               {"potential"}, {1.0}, "field/registry");
  try {
    system.mark_bound();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

bool amr_registry_bind_rejected(std::string token) {
  pops::AmrSystemConfig<Dim> config = exact_config();
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  config.distribute_coarse = true;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.mpi.amr-registry/runtime");
  system.install_block_state_route("a", "tests.mpi.multiblock-field/state/a");
  install_block_boundary(system, "a");
  install_block(system, "a");
  const pops::AmrFieldHierarchyPolicyAuthority hierarchy{
      "pops.field-hierarchy.level-local", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  system.set_field_solver_plan(
      "field/registry", std::move(token), "provider/registry", "output/registry", "a", "potential",
      {{"output/registry", "field", "potential", "value"}}, 1, {"rhs/registry"}, {"a"},
      {"potential"}, {1.0}, "geometric_mg", hierarchy,
      pops::geometric_mg_amr_field_solver_options(pops::GeometricMgOptions{},
                                                  pops::CompositeFacOptions{}));
  try {
    system.mark_bound();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

double max_difference(const std::vector<double>& left, const std::vector<double>& right) {
  if (left.size() != right.size())
    return std::numeric_limits<double>::infinity();
  double difference = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index)
    difference = std::max(difference, std::abs(left[index] - right[index]));
  return difference;
}

double field_difference_inf(const Field& left, const Field& right) {
  Field difference(left);
  pops::saxpy(difference, pops::Real(-1), right);
  return static_cast<double>(pops::reduce_norm_inf(difference, 0));
}

bool elliptic_factory_rejected(const Field& prototype, const pops::AmrSystemConfig<Dim>& config,
                               const pops::Box<Dim>& domain, EllipticFactoryFault fault,
                               int& constructions) {
  const pops::Geometry<Dim> geometry =
      pops::Geometry<Dim>::from_bounds(domain, config.lower, config.upper);
  std::array<pops::PhysicalBoundaryFace, 2 * Dim> faces{};
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  const pops::PhysicalBoundaryConditions<Dim> boundary(pops::BoundaryTopology<Dim>{}, faces,
                                                       spacing);
  pops::Extent<Dim> rhs_ghosts{};
  pops::Extent<Dim> phi_ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    phi_ghosts[axis] = 1;
  const std::size_t count = prototype.layout().size();
  try {
    (void)pops::make_elliptic_solver<ConsensusElliptic>(
        {geometry,
         prototype.layout(),
         prototype.distribution(),
         prototype.local_rank(),
         boundary,
         rhs_ghosts,
         phi_ghosts,
         {count, count > 1 ? count * (count - 1) / 2 : 0}},
        ConsensusEllipticFactory{&constructions, fault});
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

std::pair<double, double> superposition_error(const std::vector<double>& both,
                                              const std::vector<double>& only_a,
                                              const std::vector<double>& only_b,
                                              const std::vector<double>& base) {
  if (both.size() != only_a.size() || both.size() != only_b.size() || both.size() != base.size())
    return {std::numeric_limits<double>::infinity(), 0.0};
  double error = 0.0;
  double response = 0.0;
  for (std::size_t index = 0; index < both.size(); ++index) {
    const double simultaneous = both[index] - base[index];
    const double separate = only_a[index] - base[index] + only_b[index] - base[index];
    error = std::max(error, std::abs(simultaneous - separate));
    response = std::max(response, std::abs(simultaneous));
  }
  return {error, response};
}

int run_multiblock_field_solve(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  long failures = 0;
  const auto require = [&failures](bool condition, std::string_view label) {
    if (!condition) {
      std::fprintf(stderr, "rank %d: multi-block field check failed: %.*s\n", pops::my_rank(),
                   static_cast<int>(label.size()), label.data());
      ++failures;
    }
  };

  try {
    require(!system_registry_bind_rejected("tests.mpi.registry/shared@1"),
            "System accepts one exact canonical field-plan registry");
    require(system_registry_bind_rejected(pops::my_rank() == 0 ? "tests.mpi.registry/rank-zero@1"
                                                               : "tests.mpi.registry/rank-one@1"),
            "System rejects a rank-divergent field-plan registry");
    require(!amr_registry_bind_rejected("tests.mpi.amr-registry/shared@1"),
            "AmrSystem accepts one exact canonical field-plan registry");
    require(amr_registry_bind_rejected(pops::my_rank() == 0 ? "tests.mpi.amr-registry/rank-zero@1"
                                                            : "tests.mpi.amr-registry/rank-one@1"),
            "AmrSystem rejects a rank-divergent field-plan registry");

    const pops::AmrSystemConfig<Dim> config = exact_config();
    pops::AmrSystem<Dim> system(config);
    pops::test::install_amr_runtime_authority(system, "tests.mpi.multiblock-field/runtime");
    system.install_block_state_route("a", "tests.mpi.multiblock-field/state/a");
    system.install_block_state_route("b", "tests.mpi.multiblock-field/state/b");
    install_block_boundary(system, "a");
    install_block_boundary(system, "b");
    install_block(system, "a");
    install_block(system, "b");
    system.set_poisson("charge_density", "geometric_mg", "dirichlet");
    const auto output = install_field_output(system);
    install_field_plan(system, output);
    install_rejecting_field_plan(system);
    install_consensus_provider_plan(system, "field/provider-positive", AmrProviderFault::none);
    install_consensus_provider_plan(system, "field/provider-divergent",
                                    AmrProviderFault::divergent_report);
    install_consensus_provider_plan(system, "field/provider-throw",
                                    AmrProviderFault::throw_on_rank_one);
    install_consensus_provider_plan(system, "field/provider-validation-fault",
                                    AmrProviderFault::candidate_validation_throw_once_on_rank_one,
                                    true);
    pops::test::install_prepared_threshold_union(system, {{"b", "u", 0.5}},
                                                 "tests.mpi.multiblock-field/tagger-b@1");
    int rhs_calls = 0;
    bool fail_rhs_on_rank_one = false;
    system.set_block_elliptic_field(
        "a", "potential", "tests.mpi.multiblock-field.rhs-a@1",
        [&](const Field& state, Field& rhs) {
          ++rhs_calls;
          if (fail_rhs_on_rank_one && pops::my_rank() == 1)
            throw std::runtime_error("intentional rank-local RHS preparation failure");
          pops::add_scaled_component(state, pops::Real(1), 0, rhs);
        });
    system.set_block_elliptic_field("b", "potential", "tests.mpi.multiblock-field.rhs-b@1",
                                    [&rhs_calls](const Field& state, Field& rhs) {
                                      ++rhs_calls;
                                      pops::add_scaled_component(state, pops::Real(1), 0, rhs);
                                    });
    const std::vector<double> accepted_a(cell_count(config.shape), -2.0);
    const std::vector<double> accepted_b = discriminating_tag_state(config.shape);
    system.set_conservative_state("a", accepted_a);
    system.set_conservative_state("b", accepted_b);
    system.set_program_block_map({0, 1});
    const auto patches = fine_patches(config);
    system.rebuild_hierarchy(patches, {0, 1});
    pops::Extent<Dim> fine_shape{};
    for (int axis = 0; axis < Dim; ++axis)
      fine_shape[axis] = 2 * config.shape[axis];
    system.set_block_level_state("a", 1, initial_state(fine_shape, 2.5));
    system.set_block_level_state("b", 1, initial_state(fine_shape, 3.5));
    const std::vector<double> accepted_fine_a = system.block_level_state_global("a", 1);
    const std::vector<double> accepted_fine_b = system.block_level_state_global("b", 1);

    require(system.n_levels() == 2, "two-level hierarchy is materialized");
    require(system.prepared_amr_block_state(0, 0).distribution().replicated(),
            "coarse level is replicated");
    require(!system.prepared_amr_block_state(0, 1).distribution().replicated(),
            "fine level is partitioned");
    require(system.prepared_amr_block_state(0, 0).local_size() > 0,
            "every rank owns the replicated coarse carrier");
    require(system.prepared_amr_block_state(0, 1).local_size() == 1,
            "every rank owns one fine patch");
    const auto tagging = system.execute_prepared_tagging(0);
    bool exact_tag_pattern = true;
    bool primary_never_crosses = true;
    std::size_t expected_tag_count = 0;
    for (const auto& patch : tagging.refine.patches()) {
      for (std::size_t local = 0; local < patch.tags.size(); ++local) {
        std::size_t remainder = local;
        pops::Index<Dim> index{};
        for (int axis = 0; axis < Dim; ++axis) {
          const std::size_t extent =
              static_cast<std::size_t>(patch.box.hi[axis] - patch.box.lo[axis] + 1);
          index[axis] = patch.box.lo[axis] + static_cast<int>(remainder % extent);
          remainder /= extent;
        }
        const std::size_t ordinal = linear_ordinal(index, config.shape);
        const bool expected = accepted_b.at(ordinal) > 0.5;
        exact_tag_pattern &= (patch.tags[local] != 0) == expected;
        primary_never_crosses &= accepted_a.at(ordinal) <= 0.5;
        expected_tag_count += expected ? 1u : 0u;
      }
    }
    require(primary_never_crosses && expected_tag_count > 0 &&
                expected_tag_count < cell_count(config.shape),
            "tagging threshold discriminates block b from primary block a");
    require(exact_tag_pattern && tagging.refine.count() == expected_tag_count,
            "tagging reads the exact non-primary block owner cell by cell");

    auto context = pops::runtime::program::make_program_execution_provider(&system);
    context->configure_primary_clock("tests.mpi.multiblock-field");
    context->begin_step(0.01);
    auto point = context->boundary_evaluation_point(3);

    pops::SolveOutcome prepared_rhs = context->solve_default_field_on_coarse_level();
    require(prepared_rhs.report().solved_value_available(),
            "prepared block-level add_poisson_rhs solve succeeds");
    (void)prepared_rhs.consume(pops::SolveConsumption::kAccept);
    require(!system.level_potential(0).empty() && !system.level_potential(1).empty(),
            "prepared block-level RHS publishes both levels");

    Field stage_a(context->state(0));
    Field stage_b(context->state(1));
    Field common_delta_a(stage_a);
    Field common_delta_b(stage_b);
    common_delta_a.set_val(pops::Real(0.25));
    common_delta_b.set_val(pops::Real(0.25));
    pops::saxpy(stage_a, pops::Real(1), common_delta_a);
    pops::saxpy(stage_b, pops::Real(1), common_delta_b);
    const Field authored_stage_a(stage_a);
    const Field authored_stage_b(stage_b);

    auto solve = [&](std::int64_t identity, const Field* a, const Field* b) {
      const std::vector<double> visible = system.field_potential_level_global("field/coupled", 0);
      const int calls_before = rhs_calls;
      pops::SolveOutcome outcome =
          context->solve_fields_from_blocks_at(point, identity, "field/coupled", {{0, a}, {1, b}});
      require(rhs_calls == calls_before + 4, "every block and level assembles exactly once");
      require(system.block_level_state_global("a", 0) == accepted_a,
              "block a remains accepted before publication");
      require(system.block_level_state_global("b", 0) == accepted_b,
              "block b remains accepted before publication");
      require(system.block_level_state_global("a", 1) == accepted_fine_a,
              "fine block a remains accepted before publication");
      require(system.block_level_state_global("b", 1) == accepted_fine_b,
              "fine block b remains accepted before publication");
      require(system.field_potential_level_global("field/coupled", 0) == visible,
              "field candidate remains private before consumption");
      require(outcome.report().solved_value_available(), "field report is publishable");
      (void)outcome.consume(pops::SolveConsumption::kAccept);
      return system.field_potential_level_global("field/coupled", 0);
    };

    const std::vector<double> base = solve(100, nullptr, nullptr);
    const std::vector<double> only_a = solve(101, &stage_a, nullptr);
    const std::vector<double> only_b = solve(102, nullptr, &stage_b);
    const std::vector<double> both = solve(103, &stage_a, &stage_b);
    const auto [error, response] = superposition_error(both, only_a, only_b, base);
    require(response > 1.0e-9, "both detached stages affect the field");
    require(error < 5.0e-7 * response + 1.0e-10, "two-block stage superposition");
    require(max_difference(system.auxiliary_component(output, 0), both) < 1.0e-10,
            "accepted field publishes its provider carrier");

    double weighted_error = 0.0;
    double weighted_response = 0.0;
    for (std::size_t index = 0; index < base.size(); ++index) {
      weighted_error = std::max(weighted_error, std::abs((only_b[index] - base[index]) +
                                                         0.25 * (only_a[index] - base[index])));
      weighted_response = std::max(weighted_response, std::abs(only_a[index] - base[index]));
    }
    require(weighted_response > 1.0e-9, "non-unit block coefficient response is observable");
    require(weighted_error < 5.0e-7 * weighted_response + 1.0e-10,
            "field response follows the exact 2.0/-0.5 weighted oracle");

    const std::vector<double> before_failed_prepare =
        system.field_potential_level_global("field/coupled", 0);
    const std::vector<double> before_failed_provider = system.auxiliary_component(output, 0);
    fail_rhs_on_rank_one = true;
    bool preparation_rejected = false;
    try {
      (void)context->solve_fields_from_blocks_at(point, 150, "field/coupled",
                                                 {{0, &stage_a}, {1, &stage_b}});
    } catch (const std::runtime_error&) {
      preparation_rejected = true;
    }
    fail_rhs_on_rank_one = false;
    require(preparation_rejected, "rank-local RHS failure is rejected before solver entry");
    require(system.field_potential_level_global("field/coupled", 0) == before_failed_prepare,
            "failed preparation preserves accepted potential");
    require(system.auxiliary_component(output, 0) == before_failed_provider,
            "failed preparation preserves accepted provider publication");
    require(system.block_level_state_global("a", 0) == accepted_a &&
                system.block_level_state_global("b", 0) == accepted_b,
            "failed preparation preserves live block states");
    const std::vector<double> retry = solve(151, &stage_a, &stage_b);
    require(max_difference(retry, both) < 1.0e-10,
            "accepted retry publishes the same simultaneous candidate");

    const std::vector<double> validation_potential_before =
        system.field_potential_level_global("field/provider-validation-fault", 0);
    const std::vector<double> validation_provider_before = system.auxiliary_component(output, 0);
    BoundaryBindingAudit::reset();
    PreparedHierarchyLaneAudit::reset();
    bool candidate_validation_rejected = false;
    try {
      (void)context->solve_fields_from_blocks_at(point, 155, "field/provider-validation-fault",
                                                 {{0, &stage_a}, {1, &stage_b}});
    } catch (const std::runtime_error&) {
      candidate_validation_rejected = true;
    }
    require(candidate_validation_rejected,
            "rank-local field candidate validation failure is converged before outcome return");
    require(BoundaryBindingAudit::binding_count >= 3 &&
                BoundaryBindingAudit::bound_state != &stage_a &&
                BoundaryBindingAudit::bound_state_min == pops::Real(-2),
            "candidate validation rollback restores the live boundary binding without retaining "
            "the detached stage");
    require(system.block_level_state_global("a", 0) == accepted_a &&
                system.block_level_state_global("b", 0) == accepted_b,
            "candidate validation rollback preserves both accepted block states");
    require(system.field_potential_level_global("field/provider-validation-fault", 0) ==
                    validation_potential_before &&
                system.auxiliary_component(output, 0) == validation_provider_before,
            "candidate validation rollback preserves accepted field and provider publications");
    require(field_difference_inf(stage_a, authored_stage_a) == 0.0 &&
                field_difference_inf(stage_b, authored_stage_b) == 0.0,
            "candidate validation rollback preserves both detached stage candidates");

    pops::SolveOutcome validation_retry = context->solve_fields_from_blocks_at(
        point, 155, "field/provider-validation-fault", {{0, &stage_a}, {1, &stage_b}});
    require(validation_retry.report().solved_value_available(),
            "clean retry succeeds after rank-local candidate validation rollback");
    require(BoundaryBindingAudit::bound_state != &stage_a &&
                BoundaryBindingAudit::bound_state_min == pops::Real(-2),
            "validated outcome restores the accepted boundary binding before consumption");
    const int validation_bindings_before_accept = BoundaryBindingAudit::binding_count;
    bool first_accept_validation_rejected = false;
    try {
      (void)validation_retry.consume(pops::SolveConsumption::kAccept);
    } catch (const std::logic_error&) {
      first_accept_validation_rejected = true;
    }
    require(first_accept_validation_rejected,
            "rank-local accept validation failure leaves the same SolveOutcome unconsumed");
    require(system.field_potential_level_global("field/provider-validation-fault", 0) ==
                    validation_potential_before &&
                system.auxiliary_component(output, 0) == validation_provider_before &&
                system.block_level_state_global("a", 0) == accepted_a &&
                system.block_level_state_global("b", 0) == accepted_b &&
                field_difference_inf(stage_a, authored_stage_a) == 0.0 &&
                field_difference_inf(stage_b, authored_stage_b) == 0.0,
            "failed accept validation preserves candidate inputs and all accepted publications");
    require(BoundaryBindingAudit::binding_count == validation_bindings_before_accept &&
                BoundaryBindingAudit::bound_state != &stage_a &&
                BoundaryBindingAudit::bound_state_min == pops::Real(-2),
            "failed accept validation does not mutate the restored solver boundary binding");

    bool solved_reject_rejected = false;
    try {
      (void)validation_retry.consume(pops::SolveConsumption::kRejectAttempt);
    } catch (const std::logic_error&) {
      solved_reject_rejected = true;
    }
    require(solved_reject_rejected,
            "explicit RejectAttempt remains invalid for a solved unconsumed outcome");

    bool same_outcome_accepted = false;
    if (first_accept_validation_rejected) {
      try {
        same_outcome_accepted =
            validation_retry.consume(pops::SolveConsumption::kAccept).solved_value_available();
      } catch (const std::exception&) {
        same_outcome_accepted = false;
      }
    }
    require(same_outcome_accepted,
            "the same SolveOutcome accepts after its transient validation failure");
    require(BoundaryBindingAudit::bound_state != &stage_a,
            "accepted retry does not retain its detached attempt-stage boundary pointer");

    pops::SolveOutcome provider_positive = context->solve_fields_from_blocks_at(
        point, 152, "field/provider-positive", {{0, &stage_a}, {1, &stage_b}});
    require(provider_positive.report().solved_value_available(),
            "custom AMR provider returns one collective publishable report");
    require(PreparedHierarchyLaneAudit::observed && PreparedHierarchyLaneAudit::owns_communicator,
            "custom AMR provider executes on the owned PreparedHierarchy lane");
    require(PreparedHierarchyLaneAudit::has_qualified_identity,
            "PreparedHierarchy lane exposes its exact parent-qualified identity");
    require(PreparedHierarchyLaneAudit::is_distinct_world_duplicate,
            "PreparedHierarchy lane is congruent with but distinct from MPI_COMM_WORLD");
    (void)provider_positive.consume(pops::SolveConsumption::kAccept);
    const std::vector<double> provider_fault_visible = system.auxiliary_component(output, 0);
    bool provider_report_rejected = false;
    try {
      (void)context->solve_fields_from_blocks_at(point, 153, "field/provider-divergent",
                                                 {{0, &stage_a}, {1, &stage_b}});
    } catch (const std::runtime_error&) {
      provider_report_rejected = true;
    }
    require(provider_report_rejected,
            "AMR production field path rejects a provider-owned divergent SolveReport");
    require(system.auxiliary_component(output, 0) == provider_fault_visible,
            "provider report divergence preserves accepted publication");
    bool provider_throw_rejected = false;
    try {
      (void)context->solve_fields_from_blocks_at(point, 154, "field/provider-throw",
                                                 {{0, &stage_a}, {1, &stage_b}});
    } catch (const std::runtime_error&) {
      provider_throw_rejected = true;
    }
    require(provider_throw_rejected,
            "AMR production field path contains a rank-local provider solve throw");
    require(system.auxiliary_component(output, 0) == provider_fault_visible &&
                system.block_level_state_global("a", 0) == accepted_a &&
                system.block_level_state_global("b", 0) == accepted_b,
            "rank-local provider failure preserves publications and live states");

    const std::vector<double> reject_before =
        system.field_potential_level_global("field/reject", 0);
    const std::vector<double> provider_before_reject = system.auxiliary_component(output, 0);
    const std::vector<double> stage_a_before = system.block_level_state_global("a", 0);
    pops::SolveOutcome rejected = context->solve_fields_from_blocks_at(
        point, 160, "field/reject", {{0, &stage_a}, {1, &stage_b}});
    require(!rejected.report().solved_value_available() &&
                rejected.report().action == pops::SolveAction::kRejectAttempt,
            "real one-cycle field solve authors a rejected candidate");
    if (!rejected.report().solved_value_available())
      (void)rejected.consume(pops::SolveConsumption::kRejectAttempt);
    require(system.field_potential_level_global("field/reject", 0) == reject_before,
            "rejected real solve preserves its accepted potential");
    require(system.auxiliary_component(output, 0) == provider_before_reject,
            "rejected real solve preserves provider publication");
    require(system.block_level_state_global("a", 0) == stage_a_before,
            "rejected real solve preserves live state");
    require(field_difference_inf(stage_a, authored_stage_a) == 0.0 &&
                field_difference_inf(stage_b, authored_stage_b) == 0.0,
            "rejected real solve preserves detached stage candidates");
    const pops::ExecutionLane& lane = context->prepared_execution_lane();
    for (const EllipticFactoryFault fault : {
             EllipticFactoryFault::throw_on_rank_one,
             EllipticFactoryFault::null_on_rank_one,
             EllipticFactoryFault::wrong_components_on_rank_one,
             EllipticFactoryFault::aliased_fields_on_rank_one,
             EllipticFactoryFault::wrong_ghosts_on_rank_one,
             EllipticFactoryFault::wrong_operator_contract_on_rank_one,
             EllipticFactoryFault::wrong_distribution_on_rank_one,
             EllipticFactoryFault::inspection_throws_on_rank_one,
         }) {
      int constructions = 0;
      require(elliptic_factory_rejected(system.prepared_amr_block_state(0, 1), config,
                                        system.engine()->hierarchy().layout(1).domain(), fault,
                                        constructions) &&
                  constructions == 1,
              "rank-local elliptic factory/materialization fault is rejected collectively");
    }
    pops::SolveReport agreed_report;
    agreed_report.mark_solved("tests.mpi.multiblock-field/report");
    pops::ExactSolveReportConsensusScratch report_consensus;
    require(report_consensus.agrees(agreed_report, lane),
            "identical SolveReport reaches exact byte consensus");
    pops::SolveReport divergent_report = agreed_report;
    if (pops::my_rank() == 1)
      divergent_report.reason = "tests.mpi.multiblock-field/divergent-report";
    require(!report_consensus.agrees(divergent_report, lane),
            "rank-divergent SolveReport is detected exactly");
    pops::SolveOutcome report_outcome =
        pops::SolveOutcome::collective_lane(std::move(divergent_report), lane);
    bool divergent_disposition_rejected = false;
    try {
      (void)report_outcome.consume(pops::SolveConsumption::kAccept);
    } catch (const std::logic_error&) {
      divergent_disposition_rejected = true;
    }
    require(divergent_disposition_rejected,
            "rank-divergent SolveOutcome report disposition is rejected uniformly");

    pops::SolveReport publication_report;
    publication_report.mark_solved("validated-publication");
    RankLocalPublication publication{pops::my_rank() == 0, false};
    pops::SolveOutcome publication_outcome =
        pops::SolveOutcome::collective_lane(std::move(publication_report), lane,
                                            {&publication,
                                             &RankLocalPublication::accept,
                                             nullptr,
                                             nullptr,
                                             {},
                                             &RankLocalPublication::validate});
    bool publication_validation_rejected = false;
    try {
      (void)publication_outcome.consume(pops::SolveConsumption::kAccept);
    } catch (const std::logic_error&) {
      publication_validation_rejected = true;
    }
    require(publication_validation_rejected && !publication.accepted,
            "rank-local publication validation rejects before any accept hook");
    publication.layout_valid = true;
    require(publication_outcome.consume(pops::SolveConsumption::kAccept).solved_value_available() &&
                publication.accepted,
            "validated publication remains intact for a uniform accept retry");

    pops::SolveOutcome action_outcome = pops::SolveOutcome::collective_lane(agreed_report, lane);
    bool divergent_consumption_rejected = false;
    try {
      (void)action_outcome.consume(pops::my_rank() == 0 ? pops::SolveConsumption::kAccept
                                                        : pops::SolveConsumption::kFailRun);
    } catch (const std::logic_error&) {
      divergent_consumption_rejected = true;
    }
    require(divergent_consumption_rejected,
            "rank-divergent SolveOutcome consumption is rejected before publication");
    require(action_outcome.consume(pops::SolveConsumption::kAccept).solved_value_available(),
            "intact SolveOutcome accepts after a uniform retry");

    struct NonlinearFailureCase {
      pops::ImplicitEvaluationStatus status;
      pops::SolveAction action;
      pops::SolveConsumption consumption;
    };
    constexpr std::uint32_t nonlinear_reason = 0x75000001u;
    for (const NonlinearFailureCase failure : {
             NonlinearFailureCase{pops::ImplicitEvaluationStatus::kRetry,
                                  pops::SolveAction::kRejectAttempt,
                                  pops::SolveConsumption::kRejectAttempt},
             NonlinearFailureCase{pops::ImplicitEvaluationStatus::kReject,
                                  pops::SolveAction::kRejectAttempt,
                                  pops::SolveConsumption::kRejectAttempt},
             NonlinearFailureCase{pops::ImplicitEvaluationStatus::kFailed,
                                  pops::SolveAction::kFailRun, pops::SolveConsumption::kFailRun},
         }) {
      Field nonlinear_state(context->state(0));
      nonlinear_state.set_val(pops::Real(3));
      const Field accepted_nonlinear_state(nonlinear_state);
      Field rank_provider(context->state(0));
      rank_provider.set_val(static_cast<pops::Real>(pops::my_rank()));
      const Field& const_rank_provider = rank_provider;
      const auto provider_at = [&const_rank_provider](std::size_t local) {
        pops::ProviderStorageView<Dim, 1> view;
        view.storage[0] = const_rank_provider.fab(local).view();
        view.storage_components[0] = 0;
        return view;
      };
      pops::NewtonReport diagnostics;
      diagnostics.max_residual = pops::Real(42);
      pops::SolveOutcome nonlinear = pops::backward_euler_source(
          RankLocalFallibleSource{failure.status, nonlinear_reason}, provider_at, nonlinear_state,
          pops::Real(0.1), pops::NewtonOptions{}, lane, {}, &diagnostics);
      require(nonlinear.report().status == pops::SolveStatus::kInvalidEvaluation &&
                  nonlinear.report().action == failure.action,
              "rank-local nonlinear source failure becomes one collective transaction outcome");
      require(field_difference_inf(nonlinear_state, accepted_nonlinear_state) == 0.0 &&
                  !diagnostics.enabled && diagnostics.max_residual == pops::Real(42),
              "nonlinear failure keeps state and accepted diagnostics private");
      (void)nonlinear.consume(failure.consumption);
      require(field_difference_inf(nonlinear_state, accepted_nonlinear_state) == 0.0,
              "nonlinear retry/reject/fail consumption restores its transaction");
    }

    Field iterate(context->state(0));
    Field direction(iterate);
    direction.set_val(pops::Real(0.5));
    const auto boundary = context->prepare_block_boundary_session(0, iterate, point, lane);
    const auto residual_at = [&](pops::Real shift, bool refresh_field) {
      Field state(iterate);
      pops::saxpy(state, shift, direction);
      Field residual(iterate);
      residual.set_val(pops::Real(0));
      const auto evaluate = [&] {
        context->rhs_core_into_at(point, 0, state, residual, false, *boundary);
      };
      if (refresh_field)
        context->evaluate_with_field_state_at(point, "field/coupled", 0, state, iterate, evaluate);
      else
        evaluate();
      return residual;
    };
    constexpr pops::Real perturbation = pops::Real(2.0e-4);
    const Field residual_base = residual_at(pops::Real(0), false);
    const Field residual_plus = residual_at(perturbation, true);
    const Field residual_minus = residual_at(-perturbation, true);
    const Field residual_stale_plus = residual_at(perturbation, false);
    Field centered_jvp(direction);
    pops::saxpy(centered_jvp, -pops::Real(0.01) / (pops::Real(2) * perturbation), residual_plus);
    pops::saxpy(centered_jvp, pops::Real(0.01) / (pops::Real(2) * perturbation), residual_minus);
    Field stale_jvp(direction);
    pops::saxpy(stale_jvp, -pops::Real(0.01) / perturbation, residual_stale_plus);
    pops::saxpy(stale_jvp, pops::Real(0.01) / perturbation, residual_base);
    require(field_difference_inf(centered_jvp, direction) > 1.0e-8,
            "elliptic field-coupled residual has a nonzero JVP response");
    require(field_difference_inf(centered_jvp, stale_jvp) > 1.0e-8,
            "elliptic JVP rejects a frozen solved-field provider");
    require(system.field_potential_level_global("field/coupled", 0) == retry,
            "field-coupled JVP restores the complete accepted provider hierarchy");

    const auto reject_divergence = [&](std::int64_t identity,
                                       const pops::runtime::multiblock::BoundaryEvaluationPoint& p,
                                       std::string_view provider, const Field* a, const Field* b) {
      const int calls_before = rhs_calls;
      const std::vector<double> visible = system.field_potential_level_global("field/coupled", 0);
      bool rejected = false;
      try {
        (void)context->solve_fields_from_blocks_at(p, identity, provider, {{0, a}, {1, b}});
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "rank-divergent request rejected collectively");
      require(rhs_calls == calls_before, "rank divergence rejected before RHS assembly");
      require(system.field_potential_level_global("field/coupled", 0) == visible,
              "rank divergence preserves published provider");
      require(system.block_level_state_global("a", 0) == accepted_a,
              "rank divergence preserves block a");
      require(system.block_level_state_global("b", 0) == accepted_b,
              "rank divergence preserves block b");
    };

    auto divergent_point = point;
    if (pops::my_rank() == 1)
      ++divergent_point.stage;
    reject_divergence(200, divergent_point, "field/coupled", &stage_a, &stage_b);
    reject_divergence(201, point, pops::my_rank() == 1 ? "field/rank-one" : "field/coupled",
                      &stage_a, &stage_b);
    const std::vector<double> cache_retry = solve(201, &stage_a, &stage_b);
    require(max_difference(cache_retry, retry) < 1.0e-10,
            "divergent provider request leaves generated field-route cache unchanged");
    reject_divergence(202, point, "field/coupled", &stage_a,
                      pops::my_rank() == 1 ? nullptr : &stage_b);

    std::size_t fine_cells = 0;
    std::size_t covered_coarse_cells = 0;
    for (const auto& patch : patches) {
      fine_cells += static_cast<std::size_t>(patch.box.numPts());
      std::size_t coarsened_cells = 1;
      for (int axis = 0; axis < Dim; ++axis)
        coarsened_cells *=
            static_cast<std::size_t>(patch.box.hi[axis] / 2 - patch.box.lo[axis] / 2 + 1);
      covered_coarse_cells += coarsened_cells;
    }
    const std::size_t coarse_cells = cell_count(config.shape);
    const std::size_t uncovered_coarse_cells = coarse_cells - covered_coarse_cells;
    double coarse_cell_measure = 1.0;
    double fine_cell_measure = 1.0;
    for (int axis = 0; axis < Dim; ++axis) {
      coarse_cell_measure *= static_cast<double>(config.upper[axis] - config.lower[axis]) /
                             static_cast<double>(config.shape[axis]);
      fine_cell_measure *= static_cast<double>(config.upper[axis] - config.lower[axis]) /
                           static_cast<double>(2 * config.shape[axis]);
    }
    const auto close_norm = [](double actual, double expected) {
      return std::abs(actual - expected) <= 1.0e-10 * std::max(1.0, expected);
    };

    system.begin_step_transaction();
    Field fine_only_a0(system.prepared_amr_block_state(0, 0));
    Field fine_only_b0(system.prepared_amr_block_state(1, 0));
    Field fine_only_a1(system.prepared_amr_block_state(0, 1));
    Field fine_only_b1(system.prepared_amr_block_state(1, 1));
    add_constant(fine_only_a1, pops::Real(2));
    std::vector<Field*> fine_only_coarse{&fine_only_a0, &fine_only_b0};
    std::vector<Field*> fine_only_fine{&fine_only_a1, &fine_only_b1};
    system.publish_prepared_amr_program_candidates(0, fine_only_coarse);
    system.publish_prepared_amr_program_candidates(1, fine_only_fine);
    const auto fine_only_norms = system.step_change_l2();
    const double expected_fine_only =
        2.0 * std::sqrt(static_cast<double>(fine_cells) * fine_cell_measure);
    require(fine_only_norms.size() == 2 &&
                close_norm(fine_only_norms.at("a"), expected_fine_only) &&
                close_norm(fine_only_norms.at("b"), 0.0),
            "step_change_l2 applies the exact physical fine-only composite weight");
    system.rollback_step_transaction();
    require(system.block_level_state_global("a", 0) == accepted_a &&
                system.block_level_state_global("b", 0) == accepted_b &&
                system.block_level_state_global("a", 1) == accepted_fine_a &&
                system.block_level_state_global("b", 1) == accepted_fine_b,
            "fine-only transaction rollback restores every carrier");

    system.begin_step_transaction();
    Field transaction_a0(system.prepared_amr_block_state(0, 0));
    Field transaction_b0(system.prepared_amr_block_state(1, 0));
    Field transaction_a1(system.prepared_amr_block_state(0, 1));
    Field transaction_b1(system.prepared_amr_block_state(1, 1));
    add_constant(transaction_a0, pops::Real(1));
    add_constant(transaction_b0, pops::Real(2));
    add_constant(transaction_a1, pops::Real(3));
    add_constant(transaction_b1, pops::Real(4));
    std::vector<Field*> coarse_candidates{&transaction_a0, &transaction_b0};
    std::vector<Field*> fine_candidates{&transaction_a1, &transaction_b1};
    system.publish_prepared_amr_program_candidates(0, coarse_candidates);
    system.publish_prepared_amr_program_candidates(1, fine_candidates);
    const auto step_norms = system.step_change_l2();
    const double expected_a =
        std::sqrt(static_cast<double>(uncovered_coarse_cells) * coarse_cell_measure +
                  9.0 * static_cast<double>(fine_cells) * fine_cell_measure);
    const double expected_b =
        std::sqrt(4.0 * static_cast<double>(uncovered_coarse_cells) * coarse_cell_measure +
                  16.0 * static_cast<double>(fine_cells) * fine_cell_measure);
    require(step_norms.size() == 2 && close_norm(step_norms.at("a"), expected_a) &&
                close_norm(step_norms.at("b"), expected_b),
            "step_change_l2 covers every block with exact coarse/fine composite weights");
    system.rollback_step_transaction();
    require(system.block_level_state_global("a", 0) == accepted_a &&
                system.block_level_state_global("b", 0) == accepted_b &&
                system.block_level_state_global("a", 1) == accepted_fine_a &&
                system.block_level_state_global("b", 1) == accepted_fine_b,
            "real multi-level two-block transaction rollback restores every carrier");
  } catch (const std::exception& error) {
    std::fprintf(stderr, "rank %d: multi-block field fixture failed: %s\n", pops::my_rank(),
                 error.what());
    ++failures;
  }

  failures = pops::all_reduce_sum(failures);
  pops::comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_field_plan_consensus, ExactRankMultiBlockFieldSolveIsCollectiveAndTransactional) {
  EXPECT_EQ(pops::test::RunTestBody(&run_multiblock_field_solve, "test_mpi_field_plan_consensus"),
            0);
}
