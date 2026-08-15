// Public exact-rank AMR composite-field proof.
//
// This intentionally exercises the facade route rather than constructing a FAC directly:
// `set_field_solver_plan` -> materialize_field -> AmrProgramContext staged solve ->
// SolveOutcome reject/accept publication.  Both L0 and L1 are partitioned on two MPI ranks.

#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "gtest_compat.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

template <int Dim>
struct LaneCheckingDistribution {
  std::string expected_lane;

  static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.mpi.lane-checking-vector-distribution", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.text(expected_lane);
  }
  bool layout_matches(const pops::MultiFab<Dim>& field) const {
    return pops::detail::field_distribution_layout_matches(field,
                                                           pops::FieldDistribution::Distributed);
  }
  std::string layout_contract(const pops::MultiFab<Dim>& field) const {
    return pops::detail::field_distribution_layout_contract(field,
                                                            pops::FieldDistribution::Distributed);
  }
  std::size_t reduction_scratch_value_count(std::size_t) const noexcept { return 0; }
  std::size_t validation_scratch_byte_count() const noexcept { return 0; }
  pops::PreparedVectorDistributionStatus reduce_sum_values(
      std::span<double> values, std::span<double>, const char*,
      const pops::ExecutionLane& lane) const noexcept {
    if (lane.identity() != expected_lane)
      return pops::PreparedVectorDistributionStatus::failure(
          1, "Newton metric used the wrong execution lane");
    try {
      pops::all_reduce_sum_inplace(values.data(), values.size(), lane);
      return pops::PreparedVectorDistributionStatus::success();
    } catch (...) {
      return pops::PreparedVectorDistributionStatus::failure(2,
                                                             "Newton metric lane reduction failed");
    }
  }
  pops::PreparedVectorDistributionStatus reduce_max_values(
      std::span<double> values, std::span<double> scratch, const char* where,
      const pops::ExecutionLane& lane) const noexcept {
    return reduce_sum_values(values, scratch, where, lane);
  }
  pops::PreparedVectorDistributionStatus require_exact_values(
      const pops::MultiFab<Dim>&, std::span<char>, const char*,
      const pops::ExecutionLane& lane) const noexcept {
    return lane.identity() == expected_lane
               ? pops::PreparedVectorDistributionStatus::success()
               : pops::PreparedVectorDistributionStatus::failure(
                     3, "Newton vector validation used the wrong execution lane");
  }
};

template <int Dim>
struct PartitionedBoundaryProbe {
  inline static bool fail_on_rank_zero = false;
  inline static int residual_calls = 0;
  inline static int jvp_calls = 0;

  static void observe(int face, const pops::FieldBoundaryExecutionContext<Dim>& context, bool jvp) {
    if (jvp)
      ++jvp_calls;
    else
      ++residual_calls;
    if (fail_on_rank_zero && pops::my_rank() == 0) {
      context.failure->code = 750;
      context.failure->face = face;
    }
  }

  static void prepare_residual(int face, const pops::MultiFab<Dim>&, pops::MultiFab<Dim>&,
                               const pops::Geometry<Dim>&,
                               const pops::FieldBoundaryExecutionContext<Dim>& context) {
    observe(face, context, false);
  }
  static void prepare_jvp(int face, const pops::MultiFab<Dim>&, const pops::MultiFab<Dim>&,
                          pops::MultiFab<Dim>&, const pops::Geometry<Dim>&,
                          const pops::FieldBoundaryExecutionContext<Dim>& context) {
    observe(face, context, true);
  }
  static void add_residual(int face, const pops::MultiFab<Dim>&, pops::MultiFab<Dim>&,
                           const pops::Geometry<Dim>&,
                           const pops::FieldBoundaryExecutionContext<Dim>& context) {
    observe(face, context, false);
  }
  static void apply_jvp(int face, const pops::MultiFab<Dim>&, const pops::MultiFab<Dim>&,
                        pops::MultiFab<Dim>&, const pops::Geometry<Dim>&,
                        const pops::FieldBoundaryExecutionContext<Dim>& context) {
    observe(face, context, true);
  }

  static pops::CompiledFieldBoundaryKernel<Dim> kernel() {
    return {"test.mpi.partitioned-boundary",
            "test.mpi.partitioned-boundary.residual",
            "test.mpi.partitioned-boundary.jvp",
            &prepare_residual,
            &prepare_jvp,
            &add_residual,
            &apply_jvp,
            true};
  }
};

template <int Dim>
struct ScalarFieldModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;

  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_providers = 0;
  Law law{};

  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.mpi.partitioned-composite.scalar", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
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
  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return {}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

template <int Dim>
ScalarFieldModel<Dim> scalar_field_model() {
  pops::RealVector<Dim> velocity{};
  for (int axis = 0; axis < Dim; ++axis)
    velocity[axis] = pops::Real(0);
  return {pops::nd::ScalarAdvection<Dim>::prepare(velocity)};
}

template <int Dim>
void add_scalar_field_block(pops::AmrSystem<Dim>& system) {
  pops::add_compiled_model<Dim>(system, "tracer", scalar_field_model<Dim>(), "minmod", "rusanov",
                                "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "test.mpi.partitioned-composite/physical-flux");
}

template <int Dim>
std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
pops::AmrSystemConfig<Dim> distributed_two_level_config() {
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 2;
  config.regrid_every = 0;
  config.explicit_bootstrap = true;
  config.distribute_coarse = true;
  config.cluster_max_box_size = 4;
  config.load_balance_route = "round_robin";
  config.load_balance_identity = "test.mpi.partitioned-composite.round-robin@1";
  config.load_balance_options = {"pops.amr.load-balance.round-robin@1", {}};
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 8;
    config.lower[axis] = pops::Real(0);
    config.upper[axis] = pops::Real(1);
    config.periodicity[axis] = true;
    config.coarse_max_grid[axis] = 4;
    config.transition_buffers.front()[axis] = 0;
    config.transition_lookaheads.front()[axis] = 0;
  }
  // The distributed coarse route takes its exact tiles from RuntimeSpatialDomain::boxes; the
  // two slabs force an actual L0 owner split before the fully tagged L1 is load balanced.
  for (int owner = 0; owner < 2; ++owner) {
    pops::Index<Dim> lower{};
    pops::Index<Dim> upper{};
    for (int axis = 0; axis < Dim; ++axis) {
      lower[axis] = 0;
      upper[axis] = config.shape[axis] - 1;
    }
    lower[0] = owner * (config.shape[0] / 2);
    upper[0] = lower[0] + config.shape[0] / 2 - 1;
    config.boxes.emplace_back(lower, upper);
  }
  return config;
}

template <int Dim>
pops::runtime::system::AuxiliaryComponentKey install_field_output(pops::AmrSystem<Dim>& system) {
  using namespace pops::runtime::system;
  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 1;
  const AuxiliaryComponentKey key{"test.mpi.partitioned-composite", "field", "phi", "potential"};
  const AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "amr-field",
                                            "scalar"};
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "test.mpi.partitioned-composite.field-output@1",
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      {{key, contract, shape}},
      {}});
  system.seal_auxiliary_providers();
  return key;
}

template <int Dim>
pops::runtime::multiblock::BoundaryEvaluationPoint evaluation_point(int level, int stage) {
  return {.clock = "test.mpi.partitioned-composite.clock",
          .tick = 0,
          .level = level,
          .substep = 0,
          .stage = stage,
          .stage_fraction = {0, 1},
          .dt = 0.01,
          .physical_time = 0.0};
}

template <int Dim>
bool partitioned_on_every_rank(const pops::MultiFab<Dim>& field) {
  const auto& distribution = field.distribution();
  return !distribution.replicated() && field.local_size() != 0 &&
         distribution.rank_space().size() == static_cast<std::size_t>(pops::n_ranks()) &&
         distribution.rank_space().linear_rank(field.local_rank()) ==
             static_cast<std::size_t>(pops::my_rank());
}

template <int Dim, class Engine>
bool has_cross_owner_parent_child_edge(const Engine& engine) {
  if (engine.hierarchy().num_levels() < 2)
    return false;
  const auto& parent = engine.hierarchy().layout(0);
  const auto& child = engine.hierarchy().layout(1);
  if (parent.distribution().replicated() || child.distribution().replicated())
    return false;
  for (std::size_t fine_patch = 0; fine_patch < child.patches().size(); ++fine_patch) {
    const pops::Box<Dim> footprint =
        pops::amr::hierarchy::coarsen_box(child.patches()[fine_patch], child.ratio_from_parent());
    for (std::size_t coarse_patch = 0; coarse_patch < parent.patches().size(); ++coarse_patch)
      if (!footprint.intersect(parent.patches()[coarse_patch]).empty() &&
          child.distribution().owner(fine_patch) != parent.distribution().owner(coarse_patch))
        return true;
  }
  return false;
}

template <int Dim, class Engine>
void rebuild_with_cross_owner_fine_patches(pops::AmrSystem<Dim>& system, const Engine& engine,
                                           const pops::Extent<Dim>& ratio) {
  const auto& parent = engine.hierarchy().layout(0);
  const auto& rank_space = parent.distribution().rank_space();
  std::vector<pops::AmrPatch<Dim>> fine_patches;
  std::vector<int> fine_owners;
  fine_patches.reserve(parent.patches().size());
  fine_owners.reserve(parent.patches().size());
  for (std::size_t coarse_patch = 0; coarse_patch < parent.patches().size(); ++coarse_patch) {
    const pops::Box<Dim>& coarse = parent.patches()[coarse_patch];
    pops::Index<Dim> lower{};
    pops::Index<Dim> upper{};
    for (int axis = 0; axis < Dim; ++axis) {
      lower[axis] = coarse.lo[axis] * ratio[axis];
      upper[axis] = (coarse.hi[axis] + 1) * ratio[axis] - 1;
    }
    const int parent_owner =
        parent.distribution().replicated()
            ? static_cast<int>(coarse_patch % rank_space.size())
            : static_cast<int>(rank_space.linear_rank(parent.distribution().owner(coarse_patch)));
    fine_patches.push_back({1, pops::Box<Dim>{lower, upper}});
    fine_owners.push_back((parent_owner + 1) % pops::n_ranks());
  }
  system.rebuild_hierarchy(fine_patches, fine_owners);
}

template <int Dim>
double maximum_difference(const std::vector<double>& left, const std::vector<double>& right) {
  if (left.size() != right.size())
    return std::numeric_limits<double>::infinity();
  double difference = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index)
    difference = std::max(difference, std::fabs(left[index] - right[index]));
  return difference;
}

template <int Dim>
std::vector<std::vector<double>> published_hierarchy(
    pops::AmrSystem<Dim>& system, const pops::runtime::system::AuxiliaryComponentKey& output_key) {
  std::vector<std::vector<double>> levels;
  levels.reserve(static_cast<std::size_t>(system.n_levels()));
  for (int level = 0; level < system.n_levels(); ++level)
    levels.push_back(system.auxiliary_component(output_key, level));
  return levels;
}

template <int Dim>
bool same_hierarchy(const std::vector<std::vector<double>>& left,
                    const std::vector<std::vector<double>>& right) {
  if (left.size() != right.size())
    return false;
  for (std::size_t level = 0; level < left.size(); ++level)
    if (maximum_difference<Dim>(left[level], right[level]) != 0.0)
      return false;
  return true;
}

template <int Dim>
long prove_newton_uses_duplicated_lane() {
  pops::Index<Dim> domain_lo{};
  pops::Index<Dim> domain_hi{};
  pops::Index<Dim> split_hi{};
  pops::Index<Dim> split_lo{};
  pops::Extent<Dim> rank_extent{};
  pops::Index<Dim> local_rank{};
  for (int axis = 0; axis < Dim; ++axis) {
    domain_hi[axis] = 3;
    split_hi[axis] = domain_hi[axis];
    split_lo[axis] = domain_lo[axis];
    rank_extent[axis] = 1;
  }
  split_hi[0] = 1;
  split_lo[0] = 2;
  rank_extent[0] = 2;
  local_rank[0] = pops::my_rank();
  const pops::mesh::BoxArray<Dim> layout{
      std::vector<pops::Box<Dim>>{{domain_lo, split_hi}, {split_lo, domain_hi}}};
  const pops::mesh::RankSpace<Dim> rank_space{pops::Index<Dim>{}, rank_extent};
  pops::Index<Dim> rank_one{};
  rank_one[0] = 1;
  const auto distribution = pops::mesh::Distribution<Dim>::partitioned(
      layout, rank_space, {pops::Index<Dim>{}, rank_one});
  pops::MultiFab<Dim> candidate(layout, distribution, local_rank, 1, pops::Extent<Dim>{});
  pops::MultiFab<Dim> active(layout, distribution, local_rank, 1, pops::Extent<Dim>{});
  candidate.set_val(pops::Real(0));
  active.set_val(pops::Real(1));

  const std::string lane_identity = "test.mpi.partitioned-fac.newton-metric-lane";
  pops::ExecutionLane lane = pops::ExecutionLane::duplicate_world_collectively(lane_identity);
  const std::string prepared_lane_identity(lane.identity());
  const std::array<const pops::MultiFab<Dim>*, 1> layouts{&candidate};
  const std::array<const pops::MultiFab<Dim>*, 1> masks{&active};
  const std::array<pops::Real, 1> measures{pops::Real(1)};
  const std::array<pops::PreparedVectorDistribution<Dim>, 1> distributions{
      pops::PreparedVectorDistribution<Dim>{LaneCheckingDistribution<Dim>{prepared_lane_identity}}};
  pops::AmrFieldNewtonKrylovWorkspace<Dim> workspace(layouts, masks, measures, distributions, lane,
                                                     pops::FieldNewtonOptions{});
  std::array<pops::MultiFab<Dim>*, 1> destination{&candidate};
  const pops::SolveReport report = workspace.solve(
      destination,
      [](const auto&, auto& residual, int) {
        for (auto& level : residual)
          level.set_val(pops::Real(0));
      },
      [](const auto&, const auto&, auto& output, int) {
        for (auto& level : output)
          level.set_val(pops::Real(0));
      },
      [](auto&) {});
  return lane.owns_communicator() && report.solved() ? 0L : 1L;
}

template <int Dim>
long run_public_partitioned_composite() {
  long failures = 0;
  const auto require = [&failures](bool condition, const char* expectation) {
    if (!condition) {
      std::fprintf(stderr, "rank %d: unmet public partitioned composite expectation: %s\n",
                   pops::my_rank(), expectation);
      ++failures;
    }
  };
  try {
    const pops::AmrSystemConfig<Dim> config = distributed_two_level_config<Dim>();
    pops::AmrSystem<Dim> system(config);
    const pops::AmrFieldHierarchyPolicyAuthority composite{
        "pops.field-hierarchy.composite", 1, {"pops.field-hierarchy.options.empty@1", {}}};
    system.set_field_solver_plan("field/partitioned", "test.mpi.partitioned-composite.plan@1",
                                 "test.mpi.partitioned-composite.rhs@1",
                                 "test.mpi.partitioned-composite", "tracer", "phi",
                                 {"test.mpi.partitioned-composite.rhs@1"}, {"tracer"}, {"phi"},
                                 {1.0}, "geometric_mg", composite,
                                 pops::geometric_mg_amr_field_solver_options(
                                     pops::GeometricMgOptions{}, pops::CompositeFacOptions{}));
    system.set_field_reaction("field/partitioned", 2.0);
    system.set_field_boundary_dependencies("field/partitioned", {"tracer"}, {0}, {}, {}, {});
    system.set_field_boundary_parameters("field/partitioned", {0.25});
    system.set_field_boundary_kernel("field/partitioned", PartitionedBoundaryProbe<Dim>::kernel());
    system.set_field_newton_plan("field/partitioned", 1.0e-9, 4, 1.0e-10, 80, 20, 1.0e-4,
                                 1.0 / 1024.0);
    system.install_block_state_route("tracer", "test.mpi.partitioned-composite/state/tracer");
    add_scalar_field_block(system);
    const auto output_key = install_field_output(system);
    system.register_elliptic_field("tracer", "phi", {output_key}, 1);
    enum class RhsMode { finite, throw_rank_zero, nonfinite_rank_zero };
    RhsMode rhs_mode = RhsMode::finite;
    system.set_block_elliptic_field(
        "tracer", "phi", [&rhs_mode](const pops::MultiFab<Dim>& state, pops::MultiFab<Dim>& rhs) {
          if (rhs_mode == RhsMode::throw_rank_zero && pops::my_rank() == 0)
            throw std::runtime_error("injected rank-local exact field RHS failure");
          if (rhs_mode == RhsMode::nonfinite_rank_zero && pops::my_rank() == 0) {
            rhs.set_val(std::numeric_limits<pops::Real>::quiet_NaN());
            return;
          }
          for (std::size_t local = 0; local < state.local_size(); ++local) {
            const auto source = state.fab(local).view();
            const auto destination = rhs.fab(local).view();
            const pops::Box<Dim>& box = state.box(local);
            pops::for_each_cell(box, [=] POPS_HD(const pops::Index<Dim>& cell) {
              destination(cell, 0) += source(cell, 0);
            });
          }
          Kokkos::fence();
        });
    system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
    pops::test::install_prepared_threshold_union(system, {{"tracer", "u", 0.5}},
                                                 "test.mpi.partitioned-composite.tagging@1");
    system.begin_bootstrap_plan();
    if (!system.bootstrap_next_level()) {
      system.rollback_bootstrap_level();
      throw std::runtime_error("partitioned composite fixture did not create L1");
    }
    system.commit_bootstrap_level();
    auto* const bootstrapped_engine = system.engine();
    if (bootstrapped_engine == nullptr)
      throw std::runtime_error("partitioned composite fixture lost its bootstrapped AMR engine");
    rebuild_with_cross_owner_fine_patches<Dim>(system, *bootstrapped_engine,
                                               config.transition_ratios.front());
    system.set_program_block_map({0});

    require(system.n_levels() == 2, "the bootstrap hierarchy has L0 and L1");
    auto* const engine = system.engine();
    require(engine != nullptr, "the public AMR engine is materialized");
    if (engine != nullptr)
      for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level)
        require(partitioned_on_every_rank(engine->hierarchy().state(level)),
                "each live hierarchy level is locally owned on every rank");
    if (engine != nullptr)
      require(has_cross_owner_parent_child_edge<Dim>(*engine),
              "the FAC hierarchy contains a real cross-owner parent/fine transfer edge");

    auto context = pops::runtime::program::make_program_execution_provider(&system);
    context->configure_primary_clock("test.mpi.partitioned-composite.clock");
    context->begin_step(0.01);
    const std::vector<double> state_before = system.block_level_state_global("tracer", 0);

    context->with_program_resource_level(1, [&] {
      pops::MultiFab<Dim> stage = context->scratch_state_like(context->state(0));
      stage.set_val(pops::Real(2));
      pops::SolveOutcome pending = context->solve_fields_from_state_at(
          evaluation_point<Dim>(1, 1), "field/partitioned", 0, stage);
      const pops::SolveReport accepted = pending.consume(pops::SolveConsumption::kAccept);
      require(accepted.solved(), "the baseline partitioned FAC solve is accepted");
    });
    const auto accepted_before_failures = published_hierarchy(system, output_key);

    PartitionedBoundaryProbe<Dim>::fail_on_rank_zero = true;
    bool boundary_failure_rejected = false;
    context->with_program_resource_level(1, [&] {
      pops::MultiFab<Dim> stage = context->scratch_state_like(context->state(0));
      stage.set_val(pops::Real(3));
      try {
        pops::SolveOutcome pending = context->solve_fields_from_state_at(
            evaluation_point<Dim>(1, 2), "field/partitioned", 0, stage);
        boundary_failure_rejected = pending.report().failed();
        (void)pending.consume(pending.report().action == pops::SolveAction::kFailRun
                                  ? pops::SolveConsumption::kFailRun
                                  : pops::SolveConsumption::kRejectAttempt);
      } catch (const std::exception&) {
        boundary_failure_rejected = true;
      }
    });
    PartitionedBoundaryProbe<Dim>::fail_on_rank_zero = false;
    require(boundary_failure_rejected,
            "a rank-local dynamic-boundary failure is rejected collectively");
    require(same_hierarchy<Dim>(published_hierarchy(system, output_key), accepted_before_failures),
            "dynamic-boundary rejection preserves every accepted field-output level");
    require(PartitionedBoundaryProbe<Dim>::residual_calls > 0,
            "the partitioned residual boundary launcher executes");
    require(PartitionedBoundaryProbe<Dim>::jvp_calls > 0,
            "the partitioned JVP boundary launcher executes");

    bool invalid_point_rejected = false;
    context->with_program_resource_level(1, [&] {
      pops::MultiFab<Dim> stage = context->scratch_state_like(context->state(0));
      stage.set_val(pops::Real(3));
      auto point = evaluation_point<Dim>(1, 3);
      if (pops::my_rank() == 0)
        point.stage_fraction.denominator = 0;
      try {
        pops::SolveOutcome pending =
            context->solve_fields_from_state_at(point, "field/partitioned", 0, stage);
        (void)pending.consume(pops::SolveConsumption::kFailRun);
      } catch (const std::exception&) {
        invalid_point_rejected = true;
      }
    });
    require(invalid_point_rejected,
            "a rank-local invalid evaluation point is rejected collectively before FAC entry");
    require(same_hierarchy<Dim>(published_hierarchy(system, output_key), accepted_before_failures),
            "invalid request rejection preserves every accepted field-output level");

    rhs_mode = RhsMode::throw_rank_zero;
    bool local_callback_rejected = false;
    context->with_program_resource_level(1, [&] {
      pops::MultiFab<Dim> stage = context->scratch_state_like(context->state(0));
      stage.set_val(pops::Real(3));
      try {
        pops::SolveOutcome pending = context->solve_fields_from_state_at(
            evaluation_point<Dim>(1, 4), "field/partitioned", 0, stage);
        (void)pending.consume(pops::SolveConsumption::kFailRun);
      } catch (const std::exception&) {
        local_callback_rejected = true;
      }
    });
    require(local_callback_rejected,
            "a rank-local RHS exception is rejected collectively before FAC entry");
    require(same_hierarchy<Dim>(published_hierarchy(system, output_key), accepted_before_failures),
            "rank-local RHS failure preserves every accepted field-output level");

    rhs_mode = RhsMode::nonfinite_rank_zero;
    bool local_nonfinite_rejected = false;
    context->with_program_resource_level(1, [&] {
      pops::MultiFab<Dim> stage = context->scratch_state_like(context->state(0));
      stage.set_val(pops::Real(3));
      try {
        pops::SolveOutcome pending = context->solve_fields_from_state_at(
            evaluation_point<Dim>(1, 5), "field/partitioned", 0, stage);
        (void)pending.consume(pops::SolveConsumption::kFailRun);
      } catch (const std::exception&) {
        local_nonfinite_rejected = true;
      }
    });
    require(local_nonfinite_rejected,
            "a rank-local non-finite RHS is rejected collectively before FAC entry");
    require(same_hierarchy<Dim>(published_hierarchy(system, output_key), accepted_before_failures),
            "rank-local non-finite RHS preserves every accepted field-output level");
    require(
        maximum_difference<Dim>(system.block_level_state_global("tracer", 0), state_before) == 0.0,
        "all failed field solves preserve the conservative state");

    rhs_mode = RhsMode::finite;
    context->with_program_resource_level(1, [&] {
      pops::MultiFab<Dim> stage = context->scratch_state_like(context->state(0));
      stage.set_val(pops::Real(3));
      pops::SolveOutcome pending = context->solve_fields_from_state_at(
          evaluation_point<Dim>(1, 6), "field/partitioned", 0, stage);
      const pops::SolveReport accepted = pending.consume(pops::SolveConsumption::kAccept);
      require(accepted.solved(), "the finite partitioned FAC retry is accepted");
    });

    require(system.field_provider_levels("field/partitioned") == 2,
            "the public provider owns both composite levels");
    for (int level = 0; level < 2; ++level) {
      const std::vector<double> published = system.auxiliary_component(output_key, level);
      require(!published.empty(), "acceptance publishes every composite level");
      require(std::any_of(published.begin(), published.end(),
                          [](double value) { return std::fabs(value) > 1.0e-10; }),
              "published potential is non-trivial");
    }
    require(!same_hierarchy<Dim>(published_hierarchy(system, output_key), accepted_before_failures),
            "accepted retry replaces the prior accepted field-output hierarchy");
  } catch (const std::exception& error) {
    std::fprintf(stderr, "rank %d: public partitioned composite failed: %s\n", pops::my_rank(),
                 error.what());
    ++failures;
  } catch (...) {
    std::fprintf(stderr, "rank %d: public partitioned composite failed with unknown error\n",
                 pops::my_rank());
    ++failures;
  }
  return failures;
}

template <int Dim>
long run_public_singular_replicated_parent() {
  long failures = 0;
  const auto require = [&failures](bool condition, const char* expectation) {
    if (!condition) {
      std::fprintf(stderr, "rank %d: unmet singular composite expectation: %s\n", pops::my_rank(),
                   expectation);
      ++failures;
    }
  };
  try {
    auto config = distributed_two_level_config<Dim>();
    config.distribute_coarse = false;
    pops::AmrSystem<Dim> system(config);
    const pops::AmrFieldHierarchyPolicyAuthority composite{
        "pops.field-hierarchy.composite", 1, {"pops.field-hierarchy.options.empty@1", {}}};
    system.set_field_solver_plan("field/singular", "test.mpi.singular-composite.plan@1",
                                 "test.mpi.singular-composite.rhs@1", "test.mpi.singular-composite",
                                 "tracer", "phi", {"test.mpi.singular-composite.rhs@1"}, {"tracer"},
                                 {"phi"}, {1.0}, "geometric_mg", composite,
                                 pops::geometric_mg_amr_field_solver_options(
                                     pops::GeometricMgOptions{}, pops::CompositeFacOptions{}));
    system.set_field_nullspace(
        "field/singular", "pops.field-nullspace.operator-topology-derived",
        pops::PreparedProviderOptions{"pops.field-nullspace.operator-topology-derived.options@1",
                                      {{"gauge.value", 0.0}}});
    system.install_block_state_route("tracer", "test.mpi.singular-composite/state/tracer");
    add_scalar_field_block(system);
    const auto output_key = install_field_output(system);
    system.register_elliptic_field("tracer", "phi", {output_key}, 1);

    enum class RhsMode { compatible_wave, incompatible_constant };
    RhsMode rhs_mode = RhsMode::compatible_wave;
    system.set_block_elliptic_field(
        "tracer", "phi", [&rhs_mode](const pops::MultiFab<Dim>&, pops::MultiFab<Dim>& rhs) {
          if (rhs_mode == RhsMode::incompatible_constant) {
            rhs.set_val(pops::Real(1));
            return;
          }
          int lower = std::numeric_limits<int>::max();
          int upper = std::numeric_limits<int>::lowest();
          for (const pops::Box<Dim>& patch : rhs.layout().boxes()) {
            lower = std::min(lower, patch.lo[0]);
            upper = std::max(upper, patch.hi[0]);
          }
          const pops::Real period = static_cast<pops::Real>(upper - lower + 1);
          constexpr pops::Real two_pi = pops::Real(6.283185307179586476925286766559);
          for (std::size_t local = 0; local < rhs.local_size(); ++local) {
            const auto values = rhs.fab(local).view();
            pops::for_each_cell(rhs.box(local), [=] POPS_HD(const pops::Index<Dim>& cell) {
              values(cell, 0) += Kokkos::sin(
                  two_pi * (static_cast<pops::Real>(cell[0] - lower) + pops::Real(0.5)) / period);
            });
          }
          Kokkos::fence();
        });
    system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
    pops::test::install_prepared_threshold_union(system, {{"tracer", "u", 0.5}},
                                                 "test.mpi.singular-composite.tagging@1");
    system.begin_bootstrap_plan();
    if (!system.bootstrap_next_level()) {
      system.rollback_bootstrap_level();
      throw std::runtime_error("singular composite fixture did not create L1");
    }
    system.commit_bootstrap_level();
    auto* engine = system.engine();
    if (engine == nullptr)
      throw std::runtime_error("singular composite fixture lost its AMR engine");
    rebuild_with_cross_owner_fine_patches<Dim>(system, *engine, config.transition_ratios.front());
    engine = system.engine();
    system.set_program_block_map({0});
    require(engine->hierarchy().layout(0).distribution().replicated(),
            "the coarse parent is replicated");
    require(partitioned_on_every_rank(engine->hierarchy().state(1)),
            "the fine level is partitioned");

    auto context = pops::runtime::program::make_program_execution_provider(&system);
    context->configure_primary_clock("test.mpi.partitioned-composite.clock");
    context->begin_step(0.01);
    context->with_program_resource_level(1, [&] {
      pops::MultiFab<Dim> stage = context->scratch_state_like(context->state(0));
      stage.set_val(pops::Real(1));
      pops::SolveOutcome pending = context->solve_fields_from_state_at(evaluation_point<Dim>(1, 20),
                                                                       "field/singular", 0, stage);
      const bool solved = pending.report().solved();
      const pops::SolveReport report =
          pending.consume(solved ? pops::SolveConsumption::kAccept
                                 : (pending.report().action == pops::SolveAction::kFailRun
                                        ? pops::SolveConsumption::kFailRun
                                        : pops::SolveConsumption::kRejectAttempt));
      require(report.solved(), "the compatible singular composite solve converges");
    });
    const auto accepted = published_hierarchy(system, output_key);
    require(std::any_of(accepted[1].begin(), accepted[1].end(),
                        [](double value) { return std::fabs(value) > 1.0e-9; }),
            "the partitioned fine publication is non-trivial");
    for (const auto& level : accepted) {
      double sum = 0.0;
      for (double value : level)
        sum += value;
      require(std::fabs(sum) <= 1.0e-8 * std::max<std::size_t>(1, level.size()),
              "the published singular solution has the deterministic zero-mean gauge");
    }

    rhs_mode = RhsMode::incompatible_constant;
    bool incompatible_rejected = false;
    context->with_program_resource_level(1, [&] {
      pops::MultiFab<Dim> stage = context->scratch_state_like(context->state(0));
      stage.set_val(pops::Real(1));
      try {
        pops::SolveOutcome pending = context->solve_fields_from_state_at(
            evaluation_point<Dim>(1, 21), "field/singular", 0, stage);
        incompatible_rejected = pending.report().status == pops::SolveStatus::kIncompatibleRhs;
        if (pending.report().solved())
          (void)pending.consume(pops::SolveConsumption::kAccept);
        else
          (void)pending.consume(pending.report().action == pops::SolveAction::kFailRun
                                    ? pops::SolveConsumption::kFailRun
                                    : pops::SolveConsumption::kRejectAttempt);
      } catch (const std::exception&) {
        incompatible_rejected = true;
      }
    });
    require(incompatible_rejected, "an incompatible singular RHS is refused collectively");
    require(same_hierarchy<Dim>(published_hierarchy(system, output_key), accepted),
            "incompatible singular refusal preserves accepted publication");

    rhs_mode = RhsMode::compatible_wave;
    context->with_program_resource_level(1, [&] {
      pops::MultiFab<Dim> stage = context->scratch_state_like(context->state(0));
      stage.set_val(pops::Real(1));
      pops::SolveOutcome pending = context->solve_fields_from_state_at(evaluation_point<Dim>(1, 22),
                                                                       "field/singular", 0, stage);
      const bool solved = pending.report().solved();
      const pops::SolveReport report =
          pending.consume(solved ? pops::SolveConsumption::kAccept
                                 : (pending.report().action == pops::SolveAction::kFailRun
                                        ? pops::SolveConsumption::kFailRun
                                        : pops::SolveConsumption::kRejectAttempt));
      require(report.solved(), "the compatible singular retry converges");
    });
  } catch (const std::exception& error) {
    std::fprintf(stderr, "rank %d: public singular composite failed: %s\n", pops::my_rank(),
                 error.what());
    ++failures;
  }
  return failures;
}

int run_field_plan_consensus(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 1;
  {
#if defined(POPS_HAS_KOKKOS)
    Kokkos::ScopeGuard guard(argc, argv);
#endif
    result = pops::n_ranks() == 2 &&
                     prove_newton_uses_duplicated_lane<pops::kNativeDimension>() == 0 &&
                     run_public_partitioned_composite<pops::kNativeDimension>() == 0 &&
                     run_public_singular_replicated_parent<pops::kNativeDimension>() == 0
                 ? 0
                 : 1;
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_field_plan_consensus, PublicPartitionedCompositeAcceptsAndRollsBack) {
  EXPECT_EQ(pops::test::RunTestBody(&run_field_plan_consensus, "test_mpi_field_plan_consensus"), 0);
}
