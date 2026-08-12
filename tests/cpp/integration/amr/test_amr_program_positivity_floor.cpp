// End-to-end positivity-floor coverage through one exact-ranked AMR Program.
//
// A qualified ComponentKey supplies the transport speed through ProviderValues; no physical-name
// auxiliary slab participates.  The Program advances a genuine sparse refined level, accepts a
// fallible implicit-source SolveOutcome on that distributed level, and is replayed after an outer
// transaction rollback. Comparing the floor-enabled and floor-disabled accepted trajectories proves
// that the density floor reaches the WENO face reconstruction evaluated on valid fine cells rather
// than merely clamping a halo.

#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/core/state/state.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/fv/flux_interfaces.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

constexpr int kCells = 32;
constexpr double kFloor = 1e-6;
constexpr char kProviderOwner[] = "test.amr.positivity";
constexpr char kProviderSpaceKind[] = "input";
constexpr char kProviderSpaceName[] = "transport";
constexpr char kProviderComponent[] = "speed";
constexpr char kProviderRepresentation[] = "cell-average";
constexpr char kProviderCentering[] = "cell";
constexpr char kProviderUnit[] = "length/time";
constexpr char kProviderLayout[] = "amr-transport";
constexpr char kProviderValueKind[] = "scalar";
constexpr char kProviderIdentity[] = "test.amr.positivity.speed@1";
constexpr char kProviderConsumer[] = "test.amr.positivity/physical-flux";

template <int Dim>
struct DensityAdvection {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_providers = 1;
  static constexpr int n_flux_providers = 1;
  static constexpr std::array<pops::QualifiedProviderRequirement, 1> flux_provider_requirements{{
      {kProviderOwner, kProviderSpaceKind, kProviderSpaceName, kProviderComponent,
       kProviderRepresentation, kProviderCentering, kProviderUnit, kProviderLayout,
       kProviderValueKind, kProviderIdentity, true, 0},
  }};

  Law conversion{};

  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr.density-advection", 2};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{Dim});
  }

  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"rho"}, 1, {pops::VariableRole::Density}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"rho"}, 1, {pops::VariableRole::Density}};
  }
  POPS_HD pops::nd::StateConversion<Primitive> recover(const State& state) const {
    return conversion.recover(state);
  }
  POPS_HD pops::nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return conversion.make_conservative(primitive);
  }
  POPS_HD pops::nd::StateConversionStatus admissibility(const State& state) const {
    return conversion.admissibility(state);
  }

  template <int Axis>
  POPS_HD State flux(const State& state,
                     const pops::BoundFluxProviders<DensityAdvection<Dim>>& providers) const {
    static_assert(Axis >= 0 && Axis < Dim);
    return State{providers.template provider<0>() * state[Schema::scalar]};
  }
  template <int Axis>
  POPS_HD pops::Real max_wave_speed(
      const State&, const pops::BoundFluxProviders<DensityAdvection<Dim>>& providers) const {
    static_assert(Axis >= 0 && Axis < Dim);
    const pops::Real speed = providers.template provider<0>();
    return speed < pops::Real(0) ? -speed : speed;
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State&,
                           const pops::BoundFluxProviders<DensityAdvection<Dim>>& providers,
                           pops::Real& lower, pops::Real& upper) const {
    static_assert(Axis >= 0 && Axis < Dim);
    const pops::Real speed = providers.template provider<0>();
    lower = speed;
    upper = speed;
  }
  POPS_HD State source(const State&, const pops::ProviderValues<1>&) const { return State{}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

template <int Dim>
std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
pops::AmrSystemConfig<Dim> config() {
  pops::AmrSystemConfig<Dim> result;
  result.level_count = 2;
  result.regrid_every = 0;
  result.explicit_bootstrap = true;
  result.distribute_coarse = true;
  for (int axis = 0; axis < Dim; ++axis) {
    result.shape[axis] = kCells;
    result.lower[axis] = pops::Real(0);
    result.upper[axis] = pops::Real(1);
    result.periodicity[axis] = true;
    result.coarse_max_grid[axis] = kCells / 2;
    result.transition_buffers.front()[axis] = 3;
    result.transition_lookaheads.front()[axis] = 4;
  }
  return result;
}

template <int Dim>
std::vector<double> contrast_state(const pops::Extent<Dim>& shape) {
  std::vector<double> density(cell_count(shape), kFloor);
  const int band_lo = kCells / 3;
  const int band_hi = 2 * kCells / 3;
  const int spike = 3 * kCells / 5;
  for (std::size_t linear = 0; linear < density.size(); ++linear) {
    const int first_axis = static_cast<int>(linear % static_cast<std::size_t>(shape[0]));
    if (first_axis >= band_lo && first_axis < band_hi)
      density[linear] = 1.0;
    if (first_axis == spike)
      density[linear] = 0.8;
    else if (first_axis == spike + 1)
      density[linear] = 0.5;
    else if (first_axis == spike + 2 || first_axis == spike + 4)
      density[linear] = kFloor;
    else if (first_axis == spike + 3)
      density[linear] = 1.0;
  }
  return density;
}

template <int Dim>
pops::runtime::system::AuxiliaryComponentKey install_transport_speed(pops::AmrSystem<Dim>& system) {
  using namespace pops::runtime::system;
  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 3;
  const AuxiliaryComponentKey key{kProviderOwner, kProviderSpaceKind, kProviderSpaceName,
                                  kProviderComponent};
  const AuxiliaryComponentContract contract{kProviderRepresentation, kProviderCentering,
                                            kProviderUnit, kProviderLayout, kProviderValueKind};
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      kProviderIdentity,
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      {{key, contract, shape}},
      {}});
  system.install_auxiliary_consumer_plan(
      AuxiliaryConsumerProviderPlan<Dim>{kProviderConsumer, {{{key, contract, shape}, 0}}});
  system.seal_auxiliary_providers();
  return key;
}

struct ProgramEvidence {
  int accepted_source_solves = 0;
  int nonzero_block_provider_binds = 0;
};

template <int Dim>
std::shared_ptr<ProgramEvidence> install_program(pops::AmrSystem<Dim>& system) {
  auto context = pops::runtime::program::make_program_execution_provider(&system);
  auto evidence = std::make_shared<ProgramEvidence>();
  auto lane = std::make_shared<pops::ExecutionLane>(
      pops::ExecutionLane::world("pops.test.amr-program-positivity-floor"));
  context->configure_primary_clock("test.amr.positivity.clock");
  context->install([context, evidence, lane](double macro_dt) {
    context->begin_step(macro_dt);
    context->for_each_program_resource_level([context, evidence, lane, macro_dt](int selected) {
      context->set_stage_time(0, 1);
      std::vector<pops::MultiFab<Dim>*> states;
      std::vector<pops::MultiFab<Dim>*> residuals;
      states.reserve(static_cast<std::size_t>(context->n_blocks()));
      residuals.reserve(static_cast<std::size_t>(context->n_blocks()));
      for (int block = 0; block < context->n_blocks(); ++block) {
        pops::MultiFab<Dim>& state = context->state(block);
        if (selected > 0) {
          const auto provider_at = [context, block](std::size_t local_patch) {
            return context->template provider_values_view<1>(kProviderConsumer, block, local_patch);
          };
          if (block != 0)
            ++evidence->nonzero_block_provider_binds;
          pops::SolveOutcome source =
              pops::backward_euler_source(DensityAdvection<Dim>{}, provider_at, state,
                                          pops::Real(macro_dt), pops::NewtonOptions{}, *lane);
          const pops::SolveReport accepted = source.consume(pops::SolveConsumption::kAccept);
          if (!accepted.solved())
            throw std::runtime_error("positivity test Program source solve failed: " +
                                     accepted.reason);
          ++evidence->accepted_source_solves;
        }
        pops::MultiFab<Dim>& residual = context->rhs_scratch(1000 + block, 0, state);
        context->rhs_into(block, state, residual, 3000 + block);
        states.push_back(&state);
        residuals.push_back(&residual);
      }
      for (std::size_t block = 0; block < states.size(); ++block)
        context->axpy(*states[block], pops::Real(macro_dt), *residuals[block]);
    });
  });
  system.set_program_block_map({0, 1});
  return evidence;
}

bool all_finite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

double max_difference(const std::vector<double>& left, const std::vector<double>& right) {
  if (left.size() != right.size())
    return std::numeric_limits<double>::infinity();
  double difference = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index)
    difference = std::max(difference, std::fabs(left[index] - right[index]));
  return difference;
}

struct RunResult {
  int levels = 0;
  int fine_patches = 0;
  int accepted_source_solves = 0;
  int nonzero_block_provider_binds = 0;
  bool provider_values_exact = false;
  double mass_before = 0.0;
  double mass_trial = 0.0;
  double mass_rolled_back = 0.0;
  double mass_after = 0.0;
  std::vector<double> fine_before;
  std::vector<double> fine_trial;
  std::vector<double> fine_rolled_back;
  std::vector<double> fine_after;
  std::vector<double> fine_interior_before;
  std::vector<double> fine_interior_after;
};

template <int Dim>
RunResult advance_with_floor(double positivity_floor) {
  const pops::AmrSystemConfig<Dim> system_config = config<Dim>();
  pops::AmrSystem<Dim> system(system_config);
  const auto speed_key = install_transport_speed(system);
  system.install_block_state_route("density", "test.amr.positivity/state/density");
  system.install_block_state_route("density_peer", "test.amr.positivity/state/density-peer");
  pops::add_compiled_model<Dim>(system, "density", DensityAdvection<Dim>{}, "weno5", "rusanov",
                                "conservative", "explicit", 1.4, 1, 1, {}, {}, positivity_floor,
                                static_cast<double>(pops::kWenoEpsilon), false, kProviderConsumer);
  pops::add_compiled_model<Dim>(system, "density_peer", DensityAdvection<Dim>{}, "weno5", "rusanov",
                                "conservative", "explicit", 1.4, 1, 1, {}, {}, positivity_floor,
                                static_cast<double>(pops::kWenoEpsilon), false, kProviderConsumer);
  pops::test::install_prepared_threshold_union(system, {{"density", "rho", 0.25}},
                                               "test.amr.positivity.tagging@1");
  constexpr const char* state_route = "test.amr.positivity/state/density";
  constexpr const char* peer_state_route = "test.amr.positivity/state/density-peer";
  system.bind_bootstrap_subject(state_route, "density", "bound_level_zero");
  system.bind_bootstrap_subject(peer_state_route, "density_peer", "bound_level_zero");
  system.stage_bootstrap_array(state_route, "density", "cell", "cell", 1, system_config.shape,
                               contrast_state(system_config.shape));
  system.stage_bootstrap_array(peer_state_route, "density_peer", "cell", "cell", 1,
                               system_config.shape, contrast_state(system_config.shape));
  pops::Extent<Dim> transfer_ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    transfer_ghosts[axis] = 1;
  system.register_bootstrap_transfer_route("test.amr.positivity/bootstrap/prolongation",
                                           {state_route}, "test.amr.positivity/bootstrap/provider",
                                           "cell", "cell", "conservative", "dense", "prolongation",
                                           "conservative_linear", 2, transfer_ghosts,
                                           system_config.transition_ratios.front());
  system.register_bootstrap_transfer_route(
      "test.amr.positivity/bootstrap/peer-prolongation", {peer_state_route},
      "test.amr.positivity/bootstrap/provider", "cell", "cell", "conservative", "dense",
      "prolongation", "conservative_linear", 2, transfer_ghosts,
      system_config.transition_ratios.front());
  const std::vector<double> speed(cell_count(system_config.shape), 1.0);
  system.stage_auxiliary_input(speed_key, speed);
  system.refresh_auxiliary({"test.amr.positivity.clock", 0, 0, 0, 0, 0, 0,
                            pops::runtime::system::AuxiliaryEvaluationEvent::initialization});
  system.begin_bootstrap_plan();
  (void)system.materialize_bootstrap_action(state_route, "initialize_level_zero",
                                            "bound_level_zero", 0);
  (void)system.materialize_bootstrap_action(peer_state_route, "initialize_level_zero",
                                            "bound_level_zero", 0);
  if (!system.bootstrap_next_level()) {
    system.rollback_bootstrap_level();
    throw std::runtime_error("positivity test tagging did not create its refined level");
  }
  (void)system.materialize_bootstrap_action(state_route, "prolong_from_parent",
                                            "conservative_linear", 1);
  (void)system.materialize_bootstrap_action(peer_state_route, "prolong_from_parent",
                                            "conservative_linear", 1);
  system.commit_bootstrap_level();
  system.stage_auxiliary_input(speed_key, speed);
  system.refresh_auxiliary({"test.amr.positivity.clock", 0, 0, 0, 0, 0, 0,
                            pops::runtime::system::AuxiliaryEvaluationEvent::initialization});
  const std::shared_ptr<ProgramEvidence> evidence = install_program(system);

  RunResult result;
  result.levels = system.n_levels();
  auto& runtime = *system.engine();
  const pops::MultiFab<Dim>& fine = runtime.hierarchy().state(1);
  result.fine_patches = static_cast<int>(fine.layout().size());
  const pops::Box<Dim>& fine_domain = runtime.hierarchy().layout(1).domain();
  std::vector<std::size_t> fine_interior_indices;
  for (const pops::Box<Dim>& patch : fine.layout().boxes()) {
    const pops::Box<Dim> interior = patch.grow(-4);
    if (interior.empty())
      continue;
    for (std::int64_t ordinal = 0; ordinal < interior.numPts(); ++ordinal) {
      std::int64_t remainder = ordinal;
      pops::Index<Dim> cell{};
      for (int axis = 0; axis < Dim; ++axis) {
        cell[axis] = interior.lo[axis] + static_cast<int>(remainder % interior.length(axis));
        remainder /= interior.length(axis);
      }
      std::size_t linear = 0;
      std::size_t stride = 1;
      for (int axis = 0; axis < Dim; ++axis) {
        linear += static_cast<std::size_t>(cell[axis] - fine_domain.lo[axis]) * stride;
        stride *= static_cast<std::size_t>(fine_domain.length(axis));
      }
      fine_interior_indices.push_back(linear);
    }
  }
  const auto select_fine_interior = [&](const std::vector<double>& state) {
    std::vector<double> selected;
    selected.reserve(fine_interior_indices.size());
    for (std::size_t index : fine_interior_indices)
      selected.push_back(state[index]);
    return selected;
  };

  const auto provider_values = system.auxiliary_component(speed_key, 1);
  result.provider_values_exact =
      !fine_interior_indices.empty() &&
      std::all_of(fine_interior_indices.begin(), fine_interior_indices.end(),
                  [&](std::size_t index) {
                    return index < provider_values.size() && provider_values[index] == 1.0;
                  });
  result.mass_before = system.mass("density");
  result.fine_before = system.block_level_state_global("density", 1);
  result.fine_interior_before = select_fine_interior(result.fine_before);

  system.begin_step_transaction();
  system.step(2e-4);
  result.mass_trial = system.mass("density");
  result.fine_trial = system.block_level_state_global("density", 1);
  system.rollback_step_transaction();
  result.mass_rolled_back = system.mass("density");
  result.fine_rolled_back = system.block_level_state_global("density", 1);

  system.step(2e-4);
  result.mass_after = system.mass("density");
  result.fine_after = system.block_level_state_global("density", 1);
  result.fine_interior_after = select_fine_interior(result.fine_after);
  result.accepted_source_solves = evidence->accepted_source_solves;
  result.nonzero_block_provider_binds = evidence->nonzero_block_provider_binds;
  return result;
}

template <int Dim>
void verify_exact_ranked_trajectory() {
  const RunResult unfloored = advance_with_floor<Dim>(0.0);
  const RunResult floored = advance_with_floor<Dim>(kFloor);

  EXPECT_EQ(unfloored.levels, 2);
  EXPECT_EQ(floored.levels, 2);
  EXPECT_GT(unfloored.fine_patches, 0);
  EXPECT_EQ(floored.fine_patches, unfloored.fine_patches);
  EXPECT_TRUE(unfloored.provider_values_exact);
  EXPECT_TRUE(floored.provider_values_exact);
  EXPECT_FALSE(unfloored.fine_before.empty());
  EXPECT_EQ(floored.fine_before, unfloored.fine_before);
  EXPECT_FALSE(unfloored.fine_interior_before.empty());
  EXPECT_EQ(floored.fine_interior_before, unfloored.fine_interior_before);

  EXPECT_EQ(unfloored.fine_rolled_back, unfloored.fine_before);
  EXPECT_EQ(floored.fine_rolled_back, floored.fine_before);
  EXPECT_EQ(unfloored.fine_after, unfloored.fine_trial);
  EXPECT_EQ(floored.fine_after, floored.fine_trial);
  EXPECT_EQ(unfloored.accepted_source_solves, 4);
  EXPECT_EQ(floored.accepted_source_solves, 4);
  EXPECT_EQ(unfloored.nonzero_block_provider_binds, 2);
  EXPECT_EQ(floored.nonzero_block_provider_binds, 2);

  EXPECT_TRUE(all_finite(unfloored.fine_after));
  EXPECT_TRUE(all_finite(floored.fine_after));
  EXPECT_GT(max_difference(floored.fine_interior_after, floored.fine_interior_before), 0.0)
      << "the installed Program must advance valid cells on the refined level";
  EXPECT_GT(max_difference(floored.fine_interior_after, unfloored.fine_interior_after), 0.0)
      << "the density floor must alter fine cells beyond coarse/fine halo influence";

  constexpr double tolerance = 2e-10;
  EXPECT_NEAR(unfloored.mass_rolled_back, unfloored.mass_before, tolerance);
  EXPECT_NEAR(floored.mass_rolled_back, floored.mass_before, tolerance);
  EXPECT_NEAR(unfloored.mass_trial, unfloored.mass_after, tolerance);
  EXPECT_NEAR(floored.mass_trial, floored.mass_after, tolerance);
  EXPECT_NEAR(unfloored.mass_after, unfloored.mass_before, tolerance);
  EXPECT_NEAR(floored.mass_after, floored.mass_before, tolerance);
}

}  // namespace

TEST(test_amr_program_positivity_floor, RefinedProgramTrajectoryUsesFloorAndRollsBackExactly) {
#if defined(POPS_HAS_KOKKOS)
  int argc = 0;
  char** argv = nullptr;
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  verify_exact_ranked_trajectory<pops::kNativeDimension>();
}
