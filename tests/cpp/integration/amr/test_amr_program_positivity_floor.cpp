// End-to-end positivity-floor coverage through one exact-ranked AMR Program.
//
// A qualified ComponentKey supplies the transport speed through ProviderValues; no physical-name
// auxiliary slab participates.  The Program advances a genuine sparse refined level, accepts a
// fallible implicit-source SolveOutcome on that distributed level, and is replayed after an outer
// transaction rollback. Comparing the floor-enabled and floor-disabled accepted trajectories proves
// that the density floor reaches the WENO face reconstruction evaluated on valid fine cells rather
// than merely clamping a halo.

#include <gtest/gtest.h>

#include "amr_fixture_tag_mask_oracle.hpp"
#include "euler_amr_fixture_criteria.hpp"
#include "explicit_amr_program.hpp"
#include "positivity_floor_program_fixture.hpp"

#include <pops/runtime/dynamic/dynlib.hpp>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {
using namespace pops::test::positivity;

constexpr int kCells = 32;
constexpr double kFloor = 1e-6;
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

pops::runtime::system::AuxiliaryComponentKey transport_speed_key() {
  return {kProviderOwner, kProviderSpaceKind, kProviderSpaceName, kProviderComponent};
}

template <int Dim>
pops::runtime::system::AuxiliaryComponentKey install_transport_speed(pops::AmrSystem<Dim>& system) {
  using namespace pops::runtime::system;
  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 3;
  const auto key = transport_speed_key();
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
void bootstrap_density(pops::AmrSystem<Dim>& system,
                       const pops::AmrSystemConfig<Dim>& system_config, double positivity_floor,
                       const std::vector<double>& density,
                       const std::vector<double>& peer_density) {
  pops::test::install_amr_runtime_authority(system, "test.amr.positivity/runtime@1");
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  const auto speed_key = install_transport_speed(system);
  const auto& group = pops::test::positivity_floor;
  const auto& field = group.fields.front();
  const auto schema = DensityAdvection<Dim>::conservative_vars();
  ASSERT_STREQ(field.block, "density");
  ASSERT_STREQ(field.variable, "rho");
  ASSERT_EQ(field.component, 0);
  ASSERT_EQ(schema.names.at(0), field.variable);
  ASSERT_EQ(schema.roles.at(0), pops::VariableRole::Density);
  const char* state_route = field.state;
  system.install_block_state_route("density", state_route);
  system.install_block_state_route("density_peer", "test.amr.positivity/state/density-peer");
  pops::add_compiled_model<Dim>(system, "density", DensityAdvection<Dim>{}, "weno5", "rusanov",
                                "conservative", "explicit", 1.4, 1, 1, {}, {}, positivity_floor,
                                static_cast<double>(pops::kWenoEpsilon), false, kProviderConsumer);
  pops::add_compiled_model<Dim>(system, "density_peer", DensityAdvection<Dim>{}, "weno5", "rusanov",
                                "conservative", "explicit", 1.4, 1, 1, {}, {}, positivity_floor,
                                static_cast<double>(pops::kWenoEpsilon), false, kProviderConsumer);
  group.template install<Dim>(system, "test.amr.positivity.clock");
  constexpr const char* peer_state_route = "test.amr.positivity/state/density-peer";
  system.bind_bootstrap_subject(state_route, "density", "bound_level_zero");
  system.bind_bootstrap_subject(peer_state_route, "density_peer", "bound_level_zero");
  system.stage_bootstrap_array(state_route, "density", "cell", "cell", 1, system_config.shape,
                               density);
  system.stage_bootstrap_array(peer_state_route, "density_peer", "cell", "cell", 1,
                               system_config.shape, peer_density);
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
  ASSERT_FALSE(system.uses_runtime_engine());
  system.install_program(POPS_TEST_POSITIVITY_FLOOR_PROGRAM_SO);
  ASSERT_TRUE(system.uses_runtime_engine());
  ASSERT_EQ(system.n_levels(), 1);
  ASSERT_EQ(system.program_block_map(), (std::vector<int>{0, 1}));
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
  EXPECT_EQ(system.n_levels(), 2);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_EQ(system.time(), 0.);
  system.mark_bound();
}

template <int Dim>
RunResult advance_with_floor(double positivity_floor) {
  const auto system_config = config<Dim>();
  pops::AmrSystem<Dim> system(system_config);
  const auto density = contrast_state(system_config.shape);
  bootstrap_density(system, system_config, positivity_floor, density, density);
  if (::testing::Test::HasFatalFailure())
    throw std::runtime_error("positivity fixture failed explicit Program bootstrap");
  const pops::dynlib::UniqueHandle image(pops::dynlib::open(POPS_TEST_POSITIVITY_FLOOR_PROGRAM_SO));
  if (!pops::dynlib::valid(image.get()))
    throw std::runtime_error("cannot reopen the installed positivity fixture");
  const auto get = reinterpret_cast<void (*)(pops::AmrSystem<Dim>*, ProgramEvidence*)>(
      pops::dynlib::sym(image.get(), "pops_test_positivity_evidence"));
  if (get == nullptr)
    throw std::runtime_error("installed positivity fixture lacks its evidence export");
  ProgramEvidence evidence;
  get(&system, &evidence);
  EXPECT_EQ(evidence.accepted_source_solves, 0);
  EXPECT_EQ(evidence.nonzero_block_provider_binds, 0);
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

  const auto provider_values = system.auxiliary_component(transport_speed_key(), 1);
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
  get(&system, &evidence);
  result.accepted_source_solves = evidence.accepted_source_solves;
  result.nonzero_block_provider_binds = evidence.nonzero_block_provider_binds;
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
  // Two executions (trial plus replay), two fine substeps, and two blocks.
  EXPECT_EQ(unfloored.accepted_source_solves, 8);
  EXPECT_EQ(floored.accepted_source_solves, 8);
  EXPECT_EQ(unfloored.nonzero_block_provider_binds, 4);
  EXPECT_EQ(floored.nonzero_block_provider_binds, 4);

  EXPECT_TRUE(all_finite(unfloored.fine_after));
  EXPECT_TRUE(all_finite(floored.fine_after));
  EXPECT_GT(max_difference(floored.fine_interior_after, floored.fine_interior_before), 0.0)
      << "the installed Program must advance valid cells on the refined level";
  EXPECT_GT(max_difference(floored.fine_interior_after, unfloored.fine_interior_after), 0.0)
      << "the density floor must alter fine cells beyond coarse/fine halo influence";

  const auto exact_real = [](double value) {
    std::ostringstream encoded;
    encoded << std::hexfloat << value;
    return encoded.str();
  };
  ::testing::Test::RecordProperty("unfloored_mass_change",
                                  exact_real(unfloored.mass_after - unfloored.mass_before));
  ::testing::Test::RecordProperty("floored_mass_change",
                                  exact_real(floored.mass_after - floored.mass_before));
  ::testing::Test::RecordProperty(
      "fine_interior_floor_effect",
      exact_real(max_difference(floored.fine_interior_after, unfloored.fine_interior_after)));
  constexpr double tolerance = 2e-10;
  EXPECT_NEAR(unfloored.mass_rolled_back, unfloored.mass_before, tolerance);
  EXPECT_NEAR(floored.mass_rolled_back, floored.mass_before, tolerance);
  EXPECT_NEAR(unfloored.mass_trial, unfloored.mass_after, tolerance);
  EXPECT_NEAR(floored.mass_trial, floored.mass_after, tolerance);
  EXPECT_NEAR(unfloored.mass_after, unfloored.mass_before, tolerance);
  EXPECT_NEAR(floored.mass_after, floored.mass_before, tolerance);
}

template <int Dim>
void cold_contrast() {
  const pops::AmrSystemConfig<Dim> system_config = config<Dim>();
  pops::AmrSystem<Dim> system(system_config);

  const double stored_floor = static_cast<double>(static_cast<pops::Real>(kFloor));
  std::vector<double> density(cell_count(system_config.shape), stored_floor);
  std::vector<double> peer(density.size(), stored_floor);
  for (std::size_t cell = 0; cell < density.size(); ++cell) {
    const auto x = cell % kCells;
    if (x >= 5 && x < 11)
      density[cell] = 1.;
    if (x == 17)
      density[cell] = .25;
    if (x >= 23 && x < 27)
      peer[cell] = 1.;
  }
  ASSERT_NO_FATAL_FAILURE(bootstrap_density(system, system_config, kFloor, density, peer));
  ASSERT_EQ(system.n_levels(), 2);
  EXPECT_EQ(system.time(), 0.);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_EQ(system.block_level_state_global("density", 0), density);
  EXPECT_EQ(system.block_level_state_global("density_peer", 0), peer);
  const auto first = system.patch_boxes();
  const auto& fine = system.prepared_amr_block_state(0, 1);
  const auto& peer_fine = system.prepared_amr_block_state(1, 1);
  EXPECT_EQ(fine.layout(), peer_fine.layout());
  EXPECT_EQ(fine.distribution(), peer_fine.distribution());
  EXPECT_EQ(fine.local_size(), peer_fine.local_size());
  pops::Index<Dim> lower{}, upper{}, owner{};
  lower[0] = 10;
  upper[0] = 21;
  for (int axis = 1; axis < Dim; ++axis)
    upper[axis] = 63;
  EXPECT_EQ(first, (std::vector<pops::AmrPatch<Dim>>{{1, {lower, upper}}}));
  EXPECT_EQ(fine.distribution().owners(), (std::vector<pops::Index<Dim>>{owner}));
  EXPECT_EQ(fine.local_size(), fine.local_rank() == owner ? 1u : 0u);
  if (fine.local_size() != 0)
    EXPECT_EQ(fine.box(0), (pops::Box<Dim>{lower, upper}));
  const auto covered = [&](int x) {
    pops::Index<Dim> point{};
    point[0] = 2 * x;
    for (int axis = 1; axis < Dim; ++axis)
      point[axis] = 32;
    return std::any_of(first.begin(), first.end(), [&](const auto& patch) {
      return patch.level == 1 && patch.box.contains(point);
    });
  };
  EXPECT_TRUE(covered(7));
  EXPECT_FALSE(covered(17));
  EXPECT_FALSE(covered(24));
  const auto selected = system.execute_prepared_tagging(0);
  EXPECT_EQ(pops::all_reduce_sum(static_cast<long>(
                pops::test::expect_fixture_tag_mask(selected.refine, system_config.shape,
                                                    [](std::size_t cell) {
                                                      const auto x = cell % kCells;
                                                      return x >= 5 && x < 11;
                                                    }))),
            static_cast<long>(density.size()));
  EXPECT_EQ(pops::all_reduce_sum(static_cast<long>(pops::test::expect_fixture_tag_mask(
                selected.coarsen, system_config.shape, [](std::size_t) { return false; }))),
            static_cast<long>(density.size()));
  EXPECT_EQ(system.patch_boxes(), first);
}

}  // namespace

TEST(test_amr_program_positivity_floor, RefinedProgramTrajectoryUsesFloorAndRollsBackExactly) {
#if defined(POPS_HAS_KOKKOS)
  int argc = 0;
  char** argv = nullptr;
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  verify_exact_ranked_trajectory<pops::kNativeDimension>();
  ASSERT_NO_FATAL_FAILURE(cold_contrast<pops::kNativeDimension>());
}
