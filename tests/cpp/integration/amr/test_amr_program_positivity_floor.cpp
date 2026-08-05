// End-to-end positivity-floor coverage through the installed AMR Program.
//
// The local coarse/fine interpolation primitive is covered separately by test_cf_interface.  This
// test deliberately starts from the public AmrSystem facade, materializes a genuine two-level
// hierarchy, installs the same explicit Program used by compiled AMR tests, and advances it through
// AmrSystem::step.  The comparison with an otherwise identical floor-disabled run proves that the
// block's positivity option reaches the face reconstruction evaluated by ProgramGraph on the refined
// level; merely clamping an isolated ghost buffer cannot make the accepted trajectories differ.

#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"

#include <pops/core/state/state.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>

#include "amr_tagging_test_authority.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

using pops::Real;

struct DensityAdvection {
  using State = pops::StateVec<1>;
  using Prim = State;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  static constexpr int dimension = pops::kNativeDimension;

  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr.density-advection", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder&) const {}

  POPS_HD State flux(const State& state, const auto&, int direction) const {
    return direction == 0 ? state : State{Real(0)};
  }
  POPS_HD Real max_wave_speed(const State&, const auto&, int direction) const {
    return direction == 0 ? Real(1) : Real(0);
  }
  POPS_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
  POPS_HD Prim to_primitive(const State& state) const { return state; }
  POPS_HD State to_conservative(const Prim& primitive) const { return primitive; }

  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"rho"}, 1, {pops::VariableRole::Density}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"rho"}, 1, {pops::VariableRole::Density}};
  }
};

constexpr int kCells = 48;
constexpr double kFloor = 1e-6;

std::vector<double> contrast_state() {
  std::vector<double> density(static_cast<std::size_t>(kCells) * kCells, kFloor);
  const int band_lo = kCells / 3;
  const int band_hi = 2 * kCells / 3;
  const int spike = 3 * kCells / 5;
  for (int j = 0; j < kCells; ++j)
    for (int i = band_lo; i < band_hi; ++i)
      density[static_cast<std::size_t>(j) * kCells + i] = 1.0;

  // WENO5-Z reconstructs a sub-floor trace on this positive, non-monotone stencil.  Keep it inside
  // the deterministic central fine seed, so the floor must participate in a refined-level rate.
  for (int j = 0; j < kCells; ++j) {
    const std::size_t row = static_cast<std::size_t>(j) * kCells;
    density[row + spike] = 0.8;
    density[row + spike + 1] = 0.5;
    density[row + spike + 2] = kFloor;
    density[row + spike + 3] = 1.0;
    density[row + spike + 4] = kFloor;
  }
  return density;
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
  double mass_before = 0.0;
  double mass_after = 0.0;
  std::vector<double> fine_before;
  std::vector<double> fine_after;
  std::vector<double> fine_interior_before;
  std::vector<double> fine_interior_after;
};

RunResult advance_with_floor(double positivity_floor) {
  pops::AmrSystemConfig config;
  config.n = kCells;
  config.L = 1.0;
  config.level_count = 2;
  config.regrid_every = 0;
  config.periodicity = {true, true};

  pops::AmrSystem system(config);
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  pops::add_compiled_model(system, "density", DensityAdvection{}, "weno5", "rusanov",
                           "conservative", "euler", 1.4, 1, 1, {}, {}, positivity_floor);
  system.set_poisson("charge_density", "geometric_mg", "periodic");
  pops::test::install_prepared_threshold_union(system, {{"density", "rho", 1e29}});
  system.set_conservative_state("density", contrast_state());
  pops::test::install_forward_euler_program(system);

  RunResult result;
  result.levels = system.n_levels();
  pops::AmrRuntime& runtime = *system.engine();
  const pops::MultiFab& fine = runtime.levels(0)[1].U;
  result.fine_patches = static_cast<int>(fine.box_array().size());
  const pops::Box2D fine_domain = runtime.level_geom(1).domain;
  const std::size_t fine_nx = static_cast<std::size_t>(fine_domain.nx());
  std::vector<std::size_t> fine_interior_indices;
  for (const pops::Box2D& patch : fine.box_array().boxes()) {
    // One Forward-Euler residual cannot propagate a coarse FillPatch value farther than the WENO5
    // stencil plus its face divergence. Values this far inside a valid patch therefore isolate the
    // fine-level reconstruction itself from the coarse/fine halo.
    const pops::Box2D interior = patch.grow(-4);
    for (int j = interior.lo[1]; j <= interior.hi[1]; ++j)
      for (int i = interior.lo[0]; i <= interior.hi[0]; ++i)
        fine_interior_indices.push_back(static_cast<std::size_t>(j - fine_domain.lo[1]) * fine_nx +
                                        static_cast<std::size_t>(i - fine_domain.lo[0]));
  }

  auto select_fine_interior = [&](const std::vector<double>& state) {
    std::vector<double> selected;
    selected.reserve(fine_interior_indices.size());
    for (std::size_t index : fine_interior_indices)
      selected.push_back(state[index]);
    return selected;
  };

  result.mass_before = system.mass("density");
  result.fine_before = system.block_level_state_global("density", 1);
  result.fine_interior_before = select_fine_interior(result.fine_before);

  system.step(2e-4);

  result.mass_after = system.mass("density");
  result.fine_after = system.block_level_state_global("density", 1);
  result.fine_interior_after = select_fine_interior(result.fine_after);
  return result;
}

}  // namespace

TEST(test_amr_program_positivity_floor, RefinedProgramTrajectoryUsesTheFloor) {
#if defined(POPS_HAS_KOKKOS)
  int argc = 0;
  char** argv = nullptr;
  Kokkos::ScopeGuard guard(argc, argv);
#endif

  const RunResult unfloored = advance_with_floor(0.0);
  const RunResult floored = advance_with_floor(kFloor);

  ASSERT_EQ(unfloored.levels, 2);
  ASSERT_EQ(floored.levels, 2);
  ASSERT_GT(unfloored.fine_patches, 0);
  ASSERT_EQ(floored.fine_patches, unfloored.fine_patches);
  ASSERT_FALSE(unfloored.fine_before.empty());
  ASSERT_EQ(floored.fine_before, unfloored.fine_before);
  ASSERT_EQ(floored.fine_after.size(), unfloored.fine_after.size());
  ASSERT_FALSE(unfloored.fine_interior_before.empty());
  ASSERT_EQ(floored.fine_interior_before, unfloored.fine_interior_before);
  ASSERT_EQ(floored.fine_interior_after.size(), unfloored.fine_interior_after.size());

  EXPECT_TRUE(all_finite(unfloored.fine_after));
  EXPECT_TRUE(all_finite(floored.fine_after));
  EXPECT_GT(max_difference(floored.fine_interior_after, floored.fine_interior_before), 0.0)
      << "the installed Program must advance the refined level";
  EXPECT_GT(max_difference(floored.fine_interior_after, unfloored.fine_interior_after), 0.0)
      << "positivity_floor must alter valid fine cells beyond any coarse/fine halo influence";

  EXPECT_NEAR(unfloored.mass_after, unfloored.mass_before, 1e-11);
  EXPECT_NEAR(floored.mass_after, floored.mass_before, 1e-11);
}
