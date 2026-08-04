// ProgramGraph replacement for the retired low-level AMR diffusion driver test.
//
// A scalar heat model exposes its Fickian flux through diffusivity().  The hierarchy contains a
// strict fine subset of the periodic domain, so diffusion crosses a genuine coarse/fine interface.
// The explicit test Program is the only time authority: its captured face-flux ledger refluxes at
// every child catch-up before average-down.  We require both observable smoothing and round-off
// conservation of the level-composite integral.

#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include <pops/core/model/physical_model.hpp>
#include <pops/core/state/state.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>

#include "amr_tagging_test_authority.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

constexpr double kPi = 3.14159265358979323846;

struct DiffusiveScalar {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;

  Real nu = Real(0);

  POPS_HD State flux(const State&, const auto&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return Real(0); }
  POPS_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
  POPS_HD Real diffusivity() const { return nu; }

  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"temperature"}, 1, {VariableRole::Scalar}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"temperature"}, 1, {VariableRole::Scalar}};
  }
};

static_assert(PhysicalModel<DiffusiveScalar>);
static_assert(DiffusiveModel<DiffusiveScalar>);

std::vector<double> periodic_mode(int n) {
  std::vector<double> state(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (static_cast<double>(i) + 0.5) / static_cast<double>(n);
      const double y = (static_cast<double>(j) + 0.5) / static_cast<double>(n);
      state[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
            static_cast<std::size_t>(i)] =
          1.0 + 0.35 * std::cos(2.0 * kPi * (x - 0.17)) * std::cos(2.0 * kPi * (y + 0.11));
    }
  return state;
}

}  // namespace

TEST(test_amr_program_diffusion, RefinedFickianFluxSmoothsAndConservesThroughProgramReflux) {
#if defined(POPS_HAS_KOKKOS)
  int argc = 0;
  char** argv = nullptr;
  Kokkos::ScopeGuard guard(argc, argv);
#endif

  constexpr int n = 16;
  constexpr double dt = 1.0e-3;
  constexpr int steps = 5;

  AmrSystemConfig config;
  config.n = n;
  config.L = 1.0;
  config.level_count = 2;
  config.regrid_every = 0;
  config.periodicity = {true, true};

  AmrSystem simulation(config);
  add_compiled_model(simulation, "heat", DiffusiveScalar{Real(0.1)}, "none", "rusanov",
                     "conservative", "explicit");
  simulation.set_density("heat", periodic_mode(n));
  simulation.set_poisson("charge_density", "geometric_mg", "periodic");
  test::install_prepared_threshold_union(simulation, {{"heat", "temperature", 1.0e29}});
  simulation.set_temporal_relations({2}, {1}, {"integral_only"});
  test::install_forward_euler_program(simulation);

  ASSERT_TRUE(simulation.uses_runtime_engine());
  ASSERT_NE(simulation.engine(), nullptr);
  AmrRuntime& runtime = *simulation.engine();
  ASSERT_EQ(runtime.nlev(), 2);
  ASSERT_EQ(runtime.levels(0).size(), 2u);

  const BoxArray& fine_boxes = runtime.levels(0)[1].U.box_array();
  std::int64_t fine_cells = 0;
  for (const Box2D& box : fine_boxes.boxes())
    fine_cells += box.num_cells();
  ASSERT_GT(fine_cells, 0);
  ASSERT_LT(fine_cells, runtime.level_geom(1).domain.num_cells())
      << "the fixture must contain a genuine coarse/fine interface, not a uniformly fine domain";

  // Make the coarse representation exactly consistent with the bootstrapped fine patch before the
  // first diagnostic.  This is a spatial transfer only; every subsequent state update is Program-owned.
  runtime.average_down_level(0, 1);

  const double mass_before = runtime.composite_reduce("heat", "sum", 0);
  const double peak_before = runtime.composite_reduce("heat", "max", 0);
  ASSERT_TRUE(std::isfinite(mass_before));
  ASSERT_TRUE(std::isfinite(peak_before));

  simulation.advance(dt, steps);

  const double mass_after = runtime.composite_reduce("heat", "sum", 0);
  const double peak_after = runtime.composite_reduce("heat", "max", 0);
  ASSERT_TRUE(std::isfinite(mass_after));
  ASSERT_TRUE(std::isfinite(peak_after));

  EXPECT_LT(peak_after, peak_before - 1.0e-6)
      << "the Fickian face flux must smooth the non-uniform scalar field";
  EXPECT_NEAR(mass_after, mass_before, 5.0e-12)
      << "ProgramGraph reflux plus average-down must conserve the AMR composite integral";
}
