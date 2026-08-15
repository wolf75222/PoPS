/// @file
/// @brief Refined exact-ranked Fickian diffusion through Program reflux and average-down.

#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

template <int Dim>
class DiffusiveScalar : public pops::nd::ScalarAdvection<Dim> {
 public:
  using State = typename pops::nd::ScalarAdvection<Dim>::State;

  explicit DiffusiveScalar(pops::Real diffusivity) : diffusivity_(diffusivity) {}

  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr.diffusive-scalar", 2};
  }

  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.scalar(diffusivity_);
  }

  POPS_HD pops::Real diffusivity() const { return diffusivity_; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }

 private:
  pops::Real diffusivity_ = pops::Real(0);
};

static_assert(pops::DiffusiveModel<DiffusiveScalar<1>>);
static_assert(pops::nd::ConservationLaw<1, DiffusiveScalar<1>>);
static_assert(pops::nd::ConservationLaw<2, DiffusiveScalar<2>>);
static_assert(pops::nd::ConservationLaw<3, DiffusiveScalar<3>>);

template <int Dim>
std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
pops::AmrSystemConfig<Dim> config() {
  constexpr int cells = 12;
  pops::AmrSystemConfig<Dim> result;
  result.level_count = 2;
  result.regrid_every = 0;
  result.explicit_bootstrap = true;
  for (int axis = 0; axis < Dim; ++axis) {
    result.shape[axis] = cells;
    result.lower[axis] = pops::Real(0);
    result.upper[axis] = pops::Real(1);
    result.periodicity[axis] = true;
    result.coarse_max_grid[axis] = cells;
    result.transition_ratios.front()[axis] = 2;
    result.transition_buffers.front()[axis] = 1;
    result.transition_lookaheads.front()[axis] = 1;
  }
  return result;
}

template <int Dim>
std::vector<double> periodic_manufactured_state(const pops::Extent<Dim>& shape) {
  constexpr double two_pi = 6.283185307179586476925286766559;
  std::vector<double> result(cell_count(shape), 0.0);
  for (std::size_t linear = 0; linear < result.size(); ++linear) {
    std::size_t remaining = linear;
    double mode = 1.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const int coordinate = static_cast<int>(remaining % static_cast<std::size_t>(shape[axis]));
      remaining /= static_cast<std::size_t>(shape[axis]);
      const double x = (static_cast<double>(coordinate) + 0.5) / static_cast<double>(shape[axis]);
      mode *= std::cos(two_pi * (x - 0.5));
    }
    result[linear] = 1.0 + 0.2 * mode;
  }
  return result;
}

template <int Dim>
void rebuild_with_strict_fine_subset(pops::AmrSystem<Dim>& system) {
  constexpr int coarse_cells = 12;
  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = coarse_cells / 2;
    upper[axis] = 3 * coarse_cells / 2 - 1;
  }
  system.rebuild_hierarchy({pops::AmrPatch<Dim>{1, {lower, upper}}}, {0});
}

template <int Dim>
void verify_refined_program_diffusion() {
  const auto system_config = config<Dim>();
  pops::AmrSystem<Dim> system(system_config);
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  system.install_block_state_route("heat", "test.amr.diffusion/state/heat");
  pops::add_compiled_model<Dim>(system, "heat", DiffusiveScalar<Dim>{pops::Real(0.05)}, "none",
                                "rusanov", "conservative", "explicit");
  system.set_conservative_state("heat", periodic_manufactured_state(system_config.shape));
  system.set_poisson("charge_density", "geometric_mg", "periodic");
  pops::test::install_prepared_threshold_union(system, {{"heat", "scalar", 2.0}},
                                               "test.amr.diffusion.tagging@2");
  rebuild_with_strict_fine_subset(system);
  pops::test::install_forward_euler_program(system);

  auto* runtime = system.engine();
  ASSERT_NE(runtime, nullptr);
  ASSERT_EQ(runtime->hierarchy().num_levels(), 2U);
  const auto& fine_layout = runtime->hierarchy().layout(1);
  std::int64_t refined_cells = 0;
  for (const auto& patch : fine_layout.patches().boxes())
    refined_cells += patch.numPts();
  ASSERT_GT(refined_cells, 0);
  ASSERT_LT(refined_cells, fine_layout.domain().numPts());
  ASSERT_EQ(system.program_interface_flux_ledger_manifest().size(), 1U);

  const double mass_before = system.mass("heat");
  const std::vector<double> coarse_before = system.block_level_state_global("heat", 0);
  const std::vector<double> fine_before = system.block_level_state_global("heat", 1);
  ASSERT_FALSE(fine_before.empty());
  const double peak_before = *std::max_element(fine_before.begin(), fine_before.end());

  constexpr double dt = 2.0e-4;
  system.begin_step_transaction();
  system.step(dt);
  const double mass_trial = system.mass("heat");
  const std::vector<double> coarse_trial = system.block_level_state_global("heat", 0);
  const std::vector<double> fine_trial = system.block_level_state_global("heat", 1);
  system.rollback_step_transaction();

  EXPECT_EQ(system.block_level_state_global("heat", 0), coarse_before);
  EXPECT_EQ(system.block_level_state_global("heat", 1), fine_before);
  EXPECT_EQ(system.mass("heat"), mass_before);

  system.step(dt);
  const std::vector<double> coarse_after = system.block_level_state_global("heat", 0);
  const std::vector<double> fine_after = system.block_level_state_global("heat", 1);
  EXPECT_EQ(coarse_after, coarse_trial);
  EXPECT_EQ(fine_after, fine_trial);
  EXPECT_EQ(system.mass("heat"), mass_trial);
  EXPECT_NEAR(mass_trial, mass_before, 2.0e-12);
  EXPECT_LT(*std::max_element(fine_after.begin(), fine_after.end()), peak_before - 1.0e-7);
}

TEST(test_amr_program_diffusion,
     RefinedFickianFacesAreRefluxedAveragedDownAndTransactionallyPublished) {
  verify_refined_program_diffusion<pops::kNativeDimension>();
}

}  // namespace
