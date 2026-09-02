#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "amr_runtime_authority.hpp"

#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/amr_patch.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>

#include <algorithm>
#include <climits>
#include <string>
#include <vector>

namespace {

template <int Dim>
struct EulerModel {
  using Law = pops::nd::IdealGasEuler<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;

  Law law = Law::prepare(pops::Real(1.4));

  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr-regrid-variable.euler", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    law.serialize_exact_parameters(contract);
  }
  static pops::VariableSet conservative_vars() { return Law::conservative_vars(); }
  static pops::VariableSet primitive_vars() { return Law::primitive_vars(); }
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
std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
std::vector<double> energy_bump(const pops::Extent<Dim>& shape) {
  const std::size_t cells = cell_count(shape);
  std::vector<double> state(static_cast<std::size_t>(Dim + 2) * cells, 0.0);
  for (std::size_t linear = 0; linear < cells; ++linear) {
    state[linear] = 1.0;
    state[static_cast<std::size_t>(Dim + 1) * cells + linear] = 2.5;
    std::size_t remainder = linear;
    bool inside = true;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto width = static_cast<std::size_t>(shape[axis]);
      const int coordinate = static_cast<int>(remainder % width);
      remainder /= width;
      inside = inside && coordinate >= 2 && coordinate < 8;
    }
    if (inside)
      state[static_cast<std::size_t>(Dim + 1) * cells + linear] = 6.0;
  }
  return state;
}

template <int Dim>
std::vector<pops::AmrPatch<Dim>> run_selector(const std::string& variable) {
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 32;
  config.regrid_every = 1;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "test.amr-regrid-variable/runtime@1");
  system.install_block_state_route("gas", "state/gas");
  pops::add_compiled_model<Dim>(
      system, "gas", EulerModel<Dim>{}, "minmod", "rusanov", "conservative", "explicit",
      static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
      static_cast<double>(pops::kWenoEpsilon), false, "test.amr-regrid-variable/physical-flux");
  system.set_conservative_state("gas", energy_bump(config.shape));
  pops::test::install_prepared_threshold_union(system, {{"gas", variable, 4.0}});
  return system.patch_boxes();
}

template <int Dim>
int minimum_fine_corner(const std::vector<pops::AmrPatch<Dim>>& patches) {
  int result = INT_MAX;
  for (const pops::AmrPatch<Dim>& patch : patches)
    if (patch.level > 0)
      for (int axis = 0; axis < Dim; ++axis)
        result = std::min(result, patch.box.lo[axis]);
  return result;
}

}  // namespace

TEST(test_amr_regrid_variable, AuthoredVariableSelectsTheExactConservativeComponent) {
  constexpr int Dim = pops::kNativeDimension;
  const std::vector<pops::AmrPatch<Dim>> density = run_selector<Dim>("rho");
  const std::vector<pops::AmrPatch<Dim>> energy = run_selector<Dim>("E");

  EXPECT_TRUE(density.empty()) << "uniform rho must not inherit the E selector";
  ASSERT_FALSE(energy.empty()) << "the E bump must activate exact ranked refinement";
  EXPECT_LT(minimum_fine_corner(energy), 16)
      << "the prepared E selector must refine the lower-corner bump";
}

TEST(test_amr_regrid_variable, UnknownVariableFailsBeforeHierarchyPublication) {
  constexpr int Dim = pops::kNativeDimension;
  try {
    (void)run_selector<Dim>("not_a_conservative_variable");
    FAIL() << "unknown conservative tagging variables must fail before hierarchy publication";
  } catch (const std::invalid_argument& error) {
    EXPECT_STREQ(error.what(), "AMR tagging names an unknown conservative variable");
  }
}
