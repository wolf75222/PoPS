#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"

#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/amr_patch.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>

#include <array>
#include <cmath>
#include <vector>

namespace {

template <int Dim>
struct AdvectionModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  using Aux = pops::AuxState<Dim>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_aux = pops::aux_comps_for<Law, Dim>();

  Law law{};

  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr-seed.scalar-advection", 1};
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
  POPS_HD State source(const State&, const Aux&) const { return {}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

template <int Dim>
AdvectionModel<Dim> advection_model() {
  pops::RealVector<Dim> velocity{};
  return {pops::nd::ScalarAdvection<Dim>::prepare(velocity)};
}

template <int Dim>
std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
std::vector<double> gaussian(const pops::Extent<Dim>& shape) {
  std::vector<double> values(cell_count(shape));
  for (std::size_t linear = 0; linear < values.size(); ++linear) {
    std::size_t remainder = linear;
    double radius_squared = 0.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto width = static_cast<std::size_t>(shape[axis]);
      const int coordinate = static_cast<int>(remainder % width);
      remainder /= width;
      const double offset = (coordinate + 0.5) / static_cast<double>(shape[axis]) - 0.5;
      radius_squared += offset * offset;
    }
    values[linear] = 1.0 + 0.5 * std::exp(-radius_squared / 0.02);
  }
  return values;
}

template <int Dim>
struct PhysicalPatch {
  std::array<double, Dim> lower{};
  std::array<double, Dim> extent{};

  bool operator==(const PhysicalPatch&) const = default;
};

template <int Dim>
std::vector<PhysicalPatch<Dim>> physical_patches(const std::vector<pops::AmrPatch<Dim>>& patches,
                                                 const pops::RuntimeSpatialDomain<Dim>& domain) {
  std::vector<PhysicalPatch<Dim>> result;
  result.reserve(patches.size());
  for (const pops::AmrPatch<Dim>& patch : patches) {
    PhysicalPatch<Dim> projected;
    for (int axis = 0; axis < Dim; ++axis) {
      const double spacing = (domain.upper[axis] - domain.lower[axis]) /
                             std::ldexp(static_cast<double>(domain.shape[axis]), patch.level);
      projected.lower[axis] = domain.lower[axis] + patch.box.lo[axis] * spacing;
      projected.extent[axis] = patch.box.length(axis) * spacing;
    }
    result.push_back(projected);
  }
  return result;
}

template <int Dim>
pops::AmrSystem<Dim> make_system(const pops::AmrSystemConfig<Dim>& config,
                                 const std::vector<double>& initial) {
  pops::AmrSystem<Dim> system(config);
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", initial);
  return system;
}

}  // namespace

TEST(test_amr_seed_no_refine, CoarseOnlyWithoutPreparedTaggingAuthority) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 32;
  config.regrid_every = 0;
  auto system = make_system(config, gaussian(config.shape));

  EXPECT_EQ(system.n_levels(), 1);
  EXPECT_EQ(system.n_patches(), 1);
  EXPECT_TRUE(system.patch_boxes().empty());
}

TEST(test_amr_seed_no_refine, PreparedTaggingSeedsExactGeometryWithoutReadMutation) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 32;
  config.regrid_every = 4;
  auto system = make_system(config, gaussian(config.shape));
  pops::test::install_prepared_threshold_union(system, {{"tracer", "u", 1.2}});

  const std::vector<pops::AmrPatch<Dim>> first = system.patch_boxes();
  ASSERT_FALSE(first.empty());
  EXPECT_EQ(first.size(), static_cast<std::size_t>(system.n_patches()));
  const std::vector<double> state_before = system.block_level_state_global("tracer", 0);
  const std::vector<PhysicalPatch<Dim>> projected = physical_patches(first, config);
  ASSERT_EQ(projected.size(), first.size());
  for (std::size_t patch_index = 0; patch_index < first.size(); ++patch_index) {
    const pops::AmrPatch<Dim>& patch = first[patch_index];
    ASSERT_GE(patch.level, 1);
    for (int axis = 0; axis < Dim; ++axis) {
      const auto level_cells = config.shape[axis] << patch.level;
      EXPECT_LE(0, patch.box.lo[axis]);
      EXPECT_LE(patch.box.lo[axis], patch.box.hi[axis]);
      EXPECT_LT(patch.box.hi[axis], level_cells);
      EXPECT_GE(projected[patch_index].lower[axis], config.lower[axis]);
      EXPECT_GT(projected[patch_index].extent[axis], 0.0);
      EXPECT_LE(projected[patch_index].lower[axis] + projected[patch_index].extent[axis],
                config.upper[axis]);
    }
  }

  const std::vector<pops::AmrPatch<Dim>> second = system.patch_boxes();
  EXPECT_EQ(second, first);
  EXPECT_EQ(physical_patches(second, config), projected);
  EXPECT_EQ(system.block_level_state_global("tracer", 0), state_before);
}
