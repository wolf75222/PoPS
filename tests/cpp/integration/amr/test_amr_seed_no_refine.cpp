#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "amr_runtime_authority.hpp"

#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/runtime/amr/persistent_tagging_state.hpp>
#include <pops/runtime/amr_patch.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_checkpoint.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

template <int Dim>
struct AdvectionModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;

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
  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return {}; }
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
std::vector<double> corner_bump(const pops::Extent<Dim>& shape, bool upper) {
  std::vector<double> values(cell_count(shape), 0.0);
  for (std::size_t linear = 0; linear < values.size(); ++linear) {
    std::size_t remainder = linear;
    bool inside = true;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto width = static_cast<std::size_t>(shape[axis]);
      const int coordinate = static_cast<int>(remainder % width);
      remainder /= width;
      const int lower = upper ? shape[axis] - 8 : 2;
      const int upper_bound = upper ? shape[axis] - 2 : 8;
      inside = inside && coordinate >= lower && coordinate < upper_bound;
    }
    values[linear] = inside ? 1.0 : 0.0;
  }
  return values;
}

template <int Dim>
std::vector<double> affine_state(const pops::Extent<Dim>& shape) {
  std::vector<double> values(cell_count(shape), 0.0);
  for (std::size_t linear = 0; linear < values.size(); ++linear) {
    std::size_t remainder = linear;
    double value = 10.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto width = static_cast<std::size_t>(shape[axis]);
      value += static_cast<double>(remainder % width);
      remainder /= width;
    }
    values[linear] = value;
  }
  return values;
}

template <int Dim>
std::size_t flatten(const pops::Index<Dim>& index, const pops::Extent<Dim>& shape) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result += static_cast<std::size_t>(index[axis]) * stride;
    stride *= static_cast<std::size_t>(shape[axis]);
  }
  return result;
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
  pops::test::install_amr_runtime_authority(system, "test.amr-seed-no-refine.runtime@1");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>(), "minmod", "rusanov",
                                "conservative", "explicit",
                                static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "test.amr-seed-no-refine.tracer.provider-free@1");
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

TEST(test_amr_seed_no_refine, TransitionNeighborhoodOverflowFailsBeforeHierarchyAllocation) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  config.transition_buffers[0][0] = std::numeric_limits<std::int64_t>::max();
  config.transition_lookaheads[0][0] = std::numeric_limits<std::int64_t>::max();
  EXPECT_THROW((void)pops::AmrSystem<Dim>(config), std::overflow_error);
}

TEST(test_amr_seed_no_refine, AutomaticBootstrapRefusesFieldLeafBeforeMaterialization) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  auto system = make_system(config, gaussian(config.shape));
  EXPECT_THROW(system.set_bootstrap_tagging(
                   {"field"}, {"field/potential"}, {""}, {"phi"}, {0}, {POPS_TAGGING_ABOVE_V1},
                   {0.0}, {-1}, {}, {POPS_TAGGING_ABOVE_V1}, {0}, {}, {}, 0, "hold", "error",
                   "test::tagging-clock", "test::automatic-field-tagging@1"),
               std::invalid_argument);
  pops::test::install_prepared_threshold_union(system, {{"tracer", "u", 1.2}});
  EXPECT_EQ(system.n_levels(), 2)
      << "a rejected automatic field graph must not consume the unique tagging authority";
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

TEST(test_amr_seed_no_refine, AcceptedCadencePublishesTagDerivedRegridOnlyWhenDue) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 32;
  config.regrid_every = 2;
  auto system = make_system(config, corner_bump(config.shape, false));
  pops::test::install_prepared_threshold_union(system, {{"tracer", "u", 0.5}});
  system.install_program(POPS_TEST_AMR_V5_TRACER_PROGRAM);

  const std::vector<pops::AmrPatch<Dim>> initial = system.patch_boxes();
  ASSERT_FALSE(initial.empty());
  system.set_conservative_state("tracer", corner_bump(config.shape, true));

  system.step(0.01);
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_EQ(system.patch_boxes(), initial) << "non-due steps must not mutate hierarchy topology";

  system.step(0.01);
  EXPECT_EQ(system.macro_step(), 2);
  const std::vector<pops::AmrPatch<Dim>> moved = system.patch_boxes();
  ASSERT_FALSE(moved.empty());
  EXPECT_NE(moved, initial) << "the due accepted step must publish the newly tagged topology";
  const bool contains_upper_patch =
      std::any_of(moved.begin(), moved.end(), [&](const pops::AmrPatch<Dim>& patch) {
        if (patch.level == 0)
          return false;
        for (int axis = 0; axis < Dim; ++axis)
          if (patch.box.hi[axis] < 2 * (config.shape[axis] - 8))
            return false;
        return true;
      });
  EXPECT_TRUE(contains_upper_patch)
      << "the due accepted step must materialize refinement around the new upper-corner tags";
}

TEST(test_amr_seed_no_refine, RegridRetainsEvolvedFineCellsAndLinearlyProlongsNewCells) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 16;
  config.regrid_every = 1;
  auto system = make_system(config, affine_state(config.shape));
  pops::test::install_prepared_threshold_union(system, {{"tracer", "u", 0.0}});

  ASSERT_EQ(system.n_levels(), 2);
  const pops::Extent<Dim> fine_shape = [&] {
    pops::Extent<Dim> shape{};
    for (int axis = 0; axis < Dim; ++axis)
      shape[axis] = config.shape[axis] * 2;
    return shape;
  }();
  const std::vector<double> prolonged = system.block_level_state_global("tracer", 1);
  pops::Index<Dim> parent{};
  pops::Index<Dim> first_child{};
  pops::Index<Dim> second_child{};
  for (int axis = 0; axis < Dim; ++axis) {
    parent[axis] = 4;
    first_child[axis] = 8;
    second_child[axis] = 8;
  }
  second_child[0] = 9;
  EXPECT_NE(prolonged[flatten(first_child, fine_shape)],
            prolonged[flatten(second_child, fine_shape)])
      << "new fine cells must use prepared linear prolongation, not constant injection";

  double child_average = 0.0;
  const std::size_t child_count = std::size_t{1} << Dim;
  for (std::size_t mask = 0; mask < child_count; ++mask) {
    pops::Index<Dim> child{};
    for (int axis = 0; axis < Dim; ++axis)
      child[axis] = 2 * parent[axis] + static_cast<int>((mask >> axis) & 1u);
    child_average += prolonged[flatten(child, fine_shape)];
  }
  child_average /= static_cast<double>(child_count);
  EXPECT_NEAR(child_average, affine_state(config.shape)[flatten(parent, config.shape)], 1.0e-12)
      << "prepared prolongation must preserve the parent average";

  std::vector<double> evolved(prolonged.size(), 9.0);
  system.set_block_level_state("tracer", 1, evolved);
  const std::vector<double> accepted_before = system.block_level_state_global("tracer", 1);
  ASSERT_TRUE(system.regrid_from_prepared_tagging(0));
  const std::vector<double> accepted_after = system.block_level_state_global("tracer", 1);
  ASSERT_EQ(accepted_after.size(), accepted_before.size());
  std::size_t retained = 0;
  for (std::size_t cell = 0; cell < accepted_before.size(); ++cell)
    if (accepted_before[cell] == 9.0) {
      EXPECT_DOUBLE_EQ(accepted_after[cell], 9.0);
      ++retained;
    }
  EXPECT_GT(retained, 0u);

  auto injected = make_system(config, affine_state(config.shape));
  pops::Extent<Dim> no_ghost{};
  pops::Extent<Dim> ratio{};
  for (int axis = 0; axis < Dim; ++axis)
    ratio[axis] = 2;
  injected.register_bootstrap_transfer_route("test::explicit-injection-route", {"state/tracer"},
                                             "test::explicit-injection-provider@1", "cell", "cell",
                                             "conservative", "dense", "prolongation",
                                             "conservative_injection", 1, no_ghost, ratio);
  pops::test::install_prepared_threshold_union(injected, {{"tracer", "u", 0.0}});
  ASSERT_EQ(injected.n_levels(), 2);
  const std::vector<double> injected_fine = injected.block_level_state_global("tracer", 1);
  EXPECT_DOUBLE_EQ(injected_fine[flatten(first_child, fine_shape)],
                   injected_fine[flatten(second_child, fine_shape)])
      << "constant injection must run only when its exact route is selected explicitly";
}

TEST(test_amr_seed_no_refine,
     OneHierarchySweepAgesHysteresisOnceAndCheckpointRestoreIsTransactional) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 3;
  config.transition_ratios.resize(2);
  config.transition_buffers.resize(2);
  config.transition_lookaheads.resize(2);
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 8;
    for (std::size_t transition = 0; transition < 2; ++transition) {
      config.transition_ratios[transition][axis] = 2;
      config.transition_buffers[transition][axis] = 1;
      config.transition_lookaheads[transition][axis] = 1;
    }
  }
  auto system = make_system(config, affine_state(config.shape));
  constexpr int minimum_cycles = 3;
  const std::string provider = "test::persistent-hysteresis@1";
  pops::test::install_prepared_threshold_union(system, {{"tracer", "u", 0.0}}, provider,
                                               "test::tagging-clock", minimum_cycles);

  ASSERT_EQ(system.n_levels(), 3);
  const std::vector<std::uint8_t> accepted = system.program_accepted_state();
  const auto decoded =
      pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(accepted);
  std::vector<pops::Box<Dim>> parent_domains;
  parent_domains.push_back(config.index_domain());
  std::array<int, Dim> first_ratio{};
  for (int axis = 0; axis < Dim; ++axis)
    first_ratio[static_cast<std::size_t>(axis)] =
        static_cast<int>(config.transition_ratios[0][axis]);
  parent_domains.push_back(pops::amr::hierarchy::refine_box(
      parent_domains.back(), pops::amr::RefinementRatio<Dim>(first_ratio)));
  const auto hysteresis = pops::runtime::amr::PersistentTaggingState<Dim>::decode(
      decoded.tagging_hysteresis_state, minimum_cycles, provider, parent_domains);
  EXPECT_EQ(hysteresis.cycle(), 1u)
      << "all parent levels in one hierarchy sweep must share one hysteresis cycle";
  EXPECT_GT(hysteresis.active_entry_count(), 0u);

  auto restored = make_system(config, affine_state(config.shape));
  pops::test::install_prepared_threshold_union(restored, {{"tracer", "u", 0.0}}, provider,
                                               "test::tagging-clock", minimum_cycles);
  ASSERT_EQ(restored.n_levels(), 3);
  restored.restore_checkpoint_accepted_state(accepted);
  EXPECT_EQ(restored.program_accepted_state(), accepted);

  auto omitted_tagging = decoded;
  omitted_tagging.tagging_hysteresis_state.clear();
  const std::vector<std::uint8_t> omission =
      pops::runtime::program::serialize_amr_program_accepted_state(omitted_tagging);
  const std::uint64_t revision_before_omission = restored.program_accepted_state_revision();
  restored.restore_program_accepted_state(omission);
  EXPECT_EQ(restored.program_accepted_state(), accepted);
  EXPECT_EQ(restored.program_accepted_state_revision(), revision_before_omission + 1);

  const std::uint64_t revision_before_echo = restored.program_accepted_state_revision();
  restored.restore_program_accepted_state(accepted);
  EXPECT_EQ(restored.program_accepted_state(), accepted);
  EXPECT_EQ(restored.program_accepted_state_revision(), revision_before_echo + 1);

  auto corrupted = decoded;
  ASSERT_FALSE(corrupted.tagging_hysteresis_state.empty());
  corrupted.tagging_hysteresis_state.back() ^= std::uint8_t{0xff};
  const std::vector<std::uint8_t> invalid =
      pops::runtime::program::serialize_amr_program_accepted_state(corrupted);
  const auto before = restored.program_accepted_state();
  const std::uint64_t revision_before = restored.program_accepted_state_revision();
  EXPECT_THROW(restored.restore_program_accepted_state(invalid), std::invalid_argument);
  EXPECT_EQ(restored.program_accepted_state(), before);
  EXPECT_EQ(restored.program_accepted_state_revision(), revision_before);
  EXPECT_THROW(restored.restore_checkpoint_accepted_state(omission), std::invalid_argument);
  EXPECT_EQ(restored.program_accepted_state(), before);
  EXPECT_EQ(restored.program_accepted_state_revision(), revision_before);
  EXPECT_THROW(restored.restore_checkpoint_accepted_state(invalid), std::invalid_argument);
  EXPECT_EQ(restored.program_accepted_state(), before);
  EXPECT_EQ(restored.program_accepted_state_revision(), revision_before);
}
