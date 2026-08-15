/// @file
/// @brief Exact-ranked refined diffusion through the installed AMR Program.

#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

template <int Dim>
class DiffusiveScalar : public pops::nd::ScalarAdvection<Dim> {
 public:
  using State = typename pops::nd::ScalarAdvection<Dim>::State;

  explicit DiffusiveScalar(pops::Real diffusivity) : diffusivity_(diffusivity) {}

  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"tests.amr.program-diffusion.scalar", 2};
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
static_assert(pops::DiffusiveModel<DiffusiveScalar<2>>);
static_assert(pops::DiffusiveModel<DiffusiveScalar<3>>);
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
pops::Index<Dim> index_from_ordinal(const pops::Box<Dim>& box, std::size_t ordinal) {
  pops::Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t length = static_cast<std::size_t>(box.length(axis));
    result[axis] = box.lo[axis] + static_cast<std::int64_t>(ordinal % length);
    ordinal /= length;
  }
  return result;
}

template <int Dim>
std::size_t flatten(const pops::Index<Dim>& index, const pops::Box<Dim>& domain) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result += static_cast<std::size_t>(index[axis] - domain.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(domain.length(axis));
  }
  return result;
}

template <int Dim>
void expect_covered_coarse_equals_fine_restriction(pops::AmrSystem<Dim>& system,
                                                   const pops::Extent<Dim>& ratio) {
  const pops::Box<Dim> coarse_domain = system.prepared_amr_level_geometry(0).domain();
  const pops::Box<Dim> fine_domain = system.prepared_amr_level_geometry(1).domain();
  const std::vector<double> coarse = system.block_level_state_global("heat", 0);
  const std::vector<double> fine = system.block_level_state_global("heat", 1);
  std::size_t covered_cells = 0;
  for (const pops::Box<Dim>& fine_patch : system.prepared_amr_block_state(0, 1).layout().boxes()) {
    const pops::Box<Dim> covered = pops::coarsen(fine_patch, ratio);
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(covered.numPts()); ++ordinal) {
      const pops::Index<Dim> parent = index_from_ordinal(covered, ordinal);
      double restricted = 0.0;
      std::size_t child_count = 1;
      for (int axis = 0; axis < Dim; ++axis)
        child_count *= static_cast<std::size_t>(ratio[axis]);
      for (std::size_t child_ordinal = 0; child_ordinal < child_count; ++child_ordinal) {
        pops::Index<Dim> child{};
        std::size_t remaining = child_ordinal;
        for (int axis = 0; axis < Dim; ++axis) {
          const auto axis_ratio = static_cast<std::size_t>(ratio[axis]);
          child[axis] =
              parent[axis] * ratio[axis] + static_cast<std::int64_t>(remaining % axis_ratio);
          remaining /= axis_ratio;
        }
        restricted += fine.at(flatten(child, fine_domain));
      }
      restricted /= static_cast<double>(child_count);
      EXPECT_NEAR(coarse.at(flatten(parent, coarse_domain)), restricted, 2.0e-13)
          << "published covered coarse cell must equal its conservative fine restriction";
      ++covered_cells;
    }
  }
  EXPECT_GT(covered_cells, 0U);
}

template <int Dim>
pops::AmrSystemConfig<Dim> refined_config() {
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
std::vector<double> periodic_mode(const pops::Extent<Dim>& shape) {
  constexpr double two_pi = 6.283185307179586476925286766559;
  std::vector<double> result(cell_count(shape), 0.0);
  for (std::size_t linear = 0; linear < result.size(); ++linear) {
    std::size_t remaining = linear;
    double mode = 1.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto width = static_cast<std::size_t>(shape[axis]);
      const int coordinate = static_cast<int>(remaining % width);
      remaining /= width;
      const double x = (static_cast<double>(coordinate) + 0.5) / static_cast<double>(shape[axis]);
      mode *= std::cos(two_pi * (x - 0.17 * static_cast<double>(axis + 1)));
    }
    result[linear] = 1.0 + 0.2 * mode;
  }
  return result;
}

template <int Dim>
void materialize_conservative_bootstrap(pops::AmrSystem<Dim>& system,
                                        const pops::AmrSystemConfig<Dim>& config,
                                        const std::vector<double>& initial) {
  constexpr const char* state_route = "tests.amr.program-diffusion/state/heat";
  system.bind_bootstrap_subject(state_route, "heat", "bound_level_zero");
  system.stage_bootstrap_array(state_route, "heat", "cell", "cell", 1, config.shape, initial);
  pops::Extent<Dim> transfer_ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    transfer_ghosts[axis] = 1;
  system.register_bootstrap_transfer_route(
      "tests.amr.program-diffusion/bootstrap/prolongation", {state_route},
      "tests.amr.program-diffusion/bootstrap/conservative-linear@1", "cell", "cell", "conservative",
      "dense", "prolongation", "conservative_linear", 2, transfer_ghosts,
      config.transition_ratios.front());
  system.begin_bootstrap_plan();
  (void)system.materialize_bootstrap_action(state_route, "initialize_level_zero",
                                            "bound_level_zero", 0);
  if (!system.bootstrap_next_level()) {
    system.rollback_bootstrap_level();
    throw std::runtime_error("diffusion bootstrap did not create the requested refined level");
  }
  (void)system.materialize_bootstrap_action(state_route, "prolong_from_parent",
                                            "conservative_linear", 1);
  system.commit_bootstrap_level();
}

template <int Dim>
void verify_refined_program_diffusion() {
  const pops::AmrSystemConfig<Dim> config = refined_config<Dim>();
  const std::vector<double> initial = periodic_mode(config.shape);
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.amr.program-diffusion/runtime@1");
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  system.install_block_state_route("heat", "tests.amr.program-diffusion/state/heat");
  pops::add_compiled_model<Dim>(
      system, "heat", DiffusiveScalar<Dim>{pops::Real(0.05)}, "none", "rusanov", "conservative",
      "explicit", static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
      static_cast<double>(pops::kWenoEpsilon), false, "tests.amr.program-diffusion/physical-flux");
  pops::test::install_prepared_threshold_union(
      system,
      {{"heat", "scalar", 1.15, pops::test::PreparedThresholdRelation::Above,
        "tests.amr.program-diffusion/state/heat"}},
      "tests.amr.program-diffusion/tagging@2");
  materialize_conservative_bootstrap(system, config, initial);

  ASSERT_EQ(system.n_levels(), 2);
  const pops::MultiFab<Dim>& coarse = system.prepared_amr_block_state(0, 0);
  const pops::MultiFab<Dim>& fine = system.prepared_amr_block_state(0, 1);
  const pops::mesh::BoxArray<Dim>& fine_boxes = fine.layout();
  const pops::mesh::Distribution<Dim>& fine_distribution = fine.distribution();
  ASSERT_TRUE(fine_distribution.matches_layout(fine_boxes));
  ASSERT_FALSE(fine_boxes.boxes().empty());
  std::int64_t refined_cells = 0;
  for (const pops::Box<Dim>& patch : fine_boxes.boxes())
    refined_cells += patch.numPts();
  ASSERT_GT(refined_cells, 0);
  ASSERT_LT(refined_cells, system.prepared_amr_level_geometry(1).domain().numPts());
  ASSERT_EQ(coarse.ncomp(), 1);
  ASSERT_EQ(fine.ncomp(), 1);

  pops::test::install_forward_euler_program(system, false);
  EXPECT_TRUE(system.program_interface_flux_ledger_manifest().empty());
  const auto accepted_flux_before = system.program_flux_ledger_manifest();
  EXPECT_TRUE(accepted_flux_before.empty());

  pops::MultiFab<Dim> coarse_before(coarse);
  pops::MultiFab<Dim> fine_before(fine);
  const double mass_before = system.composite_reduce("heat", "sum", 0);
  const pops::Real peak_before = pops::reduce_max(fine_before);
  constexpr double dt = 2.0e-4;

  system.begin_step_transaction();
  system.step(dt);
  pops::MultiFab<Dim> coarse_trial(system.prepared_amr_block_state(0, 0));
  pops::MultiFab<Dim> fine_trial(system.prepared_amr_block_state(0, 1));
  const double mass_trial = system.composite_reduce("heat", "sum", 0);
  const auto trial_flux = system.program_flux_ledger_manifest();
  bool saw_coarse_flux = false;
  bool saw_fine_flux = false;
  for (const auto& row : trial_flux) {
    if (row.size() != 13)
      continue;
    saw_coarse_flux = saw_coarse_flux || row[10].ends_with("_coarse");
    saw_fine_flux = saw_fine_flux || row[10].ends_with("_fine");
  }
  EXPECT_TRUE(saw_coarse_flux);
  EXPECT_TRUE(saw_fine_flux);
  system.rollback_step_transaction();

  EXPECT_EQ(pops::difference_sum_sq_all(system.prepared_amr_block_state(0, 0), coarse_before),
            pops::Real(0));
  EXPECT_EQ(pops::difference_sum_sq_all(system.prepared_amr_block_state(0, 1), fine_before),
            pops::Real(0));
  EXPECT_DOUBLE_EQ(system.composite_reduce("heat", "sum", 0), mass_before);
  EXPECT_EQ(system.program_flux_ledger_manifest(), accepted_flux_before);

  system.step(dt);
  EXPECT_EQ(pops::difference_sum_sq_all(system.prepared_amr_block_state(0, 0), coarse_trial),
            pops::Real(0));
  EXPECT_EQ(pops::difference_sum_sq_all(system.prepared_amr_block_state(0, 1), fine_trial),
            pops::Real(0));
  EXPECT_NEAR(system.composite_reduce("heat", "sum", 0), mass_before, 2.0e-12);
  EXPECT_DOUBLE_EQ(system.composite_reduce("heat", "sum", 0), mass_trial);
  EXPECT_EQ(system.program_flux_ledger_manifest(), trial_flux);
  expect_covered_coarse_equals_fine_restriction(system, config.transition_ratios.front());
  EXPECT_LT(pops::reduce_max(system.prepared_amr_block_state(0, 1)),
            peak_before - pops::Real(1e-7));
}

TEST(test_amr_program_diffusion,
     RefinedFickianFacesAreRefluxedAveragedDownAndTransactionallyPublished) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  verify_refined_program_diffusion<pops::kNativeDimension>();
}

}  // namespace
