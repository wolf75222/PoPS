/// @file
/// @brief Exact-ranked refined diffusion through the installed AMR Program.

#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/diffusion/prepared_diffusion.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

template <int Dim>
class DiffusiveScalar : public pops::nd::ScalarAdvection<Dim> {
 public:
  using Base = pops::nd::ScalarAdvection<Dim>;
  using State = typename pops::nd::ScalarAdvection<Dim>::State;

  DiffusiveScalar(pops::Real diffusivity, pops::RealVector<Dim> velocity)
      : Base(Base::prepare(velocity)), diffusivity_(diffusivity) {}

  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"tests.amr.program-diffusion.scalar", 3};
  }

  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    Base::serialize_exact_parameters(contract);
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
double transported_sine_projection(const std::vector<double>& values,
                                   const pops::Extent<Dim>& shape) {
  constexpr double two_pi = 6.283185307179586476925286766559;
  double projection = 0.0;
  for (std::size_t linear = 0; linear < values.size(); ++linear) {
    std::size_t remaining = linear;
    double basis = 1.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto width = static_cast<std::size_t>(shape[axis]);
      const int coordinate = static_cast<int>(remaining % width);
      remaining /= width;
      const double x = (static_cast<double>(coordinate) + 0.5) / shape[axis];
      const double phase = two_pi * (x - 0.17 * static_cast<double>(axis + 1));
      basis *= axis == 0 ? std::sin(phase) : std::cos(phase);
    }
    projection += values[linear] * basis;
  }
  return projection / static_cast<double>(values.size());
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

// Exercise the same constitutive face -> conservative reflux adapter as generated Programs.
// The older default-provider path below has its own physical flux and cannot detect a reversed
// sign when +div(G) constitutive faces enter the shared -div(F) reflux ledger.
template <int Dim>
void install_prepared_constitutive_program(pops::AmrSystem<Dim>& system) {
  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("tests.amr.prepared-diffusion/macro");
  context->install(
      [context](double macro_dt) {
        context->advance_hierarchy(macro_dt, [context](double level_dt) {
          context->set_stage_time(0, 1);
          auto& state = context->state(0);
          auto& rhs = context->rhs_scratch(1000, 0, state);
          pops::runtime::program::PreparedDiffusion<Dim> prepared(*context, state, {}, true);
          context->prepare_generated_state(0, state, 3000);
          prepared.apply(state, rhs, [&](std::size_t local) {
            const auto values = std::as_const(state).fab(local).view();
            return [=] POPS_HD(const pops::Index<Dim>& cell) {
              std::array<pops::Real, Dim + 2> law{};
              law[0] = values(cell, 0);
              for (int axis = 0; axis < Dim; ++axis)
                law[axis + 1] = pops::Real(0.05);
              law[Dim + 1] = pops::Real(1);
              return law;
            };
          });
          context->attach_diffusive_flux_basis(0, rhs, 3000, prepared.faces(), pops::Real(1),
                                               "tests.amr.prepared-diffusion/constitutive");
          auto& transport = context->rhs_scratch(1000, 1, state);
          std::vector<pops::nd::FaceField<Dim>> transport_faces;
          context->neg_div_flux_default_with_faces_into(0, state, transport, 3000, transport_faces,
                                                        "tests.amr.prepared-diffusion/transport");
          context->axpy(rhs, pops::Real(1), transport, level_dt, {{0, 1, 1}});
          context->axpy(state, pops::Real(level_dt), rhs);
        });
      },
      context);
  system.set_program_block_map({0});
  // Declare the exact (block, RHS identity, provider) tuples before either producer runs.
  // The shared RHS has distinct constitutive and transport temporal families, as in codegen.
  context->install_flux_temporal_families(
      {{0, 3000, 1, "tests.amr.prepared-diffusion/transport"},
       {0, 3000, 4, "tests.amr.prepared-diffusion/constitutive"}});
  using FluxBudget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget(
      "tests.amr.prepared-diffusion/forward-euler@1", {FluxBudget{2, 2}}, 0, 0);
}

template <int Dim, bool PreparedConstitutive = false>
void verify_refined_program_diffusion(double dt = 2.0e-4) {
  const pops::AmrSystemConfig<Dim> config = refined_config<Dim>();
  const std::vector<double> initial = periodic_mode(config.shape);
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.amr.program-diffusion/runtime@1");
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  system.install_block_state_route("heat", "tests.amr.program-diffusion/state/heat");
  pops::RealVector<Dim> velocity{};
  for (int axis = 0; axis < Dim; ++axis)
    velocity[axis] = axis == 0 ? pops::Real(0.35) : pops::Real(-0.2);
  // The default spatial provider includes its model's Fickian flux. In the constitutive
  // branch, PreparedDiffusion supplies that term once and the default provider supplies transport.
  constexpr pops::Real model_diffusivity = PreparedConstitutive ? pops::Real(0) : pops::Real(0.05);
  pops::add_compiled_model<Dim>(system, "heat", DiffusiveScalar<Dim>{model_diffusivity, velocity},
                                "none", "rusanov", "conservative", "explicit",
                                static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.amr.program-diffusion/physical-flux");
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

  if constexpr (PreparedConstitutive)
    install_prepared_constitutive_program(system);
  else
    pops::test::install_forward_euler_program(system, false);
  EXPECT_TRUE(system.program_interface_flux_ledger_manifest().empty());
  const auto accepted_flux_before = system.program_flux_ledger_manifest();
  EXPECT_TRUE(accepted_flux_before.empty());

  pops::MultiFab<Dim> coarse_before(coarse);
  pops::MultiFab<Dim> fine_before(fine);
  const double mass_before = system.composite_reduce("heat", "sum", 0);
  const double transport_projection_before = transported_sine_projection(initial, config.shape);
  EXPECT_NEAR(transport_projection_before, 0.0, 2.0e-14);
  const pops::Real peak_before = pops::reduce_max(fine_before);
  system.begin_step_transaction();
  system.step(dt);
  pops::MultiFab<Dim> coarse_trial(system.prepared_amr_block_state(0, 0));
  pops::MultiFab<Dim> fine_trial(system.prepared_amr_block_state(0, 1));
  const double mass_trial = system.composite_reduce("heat", "sum", 0);
  const auto trial_flux = system.program_flux_ledger_manifest();
  bool saw_coarse_flux = false;
  bool saw_fine_flux = false;
  bool saw_constitutive_flux = false;
  bool saw_transport_flux = false;
  std::array<double, Dim> coarse_weighted_measure{};
  std::array<double, Dim> fine_weighted_measure{};
  std::array<std::size_t, Dim> coarse_fragments{};
  std::array<std::size_t, Dim> fine_fragments{};
  for (const auto& row : trial_flux) {
    ASSERT_EQ(row.size(), 17u);
    saw_constitutive_flux =
        saw_constitutive_flux || row[2].find("/provider/4/") != std::string::npos;
    saw_transport_flux = saw_transport_flux || row[2].find("/provider/1/") != std::string::npos;
    EXPECT_FALSE(row[13].empty());  // Exact accepted face-evidence space.
    ASSERT_GE(row[10].size(), 3u);
    const int axis = static_cast<int>(row[10].front() - 'x');
    ASSERT_GE(axis, 0);
    ASSERT_LT(axis, Dim);
    const bool coarse_role = row[10].ends_with("_coarse");
    const bool fine_role = row[10].ends_with("_fine");
    ASSERT_TRUE(coarse_role || fine_role);
    saw_coarse_flux = saw_coarse_flux || coarse_role;
    saw_fine_flux = saw_fine_flux || fine_role;

    const int level = std::stoi(row[4]);
    const double face_measure = std::stod(row[11]);
    const double duration = std::stod(row[12]);
    double expected_face_measure = 1.0;
    const auto geometry = system.prepared_amr_level_geometry(level);
    for (int tangent = 0; tangent < Dim; ++tangent)
      if (tangent != axis)
        expected_face_measure *= geometry.spacing(tangent);
    EXPECT_DOUBLE_EQ(face_measure, expected_face_measure);
    EXPECT_DOUBLE_EQ(duration, coarse_role ? dt : dt / 2.0);

    const double stage_weight =
        static_cast<double>(std::stoll(row[8])) / static_cast<double>(std::stoll(row[9]));
    const double weighted_measure = stage_weight * face_measure * duration;
    if (coarse_role) {
      coarse_weighted_measure[static_cast<std::size_t>(axis)] += weighted_measure;
      ++coarse_fragments[static_cast<std::size_t>(axis)];
    } else {
      fine_weighted_measure[static_cast<std::size_t>(axis)] += weighted_measure;
      ++fine_fragments[static_cast<std::size_t>(axis)];
    }
  }
  EXPECT_TRUE(saw_coarse_flux);
  EXPECT_TRUE(saw_fine_flux);
  for (int axis = 0; axis < Dim; ++axis) {
    const auto index = static_cast<std::size_t>(axis);
    if (coarse_fragments[index] == 0 && fine_fragments[index] == 0)
      continue;
    ASSERT_GT(coarse_fragments[index], 0u);
    ASSERT_GT(fine_fragments[index], 0u);
    const double scale = std::max(
        {1.0, std::abs(coarse_weighted_measure[index]), std::abs(fine_weighted_measure[index])});
    EXPECT_NEAR(coarse_weighted_measure[index], fine_weighted_measure[index], 5.0e-14 * scale);
  }
  if constexpr (PreparedConstitutive) {
    EXPECT_TRUE(saw_constitutive_flux);
    EXPECT_TRUE(saw_transport_flux);
  }
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
  const auto accepted_coarse = system.block_level_state_global("heat", 0);
  EXPECT_GT(transported_sine_projection(accepted_coarse, config.shape), 1.0e-6);
  expect_covered_coarse_equals_fine_restriction(system, config.transition_ratios.front());
  EXPECT_LT(pops::reduce_max(system.prepared_amr_block_state(0, 1)),
            peak_before - pops::Real(1e-7));
}

TEST(test_amr_program_diffusion, FluxLedgerManifestPreservesFractionalMetricTimeClosure) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  // Six-decimal formatting cannot reproduce this duration. The common helper
  // verifies exact manifest round trips and coarse/fine metric-time closure.
  verify_refined_program_diffusion<pops::kNativeDimension>(2.0e-4 / 3.0);
}

TEST(test_amr_program_diffusion,
     RefinedTransportDiffusionFacesAreRefluxedAveragedDownAndTransactionallyPublished) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  verify_refined_program_diffusion<pops::kNativeDimension>();
}

// Prepared scalar diffusion is supported in Dim1/Dim2; the provider test above also covers Dim3.
#if POPS_NATIVE_DIM <= 2
TEST(test_amr_program_diffusion,
     PreparedConstitutiveFacesConserveAcrossPartialRefinementAndRollback) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  verify_refined_program_diffusion<pops::kNativeDimension, true>();
}
#endif

}  // namespace
