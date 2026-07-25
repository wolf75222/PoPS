#include <gtest/gtest.h>

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/time/amr/advance/amr_advance.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/validation/physics/advection_diffusion.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace {

using pops::AmrLevelMP;
using pops::AmrTimeMethod;
using pops::Array4;
using pops::Box2D;
using pops::BoxArray;
using pops::ConstArray4;
using pops::DistributionMapping;
using pops::Geometry;
using pops::MultiFab;
using pops::NoSlope;
using pops::Periodicity;
using pops::Real;
using pops::RusanovFlux;

constexpr Real kPi = Real(3.1415926535897932384626433832795);

Real max_valid_difference(const MultiFab& lhs, const MultiFab& rhs) {
  pops::device_fence();
  Real difference = Real(0);
  for (int local = 0; local < lhs.local_size(); ++local) {
    const ConstArray4 a = lhs.fab(local).const_array();
    const ConstArray4 b = rhs.fab(local).const_array();
    const Box2D valid = lhs.box(local);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        for (int component = 0; component < lhs.ncomp(); ++component)
          difference = std::max(difference, std::abs(a(i, j, component) - b(i, j, component)));
  }
  return difference;
}

Real max_all_difference(const MultiFab& lhs, const MultiFab& rhs) {
  pops::device_fence();
  Real difference = Real(0);
  for (int local = 0; local < lhs.local_size(); ++local) {
    const ConstArray4 a = lhs.fab(local).const_array();
    const ConstArray4 b = rhs.fab(local).const_array();
    const Box2D box = lhs.box(local);
    for (int j = box.lo[1]; j <= box.hi[1]; ++j)
      for (int i = box.lo[0]; i <= box.hi[0]; ++i)
        for (int component = 0; component < lhs.ncomp(); ++component)
          difference = std::max(difference, std::abs(a(i, j, component) - b(i, j, component)));
  }
  return difference;
}

bool same_complete_storage(const MultiFab& lhs, const MultiFab& rhs) {
  pops::device_fence();
  if (lhs.local_size() != rhs.local_size() || lhs.ncomp() != rhs.ncomp() ||
      lhs.n_grow() != rhs.n_grow())
    return false;
  for (int local = 0; local < lhs.local_size(); ++local) {
    const ConstArray4 a = lhs.fab(local).const_array();
    const ConstArray4 b = rhs.fab(local).const_array();
    const Box2D grown = lhs.fab(local).grown_box();
    for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
      for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
        for (int component = 0; component < lhs.ncomp(); ++component)
          if (a(i, j, component) != b(i, j, component))
            return false;
  }
  return true;
}

struct NonFiniteSource : pops::validation::AdvectionDiffusion {
  POPS_HD State source(const State&, const Aux&) const {
    return State{std::numeric_limits<Real>::quiet_NaN()};
  }
};

struct FineOnlyNonFiniteSource : pops::validation::AdvectionDiffusion {
  POPS_HD State source(const State& state, const Aux&) const {
    return State{state[0] < Real(0) ? std::numeric_limits<Real>::quiet_NaN() : Real(0)};
  }
};

struct HugeFiniteSource : pops::validation::AdvectionDiffusion {
  POPS_HD State source(const State&, const Aux&) const {
    return State{std::numeric_limits<Real>::max()};
  }
};

struct RejectingNumericalFlux {
  template <pops::PhysicalFlux Physical>
  POPS_HD pops::FluxEvaluation<typename Physical::State> operator()(
      const Physical&, const typename Physical::Trace&, const typename Physical::Trace&,
      const pops::FaceContext&) const {
    return pops::FluxEvaluation<typename Physical::State>::reject(0x54455354u);
  }
};

}  // namespace

TEST(test_amr_ssprk2, stable_wire_is_additive_and_unknown_values_fail) {
  EXPECT_EQ(static_cast<int>(AmrTimeMethod::kEuler), 0);
  EXPECT_EQ(static_cast<int>(AmrTimeMethod::kSsprk3), 1);
  EXPECT_EQ(static_cast<int>(AmrTimeMethod::kSsprk2), 2);
  EXPECT_EQ(pops::amr_time_method_from_wire(0), AmrTimeMethod::kEuler);
  EXPECT_EQ(pops::amr_time_method_from_wire(1), AmrTimeMethod::kSsprk3);
  EXPECT_EQ(pops::amr_time_method_from_wire(2), AmrTimeMethod::kSsprk2);
  EXPECT_THROW((void)pops::amr_time_method_from_wire(99), std::runtime_error);
}

TEST(test_amr_ssprk2, facade_routes_and_effective_diagnostics_preserve_each_method) {
  pops::AmrSystemConfig config;
  config.n = 8;
  config.L = 1.0;
  config.periodicity = {true, true};
  pops::ModelSpec model;
  model.transport = "exb";
  model.source = "none";
  model.elliptic = "charge";

  struct ExpectedRoute {
    const char* input;
    const char* route;
    const char* method;
  };
  for (const ExpectedRoute expected : {
           ExpectedRoute{"explicit", "explicit", "ssprk2"},
           ExpectedRoute{"euler", "euler", "euler"},
           ExpectedRoute{"ssprk3", "ssprk3", "ssprk3"},
           ExpectedRoute{"imex", "imex", "imex"},
       }) {
    pops::AmrSystem system(config);
    system.add_block("state", model, "none", "rusanov", "conservative", expected.input);
    const pops::EffectiveOptionsReport report = system.effective_options_report();
    ASSERT_EQ(report.blocks.size(), 1u);
    EXPECT_EQ(report.blocks.front().time, expected.route) << expected.input;
    EXPECT_EQ(report.blocks.front().time_method, expected.method) << expected.input;
  }
}

TEST(test_amr_ssprk2, matches_shu_osher_identity_and_records_heun_effective_flux) {
  constexpr int cells = 24;
  const Box2D domain = Box2D::from_extents(cells, cells);
  const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
  const BoxArray cells_layout(std::vector<Box2D>{domain});
  const DistributionMapping ownership(cells_layout.size(), pops::n_ranks());
  const Periodicity periodic{true, true};
  const pops::validation::AdvectionDiffusion model{/*ax=*/Real(1), /*ay=*/Real(0.35),
                                                   /*nu=*/Real(0)};
  MultiFab aux(cells_layout, ownership, 3, 1);
  aux.set_val(Real(0));

  MultiFab initial(cells_layout, ownership, 1, NoSlope::n_ghost);
  for (int local = 0; local < initial.local_size(); ++local) {
    const Array4 state = initial.fab(local).array();
    const Box2D valid = initial.box(local);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        const Real x = geometry.x_cell(i);
        const Real y = geometry.y_cell(j);
        state(i, j, 0) =
            Real(1) + Real(0.2) * std::sin(Real(2) * kPi * x) * std::cos(Real(2) * kPi * y);
      }
  }

  const Real dt = Real(0.02);
  const Real dx = geometry.dx();
  const Real dy = geometry.dy();
  auto make_x_flux = [&] {
    return MultiFab(BoxArray(std::vector<Box2D>{pops::xface_box(domain)}), ownership, 1, 0);
  };
  auto make_y_flux = [&] {
    return MultiFab(BoxArray(std::vector<Box2D>{pops::yface_box(domain)}), ownership, 1, 0);
  };

  // Independent Shu-Osher reference: U1=U0+dt L(U0), then
  // Uref=1/2 U0+1/2 (U1+dt L(U1)). Preserve F0 and F1 for the reflux-flux oracle.
  MultiFab stage = initial;
  pops::fill_boundary(stage, domain, periodic);
  MultiFab flux0_x = make_x_flux(), flux0_y = make_y_flux();
  pops::compute_face_fluxes<NoSlope, RusanovFlux>(model, stage, aux, flux0_x, flux0_y, dx, dy);
  MultiFab residual(cells_layout, ownership, 1, 0);
  pops::mf_eval_rhs(model, stage, aux, flux0_x, flux0_y, dx, dy, residual);
  pops::saxpy(stage, dt, residual);
  pops::fill_boundary(stage, domain, periodic);
  MultiFab flux1_x = make_x_flux(), flux1_y = make_y_flux();
  pops::compute_face_fluxes<NoSlope, RusanovFlux>(model, stage, aux, flux1_x, flux1_y, dx, dy);
  pops::mf_eval_rhs(model, stage, aux, flux1_x, flux1_y, dx, dy, residual);
  MultiFab expected = stage;
  pops::saxpy(expected, dt, residual);
  pops::lincomb(expected, Real(1) / 2, initial, Real(1) / 2, expected);
  MultiFab effective_x = flux0_x, effective_y = flux0_y;
  pops::lincomb(effective_x, Real(1) / 2, flux0_x, Real(1) / 2, flux1_x);
  pops::lincomb(effective_y, Real(1) / 2, flux0_y, Real(1) / 2, flux1_y);

  // Production SSPRK2 helper starts with F(U0) and must return both the Shu-Osher state and
  // Feff=1/2 F(U0)+1/2 F(U1), which is subsequently consumed by coarse/fine reflux registers.
  MultiFab actual_state = initial;
  pops::fill_boundary(actual_state, domain, periodic);
  MultiFab actual_flux_x = make_x_flux(), actual_flux_y = make_y_flux();
  pops::compute_face_fluxes<NoSlope, RusanovFlux>(model, actual_state, aux, actual_flux_x,
                                                  actual_flux_y, dx, dy);
  AmrLevelMP level{std::move(actual_state), &aux, dx, dy};
  pops::detail::ssprk2_advance_level<NoSlope, RusanovFlux>(
      model, level, dt, actual_flux_x, actual_flux_y, /*recon_prim=*/false, /*lev=*/0, domain,
      periodic, /*pOld=*/nullptr, /*pNew=*/nullptr, /*frac=*/Real(0),
      /*parent_span=*/Real(0), /*coarse_replicated=*/true);

  EXPECT_LT(max_valid_difference(level.U, expected), Real(2e-14));
  EXPECT_LT(max_all_difference(actual_flux_x, effective_x), Real(2e-14));
  EXPECT_LT(max_all_difference(actual_flux_y, effective_y), Real(2e-14));

  std::vector<AmrLevelMP> euler_levels;
  euler_levels.push_back(AmrLevelMP{initial, &aux, dx, dy});
  pops::advance_amr<NoSlope, RusanovFlux>(model, euler_levels, domain, dt, periodic,
                                          /*coarse_replicated=*/true,
                                          /*recon_prim=*/false, /*imex=*/false,
                                          pops::NewtonOptions{}, AmrTimeMethod::kEuler);
  MultiFab euler_expected = initial;
  pops::fill_boundary(euler_expected, domain, periodic);
  pops::mf_advance_faces(euler_expected, flux0_x, flux0_y, dx, dy, dt);
  pops::mf_apply_source(model, euler_expected, aux, dt);
  EXPECT_TRUE(same_complete_storage(euler_levels.front().U, euler_expected))
      << "successful transactional Euler path changed the historical arithmetic";
  EXPECT_GT(max_valid_difference(level.U, euler_levels.front().U), Real(1e-6));
}

TEST(test_amr_ssprk2, coarse_fine_ghosts_follow_ssp_tableau_abscissae) {
  const Box2D coarse_domain = Box2D::from_extents(8, 8);
  const BoxArray coarse_layout(std::vector<Box2D>{coarse_domain});
  const DistributionMapping coarse_ownership(coarse_layout.size(), pops::n_ranks());
  MultiFab parent_old(coarse_layout, coarse_ownership, 1, 1);
  MultiFab parent_new(coarse_layout, coarse_ownership, 1, 1);
  parent_old.set_val(Real(0));
  parent_new.set_val(Real(2));

  const Box2D fine_box{{4, 4}, {11, 11}};
  const BoxArray fine_layout(std::vector<Box2D>{fine_box});
  const DistributionMapping fine_ownership(fine_layout.size(), pops::n_ranks());
  MultiFab initial(fine_layout, fine_ownership, 1, NoSlope::n_ghost);
  initial.set_val(Real(0.2));
  MultiFab aux(fine_layout, fine_ownership, 3, 1);
  aux.set_val(Real(0));
  const pops::validation::AdvectionDiffusion model{/*ax=*/Real(1), /*ay=*/Real(0),
                                                   /*nu=*/Real(0)};
  const Periodicity periodic{true, true};
  const Real dx = Real(1) / 16;
  const Real dt = Real(0.005);
  const Real parent_begin = Real(0.25);
  const Real parent_span = Real(0.5);

  auto make_x_flux = [&] {
    return MultiFab(BoxArray(std::vector<Box2D>{pops::xface_box(fine_box)}), fine_ownership, 1, 0);
  };
  auto make_y_flux = [&] {
    return MultiFab(BoxArray(std::vector<Box2D>{pops::yface_box(fine_box)}), fine_ownership, 1, 0);
  };
  auto prepared_stage0 = [&] {
    MultiFab state = initial;
    pops::detail::ssprk_refill_level_ghosts(state, /*lev=*/1, coarse_domain, periodic, &parent_old,
                                            &parent_new, parent_begin,
                                            /*coarse_replicated=*/true);
    MultiFab flux_x = make_x_flux();
    MultiFab flux_y = make_y_flux();
    pops::compute_face_fluxes<NoSlope, RusanovFlux>(model, state, aux, flux_x, flux_y, dx, dx);
    return std::tuple<MultiFab, MultiFab, MultiFab>{std::move(state), std::move(flux_x),
                                                    std::move(flux_y)};
  };

  auto manual_ssprk2 = [&](Real stage1_fraction) {
    auto [state, flux_x, flux_y] = prepared_stage0();
    MultiFab start = state;
    MultiFab residual(fine_layout, fine_ownership, 1, 0);
    pops::mf_eval_rhs(model, state, aux, flux_x, flux_y, dx, dx, residual);
    pops::saxpy(state, dt, residual);
    pops::detail::ssprk_refill_level_ghosts(state, /*lev=*/1, coarse_domain, periodic, &parent_old,
                                            &parent_new, stage1_fraction,
                                            /*coarse_replicated=*/true);
    MultiFab stage_flux_x = make_x_flux();
    MultiFab stage_flux_y = make_y_flux();
    pops::compute_face_fluxes<NoSlope, RusanovFlux>(model, state, aux, stage_flux_x, stage_flux_y,
                                                    dx, dx);
    pops::mf_eval_rhs(model, state, aux, stage_flux_x, stage_flux_y, dx, dx, residual);
    pops::saxpy(state, dt, residual);
    pops::lincomb(state, Real(1) / 2, start, Real(1) / 2, state);
    return state;
  };

  auto manual_ssprk3 = [&](Real stage1_fraction, Real stage2_fraction) {
    auto [state, flux_x, flux_y] = prepared_stage0();
    MultiFab start = state;
    MultiFab residual(fine_layout, fine_ownership, 1, 0);
    pops::mf_eval_rhs(model, state, aux, flux_x, flux_y, dx, dx, residual);
    pops::saxpy(state, dt, residual);
    pops::detail::ssprk_refill_level_ghosts(state, /*lev=*/1, coarse_domain, periodic, &parent_old,
                                            &parent_new, stage1_fraction,
                                            /*coarse_replicated=*/true);
    MultiFab stage_flux_x = make_x_flux();
    MultiFab stage_flux_y = make_y_flux();
    pops::compute_face_fluxes<NoSlope, RusanovFlux>(model, state, aux, stage_flux_x, stage_flux_y,
                                                    dx, dx);
    pops::mf_eval_rhs(model, state, aux, stage_flux_x, stage_flux_y, dx, dx, residual);
    pops::saxpy(state, dt, residual);
    pops::lincomb(state, Real(3) / 4, start, Real(1) / 4, state);
    pops::detail::ssprk_refill_level_ghosts(state, /*lev=*/1, coarse_domain, periodic, &parent_old,
                                            &parent_new, stage2_fraction,
                                            /*coarse_replicated=*/true);
    pops::compute_face_fluxes<NoSlope, RusanovFlux>(model, state, aux, stage_flux_x, stage_flux_y,
                                                    dx, dx);
    pops::mf_eval_rhs(model, state, aux, stage_flux_x, stage_flux_y, dx, dx, residual);
    pops::saxpy(state, dt, residual);
    pops::lincomb(state, Real(1) / 3, start, Real(2) / 3, state);
    return state;
  };

  const Real end_fraction = parent_begin + parent_span;
  const Real midpoint_fraction = parent_begin + Real(0.5) * parent_span;
  MultiFab expected2 = manual_ssprk2(end_fraction);
  MultiFab frozen2 = manual_ssprk2(parent_begin);
  auto [state2, flux2_x, flux2_y] = prepared_stage0();
  AmrLevelMP level2{std::move(state2), &aux, dx, dx};
  pops::detail::ssprk2_advance_level<NoSlope, RusanovFlux>(
      model, level2, dt, flux2_x, flux2_y, /*recon_prim=*/false, /*lev=*/1, coarse_domain, periodic,
      &parent_old, &parent_new, parent_begin, parent_span,
      /*coarse_replicated=*/true);
  EXPECT_LT(max_valid_difference(level2.U, expected2), Real(2e-14));
  EXPECT_GT(max_valid_difference(level2.U, frozen2), Real(1e-5));

  MultiFab expected3 = manual_ssprk3(end_fraction, midpoint_fraction);
  MultiFab frozen3 = manual_ssprk3(parent_begin, parent_begin);
  auto [state3, flux3_x, flux3_y] = prepared_stage0();
  AmrLevelMP level3{std::move(state3), &aux, dx, dx};
  pops::detail::ssprk3_advance_level<NoSlope, RusanovFlux>(
      model, level3, dt, flux3_x, flux3_y, /*recon_prim=*/false, /*lev=*/1, coarse_domain, periodic,
      &parent_old, &parent_new, parent_begin, parent_span,
      /*coarse_replicated=*/true);
  EXPECT_LT(max_valid_difference(level3.U, expected3), Real(2e-14));
  EXPECT_GT(max_valid_difference(level3.U, frozen3), Real(1e-5));
}

TEST(test_amr_ssprk2, rejects_imex_instead_of_ignoring_the_ssp_method) {
  const Box2D domain = Box2D::from_extents(8, 8);
  const BoxArray layout(std::vector<Box2D>{domain});
  const DistributionMapping ownership(layout.size(), pops::n_ranks());
  MultiFab state(layout, ownership, 1, NoSlope::n_ghost), aux(layout, ownership, 3, 1);
  state.set_val(Real(1));
  aux.set_val(Real(0));
  std::vector<AmrLevelMP> levels;
  levels.push_back(AmrLevelMP{std::move(state), &aux, Real(1) / 8, Real(1) / 8});
  const pops::validation::AdvectionDiffusion model{/*ax=*/Real(1), /*ay=*/Real(0),
                                                   /*nu=*/Real(0)};
  EXPECT_THROW((pops::advance_amr<NoSlope, RusanovFlux>(
                   model, levels, domain, Real(1e-3), Periodicity{true, true},
                   /*coarse_replicated=*/true, /*recon_prim=*/false, /*imex=*/true,
                   pops::NewtonOptions{}, AmrTimeMethod::kSsprk2)),
               std::runtime_error);
}

TEST(test_amr_ssprk2, euler_nonfinite_source_or_flux_never_publishes_state) {
  const Box2D domain = Box2D::from_extents(8, 8);
  const BoxArray layout(std::vector<Box2D>{domain});
  const DistributionMapping ownership(layout.size(), pops::n_ranks());
  MultiFab initial(layout, ownership, 1, NoSlope::n_ghost);
  MultiFab aux(layout, ownership, 3, 1);
  initial.set_val(Real(1.25));
  aux.set_val(Real(0));
  const MultiFab snapshot = initial;

  for (const bool imex : {false, true}) {
    std::vector<AmrLevelMP> levels{
        AmrLevelMP{initial, &aux, Real(1) / 8, Real(1) / 8}};
    const NonFiniteSource model{};
    EXPECT_THROW((pops::advance_amr<NoSlope, RusanovFlux>(
                     model, levels, domain, Real(1e-3), Periodicity{true, true},
                     /*coarse_replicated=*/true, /*recon_prim=*/false, imex,
                     pops::NewtonOptions{}, AmrTimeMethod::kEuler)),
                 std::runtime_error);
    EXPECT_TRUE(same_complete_storage(levels.front().U, snapshot))
        << (imex ? "implicit" : "explicit") << " source failure published state";
  }

  std::vector<AmrLevelMP> rejected_flux_levels{
      AmrLevelMP{initial, &aux, Real(1) / 8, Real(1) / 8}};
  const pops::validation::AdvectionDiffusion finite_model{/*ax=*/Real(1), /*ay=*/Real(0),
                                                          /*nu=*/Real(0)};
  EXPECT_THROW((pops::advance_amr<NoSlope, RejectingNumericalFlux>(
                   finite_model, rejected_flux_levels, domain, Real(1e-3),
                   Periodicity{true, true}, /*coarse_replicated=*/true,
                   /*recon_prim=*/false, /*imex=*/false, pops::NewtonOptions{},
                   AmrTimeMethod::kEuler)),
               std::runtime_error);
  EXPECT_TRUE(same_complete_storage(rejected_flux_levels.front().U, snapshot));
}

TEST(test_amr_ssprk2, finite_stage_data_that_overflows_is_not_published) {
  const Box2D domain = Box2D::from_extents(8, 8);
  const BoxArray layout(std::vector<Box2D>{domain});
  const DistributionMapping ownership(layout.size(), pops::n_ranks());
  MultiFab initial(layout, ownership, 1, NoSlope::n_ghost);
  MultiFab aux(layout, ownership, 3, 1);
  initial.set_val(Real(0));
  aux.set_val(Real(0));
  const MultiFab snapshot = initial;
  const HugeFiniteSource model{};

  for (const AmrTimeMethod method : {AmrTimeMethod::kSsprk2, AmrTimeMethod::kSsprk3}) {
    std::vector<AmrLevelMP> levels{
        AmrLevelMP{initial, &aux, Real(1) / 8, Real(1) / 8}};
    EXPECT_THROW((pops::advance_amr<NoSlope, RusanovFlux>(
                     model, levels, domain, Real(2), Periodicity{true, true},
                     /*coarse_replicated=*/true, /*recon_prim=*/false, /*imex=*/false,
                     pops::NewtonOptions{}, method)),
                 std::runtime_error);
    EXPECT_TRUE(same_complete_storage(levels.front().U, snapshot));
  }
}

TEST(test_amr_ssprk2, synchronization_overflow_is_rejected_before_hierarchy_publication) {
  const Box2D domain = Box2D::from_extents(4, 4);
  const BoxArray layout(std::vector<Box2D>{domain});
  const DistributionMapping ownership(layout.size(), pops::n_ranks());
  MultiFab initial(layout, ownership, 1, NoSlope::n_ghost);
  MultiFab aux(layout, ownership, 3, 1);
  initial.set_val(Real(3));
  aux.set_val(Real(0));

  std::vector<AmrLevelMP> live{
      AmrLevelMP{initial, &aux, Real(1) / 4, Real(1) / 4}};
  const MultiFab snapshot = live.front().U;
  std::vector<AmrLevelMP> attempt = live;
  MultiFab finite_increment(layout, ownership, 1, NoSlope::n_ghost);
  finite_increment.set_val(std::numeric_limits<Real>::max());
  pops::saxpy(attempt.front().U, Real(2), finite_increment);

  EXPECT_THROW(pops::detail::publish_amr_state_transaction(live, attempt), std::runtime_error);
  EXPECT_TRUE(same_complete_storage(live.front().U, snapshot));
}

TEST(test_amr_ssprk2, legacy_two_level_attempt_is_atomic_on_fine_source_failure) {
  const Box2D coarse_domain = Box2D::from_extents(8, 8);
  const BoxArray coarse_layout(std::vector<Box2D>{coarse_domain});
  const DistributionMapping coarse_ownership(coarse_layout.size(), pops::n_ranks());
  const Box2D fine_box{{4, 4}, {11, 11}};
  const BoxArray fine_layout(std::vector<Box2D>{fine_box});
  const DistributionMapping fine_ownership(fine_layout.size(), pops::n_ranks());

  MultiFab coarse(coarse_layout, coarse_ownership, 1, NoSlope::n_ghost);
  MultiFab fine(fine_layout, fine_ownership, 1, NoSlope::n_ghost);
  MultiFab coarse_aux(coarse_layout, coarse_ownership, 3, 1);
  MultiFab fine_aux(fine_layout, fine_ownership, 3, 1);
  coarse.set_val(Real(1));
  fine.set_val(Real(-1));
  coarse_aux.set_val(Real(0));
  fine_aux.set_val(Real(0));
  const MultiFab coarse_snapshot = coarse;
  const MultiFab fine_snapshot = fine;
  FineOnlyNonFiniteSource model{};
  model.ax = Real(0);
  model.ay = Real(0);

  EXPECT_THROW((pops::amr_step_2level_multipatch<NoSlope, RusanovFlux>(
                   model, coarse, coarse_domain, Real(1) / 8, Real(1) / 8, fine, coarse_aux,
                   fine_aux, Real(1e-3), Periodicity{true, true})),
               std::runtime_error);
  EXPECT_TRUE(same_complete_storage(coarse, coarse_snapshot));
  EXPECT_TRUE(same_complete_storage(fine, fine_snapshot));
}

TEST(test_amr_ssprk2, prepared_multilevel_replay_reuses_native_storage_and_transaction) {
  const Box2D coarse_domain = Box2D::from_extents(8, 8);
  const BoxArray coarse_layout(std::vector<Box2D>{coarse_domain});
  const DistributionMapping coarse_ownership(coarse_layout.size(), pops::n_ranks());
  const Box2D fine_box{{4, 4}, {11, 11}};
  const BoxArray fine_layout(std::vector<Box2D>{fine_box});
  const DistributionMapping fine_ownership(fine_layout.size(), pops::n_ranks());
  const Periodicity periodicity{true, true};

  MultiFab coarse(coarse_layout, coarse_ownership, 1, NoSlope::n_ghost);
  MultiFab fine(fine_layout, fine_ownership, 1, NoSlope::n_ghost);
  MultiFab coarse_aux(coarse_layout, coarse_ownership, 3, 1);
  MultiFab fine_aux(fine_layout, fine_ownership, 3, 1);
  coarse.set_val(Real(1));
  fine.set_val(Real(1));
  coarse_aux.set_val(Real(0));
  fine_aux.set_val(Real(0));
  std::vector<AmrLevelMP> levels{
      AmrLevelMP{std::move(coarse), &coarse_aux, Real(1) / 8, Real(1) / 8},
      AmrLevelMP{std::move(fine), &fine_aux, Real(1) / 16, Real(1) / 16}};

  constexpr std::uint64_t generation = 41;
  auto fill = pops::PreparedAmrFillPatchPlan::prepare(
      levels, coarse_domain, periodicity, /*coarse_replicated=*/true, generation);
  auto average = pops::PreparedAmrAverageDownPlan::prepare(levels, generation);
  auto scratch = pops::PreparedAmrAdvanceScratchPlan::prepare(
      levels, coarse_domain, periodicity, /*coarse_replicated=*/true,
      /*wave_speed_cache=*/false, generation);
  const pops::validation::AdvectionDiffusion model{/*ax=*/Real(0.2), /*ay=*/Real(-0.1),
                                                   /*nu=*/Real(0)};

  const auto step = [&] {
    pops::advance_amr<NoSlope, RusanovFlux>(
        model, levels, coarse_domain, Real(1e-3), periodicity,
        /*coarse_replicated=*/true, /*recon_prim=*/false, /*imex=*/false,
        pops::NewtonOptions{}, AmrTimeMethod::kSsprk2, /*pos_floor=*/Real(0),
        pops::kWenoEpsilon, /*wave_speed_cache=*/false, /*boundary_fill=*/nullptr, &fill,
        &average, &scratch);
  };

  step();  // materialize backend-internal schedules before observing the stable replay
  pops::device_fence();
  const pops::AllocationEventStats before = pops::allocation_event_stats();
  step();
  pops::device_fence();
  const pops::AllocationEventStats after = pops::allocation_event_stats();

  EXPECT_EQ(after, before);
}
