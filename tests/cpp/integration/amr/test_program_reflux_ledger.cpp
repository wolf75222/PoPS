// ADC-639: the conservative-reflux effective-flux LEDGER + the dt-integrated route_reflux variant for a
// whole-system compiled Program on AMR. Two contracts are frozen here, WITHOUT an engine:
//   - CoarseFineInterface::route_reflux_integrated: the /dt-free correction -(fL - cL)/dx (both sides
//     already dt-integrated in the ledger), bit-identical to route_reflux with dt == 1;
//   - the EdgeFlux ledger arithmetic (detail::edge_flux_axpy) reproduces Feff = sum_i w_i F_i through the
//     Program's linear combination -- the SSPRK2 and midpoint worked examples of design-639 section 2b.
// The full engine path (capture closures + route_reflux_program) is covered by the Python acceptances
// (test_amr_program_reflux.py, test_amr_history_regrid.py); this pins the arithmetic the driver relies on.

#include <gtest/gtest.h>

#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // CoarseFineInterface / FluxRegister
#include <pops/runtime/amr/amr_program_reflux.hpp>  // EdgeFlux / EdgeStrip / detail::edge_flux_axpy
#include <pops/runtime/program/amr_program_context.hpp>  // recursive clock/ledger driver compiles
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_ledger.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>

#include <set>
#include <limits>
#include <vector>

using namespace pops;

namespace {
// Compile-time acceptance for the generated driver's template surface.  It is deliberately not
// executed here because construction needs a materialized AmrSystem hierarchy.
[[maybe_unused]] void instantiate_recursive_driver(runtime::program::AmrProgramContext& context) {
  context.advance_hierarchy(0.1, [](double) {});
  context.register_history("qualified", 2, -1, 0, "fluid.U", "state[rho]", "clock.macro",
                           "dense.linear");
  (void)context.history_zero_start("qualified", 2, -1, 0);
}

// One fine patch [4..11]^2 -> coarse footprint [2..5]^2 (PatchRange). One component.
EdgeStrip make_strip(int I0, int I1, int J0, int J1, int nc) {
  EdgeStrip s;
  s.alloc(Box2D{{I0, J0}, {I1, J1}}, nc);
  return s;
}
}  // namespace

TEST(test_program_reflux_ledger, edge_strip_rejects_invalid_storage_shape) {
  EdgeStrip strip;
  EXPECT_THROW(strip.alloc(Box2D{}, 1), std::invalid_argument);
  EXPECT_THROW(strip.alloc(Box2D{{0, 0}, {1, 1}}, 0), std::invalid_argument);
}

TEST(test_program_reflux_ledger, prepared_transition_does_not_skip_right_or_top_only_roles) {
  const Box2D coarse_domain{{0, 0}, {7, 7}};
  const BoxArray coarse_boxes(std::vector<Box2D>{coarse_domain});
  const BoxArray fine_boxes(std::vector<Box2D>{Box2D{{4, 4}, {11, 11}}});
  const DistributionMapping coarse_mapping(coarse_boxes.size(), n_ranks());
  const DistributionMapping fine_mapping(fine_boxes.size(), n_ranks());
  AmrLevelMP coarse{MultiFab(coarse_boxes, coarse_mapping, 1, 0), nullptr, Real(0.5), Real(0.5)};
  AmrLevelMP fine{MultiFab(fine_boxes, fine_mapping, 1, 0), nullptr, Real(0.25), Real(0.25)};
  coarse.U.set_val(Real(0));
  fine.U.set_val(Real(0));
  auto transition = PreparedAmrProgramRefluxTransition::prepare(
      coarse, fine, coarse_domain, Periodicity{false, false}, world_communicator_view());

  EdgeStrip valid_coarse = make_strip(2, 5, 2, 5, 1);
  EdgeStrip valid_fine = make_strip(2, 5, 2, 5, 1);
  valid_coarse.cL.assign(4, Real(1));
  valid_coarse.cR.assign(4, Real(1));
  valid_coarse.cB.assign(4, Real(1));
  valid_coarse.cT.assign(4, Real(1));
  valid_fine.fL.assign(4, Real(2));
  valid_fine.fR.assign(4, Real(2));
  valid_fine.fB.assign(4, Real(2));
  valid_fine.fT.assign(4, Real(2));
  transition.synchronize_integrated(coarse.U, coarse.dx, coarse.dy,
                                    std::vector<EdgeStrip>{valid_coarse},
                                    std::vector<EdgeStrip>{valid_fine}, world_communicator_view());
  ASSERT_EQ(coarse.U.local_size(), 1);
  EXPECT_EQ(coarse.U.fab(0)(1, 2, 0), Real(-2));

  EdgeStrip right_only;
  right_only.I0 = right_only.J0 = 2;
  right_only.I1 = right_only.J1 = 5;
  right_only.cR.assign(4, Real(1));
  EXPECT_THROW(transition.synchronize_integrated(
                   coarse.U, coarse.dx, coarse.dy, std::vector<EdgeStrip>{right_only},
                   std::vector<EdgeStrip>{valid_fine}, world_communicator_view()),
               std::runtime_error);

  EdgeStrip top_only;
  top_only.I0 = top_only.J0 = 2;
  top_only.I1 = top_only.J1 = 5;
  top_only.fT.assign(4, Real(1));
  EXPECT_THROW(transition.synchronize_integrated(
                   coarse.U, coarse.dx, coarse.dy, std::vector<EdgeStrip>{valid_coarse},
                   std::vector<EdgeStrip>{top_only}, world_communicator_view()),
               std::runtime_error);

  EXPECT_THROW(
      transition.synchronize_integrated(coarse.U, coarse.dx, coarse.dy, std::vector<EdgeStrip>(2),
                                        std::vector<EdgeStrip>{}, world_communicator_view()),
      std::runtime_error);

  EXPECT_THROW(
      transition.synchronize_integrated(coarse.U, coarse.dx, coarse.dy, std::vector<EdgeStrip>(1),
                                        std::vector<EdgeStrip>(1), world_communicator_view()),
      std::runtime_error);

  EdgeStrip shifted_coarse = make_strip(1, 4, 2, 5, 1);
  EdgeStrip shifted_fine = make_strip(1, 4, 2, 5, 1);
  EXPECT_THROW(transition.synchronize_integrated(
                   coarse.U, coarse.dx, coarse.dy, std::vector<EdgeStrip>{shifted_coarse},
                   std::vector<EdgeStrip>{shifted_fine}, world_communicator_view()),
               std::runtime_error);
  EXPECT_EQ(coarse.U.fab(0)(1, 2, 0), Real(-2))
      << "every rejected preflight leaves the parent untouched";
}

TEST(test_program_reflux_ledger, prepared_local_kernel_routes_exact_face_corrections) {
  const Box2D coarse_domain{{0, 0}, {7, 7}};
  const BoxArray coarse_boxes(std::vector<Box2D>{coarse_domain});
  const BoxArray fine_boxes(std::vector<Box2D>{Box2D{{4, 4}, {11, 11}}});
  const DistributionMapping coarse_mapping(coarse_boxes.size(), n_ranks());
  const DistributionMapping fine_mapping(fine_boxes.size(), n_ranks());
  AmrLevelMP coarse{MultiFab(coarse_boxes, coarse_mapping, 1, 0), nullptr, Real(0.5), Real(0.5)};
  AmrLevelMP fine{MultiFab(fine_boxes, fine_mapping, 1, 0), nullptr, Real(0.25), Real(0.25)};
  coarse.U.set_val(Real(0));
  fine.U.set_val(Real(0));

  int calls = 0;
  const auto kernel = [&calls](const PreparedAmrRefluxLocalRequest& request) {
    ++calls;
    EXPECT_EQ(*request.transition_identity, "pops://test/reflux");
    EXPECT_EQ(*request.patch_identity, "pops://test/reflux/patch=0");
    EXPECT_EQ(request.parent_level, 0);
    EXPECT_EQ(request.child_level, 1);
    EXPECT_EQ(request.global_child, 0u);
    EXPECT_EQ(request.logical_time, (amr::ClockStamp{0, 3, amr::Rational(0, 1), 0.3}));
    const auto x_size =
        static_cast<std::size_t>(request.correction.J1 - request.correction.J0 + 1) *
        static_cast<std::size_t>(request.correction.components);
    const auto y_size =
        static_cast<std::size_t>(request.correction.I1 - request.correction.I0 + 1) *
        static_cast<std::size_t>(request.correction.components);
    std::fill_n(request.correction.x_low, x_size, Real(3));
    std::fill_n(request.correction.x_high, x_size, Real(4));
    std::fill_n(request.correction.y_low, y_size, Real(5));
    std::fill_n(request.correction.y_high, y_size, Real(6));
  };
  auto transition = PreparedAmrProgramRefluxTransition::prepare_with_local_kernel(
      coarse, fine, coarse_domain, Periodicity{false, false}, 0, "pops://test/reflux", kernel,
      world_communicator_view());

  EdgeStrip coarse_role = make_strip(2, 5, 2, 5, 1);
  EdgeStrip fine_role = make_strip(2, 5, 2, 5, 1);
  coarse_role.cL.assign(4, Real(1));
  coarse_role.cR.assign(4, Real(1));
  coarse_role.cB.assign(4, Real(1));
  coarse_role.cT.assign(4, Real(1));
  fine_role.fL.assign(4, Real(2));
  fine_role.fR.assign(4, Real(2));
  fine_role.fB.assign(4, Real(2));
  fine_role.fT.assign(4, Real(2));
  const amr::ClockStamp logical_time{0, 3, amr::Rational(0, 1), 0.3};
  transition.synchronize_integrated(
      coarse.U, coarse.dx, coarse.dy, std::vector<EdgeStrip>{coarse_role},
      std::vector<EdgeStrip>{fine_role}, world_communicator_view(), &logical_time);

  EXPECT_EQ(calls, 1);
  ASSERT_EQ(coarse.U.local_size(), 1);
  EXPECT_EQ(coarse.U.fab(0)(1, 3, 0), Real(3));
  EXPECT_EQ(coarse.U.fab(0)(6, 3, 0), Real(4));
  EXPECT_EQ(coarse.U.fab(0)(3, 1, 0), Real(5));
  EXPECT_EQ(coarse.U.fab(0)(3, 6, 0), Real(6));
  EXPECT_EQ(coarse.U.fab(0)(2, 2, 0), Real(0))
      << "covered parent cells remain outside the sparse correction";
}

TEST(test_program_reflux_ledger, prepared_local_kernel_rejects_unwritten_output_atomically) {
  const Box2D coarse_domain{{0, 0}, {7, 7}};
  const BoxArray coarse_boxes(std::vector<Box2D>{coarse_domain});
  const BoxArray fine_boxes(std::vector<Box2D>{Box2D{{4, 4}, {11, 11}}});
  const DistributionMapping coarse_mapping(coarse_boxes.size(), n_ranks());
  const DistributionMapping fine_mapping(fine_boxes.size(), n_ranks());
  AmrLevelMP coarse{MultiFab(coarse_boxes, coarse_mapping, 1, 0), nullptr, Real(0.5), Real(0.5)};
  AmrLevelMP fine{MultiFab(fine_boxes, fine_mapping, 1, 0), nullptr, Real(0.25), Real(0.25)};
  coarse.U.set_val(Real(0));
  fine.U.set_val(Real(0));

  const auto incomplete = [](const PreparedAmrRefluxLocalRequest& request) {
    const auto x_size =
        static_cast<std::size_t>(request.correction.J1 - request.correction.J0 + 1) *
        static_cast<std::size_t>(request.correction.components);
    std::fill_n(request.correction.x_low, x_size, Real(1));
  };
  auto transition = PreparedAmrProgramRefluxTransition::prepare_with_local_kernel(
      coarse, fine, coarse_domain, Periodicity{false, false}, 0, "pops://test/reflux", incomplete,
      world_communicator_view());

  EdgeStrip coarse_role = make_strip(2, 5, 2, 5, 1);
  EdgeStrip fine_role = make_strip(2, 5, 2, 5, 1);
  coarse_role.cL.assign(4, Real(1));
  coarse_role.cR.assign(4, Real(1));
  coarse_role.cB.assign(4, Real(1));
  coarse_role.cT.assign(4, Real(1));
  fine_role.fL.assign(4, Real(2));
  fine_role.fR.assign(4, Real(2));
  fine_role.fB.assign(4, Real(2));
  fine_role.fT.assign(4, Real(2));
  const amr::ClockStamp logical_time{0, 3, amr::Rational(0, 1), 0.3};
  EXPECT_THROW(transition.synchronize_integrated(
                   coarse.U, coarse.dx, coarse.dy, std::vector<EdgeStrip>{coarse_role},
                   std::vector<EdgeStrip>{fine_role}, world_communicator_view(), &logical_time),
               std::runtime_error);

  ASSERT_EQ(coarse.U.local_size(), 1);
  EXPECT_EQ(coarse.U.fab(0)(1, 3, 0), Real(0));
  EXPECT_EQ(coarse.U.fab(0)(6, 3, 0), Real(0));
  EXPECT_EQ(coarse.U.fab(0)(3, 1, 0), Real(0));
  EXPECT_EQ(coarse.U.fab(0)(3, 6, 0), Real(0));
}

TEST(test_program_reflux_ledger, edge_flux_axpy_rejects_shifted_equal_width_footprints) {
  EdgeFlux destination;
  destination.fine.push_back(make_strip(2, 5, 2, 5, 1));
  EdgeFlux shifted;
  shifted.fine.push_back(make_strip(3, 6, 2, 5, 1));
  EXPECT_THROW(detail::edge_flux_axpy(destination, Real(1), shifted), std::invalid_argument);
}

TEST(test_program_reflux_ledger, route_reflux_integrated_drops_dt) {
  // Same fixture as test_cf_interface but the strips already carry dt*Feff, so the correction is
  // -(fL - cL)/dx (NO *dt). We check the integrated variant equals route_reflux with dt == 1.
  const int nc = 1;
  BoxArray fine(std::vector<Box2D>{Box2D{{4, 4}, {11, 11}}});
  CoarseFineInterface cfi(Box2D{{0, 0}, {7, 7}}, fine, Periodicity{false, false});
  EdgeStrip g = make_strip(2, 5, 2, 5, nc);
  const int nJ = 4 * nc, nI = 4 * nc;
  g.cL.assign(nJ, Real(1));
  g.cR.assign(nJ, Real(2));
  g.cB.assign(nI, Real(3));
  g.cT.assign(nI, Real(4));
  g.fL.assign(nJ, Real(10));
  g.fR.assign(nJ, Real(20));
  g.fB.assign(nI, Real(30));
  g.fT.assign(nI, Real(40));
  const Real dx = Real(0.5), dy = Real(0.25);
  FluxRegister ref(Box2D{{1, 1}, {6, 6}}, nc);
  cfi.route_reflux_integrated(g, dx, dy, ref, nc);
  device_fence();
  // -(fL - cL)/dx = -(10 - 1)/0.5 = -18; +(fR - cR)/dx = (20 - 2)/0.5 = 36.
  EXPECT_EQ(ref.at(1, 2, 0), -(Real(10) - Real(1)) / dx) << "integrated_left";
  EXPECT_EQ(ref.at(6, 2, 0), +(Real(20) - Real(2)) / dx) << "integrated_right";
  EXPECT_EQ(ref.at(2, 1, 0), -(Real(30) - Real(3)) / dy) << "integrated_bottom";
  EXPECT_EQ(ref.at(2, 6, 0), +(Real(40) - Real(4)) / dy) << "integrated_top";

  // Equivalence to route_reflux at dt == 1 (both formulas coincide when the coarse side is not re-scaled).
  FluxRegister ref_dt1(Box2D{{1, 1}, {6, 6}}, nc);
  cfi.route_reflux(g, dx, dy, Real(1), ref_dt1, nc);
  device_fence();
  EXPECT_EQ(ref.at(1, 2, 0), ref_dt1.at(1, 2, 0)) << "integrated_eq_dt1";
}

// A helper building a single-patch EdgeFlux carrying the FINE-role strip with a scalar flux value F, so
// the ledger arithmetic can be checked on a representative face. The x-left face carries F on the whole
// strip; the other faces mirror it (a uniform flux -- enough to prove the linear-combination weights).
static EdgeFlux fine_flux(Real F, int nc = 1) {
  EdgeStrip s = make_strip(2, 5, 2, 5, nc);
  const std::size_t nJ = s.fL.size(), nI = s.fB.size();
  s.fL.assign(nJ, F);
  s.fR.assign(nJ, F);
  s.fB.assign(nI, F);
  s.fT.assign(nI, F);
  EdgeFlux ef;
  ef.fine.push_back(s);
  return ef;
}

static EdgeStrip ranked_strip(int patch, Real value) {
  EdgeStrip strip = make_strip(2 + 4 * patch, 5 + 4 * patch, 2, 5, 1);
  strip.cL.assign(strip.cL.size(), value);
  strip.cR.assign(strip.cR.size(), value);
  strip.cB.assign(strip.cB.size(), value);
  strip.cT.assign(strip.cT.size(), value);
  strip.fL.assign(strip.fL.size(), value);
  strip.fR.assign(strip.fR.size(), value);
  strip.fB.assign(strip.fB.size(), value);
  strip.fT.assign(strip.fT.size(), value);
  return strip;
}

static runtime::program::AmrProgramAcceptedState ranked_accepted_state(
    int rank, const runtime::program::AmrProgramRankOwnership& ownership) {
  runtime::program::AmrProgramAcceptedState state;
  state.level_clocks = {{0, 7, amr::Rational(0, 1), 0.7}, {1, 7, amr::Rational(0, 1), 0.7}};
  state.logical_clock_ticks = {{"clock.macro", 7}, {"clock.fine", 14}};
  state.tagging_hysteresis_state = {9, 8, 7};
  state.history_owners["rhs"] = 0;
  state.history_states["rhs"] = "fluid.U";
  state.history_spaces["rhs"] = "cell.conservative";
  state.history_clocks["rhs"] = "clock.macro";
  state.history_interpolations["rhs"] = "dense.linear";
  state.ring_clocks["rhs"] = {{{0, 7, amr::Rational(0, 1), 0.7}, {1, 7, amr::Rational(0, 1), 0.7}}};
  state.ring_identities["rhs"].resize(1, std::vector<std::optional<amr::HistoryIdentity>>(2));
  for (int level = 0; level < 2; ++level) {
    const amr::ClockStamp clock = state.ring_clocks["rhs"][0][static_cast<std::size_t>(level)];
    state.ring_identities["rhs"][0][static_cast<std::size_t>(level)] =
        amr::HistoryIdentity{"program.block.0", "fluid.U", "cell.conservative", level, clock};
  }
  state.ring_flux["rhs"].resize(1);
  state.ring_flux["rhs"][0].resize(2);
  state.ring_flux_contributions["rhs"].resize(1);
  state.ring_flux_contributions["rhs"][0].resize(2);
  state.ring_flux_initialized["rhs"] = {1, 1};

  const std::vector<int>& fine_owners = ownership.level_patch_owners[1];
  EdgeFlux& coarse_role = state.ring_flux["rhs"][0][0];
  EdgeFlux& fine_role = state.ring_flux["rhs"][0][1];
  coarse_role.coarse.resize(fine_owners.size());
  fine_role.fine.resize(fine_owners.size());
  auto& contributions = state.ring_flux_contributions["rhs"][0][1];
  contributions.push_back(
      {23, amr::Rational(2, 3), 1, 0.05, {1, 7, amr::Rational(1, 2), 0.675}, {}});
  contributions[0].payload.fine.resize(fine_owners.size());
  for (std::size_t patch = 0; patch < fine_owners.size(); ++patch)
    if (fine_owners[patch] == rank) {
      coarse_role.coarse[patch] = ranked_strip(static_cast<int>(patch), Real(10 + patch));
      fine_role.fine[patch] = ranked_strip(static_cast<int>(patch), Real(20 + patch));
      contributions[0].payload.fine[patch] =
          ranked_strip(static_cast<int>(patch), Real(30 + patch));
    }
  state.accepted_flux_ledger.push_back(
      {{"fluid", "U", "transport", "physical_flux", 1, {1, 7, amr::Rational(1, 2), 0.675}},
       {amr::Rational(2, 3), amr::FluxOrientation::XPlus, 0.25, 0.05}});
  state.accepted_interface_flux_ledger.push_back(
      {{"shared.rank-change",
        3,
        0,
        1,
        {1, 7, amr::Rational(1, 2), 0.675},
        "program.group.node.51",
        {{1, 7, amr::Rational(0, 1), 0.65}, {1, 7, amr::Rational(1, 1), 0.7}},
        amr::InterfaceFluxOrientation::FineOutward,
        0,
        1},
       {amr::Rational(2, 3), 0.125, 0.05, true}});
  state.accepted_sync.push_back({0, 1, 0, 1, {0, 7, amr::Rational(1, 1), 0.7}});
  return state;
}

TEST(test_program_reflux_ledger, ssprk2_effective_flux) {
  // SSPRK2 (design 2b): the terminal commit strip must hold dt*(1/2 F0 + 1/2 F1).
  // Emulate the codegen lowering:
  //   k0.ledger = F0;  k1.ledger = F1;
  //   U1 = axpy(0, 1*U0) + axpy(dt*k0)          -> u_U1.ledger = dt*F0
  //   acc = axpy(0.5*U1) + axpy(0.5*dt*k1)      -> acc.ledger  = 0.5*dt*F0 + 0.5*dt*F1
  //   commit: lincomb(U, 0.5, U, 1, acc)        -> U.ledger    = 0.5*0 + acc = dt*(1/2 F0 + 1/2 F1)
  const Real dt = Real(0.1), F0 = Real(3), F1 = Real(5);
  const EdgeFlux led_k0 = fine_flux(F0), led_k1 = fine_flux(F1);

  EdgeFlux u_U1;                             // U1 = U0 + dt*k0 (U0 has no flux)
  detail::edge_flux_axpy(u_U1, dt, led_k0);  // ledger[U1] += dt*F0

  EdgeFlux acc;                                         // acc = 0.5*U1 + 0.5*dt*k1
  detail::edge_flux_axpy(acc, Real(0.5), u_U1);         // += 0.5*dt*F0
  detail::edge_flux_axpy(acc, Real(0.5) * dt, led_k1);  // += 0.5*dt*F1

  EdgeFlux commit;                               // lincomb(U, 0.5, U(=0), 1, acc)
  detail::edge_flux_axpy(commit, Real(1), acc);  // U.ledger = acc

  // Same grouping as the ledger accumulation (0.5*(dt*F0) then += (0.5*dt)*F1): the exact reflux value.
  const Real expected = Real(0.5) * (dt * F0) + (Real(0.5) * dt) * F1;
  ASSERT_FALSE(commit.fine.empty());
  EXPECT_EQ(commit.fine[0].fL[0], expected) << "ssprk2_Feff";
  EXPECT_EQ(commit.fine[0].fT[0], expected) << "ssprk2_Feff_top";
}

TEST(test_program_reflux_ledger, prepared_edge_flux_copy_reuses_pinned_storage) {
  const EdgeFlux first = fine_flux(Real(3));
  const EdgeFlux second = fine_flux(Real(7));
  EdgeFlux resident;

  detail::edge_flux_copy_into(resident, first);  // cold materialisation
  const AllocationEventStats before = allocation_event_stats();
  detail::edge_flux_copy_into(resident, second);  // exact-layout hot replay
  const AllocationEventStats after = allocation_event_stats();

  EXPECT_EQ(after, before);
  ASSERT_EQ(resident.fine.size(), 1u);
  EXPECT_EQ(resident.fine[0].fL[0], Real(7));
  EXPECT_EQ(resident.fine[0].fT[0], Real(7));
}

TEST(test_program_reflux_ledger, midpoint_effective_flux_is_F1_only) {
  // Midpoint (design 2b): U1 = U0 + 0.5 dt k0; U <<= U0 + dt k1. The terminal strip must hold dt*F1
  // (the second-stage flux ONLY) -- proving the Program text, not a hard-coded RK, drives Feff.
  const Real dt = Real(0.1), F0 = Real(3), F1 = Real(5);
  const EdgeFlux led_k0 = fine_flux(F0), led_k1 = fine_flux(F1);

  EdgeFlux u_U1;  // U1 = U0 + 0.5 dt k0
  detail::edge_flux_axpy(u_U1, Real(0.5) * dt,
                         led_k0);  // ledger[U1] = 0.5 dt F0 (not read by the commit)

  EdgeFlux acc;                             // acc = dt*k1  (U0 is the base, coeff on k1 is dt)
  detail::edge_flux_axpy(acc, dt, led_k1);  // += dt*F1

  EdgeFlux commit;  // lincomb(U, 1, U(=0), 1, acc)
  detail::edge_flux_axpy(commit, Real(1), acc);

  ASSERT_FALSE(commit.fine.empty());
  EXPECT_EQ(commit.fine[0].fL[0], dt * F1) << "midpoint_Feff_is_F1";
  // and NOT the SSPRK2 answer (proves the weights come from the Program combine).
  EXPECT_NE(commit.fine[0].fL[0], dt * (Real(0.5) * F0 + Real(0.5) * F1)) << "midpoint_differs";
}

TEST(test_program_reflux_ledger, ab2_effective_flux_uses_lagged_flux) {
  // AB2 (design 2b / acceptance e): U <<= U + dt*(1.5 R_n - 0.5 R_{n-1}). The lagged R_{n-1} flux rides
  // with the history ring (ring_flux_ in the context); here we emulate the commit combine given that both
  // R_n and R_{n-1} strips are present in the ledger. The terminal strip must hold dt*(1.5 F_n - 0.5 F_nm1).
  const Real dt = Real(0.1), Fn = Real(7), Fnm1 = Real(2);
  const EdgeFlux led_Rn = fine_flux(Fn), led_Rnm1 = fine_flux(Fnm1);

  EdgeFlux acc;  // acc = 1.5 dt R_n + (-0.5) dt R_{n-1}
  detail::edge_flux_axpy(acc, Real(1.5) * dt, led_Rn);
  detail::edge_flux_axpy(acc, Real(-0.5) * dt, led_Rnm1);

  EdgeFlux commit;  // lincomb(U, 1, U(=0), 1, acc)
  detail::edge_flux_axpy(commit, Real(1), acc);

  // The ledger accumulates the two weighted terms in SEQUENCE (mirror of the codegen axpy chain), so the
  // expected value must use the SAME grouping / rounding order -- 1.5*dt*Fn then += (-0.5*dt)*Fnm1 -- not a
  // single fused dt*(1.5 Fn - 0.5 Fnm1) (a different float rounding). This IS the reflux the driver applies.
  const Real expected = Real(1.5) * dt * Fn + (Real(-0.5) * dt) * Fnm1;
  ASSERT_FALSE(commit.fine.empty());
  EXPECT_EQ(commit.fine[0].fL[0], expected) << "ab2_Feff_uses_lagged";
}

TEST(test_program_reflux_ledger, rational_arithmetic_is_checked_before_int64_overflow) {
  const std::int64_t limit = std::numeric_limits<std::int64_t>::max();
  const std::int64_t minimum = std::numeric_limits<std::int64_t>::min();
  EXPECT_THROW((void)(amr::Rational(limit, 1) + amr::Rational(1, 1)), std::overflow_error);
  EXPECT_THROW((void)(amr::Rational(limit, 1) * amr::Rational(2, 1)), std::overflow_error);
  EXPECT_THROW((void)(amr::Rational(-limit, 1) - amr::Rational(2, 1)), std::overflow_error);
  EXPECT_EQ(amr::Rational(minimum, 1).numerator, minimum);
  EXPECT_EQ(amr::Rational(minimum + 1, 1) - amr::Rational(1, 1), amr::Rational(minimum, 1));
  EXPECT_EQ(amr::Rational(minimum, 3) + amr::Rational(-limit, 3),
            amr::Rational(-6148914691236517205LL, 1));
  EXPECT_EQ(amr::Rational(minimum, 3) - amr::Rational(limit, 3),
            amr::Rational(-6148914691236517205LL, 1));
  EXPECT_EQ(amr::Rational(-limit, 5) + amr::Rational(3074457345618258602LL, 1),
            amr::Rational(6148914691236517203LL, 5));
  EXPECT_EQ(amr::Rational(minimum, 1) / amr::Rational(minimum, 1), amr::Rational(1, 1));
  EXPECT_EQ(amr::Rational(2, minimum), amr::Rational(-1, std::int64_t{1} << 62));
  EXPECT_THROW(amr::Rational(1, minimum), std::overflow_error);

  // These cross products overflow int64, but comparison is exact and multiplication cross-cancels.
  EXPECT_TRUE(amr::Rational(minimum, 1) < amr::Rational(-limit, 1));
  EXPECT_TRUE(amr::Rational(limit - 1, limit) < amr::Rational(limit, limit - 1));
  EXPECT_EQ(amr::Rational(limit, 2) * amr::Rational(2, limit), amr::Rational(1, 1));
}

TEST(test_program_reflux_ledger, accepted_exact_ledger_is_the_unique_numerical_reflux_source) {
  amr::TransactionalFluxLedger<EdgeFlux> ledger;
  amr::FluxLedgerKey key{"fluid",         "U", "transport",
                         "physical_flux", 1,   {1, 3, amr::Rational(1, 2), 0.15}};
  ledger.begin();
  ledger.accumulate(key, {amr::Rational(1, 2), amr::FluxOrientation::XMinus, 17.0, 0.1},
                    fine_flux(Real(3)));
  ledger.accumulate(key, {amr::Rational(1, 2), amr::FluxOrientation::XMinus, 0.25, 0.1},
                    fine_flux(Real(5)));
  ledger.begin();
  ledger.accumulate(key, {amr::Rational(1, 1), amr::FluxOrientation::XMinus, 9.0, 0.2},
                    fine_flux(Real(100)));
  ledger.rollback();  // rejected contribution cannot reach the numerical route

  const auto accepted_by_key =
      ledger.aggregate([](EdgeFlux& destination, double scale, const EdgeFlux& payload) {
        detail::edge_flux_axpy(destination, static_cast<Real>(scale), payload);
      });
  const EdgeFlux accepted = accepted_by_key.at(key);
  ledger.commit();
  ASSERT_EQ(accepted.fine.size(), 1u);
  const Real expected = Real(0.5) * Real(0.1) * Real(3) + Real(0.5) * Real(0.1) * Real(5);
  EXPECT_EQ(accepted.fine[0].fL[0], expected);
  EXPECT_EQ(
      amr::numerical_reflux_scale({amr::Rational(1, 2), amr::FluxOrientation::XMinus, 1000.0, 0.1}),
      0.05)
      << "face measure is audited but route_reflux_integrated applies geometry exactly once";

  BoxArray fine(std::vector<Box2D>{Box2D{{4, 4}, {11, 11}}});
  CoarseFineInterface cfi(Box2D{{0, 0}, {7, 7}}, fine, Periodicity{false, false});
  FluxRegister correction(Box2D{{1, 1}, {6, 6}}, 1);
  cfi.route_reflux_integrated(accepted.fine[0], Real(0.5), Real(0.5), correction, 1);
  device_fence();
  EXPECT_EQ(correction.at(1, 2, 0), -expected / Real(0.5));
  EXPECT_NE(correction.at(1, 2, 0), -Real(100) * Real(0.2) / Real(0.5))
      << "the rolled-back shadow never participates";
}

TEST(test_program_reflux_ledger, ratio_two_clocks_and_interpolation_are_exact) {
  const amr::ClockWindow parent{{0, 7, amr::Rational(0, 1), 2.0}, {0, 7, amr::Rational(1, 1), 2.4}};
  const amr::ParentChildClockRelation relation(0, 1, amr::Rational(2, 1),
                                               amr::RemainderPolicy::IntegralOnly);
  const auto children = relation.partition(parent);
  ASSERT_EQ(children.size(), 2u);
  EXPECT_EQ(children[0].window.begin.phase, amr::Rational(0, 1));
  EXPECT_EQ(children[0].window.end.phase, amr::Rational(1, 2));
  EXPECT_EQ(children[1].window.begin.phase, amr::Rational(1, 2));
  EXPECT_EQ(children[1].window.end.phase, amr::Rational(1, 1));
  EXPECT_EQ(parent.alpha({0, 7, amr::Rational(1, 2), 2.2}), amr::Rational(1, 2));
}

TEST(test_program_reflux_ledger,
     logical_clock_domains_have_exact_nested_ticks_and_restart_cursors) {
  runtime::program::ClockScheduleState clocks;
  clocks.configure_primary_clock("clock.macro");
  clocks.declare_relation("clock.macro", "clock.fine", 2);
  clocks.declare_relation("clock.fine", "clock.micro", 3);

  const auto accepted = clocks.accepted_ticks(4);
  EXPECT_EQ(accepted.at("clock.macro"), 4);
  EXPECT_EQ(accepted.at("clock.fine"), 8);
  EXPECT_EQ(accepted.at("clock.micro"), 24);
  clocks.restore_accepted_ticks(accepted, 4);
  EXPECT_EQ(clocks.restored_accepted_ticks(), accepted);

  const auto macro = clocks.coordinate(runtime::program::ScheduleDomainKind::kAcceptedStep,
                                       "clock.macro", "", -1, -1, 4);
  ASSERT_TRUE(macro.has_value());
  EXPECT_EQ(macro->value, 4);
  const auto level = clocks.coordinate(runtime::program::ScheduleDomainKind::kAmrLevel,
                                       "clock.macro", "", 1, 1, 4);
  ASSERT_TRUE(level.has_value());
  EXPECT_EQ(level->value, 4);
  EXPECT_FALSE(clocks.coordinate(runtime::program::ScheduleDomainKind::kAmrLevel, "clock.macro", "",
                                 1, 0, 4));

  auto fine = clocks.subcycle("clock.macro", "clock.fine", 2);
  fine.iteration(0);
  const auto fine_tick = clocks.coordinate(runtime::program::ScheduleDomainKind::kClockTick,
                                           "clock.fine", "", -1, -1, 4);
  ASSERT_TRUE(fine_tick.has_value());
  EXPECT_EQ(fine_tick->value, 8);
  const auto fine_level_tick =
      clocks.coordinate(runtime::program::ScheduleDomainKind::kAmrLevel, "clock.fine", "", 1, 1, 4);
  ASSERT_TRUE(fine_level_tick.has_value());
  EXPECT_EQ(fine_level_tick->value, 8);
  auto micro = clocks.subcycle("clock.fine", "clock.micro", 3);
  for (int k = 0; k < 3; ++k) {
    micro.iteration(k);
    const auto micro_tick = clocks.coordinate(runtime::program::ScheduleDomainKind::kClockTick,
                                              "clock.micro", "", -1, -1, 4);
    ASSERT_TRUE(micro_tick.has_value());
    EXPECT_EQ(micro_tick->value, 24 + k);
  }
  micro.finish();
  fine.iteration(1);
  const auto second_fine_tick = clocks.coordinate(runtime::program::ScheduleDomainKind::kClockTick,
                                                  "clock.fine", "", -1, -1, 4);
  ASSERT_TRUE(second_fine_tick.has_value());
  EXPECT_EQ(second_fine_tick->value, 9);
  fine.finish();

  auto corrupted = accepted;
  ++corrupted["clock.micro"];
  EXPECT_THROW(clocks.restore_accepted_ticks(corrupted, 4), std::runtime_error);
}

TEST(test_program_reflux_ledger, history_identity_includes_owner_state_space_and_clock) {
  const amr::ClockStamp clock{1, 3, amr::Rational(1, 2), 0.15};
  const amr::HistoryIdentity base{"fluid", "U", "cell-conservative", 1, clock};
  std::set<amr::HistoryIdentity> identities;
  identities.insert(base);
  identities.insert({"other", "U", "cell-conservative", 1, clock});
  identities.insert({"fluid", "V", "cell-conservative", 1, clock});
  identities.insert({"fluid", "U", "face-flux", 1, clock});
  identities.insert({"fluid", "U", "cell-conservative", 1, {1, 3, amr::Rational(3, 4), 0.2}});
  EXPECT_EQ(identities.size(), 5u);
}

TEST(test_program_reflux_ledger, non_integral_relation_requires_declared_remainder) {
  const amr::ClockWindow parent{{0, 0, amr::Rational(0, 1), 0.0}, {0, 0, amr::Rational(1, 1), 1.0}};
  const amr::ParentChildClockRelation rejected(0, 1, amr::Rational(3, 2),
                                               amr::RemainderPolicy::IntegralOnly);
  EXPECT_THROW(rejected.partition(parent), std::runtime_error);

  const amr::ParentChildClockRelation declared(0, 1, amr::Rational(3, 2),
                                               amr::RemainderPolicy::ExplicitFinalSubstep);
  const auto children = declared.partition(parent);
  ASSERT_EQ(children.size(), 2u);
  EXPECT_EQ(children[0].window.end.phase, amr::Rational(2, 3));
  EXPECT_TRUE(children[1].is_declared_remainder);
  EXPECT_EQ(children[1].window.begin.phase, amr::Rational(2, 3));
  EXPECT_EQ(children[1].window.end.phase, amr::Rational(1, 1));
}

namespace {
amr::FluxLedgerKey scalar_key() {
  return {"fluid", "U", "transport", "physical_flux", 1, {1, 3, amr::Rational(1, 2), 0.15}};
}
void scalar_axpy(double& dst, double a, const double& src) {
  dst += a * src;
}
}  // namespace

TEST(test_program_reflux_ledger, exact_rk_weights_are_distinct_and_transactional) {
  amr::TransactionalFluxLedger<double> rk2;
  rk2.begin();
  rk2.accumulate(scalar_key(), {amr::Rational(1, 2), amr::FluxOrientation::XPlus, 2.0, 0.1}, 3.0);
  rk2.accumulate(scalar_key(), {amr::Rational(1, 2), amr::FluxOrientation::XPlus, 2.0, 0.1}, 9.0);
  const double rk2_value = rk2.aggregate(scalar_axpy).begin()->second;
  rk2.commit();

  amr::TransactionalFluxLedger<double> rk3;
  rk3.begin();
  rk3.accumulate(scalar_key(), {amr::Rational(1, 6), amr::FluxOrientation::XPlus, 2.0, 0.1}, 3.0);
  rk3.accumulate(scalar_key(), {amr::Rational(1, 6), amr::FluxOrientation::XPlus, 2.0, 0.1}, 9.0);
  rk3.accumulate(scalar_key(), {amr::Rational(2, 3), amr::FluxOrientation::XPlus, 2.0, 0.1}, 15.0);
  const double rk3_value = rk3.aggregate(scalar_axpy).begin()->second;
  EXPECT_NE(rk2_value, rk3_value);

  rk3.begin();
  rk3.accumulate(scalar_key(), {amr::Rational(1, 1), amr::FluxOrientation::YMinus, 2.0, 0.05},
                 100.0);
  rk3.rollback();
  EXPECT_EQ(rk3.size(), 3u) << "inner rejection leaves no contribution";
  rk3.rollback();
  EXPECT_TRUE(rk3.empty()) << "outer rejection leaves zero residual";
}

TEST(test_program_reflux_ledger, accepted_checkpoint_state_round_trips_canonically) {
  runtime::program::AmrProgramAcceptedState state;
  state.level_clocks = {{0, 9, amr::Rational(0, 1), 0.9}, {1, 9, amr::Rational(0, 1), 0.9}};
  state.logical_clock_ticks = {{"clock.macro", 9}, {"clock.fine", 18}};
  state.tagging_hysteresis_state = {1, 3, 3, 7};
  state.history_owners["rhs"] = 0;
  state.history_states["rhs"] = "fluid.U";
  state.history_spaces["rhs"] = "cell.conservative";
  state.history_clocks["rhs"] = "clock.macro";
  state.history_interpolations["rhs"] = "dense.linear";
  state.ring_clocks["rhs"] = {{{0, 9, amr::Rational(0, 1), 0.9}, {1, 9, amr::Rational(0, 1), 0.9}},
                              {{0, 8, amr::Rational(0, 1), 0.8}, {1, 8, amr::Rational(0, 1), 0.8}}};
  state.ring_identities["rhs"].resize(2, std::vector<std::optional<amr::HistoryIdentity>>(2));
  for (int slot = 0; slot < 2; ++slot)
    for (int level = 0; level < 2; ++level) {
      const auto clock =
          state.ring_clocks["rhs"][static_cast<std::size_t>(slot)][static_cast<std::size_t>(level)];
      state
          .ring_identities["rhs"][static_cast<std::size_t>(slot)][static_cast<std::size_t>(level)] =
          amr::HistoryIdentity{"program.block.0", "fluid.U", "cell.conservative", level, clock};
    }
  state.ring_flux["rhs"] = {{fine_flux(Real(7)), fine_flux(Real(8))},
                            {fine_flux(Real(5)), fine_flux(Real(6))}};
  auto& contributions = state.ring_flux_contributions["rhs"];
  contributions.resize(2);
  for (auto& slot : contributions)
    slot.resize(2);
  contributions[1][0].push_back(
      {17, amr::Rational(-1, 2), 1, 0.1, {0, 8, amr::Rational(1, 2), 0.85}, fine_flux(Real(5))});
  state.ring_flux_initialized["rhs"] = {1, 1};
  state.accepted_flux_ledger.push_back(
      {scalar_key(), {amr::Rational(2, 3), amr::FluxOrientation::YPlus, 0.25, 0.05}});
  state.accepted_interface_flux_ledger.push_back(
      {{"shared.face",
        4,
        0,
        1,
        {1, 9, amr::Rational(1, 2), 0.85},
        "program.group.node.42",
        {{1, 9, amr::Rational(0, 1), 0.8}, {1, 9, amr::Rational(1, 1), 0.9}},
        amr::InterfaceFluxOrientation::FineOutward,
        0,
        1},
       {amr::Rational(2, 3), 0.125, 0.1, true}});
  state.accepted_sync.push_back({0, 1, 0, 0, {0, 9, amr::Rational(1, 1), 0.9}});
  state.accepted_sync.push_back({0, 1, 0, 1, {0, 9, amr::Rational(1, 1), 0.9}});

  const auto encoded = runtime::program::serialize_amr_program_accepted_state(state);
  const auto decoded = runtime::program::deserialize_amr_program_accepted_state(encoded);
  EXPECT_EQ(decoded.level_clocks, state.level_clocks);
  EXPECT_EQ(decoded.logical_clock_ticks, state.logical_clock_ticks);
  EXPECT_EQ(decoded.tagging_hysteresis_state, state.tagging_hysteresis_state);
  EXPECT_EQ(decoded.history_owners, state.history_owners);
  EXPECT_EQ(decoded.history_states, state.history_states);
  EXPECT_EQ(decoded.history_spaces, state.history_spaces);
  EXPECT_EQ(decoded.history_clocks, state.history_clocks);
  EXPECT_EQ(decoded.history_interpolations, state.history_interpolations);
  EXPECT_EQ(decoded.ring_flux.at("rhs")[1][0].fine[0].fL[0], Real(5));
  const auto& contribution = decoded.ring_flux_contributions.at("rhs")[1][0][0];
  EXPECT_EQ(contribution.rate_id, 17);
  EXPECT_EQ(contribution.weight, amr::Rational(-1, 2));
  EXPECT_EQ(contribution.dt_power, 1);
  EXPECT_EQ(contribution.duration, 0.1);
  EXPECT_EQ(contribution.evaluation_clock.phase, amr::Rational(1, 2));
  ASSERT_EQ(decoded.accepted_flux_ledger.size(), 1u);
  EXPECT_EQ(decoded.accepted_flux_ledger[0].measure.stage_weight, amr::Rational(2, 3));
  EXPECT_EQ(decoded.accepted_flux_ledger[0].measure.orientation, amr::FluxOrientation::YPlus);
  ASSERT_EQ(decoded.accepted_interface_flux_ledger.size(), 1u);
  EXPECT_EQ(decoded.accepted_interface_flux_ledger[0].key.interface_identity, "shared.face");
  EXPECT_EQ(decoded.accepted_interface_flux_ledger[0].key.topology_epoch, 4u);
  EXPECT_EQ(decoded.accepted_interface_flux_ledger[0].key.interval.end.phase, amr::Rational(1, 1));
  EXPECT_EQ(decoded.accepted_interface_flux_ledger[0].measure.stage_weight, amr::Rational(2, 3));
  EXPECT_TRUE(decoded.accepted_interface_flux_ledger[0].measure.stage_weight_resolved);
  ASSERT_EQ(decoded.accepted_sync.size(), 2u);
  EXPECT_EQ(decoded.accepted_sync[0].phase, 0);
  EXPECT_EQ(decoded.accepted_sync[1].phase, 1);
  EXPECT_EQ(runtime::program::serialize_amr_program_accepted_state(decoded), encoded)
      << "the accepted-state byte protocol is deterministic";
}

TEST(test_program_reflux_ledger, accepted_checkpoint_state_refuses_corruption) {
  runtime::program::AmrProgramAcceptedState state;
  state.level_clocks = {{0, 1, amr::Rational(0, 1), 0.1}};
  auto encoded = runtime::program::serialize_amr_program_accepted_state(state);
  encoded[0] ^= 0xffU;
  EXPECT_THROW(runtime::program::deserialize_amr_program_accepted_state(encoded),
               std::runtime_error);
  encoded = runtime::program::serialize_amr_program_accepted_state(state);
  encoded.pop_back();
  EXPECT_THROW(runtime::program::deserialize_amr_program_accepted_state(encoded),
               std::runtime_error);
}

TEST(test_program_reflux_ledger,
     accepted_checkpoint_state_rematerializes_payloads_by_explicit_ownership) {
  const runtime::program::AmrProgramRankOwnership source_ownership{2, {{0, 1}, {0, 1, 0}}};
  const runtime::program::AmrProgramRankOwnership target_ownership{3, {{2, 0}, {2, 2, 0}}};
  std::vector<runtime::program::AmrProgramAcceptedState> source_states;
  source_states.push_back(ranked_accepted_state(0, source_ownership));
  source_states.push_back(ranked_accepted_state(1, source_ownership));

  const auto target_states = runtime::program::rematerialize_amr_program_accepted_states(
      source_states, source_ownership, target_ownership);
  ASSERT_EQ(target_states.size(), 3u);
  for (const auto& state : target_states) {
    EXPECT_EQ(state.level_clocks, source_states[0].level_clocks);
    EXPECT_EQ(state.logical_clock_ticks, source_states[0].logical_clock_ticks);
    EXPECT_EQ(state.tagging_hysteresis_state, source_states[0].tagging_hysteresis_state);
    ASSERT_EQ(state.accepted_interface_flux_ledger.size(), 1u);
    EXPECT_EQ(state.accepted_interface_flux_ledger[0].key.interface_identity, "shared.rank-change");
    EXPECT_EQ(state.accepted_interface_flux_ledger[0].key.topology_epoch, 3u);
    EXPECT_EQ(state.accepted_interface_flux_ledger[0].measure.stage_weight, amr::Rational(2, 3));
    EXPECT_DOUBLE_EQ(state.accepted_interface_flux_ledger[0].measure.face_measure, 0.125);
    EXPECT_TRUE(state.accepted_interface_flux_ledger[0].measure.stage_weight_resolved);
    ASSERT_EQ(state.ring_flux.at("rhs")[0][0].coarse.size(), 3u);
    ASSERT_EQ(state.ring_flux.at("rhs")[0][1].fine.size(), 3u);
    ASSERT_EQ(state.ring_flux_contributions.at("rhs")[0][1].size(), 1u);
    EXPECT_EQ(state.ring_flux_contributions.at("rhs")[0][1][0].rate_id, 23);
    EXPECT_EQ(state.ring_flux_contributions.at("rhs")[0][1][0].duration, 0.05);
  }

  const auto active = [](const EdgeStrip& strip) { return detail::edge_strip_has_storage(strip); };
  const auto& rank_zero_flux = target_states[0].ring_flux.at("rhs")[0][1].fine;
  const auto& rank_one_flux = target_states[1].ring_flux.at("rhs")[0][1].fine;
  const auto& rank_two_flux = target_states[2].ring_flux.at("rhs")[0][1].fine;
  EXPECT_FALSE(active(rank_zero_flux[0]));
  EXPECT_FALSE(active(rank_zero_flux[1]));
  EXPECT_TRUE(active(rank_zero_flux[2]));
  EXPECT_FALSE(active(rank_one_flux[0]));
  EXPECT_FALSE(active(rank_one_flux[1]));
  EXPECT_FALSE(active(rank_one_flux[2]));
  EXPECT_TRUE(active(rank_two_flux[0]));
  EXPECT_TRUE(active(rank_two_flux[1]));
  EXPECT_FALSE(active(rank_two_flux[2]));
  EXPECT_EQ(rank_two_flux[0].fL[0], Real(20));
  EXPECT_EQ(rank_two_flux[1].fL[0], Real(21));
  EXPECT_EQ(rank_zero_flux[2].fL[0], Real(22));

  const auto& rank_two_coarse = target_states[2].ring_flux.at("rhs")[0][0].coarse;
  EXPECT_EQ(rank_two_coarse[0].cL[0], Real(10));
  EXPECT_EQ(rank_two_coarse[1].cL[0], Real(11));
  const auto& rank_zero_contribution =
      target_states[0].ring_flux_contributions.at("rhs")[0][1][0].payload.fine;
  EXPECT_EQ(rank_zero_contribution[2].fL[0], Real(32));
  const auto& rank_two_contribution =
      target_states[2].ring_flux_contributions.at("rhs")[0][1][0].payload.fine;
  EXPECT_EQ(rank_two_contribution[0].fL[0], Real(30));
  EXPECT_EQ(rank_two_contribution[1].fL[0], Real(31));

  for (const auto& state : target_states) {
    const auto bytes = runtime::program::serialize_amr_program_accepted_state(state);
    EXPECT_EQ(runtime::program::serialize_amr_program_accepted_state(
                  runtime::program::deserialize_amr_program_accepted_state(bytes)),
              bytes);
  }
  const std::vector<std::vector<std::uint8_t>> source_payloads = {
      runtime::program::serialize_amr_program_accepted_state(source_states[0]),
      runtime::program::serialize_amr_program_accepted_state(source_states[1])};
  EXPECT_EQ(runtime::program::rematerialize_amr_program_accepted_state_bytes(
                source_payloads, source_ownership, target_ownership, 2),
            runtime::program::serialize_amr_program_accepted_state(target_states[2]));
}

TEST(test_program_reflux_ledger,
     accepted_checkpoint_state_rematerialization_refuses_metadata_disagreement) {
  const runtime::program::AmrProgramRankOwnership ownership{2, {{0, 1}, {0, 1, 0}}};
  std::vector<runtime::program::AmrProgramAcceptedState> states;
  states.push_back(ranked_accepted_state(0, ownership));
  states.push_back(ranked_accepted_state(1, ownership));
  states[1].ring_flux_contributions["rhs"][0][1][0].duration = 0.075;

  EXPECT_THROW(
      runtime::program::rematerialize_amr_program_accepted_states(states, ownership, ownership),
      std::runtime_error);

  states.clear();
  states.push_back(ranked_accepted_state(0, ownership));
  states.push_back(ranked_accepted_state(1, ownership));
  states[1].tagging_hysteresis_state.push_back(6);
  EXPECT_THROW(
      runtime::program::rematerialize_amr_program_accepted_states(states, ownership, ownership),
      std::runtime_error)
      << "rank-independent tagging state must reach exact checkpoint consensus";

  states.clear();
  states.push_back(ranked_accepted_state(0, ownership));
  states.push_back(ranked_accepted_state(1, ownership));
  states[1].accepted_interface_flux_ledger[0].measure.face_measure = 0.25;
  EXPECT_THROW(
      runtime::program::rematerialize_amr_program_accepted_states(states, ownership, ownership),
      std::runtime_error)
      << "rank-independent interface audit state must reach exact checkpoint consensus";
}

TEST(test_program_reflux_ledger,
     accepted_checkpoint_state_rematerialization_refuses_duplicate_nonowner_or_axis_conflict) {
  const runtime::program::AmrProgramRankOwnership ownership{2, {{0, 1}, {0, 1, 0}}};
  std::vector<runtime::program::AmrProgramAcceptedState> states;
  states.push_back(ranked_accepted_state(0, ownership));
  states.push_back(ranked_accepted_state(1, ownership));
  states[1].ring_flux["rhs"][0][1].fine[0] = ranked_strip(0, Real(99));
  EXPECT_THROW(
      runtime::program::rematerialize_amr_program_accepted_states(states, ownership, ownership),
      std::runtime_error);

  states[0].ring_flux["rhs"][0][1].fine[0] = EdgeStrip{};
  EXPECT_THROW(
      runtime::program::rematerialize_amr_program_accepted_states(states, ownership, ownership),
      std::runtime_error);

  states.clear();
  states.push_back(ranked_accepted_state(0, ownership));
  states.push_back(ranked_accepted_state(1, ownership));
  states[1].ring_flux["rhs"][0][1].fine.clear();
  EXPECT_THROW(
      runtime::program::rematerialize_amr_program_accepted_states(states, ownership, ownership),
      std::runtime_error);
}
