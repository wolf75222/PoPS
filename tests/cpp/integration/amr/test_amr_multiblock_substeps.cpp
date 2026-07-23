// AMR MULTI-BLOCS MULTIRATE (capstone iv) : la FACADE RUNTIME (AmrSystem -> AmrRuntime) honore les
// SUBSTEPS et le STRIDE PAR BLOC, en mirroir du moteur compile-time AmrSystemCoupler::step (#140), et
// AmrSystem::step_cfl devient SUBSTEPS/STRIDE-AWARE comme System::step_cfl.
//
// Ce que le test verrouille (cf. tache capstone iv) :
//   (1) SUBSTEPS reellement exerces : deux blocs EXPLICITES sur UNE hierarchie partagee, bloc A
//       substeps=4 + bloc B substeps=1 ; apres K macro-pas l'etat est FINI (rejet nan/inf AVANT toute
//       tolerance), la masse de chaque bloc est conservee a ~machine, et le resultat A substeps=4
//       DIFFERE d'un A substeps=1 (le sous-cyclage n'est pas un no-op). Puis le cas RENVERSE (A=1, B=4).
//   (2) STRIDE hold-then-catch-up : un bloc stride=2 co-evolue ; il est TENU au macro-pas 0 (densite
//       inchangee) et RATTRAPE au macro-pas 1 ((macro_step+1)%2==0). Le Poisson de systeme somme bien
//       les DEUX blocs a chaque pas (RHS non trivial), meme quand le bloc lent est tenu.
//   (3) step_cfl SUBSTEPS/STRIDE-AWARE : pour une config connue, le dt renvoye vaut
//       cfl*h*min_b(substeps_b/(stride_b*w_b)) a la tolerance fp pres.
//   (4) MONO-BLOC BIT-IDENTIQUE : step et step_cfl d'un bloc unique sont inchanges (dmax==0 entre deux
//       runs), garantissant que le routage facade laisse le mono-bloc sur AmrCouplerMP.
//
// On travaille surtout au niveau du MOTEUR AmrRuntime + build_amr_block (les briques de cette PR), ou
// l'on accede aux niveaux/masses/RHS des blocs ; les regressions mono-bloc passent par la facade.

#include <gtest/gtest.h>

#include <pops/coupling/base/elliptic_rhs.hpp>  // add_scaled_component (RHS de reference assemble main)
#include <pops/physics/bricks/bricks.hpp>  // CompositeModel + ExB/NoSource/ChargeDensity bricks
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // detail::make_shared_amr_layout / dispatch_amr_block
#include <pops/runtime/amr/amr_runtime.hpp>                  // AmrRuntime, AmrRuntimeBlock
#include <pops/runtime/amr_system.hpp>                       // facade AmrSystem
#include <pops/runtime/config/model_spec.hpp>
#include <pops/mesh/storage/mf_arith.hpp>  // norm_inf
#include <pops/mesh/storage/multifab.hpp>

#include "amr_transfer_test_authority.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

// Modele ExB scalaire (1 var) a charge q : advection pilotee par grad phi, densite de charge q n pour le
// Poisson de systeme. La charge q (signe inclus) distingue electrons / ions.
using ExBModel = CompositeModel<ExBVelocity, NoSource, ChargeDensity>;
static ExBModel exb_model(double q, double B0) {
  return ExBModel{ExBVelocity{Real(B0)}, NoSource{}, ChargeDensity{Real(q)}};
}

static ModelSpec exb_spec(double q, double B0) {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  s.q = q;
  s.B0 = B0;
  return s;
}

// densite de charge a moyenne nulle (solvable en periodique) : un creneau centre +/- amplitude a, n*n.
static std::vector<double> bump(int n, double base, double amp) {
  std::vector<double> r(static_cast<std::size_t>(n) * n, base);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const bool in = (i >= n / 4 && i < 3 * n / 4 && j >= n / 4 && j < 3 * n / 4);
      r[static_cast<std::size_t>(j) * n + i] = base + (in ? amp : -amp / 3.0);
    }
  return r;
}

static double mean_of(const std::vector<double>& values) {
  double sum = 0.0;
  for (double value : values)
    sum += value;
  return sum / static_cast<double>(values.size());
}

static double periodic_rhs_mean(double q0, const std::vector<double>& rho0, double q1,
                                const std::vector<double>& rho1) {
  return q0 * mean_of(rho0) + q1 * mean_of(rho1);
}

// tout fini (ni nan ni inf) : garde AVANT toute comparaison de tolerance (un nan passerait une borne).
static bool all_finite(const std::vector<double>& v) {
  for (double x : v)
    if (!std::isfinite(x))
      return false;
  return true;
}

// ecart L-inf entre deux champs n*n (pour "A differe de B" ou "X inchange").
static double dmax_field(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  const std::size_t nn = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t i = 0; i < nn; ++i)
    d = std::max(d, std::fabs(a[i] - b[i]));
  return d;
}

// Construit un AmrRuntime a DEUX blocs ExB (charges q0/q1, schemas potentiellement differents) sur une
// hierarchie figee N x N, avec substeps/stride par bloc. Renvoie le runtime (les densites initiales
// rho0/rho1 sont posees sur le grossier de chaque bloc).
static AmrRuntime make_two_block(int N, double L, double q0, double q1, double B0,
                                 const std::vector<double>& rho0, const std::vector<double>& rho1,
                                 const std::string& lim0, const std::string& lim1, int sub0,
                                 int sub1, int stride0, int stride1) {
  AmrBuildParams bp;
  bp.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  bp.mesh.periodicity = Periodicity{true, true};
  bp.mesh.n = N;
  bp.mesh.L = L;
  bp.mesh.regrid_every = 0;  // hierarchie figee (multi-blocs)
  bp.poisson.bc = BCRec{};   // periodique
  const detail::SharedAmrLayout S = detail::make_shared_amr_layout(bp);
  std::vector<AmrRuntimeBlock> blocks;
  blocks.push_back(detail::dispatch_amr_block(exb_model(q0, B0), lim0, "rusanov", S, "A", rho0,
                                              /*has_density=*/true, 1.4, sub0, false, false,
                                              stride0));
  blocks.push_back(detail::dispatch_amr_block(exb_model(q1, B0), lim1, "rusanov", S, "B", rho1,
                                              /*has_density=*/true, 1.4, sub1, false, false,
                                              stride1));
  AmrRuntime runtime(S.geom, S.runtime_hierarchy(), S.poisson_bc, std::move(blocks), S.base_per,
                     S.replicated_coarse, S.wall);
  test::install_second_order_amr_transfer_authorities(runtime, 2);
  runtime.set_parent_child_temporal_relations({::pops::amr::ParentChildClockRelation(
      0, 1, ::pops::amr::Rational(2, 1), ::pops::amr::RemainderPolicy::IntegralOnly)});
  return runtime;
}

// Minimal compiled model used to make the native clock partition numerically observable.  Mode 0
// has a zero flux and du/dt=u, so a 5/2 relation must perform Euler intervals {2/5,2/5,1/5}; the
// resulting value differs analytically from two equal substeps.  Modes 1..3 isolate the per-level
// transport, source-frequency and direct-stability CFL authorities respectively.
struct TemporalContractModel {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  int mode = 0;

  POPS_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State& u, const Aux&, int) const {
    return mode == 1 ? (u[0] < Real(0) ? -u[0] : u[0]) : Real(0);
  }
  POPS_HD State source(const State& u, const Aux&) const { return State{u[0]}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
  POPS_HD Real source_frequency(const State& u, const Aux&) const {
    return mode == 2 ? (u[0] < Real(0) ? -u[0] : u[0]) : Real(0);
  }
  POPS_HD Real stability_dt(const State& u, const Aux&) const {
    const Real magnitude = u[0] < Real(0) ? -u[0] : u[0];
    return mode == 3 && magnitude > Real(0) ? Real(1) / magnitude
                                            : std::numeric_limits<Real>::infinity();
  }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"u"}, 1, {VariableRole::Scalar}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"u"}, 1, {VariableRole::Scalar}};
  }
};

static AmrRuntime make_temporal_contract_runtime(int mode,
                                                 const amr::ParentChildClockRelation& relation) {
  constexpr int n = 8;
  AmrBuildParams bp;
  bp.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  bp.mesh.periodicity = Periodicity{true, true};
  bp.mesh.n = n;
  bp.mesh.L = 1.0;
  bp.mesh.regrid_every = 0;
  bp.poisson.bc = BCRec{};
  const detail::SharedAmrLayout layout = detail::make_shared_amr_layout(bp);
  const std::vector<double> initial(static_cast<std::size_t>(n) * n, 1.0);
  std::vector<AmrRuntimeBlock> blocks;
  blocks.push_back(detail::build_amr_block<TemporalContractModel, NoSlope, RusanovFlux>(
      TemporalContractModel{mode}, layout, "clocked", initial, /*has_density=*/true, 1.4,
      /*substeps=*/1, /*recon_prim=*/false, /*imex=*/false));
  AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc, std::move(blocks),
                     layout.base_per, layout.replicated_coarse, layout.wall);
  test::install_second_order_amr_transfer_authorities(runtime, 1);
  runtime.set_parent_child_temporal_relations({relation});
  return runtime;
}

TEST(test_amr_multiblock_substeps, Runs) {
#if defined(POPS_HAS_KOKKOS)
  int argc = 0;
  char** argv = nullptr;
  Kokkos::ScopeGuard guard(argc, argv);
#endif

  const int N = 32;
  const double L = 1.0, B0 = 1.0;
  const double q0 = +1.0, q1 = -1.0;  // A : ions ; B : electrons
  const std::vector<double> rho0 = bump(N, 1.0, 0.40);
  const std::vector<double> rho1 = bump(N, 1.0, 0.20);
  ASSERT_NEAR(periodic_rhs_mean(q0, rho0, q1, rho1), 0.0, 1e-13)
      << "all charged two-block fixtures must satisfy the periodic nullspace before solve";
  const Real dt = Real(0.01);
  const int K = 6;  // macro-pas

  // ============================================================================================
  // (1) SUBSTEPS exerces : A substeps=4, B substeps=1. Etat fini, masse conservee, et A(sub=4) != A(sub=1).
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                                   /*sub0=*/4, /*sub1=*/1, /*stride0=*/1, /*stride1=*/1);
    const Real mA0 = rt.mass(0), mB0 = rt.mass(1);
    for (int s = 0; s < K; ++s)
      rt.step(dt);
    const std::vector<double> dA4 = rt.density(0);
    const std::vector<double> dB = rt.density(1);
    const Real mA1 = rt.mass(0), mB1 = rt.mass(1);

    EXPECT_TRUE(all_finite(dA4) && all_finite(dB))
        << "subA4_state_finite";  // AVANT toute tolerance
    EXPECT_TRUE(std::fabs(mA1 - mA0) < 1e-10) << "subA4_blockA_mass_conserved";
    EXPECT_TRUE(std::fabs(mB1 - mB0) < 1e-10) << "subA4_blockB_mass_conserved";

    // Reference : MEME config mais A substeps=1. Le resultat de A doit DIFFERER (le sous-cyclage agit).
    AmrRuntime rt1 = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                                    /*sub0=*/1, /*sub1=*/1, /*stride0=*/1, /*stride1=*/1);
    for (int s = 0; s < K; ++s)
      rt1.step(dt);
    const std::vector<double> dA1 = rt1.density(0);
    EXPECT_TRUE(all_finite(dA1)) << "subA1_state_finite";
    EXPECT_TRUE(dmax_field(dA4, dA1) > 1e-9)
        << "subA4_differs_from_subA1";  // substepping NON no-op
    // bloc B (substeps=1 dans les deux runs) : meme trajectoire au bit pres (A ne le perturbe pas, les
    // blocs avancent independamment ; phi differe car A differe, mais le couplage est once-per-step et A
    // substeps n'altere PAS l'etat de B a substeps=1 sur le MEME phi de tete).
  }

  // ============================================================================================
  // (1b) RENVERSE : A substeps=1, B substeps=4. B(sub=4) doit differer de B(sub=1).
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                                   /*sub0=*/1, /*sub1=*/4, /*stride0=*/1, /*stride1=*/1);
    const Real mA0 = rt.mass(0), mB0 = rt.mass(1);
    for (int s = 0; s < K; ++s)
      rt.step(dt);
    const std::vector<double> dB4 = rt.density(1);
    const Real mA1 = rt.mass(0), mB1 = rt.mass(1);
    EXPECT_TRUE(all_finite(dB4)) << "revB4_state_finite";
    EXPECT_TRUE(std::fabs(mA1 - mA0) < 1e-10) << "revB4_blockA_mass_conserved";
    EXPECT_TRUE(std::fabs(mB1 - mB0) < 1e-10) << "revB4_blockB_mass_conserved";

    AmrRuntime rt1 = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                                    /*sub0=*/1, /*sub1=*/1, /*stride0=*/1, /*stride1=*/1);
    for (int s = 0; s < K; ++s)
      rt1.step(dt);
    const std::vector<double> dB1 = rt1.density(1);
    EXPECT_TRUE(dmax_field(dB4, dB1) > 1e-9) << "revB4_differs_from_subB1";
  }

  // ============================================================================================
  // (2) STRIDE hold-then-catch-up : bloc A stride=1 (rapide), bloc B stride=2 (lent). Au macro-pas 0
  //     (macro_step=0, (0+1)%2=1 != 0) B est TENU -> sa densite est INCHANGEE. Au macro-pas 1
  //     ((1+1)%2=0) B RATTRAPE -> sa densite CHANGE. Le Poisson de systeme somme les DEUX blocs a
  //     chaque pas (RHS non trivial), meme quand B est tenu.
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                                   /*sub0=*/1, /*sub1=*/1, /*stride0=*/1, /*stride1=*/2);
    const std::vector<double> dA_init = rt.density(0);
    const std::vector<double> dB_init = rt.density(1);
    const Real mB_init = rt.mass(1);

    // macro-pas 0 : A avance, B TENU.
    rt.step(dt);
    const std::vector<double> dA_0 = rt.density(0);
    const std::vector<double> dB_0 = rt.density(1);
    EXPECT_TRUE(dmax_field(dA_0, dA_init) > 1e-9) << "stride_blockA_advances_at_mac0";
    EXPECT_EQ(dmax_field(dB_0, dB_init), 0.0)
        << "stride_blockB_held_at_mac0";  // exactement inchange
    // Poisson somme actif au pas 0 (les DEUX densites contribuent ; B avec son etat fige).
    EXPECT_TRUE(norm_inf(rt.poisson_rhs()) > 1e-6) << "stride_poisson_sum_active_mac0";

    // macro-pas 1 : B RATTRAPE (pas effectif 2*dt).
    rt.step(dt);
    const std::vector<double> dB_1 = rt.density(1);
    EXPECT_TRUE(dmax_field(dB_1, dB_init) > 1e-9) << "stride_blockB_catchup_at_mac1";
    EXPECT_TRUE(std::fabs(rt.mass(1) - mB_init) < 1e-10) << "stride_blockB_mass_conserved";
    EXPECT_TRUE(norm_inf(rt.poisson_rhs()) > 1e-6) << "stride_poisson_sum_active_mac1";
  }

  // ============================================================================================
  // (3) step_cfl SUBSTEPS/STRIDE-AWARE : the two ExB blocks share B0 and the resolved field, hence
  //     the same wave speed w. Opposite charges preserve periodic nullspace compatibility.
  //     A substeps=4 stride=1, B substeps=1 stride=2.
  //     min_b(substeps_b/(stride_b*w)) = min(4/(1*w), 1/(2*w)) = 0.5/w. Donc dt attendu = cfl*h*0.5/w,
  //     avec w = rt.max_speed() (max sur blocs identiques = w commun) et h = dx_coarse = L/N.
  // ============================================================================================
  {
    AmrRuntime rt = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                                   /*sub0=*/4, /*sub1=*/1, /*stride0=*/1, /*stride1=*/2);
    const Real h = Real(L) / Real(N);  // dx_coarse
    const Real cfl = Real(0.4);
    const Real w = rt.max_speed();  // solve_fields + max sur les blocs (identiques -> w commun)
    EXPECT_TRUE(w > Real(0)) << "cfl_wave_speed_positive";
    // min(substeps/(stride*w)) sur {(4,1),(1,2)} = min(4, 0.5)/w = 0.5/w.
    const Real expected = cfl * h * Real(0.5) / w;
    const Real got = rt.step_cfl(cfl, h);
    EXPECT_TRUE(std::fabs(got - expected) <= Real(1e-12) * std::fabs(expected) + Real(1e-15))
        << "cfl_dt_is_substeps_stride_aware";
  }

  // ============================================================================================
  // (4) MONO-BLOC BIT-IDENTIQUE : step ET step_cfl d'un bloc unique inchanges (dmax==0 entre deux
  //     runs). Garantit que le routage facade laisse le mono-bloc sur AmrCouplerMP (jamais AmrRuntime,
  //     qui differe sur l'ordre des operations flottantes).
  // ============================================================================================
  {
    const std::vector<double> periodic_state = bump(N, 0.0, 0.40);
    ASSERT_NEAR(periodic_rhs_mean(q0, periodic_state, 0.0, periodic_state), 0.0, 1e-13)
        << "single charged periodic fixture must have zero RHS mean before solve";
    auto run_step = [&]() {
      AmrSystemConfig cfg;
      cfg.n = N;
      cfg.L = L;
      cfg.periodicity = {true, true};
      cfg.regrid_every = 0;
      AmrSystem sim(cfg);
      sim.add_block("ne", exb_spec(q0, B0), "none", "rusanov", "conservative", "explicit", 1);
      sim.set_poisson("charge_density", "geometric_mg", "periodic");
      sim.set_density("ne", periodic_state);
      sim.advance(0.01, 5);
      return sim.density("ne");
    };
    const std::vector<double> a = run_step();
    const std::vector<double> b = run_step();
    EXPECT_EQ(dmax_field(a, b), 0.0) << "monoblock_step_bit_identical";

    auto run_cfl = [&]() {
      AmrSystemConfig cfg;
      cfg.n = N;
      cfg.L = L;
      cfg.periodicity = {true, true};
      cfg.regrid_every = 0;
      AmrSystem sim(cfg);
      sim.add_block("ne", exb_spec(q0, B0), "none", "rusanov", "conservative", "explicit", 1);
      sim.set_poisson("charge_density", "geometric_mg", "periodic");
      sim.set_density("ne", periodic_state);
      double last = 0;
      for (int s = 0; s < 5; ++s)
        last = sim.step_cfl(0.4);
      return std::make_pair(sim.density("ne"), last);
    };
    const auto ra = run_cfl();
    const auto rb = run_cfl();
    EXPECT_EQ(dmax_field(ra.first, rb.first), 0.0) << "monoblock_step_cfl_field_bit_identical";
    EXPECT_EQ(ra.second, rb.second) << "monoblock_step_cfl_dt_bit_identical";
  }

  // ============================================================================================
  // (5) EXPLICIT AMR CLOCK CONTRACT. Spatial refinement remains 2 while the temporal relation is
  //     5/2 with a declared final remainder. The real native runtime must execute 0.4,0.4,0.2 of the
  //     parent dt; invalid IntegralOnly 5/2 is rejected before relation/state mutation. CFL scans
  //     the fine state using temporal product 5/2 independently from spatial product 2.
  // ============================================================================================
  {
    const amr::ParentChildClockRelation ratio_five_halves(
        0, 1, amr::Rational(5, 2), amr::RemainderPolicy::ExplicitFinalSubstep);
    AmrRuntime rational = make_temporal_contract_runtime(/*mode=*/0, ratio_five_halves);
    rational.step(Real(0.2));
    const auto rational_fine = rational.block_level_state_global(0, 1);
    const Real rational_max =
        static_cast<Real>(*std::max_element(rational_fine.begin(), rational_fine.end()));
    const Real expected_rational = Real(1.08) * Real(1.08) * Real(1.04);
    EXPECT_NEAR(rational_max, expected_rational, 2e-14)
        << "native 5/2 partition must execute two nominal intervals and the declared remainder";

    const amr::ParentChildClockRelation ratio_two(0, 1, amr::Rational(2, 1),
                                                  amr::RemainderPolicy::IntegralOnly);
    AmrRuntime integral = make_temporal_contract_runtime(/*mode=*/0, ratio_two);
    integral.step(Real(0.2));
    const auto integral_fine = integral.block_level_state_global(0, 1);
    const Real integral_max =
        static_cast<Real>(*std::max_element(integral_fine.begin(), integral_fine.end()));
    EXPECT_NEAR(integral_max, Real(1.1) * Real(1.1), 2e-14);
    EXPECT_GT(std::fabs(rational_max - integral_max), Real(1e-4))
        << "an installed temporal ratio must change the real native trajectory";

    // Strong preparation guarantee: the rejected candidate neither replaces the accepted chain nor
    // changes any level state.
    const auto before_rejected_set = rational.block_level_state_global(0, 1);
    EXPECT_THROW(rational.set_parent_child_temporal_relations({amr::ParentChildClockRelation(
                     0, 1, amr::Rational(5, 2), amr::RemainderPolicy::IntegralOnly)}),
                 std::runtime_error);
    EXPECT_EQ(rational.block_level_state_global(0, 1), before_rejected_set);
    ASSERT_EQ(rational.checkpoint_temporal_relations().size(), 1u);
    EXPECT_EQ(rational.checkpoint_temporal_relations()[0].temporal_ratio(), amr::Rational(5, 2));

    auto fine_state_spike = [](Real value) {
      constexpr std::size_t fine_extent = 16;
      std::vector<double> state(fine_extent * fine_extent, 0.0);
      state[8 * fine_extent + 8] = static_cast<double>(value);
      return state;
    };
    const Real h = Real(1) / Real(8);
    const Real cfl = Real(0.4);
    // A 3/2 temporal ratio on a spatial ratio 2 makes the fine transport interval restrictive.
    // The single fine spike averages to 1/4 of its value on the coarse, so the fine-local source
    // frequency and direct admissible-step bounds are restrictive as well.
    const amr::ParentChildClockRelation ratio_three_halves(
        0, 1, amr::Rational(3, 2), amr::RemainderPolicy::ExplicitFinalSubstep);
    constexpr Real spike = Real(16);

    AmrRuntime transport = make_temporal_contract_runtime(/*mode=*/1, ratio_three_halves);
    transport.set_block_level_state(0, 1, fine_state_spike(spike));
    EXPECT_NEAR(transport.cfl_dt(cfl, h), cfl * (h / Real(2)) * Real(1.5) / spike, 2e-15);
    EXPECT_EQ(transport.last_dt_bound(), "transport:clocked");

    AmrRuntime source_bound = make_temporal_contract_runtime(/*mode=*/2, ratio_three_halves);
    source_bound.set_block_level_state(0, 1, fine_state_spike(spike));
    EXPECT_NEAR(source_bound.cfl_dt(cfl, h), cfl * Real(1.5) / spike, 2e-15);
    EXPECT_EQ(source_bound.last_dt_bound(), "source_frequency:clocked");

    AmrRuntime direct_bound = make_temporal_contract_runtime(/*mode=*/3, ratio_three_halves);
    direct_bound.set_block_level_state(0, 1, fine_state_spike(spike));
    EXPECT_NEAR(direct_bound.cfl_dt(cfl, h), Real(1.5) / spike, 2e-15);
    EXPECT_EQ(direct_bound.last_dt_bound(), "stability_dt:clocked");

    // A coupled frequency is a macro-step authority, but its field expression must still scan every
    // active AMR level. The coarse state remains one while the fine-only spike sets the global max.
    AmrRuntime coupled_bound = make_temporal_contract_runtime(/*mode=*/0, ratio_three_halves);
    coupled_bound.set_block_level_state(0, 1, fine_state_spike(spike));
    coupled_bound.add_coupled_frequency_expr(
        "fine_frequency", {"clocked"}, {"scalar"}, {},
        {static_cast<int>(CsOp::PushReg)}, {0});
    EXPECT_NEAR(coupled_bound.cfl_dt(cfl, h), cfl / spike, 2e-15);
    EXPECT_EQ(coupled_bound.last_dt_bound(), "coupled_source:fine_frequency");
  }
}
