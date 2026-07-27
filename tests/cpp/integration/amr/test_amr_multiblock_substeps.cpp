// AMR MULTI-BLOCS MULTIRATE (capstone iv / ADC-700): un Program explicite porte les SUBSTEPS et le
// STRIDE PAR BLOC; AmrRuntime reste le moteur spatial et le service de bornes CFL.
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
//   (3bis) CADENCE DE CHAMP PROGRAMMEE : placer solve_fields une fois hors d'une boucle de quatre
//       etages produit exactement 1 solve, le placer dans la boucle en produit exactement 4.
//   (4) MONO-BLOC DETERMINISTE : deux Programs identiques produisent les memes steps et step_cfl.
//
// Toutes les avances passent par AmrSystem + Program. Le test n'appelle AmrRuntime que pour inspecter
// les niveaux, masses, RHS et bornes spatiales.

#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/mesh/storage/mf_arith.hpp>  // norm_inf
#include <pops/mesh/storage/multifab.hpp>

#include "amr_transfer_test_authority.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

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

// Test-only authored Program with per-block cadence. The old AmrRuntime::step owned this loop
// implicitly; ADC-700 makes it explicit: each block's stride decides whether it is due, and its
// substeps determine the Euler intervals evaluated by the Program over every hierarchy clock.
static void install_multirate_forward_euler_program(AmrSystem& system, std::vector<int> substeps,
                                                    std::vector<int> strides) {
  if (substeps.size() != strides.size() ||
      substeps.size() != static_cast<std::size_t>(system.n_blocks()))
    throw std::invalid_argument("multirate test Program requires one cadence per block");
  for (std::size_t block = 0; block < substeps.size(); ++block)
    if (substeps[block] < 1 || strides[block] < 1)
      throw std::invalid_argument("multirate test Program cadences must be positive");

  std::vector<int> block_map(substeps.size());
  std::iota(block_map.begin(), block_map.end(), 0);
  system.set_program_block_map(block_map);
  system.install_program_step([](double) {});
  if (!system.uses_runtime_engine() || system.engine() == nullptr)
    throw std::runtime_error("multirate test Program requires the materialized AMR engine");

  auto context = std::make_shared<runtime::program::AmrProgramContext>(system.engine(), &system);
  context->configure_primary_clock("test.clock.macro");
  context->install(
      [context, substeps = std::move(substeps), strides = std::move(strides)](double macro_dt) {
        context->advance_hierarchy(macro_dt, [context, &substeps, &strides](double level_dt) {
          (void)consume_solve_outcome(context->solve_fields());
          for (int block = 0; block < context->n_blocks(); ++block) {
            const auto index = static_cast<std::size_t>(block);
            if ((context->macro_step() + 1) % strides[index] != 0)
              continue;
            const double effective_dt = level_dt * static_cast<double>(strides[index]);
            const double substep_dt = effective_dt / static_cast<double>(substeps[index]);
            for (int substep = 0; substep < substeps[index]; ++substep) {
              context->set_stage_time(substep, substeps[index]);
              MultiFab& state = context->state(block);
              MultiFab& residual = context->rhs_scratch(1000 + block, 0, state);
              context->rhs_into(block, state, residual, 3000 + block);
              context->axpy(state, Real(substep_dt), residual, Real(substep_dt), {{1, 1, 1}});
            }
          }
        });
      });
  system.set_program_block_map(block_map);
}

// Construit une facade AmrSystem a DEUX blocs ExB sur une hierarchie partagee et installe le Program
// multirate explicite ci-dessus. AmrRuntime reste uniquement le moteur spatial inspecte par le test.
static std::unique_ptr<AmrSystem> make_two_block(int N, double L, double q0, double q1, double B0,
                                                 const std::vector<double>& rho0,
                                                 const std::vector<double>& rho1,
                                                 const std::string& lim0, const std::string& lim1,
                                                 int sub0, int sub1, int stride0, int stride1) {
  AmrSystemConfig cfg;
  cfg.n = N;
  cfg.L = L;
  cfg.periodicity = {true, true};
  cfg.regrid_every = 0;
  auto system = std::make_unique<AmrSystem>(cfg);
  system->add_block("A", exb_spec(q0, B0), lim0, "rusanov", "conservative", "euler",
                    /*substeps=*/sub0, /*stride=*/stride0);
  system->add_block("B", exb_spec(q1, B0), lim1, "rusanov", "conservative", "euler",
                    /*substeps=*/sub1, /*stride=*/stride1);
  system->set_poisson("charge_density", "geometric_mg", "periodic");
  system->set_density("A", rho0);
  system->set_density("B", rho1);
  system->set_temporal_relations({2}, {1}, {"integral_only"});
  install_multirate_forward_euler_program(*system, {sub0, sub1}, {stride0, stride1});
  return system;
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

static std::unique_ptr<AmrSystem> make_temporal_contract_system(
    int mode, const amr::ParentChildClockRelation& relation) {
  constexpr int n = 8;
  const std::vector<double> initial(static_cast<std::size_t>(n) * n, 1.0);
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.level_count = 2;
  cfg.regrid_every = 0;
  cfg.periodicity = {true, true};
  auto system = std::make_unique<AmrSystem>(cfg);
  AmrCompiledBlockBuilder builder =
      [mode](const detail::SharedAmrLayout& layout, const std::string& name,
             const std::vector<double>& density, bool has_density, const std::vector<double>& state,
             bool has_state, double gamma, int substeps, bool recon_prim, bool imex, int stride,
             const std::vector<std::string>& implicit_vars,
             const std::vector<std::string>& implicit_roles, double pos_floor, double weno_epsilon,
             bool wave_speed_cache) {
        if (imex || !implicit_vars.empty() || !implicit_roles.empty())
          throw std::invalid_argument("temporal-contract test block is explicit");
        return detail::build_amr_block<TemporalContractModel, NoSlope, RusanovFlux>(
            TemporalContractModel{mode}, layout, name, density, has_density, gamma, substeps,
            recon_prim, stride, has_state ? &state : nullptr, pos_floor, weno_epsilon,
            wave_speed_cache);
      };
  system->set_compiled_block(TemporalContractModel::n_vars, 1.4, /*substeps=*/1, std::move(builder),
                             "clocked");
  system->set_density("clocked", initial);
  system->set_poisson("charge_density", "geometric_mg", "periodic");
  // A configured (but never re-evaluated) criterion selects the two-level hierarchy template.
  system->set_refinement(1e29);
  const amr::Rational ratio = relation.temporal_ratio();
  system->set_temporal_relations({ratio.numerator}, {ratio.denominator},
                                 {relation.remainder_policy() == amr::RemainderPolicy::IntegralOnly
                                      ? "integral_only"
                                      : "explicit_final_substep"});
  test::install_forward_euler_program(*system);
  return system;
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
  // (3bis) AuthoredFieldSolveCadenceCountsOneVsFour.
  //     Les deux runs ont la meme hierarchie AMR a deux niveaux et le meme probleme de Poisson.
  //     Seule la position de solve_fields dans le Program differe : une fois par macro-pas, ou dans
  //     une boucle explicite de quatre etages. Les appels du corps au niveau fin sont des cache hits ;
  //     le compteur du runtime mesure donc exactement l'intention ecrite : 1 contre 4 vrais solves.
  // ============================================================================================
  {
    auto make_field_cadence_system = [&](bool per_stage) {
      AmrSystemConfig cfg;
      cfg.n = N;
      cfg.L = L;
      cfg.level_count = 2;
      cfg.regrid_every = 0;
      cfg.periodicity = {true, true};
      auto system = std::make_unique<AmrSystem>(cfg);
      system->add_block("A", exb_spec(q0, B0), "minmod", "rusanov", "conservative", "euler", 1);
      system->add_block("B", exb_spec(q1, B0), "minmod", "rusanov", "conservative", "euler", 1);
      system->set_poisson("charge_density", "geometric_mg", "periodic");
      system->set_density("A", rho0);
      system->set_density("B", rho1);
      system->set_refinement(1e29);
      system->set_temporal_relations({2}, {1}, {"integral_only"});
      system->set_program_block_map({0, 1});
      system->install_program_step([](double) {});
      if (!system->uses_runtime_engine() || system->engine() == nullptr)
        throw std::runtime_error("field-cadence test requires the materialized AMR engine");

      auto context =
          std::make_shared<runtime::program::AmrProgramContext>(system->engine(), system.get());
      context->configure_primary_clock("test.clock.field-cadence");
      context->install([context, per_stage](double macro_dt) {
        context->advance_hierarchy(macro_dt, [context, per_stage](double) {
          if (!per_stage) {
            context->set_stage_time(0, 1);
            (void)consume_solve_outcome(context->solve_fields());
            return;
          }
          for (int stage = 0; stage < 4; ++stage) {
            context->set_stage_time(stage, 4);
            (void)consume_solve_outcome(context->solve_fields());
          }
        });
      });
      system->set_program_block_map({0, 1});
      return system;
    };

    auto once = make_field_cadence_system(false);
    AmrRuntime& once_runtime = *once->engine();
    ASSERT_EQ(once_runtime.nlev(), 2);
    const int once_before = once_runtime.solve_count();
    once->step(dt);
    EXPECT_EQ(once_runtime.solve_count() - once_before, 1)
        << "solve_fields authored outside the stage loop must solve exactly once";

    auto per_stage = make_field_cadence_system(true);
    AmrRuntime& per_stage_runtime = *per_stage->engine();
    ASSERT_EQ(per_stage_runtime.nlev(), 2);
    const int per_stage_before = per_stage_runtime.solve_count();
    per_stage->step(dt);
    EXPECT_EQ(per_stage_runtime.solve_count() - per_stage_before, 4)
        << "solve_fields authored inside four stages must solve exactly four times";
  }

  // ============================================================================================
  // (1) SUBSTEPS exerces : A substeps=4, B substeps=1. Etat fini, masse conservee, et A(sub=4) != A(sub=1).
  // ============================================================================================
  {
    auto sim = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                              /*sub0=*/4, /*sub1=*/1, /*stride0=*/1, /*stride1=*/1);
    AmrRuntime& rt = *sim->engine();
    const Real mA0 = rt.mass(0), mB0 = rt.mass(1);
    for (int s = 0; s < K; ++s)
      sim->step(dt);
    const std::vector<double> dA4 = rt.density(0);
    const std::vector<double> dB = rt.density(1);
    const Real mA1 = rt.mass(0), mB1 = rt.mass(1);

    EXPECT_TRUE(all_finite(dA4) && all_finite(dB))
        << "subA4_state_finite";  // AVANT toute tolerance
    EXPECT_TRUE(std::fabs(mA1 - mA0) < 1e-10) << "subA4_blockA_mass_conserved";
    EXPECT_TRUE(std::fabs(mB1 - mB0) < 1e-10) << "subA4_blockB_mass_conserved";

    // Reference : MEME config mais A substeps=1. Le resultat de A doit DIFFERER (le sous-cyclage agit).
    auto sim1 = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                               /*sub0=*/1, /*sub1=*/1, /*stride0=*/1, /*stride1=*/1);
    AmrRuntime& rt1 = *sim1->engine();
    for (int s = 0; s < K; ++s)
      sim1->step(dt);
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
    auto sim = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                              /*sub0=*/1, /*sub1=*/4, /*stride0=*/1, /*stride1=*/1);
    AmrRuntime& rt = *sim->engine();
    const Real mA0 = rt.mass(0), mB0 = rt.mass(1);
    for (int s = 0; s < K; ++s)
      sim->step(dt);
    const std::vector<double> dB4 = rt.density(1);
    const Real mA1 = rt.mass(0), mB1 = rt.mass(1);
    EXPECT_TRUE(all_finite(dB4)) << "revB4_state_finite";
    EXPECT_TRUE(std::fabs(mA1 - mA0) < 1e-10) << "revB4_blockA_mass_conserved";
    EXPECT_TRUE(std::fabs(mB1 - mB0) < 1e-10) << "revB4_blockB_mass_conserved";

    auto sim1 = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                               /*sub0=*/1, /*sub1=*/1, /*stride0=*/1, /*stride1=*/1);
    AmrRuntime& rt1 = *sim1->engine();
    for (int s = 0; s < K; ++s)
      sim1->step(dt);
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
    auto sim = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                              /*sub0=*/1, /*sub1=*/1, /*stride0=*/1, /*stride1=*/2);
    AmrRuntime& rt = *sim->engine();
    const std::vector<double> dA_init = rt.density(0);
    const std::vector<double> dB_init = rt.density(1);
    const Real mB_init = rt.mass(1);

    // macro-pas 0 : A avance, B TENU.
    sim->step(dt);
    const std::vector<double> dA_0 = rt.density(0);
    const std::vector<double> dB_0 = rt.density(1);
    EXPECT_TRUE(dmax_field(dA_0, dA_init) > 1e-9) << "stride_blockA_advances_at_mac0";
    EXPECT_EQ(dmax_field(dB_0, dB_init), 0.0)
        << "stride_blockB_held_at_mac0";  // exactement inchange
    // Poisson somme actif au pas 0 (les DEUX densites contribuent ; B avec son etat fige).
    EXPECT_TRUE(norm_inf(rt.poisson_rhs()) > 1e-6) << "stride_poisson_sum_active_mac0";

    // macro-pas 1 : B RATTRAPE (pas effectif 2*dt).
    sim->step(dt);
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
    auto sim = make_two_block(N, L, q0, q1, B0, rho0, rho1, "minmod", "minmod",
                              /*sub0=*/4, /*sub1=*/1, /*stride0=*/1, /*stride1=*/2);
    AmrRuntime& rt = *sim->engine();
    const Real h = Real(L) / Real(N);  // dx_coarse
    const Real cfl = Real(0.4);
    const Real w = rt.max_speed();  // solve_fields + max sur les blocs (identiques -> w commun)
    EXPECT_TRUE(w > Real(0)) << "cfl_wave_speed_positive";
    // min(substeps/(stride*w)) sur {(4,1),(1,2)} = min(4, 0.5)/w = 0.5/w.
    const Real expected = cfl * h * Real(0.5) / w;
    const Real got = static_cast<Real>(sim->step_cfl(cfl));
    EXPECT_TRUE(std::fabs(got - expected) <= Real(1e-12) * std::fabs(expected) + Real(1e-15))
        << "cfl_dt_is_substeps_stride_aware";
  }

  // ============================================================================================
  // (4) MONO-BLOC DETERMINISTE : deux executions du meme Program donnent exactement le meme step et
  //     step_cfl. Il n'existe plus de selection implicite d'un second moteur temporel mono-bloc.
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
      sim.set_temporal_relations({2}, {1}, {"integral_only"});
      test::install_forward_euler_program(sim);
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
      sim.set_temporal_relations({2}, {1}, {"integral_only"});
      test::install_forward_euler_program(sim);
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
    auto rational_system = make_temporal_contract_system(/*mode=*/0, ratio_five_halves);
    AmrRuntime& rational = *rational_system->engine();
    rational_system->step(Real(0.2));
    const auto rational_fine = rational.block_level_state_global(0, 1);
    const Real rational_max =
        static_cast<Real>(*std::max_element(rational_fine.begin(), rational_fine.end()));
    const Real expected_rational = Real(1.08) * Real(1.08) * Real(1.04);
    EXPECT_NEAR(rational_max, expected_rational, 2e-14)
        << "native 5/2 partition must execute two nominal intervals and the declared remainder";

    const amr::ParentChildClockRelation ratio_two(0, 1, amr::Rational(2, 1),
                                                  amr::RemainderPolicy::IntegralOnly);
    auto integral_system = make_temporal_contract_system(/*mode=*/0, ratio_two);
    AmrRuntime& integral = *integral_system->engine();
    integral_system->step(Real(0.2));
    const auto integral_fine = integral.block_level_state_global(0, 1);
    const Real integral_max =
        static_cast<Real>(*std::max_element(integral_fine.begin(), integral_fine.end()));
    EXPECT_NEAR(integral_max, Real(1.1) * Real(1.1), 2e-14);
    EXPECT_GT(std::fabs(rational_max - integral_max), Real(1e-4))
        << "an installed temporal ratio must change the real native trajectory";

    // Strong preparation guarantee: validating a rejected candidate neither replaces the accepted
    // chain nor changes any level state. Runtime installation is deliberately separate from clock
    // partition validation, so exercise the relation's fail-closed partition contract directly.
    const auto before_rejected_set = rational.block_level_state_global(0, 1);
    const amr::ParentChildClockRelation rejected_relation(0, 1, amr::Rational(5, 2),
                                                          amr::RemainderPolicy::IntegralOnly);
    const amr::ClockWindow parent_window{{0, 0, amr::Rational(0, 1), 0.0},
                                         {0, 0, amr::Rational(1, 1), 1.0}};
    EXPECT_THROW(rejected_relation.partition(parent_window), std::runtime_error);
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

    auto transport_system = make_temporal_contract_system(/*mode=*/1, ratio_three_halves);
    AmrRuntime& transport = *transport_system->engine();
    transport.set_block_level_state(0, 1, fine_state_spike(spike));
    EXPECT_NEAR(transport.cfl_dt(cfl, h), cfl * (h / Real(2)) * Real(1.5) / spike, 2e-15);
    EXPECT_EQ(transport.last_dt_bound(), "transport:clocked");

    auto source_bound_system = make_temporal_contract_system(/*mode=*/2, ratio_three_halves);
    AmrRuntime& source_bound = *source_bound_system->engine();
    source_bound.set_block_level_state(0, 1, fine_state_spike(spike));
    EXPECT_NEAR(source_bound.cfl_dt(cfl, h), cfl * Real(1.5) / spike, 2e-15);
    EXPECT_EQ(source_bound.last_dt_bound(), "source_frequency:clocked");

    auto direct_bound_system = make_temporal_contract_system(/*mode=*/3, ratio_three_halves);
    AmrRuntime& direct_bound = *direct_bound_system->engine();
    direct_bound.set_block_level_state(0, 1, fine_state_spike(spike));
    EXPECT_NEAR(direct_bound.cfl_dt(cfl, h), Real(1.5) / spike, 2e-15);
    EXPECT_EQ(direct_bound.last_dt_bound(), "stability_dt:clocked");

    // A coupled frequency is a macro-step authority, but its field expression must still scan every
    // active AMR level. The coarse state remains one while the fine-only spike sets the global max.
    auto coupled_bound_system = make_temporal_contract_system(/*mode=*/0, ratio_three_halves);
    AmrRuntime& coupled_bound = *coupled_bound_system->engine();
    coupled_bound.set_block_level_state(0, 1, fine_state_spike(spike));
    coupled_bound.add_coupled_frequency_expr("fine_frequency", {"clocked"}, {"scalar"}, {},
                                             {static_cast<int>(CsOp::PushReg)}, {0});
    EXPECT_NEAR(coupled_bound.cfl_dt(cfl, h), cfl / spike, 2e-15);
    EXPECT_EQ(coupled_bound.last_dt_bound(), "coupled_source:fine_frequency");
  }
}
