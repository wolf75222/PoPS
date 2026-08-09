// AMR MULTI-BLOCS COMPILES (capstone "v", DSL production multi-bloc) : la FACADE RUNTIME AmrSystem
// accepte desormais PLUSIEURS blocs COMPILES (add_compiled_model, modeles connus a la compilation)
// co-localises sur UNE hierarchie AMR PARTAGEE. Avant ce capstone, le 2e bloc compile LEVAIT
// ("set_compiled_block : un seul bloc compile est supporte"). Desormais add_compiled_model enregistre
// chaque bloc spatial concret et la facade les materialise tous dans l'unique AmrRuntime, sur le meme
// layout que les blocs natifs. Le Program installe reste l'unique autorite temporelle, quel que soit le
// nombre ou l'origine des blocs.
//
// Ce que le test verrouille :
//   (A) DEUX blocs compiles a SCHEMAS DIFFERENTS sur la hierarchie partagee : pas de crash au 2e bloc,
//       les DEUX blocs EVOLUENT (transport E x B), masse conservee PAR BLOC, potentiel de systeme
//       (Poisson somme co-localise q0 n0 + q1 n1) non trivial.
//   (B) COUPLAGE entre les deux blocs compiles : une source ionisation-like CONSERVATIVE (+S sur un
//       bloc, -S exactement sur l'autre, meme cellule) transfere de la masse entre blocs tout en
//       conservant la masse COMPOSITE globale a ~machine.
//   (C) MELANGE compile + natif : un bloc compile (add_compiled_model) et un bloc natif (add_block)
//       co-existent sur la meme hierarchie (le melange etait refuse avant ce capstone).
//   (D) MONO-BLOC COMPILE DETERMINISTE : le meme bloc spatial et le meme Program rejoues deux fois
//       donnent le MEME resultat au bit pres (dmax==0).
//   (E) NON-REGRESSION du refus : le 2e bloc compile NE LEVE PLUS (preuve directe que la file de specs
//       a remplace l'ancien throw "un seul bloc compile").
//
// CHOIX DE COMPILABILITE (limitation connue nvcc, cf. tache). Un test AMR complet avec concept + lambda
// GENERIQUE (auto m) NE COMPILE PAS sous nvcc. Ici on utilise donc des FONCTEURS / TYPES CONCRETS :
// les modeles sont des CompositeModel<ExBVelocity, NoSource, ChargeDensity> instancies a la main (pas
// de dispatch_model generique), et add_compiled_model capture ces types concrets. Le noyau AMR
// (residu spatial Limiter/Flux) reste capture par dispatch_amr_block via une fonction template NOMMEE
// (recette device-clean #64/#97), jamais une lambda etendue cross-TU. Le test compile donc partout
// (CPU + Kokkos Serial/OpenMP/Cuda), comme test_amr_coupled_source_role_strict et add_compiled_model #16/#18.

#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include <pops/coupling/source/coupled_source_program.hpp>  // CsOp (opcodes du bytecode P5)
#include <pops/physics/bricks/bricks.hpp>                   // CompositeModel
#include <pops/physics/bricks/elliptic.hpp>                 // ChargeDensity
#include <pops/physics/bricks/hyperbolic.hpp>               // ExBVelocity
#include <pops/physics/bricks/source.hpp>                   // NoSource
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // add_compiled_model(AmrSystem&, ...)
#include <pops/runtime/amr_system.hpp>                       // facade AmrSystem
#include <pops/runtime/config/model_spec.hpp>  // ModelSpec (bloc natif, melange compile + natif)
#include <pops/runtime/program/amr_program_context.hpp>

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

// Modele ExB scalaire (1 var) a charge q, CONNU A LA COMPILATION (type concret, pas de dispatch). q
// (signe inclus) distingue ions (+1) / electrons (-1) ; q=0 -> bloc neutre (pas de contribution Poisson).
using ExBModel = CompositeModel<ExBVelocity, NoSource, ChargeDensity>;
static ExBModel exb_model(double q, double B0) {
  return ExBModel{ExBVelocity{Real(B0)}, NoSource{}, ChargeDensity{Real(q)}};
}

// Spec ExB scalaire NATIVE (pour le melange compile + natif du point C) : meme physique, via ModelSpec.
static ModelSpec exb_spec(double q, double B0) {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  s.q = q;
  s.B0 = B0;
  return s;
}

// Source RAIDE CELLULE-LOCALE qui NE TOUCHE PAS la densite (composante 0) NI l'energie (composante 3) :
// relaxation de la SEULE quantite de mouvement (mx, my) vers zero, de raideur 1/eps (parente de
// StiffMomentumRelax de test_amr_multiblock_imex.cpp ; ici on ne stiffenne que mx/my pour que le MASQUE
// IMEX PARTIEL {momentum_x, momentum_y} couvre EXACTEMENT les composantes raides). En EXPLICITE (Euler
// avant) mx <- mx (1 - dt/eps) DIVERGE des que dt/eps > 2 ; en IMEX (backward Euler) mx <- mx /
// (1 + dt/eps) reste BORNE pour tout dt > 0. rho (comp 0) a source NULLE -> masse conservee a la machine.
struct StiffMomentumRelax {
  Real inv_eps = Real(0);
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr.stiff-momentum-relax", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(inv_eps);
  }
  template <class State>
  POPS_HD State apply(const State& u, const Aux&) const {
    State s{};
    if (State::size() > 1)
      s[1] = -inv_eps * u[1];  // -mx / eps
    if (State::size() > 2)
      s[2] = -inv_eps * u[2];  // -my / eps
    return s;
  }
};

// The canonical state-independent zero RHS cannot manufacture NaNs in an otherwise exact-zero
// periodic solve after the deliberately unstable explicit trajectory becomes non-finite.
using StiffCModel = CompositeModel<Euler, StiffMomentumRelax, NoElliptic>;
static StiffCModel stiff_cmodel(double eps) {
  StiffMomentumRelax r;
  r.inv_eps = static_cast<Real>(1.0 / eps);
  return StiffCModel{Euler{Real(1.4)}, r, NoElliptic{}};
}

// Modele COMPILE 4 variables NEUTRE (Euler sans source) : un bloc voisin explicite, MEME nombre de
// variables que le bloc raide (layout coherent sur la hierarchie partagee).
using NeutralCModel = CompositeModel<Euler, NoSource, NoElliptic>;
static NeutralCModel neutral_cmodel() {
  return NeutralCModel{Euler{Real(1.4)}, NoSource{}, NoElliptic{}};
}

// densite "bulle" gaussienne (gradients non triviaux -> le transport engendre de l'impulsion que la
// source raide relaxe). Sert au cas (F) sur les blocs compiles 4 variables.
static std::vector<double> bubble(int n) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      rho[static_cast<std::size_t>(j) * n + i] = 1.0 + 0.5 * std::exp(-(x * x + y * y) / 0.02);
    }
  return rho;
}

// Etat Euler complet, component-major, dont les QUATRE composantes sont partout non nulles. Le
// test (G) l'utilise comme oracle exact du transport set_conservative_state -> builder compile.
static std::vector<double> full_euler_state(int n, double scale) {
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  std::vector<double> state(4 * nn);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const std::size_t k = static_cast<std::size_t>(j) * n + i;
      const double rho = scale * (1.0 + 0.01 * static_cast<double>(1 + i + n * j));
      state[0 * nn + k] = rho;
      state[1 * nn + k] = 0.25 * rho;
      state[2 * nn + k] = -0.125 * rho;
      state[3 * nn + k] = 2.5 * rho;
    }
  return state;
}

// densite de charge a moyenne nulle (solvable en periodique) : un creneau centre +/- amplitude, n*n.
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

template <class F>
static bool raises(F&& f) {
  try {
    f();
  } catch (const std::runtime_error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

static double dmax_field(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  const std::size_t nn = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t i = 0; i < nn; ++i)
    d = std::max(d, std::fabs(a[i] - b[i]));
  return d;
}

static bool all_finite(const std::vector<double>& v) {
  for (double x : v)
    if (!std::isfinite(x))
      return false;
  return true;
}

// Enregistre une source ionisation-like CONSERVATIVE entre deux blocs nommes sur le role density :
// S = k * n_a * n_b ; gain +S sur @p block_gain, perte -S (gain + Neg) sur @p block_loss, MEME cellule.
// Bytecode postfixe construit a la main, EXACTEMENT comme pops.dsl.CoupledSource.add_pair (cf.
// test_amr_multiblock_coupled_source). Conserve n_a + n_b par cellule a ~machine.
static void register_ionization(AmrSystem& sim, const std::string& block_a,
                                const std::string& block_b, const std::string& block_gain,
                                const std::string& block_loss, double k) {
  const std::vector<std::string> in_blocks = {block_a, block_b};
  const std::vector<std::string> in_roles = {"density", "density"};
  const std::vector<double> consts = {k};
  const std::vector<std::string> out_blocks = {block_gain, block_loss};
  const std::vector<std::string> out_roles = {"density", "density"};
  const int P = static_cast<int>(CsOp::PushReg), MUL = static_cast<int>(CsOp::Mul),
            NEG = static_cast<int>(CsOp::Neg);
  // registres : r0 = n_a (entree), r1 = n_b (entree), r2 = k (constante)
  // gain  : PushReg 0, PushReg 1, Mul, PushReg 2, Mul        -> k * n_a * n_b
  // perte : <gain> puis Neg                                  -> -(k * n_a * n_b)
  std::vector<int> prog_ops = {P, P, MUL, P, MUL,        // gain
                               P, P, MUL, P, MUL, NEG};  // perte
  std::vector<int> prog_args = {0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 0};
  std::vector<int> prog_lens = {5, 6};
  // ADC-214 : description bytecode regroupee dans le POD CoupledSourceProgram (appel auto-documente).
  pops::CoupledSourceProgram prog;
  prog.in_blocks = in_blocks;
  prog.in_roles = in_roles;
  prog.consts = consts;
  prog.out_blocks = out_blocks;
  prog.out_roles = out_roles;
  prog.prog_ops = prog_ops;
  prog.prog_args = prog_args;
  prog.prog_lens = prog_lens;
  sim.add_coupled_source(prog);
}

// Program-owned split used by the compiled-block coupling proof: one explicit transport rate for
// each block, then one simultaneous coupled-source update on private candidates and one group commit.
static void install_compiled_coupling_program(AmrSystem& system) {
  system.install_program_step([](double) {});
  if (!system.uses_runtime_engine() || system.engine() == nullptr)
    throw std::runtime_error("compiled coupled-source fixture failed to materialize AmrRuntime");

  auto context = std::make_shared<runtime::program::AmrProgramContext>(system.engine(), &system);
  context->configure_primary_clock("test.compiled.coupling.macro");
  context->install([context](double macro_dt) {
    context->advance_hierarchy(macro_dt, [context](double level_dt) {
      context->set_stage_time(0, 1);
      if (context->level() == 0)
        (void)consume_solve_outcome(context->solve_default_field_on_coarse_level());

      MultiFab& ions = context->state(0);
      MultiFab& neutrals = context->state(1);
      MultiFab& ions_rate = context->rhs_scratch(1000, 0, ions);
      MultiFab& neutrals_rate = context->rhs_scratch(1001, 0, neutrals);
      context->rhs_into(0, ions, ions_rate, 2000);
      context->rhs_into(1, neutrals, neutrals_rate, 2001);
      context->axpy(ions, Real(level_dt), ions_rate, Real(level_dt), {{1, 1, 1}});
      context->axpy(neutrals, Real(level_dt), neutrals_rate, Real(level_dt), {{1, 1, 1}});

      MultiFab& ions_candidate = context->scratch_state(3000, 0, ions);
      MultiFab& neutrals_candidate = context->scratch_state(3001, 0, neutrals);
      context->lincomb(ions_candidate, Real(1), ions, Real(0), ions);
      context->lincomb(neutrals_candidate, Real(1), neutrals, Real(0), neutrals);
      context->apply_coupling_operators(Real(level_dt),
                                        {{0, &ions_candidate}, {1, &neutrals_candidate}});
      context->commit_many({{&ions, &ions_candidate}, {&neutrals, &neutrals_candidate}});
    });
  });
  system.set_program_block_map({0, 1});
}

TEST(test_amr_multiblock_compiled, Runs) {
#if defined(POPS_HAS_KOKKOS)
  int argc = 0;
  char** argv = nullptr;
  Kokkos::ScopeGuard guard(argc, argv);
#endif

  const int N = 32;
  const double L = 1.0, B0 = 1.0, k = 0.5;
  const double q0 = +1.0, q1 = -1.0;  // ions (block 0), electrons (block 1)
  const std::vector<double> rho0 = bump(N, 1.0, 0.40);
  const std::vector<double> rho1 = bump(N, 1.0, 0.20);
  ASSERT_NEAR(periodic_rhs_mean(q0, rho0, q1, rho1), 0.0, 1e-13)
      << "charged two-block fixtures must satisfy the periodic Poisson nullspace before solve";

  // ============================================================================================
  // (A) DEUX BLOCS COMPILES a schemas DIFFERENTS, sans couplage. Le 2e add_compiled_model ne leve
  //     PLUS ; les deux blocs evoluent ; masse conservee par bloc ; potentiel de systeme non trivial.
  // ============================================================================================
  {
    AmrSystemConfig cfg;
    cfg.n = N;
    cfg.L = L;
    cfg.periodicity = {true, true};
    cfg.regrid_every = 0;  // multi-blocs : hierarchie figee

    AmrSystem sim(cfg);
    sim.set_temporal_relations({2}, {1}, {"integral_only"});
    // bloc 0 : ions q=+1, schema none/rusanov.
    add_compiled_model(sim, "ions", exb_model(q0, B0), "none", "rusanov", "conservative",
                       "explicit",
                       /*gamma=*/1.4);
    // bloc 1 : electrons q=-1, schema minmod/rusanov (DIFFERENT du bloc 0). NE LEVE PLUS.
    add_compiled_model(sim, "electrons", exb_model(q1, B0), "minmod", "rusanov", "conservative",
                       "explicit", /*gamma=*/1.4);
    sim.set_poisson("charge_density", "geometric_mg", "periodic");
    sim.set_density("ions", rho0);
    sim.set_density("electrons", rho1);
    test::install_forward_euler_program(sim);

    EXPECT_EQ(sim.n_blocks(), 2) << "A_two_compiled_blocks";

    const std::vector<double> d0_before = sim.density("ions");
    const std::vector<double> d1_before = sim.density("electrons");
    const double m0_before = sim.mass("ions");
    const double m1_before = sim.mass("electrons");

    sim.advance(0.01, 5);

    const std::vector<double> d0_after = sim.density("ions");
    const std::vector<double> d1_after = sim.density("electrons");
    EXPECT_TRUE(all_finite(d0_after) && all_finite(d1_after)) << "A_state_finite";
    EXPECT_TRUE(dmax_field(d0_after, d0_before) > 1e-6) << "A_block0_evolved";
    EXPECT_TRUE(dmax_field(d1_after, d1_before) > 1e-6) << "A_block1_evolved";

    EXPECT_TRUE(std::fabs(sim.mass("ions") - m0_before) < 1e-10) << "A_block0_mass_conserved";
    EXPECT_TRUE(std::fabs(sim.mass("electrons") - m1_before) < 1e-10) << "A_block1_mass_conserved";

    const std::vector<double> phi = sim.potential();
    double pmax = 0;
    for (double v : phi)
      pmax = std::max(pmax, std::fabs(v));
    EXPECT_TRUE(pmax > 1e-8) << "A_system_potential_nonzero";
    EXPECT_TRUE(sim.n_patches() >= 1) << "A_shared_hierarchy_has_fine_patch";
  }

  // ============================================================================================
  // (B) COUPLAGE entre deux blocs compiles par un Program explicite.
  //     La masse COMPOSITE n_ions + n_neutrals est conservee globalement ; la masse ions AUGMENTE.
  //     Both blocks are field-neutral here: this section isolates the coupled source and keeps the
  //     fully periodic elliptic RHS exactly compatible with its constant nullspace.
  // ============================================================================================
  {
    ASSERT_EQ(periodic_rhs_mean(0.0, rho0, 0.0, rho1), 0.0)
        << "field-neutral source fixture must have an exactly zero periodic RHS mean before solve";
    AmrSystemConfig cfg;
    cfg.n = N;
    cfg.L = L;
    cfg.periodicity = {true, true};
    cfg.regrid_every = 0;

    AmrSystem sim(cfg);
    sim.set_temporal_relations({2}, {1}, {"integral_only"});
    add_compiled_model(sim, "ions", exb_model(0.0, B0), "minmod", "rusanov", "conservative",
                       "explicit", /*gamma=*/1.4);
    add_compiled_model(sim, "neutrals", exb_model(0.0, B0), "minmod", "rusanov", "conservative",
                       "explicit", /*gamma=*/1.4);
    sim.set_poisson("charge_density", "geometric_mg", "periodic");
    sim.set_density("ions", rho0);
    sim.set_density("neutrals", rho1);
    // gain sur "ions", perte sur "neutrals" (echange conservatif par cellule).
    register_ionization(sim, "ions", "neutrals", "ions", "neutrals", k);
    install_compiled_coupling_program(sim);

    const double tot0 = sim.mass("ions") + sim.mass("neutrals");
    const double mi0 = sim.mass("ions");

    ASSERT_TRUE(sim.uses_runtime_engine());
    ASSERT_NE(sim.engine(), nullptr);
    sim.advance(0.01, 6);

    EXPECT_TRUE(all_finite(sim.density("ions")) && all_finite(sim.density("neutrals")))
        << "B_state_finite";
    const double tot1 = sim.mass("ions") + sim.mass("neutrals");
    EXPECT_TRUE(std::fabs(tot1 - tot0) < 1e-9) << "B_composite_mass_conserved";
    EXPECT_TRUE(sim.mass("ions") > mi0 + 1e-6) << "B_source_transfers_to_ions";
  }

  // ============================================================================================
  // (C) MELANGE compile (add_compiled_model) + natif (add_block) sur la meme hierarchie partagee.
  //     Le melange etait REFUSE avant ce capstone ; il doit desormais construire et evoluer.
  // ============================================================================================
  {
    AmrSystemConfig cfg;
    cfg.n = N;
    cfg.L = L;
    cfg.periodicity = {true, true};
    cfg.regrid_every = 0;

    AmrSystem sim(cfg);
    sim.set_temporal_relations({2}, {1}, {"integral_only"});
    add_compiled_model(sim, "ions", exb_model(q0, B0), "minmod", "rusanov", "conservative",
                       "explicit", /*gamma=*/1.4);                                   // bloc COMPILE
    sim.add_block("electrons", exb_spec(q1, B0), "none", "rusanov", "conservative",  // bloc NATIF
                  "explicit", 1);
    sim.set_poisson("charge_density", "geometric_mg", "periodic");
    sim.set_density("ions", rho0);
    sim.set_density("electrons", rho1);
    test::install_forward_euler_program(sim);

    EXPECT_EQ(sim.n_blocks(), 2) << "C_mixed_two_blocks";
    const std::vector<double> d0_before = sim.density("ions");
    const std::vector<double> d1_before = sim.density("electrons");
    sim.advance(0.01, 5);
    EXPECT_TRUE(dmax_field(sim.density("ions"), d0_before) > 1e-6) << "C_compiled_block_evolved";
    EXPECT_TRUE(dmax_field(sim.density("electrons"), d1_before) > 1e-6) << "C_native_block_evolved";
    EXPECT_TRUE(std::fabs(sim.mass("ions") - 0.0) >= 0.0) << "C_mass_queryable";  // n'a pas crash
  }

  // ============================================================================================
  // (D) MONO-BLOC COMPILE DETERMINISTE : meme bloc spatial + meme Program joues deux fois -> dmax == 0.
  // ============================================================================================
  {
    const std::vector<double> periodic_state = bump(N, 0.0, 0.40);
    ASSERT_NEAR(periodic_rhs_mean(q0, periodic_state, 0.0, periodic_state), 0.0, 1e-13)
        << "single charged periodic fixture must have zero RHS mean before solve";
    auto run_mono = [&]() {
      AmrSystemConfig cfg;
      cfg.n = N;
      cfg.L = L;
      cfg.periodicity = {true, true};
      cfg.regrid_every = 0;
      AmrSystem sim(cfg);
      add_compiled_model(sim, "ne", exb_model(q0, B0), "none", "rusanov", "conservative",
                         "explicit",
                         /*gamma=*/1.4);
      sim.set_poisson("charge_density", "geometric_mg", "periodic");
      // A single periodic charged block must itself have zero mean; no projection is allowed.
      sim.set_density("ne", periodic_state);
      test::install_forward_euler_program(sim);
      sim.advance(0.01, 5);
      return sim.density("ne");
    };
    const std::vector<double> a = run_mono();
    const std::vector<double> b = run_mono();
    EXPECT_EQ(dmax_field(a, b), 0.0) << "D_monoblock_compiled_bit_identical_dmax0";
  }

  // ============================================================================================
  // (E) NON-REGRESSION DU REFUS : le 2e add_compiled_model NE LEVE PLUS (preuve directe que la file
  //     de specs + build paresseux ont remplace l'ancien throw "un seul bloc compile").
  // ============================================================================================
  {
    const bool threw = raises([&] {
      AmrSystemConfig cfg;
      cfg.n = N;
      cfg.L = L;
      cfg.periodicity = {true, true};
      cfg.regrid_every = 0;
      AmrSystem sim(cfg);
      add_compiled_model(sim, "a", exb_model(q0, B0), "minmod", "rusanov", "conservative",
                         "explicit", 1.4);
      add_compiled_model(sim, "b", exb_model(q1, B0), "minmod", "rusanov", "conservative",
                         "explicit", 1.4);  // 2e bloc compile : NE DOIT PLUS lever
    });
    EXPECT_TRUE(!threw) << "E_second_compiled_block_no_longer_throws";
  }

  // ============================================================================================
  // (F) FAIL-CLOSED : les descripteurs IMEX/stride/masque d'un bloc compile restent des metadonnees
  //     de lowering. Sans Program type qui place Newton, AmrSystem refuse avant de construire ou de
  //     modifier la hierarchie ; il ne retombe jamais sur un stepper prive. La preuve numerique IMEX
  //     Program est dans test_amr_multiblock_imex et test_amr_imex_native.
  // ============================================================================================
  {
    const int Nf = 32;
    const double eps = 1e-5;
    const std::vector<double> rhoF = bubble(Nf);
    AmrSystemConfig cfg;
    cfg.n = Nf;
    cfg.L = L;
    cfg.periodicity = {true, true};
    cfg.regrid_every = 0;
    AmrSystem sim(cfg);
    sim.set_temporal_relations({2}, {1}, {"integral_only"});
    add_compiled_model(sim, "stiff", stiff_cmodel(eps), "minmod", "rusanov", "conservative", "imex",
                       /*gamma=*/1.4, /*substeps=*/1, /*stride=*/2,
                       /*implicit_vars=*/{}, /*implicit_roles=*/{});
    add_compiled_model(sim, "neutral", neutral_cmodel(), "minmod", "rusanov", "conservative",
                       "explicit", /*gamma=*/1.4);
    sim.set_poisson("charge_density", "geometric_mg", "periodic");
    sim.set_density("stiff", rhoF);
    sim.set_density("neutral", rhoF);

    std::string refusal;
    try {
      sim.step(1e-3);
    } catch (const std::exception& error) {
      refusal = error.what();
    }
    EXPECT_TRUE(refusal.find("Program") != std::string::npos)
        << "F_compiled_imex_without_typed_program_fails_closed";

    std::string partial_mask_refusal;
    try {
      AmrSystem invalid(cfg);
      add_compiled_model(invalid, "stiff", stiff_cmodel(eps), "minmod", "rusanov", "conservative",
                         "imex", /*gamma=*/1.4, /*substeps=*/1,
                         /*stride=*/1, /*implicit_vars=*/{}, {"momentum_x"});
    } catch (const std::exception& error) {
      partial_mask_refusal = error.what();
    }
    EXPECT_NE(partial_mask_refusal.find("no executable AMR Program implicit-source primitive"),
              std::string::npos)
        << "F_partial_imex_mask_rejected_before_compiled_builder";

    const bool mask_in_explicit_threw = raises([&] {
      AmrSystem invalid(cfg);
      add_compiled_model(invalid, "stiff", stiff_cmodel(eps), "minmod", "rusanov", "conservative",
                         "explicit", /*gamma=*/1.4, /*substeps=*/1,
                         /*stride=*/1, /*implicit_vars=*/{}, {"momentum_x"});
    });
    EXPECT_TRUE(mask_in_explicit_threw) << "F_partial_mask_rejected_in_explicit_metadata";
  }

  // ============================================================================================
  // (G) ETAT CONSERVATIF COMPLET sur DEUX blocs compiles : le builder differe recoit le vecteur
  //     runtime lie APRES l'enregistrement du modele concret. Les 4 composantes non nulles doivent
  //     etre reproduites BIT POUR BIT au niveau grossier ; aucun repli implicite vers set_density.
  // ============================================================================================
  {
    const int Ns = 8;
    AmrSystemConfig cfg;
    cfg.n = Ns;
    cfg.L = L;
    cfg.periodicity = {true, true};
    cfg.regrid_every = 0;
    AmrSystem sim(cfg);
    sim.set_temporal_relations({2}, {1}, {"integral_only"});
    add_compiled_model(sim, "gas", neutral_cmodel(), "minmod", "rusanov", "conservative",
                       "explicit", /*gamma=*/1.4);
    add_compiled_model(sim, "marker", neutral_cmodel(), "minmod", "rusanov", "conservative",
                       "explicit", /*gamma=*/1.4);
    const std::vector<double> gas_state = full_euler_state(Ns, 1.0);
    const std::vector<double> marker_state = full_euler_state(Ns, 0.75);
    for (double value : gas_state)
      ASSERT_NE(value, 0.0);
    for (double value : marker_state)
      ASSERT_NE(value, 0.0);
    sim.set_conservative_state("gas", gas_state);
    sim.set_conservative_state("marker", marker_state);

    EXPECT_EQ(sim.block_level_state_global("gas", 0), gas_state)
        << "G_compiled_full_state_gas_exact";
    EXPECT_EQ(sim.block_level_state_global("marker", 0), marker_state)
        << "G_compiled_full_state_marker_exact";
  }
}
