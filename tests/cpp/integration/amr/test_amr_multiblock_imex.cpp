// AMR MULTI-BLOCS IMEX (capstone vii / ADC-700): un Program explicite compose transport sans source,
// backward_euler_source local sur un candidat prive, puis commit transactionnel. AmrRuntime reste le
// moteur spatial; il ne choisit plus implicitement le traitement temporel de ce test.
//
// Ce que le test verrouille (cf. tache capstone vii + suite revue #184) :
//   (1) STABILITE RAIDE : un bloc a SOURCE LOCALE RAIDE (relaxation, raideur 1/eps >> 1/dt) sous IMEX
//       sur une hierarchie AMR 2 NIVEAUX est FINI et BORNE, la ou le MEME bloc en EXPLICITE DIVERGE
//       (facteur |1 - dt/eps| >> 1). Rejet nan/inf AVANT toute tolerance. La stabilite est observee
//       DIRECTEMENT sur le champ STIFFENE mx/my/E (comp 1/2/3 du grossier, lues via levels(b)), pas
//       seulement par contamination indirecte de la densite (revue #184) : la source ne touche PAS rho.
//   (2) CONSERVATION : la source raide IMEX choisie est CELLULE-LOCALE et n'agit PAS sur la composante 0
//       (densite) -> la masse du bloc IMEX est conservee a ~machine (hors registres de reflux, cascade
//       fin -> grossier intacte). On le verifie sur 2 niveaux (un patch fin present).
//   (2bis) SOURCE IMPLICITE SUR AMR : une source non lineaire exerce backward-Euler sur le niveau fin,
//       puis la synchronisation Program retablit covered coarse == moyenne 2x2 des enfants. La
//       trajectoire fine suit l'oracle backward-Euler analytique et le bloc voisin reste bit-identique.
//   (3) DISABLE-AND-FAIL : forcer le MEME bloc raide en EXPLICITE (imex=false) le fait EXPLOSER -> la
//       selection IMEX est GENUINEMENT exercee (ce n'est pas un no-op silencieux). C'est le pendant
//       "negatif" de (1) : sans IMEX, la stabilite disparait. L'explosion est verifiee SUR mx/my/E.
//   (3bis) IMEX SOUS-CYCLE substeps>1 (revue #184) : un run IMEX substeps=4 est fini, borne (densite ET
//       mx/my/E), conservatif, et sa trajectoire DIFFERE d'un run IMEX substeps=1 -> le SOUS-CYCLAGE du
//       splitting IMEX porte par le Program est INTENTIONNEL et reellement execute, pas un no-op.
//   (4) OPT-IN BIT-IDENTIQUE : deux Programs TOUT-EXPLICITES identiques donnent le meme resultat
//       (dmax==0). L'IMEX porte par le Program est strictement opt-in.
//   (5) FACADE : AmrSystem.add_block(time="imex") sans Program ne peut pas avancer ; masque partiel,
//       options Newton non-defaut et diagnostics sont refuses avant le build tant qu'aucune primitive
//       Program implicite executable ne les consomme.
//
// La source RAIDE n'est pas ModelSpec-atteignable; deux builders compiles fournissent donc seulement
// les noyaux spatiaux au meme AmrSystem. Toutes les avances passent neanmoins par la facade + Program.

#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include <pops/physics/bricks/bricks.hpp>  // CompositeModel, Euler, BackgroundDensity, ChargeDensity, PotentialForce
#include <pops/numerics/fv/flux_failure.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // detail::make_shared_amr_layout / build_amr_block / dispatch_amr_block
#include <pops/runtime/amr/amr_runtime.hpp>                 // AmrRuntime, AmrRuntimeBlock
#include <pops/runtime/amr_system.hpp>                      // facade AmrSystem
#include <pops/runtime/builders/factory/model_factory.hpp>  // detail::dispatch_model
#include <pops/runtime/config/model_spec.hpp>

#include "amr_transfer_test_authority.hpp"

#include <cmath>
#include <cstdio>
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

namespace {

constexpr double kGamma = 1.4;

// Source RAIDE CELLULE-LOCALE qui NE TOUCHE PAS la densite (composante 0) : relaxation de la QUANTITE
// DE MOUVEMENT (mx, my) et de l'ENERGIE vers un equilibre, de raideur 1/eps. En EXPLICITE (Euler avant)
// mx <- mx (1 - dt/eps) DIVERGE des que dt/eps > 2 ; en IMEX (backward Euler) mx <- mx / (1 + dt/eps)
// reste BORNE pour tout dt > 0 (inconditionnellement stable). La composante 0 (rho) a source NULLE ->
// la MASSE est conservee a la machine, IMEX comme explicite (tant que ce dernier ne diverge pas).
struct StiffMomentumRelax {
  Real inv_eps = Real(0);
  Real e_eq = Real(2.5);  // energie d'equilibre (rho=1, vitesse nulle, p coherent)
  template <class State>
  POPS_HD State apply(const State& u, const Aux&) const {
    State s{};
    // s[0] (densite) = 0 : la source ne cree/detruit PAS de masse -> conservation a la machine.
    if (State::size() > 1)
      s[1] = -inv_eps * u[1];  // -mx / eps
    if (State::size() > 2)
      s[2] = -inv_eps * u[2];  // -my / eps
    if (State::size() > 3)
      s[3] = -inv_eps * (u[3] - e_eq);  // -(E - E_eq) / eps
    return s;
  }
};
// A genuinely state-independent zero elliptic brick. The deliberately unstable explicit oracle can
// reach non-finite state without turning an authored zero RHS into `0 * NaN`.
struct ZeroElliptic {
  template <class State>
  POPS_HD Real rhs(const State&) const {
    return Real(0);
  }
};

using StiffModel = CompositeModel<Euler, StiffMomentumRelax, ZeroElliptic>;
StiffModel make_stiff(double eps) {
  StiffMomentumRelax r;
  r.inv_eps = static_cast<Real>(1.0 / eps);
  return StiffModel{Euler{static_cast<Real>(kGamma)}, r, ZeroElliptic{}};
}

// Modele EXPLICITE neutre (Euler sans source, charge nulle) : un 2e bloc "voisin" pour exercer le
// MULTI-BLOCS (hierarchie partagee, Poisson somme) sans raideur. ExB scalaire ne convient pas (1 var) ;
// on prend un Euler 4 var a source nulle, MEME nombre de variables que le bloc raide (layout coherent).
using NeutralModel = CompositeModel<Euler, NoSource, ZeroElliptic>;
NeutralModel make_neutral() {
  return NeutralModel{Euler{static_cast<Real>(kGamma)}, NoSource{}, ZeroElliptic{}};
}

// Source scalaire non lineaire utilisee pour rendre la cascade fine -> grossier observable. Pour
// du/dt = -k u^2, backward Euler resout k*dt*u_{n+1}^2 + u_{n+1} - u_n = 0. Comme la racine positive
// est non lineaire en u_n, avancer le parent couvert independamment ne donne pas la moyenne des
// enfants : seule la synchronisation Program restaure l'invariant AMR.
struct NonlinearDensityDecay {
  using State = StateVec<1>;
  using Prim = State;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;

  Real rate = Real(0);

  POPS_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  POPS_HD State source(const State& u, const Aux&) const { return State{-rate * u[0] * u[0]}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
  POPS_HD Prim to_primitive(const State& state) const { return state; }
  POPS_HD State to_conservative(const Prim& primitive) const { return primitive; }

  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"rho"}, 1, {VariableRole::Density}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"rho"}, 1, {VariableRole::Density}};
  }
};

static_assert(PhysicalModel<NonlinearDensityDecay>);

double backward_euler_quadratic(double value, double dt, double rate) {
  return (2.0 * value) / (1.0 + std::sqrt(1.0 + 4.0 * rate * dt * value));
}

// densite + impulsion initiales : une bulle de densite avec une impulsion non nulle (pour que la source
// de relaxation de mx/my AIT un effet a relaxer). On pose ici la SEULE composante 0 (densite) via le
// chemin coupler_write_coarse (qui derive mx,my,E d'un Euler au repos) ; la raideur agit ensuite sur
// l'impulsion engendree par le transport (gradients de la bulle) -- suffisant pour exploser en explicite.
std::vector<double> bubble(int n) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      rho[static_cast<std::size_t>(j) * n + i] = 1.0 + 0.5 * std::exp(-(x * x + y * y) / 0.02);
    }
  return rho;
}

double mean_of(const std::vector<double>& values) {
  double sum = 0.0;
  for (double value : values)
    sum += value;
  return sum / static_cast<double>(values.size());
}

double periodic_rhs_mean(double q0, const std::vector<double>& rho0, double q1,
                         const std::vector<double>& rho1) {
  return q0 * mean_of(rho0) + q1 * mean_of(rho1);
}

bool all_finite(const std::vector<double>& v) {
  for (double x : v)
    if (!std::isfinite(x))
      return false;
  return true;
}
double maxabs(const std::vector<double>& v) {
  double m = 0;
  for (double x : v)
    m = std::fmax(m, std::fabs(x));
  return m;
}
double dmax_field(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  const std::size_t nn = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t i = 0; i < nn; ++i)
    d = std::fmax(d, std::fabs(a[i] - b[i]));
  return d;
}

template <class Model>
AmrCompiledBlockBuilder make_program_block_builder(Model model) {
  return
      [model](const detail::SharedAmrLayout& layout, const std::string& name,
              const std::vector<double>& density, bool has_density,
              const std::vector<double>& state, bool has_state, double gamma, int substeps,
              bool recon_prim, bool imex, int stride, const std::vector<std::string>& implicit_vars,
              const std::vector<std::string>& implicit_roles, double pos_floor, double weno_epsilon,
              bool wave_speed_cache) {
        if (imex || !implicit_vars.empty() || !implicit_roles.empty())
          throw std::invalid_argument(
              "the IMEX test Program owns source treatment; its spatial block must be explicit");
        return detail::build_amr_block<Model, Minmod, RusanovFlux>(
            model, layout, name, density, has_density, gamma, substeps, recon_prim, stride,
            has_state ? &state : nullptr, pos_floor, weno_epsilon, wave_speed_cache);
      };
}

// Install one explicitly authored Lie Program:
//   source-free Forward Euler transport; local backward-Euler source on a private candidate; commit.
// The explicit oracle uses the same Program skeleton but evaluates the complete stiff RHS directly.
void install_stiff_pair_program(AmrSystem& system, StiffModel stiff_model, bool implicit_stiff,
                                int stiff_substeps) {
  system.set_program_block_map({0, 1});
  system.install_program_step([](double) {});
  if (!system.uses_runtime_engine() || system.engine() == nullptr)
    throw std::runtime_error("stiff-pair test Program requires the materialized AMR engine");

  auto context = std::make_shared<runtime::program::AmrProgramContext>(system.engine(), &system);
  context->configure_primary_clock("test.clock.macro");
  context->install([context, stiff_model, implicit_stiff, stiff_substeps](double macro_dt) {
    context->advance_hierarchy(macro_dt, [context, stiff_model, implicit_stiff,
                                          stiff_substeps](double level_dt) {
      (void)consume_solve_outcome(context->solve_fields());
      MultiFab& stiff_live = context->state(0);
      MultiFab& neutral_live = context->state(1);
      MultiFab& stiff_candidate = context->scratch_state(1000, 0, stiff_live);
      MultiFab& neutral_candidate = context->scratch_state(1001, 0, neutral_live);
      context->lincomb(stiff_candidate, Real(1), stiff_live, Real(0), stiff_live);
      context->lincomb(neutral_candidate, Real(1), neutral_live, Real(0), neutral_live);

      const Real stiff_dt = Real(level_dt) / static_cast<Real>(stiff_substeps);
      for (int substep = 0; substep < stiff_substeps; ++substep) {
        context->set_stage_time(substep, stiff_substeps);
        MultiFab& stiff_rate = context->rhs_scratch(2000 + substep, 0, stiff_candidate);
        if (implicit_stiff) {
          context->neg_div_flux_default_into(0, stiff_candidate, stiff_rate, 3000 + substep);
          context->axpy(stiff_candidate, stiff_dt, stiff_rate, stiff_dt, {{1, 1, 1}});
          (void)consume_solve_outcome(
              backward_euler_source(stiff_model, context->aux(), stiff_candidate, stiff_dt,
                                    NewtonOptions{}, ImplicitMask<StiffModel::n_vars>{}, nullptr));
        } else {
          context->rhs_into(0, stiff_candidate, stiff_rate, 3000 + substep);
          context->axpy(stiff_candidate, stiff_dt, stiff_rate, stiff_dt, {{1, 1, 1}});
        }
      }

      context->set_stage_time(0, 1);
      MultiFab& neutral_rate = context->rhs_scratch(2100, 0, neutral_candidate);
      context->rhs_into(1, neutral_candidate, neutral_rate, 3100);
      context->axpy(neutral_candidate, Real(level_dt), neutral_rate, Real(level_dt), {{1, 1, 1}});
      context->commit_many({{&stiff_live, &stiff_candidate}, {&neutral_live, &neutral_candidate}});
    });
  });
  system.set_program_block_map({0, 1});
}

// Source-only Program used by the nonlinear AMR oracle below. It deliberately commits both private
// candidates: the decaying block changes through backward Euler, while the neighbouring block is
// copied and must therefore survive level traversal + synchronization bit-for-bit.
void install_nonlinear_source_program(AmrSystem& system, NonlinearDensityDecay decay) {
  system.set_program_block_map({0, 1});
  system.install_program_step([](double) {});
  if (!system.uses_runtime_engine() || system.engine() == nullptr)
    throw std::runtime_error("nonlinear-source test Program requires the materialized AMR engine");

  auto context = std::make_shared<runtime::program::AmrProgramContext>(system.engine(), &system);
  context->configure_primary_clock("test.clock.nonlinear-source");
  context->install([context, decay](double macro_dt) {
    context->advance_hierarchy(macro_dt, [context, decay](double level_dt) {
      context->set_stage_time(0, 1);
      MultiFab& decay_live = context->state(0);
      MultiFab& neutral_live = context->state(1);
      MultiFab& decay_candidate = context->scratch_state(4000, 0, decay_live);
      MultiFab& neutral_candidate = context->scratch_state(4001, 0, neutral_live);
      context->lincomb(decay_candidate, Real(1), decay_live, Real(0), decay_live);
      context->lincomb(neutral_candidate, Real(1), neutral_live, Real(0), neutral_live);
      (void)consume_solve_outcome(backward_euler_source(
          decay, context->aux(), decay_candidate, Real(level_dt), NewtonOptions{},
          ImplicitMask<NonlinearDensityDecay::n_vars>{}, nullptr));
      context->commit_many({{&decay_live, &decay_candidate}, {&neutral_live, &neutral_candidate}});
    });
  });
  system.set_program_block_map({0, 1});
}

// Construit une facade AmrSystem a deux blocs sur une hierarchie deux niveaux. Les builders ne
// transportent aucune decision temporelle : le Program ci-dessus est l'unique autorite IMEX/explicite.
std::unique_ptr<AmrSystem> make_stiff_pair(int N, double L, double eps, bool imex_stiff,
                                           const std::vector<double>& rho, int substeps = 1) {
  AmrSystemConfig cfg;
  cfg.n = N;
  cfg.L = L;
  cfg.level_count = 2;
  cfg.regrid_every = 0;
  cfg.periodicity = {true, true};
  auto system = std::make_unique<AmrSystem>(cfg);
  const StiffModel stiff_model = make_stiff(eps);
  system->set_compiled_block(StiffModel::n_vars, kGamma, /*substeps=*/1,
                             make_program_block_builder(stiff_model), "stiff");
  system->set_compiled_block(NeutralModel::n_vars, kGamma, /*substeps=*/1,
                             make_program_block_builder(make_neutral()), "neutral");
  system->set_density("stiff", rho);
  system->set_density("neutral", rho);
  system->set_poisson("charge_density", "geometric_mg", "periodic");
  system->set_refinement(1e29);
  system->set_temporal_relations({2}, {1}, {"integral_only"});
  install_stiff_pair_program(*system, stiff_model, imex_stiff, substeps);
  return system;
}

std::unique_ptr<AmrSystem> make_nonlinear_source_pair(int n, double rate,
                                                      const std::vector<double>& initial) {
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.level_count = 2;
  cfg.regrid_every = 0;
  cfg.periodicity = {true, true};
  auto system = std::make_unique<AmrSystem>(cfg);
  const NonlinearDensityDecay decay{Real(rate)};
  system->set_compiled_block(NonlinearDensityDecay::n_vars, kGamma, /*substeps=*/1,
                             make_program_block_builder(decay), "decay");
  system->set_compiled_block(NonlinearDensityDecay::n_vars, kGamma, /*substeps=*/1,
                             make_program_block_builder(NonlinearDensityDecay{}), "neutral");
  system->set_density("decay", initial);
  system->set_density("neutral", initial);
  system->set_poisson("charge_density", "geometric_mg", "periodic");
  system->set_refinement(1e29);
  system->set_temporal_relations({2}, {1}, {"integral_only"});
  install_nonlinear_source_program(*system, decay);
  return system;
}

// Lit DIRECTEMENT le grossier (niveau 0) du bloc @p b et renvoie le max |U(.,.,c)| sur les
// composantes STIFFENEES mx/my/E (c=1,2,3), tout en signalant la presence d'un non-fini. Le reviewer
// #184 a note que borner la densite (comp 0) n'observe la stabilite qu'INDIRECTEMENT (la source raide
// StiffMomentumRelax laisse rho INTACTE et ne stiffenne que mx/my/E) : on lit donc les composantes
// REELLEMENT relaxees, sans changer l'API de production (rt est non-const et expose levels(b)). On
// itere les fabs LOCAUX (local_size()==0 sur un rang
// sans boite -> max nul, MPI-safe) et les cellules VALIDES (box(li), pas les ghosts). @p finite mis a
// false des qu'une cellule est non finie.
double max_momentum_energy_coarse(AmrRuntime& rt, std::size_t b, bool& finite) {
  finite = true;
  double m = 0.0;
  const MultiFab& U = rt.levels(b)[0].U;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 a = U.fab(li).const_array();
    const Box2D box = U.box(li);
    for (int j = box.lo[1]; j <= box.hi[1]; ++j)
      for (int i = box.lo[0]; i <= box.hi[0]; ++i)
        for (int c = 1; c <= 3; ++c) {
          const double v = static_cast<double>(a(i, j, c));
          if (!std::isfinite(v))
            finite = false;
          m = std::fmax(m, std::fabs(v));
        }
  }
  return m;
}

// modeles ModelSpec pour la FACADE : ExB scalaire (charge q) et Euler+potential (source raide self-consistent).
ModelSpec exb_charge(double q, double B0) {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  s.q = q;
  s.B0 = B0;
  return s;
}
ModelSpec pot_charge(double qom) {
  ModelSpec s;
  s.transport = "compressible";
  s.source = "potential";
  s.elliptic = "charge";
  s.gamma = kGamma;
  s.qom = qom;
  s.q = 1.0;
  return s;
}
std::vector<double> bump(int n, double base, double amp) {
  std::vector<double> r(static_cast<std::size_t>(n) * n, base);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const bool in = (i >= n / 4 && i < 3 * n / 4 && j >= n / 4 && j < 3 * n / 4);
      r[static_cast<std::size_t>(j) * n + i] = base + (in ? amp : -amp / 3.0);
    }
  return r;
}

}  // namespace

TEST(test_amr_multiblock_imex, Runs) {
#if defined(POPS_HAS_KOKKOS)
  int argc = 0;
  char** argv = nullptr;
  Kokkos::ScopeGuard guard(argc, argv);
#endif

  const int N = 32;
  const double L = 1.0;
  const std::vector<double> rho = bubble(N);
  ASSERT_EQ(periodic_rhs_mean(0.0, rho, 0.0, rho), 0.0)
      << "exact-zero elliptic bricks must have zero periodic RHS mean before solve";

  const std::vector<double> facade_rho_a = bump(N, 1.0, 0.40);
  const std::vector<double> facade_rho_b = bump(N, 1.0, 0.20);
  ASSERT_NEAR(periodic_rhs_mean(+1.0, facade_rho_a, -1.0, facade_rho_b), 0.0, 1e-13)
      << "charged facade fixtures must satisfy the periodic Poisson nullspace before solve";

  // ============================================================================================
  // (2bis) BACKWARD-EULER NON LINEAIRE SUR LE PROGRAM AMR.
  //     Le niveau fin porte volontairement quatre valeurs differentes par cellule parente. Apres
  //     deux sous-pas fins dt/2, chaque valeur doit suivre la racine analytique de backward Euler,
  //     la cellule grossiere couverte doit etre leur moyenne 2x2, et le bloc voisin doit rester
  //     bit-identique sur les deux niveaux.
  // ============================================================================================
  {
    constexpr double rate = 0.7;
    constexpr double source_dt = 0.1;
    auto sim = make_nonlinear_source_pair(N, rate, rho);
    AmrRuntime& rt = *sim->engine();
    ASSERT_EQ(rt.nlev(), 2);

    MultiFab& fine_decay = rt.levels(0)[1].U;
    ASSERT_GT(fine_decay.local_size(), 0);
    for (int li = 0; li < fine_decay.local_size(); ++li) {
      Array4 values = fine_decay.fab(li).array();
      const Box2D box = fine_decay.box(li);
      for (int j = box.lo[1]; j <= box.hi[1]; ++j)
        for (int i = box.lo[0]; i <= box.hi[0]; ++i)
          values(i, j, 0) = Real(0.75) + Real(0.04) * Real(i & 1) + Real(0.07) * Real(j & 1) +
                            Real(0.001) * Real(i + j);
    }
    device_fence();

    // Normalize both accepted parents before the oracle snapshot. Repeating the same average-down
    // after an unchanged neutral trajectory must therefore be exactly idempotent.
    rt.average_down_level(0, 1);
    rt.average_down_level(1, 1);
    const std::vector<double> fine_before = sim->block_level_state_global("decay", 1);
    const std::vector<double> neutral_coarse_before = sim->block_level_state_global("neutral", 0);
    const std::vector<double> neutral_fine_before = sim->block_level_state_global("neutral", 1);
    ASSERT_FALSE(fine_before.empty());

    double nonlinear_sync_discriminator = 0.0;
    for (int li = 0; li < fine_decay.local_size(); ++li) {
      const ConstArray4 fine = fine_decay.fab(li).const_array();
      const Box2D box = fine_decay.box(li);
      for (int j = box.lo[1]; j + 1 <= box.hi[1]; j += 2)
        for (int i = box.lo[0]; i + 1 <= box.hi[0]; i += 2) {
          const double child_initial[4] = {fine(i, j, 0), fine(i + 1, j, 0), fine(i, j + 1, 0),
                                           fine(i + 1, j + 1, 0)};
          const double parent_initial =
              0.25 * (child_initial[0] + child_initial[1] + child_initial[2] + child_initial[3]);
          const double unsynchronized_parent =
              backward_euler_quadratic(parent_initial, source_dt, rate);
          double synchronized_parent = 0.0;
          for (double child : child_initial) {
            child = backward_euler_quadratic(child, source_dt / 2.0, rate);
            child = backward_euler_quadratic(child, source_dt / 2.0, rate);
            synchronized_parent += 0.25 * child;
          }
          nonlinear_sync_discriminator = std::fmax(
              nonlinear_sync_discriminator, std::fabs(unsynchronized_parent - synchronized_parent));
        }
    }
    EXPECT_GT(nonlinear_sync_discriminator, 1e-6)
        << "fixture must fail if the covered parent publishes its independent nonlinear solve";

    sim->step(source_dt);

    const std::vector<double> fine_after = sim->block_level_state_global("decay", 1);
    ASSERT_EQ(fine_after.size(), fine_before.size());
    EXPECT_EQ(sim->block_level_state_global("neutral", 0), neutral_coarse_before)
        << "implicit source on block 0 must not mutate the neighbouring coarse state";
    EXPECT_EQ(sim->block_level_state_global("neutral", 1), neutral_fine_before)
        << "implicit source on block 0 must not mutate the neighbouring fine state";

    double analytic_error = 0.0;
    double active_change = 0.0;
    for (std::size_t index = 0; index < fine_before.size(); ++index) {
      double expected = backward_euler_quadratic(fine_before[index], source_dt / 2.0, rate);
      expected = backward_euler_quadratic(expected, source_dt / 2.0, rate);
      analytic_error = std::fmax(analytic_error, std::fabs(fine_after[index] - expected));
      active_change = std::fmax(active_change, std::fabs(fine_after[index] - fine_before[index]));
    }
    EXPECT_LT(analytic_error, 2e-9)
        << "fine level must execute the two authored backward-Euler half-steps";
    EXPECT_GT(active_change, 1e-4) << "fine-level implicit source must be observable";

    device_fence();
    const MultiFab& coarse_after = rt.levels(0)[0].U;
    const MultiFab& fine_after_mf = rt.levels(0)[1].U;
    double covered_error = 0.0;
    int covered_samples = 0;
    for (int li = 0; li < fine_after_mf.local_size(); ++li) {
      const ConstArray4 fine = fine_after_mf.fab(li).const_array();
      const Box2D box = fine_after_mf.box(li);
      for (int j = box.lo[1]; j + 1 <= box.hi[1]; j += 2)
        for (int i = box.lo[0]; i + 1 <= box.hi[0]; i += 2) {
          const int parent_i = i / 2;
          const int parent_j = j / 2;
          const int parent_box = mf_find_box(coarse_after, parent_i, parent_j);
          ASSERT_GE(parent_box, 0);
          const double average = 0.25 * (fine(i, j, 0) + fine(i + 1, j, 0) + fine(i, j + 1, 0) +
                                         fine(i + 1, j + 1, 0));
          const double parent = coarse_after.fab(parent_box).const_array()(parent_i, parent_j, 0);
          ASSERT_TRUE(std::isfinite(parent) && std::isfinite(average));
          covered_error = std::fmax(covered_error, std::fabs(parent - average));
          ++covered_samples;
        }
    }
    EXPECT_GT(covered_samples, 0);
    EXPECT_LT(covered_error, 1e-12)
        << "Program synchronization must average nonlinear implicit fine state onto covered coarse "
           "cells";
  }

  // ============================================================================================
  // (1)+(2)+(3) STABILITE RAIDE + CONSERVATION + DISABLE-AND-FAIL via le Program AMR.
  //     eps << dt : explicite (facteur |1 - dt/eps| >> 1) DIVERGE, IMEX (backward Euler) reste fini.
  // ============================================================================================
  const double eps = 1e-5, dt = 1e-3;
  const int K = 12;  // macro-pas

  // (1) IMEX : bloc raide STABLE (fini + borne) sur 2 niveaux.
  {
    auto sim = make_stiff_pair(N, L, eps, /*imex_stiff=*/true, rho);
    AmrRuntime& rt = *sim->engine();
    const Real m0 = rt.mass(0);  // masse du bloc raide AVANT (sur le grossier, cascade incluse)
    EXPECT_EQ(rt.nlev(), 2)
        << "imex_two_levels_present";  // un patch fin existe (couverture exercee)
    for (int s = 0; s < K; ++s)
      sim->step(dt);
    const std::vector<double> dStiff = rt.density(0);
    const std::vector<double> dNeutral = rt.density(1);
    const Real m1 = rt.mass(0);
    // AVANT toute tolerance : etat fini (un nan passerait une borne par hasard).
    EXPECT_TRUE(all_finite(dStiff) && all_finite(dNeutral)) << "imex_stiff_state_finite";
    EXPECT_TRUE(maxabs(dStiff) < 1e3) << "imex_stiff_state_bounded";
    // (3-revue #184) STABILITE OBSERVEE SUR LE BON CHAMP : la source raide ne stiffenne PAS la densite
    // (comp 0) mais mx/my/E (comp 1/2/3). Borner la seule densite n'observe la stabilite qu'INDIRECTEMENT
    // (par contamination via le transport). On lit donc DIRECTEMENT mx/my/E du grossier et on exige
    // qu'elles restent finies ET bornees sous IMEX (la ou l'explicite, ci-dessous, les fait diverger).
    bool me_finite = false;
    const double me_max = max_momentum_energy_coarse(rt, 0, me_finite);
    EXPECT_TRUE(me_finite) << "imex_stiff_momentum_energy_finite_DIRECT";
    EXPECT_TRUE(me_max < 1e3) << "imex_stiff_momentum_energy_bounded_DIRECT";
    // (2) CONSERVATION : la source raide ne touche pas la densite (comp 0) -> masse conservee ~machine.
    const double drift =
        std::fabs(static_cast<double>(m1 - m0)) / (std::fabs(static_cast<double>(m0)) + 1e-30);
    EXPECT_TRUE(drift < 1e-12) << "imex_stiff_mass_conserved_to_machine";
    std::printf(
        "      IMEX : max(rho)=%.3e, max|mx,my,E|=%.3e, derive de masse=%.3e (eps=%.0e, dt=%.0e)\n",
        maxabs(dStiff), me_max, drift, eps, dt);
  }

  // (3) DISABLE-AND-FAIL : MEME bloc raide en EXPLICITE -> EXPLOSE. Prouve que la selection IMEX de (1)
  //     est GENUINEMENT exercee (sans elle, la stabilite disparait). On observe l'explosion SUR LE CHAMP
  //     STIFFENE (mx/my/E directement, comp 1/2/3), la ou la source agit, pas seulement sur la densite.
  {
    auto sim = make_stiff_pair(N, L, eps, /*imex_stiff=*/false, rho);
    AmrRuntime& rt = *sim->engine();
    bool explicit_rejected = false;
    try {
      for (int s = 0; s < K; ++s)
        sim->step(dt);
    } catch (const FluxEvaluationFailure& failure) {
      if (failure.status() != EvaluationStatus::kReject ||
          failure.action() != TransactionFailureAction::kRejectStep ||
          failure.reason_code() != 0x53544201u || failure.phase() != "compute_face_fluxes")
        throw;
      explicit_rejected = true;
    }
    const std::vector<double> dStiff = explicit_rejected ? std::vector<double>{} : rt.density(0);
    bool me_finite = false;
    const double me_max = explicit_rejected ? 0.0 : max_momentum_energy_coarse(rt, 0, me_finite);
    // L'explosion DOIT etre visible sur mx/my/E (le champ que la raideur attaque) ; on garde aussi le
    // critere densite (contamination par le transport) pour la lisibilite du diagnostic.
    const bool me_blew_up = explicit_rejected || (me_finite && me_max > 1e3);
    const bool rho_blew_up = explicit_rejected || (all_finite(dStiff) && maxabs(dStiff) > 1e3);
    EXPECT_TRUE(me_blew_up)
        << "explicit_stiff_momentum_energy_BLOWS_UP_DIRECT (disable-and-fail sur mx/my/E)";
    EXPECT_TRUE(rho_blew_up)
        << "explicit_stiff_BLOWS_UP (disable-and-fail : IMEX genuinement requis)";
    std::printf(
        "      EXPLICITE : %s (la stabilite vient bien du pas implicite)\n",
        explicit_rejected
            ? "REJETE AVANT PUBLICATION D'UN ETAT NON FINI"
            : (me_finite && all_finite(dStiff) ? "borne >> 1" : "ETAT NON FINI PUBLIE (ECHEC)"));
  }

  // ============================================================================================
  // (3bis) IMEX SOUS-CYCLE substeps>1 (revue #184) : le Program sous-cycle explicitement le splitting
  //     IMEX (K=substeps pas de Lie sur dt/K). Ce test verrouille cette semantique : un run IMEX
  //     substeps=4 est (a) fini, (b) borne (densite ET mx/my/E directement),
  //     (c) conservatif en masse, et SURTOUT (d) sa trajectoire DIFFERE d'un run IMEX substeps=1 (memes
  //     eps/dt/macro-pas). La difference PROUVE que le sous-cyclage est INTENTIONNEL et REELLEMENT
  //     execute (si substeps etait silencieusement ignore comme en compile-time, dmax serait nul).
  // ============================================================================================
  {
    // substeps=1 : reference (un seul pas de Lie par macro-pas).
    auto sim1 = make_stiff_pair(N, L, eps, /*imex_stiff=*/true, rho, /*substeps=*/1);
    AmrRuntime& rt1 = *sim1->engine();
    const Real m0_1 = rt1.mass(0);
    for (int s = 0; s < K; ++s)
      sim1->step(dt);
    const std::vector<double> d1 = rt1.density(0);
    const double drift1 = std::fabs(static_cast<double>(rt1.mass(0) - m0_1)) /
                          (std::fabs(static_cast<double>(m0_1)) + 1e-30);

    // substeps=4 : meme eps/dt/macro-pas, mais le moteur SOUS-CYCLE le splitting IMEX en 4 pas de dt/4.
    auto sim4 = make_stiff_pair(N, L, eps, /*imex_stiff=*/true, rho, /*substeps=*/4);
    AmrRuntime& rt4 = *sim4->engine();
    const Real m0_4 = rt4.mass(0);
    for (int s = 0; s < K; ++s)
      sim4->step(dt);
    const std::vector<double> d4 = rt4.density(0);
    bool me4_finite = false;
    const double me4_max = max_momentum_energy_coarse(rt4, 0, me4_finite);
    const double drift4 = std::fabs(static_cast<double>(rt4.mass(0) - m0_4)) /
                          (std::fabs(static_cast<double>(m0_4)) + 1e-30);

    // (a)(b) le run sous-cycle reste fini + borne (backward-Euler stable a tout pas ; transport plus sur
    //        en CFL sur dt/4) sur la densite ET sur le champ stiffene mx/my/E lu directement.
    EXPECT_TRUE(all_finite(d4) && me4_finite) << "imex_subcycled_s4_finite";
    EXPECT_TRUE(maxabs(d4) < 1e3 && me4_max < 1e3) << "imex_subcycled_s4_bounded";
    // (c) conservation : la source raide laisse rho intacte -> masse conservee ~machine, comme substeps=1.
    EXPECT_TRUE(drift4 < 1e-12) << "imex_subcycled_s4_mass_conserved";
    // (d) VERROU : substeps=4 DIFFERE de substeps=1 -> le sous-cyclage IMEX est intentionnel et execute.
    const double d14 = dmax_field(d1, d4);
    EXPECT_TRUE(d14 > 0.0) << "imex_subcycled_s4_DIFFERS_from_s1 (sous-cyclage assume, pas ignore)";
    std::printf(
        "      IMEX substeps : s1 (derive=%.2e) vs s4 (max|mx,my,E|=%.3e, derive=%.2e), "
        "dmax(rho)=%.3e\n",
        drift1, me4_max, drift4, d14);
  }

  // ============================================================================================
  // (4) OPT-IN BIT-IDENTIQUE : deux Programs TOUT-EXPLICITES identiques donnent le meme resultat
  //     (dmax==0). L'IMEX ne perturbe rien tant que le Program ne l'appelle pas.
  // ============================================================================================
  {
    auto run_all_explicit = [&]() {
      // Regime non raide : l'oracle Program reste entièrement explicite et ne diverge pas.
      auto sim = make_stiff_pair(N, L, /*eps=*/1.0, /*imex_stiff=*/false, rho);
      for (int s = 0; s < 5; ++s)
        sim->step(1e-3);
      return sim->engine()->density(0);
    };
    const std::vector<double> a = run_all_explicit();
    const std::vector<double> b = run_all_explicit();
    EXPECT_TRUE(all_finite(a)) << "all_explicit_state_finite";
    EXPECT_EQ(dmax_field(a, b), 0.0) << "all_explicit_multiblock_bit_identical";
  }

  // ============================================================================================
  // (5) FACADE AmrSystem : une configuration IMEX sans Program construit encore le moteur spatial,
  //     mais ne constitue jamais une autorite temporelle implicite.
  // ============================================================================================
  {
    // (5a) deux blocs via la facade : le drapeau time="imex" traverse encore le builder spatial,
    //      mais step refuse avant le build paresseux tant qu'aucun Program n'est installe.
    AmrSystemConfig cfg;
    cfg.n = N;
    cfg.L = L;
    cfg.periodicity = {true, true};
    cfg.regrid_every = 0;  // multi-blocs : hierarchie figee
    AmrSystem sim(cfg);
    sim.set_temporal_relations({2}, {1}, {"integral_only"});
    sim.add_block("A", pot_charge(50.0), "minmod", "rusanov", "conservative", "imex", 1, 1);
    sim.add_block("B", exb_charge(-1.0, 1.0), "none", "rusanov", "conservative", "explicit", 1, 1);
    sim.set_poisson("charge_density", "geometric_mg", "periodic");
    sim.set_density("A", facade_rho_a);
    sim.set_density("B", facade_rho_b);
    ASSERT_EQ(sim.engine(), nullptr);
    EXPECT_THROW(sim.step(5e-3), std::logic_error);
    EXPECT_EQ(sim.engine(), nullptr);
    EXPECT_DOUBLE_EQ(sim.time(), 0.0);
    ASSERT_TRUE(sim.uses_runtime_engine());
    ASSERT_NE(sim.engine(), nullptr);
    EXPECT_EQ(sim.n_blocks(), 2) << "facade_two_blocks";
    EXPECT_TRUE(all_finite(sim.density("A")) && all_finite(sim.density("B")))
        << "facade_multiblock_imex_builds_finite";

    // (5b) masque IMEX partiel REFUSE en explicite (pas d'ignore silencieux).
    {
      AmrSystem s2(cfg);
      EXPECT_THROW(
          s2.add_block("A", pot_charge(50.0), "minmod", "rusanov", "conservative", "explicit", 1, 1,
                       /*implicit_vars=*/{}, /*implicit_roles=*/{"momentum_x"}),
          std::runtime_error)
          << "facade_mask_rejected_in_explicit";
    }

    // (5c) Aucun masque IMEX partiel ne peut etre accepte tant qu'une primitive Program implicite
    //      executable ne le consomme. Le refus precede volontairement la resolution nom/role : un
    //      selecteur valide et un selecteur absent ont le meme contrat de capacite fail-closed.
    const auto expect_partial_mask_unavailable = [&](const std::vector<std::string>& implicit_vars,
                                                     const std::vector<std::string>& implicit_roles,
                                                     const char* label) {
      AmrSystem masked(cfg);
      std::string diagnostic;
      try {
        masked.add_block("A", pot_charge(50.0), "minmod", "rusanov", "conservative", "imex", 1, 1,
                         implicit_vars, implicit_roles);
        FAIL() << label;
      } catch (const std::runtime_error& error) {
        diagnostic = error.what();
      }
      EXPECT_NE(diagnostic.find("implicit_vars / implicit_roles"), std::string::npos) << diagnostic;
      EXPECT_NE(diagnostic.find("no executable AMR Program implicit-source primitive"),
                std::string::npos)
          << diagnostic;
      EXPECT_EQ(masked.engine(), nullptr);
    };
    expect_partial_mask_unavailable({"rho_u"}, {}, "facade_partial_name_mask_must_fail_closed");
    expect_partial_mask_unavailable({}, {"momentum_x"},
                                    "facade_partial_role_mask_must_fail_closed");

    // (5d) Les réglages Newton et leur rapport appartiennent au solve typé du Program. Les accepter
    // sur le bloc AMR serait un no-op silencieux puisque le moteur spatial ne les consomme pas.
    const auto expect_newton_metadata_unavailable = [&](const NewtonOptions& newton,
                                                        bool diagnostics, const char* label) {
      AmrSystem configured(cfg);
      std::string diagnostic;
      try {
        configured.add_block("A", pot_charge(50.0), "minmod", "rusanov", "conservative", "imex", 1,
                             1, {}, {}, newton, diagnostics);
        FAIL() << label;
      } catch (const std::runtime_error& error) {
        diagnostic = error.what();
      }
      EXPECT_NE(diagnostic.find("NewtonOptions / newton_diagnostics"), std::string::npos)
          << diagnostic;
      EXPECT_NE(diagnostic.find("typed Program implicit-source solve"), std::string::npos)
          << diagnostic;
      EXPECT_EQ(configured.engine(), nullptr);
    };
    NewtonOptions tuned_newton;
    ++tuned_newton.max_iters;
    expect_newton_metadata_unavailable(tuned_newton, false, "facade_tuned_newton_must_fail_closed");
    expect_newton_metadata_unavailable(NewtonOptions{}, true,
                                       "facade_newton_diagnostics_must_fail_closed");
  }
}
