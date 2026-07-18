// Contrat mono-bloc de la facade AmrSystem : les parametres NON cables doivent etre REFUSES
// explicitement, plus de no-op silencieux. Les identites de provider inconnues sont des arguments
// invalides (std::invalid_argument), tandis que les configurations runtime incoherentes restent des
// std::runtime_error. Avant ce nettoyage, set_poisson
// stockait rhs/solver sans jamais les valider (on pouvait croire que solver='fft' tournait sur la
// hierarchie alors qu'AmrCouplerMP cable toujours GeometricMG), et add_block acceptait n'importe
// quel time. Ce test verrouille les refus et les schemas temporels reellement cables. Il compile
// python/amr_system.cpp avec le test, la classe AmrSystem etant la facade des bindings.

#include <gtest/gtest.h>

#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/config/model_spec.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

// Bloc ExB scalaire minimal valide (diocotron-like), pour exercer les chemins de refus.
static ModelSpec exb_spec() {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  return s;
}

static ModelSpec magnetic_fluid_spec() {
  ModelSpec s;
  s.transport = "isothermal";
  s.source = "magnetic";
  s.elliptic = "background";
  s.cs2 = 1.0;
  s.qom = 1.0;
  s.alpha = 1.0;
  s.n0 = 1.0;
  return s;
}

TEST(test_amr_system_contract, Runs) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  AmrSystemConfig cfg;
  cfg.n = 16;
  cfg.L = 1.0;
  cfg.periodic = true;

  // --- set_poisson : refus immediat de solver/rhs hors du domaine cable ---------------------
  EXPECT_THROW(
      {
        AmrSystem s(cfg);
        s.set_poisson("charge_density", "fft");
      },
      std::invalid_argument)
      << "set_poisson refuse solver='fft' (seul geometric_mg est cable sur AMR)";
  EXPECT_THROW(
      {
        AmrSystem s(cfg);
        s.set_poisson("charge_density", "inconnu");
      },
      std::invalid_argument)
      << "set_poisson refuse un solver inconnu";
  EXPECT_THROW(
      {
        AmrSystem s(cfg);
        s.set_poisson("densite_bidon", "geometric_mg");
      },
      std::runtime_error)
      << "set_poisson refuse un rhs hors {charge_density, composite}";

  // Les valeurs supportees passent sans lever.
  EXPECT_NO_THROW({
    AmrSystem s(cfg);
    s.set_poisson("charge_density", "geometric_mg");
  }) << "set_poisson accepte charge_density + geometric_mg";
  EXPECT_NO_THROW({
    AmrSystem s(cfg);
    s.set_poisson("composite", "geometric_mg");
  }) << "set_poisson accepte rhs='composite'";

  // --- set_poisson : bc/wall valides au build (poisson_bc/wall_active), donc au 1er mass() ---
  EXPECT_THROW(
      {
        AmrSystem s(cfg);
        s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
        s.set_poisson("charge_density", "geometric_mg", "bc_bidon");
        (void)s.mass();  // declenche ensure_built -> poisson_bc()
      },
      std::runtime_error)
      << "bc inconnu refuse au build";
  EXPECT_THROW(
      {
        AmrSystem s(cfg);
        s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
        s.set_poisson("charge_density", "geometric_mg", "auto", "mur_bidon");
        (void)s.mass();  // declenche ensure_built -> wall_active()
      },
      std::runtime_error)
      << "wall inconnu refuse au build";

  // --- add_block : schemas cables ACCEPTES, valeur inconnue REFUSEE ---------------------------
  // Chaque identifiant public doit atteindre son chemin natif : ``explicit`` canonique (SSPRK2),
  // Forward Euler, SSPRK3 et source raide IMEX. Ce verrou complete les tests numeriques qui
  // distinguent ensuite les trajectoires Euler et SSPRK2.
  for (const char* method : {"explicit", "euler", "ssprk3", "imex"}) {
    EXPECT_NO_THROW({
      AmrSystem s(cfg);
      s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", method, 1);
    }) << "add_block accepte le schema temporel cable '"
       << method << "'";
  }
  EXPECT_THROW(
      {
        AmrSystem s(cfg);
        s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "time_bidon", 1);
      },
      std::runtime_error)
      << "add_block refuse un time hors {explicit, euler, ssprk3, imex}";
  EXPECT_THROW(
      {
        AmrSystem s(cfg);
        s.add_block("ne", exb_spec(), "none", "rusanov", "recon_bidon", "explicit", 1);
      },
      std::runtime_error)
      << "add_block refuse un recon hors {conservative, primitive}";
  EXPECT_THROW(
      {
        AmrSystem s(cfg);
        s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "explicit", 0);
      },
      std::runtime_error)
      << "add_block refuse substeps < 1";

  // --- multi-blocs (capstone PR1) : un 2e bloc natif est desormais ACCEPTE -------------------
  // Bascule sur le moteur runtime AmrRuntime (hierarchie partagee, Poisson somme). On verifie que
  // l'ajout passe sans lever ; la physique (evolution, masse, Poisson somme) est verrouillee par
  // test_amr_system_twoblock.
  EXPECT_NO_THROW({
    AmrSystemConfig c2 = cfg;
    c2.regrid_every = 0;  // multi-blocs PR1 : hierarchie FIGEE
    AmrSystem s(c2);
    s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
    s.add_block("ni", exb_spec(), "minmod", "rusanov", "conservative", "explicit", 1);
  }) << "add_block accepte un second bloc (multi-blocs, hierarchie partagee)";

  // --- DEVERROUILLAGE (capstone Phase 2, C.6) : multi-blocs + regrid_every > 0 est ACCEPTE ----
  // L'ancien REFUS (la hierarchie multi-blocs etait FIGEE) est leve : AmrRuntime porte le regrid
  // d'union des tags (set_regrid + set_block_tag_predicate cables dans build_multi). ensure_built
  // (1er mass()) construit le moteur avec la cadence active au lieu de lever ; le regrid d'union et
  // le mouvement effectif de la hierarchie sont verrouilles par test_amr_multiblock_regrid_union.
  EXPECT_NO_THROW({
    AmrSystemConfig c2 = cfg;
    c2.regrid_every = 5;  // > 0
    AmrSystem s(c2);
    s.set_temporal_relations({2}, {1}, {"integral_only"});
    s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
    s.add_block("ni", exb_spec(), "minmod", "rusanov", "conservative", "explicit", 1);
    (void)s.mass("ne");  // declenche ensure_built -> moteur multi-blocs avec regrid d'union actif
  }) << "multi-blocs + regrid_every > 0 ACCEPTE (regrid d'union des tags, deverrouillage Phase 2)";

  // --- mono-bloc + regrid_every > 0 reste AUTORISE (chemin AmrCouplerMP, regrid intact) -------
  EXPECT_NO_THROW({
    AmrSystemConfig c2 = cfg;
    c2.regrid_every = 5;
    AmrSystem s(c2);
    s.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
    (void)s.mass();  // ensure_built : mono-bloc avec regrid, pas de refus
  }) << "mono-bloc + regrid_every > 0 reste autorise par le runtime AMR unifie";

  // --- B_z : le champ accepte doit atteindre le vrai canal aux, en mono- ET multi-bloc --------
  // Deux runs strictement identiques, B_z=0 puis B_z=2, isolent la source de Lorentz sans dupliquer
  // ici le detail du programme temporel AMR. Une implementation qui stocke seulement B_z sans le
  // publier produit deux etats identiques et echoue.
  for (const int block_count : {1, 2}) {
    auto run = [&](double magnetic_field) {
      AmrSystemConfig magnetic_cfg = cfg;
      magnetic_cfg.n = 8;
      magnetic_cfg.regrid_every = 0;
      AmrSystem s(magnetic_cfg);
      if (block_count == 2)
        s.set_temporal_relations({2}, {1}, {"integral_only"});
      const std::size_t cells =
          static_cast<std::size_t>(magnetic_cfg.n) * static_cast<std::size_t>(magnetic_cfg.n);
      std::vector<double> state(3 * cells, 0.0);
      for (std::size_t cell = 0; cell < cells; ++cell) {
        state[cell] = 1.0;
        state[cells + cell] = 1.0;
      }
      for (int block = 0; block < block_count; ++block) {
        const std::string name = "magnetic_" + std::to_string(block);
        s.add_block(name, magnetic_fluid_spec(), "none", "rusanov", "conservative", "euler", 1);
        s.set_conservative_state(name, state);
      }
      s.set_magnetic_field(std::vector<double>(cells, magnetic_field));
      s.advance(0.01, 1);
      std::vector<std::vector<double>> states;
      states.reserve(static_cast<std::size_t>(block_count));
      for (int block = 0; block < block_count; ++block)
        states.push_back(s.block_level_state_global("magnetic_" + std::to_string(block), 0));
      return states;
    };

    const auto without_field = run(0.0);
    const auto with_field = run(2.0);
    for (int block = 0; block < block_count; ++block) {
      const auto& baseline = without_field[static_cast<std::size_t>(block)];
      const auto& actual = with_field[static_cast<std::size_t>(block)];
      ASSERT_EQ(actual.size(), baseline.size());
      const std::size_t cells = actual.size() / 3;
      double max_delta = 0.0;
      double transverse_delta = 0.0;
      for (std::size_t cell = 0; cell < cells; ++cell) {
        for (int component = 0; component < 3; ++component) {
          const std::size_t index = static_cast<std::size_t>(component) * cells + cell;
          ASSERT_TRUE(std::isfinite(actual[index]));
          max_delta = std::max(max_delta, std::fabs(actual[index] - baseline[index]));
        }
        transverse_delta += actual[2 * cells + cell] - baseline[2 * cells + cell];
      }
      transverse_delta /= static_cast<double>(cells);
      EXPECT_GT(max_delta, 1e-3) << "B_z must change the native block trajectory";
      EXPECT_LT(transverse_delta, -1e-3) << "positive B_z must rotate +m_x toward negative m_y";
    }
  }
}
