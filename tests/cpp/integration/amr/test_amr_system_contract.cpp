// Contrat mono-bloc de la facade AmrSystem : les parametres NON cables doivent etre REFUSES
// explicitement, plus de no-op silencieux. Les identites de provider inconnues sont des arguments
// invalides (std::invalid_argument), tandis que les configurations runtime incoherentes restent des
// std::runtime_error. Avant ce nettoyage, set_poisson
// stockait rhs/solver sans jamais les valider (on pouvait croire que solver='fft' tournait sur la
// hierarchie alors qu'AmrCouplerMP cable toujours GeometricMG), et add_block acceptait n'importe
// quel time. Ce test verrouille les refus et les schemas temporels reellement cables. Il compile
// python/amr_system.cpp avec le test, la classe AmrSystem etant la facade des bindings.

#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include <pops/mesh/execution/for_each.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/program/amr_program_checkpoint.hpp>

#include "amr_tagging_test_authority.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
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

static void install_regrid_state_authorities(AmrSystem& system,
                                             std::initializer_list<const char*> blocks) {
  for (const char* block : blocks) {
    const std::string subject =
        std::string("test://amr-system-contract/block/") + block + "/state/U";
    const std::string prefix =
        std::string("test://amr-system-contract/block/") + block + "/transfer/";
    system.install_block_state_route(block, subject);
    system.register_bootstrap_transfer_route(
        prefix + "prolongation", {subject}, "test::amr-system-contract-transfer@1", "cell", "cell",
        "conservative", "dense", "prolongation", "conservative_linear", 2, {1}, 2, kAmrRefRatio);
    system.register_bootstrap_transfer_route(
        prefix + "restriction", {subject}, "test::amr-system-contract-transfer@1", "cell", "cell",
        "conservative", "dense", "restriction", "volume_average", 1, {0}, 2, kAmrRefRatio);
    system.register_bootstrap_transfer_route(prefix + "coarse-fine", {subject},
                                             "test::amr-system-contract-transfer@1", "cell", "cell",
                                             "conservative", "dense", "coarse_fine_fill",
                                             "conservative_coarse_fine", 2, {2}, 2, kAmrRefRatio);
    system.register_bootstrap_transfer_route(prefix + "temporal", {subject},
                                             "test::amr-system-contract-transfer@1", "cell", "cell",
                                             "conservative", "dense", "temporal_interpolation",
                                             "linear_time_interpolation", 2, {0}, 2, kAmrRefRatio);
    system.bind_bootstrap_block_subject(subject, block);
  }
}

TEST(test_amr_system_contract, RefusesMappedPeriodicityBeforeAmrFillPatchConstruction) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  auto boundary = prepare_hyperbolic_boundary<2>(
      {"periodic", "foextrap", "foextrap", "periodic"}, std::vector<double>(4, 0.0),
      {"case::block::tracer::xlo", "case::block::tracer::xhi", "case::block::tracer::ylo",
       "case::block::tracer::yhi"},
      {"Scalar"}, true);
  EXPECT_THROW((void)boundary.periodic_axes(), std::logic_error);
}

TEST(test_amr_system_contract, Runs) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  AmrSystemConfig cfg;
  cfg.n = 16;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};

  // Every facade temporal entry refuses before the lazy hierarchy, a transaction snapshot, a field
  // solve or either clock can be touched. Even advance(..., 0) is a temporal request and therefore
  // requires the same explicit Program authority.
  {
    AmrSystem missing_program(cfg);
    missing_program.add_block("ne", exb_spec(), "none", "rusanov", "conservative", "euler", 1);
    ASSERT_EQ(missing_program.engine(), nullptr);
    const auto expect_program_required = [&](auto&& operation, const char* name) {
      try {
        operation();
        ADD_FAILURE() << name << " accepted a program-less temporal operation";
      } catch (const std::logic_error& error) {
        EXPECT_NE(std::string(error.what()).find(name), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("installed whole-system Program"),
                  std::string::npos);
      }
      EXPECT_EQ(missing_program.engine(), nullptr);
      EXPECT_DOUBLE_EQ(missing_program.time(), 0.0);
      EXPECT_EQ(missing_program.macro_step(), 0);
      EXPECT_FALSE(missing_program.has_active_step_transaction());
    };
    expect_program_required([&] { missing_program.step(0.01); }, "AmrSystem::step");
    expect_program_required([&] { missing_program.advance(0.01, 0); }, "AmrSystem::advance");
    expect_program_required([&] { (void)missing_program.step_cfl(0.4); }, "AmrSystem::step_cfl");
  }

  // The facade topology must reach the native level operator axis by axis. This x-periodic/y-open
  // run wraps only x and fills y physical ghosts by Foextrap; neither axis may inherit the other.
  {
    AmrSystemConfig physical_cfg = cfg;
    physical_cfg.n = 8;
    physical_cfg.regrid_every = 0;
    physical_cfg.periodicity = {true, false};
    AmrSystem physical(physical_cfg);
    physical.set_temporal_relations({2}, {1}, {"integral_only"});
    physical.add_block("left", exb_spec(), "none", "rusanov", "conservative", "euler", 1);
    physical.add_block("right", exb_spec(), "none", "rusanov", "conservative", "euler", 1);
    (void)physical.mass("left");
    ASSERT_TRUE(physical.uses_runtime_engine());
    AmrRuntime* runtime = physical.engine();
    ASSERT_NE(runtime, nullptr);
    EXPECT_TRUE(runtime->base_periodicity().x);
    EXPECT_FALSE(runtime->base_periodicity().y);

    MultiFab& state = runtime->level_state(0, 0);
    state.set_val(Real(-999));
    state.sync_host();
    const Box2D domain = runtime->level_geom(0).domain;
    for (int local = 0; local < state.local_size(); ++local) {
      Array4 values = state.fab(local).array();
      const Box2D valid = state.box(local);
      for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
          values(i, j, 0) = Real(i + 10 * j + 1);
    }
    state.sync_device();
    MultiFab rhs = runtime->level_scalar_field(0, state.ncomp(), 0);
    runtime->level_rhs_into(0, 0, state, rhs);
    state.sync_host();
    ASSERT_EQ(state.local_size(), 1);
    const ConstArray4 values = state.fab(0).const_array();
    const int sample_j = domain.lo[1] + 2;
    EXPECT_EQ(values(domain.lo[0] - 1, sample_j, 0), values(domain.hi[0], sample_j, 0));
    EXPECT_NE(values(domain.lo[0] - 1, sample_j, 0), values(domain.lo[0], sample_j, 0));
    const int sample_i = domain.lo[0] + 2;
    EXPECT_EQ(values(sample_i, domain.lo[1] - 1, 0), values(sample_i, domain.lo[1], 0));
    EXPECT_NE(values(sample_i, domain.lo[1] - 1, 0), values(sample_i, domain.hi[1], 0));
  }

  // The native hierarchy carries Cartesian axes independently: logical extents, physical bounds,
  // cell measures and dense row-major buffers must all agree on (ny, nx), not a square proxy.
  {
    AmrSystemConfig rectangular_cfg = cfg;
    rectangular_cfg.n = 12;
    rectangular_cfg.ny = 8;
    rectangular_cfg.L = 6.0;
    rectangular_cfg.Ly = 2.0;
    rectangular_cfg.xlo = -1.5;
    rectangular_cfg.ylo = 3.0;
    rectangular_cfg.regrid_every = 0;
    rectangular_cfg.periodicity = {true, true};
    AmrSystem rectangular(rectangular_cfg);
    rectangular.set_temporal_relations({2}, {1}, {"integral_only"});
    rectangular.add_block("left", exb_spec(), "none", "rusanov", "conservative", "euler", 1);
    rectangular.add_block("right", exb_spec(), "none", "rusanov", "conservative", "euler", 1);
    const std::size_t cells = static_cast<std::size_t>(rectangular_cfg.n) * rectangular_cfg.ny;
    rectangular.set_density("left", std::vector<double>(cells, 1.0));
    rectangular.set_density("right", std::vector<double>(cells, 0.0));

    EXPECT_DOUBLE_EQ(rectangular.mass("left"), 12.0);
    EXPECT_EQ(rectangular.nx(), 12);
    EXPECT_EQ(rectangular.ny(), 8);
    EXPECT_EQ(rectangular.density("left").size(), cells);
    ASSERT_TRUE(rectangular.uses_runtime_engine());
    const Geometry& geometry = rectangular.engine()->level_geom(0);
    EXPECT_EQ(geometry.domain.nx(), 12);
    EXPECT_EQ(geometry.domain.ny(), 8);
    EXPECT_DOUBLE_EQ(geometry.xlo, -1.5);
    EXPECT_DOUBLE_EQ(geometry.xhi, 4.5);
    EXPECT_DOUBLE_EQ(geometry.ylo, 3.0);
    EXPECT_DOUBLE_EQ(geometry.yhi, 5.0);
    EXPECT_DOUBLE_EQ(geometry.dx(), 0.5);
    EXPECT_DOUBLE_EQ(geometry.dy(), 0.25);
  }

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
  // d'union des tags (set_regrid + graphe prepare cables dans build_multi). ensure_built
  // (1er mass()) construit le moteur avec la cadence active au lieu de lever ; le regrid d'union et
  // le mouvement effectif de la hierarchie sont verrouilles par test_amr_multiblock_regrid_union.
  EXPECT_NO_THROW({
    AmrSystemConfig c2 = cfg;
    c2.regrid_every = 5;  // > 0
    AmrSystem s(c2);
    install_regrid_state_authorities(s, {"ne", "ni"});
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
    install_regrid_state_authorities(s, {"ne"});
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
      // Request the deterministic central fine seed through the prepared tagging authority.
      test::install_prepared_threshold_union(s, {{"magnetic_0", "rho", 1e29}});
      test::install_forward_euler_program(s);
      s.advance(0.01, 1);
      std::vector<std::vector<std::vector<double>>> states;
      states.reserve(static_cast<std::size_t>(block_count));
      EXPECT_EQ(s.n_levels(), 2) << "B_z Program proof must exercise the refined hierarchy";
      for (int block = 0; block < block_count; ++block) {
        std::vector<std::vector<double>> levels;
        levels.reserve(static_cast<std::size_t>(s.n_levels()));
        for (int level = 0; level < s.n_levels(); ++level)
          levels.push_back(s.block_level_state_global("magnetic_" + std::to_string(block), level));
        states.push_back(std::move(levels));
      }
      return states;
    };

    const auto without_field = run(0.0);
    const auto with_field = run(2.0);
    for (int block = 0; block < block_count; ++block) {
      const auto& baseline_levels = without_field[static_cast<std::size_t>(block)];
      const auto& actual_levels = with_field[static_cast<std::size_t>(block)];
      ASSERT_EQ(actual_levels.size(), baseline_levels.size());
      ASSERT_EQ(actual_levels.size(), 2u);
      for (std::size_t level = 0; level < actual_levels.size(); ++level) {
        const auto& baseline = baseline_levels[level];
        const auto& actual = actual_levels[level];
        ASSERT_EQ(actual.size(), baseline.size());
        ASSERT_FALSE(actual.empty()) << "the B_z Program proof requires an owned level state";
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
        EXPECT_GT(max_delta, 1e-3)
            << "B_z must change the native block trajectory at level " << level;
        EXPECT_LT(transverse_delta, -1e-3)
            << "positive B_z must rotate +m_x toward negative m_y at level " << level;
      }
    }
  }
}

TEST(test_amr_system_contract, PrimitiveFixedStateUsesTheConcreteAmrBlockModelConversion) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  AmrSystemConfig cfg;
  cfg.n = 4;
  cfg.L = 1.0;
  cfg.regrid_every = 0;
  cfg.periodicity = {false, false};
  AmrSystem system(cfg);
  const std::string state_identity = "case::block::fluid::state::U";
  system.install_block_state_route("fluid", state_identity);
  std::vector<double> face_values;
  for (const double primitive : {2.0, 3.0, -1.0})
    face_values.insert(face_values.end(), {0.0, primitive, 0.0, 0.0});
  system.install_boundary_plan("fluid", "case::block::fluid::boundary", 2,
                               {"foextrap", "dirichlet", "foextrap", "foextrap"}, face_values,
                               {"case::block::fluid::xlo", "case::block::fluid::xhi",
                                "case::block::fluid::ylo", "case::block::fluid::yhi"},
                               {"Density", "MomentumX", "MomentumY"}, {}, state_identity, {}, {},
                               {"conservative", "primitive", "conservative", "conservative"},
                               {"", "case::block::fluid::model-p2c", "", ""});
  system.add_block("fluid", magnetic_fluid_spec(), "minmod", "rusanov", "conservative", "explicit",
                   1);
  (void)system.mass("fluid");

  AmrRuntime* runtime = system.engine();
  ASSERT_NE(runtime, nullptr);
  MultiFab& state = runtime->level_state(0, 0);
  for (int local = 0; local < state.local_size(); ++local) {
    const Array4 values = state.fab(local).array();
    for_each_cell(state.box(local), [=](int i, int j) {
      for (int component = 0; component < 3; ++component)
        values(i, j, component) = Real(1);
    });
  }
  device_fence();
  MultiFab rhs = runtime->level_scalar_field(0, state.ncomp(), 0);
  runtime->level_rhs_into(0, 0, state, rhs);
  device_fence();
  state.sync_host();

  const Box2D domain = runtime->level_geom(0).domain;
  bool observed = false;
  for (int local = 0; local < state.local_size(); ++local) {
    const Fab2D& values = state.fab(local);
    if (!values.grown_box().contains(domain.hi[0] + 1, 2))
      continue;
    observed = true;
    EXPECT_EQ(values(domain.hi[0] + 1, 2, 0), Real(3));
    EXPECT_EQ(values(domain.hi[0] + 1, 2, 1), Real(11));
    EXPECT_EQ(values(domain.hi[0] + 1, 2, 2), Real(-5));
  }
  EXPECT_TRUE(observed);
}

TEST(test_amr_system_contract, VariableDtStrideUsesOneExactPublicWindow) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  AmrSystemConfig cfg;
  cfg.n = 4;
  cfg.L = 1.0;
  cfg.regrid_every = 0;
  cfg.periodicity = {true, true};

  AmrSystem system(cfg);
  system.add_block("tracer", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
  std::vector<double> times;
  std::vector<double> steps;
  std::vector<int> macro_steps;
  system.install_program_step([&](double h) {
    times.push_back(system.time());
    steps.push_back(h);
    macro_steps.push_back(system.macro_step());
  });
  system.set_program_cadence(/*substeps=*/3, /*stride=*/2);

  system.step(0.1);
  EXPECT_TRUE(times.empty());
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.1);
  EXPECT_EQ(system.program_cadence_window_steps(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.0);

  system.step(0.2);
  ASSERT_EQ(times.size(), 3u);
  EXPECT_NEAR(times[0], 0.0, 1.0e-14);
  EXPECT_NEAR(times[1], 0.1, 1.0e-14);
  EXPECT_NEAR(times[2], 0.2, 1.0e-14);
  ASSERT_EQ(steps.size(), 3u);
  for (const double h : steps)
    EXPECT_NEAR(h, 0.1, 1.0e-14);
  EXPECT_DOUBLE_EQ(times.back() + steps.back(), 0.1 + 0.2);
  EXPECT_EQ(macro_steps, (std::vector<int>{0, 0, 0}));
  EXPECT_NEAR(system.time(), 0.3, 1.0e-14);
  EXPECT_EQ(system.macro_step(), 2);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.0);
}

TEST(test_amr_system_contract, StrideHeldStepPublishesTheExactZeroBalance) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  AmrSystemConfig cfg;
  cfg.n = 4;
  cfg.L = 1.0;
  cfg.regrid_every = 0;
  cfg.periodicity = {true, true};

  AmrSystem system(cfg);
  system.add_block("tracer", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
  system.install_program_step([](double) {});
  system.set_program_cadence(/*substeps=*/1, /*stride=*/2);
  system.begin_step_transaction();
  system.step(0.1);

  const std::string route = "pops.balance-ledger-route.v1:sha256:" + std::string(64, '8');
  const auto balance = system.accepted_balance_terms(route);
  EXPECT_EQ(balance.size(), 5u);
  for (const auto& [name, value] : balance) {
    EXPECT_FALSE(name.empty());
    EXPECT_DOUBLE_EQ(value, 0.0);
  }
  system.commit_step_transaction();
  system.finalize_step_transaction();

  system.begin_step_transaction();
  system.step(0.1);
  system.rollback_step_transaction();
  system.begin_step_transaction();
  const auto restored = system.accepted_balance_terms(route);
  EXPECT_EQ(restored.size(), 5u);
  for (const auto& [name, value] : restored) {
    EXPECT_FALSE(name.empty());
    EXPECT_DOUBLE_EQ(value, 0.0);
  }
  system.rollback_step_transaction();
}

TEST(test_amr_system_contract, CadenceRestoreRejectsClockDriftWithoutMutatingAcceptedState) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  AmrSystemConfig cfg;
  cfg.n = 4;
  cfg.L = 1.0;
  cfg.regrid_every = 0;
  cfg.periodicity = {true, true};

  AmrSystem system(cfg);
  system.set_program_cadence(/*substeps=*/1, /*stride=*/2);
  system.restore_program_cadence_window(/*accumulated_dt=*/0.1, /*held_steps=*/1,
                                        /*window_start_time=*/0.0, /*accepted_last_dt=*/0.075,
                                        /*accepted_time=*/0.1,
                                        /*macro_step=*/1);

  // The candidate stays staged until set_clock authenticates the exact accepted pair.
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);
  EXPECT_DOUBLE_EQ(system.program_last_dt(), 0.0);
  EXPECT_THROW(system.set_clock(std::nextafter(0.1, 1.0), /*macro_step=*/1), std::runtime_error);
  EXPECT_DOUBLE_EQ(system.time(), 0.0);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);

  // The failed clock consumed the staged token; the exact retry can stage and commit cleanly.
  system.restore_program_cadence_window(/*accumulated_dt=*/0.1, /*held_steps=*/1,
                                        /*window_start_time=*/0.0, /*accepted_last_dt=*/0.075,
                                        /*accepted_time=*/0.1,
                                        /*macro_step=*/1);
  system.set_clock(/*t=*/0.1, /*macro_step=*/1);
  EXPECT_DOUBLE_EQ(system.time(), 0.1);
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.1);
  EXPECT_EQ(system.program_cadence_window_steps(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.0);
  EXPECT_DOUBLE_EQ(system.program_last_dt(), 0.075);
}

TEST(test_amr_system_contract,
     NonAssociativeCadenceClosesTheSerializedAcceptedClockAtTheFacadeEndpoint) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  AmrSystemConfig cfg;
  cfg.n = 4;
  cfg.L = 1.0;
  cfg.regrid_every = 0;
  cfg.periodicity = {true, true};

  AmrSystem system(cfg);
  system.add_block("tracer", exb_spec(), "none", "rusanov", "conservative", "explicit", 1);
  system.set_clock(0.1, 0);
  test::install_forward_euler_program(system);
  system.set_program_cadence(/*substeps=*/3, /*stride=*/3);

  const double after_first = 0.1 + 0.1;
  const double after_second = after_first + 0.1;
  const double accepted_endpoint = after_second + 0.3;
  const double reconstructed_endpoint = 0.1 + ((0.1 + 0.1) + 0.3);
  ASSERT_NE(std::bit_cast<std::uint64_t>(accepted_endpoint),
            std::bit_cast<std::uint64_t>(reconstructed_endpoint))
      << "fixture must exercise floating-point non-associativity";

  system.step(0.1);
  system.step(0.1);
  system.step(0.3);

  EXPECT_EQ(std::bit_cast<std::uint64_t>(system.time()),
            std::bit_cast<std::uint64_t>(accepted_endpoint));
  runtime::program::AmrProgramAcceptedState<2> accepted;
  accepted.spatial_contract = "test.non-associative-accepted-clock.dim2";
  accepted.level_clocks = {
      {0, system.macro_step(), amr::Rational(0, 1), system.time()},
  };
  const auto encoded = runtime::program::serialize_amr_program_accepted_state(accepted);
  const auto decoded = runtime::program::deserialize_amr_program_accepted_state<2>(encoded);
  EXPECT_EQ(runtime::program::serialize_amr_program_accepted_state(decoded), encoded);
  ASSERT_FALSE(decoded.level_clocks.empty());
  for (const auto& clock : decoded.level_clocks) {
    EXPECT_EQ(std::bit_cast<std::uint64_t>(clock.physical_time),
              std::bit_cast<std::uint64_t>(system.time()));
    EXPECT_EQ(clock.macro_step, system.macro_step());
  }
}
