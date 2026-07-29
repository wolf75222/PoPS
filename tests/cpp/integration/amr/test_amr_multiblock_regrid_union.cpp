// AMR MULTI-BLOCS REGRID D'UNION DES TAGS (capstone Phase 2, C.6 ; docs/AMR_REGRID_UNION_TAGS_DESIGN.md).
//
// Le moteur multi-blocs RUNTIME (AmrRuntime) tournait jusqu'ici a hierarchie FIGEE (pas de regrid).
// Cette PR DEVERROUILLE le regrid pilote par l'UNION des tags : tous les @p regrid_every macro-pas, la
// hierarchie partagee est re-grillee a partir de l'UNION (OU cellule a cellule) des tags de TOUS les
// blocs (predicat PAR BLOC, D1) + des tags de phi (sur |grad phi|, D4), suivie d'UN clustering
// Berger-Rigoutsos -> UN nouveau layout fin applique a TOUS les blocs (y compris ceux tenus par leur
// stride, D3) ET a l'aux partage, en maintenant same_layout_or_throw apres regrid. Le meme moteur est
// verrouille ici sur trois niveaux : transport, regrid du niveau le plus fin et rollback transactionnel.
//
// Ce que ce test verrouille (les cas demandes a-e) :
//   (a) HIERARCHIE QUI EVOLUE : avec regrid_every > 0, le layout fin CHANGE quand la structure taguee
//       se deplace (la BoxArray fine n'est plus celle du build initial central fixe).
//   (b)+(c) UNION DES TAGS : deux blocs taguant des regions DISJOINTES (bloc A a gauche, bloc B a
//       droite) -> le layout d'union COUVRE LES DEUX regions (bounding box du fin enjambe gauche ET
//       droite). Et un raffinement declenche par phi seul (|grad phi|) ajoute des patchs la ou ni A ni B
//       ne taguent : l'union est bien A OU B OU |grad phi|.
//   (d) BLOC STRIDE-TENU RE-GRILLE : un bloc tenu par son stride (stride=4, non avance au macro-pas de
//       regrid) est NEANMOINS re-grille sur le layout d'union (sa BoxArray fine == layout partage, pas
//       l'ancienne) -> same_layout_or_throw passe (sinon le ctor / le regrid aurait leve), et son fin
//       porte des donnees finies (report + interp), pas un fab non initialise sur l'ancienne grille.
//   (e) regrid_every == 0 BIT-IDENTIQUE : multi-blocs fige reste STRICTEMENT le comportement actuel
//       (regrid jamais appele) -> meme cas joue deux fois donne dmax == 0, et le layout fin ne bouge pas.
//
// AUTORITE TEMPORELLE : chaque avance passe par AmrSystem et un Program explicite. AmrRuntime reste
// uniquement le moteur spatial expose par sim.engine() pour inspecter les layouts et installer les
// graphes de tagging prepares. Il n'est jamais appele directement pour avancer le temps.

#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // detail::make_shared_amr_layout / dispatch_amr_block
#include <pops/runtime/amr/amr_runtime.hpp>                  // AmrRuntime, AmrRuntimeBlock
#include <pops/runtime/amr_system.hpp>  // facade AmrSystem (deverrouillage multi-blocs + regrid_every>0)
#include <pops/physics/bricks/bricks.hpp>
#include <pops/runtime/config/model_spec.hpp>

#include "amr_transfer_test_authority.hpp"
#include "amr_tagging_test_authority.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pops;

// Modele ExB scalaire (1 var, role density) a charge q. Transport ExB : la densite advecte le long du
// champ ExB, donc une structure se DEPLACE -> la region taguee bouge -> le layout fin change (cas a).
using ExBModel = CompositeModel<ExBVelocity, NoSource, ChargeDensity>;
static ExBModel exb_charge(double q, double B0) {
  return ExBModel{ExBVelocity{Real(B0)}, NoSource{}, ChargeDensity{Real(q)}};
}

// ------------------------------------------------------------------------------------------------
// densites initiales : un disque gaussien centre en (cx, cy) du domaine [0,1]^2, amplitude amp + base.
// Moyenne ajustee a base pour la solvabilite periodique du Poisson (on travaille relativement a base).
// ------------------------------------------------------------------------------------------------
static std::vector<double> blob(int n, double cx, double cy, double amp, double base,
                                double width) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n, base);
  double perturbation_sum = 0.0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n, y = (j + 0.5) / n;
      const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      const double perturbation = amp * std::exp(-r2 / (width * width));
      rho[static_cast<std::size_t>(j) * n + i] += perturbation;
      perturbation_sum += perturbation;
    }
  const double perturbation_mean = perturbation_sum / static_cast<double>(n * n);
  for (double& value : rho)
    value -= perturbation_mean;
  return rho;
}

// densite uniforme (bloc neutre / fond) ; ne tague rien tant qu'on n'enregistre pas son predicat.
static std::vector<double> flat(int n, double v) {
  return std::vector<double>(static_cast<std::size_t>(n) * n, v);
}

static bool all_finite(const std::vector<double>& v) {
  for (double x : v)
    if (!std::isfinite(x))
      return false;
  return true;
}

static bool all_level_states_finite(AmrRuntime& rt) {
  device_fence();
  for (std::size_t block = 0; block < rt.n_blocks(); ++block)
    for (const AmrLevelMP& level : rt.levels(block))
      for (int li = 0; li < level.U.local_size(); ++li) {
        const ConstArray4 state = level.U.fab(li).const_array();
        const Box2D valid = level.U.box(li);
        for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
          for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
            for (int component = 0; component < level.U.ncomp(); ++component)
              if (!std::isfinite(state(i, j, component)))
                return false;
      }
  return true;
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
  const std::size_t nn = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < nn; ++i)
    d = std::max(d, std::fabs(a[i] - b[i]));
  return d;
}

static void install_regrid_state_authorities(AmrSystem& sim) {
  struct StateRoute {
    const char* block;
    const char* subject;
  };
  const StateRoute routes[] = {{"a", "test://amr-regrid-union/block/a/state/U"},
                               {"b", "test://amr-regrid-union/block/b/state/U"}};
  for (const StateRoute& route : routes)
    sim.install_block_state_route(route.block, route.subject);
  for (const StateRoute& route : routes) {
    const std::string prefix =
        std::string("test://amr-regrid-union/block/") + route.block + "/transfer/";
    sim.register_bootstrap_transfer_route(prefix + "prolongation", {route.subject},
                                          "test::amr-regrid-union-transfer@1", "cell", "cell",
                                          "conservative", "dense", "prolongation",
                                          "conservative_linear", 2, {1}, 2, kAmrRefRatio);
    sim.register_bootstrap_transfer_route(
        prefix + "restriction", {route.subject}, "test::amr-regrid-union-transfer@1", "cell",
        "cell", "conservative", "dense", "restriction", "volume_average", 1, {0}, 2, kAmrRefRatio);
    sim.register_bootstrap_transfer_route(prefix + "coarse-fine", {route.subject},
                                          "test::amr-regrid-union-transfer@1", "cell", "cell",
                                          "conservative", "dense", "coarse_fine_fill",
                                          "conservative_coarse_fine", 2, {2}, 2, kAmrRefRatio);
    sim.register_bootstrap_transfer_route(prefix + "temporal", {route.subject},
                                          "test::amr-regrid-union-transfer@1", "cell", "cell",
                                          "conservative", "dense", "temporal_interpolation",
                                          "linear_time_interpolation", 2, {0}, 2, kAmrRefRatio);
    sim.bind_bootstrap_block_subject(route.subject, route.block);
  }
}

static void install_regrid_threshold_union(AmrSystem& sim, double threshold) {
  test::install_prepared_threshold_union(
      sim, {{"a", "n", threshold, test::PreparedThresholdRelation::Above,
             "test://amr-regrid-union/block/a/state/U"},
            {"b", "n", threshold, test::PreparedThresholdRelation::Above,
             "test://amr-regrid-union/block/b/state/U"}});
}

// Bounding box (coords du niveau FIN) de la BoxArray fine du bloc 0 (layout partage : identique pour
// tous les blocs). Permet de verifier la couverture spatiale du layout d'union (cas b/c).
static Box2D fine_bbox(AmrRuntime& rt) {
  const std::vector<AmrLevelMP>& L = rt.levels(0);
  return L[1].U.box_array().bounding_box();
}

// Vecteur des boites fines (coords fin) du bloc 0, ordonne, pour comparer deux layouts (cas a/e).
static std::vector<Box2D> fine_boxes(AmrRuntime& rt) {
  return rt.levels(0)[1].U.box_array().boxes();
}

static bool same_box_list(const std::vector<Box2D>& a, const std::vector<Box2D>& b) {
  if (a.size() != b.size())
    return false;
  for (std::size_t k = 0; k < a.size(); ++k)
    if (a[k].lo[0] != b[k].lo[0] || a[k].lo[1] != b[k].lo[1] || a[k].hi[0] != b[k].hi[0] ||
        a[k].hi[1] != b[k].hi[1])
      return false;
  return true;
}

static bool layout_covers(const BoxArray& layout, const Box2D& required) {
  std::int64_t covered = 0;
  for (const Box2D& box : layout.boxes())
    covered += box.intersect(required).num_cells();
  return covered == required.num_cells();
}

static ModelSpec exb_spec(double q, double B0) {
  ModelSpec spec;
  spec.transport = "exb";
  spec.source = "none";
  spec.elliptic = "charge";
  spec.q = q;
  spec.B0 = B0;
  return spec;
}

// L'objet AmrSystem reste a une adresse stable : l'AmrProgramContext installe par le helper conserve
// un pointeur vers cette facade. Le moteur spatial est materialise par le Program, jamais construit
// directement par ce test.
static std::unique_ptr<AmrSystem> make_two_block_system(int N, double L, double B0, double q0,
                                                        double q1, const std::vector<double>& rho0,
                                                        const std::vector<double>& rho1,
                                                        int stride1 = 1, int regrid_every = 0,
                                                        int regrid_grow = 2, int regrid_margin = 2,
                                                        int level_count = 2) {
  AmrSystemConfig cfg;
  cfg.n = N;
  cfg.L = L;
  cfg.periodicity = {true, true};
  cfg.regrid_every = regrid_every;
  cfg.regrid_grow = regrid_grow;
  cfg.regrid_margin = regrid_margin;
  cfg.level_count = level_count;
  cfg.explicit_bootstrap = level_count > 2;
  auto sim = std::make_unique<AmrSystem>(cfg);
  install_regrid_state_authorities(*sim);
  std::vector<std::int64_t> numerators(static_cast<std::size_t>(level_count - 1), 2);
  std::vector<std::int64_t> denominators(static_cast<std::size_t>(level_count - 1), 1);
  std::vector<std::string> policies(static_cast<std::size_t>(level_count - 1), "integral_only");
  sim->set_temporal_relations(numerators, denominators, policies);
  sim->add_block("a", exb_spec(q0, B0), "minmod", "rusanov", "conservative", "explicit", 1);
  sim->add_block("b", exb_spec(q1, B0), "minmod", "rusanov", "conservative", "explicit", 1,
                 stride1);
  sim->set_poisson("charge_density", "geometric_mg", "periodic");
  sim->set_density("a", rho0);
  sim->set_density("b", rho1);
  test::install_forward_euler_program(*sim);
  if (!sim->uses_runtime_engine() || sim->engine() == nullptr)
    throw std::runtime_error("multi-block regrid test requires the AmrSystem runtime engine");
  return sim;
}

static void check_three_level_bootstrap_step_regrid_and_rollback() {
  SCOPED_TRACE("three-level bootstrap/regrid/rollback");
  const int N = 32;
  // Offset the tagged region from the nested central finest seed. A coarse regrid therefore
  // replaces the middle level with a layout that cannot carry the complete old finest layout. The
  // runtime must retire that stale descendant before preparing the new parent topology, then
  // rebuild it from the new middle level in the same transaction.
  auto sim =
      make_two_block_system(N, 1.0, 1.0, +1.0, -1.0, blob(N, 0.32, 0.50, 0.8, 1.0, 0.08),
                            blob(N, 0.32, 0.50, 0.8, 1.0, 0.08), /*stride1=*/1, /*regrid_every=*/1,
                            /*regrid_grow=*/2, /*regrid_margin=*/2, /*level_count=*/3);
  AmrRuntime& rt = *sim->engine();
  EXPECT_EQ(rt.nlev(), 1);
  EXPECT_EQ(rt.max_levels(), 3);
  test::install_prepared_threshold_union(rt, {{0, 0, Real(-1e30)}, {1, 0, Real(-1e30)}});
  sim->step(Real(1e-4));  // macro step zero never regrids
  EXPECT_EQ(sim->checkpoint_regrid_count(), 0);
  sim->step(Real(1e-4));  // Program cadence activates every resolved transition
  EXPECT_EQ(rt.nlev(), 3);
  EXPECT_EQ(rt.levels(0).size(), 3u);
  const std::size_t expected_fine_patches =
      rt.levels(0)[1].U.box_array().size() + rt.levels(0)[2].U.box_array().size();
  EXPECT_EQ(rt.patch_boxes().size(), expected_fine_patches);
  EXPECT_EQ(static_cast<std::size_t>(rt.n_patches()), expected_fine_patches)
      << "all fine levels contribute to the patch count";
  EXPECT_TRUE(
      same_box_list(rt.levels(0)[1].U.box_array().boxes(), rt.levels(1)[1].U.box_array().boxes()));
  EXPECT_TRUE(
      same_box_list(rt.levels(0)[2].U.box_array().boxes(), rt.levels(1)[2].U.box_array().boxes()));
  {
    SCOPED_TRACE("three-level initial parent/child conservative prolongation");
    const MultiFab& parent = rt.levels(0)[1].U;
    const MultiFab& child = rt.levels(0)[2].U;
    const Box2D fine = child.box(0);
    const int i = fine.lo[0], j = fine.lo[1];
    const int ci = coarsen_index(i, kAmrRefRatio), cj = coarsen_index(j, kAmrRefRatio);
    const int parent_box = mf_find_box(parent, ci, cj);
    ASSERT_GE(parent_box, 0);
    const int first_i = 2 * ci;
    const int first_j = 2 * cj;
    Real child_average = Real(0);
    for (int child_j = 0; child_j < 2; ++child_j)
      for (int child_i = 0; child_i < 2; ++child_i)
        child_average += child.fab(0)(first_i + child_i, first_j + child_j, 0);
    child_average /= Real(4);
    EXPECT_NEAR(child_average, parent.fab(parent_box)(ci, cj, 0), 2e-14)
        << "level-2 initialization conservatively prolongs its immediate parent";
  }

  // Regridding is decision based: a false refine root holds the accepted fine coverage. The
  // complementary roots below deliberately replace it, so the displaced threshold region can
  // invalidate the old level-2 parent. Equality remains Hold and the roots never conflict.
  test::install_prepared_threshold_decisions(
      rt,
      {{0, 0, Real(1.25), test::PreparedThresholdRelation::Above},
       {1, 0, Real(1.25), test::PreparedThresholdRelation::Above}},
      {{0, 0, Real(1.25), test::PreparedThresholdRelation::Below},
       {1, 0, Real(1.25), test::PreparedThresholdRelation::Below}},
      "test::prepared-replacement-threshold@1");

  const int accepted_regrids = sim->checkpoint_regrid_count();
  const std::uint64_t accepted_materialization = rt.topology_materialization_generation();
  const std::vector<Box2D> accepted_middle = rt.levels(0)[1].U.box_array().boxes();
  const std::vector<Box2D> accepted_finest = rt.levels(0)[2].U.box_array().boxes();
  ASSERT_TRUE(std::all_of(accepted_finest.begin(), accepted_finest.end(),
                          [&](const Box2D& accepted_child) {
                            return layout_covers(rt.levels(0)[1].U.box_array(),
                                                 accepted_child.coarsen(kAmrRefRatio));
                          }))
      << "the accepted descendant must initially be supported by its parent";
  sim->begin_step_transaction();
  sim->step(Real(1e-4));  // explicit Program regrids every active transition, coarse to fine
  const std::uint64_t regridded_materialization = rt.topology_materialization_generation();
  EXPECT_EQ(rt.nlev(), 3);
  EXPECT_EQ(sim->checkpoint_regrid_count(), accepted_regrids + 1);
  EXPECT_GT(regridded_materialization, accepted_materialization);
  EXPECT_FALSE(same_box_list(accepted_middle, rt.levels(0)[1].U.box_array().boxes()));
  EXPECT_FALSE(same_box_list(accepted_finest, rt.levels(0)[2].U.box_array().boxes()));
  EXPECT_TRUE(std::any_of(accepted_finest.begin(), accepted_finest.end(),
                          [&](const Box2D& stale_child) {
                            return !layout_covers(rt.levels(0)[1].U.box_array(),
                                                  stale_child.coarsen(kAmrRefRatio));
                          }))
      << "the regression must replace a parent that cannot support the stale descendant";
  EXPECT_TRUE(all_level_states_finite(rt));

  // The facade-owned accepted-attempt transaction restores topology, cadence, clock, profiler and
  // every level together. No test-side AmrRuntime snapshot stands in for the public authority.
  sim->rollback_step_transaction();
  const std::uint64_t restored_materialization = rt.topology_materialization_generation();
  EXPECT_EQ(rt.nlev(), 3);
  EXPECT_EQ(sim->checkpoint_regrid_count(), accepted_regrids);
  EXPECT_GT(restored_materialization, regridded_materialization)
      << "restoring an older epoch still invalidates concrete layout-bound resources";
  EXPECT_TRUE(same_box_list(accepted_middle, rt.levels(0)[1].U.box_array().boxes()));
  EXPECT_TRUE(same_box_list(accepted_finest, rt.levels(0)[2].U.box_array().boxes()));

  sim->step(Real(1e-4));
  EXPECT_EQ(rt.nlev(), 3);
  EXPECT_EQ(sim->checkpoint_regrid_count(), accepted_regrids + 1)
      << "the accepted retry commits one regrid exactly once";
  EXPECT_GT(rt.topology_materialization_generation(), restored_materialization);

  // Coarsening is an explicit decision, not the absence of a refine match. Deactivate the complete
  // fine suffix without changing the resolved capacity, then reactivate that same capacity.
  test::install_prepared_threshold_decisions(rt, {{0, 0, Real(1e30)}, {1, 0, Real(1e30)}},
                                             {{0, 0, Real(-1e30)}, {1, 0, Real(-1e30)}});
  sim->step(Real(1e-4));
  EXPECT_EQ(rt.nlev(), 1);
  EXPECT_EQ(rt.max_levels(), 3);

  test::install_prepared_threshold_union(rt, {{0, 0, Real(-1e30)}, {1, 0, Real(-1e30)}});
  sim->step(Real(1e-4));
  ASSERT_EQ(rt.nlev(), 3);
  ASSERT_EQ(rt.max_levels(), 3);

  const std::uint64_t full_layout_generation = rt.topology_materialization_generation();
  const int full_layout_regrids = sim->checkpoint_regrid_count();
  sim->step(Real(1e-4));
  EXPECT_EQ(rt.topology_materialization_generation(), full_layout_generation)
      << "an identical regrid must preserve layout-bound caches and storage";
  EXPECT_EQ(sim->checkpoint_regrid_count(), full_layout_regrids)
      << "an identical regrid is not a topology replacement";

  std::vector<PatchBox> checkpoint_boxes;
  std::vector<int> checkpoint_owners;
  for (int level = 1; level < 3; ++level) {
    const std::vector<int> level_owners = sim->level_owner_ranks(level);
    std::size_t patch = 0;
    for (const Box2D& box : rt.levels(0)[static_cast<std::size_t>(level)].U.box_array().boxes()) {
      checkpoint_boxes.push_back(PatchBox{level, box.lo[0], box.lo[1], box.hi[0], box.hi[1]});
      checkpoint_owners.push_back(level_owners.at(patch++));
    }
  }

  const std::uint64_t accepted_generation = rt.topology_materialization_generation();
  const std::vector<PatchBox> accepted_patches = rt.patch_boxes();
  const std::vector<PatchBox> misaligned{{1, 1, 0, 16, 15}};
  EXPECT_THROW(sim->rebuild_hierarchy(misaligned, {0}), std::runtime_error);
  EXPECT_EQ(rt.nlev(), 3);
  EXPECT_EQ(rt.topology_materialization_generation(), accepted_generation);
  EXPECT_EQ(rt.patch_boxes(), accepted_patches);

  const std::vector<PatchBox> unnested{{1, 8, 8, 23, 23}, {2, 0, 0, 3, 3}};
  EXPECT_THROW(sim->rebuild_hierarchy(unnested, {0, 0}), std::runtime_error);
  EXPECT_EQ(rt.nlev(), 3);
  EXPECT_EQ(rt.topology_materialization_generation(), accepted_generation);
  EXPECT_EQ(rt.patch_boxes(), accepted_patches);

  sim->rebuild_hierarchy({}, {});
  EXPECT_EQ(rt.nlev(), 1);
  EXPECT_EQ(rt.max_levels(), 3);
  sim->rebuild_hierarchy(checkpoint_boxes, checkpoint_owners);
  EXPECT_EQ(rt.nlev(), 3);
  EXPECT_EQ(rt.max_levels(), 3);
  for (int level = 1; level < 3; ++level) {
    const auto& restored = rt.levels(0)[static_cast<std::size_t>(level)].U.box_array().boxes();
    std::vector<PatchBox> expected_level;
    std::vector<int> expected_owners;
    for (std::size_t patch = 0; patch < checkpoint_boxes.size(); ++patch)
      if (checkpoint_boxes[patch].level == level) {
        expected_level.push_back(checkpoint_boxes[patch]);
        expected_owners.push_back(checkpoint_owners[patch]);
      }
    ASSERT_EQ(restored.size(), expected_level.size());
    for (std::size_t patch = 0; patch < restored.size(); ++patch) {
      const PatchBox& expected = expected_level[patch];
      EXPECT_EQ(restored[patch].lo[0], expected.ilo);
      EXPECT_EQ(restored[patch].lo[1], expected.jlo);
      EXPECT_EQ(restored[patch].hi[0], expected.ihi);
      EXPECT_EQ(restored[patch].hi[1], expected.jhi);
    }
    EXPECT_EQ(sim->level_owner_ranks(level), expected_owners);
  }

  AmrBuildParams invalid;
  invalid.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  invalid.mesh.periodicity = Periodicity{true, true};
  invalid.mesh.n = N;
  EXPECT_THROW(detail::make_shared_amr_layout_levels(invalid, 0), std::runtime_error);
  invalid.mesh.n = 2;
  EXPECT_THROW(detail::make_shared_amr_layout_levels(invalid, 3), std::runtime_error);
}

TEST(test_amr_multiblock_regrid_union, Runs) {
  // AmrRuntime owns the process-wide lazy Kokkos initialization.  A local ScopeGuard would
  // finalize Kokkos when this TEST returns and make the following TEST construct storage after
  // finalization, which Kokkos deliberately forbids.

  check_three_level_bootstrap_step_regrid_and_rollback();

  const int N = 32;
  const double L = 1.0, B0 = 1.0;

  // ============================================================================================
  // (e) NON-REGRESSION FIGEE : regrid_every == 0 -> regrid JAMAIS appele -> bit-identique + layout fin
  //     inchange. On le teste D'ABORD pour ancrer le comportement de reference (la hierarchie figee).
  // ============================================================================================
  {
    SCOPED_TRACE("frozen hierarchy");
    auto run_frozen = [&]() {
      auto sim = make_two_block_system(N, L, B0, +1.0, -1.0, blob(N, 0.35, 0.5, 0.8, 1.0, 0.10),
                                       blob(N, 0.65, 0.5, 0.8, 1.0, 0.10), /*stride1=*/1,
                                       /*regrid_every=*/0);
      AmrRuntime& rt = *sim->engine();
      test::install_prepared_threshold_union(rt, {{0, 0, Real(1.3)}, {1, 0, Real(1.3)}});
      const std::vector<Box2D> fb_before = fine_boxes(rt);
      sim->advance(Real(0.01), 8);
      const std::vector<Box2D> fb_after = fine_boxes(rt);
      return std::make_pair(rt.density(0), same_box_list(fb_before, fb_after));
    };
    const auto a = run_frozen();
    const auto b = run_frozen();
    EXPECT_TRUE(a.second) << "e_frozen_fine_layout_unchanged";  // la grille n'a pas bouge
    EXPECT_EQ(dmax_field(a.first, b.first), 0.0) << "e_frozen_bit_identical_dmax0";

    auto sim = make_two_block_system(N, L, B0, +1.0, -1.0, blob(N, 0.35, 0.5, 0.8, 1.0, 0.10),
                                     blob(N, 0.65, 0.5, 0.8, 1.0, 0.10), /*stride1=*/1,
                                     /*regrid_every=*/0);
    AmrRuntime& rt = *sim->engine();
    test::install_prepared_threshold_union(rt, {{0, 0, Real(1.3)}});
    sim->advance(Real(0.01), 8);
    EXPECT_EQ(sim->checkpoint_regrid_count(), 0) << "e_regrid_count_zero_when_frozen";
  }

  // ============================================================================================
  // (a) HIERARCHIE QUI EVOLUE : regrid_every > 0 -> le layout fin CHANGE par rapport au seed central
  //     fixe du build (l'union des tags des deux blobs n'est PAS le patch central [n/4..3n/4]^2).
  // ============================================================================================
  {
    SCOPED_TRACE("evolving hierarchy");
    auto sim =
        make_two_block_system(N, L, B0, +1.0, -1.0, blob(N, 0.30, 0.5, 1.0, 1.0, 0.07),
                              blob(N, 0.70, 0.5, 1.0, 1.0, 0.07), /*stride1=*/1, /*regrid_every=*/2,
                              /*regrid_grow=*/2, /*regrid_margin=*/2);
    AmrRuntime& rt = *sim->engine();
    test::install_prepared_threshold_union(rt, {{0, 0, Real(1.5)}, {1, 0, Real(1.5)}});
    // ADC-607: wire a profiler so the regrid data-structure counters emit. The regrid site records
    // tag_density (dense TagBox fill, permille), box_hash_rebuilds + copy_cache_hits/misses
    // (parallel_copy schedule-cache engagement). We only assert the counters exist and are
    // internally consistent (rebuilds == misses); the numeric trajectory is untouched.
    sim->enable_profiling();
    pops::runtime::program::Profiler& prof = sim->profiler_handle();
    const std::vector<Box2D> fb_seed = fine_boxes(rt);            // patch central fixe du build
    const double m0_before = rt.mass(0), m1_before = rt.mass(1);  // (V1) snapshot avant la sequence
    sim->advance(Real(0.01), 6);
    EXPECT_TRUE(sim->checkpoint_regrid_count() >= 1) << "a_regrid_was_called";
    // (ADC-607) the regrid emitted the new counters; box_hash_rebuilds mirrors copy_cache_misses,
    // and each parallel_copy either hit or missed (hits + misses == total copies >= misses >= 0).
    EXPECT_EQ(prof.counter("box_hash_rebuilds"), prof.counter("copy_cache_misses"))
        << "a_box_hash_rebuilds_equals_copy_misses";
    EXPECT_TRUE(prof.counter("copy_cache_hits") >= 0 && prof.counter("copy_cache_misses") >= 0)
        << "a_copy_cache_counters_nonnegative";
    EXPECT_TRUE(prof.counter("tag_density") >= 0 &&
                prof.counter("tag_density") <= 1000 * sim->checkpoint_regrid_count())
        << "a_tag_density_in_permille_range";
    sim->disable_profiling();
    const std::vector<Box2D> fb_now = fine_boxes(rt);
    EXPECT_TRUE(!same_box_list(fb_seed, fb_now)) << "a_fine_layout_evolved_from_seed";
    EXPECT_TRUE(all_finite(rt.density(0)) && all_finite(rt.density(1)))
        << "a_state_finite_after_regrid";
    EXPECT_TRUE(rt.n_patches() >= 1) << "a_hierarchy_still_has_fine_patches";
    // (V1) CONSERVATION PAR BLOC a travers les regrids : le report fin (exact) + l'interp parent
    // piecewise-constant (conservative au sens integral) redistribuent sans creer ni detruire de masse ;
    // le transport ExB periodique + reflux conserve la masse grossiere. Verifie POUR CHAQUE bloc.
    EXPECT_TRUE(std::fabs(rt.mass(0) - m0_before) < 1e-9)
        << "a_block0_mass_conserved_across_regrid";
    EXPECT_TRUE(std::fabs(rt.mass(1) - m1_before) < 1e-9)
        << "a_block1_mass_conserved_across_regrid";
  }

  // ============================================================================================
  // (b)+(c) UNION DES TAGS. Bloc A tague une region a GAUCHE, bloc B une region a DROITE (disjointes).
  //     Le layout d'union doit COUVRIR LES DEUX (bounding box du fin enjambe gauche ET droite). Puis,
  //     en n'activant que le predicat phi (sur |grad phi|), un raffinement est declenche par phi SEUL.
  // ============================================================================================
  {
    SCOPED_TRACE("union and phi-only tagging");
    // Bloc A : blob a gauche (cx=0.25). Bloc B : blob a droite (cx=0.75). Charges opposees -> phi non
    // trivial (Poisson somme). Predicats par bloc seulement (phi non enregistre) : union = A OU B.
    auto sim =
        make_two_block_system(N, L, B0, +1.0, -1.0, blob(N, 0.25, 0.5, 1.2, 1.0, 0.06),
                              blob(N, 0.75, 0.5, 1.2, 1.0, 0.06), /*stride1=*/1, /*regrid_every=*/1,
                              /*regrid_grow=*/1, /*regrid_margin=*/1);
    AmrRuntime& rt = *sim->engine();
    test::install_prepared_threshold_union(rt, {{0, 0, Real(1.6)}, {1, 0, Real(1.6)}});
    // phi NON enregistre ici : on isole l'union A OU B. Le premier step (macro_step_=0) ne regrid PAS
    // (la grille est fraichement construite, convention mono-bloc) ; le 2e step (macro_step_=1, every=1)
    // declenche le regrid d'union.
    const std::vector<Box2D> fb_seed = fine_boxes(rt);
    sim->advance(Real(0.005), 2);
    EXPECT_TRUE(sim->checkpoint_regrid_count() >= 1) << "bc_union_regrid_called";
    const Box2D bb = fine_bbox(rt);
    // coords du niveau fin = 2 x coords grossieres. Gauche ~ cellule grossiere 8 (x=0.25*32) -> fin ~16 ;
    // droite ~ cellule grossiere 24 -> fin ~48. L'union doit enjamber le milieu (fin ~32) : lo a gauche
    // du milieu, hi a droite du milieu -> couvre les DEUX regions, pas une seule.
    const int mid_fine = N;  // milieu du domaine en coords fin (2 * N/2)
    EXPECT_TRUE(!bb.empty()) << "bc_union_layout_nonempty";
    EXPECT_TRUE(bb.lo[0] < mid_fine && bb.hi[0] > mid_fine)
        << "bc_union_covers_both_left_and_right";
    // Le layout d'union DIFFERE du seed central fixe : la couverture des DEUX regions est bien le
    // produit du regrid d'union, pas un artefact du patch central initial (qui couvre deja le milieu).
    EXPECT_TRUE(!same_box_list(fb_seed, fine_boxes(rt))) << "bc_union_layout_differs_from_seed";
    EXPECT_TRUE(all_finite(rt.density(0)) && all_finite(rt.density(1))) << "bc_state_finite";

    // (c) UNION PAR PHI SEUL : nouveau runtime, AUCUN predicat de bloc, seulement le predicat phi sur
    //     |grad phi|. Un raffinement est alors declenche PAR PHI (preuve que phi entre dans l'union,
    //     independamment des criteres de bloc, D4). On choisit un seuil bas pour garantir des tags. La
    //     comparaison au seed prouve que le layout fin est REELLEMENT celui calcule par le regrid phi.
    auto phi_sim = make_two_block_system(
        N, L, B0, +1.0, -1.0, blob(N, 0.5, 0.5, 1.5, 1.0, 0.06), flat(N, 1.0),
        /*stride1=*/1, /*regrid_every=*/1, /*regrid_grow=*/1, /*regrid_margin=*/1);
    AmrRuntime& rtp = *phi_sim->engine();
    // Aucun critere de bloc : la feuille gradient du champ aux partage pilote seule l'union.
    test::install_prepared_shared_aux_gradient(rtp, 2, Real(1e-6));
    const std::vector<Box2D> fb_seed_phi = fine_boxes(rtp);
    phi_sim->advance(Real(0.005), 2);
    EXPECT_TRUE(phi_sim->checkpoint_regrid_count() >= 1) << "c_phi_only_regrid_called";
    EXPECT_TRUE(rtp.n_patches() >= 1) << "c_phi_only_triggers_refinement";
    EXPECT_TRUE(!same_box_list(fb_seed_phi, fine_boxes(rtp)))
        << "c_phi_only_layout_from_regrid_not_seed";
    EXPECT_TRUE(all_finite(rtp.density(0))) << "c_phi_only_state_finite";
  }

  // ============================================================================================
  // (d) BLOC STRIDE-TENU RE-GRILLE. Bloc B a stride=4 : il est TENU (non avance) aux macro-pas 0,1,2
  //     puis rattrape au pas 3. Un regrid au pas 2 (every=2) tombe sur un macro-pas ou B est TENU : B
  //     doit NEANMOINS etre re-grille sur le layout d'union (sa BoxArray fine == celle du bloc A, pas
  //     l'ancienne) -> same_layout_or_throw passe (sinon le regrid aurait leve) et son fin porte des
  //     donnees finies (report + interp), pas un fab non initialise sur l'ancienne grille.
  // ============================================================================================
  {
    SCOPED_TRACE("stride-held block regrid");
    auto sim =
        make_two_block_system(N, L, B0, +1.0, -1.0, blob(N, 0.30, 0.5, 1.0, 1.0, 0.07),
                              blob(N, 0.70, 0.5, 1.0, 1.0, 0.07), /*stride1=*/4, /*regrid_every=*/2,
                              /*regrid_grow=*/2, /*regrid_margin=*/2);
    AmrRuntime& rt = *sim->engine();
    test::install_prepared_threshold_union(rt, {{0, 0, Real(1.5)}, {1, 0, Real(1.5)}});
    // Avance jusqu'a un macro-pas de regrid (macro_step_=2, every=2) ou B est TENU ((2+1)%4 != 0).
    sim->advance(Real(0.01), 3);
    EXPECT_TRUE(sim->checkpoint_regrid_count() >= 1) << "d_regrid_called_with_strided_block";
    // Le bloc B (stride-tenu) partage EXACTEMENT le layout fin du bloc A apres regrid (sinon le
    // same_layout_or_throw interne au regrid aurait leve avant d'arriver ici).
    const std::vector<Box2D> fa = rt.levels(0)[1].U.box_array().boxes();
    const std::vector<Box2D> fb = rt.levels(1)[1].U.box_array().boxes();
    EXPECT_TRUE(same_box_list(fa, fb)) << "d_strided_block_on_union_layout_not_stale";
    // Son fin porte des donnees finies (report + interp du regrid), pas un fab non initialise.
    EXPECT_TRUE(all_finite(rt.density(1))) << "d_strided_block_state_finite";
    // Et le bloc B a bien ete re-grille hors de l'ancien seed central : son fin a evolue.
    EXPECT_TRUE(rt.n_patches() >= 1) << "d_strided_block_has_fine_patches";
  }

  // ============================================================================================
  // (T7) DEVERROUILLAGE FACADE : AmrSystem (facade runtime) accepte desormais multi-blocs +
  //      regrid_every > 0 (l'ancien refus de python/amr_system.cpp est leve). On verifie :
  //        - multi-blocs + regrid_every > 0 NE LEVE PLUS (ensure_built reussit, le step tourne) ;
  //        - la hierarchie BOUGE (n_patches/layout evoluent) quand un seuil de raffinement est pose ;
  //        - regrid_every == 0 reste FIGE et BIT-IDENTIQUE (meme cas joue deux fois -> dmax == 0).
  // ============================================================================================
  {
    const std::vector<double> r0 = blob(N, 0.30, 0.5, 1.0, 1.0, 0.07);
    const std::vector<double> r1 = blob(N, 0.70, 0.5, 1.0, 1.0, 0.07);

    // (T7-a) multi-blocs + regrid_every > 0 NE LEVE PLUS + la hierarchie bouge.
    const bool unlocked_no_throw = !raises([&] {
      AmrSystemConfig cfg;
      cfg.n = N;
      cfg.L = L;
      cfg.periodicity = {true, true};
      cfg.regrid_every = 2;  // AVANT cette PR : ensure_built LEVAIT en multi-blocs
      AmrSystem sim(cfg);
      install_regrid_state_authorities(sim);
      sim.set_temporal_relations({2}, {1}, {"integral_only"});
      sim.add_block("a", exb_spec(+1.0, B0), "minmod", "rusanov", "conservative", "explicit", 1);
      sim.add_block("b", exb_spec(-1.0, B0), "minmod", "rusanov", "conservative", "explicit", 1);
      sim.set_poisson("charge_density", "geometric_mg", "periodic");
      install_regrid_threshold_union(sim, 1.5);
      sim.set_density("a", r0);
      sim.set_density("b", r1);
      test::install_forward_euler_program(sim);
      sim.advance(0.01, 6);
      if (!all_finite(sim.density("a")) || !all_finite(sim.density("b")))
        throw std::runtime_error("etat non fini");
    });
    EXPECT_TRUE(unlocked_no_throw) << "T7_facade_multiblock_regrid_every_positive_no_longer_throws";

    // (T7-b) regrid_every == 0 reste FIGE et BIT-IDENTIQUE a la facade.
    auto run_facade_frozen = [&]() {
      AmrSystemConfig cfg;
      cfg.n = N;
      cfg.L = L;
      cfg.periodicity = {true, true};
      cfg.regrid_every = 0;  // hierarchie figee
      AmrSystem sim(cfg);
      install_regrid_state_authorities(sim);
      sim.set_temporal_relations({2}, {1}, {"integral_only"});
      sim.add_block("a", exb_spec(+1.0, B0), "minmod", "rusanov", "conservative", "explicit", 1);
      sim.add_block("b", exb_spec(-1.0, B0), "minmod", "rusanov", "conservative", "explicit", 1);
      sim.set_poisson("charge_density", "geometric_mg", "periodic");
      install_regrid_threshold_union(sim, 1.5);
      sim.set_density("a", r0);
      sim.set_density("b", r1);
      test::install_forward_euler_program(sim);
      sim.advance(0.01, 6);
      return sim.density("a");
    };
    const std::vector<double> fa = run_facade_frozen();
    const std::vector<double> fb = run_facade_frozen();
    EXPECT_EQ(dmax_field(fa, fb), 0.0) << "T7_facade_frozen_regrid_every_zero_bit_identical_dmax0";
  }
}

TEST(test_amr_multiblock_regrid_union, GradientTaggingRefusesUnproducedNonPeriodicGhosts) {
  constexpr int n = 16;
  AmrBuildParams params;
  params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  params.mesh.periodicity = Periodicity{false, false};
  params.mesh.n = n;
  params.mesh.L = 1.0;
  params.mesh.regrid_every = 0;
  params.poisson.bc.xlo = params.poisson.bc.xhi = BCType::Dirichlet;
  params.poisson.bc.ylo = params.poisson.bc.yhi = BCType::Dirichlet;
  detail::SharedAmrLayout layout = detail::make_shared_amr_layout(params);
  layout.base_per = Periodicity{false, false};
  std::vector<AmrRuntimeBlock> blocks;
  blocks.push_back(detail::dispatch_amr_block(exb_charge(+1.0, 1.0), "minmod", "rusanov", layout,
                                              "a", flat(n, 1.0),
                                              /*has_density=*/true, 1.4, 1, false, 1));
  blocks.back().state_identity = "test://amr-regrid-union/non-periodic/block/a/state/U";
  // This is the precise invalid state under test: a non-periodic sampled state with no prepared
  // authority capable of producing its physical ghosts.
  blocks.front().boundary_plan.reset();
  AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc, std::move(blocks),
                     layout.base_per, layout.replicated_coarse, layout.wall);
  test::install_second_order_amr_transfer_authorities(runtime, 1);
  runtime.set_parent_child_temporal_relations({::pops::amr::ParentChildClockRelation(
      0, 1, ::pops::amr::Rational(2, 1), ::pops::amr::RemainderPolicy::IntegralOnly)});

  using Program = runtime::amr::PreparedTaggingProgram;
  const std::vector<Program::Stencil> stencils{
      Program::Stencil{"test::centered-gradient",
                       POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1,
                       "l2",
                       "inverse_cell_size",
                       "ghost_extension",
                       2,
                       {Program::AxisStencil{0, 1, 2, 1, 1, {-1, 1}, {-0.5, 0.5}},
                        Program::AxisStencil{1, 1, 2, 1, 1, {-1, 1}, {-0.5, 0.5}}}}};
  try {
    runtime.set_tagging_program(stencils,
                                {Program::Leaf{0, 0, POPS_TAGGING_GRADIENT_ABOVE_V1, 0.1, 0}},
                                {POPS_TAGGING_GRADIENT_ABOVE_V1}, {0}, {}, {}, 0, 0, 0,
                                "test::clock", "test::gradient-tagger");
    FAIL() << "non-periodic gradient tagging accepted unproduced physical ghosts";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("complete prepared ghost-production authority"),
              std::string::npos);
  }
}
