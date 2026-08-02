// Peuplement de B_z PAR NIVEAU dans le coupleur AMR de systeme (AmrSystemCoupler).
//
// PR #37 a rendu le canal aux extensible width-aware sur les chemins AMR : un bloc declarant
// n_aux=4 dispose de la PLACE pour B_z a chaque niveau, mais B_z restait a 0 (sauf propagation
// coarse->fine par coupler_inject_aux_mb). Ce test verifie le nouveau mecanisme : un B_z fourni
// par l'utilisateur (ScalarFieldProvider2D prepare, comme le bz_ du SystemAssembler mono-niveau)
// est POSE sur la composante B_z (indice kAuxBaseComps) du canal aux partage de CHAQUE niveau,
// echantillonne aux centres de cellule DU NIVEAU (chaque niveau a sa geometrie / dx).
//
//   (A) PEUPLEMENT PAR NIVEAU, ECHANTILLONNE A LA RESOLUTION DU NIVEAU : avec un B_z spatialement
//       variable bz(x,y)=1+x, on verifie que sim.aux(0) (grossier) et sim.aux(1) (fin) portent
//       chacun B_z = bz(x_cell_DU_NIVEAU). Sur le fin, deux cellules contenues dans une meme
//       cellule grossiere ont des B_z DISTINCTS (echantillonnage fin), ce qu'une simple injection
//       coarse->fine (constante par cellule grossiere) ne produirait pas.
//   (B) PRESERVATION par solve_fields : phi resolu (ici 0), field_postprocess n'ecrit que comp
//       0..2 ; B_z (comp 3) inchange a tous les niveaux apres solve_fields.
//   (C) SETTER set_bz : poser B_z apres construction donne le meme resultat que par le ctor.
//   (D) GARDE / BIT-IDENTITE : sans bz fourni, la composante B_z reste 0 (comportement historique).
//   (E) CONSOMMATION SPATIALE : le residu de production lit B_z sur les niveaux grossier et fin ;
//       ProgramGraph reste l'unique autorite qui integre ensuite ce residu dans le temps.

#include <gtest/gtest.h>

#include <pops/core/model/coupled_system.hpp>
#include <pops/core/model/physical_model.hpp>
#include <pops/core/state/state.hpp>
#include <pops/coupling/system/amr_system_coupler.hpp>
#include <pops/coupling/base/elliptic_rhs.hpp>  // ChargeDensityRhs
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // AmrLevelMP
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace pops;

// Croissance pilotee par B_z : flux nul, elliptique nul (phi=0), source S = B_z*u. Lit a.B_z
// -> declare n_aux=4. du/dt = B_z u, Euler avant par sous-pas.
struct BzGrowPop {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  static constexpr int n_aux = 4;  // phi, grad_x, grad_y, B_z
  POPS_HD State flux(const State&, const auto&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return Real(0); }
  POPS_HD State source(const State& u, const Aux& a) const { return State{a.B_z * u[0]}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

// Bloc de base (n_aux defaut = 3) : advection en x, ne lit pas l'aux, source nulle.
struct AdvectXPop {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  Real v = Real(1);
  POPS_HD State flux(const State& u, const auto&, int dir) const {
    return State{dir == 0 ? v * u[0] : Real(0)};
  }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return std::fabs(v); }
  POPS_HD State source(const State&, const Aux&) const { return State{}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

static_assert(PhysicalModel<BzGrowPop> && PhysicalModel<AdvectXPop>);
static_assert(aux_comps<BzGrowPop>() == 4, "BzGrowPop declare n_aux = 4");
static_assert(aux_comps<AdvectXPop>() == kAuxBaseComps, "AdvectXPop reste au contrat de base");

namespace {

// Geometrie commune des tests : domaine 16x16 sur [0,1]^2, patch fin {{8,8},{23,23}}
// (couvre les cellules grossieres [4..11]^2). Le coin (0,0) grossier est NON couvert.
constexpr int NC = 16;

// B_z spatialement variable : echantillonner a la resolution du niveau change la valeur.
Real bz_field(Real x, Real /*y*/) {
  return Real(1) + x;
}

ScalarFieldProvider2D bz_provider() {
  return ScalarFieldProvider2D::trusted_extension(
      {"pops.test.scalar-field.affine-2d", 1}, exact_provider_parameters(Real(1), Real(1), Real(0)),
      bz_field);
}

// Lit la composante B_z (comp kAuxBaseComps) d'un MultiFab a la cellule (i,j) du fab 0.
Real read_bz(const MultiFab& A, int i, int j) {
  return A.fab(0).const_array()(i, j, kAuxBaseComps);
}

template <class Model>
Real max_bz_source_residual_error(const Model& model, const MultiFab& U, const MultiFab& aux,
                                  Real dx, Real dy) {
  std::vector<Box2D> xfaces;
  std::vector<Box2D> yfaces;
  xfaces.reserve(U.box_array().size());
  yfaces.reserve(U.box_array().size());
  for (const Box2D& box : U.box_array().boxes()) {
    xfaces.push_back(xface_box(box));
    yfaces.push_back(yface_box(box));
  }

  MultiFab Fx(BoxArray(std::move(xfaces)), U.dmap(), Model::n_vars, 0);
  MultiFab Fy(BoxArray(std::move(yfaces)), U.dmap(), Model::n_vars, 0);
  MultiFab residual(U.box_array(), U.dmap(), Model::n_vars, 0);
  Fx.set_val(Real(0));
  Fy.set_val(Real(0));
  mf_eval_rhs(model, U, aux, Fx, Fy, dx, dy, residual);
  device_fence();

  Real error = Real(0);
  for (int li = 0; li < U.local_size(); ++li) {
    const Box2D valid = U.box(li);
    const ConstArray4 state = U.fab(li).const_array();
    const ConstArray4 auxiliary = aux.fab(li).const_array();
    const ConstArray4 rhs = residual.fab(li).const_array();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        error = std::max(error,
                         std::fabs(rhs(i, j, 0) - auxiliary(i, j, kAuxBaseComps) * state(i, j, 0)));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(error)));
}

}  // namespace

// Regroupe la geometrie partagee (grossier + patch fin) et la fabrique `build` qui construit un
// coupleur AMR frais (etats remis) pour chaque scenario (A-E). Geometrie/box-arrays identiques
// d'un scenario a l'autre -> construits une seule fois via SetUpTestSuite.
class AmrSystemBzPopTest : public ::testing::Test {
 protected:
  using BzBlk = EquationBlock<BzGrowPop, FirstOrder, ExplicitTime<SSPRK2, 1>>;
  using BaseBlk = EquationBlock<AdvectXPop, FirstOrder, ExplicitTime<SSPRK2, 1>>;

  static void SetUpTestSuite() {
    dom_ = new Box2D(Box2D::from_extents(NC, NC));
    geom_ = new Geometry{*dom_, 0.0, 1.0, 0.0, 1.0};
    ba_coarse_ = new BoxArray(std::vector<Box2D>{*dom_});
    dm_ = new DistributionMapping(1, n_ranks());
    fbox_ = new Box2D{{8, 8}, {23, 23}};
    ba_fine_ = new BoxArray(std::vector<Box2D>{*fbox_});
  }

  static void TearDownTestSuite() {
    delete dom_;
    delete geom_;
    delete ba_coarse_;
    delete dm_;
    delete fbox_;
    delete ba_fine_;
  }

  // Fabrique : construit un coupleur frais (etats remis). bz_user vide => garde (D).
  auto build(ScalarFieldProvider2D bz_user, bool use_setter, Real u0g) {
    const Geometry& geom = *geom_;
    const BoxArray& ba_coarse = *ba_coarse_;
    const DistributionMapping& dm = *dm_;
    const BoxArray& ba_fine = *ba_fine_;
    const Real dxc = geom.dx(), dyc = geom.dy();
    MultiFab UgC(ba_coarse, dm, 1, 2), UgF(ba_fine, dm, 1, 2);
    MultiFab UbC(ba_coarse, dm, 1, 2), UbF(ba_fine, dm, 1, 2);
    UgC.set_val(u0g);
    UgF.set_val(u0g);
    UbC.set_val(Real(1));
    UbF.set_val(Real(1));

    BzBlk g{"grow", BzGrowPop{}, UgC, BCRec{}};
    BaseBlk b{"base", AdvectXPop{Real(0)}, UbC, BCRec{}};  // v=0 : inerte
    CoupledSystem system{g, b};

    auto make = [&](MultiFab&& U, Real dx, Real dy) {
      return AmrLevelMP{std::move(U), nullptr, dx, dy};
    };
    std::vector<std::vector<AmrLevelMP>> bl;
    bl.emplace_back();
    bl.back().push_back(make(std::move(UgC), dxc, dyc));
    bl.back().push_back(make(std::move(UgF), dxc / 2, dyc / 2));
    bl.emplace_back();
    bl.back().push_back(make(std::move(UbC), dxc, dyc));
    bl.back().push_back(make(std::move(UbF), dxc / 2, dyc / 2));

    ChargeDensityRhs charge{{{Real(0), 0}, {Real(0), 0}}};  // charges nulles -> phi = 0
    using Sim = AmrSystemCoupler<decltype(system), ChargeDensityRhs>;
    auto sim = std::make_unique<Sim>(system, geom, ba_coarse, BCRec{}, charge, std::move(bl),
                                     Periodicity{true, true}, /*replicated_coarse=*/true,
                                     ActiveRegionProvider2D{},
                                     use_setter ? ScalarFieldProvider2D{} : bz_user);
    if (use_setter)
      sim->set_bz(bz_user);
    return sim;
  }

  static Box2D* dom_;
  static Geometry* geom_;
  static BoxArray* ba_coarse_;
  static DistributionMapping* dm_;
  static Box2D* fbox_;
  static BoxArray* ba_fine_;
};

Box2D* AmrSystemBzPopTest::dom_ = nullptr;
Geometry* AmrSystemBzPopTest::geom_ = nullptr;
BoxArray* AmrSystemBzPopTest::ba_coarse_ = nullptr;
DistributionMapping* AmrSystemBzPopTest::dm_ = nullptr;
Box2D* AmrSystemBzPopTest::fbox_ = nullptr;
BoxArray* AmrSystemBzPopTest::ba_fine_ = nullptr;

// --- (A)(B) peuplement par niveau et preservation par solve_fields.
TEST_F(AmrSystemBzPopTest, LevelwisePopulationAndFieldSolve) {
  const Geometry& geom = *geom_;
  const Box2D& fbox = *fbox_;
  auto sim = build(bz_provider(), /*use_setter=*/false, /*u0g=*/Real(2));
  EXPECT_EQ(sim->aux_ncomp(), 4) << "shared_aux_width_max_4";

  // grossier : B_z(i) = bz(x_cell_grossier(i)). Geometrie grossiere = geom.
  bool coarse_ok = true;
  for (int i = 0; i < NC; ++i)
    if (std::fabs(read_bz(sim->aux(0), i, 4) - bz_field(geom.x_cell(i), 0)) > Real(1e-12))
      coarse_ok = false;
  EXPECT_TRUE(coarse_ok) << "coarse_Bz_sampled_at_coarse_centers";

  // fin : B_z(I) = bz(x_cell_fin(I)) avec la geometrie raffinee (dx/2). Deux cellules fines
  // 2*ci et 2*ci+1 (dans la meme cellule grossiere ci) ont des B_z DISTINCTS -> echantillonnage
  // a la resolution fine, pas une injection constante par cellule grossiere.
  const Geometry gf = geom.refine(2);
  bool fine_ok = true;
  for (int I = fbox.lo[0]; I <= fbox.hi[0]; ++I) {
    const Real got = read_bz(sim->aux(1), I, fbox.lo[1]);
    if (std::fabs(got - bz_field(gf.x_cell(I), 0)) > Real(1e-12))
      fine_ok = false;
  }
  EXPECT_TRUE(fine_ok) << "fine_Bz_sampled_at_fine_centers";

  // les deux cellules fines d'une meme cellule grossiere different (B_z variable, fin-echantillonne)
  const Real bz_lo = read_bz(sim->aux(1), fbox.lo[0], fbox.lo[1]);
  const Real bz_hi = read_bz(sim->aux(1), fbox.lo[0] + 1, fbox.lo[1]);
  EXPECT_GT(std::fabs(bz_lo - bz_hi), Real(1e-6)) << "fine_Bz_resolves_subcoarse_variation";

  // --- (B) preservation par solve_fields (field_postprocess n'ecrit que comp 0..2) --------
  sim->solve_fields();
  EXPECT_LT(std::fabs(read_bz(sim->aux(0), 7, 7) - bz_field(geom.x_cell(7), 0)), Real(1e-12))
      << "coarse_Bz_preserved_after_solve_fields";
  EXPECT_LT(std::fabs(read_bz(sim->aux(1), fbox.lo[0], fbox.lo[1]) - bz_lo), Real(1e-12))
      << "fine_Bz_preserved_after_solve_fields";

  // Spatial proof that B_z is not merely stored: the production residual consumes it as
  // S(U,aux)=B_z*U on both hierarchy levels. ProgramGraph owns the subsequent time update.
  const auto& grow_levels = sim->levels(0);
  EXPECT_LT(max_bz_source_residual_error(BzGrowPop{}, grow_levels[0].U, sim->aux(0),
                                         grow_levels[0].dx, grow_levels[0].dy),
            Real(1e-12))
      << "coarse_residual_consumes_Bz";
  EXPECT_LT(max_bz_source_residual_error(BzGrowPop{}, grow_levels[1].U, sim->aux(1),
                                         grow_levels[1].dx, grow_levels[1].dy),
            Real(1e-12))
      << "fine_residual_consumes_Bz";
}

// --- (C) setter set_bz : meme resultat que par le ctor -----------------------------------
TEST_F(AmrSystemBzPopTest, SetterMatchesConstructor) {
  const Geometry& geom = *geom_;
  const Box2D& fbox = *fbox_;
  auto sim = build(bz_provider(), /*use_setter=*/true, /*u0g=*/Real(2));
  const Geometry gf = geom.refine(2);
  EXPECT_LT(std::fabs(read_bz(sim->aux(0), 3, 3) - bz_field(geom.x_cell(3), 0)), Real(1e-12))
      << "set_bz_populates_coarse";
  EXPECT_LT(
      std::fabs(read_bz(sim->aux(1), fbox.lo[0], fbox.lo[1]) - bz_field(gf.x_cell(fbox.lo[0]), 0)),
      Real(1e-12))
      << "set_bz_populates_fine";
}

// --- (D) garde : sans bz fourni, la composante B_z reste 0 (bit-identite historique) ------
TEST_F(AmrSystemBzPopTest, NoBzStaysZero) {
  const Box2D& fbox = *fbox_;
  auto sim = build({}, /*use_setter=*/false, /*u0g=*/Real(2));
  // le canal reste alloue a 4 (un bloc declare n_aux=4) mais comp 3 = 0 (MultiFab non
  // initialise par fill_bz -> valeur par defaut 0 de l'allocation).
  sim->solve_fields();
  EXPECT_LT(std::fabs(read_bz(sim->aux(0), 5, 5)), Real(1e-30)) << "no_bz_coarse_stays_zero";
  EXPECT_LT(std::fabs(read_bz(sim->aux(1), fbox.lo[0], fbox.lo[1])), Real(1e-30))
      << "no_bz_fine_stays_zero";
}
