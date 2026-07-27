// Peuplement de B_z PAR NIVEAU dans AmrSystemCoupler : couverture MULTI-BOX.
//
// test_amr_system_bz_pop valide le mecanisme en MONO-RANG / MONO-BOX par niveau (un seul fab par
// niveau). Ce test couvre le cas ou un niveau a PLUSIEURS boites : grossier decoupe en 2x2 = 4
// boites (BoxArray::from_domain), niveau fin = 2 patchs DISJOINTS (ilots), chacun dans une boite
// grossiere differente. fill_bz parcourt fab(li) pour TOUS les li : il doit echantillonner B_z(x,y)
// aux centres de cellule DE CHAQUE boite (valides + halos), independamment du decoupage. Avec un B_z
// SPATIALEMENT VARIABLE en x ET en y (bz(x,y)=1+x+2y), chaque boite voit une plage de valeurs
// distincte ; un bug d'indexation (ne remplir que fab(0), ou confondre les boites) serait detecte.
//
//   (A) PEUPLEMENT MULTI-BOX, ECHANTILLONNE A LA RESOLUTION DU NIVEAU : sur CHAQUE boite du grossier
//       et de CHAQUE patch fin, chaque cellule (valides ET ghosts) porte B_z = bz(centre DU NIVEAU).
//       Verifie aussi aux bords de boite / ghosts (la ou un decoupage casse l'indexation naive).
//   (B) PRESERVATION par solve_fields sur toutes les boites (field_postprocess n'ecrit que comp 0..2 ;
//       l'injection coarse->fine est re-ecrasee par le re-fill fin). B_z inchange partout.
//   (C) SETTER set_bz : meme resultat multi-box que par le ctor.
//   (D) CONSOMMATION SPATIALE : le residu de production lit B_z sur chaque boite locale des niveaux
//       grossier et fin ; ProgramGraph reste l'unique autorite temporelle.
//
// La grille AMR est PARTAGEE par tous les blocs (contrat du coupleur : meme BoxArray par niveau) ;
// un bloc lit B_z (n_aux=4, S=B_z*u), un bloc de base (n_aux=3) l'ignore. Grossier DE-REPLIQUE
// (replicated_coarse=false) : la grille grossiere multi-box est reellement repartie (SFC), ce que le
// mono-box replique n'exerce pas. Charges nulles -> phi=0, Poisson trivialement correct : le test
// isole le peuplement de B_z.

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

// Croissance pilotee par B_z (calque de BzGrowPop de test_amr_system_bz_pop) : flux nul, phi=0,
// source S = B_z*u. Lit a.B_z -> n_aux=4. du/dt = B_z u, Euler avant par sous-pas.
struct BzGrowMB {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  static constexpr int n_aux = 4;  // phi, grad_x, grad_y, B_z
  POPS_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  POPS_HD State source(const State& u, const Aux& a) const { return State{a.B_z * u[0]}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

// Bloc de base (n_aux defaut = 3) : ne lit pas l'aux, inerte (v=0, source nulle).
struct InertMB {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  POPS_HD State flux(const State&, const Aux&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  POPS_HD State source(const State&, const Aux&) const { return State{}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
};

static_assert(PhysicalModel<BzGrowMB> && PhysicalModel<InertMB>);
static_assert(aux_comps<BzGrowMB>() == 4, "BzGrowMB declare n_aux = 4");
static_assert(aux_comps<InertMB>() == kAuxBaseComps, "InertMB reste au contrat de base");

namespace {

// Domaine 16x16 sur [0,1]^2.
constexpr int NC = 16;

// B_z variable en x ET en y : exerce l'echantillonnage par boite dans les deux directions.
Real bz_field(Real x, Real y) {
  return Real(1) + x + Real(2) * y;
}

ScalarFieldProvider2D bz_provider() {
  return ScalarFieldProvider2D::trusted_extension(
      {"pops.test.scalar-field.affine-2d", 1}, exact_provider_parameters(Real(1), Real(1), Real(2)),
      bz_field);
}

// Lit la composante B_z (comp kAuxBaseComps) du fab local li d'un MultiFab, a la cellule (i,j).
Real read_bz(const MultiFab& A, int li, int i, int j) {
  return A.fab(li).const_array()(i, j, kAuxBaseComps);
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

TEST(test_amr_system_bz_multibox, Runs) {
  comm_init();

  const Box2D dom = Box2D::from_extents(NC, NC);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const Geometry gf = geom.refine(2);  // geometrie du niveau fin (dx/2)
  const Real dxc = geom.dx(), dyc = geom.dy();

  // grossier MULTI-BOX : 2x2 = 4 boites (max_grid_size=8 sur 16). Decoupage reparti (SFC, np rangs).
  const BoxArray ba_coarse = BoxArray::from_domain(dom, 8);
  const DistributionMapping dm_coarse(ba_coarse.size(), n_ranks());

  // niveau fin MULTI-BOX : 2 patchs DISJOINTS, chacun dans une boite grossiere differente.
  //   patch 0 : fin {{4,4},{11,11}}   = cellules grossieres [2..5]^2   (boite grossiere coin (0,0))
  //   patch 1 : fin {{20,20},{27,27}} = cellules grossieres [10..13]^2 (boite grossiere coin (8,8))
  const Box2D fb0{{4, 4}, {11, 11}};
  const Box2D fb1{{20, 20}, {27, 27}};
  const BoxArray ba_fine(std::vector<Box2D>{fb0, fb1});
  const DistributionMapping dm_fine(ba_fine.size(), n_ranks());

  using BzBlk = EquationBlock<BzGrowMB, FirstOrder, ExplicitTime<SSPRK2, 1>>;
  using BaseBlk = EquationBlock<InertMB, FirstOrder, ExplicitTime<SSPRK2, 1>>;

  // Fabrique : coupleur frais (etats remis a u0g pour le bloc B_z, 1 pour le bloc inerte).
  auto build = [&](ScalarFieldProvider2D bz_user, bool use_setter, Real u0g) {
    MultiFab UgC(ba_coarse, dm_coarse, 1, 2), UgF(ba_fine, dm_fine, 1, 2);
    MultiFab UbC(ba_coarse, dm_coarse, 1, 2), UbF(ba_fine, dm_fine, 1, 2);
    UgC.set_val(u0g);
    UgF.set_val(u0g);
    UbC.set_val(Real(1));
    UbF.set_val(Real(1));

    BzBlk g{"grow", BzGrowMB{}, UgC, BCRec{}};
    BaseBlk b{"base", InertMB{}, UbC, BCRec{}};
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
                                     Periodicity{true, true},
        /*replicated_coarse=*/false,  // grossier multi-box reparti
        ActiveRegionProvider2D{}, use_setter ? ScalarFieldProvider2D{} : bz_user);
    if (use_setter)
      sim->set_bz(bz_user);
    return sim;
  };

  // verifie B_z = bz(centre du niveau) sur CHAQUE cellule (valides + ghosts) de CHAQUE boite locale
  // du MultiFab A, avec la geometrie g du niveau. Renvoie false des la premiere cellule fausse.
  auto bz_all_boxes_ok = [&](const MultiFab& A, const Geometry& g) {
    for (int li = 0; li < A.local_size(); ++li) {
      const Box2D gb = A.fab(li).grown_box();  // valides + halos
      for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
        for (int i = gb.lo[0]; i <= gb.hi[0]; ++i)
          if (std::fabs(read_bz(A, li, i, j) - bz_field(g.x_cell(i), g.y_cell(j))) > Real(1e-12))
            return false;
    }
    return true;
  };

  // --- (A) peuplement multi-box, echantillonne a la resolution du niveau --------------------
  {
    auto sim = build(bz_provider(), /*use_setter=*/false, /*u0g=*/Real(2));
    EXPECT_EQ(sim->aux_ncomp(), 4) << "shared_aux_width_max_4";
    EXPECT_GE(ba_coarse.size(), 2)
        << "coarse_is_multibox";  // garde-fou : le decoupage a bien eu lieu
    EXPECT_EQ(ba_fine.size(), 2) << "fine_is_multibox";

    // grossier : B_z correct sur les 4 boites (valides + ghosts).
    EXPECT_TRUE(bz_all_boxes_ok(sim->aux(0), geom)) << "coarse_Bz_all_boxes_all_cells";
    // fin : B_z correct sur les 2 patchs (valides + ghosts), echantillonne a dx/2.
    EXPECT_TRUE(bz_all_boxes_ok(sim->aux(1), gf)) << "fine_Bz_all_boxes_all_cells";

    // les deux patchs fins voient des plages de B_z DISTINCTES (boites disjointes, B_z variable) :
    // le coin bas-gauche du patch 0 vs celui du patch 1 doivent differer nettement.
    Real bz_p0 = -1, bz_p1 = -1;
    for (int li = 0; li < sim->aux(1).local_size(); ++li) {
      const Box2D b = sim->aux(1).box(li);
      if (b.lo[0] == fb0.lo[0] && b.lo[1] == fb0.lo[1])
        bz_p0 = read_bz(sim->aux(1), li, fb0.lo[0], fb0.lo[1]);
      if (b.lo[0] == fb1.lo[0] && b.lo[1] == fb1.lo[1])
        bz_p1 = read_bz(sim->aux(1), li, fb1.lo[0], fb1.lo[1]);
    }
    // au moins un rang possede chaque patch ; si local, la valeur a ete lue. Sur np=1 les deux sont
    // locaux. On exige la difference seulement la ou les deux patchs sont locaux (np=1 garanti).
    if (bz_p0 >= 0 && bz_p1 >= 0)
      EXPECT_TRUE(std::fabs(bz_p0 - bz_p1) > Real(1)) << "fine_patches_see_distinct_Bz";

    // --- (B) preservation par solve_fields sur toutes les boites -----------------------------
    sim->solve_fields();
    EXPECT_TRUE(bz_all_boxes_ok(sim->aux(0), geom))
        << "coarse_Bz_preserved_after_solve_fields_all_boxes";
    EXPECT_TRUE(bz_all_boxes_ok(sim->aux(1), gf))
        << "fine_Bz_preserved_after_solve_fields_all_boxes";

    // B_z must be consumed by the production spatial residual on every local box and both
    // hierarchy levels; ProgramGraph owns the time integration of this residual.
    const auto& grow_levels = sim->levels(0);
    EXPECT_LT(max_bz_source_residual_error(BzGrowMB{}, grow_levels[0].U, sim->aux(0),
                                           grow_levels[0].dx, grow_levels[0].dy),
              Real(1e-12))
        << "coarse_multibox_residual_consumes_Bz";
    EXPECT_LT(max_bz_source_residual_error(BzGrowMB{}, grow_levels[1].U, sim->aux(1),
                                           grow_levels[1].dx, grow_levels[1].dy),
              Real(1e-12))
        << "fine_multibox_residual_consumes_Bz";
  }

  // --- (C) setter set_bz : meme resultat multi-box que par le ctor -------------------------
  {
    auto sim = build(bz_provider(), /*use_setter=*/true, /*u0g=*/Real(2));
    EXPECT_TRUE(bz_all_boxes_ok(sim->aux(0), geom)) << "set_bz_populates_coarse_all_boxes";
    EXPECT_TRUE(bz_all_boxes_ok(sim->aux(1), gf)) << "set_bz_populates_fine_all_boxes";
  }

  // --- (D) garde : sans bz fourni, la composante B_z reste 0 sur toutes les boites ----------
  {
    auto sim = build({}, /*use_setter=*/false, /*u0g=*/Real(2));
    sim->solve_fields();
    bool coarse_zero = true, fine_zero = true;
    for (int li = 0; li < sim->aux(0).local_size(); ++li) {
      const Box2D b = sim->aux(0).box(li);
      if (std::fabs(read_bz(sim->aux(0), li, b.lo[0], b.lo[1])) >= Real(1e-30))
        coarse_zero = false;
    }
    for (int li = 0; li < sim->aux(1).local_size(); ++li) {
      const Box2D b = sim->aux(1).box(li);
      if (std::fabs(read_bz(sim->aux(1), li, b.lo[0], b.lo[1])) >= Real(1e-30))
        fine_zero = false;
    }
    EXPECT_TRUE(coarse_zero) << "no_bz_coarse_stays_zero_all_boxes";
    EXPECT_TRUE(fine_zero) << "no_bz_fine_stays_zero_all_boxes";
  }

  comm_finalize();
}
