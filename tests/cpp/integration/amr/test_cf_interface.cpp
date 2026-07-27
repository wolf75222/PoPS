// Contrat de CoarseFineInterface. Son integration est couverte bit a bit par les tests de reflux
// (np=1/2/4 identiques). Ici on fige les mecaniques locales :
//   - couverture batie depuis un BoxArray fin (empreinte PatchRange)
//     et routage bordant du reflux (formules et garde de couverture).

#include <gtest/gtest.h>

#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // pops::CoarseFineInterface
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <vector>

using namespace pops;

namespace {
// registre minimal EdgeStrip-shaped (champs lus par route_reflux).
struct RegLite {
  int I0, I1, J0, J1;
  RefluxStorage<Real> cL, cR, cB, cT, fL, fR, fB, fT;
};
}  // namespace

TEST(test_cf_interface, Runs) {
  // --- CoarseFineInterface : couverture depuis un BoxArray fin ---
  // patch fin [4..11]^2 -> empreinte grossiere [2..5]^2 (PatchRange). Region grossiere 8x8.
  BoxArray fine(std::vector<Box2D>{Box2D{{4, 4}, {11, 11}}});
  CoarseFineInterface cfi(Box2D{{0, 0}, {7, 7}}, fine, Periodicity{false, false});
  EXPECT_TRUE(cfi.covered(2, 2) && cfi.covered(5, 5) && cfi.covered(3, 4)) << "cfi_couvert_dedans";
  EXPECT_TRUE(!cfi.covered(1, 2) && !cfi.covered(6, 5)) << "cfi_non_couvert_bordant";
  EXPECT_TRUE(!cfi.covered(-1, -1) && !cfi.covered(8, 8)) << "cfi_hors_region";

  // --- route_reflux : correction bordante coverage-aware ---
  // un patch, 1 composante, empreinte grossiere [2..5]^2. On pose des flux simples et on
  // verifie la formule -(fL - cL*dt)/dx sur le bord gauche et +(fR - cR*dt)/dx a droite.
  const int nc = 1;
  RegLite g;
  g.I0 = 2;
  g.I1 = 5;
  g.J0 = 2;
  g.J1 = 5;
  const int nJ = g.J1 - g.J0 + 1, nI = g.I1 - g.I0 + 1;
  g.cL.assign(nJ * nc, Real(1));
  g.cR.assign(nJ * nc, Real(2));
  g.cB.assign(nI * nc, Real(3));
  g.cT.assign(nI * nc, Real(4));
  g.fL.assign(nJ * nc, Real(10));
  g.fR.assign(nJ * nc, Real(20));
  g.fB.assign(nI * nc, Real(30));
  g.fT.assign(nI * nc, Real(40));
  const Real dx = Real(0.5), dy = Real(0.25), dt = Real(2);

  // registre grossier sur l'interface (boite englobante crue de 1, clampee).
  FluxRegister ref(Box2D{{1, 1}, {6, 6}}, nc);
  cfi.route_reflux(g, dx, dy, dt, ref, nc);
  device_fence();

  // bord gauche I0-1 = 1 (non couvert) : -(fL - cL*dt)/dx = -(10 - 1*2)/0.5 = -16.
  EXPECT_EQ(ref.at(1, 2, 0), -(Real(10) - Real(1) * dt) / dx) << "reflux_gauche";
  // bord droit I1+1 = 6 (non couvert) : +(fR - cR*dt)/dx = (20 - 2*2)/0.5 = 32.
  EXPECT_EQ(ref.at(6, 2, 0), +(Real(20) - Real(2) * dt) / dx) << "reflux_droite";
  // bord bas J0-1 = 1 : -(fB - cB*dt)/dy = -(30 - 3*2)/0.25 = -96.
  EXPECT_EQ(ref.at(2, 1, 0), -(Real(30) - Real(3) * dt) / dy) << "reflux_bas";
  // bord haut J1+1 = 6 : +(fT - cT*dt)/dy = (40 - 4*2)/0.25 = 128.
  EXPECT_EQ(ref.at(2, 6, 0), +(Real(40) - Real(4) * dt) / dy) << "reflux_haut";

  // garde de couverture : un patch dont le bord gauche tombe sur une cellule COUVERTE par un
  // second patch ne doit PAS verser de reflux la (joint fin-fin). Deux patchs adjacents
  // [4..11]x[4..11] et [12..19]x[4..11] -> empreintes [2..5] et [6..9] en x. Le bord droit du
  // premier (I1+1 = 6) est alors couvert par le second.
  BoxArray two(std::vector<Box2D>{Box2D{{4, 4}, {11, 11}}, Box2D{{12, 4}, {19, 11}}});
  CoarseFineInterface cfi2(Box2D{{0, 0}, {15, 7}}, two, Periodicity{false, false});
  EXPECT_TRUE(cfi2.covered(6, 2)) << "cfi2_joint_couvert";
  FluxRegister ref2(Box2D{{1, 1}, {10, 6}}, nc);
  cfi2.route_reflux(g, dx, dy, dt, ref2, nc);  // g = premier patch (empreinte [2..5])
  device_fence();
  EXPECT_EQ(ref2.at(6, 2, 0), Real(0)) << "reflux_joint_supprime";  // bord droit couvert -> rien
  EXPECT_EQ(ref2.at(1, 2, 0), -(Real(10) - Real(1) * dt) / dx) << "reflux_gauche_libre";  // libre
}

TEST(test_cf_interface, IntegratedPairRejectsPartiallyMaterializedRoles) {
  const BoxArray fine_boxes(std::vector<Box2D>{Box2D{{4, 4}, {11, 11}}});
  const CoarseFineInterface cfi(Box2D{{0, 0}, {7, 7}}, fine_boxes, Periodicity{false, false});
  FluxRegister correction(cfi.reflux_register_regions(fine_boxes), 1);

  RegLite coarse{};
  coarse.I0 = 2;
  coarse.I1 = 5;
  coarse.J0 = 2;
  coarse.J1 = 5;
  coarse.cR.assign(4, Real(1));
  RegLite absent_fine{};
  absent_fine.I1 = absent_fine.J1 = -1;
  EXPECT_THROW(
      cfi.route_reflux_integrated_pair(coarse, absent_fine, Real(1), Real(1), correction, 1),
      std::runtime_error);

  RegLite absent_coarse{};
  absent_coarse.I1 = absent_coarse.J1 = -1;
  RegLite fine{};
  fine.I0 = 2;
  fine.I1 = 5;
  fine.J0 = 2;
  fine.J1 = 5;
  fine.fT.assign(4, Real(1));
  EXPECT_THROW(
      cfi.route_reflux_integrated_pair(absent_coarse, fine, Real(1), Real(1), correction, 1),
      std::runtime_error);
}

TEST(test_cf_interface, RefluxRejectsRegisterComponentMismatchBeforeLaunching) {
  const BoxArray fine_boxes(std::vector<Box2D>{Box2D{{4, 4}, {11, 11}}});
  const CoarseFineInterface cfi(Box2D{{0, 0}, {7, 7}}, fine_boxes, Periodicity{false, false});
  RegLite strip{};
  strip.I0 = strip.J0 = 2;
  strip.I1 = strip.J1 = 5;
  constexpr int routed_components = 2;
  strip.cL.assign(8, Real(1));
  strip.cR.assign(8, Real(1));
  strip.cB.assign(8, Real(1));
  strip.cT.assign(8, Real(1));
  strip.fL.assign(8, Real(2));
  strip.fR.assign(8, Real(2));
  strip.fB.assign(8, Real(2));
  strip.fT.assign(8, Real(2));
  FluxRegister one_component(cfi.reflux_register_regions(fine_boxes), 1);

  EXPECT_THROW(cfi.route_reflux(strip, Real(1), Real(1), Real(1), one_component, routed_components),
               std::invalid_argument);
  EXPECT_THROW(
      cfi.route_reflux_integrated(strip, Real(1), Real(1), one_component, routed_components),
      std::invalid_argument);
  EXPECT_THROW(cfi.route_reflux_integrated_pair(strip, strip, Real(1), Real(1), one_component,
                                                routed_components),
               std::invalid_argument);
}

TEST(test_cf_interface, PeriodicSeamsWrapCoverageAndRefluxDestinations) {
  const Box2D coarse{{0, 0}, {7, 7}};
  const int nc = 1;
  const Real dx = Real(0.5), dy = Real(0.25), dt = Real(2);

  auto register_for = [](int I0, int I1, int J0, int J1) {
    RegLite g;
    g.I0 = I0;
    g.I1 = I1;
    g.J0 = J0;
    g.J1 = J1;
    const int nJ = J1 - J0 + 1, nI = I1 - I0 + 1;
    g.cL.assign(nJ, Real(1));
    g.cR.assign(nJ, Real(2));
    g.cB.assign(nI, Real(3));
    g.cT.assign(nI, Real(4));
    g.fL.assign(nJ, Real(10));
    g.fR.assign(nJ, Real(20));
    g.fB.assign(nI, Real(30));
    g.fT.assign(nI, Real(40));
    return g;
  };

  // Partial fine footprint [0..3]x[2..5] touches x-low only.  Its left C/F neighbour is coarse
  // cell x=7 through the periodic seam, so the correction must be deposited at (7,J).
  const BoxArray xlow(std::vector<Box2D>{Box2D{{0, 4}, {7, 11}}});
  const CoarseFineInterface cfix(coarse, xlow, Periodicity{true, false});
  EXPECT_FALSE(cfix.covered(-1, 2)) << "wrapped x-high cell remains coarse";
  FluxRegister refx(cfix.reflux_register_regions(xlow), nc);
  EXPECT_TRUE(refx.in(0, 2));
  EXPECT_TRUE(refx.in(7, 2));
  const RegLite gx = register_for(0, 3, 2, 5);
  cfix.route_reflux(gx, dx, dy, dt, refx, nc);
  EXPECT_EQ(refx.at(7, 2, 0), -(Real(10) - Real(1) * dt) / dx)
      << "x-low correction wraps onto x-high coarse cell";
  FluxRegister refx_integrated(cfix.reflux_register_regions(xlow), nc);
  cfix.route_reflux_integrated(gx, dx, dy, refx_integrated, nc);
  EXPECT_EQ(refx_integrated.at(7, 2, 0), -(Real(10) - Real(1)) / dx)
      << "compiled-Program integrated reflux wraps onto x-high coarse cell";

  // If another fine patch covers the opposite x edge at the same J, the periodic seam is fine-fine:
  // coverage wrapping must suppress the correction rather than double-reflux it.
  const BoxArray xboth(std::vector<Box2D>{Box2D{{0, 4}, {7, 11}}, Box2D{{12, 4}, {15, 11}}});
  const CoarseFineInterface cfix_both(coarse, xboth, Periodicity{true, false});
  EXPECT_TRUE(cfix_both.covered(-1, 2)) << "x-low neighbour wraps into covered x-high cell";
  FluxRegister refx_both(cfix_both.reflux_register_regions(xboth), nc);
  cfix_both.route_reflux(gx, dx, dy, dt, refx_both, nc);
  EXPECT_EQ(refx_both.at(7, 2, 0), Real(0)) << "periodic fine-fine seam is not refluxed";

  // Symmetric y-low case: footprint [2..5]x[0..3], bottom correction lands at y=7.
  const BoxArray ylow(std::vector<Box2D>{Box2D{{4, 0}, {11, 7}}});
  const CoarseFineInterface cfiy(coarse, ylow, Periodicity{false, true});
  EXPECT_FALSE(cfiy.covered(2, -1)) << "wrapped y-high cell remains coarse";
  FluxRegister refy(cfiy.reflux_register_regions(ylow), nc);
  EXPECT_TRUE(refy.in(2, 0));
  EXPECT_TRUE(refy.in(2, 7));
  const RegLite gy = register_for(2, 5, 0, 3);
  cfiy.route_reflux(gy, dx, dy, dt, refy, nc);
  EXPECT_EQ(refy.at(2, 7, 0), -(Real(30) - Real(3) * dt) / dy)
      << "y-low correction wraps onto y-high coarse cell";
}

TEST(test_cf_interface, PositivityFloorClampsOnlyTheCoarseFineGhostDensity) {
  const Box2D coarse_domain = Box2D::from_extents(8, 8);
  const BoxArray coarse_boxes(std::vector<Box2D>{coarse_domain});
  const DistributionMapping coarse_mapping(1, n_ranks());
  MultiFab parent(coarse_boxes, coarse_mapping, 2, 0);
  parent.set_val(Real(0));
  ASSERT_EQ(parent.local_size(), 1);
  {
    Array4 coarse = parent.fab(0).array();
    for_each_cell(coarse_domain, [coarse](int i, int j) {
      const bool low_density_parent = i == 1 && j == 3;
      coarse(i, j, 0) = low_density_parent ? Real(1e-10) : Real(1);
      coarse(i, j, 1) = low_density_parent ? Real(0.5) : Real(0.3);
    });
    device_fence();
  }

  const Box2D fine_box{{4, 4}, {11, 11}};
  const BoxArray fine_boxes(std::vector<Box2D>{fine_box});
  const DistributionMapping fine_mapping(1, n_ranks());
  MultiFab fine(fine_boxes, fine_mapping, 2, 2);
  ASSERT_EQ(fine.local_size(), 1);
  constexpr Real floor = Real(1e-6);

  auto count_subfloor_ghosts = [&](const MultiFab& state) {
    int count = 0;
    const ConstArray4 values = state.fab(0).const_array();
    const Box2D valid = state.box(0);
    const Box2D grown = state.fab(0).grown_box();
    for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
      for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
        if (!valid.contains(i, j) && values(i, j, 0) < floor)
          ++count;
    return count;
  };

  fine.set_val(Real(1));
  mf_fill_fine_ghosts_mb(fine, parent, parent, coarse_domain, Real(0.5),
                         /*replicated_parent=*/true, /*positivity_floor=*/Real(0),
                         /*positivity_component=*/0, Periodicity{false, false});
  device_fence();
  EXPECT_GT(count_subfloor_ghosts(fine), 0);

  fine.set_val(Real(1));
  mf_fill_fine_ghosts_mb(fine, parent, parent, coarse_domain, Real(0.5),
                         /*replicated_parent=*/true, floor,
                         /*positivity_component=*/0, Periodicity{false, false});
  device_fence();
  EXPECT_EQ(count_subfloor_ghosts(fine), 0);
  const ConstArray4 values = fine.fab(0).const_array();
  EXPECT_GE(values(3, 6, 0), floor);
  EXPECT_NEAR(values(3, 6, 1), Real(0.5), Real(1e-12));
}
