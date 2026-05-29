// Branche la Pile A (clustering Berger-Rigoutsos) sur le pas multi-patch valide :
// tag_cells -> grow_tags -> berger_rigoutsos -> patchs multi-box -> fine MultiFab ->
// amr_step_2level_multipatch. Deux blobs separes -> BR doit produire >= 2 patchs
// (multi-patch reel, pas une box unique). On verifie : BR clusterise (>=2 boxes), le
// pas tourne, la masse est conservee (reflux coverage-aware), positivite/finitude.
// C'est "la Pile A enfin branchee" sur l'integrateur conservatif.

#include <adc/amr/cluster.hpp>      // berger_rigoutsos, ClusterParams
#include <adc/amr/regrid.hpp>       // tag_cells, grow_tags
#include <adc/amr/tag_box.hpp>      // TagBox
#include <adc/integrator/amr_reflux_mf.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/model/diocotron.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace adc;
static constexpr double kPi = 3.14159265358979323846;

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) { std::printf("FAIL %s\n", w); ++fails; }
  };

  const int nc = 48;
  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const double dxc = geom.dx(), dyc = geom.dy();
  DistributionMapping dm(1, n_ranks());
  BoxArray bac(std::vector<Box2D>{dom});

  Diocotron model;
  model.B0 = 1.0;
  model.n_i0 = 1.0;
  const double gx = 0.4, gy = -0.25;  // aux uniforme -> advection
  // deux blobs bien separes (un hole au milieu -> BR coupe en 2 patchs)
  auto ne0 = [&](double x, double y) {
    const double r1 = (x - 0.3) * (x - 0.3) + (y - 0.5) * (y - 0.5);
    const double r2 = (x - 0.7) * (x - 0.7) + (y - 0.5) * (y - 0.5);
    return 1.0 + 0.6 * std::exp(-r1 / 0.004) + 0.6 * std::exp(-r2 / 0.004);
  };

  MultiFab Uc(bac, dm, 1, 1), axc(bac, dm, 3, 1);
  {
    Array4 u = Uc.fab(0).array();
    const Box2D g = Uc.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) u(i, j, 0) = ne0((i + 0.5) * dxc, (j + 0.5) * dyc);
    Array4 a = axc.fab(0).array();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) { a(i, j, 0) = 0; a(i, j, 1) = gx; a(i, j, 2) = gy; }
  }

  // --- clustering Berger-Rigoutsos sur les cellules au-dessus du fond ---
  auto crit = [&](const ConstArray4& a, int i, int j) { return a(i, j, 0) > 1.10; };
  TagBox tags = tag_cells(Uc, dom, crit);
  TagBox grown = grow_tags(tags, 2, dom);
  std::vector<Box2D> patches = berger_rigoutsos(grown, ClusterParams{});

  // clamp strictement interieur (>=1 cellule de marge : le coarsen des ghosts fins
  // doit retomber dans le grossier) ; refine ratio 2 -> boxes fines.
  std::vector<Box2D> fboxes;
  for (Box2D b : patches) {
    b.lo[0] = std::max(b.lo[0], 2); b.lo[1] = std::max(b.lo[1], 2);
    b.hi[0] = std::min(b.hi[0], nc - 3); b.hi[1] = std::min(b.hi[1], nc - 3);
    if (b.hi[0] < b.lo[0] || b.hi[1] < b.lo[1]) continue;
    fboxes.push_back(Box2D{{2 * b.lo[0], 2 * b.lo[1]}, {2 * b.hi[0] + 1, 2 * b.hi[1] + 1}});
  }
  std::printf("Berger-Rigoutsos : %d patch(s) brut(s) -> %d box(es) fine(s) interieures\n",
              (int)patches.size(), (int)fboxes.size());
  chk(fboxes.size() >= 2, "BR_produit_multi_patch");
  if (fboxes.empty()) { std::printf("FAIL pas de patch\n"); return 1; }

  // fine MultiFab multi-box, rempli par interp du grossier.
  DistributionMapping dmf((int)fboxes.size(), n_ranks());
  MultiFab Uf(BoxArray(fboxes), dmf, 1, 1), axf(BoxArray(fboxes), dmf, 3, 1);
  for (int li = 0; li < Uf.local_size(); ++li) {
    Array4 uf = Uf.fab(li).array();
    const Box2D b = Uf.box(li);
    for (int j = b.lo[1]; j <= b.hi[1]; ++j)
      for (int i = b.lo[0]; i <= b.hi[0]; ++i) uf(i, j, 0) = ne0((i + 0.5) * dxc / 2, (j + 0.5) * dyc / 2);
    Array4 af = axf.fab(li).array();
    const Box2D gb = axf.fab(li).grown_box();
    for (int j = gb.lo[1]; j <= gb.hi[1]; ++j)
      for (int i = gb.lo[0]; i <= gb.hi[0]; ++i) { af(i, j, 0) = 0; af(i, j, 1) = gx; af(i, j, 2) = gy; }
  }

  const double dt = 0.25 * dxc / std::hypot(gx, gy);
  mf_average_down_multi(Uf, Uc);  // sync init : cellules couvertes = moyenne des fins
  const double m0 = sum(Uc, 0);
  bool finite = true;
  for (int s = 0; s < 20; ++s) {
    amr_step_2level_multipatch<NoSlope, RusanovFlux>(model, Uc, dom, dxc, dyc, Uf, axc, axf, dt);
    for (double v : {sum(Uc, 0)}) if (!std::isfinite(v)) finite = false;
  }
  const double drift = std::fabs(sum(Uc, 0) - m0);
  double mn = 1e300;
  const ConstArray4 u = Uc.fab(0).const_array();
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i) mn = std::min(mn, u(i, j, 0));
  std::printf("multipatch + BR (20 pas) : drift_masse=%.3e min(ne)=%.4f\n", drift, mn);
  chk(finite && mn > 0.0, "stable_positif");
  chk(drift < 1e-11, "masse_conservee_multipatch_BR");

  if (fails == 0) std::printf("OK test_amr_cluster_step\n");
  return fails == 0 ? 0 : 1;
}
