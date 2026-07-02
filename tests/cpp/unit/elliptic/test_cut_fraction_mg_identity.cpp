// Bit-identite du REFACTOR cut_fraction au niveau GeometricMG (chantier T5-PR1).
//
// Le refactor a remplace la lambda 'cut' INLINE de GeometricMG par la primitive partagee
// detail::cut_fraction + detail::shortley_weller (cut_fraction.hpp). Ce test prouve qu'il n'y a
// AUCUN changement de comportement sur un cas de mur-disque :
//
//  (A) le champ de coefficients cut-cell ASSEMBLE par GeometricMG (5 composantes, niveau fin via
//      op_coef()) est EXACTEMENT egal (diff 0.0, operator!=) a une reference recalculee a la main
//      avec l'ANCIENNE formule inline. C'est la garantie "coef byte-identique" demandee.
//  (B) la resolution elliptique converge et le residu final est fini et < tolerance (le solveur
//      reste fonctionnel apres le refactor ; on capture aussi le residu pour comparaison eventuelle).
//
// Probleme : lap(phi) = -4 dans le disque r < R, phi = 0 sur r = R. Solution exacte R^2 - r^2.

#include <gtest/gtest.h>

#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/elliptic/eb/cut_fraction.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using namespace pops;
static constexpr double kCx = 0.5, kCy = 0.5, kR = 0.4;

// Oracle : ancienne formule inline (lambda 'cut' + 2/(axm*(axm+axp)) ...) reproduite a l'identique.
static void ref_coef(Real lc_x, Real lc_y, Real dx, Real dy,
                     const std::function<Real(Real, Real)>& ls, Real out[5]) {
  auto cut = [](Real lc, Real ln, Real h) -> Real {
    if (ln < Real(0))
      return h;
    Real th = lc / (lc - ln);
    if (th < Real(1e-3))
      th = Real(1e-3);
    if (th > Real(1))
      th = Real(1);
    return th * h;
  };
  const Real lc = ls(lc_x, lc_y);
  const Real axm = cut(lc, ls(lc_x - dx, lc_y), dx);
  const Real axp = cut(lc, ls(lc_x + dx, lc_y), dx);
  const Real aym = cut(lc, ls(lc_x, lc_y - dy), dy);
  const Real ayp = cut(lc, ls(lc_x, lc_y + dy), dy);
  out[0] = Real(2) / (axm * (axm + axp));
  out[1] = Real(2) / (axp * (axm + axp));
  out[2] = Real(2) / (aym * (aym + ayp));
  out[3] = Real(2) / (ayp * (aym + ayp));
  out[4] = Real(2) / (axm * axp) + Real(2) / (aym * ayp);
}

namespace {

// Fixture : construit UNE fois la grille cut-cell mur-disque partagee par les deux TEST (assemblage
// des coefficients (A) et resolution (B) portent sur la MEME instance mg, comme le corps historique).
class CutFractionMgIdentity : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    const int nc = 64;
    Box2D dom = Box2D::from_extents(nc, nc);
    Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
    BoxArray ba(std::vector<Box2D>{dom});
    BCRec bc;
    bc.xlo = bc.xhi = bc.ylo = bc.yhi = BCType::Dirichlet;
    std::function<Real(Real, Real)> ls = [](Real x, Real y) {
      return std::hypot(x - kCx, y - kCy) - kR;
    };
    std::function<bool(Real, Real)> active = [](Real x, Real y) {
      return std::hypot(x - kCx, y - kCy) < kR;
    };
    geom_ = new Geometry(geom);
    // (geom, ba, bc, active, replicated, min_coarse, nu1, nu2, nbottom, cut_cell=true, levelset)
    mg_ = new GeometricMG(geom, ba, bc, active, false, 2, 2, 2, 50, true, ls);
    ls_ = new std::function<Real(Real, Real)>(ls);
  }

  static void TearDownTestSuite() {
    delete mg_;
    mg_ = nullptr;
    delete geom_;
    geom_ = nullptr;
    delete ls_;
    ls_ = nullptr;
  }

  static constexpr int kNc = 64;
  static GeometricMG* mg_;
  static Geometry* geom_;
  static std::function<Real(Real, Real)>* ls_;
};

GeometricMG* CutFractionMgIdentity::mg_ = nullptr;
Geometry* CutFractionMgIdentity::geom_ = nullptr;
std::function<Real(Real, Real)>* CutFractionMgIdentity::ls_ = nullptr;

}  // namespace

// (A) coef assemble par GeometricMG == reference inline, EXACTEMENT (niveau fin, 5 composantes).
TEST_F(CutFractionMgIdentity, coef_is_byte_identical_to_inline_reference) {
  GeometricMG& mg = *mg_;
  const Geometry& geom = *geom_;
  const std::function<Real(Real, Real)>& ls = *ls_;

  const MultiFab* coef = mg.op_coef();
  ASSERT_TRUE(coef != nullptr) << "op_coef_disponible";
  const MultiFab* mask = mg.op_mask();
  ASSERT_TRUE(mask != nullptr) << "op_mask_disponible";

  const ConstArray4 c = coef->fab(0).const_array();
  const ConstArray4 m = mask->fab(0).const_array();
  const Real dx = geom.dx(), dy = geom.dy();
  Real max_diff = Real(0);
  long active_cnt = 0, mismatches = 0;
  for (int j = 0; j < kNc; ++j)
    for (int i = 0; i < kNc; ++i) {
      if (m(i, j) == Real(0)) {  // conducteur : GeometricMG met coef = 0
        for (int k = 0; k < 5; ++k)
          if (c(i, j, k) != Real(0)) {
            ++mismatches;
            max_diff = Real(1);
          }
        continue;
      }
      ++active_cnt;
      Real ref[5];
      ref_coef(geom.x_cell(i), geom.y_cell(j), dx, dy, ls, ref);
      for (int k = 0; k < 5; ++k) {
        if (c(i, j, k) != ref[k]) {  // EXACT : operator!=, aucune tolerance
          ++mismatches;
          max_diff = std::max(max_diff, std::fabs(c(i, j, k) - ref[k]));
        }
      }
    }
  std::printf("(A) coef : %ld cellules actives, %ld ecarts, max_diff=%.3e\n", active_cnt,
              mismatches, static_cast<double>(max_diff));
  EXPECT_TRUE(active_cnt > 1800) << "balayage_disque_couvert active_cnt=" << active_cnt;
  EXPECT_EQ(mismatches, 0) << "coef_byte_identique_a_la_reference_inline";
  EXPECT_EQ(max_diff, Real(0)) << "max_diff_exactement_0";
}

// (B) le solveur reste fonctionnel : residu fini, convergence sous tolerance, et la solution
// reproduit la physique attendue (R^2 - r^2) apres le refactor.
TEST_F(CutFractionMgIdentity, solver_stays_functional_and_physically_correct) {
  GeometricMG& mg = *mg_;
  const Geometry& geom = *geom_;

  mg.rhs().set_val(-4.0);
  mg.phi().set_val(0.0);
  const int cycles = mg.solve_robust(1e-10, 300);
  const Real res = mg.residual();
  std::printf("(B) solve_robust : %d cycles, residu final = %.3e\n", cycles,
              static_cast<double>(res));
  EXPECT_TRUE(std::isfinite(res)) << "residu_fini";
  EXPECT_TRUE(res < Real(1e-6)) << "residu_sous_tolerance res=" << res;

  // verification physique : phi reproduit R^2 - r^2 a l'ordre 2 (le refactor n'a pas casse la
  // solution).
  const ConstArray4 p = mg.phi().fab(0).const_array();
  const Real dx = geom.dx();
  double l2 = 0;
  long cnt = 0;
  for (int j = 0; j < kNc; ++j)
    for (int i = 0; i < kNc; ++i) {
      const double x = (i + 0.5) * dx, y = (j + 0.5) * dx;
      const double r2 = (x - kCx) * (x - kCx) + (y - kCy) * (y - kCy);
      if (r2 < kR * kR) {
        const double e = p(i, j) - (kR * kR - r2);
        l2 += e * e;
        ++cnt;
      }
    }
  l2 = std::sqrt(l2 / cnt);
  std::printf("    erreur L2 vs R^2 - r^2 = %.3e\n", l2);
  EXPECT_TRUE(l2 < 1e-3) << "solution_physique_correcte l2=" << l2;
}
