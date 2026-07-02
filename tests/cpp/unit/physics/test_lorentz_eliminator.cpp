// Tests analytiques de LorentzEliminator.
//
// Proprietes verifiees :
//   1. Round-trip : apply_Binv(apply_B(v)) == v a la precision machine.
//   2. Coherence de l'inverse : les entrees binv_ij reproduisent le resultat
//      de apply_Binv (verification directe par formule fermee).
//   3. Sanite rotation : pour un champ B_z pur et theta=dt=1,
//      le module de B*v est ||(vx,vy)|| * sqrt(1+w^2) = sqrt(det) * ||v||,
//      et B^{-1} ramene au vecteur initial (norme conservee par B^{-1} o B).
//   4. Cas degenere B_z=0 : B=I, B^{-1}=I, det=1.

#include <gtest/gtest.h>

#include <pops/numerics/linalg/lorentz_eliminator.hpp>

#include <cmath>

using pops::LorentzEliminator;
using pops::Real;

// Valeur absolue sans appel flottant conditionnel hors std::fabs.
static inline double dabs(double x) {
  return x < 0.0 ? -x : x;
}

// Seuil de tolerances : on attend une precision proche de epsilon machine (2.2e-16 en double).
static constexpr double EPS_MACHINE = 1e-14;

// --- cas de test (theta, dt, B_z, vx, vy) ---
struct Case {
  Real theta, dt, Bz, vx, vy;
  const char* name;
};
static const Case kCases[] = {
    {1.0, 0.1, 1.0, 1.0, 0.0, "theta=1 dt=0.1 Bz=1 vx=1 vy=0"},
    {0.5, 0.2, 2.5, 3.0, -1.5, "theta=0.5 dt=0.2 Bz=2.5"},
    {1.0, 1.0, 10.0, 0.7, 0.3, "strong w=10"},
    {0.75, 0.05, 0.01, 100.0, -50.0, "weak field small dt"},
    {1.0, 0.01, 1e6, 1.0, 1.0, "tres grand w"},
};

// Test 1 : round-trip apply_Binv(apply_B(v)) == v
TEST(test_lorentz_eliminator, RoundTripBinvOfB) {
  for (const auto& c : kCases) {
    LorentzEliminator le(c.theta, c.dt, c.Bz);
    Real Bx, By;
    le.apply_B(c.vx, c.vy, Bx, By);
    Real vxp, vyp;
    le.apply_Binv(Bx, By, vxp, vyp);
    // erreur relative (ou absolue si la norme est petite)
    const double nv = std::sqrt(c.vx * c.vx + c.vy * c.vy);
    const double scale = nv > 1e-12 ? nv : 1.0;
    const double err = std::sqrt((vxp - c.vx) * (vxp - c.vx) + (vyp - c.vy) * (vyp - c.vy));
    EXPECT_TRUE(err / scale < EPS_MACHINE) << "round-trip [" << c.name << "]";
  }
}

// Test 2 : les entrees binv_ij correspondent au resultat de apply_Binv
//          (formule fermee : B^{-1} v = (1/det)*(vx+w*vy, vy-w*vx))
TEST(test_lorentz_eliminator, BinvEntriesMatchApplyBinv) {
  for (const auto& c : kCases) {
    LorentzEliminator le(c.theta, c.dt, c.Bz);
    Real vxp, vyp;
    le.apply_Binv(c.vx, c.vy, vxp, vyp);
    // recalcul direct par les entrees
    const Real vxp2 = le.binv_11() * c.vx + le.binv_12() * c.vy;
    const Real vyp2 = le.binv_21() * c.vx + le.binv_22() * c.vy;
    // binv_ij et apply_Binv evaluent la MEME formule avec une association FP differente
    // ((vx+w*vy)*inv vs binv_11*vx+binv_12*vy) : egalite a quelques ULP, PAS bit-exacte
    // (l'ecart depend du compilateur/FMA -> tolerance RELATIVE, pas absolue).
    const Real scale = Real(1) + dabs(vxp) + dabs(vyp);
    EXPECT_TRUE(dabs(vxp2 - vxp) < EPS_MACHINE * scale && dabs(vyp2 - vyp) < EPS_MACHINE * scale)
        << "binv_ij vs apply_Binv [" << c.name << "]";
  }
}

// Test 3 : sanite rotation
//   ||B v||^2 = (1+w^2) * ||v||^2 = det * ||v||^2
//   Apres apply_Binv(apply_B(v)), on retrouve v -> norme conservee par l'aller-retour.
TEST(test_lorentz_eliminator, RotationSanityNormPreserved) {
  for (const auto& c : kCases) {
    LorentzEliminator le(c.theta, c.dt, c.Bz);
    Real Bx, By;
    le.apply_B(c.vx, c.vy, Bx, By);
    const double norm_v2 = c.vx * c.vx + c.vy * c.vy;
    const double norm_Bv2 = Bx * Bx + By * By;
    // ||B v||^2 == det * ||v||^2
    const double expected = le.det * norm_v2;
    const double scale = expected > 1e-30 ? expected : 1.0;
    EXPECT_TRUE(dabs(norm_Bv2 - expected) / scale < EPS_MACHINE) << "rotation-norme [" << c.name
                                                                  << "]";
  }
}

// Test 4 : cas degenere B_z = 0 -> B = I, B^{-1} = I, det = 1
TEST(test_lorentz_eliminator, DegenerateBzZeroIsIdentity) {
  LorentzEliminator le(1.0, 0.1, 0.0);
  EXPECT_TRUE(dabs(le.w) < EPS_MACHINE) << "Bz=0 : w==0";
  EXPECT_TRUE(dabs(le.det - 1.0) < EPS_MACHINE) << "Bz=0 : det==1";
  Real Bx, By;
  le.apply_B(3.0, -2.0, Bx, By);
  EXPECT_TRUE(dabs(Bx - 3.0) < EPS_MACHINE && dabs(By + 2.0) < EPS_MACHINE) << "Bz=0 : B==I";
  Real vxp, vyp;
  le.apply_Binv(3.0, -2.0, vxp, vyp);
  EXPECT_TRUE(dabs(vxp - 3.0) < EPS_MACHINE && dabs(vyp + 2.0) < EPS_MACHINE) << "Bz=0 : Binv==I";
  EXPECT_TRUE(dabs(le.binv_11() - 1.0) < EPS_MACHINE) << "Bz=0 : binv_11==1";
  EXPECT_TRUE(dabs(le.binv_12()) < EPS_MACHINE) << "Bz=0 : binv_12==0";
  EXPECT_TRUE(dabs(le.binv_21()) < EPS_MACHINE) << "Bz=0 : binv_21==0";
  EXPECT_TRUE(dabs(le.binv_22() - 1.0) < EPS_MACHINE) << "Bz=0 : binv_22==1";
}
