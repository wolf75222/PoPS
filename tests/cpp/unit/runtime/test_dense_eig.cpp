// pops::real_eig_minmax : extremes du spectre (parties reelles) d'un petit bloc dense.
//
// References EXACTES sans LAPACK : matrices compagnon (racines prescrites d'un polynome),
// matrices triangulaires (spectre = diagonale), similarites entieres S D S^-1 (spectre = D),
// rotations (paire complexe pure). Cas adverses : racines quasi-degenerees (conditionnement
// eps^(1/m), tolerance adaptee -- limite du PROBLEME, documentee dans l'en-tete), matrice
// fortement non normale (triangulaire a grand hors-diagonale). Contrat de repli : cap
// d'iterations a 0 -> bornes de Gershgorin (converged = false) qui ENCADRENT le vrai spectre.

#include <gtest/gtest.h>

#include <pops/numerics/linalg/dense_eig.hpp>

#include "test_harness.hpp"  // close_rel partage (comparaison relative+absolue, atol defaut 1e-12)

#include <cmath>
#include <limits>

using pops::Real;
using pops::EigBounds;
using pops::real_eig_minmax;
using pops::Spectrum;  // predicat de spectre reel/complexe (ADC-276)
using pops::real_spectrum;
using pops::test::close_rel;  // comparaison relative+absolue partagee (atol defaut 1e-12)

/// Matrice compagnon (premiere ligne = -coefficients) du polynome unitaire de racines @p roots :
/// spectre exactement {roots}. p(x) = prod (x - r_k) developpe par produits successifs.
template <int N>
static void companion(const Real (&roots)[N], Real (&A)[N][N]) {
  Real c[N + 1];  // coefficients de p, c[0] = terme dominant 1
  c[0] = Real(1);
  for (int k = 0; k < N; ++k)
    c[k + 1] = Real(0);
  for (int k = 0; k < N; ++k)
    for (int j = k + 1; j >= 1; --j)
      c[j] -= roots[k] * c[j - 1];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      A[i][j] = Real(0);
  for (int j = 0; j < N; ++j)
    A[0][j] = -c[j + 1];
  for (int i = 1; i < N; ++i)
    A[i][i - 1] = Real(1);
}

/// Consommateur DEVICE-SAFE (pile uniquement, ni NumPy ni MATLAB) : tient lieu du projecteur
/// HyQMOM15 qui classe un bloc 3x3 de moments puis choisit une action. Le switch est EXHAUSTIF sur
/// pops::Spectrum -- kUnknown (non-convergence) y est traite explicitement, jamais confondu avec kReal.
POPS_HD static int classify_action(const Real (&B)[3][3]) {
  switch (pops::real_spectrum(B)) {
    case Spectrum::kReal:
      return 0;
    case Spectrum::kComplexPair:
      return 1;
    case Spectrum::kUnknown:
      return 2;
  }
  return -1;  // injoignable : enumere clos
}

// --- helpers pour pops::roe_abs_apply (ADC-368) : reference |A| = R |Lambda| R^T par construction ---
template <int N>
static void roe_matmul(const Real (&A)[N][N], const Real (&B)[N][N], Real (&C)[N][N]) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      Real s = 0;
      for (int k = 0; k < N; ++k)
        s += A[i][k] * B[k][j];
      C[i][j] = s;
    }
}
template <int N>
static void roe_matvec(const Real (&A)[N][N], const Real (&x)[N], Real (&y)[N]) {
  for (int i = 0; i < N; ++i) {
    Real s = 0;
    for (int j = 0; j < N; ++j)
      s += A[i][j] * x[j];
    y[i] = s;
  }
}
// Rotation de Givens a DROITE (R <- R*G), pour batir un R orthogonal.
template <int N>
static void roe_givens_right(Real (&R)[N][N], int p, int q, Real c, Real s) {
  for (int i = 0; i < N; ++i) {
    const Real a = R[i][p], b = R[i][q];
    R[i][p] = c * a - s * b;
    R[i][q] = s * a + c * b;
  }
}
// A = R diag(lam) R^T (R orthogonal) : spectre reel exact lam ; |A| = R diag(|lam|) R^T. On verifie
// roe_abs_apply(A, dU) == |A| dU a la precision machine (le matrix-sign reproduit R |Lambda| R^-1).
template <int N>
static Real roe_abs_sym_err(const Real (&lam)[N]) {
  Real R[N][N] = {};
  for (int i = 0; i < N; ++i)
    R[i][i] = 1;
  for (int p = 0; p < N - 1; ++p) {
    const Real th = Real(0.3) + Real(0.17) * p;
    roe_givens_right(R, p, p + 1, std::cos(th), std::sin(th));
    if (p + 2 < N)
      roe_givens_right(R, p, p + 2, std::cos(Real(0.21) * p + Real(0.1)),
                       std::sin(Real(0.21) * p + Real(0.1)));
  }
  Real Rt[N][N], D[N][N] = {}, Dabs[N][N] = {}, tmp[N][N], A[N][N], Aabs[N][N];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      Rt[i][j] = R[j][i];
  for (int i = 0; i < N; ++i) {
    D[i][i] = lam[i];
    Dabs[i][i] = std::fabs(lam[i]);
  }
  roe_matmul(R, D, tmp);
  roe_matmul(tmp, Rt, A);
  roe_matmul(R, Dabs, tmp);
  roe_matmul(tmp, Rt, Aabs);
  Real dU[N];
  for (int i = 0; i < N; ++i)
    dU[i] = Real(1) + Real(0.37) * i - Real(0.11) * (i % 3);
  Real ref[N];
  roe_matvec(Aabs, dU, ref);
  Real out[N];
  if (!pops::roe_abs_apply(A, dU, out))
    return Real(1e30);
  Real m = 0;
  for (int i = 0; i < N; ++i) {
    const Real d = std::fabs(out[i] - ref[i]);
    if (d > m)
      m = d;
  }
  return m;
}

// Formes fermees N = 1, 2 : spectre trivial / reel exact / rotation pure (Re = 0, max_im = 2).
TEST(DenseEig, ClosedFormN1N2) {
  {
    const Real A1[1][1] = {{Real(-3.5)}};
    const EigBounds b = real_eig_minmax(A1);
    EXPECT_TRUE(b.converged && b.lmin == Real(-3.5) && b.lmax == Real(-3.5) && b.max_im == Real(0))
        << "N=1 : spectre trivial";
  }
  {
    const Real A2[2][2] = {{Real(1), Real(2)}, {Real(3), Real(0)}};  // lambda = 3, -2
    const EigBounds b = real_eig_minmax(A2);
    EXPECT_TRUE(b.converged && close_rel(b.lmin, Real(-2), 1e-14) &&
                close_rel(b.lmax, Real(3), 1e-14))
        << "N=2 reel : {-2, 3} exact";
  }
  {
    const Real R[2][2] = {{Real(0), Real(2)}, {Real(-2), Real(0)}};  // lambda = +-2i
    const EigBounds b = real_eig_minmax(R);
    EXPECT_TRUE(b.converged && close_rel(b.lmin, Real(0), 1e-14) &&
                close_rel(b.lmax, Real(0), 1e-14) && close_rel(b.max_im, Real(2), 1e-14))
        << "N=2 rotation : Re = 0, max_im = 2 (indicateur d'hyperbolicite)";
  }
}

// N = 3, 4, 5 : racines prescrites (compagnon), rtol 1e-10.
TEST(DenseEig, CompanionRootsN3N4N5) {
  {
    const Real roots[3] = {Real(1), Real(2), Real(3)};
    Real A[3][3];
    companion(roots, A);
    const EigBounds b = real_eig_minmax(A);
    EXPECT_TRUE(b.converged && close_rel(b.lmin, Real(1), 1e-10) &&
                close_rel(b.lmax, Real(3), 1e-10) && b.max_im < Real(1e-8))
        << "N=3 compagnon {1,2,3}";
  }
  {
    const Real roots[4] = {Real(-2), Real(-0.5), Real(1), Real(4)};
    Real A[4][4];
    companion(roots, A);
    const EigBounds b = real_eig_minmax(A);
    EXPECT_TRUE(b.converged && close_rel(b.lmin, Real(-2), 1e-10) &&
                close_rel(b.lmax, Real(4), 1e-10))
        << "N=4 compagnon {-2,-0.5,1,4}";
  }
  {
    const Real roots[5] = {Real(-3), Real(-1), Real(0), Real(2), Real(5)};
    Real A[5][5];
    companion(roots, A);
    const EigBounds b = real_eig_minmax(A);
    EXPECT_TRUE(b.converged && close_rel(b.lmin, Real(-3), 1e-10) &&
                close_rel(b.lmax, Real(5), 1e-10))
        << "N=5 compagnon {-3,-1,0,2,5} (taille des blocs HyQMOM, sans en dependre)";
  }
}

// Similarite dense S D S^-1 (spectre exact, matrice pleine).
TEST(DenseEig, DenseSimilarityExactSpectrum) {
  {
    // S unitriangulaire entiere, D = diag(-1, 0.5, 2, 7) : A = S D S^-1 est PLEINE et son
    // spectre est exactement D (similarite). Calcul de A a la main : S D S^-1 avec
    // S = [[1,0,0,0],[1,1,0,0],[0,1,1,0],[1,0,1,1]], S^-1 entiere aussi (det 1).
    const Real D[4] = {Real(-1), Real(0.5), Real(2), Real(7)};
    const Real S[4][4] = {{1, 0, 0, 0}, {1, 1, 0, 0}, {0, 1, 1, 0}, {1, 0, 1, 1}};
    const Real Sinv[4][4] = {{1, 0, 0, 0}, {-1, 1, 0, 0}, {1, -1, 1, 0}, {-2, 1, -1, 1}};
    Real A[4][4];
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        A[i][j] = Real(0);
        for (int k = 0; k < 4; ++k)
          A[i][j] += S[i][k] * D[k] * Sinv[k][j];
      }
    const EigBounds b = real_eig_minmax(A);
    EXPECT_TRUE(b.converged && close_rel(b.lmin, Real(-1), 1e-10) &&
                close_rel(b.lmax, Real(7), 1e-10))
        << "N=4 similarite dense : {-1, 7}";
  }
}

// N = 5 PLEINE par conjugaison orthogonale (exerce hessenberg_reduce).
TEST(DenseEig, DenseOrthogonalConjugationN5) {
  {
    // P = I - 2 v v^T / (v^T v) (reflecteur, P = P^-1 exactement) ; A = P C P a le spectre de C
    // mais est DENSE : la reduction de Householder travaille sur toutes ses colonnes (la matrice
    // compagnon, deja Hessenberg, ne l'exercait qu'a vide).
    const Real roots[5] = {Real(-3), Real(-1), Real(0), Real(2), Real(5)};
    Real C[5][5];
    companion(roots, C);
    const Real v[5] = {1, 2, -1, 3, 1};
    Real vv = 0;
    for (int i = 0; i < 5; ++i)
      vv += v[i] * v[i];
    Real P[5][5], T[5][5], A[5][5];
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j)
        P[i][j] = (i == j ? Real(1) : Real(0)) - 2 * v[i] * v[j] / vv;
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j) {
        T[i][j] = 0;
        for (int k = 0; k < 5; ++k)
          T[i][j] += P[i][k] * C[k][j];
      }
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j) {
        A[i][j] = 0;
        for (int k = 0; k < 5; ++k)
          A[i][j] += T[i][k] * P[k][j];
      }
    const EigBounds b = real_eig_minmax(A);
    EXPECT_TRUE(b.converged && close_rel(b.lmin, Real(-3), 1e-10) &&
                close_rel(b.lmax, Real(5), 1e-10))
        << "N=5 dense (P C P) : spectre {-3..5} conserve, Householder multi-etapes exerce";
  }
}

// N = 8 : genericite au-dela des tailles HyQMOM.
TEST(DenseEig, CompanionRootsN8) {
  const Real roots[8] = {Real(-7), Real(-5), Real(-2), Real(-1),
                         Real(1),  Real(3),  Real(6),  Real(9)};
  Real A[8][8];
  companion(roots, A);
  const EigBounds b = real_eig_minmax(A);
  EXPECT_TRUE(b.converged && close_rel(b.lmin, Real(-7), 1e-9) && close_rel(b.lmax, Real(9), 1e-9))
      << "N=8 compagnon {-7..9} (rtol 1e-9 : compagnon de degre 8)";
}

// Spectre mixte reel + paire complexe.
TEST(DenseEig, MixedRealAndComplexPairSpectrum) {
  // Bloc diagonal : rotation 2x2 (lambda = +-2i) et diag(-1, 3), plonge dans une similarite.
  Real A[4][4] = {{0, 2, 1, 0}, {-2, 0, 0, 1}, {0, 0, -1, 5}, {0, 0, 0, 3}};
  const EigBounds b = real_eig_minmax(A);
  EXPECT_TRUE(b.converged && close_rel(b.lmin, Real(-1), 1e-10) &&
              close_rel(b.lmax, Real(3), 1e-10) && close_rel(b.max_im, Real(2), 1e-10))
      << "N=4 mixte : Re dans {-1, 0, 3}, max_im = 2";
}

// Cas adverses : racines groupees (quasi-degenerees) et matrice fortement non normale.
TEST(DenseEig, AdversarialGroupedRootsAndNonNormalMatrix) {
  {
    // Racines GROUPEES {1-1e-3, 1, 1+1e-3} (separees, pas une vraie multiplicite : l'argument
    // eps^(1/m) ne s'applique qu'a la limite confondue ; ici l'erreur observee est bien moindre).
    // Tolerance ABSOLUE 1e-5 : marge volontaire couvrant la degradation a l'approche du groupe.
    const Real roots[3] = {Real(1) - Real(1e-3), Real(1), Real(1) + Real(1e-3)};
    Real A[3][3];
    companion(roots, A);
    const EigBounds b = real_eig_minmax(A);
    EXPECT_TRUE(b.converged && std::fabs(b.lmin - (Real(1) - Real(1e-3))) < Real(1e-5) &&
                std::fabs(b.lmax - (Real(1) + Real(1e-3))) < Real(1e-5))
        << "N=3 racines groupees a 1e-3 (tolerance large volontaire)";
  }
  {
    // Fortement non normale : triangulaire a hors-diagonale 1e6 -- spectre = diagonale, exact.
    const Real A[3][3] = {{Real(1), Real(1e6), Real(1e6)},
                          {Real(0), Real(2), Real(1e6)},
                          {Real(0), Real(0), Real(-3)}};
    const EigBounds b = real_eig_minmax(A);
    EXPECT_TRUE(b.converged && close_rel(b.lmin, Real(-3), 1e-10) &&
                close_rel(b.lmax, Real(2), 1e-10))
        << "N=3 non normale (hors-diagonale 1e6) : spectre = diagonale";
  }
}

// Bloc compagnon quasi-degenere (cap defaut converge, pas de repli).
TEST(DenseEig, QuasiDegenerateCompanionConvergesAtDefaultCap) {
  {
    // Bloc compagnon 5x5 reel d'un cas HyQMOM : superdiagonale de 1, derniere ligne = coefficients.
    // Spectre quasi-double (paires ~+-1.7326 et ~+-1.7527 + ~0.01) : la deflation QR rampe et
    // demande ~42 iterations. Sous l'ancien cap 30 ce bloc repliait en silence (Gershgorin ~+-15.6,
    // vitesse d'onde sur-estimee ~9x) ; le defaut a 100 le fait converger avec marge.
    // Reference numpy (np.linalg.eigvals) : min Re = -1.732589689893011, max Re = 1.752707143107345.
    Real A[5][5];
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j)
        A[i][j] = Real(0);
    for (int i = 0; i < 4; ++i)
      A[i][i + 1] = Real(1);
    const Real last[5] = {Real(0.0927583829495191), Real(-9.220453484757002),
                          Real(-0.18326928704092538), Real(6.072635227251581),
                          Real(0.05029363303583967)};
    for (int j = 0; j < 5; ++j)
      A[4][j] = last[j];
    bool fb = true;  // doit etre remis a false : aucun repli attendu au cap defaut
    const EigBounds b = real_eig_minmax(A, /*max_iter_per_eig=*/100, &fb);
    // Tolerance ABSOLUE 1e-6 (et non 1e-9) : la paire superieure est QUASI-DOUBLE, son
    // conditionnement non symetrique est ~eps^(1/2) (~1.5e-8) -- exiger 1e-9 contredirait le
    // contrat documente dans l'en-tete. 1e-6 reste a 7 ordres de grandeur du repli (~+-15.6).
    EXPECT_TRUE(b.converged && !fb)
        << "compagnon quasi-degenere : converge au cap defaut, fallback = false";
    EXPECT_TRUE(std::fabs(b.lmin - Real(-1.732589689893011)) < Real(1e-6) &&
                std::fabs(b.lmax - Real(1.752707143107345)) < Real(1e-6))
        << "min/max corrects (vs numpy) : pas le repli Gershgorin";
    EXPECT_TRUE(b.max_im < Real(1e-6)) << "spectre essentiellement reel (max_im ~ 0)";
    // Verrou du DEFAUT : meme bloc appele SANS cap explicite (donc avec le defaut de la signature).
    // Ce bloc demande ~42 iterations ; si le defaut regressait sous ce seuil (p.ex. l'ancien 30) il
    // replirait en silence et bdef.converged passerait a false. Epingle le defaut a >= 42.
    const EigBounds bdef = real_eig_minmax(A);
    EXPECT_TRUE(bdef.converged && std::fabs(bdef.lmin - Real(-1.732589689893011)) < Real(1e-6) &&
                std::fabs(bdef.lmax - Real(1.752707143107345)) < Real(1e-6))
        << "cap par DEFAUT suffit a converger (une regression 100->30 ferait echouer ce test)";
  }
}

// Contrat de repli (cap = 0 -> Gershgorin).
TEST(DenseEig, FallbackContractZeroCapUsesGershgorinBounds) {
  {
    const Real roots[5] = {Real(-3), Real(-1), Real(0), Real(2), Real(5)};
    Real A[5][5];
    companion(roots, A);
    bool fb = false;  // doit passer a true : le parametre de sortie reporte le repli
    const EigBounds b = real_eig_minmax(A, /*max_iter_per_eig=*/0, &fb);
    // Gershgorin ENCADRE le vrai spectre [-3, 5] et le flag dit la verite.
    EXPECT_TRUE(!b.converged && fb && b.lmin <= Real(-3) && b.lmax >= Real(5))
        << "cap 0 : converged = false, fallback = true, bornes de Gershgorin englobantes";
    Real glo, ghi;
    pops::detail::gershgorin_bounds(A, glo, ghi);
    EXPECT_TRUE(b.lmin == glo && b.lmax == ghi)
        << "le repli EST la borne de Gershgorin (contrat documente)";
  }
}

// Predicat de spectre reel/complexe (ADC-276).
TEST(DenseEig, RealComplexSpectrumPredicate) {
  {
    // (a) 3 reels distincts {1,2,3} -> kReal ; all_real vrai, has_complex_pair faux.
    const Real roots[3] = {Real(1), Real(2), Real(3)};
    Real A[3][3];
    companion(roots, A);
    const EigBounds b = real_eig_minmax(A);
    EXPECT_TRUE(real_spectrum(A) == Spectrum::kReal && b.all_real() && !b.has_complex_pair() &&
                classify_action(A) == 0)
        << "3x3 reels distincts {1,2,3} : kReal (all_real, !has_complex_pair)";
  }
  {
    // (b) Racine reelle DOUBLE {2,2,5} : le compagnon est defectif (conditionnement ~eps^(1/2),
    // d'ou un petit Im parasite) -- la tolerance RELATIVE l'absorbe ; ne doit PAS passer kComplexPair.
    const Real roots[3] = {Real(2), Real(2), Real(5)};
    Real A[3][3];
    companion(roots, A);
    const EigBounds b = real_eig_minmax(A);
    EXPECT_TRUE(real_spectrum(A) == Spectrum::kReal && b.all_real() && classify_action(A) == 0)
        << "3x3 racine double {2,2,5} : kReal (Im parasite absorbe par la tolerance relative)";
  }
  {
    // (b2) Racine reelle TRIPLE {2,2,2} : compagnon MAXIMALEMENT defectif (bloc de Jordan 3x3), le PIRE
    // cas d'un bloc 3x3 ; conditionnement ~eps^(1/3) ~ 6e-6 -> |Im| parasite relatif ~5e-6. Le defaut
    // 1e-5 le couvre (eps^(1/3) < 1e-5) donc spectre REEL. Cas "racines multiples" exige par ADC-276 ;
    // un defaut a 1e-7 (eps^(1/2) seulement) le classait kComplexPair a tort -- c'est la regression que
    // ce test verrouille.
    const Real roots[3] = {Real(2), Real(2), Real(2)};
    Real A[3][3];
    companion(roots, A);
    const EigBounds b = real_eig_minmax(A);
    EXPECT_TRUE(real_spectrum(A) == Spectrum::kReal && b.all_real() && classify_action(A) == 0)
        << "3x3 racine TRIPLE {2,2,2} : kReal au defaut (eps^(1/3) couvert, pas un faux "
           "kComplexPair)";
  }
  {
    // (b3) Racine QUADRUPLE {1,1,1,1} (4x4) : conditionnement ~eps^(1/4) ~ 1.2e-4 > defaut 1e-5, donc AU
    // DEFAUT le bloc est rapporte kComplexPair (LIMITE documentee : le defaut couvre m<=3, le 3x3 vise).
    // Avec une tolerance ELARGIE (~eps^(1/4)) le spectre reel est correctement kReal : le bouton im_tol
    // etend la couverture aux multiplicites superieures / aux blocs plus grands.
    const Real roots[4] = {Real(1), Real(1), Real(1), Real(1)};
    Real A[4][4];
    companion(roots, A);
    EXPECT_TRUE(real_spectrum(A) == Spectrum::kComplexPair &&
                real_spectrum(A, /*im_tol=*/1e-3) == Spectrum::kReal)
        << "4x4 racine QUADRUPLE : kComplexPair au defaut (m>3), kReal avec im_tol elargi (1e-3)";
  }
  {
    // (c) Paire complexe conjuguee + un reel : diag-bloc rotation(+-2i) et {3}. kComplexPair ;
    // has_complex_pair vrai, all_real faux, max_im ~ 2 (temoin de perte d'hyperbolicite).
    const Real A[3][3] = {
        {Real(0), Real(2), Real(0)}, {Real(-2), Real(0), Real(0)}, {Real(0), Real(0), Real(3)}};
    const EigBounds b = real_eig_minmax(A);
    EXPECT_TRUE(real_spectrum(A) == Spectrum::kComplexPair && b.has_complex_pair() &&
                !b.all_real() && close_rel(b.max_im, Real(2), 1e-12) && classify_action(A) == 1)
        << "3x3 paire complexe + reel : kComplexPair (has_complex_pair, max_im = 2)";
  }
  {
    // (d) ANTI-PIEGE de non-convergence : cap d'iterations 0 -> repli Gershgorin (converged=false,
    // max_im=0 PAR CONVENTION). Le predicat doit dire kUnknown, JAMAIS kReal, et les deux booleens
    // doivent etre FAUX (un bloc non converge n'est ni reel ni complexe : on ne sait pas).
    const Real roots[5] = {Real(-3), Real(-1), Real(0), Real(2), Real(5)};
    Real A[5][5];
    companion(roots, A);
    const EigBounds b = real_eig_minmax(A, /*max_iter_per_eig=*/0);
    EXPECT_TRUE(!b.converged &&
                real_spectrum(A, /*im_tol=*/1e-7, /*max_iter_per_eig=*/0) == Spectrum::kUnknown &&
                !b.all_real() && !b.has_complex_pair() && std::isnan(b.real_status()))
        << "non-convergence : kUnknown, all_real ET has_complex_pair faux (max_im=0 jamais lu "
           "reel)";
  }
  {
    // (e) Rotation pure : parties reelles nulles (echelle spectrale 0), Im = 2 reel. Le plancher
    // max(rho,1) empeche le seuil de s'effondrer a 0 -> kComplexPair correct (pas un faux kReal).
    const Real R[2][2] = {{Real(0), Real(2)}, {Real(-2), Real(0)}};
    const EigBounds b = real_eig_minmax(R);
    EXPECT_TRUE(real_spectrum(R) == Spectrum::kComplexPair && b.has_complex_pair() && !b.all_real())
        << "rotation pure (echelle 0) : kComplexPair (plancher max(rho,1) ne s'effondre pas)";
  }
  {
    // (f) Matrice nulle : converge, max_im = 0 -> kReal (le plancher gere l'echelle 0 sans faux complexe).
    const Real Z[3][3] = {
        {Real(0), Real(0), Real(0)}, {Real(0), Real(0), Real(0)}, {Real(0), Real(0), Real(0)}};
    const EigBounds b = real_eig_minmax(Z);
    EXPECT_TRUE(real_spectrum(Z) == Spectrum::kReal && b.all_real() && classify_action(Z) == 0)
        << "matrice nulle : kReal (max_im = 0, plancher gere l'echelle nulle)";
  }
  {
    // Une partition HLL peut contenir un bloc 1x1. NaN/Inf ne sont jamais un spectre reel : sinon
    // leurs comparaisons min/max seraient toutes fausses et un bloc voisin pourrait masquer l'erreur.
    const Real nan_block[1][1] = {{std::numeric_limits<Real>::quiet_NaN()}};
    const Real inf_block[1][1] = {{std::numeric_limits<Real>::infinity()}};
    const EigBounds nan_bounds = real_eig_minmax(nan_block);
    const EigBounds inf_bounds = real_eig_minmax(inf_block);
    EXPECT_TRUE(!nan_bounds.all_real(Real(0)) && !inf_bounds.all_real(Real(0)) &&
                std::isnan(nan_bounds.real_status(Real(0))) &&
                std::isnan(inf_bounds.real_status(Real(0))));
  }
  {
    // (g) BOUTON de tolerance, valeurs EXACTES (chemin ferme N=2) : [[1,-1e-6],[1e-6,1]] a pour VP
    // 1 +- 1e-6 i, donc max_im = 1e-6 EXACT, echelle = max(1,1,1) = 1. Lache (1e-5) -> kReal ;
    // serre (1e-7) -> kComplexPair. Le meme bloc bascule selon la tolerance : le bouton est explicite.
    const Real A[2][2] = {{Real(1), Real(-1e-6)}, {Real(1e-6), Real(1)}};
    const EigBounds b = real_eig_minmax(A);
    EXPECT_TRUE(close_rel(b.max_im, Real(1e-6), 1e-12) && b.all_real(/*im_tol=*/1e-5) &&
                !b.all_real(/*im_tol=*/1e-7) &&
                real_spectrum(A, /*im_tol=*/1e-5) == Spectrum::kReal &&
                real_spectrum(A, /*im_tol=*/1e-7) == Spectrum::kComplexPair)
        << "bouton tolerance (max_im=1e-6 exact) : kReal a 1e-5, kComplexPair a 1e-7";
  }
  {
    // (h) Pourquoi RELATIF et non absolu (valeurs exactes, N=2) : [[1e3,-1e-6],[1e-6,1e3]] a pour VP
    // 1e3 +- 1e-6 i, max_im = 1e-6 EXACT mais echelle = 1e3. En relatif a 1e-9 le seuil vaut
    // 1e-9*1e3 = 1e-6 >= max_im -> kReal (correct) ; une tolerance ABSOLUE 1e-9 le classerait
    // complexe a tort. La mise a l'echelle suit la magnitude du spectre.
    const Real A[2][2] = {{Real(1e3), Real(-1e-6)}, {Real(1e-6), Real(1e3)}};
    const EigBounds b = real_eig_minmax(A);
    // Le test passe parce que 1e-9*1e3 s'arrondit a 1.0000000000000002e-06 >= max_im (1e-6 exact) :
    // c'est la frontiere <= (non stricte) ; un futur passage a < ferait basculer ce cas.
    EXPECT_TRUE(close_rel(b.max_im, Real(1e-6), 1e-12) && b.all_real(/*im_tol=*/1e-9) &&
                real_spectrum(A, /*im_tol=*/1e-9) == Spectrum::kReal)
        << "tolerance RELATIVE : max_im=1e-6 a l'echelle 1e3 -> kReal a 1e-9 (un absolu mediterait "
           "complexe)";
  }
  {
    // (i) ASYMETRIE assumee de la tolerance RELATIVE (valeurs exactes, N=2) : une VRAIE paire complexe
    // dont |Im| est PETIT devant la magnitude reelle est rapportee kReal PAR CONSTRUCTION. [[1e8,-9],
    // [9,1e8]] a pour VP 1e8 +- 9i, max_im = 9 exact, echelle = 1e8, seuil defaut 1e-5*1e8 = 1e3 :
    // 9 <= 1e3 -> kReal (oscillation negligeable a l'echelle 1e8). En montant |Im| au-dessus du seuil
    // (2000 > 1e3) le bloc bascule kComplexPair. Verrou : un changement de defaut deplacerait ce bord.
    const Real Areal[2][2] = {{Real(1e8), Real(-9)}, {Real(9), Real(1e8)}};
    const Real Acplx[2][2] = {{Real(1e8), Real(-2e3)}, {Real(2e3), Real(1e8)}};
    EXPECT_TRUE(close_rel(real_eig_minmax(Areal).max_im, Real(9), 1e-12) &&
                real_spectrum(Areal) == Spectrum::kReal &&
                real_spectrum(Acplx) == Spectrum::kComplexPair)
        << "echelle 1e8 : |Im|=9 -> kReal (relatif), |Im|=2000 -> kComplexPair (asymetrie "
           "documentee)";
  }
}

// roe_abs_apply : |A| dU via matrix-sign (ADC-368).
TEST(DenseEig, RoeAbsApplyMatrixSign) {
  {
    // N=1 : |A| = |a|, donc |A| dU = |a| dU.
    const Real A[1][1] = {{Real(-5)}}, dU[1] = {Real(2)};
    Real out[1];
    EXPECT_TRUE(pops::roe_abs_apply(A, dU, out) && close_rel(out[0], Real(10), 1e-12))
        << "roe_abs N=1 : A=-5 -> |A|*2 = 10";
  }
  {
    // N=2 non symetrique, VP 1 et -2 : |A| = [[1,-1/3],[0,2]] (calcul a la main), dU=(1,1) -> (2/3,2).
    const Real A[2][2] = {{Real(1), Real(1)}, {Real(0), Real(-2)}}, dU[2] = {Real(1), Real(1)};
    Real out[2];
    EXPECT_TRUE(pops::roe_abs_apply(A, dU, out) && close_rel(out[0], Real(2) / Real(3), 1e-12) &&
                close_rel(out[1], Real(2), 1e-12))
        << "roe_abs N=2 non-sym (VP 1,-2) : |A| dU = (2/3, 2)";
  }
  {
    // Symetriques a spectre mixte, N croissant jusqu'a 15 (cible HyQMOM15) : |A| dU exact a eps machine.
    const Real lam3[3] = {Real(2), Real(-3), Real(0.5)};
    const Real lam4[4] = {Real(5), Real(-1), Real(2), Real(-4)};
    const Real lam6[6] = {Real(3.1), Real(-2.2), Real(0.7), Real(-1.5), Real(4), Real(-0.3)};
    Real lam15[15];
    for (int i = 0; i < 15; ++i)
      lam15[i] = (i % 2 ? Real(-1) : Real(1)) * (Real(1) + Real(0.5) * i);
    EXPECT_TRUE(roe_abs_sym_err(lam3) < 1e-9) << "roe_abs N=3 sym mixte : |A| dU == R |L| R^T dU";
    EXPECT_TRUE(roe_abs_sym_err(lam4) < 1e-9) << "roe_abs N=4 sym mixte";
    EXPECT_TRUE(roe_abs_sym_err(lam6) < 1e-9) << "roe_abs N=6 sym mixte";
    EXPECT_TRUE(roe_abs_sym_err(lam15) < 1e-8) << "roe_abs N=15 sym mixte (cible HyQMOM15)";
  }
  {
    // Spectre tout positif : |A| = A, donc |A| dU = A dU (le sign vaut l'identite).
    const Real lam5[5] = {Real(1), Real(2), Real(3), Real(4), Real(5)};
    EXPECT_TRUE(roe_abs_sym_err(lam5) < 1e-9) << "roe_abs spectre positif : |A| = A";
  }
  {
    // Spectre COMPLEXE meme tres proche de l'axe reel : le defaut Roe strict refuse, out intact.
    const Real A[2][2] = {{Real(1), Real(-1e-6)}, {Real(1e-6), Real(1)}};
    const Real dU[2] = {Real(1), Real(0)};
    Real out[2] = {Real(7), Real(7)};
    EXPECT_TRUE(!pops::roe_abs_apply(A, dU, out) && out[0] == Real(7) && out[1] == Real(7))
        << "roe_abs spectre complexe -> false, out inchange";
    Real exact_zero_out[2] = {Real(9), Real(9)};
    EXPECT_TRUE(!pops::roe_abs_apply(A, dU, exact_zero_out, 80, Real(1e-13), Real(0)) &&
                exact_zero_out[0] == Real(9) && exact_zero_out[1] == Real(9))
        << "roe_abs tolerance imaginaire nulle : une vraie paire complexe reste refusee";
  }
  {
    // A singuliere REELLE : le projecteur du noyau definit |0|=0 et conserve la branche Roe.
    const Real A[2][2] = {{Real(0), Real(0)}, {Real(0), Real(3)}}, dU[2] = {Real(1), Real(1)};
    Real out[2];
    EXPECT_TRUE(pops::roe_abs_apply(A, dU, out) && close_rel(out[0], Real(0), 1e-12) &&
                close_rel(out[1], Real(3), 1e-12))
        << "roe_abs A singuliere reelle -> |A| dU, aucun repli de schema";
  }
  {
    // Jacobienne x de la fermeture gaussienne 2D d'ordre 2 au Maxwellien centre. Son spectre
    // {-sqrt(3), -1, 0, 0, 1, sqrt(3)} est reel avec un noyau double : le fournisseur Roe dense
    // doit conserver les deux modes nuls sans confondre multiplicite reelle et paire complexe.
    const Real A[6][6] = {{0, 1, 0, 0, 0, 0},
                          {0, 0, 1, 0, 0, 0},
                          {0, 3, 0, 0, 0, 0},
                          {0, 0, 0, 0, 1, 0},
                          {0, 0, 0, 1, 0, 0},
                          {0, 1, 0, 0, 0, 0}};
    const Real dU[6] = {1, 0, 1, 0, 0, 1};
    Real out[6] = {};
    ASSERT_TRUE(pops::roe_abs_apply(A, dU, out));
    EXPECT_TRUE(close_rel(out[0], Real(1) / std::sqrt(Real(3)), 1e-11));
    EXPECT_TRUE(close_rel(out[1], Real(0), 1e-11));
    EXPECT_TRUE(close_rel(out[2], std::sqrt(Real(3)), 1e-11));
    EXPECT_TRUE(close_rel(out[3], Real(0), 1e-11));
    EXPECT_TRUE(close_rel(out[4], Real(0), 1e-11));
    EXPECT_TRUE(close_rel(out[5], Real(1) / std::sqrt(Real(3)), 1e-11));
  }
  {
    // Vrai Jacobien y d'une face gaussienne apres un pas. Son spectre est l'union des blocs
    // triangulaires [0,3,5], [1,4], [2] et reste reel. Selon le compilateur, le QR du bloc plein
    // rapporte exactement zero ou un residu imaginaire de l'ordre de l'arrondi sur la valeur propre
    // convective double. Le contrat physique porte donc sur la tolerance stricte, jamais sur ce
    // residu d'implementation, puis la fonction spectrale agit sur le Jacobien COMPLET.
    const Real A[6][6] = {
        {0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 1, 0},
        {Real(1.4152909052530569e-05), Real(-3.5510220040414111e-09),
         Real(-1.4152367117191054e-05), Real(1.000038300781064),
         Real(-0.00012545682198025873), 0},
        {0, 0, 0, 0, 0, 1},
        {Real(6.2728348106356945e-05), Real(0.99999899792384794), 0,
         Real(-3.5510220040414111e-09), Real(-2.8304734234382107e-05),
         Real(-6.2728410990129367e-05)},
        {Real(4.2457058811993561e-05), 0, 0, Real(2.9999969937715441), 0,
         Real(-4.2457101351573168e-05)}};
    const Real dU[6] = {Real(0.0013283911625738831), Real(-2.5848588284373079e-05),
                        Real(0.0013441770644624373), Real(1.7405713276659193e-06), 0,
                        Real(0.0013245323044106527)};

    const pops::EigBounds full = pops::real_eig_minmax(A);
    ASSERT_TRUE(full.valid());
    EXPECT_TRUE(full.all_real(pops::kEigStrictImagTol));

    constexpr int block0_indices[3] = {0, 3, 5};
    constexpr int block1_indices[2] = {1, 4};
    Real block0[3][3], block1[2][2];
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 3; ++col)
        block0[row][col] = A[block0_indices[row]][block0_indices[col]];
    for (int row = 0; row < 2; ++row)
      for (int col = 0; col < 2; ++col)
        block1[row][col] = A[block1_indices[row]][block1_indices[col]];
    const Real block2[1][1] = {{A[2][2]}};
    EXPECT_TRUE(pops::real_eig_minmax(block0).all_real(Real(0)));
    EXPECT_TRUE(pops::real_eig_minmax(block1).all_real(Real(0)));
    EXPECT_TRUE(pops::real_eig_minmax(block2).all_real(Real(0)));

    Real certified_out[6] = {};
    ASSERT_TRUE(pops::detail::roe_abs_apply_certified_real(A, dU, certified_out));
    for (const Real value : certified_out)
      EXPECT_TRUE(std::isfinite(value));

    Real default_out[6] = {};
    ASSERT_TRUE(pops::roe_abs_apply(A, dU, default_out));
    for (int component = 0; component < 6; ++component)
      EXPECT_EQ(default_out[component], certified_out[component]);
  }
}

// Fonction spectrale de Harten du Roe dense : meme formule que flux_ROE_local.m, mais appliquee
// sans vecteurs propres via les projecteurs de matrix-sign decales.
TEST(DenseEig, RoeEntropyFixApplyMatchesHartenFunction) {
  const Real delta = Real(1e-3);
  {
    // Les cinq branches : negative/positive exterieures, deux interieures et VP exactement nulle.
    const Real A[5][5] = {{-Real(2) * delta, 0, 0, 0, 0},
                          {0, -Real(0.5) * delta, 0, 0, 0},
                          {0, 0, 0, 0, 0},
                          {0, 0, 0, Real(0.5) * delta, 0},
                          {0, 0, 0, 0, Real(2) * delta}};
    const Real dU[5] = {Real(1), Real(2), Real(3), Real(4), Real(5)};
    Real out[5] = {};
    ASSERT_TRUE(pops::roe_entropy_fix_apply(A, dU, out, delta));
    const Real phi_half = Real(0.5) * (Real(0.25) * delta + delta);
    EXPECT_TRUE(close_rel(out[0], Real(2) * delta * dU[0], 1e-12));
    EXPECT_TRUE(close_rel(out[1], phi_half * dU[1], 1e-12));
    EXPECT_TRUE(close_rel(out[2], Real(0.5) * delta * dU[2], 1e-12));
    EXPECT_TRUE(close_rel(out[3], phi_half * dU[3], 1e-12));
    EXPECT_TRUE(close_rel(out[4], Real(2) * delta * dU[4], 1e-12));
  }
  {
    // Similarite non orthogonale : A triangulaire, VP {0, 2 delta}. Le terme hors-diagonale de
    // Phi(A) vaut 3 delta / 4 par la difference divisee de Phi entre ces deux VP.
    const Real A[2][2] = {{Real(0), delta}, {Real(0), Real(2) * delta}};
    const Real dU[2] = {Real(1), Real(1)};
    Real out[2] = {};
    ASSERT_TRUE(pops::roe_entropy_fix_apply(A, dU, out, delta));
    EXPECT_TRUE(close_rel(out[0], Real(1.25) * delta, 1e-12));
    EXPECT_TRUE(close_rel(out[1], Real(2) * delta, 1e-12));
  }
  {
    // A la frontiere |lambda| == delta les branches MATLAB coincident. Le retry decale doit
    // produire delta, pas refuser la matrice decalee singuliere.
    const Real A[2][2] = {{-delta, 0}, {0, delta}};
    const Real dU[2] = {Real(2), Real(3)};
    Real out[2] = {};
    ASSERT_TRUE(pops::roe_entropy_fix_apply(A, dU, out, delta));
    EXPECT_TRUE(close_rel(out[0], delta * dU[0], 1e-12));
    EXPECT_TRUE(close_rel(out[1], delta * dU[1], 1e-12));
  }
  {
    // delta^2 deborde ici, alors que Phi_delta(0.9 delta) reste representable. La matrice
    // A + delta I deborderait elle aussi si elle etait formee avant la normalisation.
    const Real large_delta = Real(1e308);
    const Real A[1][1] = {{Real(0.9) * large_delta}};
    const Real dU[1] = {Real(1)};
    Real out[1] = {};
    ASSERT_TRUE(pops::roe_entropy_fix_apply(A, dU, out, large_delta));
    EXPECT_TRUE(std::isfinite(out[0]));
    EXPECT_NEAR(out[0] / large_delta, Real(0.905), Real(2e-12));
  }
  {
    // Le retry de la valeur propre exactement sur le cutoff reste representable aux deux bornes
    // de Real. En particulier, delta/2 + delta/2 ne doit pas disparaitre par sous-debordement.
    const Real cutoffs[2] = {std::numeric_limits<Real>::denorm_min(),
                             std::numeric_limits<Real>::max()};
    for (const Real cutoff : cutoffs) {
      const Real A[1][1] = {{cutoff}};
      const Real dU[1] = {Real(1)};
      Real out[1] = {};
      ASSERT_TRUE(pops::roe_entropy_fix_apply(A, dU, out, cutoff)) << "cutoff=" << cutoff;
      EXPECT_EQ(out[0], cutoff);
    }
  }
  {
    // Meme fonction spectrale sur des echelles reciproques : le calcul ne doit ni former
    // delta^2, ni evaluer sqrt(norm(inv) / norm(S)) avant la racine.
    const Real scales[2] = {Real(1e200), Real(1e-200)};
    for (const Real scale : scales) {
      const Real A[3][3] = {{-Real(0.5) * scale, 0, 0},
                            {0, 0, 0},
                            {0, 0, Real(2) * scale}};
      const Real dU[3] = {Real(2), Real(3), Real(5)};
      Real out[3] = {};
      ASSERT_TRUE(pops::roe_entropy_fix_apply(A, dU, out, scale)) << "scale=" << scale;
      EXPECT_NEAR(out[0] / scale, Real(1.25), Real(2e-11));
      EXPECT_NEAR(out[1] / scale, Real(1.5), Real(2e-11));
      EXPECT_NEAR(out[2] / scale, Real(10), Real(2e-11));
    }
  }
  {
    // Similarite non orthogonale a tres grande/petite echelle. Toutes les valeurs propres sont
    // hors de la fenetre entropique : Phi_delta(A) = |A|, independamment de l'echelle physique.
    const Real scales[2] = {Real(1e250), Real(1e-250)};
    for (const Real scale : scales) {
      const Real A[2][2] = {{scale, scale}, {0, -Real(2) * scale}};
      const Real dU[2] = {Real(1), Real(1)};
      Real out[2] = {};
      ASSERT_TRUE(pops::roe_entropy_fix_apply(A, dU, out, Real(0.1) * scale))
          << "scale=" << scale;
      EXPECT_NEAR(out[0] / scale, Real(2) / Real(3), Real(2e-11));
      EXPECT_NEAR(out[1] / scale, Real(2), Real(2e-11));
    }
  }
  {
    // Spectre complexe et non-convergence restent des refus explicites, sans toucher la sortie.
    const Real complex_A[2][2] = {{Real(1), Real(-1e-6)}, {Real(1e-6), Real(1)}};
    const Real dU2[2] = {Real(1), Real(0)};
    Real out2[2] = {Real(7), Real(9)};
    EXPECT_TRUE(!pops::roe_entropy_fix_apply(complex_A, dU2, out2, delta));
    EXPECT_TRUE(out2[0] == Real(7) && out2[1] == Real(9));

    const Real roots[3] = {Real(-1), Real(0), Real(2)};
    Real unresolved_A[3][3];
    companion(roots, unresolved_A);
    const Real dU3[3] = {Real(1), Real(1), Real(1)};
    Real out3[3] = {Real(4), Real(5), Real(6)};
    EXPECT_TRUE(!pops::roe_entropy_fix_apply(
        unresolved_A, dU3, out3, delta, 80, Real(1e-13), Real(1e-5), 0));
    EXPECT_TRUE(out3[0] == Real(4) && out3[1] == Real(5) && out3[2] == Real(6));
  }
}
