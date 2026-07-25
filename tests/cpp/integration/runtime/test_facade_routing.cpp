// Chantier T5-PR3 : CABLAGE du transport disque (staircase + cut-cell EB) dans System::step.
//
// CONTEXTE (cf. docs/HOFFART_FIDELITY.md ligne 39, verrou "bords d'anneau cartesiens" ; le footgun T2 :
// set_disc_domain materialisait un masque MAIS System::step ne le consultait jamais -> le disque etait
// INERTE). Ce PR aiguille l'avance de transport de step() vers l'operateur disque selon un MODE explicite
// (none | staircase | cutcell), porte par set_disc_domain(mode=) / set_geometry_mode et lu par le stepper.
//
// On valide (vraies assertions, pas de no-op) :
//   (a) NO-DISC PAR DEFAUT : un pas avec set_disc_domain(mode='none') est BYTE-IDENTIQUE a un pas SANS
//       set_disc_domain (le masque est materialise mais le transport l'ignore) -> diff EXACTEMENT 0.
//   (b) ROUTING-LIVE (staircase) : mode='staircase' produit un etat DIFFERENT du carre sur le MEME init
//       (max|diff| > 0 : le routage N'EST PAS inerte) ET la masse sur les cellules ACTIVES du disque est
//       conservee a la machine (aucun flux ne franchit la frontiere du masque) -> propre au schema masque.
//   (c) CUTCELL : mode='cutcell' tourne, etat FINI partout (aucun NaN/Inf), DIFFERENT du carre ; et sur
//       un disque ENGLOBANT (rayon > diagonale, aucune cellule coupee) un pas est BIT-IDENTIQUE au carre.
//
// Modele : transport scalaire ExB (add_block transport='exb', source='none', elliptic='charge') -- le
// transport DIOCOTRON de production. La vitesse derive de grad phi (Poisson sur la densite) : champ a
// divergence nulle -> la masse est conservee par les schemas masque / EB. Compile python/system.cpp.

#include <gtest/gtest.h>

#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/system.hpp>

#include <cmath>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

#if defined(POPS_HAS_KOKKOS)
Kokkos::ScopeGuard& kokkos_scope() {
  static Kokkos::ScopeGuard guard;
  return guard;
}
#endif

// Densite initiale : anneau lisse (recouvre le disque interieur), perturbe en azimut pour casser la
// symetrie -> grad phi non trivial -> vitesse ExB non nulle. n*n row-major (j lent, i rapide).
std::vector<double> ring_density(int n, double L) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n, 1e-3);
  const double cx = 0.5 * L, cy = 0.5 * L;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) * L / n, y = (j + 0.5) * L / n;
      const double r = std::hypot(x - cx, y - cy);
      // anneau gaussien centre sur r0 = 0.18 L, module en sin(3 theta) (perturbation azimutale l=3).
      const double r0 = 0.18 * L, w = 0.05 * L;
      const double th = std::atan2(y - cy, x - cx);
      const double g = std::exp(-((r - r0) * (r - r0)) / (2 * w * w));
      rho[static_cast<std::size_t>(j) * n + i] = 1e-3 + g * (1.0 + 0.3 * std::sin(3 * th));
    }
  return rho;
}

std::vector<double> periodic_seam_density(int n) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  const double two_pi = 2.0 * std::acos(-1.0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n;
      const double y = (j + 0.5) / n;
      rho[static_cast<std::size_t>(j) * n + i] =
          1.0 + 0.15 * std::cos(two_pi * x) * std::sin(two_pi * y);
    }
  return rho;
}

ModelSpec periodic_exb_model() {
  ModelSpec spec;
  spec.transport = "exb";
  spec.source = "none";
  spec.elliptic = "background";
  spec.B0 = 1.0;
  spec.alpha = 1.0;
  spec.n0 = 1.0;
  return spec;
}

// Construit un System scalaire ExB diocotron pret a stepper. Le disque/mode est pose par l'appelant.
void build_exb(System& s, double R_wall) {
  ModelSpec spec;
  spec.transport = "exb";
  spec.source = "none";
  spec.elliptic = "charge";
  spec.q = 1.0;
  spec.B0 = 1.0;
  // First-order reconstruction is the native embedded-boundary provider supported by this facade.
  // Higher-order stencils require geometry-aware neighbor reconstruction and are rejected rather
  // than reading inactive cells. The same provider is used in every mode so this test isolates only
  // residual routing.
  s.add_block("n", spec, "none", "rusanov", "conservative", "explicit", 1, true);
  // Poisson sur la densite de charge, mur conducteur circulaire concentrique (comme le diocotron) :
  // donne un phi non trivial -> vitesse ExB. Le mur elliptique et le disque de transport partagent le
  // meme centre (L/2, L/2) et la meme convention de level set.
  s.set_poisson("charge_density", "geometric_mg", "dirichlet", "circle", R_wall, 1.0);
}

// max|diff| composante a composante entre deux champs de meme taille.
double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k)
    d = std::fmax(d, std::fabs(a[k] - b[k]));
  return d;
}

bool all_finite(const std::vector<double>& a) {
  for (double v : a)
    if (!std::isfinite(v))
      return false;
  return true;
}

}  // namespace

TEST(FacadeRouting, DiscModeRoutingBehavesAcrossNoneStaircaseCutcellAndSplittings) {
#if defined(POPS_HAS_KOKKOS)
  (void)kokkos_scope();
#endif
  const int n = 48;
  const double L = 1.0;
  const double R_wall = 0.45 * L;  // mur conducteur de Poisson (rayon < L/2)
  const double R_disc =
      0.30 * L;  // disque de transport (plus petit : de vraies cellules inactives)
  const double cx = 0.5 * L, cy = 0.5 * L;
  const double dt = 2e-4;  // pas court, transport ExB sous-CFL
  const int n_steps = 12;
  const std::vector<double> rho0 = ring_density(n, L);

  // ----------------------------------------------------------------------
  // (a) NO-DISC PAR DEFAUT : mode='none' (disque materialise) == jamais set_disc_domain (byte a byte).
  // ----------------------------------------------------------------------
  std::vector<double>
      ref_state;  // etat de reference (chemin plein cartesien), reutilise par (b)/(c)/(d)
  {
    System base(SystemConfig{n, L, Periodicity{false, false}});
    build_exb(base, R_wall);
    base.set_density("n", rho0);
    for (int k = 0; k < n_steps; ++k)
      base.step(dt);
    ref_state = base.get_state("n");

    System none(SystemConfig{n, L, Periodicity{false, false}});
    build_exb(none, R_wall);
    none.set_density("n", rho0);
    none.set_disc_domain(cx, cy, R_disc, "none");  // disque pose, mode none : doit rester inerte
    for (int k = 0; k < n_steps; ++k)
      none.step(dt);
    const std::vector<double> none_state = none.get_state("n");

    const double d = max_abs_diff(ref_state, none_state);
    // Egalite BYTE A BYTE : mode none emprunte exactement assemble_rhs, le disque materialise n'a AUCUN
    // effet sur le transport. Pas une tolerance -- l'invariant "inerte par defaut".
    EXPECT_TRUE(d == 0.0)
        << "(a) mode='none' BIT-IDENTIQUE au chemin sans disque (routage inerte sauf opt-in) : "
           "max|diff| = "
        << d << " (attendu 0)";
    EXPECT_TRUE(all_finite(ref_state) && ref_state.size() == static_cast<std::size_t>(n) * n)
        << "(a) etat de reference fini et de taille n*n (le pas plein a bien tourne)";
  }

  // ----------------------------------------------------------------------
  // (b) ROUTING-LIVE (staircase) : etat DIFFERENT du carre + masse active conservee a la machine.
  // ----------------------------------------------------------------------
  {
    System sc(SystemConfig{n, L, Periodicity{false, false}});
    build_exb(sc, R_wall);
    sc.set_density("n", rho0);
    sc.set_disc_domain(cx, cy, R_disc, "staircase");

    // Masse initiale sur les cellules ACTIVES (masque 0/1 du System) AVANT les pas.
    const std::vector<double> mask = sc.disc_mask();  // (ny, nx) row-major, 1.0 actif
    const std::vector<double> dens0 = sc.density("n");
    const double dx2 = (L / n) * (L / n);
    int n_active = 0, n_inactive = 0;
    double mass0 = 0.0;
    for (std::size_t k = 0; k < mask.size(); ++k) {
      if (mask[k] >= 0.5) {
        ++n_active;
        mass0 += dens0[k] * dx2;
      } else
        ++n_inactive;
    }
    ASSERT_TRUE(n_active > 0 && n_inactive > 0)
        << "(b) le disque partitionne la grille en cellules actives ET inactives (test non vide)";

    for (int k = 0; k < n_steps; ++k)
      sc.step(dt);
    const std::vector<double> sc_state = sc.get_state("n");

    // Masse active APRES les pas (meme masque : le disque est statique).
    const std::vector<double> dens1 = sc.density("n");
    double mass1 = 0.0;
    for (std::size_t k = 0; k < mask.size(); ++k)
      if (mask[k] >= 0.5)
        mass1 += dens1[k] * dx2;

    const double d_vs_square = max_abs_diff(ref_state, sc_state);
    const double rel_drift = std::fabs(mass1 - mass0) / std::fabs(mass0);

    // Le routage N'EST PAS inerte : l'operateur masque ferme les faces a la frontiere du disque, donc
    // l'etat diverge du chemin plein cartesien. C'est la preuve directe contre le footgun T2.
    EXPECT_TRUE(d_vs_square > 1e-10)
        << "(b) staircase produit un etat DIFFERENT du carre (le transport disque est REELLEMENT "
           "cable) : max|diff| = "
        << d_vs_square << " (attendu > 0)";
    EXPECT_TRUE(all_finite(sc_state)) << "(b) etat staircase fini partout (aucun NaN/Inf)";
    // La masse sur les cellules actives est conservee a la machine (flux normal nul aux faces
    // active/inactive). Borne juste au-dessus du bruit flottant des sommes telescopiques de flux.
    EXPECT_TRUE(rel_drift < 1e-12)
        << "(b) masse sur les cellules actives conservee a la machine (schema masque conservatif) "
           ": drift = "
        << rel_drift;
  }

  // ----------------------------------------------------------------------
  // (c) CUTCELL : tourne, FINI partout, DIFFERENT du carre ; disque ENGLOBANT == carre (bit a bit).
  // ----------------------------------------------------------------------
  {
    // (c1) disque coupant : etat fini + different du carre.
    System cc(SystemConfig{n, L, Periodicity{false, false}});
    build_exb(cc, R_wall);
    cc.set_density("n", rho0);
    cc.set_disc_domain(cx, cy, R_disc, "cutcell");
    for (int k = 0; k < n_steps; ++k)
      cc.step(dt);
    const std::vector<double> cc_state = cc.get_state("n");
    const double d_vs_square = max_abs_diff(ref_state, cc_state);
    EXPECT_TRUE(all_finite(cc_state))
        << "(c1) etat cutcell fini partout (clamp small-cell -> pas de NaN/Inf)";
    EXPECT_TRUE(d_vs_square > 1e-10)
        << "(c1) cutcell produit un etat DIFFERENT du carre (transport EB cable) : max|diff| = "
        << d_vs_square << " (attendu > 0)";

    // (c2) disque ENGLOBANT (rayon > demi-diagonale) : TOUTE cellule est active, AUCUNE face coupee ->
    // assemble_rhs_eb == assemble_rhs (kappa=1, alpha=1 partout, cf. test_eb_transport bit-identite).
    // Un pas cutcell doit alors etre BIT-IDENTIQUE au pas carre sur le meme init.
    const double R_big = 10.0 * L;         // englobe largement la boite
    System sq(SystemConfig{n, L, Periodicity{false, false}});  // reference 1 pas plein
    build_exb(sq, R_wall);
    sq.set_density("n", rho0);
    sq.step(dt);
    const std::vector<double> sq1 = sq.get_state("n");

    System eb(SystemConfig{n, L, Periodicity{false, false}});
    build_exb(eb, R_wall);
    eb.set_density("n", rho0);
    eb.set_disc_domain(cx, cy, R_big, "cutcell");
    eb.step(dt);
    const std::vector<double> eb1 = eb.get_state("n");

    const double d_enclosing = max_abs_diff(sq1, eb1);
    EXPECT_TRUE(d_enclosing == 0.0) << "(c2) cutcell sans coupe BIT-IDENTIQUE au carre (kappa=1, "
                                       "alpha=1 partout) : max|diff| = "
                                    << d_enclosing << " (attendu 0)";
  }
}

TEST(FacadeRouting, GenericAnalyticLevelSetMatchesDiscSugarAfterBlockConstruction) {
#if defined(POPS_HAS_KOKKOS)
  (void)kokkos_scope();
#endif
  const int n = 24;
  const double L = 1.0;
  const double cx = 0.5;
  const double cy = 0.5;
  const double radius = 0.31;
  const double wall_radius = 0.45;
  const std::vector<double> rho0 = ring_density(n, L);

  // Both transport closures are deliberately built before their geometry is installed. The stable
  // native program owner must therefore make authoring order irrelevant.
  System disc(SystemConfig{n, L, Periodicity{false, false}});
  build_exb(disc, wall_radius);
  disc.set_density("n", rho0);
  disc.set_disc_domain(cx, cy, radius, "cutcell");

  System analytic(SystemConfig{n, L, Periodicity{false, false}});
  build_exb(analytic, wall_radius);
  analytic.set_density("n", rho0);
  analytic.set_analytic_level_set(
      {"x", "constant", "sub", "y", "constant", "sub", "hypot", "constant", "sub"},
      {0.0, cx, 0.0, 0.0, cy, 0.0, 0.0, radius, 0.0}, "cutcell");

  EXPECT_EQ(disc.disc_mask(), analytic.disc_mask());
  disc.step(2e-4);
  analytic.step(2e-4);
  EXPECT_EQ(disc.get_state("n"), analytic.get_state("n"));
}

TEST(FacadeRouting, AnalyticLevelSetReplacementIsTransactionalOnNonFiniteValues) {
#if defined(POPS_HAS_KOKKOS)
  (void)kokkos_scope();
#endif
  System system(SystemConfig{20, 1.0, Periodicity{false, false}});
  system.set_analytic_level_set({"x", "constant", "sub"}, {0.0, 0.5, 0.0},
                                "staircase", 0.2, 1e-5, 0.1);
  const std::vector<double> original = system.disc_mask();

  // (x - x) / 0 is structurally valid but non-finite at every sampled cell. Rejection must happen
  // before publishing either the new program, the new mask, the thresholds, or the routing mode.
  const std::vector<std::string> invalid_ops{"x", "x", "sub", "constant", "div"};
  const std::vector<double> invalid_literals{0.0, 0.0, 0.0, 0.0, 0.0};
  EXPECT_THROW(system.set_analytic_level_set(invalid_ops, invalid_literals, "cutcell", 0.3,
                                             2e-5, 0.2),
               std::domain_error);
  EXPECT_EQ(original, system.disc_mask());
}

TEST(FacadeRouting, PeriodicAnalyticLevelSetUsesTopologyAtTheSeam) {
#if defined(POPS_HAS_KOKKOS)
  (void)kokkos_scope();
#endif
  const int n = 24;
  const std::vector<double> rho0 = periodic_seam_density(n);

  // The valid-cell expression x - 1/4 describes the same non-circular half-plane in both systems.
  // The reference spells out the low-side periodic extension only to make this regression observable:
  // a correct topology fill replaces that extension with the opposite valid cells and both prepared
  // metric fields become bit-identical. Direct evaluation at the fictitious x<0 ghost does not.
  System topology(SystemConfig{n, 1.0, Periodicity{true, true}});
  topology.add_block("n", periodic_exb_model(), "none");
  topology.set_density("n", rho0);
  topology.set_analytic_level_set({"x", "constant", "sub"},
                                  {0.0, 0.25, 0.0}, "cutcell");

  System explicit_wrap(SystemConfig{n, 1.0, Periodicity{true, true}});
  explicit_wrap.add_block("n", periodic_exb_model(), "none");
  explicit_wrap.set_density("n", rho0);
  explicit_wrap.set_analytic_level_set(
      {"x", "constant", "lt", "x", "constant", "add", "constant", "sub",
       "x", "constant", "sub", "where"},
      {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0},
      "cutcell");

  ASSERT_EQ(topology.disc_mask(), explicit_wrap.disc_mask());
  topology.step(2e-4);
  explicit_wrap.step(2e-4);
  EXPECT_EQ(topology.get_state("n"), explicit_wrap.get_state("n"));
  EXPECT_GT(max_abs_diff(topology.get_state("n"), rho0), 0.0);
}
