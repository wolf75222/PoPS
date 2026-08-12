// AmrSystem::potential() : le getter qui expose phi du NIVEAU GROSSIER (base) dans l'ordre natif
// aplati exact-rank, pendant raffine de System::potential(). Sans raffinement (seuil enorme -> un seul
// niveau grossier mono-box couvrant tout le domaine), AmrSystem resout EXACTEMENT le meme Poisson discret que System
// avec solver='geometric_mg' (meme operateur lap(phi)=f, meme rhs f = elliptic_rhs(U), meme BC, meme
// box). On verifie donc :
//   (1) forme (produit des axes), valeurs FINIES, champ NON TRIVIAL (variation spatiale reelle) ;
//   (2) Poisson periodique a source NEUTRE (alpha (n - n0), integrale nulle) -> phi de moyenne ~0 ;
//   (3) PARITE avec System::potential() (geometric_mg) sur le MEME modele / densite : meme phi a la
//       tolerance MG pres (les deux passent par GeometricMG sur le meme grossier mono-box) ;
//   (4) apres quelques pas (regrid inclus), potential() reste fini et non trivial (rafraichi).
// Le modele est un transport ExB pur + fond neutralisant (briques exb / none / background), proche du
// scenario diocotron qui echantillonne phi sur un cercle median (FFT azimutale).
#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/system.hpp>

#include <cmath>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

// Bulle de densite lisse autour du centre, periodique. Moyenne retiree pour neutraliser la source
// (fond background n0 = moyenne) : Poisson periodique exige une integrale de second membre nulle.
template <int Dim>
std::size_t cell_count(const Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
static std::vector<double> blob(const Extent<Dim>& shape, double& mean_out) {
  std::vector<double> rho(cell_count(shape));
  double s = 0;
  for (std::size_t ordinal = 0; ordinal < rho.size(); ++ordinal) {
    std::size_t remainder = ordinal;
    double radius_squared = 0.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto extent = static_cast<std::size_t>(shape[axis]);
      const double coordinate =
          (static_cast<double>(remainder % extent) + 0.5) / static_cast<double>(extent) - 0.5;
      remainder /= extent;
      radius_squared += coordinate * coordinate;
    }
    rho[ordinal] = std::exp(-radius_squared / 0.01);
    s += rho[ordinal];
  }
  mean_out = s / static_cast<double>(rho.size());
  return rho;
}

static ModelSpec exb_background(double n0) {
  ModelSpec spec;
  spec.transport = "exb";        // derive E x B (a divergence nulle)
  spec.source = "none";          // pas de force source
  spec.elliptic = "background";  // f = alpha (n - n0) : fond neutralisant
  spec.alpha = 1.0;
  spec.n0 = n0;  // fond = moyenne -> source d'integrale nulle (Poisson periodique)
  return spec;
}

TEST(test_amr_potential, Runs) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  const int n = 64;
  constexpr int Dim = kNativeDimension;
  double n0 = 0;
  AmrSystemConfig<Dim> cfg;
  for (int axis = 0; axis < Dim; ++axis) {
    cfg.shape[axis] = n;
    cfg.periodicity[axis] = true;
  }
  const std::vector<double> rho = blob(cfg.shape, n0);

  // --- AmrSystem SANS raffinement : un seul niveau grossier mono-box couvrant tout le domaine ---
  cfg.regrid_every = 0;  // pas de re-raffinement apres l'init (seuil enorme de toute facon)

  AmrSystem<Dim> amr(cfg);
  amr.add_block("phi_test", exb_background(n0), "minmod", "rusanov", "conservative", "explicit", 1);
  amr.set_poisson("charge_density", "geometric_mg", "auto");
  amr.set_density("phi_test", rho);
  test::install_forward_euler_program(amr);

  const std::vector<double> pa = amr.potential();

  // (1) forme + valeurs finies + non trivial
  EXPECT_EQ(pa.size(), cell_count(cfg.shape)) << "potential() rend le nombre exact-rank de valeurs";
  bool all_finite = true;
  double pmin = pa.empty() ? 0 : pa[0], pmax = pa.empty() ? 0 : pa[0], psum = 0;
  for (double v : pa) {
    if (!std::isfinite(v))
      all_finite = false;
    pmin = std::fmin(pmin, v);
    pmax = std::fmax(pmax, v);
    psum += v;
  }
  EXPECT_TRUE(all_finite) << "potential() : toutes les valeurs sont finies";
  EXPECT_TRUE((pmax - pmin) > 1e-6) << "potential() : champ non trivial (variation spatiale)";

  // (2) Poisson periodique a source neutre -> phi defini a une constante pres, moyenne ~ 0
  const double pmean = psum / static_cast<double>(pa.size());
  EXPECT_TRUE(std::fabs(pmean) < 1e-6 * (pmax - pmin) + 1e-9)
      << "potential() : moyenne ~0 (source neutre)";

  // --- System (solver geometric_mg) sur le MEME modele/densite : oracle de parite ---
  SystemConfig<Dim> scfg;
  for (int axis = 0; axis < Dim; ++axis) {
    scfg.shape[axis] = n;
    scfg.periodicity[axis] = true;
  }
  System<Dim> sys(scfg);
  sys.add_block("phi_test", exb_background(n0), "minmod", "rusanov", "conservative", "explicit", 1);
  sys.set_poisson("charge_density", "geometric_mg", "auto");
  sys.set_density("phi_test", rho);
  (void)pops::consume_solve_outcome(sys.solve_fields());
  const std::vector<double> ps = sys.potential();
  EXPECT_EQ(ps.size(), pa.size()) << "System.potential() meme taille qu'AmrSystem.potential()";

  // (3) parite a une constante additive pres (phi periodique defini modulo une constante) : on
  // compare apres recentrage sur la moyenne. Tolerance MG : meme operateur, meme rhs, meme box ->
  // l'ecart vient des iterations MG (rel_tol), pas du modele. Borne large mais discriminante.
  double smean = 0;
  for (double v : ps)
    smean += v;
  smean /= static_cast<double>(ps.size());
  double dmax = 0, ref = 0;
  for (std::size_t k = 0; k < pa.size() && k < ps.size(); ++k) {
    dmax = std::fmax(dmax, std::fabs((pa[k] - pmean) - (ps[k] - smean)));
    ref = std::fmax(ref, std::fabs(ps[k] - smean));
  }
  EXPECT_TRUE(ref > 1e-6) << "System phi non trivial (oracle valide)";
  EXPECT_TRUE(dmax < 1e-3 * (ref + 1e-12))
      << "AmrSystem.potential() == System.potential() (geometric_mg) a la tolerance MG pres"
      << " dmax=" << dmax << " ref=" << ref;

  // (4) apres quelques pas (transport ExB + regrid), potential() reste fini et non trivial
  amr.advance(1e-3, 8);
  const std::vector<double> pa2 = amr.potential();
  EXPECT_EQ(pa2.size(), cell_count(cfg.shape))
      << "potential() apres advance rend le nombre exact-rank de valeurs";
  bool finite2 = true;
  double p2min = pa2[0], p2max = pa2[0];
  for (double v : pa2) {
    if (!std::isfinite(v))
      finite2 = false;
    p2min = std::fmin(p2min, v);
    p2max = std::fmax(p2max, v);
  }
  EXPECT_TRUE(finite2) << "potential() apres advance : valeurs finies";
  EXPECT_TRUE((p2max - p2min) > 1e-6) << "potential() apres advance : champ non trivial";
}
