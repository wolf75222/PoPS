// AmrSystem::potential() : le getter qui expose phi du NIVEAU GROSSIER (base) dans l'ordre natif
// aplati exact-rank, pendant raffine de System::potential(). Sans raffinement (seuil enorme -> un seul
// niveau grossier mono-box couvrant tout le domaine), AmrSystem resout EXACTEMENT le meme Poisson discret que System
// avec son solveur uniforme autorise `cartesian_cg` (meme operateur lap(phi)=f, meme rhs
// f = elliptic_rhs(U), meme BC, meme box). On verifie donc :
//   (1) forme (produit des axes), valeurs FINIES, champ NON TRIVIAL (variation spatiale reelle) ;
//   (2) Poisson periodique a source NEUTRE (alpha (n - n0), integrale nulle) -> phi de moyenne ~0 ;
//   (3) PARITE avec System::potential() (cartesian_cg) sur le MEME modele / densite : meme phi a la
//       tolerance iterative pres, sans contourner la separation uniforme/AMR des solveurs ;
//   (4) apres quelques pas de transport periodique, l'etat puis potential() changent reellement :
//       l'oracle refuse un champ elliptique stale tout en gardant le test sans raffinement.
// Le modele est une advection scalaire exacte + fond neutralisant : ce test isole le contrat Poisson
// et son rafraichissement sans pretendre authentifier le transport ExB du scenario diocotron.
#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/physics/bricks/bricks.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/amr/field_solver_options.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/system.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

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

template <int Dim>
using PotentialModel = CompositeModel<nd::ScalarAdvection<Dim>, NoSource, BackgroundDensity>;

constexpr std::string_view kPotentialConsumerQid = "tests.amr.potential/phi-test/providers@1";

static_assert(PotentialModel<1>::n_providers == 0);
static_assert(PotentialModel<2>::n_providers == 0);
static_assert(PotentialModel<3>::n_providers == 0);

template <int Dim>
PotentialModel<Dim> potential_model(double n0) {
  RealVector<Dim> velocity{};
  for (int axis = 0; axis < Dim; ++axis)
    velocity[axis] = Real(0.25) / Real(axis + 1);
  PotentialModel<Dim> model{};
  model.hyp = nd::ScalarAdvection<Dim>::prepare(velocity);
  model.src = NoSource{};
  model.ell = BackgroundDensity{Real(1), Real(n0), 0};
  return model;
}

template <int Dim>
void install_system_runtime_authority(System<Dim>& system, std::string_view identity) {
  auto lane =
      std::make_shared<ExecutionLane>(ExecutionLane::duplicate_world_collectively(identity));
  system.install_prepared_boundary_execution_lane(std::move(lane));
}

AmrFieldSolverOptions periodic_amr_field_solver_options() {
  CompositeFacOptions fac;
  fac.abs_tol = Real(1e-10);
  fac.coarse_abs_tol = Real(1e-12);
  return geometric_mg_amr_field_solver_options(GeometricMgOptions{}, fac);
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
  test::install_amr_runtime_authority(amr, "tests.amr.potential/amr-runtime@1");
  amr.install_block_state_route("phi_test", "tests.amr.potential/amr-state/phi-test@1");
  add_compiled_model<Dim>(amr, "phi_test", potential_model<Dim>(n0), "minmod", "rusanov",
                          "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
                          static_cast<double>(kWenoEpsilon), false,
                          std::string(kPotentialConsumerQid));
  amr.set_poisson("charge_density", "geometric_mg", "periodic",
                  periodic_amr_field_solver_options());
  amr.set_density("phi_test", rho);
  test::install_forward_euler_program(amr, true);

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

  // --- System (solver cartesian_cg) sur le MEME modele/densite : oracle de parite ---
  SystemConfig<Dim> scfg;
  for (int axis = 0; axis < Dim; ++axis) {
    scfg.shape[axis] = n;
    scfg.periodicity[axis] = true;
  }
  System<Dim> sys(scfg);
  install_system_runtime_authority(sys, "tests.amr.potential/system-runtime@1");
  sys.install_block_state_route("phi_test", "tests.amr.potential/system-state/phi-test@1");
  add_compiled_model<Dim>(sys, "phi_test", potential_model<Dim>(n0), "minmod", "rusanov",
                          "conservative", "explicit");
  sys.set_poisson("charge_density", "cartesian_cg", "auto");
  sys.set_density("phi_test", rho);
  (void)pops::consume_solve_outcome(sys.solve_fields());
  const std::vector<double> ps = sys.potential();
  EXPECT_EQ(ps.size(), pa.size()) << "System.potential() meme taille qu'AmrSystem.potential()";

  // (3) parite a une constante additive pres (phi periodique defini modulo une constante) : on
  // compare apres recentrage sur la moyenne. Les deux solveurs partagent l'operateur et le RHS ;
  // seule leur convergence iterative differe. La borne reste large mais discriminante.
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
      << "AmrSystem.potential() == System.potential() a la tolerance iterative pres"
      << " dmax=" << dmax << " ref=" << ref;

  // (4) le transport non nul doit faire evoluer l'etat puis republier un potentiel distinct.
  amr.advance(1e-3, 8);
  const std::vector<double> rho2 = amr.density("phi_test");
  ASSERT_EQ(rho2.size(), rho.size());
  double density_change = 0.0;
  for (std::size_t cell = 0; cell < rho.size(); ++cell)
    density_change = std::fmax(density_change, std::fabs(rho2[cell] - rho[cell]));
  EXPECT_GT(density_change, 1e-8)
      << "advance doit modifier l'etat conservatif avant le rafraichissement elliptique";

  const std::vector<double> pa2 = amr.potential();
  EXPECT_EQ(pa2.size(), cell_count(cfg.shape))
      << "potential() apres advance rend le nombre exact-rank de valeurs";
  bool finite2 = true;
  double p2min = pa2[0], p2max = pa2[0], p2sum = 0.0;
  for (double v : pa2) {
    if (!std::isfinite(v))
      finite2 = false;
    p2min = std::fmin(p2min, v);
    p2max = std::fmax(p2max, v);
    p2sum += v;
  }
  EXPECT_TRUE(finite2) << "potential() apres advance : valeurs finies";
  EXPECT_TRUE((p2max - p2min) > 1e-6) << "potential() apres advance : champ non trivial";
  const double p2mean = p2sum / static_cast<double>(pa2.size());
  double potential_change = 0.0;
  for (std::size_t cell = 0; cell < pa.size(); ++cell)
    potential_change =
        std::fmax(potential_change, std::fabs((pa2[cell] - p2mean) - (pa[cell] - pmean)));
  EXPECT_GT(potential_change, 1e-9)
      << "le solve elliptique doit republier le potentiel du nouvel etat accepte";
}
