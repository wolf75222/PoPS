// AmrSystem::potential() : le getter qui expose phi du NIVEAU GROSSIER (base) en image exacte-rank,
// pendant raffine de System::potential(). Avec la configuration explicite a un niveau, AmrSystem et
// System resolvent le meme Poisson discret mono-box par leurs
// providers publics respectifs (FAC geometrique et CG cartesien). On verifie donc :
//   (1) forme n^Dim, valeurs FINIES, champ NON TRIVIAL (variation spatiale reelle) ;
//   (2) PARITE Dirichlet avec System::potential() sur le MEME modele / densite : meme phi a la tolerance des
//       deux solveurs pres ;
//   (3) apres une mise a jour de densite acceptee, potential() se rafraichit et reste fini/non trivial.
// Le modele est un scalaire exact-rank a vitesse nulle avec second membre neutralise rho-rho0.
#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/system.hpp>

#include <cmath>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace pops {

template <int Rank, class Model>
PreparedSystemBlock<Rank> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Rank, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

constexpr int Dim = kNativeDimension;

std::size_t native_cell_count(int n) {
  std::size_t count = 1;
  for (int axis = 0; axis < Dim; ++axis)
    count *= static_cast<std::size_t>(n);
  return count;
}

// Bulle de densite lisse autour du centre. Le fond n0 centre le second membre elliptique du modele;
// le probleme Dirichlet reste bien pose sans projection de nullspace.
static std::vector<double> blob(int n, double& mean_out) {
  std::vector<double> rho(native_cell_count(n));
  double s = 0;
  for (std::size_t cell = 0; cell < rho.size(); ++cell) {
    std::size_t quotient = cell;
    double radius_squared = 0.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const int coordinate = static_cast<int>(quotient % static_cast<std::size_t>(n));
      quotient /= static_cast<std::size_t>(n);
      const double x = (static_cast<double>(coordinate) + 0.5) / n - 0.5;
      radius_squared += x * x;
    }
    const double value = std::exp(-radius_squared / 0.01);
    rho[cell] = value;
    s += value;
  }
  mean_out = s / static_cast<double>(rho.size());
  return rho;
}

template <int Rank>
struct PotentialModel {
  using Law = nd::ScalarAdvection<Rank>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Rank;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 0;

  Law law{};
  Real background = Real(0);

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr-potential.scalar", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{Rank}).scalar(background);
  }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"rho"}, 1, {VariableRole::Density}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"rho"}, 1, {VariableRole::Density}};
  }
  POPS_HD nd::StateConversion<Primitive> recover(const State& state) const {
    return law.recover(state);
  }
  POPS_HD nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return law.make_conservative(primitive);
  }
  POPS_HD nd::StateConversionStatus admissibility(const State& state) const {
    return law.admissibility(state);
  }
  template <int Axis>
  POPS_HD State flux(const State& state) const {
    return law.template flux<Axis>(state);
  }
  template <int Axis>
  POPS_HD Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, Real& lower, Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State&, const ProviderValues<0>&) const { return {}; }
  POPS_HD Real elliptic_rhs(const State& state) const { return state[0] - background; }
};

template <int Rank>
PotentialModel<Rank> potential_model(double background) {
  RealVector<Rank> velocity{};
  return {nd::ScalarAdvection<Rank>::prepare(velocity), static_cast<Real>(background)};
}

template <class Facade>
void install_homogeneous_dirichlet_boundary(Facade& system) {
  std::vector<std::string> face_types(static_cast<std::size_t>(2 * Dim), "dirichlet");
  std::vector<std::string> face_identities;
  face_identities.reserve(static_cast<std::size_t>(2 * Dim));
  for (int face = 0; face < 2 * Dim; ++face)
    face_identities.push_back("tests.amr-potential/dirichlet-face-" + std::to_string(face));
  system.install_hyperbolic_boundary(
      "phi_test", "tests.amr-potential/model-qualified-dirichlet@1", 1, face_types,
      std::vector<double>(static_cast<std::size_t>(2 * Dim), 0.0), face_identities, {"density"},
      "tests.amr-potential.state/phi_test");
}

std::vector<double> refreshed_density(const std::vector<double>& rho, double background) {
  std::vector<double> result = rho;
  for (double& value : result)
    value = background + 1.25 * (value - background);
  return result;
}

}  // namespace

TEST(test_amr_potential, Runs) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  const int n = 64;
  double n0 = 0;
  const std::vector<double> rho = blob(n, n0);

  // --- AmrSystem SANS raffinement : un seul niveau grossier mono-box couvrant tout le domaine ---
  AmrSystemConfig<Dim> cfg;
  for (int axis = 0; axis < Dim; ++axis) {
    cfg.shape[axis] = n;
    cfg.periodicity[static_cast<std::size_t>(axis)] = false;
  }
  cfg.regrid_every = 0;  // la configuration a un niveau ne peut pas regrider
  cfg.level_count = 1;
  cfg.transition_ratios.clear();
  cfg.transition_buffers.clear();
  cfg.transition_lookaheads.clear();

  AmrSystem<Dim> amr(cfg);
  amr.install_block_state_route("phi_test", "tests.amr-potential.state/phi_test");
  install_homogeneous_dirichlet_boundary(amr);
  add_compiled_model<Dim>(amr, "phi_test", potential_model<Dim>(n0), "minmod", "rusanov",
                          "conservative", "explicit", 1.0, 1, 1, {}, {}, 0.0,
                          static_cast<double>(kWenoEpsilon), false,
                          "tests.amr-potential/physical_flux");
  amr.set_poisson("charge_density", "geometric_mg", "dirichlet");
  amr.set_density("phi_test", rho);
  test::install_forward_euler_program(amr);

  const std::vector<double> pa = amr.potential();

  // (1) forme + valeurs finies + non trivial
  EXPECT_EQ(pa.size(), native_cell_count(n)) << "potential() rend une image exacte-rank";
  bool all_finite = true;
  double pmin = pa.empty() ? 0 : pa[0], pmax = pa.empty() ? 0 : pa[0];
  for (double v : pa) {
    if (!std::isfinite(v))
      all_finite = false;
    pmin = std::fmin(pmin, v);
    pmax = std::fmax(pmax, v);
  }
  EXPECT_TRUE(all_finite) << "potential() : toutes les valeurs sont finies";
  EXPECT_TRUE((pmax - pmin) > 1e-6) << "potential() : champ non trivial (variation spatiale)";

  // --- System (solver cartesian_cg) sur le MEME modele/densite : oracle de parite ---
  SystemConfig<Dim> scfg;
  for (int axis = 0; axis < Dim; ++axis) {
    scfg.shape[axis] = n;
    scfg.periodicity[static_cast<std::size_t>(axis)] = false;
  }
  System<Dim> sys(scfg);
  sys.install_block_state_route("phi_test", "tests.amr-potential.state/phi_test");
  install_homogeneous_dirichlet_boundary(sys);
  add_compiled_model<Dim>(sys, "phi_test", potential_model<Dim>(n0), "minmod", "rusanov",
                          "conservative", "explicit", 1.0);
  sys.set_poisson("charge_density", "cartesian_cg", "dirichlet");
  sys.set_density("phi_test", rho);
  (void)pops::consume_solve_outcome(sys.solve_fields());
  const std::vector<double> ps = sys.potential();
  EXPECT_EQ(ps.size(), pa.size()) << "System.potential() meme taille qu'AmrSystem.potential()";

  // (2) Parite Dirichlet directe. FAC et CG ont des iterations independantes; 1e-3 est une borne
  // relative volontairement largement sous la precision scientifique, mais assez large pour leurs
  // residus arretes differemment.
  constexpr double cross_provider_relative_tolerance = 1e-3;
  double dmax = 0, ref = 0;
  for (std::size_t k = 0; k < pa.size() && k < ps.size(); ++k) {
    dmax = std::fmax(dmax, std::fabs(pa[k] - ps[k]));
    ref = std::fmax(ref, std::fabs(ps[k]));
  }
  EXPECT_TRUE(ref > 1e-6) << "System phi non trivial (oracle valide)";
  EXPECT_TRUE(dmax < cross_provider_relative_tolerance * (ref + 1e-12))
      << "AmrSystem.potential() == System.potential() a la tolerance solveur pres"
      << " dmax=" << dmax << " ref=" << ref;

  // (3) Une mise a jour acceptee du second membre suivie du solve public rafraichit phi. Ce chemin
  // est plus etroit qu'un pas d'advection a vitesse nulle et rend le rafraichissement observable.
  amr.set_density("phi_test", refreshed_density(rho, n0));
  const std::vector<double> pa2 = amr.potential();
  EXPECT_EQ(pa2.size(), native_cell_count(n)) << "potential() apres mise a jour : image exacte-rank";
  bool finite2 = true;
  double p2min = pa2[0], p2max = pa2[0];
  double refresh_delta = 0;
  for (std::size_t k = 0; k < pa2.size(); ++k) {
    const double v = pa2[k];
    if (!std::isfinite(v))
      finite2 = false;
    p2min = std::fmin(p2min, v);
    p2max = std::fmax(p2max, v);
    refresh_delta = std::fmax(refresh_delta, std::fabs(v - pa[k]));
  }
  EXPECT_TRUE(finite2) << "potential() apres mise a jour : valeurs finies";
  EXPECT_TRUE((p2max - p2min) > 1e-6) << "potential() apres mise a jour : champ non trivial";
  EXPECT_TRUE(refresh_delta > 1e-6) << "potential() se rafraichit apres la mise a jour de densite";
}
