// Chantier "Aux extensible", increment 8 : T_e, 2e champ aux supplementaire, peuple par DERIVATION
// (et non fourni par l'utilisateur comme B_z). Un bloc fluide COMPRESSIBLE fournit T = p/rho ; un
// bloc qui declare lire aux('T_e') le lit. Valide la generalisation du canal aux a un 2e
// champ (composante 4) et la population derivee cote System (set_electron_temperature_from + apply_te
// recalcule a chaque solve_fields). Chemin de production : add_compiled_model + eval_rhs.

#include <gtest/gtest.h>

#include <pops/physics/composition/composite.hpp>
#include <pops/physics/fluids/euler.hpp>                 // Euler (bloc fluide source de T_e)
#include <pops/physics/bricks/hyperbolic.hpp>            // ExBVelocity
#include <pops/physics/bricks/source.hpp>                // NoSource
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/system.hpp>

#include <cmath>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

// Source qui lit T_e : S = T_e u (composante 0).
struct TeSource {
  static constexpr int n_aux = AuxComponentLayout<kNativeDimension>::named_begin;
  template <class State>
  POPS_HD State apply(const State& u, const AuxState<kNativeDimension>& a) const {
    State s{};
    s[0] = a.T_e * u[0];
    return s;
  }
};
struct NoEll {
  template <class State>
  POPS_HD Real rhs(const State&) const {
    return Real(0);
  }
};

using ProbeModel = CompositeModel<ExBVelocity, TeSource, NoEll>;  // reads T_e
using GasModel = CompositeModel<Euler, NoSource, NoEll>;          // provides p/rho
static_assert(ProbeModel::n_aux == AuxComponentLayout<kNativeDimension>::named_begin);
static_assert(nd::ConservationLaw<kNativeDimension, GasModel>);

TEST(AuxTe, DerivedFromGasDrivesProbeSource) {
#if defined(POPS_HAS_KOKKOS)
  int argc = 0;
  char** argv = nullptr;
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int n = 16;
  const double gamma = 1.4, rho_gas = 1.0, p_gas = 3.0;
  const double Te = p_gas / rho_gas;  // T = p / rho = 3

  SystemConfig<kNativeDimension> cfg;
  for (int axis = 0; axis < kNativeDimension; ++axis) {
    cfg.shape[axis] = n;
    cfg.periodicity[axis] = true;
  }

  System<kNativeDimension> sys(cfg);
  GasModel gas_model;
  gas_model.hyp.gamma = gamma;
  add_compiled_model(sys, "gas", gas_model, "minmod", "rusanov", "conservative", "explicit", gamma);
  add_compiled_model(sys, "probe", ProbeModel{}, "minmod", "rusanov", "conservative", "explicit");
  sys.set_poisson("charge_density", "geometric_mg");

  // etat du gaz : rho=1, qte de mvt nulle, E = p/(gamma-1) -> p=3, donc T = p/rho = 3.
  std::size_t nn = 1;
  for (int axis = 0; axis < kNativeDimension; ++axis)
    nn *= static_cast<std::size_t>(n);
  std::vector<double> Ug(static_cast<std::size_t>(Euler::n_vars) * nn, 0.0);
  for (std::size_t k = 0; k < nn; ++k) {
    Ug[0 * nn + k] = rho_gas;
    Ug[static_cast<std::size_t>(Euler::energy_component) * nn + k] = p_gas / (gamma - 1.0);
  }
  sys.set_state("gas", Ug);
  sys.set_density("probe", std::vector<double>(nn, 1.0));
  sys.set_electron_temperature_from("gas");  // T_e <- p/rho du gaz, recalcule a chaque solve
  (void)pops::consume_solve_outcome(sys.solve_fields());

  // eval_rhs(probe) = -div F + S ; flux ExB(grad=0)=0 -> R = source = T_e * n = Te.
  const std::vector<double> R = sys.eval_rhs("probe");
  double err = 0;
  for (double r : R)
    err = std::fmax(err, std::fabs(r - Te));
  EXPECT_TRUE(err < 1e-12) << "te_derived_and_read (max|R - T_e|=" << err << " T_e=" << Te << ")";
}
