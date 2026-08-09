// Bug HOTE MPI de System::solve_fields() (#98) : System répartit UNE box exacte en round-robin,
// donc à np>1 un seul rang possède la box ; les autres ont
// local_size()==0. Le post-traitement par cellule de solve_fields() (derivation phi/grad,
// apply_te, apply_epsilon*, apply_reaction) appelait fab(0) SANS tester local_size() -> les rangs
// sans box locale dereferencaient un fab inexistant et segfaultaient cote hote.
//
// Ce test exerce System + Poisson + solve_fields() (mono-box) sous mpirun -np {1,2,4} et exige que
// l'appel TOURNE (pas de crash) sur tous les rangs et donne un resultat sense. Le solve elliptique
// est COLLECTIF (tous les rangs y participent) ; seul le post-traitement par cellule est local au
// rang proprietaire. Les I/O par cellule (set_density / density / get_state) ne touchent QUE le rang
// proprietaire ; eval_rhs() et mass() terminent par des reductions collectives et sont appeles par
// tous les rangs.
//
// AVANT le fix : segfault a np=2/4 sur le(s) rang(s) sans box locale. APRES : np=1/2/4 verts, et le
// resultat (potentiel, masse) est invariant au nombre de rangs (la box unique vit toujours sur rang 0).

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/physics/composition/composite.hpp>
#include <pops/physics/bricks/elliptic.hpp>
#include <pops/physics/bricks/hyperbolic.hpp>
#include <pops/physics/bricks/source.hpp>                // NoSource
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/system.hpp>

#include <pops/parallel/comm.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#ifdef POPS_HAS_MPI
#include <mpi.h>
#endif

using namespace pops;

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

constexpr int kTestDimension = kNativeDimension;
using NativeSystem = System<kTestDimension>;
using NativeSystemConfig = SystemConfig<kTestDimension>;
using NativeField = MultiFab<kTestDimension>;
using NativeGasLaw = nd::IdealGasEuler<kTestDimension>;

// Source qui lit T_e : exerce le canal auxiliaire dérivé (apply_te) dans solve_fields().
struct TeSource {
  static constexpr int n_providers = 1;
  template <class State>
  POPS_HD State apply(const State& u, const ProviderValues<1>& providers) const {
    State s{};
    s[0] = providers[0] * u[0];
    return s;
  }
};
// Bloc de CHARGE : alimente le second membre du Poisson (elliptic_rhs = densite de charge q n).
using ProbeModel =
    CompositeModel<ExBVelocity, TeSource, ChargeDensity>;             // lit T_e + charge le Poisson
using GasModel = CompositeModel<NativeGasLaw, NoSource, NoElliptic>;  // fournit p/rho

std::size_t cell_count(int n) {
  std::size_t result = 1;
  for (int axis = 0; axis < kTestDimension; ++axis)
    result *= static_cast<std::size_t>(n);
  return result;
}

NativeSystemConfig native_config(int n) {
  NativeSystemConfig config;
  for (int axis = 0; axis < kTestDimension; ++axis) {
    config.shape[axis] = n;
    config.lower[axis] = Real(0);
    config.upper[axis] = Real(1);
    config.periodicity[axis] = true;
  }
  config.boxes = {Box<kTestDimension>::from_extents(config.shape)};
  return config;
}

}  // namespace

static int pops_run_test_mpi_system_solve_fields(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int me = my_rank(), np = n_ranks();

  long fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("[rank %d/%d] FAIL %s\n", me, np, w);
      ++fails;
    }
  };

  const int n = 16;
  const double gamma = 1.4, rho_gas = 1.0, p_gas = 3.0;
  const double Te = p_gas / rho_gas;  // T = p / rho = 3

  const NativeSystemConfig cfg = native_config(n);

  NativeSystem sys(cfg);
  GasModel gas_model{};
  gas_model.hyp = NativeGasLaw::prepare(static_cast<Real>(gamma));
  sys.install_block_state_route("gas", "test.mpi-system-solve-fields.gas.state@1");
  add_compiled_model(sys, "gas", std::move(gas_model), "minmod", "rusanov", "conservative",
                     "explicit", gamma);
  using namespace runtime::system;
  AuxiliaryStorageShape<kTestDimension> temperature_shape;
  AuxiliaryComponentKey temperature_key{"test.mpi-system-solve-fields", "derived", "gas", "T_e"};
  AuxiliaryComponentContract temperature_contract{"cell-average", "cell", "unitless",
                                                  "test-constant-temperature", "scalar"};
  sys.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<kTestDimension>{
      "test.mpi-system-solve-fields.temperature",
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      {{temperature_key, temperature_contract, temperature_shape}},
      {}});
  sys.install_auxiliary_consumer_plan(AuxiliaryConsumerProviderPlan<kTestDimension>{
      "probe", {{{temperature_key, temperature_contract, temperature_shape}, 0}}});
  sys.seal_auxiliary_providers();
  sys.install_block_state_route("probe", "test.mpi-system-solve-fields.probe.state@1");
  add_compiled_model(sys, "probe", ProbeModel{}, "minmod", "rusanov", "conservative", "explicit");
  sys.set_poisson("composite",
                  "cartesian_cg");  // f = somme des briques elliptiques (ici la charge)

  // La box unique vit sur rang 0 (round-robin de 1 box), mais set_state / set_density portent un
  // contrat global COLLECTIF : chaque rang soumet donc le même buffer component-major. Cela prouve
  // que l'initialisation ne dépend pas de la propriété locale. La densité de charge a moyenne nulle
  // pour que le Poisson périodique soit soluble : on met un créneau symétrique.
  const std::size_t nn = cell_count(n);
  const bool owns = (me == 0);  // box 0 -> rang 0 sous le mapping round-robin exact
  std::vector<double> Ug(static_cast<std::size_t>(GasModel::n_vars) * nn, 0.0);
  for (std::size_t k = 0; k < nn; ++k) {
    Ug[0 * nn + k] = rho_gas;
    Ug[static_cast<std::size_t>(NativeGasLaw::Schema::energy) * nn + k] = p_gas / (gamma - 1.0);
  }
  sys.set_state("gas", Ug);
  // Charge à moyenne nulle : +1 sur la moitié basse du premier axe, -1 sur l'autre moitié.
  std::vector<double> q(nn, 0.0);
  for (std::size_t linear = 0; linear < nn; ++linear) {
    const int first_axis = static_cast<int>(linear % static_cast<std::size_t>(n));
    q[linear] = (first_axis < n / 2) ? 1.0 : -1.0;
  }
  sys.set_density("probe", q);
  // The gas state is constant in this MPI fixture, so a sealed input provider is the exact
  // provider-values equivalent of the former derived T_e channel.
  sys.stage_auxiliary_input(temperature_key, std::vector<double>(nn, Te));
  sys.refresh_auxiliary(AuxiliaryEvaluationPoint{"test.mpi-system-solve-fields", 0, 0, 0, 0, 0, 0,
                                                 AuxiliaryEvaluationEvent::initialization});

  // L'APPEL CRITIQUE : sur tous les rangs. Le solve elliptique est collectif ; sans le fix, les
  // rangs sans box locale crashaient ici (fab(0)). On l'enchaine deux fois (ensure_elliptic puis
  // resolve) pour couvrir le chemin construit-puis-reutilise.
  (void)pops::consume_solve_outcome(sys.solve_fields());
  (void)pops::consume_solve_outcome(sys.solve_fields());

  // Resultat sense : potentiel + masse, lus sur le rang proprietaire (par cellule). eval_rhs() se
  // termine par le preflight natif COLLECTIF de finitude : tous les rangs doivent donc l'appeler,
  // meme si le rang sans box renvoie naturellement un buffer local vide. mass() est egalement une
  // reduction COLLECTIVE appelee par tous les rangs.
  const std::vector<double> R = sys.eval_rhs("probe");
  if (owns) {
    const std::vector<double> phi = sys.potential();
    chk(phi.size() == nn, "potential_size");
    double maxabs = 0, sumphi = 0;
    bool finite = true;
    for (double v : phi) {
      if (!std::isfinite(v))
        finite = false;
      maxabs = std::fmax(maxabs, std::fabs(v));
      sumphi += v;
    }
    chk(finite, "potential_finite");
    chk(maxabs > 0.0, "potential_nonzero");  // une charge non nulle -> un potentiel non trivial

    // solve_fields() a peuple le canal aux (phi, grad phi, T_e) : eval_rhs(probe) = -div F + S lit
    // ce canal sans crash et reste fini (la derivation par cellule a bien tourne sur le rang
    // proprietaire ; la correction T_e exacte est verifiee a part par test_aux_te en serie).
    bool rfin = !R.empty();
    for (double r : R)
      rfin = rfin && std::isfinite(r);
    chk(rfin, "rhs_finite");

    std::printf("[rank %d/%d] np=%d  |phi|max=%.3e  sum(phi)=%.3e\n", me, np, np, maxabs, sumphi);
  }

  const double mtot = sys.mass("gas");  // collectif (sum -> all_reduce) : appele par TOUS les rangs
  chk(std::isfinite(mtot), "mass_finite");
  // masse du gaz = rho_gas * produit des cellules classées (réduite collectivement).
  chk(std::fabs(mtot - rho_gas * static_cast<double>(nn)) < 1e-9, "mass_value");

#ifdef POPS_HAS_MPI
  if (np > 1) {
    long g = 0;
    MPI_Allreduce(&fails, &g, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    fails = g;
  }
#endif
  if (me == 0 && fails == 0)
    std::printf("OK test_mpi_system_solve_fields (np=%d)\n", np);
  comm_finalize();
  return fails == 0 ? 0 : 1;
}

TEST(test_mpi_system_solve_fields, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_system_solve_fields,
                                    "test_mpi_system_solve_fields"),
            0);
}
