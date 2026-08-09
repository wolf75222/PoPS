// Chantier "Aux extensible", increment 5 : le runtime System (ce que pilote adc_cases) cable B_z.
// Un bloc COMPILE qui lit B_z (CompositeModel a source magnetisee, n_aux=4) elargit le canal aux
// PARTAGE du System (ensure_aux_width via add_compiled_model) ; set_magnetic_field(...) peuple la
// composante B_z. Le residu de production (eval_rhs = -div F + S sur les vrais MultiFab) voit alors
// B_z. On verifie le chemin de bout en bout cote runtime :
//   modele MagModel = CompositeModel<CartesianExBDrift, BzSource, NoEll>, flux ExB (grad=0 -> nul),
//   source S = B_z u, elliptic_rhs nul (phi=0) -> eval_rhs = B_z u = c (densite 1, B_z constant c).

#include <gtest/gtest.h>

#include <pops/physics/composition/composite.hpp>
#include <pops/physics/bricks/hyperbolic.hpp>            // CartesianExBDrift
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/system.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>

#include <array>
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

using runtime::system::AuxiliaryComponentContract;
using runtime::system::AuxiliaryComponentKey;
using runtime::system::AuxiliaryConsumerProviderPlan;
using runtime::system::AuxiliaryEvaluationEvent;
using runtime::system::AuxiliaryEvaluationPoint;
using runtime::system::AuxiliaryFreshness;
using runtime::system::AuxiliaryOutput;
using runtime::system::AuxiliaryProviderKind;
using runtime::system::AuxiliaryStorageShape;
using runtime::system::PreparedAuxiliaryProvider;

// Shares the Cartesian ExB B_z provider slot (grad[2] then B[3]).
struct BzSource {
  static constexpr int n_providers = 5;
  template <class State, class Providers>
  POPS_HD State apply(const State& u, const Providers& providers) const {
    State s{};
    s[0] = provider_value<4>(providers) * u[0];
    return s;
  }
};
struct NoEll {
  template <class State>
  POPS_HD Real rhs(const State&) const {
    return Real(0);
  }
};

using MagModel = CompositeModel<CartesianExBDrift, BzSource, NoEll>;
static_assert(MagModel::n_providers == 5, "the composite consumes grad[2] + B[3]");

TEST(AuxRuntimeBz, RuntimeSystemReadsSharedBzChannelAndClearsWithZero) {
#if defined(POPS_HAS_KOKKOS)
  static Kokkos::ScopeGuard guard;  // Kokkos init AVANT la 1ere allocation (ctor System)
#endif

  const int n = 32;
  const double c = 0.7;
  std::vector<double> ones(static_cast<std::size_t>(n) * n, 1.0);
  std::vector<double> bz(static_cast<std::size_t>(n) * n, c);  // B_z = c partout

  SystemConfig<kNativeDimension> cfg;
  for (int axis = 0; axis < kNativeDimension; ++axis) {
    cfg.shape[axis] = n;
    cfg.lower[axis] = Real(0);
    cfg.upper[axis] = Real(1);
    cfg.periodicity[axis] = true;
  }

  System<kNativeDimension> sys(cfg);
  const AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "input", "scalar"};
  AuxiliaryStorageShape<kNativeDimension> shape;
  for (int axis = 0; axis < kNativeDimension; ++axis)
    shape.halo[axis] = 2;
  std::array<AuxiliaryComponentKey, 5> keys{
      AuxiliaryComponentKey{"test.exb", "field", "phi", "grad-x"},
      AuxiliaryComponentKey{"test.exb", "field", "phi", "grad-y"},
      AuxiliaryComponentKey{"test.exb", "input", "magnetic", "B-x"},
      AuxiliaryComponentKey{"test.exb", "input", "magnetic", "B-y"},
      AuxiliaryComponentKey{"test.exb", "input", "magnetic", "B-z"}};
  std::vector<AuxiliaryOutput<kNativeDimension>> outputs;
  for (const auto& key : keys)
    outputs.push_back({key, contract, shape});
  sys.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<kNativeDimension>{
      "test.exb.providers",
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      std::move(outputs),
      {}});
  AuxiliaryConsumerProviderPlan<kNativeDimension> plan;
  plan.consumer_qid = "a";
  for (std::size_t slot = 0; slot < keys.size(); ++slot)
    plan.values.push_back({{keys[slot], contract, shape}, slot});
  sys.install_auxiliary_consumer_plan(std::move(plan));
  sys.seal_auxiliary_providers();
  sys.install_block_state_route("a", "test.exb.a.state@1");
  add_compiled_model(sys, "a", MagModel{}, "minmod", "rusanov", "conservative", "explicit");
  sys.set_state("a", ones);
  std::vector<double> zero(static_cast<std::size_t>(n) * n, 0.0);
  sys.stage_auxiliary_input(keys[0], zero);
  sys.stage_auxiliary_input(keys[1], zero);
  sys.stage_auxiliary_input(keys[2], zero);
  sys.stage_auxiliary_input(keys[3], zero);
  sys.stage_auxiliary_input(keys[4], bz);
  sys.refresh_auxiliary(AuxiliaryEvaluationPoint{"test.exb", 0, 0, 0, 0, 0, 0,
                                                 AuxiliaryEvaluationEvent::initialization});

  // eval_rhs = -div F + S. flux ExB(grad=0)=0 -> R = source = B_z u = c.
  const std::vector<double> R = sys.eval_rhs("a");
  double maxerr = 0;
  for (double r : R)
    maxerr = std::fmax(maxerr, std::fabs(r - c));
  EXPECT_TRUE(maxerr < 1e-12) << "runtime_system_reads_Bz (max|R - B_z| = " << maxerr
                              << ", providers=" << MagModel::n_providers << ")";

  // A zero vector has no E x B direction: the same accepted provider plan must fail closed.
  sys.stage_auxiliary_input(keys[4], zero);
  sys.refresh_auxiliary(AuxiliaryEvaluationPoint{"test.exb", 1, 0, 0, 0, 0, 0,
                                                 AuxiliaryEvaluationEvent::initialization});
  EXPECT_THROW((void)sys.eval_rhs("a"), std::runtime_error);
}
