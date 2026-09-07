// M2.3: actual native consumers accept only the methods their selected operation needs.
#include <gtest/gtest.h>

#include <pops/core/model/equation_block.hpp>
#include <pops/core/model/physical_model.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>

#include <utility>

namespace {
using namespace pops;

template <int Dim>
struct SourceOnly {
  using State = StateVec<2>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 2;
  static constexpr int n_providers = 1;

  POPS_HD State source(const State& state, const ProviderValues<1>& providers) const {
    return {providers[0] - Real(2) * state[0], Real(2) * state[0] - state[1]};
  }
};

template <int Dim>
struct TransportOnly {
  using State = StateVec<1>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 0;

  template <class Providers>
  POPS_HD State flux(const State& state, const Providers&, int axis) const {
    return {Real(axis + 1) * state[0]};
  }
  template <class Providers>
  POPS_HD Real max_wave_speed(const State&, const Providers&, int axis) const {
    return Real(axis + 1);
  }
};

template <int Dim>
struct LegacyAggregate : TransportOnly<Dim> {
  using State = typename TransportOnly<Dim>::State;
  POPS_HD State source(const State& state, const ProviderValues<0>&) const {
    return {Real(7) * state[0]};
  }
  POPS_HD Real elliptic_rhs(const State& state) const { return -state[0]; }
};

struct IncorrectSource : SourceOnly<1> {
  POPS_HD Real source(const State&, const ProviderValues<1>&) const { return Real(0); }
};

struct IncorrectProviderPack : SourceOnly<1> {
  POPS_HD State source(const State&, const ProviderValues<2>&) const { return {}; }
};

struct NegativeProviderCount : SourceOnly<1> {
  static constexpr int n_providers = -1;
};

struct RecoverableSourceOnly : SourceOnly<kNativeDimension> {
  using Prim = StateVec<2>;
  POPS_HD Prim to_primitive(const State& state) const { return {state[0], state[1] / state[0]}; }
  POPS_HD State to_conservative(const Prim& primitive) const {
    return {primitive[0], primitive[0] * primitive[1]};
  }
};

static_assert(PhysicalSourceFor<SourceOnly<1>, 1>);
static_assert(PhysicalSourceFor<SourceOnly<2>, 2>);
static_assert(PhysicalSourceFor<SourceOnly<3>, 3>);
static_assert(!PhysicalModelFor<SourceOnly<1>, 1>);
static_assert(!PhysicalTransportFor<SourceOnly<1>, 1>);
static_assert(!PhysicalEllipticRhsFor<SourceOnly<1>, 1>);
static_assert(!PhysicalSourceFor<SourceOnly<1>, 2>);
static_assert(!PhysicalSourceFor<SourceOnly<1>, 4>);
static_assert(!PhysicalSourceFor<IncorrectSource, 1>);
static_assert(!PhysicalSourceFor<IncorrectProviderPack, 1>);
static_assert(!PhysicalSourceFor<NegativeProviderCount, 1>);
static_assert(PhysicalTransportFor<TransportOnly<1>, 1>);
static_assert(!PhysicalModelFor<TransportOnly<1>, 1>);
static_assert(!PhysicalSourceFor<TransportOnly<1>, 1>);
static_assert(PhysicalModelFor<LegacyAggregate<1>, 1>);
static_assert(PhysicalModelFor<LegacyAggregate<2>, 2>);
static_assert(PhysicalModelFor<LegacyAggregate<3>, 3>);
static_assert(EquationBlockLike<EquationBlock<1, SourceOnly<1>>>);
static_assert(EquationBlockLike<EquationBlock<2, SourceOnly<2>>>);
static_assert(EquationBlockLike<EquationBlock<3, TransportOnly<3>>>);
static_assert(PhysicalFlux<PhysicalFluxView<TransportOnly<1>>>);
static_assert(PhysicalFlux<PhysicalFluxView<TransportOnly<2>>>);
static_assert(PhysicalFlux<PhysicalFluxView<TransportOnly<3>>>);
static_assert(HasPrimitiveVars<RecoverableSourceOnly>);
static_assert(!PhysicalModel<RecoverableSourceOnly>);

template <int Dim>
void materialize_source_only() {
  Extent<Dim> extent{};
  for (int axis = 0; axis < Dim; ++axis)
    extent[axis] = 4;
  const auto box = Box<Dim>::from_extents(extent);
  Fab<Dim> state(box, 2), forcing(box, 1), source(box, 2), status(box, 1);
  state.set_val(Real(3));
  forcing.set_val(Real(4));
  source.set_val(Real(-100));
  status.set_val(Real(-100));
  ProviderStorageView<Dim, 1> providers{};
  providers.storage[0] = std::as_const(forcing).view();

  // This is the production source materializer, submitted on an explicit execution instance
  // so even a small domain runs through Kokkos rather than the host small-box convenience path.
  const Kokkos::DefaultExecutionSpace execution{};
  for_each_cell(execution, box,
                generated_system_detail::MaterializeSource<Dim, SourceOnly<Dim>>{
                    {}, std::as_const(state).view(), providers, source.view(), status.view()});
  execution.fence();
  auto source_host = source.create_host_mirror();
  auto status_host = status.create_host_mirror();
  source.copy_to_host(source_host);
  status.copy_to_host(status_host);
  const auto cells = static_cast<std::size_t>(box.numPts());
  for (std::size_t cell = 0; cell < cells; ++cell) {
    EXPECT_EQ(source_host(cell), Real(-2));
    EXPECT_EQ(source_host(cells + cell), Real(3));
    EXPECT_EQ(status_host(cell), Real(0));
  }
}

template <int Dim>
void transport_only_flux() {
  using Transport = TransportOnly<Dim>;
  const auto providers = bind_flux_providers<Transport>(FluxProviderValues<Transport>{});
  for (int axis = 0; axis < Dim; ++axis) {
    const auto face = FaceContext::axis_aligned(axis);
    const auto flux = evaluate_numerical_flux(RusanovFlux{}, Transport{}, StateVec<1>{Real(3)},
                                              providers, StateVec<1>{Real(5)}, providers, face);
    ASSERT_TRUE(flux.succeeded());
    EXPECT_EQ(flux.checked_density().value[0], Real(3 * (axis + 1)));
    EXPECT_EQ(flux.stability.value, Real(axis + 1));
  }
}

TEST(NarrowPhysicalOperations, SourceOnlyUsesTheProductionRankedCellKernel) {
  materialize_source_only<1>();
  materialize_source_only<2>();
  materialize_source_only<3>();
}

TEST(NarrowPhysicalOperations, TransportOnlyUsesExistingPhysicalFluxAndRusanovContracts) {
  transport_only_flux<1>();
  transport_only_flux<2>();
  transport_only_flux<3>();
}

TEST(NarrowPhysicalOperations, LegacyAggregateRetainsMeaningfulOperations) {
  const LegacyAggregate<kNativeDimension> model{};
  const StateVec<1> state{Real(3)};
  const ProviderValues<0> providers{};
  EXPECT_EQ(model.flux(state, providers, 0)[0], Real(3));
  EXPECT_EQ(model.source(state, providers)[0], Real(21));
  EXPECT_EQ(model.elliptic_rhs(state), Real(-3));
}

TEST(NarrowPhysicalOperations, SourceOnlyRecoveryUsesDeclaredConversionInsteadOfIdentity) {
  PreparedModelVariableInversionRecovery<RecoverableSourceOnly> recovery(RecoverableSourceOnly{});
  const Real conservative[2] = {Real(2), Real(6)};
  const auto result = recovery.recover(conservative);
  ASSERT_EQ(result.outcome.status, RecoveryStatus::kRecovered);
  EXPECT_EQ(result.outcome.value[0], Real(2));
  EXPECT_EQ(result.outcome.value[1], Real(3));
}
}  // namespace
