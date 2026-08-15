// Squelette architecture multi-blocs : PhysicalModel local, EquationBlock,
// CoupledSystem, scheduler par sous-pas, RHS elliptique multi-champs.

#include <gtest/gtest.h>

#include <pops/core/model/coupled_system.hpp>
#include <pops/core/state/state.hpp>
#include <pops/coupling/base/elliptic_rhs.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include "reference_time_scheduler.hpp"

#include <cmath>
#include <type_traits>
#include <vector>

using namespace pops;

template <int Dim>
Extent<Dim> filled_extent(std::int64_t value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
struct ReadScalar {
  FieldView<const Real, Dim> field{};
  POPS_HD Real operator()(const Index<Dim>& cell) const { return field(cell); }
};

template <int Dim>
Real sum_field(const MultiFab<Dim>& field) {
  Real result = Real(0);
  for (std::size_t local = 0; local < field.local_size(); ++local)
    result += for_each_cell_reduce_sum(field.box(local), ReadScalar<Dim>{field.fab(local).view()});
  return result;
}

template <int Dim>
struct ElectronToy {
  using State = StateVec<1>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 0;
  POPS_HD State flux(const State&, const auto&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return Real(0); }
  POPS_HD State source(const State&, const ProviderValues<0>&) const { return State{Real(0)}; }
  POPS_HD Real elliptic_rhs(const State& u) const { return -u[0]; }
};

template <int Dim>
struct IonToy {
  using State = StateVec<1>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 0;
  POPS_HD State flux(const State&, const auto&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return Real(0); }
  POPS_HD State source(const State&, const ProviderValues<0>&) const { return State{Real(0)}; }
  POPS_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

template <int Dim>
using ElectronBlock =
    EquationBlock<Dim, ElectronToy<Dim>, MusclVanLeerHLLC, ImplicitTime<UserTimeIntegrator, 10>>;

template <int Dim>
using IonBlock = EquationBlock<Dim, IonToy<Dim>, MusclMinmod, ExplicitTime<SSPRK2, 1>>;

template <int Dim>
void check_ranked_elliptic_rhs() {
  static_assert(EquationBlockLike<ElectronBlock<Dim>>);
  static_assert(EquationBlockLike<IonBlock<Dim>>);
  static_assert(ElectronBlock<Dim>::Time::treatment == TimeTreatment::Implicit);
  static_assert(ElectronBlock<Dim>::Time::substeps == 10);
  static_assert(IonBlock<Dim>::Time::treatment == TimeTreatment::Explicit);
  static_assert(std::is_same_v<typename ElectronBlock<Dim>::Spatial::NumericalFlux, HLLCFlux>);
  static_assert(std::is_same_v<typename IonBlock<Dim>::Spatial::NumericalFlux, RusanovFlux>);

  const auto extent = filled_extent<Dim>(4);
  const auto domain = Box<Dim>::from_extents(extent);
  const mesh::BoxArray<Dim> layout(std::vector<Box<Dim>>{domain});
  const mesh::RankSpace<Dim> ranks(Index<Dim>{}, filled_extent<Dim>(1));
  const auto distribution = mesh::Distribution<Dim>::replicated(layout, ranks);
  const Index<Dim> local_rank{};

  MultiFab<Dim> Ue(layout, distribution, local_rank, 1, filled_extent<Dim>(0));
  MultiFab<Dim> Ui(layout, distribution, local_rank, 1, filled_extent<Dim>(0));
  MultiFab<Dim> rhs(layout, distribution, local_rank, 1, filled_extent<Dim>(0));
  Ue.set_val(2.0);
  Ui.set_val(5.0);

  ElectronBlock<Dim> electrons{"electrons", ElectronToy<Dim>{}, Ue};
  IonBlock<Dim> ions{"ions", IonToy<Dim>{}, Ui};
  CoupledSystem system{electrons, ions};
  static_assert(CoupledSystemLike<decltype(system)>);
  EXPECT_TRUE(decltype(system)::n_blocks == 2) << "two_blocks";

  int ne = 0, ni = 0;
  Real dte = 0, dti = 0;
  pops::test_support::advance_subcycled(system, Real(0.2), [&](auto& block, Real h, int, int) {
    using M = typename std::decay_t<decltype(block)>::Model;
    if constexpr (std::is_same_v<M, ElectronToy<Dim>>) {
      ++ne;
      dte += h;
    } else if constexpr (std::is_same_v<M, IonToy<Dim>>) {
      ++ni;
      dti += h;
    }
  });
  EXPECT_TRUE(ne == 10) << "electron_substeps";
  EXPECT_TRUE(ni == 1) << "ion_substeps";
  EXPECT_TRUE(std::fabs(dte - 0.2) < 1e-12) << "electron_dt_sum";
  EXPECT_TRUE(std::fabs(dti - 0.2) < 1e-12) << "ion_dt_sum";

  TwoFieldChargeDensityRhs<Dim> charge;
  charge.q0 = Real(-1);  // electrons
  charge.q1 = Real(1);   // ions
  charge(Ue, Ui, rhs);
  EXPECT_TRUE(std::fabs(sum_field(rhs) - Real(3) * Real(domain.numPts())) < 1e-12)
      << "charge_density_rhs";
}

TEST(SystemAbstraction, CoupledSystemSubcyclesBlocksAndAssemblesChargeDensityRhsInEveryRank) {
  check_ranked_elliptic_rhs<1>();
  check_ranked_elliptic_rhs<2>();
  check_ranked_elliptic_rhs<3>();
}
