// Squelette architecture multi-blocs : PhysicalModel local, EquationBlock,
// CoupledSystem, scheduler par sous-pas, RHS elliptique multi-champs.

#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
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

constexpr int kDim = kNativeDimension;

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

struct ElectronToy {
  using State = StateVec<1>;
  using Aux = pops::AuxState<kDim>;
  static constexpr int n_vars = 1;
  POPS_HD State flux(const State&, const auto&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return Real(0); }
  POPS_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  POPS_HD Real elliptic_rhs(const State& u) const { return -u[0]; }
};

struct IonToy {
  using State = StateVec<1>;
  using Aux = pops::AuxState<kDim>;
  static constexpr int n_vars = 1;
  POPS_HD State flux(const State&, const auto&, int) const { return State{Real(0)}; }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return Real(0); }
  POPS_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  POPS_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

using ElectronBlock =
    EquationBlock<kDim, ElectronToy, MusclVanLeerHLLC, ImplicitTime<UserTimeIntegrator, 10>>;
using IonBlock = EquationBlock<kDim, IonToy, MusclMinmod, ExplicitTime<SSPRK2, 1>>;

static_assert(EquationBlockLike<ElectronBlock>);
static_assert(EquationBlockLike<IonBlock>);
static_assert(ElectronBlock::Time::treatment == TimeTreatment::Implicit);
static_assert(ElectronBlock::Time::substeps == 10);
static_assert(IonBlock::Time::treatment == TimeTreatment::Explicit);
static_assert(std::is_same_v<ElectronBlock::Spatial::NumericalFlux, HLLCFlux>);
static_assert(std::is_same_v<IonBlock::Spatial::NumericalFlux, RusanovFlux>);

TEST(SystemAbstraction, CoupledSystemSubcyclesBlocksAndAssemblesChargeDensityRhs) {
  const auto extent = filled_extent<kDim>(4);
  const auto domain = Box<kDim>::from_extents(extent);
  const mesh::BoxArray<kDim> layout(std::vector<Box<kDim>>{domain});
  const mesh::RankSpace<kDim> ranks(Index<kDim>{}, filled_extent<kDim>(1));
  const auto distribution = mesh::Distribution<kDim>::replicated(layout, ranks);
  const Index<kDim> local_rank{};

  MultiFab<kDim> Ue(layout, distribution, local_rank, 1, filled_extent<kDim>(0));
  MultiFab<kDim> Ui(layout, distribution, local_rank, 1, filled_extent<kDim>(0));
  MultiFab<kDim> rhs(layout, distribution, local_rank, 1, filled_extent<kDim>(0));
  Ue.set_val(2.0);
  Ui.set_val(5.0);

  ElectronBlock electrons{"electrons", ElectronToy{}, Ue};
  IonBlock ions{"ions", IonToy{}, Ui};
  CoupledSystem system{electrons, ions};
  static_assert(CoupledSystemLike<decltype(system)>);
  EXPECT_TRUE(decltype(system)::n_blocks == 2) << "two_blocks";

  int ne = 0, ni = 0;
  Real dte = 0, dti = 0;
  pops::test_support::advance_subcycled(system, Real(0.2), [&](auto& block, Real h, int, int) {
    using M = typename std::decay_t<decltype(block)>::Model;
    if constexpr (std::is_same_v<M, ElectronToy>) {
      ++ne;
      dte += h;
    } else if constexpr (std::is_same_v<M, IonToy>) {
      ++ni;
      dti += h;
    }
  });
  EXPECT_TRUE(ne == 10) << "electron_substeps";
  EXPECT_TRUE(ni == 1) << "ion_substeps";
  EXPECT_TRUE(std::fabs(dte - 0.2) < 1e-12) << "electron_dt_sum";
  EXPECT_TRUE(std::fabs(dti - 0.2) < 1e-12) << "ion_dt_sum";

  TwoFieldChargeDensityRhs<kDim> charge;
  charge.q0 = Real(-1);  // electrons
  charge.q1 = Real(1);   // ions
  charge(Ue, Ui, rhs);
  EXPECT_TRUE(std::fabs(sum_field(rhs) - Real(3) * Real(domain.numPts())) < 1e-12)
      << "charge_density_rhs";
}
