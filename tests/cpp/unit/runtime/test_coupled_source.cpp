/// @file
/// @brief Exact-ranked inter-block source contract without a second runtime/time authority.

#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/model/coupled_system.hpp>
#include <pops/core/state/state.hpp>
#include <pops/coupling/source/coupled_source.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <vector>

namespace {

constexpr int kDim = pops::kNativeDimension;
using Field = pops::MultiFab<kDim>;

template <int Dim>
pops::Extent<Dim> filled_extent(std::int64_t value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

struct Inert {
  using State = pops::StateVec<1>;
  using Aux = pops::AuxState<kDim>;
  static constexpr int n_vars = 1;

  POPS_HD State flux(const State&, const Aux&, int) const { return State{pops::Real(0)}; }
  POPS_HD pops::Real max_wave_speed(const State&, const Aux&, int) const { return pops::Real(0); }
  POPS_HD State source(const State&, const Aux&) const { return State{pops::Real(0)}; }
  POPS_HD pops::Real elliptic_rhs(const State& state) const { return state[0]; }
};

template <int Dim>
struct ExchangeKernel {
  pops::FieldView<pops::Real, Dim> first{};
  pops::FieldView<pops::Real, Dim> second{};
  pops::Real coefficient = pops::Real(0);

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    const pops::Real transfer = coefficient * (second(cell) - first(cell));
    first(cell) += transfer;
    second(cell) -= transfer;
  }
};

template <int Dim>
struct ReadScalar {
  pops::FieldView<const pops::Real, Dim> field{};
  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const { return field(cell); }
};

struct LinearExchange {
  pops::Real rate = pops::Real(0.5);

  template <pops::CoupledSystemLike System>
  void apply(System& system, const typename System::field_type&, pops::Real dt) const {
    static_assert(System::dimension == kDim);
    auto& first = system.template block<0>().U();
    auto& second = system.template block<1>().U();
    if (first.layout() != second.layout() || first.distribution() != second.distribution() ||
        first.local_rank() != second.local_rank())
      throw std::invalid_argument(
          "LinearExchange requires one exact co-distributed field identity");
    for (std::size_t local = 0; local < first.local_size(); ++local)
      pops::for_each_cell(
          first.box(local),
          ExchangeKernel<kDim>{first.fab(local).view(), second.fab(local).view(), rate * dt});
  }
};

pops::Real sum_field(const Field& field) {
  pops::Real result = pops::Real(0);
  for (std::size_t local = 0; local < field.local_size(); ++local)
    result +=
        pops::for_each_cell_reduce_sum(field.box(local), ReadScalar<kDim>{field.fab(local).view()});
  return result;
}

using Block =
    pops::EquationBlock<kDim, Inert, pops::FirstOrder, pops::ExplicitTime<pops::SSPRK2, 1>>;
using System = pops::CoupledSystem<Block, Block>;

static_assert(pops::CoupledSourceFor<LinearExchange, System>);
static_assert(pops::CoupledSourceFor<pops::NoCoupledSource, System>);
static_assert(System::dimension == kDim);

TEST(CoupledSource, ExactRankExchangeConservesTotalMass) {
  const auto extent = filled_extent<kDim>(4);
  const auto domain = pops::Box<kDim>::from_extents(extent);
  const pops::mesh::BoxArray<kDim> layout(std::vector<pops::Box<kDim>>{domain});
  const pops::mesh::RankSpace<kDim> ranks(pops::Index<kDim>{}, filled_extent<kDim>(1));
  const auto distribution = pops::mesh::Distribution<kDim>::replicated(layout, ranks);
  const pops::Index<kDim> local_rank{};

  Field first(layout, distribution, local_rank, 1, filled_extent<kDim>(0));
  Field second(layout, distribution, local_rank, 1, filled_extent<kDim>(0));
  Field auxiliary(layout, distribution, local_rank, 1, filled_extent<kDim>(0));
  first.set_val(pops::Real(1));
  second.set_val(pops::Real(3));
  auxiliary.set_val(pops::Real(0));

  Block first_block{"first", Inert{}, first};
  Block second_block{"second", Inert{}, second};
  System system{first_block, second_block};
  const pops::Real initial_total = sum_field(first) + sum_field(second);

  LinearExchange{pops::Real(0.5)}.apply(system, auxiliary, pops::Real(0.1));

  const pops::Real cells = static_cast<pops::Real>(domain.numPts());
  EXPECT_NEAR(sum_field(first), pops::Real(1.1) * cells, pops::Real(1e-12));
  EXPECT_NEAR(sum_field(second), pops::Real(2.9) * cells, pops::Real(1e-12));
  EXPECT_NEAR(sum_field(first) + sum_field(second), initial_total, pops::Real(1e-12));

  const pops::Real first_before = sum_field(first);
  const pops::Real second_before = sum_field(second);
  pops::NoCoupledSource{}.apply(system, auxiliary, pops::Real(0.1));
  EXPECT_EQ(sum_field(first), first_before);
  EXPECT_EQ(sum_field(second), second_before);
}

}  // namespace
