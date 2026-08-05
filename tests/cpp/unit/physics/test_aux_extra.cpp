// Exact-ranked auxiliary consumers distinguish the required gradient base from named extras.

#include <gtest/gtest.h>

#include <pops/core/model/physical_model.hpp>
#include <pops/core/state/state.hpp>

using namespace pops;

namespace {

template <int Dim>
struct BzSource {
  using State = StateVec<1>;
  using Aux = AuxState<Dim>;
  static constexpr int dimension = Dim;
  static constexpr int n_aux = AuxComponentLayout<Dim>::b_z + 1;

  POPS_HD State apply(const State& state, const Aux& auxiliary) const {
    return State{auxiliary.B_z * state[0]};
  }
};

template <int Dim>
struct FirstGradientSource {
  using State = StateVec<1>;
  using Aux = AuxState<Dim>;
  static constexpr int dimension = Dim;

  POPS_HD State apply(const State& state, const Aux& auxiliary) const {
    return State{auxiliary.template gradient<0>() * state[0]};
  }
};

template <int Dim>
void check_ranked_extra_is_explicit() {
  using layout = AuxComponentLayout<Dim>;
  static_assert(aux_comps_for<BzSource<Dim>, Dim>() == layout::b_z + 1);
  static_assert(aux_comps_for<FirstGradientSource<Dim>, Dim>() == layout::base_components);

  const StateVec<1> state{Real(2)};
  AuxState<Dim> auxiliary{};
  auxiliary.template gradient<0>() = Real(0.3);
  auxiliary.B_z = Real(0.7);

  EXPECT_EQ(BzSource<Dim>{}.apply(state, auxiliary)[0], Real(1.4));
  const Real base_result = FirstGradientSource<Dim>{}.apply(state, auxiliary)[0];
  auxiliary.B_z = Real(999);
  EXPECT_EQ(FirstGradientSource<Dim>{}.apply(state, auxiliary)[0], base_result);
  EXPECT_EQ(base_result, Real(0.6));
}

}  // namespace

TEST(AuxExtra, BzWidthAndBaseIsolationAreExactInEveryRank) {
  check_ranked_extra_is_explicit<1>();
  check_ranked_extra_is_explicit<2>();
  check_ranked_extra_is_explicit<3>();
}
