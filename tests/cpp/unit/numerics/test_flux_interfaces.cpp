#include <gtest/gtest.h>

#include <pops/numerics/fv/numerical_flux.hpp>

#include <cmath>
#include <type_traits>

namespace {

struct ConstantAdvection {
  using State = pops::StateVec<1>;
  static constexpr int dimension = 1;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 1;

  POPS_HD State flux(const State& state, const auto& providers, int) const {
    return State{state[0] * providers.template provider<0>()};
  }
  POPS_HD pops::Real max_wave_speed(const State&, const auto& providers, int) const {
    const pops::Real velocity = providers.template provider<0>();
    return velocity < pops::Real(0) ? -velocity : velocity;
  }
};

struct NoProviderAdvection {
  using State = pops::StateVec<1>;
  static constexpr int dimension = 1;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 0;

  POPS_HD State flux(const State& state, const auto&, int) const { return state; }
  POPS_HD pops::Real max_wave_speed(const State&, const auto&, int) const { return pops::Real(1); }
};

struct SameSpellingOwnerA {
  static constexpr int dimension = 1;
  static constexpr int n_providers = 1;
};
struct SameSpellingOwnerB {
  static constexpr int dimension = 1;
  static constexpr int n_providers = 1;
};

}  // namespace

static_assert(pops::FluxProviderValues<ConstantAdvection>::size == 1);
static_assert(pops::FluxProviderValues<NoProviderAdvection>::size == 0);
static_assert(!std::is_same_v<pops::BoundFluxProviders<SameSpellingOwnerA>,
                              pops::BoundFluxProviders<SameSpellingOwnerB>>);

TEST(FluxProviders, ExactDensePackBindsOnlyDeclaredSlots) {
  pops::FluxProviderValues<ConstantAdvection> values{};
  values[0] = pops::Real(-3);
  const auto providers = pops::bind_flux_providers<ConstantAdvection>(values);
  EXPECT_EQ(providers.template provider<0>(), pops::Real(-3));
  EXPECT_EQ(ConstantAdvection{}.max_wave_speed({pops::Real(2)}, providers, 0), pops::Real(3));
}

TEST(FluxProviders, EmptyPackDoesNotNeedStorageOrAReservedPrefix) {
  const pops::FluxProviderValues<NoProviderAdvection> values{};
  const auto providers = pops::bind_flux_providers<NoProviderAdvection>(values);
  EXPECT_EQ(NoProviderAdvection{}.flux({pops::Real(4)}, providers, 0)[0], pops::Real(4));
}

TEST(FluxProviders, RusanovUsesExplicitProviderValueOnBothTraces) {
  pops::FluxProviderValues<ConstantAdvection> left_values{};
  pops::FluxProviderValues<ConstantAdvection> right_values{};
  left_values[0] = pops::Real(2);
  right_values[0] = pops::Real(2);
  const auto left = pops::bind_flux_providers<ConstantAdvection>(left_values);
  const auto right = pops::bind_flux_providers<ConstantAdvection>(right_values);
  const auto result = pops::evaluate_numerical_flux(
      pops::RusanovFlux{}, ConstantAdvection{}, ConstantAdvection::State{pops::Real(1)}, left,
      ConstantAdvection::State{pops::Real(3)}, right, pops::FaceContext::axis_aligned(0));
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(result.checked_density().value[0], pops::Real(2));
}

TEST(FluxProviders, IndependentConsumersWithHomonymousSlotsRemainIndependent) {
  pops::FluxProviderValues<SameSpellingOwnerA> a_values{};
  pops::FluxProviderValues<SameSpellingOwnerB> b_values{};
  a_values[0] = pops::Real(2);
  b_values[0] = pops::Real(9);
  const auto a = pops::bind_flux_providers<SameSpellingOwnerA>(a_values);
  const auto b = pops::bind_flux_providers<SameSpellingOwnerB>(b_values);
  EXPECT_EQ(a.template provider<0>(), pops::Real(2));
  EXPECT_EQ(b.template provider<0>(), pops::Real(9));
}
