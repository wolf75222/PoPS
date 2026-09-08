#include <gtest/gtest.h>

#include <pops/runtime/builders/compiled/generated_system_block.hpp>

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

template <int Dim, int Providers = 0>
struct WaveModel {
  using State = pops::StateVec<1>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = Providers;

  template <int Axis>
  POPS_HD State flux(const State& state, const auto&) const {
    return state;
  }
  template <int Axis>
  POPS_HD pops::Real max_wave_speed(const State&, const auto&) const {
    return pops::Real(Axis + 1);
  }
};

template <int Dim>
struct RankedStability : WaveModel<Dim> {
  int invalid_axis = -1;
  pops::Real invalid_value = 0;

  template <int Axis>
  POPS_HD pops::Real stability_speed(const typename WaveModel<Dim>::State& state,
                                     const auto&) const {
    return Axis == invalid_axis ? invalid_value : pops::Real(4 * (Axis + 1)) + state[0];
  }
};

template <int Dim>
struct ProviderStability : WaveModel<Dim, 2> {
  POPS_HD pops::Real stability_speed(const typename WaveModel<Dim, 2>::State&,
                                     const auto& providers, int axis) const {
    return providers.template provider<1>() + pops::Real(axis);
  }
};

template <int Dim>
void check_ranked_dispatch() {
  const typename WaveModel<Dim>::State state{0};
  const WaveModel<Dim> wave{};
  const auto fallback = pops::bind_flux_providers<WaveModel<Dim>>({});
  EXPECT_EQ((pops::generated_system_detail::maximum_axis_speed<0, Dim>(wave, state, fallback)),
            pops::Real(Dim));
  const RankedStability<Dim> stable{};
  const auto providers = pops::bind_flux_providers<RankedStability<Dim>>({});
  EXPECT_EQ((pops::generated_system_detail::maximum_axis_speed<0, Dim>(stable, state, providers)),
            pops::Real(4 * Dim));
  EXPECT_EQ(pops::detail::model_max_wave_speed_at<0>(stable, state, providers), pops::Real(1));
  const auto face = pops::evaluate_numerical_flux(
      pops::RusanovFlux{}, stable, typename WaveModel<Dim>::State{1}, providers,
      typename WaveModel<Dim>::State{3}, providers, pops::FaceContext::axis_aligned(0));
  ASSERT_TRUE(face.succeeded());
  EXPECT_EQ(face.checked_density().value[0], pops::Real(1));
}

template <int Dim>
void check_exact_provider_dispatch() {
  pops::FluxProviderValues<ProviderStability<Dim>> values{};
  values[0] = pops::Real(-1000);  // unrelated slot must not become the requested coefficient
  values[1] = pops::Real(7);
  const auto providers = pops::bind_flux_providers<ProviderStability<Dim>>(values);
  static_assert(decltype(providers)::value_count == 2);
  static_assert(
      !std::is_convertible_v<decltype(providers), pops::BoundFluxProviders<WaveModel<Dim, 2>>>);
  EXPECT_EQ((pops::generated_system_detail::maximum_axis_speed<0, Dim>(
                ProviderStability<Dim>{}, typename WaveModel<Dim>::State{0}, providers)),
            pops::Real(7 + Dim - 1));
}

template <int Dim>
void check_invalid_axis() {
  const auto providers = pops::bind_flux_providers<RankedStability<Dim>>({});
  for (int axis = 0; axis < Dim; ++axis)
    for (const auto bad : {pops::Real(-1), std::numeric_limits<pops::Real>::quiet_NaN(),
                           std::numeric_limits<pops::Real>::infinity()}) {
      RankedStability<Dim> model;
      model.invalid_axis = axis;
      model.invalid_value = bad;
      EXPECT_TRUE(std::isinf(pops::generated_system_detail::maximum_axis_speed<0, Dim>(
          model, typename WaveModel<Dim>::State{0}, providers)));
    }
}

void check_collective_reduction() {
  constexpr int Dim = pops::kNativeDimension;
  const auto lane = pops::ExecutionLane::world("tests.generated-stability-speed");
  pops::Index<Dim> zero{};
  pops::Extent<Dim> rank_extent{};
  for (int axis = 0; axis < Dim; ++axis)
    rank_extent[axis] = axis == 0 ? lane.size() : 1;
  std::vector<pops::Box<Dim>> boxes;
  std::vector<pops::Index<Dim>> owners;
  for (int rank = 0; rank < lane.size(); ++rank) {
    pops::Index<Dim> lo{}, hi{}, owner{};
    for (int axis = 0; axis < Dim; ++axis)
      hi[axis] = 3;
    lo[0] = 4 * rank;
    hi[0] = 4 * rank + 3;
    owner[0] = rank;
    boxes.push_back({lo, hi});
    owners.push_back(owner);
  }
  const pops::mesh::BoxArray<Dim> layout(boxes);
  const pops::mesh::RankSpace<Dim> ranks{zero, rank_extent};
  const auto distribution = pops::mesh::Distribution<Dim>::partitioned(layout, ranks, owners);
  auto local_rank = zero;
  local_rank[0] = lane.rank();
  pops::MultiFab<Dim> state(layout, distribution, local_rank, 1, pops::Extent<Dim>{});
  state.set_val(pops::Real(lane.rank()));
  RankedStability<Dim> model;
  const auto speed =
      pops::generated_system_detail::maximum_speed<Dim>(model, state, nullptr, nullptr, lane);
  EXPECT_EQ(speed, pops::Real(4 * Dim + lane.size() - 1));
  // Only the last rank supplies an invalid axis. Every participant must refuse the same bound.
  if (lane.rank() == lane.size() - 1) {
    model.invalid_axis = 0;
    model.invalid_value = std::numeric_limits<pops::Real>::quiet_NaN();
  }
  EXPECT_THROW(
      (pops::generated_system_detail::maximum_speed<Dim>(model, state, nullptr, nullptr, lane)),
      std::runtime_error);
  model.invalid_axis = -1;
  EXPECT_EQ(
      (pops::generated_system_detail::maximum_speed<Dim>(model, state, nullptr, nullptr, lane)),
      speed);
}

}  // namespace

TEST(test_generated_stability_speed, ranked_trait_replaces_only_timestep_speed) {
  check_ranked_dispatch<1>();
  check_ranked_dispatch<2>();
  check_ranked_dispatch<3>();
}

TEST(test_generated_stability_speed, runtime_axis_trait_uses_exact_provider_slot) {
  check_exact_provider_dispatch<1>();
  check_exact_provider_dispatch<2>();
  check_exact_provider_dispatch<3>();
}

TEST(test_generated_stability_speed, invalid_axis_cannot_hide_in_another_axis_maximum) {
  check_invalid_axis<1>();
  check_invalid_axis<2>();
  check_invalid_axis<3>();
}

TEST(test_generated_stability_speed, native_rank_reduction_refuses_one_rank_invalid_then_recovers) {
  check_collective_reduction();
}
