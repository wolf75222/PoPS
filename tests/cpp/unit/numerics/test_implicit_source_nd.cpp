#include <gtest/gtest.h>

#include <pops/numerics/time/integrators/implicit_stepper.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using namespace pops;

namespace {

template <int Dim>
Extent<Dim> uniform_extent(int value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
MultiFab<Dim> one_patch_field(const Box<Dim>& box, int ncomp) {
  const mesh::BoxArray<Dim> layout(std::vector<Box<Dim>>{box});
  const mesh::RankSpace<Dim> ranks(Index<Dim>{}, uniform_extent<Dim>(1));
  const auto distribution = mesh::Distribution<Dim>::replicated(layout, ranks);
  return MultiFab<Dim>(layout, distribution, Index<Dim>{}, ncomp, Extent<Dim>{});
}

template <int Dim>
std::size_t host_offset(const Box<Dim>& storage, const Index<Dim>& index, int component) {
  std::int64_t linear = 0;
  std::int64_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    linear += static_cast<std::int64_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= storage.length(axis);
  }
  return static_cast<std::size_t>(component * storage.numPts() + linear);
}

template <int Dim>
void expect_valid_value(const Fab<Dim>& field, Real expected) {
  auto host = field.create_host_mirror();
  field.copy_to_host(host);
  for (std::int64_t linear = 0; linear < field.box().numPts(); ++linear) {
    std::int64_t remaining = linear;
    Index<Dim> index{};
    for (int axis = 0; axis < Dim; ++axis) {
      index[axis] = field.box().lo[axis] + static_cast<int>(remaining % field.box().length(axis));
      remaining /= field.box().length(axis);
    }
    EXPECT_NEAR(host(host_offset(field.grown_box(), index, 0)), expected,
                Real(64) * std::numeric_limits<Real>::epsilon());
  }
}

template <int Dim>
struct RankedLinearImplicitModel {
  using State = StateVec<1>;
  static constexpr int n_vars = 1;
  static constexpr int n_providers = 0;

  POPS_HD State source(const State& state, const auto&) const {
    State result{};
    result[0] = -state[0];
    return result;
  }

  POPS_HD void source_jacobian(const State&, const auto&, Real (&jacobian)[1][1]) const {
    jacobian[0][0] = Real(-1);
  }
};

template <int Dim>
void check_ranked_implicit_provider() {
  using Model = RankedLinearImplicitModel<Dim>;
  const Box<Dim> box = Box<Dim>::from_extents(uniform_extent<Dim>(2));
  auto state = one_patch_field(box, Model::n_vars);
  state.set_val(Real(2));

  NewtonOptions options{};
  const auto provider_at = [](std::size_t) { return ProviderStorageView<Dim, 0>{}; };
  auto outcome = backward_euler_source(Model{}, provider_at, state, Real(0.25), options);
  ASSERT_TRUE(outcome.report().solved()) << outcome.report().reason;
  const SolveReport accepted = outcome.consume(SolveConsumption::kAccept);
  EXPECT_TRUE(accepted.solved());
  expect_valid_value(state.fab(0), Real(1.6));
}

template <int Dim>
void check_ranked_failure_collective() {
  Index<Dim> lower{};
  Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = -3 - axis;
    upper[axis] = 2 + axis;
  }
  const Box<Dim> box{lower, upper};
  auto statistics = one_patch_field(box, 13);
  statistics.set_val(Real(0));

  Index<Dim> selected = upper;
  selected[Dim - 1] = lower[Dim - 1];
  Index<Dim> competitor = lower;
  competitor[Dim - 1] = upper[Dim - 1];

  auto host = statistics.fab(0).create_host_mirror();
  statistics.fab(0).copy_to_host(host);
  host(host_offset(statistics.fab(0).grown_box(), selected, 12)) = Real(7);
  host(host_offset(statistics.fab(0).grown_box(), selected, 8)) = Real(4);
  host(host_offset(statistics.fab(0).grown_box(), competitor, 12)) = Real(7);
  host(host_offset(statistics.fab(0).grown_box(), competitor, 8)) = Real(1);
  statistics.fab(0).copy_from_host(host);

  const auto location = collective_first_local_nonlinear_failure(statistics, 7, 12, 8);
  ASSERT_TRUE(location.found);
  EXPECT_EQ(location.priority, 7);
  EXPECT_EQ(location.component, 4);
  for (int axis = 0; axis < Dim; ++axis)
    EXPECT_EQ(location.index[axis], selected[axis]);

  const SolveFailureLocation reported =
      SolveFailureLocation::from<Dim>(location.index, location.component);
  const SolveReport solve = local_nonlinear_solve_report(
      local_nonlinear_status_code(LocalNonlinearStatus::kInvalidEvaluation), 2, 3, Real(1),
      Real(0.5), Real(0.25), Real(1), 0, reported, SolveAction::kFailRun);
  ASSERT_TRUE(solve.valid());
  ASSERT_TRUE(solve.failure.found);
  EXPECT_EQ(solve.failure.rank, Dim);
  EXPECT_EQ(solve.failure.component, 4);
  for (int axis = 0; axis < SolveFailureLocation::maximum_rank; ++axis)
    EXPECT_EQ(solve.failure.index[static_cast<std::size_t>(axis)], axis < Dim ? selected[axis] : 0);
}

}  // namespace

TEST(test_implicit_source_nd, provider_instantiates_one_algorithm_in_1d_2d_3d) {
  check_ranked_implicit_provider<1>();
  check_ranked_implicit_provider<2>();
  check_ranked_implicit_provider<3>();
}

TEST(test_implicit_source_nd, collective_selects_exact_signed_indices_in_1d_2d_3d) {
  check_ranked_failure_collective<1>();
  check_ranked_failure_collective<2>();
  check_ranked_failure_collective<3>();
}
