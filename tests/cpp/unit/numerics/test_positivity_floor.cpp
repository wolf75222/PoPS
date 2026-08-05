#include <pops/numerics/spatial/primitives/positivity.hpp>

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

namespace {

using pops::FieldView;
using pops::Index;
using pops::Real;
using pops::VariableKind;
using pops::VariableRole;
using pops::VariableSet;

struct DensityModel {
  using State = std::array<Real, 3>;
  static constexpr int n_vars = 3;

  static VariableSet conservative_vars() {
    return {VariableKind::Conservative,
            {"density", "momentum", "energy"},
            n_vars,
            {VariableRole::Density, VariableRole::MomentumX, VariableRole::Energy}};
  }
};

struct ScalarModel {
  using State = std::array<Real, 1>;
  static constexpr int n_vars = 1;

  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"scalar"}, n_vars, {VariableRole::Scalar}};
  }
};

struct OpaqueModel {
  using State = std::array<Real, 1>;
  static constexpr int n_vars = 1;
};

template <int Dim>
void expect_ranked_fallback() {
  std::array<Real, DensityModel::n_vars> cell{Real(2), Real(4), Real(9)};
  Index<Dim> source{};
  FieldView<const Real, Dim> state{};
  state.data = cell.data();
  state.origin = source;
  for (int axis = 0; axis < Dim; ++axis) {
    source[axis] = 3 + axis;
    state.origin[axis] = source[axis];
    state.extents[axis] = 1;
    state.strides[axis] = 1;
  }
  state.ncomp = DensityModel::n_vars;
  state.component_stride = 1;

  DensityModel::State face{Real(-1), Real(7), Real(8)};
  pops::zhang_shu_scale<DensityModel>(face, state, source, Real(1e-8), 0);
  EXPECT_EQ(face, cell);

  DensityModel::State disabled{Real(-1), Real(7), Real(8)};
  const DensityModel::State disabled_before = disabled;
  pops::zhang_shu_scale<DensityModel>(disabled, state, source, Real(0), 0);
  EXPECT_EQ(disabled, disabled_before);

  DensityModel::State admissible{Real(1), Real(7), Real(8)};
  const DensityModel::State admissible_before = admissible;
  pops::zhang_shu_scale<DensityModel>(admissible, state, source, Real(1e-8), 0);
  EXPECT_EQ(admissible, admissible_before);
}

TEST(test_positivity_floor, UsesOneExactRankedAlgorithmInOneTwoAndThreeDimensions) {
  expect_ranked_fallback<1>();
  expect_ranked_fallback<2>();
  expect_ranked_fallback<3>();
}

TEST(test_positivity_floor, ResolvesDensityRoleAndFailsClosedOtherwise) {
  EXPECT_EQ(pops::detail::positivity_comp<DensityModel>(Real(1e-8)), 0);
  EXPECT_EQ(pops::detail::positivity_comp<ScalarModel>(Real(0)), 0);
  EXPECT_THROW((void)pops::detail::positivity_comp<ScalarModel>(Real(1e-8)), std::runtime_error);
  EXPECT_THROW((void)pops::detail::positivity_comp<OpaqueModel>(Real(1e-8)), std::runtime_error);
}

}  // namespace
