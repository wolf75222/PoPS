// Test des roles de variables : adresser une composante par son SENS (index_of(role)) plutot que
// par un indice magique. Verifie Euler / isotherme / ExB.
#include <gtest/gtest.h>

#include <pops/physics/bricks/bricks.hpp>
#include <pops/physics/fluids/euler.hpp>

using R = pops::VariableRole;

namespace {

template <int Dim>
void expect_exact_rank_roles() {
  constexpr std::array momentum_roles{R::MomentumX, R::MomentumY, R::MomentumZ};
  constexpr std::array velocity_roles{R::VelocityX, R::VelocityY, R::VelocityZ};
  const pops::VariableSet conservative = pops::EulerND<Dim>::conservative_vars();
  const pops::VariableSet primitive = pops::EulerND<Dim>::primitive_vars();
  const pops::VariableSet isothermal = pops::IsothermalFluxND<Dim>::conservative_vars();

  EXPECT_EQ(conservative.size, Dim + 2);
  EXPECT_EQ(primitive.size, Dim + 2);
  EXPECT_EQ(isothermal.size, Dim + 1);
  EXPECT_EQ(conservative.index_of(R::Density), 0);
  EXPECT_EQ(conservative.index_of(R::Energy), Dim + 1);
  EXPECT_EQ(primitive.index_of(R::Pressure), Dim + 1);
  EXPECT_EQ(conservative.index_of(R::Pressure), -1);
  for (int axis = 0; axis < 3; ++axis) {
    const int expected = axis < Dim ? axis + 1 : -1;
    EXPECT_EQ(conservative.index_of(momentum_roles[axis]), expected);
    EXPECT_EQ(isothermal.index_of(momentum_roles[axis]), expected);
    EXPECT_EQ(primitive.index_of(velocity_roles[axis]), expected);
  }
}

}  // namespace

TEST(VariableRole, IndexOfResolvesEulerIsothermalAndExBRoles) {
  expect_exact_rank_roles<1>();
  expect_exact_rank_roles<2>();
  expect_exact_rank_roles<3>();

  const pops::VariableSet p = pops::EulerND<3>::primitive_vars();
  const pops::Variable v = p.at(1);
  EXPECT_EQ(v.name, "u") << "Variable::at";
  EXPECT_EQ(v.role, R::VelocityX) << "Variable::at";
  EXPECT_EQ(v.component, 1) << "Variable::at";

  EXPECT_EQ(pops::ExBVelocity::conservative_vars().index_of(R::Density), 0) << "role ExB";
}

TEST(VariableRole, AxialRolesRoundTripThroughStableTextAbi) {
  EXPECT_STREQ(pops::role_name(R::AxialX), "axial_x");
  EXPECT_STREQ(pops::role_name(R::AxialY), "axial_y");
  EXPECT_STREQ(pops::role_name(R::AxialZ), "axial_z");
  EXPECT_EQ(pops::role_from_name("axial_x"), R::AxialX);
  EXPECT_EQ(pops::role_from_name("axial_y"), R::AxialY);
  EXPECT_EQ(pops::role_from_name("axial_z"), R::AxialZ);

  const pops::VariableSet original{
      pops::VariableKind::Conservative,
      {"rho", "bx", "by", "bz"},
      4,
      {R::Density, R::AxialX, R::AxialY, R::AxialZ},
  };
  EXPECT_EQ(pops::roles_csv(original), "density,axial_x,axial_y,axial_z");

  pops::VariableSet restored{
      pops::VariableKind::Conservative,
      original.names,
      original.size,
  };
  pops::parse_roles_into(restored, pops::roles_csv(original));
  EXPECT_TRUE(restored.user_roles.empty());
  EXPECT_EQ(restored.roles, original.roles);
}
