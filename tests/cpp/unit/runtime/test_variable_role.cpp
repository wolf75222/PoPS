// Test des roles de variables : adresser une composante par son SENS (index_of(role)) plutot que
// par un indice magique. Verifie Euler / isotherme / ExB.
#include <gtest/gtest.h>

#include <pops/physics/bricks/bricks.hpp>
#include <pops/physics/fluids/euler.hpp>

using R = pops::VariableRole;

TEST(VariableRole, IndexOfResolvesEulerIsothermalAndExBRoles) {
  const pops::VariableSet c = pops::Euler::conservative_vars();
  EXPECT_EQ(c.index_of(R::Density), 0) << "roles conservatifs Euler";
  EXPECT_EQ(c.index_of(R::MomentumX), 1) << "roles conservatifs Euler";
  EXPECT_EQ(c.index_of(R::MomentumY), 2) << "roles conservatifs Euler";
  EXPECT_EQ(c.index_of(R::Energy), 3) << "roles conservatifs Euler";
  EXPECT_EQ(c.index_of(R::Pressure), -1)
      << "Pressure devrait etre absente des conservatives";  // la pression n'est pas conservative

  const pops::VariableSet p = pops::Euler::primitive_vars();
  EXPECT_EQ(p.index_of(R::Pressure), 3) << "roles primitifs Euler";
  EXPECT_EQ(p.index_of(R::VelocityX), 1) << "roles primitifs Euler";

  const pops::Variable v = p.at(1);
  EXPECT_EQ(v.name, "u") << "Variable::at";
  EXPECT_EQ(v.role, R::VelocityX) << "Variable::at";
  EXPECT_EQ(v.component, 1) << "Variable::at";

  // isotherme (3 var) + ExB (1 var)
  EXPECT_EQ(pops::IsothermalFlux::conservative_vars().index_of(R::MomentumY), 2)
      << "roles isotherme";
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
