// Test des roles de variables : adresser une composante par son SENS (index_of(role)) plutot que
// par un indice magique. Verifie Euler / isotherme / ExB.
#include <gtest/gtest.h>

#include <pops/physics/bricks/bricks.hpp>
#include <pops/physics/fluids/euler.hpp>

using R = pops::VariableRole;

namespace {

template <int Dim>
void expect_exact_rank_roles() {
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
  for (int axis = 0; axis < Dim; ++axis) {
    const int expected = axis + 1;
    EXPECT_EQ(conservative.index_of(R::momentum(axis)), expected);
    EXPECT_EQ(isothermal.index_of(R::momentum(axis)), expected);
    EXPECT_EQ(primitive.index_of(R::velocity(axis)), expected);
  }
}

}  // namespace

TEST(VariableRole, IndexOfResolvesEulerIsothermalAndExBRoles) {
  expect_exact_rank_roles<1>();
  expect_exact_rank_roles<2>();
  expect_exact_rank_roles<3>();

  const pops::VariableSet p = pops::EulerND<3>::primitive_vars();
  const pops::Variable v = p.at(1);
  EXPECT_EQ(v.name, "velocity_0") << "Variable::at";
  EXPECT_EQ(v.role, R::velocity(0)) << "Variable::at";
  EXPECT_EQ(v.component, 1) << "Variable::at";

  EXPECT_EQ(pops::CartesianExBDrift::conservative_vars().index_of(R::Density), 0) << "role ExB";
}

TEST(VariableRole, AxialRolesRoundTripThroughStableTextAbi) {
  EXPECT_EQ(pops::role_name(R::axial(0)), "axial:0");
  EXPECT_EQ(pops::role_name(R::axial(1)), "axial:1");
  EXPECT_EQ(pops::role_name(R::axial(2)), "axial:2");
  EXPECT_EQ(pops::role_from_name("axial:0"), R::axial(0));
  EXPECT_EQ(pops::role_from_name("axial:1"), R::axial(1));
  EXPECT_EQ(pops::role_from_name("axial:2"), R::axial(2));

  const pops::VariableSet original{
      pops::VariableKind::Conservative,
      {"rho", "bx", "by", "bz"},
      4,
      {R::Density, R::axial(0), R::axial(1), R::axial(2)},
  };
  EXPECT_EQ(pops::roles_csv(original), "density,axial:0,axial:1,axial:2");

  pops::VariableSet restored{
      pops::VariableKind::Conservative,
      original.names,
      original.size,
  };
  pops::parse_roles_into(restored, pops::roles_csv(original));
  EXPECT_TRUE(restored.user_roles.empty());
  EXPECT_EQ(restored.roles, original.roles);
}

TEST(VariableRole, AxisSemanticsAreValidatedAgainstTheExactNativeRank) {
  const pops::VariableSet one_dimensional{
      pops::VariableKind::Conservative,
      {"rho", "m0"},
      2,
      {R::Density, R::momentum(0)},
  };
  EXPECT_NO_THROW(
      pops::validate_variable_semantics<1>(one_dimensional, "test", "one-dimensional state"));

  const pops::VariableSet invalid_for_one_dimensional{
      pops::VariableKind::Conservative,
      {"rho", "m1"},
      2,
      {R::Density, R::momentum(1)},
  };
  EXPECT_THROW(pops::validate_variable_semantics<1>(invalid_for_one_dimensional, "test",
                                                    "one-dimensional state"),
               std::invalid_argument);
  EXPECT_NO_THROW(pops::validate_variable_semantics<3>(invalid_for_one_dimensional, "test",
                                                       "three-dimensional state"));
}
