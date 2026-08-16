#include <gtest/gtest.h>

#include <pops/runtime/system/system_coupling_registry.hpp>

#include <string>
#include <utility>
#include <vector>

namespace {

template <int Dim>
using Operator = pops::runtime::system::PreparedCouplingOperator<Dim>;

using Group = pops::runtime::system::PreparedCouplingConservationGroup;

template <int Dim>
Operator<Dim> make_operator(std::vector<Group> groups) {
  return Operator<Dim>([](pops::Real, const std::vector<pops::MultiFab<Dim>*>&) {},
                       std::move(groups));
}

template <int Dim>
void prove_owner_qualified_roles_are_strict_and_permutable() {
  const Group ion_neutral{"mass-exchange",
                          {{"ions", 0, 0, "density"}, {"neutrals", 1, 0, "density"}}};
  const Group neutral_ion{"mass-exchange",
                          {{"neutrals", 1, 0, "density"}, {"ions", 0, 0, "density"}}};
  EXPECT_NO_THROW((void)make_operator<Dim>({ion_neutral}));
  EXPECT_NO_THROW((void)make_operator<Dim>({neutral_ion}))
      << "semantic owner qualification must not depend on Program declaration order";

  Group missing_owner = ion_neutral;
  missing_owner.members[0].owner.clear();
  EXPECT_THROW((void)make_operator<Dim>({missing_owner}), std::invalid_argument);

  Group missing_role = ion_neutral;
  missing_role.members[1].state_role.clear();
  EXPECT_THROW((void)make_operator<Dim>({missing_role}), std::invalid_argument);

  Group duplicate = ion_neutral;
  duplicate.members[1] = duplicate.members[0];
  EXPECT_THROW((void)make_operator<Dim>({duplicate}), std::invalid_argument);

  Group ambiguous_flat_role = ion_neutral;
  ambiguous_flat_role.members[0].owner.clear();
  ambiguous_flat_role.members[1].owner.clear();
  EXPECT_THROW((void)make_operator<Dim>({ambiguous_flat_role}), std::invalid_argument)
      << "a flat density label cannot replace two owner-qualified state roles";
}

}  // namespace

TEST(test_amr_coupled_source_role_strict, StructuredOwnerQualifiedRolesDim1Dim2Dim3) {
  prove_owner_qualified_roles_are_strict_and_permutable<1>();
  prove_owner_qualified_roles_are_strict_and_permutable<2>();
  prove_owner_qualified_roles_are_strict_and_permutable<3>();
}
