#include <pops/runtime/system/prepared_field_solver_component.hpp>

#include <gtest/gtest.h>

template class pops::runtime::field::PreparedFieldSolverComponent<1>;
template class pops::runtime::field::PreparedFieldSolverComponent<2>;
template class pops::runtime::field::PreparedFieldSolverComponent<3>;

static_assert(pops::runtime::field::PreparedFieldSolverComponent<1>::dimension == 1);
static_assert(pops::runtime::field::PreparedFieldSolverComponent<2>::dimension == 2);
static_assert(pops::runtime::field::PreparedFieldSolverComponent<3>::dimension == 3);

TEST(PreparedFieldSolverNd, CarriesOneExactCompileTimeRank) {
  EXPECT_EQ(pops::runtime::field::PreparedFieldSolverComponent<1>::dimension, 1);
  EXPECT_EQ(pops::runtime::field::PreparedFieldSolverComponent<2>::dimension, 2);
  EXPECT_EQ(pops::runtime::field::PreparedFieldSolverComponent<3>::dimension, 3);
}
