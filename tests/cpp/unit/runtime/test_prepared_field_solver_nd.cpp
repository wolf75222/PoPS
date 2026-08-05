#include <pops/runtime/system/exact_field_solver_backend.hpp>

#include <gtest/gtest.h>

template class pops::runtime::field::PreparedFieldSolverComponent<1>;
template class pops::runtime::field::PreparedFieldSolverComponent<2>;
template class pops::runtime::field::PreparedFieldSolverComponent<3>;
template class pops::runtime::system::CartesianCgFieldSolverBackend<1>;
template class pops::runtime::system::CartesianCgFieldSolverBackend<2>;
template class pops::runtime::system::CartesianCgFieldSolverBackend<3>;
template class pops::runtime::system::ComponentFieldSolverBackend<1>;
template class pops::runtime::system::ComponentFieldSolverBackend<2>;
template class pops::runtime::system::ComponentFieldSolverBackend<3>;

static_assert(pops::runtime::field::PreparedFieldSolverComponent<1>::dimension == 1);
static_assert(pops::runtime::field::PreparedFieldSolverComponent<2>::dimension == 2);
static_assert(pops::runtime::field::PreparedFieldSolverComponent<3>::dimension == 3);
static_assert(pops::runtime::system::ExactFieldSolverBackend<1>::dimension == 1);
static_assert(pops::runtime::system::ExactFieldSolverBackend<2>::dimension == 2);
static_assert(pops::runtime::system::ExactFieldSolverBackend<3>::dimension == 3);

TEST(PreparedFieldSolverNd, CarriesOneExactCompileTimeRank) {
  EXPECT_EQ(pops::runtime::field::PreparedFieldSolverComponent<1>::dimension, 1);
  EXPECT_EQ(pops::runtime::field::PreparedFieldSolverComponent<2>::dimension, 2);
  EXPECT_EQ(pops::runtime::field::PreparedFieldSolverComponent<3>::dimension, 3);
}
