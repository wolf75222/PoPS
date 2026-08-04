#include <gtest/gtest.h>

#include <pops/mesh/boundary/fill_boundary.hpp>

#include "nd_multifab_test_utils.hpp"

#include <array>
#include <stdexcept>
#include <vector>

using namespace pops;
using namespace pops::mesh;
using namespace pops::test::nd;

TEST(test_fill_boundary_cache, prepared_schedule_is_deterministic_and_reusable) {
  const Box<3> domain = cube<3>(-2, 1);
  const BoxArray<3> layout = BoxArray<3>::from_domain(domain, axis_sizes<3>(2, 4));
  const auto distribution = Distribution<3>::replicated(layout, one_rank_space<3>());
  HostMultiFab<3> fields(layout, distribution, Index<3>{}, 1, Extent<3>{1, 1, 1});
  std::array<bool, 3> periodic{true, false, true};
  const HaloScheduleBudget budget{{layout.size(), 1}, 1024, 2048, 64};

  const auto first =
      prepare_halo_schedule(fields, domain, BoundaryTopology<3>::axis_periodic(periodic), budget);
  const auto second =
      prepare_halo_schedule(fields, domain, BoundaryTopology<3>::axis_periodic(periodic), budget);
  EXPECT_EQ(first.local_jobs(), second.local_jobs());
  EXPECT_EQ(first.send_plans(), second.send_plans());
  EXPECT_EQ(first.receive_plans(), second.receive_plans());

  fill_valid_encoded(fields, Real{-91});
  fill_boundary(fields, first);
  const auto once = snapshot(fields);
  fill_boundary(fields, first);
  EXPECT_EQ(snapshot(fields), once);
}

TEST(test_fill_boundary_cache, deep_periodic_wrap_has_no_single_period_limit) {
  const Box<1> domain{Index<1>{4}, Index<1>{5}};
  const BoxArray<1> layout(std::vector<Box<1>>{domain});
  const auto distribution = Distribution<1>::replicated(layout, one_rank_space<1>());
  HostMultiFab<1> fields(layout, distribution, Index<1>{}, 1, Extent<1>{5});
  fill_valid_encoded(fields, Real{-200});
  const auto topology = BoundaryTopology<1>::axis_periodic(std::array<bool, 1>{true});
  const HaloScheduleBudget budget{{1, 0}, 32, 32, 7};
  fill_boundary(fields, domain, topology, budget);

  for (int coordinate = -1; coordinate <= 10; ++coordinate) {
    int wrapped = coordinate;
    while (wrapped < domain.lo[0])
      wrapped += 2;
    while (wrapped > domain.hi[0])
      wrapped -= 2;
    EXPECT_DOUBLE_EQ(value_at(fields, 0, Index<1>{coordinate}), encoded_value(Index<1>{wrapped}));
  }
}

TEST(test_fill_boundary_cache, finite_preparation_budget_is_enforced) {
  const Box<2> domain = cube<2>(0, 3);
  const BoxArray<2> layout = BoxArray<2>::from_domain(domain, std::array<int, 2>{2, 2});
  const auto distribution = Distribution<2>::replicated(layout, one_rank_space<2>());
  HostMultiFab<2> fields(layout, distribution, Index<2>{}, 1, Extent<2>{1, 1});
  EXPECT_THROW((void)prepare_halo_schedule(fields, domain, BoundaryTopology<2>::physical(),
                                           HaloScheduleBudget{{layout.size(), 6}, 0, 64, 1}),
               std::length_error);
}
