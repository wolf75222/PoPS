#include <gtest/gtest.h>

#include <pops/mesh/layout/refinement.hpp>

#include "nd_multifab_test_utils.hpp"

#include <array>
#include <stdexcept>
#include <vector>

using namespace pops;
using namespace pops::mesh;
using namespace pops::test::nd;

namespace {

template <int Dim>
void expect_exact_local_copy() {
  const Box<Dim> domain = cube<Dim>(-2, 1);
  const BoxArray<Dim> source_layout(std::vector<Box<Dim>>{domain});
  Extent<Dim> max_grid_size{};
  for (int axis = 0; axis < Dim; ++axis)
    max_grid_size[axis] = axis == 0 ? 2 : 4;
  const BoxArray<Dim> destination_layout = BoxArray<Dim>::from_domain(domain, max_grid_size);
  const auto ranks = one_rank_space<Dim>();
  const auto source_distribution = Distribution<Dim>::replicated(source_layout, ranks);
  const auto destination_distribution = Distribution<Dim>::replicated(destination_layout, ranks);
  HostMultiFab<Dim> source(source_layout, source_distribution, Index<Dim>{}, 2,
                           uniform_extent<Dim>(1));
  HostMultiFab<Dim> destination(destination_layout, destination_distribution, Index<Dim>{}, 2,
                                uniform_extent<Dim>(1));
  fill_valid_encoded(source, Real{-8});
  destination.set_val(Real{-17});

  const CopyScheduleBudget budget{
      source_layout.size() * destination_layout.size(), destination_layout.size(),
      destination_layout.size() * (destination_layout.size() - 1) / 2, 0};
  const auto schedule = prepare_copy_schedule(destination, source, budget);
  ASSERT_FALSE(schedule.has_remote_jobs());
  ASSERT_EQ(schedule.local_jobs().size(), destination_layout.size());
  parallel_copy(destination, source, schedule);

  for (const std::size_t global : destination.local_global_indices()) {
    const Box<Dim>& box = destination.layout()[global];
    for (std::size_t cell = 0; cell < static_cast<std::size_t>(box.numPts()); ++cell) {
      const Index<Dim> index = index_from_ordinal(box, cell);
      for (int component = 0; component < destination.ncomp(); ++component)
        EXPECT_DOUBLE_EQ(value_at(destination, global, index, component),
                         encoded_value(index, component));
    }
  }
}

}  // namespace

TEST(test_copy_schedule_cache, exact_redistribution_is_dimension_generic) {
  expect_exact_local_copy<1>();
  expect_exact_local_copy<2>();
  expect_exact_local_copy<3>();
}

TEST(test_copy_schedule_cache, remote_copy_refuses_before_destination_mutation) {
  const Box<1> domain{Index<1>{0}, Index<1>{3}};
  const BoxArray<1> layout = BoxArray<1>::from_domain(domain, Extent<1>{2});
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{2}};
  const auto source_distribution =
      Distribution<1>::partitioned(layout, ranks, std::vector<Index<1>>{Index<1>{0}, Index<1>{1}});
  const auto destination_distribution =
      Distribution<1>::partitioned(layout, ranks, std::vector<Index<1>>{Index<1>{1}, Index<1>{0}});
  HostMultiFab<1> source(layout, source_distribution, Index<1>{0}, 1, Extent<1>{});
  HostMultiFab<1> destination(layout, destination_distribution, Index<1>{0}, 1, Extent<1>{});
  source.set_val(Real{12});
  destination.set_val(Real{-33});
  const auto before = snapshot(destination);

  const CopyScheduleBudget budget{4, 2, 1, 1};
  const auto schedule = prepare_copy_schedule(destination, source, budget);
  ASSERT_TRUE(schedule.has_remote_jobs());
  EXPECT_THROW(parallel_copy(destination, source, schedule), std::logic_error);
  EXPECT_EQ(snapshot(destination), before);
}

TEST(test_copy_schedule_cache, mixed_replicated_and_partitioned_ownership_is_rejected) {
  const BoxArray<1> layout(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{1}}});
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{1}};
  const auto replicated = Distribution<1>::replicated(layout, ranks);
  const auto partitioned =
      Distribution<1>::partitioned(layout, ranks, std::vector<Index<1>>{Index<1>{0}});
  HostMultiFab<1> source(layout, replicated, Index<1>{0}, 1, Extent<1>{});
  HostMultiFab<1> destination(layout, partitioned, Index<1>{0}, 1, Extent<1>{});
  EXPECT_THROW((void)prepare_copy_schedule(destination, source, CopyScheduleBudget{1, 1, 0, 0}),
               std::invalid_argument);
}
