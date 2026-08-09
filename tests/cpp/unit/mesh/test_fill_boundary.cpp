#include <gtest/gtest.h>

#include <pops/mesh/boundary/fill_boundary.hpp>

#include "nd_multifab_test_utils.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace pops;
using namespace pops::mesh;
using namespace pops::test::nd;

namespace {

template <int Dim>
HaloScheduleBudget halo_budget(std::size_t boxes, std::size_t images = 64) {
  return HaloScheduleBudget{{boxes, boxes * (boxes - 1) / 2},
                            boxes * boxes * images,
                            boxes * boxes * images * static_cast<std::size_t>(2 * Dim),
                            images,
                            boxes,
                            1'000'000,
                            1'000'000,
                            1'000'000};
}

template <int Dim>
Extent<Dim> grid_size(int axis_zero, int other) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = axis == 0 ? axis_zero : other;
  return result;
}

template <int Dim>
Index<Dim> periodic_source(Index<Dim> index, const Box<Dim>& domain) {
  for (int axis = 0; axis < Dim; ++axis) {
    while (index[axis] < domain.lo[axis])
      index[axis] += static_cast<int>(domain.length(axis));
    while (index[axis] > domain.hi[axis])
      index[axis] -= static_cast<int>(domain.length(axis));
  }
  return index;
}

template <int Dim>
void expect_periodic_halo() {
  const Box<Dim> domain = cube<Dim>(0, 3);
  const BoxArray<Dim> layout = BoxArray<Dim>::from_domain(domain, grid_size<Dim>(2, 4));
  const auto ranks = one_rank_space<Dim>();
  const auto distribution = Distribution<Dim>::replicated(layout, ranks);
  HostMultiFab<Dim> fields(layout, distribution, Index<Dim>{}, 2, uniform_extent<Dim>(1));
  fill_valid_encoded(fields, Real{-777});

  std::array<bool, Dim> periodic{};
  periodic.fill(true);
  const auto schedule =
      prepare_halo_schedule(fields, domain, BoundaryTopology<Dim>::axis_periodic(periodic),
                            halo_budget<Dim>(layout.size(), 64));
  ASSERT_FALSE(schedule.has_remote_jobs());
  fill_boundary(fields, schedule);

  for (const std::size_t global : fields.local_global_indices()) {
    const auto& fab = fields.fab_global(global);
    const std::size_t cells = static_cast<std::size_t>(fab.grown_box().numPts());
    for (std::size_t cell = 0; cell < cells; ++cell) {
      const Index<Dim> index = index_from_ordinal(fab.grown_box(), cell);
      const Index<Dim> source = fab.box().contains(index) ? index : periodic_source(index, domain);
      for (int component = 0; component < fields.ncomp(); ++component)
        EXPECT_DOUBLE_EQ(value_at(fields, global, index, component),
                         encoded_value(source, component));
    }
  }
}

}  // namespace

TEST(test_fill_boundary, ordinary_periodic_halos_are_exact_in_1d_2d_and_3d) {
  expect_periodic_halo<1>();
  expect_periodic_halo<2>();
  expect_periodic_halo<3>();
}

TEST(test_fill_boundary, remote_schedule_refuses_before_local_mutation) {
  const Box<1> domain{Index<1>{0}, Index<1>{3}};
  const BoxArray<1> layout = BoxArray<1>::from_domain(domain, Extent<1>{2});
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{2}};
  const auto distribution =
      Distribution<1>::partitioned(layout, ranks, std::vector<Index<1>>{Index<1>{0}, Index<1>{1}});
  HostMultiFab<1> fields(layout, distribution, Index<1>{0}, 1, Extent<1>{1});
  fields.set_val(Real{-19});
  const auto before = snapshot(fields);

  const auto schedule = prepare_halo_schedule(fields, domain, BoundaryTopology<1>::physical(),
                                              halo_budget<1>(layout.size(), 1));
  ASSERT_TRUE(schedule.has_remote_jobs());
  EXPECT_THROW(fill_boundary(fields, schedule), std::logic_error);
  EXPECT_EQ(snapshot(fields), before);
}

TEST(test_fill_boundary, stale_field_identity_is_rejected) {
  const Box<2> domain = cube<2>(0, 1);
  const BoxArray<2> layout(std::vector<Box<2>>{domain});
  const auto ranks = one_rank_space<2>();
  const auto distribution = Distribution<2>::replicated(layout, ranks);
  HostMultiFab<2> one_ghost(layout, distribution, Index<2>{}, 1, Extent<2>{1, 1});
  HostMultiFab<2> two_ghosts(layout, distribution, Index<2>{}, 1, Extent<2>{2, 2});
  HostMultiFab<2> two_components(layout, distribution, Index<2>{}, 2, Extent<2>{1, 1});
  const auto schedule = prepare_halo_schedule(one_ghost, domain, BoundaryTopology<2>::physical(),
                                              halo_budget<2>(1, 1));
  EXPECT_THROW(fill_boundary(two_ghosts, schedule), std::invalid_argument);
  EXPECT_THROW(fill_boundary(two_components, schedule), std::invalid_argument);
}

TEST(test_fill_boundary, remote_payloads_have_exact_component_major_prefixes) {
  const Box<1> domain{Index<1>{0}, Index<1>{3}};
  const BoxArray<1> layout = BoxArray<1>::from_domain(domain, Extent<1>{2});
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{2}};
  const auto distribution =
      Distribution<1>::partitioned(layout, ranks, std::vector<Index<1>>{Index<1>{0}, Index<1>{1}});
  HostMultiFab<1> fields(layout, distribution, Index<1>{0}, 2, Extent<1>{1});
  const auto schedule = prepare_halo_schedule(fields, domain, BoundaryTopology<1>::physical(),
                                              halo_budget<1>(layout.size(), 1));

  ASSERT_EQ(schedule.send_plans().size(), 1U);
  ASSERT_EQ(schedule.receive_plans().size(), 1U);
  ASSERT_EQ(schedule.send_plans().front().jobs.size(), 1U);
  ASSERT_EQ(schedule.receive_plans().front().jobs.size(), 1U);
  EXPECT_EQ(schedule.ncomp(), 2);
  EXPECT_EQ(schedule.send_plans().front().jobs.front().offset, 0U);
  EXPECT_EQ(schedule.receive_plans().front().jobs.front().offset, 0U);
  EXPECT_EQ(schedule.send_plans().front().jobs.front().elements, 2U);
  EXPECT_EQ(schedule.receive_plans().front().jobs.front().elements, 2U);
  EXPECT_EQ(schedule.send_plans().front().elements, 2U);
  EXPECT_EQ(schedule.receive_plans().front().elements, 2U);
  EXPECT_EQ(schedule.send_elements(), 2U);
  EXPECT_EQ(schedule.receive_elements(), 2U);
}
