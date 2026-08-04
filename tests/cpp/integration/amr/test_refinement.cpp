#include <gtest/gtest.h>

#include <pops/mesh/layout/refinement.hpp>

#include "../../unit/mesh/nd_multifab_test_utils.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace pops;
using namespace pops::mesh;
using namespace pops::test::nd;

namespace {

constexpr CopyScheduleBudget kCopyBudget{4096, 4096, 4096, 4096};

template <int Dim>
Real expected_average(const Index<Dim>& coarse_index, const Extent<Dim>& ratio, int component) {
  std::int64_t children = 1;
  for (int axis = 0; axis < Dim; ++axis)
    children *= ratio[axis];
  Real sum = 0;
  for (std::int64_t ordinal = 0; ordinal < children; ++ordinal) {
    std::int64_t remainder = ordinal;
    Index<Dim> fine_index{};
    for (int axis = 0; axis < Dim; ++axis) {
      fine_index[axis] =
          static_cast<int>(coarse_index[axis] * ratio[axis] + remainder % ratio[axis]);
      remainder /= ratio[axis];
    }
    sum += encoded_value(fine_index, component);
  }
  return sum / static_cast<Real>(children);
}

template <int Dim>
void expect_transfer_round_trip() {
  const Box<Dim> coarse_domain = cube<Dim>(0, 1);
  const Extent<Dim> ratio = uniform_extent<Dim>(2);
  const Box<Dim> fine_domain = refine(coarse_domain, ratio);
  const BoxArray<Dim> coarse_layout(std::vector<Box<Dim>>{coarse_domain});
  const BoxArray<Dim> fine_layout = BoxArray<Dim>::from_domain(fine_domain, axis_sizes<Dim>(2, 2));
  const auto ranks = one_rank_space<Dim>();
  const auto coarse_distribution = Distribution<Dim>::replicated(coarse_layout, ranks);
  const auto fine_distribution = Distribution<Dim>::replicated(fine_layout, ranks);
  HostMultiFab<Dim> fine(fine_layout, fine_distribution, Index<Dim>{}, 2, Extent<Dim>{});
  HostMultiFab<Dim> coarse(coarse_layout, coarse_distribution, Index<Dim>{}, 2, Extent<Dim>{});
  fill_valid_encoded(fine, Real{-1});
  coarse.set_val(Real{-4});

  average_down(fine, coarse, ratio, kCopyBudget);
  for (std::size_t cell = 0; cell < static_cast<std::size_t>(coarse_domain.numPts()); ++cell) {
    const Index<Dim> index = index_from_ordinal(coarse_domain, cell);
    for (int component = 0; component < coarse.ncomp(); ++component)
      EXPECT_DOUBLE_EQ(value_at(coarse, 0, index, component),
                       expected_average(index, ratio, component));
  }

  fine.set_val(Real{-9});
  interpolate(coarse, fine, ratio, kCopyBudget);
  for (const std::size_t global : fine.local_global_indices()) {
    const Box<Dim>& box = fine.layout()[global];
    for (std::size_t cell = 0; cell < static_cast<std::size_t>(box.numPts()); ++cell) {
      const Index<Dim> fine_index = index_from_ordinal(box, cell);
      Index<Dim> coarse_index{};
      for (int axis = 0; axis < Dim; ++axis)
        coarse_index[axis] = fine_index[axis] / 2;
      for (int component = 0; component < fine.ncomp(); ++component)
        EXPECT_DOUBLE_EQ(value_at(fine, global, fine_index, component),
                         expected_average(coarse_index, ratio, component));
    }
  }
}

}  // namespace

TEST(test_refinement, restriction_and_prolongation_are_exact_in_1d_2d_and_3d) {
  expect_transfer_round_trip<1>();
  expect_transfer_round_trip<2>();
  expect_transfer_round_trip<3>();
}

TEST(test_refinement, anisotropic_geometry_uses_every_axis) {
  const Box<3> coarse{Index<3>{-2, -1, 3}, Index<3>{1, 2, 4}};
  const Extent<3> ratio{2, 3, 4};
  const Box<3> fine = refine(coarse, ratio);
  EXPECT_EQ(fine, (Box<3>{Index<3>{-4, -3, 12}, Index<3>{3, 8, 19}}));
  EXPECT_EQ(coarsen(fine, ratio), coarse);
}

TEST(test_refinement, remote_restriction_refuses_before_coarse_mutation) {
  const Box<1> fine_domain{Index<1>{0}, Index<1>{7}};
  const BoxArray<1> fine_layout = BoxArray<1>::from_domain(fine_domain, std::array<int, 1>{4});
  const BoxArray<1> coarse_layout(std::vector<Box<1>>{Box<1>{Index<1>{0}, Index<1>{3}}});
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{2}};
  const auto fine_distribution = Distribution<1>::partitioned(
      fine_layout, ranks, std::vector<Index<1>>{Index<1>{0}, Index<1>{1}});
  const auto coarse_distribution =
      Distribution<1>::partitioned(coarse_layout, ranks, std::vector<Index<1>>{Index<1>{0}});
  HostMultiFab<1> fine(fine_layout, fine_distribution, Index<1>{0}, 1, Extent<1>{});
  HostMultiFab<1> coarse(coarse_layout, coarse_distribution, Index<1>{0}, 1, Extent<1>{});
  fine.set_val(Real{8});
  coarse.set_val(Real{-23});
  const auto before = snapshot(coarse);
  EXPECT_THROW(average_down(fine, coarse, 2, CopyScheduleBudget{2, 2, 0, 1}), std::logic_error);
  EXPECT_EQ(snapshot(coarse), before);
}
