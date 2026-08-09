#include <gtest/gtest.h>

#include <pops/mesh/execution/for_each.hpp>
#include <pops/runtime/named_field_publication.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

template <int Dim>
pops::Extent<Dim> filled_extent(std::int64_t value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::Index<Dim> filled_index(int value) {
  pops::Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::Real host_value(const pops::Fab<Dim>& fab, const typename pops::Fab<Dim>::HostMirror& host,
                      const pops::Index<Dim>& index, int component) {
  const auto metadata = fab.view();
  std::int64_t offset = static_cast<std::int64_t>(component) * metadata.component_stride;
  for (int axis = 0; axis < Dim; ++axis)
    offset +=
        (static_cast<std::int64_t>(index[axis]) - metadata.origin[axis]) * metadata.strides[axis];
  return host(static_cast<std::size_t>(offset));
}

template <int Dim>
void expect_ranked_publication() {
  const pops::Box<Dim> domain{filled_index<Dim>(0), filled_index<Dim>(2)};
  const auto layout = pops::mesh::BoxArray<Dim>::from_domain(domain, filled_extent<Dim>(3));
  const pops::mesh::RankSpace<Dim> ranks(filled_index<Dim>(0), filled_extent<Dim>(1));
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(layout, ranks);
  pops::MultiFab<Dim> potential(layout, distribution, filled_index<Dim>(0), 1,
                                filled_extent<Dim>(1));
  pops::MultiFab<Dim> auxiliary(layout, distribution, filled_index<Dim>(0), Dim + 1,
                                filled_extent<Dim>(0));

  const auto potential_view = potential.fab(0).view();
  pops::for_each_cell(
      potential.fab(0).grown_box(), KOKKOS_LAMBDA(const pops::Index<Dim>& index) {
        pops::Real value = 0;
        for (int axis = 0; axis < Dim; ++axis)
          value += static_cast<pops::Real>(axis + 1) * static_cast<pops::Real>(index[axis]);
        potential_view(index, 0) = value;
      });

  pops::RealVector<Dim> lower{};
  pops::RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = 3.0;
  const auto geometry = pops::Geometry<Dim>::from_bounds(domain, lower, upper);
  pops::runtime::field::publish_named_field(
      potential, auxiliary, geometry,
      pops::runtime::field::NamedFieldOutput<Dim>(Dim + 1, /*gradient_sign=*/-1));

  const auto& fab = auxiliary.fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  const pops::Index<Dim> center = filled_index<Dim>(1);
  EXPECT_DOUBLE_EQ(host_value(fab, host, center, 0), static_cast<double>(Dim * (Dim + 1) / 2));
  for (int axis = 0; axis < Dim; ++axis)
    EXPECT_DOUBLE_EQ(host_value(fab, host, center, axis + 1), -static_cast<double>(axis + 1));

  const pops::Box<Dim> smaller_domain{filled_index<Dim>(0), filled_index<Dim>(1)};
  pops::RealVector<Dim> smaller_upper{};
  for (int axis = 0; axis < Dim; ++axis)
    smaller_upper[axis] = 2.0;
  const auto wrong_geometry =
      pops::Geometry<Dim>::from_bounds(smaller_domain, lower, smaller_upper);
  EXPECT_THROW(pops::runtime::field::publish_named_field(
                   potential, auxiliary, wrong_geometry,
                   pops::runtime::field::NamedFieldOutput<Dim>(Dim + 1,
                                                               /*gradient_sign=*/-1)),
               std::invalid_argument);
}

}  // namespace

TEST(NamedFieldOutput, ValidatesOneOrExactlyDimPlusOneCompactComponents) {
  EXPECT_NO_THROW((void)pops::runtime::field::NamedFieldOutput<1>(2, 1));
  EXPECT_THROW((void)pops::runtime::field::NamedFieldOutput<1>(3, 1), std::invalid_argument);
  EXPECT_NO_THROW((void)pops::runtime::field::NamedFieldOutput<2>(3, -1));
  EXPECT_THROW((void)pops::runtime::field::NamedFieldOutput<2>(4, 1), std::invalid_argument);

  using Output = pops::runtime::field::NamedFieldOutput<3>;
  const Output potential_only(1, 1);
  EXPECT_FALSE(potential_only.has_gradients());
  EXPECT_EQ(potential_only.component_count(), 1U);
  EXPECT_EQ(potential_only.potential_component(), 0);

  const Output complete(4, -1);
  EXPECT_TRUE(complete.has_gradients());
  EXPECT_EQ(complete.component_count(), 4U);
  EXPECT_EQ(complete.gradient_component(2), 3);

  EXPECT_THROW((void)Output(2, 1), std::invalid_argument);
  EXPECT_THROW((void)Output(1, -1), std::invalid_argument);
  EXPECT_NO_THROW(complete.validate_width(4, "test"));
  EXPECT_THROW(complete.validate_width(8, "test"), std::invalid_argument);
}

TEST(NamedFieldOutput, PublishesPotentialAndEveryAxisInOneRankedAlgorithm) {
  expect_ranked_publication<1>();
  expect_ranked_publication<2>();
  expect_ranked_publication<3>();
}
