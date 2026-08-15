#include <gtest/gtest.h>

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>

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
pops::Index<Dim> index_from_ordinal(const pops::Box<Dim>& box, std::size_t ordinal) {
  pops::Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t extent = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(ordinal % extent);
    ordinal /= extent;
  }
  return index;
}

template <int Dim>
std::size_t field_offset(const pops::Box<Dim>& grown, const pops::Index<Dim>& index,
                         int component = 0) {
  std::size_t cell = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    cell += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(grown.length(axis));
  }
  return static_cast<std::size_t>(component) * stride + cell;
}

template <int Dim>
POPS_HD pops::Real encoded_value(const pops::Index<Dim>& index) {
  pops::Real result = 0;
  pops::Real scale = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result += scale * static_cast<pops::Real>(index[axis]);
    scale *= pops::Real{97};
  }
  return result;
}

template <int Dim>
struct EncodeCells {
  pops::FieldView<pops::Real, Dim> values{};

  POPS_HD void operator()(const pops::Index<Dim>& index) const {
    values(index) = encoded_value(index);
  }
};

template <int Dim>
void prove_exact_rank_residence() {
  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = -axis;
    upper[axis] = 3 + axis;
  }
  const pops::Box<Dim> domain{lower, upper};
  const auto layout = pops::mesh::BoxArray<Dim>::from_domain(domain, filled_extent<Dim>(2));
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{}, filled_extent<Dim>(1)};
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(layout, ranks);
  pops::MultiFab<Dim> fields(layout, distribution, pops::Index<Dim>{}, 1, filled_extent<Dim>(1));

  fields.set_val(pops::Real{3});
  const pops::Real initial = pops::reduce_sum_local(fields);
  EXPECT_EQ(initial, pops::Real{3} * static_cast<pops::Real>(domain.numPts()));

  pops::sync_host();
  pops::sync_device();
  pops::sync_host();
  EXPECT_EQ(pops::reduce_sum_local(fields), initial);

  pops::sync_device();
  for (std::size_t local = 0; local < fields.local_size(); ++local)
    pops::for_each_cell(fields.box(local), EncodeCells<Dim>{fields.fab(local).view()});
  pops::sync_host();

  pops::Real expected_sum = 0;
  std::vector<pops::Real> first_snapshot;
  for (std::size_t local = 0; local < fields.local_size(); ++local) {
    const auto& fab = fields.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const auto& valid = fab.box();
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(valid.numPts()); ++ordinal) {
      const auto index = index_from_ordinal(valid, ordinal);
      const pops::Real expected = encoded_value(index);
      const pops::Real actual = host(field_offset(fab.grown_box(), index));
      EXPECT_EQ(actual, expected);
      expected_sum += expected;
      first_snapshot.push_back(actual);
    }
  }
  EXPECT_EQ(pops::reduce_sum_local(fields), expected_sum);

  pops::sync_host();
  std::vector<pops::Real> second_snapshot;
  for (std::size_t local = 0; local < fields.local_size(); ++local) {
    const auto& fab = fields.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const auto& valid = fab.box();
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(valid.numPts()); ++ordinal) {
      const auto index = index_from_ordinal(valid, ordinal);
      second_snapshot.push_back(host(field_offset(fab.grown_box(), index)));
    }
  }
  EXPECT_EQ(second_snapshot, first_snapshot);
}

}  // namespace

TEST(test_sync_residence, host_device_intent_and_explicit_mirrors_are_exact_in_1d_2d_3d) {
  pops::sync_host();
  pops::sync_host();
  pops::sync_device();
  pops::sync_device();

  prove_exact_rank_residence<1>();
  prove_exact_rank_residence<2>();
  prove_exact_rank_residence<3>();
}
