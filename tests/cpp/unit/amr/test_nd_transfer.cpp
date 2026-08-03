#include <gtest/gtest.h>

#include <pops/amr/transfer/nd/transfer_provider.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

using pops::Box;
using pops::FieldView;
using pops::Index;
using pops::Real;
using pops::amr::transfer::nd::Centering;
using pops::amr::transfer::nd::ComponentRange;
using pops::amr::transfer::nd::IndexMapping;
using pops::amr::transfer::nd::PreparedTransfer;
using pops::amr::transfer::nd::RefinementRatio;
using pops::amr::transfer::nd::TransferKind;
using pops::amr::transfer::nd::TransferProvider;

template <int Dim, class F>
void visit(const Box<Dim>& box, F&& function) {
  if (box.empty())
    return;
  Index<Dim> index = box.lo;
  while (true) {
    function(index);
    int axis = 0;
    for (; axis < Dim; ++axis) {
      if (index[axis] < box.hi[axis]) {
        ++index[axis];
        break;
      }
      index[axis] = box.lo[axis];
    }
    if (axis == Dim)
      return;
  }
}

template <int Dim>
class HostField {
 public:
  HostField(Box<Dim> box, int components)
      : box_(box),
        components_(components),
        values_(static_cast<std::size_t>(box.numPts()) * static_cast<std::size_t>(components)) {}

  FieldView<Real, Dim> view() {
    FieldView<Real, Dim> result{};
    populate(result);
    return result;
  }

  FieldView<const Real, Dim> const_view() const {
    FieldView<const Real, Dim> result{};
    populate(result);
    return result;
  }

  Real& operator()(const Index<Dim>& index, int component = 0) { return view()(index, component); }

  Real operator()(const Index<Dim>& index, int component = 0) const {
    return const_view()(index, component);
  }

  const Box<Dim>& box() const { return box_; }

 private:
  template <class T>
  void populate(FieldView<T, Dim>& result) const {
    result.data = values_.data();
    result.origin = box_.lo;
    result.extents = box_.extent();
    result.strides[0] = 1;
    for (int axis = 1; axis < Dim; ++axis)
      result.strides[axis] = result.strides[axis - 1] * result.extents[axis - 1];
    result.ncomp = components_;
    result.component_stride = box_.numPts();
  }

  Box<Dim> box_;
  int components_;
  mutable std::vector<Real> values_;
};

template <int Dim>
RefinementRatio<Dim> sample_ratio() {
  if constexpr (Dim == 1)
    return RefinementRatio<1>{3};
  else if constexpr (Dim == 2)
    return RefinementRatio<2>{2, 3};
  else
    return RefinementRatio<3>{2, 1, 3};
}

template <int Dim>
IndexMapping<Dim> sample_mapping() {
  if constexpr (Dim == 1)
    return {Index<1>{-3}, Index<1>{5}};
  else if constexpr (Dim == 2)
    return {Index<2>{-3, 4}, Index<2>{5, -7}};
  else
    return {Index<3>{-3, 4, -2}, Index<3>{5, -7, 11}};
}

template <int Dim>
Box<Dim> sample_coarse_region(const IndexMapping<Dim>& mapping) {
  Index<Dim> upper = mapping.coarse_origin;
  for (int axis = 0; axis < Dim; ++axis)
    ++upper[axis];
  return {mapping.coarse_origin, upper};
}

template <int Dim>
Box<Dim> sample_coarse_source(const IndexMapping<Dim>& mapping) {
  Index<Dim> lower = mapping.coarse_origin;
  Index<Dim> upper = mapping.coarse_origin;
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] -= 3;
    upper[axis] += 3;
  }
  return {lower, upper};
}

template <int Dim>
Box<Dim> refine_for_test(const Box<Dim>& coarse, const RefinementRatio<Dim>& ratio,
                         const IndexMapping<Dim>& mapping) {
  Box<Dim> fine{};
  for (int axis = 0; axis < Dim; ++axis) {
    fine.lo[axis] =
        mapping.fine_origin[axis] + (coarse.lo[axis] - mapping.coarse_origin[axis]) * ratio[axis];
    fine.hi[axis] = mapping.fine_origin[axis] +
                    (coarse.hi[axis] - mapping.coarse_origin[axis]) * ratio[axis] + ratio[axis] - 1;
  }
  return fine;
}

template <int Dim>
Real affine_coarse(const Index<Dim>& index, const IndexMapping<Dim>& mapping, int component) {
  Real value = Real(2.75) + Real(4.5) * component;
  for (int axis = 0; axis < Dim; ++axis)
    value += Real(axis + 1) * Real(index[axis] - mapping.coarse_origin[axis]);
  return value;
}

template <int Dim>
Real affine_fine(const Index<Dim>& index, const RefinementRatio<Dim>& ratio,
                 const IndexMapping<Dim>& mapping, int component) {
  Real value = Real(2.75) + Real(4.5) * component;
  for (int axis = 0; axis < Dim; ++axis) {
    const Real relative = static_cast<Real>(index[axis] - mapping.fine_origin[axis]);
    value += Real(axis + 1) * ((relative + Real(0.5)) / static_cast<Real>(ratio[axis]) - Real(0.5));
  }
  return value;
}

template <int Dim>
void fill_affine(HostField<Dim>& field, const IndexMapping<Dim>& mapping) {
  visit(field.box(), [&](const Index<Dim>& index) {
    for (int component = 0; component < 2; ++component)
      field(index, component) = affine_coarse(index, mapping, component);
  });
}

template <int Dim>
void execute(const PreparedTransfer<Dim>& prepared) {
  visit(prepared.destination_region(), [&](const Index<Dim>& index) { prepared(index); });
}

template <int Dim>
void expect_constant_restriction() {
  const auto ratio = sample_ratio<Dim>();
  const auto mapping = sample_mapping<Dim>();
  const Box<Dim> coarse_region = sample_coarse_region(mapping);
  const Box<Dim> fine_region = refine_for_test(coarse_region, ratio, mapping);
  HostField<Dim> fine(fine_region, 2);
  HostField<Dim> coarse(coarse_region, 2);
  visit(fine_region, [&](const Index<Dim>& index) {
    fine(index, 0) = Real(0.1);
    fine(index, 1) = Real(-3.25);
  });

  const auto prepared = TransferProvider<Dim, Centering::Cell>::conservative_restriction().prepare(
      fine.const_view(), coarse.view(), coarse_region, ratio, mapping, ComponentRange{0, 0, 2});
  execute(prepared);

  visit(coarse_region, [&](const Index<Dim>& index) {
    EXPECT_DOUBLE_EQ(coarse(index, 0), Real(0.1));
    EXPECT_DOUBLE_EQ(coarse(index, 1), Real(-3.25));
  });
}

template <int Dim>
void expect_affine_prolongation_and_conservative_round_trip() {
  const auto ratio = sample_ratio<Dim>();
  const auto mapping = sample_mapping<Dim>();
  const Box<Dim> coarse_region = sample_coarse_region(mapping);
  const Box<Dim> fine_region = refine_for_test(coarse_region, ratio, mapping);
  HostField<Dim> coarse_source(sample_coarse_source(mapping), 2);
  HostField<Dim> fine(fine_region, 2);
  HostField<Dim> restricted(coarse_region, 2);
  fill_affine(coarse_source, mapping);

  const auto prolongation = TransferProvider<Dim, Centering::Cell>::linear_prolongation().prepare(
      coarse_source.const_view(), fine.view(), fine_region, ratio, mapping,
      ComponentRange{0, 0, 2});
  execute(prolongation);
  visit(fine_region, [&](const Index<Dim>& index) {
    for (int component = 0; component < 2; ++component)
      EXPECT_NEAR(fine(index, component), affine_fine(index, ratio, mapping, component), 1e-13);
  });

  const auto restriction =
      TransferProvider<Dim, Centering::Cell>::conservative_restriction().prepare(
          fine.const_view(), restricted.view(), coarse_region, ratio, mapping,
          ComponentRange{0, 0, 2});
  execute(restriction);
  visit(coarse_region, [&](const Index<Dim>& index) {
    for (int component = 0; component < 2; ++component)
      EXPECT_NEAR(restricted(index, component), affine_coarse(index, mapping, component), 1e-13);
  });
}

template <int Dim>
void expect_negative_offset_ghost_interpolation() {
  const auto ratio = sample_ratio<Dim>();
  const auto mapping = sample_mapping<Dim>();
  Index<Dim> lower = mapping.fine_origin;
  Index<Dim> upper = mapping.fine_origin;
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] -= ratio[axis];
    upper[axis] = mapping.fine_origin[axis] - 1;
  }
  const Box<Dim> ghost_region{lower, upper};
  HostField<Dim> coarse(sample_coarse_source(mapping), 2);
  HostField<Dim> fine_ghosts(ghost_region, 2);
  fill_affine(coarse, mapping);

  const auto interpolation =
      TransferProvider<Dim, Centering::Cell>::coarse_fine_ghost_interpolation().prepare(
          coarse.const_view(), fine_ghosts.view(), ghost_region, ratio, mapping,
          ComponentRange{0, 0, 2});
  execute(interpolation);
  visit(ghost_region, [&](const Index<Dim>& index) {
    for (int component = 0; component < 2; ++component)
      EXPECT_NEAR(fine_ghosts(index, component), affine_fine(index, ratio, mapping, component),
                  1e-13);
  });
}

}  // namespace

TEST(test_nd_transfer, anisotropic_ratios_validate_once_and_fail_closed) {
  EXPECT_EQ((RefinementRatio<1>{3}.child_count()), 3);
  EXPECT_EQ((RefinementRatio<2>{2, 3}.child_count()), 6);
  EXPECT_EQ((RefinementRatio<3>{2, 1, 3}.child_count()), 6);
  EXPECT_THROW((void)(RefinementRatio<1>{0}), std::invalid_argument);
  EXPECT_THROW((void)(RefinementRatio<2>{2, -1}), std::invalid_argument);
  EXPECT_THROW((void)(RefinementRatio<3>{1, 1, 1}), std::invalid_argument);
  EXPECT_THROW(
      (void)(RefinementRatio<3>{std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
                                std::numeric_limits<int>::max()}),
      std::overflow_error);
}

TEST(test_nd_transfer, prepared_contract_is_fixed_size_and_reports_exact_capabilities) {
  static_assert(std::is_trivially_copyable_v<PreparedTransfer<1>>);
  static_assert(std::is_trivially_copyable_v<PreparedTransfer<2>>);
  static_assert(std::is_trivially_copyable_v<PreparedTransfer<3>>);
  static_assert(std::is_trivially_copyable_v<TransferProvider<3, Centering::Cell>>);

  EXPECT_EQ((TransferProvider<2, Centering::Cell>::conservative_restriction().capabilities()),
            (pops::amr::transfer::nd::TransferCapabilities{1, 0, true, true}));
  EXPECT_EQ((TransferProvider<2, Centering::Cell>::linear_prolongation().capabilities()),
            (pops::amr::transfer::nd::TransferCapabilities{2, 1, false, true}));
  EXPECT_THROW((void)(TransferProvider<2, Centering::Node>::linear_prolongation().capabilities()),
               std::invalid_argument);
  EXPECT_THROW(
      (void)(TransferProvider<2, Centering::Cell>{static_cast<TransferKind>(255)}.capabilities()),
      std::invalid_argument);
}

TEST(test_nd_transfer, conservative_restriction_preserves_constants_bit_exact_in_1d_2d_3d) {
  expect_constant_restriction<1>();
  expect_constant_restriction<2>();
  expect_constant_restriction<3>();
}

TEST(test_nd_transfer, linear_prolongation_and_restriction_reproduce_affine_fields_in_1d_2d_3d) {
  expect_affine_prolongation_and_conservative_round_trip<1>();
  expect_affine_prolongation_and_conservative_round_trip<2>();
  expect_affine_prolongation_and_conservative_round_trip<3>();
}

TEST(test_nd_transfer, coarse_fine_ghost_interpolation_handles_negative_offsets_in_1d_2d_3d) {
  expect_negative_offset_ghost_interpolation<1>();
  expect_negative_offset_ghost_interpolation<2>();
  expect_negative_offset_ghost_interpolation<3>();
}

TEST(test_nd_transfer, preparation_rejects_missing_stencils_components_aliases_and_regions) {
  const RefinementRatio<2> ratio{2, 3};
  const IndexMapping<2> mapping{};
  const Box<2> fine_region{Index<2>{0, 0}, Index<2>{3, 5}};
  const Box<2> coarse_without_halo{Index<2>{0, 0}, Index<2>{1, 1}};
  HostField<2> coarse(coarse_without_halo, 1);
  HostField<2> fine(fine_region, 1);
  const auto linear = TransferProvider<2, Centering::Cell>::linear_prolongation();

  EXPECT_THROW((void)linear.prepare(coarse.const_view(), fine.view(), fine_region, ratio, mapping),
               std::invalid_argument);

  const Box<2> source_with_halo{Index<2>{-1, -1}, Index<2>{2, 2}};
  HostField<2> valid_source(source_with_halo, 1);
  EXPECT_THROW((void)linear.prepare(valid_source.const_view(), fine.view(), fine_region, ratio,
                                    mapping, ComponentRange{0, 0, 2}),
               std::invalid_argument);
  EXPECT_THROW((void)linear.prepare(valid_source.const_view(), fine.view(),
                                    Box<2>{Index<2>{0, 0}, Index<2>{4, 5}}, ratio, mapping),
               std::invalid_argument);

  HostField<2> overlapping(Box<2>{Index<2>{-1, -1}, Index<2>{5, 5}}, 1);
  EXPECT_THROW((void)linear.prepare(overlapping.const_view(), overlapping.view(), fine_region,
                                    ratio, mapping),
               std::invalid_argument);

  const auto unsupported = TransferProvider<2, Centering::Face0>::linear_prolongation();
  EXPECT_THROW((void)unsupported.prepare(valid_source.const_view(), fine.view(), fine_region, ratio,
                                         mapping),
               std::invalid_argument);
  const TransferProvider<2, Centering::Cell> unknown{static_cast<TransferKind>(255)};
  EXPECT_THROW(
      (void)unknown.prepare(valid_source.const_view(), fine.view(), fine_region, ratio, mapping),
      std::invalid_argument);
}
