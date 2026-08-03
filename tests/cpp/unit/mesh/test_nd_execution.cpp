#include <gtest/gtest.h>

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/fab.hpp>

#include <Kokkos_Core.hpp>

#include <limits>
#include <type_traits>

namespace {

template <int Dim>
pops::Box<Dim> sample_box() {
  if constexpr (Dim == 1) {
    return pops::Box<1>{pops::Index<1>{-2}, pops::Index<1>{2}};
  } else if constexpr (Dim == 2) {
    return pops::Box<2>{pops::Index<2>{-2, 3}, pops::Index<2>{1, 5}};
  } else {
    return pops::Box<3>{pops::Index<3>{-1, 2, 7}, pops::Index<3>{1, 4, 8}};
  }
}

template <int Dim>
struct SetCellValue {
  pops::FieldView<pops::Real, Dim> values;
  pops::Real value;

  POPS_HD void operator()(const pops::CellIndex<Dim>& index) const { values(index) = value; }
};

template <int Dim>
struct ReadCellValue {
  pops::FieldView<const pops::Real, Dim> values;

  POPS_HD pops::Real operator()(const pops::CellIndex<Dim>& index) const {
    return values(index);
  }
};

template <int Dim, int Axis>
struct SetFaceValue {
  pops::FieldView<pops::Real, Dim> values;

  POPS_HD void operator()(const pops::FaceIndex<Dim, Axis>& face) const {
    static_assert(pops::FaceIndex<Dim, Axis>::normal_axis == Axis);
    values(face.coordinate) = static_cast<pops::Real>(Axis + 1);
  }
};

template <int Dim>
void expect_cell_and_product_execution() {
  const pops::Box<Dim> cells = sample_box<Dim>();
  pops::Fab<Dim> field(cells, 1);
  Kokkos::DefaultExecutionSpace execution;

  pops::for_each_cell(execution, cells, SetCellValue<Dim>{field.view(), pops::Real(2)});
  EXPECT_EQ(pops::for_each_cell_reduce_sum(
                execution, cells,
                ReadCellValue<Dim>{static_cast<const pops::Fab<Dim>&>(field).view()}),
            static_cast<pops::Real>(2 * cells.numPts()));

  pops::for_each_product(cells, SetCellValue<Dim>{field.view(), pops::Real(3)});
  EXPECT_EQ(pops::for_each_product_reduce_sum(
                cells, ReadCellValue<Dim>{static_cast<const pops::Fab<Dim>&>(field).view()}),
            static_cast<pops::Real>(3 * cells.numPts()));
}

template <int Dim, int Axis>
void expect_face_execution() {
  const pops::Box<Dim> cells = sample_box<Dim>();
  const pops::Box<Dim> faces = pops::face_box<Axis>(cells);
  pops::Fab<Dim> field(faces, 1);
  Kokkos::DefaultExecutionSpace execution;

  pops::for_each_face<Axis>(execution, cells, SetFaceValue<Dim, Axis>{field.view()});
  const pops::Real total = pops::for_each_cell_reduce_sum(
      execution, faces, ReadCellValue<Dim>{static_cast<const pops::Fab<Dim>&>(field).view()});
  EXPECT_EQ(total, static_cast<pops::Real>((Axis + 1) * faces.numPts()));
  EXPECT_EQ(faces.length(Axis), cells.length(Axis) + 1);
  for (int tangent = 0; tangent < Dim; ++tangent)
    if (tangent != Axis)
      EXPECT_EQ(faces.length(tangent), cells.length(tangent));
}

}  // namespace

TEST(test_nd_execution, cell_and_product_facades_share_static_1d_2d_3d_policies) {
  static_assert(std::is_same_v<pops::CellIndex<2>, pops::Index<2>>);
  static_assert(std::is_trivially_copyable_v<pops::FaceIndex<3, 2>>);

  expect_cell_and_product_execution<1>();
  expect_cell_and_product_execution<2>();
  expect_cell_and_product_execution<3>();
}

TEST(test_nd_execution, face_axis_is_compile_time_and_each_dimension_has_exact_face_extent) {
  expect_face_execution<1, 0>();
  expect_face_execution<2, 0>();
  expect_face_execution<2, 1>();
  expect_face_execution<3, 0>();
  expect_face_execution<3, 1>();
  expect_face_execution<3, 2>();
}

TEST(test_nd_execution, empty_and_non_addressable_face_domains_fail_deterministically) {
  EXPECT_TRUE(pops::face_box<0>(pops::Box<1>{}).empty());
  const pops::Box<1> overflow{pops::Index<1>{0},
                              pops::Index<1>{std::numeric_limits<int>::max()}};
  EXPECT_THROW((void)pops::face_box<0>(overflow), std::overflow_error);
}
