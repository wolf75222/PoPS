// Fab2D + Array4 + for_each_cell : remplissage de l'interieur via le dispatch
// (handle Array4 capture par valeur, comme un kernel Kokkos), ghosts intacts,
// coherence du handle avec operator(), et layout composante-lente.

#include <gtest/gtest.h>

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/index/real_vector.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/mesh/execution/for_each.hpp>

#include <limits>
#include <type_traits>

using namespace pops;

static_assert(std::is_trivially_copyable_v<FieldView<Real, 1>> &&
              std::is_trivially_copyable_v<FieldView<Real, 2>> &&
              std::is_trivially_copyable_v<FieldView<Real, 3>>);
static_assert(std::is_standard_layout_v<FieldView<Real, 1>> &&
              std::is_standard_layout_v<FieldView<Real, 2>> &&
              std::is_standard_layout_v<FieldView<Real, 3>>);

namespace {

struct NoOpCellKernel {
  POPS_HD void operator()(int, int) const {}
};

template <int Dim>
struct FillRankedFab {
  FieldView<Real, Dim> values;

  POPS_HD void operator()(const Index<Dim>& index) const {
    Real value = 0;
    for (int axis = 0; axis < Dim; ++axis)
      value += (axis + 1) * index[axis];
    values(index, 0) = value;
    values(index, 1) = -value;
  }
};

template <int Dim>
struct SumRankedIndex {
  POPS_HD Real operator()(const Index<Dim>& index) const {
    Real value = 0;
    for (int axis = 0; axis < Dim; ++axis)
      value += index[axis];
    return value;
  }
};

template <int Dim>
struct NegativeRankedIndex {
  POPS_HD Real operator()(const Index<Dim>& index) const {
    Real value = -1;
    for (int axis = 0; axis < Dim; ++axis)
      value -= Real(index[axis] * index[axis]);
    return value;
  }
};

template <int Dim>
struct NoOpRankedIndex {
  POPS_HD void operator()(const Index<Dim>&) const {}
};

}  // namespace

TEST(test_fab2d, fill_interior_leaves_ghosts_untouched) {
  Box2D valid = Box2D::from_extents(4, 3);  // [0..3] x [0..2]
  Fab2D fab(valid, /*ncomp=*/2, /*ng=*/1);
  EXPECT_TRUE(fab.grown_box().nx() == 6 && fab.grown_box().ny() == 5) << "grown";
  EXPECT_EQ(fab.size(), 6 * 5 * 2) << "alloc_size";
  EXPECT_TRUE(fab(-1, -1, 0) == 0.0 && fab(4, 3, 1) == 0.0) << "zero_init_ghost";

  // remplir l'interieur via le dispatch, handle capture par valeur
  Array4 a = fab.array();
  for_each_cell(valid, [a](int i, int j) {
    a(i, j, 0) = i + 10.0 * j;
    a(i, j, 1) = -(i + 10.0 * j);
  });

  EXPECT_TRUE(fab(0, 0, 0) == 0.0 && fab(3, 2, 0) == 23.0) << "fill_c0";
  EXPECT_EQ(fab(3, 2, 1), -23.0) << "fill_c1";
  EXPECT_TRUE(fab(-1, 0, 0) == 0.0 && fab(4, 2, 0) == 0.0 && fab(0, -1, 0) == 0.0)
      << "ghost_untouched";

  ConstArray4 ca = fab.const_array();
  EXPECT_TRUE(ca(2, 1, 0) == fab(2, 1, 0) && ca(2, 1, 1) == fab(2, 1, 1)) << "array4_matches";

  // composante-lente : le plan c=1 est un bloc contigu apres c=0,
  // de stride nx_tot * ny_tot = 6 * 5 = 30
  EXPECT_EQ(&fab(0, 0, 1) - &fab(0, 0, 0), 30) << "comp_slowest";
}

TEST(test_fab2d, set_val_fills_valid_ghosts_components_and_nonzero_origin) {
  const Box2D valid{{-7, 11}, {-4, 13}};
  Fab2D fab(valid, /*ncomp=*/3, /*ng=*/2);

  fab.set_val(Real(-3.25));

  const Box2D grown = fab.grown_box();
  for (int component = 0; component < fab.ncomp(); ++component)
    for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
      for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
        EXPECT_DOUBLE_EQ(fab(i, j, component), Real(-3.25));
}

TEST(test_fab2d, widened_offsets_support_extreme_negative_origins) {
  constexpr int lo = std::numeric_limits<int>::min();
  Fab2D fab(Box2D{{lo, lo}, {lo + 1, lo + 1}}, /*ncomp=*/1, /*ng=*/0);

  fab(lo + 1, lo + 1) = Real(4.5);
  EXPECT_DOUBLE_EQ(fab.const_array()(lo + 1, lo + 1), Real(4.5));
}

TEST(test_fab2d, rejects_noniterable_bounds_and_oversized_allocation_before_launch) {
  constexpr int lo = std::numeric_limits<int>::min();
  constexpr int hi = std::numeric_limits<int>::max();

  EXPECT_THROW((void)Fab2D(Box2D{{hi, 0}, {hi, 0}}, /*ncomp=*/1, /*ng=*/0), ValidationError);
  EXPECT_THROW((void)Fab2D(Box2D{{lo, 0}, {-1, 0}}, /*ncomp=*/1, /*ng=*/0), ValidationError);
  EXPECT_THROW((void)Fab2D(Box2D{{0, 0}, {hi - 1, hi - 1}}, /*ncomp=*/3, /*ng=*/0),
               ValidationError);

  // The generic iteration seam must make the same decision before Kokkos sees hi + 1.
  EXPECT_THROW(for_each_cell(Box2D{{hi, 0}, {hi, 0}}, NoOpCellKernel{}), std::overflow_error);
}

TEST(test_fab2d, ranked_fab_layout_and_host_mirrors_cover_1d_2d_and_3d) {
  const Box<1> line{Index<1>{-2}, Index<1>{1}};
  Fab<1> fab1(line, /*ncomp=*/2, Extent<1>{2});
  for_each_cell(line, FillRankedFab<1>{fab1.view()});
  auto host1 = fab1.create_host_mirror();
  fab1.copy_to_host(host1);
  EXPECT_EQ(fab1.ghosts(), Extent<1>{2});
  EXPECT_EQ(fab1.size(), 16u);
  EXPECT_EQ(fab1.view().strides[0], 1);
  EXPECT_EQ(fab1.view().component_stride, 8);
  EXPECT_DOUBLE_EQ(host1(2), -2.0);
  EXPECT_DOUBLE_EQ(host1(2 + 8), 2.0);
  EXPECT_DOUBLE_EQ(host1(5), 1.0);
  EXPECT_DOUBLE_EQ(host1(5 + 8), -1.0);

  const Box<2> plane{Index<2>{-1, 3}, Index<2>{1, 4}};
  Fab<2> fab2(plane, /*ncomp=*/2, Extent<2>{1, 2});
  for_each_cell(plane, FillRankedFab<2>{fab2.view()});
  auto host2 = fab2.create_host_mirror();
  fab2.copy_to_host(host2);
  EXPECT_EQ(fab2.ghosts(), (Extent<2>{1, 2}));
  EXPECT_EQ(fab2.size(), 60u);
  EXPECT_EQ(fab2.view().strides[0], 1);
  EXPECT_EQ(fab2.view().strides[1], 5);
  EXPECT_EQ(fab2.view().component_stride, 30);
  EXPECT_DOUBLE_EQ(host2(11), 5.0);  // (-1, 3), offset 1 + 2 * 5
  EXPECT_DOUBLE_EQ(host2(11 + 30), -5.0);

  const Box<3> volume{Index<3>{0, -1, 2}, Index<3>{1, 0, 3}};
  Fab<3> fab3(volume, /*ncomp=*/2, Extent<3>{1, 0, 2});
  for_each_cell(volume, FillRankedFab<3>{fab3.view()});
  auto host3 = fab3.create_host_mirror();
  fab3.copy_to_host(host3);
  EXPECT_EQ(fab3.ghosts(), (Extent<3>{1, 0, 2}));
  EXPECT_EQ(fab3.size(), 96u);
  EXPECT_EQ(fab3.view().strides[0], 1);
  EXPECT_EQ(fab3.view().strides[1], 4);
  EXPECT_EQ(fab3.view().strides[2], 8);
  EXPECT_EQ(fab3.view().component_stride, 48);
  EXPECT_DOUBLE_EQ(host3(22), 7.0);  // (1, 0, 2), offset 2 + 1 * 4 + 2 * 8
  EXPECT_DOUBLE_EQ(host3(22 + 48), -7.0);

  host3(0) = Real(17.5);
  fab3.copy_from_host(host3);
  auto copied_back = fab3.create_host_mirror();
  fab3.copy_to_host(copied_back);
  EXPECT_DOUBLE_EQ(copied_back(0), 17.5);
}

TEST(test_fab2d, ranked_fab_rejects_invalid_axis_ghosts_and_overflow_before_allocation) {
  const Box<1> line{Index<1>{0}, Index<1>{1}};
  EXPECT_THROW((void)Fab<1>(line, /*ncomp=*/1, Extent<1>{-1}), std::invalid_argument);
  EXPECT_THROW((void)Fab<2>(Box<2>{Index<2>{0, 0}, Index<2>{1, 1}}, /*ncomp=*/1, Extent<2>{0, -1}),
               std::invalid_argument);

  constexpr int maximum = std::numeric_limits<int>::max();
  EXPECT_THROW(
      (void)Fab<1>(Box<1>{Index<1>{maximum}, Index<1>{maximum}}, /*ncomp=*/1, Extent<1>{1}),
      std::overflow_error);
  EXPECT_THROW((void)Fab<1>(line, /*ncomp=*/1, Extent<1>{std::numeric_limits<std::int64_t>::max()}),
               std::overflow_error);
  EXPECT_THROW((void)Fab<2>(Box<2>{Index<2>{0, 0}, Index<2>{maximum - 1, maximum - 1}},
                            /*ncomp=*/3, Extent<2>{}),
               std::overflow_error);
}

TEST(test_fab2d, ranked_traversal_and_reductions_pass_ranked_indices) {
  const Box<1> line{Index<1>{-1}, Index<1>{2}};
  const Box<2> plane{Index<2>{0, 0}, Index<2>{1, 2}};
  const Box<3> volume{Index<3>{0, 0, 0}, Index<3>{1, 1, 1}};

  EXPECT_DOUBLE_EQ(for_each_cell_reduce_sum(line, SumRankedIndex<1>{}), 2.0);
  EXPECT_DOUBLE_EQ(for_each_cell_reduce_sum(plane, SumRankedIndex<2>{}), 9.0);
  EXPECT_DOUBLE_EQ(for_each_cell_reduce_sum(volume, SumRankedIndex<3>{}), 12.0);
  EXPECT_DOUBLE_EQ(for_each_cell_reduce_max(line, SumRankedIndex<1>{}), 2.0);
  EXPECT_DOUBLE_EQ(for_each_cell_reduce_max(plane, SumRankedIndex<2>{}), 3.0);
  EXPECT_DOUBLE_EQ(for_each_cell_reduce_max(volume, SumRankedIndex<3>{}), 3.0);
}

TEST(test_fab2d, ranked_max_reduction_preserves_least_negative_result_in_1d_2d_and_3d) {
  const Box<1> line{Index<1>{-4}, Index<1>{-2}};
  const Box<2> plane{Index<2>{-3, -3}, Index<2>{-2, -2}};
  const Box<3> volume{Index<3>{-2, -2, -2}, Index<3>{-1, -1, -1}};

  EXPECT_DOUBLE_EQ(for_each_cell_reduce_max(line, NegativeRankedIndex<1>{}), -5.0);
  EXPECT_DOUBLE_EQ(for_each_cell_reduce_max(plane, NegativeRankedIndex<2>{}), -9.0);
  EXPECT_DOUBLE_EQ(for_each_cell_reduce_max(volume, NegativeRankedIndex<3>{}), -4.0);
}

TEST(test_fab2d, ranked_small_host_boxes_use_existing_fallback_counter) {
  if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>) {
    reset_fallback_diagnostics_counters();
    if (detail::foreach_serial_threshold() > 1) {
      const Box<1> line{Index<1>{0}, Index<1>{0}};
      for_each_cell(line, NoOpRankedIndex<1>{});
      EXPECT_EQ(fallback_count(FallbackCounter::kForeachSerialSmallBox), 1u);
    }
  }
}

TEST(test_fab2d, ranked_fallback_threshold_does_not_multiply_large_extents) {
  EXPECT_TRUE(detail::foreach_small_box(63, 65, 4096));
  EXPECT_FALSE(detail::foreach_small_box(64, 64, 4096));
  EXPECT_FALSE(detail::foreach_small_box(std::numeric_limits<int>::max(),
                                         std::numeric_limits<int>::max(), 4096));

  constexpr int minimum = std::numeric_limits<int>::min();
  const Box<3> all_negative{Index<3>{minimum, minimum, minimum}, Index<3>{-1, -1, -1}};
  EXPECT_FALSE(detail::foreach_small_box(all_negative, 4096));
}

TEST(test_fab2d, ranked_value_constructors_compile_in_a_kokkos_device_lambda) {
  detail::ensure_kokkos_initialized();
  Kokkos::View<int[3]> equal_on_device("pops_ranked_box_equality_device");
  Kokkos::parallel_for(
      "pops_ranked_value_device_construction", 1, KOKKOS_LAMBDA(const int) {
        const Index<1> index1{1};
        const Index<2> index2{1, 2};
        const Index<3> index3{1, 2, 3};
        const Extent<1> extent1{1};
        const Extent<2> extent2{1, 2};
        const Extent<3> extent3{1, 2, 3};
        const RealVector<1> vector1{1.0};
        const RealVector<2> vector2{1.0, 2.0};
        const RealVector<3> vector3{1.0, 2.0, 3.0};
        const Box<1> box1{index1, index1};
        const Box<2> box2{index2, index2};
        const Box<3> box3{index3, index3};
        equal_on_device(0) = box1 == Box<1>{index1, index1};
        equal_on_device(1) = box2 == Box<2>{index2, index2};
        equal_on_device(2) = box3 == Box<3>{index3, index3};
        (void)extent1;
        (void)extent2;
        (void)extent3;
        (void)vector1;
        (void)vector2;
        (void)vector3;
        (void)box1;
        (void)box2;
        (void)box3;
      });
  Kokkos::fence();
  const auto equal_on_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, equal_on_device);
  EXPECT_EQ(equal_on_host(0), 1);
  EXPECT_EQ(equal_on_host(1), 1);
  EXPECT_EQ(equal_on_host(2), 1);
}

TEST(test_fab2d, ranked_fab_copy_owns_distinct_storage) {
  const Box<2> box{Index<2>{-1, 2}, Index<2>{1, 3}};
  Fab<2> original(box, /*ncomp=*/1, Extent<2>{});
  original.set_val(Real(3.5));

  Fab<2> copy = original;
  EXPECT_NE(copy.storage().data(), original.storage().data());

  auto original_host = original.create_host_mirror();
  auto copy_host = copy.create_host_mirror();
  original.copy_to_host(original_host);
  copy.copy_to_host(copy_host);
  EXPECT_DOUBLE_EQ(copy_host(0), original_host(0));

  copy.set_val(Real(-8.0));
  auto mutated_copy_host = copy.create_host_mirror();
  copy.copy_to_host(mutated_copy_host);
  original.copy_to_host(original_host);
  EXPECT_DOUBLE_EQ(mutated_copy_host(0), -8.0);
  EXPECT_DOUBLE_EQ(original_host(0), 3.5);

  Fab<2> assigned;
  assigned = original;
  EXPECT_NE(assigned.storage().data(), original.storage().data());
}

TEST(test_fab2d, ranked_host_mirrors_reject_cross_fab_and_stale_associations) {
  static_assert(
      std::is_same_v<decltype(std::declval<Fab<1>&>().storage()), const Fab<1>::storage_type&>);

  const Box<1> box{Index<1>{0}, Index<1>{1}};
  Fab<1> source(box, /*ncomp=*/1, Extent<1>{});
  Fab<1> other(box, /*ncomp=*/1, Extent<1>{});
  auto source_mirror = source.create_host_mirror();
  EXPECT_THROW(other.copy_to_host(source_mirror), std::invalid_argument);
  EXPECT_THROW(other.copy_from_host(source_mirror), std::invalid_argument);

  Fab<1> moved(std::move(source));
  EXPECT_EQ(source.size(), 0U);
  EXPECT_THROW(moved.copy_to_host(source_mirror), std::invalid_argument);
  EXPECT_THROW(source.copy_to_host(source_mirror), std::invalid_argument);
  auto moved_mirror = moved.create_host_mirror();
  EXPECT_NO_THROW(moved.copy_to_host(moved_mirror));

  auto other_mirror = other.create_host_mirror();
  other = std::move(moved);
  EXPECT_THROW(other.copy_to_host(other_mirror), std::invalid_argument);
  EXPECT_THROW(other.copy_to_host(moved_mirror), std::invalid_argument);
  auto rebound_mirror = other.create_host_mirror();
  EXPECT_NO_THROW(other.copy_to_host(rebound_mirror));

  Fab<1> resized(box, /*ncomp=*/1, Extent<1>{});
  auto stale_extent = resized.create_host_mirror();
  resized = Fab<1>(Box<1>{Index<1>{0}, Index<1>{2}}, /*ncomp=*/1, Extent<1>{});
  EXPECT_THROW(resized.copy_to_host(stale_extent), std::invalid_argument);

  Fab<2> empty;
  auto empty_mirror = empty.create_host_mirror();
  EXPECT_EQ(empty_mirror.size(), 0U);
  EXPECT_NO_THROW(empty.copy_to_host(empty_mirror));
  EXPECT_NO_THROW(empty.copy_from_host(empty_mirror));
}
