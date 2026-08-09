#include <gtest/gtest.h>

#include <pops/runtime/amr/prepared_amr_ghost_fill.hpp>

#include "../mesh/nd_multifab_test_utils.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace pops;
using namespace pops::mesh;
using namespace pops::runtime::amr;
using namespace pops::test::nd;

namespace {

template <int Dim>
Box<Dim> box(int lower, int upper) {
  return cube<Dim>(lower, upper);
}

template <int Dim>
AmrGhostFillBudget budget(std::size_t coarse_boxes, std::size_t fine_boxes) {
  const std::size_t pairs =
      coarse_boxes * fine_boxes + coarse_boxes * coarse_boxes + fine_boxes * fine_boxes + 16;
  return AmrGhostFillBudget{
      CoarseFineGhostScheduleBudget{fine_boxes, 32 * fine_boxes, pairs, 128 * fine_boxes, 16,
                                    1'000'000, 1'000'000, 1'000'000},
      HaloScheduleBudget{{fine_boxes, pairs},
                         fine_boxes * fine_boxes * 64,
                         fine_boxes * fine_boxes * 64 * static_cast<std::size_t>(2 * Dim),
                         64,
                         16,
                         1'000'000,
                         1'000'000,
                         1'000'000}};
}

template <int Dim>
mesh::Distribution<Dim> replicated(const mesh::BoxArray<Dim>& layout) {
  return mesh::Distribution<Dim>::replicated(layout, one_rank_space<Dim>());
}

template <int Dim>
::pops::amr::RefinementRatio<Dim> ratio_two() {
  std::array<int, Dim> values{};
  values.fill(2);
  return ::pops::amr::RefinementRatio<Dim>(values);
}

template <int Dim>
Real expected_linear_parent(const Index<Dim>& fine, int component) {
  Real result = static_cast<Real>(10000 * component);
  Real scale = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    const int parent = fine[axis] / 2;
    const int child = fine[axis] % 2;
    result += scale * (static_cast<Real>(parent) + (child == 0 ? Real(-0.25) : Real(0.25)));
    scale *= 97;
  }
  return result;
}

template <int Dim>
void prove_sparse_parent_interpolation() {
  const Box<Dim> coarse_domain = box<Dim>(0, 7);
  const Box<Dim> fine_domain = box<Dim>(0, 15);
  const BoxArray<Dim> coarse_layout(std::vector<Box<Dim>>{coarse_domain});
  const BoxArray<Dim> fine_layout(std::vector<Box<Dim>>{box<Dim>(4, 11)});
  HostMultiFab<Dim> coarse(coarse_layout, replicated(coarse_layout), Index<Dim>{}, 2,
                           uniform_extent<Dim>(0));
  HostMultiFab<Dim> fine(fine_layout, replicated(fine_layout), Index<Dim>{}, 2,
                         uniform_extent<Dim>(1));
  fill_valid_encoded(coarse, Real{-1});
  fill_valid(fine, Real{-777},
             [](const Index<Dim>&, int component) { return Real(8000 + component); });

  AmrGhostFillPreparation<Dim> request{};
  request.fine_level = 1;
  request.coarse_domain = coarse_domain;
  request.fine_domain = fine_domain;
  request.ratio = ratio_two<Dim>();
  request.topology = BoundaryTopology<Dim>::physical();
  request.topology_generation = 7;
  request.materialization_generation = 11;
  request.field_identity = "state";
  request.budget = budget<Dim>(1, 1);
  const ExecutionLane lane = ExecutionLane::world();
  const auto fill = prepare_amr_ghost_fill(coarse, fine, request, lane);

  runtime::multiblock::BoundaryEvaluationPoint point{};
  point.level = 1;
  fill(fine, point);

  const auto& fab = fine.fab_global(0);
  const std::size_t cells = static_cast<std::size_t>(fab.grown_box().numPts());
  for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
    const Index<Dim> index = index_from_ordinal(fab.grown_box(), ordinal);
    for (int component = 0; component < fine.ncomp(); ++component) {
      const Real expected = fab.box().contains(index) ? Real(8000 + component)
                                                      : expected_linear_parent(index, component);
      EXPECT_DOUBLE_EQ(value_at(fine, 0, index, component), expected);
    }
  }
}

}  // namespace

TEST(test_prepared_amr_ghost_fill, sparse_parent_interpolation_is_exact_in_1d_2d_and_3d) {
  prove_sparse_parent_interpolation<1>();
  prove_sparse_parent_interpolation<2>();
  prove_sparse_parent_interpolation<3>();
}

TEST(test_prepared_amr_ghost_fill, same_level_values_override_parent_interpolation) {
  const Box<1> coarse_domain{Index<1>{0}, Index<1>{7}};
  const Box<1> fine_domain{Index<1>{0}, Index<1>{15}};
  const BoxArray<1> coarse_layout(std::vector<Box<1>>{coarse_domain});
  const BoxArray<1> fine_layout(
      std::vector<Box<1>>{{Index<1>{4}, Index<1>{7}}, {Index<1>{8}, Index<1>{11}}});
  HostMultiFab<1> coarse(coarse_layout, replicated(coarse_layout), Index<1>{}, 1, Extent<1>{0});
  HostMultiFab<1> fine(fine_layout, replicated(fine_layout), Index<1>{}, 1, Extent<1>{1});
  fill_valid(coarse, Real{-1}, [](const Index<1>&, int) { return Real(2); });
  fill_valid(fine, Real{-777}, [](const Index<1>& index, int) { return Real(100 + index[0]); });

  AmrGhostFillPreparation<1> request{};
  request.fine_level = 1;
  request.coarse_domain = coarse_domain;
  request.fine_domain = fine_domain;
  request.ratio = ::pops::amr::RefinementRatio<1>(2);
  request.topology_generation = 1;
  request.materialization_generation = 2;
  request.field_identity = "state";
  request.budget = budget<1>(1, 2);
  const ExecutionLane lane = ExecutionLane::world();
  const auto fill = prepare_amr_ghost_fill(coarse, fine, request, lane);
  runtime::multiblock::BoundaryEvaluationPoint point{};
  point.level = 1;
  fill(fine, point);

  EXPECT_DOUBLE_EQ(value_at(fine, 0, Index<1>{8}), 108);
  EXPECT_DOUBLE_EQ(value_at(fine, 1, Index<1>{7}), 107);
  EXPECT_DOUBLE_EQ(value_at(fine, 0, Index<1>{3}), 2);
  EXPECT_DOUBLE_EQ(value_at(fine, 1, Index<1>{12}), 2);
}

TEST(test_prepared_amr_ghost_fill, physical_face_ghosts_remain_for_the_boundary_provider) {
  const Box<1> coarse_domain{Index<1>{0}, Index<1>{7}};
  const Box<1> fine_domain{Index<1>{0}, Index<1>{15}};
  const BoxArray<1> coarse_layout(std::vector<Box<1>>{coarse_domain});
  const BoxArray<1> fine_layout(std::vector<Box<1>>{{Index<1>{0}, Index<1>{3}}});
  HostMultiFab<1> coarse(coarse_layout, replicated(coarse_layout), Index<1>{}, 1, Extent<1>{0});
  HostMultiFab<1> fine(fine_layout, replicated(fine_layout), Index<1>{}, 1, Extent<1>{1});
  fill_valid(coarse, Real{-1}, [](const Index<1>&, int) { return Real(3); });
  fill_valid(fine, Real{-777}, [](const Index<1>&, int) { return Real(9); });

  AmrGhostFillPreparation<1> request{};
  request.fine_level = 1;
  request.coarse_domain = coarse_domain;
  request.fine_domain = fine_domain;
  request.ratio = ::pops::amr::RefinementRatio<1>(2);
  request.topology_generation = 3;
  request.materialization_generation = 4;
  request.field_identity = "state";
  request.budget = budget<1>(1, 1);
  const ExecutionLane lane = ExecutionLane::world();
  const auto fill = prepare_amr_ghost_fill(coarse, fine, request, lane);
  runtime::multiblock::BoundaryEvaluationPoint point{};
  point.level = 1;
  fill(fine, point);

  EXPECT_DOUBLE_EQ(value_at(fine, 0, Index<1>{-1}), -777);
  EXPECT_DOUBLE_EQ(value_at(fine, 0, Index<1>{4}), 3);
}

TEST(test_prepared_amr_ghost_fill, periodic_sparse_ghost_without_a_fine_peer_uses_the_parent) {
  const Box<1> coarse_domain{Index<1>{0}, Index<1>{7}};
  const Box<1> fine_domain{Index<1>{0}, Index<1>{15}};
  const BoxArray<1> coarse_layout(std::vector<Box<1>>{coarse_domain});
  const BoxArray<1> fine_layout(std::vector<Box<1>>{{Index<1>{0}, Index<1>{3}}});
  HostMultiFab<1> coarse(coarse_layout, replicated(coarse_layout), Index<1>{}, 1, Extent<1>{0});
  HostMultiFab<1> fine(fine_layout, replicated(fine_layout), Index<1>{}, 1, Extent<1>{1});
  fill_valid(coarse, Real{-1}, [](const Index<1>&, int) { return Real(5); });
  fill_valid(fine, Real{-777}, [](const Index<1>&, int) { return Real(9); });

  AmrGhostFillPreparation<1> request{};
  request.fine_level = 1;
  request.coarse_domain = coarse_domain;
  request.fine_domain = fine_domain;
  request.ratio = ::pops::amr::RefinementRatio<1>(2);
  request.topology = BoundaryTopology<1>::axis_periodic(std::array<bool, 1>{true});
  request.topology_generation = 5;
  request.materialization_generation = 6;
  request.field_identity = "state";
  request.budget = budget<1>(1, 1);
  const ExecutionLane lane = ExecutionLane::world();
  const auto fill = prepare_amr_ghost_fill(coarse, fine, request, lane);
  runtime::multiblock::BoundaryEvaluationPoint point{};
  point.level = 1;
  fill(fine, point);

  EXPECT_DOUBLE_EQ(value_at(fine, 0, Index<1>{-1}), 5);
}

TEST(test_prepared_amr_ghost_fill, stale_generation_rejects_before_mutation) {
  const Box<1> coarse_domain{Index<1>{0}, Index<1>{7}};
  const Box<1> fine_domain{Index<1>{0}, Index<1>{15}};
  const BoxArray<1> coarse_layout(std::vector<Box<1>>{coarse_domain});
  const BoxArray<1> fine_layout(std::vector<Box<1>>{{Index<1>{4}, Index<1>{11}}});
  HostMultiFab<1> coarse(coarse_layout, replicated(coarse_layout), Index<1>{}, 1, Extent<1>{0});
  HostMultiFab<1> fine(fine_layout, replicated(fine_layout), Index<1>{}, 1, Extent<1>{1});
  coarse.set_val(Real(1));
  fine.set_val(Real(-5));
  AmrGhostFillPreparation<1> request{};
  request.fine_level = 1;
  request.coarse_domain = coarse_domain;
  request.fine_domain = fine_domain;
  request.ratio = ::pops::amr::RefinementRatio<1>(2);
  request.topology_generation = 8;
  request.materialization_generation = 9;
  request.field_identity = "state";
  request.budget = budget<1>(1, 1);
  const ExecutionLane lane = ExecutionLane::world();
  const auto fill = prepare_amr_ghost_fill(coarse, fine, request, lane);
  const auto before = snapshot(fine);
  EXPECT_THROW(fill.execute(fine, 8, 10, lane), std::invalid_argument);
  EXPECT_EQ(snapshot(fine), before);
}
