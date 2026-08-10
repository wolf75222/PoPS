#include <gtest/gtest.h>

#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/runtime/amr/prepared_amr_ghost_fill.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace pops;
using namespace pops::mesh;
using namespace pops::runtime::amr;

namespace {

std::size_t offset(const Fab<1>& fab, int index) {
  return static_cast<std::size_t>(index - fab.grown_box().lo[0]);
}

void fill_parent(MultiFab<1>& coarse) {
  for (std::size_t local = 0; local < coarse.local_size(); ++local) {
    auto& fab = coarse.fab(local);
    auto host = fab.create_host_mirror();
    for (int i = fab.box().lo[0]; i <= fab.box().hi[0]; ++i)
      host(offset(fab, i)) = static_cast<Real>(i);
    fab.copy_from_host(host);
  }
}

Real fifth_power(Real value) {
  const Real square = value * value;
  return square * square * value;
}

Real quartic_average(Real lower, Real upper) {
  return (fifth_power(upper) - fifth_power(lower)) / (Real(5) * (upper - lower));
}

void fill_quartic_parent(MultiFab<1>& coarse) {
  for (std::size_t local = 0; local < coarse.local_size(); ++local) {
    auto& fab = coarse.fab(local);
    auto host = fab.create_host_mirror();
    for (int i = fab.box().lo[0]; i <= fab.box().hi[0]; ++i) {
      const Real center = static_cast<Real>(i);
      host(offset(fab, i)) = quartic_average(center - Real(0.5), center + Real(0.5));
    }
    fab.copy_from_host(host);
  }
}

void fill_child(MultiFab<1>& fine) {
  for (std::size_t local = 0; local < fine.local_size(); ++local) {
    auto& fab = fine.fab(local);
    auto host = fab.create_host_mirror();
    for (int i = fab.grown_box().lo[0]; i <= fab.grown_box().hi[0]; ++i)
      host(offset(fab, i)) = fab.box().contains(Index<1>{i}) ? Real(100 + i) : Real(-777);
    fab.copy_from_host(host);
  }
}

Real value(const Fab<1>& fab, int index) {
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  return host(offset(fab, index));
}

AmrGhostFillBudget budget() {
  return AmrGhostFillBudget{CoarseFineGhostScheduleBudget{2, 16, 16, 64, 4, 1024, 1024, 1024},
                            HaloScheduleBudget{{2, 1}, 32, 64, 4, 4, 1024, 1024, 1024}};
}

void prove_mpi_fifth_order();
void prove_mpi_centered_and_temporal_transfers();

}  // namespace

TEST(test_mpi_prepared_amr_ghost_fill,
     partitioned_parent_and_swapped_fine_ownership_fill_one_exact_candidate) {
  Kokkos::ScopeGuard kokkos;
  const ExecutionLane control = ExecutionLane::world();
  ASSERT_EQ(control.size(), 3);
  const int rank = control.rank();
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{3}};

  const Box<1> coarse_domain{Index<1>{0}, Index<1>{7}};
  const BoxArray<1> coarse_layout(
      std::vector<Box<1>>{{Index<1>{0}, Index<1>{3}}, {Index<1>{4}, Index<1>{7}}});
  const Distribution<1> coarse_distribution = Distribution<1>::partitioned(
      coarse_layout, ranks, std::vector<Index<1>>{Index<1>{0}, Index<1>{1}});
  MultiFab<1> coarse(coarse_layout, coarse_distribution, Index<1>{rank}, 1, Extent<1>{0});

  const Box<1> fine_domain{Index<1>{0}, Index<1>{15}};
  const BoxArray<1> fine_layout(
      std::vector<Box<1>>{{Index<1>{4}, Index<1>{7}}, {Index<1>{8}, Index<1>{11}}});
  const Distribution<1> fine_distribution = Distribution<1>::partitioned(
      fine_layout, ranks, std::vector<Index<1>>{Index<1>{1}, Index<1>{0}});
  MultiFab<1> fine(fine_layout, fine_distribution, Index<1>{rank}, 1, Extent<1>{1});
  fill_parent(coarse);
  fill_child(fine);

  ExecutionLane lane =
      ExecutionLane::duplicate_world_collectively("test-mpi-prepared-amr-ghost-fill");
  AmrGhostFillPreparation<1> request{};
  request.fine_level = 1;
  request.coarse_domain = coarse_domain;
  request.fine_domain = fine_domain;
  request.ratio = ::pops::amr::RefinementRatio<1>(2);
  request.topology_generation = 17;
  request.materialization_generation = 23;
  request.field_identity = "state";
  request.budget = budget();
  const auto fill = prepare_amr_ghost_fill(coarse, fine, request, lane);
  EXPECT_EQ(fill.has_remote_parent_jobs(), rank < 2);
  EXPECT_EQ(fill.has_remote_same_level_jobs(), rank < 2);

  runtime::multiblock::BoundaryEvaluationPoint point{};
  point.level = 1;
  fill(fine, point);

  if (rank == 2) {
    EXPECT_EQ(fine.local_size(), 0U);
  } else {
    ASSERT_EQ(fine.local_size(), 1U);
    const auto& local = fine.fab(0);
    for (int i = local.grown_box().lo[0]; i <= local.grown_box().hi[0]; ++i) {
      Real expected = Real(100 + i);
      if (!fine_layout[0].contains(Index<1>{i}) && !fine_layout[1].contains(Index<1>{i}))
        expected = static_cast<Real>(i / 2) + (i % 2 == 0 ? Real(-0.25) : Real(0.25));
      EXPECT_DOUBLE_EQ(value(local, i), expected);
    }
  }
  prove_mpi_fifth_order();
  prove_mpi_centered_and_temporal_transfers();
}

namespace {

void prove_mpi_fifth_order() {
  const ExecutionLane control = ExecutionLane::world();
  ASSERT_EQ(control.size(), 3);
  const int rank = control.rank();
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{3}};

  const Box<1> coarse_domain{Index<1>{0}, Index<1>{15}};
  const BoxArray<1> coarse_layout(
      std::vector<Box<1>>{{Index<1>{0}, Index<1>{7}}, {Index<1>{8}, Index<1>{15}}});
  const Distribution<1> coarse_distribution = Distribution<1>::partitioned(
      coarse_layout, ranks, std::vector<Index<1>>{Index<1>{0}, Index<1>{1}});
  MultiFab<1> coarse(coarse_layout, coarse_distribution, Index<1>{rank}, 1, Extent<1>{0});

  const Box<1> fine_domain{Index<1>{0}, Index<1>{31}};
  const BoxArray<1> fine_layout(
      std::vector<Box<1>>{{Index<1>{8}, Index<1>{15}}, {Index<1>{16}, Index<1>{23}}});
  const Distribution<1> fine_distribution = Distribution<1>::partitioned(
      fine_layout, ranks, std::vector<Index<1>>{Index<1>{1}, Index<1>{0}});
  MultiFab<1> fine(fine_layout, fine_distribution, Index<1>{rank}, 1, Extent<1>{3});
  fill_quartic_parent(coarse);
  fill_child(fine);

  ExecutionLane lane =
      ExecutionLane::duplicate_world_collectively("test-mpi-prepared-amr-ghost-fill-order5");
  AmrGhostFillPreparation<1> request{};
  request.fine_level = 1;
  request.coarse_domain = coarse_domain;
  request.fine_domain = fine_domain;
  request.ratio = ::pops::amr::RefinementRatio<1>(2);
  request.interpolation_kind =
      ::pops::amr::transfer::TransferKind::FifthOrderCoarseFineGhostInterpolation;
  request.topology_generation = 29;
  request.materialization_generation = 31;
  request.field_identity = "weno-state";
  request.budget = budget();
  const auto fill = prepare_amr_ghost_fill(coarse, fine, request, lane);
  EXPECT_EQ(fill.has_remote_parent_jobs(), rank < 2);

  runtime::multiblock::BoundaryEvaluationPoint point{};
  point.level = 1;
  fill(fine, point);

  if (rank == 2) {
    EXPECT_EQ(fine.local_size(), 0U);
    return;
  }
  ASSERT_EQ(fine.local_size(), 1U);
  const auto& local = fine.fab(0);
  for (int i = local.grown_box().lo[0]; i <= local.grown_box().hi[0]; ++i) {
    if (fine_layout[0].contains(Index<1>{i}) || fine_layout[1].contains(Index<1>{i})) {
      EXPECT_DOUBLE_EQ(value(local, i), Real(100 + i));
      continue;
    }
    const Real lower = static_cast<Real>(i) / Real(2) - Real(0.5);
    const Real upper = static_cast<Real>(i + 1) / Real(2) - Real(0.5);
    EXPECT_NEAR(value(local, i), quartic_average(lower, upper), Real(2e-10));
  }
}

template <int Dim, class F>
void visit_nd(const Box<Dim>& box, F&& function) {
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
class MpiHostField {
 public:
  explicit MpiHostField(Box<Dim> box, int components = 1)
      : box_(box),
        components_(components),
        values_(static_cast<std::size_t>(box.numPts()) * static_cast<std::size_t>(components)) {}

  FieldView<Real, Dim> view() {
    FieldView<Real, Dim> result{};
    populate_(result);
    return result;
  }
  FieldView<const Real, Dim> const_view() const {
    FieldView<const Real, Dim> result{};
    populate_(result);
    return result;
  }
  Real& operator()(const Index<Dim>& index, int component = 0) { return view()(index, component); }
  Real operator()(const Index<Dim>& index, int component = 0) const {
    return const_view()(index, component);
  }
  const Box<Dim>& box() const { return box_; }
  Real* data() { return values_.data(); }
  std::size_t size() const { return values_.size(); }

 private:
  template <class T>
  void populate_(FieldView<T, Dim>& result) const {
    result.data = values_.data();
    result.origin = box_.lo;
    result.extents = box_.extent();
    result.strides[0] = 1;
    for (int axis = 1; axis < Dim; ++axis)
      result.strides[axis] = result.strides[axis - 1] * result.extents[axis - 1];
    result.ncomp = components_;
    result.component_stride = box_.numPts();
  }

  Box<Dim> box_{};
  int components_ = 0;
  mutable std::vector<Real> values_{};
};

template <int Dim, class Value>
void collectively_stage_halo(MpiHostField<Dim>& field, int rank, Value&& value) {
  static_assert(std::is_same_v<Real, double>);
  visit_nd(field.box(), [&](const Index<Dim>& index) {
    const int owner = (index[0] - field.box().lo[0]) % 2;
    field(index) = rank == owner ? value(index) : Real(0);
  });
  all_reduce_sum_inplace(field.data(), field.size());
}

template <int Dim>
::pops::amr::RefinementRatio<Dim> centered_ratio() {
  if constexpr (Dim == 1)
    return ::pops::amr::RefinementRatio<1>{3};
  else if constexpr (Dim == 2)
    return ::pops::amr::RefinementRatio<2>{2, 3};
  else
    return ::pops::amr::RefinementRatio<3>{2, 1, 3};
}

template <int Dim>
runtime::amr::AmrRuntime<Dim> centered_runtime(int rank) {
  namespace hierarchy = ::pops::amr::hierarchy;
  Index<Dim> coarse_hi{};
  for (int axis = 0; axis < Dim; ++axis)
    coarse_hi[axis] = 1;
  const Box<Dim> coarse_domain{Index<Dim>{}, coarse_hi};
  const auto ratio = centered_ratio<Dim>();
  const Box<Dim> fine_domain = hierarchy::refine_box(coarse_domain, ratio);
  const BoxArray<Dim> coarse_boxes(std::vector<Box<Dim>>{coarse_domain});
  const BoxArray<Dim> fine_boxes(std::vector<Box<Dim>>{fine_domain});
  Extent<Dim> process_shape{};
  for (int axis = 0; axis < Dim; ++axis)
    process_shape[axis] = 1;
  process_shape[0] = 3;
  const RankSpace<Dim> ranks(Index<Dim>{}, process_shape);
  const auto coarse_distribution = Distribution<Dim>::replicated(coarse_boxes, ranks);
  const auto fine_distribution = Distribution<Dim>::replicated(fine_boxes, ranks);
  constexpr BoxArrayValidationBudget layout_budget{4, 16};
  hierarchy::LevelLayout<Dim> coarse_layout(0, coarse_domain, coarse_boxes, coarse_distribution,
                                            ::pops::amr::RefinementRatio<Dim>{}, layout_budget);
  hierarchy::LevelLayout<Dim> fine_layout(1, fine_domain, fine_boxes, fine_distribution, ratio,
                                          layout_budget);
  Index<Dim> local_rank{};
  local_rank[0] = rank;
  MultiFab<Dim> coarse(coarse_boxes, coarse_distribution, local_rank, 1, Extent<Dim>{});
  MultiFab<Dim> fine(fine_boxes, fine_distribution, local_rank, 1, Extent<Dim>{});
  std::vector<hierarchy::AmrLevelState<Dim>> levels;
  levels.emplace_back(std::move(coarse_layout), std::move(coarse));
  levels.emplace_back(std::move(fine_layout), std::move(fine));
  auto load_balance =
      std::make_shared<const PreparedLoadBalanceAuthority<Dim>>(prepare_load_balance_authority<Dim>(
          "space_filling_curve", "test.mpi.centered-transfer.sfc",
          PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
  return runtime::amr::AmrRuntime<Dim>(
      hierarchy::AmrHierarchy<Dim>(std::move(levels), hierarchy::HierarchyValidationBudget{2, 8}),
      std::move(load_balance), "test.mpi.centered-transfer.spatial");
}

template <int Dim>
void prove_centered_and_temporal(const ExecutionLane& lane) {
  namespace transfer = ::pops::amr::transfer;
  namespace time_amr = ::pops::numerics::time::amr;
  const int rank = lane.rank();
  auto runtime = centered_runtime<Dim>(rank);
  const auto ratio = runtime.hierarchy().layout(1).ratio_from_parent();
  const transfer::IndexMapping<Dim> mapping{runtime.hierarchy().layout(0).domain().lo,
                                            runtime.hierarchy().layout(1).domain().lo};
  const Box<Dim> coarse_cells = runtime.hierarchy().layout(0).domain();
  const Box<Dim> fine_cells = runtime.hierarchy().layout(1).domain();

  Box<Dim> coarse_nodes = coarse_cells;
  Box<Dim> fine_nodes = fine_cells;
  for (int axis = 0; axis < Dim; ++axis) {
    ++coarse_nodes.hi[axis];
    ++fine_nodes.hi[axis];
  }
  MpiHostField<Dim> parent_nodes(coarse_nodes);
  MpiHostField<Dim> child_nodes(fine_nodes);
  collectively_stage_halo(parent_nodes, rank, [&](const Index<Dim>& node) {
    Real value = Real(0.4);
    for (int axis = 0; axis < Dim; ++axis)
      value += Real(0.12 * (axis + 1)) * Real(node[axis]);
    return value;
  });
  const auto node = time_amr::prepare_node_multilinear_collectively(
      runtime, 0, parent_nodes.const_view(), child_nodes.view(), fine_nodes, lane, mapping);
  visit_nd(fine_nodes, [&](const Index<Dim>& index) { node(index); });
  Real node_error = Real(0);
  visit_nd(fine_nodes, [&](const Index<Dim>& index) {
    Real expected = Real(0.4);
    for (int axis = 0; axis < Dim; ++axis)
      expected += Real(0.12 * (axis + 1)) * Real(index[axis]) / Real(ratio[axis]);
    node_error = std::max(node_error, std::abs(child_nodes(index) - expected));
  });
  EXPECT_LT(all_reduce_max(node_error), Real(3e-13));

  Box<Dim> coarse_face_source = coarse_cells;
  for (int axis = 0; axis < Dim; ++axis) {
    --coarse_face_source.lo[axis];
    coarse_face_source.hi[axis] += 2;
  }
  std::vector<MpiHostField<Dim>> parent_faces;
  std::vector<MpiHostField<Dim>> child_faces;
  parent_faces.reserve(Dim);
  child_faces.reserve(Dim);
  std::array<FieldView<const Real, Dim>, Dim> parent_face_views{};
  std::array<FieldView<Real, Dim>, Dim> child_face_views{};
  for (int normal_axis = 0; normal_axis < Dim; ++normal_axis) {
    Box<Dim> fine_face_region = fine_cells;
    ++fine_face_region.hi[normal_axis];
    parent_faces.emplace_back(coarse_face_source);
    child_faces.emplace_back(fine_face_region);
    collectively_stage_halo(parent_faces.back(), rank, [&](const Index<Dim>& face) {
      Real result = Real(0.3 * (normal_axis + 1));
      for (int axis = 0; axis < Dim; ++axis) {
        const Real coordinate = Real(face[axis]);
        result += Real(0.08 * (normal_axis + 1) * (axis + 1)) * coordinate;
        result += Real(0.009 * (axis + 1)) * coordinate * coordinate;
      }
      return result;
    });
  }
  for (int axis = 0; axis < Dim; ++axis) {
    parent_face_views[axis] = parent_faces[static_cast<std::size_t>(axis)].const_view();
    child_face_views[axis] = child_faces[static_cast<std::size_t>(axis)].view();
  }
  const auto face = time_amr::prepare_divergence_preserving_faces_collectively(
      runtime, 0, parent_face_views, child_face_views, fine_cells, lane, mapping);
  for (int axis = 0; axis < Dim; ++axis)
    visit_nd(face.destination_face_region(axis),
             [&](const Index<Dim>& index) { face(axis, index); });
  Real divergence_error = Real(0);
  visit_nd(fine_cells, [&](const Index<Dim>& fine_cell) {
    Index<Dim> parent{};
    for (int axis = 0; axis < Dim; ++axis)
      parent[axis] = fine_cell[axis] / ratio[axis];
    Real coarse_divergence = Real(0);
    Real fine_divergence = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> coarse_upper = parent;
      Index<Dim> fine_upper = fine_cell;
      ++coarse_upper[axis];
      ++fine_upper[axis];
      coarse_divergence += parent_faces[static_cast<std::size_t>(axis)](coarse_upper) -
                           parent_faces[static_cast<std::size_t>(axis)](parent);
      fine_divergence +=
          Real(ratio[axis]) * (child_faces[static_cast<std::size_t>(axis)](fine_upper) -
                               child_faces[static_cast<std::size_t>(axis)](fine_cell));
    }
    divergence_error = std::max(divergence_error, std::abs(fine_divergence - coarse_divergence));
  });
  EXPECT_LT(all_reduce_max(divergence_error), Real(3e-12));

  MpiHostField<Dim> older(coarse_cells);
  MpiHostField<Dim> newer(coarse_cells);
  MpiHostField<Dim> candidate(coarse_cells);
  collectively_stage_halo(older, rank, [&](const Index<Dim>& cell) {
    Real result = Real(1);
    for (int axis = 0; axis < Dim; ++axis)
      result += Real(axis + 1) * Real(cell[axis]);
    return result;
  });
  collectively_stage_halo(newer, rank, [&](const Index<Dim>& cell) {
    Real result = Real(5);
    for (int axis = 0; axis < Dim; ++axis)
      result -= Real(0.25 * (axis + 1)) * Real(cell[axis]);
    return result;
  });
  const std::string local_spatial(runtime.spatial_contract());
  const transfer::QualifiedTemporalState older_state{
      "state/U", local_spatial, runtime.topology_epoch(), runtime.materialization_generation(),
      ::pops::amr::ClockStamp{0, 8, ::pops::amr::Rational(0, 1), 2.0}};
  const transfer::QualifiedTemporalState newer_state{
      "state/U", local_spatial, runtime.topology_epoch(), runtime.materialization_generation(),
      ::pops::amr::ClockStamp{0, 8, ::pops::amr::Rational(1, 1), 4.0}};
  const transfer::QualifiedTemporalState target_state{
      "state/U", local_spatial, runtime.topology_epoch(), runtime.materialization_generation(),
      ::pops::amr::ClockStamp{0, 8, ::pops::amr::Rational(1, 2), 3.0}};
  const auto temporal = time_amr::prepare_linear_time_interpolation_collectively(
      runtime, 0, older.const_view(), newer.const_view(), candidate.view(), coarse_cells,
      older_state, newer_state, target_state, lane);
  visit_nd(coarse_cells, [&](const Index<Dim>& index) { temporal(index); });
  Real temporal_error = Real(0);
  visit_nd(coarse_cells, [&](const Index<Dim>& index) {
    const Real expected = Real(0.5) * (older(index) + newer(index));
    temporal_error = std::max(temporal_error, std::abs(candidate(index) - expected));
  });
  EXPECT_LT(all_reduce_max(temporal_error), Real(2e-13));

  MpiHostField<Dim> refused_candidate(coarse_cells);
  visit_nd(coarse_cells, [&](const Index<Dim>& index) { refused_candidate(index) = Real(-321); });
  transfer::QualifiedTemporalState inconsistent = target_state;
  if (rank == 1)
    ++inconsistent.topology_generation;
  EXPECT_THROW((void)time_amr::prepare_linear_time_interpolation_collectively(
                   runtime, 0, older.const_view(), newer.const_view(), refused_candidate.view(),
                   coarse_cells, older_state, newer_state, inconsistent, lane),
               std::invalid_argument);
  visit_nd(coarse_cells,
           [&](const Index<Dim>& index) { EXPECT_EQ(refused_candidate(index), Real(-321)); });
}

void prove_mpi_centered_and_temporal_transfers() {
  const ExecutionLane lane =
      ExecutionLane::duplicate_world_collectively("test-mpi-centered-temporal-transfers");
  prove_centered_and_temporal<1>(lane);
  prove_centered_and_temporal<2>(lane);
  prove_centered_and_temporal<3>(lane);
}

}  // namespace
