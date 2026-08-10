/// @file
/// @brief Exact-ranked AMR transfer properties prepared by the live hierarchy authority.

#include <gtest/gtest.h>

#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace hierarchy = pops::amr::hierarchy;
namespace transfer = pops::amr::transfer;
namespace time_amr = pops::numerics::time::amr;

constexpr pops::mesh::BoxArrayValidationBudget kLayoutBudget{8, 28};
constexpr hierarchy::HierarchyValidationBudget kHierarchyBudget{2, 8};

template <int Dim>
pops::Extent<Dim> filled_extent(int value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::amr::RefinementRatio<Dim> refinement_ratio(int value) {
  std::array<int, Dim> values{};
  values.fill(value);
  return pops::amr::RefinementRatio<Dim>(values);
}

template <int Dim>
pops::Box<Dim> coarse_domain() {
  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = -3 + axis;
    upper[axis] = lower[axis] + 3;
  }
  return {lower, upper};
}

template <int Dim>
std::shared_ptr<const pops::PreparedLoadBalanceAuthority<Dim>> load_balance() {
  return std::make_shared<const pops::PreparedLoadBalanceAuthority<Dim>>(
      pops::prepare_load_balance_authority<Dim>(
          "space_filling_curve", "test.amr-transfer-properties.sfc",
          pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
}

template <int Dim>
pops::runtime::amr::AmrRuntime<Dim> make_runtime() {
  const pops::Box<Dim> parent_domain = coarse_domain<Dim>();
  const auto ratio = refinement_ratio<Dim>(2);
  const pops::Box<Dim> child_domain = hierarchy::refine_box(parent_domain, ratio);
  const pops::mesh::BoxArray<Dim> parent_boxes(std::vector<pops::Box<Dim>>{parent_domain});
  const pops::mesh::BoxArray<Dim> child_boxes(std::vector<pops::Box<Dim>>{child_domain});
  const pops::mesh::RankSpace<Dim> ranks(pops::Index<Dim>{}, filled_extent<Dim>(1));
  const auto parent_distribution = pops::mesh::Distribution<Dim>::replicated(parent_boxes, ranks);
  const auto child_distribution = pops::mesh::Distribution<Dim>::replicated(child_boxes, ranks);
  hierarchy::LevelLayout<Dim> parent_layout(0, parent_domain, parent_boxes, parent_distribution,
                                            pops::amr::RefinementRatio<Dim>{}, kLayoutBudget);
  hierarchy::LevelLayout<Dim> child_layout(1, child_domain, child_boxes, child_distribution, ratio,
                                           kLayoutBudget);
  pops::MultiFab<Dim> parent(parent_boxes, parent_distribution, pops::Index<Dim>{}, 1,
                             filled_extent<Dim>(2));
  pops::MultiFab<Dim> child(child_boxes, child_distribution, pops::Index<Dim>{}, 1,
                            filled_extent<Dim>(2));
  std::vector<hierarchy::AmrLevelState<Dim>> levels;
  levels.emplace_back(std::move(parent_layout), std::move(parent));
  levels.emplace_back(std::move(child_layout), std::move(child));
  return pops::runtime::amr::AmrRuntime<Dim>(
      hierarchy::AmrHierarchy<Dim>(std::move(levels), kHierarchyBudget), load_balance<Dim>(),
      "test.amr-transfer-properties.spatial");
}

template <int Dim>
POPS_HD pops::Real affine_parent(const pops::Index<Dim>& cell,
                                 const transfer::IndexMapping<Dim>& mapping) {
  pops::Real value = pops::Real(1.25);
  for (int axis = 0; axis < Dim; ++axis)
    value += pops::Real(0.2 * (axis + 1)) * pops::Real(cell[axis] - mapping.coarse_origin[axis]);
  return value;
}

template <int Dim>
POPS_HD pops::Real affine_child(const pops::Index<Dim>& cell,
                                const pops::amr::RefinementRatio<Dim>& ratio,
                                const transfer::IndexMapping<Dim>& mapping) {
  pops::Real value = pops::Real(1.25);
  for (int axis = 0; axis < Dim; ++axis) {
    const pops::Real relative = pops::Real(cell[axis] - mapping.fine_origin[axis]);
    const pops::Real parent_coordinate =
        (relative + pops::Real(0.5)) / pops::Real(ratio[axis]) - pops::Real(0.5);
    value += pops::Real(0.2 * (axis + 1)) * parent_coordinate;
  }
  return value;
}

template <int Dim>
struct FillAffineParent {
  pops::FieldView<pops::Real, Dim> values{};
  transfer::IndexMapping<Dim> mapping{};

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    values(cell) = affine_parent(cell, mapping);
  }
};

template <int Dim>
struct ParentError {
  pops::FieldView<const pops::Real, Dim> values{};
  transfer::IndexMapping<Dim> mapping{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    const pops::Real difference = values(cell) - affine_parent(cell, mapping);
    return difference < pops::Real(0) ? -difference : difference;
  }
};

template <int Dim>
struct ChildError {
  pops::FieldView<const pops::Real, Dim> values{};
  pops::amr::RefinementRatio<Dim> ratio{};
  transfer::IndexMapping<Dim> mapping{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    const pops::Real difference = values(cell) - affine_child(cell, ratio, mapping);
    return difference < pops::Real(0) ? -difference : difference;
  }
};

template <int Dim>
struct InjectionError {
  pops::FieldView<const pops::Real, Dim> child{};
  pops::FieldView<const pops::Real, Dim> parent{};
  pops::amr::RefinementRatio<Dim> ratio{};
  transfer::IndexMapping<Dim> mapping{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& fine) const {
    pops::Index<Dim> coarse{};
    for (int axis = 0; axis < Dim; ++axis) {
      const int relative = fine[axis] - mapping.fine_origin[axis];
      coarse[axis] = mapping.coarse_origin[axis] + relative / ratio[axis];
    }
    const pops::Real difference = child(fine) - parent(coarse);
    return difference < pops::Real(0) ? -difference : difference;
  }
};

template <int Dim>
struct FillNodeSource {
  pops::FieldView<pops::Real, Dim> values{};
  pops::Index<Dim> origin{};

  POPS_HD void operator()(const pops::Index<Dim>& node) const {
    pops::Real value = pops::Real(0.6);
    for (int axis = 0; axis < Dim; ++axis)
      value += pops::Real(0.17 * (axis + 1)) * pops::Real(node[axis] - origin[axis]);
    values(node) = value;
  }
};

template <int Dim>
struct NodeError {
  pops::FieldView<const pops::Real, Dim> values{};
  pops::amr::RefinementRatio<Dim> ratio{};
  transfer::IndexMapping<Dim> mapping{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& node) const {
    pops::Real expected = pops::Real(0.6);
    for (int axis = 0; axis < Dim; ++axis) {
      const pops::Real coordinate =
          pops::Real(node[axis] - mapping.fine_origin[axis]) / pops::Real(ratio[axis]);
      expected += pops::Real(0.17 * (axis + 1)) * coordinate;
    }
    const pops::Real difference = values(node) - expected;
    return difference < pops::Real(0) ? -difference : difference;
  }
};

template <int Dim>
struct FillFaceSource {
  pops::FieldView<pops::Real, Dim> values{};
  pops::Index<Dim> origin{};
  int normal_axis = 0;

  POPS_HD void operator()(const pops::Index<Dim>& face) const {
    pops::Real value = pops::Real(0.25 * (normal_axis + 1));
    for (int axis = 0; axis < Dim; ++axis) {
      const pops::Real coordinate = pops::Real(face[axis] - origin[axis]);
      value += pops::Real(0.09 * (normal_axis + 1) * (axis + 1)) * coordinate;
      value += pops::Real(0.007 * (axis + 1)) * coordinate * coordinate;
    }
    values(face) = value;
  }
};

template <int Dim>
struct FaceDivergenceError {
  std::array<pops::FieldView<const pops::Real, Dim>, Dim> coarse{};
  std::array<pops::FieldView<const pops::Real, Dim>, Dim> fine{};
  pops::amr::RefinementRatio<Dim> ratio{};
  transfer::IndexMapping<Dim> mapping{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& fine_cell) const {
    pops::Index<Dim> parent{};
    for (int axis = 0; axis < Dim; ++axis) {
      const int relative = fine_cell[axis] - mapping.fine_origin[axis];
      parent[axis] = mapping.coarse_origin[axis] +
                     static_cast<int>(transfer::detail::floor_div_positive(relative, ratio[axis]));
    }
    pops::Real coarse_divergence = pops::Real(0);
    pops::Real fine_divergence = pops::Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      pops::Index<Dim> coarse_upper = parent;
      pops::Index<Dim> fine_upper = fine_cell;
      ++coarse_upper[axis];
      ++fine_upper[axis];
      coarse_divergence += coarse[axis](coarse_upper) - coarse[axis](parent);
      fine_divergence += pops::Real(ratio[axis]) * (fine[axis](fine_upper) - fine[axis](fine_cell));
    }
    const pops::Real difference = fine_divergence - coarse_divergence;
    return difference < pops::Real(0) ? -difference : difference;
  }
};

template <int Dim>
struct FillTemporalSources {
  pops::FieldView<pops::Real, Dim> older{};
  pops::FieldView<pops::Real, Dim> newer{};

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    pops::Real coordinate = pops::Real(0);
    for (int axis = 0; axis < Dim; ++axis)
      coordinate += pops::Real(axis + 1) * pops::Real(cell[axis]);
    older(cell) = pops::Real(1) + coordinate;
    newer(cell) = pops::Real(7) - pops::Real(0.5) * coordinate;
  }
};

template <int Dim>
struct TemporalError {
  pops::FieldView<const pops::Real, Dim> older{};
  pops::FieldView<const pops::Real, Dim> newer{};
  pops::FieldView<const pops::Real, Dim> candidate{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    const pops::Real expected = older(cell) + pops::Real(0.25) * (newer(cell) - older(cell));
    const pops::Real difference = candidate(cell) - expected;
    return difference < pops::Real(0) ? -difference : difference;
  }
};

template <int Dim>
void prove_live_runtime_transfers() {
  auto runtime = make_runtime<Dim>();
  auto& parent = runtime.hierarchy().state(0);
  auto& child = runtime.hierarchy().state(1);
  ASSERT_EQ(parent.local_size(), 1U);
  ASSERT_EQ(child.local_size(), 1U);

  const auto ratio = runtime.hierarchy().layout(1).ratio_from_parent();
  const transfer::IndexMapping<Dim> mapping{runtime.hierarchy().layout(0).domain().lo,
                                            runtime.hierarchy().layout(1).domain().lo};
  pops::for_each_cell(parent.fab(0).grown_box(),
                      FillAffineParent<Dim>{parent.fab(0).view(), mapping});

  const auto prolongation = time_amr::prepare_linear_prolongation(
      runtime, 0, std::as_const(parent.fab(0)).view(), child.fab(0).view(), child.box(0), mapping);
  static_assert(std::is_same_v<decltype(prolongation), const transfer::PreparedTransfer<Dim>>);
  EXPECT_EQ(prolongation.kind(), transfer::TransferKind::LinearProlongation);
  EXPECT_EQ(prolongation.refinement_ratio(), ratio);
  time_amr::execute_prepared_transfer(prolongation);
  EXPECT_LT(pops::for_each_cell_reduce_max(
                child.box(0), ChildError<Dim>{std::as_const(child.fab(0)).view(), ratio, mapping}),
            pops::Real(2e-13));

  pops::MultiFab<Dim> restricted(parent.layout(), parent.distribution(), parent.local_rank(), 1,
                                 parent.ghosts());
  const auto restriction =
      time_amr::prepare_average_down(runtime, 1, std::as_const(child.fab(0)).view(),
                                     restricted.fab(0).view(), parent.box(0), mapping);
  EXPECT_EQ(restriction.kind(), transfer::TransferKind::ConservativeRestriction);
  time_amr::execute_prepared_transfer(restriction);
  EXPECT_LT(pops::for_each_cell_reduce_max(
                parent.box(0), ParentError<Dim>{std::as_const(restricted.fab(0)).view(), mapping}),
            pops::Real(2e-13));

  const auto injection = time_amr::prepare_constant_injection(
      runtime, 0, std::as_const(parent.fab(0)).view(), child.fab(0).view(), child.box(0), mapping);
  EXPECT_EQ(injection.kind(), transfer::TransferKind::ConstantInjection);
  time_amr::execute_prepared_transfer(injection);
  EXPECT_EQ(
      pops::for_each_cell_reduce_max(
          child.box(0), InjectionError<Dim>{std::as_const(child.fab(0)).view(),
                                            std::as_const(parent.fab(0)).view(), ratio, mapping}),
      pops::Real(0));

  pops::Box<Dim> ghost_region = child.box(0);
  ghost_region.hi[0] = ghost_region.lo[0] - 1;
  ghost_region.lo[0] = ghost_region.hi[0];
  const auto fill_patch = time_amr::prepare_fill_patch(
      runtime, 0, std::as_const(parent.fab(0)).view(), child.fab(0).view(), ghost_region, mapping);
  EXPECT_EQ(fill_patch.kind(), transfer::TransferKind::CoarseFineGhostInterpolation);
  time_amr::execute_prepared_transfer(fill_patch);
  EXPECT_LT(pops::for_each_cell_reduce_max(
                ghost_region, ChildError<Dim>{std::as_const(child.fab(0)).view(), ratio, mapping}),
            pops::Real(2e-13));

  const auto fifth_order_fill = time_amr::prepare_fifth_order_fill_patch(
      runtime, 0, std::as_const(parent.fab(0)).view(), child.fab(0).view(), child.box(0), mapping);
  EXPECT_EQ(fifth_order_fill.kind(),
            transfer::TransferKind::FifthOrderCoarseFineGhostInterpolation);
  time_amr::execute_prepared_transfer(fifth_order_fill);
  EXPECT_LT(pops::for_each_cell_reduce_max(
                child.box(0), ChildError<Dim>{std::as_const(child.fab(0)).view(), ratio, mapping}),
            pops::Real(3e-13));

  pops::Box<Dim> coarse_nodes = parent.box(0);
  pops::Box<Dim> fine_nodes = child.box(0);
  for (int axis = 0; axis < Dim; ++axis) {
    ++coarse_nodes.hi[axis];
    ++fine_nodes.hi[axis];
  }
  pops::Fab<Dim> parent_nodes(coarse_nodes, 1);
  pops::Fab<Dim> child_nodes(fine_nodes, 1);
  pops::for_each_cell(coarse_nodes,
                      FillNodeSource<Dim>{parent_nodes.view(), mapping.coarse_origin});
  const auto node_transfer = time_amr::prepare_node_multilinear(
      runtime, 0, std::as_const(parent_nodes).view(), child_nodes.view(), fine_nodes, mapping);
  EXPECT_EQ(node_transfer.kind(), transfer::TransferKind::NodeMultilinearProlongation);
  time_amr::execute_prepared_transfer(node_transfer);
  EXPECT_LT(pops::for_each_cell_reduce_max(
                fine_nodes, NodeError<Dim>{std::as_const(child_nodes).view(), ratio, mapping}),
            pops::Real(3e-13));

  const pops::Box<Dim> parent_face_source = parent.fab(0).grown_box();
  std::vector<pops::Fab<Dim>> parent_faces;
  std::vector<pops::Fab<Dim>> child_faces;
  parent_faces.reserve(Dim);
  child_faces.reserve(Dim);
  std::array<pops::FieldView<const pops::Real, Dim>, Dim> parent_face_views{};
  std::array<pops::FieldView<pops::Real, Dim>, Dim> child_face_views{};
  for (int normal_axis = 0; normal_axis < Dim; ++normal_axis) {
    pops::Box<Dim> fine_faces = child.box(0);
    ++fine_faces.hi[normal_axis];
    parent_faces.emplace_back(parent_face_source, 1);
    child_faces.emplace_back(fine_faces, 1);
    pops::for_each_cell(
        parent_face_source,
        FillFaceSource<Dim>{parent_faces.back().view(), mapping.coarse_origin, normal_axis});
  }
  for (int axis = 0; axis < Dim; ++axis) {
    parent_face_views[axis] = std::as_const(parent_faces[static_cast<std::size_t>(axis)]).view();
    child_face_views[axis] = child_faces[static_cast<std::size_t>(axis)].view();
  }
  const auto face_transfer = time_amr::prepare_divergence_preserving_faces(
      runtime, 0, parent_face_views, child_face_views, child.box(0), mapping);
  EXPECT_EQ(face_transfer.kind(), transfer::TransferKind::DivergencePreservingFaceProlongation);
  time_amr::execute_prepared_transfer(face_transfer);
  std::array<pops::FieldView<const pops::Real, Dim>, Dim> fine_face_views{};
  for (int axis = 0; axis < Dim; ++axis)
    fine_face_views[axis] = std::as_const(child_faces[static_cast<std::size_t>(axis)]).view();
  EXPECT_LT(pops::for_each_cell_reduce_max(
                child.box(0),
                FaceDivergenceError<Dim>{parent_face_views, fine_face_views, ratio, mapping}),
            pops::Real(3e-12));

  pops::Fab<Dim> older(parent.box(0), 1);
  pops::Fab<Dim> newer(parent.box(0), 1);
  pops::Fab<Dim> temporal_candidate(parent.box(0), 1);
  pops::for_each_cell(parent.box(0), FillTemporalSources<Dim>{older.view(), newer.view()});
  const std::string spatial_contract(runtime.spatial_contract());
  const transfer::QualifiedTemporalState older_state{
      "test/state", spatial_contract, runtime.topology_epoch(),
      runtime.materialization_generation(),
      pops::amr::ClockStamp{0, 5, pops::amr::Rational(0, 1), 1.0}};
  const transfer::QualifiedTemporalState newer_state{
      "test/state", spatial_contract, runtime.topology_epoch(),
      runtime.materialization_generation(),
      pops::amr::ClockStamp{0, 5, pops::amr::Rational(1, 1), 3.0}};
  const transfer::QualifiedTemporalState target_state{
      "test/state", spatial_contract, runtime.topology_epoch(),
      runtime.materialization_generation(),
      pops::amr::ClockStamp{0, 5, pops::amr::Rational(1, 4), 1.5}};
  const auto temporal = time_amr::prepare_linear_time_interpolation(
      runtime, 0, std::as_const(older).view(), std::as_const(newer).view(),
      temporal_candidate.view(), parent.box(0), older_state, newer_state, target_state);
  EXPECT_EQ(temporal.refinement_ratio(), ratio);
  time_amr::execute_prepared_transfer(temporal);
  EXPECT_LT(pops::for_each_cell_reduce_max(
                parent.box(0),
                TemporalError<Dim>{std::as_const(older).view(), std::as_const(newer).view(),
                                   std::as_const(temporal_candidate).view()}),
            pops::Real(2e-13));

  EXPECT_THROW(
      (void)time_amr::prepare_linear_prolongation(runtime, 1, std::as_const(parent.fab(0)).view(),
                                                  child.fab(0).view(), child.box(0), mapping),
      std::invalid_argument);
  EXPECT_THROW(
      (void)time_amr::prepare_average_down(runtime, 0, std::as_const(child.fab(0)).view(),
                                           restricted.fab(0).view(), parent.box(0), mapping),
      std::invalid_argument);
}

TEST(test_amr_transfer_properties,
     LiveHierarchyPreparesConservativeTransfersWithExactRankInOneTwoAndThreeDimensions) {
  prove_live_runtime_transfers<1>();
  prove_live_runtime_transfers<2>();
  prove_live_runtime_transfers<3>();
}

}  // namespace
