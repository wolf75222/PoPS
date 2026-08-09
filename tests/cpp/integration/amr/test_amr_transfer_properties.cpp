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
