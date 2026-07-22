#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "load_balance_test_authority.hpp"

#include <pops/coupling/amr/amr_coupler_mp.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr/bootstrap_transfer_builtins.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/builders/factory/model_factory.hpp>
#include <pops/runtime/config/model_spec.hpp>

#include <pops/mesh/layout/refinement.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;
using namespace pops::runtime::amr;

namespace {

MultiFab field(const Box2D& box, int ghosts = 0) {
  return MultiFab(BoxArray(std::vector<Box2D>{box}), DistributionMapping(1, n_ranks()), 1, ghosts);
}

SpatialTransferContext context(const Box2D& coarse, int coarse_level = 0, int fine_level = 1) {
  const Box2D fine_domain = coarse.refine(2);
  return SpatialTransferContext{
      coarse_level, fine_level, 1,
      IndexTransform{{coarse.lo[0], coarse.lo[1]}, {fine_domain.lo[0], fine_domain.lo[1]}, {2, 2}},
      coarse, fine_domain, true};
}

ModelSpec exb_model() {
  ModelSpec spec;
  spec.transport = "exb";
  spec.source = "none";
  spec.elliptic = "charge";
  spec.q = 1.0;
  spec.B0 = 1.0;
  return spec;
}

constexpr Real kPreparedBoundarySentinel = Real(37.25);

Real integer_power(Real value, int exponent) {
  Real result = Real(1);
  for (int power = 0; power < exponent; ++power)
    result *= value;
  return result;
}

Real monomial_cell_average(Real lower, Real upper, int degree) {
  return (integer_power(upper, degree + 1) - integer_power(lower, degree + 1)) /
         (Real(degree + 1) * (upper - lower));
}

Real degree_four_cell_average(Real x_lower, Real x_upper, Real y_lower, Real y_upper) {
  const Real x2 = monomial_cell_average(x_lower, x_upper, 2);
  const Real x4 = monomial_cell_average(x_lower, x_upper, 4);
  const Real y3 = monomial_cell_average(y_lower, y_upper, 3);
  const Real y4 = monomial_cell_average(y_lower, y_upper, 4);
  return Real(1.25) + Real(0.02) * x4 - Real(0.015) * x2 * y3 + Real(0.01) * y4;
}

Real coarse_polynomial_average(const Box2D& coarse_domain, int i, int j) {
  const Real x = Real(i - coarse_domain.lo[0]);
  const Real y = Real(j - coarse_domain.lo[1]);
  return degree_four_cell_average(x, x + Real(1), y, y + Real(1));
}

Real fine_polynomial_average(const Box2D& fine_domain, int i, int j) {
  const Real x = Real(0.5) * Real(i - fine_domain.lo[0]);
  const Real y = Real(0.5) * Real(j - fine_domain.lo[1]);
  return degree_four_cell_average(x, x + Real(0.5), y, y + Real(0.5));
}

AmrRuntime bootstrap_runtime(int cells = 8, bool install_prepared_boundary = false) {
  AmrBuildParams params;
  params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  params.mesh.periodicity = Periodicity{true, true};
  params.mesh.n = cells;
  params.mesh.L = 1.0;
  params.mesh.regrid_every = 0;
  params.poisson.bc = BCRec{};
  const detail::SharedAmrLayout layout = detail::make_shared_amr_layout_levels(params, 1);
  std::vector<AmrRuntimeBlock> blocks;
  detail::dispatch_model(exb_model(), [&](auto model) {
    blocks.push_back(detail::dispatch_amr_block(
        model, "minmod", "rusanov", layout, "transport",
        std::vector<double>(static_cast<std::size_t>(cells) * cells, 1.0), true, 1.4, 1, false,
        false, 1));
    blocks.back().state_identity = "test://amr-transfer/bootstrap/transport/state/U";
  });
  if (install_prepared_boundary) {
    auto& block = blocks.front();
    const std::string state_identity = block.state_identity;
    block.boundary_plan = std::make_shared<PreparedBoundaryPlan>(
        "case::bootstrap::transport::boundary", 1,
        std::vector<BCRec>(static_cast<std::size_t>(block.ncomp), BCRec{}), std::vector<int>{},
        state_identity);
    const PreparedBoundaryPlan* const expected_plan = block.boundary_plan.get();
    block.boundary_field_registry = std::make_shared<GridContext::BoundaryFieldRegistryFactory>();
    block.level_rhs_core_at_point_prepared =
        [expected_plan](const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                        const MultiFab&, const Geometry&, MultiFab& rhs,
                        const PreparedGridBoundarySession& boundary) {
          if (boundary.resolved_plan() != expected_plan)
            throw std::logic_error("bootstrap boundary session retained the wrong prepared plan");
          rhs.set_val(kPreparedBoundarySentinel);
        };
    block.level_boundary_residual_at_point_prepared =
        [](const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, const MultiFab&,
           const Geometry&, MultiFab&, const PreparedGridBoundarySession&) {};
    block.level_rhs_at_point = [](const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                                  const MultiFab&, const Geometry&, MultiFab&) {
      throw std::runtime_error("legacy bootstrap boundary fallback was selected");
    };
  }
  AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc, std::move(blocks),
                     layout.base_per, layout.replicated_coarse, layout.wall);
  runtime.configure_hierarchy_capacity(
      {kAmrRefRatio, kAmrRefRatio},
      {::pops::amr::ParentChildClockRelation(
           0, 1, ::pops::amr::Rational(2, 1),
           ::pops::amr::RemainderPolicy::IntegralOnly),
       ::pops::amr::ParentChildClockRelation(
           1, 2, ::pops::amr::Rational(2, 1),
           ::pops::amr::RemainderPolicy::IntegralOnly)});
  runtime.set_block_transfer_authority(
      0, prepare_conservative_linear(), prepare_volume_average(),
      prepare_conservative_coarse_fine(), prepare_linear_time_interpolation(), kAmrRefRatio);
  if (install_prepared_boundary)
    runtime.install_boundary_storage_routes({});
  return runtime;
}

AmrRuntime rectangular_offset_runtime() {
  const Box2D domain{{3, 5}, {6, 7}};
  const BoxArray boxes(
      std::vector<Box2D>{Box2D{{3, 5}, {4, 7}}, Box2D{{5, 5}, {6, 7}}});
  const DistributionMapping distribution(boxes.size(), n_ranks());
  const Geometry geometry{domain, Real(0), Real(4), Real(-1), Real(2)};
  const auto load_balance = test::prepare_test_space_filling_curve_load_balance();
  AmrHierarchyLayout hierarchy{
      {boxes}, {distribution}, {Real(1)}, {Real(1)}, {}, load_balance};

  MultiFab state(boxes, distribution, 2, 1);
  state.set_val(Real(0));
  auto levels = std::make_shared<std::vector<AmrLevelMP>>();
  levels->push_back(AmrLevelMP{std::move(state), nullptr, Real(1), Real(1)});
  AmrRuntimeBlock block;
  block.name = "rectangular-offset";
  block.ncomp = 2;
  block.aux_ncomp = 4;
  block.levels = std::move(levels);
  std::vector<AmrRuntimeBlock> blocks;
  blocks.push_back(std::move(block));
  return AmrRuntime(geometry, std::move(hierarchy), BCRec{}, std::move(blocks),
                    Periodicity{true, true}, true);
}

}  // namespace

#if defined(POPS_HAS_KOKKOS)
class KokkosEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { guard_.emplace(); }
  void TearDown() override { guard_.reset(); }

 private:
  std::optional<Kokkos::ScopeGuard> guard_;
};

::testing::Environment* const kKokkosEnv =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
#endif

TEST(test_amr_transfer_properties, CellConservationRestrictionAndNonzeroOrigin) {
  const Box2D coarse_box{{3, 5}, {6, 8}};
  const Box2D fine_box = coarse_box.refine(2);
  MultiFab coarse = field(coarse_box, 1);
  MultiFab fine = field(fine_box, 1);
  MultiFab restricted = field(coarse_box);
  Array4 c = coarse.fab(0).array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i)
      c(i, j, 0) = 2.0 + 0.4 * i - 0.3 * j;

  const auto transform = context(coarse_box);
  detail::coupler_conservative_linear_to_fine_mb(coarse, fine, transform.logical_coarse_domain,
                                                 transform.logical_fine_domain,
                                                 transform.index.coarse_origin,
                                                 transform.index.fine_origin,
                                                 transform.index.refinement_ratio);
  average_down(fine, restricted, 2);
  const ConstArray4 r = restricted.fab(0).const_array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i)
      EXPECT_NEAR(r(i, j, 0), c(i, j, 0), 1e-14);
}

TEST(test_amr_transfer_properties,
     FifthOrderCoarseFineReproducesDegreeFourCellAveragesAndConservesEveryParent) {
  const Box2D coarse_box{{3, 5}, {10, 11}};
  const Box2D fine_box = coarse_box.refine(kAmrRefRatio);
  MultiFab coarse = field(coarse_box);
  MultiFab fine = field(fine_box);
  Array4 parent = coarse.fab(0).array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i)
      parent(i, j, 0) = coarse_polynomial_average(coarse_box, i, j);

  detail::coupler_conservative_polynomial5_to_fine_mb(
      coarse, fine, coarse_box, fine_box, {coarse_box.lo[0], coarse_box.lo[1]},
      {fine_box.lo[0], fine_box.lo[1]}, {kAmrRefRatio, kAmrRefRatio},
      /*replicated_parent=*/true, Periodicity{false, false});

  const ConstArray4 child = fine.fab(0).const_array();
  for (int j = fine_box.lo[1]; j <= fine_box.hi[1]; ++j)
    for (int i = fine_box.lo[0]; i <= fine_box.hi[0]; ++i)
      EXPECT_NEAR(child(i, j, 0), fine_polynomial_average(fine_box, i, j), 2e-11);
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i) {
      const int fi = fine_box.lo[0] + kAmrRefRatio * (i - coarse_box.lo[0]);
      const int fj = fine_box.lo[1] + kAmrRefRatio * (j - coarse_box.lo[1]);
      const Real average = Real(0.25) *
                           (child(fi, fj, 0) + child(fi + 1, fj, 0) +
                            child(fi, fj + 1, 0) + child(fi + 1, fj + 1, 0));
      EXPECT_NEAR(average, parent(i, j, 0), 2e-12);
    }
}

TEST(test_amr_transfer_properties,
     FifthOrderPeriodicCoarseFineConservesBoundaryParentsForArbitraryData) {
  const Box2D coarse_box{{-3, 4}, {4, 10}};
  const Box2D fine_box = coarse_box.refine(kAmrRefRatio);
  MultiFab coarse = field(coarse_box);
  MultiFab fine = field(fine_box);
  Array4 parent = coarse.fab(0).array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i)
      parent(i, j, 0) = Real(2) + Real(0.13) * i * j + std::sin(Real(0.7) * i) -
                        std::cos(Real(0.4) * j);

  detail::coupler_conservative_polynomial5_to_fine_mb(
      coarse, fine, coarse_box, fine_box, {coarse_box.lo[0], coarse_box.lo[1]},
      {fine_box.lo[0], fine_box.lo[1]}, {kAmrRefRatio, kAmrRefRatio},
      /*replicated_parent=*/true, Periodicity{true, true});

  const ConstArray4 child = fine.fab(0).const_array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i) {
      const int fi = fine_box.lo[0] + kAmrRefRatio * (i - coarse_box.lo[0]);
      const int fj = fine_box.lo[1] + kAmrRefRatio * (j - coarse_box.lo[1]);
      const Real average = Real(0.25) *
                           (child(fi, fj, 0) + child(fi + 1, fj, 0) +
                            child(fi, fj + 1, 0) + child(fi + 1, fj + 1, 0));
      EXPECT_NEAR(average, parent(i, j, 0), 2e-12);
    }
}

TEST(test_amr_transfer_properties,
     FifthOrderFillPatchRejectsSmallDomainsAndPreservesValidCells) {
  const auto high_order = std::make_shared<const PreparedCoarseFineOperator>(
      prepare_polynomial5_coarse_fine_operator());
  {
    const Box2D small_parent{{0, 0}, {3, 5}};
    MultiFab parent = field(small_parent);
    MultiFab fine = field(small_parent.refine(kAmrRefRatio));
    EXPECT_THROW(
        detail::PreparedConservativeCellTransferWorkspace::prepare(
            parent, fine, small_parent, small_parent.refine(kAmrRefRatio),
            /*replicated_parent=*/true, detail::ConservativeCellFillRegion::Valid,
            Periodicity{}, 0, CommunicatorView{}, high_order),
        std::invalid_argument);
  }

  const Box2D coarse_box{{3, 5}, {10, 12}};
  const Box2D fine_domain = coarse_box.refine(kAmrRefRatio);
  const Box2D fine_patch{{10, 14}, {15, 19}};
  MultiFab old_parent = field(coarse_box);
  MultiFab new_parent = field(coarse_box);
  MultiFab fine = field(fine_patch, 3);
  Array4 old_values = old_parent.fab(0).array();
  Array4 new_values = new_parent.fab(0).array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i) {
      old_values(i, j, 0) = coarse_polynomial_average(coarse_box, i, j);
      new_values(i, j, 0) = old_values(i, j, 0) + Real(4);
    }
  fine.set_val(Real(-17));
  auto workspace = PreparedFillPatchWorkspace::prepare(
      fine, old_parent, new_parent, coarse_box, /*replicated_parent=*/true,
      Periodicity{false, false}, 19, CommunicatorView{}, high_order);
  ASSERT_EQ(workspace.prepared_operator().get(), high_order.get());
  workspace.publish_prepared(fine, Real(0.25));

  const ConstArray4 values = fine.fab(0).const_array();
  const int ghost_i = fine_patch.lo[0] - 1;
  const int ghost_j = fine_patch.lo[1];
  EXPECT_NEAR(values(ghost_i, ghost_j, 0),
              fine_polynomial_average(fine_domain, ghost_i, ghost_j) + Real(1), 2e-11);
  EXPECT_DOUBLE_EQ(values(fine_patch.lo[0], fine_patch.lo[1], 0), Real(-17));
}

TEST(test_amr_transfer_properties, PreparedCarrierRejectsUnrepresentableExternalReach) {
  const Box2D coarse_box{{0, 0}, {7, 7}};
  const Box2D fine_box = coarse_box.refine(kAmrRefRatio);
  MultiFab parent = field(coarse_box);
  MultiFab fine = field(fine_box, 1);
  auto excessive = std::make_shared<PreparedCoarseFineOperator>(
      prepare_limited_linear_coarse_fine_operator());
  excessive->parent_reach_x = std::numeric_limits<int>::max();

  EXPECT_THROW(
      detail::PreparedConservativeCellTransferWorkspace::prepare(
          parent, fine, coarse_box, fine_box, /*replicated_parent=*/true,
          detail::ConservativeCellFillRegion::Ghost, Periodicity{}, 31,
          CommunicatorView{}, excessive),
      std::invalid_argument);
  EXPECT_THROW(
      PreparedFillPatchWorkspace::prepare(
          fine, parent, parent, coarse_box, /*replicated_parent=*/true,
          Periodicity{}, 31, CommunicatorView{}, excessive),
      std::invalid_argument);
}

TEST(test_amr_transfer_properties, CoupledFaceProlongationCommutesWithDivergence) {
  const Box2D cell_box{{3, 5}, {6, 8}};
  const Box2D coarse_x{{3, 5}, {7, 8}};
  const Box2D coarse_y{{3, 5}, {6, 9}};
  const Box2D fine_x{{6, 10}, {14, 17}};
  const Box2D fine_y{{6, 10}, {13, 18}};
  MultiFab cx = field(coarse_x, 1), cy = field(coarse_y, 1);
  MultiFab fx = field(fine_x, 1), fy = field(fine_y, 1);
  Array4 x = cx.fab(0).array(), y = cy.fab(0).array();
  for (int j = coarse_x.lo[1]; j <= coarse_x.hi[1]; ++j)
    for (int i = coarse_x.lo[0]; i <= coarse_x.hi[0]; ++i)
      x(i, j, 0) = 0.2 * i * i + 0.1 * j;
  for (int j = coarse_y.lo[1]; j <= coarse_y.hi[1]; ++j)
    for (int i = coarse_y.lo[0]; i <= coarse_y.hi[0]; ++i)
      y(i, j, 0) = -0.15 * j * j + 0.05 * i;

  const auto transform = context(cell_box);
  detail::bootstrap_prolong_face_vector(cx, cy, fx, fy, transform);
  const ConstArray4 qx = fx.fab(0).const_array(), qy = fy.fab(0).const_array();
  for (int j = cell_box.lo[1]; j <= cell_box.hi[1]; ++j)
    for (int i = cell_box.lo[0]; i <= cell_box.hi[0]; ++i) {
      const double coarse_div = x(i + 1, j, 0) - x(i, j, 0) + y(i, j + 1, 0) - y(i, j, 0);
      for (int child_y = 0; child_y < 2; ++child_y)
        for (int child_x = 0; child_x < 2; ++child_x) {
          const int fi = 2 * i + child_x, fj = 2 * j + child_y;
          const double fine_div =
              2.0 * (qx(fi + 1, fj, 0) - qx(fi, fj, 0) + qy(fi, fj + 1, 0) - qy(fi, fj, 0));
          EXPECT_NEAR(fine_div, coarse_div, 1e-14);
        }
    }
}

TEST(test_amr_transfer_properties, NodeBilinearIsSecondOrderAtNonzeroOrigin) {
  const Box2D coarse_nodes{{3, 5}, {7, 9}};
  const Box2D fine_nodes{{6, 10}, {14, 18}};
  MultiFab coarse = field(coarse_nodes, 1), fine = field(fine_nodes, 1);
  Array4 c = coarse.fab(0).array();
  for (int j = coarse_nodes.lo[1]; j <= coarse_nodes.hi[1]; ++j)
    for (int i = coarse_nodes.lo[0]; i <= coarse_nodes.hi[0]; ++i)
      c(i, j, 0) = 1.5 + 0.7 * i - 0.2 * j;
  detail::bootstrap_prolong_staggered(coarse, fine, TransferCentering::Node,
                                      context(Box2D{{3, 5}, {6, 8}}));
  const ConstArray4 f = fine.fab(0).const_array();
  for (int j = fine_nodes.lo[1]; j <= fine_nodes.hi[1]; ++j)
    for (int i = fine_nodes.lo[0]; i <= fine_nodes.hi[0]; ++i) {
      const double expected = 1.5 + 0.7 * (0.5 * i) - 0.2 * (0.5 * j);
      EXPECT_NEAR(f(i, j, 0), expected, 1e-14);
    }
}

TEST(test_amr_transfer_properties, RuntimeFlatStateAuxPotentialAndHistoryUseExactRectangle) {
  AmrRuntime runtime = rectangular_offset_runtime();
  const Box2D domain{{3, 5}, {6, 7}};
  const std::size_t nx = static_cast<std::size_t>(domain.nx());
  const std::size_t cells = nx * static_cast<std::size_t>(domain.ny());

  MultiFab& state = runtime.level_state(0, 0);
  for (int local = 0; local < state.local_size(); ++local) {
    Array4 values = state.fab(local).array();
    const Box2D valid = state.box(local);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        for (int component = 0; component < state.ncomp(); ++component)
          values(i, j, component) =
              Real(100 * component + 10 * (j - domain.lo[1]) + (i - domain.lo[0]));
  }
  const std::vector<double> flat_state = runtime.block_level_state(0, 0);
  ASSERT_EQ(flat_state.size(), 2 * cells);
  for (int component = 0; component < 2; ++component)
    for (int j = domain.lo[1]; j <= domain.hi[1]; ++j)
      for (int i = domain.lo[0]; i <= domain.hi[0]; ++i)
        EXPECT_DOUBLE_EQ(flat_state[static_cast<std::size_t>(component) * cells +
                                    static_cast<std::size_t>(j - domain.lo[1]) * nx +
                                    static_cast<std::size_t>(i - domain.lo[0])],
                         100 * component + 10 * (j - domain.lo[1]) + (i - domain.lo[0]));

  state.set_val(Real(-1));
  runtime.set_block_level_state(0, 0, flat_state);
  EXPECT_EQ(runtime.block_level_state(0, 0), flat_state);

  std::vector<double> potential(cells);
  for (std::size_t cell = 0; cell < cells; ++cell)
    potential[cell] = 2.5 + static_cast<double>(cell);
  runtime.set_level_potential(0, potential);
  EXPECT_EQ(runtime.level_potential(0), potential);

  std::vector<Real> named(cells);
  for (std::size_t cell = 0; cell < cells; ++cell)
    named[cell] = Real(40 + cell);
  runtime.set_static_aux_component(3, named);
  const std::vector<double> flat_aux = runtime.level_aux_flat(0);
  ASSERT_EQ(flat_aux.size(), 4 * cells);
  for (std::size_t cell = 0; cell < cells; ++cell)
    EXPECT_DOUBLE_EQ(flat_aux[3 * cells + cell], named[cell]);
  runtime.set_level_aux_flat(0, flat_aux);
  EXPECT_EQ(runtime.level_aux_flat(0), flat_aux);

  detail::AmrHistoryOps::register_history(runtime, 0, "rect-history", 1);
  detail::AmrHistoryOps::store_history(runtime, "rect-history", 0, state, Real(0.25));
  EXPECT_EQ(detail::AmrHistoryOps::global(runtime, "rect-history", 0, false), flat_state);
  std::vector<double> changed = flat_state;
  for (double& value : changed)
    value += 7.0;
  detail::AmrHistoryOps::restore(runtime, "rect-history", 0, changed);
  EXPECT_EQ(detail::AmrHistoryOps::global(runtime, "rect-history", 0, false), changed);
  changed.push_back(0.0);
  EXPECT_THROW(detail::AmrHistoryOps::restore(runtime, "rect-history", 0, changed),
               std::runtime_error);
}

TEST(test_amr_transfer_properties, CoarseFineIsSecondOrderAndTemporalUsesIndependentContext) {
  const Box2D coarse_box{{3, 5}, {6, 8}};
  const Box2D fine_patch{{8, 12}, {11, 15}};
  MultiFab parent = field(coarse_box), fine = field(fine_patch, 1);
  Array4 p = parent.fab(0).array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i)
      p(i, j, 0) = 10.0 * i + j;
  fine.set_val(-7.0);
  auto spatial = prepare_conservative_coarse_fine();
  spatial.coarse_fine(parent, fine, context(coarse_box));
  const ConstArray4 filled = fine.fab(0).const_array();
  // The left ghost maps to parent (3,6). The x slope is one-sided at the domain edge, while the
  // exact linear y slope contributes -1/4 on the lower child: this distinguishes order two from
  // the former parent-cell injection.
  EXPECT_DOUBLE_EQ(filled(fine_patch.lo[0] - 1, fine_patch.lo[1], 0), p(3, 6, 0) - 0.25);
  // The bottom ghost maps to parent (4,5); its exact interior x slope contributes +10/4.
  EXPECT_DOUBLE_EQ(filled(fine_patch.lo[0] + 1, fine_patch.lo[1] - 1, 0), p(4, 5, 0) + 2.5);
  EXPECT_DOUBLE_EQ(filled(fine_patch.lo[0], fine_patch.lo[1], 0), -7.0);

  MultiFab old_value = field(coarse_box), new_value = field(coarse_box);
  MultiFab destination = field(coarse_box);
  Array4 old_data = old_value.fab(0).array(), new_data = new_value.fab(0).array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i) {
      old_data(i, j, 0) = i + j;
      new_data(i, j, 0) = i + j + 4.0;
    }
  auto temporal = prepare_linear_time_interpolation();
  temporal.temporal(old_value, new_value, destination,
                    TemporalTransferContext{{10, 1.0}, {11, 3.0}, {10, 1.5}});
  const ConstArray4 interpolated = destination.fab(0).const_array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i)
      EXPECT_DOUBLE_EQ(interpolated(i, j, 0), i + j + 1.0);
  EXPECT_THROW(temporal.temporal(old_value, new_value, destination,
                                 TemporalTransferContext{{10, 1.0}, {10, 3.0}, {10, 1.5}}),
               std::runtime_error);
  EXPECT_THROW(temporal.temporal(old_value, old_value, destination,
                                 TemporalTransferContext{{10, 1.0}, {11, 3.0}, {10, 1.5}}),
               std::runtime_error);
}

TEST(test_amr_transfer_properties, NativeSubcyclingGhostFillInterpolatesTimeThenLinearSpace) {
  const Box2D coarse_box{{3, 5}, {6, 8}};
  const Box2D fine_patch{{8, 12}, {11, 15}};
  MultiFab old_parent = field(coarse_box), new_parent = field(coarse_box);
  MultiFab fine = field(fine_patch, 1);
  Array4 old_values = old_parent.fab(0).array();
  Array4 new_values = new_parent.fab(0).array();
  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i) {
      old_values(i, j, 0) = 10.0 * i + j;
      new_values(i, j, 0) = old_values(i, j, 0) + 4.0;
    }
  fine.set_val(-7.0);

  auto workspace = PreparedFillPatchWorkspace::prepare(
      fine, old_parent, new_parent, coarse_box, /*replicated_parent=*/true,
      Periodicity{false, false}, 23, CommunicatorView{});
  workspace.publish_prepared(fine, Real(0.25));
  const ConstArray4 filled = fine.fab(0).const_array();
  EXPECT_DOUBLE_EQ(filled(fine_patch.lo[0] - 1, fine_patch.lo[1], 0),
                   old_values(3, 6, 0) + 1.0 - 0.25);
  EXPECT_DOUBLE_EQ(filled(fine_patch.lo[0] + 1, fine_patch.lo[1] - 1, 0),
                   old_values(4, 5, 0) + 1.0 + 2.5);
  EXPECT_DOUBLE_EQ(filled(fine_patch.lo[0], fine_patch.lo[1], 0), -7.0);

  for (int j = coarse_box.lo[1]; j <= coarse_box.hi[1]; ++j)
    for (int i = coarse_box.lo[0]; i <= coarse_box.hi[0]; ++i)
      new_values(i, j, 0) = old_values(i, j, 0) + 8.0;
  fine.set_val(-7.0);
  workspace.apply(fine, old_parent, new_parent, Real(0.5), Real(0), 0, 23,
                  CommunicatorView{});
  EXPECT_DOUBLE_EQ(fine.fab(0).const_array()(fine_patch.lo[0] - 1, fine_patch.lo[1], 0),
                   old_values(3, 6, 0) + 4.0 - 0.25);
  EXPECT_THROW(workspace.apply(fine, old_parent, new_parent, Real(0.5), Real(0), 0, 24,
                               CommunicatorView{}),
               std::invalid_argument);
}

TEST(test_amr_transfer_properties, AnalyticEveryLevelCacheEpochAndL0L1L2Rollback) {
  AmrRuntime runtime = bootstrap_runtime();
  ASSERT_EQ(runtime.nlev(), 1);
  const std::vector<double> baseline = runtime.block_level_state(0, 0);
  test::install_prepared_threshold_union(
      runtime, {{0, 0, Real(0.5)}}, "test.tag-provider::component-above");

  for (int failure_level = 0; failure_level <= 2; ++failure_level) {
    runtime.begin_bootstrap_plan();
    EXPECT_GT(runtime.fill_bootstrap_block_constant(0, 0, {7.0}), 0);
    for (int level = 1; level <= failure_level; ++level) {
      runtime.bootstrap_next_level(2);
      EXPECT_GT(runtime.fill_bootstrap_block_constant(0, level, {7.0}), 0);
      const std::vector<double> materialized = runtime.block_level_state(0, level);
      for (double value : materialized)
        if (value != 0.0)
          EXPECT_DOUBLE_EQ(value, 7.0);
    }
    EXPECT_EQ(runtime.nlev(), failure_level + 1);
    runtime.rollback_bootstrap_level();
    EXPECT_EQ(runtime.nlev(), 1);
    EXPECT_EQ(runtime.block_level_state(0, 0), baseline);
  }

  runtime.begin_bootstrap_plan();
  EXPECT_EQ(runtime.invalidate_bootstrap_cache("patch-topology"), 1u);
  EXPECT_TRUE(runtime.rebuild_bootstrap_topology_cache("patch-topology", 0).valid);
  EXPECT_EQ(runtime.invalidate_bootstrap_cache("patch-topology"), 2u);
  const auto& cache = runtime.rebuild_bootstrap_topology_cache("patch-topology", 0);
  EXPECT_TRUE(cache.valid);
  EXPECT_EQ(cache.epoch, 2u);
  runtime.rollback_bootstrap_level();
  EXPECT_THROW(runtime.bootstrap_cache("patch-topology"), std::runtime_error);
}

TEST(test_amr_transfer_properties, BootstrapMaterializesPreparedBoundarySessionAtNewLevel) {
  AmrRuntime runtime = bootstrap_runtime(8, true);
  const std::vector<double> baseline = runtime.block_level_state(0, 0);
  test::install_prepared_threshold_union(
      runtime, {{0, 0, Real(0.5)}}, "test.tag-provider::component-above");

  runtime.begin_bootstrap_plan();
  runtime.bootstrap_next_level(2);
  ASSERT_EQ(runtime.nlev(), 2);

  MultiFab& fine = runtime.level_state(0, 1);
  MultiFab fine_rhs(fine.box_array(), fine.dmap(), fine.ncomp(), 0);
  const runtime::multiblock::BoundaryEvaluationPoint fine_point{
      "clock.bootstrap-fine", 0, 1, 0, 0, amr::Rational(0, 1), 0.1, 0.0};
  EXPECT_NO_THROW(runtime.level_rhs_into_at(0, 1, fine_point, fine, fine_rhs));
  for (int local = 0; local < fine_rhs.local_size(); ++local) {
    const ConstArray4 values = fine_rhs.fab(local).const_array();
    const Box2D valid = fine_rhs.box(local);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        for (int component = 0; component < fine_rhs.ncomp(); ++component)
          EXPECT_DOUBLE_EQ(values(i, j, component), kPreparedBoundarySentinel);
  }

  runtime.rollback_bootstrap_level();
  ASSERT_EQ(runtime.nlev(), 1);
  EXPECT_EQ(runtime.block_level_state(0, 0), baseline);

  MultiFab& coarse = runtime.level_state(0, 0);
  MultiFab coarse_rhs(coarse.box_array(), coarse.dmap(), coarse.ncomp(), 0);
  const runtime::multiblock::BoundaryEvaluationPoint coarse_point{
      "clock.bootstrap-rollback", 0, 0, 0, 0, amr::Rational(0, 1), 0.1, 0.0};
  EXPECT_NO_THROW(runtime.level_rhs_into_at(0, 0, coarse_point, coarse, coarse_rhs));
  if (coarse_rhs.local_size() > 0)
    EXPECT_DOUBLE_EQ(
        coarse_rhs.fab(0).const_array()(coarse_rhs.box(0).lo[0], coarse_rhs.box(0).lo[1], 0),
        kPreparedBoundarySentinel);
}
