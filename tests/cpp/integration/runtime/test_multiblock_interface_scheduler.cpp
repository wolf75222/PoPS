#include <gtest/gtest.h>

#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/physics/bricks/bricks.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/system/system_block_store.hpp>

#include "amr_transfer_test_authority.hpp"
#include "amr_tagging_test_authority.hpp"

#include <array>
#include <cstdint>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;
using namespace pops::runtime::multiblock;

namespace {

void ensure_runtime() {
#if defined(POPS_HAS_KOKKOS)
  static Kokkos::ScopeGuard guard;
#endif
}

MultiFab make_field(const Box2D& box, int ncomp) {
  return MultiFab(BoxArray(std::vector<Box2D>{box}), DistributionMapping(1, n_ranks()), ncomp, 0);
}

MultiFab make_field(std::vector<Box2D> boxes, int ncomp) {
  const int count = static_cast<int>(boxes.size());
  return MultiFab(BoxArray(std::move(boxes)), DistributionMapping(count, 1), ncomp, 0);
}

void set_cell(MultiFab& field, int i, int j, int component, Real value) {
  for (int local = 0; local < field.local_size(); ++local)
    if (field.box(local).contains(i, j)) {
      field.fab(local).array()(i, j, component) = value;
      return;
    }
  throw std::out_of_range("test cell is absent from MultiFab");
}

Real get_cell(const MultiFab& field, int i, int j, int component) {
  for (int local = 0; local < field.local_size(); ++local)
    if (field.box(local).contains(i, j))
      return field.fab(local).const_array()(i, j, component);
  throw std::out_of_range("test cell is absent from MultiFab");
}

PopsExecutionContextV1 serial_interface_execution() {
  return {sizeof(PopsExecutionContextV1),
          1u,
          "test::execution-context",
          POPS_MEMORY_SPACE_HOST_V1,
          "pops.runtime-backend-manifest.v1:sha256:test",
          "host",
          POPS_SCALAR_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          0,
          "host::synchronous",
          0,
          0,
          "serial",
          "none"};
}

#if defined(POPS_HAS_MPI)
PopsExecutionContextV1 mpi_world_interface_execution() {
  comm_init();
  return {sizeof(PopsExecutionContextV1),
          1u,
          "test::mpi-execution-context",
          POPS_MEMORY_SPACE_HOST_V1,
          "pops.runtime-backend-manifest.v1:sha256:test-mpi",
          "host",
          POPS_SCALAR_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          0,
          "host::synchronous",
          static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD)),
          static_cast<std::int64_t>(MPI_Type_c2f(MPI_DOUBLE)),
          "MPI_COMM_WORLD",
          "MPI_DOUBLE"};
}
#endif

void authenticate_cell_average_trace(AxisAlignedInterface& route) {
  route.left_trace_projection_identity = route.identity + ".left-trace";
  route.right_trace_projection_identity = route.identity + ".right-trace";
  route.left_trace_provider_identity = "limiter.none";
  route.right_trace_provider_identity = "limiter.none";
  route.left_trace_operation = InterfaceTraceOperation::CellAverage;
  route.right_trace_operation = InterfaceTraceOperation::CellAverage;
  route.left_trace_required_depth = 1;
  route.right_trace_required_depth = 1;
}

AxisAlignedInterface heterogeneous_route() {
  AxisAlignedInterface route;
  route.identity = "left-right.shared_flux";
  route.left_block = 0;
  route.right_block = 1;
  route.level = 0;
  route.left_axis = InterfaceAxis::X;
  route.right_axis = InterfaceAxis::X;
  route.left_side = InterfaceSide::High;
  route.right_side = InterfaceSide::Low;
  route.tangential_orientation = TangentialOrientation::Reversed;
  route.right_component_for_left = {1, 0};
  route.affine_mapping_identity = "reverse-y-on-coincident-face";
  route.right_tangential_scale = Real(-1);
  route.right_tangential_offset = Real(3);
  authenticate_cell_average_trace(route);
  return route;
}

using ExBModel = CompositeModel<ExBVelocity, NoSource, ChargeDensity>;

ExBModel scalar_model() {
  return ExBModel{ExBVelocity{Real(1)}, NoSource{}, ChargeDensity{Real(0)}};
}

AxisAlignedInterface aligned_x_route(std::string identity) {
  AxisAlignedInterface route;
  route.identity = std::move(identity);
  route.left_block = 0;
  route.right_block = 1;
  route.left_axis = route.right_axis = InterfaceAxis::X;
  route.left_side = InterfaceSide::High;
  route.right_side = InterfaceSide::Low;
  route.right_component_for_left = {0};
  authenticate_cell_average_trace(route);
  return route;
}

template <std::size_t ConfiguredLevels>
AmrRuntime make_dynamic_interface_runtime(int cells, int active_levels,
                                          std::array<int, ConfiguredLevels>& evaluator_calls) {
  static_assert(ConfiguredLevels > 0);
  if (active_levels < 1 || active_levels > static_cast<int>(ConfiguredLevels))
    throw std::invalid_argument(
        "dynamic interface test requires one active configured-level prefix");
  AmrBuildParams params;
  params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  params.mesh.periodicity = Periodicity{true, true};
  params.mesh.n = cells;
  params.mesh.L = 1.0;
  params.mesh.regrid_every = 1;
  params.poisson.bc = BCRec{};
  detail::SharedAmrLayout layout = detail::make_shared_amr_layout_levels(params, active_levels);
  int cumulative_refinement = 1;
  for (int level = 1; level < active_levels; ++level) {
    cumulative_refinement *= kAmrRefRatio;
    layout.ba[static_cast<std::size_t>(level)] =
        BoxArray(std::vector<Box2D>{layout.geom.domain.refine(cumulative_refinement)});
    layout.dm[static_cast<std::size_t>(level)] =
        layout.load_balance->distribute(layout.ba[static_cast<std::size_t>(level)], n_ranks());
  }

  std::vector<AmrRuntimeBlock> blocks;
  for (const char* name : {"left", "right"}) {
    AmrRuntimeBlock block = detail::dispatch_amr_block(
        scalar_model(), "none", "rusanov", layout, name,
        std::vector<double>(static_cast<std::size_t>(cells) * cells, 1.0), true, 1.4, 1, false, 1);
    block.state_identity = std::string("test://dynamic-interface/block/") + name + "/state/U";
    const auto omit_local_interface = [](MultiFab&, const MultiFab&, const Geometry&, MultiFab& fx,
                                         MultiFab& fy, MultiFab& rhs) {
      fx.set_val(Real(0));
      fy.set_val(Real(0));
      rhs.set_val(Real(0));
    };
    block.level_flux_capture = omit_local_interface;
    block.level_flux_capture_neg_div = omit_local_interface;
    block.level_rhs_without_prepared_interfaces = [](const BoundaryEvaluationPoint&, MultiFab&,
                                                     const MultiFab&, const Geometry&,
                                                     MultiFab& rhs) { rhs.set_val(Real(0)); };
    block.level_neg_div_flux_without_prepared_interfaces =
        block.level_rhs_without_prepared_interfaces;
    blocks.push_back(std::move(block));
  }
  AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc, std::move(blocks),
                     layout.base_per, layout.replicated_coarse, layout.wall);
  test::install_second_order_amr_transfer_authorities(runtime, 2);
  std::vector<int> refinement_ratios;
  std::vector<amr::ParentChildClockRelation> relations;
  refinement_ratios.reserve(ConfiguredLevels - 1);
  relations.reserve(ConfiguredLevels - 1);
  for (std::size_t parent = 0; parent + 1 < ConfiguredLevels; ++parent) {
    refinement_ratios.push_back(kAmrRefRatio);
    relations.emplace_back(static_cast<int>(parent), static_cast<int>(parent + 1),
                           amr::Rational(2, 1), amr::RemainderPolicy::IntegralOnly);
  }
  if (active_levels < static_cast<int>(ConfiguredLevels))
    runtime.configure_hierarchy_capacity(std::move(refinement_ratios), std::move(relations));
  else
    runtime.set_parent_child_temporal_relations(std::move(relations));
  runtime.set_regrid(/*every=*/1, /*grow=*/0, /*margin=*/0);

  for (int level = 0; level < active_levels; ++level) {
    AxisAlignedInterface route = aligned_x_route("amr.dynamic.shared-flux");
    route.level = level;
    route.affine_mapping_identity = "periodic-x-translation";
    route.right_normal_translation = Real(1);
    runtime.install_level_interface_flux(
        level, route, serial_interface_execution(),
        [&evaluator_calls, level](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
          ++evaluator_calls[static_cast<std::size_t>(level)];
          for (int face = 0; face < batch.face_count; ++face)
            batch.shared_flux[face] = Real(level + face + 1);
        });
  }
  if (active_levels > 1)
    runtime.require_complete_active_level_interfaces();
  return runtime;
}

}  // namespace

TEST(test_multiblock_interface_scheduler,
     UniformExecutorRunsOneSharedFluxOnTwoHeterogeneousLayoutsWithoutDoubleCounting) {
  ensure_runtime();
  const Box2D left_box{{0, 0}, {3, 2}};     // 4 x 3
  const Box2D right_box{{10, 7}, {15, 9}};  // 6 x 3: distinct layout/index origin
  const Geometry left_geometry{left_box, Real(0), Real(2), Real(0), Real(3)};
  const Geometry right_geometry{right_box, Real(2), Real(5), Real(0), Real(3)};
  const BoundaryEvaluationPoint point{"clock.macro", 12, 0, 1, 3, amr::Rational(1, 2), 0.05, 0.625};

  SystemBlockStore store;
  int full_boundary_rhs_calls = 0;
  int interface_omitting_rhs_calls = 0;
  std::optional<BoundaryEvaluationPoint> residual_point;
  for (int block = 0; block < 2; ++block) {
    SystemBlockStore::BlockState state;
    state.name = block == 0 ? "left" : "right";
    state.U = make_field(block == 0 ? left_box : right_box, 2);
    state.ncomp = 2;
    state.rhs_into = [&full_boundary_rhs_calls](MultiFab&, MultiFab& rhs) {
      ++full_boundary_rhs_calls;
      rhs.set_val(Real(91));  // manufactured physical-BC flux: must never be retained/added
    };
    state.rhs_without_prepared_interfaces = [&interface_omitting_rhs_calls, &residual_point](
                                                const BoundaryEvaluationPoint& evaluation,
                                                MultiFab&, MultiFab& rhs) {
      ++interface_omitting_rhs_calls;
      residual_point = evaluation;
      rhs.set_val(Real(0));
    };
    state.rhs_flux_only_without_prepared_interfaces = state.rhs_without_prepared_interfaces;
    store.blocks.push_back(std::move(state));
  }

  store.blocks[0].U.set_val(Real(0));
  store.blocks[1].U.set_val(Real(0));
  Array4 left = store.blocks[0].U.fab(0).array();
  Array4 right = store.blocks[1].U.fab(0).array();
  for (int face = 0; face < 3; ++face) {
    left(left_box.hi[0], left_box.lo[1] + face, 0) = Real(2 + face);
    left(left_box.hi[0], left_box.lo[1] + face, 1) = Real(5 + face);
    const int mapped = 2 - face;
    right(right_box.lo[0], right_box.lo[1] + mapped, 1) = Real(10 + face);
    right(right_box.lo[0], right_box.lo[1] + mapped, 0) = Real(20 + face);
  }

  int evaluator_calls = 0;
  const Real* prepared_left_trace = nullptr;
  const Real* prepared_right_trace = nullptr;
  Real* prepared_shared_flux = nullptr;
  BoundaryEvaluationPoint observed;
  store.install_interface_flux(
      heterogeneous_route(), left_geometry, right_geometry, serial_interface_execution(),
      [&](const BoundaryEvaluationPoint& evaluation, const InterfaceFluxBatch& batch) {
        ++evaluator_calls;
        observed = evaluation;
        ASSERT_EQ(batch.face_count, 3);
        ASSERT_EQ(batch.component_count, 2);
        if (prepared_left_trace == nullptr) {
          prepared_left_trace = batch.left_state;
          prepared_right_trace = batch.right_state;
          prepared_shared_flux = batch.shared_flux;
        } else {
          EXPECT_EQ(batch.left_state, prepared_left_trace);
          EXPECT_EQ(batch.right_state, prepared_right_trace);
          EXPECT_EQ(batch.shared_flux, prepared_shared_flux);
        }
        for (int face = 0; face < batch.face_count; ++face)
          for (int component = 0; component < batch.component_count; ++component) {
            const std::size_t offset = static_cast<std::size_t>(face) * 2 + component;
            batch.shared_flux[offset] =
                Real(0.25) * (batch.left_state[offset] + batch.right_state[offset]);
          }
      });

  MultiFab left_rhs = make_field(left_box, 2);
  MultiFab right_rhs = make_field(right_box, 2);
  std::vector<MultiFab*> states{&store.blocks[0].U, &store.blocks[1].U};
  std::vector<MultiFab*> rhs{&left_rhs, &right_rhs};

  // The current pair scheduler owns Cartesian shared fluxes only.  An embedded-boundary Program
  // must fail before either local residual or pair evaluator runs; otherwise the scheduler could
  // scatter an unmasked flux back into an inactive cell after the local EB residual zeroed it.
  EXPECT_THROW(store.evaluate_rhs_with_interfaces(point, states, rhs, {}, GeometryMode::Staircase),
               std::runtime_error);
  EXPECT_EQ(evaluator_calls, 0);
  EXPECT_EQ(interface_omitting_rhs_calls, 0);

  store.evaluate_rhs_with_interfaces(point, states, rhs);

  EXPECT_EQ(evaluator_calls, 1);
  EXPECT_EQ(store.interface_evaluation_count("left-right.shared_flux", 0), 1u);
  EXPECT_EQ(observed, point);
  EXPECT_EQ(full_boundary_rhs_calls, 0) << "the physical-BC residual would double-count the face";
  EXPECT_EQ(interface_omitting_rhs_calls, 2);
  ASSERT_TRUE(residual_point.has_value());
  EXPECT_EQ(*residual_point, point)
      << "the boundary-aware residual sees the exact point before fill";

  const ConstArray4 left_result = left_rhs.fab(0).const_array();
  const ConstArray4 right_result = right_rhs.fab(0).const_array();
  for (int face = 0; face < 3; ++face) {
    const int mapped = 2 - face;
    for (int component = 0; component < 2; ++component) {
      const int right_component = component == 0 ? 1 : 0;
      const Real lhs = left_result(left_box.hi[0], left_box.lo[1] + face, component);
      const Real rhs_value =
          right_result(right_box.lo[0], right_box.lo[1] + mapped, right_component);
      EXPECT_NE(lhs, Real(0));
      EXPECT_EQ(lhs + rhs_value, Real(0)) << "the mapped pair must conserve the component exactly";
      EXPECT_NE(lhs, Real(91)) << "the old boundary residual was not replaced";
    }
  }

  // Replaying an exact prepared interface reuses its ABI buffers.  This guards against an
  // O(face_count*ncomp) allocation/copy returning to every residual stage.
  store.evaluate_rhs_with_interfaces(point, states, rhs);
  EXPECT_EQ(evaluator_calls, 2);
  EXPECT_EQ(store.interface_evaluation_count("left-right.shared_flux", 0), 2u);
  EXPECT_EQ(interface_omitting_rhs_calls, 4);
}

TEST(test_multiblock_interface_scheduler,
     SerialSchedulerEnumeratesEveryBoundaryPatchAcrossDifferentBoxDecompositions) {
  ensure_runtime();
  MultiFab left_state = make_field(
      {
          Box2D{{0, 0}, {1, 1}},
          Box2D{{2, 0}, {3, 1}},
          Box2D{{0, 2}, {1, 5}},
          Box2D{{2, 2}, {3, 5}},
      },
      1);
  MultiFab right_state = make_field(
      {
          Box2D{{10, 7}, {12, 9}},
          Box2D{{13, 7}, {15, 9}},
          Box2D{{10, 10}, {12, 12}},
          Box2D{{13, 10}, {15, 12}},
      },
      1);
  left_state.set_val(Real(0));
  right_state.set_val(Real(0));
  for (int face = 0; face < 6; ++face) {
    set_cell(left_state, 3, face, 0, Real(face + 1));
    set_cell(right_state, 10, 7 + face, 0, Real(11 + face));
  }
  MultiFab left_rhs(left_state.box_array(), left_state.dmap(), 1, 0);
  MultiFab right_rhs(right_state.box_array(), right_state.dmap(), 1, 0);

  AxisAlignedInterface route;
  route.identity = "serial.multibox.shared_flux";
  route.left_block = 0;
  route.right_block = 1;
  route.left_axis = route.right_axis = InterfaceAxis::X;
  route.left_side = InterfaceSide::High;
  route.right_side = InterfaceSide::Low;
  route.right_component_for_left = {0};
  authenticate_cell_average_trace(route);
  const Geometry left_geometry{left_state.box_array().bounding_box(), Real(0), Real(2), Real(0),
                               Real(6)};
  const Geometry right_geometry{right_state.box_array().bounding_box(), Real(2), Real(5), Real(0),
                                Real(6)};
  InterfaceFluxScheduler scheduler;
  int calls = 0;
  scheduler.install(route, left_state, left_geometry, right_state, right_geometry,
                    serial_interface_execution(),
                    [&](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
                      ++calls;
                      ASSERT_EQ(batch.face_count, 6);
                      for (int face = 0; face < batch.face_count; ++face) {
                        EXPECT_EQ(batch.left_state[face], Real(face + 1));
                        EXPECT_EQ(batch.right_state[face], Real(11 + face));
                        batch.shared_flux[face] = Real(face + 2);
                      }
                    });
  EXPECT_TRUE(scheduler.owns_face(0, 0, InterfaceAxis::X, InterfaceSide::High));
  EXPECT_TRUE(scheduler.owns_face(1, 0, InterfaceAxis::X, InterfaceSide::Low));
  EXPECT_FALSE(scheduler.owns_face(0, 0, InterfaceAxis::X, InterfaceSide::Low));
  EXPECT_FALSE(scheduler.owns_face(0, 0, InterfaceAxis::Y, InterfaceSide::High));
  EXPECT_FALSE(scheduler.owns_face(0, 1, InterfaceAxis::X, InterfaceSide::High));
  const BoundaryEvaluationPoint point{"clock.multibox", 1, 0, 0, 0, amr::Rational(0, 1), 0.1, 0.0};
  std::vector<MultiFab*> states{&left_state, &right_state};
  std::vector<MultiFab*> rhs{&left_rhs, &right_rhs};
  scheduler.apply(point, states, rhs);

  EXPECT_EQ(calls, 1);
  for (int face = 0; face < 6; ++face)
    EXPECT_EQ(get_cell(left_rhs, 3, face, 0) + get_cell(right_rhs, 10, 7 + face, 0), Real(0));
  scheduler.clear();
  EXPECT_FALSE(scheduler.owns_face(0, 0, InterfaceAxis::X, InterfaceSide::High));
}

TEST(test_multiblock_interface_scheduler,
     RematerializationRejectsPartialFaceWithoutMutatingAcceptedRegistry) {
  ensure_runtime();
  const Box2D domain{{0, 0}, {3, 3}};
  const Geometry geometry{domain, Real(0), Real(1), Real(0), Real(1)};
  MultiFab left_state = make_field(domain, 1);
  MultiFab right_state = make_field(domain, 1);
  left_state.set_val(Real(1));
  right_state.set_val(Real(2));

  AxisAlignedInterface route = aligned_x_route("dynamic.partial-face.shared-flux");
  route.affine_mapping_identity = "periodic-x-translation";
  route.right_normal_translation = Real(1);
  InterfaceFluxScheduler scheduler;
  int evaluator_calls = 0;
  scheduler.install(route, left_state, geometry, right_state, geometry,
                    serial_interface_execution(),
                    [&](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
                      ++evaluator_calls;
                      for (int face = 0; face < batch.face_count; ++face)
                        batch.shared_flux[face] = Real(face + 1);
                    });

  MultiFab partial_left = make_field(Box2D{{1, 1}, {2, 2}}, 1);
  MultiFab partial_right = make_field(Box2D{{1, 1}, {2, 2}}, 1);
  EXPECT_THROW(scheduler.rematerialized(
                   1,
                   [&](std::size_t block, int level) -> MultiFab& {
                     EXPECT_EQ(level, 0);
                     return block == 0 ? partial_left : partial_right;
                   },
                   [&](int level) {
                     EXPECT_EQ(level, 0);
                     return geometry;
                   }),
               std::invalid_argument);

  MultiFab left_rhs(left_state.box_array(), left_state.dmap(), 1, 0);
  MultiFab right_rhs(right_state.box_array(), right_state.dmap(), 1, 0);
  const BoundaryEvaluationPoint point{
      "clock.rematerialization-rollback", 1, 0, 0, 0, amr::Rational(0, 1), 0.1, 0.0};
  scheduler.apply(point, {&left_state, &right_state}, {&left_rhs, &right_rhs});
  EXPECT_EQ(evaluator_calls, 1)
      << "a rejected detached candidate must not mutate the accepted registry";
  for (int j = domain.lo[1]; j <= domain.hi[1]; ++j)
    EXPECT_EQ(get_cell(left_rhs, domain.hi[0], j, 0) + get_cell(right_rhs, domain.lo[0], j, 0),
              Real(0));
}

TEST(test_multiblock_interface_scheduler,
     RematerializationPreservesIncrementalFineRouteInstallationDuringBindBootstrap) {
  ensure_runtime();
  std::array<int, 2> evaluator_calls{0, 0};
  AmrRuntime runtime = make_dynamic_interface_runtime(4, 1, evaluator_calls);
  test::install_prepared_threshold_union(runtime, {{0, 0, Real(-1)}, {1, 0, Real(-1)}},
                                         "test::interface-bind-bootstrap@1");
  runtime.begin_bootstrap_plan();
  ASSERT_TRUE(runtime.bootstrap_next_level(kAmrRefRatio));
  ASSERT_EQ(runtime.nlev(), 2);
  runtime.commit_bootstrap_level();

  AxisAlignedInterface fine_route = aligned_x_route("amr.dynamic.shared-flux");
  fine_route.level = 1;
  fine_route.affine_mapping_identity = "periodic-x-translation";
  fine_route.right_normal_translation = Real(1);
  runtime.install_level_interface_flux(
      1, fine_route, serial_interface_execution(),
      [&evaluator_calls](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
        ++evaluator_calls[1];
        for (int face = 0; face < batch.face_count; ++face)
          batch.shared_flux[face] = Real(face + 1);
      });
  runtime.require_complete_active_level_interfaces();

  MultiFab& left = runtime.level_state(0, 1);
  MultiFab& right = runtime.level_state(1, 1);
  MultiFab left_rhs(left.box_array(), left.dmap(), 1, 0);
  MultiFab right_rhs(right.box_array(), right.dmap(), 1, 0);
  const BoundaryEvaluationPoint point{
      "clock.interface-bind-bootstrap", 0, 1, 0, 0, amr::Rational(0, 1), 0.1, 0.0};
  runtime.level_rhs_with_interfaces(1, point, {&left, &right}, {&left_rhs, &right_rhs});
  EXPECT_EQ(evaluator_calls[1], 1);
}

TEST(test_multiblock_interface_scheduler,
     BootstrapRollbackRemovesProvisionalFineRoutesAndRestoresTheAcceptedPrefix) {
  ensure_runtime();
  std::array<int, 2> evaluator_calls{0, 0};
  AmrRuntime runtime = make_dynamic_interface_runtime(4, 1, evaluator_calls);
  test::install_prepared_threshold_union(runtime, {{0, 0, Real(-1)}, {1, 0, Real(-1)}},
                                         "test::interface-bind-bootstrap-rollback@1");
  runtime.begin_bootstrap_plan();
  ASSERT_TRUE(runtime.bootstrap_next_level(kAmrRefRatio));

  AxisAlignedInterface fine_route = aligned_x_route("amr.dynamic.shared-flux");
  fine_route.level = 1;
  fine_route.affine_mapping_identity = "periodic-x-translation";
  fine_route.right_normal_translation = Real(1);
  runtime.install_level_interface_flux(
      1, fine_route, serial_interface_execution(),
      [&evaluator_calls](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
        ++evaluator_calls[1];
        for (int face = 0; face < batch.face_count; ++face)
          batch.shared_flux[face] = Real(face + 1);
      });
  runtime.require_complete_active_level_interfaces();

  runtime.rollback_bootstrap_level();
  ASSERT_EQ(runtime.nlev(), 1);
  EXPECT_TRUE(runtime.has_level_interfaces(0));
  EXPECT_FALSE(runtime.has_level_interfaces(1));
  runtime.require_complete_active_level_interfaces();

  MultiFab& left = runtime.level_state(0, 0);
  MultiFab& right = runtime.level_state(1, 0);
  MultiFab left_rhs(left.box_array(), left.dmap(), 1, 0);
  MultiFab right_rhs(right.box_array(), right.dmap(), 1, 0);
  const BoundaryEvaluationPoint point{
      "clock.interface-bind-bootstrap-rollback", 0, 0, 0, 0, amr::Rational(0, 1), 0.1, 0.0};
  runtime.level_rhs_with_interfaces(0, point, {&left, &right}, {&left_rhs, &right_rhs});
  EXPECT_EQ(evaluator_calls, (std::array<int, 2>{1, 0}));
}

TEST(test_multiblock_interface_scheduler,
     FixedTwoLevelPublicationEvaluatesOnceAndStagesOnlyItsQualifiedLevelOrientation) {
  ensure_runtime();
  const Box2D left_box{{0, 0}, {1, 2}};
  const Box2D right_box{{2, 0}, {3, 2}};
  MultiFab left_state = make_field(left_box, 1);
  MultiFab right_state = make_field(right_box, 1);
  MultiFab left_rhs(left_state.box_array(), left_state.dmap(), 1, 0);
  MultiFab right_rhs(right_state.box_array(), right_state.dmap(), 1, 0);
  left_state.set_val(Real(2));
  right_state.set_val(Real(6));
  left_rhs.set_val(Real(0));
  right_rhs.set_val(Real(0));

  const Geometry left_geometry{left_box, Real(0), Real(1), Real(0), Real(3)};
  const Geometry right_geometry{right_box, Real(1), Real(2), Real(0), Real(3)};
  AxisAlignedInterface route = aligned_x_route("amr.fixed-two-level.shared-flux");
  InterfaceFluxScheduler scheduler;
  int evaluator_calls = 0;
  scheduler.install(route, left_state, left_geometry, right_state, right_geometry,
                    serial_interface_execution(),
                    [&](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
                      ++evaluator_calls;
                      ASSERT_EQ(batch.face_count, 3);
                      ASSERT_EQ(batch.component_count, 1);
                      for (int face = 0; face < batch.face_count; ++face)
                        batch.shared_flux[face] = Real(face + 2);
                    });

  constexpr std::uint64_t topology_epoch = 17;
  InterfaceFluxFragmentLedger ledger(topology_epoch);
  ledger.begin();
  const amr::ClockWindow interval{{0, 7, amr::Rational(0, 1), 0.0},
                                  {0, 7, amr::Rational(1, 1), 0.2}};
  const amr::ClockStamp clock{0, 7, amr::Rational(1, 2), 0.1};
  InterfaceFluxFragmentPublication publication{
      &ledger, topology_epoch, 2, clock, "program.group.node.42", interval, amr::Rational(3, 4)};
  const BoundaryEvaluationPoint point{"clock.fragments",   7,   0,  0, 42,
                                      amr::Rational(1, 2), 0.2, 0.1};
  std::vector<MultiFab*> states{&left_state, &right_state};
  std::vector<MultiFab*> rhs{&left_rhs, &right_rhs};
  scheduler.apply(point, states, rhs, &publication);

  EXPECT_EQ(evaluator_calls, 1);
  EXPECT_EQ(scheduler.evaluation_count(route.identity, 0), 1u);
  EXPECT_EQ(ledger.pending_size(), 1u);
  EXPECT_EQ(ledger.published_size(), 0u);
  ledger.commit();
  ASSERT_EQ(ledger.published_size(), 1u);
  const auto& first = ledger.published_entries()[0];
  EXPECT_EQ(first.key.interface_identity, route.identity);
  EXPECT_EQ(first.key.topology_epoch, topology_epoch);
  EXPECT_EQ(first.key.coarse_level, 0);
  EXPECT_EQ(first.key.fine_level, 1);
  EXPECT_EQ(first.key.clock.level, clock.level);
  EXPECT_EQ(first.key.clock.macro_step, clock.macro_step);
  EXPECT_EQ(first.key.clock.phase, clock.phase);
  EXPECT_EQ(first.key.stage_identity, "program.group.node.42");
  EXPECT_EQ(first.key.orientation, amr::InterfaceFluxOrientation::CoarseOutward);
  EXPECT_EQ(first.key.left_block, 0u);
  EXPECT_EQ(first.key.right_block, 1u);
  EXPECT_EQ(first.measure.stage_weight, amr::Rational(3, 4));
  EXPECT_DOUBLE_EQ(first.measure.face_measure, 1.0);
  EXPECT_DOUBLE_EQ(first.measure.substep_duration, 0.2);
  EXPECT_TRUE(first.measure.stage_weight_resolved);
  EXPECT_EQ(first.payload, (InterfaceFluxFragmentPayload{Real(2), Real(3), Real(4)}));
}

TEST(test_multiblock_interface_scheduler,
     RejectedFixedTwoLevelAttemptRollsBackWithoutPublishingAnyFragment) {
  ensure_runtime();
  const Box2D left_box{{0, 0}, {1, 1}};
  const Box2D right_box{{2, 0}, {3, 1}};
  MultiFab left_state = make_field(left_box, 1);
  MultiFab right_state = make_field(right_box, 1);
  MultiFab left_rhs(left_state.box_array(), left_state.dmap(), 1, 0);
  MultiFab right_rhs(right_state.box_array(), right_state.dmap(), 1, 0);
  const Geometry left_geometry{left_box, Real(0), Real(1), Real(0), Real(2)};
  const Geometry right_geometry{right_box, Real(1), Real(2), Real(0), Real(2)};
  AxisAlignedInterface route = aligned_x_route("amr.rejected.shared-flux");
  route.level = 1;
  InterfaceFluxScheduler scheduler;
  scheduler.install(route, left_state, left_geometry, right_state, right_geometry,
                    serial_interface_execution(),
                    [](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
                      for (int face = 0; face < batch.face_count; ++face)
                        batch.shared_flux[face] = Real(5);
                    });

  InterfaceFluxFragmentLedger ledger(23);
  ledger.begin();
  const amr::ClockWindow interval{{1, 9, amr::Rational(0, 1), 0.4},
                                  {1, 9, amr::Rational(1, 1), 0.5}};
  InterfaceFluxFragmentPublication publication{&ledger,
                                               23,
                                               2,
                                               amr::ClockStamp{1, 9, amr::Rational(1, 2), 0.45},
                                               "program.group.node.8",
                                               interval,
                                               amr::Rational(1, 1)};
  const BoundaryEvaluationPoint point{"clock.fragments",   9,   1,   0, 8,
                                      amr::Rational(1, 2), 0.1, 0.45};
  std::vector<MultiFab*> states{&left_state, &right_state};
  std::vector<MultiFab*> rhs{&left_rhs, &right_rhs};
  scheduler.apply(point, states, rhs, &publication);
  ASSERT_EQ(ledger.pending_size(), 1u);
  EXPECT_EQ(ledger.published_size(), 0u);

  ledger.rollback();
  EXPECT_TRUE(ledger.empty());
  EXPECT_EQ(ledger.pending_size(), 0u);
  EXPECT_EQ(ledger.published_size(), 0u);
}

TEST(test_multiblock_interface_scheduler,
     AmrProgramPersistsInterfaceFragmentsAcrossAcceptedStateImportAndRejectedAttempt) {
  ensure_runtime();
  constexpr int cells = 4;
  AmrBuildParams params;
  params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  params.mesh.periodicity = Periodicity{true, true};
  params.mesh.n = cells;
  params.mesh.L = 1.0;
  params.mesh.regrid_every = 0;
  params.poisson.bc = BCRec{};
  detail::SharedAmrLayout layout = detail::make_shared_amr_layout_levels(params, 2);
  layout.ba[1] = BoxArray(std::vector<Box2D>{layout.geom.domain.refine(kAmrRefRatio)});
  layout.dm[1] = layout.load_balance->distribute(layout.ba[1], n_ranks());

  std::vector<AmrRuntimeBlock> blocks;
  for (const char* name : {"left", "right"}) {
    AmrRuntimeBlock block = detail::dispatch_amr_block(
        scalar_model(), "none", "rusanov", layout, name,
        std::vector<double>(static_cast<std::size_t>(cells) * cells, 1.0), true, 1.4, 1, false, 1);
    const auto omit_local_interface = [](MultiFab&, const MultiFab&, const Geometry&, MultiFab& fx,
                                         MultiFab& fy, MultiFab& rhs) {
      fx.set_val(Real(0));
      fy.set_val(Real(0));
      rhs.set_val(Real(0));
    };
    block.level_flux_capture = omit_local_interface;
    block.level_flux_capture_neg_div = omit_local_interface;
    block.level_rhs_without_prepared_interfaces = [](const BoundaryEvaluationPoint&, MultiFab&,
                                                     const MultiFab&, const Geometry&,
                                                     MultiFab& rhs) { rhs.set_val(Real(0)); };
    block.level_neg_div_flux_without_prepared_interfaces =
        block.level_rhs_without_prepared_interfaces;
    blocks.push_back(std::move(block));
  }
  AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc, std::move(blocks),
                     layout.base_per, layout.replicated_coarse, layout.wall);
  test::install_second_order_amr_transfer_authorities(runtime, 2);
  runtime.set_parent_child_temporal_relations({amr::ParentChildClockRelation(
      0, 1, amr::Rational(2, 1), amr::RemainderPolicy::IntegralOnly)});

  std::array<int, 2> evaluator_calls{0, 0};
  for (int level = 0; level < 2; ++level) {
    AxisAlignedInterface route = aligned_x_route("amr.program.shared-flux");
    route.level = level;
    route.affine_mapping_identity = "periodic-x-translation";
    route.right_normal_translation = Real(1);
    runtime.install_level_interface_flux(
        level, route, serial_interface_execution(),
        [&, level](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
          ++evaluator_calls[static_cast<std::size_t>(level)];
          for (int face = 0; face < batch.face_count; ++face)
            batch.shared_flux[face] = Real(level + face + 1);
        });
  }

  AmrSystem facade(AmrSystemConfig{});
  facade.set_program_block_map({0, 1});
  runtime::program::AmrProgramContext context(&runtime, &facade);
  context.configure_primary_clock("clock.program-fragments");
  const auto evaluate_group = [&](bool reject_after_parent, bool mismatch_consumer_weight = false) {
    context.advance_hierarchy(0.2, [&](double level_dt) {
      context.set_stage_time(1, 2);
      MultiFab& left = context.state(0);
      MultiFab& right = context.state(1);
      MultiFab& left_rhs = context.rhs_scratch(100, 0, left);
      MultiFab& right_rhs = context.rhs_scratch(101, 0, right);
      context.rhs_group(42, {{0, &left, &left_rhs, 11, 0}, {1, &right, &right_rhs, 12, 0}});
      const Box2D box = left.box(0);
      const int j = box.lo[1];
      const Real left_flux = left_rhs.fab(0).const_array()(box.hi[0], j, 0);
      const Real right_flux = right_rhs.fab(0).const_array()(box.lo[0], j, 0);
      EXPECT_NE(left_flux, Real(0));
      EXPECT_EQ(left_flux + right_flux, Real(0));
      context.axpy(left, Real(0.5 * level_dt), left_rhs, Real(level_dt), {{1, 1, 2}});
      if (mismatch_consumer_weight)
        context.axpy(right, Real(0.25 * level_dt), right_rhs, Real(level_dt), {{1, 1, 4}});
      else
        context.axpy(right, Real(0.5 * level_dt), right_rhs, Real(level_dt), {{1, 1, 2}});
      if (reject_after_parent && context.level() == 0)
        throw std::runtime_error("reject after canonical parent interface flux");
    });
  };

  evaluate_group(false);
  EXPECT_EQ(evaluator_calls[0], 1);
  EXPECT_EQ(evaluator_calls[1], 2);
  const auto& accepted = context.accepted_interface_flux_fragments();
  ASSERT_EQ(accepted.size(), 3u);
  int coarse_orientation_count = 0;
  int fine_orientation_count = 0;
  int parent_clock_count = 0;
  int child_clock_count = 0;
  int first_child_window_count = 0;
  int second_child_window_count = 0;
  for (const auto& entry : accepted) {
    EXPECT_EQ(entry.key.interface_identity, "amr.program.shared-flux");
    EXPECT_EQ(entry.key.topology_epoch, runtime.topology_epoch());
    EXPECT_EQ(entry.key.stage_identity, "program.group.node.42");
    EXPECT_EQ(entry.key.coarse_level, 0);
    EXPECT_EQ(entry.key.fine_level, 1);
    EXPECT_EQ(entry.key.interval.begin.level, entry.key.clock.level);
    EXPECT_EQ(entry.key.interval.end.level, entry.key.clock.level);
    EXPECT_EQ(entry.key.interval.begin.macro_step, 0);
    EXPECT_EQ(entry.key.interval.end.macro_step, 0);
    EXPECT_EQ(entry.key.clock.macro_step, 0);
    EXPECT_EQ(entry.key.left_block, 0u);
    EXPECT_EQ(entry.key.right_block, 1u);
    EXPECT_EQ(entry.measure.stage_weight, amr::Rational(1, 2));
    EXPECT_TRUE(entry.measure.stage_weight_resolved);
    if (entry.key.orientation == amr::InterfaceFluxOrientation::CoarseOutward)
      ++coarse_orientation_count;
    else if (entry.key.orientation == amr::InterfaceFluxOrientation::FineOutward)
      ++fine_orientation_count;
    if (entry.key.clock.level == 0) {
      ++parent_clock_count;
      EXPECT_EQ(entry.key.interval.begin.phase, amr::Rational(0, 1));
      EXPECT_EQ(entry.key.interval.end.phase, amr::Rational(1, 1));
      EXPECT_EQ(entry.key.clock.phase, amr::Rational(1, 2));
      EXPECT_DOUBLE_EQ(entry.key.interval.begin.physical_time, 0.0);
      EXPECT_DOUBLE_EQ(entry.key.interval.end.physical_time, 0.2);
      EXPECT_DOUBLE_EQ(entry.key.clock.physical_time, 0.1);
      EXPECT_DOUBLE_EQ(entry.measure.face_measure, 0.25);
      EXPECT_DOUBLE_EQ(entry.measure.substep_duration, 0.2);
    } else if (entry.key.clock.level == 1) {
      ++child_clock_count;
      EXPECT_DOUBLE_EQ(entry.measure.face_measure, 0.125);
      EXPECT_DOUBLE_EQ(entry.measure.substep_duration, 0.1);
      if (entry.key.interval.begin.phase == amr::Rational(0, 1)) {
        ++first_child_window_count;
        EXPECT_EQ(entry.key.interval.end.phase, amr::Rational(1, 2));
        EXPECT_EQ(entry.key.clock.phase, amr::Rational(1, 4));
        EXPECT_DOUBLE_EQ(entry.key.interval.begin.physical_time, 0.0);
        EXPECT_DOUBLE_EQ(entry.key.interval.end.physical_time, 0.1);
        EXPECT_DOUBLE_EQ(entry.key.clock.physical_time, 0.05);
      } else {
        ++second_child_window_count;
        EXPECT_EQ(entry.key.interval.begin.phase, amr::Rational(1, 2));
        EXPECT_EQ(entry.key.interval.end.phase, amr::Rational(1, 1));
        EXPECT_EQ(entry.key.clock.phase, amr::Rational(3, 4));
        EXPECT_DOUBLE_EQ(entry.key.interval.begin.physical_time, 0.1);
        EXPECT_DOUBLE_EQ(entry.key.interval.end.physical_time, 0.2);
        EXPECT_DOUBLE_EQ(entry.key.clock.physical_time, 0.15);
      }
    } else {
      ADD_FAILURE() << "fragment clock escaped the authored two-level test hierarchy";
    }
  }
  EXPECT_EQ(coarse_orientation_count, 1);
  EXPECT_EQ(fine_orientation_count, 2);
  EXPECT_EQ(parent_clock_count, 1);
  EXPECT_EQ(child_clock_count, 2);
  EXPECT_EQ(first_child_window_count, 1);
  EXPECT_EQ(second_child_window_count, 1);

  const std::array<int, 2> calls_before_rejection = evaluator_calls;
  EXPECT_THROW(evaluate_group(true), std::runtime_error);
  EXPECT_EQ(evaluator_calls[0], calls_before_rejection[0] + 1);
  EXPECT_EQ(evaluator_calls[1], calls_before_rejection[1]);
  EXPECT_EQ(context.accepted_interface_flux_fragments().size(), 3u)
      << "the rejected parent publication must not replace the accepted report";

  const std::array<int, 2> calls_before_mismatch = evaluator_calls;
  EXPECT_THROW(evaluate_group(false, true), std::runtime_error);
  EXPECT_EQ(evaluator_calls[0], calls_before_mismatch[0] + 1);
  EXPECT_EQ(evaluator_calls[1], calls_before_mismatch[1]);
  EXPECT_EQ(context.accepted_interface_flux_fragments().size(), 3u)
      << "different consumer weights must fail before replacing the accepted report";

  context.advance_hierarchy(0.2, [&](double) {
    for (int iteration = 0; iteration < 2; ++iteration) {
      auto logical = context.logical_evaluation_scope(iteration, 2);
      context.set_stage_time(1, 2);
      MultiFab& left = context.state(0);
      MultiFab& right = context.state(1);
      MultiFab& left_rhs = context.rhs_scratch(110, 0, left);
      MultiFab& right_rhs = context.rhs_scratch(111, 0, right);
      context.rhs_group(43, {{0, &left, &left_rhs, 13, 0}, {1, &right, &right_rhs, 14, 0}});
      const Real logical_dt = logical.dt();
      context.axpy(left, logical_dt, left_rhs, logical_dt, {{1, 1, 1}});
      context.axpy(right, logical_dt, right_rhs, logical_dt, {{1, 1, 1}});
    }
  });
  const auto& logical_fragments = context.accepted_interface_flux_fragments();
  ASSERT_EQ(logical_fragments.size(), 6u);
  int parent_logical_fragments = 0;
  int child_logical_fragments = 0;
  for (const auto& entry : logical_fragments) {
    EXPECT_EQ(entry.key.stage_identity, "program.group.node.43");
    EXPECT_EQ(entry.measure.stage_weight, amr::Rational(1, 1));
    if (entry.key.clock.level == 0) {
      ++parent_logical_fragments;
      EXPECT_DOUBLE_EQ(entry.measure.substep_duration, 0.1);
    } else {
      ++child_logical_fragments;
      EXPECT_DOUBLE_EQ(entry.measure.substep_duration, 0.05);
    }
  }
  EXPECT_EQ(parent_logical_fragments, 2);
  EXPECT_EQ(child_logical_fragments, 4);

  context.advance_synchronized_hierarchy(0.2, [&](double hierarchy_dt) {
    for (int level = 0; level < 2; ++level) {
      context.set_level(level);
      context.set_stage_time(1, 2);
      MultiFab& left = context.state(0);
      MultiFab& right = context.state(1);
      MultiFab& left_rhs = context.rhs_scratch(120, 0, left);
      MultiFab& right_rhs = context.rhs_scratch(121, 0, right);
      context.rhs_group(44, {{0, &left, &left_rhs, 15, 0}, {1, &right, &right_rhs, 16, 0}});
      context.axpy(left, Real(0.5 * hierarchy_dt), left_rhs, Real(hierarchy_dt), {{1, 1, 2}});
      context.axpy(right, Real(0.5 * hierarchy_dt), right_rhs, Real(hierarchy_dt), {{1, 1, 2}});
    }
  });
  const auto& barrier_fragments = context.accepted_interface_flux_fragments();
  ASSERT_EQ(barrier_fragments.size(), 2u);
  for (const auto& entry : barrier_fragments) {
    EXPECT_EQ(entry.key.stage_identity, "program.group.node.44");
    EXPECT_EQ(entry.key.interval.begin.level, entry.key.clock.level);
    EXPECT_EQ(entry.key.interval.end.level, entry.key.clock.level);
    EXPECT_EQ(entry.measure.stage_weight, amr::Rational(1, 2));
    EXPECT_DOUBLE_EQ(entry.measure.substep_duration, 0.2);
  }

  // The accepted audit is part of the canonical Program image, not a process-local report. Decode
  // the exact bytes published by the real interface execution, overwrite the live report with
  // another accepted step, then import and enter a rejected attempt. Import plus rollback must
  // recover the report exactly; an unresolved accepted weight is rejected by the canonical decoder
  // before facade publication.
  const std::vector<std::uint8_t> checkpoint = facade.program_accepted_state();
  const auto persisted = runtime::program::deserialize_amr_program_accepted_state(checkpoint);
  ASSERT_EQ(persisted.accepted_interface_flux_ledger.size(), 2u);
  for (const auto& entry : persisted.accepted_interface_flux_ledger) {
    EXPECT_EQ(entry.key.stage_identity, "program.group.node.44");
    EXPECT_TRUE(entry.measure.stage_weight_resolved);
  }

  evaluate_group(false);
  ASSERT_EQ(context.accepted_interface_flux_fragments().size(), 3u);
  EXPECT_EQ(context.accepted_interface_flux_fragments().front().key.stage_identity,
            "program.group.node.42");
  EXPECT_NE(facade.program_accepted_state(), checkpoint);

  auto malformed = persisted;
  malformed.accepted_interface_flux_ledger.front().measure.stage_weight_resolved = false;
  const auto malformed_bytes = runtime::program::serialize_amr_program_accepted_state(malformed);
  const auto accepted_before_malformed = facade.program_accepted_state();
  const std::uint64_t revision_before_malformed = facade.program_accepted_state_revision();
  EXPECT_THROW(runtime::program::deserialize_amr_program_accepted_state(malformed_bytes),
               std::runtime_error);
  EXPECT_EQ(facade.program_accepted_state(), accepted_before_malformed);
  EXPECT_EQ(facade.program_accepted_state_revision(), revision_before_malformed);

  facade.restore_program_accepted_state(checkpoint);
  EXPECT_THROW(
      context.advance_hierarchy(0.2,
                                [&](double) {
                                  throw std::runtime_error(
                                      "reject after strict interface-ledger restart import");
                                }),
      std::runtime_error);
  EXPECT_EQ(facade.program_accepted_state(), checkpoint);
  const auto& restored_fragments = context.accepted_interface_flux_fragments();
  ASSERT_EQ(restored_fragments.size(), 2u);
  for (const auto& entry : restored_fragments) {
    EXPECT_EQ(entry.key.stage_identity, "program.group.node.44");
    EXPECT_EQ(entry.key.topology_epoch, runtime.topology_epoch());
    EXPECT_TRUE(entry.measure.stage_weight_resolved);
    EXPECT_EQ(entry.measure.stage_weight, amr::Rational(1, 2));
    EXPECT_DOUBLE_EQ(entry.measure.substep_duration, 0.2);
  }
}

TEST(test_multiblock_interface_scheduler,
     FrozenThreeLevelProgramPublishesEverySubcycledInterfaceLevel) {
  ensure_runtime();
  constexpr int cells = 4;
  AmrBuildParams params;
  params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  params.mesh.periodicity = Periodicity{true, true};
  params.mesh.n = cells;
  params.mesh.L = 1.0;
  params.mesh.regrid_every = 0;
  params.poisson.bc = BCRec{};
  detail::SharedAmrLayout layout = detail::make_shared_amr_layout_levels(params, 3);
  for (int level = 1; level < 3; ++level) {
    const int refinement = level == 1 ? kAmrRefRatio : kAmrRefRatio * kAmrRefRatio;
    layout.ba[static_cast<std::size_t>(level)] =
        BoxArray(std::vector<Box2D>{layout.geom.domain.refine(refinement)});
    layout.dm[static_cast<std::size_t>(level)] =
        layout.load_balance->distribute(layout.ba[static_cast<std::size_t>(level)], n_ranks());
  }

  std::vector<AmrRuntimeBlock> blocks;
  for (const char* name : {"left", "right"}) {
    AmrRuntimeBlock block = detail::dispatch_amr_block(
        scalar_model(), "none", "rusanov", layout, name,
        std::vector<double>(static_cast<std::size_t>(cells) * cells, 1.0), true, 1.4, 1, false, 1);
    const auto omit_local_interface = [](MultiFab&, const MultiFab&, const Geometry&, MultiFab& fx,
                                         MultiFab& fy, MultiFab& rhs) {
      fx.set_val(Real(0));
      fy.set_val(Real(0));
      rhs.set_val(Real(0));
    };
    block.level_flux_capture = omit_local_interface;
    block.level_flux_capture_neg_div = omit_local_interface;
    block.level_rhs_without_prepared_interfaces = [](const BoundaryEvaluationPoint&, MultiFab&,
                                                     const MultiFab&, const Geometry&,
                                                     MultiFab& rhs) { rhs.set_val(Real(0)); };
    block.level_neg_div_flux_without_prepared_interfaces =
        block.level_rhs_without_prepared_interfaces;
    blocks.push_back(std::move(block));
  }
  AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc, std::move(blocks),
                     layout.base_per, layout.replicated_coarse, layout.wall);
  test::install_second_order_amr_transfer_authorities(runtime, 2);
  runtime.set_parent_child_temporal_relations(
      {amr::ParentChildClockRelation(0, 1, amr::Rational(2, 1), amr::RemainderPolicy::IntegralOnly),
       amr::ParentChildClockRelation(1, 2, amr::Rational(2, 1),
                                     amr::RemainderPolicy::IntegralOnly)});

  std::array<int, 3> evaluator_calls{0, 0, 0};
  for (int level = 0; level < 3; ++level) {
    AxisAlignedInterface route = aligned_x_route("amr.program.three-level.shared-flux");
    route.level = level;
    route.affine_mapping_identity = "periodic-x-translation";
    route.right_normal_translation = Real(1);
    runtime.install_level_interface_flux(
        level, route, serial_interface_execution(),
        [&, level](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
          ++evaluator_calls[static_cast<std::size_t>(level)];
          for (int face = 0; face < batch.face_count; ++face)
            batch.shared_flux[face] = Real(level + face + 1);
        });
  }
  runtime.require_complete_active_level_interfaces();

  AmrSystem facade(AmrSystemConfig{});
  facade.set_program_block_map({0, 1});
  runtime::program::AmrProgramContext context(&runtime, &facade);
  context.configure_primary_clock("clock.program-three-level-fragments");
  context.advance_hierarchy(0.2, [&](double level_dt) {
    context.set_stage_time(1, 2);
    MultiFab& left = context.state(0);
    MultiFab& right = context.state(1);
    MultiFab& left_rhs = context.rhs_scratch(150, 0, left);
    MultiFab& right_rhs = context.rhs_scratch(151, 0, right);
    context.rhs_group(52, {{0, &left, &left_rhs, 21, 0}, {1, &right, &right_rhs, 22, 0}});
    const Box2D box = left.box(0);
    const int j = box.lo[1];
    EXPECT_NE(left_rhs.fab(0).const_array()(box.hi[0], j, 0), Real(0));
    EXPECT_EQ(left_rhs.fab(0).const_array()(box.hi[0], j, 0) +
                  right_rhs.fab(0).const_array()(box.lo[0], j, 0),
              Real(0));
    context.axpy(left, Real(0.5 * level_dt), left_rhs, Real(level_dt), {{1, 1, 2}});
    context.axpy(right, Real(0.5 * level_dt), right_rhs, Real(level_dt), {{1, 1, 2}});
  });

  EXPECT_EQ(evaluator_calls, (std::array<int, 3>{1, 2, 4}));
  const auto& fragments = context.accepted_interface_flux_fragments();
  ASSERT_EQ(fragments.size(), 9u);
  std::array<int, 3> level_fragments{0, 0, 0};
  std::array<int, 2> pair_fragments{0, 0};
  int coarse_orientation_count = 0;
  int fine_orientation_count = 0;
  for (const auto& fragment : fragments) {
    EXPECT_EQ(fragment.key.interface_identity, "amr.program.three-level.shared-flux");
    EXPECT_EQ(fragment.key.topology_epoch, runtime.topology_epoch());
    EXPECT_EQ(fragment.key.stage_identity, "program.group.node.52");
    ASSERT_GE(fragment.key.clock.level, 0);
    ASSERT_LT(fragment.key.clock.level, 3);
    EXPECT_EQ(fragment.key.interval.begin.level, fragment.key.clock.level);
    EXPECT_EQ(fragment.key.interval.end.level, fragment.key.clock.level);
    EXPECT_EQ(fragment.key.clock.macro_step, fragment.key.interval.begin.macro_step);
    EXPECT_EQ(fragment.key.clock.macro_step, fragment.key.interval.end.macro_step);
    EXPECT_EQ(fragment.key.clock.phase,
              fragment.key.interval.begin.phase +
                  amr::Rational(1, 2) *
                      (fragment.key.interval.end.phase - fragment.key.interval.begin.phase));
    ++level_fragments[static_cast<std::size_t>(fragment.key.clock.level)];
    ASSERT_GE(fragment.key.coarse_level, 0);
    ASSERT_LT(fragment.key.coarse_level, 2);
    EXPECT_EQ(fragment.key.fine_level, fragment.key.coarse_level + 1);
    ++pair_fragments[static_cast<std::size_t>(fragment.key.coarse_level)];
    EXPECT_TRUE(fragment.key.clock.level == fragment.key.coarse_level ||
                fragment.key.clock.level == fragment.key.fine_level);
    if (fragment.key.clock.level == fragment.key.coarse_level)
      EXPECT_EQ(fragment.key.orientation, amr::InterfaceFluxOrientation::CoarseOutward);
    else
      EXPECT_EQ(fragment.key.orientation, amr::InterfaceFluxOrientation::FineOutward);
    EXPECT_EQ(fragment.measure.stage_weight, amr::Rational(1, 2));
    EXPECT_TRUE(fragment.measure.stage_weight_resolved);
    EXPECT_DOUBLE_EQ(fragment.measure.substep_duration,
                     0.2 / static_cast<double>(1 << fragment.key.clock.level));
    EXPECT_DOUBLE_EQ(fragment.measure.face_measure,
                     0.25 / static_cast<double>(1 << fragment.key.clock.level));
    if (fragment.key.orientation == amr::InterfaceFluxOrientation::CoarseOutward)
      ++coarse_orientation_count;
    else
      ++fine_orientation_count;
  }
  EXPECT_EQ(level_fragments, (std::array<int, 3>{1, 4, 4}));
  EXPECT_EQ(pair_fragments, (std::array<int, 2>{3, 6}));
  EXPECT_EQ(coarse_orientation_count, 3);
  EXPECT_EQ(fine_orientation_count, 6);
}

TEST(test_multiblock_interface_scheduler,
     DynamicTwoLevelRegridRematerializesConservativeInterfacesAndFragmentIdentity) {
  ensure_runtime();
  std::array<int, 2> evaluator_calls{0, 0};
  AmrRuntime runtime = make_dynamic_interface_runtime(4, 2, evaluator_calls);
  ASSERT_EQ(runtime.nlev(), 2);
  ASSERT_EQ(runtime.level_state(0, 1).box_array().size(), 1);
  const auto initial_fine_boxes = runtime.level_state(0, 1).box_array().boxes();

  const auto evaluate_level = [&](int level, std::int64_t tick) {
    MultiFab& left = runtime.level_state(0, level);
    MultiFab& right = runtime.level_state(1, level);
    MultiFab left_rhs(left.box_array(), left.dmap(), 1, 0);
    MultiFab right_rhs(right.box_array(), right.dmap(), 1, 0);
    const BoundaryEvaluationPoint point{"clock.dynamic-interface", tick, level, 0, 1,
                                        amr::Rational(1, 2),       0.1,  0.05};
    runtime.level_rhs_with_interfaces(level, point, {&left, &right}, {&left_rhs, &right_rhs});
    const Box2D domain = left.box_array().bounding_box();
    for (int j = domain.lo[1]; j <= domain.hi[1]; ++j)
      EXPECT_EQ(get_cell(left_rhs, domain.hi[0], j, 0) + get_cell(right_rhs, domain.lo[0], j, 0),
                Real(0));
  };
  evaluate_level(0, 0);
  evaluate_level(1, 0);
  EXPECT_EQ(evaluator_calls, (std::array<int, 2>{1, 1}));

  runtime.set_clustering(/*min_efficiency=*/1.0, /*min_box_size=*/1,
                         /*max_box_size=*/2);
  test::install_prepared_threshold_union(runtime, {{0, 0, Real(0.5)}, {1, 0, Real(0.5)}},
                                         "test::dynamic-interface-full-domain@1");
  const std::uint64_t accepted_epoch = runtime.topology_epoch();
  runtime.regrid();

  ASSERT_EQ(runtime.nlev(), 2);
  EXPECT_GT(runtime.level_state(0, 1).box_array().size(), 1)
      << "the proof requires one real fine-layout replacement";
  EXPECT_NE(runtime.level_state(0, 1).box_array().boxes(), initial_fine_boxes);
  EXPECT_GT(runtime.topology_epoch(), accepted_epoch);
  runtime.require_complete_active_level_interfaces();

  AmrSystem facade(AmrSystemConfig{});
  facade.set_program_block_map({0, 1});
  runtime::program::AmrProgramContext context(&runtime, &facade);
  context.configure_primary_clock("clock.dynamic-interface");
  context.advance_hierarchy(0.2, [&](double level_dt) {
    context.set_stage_time(1, 2);
    MultiFab& left = context.state(0);
    MultiFab& right = context.state(1);
    MultiFab& left_rhs = context.rhs_scratch(900, 0, left);
    MultiFab& right_rhs = context.rhs_scratch(901, 0, right);
    context.rhs_group(902, {{0, &left, &left_rhs, 903, 0}, {1, &right, &right_rhs, 904, 0}});
    const Box2D domain = left.box_array().bounding_box();
    for (int j = domain.lo[1]; j <= domain.hi[1]; ++j)
      EXPECT_EQ(get_cell(left_rhs, domain.hi[0], j, 0) + get_cell(right_rhs, domain.lo[0], j, 0),
                Real(0));
    context.axpy(left, Real(0.5 * level_dt), left_rhs, Real(level_dt), {{1, 1, 2}});
    context.axpy(right, Real(0.5 * level_dt), right_rhs, Real(level_dt), {{1, 1, 2}});
  });

  EXPECT_EQ(evaluator_calls, (std::array<int, 2>{2, 3}))
      << "rematerialization must preserve the prepared evaluator and its audit count";
  const auto& fragments = context.accepted_interface_flux_fragments();
  ASSERT_EQ(fragments.size(), 3u);
  for (const auto& fragment : fragments) {
    EXPECT_EQ(fragment.key.interface_identity, "amr.dynamic.shared-flux");
    EXPECT_EQ(fragment.key.topology_epoch, runtime.topology_epoch());
    EXPECT_EQ(fragment.key.stage_identity, "program.group.node.902");
    EXPECT_EQ(fragment.key.left_block, 0u);
    EXPECT_EQ(fragment.key.right_block, 1u);
  }
}

TEST(test_multiblock_interface_scheduler,
     DynamicThreeLevelFinestTransitionRematerializesTheCompletePreparedPrefix) {
  ensure_runtime();
  std::array<int, 3> evaluator_calls{0, 0, 0};
  AmrRuntime runtime = make_dynamic_interface_runtime(4, 3, evaluator_calls);
  ASSERT_EQ(runtime.nlev(), 3);
  const auto initial_middle_boxes = runtime.level_state(0, 1).box_array().boxes();
  const auto initial_finest_boxes = runtime.level_state(0, 2).box_array().boxes();

  // L1 already uses the exact max-size-4 clustering of a fully tagged L0 parent, whereas L2 starts
  // as one box. Reapplying the same full-domain tags therefore leaves L0 -> L1 unchanged and
  // replaces only L1 -> L2, while every physical interface face remains completely covered.
  runtime.set_clustering(/*min_efficiency=*/1.0, /*min_box_size=*/1,
                         /*max_box_size=*/4);
  test::install_prepared_threshold_union(runtime, {{0, 0, Real(0.5)}, {1, 0, Real(0.5)}},
                                         "test::dynamic-interface-three-level-finest@1");
  const std::uint64_t accepted_epoch = runtime.topology_epoch();
  runtime.regrid();

  ASSERT_EQ(runtime.nlev(), 3);
  EXPECT_EQ(runtime.level_state(0, 1).box_array().boxes(), initial_middle_boxes);
  EXPECT_NE(runtime.level_state(0, 2).box_array().boxes(), initial_finest_boxes);
  EXPECT_GT(runtime.topology_epoch(), accepted_epoch);
  runtime.require_complete_active_level_interfaces();

  for (int level = 0; level < 3; ++level) {
    MultiFab& left = runtime.level_state(0, level);
    MultiFab& right = runtime.level_state(1, level);
    MultiFab left_rhs(left.box_array(), left.dmap(), 1, 0);
    MultiFab right_rhs(right.box_array(), right.dmap(), 1, 0);
    const BoundaryEvaluationPoint point{
        "clock.dynamic-interface-three-level", 1,  level, 0, 0, amr::Rational(0, 1),
        0.1 / static_cast<double>(1 << level), 0.0};
    runtime.level_rhs_with_interfaces(level, point, {&left, &right}, {&left_rhs, &right_rhs});
    const Box2D domain = left.box_array().bounding_box();
    for (int j = domain.lo[1]; j <= domain.hi[1]; ++j)
      EXPECT_EQ(get_cell(left_rhs, domain.hi[0], j, 0) + get_cell(right_rhs, domain.lo[0], j, 0),
                Real(0));
  }
  EXPECT_EQ(evaluator_calls, (std::array<int, 3>{1, 1, 1}));
}

TEST(test_multiblock_interface_scheduler,
     DynamicThreeLevelNonFinestReplacementFailsClosedWithoutAFlatFallback) {
  ensure_runtime();
  std::array<int, 3> evaluator_calls{0, 0, 0};
  AmrRuntime runtime = make_dynamic_interface_runtime(4, 3, evaluator_calls);
  const auto accepted_middle_boxes = runtime.level_state(0, 1).box_array().boxes();
  const auto accepted_finest_boxes = runtime.level_state(0, 2).box_array().boxes();
  const std::uint64_t accepted_epoch = runtime.topology_epoch();

  // Max-size-2 reclusters the fully tagged L0 -> L1 transition itself. That non-finest replacement
  // transiently removes L2, which cannot be reconciled with the accepted three-level route prefix.
  runtime.set_clustering(/*min_efficiency=*/1.0, /*min_box_size=*/1,
                         /*max_box_size=*/2);
  test::install_prepared_threshold_union(runtime, {{0, 0, Real(0.5)}, {1, 0, Real(0.5)}},
                                         "test::dynamic-interface-three-level-non-finest@1");

  try {
    runtime.regrid();
    FAIL() << "non-finest replacement transiently removed L2 without failing closed";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("active hierarchy depth"), std::string::npos);
  }
  EXPECT_EQ(runtime.nlev(), 3);
  EXPECT_EQ(runtime.level_state(0, 1).box_array().boxes(), accepted_middle_boxes);
  EXPECT_EQ(runtime.level_state(0, 2).box_array().boxes(), accepted_finest_boxes);
  EXPECT_EQ(runtime.topology_epoch(), accepted_epoch);
  EXPECT_EQ(runtime.regrid_count(), 0);
  runtime.require_complete_active_level_interfaces();
}

TEST(test_multiblock_interface_scheduler,
     DynamicInterfaceActiveDepthChangeFailsClosedAndRestoresAcceptedRegistry) {
  ensure_runtime();
  std::array<int, 2> evaluator_calls{0, 0};
  AmrRuntime runtime = make_dynamic_interface_runtime(4, 2, evaluator_calls);
  const auto accepted_boxes = runtime.level_state(0, 1).box_array().boxes();
  const std::uint64_t accepted_epoch = runtime.topology_epoch();

  test::install_prepared_threshold_decisions(
      runtime, {{0, 0, Real(10)}, {1, 0, Real(10)}},
      {{0, 0, Real(10), test::PreparedThresholdRelation::Below},
       {1, 0, Real(10), test::PreparedThresholdRelation::Below}},
      "test::dynamic-interface-remove-level@1");
  EXPECT_THROW(runtime.regrid(), std::runtime_error);
  EXPECT_EQ(runtime.nlev(), 2);
  EXPECT_EQ(runtime.topology_epoch(), accepted_epoch);
  EXPECT_EQ(runtime.level_state(0, 1).box_array().boxes(), accepted_boxes);
  EXPECT_EQ(runtime.regrid_count(), 0);
  runtime.require_complete_active_level_interfaces();

  MultiFab& left = runtime.level_state(0, 1);
  MultiFab& right = runtime.level_state(1, 1);
  MultiFab left_rhs(left.box_array(), left.dmap(), 1, 0);
  MultiFab right_rhs(right.box_array(), right.dmap(), 1, 0);
  const BoundaryEvaluationPoint point{
      "clock.dynamic-interface-rollback", 1, 1, 0, 0, amr::Rational(0, 1), 0.1, 0.0};
  runtime.level_rhs_with_interfaces(1, point, {&left, &right}, {&left_rhs, &right_rhs});
  EXPECT_EQ(evaluator_calls[1], 1)
      << "rollback must leave the accepted interface registry executable";
}

TEST(test_multiblock_interface_scheduler,
     DynamicInterfaceRuntimeDepthCreationRequiresAnInstalledFineRoute) {
  ensure_runtime();
  std::array<int, 2> evaluator_calls{0, 0};
  AmrRuntime runtime = make_dynamic_interface_runtime(4, 1, evaluator_calls);
  ASSERT_EQ(runtime.nlev(), 1);
  ASSERT_EQ(runtime.max_levels(), 2);
  const std::uint64_t accepted_epoch = runtime.topology_epoch();

  test::install_prepared_threshold_union(runtime, {{0, 0, Real(-1)}, {1, 0, Real(-1)}},
                                         "test::dynamic-interface-create-level@1");
  try {
    runtime.regrid();
    FAIL() << "runtime regrid created L1 without an installed fine interface route";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("incomplete on the active hierarchy"),
              std::string::npos);
  }
  EXPECT_EQ(runtime.nlev(), 1);
  EXPECT_EQ(runtime.topology_epoch(), accepted_epoch);
  EXPECT_EQ(runtime.regrid_count(), 0);

  MultiFab& left = runtime.level_state(0, 0);
  MultiFab& right = runtime.level_state(1, 0);
  MultiFab left_rhs(left.box_array(), left.dmap(), 1, 0);
  MultiFab right_rhs(right.box_array(), right.dmap(), 1, 0);
  const BoundaryEvaluationPoint point{
      "clock.dynamic-interface-create-rollback", 1, 0, 0, 0, amr::Rational(0, 1), 0.1, 0.0};
  runtime.level_rhs_with_interfaces(0, point, {&left, &right}, {&left_rhs, &right_rhs});
  EXPECT_EQ(evaluator_calls[0], 1)
      << "rejected depth creation must restore the accepted coarse interface registry";
}

TEST(test_multiblock_interface_scheduler,
     MpiWorldSingleRankKeepsItsNativeIdentityAndExecutesTheCompleteLocalPair) {
#if !defined(POPS_HAS_MPI)
  GTEST_SKIP() << "requires a PoPS build with the native MPI transport enabled";
#else
  ensure_runtime();
  const PopsExecutionContextV1 execution = mpi_world_interface_execution();
  ASSERT_TRUE(comm_active());
  if (n_ranks() != 1)
    GTEST_SKIP() << "the distributed trace-exchange refusal is exercised by an MPI launch";

  const Box2D left_box{{0, 0}, {1, 2}};
  const Box2D right_box{{2, 0}, {3, 2}};
  MultiFab left_state = make_field(left_box, 1);
  MultiFab right_state = make_field(right_box, 1);
  MultiFab left_rhs(left_state.box_array(), left_state.dmap(), 1, 0);
  MultiFab right_rhs(right_state.box_array(), right_state.dmap(), 1, 0);
  left_state.set_val(Real(2));
  right_state.set_val(Real(6));
  left_rhs.set_val(Real(0));
  right_rhs.set_val(Real(0));

  AxisAlignedInterface route;
  route.identity = "mpi-world-one-rank.shared-flux";
  route.left_block = 0;
  route.right_block = 1;
  route.left_axis = route.right_axis = InterfaceAxis::X;
  route.left_side = InterfaceSide::High;
  route.right_side = InterfaceSide::Low;
  route.right_component_for_left = {0};
  authenticate_cell_average_trace(route);
  const Geometry left_geometry{left_box, Real(0), Real(1), Real(0), Real(3)};
  const Geometry right_geometry{right_box, Real(1), Real(2), Real(0), Real(3)};

  InterfaceFluxScheduler scheduler;
  int calls = 0;
  scheduler.install(route, left_state, left_geometry, right_state, right_geometry, execution,
                    [&](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
                      ++calls;
                      ASSERT_EQ(batch.face_count, 3);
                      for (int face = 0; face < batch.face_count; ++face)
                        batch.shared_flux[face] =
                            Real(0.5) * (batch.left_state[face] + batch.right_state[face]);
                    });
  const BoundaryEvaluationPoint point{"clock.mpi-one-rank", 1,   0,  0, 0,
                                      amr::Rational(0, 1),  0.1, 0.0};
  std::vector<MultiFab*> states{&left_state, &right_state};
  std::vector<MultiFab*> rhs{&left_rhs, &right_rhs};
  scheduler.apply(point, states, rhs);

  EXPECT_EQ(calls, 1);
  EXPECT_EQ(scheduler.evaluation_count(route.identity, 0), 1u);
  for (int j = left_box.lo[1]; j <= left_box.hi[1]; ++j)
    EXPECT_EQ(get_cell(left_rhs, left_box.hi[0], j, 0) + get_cell(right_rhs, right_box.lo[0], j, 0),
              Real(0));
#endif
}

TEST(test_multiblock_interface_scheduler, UnsupportedOrUnauthenticatedMappingsFailAtInstall) {
  ensure_runtime();
  const Box2D left_box{{0, 0}, {3, 2}};
  const Box2D right_box{{10, 7}, {15, 9}};
  const Geometry left_geometry{left_box, Real(0), Real(2), Real(0), Real(3)};
  const Geometry coincident_right{right_box, Real(2), Real(5), Real(0), Real(3)};
  const Geometry detached_right{right_box, Real(4), Real(7), Real(0), Real(3)};
  SystemBlockStore store;
  for (int block = 0; block < 2; ++block) {
    SystemBlockStore::BlockState state;
    state.name = block == 0 ? "left" : "right";
    state.U = make_field(block == 0 ? left_box : right_box, 2);
    state.ncomp = 2;
    state.rhs_into = [](MultiFab&, MultiFab& rhs) { rhs.set_val(Real(0)); };
    state.rhs_without_prepared_interfaces = [](const BoundaryEvaluationPoint&, MultiFab&,
                                               MultiFab& rhs) { rhs.set_val(Real(0)); };
    state.rhs_flux_only_without_prepared_interfaces = state.rhs_without_prepared_interfaces;
    store.blocks.push_back(std::move(state));
  }
  const InterfaceFluxEvaluator evaluator = [](const BoundaryEvaluationPoint&,
                                              const InterfaceFluxBatch&) {};
  int prepare_calls = 0;
  const InterfaceFluxEvaluatorFactory evaluator_factory = [&] {
    ++prepare_calls;
    return evaluator;
  };

  AxisAlignedInterface route = heterogeneous_route();
  route.identity = "detached";
  route.affine_mapping_identity.clear();
  route.right_tangential_scale = Real(1);
  route.right_tangential_offset = Real(0);
  route.tangential_orientation = TangentialOrientation::Aligned;
  EXPECT_THROW(store.install_interface_flux(route, left_geometry, detached_right,
                                            serial_interface_execution(), evaluator_factory),
               std::invalid_argument);

  route.identity = "cross-axis";
  route.right_axis = InterfaceAxis::Y;
  EXPECT_THROW(store.install_interface_flux(route, left_geometry, coincident_right,
                                            serial_interface_execution(), evaluator_factory),
               std::invalid_argument);

  route = heterogeneous_route();
  route.identity = "non-bijection";
  route.right_component_for_left = {0, 0};
  EXPECT_THROW(store.install_interface_flux(route, left_geometry, coincident_right,
                                            serial_interface_execution(), evaluator_factory),
               std::invalid_argument);
  route = heterogeneous_route();
  route.identity = "missing-trace-provider";
  route.left_trace_provider_identity.clear();
  EXPECT_THROW(store.install_interface_flux(route, left_geometry, coincident_right,
                                            serial_interface_execution(), evaluator_factory),
               std::invalid_argument);
  route = heterogeneous_route();
  route.identity = "unavailable-reconstructed-trace";
  route.left_trace_provider_identity = "limiter.minmod";
  route.left_trace_operation = InterfaceTraceOperation::ReconstructedFace;
  route.left_trace_required_depth = 2;
  EXPECT_THROW(store.install_interface_flux(route, left_geometry, coincident_right,
                                            serial_interface_execution(), evaluator_factory),
               std::invalid_argument);
  EXPECT_EQ(prepare_calls, 0)
      << "invalid topology/geometry/trace projection must fail before component prepare";
  EXPECT_THROW(store.interface_evaluation_count("non-bijection", 0), std::out_of_range);

  route = heterogeneous_route();
  route.identity = "first-owner";
  store.install_interface_flux(route, left_geometry, coincident_right, serial_interface_execution(),
                               evaluator_factory);
  EXPECT_EQ(prepare_calls, 1);
  route.identity = "competing-owner";
  EXPECT_THROW(store.install_interface_flux(route, left_geometry, coincident_right,
                                            serial_interface_execution(), evaluator_factory),
               std::invalid_argument);
  EXPECT_EQ(prepare_calls, 1)
      << "a face ownership conflict must fail before preparing a second component";

  // A refined patch that merely sits inside the physical domain is not a complete shared face.
  // Refusing it here prevents a fixed L1 seed from being mistaken for a coarse/fine block boundary.
  MultiFab partial_left = make_field(Box2D{{1, 1}, {2, 2}}, 1);
  MultiFab partial_right = make_field(Box2D{{5, 1}, {6, 2}}, 1);
  const Geometry declared_left{Box2D{{0, 0}, {3, 3}}, Real(0), Real(4), Real(0), Real(4)};
  const Geometry declared_right{Box2D{{4, 0}, {7, 3}}, Real(4), Real(8), Real(0), Real(4)};
  InterfaceFluxScheduler partial_scheduler;
  AxisAlignedInterface partial_route = aligned_x_route("partial-refined-face");
  EXPECT_THROW(
      partial_scheduler.install(partial_route, partial_left, declared_left, partial_right,
                                declared_right, serial_interface_execution(), evaluator_factory),
      std::invalid_argument);
}

TEST(test_multiblock_interface_scheduler,
     AmrRuntimeInstallsAndInvokesTheSamePairSchedulerAtOneExactPoint) {
  ensure_runtime();
  constexpr int cells = 4;
  AmrBuildParams params;
  params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  params.mesh.periodicity = Periodicity{true, true};
  params.mesh.n = cells;
  params.mesh.L = 1.0;
  params.mesh.regrid_every = 0;
  params.poisson.bc = BCRec{};
  const detail::SharedAmrLayout layout = detail::make_shared_amr_layout_levels(params, 1);
  std::vector<AmrRuntimeBlock> blocks;
  int full_rhs_calls = 0;
  int interface_omitting_rhs_calls = 0;
  const BoundaryEvaluationPoint point{"clock.fine", 9, 0, 2, 1, amr::Rational(2, 3), 0.125, 0.375};
  std::optional<BoundaryEvaluationPoint> residual_point;
  for (const char* name : {"left", "right"}) {
    blocks.push_back(detail::dispatch_amr_block(
        scalar_model(), "none", "rusanov", layout, name,
        std::vector<double>(static_cast<std::size_t>(cells) * cells, 1.0), true, 1.4, 1, false, 1));
    blocks.back().level_rhs = [&full_rhs_calls](MultiFab&, const MultiFab&, const Geometry&,
                                                MultiFab& rhs) {
      ++full_rhs_calls;
      rhs.set_val(Real(73));
    };
    blocks.back().level_rhs_without_prepared_interfaces =
        [&interface_omitting_rhs_calls, &residual_point](const BoundaryEvaluationPoint& evaluation,
                                                         MultiFab&, const MultiFab&,
                                                         const Geometry&, MultiFab& rhs) {
          ++interface_omitting_rhs_calls;
          residual_point = evaluation;
          rhs.set_val(Real(0));
        };
    blocks.back().level_neg_div_flux_without_prepared_interfaces =
        blocks.back().level_rhs_without_prepared_interfaces;
  }
  AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc, std::move(blocks),
                     layout.base_per, layout.replicated_coarse, layout.wall);
  test::install_second_order_amr_transfer_authorities(runtime, 2);

  AxisAlignedInterface route;
  route.identity = "amr.level0.shared_flux";
  route.left_block = 0;
  route.right_block = 1;
  route.level = 0;
  route.left_axis = route.right_axis = InterfaceAxis::X;
  route.left_side = InterfaceSide::High;
  route.right_side = InterfaceSide::Low;
  route.right_component_for_left = {0};
  authenticate_cell_average_trace(route);
  route.affine_mapping_identity = "periodic-x-translation";
  route.right_normal_translation = Real(1);

  int evaluator_calls = 0;
  runtime.install_level_interface_flux(
      0, route, serial_interface_execution(),
      [&](const BoundaryEvaluationPoint& observed, const InterfaceFluxBatch& batch) {
        ++evaluator_calls;
        EXPECT_EQ(observed, point);
        for (int face = 0; face < batch.face_count; ++face)
          batch.shared_flux[face] = Real(2.5);
      });

  MultiFab& left_state = runtime.level_state(0, 0);
  MultiFab& right_state = runtime.level_state(1, 0);
  MultiFab left_rhs(left_state.box_array(), left_state.dmap(), 1, 0);
  MultiFab right_rhs(right_state.box_array(), right_state.dmap(), 1, 0);
  std::vector<MultiFab*> states{&left_state, &right_state};
  std::vector<MultiFab*> rhs{&left_rhs, &right_rhs};
  runtime.level_rhs_with_interfaces(0, point, states, rhs);

  EXPECT_EQ(evaluator_calls, 1);
  EXPECT_EQ(runtime.interface_evaluation_count(route.identity, 0), 1u);
  EXPECT_EQ(full_rhs_calls, 0);
  EXPECT_EQ(interface_omitting_rhs_calls, 2);
  ASSERT_TRUE(residual_point.has_value());
  EXPECT_EQ(*residual_point, point);
  const ConstArray4 left_result = left_rhs.fab(0).const_array();
  const ConstArray4 right_result = right_rhs.fab(0).const_array();
  const Box2D box = left_state.box(0);
  for (int j = box.lo[1]; j <= box.hi[1]; ++j)
    EXPECT_EQ(left_result(box.hi[0], j, 0) + right_result(box.lo[0], j, 0), Real(0));
}

TEST(test_multiblock_interface_scheduler, AmrBoundaryRegistryUsesOtherBlocksProvisionalStageState) {
  ensure_runtime();
  AmrBuildParams params;
  params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  params.mesh.periodicity = Periodicity{true, true};
  params.mesh.n = 3;
  params.mesh.L = 1.0;
  params.mesh.regrid_every = 0;
  params.poisson.bc = BCRec{};
  const detail::SharedAmrLayout layout = detail::make_shared_amr_layout_levels(params, 1);
  std::vector<AmrRuntimeBlock> blocks;
  for (const char* name : {"a", "b"}) {
    blocks.push_back(detail::dispatch_amr_block(scalar_model(), "none", "rusanov", layout, name,
                                                std::vector<double>(9, 1.0), true, 1.4, 1, false,
                                                1));
  }
  const std::string a_state = "case::amr::a::state::U";
  const std::string b_state = "case::amr::b::state::U";
  blocks[0].state_identity = a_state;
  blocks[1].state_identity = b_state;
  blocks[0].boundary_plan = std::make_shared<PreparedBoundaryPlan>(
      "case::amr::a::boundary", 1, std::vector<BCRec>{BCRec{}}, std::vector<int>{}, a_state,
      PreparedBoundaryReadDependencies{{b_state}, {}});
  const auto b_read = blocks[0].boundary_plan->prepare_state_read(b_state);
  blocks[0].boundary_field_registry = std::make_shared<GridContext::BoundaryFieldRegistryFactory>();
  blocks[0].level_rhs_core_at_point_prepared =
      [b_read](const BoundaryEvaluationPoint& point, MultiFab& U, const MultiFab&, const Geometry&,
               MultiFab& R, const PreparedGridBoundarySession& boundary) {
        const auto reads = boundary.bind_reads(point, U);
        R.set_val(reads.state(b_read).fab(0).const_array()(0, 0, 0));
      };
  blocks[1].level_rhs_at_point = [](const BoundaryEvaluationPoint&, MultiFab&, const MultiFab&,
                                    const Geometry&, MultiFab& R) { R.set_val(Real(0)); };
  AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc, std::move(blocks),
                     layout.base_per, layout.replicated_coarse, layout.wall);
  test::install_second_order_amr_transfer_authorities(runtime, 2);
  runtime.install_boundary_storage_routes({});
  runtime.level_state(0, 0).set_val(Real(1));
  runtime.level_state(1, 0).set_val(Real(2));
  MultiFab stage_a = runtime.level_state(0, 0);
  MultiFab stage_b = runtime.level_state(1, 0);
  stage_a.set_val(Real(7));
  stage_b.set_val(Real(11));
  MultiFab rhs_a(stage_a.box_array(), stage_a.dmap(), 1, 0);
  MultiFab rhs_b(stage_b.box_array(), stage_b.dmap(), 1, 0);
  const BoundaryEvaluationPoint point{"clock.amr-stage", 4, 0, 0, 1, amr::Rational(1, 3), 0.2, 0.4};
  std::vector<MultiFab*> states{&stage_a, &stage_b};
  std::vector<MultiFab*> rhs{&rhs_a, &rhs_b};
  runtime.level_rhs_with_interfaces(0, point, states, rhs);

  EXPECT_EQ(rhs_a.fab(0).const_array()(0, 0, 0), Real(11));
  EXPECT_EQ(runtime.level_state(1, 0).fab(0).const_array()(0, 0, 0), Real(2));

  // The sparse grouped scope is the primitive used by independent reflux captures. It must expose
  // the sibling's provisional state to prepared boundary reads for the entire group, and an
  // exception must not leave the process-local registry active.
  const std::vector<int> requested_blocks{0, 1};
  rhs_a.set_val(Real(0));
  runtime.with_boundary_stage_states(point, requested_blocks, states, [&] {
    runtime.level_rhs_core_into_at(0, 0, point, stage_a, rhs_a, /*flux_only=*/false);
  });
  EXPECT_EQ(rhs_a.fab(0).const_array()(0, 0, 0), Real(11));
  EXPECT_THROW(runtime.with_boundary_stage_states(point, requested_blocks, states,
                                                  [] { throw std::runtime_error("stage abort"); }),
               std::runtime_error);
  EXPECT_NO_THROW(runtime.with_boundary_stage_states(point, requested_blocks, states, [] {}));
}
