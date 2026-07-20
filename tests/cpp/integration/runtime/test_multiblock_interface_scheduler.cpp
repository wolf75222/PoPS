#include <gtest/gtest.h>

#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/builders/factory/model_factory.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/system/system_block_store.hpp>

#include <cmath>
#include <optional>
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
  return route;
}

ModelSpec scalar_model() {
  ModelSpec spec;
  spec.transport = "exb";
  spec.source = "none";
  spec.elliptic = "charge";
  spec.q = 0.0;
  spec.B0 = 1.0;
  return spec;
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
  BoundaryEvaluationPoint observed;
  store.install_interface_flux(
      heterogeneous_route(), left_geometry, right_geometry, serial_interface_execution(),
      [&](const BoundaryEvaluationPoint& evaluation, const InterfaceFluxBatch& batch) {
        ++evaluator_calls;
        observed = evaluation;
        ASSERT_EQ(batch.face_count, 3);
        ASSERT_EQ(batch.component_count, 2);
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
  EXPECT_THROW(
      store.evaluate_rhs_with_interfaces(point, states, rhs, {}, GeometryMode::Staircase),
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
  const BoundaryEvaluationPoint point{"clock.multibox", 1, 0, 0, 0, amr::Rational(0, 1), 0.1, 0.0};
  std::vector<MultiFab*> states{&left_state, &right_state};
  std::vector<MultiFab*> rhs{&left_rhs, &right_rhs};
  scheduler.apply(point, states, rhs);

  EXPECT_EQ(calls, 1);
  for (int face = 0; face < 6; ++face)
    EXPECT_EQ(get_cell(left_rhs, 3, face, 0) + get_cell(right_rhs, 10, 7 + face, 0), Real(0));
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
  EXPECT_EQ(prepare_calls, 0)
      << "invalid topology/geometry must be rejected before component prepare";
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
}

TEST(test_multiblock_interface_scheduler,
     AmrRuntimeInstallsAndInvokesTheSamePairSchedulerAtOneExactPoint) {
  ensure_runtime();
  constexpr int cells = 4;
  AmrBuildParams params;
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
    detail::dispatch_model(scalar_model(), [&](auto model) {
      blocks.push_back(detail::dispatch_amr_block(
          model, "none", "rusanov", layout, name,
          std::vector<double>(static_cast<std::size_t>(cells) * cells, 1.0), true, 1.4, 1, false,
          false, 1));
    });
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

  AxisAlignedInterface route;
  route.identity = "amr.level0.shared_flux";
  route.left_block = 0;
  route.right_block = 1;
  route.level = 0;
  route.left_axis = route.right_axis = InterfaceAxis::X;
  route.left_side = InterfaceSide::High;
  route.right_side = InterfaceSide::Low;
  route.right_component_for_left = {0};
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
  params.mesh.n = 3;
  params.mesh.L = 1.0;
  params.mesh.regrid_every = 0;
  params.poisson.bc = BCRec{};
  const detail::SharedAmrLayout layout = detail::make_shared_amr_layout_levels(params, 1);
  std::vector<AmrRuntimeBlock> blocks;
  for (const char* name : {"a", "b"}) {
    detail::dispatch_model(scalar_model(), [&](auto model) {
      blocks.push_back(detail::dispatch_amr_block(model, "none", "rusanov", layout, name,
                                                  std::vector<double>(9, 1.0), true, 1.4, 1, false,
                                                  false, 1));
    });
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
}
