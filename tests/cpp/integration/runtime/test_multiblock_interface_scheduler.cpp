#include <gtest/gtest.h>

#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/multiblock/interface_flux_scheduler.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pops;
using namespace pops::runtime::multiblock;

namespace {

void ensure_runtime() {
  static Kokkos::ScopeGuard guard;
}

PopsExecutionContextV1 serial_execution() {
  return {sizeof(PopsExecutionContextV1),
          1u,
          "test::nd-interface-execution",
          POPS_MEMORY_SPACE_HOST_V1,
          "pops.runtime-backend-manifest.v1:sha256:nd-interface",
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

template <int Dim>
Extent<Dim> unit_extent() {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = 1;
  return result;
}

template <int Dim>
MultiFab<Dim> make_field(std::vector<Box<Dim>> boxes, int components) {
  mesh::BoxArray<Dim> layout(std::move(boxes));
  const mesh::RankSpace<Dim> ranks(Index<Dim>{}, unit_extent<Dim>());
  std::vector<Index<Dim>> owners(layout.size(), Index<Dim>{});
  auto distribution = mesh::Distribution<Dim>::partitioned(layout, ranks, std::move(owners));
  return MultiFab<Dim>(std::move(layout), std::move(distribution), Index<Dim>{}, components,
                       Extent<Dim>{});
}

template <int Dim>
MultiFab<Dim> make_field(Box<Dim> box, int components) {
  return make_field<Dim>(std::vector<Box<Dim>>{box}, components);
}

template <int Dim>
Geometry<Dim> geometry(Box<Dim> domain, std::array<Real, Dim> lower, std::array<Real, Dim> upper) {
  RealVector<Dim> lo{};
  RealVector<Dim> hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    lo[axis] = lower[axis];
    hi[axis] = upper[axis];
  }
  return Geometry<Dim>::from_bounds(domain, lo, hi);
}

template <int Dim>
std::size_t host_offset(const Fab<Dim>& fab, const Index<Dim>& index, int component) {
  const Box<Dim>& grown = fab.grown_box();
  std::size_t stride = 1;
  std::size_t offset = 0;
  for (int axis = 0; axis < Dim; ++axis) {
    offset += static_cast<std::size_t>(index[axis] - grown.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(grown.length(axis));
  }
  return offset + static_cast<std::size_t>(component) * stride;
}

template <int Dim>
void set_cell(MultiFab<Dim>& field, const Index<Dim>& index, int component, Real value) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    Fab<Dim>& fab = field.fab(local);
    if (!fab.box().contains(index))
      continue;
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    host(host_offset(fab, index, component)) = value;
    fab.copy_from_host(host);
    return;
  }
  throw std::out_of_range("test cell is absent from MultiFab");
}

template <int Dim>
Real get_cell(const MultiFab<Dim>& field, const Index<Dim>& index, int component) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const Fab<Dim>& fab = field.fab(local);
    if (!fab.box().contains(index))
      continue;
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    return host(host_offset(fab, index, component));
  }
  throw std::out_of_range("test cell is absent from MultiFab");
}

template <int Dim>
void authenticate(AxisAlignedInterface<Dim>& route) {
  route.left_trace_projection_identity = route.identity + ".left.trace";
  route.right_trace_projection_identity = route.identity + ".right.trace";
  route.left_trace_provider_identity = "test.cell-average.left";
  route.right_trace_provider_identity = "test.cell-average.right";
  route.left_trace_operation = InterfaceTraceOperation::CellAverage;
  route.right_trace_operation = InterfaceTraceOperation::CellAverage;
  route.left_trace_required_depth = 1;
  route.right_trace_required_depth = 1;
}

BoundaryEvaluationPoint point(int level = 0) {
  return {"test.clock", 1, level, 0, 0, amr::Rational(0, 1), 0.25, 0.0};
}

template <int Dim, class View, class Pointer>
View exact_ranked_field_view(Pointer data, std::size_t components) {
  static_assert(Dim >= 1 && Dim <= 3);
  View view{};
  view.struct_size = sizeof(View);
  view.data = data;
  view.dimension = Dim;
  std::array<std::size_t, 3> extents{1, 1, 1};
  for (std::int32_t axis = 0; axis < Dim; ++axis)
    extents[axis] = static_cast<std::size_t>(axis + 2);
  std::array<std::ptrdiff_t, 3> strides{0, 0, 0};
  std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(components);
  for (std::int32_t axis = Dim; axis-- > 0;) {
    strides[axis] = stride;
    stride *= static_cast<std::ptrdiff_t>(extents[axis]);
  }
  for (std::int32_t axis = 0; axis < 3; ++axis) {
    view.extents[axis] = extents[axis];
    view.axis_strides[axis] = strides[axis];
    view.ghost_lower[axis] = 0;
    view.ghost_upper[axis] = 0;
  }
  view.component_count = components;
  view.component_stride = 1;
  view.centering = POPS_FIELD_CENTERING_CELL_V1;
  view.centering_axes = 0;
  view.scalar_type = POPS_SCALAR_FLOAT64_V1;
  view.memory_space = POPS_MEMORY_SPACE_HOST_V1;
  view.layout_identity = "test::exact-ranked-layout";
  view.patch_identity = "test::exact-ranked-patch";
  view.ownership = POPS_FIELD_OWNERSHIP_RUNTIME_BORROWED_V1;
  return view;
}

template <int Dim>
void prove_exact_ranked_numerical_flux_consumer() {
  constexpr std::size_t components = 2;
  std::size_t points = 1;
  for (std::int32_t axis = 0; axis < Dim; ++axis)
    points *= static_cast<std::size_t>(axis + 2);
  std::vector<double> left(points * components, 1.0);
  std::vector<double> right(points * components, 2.0);
  std::vector<double> normals(points * Dim, 0.0);
  std::vector<double> flux(points * components, 0.0);
  std::vector<double> stability(points, 0.0);
  std::vector<PopsComponentActionV1> actions(points, POPS_COMPONENT_CONTINUE_V1);
  bool invoked = false;
  PopsNumericalFluxApiV1 api{};
  api.evaluate_faces =
      +[](void* state, const PopsNumericalFluxRequestV1*, PopsNumericalFluxResultV1*) {
        *static_cast<bool*>(state) = true;
        return 0;
      };
  const PopsNumericalFluxRequestV1 request{
      sizeof(PopsNumericalFluxRequestV1),
      exact_ranked_field_view<Dim, PopsConstFieldViewV1>(left.data(), components),
      exact_ranked_field_view<Dim, PopsConstFieldViewV1>(right.data(), components),
      exact_ranked_field_view<Dim, PopsConstFieldViewV1>(normals.data(), Dim),
      nullptr,
      {sizeof(PopsLogicalTimeV1), "test.clock", 1, 0, 0, 0, 0, 1, 0.25, 0.0},
      serial_execution()};
  PopsNumericalFluxResultV1 result{
      sizeof(PopsNumericalFluxResultV1),
      exact_ranked_field_view<Dim, PopsFieldViewV1>(flux.data(), components),
      stability.data(),
      actions.data(),
      {}};

  EXPECT_EQ(component::evaluate_faces<Dim>(api, &invoked, request, result), 0);
  EXPECT_TRUE(invoked);

  auto wrong_rank = request;
  wrong_rank.left.dimension = Dim % 3 + 1;
  invoked = false;
  EXPECT_THROW(component::evaluate_faces<Dim>(api, &invoked, wrong_rank, result),
               std::invalid_argument);
  EXPECT_FALSE(invoked);

  auto wrong_stride = request;
  wrong_stride.right.axis_strides[Dim - 1] = 0;
  EXPECT_THROW(component::evaluate_faces<Dim>(api, &invoked, wrong_stride, result),
               std::invalid_argument);
  EXPECT_FALSE(invoked);
}

TEST(test_multiblock_interface_scheduler, NumericalFluxConsumerCarriesExactCompileTimeRank) {
  prove_exact_ranked_numerical_flux_consumer<1>();
  prove_exact_ranked_numerical_flux_consumer<2>();
  prove_exact_ranked_numerical_flux_consumer<3>();
}

TEST(test_multiblock_interface_scheduler, OneDimensionalFaceHasNoSyntheticTangent) {
  ensure_runtime();
  const Box<1> left_box(Index<1>(0), Index<1>(3));
  const Box<1> right_box(Index<1>(10), Index<1>(13));
  MultiFab<1> left = make_field<1>(left_box, 1);
  MultiFab<1> right = make_field<1>(right_box, 1);
  MultiFab<1> left_rhs = make_field<1>(left_box, 1);
  MultiFab<1> right_rhs = make_field<1>(right_box, 1);
  set_cell(left, Index<1>(3), 0, Real(2));
  set_cell(right, Index<1>(10), 0, Real(5));

  AxisAlignedInterface<1> route;
  route.identity = "nd.1d.interface";
  route.left_block = 0;
  route.right_block = 1;
  route.left_axis = route.right_axis = 0;
  route.left_side = InterfaceSide::High;
  route.right_side = InterfaceSide::Low;
  route.right_component_for_left = {0};
  authenticate(route);

  InterfaceFluxScheduler<1> scheduler;
  scheduler.install(route, left, geometry<1>(left_box, {Real(0)}, {Real(1)}), right,
                    geometry<1>(right_box, {Real(1)}, {Real(2)}), serial_execution(),
                    [](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
                      ASSERT_EQ(batch.face_count, 1);
                      EXPECT_EQ(batch.memory_space, POPS_MEMORY_SPACE_HOST_V1);
                      EXPECT_EQ(batch.face_measure, Real(1));
                      EXPECT_EQ(batch.outward_normals[0], Real(1));
                      EXPECT_EQ(batch.left_state[0], Real(2));
                      EXPECT_EQ(batch.right_state[0], Real(5));
                      batch.shared_flux[0] = Real(3);
                    });
  scheduler.apply(point(), std::vector<MultiFab<1>*>{&left, &right},
                  std::vector<MultiFab<1>*>{&left_rhs, &right_rhs});

  EXPECT_EQ(get_cell(left_rhs, Index<1>(3), 0), Real(-12));
  EXPECT_EQ(get_cell(right_rhs, Index<1>(10), 0), Real(12));
  EXPECT_TRUE(scheduler.owns_face(0, 0, 0, InterfaceSide::High));
}

TEST(test_multiblock_interface_scheduler,
     TwoDimensionalReflectionAndComponentPermutationAreConservative) {
  ensure_runtime();
  const Box<2> left_box(Index<2>(0, 0), Index<2>(1, 2));
  const Box<2> right_box(Index<2>(5, 7), Index<2>(6, 9));
  MultiFab<2> left = make_field<2>(left_box, 2);
  MultiFab<2> right = make_field<2>(right_box, 2);
  MultiFab<2> left_rhs = make_field<2>(left_box, 2);
  MultiFab<2> right_rhs = make_field<2>(right_box, 2);
  for (int face = 0; face < 3; ++face) {
    set_cell(left, Index<2>(1, face), 0, Real(10 + face));
    set_cell(left, Index<2>(1, face), 1, Real(20 + face));
    set_cell(right, Index<2>(5, 9 - face), 1, Real(30 + face));
    set_cell(right, Index<2>(5, 9 - face), 0, Real(40 + face));
  }

  AxisAlignedInterface<2> route;
  route.identity = "nd.2d.reflected";
  route.left_block = 0;
  route.right_block = 1;
  route.left_axis = route.right_axis = 0;
  route.left_side = InterfaceSide::High;
  route.right_side = InterfaceSide::Low;
  route.right_component_for_left = {1, 0};
  route.affine_mapping_identity = "nd.2d.reflect-y";
  route.tangential_transform.sign[0] = -1;
  route.tangential_transform.offset[0] = Real(3);
  authenticate(route);

  InterfaceFluxScheduler<2> scheduler;
  scheduler.install(route, left, geometry<2>(left_box, {Real(0), Real(0)}, {Real(1), Real(3)}),
                    right, geometry<2>(right_box, {Real(1), Real(0)}, {Real(2), Real(3)}),
                    serial_execution(),
                    [](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
                      ASSERT_EQ(batch.face_count, 3);
                      ASSERT_EQ(batch.component_count, 2);
                      EXPECT_EQ(batch.face_measure, Real(1));
                      for (int face = 0; face < batch.face_count; ++face) {
                        EXPECT_EQ(batch.outward_normals[2 * face], Real(1));
                        EXPECT_EQ(batch.outward_normals[2 * face + 1], Real(0));
                        EXPECT_EQ(batch.left_state[2 * face], Real(10 + face));
                        EXPECT_EQ(batch.right_state[2 * face], Real(30 + face));
                        EXPECT_EQ(batch.left_state[2 * face + 1], Real(20 + face));
                        EXPECT_EQ(batch.right_state[2 * face + 1], Real(40 + face));
                        batch.shared_flux[2 * face] = Real(face + 1);
                        batch.shared_flux[2 * face + 1] = Real(face + 4);
                      }
                    });
  scheduler.apply(point(), std::vector<MultiFab<2>*>{&left, &right},
                  std::vector<MultiFab<2>*>{&left_rhs, &right_rhs});
  for (int face = 0; face < 3; ++face) {
    EXPECT_EQ(get_cell(left_rhs, Index<2>(1, face), 0), Real(-2 * (face + 1)));
    EXPECT_EQ(get_cell(right_rhs, Index<2>(5, 9 - face), 1), Real(2 * (face + 1)));
    EXPECT_EQ(get_cell(left_rhs, Index<2>(1, face), 1), Real(-2 * (face + 4)));
    EXPECT_EQ(get_cell(right_rhs, Index<2>(5, 9 - face), 0), Real(2 * (face + 4)));
  }
}

TEST(test_multiblock_interface_scheduler,
     ThreeDimensionalTangentPermutationAndReflectionUseCanonicalLeftOrder) {
  ensure_runtime();
  const Box<3> left_box(Index<3>(0, 0, 0), Index<3>(1, 1, 2));
  const Box<3> right_box(Index<3>(5, 7, 9), Index<3>(6, 9, 10));
  MultiFab<3> left = make_field<3>(left_box, 1);
  MultiFab<3> right = make_field<3>(right_box, 1);
  MultiFab<3> left_rhs = make_field<3>(left_box, 1);
  MultiFab<3> right_rhs = make_field<3>(right_box, 1);
  for (int z = 0; z < 3; ++z)
    for (int y = 0; y < 2; ++y) {
      const int face = y + 2 * z;
      set_cell(left, Index<3>(1, y, z), 0, Real(100 + face));
      set_cell(right, Index<3>(5, 9 - z, 9 + y), 0, Real(200 + face));
    }

  AxisAlignedInterface<3> route;
  route.identity = "nd.3d.permuted";
  route.left_block = 0;
  route.right_block = 1;
  route.left_axis = route.right_axis = 0;
  route.left_side = InterfaceSide::High;
  route.right_side = InterfaceSide::Low;
  route.right_component_for_left = {0};
  route.affine_mapping_identity = "nd.3d.swap-yz-reflect-z";
  route.tangential_transform.right_tangent_for_left = {1, 0};
  route.tangential_transform.sign = {1, -1};
  route.tangential_transform.offset = {Real(0), Real(0)};
  authenticate(route);

  InterfaceFluxScheduler<3> scheduler;
  scheduler.install(
      route, left,
      geometry<3>(left_box, {Real(0), Real(0), Real(10)}, {Real(1), Real(2), Real(13)}), right,
      geometry<3>(right_box, {Real(1), Real(-13), Real(0)}, {Real(2), Real(-10), Real(2)}),
      serial_execution(), [](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
        ASSERT_EQ(batch.face_count, 6);
        EXPECT_EQ(batch.face_measure, Real(1));
        for (int face = 0; face < batch.face_count; ++face) {
          EXPECT_EQ(batch.outward_normals[3 * face], Real(1));
          EXPECT_EQ(batch.outward_normals[3 * face + 1], Real(0));
          EXPECT_EQ(batch.outward_normals[3 * face + 2], Real(0));
          EXPECT_EQ(batch.left_state[face], Real(100 + face));
          EXPECT_EQ(batch.right_state[face], Real(200 + face));
          batch.shared_flux[face] = Real(face + 1);
        }
      });
  scheduler.apply(point(), std::vector<MultiFab<3>*>{&left, &right},
                  std::vector<MultiFab<3>*>{&left_rhs, &right_rhs});
  for (int z = 0; z < 3; ++z)
    for (int y = 0; y < 2; ++y) {
      const int face = y + 2 * z;
      EXPECT_EQ(get_cell(left_rhs, Index<3>(1, y, z), 0), Real(-2 * (face + 1)));
      EXPECT_EQ(get_cell(right_rhs, Index<3>(5, 9 - z, 9 + y), 0), Real(2 * (face + 1)));
    }
}

TEST(test_multiblock_interface_scheduler, InvalidThreeDimensionalMapFailsBeforeMutation) {
  ensure_runtime();
  const Box<3> box(Index<3>(0, 0, 0), Index<3>(1, 1, 1));
  MultiFab<3> left = make_field<3>(box, 1);
  MultiFab<3> right = make_field<3>(box, 1);
  AxisAlignedInterface<3> route;
  route.identity = "nd.3d.invalid-map";
  route.left_block = 0;
  route.right_block = 1;
  route.left_axis = route.right_axis = 0;
  route.left_side = InterfaceSide::High;
  route.right_side = InterfaceSide::Low;
  route.right_component_for_left = {0};
  route.affine_mapping_identity = "invalid-duplicate-permutation";
  route.right_normal_translation = Real(1);
  route.tangential_transform.right_tangent_for_left = {0, 0};
  authenticate(route);
  InterfaceFluxScheduler<3> scheduler;
  bool factory_called = false;
  EXPECT_THROW(
      scheduler.install(
          route, left, geometry<3>(box, {Real(0), Real(0), Real(0)}, {Real(1), Real(1), Real(1)}),
          right, geometry<3>(box, {Real(0), Real(0), Real(0)}, {Real(1), Real(1), Real(1)}),
          serial_execution(), InterfaceFluxEvaluatorFactory([&] {
            factory_called = true;
            return InterfaceFluxEvaluator{};
          })),
      std::invalid_argument);
  EXPECT_FALSE(factory_called);
  EXPECT_EQ(scheduler.size(), 0u);
}

TEST(test_multiblock_interface_scheduler, NonFiniteFluxDoesNotScatterEitherSide) {
  ensure_runtime();
  const Box<2> left_box(Index<2>(0, 0), Index<2>(1, 1));
  const Box<2> right_box(Index<2>(2, 0), Index<2>(3, 1));
  MultiFab<2> left = make_field<2>(left_box, 1);
  MultiFab<2> right = make_field<2>(right_box, 1);
  MultiFab<2> left_rhs = make_field<2>(left_box, 1);
  MultiFab<2> right_rhs = make_field<2>(right_box, 1);
  AxisAlignedInterface<2> route;
  route.identity = "nd.nonfinite";
  route.left_block = 0;
  route.right_block = 1;
  route.left_axis = route.right_axis = 0;
  route.left_side = InterfaceSide::High;
  route.right_side = InterfaceSide::Low;
  route.right_component_for_left = {0};
  authenticate(route);
  InterfaceFluxScheduler<2> scheduler;
  scheduler.install(route, left, geometry<2>(left_box, {Real(0), Real(0)}, {Real(1), Real(1)}),
                    right, geometry<2>(right_box, {Real(1), Real(0)}, {Real(2), Real(1)}),
                    serial_execution(),
                    [](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
                      for (int face = 0; face < batch.face_count; ++face)
                        batch.shared_flux[face] = std::numeric_limits<Real>::quiet_NaN();
                    });
  EXPECT_THROW(scheduler.apply(point(), std::vector<MultiFab<2>*>{&left, &right},
                               std::vector<MultiFab<2>*>{&left_rhs, &right_rhs}),
               std::runtime_error);
  EXPECT_EQ(get_cell(left_rhs, Index<2>(1, 0), 0), Real(0));
  EXPECT_EQ(get_cell(right_rhs, Index<2>(2, 0), 0), Real(0));
  EXPECT_EQ(scheduler.evaluation_count(route.identity, 0), 0u);
}

TEST(test_multiblock_interface_scheduler,
     RematerializationRebuildsExactFacePlansWithoutMutatingAcceptedScheduler) {
  ensure_runtime();
  const Box<2> left_box(Index<2>(0, 0), Index<2>(1, 3));
  const Box<2> right_box = left_box;
  MultiFab<2> left = make_field<2>(left_box, 1);
  MultiFab<2> right = make_field<2>(right_box, 1);
  AxisAlignedInterface<2> route;
  route.identity = "nd.rematerialized";
  route.left_block = 0;
  route.right_block = 1;
  route.left_axis = route.right_axis = 0;
  route.left_side = InterfaceSide::High;
  route.right_side = InterfaceSide::Low;
  route.right_component_for_left = {0};
  route.affine_mapping_identity = "nd.periodic-x-translation";
  route.right_normal_translation = Real(1);
  authenticate(route);
  const Geometry<2> level_geometry = geometry<2>(left_box, {Real(0), Real(0)}, {Real(1), Real(4)});
  InterfaceFluxScheduler<2> scheduler;
  int calls = 0;
  scheduler.install(route, left, level_geometry, right, level_geometry, serial_execution(),
                    [&](const BoundaryEvaluationPoint&, const InterfaceFluxBatch& batch) {
                      ++calls;
                      for (int face = 0; face < batch.face_count; ++face)
                        batch.shared_flux[face] = Real(face + 1);
                    });

  MultiFab<2> replacement_left = make_field<2>(
      {Box<2>(Index<2>(0, 0), Index<2>(1, 1)), Box<2>(Index<2>(0, 2), Index<2>(1, 3))}, 1);
  MultiFab<2> replacement_right = make_field<2>(
      {Box<2>(Index<2>(0, 0), Index<2>(1, 1)), Box<2>(Index<2>(0, 2), Index<2>(1, 3))}, 1);
  InterfaceFluxScheduler<2> replacement = scheduler.rematerialized(
      1,
      [&](std::size_t block, int level) -> MultiFab<2>& {
        EXPECT_EQ(level, 0);
        return block == 0 ? replacement_left : replacement_right;
      },
      [&](int level) {
        EXPECT_EQ(level, 0);
        return level_geometry;
      });
  MultiFab<2> replacement_left_rhs = make_field<2>(replacement_left.layout().boxes(), 1);
  MultiFab<2> replacement_right_rhs = make_field<2>(replacement_right.layout().boxes(), 1);
  replacement.apply(point(), std::vector<MultiFab<2>*>{&replacement_left, &replacement_right},
                    std::vector<MultiFab<2>*>{&replacement_left_rhs, &replacement_right_rhs});
  EXPECT_EQ(scheduler.size(), 1u);
  EXPECT_EQ(scheduler.evaluation_count(route.identity, 0), 0u);
  EXPECT_EQ(replacement.size(), 1u);
  EXPECT_EQ(replacement.evaluation_count(route.identity, 0), 1u);
  EXPECT_EQ(calls, 1);
  for (int face = 0; face < 4; ++face) {
    EXPECT_EQ(get_cell(replacement_left_rhs, Index<2>(1, face), 0), Real(-2 * (face + 1)));
    EXPECT_EQ(get_cell(replacement_right_rhs, Index<2>(0, face), 0), Real(2 * (face + 1)));
  }
}

}  // namespace
