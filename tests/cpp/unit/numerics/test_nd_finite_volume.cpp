#include <gtest/gtest.h>

#include <pops/numerics/spatial/nd/finite_volume.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using namespace pops;

namespace {

template <int Dim>
Box<Dim> make_box(const std::array<int, Dim>& extents) {
  Index<Dim> lower{};
  Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = extents[axis] - 1;
  return {lower, upper};
}

template <int Dim, class Function>
void for_each_index(const Box<Dim>& box, Function&& function) {
  const std::int64_t count = box.numPts();
  for (std::int64_t linear = 0; linear < count; ++linear) {
    std::int64_t remaining = linear;
    Index<Dim> index{};
    for (int axis = 0; axis < Dim; ++axis) {
      index[axis] = box.lo[axis] + static_cast<int>(remaining % box.length(axis));
      remaining /= box.length(axis);
    }
    function(index);
  }
}

template <int Dim, int N>
class HostFaceStorage {
 public:
  explicit HostFaceStorage(Box<Dim> cells) {
    view_.cells = cells;
    view_.ncomp = N;
    for (int axis = 0; axis < Dim; ++axis) {
      boxes_[axis] = nd::face_box(cells, axis);
      const std::int64_t count = boxes_[axis].numPts();
      values_[axis].resize(static_cast<std::size_t>(count) * N);
      FieldView<const Real, Dim> axis_view{};
      axis_view.data = values_[axis].data();
      axis_view.origin = boxes_[axis].lo;
      axis_view.extents = boxes_[axis].extent();
      std::int64_t stride = 1;
      for (int direction = 0; direction < Dim; ++direction) {
        axis_view.strides[direction] = stride;
        stride *= axis_view.extents[direction];
      }
      axis_view.ncomp = N;
      axis_view.component_stride = count;
      view_.axes[axis] = axis_view;
    }
  }

  const Box<Dim>& box(int axis) const { return boxes_[axis]; }
  const nd::FaceFieldView<const Real, Dim>& view() const { return view_; }

  void set(int axis, const Index<Dim>& index, int component, Real value) {
    values_[axis][offset(axis, index, component)] = value;
  }

  void fill(Real value) {
    for (auto& axis : values_)
      std::fill(axis.begin(), axis.end(), value);
  }

 private:
  std::size_t offset(int axis, const Index<Dim>& index, int component) const {
    std::int64_t linear = 0;
    std::int64_t stride = 1;
    for (int direction = 0; direction < Dim; ++direction) {
      linear += static_cast<std::int64_t>(index[direction] - boxes_[axis].lo[direction]) * stride;
      stride *= boxes_[axis].length(direction);
    }
    return static_cast<std::size_t>(component * boxes_[axis].numPts() + linear);
  }

  std::array<Box<Dim>, Dim> boxes_{};
  std::array<std::vector<Real>, Dim> values_{};
  nd::FaceFieldView<const Real, Dim> view_{};
};

template <int Axis, int Dim>
void check_scalar_axis(const nd::ScalarAdvection<Dim>& model) {
  using State = typename nd::ScalarAdvection<Dim>::State;
  const State left{Real(1.25)};
  const State right{Real(2.75)};
  const Real speed = model.velocity()[Axis];
  const Real expected = speed * (speed >= Real(0) ? left[0] : right[0]);

  const auto rusanov = nd::evaluate_axis_flux<Axis>(RusanovFlux{}, model, left, right);
  ASSERT_TRUE(rusanov.succeeded());
  EXPECT_EQ(rusanov.requested_solver, RiemannSolverId::kRusanov);
  EXPECT_EQ(rusanov.used_solver, RiemannSolverId::kRusanov);
  EXPECT_NEAR(rusanov.checked_density().value[0], expected, Real(2e-14));

  const auto hll = nd::evaluate_axis_flux<Axis>(HLLFlux{}, model, left, right);
  ASSERT_TRUE(hll.succeeded());
  EXPECT_EQ(hll.requested_solver, RiemannSolverId::kHll);
  EXPECT_NEAR(hll.checked_density().value[0], expected, Real(2e-14));

  if constexpr (Axis + 1 < Dim)
    check_scalar_axis<Axis + 1>(model);
}

template <int Dim>
void check_scalar_law() {
  RealVector<Dim> velocity{};
  for (int axis = 0; axis < Dim; ++axis)
    velocity[axis] = axis % 2 == 0 ? Real(0.35 + 0.1 * axis) : Real(-0.45 - 0.1 * axis);
  check_scalar_axis<0>(nd::ScalarAdvection<Dim>::prepare(velocity));
}

template <int Axis, int Dim>
void check_euler_axis(const nd::IdealGasEuler<Dim>& model,
                      const typename nd::IdealGasEuler<Dim>::State& conservative,
                      const typename nd::IdealGasEuler<Dim>::Primitive& primitive) {
  using Schema = nd::EulerStateSchema<Dim>;
  const auto flux = model.template flux<Axis>(conservative);
  const Real normal_velocity = primitive[Schema::template velocity<Axis>];
  EXPECT_NEAR(flux[Schema::density], conservative[Schema::template momentum<Axis>], Real(2e-14));
  for (int momentum_axis = 0; momentum_axis < Dim; ++momentum_axis) {
    Real expected = conservative[momentum_axis + 1] * normal_velocity;
    if (momentum_axis == Axis)
      expected += primitive[Schema::pressure];
    EXPECT_NEAR(flux[momentum_axis + 1], expected, Real(2e-14));
  }
  EXPECT_NEAR(flux[Schema::energy],
              (conservative[Schema::energy] + primitive[Schema::pressure]) * normal_velocity,
              Real(2e-14));

  const auto rusanov =
      nd::evaluate_axis_flux<Axis>(RusanovFlux{}, model, conservative, conservative);
  const auto hll = nd::evaluate_axis_flux<Axis>(HLLFlux{}, model, conservative, conservative);
  const auto hllc = nd::evaluate_axis_flux<Axis>(HLLCFlux{}, model, conservative, conservative);
  const auto roe = nd::evaluate_axis_flux<Axis>(RoeFlux{}, model, conservative, conservative);
  ASSERT_TRUE(rusanov.succeeded());
  ASSERT_TRUE(hll.succeeded());
  ASSERT_TRUE(hllc.succeeded());
  ASSERT_TRUE(roe.succeeded());
  for (int component = 0; component < Schema::nvars; ++component) {
    EXPECT_NEAR(rusanov.checked_density().value[component], flux[component], Real(4e-14));
    EXPECT_NEAR(hll.checked_density().value[component], flux[component], Real(4e-14));
    EXPECT_NEAR(hllc.checked_density().value[component], flux[component], Real(4e-14));
    EXPECT_NEAR(roe.checked_density().value[component], flux[component], Real(4e-13));
  }

  if constexpr (Axis + 1 < Dim)
    check_euler_axis<Axis + 1>(model, conservative, primitive);
}

template <int Axis, int Dim>
void check_euler_contact_and_shear_axis(const nd::IdealGasEuler<Dim>& model) {
  using Schema = nd::EulerStateSchema<Dim>;
  using Primitive = typename nd::IdealGasEuler<Dim>::Primitive;
  for (const Real normal_velocity : std::array<Real, 2>{Real(0.37), Real(-0.29)}) {
    Primitive left{};
    Primitive right{};
    left[Schema::density] = Real(1.4);
    right[Schema::density] = Real(0.8);
    left[Schema::pressure] = right[Schema::pressure] = Real(1.1);
    for (int velocity_axis = 0; velocity_axis < Dim; ++velocity_axis) {
      left[velocity_axis + 1] = Real(0.12 * (velocity_axis + 1));
      right[velocity_axis + 1] = Real(-0.09 * (velocity_axis + 1));
    }
    left[Schema::template velocity<Axis>] = normal_velocity;
    right[Schema::template velocity<Axis>] = normal_velocity;
    const auto left_state = model.make_conservative(left);
    const auto right_state = model.make_conservative(right);
    ASSERT_TRUE(left_state.succeeded());
    ASSERT_TRUE(right_state.succeeded());

    const auto hllc =
        nd::evaluate_axis_flux<Axis>(HLLCFlux{}, model, left_state.value, right_state.value);
    const auto roe =
        nd::evaluate_axis_flux<Axis>(RoeFlux{}, model, left_state.value, right_state.value);
    ASSERT_TRUE(hllc.succeeded());
    ASSERT_TRUE(roe.succeeded());
    const auto expected =
        model.template flux<Axis>(normal_velocity > Real(0) ? left_state.value : right_state.value);
    for (int component = 0; component < Schema::nvars; ++component) {
      EXPECT_NEAR(hllc.checked_density().value[component], expected[component], Real(2e-12));
      EXPECT_NEAR(roe.checked_density().value[component], expected[component], Real(2e-12));
    }
  }

  if constexpr (Axis + 1 < Dim)
    check_euler_contact_and_shear_axis<Axis + 1>(model);
}

template <int Dim>
void check_euler_law() {
  using Schema = nd::EulerStateSchema<Dim>;
  const auto model = nd::IdealGasEuler<Dim>::prepare(Real(1.4));
  typename nd::IdealGasEuler<Dim>::Primitive primitive{};
  primitive[Schema::density] = Real(1.25);
  primitive[Schema::pressure] = Real(0.9);
  for (int axis = 0; axis < Dim; ++axis)
    primitive[axis + 1] = axis % 2 == 0 ? Real(0.2 * (axis + 1)) : Real(-0.15 * (axis + 1));
  const auto conservative = model.make_conservative(primitive);
  ASSERT_TRUE(conservative.succeeded());
  const auto recovered = model.recover(conservative.value);
  ASSERT_TRUE(recovered.succeeded());
  for (int component = 0; component < Schema::nvars; ++component)
    EXPECT_NEAR(recovered.value[component], primitive[component], Real(3e-14));
  check_euler_axis<0>(model, conservative.value, primitive);
  check_euler_contact_and_shear_axis<0>(model);
}

template <int Axis, int Dim, class Model, class Metric>
void fill_constant_physical_flux(HostFaceStorage<Dim, Model::n_vars>& faces, const Model& model,
                                 const typename Model::State& state, const Metric& metric,
                                 const Box<Dim>& cells) {
  for_each_index(faces.box(Axis), [&](const Index<Dim>& face) {
    Index<Dim> left_cell = face;
    if (face[Axis] == cells.lo[Axis])
      left_cell[Axis] = cells.hi[Axis];
    else
      --left_cell[Axis];
    const auto evaluation = nd::evaluate_metric_face_flux<Axis, MetricFaceSide::Upper>(
        RusanovFlux{}, model, state, state, metric, left_cell);
    ASSERT_TRUE(evaluation.succeeded());
    const FaceContext context =
        nd::metric_face_context<Axis, MetricFaceSide::Upper>(metric, left_cell);
    const auto integrated = apply_face_measure(evaluation.checked_density(), context);
    for (int component = 0; component < Model::n_vars; ++component)
      faces.set(Axis, face, component, integrated.value[component]);
  });
  if constexpr (Axis + 1 < Dim)
    fill_constant_physical_flux<Axis + 1>(faces, model, state, metric, cells);
}

template <int Dim>
void check_metric_cfl_and_divergence() {
  std::array<int, Dim> extents{};
  RealVector<Dim> lengths{};
  RealVector<Dim> origin{};
  RealVector<Dim> velocity{};
  for (int axis = 0; axis < Dim; ++axis) {
    extents[axis] = 4 + axis;
    lengths[axis] = Real(1.5 + 0.5 * axis);
    velocity[axis] = axis % 2 == 0 ? Real(0.3 + 0.1 * axis) : Real(-0.4 - 0.1 * axis);
  }
  const Box<Dim> cells = make_box<Dim>(extents);
  const auto map = CartesianCoordinateMap<Dim>::make(origin, lengths);
  const auto metric = prepare_metric_provider(cells, map);
  const auto model = nd::ScalarAdvection<Dim>::prepare(velocity);
  const typename nd::ScalarAdvection<Dim>::State state{Real(1.7)};
  Index<Dim> sample{};
  for (int axis = 0; axis < Dim; ++axis)
    sample[axis] = extents[axis] / 2;

  const auto cfl = nd::cell_cfl_bound<Dim>(model, state, metric, sample);
  ASSERT_TRUE(cfl.succeeded());
  Real expected_inverse_dt = Real(0);
  for (int axis = 0; axis < Dim; ++axis)
    expected_inverse_dt +=
        std::abs(velocity[axis]) / (lengths[axis] / static_cast<Real>(extents[axis]));
  EXPECT_NEAR(cfl.inverse_dt, expected_inverse_dt, Real(2e-13));
  const auto step = nd::cell_time_step<Dim>(model, state, metric, sample, Real(0.4));
  ASSERT_TRUE(step.succeeded());
  EXPECT_NEAR(step.value, Real(0.4) / expected_inverse_dt, Real(2e-14));

  HostFaceStorage<Dim, 1> faces(cells);
  fill_constant_physical_flux<0>(faces, model, state, metric, cells);
  for_each_index(cells, [&](const Index<Dim>& cell) {
    const auto residual = nd::conservative_residual<1>(metric, faces.view(), cell);
    ASSERT_TRUE(residual.succeeded());
    EXPECT_NEAR(residual.value[0], Real(0), Real(3e-14));
  });

  constexpr Real two_pi = Real(6.283185307179586476925286766559);
  for (int axis = 0; axis < Dim; ++axis) {
    for_each_index(faces.box(axis), [&](const Index<Dim>& face) {
      const int periodic_coordinate =
          (face[axis] - cells.lo[axis]) % static_cast<int>(cells.length(axis));
      Real value = std::sin(two_pi * static_cast<Real>(periodic_coordinate) /
                            static_cast<Real>(cells.length(axis)));
      for (int tangent = 0; tangent < Dim; ++tangent)
        if (tangent != axis)
          value += Real(0.03 * (tangent + 1)) * static_cast<Real>(face[tangent]);
      faces.set(axis, face, 0, value);
    });
  }
  Real global_residual = Real(0);
  for_each_index(cells, [&](const Index<Dim>& cell) {
    const auto residual = nd::conservative_residual<1>(metric, faces.view(), cell);
    ASSERT_TRUE(residual.succeeded());
    global_residual += residual.value[0] * metric.cell_measure(cell);
  });
  EXPECT_NEAR(global_residual, Real(0), Real(3e-13));
}

}  // namespace

TEST(test_nd_finite_volume, state_schemas_are_axis_indexed_at_compile_time) {
  static_assert(nd::EulerStateSchema<1>::density == 0);
  static_assert(nd::EulerStateSchema<1>::energy == 2);
  static_assert(nd::EulerStateSchema<2>::template momentum<1> == 2);
  static_assert(nd::EulerStateSchema<3>::template momentum<2> == 3);
  static_assert(nd::EulerStateSchema<3>::template tangent_momentum<1, 0> == 1);
  static_assert(nd::EulerStateSchema<3>::template tangent_momentum<1, 1> == 3);
  constexpr auto tangents = nd::EulerStateSchema<3>::tangent_axes<1>();
  static_assert(tangents[0] == 0 && tangents[1] == 2);
  SUCCEED();
}

TEST(test_nd_finite_volume, scalar_advection_uses_the_same_rusanov_and_hll_templates_in_1d_2d_3d) {
  check_scalar_law<1>();
  check_scalar_law<2>();
  check_scalar_law<3>();
}

TEST(test_nd_finite_volume, euler_dim_plus_two_flux_and_fallible_recovery_work_in_1d_2d_3d) {
  check_euler_law<1>();
  check_euler_law<2>();
  check_euler_law<3>();
}

TEST(test_nd_finite_volume, euler_flux_is_invariant_under_an_axis_permutation) {
  using Schema = nd::EulerStateSchema<3>;
  constexpr std::array<int, 3> permutation{2, 0, 1};
  const auto model = nd::IdealGasEuler<3>::prepare(Real(1.4));
  nd::IdealGasEuler<3>::Primitive original{};
  original[Schema::density] = Real(1.3);
  original[1] = Real(0.2);
  original[2] = Real(-0.4);
  original[3] = Real(0.7);
  original[Schema::pressure] = Real(0.8);
  nd::IdealGasEuler<3>::Primitive permuted{};
  permuted[Schema::density] = original[Schema::density];
  permuted[Schema::pressure] = original[Schema::pressure];
  for (int axis = 0; axis < 3; ++axis)
    permuted[axis + 1] = original[permutation[axis] + 1];
  const auto original_state = model.make_conservative(original);
  const auto permuted_state = model.make_conservative(permuted);
  ASSERT_TRUE(original_state.succeeded());
  ASSERT_TRUE(permuted_state.succeeded());
  const auto original_flux = model.flux<2>(original_state.value);
  const auto permuted_flux = model.flux<0>(permuted_state.value);
  EXPECT_NEAR(permuted_flux[Schema::density], original_flux[Schema::density], Real(2e-14));
  for (int axis = 0; axis < 3; ++axis)
    EXPECT_NEAR(permuted_flux[axis + 1], original_flux[permutation[axis] + 1], Real(2e-14));
  EXPECT_NEAR(permuted_flux[Schema::energy], original_flux[Schema::energy], Real(2e-14));
}

TEST(test_nd_finite_volume, prepared_metric_drives_cfl_and_conservative_face_divergence) {
  check_metric_cfl_and_divergence<1>();
  check_metric_cfl_and_divergence<2>();
  check_metric_cfl_and_divergence<3>();
}

TEST(test_nd_finite_volume, embedded_axis_permutation_does_not_change_logical_cfl) {
  const Box<3> cells = make_box<3>({4, 5, 6});
  const RealVector<3> lengths{Real(2), Real(3), Real(4)};
  const auto canonical =
      prepare_metric_provider(cells, CartesianCoordinateMap<3>::make(RealVector<3>{}, lengths));
  const auto permuted = prepare_metric_provider(
      cells, CartesianCoordinateMap<3>::make(RealVector<3>{}, lengths, {2, 0, 1}, {-1, 1, -1}));
  const auto model =
      nd::ScalarAdvection<3>::prepare(RealVector<3>{Real(0.3), Real(-0.5), Real(0.7)});
  const nd::ScalarAdvection<3>::State state{Real(1)};
  const Index<3> cell{1, 2, 3};
  const auto left = nd::cell_cfl_bound<3>(model, state, canonical, cell);
  const auto right = nd::cell_cfl_bound<3>(model, state, permuted, cell);
  ASSERT_TRUE(left.succeeded());
  ASSERT_TRUE(right.succeeded());
  EXPECT_NEAR(left.inverse_dt, right.inverse_dt, Real(2e-14));
}

TEST(test_nd_finite_volume, face_field_owns_one_axis_static_fab_per_direction) {
  const Box<3> cells = make_box<3>({3, 4, 5});
  nd::FaceField<3> faces(cells, nd::EulerStateSchema<3>::nvars);
  EXPECT_EQ(faces.ncomp(), 5);
  EXPECT_EQ(faces.field<0>().box(), nd::face_box<0>(cells));
  EXPECT_EQ(faces.field<1>().box(), nd::face_box<1>(cells));
  EXPECT_EQ(faces.field<2>().box(), nd::face_box<2>(cells));
  EXPECT_EQ(faces.view().ncomp, 5);
  const auto metric = prepare_metric_provider(
      cells, CartesianCoordinateMap<3>::make(RealVector<3>{}, RealVector<3>{1, 1, 1}));
  EXPECT_TRUE(nd::conservative_residual<5>(metric, faces.view(), Index<3>{}).succeeded());
}

TEST(test_nd_finite_volume, inadmissible_states_and_invalid_metric_inputs_fail_closed) {
  EXPECT_THROW((void)nd::IdealGasEuler<3>::prepare(Real(1)), std::invalid_argument);
  EXPECT_THROW((void)nd::ScalarAdvection<2>::prepare(
                   RealVector<2>{Real(0), std::numeric_limits<Real>::infinity()}),
               std::invalid_argument);

  using Schema = nd::EulerStateSchema<3>;
  const auto model = nd::IdealGasEuler<3>::prepare(Real(1.4));
  nd::IdealGasEuler<3>::Primitive primitive{};
  primitive[Schema::density] = Real(1);
  primitive[Schema::pressure] = Real(1);
  const auto valid = model.make_conservative(primitive);
  ASSERT_TRUE(valid.succeeded());

  auto vacuum = valid.value;
  vacuum[Schema::density] = Real(0);
  EXPECT_EQ(model.recover(vacuum).status, nd::StateConversionStatus::NonPositiveDensity);
  auto cold = valid.value;
  cold[Schema::energy] = Real(-1);
  EXPECT_EQ(model.recover(cold).status, nd::StateConversionStatus::NonPositivePressure);
  auto nonfinite = valid.value;
  nonfinite[1] = std::numeric_limits<Real>::quiet_NaN();
  EXPECT_EQ(model.recover(nonfinite).status, nd::StateConversionStatus::NonFiniteState);

  const auto refused = nd::evaluate_axis_flux<0>(RusanovFlux{}, model, cold, valid.value);
  EXPECT_FALSE(refused.succeeded());
  EXPECT_EQ(refused.status, EvaluationStatus::kReject);
  EXPECT_EQ(refused.requested_solver, RiemannSolverId::kRusanov);
  EXPECT_EQ(refused.used_solver, RiemannSolverId::kReject);

  const Box<3> cells = make_box<3>({2, 2, 2});
  const auto metric = prepare_metric_provider(
      cells, CartesianCoordinateMap<3>::make(RealVector<3>{}, RealVector<3>{1, 1, 1}));
  EXPECT_EQ(nd::cell_cfl_bound<3>(model, cold, metric, Index<3>{}).status,
            nd::FiniteVolumeStatus::NonPositivePressure);
  EXPECT_EQ(nd::cell_time_step<3>(model, valid.value, metric, Index<3>{}, Real(0)).status,
            nd::FiniteVolumeStatus::InvalidCourantNumber);
  EXPECT_FALSE(
      nd::evaluate_axis_flux<0>(RusanovFlux{}, model, valid.value, valid.value, Real(0), Real(1))
          .succeeded());

  HostFaceStorage<3, 5> faces(cells);
  auto forged = faces.view();
  forged.ncomp = 4;
  EXPECT_EQ(nd::conservative_residual<5>(metric, forged, Index<3>{}).status,
            nd::FiniteVolumeStatus::InvalidFaceField);

  const Box<3> other_cells = make_box<3>({1, 2, 2});
  const auto other_metric = prepare_metric_provider(
      other_cells, CartesianCoordinateMap<3>::make(RealVector<3>{}, RealVector<3>{1, 1, 1}));
  EXPECT_EQ(nd::conservative_residual<5>(other_metric, faces.view(), Index<3>{}).status,
            nd::FiniteVolumeStatus::InvalidMetric);
  EXPECT_FALSE((nd::evaluate_metric_face_flux<0, MetricFaceSide::Upper>(
                    RusanovFlux{}, model, valid.value, valid.value, metric, Index<3>{2, 0, 0})
                    .succeeded()));
}
