#include <gtest/gtest.h>

#include <pops/numerics/spatial/operators/cartesian_operator.hpp>
#include <pops/numerics/spatial/operators/masked_operator.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

using namespace pops;

namespace {

template <int Dim, class Function>
void for_each_host_index(const Box<Dim>& box, Function&& function) {
  for (std::int64_t linear = 0; linear < box.numPts(); ++linear) {
    std::int64_t remaining = linear;
    Index<Dim> index{};
    for (int axis = 0; axis < Dim; ++axis) {
      index[axis] = box.lo[axis] + static_cast<int>(remaining % box.length(axis));
      remaining /= box.length(axis);
    }
    function(index);
  }
}

template <int Dim>
std::size_t host_offset(const Box<Dim>& storage, const Index<Dim>& index, int component) {
  std::int64_t linear = 0;
  std::int64_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    linear += static_cast<std::int64_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= storage.length(axis);
  }
  return static_cast<std::size_t>(component * storage.numPts() + linear);
}

template <int Dim>
Index<Dim> periodic_image(Index<Dim> index, const Box<Dim>& valid) {
  for (int axis = 0; axis < Dim; ++axis) {
    const int extent = static_cast<int>(valid.length(axis));
    while (index[axis] < valid.lo[axis])
      index[axis] += extent;
    while (index[axis] > valid.hi[axis])
      index[axis] -= extent;
  }
  return index;
}

template <int Dim, class Function>
void fill_periodic(Fab<Dim>& field, Function&& value) {
  auto host = field.create_host_mirror();
  const Box<Dim> storage = field.grown_box();
  for_each_host_index(storage, [&](const Index<Dim>& index) {
    const Index<Dim> wrapped = periodic_image(index, field.box());
    for (int component = 0; component < field.ncomp(); ++component)
      host(host_offset(storage, index, component)) = value(wrapped, component);
  });
  field.copy_from_host(host);
}

template <int Dim>
std::vector<Real> valid_values(const Fab<Dim>& field, int component = 0) {
  auto host = field.create_host_mirror();
  field.copy_to_host(host);
  std::vector<Real> result;
  result.reserve(static_cast<std::size_t>(field.box().numPts()));
  for_each_host_index(field.box(), [&](const Index<Dim>& index) {
    result.push_back(host(host_offset(field.grown_box(), index, component)));
  });
  return result;
}

template <int Dim>
Geometry<Dim> unit_geometry(const Box<Dim>& domain) {
  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = Real(1 + axis);
  return Geometry<Dim>::from_bounds(domain, lower, upper);
}

template <int Dim>
Extent<Dim> uniform_extent(int value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
class DiffusiveStaticScalar : public nd::ScalarAdvection<Dim> {
 public:
  using State = typename nd::ScalarAdvection<Dim>::State;

  explicit DiffusiveStaticScalar(Real diffusivity) : diffusivity_(diffusivity) {}

  POPS_HD Real diffusivity() const { return diffusivity_; }

 private:
  Real diffusivity_ = Real(0);
};

static_assert(DiffusiveModel<DiffusiveStaticScalar<1>>);
static_assert(DiffusiveModel<DiffusiveStaticScalar<2>>);
static_assert(DiffusiveModel<DiffusiveStaticScalar<3>>);

template <int Dim>
void check_centered_fickian_flux() {
  constexpr Real diffusivity = Real(0.125);
  const Box<Dim> domain = Box<Dim>::from_extents(uniform_extent<Dim>(7));
  const auto geometry = unit_geometry(domain);
  const auto op = nd::prepare_cartesian_operator<Dim>(
      geometry, DiffusiveStaticScalar<Dim>{diffusivity}, NoSlope{}, RusanovFlux{});
  Fab<Dim> state(domain, 1, uniform_extent<Dim>(NoSlope::n_ghost));
  fill_periodic(state, [](const Index<Dim>& index, int) {
    Real value = Real(0);
    for (int axis = 0; axis < Dim; ++axis)
      value += Real(index[axis] * index[axis]);
    return value;
  });

  Fab<Dim> residual(domain, 1);
  op.assemble_residual(state, residual);
  auto host = residual.create_host_mirror();
  residual.copy_to_host(host);
  Index<Dim> center{};
  Real expected = Real(0);
  for (int axis = 0; axis < Dim; ++axis) {
    center[axis] = 3;
    const Real spacing = geometry.spacing(axis);
    expected += Real(2) * diffusivity / (spacing * spacing);
  }
  EXPECT_NEAR(host(host_offset(residual.grown_box(), center, 0)), expected, Real(2e-12));
}

template <int Dim>
void check_diffusive_cfl_bound() {
  constexpr Real diffusivity = Real(0.125);
  const Box<Dim> domain = Box<Dim>::from_extents(uniform_extent<Dim>(7));
  const auto geometry = unit_geometry(domain);
  const DiffusiveStaticScalar<Dim> model{diffusivity};
  const auto op = nd::prepare_cartesian_operator<Dim>(geometry, model, NoSlope{}, RusanovFlux{});
  const auto& metric = op.metric();
  typename DiffusiveStaticScalar<Dim>::State state{Real(1)};
  const Index<Dim> cell{};

  Real expected_inverse_dt = Real(0);
  for (int axis = 0; axis < Dim; ++axis) {
    const Real spacing = geometry.spacing(axis);
    expected_inverse_dt += Real(2) * diffusivity / (spacing * spacing);
  }
  const auto bound = nd::cell_cfl_bound<Dim>(model, state, metric, cell);
  ASSERT_TRUE(bound.succeeded());
  EXPECT_NEAR(bound.inverse_dt, expected_inverse_dt, Real(2e-12));

  const auto frequency = nd::cell_parabolic_frequency<Dim>(model, metric, cell);
  ASSERT_TRUE(frequency.succeeded());
  EXPECT_NEAR(frequency.value, expected_inverse_dt, Real(2e-12));
  const auto parabolic = nd::cell_parabolic_time_step<Dim>(model, metric, cell);
  ASSERT_TRUE(parabolic.succeeded());
  EXPECT_NEAR(parabolic.value, Real(1) / expected_inverse_dt, Real(2e-12));
  const auto stepped = nd::cell_time_step<Dim>(model, state, metric, cell, Real(0.4));
  ASSERT_TRUE(stepped.succeeded());
  EXPECT_NEAR(stepped.value, Real(0.4) / expected_inverse_dt, Real(2e-12));

  const DiffusiveStaticScalar<Dim> invalid{std::numeric_limits<Real>::quiet_NaN()};
  EXPECT_EQ(nd::cell_cfl_bound<Dim>(invalid, state, metric, cell).status,
            nd::FiniteVolumeStatus::InvalidWaveSpeed);
}

template <int Dim>
void check_masked_diffusion_is_refused() {
  constexpr Real diffusivity = Real(0.125);
  const Box<Dim> domain = Box<Dim>::from_extents(uniform_extent<Dim>(7));
  const auto geometry = unit_geometry(domain);
  const auto regular = nd::prepare_cartesian_operator<Dim>(
      geometry, DiffusiveStaticScalar<Dim>{diffusivity}, NoSlope{}, RusanovFlux{});
  const auto op = nd::prepare_masked_cartesian_operator<Dim>(
      DiffusiveStaticScalar<Dim>{diffusivity}, regular.metric(), NoSlope{}, RusanovFlux{});
  Fab<Dim> state(domain, 1, uniform_extent<Dim>(NoSlope::n_ghost));
  fill_periodic(state, [](const Index<Dim>& index, int) {
    Real value = Real(0);
    for (int axis = 0; axis < Dim; ++axis)
      value += Real(index[axis] * index[axis]);
    return value;
  });
  Fab<Dim> active(domain, 1, uniform_extent<Dim>(1));
  fill_periodic(active,
                [](const Index<Dim>& index, int) { return index[0] == 0 ? Real(0) : Real(1); });
  Fab<Dim> residual(domain, 1);
  EXPECT_THROW(op.assemble_residual(state, active, residual), std::invalid_argument);
}

template <int Dim>
void check_constant_and_conservation(const Extent<Dim>& extents) {
  const Box<Dim> domain = Box<Dim>::from_extents(extents);
  RealVector<Dim> velocity{};
  for (int axis = 0; axis < Dim; ++axis)
    velocity[axis] = axis % 2 == 0 ? Real(0.7 + 0.1 * axis) : Real(-0.4 - 0.1 * axis);
  const auto geometry = unit_geometry(domain);
  const auto model = nd::ScalarAdvection<Dim>::prepare(velocity);
  const auto op = nd::prepare_cartesian_operator<Dim>(geometry, model, VanLeer{}, HLLFlux{});

  Fab<Dim> state(domain, 1, uniform_extent<Dim>(VanLeer::n_ghost));
  Fab<Dim> residual(domain, 1);
  fill_periodic(state, [](const Index<Dim>&, int) { return Real(2.5); });
  op.assemble_residual(state, residual);
  for (const Real value : valid_values(residual))
    EXPECT_EQ(value, Real(0));

  constexpr Real two_pi = Real(6.283185307179586476925286766559);
  fill_periodic(state, [&](const Index<Dim>& index, int) {
    Real value = Real(0.75);
    for (int axis = 0; axis < Dim; ++axis)
      value += (Real(0.1) + Real(0.03) * axis) *
               std::sin(two_pi * (Real(index[axis] - domain.lo[axis]) + Real(0.5)) /
                        static_cast<Real>(domain.length(axis)));
    return value;
  });
  op.assemble_residual(state, residual);
  const auto values = valid_values(residual);
  EXPECT_TRUE(std::any_of(values.begin(), values.end(),
                          [](Real value) { return std::abs(value) > Real(1e-8); }));
  const Real volume = op.metric().cell_measure(domain.lo);
  EXPECT_NEAR(std::accumulate(values.begin(), values.end(), Real(0)) * volume, Real(0),
              Real(4e-13));
}

template <int Axis, int Dim>
void check_constant_face_axis(const nd::FaceField<Dim>& fluxes,
                              const std::array<Real, Dim>& expected) {
  const auto& field = fluxes.template field<Axis>();
  for (const Real value : valid_values(field))
    EXPECT_NEAR(value, expected[Axis], Real(4e-14));
  if constexpr (Axis + 1 < Dim)
    check_constant_face_axis<Axis + 1, Dim>(fluxes, expected);
}

struct TwoComponentEllipticModel {
  using State = StateVec<2>;
  static constexpr int n_vars = 2;

  POPS_HD Real elliptic_rhs(const State& state) const { return state[0] + Real(2) * state[1]; }
};

template <int Dim>
void check_ranked_state_access() {
  const Box<Dim> box = Box<Dim>::from_extents(uniform_extent<Dim>(2));
  constexpr int ncomp = Dim < 2 ? 2 : Dim;
  Fab<Dim> field(box, ncomp);
  fill_periodic(field, [](const Index<Dim>& index, int component) {
    Real value = Real(10 * component);
    for (int axis = 0; axis < Dim; ++axis)
      value += Real(index[axis]);
    return value;
  });
  const Index<Dim> sample = box.hi;
  const auto view = static_cast<const Fab<Dim>&>(field).view();
  const auto state = load_state<TwoComponentEllipticModel>(view, sample);
  Real coordinate_sum = Real(0);
  for (int axis = 0; axis < Dim; ++axis)
    coordinate_sum += Real(sample[axis]);
  EXPECT_EQ(state[0], coordinate_sum);
  EXPECT_EQ(state[1], Real(10) + coordinate_sum);

  ProviderStorageView<Dim, Dim> mapped{};
  for (int slot = 0; slot < Dim; ++slot) {
    mapped.storage[slot] = view;
    mapped.storage_components[slot] = ncomp - 1 - slot;
  }
  const ProviderValues<Dim> providers = load_provider_values<Dim>(mapped, sample);
  for (int slot = 0; slot < Dim; ++slot)
    EXPECT_EQ(providers[slot], Real(10 * (ncomp - 1 - slot)) + coordinate_sum);
}

}  // namespace

TEST(test_prepared_cartesian_nd, one_dimensional_kernel_preserves_constant_state_and_conservation) {
  check_constant_and_conservation<1>(Extent<1>{32});
}

TEST(test_prepared_cartesian_nd, centered_fickian_flux_survives_the_canonical_ranked_face_path) {
  check_centered_fickian_flux<1>();
  check_centered_fickian_flux<2>();
  check_centered_fickian_flux<3>();
}

TEST(test_prepared_cartesian_nd, diffusive_cfl_bound_is_parabolic_and_rejects_invalid_nu) {
  check_diffusive_cfl_bound<1>();
  check_diffusive_cfl_bound<2>();
  check_diffusive_cfl_bound<3>();
}

TEST(test_prepared_cartesian_nd, masked_diffusion_is_refused_without_embedded_face_geometry) {
  check_masked_diffusion_is_refused<1>();
  check_masked_diffusion_is_refused<2>();
  check_masked_diffusion_is_refused<3>();
}

TEST(test_prepared_cartesian_nd,
     three_dimensional_kernel_is_axis_permutation_invariant_and_conservative) {
  check_constant_and_conservation<3>(Extent<3>{8, 7, 6});

  const Box<3> xyz = Box<3>::from_extents(Extent<3>{8, 7, 6});
  const Box<3> zyx = Box<3>::from_extents(Extent<3>{6, 7, 8});
  const auto xyz_geometry =
      Geometry<3>::from_bounds(xyz, RealVector<3>{}, RealVector<3>{Real(1), Real(2), Real(3)});
  const auto zyx_geometry =
      Geometry<3>::from_bounds(zyx, RealVector<3>{}, RealVector<3>{Real(3), Real(2), Real(1)});
  const auto xyz_model =
      nd::ScalarAdvection<3>::prepare(RealVector<3>{Real(0.7), Real(-0.5), Real(0.9)});
  const auto zyx_model =
      nd::ScalarAdvection<3>::prepare(RealVector<3>{Real(0.9), Real(-0.5), Real(0.7)});
  const auto xyz_operator =
      nd::prepare_cartesian_operator<3>(xyz_geometry, xyz_model, VanLeer{}, HLLFlux{});
  const auto zyx_operator =
      nd::prepare_cartesian_operator<3>(zyx_geometry, zyx_model, VanLeer{}, HLLFlux{});

  Fab<3> xyz_state(xyz, 1, uniform_extent<3>(VanLeer::n_ghost));
  Fab<3> zyx_state(zyx, 1, uniform_extent<3>(VanLeer::n_ghost));
  constexpr Real two_pi = Real(6.283185307179586476925286766559);
  const auto xyz_value = [&](const Index<3>& index) {
    return Real(0.9) +
           Real(0.13) * std::sin(two_pi * (Real(index[0]) + Real(0.5)) / Real(xyz.length(0))) +
           Real(0.07) * std::cos(two_pi * (Real(index[1]) + Real(0.5)) / Real(xyz.length(1))) +
           Real(0.04) * std::sin(two_pi * (Real(index[2]) + Real(0.5)) / Real(xyz.length(2)));
  };
  fill_periodic(xyz_state, [&](const Index<3>& index, int) { return xyz_value(index); });
  fill_periodic(zyx_state, [&](const Index<3>& index, int) {
    return xyz_value(Index<3>{index[2], index[1], index[0]});
  });

  Fab<3> xyz_residual(xyz, 1);
  Fab<3> zyx_residual(zyx, 1);
  xyz_operator.assemble_residual(xyz_state, xyz_residual);
  zyx_operator.assemble_residual(zyx_state, zyx_residual);
  auto xyz_host = xyz_residual.create_host_mirror();
  auto zyx_host = zyx_residual.create_host_mirror();
  xyz_residual.copy_to_host(xyz_host);
  zyx_residual.copy_to_host(zyx_host);
  for_each_host_index(zyx, [&](const Index<3>& zyx_index) {
    const Index<3> xyz_index{zyx_index[2], zyx_index[1], zyx_index[0]};
    EXPECT_NEAR(xyz_host(host_offset(xyz_residual.grown_box(), xyz_index, 0)),
                zyx_host(host_offset(zyx_residual.grown_box(), zyx_index, 0)), Real(3e-12));
  });
}

TEST(test_prepared_cartesian_nd, one_face_field_carries_every_compile_time_axis) {
  const Box<3> domain = Box<3>::from_extents(Extent<3>{4, 5, 6});
  const auto geometry =
      Geometry<3>::from_bounds(domain, RealVector<3>{}, RealVector<3>{Real(1), Real(1), Real(1)});
  const auto model =
      nd::ScalarAdvection<3>::prepare(RealVector<3>{Real(0.5), Real(0.5), Real(0.5)});
  const auto op = nd::prepare_cartesian_operator<3>(geometry, model);
  Fab<3> state(domain, 1, uniform_extent<3>(NoSlope::n_ghost));
  fill_periodic(state, [](const Index<3>&, int) { return Real(2); });

  nd::FaceField<3> fluxes(domain, 1);
  op.materialize_face_fluxes(state, fluxes);
  check_constant_face_axis<0, 3>(
      fluxes, std::array<Real, 3>{Real(1) / Real(30), Real(1) / Real(24), Real(1) / Real(20)});
}

TEST(test_prepared_cartesian_nd, prepared_face_field_is_the_public_boundary_to_divergence_seam) {
  const Box<1> domain = Box<1>::from_extents(Extent<1>{4});
  const auto model = nd::ScalarAdvection<1>::prepare(RealVector<1>{Real(1)});
  const auto op = nd::prepare_cartesian_operator<1>(unit_geometry(domain), model);
  nd::FaceField<1> fluxes(domain, 1);
  Fab<1>& faces = fluxes.field<0>();
  auto face_host = faces.create_host_mirror();
  faces.copy_to_host(face_host);
  for_each_host_index(faces.box(), [&](const Index<1>& face) {
    face_host(host_offset(faces.grown_box(), face, 0)) = Real(face[0]);
  });
  faces.copy_from_host(face_host);

  Fab<1> residual(domain, 1);
  op.assemble_residual_from_face_fluxes(fluxes, residual);
  for (const Real value : valid_values(residual))
    EXPECT_EQ(value, Real(-4));

  residual.set_val(Real(9));
  faces.copy_to_host(face_host);
  face_host(host_offset(faces.grown_box(), Index<1>{1}, 0)) =
      std::numeric_limits<Real>::quiet_NaN();
  faces.copy_from_host(face_host);
  EXPECT_THROW(op.assemble_residual_from_face_fluxes(fluxes, residual), std::runtime_error);
  for (const Real value : valid_values(residual))
    EXPECT_EQ(value, Real(9));
}

TEST(test_prepared_cartesian_nd, primitive_euler_hllc_uses_the_same_three_dimensional_path) {
  using Model = nd::IdealGasEuler<3>;
  using Schema = Model::Schema;
  const Box<3> domain = Box<3>::from_extents(Extent<3>{4, 4, 4});
  const auto geometry = unit_geometry(domain);
  const auto model = Model::prepare(Real(1.4));
  typename Model::Primitive primitive{};
  primitive[Schema::density] = Real(1.2);
  primitive[Schema::template velocity<0>] = Real(0.3);
  primitive[Schema::template velocity<1>] = Real(-0.2);
  primitive[Schema::template velocity<2>] = Real(0.1);
  primitive[Schema::pressure] = Real(0.9);
  const auto conservative = model.make_conservative(primitive);
  ASSERT_TRUE(conservative.succeeded());

  RealVector<3> lengths{};
  for (int axis = 0; axis < 3; ++axis)
    lengths[axis] = geometry.upper()[axis] - geometry.lower()[axis];
  const auto metric =
      prepare_metric_provider(domain, CartesianCoordinateMap<3>::make(geometry.lower(), lengths));
  const nd::PreparedCartesianOperator<3, Model, decltype(metric), VanLeer, HLLCFlux,
                                      nd::ReconstructionVariables::Primitive>
      op(model, metric);
  Fab<3> state(domain, Model::n_vars, uniform_extent<3>(VanLeer::n_ghost));
  fill_periodic(state,
                [&](const Index<3>&, int component) { return conservative.value[component]; });
  Fab<3> residual(domain, Model::n_vars);
  op.assemble_residual(state, residual);
  for (int component = 0; component < Model::n_vars; ++component)
    for (const Real value : valid_values(residual, component))
      EXPECT_NEAR(value, Real(0), Real(8e-14));
}

TEST(test_prepared_cartesian_nd, preflight_failure_leaves_the_residual_unpublished) {
  const Box<2> domain = Box<2>::from_extents(Extent<2>{4, 4});
  const auto geometry = unit_geometry(domain);
  const auto model = nd::ScalarAdvection<2>::prepare(RealVector<2>{Real(1), Real(1)});
  const auto op = nd::prepare_cartesian_operator<2>(geometry, model, Weno5{});
  Fab<2> state(domain, 1, Extent<2>{2, 2});
  Fab<2> residual(domain, 1);
  residual.set_val(Real(7));
  EXPECT_THROW(op.assemble_residual(state, residual), std::invalid_argument);
  for (const Real value : valid_values(residual))
    EXPECT_EQ(value, Real(7));

  Fab<2> alias(domain, 1, Extent<2>{Weno5::n_ghost, Weno5::n_ghost});
  EXPECT_THROW(op.assemble_residual(alias, alias), std::invalid_argument);
}

TEST(test_prepared_cartesian_nd, multi_patch_failure_is_transactional_before_first_publication) {
  const Box<1> domain{Index<1>{0}, Index<1>{7}};
  const Box<1> first{Index<1>{0}, Index<1>{3}};
  const Box<1> second{Index<1>{4}, Index<1>{7}};
  const mesh::BoxArray<1> layout(std::vector<Box<1>>{first, second});
  const mesh::RankSpace<1> ranks{Index<1>{0}, Extent<1>{1}};
  const auto distribution = mesh::Distribution<1>::replicated(layout, ranks);
  MultiFab<1> state(layout, distribution, Index<1>{0}, 1, Extent<1>{1});
  MultiFab<1> residual(layout, distribution, Index<1>{0}, 1, Extent<1>{0});
  state.set_val(Real(1));
  residual.set_val(Real(7));

  Fab<1>& invalid_patch = state.fab(1);
  auto invalid_host = invalid_patch.create_host_mirror();
  invalid_patch.copy_to_host(invalid_host);
  invalid_host(host_offset(invalid_patch.grown_box(), Index<1>{4}, 0)) =
      std::numeric_limits<Real>::quiet_NaN();
  invalid_patch.copy_from_host(invalid_host);

  const auto model = nd::ScalarAdvection<1>::prepare(RealVector<1>{Real(1)});
  const auto op = nd::prepare_cartesian_operator<1>(unit_geometry(domain), model);
  EXPECT_THROW(op.assemble_residual(state, residual), std::runtime_error);
  for (std::size_t local = 0; local < residual.local_size(); ++local)
    for (const Real value : valid_values(residual.fab(local)))
      EXPECT_EQ(value, Real(7));
}

TEST(test_prepared_cartesian_nd, state_and_aux_access_share_one_ranked_pointwise_authority) {
  check_ranked_state_access<1>();
  check_ranked_state_access<2>();
  check_ranked_state_access<3>();
}
