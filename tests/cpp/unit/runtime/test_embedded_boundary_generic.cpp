/// @file
/// @brief Exact-ranked staircase and cut-cell transport capability.

#include <gtest/gtest.h>

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/coordinate_map.hpp>
#include <pops/mesh/geometry/prepared_metric_provider.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/embedded_boundary/characteristic.hpp>
#include <pops/numerics/spatial/embedded_boundary/cut_geometry.hpp>
#include <pops/numerics/spatial/embedded_boundary/operator.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/spatial/operators/masked_operator.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <int Dim>
pops::Extent<Dim> filled_extent(int value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::RealVector<Dim> filled_real(pops::Real value) {
  pops::RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
using CartesianMetric = pops::PreparedMappedMetricProvider<pops::CartesianCoordinateMap<Dim>>;

template <int Dim>
struct ExactFixture {
  using field_type = pops::MultiFab<Dim>;

  explicit ExactFixture(int cells_per_axis)
      : domain(pops::Box<Dim>::from_extents(filled_extent<Dim>(cells_per_axis))),
        metric(pops::prepare_metric_provider(
            domain, pops::CartesianCoordinateMap<Dim>::make(pops::RealVector<Dim>{},
                                                            filled_real<Dim>(pops::Real(1))))),
        layout(std::vector<pops::Box<Dim>>{domain}),
        ranks(pops::Index<Dim>{}, filled_extent<Dim>(1)),
        distribution(pops::mesh::Distribution<Dim>::replicated(layout, ranks)),
        state(layout, distribution, local_rank, 1, filled_extent<Dim>(1)) {}

  field_type make_field(int components, int ghost_depth) const {
    return field_type(layout, distribution, local_rank, components,
                      filled_extent<Dim>(ghost_depth));
  }

  pops::Box<Dim> domain;
  CartesianMetric<Dim> metric;
  pops::mesh::BoxArray<Dim> layout;
  pops::mesh::RankSpace<Dim> ranks;
  pops::mesh::Distribution<Dim> distribution;
  pops::Index<Dim> local_rank{};
  field_type state;
};

template <int Dim, class Metric>
struct FillSmoothState {
  pops::FieldView<pops::Real, Dim> state{};
  Metric metric;

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    const auto point = metric.cell_center(cell);
    pops::Real value = pops::Real(1);
    for (int axis = 0; axis < Dim; ++axis)
      value += pops::Real(0.04 * (axis + 1)) * (point[axis] + point[axis] * point[axis]);
    state(cell) = value;
  }
};

template <int Dim, class Metric>
struct MaterializeStaircaseMask {
  pops::FieldView<pops::Real, Dim> active{};
  Metric metric;

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    active(cell) = metric.cell_center(cell)[0] < pops::Real(0.57) ? pops::Real(1) : pops::Real(0);
  }
};

template <int Dim>
struct CountActive {
  pops::FieldView<const pops::Real, Dim> active{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    return active(cell) >= pops::Real(0.5) ? pops::Real(1) : pops::Real(0);
  }
};

template <int Dim>
struct CountInactive {
  pops::FieldView<const pops::Real, Dim> active{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    return active(cell) < pops::Real(0.5) ? pops::Real(1) : pops::Real(0);
  }
};

template <int Dim>
struct CountNonFinite {
  pops::FieldView<const pops::Real, Dim> values{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    return Kokkos::isfinite(values(cell)) ? pops::Real(0) : pops::Real(1);
  }
};

template <int Dim>
struct MaximumInactiveResidual {
  pops::FieldView<const pops::Real, Dim> residual{};
  pops::FieldView<const pops::Real, Dim> active{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    if (active(cell) >= pops::Real(0.5))
      return pops::Real(0);
    const pops::Real value = residual(cell);
    return value < pops::Real(0) ? -value : value;
  }
};

template <int Dim>
struct MaximumActiveResidual {
  pops::FieldView<const pops::Real, Dim> residual{};
  pops::FieldView<const pops::Real, Dim> active{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    if (active(cell) < pops::Real(0.5))
      return pops::Real(0);
    const pops::Real value = residual(cell);
    return value < pops::Real(0) ? -value : value;
  }
};

template <int Dim>
struct MaximumDifference {
  pops::FieldView<const pops::Real, Dim> left{};
  pops::FieldView<const pops::Real, Dim> right{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    const pops::Real difference = left(cell) - right(cell);
    return difference < pops::Real(0) ? -difference : difference;
  }
};

template <int Dim>
pops::RealVector<Dim> velocity() {
  pops::RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = pops::Real(0.25 + 0.1 * axis);
  return result;
}

template <int Dim>
void prove_staircase_operator() {
  ExactFixture<Dim> fixture(18);
  auto active = fixture.make_field(1, 1);
  auto residual = fixture.make_field(1, 0);

  pops::for_each_cell(
      fixture.state.fab(0).grown_box(),
      FillSmoothState<Dim, CartesianMetric<Dim>>{fixture.state.fab(0).view(), fixture.metric});
  pops::for_each_cell(
      active.fab(0).grown_box(),
      MaterializeStaircaseMask<Dim, CartesianMetric<Dim>>{active.fab(0).view(), fixture.metric});

  const auto model = pops::nd::ScalarAdvection<Dim>::prepare(velocity<Dim>());
  const auto masked = pops::nd::prepare_masked_cartesian_operator<Dim>(model, fixture.metric);
  masked.assemble_residual(fixture.state, active, residual);

  const auto active_view = std::as_const(active.fab(0)).view();
  const auto residual_view = std::as_const(residual.fab(0)).view();
  const pops::Real active_count =
      pops::for_each_cell_reduce_sum(fixture.domain, CountActive<Dim>{active_view});
  const pops::Real inactive_count =
      pops::for_each_cell_reduce_sum(fixture.domain, CountInactive<Dim>{active_view});
  EXPECT_GT(active_count, pops::Real(0));
  EXPECT_GT(inactive_count, pops::Real(0));
  EXPECT_EQ(pops::for_each_cell_reduce_sum(fixture.domain, CountNonFinite<Dim>{residual_view}),
            pops::Real(0));
  EXPECT_EQ(pops::for_each_cell_reduce_max(
                fixture.domain, MaximumInactiveResidual<Dim>{residual_view, active_view}),
            pops::Real(0));
  EXPECT_GT(pops::for_each_cell_reduce_max(fixture.domain,
                                           MaximumActiveResidual<Dim>{residual_view, active_view}),
            pops::Real(0));

  auto all_active = fixture.make_field(1, 1);
  auto masked_all_active = fixture.make_field(1, 0);
  auto cartesian = fixture.make_field(1, 0);
  all_active.set_val(pops::Real(1));
  masked.assemble_residual(fixture.state, all_active, masked_all_active);
  pops::nd::prepare_cartesian_operator<Dim>(model, fixture.metric)
      .assemble_residual(fixture.state, cartesian);
  EXPECT_EQ(
      pops::for_each_cell_reduce_max(
          fixture.domain, MaximumDifference<Dim>{std::as_const(masked_all_active.fab(0)).view(),
                                                 std::as_const(cartesian.fab(0)).view()}),
      pops::Real(0));
}

template <int Dim>
struct MaterializeCutMask {
  pops::FieldView<pops::Real, Dim> active{};
  CartesianMetric<Dim> metric;

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    const auto point = metric.cell_center(cell);
    bool inside = true;
    for (int axis = 0; axis < Dim; ++axis)
      inside = inside && point[axis] > pops::Real(0.18) && point[axis] < pops::Real(0.82);
    active(cell) = inside ? pops::Real(1) : pops::Real(0);
  }
};

template <int Dim>
struct MaterializeInverseVolume {
  pops::FieldView<pops::Real, Dim> inverse_volume{};
  CartesianMetric<Dim> metric;
  int cut_axis = 0;

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    const auto point = metric.cell_center(cell);
    bool inside = true;
    for (int axis = 0; axis < Dim; ++axis)
      inside = inside && point[axis] > pops::Real(0.18) && point[axis] < pops::Real(0.82);
    if (!inside) {
      inverse_volume(cell) = pops::Real(0);
      return;
    }
    inverse_volume(cell) = point[cut_axis] < pops::Real(0.27) || point[cut_axis] > pops::Real(0.73)
                               ? pops::Real(2)
                               : pops::Real(1);
  }
};

template <int Dim>
struct CountCutCells {
  pops::FieldView<const pops::Real, Dim> inverse_volume{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    return inverse_volume(cell) > pops::Real(1) + pops::Real(1e-9) ? pops::Real(1) : pops::Real(0);
  }
};

template <int Dim>
struct SumVolumeWeightedResidual {
  pops::FieldView<const pops::Real, Dim> residual{};
  pops::FieldView<const pops::Real, Dim> active{};
  pops::FieldView<const pops::Real, Dim> inverse_volume{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    if (active(cell) < pops::Real(0.5))
      return pops::Real(0);
    return residual(cell) / inverse_volume(cell);
  }
};

template <int Dim>
struct SetOneNonFiniteCell {
  pops::FieldView<pops::Real, Dim> state{};
  pops::Index<Dim> target{};

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    if (cell == target)
      state(cell) = std::numeric_limits<pops::Real>::quiet_NaN();
  }
};

template <int Dim>
struct MaximumSentinelDifference {
  pops::FieldView<const pops::Real, Dim> values{};
  pops::Real sentinel = pops::Real(0);

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    const pops::Real difference = values(cell) - sentinel;
    return difference < pops::Real(0) ? -difference : difference;
  }
};

TEST(EmbeddedBoundaryGeneric, StaircaseKernelUsesOneExactAlgorithmInOneTwoAndThreeDimensions) {
  prove_staircase_operator<1>();
  prove_staircase_operator<2>();
  prove_staircase_operator<3>();
}

template <int Dim>
void prove_cut_cell_operator() {
  ExactFixture<Dim> fixture(32);
  pops::for_each_cell(
      fixture.state.fab(0).grown_box(),
      FillSmoothState<Dim, CartesianMetric<Dim>>{fixture.state.fab(0).view(), fixture.metric});
  const auto model = pops::nd::ScalarAdvection<Dim>::prepare(velocity<Dim>());
  const auto embedded = pops::nd::prepare_embedded_boundary_operator(model, fixture.metric);
  static_assert(decltype(embedded)::dimension == Dim);
  constexpr auto capabilities = decltype(embedded)::capabilities();
  static_assert(capabilities.centre_sampled_activity);
  static_assert(!capabilities.binary_face_aperture);
  static_assert(capabilities.prepared_inverse_volume);

  for (int cut_axis = 0; cut_axis < Dim; ++cut_axis) {
    auto active = fixture.make_field(1, 1);
    auto inverse_volume = fixture.make_field(1, 0);
    auto residual = fixture.make_field(1, 0);
    pops::for_each_cell(active.fab(0).grown_box(),
                        MaterializeCutMask<Dim>{active.fab(0).view(), fixture.metric});
    pops::for_each_cell(fixture.domain, MaterializeInverseVolume<Dim>{inverse_volume.fab(0).view(),
                                                                      fixture.metric, cut_axis});
    embedded.assemble_residual(fixture.state, active, inverse_volume, residual);

    const auto active_view = std::as_const(active.fab(0)).view();
    const auto inverse_view = std::as_const(inverse_volume.fab(0)).view();
    const auto residual_view = std::as_const(residual.fab(0)).view();
    EXPECT_GT(pops::for_each_cell_reduce_sum(fixture.domain, CountActive<Dim>{active_view}),
              pops::Real(0));
    EXPECT_GT(pops::for_each_cell_reduce_sum(fixture.domain, CountInactive<Dim>{active_view}),
              pops::Real(0));
    EXPECT_GT(pops::for_each_cell_reduce_sum(fixture.domain, CountCutCells<Dim>{inverse_view}),
              pops::Real(0));
    EXPECT_EQ(pops::for_each_cell_reduce_sum(fixture.domain, CountNonFinite<Dim>{residual_view}),
              pops::Real(0));
    EXPECT_EQ(pops::for_each_cell_reduce_max(
                  fixture.domain, MaximumInactiveResidual<Dim>{residual_view, active_view}),
              pops::Real(0));
    const pops::Real balance = pops::for_each_cell_reduce_sum(
        fixture.domain, SumVolumeWeightedResidual<Dim>{residual_view, active_view, inverse_view});
    EXPECT_NEAR(balance, pops::Real(0), pops::Real(2e-10));
  }

  auto all_active = fixture.make_field(1, 1);
  auto full_volume = fixture.make_field(1, 0);
  auto embedded_no_cut = fixture.make_field(1, 0);
  auto cartesian = fixture.make_field(1, 0);
  all_active.set_val(pops::Real(1));
  full_volume.set_val(pops::Real(1));
  embedded.assemble_residual(fixture.state, all_active, full_volume, embedded_no_cut);
  pops::nd::prepare_cartesian_operator<Dim>(model, fixture.metric)
      .assemble_residual(fixture.state, cartesian);
  EXPECT_EQ(pops::for_each_cell_reduce_max(
                fixture.domain, MaximumDifference<Dim>{std::as_const(embedded_no_cut.fab(0)).view(),
                                                       std::as_const(cartesian.fab(0)).view()}),
            pops::Real(0));

  auto active = fixture.make_field(1, 1);
  auto inverse_volume = fixture.make_field(1, 0);
  auto refused = fixture.make_field(1, 0);
  pops::for_each_cell(active.fab(0).grown_box(),
                      MaterializeCutMask<Dim>{active.fab(0).view(), fixture.metric});
  pops::for_each_cell(fixture.domain, MaterializeInverseVolume<Dim>{inverse_volume.fab(0).view(),
                                                                    fixture.metric, 0});
  constexpr pops::Real sentinel = pops::Real(17);
  refused.set_val(sentinel);
  pops::Index<Dim> target{};
  for (int axis = 0; axis < Dim; ++axis)
    target[axis] = fixture.domain.length(axis) / 2;
  pops::for_each_cell(fixture.domain,
                      SetOneNonFiniteCell<Dim>{fixture.state.fab(0).view(), target});
  EXPECT_THROW(embedded.assemble_residual(fixture.state, active, inverse_volume, refused),
               std::runtime_error);
  EXPECT_EQ(pops::for_each_cell_reduce_max(
                fixture.domain,
                MaximumSentinelDifference<Dim>{std::as_const(refused.fab(0)).view(), sentinel}),
            pops::Real(0));
}

TEST(EmbeddedBoundaryGeneric, CutCellUsesOneExactAlgorithmInOneTwoAndThreeDimensions) {
  prove_cut_cell_operator<1>();
  prove_cut_cell_operator<2>();
  prove_cut_cell_operator<3>();
}

template <int Dim>
struct FillPairState {
  pops::FieldView<pops::Real, Dim> state{};

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    state(cell) = cell[0] == 0 ? pops::Real(1) : pops::Real(0);
  }
};

template <int Dim>
struct FillContinuousApertures {
  pops::FieldView<pops::Real, Dim> lower{};
  pops::FieldView<pops::Real, Dim> upper{};
  pops::Real shared = pops::Real(1);

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    for (int axis = 0; axis < Dim; ++axis) {
      lower(cell, axis) = pops::Real(1);
      upper(cell, axis) = pops::Real(1);
    }
    if (cell[0] == 0)
      upper(cell, 0) = shared;
    if (cell[0] == 1)
      lower(cell, 0) = shared;
  }
};

template <int Dim>
pops::Real assemble_shared_face_residual(pops::Real aperture) {
  pops::Extent<Dim> cells{};
  for (int axis = 0; axis < Dim; ++axis)
    cells[axis] = axis == 0 ? 2 : 1;
  const auto domain = pops::Box<Dim>::from_extents(cells);
  const auto metric = pops::prepare_metric_provider(
      domain, pops::CartesianCoordinateMap<Dim>::make(pops::RealVector<Dim>{},
                                                      filled_real<Dim>(pops::Real(1))));
  const pops::mesh::BoxArray<Dim> layout{std::vector<pops::Box<Dim>>{domain}};
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{}, filled_extent<Dim>(1)};
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(layout, ranks);
  const pops::Index<Dim> local{};
  pops::MultiFab<Dim> state(layout, distribution, local, 1, filled_extent<Dim>(1));
  pops::MultiFab<Dim> active(layout, distribution, local, 1, filled_extent<Dim>(1));
  pops::MultiFab<Dim> inverse(layout, distribution, local, 1, pops::Extent<Dim>{});
  pops::MultiFab<Dim> face_lower(layout, distribution, local, Dim, pops::Extent<Dim>{});
  pops::MultiFab<Dim> face_upper(layout, distribution, local, Dim, pops::Extent<Dim>{});
  pops::MultiFab<Dim> residual(layout, distribution, local, 1, pops::Extent<Dim>{});
  pops::for_each_cell(state.fab(0).grown_box(), FillPairState<Dim>{state.fab(0).view()});
  active.set_val(pops::Real(1));
  inverse.set_val(pops::Real(1));
  pops::for_each_cell(domain, FillContinuousApertures<Dim>{face_lower.fab(0).view(),
                                                           face_upper.fab(0).view(), aperture});
  pops::nd::BoundaryFaceOmission<Dim> omission{};
  omission.domain = domain;
  for (int axis = 0; axis < Dim; ++axis) {
    omission.lower[axis] = true;
    omission.upper[axis] = true;
  }
  pops::RealVector<Dim> advect{};
  advect[0] = pops::Real(1);
  const auto model = pops::nd::ScalarAdvection<Dim>::prepare(advect);
  const auto embedded = pops::nd::prepare_embedded_boundary_operator(model, metric);
  embedded.assemble_residual(state, active, inverse, face_lower, face_upper, residual, omission);
  pops::Index<Dim> left{};
  return std::as_const(residual.fab(0)).view()(left);
}

template <int Dim>
void prove_continuous_aperture_scales_flux() {
  const pops::Real full = assemble_shared_face_residual<Dim>(pops::Real(1));
  const pops::Real half = assemble_shared_face_residual<Dim>(pops::Real(0.5));
  const pops::Real closed = assemble_shared_face_residual<Dim>(pops::Real(0));
  EXPECT_NE(full, pops::Real(0));
  EXPECT_NEAR(closed, pops::Real(0), pops::Real(1e-12));
  EXPECT_NEAR(half, pops::Real(0.5) * full, pops::Real(1e-12));
}

TEST(EmbeddedBoundaryGeneric, ContinuousFaceApertureScalesFluxInOneTwoAndThreeDimensions) {
  prove_continuous_aperture_scales_flux<1>();
  prove_continuous_aperture_scales_flux<2>();
  prove_continuous_aperture_scales_flux<3>();
}

template <int Dim>
struct FillDiagonalCutLevelSet {
  pops::FieldView<pops::Real, Dim> phi{};
  pops::FieldView<pops::Real, Dim> active{};

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    bool cut = true;
    bool lower_neighbour = true;
    for (int axis = 0; axis < Dim; ++axis) {
      cut = cut && cell[axis] == 0;
      lower_neighbour = lower_neighbour && cell[axis] <= 0;
    }
    if (cut) {
      phi(cell) = pops::Real(-0.1);
      active(cell) = pops::Real(1);
    } else if (lower_neighbour) {
      phi(cell) = pops::Real(-1);
      active(cell) = pops::Real(1);
    } else {
      phi(cell) = pops::Real(1);
      active(cell) = pops::Real(0);
    }
  }
};

template <int Dim>
void prove_characteristic_uses_interface_normal() {
  ExactFixture<Dim> fixture(3);
  auto phi = fixture.make_field(1, 1);
  auto active = fixture.make_field(1, 1);
  auto exterior = fixture.make_field(1, 0);
  fixture.state.set_val(pops::Real(1));
  pops::for_each_cell(phi.fab(0).grown_box(),
                      FillDiagonalCutLevelSet<Dim>{phi.fab(0).view(), active.fab(0).view()});

  pops::RealVector<Dim> velocity{};
  velocity[0] = pops::Real(1);
  if constexpr (Dim >= 2)
    velocity[1] = pops::Real(-2);
  const auto model = pops::nd::ScalarAdvection<Dim>::prepare(velocity);
  typename std::remove_cvref_t<decltype(model)>::State reference{};
  pops::nd::fill_embedded_characteristic_no_inflow(model, fixture.state.fab(0), phi.fab(0),
                                                   active.fab(0), reference, exterior.fab(0));

  const pops::Index<Dim> cut{};
  const auto fractions = pops::nd::cut_cell_fractions_from_phi_cell(
      std::as_const(phi.fab(0)).view(), cut);
  pops::RealVector<Dim> normal{};
  ASSERT_TRUE(pops::nd::cut_cell_interface_normal(fractions, normal));
  if constexpr (Dim >= 2) {
    EXPECT_GT(normal[0], pops::Real(0));
    EXPECT_GT(normal[1], pops::Real(0));
    EXPECT_NEAR(normal[0], normal[1], pops::Real(1e-12));
  }

  typename std::remove_cvref_t<decltype(model)>::State interior{};
  interior[0] = pops::Real(1);
  typename std::remove_cvref_t<decltype(model)>::State expected{};
  ASSERT_TRUE(pops::nd::apply_characteristic_no_inflow_on_normal<Dim>(model, interior, reference,
                                                                      normal, expected));
  EXPECT_NEAR(std::as_const(exterior.fab(0)).view()(cut), expected[0], pops::Real(1e-12));

  if constexpr (Dim >= 2) {
    typename std::remove_cvref_t<decltype(model)>::State axis_alias{};
    ASSERT_TRUE(model.characteristic_no_inflow(interior, reference, 0, 1, axis_alias));
    EXPECT_NE(expected[0], axis_alias[0]);
    EXPECT_NEAR(axis_alias[0], pops::Real(1), pops::Real(1e-12));
    EXPECT_NEAR(expected[0], pops::Real(-1), pops::Real(1e-12));
  }
}

TEST(EmbeddedBoundaryGeneric, CharacteristicNoInflowUsesInterfaceNormalInOneTwoAndThreeDimensions) {
  prove_characteristic_uses_interface_normal<1>();
  prove_characteristic_uses_interface_normal<2>();
  prove_characteristic_uses_interface_normal<3>();
}

}  // namespace
