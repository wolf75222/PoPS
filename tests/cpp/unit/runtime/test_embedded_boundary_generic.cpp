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
#include <pops/numerics/spatial/embedded_boundary/operator.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/spatial/operators/masked_operator.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <cstddef>
#include <limits>
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
  static_assert(capabilities.binary_face_aperture);
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

}  // namespace
