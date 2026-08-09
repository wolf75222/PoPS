/// @file
/// @brief Exact-ranked staircase transport and explicit two-dimensional cut-cell capability.

#include <gtest/gtest.h>

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/coordinate_map.hpp>
#include <pops/mesh/geometry/prepared_metric_provider.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/eb/cut_fraction.hpp>
#include <pops/numerics/spatial/embedded_boundary/domain.hpp>
#include <pops/numerics/spatial/embedded_boundary/operator.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/spatial/operators/masked_operator.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <cstddef>
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
  auto providers = fixture.make_field(pops::flux_provider_count<decltype(model)>, 1);
  providers.set_val(pops::Real(0));
  const auto masked = pops::nd::prepare_masked_cartesian_operator<Dim>(model, fixture.metric);
  masked.assemble_residual(fixture.state, providers, active, residual);

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
  masked.assemble_residual(fixture.state, providers, all_active, masked_all_active);
  pops::nd::prepare_cartesian_operator<Dim>(model, fixture.metric)
      .assemble_residual(fixture.state, providers, cartesian);
  EXPECT_EQ(
      pops::for_each_cell_reduce_max(
          fixture.domain, MaximumDifference<Dim>{std::as_const(masked_all_active.fab(0)).view(),
                                                 std::as_const(cartesian.fab(0)).view()}),
      pops::Real(0));
}

template <int Dim>
concept HasExplicitCutCellOperator = requires {
  typename pops::nd::PreparedEmbeddedBoundaryOperator2D<pops::nd::ScalarAdvection<Dim>,
                                                         CartesianMetric<Dim>>;
};

static_assert(!HasExplicitCutCellOperator<1>);
static_assert(HasExplicitCutCellOperator<2>);
static_assert(!HasExplicitCutCellOperator<3>);

struct MaterializeCutMask2D {
  pops::FieldView<pops::Real, 2> active{};
  CartesianMetric<2> metric;
  pops::detail::HalfPlaneDomain level_set;

  POPS_HD void operator()(const pops::Index<2>& cell) const {
    const auto point = metric.cell_center(cell);
    active(cell) = level_set.cell_active(point[0], point[1]) ? pops::Real(1) : pops::Real(0);
  }
};

struct MaterializeInverseVolume2D {
  pops::FieldView<pops::Real, 2> inverse_volume{};
  CartesianMetric<2> metric;
  pops::detail::HalfPlaneDomain level_set;
  pops::Real dx = pops::Real(0);
  pops::Real dy = pops::Real(0);

  POPS_HD void operator()(const pops::Index<2>& cell) const {
    const auto point = metric.cell_center(cell);
    if (!level_set.cell_active(point[0], point[1])) {
      inverse_volume(cell) = pops::Real(0);
      return;
    }
    const auto cut = pops::detail::cut_fraction(level_set, point[0], point[1], dx, dy);
    const pops::Real kappa = cut.kappa < pops::kEbKappaMin ? pops::kEbKappaMin : cut.kappa;
    inverse_volume(cell) = pops::Real(1) / kappa;
  }
};

struct CountCutCells2D {
  pops::FieldView<const pops::Real, 2> inverse_volume{};

  POPS_HD pops::Real operator()(const pops::Index<2>& cell) const {
    return inverse_volume(cell) > pops::Real(1) + pops::Real(1e-9) ? pops::Real(1) : pops::Real(0);
  }
};

TEST(EmbeddedBoundaryGeneric, StaircaseKernelUsesOneExactAlgorithmInOneTwoAndThreeDimensions) {
  prove_staircase_operator<1>();
  prove_staircase_operator<2>();
  prove_staircase_operator<3>();
}

TEST(EmbeddedBoundaryGeneric, CutCellCapabilityIsExplicitAndOperationalOnlyInTwoDimensions) {
  ExactFixture<2> fixture(32);
  auto active = fixture.make_field(1, 1);
  auto inverse_volume = fixture.make_field(1, 0);
  auto residual = fixture.make_field(1, 0);

  pops::for_each_cell(
      fixture.state.fab(0).grown_box(),
      FillSmoothState<2, CartesianMetric<2>>{fixture.state.fab(0).view(), fixture.metric});
  const pops::detail::HalfPlaneDomain level_set{1.0, 0.6, 0.75};
  pops::for_each_cell(active.fab(0).grown_box(),
                      MaterializeCutMask2D{active.fab(0).view(), fixture.metric, level_set});
  const pops::Real spacing = pops::Real(1) / pops::Real(32);
  pops::for_each_cell(fixture.domain,
                      MaterializeInverseVolume2D{inverse_volume.fab(0).view(), fixture.metric,
                                                 level_set, spacing, spacing});

  const auto model = pops::nd::ScalarAdvection<2>::prepare(velocity<2>());
  auto providers = fixture.make_field(pops::flux_provider_count<decltype(model)>, 1);
  providers.set_val(pops::Real(0));
  const auto embedded = pops::nd::prepare_embedded_boundary_operator(model, fixture.metric);
  static_assert(decltype(embedded)::dimension == 2);
  constexpr auto capabilities = decltype(embedded)::capabilities();
  static_assert(capabilities.centre_sampled_activity);
  static_assert(capabilities.binary_face_aperture);
  static_assert(capabilities.prepared_inverse_volume);
  embedded.assemble_residual(fixture.state, providers, active, inverse_volume, residual);

  const auto active_view = std::as_const(active.fab(0)).view();
  const auto inverse_view = std::as_const(inverse_volume.fab(0)).view();
  const auto residual_view = std::as_const(residual.fab(0)).view();
  EXPECT_GT(pops::for_each_cell_reduce_sum(fixture.domain, CountActive<2>{active_view}),
            pops::Real(0));
  EXPECT_GT(pops::for_each_cell_reduce_sum(fixture.domain, CountInactive<2>{active_view}),
            pops::Real(0));
  EXPECT_GT(pops::for_each_cell_reduce_sum(fixture.domain, CountCutCells2D{inverse_view}),
            pops::Real(0));
  EXPECT_EQ(pops::for_each_cell_reduce_sum(fixture.domain, CountNonFinite<2>{residual_view}),
            pops::Real(0));
  EXPECT_EQ(pops::for_each_cell_reduce_max(fixture.domain,
                                           MaximumInactiveResidual<2>{residual_view, active_view}),
            pops::Real(0));

  auto all_active = fixture.make_field(1, 1);
  auto full_volume = fixture.make_field(1, 0);
  auto embedded_no_cut = fixture.make_field(1, 0);
  auto cartesian = fixture.make_field(1, 0);
  all_active.set_val(pops::Real(1));
  full_volume.set_val(pops::Real(1));
  embedded.assemble_residual(fixture.state, providers, all_active, full_volume, embedded_no_cut);
  pops::nd::prepare_cartesian_operator<2>(model, fixture.metric)
      .assemble_residual(fixture.state, providers, cartesian);
  EXPECT_EQ(pops::for_each_cell_reduce_max(
                fixture.domain, MaximumDifference<2>{std::as_const(embedded_no_cut.fab(0)).view(),
                                                     std::as_const(cartesian.fab(0)).view()}),
            pops::Real(0));
}

}  // namespace
