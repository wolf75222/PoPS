#include <gtest/gtest.h>

#include <pops/amr/hierarchy/level_layout.hpp>
#include <pops/amr/transfer/temporal_interpolation_provider.hpp>
#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/numerics/time/amr/prepared_coarse_fine_operator.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

using pops::Box;
using pops::FieldView;
using pops::Index;
using pops::PreparedCoarseFineOperator;
using pops::PreparedCoarseFineTransform;
using pops::Real;
using pops::amr::transfer::Centering;
using pops::amr::transfer::ComponentRange;
using pops::amr::transfer::DivergencePreservingFaceTransferProvider;
using pops::amr::transfer::IndexMapping;
using pops::amr::transfer::LinearTemporalInterpolationProvider;
using pops::amr::transfer::PreparedTransfer;
using pops::amr::transfer::QualifiedTemporalState;
using pops::amr::transfer::TemporalComponentRange;
using pops::amr::RefinementRatio;
using pops::amr::transfer::SlopeLimiter;
using pops::amr::transfer::TransferKind;
using pops::amr::transfer::TransferProvider;

template <int Dim>
PreparedCoarseFineOperator<Dim> complete_coarse_fine_operator() {
  PreparedCoarseFineOperator<Dim> prepared;
  for (int axis = 0; axis < Dim; ++axis) {
    prepared.parent_reach[axis] = axis + 1;
    prepared.minimum_axis_cells[axis] = axis + 2;
  }
  prepared.launch_spatial = [](FieldView<Real, Dim>, FieldView<const Real, Dim>, const Box<Dim>&,
                               const Box<Dim>&, const Box<Dim>&, const Box<Dim>&,
                               const PreparedCoarseFineTransform<Dim>&, int, bool, bool,
                               const pops::BoundaryTopology<Dim>&) {};
  prepared.launch_space_time = [](FieldView<Real, Dim>, FieldView<const Real, Dim>,
                                  FieldView<const Real, Dim>, const Box<Dim>&, const Box<Dim>&,
                                  const Box<Dim>&, const Box<Dim>&,
                                  const PreparedCoarseFineTransform<Dim>&, int, Real, Real, int,
                                  const pops::BoundaryTopology<Dim>&) {};
  return prepared;
}

template <int Dim>
void expect_prepared_coarse_fine_metadata() {
  auto prepared = complete_coarse_fine_operator<Dim>();
  prepared.validate();

  Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = axis + 1;
  prepared.validate_domain(Box<Dim>{Index<Dim>{}, upper});

  for (int axis = 0; axis < Dim; ++axis)
    --upper[axis];
  EXPECT_THROW(prepared.validate_domain(Box<Dim>{Index<Dim>{}, upper}), std::invalid_argument);

  pops::Extent<Dim> ghosts{};
  pops::Extent<Dim> reach{};
  std::array<int, Dim> ratios{};
  for (int axis = 0; axis < Dim; ++axis) {
    ghosts[axis] = axis + 1;
    reach[axis] = axis + 2;
    ratios[static_cast<std::size_t>(axis)] = axis + 2;
  }
  const RefinementRatio<Dim> ratio{ratios};
  const pops::Extent<Dim> growth =
      pops::detail::checked_coarse_fine_carrier_growth(ghosts, ratio, reach);
  for (int axis = 0; axis < Dim; ++axis)
    EXPECT_EQ(growth[axis], ghosts[axis] + reach[axis] * ratio[axis]);
}

template <int Dim, class F>
void visit(const Box<Dim>& box, F&& function) {
  if (box.empty())
    return;
  Index<Dim> index = box.lo;
  while (true) {
    function(index);
    int axis = 0;
    for (; axis < Dim; ++axis) {
      if (index[axis] < box.hi[axis]) {
        ++index[axis];
        break;
      }
      index[axis] = box.lo[axis];
    }
    if (axis == Dim)
      return;
  }
}

template <int Dim>
class HostField {
 public:
  HostField(Box<Dim> box, int components)
      : box_(box),
        components_(components),
        values_(static_cast<std::size_t>(box.numPts()) * static_cast<std::size_t>(components)) {}

  FieldView<Real, Dim> view() {
    FieldView<Real, Dim> result{};
    populate(result);
    return result;
  }

  FieldView<const Real, Dim> const_view() const {
    FieldView<const Real, Dim> result{};
    populate(result);
    return result;
  }

  Real& operator()(const Index<Dim>& index, int component = 0) { return view()(index, component); }

  Real operator()(const Index<Dim>& index, int component = 0) const {
    return const_view()(index, component);
  }

  const Box<Dim>& box() const { return box_; }

 private:
  template <class T>
  void populate(FieldView<T, Dim>& result) const {
    result.data = values_.data();
    result.origin = box_.lo;
    result.extents = box_.extent();
    result.strides[0] = 1;
    for (int axis = 1; axis < Dim; ++axis)
      result.strides[axis] = result.strides[axis - 1] * result.extents[axis - 1];
    result.ncomp = components_;
    result.component_stride = box_.numPts();
  }

  Box<Dim> box_;
  int components_;
  mutable std::vector<Real> values_;
};

template <int Dim>
RefinementRatio<Dim> sample_ratio() {
  if constexpr (Dim == 1)
    return RefinementRatio<1>{3};
  else if constexpr (Dim == 2)
    return RefinementRatio<2>{2, 3};
  else
    return RefinementRatio<3>{2, 1, 3};
}

template <int Dim>
IndexMapping<Dim> sample_mapping() {
  if constexpr (Dim == 1)
    return {Index<1>{-3}, Index<1>{5}};
  else if constexpr (Dim == 2)
    return {Index<2>{-3, 4}, Index<2>{5, -7}};
  else
    return {Index<3>{-3, 4, -2}, Index<3>{5, -7, 11}};
}

template <int Dim>
Box<Dim> sample_coarse_region(const IndexMapping<Dim>& mapping) {
  Index<Dim> upper = mapping.coarse_origin;
  for (int axis = 0; axis < Dim; ++axis)
    ++upper[axis];
  return {mapping.coarse_origin, upper};
}

template <int Dim>
Box<Dim> sample_coarse_source(const IndexMapping<Dim>& mapping) {
  Index<Dim> lower = mapping.coarse_origin;
  Index<Dim> upper = mapping.coarse_origin;
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] -= 3;
    upper[axis] += 3;
  }
  return {lower, upper};
}

template <int Dim>
Box<Dim> refine_for_test(const Box<Dim>& coarse, const RefinementRatio<Dim>& ratio,
                         const IndexMapping<Dim>& mapping) {
  Box<Dim> fine{};
  for (int axis = 0; axis < Dim; ++axis) {
    fine.lo[axis] =
        mapping.fine_origin[axis] + (coarse.lo[axis] - mapping.coarse_origin[axis]) * ratio[axis];
    fine.hi[axis] = mapping.fine_origin[axis] +
                    (coarse.hi[axis] - mapping.coarse_origin[axis]) * ratio[axis] + ratio[axis] - 1;
  }
  return fine;
}

template <int Dim>
Real affine_coarse(const Index<Dim>& index, const IndexMapping<Dim>& mapping, int component) {
  Real value = Real(2.75) + Real(4.5) * component;
  for (int axis = 0; axis < Dim; ++axis)
    value += Real(axis + 1) * Real(index[axis] - mapping.coarse_origin[axis]);
  return value;
}

template <int Dim>
Real affine_fine(const Index<Dim>& index, const RefinementRatio<Dim>& ratio,
                 const IndexMapping<Dim>& mapping, int component) {
  Real value = Real(2.75) + Real(4.5) * component;
  for (int axis = 0; axis < Dim; ++axis) {
    const Real relative = static_cast<Real>(index[axis] - mapping.fine_origin[axis]);
    value += Real(axis + 1) * ((relative + Real(0.5)) / static_cast<Real>(ratio[axis]) - Real(0.5));
  }
  return value;
}

template <int Dim>
void fill_affine(HostField<Dim>& field, const IndexMapping<Dim>& mapping) {
  visit(field.box(), [&](const Index<Dim>& index) {
    for (int component = 0; component < 2; ++component)
      field(index, component) = affine_coarse(index, mapping, component);
  });
}

Real test_integer_power(Real value, int exponent) {
  Real result = Real(1);
  for (int power = 0; power < exponent; ++power)
    result *= value;
  return result;
}

Real quartic_axis_average(Real lower, Real upper, int axis) {
  const Real coefficients[5] = {Real(1) + Real(0.125) * axis, Real(0.07) * (axis + 1),
                                -Real(0.013) * (axis + 1), Real(0.004) * (axis + 1),
                                Real(0.001) * (axis + 1)};
  Real integral = Real(0);
  for (int degree = 0; degree < 5; ++degree) {
    const int power = degree + 1;
    integral += coefficients[degree] *
                (test_integer_power(upper, power) - test_integer_power(lower, power)) /
                static_cast<Real>(power);
  }
  return integral / (upper - lower);
}

template <int Dim>
Real quartic_coarse_average(const Index<Dim>& index, const IndexMapping<Dim>& mapping) {
  Real result = Real(1);
  for (int axis = 0; axis < Dim; ++axis) {
    const Real center = static_cast<Real>(index[axis] - mapping.coarse_origin[axis]);
    result *= quartic_axis_average(center - Real(0.5), center + Real(0.5), axis);
  }
  return result;
}

template <int Dim>
Real quartic_fine_average(const Index<Dim>& index, const RefinementRatio<Dim>& ratio,
                          const IndexMapping<Dim>& mapping) {
  Real result = Real(1);
  for (int axis = 0; axis < Dim; ++axis) {
    const Real relative = static_cast<Real>(index[axis] - mapping.fine_origin[axis]);
    const Real lower = relative / static_cast<Real>(ratio[axis]) - Real(0.5);
    const Real upper = (relative + Real(1)) / static_cast<Real>(ratio[axis]) - Real(0.5);
    result *= quartic_axis_average(lower, upper, axis);
  }
  return result;
}

template <int Dim>
void expect_fifth_order_quartic_and_parent_average(const RefinementRatio<Dim>& ratio) {
  const auto mapping = sample_mapping<Dim>();
  const Box<Dim> coarse_region = sample_coarse_region(mapping);
  const Box<Dim> fine_region = refine_for_test(coarse_region, ratio, mapping);
  HostField<Dim> coarse(sample_coarse_source(mapping), 1);
  HostField<Dim> fine(fine_region, 1);
  visit(coarse.box(),
        [&](const Index<Dim>& index) { coarse(index) = quartic_coarse_average(index, mapping); });

  const auto provider =
      TransferProvider<Dim, Centering::Cell>::fifth_order_coarse_fine_ghost_interpolation();
  const auto prepared =
      provider.prepare(coarse.const_view(), fine.view(), fine_region, ratio, mapping);
  EXPECT_EQ(prepared.kind(), TransferKind::FifthOrderCoarseFineGhostInterpolation);
  EXPECT_EQ(prepared.slope_limiter(), SlopeLimiter::None);
  visit(prepared.destination_region(), [&](const Index<Dim>& index) { prepared(index); });

  visit(fine_region, [&](const Index<Dim>& index) {
    EXPECT_NEAR(fine(index), quartic_fine_average(index, ratio, mapping), Real(3e-12));
  });
  visit(coarse_region, [&](const Index<Dim>& parent) {
    const Box<Dim> children = refine_for_test(Box<Dim>{parent, parent}, ratio, mapping);
    Real average = Real(0);
    visit(children, [&](const Index<Dim>& child) { average += fine(child); });
    average /= static_cast<Real>(ratio.child_count());
    EXPECT_NEAR(average, coarse(parent), Real(3e-12));
  });
}

template <int Dim>
void execute(const PreparedTransfer<Dim>& prepared) {
  visit(prepared.destination_region(), [&](const Index<Dim>& index) { prepared(index); });
}

template <int Dim>
Index<Dim> parent_of(const Index<Dim>& fine, const RefinementRatio<Dim>& ratio,
                     const IndexMapping<Dim>& mapping) {
  Index<Dim> parent{};
  for (int axis = 0; axis < Dim; ++axis) {
    const int relative = fine[axis] - mapping.fine_origin[axis];
    parent[axis] = mapping.coarse_origin[axis] + relative / ratio[axis];
  }
  return parent;
}

template <int Dim>
Real nonlinear_coarse(const Index<Dim>& index, const IndexMapping<Dim>& mapping) {
  Real value = Real(0.75);
  for (int axis = 0; axis < Dim; ++axis) {
    const Real relative = Real(index[axis] - mapping.coarse_origin[axis]);
    value += Real(0.25 * (axis + 1)) * relative * relative;
    if (index[axis] > mapping.coarse_origin[axis])
      value += Real(0.5 * (axis + 1));
  }
  return value;
}

template <int Dim>
void expect_limited_prolongation_is_conservative() {
  const auto ratio = sample_ratio<Dim>();
  const auto mapping = sample_mapping<Dim>();
  const Box<Dim> coarse_region = sample_coarse_region(mapping);
  const Box<Dim> fine_region = refine_for_test(coarse_region, ratio, mapping);
  HostField<Dim> coarse(sample_coarse_source(mapping), 1);
  HostField<Dim> fine(fine_region, 1);
  visit(coarse.box(),
        [&](const Index<Dim>& index) { coarse(index) = nonlinear_coarse(index, mapping); });

  const auto prepared = TransferProvider<Dim, Centering::Cell>::linear_prolongation().prepare(
      coarse.const_view(), fine.view(), fine_region, ratio, mapping);
  EXPECT_EQ(prepared.slope_limiter(), SlopeLimiter::MonotonizedCentral);
  execute(prepared);

  visit(coarse_region, [&](const Index<Dim>& parent) {
    const Box<Dim> children = refine_for_test(Box<Dim>{parent, parent}, ratio, mapping);
    Real child_sum = Real(0);
    visit(children, [&](const Index<Dim>& child) { child_sum += fine(child); });
    EXPECT_NEAR(child_sum / static_cast<Real>(ratio.child_count()), coarse(parent), 2e-13);

    Index<Dim> neighborhood_lo = parent;
    Index<Dim> neighborhood_hi = parent;
    for (int axis = 0; axis < Dim; ++axis) {
      --neighborhood_lo[axis];
      ++neighborhood_hi[axis];
    }
    Real lower_bound = coarse(neighborhood_lo);
    Real upper_bound = lower_bound;
    visit(Box<Dim>{neighborhood_lo, neighborhood_hi}, [&](const Index<Dim>& neighbor) {
      const Real value = coarse(neighbor);
      if (value < lower_bound)
        lower_bound = value;
      if (value > upper_bound)
        upper_bound = value;
    });
    visit(children, [&](const Index<Dim>& child) {
      EXPECT_GE(fine(child), lower_bound - Real(2e-13));
      EXPECT_LE(fine(child), upper_bound + Real(2e-13));
    });
  });
}

template <int Dim>
void expect_injection_is_explicit() {
  const auto ratio = sample_ratio<Dim>();
  const auto mapping = sample_mapping<Dim>();
  const Box<Dim> coarse_region = sample_coarse_region(mapping);
  const Box<Dim> fine_region = refine_for_test(coarse_region, ratio, mapping);
  HostField<Dim> coarse(coarse_region, 1);
  HostField<Dim> injected(fine_region, 1);
  HostField<Dim> rejected_linear(fine_region, 1);
  visit(coarse_region,
        [&](const Index<Dim>& index) { coarse(index) = nonlinear_coarse(index, mapping); });
  visit(fine_region, [&](const Index<Dim>& index) {
    injected(index) = Real(-31);
    rejected_linear(index) = Real(-47);
  });

  const auto injection = TransferProvider<Dim, Centering::Cell>::constant_injection().prepare(
      coarse.const_view(), injected.view(), fine_region, ratio, mapping);
  EXPECT_EQ(injection.kind(), TransferKind::ConstantInjection);
  EXPECT_EQ(injection.slope_limiter(), SlopeLimiter::None);
  execute(injection);
  visit(fine_region, [&](const Index<Dim>& child) {
    EXPECT_DOUBLE_EQ(injected(child), coarse(parent_of(child, ratio, mapping)));
  });

  EXPECT_THROW((void)(TransferProvider<Dim, Centering::Cell>::linear_prolongation().prepare(
                   coarse.const_view(), rejected_linear.view(), fine_region, ratio, mapping)),
               std::invalid_argument);
  visit(fine_region,
        [&](const Index<Dim>& child) { EXPECT_DOUBLE_EQ(rejected_linear(child), Real(-47)); });
}

Real exponential_cell_average(int cell, int cells, int axis) {
  const Real alpha = Real(0.2 * (axis + 1));
  const Real lower = Real(cell) / Real(cells);
  const Real upper = Real(cell + 1) / Real(cells);
  return (std::exp(alpha * upper) - std::exp(alpha * lower)) / (alpha * (upper - lower));
}

template <int Dim>
Real smooth_cell_average(const Index<Dim>& index, int cells) {
  Real value = Real(0.125);
  for (int axis = 0; axis < Dim; ++axis)
    value += exponential_cell_average(index[axis], cells, axis);
  return value;
}

template <int Dim>
Real prolongation_l1_error(int cells) {
  Index<Dim> source_lo{};
  Index<Dim> source_hi{};
  Index<Dim> fine_lo{};
  Index<Dim> fine_hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    source_lo[axis] = -1;
    source_hi[axis] = cells;
    fine_hi[axis] = 2 * cells - 1;
  }
  const Box<Dim> source_region{source_lo, source_hi};
  const Box<Dim> fine_region{fine_lo, fine_hi};
  std::array<int, Dim> ratio_values{};
  ratio_values.fill(2);
  const RefinementRatio<Dim> ratio{ratio_values};
  const IndexMapping<Dim> mapping{};
  HostField<Dim> coarse(source_region, 1);
  HostField<Dim> fine(fine_region, 1);
  visit(source_region,
        [&](const Index<Dim>& index) { coarse(index) = smooth_cell_average(index, cells); });
  const auto prepared = TransferProvider<Dim, Centering::Cell>::linear_prolongation().prepare(
      coarse.const_view(), fine.view(), fine_region, ratio, mapping);
  execute(prepared);
  Real error = Real(0);
  visit(fine_region, [&](const Index<Dim>& index) {
    error += std::abs(fine(index) - smooth_cell_average(index, 2 * cells));
  });
  return error / static_cast<Real>(fine_region.numPts());
}

template <int Dim>
void expect_constant_restriction() {
  const auto ratio = sample_ratio<Dim>();
  const auto mapping = sample_mapping<Dim>();
  const Box<Dim> coarse_region = sample_coarse_region(mapping);
  const Box<Dim> fine_region = refine_for_test(coarse_region, ratio, mapping);
  HostField<Dim> fine(fine_region, 2);
  HostField<Dim> coarse(coarse_region, 2);
  visit(fine_region, [&](const Index<Dim>& index) {
    fine(index, 0) = Real(0.1);
    fine(index, 1) = Real(-3.25);
  });

  const auto prepared = TransferProvider<Dim, Centering::Cell>::conservative_restriction().prepare(
      fine.const_view(), coarse.view(), coarse_region, ratio, mapping, ComponentRange{0, 0, 2});
  execute(prepared);

  visit(coarse_region, [&](const Index<Dim>& index) {
    EXPECT_DOUBLE_EQ(coarse(index, 0), Real(0.1));
    EXPECT_DOUBLE_EQ(coarse(index, 1), Real(-3.25));
  });
}

template <int Dim>
void expect_affine_prolongation_and_conservative_round_trip() {
  const auto ratio = sample_ratio<Dim>();
  const auto mapping = sample_mapping<Dim>();
  const Box<Dim> coarse_region = sample_coarse_region(mapping);
  const Box<Dim> fine_region = refine_for_test(coarse_region, ratio, mapping);
  HostField<Dim> coarse_source(sample_coarse_source(mapping), 2);
  HostField<Dim> fine(fine_region, 2);
  HostField<Dim> restricted(coarse_region, 2);
  fill_affine(coarse_source, mapping);

  const auto prolongation = TransferProvider<Dim, Centering::Cell>::linear_prolongation().prepare(
      coarse_source.const_view(), fine.view(), fine_region, ratio, mapping,
      ComponentRange{0, 0, 2});
  execute(prolongation);
  visit(fine_region, [&](const Index<Dim>& index) {
    for (int component = 0; component < 2; ++component)
      EXPECT_NEAR(fine(index, component), affine_fine(index, ratio, mapping, component), 1e-13);
  });

  const auto restriction =
      TransferProvider<Dim, Centering::Cell>::conservative_restriction().prepare(
          fine.const_view(), restricted.view(), coarse_region, ratio, mapping,
          ComponentRange{0, 0, 2});
  execute(restriction);
  visit(coarse_region, [&](const Index<Dim>& index) {
    for (int component = 0; component < 2; ++component)
      EXPECT_NEAR(restricted(index, component), affine_coarse(index, mapping, component), 1e-13);
  });
}

template <int Dim>
void expect_negative_offset_ghost_interpolation() {
  const auto ratio = sample_ratio<Dim>();
  const auto mapping = sample_mapping<Dim>();
  Index<Dim> lower = mapping.fine_origin;
  Index<Dim> upper = mapping.fine_origin;
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] -= ratio[axis];
    upper[axis] = mapping.fine_origin[axis] - 1;
  }
  const Box<Dim> ghost_region{lower, upper};
  HostField<Dim> coarse(sample_coarse_source(mapping), 2);
  HostField<Dim> fine_ghosts(ghost_region, 2);
  fill_affine(coarse, mapping);

  const auto interpolation =
      TransferProvider<Dim, Centering::Cell>::coarse_fine_ghost_interpolation().prepare(
          coarse.const_view(), fine_ghosts.view(), ghost_region, ratio, mapping,
          ComponentRange{0, 0, 2});
  execute(interpolation);
  visit(ghost_region, [&](const Index<Dim>& index) {
    for (int component = 0; component < 2; ++component)
      EXPECT_NEAR(fine_ghosts(index, component), affine_fine(index, ratio, mapping, component),
                  1e-13);
  });
}

template <int Dim>
Box<Dim> two_parent_cells(const IndexMapping<Dim>& mapping) {
  Index<Dim> upper = mapping.coarse_origin;
  for (int axis = 0; axis < Dim; ++axis)
    ++upper[axis];
  return {mapping.coarse_origin, upper};
}

template <int Dim>
Real multilinear_value(const Index<Dim>& index, const Index<Dim>& origin) {
  Real coordinates[Dim]{};
  for (int axis = 0; axis < Dim; ++axis)
    coordinates[axis] = static_cast<Real>(index[axis] - origin[axis]);
  Real value = Real(0.75);
  for (int mask = 1; mask < (1 << Dim); ++mask) {
    Real term = Real(0.03 * (mask + 1));
    for (int axis = 0; axis < Dim; ++axis)
      if ((mask & (1 << axis)) != 0)
        term *= coordinates[axis];
    value += term;
  }
  return value;
}

template <int Dim>
Real multilinear_fine_value(const Index<Dim>& index, const RefinementRatio<Dim>& ratio,
                            const IndexMapping<Dim>& mapping) {
  Real coordinates[Dim]{};
  for (int axis = 0; axis < Dim; ++axis)
    coordinates[axis] =
        static_cast<Real>(index[axis] - mapping.fine_origin[axis]) / static_cast<Real>(ratio[axis]);
  Real value = Real(0.75);
  for (int mask = 1; mask < (1 << Dim); ++mask) {
    Real term = Real(0.03 * (mask + 1));
    for (int axis = 0; axis < Dim; ++axis)
      if ((mask & (1 << axis)) != 0)
        term *= coordinates[axis];
    value += term;
  }
  return value;
}

template <int Dim>
void expect_node_multilinear(const RefinementRatio<Dim>& ratio) {
  const auto mapping = sample_mapping<Dim>();
  Index<Dim> coarse_upper = mapping.coarse_origin;
  Index<Dim> fine_upper = mapping.fine_origin;
  for (int axis = 0; axis < Dim; ++axis) {
    coarse_upper[axis] += 2;
    fine_upper[axis] += 2 * ratio[axis];
  }
  const Box<Dim> coarse_nodes{mapping.coarse_origin, coarse_upper};
  const Box<Dim> fine_nodes{mapping.fine_origin, fine_upper};
  HostField<Dim> coarse(coarse_nodes, 1);
  HostField<Dim> fine(fine_nodes, 1);
  visit(coarse_nodes, [&](const Index<Dim>& index) {
    coarse(index) = multilinear_value(index, mapping.coarse_origin);
  });
  const auto prepared =
      TransferProvider<Dim, Centering::Node>::node_multilinear_prolongation().prepare(
          coarse.const_view(), fine.view(), fine_nodes, ratio, mapping);
  execute(prepared);
  visit(fine_nodes, [&](const Index<Dim>& index) {
    EXPECT_NEAR(fine(index), multilinear_fine_value(index, ratio, mapping), Real(3e-13));
  });
}

template <int Dim>
Real face_source_value(int normal_axis, const Index<Dim>& face, const IndexMapping<Dim>& mapping) {
  Real value = Real(0.2 * (normal_axis + 1));
  for (int axis = 0; axis < Dim; ++axis) {
    const Real coordinate = static_cast<Real>(face[axis] - mapping.coarse_origin[axis]);
    value += Real(0.07 * (normal_axis + 1) * (axis + 1)) * coordinate;
    value += Real(0.011 * (axis + 1)) * coordinate * coordinate;
  }
  if constexpr (Dim > 1)
    value += Real(0.013 * (normal_axis + 1)) *
             static_cast<Real>(face[0] - mapping.coarse_origin[0]) *
             static_cast<Real>(face[Dim - 1] - mapping.coarse_origin[Dim - 1]);
  return value;
}

template <int Dim>
Real affine_face_value(int normal_axis, const Index<Dim>& face, const IndexMapping<Dim>& mapping) {
  Real value = Real(0.35 * (normal_axis + 1));
  for (int axis = 0; axis < Dim; ++axis)
    value += Real(0.06 * (normal_axis + 1) * (axis + 1)) *
             Real(face[axis] - mapping.coarse_origin[axis]);
  return value;
}

template <int Dim>
Real affine_fine_face_value(int normal_axis, const Index<Dim>& fine_face,
                            const RefinementRatio<Dim>& ratio, const IndexMapping<Dim>& mapping) {
  Real value = Real(0.35 * (normal_axis + 1));
  for (int axis = 0; axis < Dim; ++axis) {
    const Real relative = Real(fine_face[axis] - mapping.fine_origin[axis]);
    const Real coordinate = axis == normal_axis
                                ? relative / Real(ratio[axis])
                                : (relative + Real(0.5)) / Real(ratio[axis]) - Real(0.5);
    value += Real(0.06 * (normal_axis + 1) * (axis + 1)) * coordinate;
  }
  return value;
}

template <int Dim>
void expect_divergence_preserving_faces(const RefinementRatio<Dim>& ratio) {
  const auto mapping = sample_mapping<Dim>();
  const Box<Dim> coarse_cells = two_parent_cells(mapping);
  const Box<Dim> fine_cells = refine_for_test(coarse_cells, ratio, mapping);
  Index<Dim> source_lower = mapping.coarse_origin;
  Index<Dim> source_upper = coarse_cells.hi;
  for (int axis = 0; axis < Dim; ++axis) {
    --source_lower[axis];
    source_upper[axis] += 2;
  }
  const Box<Dim> source_box{source_lower, source_upper};

  std::vector<HostField<Dim>> coarse_faces;
  std::vector<HostField<Dim>> fine_faces;
  coarse_faces.reserve(Dim);
  fine_faces.reserve(Dim);
  std::array<Box<Dim>, Dim> fine_face_regions{};
  for (int normal_axis = 0; normal_axis < Dim; ++normal_axis) {
    coarse_faces.emplace_back(source_box, 1);
    fine_face_regions[normal_axis] = fine_cells;
    ++fine_face_regions[normal_axis].hi[normal_axis];
    fine_faces.emplace_back(fine_face_regions[normal_axis], 1);
    visit(source_box, [&](const Index<Dim>& face) {
      coarse_faces[static_cast<std::size_t>(normal_axis)](face) =
          face_source_value(normal_axis, face, mapping);
    });
  }

  std::array<FieldView<const Real, Dim>, Dim> source_views{};
  std::array<FieldView<Real, Dim>, Dim> destination_views{};
  for (int axis = 0; axis < Dim; ++axis) {
    source_views[axis] = coarse_faces[static_cast<std::size_t>(axis)].const_view();
    destination_views[axis] = fine_faces[static_cast<std::size_t>(axis)].view();
  }
  const auto prepared = DivergencePreservingFaceTransferProvider<Dim>{}.prepare(
      source_views, destination_views, fine_cells, ratio, mapping);
  for (int axis = 0; axis < Dim; ++axis)
    visit(prepared.destination_face_region(axis),
          [&](const Index<Dim>& face) { prepared(axis, face); });

  visit(fine_cells, [&](const Index<Dim>& fine) {
    const Index<Dim> parent = parent_of(fine, ratio, mapping);
    Real coarse_divergence = Real(0);
    Real fine_divergence = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> coarse_upper_face = parent;
      Index<Dim> fine_upper_face = fine;
      ++coarse_upper_face[axis];
      ++fine_upper_face[axis];
      coarse_divergence += coarse_faces[static_cast<std::size_t>(axis)](coarse_upper_face) -
                           coarse_faces[static_cast<std::size_t>(axis)](parent);
      fine_divergence += static_cast<Real>(ratio[axis]) *
                         (fine_faces[static_cast<std::size_t>(axis)](fine_upper_face) -
                          fine_faces[static_cast<std::size_t>(axis)](fine));
    }
    EXPECT_NEAR(fine_divergence, coarse_divergence, Real(2e-12));
  });

  visit(coarse_cells, [&](const Index<Dim>& parent) {
    for (int normal_axis = 0; normal_axis < Dim; ++normal_axis) {
      for (int side = 0; side < 2; ++side) {
        Index<Dim> child_lower{};
        Index<Dim> child_upper{};
        for (int axis = 0; axis < Dim; ++axis)
          child_upper[axis] = axis == normal_axis ? 0 : ratio[axis] - 1;
        Real sum = Real(0);
        int count = 0;
        visit(Box<Dim>{child_lower, child_upper}, [&](const Index<Dim>& child) {
          Index<Dim> fine_face{};
          for (int axis = 0; axis < Dim; ++axis) {
            fine_face[axis] = mapping.fine_origin[axis] +
                              (parent[axis] - mapping.coarse_origin[axis]) * ratio[axis] +
                              child[axis];
          }
          fine_face[normal_axis] += side * ratio[normal_axis];
          sum += fine_faces[static_cast<std::size_t>(normal_axis)](fine_face);
          ++count;
        });
        Index<Dim> coarse_face = parent;
        coarse_face[normal_axis] += side;
        EXPECT_NEAR(sum / static_cast<Real>(count),
                    coarse_faces[static_cast<std::size_t>(normal_axis)](coarse_face), Real(8e-13));
      }
    }
  });

  for (int normal_axis = 0; normal_axis < Dim; ++normal_axis) {
    visit(source_box, [&](const Index<Dim>& face) {
      coarse_faces[static_cast<std::size_t>(normal_axis)](face) =
          affine_face_value(normal_axis, face, mapping);
    });
  }
  for (int axis = 0; axis < Dim; ++axis)
    source_views[axis] = coarse_faces[static_cast<std::size_t>(axis)].const_view();
  const auto affine = DivergencePreservingFaceTransferProvider<Dim>{}.prepare(
      source_views, destination_views, fine_cells, ratio, mapping);
  for (int axis = 0; axis < Dim; ++axis) {
    visit(affine.destination_face_region(axis), [&](const Index<Dim>& face) {
      affine(axis, face);
      EXPECT_NEAR(fine_faces[static_cast<std::size_t>(axis)](face),
                  affine_fine_face_value(axis, face, ratio, mapping), Real(2e-12));
    });
  }
}

template <int Dim>
void expect_linear_temporal_interpolation(const RefinementRatio<Dim>& ratio) {
  Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = 1;
  const Box<Dim> region{Index<Dim>{}, upper};
  HostField<Dim> older(region, 2);
  HostField<Dim> newer(region, 2);
  HostField<Dim> candidate(region, 2);
  visit(region, [&](const Index<Dim>& index) {
    Real coordinate = Real(0);
    for (int axis = 0; axis < Dim; ++axis)
      coordinate += Real(axis + 1) * static_cast<Real>(index[axis]);
    older(index, 0) = Real(2) + coordinate;
    older(index, 1) = Real(-4) + Real(0.5) * coordinate;
    newer(index, 0) = Real(10) - Real(2) * coordinate;
    newer(index, 1) = Real(8) + coordinate;
    candidate(index, 0) = Real(-99);
    candidate(index, 1) = Real(-99);
  });

  const std::string spatial = "test.temporal.spatial";
  QualifiedTemporalState older_state{"state/U", spatial, 7, 11,
                                     pops::amr::ClockStamp{0, 4, pops::amr::Rational(0, 1), 2.0}};
  QualifiedTemporalState newer_state{"state/U", spatial, 7, 11,
                                     pops::amr::ClockStamp{0, 4, pops::amr::Rational(1, 1), 5.0}};
  QualifiedTemporalState target_state{"state/U", spatial, 7, 11,
                                      pops::amr::ClockStamp{0, 4, pops::amr::Rational(1, 3), 3.0}};
  const auto prepared = LinearTemporalInterpolationProvider<Dim>{}.prepare(
      older.const_view(), newer.const_view(), candidate.view(), region, ratio, older_state,
      newer_state, target_state, TemporalComponentRange{0, 0, 0, 2});
  EXPECT_NEAR(prepared.alpha(), Real(1.0 / 3.0), Real(1e-15));
  visit(region, [&](const Index<Dim>& index) { prepared(index); });
  visit(region, [&](const Index<Dim>& index) {
    for (int component = 0; component < 2; ++component)
      EXPECT_NEAR(candidate(index, component),
                  older(index, component) +
                      Real(1.0 / 3.0) * (newer(index, component) - older(index, component)),
                  Real(4e-15));
  });

  QualifiedTemporalState wrong_target = target_state;
  ++wrong_target.topology_generation;
  visit(region, [&](const Index<Dim>& index) {
    candidate(index, 0) = Real(-101);
    candidate(index, 1) = Real(-101);
  });
  EXPECT_THROW((void)LinearTemporalInterpolationProvider<Dim>{}.prepare(
                   older.const_view(), newer.const_view(), candidate.view(), region, ratio,
                   older_state, newer_state, wrong_target, TemporalComponentRange{0, 0, 0, 2}),
               std::invalid_argument);
  visit(region, [&](const Index<Dim>& index) {
    EXPECT_DOUBLE_EQ(candidate(index, 0), Real(-101));
    EXPECT_DOUBLE_EQ(candidate(index, 1), Real(-101));
  });
}

}  // namespace

TEST(test_nd_transfer, anisotropic_ratios_validate_once_and_fail_closed) {
  const RefinementRatio<3> identity{};
  EXPECT_TRUE(identity.is_identity());
  EXPECT_FALSE(identity.refines_any_axis());
  EXPECT_EQ(identity.child_count(), 1);
  EXPECT_EQ((RefinementRatio<3>{1, 1, 1}), identity);
  EXPECT_EQ((RefinementRatio<1>{3}.child_count()), 3);
  EXPECT_EQ((RefinementRatio<2>{2, 3}.child_count()), 6);
  EXPECT_EQ((RefinementRatio<3>{2, 1, 3}.child_count()), 6);
  EXPECT_TRUE((RefinementRatio<3>{2, 1, 3}.refines_any_axis()));
  EXPECT_THROW((void)(RefinementRatio<1>{0}), std::invalid_argument);
  EXPECT_THROW((void)(RefinementRatio<2>{2, -1}), std::invalid_argument);
  EXPECT_THROW(
      (void)(RefinementRatio<3>{std::numeric_limits<int>::max(), std::numeric_limits<int>::max(),
                                std::numeric_limits<int>::max()}),
      std::overflow_error);
}

TEST(test_nd_transfer, prepared_contract_is_fixed_size_and_reports_exact_capabilities) {
  static_assert(std::is_trivially_copyable_v<PreparedTransfer<1>>);
  static_assert(std::is_trivially_copyable_v<PreparedTransfer<2>>);
  static_assert(std::is_trivially_copyable_v<PreparedTransfer<3>>);
  static_assert(std::is_trivially_copyable_v<TransferProvider<3, Centering::Cell>>);
  static_assert(std::is_trivially_copyable_v<
                pops::amr::transfer::PreparedDivergencePreservingFaceTransfer<3>>);
  static_assert(
      std::is_trivially_copyable_v<pops::amr::transfer::PreparedLinearTemporalInterpolation<3>>);

  EXPECT_EQ((TransferProvider<2, Centering::Cell>::conservative_restriction().capabilities()),
            (pops::amr::transfer::TransferCapabilities{1, 0, true, true, SlopeLimiter::None}));
  EXPECT_EQ((TransferProvider<2, Centering::Cell>::linear_prolongation().capabilities()),
            (pops::amr::transfer::TransferCapabilities{2, 1, true, true,
                                                       SlopeLimiter::MonotonizedCentral}));
  EXPECT_EQ((TransferProvider<2, Centering::Cell>::constant_injection().capabilities()),
            (pops::amr::transfer::TransferCapabilities{1, 0, true, true, SlopeLimiter::None}));
  EXPECT_EQ((TransferProvider<2, Centering::Cell>::fifth_order_coarse_fine_ghost_interpolation()
                 .capabilities()),
            (pops::amr::transfer::TransferCapabilities{5, 2, false, true, SlopeLimiter::None}));
  EXPECT_EQ((TransferProvider<2, Centering::Node>::node_multilinear_prolongation().capabilities()),
            (pops::amr::transfer::TransferCapabilities{2, 0, false, true, SlopeLimiter::None}));
  EXPECT_EQ((DivergencePreservingFaceTransferProvider<2>::capabilities()),
            (pops::amr::transfer::TransferCapabilities{2, 1, true, true,
                                                       SlopeLimiter::MonotonizedCentral}));
  EXPECT_EQ((LinearTemporalInterpolationProvider<2>::capabilities()),
            (pops::amr::transfer::TransferCapabilities{2, 0, true, true, SlopeLimiter::None}));
  EXPECT_THROW((void)(TransferProvider<2, Centering::Node>::linear_prolongation().capabilities()),
               std::invalid_argument);
  EXPECT_THROW(
      (void)(TransferProvider<2, Centering::Cell>{static_cast<TransferKind>(255)}.capabilities()),
      std::invalid_argument);
}

TEST(test_nd_transfer, conservative_restriction_preserves_constants_bit_exact_in_1d_2d_3d) {
  expect_constant_restriction<1>();
  expect_constant_restriction<2>();
  expect_constant_restriction<3>();
}

TEST(test_nd_transfer, linear_prolongation_and_restriction_reproduce_affine_fields_in_1d_2d_3d) {
  expect_affine_prolongation_and_conservative_round_trip<1>();
  expect_affine_prolongation_and_conservative_round_trip<2>();
  expect_affine_prolongation_and_conservative_round_trip<3>();
}

TEST(test_nd_transfer, limited_linear_prolongation_preserves_every_parent_average_in_1d_2d_3d) {
  expect_limited_prolongation_is_conservative<1>();
  expect_limited_prolongation_is_conservative<2>();
  expect_limited_prolongation_is_conservative<3>();
}

TEST(test_nd_transfer, constant_injection_is_explicit_and_never_a_linear_fallback) {
  expect_injection_is_explicit<1>();
  expect_injection_is_explicit<2>();
  expect_injection_is_explicit<3>();
}

TEST(test_nd_transfer, limited_linear_prolongation_converges_at_second_order_in_1d_2d_3d) {
  const Real coarse_1d = prolongation_l1_error<1>(8);
  const Real coarse_2d = prolongation_l1_error<2>(8);
  const Real coarse_3d = prolongation_l1_error<3>(8);
  EXPECT_LT(prolongation_l1_error<1>(16), coarse_1d * Real(0.3));
  EXPECT_LT(prolongation_l1_error<2>(16), coarse_2d * Real(0.3));
  EXPECT_LT(prolongation_l1_error<3>(16), coarse_3d * Real(0.3));
}

TEST(test_nd_transfer, coarse_fine_ghost_interpolation_handles_negative_offsets_in_1d_2d_3d) {
  expect_negative_offset_ghost_interpolation<1>();
  expect_negative_offset_ghost_interpolation<2>();
  expect_negative_offset_ghost_interpolation<3>();
}

TEST(test_nd_transfer,
     fifth_order_coarse_fine_reproduces_quartics_and_parent_averages_for_ratios_two_and_three) {
  expect_fifth_order_quartic_and_parent_average<1>(RefinementRatio<1>{2});
  expect_fifth_order_quartic_and_parent_average<1>(RefinementRatio<1>{3});
  expect_fifth_order_quartic_and_parent_average<2>(RefinementRatio<2>{2, 2});
  expect_fifth_order_quartic_and_parent_average<2>(RefinementRatio<2>{3, 3});
  expect_fifth_order_quartic_and_parent_average<3>(RefinementRatio<3>{2, 2, 2});
  expect_fifth_order_quartic_and_parent_average<3>(RefinementRatio<3>{3, 3, 3});
  expect_fifth_order_quartic_and_parent_average<3>(RefinementRatio<3>{2, 1, 3});
}

TEST(test_nd_transfer,
     node_multilinear_prolongation_reproduces_multilinear_fields_for_exact_ranked_ratios) {
  expect_node_multilinear<1>(RefinementRatio<1>{2});
  expect_node_multilinear<1>(RefinementRatio<1>{3});
  expect_node_multilinear<2>(RefinementRatio<2>{2, 2});
  expect_node_multilinear<2>(RefinementRatio<2>{3, 3});
  expect_node_multilinear<3>(RefinementRatio<3>{2, 2, 2});
  expect_node_multilinear<3>(RefinementRatio<3>{3, 3, 3});
  expect_node_multilinear<3>(RefinementRatio<3>{2, 1, 3});
}

TEST(test_nd_transfer,
     coupled_face_prolongation_preserves_boundary_fluxes_and_cell_divergence_in_1d_2d_3d) {
  expect_divergence_preserving_faces<1>(RefinementRatio<1>{2});
  expect_divergence_preserving_faces<1>(RefinementRatio<1>{3});
  expect_divergence_preserving_faces<2>(RefinementRatio<2>{2, 2});
  expect_divergence_preserving_faces<2>(RefinementRatio<2>{3, 3});
  expect_divergence_preserving_faces<3>(RefinementRatio<3>{2, 2, 2});
  expect_divergence_preserving_faces<3>(RefinementRatio<3>{3, 3, 3});
  expect_divergence_preserving_faces<3>(RefinementRatio<3>{2, 1, 3});
}

TEST(test_nd_transfer,
     linear_temporal_provider_interpolates_two_qualified_states_for_exact_ranked_transitions) {
  expect_linear_temporal_interpolation<1>(RefinementRatio<1>{3});
  expect_linear_temporal_interpolation<2>(RefinementRatio<2>{2, 3});
  expect_linear_temporal_interpolation<3>(RefinementRatio<3>{2, 1, 3});
}

TEST(test_nd_transfer, preparation_rejects_missing_stencils_components_aliases_and_regions) {
  const RefinementRatio<2> ratio{2, 3};
  const IndexMapping<2> mapping{};
  const Box<2> fine_region{Index<2>{0, 0}, Index<2>{3, 5}};
  const Box<2> coarse_without_halo{Index<2>{0, 0}, Index<2>{1, 1}};
  HostField<2> coarse(coarse_without_halo, 1);
  HostField<2> fine(fine_region, 1);
  const auto linear = TransferProvider<2, Centering::Cell>::linear_prolongation();
  const auto fifth_order =
      TransferProvider<2, Centering::Cell>::fifth_order_coarse_fine_ghost_interpolation();

  EXPECT_THROW((void)linear.prepare(coarse.const_view(), fine.view(), fine_region, ratio, mapping),
               std::invalid_argument);
  EXPECT_THROW(
      (void)fifth_order.prepare(coarse.const_view(), fine.view(), fine_region, ratio, mapping),
      std::invalid_argument);

  visit(fine_region, [&](const Index<2>& index) { fine(index) = Real(-17); });
  EXPECT_THROW((void)(TransferProvider<2, Centering::Node>::node_multilinear_prolongation().prepare(
                   coarse.const_view(), fine.view(), fine_region, ratio, mapping)),
               std::invalid_argument);
  visit(fine_region, [&](const Index<2>& index) { EXPECT_DOUBLE_EQ(fine(index), Real(-17)); });

  Box<2> first_face_region = fine_region;
  Box<2> second_face_region = fine_region;
  ++first_face_region.hi[0];
  ++second_face_region.hi[1];
  HostField<2> first_parent_faces(coarse_without_halo, 1);
  HostField<2> second_parent_faces(coarse_without_halo, 1);
  HostField<2> first_child_faces(first_face_region, 1);
  HostField<2> second_child_faces(second_face_region, 1);
  visit(first_face_region, [&](const Index<2>& index) { first_child_faces(index) = Real(-29); });
  visit(second_face_region, [&](const Index<2>& index) { second_child_faces(index) = Real(-29); });
  const std::array<FieldView<const Real, 2>, 2> parent_faces{first_parent_faces.const_view(),
                                                             second_parent_faces.const_view()};
  const std::array<FieldView<Real, 2>, 2> child_faces{first_child_faces.view(),
                                                      second_child_faces.view()};
  EXPECT_THROW((void)DivergencePreservingFaceTransferProvider<2>{}.prepare(
                   parent_faces, child_faces, fine_region, ratio, mapping),
               std::invalid_argument);
  visit(first_face_region,
        [&](const Index<2>& index) { EXPECT_DOUBLE_EQ(first_child_faces(index), Real(-29)); });
  visit(second_face_region,
        [&](const Index<2>& index) { EXPECT_DOUBLE_EQ(second_child_faces(index), Real(-29)); });

  const Box<2> source_with_halo{Index<2>{-1, -1}, Index<2>{2, 2}};
  HostField<2> valid_source(source_with_halo, 1);
  EXPECT_THROW((void)linear.prepare(valid_source.const_view(), fine.view(), fine_region,
                                    RefinementRatio<2>{1, 1}, mapping),
               std::invalid_argument);
  EXPECT_THROW((void)linear.prepare(valid_source.const_view(), fine.view(), fine_region, ratio,
                                    mapping, ComponentRange{0, 0, 2}),
               std::invalid_argument);
  EXPECT_THROW((void)linear.prepare(valid_source.const_view(), fine.view(),
                                    Box<2>{Index<2>{0, 0}, Index<2>{4, 5}}, ratio, mapping),
               std::invalid_argument);

  HostField<2> overlapping(Box<2>{Index<2>{-1, -1}, Index<2>{5, 5}}, 1);
  EXPECT_THROW((void)linear.prepare(overlapping.const_view(), overlapping.view(), fine_region,
                                    ratio, mapping),
               std::invalid_argument);

  const auto unsupported = TransferProvider<2, Centering::Face0>::linear_prolongation();
  EXPECT_THROW((void)unsupported.prepare(valid_source.const_view(), fine.view(), fine_region, ratio,
                                         mapping),
               std::invalid_argument);
  const TransferProvider<2, Centering::Cell> unknown{static_cast<TransferKind>(255)};
  EXPECT_THROW(
      (void)unknown.prepare(valid_source.const_view(), fine.view(), fine_region, ratio, mapping),
      std::invalid_argument);
}

TEST(test_nd_transfer, prepared_coarse_fine_authority_is_exact_ranked_in_1d_2d_3d) {
  expect_prepared_coarse_fine_metadata<1>();
  expect_prepared_coarse_fine_metadata<2>();
  expect_prepared_coarse_fine_metadata<3>();
}

TEST(test_nd_transfer,
     prepared_coarse_fine_authority_fails_closed_on_incomplete_or_overflowing_metadata) {
  PreparedCoarseFineOperator<2> incomplete;
  EXPECT_THROW(incomplete.validate(), std::invalid_argument);

  pops::Extent<2> ghosts{1, 1};
  pops::Extent<2> reach{std::numeric_limits<int>::max(), 1};
  EXPECT_THROW((void)pops::detail::checked_coarse_fine_carrier_growth(
                   ghosts, RefinementRatio<2>{2, 2}, reach),
               std::invalid_argument);
}
