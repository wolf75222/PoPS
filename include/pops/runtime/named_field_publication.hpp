/// @file
/// @brief Dimension-generic publication of solved named fields into an auxiliary carrier.

#pragma once

#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/named_field_output.hpp>

#include <stdexcept>

namespace pops::runtime::field {
namespace detail {

template <int Dim>
struct NamedFieldPublicationKernel {
  FieldView<const Real, Dim> potential;
  FieldView<Real, Dim> auxiliary;
  NamedFieldOutput<Dim> output;
  Real inverse_two_spacing[Dim]{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    auxiliary(index, output.potential_component()) = potential(index, 0);
    if (!output.has_gradients())
      return;
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> lower = index;
      Index<Dim> upper = index;
      --lower[axis];
      ++upper[axis];
      auxiliary(index, output.gradient_component(axis)) =
          static_cast<Real>(output.gradient_sign()) * (potential(upper, 0) - potential(lower, 0)) *
          inverse_two_spacing[axis];
    }
  }
};

}  // namespace detail

/// Publish a solved scalar potential and its optional signed centered gradient.
///
/// The solver must have completed and populated one ghost cell on every axis before this call. The
/// publication itself is a single rank-specialized Kokkos launch per local patch; no rank branch or
/// 2D compatibility layout is retained.
template <int Dim>
void publish_named_field(const MultiFab<Dim>& potential, MultiFab<Dim>& auxiliary,
                         const Geometry<Dim>& geometry, const NamedFieldOutput<Dim>& output) {
  if (potential.ncomp() != 1)
    throw std::invalid_argument("named elliptic potential must contain exactly one component");
  if (potential.layout() != auxiliary.layout() ||
      potential.distribution() != auxiliary.distribution() ||
      potential.local_rank() != auxiliary.local_rank())
    throw std::invalid_argument(
        "named elliptic potential and auxiliary publication carriers must share one exact layout");
  for (const Box<Dim>& patch : potential.layout().boxes())
    if (!geometry.domain().contains(patch))
      throw std::invalid_argument(
          "named elliptic publication layout lies outside its exact geometry domain");
  output.validate_width(auxiliary.ncomp(), "named elliptic field");
  if (output.has_gradients())
    for (int axis = 0; axis < Dim; ++axis)
      if (potential.ghosts()[axis] < 1)
        throw std::invalid_argument(
            "named elliptic gradient publication requires one potential ghost per axis");

  Real inverse_two_spacing[Dim]{};
  for (int axis = 0; axis < Dim; ++axis)
    inverse_two_spacing[axis] = Real(1) / (Real(2) * geometry.spacing(axis));

  device_fence();
  for (std::size_t local = 0; local < potential.local_size(); ++local) {
    detail::NamedFieldPublicationKernel<Dim> kernel{
        potential.fab(local).view(), auxiliary.fab(local).view(), output, {}};
    for (int axis = 0; axis < Dim; ++axis)
      kernel.inverse_two_spacing[axis] = inverse_two_spacing[axis];
    for_each_cell(potential.fab(local).box(), kernel);
  }
  device_fence();
}

}  // namespace pops::runtime::field
