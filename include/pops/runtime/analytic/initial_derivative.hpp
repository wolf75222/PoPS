/// @file
/// @brief Qualified initial-grid derivative with physical-edge second-order stencils.
#pragma once

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/parallel/collective_exception.hpp>
#include <pops/runtime/system/auxiliary_ghost_fill.hpp>

#include <cmath>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace pops::analytic {
namespace initial_derivative_detail {

template <int Dim>
struct CopyScalar {
  FieldView<const Real, Dim> source;
  FieldView<Real, Dim> destination;
  int component;
  POPS_HD void operator()(const Index<Dim>& index) const {
    destination(index, 0) = source(index, component);
  }
};

template <int Dim>
struct Derivative {
  FieldView<const Real, Dim> input;
  Box<Dim> domain;
  int axis;
  Real spacing;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    Index<Dim> left = index;
    Index<Dim> right = index;
    if (index[axis] == domain.lo[axis]) {
      ++left[axis];
      right[axis] += 2;
      return (Real(-1.5) / spacing) * input(index, 0) + (Real(2) / spacing) * input(left, 0) +
             (Real(-0.5) / spacing) * input(right, 0);
    }
    if (index[axis] == domain.hi[axis]) {
      left[axis] -= 2;
      --right[axis];
      return (Real(0.5) / spacing) * input(left, 0) + (Real(-2) / spacing) * input(right, 0) +
             (Real(1.5) / spacing) * input(index, 0);
    }
    --left[axis];
    ++right[axis];
    return (input(right, 0) - input(left, 0)) / (Real(2) * spacing);
  }
};

template <int Dim>
struct WriteDerivative {
  Derivative<Dim> derivative;
  FieldView<Real, Dim> output;
  POPS_HD void operator()(const Index<Dim>& index) const { output(index, 0) = derivative(index); }
};

template <int Dim>
struct InvalidDerivative {
  Derivative<Dim> derivative;
  POPS_HD Real operator()(const Index<Dim>& index) const {
    return Kokkos::isfinite(derivative(index)) ? Real(0) : Real(1);
  }
};

}  // namespace initial_derivative_detail

/// Copy an authenticated scalar provider component and compute its initial-grid derivative.
/// The complete-domain halo schedule refuses sparse AMR coverage: this function cannot silently
/// turn a fine-patch edge into a physical one-sided boundary or invent a coarse-fine sample.
template <int Dim>
std::unique_ptr<MultiFab<Dim>> materialize_initial_derivative(const MultiFab<Dim>& source,
                                                              int component,
                                                              const Geometry<Dim>& geometry,
                                                              int axis, int edge_order,
                                                              const ExecutionLane& lane) {
  std::unique_ptr<MultiFab<Dim>> halo;
  std::unique_ptr<MultiFab<Dim>> result;
  std::optional<HaloSchedule<Dim>> schedule;
  std::exception_ptr preparation_error;
  try {
    if (component < 0 || component >= source.ncomp() || axis < 0 || axis >= Dim ||
        edge_order != 2 || geometry.domain().length(axis) < 3 ||
        !(geometry.spacing(axis) > Real(0)) || !std::isfinite(geometry.spacing(axis)))
      throw std::invalid_argument(
          "initial provider derivative requires rank-valid axis, "
          "edge_order=2, and at least three global cells");
    Extent<Dim> ghosts{};
    for (int dimension = 0; dimension < Dim; ++dimension)
      ghosts[dimension] = 2;
    halo = std::make_unique<MultiFab<Dim>>(source.layout(), source.distribution(),
                                           source.local_rank(), 1, ghosts);
    result = std::make_unique<MultiFab<Dim>>(source.layout(), source.distribution(),
                                             source.local_rank(), 1, Extent<Dim>{});
    halo->set_val(Real(0));
    result->set_val(Real(0));
    const auto topology = BoundaryTopology<Dim>::physical();
    schedule.emplace(
        prepare_halo_schedule(*halo, geometry.domain(), topology,
                              runtime::system::auxiliary_ghost_detail::exact_halo_budget(
                                  *halo, geometry.domain(), topology)));
  } catch (...) {
    preparation_error = std::current_exception();
  }
  collectively_rethrow_exception(preparation_error, lane.communicator(),
                                 "initial provider derivative preparation failed collectively");
  for (std::size_t local = 0; local < source.local_size(); ++local)
    for_each_cell(source.box(local),
                  initial_derivative_detail::CopyScalar<Dim>{source.fab(local).view(),
                                                             halo->fab(local).view(), component});
  Kokkos::fence();
  const long remote_any =
      all_reduce_max(schedule->has_remote_jobs() ? 1L : 0L, lane.communicator());
  if (remote_any != 0) {
    // This private schedule is constructed once, executed synchronously and destroyed.
    // Its initial generation is one; zero denotes an unauthenticated exchange context.
    // HaloExchange retains its owning-lane and exact-rank validation.
    fill_boundary(*halo, *schedule, lane,
                  HaloExchangeContext{.context_generation = 1, .schedule_generation = 1});
  } else {
    // Local replay needs no communicator transport, including empty peer ranks.
    fill_boundary(*halo, *schedule);
  }
  long invalid_local = 0;
  for (std::size_t local = 0; local < halo->local_size(); ++local) {
    const initial_derivative_detail::Derivative<Dim> derivative{
        std::as_const(*halo).fab(local).view(), geometry.domain(), axis, geometry.spacing(axis)};
    invalid_local += static_cast<long>(for_each_cell_reduce_sum(
        halo->box(local), initial_derivative_detail::InvalidDerivative<Dim>{derivative}));
  }
  if (all_reduce_sum(invalid_local, lane.communicator()) != 0)
    throw std::runtime_error("initial provider derivative produced non-finite valid-cell values");
  for (std::size_t local = 0; local < halo->local_size(); ++local) {
    const initial_derivative_detail::Derivative<Dim> derivative{
        std::as_const(*halo).fab(local).view(), geometry.domain(), axis, geometry.spacing(axis)};
    for_each_cell(halo->box(local), initial_derivative_detail::WriteDerivative<Dim>{
                                        derivative, result->fab(local).view()});
  }
  Kokkos::fence();
  return result;
}

// Internal hierarchy caller authenticates the solved potential family before this kernel.
template <int Dim>
std::unique_ptr<MultiFab<Dim>> materialize_initial_derivative_from_prepared_halo(
    const MultiFab<Dim>& phi, const Geometry<Dim>& geometry, int axis, const ExecutionLane& lane) {
  std::unique_ptr<MultiFab<Dim>> candidate;
  std::exception_ptr preparation_error;
  try {
    if (phi.ncomp() != 1 || axis < 0 || axis >= Dim || phi.ghosts()[axis] < 1 ||
        geometry.domain().length(axis) < 3 || !(geometry.spacing(axis) > Real(0)) ||
        !std::isfinite(geometry.spacing(axis)))
      throw std::invalid_argument("prepared initial derivative lost its scalar/rank/halo contract");
    for (std::size_t local = 0; local < phi.local_size(); ++local) {
      const auto& box = phi.box(local);
      if (!geometry.domain().contains(box))
        throw std::invalid_argument(
            "prepared initial derivative patch lies outside its level domain");
      if ((box.lo[axis] == geometry.domain().lo[axis] ||
           box.hi[axis] == geometry.domain().hi[axis]) &&
          box.length(axis) < 3)
        throw std::invalid_argument(
            "global edge_order2 requires three valid patch cells or a qualified FAC halo2");
    }
    candidate = std::make_unique<MultiFab<Dim>>(phi.layout(), phi.distribution(), phi.local_rank(),
                                                1, Extent<Dim>{});
    candidate->set_val(Real(0));
  } catch (...) {
    preparation_error = std::current_exception();
  }
  collectively_rethrow_exception(
      preparation_error, lane.communicator(),
      "prepared initial derivative allocation/authority failed collectively");
  long invalid = 0;
  for (std::size_t local = 0; local < phi.local_size(); ++local) {
    const initial_derivative_detail::Derivative<Dim> stencil{
        phi.fab(local).view(), geometry.domain(), axis, geometry.spacing(axis)};
    invalid += static_cast<long>(for_each_cell_reduce_sum(
        phi.box(local), initial_derivative_detail::InvalidDerivative<Dim>{stencil}));
  }
  if (all_reduce_sum(invalid, lane.communicator()) != 0)
    throw std::runtime_error("prepared initial derivative contains non-finite valid values");
  for (std::size_t local = 0; local < phi.local_size(); ++local) {
    const initial_derivative_detail::Derivative<Dim> stencil{
        phi.fab(local).view(), geometry.domain(), axis, geometry.spacing(axis)};
    for_each_cell(phi.box(local), initial_derivative_detail::WriteDerivative<Dim>{
                                      stencil, candidate->fab(local).view()});
  }
  Kokkos::fence();
  return candidate;
}

}  // namespace pops::analytic
