/// @file
/// @brief Characteristic no-inflow on a cut-cell / staircase interface normal.
///
/// The incoming-mode projector is the same `characteristic_incoming_apply` kernel used on
/// Cartesian faces.  The flux Jacobian is oriented by the EB interface normal, never by a
/// cell-centred Cartesian-axis alias of that cut.

#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/embedded_boundary/cut_geometry.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <concepts>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace pops::nd {

template <class Model>
concept CharacteristicNoInflowNormalModel = requires(
    const Model& model, const typename Model::State& interior,
    const typename Model::State& reference, const Real* normal, typename Model::State& ghost) {
  { model.characteristic_no_inflow(interior, reference, normal, ghost) } -> std::same_as<bool>;
};

template <class Model>
concept CharacteristicNoInflowAxisModel = requires(
    const Model& model, const typename Model::State& interior,
    const typename Model::State& reference, typename Model::State& ghost) {
  { model.characteristic_no_inflow(interior, reference, 0, -1, ghost) } -> std::same_as<bool>;
};

template <int Dim>
POPS_HD bool axis_aligned_unit_normal(const RealVector<Dim>& normal, int& axis, int& outward_sign) {
  axis = -1;
  outward_sign = 0;
  int nonzero = 0;
  for (int direction = 0; direction < Dim; ++direction) {
    const Real value = normal[direction];
    if (value == Real(0))
      continue;
    ++nonzero;
    axis = direction;
    outward_sign = value > Real(0) ? 1 : -1;
    if (value != static_cast<Real>(outward_sign))
      return false;
  }
  return nonzero == 1 && axis >= 0;
}

/// Apply the model flux-Jacobian no-inflow kernel on an outward EB interface normal.
template <int Dim, class Model>
POPS_HD bool apply_characteristic_no_inflow_on_normal(
    const Model& model, const typename Model::State& interior,
    const typename Model::State& reference, const RealVector<Dim>& normal,
    typename Model::State& ghost) {
  Real mag2 = Real(0);
  for (int axis = 0; axis < Dim; ++axis) {
    if (!Kokkos::isfinite(normal[axis]))
      return false;
    mag2 += normal[axis] * normal[axis];
  }
  if (!(mag2 > Real(0)))
    return false;

  if constexpr (CharacteristicNoInflowNormalModel<Model>) {
    Real packed[3]{};
    for (int axis = 0; axis < Dim; ++axis)
      packed[axis] = normal[axis];
    return model.characteristic_no_inflow(interior, reference, packed, ghost);
  } else if constexpr (CharacteristicNoInflowAxisModel<Model>) {
    int axis = -1;
    int outward_sign = 0;
    if (!axis_aligned_unit_normal(normal, axis, outward_sign))
      return false;
    return model.characteristic_no_inflow(interior, reference, axis, outward_sign, ghost);
  } else {
    return false;
  }
}

template <int Dim, class Model>
struct FillEmbeddedCharacteristicKernel {
  Model model;
  FieldView<const Real, Dim> state{};
  FieldView<const Real, Dim> phi{};
  FieldView<const Real, Dim> active{};
  FieldView<Real, Dim> exterior{};
  typename Model::State reference{};
  Real theta_min = kEbCutFractionFloor;

  POPS_HD void operator()(const Index<Dim>& cell) const {
    if (active(cell) < Real(0.5)) {
      for (int component = 0; component < Model::n_vars; ++component)
        exterior(cell, component) = Real(0);
      return;
    }
    const CutCellFractions<Dim> fractions =
        cut_cell_fractions_from_phi_cell(phi, cell, theta_min);
    RealVector<Dim> normal{};
    if (!cut_cell_interface_normal(fractions, normal)) {
      for (int component = 0; component < Model::n_vars; ++component)
        exterior(cell, component) = state(cell, component);
      return;
    }
    typename Model::State interior{};
    typename Model::State ghost{};
    for (int component = 0; component < Model::n_vars; ++component)
      interior[component] = state(cell, component);
    if (!apply_characteristic_no_inflow_on_normal<Dim>(model, interior, reference, normal,
                                                       ghost)) {
      for (int component = 0; component < Model::n_vars; ++component)
        exterior(cell, component) = std::numeric_limits<Real>::quiet_NaN();
      return;
    }
    for (int component = 0; component < Model::n_vars; ++component)
      exterior(cell, component) = ghost[component];
  }
};

template <int Dim, class Model, class MemorySpace>
void fill_embedded_characteristic_no_inflow(
    const Model& model, const Fab<Dim, MemorySpace>& state, const Fab<Dim, MemorySpace>& phi,
    const Fab<Dim, MemorySpace>& active, const typename Model::State& reference,
    Fab<Dim, MemorySpace>& exterior, Real theta_min = kEbCutFractionFloor) {
  if (state.ncomp() != Model::n_vars || exterior.ncomp() != Model::n_vars || phi.ncomp() != 1 ||
      active.ncomp() != 1 || !(phi.box() == state.box()) || !(active.box() == state.box()) ||
      !(exterior.box() == state.box()) || !phi.grown_box().contains(state.box().grow(1)) ||
      !active.grown_box().contains(state.box().grow(1)))
    throw std::invalid_argument(
        "embedded characteristic no-inflow requires state, phi, active mask, and exterior on the "
        "same patch, with one phi/mask ghost");
  for_each_cell(state.box(),
                FillEmbeddedCharacteristicKernel<Dim, Model>{
                    model, state.view(), phi.view(), active.view(), exterior.view(), reference,
                    theta_min});
  device_fence();
}

template <int Dim, class Model, class MemorySpace>
void fill_embedded_characteristic_no_inflow(
    const Model& model, const MultiFab<Dim, MemorySpace>& state,
    const MultiFab<Dim, MemorySpace>& phi, const MultiFab<Dim, MemorySpace>& active,
    const typename Model::State& reference, MultiFab<Dim, MemorySpace>& exterior,
    Real theta_min = kEbCutFractionFloor) {
  if (!(state.layout() == phi.layout()) || !(state.layout() == active.layout()) ||
      !(state.layout() == exterior.layout()) || !(state.distribution() == phi.distribution()) ||
      !(state.distribution() == active.distribution()) ||
      !(state.distribution() == exterior.distribution()) ||
      !(state.local_rank() == phi.local_rank()) || !(state.local_rank() == active.local_rank()) ||
      !(state.local_rank() == exterior.local_rank()) || state.local_size() != phi.local_size() ||
      state.local_size() != active.local_size() || state.local_size() != exterior.local_size())
    throw std::invalid_argument(
        "embedded characteristic no-inflow MultiFab layouts differ");
  for (std::size_t local = 0; local < state.local_size(); ++local)
    ::pops::nd::fill_embedded_characteristic_no_inflow(
        model, state.fab(local), phi.fab(local), active.fab(local), reference,
        exterior.fab(local), theta_min);
}

}  // namespace pops::nd
