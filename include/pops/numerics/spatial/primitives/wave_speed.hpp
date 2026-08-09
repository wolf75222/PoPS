/// @file
/// @brief Compile-time-ranked collective wave-speed and step-bound reductions.

#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>
#include <pops/parallel/comm.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace pops::nd {

namespace wave_speed_detail {

POPS_HD inline bool nonnegative_finite(Real value) {
  return value >= Real(0) && Kokkos::isfinite(value);
}

template <int Axis, int Dim, class Model>
POPS_HD Real maximum_axis_wave_speed(const Model& model, const typename Model::State& state) {
  static_assert(Axis >= 0 && Axis < Dim);
  const Real current = model.template max_wave_speed<Axis>(state);
  if (!nonnegative_finite(current))
    return std::numeric_limits<Real>::infinity();
  if constexpr (Axis + 1 < Dim) {
    const Real remaining = maximum_axis_wave_speed<Axis + 1, Dim>(model, state);
    return current > remaining ? current : remaining;
  }
  return current;
}

template <int Axis, int Dim, class Model>
POPS_HD Real maximum_axis_stability_speed(const Model& model, const typename Model::State& state) {
  static_assert(Axis >= 0 && Axis < Dim);
  const Real current = model.template stability_speed<Axis>(state);
  if (!nonnegative_finite(current))
    return std::numeric_limits<Real>::infinity();
  if constexpr (Axis + 1 < Dim) {
    const Real remaining = maximum_axis_stability_speed<Axis + 1, Dim>(model, state);
    return current > remaining ? current : remaining;
  }
  return current;
}

struct WaveSpeedQuantity {
  template <int Dim, class Model>
  POPS_HD Real operator()(const Model& model, const typename Model::State& state) const {
    return maximum_axis_wave_speed<0, Dim>(model, state);
  }
};

struct StabilitySpeedQuantity {
  template <int Dim, class Model>
  POPS_HD Real operator()(const Model& model, const typename Model::State& state) const {
    return maximum_axis_stability_speed<0, Dim>(model, state);
  }
};

struct SourceFrequencyQuantity {
  template <int Dim, class Model>
  POPS_HD Real operator()(const Model& model, const typename Model::State& state) const {
    (void)Dim;
    const Real value = model.source_frequency(state);
    return nonnegative_finite(value) ? value : std::numeric_limits<Real>::infinity();
  }
};

template <int Dim, class Model, class Quantity>
struct MaximumCellQuantity {
  static_assert(Model::dimension == Dim,
                "wave-speed model rank must match the compiled field specialization");
  Model model;
  FieldView<const Real, Dim> state{};
  Quantity quantity{};

  POPS_HD Real operator()(const Index<Dim>& index) const {
    return quantity.template operator()<Dim>(model, pops::load_state<Model>(state, index));
  }
};

template <int Dim, class CellQuantity>
struct ActiveCellQuantity {
  CellQuantity quantity;
  FieldView<const Real, Dim> active_cells{};

  POPS_HD Real operator()(const Index<Dim>& index) const {
    return active_cells(index) >= Real(0.5) ? quantity(index) : std::numeric_limits<Real>::lowest();
  }
};

template <int Dim, class CellQuantity>
struct CutCellQuantity {
  CellQuantity quantity;
  FieldView<const Real, Dim> active_cells{};
  FieldView<const Real, Dim> inverse_volume_fraction{};

  POPS_HD Real operator()(const Index<Dim>& index) const {
    if (active_cells(index) < Real(0.5))
      return std::numeric_limits<Real>::lowest();
    const Real value = quantity(index);
    const Real inverse = inverse_volume_fraction(index);
    if (!nonnegative_finite(value) || !nonnegative_finite(inverse))
      return std::numeric_limits<Real>::infinity();
    const Real scaled = value * inverse;
    return nonnegative_finite(scaled) ? scaled : std::numeric_limits<Real>::infinity();
  }
};

template <int Dim, class LeftMemory, class RightMemory>
void require_same_layout(const MultiFab<Dim, LeftMemory>& left,
                         const MultiFab<Dim, RightMemory>& right, const char* operation) {
  if (!(left.layout() == right.layout()) || !(left.distribution() == right.distribution()) ||
      !(left.local_rank() == right.local_rank()) || left.local_size() != right.local_size())
    throw std::invalid_argument(std::string(operation) + ": field layouts differ");
}

template <int Dim, class MemorySpace>
void require_mask(const MultiFab<Dim, MemorySpace>& state,
                  const MultiFab<Dim, MemorySpace>& active_cells, const char* operation) {
  require_same_layout(state, active_cells, operation);
  if (active_cells.ncomp() != 1)
    throw std::invalid_argument(std::string(operation) + ": active-cell mask must be scalar");
}

template <int Dim, class MemorySpace>
void require_cut_metrics(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         const MultiFab<Dim, MemorySpace>& inverse_volume_fraction,
                         const char* operation) {
  require_mask(state, active_cells, operation);
  require_same_layout(state, inverse_volume_fraction, operation);
  if (inverse_volume_fraction.ncomp() != 1)
    throw std::invalid_argument(std::string(operation) +
                                ": inverse-volume fraction must be scalar");
}

inline Real publish_nonnegative_maximum(Real local_maximum, const char* quantity) {
  const bool invalid = !std::isfinite(static_cast<double>(local_maximum)) &&
                       local_maximum != std::numeric_limits<Real>::lowest();
  double payload[2] = {invalid ? 1.0 : 0.0, local_maximum == std::numeric_limits<Real>::lowest()
                                                ? 0.0
                                                : static_cast<double>(local_maximum)};
  all_reduce_max_inplace(payload, 2);
  if (payload[0] != 0.0)
    throw std::domain_error(std::string(quantity) +
                            " returned a negative or non-finite value on an active cell");
  return static_cast<Real>(payload[1]);
}

template <int Dim, class MemorySpace, class MakeKernel>
Real collective_maximum(const MultiFab<Dim, MemorySpace>& state, MakeKernel make_kernel,
                        const char* quantity) {
  Real local_maximum = std::numeric_limits<Real>::lowest();
  for (std::size_t local = 0; local < state.local_size(); ++local) {
    const Real patch = for_each_cell_reduce_max(state.box(local), make_kernel(local));
    if (patch > local_maximum)
      local_maximum = patch;
  }
  return publish_nonnegative_maximum(local_maximum, quantity);
}

template <int Axis, int Dim, class CellQuantity>
struct MatchingCoordinate {
  CellQuantity quantity;
  Real target = Real(0);
  Index<Dim> selected{};

  POPS_HD Real operator()(const Index<Dim>& index) const {
    if (quantity(index) != target)
      return std::numeric_limits<Real>::lowest();
    for (int higher = Axis + 1; higher < Dim; ++higher)
      if (index[higher] != selected[higher])
        return std::numeric_limits<Real>::lowest();
    return -static_cast<Real>(index[Axis]);
  }
};

template <int Axis, int Dim, class MemorySpace, class MakeKernel>
bool select_hotspot_coordinates(const MultiFab<Dim, MemorySpace>& state, Real target,
                                MakeKernel& make_kernel, Index<Dim>& selected) {
  Real local_best = std::numeric_limits<Real>::lowest();
  for (std::size_t local = 0; local < state.local_size(); ++local) {
    const auto quantity = make_kernel(local);
    const Real patch = for_each_cell_reduce_max(
        state.box(local),
        MatchingCoordinate<Axis, Dim, decltype(quantity)>{quantity, target, selected});
    if (patch > local_best)
      local_best = patch;
  }
  const double global_best = all_reduce_max(static_cast<double>(local_best));
  if (global_best == static_cast<double>(std::numeric_limits<Real>::lowest()))
    return false;
  selected[Axis] = static_cast<int>(-global_best);
  if constexpr (Axis > 0)
    return select_hotspot_coordinates<Axis - 1>(state, target, make_kernel, selected);
  return true;
}

template <int Dim, class MemorySpace, class MakeKernel>
Index<Dim> locate_hotspot(const MultiFab<Dim, MemorySpace>& state, Real target,
                          MakeKernel make_kernel, bool& found) {
  Index<Dim> selected{};
  found = select_hotspot_coordinates<Dim - 1>(state, target, make_kernel, selected);
  return selected;
}

template <int Dim, class Model, class MemorySpace, class Quantity>
Real maximum_unmasked(const Model& model, const MultiFab<Dim, MemorySpace>& state,
                      Quantity quantity, const char* name) {
  if (state.ncomp() != Model::n_vars)
    throw std::invalid_argument(std::string(name) + ": state component count differs from model");
  return collective_maximum(
      state,
      [&](std::size_t local) {
        return MaximumCellQuantity<Dim, Model, Quantity>{model, state.fab(local).view(), quantity};
      },
      name);
}

template <int Dim, class Model, class MemorySpace, class Quantity>
Real maximum_active(const Model& model, const MultiFab<Dim, MemorySpace>& state,
                    const MultiFab<Dim, MemorySpace>& active_cells, Quantity quantity,
                    const char* name) {
  if (state.ncomp() != Model::n_vars)
    throw std::invalid_argument(std::string(name) + ": state component count differs from model");
  require_mask(state, active_cells, name);
  return collective_maximum(
      state,
      [&](std::size_t local) {
        const MaximumCellQuantity<Dim, Model, Quantity> cell{model, state.fab(local).view(),
                                                             quantity};
        return ActiveCellQuantity<Dim, decltype(cell)>{cell, active_cells.fab(local).view()};
      },
      name);
}

template <int Dim, class Model, class MemorySpace, class Quantity>
Real maximum_cut(const Model& model, const MultiFab<Dim, MemorySpace>& state,
                 const MultiFab<Dim, MemorySpace>& active_cells,
                 const MultiFab<Dim, MemorySpace>& inverse_volume_fraction, Quantity quantity,
                 const char* name) {
  if (state.ncomp() != Model::n_vars)
    throw std::invalid_argument(std::string(name) + ": state component count differs from model");
  require_cut_metrics(state, active_cells, inverse_volume_fraction, name);
  return collective_maximum(
      state,
      [&](std::size_t local) {
        const MaximumCellQuantity<Dim, Model, Quantity> cell{model, state.fab(local).view(),
                                                             quantity};
        return CutCellQuantity<Dim, decltype(cell)>{cell, active_cells.fab(local).view(),
                                                    inverse_volume_fraction.fab(local).view()};
      },
      name);
}

struct MinimumStabilityStep {
  template <int Dim, class Model>
  POPS_HD Real operator()(const Model& model, const typename Model::State& state) const {
    (void)Dim;
    const Real value = model.stability_dt(state);
    if (value == std::numeric_limits<Real>::infinity())
      return std::numeric_limits<Real>::lowest();
    if (!(value > Real(0)) || !Kokkos::isfinite(value))
      return std::numeric_limits<Real>::infinity();
    return -value;
  }
};

struct FiniteStabilityStep {
  template <int Dim, class Model>
  POPS_HD Real operator()(const Model& model, const typename Model::State& state) const {
    (void)Dim;
    const Real value = model.stability_dt(state);
    return value > Real(0) && Kokkos::isfinite(value) ? Real(1) : Real(0);
  }
};

template <int Dim, class Model, class MemorySpace, class MakeKernel, class MakeConstrainedKernel>
Real minimum_step(const MultiFab<Dim, MemorySpace>& state, MakeKernel make_kernel,
                  MakeConstrainedKernel make_constrained_kernel) {
  Real local = std::numeric_limits<Real>::lowest();
  for (std::size_t patch = 0; patch < state.local_size(); ++patch) {
    const Real encoded = for_each_cell_reduce_max(state.box(patch), make_kernel(patch));
    if (encoded > local)
      local = encoded;
  }
  const double global = all_reduce_max(static_cast<double>(local));
  if (!std::isfinite(global) && global > 0.0)
    throw std::domain_error(
        "stability_dt returned zero, a negative value, or a non-finite value other than +inf "
        "on an active cell");
  if (global != static_cast<double>(std::numeric_limits<Real>::lowest()))
    return static_cast<Real>(-global);

  Real local_constrained = Real(0);
  for (std::size_t patch = 0; patch < state.local_size(); ++patch) {
    const Real constrained =
        for_each_cell_reduce_max(state.box(patch), make_constrained_kernel(patch));
    if (constrained > local_constrained)
      local_constrained = constrained;
  }
  return all_reduce_max(static_cast<double>(local_constrained)) > 0.0
             ? std::numeric_limits<Real>::max()
             : Real(0);
}

}  // namespace wave_speed_detail

template <int Dim>
struct WaveSpeedHotspot {
  Real speed = Real(0);
  Index<Dim> cell{};
  bool found = false;
};

template <int Dim, class Model, class MemorySpace>
Real max_wave_speed_mf(const Model& model, const MultiFab<Dim, MemorySpace>& state) {
  return wave_speed_detail::maximum_unmasked(model, state, wave_speed_detail::WaveSpeedQuantity{},
                                             "max_wave_speed");
}

template <int Dim, class Model, class MemorySpace>
Real max_wave_speed_mf(const Model& model, const MultiFab<Dim, MemorySpace>& state,
                       const MultiFab<Dim, MemorySpace>& active_cells) {
  return wave_speed_detail::maximum_active(
      model, state, active_cells, wave_speed_detail::WaveSpeedQuantity{}, "max_wave_speed");
}

template <int Dim, class Model, class MemorySpace>
Real max_wave_speed_mf(const Model& model, const MultiFab<Dim, MemorySpace>& state,
                       const MultiFab<Dim, MemorySpace>& active_cells,
                       const MultiFab<Dim, MemorySpace>& inverse_volume_fraction) {
  return wave_speed_detail::maximum_cut(model, state, active_cells, inverse_volume_fraction,
                                        wave_speed_detail::WaveSpeedQuantity{}, "max_wave_speed");
}

template <int Dim, class Model, class MemorySpace>
WaveSpeedHotspot<Dim> max_wave_speed_hotspot_mf(const Model& model,
                                                const MultiFab<Dim, MemorySpace>& state) {
  WaveSpeedHotspot<Dim> result{};
  result.speed = max_wave_speed_mf(model, state);
  auto make_kernel = [&](std::size_t local) {
    return wave_speed_detail::MaximumCellQuantity<Dim, Model, wave_speed_detail::WaveSpeedQuantity>{
        model, state.fab(local).view(), {}};
  };
  result.cell = wave_speed_detail::locate_hotspot(state, result.speed, make_kernel, result.found);
  return result;
}

template <int Dim, class Model, class MemorySpace>
WaveSpeedHotspot<Dim> max_wave_speed_hotspot_mf(const Model& model,
                                                const MultiFab<Dim, MemorySpace>& state,
                                                const MultiFab<Dim, MemorySpace>& active_cells) {
  wave_speed_detail::require_mask(state, active_cells, "max_wave_speed_hotspot");
  WaveSpeedHotspot<Dim> result{};
  result.speed = max_wave_speed_mf(model, state, active_cells);
  auto make_kernel = [&](std::size_t local) {
    const wave_speed_detail::MaximumCellQuantity<Dim, Model, wave_speed_detail::WaveSpeedQuantity>
        cell{model, state.fab(local).view(), {}};
    return wave_speed_detail::ActiveCellQuantity<Dim, decltype(cell)>{
        cell, active_cells.fab(local).view()};
  };
  result.cell = wave_speed_detail::locate_hotspot(state, result.speed, make_kernel, result.found);
  return result;
}

template <int Dim, class Model, class MemorySpace>
WaveSpeedHotspot<Dim> max_wave_speed_hotspot_mf(
    const Model& model, const MultiFab<Dim, MemorySpace>& state,
    const MultiFab<Dim, MemorySpace>& active_cells,
    const MultiFab<Dim, MemorySpace>& inverse_volume_fraction) {
  wave_speed_detail::require_cut_metrics(state, active_cells, inverse_volume_fraction,
                                         "max_wave_speed_hotspot");
  WaveSpeedHotspot<Dim> result{};
  result.speed = max_wave_speed_mf(model, state, active_cells, inverse_volume_fraction);
  auto make_kernel = [&](std::size_t local) {
    const wave_speed_detail::MaximumCellQuantity<Dim, Model, wave_speed_detail::WaveSpeedQuantity>
        cell{model, state.fab(local).view(), {}};
    return wave_speed_detail::CutCellQuantity<Dim, decltype(cell)>{
        cell, active_cells.fab(local).view(), inverse_volume_fraction.fab(local).view()};
  };
  result.cell = wave_speed_detail::locate_hotspot(state, result.speed, make_kernel, result.found);
  return result;
}

template <int Dim, class Model, class MemorySpace>
Real max_stability_speed_mf(const Model& model, const MultiFab<Dim, MemorySpace>& state) {
  return wave_speed_detail::maximum_unmasked(
      model, state, wave_speed_detail::StabilitySpeedQuantity{}, "stability_speed");
}

template <int Dim, class Model, class MemorySpace>
Real max_stability_speed_mf(const Model& model, const MultiFab<Dim, MemorySpace>& state,
                            const MultiFab<Dim, MemorySpace>& active_cells) {
  return wave_speed_detail::maximum_active(
      model, state, active_cells, wave_speed_detail::StabilitySpeedQuantity{}, "stability_speed");
}

template <int Dim, class Model, class MemorySpace>
Real max_stability_speed_mf(const Model& model, const MultiFab<Dim, MemorySpace>& state,
                            const MultiFab<Dim, MemorySpace>& active_cells,
                            const MultiFab<Dim, MemorySpace>& inverse_volume_fraction) {
  return wave_speed_detail::maximum_cut(model, state, active_cells, inverse_volume_fraction,
                                        wave_speed_detail::StabilitySpeedQuantity{},
                                        "stability_speed");
}

template <int Dim, class Model, class MemorySpace>
Real max_source_frequency_mf(const Model& model, const MultiFab<Dim, MemorySpace>& state) {
  return wave_speed_detail::maximum_unmasked(
      model, state, wave_speed_detail::SourceFrequencyQuantity{}, "source_frequency");
}

template <int Dim, class Model, class MemorySpace>
Real max_source_frequency_mf(const Model& model, const MultiFab<Dim, MemorySpace>& state,
                             const MultiFab<Dim, MemorySpace>& active_cells) {
  return wave_speed_detail::maximum_active(
      model, state, active_cells, wave_speed_detail::SourceFrequencyQuantity{}, "source_frequency");
}

template <int Dim, class Model, class MemorySpace>
Real min_stability_dt_mf(const Model& model, const MultiFab<Dim, MemorySpace>& state) {
  if (state.ncomp() != Model::n_vars)
    throw std::invalid_argument("stability_dt: state component count differs from model");
  return wave_speed_detail::minimum_step<Dim, Model>(
      state,
      [&](std::size_t local) {
        return wave_speed_detail::MaximumCellQuantity<Dim, Model,
                                                      wave_speed_detail::MinimumStabilityStep>{
            model, state.fab(local).view(), {}};
      },
      [&](std::size_t local) {
        return wave_speed_detail::MaximumCellQuantity<Dim, Model,
                                                      wave_speed_detail::FiniteStabilityStep>{
            model, state.fab(local).view(), {}};
      });
}

template <int Dim, class Model, class MemorySpace>
Real min_stability_dt_mf(const Model& model, const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& active_cells) {
  if (state.ncomp() != Model::n_vars)
    throw std::invalid_argument("stability_dt: state component count differs from model");
  wave_speed_detail::require_mask(state, active_cells, "stability_dt");
  return wave_speed_detail::minimum_step<Dim, Model>(
      state,
      [&](std::size_t local) {
        const wave_speed_detail::MaximumCellQuantity<Dim, Model,
                                                     wave_speed_detail::MinimumStabilityStep>
            cell{model, state.fab(local).view(), {}};
        return wave_speed_detail::ActiveCellQuantity<Dim, decltype(cell)>{
            cell, active_cells.fab(local).view()};
      },
      [&](std::size_t local) {
        const wave_speed_detail::MaximumCellQuantity<Dim, Model,
                                                     wave_speed_detail::FiniteStabilityStep>
            cell{model, state.fab(local).view(), {}};
        return wave_speed_detail::ActiveCellQuantity<Dim, decltype(cell)>{
            cell, active_cells.fab(local).view()};
      });
}

}  // namespace pops::nd
