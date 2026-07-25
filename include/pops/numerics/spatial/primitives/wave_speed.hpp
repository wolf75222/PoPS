/// @file
/// @brief Global wave-speed and step-bound reductions.
///
/// CONTRACT: the CFL and step-bound reductions over a whole MultiFab (device reduction + MPI
/// all_reduce).
///   - max_wave_speed_mf: global max CFL speed (collective under MPI).
///   - max_wave_speed_hotspot_mf: cell dominating the CFL bound (diagnostic, ADC-182).
///   - max_stability_speed_mf / max_source_frequency_mf / min_stability_dt_mf: optional
///     step-bound traits (HasStabilitySpeed / HasSourceFrequency / HasStabilityDt).
///
/// COLLECTIVE UNDER MPI: every reduction aggregates via all_reduce over ALL ranks; without it
/// each rank would choose a different dt and the simulation desynchronizes (see notes below).
/// A negative or non-finite speed/frequency is a model-contract failure: one rank observing it
/// makes every rank throw at the reduction boundary.

#pragma once

#include <pops/core/state/state.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>  // reduce_max_cell, reduce_min_cell
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>  // load_state, load_aux, aux_comps
#include <pops/parallel/comm.hpp>                             // all_reduce_max, all_reduce_min

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace pops {

namespace detail {
POPS_HD inline bool is_nonnegative_finite(Real value) {
  return value >= Real(0) && value < std::numeric_limits<Real>::infinity();
}

/// Device-side accumulator for quantities whose mathematical contract is a finite,
/// non-negative scalar.  Invalid samples are encoded as +inf because Kokkos::Max must not be
/// allowed to swallow NaN or a negative value behind its zero identity.  The host seam turns that
/// marker into an explicit collective status before publishing the reduced maximum.
POPS_HD inline void accumulate_nonnegative_finite(Real value, Real& maximum) {
  if (!is_nonnegative_finite(value)) {
    maximum = std::numeric_limits<Real>::infinity();
    return;
  }
  if (value > maximum)
    maximum = value;
}

/// Publish a validated maximum with one MPI collective carrying both the invalid marker and the
/// value.  Ranks with no local Fab contribute {0, 0}; if one rank observed an invalid sample, every
/// rank receives the same status and throws at the same collective boundary.
inline Real publish_nonnegative_maximum(Real local_maximum, const char* quantity) {
  const bool local_invalid = local_maximum < Real(0) ||
                             !std::isfinite(static_cast<double>(local_maximum));
  double reduction[2] = {local_invalid ? 1.0 : 0.0,
                         local_invalid ? 0.0 : static_cast<double>(local_maximum)};
  all_reduce_max_inplace(reduction, 2);
  if (reduction[0] != 0.0)
    throw std::domain_error(std::string(quantity) +
                            " returned a negative or non-finite value on an active cell");
  return static_cast<Real>(reduction[1]);
}

/// MaxWaveSpeedKernel<Model>: device reduction functor for max_wave_speed_mf.
///
/// Accumulates the max of the wave speeds in both directions at cell (i,j).
/// Named functor (and not an extended lambda): robust device emission from an external TU
/// (add_compiled_model). POPS_HD; invalid model outputs become a host-visible marker.
template <class Model>
struct MaxWaveSpeedKernel {
  Model model;
  ConstArray4 u, a;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.max_wave_speed(s, ax, 0);
    const Real wy = model.max_wave_speed(s, ax, 1);
    accumulate_nonnegative_finite(wx, acc);
    accumulate_nonnegative_finite(wy, acc);
  }
};

template <class Kernel>
struct ActiveCellReductionKernel {
  Kernel kernel;
  ConstArray4 active_cells;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (active_cells(i, j, 0) >= Real(0.5))
      kernel(i, j, acc);
  }
};

template <class Model>
struct CutCellWaveSpeedKernel {
  Model model;
  ConstArray4 u, a, active_cells, inverse_volume_fraction;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (active_cells(i, j, 0) < Real(0.5))
      return;
    const auto state = load_state<Model>(u, i, j);
    const Aux aux = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.max_wave_speed(state, aux, 0);
    const Real wy = model.max_wave_speed(state, aux, 1);
    if (!is_nonnegative_finite(wx) || !is_nonnegative_finite(wy)) {
      acc = std::numeric_limits<Real>::infinity();
      return;
    }
    const Real wave = (wx > wy ? wx : wy) * inverse_volume_fraction(i, j, 0);
    accumulate_nonnegative_finite(wave, acc);
  }
};
}  // namespace detail

/// max_wave_speed_mf: global max of the wave speed over the whole MultiFab (CFL).
///
/// Reduce over all local boxes then all_reduce_max over all MPI ranks.
/// Without the all_reduce, each rank only sees its boxes and step_cfl computes a different
/// dt per rank (desynchronization / divergence). In serial all_reduce_max is the identity.
/// For a model without transport (max_wave_speed = 0 everywhere) -> returns 0 (step unconstrained).
/// @throws std::domain_error collectively if an active cell returns a negative or non-finite speed.
//
// COLLECTIVE UNDER MPI: we aggregate via all_reduce_max over ALL ranks (same convention as
// AmrCouplerMp::max_wave_speed and GeometricMG::current_residual). Without this all-reduce, each
// rank only sees the max of ITS boxes: step_cfl / step_adaptive then choose a DIFFERENT dt per
// rank (the rank whose local max is lower takes too large a step) and the simulation diverges or
// desynchronizes the ranks. In serial all_reduce_max is the identity (behavior unchanged).
template <class Model>
inline Real max_wave_speed_mf(const Model& model, const MultiFab& U, const MultiFab& aux) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(m, reduce_max_cell(U.box(li), detail::MaxWaveSpeedKernel<Model>{model, u, a}));
  }
  return detail::publish_nonnegative_maximum(m, "max_wave_speed");
}

/// Same collective reduction restricted to a prepared active-cell set.  The mask is a generic
/// geometry-provider result, independent of the analytic shape that produced it.
template <class Model>
inline Real max_wave_speed_mf(const Model& model, const MultiFab& U, const MultiFab& aux,
                              const MultiFab& active_cells) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(
        m, reduce_max_cell(U.box(li),
                           detail::ActiveCellReductionKernel<detail::MaxWaveSpeedKernel<Model>>{
                               detail::MaxWaveSpeedKernel<Model>{model, u, a},
                               active_cells.fab(li).const_array()}));
  }
  return detail::publish_nonnegative_maximum(m, "max_wave_speed");
}

/// Cut-cell transport bound. The prepared operator multiplies flux divergence by 1/kappa, so the
/// stable effective speed is bounded by lambda/kappa on each active cell. This conservative bound
/// deliberately uses the same frozen inverse-volume metric as the residual.
template <class Model>
inline Real max_wave_speed_mf(const Model& model, const MultiFab& U, const MultiFab& aux,
                              const MultiFab& active_cells,
                              const MultiFab& inverse_volume_fraction) {
  Real maximum = 0;
  for (int local = 0; local < U.local_size(); ++local)
    maximum = std::max(
        maximum,
        reduce_max_cell(U.box(local),
                        detail::CutCellWaveSpeedKernel<Model>{
                            model, U.fab(local).const_array(), aux.fab(local).const_array(),
                            active_cells.fab(local).const_array(),
                            inverse_volume_fraction.fab(local).const_array()}));
  return detail::publish_nonnegative_maximum(maximum, "max_wave_speed");
}

namespace detail {
/// Locates the cell DOMINATING the CFL (dt_hotspot diagnostic, ADC-182): EQUALITY scan
/// of the recomputed w -- same functor and same data as MaxWaveSpeedKernel, hence bit-equal
/// to the max returned by max_wave_speed_mf -- which encodes the GLOBAL index j*nx + i as
/// Real (exact as long as nx*ny < 2^53) and reduces to the MIN (first cell in lexicographic
/// order: deterministic). NAMED functor (cross-TU instantiation under nvcc).
template <class Model>
struct WaveSpeedMatchKernel {
  Model model;
  ConstArray4 u, a;
  Real target;
  Real nx;  // encoding stride (nx of the DOMAIN, global indices)
  POPS_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.max_wave_speed(s, ax, 0);
    const Real wy = model.max_wave_speed(s, ax, 1);
    const Real w = wx > wy ? wx : wy;
    if (w == target) {
      const Real idx = static_cast<Real>(j) * nx + static_cast<Real>(i);
      if (idx < acc)
        acc = idx;
    }
  }
};

template <class Model>
struct ActiveWaveSpeedMatchKernel {
  WaveSpeedMatchKernel<Model> match;
  ConstArray4 active_cells;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (active_cells(i, j, 0) >= Real(0.5))
      match(i, j, acc);
  }
};

template <class Model>
struct CutCellWaveSpeedMatchKernel {
  Model model;
  ConstArray4 u, a, active_cells, inverse_volume_fraction;
  Real target;
  Real nx;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (active_cells(i, j, 0) < Real(0.5))
      return;
    const auto state = load_state<Model>(u, i, j);
    const Aux aux = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.max_wave_speed(state, aux, 0);
    const Real wy = model.max_wave_speed(state, aux, 1);
    const Real wave = (wx > wy ? wx : wy) * inverse_volume_fraction(i, j, 0);
    if (wave == target) {
      const Real index = static_cast<Real>(j) * nx + static_cast<Real>(i);
      if (index < acc)
        acc = index;
    }
  }
};
}  // namespace detail

/// dt_hotspot diagnostic (ADC-182): the cell (GLOBAL indices) that dominates the block's transport
/// CFL bound, and its speed w = max(wx, wy). ON DEMAND only -- two full passes (max then location
/// by bit-exact equality), step_cfl does not touch it (bit-identical). MPI: all_reduce of the max
/// then all_reduce_min of the encoded index (+inf on the non-holder ranks). @p nx: domain width
/// (encoding j*nx + i).
template <class Model>
inline void max_wave_speed_hotspot_mf(const Model& model, const MultiFab& U, const MultiFab& aux,
                                      int nx, Real& w_out, int& i_out, int& j_out) {
  const Real w = max_wave_speed_mf(model, U, aux);
  Real best = std::numeric_limits<Real>::infinity();
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    best = std::min(best, reduce_min_cell(U.box(li), detail::WaveSpeedMatchKernel<Model>{
                                                         model, u, a, w, static_cast<Real>(nx)}));
  }
  best = static_cast<Real>(all_reduce_min(static_cast<double>(best)));
  w_out = w;
  // identity of Kokkos::Min = max_real (finite): a rank/box without a cell equaling the max
  // leaves this value -> we only decode if a REAL index was encoded.
  if (best >= Real(0) && best < std::numeric_limits<Real>::max() * Real(0.5)) {
    const long long idx = static_cast<long long>(best);
    i_out = static_cast<int>(idx % nx);
    j_out = static_cast<int>(idx / nx);
  } else {  // empty domain / degenerate state: no cell (w may be 0)
    i_out = -1;
    j_out = -1;
  }
}

template <class Model>
inline void max_wave_speed_hotspot_mf(const Model& model, const MultiFab& U, const MultiFab& aux,
                                      const MultiFab& active_cells, int nx, Real& w_out, int& i_out,
                                      int& j_out) {
  const Real w = max_wave_speed_mf(model, U, aux, active_cells);
  Real best = std::numeric_limits<Real>::infinity();
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    best = std::min(best, reduce_min_cell(U.box(li), detail::ActiveWaveSpeedMatchKernel<Model>{
                                                         detail::WaveSpeedMatchKernel<Model>{
                                                             model, u, a, w, static_cast<Real>(nx)},
                                                         active_cells.fab(li).const_array()}));
  }
  best = static_cast<Real>(all_reduce_min(static_cast<double>(best)));
  w_out = w;
  if (best >= Real(0) && best < std::numeric_limits<Real>::max() * Real(0.5)) {
    const long long index = static_cast<long long>(best);
    i_out = static_cast<int>(index % nx);
    j_out = static_cast<int>(index / nx);
  } else {
    i_out = -1;
    j_out = -1;
  }
}

template <class Model>
inline void max_wave_speed_hotspot_mf(const Model& model, const MultiFab& U, const MultiFab& aux,
                                      const MultiFab& active_cells,
                                      const MultiFab& inverse_volume_fraction, int nx, Real& w_out,
                                      int& i_out, int& j_out) {
  const Real wave = max_wave_speed_mf(model, U, aux, active_cells, inverse_volume_fraction);
  Real best = std::numeric_limits<Real>::infinity();
  for (int local = 0; local < U.local_size(); ++local)
    best = std::min(
        best, reduce_min_cell(U.box(local),
                              detail::CutCellWaveSpeedMatchKernel<Model>{
                                  model, U.fab(local).const_array(), aux.fab(local).const_array(),
                                  active_cells.fab(local).const_array(),
                                  inverse_volume_fraction.fab(local).const_array(), wave,
                                  static_cast<Real>(nx)}));
  best = static_cast<Real>(all_reduce_min(static_cast<double>(best)));
  w_out = wave;
  if (best >= Real(0) && best < std::numeric_limits<Real>::max() * Real(0.5)) {
    const long long index = static_cast<long long>(best);
    i_out = static_cast<int>(index % nx);
    j_out = static_cast<int>(index / nx);
  } else {
    i_out = -1;
    j_out = -1;
  }
}

// ============================================================================
// OPTIONAL STEP-BOUND REDUCTIONS (audit 2026-06, step_cfl effort).
// Counterparts of max_wave_speed_mf for the HasStabilitySpeed / HasSourceFrequency /
// HasStabilityDt traits (cf. core/physical_model.hpp). Same conventions: reduction via the seam
// (device under Kokkos), MPI all_reduce (without which each rank would choose a different dt).
// Instantiated ONLY for a model declaring the trait (if constexpr on the block_builder side):
// zero codegen, zero cost for a legacy model.
// ============================================================================

namespace detail {
/// StabilitySpeedKernel: max over cells/directions of model.stability_speed (replaces
/// MaxWaveSpeedKernel when the trait is declared). Named functor (device-clean cross-TU).
template <class Model>
struct StabilitySpeedKernel {
  Model model;
  ConstArray4 u, a;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.stability_speed(s, ax, 0);
    const Real wy = model.stability_speed(s, ax, 1);
    accumulate_nonnegative_finite(wx, acc);
    accumulate_nonnegative_finite(wy, acc);
  }
};

template <class Model>
struct CutCellStabilitySpeedKernel {
  Model model;
  ConstArray4 u, a, active_cells, inverse_volume_fraction;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (active_cells(i, j, 0) < Real(0.5))
      return;
    const auto state = load_state<Model>(u, i, j);
    const Aux aux = load_aux<aux_comps<Model>()>(a, i, j);
    const Real wx = model.stability_speed(state, aux, 0);
    const Real wy = model.stability_speed(state, aux, 1);
    if (!is_nonnegative_finite(wx) || !is_nonnegative_finite(wy)) {
      acc = std::numeric_limits<Real>::infinity();
      return;
    }
    const Real wave = (wx > wy ? wx : wy) * inverse_volume_fraction(i, j, 0);
    accumulate_nonnegative_finite(wave, acc);
  }
};

/// SourceFrequencyKernel: max over cells of model.source_frequency (mu >= 0, 1/s).
template <class Model>
struct SourceFrequencyKernel {
  Model model;
  ConstArray4 u, a;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    const Real mu = model.source_frequency(s, ax);
    accumulate_nonnegative_finite(mu, acc);
  }
};

/// Exact local reduction state for the direct admissible-step contract.  ``minimum`` stays at
/// +infinity when no active cell constrains the step; ``invalid`` is independent so NaN, zero and
/// negative values can never be hidden by a valid minimum from another cell.
struct StabilityDtReduction {
  Real minimum;
  int constrained;
  int invalid;
};

POPS_HD inline void accumulate_stability_dt(Real value, StabilityDtReduction& reduction) {
  if (value == std::numeric_limits<Real>::infinity())
    return;  // the sole public spelling of "no direct bound on this cell"
  if (!(value > Real(0)) || !Kokkos::isfinite(value)) {
    reduction.invalid = 1;
    return;
  }
  reduction.constrained = 1;
  if (value < reduction.minimum)
    reduction.minimum = value;
}

/// Direct minimum of model.stability_dt.  A custom reduction carries the validity bit and the
/// minimum together, so the model is evaluated exactly once per active cell and tiny finite steps
/// never overflow through a reciprocal encoding.
template <class Model>
struct StabilityDtKernel {
  using value_type = StabilityDtReduction;

  Model model;
  ConstArray4 u, a;

  POPS_HD void init(value_type& value) const {
    value = {std::numeric_limits<Real>::infinity(), 0, 0};
  }
  POPS_HD void join(value_type& destination, const value_type& source) const {
    if (source.minimum < destination.minimum)
      destination.minimum = source.minimum;
    destination.constrained = destination.constrained || source.constrained;
    destination.invalid = destination.invalid || source.invalid;
  }
  POPS_HD void operator()(int i, int j, value_type& reduction) const {
    const auto s = load_state<Model>(u, i, j);
    const Aux ax = load_aux<aux_comps<Model>()>(a, i, j);
    accumulate_stability_dt(model.stability_dt(s, ax), reduction);
  }
};

template <class Model>
struct ActiveStabilityDtKernel : StabilityDtKernel<Model> {
  ConstArray4 active_cells;

  using value_type = StabilityDtReduction;
  POPS_HD void operator()(int i, int j, value_type& reduction) const {
    if (active_cells(i, j, 0) >= Real(0.5))
      StabilityDtKernel<Model>::operator()(i, j, reduction);
  }
};

template <class Kernel>
inline StabilityDtReduction reduce_stability_dt_cell(const Box2D& box, const Kernel& kernel) {
  ensure_kokkos_initialized();
  StabilityDtReduction result{};
  Kokkos::parallel_reduce(
      "pops_reduce_stability_dt_cell",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::IndexType<int>>(
          {box.lo[0], box.lo[1]}, {box.hi[0] + 1, box.hi[1] + 1}),
      kernel, result);
  return result;
}

inline void merge_stability_dt_reduction(StabilityDtReduction& destination,
                                         const StabilityDtReduction& source) {
  if (source.minimum < destination.minimum)
    destination.minimum = source.minimum;
  destination.constrained = destination.constrained || source.constrained;
  destination.invalid = destination.invalid || source.invalid;
}

/// One native collective publishes both status and the exact minimum.  ``max(-dt)`` is exactly
/// ``-min(dt)`` for positive binary64 values; -infinity is the neutral contribution of a rank with
/// no bound.  The first lane propagates any invalid active-cell result to every rank.
inline Real publish_stability_dt_minimum(const StabilityDtReduction& local) {
  double payload[2] = {
      local.invalid ? 1.0 : 0.0,
      local.constrained ? -static_cast<double>(local.minimum)
                        : -std::numeric_limits<double>::infinity(),
  };
  all_reduce_max_inplace(payload, 2);
  if (payload[0] != 0.0)
    throw std::domain_error(
        "stability_dt returned zero, a negative value, or a non-finite value other than +inf "
        "on an active cell");
  return payload[1] == -std::numeric_limits<double>::infinity()
             ? Real(0)
             : static_cast<Real>(-payload[1]);
}
}  // namespace detail

/// Global max of the STABILITY speed (HasStabilitySpeed trait) -- counterpart of max_wave_speed_mf.
/// Rejects a negative or non-finite active-cell value collectively.
template <class Model>
inline Real max_stability_speed_mf(const Model& model, const MultiFab& U, const MultiFab& aux) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(m, reduce_max_cell(U.box(li), detail::StabilitySpeedKernel<Model>{model, u, a}));
  }
  return detail::publish_nonnegative_maximum(m, "stability_speed");
}

template <class Model>
inline Real max_stability_speed_mf(const Model& model, const MultiFab& U, const MultiFab& aux,
                                   const MultiFab& active_cells) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(
        m, reduce_max_cell(U.box(li),
                           detail::ActiveCellReductionKernel<detail::StabilitySpeedKernel<Model>>{
                               detail::StabilitySpeedKernel<Model>{model, u, a},
                               active_cells.fab(li).const_array()}));
  }
  return detail::publish_nonnegative_maximum(m, "stability_speed");
}

template <class Model>
inline Real max_stability_speed_mf(const Model& model, const MultiFab& U, const MultiFab& aux,
                                   const MultiFab& active_cells,
                                   const MultiFab& inverse_volume_fraction) {
  Real maximum = 0;
  for (int local = 0; local < U.local_size(); ++local)
    maximum = std::max(
        maximum,
        reduce_max_cell(U.box(local),
                        detail::CutCellStabilitySpeedKernel<Model>{
                            model, U.fab(local).const_array(), aux.fab(local).const_array(),
                            active_cells.fab(local).const_array(),
                            inverse_volume_fraction.fab(local).const_array()}));
  return detail::publish_nonnegative_maximum(maximum, "stability_speed");
}

/// Global max of the source frequency (HasSourceFrequency trait). 0 if the source does not constrain;
/// a negative or non-finite active-cell value is rejected collectively.
template <class Model>
inline Real max_source_frequency_mf(const Model& model, const MultiFab& U, const MultiFab& aux) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(m, reduce_max_cell(U.box(li), detail::SourceFrequencyKernel<Model>{model, u, a}));
  }
  return detail::publish_nonnegative_maximum(m, "source_frequency");
}

template <class Model>
inline Real max_source_frequency_mf(const Model& model, const MultiFab& U, const MultiFab& aux,
                                    const MultiFab& active_cells) {
  Real m = 0;
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    m = std::max(
        m, reduce_max_cell(U.box(li),
                           detail::ActiveCellReductionKernel<detail::SourceFrequencyKernel<Model>>{
                               detail::SourceFrequencyKernel<Model>{model, u, a},
                               active_cells.fab(li).const_array()}));
  }
  return detail::publish_nonnegative_maximum(m, "source_frequency");
}

/// Global min of the declared admissible step (HasStabilityDt trait).  A finite value must be
/// strictly positive; +infinity alone means "no bound on this cell".  Every other non-finite or
/// non-positive value is rejected collectively.  @return 0 if NO cell constrains.
template <class Model>
inline Real min_stability_dt_mf(const Model& model, const MultiFab& U, const MultiFab& aux) {
  detail::StabilityDtReduction local{std::numeric_limits<Real>::infinity(), 0, 0};
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    detail::merge_stability_dt_reduction(
        local, detail::reduce_stability_dt_cell(
                   U.box(li), detail::StabilityDtKernel<Model>{model, u, a}));
  }
  return detail::publish_stability_dt_minimum(local);
}

template <class Model>
inline Real min_stability_dt_mf(const Model& model, const MultiFab& U, const MultiFab& aux,
                                const MultiFab& active_cells) {
  detail::StabilityDtReduction local{std::numeric_limits<Real>::infinity(), 0, 0};
  for (int li = 0; li < U.local_size(); ++li) {
    const ConstArray4 u = U.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    detail::merge_stability_dt_reduction(
        local, detail::reduce_stability_dt_cell(
                   U.box(li), detail::ActiveStabilityDtKernel<Model>{
                                      {model, u, a}, active_cells.fab(li).const_array()}));
  }
  return detail::publish_stability_dt_minimum(local);
}

}  // namespace pops
