/// @file
/// @brief MultiFab arithmetic (saxpy, lincomb, norm_inf, dot) over VALID cells.
///
/// Building blocks for integrator stages and Krylov solvers. Assumes IDENTICAL layouts
/// (same BoxArray, same DistributionMapping). Pointwise operations -> ALIASING is safe
/// (x or y == z allowed). norm_inf / dot go through the reducer seam (true Kokkos reduction).
/// dot performs a COLLECTIVE all_reduce: it MUST be called on EVERY rank (including a rank with no
/// box) under MPI, otherwise deadlock. FP NOTE: dot/sum are re-associated per tile (Kokkos::Sum,
/// deterministic/idempotent but not bit-identical to a lexicographic sum, for all Kokkos
/// spaces); norm_inf is exact everywhere. The kernels are device-clean NAMED FUNCTORS (nvcc cross-TU).

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>  // all_reduce_sum: COLLECTIVE dot product (Krylov under MPI)

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace pops {

/// Prepared relative measure for reductions over a physical cell domain.
///
/// `active_cells == nullptr` denotes the full valid-cell domain.  Otherwise only cells whose mask
/// is at least 0.5 participate.  `inverse_volume_fraction` is optional: when present, an active
/// cell contributes with relative volume `1 / inverse_volume_fraction`; when absent every selected
/// cell has unit relative volume.  This one data contract covers a full grid, a staircase mask and
/// cut-cell metrics without exposing a geometry/shape type to the reduction kernels.
struct RelativeCellMeasure {
  const MultiFab* active_cells = nullptr;
  const MultiFab* inverse_volume_fraction = nullptr;
};

namespace detail {
// NAMED FUNCTORS (not POPS_HD lambdas) for the MultiFab arithmetic kernels. Same recipe as
// the block path (#64): these operations are first-instantiated from the MG V-cycle, itself pulled
// from an external TU (native harness/loader); an extended lambda at this spot makes nvcc stumble on
// device kernel emission (null kernel-stub -> Cuda segfault in -O Release without -g, #93). Body
// strictly identical to the old lambdas -> bit-identical on CPU and device.
struct SaxpyKernel {
  Array4 Y;
  ConstArray4 X;
  Real a;
  int c;
  POPS_HD void operator()(int i, int j) const { Y(i, j, c) += a * X(i, j, c); }
};

struct ActiveSaxpyKernel {
  Array4 Y;
  ConstArray4 X, active_cells;
  Real a;
  int c;
  POPS_HD void operator()(int i, int j) const {
    if (active_cells(i, j, 0) >= Real(0.5))
      Y(i, j, c) += a * X(i, j, c);
  }
};

struct ScaleKernel {
  Array4 values;
  Real factor;
  int comp;
  POPS_HD void operator()(int i, int j) const { values(i, j, comp) *= factor; }
};

struct LincombKernel {
  Array4 Z;
  ConstArray4 X, Y;
  Real a, b;
  int c;
  POPS_HD void operator()(int i, int j) const { Z(i, j, c) = a * X(i, j, c) + b * Y(i, j, c); }
};

struct ActiveLincombKernel {
  Array4 Z;
  ConstArray4 X, Y, active_cells;
  Real a, b;
  int c;
  POPS_HD void operator()(int i, int j) const {
    if (active_cells(i, j, 0) >= Real(0.5))
      Z(i, j, c) = a * X(i, j, c) + b * Y(i, j, c);
  }
};

// Reducer |f(i,j,comp)| -> max, passed DIRECTLY to reduce_max_cell (no wrapping extended
// lambda, unlike for_each_cell_reduce_max). This is the device-clean path documented
// in for_each.hpp. Reducer signature (i, j, Real& acc); same Kokkos::Max / same sequential host
// loop -> bit-identical to the old norm_inf for finite inputs (max and fabs without rounding).
// NaN and either infinity are mapped to +infinity before the max reduction.  This is deliberate:
// IEEE max reductions may otherwise ignore NaN depending on operand order, while the +infinity
// sentinel propagates deterministically through both Kokkos::Max and the later MPI_MAX.
struct NormInfKernel {
  ConstArray4 a;
  int comp;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    const Real v = a(i, j, comp);
    const Real av = v < 0 ? -v : v;
    if (!(av <= std::numeric_limits<Real>::max())) {
      acc = std::numeric_limits<Real>::infinity();
      return;
    }
    if (av > acc)
      acc = av;
  }
};

// Reducer x(i,j,comp) * y(i,j,comp) -> sum, passed DIRECTLY to reduce_sum_cell (no wrapping
// extended lambda). Device-clean NAMED functor (same recipe as NormInfKernel) for the Krylov
// solver dot product, pulled from an external TU. Reducer signature (i, j, Real& acc).
struct DotKernel {
  ConstArray4 x, y;
  int comp;
  POPS_HD void operator()(int i, int j, Real& acc) const { acc += x(i, j, comp) * y(i, j, comp); }
};

struct DifferenceSqKernel {
  ConstArray4 current, previous;
  int comp;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    const Real difference = current(i, j, comp) - previous(i, j, comp);
    acc += difference * difference;
  }
};

// Reducer f(i,j,comp) -> sum / signed max / signed min over one component. Same device-clean named
// functor recipe as DotKernel / NormInfKernel (the compiled time Program reductions are first
// instantiated from a generated problem.so, an external TU). MaxKernel/MinKernel are SIGNED (no fabs,
// unlike NormInfKernel): they reduce the value itself, the contract of P.max / P.min.
struct SumKernel {
  ConstArray4 a;
  int comp;
  POPS_HD void operator()(int i, int j, Real& acc) const { acc += a(i, j, comp); }
};
struct MaxKernel {
  ConstArray4 a;
  int comp;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    const Real v = a(i, j, comp);
    if (v > acc)
      acc = v;
  }
};
struct MinKernel {
  ConstArray4 a;
  int comp;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    const Real v = a(i, j, comp);
    if (v < acc)
      acc = v;
  }
};

// Reducer |f(i,j,comp)| -> sum over one component -- the L1 (absolute-sum) reduction. Same
// device-clean named functor recipe as SumKernel / NormInfKernel (first instantiated from a generated
// problem.so, an external TU). The abs is a branch (v < 0 ? -v : v), NOT std::fabs, for bit-parity
// with NormInfKernel above (identical magnitude rounding, none, on every backend).
struct AbsSumKernel {
  ConstArray4 a;
  int comp;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    const Real v = a(i, j, comp);
    acc += v < 0 ? -v : v;
  }
};

struct RelativeCellSumKernel {
  ConstArray4 values, active_cells, inverse_volume_fraction;
  int comp;
  bool has_inverse_volume_fraction;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (active_cells(i, j, 0) < Real(0.5))
      return;
    const Real value = values(i, j, comp);
    acc += has_inverse_volume_fraction ? value / inverse_volume_fraction(i, j, 0) : value;
  }
};

struct RelativeCellAbsSumKernel {
  ConstArray4 values, active_cells, inverse_volume_fraction;
  int comp;
  bool has_inverse_volume_fraction;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (active_cells(i, j, 0) < Real(0.5))
      return;
    const Real value = values(i, j, comp);
    const Real magnitude = value < Real(0) ? -value : value;
    acc += has_inverse_volume_fraction
               ? magnitude / inverse_volume_fraction(i, j, 0)
               : magnitude;
  }
};

struct RelativeCellDotKernel {
  ConstArray4 left, right, active_cells, inverse_volume_fraction;
  int comp;
  bool has_inverse_volume_fraction;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (active_cells(i, j, 0) < Real(0.5))
      return;
    const Real product = left(i, j, comp) * right(i, j, comp);
    acc += has_inverse_volume_fraction
               ? product / inverse_volume_fraction(i, j, 0)
               : product;
  }
};

struct RelativeCellDifferenceSqKernel {
  ConstArray4 current, previous, active_cells, inverse_volume_fraction;
  int comp;
  bool has_inverse_volume_fraction;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (active_cells(i, j, 0) < Real(0.5))
      return;
    const Real difference = current(i, j, comp) - previous(i, j, comp);
    const Real square = difference * difference;
    acc += has_inverse_volume_fraction
               ? square / inverse_volume_fraction(i, j, 0)
               : square;
  }
};

struct RelativeCellMaxKernel {
  ConstArray4 values, active_cells;
  int comp;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (active_cells(i, j, 0) < Real(0.5))
      return;
    const Real value = values(i, j, comp);
    if (value > acc)
      acc = value;
  }
};

struct RelativeCellMinKernel {
  ConstArray4 values, active_cells;
  int comp;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (active_cells(i, j, 0) < Real(0.5))
      return;
    const Real value = values(i, j, comp);
    if (value < acc)
      acc = value;
  }
};

struct RelativeCellNormInfKernel {
  ConstArray4 values, active_cells;
  int comp;
  POPS_HD void operator()(int i, int j, Real& acc) const {
    if (active_cells(i, j, 0) < Real(0.5))
      return;
    const Real value = values(i, j, comp);
    const Real magnitude = value < Real(0) ? -value : value;
    if (!(magnitude <= std::numeric_limits<Real>::max())) {
      acc = std::numeric_limits<Real>::infinity();
      return;
    }
    if (magnitude > acc)
      acc = magnitude;
  }
};

inline void validate_relative_cell_measure(const MultiFab& field,
                                           const RelativeCellMeasure& measure,
                                           const char* operation) {
  if (measure.active_cells == nullptr) {
    if (measure.inverse_volume_fraction != nullptr)
      throw std::invalid_argument(std::string(operation) +
                                  ": an inverse volume fraction requires an active-cell mask");
    return;
  }
  const auto require_same_layout = [&](const MultiFab& metric, const char* metric_name) {
    if (metric.ncomp() != 1 || metric.box_array().boxes() != field.box_array().boxes() ||
        metric.dmap().ranks() != field.dmap().ranks() ||
        metric.local_size() != field.local_size())
      throw std::invalid_argument(std::string(operation) + ": " + metric_name +
                                  " must be a one-component metric on the field layout");
  };
  require_same_layout(*measure.active_cells, "active-cell mask");
  if (measure.inverse_volume_fraction != nullptr)
    require_same_layout(*measure.inverse_volume_fraction, "inverse volume fraction");
}
}  // namespace detail

/// y <- y + a x over ALL components of the valid cells. Identical layouts required.
inline void saxpy(MultiFab& y, Real a, const MultiFab& x) {
  const int nc = y.ncomp();
  for (int li = 0; li < y.local_size(); ++li) {
    Array4 Y = y.fab(li).array();
    const ConstArray4 X = x.fab(li).const_array();
    const Box2D b = y.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(b, detail::SaxpyKernel{Y, X, a, c});
  }
}

/// Active-domain twin of saxpy. Inactive target cells are not loaded or written, so their bit
/// pattern survives every RK stage even when a mathematically neutral floating recombination would
/// round differently.
inline void saxpy_active(MultiFab& y, Real a, const MultiFab& x, const MultiFab& active_cells) {
  const int nc = y.ncomp();
  for (int li = 0; li < y.local_size(); ++li) {
    Array4 Y = y.fab(li).array();
    const ConstArray4 X = x.fab(li).const_array();
    const ConstArray4 active = active_cells.fab(li).const_array();
    const Box2D b = y.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(b, detail::ActiveSaxpyKernel{Y, X, active, a, c});
  }
}

/// x <- factor * x over ALL components of the valid cells.
///
/// This is the device-resident scalar multiplication primitive. In particular, callers must use it
/// instead of following an asynchronous assembly kernel with raw host Array4 loops: launches remain
/// ordered in the execution space without inserting a global fence or forcing a host round-trip.
inline void scale(MultiFab& x, Real factor) {
  const int nc = x.ncomp();
  for (int li = 0; li < x.local_size(); ++li) {
    Array4 values = x.fab(li).array();
    const Box2D valid = x.box(li);
    for (int c = 0; c < nc; ++c)
      for_each_cell(valid, detail::ScaleKernel{values, factor, c});
  }
}

// Infinity norm over the valid cells of one component. Each local fab is
// reduced by for_each_cell_reduce_max over |f(i,j,comp)| (true Kokkos reduction,
// Kokkos::Max), aggregated by host max over the fabs.
//
// No more device_fence() up front: under Kokkos parallel_reduce is blocking and
// absorbs the barrier. EXACT for finite data: max and fabs are without rounding and max
// is associative/commutative in IEEE754, so bit-identical to the old norm_inf regardless of
// backend (the reduction order changes no bit). Any non-finite sample returns +infinity instead
// of allowing NaN to be silently ignored by a max reduction.
/// Infinity norm max |f(.,.,comp)| over the valid cells (LOCAL, without MPI all_reduce). Exact
/// on finite data; returns +infinity if any selected value is NaN or infinite.
inline Real norm_inf(const MultiFab& mf, int comp = 0) {
  Real m = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 a = mf.fab(li).const_array();
    m = std::max(m, reduce_max_cell(mf.box(li), detail::NormInfKernel{a, comp}));
  }
  return m;  // MPI all-reduce max later (iso-behavior, not added here)
}

/// z <- a x + b y over ALL components of the valid cells. Identical layouts; aliasing safe.
inline void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y) {
  const int nc = z.ncomp();
  for (int li = 0; li < z.local_size(); ++li) {
    Array4 Z = z.fab(li).array();
    const ConstArray4 X = x.fab(li).const_array();
    const ConstArray4 Y = y.fab(li).const_array();
    const Box2D bb = z.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(bb, detail::LincombKernel{Z, X, Y, a, b, c});
  }
}

/// Active-domain twin of lincomb. Inactive target cells remain exactly untouched.
inline void lincomb_active(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y,
                           const MultiFab& active_cells) {
  const int nc = z.ncomp();
  for (int li = 0; li < z.local_size(); ++li) {
    Array4 Z = z.fab(li).array();
    const ConstArray4 X = x.fab(li).const_array();
    const ConstArray4 Y = y.fab(li).const_array();
    const ConstArray4 active = active_cells.fab(li).const_array();
    const Box2D bb = z.fab(li).box();
    for (int c = 0; c < nc; ++c)
      for_each_cell(bb, detail::ActiveLincombKernel{Z, X, Y, active, a, b, c});
  }
}

// Dot product sum_cells x . y over the VALID cells of component comp, reduced over all
// ranks (all-reduce). Building block of Krylov solvers (BiCGStab: rho, alpha, omega, betas). Each
// local fab is reduced by reduce_sum_cell (true Kokkos reduction, Kokkos::Sum), the local fabs
// aggregated by host sum, then all_reduce_sum aggregates the ranks.
//
// COLLECTIVE, MANDATORY UNDER MPI: all_reduce_sum is called on EVERY rank, including a rank
// WITH NO box (local_size()==0, which then contributes 0 to the local sum). Without this call on all
// ranks, MPI_Allreduce deadlocks (desynchronized collective); the Krylov solver must therefore
// NEVER short-circuit dot() on an empty rank. In serial all_reduce_sum is the identity.
//
// FP NOTE (like sum()): Kokkos::Sum re-associates the sum per tile, so dot is not bit-identical
// to a lexicographic sum (deterministic/idempotent nonetheless, all Kokkos spaces). Under MPI, the all-reduce
// returns the SAME value to all ranks (MPI_SUM over one same set of local contributions), so the
// Krylov stopping criterion triggers at the SAME iteration everywhere (no desynchronization).
/// Dot product Sum_cells x.y over component comp, reduced over ALL ranks (all_reduce).
/// COLLECTIVE, MANDATORY UNDER MPI: must be called on every rank (including empty), otherwise
/// deadlock. FP NOTE: not bit-identical across backends under Kokkos; the all-reduce returns the same
/// value to all ranks (no desynchronization of the Krylov stopping criterion).
/// Rank-local component dot. This is the non-collective building block for algorithms that batch
/// several products into one explicit vector all-reduce. Every caller must still participate in
/// that collective, including ranks owning no box.
inline Real dot_local(const MultiFab& x, const MultiFab& y, int comp = 0) {
  Real s = 0;
  for (int li = 0; li < x.local_size(); ++li) {
    const ConstArray4 X = x.fab(li).const_array();
    const ConstArray4 Y = y.fab(li).const_array();
    s += reduce_sum_cell(x.box(li), detail::DotKernel{X, Y, comp});
  }
  return s;
}

inline Real dot(const MultiFab& x, const MultiFab& y, int comp = 0) {
  return static_cast<Real>(all_reduce_sum(static_cast<double>(dot_local(x, y, comp))));
}

/// FULL-component dot Sum_{cells, c} x(.,.,c) * y(.,.,c) over ALL components, reduced over ALL ranks
/// (all_reduce). The vector inner product for a MULTI-component (vector / state-valued) Krylov solve:
/// the residual / search-direction norms must cover EVERY component, not just component 0, or the loop
/// converges on component 0 alone and leaves the others unsolved. For a single-component field this is
/// exactly dot(x, y) (one component, component 0), so the scalar Krylov path stays BIT-IDENTICAL.
///
/// COLLECTIVE, MANDATORY UNDER MPI: like dot, all_reduce_sum runs on every rank (an empty rank
/// contributes 0); the per-component local sums are summed BEFORE the single all-reduce so the
/// reduction structure matches dot per component (same per-tile Kokkos::Sum, deterministic).
/// Rank-local full-component dot, paired with an explicit batched collective by prepared solvers.
inline Real dot_all_local(const MultiFab& x, const MultiFab& y) {
  const int nc = x.ncomp();
  Real s = 0;
  for (int li = 0; li < x.local_size(); ++li) {
    const ConstArray4 X = x.fab(li).const_array();
    const ConstArray4 Y = y.fab(li).const_array();
    const Box2D b = x.box(li);
    for (int c = 0; c < nc; ++c)
      s += reduce_sum_cell(b, detail::DotKernel{X, Y, c});
  }
  return s;
}

inline Real dot_all(const MultiFab& x, const MultiFab& y) {
  return static_cast<Real>(all_reduce_sum(static_cast<double>(dot_all_local(x, y))));
}

/// Sum of squared component-wise changes over every valid cell.  The subtraction happens before
/// squaring, avoiding the cancellation in ||current||² + ||previous||² - 2 current.previous.
/// COLLECTIVE under MPI; no field leaves native Kokkos storage.
inline Real difference_sum_sq_all(const MultiFab& current, const MultiFab& previous) {
  if (current.ncomp() != previous.ncomp() ||
      current.box_array().boxes() != previous.box_array().boxes() ||
      current.dmap().ranks() != previous.dmap().ranks() ||
      current.local_size() != previous.local_size())
    throw std::invalid_argument(
        "difference_sum_sq_all: fields must have the same component layout");
  Real local = 0;
  for (int li = 0; li < current.local_size(); ++li) {
    const ConstArray4 now = current.fab(li).const_array();
    const ConstArray4 before = previous.fab(li).const_array();
    for (int comp = 0; comp < current.ncomp(); ++comp)
      local += reduce_sum_cell(
          current.box(li), detail::DifferenceSqKernel{now, before, comp});
  }
  return static_cast<Real>(all_reduce_sum(static_cast<double>(local)));
}

/// Sum Sum_cells f(.,.,comp) over component comp, reduced over ALL ranks (all_reduce_sum) -- the
/// compiled-Program P.sum / P.sum_component reduction. COLLECTIVE, MANDATORY UNDER MPI: called on every
/// rank (an empty rank contributes 0), like dot. Same per-tile Kokkos::Sum FP guarantees as dot.
inline Real reduce_sum(const MultiFab& mf, int comp = 0) {
  Real s = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 a = mf.fab(li).const_array();
    s += reduce_sum_cell(mf.box(li), detail::SumKernel{a, comp});
  }
  return static_cast<Real>(all_reduce_sum(static_cast<double>(s)));
}

/// Signed maximum max_cells f(.,.,comp) over component comp, reduced over ALL ranks (all_reduce_max)
/// -- the compiled-Program P.max reduction (SIGNED, not the magnitude -- use norm_inf for max|f|).
/// COLLECTIVE, MANDATORY UNDER MPI: an empty rank seeds -inf so the all_reduce_max ignores it. EXACT
/// everywhere (max without rounding, associative/commutative).
inline Real reduce_max(const MultiFab& mf, int comp = 0) {
  Real m = -std::numeric_limits<Real>::infinity();
  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 a = mf.fab(li).const_array();
    m = std::max(m, reduce_max_cell(mf.box(li), detail::MaxKernel{a, comp}));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(m)));
}

/// Signed minimum min_cells f(.,.,comp) over component comp, reduced over ALL ranks (all_reduce_min)
/// -- the compiled-Program P.min reduction. COLLECTIVE, MANDATORY UNDER MPI: an empty rank seeds +inf
/// so the all_reduce_min ignores it. EXACT everywhere (min without rounding, associative/commutative).
inline Real reduce_min(const MultiFab& mf, int comp = 0) {
  Real m = std::numeric_limits<Real>::infinity();
  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 a = mf.fab(li).const_array();
    m = std::min(m, reduce_min_cell(mf.box(li), detail::MinKernel{a, comp}));
  }
  return static_cast<Real>(all_reduce_min(static_cast<double>(m)));
}

/// Absolute sum Sum_cells |f(.,.,comp)| over component comp, reduced over ALL ranks (all_reduce_sum)
/// -- the L1 reduction (compiled-Program P.norm1 / the Norm(L1) measure). reduce_sum is SIGNED; this
/// folds magnitudes. COLLECTIVE, MANDATORY UNDER MPI: called on every rank (an empty rank contributes
/// 0), like dot. Same per-tile Kokkos::Sum FP guarantees as dot/reduce_sum (deterministic/idempotent,
/// not bit-identical to a lexicographic sum across backends). Ghost exclusion is the valid-box
/// contract: the reduction domain is mf.box(li) (the VALID box), never the grown fab box, exactly as
/// reduce_sum excludes ghosts -- no mask needed.
inline Real reduce_abs_sum(const MultiFab& mf, int comp = 0) {
  Real s = 0;
  for (int li = 0; li < mf.local_size(); ++li) {
    const ConstArray4 a = mf.fab(li).const_array();
    s += reduce_sum_cell(mf.box(li), detail::AbsSumKernel{a, comp});
  }
  return static_cast<Real>(all_reduce_sum(static_cast<double>(s)));
}

/// Measure-aware variants used by physical-domain diagnostics and compiled Programs.  The empty
/// measure delegates to the historical full-grid kernels, preserving their exact no-EB path.
inline Real reduce_sum(const MultiFab& field, int comp, const RelativeCellMeasure& measure) {
  detail::validate_relative_cell_measure(field, measure, "reduce_sum(measure)");
  if (measure.active_cells == nullptr)
    return reduce_sum(field, comp);
  Real local = 0;
  for (int li = 0; li < field.local_size(); ++li) {
    const ConstArray4 inverse = measure.inverse_volume_fraction == nullptr
                                    ? ConstArray4{}
                                    : measure.inverse_volume_fraction->fab(li).const_array();
    local += reduce_sum_cell(
        field.box(li),
        detail::RelativeCellSumKernel{field.fab(li).const_array(),
                                      measure.active_cells->fab(li).const_array(), inverse, comp,
                                      measure.inverse_volume_fraction != nullptr});
  }
  return static_cast<Real>(all_reduce_sum(static_cast<double>(local)));
}

inline Real reduce_abs_sum(const MultiFab& field, int comp,
                           const RelativeCellMeasure& measure) {
  detail::validate_relative_cell_measure(field, measure, "reduce_abs_sum(measure)");
  if (measure.active_cells == nullptr)
    return reduce_abs_sum(field, comp);
  Real local = 0;
  for (int li = 0; li < field.local_size(); ++li) {
    const ConstArray4 inverse = measure.inverse_volume_fraction == nullptr
                                    ? ConstArray4{}
                                    : measure.inverse_volume_fraction->fab(li).const_array();
    local += reduce_sum_cell(
        field.box(li),
        detail::RelativeCellAbsSumKernel{field.fab(li).const_array(),
                                         measure.active_cells->fab(li).const_array(), inverse, comp,
                                         measure.inverse_volume_fraction != nullptr});
  }
  return static_cast<Real>(all_reduce_sum(static_cast<double>(local)));
}

inline Real dot(const MultiFab& left, const MultiFab& right, int comp,
                const RelativeCellMeasure& measure) {
  detail::validate_relative_cell_measure(left, measure, "dot(measure)");
  if (left.box_array().boxes() != right.box_array().boxes() ||
      left.dmap().ranks() != right.dmap().ranks() || left.local_size() != right.local_size())
    throw std::invalid_argument("dot(measure): left and right fields must have the same layout");
  if (measure.active_cells == nullptr)
    return dot(left, right, comp);
  Real local = 0;
  for (int li = 0; li < left.local_size(); ++li) {
    const ConstArray4 inverse = measure.inverse_volume_fraction == nullptr
                                    ? ConstArray4{}
                                    : measure.inverse_volume_fraction->fab(li).const_array();
    local += reduce_sum_cell(
        left.box(li),
        detail::RelativeCellDotKernel{left.fab(li).const_array(), right.fab(li).const_array(),
                                      measure.active_cells->fab(li).const_array(), inverse, comp,
                                      measure.inverse_volume_fraction != nullptr});
  }
  return static_cast<Real>(all_reduce_sum(static_cast<double>(local)));
}

inline Real dot_all(const MultiFab& left, const MultiFab& right,
                    const RelativeCellMeasure& measure) {
  detail::validate_relative_cell_measure(left, measure, "dot_all(measure)");
  if (left.ncomp() != right.ncomp() ||
      left.box_array().boxes() != right.box_array().boxes() ||
      left.dmap().ranks() != right.dmap().ranks() || left.local_size() != right.local_size())
    throw std::invalid_argument(
        "dot_all(measure): left and right fields must have the same component layout");
  if (measure.active_cells == nullptr)
    return dot_all(left, right);
  Real local = 0;
  for (int li = 0; li < left.local_size(); ++li) {
    const ConstArray4 inverse = measure.inverse_volume_fraction == nullptr
                                    ? ConstArray4{}
                                    : measure.inverse_volume_fraction->fab(li).const_array();
    for (int comp = 0; comp < left.ncomp(); ++comp)
      local += reduce_sum_cell(
          left.box(li),
          detail::RelativeCellDotKernel{left.fab(li).const_array(), right.fab(li).const_array(),
                                        measure.active_cells->fab(li).const_array(), inverse, comp,
                                        measure.inverse_volume_fraction != nullptr});
  }
  return static_cast<Real>(all_reduce_sum(static_cast<double>(local)));
}

inline Real difference_sum_sq_all(const MultiFab& current, const MultiFab& previous,
                                  const RelativeCellMeasure& measure) {
  detail::validate_relative_cell_measure(
      current, measure, "difference_sum_sq_all(measure)");
  if (current.ncomp() != previous.ncomp() ||
      current.box_array().boxes() != previous.box_array().boxes() ||
      current.dmap().ranks() != previous.dmap().ranks() ||
      current.local_size() != previous.local_size())
    throw std::invalid_argument(
        "difference_sum_sq_all(measure): fields must have the same component layout");
  if (measure.active_cells == nullptr)
    return difference_sum_sq_all(current, previous);
  Real local = 0;
  for (int li = 0; li < current.local_size(); ++li) {
    const ConstArray4 inverse = measure.inverse_volume_fraction == nullptr
                                    ? ConstArray4{}
                                    : measure.inverse_volume_fraction->fab(li).const_array();
    for (int comp = 0; comp < current.ncomp(); ++comp)
      local += reduce_sum_cell(
          current.box(li),
          detail::RelativeCellDifferenceSqKernel{
              current.fab(li).const_array(), previous.fab(li).const_array(),
              measure.active_cells->fab(li).const_array(), inverse, comp,
              measure.inverse_volume_fraction != nullptr});
  }
  return static_cast<Real>(all_reduce_sum(static_cast<double>(local)));
}

inline Real reduce_max(const MultiFab& field, int comp, const RelativeCellMeasure& measure) {
  detail::validate_relative_cell_measure(field, measure, "reduce_max(measure)");
  if (measure.active_cells == nullptr)
    return reduce_max(field, comp);
  Real local = -std::numeric_limits<Real>::infinity();
  for (int li = 0; li < field.local_size(); ++li)
    local = std::max(
        local,
        reduce_max_cell(field.box(li),
                        detail::RelativeCellMaxKernel{field.fab(li).const_array(),
                                                      measure.active_cells->fab(li).const_array(),
                                                      comp}));
  return static_cast<Real>(all_reduce_max(static_cast<double>(local)));
}

inline Real reduce_min(const MultiFab& field, int comp, const RelativeCellMeasure& measure) {
  detail::validate_relative_cell_measure(field, measure, "reduce_min(measure)");
  if (measure.active_cells == nullptr)
    return reduce_min(field, comp);
  Real local = std::numeric_limits<Real>::infinity();
  for (int li = 0; li < field.local_size(); ++li)
    local = std::min(
        local,
        reduce_min_cell(field.box(li),
                        detail::RelativeCellMinKernel{field.fab(li).const_array(),
                                                      measure.active_cells->fab(li).const_array(),
                                                      comp}));
  return static_cast<Real>(all_reduce_min(static_cast<double>(local)));
}

inline Real reduce_norm_inf(const MultiFab& field, int comp,
                            const RelativeCellMeasure& measure) {
  detail::validate_relative_cell_measure(field, measure, "reduce_norm_inf(measure)");
  if (measure.active_cells == nullptr)
    return static_cast<Real>(all_reduce_max(static_cast<double>(norm_inf(field, comp))));
  Real local = 0;
  for (int li = 0; li < field.local_size(); ++li)
    local = std::max(
        local,
        reduce_max_cell(field.box(li),
                        detail::RelativeCellNormInfKernel{
                            field.fab(li).const_array(),
                            measure.active_cells->fab(li).const_array(), comp}));
  return static_cast<Real>(all_reduce_max(static_cast<double>(local)));
}

}  // namespace pops
