#pragma once

#include <pops/mesh/index/box2d.hpp>
#include <pops/amr/hierarchy/refinement_ratio.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/layout/refinement.hpp>     // coarsen_index
#include <pops/numerics/spatial_operator.hpp>  // compute_face_fluxes, xface_box, yface_box
#include <pops/numerics/spatial/primitives/finite.hpp>
#include <pops/numerics/time/amr/prepared_coarse_fine_operator.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

/// @file
/// @brief Basic MultiFab spatial building blocks: device-clean functors and helpers
///        (flux/source residual evaluation, average-down and coarse-fine interpolation).
///
/// Layer: `include/pops/numerics/time`.
/// Role: provide the spatial kernels reused by AMR runtime blocks and prepared hierarchy
///        transfers. ProgramGraph owns every state update and source-treatment composition.
///
/// Invariants:
/// - kernels = NAMED functors and not lambdas:
///   a first instantiation from an external loader TU or an extended lambda would
///   make nvcc choke;
/// - the device paths read/write unified memory: device_fence() before host read.

namespace pops {

static_assert(kAmrRefRatio == 2, "ratio-2-structural kernels below assume kAmrRefRatio == 2");

// Device-clean NAMED functor (same recipe as mf_arith.hpp: a first instantiation possible
// from an external loader TU, or an extended lambda makes nvcc choke) of the method-of-lines RHS
// at ONE AMR level: R = -div(Fx,Fy) + S(U, aux), evaluated at ONE SAME state. Program stages consume
// this residual; without a source (model with S == 0) R reduces to -div F.
template <class Model>
struct AmrResidualKernel {
  Model m;
  ConstArray4 u, ax, fx, fy;
  Array4 R;
  Real dx, dy;
  POPS_HD void operator()(int i, int j) const {
    const auto S = m.source(load_state<Model>(u, i, j), load_aux<aux_comps<Model>()>(ax, i, j));
    for (int c = 0; c < Model::n_vars; ++c)
      R(i, j, c) =
          -((fx(i + 1, j, c) - fx(i, j, c)) / dx + (fy(i, j + 1, c) - fy(i, j, c)) / dy) + S[c];
  }
};

namespace detail {

// Internal arithmetic-only form used when the owning integrator validates the post-combination
// stage instead.  Keeping it separate preserves mf_eval_rhs as a safe standalone publication
// boundary for compiled blocks and direct callers.
template <class Model>
inline void mf_eval_rhs_unchecked(const Model& m, const MultiFab& U, const MultiFab& aux,
                                  const MultiFab& Fx, const MultiFab& Fy, Real dx, Real dy,
                                  MultiFab& R) {
  for (int li = 0; li < U.local_size(); ++li)
    for_each_cell(U.box(li),
                  AmrResidualKernel<Model>{m, U.fab(li).const_array(), aux.fab(li).const_array(),
                                           Fx.fab(li).const_array(), Fy.fab(li).const_array(),
                                           R.fab(li).array(), dx, dy});
}

}  // namespace detail

// R <- -div(Fx,Fy) + S(U, aux) on the valid cells (method-of-lines RHS at ONE level, evaluated
// at the state U). The stage flux Fx/Fy is assumed already computed by compute_face_fluxes at U.
// Standalone callers publish R directly, so this form validates R and its two input flux fields
// atomically.
template <class Model>
inline void mf_eval_rhs(const Model& m, const MultiFab& U, const MultiFab& aux, const MultiFab& Fx,
                        const MultiFab& Fy, Real dx, Real dy, MultiFab& R) {
  detail::mf_eval_rhs_unchecked(m, U, aux, Fx, Fy, dx, dy, R);
  detail::reject_nonfinite_finite_volume_data("mf_eval_rhs", R, Fx, Fy);
}

/// Device-clean NAMED functor: 2x2 average fine -> coarse on a coarse cell.
struct AmrAverageDownKernel {
  ConstArray4 f;
  Array4 c;
  int nc;
  POPS_HD void operator()(int I, int J) const {
    for (int k = 0; k < nc; ++k)
      c(I, J, k) = Real(0.25) * (f(2 * I, 2 * J, k) + f(2 * I + 1, 2 * J, k) +
                                 f(2 * I, 2 * J + 1, k) + f(2 * I + 1, 2 * J + 1, k));
  }
};

// average fine -> coarse (ratio 2) on the covered region (coarse coords).
inline void mf_average_down(const MultiFab& Uf, MultiFab& Uc, int CI0, int CI1, int CJ0, int CJ1) {
  const int nc = Uc.ncomp();
  const ConstArray4 f = Uf.fab(0).const_array();
  Array4 c = Uc.fab(0).array();
  for_each_cell(Box2D{{CI0, CJ0}, {CI1, CJ1}}, AmrAverageDownKernel{f, c, nc});
}

// Fills one fine ghost from a time-interpolated parent state followed by conservative, minmod-limited
// piecewise-linear spatial reconstruction.  Performing time interpolation first gives the spatial
// provider one physical parent snapshot at the requested substep time. Ratio two makes the four
// child offsets +/-1/4, hence their average is exactly the parent mean.
POPS_HD inline Real coarse_fine_time_value(const ConstArray4& old_value,
                                           const ConstArray4& new_value, int i, int j,
                                           int component, Real fraction) {
  return (Real(1) - fraction) * old_value(i, j, component) + fraction * new_value(i, j, component);
}

/// One-dimensional weights for the average of either ratio-two child of a coarse cell.
///
/// The five inputs are coarse-cell averages on one contiguous stencil.  `parent_position`
/// identifies the parent within that stencil and therefore also covers the one-sided stencils
/// required next to a non-periodic boundary.  These dyadic coefficients are obtained by integrating
/// the unique degree-four polynomial whose five coarse-cell averages match the inputs.  For every
/// parent position, the arithmetic mean of the two child rows is exactly the Kronecker row of the
/// parent; the tensor product below consequently conserves every coarse-cell mean, not merely the
/// global integral.
POPS_HD inline Real conservative_polynomial5_row_weight(int stencil_index, int a, int b, int c,
                                                        int d, int e) {
  switch (stencil_index) {
    case 0:
      return Real(a) / Real(128);
    case 1:
      return Real(b) / Real(128);
    case 2:
      return Real(c) / Real(128);
    case 3:
      return Real(d) / Real(128);
    default:
      return Real(e) / Real(128);
  }
}

POPS_HD inline Real conservative_polynomial5_weight(int parent_position, int child,
                                                    int stencil_index) {
  switch (2 * parent_position + child) {
    case 0:
      return conservative_polynomial5_row_weight(stencil_index, 193, -122, 88, -38, 7);
    case 1:
      return conservative_polynomial5_row_weight(stencil_index, 63, 122, -88, 38, -7);
    case 2:
      return conservative_polynomial5_row_weight(stencil_index, 7, 158, -52, 18, -3);
    case 3:
      return conservative_polynomial5_row_weight(stencil_index, -7, 98, 52, -18, 3);
    case 4:
      return conservative_polynomial5_row_weight(stencil_index, -3, 22, 128, -22, 3);
    case 5:
      return conservative_polynomial5_row_weight(stencil_index, 3, -22, 128, 22, -3);
    case 6:
      return conservative_polynomial5_row_weight(stencil_index, 3, -18, 52, 98, -7);
    case 7:
      return conservative_polynomial5_row_weight(stencil_index, -3, 18, -52, 158, 7);
    case 8:
      return conservative_polynomial5_row_weight(stencil_index, -7, 38, -88, 122, 63);
    default:
      return conservative_polynomial5_row_weight(stencil_index, 7, -38, 88, -122, 193);
  }
}

struct ConservativePolynomial5Stencil {
  int x_begin;
  int y_begin;
  int x_parent_position;
  int y_parent_position;
  int x_child;
  int y_child;
};

/// Resolve a full degree-four stencil without changing order near physical boundaries.  Periodic
/// axes retain the centred stencil and read periodic images from the prepared parent carrier;
/// non-periodic axes shift the same five-cell stencil into the logical domain.  The host-side
/// preparation contract rejects domains smaller than five cells, so this routine never clamps to a
/// lower-order formula.
POPS_HD inline ConservativePolynomial5Stencil conservative_polynomial5_stencil(
    int i, int j, const Box2D& coarse_domain, const Box2D& fine_domain, Periodicity periodicity) {
  const int x_fine_offset = i - fine_domain.lo[0];
  const int y_fine_offset = j - fine_domain.lo[1];
  const int x_coarse_offset = floor_div(x_fine_offset, kAmrRefRatio);
  const int y_coarse_offset = floor_div(y_fine_offset, kAmrRefRatio);
  const int ci = coarse_domain.lo[0] + x_coarse_offset;
  const int cj = coarse_domain.lo[1] + y_coarse_offset;
  int x_begin = ci - 2;
  int y_begin = cj - 2;
  if (!periodicity.x) {
    if (x_begin < coarse_domain.lo[0])
      x_begin = coarse_domain.lo[0];
    if (x_begin + 4 > coarse_domain.hi[0])
      x_begin = coarse_domain.hi[0] - 4;
  }
  if (!periodicity.y) {
    if (y_begin < coarse_domain.lo[1])
      y_begin = coarse_domain.lo[1];
    if (y_begin + 4 > coarse_domain.hi[1])
      y_begin = coarse_domain.hi[1] - 4;
  }
  return ConservativePolynomial5Stencil{x_begin,
                                        y_begin,
                                        ci - x_begin,
                                        cj - y_begin,
                                        x_fine_offset - kAmrRefRatio * x_coarse_offset,
                                        y_fine_offset - kAmrRefRatio * y_coarse_offset};
}

POPS_HD inline Real conservative_polynomial5_value(const ConstArray4& value, int i, int j,
                                                   int component, const Box2D& coarse_domain,
                                                   const Box2D& fine_domain,
                                                   Periodicity periodicity) {
  const ConservativePolynomial5Stencil stencil =
      conservative_polynomial5_stencil(i, j, coarse_domain, fine_domain, periodicity);
  Real result = Real(0);
  for (int sy = 0; sy < 5; ++sy) {
    const Real wy = conservative_polynomial5_weight(stencil.y_parent_position, stencil.y_child, sy);
    Real x_value = Real(0);
    for (int sx = 0; sx < 5; ++sx)
      x_value += conservative_polynomial5_weight(stencil.x_parent_position, stencil.x_child, sx) *
                 value(stencil.x_begin + sx, stencil.y_begin + sy, component);
    result += wy * x_value;
  }
  return result;
}

POPS_HD inline Real conservative_polynomial5_time_value(
    const ConstArray4& old_value, const ConstArray4& new_value, int i, int j, int component,
    Real fraction, const Box2D& coarse_domain, const Box2D& fine_domain, Periodicity periodicity) {
  const ConservativePolynomial5Stencil stencil =
      conservative_polynomial5_stencil(i, j, coarse_domain, fine_domain, periodicity);
  Real result = Real(0);
  for (int sy = 0; sy < 5; ++sy) {
    const Real wy = conservative_polynomial5_weight(stencil.y_parent_position, stencil.y_child, sy);
    Real x_value = Real(0);
    for (int sx = 0; sx < 5; ++sx) {
      const int ci = stencil.x_begin + sx;
      const int cj = stencil.y_begin + sy;
      x_value += conservative_polynomial5_weight(stencil.x_parent_position, stencil.x_child, sx) *
                 coarse_fine_time_value(old_value, new_value, ci, cj, component, fraction);
    }
    result += wy * x_value;
  }
  return result;
}

struct ConservativeLinearCellFillKernel {
  Array4 fine;
  ConstArray4 coarse;
  Box2D valid;
  Box2D coarse_domain;
  Box2D fine_domain;
  int component;
  bool fill_valid;
  bool fill_ghost;
  Periodicity periodicity;

  POPS_HD void operator()(int i, int j) const {
    const bool is_valid = valid.contains(i, j);
    if ((is_valid && !fill_valid) || (!is_valid && !fill_ghost))
      return;
    const int ci = coarse_domain.lo[0] + floor_div(i - fine_domain.lo[0], kAmrRefRatio);
    const int cj = coarse_domain.lo[1] + floor_div(j - fine_domain.lo[1], kAmrRefRatio);
    if ((!periodicity.x && (ci < coarse_domain.lo[0] || ci > coarse_domain.hi[0])) ||
        (!periodicity.y && (cj < coarse_domain.lo[1] || cj > coarse_domain.hi[1])))
      return;
    const Real center = coarse(ci, cj, component);
    Real sx = Real(0), sy = Real(0);
    if (periodicity.x || (ci > coarse_domain.lo[0] && ci < coarse_domain.hi[0])) {
      const Real left = center - coarse(ci - 1, cj, component);
      const Real right = coarse(ci + 1, cj, component) - center;
      if (left * right > Real(0))
        sx = ((left < Real(0) ? -left : left) < (right < Real(0) ? -right : right)) ? left : right;
    }
    if (periodicity.y || (cj > coarse_domain.lo[1] && cj < coarse_domain.hi[1])) {
      const Real down = center - coarse(ci, cj - 1, component);
      const Real up = coarse(ci, cj + 1, component) - center;
      if (down * up > Real(0))
        sy = ((down < Real(0) ? -down : down) < (up < Real(0) ? -up : up)) ? down : up;
    }
    const Real ox = ((i - fine_domain.lo[0]) & 1) ? Real(0.25) : Real(-0.25);
    const Real oy = ((j - fine_domain.lo[1]) & 1) ? Real(0.25) : Real(-0.25);
    fine(i, j, component) = center + ox * sx + oy * sy;
  }
};

struct ConservativePolynomial5CellFillKernel {
  Array4 fine;
  ConstArray4 coarse;
  Box2D valid;
  Box2D coarse_domain;
  Box2D fine_domain;
  int component;
  bool fill_valid;
  bool fill_ghost;
  Periodicity periodicity;

  POPS_HD void operator()(int i, int j) const {
    const bool is_valid = valid.contains(i, j);
    if ((is_valid && !fill_valid) || (!is_valid && !fill_ghost))
      return;
    const int ci = coarse_domain.lo[0] + floor_div(i - fine_domain.lo[0], kAmrRefRatio);
    const int cj = coarse_domain.lo[1] + floor_div(j - fine_domain.lo[1], kAmrRefRatio);
    if ((!periodicity.x && (ci < coarse_domain.lo[0] || ci > coarse_domain.hi[0])) ||
        (!periodicity.y && (cj < coarse_domain.lo[1] || cj > coarse_domain.hi[1])))
      return;
    fine(i, j, component) = conservative_polynomial5_value(coarse, i, j, component, coarse_domain,
                                                           fine_domain, periodicity);
  }
};

POPS_HD inline void fill_cf_ghost_cell(Array4 f, const ConstArray4& co, const ConstArray4& cn,
                                       int i, int j, int nc, Real frac, const Box2D& coarse_domain,
                                       const Box2D& fine_domain, Periodicity periodicity = {},
                                       Real pos_floor = Real(0), int pos_comp = 0) {
  const int ci = coarse_domain.lo[0] + floor_div(i - fine_domain.lo[0], kAmrRefRatio);
  const int cj = coarse_domain.lo[1] + floor_div(j - fine_domain.lo[1], kAmrRefRatio);
  if ((!periodicity.x && (ci < coarse_domain.lo[0] || ci > coarse_domain.hi[0])) ||
      (!periodicity.y && (cj < coarse_domain.lo[1] || cj > coarse_domain.hi[1])))
    return;
  const std::int64_t local_fine_x =
      static_cast<std::int64_t>(i) - static_cast<std::int64_t>(fine_domain.lo[0]);
  const std::int64_t local_fine_y =
      static_cast<std::int64_t>(j) - static_cast<std::int64_t>(fine_domain.lo[1]);
  const Real ox = (local_fine_x & std::int64_t{1}) ? Real(0.25) : Real(-0.25);
  const Real oy = (local_fine_y & std::int64_t{1}) ? Real(0.25) : Real(-0.25);
  for (int k = 0; k < nc; ++k) {
    const Real center = coarse_fine_time_value(co, cn, ci, cj, k, frac);
    Real sx = Real(0), sy = Real(0);
    if (periodicity.x || (ci > coarse_domain.lo[0] && ci < coarse_domain.hi[0])) {
      const Real left = center - coarse_fine_time_value(co, cn, ci - 1, cj, k, frac);
      const Real right = coarse_fine_time_value(co, cn, ci + 1, cj, k, frac) - center;
      if (left * right > Real(0))
        sx = (left < Real(0) ? -left : left) < (right < Real(0) ? -right : right) ? left : right;
    }
    if (periodicity.y || (cj > coarse_domain.lo[1] && cj < coarse_domain.hi[1])) {
      const Real down = center - coarse_fine_time_value(co, cn, ci, cj - 1, k, frac);
      const Real up = coarse_fine_time_value(co, cn, ci, cj + 1, k, frac) - center;
      if (down * up > Real(0))
        sy = (down < Real(0) ? -down : down) < (up < Real(0) ? -up : up) ? down : up;
    }
    f(i, j, k) = center + ox * sx + oy * sy;
  }
  // Zhang-Shu positivity floor on the C/F fine GHOST MEAN (ADC-259): clamp the Density role only
  // (pos_comp, resolved on the host by the caller via positivity_comp<Model>) to >= pos_floor. The
  // refined-patch C/F interface is the highest-risk site: reconstruct_pp's order-1 fallback brings a
  // sub-floor face back to its SOURCE-CELL mean, and at a fine cell bordering the interface that
  // source is a ghost; without this clamp the fallback target itself could be sub-floor (the coarse
  // mean is not floored), defeating the guarantee. Momenta/energy stay interpolated -> the ghost
  // velocity m/rho only DROPS at quasi-vacuum (bounded, mirror of the single-block mean fallback).
  // pos_floor <= 0 short-circuits (bit-identical). Ghost cells are never averaged-down nor summed in
  // mass, so the clamp is conservation-safe (cf. ADC-259 design: average-down immunity + the reflux
  // coarse-side register reads a separate fab, so the two-sided telescoping is preserved exactly).
  if (pos_floor > Real(0) && f(i, j, pos_comp) < pos_floor)
    f(i, j, pos_comp) = pos_floor;
}

struct CoarseFineTemporalGhostKernel {
  Array4 fine;
  ConstArray4 old_parent;
  ConstArray4 new_parent;
  Box2D valid;
  Box2D coarse_domain;
  Box2D fine_domain;
  int components;
  Real fraction;
  Real positivity_floor;
  int positivity_component;
  Periodicity periodicity;

  POPS_HD void operator()(int i, int j) const {
    if (i >= valid.lo[0] && i <= valid.hi[0] && j >= valid.lo[1] && j <= valid.hi[1])
      return;
    fill_cf_ghost_cell(fine, old_parent, new_parent, i, j, components, fraction, coarse_domain,
                       fine_domain, periodicity, positivity_floor, positivity_component);
  }
};

struct ConservativePolynomial5TemporalGhostKernel {
  Array4 fine;
  ConstArray4 old_parent;
  ConstArray4 new_parent;
  Box2D valid;
  Box2D coarse_domain;
  Box2D fine_domain;
  int components;
  Real fraction;
  Real positivity_floor;
  int positivity_component;
  Periodicity periodicity;

  POPS_HD void operator()(int i, int j) const {
    if (i >= valid.lo[0] && i <= valid.hi[0] && j >= valid.lo[1] && j <= valid.hi[1])
      return;
    const int ci = coarse_domain.lo[0] + floor_div(i - fine_domain.lo[0], kAmrRefRatio);
    const int cj = coarse_domain.lo[1] + floor_div(j - fine_domain.lo[1], kAmrRefRatio);
    if ((!periodicity.x && (ci < coarse_domain.lo[0] || ci > coarse_domain.hi[0])) ||
        (!periodicity.y && (cj < coarse_domain.lo[1] || cj > coarse_domain.hi[1])))
      return;
    for (int component = 0; component < components; ++component)
      fine(i, j, component) =
          conservative_polynomial5_time_value(old_parent, new_parent, i, j, component, fraction,
                                              coarse_domain, fine_domain, periodicity);
    if (positivity_floor > Real(0) && fine(i, j, positivity_component) < positivity_floor)
      fine(i, j, positivity_component) = positivity_floor;
  }
};

inline void validate_builtin_ratio2_coarse_fine_transform(
    const PreparedCoarseFineTransform2D& transform, const Box2D& coarse_domain,
    const Box2D& fine_domain) {
  if (transform.refinement_ratio_x != 2 || transform.refinement_ratio_y != 2 ||
      transform.coarse_origin_x != coarse_domain.lo[0] ||
      transform.coarse_origin_y != coarse_domain.lo[1] ||
      transform.fine_origin_x != fine_domain.lo[0] ||
      transform.fine_origin_y != fine_domain.lo[1] || fine_domain != coarse_domain.refine(2))
    throw std::invalid_argument(
        "builtin conservative coarse/fine operator requires an explicit ratio-2 2D transform");
}

/// Builtin order-two authority.  This factory is also the explicit compatibility route used by
/// low-level callers that do not own a transfer registry; production AmrRuntime receives the
/// registry-selected authority instead.
inline PreparedCoarseFineOperator prepare_limited_linear_coarse_fine_operator() {
  PreparedCoarseFineOperator prepared;
  prepared.parent_reach_x = 1;
  prepared.parent_reach_y = 1;
  prepared.minimum_axis_cells_x = 1;
  prepared.minimum_axis_cells_y = 1;
  prepared.launch_spatial = [](Array4 fine, ConstArray4 coarse, const Box2D& target,
                               const Box2D& valid, const Box2D& coarse_domain,
                               const Box2D& fine_domain,
                               const PreparedCoarseFineTransform2D& transform, int component,
                               bool fill_valid, bool fill_ghost, Periodicity periodicity) {
    validate_builtin_ratio2_coarse_fine_transform(transform, coarse_domain, fine_domain);
    for_each_cell(target,
                  ConservativeLinearCellFillKernel{fine, coarse, valid, coarse_domain, fine_domain,
                                                   component, fill_valid, fill_ghost, periodicity});
  };
  prepared.launch_space_time =
      [](Array4 fine, ConstArray4 old_parent, ConstArray4 new_parent, const Box2D& target,
         const Box2D& valid, const Box2D& coarse_domain, const Box2D& fine_domain,
         const PreparedCoarseFineTransform2D& transform, int components, Real fraction,
         Real positivity_floor, int positivity_component, Periodicity periodicity) {
        validate_builtin_ratio2_coarse_fine_transform(transform, coarse_domain, fine_domain);
        for_each_cell(target, CoarseFineTemporalGhostKernel{fine, old_parent, new_parent, valid,
                                                            coarse_domain, fine_domain, components,
                                                            fraction, positivity_floor,
                                                            positivity_component, periodicity});
      };
  return prepared;
}

inline PreparedCoarseFineOperator prepare_polynomial5_coarse_fine_operator() {
  PreparedCoarseFineOperator prepared;
  prepared.parent_reach_x = 4;
  prepared.parent_reach_y = 4;
  prepared.minimum_axis_cells_x = 5;
  prepared.minimum_axis_cells_y = 5;
  prepared.launch_spatial = [](Array4 fine, ConstArray4 coarse, const Box2D& target,
                               const Box2D& valid, const Box2D& coarse_domain,
                               const Box2D& fine_domain,
                               const PreparedCoarseFineTransform2D& transform, int component,
                               bool fill_valid, bool fill_ghost, Periodicity periodicity) {
    validate_builtin_ratio2_coarse_fine_transform(transform, coarse_domain, fine_domain);
    for_each_cell(target, ConservativePolynomial5CellFillKernel{fine, coarse, valid, coarse_domain,
                                                                fine_domain, component, fill_valid,
                                                                fill_ghost, periodicity});
  };
  prepared.launch_space_time =
      [](Array4 fine, ConstArray4 old_parent, ConstArray4 new_parent, const Box2D& target,
         const Box2D& valid, const Box2D& coarse_domain, const Box2D& fine_domain,
         const PreparedCoarseFineTransform2D& transform, int components, Real fraction,
         Real positivity_floor, int positivity_component, Periodicity periodicity) {
        validate_builtin_ratio2_coarse_fine_transform(transform, coarse_domain, fine_domain);
        for_each_cell(
            target, ConservativePolynomial5TemporalGhostKernel{
                        fine, old_parent, new_parent, valid, coarse_domain, fine_domain, components,
                        fraction, positivity_floor, positivity_component, periodicity});
      };
  return prepared;
}

}  // namespace pops
