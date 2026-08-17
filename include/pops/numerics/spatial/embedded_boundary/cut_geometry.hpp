/// @file
/// @brief Compile-time-ranked level-set cut geometry shared by Cartesian EB consumers.

#pragma once

#include <pops/amr/refinement_ratio.hpp>
#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/index/real_vector.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <stdexcept>

namespace pops::nd {

/// Normalized centre-to-neighbour crossings, independent face apertures, and retained volume.
///
/// `lower` / `upper` are Shortley-Weller centre-to-neighbour distances. Dim=1/2 `volume_fraction`
/// stays the historical axis product of those rays (bit-compatible). Dim=3 reconstructs the eight
/// unit-cube corners from the same seven centre/neighbour samples and measures `{phi < 0}` by
/// marching tetrahedra on that cube, clamped to `(0, 1]` on fluid cells. `face_lower` /
/// `face_upper` are the independent `{phi < 0}` areas of each unit-cell face from those corners —
/// not the 1-D crossing product, and not a Shortley-Weller floor. Polar/Disc remain planar
/// consumers of this Cartesian product.
template <int Dim>
struct CutCellFractions {
  static_assert(Dim >= 1 && Dim <= 3);
  static constexpr int dimension = Dim;

  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  RealVector<Dim> face_lower{};
  RealVector<Dim> face_upper{};
  Real volume_fraction = Real(0);
};

/// Linear centre-to-neighbour crossing, normalized by the axis spacing.
POPS_HD inline Real cut_cell_crossing_fraction(Real center, Real neighbor,
                                               Real theta_min = kEbCutFractionFloor) {
  if (neighbor < Real(0))
    return Real(1);
  Real theta = center / (center - neighbor);
  if (theta < theta_min)
    theta = theta_min;
  if (theta > Real(1))
    theta = Real(1);
  return theta;
}

namespace cut_geometry_detail {

POPS_HD inline Real clamp_unit_interval(Real value) {
  if (value < Real(0))
    return Real(0);
  if (value > Real(1))
    return Real(1);
  return value;
}

POPS_HD inline Real clamp_cut_fraction(Real value, Real theta_min) {
  if (value < theta_min)
    return theta_min;
  if (value > Real(1))
    return Real(1);
  return value;
}

/// `{n x < alpha} ∩ [0, 1]` with `n >= 0`.
POPS_HD inline Real unit_interval_halfspace(Real n, Real alpha) {
  constexpr Real kEps = Real(1e-14);
  if (n <= kEps)
    return alpha > Real(0) ? Real(1) : Real(0);
  return clamp_unit_interval(alpha / n);
}

/// `{nx x + ny y < alpha} ∩ [0, 1]^2` with `nx, ny >= 0`.
POPS_HD inline Real unit_square_halfspace(Real nx, Real ny, Real alpha) {
  constexpr Real kEps = Real(1e-14);
  if (nx <= kEps)
    return unit_interval_halfspace(ny, alpha);
  if (ny <= kEps)
    return unit_interval_halfspace(nx, alpha);
  if (alpha <= Real(0))
    return Real(0);
  if (alpha >= nx + ny)
    return Real(1);
  Real area = alpha * alpha;
  if (alpha > nx) {
    const Real tail = alpha - nx;
    area -= tail * tail;
  }
  if (alpha > ny) {
    const Real tail = alpha - ny;
    area -= tail * tail;
  }
  if (alpha > nx + ny) {
    const Real tail = alpha - nx - ny;
    area += tail * tail;
  }
  return area / (Real(2) * nx * ny);
}

/// `{nx x + ny y + nz z < alpha} ∩ [0, 1]^3` with `nx, ny, nz >= 0`.
POPS_HD inline Real unit_cube_halfspace(Real nx, Real ny, Real nz, Real alpha) {
  constexpr Real kEps = Real(1e-14);
  const int tiny_x = nx <= kEps ? 1 : 0;
  const int tiny_y = ny <= kEps ? 1 : 0;
  const int tiny_z = nz <= kEps ? 1 : 0;
  if (tiny_x + tiny_y + tiny_z == 3)
    return alpha > Real(0) ? Real(1) : Real(0);
  if (tiny_z)
    return unit_square_halfspace(nx, ny, alpha);
  if (tiny_y)
    return unit_square_halfspace(nx, nz, alpha);
  if (tiny_x)
    return unit_square_halfspace(ny, nz, alpha);
  if (alpha <= Real(0))
    return Real(0);
  if (alpha >= nx + ny + nz)
    return Real(1);
  Real volume = alpha * alpha * alpha;
  if (alpha > nx) {
    const Real tail = alpha - nx;
    volume -= tail * tail * tail;
  }
  if (alpha > ny) {
    const Real tail = alpha - ny;
    volume -= tail * tail * tail;
  }
  if (alpha > nz) {
    const Real tail = alpha - nz;
    volume -= tail * tail * tail;
  }
  if (alpha > nx + ny) {
    const Real tail = alpha - nx - ny;
    volume += tail * tail * tail;
  }
  if (alpha > nx + nz) {
    const Real tail = alpha - nx - nz;
    volume += tail * tail * tail;
  }
  if (alpha > ny + nz) {
    const Real tail = alpha - ny - nz;
    volume += tail * tail * tail;
  }
  return volume / (Real(6) * nx * ny * nz);
}

POPS_HD inline Real abs_real(Real value) { return value < Real(0) ? -value : value; }

POPS_HD inline Real tet_volume(const Real a[3], const Real b[3], const Real c[3], const Real d[3]) {
  const Real ab0 = b[0] - a[0];
  const Real ab1 = b[1] - a[1];
  const Real ab2 = b[2] - a[2];
  const Real ac0 = c[0] - a[0];
  const Real ac1 = c[1] - a[1];
  const Real ac2 = c[2] - a[2];
  const Real ad0 = d[0] - a[0];
  const Real ad1 = d[1] - a[1];
  const Real ad2 = d[2] - a[2];
  const Real det = ab0 * (ac1 * ad2 - ac2 * ad1) - ab1 * (ac0 * ad2 - ac2 * ad0) +
                   ab2 * (ac0 * ad1 - ac1 * ad0);
  return abs_real(det) / Real(6);
}

POPS_HD inline Real triangle_area(const Real a[2], const Real b[2], const Real c[2]) {
  return Real(0.5) * abs_real((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]));
}

POPS_HD inline Real iso_edge_parameter(Real inside, Real outside) {
  const Real den = inside - outside;
  constexpr Real kEps = Real(1e-14);
  if (abs_real(den) <= kEps)
    return inside < Real(0) ? Real(1) : Real(0);
  return clamp_unit_interval(inside / den);
}

POPS_HD inline void lerp3(const Real a[3], const Real b[3], Real t, Real out[3]) {
  out[0] = a[0] + t * (b[0] - a[0]);
  out[1] = a[1] + t * (b[1] - a[1]);
  out[2] = a[2] + t * (b[2] - a[2]);
}

POPS_HD inline void lerp2(const Real a[2], const Real b[2], Real t, Real out[2]) {
  out[0] = a[0] + t * (b[0] - a[0]);
  out[1] = a[1] + t * (b[1] - a[1]);
}

POPS_HD inline Real tet_negative_volume(const Real v[4][3], const Real phi[4]) {
  int inside[4];
  int outside[4];
  int n_inside = 0;
  int n_outside = 0;
  for (int vertex = 0; vertex < 4; ++vertex) {
    if (phi[vertex] < Real(0))
      inside[n_inside++] = vertex;
    else
      outside[n_outside++] = vertex;
  }
  if (n_inside == 0)
    return Real(0);
  const Real full = tet_volume(v[0], v[1], v[2], v[3]);
  if (n_inside == 4)
    return full;
  if (n_inside == 1) {
    const int i = inside[0];
    Real q[3][3];
    for (int k = 0; k < 3; ++k)
      lerp3(v[i], v[outside[k]], iso_edge_parameter(phi[i], phi[outside[k]]), q[k]);
    return tet_volume(v[i], q[0], q[1], q[2]);
  }
  if (n_inside == 3) {
    const int o = outside[0];
    Real q[3][3];
    for (int k = 0; k < 3; ++k)
      lerp3(v[o], v[inside[k]], iso_edge_parameter(phi[o], phi[inside[k]]), q[k]);
    const Real solid = tet_volume(v[o], q[0], q[1], q[2]);
    return full > solid ? full - solid : Real(0);
  }
  const int i0 = inside[0];
  const int i1 = inside[1];
  const int o0 = outside[0];
  const int o1 = outside[1];
  Real a[3];
  Real b[3];
  Real c[3];
  Real d[3];
  lerp3(v[i0], v[o0], iso_edge_parameter(phi[i0], phi[o0]), a);
  lerp3(v[i0], v[o1], iso_edge_parameter(phi[i0], phi[o1]), b);
  lerp3(v[i1], v[o0], iso_edge_parameter(phi[i1], phi[o0]), c);
  lerp3(v[i1], v[o1], iso_edge_parameter(phi[i1], phi[o1]), d);
  return tet_volume(v[i0], v[i1], a, b) + tet_volume(v[i1], a, c, d) + tet_volume(v[i1], a, b, d);
}

POPS_HD inline Real triangle_negative_area(const Real v[3][2], const Real phi[3]) {
  int inside[3];
  int outside[3];
  int n_inside = 0;
  int n_outside = 0;
  for (int vertex = 0; vertex < 3; ++vertex) {
    if (phi[vertex] < Real(0))
      inside[n_inside++] = vertex;
    else
      outside[n_outside++] = vertex;
  }
  if (n_inside == 0)
    return Real(0);
  const Real full = triangle_area(v[0], v[1], v[2]);
  if (n_inside == 3)
    return full;
  if (n_inside == 1) {
    const int i = inside[0];
    Real q[2][2];
    for (int k = 0; k < 2; ++k)
      lerp2(v[i], v[outside[k]], iso_edge_parameter(phi[i], phi[outside[k]]), q[k]);
    return triangle_area(v[i], q[0], q[1]);
  }
  const int o = outside[0];
  Real q[2][2];
  for (int k = 0; k < 2; ++k)
    lerp2(v[o], v[inside[k]], iso_edge_parameter(phi[o], phi[inside[k]]), q[k]);
  const Real solid = triangle_area(v[o], q[0], q[1]);
  return full > solid ? full - solid : Real(0);
}

POPS_HD inline Real square_negative_area(Real p00, Real p10, Real p01, Real p11) {
  const Real v0[3][2] = {{0, 0}, {1, 0}, {1, 1}};
  const Real phi0[3] = {p00, p10, p11};
  const Real v1[3][2] = {{0, 0}, {0, 1}, {1, 1}};
  const Real phi1[3] = {p00, p01, p11};
  return triangle_negative_area(v0, phi0) + triangle_negative_area(v1, phi1);
}

POPS_HD inline void cube_corner_samples(Real center, const RealVector<3>& lower_samples,
                                        const RealVector<3>& upper_samples, Real corners[8]) {
  for (int mask = 0; mask < 8; ++mask) {
    Real face_sum = Real(0);
    for (int axis = 0; axis < 3; ++axis) {
      const Real face = (mask & (1 << axis)) != 0 ? upper_samples[axis] : lower_samples[axis];
      face_sum += face;
    }
    corners[mask] = Real(0.5) * (face_sum - center);
  }
}

POPS_HD inline Real cube_triangulation_volume(const Real corners[8]) {
  constexpr int tets[6][4] = {{0, 1, 3, 7}, {0, 1, 5, 7}, {0, 2, 3, 7},
                              {0, 2, 6, 7}, {0, 4, 5, 7}, {0, 4, 6, 7}};
  Real volume = Real(0);
  for (int tet = 0; tet < 6; ++tet) {
    Real vertices[4][3];
    Real phi[4];
    for (int local = 0; local < 4; ++local) {
      const int corner = tets[tet][local];
      vertices[local][0] = Real(corner & 1);
      vertices[local][1] = Real((corner >> 1) & 1);
      vertices[local][2] = Real((corner >> 2) & 1);
      phi[local] = corners[corner];
    }
    volume += tet_negative_volume(vertices, phi);
  }
  return clamp_unit_interval(volume);
}

POPS_HD inline Real cube_face_negative_area(const Real corners[8], int face_axis, bool upper_face) {
  const int bit = 1 << face_axis;
  const int fixed = upper_face ? bit : 0;
  Real p00 = Real(0);
  Real p10 = Real(0);
  Real p01 = Real(0);
  Real p11 = Real(0);
  for (int mask = 0; mask < 8; ++mask) {
    if ((mask & bit) != fixed)
      continue;
    int u = -1;
    int w = -1;
    for (int axis = 0; axis < 3; ++axis) {
      if (axis == face_axis)
        continue;
      if (u < 0)
        u = (mask >> axis) & 1;
      else
        w = (mask >> axis) & 1;
    }
    if (u == 0 && w == 0)
      p00 = corners[mask];
    else if (u == 1 && w == 0)
      p10 = corners[mask];
    else if (u == 0 && w == 1)
      p01 = corners[mask];
    else
      p11 = corners[mask];
  }
  return square_negative_area(p00, p10, p01, p11);
}

template <int Dim>
POPS_HD void linear_level_set_from_samples(Real center, const RealVector<Dim>& lower_samples,
                                           const RealVector<Dim>& upper_samples,
                                           RealVector<Dim>& gradient, Real& intercept) {
  intercept = center;
  for (int axis = 0; axis < Dim; ++axis) {
    gradient[axis] = Real(0.5) * (upper_samples[axis] - lower_samples[axis]);
    intercept -= gradient[axis] * Real(0.5);
  }
}

/// Measure of `{gradient · x + intercept < 0}` in the unit box, reflecting negative axes.
template <int Rank>
POPS_HD Real unit_box_negative_halfspace(const Real* gradient, Real intercept) {
  static_assert(Rank >= 1 && Rank <= 3);
  Real n[3]{};
  Real alpha = -intercept;
  for (int axis = 0; axis < Rank; ++axis) {
    const Real g = gradient[axis];
    if (g < Real(0)) {
      n[axis] = -g;
      alpha -= g;
    } else {
      n[axis] = g;
    }
  }
  if constexpr (Rank == 1)
    return unit_interval_halfspace(n[0], alpha);
  else if constexpr (Rank == 2)
    return unit_square_halfspace(n[0], n[1], alpha);
  else
    return unit_cube_halfspace(n[0], n[1], n[2], alpha);
}

template <int Dim>
POPS_HD Real reconstructed_volume_fraction(Real center, const RealVector<Dim>& lower_samples,
                                           const RealVector<Dim>& upper_samples) {
  if constexpr (Dim == 3) {
    Real corners[8];
    cube_corner_samples(center, lower_samples, upper_samples, corners);
    return cube_triangulation_volume(corners);
  } else {
    RealVector<Dim> gradient{};
    Real intercept = Real(0);
    linear_level_set_from_samples(center, lower_samples, upper_samples, gradient, intercept);
    Real g[3]{};
    for (int axis = 0; axis < Dim; ++axis)
      g[axis] = gradient[axis];
    return unit_box_negative_halfspace<Dim>(g, intercept);
  }
}

template <int Dim>
POPS_HD Real reconstructed_face_aperture(Real center, const RealVector<Dim>& lower_samples,
                                         const RealVector<Dim>& upper_samples, int face_axis,
                                         bool upper_face) {
  if constexpr (Dim == 3) {
    Real corners[8];
    cube_corner_samples(center, lower_samples, upper_samples, corners);
    return cube_face_negative_area(corners, face_axis, upper_face);
  } else {
    RealVector<Dim> gradient{};
    Real intercept = Real(0);
    linear_level_set_from_samples(center, lower_samples, upper_samples, gradient, intercept);
    intercept += gradient[face_axis] * (upper_face ? Real(1) : Real(0));
    if constexpr (Dim == 1) {
      return intercept < Real(0) ? Real(1) : Real(0);
    } else {
      Real g[2]{};
      int packed = 0;
      for (int axis = 0; axis < Dim; ++axis) {
        if (axis == face_axis)
          continue;
        g[packed] = gradient[axis];
        ++packed;
      }
      return unit_box_negative_halfspace<Dim - 1>(g, intercept);
    }
  }
}

}  // namespace cut_geometry_detail

/// Independent `{phi < 0}` area of one unit-cell face from the same plane as the ranked volume.
///
/// The Shortley-Weller `theta_min` floor applies to centre-to-neighbour distances, not to these
/// conservative face areas: a covered face is 0 and an open face is 1.
template <int Dim>
POPS_HD Real cut_cell_independent_face_aperture(Real center, const RealVector<Dim>& lower_samples,
                                                const RealVector<Dim>& upper_samples, int axis,
                                                bool upper_face,
                                                Real theta_min = kEbCutFractionFloor) {
  (void)theta_min;
  const Real area = cut_geometry_detail::reconstructed_face_aperture<Dim>(
      center, lower_samples, upper_samples, axis, upper_face);
  return cut_geometry_detail::clamp_unit_interval(area);
}

/// Materialize independent directional crossings, independent face apertures, and retained volume.
///
/// Dim=1/2 volume stays the axis product of the 1-D rays (bit-compatible). Dim=3 reconstructs the
/// eight unit-cube corners from the same seven samples and measures `{phi < 0}` by marching
/// tetrahedra, in `(0, 1]` when the centre sample is fluid. Face apertures are independent per
/// face from that same cube triangulation; they are not a second EB library. Polar/Disc stay
/// planar.
template <int Dim>
POPS_HD CutCellFractions<Dim> cut_cell_fractions_from_samples(
    Real center, const RealVector<Dim>& lower_samples, const RealVector<Dim>& upper_samples,
    Real theta_min = kEbCutFractionFloor) {
  CutCellFractions<Dim> result;
  Real product_volume = Real(1);
  for (int axis = 0; axis < Dim; ++axis) {
    result.lower[axis] = cut_cell_crossing_fraction(center, lower_samples[axis], theta_min);
    result.upper[axis] = cut_cell_crossing_fraction(center, upper_samples[axis], theta_min);
    product_volume *= Real(0.5) * (result.lower[axis] + result.upper[axis]);
    result.face_lower[axis] = cut_cell_independent_face_aperture(
        center, lower_samples, upper_samples, axis, false, theta_min);
    result.face_upper[axis] = cut_cell_independent_face_aperture(
        center, lower_samples, upper_samples, axis, true, theta_min);
  }
  if constexpr (Dim < 3) {
    result.volume_fraction = product_volume;
  } else {
    result.volume_fraction = cut_geometry_detail::reconstructed_volume_fraction<Dim>(
        center, lower_samples, upper_samples);
    if (center < Real(0))
      result.volume_fraction =
          cut_geometry_detail::clamp_cut_fraction(result.volume_fraction, theta_min);
    else
      result.volume_fraction = cut_geometry_detail::clamp_unit_interval(result.volume_fraction);
  }
  return result;
}

/// Exact-rank Shortley-Weller coefficients associated with the same sampled cut geometry.
template <int Dim>
struct ShortleyWellerStencil {
  static_assert(Dim >= 1 && Dim <= 3);
  static constexpr int dimension = Dim;

  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  Real diagonal = Real(0);
};

template <int Dim>
POPS_HD ShortleyWellerStencil<Dim> shortley_weller_stencil(const CutCellFractions<Dim>& fractions,
                                                           const RealVector<Dim>& spacing) {
  ShortleyWellerStencil<Dim> result;
  for (int axis = 0; axis < Dim; ++axis) {
    const Real lower_distance = fractions.lower[axis] * spacing[axis];
    const Real upper_distance = fractions.upper[axis] * spacing[axis];
    const Real span = lower_distance + upper_distance;
    result.lower[axis] = Real(2) / (lower_distance * span);
    result.upper[axis] = Real(2) / (upper_distance * span);
    result.diagonal += Real(2) / (lower_distance * upper_distance);
  }
  return result;
}

/// Conservative restriction of one coarse cut cell from its uniform fine children.
/// Volume is the arithmetic mean (uniform Cartesian cell volumes). Face apertures on each
/// coarse face are the mean of the fine faces that cover that coarse face. Polar/Disc stay
/// planar consumers of this same ranked metric — this is not a second EB engine.
template <int Dim>
POPS_HD CutCellFractions<Dim> restrict_cut_cell_fractions(const CutCellFractions<Dim>* fine,
                                                          int fine_count) {
  CutCellFractions<Dim> coarse{};
  if (fine == nullptr || fine_count <= 0)
    return coarse;
  const Real inv = Real(1) / static_cast<Real>(fine_count);
  for (int child = 0; child < fine_count; ++child)
    coarse.volume_fraction += fine[child].volume_fraction * inv;
  for (int axis = 0; axis < Dim; ++axis) {
    for (int child = 0; child < fine_count; ++child) {
      coarse.lower[axis] += fine[child].lower[axis] * inv;
      coarse.upper[axis] += fine[child].upper[axis] * inv;
      coarse.face_lower[axis] += fine[child].face_lower[axis] * inv;
      coarse.face_upper[axis] += fine[child].face_upper[axis] * inv;
    }
  }
  return coarse;
}

/// Conservative injection: every fine child receives the coarse volume and apertures.
template <int Dim>
POPS_HD CutCellFractions<Dim> prolong_cut_cell_fractions(const CutCellFractions<Dim>& coarse) {
  return coarse;
}

/// Outward (fluid → solid) Cartesian area vector of the cut interface from independent face
/// apertures.  This is the closed-surface remainder of the fluid region, not a cell-centred
/// Cartesian face alias.
template <int Dim>
POPS_HD RealVector<Dim> cut_cell_interface_area_vector(const CutCellFractions<Dim>& fractions) {
  RealVector<Dim> area{};
  for (int axis = 0; axis < Dim; ++axis)
    area[axis] = fractions.face_lower[axis] - fractions.face_upper[axis];
  return area;
}

template <int Dim>
POPS_HD Real cut_cell_interface_area(const CutCellFractions<Dim>& fractions) {
  const auto area = cut_cell_interface_area_vector(fractions);
  Real mag2 = Real(0);
  for (int axis = 0; axis < Dim; ++axis)
    mag2 += area[axis] * area[axis];
  return Kokkos::sqrt(mag2);
}

/// Unit outward interface normal.  Returns false when the cell has no cut (area = 0).
template <int Dim>
POPS_HD bool cut_cell_interface_normal(const CutCellFractions<Dim>& fractions,
                                       RealVector<Dim>& normal) {
  const auto area = cut_cell_interface_area_vector(fractions);
  Real mag2 = Real(0);
  for (int axis = 0; axis < Dim; ++axis)
    mag2 += area[axis] * area[axis];
  if (!(mag2 > Real(0)) || !Kokkos::isfinite(mag2))
    return false;
  const Real inv = Real(1) / Kokkos::sqrt(mag2);
  for (int axis = 0; axis < Dim; ++axis)
    normal[axis] = area[axis] * inv;
  return true;
}

/// Face reflux residual: sum of fine apertures covering one coarse face minus the coarse aperture.
POPS_HD inline Real reflux_cut_face_aperture(Real coarse_aperture, const Real* fine_apertures,
                                             int fine_count) {
  Real sum = Real(0);
  if (fine_apertures != nullptr) {
    for (int child = 0; child < fine_count; ++child)
      sum += fine_apertures[child];
  }
  return sum - coarse_aperture;
}

/// Uniform ratio-2 children fit in one ranked cube (2^3). Larger ratios stay refused here so this
/// remains one CutCellFractions engine, not a second transfer library.
inline constexpr int kMaxUniformCutCellFineChildren = 8;

template <int Dim>
POPS_HD CutCellFractions<Dim> cut_cell_fractions_from_phi_cell(FieldView<const Real, Dim> phi,
                                                               const Index<Dim>& index,
                                                               Real theta_min = kEbCutFractionFloor) {
  RealVector<Dim> lower{};
  RealVector<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    Index<Dim> lo = index;
    Index<Dim> hi = index;
    --lo[axis];
    ++hi[axis];
    lower[axis] = phi(lo);
    upper[axis] = phi(hi);
  }
  return cut_cell_fractions_from_samples<Dim>(phi(index), lower, upper, theta_min);
}

namespace cut_geometry_detail {

template <int Dim>
POPS_HD Index<Dim> fine_child_index(const Index<Dim>& coarse,
                                    const ::pops::amr::RefinementRatio<Dim>& ratio,
                                    const ::pops::amr::transfer::IndexMapping<Dim>& mapping,
                                    int child_id) {
  Index<Dim> fine{};
  int rem = child_id;
  for (int axis = 0; axis < Dim; ++axis) {
    const std::int64_t rel =
        static_cast<std::int64_t>(coarse[axis]) - mapping.coarse_origin[axis];
    fine[axis] = static_cast<int>(static_cast<std::int64_t>(mapping.fine_origin[axis]) +
                                  rel * ratio[axis] + rem % ratio[axis]);
    rem /= ratio[axis];
  }
  return fine;
}

inline void require_uniform_cut_cell_ratio(std::int64_t fine_count) {
  if (fine_count <= 1 || fine_count > kMaxUniformCutCellFineChildren)
    throw std::invalid_argument(
        "cut-cell fraction transfer supports one uniform-ratio path with at most 8 fine children");
}

template <int Dim>
FieldView<const Real, Dim> as_const_field(FieldView<Real, Dim> view) {
  FieldView<const Real, Dim> out{};
  out.data = view.data;
  out.origin = view.origin;
  out.extents = view.extents;
  for (int axis = 0; axis < Dim; ++axis)
    out.strides[axis] = view.strides[axis];
  out.ncomp = view.ncomp;
  out.component_stride = view.component_stride;
  return out;
}

template <int Dim>
struct RestrictCutCellVolumeKernel {
  FieldView<const Real, Dim> fine_phi{};
  FieldView<Real, Dim> coarse_volume{};
  ::pops::amr::RefinementRatio<Dim> ratio{};
  ::pops::amr::transfer::IndexMapping<Dim> mapping{};
  Real theta_min = kEbCutFractionFloor;
  int fine_count = 0;

  POPS_HD void operator()(const Index<Dim>& coarse) const {
    CutCellFractions<Dim> fine[kMaxUniformCutCellFineChildren];
    for (int child = 0; child < fine_count; ++child)
      fine[child] = cut_cell_fractions_from_phi_cell(fine_phi,
                                                     fine_child_index(coarse, ratio, mapping, child),
                                                     theta_min);
    coarse_volume(coarse) = restrict_cut_cell_fractions(fine, fine_count).volume_fraction;
  }
};

template <int Dim>
struct ProlongCutCellVolumeKernel {
  FieldView<const Real, Dim> coarse_volume{};
  FieldView<Real, Dim> fine_volume{};
  ::pops::amr::RefinementRatio<Dim> ratio{};
  ::pops::amr::transfer::IndexMapping<Dim> mapping{};
  int fine_count = 0;

  POPS_HD void operator()(const Index<Dim>& coarse) const {
    CutCellFractions<Dim> parent{};
    parent.volume_fraction = coarse_volume(coarse);
    const CutCellFractions<Dim> injected = prolong_cut_cell_fractions(parent);
    for (int child = 0; child < fine_count; ++child)
      fine_volume(fine_child_index(coarse, ratio, mapping, child)) = injected.volume_fraction;
  }
};

template <int Dim>
struct RefluxCutCellFaceKernel {
  FieldView<const Real, Dim> fine_phi{};
  FieldView<Real, Dim> coarse_residual{};
  ::pops::amr::RefinementRatio<Dim> ratio{};
  ::pops::amr::transfer::IndexMapping<Dim> mapping{};
  Real theta_min = kEbCutFractionFloor;
  int fine_count = 0;
  int axis = 0;
  bool upper = false;

  POPS_HD void operator()(const Index<Dim>& coarse) const {
    CutCellFractions<Dim> fine[kMaxUniformCutCellFineChildren];
    Real covering[kMaxUniformCutCellFineChildren];
    int covering_count = 0;
    for (int child = 0; child < fine_count; ++child) {
      const Index<Dim> fine_index = fine_child_index(coarse, ratio, mapping, child);
      fine[child] = cut_cell_fractions_from_phi_cell(fine_phi, fine_index, theta_min);
      int rem = child;
      int child_axis = 0;
      for (int dir = 0; dir < Dim; ++dir) {
        if (dir == axis)
          child_axis = rem % ratio[dir];
        rem /= ratio[dir];
      }
      const bool on_face = upper ? child_axis == ratio[axis] - 1 : child_axis == 0;
      if (!on_face)
        continue;
      covering[covering_count++] =
          upper ? fine[child].face_upper[axis] : fine[child].face_lower[axis];
    }
    const CutCellFractions<Dim> restricted = restrict_cut_cell_fractions(fine, fine_count);
    const Real coarse_aperture =
        upper ? restricted.face_upper[axis] : restricted.face_lower[axis];
    coarse_residual(coarse) =
        reflux_cut_face_aperture(coarse_aperture, covering, covering_count);
  }
};

}  // namespace cut_geometry_detail

/// Fine-to-coarse volume restriction on the existing CutCellFractions metric.
template <int Dim>
void apply_cut_cell_fraction_restriction(
    FieldView<const Real, Dim> fine_phi, FieldView<Real, Dim> coarse_volume,
    const Box<Dim>& coarse_region, const ::pops::amr::RefinementRatio<Dim>& ratio,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    Real theta_min = kEbCutFractionFloor) {
  const int fine_count = static_cast<int>(ratio.child_count());
  cut_geometry_detail::require_uniform_cut_cell_ratio(fine_count);
  ::pops::for_each_cell(coarse_region,
                        cut_geometry_detail::RestrictCutCellVolumeKernel<Dim>{
                            fine_phi, coarse_volume, ratio, mapping, theta_min, fine_count});
  ::pops::device_fence();
}

/// Coarse-to-fine volume injection on the existing CutCellFractions metric.
template <int Dim>
void apply_cut_cell_fraction_prolongation(
    FieldView<const Real, Dim> coarse_volume, FieldView<Real, Dim> fine_volume,
    const Box<Dim>& coarse_region, const ::pops::amr::RefinementRatio<Dim>& ratio,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {}) {
  const int fine_count = static_cast<int>(ratio.child_count());
  cut_geometry_detail::require_uniform_cut_cell_ratio(fine_count);
  ::pops::for_each_cell(coarse_region,
                        cut_geometry_detail::ProlongCutCellVolumeKernel<Dim>{
                            coarse_volume, fine_volume, ratio, mapping, fine_count});
  ::pops::device_fence();
}

/// Coarse-face aperture reflux residual on the existing CutCellFractions metric.
template <int Dim>
void apply_cut_cell_face_aperture_reflux(
    FieldView<const Real, Dim> fine_phi, FieldView<Real, Dim> coarse_residual,
    const Box<Dim>& coarse_region, int axis, bool upper,
    const ::pops::amr::RefinementRatio<Dim>& ratio,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    Real theta_min = kEbCutFractionFloor) {
  if (axis < 0 || axis >= Dim)
    throw std::invalid_argument("cut-cell face reflux axis is outside the compile-time rank");
  const int fine_count = static_cast<int>(ratio.child_count());
  cut_geometry_detail::require_uniform_cut_cell_ratio(fine_count);
  ::pops::for_each_cell(coarse_region,
                        cut_geometry_detail::RefluxCutCellFaceKernel<Dim>{
                            fine_phi, coarse_residual, ratio, mapping, theta_min, fine_count, axis,
                            upper});
  ::pops::device_fence();
}

/// One uniform-ratio coarse-fine volume/aperture path: restrict, prolong, then reflux axis 0 lower.
template <int Dim>
void apply_cut_cell_fraction_amr_transfer(
    FieldView<const Real, Dim> fine_phi, FieldView<Real, Dim> coarse_volume,
    FieldView<Real, Dim> fine_volume, FieldView<Real, Dim> coarse_aperture_residual,
    const Box<Dim>& coarse_region, const ::pops::amr::RefinementRatio<Dim>& ratio,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    Real theta_min = kEbCutFractionFloor) {
  apply_cut_cell_fraction_restriction(fine_phi, coarse_volume, coarse_region, ratio, mapping,
                                      theta_min);
  apply_cut_cell_fraction_prolongation(cut_geometry_detail::as_const_field(coarse_volume),
                                       fine_volume, coarse_region, ratio, mapping);
  apply_cut_cell_face_aperture_reflux(fine_phi, coarse_aperture_residual, coarse_region, 0, false,
                                      ratio, mapping, theta_min);
}

}  // namespace pops::nd
