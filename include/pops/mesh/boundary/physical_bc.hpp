/// @file
/// @brief PHYSICAL boundary conditions at the domain edge (BCType, BCRec, fill_physical_bc,
///        fill_ghosts).
///
/// fill_boundary already fills the INTERIOR and periodic ghosts; here we fill the ghosts that
/// fall OUTSIDE the domain on non-periodic faces. Foextrap: zero-order extrapolation
/// (zero gradient), ghost = mirror interior cell (outflow / order-0 wall). Dirichlet: value
/// imposed at the FACE, ghost = 2 v - mirror interior (the ghost/interior average equals v at the face).
/// fill_ghosts composes both in the right order (interior/periodic THEN physical edge) and
/// fills the corners via x-faces then y-faces over the full extension. The edge kernels are
/// device-clean NAMED FUNCTORS (nvcc limitation).

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace pops {

/// Boundary condition type for a face: Periodic (handled by fill_boundary), Foextrap (zero gradient,
/// outflow/order-0 wall), Dirichlet (value imposed at the face by reflection).
enum class BCType { Periodic, Foextrap, Dirichlet, Robin, External };

/// Boundary conditions for the FOUR faces of the domain (type + associated Dirichlet value). Default
/// is all periodic (xlo_val... ignored for non-Dirichlet faces).
struct BCRec {
  BCType xlo = BCType::Periodic, xhi = BCType::Periodic;
  BCType ylo = BCType::Periodic, yhi = BCType::Periodic;
  Real xlo_val = 0, xhi_val = 0, ylo_val = 0, yhi_val = 0;
  Real xlo_alpha = 0, xlo_beta = 1, xhi_alpha = 0, xhi_beta = 1;
  Real ylo_alpha = 0, ylo_beta = 1, yhi_alpha = 0, yhi_beta = 1;
  Real dx = 1, dy = 1;
};

namespace detail {
// A physical halo can be deeper than the complete domain (for example, a one-cell axis with a
// fourth-order reconstruction). A single mirror then points through the opposite boundary instead
// of into valid storage. Represent every reflection as u(out) = scale*u(mapped) + offset and compose
// reflections until mapped is valid. This gives the usual first-layer formula exactly, while making
// arbitrary-depth Dirichlet and Robin extensions independent of previously written ghost cells.
struct BoundaryFaceData {
  BCType type;
  Real value;
  Real alpha;
  Real beta;
};

struct BoundarySample1D {
  int source;
  Real scale;
  Real offset;
};

POPS_HD inline BoundarySample1D boundary_sample_1d(int index, int lo, int hi,
                                                    BoundaryFaceData low,
                                                    BoundaryFaceData high, Real h) {
  std::int64_t current = index;
  Real scale = Real(1);
  Real offset = Real(0);
  while (current < lo || current > hi) {
    const bool below = current < lo;
    const BoundaryFaceData face = below ? low : high;
    const std::int64_t boundary = below ? lo : hi;
    if (face.type == BCType::Foextrap) {
      current = boundary;
      break;
    }

    const std::int64_t layer = below ? boundary - current : current - boundary;
    const Real distance = (Real(2) * static_cast<Real>(layer) - Real(1)) * h;
    Real face_scale = Real(-1);
    Real face_offset = Real(2) * face.value;
    if (face.type == BCType::Robin) {
      const Real denominator = face.alpha / Real(2) + face.beta / distance;
      face_scale = -(face.alpha / Real(2) - face.beta / distance) / denominator;
      face_offset = face.value / denominator;
    }
    offset += scale * face_offset;
    scale *= face_scale;
    current = below ? 2 * boundary - current - 1 : 2 * boundary - current + 1;
  }
  return {static_cast<int>(current), scale, offset};
}

inline bool is_native_physical_bc(BCType type) {
  return type == BCType::Foextrap || type == BCType::Dirichlet || type == BCType::Robin;
}

inline const char* bc_type_name(BCType type) {
  switch (type) {
    case BCType::Periodic:
      return "Periodic";
    case BCType::Foextrap:
      return "Foextrap";
    case BCType::Dirichlet:
      return "Dirichlet";
    case BCType::Robin:
      return "Robin";
    case BCType::External:
      return "External";
  }
  return "unknown";
}

inline void validate_face_data(BoundaryFaceData face, Real h, const char* axis,
                               const char* side) {
  if (!is_native_physical_bc(face.type))
    return;
  if (face.type != BCType::Foextrap && !std::isfinite(face.value))
    throw std::invalid_argument(std::string("fill_physical_bc: non-finite ") + axis + side +
                                " boundary value");
  if (face.type == BCType::Robin &&
      (!std::isfinite(face.alpha) || !std::isfinite(face.beta) || !std::isfinite(h) || h <= 0))
    throw std::invalid_argument(std::string("fill_physical_bc: invalid Robin coefficients or ") +
                                axis + " spacing on the " + side + " face");
}

// Host preflight mirrors boundary_sample_1d. It rejects a deep extension which would require data
// owned by an External/Periodic opposite face, and rejects singular/non-finite Robin transforms
// before any asynchronous device kernel is launched.
inline void validate_boundary_sample_1d(std::int64_t index, int lo, int hi,
                                        BoundaryFaceData low, BoundaryFaceData high, Real h,
                                        const char* axis) {
  std::int64_t current = index;
  Real scale = Real(1);
  Real offset = Real(0);
  while (current < lo || current > hi) {
    const bool below = current < lo;
    const BoundaryFaceData face = below ? low : high;
    const std::int64_t boundary = below ? lo : hi;
    if (!is_native_physical_bc(face.type))
      throw std::invalid_argument(std::string("fill_physical_bc: deep ") + axis +
                                  " halo reaches a " + bc_type_name(face.type) +
                                  " face whose values are not owned by the physical BC fill");
    if (face.type == BCType::Foextrap)
      return;

    const std::int64_t layer = below ? boundary - current : current - boundary;
    const Real distance = (Real(2) * static_cast<Real>(layer) - Real(1)) * h;
    Real face_scale = Real(-1);
    Real face_offset = Real(2) * face.value;
    if (face.type == BCType::Robin) {
      const Real denominator = face.alpha / Real(2) + face.beta / distance;
      if (!std::isfinite(distance) || distance <= 0 || !std::isfinite(denominator) ||
          denominator == 0)
        throw std::invalid_argument(std::string("fill_physical_bc: singular Robin extension on ") +
                                    axis + (below ? "lo" : "hi") + " face");
      face_scale = -(face.alpha / Real(2) - face.beta / distance) / denominator;
      face_offset = face.value / denominator;
    }
    offset += scale * face_offset;
    scale *= face_scale;
    if (!std::isfinite(scale) || !std::isfinite(offset))
      throw std::overflow_error(std::string("fill_physical_bc: non-finite deep ") + axis +
                                " boundary extension");
    current = below ? 2 * boundary - current - 1 : 2 * boundary - current + 1;
  }
}

inline void validate_axis_extension(int lo, int hi, int ng, BoundaryFaceData low,
                                    BoundaryFaceData high, Real h, const char* axis) {
  if (lo > hi)
    throw std::invalid_argument(std::string("fill_physical_bc: empty ") + axis + " domain");
  validate_face_data(low, h, axis, "lo");
  validate_face_data(high, h, axis, "hi");
  for (int k = 1; k <= ng; ++k) {
    if (is_native_physical_bc(low.type))
      validate_boundary_sample_1d(static_cast<std::int64_t>(lo) - k, lo, hi, low, high, h, axis);
    if (is_native_physical_bc(high.type))
      validate_boundary_sample_1d(static_cast<std::int64_t>(hi) + k, lo, hi, low, high, h, axis);
  }
}

inline void validate_periodic_pairs(const BCRec& bc) {
  if ((bc.xlo == BCType::Periodic) != (bc.xhi == BCType::Periodic) ||
      (bc.ylo == BCType::Periodic) != (bc.yhi == BCType::Periodic))
    throw std::invalid_argument(
        "fill_physical_bc: periodicity must be declared on both faces of an axis");
}

// NAMED FUNCTORS (not POPS_HD lambdas) keep the boundary path device-clean under nvcc. Component
// range [c0, c1) is either the complete channel or one per-field aux component.
struct BCFaceXKernel {
  Array4 a;
  int c0, c1, lo, hi;
  BoundaryFaceData low, high;
  Real h;
  POPS_HD void operator()(int i, int j) const {
    const BoundarySample1D sample = boundary_sample_1d(i, lo, hi, low, high, h);
    for (int c = c0; c < c1; ++c)
      a(i, j, c) = sample.scale * a(sample.source, j, c) + sample.offset;
  }
};

struct BCFaceYKernel {
  Array4 a;
  int c0, c1, lo, hi;
  BoundaryFaceData low, high;
  Real h;
  POPS_HD void operator()(int i, int j) const {
    const BoundarySample1D sample = boundary_sample_1d(j, lo, hi, low, high, h);
    for (int c = c0; c < c1; ++c)
      a(i, j, c) = sample.scale * a(i, sample.source, c) + sample.offset;
  }
};
}  // namespace detail

/// Fills the OUT-OF-domain ghosts of the NON-periodic faces of @p mf according to @p bc (Foextrap or
/// Dirichlet), for the COMPONENT RANGE [c0, c1). No-op if there is no ghost or everything is periodic.
/// PRECONDITION: fill_boundary has already filled the interior/periodic (the x-faces read the y/theta
/// ghosts already filled to extend the radial BC into the halo, and the y-faces read the x ghosts for
/// the corners). CORNERS of the 9-point stencil: the x-face BC is extended to the EXTENDED j range
/// (y/theta ghosts included), so that the corner (x-physical CROSSED with y-periodic/neighbor) -- read
/// by the cross terms of a 9-point operator (e.g. PolarTensorKrylovSolver) -- is correct even in
/// MULTI-BOX. The all-component entry point fill_physical_bc(mf, domain, bc) and the single-component
/// override fill_physical_bc(mf, domain, bc, comp) (ADC-369, per-field aux halo) both delegate here.
inline void fill_physical_bc_range(MultiFab& mf, const Box2D& domain, const BCRec& bc, int c0,
                                   int c1) {
  const int ng = mf.n_grow();
  if (ng == 0)
    return;
  detail::validate_periodic_pairs(bc);
  // All periodic: fill_boundary has already done everything, nothing to read/write here (and we
  // avoid a useless barrier on the hot path of the periodic multigrid).
  if (bc.xlo == BCType::Periodic && bc.xhi == BCType::Periodic && bc.ylo == BCType::Periodic &&
      bc.yhi == BCType::Periodic)
    return;
  // A malformed range is an authoring/programming error.  Clamping used to turn an invalid
  // per-field component into a partial fill or a silent no-op, which is especially dangerous when
  // different model fields own different boundary laws.  Internal callers pass either the exact
  // full channel or one validated component, so rejecting here keeps every valid route unchanged.
  if (c0 < 0 || c1 < 0 || c0 >= c1 || c1 > mf.ncomp())
    throw std::out_of_range(
        "fill_physical_bc: component range must be non-empty and lie inside the MultiFab channel");
  const detail::BoundaryFaceData xlow{bc.xlo, bc.xlo_val, bc.xlo_alpha, bc.xlo_beta};
  const detail::BoundaryFaceData xhigh{bc.xhi, bc.xhi_val, bc.xhi_alpha, bc.xhi_beta};
  const detail::BoundaryFaceData ylow{bc.ylo, bc.ylo_val, bc.ylo_alpha, bc.ylo_beta};
  const detail::BoundaryFaceData yhigh{bc.yhi, bc.yhi_val, bc.yhi_alpha, bc.yhi_beta};
  detail::validate_axis_extension(domain.lo[0], domain.hi[0], ng, xlow, xhigh, bc.dx, "x");
  detail::validate_axis_extension(domain.lo[1], domain.hi[1], ng, ylow, yhigh, bc.dy, "y");

  // Physical edges on DEVICE. Every output reads a mapped VALID cell, never a previous halo layer.
  // The host preflight above proves that every composed reflection is finite and reaches valid
  // storage before these asynchronous kernels are launched.
  for (int li = 0; li < mf.local_size(); ++li) {
    Fab2D& F = mf.fab(li);
    const Box2D v = F.box();
    Array4 a = F.array();

    // --- x-faces ---
    // Periodic y ghosts were filled by fill_boundary and may be used to extend an x BC into a
    // nine-point corner. At a physical/external y edge they are not initialized yet, so the x pass
    // stops at the domain; the following y pass produces double-physical corners from initialized
    // x ghosts, while External corners remain owned by the caller.
    int jglo = v.lo[1] - ng;
    int jghi = v.hi[1] + ng;
    if (bc.ylo != BCType::Periodic)
      jglo = std::max(jglo, domain.lo[1]);
    if (bc.yhi != BCType::Periodic)
      jghi = std::min(jghi, domain.hi[1]);
    if (detail::is_native_physical_bc(bc.xlo) && v.lo[0] == domain.lo[0]) {
      for_each_cell(Box2D{{domain.lo[0] - ng, jglo}, {domain.lo[0] - 1, jghi}},
                    detail::BCFaceXKernel{a, c0, c1, domain.lo[0], domain.hi[0], xlow,
                                          xhigh, bc.dx});
    }
    if (detail::is_native_physical_bc(bc.xhi) && v.hi[0] == domain.hi[0]) {
      for_each_cell(Box2D{{domain.hi[0] + 1, jglo}, {domain.hi[0] + ng, jghi}},
                    detail::BCFaceXKernel{a, c0, c1, domain.lo[0], domain.hi[0], xlow,
                                          xhigh, bc.dx});
    }

    // --- y-faces, over the EXTENDED i range (corners via the already-filled x-ghosts) ---
    int iglo = v.lo[0] - ng;
    int ighi = v.hi[0] + ng;
    if (bc.xlo == BCType::External)
      iglo = std::max(iglo, domain.lo[0]);
    if (bc.xhi == BCType::External)
      ighi = std::min(ighi, domain.hi[0]);
    if (detail::is_native_physical_bc(bc.ylo) && v.lo[1] == domain.lo[1]) {
      for_each_cell(Box2D{{iglo, domain.lo[1] - ng}, {ighi, domain.lo[1] - 1}},
                    detail::BCFaceYKernel{a, c0, c1, domain.lo[1], domain.hi[1], ylow,
                                          yhigh, bc.dy});
    }
    if (detail::is_native_physical_bc(bc.yhi) && v.hi[1] == domain.hi[1]) {
      for_each_cell(Box2D{{iglo, domain.hi[1] + 1}, {ighi, domain.hi[1] + ng}},
                    detail::BCFaceYKernel{a, c0, c1, domain.lo[1], domain.hi[1], ylow,
                                          yhigh, bc.dy});
    }
  }
}

/// Fills the physical-face ghosts of ALL components per @p bc (historical entry point, bit-identical).
inline void fill_physical_bc(MultiFab& mf, const Box2D& domain, const BCRec& bc) {
  fill_physical_bc_range(mf, domain, bc, 0, mf.ncomp());
}

/// ADC-369: fills the physical-face ghosts of a SINGLE component @p comp per @p bc -- the per-field aux
/// halo override. Applied AFTER the shared aux fill so a model-named field (component kAuxNamedBase+k)
/// can carry its own boundary policy (foextrap / dirichlet), overriding the shared aux BC for that
/// component only. It can even override the domain periodicity for that component (a Foextrap/Dirichlet
/// face re-fills a ghost that the shared periodic wrap had filled). Default paths never call this.
inline void fill_physical_bc(MultiFab& mf, const Box2D& domain, const BCRec& bc, int comp) {
  fill_physical_bc_range(mf, domain, bc, comp, comp + 1);
}

/// Per-field aux halo policy (ADC-369): a UNIFORM boundary policy for ONE model-named aux component,
/// declared via pops.AuxHalo. It is applied (aux_halo_override + the single-component fill_physical_bc)
/// to the NON-PERIODIC faces only -- periodic faces (a fully-periodic Cartesian domain, the polar
/// theta direction) keep their wrap, so a per-field policy never breaks the domain's periodic structure.
/// type is Foextrap (zero-gradient) or Dirichlet; value is the Dirichlet boundary value (ignored for
/// Foextrap). Default (no policy declared) leaves the shared aux BC untouched -> bit-identical.
struct AuxHaloPolicy {
  BCType type = BCType::Foextrap;
  Real value = Real(0);
};

/// Builds the effective override BCRec for a per-field aux halo: starts from the SHARED aux BC @p shared
/// (so periodic faces stay periodic) and replaces each NON-PERIODIC face with the policy @p p
/// (type + Dirichlet value). Feeding the result to fill_physical_bc(mf, domain, bc, comp) re-fills only
/// that component's physical-face ghosts with the field's own policy.
inline BCRec aux_halo_override(const BCRec& shared, const AuxHaloPolicy& p) {
  BCRec b = shared;
  if (b.xlo != BCType::Periodic) {
    b.xlo = p.type;
    b.xlo_val = p.value;
  }
  if (b.xhi != BCType::Periodic) {
    b.xhi = p.type;
    b.xhi_val = p.value;
  }
  if (b.ylo != BCType::Periodic) {
    b.ylo = p.type;
    b.ylo_val = p.value;
  }
  if (b.yhi != BCType::Periodic) {
    b.yhi = p.type;
    b.yhi_val = p.value;
  }
  return b;
}

/// COMPLETE ghost filling: fill_boundary (interior + periodic, periodicity deduced from
/// @p bc) THEN fill_physical_bc (physical edges). Usual entry point before assembling a residual.
inline void fill_ghosts(MultiFab& mf, const Box2D& domain, const BCRec& bc) {
  detail::validate_periodic_pairs(bc);
  Periodicity per{bc.xlo == BCType::Periodic, bc.ylo == BCType::Periodic};
  fill_boundary(mf, domain, per);
  fill_physical_bc(mf, domain, bc);
}

/// Execution-lane twin: physical faces remain rank-local while same-level/periodic exchange uses
/// the lane's isolated communicator. It never falls back to MPI_COMM_WORLD.
inline void fill_ghosts(MultiFab& mf, const Box2D& domain, const BCRec& bc,
                        const ExecutionLane& lane) {
  detail::validate_periodic_pairs(bc);
  Periodicity per{bc.xlo == BCType::Periodic, bc.ylo == BCType::Periodic};
  fill_boundary(mf, domain, lane, per);
  fill_physical_bc(mf, domain, bc);
}

}  // namespace pops
