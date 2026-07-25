/// @file
/// @brief Interface reconstruction policies: MUSCL limiters and WENO5-Z.
///
/// Each policy exposes:
///   - `formal_order`: formal accuracy in smooth regions.
///   - `n_ghost`: required face-operator ghost storage (never used to infer the algorithm).
///   - exactly one pointwise protocol: cell value, limited slope, or sampled stencil.
///
/// All policies are POPS_HD (no std::, no branch to UB). The limiter is a template parameter
/// in assemble_rhs / reconstruct (static polymorphism, inlined on device). INVARIANT: a
/// reconstruction policy is POINTWISE -- it does not own or capture a global array.  A sampled
/// stencil policy receives a device-callable scalar sampler and chooses the integer offsets it
/// needs.  The mesh access and orientation remain encapsulated by reconstruct (face_flux.hpp).

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <cmath>
#include <concepts>
#include <type_traits>

namespace pops {

/// First-order reconstruction (piecewise constant): zero slope, 1 ghost.
///
/// Minimal policy: no slope computation and no neighbour read. POPS_HD.
struct NoSlope {
  static constexpr int formal_order = 1;
  static constexpr int n_ghost = 1;
  POPS_HD Real cell_face_value(Real value) const { return value; }
};

/// Explicit embedded-boundary reconstruction capability.
///
/// A stencil radius is not a geometry contract: a future one-ghost policy may still inspect an
/// inactive neighbour.  New reconstruction policies therefore opt in deliberately after proving
/// that their face states are defined using only the two active cells selected by the EB operator.
/// The default stays fail-closed.
template <class Reconstruction>
struct EmbeddedBoundaryReconstructionSupport : std::false_type {};

template <>
struct EmbeddedBoundaryReconstructionSupport<NoSlope> : std::true_type {};

template <class Reconstruction>
inline constexpr bool supports_embedded_boundary_reconstruction_v =
    EmbeddedBoundaryReconstructionSupport<Reconstruction>::value;

/// minmod limiter: TVD (Total Variation Diminishing), 2 ghosts, order 2 in smooth regions.
///
/// Returns min(|a|,|b|)*sgn(a) if a and b have the same sign, 0 otherwise. Implemented without
/// std::min / std::abs to stay device-safe (no <cmath> required). Order 1 locally at extrema
/// (clips smooth peaks): prefer VanLeer when smooth growth modes must survive.
struct Minmod {
  static constexpr int formal_order = 2;
  static constexpr int n_ghost = 2;
  POPS_HD Real limited_slope(Real backward, Real forward) const {
    const Real a = backward;
    const Real b = forward;
    if (a * b <= Real(0))
      return Real(0);
    const Real fa = a < 0 ? -a : a, fb = b < 0 ? -b : b;  // |.| device-safe
    return (fa < fb) ? a : b;
  }
};

/// van Leer limiter: smooth, 2 ghosts, better order at extrema than Minmod.
///
/// Harmonic average of the differences: 2ab/(a+b) if same sign, 0 otherwise. No sign branch
/// (no std::abs). Preferred over Minmod for preserving smooth growth modes (less
/// dissipative at the density profile extrema).
struct VanLeer {
  static constexpr int formal_order = 2;
  static constexpr int n_ghost = 2;
  POPS_HD Real limited_slope(Real backward, Real forward) const {
    const Real a = backward;
    const Real b = forward;
    const Real ab = a * b;
    if (ab <= Real(0))
      return Real(0);
    return Real(2) * ab / (a + b);
  }
};

/// weno5z: WENO5-Z reconstruction (Borges 2008) at one interface, on a 5-point stencil.
///
/// Returns the reconstructed value at the face BETWEEN v0 and vp1 (face +dir of cell v0).
/// For the -dir face, call weno5z(vp2, vp1, v0, vm1, vm2) (reversed stencil). POPS_HD.
/// INVARIANT: purely combinatorial computation, no branch on signs -- the beta and tau5
/// indicators are squares, always >= 0; only the absolute value of (b0-b2) is taken via a
/// ternary (device-safe, avoids std::abs).
/// Must NOT be called directly by a mesh user: go through the Weno5 policy and the reconstruct
/// function of spatial_operator.hpp.
POPS_HD inline Real weno5z(Real vm2, Real vm1, Real v0, Real vp1, Real vp2,
                           Real eps = kWenoEpsilon) {
  // ADC-645: eps is the WENO-Z smoothness regulariser (default = the historical kWenoEpsilon
  // literal, bit-identical); a per-block override reaches here through Weno5::eps.
  // three third-order reconstructions of the +x face of v0
  const Real q0 = (Real(2) * vm2 - Real(7) * vm1 + Real(11) * v0) / Real(6);
  const Real q1 = (-vm1 + Real(5) * v0 + Real(2) * vp1) / Real(6);
  const Real q2 = (Real(2) * v0 + Real(5) * vp1 - vp2) / Real(6);
  // smoothness indicators (Jiang-Shu)
  const Real b0 = Real(13) / Real(12) * (vm2 - 2 * vm1 + v0) * (vm2 - 2 * vm1 + v0) +
                  Real(0.25) * (vm2 - 4 * vm1 + 3 * v0) * (vm2 - 4 * vm1 + 3 * v0);
  const Real b1 = Real(13) / Real(12) * (vm1 - 2 * v0 + vp1) * (vm1 - 2 * v0 + vp1) +
                  Real(0.25) * (vm1 - vp1) * (vm1 - vp1);
  const Real b2 = Real(13) / Real(12) * (v0 - 2 * vp1 + vp2) * (v0 - 2 * vp1 + vp2) +
                  Real(0.25) * (3 * v0 - 4 * vp1 + vp2) * (3 * v0 - 4 * vp1 + vp2);
  // WENO-Z weights: alpha_k = d_k (1 + (tau5/(eps+beta_k))^2), tau5 = |beta0 - beta2|
  const Real tau5 = (b0 - b2 < 0 ? b2 - b0 : b0 - b2);
  const Real a0 = (Real(1) / Real(10)) * (Real(1) + (tau5 / (eps + b0)) * (tau5 / (eps + b0)));
  const Real a1 = (Real(6) / Real(10)) * (Real(1) + (tau5 / (eps + b1)) * (tau5 / (eps + b1)));
  const Real a2 = (Real(3) / Real(10)) * (Real(1) + (tau5 / (eps + b2)) * (tau5 / (eps + b2)));
  const Real inv = Real(1) / (a0 + a1 + a2);
  return (a0 * q0 + a1 * q1 + a2 * q2) * inv;
}

/// WENO5 policy: declares a three-cell storage requirement and delegates its sampled stencil to
/// weno5z.
///
/// The storage requirement is not used to identify WENO; stencil_face_value is the explicit
/// protocol.  The sampler maps offsets relative to the source cell into the oriented face
/// direction, so the same policy reconstructs both faces without the core knowing its stencil.
struct Weno5 {
  static constexpr int formal_order = 5;
  static constexpr int n_ghost = 3;
  static constexpr int stencil_min_offset = -2;
  static constexpr int stencil_max_offset = 2;
  // ADC-645: the WENO-Z smoothness regulariser (default = the historical kWenoEpsilon, so a
  // default-constructed Weno5 is bit-identical); the reconstruction passes it to weno5z.
  Real eps = kWenoEpsilon;
  POPS_HD void set_smoothness_epsilon(Real value) { eps = value; }
  template <class Sample>
  POPS_HD Real stencil_face_value(const Sample& sample) const {
    const Real vm2 = sample(-2);
    const Real vm1 = sample(-1);
    const Real v0 = sample(0);
    const Real vp1 = sample(1);
    const Real vp2 = sample(2);
    return weno5z(vm2, vm1, v0, vp1, vp2, eps);
  }
};

/// Small reconstruction-policy protocols.  The ghost count remains a storage-capacity contract;
/// it never selects the numerical algorithm or the sampled offsets.  A new policy opts into
/// exactly one pointwise protocol, so a four-ghost MUSCL policy cannot accidentally become a
/// sampled stencil merely because its storage envelope is wider.
template <class Reconstruction>
concept CellValueReconstruction = requires(const Reconstruction& reconstruction, Real value) {
  { reconstruction.cell_face_value(value) } -> std::convertible_to<Real>;
};

template <class Reconstruction>
concept SlopeReconstruction =
    requires(const Reconstruction& reconstruction, Real backward, Real forward) {
      { reconstruction.limited_slope(backward, forward) } -> std::convertible_to<Real>;
    };

/// Minimal compile-time probe for the sampled-stencil protocol.  Production reconstruction passes
/// an equally small POD sampler backed by ConstArray4.  Policies are expected to use only
/// `sample(integer_offset)`; they never receive a mesh, direction, component, or host callback.
struct ReconstructionSamplerProbe {
  POPS_HD Real operator()(int) const { return Real(0); }
};

/// SFINAE envelope trait keeps cell-value and slope policies completely free of stencil metadata.
/// Only a policy that declares both bounds can opt into the sampled-stencil protocol.
template <class Reconstruction, class = void>
struct ReconstructionStencilEnvelope {
  static constexpr bool declared = false;
  static constexpr bool ordered = false;
  static constexpr int min_offset = 0;
  static constexpr int max_offset = -1;
};

template <class Reconstruction>
struct ReconstructionStencilEnvelope<
    Reconstruction,
    std::void_t<decltype(Reconstruction::stencil_min_offset),
                decltype(Reconstruction::stencil_max_offset)>> {
  static constexpr bool declared = true;
  static constexpr int min_offset = static_cast<int>(Reconstruction::stencil_min_offset);
  static constexpr int max_offset = static_cast<int>(Reconstruction::stencil_max_offset);
  static constexpr bool ordered = min_offset <= max_offset;
};

template <class Reconstruction>
concept StencilReconstruction =
    ReconstructionStencilEnvelope<Reconstruction>::declared &&
    ReconstructionStencilEnvelope<Reconstruction>::ordered &&
    requires(const Reconstruction& reconstruction, const ReconstructionSamplerProbe& sample) {
      { reconstruction.stencil_face_value(sample) } -> std::convertible_to<Real>;
    };

template <class Reconstruction>
inline constexpr int reconstruction_protocol_count =
    static_cast<int>(CellValueReconstruction<Reconstruction>) +
    static_cast<int>(SlopeReconstruction<Reconstruction>) +
    static_cast<int>(StencilReconstruction<Reconstruction>);

template <class Reconstruction>
concept ReconstructionMetadata = requires {
  { Reconstruction::formal_order } -> std::convertible_to<int>;
  { Reconstruction::n_ghost } -> std::convertible_to<int>;
  requires Reconstruction::formal_order > 0;
  requires Reconstruction::n_ghost > 0;
};

template <class Reconstruction>
concept ReconstructionPolicy =
    ReconstructionMetadata<Reconstruction> && reconstruction_protocol_count<Reconstruction> == 1;

/// A sampled policy explicitly declares the envelope it may query.  `n_ghost` remains the storage
/// capacity required by the surrounding face operator (one cell more than a source-centred
/// offset at a valid-box edge); it neither selects a protocol nor supplies the offsets.
template <class Reconstruction>
consteval bool stencil_envelope_fits_storage_contract() {
  if constexpr (!StencilReconstruction<Reconstruction>) {
    return true;
  } else {
    using Envelope = ReconstructionStencilEnvelope<Reconstruction>;
    return Reconstruction::n_ghost > 0 &&
           Envelope::min_offset >= 1 - Reconstruction::n_ghost &&
           Envelope::max_offset <= Reconstruction::n_ghost - 1;
  }
}

template <class Reconstruction>
inline constexpr bool stencil_envelope_fits_storage =
    stencil_envelope_fits_storage_contract<Reconstruction>();

template <class Reconstruction>
inline Reconstruction configured_reconstruction(Real smoothness_epsilon = kWenoEpsilon) {
  static_assert(
      ReconstructionPolicy<Reconstruction>,
      "a reconstruction policy must declare positive formal_order/n_ghost metadata and implement "
      "exactly one pointwise protocol");
  Reconstruction reconstruction{};
  if constexpr (requires(Reconstruction& value, Real eps) {
                  value.set_smoothness_epsilon(eps);
                })
    reconstruction.set_smoothness_epsilon(smoothness_epsilon);
  return reconstruction;
}

}  // namespace pops
