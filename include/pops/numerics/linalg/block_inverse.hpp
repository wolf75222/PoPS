/// @file
/// @brief Closed-form device inverse of a small dense block, with a PINNED operation order.
///
/// ``pops::detail::block_inverse<N>`` inverts a 2x2 or 3x3 ``Real[N][N]`` block by the
/// analytic adjugate/determinant formula, with EVERY floating-point operation written out in a
/// fixed order so the result does not depend on the compiler's freedom to reassociate. N>3 (and any
/// runtime request for a larger block) falls through to the generic Gauss-Jordan
/// ``pops::detail::mat_inverse<N>`` (dense_eig.hpp) -- so the intrinsic is a bit-stable FAST PATH for
/// the small blocks, never a new capability.
///
/// WHY A SEPARATE, PINNED-ORDER INTRINSIC. The condensed-implicit codegen (ADC-637) assembles the
/// tensor elliptic coefficient ``A = I + c*rho*M^{-1}`` from a per-cell block ``M = I - theta*dt*J``
/// and MUST reproduce, bit-for-bit, the entries the hand-written ``LorentzEliminator`` (the retiring
/// Schur brick's B^{-1}) produced -- the retirement parity gate rests on it. For the Lorentz block
/// ``M = [[1, -w], [w, 1]]`` (``w = theta*dt*B_z``) the pinned 2x2 formula below reduces EXACTLY to
/// ``LorentzEliminator``: ``det = 1*1 - (-w)*w`` is ``1 + w*w`` in IEEE-754 (the sign flip and the
/// subtract-of-a-negation are exact), and the four entries ``d/det, -b/det, -c/det, a/det`` become
/// ``1/det, w/det, -w/det, 1/det`` -- the same four DIRECT divisions ``binv_11..22`` return. Because
/// the operation tree is fixed HERE (not left to the symbolic simplifier or the C++ optimizer),
/// bit-identity holds independently of ``-O`` level (fast-math stays forbidden on the elliptic TUs).
///
/// DEVICE / ALLOCATION CONTRACT (mirrors mat_inverse): POPS_HD, stack-only fixed buffers, bounded
/// loops, no allocation, no std:: call on the closed-form paths -- capturable by value in a
/// Kokkos/CUDA/HIP kernel. Returns false (``inv`` untouched-meaningful) when the determinant/pivot is
/// below @p tol (singular); the closed-form paths never throw.

#pragma once

#include <pops/core/foundation/types.hpp>       // Real, POPS_HD
#include <pops/numerics/linalg/dense_eig.hpp>    // pops::detail::mat_inverse<N> (N>3 fallback)

namespace pops {
namespace detail {

/// Closed-form inverse of a small dense block into @p inv, with a pinned operation order.
///
/// N == 2, 3: analytic adjugate / determinant (each entry a DIRECT division by @p det, so the
/// rotation-block case reduces bit-for-bit to LorentzEliminator -- see the file header). N > 3:
/// delegates to ``mat_inverse<N>`` (Gauss-Jordan, partial pivot). Returns false without touching
/// @p inv when the block is singular (|det| < @p tol on the closed-form paths, the pivot test in the
/// fallback); the closed-form paths are branch-free otherwise (no throw on device).
template <int N>
POPS_HD inline bool block_inverse(const Real (&A)[N][N], Real (&inv)[N][N],
                                  Real tol = Real(1e-300)) {
  return mat_inverse<N>(A, inv, tol);  // N != 2, 3: the generic dense solve (specializations below).
}

/// 2x2 closed form. A = [[a, b], [c, d]], det = a*d - b*c, A^{-1} = (1/det) [[d, -b], [-c, a]].
/// Each entry is a DIRECT division adj/det (not adj*(1/det)): for the rotation block
/// M = [[1, -w], [w, 1]] this yields 1/det, w/det, -w/det, 1/det -- LorentzEliminator's binv_11..22
/// bit-for-bit (the negation -(-w) is exact, det = 1 - (-w)*w == 1 + w*w).
template <>
POPS_HD inline bool block_inverse<2>(const Real (&A)[2][2], Real (&inv)[2][2], Real tol) {
  const Real a = A[0][0];
  const Real b = A[0][1];
  const Real c = A[1][0];
  const Real d = A[1][1];
  const Real det = a * d - b * c;  // = 1 - (-w)*w == 1 + w*w for the Lorentz rotation block
  if (det < tol && det > -tol)
    return false;
  inv[0][0] = d / det;
  inv[0][1] = -b / det;
  inv[1][0] = -c / det;
  inv[1][1] = a / det;
  return true;
}

/// 3x3 closed form via the cofactor/adjugate, pinned order. det = a00*C00 + a01*C01 + a02*C02 with
/// the cofactors Cij; A^{-1}[i][j] = Cji / det (adjugate = cofactor transpose). Each output entry is a
/// DIRECT division by det. For a block-diagonal 3D rotation ([[1,-w,0],[w,1,0],[0,0,1]]) the (0,1)
/// 2x2 sub-block matches the 2x2 case above and the z row/col reduces to the identity.
template <>
POPS_HD inline bool block_inverse<3>(const Real (&A)[3][3], Real (&inv)[3][3], Real tol) {
  const Real a00 = A[0][0], a01 = A[0][1], a02 = A[0][2];
  const Real a10 = A[1][0], a11 = A[1][1], a12 = A[1][2];
  const Real a20 = A[2][0], a21 = A[2][1], a22 = A[2][2];
  // Cofactors (signed 2x2 minors), pinned order.
  const Real c00 = a11 * a22 - a12 * a21;
  const Real c01 = a12 * a20 - a10 * a22;
  const Real c02 = a10 * a21 - a11 * a20;
  const Real c10 = a02 * a21 - a01 * a22;
  const Real c11 = a00 * a22 - a02 * a20;
  const Real c12 = a01 * a20 - a00 * a21;
  const Real c20 = a01 * a12 - a02 * a11;
  const Real c21 = a02 * a10 - a00 * a12;
  const Real c22 = a00 * a11 - a01 * a10;
  const Real det = a00 * c00 + a01 * c01 + a02 * c02;
  if (det < tol && det > -tol)
    return false;
  // inv = adj / det = cofactor^T / det.
  inv[0][0] = c00 / det;
  inv[0][1] = c10 / det;
  inv[0][2] = c20 / det;
  inv[1][0] = c01 / det;
  inv[1][1] = c11 / det;
  inv[1][2] = c21 / det;
  inv[2][0] = c02 / det;
  inv[2][1] = c12 / det;
  inv[2][2] = c22 / det;
  return true;
}

}  // namespace detail
}  // namespace pops
