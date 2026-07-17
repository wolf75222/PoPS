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
///
/// APPLYING THE INVERSE TO A VECTOR (``block_apply_inverse<N>``). Assembling the tensor COEFFICIENT
/// ``A = I + c*rho*M^{-1}`` reads the four entries of ``M^{-1}`` directly, so ``block_inverse<2>`` (each
/// entry a DIRECT division) is the right primitive there and is bit-identical to ``LorentzEliminator``'s
/// ``binv_11..22``. But the RHS flux ``F = M^{-1}(mx, my)`` and the reconstruct
/// ``v = M^{-1}(v^n - theta*dt*grad phi)`` apply the inverse to a VECTOR, and the retiring Schur brick
/// applied it with ``LorentzEliminator::apply_Binv`` = ``inv*(vx + w*vy)`` -- one reciprocal FACTORED
/// out of the bracket. Summing the pre-divided entries instead (``(1/det)*vx + (w/det)*vy``) rounds
/// differently, so ~1/3 of cells drift by a ULP each step and the trajectory leaves ``np.array_equal``.
/// ``block_apply_inverse<N>`` reproduces the factored order EXACTLY: it forms the adjugate ``adj`` (the
/// numerators of the inverse) and the single reciprocal ``inv = 1/det``, then ``out = inv*(adj . v)`` --
/// bit-for-bit ``apply_Binv`` for the Lorentz block, generic for any small block. It is the vector-apply
/// companion of ``block_inverse`` (which stays the coefficient primitive).

#pragma once

#include <pops/core/foundation/types.hpp>      // Real, POPS_HD
#include <pops/numerics/linalg/dense_eig.hpp>  // pops::detail::mat_inverse<N> (N>3 fallback)

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
  return mat_inverse<N>(A, inv,
                        tol);  // N != 2, 3: the generic dense solve (specializations below).
}

/// 2x2 closed form. A = [[a, b], [c, d]], det = a*d - b*c, A^{-1} = (1/det) [[d, -b], [-c, a]].
/// Each entry is a DIRECT division adj/det (not adj*(1/det)): for the rotation block
/// M = [[1, -w], [w, 1]] this yields 1/det, w/det, -w/det, 1/det -- LorentzEliminator's binv_11..22
/// bit-for-bit (the negation -(-w) is exact, det = 1 - (-w)*w == 1 + w*w).
///
/// FP-CONTRACTION SHAPE of det (the last-ULP trap; clang contracts fma at EVERY -O level).
/// LorentzEliminator's ``det = 1 + w*w`` has one multiply adjacent to the add, contracting to a single
/// fused fma(w, w, 1). Written as ``a*d - b*c`` the frontend may fuse the WRONG product (an extra
/// rounding, ~7% of random w one ULP off). Hoisting ``a*d`` into its own statement (exact for the
/// rotation block) leaves one multiply in the subtract, so it contracts to fma(-b, c, t) == fma(w, w, 1)
/// -- and compiles to the same two roundings as the eliminator when contraction is off.
template <>
POPS_HD inline bool block_inverse<2>(const Real (&A)[2][2], Real (&inv)[2][2], Real tol) {
  const Real a = A[0][0];
  const Real b = A[0][1];
  const Real c = A[1][0];
  const Real d = A[1][1];
  const Real t = a * d;  // hoisted: exact for the rotation block (1*1), own rounding otherwise
  const Real det = t - b * c;  // = 1 - (-w)*w == 1 + w*w (fma pairs on b*c, matching 1 + w*w)
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

/// Apply ``M^{-1}`` to the vector @p v into @p out, in the FACTORED order ``out = (1/det) * (adj . v)``
/// (one reciprocal outside the bracket). N == 2, 3: closed-form adjugate. N > 3: delegates to
/// ``block_inverse<N>`` (the dense inverse) and multiplies out -- no factored guarantee there, but the
/// small blocks the condensed solve uses are always N in {2, 3}. Returns false without touching @p out
/// when the block is singular. See the file header: for the Lorentz block this is ``apply_Binv``
/// bit-for-bit (``inv*(vx + w*vy)``), the parity the RHS-flux / reconstruct kernels rest on.
template <int N>
POPS_HD inline bool block_apply_inverse(const Real (&M)[N][N], const Real (&v)[N], Real (&out)[N],
                                        Real tol = Real(1e-300)) {
  Real inv[N][N];
  if (!block_inverse<N>(M, inv, tol))
    return false;
  for (int r = 0; r < N; ++r) {
    Real acc = Real(0);
    for (int c = 0; c < N; ++c)
      acc += inv[r][c] * v[c];
    out[r] = acc;
  }
  return true;
}

/// 2x2 factored apply. adj = [[d, -b], [-c, a]], inv = 1/det, out = inv*(adj . v). For the rotation
/// block M = [[1, -w], [w, 1]] this is out[0] = inv*(vx + w*vy), out[1] = inv*(vy - w*vx) -- bit-for-bit
/// LorentzEliminator::apply_Binv (d*vx = 1*vx and -b*vy = w*vy are exact; the single inv multiply of the
/// bracket matches, whereas summing pre-divided entries would not).
///
/// FP-CONTRACTION SHAPE (the last-ULP trap, measured in situ). apply_Binv's bracket ``vx + w*vy`` has
/// ONE multiply adjacent to the add, so under ``-ffp-contract=on`` (clang's default) it contracts to a
/// single fused fma(w, vy, vx). Writing the generic bracket as ``d*v[0] + (-b)*v[1]`` leaves TWO
/// multiplies and lets the compiler fuse the WRONG one (fma(d, v0, round(-b*v1)) -- an extra rounding):
/// ~7% of random inputs drifted by one ULP at -O3 (and 3/1024 cells of the in-situ golden). Hoisting
/// the diagonal product into its OWN statement (rounded separately -- exact for the rotation block's
/// d = 1) leaves exactly one multiply in the bracket, so the contraction pairs identically to
/// apply_Binv under BOTH regimes: fused -> fma(-b, v1, t0) == fma(w, vy, vx); unfused -> round(-b*v1)
/// then the add == apply_Binv compiled unfused. Same pinned shape for out[1] (hoist a*v[1]).
template <>
POPS_HD inline bool block_apply_inverse<2>(const Real (&M)[2][2], const Real (&v)[2],
                                           Real (&out)[2], Real tol) {
  const Real a = M[0][0];
  const Real b = M[0][1];
  const Real c = M[1][0];
  const Real d = M[1][1];
  const Real t = a * d;        // hoisted det shape, same as block_inverse<2> (fma pairs on b*c)
  const Real det = t - b * c;  // = 1 + w*w for the Lorentz rotation block
  if (det < tol && det > -tol)
    return false;
  const Real inv = Real(1) / det;
  const Real t0 =
      d * v[0];  // hoisted: exact for the rotation block (d = 1), own rounding otherwise
  const Real t1 = a * v[1];
  out[0] = inv * (t0 + (-b) * v[1]);  // inv*(vx + w*vy) for the rotation block (fma pairs on -b*v1)
  out[1] = inv * ((-c) * v[0] + t1);  // inv*(vy - w*vx)                       (fma pairs on -c*v0)
  return true;
}

/// 3x3 factored apply: out = (1/det) * (adj . v), adj = cofactor^T (same cofactors / det as
/// block_inverse<3>, pinned order), the single reciprocal factored out of each row bracket.
template <>
POPS_HD inline bool block_apply_inverse<3>(const Real (&M)[3][3], const Real (&v)[3],
                                           Real (&out)[3], Real tol) {
  const Real a00 = M[0][0], a01 = M[0][1], a02 = M[0][2];
  const Real a10 = M[1][0], a11 = M[1][1], a12 = M[1][2];
  const Real a20 = M[2][0], a21 = M[2][1], a22 = M[2][2];
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
  const Real inv = Real(1) / det;
  // adj = cofactor^T: adj[r][c] = C[c][r]. out[r] = inv * sum_c adj[r][c] * v[c].
  out[0] = inv * (c00 * v[0] + c10 * v[1] + c20 * v[2]);
  out[1] = inv * (c01 * v[0] + c11 * v[1] + c21 * v[2]);
  out[2] = inv * (c02 * v[0] + c12 * v[1] + c22 * v[2]);
  return true;
}

}  // namespace detail
}  // namespace pops
