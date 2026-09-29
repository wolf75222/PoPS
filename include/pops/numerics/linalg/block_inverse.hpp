/// @file
/// @brief Closed-form device inverse of a small dense block, with a PINNED operation order.
///
/// ``pops::detail::block_inverse<N>`` inverts a 2x2 or 3x3 ``Real[N][N]`` block by the
/// analytic adjugate/determinant formula in its safe exponent range, with each floating-point
/// operation written in a pinned order. All sizes first equilibrate rows and use partial-pivot
/// Gauss-Jordan to decide numerical admissibility. Outside the closed-form range, that balanced
/// inverse supplies the result. The relative pivot threshold is max(tol, N*epsilon), never a
/// determinant threshold carrying physical units. See docs/development/api_040/block_inverse_scaling_v1.md.
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
/// loops, no allocation -- capturable by value in a Kokkos/CUDA/HIP kernel. Returns false with
/// the destination untouched for invalid input, relative pivot refusal or a nonrepresentable
/// result. These device functions never throw. Host execution does not qualify a GPU backend.
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

// Version 1 numerical admission: scale each nonzero finite row to unit max norm,
// then use partial pivoting with a relative floor N*epsilon. Both the closed-form
// and scaled output paths share this decision, independently of row units.
template <int N>
POPS_HD inline bool block_equilibrated_inverse(const Real (&A)[N][N],
                                               Real (&inverse)[N][N], Real (&scales)[N],
                                               Real tolerance) {
  if (!std::isfinite(tolerance) || tolerance < Real(0))
    return false;
  Real normalized[N][N];
  for (int row = 0; row < N; ++row) {
    scales[row] = Real(0);
    for (int column = 0; column < N; ++column) {
      if (!std::isfinite(A[row][column]))
        return false;
      const Real magnitude = std::fabs(A[row][column]);
      if (magnitude > scales[row])
        scales[row] = magnitude;
    }
    if (scales[row] == Real(0))
      return false;
    for (int column = 0; column < N; ++column)
      normalized[row][column] = A[row][column] / scales[row];
  }
  const Real floor = Real(N) * std::numeric_limits<Real>::epsilon();
  if (!mat_inverse<N>(normalized, inverse, tolerance > floor ? tolerance : floor))
    return false;
  for (int row = 0; row < N; ++row)
    for (int column = 0; column < N; ++column)
      if (!std::isfinite(inverse[row][column]))
        return false;
  return true;
}

template <int N>
POPS_HD inline bool block_closed_form_range(const Real (&scales)[N]) {
  // N=2/3 adjugate, determinant and factored-vector expressions have degree at
  // most four. Fourth-root normal bounds, with room for their sums, keep their
  // intermediate products away from representational overflow/underflow.
  const Real upper = std::sqrt(std::sqrt(std::numeric_limits<Real>::max())) / Real(N + 1);
  const Real lower = std::sqrt(std::sqrt(std::numeric_limits<Real>::min())) * Real(N + 1);
  for (int row = 0; row < N; ++row)
    if (!std::isfinite(scales[row]) || scales[row] < lower || scales[row] > upper)
      return false;
  return true;
}

template <int N>
POPS_HD inline bool block_publish_scaled_inverse(const Real (&normalized_inverse)[N][N],
                                                  const Real (&scales)[N], Real (&out)[N][N]) {
  Real candidate[N][N];
  for (int row = 0; row < N; ++row)
    for (int column = 0; column < N; ++column) {
      candidate[row][column] = normalized_inverse[row][column] / scales[column];
      if (!std::isfinite(candidate[row][column]))
        return false;
    }
  for (int row = 0; row < N; ++row)
    for (int column = 0; column < N; ++column)
      out[row][column] = candidate[row][column];
  return true;
}

// Apply R^-1 D^-1 v without materializing A^-1. A representable solution can
// exist when the inverse itself is not representable. Each row is accumulated
// by decreasing binary exponent, renormalizing after cancellation. In
// particular a small independent component is not lost by one global RHS scale.
template <int N>
POPS_HD inline bool block_publish_scaled_apply(const Real (&inverse)[N][N],
                                                const Real (&scales)[N], const Real (&v)[N],
                                                Real (&out)[N]) {
  for (int column = 0; column < N; ++column)
    if (!std::isfinite(v[column]))
      return false;
  Real candidate[N];
  for (int row = 0; row < N; ++row) {
    Real fractions[N]{};
    int exponents[N]{};
    for (int column = 0; column < N; ++column) {
      if (inverse[row][column] == Real(0) || v[column] == Real(0))
        continue;
      int ei = 0, ev = 0, es = 0, adjustment = 0;
      const Real fi = std::frexp(inverse[row][column], &ei);
      const Real fv = std::frexp(v[column], &ev);
      const Real fs = std::frexp(scales[column], &es);
      fractions[column] = std::frexp((fi * fv) / fs, &adjustment);
      exponents[column] = ei + ev - es + adjustment;
    }
    for (int first = 0; first < N; ++first)
      for (int second = first + 1; second < N; ++second)
        if (exponents[second] > exponents[first]) {
          const int exponent = exponents[first];
          exponents[first] = exponents[second];
          exponents[second] = exponent;
          const Real fraction = fractions[first];
          fractions[first] = fractions[second];
          fractions[second] = fraction;
        }
    Real sum = Real(0);
    int sum_exponent = 0;
    for (int column = 0; column < N; ++column) {
      if (fractions[column] == Real(0))
        continue;
      if (sum == Real(0)) {
        sum = fractions[column];
        sum_exponent = exponents[column];
      } else {
        const int common = sum_exponent > exponents[column] ? sum_exponent : exponents[column];
        const Real combined = std::ldexp(sum, sum_exponent - common) +
                              std::ldexp(fractions[column], exponents[column] - common);
        int adjustment = 0;
        sum = std::frexp(combined, &adjustment);
        sum_exponent = common + adjustment;
      }
    }
    candidate[row] = std::ldexp(sum, sum_exponent);
    if (!std::isfinite(candidate[row]))
      return false;
  }
  for (int row = 0; row < N; ++row)
    out[row] = candidate[row];
  return true;
}

/// Closed-form inverse of a small dense block into @p inv, with a pinned operation order.
///
/// N == 2, 3: retain the pinned direct divisions when row scales are in the closed-form range.
/// All sizes use the same row-equilibrated partial-pivot admission, with relative threshold
/// max(tol, N*epsilon). @p tol must be finite and nonnegative. Return false without touching
/// @p inv on refusal, invalid input, or an inverse outside Real's representable range.
template <int N>
POPS_HD inline bool block_inverse(const Real (&A)[N][N], Real (&inv)[N][N],
                                  Real tol = Real(1e-300)) {
  Real normalized_inverse[N][N], scales[N];
  return block_equilibrated_inverse(A, normalized_inverse, scales, tol) &&
         block_publish_scaled_inverse(normalized_inverse, scales, inv);
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
  Real normalized_inverse[2][2], scales[2];
  if (!block_equilibrated_inverse(A, normalized_inverse, scales, tol))
    return false;
  if (!block_closed_form_range(scales))
    return block_publish_scaled_inverse(normalized_inverse, scales, inv);
  const Real a = A[0][0];
  const Real b = A[0][1];
  const Real c = A[1][0];
  const Real d = A[1][1];
  const Real t = a * d;  // hoisted: exact for the rotation block (1*1), own rounding otherwise
  const Real det = t - b * c;  // = 1 - (-w)*w == 1 + w*w (fma pairs on b*c, matching 1 + w*w)
  if (!std::isfinite(det) || det == Real(0))
    return block_publish_scaled_inverse(normalized_inverse, scales, inv);
  const Real candidate[2][2] = {{d / det, -b / det}, {-c / det, a / det}};
  for (int row = 0; row < 2; ++row)
    for (int column = 0; column < 2; ++column)
      if (!std::isfinite(candidate[row][column]))
        return block_publish_scaled_inverse(normalized_inverse, scales, inv);
  for (int row = 0; row < 2; ++row)
    for (int column = 0; column < 2; ++column)
      inv[row][column] = candidate[row][column];
  return true;
}

/// 3x3 closed form via the cofactor/adjugate, pinned order. det = a00*C00 + a01*C01 + a02*C02 with
/// the cofactors Cij; A^{-1}[i][j] = Cji / det (adjugate = cofactor transpose). Each output entry is a
/// DIRECT division by det. For a block-diagonal 3D rotation ([[1,-w,0],[w,1,0],[0,0,1]]) the (0,1)
/// 2x2 sub-block matches the 2x2 case above and the z row/col reduces to the identity.
template <>
POPS_HD inline bool block_inverse<3>(const Real (&A)[3][3], Real (&inv)[3][3], Real tol) {
  Real normalized_inverse[3][3], scales[3];
  if (!block_equilibrated_inverse(A, normalized_inverse, scales, tol))
    return false;
  if (!block_closed_form_range(scales))
    return block_publish_scaled_inverse(normalized_inverse, scales, inv);
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
  if (!std::isfinite(det) || det == Real(0))
    return block_publish_scaled_inverse(normalized_inverse, scales, inv);
  // inv = adj / det = cofactor^T / det.
  const Real candidate[3][3] = {{c00 / det, c10 / det, c20 / det},
                               {c01 / det, c11 / det, c21 / det},
                               {c02 / det, c12 / det, c22 / det}};
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      if (!std::isfinite(candidate[row][column]))
        return block_publish_scaled_inverse(normalized_inverse, scales, inv);
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      inv[row][column] = candidate[row][column];
  return true;
}

/// Apply ``M^{-1}`` to the vector @p v into @p out, in the FACTORED order ``out = (1/det) * (adj . v)``
/// (one reciprocal outside the bracket) within the N=2/3 closed-form range. Otherwise apply the
/// balanced inverse using binary-scaled products and sums, without materializing M^-1. A finite
/// solution can therefore succeed even when M^-1 is not representable. The same relative pivot
/// admission governs both paths. Return false without touching @p out for invalid input, refusal
/// or a nonrepresentable solution. The ordinary rotation-block operation tree remains pinned.
template <int N>
POPS_HD inline bool block_apply_inverse(const Real (&M)[N][N], const Real (&v)[N], Real (&out)[N],
                                        Real tol = Real(1e-300)) {
  Real normalized_inverse[N][N], scales[N];
  return block_equilibrated_inverse(M, normalized_inverse, scales, tol) &&
         block_publish_scaled_apply(normalized_inverse, scales, v, out);
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
  Real normalized_inverse[2][2], scales[2];
  if (!block_equilibrated_inverse(M, normalized_inverse, scales, tol))
    return false;
  Real vector_scales[2];
  for (int row = 0; row < 2; ++row)
    vector_scales[row] = v[row] == Real(0) ? Real(1) : std::fabs(v[row]);
  if (!block_closed_form_range(scales) || !block_closed_form_range(vector_scales))
    return block_publish_scaled_apply(normalized_inverse, scales, v, out);
  const Real a = M[0][0];
  const Real b = M[0][1];
  const Real c = M[1][0];
  const Real d = M[1][1];
  const Real t = a * d;        // hoisted det shape, same as block_inverse<2> (fma pairs on b*c)
  const Real det = t - b * c;  // = 1 + w*w for the Lorentz rotation block
  if (!std::isfinite(det) || det == Real(0))
    return block_publish_scaled_apply(normalized_inverse, scales, v, out);
  const Real inv = Real(1) / det;
  const Real t0 =
      d * v[0];  // hoisted: exact for the rotation block (d = 1), own rounding otherwise
  const Real t1 = a * v[1];
  const Real candidate[2] = {
      inv * (t0 + (-b) * v[1]),  // pinned fma pairing on -b*v1
      inv * ((-c) * v[0] + t1)}; // pinned fma pairing on -c*v0
  for (int row = 0; row < 2; ++row)
    if (!std::isfinite(candidate[row]))
      return block_publish_scaled_apply(normalized_inverse, scales, v, out);
  for (int row = 0; row < 2; ++row)
    out[row] = candidate[row];
  return true;
}

/// 3x3 factored apply: out = (1/det) * (adj . v), adj = cofactor^T (same cofactors / det as
/// block_inverse<3>, pinned order), the single reciprocal factored out of each row bracket.
template <>
POPS_HD inline bool block_apply_inverse<3>(const Real (&M)[3][3], const Real (&v)[3],
                                           Real (&out)[3], Real tol) {
  Real normalized_inverse[3][3], scales[3];
  if (!block_equilibrated_inverse(M, normalized_inverse, scales, tol))
    return false;
  Real vector_scales[3];
  for (int row = 0; row < 3; ++row)
    vector_scales[row] = v[row] == Real(0) ? Real(1) : std::fabs(v[row]);
  if (!block_closed_form_range(scales) || !block_closed_form_range(vector_scales))
    return block_publish_scaled_apply(normalized_inverse, scales, v, out);
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
  if (!std::isfinite(det) || det == Real(0))
    return block_publish_scaled_apply(normalized_inverse, scales, v, out);
  const Real inv = Real(1) / det;
  // adj = cofactor^T: adj[r][c] = C[c][r]. out[r] = inv * sum_c adj[r][c] * v[c].
  const Real candidate[3] = {
      inv * (c00 * v[0] + c10 * v[1] + c20 * v[2]),
      inv * (c01 * v[0] + c11 * v[1] + c21 * v[2]),
      inv * (c02 * v[0] + c12 * v[1] + c22 * v[2])};
  for (int row = 0; row < 3; ++row)
    if (!std::isfinite(candidate[row]))
      return block_publish_scaled_apply(normalized_inverse, scales, v, out);
  for (int row = 0; row < 3; ++row)
    out[row] = candidate[row];
  return true;
}

}  // namespace detail
}  // namespace pops
