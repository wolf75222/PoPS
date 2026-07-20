#pragma once
#include <pops/diagnostics/fallback_diagnostics.hpp>

/// @file
/// @brief Spectrum extremes (real parts) of a small dense matrix: signed wave-speed bounds
/// supplied by a model (exact HLL via flux Jacobian).
///
/// GENERIC utility: the core knows nothing of the calling model -- it receives a dense block
/// Real[N][N] and returns min/max of the real parts of its spectrum. Intended consumer: the DSL
/// codegen wave_speeds_from_jacobian (the model supplies dF/dU, possibly by diagonal blocks;
/// each block passes through here). Any other "small eigenvalues on the stack" use is legitimate.
///
/// Algorithm (eigenvalues ONLY, never the vectors):
///   N == 1, 2: closed form (trace/determinant);
///   N >= 3: Hessenberg reduction (Householder, without accumulation) then QR iteration with
///               implicit Francis DOUBLE SHIFT and deflation (EISPACK/hqr formulation) --
///               complex conjugate pairs stay in real arithmetic (2x2 blocks),
///               exceptional shifts after 10 and 20 iterations on a single block.
///
/// ROBUSTNESS CONTRACT: if a block does not converge under the iteration cap, the result is the
/// Gershgorin diagnostic enclosure over the WHOLE matrix (converged = false): an EXTERNAL bound
/// on all real parts. It is not a certified spectrum and the HLL/Roe providers reject it. If
/// the fallback triggers, lmin/lmax are NOT the eigenvalues and max_im is 0 by CONVENTION (nothing
/// was computed, this is NOT a real-spectrum signal): any bit-match against an eig reference is then
/// void. converged (or the fallback output parameter) decides: never read lmin/lmax/max_im without
/// consulting it. The default cap (100) is sized so that the usual near-degenerate companion blocks
/// converge; the enclosure remains available for diagnostics of out-of-envelope cases.
///
/// HYPERBOLICITY: max_im returns the largest |Im(lambda)| encountered. A hyperbolic system has a
/// real spectrum (max_im ~ 0); a model that loses hyperbolicity does not silently receive a
/// plausible-but-wrong speed -- the caller decides (assert, warning, clamp). A caller wanting only a
/// real/complex verdict (not the extremes) uses pops::real_spectrum / EigBounds::all_real (ADC-276),
/// which couple this max_im check with convergence so a non-converged block is never reported real.
///
/// PRECISION: simple, separated eigenvalues -> machine precision (tested at rtol 1e-10).
/// CLUSTERED eigenvalues of a non-symmetric matrix: conditioning ~ eps^(1/m) for a near-multiplicity
/// m (a limit of the problem, not of the algorithm) -- do not expect 1e-10 at a triple point. No
/// exactness claim is made under the iteration cap: the cap bounds the COST, the fallback bounds the
/// RESULT.
///
/// Device: POPS_HD, stack buffers (O(N^2)), zero allocation, zero std:: container, bounded loops,
/// no recursion -- only std::sqrt / std::fabs (resolved on device under nvcc/Kokkos, as on the flux
/// path).

#include <cmath>
#include <limits>

#include <pops/core/foundation/types.hpp>

namespace pops {

/// Default RELATIVE imaginary tolerance for the small-block spectrum classifier (all_real /
/// has_complex_pair / real_spectrum). Covers a real multiplicity up to m=3 (eps^(1/3) ~ 6e-6); see
/// EigBounds::all_real for the tolerance contract. ADC-643 single-sources the default; ADC-645 will
/// later promote it to a DSL knob.
inline constexpr Real kEigImagTol = Real(1e-5);

/// Strict RELATIVE tolerance used by physical Roe providers.  It only absorbs the bounded
/// round-off floor of the native QR implementation; it is deliberately distinct from the much
/// looser diagnostic classifier default above.  Callers that require an exact-zero comparison can
/// still pass im_tol = 0 explicitly.
inline constexpr Real kEigStrictImagTol =
    Real(64) * std::numeric_limits<Real>::epsilon();

/// Result of real_eig_minmax: real-part extremes + diagnostic. The consumer (DSL codegen
/// wave_speeds_from_jacobian, ADC-87) receives the WHOLE structure: converged and max_im are part
/// of the safety contract, no overload silently drops them.
struct EigBounds {
  Real lmin;    ///< smallest real part (or Gershgorin lower bound if !converged)
  Real lmax;    ///< largest real part (or Gershgorin upper bound if !converged)
  Real max_im;  ///< largest |Im(lambda)| encountered (0 = real spectrum: hyperbolic). Has this
                ///< meaning ONLY if converged: under fallback it is 0 by CONVENTION (the spectrum
                ///< is not computed), certainly not a hyperbolicity signal -- read converged first.
  bool converged;  ///< false -> Gershgorin fallback (valid external bound, NOT the spectrum)

  /// True only when the eigensolve converged and every returned spectral diagnostic is finite.
  /// This is the common gate for consumers that must not turn a diagnostic enclosure or NaN into
  /// physics. max_im is a magnitude and therefore additionally required to be non-negative.
  POPS_HD bool valid() const {
    return converged && std::isfinite(static_cast<double>(lmin)) &&
           std::isfinite(static_cast<double>(lmax)) &&
           std::isfinite(static_cast<double>(max_im)) && max_im >= Real(0);
  }

  /// Tri-state scalar for branchless generated kernels: 1 = certified real at @p im_tol,
  /// 0 = converged finite complex spectrum, NaN = invalid/non-converged solve. Unlike all_real(),
  /// this preserves the distinction needed by a fail-closed physical transform.
  POPS_HD Real real_status(Real im_tol = kEigImagTol) const {
    if (!valid() || !std::isfinite(static_cast<double>(im_tol)) || im_tol < Real(0))
      return std::numeric_limits<Real>::quiet_NaN();
    const Real rho = std::fabs(lmin) > std::fabs(lmax) ? std::fabs(lmin) : std::fabs(lmax);
    const Real scale = rho > Real(1) ? rho : Real(1);
    return max_im <= im_tol * scale ? Real(1) : Real(0);
  }

  /// REAL-SPECTRUM PREDICATE (ADC-276). True iff the block CONVERGED and the largest imaginary part is
  /// within a RELATIVE tolerance of zero: max_im <= im_tol * max(|lmin|, |lmax|, 1). The threshold is
  /// RELATIVE to the spectral magnitude, so the verdict is scale-invariant; the floor of 1 keeps a pure
  /// rotation / zero matrix (real-part scale 0) from collapsing the threshold to 0.
  /// MULTIPLICITY: a near m-fold REAL root is ill-conditioned and acquires a spurious relative |Im| of
  /// order eps^(1/m) (see PRECISION in the file header). The default im_tol = 1e-5 covers m up to 3 (the
  /// 3x3 priority target) since eps^(1/3) ~ 6e-6; a higher multiplicity or a larger block needs a larger
  /// im_tol (~ eps^(1/m), e.g. eps^(1/4) ~ 1.2e-4 for a 4-fold root) or it is reported complex.
  /// ASYMMETRY: because the tolerance is relative, a GENUINE complex pair whose |Im| is below
  /// im_tol * scale is reported real BY DESIGN (e.g. 1e8 +- i at the default, or any pair below im_tol
  /// when |Re| < 1 and the floor pins the threshold to im_tol). For an absolute test, read max_im.
  /// NON-CONVERGENCE => false: the Gershgorin fallback sets max_im = 0 by CONVENTION (nothing computed),
  /// never read as a real spectrum. Any non-finite bound, non-finite tolerance, negative imaginary
  /// magnitude, or negative tolerance also returns false. This matters for partitioned providers: a
  /// NaN 1x1 block must invalidate the complete spectrum instead of disappearing in min/max comparisons.
  /// POPS_HD, no allocation (only std::isfinite, std::fabs and comparisons, like the rest of this header).
  POPS_HD bool all_real(Real im_tol = kEigImagTol) const {
    return real_status(im_tol) == Real(1);
  }
  /// Complement of all_real RESTRICTED to converged blocks: true iff the spectrum was computed AND
  /// carries a complex conjugate pair beyond @p im_tol. NON-CONVERGENCE => false (NOT a complex signal:
  /// nothing was computed -- tell it apart via converged, or via pops::real_spectrum's kUnknown).
  POPS_HD bool has_complex_pair(Real im_tol = kEigImagTol) const {
    return valid() && !all_real(im_tol);
  }
};

/// Tri-state classification of a small block's spectrum (ADC-276), returned by pops::real_spectrum.
/// kUnknown is the NON-CONVERGENCE outcome and is NEVER kReal: a switch over it that omits kUnknown is
/// a visible bug, which is the point -- the caller (e.g. a native realizability projector) must treat a
/// non-converged block conservatively, not assume a real spectrum.
enum class Spectrum : int { kReal = 0, kComplexPair = 1, kUnknown = 2 };

namespace detail {

/// Gershgorin bound on the REAL PARTS: every lambda of the spectrum satisfies
/// lo <= Re(lambda) <= hi (disks centered at a_ii of radius the sum of the |off-diagonal| terms of
/// the row). Safe external bound for HLL, attained only if the matrix is diagonal.
template <int N>
POPS_HD inline void gershgorin_bounds(const Real (&A)[N][N], Real& lo, Real& hi) {
  for (int i = 0; i < N; ++i) {
    Real r = Real(0);
    for (int j = 0; j < N; ++j)
      if (j != i)
        r += std::fabs(A[i][j]);
    const Real l = A[i][i] - r, h = A[i][i] + r;
    if (i == 0 || l < lo)
      lo = l;
    if (i == 0 || h > hi)
      hi = h;
  }
}

/// Upper Hessenberg reduction by Householder reflections, IN PLACE, without accumulating the
/// transformations (eigenvalues only). Unconditionally stable.
template <int N>
POPS_HD inline void hessenberg_reduce(Real (&H)[N][N]) {
  Real v[N];  // Householder vector of the current step (components k..N-1)
  for (int k = 1; k <= N - 2; ++k) {
    Real scale = Real(0);
    for (int i = k; i < N; ++i)
      scale += std::fabs(H[i][k - 1]);
    if (scale == Real(0))
      continue;  // column already zero below the subdiagonal
    Real h = Real(0);
    for (int i = k; i < N; ++i) {
      v[i] = H[i][k - 1] / scale;
      h += v[i] * v[i];
    }
    Real g = std::sqrt(h);
    if (v[k] > Real(0))
      g = -g;
    h -= v[k] * g;  // h = v.v / 2 after updating v[k]
    v[k] -= g;
    if (h == Real(0))
      continue;
    // P = I - v v^T / h; H <- P H P (column k-1 set explicitly, exact zeros below)
    for (int j = k; j < N; ++j) {  // P * H on rows k..N-1
      Real f = Real(0);
      for (int i = k; i < N; ++i)
        f += v[i] * H[i][j];
      f /= h;
      for (int i = k; i < N; ++i)
        H[i][j] -= f * v[i];
    }
    for (int i = 0; i < N; ++i) {  // H * P on columns k..N-1
      Real f = Real(0);
      for (int j = k; j < N; ++j)
        f += H[i][j] * v[j];
      f /= h;
      for (int j = k; j < N; ++j)
        H[i][j] -= f * v[j];
    }
    H[k][k - 1] = scale * g;
    for (int i = k + 1; i < N; ++i)
      H[i][k - 1] = Real(0);
  }
}

POPS_HD inline Real hqr_copysign(Real mag, Real sgn) {
  return sgn >= Real(0) ? std::fabs(mag) : -std::fabs(mag);
}

/// Accumulate an eigenvalue (re, im) into the current extremes. A named function rather than a
/// local lambda: device caution (nvcc and lambdas inside __host__ __device__ code).
POPS_HD inline void record_eig(Real re, Real im, Real& lmin, Real& lmax, Real& max_im,
                               bool& first) {
  if (first || re < lmin)
    lmin = re;
  if (first || re > lmax)
    lmax = re;
  const Real ai = std::fabs(im);
  if (first || ai > max_im)
    max_im = ai;
  first = false;
}

/// QR iteration with implicit Francis double shift on a Hessenberg matrix (EISPACK/hqr
/// formulation, eigenvalues only, blocks processed bottom-up with deflation). Accumulates min/max
/// of the real parts and max|Im| directly. @return true if the WHOLE spectrum is extracted under
/// the cap (@p max_iter_per_eig iterations per active block), false otherwise.
template <int N>
POPS_HD inline bool hqr_minmax(Real (&H)[N][N], Real& lmin, Real& lmax, Real& max_im,
                               int max_iter_per_eig) {
  constexpr Real kEps = std::numeric_limits<Real>::epsilon();  // follows the Real type
  Real anorm = Real(0);  // norm of the Hessenberg part (deflation criterion for the s == 0 cases)
  for (int i = 0; i < N; ++i)
    for (int j = (i > 0 ? i - 1 : 0); j < N; ++j)
      anorm += std::fabs(H[i][j]);
  if (anorm == Real(0)) {  // null matrix: spectrum {0}
    lmin = lmax = max_im = Real(0);
    return true;
  }

  bool first = true;

  int nn = N - 1;    // top index of the active block
  Real t = Real(0);  // cumulative shift (exceptional shifts)
  Real p = Real(0), q = Real(0), r = Real(0), x, y, z, w, s;
  while (nn >= 0) {
    int its = 0;
    int l;
    do {
      // deflation: smallest l such that H[l][l-1] is negligible
      for (l = nn; l >= 1; --l) {
        s = std::fabs(H[l - 1][l - 1]) + std::fabs(H[l][l]);
        if (s == Real(0))
          s = anorm;
        if (std::fabs(H[l][l - 1]) <= kEps * s) {
          H[l][l - 1] = Real(0);
          break;
        }
      }
      x = H[nn][nn];
      if (l == nn) {  // real 1x1 eigenvalue
        record_eig(x + t, Real(0), lmin, lmax, max_im, first);
        --nn;
      } else {
        y = H[nn - 1][nn - 1];
        w = H[nn][nn - 1] * H[nn - 1][nn];
        if (l == nn - 1) {  // 2x2 block: real pair or complex conjugate pair
          p = Real(0.5) * (y - x);
          q = p * p + w;
          z = std::sqrt(std::fabs(q));
          x += t;
          if (q >= Real(0)) {  // two real values
            z = p + hqr_copysign(z, p);
            record_eig(x + z, Real(0), lmin, lmax, max_im, first);
            record_eig(z != Real(0) ? x - w / z : x + z, Real(0), lmin, lmax, max_im, first);
          } else {  // complex pair: Re = x + p, |Im| = z
            record_eig(x + p, z, lmin, lmax, max_im, first);
            record_eig(x + p, -z, lmin, lmax, max_im, first);
          }
          nn -= 2;
        } else {  // block > 2: one Francis double-shift iteration
          if (its == max_iter_per_eig)
            return false;                // cap reached -> caller fallback
          if (its == 10 || its == 20) {  // exceptional shift (slow cycles)
            t += x;
            for (int i = 0; i <= nn; ++i)
              H[i][i] -= x;
            s = std::fabs(H[nn][nn - 1]) + std::fabs(H[nn - 1][nn - 2]);
            y = x = Real(0.75) * s;
            w = Real(-0.4375) * s * s;
          }
          ++its;
          int m;
          for (m = nn - 2; m >= l; --m) {  // two consecutive small subdiagonals
            z = H[m][m];
            r = x - z;
            s = y - z;
            p = (r * s - w) / H[m + 1][m] + H[m][m + 1];
            q = H[m + 1][m + 1] - z - r - s;
            r = H[m + 2][m + 1];
            s = std::fabs(p) + std::fabs(q) + std::fabs(r);
            p /= s;
            q /= s;
            r /= s;
            if (m == l)
              break;
            const Real u = std::fabs(H[m][m - 1]) * (std::fabs(q) + std::fabs(r));
            const Real v = std::fabs(p) *
                           (std::fabs(H[m - 1][m - 1]) + std::fabs(z) + std::fabs(H[m + 1][m + 1]));
            if (u <= kEps * v)
              break;
          }
          for (int i = m + 2; i <= nn; ++i) {
            H[i][i - 2] = Real(0);
            if (i > m + 2)
              H[i][i - 3] = Real(0);
          }
          for (int k = m; k <= nn - 1; ++k) {  // QR double-shift sweep on columns m..nn-1
            if (k != m) {
              p = H[k][k - 1];
              q = H[k + 1][k - 1];
              r = (k != nn - 1) ? H[k + 2][k - 1] : Real(0);
              x = std::fabs(p) + std::fabs(q) + std::fabs(r);
              if (x != Real(0)) {
                p /= x;
                q /= x;
                r /= x;
              }
            }
            s = hqr_copysign(std::sqrt(p * p + q * q + r * r), p);
            if (s == Real(0))
              continue;
            if (k == m) {
              if (l != m)
                H[k][k - 1] = -H[k][k - 1];
            } else {
              H[k][k - 1] = -s * x;
            }
            p += s;
            x = p / s;
            y = q / s;
            z = r / s;
            q /= p;
            r /= p;
            for (int j = k; j <= nn; ++j) {  // transform rows k..k+2
              p = H[k][j] + q * H[k + 1][j];
              if (k != nn - 1) {
                p += r * H[k + 2][j];
                H[k + 2][j] -= p * z;
              }
              H[k + 1][j] -= p * y;
              H[k][j] -= p * x;
            }
            const int mmin = (nn < k + 3) ? nn : k + 3;
            for (int i = l; i <= mmin; ++i) {  // transform columns k..k+2
              p = x * H[i][k] + y * H[i][k + 1];
              if (k != nn - 1) {
                p += z * H[i][k + 2];
                H[i][k + 2] -= p * r;
              }
              H[i][k + 1] -= p * q;
              H[i][k] -= p;
            }
          }
        }
      }
    } while (l < nn - 1);
  }
  return true;
}

}  // namespace detail

/// Extremes of the REAL PARTS of the spectrum of a small dense block @p A, plus the largest |Im|
/// encountered and a convergence indicator (see the file header for the full contract: Gershgorin
/// fallback on non-convergence, max_im as a hyperbolicity-loss detector).
/// @p max_iter_per_eig: QR iteration cap per active block (default 100). The historical EISPACK
/// heuristic (30) does not suffice on near-degenerate companion blocks (near-double eigenvalues)
/// where deflation crawls: such a 5x5 block needs ~42 iterations, and below 30 it silently fell
/// back (wave speed over-estimated ~9x). 100 leaves more than double the margin; the overhead is
/// paid ONLY by pathological blocks (healthy cases converge in a few iterations). 0 forces the
/// fallback AS SOON AS an active block >= 3 exists (useful for testing the caller's contract); a
/// matrix that deflates entirely into 1x1 / 2x2 blocks (quasi-triangular) never iterates and
/// converges even at cap 0.
/// @p fallback: if non-null, receives true when the Gershgorin fallback triggered (spectrum NOT
/// computed), false otherwise. Default nullptr -> behavior unchanged for any existing caller;
/// mirror of !EigBounds::converged, for whoever wants only the flag (e.g. OR over several blocks).
template <int N>
POPS_HD inline EigBounds real_eig_minmax(const Real (&A)[N][N], int max_iter_per_eig = 100,
                                         bool* fallback = nullptr) {
  static_assert(N >= 1, "real_eig_minmax: N >= 1");
  static_assert(N <= 16,
                "real_eig_minmax: block limited to 16x16 (stack buffer O(N^2) per device "
                "thread, ~2 KB; beyond that, a dense solver with allocation is more "
                "appropriate than this path)");
  EigBounds b{Real(0), Real(0), Real(0), true};
  if constexpr (N == 1) {
    b.lmin = b.lmax = A[0][0];
  } else if constexpr (N == 2) {  // closed form: trace / determinant
    // Scale before the discriminant: the eigenvalues of a representable matrix can be finite even
    // when (a-d)^2 or b*c overflows (and conversely can remain representable in the subnormal range).
    Real scale = std::fabs(A[0][0]);
    const Real abs01 = std::fabs(A[0][1]);
    const Real abs10 = std::fabs(A[1][0]);
    const Real abs11 = std::fabs(A[1][1]);
    if (abs01 > scale)
      scale = abs01;
    if (abs10 > scale)
      scale = abs10;
    if (abs11 > scale)
      scale = abs11;
    if (scale == Real(0)) {
      b.lmin = b.lmax = Real(0);
    } else if (!std::isfinite(static_cast<double>(scale))) {
      b.lmin = b.lmax = b.max_im = std::numeric_limits<Real>::quiet_NaN();
    } else {
      const Real a = A[0][0] / scale;
      const Real d = A[1][1] / scale;
      const Real tr2 = Real(0.5) * a + Real(0.5) * d;
      const Real half_diff = Real(0.5) * a - Real(0.5) * d;
      const Real disc = half_diff * half_diff + (A[0][1] / scale) * (A[1][0] / scale);
      if (disc >= Real(0)) {
        const Real z = std::sqrt(disc);
        b.lmin = (tr2 - z) * scale;
        b.lmax = (tr2 + z) * scale;
      } else {
        b.lmin = b.lmax = tr2 * scale;
        b.max_im = std::sqrt(-disc) * scale;
      }
    }
  } else {
    Real H[N][N];  // working copy (A is not modified)
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        H[i][j] = A[i][j];
    detail::hessenberg_reduce(H);
    if (!detail::hqr_minmax(H, b.lmin, b.lmax, b.max_im, max_iter_per_eig)) {
      // non-convergence: external Gershgorin bound on the ORIGINAL matrix (the working copy is in
      // an intermediate state) -- safe bound, not the spectrum. max_im forced to 0 by CONVENTION
      // (nothing was computed), never to be interpreted as a real spectrum.
      detail::gershgorin_bounds(A, b.lmin, b.lmax);
      b.max_im = Real(0);
      b.converged = false;
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
      record_fallback(FallbackCounter::kDenseEigGershgorin);
#endif
    }
  }
  if (fallback)
    *fallback = !b.converged;
  return b;
}

/// Classify the spectrum of a small dense block @p A as kReal / kComplexPair / kUnknown (ADC-276): a
/// GENERIC, device-safe predicate over the SAME Francis-QR path as real_eig_minmax (no second
/// algorithm to keep in sync). @p im_tol is the RELATIVE imaginary tolerance of EigBounds::all_real
/// (default 1e-5, scaled by max(|lmin|, |lmax|, 1); covers a real multiplicity up to m=3, the 3x3
/// target -- see EigBounds::all_real for the tolerance contract, multiplicity coverage, and the
/// relative-tolerance asymmetry). NON-CONVERGENCE under @p max_iter_per_eig returns kUnknown BEFORE any
/// max_im read, so the Gershgorin fallback's max_im = 0 convention can never be mistaken for a real
/// spectrum. Intended consumer: a native realizability / hyperbolicity check on small Jacobian or
/// small dense moment blocks with no NumPy and no host callback; the core stays free of any model
/// specifics. Need the extremes too? Call real_eig_minmax and use the EigBounds
/// predicates directly.
template <int N>
POPS_HD inline Spectrum real_spectrum(const Real (&A)[N][N], Real im_tol = kEigImagTol,
                                      int max_iter_per_eig = 100) {
  const EigBounds b = real_eig_minmax(A, max_iter_per_eig);
  if (!b.valid() || !std::isfinite(static_cast<double>(im_tol)) || im_tol < Real(0))
    return Spectrum::kUnknown;  // invalid/non-converged is EXPLICIT, never real or complex
  return b.all_real(im_tol) ? Spectrum::kReal : Spectrum::kComplexPair;
}

namespace detail {

/// Inverse of a dense N x N matrix by Gauss-Jordan elimination with partial pivoting, into @p inv.
/// Returns false (inv untouched-meaningful) if a pivot falls below @p pivot_tol (singular). Device
/// clean: fixed stack buffers, bounded loops, no allocation.
template <int N>
POPS_HD inline bool mat_inverse(const Real (&A)[N][N], Real (&inv)[N][N],
                                Real pivot_tol = Real(1e-300)) {
  Real M[N][N];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      M[i][j] = A[i][j];
      inv[i][j] = (i == j) ? Real(1) : Real(0);
    }
  for (int col = 0; col < N; ++col) {
    int piv = col;
    Real best = std::fabs(M[col][col]);
    for (int r = col + 1; r < N; ++r) {
      const Real v = std::fabs(M[r][col]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < pivot_tol)
      return false;
    if (piv != col)
      for (int j = 0; j < N; ++j) {
        Real t = M[col][j];
        M[col][j] = M[piv][j];
        M[piv][j] = t;
        t = inv[col][j];
        inv[col][j] = inv[piv][j];
        inv[piv][j] = t;
      }
    const Real invd = Real(1) / M[col][col];
    for (int j = 0; j < N; ++j) {
      M[col][j] *= invd;
      inv[col][j] *= invd;
    }
    for (int r = 0; r < N; ++r) {
      if (r == col)
        continue;
      const Real f = M[r][col];
      if (f != Real(0))
        for (int j = 0; j < N; ++j) {
          M[r][j] -= f * M[col][j];
          inv[r][j] -= f * inv[col][j];
        }
    }
  }
  return true;
}

/// Max absolute row sum (infinity norm) of a dense N x N matrix.
template <int N>
POPS_HD inline Real mat_norm_inf(const Real (&A)[N][N]) {
  const Real real_max = std::numeric_limits<Real>::max();
  Real m = Real(0);
  for (int i = 0; i < N; ++i) {
    Real r = Real(0);
    for (int j = 0; j < N; ++j) {
      const Real v = std::fabs(A[i][j]);
      // Saturation keeps this scaling norm finite even when the mathematical row sum exceeds the
      // largest Real.  The Newton balance only needs a positive magnitude estimate; overflowing
      // to infinity would instead poison sqrt(norm(inv)) / sqrt(norm(S)).
      r = v <= real_max - r ? r + v : real_max;
    }
    if (r > m)
      m = r;
  }
  return m;
}

/// Scaled Newton matrix sign of A + shift I.  The shift is part of the spectral function, not a
/// diagonal regularisation: callers use it to construct exact spectral projectors.  Returns false
/// when the shifted matrix is singular or the bounded iteration does not converge.  S is only
/// meaningful on success.
template <int N>
POPS_HD inline bool matrix_sign_shifted(const Real (&A)[N][N], Real shift, Real (&S)[N][N],
                                        int max_iter, Real tol) {
  const Real real_max = std::numeric_limits<Real>::max();
  Real input_scale = std::fabs(shift);
  if (!(input_scale <= real_max))
    return false;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      const Real v = std::fabs(A[i][j]);
      if (!(v <= real_max))
        return false;
      if (v > input_scale)
        input_scale = v;
    }
  if (!(input_scale > Real(0)))
    return false;

  Real Sinv[N][N];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      // sign(alpha B) == sign(B) for alpha > 0.  Forming the shifted matrix after this
      // normalisation avoids overflow in A_ii + shift and makes tiny, otherwise subnormal,
      // matrices visible to the fixed absolute pivot threshold in mat_inverse.
      S[i][j] = A[i][j] / input_scale +
                (i == j ? shift / input_scale : Real(0));
  bool converged = false;
  for (int it = 0; it < max_iter; ++it) {
    if (!mat_inverse(S, Sinv))
      return false;
    const Real ns = mat_norm_inf(S), nsi = mat_norm_inf(Sinv);
    Real mu = Real(1);
    if (ns > Real(0) && nsi > Real(0)) {
      // The ratio nsi/ns can overflow (or underflow) even when its square root is representable.
      mu = std::sqrt(nsi) / std::sqrt(ns);
      if (!(mu > Real(0)) || !(mu <= real_max))
        return false;
    }
    const Real a = Real(0.5) * mu, b = Real(0.5) / mu;
    Real diff = Real(0), nrm = Real(0);
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        const Real snext = a * S[i][j] + b * Sinv[i][j];
        const Real d = snext - S[i][j];
        const Real abs_next = std::fabs(snext), abs_diff = std::fabs(d);
        if (!(abs_next <= real_max) || !(abs_diff <= real_max))
          return false;
        if (abs_diff > diff)
          diff = abs_diff;
        if (abs_next > nrm)
          nrm = abs_next;
        S[i][j] = snext;
      }
    if (nrm > Real(0) && diff <= tol * nrm) {
      converged = true;
      break;
    }
  }
  return converged;
}

template <int N>
POPS_HD inline void matvec(const Real (&A)[N][N], const Real (&x)[N], Real (&out)[N]) {
  for (int i = 0; i < N; ++i) {
    Real value = Real(0);
    for (int j = 0; j < N; ++j)
      value += A[i][j] * x[j];
    out[i] = value;
  }
}

/// Arithmetic mean without first forming a+b (overflow) or a/2 and b/2 (underflow).  The
/// same-sign branch interpolates from the smaller magnitude; opposite signs can be added safely.
POPS_HD inline Real safe_average(Real a, Real b) {
  if ((a > Real(0) && b > Real(0)) || (a < Real(0) && b < Real(0)))
    return std::fabs(a) < std::fabs(b) ? a + Real(0.5) * (b - a)
                                      : b + Real(0.5) * (a - b);
  return Real(0.5) * (a + b);
}

/// Actions of sign(A + cutoff I) and sign(A - cutoff I) on x.  A representable outward (or, near
/// max(Real), inward) retry handles the measure-zero case in which an eigenvalue lies exactly on a
/// cutoff; both Roe spectral functions using this helper are continuous at that boundary.
template <int N>
POPS_HD inline bool shifted_sign_actions(const Real (&A)[N][N], const Real (&x)[N], Real cutoff,
                                         Real (&plus_x)[N], Real (&minus_x)[N], int max_iter,
                                         Real tol) {
  Real S[N][N];
  const Real boundary_step = std::sqrt(std::numeric_limits<Real>::epsilon());
  for (int attempt = 0; attempt < 4; ++attempt) {
    Real selected = cutoff;
    if (attempt != 0) {
      const Real relative_step = Real(attempt) * boundary_step;
      const Real relative_increment = cutoff * relative_step;
      const Real increment = relative_increment > std::numeric_limits<Real>::denorm_min()
                                 ? relative_increment
                                 : std::numeric_limits<Real>::denorm_min();
      // Outward is preferable, but cannot be represented above a cutoff near max(Real).  An
      // inward retry is equally valid at the boundary because the Harten branches agree there.
      selected = cutoff <= std::numeric_limits<Real>::max() - increment
                     ? cutoff + increment
                     : cutoff - increment;
    }
    if (!matrix_sign_shifted(A, selected, S, max_iter, tol))
      continue;
    matvec(S, x, plus_x);
    if (!matrix_sign_shifted(A, -selected, S, max_iter, tol))
      continue;
    matvec(S, x, minus_x);
    return true;
  }
  return false;
}

/// Apply the Roe matrix absolute value after an external provider has certified that A has a real
/// spectrum.  This helper deliberately performs no eigensolve: a block-triangular provider may
/// certify the full spectrum from smaller diagonal blocks, while the spectral function must still
/// act on the complete matrix.  Failure of either bounded matrix-sign path remains explicit; no
/// alternate Riemann solver or spectral enclosure is selected.
template <int N>
POPS_HD inline bool roe_abs_apply_certified_real(const Real (&A)[N][N], const Real (&dU)[N],
                                                 Real (&out)[N], int max_iter = 80,
                                                 Real tol = Real(1e-13)) {
  static_assert(N >= 1 && N <= 16, "roe_abs_apply_certified_real: 1 <= N <= 16");
  Real S[N][N];
  Real SdU[N];
  if (matrix_sign_shifted(A, Real(0), S, max_iter, tol)) {
    matvec(S, dU, SdU);
  } else {
    // sign(0) is undefined, but A sign(A) is not: its zero-mode value is exactly zero.  Build the
    // positive/negative resolved projector around a scale-relative numerical nullspace.
    const Real scale = mat_norm_inf(A);
    if (scale == Real(0)) {
      for (int i = 0; i < N; ++i)
        out[i] = Real(0);
      return true;
    }
    const Real cutoff = Real(64) * std::numeric_limits<Real>::epsilon() * scale;
    if (!(cutoff > Real(0)))
      return false;
    Real plus_du[N], minus_du[N];
    if (!shifted_sign_actions(A, dU, cutoff, plus_du, minus_du, max_iter, tol))
      return false;
    for (int i = 0; i < N; ++i)
      SdU[i] = Real(0.5) * (plus_du[i] + minus_du[i]);
  }
  // |A| dU = A sign(A) dU; the regularized zero projector above defines sign(0)=0.
  for (int i = 0; i < N; ++i) {
    Real value = Real(0);
    for (int j = 0; j < N; ++j)
      value += A[i][j] * SdU[j];
    out[i] = value;
  }
  return true;
}

/// Apply the Harten entropy-fixed Roe spectral function after external real-spectrum
/// certification.  As above, the complete matrix is retained and every numerical failure is
/// returned to the caller; certification is never replaced by a fallback.
template <int N>
POPS_HD inline bool roe_entropy_fix_apply_certified_real(
    const Real (&A)[N][N], const Real (&dU)[N], Real (&out)[N], Real delta,
    int max_iter = 80, Real tol = Real(1e-13)) {
  static_assert(N >= 1 && N <= 16,
                "roe_entropy_fix_apply_certified_real: 1 <= N <= 16");
  if (!(delta > Real(0)) || !(delta <= std::numeric_limits<Real>::max()))
    return false;

  Real plus_du[N], minus_du[N];
  if (!shifted_sign_actions(A, dU, delta, plus_du, minus_du, max_iter, tol))
    return false;

  Real outside_sign_du[N], inside_du[N], A_inside_du[N], scaled_A_inside_du[N];
  for (int i = 0; i < N; ++i) {
    outside_sign_du[i] = Real(0.5) * (plus_du[i] + minus_du[i]);
    inside_du[i] = Real(0.5) * (plus_du[i] - minus_du[i]);
  }
  matvec(A, inside_du, A_inside_du);
  for (int i = 0; i < N; ++i)
    scaled_A_inside_du[i] = A_inside_du[i] / delta;
  for (int i = 0; i < N; ++i) {
    Real outside = Real(0), scaled_inside_quadratic = Real(0);
    for (int j = 0; j < N; ++j) {
      outside += A[i][j] * outside_sign_du[j];
      // Evaluate A^2/delta as A*(A/delta): neither delta^2 nor the intermediate A^2 needs to
      // exist in Real when the final Harten value itself is finite.
      scaled_inside_quadratic += A[i][j] * scaled_A_inside_du[j];
    }
    // Factor delta only after forming the dimensionless average.  This preserves both
    // Phi_delta(delta)=delta at denorm_min and finite O(delta) values near max(Real).
    const Real inside =
        delta * safe_average(scaled_inside_quadratic / delta, inside_du[i]);
    out[i] = outside + inside;
  }
  return true;
}

}  // namespace detail

/// Roe matrix-absolute-value applied to a state jump: out = |A| dU, with |A| the SPECTRAL absolute
/// value A * sign(A). sign(A) is computed by the determinant-free, infinity-norm-SCALED Newton
/// matrix-sign iteration S_{k+1} = 1/2 (mu S_k + 1/mu S_k^-1), mu = sqrt(||S^-1||/||S||), which
/// converges quadratically for a real spectrum off the imaginary axis. For a real-diagonalizable A
/// this is EXACTLY R |Lambda| R^-1 dU.  The configurable Harten function used by
/// flux_ROE_local.m is implemented separately by roe_entropy_fix_apply below.
///
/// Returns false (out untouched) when |A| is not a faithful real spectral function:
///   - the spectrum is not real (real_spectrum != kReal): A * sign(A) would keep the sign-of-real-part
///     of a complex eigenvalue, NOT its modulus, so it would diverge from the reference;
///   - the real spectral solve or the bounded matrix-sign iterations do not converge.
/// A singular REAL A is handled without changing solver: after the exact sign iteration detects the
/// zero mode, shifted projectors define sign(0)=0 at a relative numerical-rank threshold and return
/// the same Roe matrix absolute value on every resolved nonzero mode.
/// POPS_HD, no allocation, N <= 16 (the dense-eig stack-buffer limit).
template <int N>
POPS_HD inline bool roe_abs_apply(const Real (&A)[N][N], const Real (&dU)[N], Real (&out)[N],
                                  int max_iter = 80, Real tol = Real(1e-13),
                                  Real im_tol = kEigStrictImagTol,
                                  int max_iter_per_eig = 100) {
  // The physical default accepts only the bounded QR round-off floor.  An explicit zero remains
  // available to callers that want an exact-zero diagnostic rather than this physical default.
  static_assert(N >= 1 && N <= 16, "roe_abs_apply: 1 <= N <= 16");
  if (real_spectrum(A, im_tol, max_iter_per_eig) != Spectrum::kReal)
    return false;
  return detail::roe_abs_apply_certified_real(A, dU, out, max_iter, tol);
}

/// Harten entropy-fixed Roe spectral function applied to a state jump:
///
///   Phi_delta(lambda) = |lambda|                         for |lambda| >= delta
///                     = (lambda^2 / delta + delta) / 2   for |lambda| < delta.
///
/// For a real-diagonalizable A, the two shifted matrix signs construct the three invariant
/// spectral projectors without eigenvectors:
///
///   P_inside = (sign(A + delta I) - sign(A - delta I)) / 2,
///   P_pos - P_neg = (sign(A + delta I) + sign(A - delta I)) / 2.
///
/// Consequently the returned vector is exactly Phi_delta(A) dU.  Unlike roe_abs_apply, a zero
/// eigenvalue is regular: its dissipation is delta/2.  Complex and non-converged spectra return
/// false before touching out.  If an eigenvalue lies exactly on +/-delta, the two formula branches
/// agree to first order; a representable near-boundary retry avoids an undefined shifted sign while
/// preserving the rounded spectral value.
template <int N>
POPS_HD inline bool roe_entropy_fix_apply(const Real (&A)[N][N], const Real (&dU)[N],
                                          Real (&out)[N], Real delta, int max_iter = 80,
                                          Real tol = Real(1e-13),
                                          Real im_tol = kEigStrictImagTol,
                                          int max_iter_per_eig = 100) {
  static_assert(N >= 1 && N <= 16, "roe_entropy_fix_apply: 1 <= N <= 16");
  if (!(delta > Real(0)) || !(delta <= std::numeric_limits<Real>::max()))
    return false;
  if (real_spectrum(A, im_tol, max_iter_per_eig) != Spectrum::kReal)
    return false;
  return detail::roe_entropy_fix_apply_certified_real(A, dU, out, delta, max_iter, tol);
}

}  // namespace pops
