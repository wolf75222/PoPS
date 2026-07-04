// Tests of pops::detail::block_inverse<N> (ADC-637), the pinned-order closed-form block inverse.
//
// Properties verified:
//   1. BIT-FOR-BIT parity with LorentzEliminator. For the rotation block M = [[1, -w], [w, 1]]
//      (w = theta*dt*B_z) the four block_inverse<2> entries MUST equal binv_11/12/21/22 EXACTLY
//      (operator==, not a tolerance). This is the load-bearing gate of the ADC-637 retirement plan:
//      the condensed-implicit codegen assembles A = I + c*rho*M^{-1} and must reproduce the retiring
//      Schur brick's coefficients to the last bit.
//   2. Round-trip M * M^{-1} == I to round-off for a general 2x2 and 3x3.
//   3. 3x3 vs the generic mat_inverse<N> agree to round-off (the closed form is a bit-stable fast
//      path, not a different result).
//   4. block_inverse<4> falls through to mat_inverse<4> (the >3 generic path).
//   5. Singular block: det ~ 0 returns false without writing inv.

#include <gtest/gtest.h>

#include <pops/numerics/linalg/block_inverse.hpp>
#include <pops/numerics/linalg/dense_eig.hpp>          // mat_inverse<N> (the 3x3 / >3 reference)
#include <pops/numerics/linalg/lorentz_eliminator.hpp>  // the bit-parity reference

#include <cmath>

using pops::LorentzEliminator;
using pops::Real;
using pops::detail::block_inverse;
using pops::detail::mat_inverse;

static inline double dabs(double x) { return x < 0.0 ? -x : x; }

static constexpr double EPS_MACHINE = 1e-14;

// The (theta, dt, B_z) cases mirror test_lorentz_eliminator: the same w = theta*dt*B_z sweep, from a
// weak field to a very large w, so the parity claim is exercised across the dynamic range.
struct BzCase {
  Real theta, dt, Bz;
  const char* name;
};
static const BzCase kBz[] = {
    {1.0, 0.1, 1.0, "theta=1 dt=0.1 Bz=1"},
    {0.5, 0.2, 2.5, "theta=0.5 dt=0.2 Bz=2.5"},
    {1.0, 1.0, 10.0, "strong w=10"},
    {0.75, 0.05, 0.01, "weak field small dt"},
    {1.0, 0.01, 1e6, "very large w"},
    {1.0, 0.1, 0.0, "degenerate Bz=0"},
};

// Test 1 (THE GATE): block_inverse<2> of the Lorentz rotation block == LorentzEliminator, BIT-EXACT.
//
// M = [[1, -w], [w, 1]] with w = th_dt*B_z (th_dt = theta*dt, and the eliminator is built as
// (theta=th_dt, dt=1, B_z) to obtain the same w, exactly as SchurOperatorCoeffKernel does). The
// closed 2x2 entries d/det, -b/det, -c/det, a/det reduce to 1/det, w/det, -w/det, 1/det -- the same
// four DIRECT divisions binv_11..22 return, with det = 1 - (-w)*w == 1 + w*w. == is intentional.
TEST(test_block_inverse, RotationBlockMatchesLorentzBitForBit) {
  for (const auto& c : kBz) {
    const Real th_dt = c.theta * c.dt;
    const LorentzEliminator le(th_dt, Real(1), c.Bz);
    const Real w = th_dt * c.Bz;
    const Real M[2][2] = {{Real(1), -w}, {w, Real(1)}};
    Real Mi[2][2];
    ASSERT_TRUE(block_inverse<2>(M, Mi)) << "non-singular rotation block [" << c.name << "]";
    // BIT-FOR-BIT: the pinned operation order fixes the tree, so this is exact, not a tolerance.
    EXPECT_EQ(Mi[0][0], le.binv_11()) << "binv_11 [" << c.name << "]";
    EXPECT_EQ(Mi[0][1], le.binv_12()) << "binv_12 [" << c.name << "]";
    EXPECT_EQ(Mi[1][0], le.binv_21()) << "binv_21 [" << c.name << "]";
    EXPECT_EQ(Mi[1][1], le.binv_22()) << "binv_22 [" << c.name << "]";
    // And the determinant reduces to LorentzEliminator's 1 + w*w bit-for-bit -- in the intrinsic's
    // PINNED shape (hoist a*d, subtract b*c). Written as one `a*d - b*c` expression the frontend may
    // fma-contract the WRONG product (clang contracts at EVERY -O level) and drift a ULP.
    const Real t = M[0][0] * M[1][1];
    const Real det = t - M[0][1] * M[1][0];
    EXPECT_EQ(det, le.det) << "det [" << c.name << "]";
  }
}

// Test 1b (the contraction sweep): the bit-parity holds on RUNTIME values, not only compile-time
// constants. A fixed-value table can pass by constant folding while the fma-contraction of the
// runtime code pairs a product differently (measured: ~7% of random w one ULP off before the det /
// bracket shapes were pinned); a pseudo-random volatile sweep defeats the folding.
TEST(test_block_inverse, RotationBlockMatchesLorentzOnRuntimeValues) {
  unsigned s = 12345u;
  auto next = [&s]() {  // xorshift; volatile write defeats constant propagation
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    volatile Real r = Real(s) / Real(4294967295.0) * Real(12) - Real(6);
    return Real(r);
  };
  for (int k = 0; k < 10000; ++k) {
    const Real w = next();
    const LorentzEliminator le(w, Real(1), Real(1));
    const Real M[2][2] = {{Real(1), -w}, {w, Real(1)}};
    Real Mi[2][2];
    ASSERT_TRUE(block_inverse<2>(M, Mi));
    ASSERT_EQ(Mi[0][0], le.binv_11()) << "w=" << w;
    ASSERT_EQ(Mi[0][1], le.binv_12()) << "w=" << w;
    ASSERT_EQ(Mi[1][0], le.binv_21()) << "w=" << w;
    ASSERT_EQ(Mi[1][1], le.binv_22()) << "w=" << w;
    const Real v[2] = {next(), next()};
    Real got[2];
    ASSERT_TRUE(pops::detail::block_apply_inverse<2>(M, v, got));
    Real ex, ey;
    le.apply_Binv(v[0], v[1], ex, ey);
    ASSERT_EQ(got[0], ex) << "apply x, w=" << w;
    ASSERT_EQ(got[1], ey) << "apply y, w=" << w;
  }
}

// Test 2 : round-trip M * M^{-1} == I to round-off for a general (non-rotation) 2x2.
TEST(test_block_inverse, RoundTrip2x2) {
  const Real M[2][2] = {{Real(2.0), Real(-1.5)}, {Real(0.7), Real(3.1)}};
  Real Mi[2][2];
  ASSERT_TRUE(block_inverse<2>(M, Mi));
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      Real p = Real(0);
      for (int k = 0; k < 2; ++k) p += M[i][k] * Mi[k][j];
      const Real want = (i == j) ? Real(1) : Real(0);
      EXPECT_TRUE(dabs(p - want) < EPS_MACHINE) << "M*Minv[" << i << "][" << j << "]";
    }
}

// Test 3 : 3x3 closed form -- round-trip to I AND agreement with the generic mat_inverse<3>.
TEST(test_block_inverse, ClosedForm3x3MatchesMatInverse) {
  const Real M[3][3] = {{Real(4.0), Real(1.0), Real(-2.0)},
                        {Real(0.5), Real(3.0), Real(1.5)},
                        {Real(-1.0), Real(2.0), Real(5.0)}};
  Real Mi[3][3];
  ASSERT_TRUE(block_inverse<3>(M, Mi));
  // Round-trip M * M^{-1} == I.
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      Real p = Real(0);
      for (int k = 0; k < 3; ++k) p += M[i][k] * Mi[k][j];
      const Real want = (i == j) ? Real(1) : Real(0);
      EXPECT_TRUE(dabs(p - want) < EPS_MACHINE) << "M*Minv[" << i << "][" << j << "]";
    }
  // Same result as the generic Gauss-Jordan inverse to round-off (a bit-stable fast path).
  Real Mg[3][3];
  ASSERT_TRUE(mat_inverse<3>(M, Mg));
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      const Real scale = Real(1) + dabs(Mi[i][j]);
      EXPECT_TRUE(dabs(Mi[i][j] - Mg[i][j]) < EPS_MACHINE * scale)
          << "closed vs mat_inverse [" << i << "][" << j << "]";
    }
}

// Test 3b : a 3D block-diagonal rotation ([[1,-w,0],[w,1,0],[0,0,1]]) inverts with the (0,1) sub-block
// matching the 2x2 Lorentz entries and the z axis untouched (the natural 3D-momentum instance).
TEST(test_block_inverse, BlockDiagonalRotation3x3) {
  const Real w = Real(0.3);
  const Real M[3][3] = {{Real(1), -w, Real(0)}, {w, Real(1), Real(0)}, {Real(0), Real(0), Real(1)}};
  Real Mi[3][3];
  ASSERT_TRUE(block_inverse<3>(M, Mi));
  const LorentzEliminator le(w, Real(1), Real(1));  // w = th_dt*B_z with th_dt=w, B_z=1
  EXPECT_TRUE(dabs(Mi[0][0] - le.binv_11()) < EPS_MACHINE);
  EXPECT_TRUE(dabs(Mi[0][1] - le.binv_12()) < EPS_MACHINE);
  EXPECT_TRUE(dabs(Mi[1][0] - le.binv_21()) < EPS_MACHINE);
  EXPECT_TRUE(dabs(Mi[1][1] - le.binv_22()) < EPS_MACHINE);
  EXPECT_TRUE(dabs(Mi[2][2] - Real(1)) < EPS_MACHINE) << "z axis identity";
  EXPECT_TRUE(dabs(Mi[0][2]) < EPS_MACHINE && dabs(Mi[2][0]) < EPS_MACHINE) << "no x-z coupling";
}

// Test 4 : block_inverse<4> falls through to mat_inverse<4> (the generic >3 path, same result).
TEST(test_block_inverse, N4FallsThroughToMatInverse) {
  const Real M[4][4] = {{Real(3), Real(1), Real(0), Real(2)},
                        {Real(0), Real(4), Real(1), Real(0)},
                        {Real(1), Real(0), Real(5), Real(1)},
                        {Real(0), Real(2), Real(0), Real(6)}};
  Real Mi[4][4];
  Real Mg[4][4];
  ASSERT_TRUE(block_inverse<4>(M, Mi));
  ASSERT_TRUE(mat_inverse<4>(M, Mg));
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_EQ(Mi[i][j], Mg[i][j]) << "fallthrough is mat_inverse [" << i << "][" << j << "]";
}

// Test 5 : a singular block returns false and does not write inv.
TEST(test_block_inverse, SingularReturnsFalse) {
  const Real M[2][2] = {{Real(1), Real(2)}, {Real(2), Real(4)}};  // det = 0
  Real Mi[2][2] = {{Real(-1), Real(-1)}, {Real(-1), Real(-1)}};   // sentinel
  EXPECT_FALSE(block_inverse<2>(M, Mi)) << "singular 2x2 -> false";
  EXPECT_EQ(Mi[0][0], Real(-1)) << "inv untouched on singular";
}

// Test 6 (THE APPLY GATE): block_apply_inverse<2> of the Lorentz rotation block applied to an arbitrary
// vector == LorentzEliminator::apply_Binv, BIT-EXACT. The condensed RHS-flux and reconstruct kernels
// apply M^{-1} to a VECTOR (F = M^{-1}(mx,my); v = M^{-1}(v^n - theta dt grad phi)); the retiring Schur
// brick applied it with apply_Binv = inv*(vx + w*vy) -- ONE reciprocal factored out of the bracket.
// block_apply_inverse reproduces that factored order; summing the pre-divided block_inverse<2> entries
// (1/det)*vx + (w/det)*vy would round differently (a per-step ULP drift off np.array_equal). == is the
// point.
TEST(test_block_inverse, ApplyInverseMatchesApplyBinvBitForBit) {
  using pops::detail::block_apply_inverse;
  // A spread of input vectors: axis-aligned, mixed sign, and large magnitude.
  const Real vs[][2] = {{Real(1), Real(0)},       {Real(0), Real(1)},   {Real(0.4), Real(-0.2)},
                        {Real(-3.7), Real(2.1)},  {Real(1e3), Real(-7)}};
  for (const auto& cc : kBz) {
    const Real th_dt = cc.theta * cc.dt;
    const LorentzEliminator le(th_dt, Real(1), cc.Bz);
    const Real w = th_dt * cc.Bz;
    const Real M[2][2] = {{Real(1), -w}, {w, Real(1)}};
    for (const auto& v : vs) {
      Real got[2];
      ASSERT_TRUE(block_apply_inverse<2>(M, v, got)) << "non-singular [" << cc.name << "]";
      Real wx, wy;
      le.apply_Binv(v[0], v[1], wx, wy);
      EXPECT_EQ(got[0], wx) << "apply_Binv x [" << cc.name << "] v=(" << v[0] << "," << v[1] << ")";
      EXPECT_EQ(got[1], wy) << "apply_Binv y [" << cc.name << "] v=(" << v[0] << "," << v[1] << ")";
    }
  }
}

// Test 7 : block_apply_inverse<3> of a general 3x3 equals inv * (adj . v) to round-off (round-trip via
// the block inverse), and a singular block returns false without touching out.
TEST(test_block_inverse, ApplyInverse3x3AndSingular) {
  using pops::detail::block_apply_inverse;
  const Real M[3][3] = {{Real(4.0), Real(1.0), Real(-2.0)},
                        {Real(0.5), Real(3.0), Real(1.5)},
                        {Real(-1.0), Real(2.0), Real(5.0)}};
  const Real v[3] = {Real(1.3), Real(-0.7), Real(2.2)};
  Real got[3];
  ASSERT_TRUE(block_apply_inverse<3>(M, v, got));
  Real Mi[3][3];
  ASSERT_TRUE(block_inverse<3>(M, Mi));
  for (int r = 0; r < 3; ++r) {
    Real ref = Real(0);
    for (int c = 0; c < 3; ++c) ref += Mi[r][c] * v[c];
    const Real scale = Real(1) + dabs(ref);
    EXPECT_TRUE(dabs(got[r] - ref) < EPS_MACHINE * scale) << "apply vs Minv.v [" << r << "]";
  }
  const Real S[3][3] = {{Real(1), Real(2), Real(3)}, {Real(2), Real(4), Real(6)}, {Real(0), Real(1), Real(1)}};
  Real out[3] = {Real(-9), Real(-9), Real(-9)};  // sentinel
  EXPECT_FALSE(block_apply_inverse<3>(S, v, out)) << "singular 3x3 -> false";
  EXPECT_EQ(out[0], Real(-9)) << "out untouched on singular";
}
