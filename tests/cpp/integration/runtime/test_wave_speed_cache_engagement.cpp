// Cache des vitesses d'onde HLL (opt-in) : PREUVE D'ENGAGEMENT cote coeur C++. Le test Python
// (tests/python/unit/physics/test_wave_speed_cache.py) verifie ON == OFF + les gardes, mais sa bit-identite
// reussirait TRIVIALEMENT si le chemin cache devenait un no-op silencieux (repli par face). Ce test
// instrumente model.wave_speeds par un COMPTEUR et exerce le chemin HLL OFF puis ON :
//   (1) bit-exactitude : les vitesses sont calculees sur les traces reconstruites exactes, donc
//       cache ON == OFF a 0 ulp avec NoSlope, Minmod et WENO5 ;
//   (2) engagement      : chaque face partagee est evaluee une seule fois dans la pre-passe ->
//       calls_on < calls_off STRICTEMENT. Si le cache devenait un no-op, ce test echouerait.
// Header-only (pops::pops seul), aucun modele physique : le compteur vit dans une Kokkos::View
// device-accessible (atomic), portable Serial / OpenMP / Cuda.

#include <gtest/gtest.h>

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/numerics/spatial_operator.hpp>
#include <pops/runtime/builders/block/block_builder.hpp>

#include <Kokkos_Core.hpp>  // Kokkos::View / atomic_add / deep_copy (compteur d'appels device-accessible)

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

using namespace pops;
static constexpr double kPi = 3.14159265358979323846;

// Compteur d'appels a wave_speeds : Kokkos::View<long long> (memoire de l'espace d'execution par
// defaut, accessible dans le kernel device). atomic_increment -> correct sous Serial/OpenMP/Cuda.
using Counter = Kokkos::View<long long>;

// Modele isotherme 3-var (rho, mx, my), valide hyperbolique, dont wave_speeds INCREMENTE un compteur
// (et porte un cout factice DETERMINISTE busy >= 0 pour le volet timing, sans alterer lo/hi). Sert a
// PROUVER que le cache reduit le nombre d'appels (engagement) et reste bit-exact.
struct CountingIsothermal {
  static constexpr int n_vars = 3;
  using State = StateVec<3>;
  using Aux = pops::Aux;
  Real c0 = Real(1);
  int busy = 0;
  Counter calls;  // handle capture par valeur dans le kernel (donnees partagees)

  POPS_HD State flux(const State& u, const Aux&, int dir) const {
    const Real rho = u[0];
    const Real vx = u[1] / rho, vy = u[2] / rho;
    const Real p = c0 * c0 * rho;
    State F{};
    if (dir == 0) {
      F[0] = u[1];
      F[1] = u[1] * vx + p;
      F[2] = u[2] * vx;
    } else {
      F[0] = u[2];
      F[1] = u[1] * vy;
      F[2] = u[2] * vy + p;
    }
    return F;
  }
  POPS_HD Real max_wave_speed(const State& u, const Aux&, int dir) const {
    const Real v = (dir == 0 ? u[1] : u[2]) / u[0];
    const Real av = v < 0 ? -v : v;
    return av + c0;
  }
  POPS_HD void wave_speeds(const State& u, const Aux&, int dir, Real& lo, Real& hi) const {
    Kokkos::atomic_add(&calls(), 1LL);
    const Real v = (dir == 0 ? u[1] : u[2]) / u[0];
    Real acc = Real(0);
    for (int k = 0; k < busy; ++k)
      acc += std::sin(v + Real(k)) * std::cos(v - Real(k));
    const Real c = c0 + (acc - acc);  // acc-acc == 0 exact : lo/hi bit-stables quel que soit busy
    lo = v - c;
    hi = v + c;
  }
  POPS_HD State source(const State&, const Aux&) const { return State{}; }
};

static void init_state(MultiFab& U, const Geometry& geom, const Box2D& dom) {
  Array4 a = U.fab(0).array();
  for_each_cell(dom, [a, geom](int i, int j) {
    const double x = geom.x_cell(i), y = geom.y_cell(j);
    const double rho = 1.0 + 0.3 * std::sin(2 * kPi * x) * std::cos(2 * kPi * y);
    a(i, j, 0) = rho;
    a(i, j, 1) = 0.2 * rho * std::sin(2 * kPi * x);
    a(i, j, 2) = -0.15 * rho * std::cos(2 * kPi * y);
  });
}

// Compte les bits qui different entre deux MultiFab sur la boite valide (memcmp par valeur double).
static long long count_diff_bits(const MultiFab& A, const MultiFab& B, const Box2D& dom) {
  long long ndiff = 0;
  const ConstArray4 a = A.fab(0).const_array(), b = B.fab(0).const_array();
  for (int c = 0; c < 3; ++c)
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int i = dom.lo[0]; i <= dom.hi[0]; ++i) {
        const double va = a(i, j, c), vb = b(i, j, c);
        if (std::memcmp(&va, &vb, sizeof(double)) != 0)
          ++ndiff;
      }
  return ndiff;
}

static long long read_counter(const Counter& c) {
  device_fence();
  auto h = Kokkos::create_mirror_view(c);
  Kokkos::deep_copy(h, c);
  return h();
}

template <class Limiter>
static void expect_reconstructed_face_cache_exact_and_engaged(const char* reconstruction_name) {
  const int n = 24;
  const double L = 1.0;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, L, 0.0, L};
  const BoxArray ba = BoxArray::from_domain(dom, n);
  const DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;
  MultiFab state(ba, dm, 3, Limiter::n_ghost);
  MultiFab aux(ba, dm, 3, 1);
  MultiFab residual_direct(ba, dm, 3, 0), residual_cached(ba, dm, 3, 0);
  MultiFab cache(ba, dm, 4, 1);
  const Box2D xfaces = xface_box(dom), yfaces = yface_box(dom);
  const BoxArray xba(std::vector<Box2D>{xfaces}), yba(std::vector<Box2D>{yfaces});
  MultiFab flux_x_direct(xba, dm, 3, 0), flux_y_direct(yba, dm, 3, 0);
  MultiFab flux_x_cached(xba, dm, 3, 0), flux_y_cached(yba, dm, 3, 0);
  aux.set_val(0.0);
  init_state(state, geom, dom);
  fill_ghosts(state, geom.domain, bc);

  Counter calls("ws_calls_reconstructed");
  const CountingIsothermal model{Real(1), /*busy=*/0, calls};
  Kokkos::deep_copy(calls, 0LL);
  assemble_rhs<Limiter, HLLFlux>(model, state, aux, geom, residual_direct);
  const long long calls_direct = read_counter(calls);

  Kokkos::deep_copy(calls, 0LL);
  assemble_rhs_hll_cached<Limiter>(model, state, aux, geom, residual_cached, cache);
  const long long calls_cached = read_counter(calls);

  device_fence();
  EXPECT_EQ(count_diff_bits(residual_direct, residual_cached, dom), 0)
      << reconstruction_name
      << ": exact face-trace cache must preserve every residual bit";
  EXPECT_LT(calls_cached, calls_direct)
      << reconstruction_name << ": shared faces must reduce wave_speeds calls (direct="
      << calls_direct << ", cached=" << calls_cached << ")";
  EXPECT_GT(calls_cached, 0) << reconstruction_name << ": cache pre-pass really evaluated waves";

  compute_face_fluxes<Limiter, HLLFlux>(model, state, aux, flux_x_direct, flux_y_direct,
                                        geom.dx(), geom.dy());
  compute_face_fluxes_hll_cached<Limiter>(model, state, aux, flux_x_cached, flux_y_cached, cache,
                                          geom.dx(), geom.dy());
  device_fence();
  EXPECT_EQ(count_diff_bits(flux_x_direct, flux_x_cached, xfaces), 0)
      << reconstruction_name << ": x-face materialization must consume the exact cached interval";
  EXPECT_EQ(count_diff_bits(flux_y_direct, flux_y_cached, yfaces), 0)
      << reconstruction_name << ": y-face materialization must consume the exact cached interval";
}

TEST(WaveSpeedCacheEngagement, MusclCacheMatchesExactReconstructedTraces) {
  expect_reconstructed_face_cache_exact_and_engaged<Minmod>("Minmod");
}

TEST(WaveSpeedCacheEngagement, Weno5CacheMatchesExactReconstructedTraces) {
  expect_reconstructed_face_cache_exact_and_engaged<Weno5>("WENO5-Z");
}

// (1) bit-exactitude + (2) engagement (comptage), wave_speeds bon marche.
TEST(WaveSpeedCacheEngagement, CacheIsBitExactAndCallsWaveSpeedsFewerTimes) {
  const int n = 48;
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique
  MultiFab aux(ba, dm, 3,
               1);  // construit AVANT la View : initialise Kokkos via l'allocateur unifie
  aux.set_val(0.0);
  const GridContext ctx{dom, bc, geom, &aux};

  Counter calls("ws_calls");
  CountingIsothermal model{Real(1), /*busy=*/0, calls};
  BlockClosures off = make_block(model, "none", "hll", ctx, false, false, "explicit", {}, {},
                                 nullptr, Real(0), /*wave_speed_cache=*/false);
  BlockClosures on = make_block(model, "none", "hll", ctx, false, false, "explicit", {}, {},
                                nullptr, Real(0), /*wave_speed_cache=*/true);

  MultiFab Uoff(ba, dm, 3, 1), Uon(ba, dm, 3, 1), U0(ba, dm, 3, 1);
  init_state(Uoff, geom, dom);
  init_state(Uon, geom, dom);
  init_state(U0, geom, dom);

  const double dt = 0.2 * (L / n) / 2.0;  // CFL prudente (max |v|+c ~ 1.4)
  const int nsteps = 25;

  Kokkos::deep_copy(calls, 0LL);
  for (int s = 0; s < nsteps; ++s)
    off.advance(Uoff, dt, 1);
  const long long calls_off = read_counter(calls);

  Kokkos::deep_copy(calls, 0LL);
  for (int s = 0; s < nsteps; ++s)
    on.advance(Uon, dt, 1);
  const long long calls_on = read_counter(calls);

  device_fence();
  const long long ndiff = count_diff_bits(Uoff, Uon, dom);
  const long long evolved = count_diff_bits(Uoff, U0, dom);
  EXPECT_TRUE(evolved > 0) << "l'etat a reellement evolue (test non creux)";
  EXPECT_TRUE(ndiff == 0) << "bit-exact NoSlope+HLL : cache ON == OFF (0 ulp), ndiff_bits="
                          << ndiff;
  // PREUVE D'ENGAGEMENT : la pre-passe calcule chaque face partagee une seule fois, contrairement au
  // residu direct qui la re-evalue depuis chacune de ses deux cellules adjacentes.
  EXPECT_TRUE(calls_on < calls_off)
      << "cache ENGAGE : moins d'appels wave_speeds que le chemin par face (OFF=" << calls_off
      << " ON=" << calls_on << ")";
  EXPECT_TRUE(calls_off > 0 && calls_on > 0) << "les deux chemins evaluent reellement wave_speeds";
}

// (3) gain de temps (mesure diagnostique) : wave_speeds couteux reste bit-exact ON == OFF. Le temps
// n'est pas asserte (bruit de mesure) ; seule la bit-exactitude l'est.
TEST(WaveSpeedCacheEngagement, CostlyWaveSpeedsStaysBitExact) {
  const int n = 48;
  const double L = 1.0;
  Box2D dom = Box2D::from_extents(n, n);
  Geometry geom{dom, 0.0, L, 0.0, L};
  BoxArray ba = BoxArray::from_domain(dom, n);
  DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;  // periodique
  MultiFab aux(ba, dm, 3, 1);
  aux.set_val(0.0);
  const GridContext ctx{dom, bc, geom, &aux};

  Counter calls("ws_calls_costly");
  CountingIsothermal model{Real(1), /*busy=*/100, calls};  // emule moments + factorisations
  BlockClosures off = make_block(model, "none", "hll", ctx, false, false, "explicit", {}, {},
                                 nullptr, Real(0), false);
  BlockClosures on = make_block(model, "none", "hll", ctx, false, false, "explicit", {}, {},
                                nullptr, Real(0), true);
  MultiFab Uoff(ba, dm, 3, 1), Uon(ba, dm, 3, 1);
  init_state(Uoff, geom, dom);
  init_state(Uon, geom, dom);
  const double dt = 0.2 * (L / n) / 2.0;
  const int nsteps = 10;

  auto t0 = std::chrono::steady_clock::now();
  for (int s = 0; s < nsteps; ++s)
    off.advance(Uoff, dt, 1);
  device_fence();
  auto t1 = std::chrono::steady_clock::now();
  for (int s = 0; s < nsteps; ++s)
    on.advance(Uon, dt, 1);
  device_fence();
  auto t2 = std::chrono::steady_clock::now();
  const double ms_off = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double ms_on = std::chrono::duration<double, std::milli>(t2 - t1).count();
  const long long ndiff = count_diff_bits(Uoff, Uon, dom);
  EXPECT_TRUE(ndiff == 0) << "bit-exact (cas wave_speeds couteux) : cache ON == OFF ; OFF="
                          << ms_off << " ms ON=" << ms_on
                          << " ms speedup=" << (ms_on > 0 ? ms_off / ms_on : 0.0) << "x";
}
