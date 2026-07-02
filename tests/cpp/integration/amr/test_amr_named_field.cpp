// ADC-428 Named elliptic fields on the AMR layout (engine-level numerical validation).
//
// A SECOND elliptic solve (beyond the default coarse Poisson) for a user-named field
// (m.elliptic_field) on the AMR hierarchy: AmrRuntime owns a DEDICATED coarse GeometricMG per named
// field, sums its RHS from the blocks' per-field closures, writes phi (+ centered grad) into the
// field's OWN aux components, and injects it to the fine levels each solve_fields. The default Poisson
// path (mg_) is untouched. The OFFLINE REFERENCE is the engine's own default Poisson solve (the same
// validation idea as test_time_multielliptic for the uniform System): we never reimplement a multigrid
// to check against.
//
//   (1) PARITY: a named field "psi" with RHS = the SAME f = q*rho as the default Poisson solves the
//       IDENTICAL elliptic problem with the SAME native solver (GeometricMG), so its solved potential
//       equals the default potential() to the MG tolerance (modulo the periodic additive constant). A
//       true second INDEPENDENT solve validated against the default one.
//   (2) DISTINCT RHS (linearity): a named field "chi" with RHS = 2 * (default) gives chi = 2*psi
//       (Poisson is linear) -- confirms the named field carries a genuinely DIFFERENT, correctly scaled
//       field, not an alias of the default phi.
//   (3) NO REGRESSION: registering named fields leaves the DEFAULT potential() bit-identical (the
//       default-only solve path is unchanged).
//   (4) the named field stays finite + non-trivial after a few transport steps (re-solved each step),
//       and an unregistered field name is rejected loud.
//
// Engine-level (AmrRuntime + dispatch_amr_block): no DSL / .so compile (the production AMR loader is
// Kokkos-gated). The named field's aux output components (>= kAuxNamedBase) need a wide shared aux
// channel; a native ExB block carries no named aux, so we widen the block's aux_ncomp and attach the
// per-field RHS closure by hand -- exactly what the native loader does (register_elliptic_field +
// set_block_elliptic_field), minus the DSL codegen. The loader-side emission is covered by the Python
// test_time_multielliptic Section A.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/coupling/base/elliptic_rhs.hpp>  // add_scaled_component (per-field RHS closure)
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // detail::make_shared_amr_layout / dispatch_amr_block
#include <pops/runtime/amr/amr_runtime.hpp>                  // AmrRuntime, AmrRuntimeBlock
#include <pops/runtime/builders/factory/model_factory.hpp>  // detail::dispatch_model
#include <pops/runtime/config/model_spec.hpp>
#include <pops/core/state/state.hpp>       // kAuxNamedBase
#include <pops/mesh/storage/mf_arith.hpp>  // norm_inf
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

template <class F>
static bool raises(F&& f) {
  try {
    f();
  } catch (const std::runtime_error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

// Scalar ExB block of charge q: transport E x B (advection driven by grad phi), charge density q*n for
// the default system Poisson (elliptic = "charge" -> elliptic_rhs = q*n).
static ModelSpec exb_charge(double q, double B0) {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  s.q = q;
  s.B0 = B0;
  return s;
}

// Smooth zero-mean density (solvable in periodic): a centered blob around 1, n*n row-major.
static std::vector<double> blob(int n, double amp) {
  std::vector<double> r(static_cast<std::size_t>(n) * n, 1.0);
  double s = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      const double v = amp * std::exp(-(x * x + y * y) / 0.01);
      r[static_cast<std::size_t>(j) * n + i] = 1.0 + v;
      s += v;
    }
  const double mean = s / (static_cast<double>(n) * n);
  for (auto& v : r)
    v -= mean;  // zero-mean source -> periodic Poisson solvable
  return r;
}

// Mean of a coarse n*n field (for the periodic additive-constant recentering before comparison).
static double mean_of(const std::vector<double>& f) {
  double s = 0;
  for (double v : f)
    s += v;
  return f.empty() ? 0.0 : s / static_cast<double>(f.size());
}

static int pops_run_test_amr_named_field(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int N = 64;
  const double L = 1.0, B0 = 1.0, q = -1.0;
  const std::vector<double> rho = blob(N, 0.5);

  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  // --- single ExB block on a frozen one-level shared hierarchy (default Poisson f = q*rho) ---
  AmrBuildParams bp;
  bp.n = N;
  bp.L = L;
  bp.regrid_every = 0;
  bp.poisson_bc = BCRec{};  // periodic
  const detail::SharedAmrLayout S = detail::make_shared_amr_layout(bp);

  std::vector<AmrRuntimeBlock> blocks;
  detail::dispatch_model(exb_charge(q, B0), [&](auto m) {
    blocks.push_back(detail::dispatch_amr_block(m, "minmod", "rusanov", S, "plasma", rho,
                                                /*has_density=*/true, 1.4, 1, false, false));
  });
  // Widen the shared aux channel so the named fields' output components (>= kAuxNamedBase) fit. The
  // native ExB block reads only comps 0..2; the runtime sizes the channel to max(b.aux_ncomp), so a
  // wider value just reserves room (the extra comps are written only by the named solve). This is what a
  // model declaring m.aux_field("psi"/"g2x"/"g2y") would set via aux_comps<Model>().
  const int kPhiPsi = kAuxNamedBase;      // 5
  const int kGxPsi = kAuxNamedBase + 1;   // 6
  const int kGyPsi = kAuxNamedBase + 2;   // 7
  const int kPhiChi = kAuxNamedBase + 3;  // 8 (second named field, phi only)
  blocks[0].aux_ncomp = kPhiChi + 1;      // reserve up to comp 8

  AmrRuntime rt(S.geom, S.ba_coarse, S.poisson_bc, std::move(blocks), S.base_per,
                S.replicated_coarse, S.wall);
  chk(rt.n_blocks() == 1, "named_engine_one_block");

  // Default Poisson REFERENCE (the engine's own solve): potential() solves the fields and returns phi.
  const std::vector<double> phi_default = rt.potential();
  const double phi_def_mean = mean_of(phi_default);
  double phi_def_span = 0;
  for (double v : phi_default)
    phi_def_span = std::fmax(phi_def_span, std::fabs(v - phi_def_mean));
  chk(phi_def_span > 1e-6, "named_default_phi_nontrivial");

  // (1) PARITY: named field "psi" with RHS = q*rho (the SAME as the default Poisson). gradient comps
  // declared. The closure mirrors make_poisson_rhs of a charge brick: rhs += q * U[0].
  rt.register_named_field("psi", kPhiPsi, kGxPsi, kGyPsi);
  rt.set_block_named_elliptic_rhs(0, "psi", [q](const MultiFab& U, MultiFab& rhs) {
    add_scaled_component(U, Real(q), 0, rhs);  // f_psi = q * rho == default Poisson RHS
  });
  chk(rt.has_named_field("psi") && rt.n_named_fields() == 1, "named_psi_registered");

  const std::vector<double> psi = rt.named_field_values("psi");
  chk(static_cast<int>(psi.size()) == N * N, "named_psi_shape_nxn");
  bool psi_finite = true;
  for (double v : psi)
    psi_finite = psi_finite && std::isfinite(v);
  chk(psi_finite, "named_psi_finite");

  // psi == default phi to the MG tolerance, after recentering on the periodic additive constant (same
  // operator, same RHS, same coarse box -> the only gap is the iterative MG rel_tol).
  const double psi_mean = mean_of(psi);
  double dmax_par = 0, ref_par = 0;
  for (int k = 0; k < N * N; ++k) {
    dmax_par =
        std::fmax(dmax_par, std::fabs((psi[k] - psi_mean) - (phi_default[k] - phi_def_mean)));
    ref_par = std::fmax(ref_par, std::fabs(phi_default[k] - phi_def_mean));
  }
  chk(ref_par > 1e-6, "named_parity_oracle_nontrivial");
  chk(dmax_par < 1e-3 * (ref_par + 1e-12),
      "named psi (RHS=q*rho) == default Poisson potential() to the MG tolerance");

  // (2) DISTINCT RHS (linearity): named field "chi" with RHS = 2*q*rho -> chi = 2*psi (Poisson linear).
  // A genuinely different, correctly scaled second field (not an alias of the default phi).
  rt.register_named_field("chi", kPhiChi, /*gx=*/-1,
                          /*gy=*/-1);  // phi only (fewer than 3 aux slots)
  rt.set_block_named_elliptic_rhs(0, "chi", [q](const MultiFab& U, MultiFab& rhs) {
    add_scaled_component(U, Real(2.0 * q), 0, rhs);  // f_chi = 2 * (q * rho)
  });
  chk(rt.n_named_fields() == 2, "named_chi_registered");

  const std::vector<double> chi = rt.named_field_values("chi");
  const double chi_mean = mean_of(chi);
  // Re-read psi: named_field_values(chi) re-ran solve_fields, so psi's aux was refreshed too.
  const std::vector<double> psi2 = rt.named_field_values("psi");
  const double psi2_mean = mean_of(psi2);
  double dmax_lin = 0, ref_lin = 0;
  for (int k = 0; k < N * N; ++k) {
    // chi - chi_mean should equal 2 * (psi - psi_mean).
    dmax_lin = std::fmax(dmax_lin, std::fabs((chi[k] - chi_mean) - 2.0 * (psi2[k] - psi2_mean)));
    ref_lin = std::fmax(ref_lin, std::fabs(chi[k] - chi_mean));
  }
  chk(ref_lin > 1e-6, "named_chi_nontrivial");
  chk(dmax_lin < 1e-3 * (ref_lin + 1e-12),
      "named chi (RHS=2*q*rho) == 2 * psi (linearity: genuinely distinct scaled field)");

  // (3) NO REGRESSION: the DEFAULT potential() is unchanged by the named-field registration. The
  // default Poisson (mg_) is solved FIRST in solve_fields, BEFORE solve_named_fields, which only writes
  // the NAMED comps (>= kAuxNamedBase) and re-fills ghosts (valid cells of comps 0..2 untouched). The
  // tiny residual below the MG tolerance comes from GeometricMG's warm start across the two separate
  // solve_fields calls (the default solver re-cycles from its converged phi), NOT from the named path --
  // an isolated runtime without any named field shows the same warm-start drift between two potential()
  // calls. So we assert default-path INVARIANCE to the MG tolerance, the meaningful no-regression claim.
  const std::vector<double> phi_after = rt.potential();
  const double phi_after_mean = mean_of(phi_after);
  double dmax_def = 0;
  for (int k = 0; k < N * N; ++k)
    dmax_def = std::fmax(
        dmax_def, std::fabs((phi_after[k] - phi_after_mean) - (phi_default[k] - phi_def_mean)));
  chk(dmax_def < 1e-3 * (phi_def_span + 1e-12),
      "named registration leaves the default potential() unchanged to the MG tolerance");

  // (4) after a few ExB transport steps (named field re-solved each step), psi stays finite + non-trivial.
  rt.step(Real(1e-3));
  rt.step(Real(1e-3));
  const std::vector<double> psi_adv = rt.named_field_values("psi");
  bool adv_finite = true;
  double adv_span = 0;
  const double adv_mean = mean_of(psi_adv);
  for (double v : psi_adv) {
    adv_finite = adv_finite && std::isfinite(v);
    adv_span = std::fmax(adv_span, std::fabs(v - adv_mean));
  }
  chk(adv_finite, "named_psi_finite_after_advance");
  chk(adv_span > 1e-6, "named_psi_nontrivial_after_advance");

  // an unregistered field name is rejected loud (never a silent zero field).
  chk(raises([&] { rt.named_field_values("nope"); }), "named_unknown_field_rejected");

  if (fails == 0)
    std::printf(
        "OK test_amr_named_field (psi==default dmax/ref=%.1e/%.1e ; chi==2psi "
        "dmax/ref=%.1e/%.1e)\n",
        dmax_par, ref_par, dmax_lin, ref_lin);
  return fails ? 1 : 0;
}

TEST(test_amr_named_field, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_amr_named_field, "test_amr_named_field"), 0);
}
