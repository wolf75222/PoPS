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
//   (4) a late rejected field solve rolls back every warm start and every published aux component,
//       while an unregistered field name is rejected loud.
//
// Engine-level (AmrRuntime + dispatch_amr_block): no DSL / .so compile (the production AMR loader is
// Kokkos-gated). The named field's aux output components (>= kAuxNamedBase) need a wide shared aux
// channel; a native ExB block carries no named aux, so we widen the block's aux_ncomp and attach the
// per-field RHS closure by hand -- exactly what the native loader does (register_elliptic_field +
// set_block_elliptic_field), minus the DSL codegen. The loader-side emission is covered by the Python
// test_time_multielliptic Section A.

#include <gtest/gtest.h>

#include <pops/coupling/base/elliptic_rhs.hpp>  // add_scaled_component (per-field RHS closure)
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // detail::make_shared_amr_layout / dispatch_amr_block
#include <pops/runtime/amr/amr_runtime.hpp>                  // AmrRuntime, AmrRuntimeBlock
#include <pops/runtime/builders/factory/model_factory.hpp>  // detail::dispatch_model
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/core/state/state.hpp>       // kAuxNamedBase
#include <pops/mesh/storage/mf_arith.hpp>  // norm_inf
#include <pops/mesh/storage/multifab.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

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
  const double mean = 1.0 + s / (static_cast<double>(n) * n);
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

static Real max_abs_diff(const MultiFab& lhs, const MultiFab& rhs) {
  device_fence();
  Real result = Real(0);
  for (int li = 0; li < lhs.local_size(); ++li) {
    const ConstArray4 left = lhs.fab(li).const_array();
    const ConstArray4 right = rhs.fab(li).const_array();
    const Box2D grown = lhs.fab(li).grown_box();
    for (int component = 0; component < lhs.ncomp(); ++component)
      for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
        for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
          result = std::max(
              result, std::fabs(left(i, j, component) - right(i, j, component)));
  }
  return result;
}

static Real max_abs_component_diff(const MultiFab& lhs, const MultiFab& rhs, int component) {
  device_fence();
  Real result = Real(0);
  for (int li = 0; li < lhs.local_size(); ++li) {
    const ConstArray4 left = lhs.fab(li).const_array();
    const ConstArray4 right = rhs.fab(li).const_array();
    const Box2D grown = lhs.fab(li).grown_box();
    for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
      for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
        result = std::max(result, std::fabs(left(i, j, component) - right(i, j, component)));
  }
  return result;
}

static std::vector<double> valid_values(const MultiFab& field, int n) {
  device_fence();
  std::vector<double> values(static_cast<std::size_t>(n) * n, 0.0);
  for (int li = 0; li < field.local_size(); ++li) {
    const ConstArray4 source = field.fab(li).const_array();
    const Box2D valid = field.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        values[static_cast<std::size_t>(j) * n + i] = source(i, j);
  }
  return values;
}

static void add_valid_constant(MultiFab& field, Real value) {
  device_fence();
  for (int li = 0; li < field.local_size(); ++li) {
    Array4 destination = field.fab(li).array();
    const Box2D valid = field.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        destination(i, j, 0) += value;
  }
}

TEST(test_amr_named_field, Runs) {
#if defined(POPS_HAS_KOKKOS)
  int argc = 0;
  char** argv = nullptr;
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int N = 64;
  const double L = 1.0, B0 = 1.0, q = -1.0;
  const std::vector<double> rho = blob(N, 0.5);

  // --- single ExB block on a frozen one-level shared hierarchy (default Poisson f = q*rho) ---
  AmrBuildParams bp;
  bp.mesh.n = N;
  bp.mesh.L = L;
  bp.mesh.regrid_every = 0;
  bp.poisson.bc = BCRec{};  // periodic
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
  const int kPhiFail = kAuxNamedBase + 4;  // 9 (forced late failure, phi only)
  blocks[0].aux_ncomp = kPhiFail + 1;      // reserve up to comp 9

  AmrRuntime rt(S.geom, S.runtime_hierarchy(), S.poisson_bc, std::move(blocks), S.base_per,
                S.replicated_coarse, S.wall);
  EXPECT_EQ(rt.n_blocks(), 1) << "named_engine_one_block";

  const SolveReport default_report = rt.solve_default_field();
  ASSERT_TRUE(default_report.solved())
      << "status=" << default_report.status_name() << " action=" << default_report.action_name()
      << " iters=" << default_report.iters
      << " rel_residual=" << default_report.rel_residual;
  EXPECT_GT(default_report.iters, 0) << "default field returns its real GeometricMG report";

  // Default Poisson REFERENCE: read the accepted warm start without launching another relative solve.
  const std::vector<double> phi_default = rt.level_potential(0);
  const double phi_def_mean = mean_of(phi_default);
  double phi_def_span = 0;
  for (double v : phi_default)
    phi_def_span = std::fmax(phi_def_span, std::fabs(v - phi_def_mean));
  EXPECT_GT(phi_def_span, 1e-6) << "named_default_phi_nontrivial";

  // (1) PARITY: named field "psi" with RHS = q*rho (the SAME as the default Poisson). gradient comps
  // declared. The closure mirrors make_poisson_rhs of a charge brick: rhs += q * U[0].
  AmrFieldSolveConfig psi_plan;
  psi_plan.provider_identity = "test:plasma/psi";
  psi_plan.topology_provider_kind = "structured";
  psi_plan.topology_provenance = "test:periodic-cartesian";
  psi_plan.topology_digest = "test:periodic-cartesian:v1";
  psi_plan.output_owner_identity = "test:plasma";
  psi_plan.output_block = "plasma";
  psi_plan.output_key = "psi";
  psi_plan.nullspace_assertion = "constant";
  psi_plan.gauge = "mean_zero";
  psi_plan.providers.push_back(
      FieldProviderBinding{"test:plasma/psi/rhs", "plasma", "psi", Real(1)});
  rt.install_field_plan("psi", psi_plan);
  rt.register_named_field("plasma", "psi", kPhiPsi, kGxPsi, kGyPsi);
  rt.set_block_named_elliptic_rhs(0, "psi", [q](const MultiFab& U, MultiFab& rhs) {
    add_scaled_component(U, Real(q), 0, rhs);  // f_psi = q * rho == default Poisson RHS
  });
  EXPECT_TRUE(rt.has_named_field("psi") && rt.n_named_fields() == 1) << "named_psi_registered";

  const std::string psi_field = "psi";
  ASSERT_TRUE(rt.solve_named_fields(&psi_field).solved());
  const std::vector<double> psi = valid_values(rt.provider_potential("psi"), N);
  EXPECT_EQ(static_cast<int>(psi.size()), N * N) << "named_psi_shape_nxn";
  bool psi_finite = true;
  for (double v : psi)
    psi_finite = psi_finite && std::isfinite(v);
  EXPECT_TRUE(psi_finite) << "named_psi_finite";

  // psi == default phi to the MG tolerance, after recentering on the periodic additive constant (same
  // operator, same RHS, same coarse box -> the only gap is the iterative MG rel_tol).
  const double psi_mean = mean_of(psi);
  double dmax_par = 0, ref_par = 0;
  for (int k = 0; k < N * N; ++k) {
    dmax_par =
        std::fmax(dmax_par, std::fabs((psi[k] - psi_mean) - (phi_default[k] - phi_def_mean)));
    ref_par = std::fmax(ref_par, std::fabs(phi_default[k] - phi_def_mean));
  }
  EXPECT_GT(ref_par, 1e-6) << "named_parity_oracle_nontrivial";
  EXPECT_LT(dmax_par, 1e-2 * (ref_par + 1e-12))
      << "named psi (RHS=q*rho) tracks the default Poisson potential within iterative accuracy";

  // (2) DISTINCT RHS (linearity): named field "chi" with RHS = 2*q*rho -> chi = 2*psi (Poisson linear).
  // A genuinely different, correctly scaled second field (not an alias of the default phi).
  AmrFieldSolveConfig chi_plan;
  chi_plan.provider_identity = "test:plasma/chi";
  chi_plan.topology_provider_kind = "structured";
  chi_plan.topology_provenance = "test:periodic-cartesian";
  chi_plan.topology_digest = "test:periodic-cartesian:v1";
  chi_plan.output_owner_identity = "test:plasma";
  chi_plan.output_block = "plasma";
  chi_plan.output_key = "chi";
  chi_plan.nullspace_assertion = "constant";
  chi_plan.gauge = "mean_zero";
  chi_plan.providers.push_back(
      FieldProviderBinding{"test:plasma/chi/rhs", "plasma", "chi", Real(1)});
  rt.install_field_plan("chi", chi_plan);
  rt.register_named_field("plasma", "chi", kPhiChi, /*gx=*/-1,
                          /*gy=*/-1);  // phi only (fewer than 3 aux slots)
  rt.set_block_named_elliptic_rhs(0, "chi", [q](const MultiFab& U, MultiFab& rhs) {
    add_scaled_component(U, Real(2.0 * q), 0, rhs);  // f_chi = 2 * (q * rho)
  });
  EXPECT_EQ(rt.n_named_fields(), 2) << "named_chi_registered";

  const MultiFab aux_before_selected_chi = rt.aux(0);
  const std::string chi_field = "chi";
  ASSERT_TRUE(rt.solve_named_fields(&chi_field).solved());
  for (const int untouched_component : {0, 1, 2, kPhiPsi, kGxPsi, kGyPsi})
    EXPECT_EQ(max_abs_component_diff(rt.aux(0), aux_before_selected_chi, untouched_component),
              Real(0))
        << "selected named publication preserves unrelated valid cells and ghosts";
  const std::vector<double> chi = valid_values(rt.provider_potential("chi"), N);
  const double chi_mean = mean_of(chi);
  const std::vector<double> psi2 = valid_values(rt.provider_potential("psi"), N);
  const double psi2_mean = mean_of(psi2);
  double dmax_lin = 0, ref_lin = 0;
  for (int k = 0; k < N * N; ++k) {
    // chi - chi_mean should equal 2 * (psi - psi_mean).
    dmax_lin = std::fmax(dmax_lin, std::fabs((chi[k] - chi_mean) - 2.0 * (psi2[k] - psi2_mean)));
    ref_lin = std::fmax(ref_lin, std::fabs(chi[k] - chi_mean));
  }
  EXPECT_GT(ref_lin, 1e-6) << "named_chi_nontrivial";
  EXPECT_LT(dmax_lin, 1e-3 * (ref_lin + 1e-12))
      << "named chi (RHS=2*q*rho) == 2 * psi (linearity: genuinely distinct scaled field)";

  // (3) NO REGRESSION: selected named solves do not mutate or snapshot the default warm start.
  const std::vector<double> phi_after = rt.level_potential(0);
  const double phi_after_mean = mean_of(phi_after);
  double dmax_def = 0;
  for (int k = 0; k < N * N; ++k)
    dmax_def = std::fmax(
        dmax_def, std::fabs((phi_after[k] - phi_after_mean) - (phi_default[k] - phi_def_mean)));
  EXPECT_LT(dmax_def, 1e-3 * (phi_def_span + 1e-12))
      << "named registration leaves the default potential() unchanged to the MG tolerance";

  // AmrProgramContext must forward the true default-field rejection, not replace it with a fabricated
  // cache success. A constant offset makes the periodic RHS deliberately incompatible.
  MultiFab& live_state = rt.level_state(0, 0);
  MultiFab accepted_state = live_state;
  const MultiFab context_phi_before = rt.phi();
  std::vector<MultiFab> context_aux_before;
  for (int level = 0; level < rt.nlev(); ++level)
    context_aux_before.push_back(rt.aux(level));
  add_valid_constant(live_state, Real(1));
  runtime::program::AmrProgramContext context(&rt, nullptr);
  context.reset_step();
  context.set_level(0);
  std::string context_diagnostic;
  try {
    (void)context.solve_fields();
    FAIL() << "periodic default RHS with non-zero mean was accepted or silently projected";
  } catch (const std::runtime_error& error) {
    context_diagnostic = error.what();
  }
  EXPECT_NE(context_diagnostic.find("incompatible with nullspace"), std::string::npos)
      << context_diagnostic;
  EXPECT_NE(context_diagnostic.find("silent projection is forbidden"), std::string::npos)
      << context_diagnostic;
  EXPECT_EQ(max_abs_diff(rt.phi(), context_phi_before), Real(0));
  for (int level = 0; level < rt.nlev(); ++level)
    EXPECT_EQ(max_abs_diff(rt.aux(level),
                           context_aux_before[static_cast<std::size_t>(level)]),
              Real(0));
  live_state = std::move(accepted_state);

  // an unregistered field name is rejected loud (never a silent zero field).
  EXPECT_THROW(rt.named_field_values("nope"), std::runtime_error) << "named_unknown_field_rejected";

  // A late named-field failure must roll back the complete solve set: the default warm start, every
  // already-solved named warm start, and every aux level. The failing field is ordered after chi and
  // psi, so the attempt has already advanced the complete previously published solve set before it
  // fails.
  rt.phi().set_val(Real(0));
  rt.provider_potential("psi").set_val(Real(0));
  rt.provider_potential("chi").set_val(Real(0));
  const MultiFab default_before = rt.phi();
  const MultiFab psi_before = rt.provider_potential("psi");
  const MultiFab chi_before = rt.provider_potential("chi");
  std::vector<MultiFab> aux_before;
  for (int level = 0; level < rt.nlev(); ++level)
    aux_before.push_back(rt.aux(level));

  AmrFieldSolveConfig fail_plan;
  fail_plan.provider_identity = "test:plasma/zeta";
  fail_plan.topology_provider_kind = "structured";
  fail_plan.topology_provenance = "test:periodic-cartesian";
  fail_plan.topology_digest = "test:periodic-cartesian:v1";
  fail_plan.output_owner_identity = "test:plasma";
  fail_plan.output_block = "plasma";
  fail_plan.output_key = "zeta";
  fail_plan.nullspace_assertion = "constant";
  fail_plan.gauge = "mean_zero";
  fail_plan.mg_opts.rel_tol = Real(1e-30);
  fail_plan.mg_opts.max_cycles = 1;
  fail_plan.providers.push_back(
      FieldProviderBinding{"test:plasma/zeta/rhs", "plasma", "zeta", Real(1)});
  rt.install_field_plan("zeta", fail_plan);
  rt.register_named_field("plasma", "zeta", kPhiFail, /*gx=*/-1, /*gy=*/-1);
  rt.set_block_named_elliptic_rhs(0, "zeta", [q](const MultiFab& U, MultiFab& rhs) {
    add_scaled_component(U, Real(q), 0, rhs);
  });

  const SolveReport failed = rt.solve_fields();
  EXPECT_EQ(failed.status, SolveStatus::kIterationLimit);
  EXPECT_EQ(failed.action, SolveAction::kRejectAttempt);
  EXPECT_EQ(failed.iters, 1);
  EXPECT_EQ(max_abs_diff(rt.phi(), default_before), Real(0));
  EXPECT_EQ(max_abs_diff(rt.provider_potential("psi"), psi_before), Real(0));
  EXPECT_EQ(max_abs_diff(rt.provider_potential("chi"), chi_before), Real(0));
  for (int level = 0; level < rt.nlev(); ++level)
    EXPECT_EQ(max_abs_diff(rt.aux(level), aux_before[static_cast<std::size_t>(level)]), Real(0));
  EXPECT_EQ(norm_inf(rt.provider_potential("zeta")), Real(0))
      << "a solver allocated by the rejected attempt does not retain its partial iterate";

  // Bootstrap recomputation cannot materialize/cache a field from a rejected solve report.
  rt.begin_bootstrap_plan();
  EXPECT_THROW(rt.recompute_bootstrap_field("zeta"), std::runtime_error);
  rt.rollback_bootstrap_level();
}
