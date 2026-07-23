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
#include <pops/physics/bricks/bricks.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/core/state/state.hpp>       // kAuxNamedBase
#include <pops/mesh/layout/refinement.hpp>  // parallel_copy
#include <pops/mesh/storage/mf_arith.hpp>  // norm_inf
#include <pops/mesh/storage/multifab.hpp>

#include "amr_transfer_test_authority.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

static AmrFieldHierarchyPolicyAuthority level_local_hierarchy_policy() {
  return {
      "pops.field-hierarchy.level-local",
      1,
      {"pops.field-hierarchy.options.empty@1", {}},
  };
}

static AmrFieldHierarchyPolicyAuthority composite_hierarchy_policy() {
  return {
      "pops.field-hierarchy.composite",
      1,
      {"pops.field-hierarchy.options.empty@1", {}},
  };
}

static AmrFieldHierarchyPolicyAuthority external_graph_hierarchy_policy() {
  return {
      "tests.field-hierarchy.coupled-graph",
      3,
      {"tests.field-hierarchy.coupled-graph.options@3",
       {{"coupling_radius", std::uint64_t{2}}}},
  };
}

class ExternalGraphIdentityPrepared final : public AmrPreparedFieldSolver {
 public:
  ExternalGraphIdentityPrepared(const AmrFieldSolverBuildRequest& request, std::string contract)
      : contract_(std::move(contract)),
        distribution_(request.replicated_coarse ? FieldDistribution::Replicated
                                                : FieldDistribution::Distributed),
        rhs_(request.hierarchy.ba.front(), request.hierarchy.dm.front(), 1, 0),
        phi_(request.hierarchy.ba.front(), request.hierarchy.dm.front(), 1, 1) {
    rhs_.set_val(Real(0));
    phi_.set_val(Real(0));
  }

  std::string_view provider_identity() const noexcept override {
    return "tests.amr.field-solver.graph-identity";
  }
  std::string_view exact_prepared_contract() const noexcept override { return contract_; }
  bool couples_hierarchy_levels() const noexcept override { return false; }
  int level_count() const noexcept override { return 1; }
  FieldDistribution level_distribution(int level) const override {
    if (level != 0)
      throw std::out_of_range("graph-identity provider has exactly one level");
    return distribution_;
  }
  MultiFab& rhs_level(int level) override {
    if (level != 0)
      throw std::out_of_range("graph-identity provider has exactly one level");
    return rhs_;
  }
  MultiFab& phi_level(int level) override {
    if (level != 0)
      throw std::out_of_range("graph-identity provider has exactly one level");
    return phi_;
  }
  void set_boundary_context(const FieldBoundaryExecutionContext&) override {}
  SolveReport solve() override {
    phi_.set_val(Real(0));
    parallel_copy(phi_, rhs_);
    report_.iters = 0;
    report_.reference_residual_norm = norm_inf(rhs_);
    report_.residual_norm = Real(0);
    report_.rel_residual = Real(0);
    report_.mark_solved("external graph identity exact inverse");
    return report_;
  }
  const SolveReport& last_solve_report() const noexcept override { return report_; }

 private:
  std::string contract_;
  FieldDistribution distribution_;
  MultiFab rhs_;
  MultiFab phi_;
  SolveReport report_{};
};

class ExternalGraphIdentityProvider final : public AmrFieldSolverProvider {
 public:
  std::string_view identity() const noexcept override {
    return "tests.amr.field-solver.graph-identity";
  }
  std::uint64_t interface_version() const noexcept override { return 4; }
  std::string_view collective_contract() const noexcept override {
    return "tests.amr.field-solver.graph-identity@4";
  }
  std::vector<std::string> capability_contracts() const override { return {}; }
  AmrFieldSolverOptions default_field_options() const override {
    return {"tests.amr.field-solver.graph-identity.options@4", {}};
  }
  std::optional<AmrFieldHierarchyPolicyAuthority> default_hierarchy_policy(
      std::string_view) const override {
    return std::nullopt;
  }
  PreparedProviderSupport accepts_options(
      const AmrFieldSolverOptions& options) const noexcept override {
    return options.schema_identity == "tests.amr.field-solver.graph-identity.options@4" &&
                   options.values.empty()
               ? PreparedProviderSupport::accept()
               : PreparedProviderSupport::reject(1, "graph identity options are invalid");
  }
  PreparedProviderSupport supports(
      const AmrFieldSolverBuildRequest& request) const noexcept override {
    const auto& policy = request.plan.hierarchy_policy;
    const auto radius = policy.options.values.find("coupling_radius");
    const bool accepted =
        request.use_contract_identity == "pops.amr.field-solver-use.named@1" &&
        request.hierarchy.nlev() == 1 && request.plan.has_reaction &&
        request.plan.reaction == Real(1) &&
        accepts_options(request.plan.solver_options).accepted() &&
        policy.policy_id == "tests.field-hierarchy.coupled-graph" &&
        policy.interface_version == 3 &&
        policy.options.schema_identity == "tests.field-hierarchy.coupled-graph.options@3" &&
        radius != policy.options.values.end() &&
        std::holds_alternative<std::uint64_t>(radius->second) &&
        std::get<std::uint64_t>(radius->second) == 2;
    return accepted ? PreparedProviderSupport::accept()
                    : PreparedProviderSupport::reject(2, "graph identity request is invalid");
  }
  std::string expected_prepared_contract(
      const AmrFieldSolverBuildRequest& request) const override {
    if (!supports(request).accepted())
      throw std::invalid_argument("external graph-identity provider rejected the request");
    return make_amr_field_solver_contract(identity(), request);
  }
  std::unique_ptr<AmrPreparedFieldSolver> build(
      const AmrFieldSolverBuildRequest& request) const override {
    return std::make_unique<ExternalGraphIdentityPrepared>(
        request, expected_prepared_contract(request));
  }
};

#if defined(POPS_HAS_KOKKOS)
class KokkosEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { guard_.emplace(); }
  void TearDown() override { guard_.reset(); }

 private:
  std::optional<Kokkos::ScopeGuard> guard_;
};

::testing::Environment* const kKokkosEnv =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
#endif

// Scalar ExB block of charge q: transport E x B (advection driven by grad phi), charge density q*n for
// the default system Poisson (elliptic = "charge" -> elliptic_rhs = q*n).
using ExBModel = CompositeModel<ExBVelocity, NoSource, ChargeDensity>;
static ExBModel exb_charge(double q, double B0) {
  return ExBModel{ExBVelocity{Real(B0)}, NoSource{}, ChargeDensity{Real(q)}};
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
          result = std::max(result, std::fabs(left(i, j, component) - right(i, j, component)));
  }
  return result;
}

static Real max_valid_scalar_diff(const MultiFab& lhs, const MultiFab& rhs) {
  device_fence();
  Real result = Real(0);
  for (int li = 0; li < lhs.local_size(); ++li) {
    const ConstArray4 left = lhs.fab(li).const_array();
    const int rhs_local = rhs.local_index_of(lhs.global_index(li));
    if (rhs_local < 0)
      throw std::logic_error("scalar comparison layouts have different local ownership");
    const ConstArray4 right = rhs.fab(rhs_local).const_array();
    const Box2D valid = lhs.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        result = std::max(result, std::fabs(left(i, j, 0) - right(i, j, 0)));
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

static Real max_gradient_error(const MultiFab& phi, const MultiFab& aux, int gx_comp, int gy_comp,
                               Real sign, Real dx, Real dy) {
  device_fence();
  Real result = Real(0);
  for (int li = 0; li < aux.local_size(); ++li) {
    const ConstArray4 p = phi.fab(li).const_array();
    const ConstArray4 a = aux.fab(li).const_array();
    const Box2D valid = aux.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        const Real expected_x = sign * (p(i + 1, j) - p(i - 1, j)) / (Real(2) * dx);
        const Real expected_y = sign * (p(i, j + 1) - p(i, j - 1)) / (Real(2) * dy);
        result = std::max(result, std::fabs(a(i, j, gx_comp) - expected_x));
        result = std::max(result, std::fabs(a(i, j, gy_comp) - expected_y));
      }
  }
  return result;
}

static Real max_valid_component_error(const MultiFab& expected, const MultiFab& aux,
                                      int component) {
  device_fence();
  Real result = Real(0);
  for (int li = 0; li < aux.local_size(); ++li) {
    const ConstArray4 source = expected.fab(li).const_array();
    const ConstArray4 published = aux.fab(li).const_array();
    const Box2D valid = aux.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        result = std::max(result, std::fabs(published(i, j, component) - source(i, j)));
  }
  return result;
}

struct CoarseFineGhostCheck {
  Real error = Real(0);
  Real reference = Real(0);
  int samples = 0;
};

static CoarseFineGhostCheck coarse_fine_ghost_check(const MultiFab& coarse, const MultiFab& fine,
                                                    int component) {
  device_fence();
  CoarseFineGhostCheck result;
  const BoxArray& fine_boxes = fine.box_array();
  auto covered_by_fine = [&](int i, int j) {
    for (int box = 0; box < fine_boxes.size(); ++box)
      if (fine_boxes[box].contains(i, j))
        return true;
    return false;
  };
  for (int li = 0; li < fine.local_size(); ++li) {
    const ConstArray4 child = fine.fab(li).const_array();
    const Box2D valid = fine.box(li), grown = fine.fab(li).grown_box();
    for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
      for (int i = grown.lo[0]; i <= grown.hi[0]; ++i) {
        if (valid.contains(i, j) || covered_by_fine(i, j))
          continue;
        const int ci = coarsen_index(i, kAmrRefRatio);
        const int cj = coarsen_index(j, kAmrRefRatio);
        const int parent_box = mf_find_box(coarse, ci, cj);
        if (parent_box < 0)
          continue;
        const Real expected = coarse.fab(parent_box).const_array()(ci, cj, component);
        result.error = std::max(result.error, std::fabs(child(i, j, component) - expected));
        result.reference = std::max(result.reference, std::fabs(expected));
        ++result.samples;
      }
  }
  return result;
}

static Real max_valid_coarse_injection_gap(const MultiFab& coarse, const MultiFab& fine,
                                           int component) {
  device_fence();
  Real result = Real(0);
  for (int li = 0; li < fine.local_size(); ++li) {
    const ConstArray4 child = fine.fab(li).const_array();
    const Box2D valid = fine.box(li);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        const int ci = coarsen_index(i, kAmrRefRatio);
        const int cj = coarsen_index(j, kAmrRefRatio);
        const int parent_box = mf_find_box(coarse, ci, cj);
        if (parent_box < 0)
          continue;
        const Real parent = coarse.fab(parent_box).const_array()(ci, cj, component);
        result = std::max(result, std::fabs(child(i, j, component) - parent));
      }
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

TEST(test_amr_named_field, ExternalPolicyAndEmptyCapabilityProviderRunWithoutCoreBranch) {
  constexpr int n = 16;
  constexpr Real charge = Real(-1);
  AmrBuildParams params;
  params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  params.mesh.periodicity = Periodicity{true, true};
  params.mesh.n = n;
  params.mesh.L = 1.0;
  params.mesh.regrid_every = 0;
  params.poisson.bc = BCRec{};
  const detail::SharedAmrLayout layout = detail::make_shared_amr_layout_levels(params, 1);

  std::vector<AmrRuntimeBlock> blocks;
  blocks.push_back(detail::dispatch_amr_block(exb_charge(charge, 1.0), "minmod", "rusanov", layout,
                                              "plasma", blob(n, 0.25),
                                              /*has_density=*/true, 1.4, 1, false, false));
  blocks[0].aux_ncomp = kAuxNamedBase + 1;

  auto registry = make_default_amr_field_solver_registry();
  registry->add(std::make_shared<ExternalGraphIdentityProvider>());
  const auto external = registry->resolve("tests.amr.field-solver.graph-identity");
  EXPECT_TRUE(external->capability_contracts().empty());
  EXPECT_NO_THROW((void)exact_amr_field_solver_provider_declaration(*external));

  AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc,
                     std::move(blocks), layout.base_per, layout.replicated_coarse, layout.wall,
                     registry);
  test::install_second_order_amr_transfer_authorities(runtime, 1);
  AmrFieldSolveConfig plan;
  plan.plan_identity = "tests.amr.field-solver.graph-identity.plan@4";
  plan.provider_identity = "tests:plasma/graph-identity";
  plan.topology_provider_kind = "tests.graph-topology";
  plan.topology_provenance = "tests:external-coupled-graph";
  plan.topology_digest = "tests:external-coupled-graph:layout@3";
  plan.output_owner_identity = "tests:plasma";
  plan.output_block = "plasma";
  plan.output_key = "graph_identity";
  plan.solver = "tests.amr.field-solver.graph-identity";
  plan.hierarchy_policy = external_graph_hierarchy_policy();
  plan.solver_options = external->default_field_options();
  plan.nullspace = operator_topology_zero_mean_nullspace();
  plan.has_reaction = true;
  plan.reaction = Real(1);
  plan.providers.push_back(FieldProviderBinding{
      "tests:plasma/graph-identity/rhs", "plasma", "graph_identity", Real(1)});
  runtime.install_field_plan("graph_identity", plan);
  runtime.register_named_field("plasma", "graph_identity", kAuxNamedBase, -1, -1,
                               /*gradient_sign=*/Real(1));
  runtime.set_block_named_elliptic_rhs(
      0, "graph_identity", [charge](const MultiFab& state, MultiFab& rhs) {
        add_scaled_component(state, charge, 0, rhs);
      });

  const std::string selected = "graph_identity";
  const SolveReport report = runtime.solve_named_fields(&selected);
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_EQ(report.iters, 0);
  EXPECT_GT(report.reference_residual_norm, Real(0));

  const MultiFab& state = runtime.level_state(0, 0);
  MultiFab expected(state.box_array(), state.dmap(), 1, 1);
  expected.set_val(Real(0));
  // Named fields expose -div(A grad(phi)) + kappa phi = rhs.  The AMR runtime converts that
  // public RHS once to the native div(A grad(phi)) - kappa phi convention before invoking a
  // provider, so an identity provider must observe the negated public closure output.
  add_scaled_component(state, -charge, 0, expected);
  EXPECT_EQ(max_valid_scalar_diff(runtime.provider_potential(selected), expected), Real(0));
}

TEST(test_amr_named_field, Runs) {
  const int N = 64;
  const double L = 1.0, B0 = 1.0, q = -1.0;
  const std::vector<double> rho = blob(N, 0.5);

  // --- single ExB block on a frozen one-level shared hierarchy (default Poisson f = q*rho) ---
  AmrBuildParams bp;
  bp.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  bp.mesh.periodicity = Periodicity{true, true};
  bp.mesh.n = N;
  bp.mesh.L = L;
  bp.mesh.regrid_every = 0;
  bp.poisson.bc = BCRec{};  // periodic
  const detail::SharedAmrLayout S = detail::make_shared_amr_layout(bp);

  std::vector<AmrRuntimeBlock> blocks;
  blocks.push_back(detail::dispatch_amr_block(exb_charge(q, B0), "minmod", "rusanov", S, "plasma",
                                              rho, /*has_density=*/true, 1.4, 1, false, false));
  // Widen the shared aux channel so the named fields' output components (>= kAuxNamedBase) fit. The
  // native ExB block reads only comps 0..2; the runtime sizes the channel to max(b.aux_ncomp), so a
  // wider value just reserves room (the extra comps are written only by the named solve). This is what a
  // model declaring m.aux_field("psi"/"g2x"/"g2y") would set via aux_comps<Model>().
  const int kPhiPsi = kAuxNamedBase;       // 5
  const int kGxPsi = kAuxNamedBase + 1;    // 6
  const int kGyPsi = kAuxNamedBase + 2;    // 7
  const int kPhiChi = kAuxNamedBase + 3;   // 8
  const int kGxChi = kAuxNamedBase + 4;    // 9
  const int kGyChi = kAuxNamedBase + 5;    // 10
  const int kPhiFail = kAuxNamedBase + 6;  // 11 (forced late failure, phi only)
  blocks[0].aux_ncomp = kPhiFail + 1;

  AmrRuntime rt(S.geom, S.runtime_hierarchy(), S.poisson_bc, std::move(blocks), S.base_per,
                S.replicated_coarse, S.wall);
  rt.set_parent_child_temporal_relations({::pops::amr::ParentChildClockRelation(
      0, 1, ::pops::amr::Rational(2, 1), ::pops::amr::RemainderPolicy::IntegralOnly)});
  EXPECT_EQ(rt.n_blocks(), 1) << "named_engine_one_block";

  const SolveReport default_report = rt.solve_default_field();
  ASSERT_TRUE(default_report.solved())
      << "status=" << default_report.status_name() << " action=" << default_report.action_name()
      << " iters=" << default_report.iters << " rel_residual=" << default_report.rel_residual;
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
  psi_plan.solver_options =
      geometric_mg_amr_field_solver_options(GeometricMgOptions{}, CompositeFacOptions{});
  psi_plan.plan_identity = "test:plasma/psi:plan:v1";
  psi_plan.provider_identity = "test:plasma/psi";
  psi_plan.topology_provider_kind = "structured";
  psi_plan.topology_provenance = "test:periodic-cartesian";
  psi_plan.topology_digest = "test:periodic-cartesian:v1";
  psi_plan.output_owner_identity = "test:plasma";
  psi_plan.output_block = "plasma";
  psi_plan.output_key = "psi";
  psi_plan.hierarchy_policy = composite_hierarchy_policy();
  psi_plan.nullspace = operator_topology_zero_mean_nullspace();
  psi_plan.providers.push_back(
      FieldProviderBinding{"test:plasma/psi/rhs", "plasma", "psi", Real(1)});
  rt.install_field_plan("psi", psi_plan);
  rt.register_named_field("plasma", "psi", kPhiPsi, kGxPsi, kGyPsi, /*gradient_sign=*/-1);
  rt.set_block_named_elliptic_rhs(0, "psi", [q](const MultiFab& U, MultiFab& rhs) {
    add_scaled_component(U, Real(q), 0, rhs);  // f_psi = q * rho == default Poisson RHS
  });
  EXPECT_TRUE(rt.has_named_field("psi") && rt.n_named_fields() == 1) << "named_psi_registered";

  const std::string psi_field = "psi";
  ASSERT_TRUE(rt.solve_named_fields(&psi_field).solved());
  ASSERT_EQ(rt.provider_potential_levels("psi"), rt.nlev());
  for (int level = 0; level < rt.nlev(); ++level) {
    const Real spacing = S.geom.dx() / Real(1 << level);
    EXPECT_EQ(max_gradient_error(rt.provider_potential_level("psi", level), rt.aux(level), kGxPsi,
                                 kGyPsi, Real(-1), spacing, spacing),
              Real(0))
        << "GradientOutput(sign=-1) publishes -grad(phi) on FAC level " << level;
  }
  const std::vector<double> psi = valid_values(rt.provider_potential("psi"), N);
  EXPECT_EQ(static_cast<int>(psi.size()), N * N) << "named_psi_shape_nxn";
  bool psi_finite = true;
  for (double v : psi)
    psi_finite = psi_finite && std::isfinite(v);
  EXPECT_TRUE(psi_finite) << "named_psi_finite";

  // The public named equation is -lap(psi)=rhs, whereas the legacy default field keeps the native
  // lap(phi)=rhs convention. Therefore psi == -default phi after recentering on the periodic
  // additive constant (same discrete operator/RHS; only the public/native sign differs).
  const double psi_mean = mean_of(psi);
  double dmax_par = 0, ref_par = 0;
  for (int k = 0; k < N * N; ++k) {
    dmax_par =
        std::fmax(dmax_par, std::fabs((psi[k] - psi_mean) + (phi_default[k] - phi_def_mean)));
    ref_par = std::fmax(ref_par, std::fabs(phi_default[k] - phi_def_mean));
  }
  EXPECT_GT(ref_par, 1e-6) << "named_parity_oracle_nontrivial";
  EXPECT_LT(dmax_par, 1e-2 * (ref_par + 1e-12))
      << "public -lap(psi)=q*rho tracks the negative native default potential";

  // (2) DISTINCT RHS (linearity): named field "chi" with RHS = 2*q*rho -> chi = 2*psi (Poisson linear).
  // A genuinely different, correctly scaled second field (not an alias of the default phi).
  AmrFieldSolveConfig chi_plan;
  chi_plan.solver_options =
      geometric_mg_amr_field_solver_options(GeometricMgOptions{}, CompositeFacOptions{});
  chi_plan.plan_identity = "test:plasma/chi:plan:v1";
  chi_plan.provider_identity = "test:plasma/chi";
  chi_plan.topology_provider_kind = "structured";
  chi_plan.topology_provenance = "test:periodic-cartesian";
  chi_plan.topology_digest = "test:periodic-cartesian:v1";
  chi_plan.output_owner_identity = "test:plasma";
  chi_plan.output_block = "plasma";
  chi_plan.output_key = "chi";
  chi_plan.hierarchy_policy = composite_hierarchy_policy();
  chi_plan.nullspace = operator_topology_zero_mean_nullspace();
  chi_plan.providers.push_back(
      FieldProviderBinding{"test:plasma/chi/rhs", "plasma", "chi", Real(1)});
  rt.install_field_plan("chi", chi_plan);
  rt.register_named_field("plasma", "chi", kPhiChi, kGxChi, kGyChi, /*gradient_sign=*/1);
  rt.set_block_named_elliptic_rhs(0, "chi", [q](const MultiFab& U, MultiFab& rhs) {
    add_scaled_component(U, Real(2.0 * q), 0, rhs);  // f_chi = 2 * (q * rho)
  });
  EXPECT_EQ(rt.n_named_fields(), 2) << "named_chi_registered";

  const MultiFab aux_before_selected_chi = rt.aux(0);
  const std::string chi_field = "chi";
  ASSERT_TRUE(rt.solve_named_fields(&chi_field).solved());
  ASSERT_EQ(rt.provider_potential_levels("chi"), rt.nlev());
  for (int level = 0; level < rt.nlev(); ++level) {
    const Real spacing = S.geom.dx() / Real(1 << level);
    EXPECT_EQ(max_gradient_error(rt.provider_potential_level("chi", level), rt.aux(level), kGxChi,
                                 kGyChi, Real(1), spacing, spacing),
              Real(0))
        << "GradientOutput(sign=+1) publishes +grad(phi) on level-local level " << level;
  }
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
  } catch (const FieldNullspaceIncompatibleRhs& error) {
    context_diagnostic = error.what();
  }
  EXPECT_NE(context_diagnostic.find("incompatible with prepared nullspace basis"),
            std::string::npos)
      << context_diagnostic;
  EXPECT_NE(context_diagnostic.find("silent projection is forbidden"), std::string::npos)
      << context_diagnostic;
  EXPECT_EQ(max_abs_diff(rt.phi(), context_phi_before), Real(0));
  for (int level = 0; level < rt.nlev(); ++level)
    EXPECT_EQ(max_abs_diff(rt.aux(level), context_aux_before[static_cast<std::size_t>(level)]),
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
  fail_plan.solver_options =
      geometric_mg_amr_field_solver_options(GeometricMgOptions{}, CompositeFacOptions{});
  fail_plan.plan_identity = "test:plasma/zeta:plan:v1";
  fail_plan.provider_identity = "test:plasma/zeta";
  fail_plan.topology_provider_kind = "structured";
  fail_plan.topology_provenance = "test:periodic-cartesian";
  fail_plan.topology_digest = "test:periodic-cartesian:v1";
  fail_plan.output_owner_identity = "test:plasma";
  fail_plan.output_block = "plasma";
  fail_plan.output_key = "zeta";
  fail_plan.hierarchy_policy = level_local_hierarchy_policy();
  fail_plan.nullspace = operator_topology_zero_mean_nullspace();
  // This rollback oracle is about a late iteration-limit failure. A level-local periodic Poisson
  // over a refined subdomain has one compatibility condition per level, so make the operator
  // genuinely invertible instead of relying on the old silent RHS projection.
  fail_plan.has_reaction = true;
  fail_plan.reaction = Real(1);
  fail_plan.solver_options.values["mg.rel_tol"] = 1e-30;
  fail_plan.solver_options.values["mg.max_cycles"] = std::int64_t{1};
  fail_plan.providers.push_back(
      FieldProviderBinding{"test:plasma/zeta/rhs", "plasma", "zeta", Real(1)});
  rt.install_field_plan("zeta", fail_plan);
  rt.register_named_field("plasma", "zeta", kPhiFail, /*gx=*/-1, /*gy=*/-1, /*gradient_sign=*/1);
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

TEST(test_amr_named_field, RefinedPublicationPreservesValidAndRefreshesGhosts) {
  constexpr int n = 32;
  constexpr Real reaction = Real(2);
  constexpr double charge = -1.0;
  AmrBuildParams params;
  params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
  params.mesh.periodicity = Periodicity{true, true};
  params.mesh.n = n;
  params.mesh.L = 1.0;
  params.mesh.regrid_every = 0;
  params.poisson.bc = BCRec{};
  const detail::SharedAmrLayout layout = detail::make_shared_amr_layout(params);

  std::vector<AmrRuntimeBlock> blocks;
  blocks.push_back(detail::dispatch_amr_block(exb_charge(charge, 1.0), "minmod", "rusanov", layout,
                                              "plasma", blob(n, 0.5),
                                              /*has_density=*/true, 1.4, 1, false, false));
  constexpr int phi_component = kAuxNamedBase;
  constexpr int gx_component = kAuxNamedBase + 1;
  constexpr int gy_component = kAuxNamedBase + 2;
  blocks[0].aux_ncomp = gy_component + 1;

  AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc, std::move(blocks),
                     layout.base_per, layout.replicated_coarse, layout.wall);
  test::install_second_order_amr_transfer_authorities(runtime, 1);
  runtime.set_parent_child_temporal_relations({::pops::amr::ParentChildClockRelation(
      0, 1, ::pops::amr::Rational(2, 1), ::pops::amr::RemainderPolicy::IntegralOnly)});
  AmrFieldSolveConfig plan;
  plan.solver_options =
      geometric_mg_amr_field_solver_options(GeometricMgOptions{}, CompositeFacOptions{});
  plan.plan_identity = "test:plasma/screened:plan:v1";
  plan.provider_identity = "test:plasma/screened";
  plan.topology_provider_kind = "structured";
  plan.topology_provenance = "test:periodic-cartesian";
  plan.topology_digest = "test:periodic-cartesian:v1";
  plan.output_owner_identity = "test:plasma";
  plan.output_block = "plasma";
  plan.output_key = "screened";
  plan.hierarchy_policy = composite_hierarchy_policy();
  plan.nullspace = operator_topology_zero_mean_nullspace();
  plan.has_reaction = true;
  plan.reaction = reaction;
  plan.providers.push_back(
      FieldProviderBinding{"test:plasma/screened/rhs", "plasma", "screened", Real(1)});
  runtime.install_field_plan("screened", plan);
  runtime.register_named_field("plasma", "screened", phi_component, gx_component, gy_component,
                               /*gradient_sign=*/-1);
  runtime.set_block_named_elliptic_rhs(0, "screened",
                                       [charge](const MultiFab& state, MultiFab& rhs) {
                                         add_scaled_component(state, Real(charge), 0, rhs);
                                       });

  const std::string field = "screened";
  ASSERT_TRUE(runtime.solve_named_fields(&field).solved());
  ASSERT_EQ(runtime.nlev(), 2);
  for (int level = 0; level < runtime.nlev(); ++level)
    EXPECT_EQ(max_valid_component_error(runtime.provider_potential_level(field, level),
                                        runtime.aux(level), phi_component),
              Real(0))
        << "resolved FAC valid cells remain authoritative on level " << level;

  EXPECT_GT(max_valid_coarse_injection_gap(
                runtime.aux(0), runtime.provider_potential_level(field, 1), phi_component),
            Real(1e-8))
      << "the fine FAC solution must differ from full-grown coarse injection for this oracle";
  const CoarseFineGhostCheck ghosts =
      coarse_fine_ghost_check(runtime.aux(0), runtime.aux(1), phi_component);
  ASSERT_GT(ghosts.samples, 0) << "the refined patch exposes coarse/fine ghosts";
  ASSERT_GT(ghosts.reference, Real(1e-8)) << "the ghost oracle is nontrivial";
  EXPECT_EQ(ghosts.error, Real(0))
      << "coarse/fine ghosts must come from the freshly published coarse solution";
}

TEST(test_amr_named_field, ProviderSupportDistinguishesRepresentedAndUnrepresentedTopologies) {
  auto make_plan = [](const std::string& name,
                      const AmrFieldHierarchyPolicyAuthority& hierarchy_policy) {
    AmrFieldSolveConfig plan;
    plan.solver_options =
        geometric_mg_amr_field_solver_options(GeometricMgOptions{}, CompositeFacOptions{});
    plan.plan_identity = "test:" + name + ":plan:v1";
    plan.provider_identity = "test:" + name;
    plan.topology_provider_kind = "builtin_rectangular_cell_graph_v1";
    plan.topology_provenance = "test:rectangular-cell-graph";
    plan.topology_digest = "test:rectangular-cell-graph:v1";
    plan.output_owner_identity = "test:plasma";
    plan.output_block = "plasma";
    plan.output_key = name;
    plan.hierarchy_policy = hierarchy_policy;
    plan.nullspace = operator_topology_zero_mean_nullspace();
    plan.providers.push_back(
        FieldProviderBinding{"test:" + name + ":rhs", "plasma", name, Real(1)});
    return plan;
  };

  {
    AmrBuildParams params;
    params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
    params.mesh.periodicity = Periodicity{true, true};
    params.mesh.n = 16;
    params.mesh.regrid_every = 0;
    params.mesh.distribute_coarse = true;
    params.mesh.coarse_max_grid = 8;
    const detail::SharedAmrLayout layout = detail::make_shared_amr_layout(params);
    std::vector<AmrRuntimeBlock> blocks;
    blocks.push_back(detail::dispatch_amr_block(
        exb_charge(-1.0, 1.0), "minmod", "rusanov", layout, "plasma",
        blob(params.mesh.n, 0.25), /*has_density=*/true, 1.4, 1, false, false));
    AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc,
                       std::move(blocks), layout.base_per, layout.replicated_coarse, layout.wall);
    test::install_second_order_amr_transfer_authorities(runtime, 1);
    std::string diagnostic;
    try {
      runtime.install_field_plan("distributed-composite",
                                 make_plan("distributed-composite",
                                           composite_hierarchy_policy()));
      ADD_FAILURE()
          << "FAC's replicated coarse storage must not be paired with distributed runtime coverage";
    } catch (const std::invalid_argument& error) {
      diagnostic = error.what();
    }
    EXPECT_NE(diagnostic.find("provider rejected request (code 14)"), std::string::npos)
        << diagnostic;
    EXPECT_NE(diagnostic.find("coarse distribution or active region"), std::string::npos)
        << diagnostic;
  }

  {
    AmrBuildParams params;
    params.mesh.load_balance = test::prepare_test_space_filling_curve_load_balance();
    params.mesh.periodicity = Periodicity{true, true};
    params.mesh.n = 16;
    params.mesh.regrid_every = 0;
    params.poisson.wall = ActiveRegionProvider2D::trusted_extension(
        {"pops.test.amr-named-field.quarter-wall", 1}, exact_provider_parameters(Real(0.25)),
        [](Real x, Real y) { return x < Real(0.25) && y < Real(0.25); });
    const detail::SharedAmrLayout layout = detail::make_shared_amr_layout(params);
    std::vector<AmrRuntimeBlock> blocks;
    blocks.push_back(detail::dispatch_amr_block(
        exb_charge(-1.0, 1.0), "minmod", "rusanov", layout, "plasma",
        blob(params.mesh.n, 0.25), /*has_density=*/true, 1.4, 1, false, false));
    blocks[0].aux_ncomp = kAuxNamedBase + 1;
    AmrRuntime runtime(layout.geom, layout.runtime_hierarchy(), layout.poisson_bc,
                       std::move(blocks), layout.base_per, layout.replicated_coarse, layout.wall);
    test::install_second_order_amr_transfer_authorities(runtime, 1);
    runtime.install_field_plan("wall-field",
                               make_plan("wall-field", level_local_hierarchy_policy()));
    runtime.register_named_field("plasma", "wall-field", kAuxNamedBase, -1, -1,
                                 /*gradient_sign=*/Real(1));
    runtime.set_block_named_elliptic_rhs(0, "wall-field",
                                         [](const MultiFab& state, MultiFab& rhs) {
                                           add_scaled_component(state, Real(-1), 0, rhs);
                                         });
    const std::string wall_field = "wall-field";
    EXPECT_GT(norm_inf(runtime.level_state(0, 0)), Real(0))
        << "the embedded-Dirichlet solve must receive a non-trivial source field";
    const SolveReport wall_report = runtime.solve_named_fields(&wall_field);
    ASSERT_TRUE(wall_report.solved()) << wall_report.reason;
    const MultiFab first_wall_solution = runtime.provider_potential(wall_field);
    EXPECT_GT(norm_inf(first_wall_solution), Real(0))
        << "the level-local active-region provider must execute a non-trivial embedded-Dirichlet solve";
    runtime.provider_potential(wall_field).set_val(Real(0));
    const SolveReport repeated_wall_report = runtime.solve_named_fields(&wall_field);
    ASSERT_TRUE(repeated_wall_report.solved()) << repeated_wall_report.reason;
    EXPECT_EQ(max_valid_scalar_diff(runtime.provider_potential(wall_field), first_wall_solution),
              Real(0))
        << "the represented wall topology must reproduce the same solution from the same zero start";
    AmrFieldSolveConfig screened =
        make_plan("wall-screened", level_local_hierarchy_policy());
    screened.has_reaction = true;
    screened.reaction = Real(1);
    EXPECT_NO_THROW(runtime.install_field_plan("wall-screened", screened))
        << "an invertible level-local Helmholtz field needs no nullspace component provider";
    AmrFieldSolveConfig composite_screened =
        make_plan("wall-composite", composite_hierarchy_policy());
    composite_screened.has_reaction = true;
    composite_screened.reaction = Real(1);
    std::string diagnostic;
    try {
      runtime.install_field_plan("wall-composite", composite_screened);
      ADD_FAILURE()
          << "CompositeFacPoisson has no wall-mask carrier even when the operator is invertible";
    } catch (const std::invalid_argument& error) {
      diagnostic = error.what();
    }
    EXPECT_NE(diagnostic.find("provider rejected request (code 14)"), std::string::npos)
        << diagnostic;
    EXPECT_NE(diagnostic.find("coarse distribution or active region"), std::string::npos)
        << diagnostic;
  }
}
