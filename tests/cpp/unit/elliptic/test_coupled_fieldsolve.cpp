// Coupled multi-block field solve (Spec 3 section 12.3, criterion 24; ADC-457).
//
// The exact-ranked field plan assembles the system Poisson RHS as
// f = Sum_s elliptic_rhs_s(U_s) reading EVERY block's stage state at once (indexed by block index;
// a nullptr entry uses the block's live state) -- the SIMULTANEOUS multi-target counterpart of the
// single-target assemble_poisson_rhs. System::solve_fields_from_blocks wraps it (solve + aux derive),
// the seam the compiled-Program codegen lowers P.solve_fields_from_blocks([...]) to (ProgramContext).
//
// This test exercises that path through the PUBLIC System API (assemble_poisson_rhs_* are private to
// the templated field solver). Two charge (ExB) blocks with distinct exact-rank mean-zero densities:
//   (a) SUM over all blocks: solve_fields_from_blocks({&U0, &U1}) (every block at its LIVE state) gives
//       a potential matching the historical solve_fields() to round-off -- both assemble Sum_s
//       elliptic_rhs_s(s.U). This is the "RHS == sum of the per-block contributions" assertion. (Not
//       bit-for-bit: the GeometricMG is iterative + warm-started, so a redundant solve stops on a
//       relative tolerance -- the result matches to ~ulp*|phi|, not exactly.)
//   (b) PER-BLOCK stage override: with block 1's slot pointing at a stage state whose charge is ZEROED,
//       the potential matches (to round-off) a reference where block 1's LIVE density is zero (only
//       block 0 contributes) -- so block 1 read its STAGE override, not its live state (the per-block
//       sum is honored per slot, the coupled commit_many guarantee), and differs from the all-live
//       solve by far more than the tolerance.
//   (c) SIZE guard: a wrong-sized U_stages vector throws (a stale binding cannot silently mis-route).
//
// The source contains one kNativeDimension algorithm. Native dimension builds qualify it in 1D,
// 2D, and 3D; the np2 variant retains collective request/RHS failure and recovery checks.

#include <gtest/gtest.h>

#include <pops/runtime/system.hpp>

#include <memory>
#include <pops/coupling/base/elliptic_rhs.hpp>
#include <pops/core/state/state.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_builtins.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/physics/bricks/elliptic.hpp>
#include <pops/physics/bricks/source.hpp>
#include <pops/physics/composition/composite.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>

#include "test_harness.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

using namespace pops;

namespace {

constexpr int kDim = kNativeDimension;
using NativeMultiFab = MultiFab<kDim>;
using NativeSystem = System<kDim>;
using NativeSystemConfig = SystemConfig<kDim>;
using ChargeModel =
    CompositeModel<nd::ScalarAdvection<kDim>, NoSource, ChargeDensity>;

using runtime::system::AuxiliaryComponentContract;
using runtime::system::AuxiliaryComponentKey;
using runtime::system::AuxiliaryEvaluationEvent;
using runtime::system::AuxiliaryEvaluationPolicy;
using runtime::system::AuxiliaryFreshness;
using runtime::system::AuxiliaryOutput;
using runtime::system::AuxiliaryProviderKind;
using runtime::system::AuxiliaryStorageShape;
using runtime::system::PreparedAuxiliaryProvider;

class CollectiveRuntimeEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override { comm_init(); }
  void TearDown() override { comm_finalize(); }
};

[[maybe_unused]] const ::testing::Environment* const kCollectiveRuntimeEnvironment =
    ::testing::AddGlobalTestEnvironment(new CollectiveRuntimeEnvironment);

std::vector<AuxiliaryComponentKey> install_field_outputs(NativeSystem& system,
                                                         const std::string& owner,
                                                         const std::string& field, int count) {
  AuxiliaryStorageShape<kDim> shape;
  for (int axis = 0; axis < kDim; ++axis)
    shape.halo[axis] = 1;
  const AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "field", "scalar"};
  std::vector<AuxiliaryOutput<kDim>> outputs;
  std::vector<AuxiliaryComponentKey> keys;
  outputs.reserve(static_cast<std::size_t>(count));
  keys.reserve(static_cast<std::size_t>(count));
  for (int component = 0; component < count; ++component) {
    AuxiliaryComponentKey key{
        owner, "field", field,
        component == 0 ? "potential" : "gradient-" + std::to_string(component - 1)};
    keys.push_back(key);
    outputs.push_back({std::move(key), contract, shape});
  }
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<kDim>{
      "test.field-output/" + owner + "/" + field,
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      std::move(outputs),
      {}});
  system.seal_auxiliary_providers();
  return keys;
}

class DecoratedNullspaceProvider final : public FieldNullspaceProvider<kDim> {
 public:
  DecoratedNullspaceProvider()
      : inner_(make_default_field_nullspace_provider_registry<kDim>()->resolve(
            "pops.field-nullspace.operator-topology-derived")) {}

  [[nodiscard]] std::string_view identity() const noexcept override {
    return "test.field-nullspace.decorated";
  }
  [[nodiscard]] std::uint64_t interface_version() const noexcept override { return 1; }
  [[nodiscard]] std::string_view collective_contract() const noexcept override {
    return "test.field-nullspace.decorated@1";
  }
  [[nodiscard]] PreparedProviderOptions default_options() const override {
    return inner_->default_options();
  }
  [[nodiscard]] bool accepts_options(
      const PreparedProviderOptions& options) const noexcept override {
    return inner_->accepts_options(options);
  }
  [[nodiscard]] PreparedProviderSupport supports(
      const FieldNullspaceProviderRequest<kDim>& request) const noexcept override {
    return inner_->supports(request);
  }
  [[nodiscard]] std::string expected_prepared_contract(
      const FieldNullspaceProviderRequest<kDim>& request) const override {
    ExactContractBuilder contract;
    contract.text("test.prepared-field-nullspace.decorator")
        .scalar(std::uint32_t{1})
        .bytes(inner_->expected_prepared_contract(request));
    return std::move(contract).release();
  }
  [[nodiscard]] PreparedFieldNullspace<kDim> prepare(
      const FieldNullspaceProviderRequest<kDim>& request) const override {
    PreparedFieldNullspace<kDim> prepared = inner_->prepare(request);
    prepared.provider_identity = std::string(identity());
    prepared.provider_version = interface_version();
    prepared.exact_prepared_contract = expected_prepared_contract(request);
    return prepared;
  }

 private:
  std::shared_ptr<const FieldNullspaceProvider<kDim>> inner_;
};

}  // namespace

namespace {

std::size_t cell_count(int cells_per_axis) {
  std::size_t result = 1;
  for (int axis = 0; axis < kDim; ++axis)
    result *= static_cast<std::size_t>(cells_per_axis);
  return result;
}

NativeSystemConfig periodic_unit_config(int cells_per_axis) {
  NativeSystemConfig config;
  for (int axis = 0; axis < kDim; ++axis) {
    config.shape[axis] = cells_per_axis;
    config.lower[axis] = Real(0);
    config.upper[axis] = Real(1);
    config.periodicity[axis] = true;
  }
  return config;
}

void install_periodic_field_boundary(NativeSystem& system, const std::string& slot) {
  const std::vector<std::string> kinds(static_cast<std::size_t>(2 * kDim), "periodic");
  const std::vector<double> values(static_cast<std::size_t>(2 * kDim), 0.0);
  system.set_field_boundary_plan(slot, kinds, values, values, values);
}

ChargeModel charge_model() {
  RealVector<kDim> velocity{};
  ChargeModel model{};
  model.hyp = nd::ScalarAdvection<kDim>::prepare(velocity);
  model.src = NoSource{};
  model.ell = ChargeDensity{Real(1)};
  return model;
}

// A charge density with exactly zero discrete mean on a periodic Cartesian grid. Every native axis
// contributes one complete cosine period and a deterministic phase multiple which distinguishes
// blocks. Flattening keeps axis zero contiguous, matching exact field marshaling.
std::vector<double> charge_density(int n, double amp, double phase) {
  std::vector<double> q(cell_count(n), 0.0);
  for (std::size_t cell = 0; cell < q.size(); ++cell) {
    std::size_t quotient = cell;
    double value = amp;
    for (int axis = 0; axis < kDim; ++axis) {
      const int coordinate = static_cast<int>(quotient % static_cast<std::size_t>(n));
      quotient /= static_cast<std::size_t>(n);
      const double x = (coordinate + 0.5) / n + phase * static_cast<double>(axis + 1);
      value *= std::cos(2.0 * test::kPi * x);
    }
    q[cell] = value;
  }
  return q;
}

// Blocks declared "n0" then "n1" -> runtime indices 0 and 1 (the order the coupled vector expects).
void add_two_charge_blocks(NativeSystem& s) {
  for (const std::string block : {std::string("n0"), std::string("n1")}) {
    s.install_block_state_route(block, "test.coupled-fieldsolve/" + block + "/state@1");
    add_compiled_model(s, block, charge_model(), "none", "rusanov", "conservative", "explicit");
  }
}

// Builds the same two charge blocks with the principal exact Cartesian field route.
void build_two_charge_blocks(NativeSystem& s) {
  add_two_charge_blocks(s);
  s.set_poisson("composite", "cartesian_cg");  // f = sum of the per-block elliptic bricks
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size())
    return 1e300;  // a size mismatch is a hard failure (compared against 0 by the caller)
  double d = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k)
    d = std::fmax(d, std::fabs(a[k] - b[k]));
  return d;
}

}  // namespace

TEST(test_coupled_fieldsolve, coupled_solve_matches_solve_fields_and_honors_stage_overrides) {
  // System storage initializes Kokkos lazily through the PoPS runtime.  That process-wide lifetime
  // is finalized at exit, after every local System/MultiFab has been destroyed; a per-test
  // ScopeGuard would finalize Kokkos here and make the next TEST attempt an illegal reinitialize.
  test::Checker chk(test::Checker::Style::Verbose);

  const int n = 32;
  const NativeSystemConfig cfg = periodic_unit_config(n);
  const std::vector<double> q0 = charge_density(n, 1.0, 0.0);
  const std::vector<double> q1 = charge_density(n, 0.6, 0.25);  // distinct from block 0

  // (a) SUM over all blocks: a coupled solve from the LIVE states == historical solve_fields ----------
  NativeSystem s(cfg);
  build_two_charge_blocks(s);
  s.set_density("n0", q0);
  s.set_density("n1", q1);
  chk(s.n_blocks() == 2, "two blocks installed");

  (void)consume_solve_outcome(
      s.solve_fields());  // historical: f = elliptic_rhs(n0.U) + elliptic_rhs(n1.U)
  const std::vector<double> phi_ref = s.potential_global();

  NativeMultiFab& U0 = s.block_state(0);
  NativeMultiFab& U1 = s.block_state(1);
  std::vector<const NativeMultiFab*> stages_live{&U0, &U1};
  (void)consume_solve_outcome(
      s.solve_fields_from_blocks(stages_live));  // coupled: every block at its live state
  const std::vector<double> phi_blocks = s.potential_global();

  chk(!phi_ref.empty() && phi_ref.size() == cell_count(n), "potential size");
  bool finite = true;
  for (double v : phi_blocks)
    finite = finite && std::isfinite(v);
  chk(finite, "coupled potential is finite");
  double maxabs = 0.0;
  for (double v : phi_ref)
    maxabs = std::fmax(maxabs, std::fabs(v));
  // The two solves assemble the SAME RHS (Sum_s elliptic_rhs_s(s.U)); the GeometricMG is iterative and
  // WARM-STARTED, so the second solve resumes from the first's converged phi and the V-cycle stops on a
  // RELATIVE tolerance -- the result matches to round-off, not bit-for-bit (a redundant iterative solve
  // is rarely a true no-op). We assert a tight relative agreement (~few ulp * |phi|): proves the
  // from-blocks RHS == the historical sum, the field-solve numerics are identical.
  const double d_sum = max_abs_diff(phi_blocks, phi_ref);
  const double tol = 1e-12 * std::fmax(maxabs, 1.0);
  std::printf("  d_sum=%.3e (tol=%.3e, |phi|max=%.3e)\n", d_sum, tol, maxabs);
  chk(d_sum <= tol,
      "coupled solve from live states matches solve_fields to round-off (RHS == sum of blocks)");
  // The potential is non-trivial (a genuine solve, not a zero field) -- guards against a vacuous pass.
  chk(maxabs > 0.0, "the coupled solve produces a non-trivial potential");

  // (b) PER-BLOCK stage override: block 1 reads a ZEROED stage state (drops its contribution) ----------
  // Reference: only block 0 contributes (block 1's live density zeroed), via the historical path.
  NativeSystem ref(cfg);
  build_two_charge_blocks(ref);
  ref.set_density("n0", q0);
  ref.set_density("n1", std::vector<double>(cell_count(n), 0.0));  // block 1 = 0
  (void)consume_solve_outcome(ref.solve_fields());
  const std::vector<double> phi_only0 = ref.potential_global();

  // Coupled solve on the original system: block 0 at its live state, block 1 at a ZEROED stage copy.
  NativeMultiFab stage1 = s.block_state(1);  // deep copy of block 1's exact-ranked live state
  stage1.set_val(Real(0));                   // zero the stage charge
  std::vector<const NativeMultiFab*> stages_override{&s.block_state(0), &stage1};
  (void)consume_solve_outcome(s.solve_fields_from_blocks(stages_override));
  const std::vector<double> phi_override = s.potential_global();

  const double d_override = max_abs_diff(phi_override, phi_only0);
  const double d_vs_all = max_abs_diff(phi_override, phi_blocks);
  std::printf("  d_override=%.3e (tol=%.3e)  d_vs_all=%.3e\n", d_override, tol, d_vs_all);
  // Same RHS as the only-block-0 reference (block 1 contributes zero), warm-started GeometricMG -> a
  // round-off match (see the d_sum note), not bit-for-bit.
  chk(d_override <= tol,
      "block 1's ZEROED stage override drops its charge (== only-block-0 reference, to round-off)");
  // And it differs from the all-blocks solve by FAR more than the tolerance (block 1's live charge
  // mattered) -- the override was honored, not ignored.
  chk(d_vs_all > 1e-4, "the stage override changes the potential vs the all-live coupled solve");

  // (b2) ALL-nullptr U_stages: every slot falls back to its block's LIVE state == the all-live solve --
  std::vector<const NativeMultiFab*> stages_null{nullptr, nullptr};
  (void)consume_solve_outcome(s.solve_fields_from_blocks(stages_null));
  const std::vector<double> phi_null = s.potential_global();
  chk(max_abs_diff(phi_null, phi_blocks) <= tol,
      "an all-nullptr U_stages falls back to every block's live state (== the all-live coupled "
      "solve)");

  // (c) SIZE guard: a U_stages not sized to n_blocks() throws (fail-loud on a stale binding) ----------
  std::vector<const NativeMultiFab*> bad{&s.block_state(0)};  // size 1 != 2 blocks
  if (n_ranks() == 1) {
    EXPECT_THROW((void)s.solve_fields_from_blocks(bad), std::invalid_argument);
  } else {
    EXPECT_THROW((void)s.solve_fields_from_blocks(bad), std::runtime_error);
  }

  if (!chk.failed())
    std::printf("OK test_coupled_fieldsolve\n");
}

TEST(test_coupled_fieldsolve, named_solve_honors_every_qualified_stage_without_live_mutation) {
  const int n = 32;
  const NativeSystemConfig cfg = periodic_unit_config(n);
  NativeSystem system(cfg);
  add_two_charge_blocks(system);

  const std::string slot = "qualified-coupled-provider";
  const PreparedProviderOptions backend_options{
      "pops.system.cartesian-cg-options@1",
      {{"abs_tol", 0.0}, {"max_iterations", std::int64_t{2000}}, {"rel_tol", 1.0e-10}}};
  system.register_configured_field_solver_provider("cartesian_cg", slot, backend_options);
  system.set_field_solver_plan(slot, "test:qualified-coupled-plan",
                               "test:qualified-coupled-provider", "test:qualified-coupled-field",
                               "n0", "potential",
                               {"test:n0/potential/rhs", "test:n1/potential/rhs"}, {"n0", "n1"},
                               {"potential", "potential"}, {1.0, 1.0}, slot);
  const std::string conflicting_slot = "conflicting-output-provider";
  system.register_configured_field_solver_provider("cartesian_cg", conflicting_slot,
                                                   backend_options);
  EXPECT_THROW(system.set_field_solver_plan(conflicting_slot, "test:conflicting-output-plan",
                                            "test:conflicting-output-provider", "test:other-owner",
                                            "n0", "potential", {"test:n0/other/rhs"}, {"n0"},
                                            {"other"}, {1.0}, conflicting_slot),
               std::runtime_error)
      << "one output block/key must identify exactly one qualified provider slot";
  system.set_field_topology_authority(slot, "builtin_rectangular_cell_graph_v1",
                                      "test:periodic-cartesian", "test:periodic-cartesian:v1");
  install_periodic_field_boundary(system, slot);
  system.register_field_nullspace_provider(std::make_shared<DecoratedNullspaceProvider>());
  system.set_field_nullspace(
      slot, "test.field-nullspace.decorated",
      PreparedProviderOptions{"pops.field-nullspace.operator-topology-derived.options@1",
                              {{"gauge.value", 0.0}}});
  const auto potential_key =
      install_field_outputs(system, "test.qualified-coupled", "potential", 1);
  system.register_elliptic_field("n0", "potential", potential_key, 1);
  system.set_block_elliptic_field(
      "n0", "potential", [](const NativeMultiFab& state, NativeMultiFab& rhs) {
        add_scaled_component(state, Real(1), 0, rhs);
      });
  bool fail_rank_local_rhs = false;
  system.set_block_elliptic_field(
      "n1", "potential", [&](const NativeMultiFab& state, NativeMultiFab& rhs) {
        if (fail_rank_local_rhs)
          throw std::runtime_error("intentional rank-local RHS failure");
        add_scaled_component(state, Real(1), 0, rhs);
      });

  const std::vector<double> q0 = charge_density(n, 1.0, 0.0);
  const std::vector<double> q1 = charge_density(n, 0.6, 0.25);
  system.set_density("n0", q0);
  system.set_density("n1", q1);

  std::vector<const NativeMultiFab*> all_live{&system.block_state(0), &system.block_state(1)};
  const SolveReport all_report =
      consume_solve_outcome(system.solve_fields_from_blocks(slot, all_live));
  ASSERT_TRUE(all_report.solved_value_available()) << all_report.status_name();
  const std::vector<double> all_phi = system.field_potential_global(slot);

  // The removed formula-specific ``eps`` facade encoded this as 1/eps.  The generic field plan
  // carries one explicit coefficient per qualified RHS provider instead.  A fresh system with both
  // coefficients halved must therefore publish half the all-live potential in every native rank.
  NativeSystem scaled(cfg);
  add_two_charge_blocks(scaled);
  const std::string scaled_slot = "qualified-scaled-provider";
  scaled.register_configured_field_solver_provider("cartesian_cg", scaled_slot, backend_options);
  scaled.set_field_solver_plan(
      scaled_slot, "test:qualified-scaled-plan", "test:qualified-scaled-provider",
      "test:qualified-scaled-field", "n0", "potential",
      {"test:n0/potential/scaled-rhs", "test:n1/potential/scaled-rhs"}, {"n0", "n1"},
      {"potential", "potential"}, {0.5, 0.5}, scaled_slot);
  scaled.set_field_topology_authority(scaled_slot, "builtin_rectangular_cell_graph_v1",
                                      "test:periodic-cartesian", "test:periodic-cartesian:v1");
  install_periodic_field_boundary(scaled, scaled_slot);
  scaled.register_field_nullspace_provider(std::make_shared<DecoratedNullspaceProvider>());
  scaled.set_field_nullspace(
      scaled_slot, "test.field-nullspace.decorated",
      PreparedProviderOptions{"pops.field-nullspace.operator-topology-derived.options@1",
                              {{"gauge.value", 0.0}}});
  const auto scaled_output =
      install_field_outputs(scaled, "test.qualified-scaled", "potential", 1);
  scaled.register_elliptic_field("n0", "potential", scaled_output, 1);
  for (const std::string block : {std::string("n0"), std::string("n1")})
    scaled.set_block_elliptic_field(
        block, "potential", [](const NativeMultiFab& state, NativeMultiFab& rhs) {
          add_scaled_component(state, Real(1), 0, rhs);
        });
  scaled.set_density("n0", q0);
  scaled.set_density("n1", q1);
  std::vector<const NativeMultiFab*> scaled_live{&scaled.block_state(0), &scaled.block_state(1)};
  const SolveReport scaled_report =
      consume_solve_outcome(scaled.solve_fields_from_blocks(scaled_slot, scaled_live));
  ASSERT_TRUE(scaled_report.solved_value_available()) << scaled_report.status_name();
  const std::vector<double> scaled_phi = scaled.field_potential_global(scaled_slot);
  ASSERT_EQ(scaled_phi.size(), all_phi.size());
  double coefficient_error = 0.0;
  double coefficient_scale = 0.0;
  for (std::size_t cell = 0; cell < all_phi.size(); ++cell) {
    coefficient_error =
        std::fmax(coefficient_error, std::fabs(scaled_phi[cell] - 0.5 * all_phi[cell]));
    coefficient_scale = std::fmax(coefficient_scale, std::fabs(all_phi[cell]));
  }
  EXPECT_LE(coefficient_error, 1.0e-9 * std::max(1.0, coefficient_scale))
      << "qualified RHS coefficients must replace the removed formula-specific permittivity seam";

  NativeMultiFab stage1 = system.block_state(1);
  stage1.set_val(Real(0));
  std::vector<const NativeMultiFab*> override_states{&system.block_state(0), &stage1};
  const SolveReport override_report =
      consume_solve_outcome(system.solve_fields_from_blocks(slot, override_states));
  ASSERT_TRUE(override_report.solved_value_available()) << override_report.status_name();
  const std::vector<double> override_phi = system.field_potential_global(slot);

  EXPECT_GT(max_abs_diff(all_phi, override_phi), 1.0e-4)
      << "the named solve ignored the second qualified stage slot";
  EXPECT_EQ(max_abs_diff(system.density_global("n0"), q0), 0.0);
  EXPECT_EQ(max_abs_diff(system.density_global("n1"), q1), 0.0);

  std::vector<const NativeMultiFab*> bad{&system.block_state(0)};
  std::vector<const NativeMultiFab*> duplicate_stage{&system.block_state(0),
                                                      &system.block_state(0)};
  if (n_ranks() == 1) {
    EXPECT_THROW((void)consume_solve_outcome(system.solve_fields_from_blocks(slot, bad)),
                 std::invalid_argument);
  } else {
    EXPECT_THROW((void)consume_solve_outcome(system.solve_fields_from_blocks(slot, bad)),
                 std::runtime_error);
  }

  // Stage identity is the explicit vector slot, never pointer ownership. Reusing one immutable
  // image for two qualified RHS providers is valid and must remain collective; assigning block 1's
  // live image to block 0 through the single-stage API is likewise an explicit value override.
  const SolveReport duplicate_report =
      consume_solve_outcome(system.solve_fields_from_blocks(slot, duplicate_stage));
  ASSERT_TRUE(duplicate_report.solved_value_available()) << duplicate_report.status_name();
  const std::vector<double> duplicate_phi = system.field_potential_global(slot);
  const SolveReport cross_stage_report = consume_solve_outcome(
      system.solve_fields_from_state(slot, 0, system.block_state(1)));
  ASSERT_TRUE(cross_stage_report.solved_value_available()) << cross_stage_report.status_name();
  const std::vector<double> cross_stage_phi = system.field_potential_global(slot);
  EXPECT_GT(max_abs_diff(duplicate_phi, cross_stage_phi), 1.0e-4)
      << "qualified stage slots must select values without inferring block identity from an address";
  EXPECT_EQ(max_abs_diff(system.density_global("n0"), q0), 0.0);
  EXPECT_EQ(max_abs_diff(system.density_global("n1"), q1), 0.0);

  if (n_ranks() > 1) {
    const std::vector<double> accepted_potential = system.field_potential_global(slot);
    const std::vector<double> accepted_output = system.auxiliary_component(potential_key.front());
    std::vector<const NativeMultiFab*> rank_local_shape = all_live;
    if (my_rank() == 0)
      rank_local_shape.pop_back();
    EXPECT_THROW(
        (void)consume_solve_outcome(system.solve_fields_from_blocks(slot, rank_local_shape)),
        std::runtime_error)
        << "rank-local request-shape drift must fail collectively without stranding peers";
    EXPECT_EQ(max_abs_diff(system.field_potential_global(slot), accepted_potential), 0.0);
    EXPECT_EQ(max_abs_diff(system.auxiliary_component(potential_key.front()), accepted_output), 0.0);
    fail_rank_local_rhs = my_rank() == 0;
    EXPECT_THROW((void)consume_solve_outcome(system.solve_fields_from_blocks(slot, all_live)),
                 std::runtime_error)
        << "a rank-local provider callback failure must be published before prepare_rhs";
    fail_rank_local_rhs = false;
    EXPECT_EQ(max_abs_diff(system.field_potential_global(slot), accepted_potential), 0.0);
    EXPECT_EQ(max_abs_diff(system.auxiliary_component(potential_key.front()), accepted_output), 0.0);
    EXPECT_EQ(max_abs_diff(system.density_global("n0"), q0), 0.0);
    EXPECT_EQ(max_abs_diff(system.density_global("n1"), q1), 0.0);
    const SolveReport recovered =
        consume_solve_outcome(system.solve_fields_from_blocks(slot, all_live));
    EXPECT_TRUE(recovered.solved_value_available())
        << "the communicator must remain usable after both fail-closed collective paths";
  }
}

TEST(test_coupled_fieldsolve, named_gradient_output_applies_the_registered_sign) {
  const int n = 32;
  NativeSystem system(periodic_unit_config(n));
  const std::string slot = "signed-gradient-provider";
  const PreparedProviderOptions backend_options{
      "pops.system.cartesian-cg-options@1",
      {{"abs_tol", 0.0}, {"max_iterations", std::int64_t{2000}}, {"rel_tol", 1.0e-10}}};
  system.register_configured_field_solver_provider("cartesian_cg", slot, backend_options);
  system.set_field_solver_plan(slot, "test:signed-gradient-plan", "test:signed-gradient-provider",
                               "test:plasma", "plasma", "potential", {"test:plasma/potential/rhs"},
                               {"plasma"}, {"potential"}, {1.0}, slot);
  system.set_field_topology_authority(slot, "builtin_rectangular_cell_graph_v1",
                                      "test:periodic-cartesian", "test:periodic-cartesian:v1");
  install_periodic_field_boundary(system, slot);
  system.register_field_nullspace_provider(std::make_shared<DecoratedNullspaceProvider>());
  system.set_field_nullspace(
      slot, "test.field-nullspace.decorated",
      PreparedProviderOptions{"pops.field-nullspace.operator-topology-derived.options@1",
                              {{"gauge.value", 0.0}}});

  system.install_block_state_route("plasma", "test.coupled-fieldsolve/plasma/state@1");
  add_compiled_model(system, "plasma", charge_model(), "none", "rusanov", "conservative",
                     "explicit");
  const auto potential_outputs =
      install_field_outputs(system, "test.gradient-sign", "potential", kDim + 1);
  EXPECT_THROW(system.register_elliptic_field("plasma", "potential", potential_outputs, 0),
               std::invalid_argument);
  system.register_elliptic_field("plasma", "potential", potential_outputs, -1);
  system.set_block_elliptic_field(
      "plasma", "potential", [](const NativeMultiFab& state, NativeMultiFab& rhs) {
        add_scaled_component(state, Real(1), 0, rhs);
      });
  system.set_density("plasma", charge_density(n, 1.0, 0.0));

  const SolveReport report =
      consume_solve_outcome(system.solve_fields_from_state(slot, 0, system.block_state(0)));
  ASSERT_TRUE(report.solved()) << report.status_name();
  const std::vector<double> phi = system.field_potential_global(slot);
  std::array<std::vector<double>, static_cast<std::size_t>(kDim)> gradients;
  for (int axis = 0; axis < kDim; ++axis) {
    gradients[static_cast<std::size_t>(axis)] =
        system.auxiliary_component(potential_outputs[static_cast<std::size_t>(axis + 1)]);
    ASSERT_EQ(gradients[static_cast<std::size_t>(axis)].size(), phi.size());
  }
  ASSERT_EQ(phi.size(), cell_count(n));
  std::array<std::size_t, static_cast<std::size_t>(kDim)> strides{};
  strides[0] = 1;
  for (int axis = 1; axis < kDim; ++axis)
    strides[static_cast<std::size_t>(axis)] =
        strides[static_cast<std::size_t>(axis - 1)] * static_cast<std::size_t>(n);
  const double unsigned_scale = 0.5 * n;
  double error = 0.0;
  double reference = 0.0;
  double signed_observed = 0.0;
  double unsigned_reference = 0.0;
  for (std::size_t cell = 0; cell < phi.size(); ++cell) {
    for (int axis = 0; axis < kDim; ++axis) {
      const std::size_t stride = strides[static_cast<std::size_t>(axis)];
      const std::size_t coordinate = (cell / stride) % static_cast<std::size_t>(n);
      const std::size_t previous = coordinate == 0
                                       ? cell + static_cast<std::size_t>(n - 1) * stride
                                       : cell - stride;
      const std::size_t next = coordinate + 1 == static_cast<std::size_t>(n)
                                   ? cell - static_cast<std::size_t>(n - 1) * stride
                                   : cell + stride;
      const double unsigned_gradient = unsigned_scale * (phi[next] - phi[previous]);
      const double observed = gradients[static_cast<std::size_t>(axis)][cell];
      error = std::fmax(error, std::fabs(observed + unsigned_gradient));
      if (std::fabs(unsigned_gradient) > reference) {
        reference = std::fabs(unsigned_gradient);
        signed_observed = observed;
        unsigned_reference = unsigned_gradient;
      }
    }
  }
  const double epsilon = std::numeric_limits<Real>::epsilon();
  ASSERT_GT(reference, 1024.0 * epsilon) << "the signed-gradient oracle must be nontrivial";
  EXPECT_LT(signed_observed * unsigned_reference, 0.0)
      << "GradientOutput(sign=-1) must reverse the physical gradient direction";
  // Device backends may contract the multiply/divide sequence (or use an FMA), so exact host bits
  // are not a portable oracle.  Keep the allowance tied to machine precision and field scale: a
  // missing or inverted sign remains many orders of magnitude outside this bound.
  const double tolerance = 16.0 * epsilon * std::max(1.0, reference);
  EXPECT_LE(error, tolerance) << "GradientOutput(sign=-1) must publish -grad(phi)";
}
