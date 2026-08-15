// Compiled time-program LOADER path (epic ADC-399 / ADC-401 Phase 2c-i): System::install_program
// dlopens a generated problem.so and installs its compiled time Program across the ABI boundary.
//
// We compile AT RUNTIME a stub problem.so -- the role the codegen (Phase 2c-ii) will fill -- that
// exports pops_program_abi_key(), the required block-identity table, and
// pops_install_program(System* sys); the installer selects the System provider and installs the
// SAME Forward-Euler closure as the in-process test_program_runtime.
// We then sim.install_program(so) + sim.step(dt) and check bit-parity against a reference Forward-Euler
// step computed from the same primitives (solve_fields + eval_rhs + U + dt*R). This validates the
// dlopen + ABI-key guard + globally visible host seams with a locally scoped package, end to end.
//
// The runtime package is compiled with the exact compiler/Kokkos contract injected by CMake. A
// missing compiler or a package compilation failure is a test failure: this proof never self-skips.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/physics/bricks/elliptic.hpp>
#include <pops/physics/bricks/source.hpp>                // NoSource
#include <pops/physics/composition/composite.hpp>        // CompositeModel
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/program/cache_manager.hpp>
#include <pops/runtime/system.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

constexpr int kTestDimension = kNativeDimension;
using NativeSystem = System<kTestDimension>;
using NativeSystemConfig = SystemConfig<kTestDimension>;
using NativeField = MultiFab<kTestDimension>;
using GasLaw = nd::IdealGasEuler<kTestDimension>;
using GasModel = CompositeModel<GasLaw, NoSource, NoElliptic>;
constexpr double kGamma = 1.4;
constexpr int kGasComponents = GasModel::n_vars;

template <int Dim>
void install_runtime_authority(System<Dim>& system, std::string_view identity) {
  auto lane =
      std::make_shared<ExecutionLane>(ExecutionLane::duplicate_world_collectively(identity));
  system.install_prepared_boundary_execution_lane(std::move(lane));
}

std::size_t cell_count(int n) {
  std::size_t count = 1;
  for (int axis = 0; axis < kTestDimension; ++axis)
    count *= static_cast<std::size_t>(n);
  return count;
}

NativeSystemConfig native_config(int n) {
  NativeSystemConfig config;
  for (int axis = 0; axis < kTestDimension; ++axis) {
    config.shape[axis] = n;
    config.lower[axis] = Real(0);
    config.upper[axis] = Real(1);
    config.periodicity[static_cast<std::size_t>(axis)] = true;
  }
  config.boxes = {Box<kTestDimension>::from_extents(config.shape)};
  return config;
}

void fill_ic(std::vector<double>& U, int n) {
  const std::size_t nn = cell_count(n);
  const double pi = 3.14159265358979323846;
  for (std::size_t cell = 0; cell < nn; ++cell) {
    std::size_t remaining = cell;
    double mode = 1.0;
    for (int axis = 0; axis < kTestDimension; ++axis) {
      const int index = static_cast<int>(remaining % static_cast<std::size_t>(n));
      remaining /= static_cast<std::size_t>(n);
      const double coordinate = (static_cast<double>(index) + 0.5) / n;
      mode *= std::cos(2 * pi * coordinate);
    }
    const double pressure = 3.0 + 0.5 * mode;
    U[cell] = 1.0;
    for (int axis = 0; axis < kTestDimension; ++axis)
      U[static_cast<std::size_t>(axis + 1) * nn + cell] = 0.0;
    U[static_cast<std::size_t>(kTestDimension + 1) * nn + cell] = pressure / (kGamma - 1.0);
  }
}

void add_gas(NativeSystem& system) {
  GasModel model{};
  model.hyp = GasLaw::prepare(static_cast<Real>(kGamma));
  system.install_block_state_route("gas", "test:program-loader/gas/state");
  add_compiled_model(system, "gas", std::move(model), "minmod", "rusanov", "conservative",
                     "explicit", kGamma);
  system.set_poisson("charge_density", "cartesian_cg");
}

runtime::system::AuxiliaryComponentKey install_field_output(NativeSystem& system,
                                                            const std::string& owner,
                                                            const std::string& field) {
  using namespace runtime::system;
  AuxiliaryStorageShape<kTestDimension> shape;
  for (int axis = 0; axis < kTestDimension; ++axis)
    shape.halo[axis] = 1;
  AuxiliaryComponentKey key{owner, "field", field, "potential"};
  AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "field", "scalar"};
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<kTestDimension>{
      "test.field-output/" + owner + "/" + field,
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      {{key, contract, shape}},
      {}});
  system.seal_auxiliary_providers();
  return key;
}

// The generated problem.so: a Forward-Euler Program installed via ProgramContext. This is exactly the
// source the Phase 2c-ii codegen will emit (here hand-written for an autonomous C++ test). The ABI key
// is the preprocessor LITERAL (not the inline abi_key_string(), which would be interposed via RTLD).
std::string loader_source(bool include_block_identities = true, bool install_step = true,
                          bool incomplete_dt_bound = false,
                          const std::string& dynamic_boundary_slot = {},
                          bool register_history = false, bool boundary_install_throws = false) {
  // clang-format off
  std::string source = R"CPP(
#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/core/foundation/types.hpp>
#include <cstdint>
#include <stdexcept>
extern "C" const char* pops_program_abi_key() { return POPS_ABI_KEY_LITERAL; }
extern "C" const char* pops_program_route_manifest() { return pops::kRouteRegistrySignature; }
extern "C" const char* pops_program_name() { return "forward_euler_stub"; }
extern "C" int pops_program_operator_authority_count() { return 0; }
extern "C" std::uint64_t pops_program_operator_authority_word(int, int) { return 0; }
extern "C" int pops_module_operator_count() { return 1; }
extern "C" int pops_module_state_space_count() { return 1; }
extern "C" int pops_module_field_space_count() { return 0; }
extern "C" const char* pops_module_operator_owner(int) { return "gas"; }
extern "C" const char* pops_module_operator_name(int) { return "rhs"; }
extern "C" const char* pops_module_operator_kind(int) { return "local_rate"; }
extern "C" const char* pops_module_operator_signature(int) { return "(U) -> Rate(U)"; }
extern "C" const char* pops_module_operator_requirements(int) {
  return "{\"kind\":\"local_rate\"}";
}
extern "C" const char* pops_module_state_space_name(int) { return "U"; }
extern "C" const char* pops_module_state_space_owner(int) { return "gas"; }
extern "C" const char* pops_module_field_space_name(int) { return ""; }
extern "C" const char* pops_module_field_space_owner(int) { return ""; }
)CPP";
  if (include_block_identities) {
    source += R"CPP(
extern "C" int pops_program_block_count() { return 1; }
extern "C" const char* pops_program_block_name(int i) { return i == 0 ? "gas" : ""; }
)CPP";
  }
  if (incomplete_dt_bound) {
    source += R"CPP(
extern "C" bool pops_program_has_dt_bound() { return true; }
)CPP";
  }
  if (install_step) {
    source += R"CPP(
extern "C" void pops_install_program(pops::System<pops::kNativeDimension>* sys) {
  pops::runtime::program::ProgramContext<pops::kNativeDimension> ctx(sys);
)CPP";
    if (register_history) {
      source += R"CPP(
  ctx.register_history("artifact.history", 1, 1, 0, "test:artifact/state",
                       "test:artifact/space", "clock.macro", "test:artifact/interp");
)CPP";
    }
    source += R"CPP(
  ctx.configure_primary_clock("clock.macro");
  ctx.install([ctx](double dt) {
    ctx.begin_step(dt);
    ctx.set_stage_time(0, 1);
    auto field_outcome = ctx.solve_fields();
    (void)field_outcome.consume(pops::SolveConsumption::kAccept);
    for (int b = 0; b < ctx.n_blocks(); ++b) {
      pops::MultiFab<pops::kNativeDimension>& U = ctx.state(b);
      pops::MultiFab<pops::kNativeDimension> R = ctx.rhs_scratch_like(U);
      ctx.rhs_into(b, U, R, 0);
      ctx.axpy(U, static_cast<pops::Real>(dt), R);
    }
  });
}
)CPP";
  } else {
    source += R"CPP(
extern "C" void pops_install_program(pops::System<pops::kNativeDimension>* sys) {
  pops::runtime::program::ProgramContext<pops::kNativeDimension> ctx(sys);
  ctx.register_history("poison", 1, 1, 0, "test:poison/state", "test:poison/space",
                       "clock.macro", "test:poison/interp");
}
)CPP";
  }
  if (!dynamic_boundary_slot.empty()) {
    source += R"CPP(
namespace {
void prepare_boundary_residual(
    int, const pops::MultiFab<pops::kNativeDimension>&,
    pops::MultiFab<pops::kNativeDimension>&, const pops::Geometry<pops::kNativeDimension>&,
    const pops::FieldBoundaryExecutionContext<pops::kNativeDimension>&) {}
void prepare_boundary_jvp(
    int, const pops::MultiFab<pops::kNativeDimension>&,
    const pops::MultiFab<pops::kNativeDimension>&,
    pops::MultiFab<pops::kNativeDimension>&, const pops::Geometry<pops::kNativeDimension>&,
    const pops::FieldBoundaryExecutionContext<pops::kNativeDimension>&) {}
void add_boundary_residual(
    int, const pops::MultiFab<pops::kNativeDimension>&,
    pops::MultiFab<pops::kNativeDimension>&, const pops::Geometry<pops::kNativeDimension>&,
    const pops::FieldBoundaryExecutionContext<pops::kNativeDimension>&) {}
void add_boundary_jvp(
    int, const pops::MultiFab<pops::kNativeDimension>&,
    const pops::MultiFab<pops::kNativeDimension>&,
    pops::MultiFab<pops::kNativeDimension>&, const pops::Geometry<pops::kNativeDimension>&,
    const pops::FieldBoundaryExecutionContext<pops::kNativeDimension>&) {}
}  // namespace
extern "C" void pops_install_field_boundaries(pops::System<pops::kNativeDimension>* sys) {
  pops::runtime::program::ProgramContext<pops::kNativeDimension> ctx(sys);
  ctx.set_field_boundary_kernel(
)CPP";
    source += "\"" + dynamic_boundary_slot + "\"";
    source += R"CPP(,
      pops::CompiledFieldBoundaryKernel<pops::kNativeDimension>{
          "test:program-boundary", "test:program-boundary-residual",
          "test:program-boundary-jvp", prepare_boundary_residual,
          prepare_boundary_jvp, add_boundary_residual, add_boundary_jvp, false});
)CPP";
    if (boundary_install_throws) {
      source += "  ctx.set_field_logical_timepoint(\"" + dynamic_boundary_slot +
                "\", pops::FieldLogicalTimePoint{0.375, 0.025, 2, 1, 3, 0, 7, 1, 4});\n";
      source += "  ctx.set_field_boundary_parameters(\"" + dynamic_boundary_slot +
                "\", std::vector<double>{0.125, 0.25});\n";
      source += R"CPP(
  throw std::runtime_error("injected boundary publication failure");
)CPP";
    }
    source += R"CPP(
}
)CPP";
  }
  // clang-format on
  return source;
}

}  // namespace

static int pops_run_test_program_loader(int argc, char** argv) {
  (void)argc;
  (void)argv;

  const int n = 16;
  const double dt = 1e-3;
  const std::size_t nn = cell_count(n);
  std::vector<double> U0(static_cast<std::size_t>(kGasComponents) * nn);
  fill_ic(U0, n);

  const NativeSystemConfig cfg = native_config(n);

  // Reference: one Forward-Euler step via the existing primitives, combined on the host.
  NativeSystem ref(cfg);
  install_runtime_authority(ref, "test.program-loader/runtime-reference@1");
  add_gas(ref);
  ref.set_state("gas", U0);
  (void)pops::consume_solve_outcome(ref.solve_fields());
  const std::vector<double> R0 = ref.eval_rhs("gas");
  std::vector<double> Uref(static_cast<std::size_t>(kGasComponents) * nn);
  for (std::size_t k = 0; k < Uref.size(); ++k)
    Uref[k] = U0[k] + dt * R0[k];

  // Compile the stub problem.so and load it via System::install_program.
  const std::string tmp = std::string(POPS_TEST_TMPDIR) + "/program_loader_" +
                          std::to_string(static_cast<long>(std::clock()));
  const std::string src = tmp + ".cpp";
  const std::string so = tmp + ".so";
  const std::string legacy_src = tmp + "_missing_block_identities.cpp";
  const std::string legacy_so = tmp + "_missing_block_identities.so";
  const std::string no_op_src = tmp + "_no_op_installer.cpp";
  const std::string no_op_so = tmp + "_no_op_installer.so";
  const std::string incomplete_dt_src = tmp + "_incomplete_dt.cpp";
  const std::string incomplete_dt_so = tmp + "_incomplete_dt.so";
  const std::string dynamic_boundary_src = tmp + "_dynamic_boundary.cpp";
  const std::string dynamic_boundary_so = tmp + "_dynamic_boundary.so";
  const std::string failing_boundary_src = tmp + "_failing_boundary.cpp";
  const std::string failing_boundary_so = tmp + "_failing_boundary.so";
  {
    std::ofstream f(src);
    f << loader_source();
  }
  {
    std::ofstream f(legacy_src);
    f << loader_source(false);
  }
  {
    std::ofstream f(no_op_src);
    f << loader_source(true, false);
  }
  {
    std::ofstream f(incomplete_dt_src);
    f << loader_source(true, true, true);
  }
  {
    std::ofstream f(dynamic_boundary_src);
    f << loader_source(true, true, false, "program-boundary-field", true);
  }
  {
    std::ofstream f(failing_boundary_src);
    f << loader_source(true, true, false, "program-boundary-field", true, true);
  }
  const auto package = pops::test::native_dso::compile_shared(src, so);
  if (!package.ok) {
    pops::test::native_dso::report_compile_failure("test_program_loader", package);
    return 1;
  }
  const auto legacy_package = pops::test::native_dso::compile_shared(legacy_src, legacy_so);
  if (!legacy_package.ok) {
    pops::test::native_dso::report_compile_failure("test_program_loader legacy package",
                                                   legacy_package);
    return 1;
  }
  const auto no_op_package = pops::test::native_dso::compile_shared(no_op_src, no_op_so);
  if (!no_op_package.ok) {
    pops::test::native_dso::report_compile_failure("test_program_loader no-op package",
                                                   no_op_package);
    return 1;
  }
  const auto incomplete_dt_package =
      pops::test::native_dso::compile_shared(incomplete_dt_src, incomplete_dt_so);
  if (!incomplete_dt_package.ok) {
    pops::test::native_dso::report_compile_failure("test_program_loader incomplete-dt package",
                                                   incomplete_dt_package);
    return 1;
  }
  const auto dynamic_boundary_package =
      pops::test::native_dso::compile_shared(dynamic_boundary_src, dynamic_boundary_so);
  if (!dynamic_boundary_package.ok) {
    pops::test::native_dso::report_compile_failure("test_program_loader dynamic-boundary package",
                                                   dynamic_boundary_package);
    return 1;
  }
  const auto failing_boundary_package =
      pops::test::native_dso::compile_shared(failing_boundary_src, failing_boundary_so);
  if (!failing_boundary_package.ok) {
    pops::test::native_dso::report_compile_failure("test_program_loader failing-boundary package",
                                                   failing_boundary_package);
    return 1;
  }

  int fails = 0;
  // A pre-spec library with no explicit block identity table must never install by add-order. The
  // old positional fallback could silently bind the right equations to the wrong instances.
  NativeSystem missing_identity(cfg);
  install_runtime_authority(missing_identity, "test.program-loader/runtime-missing-identity@1");
  add_gas(missing_identity);
  try {
    missing_identity.install_program(legacy_so);
    std::printf("FAIL Program without a block identity table installed positionally\n");
    ++fails;
  } catch (const std::runtime_error& e) {
    const std::string message = e.what();
    if (message.find("block identity table") == std::string::npos ||
        message.find("pops_program_block_count") == std::string::npos ||
        message.find("pops_program_block_name") == std::string::npos ||
        message.find("Positional") == std::string::npos) {
      std::printf("FAIL missing block identity table diagnostic: %s\n", message.c_str());
      ++fails;
    }
  }

  // A prelude-only installer that registers a history but omits the Program step must not inherit its
  // candidate block map/history or replace an already usable direct step. The loader's generation
  // witness fails and restores the exact image.
  NativeSystem no_op(cfg);
  install_runtime_authority(no_op, "test.program-loader/runtime-no-op@1");
  add_gas(no_op);
  no_op.install_program_step([](double) {});
  no_op.program_cache().store(7, no_op.block_state(0), 0, "kept-cache");
  no_op.record_program_diagnostic("kept-diagnostic", Real(2.5));
  const auto histories_before_no_op = no_op.history_names();
  try {
    no_op.install_program(no_op_so);
    std::printf("FAIL no-op artifact installer was accepted\\n");
    ++fails;
  } catch (const std::runtime_error& e) {
    if (std::string(e.what()).find("exactly one new unverified") == std::string::npos) {
      std::printf("FAIL no-op installer diagnostic: %s\\n", e.what());
      ++fails;
    }
  }
  if (!no_op.program_block_map().empty()) {
    std::printf("FAIL no-op installer leaked its candidate block map\\n");
    ++fails;
  }
  if (no_op.history_names() != histories_before_no_op) {
    std::printf("FAIL no-op installer leaked its candidate history ring\\n");
    ++fails;
  }
  if (no_op.program_cache_nodes() != std::vector<int>{7} ||
      no_op.program_diagnostics() != std::map<std::string, Real>{{"kept-diagnostic", Real(2.5)}}) {
    std::printf("FAIL prelude-only installer did not restore cache/diagnostics\\n");
    ++fails;
  }
  try {
    no_op.step(dt);
  } catch (const std::exception& e) {
    std::printf("FAIL no-op installer did not restore the prior Program: %s\\n", e.what());
    ++fails;
  }

  // A declared-but-missing dt-bound entry is rejected before any candidate facade state is
  // installed. Falling back to the native CFL would silently change the authored numerics.
  NativeSystem incomplete_dt(cfg);
  install_runtime_authority(incomplete_dt, "test.program-loader/runtime-incomplete-dt@1");
  add_gas(incomplete_dt);
  incomplete_dt.install_program_step([](double) {});
  try {
    incomplete_dt.install_program(incomplete_dt_so);
    std::printf("FAIL incomplete dt-bound artifact was accepted\\n");
    ++fails;
  } catch (const std::runtime_error& e) {
    const std::string message = e.what();
    if (message.find("declares a dt bound") == std::string::npos ||
        message.find("pops_program_dt_bound") == std::string::npos) {
      std::printf("FAIL incomplete dt-bound diagnostic: %s\\n", message.c_str());
      ++fails;
    }
  }
  if (!incomplete_dt.program_block_map().empty()) {
    std::printf("FAIL incomplete dt-bound preflight mutated the block map\\n");
    ++fails;
  }

  // Program-owned field-boundary kernels are an artifact overlay, not durable System authoring.
  // Replacing artifact A (dynamic boundary export) with artifact B (no export) must therefore
  // restore the static baseline while retaining the exact configured backend route.
  {
    NativeSystem replacement(cfg);
    install_runtime_authority(replacement, "test.program-loader/runtime-replacement@1");
    add_gas(replacement);
    constexpr const char* slot = "program-boundary-field";
    constexpr const char* backend = "program-boundary-cartesian-cg";
    replacement.register_configured_field_solver_provider(
        "cartesian_cg", backend,
        PreparedProviderOptions{
            "pops.system.cartesian-cg-options@1",
            {{"abs_tol", 0.0}, {"max_iterations", std::int64_t{200}}, {"rel_tol", 1.0e-8}}});
    replacement.set_field_solver_plan(
        slot, "test:program-boundary-plan", "test:program-boundary-provider", "test:gas", "gas",
        "program-boundary-potential", {"test:gas/program-boundary-rhs"}, {"gas"},
        {"program-boundary-potential"}, {1.0}, backend);
    replacement.set_field_topology_authority(slot, "builtin_rectangular_cell_graph_v1",
                                             "test:periodic-cartesian",
                                             "test:periodic-cartesian:v1");
    const std::vector<std::string> periodic_faces(static_cast<std::size_t>(2 * kTestDimension),
                                                  "periodic");
    const std::vector<double> zero_faces(static_cast<std::size_t>(2 * kTestDimension), 0.0);
    replacement.set_field_boundary_plan(slot, periodic_faces, zero_faces, zero_faces, zero_faces);
    const auto field_output =
        install_field_output(replacement, "test.program-boundary", "program-boundary-potential");
    replacement.register_elliptic_field("gas", "program-boundary-potential", {field_output}, 1);
    replacement.set_block_elliptic_field("gas", "program-boundary-potential",
                                         [](const NativeField&, NativeField&) {});
    replacement.set_state("gas", U0);

    replacement.install_program(dynamic_boundary_so);
    if (replacement.history_names() != std::vector<std::string>{"artifact.history"}) {
      std::printf("FAIL artifact A did not materialize its qualified history\\n");
      ++fails;
    }
    replacement.program_cache().store(11, replacement.block_state(0), 0, "artifact-A-cache");
    replacement.record_program_diagnostic("artifact-A-diagnostic", Real(1));
    try {
      const SolveReport report = consume_solve_outcome(
          replacement.solve_fields_from_state(slot, 0, replacement.block_state(0)));
      if (!report.solved()) {
        std::printf("FAIL dynamic-boundary field solve returned %s\\n", report.status_name());
        ++fails;
      }
    } catch (const std::exception& e) {
      std::printf("FAIL dynamic-boundary artifact was not executable: %s\\n", e.what());
      ++fails;
    }

    // A later artifact that stages a valid overlay and then throws must publish neither its
    // boundary nor its Program/history/cache image. Artifact A remains the accepted owner and its
    // function pointers remain backed by the still-live accepted DSO.
    try {
      replacement.install_program(failing_boundary_so);
      std::printf("FAIL partially failing boundary artifact was accepted\\n");
      ++fails;
    } catch (const std::runtime_error& e) {
      if (std::string(e.what()).find("injected boundary publication failure") ==
          std::string::npos) {
        std::printf("FAIL partial boundary rollback diagnostic: %s\\n", e.what());
        ++fails;
      }
    }
    if (replacement.history_names() != std::vector<std::string>{"artifact.history"} ||
        replacement.program_cache_nodes() != std::vector<int>{11} ||
        replacement.program_diagnostics() !=
            std::map<std::string, Real>{{"artifact-A-diagnostic", Real(1)}}) {
      std::printf("FAIL partial boundary artifact mutated accepted Program state\\n");
      ++fails;
    }
    try {
      const SolveReport report = consume_solve_outcome(
          replacement.solve_fields_from_state(slot, 0, replacement.block_state(0)));
      if (!report.solved()) {
        std::printf("FAIL accepted boundary after rollback returned %s\\n", report.status_name());
        ++fails;
      }
    } catch (const std::exception& e) {
      std::printf("FAIL accepted boundary was lost after rollback: %s\\n", e.what());
      ++fails;
    }

    replacement.install_program(so);
    if (!replacement.history_names().empty() || !replacement.program_cache_nodes().empty() ||
        !replacement.program_diagnostics().empty()) {
      std::printf("FAIL artifact B retained Program-owned state from artifact A\\n");
      ++fails;
    }
    try {
      const SolveReport report = consume_solve_outcome(
          replacement.solve_fields_from_state(slot, 0, replacement.block_state(0)));
      if (!report.solved()) {
        std::printf("FAIL static replacement field solve returned %s\\n", report.status_name());
        ++fails;
      }
    } catch (const std::exception& e) {
      std::printf("FAIL static artifact inherited A's dynamic boundary: %s\\n", e.what());
      ++fails;
    }
  }

  NativeSystem sim(cfg);
  install_runtime_authority(sim, "test.program-loader/runtime-simulation@1");
  add_gas(sim);
  sim.set_state("gas", U0);
  sim.install_program(so);  // dlopen + ABI check + pops_install_program(this)
  const int step0 = sim.macro_step();
  sim.step(dt);  // The exact-ranked System facade dispatches to the installed Program.
  const std::vector<double> Up = sim.get_state("gas");

  double err = 0, change = 0;
  for (std::size_t k = 0; k < Up.size(); ++k) {
    err = std::fmax(err, std::fabs(Up[k] - Uref[k]));
    change = std::fmax(change, std::fabs(Up[k] - U0[k]));
  }
  if (!(err < 1e-12)) {
    std::printf("FAIL parity: max|Up - Uref| = %.3e\n", err);
    ++fails;
  }
  if (sim.macro_step() != step0 + 1) {
    std::printf("FAIL macro_step not advanced (%d -> %d)\n", step0, sim.macro_step());
    ++fails;
  }
  if (!(change > 1e-9)) {
    std::printf("FAIL loaded program did not change the state (change = %.3e)\n", change);
    ++fails;
  }

  if (fails == 0)
    std::printf(
        "OK test_program_loader (problem.so Forward Euler via install_program == reference; "
        "max|d| = %.2e, change = %.2e)\n",
        err, change);
  return fails ? 1 : 0;
}

TEST(test_program_loader, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_program_loader, "test_program_loader"), 0);
}
