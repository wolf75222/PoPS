#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"
#include <pops/runtime/program/step_transaction.hpp>
#include <pops/runtime/system.hpp>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string package_source() {
  return R"CPP(
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

struct DecayingScalar {
  using State = pops::StateVec<1>;
  using Prim = pops::StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;

  POPS_HD State flux(const State&, const Aux&, int) const { return State{}; }
  POPS_HD pops::Real max_wave_speed(const State&, const Aux&, int) const {
    return pops::Real(1);
  }
  POPS_HD State source(const State& state, const Aux&) const { return State{-state[0]}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
  POPS_HD Prim to_primitive(const State& state) const { return state; }
  POPS_HD State to_conservative(const Prim& primitive) const { return primitive; }

  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"u"}, 1, {pops::VariableRole::Custom}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"u"}, 1, {pops::VariableRole::Custom}};
  }
};

template <pops::EvaluationStatus Status, std::uint32_t Reason>
struct AttemptControlFlux {
  template <pops::PhysicalFlux Physical>
  POPS_HD pops::FluxEvaluation<typename Physical::State> operator()(
      const Physical&, const typename Physical::Trace& left,
      const typename Physical::Trace&, const pops::FaceContext&) const {
    // With dt=0.2 and two SSPRK2 substeps, u'= -u advances the first accepted substep
    // 1 -> 0.905.  Its two stage inputs are 1 and 0.9, so only the next substep enters this window.
    // The runtime has therefore mutated the live state before this loader requests rollback.
    if (left.state[0] > pops::Real(0.902) && left.state[0] < pops::Real(0.908)) {
      if constexpr (Status == pops::EvaluationStatus::kRetry)
        return pops::FluxEvaluation<typename Physical::State>::retry(Reason);
      else
        return pops::FluxEvaluation<typename Physical::State>::reject(Reason);
    }
    return pops::FluxEvaluation<typename Physical::State>::ok(
        typename Physical::State{},
        {pops::Real(1), pops::StabilityUnit::kLengthPerTime,
         pops::StabilityConvention::kNormalSpectralRadius});
  }
};

using RetryFlux = AttemptControlFlux<pops::EvaluationStatus::kRetry, 0x52545259u>;
using RejectFlux = AttemptControlFlux<pops::EvaluationStatus::kReject, 0x524a4354u>;

template <class Flux>
void install_attempt_block(pops::System& system, const char* name, int substeps, bool evolve,
                           int stride) {
  const DecayingScalar model{};
  const pops::GridContext context = system.grid_context(name);
  pops::BlockClosures closures = pops::build_block<pops::NoSlope, Flux>(
      model, context, /*imex=*/false, /*recon_prim=*/false, "explicit");
  system.install_block(name, DecayingScalar::n_vars, DecayingScalar::conservative_vars(),
                       DecayingScalar::primitive_vars(), 1.4, std::move(closures),
                       pops::make_max_speed(model, context), pops::make_poisson_rhs(model),
                       substeps, evolve, stride);
  auto conversion = pops::make_cell_convert(model);
  system.set_block_conversion(name, std::move(conversion.first), std::move(conversion.second));
  system.set_block_ghosts(name, pops::NoSlope::n_ghost);
}

extern "C" const char* pops_native_abi_key() { return POPS_ABI_KEY_LITERAL; }
extern "C" const char* pops_compiled_route_manifest() { return pops::kRouteRegistrySignature; }
extern "C" int pops_compiled_nparams() { return 1; }
extern "C" const char* pops_compiled_param_names() { return "attempt_mode"; }

extern "C" void pops_install_native(void* raw, const char* name, const char* limiter,
                                    const char*, const char* recon, const char* time, double,
                                    int substeps, int evolve, int stride, const double* params,
                                    int nparams, double) {
  if (std::string(limiter) != "none" || std::string(recon) != "conservative" ||
      std::string(time) != "explicit" || params == nullptr || nparams != 1)
    throw std::invalid_argument("attempt-control test package received an invalid route");
  auto& system = *reinterpret_cast<pops::System*>(raw);
  if (params[0] == 0.0)
    install_attempt_block<RetryFlux>(system, name, substeps, evolve != 0, stride);
  else
    install_attempt_block<RejectFlux>(system, name, substeps, evolve != 0, stride);
}
)CPP";
}

int exercise_attempt(const std::string& library, double mode,
                     pops::runtime::program::StepAttemptDisposition expected_disposition,
                     std::uint32_t expected_reason) {
  constexpr int n = 8;
  pops::SystemConfig config;
  config.n = n;
  config.L = 1.0;
  config.periodicity = {true, true};
  pops::System system(config);
  system.add_native_block("scalar", library, "none", "rusanov", "conservative", "explicit",
                          1.4, /*substeps=*/2, true, 1, {mode});
  const std::vector<double> accepted(static_cast<std::size_t>(n) * n, 1.0);
  system.set_state("scalar", accepted);

  bool caught = false;
  try {
    system.step(0.2);
  } catch (const pops::runtime::program::StepAttemptRejected& rejected) {
    caught = rejected.status() == pops::SolveStatus::kInvalidEvaluation &&
             rejected.disposition() == expected_disposition &&
             rejected.reason_code() == expected_reason;
  } catch (...) {
    return 1;  // FluxEvaluationFailure must not escape the host transaction boundary.
  }
  if (!caught || system.time() != 0.0 || system.get_state("scalar") != accepted)
    return 1;
  return 0;
}

int run_flux_failure_loader_transaction() {
  const std::string stem = std::string(POPS_TEST_TMPDIR) + "/flux_failure_loader_" +
                           std::to_string(static_cast<long>(std::clock()));
  const std::string source = stem + ".cpp";
  const std::string library = stem + ".so";
  {
    std::ofstream output(source);
    output << package_source();
  }
  const auto package = pops::test::native_dso::compile_shared(
      source, library, "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI");
  if (!package.ok) {
    pops::test::native_dso::report_compile_failure("test_flux_failure_loader_transaction",
                                                   package);
    return 1;
  }

  int failures = 0;
  failures += exercise_attempt(library, 0.0,
                               pops::runtime::program::StepAttemptDisposition::kRetry,
                               0x52545259u);
  failures += exercise_attempt(library, 1.0,
                               pops::runtime::program::StepAttemptDisposition::kReject,
                               0x524a4354u);
  std::remove(source.c_str());
  std::remove(library.c_str());
  std::remove((library + ".log").c_str());
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_flux_failure_loader_transaction, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&run_flux_failure_loader_transaction,
                                    "test_flux_failure_loader_transaction"),
            0);
}
