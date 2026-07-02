// Capacity audit: of the four fixed capacities bounding a CoupledSourceProgram bytecode term
// (include/pops/coupling/source/coupled_source_program.hpp:42-45 -- kCsMaxReg, kCsMaxStack, kCsMaxProg,
// kCsMaxTerms), kCsMaxReg / kCsMaxProg / kCsMaxTerms are already exercised as OVERFLOW checks against the
// C++ guard reachable from the public path: System::add_coupled_source
// (python/bindings/system/base/system.cpp) throws EXPLICITLY when in_blocks+consts > kCsMaxReg,
// out_blocks > kCsMaxTerms, or a term's prog_lens > kCsMaxProg -- BEFORE calling
// validate_cs_program_stack. kCsMaxStack had no equivalent OVERFLOW test: only the MALFORMED-program
// cases (underflow, leftover stack) are covered by test_public_validation_errors.cpp, calling
// validate_cs_program_stack directly rather than through the public add_coupled_source path.
//
// This closes that gap at the HONEST level: validate_cs_program_stack (coupled_source_program.hpp:171)
// IS a real C++ guard reachable through the public API -- System::add_coupled_source calls it for every
// term (system.cpp, "System::add_coupled_source term " + index) BEFORE the coupling is registered, so a
// program whose postfix stack would exceed kCsMaxStack is rejected with an EXPLICIT std::runtime_error,
// never reaching the device kernel. We build such a program (kCsMaxStack + 1 consecutive PushReg with no
// operator to drain the stack) through the SAME public entry point a real coupling would use
// (System::add_coupled_source, not a direct call to validate_cs_program_stack) and assert the throw,
// naming the capacity in the message.
//
// A program AT the capacity (exactly kCsMaxStack PushReg, still short of a valid single-result program --
// it would fail the "leaves exactly one result" check first) is not what we probe here: the boundary
// condition of PushReg's OWN bound (sp < kCsMaxStack before incrementing) already means kCsMaxStack pushes
// are accepted and the (kCsMaxStack+1)-th is rejected; that is the overflow this test locks.
#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "test_harness.hpp"  // pops::test::Checker, raises
#include <pops/coupling/source/coupled_source_program.hpp>  // CsOp, kCsMaxStack
#include <pops/physics/bricks/hyperbolic.hpp>  // ExBVelocity (scalar 1-var block, role Density)
#include <pops/physics/bricks/source.hpp>      // NoSource
#include <pops/physics/composition/composite.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/facade_options.hpp>               // CoupledSourceProgram
#include <pops/runtime/system.hpp>

#include <cstdio>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

struct NoEll {
  template <class State>
  POPS_HD Real rhs(const State&) const {
    return Real(0);
  }
};
using Dens = CompositeModel<ExBVelocity, NoSource, NoEll>;  // scalar density block, role "density"

// Builds a CoupledSourceProgram with ONE input register (block "a" / role "density") and a single
// output term whose postfix program is @p n_pushes consecutive PushReg(0) opcodes (no operator to drain
// the stack): a well-formed source program never does this (it always leaves exactly one result), but a
// generated / hand-built program that DOES is exactly the shape the overflow guard exists to reject.
CoupledSourceProgram all_pushes_program(int n_pushes) {
  CoupledSourceProgram prog;
  prog.in_blocks = {"a"};
  prog.in_roles = {"density"};
  prog.out_blocks = {"a"};
  prog.out_roles = {"density"};
  const int push = static_cast<int>(CsOp::PushReg);
  std::vector<int> ops(static_cast<std::size_t>(n_pushes), push);
  std::vector<int> args(static_cast<std::size_t>(n_pushes), 0);  // always read register 0 (the input)
  prog.prog_ops = ops;
  prog.prog_args = args;
  prog.prog_lens = {n_pushes};
  return prog;
}

}  // namespace

static int pops_run_test_coupled_source_stack_limit(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  pops::test::Checker chk;

  const int n = 8;
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;

  // (1) at EXACTLY kCsMaxStack consecutive PushReg (no operator), validate_cs_program_stack's own
  // "leaves exactly one result" check rejects it FIRST (sp == kCsMaxStack != 1) -- so this shape never
  // reaches a device kernel either way, but the diagnostic is the SHAPE check, not the stack-overflow
  // check. We assert it is rejected (for completeness) without asserting which specific message it is.
  {
    System sys(cfg);
    add_compiled_model(sys, "a", Dens{}, "none", "rusanov", "conservative", "explicit");
    sys.set_poisson("charge_density", "geometric_mg");
    sys.set_density("a", std::vector<double>(static_cast<std::size_t>(n) * n, 1.0));
    const CoupledSourceProgram prog = all_pushes_program(kCsMaxStack);
    chk(pops::test::raises([&] { sys.add_coupled_source(prog); }),
        "exactly kCsMaxStack pushes (no operator) rejected (malformed program)");
  }

  // (2) kCsMaxStack + 1 consecutive PushReg: the (kCsMaxStack+1)-th push finds sp == kCsMaxStack BEFORE
  // incrementing -> validate_cs_program_stack's OVERFLOW branch (coupled_source_program.hpp:190-193)
  // throws EXPLICITLY, reached through the PUBLIC path System::add_coupled_source (system.cpp:1649),
  // never through a direct call to validate_cs_program_stack. This is the capacity this test locks.
  {
    System sys(cfg);
    add_compiled_model(sys, "a", Dens{}, "none", "rusanov", "conservative", "explicit");
    sys.set_poisson("charge_density", "geometric_mg");
    sys.set_density("a", std::vector<double>(static_cast<std::size_t>(n) * n, 1.0));
    const CoupledSourceProgram prog = all_pushes_program(kCsMaxStack + 1);
    bool raised = false;
    std::string what;
    try {
      sys.add_coupled_source(prog);
    } catch (const std::runtime_error& e) {
      raised = true;
      what = e.what();
    }
    chk(raised, "kCsMaxStack + 1 pushes: System::add_coupled_source throws std::runtime_error");
    chk(what.find("postfix stack depth") != std::string::npos,
        "overflow message names the postfix stack depth capacity");
    chk(what.find(std::to_string(kCsMaxStack)) != std::string::npos,
        "overflow message names the kCsMaxStack bound");
    chk(what.find("stack overflow") != std::string::npos,
        "overflow message says 'stack overflow' (not underflow/leftover)");
    std::printf("  message: %s\n", what.c_str());
  }

  if (chk.fails() == 0)
    std::printf(
        "OK test_coupled_source_stack_limit (kCsMaxStack overflow rejected by "
        "System::add_coupled_source, the public path validate_cs_program_stack guards)\n");
  return chk.failed();
}

TEST(test_coupled_source_stack_limit, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_coupled_source_stack_limit,
                                    "test_coupled_source_stack_limit"),
            0);
}
