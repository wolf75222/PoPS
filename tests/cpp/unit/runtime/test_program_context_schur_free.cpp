// ADC-587 -- program_context.hpp is self-contained and Schur/Lorentz-free (compile-fire).
//
// The Phase-4 refactor split the condensed-Schur / Lorentz operator out of the generic runtime facade
// include/pops/runtime/program/program_context.hpp into include/pops/coupling/schur/program/. This TU
// includes ONLY program_context.hpp: it must compile on its own (the facade no longer depends on the
// Schur condensation / geometric multigrid / Lorentz eliminator headers it used to pull in), proving a
// generated Schur-free problem.so -- which includes program_context.hpp and nothing under
// coupling/schur/** -- still builds. The source-parse architecture gate
// (tests/python/architecture/test_no_schur_header_leak.py) pins the token / include hygiene; this test
// pins that the trimmed facade is a COMPLETE, buildable translation unit by itself.
//
// A named-check twist: pops::runtime::program is in scope (ProgramContext lives there), but the Schur
// operator lives in the SEPARATE namespace pops::coupling::schur::program, which program_context.hpp
// does not declare -- so a use of it here would fail to compile. We therefore only touch the facade
// type, and rely on the include-graph gate for the negative.

#include <gtest/gtest.h>

#include <pops/runtime/program/program_context.hpp>

#include <type_traits>

// The facade type is complete and usable from program_context.hpp alone (a self-contained TU). If the
// header had lost a needed include when the Schur material moved out, this static_assert would not
// compile -- the compile-fire guarantee.
static_assert(std::is_class<pops::runtime::program::ProgramContext>::value,
              "ProgramContext must be a complete class type from program_context.hpp alone");

// ProgramContext holds a System* and forwards to public System accessors; it is NOT trivially
// constructible (its only constructors take a System* / void*), which pins that the trimmed facade
// still carries its seam constructor after the Schur split.
static_assert(!std::is_trivially_constructible<pops::runtime::program::ProgramContext>::value,
              "ProgramContext keeps its System-wrapping constructor after the split");

TEST(ProgramContextSchurFree, HeaderIsSelfContainedAndBuilds) {
  // Reaching this TEST means program_context.hpp compiled standalone (no Schur/MG/Lorentz headers).
  // Constructing a ProgramContext from a null System* is well-defined here: we never dereference it,
  // we only exercise that the type is instantiable from the facade header by itself.
  pops::runtime::program::ProgramContext ctx(static_cast<void*>(nullptr));
  (void)ctx;
  SUCCEED() << "program_context.hpp builds without any coupling/schur/** dependency";
}
