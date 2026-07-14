// ADC-578: the typed System runtime registries + lifecycle state machine extracted out of the
// System::Impl god-object. This suite pins two acceptance properties WITHOUT building a full System:
//
//   1. SystemLifecycle -- the typed freeze state machine that replaced `bool bound_`. It PRESERVES the
//      historical observable strings ("assembling"/"bound"/"running") for every pre-existing call
//      sequence, and the NEW checkpointed / finalized states + refusals are reachable only through the
//      new explicit transitions (no current caller). The inverted refusals are argued here.
//   2. The structured reports (options_report / layout_report / newton_report_ptr) the
//      registries expose for a runtime report -- the ADC-578 "define ownerships + structured reports"
//      acceptance.
#include <gtest/gtest.h>

#include <memory>
#include <string>

#include <pops/runtime/system/system_lifecycle.hpp>
#include <pops/runtime/system/system_diagnostics_registry.hpp>
#include <pops/runtime/system/system_coupling_registry.hpp>
#include <pops/runtime/system/system_domain.hpp>

namespace {

using pops::runtime::system::LifecyclePhase;
using pops::runtime::system::SystemLifecycle;

// --- 1. Lifecycle: the historical three strings are preserved bit-for-bit ------------------------
TEST(SystemLifecycle, PreservesHistoricalObservableStrings) {
  SystemLifecycle lc;
  EXPECT_FALSE(lc.frozen());
  EXPECT_EQ(lc.state(0), "assembling");
  EXPECT_EQ(lc.state(5), "assembling") << "assembling ignores the macro-step counter";

  lc.to_bound();
  EXPECT_TRUE(lc.frozen());
  EXPECT_EQ(lc.state(0), "bound") << "bound with no macro-step advanced";
  EXPECT_EQ(lc.state(1), "running") << "running is DERIVED from the macro-step > 0";
  EXPECT_EQ(lc.state(42), "running");
}

TEST(SystemLifecycle, DoubleBindThrowsTheSameMessage) {
  SystemLifecycle lc;
  lc.to_bound();
  // Same message text the old `bool bound_` guard raised in System::mark_bound.
  EXPECT_THROW(
      {
        try {
          lc.to_bound();
        } catch (const std::runtime_error& e) {
          EXPECT_NE(std::string(e.what()).find("already bound"), std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

// --- 1b. NEW states: reachable only through the new transitions -----------------------------------
TEST(SystemLifecycle, CheckpointedIsInformationalAndReversible) {
  SystemLifecycle lc;
  lc.to_bound();
  lc.to_checkpointed();
  EXPECT_EQ(lc.state(0), "checkpointed") << "checkpointed surfaces ONLY after the explicit mark";
  EXPECT_TRUE(lc.frozen()) << "checkpointed is still frozen for structural setters";
}

TEST(SystemLifecycle, FinalizedIsTerminalAndSupersetOfBoundForRefusals) {
  SystemLifecycle lc;
  lc.to_bound();
  lc.to_finalized();
  EXPECT_EQ(lc.state(0), "finalized");
  EXPECT_EQ(lc.state(9), "finalized") << "finalized ignores the macro-step counter (terminal)";
  EXPECT_TRUE(lc.frozen()) << "a structural setter after finalize is refused (superset of bound)";
}

TEST(SystemLifecycle, InvertedRefusals) {
  // (a) finalize before bind is refused.
  {
    SystemLifecycle lc;
    EXPECT_THROW(lc.to_finalized(), std::runtime_error);
  }
  // (b) double-finalize is refused.
  {
    SystemLifecycle lc;
    lc.to_bound();
    lc.to_finalized();
    EXPECT_THROW(lc.to_finalized(), std::runtime_error);
  }
  // (c) to_bound after finalize is refused (the terminal state cannot re-bind).
  {
    SystemLifecycle lc;
    lc.to_bound();
    lc.to_finalized();
    EXPECT_THROW(lc.to_bound(), std::runtime_error);
  }
  // (d) checkpoint before bind is refused; checkpoint after finalize is refused.
  {
    SystemLifecycle lc;
    EXPECT_THROW(lc.to_checkpointed(), std::runtime_error);
    lc.to_bound();
    lc.to_finalized();
    EXPECT_THROW(lc.to_checkpointed(), std::runtime_error);
  }
}

TEST(SystemLifecycle, NewStatesNeverSurfaceWithoutTheExplicitTransition) {
  // The historical sequence (bind then step) NEVER yields checkpointed / finalized: those are
  // reachable only via to_checkpointed / to_finalized, which have no current caller -> bit-identity.
  SystemLifecycle lc;
  lc.to_bound();
  for (int step = 0; step <= 3; ++step) {
    const std::string s = lc.state(step);
    EXPECT_TRUE(s == "bound" || s == "running") << "unexpected state for step " << step << ": " << s;
  }
}

// --- 2. Registry structured reports --------------------------------------------------------------
TEST(SystemDiagnosticsRegistry, OptionsAndNewtonReportsRoundTrip) {
  pops::runtime::system::SystemDiagnosticsRegistry reg;
  pops::EffectiveBlockOptions opt;
  opt.name = "ions";
  opt.route = "native_model";
  reg.block_options["ions"] = opt;
  EXPECT_EQ(reg.block_options_ptr("ions")->route, "native_model");
  EXPECT_EQ(reg.block_options_ptr("absent"), nullptr);

  const auto rows = reg.options_report();
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_EQ(rows[0].name, "ions");

  EXPECT_EQ(reg.newton_report_ptr("ions"), nullptr) << "absent -> nullptr, not a silently empty report";
  auto rep = std::make_shared<pops::NewtonReport>();
  rep->enabled = true;
  reg.newton_reports["ions"] = rep;
  ASSERT_NE(reg.newton_report_ptr("ions"), nullptr);
  EXPECT_TRUE(reg.newton_report_ptr("ions")->enabled);
}

TEST(SystemCouplingRegistry, HoldsOperatorsAndBounds) {
  pops::runtime::system::SystemCouplingRegistry reg;
  reg.dt_bounds.push_back({"schur", [] { return 0.5; }});
  reg.coupled_freqs.push_back({"ionization", 3.0});
  EXPECT_EQ(reg.dt_bounds.size(), 1u);
  EXPECT_EQ(reg.dt_bounds[0].label, "schur");
  EXPECT_DOUBLE_EQ(reg.dt_bounds[0].fn(), 0.5);
  EXPECT_EQ(reg.coupled_freqs[0].mu, 3.0);
  EXPECT_TRUE(reg.operators.empty());
  EXPECT_TRUE(reg.coupled_operators.empty());
}

TEST(SystemDomain, LayoutReportReflectsCartesianConstruction) {
  pops::SystemConfig c;
  c.n = 16;
  c.L = 1.0;
  c.periodic = true;
  c.geometry = "cartesian";
  pops::runtime::system::SystemDomain domain(c);
  const auto rep = domain.layout_report();
  EXPECT_FALSE(rep.polar);
  EXPECT_EQ(rep.nx, 16);
  EXPECT_EQ(rep.ny, 16);
  EXPECT_EQ(rep.n_boxes, 1) << "Cartesian is a single box";
  EXPECT_TRUE(rep.periodic);
  EXPECT_FALSE(rep.eb_active);
  EXPECT_GE(rep.aux_ncomp, 3) << "the shared aux channel is at least 3 wide";
}

}  // namespace
