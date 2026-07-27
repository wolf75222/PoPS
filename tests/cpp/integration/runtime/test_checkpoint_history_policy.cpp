// Selective history persistence authority (ADC-626). Rebuilding omitted Interval / Revolve slots
// executes scientific code and is therefore reserved for a loader-authenticated Program artifact
// whose metadata names the exact `(ring, depth)` pair. A direct install_program_step lambda remains
// useful for ordinary low-level stepping tests, but must fail closed for selective replay. Dense
// persistence stores every slot and remains a strict no-op that needs no replay authority.

#include <gtest/gtest.h>

#include <pops/mesh/storage/mf_arith.hpp>  // saxpy / lincomb
#include <pops/mesh/storage/multifab.hpp>
#include <pops/physics/bricks/source.hpp>                // NoSource
#include <pops/physics/composition/composite.hpp>        // CompositeModel
#include <pops/physics/fluids/euler.hpp>                 // Euler
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/system.hpp>

#include <limits>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

// ONE Kokkos ScopeGuard for the whole TU: a function-local static initialized on the first call and
// shared by every TEST (a second ScopeGuard while Kokkos is live is illegal). Each TEST calls kokkos().
void kokkos() {
#if defined(POPS_HAS_KOKKOS)
  static Kokkos::ScopeGuard guard;
  (void)guard;
#endif
}

struct NoEll {
  template <class State>
  POPS_HD Real rhs(const State&) const {
    return Real(0);
  }
};
using GasModel = CompositeModel<Euler, NoSource, NoEll>;
constexpr double kGamma = 1.4;

void add_gas_block(System& s, const std::string& name) {
  add_compiled_model(s, name, GasModel{Euler{kGamma}, NoSource{}, NoEll{}}, "minmod", "rusanov",
                     "conservative", "explicit", kGamma);
}

void add_gas(System& s) {
  add_gas_block(s, "gas");
  s.set_poisson("charge_density", "geometric_mg");
}

void register_state_history(System& s, const std::string& ring, int depth, int owner = 0) {
  s.register_history(ring, depth - 1, -1, owner, "test.state." + std::to_string(owner),
                     "test.space", "test.clock", "test.exact");
}

// Install a deterministic macro-step closure with the exact compiled keep_history phase: snapshot
// U^n first, advance the qualified owner by +inc, then rotate after the commit. The dt ledger therefore travels
// with its starting sample (the outgoing interval toward the newer sample). `inc` scales with dt so
// a variable-dt run with multiple independent gaps proves that exact provenance. The closure
// captures &s; s must outlive it.
void install_ramp_program(System& s, const std::string& ring, double rate, int owner = 0,
                          int* executed_steps = nullptr) {
  System* self = &s;
  self->install_program_step([self, ring, rate, owner, executed_steps](double dt) {
    if (executed_steps != nullptr)
      ++*executed_steps;
    MultiFab& U = self->block_state(owner);
    self->store_history(ring, U);
    // Advance: U += rate*dt (a deterministic, dt-dependent conservative increment on every component).
    MultiFab bump = U;  // same layout
    bump.set_val(Real(rate) * Real(dt));
    pops::saxpy(U, Real(1), bump);
    self->rotate_histories();
  });
}

SystemConfig make_cfg() {
  SystemConfig cfg;
  cfg.n = 8;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};
  return cfg;
}

}  // namespace

TEST(CheckpointHistoryPolicy, DirectStepCannotAuthorizeIntervalOrRevolveSelectiveReplay) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "state_prev";
  constexpr int depth = 5;
  const std::vector<std::vector<int>> policies = {{0, 2, 4}, {0, 1, 4}};

  for (const auto& stored : policies) {
    System system(cfg);
    add_gas(system);
    register_state_history(system, ring, depth);
    int executed_steps = 0;
    install_ramp_program(system, ring, 3.0, /*owner=*/0, &executed_steps);
    const std::vector<double> token = system.history_global(ring, 0);
    for (const int slot : stored)
      system.restore_history(ring, slot, token);
    for (int slot = 0; slot < depth; ++slot)
      system.restore_history_slot_dt(ring, slot, 0.05);
    system.set_history_initialized(ring, true);
    const std::vector<double> live_before = system.state_global("gas");

    try {
      (void)system.rebuild_history_slots(ring, stored);
      FAIL() << "a direct Program step forged selective replay authority";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string(error.what()).find("validated native authority"), std::string::npos);
      EXPECT_NE(std::string(error.what()).find("Dense()"), std::string::npos);
    }
    EXPECT_EQ(executed_steps, 0);
    EXPECT_EQ(system.state_global("gas"), live_before);
  }
}

TEST(CheckpointHistoryPolicy, DenseIsANoOpWithoutArtifactReplayAuthority) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "state_prev";
  constexpr int depth = 5;
  System system(cfg);
  add_gas(system);
  register_state_history(system, ring, depth);
  int executed_steps = 0;
  install_ramp_program(system, ring, 2.0, /*owner=*/0, &executed_steps);
  const std::vector<double> token = system.history_global(ring, 0);
  const std::vector<int> stored = {0, 1, 2, 3, 4};
  for (const int slot : stored)
    system.restore_history(ring, slot, token);
  system.set_history_initialized(ring, true);
  const std::vector<double> live_before = system.state_global("gas");

  EXPECT_EQ(system.rebuild_history_slots(ring, stored), 0);
  EXPECT_EQ(executed_steps, 0);
  EXPECT_EQ(system.state_global("gas"), live_before);
}

TEST(CheckpointHistoryPolicy, VariableDtDirectStepStillLacksArtifactReplayAuthority) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "state_prev";
  constexpr int depth = 5;
  System system(cfg);
  add_gas(system);
  register_state_history(system, ring, depth);
  int executed_steps = 0;
  install_ramp_program(system, ring, 2.0, /*owner=*/0, &executed_steps);
  const std::vector<int> stored = {0, 2, 4};
  const std::vector<double> token = system.history_global(ring, 0);
  for (const int slot : stored)
    system.restore_history(ring, slot, token);
  const std::vector<double> slot_dt = {0.03, 0.07, 0.05, 0.11, 0.02};
  for (int slot = 0; slot < depth; ++slot)
    system.restore_history_slot_dt(ring, slot, slot_dt[static_cast<std::size_t>(slot)]);
  system.set_history_initialized(ring, true);
  const std::vector<double> live_before = system.state_global("gas");

  try {
    (void)system.rebuild_history_slots(ring, stored);
    FAIL() << "variable outgoing dt values bypassed artifact replay authority";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("validated native authority"), std::string::npos);
  }
  EXPECT_EQ(executed_steps, 0);
  EXPECT_EQ(system.state_global("gas"), live_before);
}

TEST(CheckpointHistoryPolicy, QualifiedNonzeroOwnerStillRequiresArtifactReplayAuthority) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "second_state_prev";
  constexpr int depth = 5;
  System system(cfg);
  add_gas_block(system, "first");
  add_gas_block(system, "second");
  system.set_poisson("charge_density", "geometric_mg");
  register_state_history(system, ring, depth, /*owner=*/1);
  int executed_steps = 0;
  install_ramp_program(system, ring, 2.0, /*owner=*/1, &executed_steps);
  const std::vector<int> stored = {0, 2, 4};
  const std::vector<double> token = system.history_global(ring, 0);
  for (const int slot : stored)
    system.restore_history(ring, slot, token);
  for (int slot = 0; slot < depth; ++slot)
    system.restore_history_slot_dt(ring, slot, 0.05);
  system.set_history_initialized(ring, true);
  const std::vector<double> first_before = system.state_global("first");
  const std::vector<double> second_before = system.state_global("second");

  try {
    (void)system.rebuild_history_slots(ring, stored);
    FAIL() << "a qualified owner bypassed artifact replay authority";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("validated native authority"), std::string::npos);
  }
  EXPECT_EQ(executed_steps, 0);
  EXPECT_EQ(system.state_global("first"), first_before);
  EXPECT_EQ(system.state_global("second"), second_before);
}

TEST(CheckpointHistoryPolicy, SelectiveReplayRefusesNonDefaultProgramCadence) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "state_prev";
  constexpr int depth = 5;
  System system(cfg);
  add_gas(system);
  register_state_history(system, ring, depth);
  install_ramp_program(system, ring, 1.0);
  system.set_program_cadence(/*substeps=*/2, /*stride=*/3);

  const std::vector<double> token = system.history_global(ring, 0);
  for (const int slot : {0, 2, 4})
    system.restore_history(ring, slot, token);
  system.set_history_initialized(ring, true);

  try {
    (void)system.rebuild_history_slots(ring, {0, 2, 4});
    FAIL() << "a selective replay under non-default Program cadence must fail closed";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("substeps=1, stride=1"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("Dense()"), std::string::npos);
  }
}

TEST(CheckpointHistoryPolicy, InvalidOutgoingDtFailsBeforeSelectiveReplayMutation) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "state_prev";
  constexpr int depth = 3;
  System system(cfg);
  add_gas(system);
  register_state_history(system, ring, depth);
  int executed_steps = 0;
  install_ramp_program(system, ring, 1.0, /*owner=*/0, &executed_steps);

  EXPECT_THROW(system.restore_history_slot_dt(ring, 0, std::numeric_limits<double>::quiet_NaN()),
               std::runtime_error);
  EXPECT_THROW(system.restore_history_slot_dt(ring, 0, std::numeric_limits<double>::infinity()),
               std::runtime_error);
  EXPECT_THROW(system.restore_history_slot_dt(ring, 0, -0.01), std::runtime_error);

  const std::vector<double> anchor = system.history_global(ring, 0);
  system.restore_history(ring, 0, anchor);
  system.restore_history(ring, 2, anchor);
  for (int slot = 0; slot < depth; ++slot)
    system.restore_history_slot_dt(ring, slot, 0.0);
  system.set_history_initialized(ring, true);
  const std::vector<double> live_before = system.state_global("gas");
  const std::vector<double> slot_before = system.history_global(ring, 0);

  EXPECT_THROW((void)system.rebuild_history_slots(ring, {0, 2}), std::runtime_error);
  EXPECT_EQ(executed_steps, 0);
  EXPECT_EQ(system.state_global("gas"), live_before);
  EXPECT_EQ(system.history_global(ring, 0), slot_before);
}

// (D) The oldest slot MUST be stored: a policy whose stored set omits slot depth-1 is refused verbatim.
TEST(CheckpointHistoryPolicy, RebuildRefusesMissingOldestSlot) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "state_prev";
  const int depth = 5;
  System s(cfg);
  add_gas(s);
  install_ramp_program(s, ring, 1.0);
  register_state_history(s, ring, depth);
  s.restore_history(ring, 0, s.history_global(ring, 0));  // register + a token slot
  s.set_history_initialized(ring, true);
  // stored = {0, 2} omits the oldest slot 4 -> unreconstructable.
  bool threw = false;
  std::string what;
  try {
    s.rebuild_history_slots(ring, std::vector<int>{0, 2});
  } catch (const std::runtime_error& e) {
    threw = true;
    what = e.what();
  }
  EXPECT_TRUE(threw) << "missing_oldest_slot_refused";
  EXPECT_TRUE(what.find("oldest slot") != std::string::npos)
      << "verbatim_oldest_slot_message: " << what;
}

// (E) The newest slot MUST be stored: there is no newer anchor from which to fill it backwards.
TEST(CheckpointHistoryPolicy, RebuildRefusesMissingNewestSlot) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "state_prev";
  const int depth = 5;
  System s(cfg);
  add_gas(s);
  register_state_history(s, ring, depth);
  install_ramp_program(s, ring, 1.0);
  const std::vector<double> token = s.history_global(ring, 0);
  s.restore_history(ring, 2, token);
  s.restore_history(ring, 4, token);
  s.set_history_initialized(ring, true);
  bool threw = false;
  std::string what;
  try {
    s.rebuild_history_slots(ring, std::vector<int>{2, 4});
  } catch (const std::runtime_error& e) {
    threw = true;
    what = e.what();
  }
  EXPECT_TRUE(threw) << "missing_newest_slot_refused";
  EXPECT_TRUE(what.find("newest slot") != std::string::npos)
      << "verbatim_newest_slot_message: " << what;
}

// (F) Replay requires an installed Program: rebuild without a program fails loud (never a silent skip).
TEST(CheckpointHistoryPolicy, RebuildRefusesWithoutInstalledProgram) {
  kokkos();
  const SystemConfig cfg = make_cfg();
  const std::string ring = "state_prev";
  System s(cfg);
  add_gas(s);
  register_state_history(s, ring, /*depth=*/4);
  s.restore_history(ring, 0, s.history_global(ring, 0));
  s.restore_history(ring, 3, s.history_global(ring, 3));
  s.set_history_initialized(ring, true);
  bool threw = false;
  std::string what;
  try {
    s.rebuild_history_slots(ring, std::vector<int>{0, 3});
  } catch (const std::runtime_error& e) {
    threw = true;
    what = e.what();
  }
  EXPECT_TRUE(threw) << "no_program_refused";
  EXPECT_TRUE(what.find("no compiled Program") != std::string::npos)
      << "verbatim_no_program: " << what;
}
