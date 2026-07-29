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

#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

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
static_assert(
    std::is_base_of_v<
        pops::runtime::program::ProgramExecutionServices<pops::runtime::program::ProgramContext>,
        pops::runtime::program::ProgramContext>,
    "ProgramContext must consume the one shared Program execution-service implementation");

TEST(ProgramContextSchurFree, HeaderIsSelfContainedAndBuilds) {
  // Reaching this TEST means program_context.hpp compiled standalone (no Schur/MG/Lorentz headers).
  // Constructing a ProgramContext from a null System* is well-defined here: we never dereference it,
  // we only exercise that the type is instantiable from the facade header by itself.
  pops::runtime::program::ProgramContext ctx(static_cast<pops::System*>(nullptr));
  (void)ctx;
  SUCCEED() << "program_context.hpp builds without any coupling/schur/** dependency";
}

namespace {

template <bool Amr>
class ExecutionServicesFixture
    : public pops::runtime::program::ProgramExecutionServices<ExecutionServicesFixture<Amr>> {
 public:
  using SharedServices =
      pops::runtime::program::ProgramExecutionServices<ExecutionServicesFixture<Amr>>;

  explicit ExecutionServicesFixture(int active_level) : active_level_(active_level) {}

  int last_params_block() const { return last_params_block_; }
  int field_update_count() const { return field_update_count_; }
  pops::Real diagnostic(const std::string& name) const { return diagnostics_.at(name); }
  double logical_dt() const { return logical_dt_; }
  void fail_next_logical_apply() { fail_logical_apply_ = true; }
  int history_register_count() const { return history_register_count_; }
  int history_read_count() const { return history_read_count_; }
  int history_store_count() const { return history_store_count_; }
  int history_rotate_count() const { return history_rotate_count_; }
  int history_runtime_owner() const { return history_runtime_owner_; }
  bool history_initialized() const { return history_initialized_; }
  pops::Real history_outgoing_dt() const { return history_outgoing_dt_; }
  const std::string& history_rotation_clock() const { return history_rotation_clock_; }

 private:
  friend class pops::runtime::program::ProgramExecutionServices<ExecutionServicesFixture<Amr>>;

  struct LogicalRollback {
    double dt = 0.0;
  };

  double program_execution_logical_parent_dt_() const noexcept { return logical_dt_; }
  LogicalRollback program_execution_capture_logical_evaluation_() const noexcept {
    return {logical_dt_};
  }
  void program_execution_apply_logical_evaluation_(
      const typename SharedServices::LogicalEvaluationInterval& interval) const {
    logical_dt_ = interval.child_dt;
    if (fail_logical_apply_) {
      fail_logical_apply_ = false;
      throw std::runtime_error("injected logical projection failure");
    }
  }
  void program_execution_restore_logical_evaluation_(
      const LogicalRollback& rollback) const noexcept {
    logical_dt_ = rollback.dt;
  }
  const std::vector<int>& program_execution_block_map_() const { return block_map_; }
  int program_execution_block_count_() const { return 2; }
  pops::Real program_execution_physical_time_() const { return pops::Real(3.5); }
  void program_execution_record_scalar_(const std::string& name, pops::Real value) const {
    diagnostics_[name] = value;
  }
  pops::RuntimeParams program_execution_params_(int block) const {
    last_params_block_ = block;
    return {};
  }
  void program_execution_set_field_timepoint_(const std::string&,
                                              const pops::FieldLogicalTimePoint&) const {
    ++field_update_count_;
  }
  void program_execution_set_field_parameters_(const std::string&,
                                               const std::vector<double>&) const {
    ++field_update_count_;
  }
  void program_execution_set_field_kernel_(const std::string&,
                                           const pops::CompiledFieldBoundaryKernel&) const {
    ++field_update_count_;
  }
  void program_execution_register_history_storage_(
      const typename SharedServices::HistoryRegistration& registration) const {
    ++history_register_count_;
    history_runtime_owner_ = registration.runtime_owner;
  }
  pops::MultiFab& program_execution_read_history_storage_(
      const typename SharedServices::HistoryRegistration&, int,
      typename SharedServices::HistoryReadMode) const {
    ++history_read_count_;
    return history_field_;
  }
  bool program_execution_history_initialized_storage_(
      const typename SharedServices::HistoryRegistration&) const {
    return history_initialized_;
  }
  void program_execution_set_history_initialized_storage_(
      const typename SharedServices::HistoryRegistration&, bool initialized) const {
    history_initialized_ = initialized;
  }
  typename SharedServices::HistoryStorePlan program_execution_history_store_plan_(
      const typename SharedServices::HistoryRegistration&) const {
    return {true, pops::Real(0.25)};
  }
  void program_execution_store_history_storage_(
      const typename SharedServices::HistoryRegistration&, const pops::MultiFab&,
      const std::optional<pops::Real>& outgoing_dt) const {
    ++history_store_count_;
    history_outgoing_dt_ = outgoing_dt.value_or(pops::Real(-1));
  }
  bool program_execution_history_supports_selective_rotation_() const noexcept { return !Amr; }
  typename SharedServices::HistoryRotationAction program_execution_history_rotation_action_()
      const noexcept {
    return SharedServices::HistoryRotationAction::Rotate;
  }
  void program_execution_defer_history_rotation_() const noexcept {}
  void program_execution_rotate_history_storage_(const std::string& clock_identity) const {
    ++history_rotate_count_;
    history_rotation_clock_ = clock_identity;
  }
  pops::runtime::program::Profiler& program_execution_profiler_() const { return profiler_; }
  int program_execution_macro_step_() const { return 4; }
  int program_execution_active_level_() const { return active_level_; }
  typename SharedServices::ProgramResourceTopology program_execution_resource_topology_()
      const noexcept {
    return {11, 17, Amr ? 3 : 1};
  }
  int program_execution_resource_level_() const noexcept { return resource_level_; }
  void program_execution_select_resource_level_(int selected) const noexcept {
    resource_level_ = selected;
  }

  int active_level_ = -1;
  mutable int resource_level_ = Amr ? 1 : 0;
  std::vector<int> block_map_{1, 0};
  mutable std::map<std::string, pops::Real> diagnostics_;
  mutable int last_params_block_ = -1;
  mutable int field_update_count_ = 0;
  mutable int history_register_count_ = 0;
  mutable int history_read_count_ = 0;
  mutable int history_store_count_ = 0;
  mutable int history_rotate_count_ = 0;
  mutable int history_runtime_owner_ = -1;
  mutable bool history_initialized_ = false;
  mutable pops::Real history_outgoing_dt_ = pops::Real(-1);
  mutable std::string history_rotation_clock_;
  mutable pops::MultiFab history_field_;
  mutable pops::runtime::program::Profiler profiler_;
  mutable double logical_dt_ = 0.4;
  mutable bool fail_logical_apply_ = false;
};

template <class Context>
void expect_shared_program_services(Context& context, bool amr) {
  using pops::runtime::program::ScheduleDomainKind;

  context.configure_primary_clock("clock.macro");
  context.declare_clock_relation("clock.macro", "clock.fast", 2);
  EXPECT_TRUE(
      context.schedule_domain_occurs(ScheduleDomainKind::kAcceptedStep, "clock.macro", "", -1));
  EXPECT_TRUE(
      context.schedule_is_due(7, 2, ScheduleDomainKind::kAcceptedStep, "clock.macro", "", -1));
  EXPECT_FALSE(context.schedule_at_start(ScheduleDomainKind::kAcceptedStep, "clock.macro", "", -1));
  EXPECT_EQ(context.schedule_domain_occurs(ScheduleDomainKind::kAmrLevel, "clock.macro", "", 1),
            amr);

  auto subcycle = context.subcycle_scope("clock.macro", "clock.fast", 2);
  for (int iteration = 0; iteration < 2; ++iteration) {
    subcycle.iteration(iteration);
    EXPECT_TRUE(
        context.schedule_domain_occurs(ScheduleDomainKind::kClockTick, "clock.fast", "", -1));
  }
  subcycle.finish();
  context.synchronize_sample_and_hold("clock.macro", "clock.fast", 4, pops::Real(0));

  EXPECT_DOUBLE_EQ(context.logical_dt(), 0.4);
  {
    auto child = context.logical_evaluation_scope(1, 2);
    EXPECT_DOUBLE_EQ(static_cast<double>(child.dt()), 0.2);
    EXPECT_DOUBLE_EQ(context.logical_dt(), 0.2);
  }
  EXPECT_DOUBLE_EQ(context.logical_dt(), 0.4);
  context.fail_next_logical_apply();
  EXPECT_THROW((void)context.logical_evaluation_scope(0, 2), std::runtime_error);
  EXPECT_DOUBLE_EQ(context.logical_dt(), 0.4)
      << "the shared RAII scope restores a provider that throws after partial mutation";

  const auto topology = context.program_resource_topology();
  EXPECT_EQ(topology.epoch, 11);
  EXPECT_EQ(topology.generation, 17);
  EXPECT_EQ(topology.levels, amr ? 3 : 1);
  const int incoming_resource_level = context.level();
  const int selected_resource_level = amr ? 2 : 0;
  context.with_program_resource_level(
      selected_resource_level, [&]() { EXPECT_EQ(context.level(), selected_resource_level); });
  EXPECT_EQ(context.level(), incoming_resource_level);
  EXPECT_THROW(
      context.with_program_resource_level(
          selected_resource_level, []() { throw std::runtime_error("injected body failure"); }),
      std::runtime_error);
  EXPECT_EQ(context.level(), incoming_resource_level)
      << "the shared topology scope restores the provider cursor after a body failure";
  std::vector<int> visited_levels;
  context.for_each_program_resource_level([&](int level) {
    EXPECT_EQ(context.level(), level);
    visited_levels.push_back(level);
  });
  EXPECT_EQ(visited_levels, amr ? std::vector<int>({0, 1, 2}) : std::vector<int>({0}));
  EXPECT_EQ(context.level(), incoming_resource_level);
  EXPECT_THROW(context.set_level(topology.levels), std::out_of_range);

  EXPECT_EQ(context.sys_block(0), 1);
  EXPECT_EQ(context.sys_block(1), 0);
  EXPECT_EQ(context.n_blocks(), 2);
  EXPECT_EQ(context.macro_step(), 4);
  EXPECT_DOUBLE_EQ(static_cast<double>(context.physical_time()), 3.5);

  context.set_stage_time(1, 2);
  context.record_scalar("mass", pops::Real(9));
  EXPECT_EQ(context.diagnostic("mass"), pops::Real(9));
  (void)context.program_params(1);
  EXPECT_EQ(context.last_params_block(), 1);
  context.set_field_logical_timepoint("potential", {});
  context.set_field_boundary_parameters("potential", {1.0, 2.0});
  context.set_field_boundary_kernel("potential", {});
  EXPECT_EQ(context.field_update_count(), 3);

  context.register_history("rate", 2, 1, 0, "block.U", "cell", "clock.macro", "dense.linear");
  EXPECT_EQ(context.history_runtime_owner(), 1);
  pops::MultiFab history_value;
  context.store_history("rate", history_value, 0);
  EXPECT_EQ(context.history_store_count(), 1);
  EXPECT_EQ(context.history_outgoing_dt(), pops::Real(0.25));
  (void)context.history("rate", 1, 0);
  EXPECT_EQ(context.history_read_count(), 1);
  (void)context.history_zero_start("cold", 1, 1, 0);
  EXPECT_TRUE(context.history_initialized());
  EXPECT_EQ(context.history_read_count(), 2);
  context.rotate_histories("clock.macro");
  EXPECT_EQ(context.history_rotate_count(), 1);
  EXPECT_EQ(context.history_rotation_clock(), amr ? "" : "clock.macro");
  const int registrations_before_drift = context.history_register_count();
  EXPECT_THROW(
      context.register_history("rate", 2, 1, 1, "block.U", "cell", "clock.macro", "dense.linear"),
      std::runtime_error);
  EXPECT_EQ(context.history_register_count(), registrations_before_drift)
      << "shared identity validation rejects owner drift before provider storage";

  context.profiler().enable();
  EXPECT_TRUE(context.schedule_decision(3, true, true));
  EXPECT_EQ(context.profiler().counter("nodes_due"), 1);
  EXPECT_EQ(context.profiler().counter("cache_misses"), 1);
  context.count_kernel(2);
  EXPECT_EQ(context.profiler().counter("kernels"), 2);

  EXPECT_THROW(context.set_stage_time(2, 1), std::runtime_error);
  EXPECT_THROW(
      context.schedule_is_due(-1, 2, ScheduleDomainKind::kAcceptedStep, "clock.macro", "", -1),
      std::runtime_error);
  EXPECT_THROW(context.sys_block(2), std::runtime_error);
  EXPECT_THROW(context.scheduler_error("stale value"), std::runtime_error);
}

}  // namespace

TEST(ProgramExecutionServices, UniformAndAmrProvidersRunTheSameContractFixture) {
  ExecutionServicesFixture<false> uniform(-1);
  ExecutionServicesFixture<true> amr(1);
  expect_shared_program_services(uniform, false);
  expect_shared_program_services(amr, true);
}
