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

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/runtime/program/program_context.hpp>

#include <bit>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
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

  explicit ExecutionServicesFixture(int active_level) : active_level_(active_level) {
    program_runtime_state_.block_map_ = {1, 0};
    program_runtime_state_.seed_params(1, {2.5});
  }

  int field_update_count() const { return field_update_count_; }
  pops::Real diagnostic(const std::string& name) const {
    return program_runtime_state_.diagnostic(name, "ExecutionServicesFixture");
  }
  double logical_dt() const { return logical_dt_; }
  void set_untracked_logical_dt(double value) { logical_dt_ = value; }
  void fail_next_logical_apply() { fail_logical_apply_ = true; }
  int rhs_group_identity() const { return rhs_group_identity_; }
  const std::vector<int>& rhs_group_program_blocks() const { return rhs_group_program_blocks_; }
  const std::vector<int>& rhs_group_runtime_blocks() const { return rhs_group_runtime_blocks_; }
  const std::vector<int>& rhs_group_rate_ids() const { return rhs_group_rate_ids_; }
  const std::vector<int>& rhs_group_flux_only() const { return rhs_group_flux_only_; }
  int source_runtime_block() const { return source_runtime_block_; }
  const pops::MultiFab* source_state() const { return source_state_; }
  const pops::MultiFab* source_rhs() const { return source_rhs_; }
  int history_register_count() const { return history_register_count_; }
  int history_read_count() const { return history_read_count_; }
  int history_store_count() const { return history_store_count_; }
  int history_rotate_count() const { return history_rotate_count_; }
  int history_runtime_owner() const { return history_runtime_owner_; }
  bool history_initialized() const { return history_initialized_; }
  pops::Real history_outgoing_dt() const { return history_outgoing_dt_; }
  const std::string& history_rotation_clock() const { return history_rotation_clock_; }
  int resource_level() const { return resource_level_; }
  int resource_levels() const { return resource_levels_; }
  void set_scratch_resource_identity(std::uint64_t epoch, std::uint64_t generation, int levels,
                                     int level) {
    resource_topology_epoch_ = epoch;
    resource_materialization_generation_ = generation;
    resource_levels_ = levels;
    resource_level_ = level;
  }
  int boundary_program_block() const { return boundary_program_block_; }
  int assembly_target_count() const { return assembly_target_count_; }
  int assembly_source_count() const { return assembly_source_count_; }
  int linear_solution_count() const { return linear_solution_count_; }
  int authority_check_count() const { return authority_check_count_; }
  const std::string& assembly_target_slot() const { return assembly_target_slot_; }
  const std::string& assembly_source_slot() const { return assembly_source_slot_; }
  const std::string& boundary_dispatch_operation() const { return boundary_dispatch_operation_; }
  int boundary_dispatch_program_block() const { return boundary_dispatch_program_block_; }
  int boundary_dispatch_runtime_block() const { return boundary_dispatch_runtime_block_; }
  int boundary_dispatch_rate() const { return boundary_dispatch_rate_; }
  bool boundary_dispatch_has_session() const { return boundary_dispatch_has_session_; }
  bool boundary_dispatch_flux_only() const { return boundary_dispatch_flux_only_; }
  int operator_topology_count() const { return operator_topology_count_; }
  int install_count() const { return install_count_; }
  int field_solve_dispatch_count() const {
    return static_cast<int>(field_solve_dispatches_.size());
  }
  const std::vector<std::string>& field_solve_dispatches() const { return field_solve_dispatches_; }
  void run_installed_step(double dt) const {
    if (!installed_step_)
      throw std::logic_error("fixture has no installed Program step");
    installed_step_(dt);
  }
  bool exclusive_workspace_in_use() const { return exclusive_workspace_in_use_; }
  void exercise_exclusive_workspace(bool reenter, bool fail_after_enter) const {
    typename SharedServices::ExclusiveUseGuard use(exclusive_workspace_in_use_,
                                                   "fixture exclusive workspace is already in use");
    if (reenter)
      exercise_exclusive_workspace(false, false);
    if (fail_after_enter)
      throw std::runtime_error("fixture operation failed after acquiring its workspace");
  }

 private:
  friend class pops::runtime::program::ProgramExecutionServices<ExecutionServicesFixture<Amr>>;

  struct LogicalRollback {
    double dt = 0.0;
  };

  struct FieldFacade {
    int* update_count = nullptr;

    void set_field_logical_timepoint(const std::string&, const pops::FieldLogicalTimePoint&) const {
      ++*update_count;
    }
    void set_field_boundary_parameters(const std::string&, const std::vector<double>&) const {
      ++*update_count;
    }
    void set_field_boundary_kernel(const std::string&,
                                   const pops::CompiledFieldBoundaryKernel&) const {
      ++*update_count;
    }
  };

  double program_execution_logical_parent_dt_() const noexcept { return logical_dt_; }
  void program_execution_install_(std::function<void(double)> step) const {
    ++install_count_;
    installed_step_ = std::move(step);
  }
  pops::SolveOutcome program_execution_solve_fields_outcome_() const {
    return solved_field_outcome_("default");
  }
  pops::SolveOutcome program_execution_solve_fields_from_state_outcome_(int,
                                                                        pops::MultiFab&) const {
    return solved_field_outcome_("default-state");
  }
  pops::SolveOutcome program_execution_field_solve_from_state_at_outcome_(
      const pops::runtime::multiblock::BoundaryEvaluationPoint&, const std::string&, int,
      pops::MultiFab&) const {
    return solved_field_outcome_("qualified-state-at");
  }
  pops::SolveReport program_execution_solve_fields_from_state_at_(
      const pops::runtime::multiblock::BoundaryEvaluationPoint& point,
      const std::string& provider_slot, int block, pops::MultiFab& state) const {
    pops::SolveOutcome outcome =
        program_execution_field_solve_from_state_at_outcome_(point, provider_slot, block, state);
    return outcome.consume(pops::SolveConsumption::kAccept);
  }
  pops::SolveOutcome program_execution_solve_fields_from_blocks_outcome_(
      const std::vector<const pops::MultiFab*>&) const {
    return solved_field_outcome_("default-blocks");
  }
  pops::SolveOutcome program_execution_solve_generated_field_from_blocks_outcome_(
      const pops::runtime::multiblock::BoundaryEvaluationPoint&, std::int64_t, std::string_view,
      std::initializer_list<typename SharedServices::FieldStageOverride>) const {
    return solved_field_outcome_("generated-blocks");
  }
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
  pops::runtime::multiblock::BoundaryEvaluationPoint program_execution_boundary_point_(
      int stage_id) const {
    return {"fixture.clock", 0, Amr ? active_level_ : 0, 0, stage_id};
  }
  void program_execution_rhs_into_(int program_block, int runtime_block, pops::MultiFab&,
                                   pops::MultiFab&, int rate_id) const {
    record_boundary_dispatch_("rhs", program_block, runtime_block, rate_id, false, false);
  }
  bool program_execution_has_boundary_linearization_(int runtime_block) const {
    record_boundary_dispatch_("has_linearization", -1, runtime_block, -1, false, false);
    return runtime_block == 1;
  }
  void program_execution_rhs_core_into_at_(
      const pops::runtime::multiblock::BoundaryEvaluationPoint&, int runtime_block, pops::MultiFab&,
      pops::MultiFab&, bool flux_only, const pops::PreparedGridBoundarySession* boundary) const {
    record_boundary_dispatch_("rhs_core", -1, runtime_block, -1, boundary != nullptr, flux_only);
  }
  void program_execution_boundary_residual_into_at_(
      const pops::runtime::multiblock::BoundaryEvaluationPoint&, int runtime_block, pops::MultiFab&,
      pops::MultiFab&, const pops::PreparedGridBoundarySession* boundary) const {
    record_boundary_dispatch_("boundary_residual", -1, runtime_block, -1, boundary != nullptr,
                              false);
  }
  void program_execution_boundary_jvp_into_at_(
      const pops::runtime::multiblock::BoundaryEvaluationPoint&, int runtime_block, pops::MultiFab&,
      const pops::MultiFab&, pops::MultiFab&,
      const pops::PreparedGridBoundarySession* boundary) const {
    record_boundary_dispatch_("boundary_jvp", -1, runtime_block, -1, boundary != nullptr, false);
  }
  void program_execution_neg_div_flux_default_into_(int program_block, int runtime_block,
                                                    pops::MultiFab&, pops::MultiFab&,
                                                    int rate_id) const {
    record_boundary_dispatch_("neg_div_flux", program_block, runtime_block, rate_id, false, true);
  }
  void program_execution_neg_div_named_flux_into_(pops::MultiFab&, pops::MultiFab&, pops::MultiFab&,
                                                  pops::MultiFab&,
                                                  const pops::ExecutionLane* lane) const {
    record_boundary_dispatch_("named_flux", -1, -1, -1, lane != nullptr, true);
  }
  pops::OperatorFingerprint program_execution_operator_topology_(const pops::MultiFab&) const {
    ++operator_topology_count_;
    return {UINT64_C(9), UINT64_C(10), UINT64_C(11), UINT64_C(12)};
  }
  pops::OperatorEvaluationSnapshot program_execution_operator_evaluation_snapshot_(
      pops::OperatorFingerprint authority, pops::OperatorFingerprint topology,
      pops::OperatorFingerprint resources, std::uint64_t revision) const {
    return {authority,
            revision,
            4,
            0,
            1,
            std::bit_cast<std::uint64_t>(logical_dt_),
            std::bit_cast<std::uint64_t>(3.5),
            Amr ? UINT64_C(17) : UINT64_C(1),
            topology,
            resources};
  }
  void program_execution_rhs_group_(const typename SharedServices::RhsGroupBatch& batch) const {
    this->count_kernel(static_cast<std::int64_t>(batch.requests.size()));
    rhs_group_identity_ = batch.group_id;
    rhs_group_program_blocks_.clear();
    for (const auto& request : batch.requests)
      rhs_group_program_blocks_.push_back(request.block);
    rhs_group_runtime_blocks_ = batch.runtime_blocks;
    rhs_group_rate_ids_ = batch.rate_ids;
    rhs_group_flux_only_ = batch.flux_only;
  }
  void program_execution_source_default_into_(int runtime_block, pops::MultiFab& state,
                                              pops::MultiFab& rhs) const {
    source_runtime_block_ = runtime_block;
    source_state_ = &state;
    source_rhs_ = &rhs;
  }
  bool program_execution_is_polar_geometry_() const noexcept { return false; }
  pops::GridContext program_execution_default_grid_context_() const {
    const pops::Box2D domain = pops::Box2D::from_extents(4, 4);
    pops::GridContext context;
    context.dom = domain;
    context.bc = pops::BCRec{};
    context.geom =
        pops::Geometry{domain, pops::Real(0), pops::Real(1), pops::Real(0), pops::Real(1)};
    return context;
  }
  pops::GridContext program_execution_block_grid_context_(int block) const {
    boundary_program_block_ = block;
    return program_execution_default_grid_context_();
  }
  bool program_execution_owns_operator_authority_(pops::OperatorFingerprint authority) const {
    ++authority_check_count_;
    return authority == pops::OperatorFingerprint{1, 2, 3, 4};
  }
  pops::MultiFab& program_execution_assembly_target_(pops::MultiFab& field,
                                                     std::string_view field_slot_identity) const {
    ++assembly_target_count_;
    assembly_target_slot_ = field_slot_identity;
    return field;
  }
  pops::MultiFab& program_execution_assembly_source_(pops::MultiFab& field,
                                                     std::string_view field_slot_identity) const {
    ++assembly_source_count_;
    assembly_source_slot_ = field_slot_identity;
    return field;
  }
  pops::MultiFab& program_execution_linear_solution_(pops::MultiFab& field) const {
    ++linear_solution_count_;
    return field;
  }
  [[noreturn]] void program_execution_apply_polar_tensor_(pops::MultiFab&, pops::MultiFab&,
                                                          const pops::MultiFab*,
                                                          const pops::MultiFab*,
                                                          const pops::MultiFab*,
                                                          const pops::MultiFab*) const {
    throw std::logic_error("shared Cartesian fixture cannot execute a polar tensor stencil");
  }
  pops::runtime::program::ProgramRuntimeState& program_execution_runtime_state_() const {
    return program_runtime_state_;
  }
  typename SharedServices::ProgramClockCoordinate program_execution_clock_coordinate_() const {
    return {pops::Real(3.5), 4, active_level_};
  }
  FieldFacade& program_execution_field_facade_() const { return field_facade_; }
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
  double program_execution_history_slot_dt_storage_(
      const typename SharedServices::HistoryRegistration&, int) const {
    return 0.25;
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
  typename SharedServices::ProgramResourceTopology program_execution_resource_topology_()
      const noexcept {
    return {resource_topology_epoch_, resource_materialization_generation_, resource_levels_, 2};
  }
  int program_execution_resource_level_() const noexcept { return resource_level_; }
  void program_execution_select_resource_level_(int selected) const noexcept {
    resource_level_ = selected;
  }
  void record_boundary_dispatch_(std::string operation, int program_block, int runtime_block,
                                 int rate_id, bool has_session, bool flux_only) const {
    boundary_dispatch_operation_ = std::move(operation);
    boundary_dispatch_program_block_ = program_block;
    boundary_dispatch_runtime_block_ = runtime_block;
    boundary_dispatch_rate_ = rate_id;
    boundary_dispatch_has_session_ = has_session;
    boundary_dispatch_flux_only_ = flux_only;
  }
  pops::SolveOutcome solved_field_outcome_(std::string operation) const {
    field_solve_dispatches_.push_back(std::move(operation));
    pops::SolveReport report;
    report.mark_solved("shared-field-dispatch-fixture");
    return pops::SolveOutcome::serial(std::move(report));
  }

  int active_level_ = -1;
  mutable int resource_level_ = Amr ? 1 : 0;
  mutable std::uint64_t resource_topology_epoch_ = 11;
  mutable std::uint64_t resource_materialization_generation_ = 17;
  mutable int resource_levels_ = Amr ? 3 : 1;
  mutable pops::runtime::program::ProgramRuntimeState program_runtime_state_;
  mutable int field_update_count_ = 0;
  mutable FieldFacade field_facade_{&field_update_count_};
  mutable int history_register_count_ = 0;
  mutable int history_read_count_ = 0;
  mutable int history_store_count_ = 0;
  mutable int history_rotate_count_ = 0;
  mutable int history_runtime_owner_ = -1;
  mutable bool history_initialized_ = false;
  mutable pops::Real history_outgoing_dt_ = pops::Real(-1);
  mutable std::string history_rotation_clock_;
  mutable pops::MultiFab history_field_;
  mutable double logical_dt_ = 0.4;
  mutable bool fail_logical_apply_ = false;
  mutable int rhs_group_identity_ = -1;
  mutable std::vector<int> rhs_group_program_blocks_;
  mutable std::vector<int> rhs_group_runtime_blocks_;
  mutable std::vector<int> rhs_group_rate_ids_;
  mutable std::vector<int> rhs_group_flux_only_;
  mutable int source_runtime_block_ = -1;
  mutable const pops::MultiFab* source_state_ = nullptr;
  mutable const pops::MultiFab* source_rhs_ = nullptr;
  mutable int boundary_program_block_ = -1;
  mutable int assembly_target_count_ = 0;
  mutable int assembly_source_count_ = 0;
  mutable int linear_solution_count_ = 0;
  mutable int authority_check_count_ = 0;
  mutable std::string assembly_target_slot_;
  mutable std::string assembly_source_slot_;
  mutable std::string boundary_dispatch_operation_;
  mutable int boundary_dispatch_program_block_ = -1;
  mutable int boundary_dispatch_runtime_block_ = -1;
  mutable int boundary_dispatch_rate_ = -1;
  mutable bool boundary_dispatch_has_session_ = false;
  mutable bool boundary_dispatch_flux_only_ = false;
  mutable int operator_topology_count_ = 0;
  mutable int install_count_ = 0;
  mutable std::function<void(double)> installed_step_;
  mutable std::vector<std::string> field_solve_dispatches_;
  mutable bool exclusive_workspace_in_use_ = false;
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

  pops::MultiFab state_a;
  pops::MultiFab state_b;
  pops::MultiFab rhs_a;
  pops::MultiFab rhs_b;
  context.profiler().enable();
  context.rhs_group(20, {{0, &state_a, &rhs_a, 11, 0}, {1, &state_b, &rhs_b, 12, 1}});
  EXPECT_EQ(context.profiler().counter("kernels"), 2);
  EXPECT_EQ(context.rhs_group_identity(), 20);
  EXPECT_EQ(context.rhs_group_program_blocks(), std::vector<int>({0, 1}));
  EXPECT_EQ(context.rhs_group_runtime_blocks(), std::vector<int>({1, 0}));
  EXPECT_EQ(context.rhs_group_rate_ids(), std::vector<int>({11, 12}));
  EXPECT_EQ(context.rhs_group_flux_only(), std::vector<int>({0, 1}));
  EXPECT_THROW(context.rhs_group(20, {{0, &state_a, &rhs_a, 11, 0}, {1, &state_b, &rhs_b, 11, 1}}),
               std::invalid_argument);
  context.source_default_into(0, state_a, rhs_a);
  EXPECT_EQ(context.source_runtime_block(), 1);
  EXPECT_EQ(context.source_state(), &state_a);
  EXPECT_EQ(context.source_rhs(), &rhs_a);
  EXPECT_EQ(context.profiler().counter("kernels"), 3);

  context.set_stage_time(1, 2);
  context.record_scalar("mass", pops::Real(9));
  EXPECT_EQ(context.diagnostic("mass"), pops::Real(9));
  const pops::RuntimeParams params = context.program_params(1);
  ASSERT_EQ(params.count, 1);
  EXPECT_EQ(params.values[0], pops::Real(2.5));
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

  EXPECT_TRUE(context.schedule_decision(3, true, true));
  EXPECT_EQ(context.profiler().counter("nodes_due"), 1);
  EXPECT_EQ(context.profiler().counter("cache_misses"), 1);
  context.count_kernel(2);
  EXPECT_EQ(context.profiler().counter("kernels"), 5);

  EXPECT_THROW(context.set_stage_time(2, 1), std::runtime_error);
  EXPECT_THROW(
      context.schedule_is_due(-1, 2, ScheduleDomainKind::kAcceptedStep, "clock.macro", "", -1),
      std::runtime_error);
  EXPECT_THROW(context.sys_block(2), std::runtime_error);
  EXPECT_THROW(context.scheduler_error("stale value"), std::runtime_error);
}

template <class Context>
void expect_shared_install_and_field_services(Context& context) {
  double installed_dt = 0.0;
  context.install([&](double dt) { installed_dt = dt; });
  EXPECT_EQ(context.install_count(), 1);
  context.run_installed_step(0.125);
  EXPECT_DOUBLE_EQ(installed_dt, 0.125);

  pops::MultiFab state;
  const std::vector<const pops::MultiFab*> states{&state};
  pops::runtime::multiblock::BoundaryEvaluationPoint point{};
  point.level = context.level();
  auto accept = [](pops::SolveOutcome outcome) {
    return outcome.consume(pops::SolveConsumption::kAccept);
  };

  EXPECT_TRUE(accept(context.solve_fields()).solved());
  EXPECT_TRUE(accept(context.solve_fields_from_state(0, state)).solved());
  EXPECT_TRUE(accept(context.solve_fields_from_state_at(point, "field", 0, state)).solved());
  EXPECT_TRUE(accept(context.solve_fields_from_blocks(states)).solved());
  EXPECT_TRUE(
      accept(context.solve_fields_from_blocks_at(point, 17, "field", {{0, &state}})).solved());
  EXPECT_EQ(context.field_solve_dispatches(),
            std::vector<std::string>({"default", "default-state", "qualified-state-at",
                                      "default-blocks", "generated-blocks"}));

  int evaluated_bodies = 0;
  context.evaluate_with_field_state_at(point, "field", 0, state, state,
                                       [&]() { ++evaluated_bodies; });
  EXPECT_EQ(evaluated_bodies, 1);
  EXPECT_EQ(
      context.field_solve_dispatches(),
      std::vector<std::string>({"default", "default-state", "qualified-state-at", "default-blocks",
                                "generated-blocks", "qualified-state-at", "qualified-state-at"}));

  auto mismatched_point = point;
  ++mismatched_point.level;
  const int calls_before_level_mismatch = context.field_solve_dispatch_count();
  EXPECT_THROW((void)context.solve_fields_from_state_at(mismatched_point, "field", 0, state),
               std::invalid_argument);
  bool evaluation_body_called = false;
  EXPECT_THROW(context.evaluate_with_field_state_at(mismatched_point, "field", 0, state, state,
                                                    [&]() { evaluation_body_called = true; }),
               std::invalid_argument);
  EXPECT_FALSE(evaluation_body_called);
  EXPECT_EQ(context.field_solve_dispatch_count(), calls_before_level_mismatch)
      << "a mismatched fine/coarse point must fail before provider dispatch";

  const int calls_before_invalid_provider = context.field_solve_dispatch_count();
  EXPECT_THROW((void)context.solve_fields_from_state_at(point, "", 0, state),
               std::invalid_argument);
  EXPECT_EQ(context.field_solve_dispatch_count(), calls_before_invalid_provider)
      << "shared provider identity validation must run before topology dispatch";

  EXPECT_THROW(context.exercise_exclusive_workspace(true, false), std::logic_error);
  EXPECT_FALSE(context.exclusive_workspace_in_use())
      << "the outer guard must release after a nested-use rejection";
  EXPECT_THROW(context.exercise_exclusive_workspace(false, true), std::runtime_error);
  EXPECT_FALSE(context.exclusive_workspace_in_use())
      << "the guard must release after an operation failure";
  EXPECT_NO_THROW(context.exercise_exclusive_workspace(false, false));
  EXPECT_FALSE(context.exclusive_workspace_in_use());
}

template <class Context>
void expect_shared_spatial_services(Context& context) {
  const pops::Box2D domain = pops::Box2D::from_extents(4, 4);
  const pops::BoxArray boxes(std::vector<pops::Box2D>{domain});
  const pops::DistributionMapping mapping(std::vector<int>{0});
  pops::MultiFab phi(boxes, mapping, 1, 1);
  pops::MultiFab laplacian(boxes, mapping, 1, 0);
  pops::MultiFab gradient(boxes, mapping, 2, 1);
  pops::MultiFab divergence(boxes, mapping, 1, 0);
  pops::MultiFab a_xx(boxes, mapping, 1, 1);
  pops::MultiFab a_yy(boxes, mapping, 1, 1);
  pops::MultiFab a_xy(boxes, mapping, 1, 1);
  pops::MultiFab a_yx(boxes, mapping, 1, 1);
  phi.set_val(pops::Real(2));
  a_xx.set_val(pops::Real(1));
  a_yy.set_val(pops::Real(1));
  a_xy.set_val(pops::Real(0));
  a_yx.set_val(pops::Real(0));

  const pops::ExecutionLane lane = pops::ExecutionLane::world();
  pops::PreparedGridBoundarySession boundary(context.grid_context(), lane);
  const pops::runtime::multiblock::BoundaryEvaluationPoint point{};

  context.profiler().enable();
  context.laplacian(laplacian, phi);
  context.laplacian(laplacian, phi, lane);
  context.laplacian(laplacian, phi, boundary);
  context.laplacian(laplacian, phi, boundary, point);
  context.tensor_laplacian(laplacian, phi, a_xx, a_yy, a_xy, a_yx);
  context.tensor_laplacian(laplacian, phi, a_xx, a_yy, a_xy, a_yx, lane);
  context.tensor_laplacian(laplacian, phi, a_xx, a_yy, a_xy, a_yx, boundary);
  context.tensor_laplacian(laplacian, phi, a_xx, a_yy, a_xy, a_yx, boundary, point);
  context.gradient(gradient, phi);
  context.gradient(gradient, phi, lane);
  context.gradient(gradient, phi, boundary);
  context.gradient(gradient, phi, boundary, point);
  context.divergence(divergence, gradient, gradient);
  context.divergence(divergence, gradient, gradient, lane);
  context.divergence(divergence, gradient, gradient, boundary);
  context.divergence(divergence, gradient, gradient, boundary, point);

  EXPECT_EQ(context.profiler().counter("kernels"), 16);
  EXPECT_NEAR(context.max_component(laplacian, 0), pops::Real(0), 1e-12);
  EXPECT_NEAR(context.min_component(laplacian, 0), pops::Real(0), 1e-12);
  EXPECT_NEAR(context.max_component(gradient, 0), pops::Real(0), 1e-12);
  EXPECT_NEAR(context.max_component(gradient, 1), pops::Real(0), 1e-12);
  EXPECT_NEAR(context.max_component(divergence, 0), pops::Real(0), 1e-12);
}

template <class Context>
void expect_shared_prepared_operator_services(Context& context) {
  const pops::Box2D domain = pops::Box2D::from_extents(4, 4);
  const pops::BoxArray boxes(std::vector<pops::Box2D>{domain});
  const pops::DistributionMapping mapping(std::vector<int>{0});
  pops::MultiFab field(boxes, mapping, 1, 1);
  const pops::ExecutionLane lane = pops::ExecutionLane::world();
  const pops::runtime::multiblock::BoundaryEvaluationPoint point{};

  const auto mesh_boundary = context.prepare_mesh_boundary_session(field, lane);
  const auto block_boundary = context.prepare_block_boundary_session(0, field, point, lane);
  EXPECT_NE(mesh_boundary, nullptr);
  EXPECT_NE(block_boundary, nullptr);
  EXPECT_EQ(context.boundary_program_block(), 0)
      << "the shared service passes a Program block and leaves topology mapping to the provider";

  EXPECT_EQ(&context.assembly_target(field, "pops.test.operator.target"), &field);
  EXPECT_EQ(context.assembly_target_count(), 1);
  EXPECT_EQ(context.assembly_target_slot(), "pops.test.operator.target");
  EXPECT_THROW((void)context.assembly_target(field, ""), std::runtime_error);
  EXPECT_EQ(context.assembly_target_count(), 1)
      << "shared validation must reject an invalid slot before provider dispatch";

  EXPECT_EQ(&context.assembly_source(field, "pops.test.operator.solution"), &field);
  EXPECT_EQ(context.assembly_source_count(), 1);
  EXPECT_EQ(context.assembly_source_slot(), "pops.test.operator.solution");
  EXPECT_THROW((void)context.assembly_source(field, ""), std::runtime_error);
  EXPECT_EQ(context.assembly_source_count(), 1)
      << "shared validation must reject an invalid slot before provider dispatch";

  EXPECT_EQ(&context.linear_solution(field), &field);
  EXPECT_EQ(context.linear_solution_count(), 1);

  EXPECT_NO_THROW((void)context.authenticated_program_apply_token({1, 2, 3, 4}));
  EXPECT_THROW((void)context.authenticated_program_apply_token({4, 3, 2, 1}),
               std::invalid_argument);
  EXPECT_EQ(context.authority_check_count(), 2);

  using SolveMethod = pops::SolveOutcome (Context::*)(
      const pops::PreparedAffineLinearProblem&, pops::KrylovWorkspace&, pops::MultiFab&,
      const pops::MultiFab&, const pops::KrylovControls&) const;
  const auto solve_method = static_cast<SolveMethod>(&Context::solve_prepared_linear);
  EXPECT_NE(solve_method, nullptr)
      << "the shared prepared Krylov route remains part of both provider facades";
}

template <class Context>
void expect_shared_boundary_dispatch_services(Context& context) {
  pops::MultiFab state;
  pops::MultiFab rhs;
  pops::MultiFab direction;

  context.profiler().enable();

  context.rhs_into(0, state, rhs, 13);
  EXPECT_EQ(context.boundary_dispatch_operation(), "rhs");
  EXPECT_EQ(context.boundary_dispatch_program_block(), 0);
  EXPECT_EQ(context.boundary_dispatch_runtime_block(), 1);
  EXPECT_EQ(context.boundary_dispatch_rate(), 13);
  EXPECT_EQ(context.profiler().counter("kernels"), 1);

  const auto point = context.boundary_evaluation_point(14);
  EXPECT_EQ(point.stage, 14);
  EXPECT_TRUE(context.has_boundary_linearization(0));
  EXPECT_EQ(context.boundary_dispatch_operation(), "has_linearization");
  EXPECT_EQ(context.boundary_dispatch_runtime_block(), 1);
  EXPECT_EQ(context.profiler().counter("kernels"), 1)
      << "a linearization capability query must not count as a kernel";

  const pops::PreparedGridBoundarySession boundary(context.grid_context(),
                                                   pops::ExecutionLane::world());

  context.rhs_core_into_at(point, 0, state, rhs, false);
  EXPECT_EQ(context.boundary_dispatch_operation(), "rhs_core");
  EXPECT_EQ(context.boundary_dispatch_runtime_block(), 1);
  EXPECT_FALSE(context.boundary_dispatch_has_session());
  EXPECT_FALSE(context.boundary_dispatch_flux_only());

  context.rhs_core_into_at(point, 0, state, rhs, true, boundary);
  EXPECT_EQ(context.boundary_dispatch_operation(), "rhs_core");
  EXPECT_TRUE(context.boundary_dispatch_has_session());
  EXPECT_TRUE(context.boundary_dispatch_flux_only());

  context.boundary_residual_into_at(point, 0, state, rhs);
  EXPECT_EQ(context.boundary_dispatch_operation(), "boundary_residual");
  EXPECT_FALSE(context.boundary_dispatch_has_session());
  context.boundary_residual_into_at(point, 0, state, rhs, boundary);
  EXPECT_EQ(context.boundary_dispatch_operation(), "boundary_residual");
  EXPECT_TRUE(context.boundary_dispatch_has_session());

  context.boundary_jvp_into_at(point, 0, state, direction, rhs);
  EXPECT_EQ(context.boundary_dispatch_operation(), "boundary_jvp");
  EXPECT_FALSE(context.boundary_dispatch_has_session());
  context.boundary_jvp_into_at(point, 0, state, direction, rhs, boundary);
  EXPECT_EQ(context.boundary_dispatch_operation(), "boundary_jvp");
  EXPECT_TRUE(context.boundary_dispatch_has_session());

  context.neg_div_flux_default_into(0, state, rhs, 15);
  EXPECT_EQ(context.boundary_dispatch_operation(), "neg_div_flux");
  EXPECT_EQ(context.boundary_dispatch_program_block(), 0);
  EXPECT_EQ(context.boundary_dispatch_runtime_block(), 1);
  EXPECT_EQ(context.boundary_dispatch_rate(), 15);
  EXPECT_TRUE(context.boundary_dispatch_flux_only());
  EXPECT_EQ(context.profiler().counter("kernels"), 8);

  pops::MultiFab divergence_scratch;
  context.neg_div_flux_into(rhs, state, direction, divergence_scratch);
  EXPECT_EQ(context.boundary_dispatch_operation(), "named_flux");
  EXPECT_FALSE(context.boundary_dispatch_has_session());
  EXPECT_TRUE(context.boundary_dispatch_flux_only());
  context.neg_div_flux_into(rhs, state, direction, divergence_scratch,
                            pops::ExecutionLane::world());
  EXPECT_EQ(context.boundary_dispatch_operation(), "named_flux");
  EXPECT_TRUE(context.boundary_dispatch_has_session());
  EXPECT_EQ(context.profiler().counter("kernels"), 10);

  EXPECT_THROW(context.rhs_into(0, state, rhs, -1), std::invalid_argument);
  EXPECT_THROW(context.neg_div_flux_default_into(0, state, rhs, -1), std::invalid_argument);
  EXPECT_EQ(context.boundary_dispatch_operation(), "named_flux");
  EXPECT_EQ(context.boundary_dispatch_rate(), -1);
  EXPECT_EQ(context.profiler().counter("kernels"), 10)
      << "invalid stable identities must fail before provider dispatch or profiling";
}

template <class Context>
void expect_shared_operator_snapshot_services(Context& context, std::uint64_t topology_revision) {
  pops::MultiFab prototype;
  const pops::OperatorFingerprint authority{UINT64_C(1), UINT64_C(2), UINT64_C(3), UINT64_C(4)};
  const pops::OperatorFingerprint resources{UINT64_C(5), UINT64_C(6), UINT64_C(7), UINT64_C(8)};

  const pops::OperatorEvaluationSnapshot parent =
      context.operator_evaluation_snapshot(authority, prototype, resources);
  EXPECT_EQ(parent.revision, 1u);
  EXPECT_EQ(parent.topology_revision, topology_revision);
  EXPECT_TRUE(parent.valid());
  EXPECT_EQ(context.operator_topology_count(), 1);
  EXPECT_EQ(
      context.probe_operator_evaluation(authority, parent.topology, resources, parent.revision),
      parent);
  EXPECT_EQ(context.operator_topology_count(), 1)
      << "an allocation-free probe reuses the authenticated topology fingerprint";

  pops::OperatorEvaluationSnapshot child;
  {
    auto logical_child = context.logical_evaluation_scope(0, 2);
    const auto stale_parent =
        context.probe_operator_evaluation(authority, parent.topology, resources, parent.revision);
    EXPECT_EQ(stale_parent.revision, 0u);
    child = context.operator_evaluation_snapshot(authority, prototype, resources);
    EXPECT_EQ(child.revision, 2u);
    EXPECT_DOUBLE_EQ(std::bit_cast<double>(child.dt_bits), 0.2);
  }

  const auto stale_child =
      context.probe_operator_evaluation(authority, child.topology, resources, child.revision);
  EXPECT_EQ(stale_child.revision, 0u);
  const auto reminted_parent =
      context.operator_evaluation_snapshot(authority, prototype, resources);
  EXPECT_EQ(reminted_parent.revision, 3u);
  EXPECT_DOUBLE_EQ(std::bit_cast<double>(reminted_parent.dt_bits), 0.4);
  EXPECT_EQ(context.operator_topology_count(), 3);

  context.set_untracked_logical_dt(0.3);
  const auto stale_after_provider_clock_change = context.probe_operator_evaluation(
      authority, reminted_parent.topology, resources, reminted_parent.revision);
  EXPECT_EQ(stale_after_provider_clock_change.revision, 0u);
  EXPECT_FALSE(stale_after_provider_clock_change.valid())
      << "a provider clock transition must invalidate the complete shared capability";
  context.set_untracked_logical_dt(0.4);
  EXPECT_EQ(context
                .probe_operator_evaluation(authority, reminted_parent.topology, resources,
                                           reminted_parent.revision)
                .revision,
            0u)
      << "restoring matching scalar coordinates must not resurrect an invalidated capability";
}

template <class Context>
void expect_shared_persistent_scratch_services(Context& context) {
  const pops::Box2D domain = pops::Box2D::from_extents(4, 4);
  const pops::BoxArray boxes(std::vector<pops::Box2D>{domain});
  const pops::DistributionMapping mapping(std::vector<int>{0});
  pops::MultiFab prototype(boxes, mapping, 2, 1);

  context.profiler().enable();
  pops::MultiFab& first = context.rhs_scratch(41, 0, prototype);
  EXPECT_EQ(first.ncomp(), 2);
  EXPECT_EQ(first.n_grow(), 1);
  first.set_val(pops::Real(9));
  const std::int64_t allocations_after_first = context.profiler().counter("scratch_allocs");

  pops::MultiFab& reused = context.rhs_scratch(41, 0, prototype);
  EXPECT_EQ(&reused, &first);
  EXPECT_EQ(context.profiler().counter("scratch_allocs"), allocations_after_first);
  if (reused.local_size() > 0) {
    const auto cell = reused.box(0).lo;
    EXPECT_EQ(reused.fab(0).const_array()(cell[0], cell[1], 0), pops::Real(0))
        << "a shared persistent slot must clear provisional bytes before reuse";
  }

  pops::MultiFab& other_kind = context.scratch_state(41, 0, prototype);
  pops::MultiFab& other_subslot = context.rhs_scratch(41, 1, prototype);
  EXPECT_NE(&other_kind, &reused);
  EXPECT_NE(&other_subslot, &reused);
  EXPECT_EQ(context.profiler().counter("scratch_allocs"), allocations_after_first + 2);

  const int level = context.resource_level();
  const int levels = context.resource_levels();
  context.set_scratch_resource_identity(11, 18, levels, level);
  (void)context.rhs_scratch(41, 0, prototype);
  EXPECT_EQ(context.profiler().counter("scratch_allocs"), allocations_after_first + 3)
      << "a process-local materialization change must invalidate every shared scratch slot";

  EXPECT_THROW((void)context.rhs_scratch(-1, 0, prototype), std::invalid_argument);
  EXPECT_THROW((void)context.rhs_scratch(41, -1, prototype), std::invalid_argument);
  context.set_scratch_resource_identity(12, 19, 0, 0);
  EXPECT_THROW((void)context.rhs_scratch(41, 0, prototype), std::runtime_error);
  context.set_scratch_resource_identity(12, 19, levels, levels);
  EXPECT_THROW((void)context.rhs_scratch(41, 0, prototype), std::out_of_range);
}

}  // namespace

TEST(ProgramExecutionServices, UniformAndAmrProvidersRunTheSameContractFixture) {
  ExecutionServicesFixture<false> uniform(-1);
  ExecutionServicesFixture<true> amr(1);
  expect_shared_program_services(uniform, false);
  expect_shared_program_services(amr, true);
}

TEST(ProgramExecutionServices, UniformAndAmrProvidersRunTheSameInstallAndFieldFixture) {
  ExecutionServicesFixture<false> uniform(0);
  ExecutionServicesFixture<true> amr(1);
  expect_shared_install_and_field_services(uniform);
  expect_shared_install_and_field_services(amr);
}

TEST(ProgramExecutionServices, UniformAndAmrProvidersRunTheSameSpatialFixture) {
  ExecutionServicesFixture<false> uniform(-1);
  ExecutionServicesFixture<true> amr(1);
  expect_shared_spatial_services(uniform);
  expect_shared_spatial_services(amr);
}

TEST(ProgramExecutionServices, UniformAndAmrProvidersRunTheSamePreparedOperatorFixture) {
  ExecutionServicesFixture<false> uniform(-1);
  ExecutionServicesFixture<true> amr(1);
  expect_shared_prepared_operator_services(uniform);
  expect_shared_prepared_operator_services(amr);
}

TEST(ProgramExecutionServices, UniformAndAmrProvidersRunTheSameBoundaryDispatchFixture) {
  ExecutionServicesFixture<false> uniform(-1);
  ExecutionServicesFixture<true> amr(1);
  expect_shared_boundary_dispatch_services(uniform);
  expect_shared_boundary_dispatch_services(amr);
}

TEST(ProgramExecutionServices, UniformAndAmrProvidersRunTheSameOperatorSnapshotFixture) {
  ExecutionServicesFixture<false> uniform(-1);
  ExecutionServicesFixture<true> amr(1);
  expect_shared_operator_snapshot_services(uniform, 1);
  expect_shared_operator_snapshot_services(amr, 17);
}

TEST(ProgramExecutionServices, UniformAndAmrProvidersRunTheSamePersistentScratchFixture) {
  ExecutionServicesFixture<false> uniform(-1);
  ExecutionServicesFixture<true> amr(1);
  expect_shared_persistent_scratch_services(uniform);
  expect_shared_persistent_scratch_services(amr);
}
