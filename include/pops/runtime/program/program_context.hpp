#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pops/core/foundation/types.hpp>     // Real, POPS_HD
#include <pops/runtime/program/profiler.hpp>  // Profiler / ProfileScope (per-node timing, ADC-459)
#include <pops/runtime/program/step_transaction.hpp>
#include <pops/runtime/program/wire_ids.hpp>   // stable compiled-Program numeric protocol
#include <pops/mesh/boundary/physical_bc.hpp>  // fill_ghosts (periodic / physical halo exchange)
#include <pops/mesh/execution/for_each.hpp>  // for_each_cell (per-cell coeff / reconstruct kernels + negated divergence copy)
#include <pops/mesh/geometry/geometry.hpp>  // Geometry (mesh metric of the Laplacian / gradient)
#include <pops/mesh/layout/field_distribution.hpp>  // FieldDistribution
#include <pops/mesh/storage/fab2d.hpp>              // Array4 / ConstArray4 (per-cell handles)
#include <pops/mesh/storage/mf_arith.hpp>           // saxpy (linear combine over a MultiFab)
#include <pops/mesh/storage/multifab.hpp>           // MultiFab
#include <pops/parallel/execution_lane.hpp>
#include <pops/parallel/solve_report_consensus.hpp>
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/linear/pure_field_algebra.hpp>
#include <pops/numerics/elliptic/linear/vector_distribution.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>  // apply_laplacian (shared 5-point matvec)
#include <pops/numerics/elliptic/polar/polar_tensor_operator.hpp>  // metric-aware generated tensor solve
#include <pops/runtime/config/runtime_params.hpp>  // RuntimeParams (compiled-Program runtime params, ADC-510)
#include <pops/runtime/context/grid_context.hpp>    // GridContext (System aux seam)
#include <pops/runtime/program/clock_schedule.hpp>  // nested logical-clock cursor validation
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/system.hpp>  // System (the runtime this facade forwards to)

/// @file
/// @brief ProgramContext -- the C++-side facade a generated problem.so calls to run a compiled time
///        Program during sim.step(dt) (epic ADC-399, ADC-401 Phase 2b).
///
/// It REIMPLEMENTS NOTHING. Each method forwards to an existing pops::System primitive:
///   install(fn)          -> System::install_program_step(fn)   (registers the macro-step body)
///   solve_fields()       -> System::solve_fields()             (elliptic solve + aux at current U)
///   solve_fields_from_state(b, U) -> System::solve_fields_from_state(b, U) (aux at a stage state)
///   n_blocks()           -> System::n_blocks()
///   state(b)             -> System::block_state(b)             (the block's live MultiFab, zero-copy)
///   rhs_into(b, U, R, rate_id) -> System::block_rhs_into_at(...) (point-qualified -div F + S)
///   neg_div_flux_default_into(b, U, R, rate_id) -> point-qualified -div F with no source
///   axpy(U, a, R)        -> pops::saxpy(U, a, R)                (U <- U + a R, device-dispatched)
///
/// The Program composes the chain (e.g. Forward Euler = solve_fields(); for each block:
/// rhs_into(b, U, R, rate_id); axpy(U, dt, R)) and installs it via install(...). The .so NEVER touches
/// System::Impl / Array4 / fill_boundary / the elliptic solver / Kokkos / MPI / CFL / substeps.
///
/// IDIOM: ProgramContext is a plain (non-template) class holding a System*. A generated .so receives
/// the typed System facade across the authenticated dlopen boundary and asks the shared provider
/// factory to construct this topology/storage provider; it reaches per-block storage through the
/// System's public accessors because System::Impl is private to the _pops translation unit.
namespace pops {
namespace runtime {
namespace program {

class ProgramContext : public ProgramExecutionServices<ProgramContext> {
 public:
  explicit ProgramContext(System* sys) : sys_(sys) {}

  /// Start one generated Program body.  The native stepper supplies the accepted local dt; every
  /// boundary evaluation in the body derives its physical time from this exact value and the
  /// authored rational stage fraction.
  void begin_step(double dt) const {
    if (!std::isfinite(dt) || dt <= 0.0)
      throw std::invalid_argument("Program boundary clock requires a finite positive dt");
    current_dt_ = dt;
    stage_time_ = amr::Rational(0, 1);
    logical_phase_begin_ = amr::Rational(0, 1);
    logical_phase_span_ = amr::Rational(1, 1);
    logical_physical_time_offset_ = 0.0;
  }

 private:
  SolveOutcome program_execution_solve_fields_outcome_() const {
    // No count_kernel() here: System's private in-place default provider seam already counts it.
    // The from_state/from_blocks/named seams below do not, so those routes count explicitly.
    sys_->prepare_default_field_publication_storage_();
    return run_field_solve_transaction_([&]() { return sys_->solve_fields_in_place_(); });
  }
  /// Per-stage field solve (ADC-409): re-solve the elliptic fields and re-fill the shared aux from
  /// block @p b's STAGE state @p u_stage (not its live state), so a field-coupled multi-stage
  /// Program's stage k reads phi solved from stage k's own state. Forwards to
  /// System::solve_fields_from_state. With b = 0 and u_stage = U^n (the first stage) it matches
  /// solve_fields(); the codegen lowers every solve_fields op to this, passing the stage's state var.
  SolveOutcome program_execution_solve_fields_from_state_outcome_(int b, MultiFab& u_stage) const {
    count_kernel();
    sys_->prepare_default_field_publication_storage_();
    return run_field_solve_transaction_(
        [&]() { return sys_->solve_fields_from_state_in_place_(sys_block(b), u_stage); });
  }
  SolveOutcome program_execution_field_solve_from_state_at_outcome_(
      const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
      int b, MultiFab& u_stage) const {
    count_kernel();
    sys_->prepare_named_field_publication_storage_(provider_slot);
    return run_field_solve_transaction_([&]() {
      return sys_->solve_fields_from_state_at_in_place_(point, provider_slot, sys_block(b),
                                                        u_stage);
    });
  }
  /// Named multi-elliptic field solve (ADC-428): re-solve the SECOND elliptic field @p field from block
  /// @p b's stage state @p u_stage and write its phi (+ centered grad) into the field's OWN aux
  /// components (distinct from the shared phi/grad the default solve_fields fills). Forwards to
  /// System::solve_fields_from_state(field, b, u_stage). The codegen lowers
  /// P.solve_fields(field=name, state=U) to this; a default (unnamed) solve_fields keeps the overload
  /// above, byte-identical.
  SolveOutcome program_execution_solve_named_field_from_state_outcome_(const std::string& field,
                                                                       int b,
                                                                       MultiFab& u_stage) const {
    count_kernel();
    sys_->prepare_named_field_publication_storage_(field);
    return run_field_solve_transaction_(
        [&]() { return sys_->solve_fields_from_state_in_place_(field, sys_block(b), u_stage); });
  }
  /// Coupled multi-block field solve (Spec 3 criterion 24, ADC-457): re-solve the elliptic fields and
  /// re-fill the shared aux from the SIMULTANEOUS stage states of MULTIPLE blocks at once -- the system
  /// Poisson RHS is Sum_s elliptic_rhs_s(U_s), every coupled block reading its OWN stage state (not a
  /// single-target override). @p u_stages is indexed BY BLOCK INDEX (size == n_blocks()); a nullptr
  /// entry uses that block's live state. Forwards to System::solve_fields_from_blocks. The codegen
  /// Manual callers may provide the historical pointer vector. Generated Programs use the exact-IR
  /// initializer-list overload below, which fills the same context-owned workspace without allocating
  /// a pointer vector in the step body. This is the multi-target counterpart of solve_fields_from_state.
  SolveOutcome program_execution_solve_fields_from_blocks_outcome_(
      const std::vector<const MultiFab*>& u_stages) const {
    count_kernel();
    // The codegen builds @p u_stages indexed BY PROGRAM block index (a stage state slotted at its own
    // Program index, the rest nullptr). The System solver expects it indexed by SYSTEM block index, so
    // re-slot each Program entry p at its name-matched System index sys_block(p) (Spec 3 criterion 23,
    // ADC-457). Even an order-matching Program carries an explicit identity map.
    const std::vector<int>& block_map = sys_->program_block_map();
    if (block_map.empty())
      throw block_map_error_(
          "ProgramContext::solve_fields_from_blocks: no explicit program-to-system block map is "
          "installed; positional block identity is not supported");
    if (u_stages.size() < block_map.size())
      throw block_map_error_("ProgramContext::solve_fields_from_blocks: received " +
                             std::to_string(u_stages.size()) +
                             " Program stage slots for an explicit block map with " +
                             std::to_string(block_map.size()) + " entries");
    FieldSolveWorkspace& workspace = manual_default_field_solve_workspace_();
    fill_manual_field_stages_(workspace, u_stages, /*require_exact_size=*/false);
    // Iterate the PROGRAM block indices [0, m.size()) -- NOT u_stages.size(), which is the larger
    // SYSTEM block count. The codegen sizes u_stages to ctx.n_blocks() but only fills Program slots
    // [0, n_program_blocks); when the System has MORE blocks than the Program declares (a subset
    // install), walking the System-sized range would re-map the nullptr padding through the identity
    // fallthrough and clobber real entries. m[p] is Program block p's System index (install-validated
    // in range); the unlisted System slots stay nullptr = their live state. sys_block validates every
    // mapped value before it is used as a vector index.
    sys_->prepare_default_field_publication_storage_();
    return run_field_solve_transaction_(
        [&]() { return solve_default_field_workspace_(workspace); });
  }

  SolveOutcome program_execution_solve_named_field_from_blocks_outcome_(
      const std::string& field, const std::vector<const MultiFab*>& u_stages) const {
    count_kernel();
    FieldSolveWorkspace& workspace = manual_named_field_solve_workspace_(field);
    fill_manual_field_stages_(workspace, u_stages, /*require_exact_size=*/true);
    sys_->prepare_named_field_publication_storage_(field);
    return run_field_solve_transaction_(
        [&]() { return solve_named_field_workspace_(field, workspace); });
  }

  /// Allocation-free generated route.  The exact IR identity owns one context-local pointer/snapshot
  /// workspace; @p field and the ordered Program block pack are authenticated on every replay.  The
  /// old vector overloads above remain available for manual C++ callers.
  SolveOutcome program_execution_solve_generated_field_from_blocks_outcome_(
      std::int64_t value_id, std::string_view field,
      std::initializer_list<FieldStageOverride> overrides) const {
    count_kernel();
    FieldSolveWorkspace& workspace = generated_field_solve_workspace_(value_id, field, overrides);
    sys_->prepare_named_field_publication_storage_(workspace.generated_field_identity);
    return run_field_solve_transaction_([&]() {
      return solve_named_field_workspace_(workspace.generated_field_identity, workspace);
    });
  }

 public:
  /// Reconstruct one primary-clock retained state at an exact target-clock coordinate. The
  /// bracketing slots and every intervening accepted interval come from the native history ledger;
  /// no Python callback, current-state alias, or fixed-dt inference participates.
  void interpolate_history_linear(MultiFab& out, const std::string& name, int max_lag, int owner,
                                  const std::string& source_clock, const std::string& target_clock,
                                  int target_step, Real target_offset) const {
    (void)sys_block(owner);
    if (max_lag < 1)
      throw std::invalid_argument(
          "linear history interpolation requires at least one retained lag");
    if (!std::isfinite(static_cast<double>(target_offset)))
      throw std::invalid_argument("linear history interpolation offset must be finite");
    if (!sys_->history_initialized(name))
      throw std::runtime_error(
          "linear history interpolation requires an initialized native history");
    const double source_ticks = static_cast<double>(clock_schedule_.ticks_per_macro(source_clock));
    const double target_ticks = static_cast<double>(clock_schedule_.ticks_per_macro(target_clock));
    const double coordinate =
        (static_cast<double>(target_step) + static_cast<double>(target_offset)) * source_ticks /
        target_ticks;
    if (!std::isfinite(coordinate) || coordinate > 0.0 ||
        coordinate < -static_cast<double>(max_lag))
      throw std::runtime_error(
          "linear history interpolation target lies outside retained timestamps");

    if (coordinate == 0.0) {
      MultiFab& exact = history(name, 0);
      lincomb(out, Real(1), exact, Real(0), exact);
      return;
    }
    const int older_lag = static_cast<int>(std::ceil(-coordinate));
    if (older_lag < 1 || older_lag > max_lag)
      throw std::runtime_error("linear history interpolation could not select bracketing slots");

    double newer_time = static_cast<double>(physical_time());
    double older_time = newer_time;
    double bracket_dt = 0.0;
    for (int lag = 1; lag <= older_lag; ++lag) {
      const double interval = sys_->history_slot_dt(name, lag);
      if (!std::isfinite(interval) || interval <= 0.0)
        throw std::runtime_error(
            "linear history interpolation requires positive exact slot timestamps");
      bracket_dt = interval;
      older_time = newer_time - interval;
      if (lag != older_lag)
        newer_time = older_time;
    }
    const double logical_fraction = coordinate + static_cast<double>(older_lag);
    const double target_time = older_time + logical_fraction * bracket_dt;
    const double timestamp_fraction = (target_time - older_time) / (newer_time - older_time);
    if (!std::isfinite(timestamp_fraction) || timestamp_fraction < 0.0 || timestamp_fraction > 1.0)
      throw std::runtime_error(
          "linear history interpolation target does not bracket native timestamps");

    MultiFab& older = history(name, older_lag);
    MultiFab& newer = history(name, older_lag - 1);
    const Real alpha = static_cast<Real>(timestamp_fraction);
    lincomb(out, Real(1) - alpha, older, alpha, newer);
  }

 private:
  struct FieldSolveWorkspace {
    std::vector<int> program_to_system;
    std::vector<const MultiFab*> program_stages;
    std::vector<const MultiFab*> system_stages;
    std::vector<int> expected_program_blocks;
    std::string generated_field_identity;
    bool expected_program_blocks_initialized = false;
    bool in_use = false;
  };

  struct FieldPublicationTransaction {
    System* system = nullptr;
    bool active = false;

    void validate_accept() {
      if (!active || system == nullptr)
        throw std::logic_error("Program field publication has no staged candidate");
      system->validate_field_publication_candidate();
    }

    void accept() noexcept {
      if (!active || system == nullptr)
        std::terminate();
      system->accept_field_publication_candidate();
      system = nullptr;
      active = false;
    }

    void rollback() noexcept {
      if (!active)
        return;
      try {
        if (system == nullptr)
          std::terminate();
        system->rollback_field_publication_transaction();
      } catch (...) {
        std::terminate();
      }
      release();
    }

    void release() noexcept {
      system = nullptr;
      active = false;
    }
  };

  struct FieldSolveWorkspaceRegistry {
    FieldSolveWorkspace manual_default;
    std::map<std::string, FieldSolveWorkspace, std::less<>> manual_named;
    std::map<std::int64_t, FieldSolveWorkspace> generated;
    FieldPublicationTransaction publication;
  };

  void capture_field_publication_(FieldPublicationTransaction& transaction) const {
    // SystemFieldSolver's uniform reductions use MPI_COMM_WORLD, so this transaction must
    // authenticate and release on that exact communicator rather than inventing a private lane.
    if (all_reduce_max(transaction.active ? 1L : 0L) != 0)
      throw std::logic_error(
          "ProgramContext field solves are sequential until their SolveOutcome is consumed");
    if (all_reduce_max(sys_->field_publication_transaction_active_() ? 1L : 0L) != 0)
      throw std::logic_error(
          "System field solves are sequential until their prior SolveOutcome is consumed");

    long capture_failure_local = 0;
    try {
      sys_->begin_field_publication_transaction();
    } catch (...) {
      capture_failure_local = 1;
    }
    if (all_reduce_max(capture_failure_local) != 0) {
      try {
        sys_->rollback_field_publication_transaction();
      } catch (...) {
        std::terminate();
      }
      throw std::runtime_error(
          "ProgramContext field publication snapshot failed on at least one MPI rank");
    }
    transaction.system = sys_;
    transaction.active = true;
  }

  template <class Solve>
  SolveOutcome run_field_solve_transaction_(Solve&& solve) const {
    if (!field_solve_workspace_registry_)
      throw std::logic_error("Program field-solve workspace registry is unavailable");
    const std::shared_ptr<FieldSolveWorkspaceRegistry> registry = field_solve_workspace_registry_;
    FieldPublicationTransaction& transaction = registry->publication;
    capture_field_publication_(transaction);

    SolveReport report;
    std::exception_ptr solve_error;
    long solve_failure_local = 0;
    try {
      report = std::forward<Solve>(solve)();
    } catch (...) {
      solve_error = std::current_exception();
      solve_failure_local = 1;
    }
    if (all_reduce_max(solve_failure_local) != 0) {
      transaction.rollback();
      if (n_ranks() == 1 && solve_error != nullptr)
        std::rethrow_exception(solve_error);
      throw std::runtime_error("ProgramContext field solver failed on at least one MPI rank");
    }
    const bool malformed = !solve_report_is_publishable(report, std::numeric_limits<int>::max());
    if (all_reduce_max(malformed ? 1L : 0L) != 0) {
      transaction.rollback();
      throw std::runtime_error("ProgramContext field solver published a malformed SolveReport");
    }
    ExactSolveReportConsensusScratch consensus;
    if (!consensus.agrees(report)) {
      transaction.rollback();
      throw std::runtime_error("ProgramContext field solver report differs between MPI ranks");
    }
    if (!report.solved_value_available()) {
      // The uniform backends already restore their potential warm start on a failed report. This
      // outer graph transaction additionally restores the complete shared aux channel before the
      // failure can be inspected or acted upon.
      transaction.rollback();
      return SolveOutcome::collective_world(std::move(report));
    }

    long staging_failure_local = 0;
    try {
      sys_->stage_field_publication_candidate();
    } catch (...) {
      staging_failure_local = 1;
    }
    if (all_reduce_max(staging_failure_local) != 0) {
      transaction.rollback();
      throw std::runtime_error(
          "ProgramContext field candidate staging failed on at least one MPI rank");
    }

    // Aux and every backend potential now contain the previous accepted state. The shared registry
    // keeps the callback context alive even if a direct C++ caller moves the outcome beyond this
    // ProgramContext; only collective Accept restores the staged candidate.
    return SolveOutcome::collective_world(
        std::move(report),
        SolveOutcome::PublicationHooks{
            &transaction,
            [](void* context) noexcept {
              static_cast<FieldPublicationTransaction*>(context)->accept();
            },
            nullptr,
            [](void* context) noexcept {
              static_cast<FieldPublicationTransaction*>(context)->rollback();
            },
            std::static_pointer_cast<void>(registry),
            [](void* context) {
              static_cast<FieldPublicationTransaction*>(context)->validate_accept();
            }});
  }

  void prepare_field_solve_structure_(FieldSolveWorkspace& workspace) const {
    const std::vector<int>& block_map = sys_->program_block_map();
    if (block_map.empty())
      throw block_map_error_(
          "ProgramContext::solve_fields_from_blocks: no explicit program-to-system block map is "
          "installed; positional block identity is not supported");
    const std::size_t system_blocks = static_cast<std::size_t>(sys_->n_blocks());
    const bool unchanged = workspace.program_to_system == block_map &&
                           workspace.program_stages.size() == block_map.size() &&
                           workspace.system_stages.size() == system_blocks;
    if (unchanged)
      return;

    // Authenticate the complete map before releasing a previously valid workspace.  A malformed map
    // cannot leave a partially reconfigured context behind.
    for (std::size_t p = 0; p < block_map.size(); ++p) {
      const int mapped = sys_block(static_cast<int>(p));
      for (std::size_t previous = 0; previous < p; ++previous) {
        if (block_map[previous] == mapped)
          throw block_map_error_("ProgramContext::solve_fields_from_blocks: Program blocks " +
                                 std::to_string(previous) + " and " + std::to_string(p) +
                                 " both map to System block " + std::to_string(mapped));
      }
    }
    workspace.program_to_system.assign(block_map.begin(), block_map.end());
    workspace.program_stages.assign(block_map.size(), nullptr);
    workspace.system_stages.assign(system_blocks, nullptr);
    workspace.expected_program_blocks.clear();
    workspace.expected_program_blocks_initialized = false;
  }

  void require_program_stage_layout_(int program_block, const MultiFab& stage) const {
    const MultiFab& live = sys_->block_state(sys_block(program_block));
    if (!field_layout_matches_(stage, live, live.ncomp(), live.n_grow()))
      throw std::invalid_argument(
          "Program simultaneous field solve requires each stage state to match its block's exact "
          "distributed layout");
    const int qualified_system_block = sys_block(program_block);
    for (int other = 0; other < sys_->n_blocks(); ++other) {
      if (other == qualified_system_block)
        continue;
      if (&stage == &sys_->block_state(other))
        throw std::invalid_argument(
            "Program stage override cannot borrow another block's live state");
    }
  }

  void fill_manual_field_stages_(FieldSolveWorkspace& workspace,
                                 const std::vector<const MultiFab*>& stages,
                                 bool require_exact_size) const {
    prepare_field_solve_structure_(workspace);
    const std::size_t required = workspace.program_to_system.size();
    if ((require_exact_size && stages.size() != required) ||
        (!require_exact_size && stages.size() < required))
      throw std::runtime_error(
          "ProgramContext::solve_fields_from_blocks: stage vector size mismatch");
    std::fill(workspace.program_stages.begin(), workspace.program_stages.end(), nullptr);
    for (std::size_t p = 0; p < required; ++p) {
      const MultiFab* stage = stages[p];
      if (stage != nullptr)
        require_program_stage_layout_(static_cast<int>(p), *stage);
      workspace.program_stages[p] = stage;
    }
  }

  FieldSolveWorkspace& manual_default_field_solve_workspace_() const {
    if (!field_solve_workspace_registry_)
      throw std::logic_error("Program field-solve workspace registry is unavailable");
    FieldSolveWorkspace& workspace = field_solve_workspace_registry_->manual_default;
    prepare_field_solve_structure_(workspace);
    return workspace;
  }

  FieldSolveWorkspace& manual_named_field_solve_workspace_(const std::string& field) const {
    if (field.empty())
      throw std::invalid_argument(
          "Program named simultaneous field solve requires a field identity");
    if (!field_solve_workspace_registry_)
      throw std::logic_error("Program field-solve workspace registry is unavailable");
    auto& workspaces = field_solve_workspace_registry_->manual_named;
    auto found = workspaces.find(field);
    if (found == workspaces.end())
      found = workspaces.try_emplace(field).first;
    prepare_field_solve_structure_(found->second);
    return found->second;
  }

  FieldSolveWorkspace& generated_field_solve_workspace_(
      std::int64_t value_id, std::string_view field,
      std::initializer_list<FieldStageOverride> overrides) const {
    if (value_id < 0)
      throw std::invalid_argument(
          "generated simultaneous field solve requires a non-negative IR identity");
    if (field.empty())
      throw std::invalid_argument("generated simultaneous field solve requires a field identity");
    if (overrides.size() == 0)
      throw std::invalid_argument(
          "generated simultaneous field solve requires at least one stage override");
    if (!field_solve_workspace_registry_)
      throw std::logic_error("Program field-solve workspace registry is unavailable");

    auto [entry, inserted] = field_solve_workspace_registry_->generated.try_emplace(value_id);
    FieldSolveWorkspace& workspace = entry->second;
    if (inserted)
      workspace.generated_field_identity.assign(field.data(), field.size());
    else if (std::string_view(workspace.generated_field_identity) != field)
      throw std::logic_error(
          "generated simultaneous field solve IR identity was reused for a different field");
    prepare_field_solve_structure_(workspace);

    const bool learn_blocks = !workspace.expected_program_blocks_initialized;
    if (learn_blocks) {
      workspace.expected_program_blocks.clear();
      workspace.expected_program_blocks.reserve(overrides.size());
    } else if (workspace.expected_program_blocks.size() != overrides.size()) {
      throw std::logic_error(
          "generated simultaneous field solve IR identity changed its block pack");
    }
    std::fill(workspace.program_stages.begin(), workspace.program_stages.end(), nullptr);
    std::size_t ordinal = 0;
    for (const FieldStageOverride& override_value : overrides) {
      if (override_value.program_block < 0 ||
          static_cast<std::size_t>(override_value.program_block) >= workspace.program_stages.size())
        throw std::out_of_range("generated simultaneous field solve Program block is out of range");
      if (override_value.state == nullptr)
        throw std::invalid_argument(
            "generated simultaneous field solve stage override cannot be null");
      if (workspace.program_stages[static_cast<std::size_t>(override_value.program_block)] !=
          nullptr)
        throw std::invalid_argument(
            "generated simultaneous field solve contains a duplicate Program block");
      if (learn_blocks)
        workspace.expected_program_blocks.push_back(override_value.program_block);
      else if (workspace.expected_program_blocks[ordinal] != override_value.program_block)
        throw std::logic_error(
            "generated simultaneous field solve IR identity changed its ordered block pack");
      require_program_stage_layout_(override_value.program_block, *override_value.state);
      workspace.program_stages[static_cast<std::size_t>(override_value.program_block)] =
          override_value.state;
      ++ordinal;
    }
    workspace.expected_program_blocks_initialized = true;
    return workspace;
  }

  SolveReport solve_default_field_workspace_(FieldSolveWorkspace& workspace) const {
    std::fill(workspace.system_stages.begin(), workspace.system_stages.end(), nullptr);
    for (std::size_t p = 0; p < workspace.program_to_system.size(); ++p) {
      const int mapped = workspace.program_to_system[p];
      workspace.system_stages[static_cast<std::size_t>(mapped)] = workspace.program_stages[p];
    }
    return sys_->solve_fields_from_blocks_in_place_(workspace.system_stages);
  }

  SolveReport solve_named_field_workspace_(const std::string& field,
                                           FieldSolveWorkspace& workspace) const {
    ExclusiveUseGuard use(workspace.in_use,
                          "Program simultaneous field-solve workspace is already in use");
    std::fill(workspace.system_stages.begin(), workspace.system_stages.end(), nullptr);
    bool has_override = false;
    for (std::size_t p = 0; p < workspace.program_stages.size(); ++p) {
      if (workspace.program_stages[p] == nullptr)
        continue;
      workspace.system_stages[static_cast<std::size_t>(workspace.program_to_system[p])] =
          workspace.program_stages[p];
      has_override = true;
    }
    if (!has_override)
      throw std::runtime_error(
          "ProgramContext::solve_fields_from_blocks(field): no stage override was supplied");
    return sys_->solve_fields_from_blocks_in_place_(field, workspace.system_stages);
  }

  struct ScratchKey {
    ScratchKind kind = ScratchKind::Rhs;
    std::int64_t value_id = -1;
    int subslot = -1;

    friend bool operator<(const ScratchKey& lhs, const ScratchKey& rhs) noexcept {
      if (lhs.kind != rhs.kind)
        return lhs.kind < rhs.kind;
      if (lhs.value_id != rhs.value_id)
        return lhs.value_id < rhs.value_id;
      return lhs.subslot < rhs.subslot;
    }
  };

  struct ScratchRegistry {
    std::map<ScratchKey, MultiFab> fields;
  };

  MultiFab& program_scratch_for_(ScratchKind kind, std::int64_t value_id, int subslot,
                                 const MultiFab& prototype, int n_comp, int n_ghost) const {
    if (value_id < 0 || subslot < 0)
      throw std::invalid_argument(
          "Program persistent scratch requires non-negative IR value and sub-slot identities");
    if (!scratch_registry_)
      throw std::logic_error("Program persistent scratch registry is unavailable");
    const ScratchKey key{kind, value_id, subslot};
    auto [entry, inserted] = scratch_registry_->fields.try_emplace(key);
    MultiFab& field = entry->second;
    if (inserted || !field_layout_matches_(field, prototype, n_comp, n_ghost)) {
      field = MultiFab(prototype.box_array(), prototype.dmap(), n_comp, n_ghost);
      count_scratch(field);
    }
    field.set_val(Real(0));
    return field;
  }

  runtime::multiblock::BoundaryEvaluationPoint boundary_point_(int stage) const {
    require_rate_identity_(stage);
    if (primary_clock_.empty() || !std::isfinite(current_dt_) || current_dt_ <= 0.0)
      throw std::runtime_error("Program boundary evaluation has no prepared clock/dt");
    const amr::Rational evaluation_stage = logical_phase_begin_ + stage_time_ * logical_phase_span_;
    return {primary_clock_,
            static_cast<std::int64_t>(macro_step()),
            0,
            0,
            stage,
            evaluation_stage,
            current_dt_,
            static_cast<double>(physical_time()) + logical_physical_time_offset_ +
                stage_time_.value() * current_dt_};
  }

  friend class ProgramExecutionServices<ProgramContext>;

  void program_execution_install_(std::function<void(double)> step) const {
    sys_->install_program_step(std::move(step));
  }

  runtime::multiblock::BoundaryEvaluationPoint program_execution_boundary_point_(
      int stage_id) const {
    return boundary_point_(stage_id);
  }
  void program_execution_rhs_into_(int /*program_block*/, int runtime_block, MultiFab& state,
                                   MultiFab& rhs, int rate_id) const {
    sys_->block_rhs_into_at(boundary_point_(rate_id), runtime_block, state, rhs);
  }
  bool program_execution_has_boundary_linearization_(int runtime_block) const {
    return sys_->block_has_boundary_linearization(runtime_block);
  }
  void program_execution_require_cartesian_generated_operator_(int runtime_block,
                                                               const std::string& operation) const {
    sys_->require_cartesian_generated_operator(runtime_block, operation);
  }
  void program_execution_rhs_core_into_at_(
      const runtime::multiblock::BoundaryEvaluationPoint& point, int runtime_block, MultiFab& state,
      MultiFab& rhs, bool flux_only, const PreparedGridBoundarySession* boundary) const {
    if (boundary == nullptr)
      sys_->block_rhs_core_into_at(point, runtime_block, state, rhs, flux_only);
    else
      sys_->block_rhs_core_into_at(point, runtime_block, state, rhs, flux_only, *boundary);
  }
  void program_execution_boundary_residual_into_at_(
      const runtime::multiblock::BoundaryEvaluationPoint& point, int runtime_block, MultiFab& state,
      MultiFab& residual, const PreparedGridBoundarySession* boundary) const {
    if (boundary == nullptr)
      sys_->block_boundary_residual_into_at(point, runtime_block, state, residual);
    else
      sys_->block_boundary_residual_into_at(point, runtime_block, state, residual, *boundary);
  }
  void program_execution_boundary_jvp_into_at_(
      const runtime::multiblock::BoundaryEvaluationPoint& point, int runtime_block, MultiFab& state,
      const MultiFab& direction, MultiFab& result,
      const PreparedGridBoundarySession* boundary) const {
    if (boundary == nullptr)
      sys_->block_boundary_jvp_into_at(point, runtime_block, state, direction, result);
    else
      sys_->block_boundary_jvp_into_at(point, runtime_block, state, direction, result, *boundary);
  }
  void program_execution_neg_div_flux_default_into_(int /*program_block*/, int runtime_block,
                                                    MultiFab& state, MultiFab& rhs,
                                                    int rate_id) const {
    sys_->block_neg_div_flux_into_at(boundary_point_(rate_id), runtime_block, state, rhs);
  }
  void program_execution_neg_div_named_flux_into_(MultiFab& rhs, MultiFab& flux_x, MultiFab& flux_y,
                                                  MultiFab& divergence_scratch,
                                                  const ExecutionLane* lane) const {
    const GridContext context = sys_->grid_context();
    if (lane == nullptr) {
      fill_ghosts(flux_x, context.geom.domain, context.bc);
      fill_ghosts(flux_y, context.geom.domain, context.bc);
    } else {
      fill_ghosts(flux_x, context.geom.domain, context.bc, *lane);
      fill_ghosts(flux_y, context.geom.domain, context.bc, *lane);
    }
    if (!field_layout_matches_(divergence_scratch, rhs, 1, 0))
      throw std::invalid_argument(
          "Program named-flux divergence scratch must match the RHS distributed layout");
    for (int component = 0; component < rhs.ncomp(); ++component) {
      apply_divergence(flux_x, flux_y, context.geom, divergence_scratch, component, component);
      for (int local = 0; local < rhs.local_size(); ++local) {
        const ConstArray4 divergence = divergence_scratch.fab(local).const_array();
        Array4 result = rhs.fab(local).array();
        const int output_component = component;
        for_each_cell(rhs.box(local), [=] POPS_HD(int i, int j) {
          result(i, j, output_component) = -divergence(i, j, 0);
        });
      }
    }
  }
  void program_execution_rhs_group_(const RhsGroupBatch& batch) const {
    count_kernel(static_cast<std::int64_t>(batch.requests.size()));
    sys_->block_rhs_group(boundary_point_(batch.group_id), batch.runtime_blocks, batch.states,
                          batch.rhs, batch.flux_only);
  }
  void program_execution_source_default_into_(int runtime_block, MultiFab& state,
                                              MultiFab& rhs) const {
    sys_->block_source_into(runtime_block, state, rhs);
  }
  void program_execution_apply_projection_(int runtime_block, MultiFab& state) const {
    sys_->block_project(runtime_block, state);
  }
  Real program_execution_hmin_() const { return sys_->cfl_min_dx(); }
  Real program_execution_max_wave_speed_(int runtime_block, const MultiFab& state) const {
    return sys_->block_max_speed(runtime_block, state);
  }
  ProgramMetricGeometry program_execution_metric_geometry_() const {
    if (sys_->program_is_polar()) {
      const PolarGeometry& geometry = sys_->program_polar_geometry();
      return {true, geometry.r_min, geometry.dr()};
    }
    return {false, Real(0), sys_->grid_context().geom.dx()};
  }
  void program_execution_apply_polar_tensor_(MultiFab& out, MultiFab& in, const MultiFab* a_xx,
                                             const MultiFab* a_yy, const MultiFab* a_xy,
                                             const MultiFab* a_yx) const {
    if (!sys_->program_is_polar())
      throw std::logic_error("Cartesian Program provider cannot execute a polar tensor stencil");
    if (a_xx == nullptr) {
      if (a_yy != nullptr || a_xy != nullptr || a_yx != nullptr)
        throw std::logic_error("isotropic polar Program Laplacian received a partial tensor");
      if (!polar_unit_rr_ || !field_layout_matches_(*polar_unit_rr_, in, 1, 1) || !polar_unit_tt_ ||
          !field_layout_matches_(*polar_unit_tt_, in, 1, 1)) {
        polar_unit_rr_ = std::make_shared<MultiFab>(in.box_array(), in.dmap(), 1, 1);
        polar_unit_tt_ = std::make_shared<MultiFab>(in.box_array(), in.dmap(), 1, 1);
        polar_unit_rr_->set_val(Real(1));
        polar_unit_tt_->set_val(Real(1));
      }
      apply_polar_tensor(in, sys_->program_polar_geometry(), out, polar_unit_rr_.get(),
                         polar_unit_tt_.get(), nullptr, nullptr);
      return;
    }
    apply_polar_tensor(in, sys_->program_polar_geometry(), out, a_xx, a_yy, a_xy, a_yx);
  }

  struct LogicalEvaluationRollback {
    double parent_dt = 0.0;
    amr::Rational stage{0, 1};
    amr::Rational phase_begin{0, 1};
    amr::Rational phase_span{1, 1};
    double physical_time_offset = 0.0;
  };

  GridContext program_execution_default_grid_context_() const { return sys_->grid_context(); }
  GridContext program_execution_block_grid_context_(int owner) const {
    return sys_->grid_context(sys_block(owner));
  }
  bool program_execution_owns_operator_authority_(OperatorFingerprint authority) const {
    return sys_ != nullptr && sys_->program_owns_operator_authority(authority);
  }
  OperatorFingerprint program_execution_operator_topology_(const MultiFab& prototype) const {
    if (!std::isfinite(current_dt_) || current_dt_ <= 0.0)
      throw std::logic_error("operator snapshot requested outside a prepared Program step");
    const GridContext context = sys_->grid_context();
    OperatorFingerprint topology =
        ::pops::detail::layout_fingerprint(prototype, program_resource_vector_distribution());
    if (sys_->program_is_polar())
      ::pops::detail::fingerprint_geometry(topology, sys_->program_polar_geometry());
    else
      ::pops::detail::fingerprint_geometry(topology, context.geom);
    ::pops::detail::fingerprint_boundary(topology, context.bc);
    if (context.boundary_plan) {
      ::pops::detail::fingerprint_mix(topology, context.boundary_plan->identity());
      ::pops::detail::fingerprint_mix(topology, context.boundary_plan->state_identity());
      ::pops::detail::fingerprint_mix(
          topology, static_cast<std::uint64_t>(context.boundary_plan->required_depth()));
    } else {
      ::pops::detail::fingerprint_mix(topology, "legacy-bcrec-boundary");
    }
    return topology;
  }
  OperatorEvaluationSnapshot program_execution_operator_evaluation_snapshot_(
      OperatorFingerprint authority, OperatorFingerprint topology, OperatorFingerprint resources,
      std::uint64_t revision) const {
    if (!std::isfinite(current_dt_) || current_dt_ <= 0.0)
      throw std::logic_error("operator snapshot requested outside a prepared Program step");
    const amr::Rational evaluation_stage = logical_phase_begin_ + stage_time_ * logical_phase_span_;
    const double evaluation_time = static_cast<double>(physical_time()) +
                                   logical_physical_time_offset_ +
                                   stage_time_.value() * current_dt_;
    return {authority,
            revision,
            static_cast<std::int64_t>(macro_step()),
            evaluation_stage.numerator,
            evaluation_stage.denominator,
            std::bit_cast<std::uint64_t>(current_dt_),
            std::bit_cast<std::uint64_t>(evaluation_time),
            UINT64_C(1),
            topology,
            resources};
  }
  MultiFab& program_execution_assembly_target_(MultiFab& field, std::string_view) const {
    return field;
  }
  MultiFab& program_execution_assembly_source_(MultiFab& field, std::string_view) const {
    return field;
  }
  MultiFab& program_execution_linear_solution_(MultiFab& field) const { return field; }
  MultiFab& program_execution_state_(int runtime_block) const {
    return sys_->block_state(runtime_block);
  }
  MultiFab program_execution_alloc_scalar_field_(int n_comp, int n_ghost) const {
    return sys_->alloc_scalar_field(n_comp, n_ghost);
  }
  std::size_t program_execution_apply_coupling_(
      Real dt, const std::vector<MultiFab*>& runtime_states) const {
    return sys_->apply_coupling_operators(dt, runtime_states);
  }
  void program_execution_register_history_storage_(const HistoryRegistration& registration) const {
    sys_->register_history(registration.name, registration.lag, registration.ncomp,
                           registration.qualified ? registration.runtime_owner : -1,
                           registration.state_identity, registration.space_identity,
                           registration.clock_identity, registration.interpolation_identity);
  }
  MultiFab& program_execution_read_history_storage_(const HistoryRegistration& registration,
                                                    int lag, HistoryReadMode /*mode*/) const {
    return sys_->read_history(registration.name, lag);
  }
  bool program_execution_history_initialized_storage_(
      const HistoryRegistration& registration) const {
    return sys_->history_initialized(registration.name);
  }
  void program_execution_set_history_initialized_storage_(const HistoryRegistration& registration,
                                                          bool initialized) const {
    sys_->set_history_initialized(registration.name, initialized);
  }
  HistoryStorePlan program_execution_history_store_plan_(
      const HistoryRegistration& /*registration*/) const {
    if (std::isfinite(current_dt_) && current_dt_ > 0.0)
      return {true, static_cast<Real>(current_dt_)};
    return {true, std::nullopt};
  }
  void program_execution_store_history_storage_(const HistoryRegistration& registration,
                                                const MultiFab& value,
                                                const std::optional<Real>& outgoing_dt) const {
    if (outgoing_dt)
      sys_->store_history(registration.name, value, static_cast<double>(*outgoing_dt));
    else
      sys_->store_history(registration.name, value);
  }
  bool program_execution_history_supports_selective_rotation_() const noexcept { return true; }
  HistoryRotationAction program_execution_history_rotation_action_() const noexcept {
    return HistoryRotationAction::Rotate;
  }
  void program_execution_defer_history_rotation_() const noexcept {}
  void program_execution_rotate_history_storage_(const std::string& clock_identity) const {
    if (clock_identity.empty())
      sys_->rotate_histories();
    else
      sys_->rotate_histories(clock_identity);
  }
  CacheManager& program_execution_cache_(SchedulerCacheOperation /*operation*/) const {
    return sys_->program_cache();
  }
  ProgramResourceTopology program_execution_resource_topology_() const noexcept {
    return {0, 0, 1, sys_->n_blocks()};
  }
  int program_execution_resource_level_() const noexcept { return 0; }
  void program_execution_select_resource_level_(int /*level*/) const noexcept {}
  ProgramResourceStorage program_execution_resource_storage_() const noexcept {
    return {PreparedVectorDistribution::Distributed, FieldDistribution::Distributed, 0};
  }
  std::vector<Real> program_execution_resource_cell_measures_() const {
    const Geometry geometry = sys_->grid_context().geom;
    return {geometry.dx() * geometry.dy()};
  }
  void program_execution_publish_axpy_(MultiFab&, Real, const MultiFab&) const noexcept {}
  void program_execution_publish_exact_axpy_(
      MultiFab&, Real, const MultiFab&, Real,
      std::initializer_list<ExactCoefficientTerm>) const noexcept {}
  void program_execution_publish_lincomb_(MultiFab&, Real, const MultiFab&, Real,
                                          const MultiFab&) const noexcept {}
  void program_execution_publish_exact_lincomb_(
      MultiFab&, Real, const MultiFab&, Real, const MultiFab&, Real,
      std::initializer_list<ExactCoefficientTerm>,
      std::initializer_list<ExactCoefficientTerm>) const noexcept {}

  double program_execution_logical_parent_dt_() const noexcept { return current_dt_; }
  LogicalEvaluationRollback program_execution_capture_logical_evaluation_() const noexcept {
    return {current_dt_, stage_time_, logical_phase_begin_, logical_phase_span_,
            logical_physical_time_offset_};
  }
  void program_execution_apply_logical_evaluation_(
      const LogicalEvaluationInterval& interval) const {
    const double child_offset =
        logical_physical_time_offset_ + static_cast<double>(interval.iteration) * interval.child_dt;
    if (!std::isfinite(child_offset))
      throw std::overflow_error("Program logical evaluation child window is not finite");
    const amr::Rational child_span = logical_phase_span_ * interval.child_span;
    const amr::Rational child_begin =
        logical_phase_begin_ + logical_phase_span_ * interval.child_begin;

    current_dt_ = interval.child_dt;
    stage_time_ = amr::Rational(0, 1);
    logical_phase_begin_ = child_begin;
    logical_phase_span_ = child_span;
    logical_physical_time_offset_ = child_offset;
  }
  void program_execution_restore_logical_evaluation_(
      const LogicalEvaluationRollback& rollback) const noexcept {
    current_dt_ = rollback.parent_dt;
    stage_time_ = rollback.stage;
    logical_phase_begin_ = rollback.phase_begin;
    logical_phase_span_ = rollback.phase_span;
    logical_physical_time_offset_ = rollback.physical_time_offset;
  }
  SolveReport program_execution_solve_fields_from_state_at_(
      const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
      int block, MultiFab& state) const {
    return consume_field_outcome_(solve_fields_from_state_at(point, provider_slot, block, state));
  }
  MultiFab& program_execution_scratch_(ScratchKind kind, std::int64_t value_id, int subslot,
                                       const MultiFab& prototype, int n_comp, int n_ghost) const {
    return program_scratch_for_(kind, value_id, subslot, prototype, n_comp, n_ghost);
  }
  void program_execution_validate_commit_aliases_(bool /*has_aliased_source*/) const noexcept {}
  ProgramRuntimeState& program_execution_runtime_state_() const {
    return sys_->program_runtime_state_();
  }
  ProgramClockCoordinate program_execution_clock_coordinate_() const {
    return {static_cast<Real>(sys_->time()), sys_->macro_step(), -1};
  }
  void program_execution_set_field_timepoint_(const std::string& field,
                                              const FieldLogicalTimePoint& point) const {
    sys_->set_field_logical_timepoint(field, point);
  }
  void program_execution_set_field_parameters_(const std::string& field,
                                               const std::vector<double>& parameters) const {
    sys_->set_field_boundary_parameters(field, parameters);
  }
  void program_execution_set_field_kernel_(const std::string& field,
                                           const CompiledFieldBoundaryKernel& kernel) const {
    sys_->set_field_boundary_kernel(field, kernel);
  }
  mutable double current_dt_ = 0.0;
  mutable amr::Rational logical_phase_begin_{0, 1};
  mutable amr::Rational logical_phase_span_{1, 1};
  mutable double logical_physical_time_offset_ = 0.0;
  mutable std::shared_ptr<MultiFab> polar_unit_rr_;
  mutable std::shared_ptr<MultiFab> polar_unit_tt_;
  mutable std::shared_ptr<FieldSolveWorkspaceRegistry> field_solve_workspace_registry_ =
      std::make_shared<FieldSolveWorkspaceRegistry>();
  mutable std::shared_ptr<ScratchRegistry> scratch_registry_ = std::make_shared<ScratchRegistry>();
  System* sys_;
};

template <>
struct ProgramExecutionProviderFor<System> {
  using type = ProgramContext;
};

}  // namespace program
}  // namespace runtime
}  // namespace pops
