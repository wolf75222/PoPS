// ADC-632: field/state seam of the System facade -- density/primitive-state setters, the elliptic
// field solve entry points (solve_fields / *_from_state), potential, get/set_state, variable
// names/roles, reduce_component, mass/density/potential and their global gathers, and the local-box
// accessors. This TU is a subdivision of system.cpp (state marshaling + field derivation surface).
// Pure body move from system.cpp, no logic changed -> production trajectories bit-identical.
#include "system_impl.hpp"  // ADC-632: shared System<kNativeDimension>::Impl + facade helpers (runtime-private)
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/numerics/elliptic/linear/pure_field_algebra.hpp>
#include <pops/parallel/solve_report_consensus.hpp>
#include <pops/runtime/analytic/collective_preflight.hpp>
#include <pops/runtime/output_piece_collective.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <tuple>

namespace pops {
namespace {

template <class Species>
void require_recoverable_system_candidate(const Species& state, const MultiFab& candidate,
                                          std::string_view operation) {
  const long missing_recovery = all_reduce_sum(state.cons_to_prim ? 0L : 1L);
  if (missing_recovery != 0)
    throw std::runtime_error(std::string(operation) +
                             ": target block has no prepared variable-recovery authority");

  // Candidate kernels may execute asynchronously. The type-erased prepared recovery is a host
  // closure over one cell, so make the latest device values visible before inspecting storage.
  candidate.sync_host();
  std::vector<double> conserved(static_cast<std::size_t>(state.ncomp));
  std::vector<double> primitive(static_cast<std::size_t>(state.ncomp));
  long local_failures = 0;
  for (int local = 0; local < candidate.local_size(); ++local) {
    const ConstArray4 values = candidate.fab(local).const_array();
    const Box2D valid = candidate.box(local);
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        for (int component = 0; component < state.ncomp; ++component)
          conserved[static_cast<std::size_t>(component)] = values(i, j, component);
        try {
          const RecoveryReport report = state.cons_to_prim(conserved.data(), primitive.data());
          const bool finite_candidate =
              std::all_of(conserved.begin(), conserved.end(),
                          [](double value) { return std::isfinite(value); }) &&
              std::all_of(primitive.begin(), primitive.end(),
                          [](double value) { return std::isfinite(value); });
          if (!report.publication_permitted() || !finite_candidate)
            ++local_failures;
        } catch (...) {
          // Do not let a rank-local provider exception strand peers before the collective verdict.
          ++local_failures;
        }
      }
  }

  const long failures = all_reduce_sum(local_failures);
  if (failures != 0)
    throw std::runtime_error(std::string(operation) +
                             ": prepared variable recovery rejected the candidate before "
                             "publication (failed cells=" +
                             std::to_string(failures) + ")");
}

template <class Species>
void publish_recovered_initial_candidate(Species& state, MultiFab& candidate,
                                         std::string_view operation) {
  require_recoverable_system_candidate(state, candidate, operation);

  PureFieldAlgebra::copy(state.U, candidate);
  // candidate is setup-local storage; publication must finish before it is destroyed.
  device_fence();
}

void require_exact_field_evaluation_request(
    const runtime::multiblock::BoundaryEvaluationPoint& point, std::string_view provider_slot,
    std::string_view request_kind) {
  const bool invalid =
      request_kind.empty() || provider_slot.empty() || point.clock.empty() || point.tick < 0 ||
      point.level != 0 || point.substep < 0 || point.stage < 0 || !std::isfinite(point.dt) ||
      point.dt <= 0.0 || !std::isfinite(point.physical_time) ||
      point.stage_fraction < amr::Rational(0, 1) || amr::Rational(1, 1) < point.stage_fraction;
  if (all_reduce_max(invalid ? 1L : 0L) != 0)
    throw std::invalid_argument(
        "System exact field evaluation requires one complete level-zero point and provider slot");

  ExactContractBuilder request;
  request.text("pops.system.exact-field-evaluation")
      .scalar(std::uint32_t{1})
      .text(request_kind)
      .text(provider_slot)
      .text(point.clock)
      .scalar(point.tick)
      .scalar(static_cast<std::int32_t>(point.level))
      .scalar(static_cast<std::int32_t>(point.substep))
      .scalar(static_cast<std::int32_t>(point.stage))
      .scalar(point.stage_fraction.numerator)
      .scalar(point.stage_fraction.denominator)
      .scalar(point.dt)
      .scalar(point.physical_time);
  const std::string exact_request = std::move(request).release();
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"system-exact-field-evaluation", exact_request}}))
    throw std::invalid_argument(
        "System exact field evaluation point differs between communicator ranks");
}

}  // namespace

void System<kNativeDimension>::validate_program_state_publication_candidate(int block,
                                                          const MultiFab& candidate) const {
  const long invalid_block =
      all_reduce_sum(block >= 0 && block < static_cast<int>(p_->sp.size()) ? 0L : 1L);
  if (invalid_block != 0)
    throw std::out_of_range(
        "System Program state publication block index differs across communicator ranks");
  const Impl::Species& state = p_->sp[static_cast<std::size_t>(block)];
  const bool exact_layout = candidate.box_array().boxes() == state.U.box_array().boxes() &&
                            candidate.dmap().ranks() == state.U.dmap().ranks() &&
                            candidate.ncomp() == state.U.ncomp() &&
                            candidate.n_grow() == state.U.n_grow();
  if (all_reduce_sum(exact_layout ? 0L : 1L) != 0)
    throw std::invalid_argument(
        "System Program state publication candidate differs from its block layout");
  require_recoverable_system_candidate(
      state, candidate,
      "System Program terminal state publication for block '" + state.name + "'");
}

void System<kNativeDimension>::set_density(const std::string& name, const std::vector<double>& rho) {
  Impl::Species& s = p_->find(name);
  const Real gm1 = Real(s.gamma) - Real(1);
  // Local helper: sets density + rest state on ONE cell (same formulas as the historical).
  auto set_cell = [&](Array4& u, int i, int j, Real r) {
    u(i, j, 0) = r;
    if (s.ncomp >= 3) {
      u(i, j, 1) = 0;
      u(i, j, 2) = 0;
    }  // momentum at rest
    if (s.ncomp == 4)
      u(i, j, 3) = r / gm1;  // E = p/(g-1), p = rho
  };
  // MULTI-BOX (theta_boxes > 1, polar): @p rho is the GLOBAL field (nr x ntheta, layout flat[j*gnx+i]
  // identical to the mono-box below). We write each local box at its GLOBAL indices. local_size() <= 1
  // (Cartesian / polar mono-box, including MPI mono-box): historical path UNCHANGED, bit-identical.
  if (s.U.local_size() > 1) {
    const int gnx = p_->dom.nx(), gny = p_->dom.ny();
    if (static_cast<int>(rho.size()) != gnx * gny)
      throw std::runtime_error("System<kNativeDimension>::set_density : size != nr*ntheta (multi-box theta)");
    for (int li = 0; li < s.U.local_size(); ++li) {
      Array4 u = s.U.fab(li).array();
      const Box2D b = s.U.box(li);
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          set_cell(u, i, j, rho[static_cast<std::size_t>(j) * gnx + i]);
    }
    return;
  }
  // Row-major layout of the input array: (ni x nj) = extents of the state box. In Cartesian
  // ni = nj = cfg.n (indexing and size bit-identical to before). In polar ni = nr, nj = ntheta:
  // we index by the real extents of the box (and not n*n), so nr != ntheta is correctly handled.
  const Box2D v = s.U.box(0);
  const int ni = v.nx(), nj = v.ny();
  if (static_cast<int>(rho.size()) != ni * nj)
    throw std::runtime_error("System<kNativeDimension>::set_density : size != nr*ntheta (or n*n in Cartesian)");
  Array4 u = s.U.fab(0).array();
  // LAYOUT CONVENTION (unchanged vs the historical): slow axis = 2nd box index (j), fast axis =
  // 1st (i), i.e. flat[(j-lo) * ni + (i-lo)]. In Cartesian ni = n, lo = 0 -> flat[j*n+i] (bit-identical
  // to before). In polar the array is thus (nr, ntheta) radial-line-by-line: j = theta (slow
  // axis), i = r (fast axis), SAME order as density()/copy_comp0 -> consistent.
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      set_cell(u, i, j, rho[static_cast<std::size_t>(j - v.lo[1]) * ni + (i - v.lo[0])]);
}

POPS_EXPORT void System<kNativeDimension>::set_block_conversion(const std::string& name, CellConvert prim_to_cons,
                                              CellRecovery cons_to_prim) {
  Impl::Species& s = p_->find(name);
  const auto boundary = p_->boundary_plans_.find(name);
  if (boundary != p_->boundary_plans_.end()) {
    if (!cons_to_prim)
      throw std::runtime_error(
          "System prepared boundary traces require the block-model variable-recovery authority");
    if (boundary->second->requires_fixed_state_conversion()) {
      if (!prim_to_cons)
        throw std::runtime_error(
            "System primitive fixed-state boundary requires the block-model conversion");
      const int ncomp = s.ncomp;
      boundary->second->prepare_fixed_state_conversion(
          [prim_to_cons, cons_to_prim, ncomp](const double* primitive, double* conservative) {
            prim_to_cons(primitive, conservative);
            std::vector<double> recovered(static_cast<std::size_t>(ncomp));
            const RecoveryReport report = cons_to_prim(conservative, recovered.data());
            if (!report.publication_permitted())
              throw std::runtime_error(
                  "primitive fixed-state boundary conversion failed prepared variable recovery");
          });
    }
    boundary->second->prepare_trace_recovery(cons_to_prim);
  }
  // A replacement pointwise authority must never inherit warm starts produced by the previous
  // model/provider. The matching batch authority is installed explicitly immediately afterwards
  // by every supported native and compiled builder. Until then primitive-field materialization
  // fails closed instead of reviving a second cell-by-cell recovery engine.
  s.batch_cons_to_prim = {};
  s.prim_to_cons = std::move(prim_to_cons);
  s.cons_to_prim = std::move(cons_to_prim);
}

POPS_EXPORT void System<kNativeDimension>::set_block_characteristic_no_inflow(const std::string& name,
                                                            CharacteristicNoInflowFill fill) {
  Impl::Species& block = p_->find(name);
  const auto boundary = p_->boundary_plans_.find(name);
  if (boundary == p_->boundary_plans_.end() ||
      !boundary->second->requires_characteristic_no_inflow())
    throw std::runtime_error(
        "System characteristic no-inflow was not requested by the exact block boundary plan");
  const SpatialProviderGeometry geometry = p_->geometry_mode_ == GeometryMode::None
                                               ? block.base_spatial_geometry
                                               : spatial_provider_geometry(p_->geometry_mode_);
  if (!block.spatial_provider.supports(
          {kNativeDimension, geometry, SpatialProviderOperation::CharacteristicNoInflow}))
    throw std::runtime_error(
        "System characteristic no-inflow has no qualified provider for the active spatial "
        "geometry");
  boundary->second->prepare_characteristic_no_inflow(std::move(fill));
}

POPS_EXPORT void System<kNativeDimension>::set_block_batch_recovery(const std::string& name,
                                                  CellBatchRecovery batch_cons_to_prim) {
  Impl::Species& state = p_->find(name);
  if (!state.cons_to_prim)
    throw std::runtime_error(
        "System batch variable recovery requires the pointwise prepared recovery authority");
  if (!batch_cons_to_prim)
    throw std::invalid_argument("System batch variable recovery callback must not be empty");
  state.batch_cons_to_prim = std::move(batch_cons_to_prim);
}

void System<kNativeDimension>::set_primitive_state(const std::string& name, const std::vector<double>& prim) {
  Impl::Species& s = p_->find(name);
  const int nc = s.ncomp;
  // Number of cells = REAL EXTENTS of the index domain (n*n Cartesian, nr*ntheta polar), NOT
  // cfg.n*cfg.n: in polar cfg.n = nr, so cfg.n^2 != nr*ntheta -> heap overflow (ntheta<nr) or
  // partial/wrong content (ntheta>nr). Cartesian bit-identical (dom.nx()==dom.ny()==n).
  const std::size_t nn =
      static_cast<std::size_t>(p_->dom.nx()) * static_cast<std::size_t>(p_->dom.ny());
  if (prim.size() != static_cast<std::size_t>(nc) * nn)
    throw std::runtime_error(
        "System<kNativeDimension>::set_primitive_state : size != ncomp*nr*ntheta (n*n Cartesian) (block '" + name +
        "' has " + std::to_string(nc) + " variables)");
  if (!s.prim_to_cons)
    throw std::runtime_error(
        "System<kNativeDimension>::set_primitive_state : the model of block '" + name +
        "' does not expose a primitive -> conservative conversion (.so generated before "
        "this project ?) ; use set_state (direct conservative state)");
  if (!s.cons_to_prim)
    throw std::runtime_error(
        "System<kNativeDimension>::set_primitive_state : the model of block '" + name +
        "' has no prepared variable-recovery authority for validating conservative publication");
  // CELL-BY-CELL conversion via the block model: we read the nc primitives component-major
  // (prim[c*nn + k]) into a small contiguous buffer, convert, and write the conservatives at the
  // same place in an output buffer. Then write_state pushes everything to the MultiFab (set_state
  // path, identical marshaling). Reuses therefore the existing marshaling (copy/write_state).
  std::vector<double> cons(prim.size());
  std::vector<double> cell_in(static_cast<std::size_t>(nc));
  std::vector<double> cell_out(static_cast<std::size_t>(nc));
  std::vector<double> recovered(static_cast<std::size_t>(nc));
  long local_failures = 0;
  for (std::size_t k = 0; k < nn; ++k) {
    for (int c = 0; c < nc; ++c)
      cell_in[c] = prim[static_cast<std::size_t>(c) * nn + k];
    std::fill(cell_out.begin(), cell_out.end(), std::numeric_limits<double>::quiet_NaN());
    bool accepted = false;
    try {
      s.prim_to_cons(cell_in.data(), cell_out.data());
      const bool finite = std::all_of(cell_out.begin(), cell_out.end(),
                                      [](double value) { return std::isfinite(value); });
      if (finite) {
        const RecoveryReport report = s.cons_to_prim(cell_out.data(), recovered.data());
        accepted = report.publication_permitted();
      }
    } catch (...) {
      accepted = false;
    }
    if (!accepted) {
      ++local_failures;
      continue;
    }
    for (int c = 0; c < nc; ++c)
      cons[static_cast<std::size_t>(c) * nn + k] = cell_out[c];
  }
  const long failures = all_reduce_sum(local_failures);
  if (failures != 0)
    throw std::runtime_error(
        "System<kNativeDimension>::set_primitive_state : prepared variable recovery rejected conservative "
        "publication (failed cells=" +
        std::to_string(failures) + ")");
  p_->write_state(s.U, nc, cons);
}

std::vector<double> System<kNativeDimension>::get_primitive_state(const std::string& name) {
  Impl::Species& s = p_->find(name);
  const int nc = s.ncomp;
  if (!s.cons_to_prim)
    throw std::runtime_error(
        "System<kNativeDimension>::get_primitive_state : the model of block '" + name +
        "' does not expose a conservative -> primitive conversion (.so generated before "
        "this project ?) ; use get_state (direct conservative state)");
  if (!s.batch_cons_to_prim)
    throw std::runtime_error(
        "System<kNativeDimension>::get_primitive_state : block '" + name +
        "' has no generation-qualified prepared batch recovery consumer");
  const std::vector<double> cons = p_->copy_state(s.U, nc);  // get_state path (same marshaling)
  std::vector<double> prim;
  const UniformRecoveryBatchReport batch = s.batch_cons_to_prim(cons, prim);
  if (!batch.publication_permitted()) {
    const RecoveryReport& recovery = batch.recovery;
    throw std::runtime_error(
        "System<kNativeDimension>::get_primitive_state : variable recovery failed for block '" + name +
        "' at local cell " + std::to_string(batch.failed_cell) + " (status=" +
        recovery_status_name(recovery.status) + ", cause=" + recovery_cause_name(recovery.cause) +
        ", failing_component=" + std::to_string(recovery.failing_component) +
        ", attempted_methods=" + std::to_string(recovery.attempted_methods) +
        ", last_method=" + recovery_method_kind_name(recovery.last_method_kind) +
        ", last_method_index=" + std::to_string(recovery.last_method) + ")");
  }
  return prim;
}

SolveReport System<kNativeDimension>::solve_fields_in_place_() {
  pops::runtime::program::ProfileScope s(p_->program_.profiler_, "field_solve");
  const SolveReport report = p_->solve_fields();
  // ELLIPTIC-SOLVER NATIVE COUNTERS (Spec 5 sec.13.11.1, ADC-479 criteria 42/43). The opaque
  // "field_solve" scope hides where the elliptic solve (96-99.9% of step cost) spends its time: read
  // the active solver's per-solve stats back HERE -- after p_->solve_fields() returns, so AFTER its
  // internal device_fence() (system_field_solver.hpp CRITICAL invariant: the V-cycle must be done
  // before phi is read), preserving the device-fence ordering. Cheap int/double reads, all guarded
  // by enabled() -> ZERO cost when profiling is off (count/record are no-ops too, but the accessor
  // reads are skipped entirely).
  if (p_->program_.profiler_.enabled()) {
    // mg_cycles / krylov_iters ACCUMULATE (total elliptic iteration work over the run); elliptic_bottom
    // records the coarsest-grid self-time as a timing sample. mg_levels is a STRUCTURAL CONSTANT (the
    // hierarchy depth), so count_max (peak) reports the actual level count instead of summing it per
    // step (same idiom as scratch_peak_bytes). All four are honest 0 for a direct FFT solver.
    p_->program_.profiler_.count("mg_cycles", p_->fields_.last_mg_cycles());
    p_->program_.profiler_.count("krylov_iters", p_->fields_.last_krylov_iters());
    p_->program_.profiler_.count_max("mg_levels", p_->fields_.last_num_levels());
    p_->program_.profiler_.record("elliptic_bottom", p_->fields_.last_bottom_seconds());
  }
  return report;
}

SolveReport System<kNativeDimension>::solve_fields_from_state_in_place_(int block_idx, const MultiFab& U_stage) {
  return p_->solve_fields_from_state(block_idx, U_stage);
}

SolveReport System<kNativeDimension>::solve_fields_from_state_at_in_place_(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
    int block_idx, const MultiFab& U_stage) {
  require_exact_field_evaluation_request(point, provider_slot, "single-stage");
  return p_->solve_named_field_from_state(provider_slot, block_idx, U_stage);
}

// Coupled multi-block field solve (Spec 3 criterion 24, ADC-457): forwards to the field solver, which
// assembles the system Poisson RHS as Sum_s elliptic_rhs_s(U_s) reading EVERY block's stage state at
// once (U_stages indexed by block index; nullptr -> the block's live state), then re-fills the shared
// aux. POPS_EXPORT: resolved by a generated problem.so (ProgramContext) across the dlopen boundary.
POPS_EXPORT SolveReport
System<kNativeDimension>::solve_fields_from_blocks_in_place_(const std::vector<const MultiFab*>& U_stages) {
  pops::runtime::program::ProfileScope s(p_->program_.profiler_, "field_solve");
  const SolveReport report = p_->solve_fields_from_blocks(U_stages);
  // Same elliptic-solver counters as System<kNativeDimension>::solve_fields (ADC-479 criteria 42/43), read back AFTER
  // the coupled solve returns -- i.e. after its internal device_fence() (system_field_solver.hpp). The
  // coupled multi-block solve uses the SAME ell_ solver, so the stats are populated identically.
  if (p_->program_.profiler_.enabled()) {
    p_->program_.profiler_.count("mg_cycles", p_->fields_.last_mg_cycles());
    p_->program_.profiler_.count("krylov_iters", p_->fields_.last_krylov_iters());
    p_->program_.profiler_.count_max("mg_levels", p_->fields_.last_num_levels());
    p_->program_.profiler_.record("elliptic_bottom", p_->fields_.last_bottom_seconds());
  }
  return report;
}

// NAMED multi-elliptic field (ADC-428): a SECOND elliptic solve for @p field from block @p block_idx's
// stage state. Forwards to the field solver, which assembles the per-field RHS (sum of the blocks'
// named bricks), solves with a dedicated native solver, and writes the field's OWN aux components.
POPS_EXPORT SolveReport System<kNativeDimension>::solve_fields_from_state_in_place_(const std::string& field,
                                                                  int block_idx,
                                                                  const MultiFab& U_stage) {
  return p_->solve_named_field_from_state(field, block_idx, U_stage);
}

POPS_EXPORT SolveReport System<kNativeDimension>::solve_fields_from_blocks_in_place_(
    const std::string& field, const std::vector<const MultiFab*>& U_stages) {
  return p_->solve_named_field_from_blocks(field, U_stages);
}

POPS_EXPORT SolveReport System<kNativeDimension>::solve_fields_from_blocks_at_in_place_(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& field,
    const std::vector<const MultiFab*>& U_stages) {
  require_exact_field_evaluation_request(point, field, "simultaneous-stages");
  return p_->solve_named_field_from_blocks(field, U_stages);
}

SolveOutcome System<kNativeDimension>::solve_fields() {
  prepare_default_field_publication_storage_();
  return run_field_publication_outcome_([this]() { return solve_fields_in_place_(); });
}

SolveOutcome System<kNativeDimension>::solve_fields_from_state(int block_idx, const MultiFab& U_stage) {
  prepare_default_field_publication_storage_();
  return run_field_publication_outcome_([this, block_idx, &U_stage]() {
    return solve_fields_from_state_in_place_(block_idx, U_stage);
  });
}

SolveOutcome System<kNativeDimension>::solve_fields_from_state_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
    int block_idx, const MultiFab& U_stage) {
  if (provider_slot.empty())
    throw std::invalid_argument(
        "System<kNativeDimension>::solve_fields_from_state_at requires an exact provider slot");
  prepare_named_field_publication_storage_(provider_slot);
  return run_field_publication_outcome_([this, &point, &provider_slot, block_idx, &U_stage]() {
    return solve_fields_from_state_at_in_place_(point, provider_slot, block_idx, U_stage);
  });
}

SolveOutcome System<kNativeDimension>::solve_fields_from_blocks(const std::vector<const MultiFab*>& U_stages) {
  prepare_default_field_publication_storage_();
  return run_field_publication_outcome_(
      [this, &U_stages]() { return solve_fields_from_blocks_in_place_(U_stages); });
}

SolveOutcome System<kNativeDimension>::solve_fields_from_state(const std::string& field, int block_idx,
                                             const MultiFab& U_stage) {
  prepare_named_field_publication_storage_(field);
  return run_field_publication_outcome_([this, &field, block_idx, &U_stage]() {
    return solve_fields_from_state_in_place_(field, block_idx, U_stage);
  });
}

SolveOutcome System<kNativeDimension>::solve_fields_from_blocks(const std::string& field,
                                              const std::vector<const MultiFab*>& U_stages) {
  prepare_named_field_publication_storage_(field);
  return run_field_publication_outcome_(
      [this, &field, &U_stages]() { return solve_fields_from_blocks_in_place_(field, U_stages); });
}

void System<kNativeDimension>::prepare_default_field_publication_storage_() {
  p_->fields_.prepare_default_publication_storage();
}

void System<kNativeDimension>::prepare_named_field_publication_storage_(const std::string& field) {
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"system-named-field-publication", field}}))
    throw std::invalid_argument("System named field publication request differs between MPI ranks");
  p_->fields_.prepare_named_publication_storage(field);
}

SolveOutcome System<kNativeDimension>::run_field_publication_outcome_(const std::function<SolveReport()>& solve) {
  begin_field_publication_outcome_();
  SolveReport report;
  std::exception_ptr local_error;
  bool solve_failed_local = false;
  try {
    report = solve();
  } catch (...) {
    local_error = std::current_exception();
    solve_failed_local = true;
  }
  if (all_reduce_max(solve_failed_local ? 1L : 0L) != 0) {
    rollback_field_publication_transaction();
    if (n_ranks() == 1 && local_error != nullptr)
      std::rethrow_exception(local_error);
    throw std::runtime_error("System field solver failed on at least one MPI rank");
  }
  return stage_field_publication_outcome_(std::move(report));
}

void System<kNativeDimension>::begin_field_publication_outcome_() {
  const bool active_local = p_->field_publication_active_;
  if (all_reduce_max(active_local ? 1L : 0L) != 0)
    throw std::logic_error(
        "System field solves are sequential until their prior SolveOutcome is consumed");

  bool begin_failed_local = false;
  try {
    begin_field_publication_transaction();
  } catch (...) {
    begin_failed_local = true;
  }
  if (all_reduce_max(begin_failed_local ? 1L : 0L) != 0) {
    rollback_field_publication_transaction();
    throw std::runtime_error("System field publication snapshot failed on at least one MPI rank");
  }
}

SolveOutcome System<kNativeDimension>::stage_field_publication_outcome_(SolveReport report) {
  const bool malformed = !solve_report_is_publishable(report, std::numeric_limits<int>::max());
  if (all_reduce_max(malformed ? 1L : 0L) != 0) {
    rollback_field_publication_transaction();
    throw std::runtime_error("System field solver published a malformed SolveReport");
  }
  ExactSolveReportConsensusScratch consensus;
  if (!consensus.agrees(report)) {
    rollback_field_publication_transaction();
    throw std::runtime_error("System field solver report differs between MPI ranks");
  }
  if (!report.solved_value_available()) {
    rollback_field_publication_transaction();
    return SolveOutcome::collective_world(std::move(report));
  }

  bool stage_failed_local = false;
  try {
    stage_field_publication_candidate();
  } catch (...) {
    stage_failed_local = true;
  }
  if (all_reduce_max(stage_failed_local ? 1L : 0L) != 0) {
    rollback_field_publication_transaction();
    throw std::runtime_error("System field candidate staging failed on at least one MPI rank");
  }
  return SolveOutcome::collective_world(
      std::move(report),
      SolveOutcome::PublicationHooks{
          this,
          [](void* context) noexcept {
            static_cast<System*>(context)->accept_field_publication_candidate();
          },
          nullptr,
          [](void* context) noexcept {
            try {
              static_cast<System*>(context)->rollback_field_publication_transaction();
            } catch (...) {
              std::terminate();
            }
          },
          {},
          [](void* context) {
            static_cast<System*>(context)->validate_field_publication_candidate();
          }});
}

void System<kNativeDimension>::begin_field_publication_transaction() {
  if (p_->field_publication_active_)
    throw std::logic_error(
        "System field solves are sequential until their publication outcome is consumed");
  if (p_->accepted_field_publication_)
    p_->accepted_field_publication_->capture(*p_);
  else
    p_->accepted_field_publication_ = std::make_unique<Impl::FieldPublicationSnapshot>(*p_);
  p_->field_publication_active_ = true;
  p_->field_publication_candidate_ready_ = false;
}

void System<kNativeDimension>::stage_field_publication_candidate() {
  if (!p_->field_publication_active_ || !p_->accepted_field_publication_ ||
      p_->field_publication_candidate_ready_)
    throw std::logic_error("System field publication has no unique active candidate slot");
  p_->fields_.stage_named_topology_reports();
  if (p_->candidate_field_publication_)
    p_->candidate_field_publication_->capture(*p_);
  else
    p_->candidate_field_publication_ = std::make_unique<Impl::FieldPublicationSnapshot>(*p_);
  p_->field_publication_candidate_ready_ = true;
  p_->accepted_field_publication_->restore(*p_);
}

void System<kNativeDimension>::validate_field_publication_candidate() {
  if (!p_->field_publication_active_ || !p_->accepted_field_publication_ ||
      !p_->candidate_field_publication_ || !p_->field_publication_candidate_ready_)
    throw std::logic_error("System field publication has no staged candidate");
  if (!p_->candidate_field_publication_->publication_layout_matches(*p_))
    throw std::logic_error("System field publication snapshot layout changed before Accept");
}

void System<kNativeDimension>::accept_field_publication_candidate() noexcept {
  if (!p_->field_publication_active_ || !p_->accepted_field_publication_ ||
      !p_->candidate_field_publication_ || !p_->field_publication_candidate_ready_)
    std::terminate();
  p_->candidate_field_publication_->restore_copy_only(*p_);
  p_->field_publication_candidate_ready_ = false;
  p_->field_publication_active_ = false;
}

void System<kNativeDimension>::rollback_field_publication_transaction() {
  if (!p_->field_publication_active_)
    return;
  if (!p_->accepted_field_publication_)
    std::terminate();
  p_->accepted_field_publication_->restore(*p_);
  p_->field_publication_candidate_ready_ = false;
  p_->field_publication_active_ = false;
}

bool System<kNativeDimension>::field_publication_transaction_active_() const noexcept {
  return p_->field_publication_active_;
}

// Register a named elliptic field (ADC-428): records WHERE the field's solved phi / centered grad land
// in the aux channel (@p phi_comp / @p gx_comp / @p gy_comp, the model's named aux slots). The native
// loader calls this for each m.elliptic_field after the block is installed. POPS_EXPORT: resolved by the
// generated problem.so / native loader across the dlopen boundary.
POPS_EXPORT void System<kNativeDimension>::register_elliptic_field(const std::string& block, const std::string& field,
                                                 int phi_comp, int gx_comp, int gy_comp,
                                                 int gradient_sign) {
  p_->register_elliptic_field(block, field, phi_comp, gx_comp, gy_comp, gradient_sign);
}

// Attach a named elliptic-field RHS closure to block @p block_name (ADC-428): the per-field Poisson
// right-hand side brick += elliptic_field_rhs(U). The native loader builds it (make_poisson_rhs of the
// named brick) and attaches it here; solve_fields_from_state(field, ...) then sums it over the blocks.
// @throws if the block is unknown. POPS_EXPORT: resolved across the dlopen boundary.
POPS_EXPORT void System<kNativeDimension>::set_block_elliptic_field(
    const std::string& block_name, const std::string& field,
    std::function<void(const MultiFab&, MultiFab&)> rhs) {
  p_->blocks_.find(block_name).named_poisson_rhs[field] = std::move(rhs);
}

// Potential phi restoration (IO v1, restart): writes the VALID cells of component 0 of the
// solver phi (multigrid warm start). Mono-box
// (same marshaling convention as potential / set_density).
void System<kNativeDimension>::set_potential(const std::vector<double>& phi) {
  Impl* P = p_.get();
  device_fence();
  if (P->polar_) {
    P->fields_.ensure_elliptic_polar();
    MultiFab& ph = P->fields_.pell_->phi();
    // Rank without a box (MPI mono-box): NO-OP (the owning rank restores phi). Allows restart on
    // all ranks with the GLOBAL field. Mono-rank: local_size()==1, UNCHANGED.
    if (ph.local_size() == 0)
      return;
    const Box2D v = ph.box(0);
    if (static_cast<int>(phi.size()) != v.nx() * v.ny())
      throw std::runtime_error("System<kNativeDimension>::set_potential : size != nr*ntheta");
    Array4 a = ph.fab(0).array();
    std::size_t k = 0;
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        a(i, j, 0) = phi[k++];
    return;
  }
  P->fields_.ensure_elliptic();
  MultiFab& ph = P->fields_.ell_phi();
  if (ph.local_size() == 0)
    return;  // rank without a box: no-op (cf. polar branch)
  const Box2D v = ph.box(0);
  if (static_cast<int>(phi.size()) != v.nx() * v.ny())
    throw std::runtime_error("System<kNativeDimension>::set_potential : size != n*n");
  Array4 a = ph.fab(0).array();
  std::size_t k = 0;
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      a(i, j, 0) = phi[k++];
}

std::vector<std::string> System<kNativeDimension>::field_provider_slots() const {
  return p_->fields_.provider_slots();
}

void System<kNativeDimension>::set_field_potential(const std::string& provider_slot, const std::vector<double>& phi) {
  MultiFab& field = p_->fields_.provider_potential(provider_slot);
  if (field.local_size() == 0)
    return;
  const Box2D valid = field.box(0);
  if (static_cast<int>(phi.size()) != valid.nx() * valid.ny())
    throw std::runtime_error("System<kNativeDimension>::set_field_potential size != nx*ny");
  Array4 values = field.fab(0).array();
  std::size_t index = 0;
  for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
    for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
      values(i, j, 0) = static_cast<Real>(phi[index++]);
}
std::vector<double> System<kNativeDimension>::eval_rhs(const std::string& name) {
  Impl::Species& s = p_->find(name);
  MultiFab R(p_->ba, p_->dm, s.ncomp, 0);
  block_rhs_into(p_->index(name), s.U, R);
  return p_->copy_state(R, s.ncomp);
}

// Collective scalar reduction over a NAMED block's state -- the native seam the Python diagnostics
// driver (ADC-542) drives to fire a declared typed measure (Norm / Integral / MinMax) each cadence
// tick. Resolves the block by name (Impl::find, insertion order) and folds its U with the pops::
// free functions. Per-component kinds read component @p comp; the full-state "_all" kinds fold over
// EVERY component. Unknown kind -> throw (fail loud, no silent 0). COLLECTIVE like dot.
double System<kNativeDimension>::reduce_component(const std::string& block, const std::string& kind, int comp) const {
  const Impl::Species& s = p_->find(block);
  const MultiFab& u = s.U;
  const int nc = s.ncomp;
  if (comp < 0 || comp >= nc)
    throw std::out_of_range("System<kNativeDimension>::reduce_component: component " + std::to_string(comp) +
                            " is outside block '" + block + "' with " + std::to_string(nc) +
                            " components");
  RelativeCellMeasure measure;
  if (p_->eb_set_ && p_->geometry_mode_ != GeometryMode::None) {
    measure.active_cells = &p_->domain_mask_;
    if (p_->geometry_mode_ == GeometryMode::CutCell)
      measure.inverse_volume_fraction = &p_->eb_inverse_volume_fraction_;
  }
  if (kind == "sum")
    return static_cast<double>(pops::reduce_sum(u, comp, measure));
  if (kind == "min")
    return static_cast<double>(pops::reduce_min(u, comp, measure));
  if (kind == "max")
    return static_cast<double>(pops::reduce_max(u, comp, measure));
  if (kind == "abs_sum")
    return static_cast<double>(pops::reduce_abs_sum(u, comp, measure));
  if (kind == "sum_sq")  // L2 squared: dot(u, u, comp); the driver takes sqrt
    return static_cast<double>(pops::dot(u, u, comp, measure));
  if (kind == "abs_max")  // LInf: collective max |u(.,.,comp)|
    return static_cast<double>(pops::reduce_norm_inf(u, comp, measure));
  // Full-state (unscoped) folds over ALL components -- host O(ncomp) composition of the native
  // per-component collectives (no field leaves the ranks; only ncomp scalars).
  if (kind == "sum_all") {
    double acc = 0.0;
    for (int c = 0; c < nc; ++c)
      acc += static_cast<double>(pops::reduce_sum(u, c, measure));
    return acc;
  }
  if (kind == "abs_sum_all") {
    double acc = 0.0;
    for (int c = 0; c < nc; ++c)
      acc += static_cast<double>(pops::reduce_abs_sum(u, c, measure));
    return acc;
  }
  if (kind == "sum_sq_all")
    return static_cast<double>(pops::dot_all(u, u, measure));
  if (kind == "abs_max_all") {
    double m = 0.0;
    for (int c = 0; c < nc; ++c)
      m = std::max(m, static_cast<double>(pops::reduce_norm_inf(u, c, measure)));
    return m;
  }
  throw std::runtime_error("System<kNativeDimension>::reduce_component: unknown reduction kind '" + kind +
                           "' for block '" + block +
                           "' (expected one of: sum, min, max, abs_sum, sum_sq, abs_max, "
                           "sum_all, abs_sum_all, sum_sq_all, abs_max_all)");
}
MultiFab System<kNativeDimension>::alloc_scalar_field(int n_comp, int n_ghost) {
  // Co-distributed with the block storage (Impl::ba / Impl::dm -- the same (ba, dm) every block U is
  // built with, P->ba/P->dm above), so a matrix-free apply pairs this field with the state/aux by
  // local fab index. Zero-initialized like a fresh block state (install_block sets U to 0).
  MultiFab f(p_->ba, p_->dm, n_comp, n_ghost);
  f.set_val(Real(0));
  return f;
}

// Multistep history seam (ADC-406a): a generated problem.so declares / reads / writes a named history
// field across macro-steps (Adams-Bashforth), reaching the SYSTEM-OWNED ring buffers through these
// accessors. The rings live in Impl::program_.hist_ (the extracted Program subsystem, ADC-594) so a
// later checkpoint slice (ADC-406b) can serialize them without touching the .so ABI.
MultiFab& System<kNativeDimension>::register_history(const std::string& name, int lag, int ncomp, int owner,
                                   const std::string& state_identity,
                                   const std::string& space_identity,
                                   const std::string& clock_identity,
                                   const std::string& interpolation_identity) {
  if (lag < 1)
    throw std::runtime_error("System<kNativeDimension>::register_history: lag must be >= 1 (got " +
                             std::to_string(lag) + ") for history '" + name + "'");
  if (p_->sp.empty())
    throw std::runtime_error(
        "System<kNativeDimension>::register_history: no block exists yet; a history is co-distributed with block 0's "
        "state (add the block before installing the program)");
  const bool qualified = owner >= 0 || !state_identity.empty() || !space_identity.empty() ||
                         !clock_identity.empty() || !interpolation_identity.empty();
  if (qualified &&
      (owner < 0 || owner >= static_cast<int>(p_->sp.size()) || state_identity.empty() ||
       space_identity.empty() || clock_identity.empty() || interpolation_identity.empty()))
    throw std::runtime_error(
        "System<kNativeDimension>::register_history: qualified registration requires owner/state/space/clock/"
        "interpolation identities for history '" +
        name + "'");
  const int want_depth = lag + 1;
  auto it = p_->program_.hist_.histories.find(name);
  if (it != p_->program_.hist_.histories.end()) {
    if (qualified) {
      auto& histories = p_->program_.hist_;
      const auto prior = histories.clock_identity.find(name);
      if (prior == histories.clock_identity.end()) {
        histories.owner[name] = owner;
        histories.state_identity[name] = state_identity;
        histories.space_identity[name] = space_identity;
        histories.clock_identity[name] = clock_identity;
        histories.interpolation_identity[name] = interpolation_identity;
      } else if (histories.owner.at(name) != owner ||
                 histories.state_identity.at(name) != state_identity ||
                 histories.space_identity.at(name) != space_identity ||
                 prior->second != clock_identity ||
                 histories.interpolation_identity.at(name) != interpolation_identity) {
        throw std::runtime_error("System<kNativeDimension>::register_history: history '" + name +
                                 "' cannot be re-registered with a different qualified identity");
      }
    }
    if (ncomp >= 1 && it->second[0].ncomp() != ncomp)
      throw std::runtime_error("System<kNativeDimension>::register_history: ncomp mismatch for history '" + name +
                               "'");
    // Idempotent re-registration: the ring depth is the MAX lag any caller requests. A read at the
    // declared max lag and the store (which only needs the current slot, register_history(name, 1))
    // can register in EITHER order without conflict -- a smaller request is a no-op (returns the
    // existing current slot), a larger one grows the ring (appending zero-filled deeper slots; the
    // current slot [0] and the already-stored slots are preserved). A program reads each name at one
    // fixed lag, so the depth converges in the first step and never changes again. The @p ncomp
    // request is ignored on re-registration: a name binds one component count at its first register.
    if (want_depth > p_->program_.hist_.depth[name]) {
      const int slot_ncomp = it->second[0].ncomp();
      for (int k = p_->program_.hist_.depth[name]; k < want_depth; ++k) {
        MultiFab slot(p_->ba, p_->dm, slot_ncomp, 1);
        slot.set_val(Real(0));
        it->second.push_back(std::move(slot));
      }
      p_->program_.hist_.depth[name] = want_depth;
    }
    return it->second[0];
  }
  // The ring holds @p ncomp components, co-distributed with the block storage (ba/dm) so a per-cell
  // kernel and the arithmetic pair it with the state by local fab index. One ghost layer like a block
  // state; zero-initialized (the cold-start fill happens on the first store, but a never-stored read
  // still fails loud on the !initialized flag below). @p ncomp < 0 (the default) resolves to block 0's
  // ncomp -- so a slot can carry a full RHS / state, byte-identical to the historical multistep ring
  // (ADC-406a); a caller that needs a narrower ring (ADC-427: the 1-component condensed-Schur phi^n
  // carry) passes an explicit ncomp >= 1.
  const int resolved_ncomp = ncomp < 0 ? p_->sp[qualified ? owner : 0].ncomp : ncomp;
  if (resolved_ncomp < 1)
    throw std::runtime_error("System<kNativeDimension>::register_history: ncomp must be >= 1 (got " +
                             std::to_string(ncomp) + ") for history '" + name + "'");
  std::vector<MultiFab> ring;
  ring.reserve(static_cast<std::size_t>(want_depth));
  for (int k = 0; k < want_depth; ++k) {
    MultiFab slot(p_->ba, p_->dm, resolved_ncomp, 1);
    slot.set_val(Real(0));
    ring.push_back(std::move(slot));
  }
  auto& stored = p_->program_.hist_.histories.emplace(name, std::move(ring)).first->second;
  p_->program_.hist_.depth[name] = want_depth;
  p_->program_.hist_.initialized[name] = false;
  p_->program_.hist_.fill_count[name] = 0;
  p_->program_.hist_.store_pending[name] = false;
  p_->program_.hist_.owner[name] = qualified ? owner : -1;
  if (qualified) {
    p_->program_.hist_.state_identity[name] = state_identity;
    p_->program_.hist_.space_identity[name] = space_identity;
    p_->program_.hist_.clock_identity[name] = clock_identity;
    p_->program_.hist_.interpolation_identity[name] = interpolation_identity;
  }
  return stored[0];
}

MultiFab& System<kNativeDimension>::read_history(const std::string& name, int lag) {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System<kNativeDimension>::read_history: unknown history '" + name +
                             "' (register it first)");
  if (lag < 0 || lag >= p_->program_.hist_.depth[name])
    throw std::runtime_error("System<kNativeDimension>::read_history: lag=" + std::to_string(lag) +
                             " out of range for history '" + name + "' (depth " +
                             std::to_string(p_->program_.hist_.depth[name]) + ")");
  if (!p_->program_.hist_.initialized[name])
    throw std::runtime_error("history '" + name + "' with lag=" + std::to_string(lag) +
                             " was requested but not initialized");
  return it->second[static_cast<std::size_t>(lag)];
}

std::vector<double> System<kNativeDimension>::get_state(const std::string& name) {
  Impl::Species& s = p_->find(name);
  return p_->copy_state(s.U, s.ncomp);
}
void System<kNativeDimension>::set_state(const std::string& name, const std::vector<double>& u) {
  Impl::Species& s = p_->find(name);
  p_->write_state(s.U, s.ncomp, u);
}
template <int Dim>
std::int64_t System<Dim>::set_analytic_expression_state(
    const std::string& name, const std::string& space, const std::string& centering,
    const std::string& projection, const std::vector<std::vector<std::string>>& opcodes,
    const std::vector<std::vector<double>>& literals) {
  auto prepared = analytic::collectively_prepare_analytic_request(
      "System<kNativeDimension>::set_analytic_expression_state",
      {{"centering", centering}, {"name", name}, {"projection", projection}, {"space", space}}, {},
      opcodes, literals, [&]() {
        require_assembling(p_->lifecycle_, "set_analytic_expression_state");
        if (space != "cell" || centering != "cell" || projection != "conservative_cell_average")
          throw std::runtime_error(
              "System<kNativeDimension>::set_analytic_expression_state requires cell-centred "
              "conservative_cell_average projection");
        typename Impl::Species& state = p_->find(name);
        std::vector<analytic::AnalyticProgram> programs =
            analytic::compile_component_programs(opcodes, literals);
        if (programs.size() != static_cast<std::size_t>(state.ncomp))
          throw std::runtime_error(
              "System<kNativeDimension>::set_analytic_expression_state component count differs from target state");
        return std::pair<typename Impl::Species*, std::vector<analytic::AnalyticProgram>>{
            &state, std::move(programs)};
      });
  MultiFab<Dim> candidate(prepared.first->U.layout(), prepared.first->U.distribution(),
                          prepared.first->U.local_rank(), prepared.first->U.ncomp(),
                          prepared.first->U.ghosts());
  const std::int64_t materialized =
      analytic::materialize_cell_average(candidate, p_->geom, prepared.second);
  publish_recovered_initial_candidate(*prepared.first, candidate,
                                      "System<kNativeDimension>::set_analytic_expression_state");
  return materialized;
}
template <int Dim>
std::int64_t System<Dim>::set_analytic_mapped_state(const std::string& name,
                                               const std::vector<std::vector<std::string>>& opcodes,
                                               const std::vector<std::vector<double>>& literals,
                                               const std::vector<std::string>& input_sources) {
  auto prepared = analytic::collectively_prepare_analytic_request(
      "System<kNativeDimension>::set_analytic_mapped_state", {{"name", name}}, {}, opcodes, literals, [&]() {
        require_assembling(p_->lifecycle_, "set_analytic_mapped_state");
        typename Impl::Species& state = p_->find(name);
        std::vector<analytic::AnalyticProgram> programs =
            analytic::compile_component_programs(opcodes, literals);
        if (programs.size() != static_cast<std::size_t>(state.ncomp))
          throw std::runtime_error(
              "System<kNativeDimension>::set_analytic_mapped_state component count differs from target state");
        if (input_sources.empty() || input_sources.size() > analytic::kAnalyticMaxStack)
          throw std::runtime_error(
              "System<kNativeDimension>::set_analytic_mapped_state requires one bounded input table");
        std::vector<analytic::detail::AnalyticInputBinding> bindings;
        bindings.reserve(input_sources.size());
        for (const auto& source : input_sources) {
          const auto sep = source.find(':');
          if (sep == std::string::npos)
            throw std::runtime_error(
                "System<kNativeDimension>::set_analytic_mapped_state input source must be 'state:N' or 'aux:N'");
          const std::string kind = source.substr(0, sep);
          int component = -1;
          try {
            component = std::stoi(source.substr(sep + 1));
          } catch (...) {
            throw std::runtime_error(
                "System<kNativeDimension>::set_analytic_mapped_state input component is not an integer");
          }
          if (component < 0)
            throw std::runtime_error(
                "System<kNativeDimension>::set_analytic_mapped_state input component must be non-negative");
          if (kind == "state")
            bindings.push_back({0, component});
          else if (kind == "aux")
            bindings.push_back({1, component});
          else
            throw std::runtime_error(
                "System<kNativeDimension>::set_analytic_mapped_state input source must be 'state' or 'aux'");
        }
        return std::tuple<typename Impl::Species*, std::vector<analytic::AnalyticProgram>,
                          std::vector<analytic::detail::AnalyticInputBinding>>{
            &state, std::move(programs), std::move(bindings)};
      });
  typename Impl::Species* state = std::get<0>(prepared);
  const auto& programs = std::get<1>(prepared);
  const auto& bindings = std::get<2>(prepared);
  MultiFab<Dim> seed(state->U);
  MultiFab<Dim> candidate(state->U.layout(), state->U.distribution(), state->U.local_rank(),
                          state->U.ncomp(), state->U.ghosts());
  const std::int64_t materialized = analytic::materialize_discrete_mapped_state(
      candidate, seed, p_->aux, p_->geom, programs, bindings);
  publish_recovered_initial_candidate(*state, candidate, "System<kNativeDimension>::set_analytic_mapped_state");
  return materialized;
}
template <int Dim>
std::int64_t System<Dim>::set_analytic_gaussian_state(const std::string& name,
                                                      const RealVector<Dim>& center,
                                                      double background, double amplitude,
                                                      double inverse_width) {
  require_assembling(p_->lifecycle_, "set_analytic_gaussian_state");
  typename Impl::Species& state = p_->find(name);
  MultiFab<Dim> candidate(state.U.layout(), state.U.distribution(), state.U.local_rank(),
                          state.U.ncomp(), state.U.ghosts());
  const std::int64_t materialized = analytic::materialize_gaussian_cell_average(
      candidate, p_->geom, center, static_cast<Real>(background), static_cast<Real>(amplitude),
      static_cast<Real>(inverse_width));
  publish_recovered_initial_candidate(state, candidate, "System<kNativeDimension>::set_analytic_gaussian_state");
  return materialized;
}
int System<kNativeDimension>::n_vars(const std::string& name) const {
  return p_->find(name).ncomp;
}
std::vector<std::string> System<kNativeDimension>::variable_names(const std::string& name,
                                                const std::string& kind) const {
  const Impl::Species& s = p_->find(name);
  if (kind == "conservative")
    return s.cons_vars.names;
  if (kind == "primitive")
    return s.prim_vars.names;
  throw std::runtime_error(
      "System<kNativeDimension>::variable_names : kind 'conservative' | 'primitive' (received '" + kind + "')");
}
std::vector<std::string> System<kNativeDimension>::variable_roles(const std::string& name,
                                                const std::string& kind) const {
  const Impl::Species& s = p_->find(name);
  const VariableSet* vs = nullptr;
  if (kind == "conservative")
    vs = &s.cons_vars;
  else if (kind == "primitive")
    vs = &s.prim_vars;
  else
    throw std::runtime_error(
        "System<kNativeDimension>::variable_roles : kind 'conservative' | 'primitive' (received '" + kind + "')");
  std::vector<std::string> out;
  out.reserve(static_cast<std::size_t>(vs->size));
  for (int i = 0; i < vs->size; ++i)
    out.push_back(role_name(vs->at(i).role));  // 'custom' if absent
  return out;
}
double System<kNativeDimension>::block_gamma(const std::string& name) const {
  return p_->find(name).gamma;
}

double System<kNativeDimension>::mass(const std::string& name) const {
  const Impl::Species& s = p_->find(name);
  if (!p_->polar_) {
    RelativeCellMeasure measure;
    if (p_->eb_set_ && p_->geometry_mode_ != GeometryMode::None) {
      measure.active_cells = &p_->domain_mask_;
      if (p_->geometry_mode_ == GeometryMode::CutCell)
        measure.inverse_volume_fraction = &p_->eb_inverse_volume_fraction_;
    }
    return static_cast<double>(pops::reduce_sum(s.U, 0, measure));
  }
  // POLAR: FV mass = Sum_ij n_ij r_i dr dtheta (annular cell volume r dr dtheta). This is the
  // quantity CONSERVED by assemble_rhs_polar (cf. test_polar_transport_mms). Host loop over the valid
  // cells (mono-rank: a single local fab), reduced over the ranks by symmetry (n_ranks==1).
  device_fence();
  const PolarGeometry& g = p_->pgeom_;
  const Real dr = g.dr(), dth = g.dtheta();
  double m = 0.0;
  for (int li = 0; li < s.U.local_size(); ++li) {
    const ConstArray4 u = s.U.fab(li).const_array();
    const Box2D v = s.U.box(li);
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        m += static_cast<double>(u(i, j, 0)) * static_cast<double>(g.r_cell(i) * dr * dth);
  }
  return all_reduce_sum(m);
}
std::vector<double> System<kNativeDimension>::density(const std::string& name) const {
  return p_->copy_comp0(p_->find(name).U);
}
std::vector<double> System<kNativeDimension>::potential() {
  device_fence();
  // POLAR: phi comes from the polar Poisson (pell_), not from the Cartesian solver (ell_). We build it
  // lazily if needed (a call before any step) and we read phi() of PolarPoissonSolver.
  if (p_->polar_) {
    p_->fields_.ensure_elliptic_polar();
    // Rank without a box (MPI mono-box): EMPTY return (no fab(0)). Cf. copy_comp0; the multi-rank
    // global field goes through System<kNativeDimension>::potential_global.
    if (p_->aux.local_size() == 0)
      return {};
    const ConstArray4 ph = p_->fields_.pell_->phi().fab(0).const_array();
    const Box2D v = p_->aux.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out.push_back(ph(i, j));
    return out;
  }
  p_->fields_.ensure_elliptic();
  if (p_->aux.local_size() == 0)
    return {};  // rank without a box: empty (cf. potential_global)
  const ConstArray4 ph = p_->fields_.ell_phi().fab(0).const_array();
  const Box2D v = p_->aux.box(0);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      out.push_back(ph(i, j));
  return out;
}

// --- GLOBAL accessors (collective MPI-safe), IO v1 multi-rank --------------------------------
// All three delegate to gather_global (anon namespace, top of file): a GLOBAL buffer filled by the
// LOCAL fabs at GLOBAL indices then all_reduce_sum_inplace, component-major. Mono-rank: the box
// covers the domain and the reduce is the identity -> array bit-identical to the non-global
// accessors (density / get_state / potential). The device_fence is owned here (before the gather).
std::vector<double> System<kNativeDimension>::density_global(const std::string& name) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  return gather_global(s.U, 1, nx(), ny());
}
std::vector<double> System<kNativeDimension>::state_global(const std::string& name) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  return gather_global(s.U, s.ncomp, nx(), ny());
}
std::vector<double> System<kNativeDimension>::potential_global() {
  device_fence();
  // Resolve phi, solving the Poisson (polar or Cartesian) if needed: COLLECTIVE, like the gather.
  const MultiFab* phi = nullptr;
  if (p_->polar_) {
    p_->fields_.ensure_elliptic_polar();
    phi = &p_->fields_.pell_->phi();
  } else {
    p_->fields_.ensure_elliptic();
    phi = &p_->fields_.ell_phi();
  }
  return gather_global(*phi, 1, nx(), ny());
}

std::vector<double> System<kNativeDimension>::field_potential_global(const std::string& provider_slot) {
  device_fence();
  MultiFab& field = p_->fields_.provider_potential(provider_slot);
  return gather_global(field, 1, nx(), ny());
}

std::vector<OutputPiece<2>> System<kNativeDimension>::output_state_local_pieces(const std::string& name,
                                                              int level) const {
  if (level != 0)
    throw std::out_of_range(
        "System<kNativeDimension>::output_state_local_pieces: uniform layout has only level zero");
  const Impl::Species& species = p_->find(name);
  return output_local_pieces(species.U, 0, false);
}

std::vector<OutputPiece<2>> System<kNativeDimension>::output_field_local_pieces(const std::string& provider_slot,
                                                              int level) {
  if (level != 0)
    throw std::out_of_range(
        "System<kNativeDimension>::output_field_local_pieces: uniform layout has only level zero");
  MultiFab& field = p_->fields_.provider_potential(provider_slot);
  return output_local_pieces(field, 0, false);
}

std::vector<OutputPiece<2>> System<kNativeDimension>::output_state_root_pieces(const ObserverMpiLane& lane,
                                                             const std::string& name,
                                                             int level) const {
  return output_pieces_to_root(lane,
                               detail::output_collective_identity("System", "state", name, level),
                               [&] { return output_state_local_pieces(name, level); });
}

std::vector<OutputPiece<2>> System<kNativeDimension>::output_field_root_pieces(const ObserverMpiLane& lane,
                                                             const std::string& provider_slot,
                                                             int level) {
  return output_pieces_to_root(
      lane, detail::output_collective_identity("System", "field", provider_slot, level),
      [&] { return output_field_local_pieces(provider_slot, level); });
}

// --- LOCAL per-fab accessors (NON collective): exact native ownership inspection ----------------
// Local counterpart of the _global accessors: they aggregate nothing (no MPI comm), they expose per
// rank the LOCAL boxes (in GLOBAL indices, as carried by the fab box) and the state of each fab. The
// typed scientific-output bridge consumes OutputPiece instead; these lower-level views remain useful
// for native ownership verification. A rank without a box returns an empty list.
std::vector<std::array<int, 4>> System<kNativeDimension>::local_boxes(const std::string& name) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  std::vector<std::array<int, 4>> out;
  out.reserve(s.U.local_size());
  for (int li = 0; li < s.U.local_size(); ++li) {
    const Box2D v = s.U.box(li);
    out.push_back({v.lo[0], v.lo[1], v.hi[0], v.hi[1]});  // (ilo, jlo, ihi, jhi) GLOBAL
  }
  return out;
}
std::vector<double> System<kNativeDimension>::local_state(const std::string& name, int li) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  if (li < 0 || li >= s.U.local_size())
    throw std::out_of_range("System<kNativeDimension>::local_state : local fab index out of bounds (0.." +
                            std::to_string(s.U.local_size() - 1) + ")");
  const int nc = s.ncomp;
  const ConstArray4 u = s.U.fab(li).const_array();
  const Box2D v = s.U.box(li);
  const int bnx = v.nx(), bny = v.ny();  // dimensions of the LOCAL box (valid cells)
  std::vector<double> out(static_cast<std::size_t>(nc) * bnx * bny, 0.0);
  // Layout = state_global mapped to the local box: (c*bny + jl)*bnx + il, component-major, so
  // reshapeable into (nc, bny, bnx) for a hyperslab dset[:, jlo:jhi+1, ilo:ihi+1].
  for (int c = 0; c < nc; ++c)
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out[(static_cast<std::size_t>(c) * bny + (j - v.lo[1])) * bnx + (i - v.lo[0])] =
            static_cast<double>(u(i, j, c));
  return out;
}

}  // namespace pops
