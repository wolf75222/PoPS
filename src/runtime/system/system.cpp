// ADC-632: CORE System-facade TU. Impl, the anon-namespace helpers, the config/geometry/lifecycle
// guards, the includes and the ctor/dtor/abi_key/lifecycle live here or in the shared private
// header system_impl.hpp; the responsibility split (install / fields / io / profiling / program)
// moves the remaining method bodies into sibling TUs that all include system_impl.hpp. This file
// keeps: abi_key, the ctor/dtor/move, mark_bound / lifecycle_state, and the thin step forwards.
#include "system_impl.hpp"  // ADC-632: System::Impl + shared facade helpers (runtime-private)

// native_loader.hpp templates instantiate on Impl; included AFTER system_impl.hpp so the Impl
// definition is complete (the historical "templates instantiated lower down" ordering, per-TU).
#include <pops/runtime/builders/compiled/native_loader.hpp>  // .so loading (JIT/AOT/native) + ABI guard

namespace pops {

// MODULE ABI key (frozen at compile time of this TU). Defined here so the _pops module
// exports it (POPS_EXPORT): add_native_block compares it to the key baked into the loader .so.
POPS_EXPORT std::string abi_key() {
  return detail::abi_key_string();
}

// Convenience static method (Python binding + add_native_block): delegates to the module's free key.
std::string System::abi_key() {
  return pops::abi_key();
}

System::System(const SystemConfig& c) {
  validate_system_config(c);  // BEFORE any allocation/derivation (Impl builds geom/ba/dm/aux)
  p_ = std::make_unique<Impl>(c);
}
System::~System() = default;
System::System(System&&) noexcept = default;
System& System::operator=(System&&) noexcept = default;

// Time advance EXTRACTED into stepper_ (SystemStepper, Batch B). Pure delegation: the Cartesian/polar
// dispatch of the physical step h, the per-block CFL formula (substeps/stride), the
// hold-then-catch-up semantics of the macro-step counter, the condensed source stage and the couplings live
// now in the header (bit-identical). The public API stays unchanged.
void System::step(double dt) {
  pops::runtime::program::ProfileScope s(p_->program_.profiler_, "step");
  p_->program_.profiler_.count("steps");
  p_->execute_step_transaction([&] { p_->stepper_.step(dt); });
}
void System::advance(double dt, int nsteps) {
  for (int i = 0; i < nsteps; ++i)
    step(dt);
}
void System::begin_step_transaction() {
  if (p_->external_step_transaction_)
    throw std::runtime_error("System::begin_step_transaction: transaction already active");
  p_->external_step_transaction_ = std::make_unique<Impl::AcceptedSnapshot>(*p_);
  p_->external_step_transaction_committed_ = false;
}
void System::commit_step_transaction() {
  if (!p_->external_step_transaction_)
    throw std::runtime_error("System::commit_step_transaction: no active transaction");
  if (p_->external_step_transaction_committed_)
    throw std::runtime_error("System::commit_step_transaction: transaction already committed");
  p_->external_step_transaction_committed_ = true;
}
std::map<std::string, double> System::step_change_l2() const {
  if (!p_->external_step_transaction_ || !p_->external_step_transaction_committed_)
    throw std::runtime_error(
        "System::step_change_l2 requires a committed external step transaction");
  if (p_->polar_)
    throw std::runtime_error(
        "System::step_change_l2 does not yet define the polar cell measure");
  const auto& previous = p_->external_step_transaction_->states;
  if (previous.size() != p_->sp.size())
    throw std::runtime_error("System::step_change_l2 snapshot composition mismatch");
  RelativeCellMeasure measure;
  if (p_->eb_set_ && p_->geometry_mode_ != GeometryMode::None) {
    measure.active_cells = &p_->domain_mask_;
    if (p_->geometry_mode_ == GeometryMode::CutCell)
      measure.inverse_volume_fraction = &p_->eb_inverse_volume_fraction_;
  }
  const double cell_area =
      static_cast<double>(p_->geom.dx()) * static_cast<double>(p_->geom.dy());
  std::map<std::string, double> result;
  for (std::size_t block = 0; block < p_->sp.size(); ++block) {
    const double sum_sq = static_cast<double>(
        pops::difference_sum_sq_all(p_->sp[block].U, previous[block], measure));
    result.emplace(p_->sp[block].name, std::sqrt(cell_area * sum_sq));
  }
  return result;
}
void System::finalize_step_transaction() {
  if (!p_->external_step_transaction_ || !p_->external_step_transaction_committed_)
    throw std::runtime_error("System::finalize_step_transaction: no committed transaction");
  p_->external_step_transaction_.reset();
  p_->external_step_transaction_committed_ = false;
}
void System::rollback_step_transaction() {
  if (!p_->external_step_transaction_)
    throw std::runtime_error("System::rollback_step_transaction: no active transaction");
  p_->external_step_transaction_->restore(*p_);
  p_->external_step_transaction_.reset();
  p_->external_step_transaction_committed_ = false;
}
double System::step_cfl(double cfl, double speed_floor, double max_dt, double min_dt) {
  return p_->execute_step_transaction(
      [&] { return p_->stepper_.step_cfl(cfl, speed_floor, max_dt, min_dt); });
}
double System::step_adaptive(double cfl) {
  return p_->execute_step_transaction([&] { return p_->stepper_.step_adaptive(cfl); });
}

// System clock (IO v1, audit wave 2): macro_step is REQUIRED by the restart (the
// hold-then-catch-up stride cadence reads macro_step % stride; t alone is not enough).
int System::macro_step() const {
  return p_->macro_step_;
}

// RUNTIME FREEZE LIFECYCLE (ADC-592 / ADC-578). mark_bound() is the ONE transition into the frozen
// state; the Python bind flow calls it LAST (after every install call), so the install sequence itself
// never trips require_assembling. A second call throws (a composition binds exactly once).
// lifecycle_state() reports "assembling" (not bound), "bound" (bound, no macro-step advanced),
// "running" (bound AND macro_step_ > 0) -- the running edge is derived from the macro-step counter, so
// it needs no extra state (and SystemStepper never reads lifecycle_ -> no MockImpl impact). The new
// checkpointed / finalized phases (SystemLifecycle) are reachable only through explicit transitions
// with no current caller, so the observable strings above are preserved bit-for-bit.
void System::mark_bound() {
  if (p_->lifecycle_.frozen())
    p_->lifecycle_.to_bound();  // raises the canonical second-bind refusal before any collective
  // All resolved field plans have now been installed and no named backend has been
  // materialized yet. Agree the complete registry before rank-local validation so a
  // divergent rank cannot throw locally while its peers enter the collective.
  p_->fields_.require_field_plan_consensus();
  if (!p_->block_state_identities_.empty() && p_->block_state_identities_.size() != p_->sp.size())
    throw std::runtime_error(
        "System::mark_bound: block state routes do not exactly cover materialized blocks");
  for (const auto& block : p_->sp)
    if (!p_->block_state_identities_.empty() &&
        (block.state_identity.empty() ||
         p_->block_state_identities_.find(block.name) == p_->block_state_identities_.end()))
      throw std::runtime_error(
          "System::mark_bound: materialized block lacks its exact state route");
  for (const auto& [name, plan] : p_->boundary_plans_) {
    if (p_->eb_set_ && p_->geometry_mode_ != GeometryMode::None &&
        plan->has_component_boundaries())
      throw std::runtime_error(
          "System::mark_bound: embedded-boundary block '" + name +
          "' has a native boundary component without a geometry-aware provider");
    auto found = std::find_if(p_->sp.begin(), p_->sp.end(),
                              [&name](const Impl::Species& block) { return block.name == name; });
    if (found == p_->sp.end())
      throw std::runtime_error(
          "System::mark_bound: prepared boundary plan references unknown block '" + name + "'");
    if (plan->ncomp() != found->ncomp)
      throw std::runtime_error(
          "System::mark_bound: prepared boundary component count differs from block '" + name +
          "'");
    (void)plan->has_boundary_linearization();
    runtime::multiblock::BoundaryEvaluationPoint preparation_point;
    preparation_point.clock = plan->identity() + "::bound-runtime";
    preparation_point.level = 0;
    preparation_point.dt = 0.0;
    preparation_point.physical_time = p_->t;
    found->boundary_lane =
        std::make_shared<ExecutionLane>(ExecutionLane::world(plan->identity(), "::runtime"));
    found->boundary_session = std::make_shared<PreparedGridBoundarySession>(
        p_->grid_ctx(name), *found->boundary_lane, found->U, preparation_point);
  }
  p_->lifecycle_.to_bound();  // Assembling -> Bound; throws the same message on a second bind
}
std::string System::lifecycle_state() const {
  // "running" stays DERIVED from the macro-step counter (the stepper never touches lifecycle_), so
  // the observable three strings are unchanged; the new checkpointed / finalized states surface only
  // when explicitly transitioned (no current caller).
  return p_->lifecycle_.state(p_->macro_step_);
}
// SCHEDULER VALUE CACHE (ADC-458): the System-owned CacheManager every ProgramContext forwards to. The
// .so resolves this across the dlopen boundary (POPS_EXPORT), so the step closure's cache_store_aux /
// cache_should_update reach the SAME manager the checkpoint serializes.
POPS_EXPORT pops::runtime::program::CacheManager& System::program_cache() {
  return p_->program_.cache_;
}
int System::nx() const {
  return p_->cfg.n;
}
// SLOW axis of the field (rows of the (ny, nx) array). We read it from the INDEX domain (dom = nx() x ny()),
// SINGLE SOURCE of the extents for both geometries: Cartesian dom = n x n -> ny() == nx() == n (square,
// UNCHANGED); polar dom = nr x ntheta -> nx() == nr (fast, i), ny() == ntheta (slow, j). It is this
// dimension that sizes the numpy array on the bindings side: a polar field has nx()*ny() = nr*ntheta
// values, and with nr != ntheta the square reshape (nx, nx) overflows the buffer (teardown bug).
int System::ny() const {
  return p_->dom.ny();
}
double System::time() const {
  return p_->t;
}
int System::n_species() const {
  return p_->blocks_.size();
}
std::vector<std::string> System::block_names() const {
  // SINGLE block registry (store), populated by the native install paths.
  return p_->blocks_.names();
}

EffectiveOptionsReport System::effective_options_report() const {
  EffectiveOptionsReport report;
  report.runtime = "system";
  report.poisson.rhs = p_->fields_.p_rhs;
  report.poisson.solver = p_->fields_.p_solver;
  report.poisson.bc = p_->fields_.p_bc;
  report.poisson.wall = p_->fields_.p_wall;
  report.poisson.wall_radius = p_->fields_.p_wall_radius;
  report.poisson.epsilon = static_cast<double>(p_->fields_.p_eps_);
  p_->fields_.write_effective_poisson_options(report.poisson);
  report.poisson.has_epsilon_field = p_->fields_.has_scalar_diffusion_coefficient();
  report.poisson.has_anisotropic_epsilon = p_->fields_.has_anisotropic_diffusion_coefficient();
  report.poisson.has_reaction_field = p_->fields_.has_kappa_field_;
  // ADC-615: effective cut-cell / EB thresholds (default kEb* unless overridden by set_disc_domain).
  report.eb.enabled = p_->eb_set_ && p_->geometry_mode_ == GeometryMode::CutCell;
  report.eb.geometry_mode =
      (p_->geometry_mode_ == GeometryMode::CutCell)
          ? "cutcell"
          : (p_->geometry_mode_ == GeometryMode::Staircase ? "staircase" : "none");
  report.eb.kappa_min = static_cast<double>(p_->eb_thresholds_.kappa_min);
  report.eb.face_open_eps = static_cast<double>(p_->eb_thresholds_.face_open_eps);
  report.eb.cut_theta_min = static_cast<double>(p_->eb_thresholds_.cut_theta_min);

  for (const Impl::Species& s : p_->sp) {
    EffectiveBlockOptions row;
    if (const EffectiveBlockOptions* opt = p_->diagnostics_.block_options_ptr(s.name))
      row = *opt;
    row.name = s.name;
    row.ncomp = s.ncomp;
    row.n_ghost = s.U.n_grow();
    row.substeps = s.substeps;
    row.stride = s.stride;
    row.evolve = s.evolve;
    row.gamma = s.gamma;
    row.conservative_vars = s.cons_vars.names;
    row.primitive_vars = s.prim_vars.names;
    report.blocks.push_back(std::move(row));
  }
  return report;
}

}  // namespace pops
