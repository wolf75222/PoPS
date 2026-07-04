// ADC-632: CORE System-facade TU. Impl, the anon-namespace helpers, the config/geometry/lifecycle
// guards, the includes and the ctor/dtor/abi_key/lifecycle live here or in the shared private
// header system_impl.hpp; the responsibility split (install / fields / io / profiling / program)
// moves the remaining method bodies into sibling TUs that all include system_impl.hpp. This file
// keeps: abi_key, the ctor/dtor/move, mark_bound / lifecycle_state, and the thin step forwards.
#include "system_impl.hpp"  // ADC-632: System::Impl + shared facade helpers (binding-private)

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
  p_->stepper_.step(dt);
}
void System::advance(double dt, int nsteps) {
  p_->stepper_.advance(dt, nsteps);
}
double System::step_cfl(double cfl) {
  return p_->stepper_.step_cfl(cfl);
}
double System::step_adaptive(double cfl) {
  return p_->stepper_.step_adaptive(cfl);
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
POPS_EXPORT pops::runtime::program::CacheManager& System::program_cache() { return p_->program_.cache_; }
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
  // SINGLE block registry (store), populated by all add paths: a block loaded via
  // add_dynamic_block / add_compiled_block (.so) appears there just like an add_block.
  return p_->blocks_.names();
}

EffectiveOptionsReport System::effective_options_report() const {
  EffectiveOptionsReport report;
  report.runtime = "system";
  report.time_scheme = p_->time_scheme_;
  report.gauss_policy = p_->gauss_policy_;
  report.poisson.rhs = p_->fields_.p_rhs;
  report.poisson.solver = p_->fields_.p_solver;
  report.poisson.bc = p_->fields_.p_bc;
  report.poisson.wall = p_->fields_.p_wall;
  report.poisson.wall_radius = p_->fields_.p_wall_radius;
  report.poisson.epsilon = static_cast<double>(p_->fields_.p_eps_);
  report.poisson.abs_tol = static_cast<double>(p_->fields_.p_abs_tol_);
  report.poisson.has_epsilon_field = p_->fields_.has_eps_field_;
  report.poisson.has_anisotropic_epsilon = p_->fields_.has_eps_xy_field_;
  report.poisson.has_reaction_field = p_->fields_.has_kappa_field_;

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

  for (const auto& kv : p_->diagnostics_.source_stage_options)
    report.source_stages.push_back(kv.second);
  return report;
}

}  // namespace pops
