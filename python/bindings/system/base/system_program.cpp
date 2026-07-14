// ADC-632: program-forward seam of the System facade -- the thin delegations to the compiled
// ProgramRuntimeState (install_program_step, cadence, substeps/stride, the block_* rhs/flux/source
// evaluators, program block map, block_project, program diagnostics and params, installed hash and
// poisson_solver). This TU is a subdivision of system.cpp isolating the Program forwards.
// Pure body move from system.cpp, no logic changed -> production trajectories bit-identical.
#include "system_impl.hpp"  // ADC-632: shared System::Impl + facade helpers (binding-private)

namespace pops {

// Compiled time-program seam (epic ADC-399 / ADC-401): a generated problem.so installs its macro-step
// body and reaches per-block storage through these accessors (Impl is private to this TU).
void System::install_program_step(std::function<void(double)> step) {
  p_->program_.step_ = std::move(step);
}
// Compiled-Program macro-step cadence (ADC-411): SYSTEM-level substeps + stride around the installed
// program closure (cf. SystemStepper::step). Kept separate from install_program so the .so ABI is
// untouched. Validates substeps >= 1 && stride >= 1 (fail-loud: a non-positive cadence is meaningless).
void System::set_program_cadence(int substeps, int stride) {
  require_assembling(p_->lifecycle_, "set_program_cadence");  // frozen once pops.bind completes (ADC-592)
  // Program subsystem owns the cadence validation + storage (ADC-594): the guard message names
  // "System::set_program_cadence" verbatim (unchanged wording), keeping the pinned error intact.
  p_->program_.set_cadence(substeps, stride, "System");
}
// Read the installed GLOBAL cadence (ADC-594): the tiny const getters the ProgramRuntimeReport reads
// through the bindings (there was no Python-visible getter before). Default 1/1 with no program.
int System::program_substeps() const {
  return p_->program_.substeps_;
}
int System::program_stride() const {
  return p_->program_.stride_;
}
int System::n_blocks() const {
  return static_cast<int>(p_->sp.size());
}
MultiFab& System::block_state(int b) {
  return p_->sp[static_cast<std::size_t>(b)].U;
}
void System::block_rhs_into(int b, MultiFab& U, MultiFab& R) {
  p_->sp[static_cast<std::size_t>(b)].rhs_into(U, R);
}
// FLUX-ONLY residual R <- -div F(U) (ADC-425): the block's SourceFreeModel<Model> rhs path (built in
// build_block), bit-identical to rhs_into minus the default source. Fails loud on a block that did not
// build it (the host .so prototype loader) instead of silently leaking the source.
void System::block_neg_div_flux_into(int b, MultiFab& U, MultiFab& R) {
  Impl::Species& s = p_->sp[static_cast<std::size_t>(b)];
  if (!s.rhs_flux_only)
    throw std::runtime_error(
        "System::block_neg_div_flux_into: block '" + s.name +
        "' has no flux-only residual closure (the host .so prototype loader does not build one); a "
        "flux-only RHS (P.rhs(flux=True, sources without 'default')) needs a native block "
        "(add_block / production-backend compiled block)");
  s.rhs_flux_only(U, R);
}
// SOURCE-ONLY residual R <- S(U, aux) (ADC-430): the block's SourceInto<Model> path (built in
// build_block), the exact mirror of block_neg_div_flux_into and bit-identical to the source half of
// rhs_into. Fails loud on a block that did not build it (the host .so prototype loader) instead of
// silently leaking the flux.
void System::block_source_into(int b, MultiFab& U, MultiFab& R) {
  Impl::Species& s = p_->sp[static_cast<std::size_t>(b)];
  if (!s.source_only)
    throw std::runtime_error(
        "System::block_source_into: block '" + s.name +
        "' has no source-only residual closure (the host .so prototype loader does not build one); "
        "a "
        "source-only RHS (P.rhs(flux=False, sources with 'default')) needs a native block "
        "(add_block / production-backend compiled block)");
  s.source_only(U, R);
}
// Max |wave speed| of block b on U: the SAME BlockState::max_speed closure step_cfl reads (set at
// add_block time -- HasStabilitySpeed / max_wave_speed of the model). REUSES it, does not recompute.
Real System::block_max_speed(int b, const MultiFab& U) const {
  return p_->sp[static_cast<std::size_t>(b)].max_speed(U);
}
// MIN physical cell size of the grid: Cartesian min(dx, dy) / polar min(dr, r_min*dtheta), the exact
// formula SystemStepper::cfl_grid_h uses for the native CFL (kept consistent so a Program dt bound and
// the native CFL share the same hmin).
Real System::cfl_min_dx() const {
  return p_->polar_ ? std::min(p_->pgeom_.dr(), p_->pgeom_.r_min * p_->pgeom_.dtheta())
                    : std::min(p_->geom.dx(), p_->geom.dy());
}
bool System::program_is_polar() const {
  return p_->polar_;
}
PolarGeometry System::program_polar_geometry() const {
  if (!p_->polar_)
    throw std::runtime_error(
        "System::program_polar_geometry: the installed Program is not bound to a polar mesh");
  return p_->pgeom_;
}
std::string System::installed_program_hash() const {
  return p_->program_.installed_hash_;
}
// Configured field (Poisson) solver token, owned by SystemFieldSolver (p_solver, default
// "geometric_mg"). Read by install_program (Spec criterion 24, solver requirement) and exposed for
// introspection. Returns the last set_poisson solver, never empty (the default stands).
std::string System::poisson_solver() const {
  return p_->fields_.p_solver;
}
// NAME-based block binding seam (Spec 3 criterion 23, ADC-457). install_program builds the map after
// matching the .so's block names; ProgramContext reads it to translate a Program block index to the
// name-matched System block index. POPS_EXPORT: resolved by the generated .so across the dlopen boundary.
void System::set_program_block_map(const std::vector<int>& prog_to_sys) {
  p_->program_.block_map_ = prog_to_sys;
}
const std::vector<int>& System::program_block_map() const {
  return p_->program_.block_map_;
}
// Block positivity projection (ADC-177) reached by a compiled Program (ProgramContext::apply_projection,
// spec op 21). REUSES the block's own projection closure and rejects an absent capability.
void System::block_project(int b, MultiFab& u) {
  std::function<void(MultiFab&)>& proj = p_->sp[static_cast<std::size_t>(b)].project;
  if (!proj)
    throw std::runtime_error("System::block_project: owning block declares no pointwise projection");
  proj(u);
}
// Compiled-Program scalar diagnostics (ADC-414, spec op 23): the installed program writes named scalars
// via P.record_scalar (ProgramContext::record_scalar); Python reads them after the step. Delegated to
// the extracted Program subsystem (ADC-594); the read keeps the "System::program_diagnostic" wording.
void System::record_program_diagnostic(const std::string& name, Real value) {
  p_->program_.record_diagnostic(name, value);
}
Real System::program_diagnostic(const std::string& name) const {
  return p_->program_.diagnostic(name, "System");
}
std::map<std::string, Real> System::program_diagnostics() const {
  return p_->program_.diagnostics();
}
// COMPILED-PROGRAM RUNTIME PARAMETERS (ADC-510, Spec 5 C5). Seed/overwrite/read the per-PROGRAM-block
// RuntimeParams the installed step closure reads through ProgramContext::program_params. Delegated to
// the extracted Program subsystem (ADC-594): the store lives in program_ so a value change reaches the
// captured ctx -- the Program parameter carrier is independent from immutable model-package params. The fail-loud
// messages keep the "System::set_program_params" wording (unchanged). install_program seeds the
// defaults; Python installs the resolved Program vector (validated against the .so metadata).
void System::seed_program_params(int prog_block, const std::vector<double>& defaults) {
  p_->program_.seed_params(prog_block, defaults);  // idempotent: re-seeding resets to the baseline
}
void System::set_program_params(int prog_block, const std::vector<double>& values) {
  p_->program_.set_params(prog_block, values, "System");
}
RuntimeParams System::program_params(int prog_block) const {
  return p_->program_.params(prog_block);
}

}  // namespace pops
