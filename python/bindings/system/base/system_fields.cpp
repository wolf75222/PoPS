// ADC-632: field/state seam of the System facade -- density/primitive-state setters, the elliptic
// field solve entry points (solve_fields / *_from_state), potential, get/set_state, variable
// names/roles, reduce_component, mass/density/potential and their global gathers, and the local-box
// accessors. This TU is a subdivision of system.cpp (state marshaling + field derivation surface).
// Pure body move from system.cpp, no logic changed -> production trajectories bit-identical.
#include "system_impl.hpp"  // ADC-632: shared System::Impl + facade helpers (binding-private)

namespace pops {

void System::set_density(const std::string& name, const std::vector<double>& rho) {
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
      throw std::runtime_error("System::set_density : size != nr*ntheta (multi-box theta)");
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
    throw std::runtime_error("System::set_density : size != nr*ntheta (or n*n in Cartesian)");
  Array4 u = s.U.fab(0).array();
  // LAYOUT CONVENTION (unchanged vs the historical): slow axis = 2nd box index (j), fast axis =
  // 1st (i), i.e. flat[(j-lo) * ni + (i-lo)]. In Cartesian ni = n, lo = 0 -> flat[j*n+i] (bit-identical
  // to before). In polar the array is thus (nr, ntheta) radial-line-by-line: j = theta (slow
  // axis), i = r (fast axis), SAME order as density()/copy_comp0 -> consistent.
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      set_cell(u, i, j, rho[static_cast<std::size_t>(j - v.lo[1]) * ni + (i - v.lo[0])]);
}

POPS_EXPORT void System::set_block_conversion(const std::string& name, CellConvert prim_to_cons,
                                             CellConvert cons_to_prim) {
  Impl::Species& s = p_->find(name);
  s.prim_to_cons = std::move(prim_to_cons);
  s.cons_to_prim = std::move(cons_to_prim);
}

void System::set_primitive_state(const std::string& name, const std::vector<double>& prim) {
  Impl::Species& s = p_->find(name);
  const int nc = s.ncomp;
  // Number of cells = REAL EXTENTS of the index domain (n*n Cartesian, nr*ntheta polar), NOT
  // cfg.n*cfg.n: in polar cfg.n = nr, so cfg.n^2 != nr*ntheta -> heap overflow (ntheta<nr) or
  // partial/wrong content (ntheta>nr). Cartesian bit-identical (dom.nx()==dom.ny()==n).
  const std::size_t nn =
      static_cast<std::size_t>(p_->dom.nx()) * static_cast<std::size_t>(p_->dom.ny());
  if (prim.size() != static_cast<std::size_t>(nc) * nn)
    throw std::runtime_error(
        "System::set_primitive_state : size != ncomp*nr*ntheta (n*n Cartesian) (block '" + name +
        "' has " + std::to_string(nc) + " variables)");
  if (!s.prim_to_cons)
    throw std::runtime_error(
        "System::set_primitive_state : the model of block '" + name +
        "' does not expose a primitive -> conservative conversion (.so generated before "
        "this project ?) ; use set_state (direct conservative state)");
  // CELL-BY-CELL conversion via the block model: we read the nc primitives component-major
  // (prim[c*nn + k]) into a small contiguous buffer, convert, and write the conservatives at the
  // same place in an output buffer. Then write_state pushes everything to the MultiFab (set_state
  // path, identical marshaling). Reuses therefore the existing marshaling (copy/write_state).
  std::vector<double> cons(prim.size());
  std::vector<double> cell_in(static_cast<std::size_t>(nc)), cell_out(static_cast<std::size_t>(nc));
  for (std::size_t k = 0; k < nn; ++k) {
    for (int c = 0; c < nc; ++c)
      cell_in[c] = prim[static_cast<std::size_t>(c) * nn + k];
    s.prim_to_cons(cell_in.data(), cell_out.data());
    for (int c = 0; c < nc; ++c)
      cons[static_cast<std::size_t>(c) * nn + k] = cell_out[c];
  }
  p_->write_state(s.U, nc, cons);
}

std::vector<double> System::get_primitive_state(const std::string& name) {
  Impl::Species& s = p_->find(name);
  const int nc = s.ncomp;
  // Number of cells = REAL EXTENTS of the index domain (n*n Cartesian, nr*ntheta polar), NOT
  // cfg.n*cfg.n: in polar cfg.n = nr, so cfg.n^2 != nr*ntheta -> heap overflow (ntheta<nr) or
  // partial/wrong content (ntheta>nr). Cartesian bit-identical (dom.nx()==dom.ny()==n).
  const std::size_t nn =
      static_cast<std::size_t>(p_->dom.nx()) * static_cast<std::size_t>(p_->dom.ny());
  if (!s.cons_to_prim)
    throw std::runtime_error(
        "System::get_primitive_state : the model of block '" + name +
        "' does not expose a conservative -> primitive conversion (.so generated before "
        "this project ?) ; use get_state (direct conservative state)");
  const std::vector<double> cons = p_->copy_state(s.U, nc);  // get_state path (same marshaling)
  std::vector<double> prim(cons.size());
  std::vector<double> cell_in(static_cast<std::size_t>(nc)), cell_out(static_cast<std::size_t>(nc));
  for (std::size_t k = 0; k < nn; ++k) {
    for (int c = 0; c < nc; ++c)
      cell_in[c] = cons[static_cast<std::size_t>(c) * nn + k];
    s.cons_to_prim(cell_in.data(), cell_out.data());
    for (int c = 0; c < nc; ++c)
      prim[static_cast<std::size_t>(c) * nn + k] = cell_out[c];
  }
  return prim;
}

SolveReport System::solve_fields() {
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

SolveReport System::solve_fields_from_state(int block_idx, const MultiFab& U_stage) {
  return p_->solve_fields_from_state(block_idx, U_stage);
}

// Coupled multi-block field solve (Spec 3 criterion 24, ADC-457): forwards to the field solver, which
// assembles the system Poisson RHS as Sum_s elliptic_rhs_s(U_s) reading EVERY block's stage state at
// once (U_stages indexed by block index; nullptr -> the block's live state), then re-fills the shared
// aux. POPS_EXPORT: resolved by a generated problem.so (ProgramContext) across the dlopen boundary.
POPS_EXPORT SolveReport System::solve_fields_from_blocks(
    const std::vector<const MultiFab*>& U_stages) {
  pops::runtime::program::ProfileScope s(p_->program_.profiler_, "field_solve");
  const SolveReport report = p_->solve_fields_from_blocks(U_stages);
  // Same elliptic-solver counters as System::solve_fields (ADC-479 criteria 42/43), read back AFTER
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
POPS_EXPORT SolveReport System::solve_fields_from_state(const std::string& field, int block_idx,
                                                       const MultiFab& U_stage) {
  return p_->solve_named_field_from_state(field, block_idx, U_stage);
}

// Register a named elliptic field (ADC-428): records WHERE the field's solved phi / centered grad land
// in the aux channel (@p phi_comp / @p gx_comp / @p gy_comp, the model's named aux slots). The native
// loader calls this for each m.elliptic_field after the block is installed. POPS_EXPORT: resolved by the
// generated problem.so / native loader across the dlopen boundary.
POPS_EXPORT void System::register_elliptic_field(const std::string& block,
                                                const std::string& field, int phi_comp,
                                                int gx_comp, int gy_comp) {
  p_->register_elliptic_field(block, field, phi_comp, gx_comp, gy_comp);
}

// Attach a named elliptic-field RHS closure to block @p block_name (ADC-428): the per-field Poisson
// right-hand side brick += elliptic_field_rhs(U). The native loader builds it (make_poisson_rhs of the
// named brick) and attaches it here; solve_fields_from_state(field, ...) then sums it over the blocks.
// @throws if the block is unknown. POPS_EXPORT: resolved across the dlopen boundary.
POPS_EXPORT void System::set_block_elliptic_field(
    const std::string& block_name, const std::string& field,
    std::function<void(const MultiFab&, MultiFab&)> rhs) {
  p_->blocks_.find(block_name).named_poisson_rhs[field] = std::move(rhs);
}

// Potential phi restoration (IO v1, restart): writes the VALID cells of component 0 of the
// solver phi (multigrid warm start). Mono-box
// (same marshaling convention as potential / set_density).
void System::set_potential(const std::vector<double>& phi) {
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
      throw std::runtime_error("System::set_potential : size != nr*ntheta");
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
    throw std::runtime_error("System::set_potential : size != n*n");
  Array4 a = ph.fab(0).array();
  std::size_t k = 0;
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      a(i, j, 0) = phi[k++];
}

std::vector<std::string> System::field_provider_slots() const {
  return p_->fields_.provider_slots();
}

void System::set_field_potential(const std::string& provider_slot,
                                 const std::vector<double>& phi) {
  MultiFab& field = p_->fields_.provider_potential(provider_slot);
  if (field.local_size() == 0)
    return;
  const Box2D valid = field.box(0);
  if (static_cast<int>(phi.size()) != valid.nx() * valid.ny())
    throw std::runtime_error("System::set_field_potential size != nx*ny");
  Array4 values = field.fab(0).array();
  std::size_t index = 0;
  for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
    for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
      values(i, j, 0) = static_cast<Real>(phi[index++]);
}
std::vector<double> System::eval_rhs(const std::string& name) {
  Impl::Species& s = p_->find(name);
  MultiFab R(p_->ba, p_->dm, s.ncomp, 0);
  s.rhs_into(s.U, R);
  return p_->copy_state(R, s.ncomp);
}

// Collective scalar reduction over a NAMED block's state -- the native seam the Python diagnostics
// driver (ADC-542) drives to fire a declared typed measure (Norm / Integral / MinMax) each cadence
// tick. Resolves the block by name (Impl::find, insertion order) and folds its U with the pops::
// free functions. Per-component kinds read component @p comp; the full-state "_all" kinds fold over
// EVERY component. Unknown kind -> throw (fail loud, no silent 0). COLLECTIVE like dot.
double System::reduce_component(const std::string& block, const std::string& kind, int comp) const {
  const Impl::Species& s = p_->find(block);
  const MultiFab& u = s.U;
  const int nc = s.ncomp;
  if (kind == "sum")
    return static_cast<double>(pops::reduce_sum(u, comp));
  if (kind == "min")
    return static_cast<double>(pops::reduce_min(u, comp));
  if (kind == "max")
    return static_cast<double>(pops::reduce_max(u, comp));
  if (kind == "abs_sum")
    return static_cast<double>(pops::reduce_abs_sum(u, comp));
  if (kind == "sum_sq")  // L2 squared: dot(u, u, comp); the driver takes sqrt
    return static_cast<double>(pops::dot(u, u, comp));
  if (kind == "abs_max")  // LInf: collective max |u(.,.,comp)|
    return all_reduce_max(static_cast<double>(pops::norm_inf(u, comp)));
  // Full-state (unscoped) folds over ALL components -- host O(ncomp) composition of the native
  // per-component collectives (no field leaves the ranks; only ncomp scalars).
  if (kind == "sum_all") {
    double acc = 0.0;
    for (int c = 0; c < nc; ++c)
      acc += static_cast<double>(pops::reduce_sum(u, c));
    return acc;
  }
  if (kind == "abs_sum_all") {
    double acc = 0.0;
    for (int c = 0; c < nc; ++c)
      acc += static_cast<double>(pops::reduce_abs_sum(u, c));
    return acc;
  }
  if (kind == "sum_sq_all")
    return static_cast<double>(pops::dot_all(u, u));
  if (kind == "abs_max_all") {
    double m = 0.0;
    for (int c = 0; c < nc; ++c)
      m = std::max(m, all_reduce_max(static_cast<double>(pops::norm_inf(u, c))));
    return m;
  }
  throw std::runtime_error(
      "System::reduce_component: unknown reduction kind '" + kind + "' for block '" + block +
      "' (expected one of: sum, min, max, abs_sum, sum_sq, abs_max, "
      "sum_all, abs_sum_all, sum_sq_all, abs_max_all)");
}
MultiFab System::alloc_scalar_field(int n_comp, int n_ghost) {
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
MultiFab& System::register_history(const std::string& name, int lag, int ncomp) {
  if (lag < 1)
    throw std::runtime_error("System::register_history: lag must be >= 1 (got " +
                             std::to_string(lag) + ") for history '" + name + "'");
  if (p_->sp.empty())
    throw std::runtime_error(
        "System::register_history: no block exists yet; a history is co-distributed with block 0's "
        "state (add the block before installing the program)");
  const int want_depth = lag + 1;
  auto it = p_->program_.hist_.histories.find(name);
  if (it != p_->program_.hist_.histories.end()) {
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
  const int resolved_ncomp = ncomp < 0 ? p_->sp[0].ncomp : ncomp;
  if (resolved_ncomp < 1)
    throw std::runtime_error("System::register_history: ncomp must be >= 1 (got " +
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
  return stored[0];
}

MultiFab& System::read_history(const std::string& name, int lag) {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::read_history: unknown history '" + name +
                             "' (register it first)");
  if (lag < 0 || lag >= p_->program_.hist_.depth[name])
    throw std::runtime_error("System::read_history: lag=" + std::to_string(lag) +
                             " out of range for history '" + name + "' (depth " +
                             std::to_string(p_->program_.hist_.depth[name]) + ")");
  if (!p_->program_.hist_.initialized[name])
    throw std::runtime_error("history '" + name + "' with lag=" + std::to_string(lag) +
                             " was requested but not initialized");
  return it->second[static_cast<std::size_t>(lag)];
}

std::vector<double> System::get_state(const std::string& name) {
  Impl::Species& s = p_->find(name);
  return p_->copy_state(s.U, s.ncomp);
}
void System::set_state(const std::string& name, const std::vector<double>& u) {
  Impl::Species& s = p_->find(name);
  p_->write_state(s.U, s.ncomp, u);
}
int System::n_vars(const std::string& name) const {
  return p_->find(name).ncomp;
}
std::vector<std::string> System::variable_names(const std::string& name,
                                                const std::string& kind) const {
  const Impl::Species& s = p_->find(name);
  if (kind == "conservative")
    return s.cons_vars.names;
  if (kind == "primitive")
    return s.prim_vars.names;
  throw std::runtime_error(
      "System::variable_names : kind 'conservative' | 'primitive' (received '" + kind + "')");
}
std::vector<std::string> System::variable_roles(const std::string& name,
                                                const std::string& kind) const {
  const Impl::Species& s = p_->find(name);
  const VariableSet* vs = nullptr;
  if (kind == "conservative")
    vs = &s.cons_vars;
  else if (kind == "primitive")
    vs = &s.prim_vars;
  else
    throw std::runtime_error(
        "System::variable_roles : kind 'conservative' | 'primitive' (received '" + kind + "')");
  std::vector<std::string> out;
  out.reserve(static_cast<std::size_t>(vs->size));
  for (int i = 0; i < vs->size; ++i)
    out.push_back(role_name(vs->at(i).role));  // 'custom' if absent
  return out;
}
double System::block_gamma(const std::string& name) const {
  return p_->find(name).gamma;
}

double System::mass(const std::string& name) const {
  const Impl::Species& s = p_->find(name);
  if (!p_->polar_)
    return sum(s.U, 0);  // Cartesian: bare sum of the cells (bit-identical)
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
std::vector<double> System::density(const std::string& name) const {
  return p_->copy_comp0(p_->find(name).U);
}
std::vector<double> System::potential() {
  device_fence();
  // POLAR: phi comes from the polar Poisson (pell_), not from the Cartesian solver (ell_). We build it
  // lazily if needed (a call before any step) and we read phi() of PolarPoissonSolver.
  if (p_->polar_) {
    p_->fields_.ensure_elliptic_polar();
    // Rank without a box (MPI mono-box): EMPTY return (no fab(0)). Cf. copy_comp0; the multi-rank
    // global field goes through System::potential_global.
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
std::vector<double> System::density_global(const std::string& name) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  return gather_global(s.U, 1, nx(), ny());
}
std::vector<double> System::state_global(const std::string& name) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  return gather_global(s.U, s.ncomp, nx(), ny());
}
std::vector<double> System::potential_global() {
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

std::vector<double> System::field_potential_global(const std::string& provider_slot) {
  device_fence();
  MultiFab& field = p_->fields_.provider_potential(provider_slot);
  return gather_global(field, 1, nx(), ny());
}

// --- LOCAL per-fab accessors (NON collective): parallel HDF5 write by hyperslabs (PR-IO-3) --
// Local counterpart of the _global accessors: they aggregate nothing (no MPI comm), they expose per rank
// the LOCAL boxes (in GLOBAL indices, as carried by the fab box) and the state of each fab.
// The facade sim.write(format='hdf5', parallel=True) creates the global datasets then each rank writes
// ITS boxes in hyperslabs. A rank without a box -> local_size()==0 -> empty list (never a hard-coded fab(0)).
std::vector<std::array<int, 4>> System::local_boxes(const std::string& name) const {
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
std::vector<double> System::local_state(const std::string& name, int li) const {
  device_fence();
  const Impl::Species& s = p_->find(name);
  if (li < 0 || li >= s.U.local_size())
    throw std::out_of_range("System::local_state : local fab index out of bounds (0.." +
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
