// ADC-632: io/history seam of the System facade -- the clock, the multistep history rings
// (store/rotate/restore/rebuild_history_slots, the ADC-631 replay reference body) and the program
// scheduler-cache save/restore accessors. This TU is a subdivision of system.cpp (persistence and
// checkpoint surface of the compiled program runtime state).
// Pure body move from system.cpp, no logic changed -> production trajectories bit-identical.
#include "system_impl.hpp"  // ADC-632: shared System::Impl + facade helpers (runtime-private)

namespace pops {

void System::set_clock(double t, int macro_step) {
  if (macro_step < 0)
    throw std::runtime_error("System::set_clock : macro_step >= 0 (restart)");
  p_->t = t;
  p_->macro_step_ = macro_step;
}

void System::store_history(const std::string& name, const MultiFab& value) {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::store_history: unknown history '" + name +
                             "' (register it first)");
  std::vector<MultiFab>& ring = it->second;
  // Copy the valid cells of value into the current slot [0] (identical layout: ring slots and the
  // block state share (ba, dm); lincomb(dst, 1, src, 0, src) is a valid-cell deep copy).
  pops::lincomb(ring[0], Real(1), value, Real(0), value);
  // PER-SLOT dt (ADC-626): tag slot 0 with the dt that produced it (the last dt the stepper handed to
  // program_.step_). slot_dt is co-sized with the ring and rotated alongside it, so a selective-
  // persistence restart re-steps the recomputed slots with the exact dt sequence. Grown lazily here so
  // a program that never uses a checkpoint policy still pays only a small scalar vector.
  std::vector<Real>& dts = p_->program_.hist_.slot_dt[name];
  if (dts.size() != ring.size())
    dts.assign(ring.size(), Real(0));
  dts[0] = p_->program_.last_dt_;
  if (!p_->program_.hist_.initialized[name]) {
    // COLD START (first store): broadcast into every deeper slot so a multistep step 0 reads the same
    // value at every lag (degenerating to a one-step method). Deterministic + machine-precision exact.
    // The dt broadcasts the same way so every cold-start slot carries the step-0 dt.
    for (std::size_t k = 1; k < ring.size(); ++k) {
      pops::lincomb(ring[k], Real(1), value, Real(0), value);
      dts[k] = p_->program_.last_dt_;
    }
    p_->program_.hist_.initialized[name] = true;
  }
}

void System::rotate_histories() {
  // Shift each ring one step at the end of a macro-step (O(1) std::swap chain, buffer recycled into
  // slot [0]); the grid-free ring bookkeeping lives in the extracted Program subsystem (ADC-594).
  p_->program_.hist_.rotate();
}

void System::rotate_histories(const std::string& clock_identity) {
  if (clock_identity.empty())
    throw std::runtime_error("System::rotate_histories: clock identity must be non-empty");
  p_->program_.hist_.rotate(clock_identity);
}

// Multistep history checkpoint/restart seam (ADC-406b): the System owns the rings, so the checkpoint
// facade (sim.checkpoint / sim.restart) gathers and restores them DIRECTLY -- reusing the SAME global
// gather (gather_global) / scatter (write_state) machinery as the block state, so the round-trip is
// MPI-safe and bit-identical under np>1. No .so checkpoint_extra ABI is needed for the buffers.
std::vector<std::string> System::history_names() const {
  // enumeration lives in the extracted Program subsystem (ADC-594)
  return p_->program_.hist_.names();
}
int System::history_depth(const std::string& name) const {
  auto it = p_->program_.hist_.depth.find(name);
  if (it == p_->program_.hist_.depth.end())
    throw std::runtime_error("System::history_depth: unknown history '" + name + "'");
  return it->second;
}
int System::history_ncomp(const std::string& name) const {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::history_ncomp: unknown history '" + name + "'");
  return it->second[0].ncomp();
}
std::vector<double> System::history_global(const std::string& name, int slot) const {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::history_global: unknown history '" + name + "'");
  const std::vector<MultiFab>& ring = it->second;
  if (slot < 0 || slot >= static_cast<int>(ring.size()))
    throw std::runtime_error("System::history_global: slot=" + std::to_string(slot) +
                             " out of range for history '" + name + "' (depth " +
                             std::to_string(ring.size()) + ")");
  device_fence();
  return gather_global(ring[static_cast<std::size_t>(slot)], ring[0].ncomp(), nx(), ny());
}
bool System::history_initialized(const std::string& name) const {
  auto it = p_->program_.hist_.initialized.find(name);
  if (it == p_->program_.hist_.initialized.end())
    throw std::runtime_error("System::history_initialized: unknown history '" + name + "'");
  return it->second;
}
void System::restore_history(const std::string& name, int slot, const std::vector<double>& values) {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end()) {
    // The program will re-register the ring on its first post-restart step, but we restore BEFORE that
    // step; register it now (depth = slot + 1, grown as deeper slots arrive) so the values land. Uses
    // the SAME co-distributed (ba, dm, block 0 ncomp) ring as register_history.
    register_history(name, slot >= 1 ? slot : 1);
    it = p_->program_.hist_.histories.find(name);
  }
  std::vector<MultiFab>& ring = it->second;
  if (slot < 0)
    throw std::runtime_error("System::restore_history: slot=" + std::to_string(slot) +
                             " must be >= 0 for history '" + name + "'");
  if (slot >= static_cast<int>(ring.size())) {
    // A deeper slot than currently registered: grow the ring (zero-filled tail) so it fits, matching
    // register_history's idempotent growth.
    const int ncomp = ring[0].ncomp();
    for (int k = static_cast<int>(ring.size()); k <= slot; ++k) {
      MultiFab s(p_->ba, p_->dm, ncomp, 1);
      s.set_val(Real(0));
      ring.push_back(std::move(s));
    }
    p_->program_.hist_.depth[name] = static_cast<int>(ring.size());
  }
  // Scatter the GLOBAL component-major buffer into the slot's fab: reuse the Impl multi-box
  // write_state (the SAME scatter set_state uses), the true inverse of the multi-box gather
  // (gather_global / state_global). It dispatches on the slot's local_size(): the mono-box / MPI
  // mono-box path (owner rank writes its box, others no-op) and, for theta_boxes > 1, the multi-box
  // scatter that places each local band at its global indices -- matching how history_global gathers.
  p_->write_state(ring[static_cast<std::size_t>(slot)], ring[0].ncomp(), values);
}
void System::set_history_initialized(const std::string& name, bool initialized) {
  auto it = p_->program_.hist_.initialized.find(name);
  if (it == p_->program_.hist_.initialized.end())
    throw std::runtime_error("System::set_history_initialized: unknown history '" + name +
                             "' (restore its slots first)");
  it->second = initialized;
}

// Selective history persistence + deterministic ring replay (ADC-626). A history-persistence policy
// (pops.time.Dense / Interval / Revolve) stores only a SUBSET of a ring's slots in a checkpoint; the
// per-slot dt is serialized alongside so the restart can replay the recomputed slots with the exact dt
// sequence (variable-dt histories round-trip bit-for-bit). rebuild_history_slots reconstructs the
// missing slots by re-stepping the installed Program from the nearest older stored slot.
double System::history_slot_dt(const std::string& name, int slot) const {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::history_slot_dt: unknown history '" + name + "'");
  if (slot < 0 || slot >= static_cast<int>(it->second.size()))
    throw std::runtime_error("System::history_slot_dt: slot=" + std::to_string(slot) +
                             " out of range for history '" + name + "' (depth " +
                             std::to_string(it->second.size()) + ")");
  auto dt_it = p_->program_.hist_.slot_dt.find(name);
  if (dt_it == p_->program_.hist_.slot_dt.end() ||
      slot >= static_cast<int>(dt_it->second.size()))
    return 0.0;  // a never-stepped ring: no dt recorded yet (the dense/zero-fill case)
  return static_cast<double>(dt_it->second[static_cast<std::size_t>(slot)]);
}

void System::restore_history_slot_dt(const std::string& name, int slot, double dt) {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::restore_history_slot_dt: unknown history '" + name +
                             "' (restore its slots first)");
  if (slot < 0)
    throw std::runtime_error("System::restore_history_slot_dt: slot=" + std::to_string(slot) +
                             " must be >= 0 for history '" + name + "'");
  std::vector<Real>& dts = p_->program_.hist_.slot_dt[name];
  if (slot >= static_cast<int>(dts.size()))
    dts.resize(static_cast<std::size_t>(slot) + 1, Real(0));
  dts[static_cast<std::size_t>(slot)] = static_cast<Real>(dt);
}

int System::rebuild_history_slots(const std::string& name, const std::vector<int>& stored_slots) {
  // Contract (ADC-626): the STORED slots of ring `name` are already restored (restore_history), the
  // per-slot dt is restored (restore_history_slot_dt), and the SAME Program the checkpoint recorded is
  // installed (the program-hash guard upstream ensures this). The ring stores the block-0 state (the
  // keep_history lowering emits store_history(name, U.n)), so a stored slot IS that block's state at
  // that lag. We reconstruct the missing slots by seeding block 0 from the nearest OLDER stored slot
  // and re-stepping the installed Program forward, capturing the intermediate block states.
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::rebuild_history_slots: unknown history '" + name + "'");
  if (!p_->program_.step_)
    throw std::runtime_error(
        "System::rebuild_history_slots: no compiled Program is installed; the ring cannot be replayed "
        "(install_program before restart, or checkpoint the ring with Dense())");
  std::vector<MultiFab>& ring = it->second;
  const int depth = static_cast<int>(ring.size());
  std::vector<int> anchors = stored_slots;
  std::sort(anchors.begin(), anchors.end());
  anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());
  if (anchors.empty() || anchors.back() != depth - 1)
    throw std::runtime_error(
        "System::rebuild_history_slots: the oldest slot " + std::to_string(depth - 1) +
        " of history '" + name + "' is not stored; the ring is unreconstructable (nothing older to "
        "replay it from). The persistence policy must store the oldest slot.");
  // A fully-stored ring (Dense): nothing to recompute.
  const std::size_t stored_count = anchors.size();
  if (static_cast<int>(stored_count) == depth)
    return 0;
  // SAVE bracket: deep-copy every block state, the scheduler cache, and the WHOLE history subsystem
  // (rings + slot_dt + initialized) so the replay's own store_history / rotate_histories side effects
  // are fully undone -- the live state U and cache_ are identity after replay, and only the missing
  // ring slots we place by index below survive.
  std::vector<MultiFab> saved_states;
  saved_states.reserve(p_->sp.size());
  for (auto& block : p_->sp)
    saved_states.push_back(block.U);  // deep copy
  const pops::runtime::program::CacheManager saved_cache = p_->program_.cache_;
  const pops::runtime::program::HistoryManager saved_hist = p_->program_.hist_;

  // The per-slot dt each store produced, captured from the SAVED snapshot into a stable local vector.
  // CRITICAL: the replay's own store_history / rotate_histories MUTATE p_->program_.hist_.slot_dt, so
  // reading the live map inside the loop would give a moving target -- dts[j] is the dt that produced
  // the state now in slot j on the ORIGINAL forward run, which is exactly what re-stepping needs.
  std::vector<Real> dts(static_cast<std::size_t>(depth), Real(0));
  auto saved_dt_it = saved_hist.slot_dt.find(name);
  if (saved_dt_it != saved_hist.slot_dt.end()) {
    const std::vector<Real>& sd = saved_dt_it->second;
    for (int k = 0; k < depth && k < static_cast<int>(sd.size()); ++k)
      dts[static_cast<std::size_t>(k)] = sd[static_cast<std::size_t>(k)];
  }

  // Reconstruct the block-0 state trajectory: for each gap between adjacent anchors (older anchor at a
  // LARGER index, newer at a SMALLER one; time increases as the index decreases), seed block 0 from the
  // older stored slot then step forward, recording the post-step block state into each intervening slot.
  // Placement is BY INDEX (no rotate) -> the ADC-538 rotation-invalidation edge is sidestepped.
  std::vector<MultiFab> reconstructed(static_cast<std::size_t>(depth));
  for (std::size_t a = 0; a + 1 < anchors.size(); ++a) {
    const int older = anchors[a + 1];  // larger index = further back in time
    const int newer = anchors[a];       // smaller index = closer to now
    // Seed block 0 with the older stored slot's state (the ring holds the block-0 state at that lag).
    pops::lincomb(p_->sp[0].U, Real(1), saved_hist.histories.at(name)[static_cast<std::size_t>(older)],
                  Real(0), p_->sp[0].U);
    // Step forward from `older` down to `newer`, capturing each intermediate slot. The dt for the store
    // that produced slot j is dts[j] (recorded on the forward run), so re-stepping with it reproduces a
    // variable-dt history exactly.
    for (int j = older - 1; j >= newer; --j) {
      p_->program_.last_dt_ = dts[static_cast<std::size_t>(j)];
      p_->program_.step_(static_cast<double>(dts[static_cast<std::size_t>(j)]));
      // Record the fresh block state for slot j. Slot `newer` is a stored anchor (its restored value is
      // reinstated below), so recording it here is harmless; the non-anchor slots are the real output.
      reconstructed[static_cast<std::size_t>(j)] = p_->sp[0].U;  // deep copy the fresh block state
    }
  }

  // RESTORE bracket: undo every replay side effect (block states, cache, whole history subsystem).
  for (std::size_t b = 0; b < p_->sp.size(); ++b)
    p_->sp[b].U = std::move(saved_states[b]);
  p_->program_.cache_ = saved_cache;
  p_->program_.hist_ = saved_hist;

  // Place ONLY the recomputed slots (the anchors keep their restored values). Re-fetch the ring after
  // restoring hist_ (the restore replaced the vector).
  std::vector<MultiFab>& out_ring = p_->program_.hist_.histories.at(name);
  std::vector<bool> is_stored(static_cast<std::size_t>(depth), false);
  for (int s : anchors)
    is_stored[static_cast<std::size_t>(s)] = true;
  int recomputed = 0;
  for (int j = 0; j < depth; ++j) {
    if (is_stored[static_cast<std::size_t>(j)])
      continue;
    pops::lincomb(out_ring[static_cast<std::size_t>(j)], Real(1),
                  reconstructed[static_cast<std::size_t>(j)], Real(0),
                  out_ring[static_cast<std::size_t>(j)]);
    ++recomputed;
  }
  return recomputed;
}

// Load a generated problem.so and install its compiled time Program. Mirrors add_native_block
// (native_loader.hpp): self-promote this module to the global scope so the .so resolves the System
// seam accessors (POPS_EXPORT) against it, load the generated package locally, fail-loud on ABI-key
// mismatch, then call pops_install_program(this), which wraps the System in a ProgramContext and
// installs the macro-step closure. The .so stays loaded for the process lifetime.
POPS_EXPORT void System::install_program(const std::string& so_path) {
  require_assembling(p_->lifecycle_, "install_program");  // frozen once pops.bind completes (ADC-592)
#if defined(_WIN32)
  // Windows: the generated .dll links against _pops.lib at compile time; no global promotion needed.
  pops::dynlib::handle h = pops::dynlib::open(so_path);
  if (!h) {
    throw std::runtime_error("System::install_program: LoadLibrary('" + so_path +
                             "'): " + pops::dynlib::last_error());
  }
#else
  {
    // Promote the already-loaded module (found via an exported symbol) to the global scope so the
    // .so's undefined System seam symbols (POPS_EXPORT) resolve against it. macOS: harmless (the .so
    // is built with -undefined dynamic_lookup).
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&pops::abi_key), &info) && info.dli_fname)
      dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  }
  // The host must be visible to the package, but the package itself must remain local: generated
  // Programs deliberately reuse fixed ABI and C++ template names across semantic identities.
  pops::dynlib::handle h = pops::dynlib::open(so_path);
  if (!h) {
    throw std::runtime_error(
        "System::install_program: dlopen('" + so_path + "'): " + pops::dynlib::last_error() +
        " (the pops::System seam accessors must be exported and the host module promoted "
        "globally; cf. POPS_EXPORT)");
  }
#endif
  auto key_fn = reinterpret_cast<const char* (*)()>(pops::dynlib::sym(h, "pops_program_abi_key"));
  if (!key_fn) {
    pops::dynlib::close(h);
    throw std::runtime_error("System::install_program: pops_program_abi_key missing from '" +
                             so_path +
                             "' (regenerate the problem module with the current pops headers)");
  }
  const std::string loader_key = key_fn();
  const std::string module_key = pops::abi_key();
  if (loader_key != module_key) {
    pops::dynlib::close(h);
    throw std::runtime_error(
        "System::install_program: compiled program ABI mismatch: expected '" + module_key +
        "', got '" + loader_key +
        "'. Recompile the problem module with the SAME compiler, C++ standard and "
        "pops headers as the _pops module.");
  }
  // Route registry guard: the manifest is mandatory and must match before any installer is called.
  {
    auto manifest_fn = reinterpret_cast<const char* (*)()>(
        pops::dynlib::sym(h, "pops_program_route_manifest"));
    if (!manifest_fn) {
      pops::dynlib::close(h);
      throw std::runtime_error(
          "System::install_program: pops_program_route_manifest missing; regenerate artifact");
    }
    try {
      const char* raw = manifest_fn();
      if (!raw || raw[0] == '\0')
        throw std::runtime_error(
            "System::install_program: pops_program_route_manifest returned empty data");
      pops::verify_route_manifest(std::string(raw), "install_program");
    } catch (...) {
      pops::dynlib::close(h);
      throw;
    }
  }
  std::vector<pops::runtime::program::ProgramOperatorAuthority> operator_authorities;
  try {
    operator_authorities =
        pops::runtime::program::read_program_operator_authorities(h);
  } catch (...) {
    pops::dynlib::close(h);
    throw;
  }
  auto install = reinterpret_cast<void (*)(void*)>(pops::dynlib::sym(h, "pops_install_program"));
  if (!install) {
    pops::dynlib::close(h);
    throw std::runtime_error("System::install_program: pops_install_program missing from '" +
                             so_path + "'");
  }
  // Mandatory install-time requirement validation. The complete owner-qualified metadata table is
  // authenticated before installation on every platform; no pre-metadata artifact can bypass it.
  try {
    const auto meta = pops::runtime::program::read_module_metadata(h);
    const std::vector<std::string> sys_block_names = block_names();
    const std::string configured_solver = poisson_solver();
    auto has_block = [&sys_block_names](const std::string& want) {
      for (const auto& nm : sys_block_names) {
        if (nm == want) {
          return true;
        }
      }
      return false;
    };
    for (const auto& op : meta.operators) {
      // (a) AUX FIELD requirements (ADC-446): the user-supplied application fields B_z / T_e. Only
      // these are hard requirements (provides_aux); the derived fields phi/grad cannot block.
      for (const auto& aux : pops::runtime::program::required_aux(op.requirements)) {
        if (!p_->fields_.provides_aux(aux)) {
          throw std::runtime_error(
              "System::install_program: operator '" + op.name + "' requires aux field '" + aux +
              "', but simulation did not provide it (B_z -> set_magnetic_field, T_e -> "
              "set_electron_temperature_from, before install_program)");
        }
      }
      // (b) BLOCK-INSTANCE requirements (ADC-466, Spec criterion 24): an operator that reads another
      // species (e.g. collisions) names the block instance it needs; reject if it was not added. The
      // verbatim spec message names the operator and the missing instance.
      for (const auto& blk : pops::runtime::program::required_blocks(op.requirements)) {
        if (!has_block(blk)) {
          throw std::runtime_error("operator '" + op.name + "' requires block instance '" + blk +
                                   "'");
        }
      }
      // (c) SOLVER requirement (ADC-466): a field operator that requires a named field solver is
      // rejected at install when the configured Poisson solver (set_poisson) does not match. The
      // verbatim spec message names the field operator and the required solver.
      const std::string need_solver = pops::runtime::program::required_solver(op.requirements);
      if (!need_solver.empty() && need_solver != configured_solver) {
        throw std::runtime_error("field operator '" + op.name + "' requires solver '" + need_solver +
                                 "'");
      }
    }
  } catch (...) {
    pops::dynlib::close(h);
    throw;
  }
  // NAME-based block binding (Spec 3 criterion 23, ADC-457). A compiled Program numbers its blocks in
  // P.state declaration order (the .so's pops_program_block_name table); the System numbers its blocks
  // in add order (block_names). They need NOT agree -- bind by NAME, not add-order. Read the .so's
  // block names, map each Program block index to the System block of that name, and store the
  // program-index -> system-index map (read by ProgramContext to resolve every ctx.state / rhs_into /
  // commit). A Program block whose name has no instantiated System block fails loud with the spec
  // message. The table is REQUIRED: a library without explicit block identities is ambiguous and
  // must be regenerated; the historical positional convention is no longer a binding contract.
  // Built BEFORE install() so the step closure (which captures a ProgramContext) sees the map on its
  // first run.
  {
    using count_t = int (*)();
    using name_t = const char* (*)(int);
    auto block_count = reinterpret_cast<count_t>(pops::dynlib::sym(h, "pops_program_block_count"));
    auto block_name = reinterpret_cast<name_t>(pops::dynlib::sym(h, "pops_program_block_name"));
    if (!block_count || !block_name) {
      pops::dynlib::close(h);
      throw std::runtime_error(
          "System::install_program: compiled Program '" + so_path +
          "' does not export the required block identity table "
          "(pops_program_block_count + pops_program_block_name). Positional Program-to-System "
          "binding has been removed; regenerate the Program library with the current PoPS "
          "codegen and headers.");
    }
    const std::vector<std::string> sys_names = block_names();
    const int n = block_count();
    std::vector<int> prog_to_sys(static_cast<std::size_t>(n), -1);
    for (int p = 0; p < n; ++p) {
      const std::string want = block_name(p);
      int found = -1;
      for (std::size_t s = 0; s < sys_names.size(); ++s)
        if (sys_names[s] == want) {
          found = static_cast<int>(s);
          break;
        }
      if (found < 0) {
        pops::dynlib::close(h);
        throw std::runtime_error("Program requires block instance '" + want +
                                 "', but simulation did not instantiate it");
      }
      prog_to_sys[static_cast<std::size_t>(p)] = found;
    }
    set_program_block_map(prog_to_sys);
  }
  // RUNTIME PARAMETERS (ADC-510, Spec 5 C5). A Program whose physics reads dsl.Param(..., kind="runtime")
  // exports a pops_program_param_* table: per flat parameter, its PROGRAM block index, its stable index
  // WITHIN that block (sorted-name order, matching the lowered params.get(index)) and its declaration
  // default. Group the defaults per block (in index order) and seed each block's RuntimeParams to those
  // defaults, so an install WITHOUT a runtime set behaves as with a const param. A later Python params=
  // route overwrites the supplied values via set_program_params. A Program with no runtime param (the
  // count symbol absent or 0) seeds nothing -> the param store stays empty (program_params returns
  // count 0, the lowered kernels read no param). Built BEFORE install() so the step closure (which
  // captures a ProgramContext) reads the seeded value on its first run.
  {
    using count_t = int (*)();
    using ival_t = int (*)(int);
    using dval_t = double (*)(int);
    auto pcount = reinterpret_cast<count_t>(pops::dynlib::sym(h, "pops_program_param_count"));
    auto pblock = reinterpret_cast<ival_t>(pops::dynlib::sym(h, "pops_program_param_block"));
    auto pindex = reinterpret_cast<ival_t>(pops::dynlib::sym(h, "pops_program_param_index"));
    auto pdef = reinterpret_cast<dval_t>(pops::dynlib::sym(h, "pops_program_param_default"));
    if (pcount && pblock && pindex && pdef) {
      const int np = pcount();
      std::map<int, std::vector<double>> defaults_by_block;  // program block -> defaults in index order
      for (int i = 0; i < np; ++i) {
        const int blk = pblock(i);
        const int idx = pindex(i);
        std::vector<double>& d = defaults_by_block[blk];
        if (static_cast<int>(d.size()) <= idx)
          d.resize(static_cast<std::size_t>(idx) + 1, 0.0);
        d[static_cast<std::size_t>(idx)] = pdef(i);
      }
      for (const auto& kv : defaults_by_block)
        seed_program_params(kv.first, kv.second);
    }
  }
  // Dynamic field-boundary launchers are installed from the same problem.so that owns their direct
  // function pointers.  Static-boundary artifacts export no entry and keep the historical fast path.
  // Install only after ABI/requirements/block/parameter preflight has completed.
  const auto previous_operator_authorities = p_->program_.operator_authorities_;
  p_->program_.operator_authorities_ = operator_authorities;
  try {
    if (auto install_boundaries = reinterpret_cast<void (*)(void*)>(
            pops::dynlib::sym(h, "pops_install_field_boundaries")))
      install_boundaries(static_cast<void*>(this));
    install(static_cast<void*>(this));
  } catch (...) {
    p_->program_.operator_authorities_ = previous_operator_authorities;
    throw;
  }
  // Record the program's IR hash (ADC-406b): the optional pops_program_hash export (a stable IR key,
  // cf. _PROGRAM_CPP_TEMPLATE) is serialized in the checkpoint so a restart against a DIFFERENT
  // compiled Program is rejected fail-loud. Missing symbol (older module) -> empty hash, no guard.
  auto hash_fn = reinterpret_cast<const char* (*)()>(pops::dynlib::sym(h, "pops_program_hash"));
  p_->program_.installed_hash_ = hash_fn ? std::string(hash_fn()) : std::string();
  // OPTIONAL dt bound (epic ADC-399 / ADC-417, spec s18). A Program may export a SECOND ABI pair --
  // pops_program_has_dt_bound() and pops_program_dt_bound(ProgramContext*, Real cfl) -- alongside
  // pops_install_program. When present AND has_dt_bound() is true, store a closure that builds a
  // ProgramContext over THIS System and runs the .so's lowered dt_bound expression for a given cfl;
  // step_cfl tightens dt to min(native CFL, program dt bound). A Program WITHOUT a dt bound (older
  // module / has_dt_bound() == false) clears the closure -> the native CFL is used UNCHANGED.
  using has_dt_t = bool (*)();
  using dt_bound_t = pops::Real (*)(pops::runtime::program::ProgramContext*, pops::Real);
  auto has_dt = reinterpret_cast<has_dt_t>(pops::dynlib::sym(h, "pops_program_has_dt_bound"));
  auto dt_bound = reinterpret_cast<dt_bound_t>(pops::dynlib::sym(h, "pops_program_dt_bound"));
  if (has_dt && dt_bound && has_dt()) {
    System* self = this;
    p_->program_.dt_bound_ = [self, dt_bound](Real cfl) -> Real {
      pops::runtime::program::ProgramContext ctx(self);
      return dt_bound(&ctx, cfl);
    };
  } else {
    p_->program_.dt_bound_ = nullptr;  // no program dt bound -> native CFL unchanged
  }
  // .so left loaded for the duration of the process (the installed closure points to code in it).
}
// Scheduler-cache checkpoint/restart seam (ADC-458, Spec 3 section 30): the System owns the cache, so
// the facade (sim.checkpoint / sim.restart) gathers and restores it DIRECTLY -- reusing the SAME global
// gather (gather_global, via copy_state) / scatter (write_state) machinery as the block state and the
// history rings, so the round-trip is MPI-safe and bit-identical under np>1. Mirrors the history seam.
std::vector<int> System::program_cache_nodes() const { return p_->program_.cache_.node_ids(); }
std::string System::program_cache_name(int node_id) const {
  return p_->program_.cache_.name_of(node_id);
}
int System::program_cache_last_update_step(int node_id) const {
  return p_->program_.cache_.last_update_step(node_id);
}
double System::program_cache_accumulated_dt(int node_id) const {
  return static_cast<double>(p_->program_.cache_.accumulated_dt_of(node_id));
}
int System::program_cache_ncomp(int node_id) const { return p_->program_.cache_.ncomp_of(node_id); }
int System::program_cache_ngrow(int node_id) const { return p_->program_.cache_.ngrow_of(node_id); }
std::vector<double> System::program_cache_global(int node_id) const {
  // Reuse the Impl multi-box gather (copy_state -> gather_global): the cache value is co-distributed
  // with block 0's storage (ba/dm), so this is the SAME component-major gather state_global / history_
  // global use (device_fence + all_reduce). All ranks call it; @throws if @p node_id is absent.
  const MultiFab& v = p_->program_.cache_.value_of(node_id);
  return p_->copy_state(v, v.ncomp());
}
void System::restore_program_cache(int node_id, int ncomp, int ngrow, int last_update_step,
                                   double accumulated_dt, const std::string& name,
                                   const std::vector<double>& values) {
  if (p_->sp.empty())
    throw std::runtime_error(
        "System::restore_program_cache: no block exists yet; the cache value is co-distributed with "
        "block 0's storage (replay the composition before restart)");
  // Allocate a value co-distributed with block 0 (ba/dm, @p ncomp comps, @p ngrow ghosts -- the SAME
  // ghost width the slot was cached with: 1 for the aux, the block-state width for a held scratch) and
  // scatter the GLOBAL buffer into it via the SAME write_state set_state uses (owner rank writes,
  // others no-op) -- the true inverse of program_cache_global. Then re-key the slot with its
  // bookkeeping. MPI-safe (all ranks call), bit-identical under np>1.
  MultiFab value(p_->ba, p_->dm, ncomp, ngrow);
  value.set_val(Real(0));
  p_->write_state(value, ncomp, values);
  p_->program_.cache_.restore_slot(node_id, std::move(value), last_update_step,
                                  static_cast<Real>(accumulated_dt), name);
}

}  // namespace pops
