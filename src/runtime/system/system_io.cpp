// ADC-632: io/history seam of the System facade -- the clock, the multistep history rings
// (store/rotate/restore/rebuild_history_slots, the ADC-631 replay reference body) and the program
// scheduler-cache save/restore accessors. This TU is a subdivision of system.cpp (persistence and
// checkpoint surface of the compiled program runtime state).
// Pure body move from system.cpp, no logic changed -> production trajectories bit-identical.
#include <cmath>

#include "system_impl.hpp"  // ADC-632: shared System::Impl + facade helpers (runtime-private)

#include <cmath>
#include <exception>
#include <optional>

namespace pops {

void System::set_clock(double t, int macro_step) {
  try {
    if (macro_step < 0)
      throw std::runtime_error("System::set_clock : macro_step >= 0 (restart)");
    if (!std::isfinite(t))
      throw std::runtime_error("System::set_clock : time must be finite");
    p_->program_.consume_cadence_clock_restore(t, macro_step, "System");
  } catch (...) {
    p_->program_.cancel_cadence_clock_restore();
    throw;
  }
  p_->t = t;
  p_->macro_step_ = macro_step;
}

void System::store_history(const std::string& name, const MultiFab& value) {
  store_history(name, value, static_cast<double>(p_->program_.last_dt_));
}

void System::store_history(const std::string& name, const MultiFab& value, double outgoing_dt) {
  if (!std::isfinite(outgoing_dt) || outgoing_dt < 0.0)
    throw std::runtime_error(
        "System::store_history: outgoing logical-clock dt must be finite and non-negative");
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::store_history: unknown history '" + name +
                             "' (register it first)");
  std::vector<MultiFab>& ring = it->second;
  // Copy the valid cells of value into the current slot [0] (identical layout: ring slots and the
  // block state share (ba, dm); lincomb(dst, 1, src, 0, src) is a valid-cell deep copy).
  pops::lincomb(ring[0], Real(1), value, Real(0), value);
  // PER-SLOT outgoing dt (ADC-626): keep_history stores U^n before the tail commit, so the current
  // step dt is the interval that advances this sample to the next accepted sample. slot_dt rotates
  // alongside the ring; after rotation slot k therefore carries the exact interval from slot k to
  // slot k-1. A selective restart reconstructing slot j from slot j+1 uses slot_dt[j+1]. Grown
  // lazily here so a program that never uses a checkpoint policy still pays only a small scalar
  // vector.
  std::vector<Real>& dts = p_->program_.hist_.slot_dt[name];
  if (dts.size() != ring.size())
    dts.assign(ring.size(), Real(0));
  dts[0] = static_cast<Real>(outgoing_dt);
  if (!p_->program_.hist_.initialized[name]) {
    // COLD START (first store): broadcast into every deeper slot so a multistep step 0 reads the same
    // value at every lag (degenerating to a one-step method). Deterministic + machine-precision exact.
    // The dt broadcasts the same way so every cold-start slot carries the step-0 dt.
    for (std::size_t k = 1; k < ring.size(); ++k) {
      pops::lincomb(ring[k], Real(1), value, Real(0), value);
      dts[k] = static_cast<Real>(outgoing_dt);
    }
    p_->program_.hist_.initialized[name] = true;
  }
  p_->program_.hist_.store_pending[name] = true;
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
int System::history_fill_count(const std::string& name) const {
  auto it = p_->program_.hist_.fill_count.find(name);
  if (it == p_->program_.hist_.fill_count.end())
    throw std::runtime_error("System::history_fill_count: unknown history '" + name + "'");
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
  p_->program_.hist_.fill_count[name] = initialized ? p_->program_.hist_.depth.at(name) : 0;
  p_->program_.hist_.store_pending[name] = false;
}
void System::restore_history_fill_count(const std::string& name, int fill_count) {
  auto depth = p_->program_.hist_.depth.find(name);
  if (depth == p_->program_.hist_.depth.end())
    throw std::runtime_error("System::restore_history_fill_count: unknown history '" + name +
                             "' (restore its slots first)");
  if (fill_count < 0 || fill_count > depth->second)
    throw std::runtime_error("System::restore_history_fill_count: fill count " +
                             std::to_string(fill_count) + " is outside [0, " +
                             std::to_string(depth->second) + "] for history '" + name + "'");
  p_->program_.hist_.fill_count[name] = fill_count;
  p_->program_.hist_.initialized[name] = fill_count > 0;
  p_->program_.hist_.store_pending[name] = false;
}

// Selective history persistence + deterministic ring replay (ADC-626). A history-persistence policy
// (pops.time.Dense / Interval / Revolve) stores only a SUBSET of a ring's slots in a checkpoint; the
// per-slot outgoing interval is serialized alongside so restart can replay the recomputed slots with
// the exact dt sequence (variable-dt histories round-trip bit-for-bit). rebuild_history_slots
// reconstructs the missing slots by re-stepping the installed Program from the nearest older slot.
double System::history_slot_dt(const std::string& name, int slot) const {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::history_slot_dt: unknown history '" + name + "'");
  if (slot < 0 || slot >= static_cast<int>(it->second.size()))
    throw std::runtime_error("System::history_slot_dt: slot=" + std::to_string(slot) +
                             " out of range for history '" + name + "' (depth " +
                             std::to_string(it->second.size()) + ")");
  auto dt_it = p_->program_.hist_.slot_dt.find(name);
  if (dt_it == p_->program_.hist_.slot_dt.end() || slot >= static_cast<int>(dt_it->second.size()))
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
  const Real native_dt = static_cast<Real>(dt);
  if (!std::isfinite(dt) || dt < 0.0 || !std::isfinite(static_cast<double>(native_dt)))
    throw std::runtime_error(
        "System::restore_history_slot_dt: dt must be finite and >= 0 for "
        "history '" +
        name + "'");
  std::vector<Real>& dts = p_->program_.hist_.slot_dt[name];
  if (slot >= static_cast<int>(dts.size()))
    dts.resize(static_cast<std::size_t>(slot) + 1, Real(0));
  dts[static_cast<std::size_t>(slot)] = native_dt;
}

int System::rebuild_history_slots(const std::string& name, const std::vector<int>& stored_slots) {
  // Contract (ADC-626): the STORED slots of ring `name` are already restored (restore_history), the
  // per-slot outgoing dt is restored (restore_history_slot_dt), and the SAME Program the checkpoint
  // recorded is
  // installed (the program-hash guard upstream ensures this). A selective ring is owner-qualified
  // by keep_history, so a stored slot IS that exact block's state at that lag. We reconstruct the
  // missing slots by seeding the qualified owner from the nearest OLDER stored slot and re-stepping
  // the installed Program forward, capturing the intermediate owner states.
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::rebuild_history_slots: unknown history '" + name + "'");
  if (!p_->program_.step_)
    throw std::runtime_error(
        "System::rebuild_history_slots: no compiled Program is installed; the ring cannot be "
        "replayed "
        "(install_program before restart, or checkpoint the ring with Dense())");
  std::vector<MultiFab>& ring = it->second;
  const int depth = static_cast<int>(ring.size());
  std::vector<int> anchors = stored_slots;
  std::sort(anchors.begin(), anchors.end());
  anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());
  if (anchors.empty() || anchors.back() != depth - 1)
    throw std::runtime_error("System::rebuild_history_slots: the oldest slot " +
                             std::to_string(depth - 1) + " of history '" + name +
                             "' is not stored; the ring is unreconstructable (nothing older to "
                             "replay it from). The persistence policy must store the oldest slot.");
  if (anchors.front() != 0)
    throw std::runtime_error(
        "System::rebuild_history_slots: the newest slot 0 of history '" + name +
        "' is not stored; slots newer than the first anchor are unbracketed. The persistence "
        "policy must store the newest slot.");
  // A fully-stored ring (Dense): nothing to recompute.
  const std::size_t stored_count = anchors.size();
  if (static_cast<int>(stored_count) == depth)
    return 0;
  if (p_->program_.substeps_ != 1 || p_->program_.stride_ != 1)
    throw std::runtime_error(
        "System::rebuild_history_slots: selective replay requires Program cadence "
        "(substeps=1, stride=1); checkpoint this ring with Dense() for a subcycled or held "
        "Program");
  if (!p_->program_.authorizes_history_replay(name, depth))
    throw std::runtime_error(
        "System::rebuild_history_slots: selective replay for history '" + name +
        "' lacks the installed Program's validated native authority; compile this ring with a "
        "selective checkpoint policy or use Dense()");
  const auto owner_it = p_->program_.hist_.owner.find(name);
  if (owner_it == p_->program_.hist_.owner.end() || owner_it->second < 0 ||
      owner_it->second >= static_cast<int>(p_->sp.size()))
    throw std::runtime_error(
        "System::rebuild_history_slots: selective replay requires an exact owner-qualified "
        "keep_history ring (legacy/unowned history '" +
        name + "' must use Dense())");
  const std::size_t owner = static_cast<std::size_t>(owner_it->second);
  // SAVE bracket: deep-copy every block state, the scheduler cache, the WHOLE history subsystem
  // (rings + slot_dt + initialized), and the accepted Program last-dt ledger. The replay overwrites
  // last_dt_ before every historical call so its own store_history records the proper outgoing
  // interval; that temporary value must not become the accepted runtime's next-store provenance.
  // After restoration the live state U, cache_ and last_dt_ are identities, and only the missing ring
  // slots placed by index below survive.
  std::vector<MultiFab> saved_states;
  saved_states.reserve(p_->sp.size());
  for (auto& block : p_->sp)
    saved_states.push_back(block.U);  // deep copy
  const pops::runtime::program::CacheManager saved_cache = p_->program_.cache_;
  const pops::runtime::program::HistoryManager saved_hist = p_->program_.hist_;
  const Real saved_last_dt = p_->program_.last_dt_;

  // The per-slot outgoing dt, captured from the SAVED snapshot into a stable local vector.
  // CRITICAL: the replay's own store_history / rotate_histories MUTATE p_->program_.hist_.slot_dt, so
  // reading the live map inside the loop would give a moving target. keep_history snapshots the
  // pre-commit state, so dts[j+1] is the interval that produced slot j from older slot j+1.
  std::vector<Real> dts(static_cast<std::size_t>(depth), Real(0));
  auto saved_dt_it = saved_hist.slot_dt.find(name);
  if (saved_dt_it != saved_hist.slot_dt.end()) {
    const std::vector<Real>& sd = saved_dt_it->second;
    for (int k = 0; k < depth && k < static_cast<int>(sd.size()); ++k)
      dts[static_cast<std::size_t>(k)] = sd[static_cast<std::size_t>(k)];
  }
  for (std::size_t anchor = 0; anchor + 1 < anchors.size(); ++anchor) {
    const int newer = anchors[anchor];
    const int older = anchors[anchor + 1];
    for (int j = older - 1; j > newer; --j) {
      const Real replay_dt = dts[static_cast<std::size_t>(j + 1)];
      if (!std::isfinite(static_cast<double>(replay_dt)) || replay_dt <= Real(0))
        throw std::runtime_error(
            "System::rebuild_history_slots: every replayed outgoing dt must be finite and > 0 "
            "before Program execution");
    }
  }

  // Reconstruct the owner state trajectory: for each gap between adjacent anchors (older anchor at a
  // LARGER index, newer at a SMALLER one; time increases as the index decreases), restore the same
  // accepted image, seed the owner from the older stored slot, then step forward through omitted
  // slots. Placement is BY INDEX (no rotate) -> the ADC-538 rotation-invalidation edge is sidestepped.
  std::vector<MultiFab> reconstructed(static_cast<std::size_t>(depth));
  const auto restore_saved = [&] {
    for (std::size_t b = 0; b < p_->sp.size(); ++b)
      p_->sp[b].U = saved_states[b];
    p_->program_.cache_ = saved_cache;
    p_->program_.hist_ = saved_hist;
    p_->program_.last_dt_ = saved_last_dt;
  };
  try {
    for (std::size_t a = 0; a + 1 < anchors.size(); ++a) {
      const int older = anchors[a + 1];  // larger index = further back in time
      const int newer = anchors[a];      // smaller index = closer to now
      restore_saved();
      // Seed the qualified owner with the older stored slot's state.
      pops::lincomb(p_->sp[owner].U, Real(1),
                    saved_hist.histories.at(name)[static_cast<std::size_t>(older)], Real(0),
                    p_->sp[owner].U);
      // Adjacent anchors bracket only omitted slots. Stop before `newer`: that exact checkpoint
      // value needs no Program execution. The interval from slot j+1 to j is dts[j+1].
      for (int j = older - 1; j > newer; --j) {
        p_->program_.last_dt_ = dts[static_cast<std::size_t>(j + 1)];
        p_->program_.step_(static_cast<double>(dts[static_cast<std::size_t>(j + 1)]));
        reconstructed[static_cast<std::size_t>(j)] =
            p_->sp[owner].U;  // deep copy the fresh owner state
      }
    }
  } catch (...) {
    restore_saved();
    throw;
  }

  // RESTORE bracket: undo every replay side effect (block states, cache, history and last_dt_).
  restore_saved();

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
// mismatch, then call pops_install_program(this), whose shared facade factory selects the provider
// and installs the macro-step closure. The .so stays loaded for the process lifetime.
POPS_EXPORT void System::install_program(const std::string& so_path) {
  require_assembling(p_->lifecycle_,
                     "install_program");  // frozen once pops.bind completes (ADC-592)
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
    auto manifest_fn =
        reinterpret_cast<const char* (*)()>(pops::dynlib::sym(h, "pops_program_route_manifest"));
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
  std::vector<pops::runtime::program::ProgramHistoryReplayAuthority> history_replay_authorities;
  try {
    operator_authorities = pops::runtime::program::read_program_operator_authorities(h);
    history_replay_authorities = pops::runtime::program::read_program_history_replay_authorities(h);
  } catch (...) {
    pops::dynlib::close(h);
    throw;
  }
  auto install = reinterpret_cast<void (*)(System*)>(pops::dynlib::sym(h, "pops_install_program"));
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
        throw std::runtime_error("field operator '" + op.name + "' requires solver '" +
                                 need_solver + "'");
      }
    }
  } catch (...) {
    pops::dynlib::close(h);
    throw;
  }
  // Resolve every optional scalar ABI before the first facade mutation. A module that declares a
  // dt bound but omits its target entry is malformed; silently falling back to native CFL would
  // execute different numerics from the authored Program.
  using has_dt_t = bool (*)();
  using dt_bound_t = pops::Real (*)(System*, pops::Real);
  auto has_dt = reinterpret_cast<has_dt_t>(pops::dynlib::sym(h, "pops_program_has_dt_bound"));
  auto dt_bound = reinterpret_cast<dt_bound_t>(pops::dynlib::sym(h, "pops_program_dt_bound"));
  const bool program_has_dt_bound = has_dt && has_dt();
  if (program_has_dt_bound && !dt_bound) {
    pops::dynlib::close(h);
    throw std::runtime_error(
        "System::install_program: Program declares a dt bound but pops_program_dt_bound is "
        "missing; regenerate the System artifact");
  }
  auto hash_fn = reinterpret_cast<const char* (*)()>(pops::dynlib::sym(h, "pops_program_hash"));
  const std::string installed_hash = hash_fn ? std::string(hash_fn()) : std::string();
  auto install_boundaries =
      reinterpret_cast<void (*)(System*)>(pops::dynlib::sym(h, "pops_install_field_boundaries"));

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
  std::vector<int> program_block_map;
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
    program_block_map.assign(static_cast<std::size_t>(n), -1);
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
      program_block_map[static_cast<std::size_t>(p)] = found;
    }
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
  std::map<int, std::vector<double>> program_param_defaults;
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
      for (int i = 0; i < np; ++i) {
        const int blk = pblock(i);
        const int idx = pindex(i);
        std::vector<double>& d = program_param_defaults[blk];
        if (static_cast<int>(d.size()) <= idx)
          d.resize(static_cast<std::size_t>(idx) + 1, 0.0);
        d[static_cast<std::size_t>(idx)] = pdef(i);
      }
    }
  }
  // Dynamic field-boundary launchers are installed from the same problem.so that owns their direct
  // function pointers.  Static-boundary artifacts export no entry and keep the historical fast path.
  // Install only after ABI/requirements/block/parameter preflight has completed.
  auto previous_install = p_->program_.capture_artifact_step_install();
  using FieldInstallSnapshot =
      typename field_solver::SystemFieldSolver<Impl>::ProgramInstallSnapshot;
  std::optional<FieldInstallSnapshot> previous_fields;
  try {
    previous_fields.emplace(p_->fields_.begin_program_install());
    p_->program_.reset_artifact_candidate_state();
    // The generated prelude may resolve blocks and parameters before ctx.install() publishes the
    // closure. Install the candidate image first; install_unverified_step then revokes it, and the
    // authenticated image is republished only after the exact one-step witness succeeds.
    p_->program_.block_map_ = program_block_map;
    p_->program_.block_params_.clear();
    for (const auto& [block, defaults] : program_param_defaults)
      seed_program_params(block, defaults);
    p_->program_.operator_authorities_ = operator_authorities;
    install(this);
    p_->program_.require_exact_artifact_step_install(previous_install, "System::install_program:");

    p_->program_.block_map_ = std::move(program_block_map);
    for (const auto& [block, defaults] : program_param_defaults)
      seed_program_params(block, defaults);
    p_->program_.operator_authorities_ = std::move(operator_authorities);
    p_->program_.history_replay_authorities_ = std::move(history_replay_authorities);
    p_->program_.installed_hash_ = installed_hash;
    if (program_has_dt_bound) {
      System* self = this;
      p_->program_.dt_bound_ = [self, dt_bound](Real cfl) -> Real { return dt_bound(self, cfl); };
    }
    p_->program_.artifact_backed_ = true;

    // Dynamic field kernels are installed last. Their structural snapshot is restored before the
    // DSO is closed if any generated setter fails.
    if (install_boundaries)
      install_boundaries(this);
    p_->fields_.commit_program_install();
  } catch (...) {
    const std::exception_ptr failure = std::current_exception();
    if (previous_fields)
      p_->fields_.restore_program_install_snapshot(std::move(*previous_fields));
    p_->program_.rollback_artifact_step_install(std::move(previous_install));
    pops::dynlib::close(h);
    std::rethrow_exception(failure);
  }
  // .so left loaded for the duration of the process (the installed closure points to code in it).
}
// Scheduler-cache checkpoint/restart seam (ADC-458, Spec 3 section 30): the System owns the cache, so
// the facade (sim.checkpoint / sim.restart) gathers and restores it DIRECTLY -- reusing the SAME global
// gather (gather_global, via copy_state) / scatter (write_state) machinery as the block state and the
// history rings, so the round-trip is MPI-safe and bit-identical under np>1. Mirrors the history seam.
std::vector<int> System::program_cache_nodes() const {
  return p_->program_.cache_.node_ids();
}
std::string System::program_cache_name(int node_id) const {
  return p_->program_.cache_.name_of(node_id);
}
int System::program_cache_last_update_step(int node_id) const {
  return p_->program_.cache_.last_update_step(node_id);
}
double System::program_cache_accumulated_dt(int node_id) const {
  return static_cast<double>(p_->program_.cache_.accumulated_dt_of(node_id));
}
int System::program_cache_ncomp(int node_id) const {
  return p_->program_.cache_.ncomp_of(node_id);
}
int System::program_cache_ngrow(int node_id) const {
  return p_->program_.cache_.ngrow_of(node_id);
}
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
        "System::restore_program_cache: no block exists yet; the cache value is co-distributed "
        "with "
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
