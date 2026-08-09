/// @file
/// @brief Exact compile-time-ranked System persistence and Program-install surface.

#include "system_impl.hpp"  // ADC-632: shared System::Impl + facade helpers (runtime-private)

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/dynamic/dynlib.hpp>
#include <pops/runtime/program/module_metadata.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {
namespace {

template <int Dim>
Extent<Dim> uniform_ghosts(int width) {
  if (width < 0)
    throw std::invalid_argument("System field ghost width must be non-negative");
  Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    ghosts[axis] = width;
  return ghosts;
}

template <int Dim>
void require_collective_exact_layout(const MultiFab<Dim>& value, const mesh::BoxArray<Dim>& layout,
                                     const mesh::Distribution<Dim>& distribution,
                                     const Index<Dim>& local_rank, int components,
                                     const std::string& operation) {
  const long local_invalid = value.layout() != layout || value.distribution() != distribution ||
                                     value.local_rank() != local_rank || value.ncomp() != components
                                 ? 1L
                                 : 0L;
  if (all_reduce_max(local_invalid) != 0)
    throw std::invalid_argument(operation + ": value does not match the exact System layout");
}

}  // namespace

template <int Dim>
void System<Dim>::set_clock(double t, int macro_step) {
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

template <int Dim>
void System<Dim>::store_history(const std::string& name, const MultiFab<Dim>& value) {
  store_history(name, value, static_cast<double>(p_->program_.last_dt_));
}

template <int Dim>
void System<Dim>::store_history(const std::string& name, const MultiFab<Dim>& value,
                                double outgoing_dt) {
  if (all_reduce_max(!std::isfinite(outgoing_dt) || outgoing_dt < 0.0 ? 1L : 0L) != 0)
    throw std::runtime_error(
        "System::store_history: outgoing logical-clock dt must be finite and non-negative");
  auto it = p_->program_.hist_.histories.find(name);
  const bool exists = it != p_->program_.hist_.histories.end();
  const std::string store_request =
      std::string(exists ? "1:" : "0:") + std::to_string(std::bit_cast<std::uint64_t>(outgoing_dt));
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view(name), std::string_view(store_request)}}))
    throw std::invalid_argument("System::store_history request differs between MPI ranks");
  if (!exists)
    throw std::runtime_error("System::store_history: unknown history '" + name +
                             "' (register it first)");
  std::vector<MultiFab<Dim>>& ring = it->second;
  if (all_reduce_max(ring.empty() ? 1L : 0L) != 0)
    throw std::runtime_error("System::store_history: registered history ring is empty");
  const std::string ring_contract =
      std::to_string(ring.size()) + ":" + std::to_string(ring.front().ncomp());
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view(name), std::string_view(ring_contract)}}))
    throw std::invalid_argument("System::store_history ring differs between MPI ranks");
  require_collective_exact_layout(value, p_->ba, p_->dm, p_->local_rank, ring.front().ncomp(),
                                  "System::store_history");
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

template <int Dim>
void System<Dim>::rotate_histories() {
  // Shift each ring one step at the end of a macro-step (O(1) std::swap chain, buffer recycled into
  // slot [0]); the grid-free ring bookkeeping lives in the extracted Program subsystem (ADC-594).
  p_->program_.hist_.rotate();
}

template <int Dim>
void System<Dim>::rotate_histories(const std::string& clock_identity) {
  if (clock_identity.empty())
    throw std::runtime_error("System::rotate_histories: clock identity must be non-empty");
  p_->program_.hist_.rotate(clock_identity);
}

// Multistep history checkpoint/restart seam (ADC-406b): the System owns the rings, so the checkpoint
// facade (sim.checkpoint / sim.restart) gathers and restores them DIRECTLY -- reusing the SAME global
// exact-ranked gather/write machinery as the block state, so the round-trip is
// MPI-safe and bit-identical under np>1. No .so checkpoint_extra ABI is needed for the buffers.
template <int Dim>
std::vector<std::string> System<Dim>::history_names() const {
  // enumeration lives in the extracted Program subsystem (ADC-594)
  return p_->program_.hist_.names();
}
template <int Dim>
int System<Dim>::history_depth(const std::string& name) const {
  auto it = p_->program_.hist_.depth.find(name);
  if (it == p_->program_.hist_.depth.end())
    throw std::runtime_error("System::history_depth: unknown history '" + name + "'");
  return it->second;
}
template <int Dim>
int System<Dim>::history_ncomp(const std::string& name) const {
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::history_ncomp: unknown history '" + name + "'");
  return it->second[0].ncomp();
}
template <int Dim>
std::vector<double> System<Dim>::history_global(const std::string& name, int slot) const {
  const std::string request = std::to_string(slot);
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view(name), std::string_view(request)}}))
    throw std::invalid_argument("System::history_global request differs between MPI ranks");
  auto it = p_->program_.hist_.histories.find(name);
  const bool valid = it != p_->program_.hist_.histories.end() && !it->second.empty() && slot >= 0 &&
                     slot < static_cast<int>(it->second.size());
  if (all_reduce_max(valid ? 0L : 1L) != 0)
    throw std::runtime_error("System::history_global request does not name one valid ring slot");
  const std::vector<MultiFab<Dim>>& ring = it->second;
  return runtime::system::marshaling::gather_global(ring[static_cast<std::size_t>(slot)], p_->dom,
                                                    ring.front().ncomp());
}
template <int Dim>
bool System<Dim>::history_initialized(const std::string& name) const {
  auto it = p_->program_.hist_.initialized.find(name);
  if (it == p_->program_.hist_.initialized.end())
    throw std::runtime_error("System::history_initialized: unknown history '" + name + "'");
  return it->second;
}
template <int Dim>
int System<Dim>::history_fill_count(const std::string& name) const {
  auto it = p_->program_.hist_.fill_count.find(name);
  if (it == p_->program_.hist_.fill_count.end())
    throw std::runtime_error("System::history_fill_count: unknown history '" + name + "'");
  return it->second;
}
template <int Dim>
void System<Dim>::restore_history(const std::string& name, int slot,
                                  const std::vector<double>& values) {
  if (all_reduce_max(slot < 0 ? 1L : 0L) != 0)
    throw std::runtime_error("System::restore_history: slot=" + std::to_string(slot) +
                             " must be >= 0 for history '" + name + "'");
  if (all_reduce_max(p_->sp.empty() ? 1L : 0L) != 0)
    throw std::runtime_error(
        "System::restore_history: a block must exist before restoring a history ring");
  auto it = p_->program_.hist_.histories.find(name);
  const bool exists = it != p_->program_.hist_.histories.end();
  if (all_reduce_max(exists && it->second.empty() ? 1L : 0L) != 0)
    throw std::runtime_error("System::restore_history: registered history ring is empty");
  const int expected_components = exists ? it->second.front().ncomp() : p_->sp.front().ncomp;
  const std::string request =
      std::to_string(slot) + ":" + (exists ? "1" : "0") + ":" + std::to_string(expected_components);
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view(name), std::string_view(request)}}))
    throw std::invalid_argument("System::restore_history request differs between MPI ranks");
  const std::size_t cells = runtime::system::marshaling::checked_cell_count(p_->dom);
  if (cells != 0 && static_cast<std::size_t>(expected_components) >
                        std::numeric_limits<std::size_t>::max() / cells)
    throw std::overflow_error("System::restore_history: payload size exceeds size_t");
  if (all_reduce_max(
          values.size() != static_cast<std::size_t>(expected_components) * cells ? 1L : 0L) != 0)
    throw std::invalid_argument("System::restore_history: payload has the wrong exact shape");
  if (!exists) {
    // The program will re-register the ring on its first post-restart step, but we restore BEFORE that
    // step; register it now (depth = slot + 1, grown as deeper slots arrive) so the values land. Uses
    // the SAME co-distributed (ba, dm, block 0 ncomp) ring as register_history.
    register_history(name, slot >= 1 ? slot : 1);
    it = p_->program_.hist_.histories.find(name);
  }
  std::vector<MultiFab<Dim>>& ring = it->second;
  if (ring.front().ncomp() != expected_components)
    throw std::runtime_error(
        "System::restore_history: registered ring component count differs from block 0");
  if (slot >= static_cast<int>(ring.size())) {
    // A deeper slot than currently registered: grow the ring (zero-filled tail) so it fits, matching
    // register_history's idempotent growth.
    const int ncomp = ring[0].ncomp();
    for (int k = static_cast<int>(ring.size()); k <= slot; ++k) {
      MultiFab<Dim> s(p_->ba, p_->dm, p_->local_rank, ncomp, uniform_ghosts<Dim>(1));
      s.set_val(Real(0));
      ring.push_back(std::move(s));
    }
    p_->program_.hist_.depth[name] = static_cast<int>(ring.size());
  }
  // Scatter the GLOBAL component-major buffer into the exact ranked slot. Structural ownership and
  // payload consensus are authenticated collectively before any resident Fab is mutated.
  runtime::system::marshaling::write_global(ring[static_cast<std::size_t>(slot)], p_->dom, values,
                                            ring.front().ncomp());
}
template <int Dim>
void System<Dim>::set_history_initialized(const std::string& name, bool initialized) {
  auto it = p_->program_.hist_.initialized.find(name);
  if (it == p_->program_.hist_.initialized.end())
    throw std::runtime_error("System::set_history_initialized: unknown history '" + name +
                             "' (restore its slots first)");
  it->second = initialized;
  p_->program_.hist_.fill_count[name] = initialized ? p_->program_.hist_.depth.at(name) : 0;
  p_->program_.hist_.store_pending[name] = false;
}
template <int Dim>
void System<Dim>::restore_history_fill_count(const std::string& name, int fill_count) {
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
template <int Dim>
double System<Dim>::history_slot_dt(const std::string& name, int slot) const {
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

template <int Dim>
void System<Dim>::restore_history_slot_dt(const std::string& name, int slot, double dt) {
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

template <int Dim>
int System<Dim>::rebuild_history_slots(const std::string& name,
                                       const std::vector<int>& stored_slots) {
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
  std::vector<MultiFab<Dim>>& ring = it->second;
  const int depth = static_cast<int>(ring.size());
  std::vector<int> anchors = stored_slots;
  std::sort(anchors.begin(), anchors.end());
  anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());
  for (const int anchor : anchors)
    if (anchor < 0 || anchor >= depth)
      throw std::out_of_range(
          "System::rebuild_history_slots: stored slot lies outside the history ring");
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
  std::vector<MultiFab<Dim>> saved_states;
  saved_states.reserve(p_->sp.size());
  for (auto& block : p_->sp)
    saved_states.push_back(block.U);  // deep copy
  const pops::runtime::program::CacheManager<Dim> saved_cache = p_->program_.cache_;
  const pops::runtime::program::HistoryManager<Dim> saved_hist = p_->program_.hist_;
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
  std::vector<MultiFab<Dim>> reconstructed(static_cast<std::size_t>(depth));
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
        p_->program_.run_balance_replay("System::rebuild_history_slots", [&] {
          p_->program_.step_(static_cast<double>(dts[static_cast<std::size_t>(j + 1)]));
        });
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
  std::vector<MultiFab<Dim>>& out_ring = p_->program_.hist_.histories.at(name);
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
template <int Dim>
POPS_EXPORT void System<Dim>::install_program(const std::string& so_path) {
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
  auto install =
      reinterpret_cast<void (*)(System<Dim>*)>(pops::dynlib::sym(h, "pops_install_program"));
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
      // (a) Auxiliary requirements are no longer inferred from a physical string.  Every consumer
      // carries complete owner-qualified ComponentKeys and the sealed provider DAG rejects a missing
      // producer or a contract mismatch before allocation/publication.
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
  using dt_bound_t = pops::Real (*)(System<Dim>*, pops::Real);
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
  auto install_boundaries = reinterpret_cast<void (*)(System<Dim>*)>(
      pops::dynlib::sym(h, "pops_install_field_boundaries"));

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

  // Program-owned dynamic boundary launchers are an overlay on the static field-plan registry.
  // The generated entry receives the real exact-ranked facade, but its kernel, logical-timepoint,
  // and parameter setters are redirected into this attempt-local stage. Nothing live is mutated
  // until every authority has validated, all ranks agree on the ordered registry, and every
  // already-materialized solver has confirmed that it can consume the candidate. A missing entry is
  // meaningful: it selects the static baseline and therefore removes any overlay owned by the
  // previous Program artifact.
  using boundary_registry = runtime::program::ArtifactFieldBoundaryAuthorityRegistry<Dim>;
  using kernel_registry = std::map<std::string, std::optional<CompiledFieldBoundaryKernel<Dim>>>;
  using field_plan_registry = decltype(p_->field_plans_);
  boundary_registry static_boundary_baseline;
  kernel_registry materialized_candidate;
  kernel_registry materialized_previous;
  field_plan_registry candidate_field_plans;
  std::string candidate_boundary_contract;
  std::exception_ptr boundary_preparation_error;

  // Baseline preparation and stage allocation finish collectively before the DSO entry is invoked.
  // Otherwise one rank rejecting a changed registry could skip the ProgramContext communicator
  // construction while a peer entered it.
  try {
    if (p_->program_.artifact_field_boundary_stage_)
      throw std::logic_error(
          "System::install_program: a field-boundary artifact transaction is already active");
    auto capture_authorities = [](const field_plan_registry& plans) {
      boundary_registry result;
      for (const auto& [slot, plan] : plans)
        result.emplace(slot,
                       runtime::program::ArtifactFieldBoundaryAuthority<Dim>{
                           plan.boundary_kernel, plan.boundary_point, plan.boundary_parameters});
      return result;
    };
    const boundary_registry current = capture_authorities(p_->field_plans_);
    if (p_->program_.artifact_field_boundary_baseline_) {
      static_boundary_baseline = *p_->program_.artifact_field_boundary_baseline_;
      if (static_boundary_baseline.size() != current.size())
        throw std::logic_error(
            "System::install_program: field-plan registry changed after an artifact boundary "
            "baseline was established; create a fresh runtime");
      for (const auto& [slot, authority] : current) {
        (void)authority;
        if (!static_boundary_baseline.contains(slot))
          throw std::logic_error(
              "System::install_program: field-plan registry changed after an artifact boundary "
              "baseline was established; create a fresh runtime");
      }
    } else {
      static_boundary_baseline = current;
    }

    p_->program_.artifact_field_boundary_stage_.emplace();
    p_->program_.artifact_field_boundary_stage_->authorities = static_boundary_baseline;
  } catch (...) {
    boundary_preparation_error = std::current_exception();
  }
  if (all_reduce_max(boundary_preparation_error ? 1L : 0L) != 0) {
    p_->program_.artifact_field_boundary_stage_.reset();
    pops::dynlib::close(h);
    if (n_ranks() == 1 && boundary_preparation_error)
      std::rethrow_exception(boundary_preparation_error);
    throw std::runtime_error(
        "System::install_program: field-boundary baseline preparation failed collectively");
  }

  std::exception_ptr boundary_installer_error;
  try {
    if (install_boundaries)
      install_boundaries(this);
  } catch (...) {
    boundary_installer_error = std::current_exception();
  }
  if (all_reduce_max(boundary_installer_error ? 1L : 0L) != 0) {
    p_->program_.artifact_field_boundary_stage_.reset();
    pops::dynlib::close(h);
    if (n_ranks() == 1 && boundary_installer_error)
      std::rethrow_exception(boundary_installer_error);
    throw std::runtime_error(
        "System::install_program: generated field-boundary installer failed collectively");
  }

  boundary_preparation_error = nullptr;
  try {
    candidate_field_plans = p_->field_plans_;
    const auto& staged = p_->program_.artifact_field_boundary_stage_->authorities;
    if (staged.size() != candidate_field_plans.size())
      throw std::logic_error(
          "System::install_program: artifact boundary candidate does not exactly cover the "
          "field-plan registry");
    for (const auto& [slot, authority] : staged) {
      const auto plan = candidate_field_plans.find(slot);
      if (plan == candidate_field_plans.end())
        throw std::logic_error(
            "System::install_program: artifact boundary candidate names an unknown field plan");
      if (authority.kernel)
        authority.kernel->validate();
      if ((!plan->second.boundary_state_blocks.empty() ||
           !plan->second.boundary_field_blocks.empty()) &&
          !authority.kernel)
        throw std::logic_error(
            "System::install_program: field boundary dependencies require one complete generated "
            "kernel authority");
      plan->second.boundary_kernel = authority.kernel;
      plan->second.boundary_point = authority.point;
      plan->second.boundary_parameters = authority.parameters;
    }

    ExactContractBuilder contract;
    contract.text("pops.system.artifact-field-boundary-registry")
        .scalar(std::uint32_t{2})
        .scalar(static_cast<std::uint32_t>(Dim))
        .scalar(static_cast<std::uint64_t>(candidate_field_plans.size()));
    for (const auto& [slot, plan] : candidate_field_plans) {
      contract.text(slot).presence(plan.boundary_kernel.has_value());
      if (plan.boundary_kernel)
        contract.text(plan.boundary_kernel->identity)
            .text(plan.boundary_kernel->residual_identity)
            .text(plan.boundary_kernel->jvp_identity)
            .presence(plan.boundary_kernel->observes_iteration);
      contract.presence(plan.boundary_point.has_value());
      if (plan.boundary_point)
        contract.scalar(plan.boundary_point->time)
            .scalar(plan.boundary_point->dt)
            .scalar(plan.boundary_point->clock_slot)
            .scalar(plan.boundary_point->partition_slot)
            .scalar(plan.boundary_point->stage_slot)
            .scalar(plan.boundary_point->level)
            .scalar(plan.boundary_point->step)
            .scalar(plan.boundary_point->substep)
            .scalar(plan.boundary_point->iteration);
      contract.sequence(plan.boundary_parameters);
    }
    candidate_boundary_contract = std::move(contract).release();

    // Pre-copy both materialization images while failure is still harmless.  After the plan-registry
    // swap these optionals are moved into the solvers through a noexcept publication seam.
    for (const auto& [slot, field] : p_->named_fields_) {
      const auto candidate = candidate_field_plans.find(slot);
      const auto previous = p_->field_plans_.find(slot);
      if (candidate == candidate_field_plans.end() || previous == p_->field_plans_.end())
        throw std::logic_error(
            "System::install_program: materialized field lacks its qualified field plan");
      field->validate_boundary_kernel_replacement(candidate->second.boundary_kernel);
      materialized_candidate.emplace(slot, candidate->second.boundary_kernel);
      materialized_previous.emplace(slot, previous->second.boundary_kernel);
    }
  } catch (...) {
    boundary_preparation_error = std::current_exception();
  }
  p_->program_.artifact_field_boundary_stage_.reset();
  if (all_reduce_max(boundary_preparation_error ? 1L : 0L) != 0) {
    pops::dynlib::close(h);
    if (n_ranks() == 1 && boundary_preparation_error)
      std::rethrow_exception(boundary_preparation_error);
    throw std::runtime_error(
        "System::install_program: field-boundary artifact preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-artifact-field-boundary-registry", candidate_boundary_contract}})) {
    pops::dynlib::close(h);
    throw std::runtime_error(
        "System::install_program: generated field-boundary authorities differ between MPI ranks");
  }

  // Install the boundary registry before the Program materializes any closure that may call a field
  // solve.  candidate_field_plans retains the complete prior image after the noexcept swap and is
  // therefore the rollback journal for every subsequent installer failure.
  using artifact_install_snapshot = decltype(p_->program_.capture_artifact_step_install());
  std::optional<artifact_install_snapshot> previous_install;
  std::exception_ptr snapshot_error;
  try {
    previous_install.emplace(p_->program_.capture_artifact_step_install());
  } catch (...) {
    snapshot_error = std::current_exception();
  }
  if (all_reduce_max(snapshot_error ? 1L : 0L) != 0) {
    pops::dynlib::close(h);
    if (n_ranks() == 1 && snapshot_error)
      std::rethrow_exception(snapshot_error);
    throw std::runtime_error(
        "System::install_program: Program rollback snapshot failed collectively");
  }
  const bool previous_field_plan_consensus = p_->field_plan_consensus_verified_;
  bool boundary_registry_published = false;
  try {
    p_->field_plans_.swap(candidate_field_plans);
    p_->field_plan_consensus_verified_ = false;
    for (auto& [slot, kernel] : materialized_candidate) {
      const auto field = p_->named_fields_.find(slot);
      if (field == p_->named_fields_.end())
        std::terminate();
      field->second->replace_boundary_kernel(std::move(kernel));
    }
    boundary_registry_published = true;

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
    p_->program_.require_exact_artifact_step_install(*previous_install, "System::install_program:");

    p_->program_.block_map_ = std::move(program_block_map);
    for (const auto& [block, defaults] : program_param_defaults)
      seed_program_params(block, defaults);
    p_->program_.operator_authorities_ = std::move(operator_authorities);
    p_->program_.history_replay_authorities_ = std::move(history_replay_authorities);
    p_->program_.installed_hash_ = installed_hash;
    if (program_has_dt_bound) {
      System<Dim>* self = this;
      p_->program_.dt_bound_ = [self, dt_bound](Real cfl) -> Real { return dt_bound(self, cfl); };
    }
    p_->program_.artifact_backed_ = true;

    if (!p_->program_.artifact_field_boundary_baseline_)
      p_->program_.artifact_field_boundary_baseline_.emplace(std::move(static_boundary_baseline));

  } catch (...) {
    const std::exception_ptr failure = std::current_exception();
    p_->program_.rollback_artifact_step_install(std::move(*previous_install));
    if (boundary_registry_published) {
      p_->field_plans_.swap(candidate_field_plans);
      p_->field_plan_consensus_verified_ = previous_field_plan_consensus;
      for (auto& [slot, kernel] : materialized_previous) {
        const auto field = p_->named_fields_.find(slot);
        if (field == p_->named_fields_.end())
          std::terminate();
        field->second->replace_boundary_kernel(std::move(kernel));
      }
    }
    pops::dynlib::close(h);
    std::rethrow_exception(failure);
  }
  // .so left loaded for the duration of the process (the installed closure points to code in it).
}
// Scheduler-cache checkpoint/restart seam (ADC-458, Spec 3 section 30): the System owns the cache, so
// the facade (sim.checkpoint / sim.restart) gathers and restores it DIRECTLY -- reusing the SAME global
// exact-ranked global gather/write machinery as the block state and the
// history rings, so the round-trip is MPI-safe and bit-identical under np>1. Mirrors the history seam.
template <int Dim>
std::vector<int> System<Dim>::program_cache_nodes() const {
  return p_->program_.cache_.node_ids();
}
template <int Dim>
std::string System<Dim>::program_cache_name(int node_id) const {
  return p_->program_.cache_.name_of(node_id);
}
template <int Dim>
int System<Dim>::program_cache_last_update_step(int node_id) const {
  return p_->program_.cache_.last_update_step(node_id);
}
template <int Dim>
double System<Dim>::program_cache_accumulated_dt(int node_id) const {
  return static_cast<double>(p_->program_.cache_.accumulated_dt_of(node_id));
}
template <int Dim>
int System<Dim>::program_cache_ncomp(int node_id) const {
  return p_->program_.cache_.ncomp_of(node_id);
}
template <int Dim>
int System<Dim>::program_cache_ngrow(int node_id) const {
  const Extent<Dim> ghosts = p_->program_.cache_.ghosts_of(node_id);
  const int width = ghosts[0];
  for (int axis = 1; axis < Dim; ++axis)
    if (ghosts[axis] != width)
      throw std::runtime_error(
          "System::program_cache_ngrow: checkpoint schema requires one uniform ND ghost width");
  return width;
}
template <int Dim>
std::vector<double> System<Dim>::program_cache_global(int node_id) const {
  // The cache value is co-distributed with block 0 and uses the same authenticated component-major
  // collective gather as state_global and history_global.
  const std::string request = std::to_string(node_id);
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("program-cache-global"), std::string_view(request)}}))
    throw std::invalid_argument("System::program_cache_global request differs between MPI ranks");
  if (all_reduce_max(p_->program_.cache_.valid(node_id) ? 0L : 1L) != 0)
    throw std::out_of_range("System::program_cache_global node is absent on at least one MPI rank");
  const MultiFab<Dim>& value = p_->program_.cache_.value_of(node_id);
  return runtime::system::marshaling::gather_global(value, p_->dom, value.ncomp());
}
template <int Dim>
void System<Dim>::restore_program_cache(int node_id, int ncomp, int ngrow, int last_update_step,
                                        double accumulated_dt, const std::string& name,
                                        const std::vector<double>& values) {
  if (all_reduce_max(p_->sp.empty() ? 1L : 0L) != 0)
    throw std::runtime_error(
        "System::restore_program_cache: no block exists yet; the cache value is co-distributed "
        "with "
        "block 0's storage (replay the composition before restart)");
  // Allocate a value co-distributed with block 0 (ba/dm, @p ncomp comps, @p ngrow ghosts -- the SAME
  // ghost width the slot was cached with: 1 for the aux, the block-state width for a held scratch) and
  // stage the GLOBAL payload into it through the exact-ranked collective writer, then re-key the
  // slot with its bookkeeping. MPI-safe (all ranks call), bit-identical under np>1.
  const std::string request = std::to_string(node_id) + ":" + std::to_string(ncomp) + ":" +
                              std::to_string(ngrow) + ":" + std::to_string(last_update_step) + ":" +
                              std::to_string(std::bit_cast<std::uint64_t>(accumulated_dt));
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view(name), std::string_view(request)}}))
    throw std::invalid_argument("System::restore_program_cache request differs between MPI ranks");
  const long local_metadata_invalid =
      ncomp < 1 || ngrow < 0 || last_update_step < -1 || !std::isfinite(accumulated_dt) ||
              accumulated_dt < 0.0 ||
              !std::isfinite(static_cast<double>(static_cast<Real>(accumulated_dt)))
          ? 1L
          : 0L;
  if (all_reduce_max(local_metadata_invalid) != 0)
    throw std::invalid_argument("System::restore_program_cache: checkpoint metadata is invalid");
  const std::size_t cells = runtime::system::marshaling::checked_cell_count(p_->dom);
  if (cells != 0 &&
      static_cast<std::size_t>(ncomp) > std::numeric_limits<std::size_t>::max() / cells)
    throw std::overflow_error("System::restore_program_cache: payload size exceeds size_t");
  if (all_reduce_max(values.size() != static_cast<std::size_t>(ncomp) * cells ? 1L : 0L) != 0)
    throw std::invalid_argument("System::restore_program_cache: payload has the wrong exact shape");

  MultiFab<Dim> value(p_->ba, p_->dm, p_->local_rank, ncomp, uniform_ghosts<Dim>(ngrow));
  value.set_val(Real(0));
  runtime::system::marshaling::write_global(value, p_->dom, values, ncomp);
  p_->program_.cache_.restore_slot(node_id, std::move(value), last_update_step,
                                   static_cast<Real>(accumulated_dt), name);
}

template void System<kNativeDimension>::set_clock(double, int);
template void System<kNativeDimension>::store_history(const std::string&,
                                                      const MultiFab<kNativeDimension>&);
template void System<kNativeDimension>::store_history(const std::string&,
                                                      const MultiFab<kNativeDimension>&, double);
template void System<kNativeDimension>::rotate_histories();
template void System<kNativeDimension>::rotate_histories(const std::string&);
template std::vector<std::string> System<kNativeDimension>::history_names() const;
template int System<kNativeDimension>::history_depth(const std::string&) const;
template int System<kNativeDimension>::history_ncomp(const std::string&) const;
template std::vector<double> System<kNativeDimension>::history_global(const std::string&,
                                                                      int) const;
template bool System<kNativeDimension>::history_initialized(const std::string&) const;
template int System<kNativeDimension>::history_fill_count(const std::string&) const;
template void System<kNativeDimension>::restore_history(const std::string&, int,
                                                        const std::vector<double>&);
template void System<kNativeDimension>::set_history_initialized(const std::string&, bool);
template void System<kNativeDimension>::restore_history_fill_count(const std::string&, int);
template double System<kNativeDimension>::history_slot_dt(const std::string&, int) const;
template void System<kNativeDimension>::restore_history_slot_dt(const std::string&, int, double);
template int System<kNativeDimension>::rebuild_history_slots(const std::string&,
                                                             const std::vector<int>&);
template void System<kNativeDimension>::install_program(const std::string&);
template std::vector<int> System<kNativeDimension>::program_cache_nodes() const;
template std::string System<kNativeDimension>::program_cache_name(int) const;
template int System<kNativeDimension>::program_cache_last_update_step(int) const;
template double System<kNativeDimension>::program_cache_accumulated_dt(int) const;
template int System<kNativeDimension>::program_cache_ncomp(int) const;
template int System<kNativeDimension>::program_cache_ngrow(int) const;
template std::vector<double> System<kNativeDimension>::program_cache_global(int) const;
template void System<kNativeDimension>::restore_program_cache(int, int, int, int, double,
                                                              const std::string&,
                                                              const std::vector<double>&);

}  // namespace pops
