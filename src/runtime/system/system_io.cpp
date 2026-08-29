/// @file
/// @brief Exact compile-time-ranked System persistence and Program-install surface.

#include "system_impl.hpp"  // ADC-632: shared System::Impl + facade helpers (runtime-private)

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/dynamic/dynlib.hpp>
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/program/program_loader.hpp>
#include <pops/runtime/program/module_metadata.hpp>
#include <pops/runtime/program/program_persistent_value_checkpoint.hpp>
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
std::vector<std::uint8_t> System<Dim>::capture_program_persistent_value_checkpoint() const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return runtime::program::serialize_program_persistent_value_checkpoint(
      runtime::program::capture_program_persistent_value_checkpoint(
          p_->program_.persistent_values()));
}

template <int Dim>
runtime::program::PreparedProgramPersistentValueRestore
System<Dim>::prepare_program_persistent_value_restore(
    const std::vector<std::uint8_t>& payload) const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  const auto checkpoint = runtime::program::deserialize_program_persistent_value_checkpoint(
      std::span<const std::uint8_t>(payload.data(), payload.size()));
  const auto& receipt = p_->program_.artifact_publication_receipt();
  if (!receipt)
    throw std::logic_error(
        "System Program persistent restore requires an installed prepared artifact");
  auto prepared = runtime::program::prepare_program_persistent_value_restore(
      checkpoint, receipt->resource_plan);
  return runtime::program::PreparedProgramPersistentValueRestore(
      std::move(prepared), p_->program_.step_install_generation_);
}

template <int Dim>
void System<Dim>::publish_program_persistent_value_restore(
    runtime::program::PreparedProgramPersistentValueRestore& prepared) {
  if (!p_->external_program_transaction_ || p_->external_step_transaction_committed_ ||
      p_->external_program_transaction_->phase() !=
          runtime::program::ProgramTransactionPhase::kCandidate)
    throw std::logic_error(
        "System Program persistent restore publication requires the restart candidate writer");
  std::exception_ptr validation_error;
  try {
    prepared.validate_publication(p_->program_.step_install_generation_);
  } catch (...) {
    validation_error = std::current_exception();
  }
  if (all_reduce_max(validation_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && validation_error)
      std::rethrow_exception(validation_error);
    throw std::runtime_error(
        "System Program persistent restore publication failed collective validation");
  }
  prepared.publish_validated_into(p_->program_.persistent_values());
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
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  // enumeration lives in the extracted Program subsystem (ADC-594)
  return p_->program_.hist_.names();
}
template <int Dim>
int System<Dim>::history_depth(const std::string& name) const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  auto it = p_->program_.hist_.depth.find(name);
  if (it == p_->program_.hist_.depth.end())
    throw std::runtime_error("System::history_depth: unknown history '" + name + "'");
  return it->second;
}
template <int Dim>
int System<Dim>::history_ncomp(const std::string& name) const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  auto it = p_->program_.hist_.histories.find(name);
  if (it == p_->program_.hist_.histories.end())
    throw std::runtime_error("System::history_ncomp: unknown history '" + name + "'");
  return it->second[0].ncomp();
}
template <int Dim>
std::vector<double> System<Dim>::history_global(const std::string& name, int slot) const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
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
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  auto it = p_->program_.hist_.initialized.find(name);
  if (it == p_->program_.hist_.initialized.end())
    throw std::runtime_error("System::history_initialized: unknown history '" + name + "'");
  return it->second;
}
template <int Dim>
int System<Dim>::history_fill_count(const std::string& name) const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
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
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
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
        p_->program_.replay_step(static_cast<double>(dts[static_cast<std::size_t>(j + 1)]),
                                 "System::rebuild_history_slots");
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
// mismatch, then prepare the generated v5 candidate against the exact-ranked host services.  The
// committed ProgramRuntimeState owns the private image for exactly as long as its callbacks exist.
template <int Dim>
POPS_EXPORT void System<Dim>::install_program(const std::string& so_path) {
  require_assembling(p_->lifecycle_,
                     "install_program");  // frozen once pops.bind completes (ADC-592)
  if (p_->program_.step_install_generation_ == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("System::install_program: Program generation overflow");
  const std::uint64_t preparation_generation = p_->program_.step_install_generation_ + 1;
#if defined(_WIN32)
  // Windows: the generated .dll links against _pops.lib at compile time; no global promotion needed.
  pops::dynlib::UniqueHandle image(pops::dynlib::open_private_image(so_path));
  if (!pops::dynlib::valid(image.get())) {
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
  pops::dynlib::UniqueHandle image(pops::dynlib::open_private_image(so_path));
  if (!pops::dynlib::valid(image.get())) {
    throw std::runtime_error(
        "System::install_program: dlopen('" + so_path + "'): " + pops::dynlib::last_error() +
        " (the pops::System seam accessors must be exported and the host module promoted "
        "globally; cf. POPS_EXPORT)");
  }
#endif
  // Build a typed, retained preparation image before entering the private DSO.  It is the only
  // object candidates can use to construct execution services; the raw service registry never
  // grants a generated artifact a System facade during its prelude.
  auto preparation_image =
      pops::runtime::program::make_program_preparation_image<Dim>(this, preparation_generation);
  auto preparation_host = program_host_descriptor();
  pops::runtime::program::bind_program_preparation_image(preparation_host, preparation_image);
  // Exactly one Program symbol is resolved by inspect_program_installation.  It copies every
  // descriptor view while the private image is resident; all checks below consume that host image.
  auto inspected =
      pops::runtime::program::inspect_program_installation(std::move(image), preparation_host);
  inspected.set_preparation_image(preparation_image);
  const auto& candidate_metadata = inspected.metadata();
  const auto& candidate_tables = inspected.tables();
  const std::string loader_key = candidate_metadata.abi_key;
  const std::string module_key = pops::abi_key();
  if (loader_key != module_key) {
    throw std::runtime_error(
        "System::install_program: compiled program ABI mismatch: expected '" + module_key +
        "', got '" + loader_key +
        "'. Recompile the problem module with the SAME compiler, C++ standard and "
        "pops headers as the _pops module.");
  }
  pops::verify_route_manifest(candidate_metadata.route_manifest, "install_program");
  std::vector<pops::runtime::program::ProgramOperatorAuthority> operator_authorities;
  std::vector<pops::runtime::program::ProgramHistoryReplayAuthority> history_replay_authorities;
  try {
    operator_authorities =
        pops::runtime::program::read_program_operator_authorities(candidate_tables);
    history_replay_authorities =
        pops::runtime::program::read_program_history_replay_authorities(candidate_tables);
  } catch (...) {
    throw;
  }
  // Mandatory install-time requirement validation. The complete owner-qualified metadata table is
  // authenticated before installation on every platform; no pre-metadata artifact can bypass it.
  try {
    const auto meta = pops::runtime::program::read_module_metadata(candidate_tables);
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
    throw;
  }
  const bool program_has_dt_bound = inspected.candidate().dt_bound != nullptr;
  const std::string installed_hash = candidate_metadata.artifact_identity;
  const bool state_free_program = candidate_tables.blocks.empty();
  if (state_free_program &&
      (!candidate_tables.parameters.empty() || !candidate_tables.operator_authorities.empty() ||
       !candidate_tables.history_authorities.empty() || !candidate_tables.checkpoint_shape.empty()))
    throw std::runtime_error(
        "System::install_program: state-free Program declares block-owned authority before "
        "candidate_prepare");
  // Bind all compact symbolic resource identities before the DSO gets its descriptor.  A
  // runtime-sized plan deliberately has no byte authority yet: prepare_* below records exact
  // local MultiFab prototypes and the host seals one collective plan before publication.  The
  // candidate therefore still has dense slots (never value ids), but cannot fabricate an extent.
  std::vector<int> preparation_block_map;
  {
    const std::vector<std::string> sys_names = block_names();
    const int count = static_cast<int>(candidate_tables.blocks.size());
    if (state_free_program && !sys_names.empty())
      throw std::runtime_error(
          "System::install_program: a state-free Program has an empty block identity table, but "
          "this System has accepted blocks; positional Program-to-System binding is not supported");
    preparation_block_map.assign(static_cast<std::size_t>(count), -1);
    for (int program_block = 0; program_block < count; ++program_block) {
      const auto& want = candidate_tables.blocks.at(static_cast<std::size_t>(program_block)).name;
      const auto found = std::find(sys_names.begin(), sys_names.end(), want);
      if (found == sys_names.end())
        throw std::runtime_error("Program requires block instance '" + want +
                                 "', but simulation did not instantiate it");
      preparation_block_map[static_cast<std::size_t>(program_block)] =
          static_cast<int>(std::distance(sys_names.begin(), found));
    }
  }
  pops::runtime::program::bind_staged_uniform_program_resource_declaration<Dim>(
      preparation_image, candidate_tables.resource_declarations(), preparation_block_map);

  using preparation_boundary_registry =
      runtime::program::ArtifactFieldBoundaryAuthorityRegistry<Dim>;
  preparation_boundary_registry preparation_boundary_baseline;
  {
    auto capture_authorities = [](const auto& plans) {
      preparation_boundary_registry result;
      for (const auto& [slot, plan] : plans)
        result.emplace(slot,
                       runtime::program::ArtifactFieldBoundaryAuthority<Dim>{
                           plan.boundary_kernel, plan.boundary_point, plan.boundary_parameters});
      return result;
    };
    const auto current = capture_authorities(p_->field_plans_);
    if (p_->program_.artifact_field_boundary_baseline_) {
      preparation_boundary_baseline = *p_->program_.artifact_field_boundary_baseline_;
      if (preparation_boundary_baseline.size() != current.size())
        throw std::logic_error(
            "System::install_program: field-plan registry changed after an artifact boundary "
            "baseline was established; create a fresh runtime");
    } else {
      preparation_boundary_baseline = current;
    }
  }
  pops::runtime::program::seed_staged_uniform_field_boundaries<Dim>(preparation_image,
                                                                    preparation_boundary_baseline);

  // Run the candidate prelude against the detached preparation image before a live Program,
  // boundary, persistent carrier, or auxiliary registry is changed.  Provider-consumer plans are
  // copied into a complete registry image and authenticated collectively; publication below is an
  // allocation-free swap and therefore cannot expose a rank-local half-installation.
  using auxiliary_registry = decltype(p_->auxiliary_registry_);
  using staged_history_request =
      typename runtime::program::ProgramExecutionPreparationImage<Dim>::HistoryRequest;
  std::optional<auxiliary_registry> candidate_auxiliary_registry;
  std::vector<staged_history_request> staged_histories;
  std::optional<preparation_boundary_registry> staged_field_boundaries;
  std::vector<typename runtime::program::ProgramExecutionPreparationImage<Dim>::CacheRequest>
      staged_cache_requests;
  std::optional<runtime::program::ProgramRuntimeState<Dim>> staged_transaction_state;
  std::string staged_transaction_authority_contract;
  bool has_staged_auxiliary_registry = false;
  std::exception_ptr detached_prepare_error;
  try {
    inspected.prepare(preparation_host);
    // The DSO has finished declaring its local clock image. Seal that host-owned provider before
    // reading any other staged carrier; no System state is touched by this transition.
    pops::runtime::program::seal_staged_uniform_program_execution_services<Dim>(preparation_image);
    auto staged =
        pops::runtime::program::take_staged_auxiliary_consumer_plans<Dim>(preparation_image);
    staged_histories = pops::runtime::program::take_staged_histories<Dim>(preparation_image);
    if (state_free_program && !staged_histories.empty())
      throw std::logic_error("System::install_program: state-free Program staged histories");
    staged_field_boundaries =
        pops::runtime::program::take_staged_uniform_field_boundaries<Dim>(preparation_image);
    staged_cache_requests =
        pops::runtime::program::take_staged_uniform_cache_requests<Dim>(preparation_image);
    staged_transaction_authority_contract =
        pops::runtime::program::staged_program_transaction_authority_contract<Dim>(
            preparation_image);
    staged_transaction_state.emplace(
        pops::runtime::program::take_staged_uniform_transaction_authority_state<Dim>(
            preparation_image));
    if (!staged.empty()) {
      candidate_auxiliary_registry.emplace(p_->auxiliary_registry_);
      for (auto& plan : staged)
        candidate_auxiliary_registry->add_consumer_plan(std::move(plan));
      has_staged_auxiliary_registry = true;
    }
  } catch (...) {
    detached_prepare_error = std::current_exception();
  }
  if (all_reduce_max(detached_prepare_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && detached_prepare_error)
      std::rethrow_exception(detached_prepare_error);
    throw std::runtime_error(
        "System::install_program: Program detached preparation failed collectively");
  }
  if (all_reduce_min(has_staged_auxiliary_registry ? 1L : 0L) !=
      all_reduce_max(has_staged_auxiliary_registry ? 1L : 0L))
    throw std::runtime_error(
        "System::install_program: Program provider-plan presence differs between MPI ranks");
  if (has_staged_auxiliary_registry &&
      !all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-program-provider-plans", candidate_auxiliary_registry->collective_contract()}}))
    throw std::runtime_error(
        "System::install_program: staged Program provider plans differ between MPI ranks");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-program-transaction-authorities",
            staged_transaction_authority_contract}}))
    throw std::runtime_error(
        "System::install_program: staged transaction authorities differ between MPI ranks");

  // Runtime-sized resources are intentionally materialized only after the DSO has completed its
  // detached prelude.  First authenticate the exact finite family/key/shape set, then reduce only
  // the rank-local allocated-cell footprint.  A rank with no local fabs contributes cells==0;
  // using max (rather than a global-cell sum) preserves the exact worst-rank memory ceiling.
  using resource_prototype = runtime::program::ProgramInstallationTables::ResourcePrototype;
  std::vector<resource_prototype> prepared_resource_prototypes;
  std::exception_ptr resource_materialization_error;
  try {
    prepared_resource_prototypes =
        pops::runtime::program::take_staged_uniform_resource_prototypes<Dim>(preparation_image);
    std::sort(prepared_resource_prototypes.begin(), prepared_resource_prototypes.end(),
              [](const resource_prototype& left, const resource_prototype& right) {
                return std::tie(left.kind, left.slot, left.subslot) <
                       std::tie(right.kind, right.slot, right.subslot);
              });
    for (std::size_t index = 1; index < prepared_resource_prototypes.size(); ++index) {
      const auto& previous = prepared_resource_prototypes[index - 1];
      const auto& current = prepared_resource_prototypes[index];
      if (std::tie(previous.kind, previous.slot, previous.subslot) ==
          std::tie(current.kind, current.slot, current.subslot))
        throw std::logic_error(
            "System::install_program: detached prepare emitted a duplicate resource prototype");
    }
  } catch (...) {
    resource_materialization_error = std::current_exception();
  }
  if (all_reduce_max(resource_materialization_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && resource_materialization_error)
      std::rethrow_exception(resource_materialization_error);
    throw std::runtime_error(
        "System::install_program: Program resource prototype capture failed collectively");
  }

  std::string resource_family_contract;
  std::exception_ptr resource_family_contract_error;
  try {
    ExactContractBuilder contract;
    contract.text("pops.system.detached-uniform-resource-families")
        .scalar(static_cast<std::uint32_t>(Dim))
        .scalar(preparation_generation)
        .scalar(static_cast<std::uint64_t>(prepared_resource_prototypes.size()));
    for (const auto& prototype : prepared_resource_prototypes) {
      contract.scalar(static_cast<std::uint8_t>(prototype.kind))
          .scalar(prototype.slot)
          .scalar(prototype.subslot)
          .scalar(prototype.layout.itemsize)
          .scalar(prototype.layout.components)
          .scalar(prototype.layout.ghosts);
    }
    resource_family_contract = std::move(contract).release();
  } catch (...) {
    resource_family_contract_error = std::current_exception();
  }
  if (all_reduce_max(resource_family_contract_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && resource_family_contract_error)
      std::rethrow_exception(resource_family_contract_error);
    throw std::runtime_error(
        "System::install_program: Program resource-family contract preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-detached-uniform-resource-families", resource_family_contract}})) {
    throw std::runtime_error(
        "System::install_program: detached Program resource families differ between MPI ranks");
  }
  for (auto& prototype : prepared_resource_prototypes) {
    prototype.layout.cells = all_reduce_max(prototype.layout.cells);
    if (prototype.layout.cells == 0)
      throw std::runtime_error(
          "System::install_program: runtime-sized Program resource has no local allocation on any rank");
    if (prototype.layout.itemsize == 0 || prototype.layout.components == 0 ||
        prototype.layout.cells > std::numeric_limits<std::uint64_t>::max() /
                                     prototype.layout.itemsize ||
        prototype.layout.cells * prototype.layout.itemsize >
            std::numeric_limits<std::uint64_t>::max() / prototype.layout.components)
      throw std::overflow_error(
          "System::install_program: runtime-sized Program resource byte size overflows uint64");
    const std::uint64_t exact_bytes =
        prototype.layout.cells * prototype.layout.itemsize * prototype.layout.components;
    prototype.layout.bytes = exact_bytes;
    prototype.layout.maximum_bytes = exact_bytes;
  }

  std::optional<pops::runtime::program::PreparedProgramInstallation> sealed_installation;
  resource_materialization_error = nullptr;
  try {
    sealed_installation.emplace(std::move(inspected));
    if (!sealed_installation->resource_plan_sealed())
      sealed_installation->seal_resource_plan(prepared_resource_prototypes);
  } catch (...) {
    resource_materialization_error = std::current_exception();
  }
  if (all_reduce_max(resource_materialization_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && resource_materialization_error)
      std::rethrow_exception(resource_materialization_error);
    throw std::runtime_error(
        "System::install_program: Program resource-plan materialization failed collectively");
  }
  const auto& sealed_tables = sealed_installation->tables();
  if (sealed_tables.blocks.empty() != state_free_program)
    throw std::logic_error(
        "System::install_program: block identity table changed after descriptor authentication");

  // NAME-based block binding (Spec 3 criterion 23, ADC-457). A compiled Program carries its block
  // identities in the v5 descriptor; the System numbers its blocks
  // in add order (block_names). They need NOT agree -- bind by NAME, not add-order. Read the .so's
  // block names, map each Program block index to the System block of that name, and store the
  // program-index -> system-index map (read by ProgramExecutionServices to resolve every ctx.state / rhs_into /
  // commit). A Program block whose name has no instantiated System block fails loud with the spec
  // message. The table is REQUIRED whenever the Program owns block state: a state-free Program
  // instead carries the authenticated empty table accepted above. The historical positional
  // convention is no longer a binding contract.
  // Built BEFORE install() so the step closure (which captures a ProgramExecutionServices) sees the map on its
  // first run.
  std::vector<int> program_block_map;
  {
    const std::vector<std::string> sys_names = block_names();
    const int n = static_cast<int>(sealed_tables.blocks.size());
    if (state_free_program && !sys_names.empty())
      throw std::runtime_error(
          "System::install_program: a state-free Program has an empty block identity table, but "
          "this System has accepted blocks; positional Program-to-System binding is not supported");
    program_block_map.assign(static_cast<std::size_t>(n), -1);
    for (int p = 0; p < n; ++p) {
      const std::string& want = sealed_tables.blocks.at(static_cast<std::size_t>(p)).name;
      int found = -1;
      for (std::size_t s = 0; s < sys_names.size(); ++s)
        if (sys_names[s] == want) {
          found = static_cast<int>(s);
          break;
        }
      if (found < 0) {
        throw std::runtime_error("Program requires block instance '" + want +
                                 "', but simulation did not instantiate it");
      }
      program_block_map[static_cast<std::size_t>(p)] = found;
    }
  }
  // RUNTIME PARAMETERS (ADC-510, Spec 5 C5). A Program whose physics reads dsl.Param(..., kind="runtime")
  // carries a v5 parameter table: per flat parameter, its PROGRAM block index, its stable index
  // WITHIN that block (sorted-name order, matching the lowered params.get(index)) and its declaration
  // default. Group the defaults per block (in index order) and seed each block's RuntimeParams to those
  // defaults, so an install WITHOUT a runtime set behaves as with a const param. A later Python params=
  // route overwrites the supplied values via set_program_params. A Program with no runtime param (the
  // count symbol absent or 0) seeds nothing -> the param store stays empty (program_params returns
  // count 0, the lowered kernels read no param). Built BEFORE install() so the step closure (which
  // captures a ProgramExecutionServices) reads the seeded value on its first run.
  std::map<int, std::vector<double>> program_param_defaults;
  {
    for (const auto& parameter : sealed_tables.parameters) {
      const int blk = parameter.block;
      const int idx = parameter.index;
      std::vector<double>& d = program_param_defaults[blk];
      if (static_cast<int>(d.size()) <= idx)
        d.resize(static_cast<std::size_t>(idx) + 1, 0.0);
      d[static_cast<std::size_t>(idx)] = parameter.default_value;
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
  field_plan_registry candidate_field_plans;
  std::string candidate_boundary_contract;
  std::exception_ptr boundary_preparation_error;

  // Baseline preparation and stage allocation finish collectively before the DSO entry is invoked.
  // Otherwise one rank rejecting a changed registry could skip the ProgramExecutionServices communicator
  // construction while a peer entered it.
  try {
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

  } catch (...) {
    boundary_preparation_error = std::current_exception();
  }
  if (all_reduce_max(boundary_preparation_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && boundary_preparation_error)
      std::rethrow_exception(boundary_preparation_error);
    throw std::runtime_error(
        "System::install_program: field-boundary baseline preparation failed collectively");
  }

  // Boundary routes are part of the candidate table.  The current generated routes are pure
  // metadata; actual kernel authorities are staged by candidate preparation, never a second DSO
  // entry point.

  boundary_preparation_error = nullptr;
  try {
    candidate_field_plans = p_->field_plans_;
    if (!staged_field_boundaries)
      throw std::logic_error(
          "System::install_program: detached field-boundary stage was not returned by prepare");
    const auto& staged = *staged_field_boundaries;
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
    }
  } catch (...) {
    boundary_preparation_error = std::current_exception();
  }
  if (all_reduce_max(boundary_preparation_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && boundary_preparation_error)
      std::rethrow_exception(boundary_preparation_error);
    throw std::runtime_error(
        "System::install_program: field-boundary artifact preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-artifact-field-boundary-registry", candidate_boundary_contract}})) {
    throw std::runtime_error(
        "System::install_program: generated field-boundary authorities differ between MPI ranks");
  }
  // The first generated step must never be the field-plan consensus boundary.  Authenticate the
  // complete candidate registry (not merely its staged boundary overlay) while refusal remains
  // harmless, then publish its already-verified flag with the no-throw swap below.
  std::string candidate_field_plan_contract;
  boundary_preparation_error = nullptr;
  try {
    ExactContractBuilder contract;
    contract.text("pops.system.prepared-field-plan-registry")
        .scalar(std::uint32_t{1})
        .scalar(static_cast<std::uint64_t>(candidate_field_plans.size()));
    for (const auto& [slot, plan] : candidate_field_plans)
      contract.text(slot).bytes(Impl::exact_field_plan_contract(plan));
    candidate_field_plan_contract = std::move(contract).release();
  } catch (...) {
    boundary_preparation_error = std::current_exception();
  }
  if (all_reduce_max(boundary_preparation_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && boundary_preparation_error)
      std::rethrow_exception(boundary_preparation_error);
    throw std::runtime_error(
        "System::install_program: candidate field-plan consensus preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-prepared-field-plan-registry", candidate_field_plan_contract}})) {
    throw std::runtime_error(
        "System::install_program: candidate field-plan registry differs between MPI ranks");
  }

  // The DSO retained only the detached image while its callback ran. Its execution-services
  // adapter becomes eligible to borrow System after every collective candidate contract above has
  // succeeded; this changes no accepted value and keeps a failed prepare completely facade-cold.
  pops::runtime::program::activate_staged_uniform_program_execution_services<Dim>(
      preparation_image);

  // Build the complete Program-owned publication image while refusal is still harmless.  Histories,
  // cache images, block parameters and DSO closures all live in this disconnected carrier; there is
  // no live ProgramRuntimeState rollback journal because nothing accepted has changed yet.
  using publication_type =
      typename runtime::program::ProgramRuntimeState<Dim>::PreparedArtifactPublication;
  std::optional<publication_type> prepared_publication;
  std::exception_ptr publication_prepare_error;
  try {
    runtime::program::HistoryManager<Dim> prepared_histories;
    for (const auto& history : staged_histories) {
      if (history.lag < 1 || history.name.empty())
        throw std::invalid_argument("System::install_program: staged history is incomplete");
      const int owner =
          history.program_owner < 0
              ? -1
              : preparation_block_map.at(static_cast<std::size_t>(history.program_owner));
      const int components = history.components < 0
                                 ? program_block_state_(owner < 0 ? 0 : owner).ncomp()
                                 : history.components;
      if (components < 1)
        throw std::invalid_argument("System::install_program: staged history has no components");
      const int depth = history.lag + 1;
      Extent<Dim> ghosts{};
      for (int axis = 0; axis < Dim; ++axis)
        ghosts[axis] = 1;
      std::vector<MultiFab<Dim>> ring;
      ring.reserve(static_cast<std::size_t>(depth));
      for (int slot = 0; slot < depth; ++slot)
        ring.emplace_back(p_->ba, p_->dm, p_->local_rank, components, ghosts);
      const auto [found, inserted] =
          prepared_histories.histories.emplace(history.name, std::move(ring));
      if (!inserted)
        throw std::invalid_argument("System::install_program: duplicate staged history identity");
      prepared_histories.depth[history.name] = depth;
      prepared_histories.initialized[history.name] = false;
      prepared_histories.fill_count[history.name] = 0;
      prepared_histories.store_pending[history.name] = false;
      prepared_histories.owner[history.name] = owner;
      prepared_histories.slot_dt[history.name] =
          std::vector<Real>(static_cast<std::size_t>(depth), Real(0));
      if (owner >= 0) {
        if (history.state_identity.empty() || history.space_identity.empty() ||
            history.clock_identity.empty() || history.interpolation_identity.empty())
          throw std::invalid_argument(
              "System::install_program: qualified staged history is incomplete");
        prepared_histories.state_identity[history.name] = history.state_identity;
        prepared_histories.space_identity[history.name] = history.space_identity;
        prepared_histories.clock_identity[history.name] = history.clock_identity;
        prepared_histories.interpolation_identity[history.name] = history.interpolation_identity;
      }
    }

    runtime::program::ProgramRuntimeState<Dim> prepared_execution_state;
    std::map<int, RuntimeParams> prepared_params;
    for (const auto& [block, defaults] : program_param_defaults) {
      prepared_execution_state.seed_params(block, defaults);
      prepared_params.emplace(block, prepared_execution_state.params(block));
    }
    prepared_publication.emplace(
        publication_type::prepare(std::move(*sealed_installation), preparation_generation));
    for (const auto& request : staged_cache_requests)
      prepared_publication->prime_cache_slot(request.slot, request.prototype);
    prepared_publication->set_resolved_authority(
        installed_hash, operator_authorities, history_replay_authorities,
        runtime::program::read_program_checkpoint_metadata(prepared_publication->tables()),
        program_block_map, std::move(prepared_params), state_free_program);
    if (!staged_transaction_state)
      throw std::logic_error(
          "System::install_program: staged transaction-authority image is absent");
    prepared_publication->adopt_prepared_transaction_authorities(*staged_transaction_state);
    prepared_publication->adopt_prepared_histories(std::move(prepared_histories));
    prepared_publication->set_field_boundary_baseline(
        std::optional<boundary_registry>{std::move(static_boundary_baseline)});
  } catch (...) {
    publication_prepare_error = std::current_exception();
  }
  if (all_reduce_max(publication_prepare_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && publication_prepare_error)
      std::rethrow_exception(publication_prepare_error);
    throw std::runtime_error(
        "System::install_program: Program publication image preparation failed collectively");
  }
  std::string publication_contract;
  std::exception_ptr publication_contract_error;
  try {
    ExactContractBuilder contract;
    const auto& final_resource_plan = prepared_publication->resource_plan();
    const auto& final_metadata = prepared_publication->metadata();
    const auto& final_tables = prepared_publication->tables();
    contract.text("pops.system.prepared-uniform-program-publication")
        .scalar(static_cast<std::uint32_t>(Dim))
        .scalar(preparation_generation)
        .text(final_metadata.artifact_identity)
        .text(final_metadata.abi_key)
        .text(final_metadata.route_manifest)
        .text(final_metadata.boundary_manifest)
        .text(final_metadata.persistent_resource_manifest)
        .text(final_metadata.checkpoint_identity)
        .text(final_metadata.program_name)
        .bytes(staged_transaction_authority_contract)
        .text(final_resource_plan.schema())
        .text(final_resource_plan.digest())
        .scalar(final_resource_plan.maximum_bytes())
        .text(final_tables.canonical_resource_digest_payload(final_resource_plan.maximum_bytes()))
        .scalar(static_cast<std::uint64_t>(staged_histories.size()))
        .scalar(static_cast<std::uint64_t>(staged_cache_requests.size()));
    contract.scalar(static_cast<std::uint64_t>(final_tables.blocks.size()));
    for (const auto& block : final_tables.blocks)
      contract.text(block.name);
    contract.scalar(static_cast<std::uint64_t>(final_tables.parameters.size()));
    for (const auto& parameter : final_tables.parameters)
      contract.scalar(parameter.block)
          .scalar(parameter.index)
          .scalar(std::bit_cast<std::uint64_t>(parameter.default_value))
          .text(parameter.name);
    contract.scalar(static_cast<std::uint64_t>(final_tables.operator_authorities.size()));
    for (const auto& authority : final_tables.operator_authorities)
      for (const auto word : authority.words)
        contract.scalar(word);
    contract.scalar(static_cast<std::uint64_t>(final_tables.history_authorities.size()));
    for (const auto& authority : final_tables.history_authorities)
      contract.text(authority.identity).scalar(authority.depth);
    contract.scalar(static_cast<std::uint64_t>(final_tables.checkpoint_shape.size()));
    for (const auto& checkpoint : final_tables.checkpoint_shape)
      contract.text(checkpoint.identity)
          .text(checkpoint.owner)
          .text(checkpoint.space)
          .text(checkpoint.clock)
          .text(checkpoint.transfer)
          .scalar(checkpoint.block)
          .scalar(checkpoint.components)
          .scalar(checkpoint.retained_images);
    contract.scalar(static_cast<std::uint64_t>(final_tables.flux_budgets.size()));
    for (const auto& budget : final_tables.flux_budgets)
      contract.scalar(budget.rhs_basis_bound)
          .scalar(budget.coefficient_term_bound)
          .scalar(budget.interface_application_bound)
          .scalar(budget.interface_identity_character_bound);
    const auto append_routes = [&contract](const auto& routes) {
      contract.scalar(static_cast<std::uint64_t>(routes.size()));
      for (const auto& route : routes)
        contract.text(route.identity).text(route.kind).scalar(route.capability_bits);
    };
    append_routes(final_tables.boundary_routes);
    append_routes(final_tables.provider_routes);
    const auto append_modules = [&contract](const auto& modules) {
      contract.scalar(static_cast<std::uint64_t>(modules.size()));
      for (const auto& module : modules)
        contract.text(module.identity)
            .text(module.kind)
            .text(module.signature)
            .text(module.requirements)
            .text(module.owner);
    };
    append_modules(final_tables.module_operators);
    append_modules(final_tables.module_state_spaces);
    append_modules(final_tables.module_field_spaces);
    contract.scalar(static_cast<std::uint64_t>(program_block_map.size()));
    for (const int block : program_block_map)
      contract.scalar(block);
    contract.scalar(static_cast<std::uint64_t>(program_param_defaults.size()));
    for (const auto& [block, defaults] : program_param_defaults) {
      contract.scalar(block).scalar(static_cast<std::uint64_t>(defaults.size()));
      for (const double value : defaults)
        contract.scalar(std::bit_cast<std::uint64_t>(value));
    }
    for (const auto& history : staged_histories)
      contract.text(history.name)
          .scalar(history.lag)
          .scalar(history.components)
          .scalar(history.program_owner)
          .text(history.state_identity)
          .text(history.space_identity)
          .text(history.clock_identity)
          .text(history.interpolation_identity);
    for (const auto& request : staged_cache_requests)
      contract.scalar(request.slot)
          .scalar(request.program_block)
          .scalar(static_cast<std::uint64_t>(request.prototype.ncomp()))
          .scalar(static_cast<std::uint64_t>(request.prototype.ghosts()[0]))
          .text(final_resource_plan.entry(request.slot).identity);
    publication_contract = std::move(contract).release();
  } catch (...) {
    publication_contract_error = std::current_exception();
  }
  if (all_reduce_max(publication_contract_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && publication_contract_error)
      std::rethrow_exception(publication_contract_error);
    throw std::runtime_error(
        "System::install_program: Program publication contract preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-prepared-uniform-program-publication", publication_contract}})) {
    throw std::runtime_error(
        "System::install_program: prepared Program publication differs between MPI ranks");
  }

  // All collective checks and allocations have completed.  This is deliberately a closed
  // no-throw publication: readers either see the old complete authority or the new one, never an
  // install-time boundary/provider/history fragment.  ProgramRuntimeState publishes its owner last.
  [[maybe_unused]] auto accepted_write = p_->acquire_accepted_write_lease();
  if (has_staged_auxiliary_registry) {
    p_->auxiliary_registry_.swap_complete(*candidate_auxiliary_registry);
    p_->auxiliary_registry_consensus_verified_ = true;
  }
  p_->field_plans_.swap(candidate_field_plans);
  p_->field_plan_consensus_verified_ = true;
  for (auto& [slot, kernel] : materialized_candidate) {
    const auto field = p_->named_fields_.find(slot);
    if (field == p_->named_fields_.end())
      std::terminate();
    field->second->replace_boundary_kernel(std::move(kernel));
  }
  p_->program_.publish_prepared_artifact(std::move(*prepared_publication));
  (void)program_has_dt_bound;
  // The committed ProgramRuntimeState keeps the DSO resident until its candidate and closures are gone.
}
// Scheduler-cache checkpoint/restart seam (ADC-458, Spec 3 section 30): the System owns the cache, so
// the facade (sim.checkpoint / sim.restart) gathers and restores it DIRECTLY -- reusing the SAME global
// exact-ranked global gather/write machinery as the block state and the
// history rings, so the round-trip is MPI-safe and bit-identical under np>1. Mirrors the history seam.
template <int Dim>
std::vector<std::size_t> System<Dim>::program_cache_slots() const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->program_.cache_.checkpoint_slot_indices();
}
template <int Dim>
std::string System<Dim>::program_cache_plan_schema() const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return std::string(p_->program_.cache_.plan_schema());
}
template <int Dim>
std::string System<Dim>::program_cache_plan_digest() const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return std::string(p_->program_.cache_.plan_digest());
}
template <int Dim>
bool System<Dim>::program_cache_valid(std::size_t slot) const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->program_.cache_.valid(slot);
}
template <int Dim>
bool System<Dim>::program_cache_cold(std::size_t slot) const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->program_.cache_.cold(slot);
}
template <int Dim>
std::string System<Dim>::program_cache_name(std::size_t slot) const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->program_.cache_.name_of(slot);
}
template <int Dim>
int System<Dim>::program_cache_last_update_step(std::size_t slot) const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->program_.cache_.last_update_step(slot);
}
template <int Dim>
double System<Dim>::program_cache_accumulated_dt(std::size_t slot) const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return static_cast<double>(p_->program_.cache_.accumulated_dt(slot));
}
template <int Dim>
int System<Dim>::program_cache_ncomp(std::size_t slot) const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  return p_->program_.cache_.ncomp_of(slot);
}
template <int Dim>
int System<Dim>::program_cache_ngrow(std::size_t slot) const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  const Extent<Dim> ghosts = p_->program_.cache_.ghosts_of(slot);
  const int width = ghosts[0];
  for (int axis = 1; axis < Dim; ++axis)
    if (ghosts[axis] != width)
      throw std::runtime_error(
          "System::program_cache_ngrow: checkpoint schema requires one uniform ND ghost width");
  return width;
}
template <int Dim>
std::vector<double> System<Dim>::program_cache_global(std::size_t slot) const {
  [[maybe_unused]] auto accepted_read = p_->acquire_accepted_read_lease();
  // The cache value is co-distributed with block 0 and uses the same authenticated component-major
  // collective gather as state_global and history_global.
  const std::string request = std::to_string(slot);
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("program-cache-global"), std::string_view(request)}}))
    throw std::invalid_argument("System::program_cache_global request differs between MPI ranks");
  if (all_reduce_max(p_->program_.cache_.valid(slot) ? 0L : 1L) != 0)
    throw std::out_of_range(
        "System::program_cache_global slot is invalid on at least one MPI rank");
  const MultiFab<Dim>& value = p_->program_.cache_.value_of(slot);
  return runtime::system::marshaling::gather_global(value, p_->dom, value.ncomp());
}
template <int Dim>
void System<Dim>::restore_program_cache(std::size_t slot, int ncomp, int ngrow,
                                        int last_update_step, double accumulated_dt,
                                        const std::string& name,
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
  const std::string request = std::to_string(slot) + ":" + std::to_string(ncomp) + ":" +
                              std::to_string(ngrow) + ":" + std::to_string(last_update_step) + ":" +
                              std::to_string(std::bit_cast<std::uint64_t>(accumulated_dt));
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view(name), std::string_view(request)}}))
    throw std::invalid_argument("System::restore_program_cache request differs between MPI ranks");
  const long local_metadata_invalid =
      ncomp < 1 || ngrow < 0 || last_update_step < 0 || !std::isfinite(accumulated_dt) ||
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
  p_->program_.cache_.restore_slot(slot, std::move(value), last_update_step,
                                   static_cast<Real>(accumulated_dt), name);
}

template <int Dim>
void System<Dim>::restore_program_cache_pending(std::size_t slot, double accumulated_dt,
                                                const std::string& name) {
  const std::string request =
      std::to_string(slot) + ":" + std::to_string(std::bit_cast<std::uint64_t>(accumulated_dt));
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view(name), std::string_view(request)}}))
    throw std::invalid_argument(
        "System::restore_program_cache_pending request differs between MPI ranks");
  const long invalid =
      !std::isfinite(accumulated_dt) || accumulated_dt < 0.0 ||
              !std::isfinite(static_cast<double>(static_cast<Real>(accumulated_dt)))
          ? 1L
          : 0L;
  if (all_reduce_max(invalid) != 0)
    throw std::invalid_argument("System::restore_program_cache_pending metadata is invalid");
  p_->program_.cache_.restore_pending_slot(slot, static_cast<Real>(accumulated_dt), name);
}

template void System<kNativeDimension>::set_clock(double, int);
template std::vector<std::uint8_t>
System<kNativeDimension>::capture_program_persistent_value_checkpoint() const;
template runtime::program::PreparedProgramPersistentValueRestore
System<kNativeDimension>::prepare_program_persistent_value_restore(
    const std::vector<std::uint8_t>&) const;
template void System<kNativeDimension>::publish_program_persistent_value_restore(
    runtime::program::PreparedProgramPersistentValueRestore&);
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
template std::vector<std::size_t> System<kNativeDimension>::program_cache_slots() const;
template std::string System<kNativeDimension>::program_cache_plan_schema() const;
template std::string System<kNativeDimension>::program_cache_plan_digest() const;
template bool System<kNativeDimension>::program_cache_valid(std::size_t) const;
template bool System<kNativeDimension>::program_cache_cold(std::size_t) const;
template std::string System<kNativeDimension>::program_cache_name(std::size_t) const;
template int System<kNativeDimension>::program_cache_last_update_step(std::size_t) const;
template double System<kNativeDimension>::program_cache_accumulated_dt(std::size_t) const;
template int System<kNativeDimension>::program_cache_ncomp(std::size_t) const;
template int System<kNativeDimension>::program_cache_ngrow(std::size_t) const;
template std::vector<double> System<kNativeDimension>::program_cache_global(std::size_t) const;
template void System<kNativeDimension>::restore_program_cache(std::size_t, int, int, int, double,
                                                              const std::string&,
                                                              const std::vector<double>&);
template void System<kNativeDimension>::restore_program_cache_pending(std::size_t, double,
                                                                      const std::string&);

}  // namespace pops
