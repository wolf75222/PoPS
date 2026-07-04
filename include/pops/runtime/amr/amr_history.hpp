/// @file
/// @brief Multistep HISTORY RINGS on the compiled-Program AMR route (ADC-631). Kept out of
/// amr_runtime.hpp (its line budget): all the logic lives in `struct detail::AmrHistoryOps`, a FRIEND
/// of AmrRuntime whose static methods take the engine by reference (move-safe -- no back-pointer) and
/// reach the four data-only ring members directly. Included at the END of amr_runtime.hpp (the full
/// AmrRuntime class is visible); the AmrProgramContext seam, the regrid remap hook and the AmrSystem
/// facade all call `detail::AmrHistoryOps::<op>(engine, ...)`.
///
/// SEMANTIC (design plan section 2): a ring slot on AMR is a snapshot of the FULL hierarchy state --
/// one MultiFab per level, co-distributed with each level's U. `hist_rings_[name][slot][level]` is
/// block 0's level-`level` state at macro-step lag `slot` (slot 0 = newest). Rotation swaps the whole
/// per-level buffer set atomically (a vector-of-handles swap, O(1), no deep copy). `slot_dt` is ONE
/// scalar per slot (the synchronous AMR driver advances every level with the SAME dt), byte-identical
/// to the Uniform ring's per-slot dt, so the ADC-626 variable-dt replay math is reused unchanged.
///
/// REGRID REMAP (section 3): after any hierarchy change every ring slot's per-level data is remapped
/// onto the NEW (fb, dmap) with the SAME R6/R7 machinery the live U uses (regrid_field_on_layout, the
/// ring slot's own inherited ghost width), so `prev(k)` reads stay layout-consistent with the current
/// U. The coarse slot is stable; finer slots are prolonged/restricted from it exactly like U.
///
/// NATIVE REPLAY (section 5): rebuild_history_slots reconstructs the policy-recomputed slots by
/// re-stepping the installed Program (passed as a closure -- it lives on the AmrSystem facade, not the
/// engine) from the nearest OLDER stored slot. The replay re-executes the ORIGINAL step sequence on the
/// checkpoint hierarchy: the v3 reader REFUSES a replay whose reconstruction window straddles a
/// head-of-step regrid (the anchors are stored REMAPPED, so the pre-regrid fine data an exact re-step
/// needs no longer exists -- _amr_checkpoint_v3._restore_histories_v3), and inside a clean window the
/// original steps saw NO regrid, so freezing the regrid cadence here (regrid_every_ = 0,
/// saved/restored) makes the re-stepped schedule IDENTICAL to the original one. The freeze also
/// hardens a direct rebuild_history_slots call made after set_clock (where the constant macro-step
/// cursor could otherwise re-fire regrid_if_due on every re-step). The live block states, the shared
/// aux and the whole ring store are SAVE/RESTORE bracketed so replay is identity on them (parity
/// System::rebuild_history_slots, ADC-626); the hierarchy is invariant through replay by construction.
/// A single-block AMR Program is reconstructed bit-for-bit; a multi-block Program inherits the Uniform
/// limitation (only ring block 0 is re-seeded).

#pragma once

#include <pops/runtime/amr/amr_runtime.hpp>  // AmrRuntime (befriends detail::AmrHistoryOps)

#include <algorithm>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {
namespace detail {

/// Static operations over the AmrRuntime history-ring members (a friend of AmrRuntime). No state of
/// its own: every method takes the engine by reference, so the engine stays movable and the ring data
/// stays co-located with the per-level U/aux it mirrors.
struct AmrHistoryOps {
  // One per-level buffer set (a MultiFab per level, co-distributed with block 0's per-level U),
  // @p ncomp components, one ghost, zero-initialized -- one ring slot on the shared hierarchy.
  static std::vector<MultiFab> alloc_slot(const AmrRuntime& eng, int ncomp) {
    std::vector<MultiFab> per_level;
    per_level.reserve(static_cast<std::size_t>(eng.nlev_));
    for (int k = 0; k < eng.nlev_; ++k) {
      const MultiFab& U = (*eng.blocks_[0].levels)[static_cast<std::size_t>(k)].U;
      MultiFab slot(U.box_array(), U.dmap(), ncomp, 1);
      slot.set_val(Real(0));
      per_level.push_back(std::move(slot));
    }
    return per_level;
  }

  // --- AmrProgramContext-facing seams (register / read / store / rotate) --------------------------

  static void register_history(AmrRuntime& eng, const std::string& name, int lag) {
    if (lag < 1)
      throw std::runtime_error("AmrRuntime::register_history: lag must be >= 1 (got " +
                               std::to_string(lag) + ") for history '" + name + "'");
    const int want_depth = lag + 1;
    auto it = eng.hist_rings_.find(name);
    if (it != eng.hist_rings_.end()) {
      // Idempotent re-registration: the ring depth is the MAX lag any caller requests. A larger lag
      // grows the ring (append zero-filled deeper slots on the CURRENT layout, all levels); a smaller
      // one is a no-op. The already-stored slots and the current slot [0] are preserved.
      if (want_depth > eng.hist_depth_[name]) {
        const int ncomp = it->second[0][0].ncomp();
        for (int s = eng.hist_depth_[name]; s < want_depth; ++s)
          it->second.push_back(alloc_slot(eng, ncomp));
        eng.hist_depth_[name] = want_depth;
        std::vector<Real>& dts = eng.hist_slot_dt_[name];
        if (static_cast<int>(dts.size()) < want_depth)
          dts.resize(static_cast<std::size_t>(want_depth), Real(0));
      }
      return;
    }
    const int ncomp = eng.blocks_[0].ncomp;
    std::vector<std::vector<MultiFab>> ring;
    ring.reserve(static_cast<std::size_t>(want_depth));
    for (int s = 0; s < want_depth; ++s)
      ring.push_back(alloc_slot(eng, ncomp));
    eng.hist_rings_.emplace(name, std::move(ring));
    eng.hist_depth_[name] = want_depth;
    eng.hist_init_[name] = std::vector<char>(static_cast<std::size_t>(eng.nlev_), 0);
    eng.hist_slot_dt_[name] = std::vector<Real>(static_cast<std::size_t>(want_depth), Real(0));
  }

  static MultiFab& read_history(AmrRuntime& eng, const std::string& name, int lag, int level) {
    auto it = eng.hist_rings_.find(name);
    if (it == eng.hist_rings_.end())
      throw std::runtime_error("AmrRuntime::read_history: unknown history '" + name +
                               "' (register it first)");
    if (lag < 0 || lag >= eng.hist_depth_[name])
      throw std::runtime_error("AmrRuntime::read_history: lag=" + std::to_string(lag) +
                               " out of range for history '" + name + "' (depth " +
                               std::to_string(eng.hist_depth_[name]) + ")");
    if (level < 0 || level >= eng.nlev_)
      throw std::runtime_error("AmrRuntime::read_history: level out of bounds");
    const std::vector<char>& init = eng.hist_init_[name];
    if (level >= static_cast<int>(init.size()) || !init[static_cast<std::size_t>(level)])
      throw std::runtime_error("history '" + name + "' with lag=" + std::to_string(lag) +
                               " was requested but not initialized (level " +
                               std::to_string(level) + ")");
    return it->second[static_cast<std::size_t>(lag)][static_cast<std::size_t>(level)];
  }

  static void store_history(AmrRuntime& eng, const std::string& name, int level,
                            const MultiFab& value, Real dt) {
    auto it = eng.hist_rings_.find(name);
    if (it == eng.hist_rings_.end())
      throw std::runtime_error("AmrRuntime::store_history: unknown history '" + name +
                               "' (register it first)");
    if (level < 0 || level >= eng.nlev_)
      throw std::runtime_error("AmrRuntime::store_history: level out of bounds");
    std::vector<std::vector<MultiFab>>& ring = it->second;
    // Copy value's valid cells into slot [0] of THIS level (ring slot and level state share (ba, dm);
    // lincomb(dst, 1, src, 0, src) is a valid-cell deep copy).
    pops::lincomb(ring[0][static_cast<std::size_t>(level)], Real(1), value, Real(0), value);
    // PER-SLOT dt (ADC-626): scalar per slot (same dt for every level -- the synchronous driver).
    std::vector<Real>& dts = eng.hist_slot_dt_[name];
    if (dts.size() != ring.size())
      dts.assign(ring.size(), Real(0));
    dts[0] = dt;
    std::vector<char>& init = eng.hist_init_[name];
    if (static_cast<int>(init.size()) < eng.nlev_)
      init.assign(static_cast<std::size_t>(eng.nlev_), 0);
    if (!init[static_cast<std::size_t>(level)]) {
      // PER-LEVEL COLD START (first store of this level): broadcast into every deeper slot so a
      // multistep step 0 reads the same value at every lag (degenerating to a one-step method).
      for (std::size_t s = 1; s < ring.size(); ++s) {
        pops::lincomb(ring[s][static_cast<std::size_t>(level)], Real(1), value, Real(0), value);
        dts[s] = dt;
      }
      init[static_cast<std::size_t>(level)] = 1;
    }
  }

  static void rotate_histories(AmrRuntime& eng) {
    // Shift each ring one step at the MACRO-step boundary (the AmrProgramContext guards this to fire
    // once per macro-step, on the last level). O(1) swap of the per-level buffer sets (vector-of-
    // handles swap, no deep copy); slot_dt rotates on the same chain.
    for (auto& [name, ring] : eng.hist_rings_) {
      for (std::size_t s = ring.size(); s-- > 1;)
        std::swap(ring[s], ring[s - 1]);
      auto dt_it = eng.hist_slot_dt_.find(name);
      if (dt_it != eng.hist_slot_dt_.end()) {
        std::vector<Real>& dts = dt_it->second;
        for (std::size_t s = dts.size(); s-- > 1;)
          std::swap(dts[s], dts[s - 1]);
      }
    }
  }

  // --- checkpoint accessors (Uniform seam names; per-level flat concatenated across levels) -------

  static std::vector<std::string> names(const AmrRuntime& eng) {
    std::vector<std::string> out;
    out.reserve(eng.hist_rings_.size());
    for (const auto& [name, ring] : eng.hist_rings_) {
      (void)ring;
      out.push_back(name);
    }
    return out;
  }

  static int depth(const AmrRuntime& eng, const std::string& name) {
    auto it = eng.hist_depth_.find(name);
    if (it == eng.hist_depth_.end())
      throw std::runtime_error("AmrRuntime::history_depth: unknown history '" + name + "'");
    return it->second;
  }

  static int ncomp(const AmrRuntime& eng, const std::string& name) {
    auto it = eng.hist_rings_.find(name);
    if (it == eng.hist_rings_.end())
      throw std::runtime_error("AmrRuntime::history_ncomp: unknown history '" + name + "'");
    return it->second[0][0].ncomp();
  }

  static bool initialized(const AmrRuntime& eng, const std::string& name) {
    auto it = eng.hist_init_.find(name);
    if (it == eng.hist_init_.end())
      throw std::runtime_error("AmrRuntime::history_initialized: unknown history '" + name + "'");
    // Every level initializes in the SAME macro-step; level 0 is a faithful representative.
    return !it->second.empty() && it->second[0] != 0;
  }

  static void set_initialized(AmrRuntime& eng, const std::string& name, bool value) {
    auto it = eng.hist_init_.find(name);
    if (it == eng.hist_init_.end())
      throw std::runtime_error("AmrRuntime::set_history_initialized: unknown history '" + name +
                               "' (restore its slots first)");
    std::fill(it->second.begin(), it->second.end(), value ? char(1) : char(0));
  }

  // FULL history slot @p slot of ring @p name as ONE flat buffer, the per-level slices concatenated
  // (level 0 then 1 ...), each at the v3 convention c*nf*nf+j*nf+i (nf = nx << level; zeros outside
  // the patches at a fine level) -- the SAME layout level_aux_flat uses, hiding the level axis inside
  // the accessor so _system_io_history.py is reused verbatim. @p gather -> np>1 all_reduce_sum.
  static std::vector<double> global(const AmrRuntime& eng, const std::string& name, int slot,
                                    bool gather) {
    auto it = eng.hist_rings_.find(name);
    if (it == eng.hist_rings_.end())
      throw std::runtime_error("AmrRuntime::history_global: unknown history '" + name + "'");
    const std::vector<std::vector<MultiFab>>& ring = it->second;
    if (slot < 0 || slot >= static_cast<int>(ring.size()))
      throw std::runtime_error("AmrRuntime::history_global: slot=" + std::to_string(slot) +
                               " out of range for history '" + name + "'");
    const int nc = ring[static_cast<std::size_t>(slot)][0].ncomp();
    std::vector<double> out;
    device_fence();
    for (int k = 0; k < eng.nlev_; ++k) {
      const MultiFab& S = ring[static_cast<std::size_t>(slot)][static_cast<std::size_t>(k)];
      const std::size_t nf = static_cast<std::size_t>(eng.dom_.nx()) << k;
      std::vector<double> lvl(static_cast<std::size_t>(nc) * nf * nf, 0.0);
      for (int li = 0; li < S.local_size(); ++li) {
        const ConstArray4 a = S.fab(li).const_array();
        const Box2D v = S.box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            for (int c = 0; c < nc; ++c)
              lvl[static_cast<std::size_t>(c) * nf * nf + static_cast<std::size_t>(j) * nf +
                  static_cast<std::size_t>(i)] = a(i, j, c);
      }
      out.insert(out.end(), lvl.begin(), lvl.end());
    }
    if (gather)
      all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
    return out;
  }

  // Restore history slot @p slot of ring @p name from the flat concatenated buffer above. Registers
  // the ring (all levels) if unknown / grows it if @p slot is deeper (mirror of System::restore_
  // history), then scatters each per-level slice into the slot's valid cells (owner-rank).
  static void restore(AmrRuntime& eng, const std::string& name, int slot,
                      const std::vector<double>& flat) {
    if (slot < 0)
      throw std::runtime_error("AmrRuntime::restore_history: slot must be >= 0 for history '" +
                               name + "'");
    auto it = eng.hist_rings_.find(name);
    if (it == eng.hist_rings_.end()) {
      register_history(eng, name, slot >= 1 ? slot : 1);
      it = eng.hist_rings_.find(name);
    }
    std::vector<std::vector<MultiFab>>& ring = it->second;
    if (slot >= static_cast<int>(ring.size())) {
      const int nco = ring[0][0].ncomp();
      for (int s = static_cast<int>(ring.size()); s <= slot; ++s)
        ring.push_back(alloc_slot(eng, nco));
      eng.hist_depth_[name] = static_cast<int>(ring.size());
      std::vector<Real>& dts = eng.hist_slot_dt_[name];
      if (static_cast<int>(dts.size()) < static_cast<int>(ring.size()))
        dts.resize(ring.size(), Real(0));
    }
    const int nc = ring[static_cast<std::size_t>(slot)][0].ncomp();
    device_fence();
    std::size_t off = 0;
    for (int k = 0; k < eng.nlev_; ++k) {
      MultiFab& S = ring[static_cast<std::size_t>(slot)][static_cast<std::size_t>(k)];
      const std::size_t nf = static_cast<std::size_t>(eng.dom_.nx()) << k;
      for (int li = 0; li < S.local_size(); ++li) {
        Array4 a = S.fab(li).array();
        const Box2D v = S.box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            for (int c = 0; c < nc; ++c)
              a(i, j, c) = flat[off + static_cast<std::size_t>(c) * nf * nf +
                                static_cast<std::size_t>(j) * nf + static_cast<std::size_t>(i)];
      }
      off += static_cast<std::size_t>(nc) * nf * nf;
    }
  }

  static double slot_dt(const AmrRuntime& eng, const std::string& name, int slot) {
    auto it = eng.hist_rings_.find(name);
    if (it == eng.hist_rings_.end())
      throw std::runtime_error("AmrRuntime::history_slot_dt: unknown history '" + name + "'");
    if (slot < 0 || slot >= static_cast<int>(it->second.size()))
      throw std::runtime_error("AmrRuntime::history_slot_dt: slot out of range for history '" +
                               name + "'");
    auto dt_it = eng.hist_slot_dt_.find(name);
    if (dt_it == eng.hist_slot_dt_.end() || slot >= static_cast<int>(dt_it->second.size()))
      return 0.0;
    return static_cast<double>(dt_it->second[static_cast<std::size_t>(slot)]);
  }

  static void restore_slot_dt(AmrRuntime& eng, const std::string& name, int slot, double dt) {
    auto it = eng.hist_rings_.find(name);
    if (it == eng.hist_rings_.end())
      throw std::runtime_error("AmrRuntime::restore_history_slot_dt: unknown history '" + name +
                               "' (restore its slots first)");
    if (slot < 0)
      throw std::runtime_error("AmrRuntime::restore_history_slot_dt: slot must be >= 0");
    std::vector<Real>& dts = eng.hist_slot_dt_[name];
    if (slot >= static_cast<int>(dts.size()))
      dts.resize(static_cast<std::size_t>(slot) + 1, Real(0));
    dts[static_cast<std::size_t>(slot)] = static_cast<Real>(dt);
  }

  // --- regrid / rebuild-hierarchy remap hook ------------------------------------------------------

  // Remap every registered ring's fine-level slot onto the NEW (fb, dmap) with the SAME machinery the
  // live U uses (regrid_field_on_layout: parent prolong + old-fine carry-over, the slot's own
  // inherited ghost width). @p prolong=false (rebuild_hierarchy) reallocates the fine slot WITHOUT
  // interpolation (the per-slot restore overwrites every valid cell afterwards); the coarse slot
  // (level pk) is stable. No-op when no ring exists.
  static void remap_rings(AmrRuntime& eng, const BoxArray& fb, const DistributionMapping& dmap,
                          int fk, int pk, bool prolong) {
    for (auto& [name, ring] : eng.hist_rings_) {
      (void)name;
      for (auto& slot : ring) {  // slot = per-level vector<MultiFab>
        MultiFab& fine = slot[static_cast<std::size_t>(fk)];
        const int ngf = fine.n_grow();
        if (prolong)
          fine = regrid_field_on_layout(fb, dmap, slot[static_cast<std::size_t>(pk)], fine, pk, ngf,
                                        eng.replicated_coarse_);
        else
          fine = MultiFab(fb, dmap, fine.ncomp(), ngf);
      }
    }
  }

  // --- native selective-persistence replay (ADC-626 on AMR) ---------------------------------------

  static int rebuild_slots(AmrRuntime& eng, const std::string& name,
                           const std::vector<int>& stored_slots,
                           const std::function<void(double)>& program_step) {
    auto it = eng.hist_rings_.find(name);
    if (it == eng.hist_rings_.end())
      throw std::runtime_error("AmrRuntime::rebuild_history_slots: unknown history '" + name + "'");
    if (!program_step)
      throw std::runtime_error(
          "AmrRuntime::rebuild_history_slots: no compiled Program is installed; the ring cannot be "
          "replayed (install_program before restart, or checkpoint the ring with Dense())");
    std::vector<std::vector<MultiFab>>& ring = it->second;
    const int d = static_cast<int>(ring.size());
    std::vector<int> anchors = stored_slots;
    std::sort(anchors.begin(), anchors.end());
    anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());
    if (anchors.empty() || anchors.back() != d - 1)
      throw std::runtime_error(
          "AmrRuntime::rebuild_history_slots: the oldest slot " + std::to_string(d - 1) +
          " of history '" + name + "' is not stored; the ring is unreconstructable (the persistence "
          "policy must store the oldest slot).");
    if (static_cast<int>(anchors.size()) == d)
      return 0;  // Dense: nothing to recompute.

    // SAVE bracket (extended for AMR): deep-copy every block's per-level U (all levels), the shared
    // aux (all levels), the WHOLE ring store, and FREEZE regrid so the internal re-steps stay on the
    // checkpoint hierarchy (every reconstructed slot lands on it exactly like the stored anchors, so
    // placement by index is layout-consistent). All undone below; only the missing ring slots survive.
    std::vector<std::vector<MultiFab>> saved_states;  // [block][level]
    saved_states.reserve(eng.blocks_.size());
    for (auto& b : eng.blocks_) {
      std::vector<MultiFab> per_level;
      per_level.reserve(b.levels->size());
      for (auto& lvl : *b.levels)
        per_level.push_back(lvl.U);  // deep copy
      saved_states.push_back(std::move(per_level));
    }
    std::vector<MultiFab> saved_aux = eng.aux_;  // deep copy (all levels)
    // The multigrid warm-start iterate (mg_.phi()) is STATEFUL across solve_fields: a re-step reads it
    // as the seed of the next Poisson solve, so leaving it in the replayed state would perturb the
    // post-restart continuation (the solver converges to a slightly different residual). Bracket it too
    // so replay is a true identity on every stateful buffer the macro-step touches.
    MultiFab saved_phi = eng.mg_.phi();  // deep copy of the warm-start iterate
    const std::map<std::string, std::vector<std::vector<MultiFab>>> saved_rings = eng.hist_rings_;
    const std::map<std::string, std::vector<char>> saved_init = eng.hist_init_;
    const std::map<std::string, std::vector<Real>> saved_slot_dt = eng.hist_slot_dt_;
    const int saved_regrid_every = eng.regrid_every_;
    eng.regrid_every_ = 0;  // freeze the hierarchy for the internal replay

    // Per-slot dt each store produced (from the SAVED snapshot -- the replay MUTATES hist_slot_dt_).
    std::vector<Real> dts(static_cast<std::size_t>(d), Real(0));
    auto sd_it = saved_slot_dt.find(name);
    if (sd_it != saved_slot_dt.end())
      for (int k = 0; k < d && k < static_cast<int>(sd_it->second.size()); ++k)
        dts[static_cast<std::size_t>(k)] = sd_it->second[static_cast<std::size_t>(k)];

    // Reconstruct block-0's per-level trajectory: for each gap (older at a LARGER index), seed block 0
    // every level from the older stored slot, step forward, record each intervening slot. Placement BY
    // INDEX (no rotate) sidesteps ADC-538.
    std::vector<std::vector<MultiFab>> reconstructed(static_cast<std::size_t>(d));  // [slot][level]
    for (std::size_t a = 0; a + 1 < anchors.size(); ++a) {
      const int older = anchors[a + 1];
      const int newer = anchors[a];
      for (int k = 0; k < eng.nlev_; ++k)
        pops::lincomb(
            (*eng.blocks_[0].levels)[static_cast<std::size_t>(k)].U, Real(1),
            saved_rings.at(name)[static_cast<std::size_t>(older)][static_cast<std::size_t>(k)],
            Real(0), (*eng.blocks_[0].levels)[static_cast<std::size_t>(k)].U);
      for (int j = older - 1; j >= newer; --j) {
        program_step(static_cast<double>(dts[static_cast<std::size_t>(j)]));
        std::vector<MultiFab> snap;
        snap.reserve(static_cast<std::size_t>(eng.nlev_));
        for (int k = 0; k < eng.nlev_; ++k)
          snap.push_back((*eng.blocks_[0].levels)[static_cast<std::size_t>(k)].U);  // deep copy
        reconstructed[static_cast<std::size_t>(j)] = std::move(snap);
      }
    }

    // RESTORE bracket: undo every replay side effect (block states, aux, ring store, regrid cadence).
    for (std::size_t b = 0; b < eng.blocks_.size(); ++b)
      for (int k = 0; k < static_cast<int>(eng.blocks_[b].levels->size()); ++k)
        (*eng.blocks_[b].levels)[static_cast<std::size_t>(k)].U =
            std::move(saved_states[b][static_cast<std::size_t>(k)]);
    eng.aux_ = std::move(saved_aux);
    eng.mg_.phi() = std::move(saved_phi);  // restore the multigrid warm-start iterate
    eng.hist_rings_ = saved_rings;
    eng.hist_init_ = saved_init;
    eng.hist_slot_dt_ = saved_slot_dt;
    eng.regrid_every_ = saved_regrid_every;

    // Place ONLY the recomputed slots (the anchors keep their restored values), all levels.
    std::vector<std::vector<MultiFab>>& out_ring = eng.hist_rings_.at(name);
    std::vector<char> is_stored(static_cast<std::size_t>(d), 0);
    for (int s : anchors)
      is_stored[static_cast<std::size_t>(s)] = 1;
    int recomputed = 0;
    for (int j = 0; j < d; ++j) {
      if (is_stored[static_cast<std::size_t>(j)])
        continue;
      for (int k = 0; k < eng.nlev_; ++k)
        pops::lincomb(out_ring[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)], Real(1),
                      reconstructed[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)],
                      Real(0), out_ring[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)]);
      ++recomputed;
    }
    return recomputed;
  }
};

}  // namespace detail

// Out-of-line AmrRuntime remap-hook member (declared in amr_runtime.hpp): the INLINE regrid() and
// rebuild_hierarchy call it (a member call needs only the declaration), and it forwards to the now
// complete detail::AmrHistoryOps. Keeps the ring logic in one place while sidestepping the
// incomplete-type call from regrid()'s in-class body.
inline void AmrRuntime::remap_history_rings_(const BoxArray& fb, const DistributionMapping& dmap,
                                             int fk, int pk, bool prolong) {
  detail::AmrHistoryOps::remap_rings(*this, fb, dmap, fk, pk, prolong);
}

}  // namespace pops
