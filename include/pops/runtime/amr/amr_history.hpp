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
/// NATIVE REPLAY (section 5, ADC-631/ADC-635): rebuild_history_slots reconstructs the policy-recomputed
/// slots as ONE continuous forward sweep of the installed Program (passed as a closure -- it lives on
/// the AmrSystem facade, not the engine) from the OLDEST stored slot. Regrid stays ACTIVE during the
/// re-steps: the closure drives the facade cursor to m-1-j for the re-step producing slot j, so the
/// head-of-step ctx.regrid_if_due(ctx.macro_step()) reproduces the ORIGINAL in-window regrid schedule,
/// and each regrid remaps every live ring slot (remap_history_rings_) exactly as the original run did.
/// Because the stored anchors are REMAPPED onto the checkpoint hierarchy, an exact reconstruction must
/// ride the SAME incremental remap chain: the re-stepped program's OWN store/rotate mechanics rebuild
/// the ring in place, so the rebuilt values ride every later in-window regrid forward onto the
/// checkpoint grid (a frozen one-shot re-step could not -- that was the ADC-631 straddle refusal, now
/// lifted). A coherence guard refuses
/// any regrid completed OFF the due schedule derived from (depth, m, regrid_every) -- broken cursor
/// driving or a divergent restart build fails LOUD, never silently wrong (a due step whose regrid
/// no-ops deterministically is legitimate: the original run no-oped there identically). The live block
/// states, the shared aux, the warm start, the ring store and the regrid count are SAVE/RESTORE
/// bracketed so replay is identity on them (parity System::rebuild_history_slots, ADC-626). A
/// single-block AMR Program is reconstructed bit-for-bit; a multi-block Program inherits the Uniform
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

  static void register_history(AmrRuntime& eng, const std::string& name, int lag, int ncomp = -1) {
    if (lag < 1)
      throw std::runtime_error("AmrRuntime::register_history: lag must be >= 1 (got " +
                               std::to_string(lag) + ") for history '" + name + "'");
    const int want_depth = lag + 1;
    auto it = eng.hist_rings_.find(name);
    if (it != eng.hist_rings_.end()) {
      // Idempotent re-registration: the ring depth is the MAX lag any caller requests. A larger lag
      // grows the ring (append zero-filled deeper slots on the CURRENT layout, all levels); a smaller
      // one is a no-op. The already-stored slots and the current slot [0] are preserved. The @p ncomp
      // request is IGNORED on re-registration: a name binds one component count at its first register
      // (parity with System::register_history).
      if (want_depth > eng.hist_depth_[name]) {
        const int slot_ncomp = it->second[0][0].ncomp();
        for (int s = eng.hist_depth_[name]; s < want_depth; ++s)
          it->second.push_back(alloc_slot(eng, slot_ncomp));
        eng.hist_depth_[name] = want_depth;
        std::vector<Real>& dts = eng.hist_slot_dt_[name];
        if (static_cast<int>(dts.size()) < want_depth)
          dts.resize(static_cast<std::size_t>(want_depth), Real(0));
      }
      return;
    }
    // @p ncomp < 0 (the default) resolves to block 0's ncomp -- byte-identical to the historical
    // full-state multistep ring (ADC-631); a caller that needs a narrower ring (ADC-427: the
    // 1-component condensed-Schur phi^n carry) passes an explicit ncomp >= 1. The narrow ring rides the
    // same alloc_slot / remap / replay machinery (each slot is sized by ncomp internally).
    const int resolved_ncomp = ncomp < 0 ? eng.blocks_[0].ncomp : ncomp;
    if (resolved_ncomp < 1)
      throw std::runtime_error("AmrRuntime::register_history: ncomp must be >= 1 (got " +
                               std::to_string(ncomp) + ") for history '" + name + "'");
    std::vector<std::vector<MultiFab>> ring;
    ring.reserve(static_cast<std::size_t>(want_depth));
    for (int s = 0; s < want_depth; ++s)
      ring.push_back(alloc_slot(eng, resolved_ncomp));
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

  // --- native selective-persistence replay (ADC-626 on AMR, ADC-635 through in-window regrids) -----

  // Outcome of a ring replay: how many slots were recomputed and, for the coherence guard, the sorted
  // macro-step cursors at which an in-window regrid actually fired during the internal re-steps. The
  // v3 reader asserts this fired schedule against the WRITE-time fingerprint (history_regrid_steps_).
  struct ReplayOutcome {
    int recomputed = 0;
    std::vector<int> fired_regrid_steps;
  };

  // The macro-step cursors at which the replay of a depth-@p d ring (checkpointed at macro-step @p m,
  // cadence @p regrid_every) is EXPECTED to fire a head-of-step regrid. The replay is ONE continuous
  // forward sweep from the oldest anchor (slot d-1) to slot 0, re-stepping slot by slot; the re-step
  // producing slot j runs at cursor m-1-j (the ORIGINAL step that landed the ring on macro-step m-j ran
  // with ctx.macro_step()==m-j-1, pre-increment). A regrid is due when that cursor is > 0 and divisible
  // by regrid_every. This is exactly the set the correct replay drives -- the fingerprint the v3 write
  // records (matching python replay_regrid_steps over the single full gap [0, d-1)).
  static std::vector<int> expected_regrid_steps(int d, int m, int regrid_every) {
    std::vector<int> steps;
    if (regrid_every <= 0 || d < 2)
      return steps;
    for (int j = d - 2; j >= 0; --j) {
      const int cursor = m - 1 - j;
      if (cursor > 0 && cursor % regrid_every == 0)
        steps.push_back(cursor);
    }
    std::sort(steps.begin(), steps.end());
    steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
    return steps;
  }

  // ADC-635: reconstruct the policy-recomputed slots of ring @p name by re-stepping the installed
  // Program with regrid ACTIVE, as ONE continuous forward sweep from the oldest stored anchor (slot
  // d-1) to slot 0. @p m is the checkpoint macro-step (the facade's cursor, primed by the reader); @p
  // program_step(dt, cursor) drives one re-step at the given facade macro-step so the head-of-step
  // ctx.regrid_if_due(ctx.macro_step()) reproduces the ORIGINAL in-window regrid schedule (cursor =
  // m-1-j for the re-step producing slot j). Each regrid remaps EVERY live ring slot through
  // remap_history_rings_; the ring is rebuilt by the re-stepped program's OWN store/rotate mechanics
  // (each re-step stores and rotates exactly like the original macro-step), so every rebuilt value
  // rides the SAME incremental remap chain the stored anchors rode (that chain -- not a frozen one-shot
  // re-step -- is what makes the straddling window reconstructable) and the sweep ends at cursor m-1,
  // back on the checkpoint hierarchy H_m. The single-seed sweep reconstructs a single-step recurrence
  // bit-for-bit (the documented replay class); the stored anchors are restored pristine at the end, so
  // only the recomputed slots survive on H_m.
  static ReplayOutcome rebuild_slots(AmrRuntime& eng, const std::string& name,
                                     const std::vector<int>& stored_slots, int m,
                                     const std::function<void(double, int)>& program_step) {
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
      return {};  // Dense: nothing to recompute.

    // The regrid cursors the replay MUST fire (a pure function of the ring depth, m and the cadence).
    ReplayOutcome outcome;
    const std::vector<int> expected = expected_regrid_steps(d, m, eng.regrid_every_);

    // SAVE bracket (extended for AMR): deep-copy every block's per-level U (all levels), the shared
    // aux (all levels), the multigrid warm start, the WHOLE ring store, and the regrid cadence + count.
    // Regrid is NOT frozen (ADC-635): the internal re-steps regrid exactly as the original run did, so
    // every reconstructed slot rides the same remap chain onto the checkpoint hierarchy. All state is
    // undone below; only the missing ring slots survive.
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
    const int saved_regrid_count = eng.regrid_count_;  // ADC-635: in-window regrid() bumps it

    // Per-slot dt each store produced (from the SAVED snapshot -- the replay MUTATES hist_slot_dt_).
    std::vector<Real> dts(static_cast<std::size_t>(d), Real(0));
    auto sd_it = saved_slot_dt.find(name);
    if (sd_it != saved_slot_dt.end())
      for (int k = 0; k < d && k < static_cast<int>(sd_it->second.size()); ++k)
        dts[static_cast<std::size_t>(k)] = sd_it->second[static_cast<std::size_t>(k)];

    // Reconstruct block-0's per-level trajectory as ONE continuous forward sweep: seed every level from
    // the OLDEST stored slot (d-1), then re-step slot by slot down to slot 0. The re-step producing slot
    // j runs at cursor m-1-j so its head-of-step regrid_if_due reproduces the original in-window regrid.
    // The ring itself is rebuilt by the RE-STEPPED PROGRAM'S OWN store/rotate mechanics (each re-step's
    // body stores the committed state and rotates the ring exactly as the original macro-step did), so
    // every rebuilt slot sits in the LIVE ring and rides the in-window regrids' remap_history_rings_
    // through the SAME incremental chain the original values rode, landing back on H_m at the sweep's
    // end (cursor m-1). No side placement: writing slots by index would fight the rotation (the ADC-635
    // depth-5 corruption). Every regrid that completes is recorded (regrid_count_ delta) for the guard.
    std::vector<char> is_stored(static_cast<std::size_t>(d), 0);
    for (int s : anchors)
      is_stored[static_cast<std::size_t>(s)] = 1;
    for (int k = 0; k < eng.nlev_; ++k)
      pops::lincomb(
          (*eng.blocks_[0].levels)[static_cast<std::size_t>(k)].U, Real(1),
          saved_rings.at(name)[static_cast<std::size_t>(d - 1)][static_cast<std::size_t>(k)], Real(0),
          (*eng.blocks_[0].levels)[static_cast<std::size_t>(k)].U);
    for (int j = d - 2; j >= 0; --j) {
      const int cursor = m - 1 - j;
      const int rc_before = eng.regrid_count_;
      program_step(static_cast<double>(dts[static_cast<std::size_t>(j)]), cursor);
      if (eng.regrid_count_ != rc_before)
        outcome.fired_regrid_steps.push_back(cursor);  // a regrid completed at this cursor
    }
    std::sort(outcome.fired_regrid_steps.begin(), outcome.fired_regrid_steps.end());
    outcome.fired_regrid_steps.erase(
        std::unique(outcome.fired_regrid_steps.begin(), outcome.fired_regrid_steps.end()),
        outcome.fired_regrid_steps.end());

    // COHERENCE GUARD (ADC-635): every regrid the replay COMPLETED must sit on the due schedule
    // derived from (depth, m, regrid_every) -- a completed regrid at an off-schedule cursor means the
    // cursor driving or the cadence is wrong; hard error, never silent. The converse is legitimate: a
    // due cursor whose regrid() no-ops (single-level hierarchy, no wired predicate, empty tags)
    // completes nothing, and by determinism the ORIGINAL run no-oped there identically, so the
    // reconstruction is unharmed (the v3 reader still asserts the recorded fingerprint separately).
    for (int s : outcome.fired_regrid_steps)
      if (std::find(expected.begin(), expected.end(), s) == expected.end())
        throw std::runtime_error(
            "AmrRuntime::rebuild_history_slots: the replay of history '" + name +
            "' completed a regrid at macro-step " + std::to_string(s) +
            " which is OFF the due schedule (checkpoint macro-step " + std::to_string(m) +
            ", regrid_every " + std::to_string(eng.regrid_every_) +
            "); the replay cursor driving or the restart composition is inconsistent with the "
            "recorded in-window regrid schedule.");

    // Extract the recomputed slots (already remapped through the in-window chain onto the live layout),
    // then RESTORE every replay side effect: block states, aux, warm start, regrid count. The saved ring
    // supplies the pristine anchors; the recomputed slots keep the values they rode to.
    std::map<std::string, std::vector<std::vector<MultiFab>>> replayed_rings = eng.hist_rings_;
    for (std::size_t b = 0; b < eng.blocks_.size(); ++b)
      for (int k = 0; k < static_cast<int>(eng.blocks_[b].levels->size()); ++k)
        (*eng.blocks_[b].levels)[static_cast<std::size_t>(k)].U =
            std::move(saved_states[b][static_cast<std::size_t>(k)]);
    eng.aux_ = std::move(saved_aux);
    eng.mg_.phi() = std::move(saved_phi);  // restore the multigrid warm-start iterate
    eng.hist_rings_ = saved_rings;         // pristine anchors on the checkpoint hierarchy
    eng.hist_init_ = saved_init;
    eng.hist_slot_dt_ = saved_slot_dt;
    eng.regrid_count_ = saved_regrid_count;

    // Copy ONLY the recomputed slots back from the replayed ring (the anchors keep their restored
    // values), all levels. The recomputed slots share the checkpoint hierarchy's layout: they rode the
    // in-window remap chain back onto it (the last re-step lands on macro-step m, the checkpoint grid).
    std::vector<std::vector<MultiFab>>& out_ring = eng.hist_rings_.at(name);
    const std::vector<std::vector<MultiFab>>& src = replayed_rings.at(name);
    for (int j = 0; j < d; ++j) {
      if (is_stored[static_cast<std::size_t>(j)])
        continue;
      for (int k = 0; k < eng.nlev_; ++k)
        out_ring[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] =
            src[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)];  // deep copy (rode the chain)
      ++outcome.recomputed;
    }
    return outcome;
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
