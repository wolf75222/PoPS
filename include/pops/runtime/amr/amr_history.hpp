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
/// its explicitly bound block's level-`level` state at macro-step lag `slot` (slot 0 = newest). Rotation swaps the whole
/// per-level buffer set atomically (a vector-of-handles swap, O(1), no deep copy). `slot_dt` is ONE
/// scalar per slot (the synchronous AMR driver advances every level with the SAME dt), byte-identical
/// to the Uniform ring's per-slot dt, so the ADC-626 variable-dt replay math is reused unchanged.
///
/// REGRID REMAP (section 3): after any hierarchy change every ring slot's per-level data is remapped
/// onto the NEW (fb, dmap) with the SAME R6/R7 machinery the live U uses (regrid_field_on_layout, the
/// ring slot's own inherited ghost width), so `prev(k)` reads stay layout-consistent with the current
/// U. The coarse slot is stable; finer slots are prolonged/restricted from it exactly like U.
///
/// NATIVE REPLAY (section 5, ADC-631): rebuild_history_slots reconstructs every policy-omitted gap
/// from its exact older stored anchor and publishes each accepted post-step state by logical slot
/// index. This is exact only while the replay window retains one hierarchy. The authenticated Python
/// capture plan promotes a selective policy to dense storage when a regrid is scheduled in that
/// window; the native seam also refuses such a call defensively. A complete StepSnapshot brackets
/// every gap so replay is identity on live states, fields, caches, diagnostics, hierarchy and rings.
/// Every ring re-seeds only its authenticated block owner; independent rings may target independent
/// blocks.

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
  // One per-level buffer set, co-distributed with the runtime-owned hierarchy.
  // @p ncomp components, one ghost, zero-initialized -- one ring slot on the shared hierarchy.
  static std::vector<MultiFab> alloc_slot(const AmrRuntime& eng, int ncomp) {
    std::vector<MultiFab> per_level;
    per_level.reserve(static_cast<std::size_t>(eng.nlev_));
    for (int k = 0; k < eng.nlev_; ++k) {
      MultiFab slot(eng.hierarchy_.ba[static_cast<std::size_t>(k)],
                    eng.hierarchy_.dm[static_cast<std::size_t>(k)], ncomp, 1);
      slot.set_val(Real(0));
      per_level.push_back(std::move(slot));
    }
    return per_level;
  }

  // --- AmrProgramContext-facing seams (register / read / store / rotate) --------------------------

  static void register_history(AmrRuntime& eng, std::size_t block, const std::string& name, int lag,
                               int ncomp = -1) {
    if (block >= eng.blocks_.size())
      throw std::runtime_error("AmrRuntime::register_history: block owner out of bounds");
    if (lag < 1)
      throw std::runtime_error("AmrRuntime::register_history: lag must be >= 1 (got " +
                               std::to_string(lag) + ") for history '" + name + "'");
    const int want_depth = lag + 1;
    auto it = eng.hist_rings_.find(name);
    if (it != eng.hist_rings_.end()) {
      if (eng.hist_block_owner_.at(name) != block)
        throw std::runtime_error("AmrRuntime::register_history: owner mismatch for history '" +
                                 name + "'");
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
    // @p ncomp < 0 resolves to the explicitly bound block's width; a caller that needs a narrower ring
    // 1-component condensed-Schur phi^n carry) passes an explicit ncomp >= 1. The narrow ring rides the
    // same alloc_slot / remap / replay machinery (each slot is sized by ncomp internally).
    const int resolved_ncomp = ncomp < 0 ? eng.blocks_[block].ncomp : ncomp;
    if (resolved_ncomp < 1)
      throw std::runtime_error("AmrRuntime::register_history: ncomp must be >= 1 (got " +
                               std::to_string(ncomp) + ") for history '" + name + "'");
    std::vector<std::vector<MultiFab>> ring;
    ring.reserve(static_cast<std::size_t>(want_depth));
    for (int s = 0; s < want_depth; ++s)
      ring.push_back(alloc_slot(eng, resolved_ncomp));
    eng.hist_rings_.emplace(name, std::move(ring));
    eng.hist_depth_[name] = want_depth;
    eng.hist_block_owner_[name] = block;
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
  // (level 0 then 1 ...), each at the v3 convention c*nf*nf+j*nf+i (nf derives from the runtime
  // hierarchy transition product; zeros outside
  // the patches at a fine level) -- the SAME layout level_aux_flat uses, hiding the level axis inside
  // the accessor so _system_io_history.py is reused verbatim. @p gather collects only the
  // ownership-distributed level slices; replicated level 0 is already global on every rank.
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
      const Box2D domain = eng.dom_.refine(eng.level_refinement(k));
      const std::size_t nf = static_cast<std::size_t>(domain.nx());
      std::vector<double> lvl(static_cast<std::size_t>(nc) * nf * nf, 0.0);
      for (int li = 0; li < S.local_size(); ++li) {
        const ConstArray4 a = S.fab(li).const_array();
        const Box2D v = S.box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            for (int c = 0; c < nc; ++c)
              lvl[static_cast<std::size_t>(c) * nf * nf +
                  static_cast<std::size_t>(j - domain.lo[1]) * nf +
                  static_cast<std::size_t>(i - domain.lo[0])] = a(i, j, c);
      }
      if (gather && (k > 0 || !eng.replicated_coarse_))
        all_reduce_sum_inplace(lvl.data(), static_cast<int>(lvl.size()));
      out.insert(out.end(), lvl.begin(), lvl.end());
    }
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
      throw std::runtime_error("AmrRuntime::restore_history: history '" + name +
                               "' has no owner-qualified registration in the installed Program");
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
      const Box2D domain = eng.dom_.refine(eng.level_refinement(k));
      const std::size_t nf = static_cast<std::size_t>(domain.nx());
      for (int li = 0; li < S.local_size(); ++li) {
        Array4 a = S.fab(li).array();
        const Box2D v = S.box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            for (int c = 0; c < nc; ++c)
              a(i, j, c) = flat[off + static_cast<std::size_t>(c) * nf * nf +
                                static_cast<std::size_t>(j - domain.lo[1]) * nf +
                                static_cast<std::size_t>(i - domain.lo[0])];
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

  // --- native selective-persistence replay on a stable AMR hierarchy -------------------------------

  // Outcome of a ring replay. A non-empty fired schedule is a hard contract violation: the resolved
  // checkpoint plan must promote a replay window containing a scheduled regrid to dense storage.
  struct ReplayOutcome {
    int recomputed = 0;
    std::vector<int> fired_regrid_steps;
  };

  // Scheduled head-of-step regrid cursors inside the prospective replay window of a depth-@p d ring
  // checkpointed at macro-step @p m. Replay starts independently from each exact older anchor and
  // sweeps its gap toward the newer anchor; across those gaps the re-step producing slot j runs at
  // cursor m-1-j (the ORIGINAL step that landed the ring on macro-step m-j ran with
  // ctx.macro_step()==m-j-1, pre-increment). A regrid is due when that cursor is > 0 and divisible by
  // regrid_every. A non-empty set makes selective replay unsafe and forces dense safety storage. The
  // v3 payload records the same fingerprint (matching Python ``replay_regrid_steps``).
  static std::vector<int> scheduled_regrid_steps(int d, int m, int regrid_every) {
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

  // Reconstruct a clean-window ring by re-stepping the installed Program independently between each
  // pair of stored anchors. A scheduled regrid is refused: stored anchors already live on the
  // checkpoint hierarchy, so replaying a historical remap from that future layout is not exact.
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
      throw std::runtime_error("AmrRuntime::rebuild_history_slots: the oldest slot " +
                               std::to_string(d - 1) + " of history '" + name +
                               "' is not stored; the ring is unreconstructable (the persistence "
                               "policy must store the oldest slot).");
    if (static_cast<int>(anchors.size()) == d)
      return {};  // Dense: nothing to recompute.

    // Resolve the structural schedule defensively even when a handcrafted caller bypasses Python.
    ReplayOutcome outcome;
    const std::vector<int> scheduled = scheduled_regrid_steps(d, m, eng.regrid_every_);
    if (!scheduled.empty())
      throw std::runtime_error(
          "AmrRuntime::rebuild_history_slots: selective replay window contains scheduled regrid "
          "steps; the resolved checkpoint plan must use dense_regrid_safety storage");

    // SAVE bracket: use the accepted-state snapshot rather than a partial hand-maintained list.
    // Besides all block states and rings it carries aux, every default/named elliptic warm start,
    // bootstrap caches, hierarchy/cadence counters and diagnostics.  Each anchor gap starts from
    // this same image, and every replay side effect is undone before the reconstructed slots are
    // published.  This keeps restart replay aligned with the ordinary step-transaction boundary.
    const AmrRuntime::StepSnapshot saved = eng.step_snapshot();

    // Per-slot dt each store produced (from the SAVED snapshot -- the replay MUTATES hist_slot_dt_).
    std::vector<Real> dts(static_cast<std::size_t>(d), Real(0));
    auto sd_it = saved.history_slot_dt.find(name);
    if (sd_it != saved.history_slot_dt.end())
      for (int k = 0; k < d && k < static_cast<int>(sd_it->second.size()); ++k)
        dts[static_cast<std::size_t>(k)] = sd_it->second[static_cast<std::size_t>(k)];

    // Reconstruct every gap independently from its exact older stored anchor.  The Program's own
    // store/rotate operations are replay side effects, not the output: trusting their final rotated
    // positions shifts omitted slots by one and bypasses intermediate anchors.  Capture the accepted
    // block state after each historical step and publish it explicitly at its logical slot index,
    // exactly like the Uniform engine's replay.
    std::vector<char> is_stored(static_cast<std::size_t>(d), 0);
    for (int s : anchors)
      is_stored[static_cast<std::size_t>(s)] = 1;
    const auto owner = saved.history_block_owner.at(name);
    std::vector<std::vector<MultiFab>> reconstructed(static_cast<std::size_t>(d));
    const auto restore_saved = [&] { eng.restore_step_snapshot(saved); };
    try {
      for (std::size_t anchor = 0; anchor + 1 < anchors.size(); ++anchor) {
        const int newer = anchors[anchor];
        const int older = anchors[anchor + 1];
        restore_saved();
        for (int level = 0; level < eng.nlev_; ++level) {
          MultiFab& state = (*eng.blocks_[owner].levels)[static_cast<std::size_t>(level)].U;
          pops::lincomb(state, Real(1),
                        saved.history_rings.at(
                            name)[static_cast<std::size_t>(older)][static_cast<std::size_t>(level)],
                        Real(0), state);
        }
        for (int j = older - 1; j >= newer; --j) {
          const int cursor = m - 1 - j;
          const int rc_before = eng.regrid_count_;
          program_step(static_cast<double>(dts[static_cast<std::size_t>(j)]), cursor);
          if (eng.regrid_count_ != rc_before)
            outcome.fired_regrid_steps.push_back(cursor);
          if (is_stored[static_cast<std::size_t>(j)])
            continue;
          auto& slot = reconstructed[static_cast<std::size_t>(j)];
          slot.reserve(static_cast<std::size_t>(eng.nlev_));
          for (int level = 0; level < eng.nlev_; ++level)
            slot.push_back((*eng.blocks_[owner].levels)[static_cast<std::size_t>(level)].U);
        }
      }
    } catch (...) {
      restore_saved();
      throw;
    }
    restore_saved();
    std::sort(outcome.fired_regrid_steps.begin(), outcome.fired_regrid_steps.end());
    outcome.fired_regrid_steps.erase(
        std::unique(outcome.fired_regrid_steps.begin(), outcome.fired_regrid_steps.end()),
        outcome.fired_regrid_steps.end());

    if (!outcome.fired_regrid_steps.empty())
      throw std::runtime_error("AmrRuntime::rebuild_history_slots: history '" + name +
                               "' changed hierarchy during a stable-window replay; the resolved "
                               "checkpoint plan must use "
                               "dense_regrid_safety storage");

    // The snapshot restore above reinstated every accepted value and the pristine stored anchors.
    // Publish only the explicitly reconstructed gaps; all slots share the unchanged checkpoint
    // hierarchy and every stored anchor remains byte-for-byte the payload value.
    std::vector<std::vector<MultiFab>>& out_ring = eng.hist_rings_.at(name);
    for (int j = 0; j < d; ++j) {
      if (is_stored[static_cast<std::size_t>(j)])
        continue;
      const auto& slot = reconstructed[static_cast<std::size_t>(j)];
      if (slot.size() != static_cast<std::size_t>(eng.nlev_))
        throw std::runtime_error(
            "AmrRuntime::rebuild_history_slots: stored anchors do not bracket omitted slot " +
            std::to_string(j) + " of history '" + name + "'");
      for (int k = 0; k < eng.nlev_; ++k)
        out_ring[static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] =
            slot[static_cast<std::size_t>(k)];
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
