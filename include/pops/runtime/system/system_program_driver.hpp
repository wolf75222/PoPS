#pragma once

#include <pops/core/foundation/types.hpp>                   // Real
#include <pops/coupling/source/coupled_source_program.hpp>  // CoupledFreqKernel (per-cell coupled frequency)
#include <pops/mesh/execution/for_each.hpp>  // reduce_max_cell (max mu over the cells, device-clean functor)
#include <pops/parallel/comm.hpp>  // all_reduce_min/max (global bounds: identical dt on all ranks)
#include <pops/runtime/numerical_defaults.hpp>

#include <stdexcept>

#include <algorithm>  // std::min, std::max (CFL: min grid physical step, min dt over the blocks)
#include <cmath>      // std::isfinite (step_cfl)
#include <limits>     // std::numeric_limits (per-block CFL: dt = min over the blocks)
#include <string>     // last_dt_bound (name of the active bound of the last step_cfl)

/// @file
/// @brief Uniform facade time policy after the Program cutover.
///
/// CONTRACT / INVARIANTS
/// - `step` and `step_cfl` advance exclusively through the installed whole-system Program.
/// - Native block closures remain available to ProgramContext for spatial RHS, source, projection
///   and field operations; this helper never assembles an implicit fallback macro-step.
/// - `step_cfl` retains the native, model-aware stability-bound calculation, then hands the selected
///   dt to the Program.
/// - CFL PHYSICAL STEP h: Cartesian = min(dx, dy); POLAR = min(dr, r_min * dtheta) (the azimuthal step
///   r*dtheta is minimal at the inner radius r_min of the ring -> most constraining edge).
/// - PROGRAM CADENCE INVARIANT (hold-then-catch-up): a whole-system Program of cadence M is held until
///   the end of its window, then receives M*dt; the facade clock advances once per accepted macro-step.
/// - PER-BLOCK CFL FORMULA (substeps-aware, post-#121): dt <= cfl * h * substeps_b / (stride_b * w_b);
///   the global dt is the min over the evolving blocks.
///
/// Since System::Impl stays PRIVATE to python/system.cpp, this helper is a TEMPLATE parameterized on the
/// real Impl type (same technique as system_field_solver / native_loader): python/system.cpp instantiates
/// it with System::Impl after defining Impl. owner_ is an Impl* (the helper lifetime is subordinate to
/// that of Impl). System::step and step_cfl delegate here after the facade's fail-before-mutation guard.

namespace pops::runtime::system {

/// SystemProgramDriver<Impl>: see the contract above. All methods are MEMBERS because they share
/// Program cadence and stability-bound evaluation; accesses to the SHARED state of Impl go through
/// owner_-> verbatim.
/// Templated on Impl to stay free of any dependency on the (private) definition of System::Impl.
template <class Impl>
class SystemProgramDriver {
 public:
  /// @param owner back-pointer to System::Impl (lifetime subordinate to that of Impl).
  explicit SystemProgramDriver(Impl* owner) : owner_(owner) {}

  /// True if a block of cadence @p stride CATCHES UP at this macro-step (END of window).
  /// STRIDE SEMANTICS = HOLD-THEN-CATCH-UP (catch-up at the END of the window). A block of cadence M is
  /// HELD (not advanced) on the macro-steps where (macro_step + 1) % M != 0, then advances by an effective
  /// step M*dt at the macro-step where (macro_step + 1) % M == 0, i.e. at the END of its window of M
  /// macro-steps. At macro-step k, the system time is (k+1)*dt and the block that CATCHES UP has then
  /// advanced by the same cumulative (k+1)*dt: it is temporally CONSISTENT with the fast blocks, never
  /// "in the future". (The old semantics advanced at the START of the window, macro_step % M == 0: at k=0
  /// the block already advanced M*dt while the system advanced only dt -> anticipated block, wrong
  /// Poisson/source coupling.)
  static bool stride_due(int macro_step, int stride) { return (macro_step + 1) % stride == 0; }

  /// Step bound from PER-CELL COUPLED FREQUENCIES (CoupledSource.frequency with an Expr,
  /// refinement of the CONSTANT frequency). For each registered program: reduces the MAX of mu(U)
  /// over the LOCAL fabs of the FIRST input block (CoupledFreqKernel, named device-clean functor;
  /// same rank-local ownership convention as the registered operator), GLOBAL all_reduce_max, then
  /// dt <= cfl / max(mu).
  /// Updates @p dt (and @p reason if non-null) if the bound is tighter. max(mu) <= 0 = no
  /// bound this step. Reason "coupled_source:<label>" -- SAME prefix as the constant frequency, for a
  /// uniform diagnostic. Per-cell counterpart of the constant loop of step_cfl;
  /// no per-cell source registered -> empty loop, bit-identical trajectory.
  ///
  /// MPI: all_reduce_max is called by ALL ranks, the SAME number of times (coupled_freq_exprs_ is
  /// identical on all ranks) -> symmetric collective, identical dt everywhere (no deadlock). A
  /// rank with no local box reduces m=0 (neutral for MAX). WARNING: the Array4 are rebuilt at
  /// EACH step because the fabs may be reallocated.
  void apply_coupled_freq_expr_bounds(double cfl, double& dt, std::string* reason) const {
    Impl* P = owner_;
    for (const auto& ce : P->coupled_freq_exprs_) {
      Real m = 0;
      if (ce.n_in > 0) {
        auto& Uref = P->sp[static_cast<std::size_t>(ce.ins[0].sidx)].U;
        for (int li = 0; li < Uref.local_size(); ++li) {
          CoupledFreqKernel kern;
          kern.n_in = ce.n_in;
          kern.n_const = static_cast<int>(ce.kconsts.size());
          for (int c = 0; c < ce.n_in; ++c) {
            kern.in[c] = P->sp[static_cast<std::size_t>(ce.ins[static_cast<std::size_t>(c)].sidx)]
                             .U.fab(li)
                             .array();
            kern.in_comp[c] = ce.ins[static_cast<std::size_t>(c)].comp;
          }
          for (int c = 0; c < kern.n_const; ++c)
            kern.consts[c] = ce.kconsts[static_cast<std::size_t>(c)];
          kern.prog = ce.prog;
          m = std::max(m, reduce_max_cell(Uref.box(li), kern));
        }
      } else {
        // Program WITHOUT an input field (constant frequency expressed in bytecode): evaluated once
        // on the constants alone (no box to traverse); identical on all ranks.
        Real reg[kCsMaxReg];
        const int nc = static_cast<int>(ce.kconsts.size());
        for (int c = 0; c < nc; ++c)
          reg[c] = ce.kconsts[static_cast<std::size_t>(c)];
        const Real mu0 = ce.prog.eval(reg);
        if (mu0 > Real(0))
          m = mu0;
      }
      const double mu = all_reduce_max(static_cast<double>(m));  // ALL ranks (collective symmetry)
      if (mu > 0.0) {
        const double dt_cs = cfl / mu;
        if (dt_cs < dt) {
          dt = dt_cs;
          if (reason)
            *reason = "coupled_source:" + ce.label;
        }
      }
    }
  }

  /// MIN physical step of the grid for step_cfl: Cartesian = min(dx, dy);
  /// POLAR = min(dr, r_min * dtheta) -- the azimuthal physical step r*dtheta is minimal at the inner
  /// radius r_min of the ring (the most constraining edge for the CFL). Reads rank-local geometry
  /// only (no collective).
  Real cfl_grid_h() const {
    Impl* P = owner_;
    return P->polar_ ? std::min(P->pgeom_.dr(), P->pgeom_.r_min * P->pgeom_.dtheta())
                     : std::min(P->geom.dx(), P->geom.dy());
  }

  /// GLOBAL step bounds (System::add_dt_bound): multi-block coupling, Schur/Poisson, AMR/scheduler.
  /// One HOST evaluation per step and per bound; <= 0 or non-finite = does not constrain this step
  /// (neutralized to +inf BEFORE the global min). ALL_REDUCE_MIN mandatory: the callback is
  /// evaluated PER RANK (it may read a rank-local state); without the global min each rank would
  /// choose a different dt -> desynchronized step collectives (Krylov / fill_boundary) -> MPI
  /// deadlock. In serial all_reduce_min is the identity (bit-identical). @p reason, if non-null, is
  /// set to "global:<label>" for the winning bound.
  /// MPI: dt_bounds_ is identical on all ranks and `if(!g.fn)` is rank-uniform, so the collective is
  /// symmetric (same count/order on every rank) -- factoring keeps the deadlock-safety unchanged.
  void apply_global_dt_bounds(double& dt, std::string* reason) const {
    Impl* P = owner_;
    for (const auto& g : P->dt_bounds_) {
      if (!g.fn)
        continue;
      double v = g.fn();
      if (!(v > 0.0) || !std::isfinite(v))
        v = std::numeric_limits<double>::infinity();
      v = all_reduce_min(v);
      if (v < dt) {
        dt = v;
        if (reason)
          *reason = "global:" + g.label;
      }
    }
  }

  /// Runs ONE macro-step of length @p dt through an INSTALLED compiled time Program (epic ADC-399):
  /// the SYSTEM-level cadence (substeps + stride, ADC-411) wrapped around the opaque program closure,
  /// then the clock tick. Shared by step() (Lie path) and step_cfl() (CFL path) so both route a
  /// compiled program through the SAME cadence, keeping them consistent. The Program OWNS the whole
  /// step body (solve_fields, RHS, combine, commit -- all via ProgramContext); the runtime adds no
  /// implicit solve_fields / couplings / projections here (cf. step()): they are the Program's job.
  ///
  /// SUBSTEPS + STRIDE (ADC-411):
  ///   - stride M: GLOBAL hold-then-catch-up. The whole program is HELD on the macro-steps where
  ///     stride_due is false, then runs ONCE with the effective step eff_dt = M*dt at the window end.
  ///     A compiled program is ONE whole-system closure, so the stride is GLOBAL (whole-system); this
  ///     equals native per-block stride ONLY for a single-block system (or all blocks sharing M).
  ///   - substeps n: subdivides the EFFECTIVE step into n calls program_.step_(eff_dt/n). Each call
  ///     executes the complete authored Program; no hidden block-local subcycling is inferred.
  /// The clock ticks EVERY macro-step (held steps included), matching native. Default cadence 1/1 is
  /// byte-identical to the single program_.step_(dt) call: stride_due(_, 1) is always true and n == 1
  /// collapses the loop to one call with h == dt.
  void run_program_cadence(double dt) {
    Impl* P = owner_;
    if (stride_due(P->macro_step_, P->program_.stride_)) {
      const Real eff_dt = Real(dt) * Real(P->program_.stride_);  // catch-up: effective step M*dt
      const int n = P->program_.substeps_;
      const Real h = eff_dt / Real(n);  // substeps subdivide the EFFECTIVE step (native: eff_dt/n)
      for (int sub = 0; sub < n; ++sub) {
        // Record the dt handed to the program BEFORE the call so the runtime's store_history can tag
        // the slot it produces with the exact dt (ADC-626 variable-dt replay). Shared by step() and
        // step_cfl() (both route here), so no call site is missed. A plain field assignment.
        P->program_.last_dt_ = h;
        P->program_.step_(h);
      }
    }
    P->t += dt;  // clock ticks EVERY macro-step (held steps included), like native
    P->macro_step_++;
  }

  /// One macro-step of length @p dt through the installed whole-system Program.
  void step(double dt) {
    Impl* P = owner_;
    P->program_.require_step_installed("System::step");
    run_program_cadence(dt);
  }

  /// One macro-step at CFL dt: dt = min over the evolving blocks of the block step BOUNDS, then advances
  /// like step. @return the dt used. SUBSTEPS-AWARE (post-#121): bit-identical to the old
  /// formula only for substeps=1 (cf. backward-compatibility note).
  ///
  /// STEP POLICY (audit 2026-06, step_cfl worksite): the historical TRANSPORT bound
  /// dt <= cfl*h*substeps_b/(stride_b*w_b) stays the base, but the step now AGGREGATES, per block:
  ///   - the SOURCE FREQUENCY bound (s.source_frequency, HasSourceFrequency trait):
  ///     effective substep stride*dt/substeps <= cfl/mu -> dt <= cfl*substeps/(stride*mu), WITHOUT h
  ///     (a local source bounds in 1/time, not in length/time);
  ///   - the direct ADMISSIBLE STEP (s.stability_dt, HasStabilityDt trait):
  ///     stride*dt/substeps <= dt_adm -> dt <= dt_adm*substeps/stride, WITHOUT cfl (the model already
  ///     declares an admissible step);
  ///   - the CFL speed itself can be the declared STABILITY speed (HasStabilitySpeed
  ///     trait): s.max_speed is then wired onto stability_speed (cf. make_max_speed).
  /// Then the GLOBAL bounds (P->dt_bounds_: multi-block coupling, Schur/Poisson, AMR/scheduler,
  /// set by System::add_dt_bound): dt <= fn() each, one HOST evaluation per step (no per-cell
  /// callback). A block / a system WITHOUT optional bounds keeps a step STRICTLY identical to
  /// history (empty functions are not queried). The ACTIVE bound of the last step is consultable via
  /// last_dt_bound() ("transport:<block>", "source_frequency:<block>",
  /// "stability_dt:<block>", "global:<label>", "degenerate").
  /// 2D CONVENTION NOTE (audit ADC-182): the per-cell speed is w = max(wx, wy) -- NOT the
  /// sum. In unsplit 2D the effective Courant number thus reaches 2*cfl when wx ~ wy:
  /// cfl = 0.4 (case default) stays < 1 (safe), this is also the convention of the sweep
  /// references (HLL step); cfl >= 0.5 in unsplit 2D is MARGINAL -- to avoid without study.
  double step_cfl(double cfl, double speed_floor = static_cast<double>(kCflSpeedFloor),
                  double max_dt = std::numeric_limits<double>::infinity(), double min_dt = 0.0) {
    Impl* P = owner_;
    P->program_.require_step_installed("System::step_cfl");
    P->solve_fields();
    // MIN physical step of the grid (Cartesian min(dx,dy) / polar min(dr, r_min*dtheta), cf.
    // cfl_grid_h). The rest of the CFL formula (per block, substeps/stride) is unchanged.
    const Real h = cfl_grid_h();
    // PER-BLOCK CFL, STRIDE AND SUBSTEPS FACTOR INCLUDED. A block of cadence M advances by an effective
    // step M*dt in substeps_b substeps, so each substep is worth stride_b * dt / substeps_b: the stable
    // condition per substep is stride_b * dt / substeps_b <= cfl * h / w_b, that is
    //   dt <= cfl * h * substeps_b / (stride_b * w_b).
    // The GLOBAL dt is the min over the evolving blocks (the most constraining). Without this, the step
    // computed on w_max alone then multiplied by M would violate the CFL by a factor M on the stride block.
    //
    // BACKWARD COMPATIBILITY (post-#121). The formula is SUBSTEPS-AWARE: with substeps_b > 1, the dt
    // returned is substeps_b times larger than the old formula dt = cfl*h/(stride*w).
    // bit-identical only for substeps=1 (at any stride); step_cfl is now substeps-aware
    // (dt = cfl*h*substeps/(stride*w)), so a step_cfl run with substeps>1 advances a larger dt
    // than before #121 (CFL-maximal step, each substep at the stability limit).
    // To reproduce a run calibrated with the old formula, use step(dt) with the explicit historical
    // dt, NOT step_cfl.
    double dt = std::numeric_limits<double>::infinity();
    std::string reason = "degenerate";
    for (auto& s : P->sp) {
      if (!s.evolve)
        continue;  // frozen block: does not constrain the step
      // ADC-645: the caller-facing speed floor (default = the historical kCflSpeedFloor literal,
      // bit-identical): w floors the reduced wave speed so a quiescent block cannot divide by zero.
      const Real w = std::max(s.max_speed(s.U), static_cast<Real>(speed_floor));
      double dt_b = cfl * static_cast<double>(h) * static_cast<double>(s.substeps) /
                    (static_cast<double>(s.stride) * static_cast<double>(w));
      const char* why = "transport";
      // SOURCE FREQUENCY bound (optional; mu <= 0 = does not constrain).
      if (s.source_frequency) {
        const Real mu = s.source_frequency(s.U);
        if (mu > Real(0)) {
          const double dt_src = cfl * static_cast<double>(s.substeps) /
                                (static_cast<double>(s.stride) * static_cast<double>(mu));
          if (dt_src < dt_b) {
            dt_b = dt_src;
            why = "source_frequency";
          }
        }
      }
      // Direct ADMISSIBLE STEP (optional; <= 0 = does not constrain; cfl NOT applied).
      if (s.stability_dt) {
        const Real db = s.stability_dt(s.U);
        if (db > Real(0)) {
          const double dt_adm = static_cast<double>(db) * static_cast<double>(s.substeps) /
                                static_cast<double>(s.stride);
          if (dt_adm < dt_b) {
            dt_b = dt_adm;
            why = "stability_dt";
          }
        }
      }
      if (dt_b < dt) {
        dt = dt_b;
        reason = std::string(why) + ":" + s.name;
      }
    }
    // DECLARED system-coupling frequencies constrain the whole Program macro-dt directly:
    // dt <= cfl / mu (no block-local substeps/stride factor).
    for (const auto& cs : P->coupled_freqs_) {
      if (!(cs.mu > 0.0))
        continue;
      const double dt_cs = cfl / cs.mu;
      if (dt_cs < dt) {
        dt = dt_cs;
        reason = "coupled_source:" + cs.label;
      }
    }
    // PER-CELL frequencies (CoupledSource.frequency with an Expr): mu(U) reduced (MAX) per cell at
    // this step, global all_reduce_max, dt <= cfl / max(mu). Same reason "coupled_source:<label>" as the
    // constant. No per-cell source -> no-op (bit-identical).
    apply_coupled_freq_expr_bounds(cfl, dt, &reason);
    // GLOBAL bounds (System::add_dt_bound): all_reduce_min over the registered bounds, tracking the
    // winning reason (see apply_global_dt_bounds for the MPI deadlock-safety rationale).
    apply_global_dt_bounds(dt, &reason);
    // OPTIONAL compiled-Program dt bound (epic ADC-399 / ADC-417, spec s18). When the installed Program
    // exported one (System::install_program stored program_.dt_bound_), it TIGHTENS dt to the min of the
    // native CFL dt above and the program's own bound. No program / no bound -> the closure is empty and
    // dt is the native CFL UNCHANGED. The native CFL logic above is left intact: this only reduces dt.
    // MPI-SAFE: program_.dt_bound_ runs the SAME collective reduction (block_max_speed / reductions) on
    // every rank, so the bound is rank-uniform -- like apply_global_dt_bounds, the min keeps the step
    // collectives symmetric (no desync / deadlock).
    if (P->program_.dt_bound_) {
      const double pb = static_cast<double>(P->program_.dt_bound_(static_cast<Real>(cfl)));
      if (std::isfinite(pb) && pb > 0.0 && pb < dt) {
        dt = pb;
        reason = "program:dt_bound";
      }
    }
    if (!std::isfinite(dt)) {
      dt = cfl * static_cast<double>(h) /
           static_cast<double>(kCflSpeedFloor);  // all frozen: degenerate step
      reason = "degenerate";
    }
    if (std::isnan(max_dt) || max_dt <= 0.0)
      throw std::invalid_argument("System::step_cfl max_dt must be positive or +infinity");
    if (std::isfinite(max_dt)) {
      if (max_dt < dt) {
        dt = max_dt;
        reason = "strategy:max_dt";
      }
    }
    if (std::isnan(min_dt) || min_dt < 0.0)
      throw std::invalid_argument("System::step_cfl min_dt must be finite and >= 0");
    if (dt < min_dt)
      throw std::runtime_error("System::step_cfl stability bound is below declared min_dt");
    last_dt_reason_ = std::move(reason);
    // CFL remains a native bound calculation, but the accepted advance is exclusively the Program.
    run_program_cadence(dt);
    return dt;
  }

  /// Name of the ACTIVE bound (the one that fixed dt) of the last step_cfl: "transport:<block>",
  /// "source_frequency:<block>", "stability_dt:<block>", "global:<label>", "degenerate", or "" if
  /// no step_cfl has run yet. Diagnostic (System::last_dt_bound).
  const std::string& last_dt_reason() const { return last_dt_reason_; }
  void restore_last_dt_reason(std::string reason) { last_dt_reason_ = std::move(reason); }

 private:
  Impl* owner_;
  std::string last_dt_reason_;  // active bound of the last step_cfl (diagnostic)
};

}  // namespace pops::runtime::system
