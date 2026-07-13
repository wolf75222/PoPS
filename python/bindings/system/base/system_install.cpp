// ADC-632: install/composition seam of the System facade -- the structural setters guarded by
// require_assembling (blocks, aux, elliptic/reaction/epsilon fields, disc domain, geometry mode,
// coupled sources, source stages, time scheme, gauss policy) plus install_program. This TU is the
// subdivision of system.cpp that instantiates the native_loader templates for the dynamic/compiled/
// native block loaders; it includes native_loader.hpp AFTER system_impl.hpp so Impl is complete.
// Pure body move from system.cpp, no logic changed -> production trajectories bit-identical.
#include "system_impl.hpp"  // ADC-632: shared System::Impl + facade helpers (binding-private)

// native_loader.hpp templates instantiate on Impl (add_dynamic/compiled/native_block); included
// AFTER system_impl.hpp per the historical "templates instantiated lower down" per-TU ordering.
#include <pops/runtime/builders/compiled/native_loader.hpp>  // .so loading (JIT/AOT/native) + ABI guard
#include <pops/runtime/config/route_ids.hpp>  // ADC-641: parse_{transport,riemann,time}_route typed switches

namespace pops {

void System::add_block(const std::string& name, const ModelSpec& model, const std::string& limiter,
                       const std::string& riemann, const std::string& recon,
                       const std::string& time, int substeps, bool evolve, int stride,
                       const std::vector<std::string>& implicit_vars,
                       const std::vector<std::string>& implicit_roles, const NewtonOptions& newton,
                       bool newton_diagnostics, double positivity_floor, bool wave_speed_cache,
                       double weno_epsilon) {
  Impl* P = p_.get();
  require_assembling(P->lifecycle_, "add_block");  // frozen once pops.bind completes (ADC-592)
  // Completeness contract of the model (ADC-290): transport / elliptic must be chosen explicitly.
  // Validated HERE, before the transport string routing below (which would otherwise report a
  // cryptic "unknown transport ''" for an unset tag) -- a default-constructed ModelSpec no longer
  // means a silent Euler + Poisson-charge composition.
  detail::validate_model_spec(model);
  if (substeps < 1)
    throw std::runtime_error("System::add_block : substeps >= 1");
  if (stride < 1)
    throw std::runtime_error("System::add_block : stride >= 1");
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::runtime_error("System::add_block : positivity_floor >= 0 and finite (0 = inactive)");
  // Validation of the NEWTON OPTIONS POD (ADC-214): range check shared with AmrSystem::add_block
  // (validate_newton_options, in implicit_stepper.hpp). Whether non-default options are ALLOWED
  // (the time='imex' gate below) stays here -- it differs from the AMR path.
  validate_newton_options(newton, "System::add_block");
  // @p time carries the TREATMENT and, in explicit, the RK SCHEME: "explicit"/"ssprk2" = SSPRK2
  // (historical default), "ssprk3" = SSPRK3 (order 3), "euler" = ForwardEuler (order 1, fidelity to
  // first-order references -- validation), "imex" = explicit transport + local backward-Euler implicit
  // stiff source (order 1), "imexrk_ars222" = IMEX-RK family scheme ARS(2,2,2)
  // (order 2, distinct PARALLEL advance, Cartesian only). The RK math stays a CORE FUNCTOR
  // (build_block). "imex" and "imexrk_ars222" share the @c imex flag; @c method distinguishes them.
  if (time != "explicit" && time != "ssprk2" && time != "ssprk3" && time != "euler" &&
      time != "imex" && time != "imexrk_ars222")
    throw std::runtime_error(
        "System::add_block : time 'explicit'|'ssprk2'|'ssprk3'|'euler'|'imex'|'imexrk_ars222' "
        "(received '" +
        time + "')");
  if (recon != "conservative" && recon != "primitive")
    throw std::runtime_error("System::add_block : recon 'conservative' | 'primitive' (received '" +
                             recon + "')");
  const bool imexrk = (time == "imexrk_ars222");
  const bool imex = (time == "imex" || imexrk);  // both go through the implicit source step
  const bool recon_prim = (recon == "primitive");
  // Wave speed cache (opt-in): only engages for the HLL flux and the explicit advance. Requesting it
  // elsewhere would be SILENTLY without effect -> explicit error (no silent ignore). The polar path has
  // its own factory (make_block_polar) without this cache.
  if (wave_speed_cache) {
    if (riemann != "hll")
      throw std::runtime_error(
          "System::add_block : wave_speed_cache requires riemann='hll' (the wave "
          "speed cache only applies to the HLL flux ; received riemann='" +
          riemann + "')");
    if (imex)
      throw std::runtime_error("System::add_block : wave_speed_cache not supported with time='" +
                               time +
                               "' (wired on the explicit advance ; use time "
                               "'explicit'/'ssprk2'/'ssprk3'/'euler')");
    if (P->polar_)
      throw std::runtime_error(
          "System::add_block : wave_speed_cache not supported on the polar "
          "geometry (ring)");
    // EMBEDDED-BOUNDARY transport mode already active: the stepper routes to advance_masked /
    // advance_eb, which do not carry the cache -> requesting it would be WITHOUT EFFECT. Explicit
    // rejection (no silent ignore). The reverse order (set_disc_domain AFTER a cached block) is
    // rejected by set_disc_domain / set_geometry_mode.
    if (P->eb_set_ && P->geometry_mode_ != GeometryMode::None)
      throw std::runtime_error(
          "System::add_block : wave_speed_cache incompatible with an active "
          "embedded-boundary transport mode (staircase/cutcell) ; the cache is only "
          "wired on the full Cartesian advance (remove wave_speed_cache or mode='none')");
    P->ws_cache_block_ = true;  // a block requested the cache -> locks the switch to disc mode
  }
  // The EXPLICIT RK scheme threaded to build_block: the ONE canonical spelling of the typed route
  // (ADC-641), replacing the imexrk?...:(ssprk3?...) string ladder. build_block decodes it once via
  // parse_time_route ("explicit" resolves to the SSPRK2 advance, "imex" is ignored past the imex flag),
  // so the advance selected is bit-identical.
  const std::string method = route_token(parse_time_route(time, "System::add_block"));
  // The implicit mask (implicit_vars / implicit_roles) applies only to the IMEX source step. Requesting
  // it in explicit is an ERROR (no silent ignore): the explicit has no implicit step.
  if (!imex && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error(
        "System::add_block : implicit_vars / implicit_roles require time='imex' "
        "(the implicit mask applies only to the IMEX source step ; received time='" +
        time + "')");
  // IMEX-RK ARS(2,2,2): FULLY implicit source (the stage consistency relation assumes a homogeneous
  // solve). A partial mask would be SILENTLY ignored there -> we reject it explicitly. The
  // partial mask stays available on time='imex' (local backward-Euler).
  if (imexrk && (!implicit_vars.empty() || !implicit_roles.empty()))
    throw std::runtime_error(
        "System::add_block : implicit_vars / implicit_roles (partial IMEX mask) unsupported by "
        "time='imexrk_ars222' (its source is FULLY implicit). Use time='imex' for a "
        "partial mask, or remove implicit_vars / implicit_roles.");
  // Same rules for the Newton options/diagnostics: they only drive the IMEX source step.
  // Non-default values in explicit would be SILENTLY ignored -> explicit error.
  const bool newton_non_default = newton_options_non_default(newton, newton_diagnostics);
  if (!imex && newton_non_default)
    throw std::runtime_error(
        "System::add_block : the Newton options (newton_max_iters/rel_tol/"
        "abs_tol/fd_eps/diagnostics) require time='imex' (received time='" +
        time + "')");

  // ADC-645: the WENO-Z regulariser. Only meaningful with limiter='weno5'; on any other limiter a
  // non-default value would be silently ignored -> refuse loud. The POLAR path keeps the default
  // Weno5 (its builder is not threaded); refuse a non-default eps there too rather than drop it.
  if (weno_epsilon <= 0.0)
    throw std::runtime_error("System::add_block : weno_epsilon > 0 required");
  if (weno_epsilon != static_cast<double>(kWenoEpsilon)) {
    if (limiter != "weno5")
      throw std::runtime_error(
          "System::add_block : weno_epsilon applies to limiter='weno5' only (received limiter='" +
          limiter + "')");
    if (P->polar_)
      throw std::runtime_error(
          "System::add_block : weno_epsilon is wired on the cartesian path only (the polar "
          "builder keeps the default kWenoEpsilon; wiring it is a follow-up)");
    // The masked/EB advances keep the default-constructed Weno5 (mirror of the wave_speed_cache
    // guard above): requesting a non-default eps with an active disc transport mode would be
    // WITHOUT EFFECT on those closures -> explicit rejection, never a silent drop.
    if (P->eb_set_ && P->geometry_mode_ != GeometryMode::None)
      throw std::runtime_error(
          "System::add_block : weno_epsilon incompatible with an active embedded-boundary "
          "transport mode (staircase/cutcell) ; it is only wired on the full Cartesian advance "
          "(leave weno_epsilon default or mode='none')");
  }

  int ncomp = 1;
  BlockClosures clo;
  std::function<Real(const MultiFab&)> max_speed;
  std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;
  std::function<Real(const MultiFab&)> src_freq, stab_dt;  // optional step bounds (model traits)
  CellConvert prim_to_cons, cons_to_prim;  // pointwise model conversions (set/get_primitive_state)
  VariableSet cons_vs, prim_vs;
  detail::BuiltBlock bb;
  if (P->polar_) {
    // POLAR PATH (ring): closures built by block_builder_polar.hpp (assemble_rhs_polar + scalar polar
    // transport ExBVelocityPolar OR fluid IsothermalFluxPolar + scalar polar Poisson), via the polar
    // seam (python/system_polar.cpp, ADC-335). IMEX is not supported on the ring at this stage: the
    // electrostatic coupling goes through an explicit LOCAL source (non-stiff regime, Path A step 1);
    // we reject it explicitly rather than silently running the transport alone.
    if (imex)
      throw std::runtime_error(
          "System::add_block (polar) : time='" + time +
          "' (IMEX / IMEX-RK ARS(2,2,2)) unsupported "
          "(ring : coupling by explicit local source, no stiff source to handle implicitly "
          "at this stage). Use 'explicit'/'ssprk2'/'ssprk3'.");
    const PolarGridContext pctx = P->grid_ctx_polar();
    bb = detail::build_block_polar(model, limiter, riemann, pctx, recon_prim, method,
                                   static_cast<Real>(positivity_floor), &P->aux);
    // ADC-291: widen the shared aux to the polar block's read width (canonical extras AND model-named
    // extra[k]), mirroring the Cartesian branch below. ensure_aux_width keeps the aux ADDRESS captured
    // by the closures and re-applies B_z / named aux on realloc; without it a polar n_aux>3 model read
    // out of bounds. No-op for a base (n_aux=3) model -> bit-identical.
    P->ensure_aux_width(bb.aux_width);
  } else {
    const GridContext ctx = P->grid_ctx();
    // Newton options of the IMEX implicit source (defaults = historical constants, bit-identical).
    // The report lives in diagnostics_.newton_reports in a shared_ptr -> STABLE address captured by
    // the closures even when the map reallocates at a later add_block. It is allocated for explicit
    // diagnostics and for fail_policy warn/throw, because those policies must surface as structured
    // report events rather than stderr text.
    const NewtonOptions& nopts = newton;
    NewtonReport* nreport = nullptr;
    if (newton_diagnostics || nopts.fail_policy != NewtonOptions::kFailNone) {
      auto rep = std::make_shared<NewtonReport>();
      P->diagnostics_.newton_reports[name] = rep;
      nreport = rep.get();
    }
    // Transport-axis seam (ADC-335): each per-transport TU (python/system_<transport>.cpp) runs the
    // SAME source/elliptic dispatch + make_block + makers as before (detail::build_block_for), but
    // instantiates ONLY its own transport's leaves -- so the combinatorial product splits across files
    // for `-j`. This string if/else mirrors detail::dispatch_transport (same unknown-transport message).
    // aux_width is widened host-side AFTER the build (was P->ensure_aux_width inside the visitor;
    // ensure_aux_width keeps the aux ADDRESS captured by the closures, so order vs make_block is
    // immaterial -- byte-identical).
    const detail::BlockBuildArgs args{name,
                                      limiter,
                                      riemann,
                                      ctx,
                                      imex,
                                      recon_prim,
                                      method,
                                      implicit_vars,
                                      implicit_roles,
                                      nopts,
                                      nreport,
                                      static_cast<Real>(positivity_floor),
                                      wave_speed_cache,
                                      static_cast<Real>(weno_epsilon)};
    // Transport dispatch mirrors detail::dispatch_transport (ADC-641): validate_transport preserves the
    // unknown_transport_msg byte-for-byte, then the switch on the typed TransportRouteId routes to the
    // per-transport seam. Every case terminates (assigns bb); the compressible/isothermal flux ladders
    // are their own switch on parse_riemann_route (default -> the registry/dispatch guard).
    validate_transport(model.transport);
    switch (parse_transport_route(model.transport)) {
      case TransportRouteId::kExb:
        bb = detail::build_block_exb(model, args);
        break;
      case TransportRouteId::kCompressible: {
        // Compressible/Euler is flux-subdivided (ADC-335): all four fluxes are valid (4-var + pressure),
        // so we run the SAME validation as make_block (validate_riemann then validate_limiter, identical
        // messages) and dispatch the riemann route to the matching per-flux sub-TU. An unknown flux hits
        // the same registry throw as make_block's tail (validate_riemann already rejected it).
        validate_riemann(riemann, /*polar=*/false, "System");
        validate_limiter(limiter, "System");
        switch (parse_riemann_route(riemann, "System")) {
          case RiemannRouteId::kRusanov:
            bb = detail::build_block_compressible_rusanov(model, args);
            break;
          case RiemannRouteId::kHll:
            bb = detail::build_block_compressible_hll(model, args);
            break;
          // On the true Euler brick the EXPLICIT euler_hllc route and the generic hllc route are the
          // SAME arithmetic (the native Euler now provides HasHLLCStructure with the canonical-Euler
          // formulas, so HLLCFlux == the former EulerHLLCFlux2D fallback bit-for-bit): both share this
          // seam leaf (ADC-590). euler_hllc's 4-var+pressure gate is satisfied by CompressibleFlux.
          case RiemannRouteId::kHllc:
          case RiemannRouteId::kEulerHllc:
            bb = detail::build_block_compressible_hllc(model, args);
            break;
          case RiemannRouteId::kRoe:
          case RiemannRouteId::kEulerRoe:
            bb = detail::build_block_compressible_roe(model, args);
            break;
          default:
            throw_registry_dispatch_mismatch("System", "flux", riemann);
        }
        break;
      }
      case TransportRouteId::kIsothermal: {
        // Isothermal is flux-subdivided (ADC-342): only rusanov + hll are reachable (3-var, no pressure
        // for hllc/roe). The per-flux seams call make_block_<flux> directly, so -- like compressible --
        // we run make_block's validation here (validate_riemann then validate_limiter, identical
        // messages) before dispatching; hllc/roe and any unknown flux hit the registry throw (explicit,
        // no UB). The default preserves isothermal+hllc -> registry-mismatch throw exactly.
        validate_riemann(riemann, /*polar=*/false, "System");
        validate_limiter(limiter, "System");
        switch (parse_riemann_route(riemann, "System")) {
          case RiemannRouteId::kRusanov:
            bb = detail::build_block_isothermal_rusanov(model, args);
            break;
          case RiemannRouteId::kHll:
            bb = detail::build_block_isothermal_hll(model, args);
            break;
          default:
            throw_registry_dispatch_mismatch("System", "flux", riemann);
        }
        break;
      }
    }
    P->ensure_aux_width(bb.aux_width);
  }
  ncomp = bb.ncomp;
  cons_vs = std::move(bb.cons_vs);
  prim_vs = std::move(bb.prim_vs);
  clo = std::move(bb.clo);
  max_speed = std::move(bb.max_speed);
  add_poisson_rhs = std::move(bb.add_poisson_rhs);
  src_freq = std::move(bb.src_freq);
  stab_dt = std::move(bb.stab_dt);
  prim_to_cons = std::move(bb.prim_to_cons);
  cons_to_prim = std::move(bb.cons_to_prim);
  // Common installation (same path as add_compiled_model for a DSL-generated model):
  // the closures run on the REAL System MultiFabs (MPI halos via fill_boundary, device
  // via Kokkos), without copy.
  install_block(name, ncomp, cons_vs, prim_vs, model.gamma, std::move(clo), std::move(max_speed),
                std::move(add_poisson_rhs), substeps, evolve, stride);
  EffectiveBlockOptions block_options =
      make_system_block_options(name, model, "native_model", limiter, riemann, recon, time, method,
                                imex, substeps, evolve, stride, implicit_vars, implicit_roles,
                                newton, newton_diagnostics, positivity_floor, wave_speed_cache);
  block_options.ncomp = ncomp;
  block_options.conservative_vars = cons_vs.names;
  block_options.primitive_vars = prim_vs.names;
  P->diagnostics_.block_options[name] = std::move(block_options);
  set_block_conversion(name, std::move(prim_to_cons), std::move(cons_to_prim));
  set_block_dt_bounds(name, std::move(src_freq), std::move(stab_dt));
  // SCHEME GHOSTS: WENO5 reads a 5-point stencil (3 ghosts) > the 2 allocated by default in
  // install_block. We reallocate the block state with block_n_ghost(limiter) if needed (cf. AmrSystem which
  // allocates with Limiter::n_ghost, PR #22) so that fill_ghosts + assemble_rhs do not read out of
  // bounds. minmod/vanleer (2 ghosts): no-op, allocation and result bit-identical to before.
  P->set_block_ghosts(name, block_n_ghost(limiter));
}

// Real grid context (mesh + BC + aux): used by the add_compiled_model template to build
// the closures of an AOT-compiled model on the real System fields (native parity, without marshaling).
POPS_EXPORT GridContext System::grid_context() {
  return p_->grid_ctx();
}

// Installs a block from already-built closures (by dispatch_model on the add_block side, or by
// block_builder on the add_compiled_model side). Centralizes the creation of the species (U, names, scheme).
POPS_EXPORT void System::install_block(const std::string& name, int ncomp,
                                      const VariableSet& cons_vars, const VariableSet& prim_vars,
                                      double gamma, BlockClosures closures,
                                      std::function<Real(const MultiFab&)> max_speed,
                                      std::function<void(const MultiFab&, MultiFab&)> poisson_rhs,
                                      int substeps, bool evolve, int stride) {
  if (stride < 1)
    throw std::runtime_error("System::install_block : stride >= 1");
  Impl* P = p_.get();
  P->sp.push_back(Impl::Species{name, MultiFab(P->ba, P->dm, ncomp, 2), ncomp, substeps, evolve,
                                stride, gamma, std::move(closures.advance),
                                std::move(closures.rhs_into), std::move(max_speed),
                                std::move(poisson_rhs)});
  P->sp.back().U.set_val(Real(0));
  P->sp.back().cons_vars = cons_vars;
  P->sp.back().prim_vars = prim_vars;
  // EMBEDDED-BOUNDARY transport advances (project T5-PR3): empty unless build_block built them
  // (Cartesian block with domain_mask_/eb_domain_ provided). Empty -> the stepper falls back on advance
  // (bit-identical).
  P->sp.back().advance_masked = std::move(closures.advance_masked);
  P->sp.back().advance_eb = std::move(closures.advance_eb);
  P->sp.back().hotspot = std::move(closures.hotspot);  // dt_hotspot diagnostic (ADC-182)
  // Projection ponctuelle post-pas (ADC-177) : vide sauf si le modele declare le trait
  // HasPointwiseProjection (make_block). Vide -> le stepper ne l'interroge pas (bit-identique).
  P->sp.back().project = std::move(closures.project);
  // FLUX-ONLY residual -div F(U) (ADC-425): set for native blocks (build_block builds it via
  // SourceFreeModel<Model>); empty for paths that do not (the host .so prototype loader) ->
  // block_neg_div_flux_into fails loud rather than silently leaking the default source.
  P->sp.back().rhs_flux_only = std::move(closures.rhs_flux_only);
  // SOURCE-ONLY residual S(U, aux) (ADC-430): set for native blocks (build_block builds it via
  // SourceInto<Model>); empty for paths that do not (the host .so prototype loader) ->
  // block_source_into fails loud rather than silently leaking the flux.
  P->sp.back().source_only = std::move(closures.source_only);
  EffectiveBlockOptions& opt = P->diagnostics_.block_options[name];
  opt.name = name;
  if (opt.route.empty())
    opt.route = "closure_install";
  opt.ncomp = ncomp;
  opt.n_ghost = P->sp.back().U.n_grow();
  opt.substeps = substeps;
  opt.stride = stride;
  opt.evolve = evolve;
  opt.gamma = gamma;
  opt.conservative_vars = cons_vars.names;
  opt.primitive_vars = prim_vars.names;
}

// Width-aware reallocation of a block state (delegates to Impl::set_block_ghosts). Exposed
// (POPS_EXPORT) so that the add_compiled_model header template (native path, .so loader) can
// widen the compiled block to block_n_ghost(limiter) -- 3 for weno5 -- as add_block does.
POPS_EXPORT void System::set_block_ghosts(const std::string& name, int n_ghost) {
  p_->set_block_ghosts(name, n_ghost);
  if (EffectiveBlockOptions* opt = p_->diagnostics_.block_options_ptr(name))
    opt->n_ghost = p_->find(name).U.n_grow();
}

// OPTIONAL step bounds of a block (model traits): set after install_block, read by
// step_cfl / step_adaptive. Empty functions = the block imposes no bound (historical).
void System::set_block_dt_bounds(const std::string& name,
                                 std::function<Real(const MultiFab&)> source_frequency,
                                 std::function<Real(const MultiFab&)> stability_dt) {
  Impl::Species& s = p_->find(name);  // raises if unknown block
  s.source_frequency = std::move(source_frequency);
  s.stability_dt = std::move(stability_dt);
}

// GLOBAL step bound (host, one evaluation per step): multi-block coupling, Schur/Poisson,
// scheduler, user policy. cf. SystemStepper::step_cfl for the aggregation.
void System::add_dt_bound(const std::string& label, std::function<double()> fn) {
  require_assembling(p_->lifecycle_, "add_dt_bound");  // frozen once pops.bind completes (ADC-592)
  if (!fn)
    throw std::runtime_error("System::add_dt_bound : empty bound function");
  p_->dt_bounds_.push_back(Impl::GlobalDtBound{label, std::move(fn)});
}

// ACTIVE bound of the last step_cfl (step-policy diagnostic). "" before the first step.
std::string System::last_dt_bound() const {
  return p_->stepper_.last_dt_reason();
}

// dt_hotspot diagnostic (ADC-182): the GLOBAL cell (i, j) that dominates the transport CFL
// bound of block @p name, and its speed w = max(wx, wy). ON DEMAND (two reduction
// passes, cf. max_wave_speed_hotspot_mf) -- step/step_cfl do not touch it. Block without
// closure (historical non-rewireable paths, e.g. dynamic) -> EXPLICIT error.
std::array<double, 3> System::dt_hotspot(const std::string& name) {
  Impl::Species& s = p_->find(name);
  if (!s.hotspot)
    throw std::runtime_error("System::dt_hotspot : block '" + name +
                             "' without hotspot diagnostic (non-rewireable add path)");
  Real w = 0;
  int i = -1, j = -1;
  s.hotspot(s.U, w, i, j);
  return {static_cast<double>(w), static_cast<double>(i), static_cast<double>(j)};
}

// Newton report (OPT-IN IMEX diagnostics) of the block: flat copy of the NewtonReport aggregated by the
// LAST advance of the block (reset at the start of the advance by AdvanceImex*). Clear error if the block did
// not enable newton_diagnostics (no silently empty report).
System::SourceNewtonReport System::newton_report(const std::string& name) const {
  p_->index(name);  // raises if unknown block
  const NewtonReport* rp = p_->diagnostics_.newton_report_ptr(name);
  if (rp == nullptr)
    throw std::runtime_error(
        "System::newton_report : Newton diagnostics not enabled for block '" + name +
        "' ; add the block with newton_diagnostics=true "
        "(pops.IMEX(newton_diagnostics=True) / pops.SourceImplicit(newton_diagnostics=True)) "
        "or newton_fail_policy='warn'/'throw'");
  const NewtonReport& r = *rp;
  return SourceNewtonReport{r.enabled,
                            r.converged,
                            static_cast<double>(r.max_residual),
                            static_cast<double>(r.max_iters_used),
                            r.n_failed,
                            r.failed_i,
                            r.failed_j,
                            r.failed_comp,
                            r.diagnostics.events};
}

// Body EXTRACTED VERBATIM into pops::native_loader::add_dynamic_block (native_loader.hpp); instantiated
// here with System::Impl (defined above, private to this TU). Bit-identical: pure delegation.
void System::add_dynamic_block(const std::string& name, const std::string& so_path, int substeps,
                               const std::vector<std::string>& names, const std::string& recon) {
  require_assembling(p_->lifecycle_, "add_dynamic_block");  // frozen once pops.bind completes (ADC-592)
  native_loader::add_dynamic_block(this, p_.get(), name, so_path, substeps, names, recon);
  EffectiveBlockOptions& opt = p_->diagnostics_.block_options[name];
  opt.route = "dynamic_loader";
  opt.compiled = false;
  opt.transport = "dynamic_model";
  opt.source = "dynamic_model";
  opt.elliptic = "dynamic_model";
  opt.limiter = recon;
  opt.riemann = "rusanov_global";
  opt.recon = recon;
  opt.time = "explicit";
  opt.time_method = "host_euler";
}

// Body EXTRACTED VERBATIM into pops::native_loader::add_compiled_block (native_loader.hpp); instantiated
// here with System::Impl. Bit-identical: pure delegation.
void System::add_compiled_block(const std::string& name, const std::string& so_path,
                                const std::string& limiter, const std::string& riemann,
                                const std::string& recon, const std::string& time, int substeps,
                                const std::vector<std::string>& names, double positivity_floor) {
  require_assembling(p_->lifecycle_, "add_compiled_block");  // frozen once pops.bind completes (ADC-592)
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::runtime_error(
        "System::add_compiled_block : positivity_floor >= 0 and finite (0 = inactive)");
  native_loader::add_compiled_block(this, p_.get(), name, so_path, limiter, riemann, recon, time,
                                    substeps, names, positivity_floor);
  EffectiveBlockOptions& opt = p_->diagnostics_.block_options[name];
  opt.route = "aot_loader";
  opt.compiled = true;
  opt.transport = "compiled_artifact";
  opt.source = "compiled_artifact";
  opt.elliptic = "compiled_artifact";
  opt.limiter = limiter;
  opt.riemann = riemann;
  opt.recon = recon;
  opt.time = time;
  opt.time_method = time;
  opt.imex = (time == "imex");
  opt.substeps = substeps;
  opt.positivity_floor = positivity_floor;
}

// P7-b: overwrites the SHARED vector of runtime parameter values of block @p name. add_compiled_block
// registered this vector in p_->block_params_ AND captured it in the block closures: writing
// into it suffices to change the behavior at the next step, WITHOUT recompiling the .so. Explicit error if
// the block has no runtime params (vector absent) or if values does not have the right size.
void System::set_block_params(const std::string& name, const std::vector<double>& values) {
  // index() raises "System: unknown block '...'" if the block does not exist (same diagnostic as everywhere).
  (void)p_->blocks_.index(name);
  auto it = p_->block_params_.find(name);
  if (it == p_->block_params_.end())
    throw std::runtime_error(
        "System::set_block_params : block '" + name +
        "' has no runtime parameter (declare dsl.Param(..., kind='runtime') and wire via "
        "backend='aot' / add_compiled_block ; const params are frozen at compile time)");
  std::vector<double>& pv = *it->second;
  if (values.size() != pv.size())
    throw std::runtime_error("System::set_block_params : block '" + name + "' expects " +
                             std::to_string(pv.size()) + " runtime parameters, received " +
                             std::to_string(values.size()));
  pv = values;  // the vector is SHARED with the closures (shared_ptr): effect at the next step
}

// Body EXTRACTED VERBATIM into pops::native_loader::add_native_block (native_loader.hpp); instantiated
// here with System::Impl. Bit-identical: pure delegation (this marshals to the unchanged native loader).
void System::add_native_block(const std::string& name, const std::string& so_path,
                              const std::string& limiter, const std::string& riemann,
                              const std::string& recon, const std::string& time, double gamma,
                              int substeps, bool evolve, int stride, double positivity_floor) {
  require_assembling(p_->lifecycle_, "add_native_block");  // frozen once pops.bind completes (ADC-592)
  if (!(positivity_floor >= 0.0) || !std::isfinite(positivity_floor))
    throw std::runtime_error(
        "System::add_native_block : positivity_floor >= 0 and finite (0 = inactive)");
  native_loader::add_native_block(this, p_.get(), name, so_path, limiter, riemann, recon, time,
                                  gamma, substeps, evolve, stride, positivity_floor);
  EffectiveBlockOptions& opt = p_->diagnostics_.block_options[name];
  opt.route = "native_loader";
  opt.compiled = true;
  opt.transport = "compiled_artifact";
  opt.source = "compiled_artifact";
  opt.elliptic = "compiled_artifact";
  opt.limiter = limiter;
  opt.riemann = riemann;
  opt.recon = recon;
  opt.time = time;
  // The canonical spelling of the typed route (ADC-641), replacing the if (time=="imex")...else "ssprk2"
  // string ladder. Diagnostic-only (EffectiveBlockOptions.time_method); the advance is selected by the
  // native loader from @p time itself.
  opt.time_method = route_token(parse_time_route(time, "System::add_native_block"));
  opt.imex = (time == "imex");
  opt.substeps = substeps;
  opt.stride = stride;
  opt.evolve = evolve;
  opt.gamma = gamma;
  opt.positivity_floor = positivity_floor;
}

void System::set_poisson(const std::string& rhs, const std::string& solver, const std::string& bc,
                         const std::string& wall, double wall_radius, double epsilon, double abs_tol,
                         double rel_tol, int max_cycles, int min_coarse, int pre_smooth,
                         int post_smooth, int bottom_sweeps, int coarse_threshold) {
  require_assembling(p_->lifecycle_, "set_poisson");  // frozen once pops.bind completes (ADC-592)
  if (epsilon == 0.0)
    throw std::runtime_error("System::set_poisson : epsilon != 0 required");
  if (abs_tol < 0.0)
    throw std::runtime_error("System::set_poisson : abs_tol >= 0 required");
  // ADC-613: the GeometricMG V-cycle knobs. Refuse out-of-domain values STRUCTURALLY here (the
  // Python descriptor already refuses, but the native seam is a public API in its own right and
  // must never silently accept a degenerate cycle). Defaults are the kMG* constants -> historical.
  if (rel_tol <= 0.0)
    throw std::runtime_error("System::set_poisson : rel_tol > 0 required");
  if (max_cycles < 1)
    throw std::runtime_error("System::set_poisson : max_cycles >= 1 required");
  if (min_coarse < 1)
    throw std::runtime_error("System::set_poisson : min_coarse >= 1 required");
  if (pre_smooth < 0 || post_smooth < 0 || bottom_sweeps < 0)
    throw std::runtime_error("System::set_poisson : pre_smooth/post_smooth/bottom_sweeps >= 0 "
                             "required");
  // ADC-644: the total-cell coarsening ceiling. 0 (the default sentinel) = disabled (only min_coarse
  // governs); a positive value stops coarsening at that unknown count. Negative is refused.
  if (coarse_threshold < 0)
    throw std::runtime_error("System::set_poisson : coarse_threshold >= 0 required (0 = disabled)");
  p_->fields_.p_rhs = rhs;
  p_->fields_.p_solver = solver;
  p_->fields_.p_bc = bc;
  p_->fields_.p_has_explicit_bc = false;
  p_->fields_.p_nullspace_const = false;
  p_->fields_.p_mean_zero_gauge = false;
  p_->fields_.p_wall = wall;
  p_->fields_.p_wall_radius = wall_radius;
  p_->fields_.p_eps_ = static_cast<Real>(epsilon);
  p_->fields_.p_abs_tol_ =
      static_cast<Real>(abs_tol);  // absolute floor of the V-cycle (0 = relative only)
  // Resolve the V-cycle knobs into the options POD the field solver forwards to GeometricMG (ctor
  // args + solve(rel, cyc, abs)). abs_tol feeds both p_abs_tol_ (the pre-613 field) and the POD.
  p_->fields_.p_mg_opts_.rel_tol = static_cast<Real>(rel_tol);
  p_->fields_.p_mg_opts_.abs_tol = static_cast<Real>(abs_tol);
  p_->fields_.p_mg_opts_.max_cycles = max_cycles;
  p_->fields_.p_mg_opts_.min_coarse = min_coarse;
  p_->fields_.p_mg_opts_.nu1 = pre_smooth;
  p_->fields_.p_mg_opts_.nu2 = post_smooth;
  p_->fields_.p_mg_opts_.nbottom = bottom_sweeps;
  p_->fields_.p_mg_opts_.coarse_threshold = coarse_threshold;  // ADC-644: total-cell coarsening ceiling.
  p_->fields_.ell_.reset();
}

void System::set_field_solver_plan(const std::string& provider_slot,
                                   const std::string& provider_identity,
                                   const std::string& output_owner_identity,
                                   const std::string& output_block,
                                   const std::string& output_key,
                                   const std::vector<std::string>& provider_identities,
                                   const std::vector<std::string>& provider_blocks,
                                   const std::vector<std::string>& provider_keys,
                                   const std::vector<double>& provider_coefficients,
                                   const std::string& solver,
                                   double abs_tol, double rel_tol,
                                   int max_cycles, int min_coarse, int pre_smooth,
                                   int post_smooth, int bottom_sweeps, int coarse_threshold) {
  require_assembling(p_->lifecycle_, "set_field_solver_plan");
  if (provider_slot.empty() || provider_identity.empty() || output_owner_identity.empty() ||
      output_block.empty() ||
      output_key.empty())
    throw std::runtime_error("System::set_field_solver_plan requires a qualified provider identity");
  const std::size_t provider_count = provider_identities.size();
  if (provider_count == 0 || provider_blocks.size() != provider_count ||
      provider_keys.size() != provider_count || provider_coefficients.size() != provider_count)
    throw std::runtime_error("System::set_field_solver_plan invalid provider-pack shape");
  for (std::size_t i = 0; i < provider_count; ++i)
    if (provider_identities[i].empty() || provider_blocks[i].empty() || provider_keys[i].empty() ||
        !std::isfinite(provider_coefficients[i]))
      throw std::runtime_error("System::set_field_solver_plan invalid provider-pack entry");
  if (solver != "geometric_mg" && solver != "fft" && solver != "fft_spectral")
    throw std::runtime_error("System::set_field_solver_plan unknown solver '" + solver + "'");
  if (abs_tol < 0.0 || rel_tol <= 0.0 || max_cycles < 1 || min_coarse < 1 ||
      pre_smooth < 0 || post_smooth < 0 || bottom_sweeps < 0 || coarse_threshold < 0)
    throw std::runtime_error("System::set_field_solver_plan invalid multigrid options");
  const auto existing = p_->fields_.named_field_plans_.find(provider_slot);
  if (existing != p_->fields_.named_field_plans_.end() &&
      existing->second.provider_identity != provider_identity)
    throw std::runtime_error("System::set_field_solver_plan provider digest collision");
  auto& plan = p_->fields_.named_field_plans_[provider_slot];
  plan.provider_identity = provider_identity;
  plan.output_owner_identity = output_owner_identity;
  plan.output_block = output_block;
  plan.output_key = output_key;
  plan.providers.clear();
  plan.providers.reserve(provider_count);
  for (std::size_t i = 0; i < provider_count; ++i)
    plan.providers.push_back({provider_identities[i], provider_blocks[i], provider_keys[i],
                              static_cast<Real>(provider_coefficients[i])});
  plan.solver = solver;
  plan.mg_opts.rel_tol = static_cast<Real>(rel_tol);
  plan.mg_opts.abs_tol = static_cast<Real>(abs_tol);
  plan.mg_opts.max_cycles = max_cycles;
  plan.mg_opts.min_coarse = min_coarse;
  plan.mg_opts.nu1 = pre_smooth;
  plan.mg_opts.nu2 = post_smooth;
  plan.mg_opts.nbottom = bottom_sweeps;
  plan.mg_opts.coarse_threshold = coarse_threshold;
  auto registered = p_->fields_.named_fields_.find(provider_slot);
  if (registered != p_->fields_.named_fields_.end()) {
    registered->second.has_plan = true;
    registered->second.plan = plan;
    registered->second.prepared_providers.clear();
    registered->second.ell.reset();
  }
}

void System::set_field_boundary_plan(const std::string& provider_slot,
                                     const std::vector<std::string>& kind,
                                     const std::vector<double>& alpha,
                                     const std::vector<double>& beta,
                                     const std::vector<double>& value) {
  require_assembling(p_->lifecycle_, "set_field_boundary_plan");
  if (kind.size() != 4 || alpha.size() != 4 || beta.size() != 4 || value.size() != 4)
    throw std::runtime_error(
        "System::set_field_boundary_plan requires four xlo/xhi/ylo/yhi entries");
  BCRec bc;
  bc.dx = p_->geom.dx();
  bc.dy = p_->geom.dy();
  BCType* types[] = {&bc.xlo, &bc.xhi, &bc.ylo, &bc.yhi};
  Real* vals[] = {&bc.xlo_val, &bc.xhi_val, &bc.ylo_val, &bc.yhi_val};
  Real* alphas[] = {&bc.xlo_alpha, &bc.xhi_alpha, &bc.ylo_alpha, &bc.yhi_alpha};
  Real* betas[] = {&bc.xlo_beta, &bc.xhi_beta, &bc.ylo_beta, &bc.yhi_beta};
  for (int face = 0; face < 4; ++face) {
    const Real a = static_cast<Real>(alpha[face]);
    const Real b = static_cast<Real>(beta[face]);
    const Real v = static_cast<Real>(value[face]);
    if (!std::isfinite(alpha[face]) || !std::isfinite(beta[face]) ||
        !std::isfinite(value[face]) || (a == Real(0) && b == Real(0) && kind[face] != "periodic"))
      throw std::runtime_error("System::set_field_boundary_plan invalid Robin coefficients");
    if (kind[face] == "periodic") {
      *types[face] = BCType::Periodic;
    } else if (kind[face] == "dirichlet" || (kind[face] == "mixed" && b == Real(0))) {
      if (a == Real(0))
        throw std::runtime_error("System::set_field_boundary_plan Dirichlet alpha is zero");
      *types[face] = BCType::Dirichlet;
      *vals[face] = v / a;
    } else if (kind[face] == "neumann" && v == Real(0)) {
      *types[face] = BCType::Foextrap;
    } else if (kind[face] == "neumann" || kind[face] == "mixed") {
      *types[face] = BCType::Robin;
      *vals[face] = v;
      *alphas[face] = a;
      *betas[face] = b;
      const Real h = face < 2 ? bc.dx : bc.dy;
      if (a / Real(2) + b / h == Real(0))
        throw std::runtime_error(
            "System::set_field_boundary_plan singular cell-centred Robin denominator");
    } else {
      throw std::runtime_error("System::set_field_boundary_plan unknown kind '" + kind[face] + "'");
    }
  }
  auto plan_it = p_->fields_.named_field_plans_.find(provider_slot);
  if (plan_it == p_->fields_.named_field_plans_.end())
    throw std::runtime_error("System::set_field_boundary_plan unknown provider slot");
  auto& plan = plan_it->second;
  plan.explicit_bc = bc;
  plan.has_explicit_bc = true;
  auto registered = p_->fields_.named_fields_.find(provider_slot);
  if (registered != p_->fields_.named_fields_.end()) {
    registered->second.has_plan = true;
    registered->second.plan = plan;
    registered->second.ell.reset();
  }
}

void System::set_field_boundary_dependencies(
    const std::string& provider_slot, const std::vector<std::string>& state_blocks,
    const std::vector<int>& state_components,
    const std::vector<std::string>& field_blocks,
    const std::vector<std::string>& field_keys,
    const std::vector<int>& field_components) {
  require_assembling(p_->lifecycle_, "set_field_boundary_dependencies");
  if (state_blocks.size() != state_components.size() ||
      field_blocks.size() != field_keys.size() || field_blocks.size() != field_components.size())
    throw std::runtime_error("System::set_field_boundary_dependencies pack shape mismatch");
  const auto invalid_text = [](const auto& value) { return value.empty(); };
  const auto invalid_component = [](int value) { return value < 0; };
  if (std::any_of(state_blocks.begin(), state_blocks.end(), invalid_text) ||
      std::any_of(field_blocks.begin(), field_blocks.end(), invalid_text) ||
      std::any_of(field_keys.begin(), field_keys.end(), invalid_text) ||
      std::any_of(state_components.begin(), state_components.end(), invalid_component) ||
      std::any_of(field_components.begin(), field_components.end(), invalid_component))
    throw std::runtime_error("System::set_field_boundary_dependencies contains invalid entries");
  auto plan_it = p_->fields_.named_field_plans_.find(provider_slot);
  if (plan_it == p_->fields_.named_field_plans_.end())
    throw std::runtime_error("System::set_field_boundary_dependencies unknown provider slot");
  auto& plan = plan_it->second;
  plan.boundary_state_blocks = state_blocks;
  plan.boundary_state_components = state_components;
  plan.boundary_field_blocks = field_blocks;
  plan.boundary_field_keys = field_keys;
  plan.boundary_field_components = field_components;
  auto registered = p_->fields_.named_fields_.find(provider_slot);
  if (registered != p_->fields_.named_fields_.end()) {
    registered->second.plan = plan;
    registered->second.has_plan = true;
    registered->second.ell.reset();
  }
}

void System::set_field_boundary_kernel(const std::string& provider_slot,
                                       const CompiledFieldBoundaryKernel& kernel) {
  require_assembling(p_->lifecycle_, "set_field_boundary_kernel");
  kernel.validate();
  auto plan_it = p_->fields_.named_field_plans_.find(provider_slot);
  if (plan_it == p_->fields_.named_field_plans_.end())
    throw std::runtime_error("System::set_field_boundary_kernel unknown provider slot");
  auto& plan = plan_it->second;
  plan.boundary_kernel = kernel;
  plan.has_boundary_kernel = true;
  if (!kernel.observes_iteration)
    plan.boundary_context.point.iteration = 0;
  auto registered = p_->fields_.named_fields_.find(provider_slot);
  if (registered != p_->fields_.named_fields_.end()) {
    registered->second.plan = plan;
    registered->second.has_plan = true;
    registered->second.ell.reset();
  }
}

void System::set_field_logical_timepoint(const std::string& provider_slot,
                                         const FieldLogicalTimePoint& point) {
  auto plan_it = p_->fields_.named_field_plans_.find(provider_slot);
  if (plan_it == p_->fields_.named_field_plans_.end())
    throw std::runtime_error("System::set_field_logical_timepoint unknown provider slot");
  auto& plan = plan_it->second;
  plan.boundary_context.point = point;
  if (!plan.has_boundary_kernel || !plan.boundary_kernel.observes_iteration)
    plan.boundary_context.point.iteration = 0;
  auto registered = p_->fields_.named_fields_.find(provider_slot);
  if (registered != p_->fields_.named_fields_.end()) {
    registered->second.plan.boundary_context = plan.boundary_context;
    if (registered->second.ell && registered->second.plan.has_boundary_kernel) {
      auto* geometric = std::get_if<GeometricMG>(&*registered->second.ell);
      if (geometric != nullptr)
        geometric->set_boundary_context(plan.boundary_context);
    }
  }
}

void System::set_field_boundary_parameters(const std::string& provider_slot,
                                           const std::vector<double>& parameters) {
  auto plan_it = p_->fields_.named_field_plans_.find(provider_slot);
  if (plan_it == p_->fields_.named_field_plans_.end())
    throw std::runtime_error("System::set_field_boundary_parameters unknown provider slot");
  auto& plan = plan_it->second;
  if (!plan.boundary_parameters)
    plan.boundary_parameters = std::make_shared<std::vector<Real>>();
  plan.boundary_parameters->assign(parameters.begin(), parameters.end());
  plan.boundary_context.parameters = plan.boundary_parameters.get();
  plan.boundary_context.parameter_count = static_cast<int>(parameters.size());
  auto registered = p_->fields_.named_fields_.find(provider_slot);
  if (registered != p_->fields_.named_fields_.end()) {
    auto& installed = registered->second.plan;
    installed.boundary_parameters = plan.boundary_parameters;
    installed.boundary_context.parameters = plan.boundary_parameters.get();
    installed.boundary_context.parameter_count = plan.boundary_context.parameter_count;
    if (registered->second.ell && installed.has_boundary_kernel) {
      auto* geometric = std::get_if<GeometricMG>(&*registered->second.ell);
      if (geometric != nullptr)
        geometric->set_boundary_context(installed.boundary_context);
    }
  }
}

void System::set_field_newton_plan(const std::string& provider_slot, double tolerance,
                                   int max_iterations, double linear_tolerance,
                                   int linear_max_iterations, int restart, double armijo,
                                   double minimum_step) {
  require_assembling(p_->lifecycle_, "set_field_newton_plan");
  auto found = p_->fields_.named_field_plans_.find(provider_slot);
  if (found == p_->fields_.named_field_plans_.end())
    throw std::runtime_error("System::set_field_newton_plan unknown field provider slot");
  FieldNewtonOptions options{static_cast<Real>(tolerance), max_iterations,
                             static_cast<Real>(linear_tolerance), linear_max_iterations,
                             restart, static_cast<Real>(armijo),
                             static_cast<Real>(minimum_step)};
  validate_field_newton_options(options);
  found->second.has_newton = true;
  found->second.newton = options;
  auto installed = p_->fields_.named_fields_.find(provider_slot);
  if (installed != p_->fields_.named_fields_.end()) {
    installed->second.plan.has_newton = true;
    installed->second.plan.newton = options;
    installed->second.ell.reset();
  }
}

void System::set_field_nullspace(const std::string& provider_slot, bool constant_kernel,
                                 bool mean_zero_gauge) {
  require_assembling(p_->lifecycle_, "set_field_nullspace");
  if (mean_zero_gauge && !constant_kernel)
    throw std::runtime_error("System::set_field_nullspace mean-zero gauge requires constant kernel");
  auto plan_it = p_->fields_.named_field_plans_.find(provider_slot);
  if (plan_it == p_->fields_.named_field_plans_.end())
    throw std::runtime_error("System::set_field_nullspace unknown provider slot");
  auto& plan = plan_it->second;
  plan.nullspace = {};
  if (constant_kernel) {
    plan.nullspace = constant_mean_zero_nullspace(
        provider_slot + ":topology-nullspace", "derived:uniform-connected-component:0",
        p_->geom.dx() * p_->geom.dy());
    if (!mean_zero_gauge)
      plan.nullspace.gauges.clear();
  }
  auto registered = p_->fields_.named_fields_.find(provider_slot);
  if (registered != p_->fields_.named_fields_.end()) {
    registered->second.has_plan = true;
    registered->second.plan = plan;
    registered->second.ell.reset();
  }
}

namespace {
// Translates the Python disc transport mode ("none"|"staircase"|"cutcell") into a GeometryMode. EXPLICIT
// error on an unknown mode (never a silent fallback). Single source of the name table.
GeometryMode parse_geometry_mode(const std::string& mode, const char* err_context) {
  if (mode == "none")
    return GeometryMode::None;
  if (mode == "staircase")
    return GeometryMode::Staircase;
  if (mode == "cutcell")
    return GeometryMode::CutCell;
  throw std::runtime_error(std::string(err_context) + " : unknown geometry mode '" + mode +
                           "' (none|staircase|cutcell)");
}
}  // namespace

void System::set_disc_domain(double cx, double cy, double R, const std::string& mode,
                             double kappa_min, double face_open_eps, double cut_theta_min) {
  Impl* P = p_.get();
  require_assembling(P->lifecycle_, "set_disc_domain");  // frozen once pops.bind completes (ADC-592)
  // ADC-615: resolve the cut-cell thresholds (each <= 0 keeps the kEb* default). Refuse out-of-domain
  // values STRUCTURALLY -- a degenerate clamp is a structural error, never a silent fallback.
  if (kappa_min < 0.0 || face_open_eps < 0.0 || cut_theta_min < 0.0)
    throw std::runtime_error("System::set_disc_domain : kappa_min / face_open_eps / cut_theta_min "
                             ">= 0 required (0 = keep the default)");
  if (kappa_min > 1.0 || cut_theta_min > 1.0)
    throw std::runtime_error("System::set_disc_domain : kappa_min / cut_theta_min must be in (0, 1]");
  // CARTESIAN only: polar already bounds the ring by its radial walls (r_min / r_max,
  // zero radial flux) -> a Cartesian disc mask makes no sense on the (r, theta) grid.
  if (P->polar_)
    throw std::runtime_error(
        "System::set_disc_domain : polar geometry (the ring is already bounded by its radial "
        "walls r_min/r_max ; the Cartesian disc mask does not apply)");
  if (!(R > 0.0))
    throw std::runtime_error("System::set_disc_domain : radius R > 0 required");
  // Validate the mode BEFORE any mutation (an unknown mode must not leave the disc half-set).
  const GeometryMode gmode = parse_geometry_mode(mode, "System::set_disc_domain");
  // wave_speed_cache (ADC-199) is only wired on the full Cartesian advance: a disc mode
  // (staircase/cutcell) borrows advance_masked / advance_eb which ignore the cache -> explicit rejection.
  if (gmode != GeometryMode::None && P->ws_cache_block_)
    throw std::runtime_error(
        "System::set_disc_domain : mode '" + mode +
        "' incompatible with wave_speed_cache (a block enabled the HLL wave speed "
        "cache, only wired on the full Cartesian advance ; remove wave_speed_cache "
        "or use mode='none')");
  P->eb_domain_ = detail::DiscDomain{cx, cy, R};
  P->eb_set_ = true;
  // ADC-615: store the resolved thresholds (0 -> keep the kEb* default). Consumed by the EB transport
  // (assemble_rhs_eb) and the elliptic Shortley-Weller wall (cut_theta_min), single source of truth.
  if (kappa_min > 0.0)
    P->eb_thresholds_.kappa_min = static_cast<Real>(kappa_min);
  if (face_open_eps > 0.0)
    P->eb_thresholds_.face_open_eps = static_cast<Real>(face_open_eps);
  if (cut_theta_min > 0.0)
    P->eb_thresholds_.cut_theta_min = static_cast<Real>(cut_theta_min);
  // Materializes the 0/1 cell-centered mask (1 ghost, so the mask-aware transport reads the
  // i-1/i+1/j-1/j+1 neighbors up to the edge). Same layout as the blocks (ba/dm). Cell active when
  // its CENTER is inside the disc (level set < 0, SAME convention as the conducting wall).
  P->domain_mask_ = MultiFab(P->ba, P->dm, 1, 1);
  const detail::DiscDomain disc = P->eb_domain_;
  const Geometry geom = P->geom;
  for (int li = 0; li < P->domain_mask_.local_size(); ++li) {
    Array4 m = P->domain_mask_.fab(li).array();
    // box WITH ghosts: we also classify the ghosts (the mask-aware transport reads the edge neighbors).
    const Box2D g = P->domain_mask_.fab(li).grown_box();
    for_each_cell(g, [=] POPS_HD(int i, int j) {
      m(i, j, 0) = disc.cell_active(geom.x_cell(i), geom.y_cell(j)) ? Real(1) : Real(0);
    });
  }
  // TRANSPORT ROUTING (project T5-PR3). mode == "none": the mask is materialized (queryable
  // via disc_mask()) but the transport stays FULL Cartesian -> bit-identical. mode != "none": the
  // stepper routes the advance to assemble_rhs_masked (staircase) / assemble_rhs_eb (cutcell).
  P->geometry_mode_ = gmode;
}

void System::set_geometry_mode(const std::string& mode) {
  Impl* P = p_.get();
  require_assembling(P->lifecycle_, "set_geometry_mode");  // frozen once pops.bind completes (ADC-592)
  const GeometryMode gmode = parse_geometry_mode(mode, "System::set_geometry_mode");
  // An embedded-boundary mode (staircase/cutcell) only makes sense with a fixed domain: otherwise the
  // stepper would fall back on the full transport (the mask / level set does not exist), a silent
  // footgun -> we reject.
  if (gmode != GeometryMode::None && !P->eb_set_)
    throw std::runtime_error(
        "System::set_geometry_mode : embedded-boundary mode '" + mode +
        "' requested without a fixed level-set domain ; call set_disc_domain(cx, cy, R) first");
  // wave_speed_cache (ADC-199) is not carried by the disc advances -> explicit rejection (cf.
  // set_disc_domain) rather than a cache silently ignored in staircase/cutcell mode.
  if (gmode != GeometryMode::None && P->ws_cache_block_)
    throw std::runtime_error(
        "System::set_geometry_mode : mode '" + mode +
        "' incompatible with wave_speed_cache (a block enabled the HLL wave speed "
        "cache, only wired on the full Cartesian advance ; remove wave_speed_cache "
        "or use mode='none')");
  P->geometry_mode_ = gmode;
}

std::vector<double> System::disc_mask() const {
  Impl* P = p_.get();
  device_fence();
  const Box2D v = P->dom;
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  if (!P->eb_set_) {
    // CONTRACT: without a fixed domain, the transport subdomain is the whole domain -> all active.
    out.assign(static_cast<std::size_t>(v.nx()) * v.ny(), 1.0);
    return out;
  }
  const ConstArray4 m = P->domain_mask_.fab(0).const_array();
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      out.push_back(static_cast<double>(m(i, j, 0)));
  return out;
}

void System::set_epsilon_field(const std::vector<double>& eps) {
  require_assembling(p_->lifecycle_, "set_epsilon_field");  // frozen once pops.bind completes (ADC-592)
  const int n = p_->cfg.n;
  if (static_cast<int>(eps.size()) != n * n)
    throw std::runtime_error("System::set_epsilon_field : size != n*n");
  for (double e : eps)
    if (!(e > 0.0))
      throw std::runtime_error("System::set_epsilon_field : permittivity eps(x) > 0 required");
  p_->fields_.p_eps_field_ = eps;
  p_->fields_.has_eps_field_ = true;
  p_->fields_.ell_
      .reset();  // the operator will be rebuilt with the eps field at the next solve_fields
}

void System::set_epsilon_anisotropic_field(const std::vector<double>& eps_x,
                                           const std::vector<double>& eps_y) {
  require_assembling(p_->lifecycle_,
                     "set_epsilon_anisotropic_field");  // frozen once pops.bind completes (ADC-592)
  const int n = p_->cfg.n;
  if (static_cast<int>(eps_x.size()) != n * n || static_cast<int>(eps_y.size()) != n * n)
    throw std::runtime_error(
        "System::set_epsilon_anisotropic_field : size != n*n (eps_x and eps_y)");
  for (double e : eps_x)
    if (!(e > 0.0))
      throw std::runtime_error(
          "System::set_epsilon_anisotropic_field : permittivity eps_x(x) > 0 required");
  for (double e : eps_y)
    if (!(e > 0.0))
      throw std::runtime_error(
          "System::set_epsilon_anisotropic_field : permittivity eps_y(x) > 0 required");
  p_->fields_.p_eps_x_field_ = eps_x;
  p_->fields_.p_eps_y_field_ = eps_y;
  p_->fields_.has_eps_xy_field_ = true;
  p_->fields_.ell_
      .reset();  // operator rebuilt as div(diag(eps_x, eps_y) grad phi) at the next solve_fields
}

void System::set_reaction_field(const std::vector<double>& kappa) {
  require_assembling(p_->lifecycle_, "set_reaction_field");  // frozen once pops.bind completes (ADC-592)
  const int n = p_->cfg.n;
  if (static_cast<int>(kappa.size()) != n * n)
    throw std::runtime_error("System::set_reaction_field : size != n*n");
  for (double k : kappa)
    if (!(k >= 0.0))
      throw std::runtime_error(
          "System::set_reaction_field : reaction term kappa(x) >= 0 required "
          "(well-posed elliptic operator and convergent multigrid)");
  p_->fields_.p_kappa_field_ = kappa;
  p_->fields_.has_kappa_field_ = true;
  p_->fields_.ell_.reset();  // operator rebuilt with - kappa phi at the next solve_fields
}

POPS_EXPORT void System::ensure_aux_width(int ncomp) {
  p_->ensure_aux_width(ncomp);
}

void System::set_magnetic_field(const std::vector<double>& bz) {
  // Expected size of the B_z(x) field row-major (slow axis = 2nd box index, fast axis = 1st):
  //   Cartesian = n * n (square, BIT-IDENTICAL); POLAR = nr * ntheta (ring, i = r fast, cf.
  //   apply_bz / polar set_density). The layout is the SAME as set_density (flat[j * nr + i]).
  if (p_->polar_) {
    const int nr = Impl::polar_nr(p_->cfg), nth = Impl::polar_ntheta(p_->cfg);
    if (static_cast<int>(bz.size()) != nr * nth)
      throw std::runtime_error("System::set_magnetic_field : size != nr*ntheta (polar)");
  } else {
    const int n = p_->cfg.n;
    if (static_cast<int>(bz.size()) != n * n)
      throw std::runtime_error("System::set_magnetic_field : size != n*n");
  }
  p_->fields_.bz_field_.assign(bz.begin(), bz.end());
  p_->fields_
      .apply_bz();  // apply right away if a block already reads B_z; otherwise keep for ensure_aux_width
}

void System::set_electron_temperature_from(const std::string& name) {
  require_assembling(p_->lifecycle_,
                     "set_electron_temperature_from");  // frozen once pops.bind completes (ADC-592)
  const int idx = p_->index(name);  // raises if unknown block
  if (p_->sp[static_cast<std::size_t>(idx)].ncomp != 4)
    throw std::runtime_error(
        "System::set_electron_temperature_from : block '" + name +
        "' must be compressible (4 vars : rho, rho u, rho v, E) for T = p/rho");
  p_->fields_.te_src_ = idx;
  // T_e (canonical comp 4) DERIVED: recomputed at each solve_fields. Inert as long as no block
  // reads T_e (n_aux=5 -> ensure_aux_width(5)), like set_magnetic_field for B_z.
  p_->fields_.apply_te();
}

// Expected size of a cell-defined field (Cartesian n*n / polar nr*ntheta). Member of Impl:
// a free caller could not name the private type System::Impl.
std::size_t System::Impl::aux_field_cell_count() const {
  if (polar_) {
    const int nr = polar_nr(cfg), nth = polar_ntheta(cfg);
    return static_cast<std::size_t>(nr) * nth;
  }
  return static_cast<std::size_t>(cfg.n) * cfg.n;
}

void System::set_aux_field_component(int comp, const std::vector<double>& field) {
  Impl* P = p_.get();
  // RESERVED components (phi/grad/B_z/T_e): a named aux field starts at kAuxNamedBase (= 5).
  // B_z and T_e keep their dedicated paths -> redirecting message (the Python facade already intercepts
  // the canonical names, this guard covers a direct C++ call).
  if (comp < kAuxNamedBase)
    throw std::runtime_error(
        "System::set_aux_field : component " + std::to_string(comp) +
        " reserved (phi/grad_x/grad_y/B_z/T_e) ; a named aux field starts at index " +
        std::to_string(kAuxNamedBase) +
        " (B_z -> set_magnetic_field, T_e -> "
        "set_electron_temperature_from)");
  const std::size_t expect = P->aux_field_cell_count();
  if (field.size() != expect)
    throw std::runtime_error("System::set_aux_field : size " + std::to_string(field.size()) +
                             " != " + std::to_string(expect) + " (grid cells)");
  // The aux channel must be wide enough: a block declaring this field (n_aux = kAuxNamedBase + k + 1) has
  // already called ensure_aux_width at its add time. Otherwise the field would be read by no model -> error.
  if (comp >= P->aux_ncomp_)
    throw std::runtime_error(
        "System::set_aux_field : the aux channel has only " + std::to_string(P->aux_ncomp_) +
        " components ; no block declares an aux field at index " + std::to_string(comp) +
        " (add the block that reads it before set_aux_field)");
  std::vector<Real> f(field.begin(), field.end());
  p_->fields_.apply_named_aux_one(comp, f);     // populate right away (channel wide enough)
  p_->fields_.named_aux_[comp] = std::move(f);  // keep for a later reallocation of the channel
}

void System::set_aux_field_halo_component(int comp, int bc_type, double value) {
  Impl* P = p_.get();
  if (comp < kAuxNamedBase)
    throw std::runtime_error(
        "System::set_aux_field (halo) : component " + std::to_string(comp) +
        " reserved (phi/grad_x/grad_y/B_z/T_e) ; a named aux field starts at index " +
        std::to_string(kAuxNamedBase));
  if (comp >= P->aux_ncomp_)
    throw std::runtime_error(
        "System::set_aux_field (halo) : the aux channel has only " + std::to_string(P->aux_ncomp_) +
        " components ; no block declares an aux field at index " + std::to_string(comp));
  // Only the PHYSICAL-face policies are meaningful per field (Foextrap / Dirichlet). A periodic face is
  // a domain property kept by aux_halo_override, so a per-field 'periodic' is not offered.
  if (bc_type != static_cast<int>(BCType::Foextrap) &&
      bc_type != static_cast<int>(BCType::Dirichlet))
    throw std::runtime_error("System::set_aux_field (halo) : unsupported halo type " +
                             std::to_string(bc_type) + " ; use foextrap or dirichlet");
  P->fields_.named_aux_bc_[comp] =
      AuxHaloPolicy{static_cast<BCType>(bc_type), static_cast<Real>(value)};
}

std::vector<double> System::aux_field_component(int comp) const {
  Impl* P = p_.get();
  if (comp < kAuxNamedBase)
    throw std::runtime_error("System::aux_field : component " + std::to_string(comp) +
                             " reserved (phi/grad_x/grad_y/B_z/T_e) ; read phi via potential(), a "
                             "named aux field starts "
                             "at index " +
                             std::to_string(kAuxNamedBase));
  if (comp >= P->aux_ncomp_)
    throw std::runtime_error(
        "System::aux_field : the aux channel has only " + std::to_string(P->aux_ncomp_) +
        " components ; no block declares an aux field at index " + std::to_string(comp));
  device_fence();
  // Rank without a box (MPI mono-box): EMPTY return (cf. potential / copy_comp0). The Python facade is
  // mono-rank; the multi-rank global field would be a dedicated collective accessor (follow-up).
  if (P->aux.local_size() == 0)
    return {};
  const ConstArray4 a = P->aux.fab(0).const_array();
  const Box2D v = P->aux.box(0);
  std::vector<double> out;
  out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
  for (int j = v.lo[1]; j <= v.hi[1]; ++j)
    for (int i = v.lo[0]; i <= v.hi[0]; ++i)
      out.push_back(static_cast<double>(a(i, j, comp)));
  return out;
}

// The named inter-species couplings (System::add_ionization / add_collision / add_thermal_exchange)
// are removed (ADC-595): they are Python presets (python/pops/physics/coupling_presets.py) that emit the
// same formulas as a generic CoupledSource and register through add_coupling_operator with a declared
// conservation contract. Impl::couplings / coupled_freqs_ / coupled_freq_exprs_ STORAGE stays untouched
// (SystemStepper::apply_couplings / step_cfl read them); only the entry methods go.

void System::add_coupled_source(const CoupledSourceProgram& prog_desc, double frequency,
                                const std::string& label) {
  require_assembling(p_->lifecycle_, "add_coupled_source");  // frozen once pops.bind completes (ADC-592)
  // Bytecode description grouped into a POD (ADC-214): local aliases to keep the body readable (the
  // names and the semantics are strictly those of the old flat parameters).
  const std::vector<std::string>& in_blocks = prog_desc.in_blocks;
  const std::vector<std::string>& in_roles = prog_desc.in_roles;
  const std::vector<double>& consts = prog_desc.consts;
  const std::vector<std::string>& out_blocks = prog_desc.out_blocks;
  const std::vector<std::string>& out_roles = prog_desc.out_roles;
  const std::vector<int>& prog_ops = prog_desc.prog_ops;
  const std::vector<int>& prog_args = prog_desc.prog_args;
  const std::vector<int>& prog_lens = prog_desc.prog_lens;
  const std::vector<int>& freq_prog_ops = prog_desc.freq_prog_ops;
  const std::vector<int>& freq_prog_args = prog_desc.freq_prog_args;
  Impl* P = p_.get();
  const int n_in = static_cast<int>(in_blocks.size());
  const int n_const = static_cast<int>(consts.size());
  const int n_terms = static_cast<int>(out_blocks.size());
  // --- shape validation (before any step, EXPLICIT errors) ------------------------------------
  if (n_terms == 0)
    throw std::runtime_error("System::add_coupled_source : no source term (out_blocks empty)");
  if (static_cast<int>(in_roles.size()) != n_in)
    throw std::runtime_error(
        "System::add_coupled_source : in_blocks / in_roles of different sizes");
  if (static_cast<int>(out_roles.size()) != n_terms ||
      static_cast<int>(prog_lens.size()) != n_terms)
    throw std::runtime_error(
        "System::add_coupled_source : out_blocks / out_roles / prog_lens of different "
        "sizes");
  if (prog_ops.size() != prog_args.size())
    throw std::runtime_error(
        "System::add_coupled_source : prog_ops / prog_args of different sizes");
  if (n_in + n_const > kCsMaxReg)
    throw std::runtime_error(
        "System::add_coupled_source : too many registers (inputs + constants > " +
        std::to_string(kCsMaxReg) + ")");
  if (n_terms > kCsMaxTerms)
    throw std::runtime_error("System::add_coupled_source : too many source terms (> " +
                             std::to_string(kCsMaxTerms) + ")");
  // Resolves role -> component via the CONSERVATIVE descriptor of the block. The role is addressed BY
  // NAME: a canonical role name OR a user-defined role label (index_of(string), ADC-292). An unknown
  // block raises via P->index().
  auto resolve = [&](const std::string& block, const std::string& role) -> std::pair<int, int> {
    const int sidx = P->index(block);  // raises if unknown block
    const VariableSet& vs = P->sp[static_cast<std::size_t>(sidx)].cons_vars;
    // STRICT (no silent fallback): a DSL coupled source targets a (block, role) EXPLICITLY requested
    // by the user. If the block does NOT expose this role (neither a canonical role nor a declared
    // user-role label), it is an error: a fallback on component 0 would apply the source to the wrong
    // field SILENTLY. We raise, listing what the block actually exposes.
    const int comp = vs.index_of(role);
    if (comp < 0)
      throw std::runtime_error(
          "System::add_coupled_source : block '" + block + "' does not expose role '" + role +
          "' (roles: " + (vs.roles.empty() ? std::string("<none>") : roles_csv(vs)) +
          ", no silent fallback on component 0)");
    return {sidx, comp};
  };
  // Inputs: (species, component) read per cell. Captured by INDEX (the fabs may be
  // reallocated between registration and application: we rebuild the Array4 at EACH step).
  struct InRef {
    int sidx, comp;
  };
  std::vector<InRef> ins(static_cast<std::size_t>(n_in));
  for (int c = 0; c < n_in; ++c) {
    auto [s, comp] =
        resolve(in_blocks[static_cast<std::size_t>(c)], in_roles[static_cast<std::size_t>(c)]);
    ins[static_cast<std::size_t>(c)] = {s, comp};
  }
  struct OutRef {
    int sidx, comp;
    CsProgram prog;
  };
  std::vector<OutRef> outs(static_cast<std::size_t>(n_terms));
  int off = 0;
  for (int t = 0; t < n_terms; ++t) {
    auto [s, comp] =
        resolve(out_blocks[static_cast<std::size_t>(t)], out_roles[static_cast<std::size_t>(t)]);
    const int len = prog_lens[static_cast<std::size_t>(t)];
    if (len < 0 || len > kCsMaxProg)
      throw std::runtime_error("System::add_coupled_source : program of term " + std::to_string(t) +
                               " too long (> " + std::to_string(kCsMaxProg) + ")");
    if (off + len > static_cast<int>(prog_ops.size()))
      throw std::runtime_error("System::add_coupled_source : prog_lens inconsistent with prog_ops");
    CsProgram pg;
    pg.len = len;
    for (int k = 0; k < len; ++k) {
      const int opc = prog_ops[static_cast<std::size_t>(off + k)];
      const int a = prog_args[static_cast<std::size_t>(off + k)];
      if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
        throw std::runtime_error("System::add_coupled_source : invalid opcode");
      if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
        throw std::runtime_error(
            "System::add_coupled_source : register out of bounds in the program");
      pg.op[k] = opc;
      pg.arg[k] = a;
    }
    validate_cs_program_stack(pg, "System::add_coupled_source term " + std::to_string(t));
    outs[static_cast<std::size_t>(t)] = {s, comp, pg};
    off += len;
  }
  // All touched species (inputs + outputs) share the System DistributionMapping (one box
  // round-robin distributed), so same local_size() and same local indexing -> we would iterate in parallel
  // over the local fabs. Conversion to CAPTURED values (no reference to the C++ lambda's 'this').
  std::vector<Real> kconsts(consts.begin(), consts.end());
  // Optional PER-CELL frequency (CoupledSource.frequency with an Expr, refinement of the
  // CONSTANT frequency): a bytecode program mu(U) on the SAME register table as the terms
  // (inputs then constants). Validates HERE its SHAPE (opcodes / bounded registers) BEFORE any push -- the
  // bound must be registered only after a complete validation (anti-phantom-bound rule). Empty
  // (default) -> no per-cell frequency (historical path).
  const bool has_freq_expr = !freq_prog_ops.empty() || !freq_prog_args.empty();
  CsProgram freq_pg;
  if (has_freq_expr) {
    if (freq_prog_ops.size() != freq_prog_args.size())
      throw std::runtime_error(
          "System::add_coupled_source : freq_prog_ops / freq_prog_args of different "
          "sizes");
    if (static_cast<int>(freq_prog_ops.size()) > kCsMaxProg)
      throw std::runtime_error("System::add_coupled_source : frequency program too long (> " +
                               std::to_string(kCsMaxProg) + ")");
    freq_pg.len = static_cast<int>(freq_prog_ops.size());
    for (int k = 0; k < freq_pg.len; ++k) {
      const int opc = freq_prog_ops[static_cast<std::size_t>(k)];
      const int a = freq_prog_args[static_cast<std::size_t>(k)];
      if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
        throw std::runtime_error("System::add_coupled_source : invalid opcode in the frequency");
      if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
        throw std::runtime_error(
            "System::add_coupled_source : register out of bounds in the frequency");
      freq_pg.op[k] = opc;
      freq_pg.arg[k] = a;
    }
    validate_cs_program_stack(freq_pg, "System::add_coupled_source frequency");
  }
  // CONSTANT declared frequency of the coupling (audit wave 3): registered for the step bound of
  // step_cfl / step_adaptive (dt <= cfl/mu on the MACRO-step). <= 0 = no bound (historical). Pushed
  // AFTER all the validation (source AND frequency have raised if invalid): a rejected coupling must
  // leave NO phantom bound -- otherwise a script that try/excepts the failure would keep a throttled step without
  // matching physics.
  if (frequency > 0.0)
    P->coupled_freqs_.push_back(Impl::CoupledFreq{label, frequency});
  // PER-CELL frequency: same rule (push after complete validation). The inputs REUSE the
  // resolve() resolution (ins); the constants are the same as the source (kconsts). The program
  // mu(U) is reduced (MAX) at each step in step_cfl / step_adaptive.
  if (has_freq_expr) {
    Impl::CoupledFreqExpr ce;
    ce.label = label;
    ce.prog = freq_pg;
    ce.n_in = n_in;
    ce.ins.resize(static_cast<std::size_t>(n_in));
    for (int c = 0; c < n_in; ++c)
      ce.ins[static_cast<std::size_t>(c)] = {ins[static_cast<std::size_t>(c)].sidx,
                                             ins[static_cast<std::size_t>(c)].comp};
    ce.kconsts = kconsts;
    P->coupled_freq_exprs_.push_back(std::move(ce));
  }
  P->couplings.push_back([P, ins, outs, kconsts, n_in, n_const, n_terms](Real dt) {
    // MPI-safe: iteration over the LOCAL fabs of the first input block (or output if no
    // input). local_size()==0 on a rank without a box -> empty loop, no-op (no hard-coded fab(0)).
    const int sref = n_in > 0 ? ins[0].sidx : outs[0].sidx;
    MultiFab& Uref = P->sp[static_cast<std::size_t>(sref)].U;
    for (int li = 0; li < Uref.local_size(); ++li) {
      CoupledSourceKernel kern;
      kern.dt = dt;
      kern.n_in = n_in;
      kern.n_const = n_const;
      kern.n_terms = n_terms;
      for (int c = 0; c < n_in; ++c) {
        kern.in[c] = P->sp[static_cast<std::size_t>(ins[static_cast<std::size_t>(c)].sidx)]
                         .U.fab(li)
                         .array();
        kern.in_comp[c] = ins[static_cast<std::size_t>(c)].comp;
      }
      for (int c = 0; c < n_const; ++c)
        kern.consts[c] = kconsts[static_cast<std::size_t>(c)];
      for (int t = 0; t < n_terms; ++t) {
        kern.out[t] = P->sp[static_cast<std::size_t>(outs[static_cast<std::size_t>(t)].sidx)]
                          .U.fab(li)
                          .array();
        kern.out_comp[t] = outs[static_cast<std::size_t>(t)].comp;
        kern.prog[t] = outs[static_cast<std::size_t>(t)].prog;
      }
      for_each_cell(Uref.box(li), kern);  // NAMED functor (device-clean), additive forward-Euler
    }
  });
  // Inspect metadata (ADC-595): a raw add_coupled_source declares NO conservation contract, so it
  // registers an "unchecked" view (empty ConservationContract) carrying the label and the frequency
  // bound. add_coupling_operator overwrites this behavior by pushing the DECLARED contract instead.
  CouplingOperatorView view;
  view.label = label;
  view.frequency.constant_mu = frequency;
  view.frequency.per_cell = has_freq_expr;
  P->coupling_.coupled_operators.push_back(std::move(view));
}

void System::add_coupling_operator(const CouplingOperator& op) {
  // Validate the DECLARED conservation contract against the actual output terms BEFORE anything is
  // stored (host, fail-loud): a coupling that declares a role conserved whose terms do not cancel
  // raises here and leaves no partial state (anti-phantom-registration, like add_coupled_source's
  // frequency-bound rule). An unchecked (empty) contract is a no-op check.
  validate_coupling_contract(op, "System::add_coupling_operator");
  // Lower through the SAME flat path (bit-identical numerics); it pushes an "unchecked" inspect view
  // at its tail. We then replace that view's contract with the DECLARED one so coupled_operators()
  // reports the typed contract rather than "unchecked".
  add_coupled_source(op.program, op.frequency.constant_mu, op.label);
  p_->coupling_.coupled_operators.back().conservation = op.conservation;
}

const std::vector<CouplingOperatorView>& System::coupled_operators() const {
  return p_->coupling_.coupled_operators;
}

void System::set_source_stage(const std::string& name, const std::string& kind, double theta,
                              double alpha, const SourceStageOptions& opts) {
  require_assembling(p_->lifecycle_, "set_source_stage");  // frozen once pops.bind completes (ADC-592)
  // Settings grouped into a POD (ADC-214): local aliases to keep the body readable (the names and the
  // semantics are strictly those of the old flat parameters).
  const double krylov_tol = opts.krylov_tol;
  const int krylov_max_iters = opts.krylov_max_iters;
  const std::string& density = opts.density;
  const std::string& momentum_x = opts.momentum_x;
  const std::string& momentum_y = opts.momentum_y;
  const std::string& energy = opts.energy;
  const int bz_aux_component = opts.bz_aux_component;
  // ADC-645: Krylov-preconditioner knobs of the stage. 0/"" = the historical stepper defaults
  // (ONE V-cycle; RadialLine on polar), bit-identical; out-of-domain values refuse loud here.
  const int n_precond_vcycles = opts.n_precond_vcycles;
  const std::string& polar_precond = opts.polar_precond;
  if (n_precond_vcycles < 0 || n_precond_vcycles > 2)
    throw std::runtime_error(
        "System::set_source_stage : n_precond_vcycles must be 1 or 2 (0 = default)");
  if (!polar_precond.empty() && polar_precond != "radial_line" && polar_precond != "jacobi")
    throw std::runtime_error(
        "System::set_source_stage : polar_precond must be 'radial_line' or 'jacobi' ('' = "
        "default RadialLine)");
  Impl* P = p_.get();
  Impl::Species& s = P->find(name);  // raises if unknown block
  // ONLY kind wired for now: ElectrostaticLorentzCondensation (cf. CondensedSchurSourceStepper).
  // Other kinds may be added without touching the facade (explicit rejection, no silent ignore).
  if (kind != "electrostatic_lorentz")
    throw std::runtime_error("System::set_source_stage : kind '" + kind +
                             "' unknown (only 'electrostatic_lorentz' is supported)");
  if (!(theta > 0.0 && theta <= 1.0))
    throw std::runtime_error("System::set_source_stage : theta must be in (0, 1] (received " +
                             std::to_string(theta) + ")");
  // Tolerance / budget of the stage Krylov solve (audit 2026-06: the constants 1e-10 / 400 (cart)
  // / 600 (polar) are no longer frozen). krylov_tol <= 0 / krylov_max_iters <= 0 = "historical
  // stepper default" (we do not touch the setting of the constructed stepper).
  if (krylov_tol > 0.0 && !(krylov_tol < 1.0))
    throw std::runtime_error("System::set_source_stage : krylov_tol must be in (0, 1)");
  // GEOMETRY: the condensed source stage is wired in CARTESIAN (CondensedSchurSourceStepper, #126) AND in
  // POLAR (PolarCondensedSchurSourceStepper, #212, Path A step 2c). The dispatch below builds the
  // stepper adapted to the System geometry. Any other geometry is REJECTED explicitly (no
  // silent ignore).
  const bool polar = (P->cfg.geometry == "polar");
  if (P->cfg.geometry != "cartesian" && !polar)
    throw std::runtime_error(
        "System::set_source_stage : condensed source stage supports the "
        "cartesian and polar geometries (received '" +
        P->cfg.geometry + "')");
  // The POLAR condensed source stage is now MULTI-RANK MPI (PolarTensorKrylovSolver / polar
  // Schur distributed by AZIMUTHAL split; check_radial_columns layout guard in the
  // solver). On the FACADE side, the System builds for now ONE box covering the ring (P->ba mono-box),
  // so under MPI the box lives on rank 0 and the other ranks have local_size()==0: the solve stays CORRECT
  // (collective dot/project_mean called on all ranks, zero contributions from the empty ranks) and
  // BIT-IDENTICAL to the mono-rank, but without real parallelism at this level. The effective theta split
  // (true multi-rank scaling) takes place at the C++ API level (PolarCondensedSchurSourceStepper with a
  // BoxArray split in theta); the facade-side theta distribution is deferred (Extend). No mono-rank
  // guard here: the PolarTensorKrylovSolver raises a clear error if the layout ever cuts r.
  // ROLE CONTRACT: the block must expose Density / MomentumX / MomentumY (Energy optional). We read the
  // CONSERVATIVE descriptor of the block (populated by add_block / the .so with roles, including the compiled DSL which
  // declares the electrons with roles). A required role absent raises an EXPLICIT error HERE (before the step)
  // -- the stepper constructor would raise it too, but we diagnose on the named-block side.
  const VariableSet& vs = s.cons_vars;
  // DESCRIPTOR RESOLUTION (audit wave 2: roles/transported fields in the ABI). An
  // EMPTY descriptor = canonical role (historical, bit-identical). Otherwise: stable ROLE name
  // first (role_from_name), then block VARIABLE name. Failure = explicit error with remedy.
  auto resolve_field = [&](const std::string& spec, VariableRole canonical,
                           const char* label) -> int {
    if (spec.empty()) {
      const int idx = vs.index_of(canonical);
      if (idx < 0)
        throw std::runtime_error(
            "System::set_source_stage : block '" + name + "' does not expose the role " + label +
            " required by pops.CondensedSchur (the model must declare Density / MomentumX / "
            "MomentumY ; Energy optional), and no explicit descriptor is provided (pass "
            "density=/momentum=... with a role name or a block variable name).");
      return idx;
    }
    const VariableRole r = role_from_name(spec);
    if (r != VariableRole::Custom) {
      const int idx = vs.index_of(r);
      if (idx < 0)
        throw std::runtime_error("System::set_source_stage : block '" + name +
                                 "' does not expose role '" + spec + "' (" + label + ")");
      return idx;
    }
    for (std::size_t i = 0; i < vs.names.size(); ++i)
      if (vs.names[i] == spec)
        return static_cast<int>(i);
    throw std::runtime_error("System::set_source_stage : '" + spec +
                             "' is neither a stable role nor a variable of block '" + name + "' (" +
                             label + ")");
  };
  const int c_rho = resolve_field(density, VariableRole::Density, "Density");
  const int c_mx = resolve_field(momentum_x, VariableRole::MomentumX, "MomentumX");
  const int c_my = resolve_field(momentum_y, VariableRole::MomentumY, "MomentumY");
  const int c_E = (energy == "none")
                      ? -1
                      : (energy.empty() ? vs.index_of(VariableRole::Energy)
                                        : resolve_field(energy, VariableRole::Energy, "Energy"));
  // B_z MANDATORY: the Lorentz stage reads Omega = B_z. We require set_magnetic_field called
  // (bz_field_ provided) and we widen the aux channel to the B_z channel (kAuxBaseComps) so that apply_bz
  // populates it and solve_fields fills its ghosts. An absent B_z raises an EXPLICIT error.
  if (P->fields_.bz_field_.empty())
    throw std::runtime_error("System::set_source_stage : block '" + name +
                             "' has no B_z field (aux Omega) ; "
                             "pops.CondensedSchur requires set_magnetic_field(B_z) (the Lorentz "
                             "term reads Omega = B_z).");
  // Aux channel of the magnetic field: canonical (kAuxBaseComps) by default, redirectable by
  // bz_aux_component (transported descriptor). NOTE: apply_bz populates the CANONICAL channel; a
  // different component assumes the caller populates it itself (derived/custom aux field).
  const int c_bz = bz_aux_component >= 0 ? bz_aux_component : kAuxBaseComps;
  P->ensure_aux_width(c_bz + 1);  // guarantees the channel in the shared aux + re-applies B_z
  const double effective_krylov_tol =
      krylov_tol > 0.0 ? krylov_tol : static_cast<double>(kKrylovDefaultRelTol);
  const int effective_krylov_max_iters =
      krylov_max_iters > 0
          ? krylov_max_iters
          : (polar ? kSchurKrylovPolarMaxIters : kSchurKrylovCartesianMaxIters);
  // Builds the condensed source stage on the REAL System layout (ba/dm/geom) with the Poisson BC.
  // The stepper allocates its buffers ONCE; step() reuses them (cf. its lifecycle). alpha =
  // electrostatic coupling constant of the source subsystem.
  if (polar) {
    // POLAR (Path A step 2c): PolarCondensedSchurSourceStepper on the ring pgeom_, SAME Poisson BC
    // (radial Dirichlet/Neumann, theta always periodic on the solver side). RadialLine preconditioner
    // (default). run_source_stage invokes it exactly like the Cartesian (identical step() signature).
    // schur stays nullptr (Cartesian path untouched). EXPLICIT components resolved above
    // (empty descriptors -> canonical roles -> bit-identical): the POLAR stepper accepts the
    // overrides since wave 3 (ctor with explicit components, Cartesian parity).
    // ADC-645: the polar stage's preconditioner is the PolarPrecond enum, not MG V-cycles; a
    // cartesian-only n_precond_vcycles here would be silently ignored -> refuse loud instead.
    if (n_precond_vcycles > 0)
      throw std::runtime_error(
          "System::set_source_stage : n_precond_vcycles applies to the CARTESIAN stage only (the "
          "polar preconditioner is selected by polar_precond='radial_line'|'jacobi')");
    const PolarPrecond precond =
        polar_precond == "jacobi" ? PolarPrecond::Jacobi : PolarPrecond::RadialLine;
    s.schur_polar = std::make_shared<PolarCondensedSchurSourceStepper>(
        vs, c_rho, c_mx, c_my, c_E, P->pgeom_, P->ba, P->fields_.poisson_bc(),
        static_cast<Real>(alpha), precond);
    if (krylov_tol > 0.0 || krylov_max_iters > 0)
      s.schur_polar->set_krylov(static_cast<Real>(effective_krylov_tol),
                                effective_krylov_max_iters);
  } else {
    // CARTESIAN (#126): EXPLICIT components resolved above (empty descriptors -> canonical
    // roles -> same indices as the historical, bit-identical).
    // ADC-645: polar_precond names the POLAR preconditioner; on cartesian it would be silently
    // ignored -> refuse loud. n_precond_vcycles 0 = the historical ctor default (ONE V-cycle).
    if (!polar_precond.empty())
      throw std::runtime_error(
          "System::set_source_stage : polar_precond applies to the POLAR stage only (the "
          "cartesian preconditioner is the MG V-cycle; tune it with n_precond_vcycles)");
    s.schur = std::make_shared<CondensedSchurSourceStepper>(
        vs, c_rho, c_mx, c_my, c_E, P->geom, P->ba, P->fields_.poisson_bc(),
        static_cast<Real>(alpha), n_precond_vcycles > 0 ? n_precond_vcycles : 1);
    if (krylov_tol > 0.0 || krylov_max_iters > 0)
      s.schur->set_krylov(static_cast<Real>(effective_krylov_tol), effective_krylov_max_iters);
  }
  s.schur_bz_comp = c_bz;
  s.schur_theta = theta;
  EffectiveSourceStageOptions stage;
  stage.block = name;
  stage.kind = kind;
  stage.geometry = polar ? "polar" : "cartesian";
  stage.theta = theta;
  stage.alpha = alpha;
  stage.requested_krylov_tol = krylov_tol;
  stage.requested_krylov_max_iters = krylov_max_iters;
  stage.effective_krylov_tol = effective_krylov_tol;
  stage.effective_krylov_max_iters = effective_krylov_max_iters;
  stage.density = density;
  stage.momentum_x = momentum_x;
  stage.momentum_y = momentum_y;
  stage.energy = energy;
  stage.bz_aux_component = c_bz;
  // ADC-645: the stage's Krylov-preconditioner knobs (requested vs effective).
  stage.requested_n_precond_vcycles = n_precond_vcycles;
  stage.effective_n_precond_vcycles = polar ? 1 : (n_precond_vcycles > 0 ? n_precond_vcycles : 1);
  stage.polar_precond =
      polar ? (polar_precond.empty() ? std::string("radial_line") : polar_precond) : "";
  P->diagnostics_.source_stage_options[name] = std::move(stage);
}

void System::set_time_scheme(const std::string& scheme) {
  require_assembling(p_->lifecycle_, "set_time_scheme");  // frozen once pops.bind completes (ADC-592)
  // Routes the splitting policy of the system stepper (default Lie = bit-identical). The Strang
  // scheme reuses the SAME bricks (s.advance for the transport half-advances, run_source_stage
  // for the full source stage); it RE-SOLVES solve_fields between the stages (cf. SystemStepper::step_strang
  // and docs/HOFFART_STEP_SEQUENCE.md). An unknown scheme raises an EXPLICIT error (no silent ignore).
  if (scheme == "lie") {
    p_->stepper_.set_scheme(stepper::SplitScheme::Lie);
    p_->time_scheme_ = "lie";
  } else if (scheme == "strang") {
    p_->stepper_.set_scheme(stepper::SplitScheme::Strang);
    p_->time_scheme_ = "strang";
  } else {
    throw std::runtime_error("System::set_time_scheme : scheme '" + scheme +
                             "' unknown (expected 'lie' or 'strang')");
  }
}

void System::set_gauss_policy(const std::string& policy) {
  require_assembling(p_->lifecycle_, "set_gauss_policy");  // frozen once pops.bind completes (ADC-592)
  // Gauss's law policy (project R0, Hoffart reproduction). "restart" (default): solve_fields
  // re-solves -Delta phi = f at each step (bit-identical to the historical). "evolve": after the first
  // solve (phi^0), solve_fields NO LONGER re-solves the Poisson; it derives the aux from the CURRENT phi that
  // the Schur source stage evolves in-place in ell_phi() -> -Delta phi evolution without restart of the
  // paper (the Gauss constraint is imposed only at t=0). Has effect ONLY with a condensed source stage
  // (without it phi would stay frozen after t=0). The gauss_solved_once_ lock is reset to zero here so
  // that a policy change BEFORE the first solve stays consistent (the 1st solve always solves).
  if (policy == "restart") {
    p_->fields_.gauss_evolve_ = false;
    p_->gauss_policy_ = "restart";
  } else if (policy == "evolve") {
    p_->fields_.gauss_evolve_ = true;
    p_->gauss_policy_ = "evolve";
  } else {
    throw std::runtime_error("System::set_gauss_policy : policy '" + policy +
                             "' unknown (expected 'restart' or 'evolve')");
  }
  p_->fields_.gauss_solved_once_ = false;
}


}  // namespace pops
