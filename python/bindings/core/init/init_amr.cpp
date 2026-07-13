#include "../bindings_detail.hpp"

// ADC-365: the AMR (AmrSystemConfig + AmrSystem) bindings.
//
// ADC-593: like init_system, the AmrSystem .def registrations are INTERNAL seams of the bind flow (the
// AMR target reaches them through the typed layout, not as public vocabulary). The AmrSystem chain is
// split into concern-grouped static helpers (assembly / physics / stepping / program / data), each taking
// the class handle and adding its slice. PURE reorganization: same .def names, docstrings, args, and
// RELATIVE order (no overload set is reordered -- every AmrSystem method name here is unique). The class
// name and the .def names are unchanged, so the legacy-name architecture gate still finds them here.
namespace {

// Assembly seams: per-block composition, native block, and refinement tagging.
void bind_amr_assembly(py::class_<AmrSystem>& cls) {
  cls.def(py::init<const AmrSystemConfig&>())
      // ADC-214: Python surface UNCHANGED (same flat newton_* kwargs, same defaults). The lambda
      // assembles the NewtonOptions POD before the C++ call (parity with System.add_block).
      .def(
          "add_block",
          [](AmrSystem& s, const std::string& name, const ModelSpec& model,
             const std::string& limiter, const std::string& riemann, const std::string& recon,
             const std::string& time, int substeps, int stride,
             const std::vector<std::string>& implicit_vars,
             const std::vector<std::string>& implicit_roles, int newton_max_iters,
             double newton_rel_tol, double newton_abs_tol, double newton_fd_eps,
             double newton_damping, const std::string& newton_fail_policy, bool newton_diagnostics,
             double positivity_floor) {
            NewtonOptions newton;
            newton.max_iters = newton_max_iters;
            newton.rel_tol = static_cast<Real>(newton_rel_tol);
            newton.abs_tol = static_cast<Real>(newton_abs_tol);
            newton.fd_eps = static_cast<Real>(newton_fd_eps);
            newton.damping = static_cast<Real>(newton_damping);
            newton.fail_policy =
                newton_fail_policy_from_string(newton_fail_policy, "AmrSystem::add_block");
            s.add_block(name, model, limiter, riemann, recon, time, substeps, stride, implicit_vars,
                        implicit_roles, newton, newton_diagnostics, positivity_floor);
          },
          py::arg("name"), py::arg("model"), py::arg("limiter") = "minmod",
          py::arg("riemann") = "rusanov", py::arg("recon") = "conservative",
          py::arg("time") = "explicit", py::arg("substeps") = 1, py::arg("stride") = 1,
          // Partial IMEX mask CARRIED BY THE BLOCK (capstone vii): conserved variables treated
          // implicitly by NAME (implicit_vars) or by physical ROLE (implicit_roles). Empty (default)
          // -> full backward-Euler. Only meaningful with time="imex" and MULTI-BLOCK (cf. add_block).
          py::arg("implicit_vars") = std::vector<std::string>{},
          py::arg("implicit_roles") = std::vector<std::string>{},
          // IMEX Newton options (wave 3, System parity): OPTIONS wired in MONO-BLOCK (coupler)
          // AND MULTI-BLOCK (engine). newton_diagnostics (newton_report report): native MULTI-BLOCK
          // only (mono-block rejected at build; .so loaders rejected at the Python facade).
          py::arg("newton_max_iters") = kNewtonDefaultMaxIters,
          py::arg("newton_rel_tol") = static_cast<double>(kNewtonDefaultRelTol),
          py::arg("newton_abs_tol") = static_cast<double>(kNewtonDefaultAbsTol),
          py::arg("newton_fd_eps") = static_cast<double>(kNewtonDefaultFdEps),
          py::arg("newton_damping") = static_cast<double>(kNewtonDefaultDamping),
          py::arg("newton_fail_policy") = "none",
          py::arg("newton_diagnostics") = false,
          // Zhang-Shu positivity floor (ADC-259): Density-role face-state + C/F-ghost-mean floor on
          // the AMR transport. 0 (default) = inactive, bit-identical. Marshaled from spatial.positivity_floor
          // by the AmrSystem.add_block / add_equation Python facade.
          py::arg("positivity_floor") = 0.0)
      // Newton report (IMEX diagnostics OPT-IN, native MULTI-BLOCK): dict {enabled, converged,
      // max_residual, max_iters_used, n_failed, failed_cell, failed_component}, aggregated over the
      // levels/substeps of the LAST advance of the block. failed_cell = (i, j) or None. EXACT shape of
      // the System.newton_report binding (parity, including failed_cell tuple/None).
      .def(
          "newton_report",
          [](AmrSystem& s, const std::string& name) {
            const AmrSystem::SourceNewtonReport r = s.newton_report(name);
            py::dict d;
            d["enabled"] = r.enabled;
            d["converged"] = r.converged;
            d["max_residual"] = r.max_residual;
            d["max_iters_used"] = r.max_iters_used;
            d["n_failed"] = r.n_failed;
            if (r.failed_i >= 0)
              d["failed_cell"] =
                  py::make_tuple(static_cast<int>(r.failed_i), static_cast<int>(r.failed_j));
            else
              d["failed_cell"] = py::none();
            d["failed_component"] = static_cast<int>(r.failed_comp);
            py::list diagnostics;
            for (const RuntimeDiagnosticEvent& event : r.diagnostics) {
              py::dict row;
              row["code"] = event.code;
              row["component"] = event.component;
              row["severity"] = event.severity;
              row["message"] = event.message;
              row["iteration"] = event.iteration;
              row["value"] = event.value;
              diagnostics.append(row);
            }
            d["diagnostics"] = diagnostics;
            return d;
          },
          py::arg("name"))
      // NATIVE AMR block loaded from a .so loader generated by the DSL (backend "production",
      // target="amr_system"): the .so inlines add_compiled_model(AmrSystem&) -> native block on the
      // AMR hierarchy (reflux, regrid), ABI key verified. cf. AmrSystem::add_native_block. NO
      // evolve (mono-block AMR). The AMR LIMITS (primitive/roe/hllc/weno5) are guarded on the Python
      // facade side (AmrSystem.add_equation) before this binding.
      .def("add_native_block", &AmrSystem::add_native_block, py::arg("name"), py::arg("so_path"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit",
           py::arg("gamma") = static_cast<double>(kPhysicalDefaultGamma),
           py::arg("substeps") = 1,
           // Zhang-Shu positivity floor (ADC-322): marshaled down the regenerated .so loader
           // (pops_install_native_amr). 0 (default) = inactive, bit-identical.
           py::arg("positivity_floor") = 0.0)
      // Regrid criterion: refine where the SELECTED variable exceeds threshold. Default = component 0
      // (historical density), bit-identical 1e30 no-op. ADC-296: select it PER BLOCK by NAME (variable=)
      // or physical ROLE (role=); a block lacking it raises at build (no silent comp-0 fallback). A
      // non-default selector is MULTI-BLOCK only (mono-block / compiled .so refine on component 0).
      .def("set_refinement", &AmrSystem::set_refinement, py::arg("threshold"),
           py::arg("variable") = "", py::arg("role") = "",
           "Refine where the selected conserved variable exceeds threshold. variable=/role= pick "
           "it per "
           "block by name or physical role (default: component 0, the historical density). "
           "Selecting by "
           "name and role at once, or a name/role absent from a block, raises. Non-default "
           "selector is "
           "multi-block only.")
      .def("_set_bootstrap_refinement", &AmrSystem::set_bootstrap_refinement,
           py::arg("block"), py::arg("variable"), py::arg("threshold"),
           py::arg("provider_identity"))
      // PHI tag on |grad phi| (D4) added to the union of regrid tags: also refines where the
      // norm of the potential gradient exceeds grad_threshold (diocotron ring edge). MULTI-BLOCK
      // + regrid_every > 0. <= 0 (default) -> phi DISABLED (bit-identical). cf. AmrSystem::set_phi_refinement.
      .def("set_phi_refinement", &AmrSystem::set_phi_refinement, py::arg("grad_threshold"))
      .def(
          "set_poisson", &AmrSystem::set_poisson,
          "Configures the coarse Poisson of the AMR hierarchy (cf. System.set_poisson). On AMR the "
          "solver is ALWAYS GeometricMG and the right-hand side ALWAYS the sum of the elliptic "
          "bricks. rhs: 'charge_density' | 'composite'. solver: 'geometric_mg' only (no "
          "FFT on the hierarchy). bc: 'auto' | 'periodic' | 'dirichlet' | 'neumann'. wall: "
          "'none' | 'circle' (circular conducting wall, requires wall_radius > 0). "
          "composite (ADC-645): True opts the FIELD solve into the composite FAC path (the fine "
          "patch refines the elliptic); scope = single block, 2 levels, one mono-box fine patch, "
          "replicated coarse -- out of scope REFUSES at build (never a silent fallback). The fac_* "
          "knobs (<= 0 = the kFAC* defaults) tune that solve; "
          "inert when composite is False (the historical Option A solve, bit-identical).",
          py::arg("rhs") = "charge_density", py::arg("solver") = "geometric_mg",
          py::arg("bc") = "auto", py::arg("wall") = "none", py::arg("wall_radius") = 0.0,
          py::arg("composite") = false, py::arg("fac_max_iters") = 0,
          py::arg("fac_fine_sweeps") = 0, py::arg("fac_tol") = 0.0,
          py::arg("fac_coarse_rel_tol") = 0.0, py::arg("fac_coarse_cycles") = 0,
          py::arg("fac_verbose") = false)
      .def("set_field_solver_plan", &AmrSystem::set_field_solver_plan,
           py::arg("provider_slot"), py::arg("provider_identity"),
           py::arg("output_owner_identity"),
           py::arg("output_block"), py::arg("output_key"),
           py::arg("provider_identities"), py::arg("provider_blocks"),
           py::arg("provider_keys"), py::arg("provider_coefficients"),
           py::arg("solver"), py::arg("hierarchy"), py::arg("abs_tol"),
           py::arg("rel_tol"), py::arg("max_cycles"), py::arg("min_coarse"),
           py::arg("pre_smooth"), py::arg("post_smooth"), py::arg("bottom_sweeps"),
           py::arg("coarse_threshold"))
      .def("set_field_boundary_plan", &AmrSystem::set_field_boundary_plan,
           py::arg("provider_slot"), py::arg("kind"), py::arg("alpha"), py::arg("beta"),
           py::arg("value"))
      .def("set_field_boundary_dependencies", &AmrSystem::set_field_boundary_dependencies,
           py::arg("provider_slot"), py::arg("state_blocks"),
           py::arg("state_components"), py::arg("field_blocks"),
           py::arg("field_keys"), py::arg("field_components"))
      .def("set_field_boundary_parameters", &AmrSystem::set_field_boundary_parameters,
           py::arg("provider_slot"), py::arg("parameters"))
      .def("set_field_newton_plan", &AmrSystem::set_field_newton_plan,
           py::arg("provider_slot"), py::arg("tolerance"), py::arg("max_iterations"),
           py::arg("linear_tolerance"), py::arg("linear_max_iterations"),
           py::arg("restart"), py::arg("armijo"), py::arg("minimum_step"))
      .def("set_field_nullspace", &AmrSystem::set_field_nullspace,
           py::arg("provider_slot"), py::arg("constant_kernel"),
           py::arg("mean_zero_gauge"));
}

// Physics wiring: dt bounds, fields, and coupled source stages.
void bind_amr_physics(py::class_<AmrSystem>& cls) {
  cls
      // GLOBAL step bound + ACTIVE bound (AMR StabilityPolicy, System.add_dt_bound parity).
      .def("add_dt_bound", &AmrSystem::add_dt_bound, py::arg("label"), py::arg("fn"))
      .def("last_dt_bound", &AmrSystem::last_dt_bound)
      // B_z accepts a flattened numpy (n, n) and populates the Program-visible aux channel.
      .def(
          "set_magnetic_field",
          [](AmrSystem& s, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_magnetic_field(flat(arr));
          },
          py::arg("bz"))
      // ADC-291: model-NAMED aux field at a resolved channel component (>= kAuxNamedBase). The Python
      // facade (AmrSystem.set_aux_field) resolves the name -> comp and reshapes (n, n) -> flat n*n.
      .def(
          "set_aux_field_component",
          [](AmrSystem& s, int comp,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_aux_field_component(comp, flat(arr));
          },
          py::arg("comp"), py::arg("field"))
      // ADC-369: per-field aux halo policy (bc_type = pops::BCType Foextrap=1 / Dirichlet=2).
      .def(
          "set_aux_field_halo_component",
          [](AmrSystem& s, int comp, int bc_type, double value) {
            s.set_aux_field_halo_component(comp, bc_type, value);
          },
          py::arg("comp"), py::arg("bc_type"), py::arg("value"))
      .def(
          "set_density",
          [](AmrSystem& s, const std::string& name,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_density(name, flat(arr));
          },
          py::arg("name"), py::arg("rho"))
      // Full initial conservative state (ncomp, n, n) -> starts the AMR from the paper's drift
      // state (rho, rho*u, rho*v) instead of m=0. Keeps ndim==3 EXPLICIT: flat() flattens
      // any C-contiguous array, so a 2D density (n, n) passed by mistake would become a
      // 1-component state (comp 0 = density, momentum left at 0) -- a silent density masquerade
      // with the wrong physics. We require (ncomp, n, n). flat() then flattens in
      // component-major c*n*n + j*n + i (same convention as to_3d / set_state).
      .def(
          "set_conservative_state",
          [](AmrSystem& s, const std::string& name,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            if (arr.ndim() != 3)
              throw std::runtime_error(
                  "AmrSystem.set_conservative_state: state expected of shape (ncomp, n, n); got "
                  "a " +
                  std::to_string(arr.ndim()) +
                  "D array (a 2D density? use "
                  "set_density)");
            s.set_conservative_state(name, flat(arr));
          },
          py::arg("name"), py::arg("U"))
      .def("_begin_bootstrap_plan", &AmrSystem::begin_bootstrap_plan)
      .def("_bootstrap_next_level", &AmrSystem::bootstrap_next_level)
      .def("_commit_bootstrap_level", &AmrSystem::commit_bootstrap_level)
      .def("_rollback_bootstrap_level", &AmrSystem::rollback_bootstrap_level)
      .def("_register_bootstrap_transfer_route",
           &AmrSystem::register_bootstrap_transfer_route,
           py::arg("identity"), py::arg("subjects"), py::arg("provider_identity"),
           py::arg("space"), py::arg("centering"), py::arg("representation"),
           py::arg("storage"), py::arg("operation"), py::arg("kernel"),
           py::arg("order"), py::arg("ghost_depth"), py::arg("dimension"),
           py::arg("refinement_ratio"))
      .def("_register_bootstrap_face_vector", &AmrSystem::register_bootstrap_face_vector,
           py::arg("subjects"))
      .def(
          "_register_bootstrap_array",
          [](AmrSystem& s, const std::string& subject, const std::string& centering,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            if (arr.ndim() != 3)
              throw std::runtime_error(
                  "AmrSystem._register_bootstrap_array expects (ncomp, ny, nx)");
            s.register_bootstrap_array(subject, centering, static_cast<int>(arr.shape(0)),
                                       static_cast<int>(arr.shape(1)),
                                       static_cast<int>(arr.shape(2)), flat(arr));
          },
          py::arg("subject"), py::arg("centering"), py::arg("values"))
      .def("_bind_bootstrap_block_subject", &AmrSystem::bind_bootstrap_block_subject,
           py::arg("subject"), py::arg("block"))
      .def("_register_analytic_constant", &AmrSystem::register_analytic_constant,
           py::arg("subject"), py::arg("block"), py::arg("space"),
           py::arg("centering"), py::arg("components"))
      .def("_bootstrap_analytic_reproject", &AmrSystem::bootstrap_analytic_reproject,
           py::arg("subject"), py::arg("level"))
      .def("_apply_bootstrap_component_floor", &AmrSystem::apply_bootstrap_component_floor,
           py::arg("subject"), py::arg("level"), py::arg("component"), py::arg("floor"))
      .def("_recompute_bootstrap_field", &AmrSystem::recompute_bootstrap_field,
           py::arg("subject"), py::arg("field_name"))
      .def("_bootstrap_prolong_array", &AmrSystem::bootstrap_prolong_array,
           py::arg("subject"), py::arg("level"))
      .def("_synchronize_bootstrap_state", &AmrSystem::synchronize_bootstrap_state,
           py::arg("subject"), py::arg("fine_level"))
      .def("_bootstrap_array_level", &AmrSystem::bootstrap_array_level,
           py::arg("subject"), py::arg("level"))
      .def("_invalidate_bootstrap_cache", &AmrSystem::invalidate_bootstrap_cache,
           py::arg("subject"), py::arg("level"))
      .def(
          "_rebuild_bootstrap_topology_cache",
          [](AmrSystem& s, const std::string& subject, int level) {
            py::list out;
            for (const pops::PatchBox& b :
                 s.rebuild_bootstrap_topology_cache(subject, level))
              out.append(py::make_tuple(b.level, b.ilo, b.jlo, b.ihi, b.jhi));
            return out;
          },
          py::arg("subject"), py::arg("level"))
      .def("_bootstrap_cache_epoch", &AmrSystem::bootstrap_cache_epoch,
           py::arg("subject"))
      // Inter-species COUPLED source (compiled pops.dsl.CoupledSource, P5 bytecode), MULTI-BLOCK on the
      // SHARED AMR hierarchy: applied after the transport at each macro-step, by explicit
      // splitting, level by level + fine -> coarse cascade (consistent covered cells). SAME
      // flat ABI as System.add_coupled_source. Without the call, unchanged. cf. AmrSystem::add_coupled_source.
      // ADC-214: Python surface UNCHANGED (same flat kwargs, same defaults). The lambda assembles the
      // CoupledSourceProgram POD before the C++ call (parity with System.add_coupled_source).
      // INTERNAL raw coupled-source ABI (ADC-595): flat 12-kwarg bytecode form, called only by the
      // typed lowering (AmrSystem.add_coupling -> add_coupling_operator) and low-level tests. End users
      // register through sim.add_coupling(...); parity with System._add_coupled_source.
      .def(
          "_add_coupled_source",
          [](AmrSystem& s, const std::vector<std::string>& in_blocks,
             const std::vector<std::string>& in_roles, const std::vector<double>& consts,
             const std::vector<std::string>& out_blocks, const std::vector<std::string>& out_roles,
             const std::vector<int>& prog_ops, const std::vector<int>& prog_args,
             const std::vector<int>& prog_lens, double frequency, const std::string& label,
             const std::vector<int>& freq_prog_ops, const std::vector<int>& freq_prog_args) {
            CoupledSourceProgram prog{in_blocks,     in_roles,      consts,    out_blocks,
                                      out_roles,     prog_ops,      prog_args, prog_lens,
                                      freq_prog_ops, freq_prog_args};
            s.add_coupled_source(prog, frequency, label);
          },
          py::arg("in_blocks"), py::arg("in_roles"), py::arg("consts"), py::arg("out_blocks"),
          py::arg("out_roles"), py::arg("prog_ops"), py::arg("prog_args"), py::arg("prog_lens"),
          py::arg("frequency") = 0.0, py::arg("label") = "coupled_source",
          // Optional PER-CELL frequency mu(U): evaluated on the coarse level (cf. System).
          py::arg("freq_prog_ops") = std::vector<int>{},
          py::arg("freq_prog_args") = std::vector<int>{})
      // Typed COUPLING OPERATOR (ADC-595, parity with System): the same flat program PLUS the DECLARED
      // conservation contract (conserved / created roles) and frequency bound, validated at
      // registration (host, fail-loud) then lowered through the SAME add_coupled_source path.
      .def(
          "add_coupling_operator",
          [](AmrSystem& s, const std::vector<std::string>& in_blocks,
             const std::vector<std::string>& in_roles, const std::vector<double>& consts,
             const std::vector<std::string>& out_blocks, const std::vector<std::string>& out_roles,
             const std::vector<int>& prog_ops, const std::vector<int>& prog_args,
             const std::vector<int>& prog_lens, double frequency, const std::string& label,
             const std::vector<int>& freq_prog_ops, const std::vector<int>& freq_prog_args,
             const std::vector<std::string>& conserved_roles,
             const std::vector<std::string>& created_roles) {
            CouplingOperator op;
            op.label = label;
            op.program = CoupledSourceProgram{in_blocks,     in_roles,      consts,    out_blocks,
                                              out_roles,     prog_ops,      prog_args, prog_lens,
                                              freq_prog_ops, freq_prog_args};
            op.conservation.conserved_roles = conserved_roles;
            op.conservation.created_roles = created_roles;
            op.frequency.constant_mu = frequency;
            op.frequency.per_cell = !freq_prog_ops.empty() || !freq_prog_args.empty();
            s.add_coupling_operator(op);
          },
          py::arg("in_blocks"), py::arg("in_roles"), py::arg("consts"), py::arg("out_blocks"),
          py::arg("out_roles"), py::arg("prog_ops"), py::arg("prog_args"), py::arg("prog_lens"),
          py::arg("frequency") = 0.0, py::arg("label") = "coupled_source",
          py::arg("freq_prog_ops") = std::vector<int>{},
          py::arg("freq_prog_args") = std::vector<int>{},
          py::arg("conserved_roles") = std::vector<std::string>{},
          py::arg("created_roles") = std::vector<std::string>{})
      // Read-only view of the registered coupling operators (ADC-595): one dict per coupling
      // {label, conserved_roles, created_roles, frequency_mu, per_cell_frequency}.
      .def("coupled_operators", [](const AmrSystem& s) {
        py::list out;
        for (const CouplingOperatorView& v : s.coupled_operators()) {
          py::dict row;
          row["label"] = v.label;
          row["conserved_roles"] = v.conservation.conserved_roles;
          row["created_roles"] = v.conservation.created_roles;
          row["frequency_mu"] = v.frequency.constant_mu;
          row["per_cell_frequency"] = v.frequency.per_cell;
          out.append(row);
        }
        return out;
      });
}

// Stepping + profiling: step/advance/CFL/adaptive and the profiler surface.
void bind_amr_stepping(py::class_<AmrSystem>& cls) {
  cls.def("step", &AmrSystem::step, py::arg("dt"))
      .def("advance", &AmrSystem::advance, py::arg("dt"), py::arg("nsteps"))
      .def("step_cfl", &AmrSystem::step_cfl,
           "Advances by one AMR macro-step at dt = cfl * dx_coarse / max wave speed (also honors "
           "the substeps/stride cadence in multi-block and the optional bounds). Returns the dt "
           "used. speed_floor (ADC-645): the floor applied to the reduced max wave speed on the "
           "multi-block runtime engine (default = the historical kCflSpeedFloor, bit-identical); "
           "refused non-default on the single-block coupler (no historical floor site there).",
           py::arg("cfl"), py::arg("speed_floor") = static_cast<double>(kCflSpeedFloor))
      // AMR / MPI profiling (Spec 5 criterion 43, ADC-479): the multi-block engine times its
      // non-numeric phases (regrid / fill_boundary / average_down) + MPI counters into the
      // facade-owned Profiler. PerformanceSummary.by_amr_mpi() surfaces them. Off by default.
      .def("enable_profiling", &AmrSystem::enable_profiling,
           "Spec 5 profiling (ADC-479): time the AMR phases (regrid, fill_boundary, average_down) "
           "and MPI counters. Disabled by default; off the hot path when off.")
      .def("disable_profiling", &AmrSystem::disable_profiling,
           "Stop profiling (keeps accumulated data).")
      .def("is_profiling", &AmrSystem::is_profiling)
      .def("reset_profiling", &AmrSystem::reset_profiling, "Clear accumulated profiling data.")
      .def("profile_report", &AmrSystem::profile_report,
           "Per-phase wall-clock report of the AMR runtime (count / total / mean / min / max per "
           "scope, plus counters regrid / fill_boundary / mpi_reductions / mpi_messages). "
           "Per-rank.")
      .def("profile_snapshot",
           [](AmrSystem& s) { return profile_snapshot_to_dict(s.profiler_handle().snapshot()); },
           "Structured AMR profiling snapshot: schema_version, enabled, scopes and counters.");
}

// Clock + compiled-Program install/introspection + runtime freeze lifecycle.
void bind_amr_program(py::class_<AmrSystem>& cls) {
  cls.def("nx", &AmrSystem::nx)
      .def("time", &AmrSystem::time)
      // AMR clock (IO v1, System parity): macro-step counter + restoration (t, macro_step) ->
      // the regrid/stride cadence resumes exactly after a set_clock. Prerequisite PR-IO-3.
      .def("macro_step", &AmrSystem::macro_step)
      .def("set_clock", &AmrSystem::set_clock, py::arg("t"), py::arg("macro_step"))
      .def("field_provider_slots", &AmrSystem::field_provider_slots)
      .def("field_provider_levels", &AmrSystem::field_provider_levels,
           py::arg("provider_slot"))
      .def("set_field_potential", &AmrSystem::set_field_potential,
           py::arg("provider_slot"), py::arg("phi"))
      .def("set_field_potential_level", &AmrSystem::set_field_potential_level,
           py::arg("provider_slot"), py::arg("level"), py::arg("phi"))
      // Compiled time Program on the AMR hierarchy (epic ADC-511 / ADC-508, Spec 6): dlopen a generated
      // problem.so (target='amr_system'), verify its ABI key, run the section-24 requirement validation
      // (block instance / solver), bind the Program blocks by name, seed the runtime params, and install
      // the per-level Lie/Strang macro-step body. The block(s) must already exist (add_equation). cf.
      // AmrSystem::install_program (the AMR counterpart of System::install_program).
      .def("install_program", &AmrSystem::install_program, py::arg("so_path"))
      // Compiled-Program macro-step cadence (parity System::set_program_cadence, ADC-411): GLOBAL
      // substeps + stride around the installed program closure. Both must be >= 1. Separate from
      // install_program so the .so ABI is untouched; CompiledTime(substeps=, stride=) threads here.
      .def("set_program_cadence", &AmrSystem::set_program_cadence, py::arg("substeps"),
           py::arg("stride"))
      // ADC-594: read the installed GLOBAL cadence (substeps / stride) for the ProgramRuntimeReport.
      // Const getters (default 1/1 with no program); there was no Python-visible getter before.
      .def("program_substeps", &AmrSystem::program_substeps)
      .def("program_stride", &AmrSystem::program_stride)
      // Changes the RUNTIME parameters of a compiled time PROGRAM block WITHOUT recompiling the .so
      // (ADC-508, parity ADC-510). prog_block = the PROGRAM block index (P.state order); values = that
      // block's params in sorted-name order. Python's _install_program_params routes params={name: value}
      // here. cf. AmrSystem::set_program_params.
      .def("set_program_params", &AmrSystem::set_program_params, py::arg("prog_block"),
           py::arg("values"))
      // Changes the NATIVE per-block RUNTIME parameters of a production block WITHOUT recompiling the
      // .so (ADC-514, parity System.set_block_params). name = the block name; values = that block's
      // params in the model's runtime_param_names order. Python's step-4b route in _amr_system_install
      // sends params={name: value} here after validating the names against pops_compiled_param_names.
      .def("set_block_params", &AmrSystem::set_block_params, py::arg("name"), py::arg("values"))
      // IR hash of the installed compiled Program (the .so's pops_program_hash), or "" if none. Parity
      // System::installed_program_hash (the checkpoint guard).
      .def("installed_program_hash", &AmrSystem::installed_program_hash)
      .def("program_accepted_state", [](const AmrSystem& s) {
        const auto bytes = s.program_accepted_state();
        return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      })
      .def("restore_program_accepted_state", [](AmrSystem& s, py::bytes payload) {
        std::string bytes = payload;
        s.restore_program_accepted_state(
            std::vector<std::uint8_t>(bytes.begin(), bytes.end()));
      }, py::arg("payload"))
      .def("program_accepted_state_manifest", &AmrSystem::program_accepted_state_manifest)
      // ADC-631: True on the multi-block AmrRuntime engine (a compiled Program forces it even for ONE
      // block), False on the single-block coupler. The v3 checkpoint routes per-block vs mono state I/O
      // on this (n_blocks()==1 does not imply the coupler under a Program).
      .def("uses_runtime_engine", &AmrSystem::uses_runtime_engine)
      // ADC-414 / ADC-542: scalar Program diagnostics (parity System). program_diagnostic(name) reads
      // one, program_diagnostics() the whole map; record_program_diagnostic is the sink the diagnostics
      // driver records a measured scalar into each cadence tick.
      .def("program_diagnostic", &AmrSystem::program_diagnostic, py::arg("name"))
      .def("program_diagnostics", &AmrSystem::program_diagnostics)
      .def("record_program_diagnostic", &AmrSystem::record_program_diagnostic, py::arg("name"),
           py::arg("value"))
      // ADC-542: the level-composite collective reduction over a named block the AMR diagnostics
      // path drives -- volume-weighted sums with covered-cell exclusion, extrema folded over all
      // levels (a covered coarse cell is the average of its children, provably inside their extrema).
      .def("composite_reduce", &AmrSystem::composite_reduce, py::arg("block"), py::arg("kind"),
           py::arg("comp") = 0)
      // ADC-592: runtime freeze lifecycle (parity with System). mark_bound() (called LAST by the
      // Python bind flow) freezes the composition; lifecycle_state() reports assembling / bound /
      // running (running derived from macro_step()).
      .def("mark_bound", &AmrSystem::mark_bound)
      .def("lifecycle_state", &AmrSystem::lifecycle_state);
}

// Data + IO accessors: block/patch introspection, mass/density/potential, level/var shape.
void bind_amr_data(py::class_<AmrSystem>& cls) {
  cls.def("n_blocks", &AmrSystem::n_blocks)
      .def("block_names", &AmrSystem::block_names)
      .def("effective_options_report",
           [](const AmrSystem& s) {
             return effective_options_report_to_dict(s.effective_options_report());
           },
           "Structured effective numerical/solver/physical options for this AmrSystem.")
      .def("n_patches", &AmrSystem::n_patches)
      // Index-space footprints of the fine patches: list of tuples (level, ilo, jlo, ihi, jhi), INCLUSIVE
      // corners, in the index space of the level (n << level cells/direction, ratio 2). SAME
      // source as n_patches() (the GLOBAL fine BoxArray) -> rank-independent, MPI-safe. Query between
      // steps, zero cost on the hot path. The Python wrapper converts to [0, L]^2 (it knows n via nx() and
      // L); cf. AmrSystem.patch_rectangles() on the facade side.
      .def("patch_boxes",
           [](AmrSystem& s) {
             py::list out;
             for (const pops::PatchBox& b : s.patch_boxes())
               out.append(py::make_tuple(b.level, b.ilo, b.jlo, b.ihi, b.jhi));
             return out;
           })
      // COARSE-level (base) box counts (ADC-319, MPI ownership diagnostic): coarse_local_boxes() = base
      // boxes OWNED by this rank (level-0 local_size()); coarse_total_boxes() = total base boxes (BoxArray
      // size, identical on all ranks). distribute_coarse=True -> local < total per rank (distributed
      // coarse transport); replicated / single-box -> local == total. Query between steps, no hot cost.
      .def("coarse_local_boxes", &AmrSystem::coarse_local_boxes)
      .def("coarse_total_boxes", &AmrSystem::coarse_total_boxes)
      // mass / density: overload by BLOCK NAME (multi-block; empty name -> 1st block, mono-block
      // compat or cosmetic name). The name INDEXES the block in multi-block (each block has its mass /
      // density, conserved PER BLOCK at reflux). Without argument -> 1st block (mono-block back-compat).
      .def("mass", [](AmrSystem& s) { return s.mass(); })
      .def(
          "mass", [](AmrSystem& s, const std::string& name) { return s.mass(name); },
          py::arg("name"))
      // AMR: SQUARE domain (n x n), no polar geometry -> rows == cols == nx() (unchanged).
      .def("density", [](AmrSystem& s) { return to_2d(s.density(), s.nx(), s.nx()); })
      .def(
          "density",
          [](AmrSystem& s, const std::string& name) {
            return to_2d(s.density(name), s.nx(), s.nx());
          },
          py::arg("name"))
      // phi of the coarse (base) level, (n, n). SAME observable as System.potential(): level 0
      // covers the whole domain -> enough to sample a median circle (azimuthal FFT). In
      // multi-block, phi results from the SYSTEM Poisson (Sum_b q_b n_b co-located), shared by all.
      .def("potential", [](AmrSystem& s) { return to_2d(s.potential(), s.nx(), s.nx()); })
      // ADC-428: solved potential of a NAMED elliptic field (m.elliptic_field) on the coarse level,
      // (n, n). Read-back counterpart of potential() for a second elliptic field; the Python
      // AmrSystem.field(name) resolves the field name to this. Solves the hierarchy if needed.
      .def(
          "named_field_values",
          [](AmrSystem& s, const std::string& field) {
            return to_2d(s.named_field_values(field), s.nx(), s.nx());
          },
          py::arg("field"))
      // AMR CHECKPOINT / RESTART single-rank (ADC-65): full conservative state per level + phi
      // (warm-start) + imposition of the saved fine hierarchy. SERIAL MONO-BLOCK (multi-block: C++
      // rejection; np>1: facade rejection -- per-level gather = future). level_state / level_potential return
      // FLAT fields (c*nf*nf + j*nf + i / nf*nf, nf = nx << k); the facade reshapes. set_*
      // flatten any C-contiguous array (flat). set_hierarchy: list of tuples
      // (level, ilo, jlo, ihi, jhi) like patch_boxes() (the coupler filters level 1).
      .def("n_levels", &AmrSystem::n_levels)
      .def("n_vars", [](AmrSystem& s) { return s.n_vars(); })
      .def(
          "level_state", [](AmrSystem& s, int k) { return s.level_state(k); }, py::arg("k"))
      .def(
          "set_level_state",
          [](AmrSystem& s, int k,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_level_state(k, flat(arr));
          },
          py::arg("k"), py::arg("state"))
      .def(
          "level_potential", [](AmrSystem& s, int k) { return s.level_potential(k); }, py::arg("k"))
      .def(
          "set_level_potential",
          [](AmrSystem& s, int k,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_level_potential(k, flat(arr));
          },
          py::arg("k"), py::arg("phi"))
      .def(
          "set_hierarchy",
          [](AmrSystem& s, const std::vector<std::tuple<int, int, int, int, int>>& boxes) {
            std::vector<pops::PatchBox> bx;
            bx.reserve(boxes.size());
            for (const auto& b : boxes)
              bx.push_back(pops::PatchBox{std::get<0>(b), std::get<1>(b), std::get<2>(b),
                                         std::get<3>(b), std::get<4>(b)});
            s.set_hierarchy(bx);
          },
          py::arg("boxes"))
      // GLOBAL (np>1 gather) variants of the per-level accessors (ADC-509): the checkpoint facade
      // routes to them under MPI np>1 so the distributed fabs are gathered onto rank 0 (COLLECTIVE:
      // all ranks call). Mono-rank they return the same array as the non-global accessors.
      .def(
          "level_state_global", [](AmrSystem& s, int k) { return s.level_state_global(k); },
          py::arg("k"))
      .def(
          "level_potential_global", [](AmrSystem& s, int k) { return s.level_potential_global(k); },
          py::arg("k"))
      .def("field_potential_global", &AmrSystem::field_potential_global,
           py::arg("provider_slot"))
      .def("field_potential_level_global", &AmrSystem::field_potential_level_global,
           py::arg("provider_slot"), py::arg("level"))
      // MULTI-BLOCK per-BLOCK per-level state (ADC-509): the AmrRuntime engine shares the layout +
      // aux, so the per-level STATE is read/restored PER BLOCK (by name) while phi stays shared
      // (level_potential). block_level_state returns a FLAT field (c*nf*nf + j*nf + i); the _global
      // variant gathers under np>1; set_block_level_state flattens any C-contiguous array.
      .def(
          "block_n_vars", [](AmrSystem& s, const std::string& name) { return s.block_n_vars(name); },
          py::arg("name"))
      .def(
          "block_level_state",
          [](AmrSystem& s, const std::string& name, int k) { return s.block_level_state(name, k); },
          py::arg("name"), py::arg("k"))
      .def(
          "block_level_state_global",
          [](AmrSystem& s, const std::string& name, int k) {
            return s.block_level_state_global(name, k);
          },
          py::arg("name"), py::arg("k"))
      .def(
          "set_block_level_state",
          [](AmrSystem& s, const std::string& name, int k,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_block_level_state(name, k, flat(arr));
          },
          py::arg("name"), py::arg("k"), py::arg("state"))
      // ADC-542: owner rank per box of a level (the shared DistributionMapping), for the v3 checkpoint
      // to reproduce the local-fab iteration order at restart. Empty on the single-block coupler path.
      .def(
          "level_owner_ranks", [](AmrSystem& s, int k) { return s.level_owner_ranks(k); },
          py::arg("k"))
      // ADC-542: the FULL shared aux of a level (ALL components, flat c*nf*nf+j*nf+i) -- the v3
      // checkpoint aux payload. _global gathers under np>1 (COLLECTIVE); the setter restores the
      // valid cells owner-rank. Empty read / throwing write on the single-block coupler path.
      .def(
          "level_aux_flat", [](AmrSystem& s, int k) { return s.level_aux_flat(k); }, py::arg("k"))
      .def(
          "level_aux_flat_global", [](AmrSystem& s, int k) { return s.level_aux_flat_global(k); },
          py::arg("k"))
      .def(
          "set_level_aux_flat",
          [](AmrSystem& s, int k,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_level_aux_flat(k, flat(arr));
          },
          py::arg("k"), py::arg("aux"))
      // ADC-542: impose a mid-run MULTI-BLOCK hierarchy from a v3 checkpoint. @p boxes are the
      // level-tagged patch signatures (level, ilo, jlo, ihi, jhi); @p owner_ranks is the per-box owner
      // rank aligned with @p boxes. Routes to AmrRuntime::rebuild_hierarchy (all levels rebuilt).
      .def(
          "rebuild_hierarchy",
          [](AmrSystem& s, const std::vector<std::tuple<int, int, int, int, int>>& boxes,
             const std::vector<int>& owner_ranks) {
            std::vector<pops::PatchBox> bx;
            bx.reserve(boxes.size());
            for (const auto& b : boxes)
              bx.push_back(pops::PatchBox{std::get<0>(b), std::get<1>(b), std::get<2>(b),
                                         std::get<3>(b), std::get<4>(b)});
            s.rebuild_hierarchy(bx, owner_ranks);
          },
          py::arg("boxes"), py::arg("owner_ranks"))
      .def("begin_restart_transaction", &AmrSystem::begin_restart_transaction)
      .def("commit_restart_transaction", &AmrSystem::commit_restart_transaction)
      .def("rollback_restart_transaction", &AmrSystem::rollback_restart_transaction)
      .def("checkpoint_regrid_count", &AmrSystem::checkpoint_regrid_count)
      .def("checkpoint_topology_epoch", &AmrSystem::checkpoint_topology_epoch)
      .def("restore_checkpoint_counters", &AmrSystem::restore_checkpoint_counters,
           py::arg("regrid_count"), py::arg("topology_epoch"))
      .def("checkpoint_temporal_ratios", &AmrSystem::checkpoint_temporal_ratios)
      .def("checkpoint_transfer_routes", &AmrSystem::checkpoint_transfer_routes)
      // ADC-631 multistep history rings on the compiled-Program AMR route: the SAME seam names as
      // System (init_system.cpp) so _system_io_history.py serialize/restore is reused verbatim.
      // history_global returns the per-level slices concatenated into ONE flat buffer (level axis
      // hidden inside the facade, parity level_aux_flat); restore_history flattens any C-contiguous
      // array and scatters it back per level; rebuild_history_slots replays the recomputed slots.
      .def("history_names", &AmrSystem::history_names)
      .def("history_depth", &AmrSystem::history_depth, py::arg("name"))
      .def("history_ncomp", &AmrSystem::history_ncomp, py::arg("name"))
      .def(
          "history_global",
          [](const AmrSystem& s, const std::string& name, int slot) {
            return s.history_global(name, slot);
          },
          py::arg("name"), py::arg("slot"))
      .def("history_initialized", &AmrSystem::history_initialized, py::arg("name"))
      .def(
          "restore_history",
          [](AmrSystem& s, const std::string& name, int slot,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.restore_history(name, slot, flat(arr));
          },
          py::arg("name"), py::arg("slot"), py::arg("values"))
      .def("set_history_initialized", &AmrSystem::set_history_initialized, py::arg("name"),
           py::arg("initialized"))
      .def("history_slot_dt", &AmrSystem::history_slot_dt, py::arg("name"), py::arg("slot"))
      .def("restore_history_slot_dt", &AmrSystem::restore_history_slot_dt, py::arg("name"),
           py::arg("slot"), py::arg("dt"))
      .def("rebuild_history_slots", &AmrSystem::rebuild_history_slots, py::arg("name"),
           py::arg("stored_slots"))
      .def("last_replay_regrid_steps", &AmrSystem::last_replay_regrid_steps);
}

}  // namespace

// Registers AmrSystemConfig, then the AmrSystem facade and each concern's bindings IN ORDER (assembly
// first, so the class exists before the other groups extend it). The per-concern order matches the
// historical single chain; no overload set spans two concerns.
void init_amr(py::module_& m) {
  // --- AMR: single-species composition on multi-patch AMR (generic composable brick) ---
  // adc_cases DRIVES it from Python (no C++ on the cases side) just like System.
  //
  // NB: the two-fluid AP integrator (BESPOKE asymptotic-preserving scheme, not composable
  // block by block) has left the core: it is not a generic brick but a SCENARIO. It now lives
  // in adc_cases (cf. adc_cases/two_fluid_ap/), compiled on the fly against the generic
  // headers of PoPS; it is no longer exposed by the _pops module.
  py::class_<AmrSystemConfig>(m, "AmrSystemConfig")
      .def(py::init<>())
      .def_readwrite("n", &AmrSystemConfig::n)
      .def_readwrite("L", &AmrSystemConfig::L)
      .def_readwrite("regrid_every", &AmrSystemConfig::regrid_every)
      .def_readwrite("level_count", &AmrSystemConfig::level_count)
      .def_readwrite("regrid_grow", &AmrSystemConfig::regrid_grow)
      .def_readwrite("regrid_margin", &AmrSystemConfig::regrid_margin)
      .def_readwrite("explicit_bootstrap", &AmrSystemConfig::explicit_bootstrap)
      .def_readwrite("periodic", &AmrSystemConfig::periodic)
      .def_readwrite("distribute_coarse", &AmrSystemConfig::distribute_coarse)
      .def_readwrite("coarse_max_grid", &AmrSystemConfig::coarse_max_grid)
      // ADC-616: Berger-Rigoutsos clustering params (<= 0 = the historical {0.7, 1, 32} default).
      .def_readwrite("cluster_min_efficiency", &AmrSystemConfig::cluster_min_efficiency)
      .def_readwrite("cluster_min_box_size", &AmrSystemConfig::cluster_min_box_size)
      .def_readwrite("cluster_max_box_size", &AmrSystemConfig::cluster_max_box_size);

  // AmrSystem: generic single-species composition on AMR.
  py::class_<AmrSystem> cls(m, "AmrSystem");
  bind_amr_assembly(cls);
  bind_amr_physics(cls);
  bind_amr_stepping(cls);
  bind_amr_program(cls);
  bind_amr_data(cls);
}
