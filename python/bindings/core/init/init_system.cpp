#include "../bindings_detail.hpp"
#include <pops/parallel/world_communicator.hpp>
#include "boundary_component_install.hpp"
#include "output_geometry_binding.hpp"

#include <pops/runtime/dynamic/component_loader.hpp>

#include <initializer_list>
#include <limits>

// ADC-365: the System runtime-composition facade bindings.
//
// ADC-593: these .def registrations are INTERNAL seams of the bind flow (pops.bind reaches them through
// compile / bind, not as public vocabulary). To keep this adapter TU readable as it grew, the chain is
// split into concern-grouped static helpers (assembly / program+lifecycle / checkpoint / physics /
// stepping / data+io), each taking the class handle and adding its slice. This is a PURE reorganization:
// the SAME .def names, docstrings, args, and RELATIVE registration order (no overload set is reordered --
// every System method name here is unique). The class name and the .def names are unchanged, so the
// legacy-name architecture gate still finds them in this exact file.
namespace {

void require_exact_keys(const py::dict& value, std::initializer_list<const char*> expected,
                        const char* where) {
  if (value.size() != static_cast<py::ssize_t>(expected.size()))
    throw py::value_error(std::string(where) + " keys are not exact");
  for (const char* key : expected)
    if (!value.contains(key))
      throw py::value_error(std::string(where) + " keys are not exact");
}

SystemLayoutTransferSpec layout_transfer_spec_from_python(const py::dict& row) {
  require_exact_keys(
      row,
      {"mapping_identity", "provider_identity", "provider_component_identity",
       "provider_manifest_identity", "source_layout_identity", "target_layout_identity",
       "source_block", "target_block", "source_representation", "target_representation",
       "synchronization_identity", "refinement_ratio", "operation"},
      "prepared layout-transfer spec");
  return {py::cast<std::string>(row["mapping_identity"]),
          py::cast<std::string>(row["provider_identity"]),
          py::cast<std::string>(row["provider_component_identity"]),
          py::cast<std::string>(row["provider_manifest_identity"]),
          py::cast<std::string>(row["source_layout_identity"]),
          py::cast<std::string>(row["target_layout_identity"]),
          py::cast<std::string>(row["source_block"]),
          py::cast<std::string>(row["target_block"]),
          py::cast<std::string>(row["source_representation"]),
          py::cast<std::string>(row["target_representation"]),
          py::cast<std::string>(row["synchronization_identity"]),
          py::cast<std::array<std::int32_t, 2>>(row["refinement_ratio"]),
          py::cast<std::int32_t>(row["operation"])};
}

SystemLayoutTransferExecution layout_transfer_execution_from_python(const py::dict& row) {
  require_exact_keys(
      row,
      {"execution_identity", "context_version", "memory_space", "backend_identity",
       "device_identity", "scalar_type", "storage_precision", "compute_precision",
       "accumulation_precision", "reduction_precision", "stream_handle", "stream_identity",
       "communicator_f_handle", "communicator_datatype_f_handle", "communicator_identity",
       "communicator_datatype_identity"},
      "prepared layout-transfer execution context");
  return {py::cast<std::uint32_t>(row["context_version"]),
          py::cast<std::string>(row["execution_identity"]),
          py::cast<std::int32_t>(row["memory_space"]),
          py::cast<std::string>(row["backend_identity"]),
          py::cast<std::string>(row["device_identity"]),
          py::cast<std::int32_t>(row["scalar_type"]),
          py::cast<std::int32_t>(row["storage_precision"]),
          py::cast<std::int32_t>(row["compute_precision"]),
          py::cast<std::int32_t>(row["accumulation_precision"]),
          py::cast<std::int32_t>(row["reduction_precision"]),
          py::cast<std::uint64_t>(row["stream_handle"]),
          py::cast<std::string>(row["stream_identity"]),
          py::cast<std::int64_t>(row["communicator_f_handle"]),
          py::cast<std::int64_t>(row["communicator_datatype_f_handle"]),
          py::cast<std::string>(row["communicator_identity"]),
          py::cast<std::string>(row["communicator_datatype_identity"])};
}

PreparedProviderOptions prepared_provider_options_from_python(const std::string& schema_identity,
                                                              const py::dict& values) {
  PreparedProviderOptions options;
  options.schema_identity = schema_identity;
  for (const auto& item : values) {
    if (!py::isinstance<py::str>(item.first))
      throw py::type_error("prepared provider option names must be exact strings");
    const std::string name = py::cast<std::string>(item.first);
    const py::handle value = item.second;
    PreparedProviderOptionValue converted;
    if (py::isinstance<py::bool_>(value)) {
      converted = py::cast<bool>(value);
    } else if (py::isinstance<py::int_>(value)) {
      try {
        converted = py::cast<std::int64_t>(value);
      } catch (const py::cast_error&) {
        converted = py::cast<std::uint64_t>(value);
      }
    } else if (py::isinstance<py::float_>(value)) {
      converted = py::cast<double>(value);
    } else if (py::isinstance<py::str>(value)) {
      converted = py::cast<std::string>(value);
    } else {
      throw py::type_error(
          "prepared provider option values must be exact bool/int/float/string scalars");
    }
    options.values.emplace(name, std::move(converted));
  }
  (void)options.exact_contract();
  return options;
}

// Assembly seams: per-block composition + compiled/native/program install (the "what to assemble" API).
void bind_system_assembly(py::class_<System>& cls) {
  cls.def(py::init<const SystemConfig&>())
      // Per-block composition: model (bricks) + spatial scheme (limiter/riemann) + time
      // (explicit/imex) + substeps. Python says WHAT, the compiled C++ does the compute.
      // ADC-214: the Python SURFACE is UNCHANGED (same flat newton_* kwargs, same defaults). The
      // lambda receives them flat and BUILDS the NewtonOptions POD internally before calling the
      // new C++ method (which groups these homogeneous parameters). adc_cases sees no change.
      .def(
          "add_block",
          [](System& s, const std::string& name, const ModelSpec& model, const std::string& limiter,
             const std::string& riemann, const std::string& recon, const std::string& time,
             int substeps, bool evolve, int stride, const std::vector<std::string>& implicit_vars,
             const std::vector<std::string>& implicit_roles, int newton_max_iters,
             double newton_rel_tol, double newton_abs_tol, double newton_fd_eps,
             bool newton_diagnostics, double newton_damping, const std::string& newton_fail_policy,
             double positivity_floor, bool wave_speed_cache, double weno_epsilon) {
            NewtonOptions newton;
            newton.max_iters = newton_max_iters;
            newton.rel_tol = static_cast<Real>(newton_rel_tol);
            newton.abs_tol = static_cast<Real>(newton_abs_tol);
            newton.fd_eps = static_cast<Real>(newton_fd_eps);
            newton.damping = static_cast<Real>(newton_damping);
            newton.fail_policy =
                newton_fail_policy_from_string(newton_fail_policy, "System::add_block");
            s.add_block(name, model, limiter, riemann, recon, time, substeps, evolve, stride,
                        implicit_vars, implicit_roles, newton, newton_diagnostics, positivity_floor,
                        wave_speed_cache, weno_epsilon);
          },
          py::arg("name"), py::arg("model"), py::arg("limiter") = "minmod",
          py::arg("riemann") = "rusanov", py::arg("recon") = "conservative",
          py::arg("time") = "explicit", py::arg("substeps") = 1, py::arg("evolve") = true,
          py::arg("stride") = 1,
          // Implicit mask CARRIED BY THE BLOCK (IMEX): conserved variables treated implicitly by
          // NAME (implicit_vars) or by physical ROLE (implicit_roles). Empty (default) -> model default,
          // bit-identical. Resolved on the C++ side against the block's names/roles (error on a missing name/role).
          py::arg("implicit_vars") = std::vector<std::string>{},
          py::arg("implicit_roles") = std::vector<std::string>{},
          // Options of the implicit IMEX source Newton (defaults = historical constants 2 / 1e-7,
          // bit-identical). newton_diagnostics=True enables the report (newton_report(name)).
          py::arg("newton_max_iters") = kNewtonDefaultMaxIters,
          py::arg("newton_rel_tol") = static_cast<double>(kNewtonDefaultRelTol),
          py::arg("newton_abs_tol") = static_cast<double>(kNewtonDefaultAbsTol),
          py::arg("newton_fd_eps") = static_cast<double>(kNewtonDefaultFdEps),
          py::arg("newton_diagnostics") = false,
          py::arg("newton_damping") = static_cast<double>(kNewtonDefaultDamping),
          py::arg("newton_fail_policy") = "none",
          // Zhang-Shu POSITIVITY limiter (ADC-76): density floor of the reconstructed face states
          // (conservative scaling toward the cell mean). 0 (default) = inactive,
          // bit-identical path. Requires a model exposing the Density role.
          py::arg("positivity_floor") = 0.0,
          // HLL wave speed cache (opt-in): evaluates model.wave_speeds once per cell instead of per
          // face. riemann='hll' + explicit only (explicit error otherwise). NoSlope + conservative
          // recon -> bit-identical to the per-face path. False (default) = path unchanged.
          py::arg("wave_speed_cache") = false,
          // ADC-645: the WENO-Z smoothness regulariser of limiter='weno5' (default = the historical
          // kWenoEpsilon literal, bit-identical; refused on another limiter / the polar path).
          py::arg("weno_epsilon") = static_cast<double>(kWenoEpsilon))
      .def(
          "_install_boundary_plan",
          [](System& system, const std::string& name, const std::string& identity,
             int required_depth, const std::vector<std::string>& face_types,
             const std::vector<double>& face_values, int ncomp,
             const std::vector<int>& omitted_interface_faces, const std::string& state_identity) {
            system.install_boundary_plan(name, identity, required_depth, face_types, face_values,
                                         ncomp, omitted_interface_faces, state_identity,
                                         PreparedBoundaryReadDependencies{});
          },
          py::arg("name"), py::arg("identity"), py::arg("required_depth"), py::arg("face_types"),
          py::arg("face_values"), py::arg("ncomp"),
          py::arg("omitted_interface_faces") = std::vector<int>{},
          py::arg("state_identity") = std::string{},
          "Install one resolved per-block ghost-production plan before block construction.")
      .def("_install_block_state_route", &System::install_block_state_route, py::arg("name"),
           py::arg("state_identity"),
           "Bind one exact state Handle identity to native block storage.")
      .def("_install_boundary_field_route", &System::install_boundary_field_route,
           py::arg("field_identity"), py::arg("provider_slot"),
           "Bind one exact boundary field Handle to native provider storage.")
      .def("_discard_boundary_plans", &System::discard_boundary_plans,
           "Roll back one failed pre-block boundary authority transaction.")
      .def(
          "_install_ghost_boundary_component",
          [](System& system, const std::string& name,
             std::shared_ptr<pops::component::LoadedComponent> component, const py::dict& row,
             const std::string& parameters_json, const std::string& target_json,
             const py::dict& execution) {
            system.install_ghost_boundary_component(
                name,
                pops::python::detail::boundary_component_spec_from_python(row, parameters_json,
                                                                          target_json, execution),
                std::move(component));
          },
          py::arg("name"), py::arg("component"), py::arg("binding"), py::arg("parameters_json"),
          py::arg("target_json"), py::arg("execution_context"))
      .def(
          "_install_field_boundary_residual_component",
          [](System& system, const std::string& name,
             std::shared_ptr<pops::component::LoadedComponent> component, const py::dict& row,
             const std::string& parameters_json, const std::string& target_json,
             const py::dict& execution) {
            system.install_field_boundary_residual_component(
                name,
                pops::python::detail::boundary_component_spec_from_python(row, parameters_json,
                                                                          target_json, execution),
                std::move(component));
          },
          py::arg("name"), py::arg("component"), py::arg("binding"), py::arg("parameters_json"),
          py::arg("target_json"), py::arg("execution_context"))
      .def(
          "_install_field_boundary_jvp_component",
          [](System& system, const std::string& name,
             std::shared_ptr<pops::component::LoadedComponent> component, const py::dict& row,
             const std::string& parameters_json, const std::string& target_json,
             const py::dict& execution) {
            system.install_field_boundary_jvp_component(
                name,
                pops::python::detail::boundary_component_spec_from_python(row, parameters_json,
                                                                          target_json, execution),
                std::move(component));
          },
          py::arg("name"), py::arg("component"), py::arg("binding"), py::arg("parameters_json"),
          py::arg("target_json"), py::arg("execution_context"))
      .def(
          "_install_interface_flux_component",
          [](System& system, std::size_t left_block, std::size_t right_block, int level,
             std::shared_ptr<pops::component::LoadedComponent> component, const py::dict& interface,
             const py::dict& binding, const std::string& parameters_json,
             const std::string& target_json, const py::dict& execution) {
            auto route = pops::python::detail::interface_route_from_python(interface, left_block,
                                                                           right_block, level);
            auto spec = pops::python::detail::interface_flux_spec_from_python(
                interface, binding, parameters_json, target_json, execution);
            system.install_interface_flux_component(std::move(route), std::move(spec),
                                                    std::move(component));
          },
          py::arg("left_block"), py::arg("right_block"), py::arg("level"), py::arg("component"),
          py::arg("interface"), py::arg("binding"), py::arg("parameters_json"),
          py::arg("target_json"), py::arg("execution_context"))
      .def("_interface_evaluation_count", &System::interface_evaluation_count, py::arg("identity"),
           py::arg("level") = 0)
      .def("_discard_interface_flux_components", &System::discard_interface_flux_components,
           "Roll back one failed post-block interface authority transaction.")
      // Newton report (IMEX diagnostics OPT-IN): dict {enabled, converged, max_residual,
      // max_iters_used, n_failed, failed_cell, failed_component}, aggregated over the substeps of the
      // LAST advance of the block. failed_cell = (i, j) of ONE faulty cell or None.
      .def(
          "newton_report",
          [](const System& s, const std::string& name) {
            const System::SourceNewtonReport r = s.newton_report(name);
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
      // ADC-510 (Spec 5 C5): changes the RUNTIME parameters of a compiled time PROGRAM block WITHOUT
      // recompiling the .so. prog_block = the PROGRAM block index (P.state order); values = that block's
      // params in sorted-name order (the .so pops_program_param_* metadata). cf.
      // System::set_program_params. Python's _install_params routes a params={name: value} dict here.
      .def("set_program_params", &System::set_program_params, py::arg("prog_block"),
           py::arg("values"))
      // Private package-install seam. The resolved parameter vector is injected before native
      // closures are built; no mutable per-block parameter side channel exists.
      .def("_install_native_block", &System::add_native_block, py::arg("name"), py::arg("so_path"),
           py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit",
           py::arg("gamma") = static_cast<double>(kPhysicalDefaultGamma), py::arg("substeps") = 1,
           py::arg("evolve") = true, py::arg("stride") = 1,
           py::arg("params") = std::vector<double>{}, py::arg("positivity_floor") = 0.0)
      .def("_install_external_riemann_block", &System::add_external_riemann_block,
           py::arg("name"), py::arg("so_path"), py::arg("brick_id"), py::arg("sha256"),
           py::arg("limiter"), py::arg("recon"), py::arg("time"), py::arg("gamma"),
           py::arg("substeps"), py::arg("evolve"), py::arg("stride"),
           py::arg("expected_nvars"), py::arg("expected_naux"),
           py::arg("expected_model_identity"),
           py::arg("positivity_floor") = 0.0,
           py::arg("weno_epsilon") = static_cast<double>(kWenoEpsilon))
      // Compiled time Program (epic ADC-399 / ADC-401): dlopen a generated problem.so, verify its
      // ABI key against this module (fail-loud -> RuntimeError), and install its macro-step body. The
      // block(s) must already exist (add_equation); the Program drives sim.step(dt) via ProgramContext.
      .def("install_program", &System::install_program, py::arg("so_path"))
      // Compiled-Program macro-step cadence (ADC-411): SYSTEM-level substeps + stride around the
      // installed program closure (cf. SystemStepper::step). Separate from install_program so the .so
      // Internal compiled-kernel cadence seam; the public controller is Program.step_strategy().
      .def("set_program_cadence", &System::set_program_cadence, py::arg("substeps"),
           py::arg("stride"));
}

// Program introspection + runtime freeze lifecycle + field-solver token.
void bind_system_program(py::class_<System>& cls) {
  cls
      // ADC-594: read the installed GLOBAL cadence (substeps / stride) for the ProgramRuntimeReport.
      // Const getters (default 1/1 with no program); there was no Python-visible getter before.
      .def("program_substeps", &System::program_substeps)
      .def("program_stride", &System::program_stride)
      // ADC-406b: IR hash of the installed compiled Program (the .so's pops_program_hash), or "" if
      // none. sim.checkpoint records it; sim.restart rejects a restart against a DIFFERENT Program.
      .def("installed_program_hash", &System::installed_program_hash)
      // ADC-592: runtime freeze lifecycle. mark_bound() (called LAST by the Python bind flow) freezes
      // the composition -> every structural setter then rejects; lifecycle_state() reports
      // assembling / bound / running (running derived from macro_step()).
      .def("mark_bound", &System::mark_bound)
      .def("lifecycle_state", &System::lifecycle_state)
      // ADC-466 (Spec criterion 24): configured field (Poisson) solver token (the last set_poisson
      // solver, default "geometric_mg"). install_program reads it to validate a field operator's
      // solver requirement; exposed so the unified sim.install can pre-validate host-side too.
      .def("poisson_solver", &System::poisson_solver)
      // ADC-414 (spec op 23): scalar diagnostics a compiled Program records via P.record_scalar,
      // retrievable AFTER sim.step. program_diagnostic(name) reads one (raises if never recorded);
      // program_diagnostics() returns the whole name -> value dict.
      .def("program_diagnostic", &System::program_diagnostic, py::arg("name"))
      .def("program_diagnostics", &System::program_diagnostics)
      // ADC-542: the native collective reduction over a named block the diagnostics driver drives to
      // fire a declared typed measure (Norm / Integral / MinMax) each cadence tick, and the sink the
      // driver records the measured scalar into (readable via program_diagnostics, same map a
      // compiled Program's P.record_scalar writes).
      .def("reduce_component", &System::reduce_component, py::arg("block"), py::arg("kind"),
           py::arg("comp") = 0)
      .def("record_program_diagnostic", &System::record_program_diagnostic, py::arg("name"),
           py::arg("value"));
}

// Checkpoint/restart seams: multistep history rings + scheduler value-cache (gathered/restored directly).
void bind_system_checkpoint(py::class_<System>& cls) {
  cls
      // Multistep history checkpoint/restart seam (ADC-406b): the facade gathers/restores the
      // System-owned rings DIRECTLY (no .so checkpoint_extra ABI). history_global mirrors state_global
      // (collective gather, component-major); restore_history mirrors set_state (owner-rank scatter).
      .def("history_names", &System::history_names)
      .def("history_depth", &System::history_depth, py::arg("name"))
      .def("history_ncomp", &System::history_ncomp, py::arg("name"))
      .def(
          "history_global",
          [](const System& s, const std::string& name, int slot) {
            return to_3d(s.history_global(name, slot), s.history_ncomp(name), s.ny(), s.nx());
          },
          py::arg("name"), py::arg("slot"))
      .def("history_initialized", &System::history_initialized, py::arg("name"))
      .def(
          "restore_history",
          [](System& s, const std::string& name, int slot,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.restore_history(name, slot, flat(arr));
          },
          py::arg("name"), py::arg("slot"), py::arg("values"))
      .def("set_history_initialized", &System::set_history_initialized, py::arg("name"),
           py::arg("initialized"))
      // Selective history persistence + deterministic ring replay (ADC-626): the checkpoint stores only
      // the policy-selected slots + the per-slot dt; the restart replays the gaps via
      // rebuild_history_slots (re-stepping the installed Program from the nearest older stored slot).
      .def("history_slot_dt", &System::history_slot_dt, py::arg("name"), py::arg("slot"))
      .def("restore_history_slot_dt", &System::restore_history_slot_dt, py::arg("name"),
           py::arg("slot"), py::arg("dt"))
      .def("rebuild_history_slots", &System::rebuild_history_slots, py::arg("name"),
           py::arg("stored_slots"))
      // Scheduler value-cache checkpoint/restart seam (ADC-458, Spec 3 section 30): the facade gathers/
      // restores the System-owned held-node cache DIRECTLY (no .so checkpoint_extra ABI), mirroring the
      // history seam. program_cache_global mirrors history_global (collective gather, component-major);
      // restore_program_cache mirrors restore_history (owner-rank scatter + re-key). program_cache_nodes
      // is empty unless a held schedule cached a value, so a program without one writes no cache keys.
      .def("program_cache_nodes", &System::program_cache_nodes)
      .def("program_cache_name", &System::program_cache_name, py::arg("node_id"))
      .def("program_cache_last_update_step", &System::program_cache_last_update_step,
           py::arg("node_id"))
      .def("program_cache_accumulated_dt", &System::program_cache_accumulated_dt,
           py::arg("node_id"))
      .def("program_cache_ncomp", &System::program_cache_ncomp, py::arg("node_id"))
      .def("program_cache_ngrow", &System::program_cache_ngrow, py::arg("node_id"))
      .def(
          "program_cache_global",
          [](const System& s, int node_id) {
            return to_3d(s.program_cache_global(node_id), s.program_cache_ncomp(node_id), s.ny(),
                         s.nx());
          },
          py::arg("node_id"))
      .def(
          "restore_program_cache",
          [](System& s, int node_id, int ncomp, int ngrow, int last_update_step,
             double accumulated_dt, const std::string& name,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.restore_program_cache(node_id, ncomp, ngrow, last_update_step, accumulated_dt, name,
                                    flat(arr));
          },
          py::arg("node_id"), py::arg("ncomp"), py::arg("ngrow"), py::arg("last_update_step"),
          py::arg("accumulated_dt"), py::arg("name"), py::arg("values"));
}

// Physics wiring: inter-species couplings, Poisson/field config, geometry (disc),
// epsilon/reaction/magnetic/aux fields, and state initialization.
void bind_system_physics(py::class_<System>& cls) {
  // The named inter-species couplings (add_ionization / add_collision / add_thermal_exchange) are no
  // longer bound (ADC-595): they are Python presets lowering to add_coupling_operator. A new coupling
  // needs no new pybind def.
  cls
      // (System) -- see also AmrSystem.add_coupled_source below for the AMR counterpart.
      // GLOBAL time-step bound (step_cfl audit): fn() evaluated ONCE per step (host) by
      // step_cfl / step_adaptive; dt <= fn() when fn() > 0 and finite. Hook for non
      // cell-local constraints (coupling, Schur/Poisson, scheduler, user ramp). A Python
      // callback is acceptable here (never per cell).
      .def("add_dt_bound", &System::add_dt_bound, py::arg("label"), py::arg("fn"))
      // ACTIVE bound of the last step_cfl: "transport:<block>" | "source_frequency:<block>" |
      // "stability_dt:<block>" | "global:<label>" | "degenerate" | "" (no CFL step yet).
      .def("last_dt_bound", &System::last_dt_bound)
      // Clock (IO v1): macro_step exposed + restoration (t, macro_step) for the restart -- the
      // stride cadence depends on macro_step % stride, not only on t.
      .def("macro_step", &System::macro_step)
      .def("set_clock", &System::set_clock, py::arg("t"), py::arg("macro_step"))
      .def("set_potential", &System::set_potential, py::arg("phi"))
      .def("field_provider_slots", &System::field_provider_slots)
      .def("set_field_potential", &System::set_field_potential, py::arg("provider_slot"),
           py::arg("phi"))
      // INTERNAL raw coupled-source ABI (ADC-595): the flat 12-kwarg bytecode form is now an INTERNAL
      // escape hatch (leading underscore), called only by the typed lowering (add_coupling ->
      // add_coupling_operator) and by the low-level ABI-validation tests. End users register a coupling
      // through sim.add_coupling(CoupledSource(...).compile()) or a named preset, never this raw form.
      // The lambda assembles the CoupledSourceProgram POD before the C++ call.
      .def(
          "_add_coupled_source",
          [](System& s, const std::vector<std::string>& in_blocks,
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
          // CONSTANT declared frequency mu of the coupling (CoupledSource.frequency, wave 3): step
          // bound dt <= cfl/mu on the macro-step; <= 0 = no bound (historical).
          py::arg("frequency") = 0.0, py::arg("label") = "coupled_source",
          // Optional PER-CELL frequency mu(U): bytecode program (same stack machine / register
          // table as the terms). EMPTY (default) = constant frequency only, bit-identical.
          py::arg("freq_prog_ops") = std::vector<int>{},
          py::arg("freq_prog_args") = std::vector<int>{})
      // Typed COUPLING OPERATOR (ADC-595): the same flat coupled-source program PLUS the DECLARED
      // conservation contract (conserved / created roles) and frequency bound. The declared contract is
      // validated at registration (host, fail-loud) against the actual terms, then the program lowers
      // through the SAME add_coupled_source path (bit-identical). Used by the typed named-coupling
      // presets; the raw add_coupled_source above stays the unchecked (empty-contract) entry.
      .def(
          "add_coupling_operator",
          [](System& s, const std::vector<std::string>& in_blocks,
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
      // {label, conserved_roles, created_roles, frequency_mu, per_cell_frequency}, in registration
      // order, so a Program / report enumerates couplings as typed operators (never raw bytecode).
      .def("coupled_operators",
           [](const System& s) {
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
           })
      .def("variable_names", &System::variable_names,
           "Variable names of a block (introspection). kind = 'conservative' | 'primitive'.",
           py::arg("name"), py::arg("kind") = "conservative")
      .def("variable_roles", &System::variable_roles,
           "PHYSICAL roles of a block's variables, parallel to variable_names: 'density', "
           "'momentum_x', 'energy', ... or 'custom' if the block does not declare its roles. This "
           "is what "
           "the inter-species couplings resolve (index_of(role)). kind = 'conservative' | "
           "'primitive'.",
           py::arg("name"), py::arg("kind") = "conservative")
      .def("block_gamma", &System::block_gamma, py::arg("name"))
      .def(
          "set_poisson", &System::set_poisson,
          "Configures the shared system Poisson. rhs: 'charge_density' | 'composite' (labels "
          "of the SAME right-hand side f = sum of the elliptic bricks per block; charge_density = "
          "historical "
          "alias). solver: 'geometric_mg' (any case, wall included) | 'fft' (periodic, "
          "discrete stencil; n = 2^k for the fast FFT, otherwise direct DFT O(n^2)) | "
          "'fft_spectral' "
          "(periodic, continuous symbol -(kx^2+ky^2)). bc: 'auto' | 'periodic' | 'dirichlet' | "
          "'neumann'. wall: 'none' | "
          "'circle' (conducting wall centered at (L/2, L/2), radius wall_radius). epsilon: "
          "CONSTANT permittivity of div(eps grad phi) = f (for variable eps(x): "
          "set_epsilon_field). "
          "abs_tol: absolute floor of the GeometricMG V-cycle stopping criterion (0 = relative "
          "criterion, "
          "historical; no effect on FFT). rel_tol / max_cycles / min_coarse / pre_smooth / "
          "post_smooth / bottom_sweeps: the GeometricMG V-cycle knobs (ADC-613); they default to "
          "the native kMG* constants so an omitting call is bit-identical to the historical "
          "V-cycle, and are inert for the FFT solver. coarse_threshold (ADC-644): a total-cell "
          "coarsening ceiling -- coarsening stops once a level's nx*ny is at or below it; distinct "
          "from the per-axis min_coarse. Default 0 = disabled (only min_coarse governs), "
          "bit-identical to the historical hierarchy; inert for the FFT solver.",
          py::arg("rhs") = "charge_density", py::arg("solver") = "geometric_mg",
          py::arg("bc") = "auto", py::arg("wall") = "none", py::arg("wall_radius") = 0.0,
          py::arg("epsilon") = 1.0, py::arg("abs_tol") = 0.0,
          py::arg("rel_tol") = static_cast<double>(kMGDefaultRelTol),
          py::arg("max_cycles") = kMGDefaultMaxCycles, py::arg("min_coarse") = kMGDefaultMinCoarse,
          py::arg("pre_smooth") = kMGDefaultPreSmooth,
          py::arg("post_smooth") = kMGDefaultPostSmooth,
          py::arg("bottom_sweeps") = kMGDefaultBottomSweeps,
          py::arg("coarse_threshold") = kMGDefaultCoarseThreshold)
      .def(
          "register_configured_field_solver_provider",
          [](System& system, const std::string& family_route, const std::string& provider_route,
             const std::string& schema_identity, const py::dict& options) {
            return system.register_configured_field_solver_provider(
                family_route, provider_route,
                prepared_provider_options_from_python(schema_identity, options));
          },
          py::arg("family_route"), py::arg("provider_route"), py::arg("schema_identity"),
          py::arg("options"))
      .def("set_field_solver_plan", &System::set_field_solver_plan, py::arg("provider_slot"),
           py::arg("plan_identity"), py::arg("provider_identity"), py::arg("output_owner_identity"),
           py::arg("output_block"), py::arg("output_key"), py::arg("provider_identities"),
           py::arg("provider_blocks"), py::arg("provider_keys"), py::arg("provider_coefficients"),
           py::arg("backend_provider_route"))
      .def("set_field_reaction", &System::set_field_reaction, py::arg("provider_slot"),
           py::arg("reaction"))
      .def(
          "register_field_solver_provider",
          [](System& system, const std::string& provider_slot,
             std::shared_ptr<pops::component::LoadedComponent> topology,
             std::shared_ptr<pops::component::LoadedComponent> solver,
             const py::dict& topology_binding, const py::dict& solver_binding,
             const std::string& topology_parameters_json, const std::string& solver_parameters_json,
             const std::string& source_layout_identity, const std::string& topology_recipe_identity,
             const std::string& boundary_contract_json, double relative_tolerance,
             double absolute_tolerance, std::int32_t max_iterations, const py::dict& execution) {
            auto spec = pops::python::detail::field_solver_spec_from_python(
                provider_slot, topology_binding, solver_binding, topology_parameters_json,
                solver_parameters_json, source_layout_identity, topology_recipe_identity,
                boundary_contract_json, relative_tolerance, absolute_tolerance, max_iterations,
                execution);
            return system.register_field_solver_provider(provider_slot, std::move(spec),
                                                         std::move(topology), std::move(solver));
          },
          py::arg("provider_slot"), py::arg("topology_component"), py::arg("solver_component"),
          py::arg("topology_binding"), py::arg("solver_binding"),
          py::arg("topology_parameters_json"), py::arg("solver_parameters_json"),
          py::arg("source_layout_identity"), py::arg("topology_recipe_identity"),
          py::arg("boundary_contract_json"), py::arg("relative_tolerance"),
          py::arg("absolute_tolerance"), py::arg("max_iterations"), py::arg("execution_context"))
      .def("_set_field_topology_authority", &System::set_field_topology_authority,
           py::arg("provider_slot"), py::arg("provider_kind"), py::arg("provenance"),
           py::arg("topology_digest"))
      .def(
          "_field_topology_report",
          [](const System& system, const std::string& provider_slot) {
            py::list report;
            for (const auto& row : system.field_topology_report(provider_slot)) {
              py::dict item;
              item["patch_identity"] = row.patch_identity;
              item["topology_digest"] = row.topology_digest;
              item["provenance"] = row.provenance;
              item["material_points"] = row.material_points;
              item["connected_components"] = row.connected_components;
              item["source_layout_identity"] = row.source_layout_identity;
              item["materialized_layout_identity"] = row.materialized_layout_identity;
              report.append(std::move(item));
            }
            return report;
          },
          py::arg("provider_slot"))
      .def("register_elliptic_field", &System::register_elliptic_field, py::arg("block"),
           py::arg("field"), py::arg("phi_comp"), py::arg("gx_comp"), py::arg("gy_comp"),
           py::arg("gradient_sign"))
      .def("set_field_boundary_plan", &System::set_field_boundary_plan, py::arg("provider_slot"),
           py::arg("kind"), py::arg("alpha"), py::arg("beta"), py::arg("value"))
      .def("set_field_boundary_dependencies", &System::set_field_boundary_dependencies,
           py::arg("provider_slot"), py::arg("state_blocks"), py::arg("state_components"),
           py::arg("field_blocks"), py::arg("field_keys"), py::arg("field_components"))
      .def("set_field_boundary_parameters", &System::set_field_boundary_parameters,
           py::arg("provider_slot"), py::arg("parameters"))
      .def(
          "set_default_field_nullspace",
          [](System& system, const std::string& provider_identity,
             const std::string& schema_identity, const py::dict& options) {
            system.set_default_field_nullspace(
                provider_identity, prepared_provider_options_from_python(schema_identity, options));
          },
          py::arg("provider_identity"), py::arg("schema_identity"), py::arg("options"))
      .def(
          "set_field_nullspace",
          [](System& system, const std::string& provider_slot, const std::string& provider_identity,
             const std::string& schema_identity, const py::dict& options) {
            system.set_field_nullspace(
                provider_slot, provider_identity,
                prepared_provider_options_from_python(schema_identity, options));
          },
          py::arg("provider_slot"), py::arg("provider_identity"), py::arg("schema_identity"),
          py::arg("options"))
      .def("set_field_newton_plan", &System::set_field_newton_plan, py::arg("provider_slot"),
           py::arg("tolerance"), py::arg("max_iterations"), py::arg("linear_tolerance"),
           py::arg("linear_max_iterations"), py::arg("restart"), py::arg("armijo"),
           py::arg("minimum_step"))
      // Runtime-private lowering seam for every public analytic LevelSet.  The native System owns,
      // validates and materializes the scalar postfix program; no Python callback reaches a cell
      // kernel.  Active is the strict convention phi < 0.
      .def("_set_analytic_level_set", &System::set_analytic_level_set, py::arg("opcodes"),
           py::arg("literals"), py::arg("mode") = "none", py::arg("kappa_min") = 0.0,
           py::arg("face_open_eps") = 0.0, py::arg("cut_theta_min") = 0.0)
      // Disc convenience constructor: lowers to the same generic analytic program and EB path.
      .def("set_disc_domain", &System::set_disc_domain, py::arg("cx"), py::arg("cy"), py::arg("R"),
           py::arg("mode") = "none", py::arg("kappa_min") = 0.0, py::arg("face_open_eps") = 0.0,
           py::arg("cut_theta_min") = 0.0)
      // Toggles only the installed level-set transport mode without redefining its expression.
      .def("set_geometry_mode", &System::set_geometry_mode, py::arg("mode"))
      // Domain 0/1 mask (ny, nx) row-major. Historical name retained for compatibility; it reports
      // the mask of any analytic level set and is all 1.0 when none is installed.
      .def("disc_mask", [](const System& s) { return to_2d(s.disc_mask(), s.ny(), s.nx()); })
      .def(
          "set_epsilon_field",
          [](System& s, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_epsilon_field(flat(arr));
          },
          py::arg("eps"))
      .def(
          "set_epsilon_anisotropic_field",
          [](System& s, py::array_t<double, py::array::c_style | py::array::forcecast> eps_x,
             py::array_t<double, py::array::c_style | py::array::forcecast> eps_y) {
            s.set_epsilon_anisotropic_field(flat(eps_x), flat(eps_y));
          },
          py::arg("eps_x"), py::arg("eps_y"))
      .def(
          "set_reaction_field",
          [](System& s, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_reaction_field(flat(arr));
          },
          py::arg("kappa"))
      .def(
          "set_magnetic_field",
          [](System& s, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_magnetic_field(flat(arr));
          },
          py::arg("bz"))
      // NAMED aux fields (ADC-70 phase 1): by canonical COMPONENT (>= 5). The name -> comp
      // resolution lives in the private Python System facade, which calls these two methods.
      .def(
          "set_aux_field_component",
          [](System& s, int comp,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_aux_field_component(comp, flat(arr));
          },
          py::arg("comp"), py::arg("field"))
      // ADC-369: per-field aux halo policy (bc_type = pops::BCType Foextrap=1 / Dirichlet=2). The Python
      // facade (System.set_aux_field(..., halo=pops.mesh.AuxHalo(...))) resolves name -> comp and calls this.
      .def(
          "set_aux_field_halo_component",
          [](System& s, int comp, int bc_type, double value) {
            s.set_aux_field_halo_component(comp, bc_type, value);
          },
          py::arg("comp"), py::arg("bc_type"), py::arg("value"))
      .def(
          "aux_field_component",
          [](const System& s, int comp) {
            return to_2d(s.aux_field_component(comp), s.ny(), s.nx());
          },
          py::arg("comp"))
      .def("set_electron_temperature_from", &System::set_electron_temperature_from, py::arg("name"))
      .def(
          "set_density",
          [](System& s, const std::string& name,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_density(name, flat(arr));
          },
          py::arg("name"), py::arg("rho"))
      // Init from the PRIMITIVES: prim = array (ncomp, n, n) component-major in the order of
      // primitive_vars(name); converted to conservative by the block's model. The Python facade
      // System.set_primitive_state(**prims) assembles this array from the named kwargs.
      .def(
          "set_primitive_state",
          [](System& s, const std::string& name,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_primitive_state(name, flat(arr));
          },
          py::arg("name"), py::arg("prim"))
      // Diagnostic: conservative state -> primitive (ncomp, n, n), order of primitive_vars(name).
      .def(
          "get_primitive_state",
          [](System& s, const std::string& name) {
            return to_3d(s.get_primitive_state(name), s.n_vars(name), s.ny(), s.nx());
          },
          py::arg("name"));
}

// Stepping + profiling + custom-integrator primitives (field solve, step/advance/CFL, eval_rhs/state).
void bind_system_stepping(py::class_<System>& cls) {
  cls.def("solve_fields", &System::solve_fields)
      .def("step", &System::step, py::arg("dt"))
      .def("advance", &System::advance, py::arg("dt"), py::arg("nsteps"))
      .def("_begin_step_transaction", &System::begin_step_transaction)
      .def("_commit_step_transaction", &System::commit_step_transaction)
      .def("_finalize_step_transaction", &System::finalize_step_transaction)
      .def("_rollback_step_transaction", &System::rollback_step_transaction)
      .def(
          "_prepare_layout_transfer",
          [](System& source, System& target,
             std::shared_ptr<pops::component::LoadedComponent> component,
             const py::dict& spec, const py::dict& execution) {
            return PreparedSystemLayoutTransfer::prepare(
                source, target, std::move(component),
                layout_transfer_spec_from_python(spec),
                layout_transfer_execution_from_python(execution));
          },
          py::arg("target"), py::arg("component"), py::arg("spec"),
          py::arg("execution_context"), py::keep_alive<0, 1>(), py::keep_alive<0, 2>(),
          "Prepare one persistent native System-to-System mapping session.")
      .def("step_cfl", &System::step_cfl,
           "Advances by ONE step at dt = cfl * h / max wave speed of the system (also honors the "
           "optional bounds: substeps, stride, source_frequency, couplings, add_dt_bound). Returns "
           "the dt used. speed_floor (ADC-645): the floor applied to the reduced max wave speed "
           "(w = max(w, speed_floor), so a quiescent system cannot divide by zero); defaults to "
           "the historical kCflSpeedFloor (1e-30), bit-identical.",
           py::arg("cfl"), py::arg("speed_floor") = static_cast<double>(kCflSpeedFloor),
           py::arg("max_dt") = std::numeric_limits<double>::infinity(), py::arg("min_dt") = 0.0)
      .def("enable_profiling", &System::enable_profiling,
           "Spec 3 profiling (ADC-459): start timing the step phases (step, field_solve). Disabled "
           "by default; off the hot path when off.")
      .def("disable_profiling", &System::disable_profiling,
           "Stop profiling (keeps accumulated data).")
      .def("is_profiling", &System::is_profiling)
      .def("reset_profiling", &System::reset_profiling, "Clear accumulated profiling data.")
      .def("profile_report", &System::profile_report,
           "Per-phase / per-brick wall-clock report (count / total / mean / min / max per scope, "
           "plus counters). Per-rank.")
      .def(
          "profile_snapshot",
          [](System& s) { return profile_snapshot_to_dict(s.profiler().snapshot()); },
          "Structured profiling snapshot: schema_version, enabled, scopes and counters.")
      .def(
          "solver_diagnostics",
          [](const System& s) {
            py::list rows;
            for (const RuntimeDiagnosticEvent& event : s.solver_diagnostics()) {
              py::dict row;
              row["code"] = event.code;
              row["component"] = event.component;
              row["severity"] = event.severity;
              row["message"] = event.message;
              row["iteration"] = event.iteration;
              row["value"] = event.value;
              rows.append(row);
            }
            return rows;
          },
          "Structured solver/runtime diagnostic events; empty unless diagnostics were enabled.")
      .def("dt_hotspot", &System::dt_hotspot,
           "Diagnostic (ADC-182): (w, i, j) of the GLOBAL cell that dominates the transport CFL "
           "bound "
           "of block 'name' -- to locate a collapsing dt. On demand, off the hot path.",
           py::arg("name"))
      .def("step_adaptive", &System::step_adaptive,
           "Advances by ONE MULTIRATE macro-step: the slowest block sets the macro-step, each "
           "faster "
           "block is sub-cycled n = ceil(w_block / w_min) times. Returns the macro-step.",
           py::arg("cfl"))
      // Explicit host inspection/state-transfer primitives.  Production time programs execute in
      // the prepared native runtime; these bulk copies exist for initialization, checkpoints,
      // diagnostics and numerical verification, never as a per-cell Python stepping route.
      .def(
          "eval_rhs",
          [](System& s, const std::string& name) {
            return to_3d(s.eval_rhs(name), s.n_vars(name), s.ny(), s.nx());
          },
          py::arg("name"))
      .def(
          "get_state",
          [](System& s, const std::string& name) {
            return to_3d(s.get_state(name), s.n_vars(name), s.ny(), s.nx());
          },
          py::arg("name"))
      .def(
          "set_state",
          [](System& s, const std::string& name,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_state(name, flat(arr));
          },
          py::arg("name"), py::arg("u"))
      .def("_set_analytic_expression_state", &System::set_analytic_expression_state,
           py::arg("name"), py::arg("space"), py::arg("centering"), py::arg("projection"),
           py::arg("opcodes"), py::arg("literals"))
      .def("_set_analytic_mapped_state", &System::set_analytic_mapped_state,
           py::arg("name"), py::arg("opcodes"), py::arg("literals"),
           py::arg("input_sources"))
      .def("_set_analytic_gaussian_state", &System::set_analytic_gaussian_state,
           py::arg("name"), py::arg("center_x"), py::arg("center_y"),
           py::arg("background"), py::arg("amplitude"), py::arg("inverse_width"));
}

// Data + IO accessors: shape/introspection, mass/density/potential, MPI-safe globals, local hyperslabs.
void bind_system_data(py::class_<System>& cls) {
  cls.def("n_vars", &System::n_vars, py::arg("name"))
      .def("nx", &System::nx)
      .def("ny", &System::ny)
      .def("time", &System::time)
      .def("n_species", &System::n_species)
      .def("block_names", &System::block_names)
      .def(
          "effective_options_report",
          [](const System& s) {
            return effective_options_report_to_dict(s.effective_options_report());
          },
          "Structured effective numerical/solver/physical options for this System.")
      .def("mass", &System::mass, py::arg("name"))
      .def(
          "density",
          [](const System& s, const std::string& name) {
            return to_2d(s.density(name), s.ny(), s.nx());
          },
          py::arg("name"))
      .def("potential", [](System& s) { return to_2d(s.potential(), s.ny(), s.nx()); })
      // GLOBAL accessors (MPI-safe collectives): accepted-state checkpoint capture. Each
      // rank MUST call them (internal all_reduce); they return the COMPLETE field (rank-0 gather
      // implicit via all_reduce_sum) -- single-rank: bit-identical to density / get_state / potential.
      // RuntimeInstance seals and publishes the resulting checkpoint only on rank 0.
      .def(
          "density_global",
          [](const System& s, const std::string& name) {
            return to_2d(s.density_global(name), s.ny(), s.nx());
          },
          py::arg("name"))
      .def(
          "state_global",
          [](const System& s, const std::string& name) {
            return to_3d(s.state_global(name), s.n_vars(name), s.ny(), s.nx());
          },
          py::arg("name"))
      .def("potential_global",
           [](System& s) { return to_2d(s.potential_global(), s.ny(), s.nx()); })
      .def(
          "field_potential_global",
          [](System& s, const std::string& slot) {
            return to_2d(s.field_potential_global(slot), s.ny(), s.nx());
          },
          py::arg("provider_slot"))
      .def(
          "output_state_local_pieces",
          [](const System& s, const std::string& block, int level) {
            return output_pieces_to_python(s.output_state_local_pieces(block, level));
          },
          py::arg("block"), py::arg("level"),
          "Exact compact valid-cell state pieces owned by this rank.")
      .def(
          "output_field_local_pieces",
          [](System& s, const std::string& provider_slot, int level) {
            return output_pieces_to_python(s.output_field_local_pieces(provider_slot, level));
          },
          py::arg("provider_slot"), py::arg("level"),
          "Exact compact valid-cell field pieces owned by this rank.")
      .def(
          "output_state_root_pieces",
          [](const System& s, const WorldCommunicator& world, const std::string& block, int level) {
            std::vector<OutputPiece> pieces;
            {
              py::gil_scoped_release release;
              pieces = s.output_state_root_pieces(world, block, level);
            }
            return output_pieces_to_python(pieces);
          },
          py::arg("world"), py::arg("block"), py::arg("level"),
          "Collectively gather compact state pieces in C++; complete only on MPI rank zero.")
      .def(
          "output_field_root_pieces",
          [](System& s, const WorldCommunicator& world, const std::string& provider_slot,
             int level) {
            std::vector<OutputPiece> pieces;
            {
              py::gil_scoped_release release;
              pieces = s.output_field_root_pieces(world, provider_slot, level);
            }
            return output_pieces_to_python(pieces);
          },
          py::arg("world"), py::arg("provider_slot"), py::arg("level"),
          "Collectively gather compact field pieces in C++; complete only on MPI rank zero.")
      .def(
          "_output_geometry_snapshot",
          [](const System& s, const std::array<double, 2>& origin,
             const std::array<double, 2>& spacing, const std::array<std::int64_t, 2>& cell_shape,
             const std::string& cell_measure) {
            if (cell_shape[0] != s.ny() || cell_shape[1] != s.nx())
              throw std::invalid_argument(
                  "System output geometry shape differs from the native domain");
            return pops::python::detail::native_output_geometry_snapshot(
                0, 0, origin, spacing, cell_shape, cell_measure, {}, 0, false);
          },
          py::arg("origin"), py::arg("spacing"), py::arg("cell_shape"), py::arg("cell_measure"),
          "Private Writer geometry view: native, immutable, and cacheable by the runtime.")
      // LOCAL per-fab accessors (NOT collective): native ownership inspection. ScientificOutput
      // consumes the typed output_*_local_pieces API above; local_boxes returns the list of boxes
      // (ilo, jlo, ihi, jhi) in GLOBAL indices; local_state returns the state of fab li reshaped
      // (n_vars, bny, bnx) for a hyperslab dset[:, jlo:jhi+1, ilo:ihi+1]. A rank without a box returns an
      // empty list. Since the System is single-box, real parallelism only appears on a multi-box
      // geometry (cf. AMR); the API stays correct in the general case.
      .def("local_boxes", &System::local_boxes, py::arg("name"))
      .def(
          "local_state",
          [](const System& s, const std::string& name, int li) {
            const auto boxes = s.local_boxes(name);
            if (li < 0 || li >= static_cast<int>(boxes.size()))
              throw std::out_of_range("System.local_state: local fab index out of bounds");
            const int bnx = boxes[li][2] - boxes[li][0] + 1;  // ihi - ilo + 1
            const int bny = boxes[li][3] - boxes[li][1] + 1;  // jhi - jlo + 1
            return to_3d(s.local_state(name, li), s.n_vars(name), bny, bnx);
          },
          py::arg("name"), py::arg("li"))
      .def_static("abi_key", &System::abi_key,
                  "Module ABI key (cf. pops.abi_key); compared to that of a native loader.");
}

}  // namespace

// Registers the System facade class, then adds each concern's bindings IN ORDER (assembly first, so the
// class exists before the other groups extend it). The per-concern order matches the historical single
// chain; no overload set spans two concerns.
void init_system(py::module_& m) {
  py::class_<SystemLayoutTransferReceipt>(m, "_SystemLayoutTransferReceipt")
      .def_readonly("applied", &SystemLayoutTransferReceipt::applied)
      .def_readonly("mapping_identity", &SystemLayoutTransferReceipt::mapping_identity)
      .def_readonly("provider_identity", &SystemLayoutTransferReceipt::provider_identity)
      .def_readonly("provider_component_identity",
                    &SystemLayoutTransferReceipt::provider_component_identity)
      .def_readonly("provider_manifest_identity",
                    &SystemLayoutTransferReceipt::provider_manifest_identity)
      .def_readonly("source_layout_identity",
                    &SystemLayoutTransferReceipt::source_layout_identity)
      .def_readonly("target_layout_identity",
                    &SystemLayoutTransferReceipt::target_layout_identity)
      .def_readonly("source_block", &SystemLayoutTransferReceipt::source_block)
      .def_readonly("target_block", &SystemLayoutTransferReceipt::target_block)
      .def_readonly("execution_identity", &SystemLayoutTransferReceipt::execution_identity)
      .def_readonly("operation", &SystemLayoutTransferReceipt::operation)
      .def_readonly("generation", &SystemLayoutTransferReceipt::generation)
      .def_readonly("attempt", &SystemLayoutTransferReceipt::attempt)
      .def_readonly("source_element_count",
                    &SystemLayoutTransferReceipt::source_element_count)
      .def_readonly("destination_element_count",
                    &SystemLayoutTransferReceipt::destination_element_count);
  py::class_<PreparedSystemLayoutTransfer,
             std::shared_ptr<PreparedSystemLayoutTransfer>>(
      m, "_PreparedSystemLayoutTransfer")
      .def("begin_transaction", &PreparedSystemLayoutTransfer::begin_transaction,
           py::arg("generation"))
      .def("capture", &PreparedSystemLayoutTransfer::capture, py::arg("generation"),
           py::arg("attempt"))
      .def("apply", &PreparedSystemLayoutTransfer::apply, py::arg("generation"),
           py::arg("attempt"))
      .def("reject_attempt", &PreparedSystemLayoutTransfer::reject_attempt,
           py::arg("generation"), py::arg("attempt"))
      .def("finalize_transaction", &PreparedSystemLayoutTransfer::finalize_transaction,
           py::arg("generation"))
      .def("rollback_transaction", &PreparedSystemLayoutTransfer::rollback_transaction,
           py::arg("generation"));
  py::class_<System> cls(m, "System");
  bind_system_assembly(cls);
  bind_system_program(cls);
  bind_system_checkpoint(cls);
  bind_system_physics(cls);
  bind_system_stepping(cls);
  bind_system_data(cls);
}
