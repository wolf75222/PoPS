#include "../bindings_detail.hpp"
#include <pops/parallel/world_communicator.hpp>
#include "boundary_component_install.hpp"
#include "output_geometry_binding.hpp"

#include <pops/runtime/amr/prepared_component_providers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>

#include <limits>
#include <string_view>

// ADC-365: the AMR (AmrSystemConfig + AmrSystem) bindings.
//
// ADC-593: like init_system, the AmrSystem .def registrations are INTERNAL seams of the bind flow (the
// AMR target reaches them through the typed layout, not as public vocabulary). The AmrSystem chain is
// split into concern-grouped static helpers (assembly / physics / stepping / program / data), each taking
// the class handle and adding its slice. PURE reorganization: same .def names, docstrings, args, and
// RELATIVE order (no overload set is reordered -- every AmrSystem method name here is unique). The class
// name and the .def names are unchanged, so the legacy-name architecture gate still finds them here.
namespace {

pops::PreparedProviderOptionValue prepared_provider_option_from_python(const py::handle& value,
                                                                       std::string_view key) {
  if (PyBool_Check(value.ptr()))
    return value.ptr() == Py_True;
  if (PyLong_CheckExact(value.ptr())) {
    int overflow = 0;
    const long long signed_value = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
    if (PyErr_Occurred())
      throw py::error_already_set();
    if (overflow == 0)
      return static_cast<std::int64_t>(signed_value);
    if (overflow < 0)
      throw std::overflow_error("AMR provider option '" + std::string(key) +
                                "' is below int64 range");
    const unsigned long long unsigned_value = PyLong_AsUnsignedLongLong(value.ptr());
    if (PyErr_Occurred())
      throw py::error_already_set();
    return static_cast<std::uint64_t>(unsigned_value);
  }
  if (PyFloat_CheckExact(value.ptr()))
    return PyFloat_AS_DOUBLE(value.ptr());
  if (PyUnicode_CheckExact(value.ptr()))
    return py::cast<std::string>(value);
  throw py::type_error("AMR provider option '" + std::string(key) +
                       "' must be exactly bool, int64/uint64, float64 or str");
}

pops::PreparedProviderOptions prepared_provider_options_from_python(
    const std::string& schema_identity, const py::dict& values) {
  if (!PyDict_CheckExact(values.ptr()))
    throw py::type_error("AMR provider options must be an exact dict");
  pops::PreparedProviderOptions options;
  options.schema_identity = schema_identity;
  for (const auto pair : values) {
    if (!PyUnicode_CheckExact(pair.first.ptr()))
      throw py::type_error("AMR provider option keys must be exact strings");
    const std::string key = py::cast<std::string>(pair.first);
    if (key.empty())
      throw py::value_error("AMR provider option keys must be non-empty");
    options.values.emplace(key, prepared_provider_option_from_python(pair.second, key));
  }
  (void)options.exact_contract();
  return options;
}

py::dict prepared_provider_options_to_python(const pops::PreparedProviderOptions& options) {
  py::dict result;
  for (const auto& [key, value] : options.values) {
    std::visit([&](const auto& typed) { result[py::str(key)] = py::cast(typed); }, value);
  }
  return result;
}

void require_amr_cell_array_shape(const AmrSystem& system, const py::array& array,
                                  std::string_view operation) {
  const auto expected_ny = static_cast<py::ssize_t>(system.ny());
  const auto expected_nx = static_cast<py::ssize_t>(system.nx());
  if (array.ndim() != 2)
    throw py::value_error(std::string(operation) +
                          ": expected one 2D Cartesian cell array of shape (ny, nx) = (" +
                          std::to_string(expected_ny) + ", " + std::to_string(expected_nx) +
                          "); got ndim=" + std::to_string(array.ndim()));
  if (array.shape(0) != expected_ny || array.shape(1) != expected_nx)
    throw py::value_error(std::string(operation) +
                          ": expected Cartesian cell shape (ny, nx) = (" +
                          std::to_string(expected_ny) + ", " + std::to_string(expected_nx) +
                          "); got (" + std::to_string(array.shape(0)) + ", " +
                          std::to_string(array.shape(1)) + ")");
}

pops::runtime::amr::PreparedTaggerSpec amr_tagger_spec_from_python(const py::dict& row,
                                                                   const py::dict& execution) {
  pops::runtime::amr::PreparedTaggerSpec spec;
  spec.provider_identity = py::cast<std::string>(row["provider_identity"]);
  spec.component_id = py::cast<std::string>(row["component_id"]);
  spec.manifest_identity = py::cast<std::string>(row["component_manifest_identity"]);
  spec.layout_identity = py::cast<std::string>(row["layout_identity"]);
  spec.clock_identity = py::cast<std::string>(row["clock_identity"]);
  const py::dict capability = py::cast<py::dict>(row["tagging_capability"]);
  spec.leaf_opcodes = py::cast<std::vector<std::int32_t>>(capability["leaf_opcode_ids"]);
  spec.logical_opcodes = py::cast<std::vector<std::int32_t>>(capability["logical_opcode_ids"]);
  spec.indicator_stencil_routes =
      py::cast<std::vector<std::string>>(capability["indicator_stencil_routes"]);
  spec.maximum_stencil_terms = py::cast<std::size_t>(capability["maximum_stencil_terms"]);
  spec.maximum_instruction_count = py::cast<std::size_t>(capability["maximum_instruction_count"]);
  const std::string execution_mode = py::cast<std::string>(capability["execution_mode"]);
  if (execution_mode == "native_backend")
    spec.execution_mode = POPS_TAGGER_EXECUTION_NATIVE_BACKEND_V2;
  else if (execution_mode == "host")
    spec.execution_mode = POPS_TAGGER_EXECUTION_HOST_V2;
  else
    throw std::invalid_argument("AMR Tagger declares an unknown execution_mode");
  const std::string collective_scope = py::cast<std::string>(capability["collective_scope"]);
  if (collective_scope != "none")
    throw std::invalid_argument("AMR Tagger callbacks must be explicitly noncollective");
  spec.collective_scope = POPS_TAGGER_COLLECTIVE_NONE_V2;
  for (const std::string& memory_space :
       py::cast<std::vector<std::string>>(capability["memory_spaces"])) {
    if (memory_space == "host")
      spec.memory_spaces.push_back(POPS_MEMORY_SPACE_HOST_V1);
    else if (memory_space == "managed")
      spec.memory_spaces.push_back(POPS_MEMORY_SPACE_MANAGED_V1);
    else if (memory_space == "device")
      spec.memory_spaces.push_back(POPS_MEMORY_SPACE_DEVICE_V1);
    else
      throw std::invalid_argument("AMR Tagger declares an unknown memory space");
  }
  const std::string non_finite_policy = py::cast<std::string>(capability["non_finite_policy"]);
  if (non_finite_policy != "reject")
    throw std::invalid_argument("AMR Tagger native adapter requires non_finite_policy='reject'");
  spec.non_finite_policy = POPS_TAGGING_NON_FINITE_REJECT_V1;
  spec.interface_version = py::cast<std::uint32_t>(row["interface_version"]);
  spec.execution = pops::python::detail::make_component_execution_context(execution);
  return spec;
}

pops::runtime::amr::PreparedTaggingProgram::Stencil amr_tagging_stencil_from_python(
    const py::dict& row) {
  using Program = pops::runtime::amr::PreparedTaggingProgram;
  Program::Stencil result;
  result.identity = py::cast<std::string>(row["identity"]);
  result.route = py::cast<std::string>(row["route"]);
  result.norm = py::cast<std::string>(row["norm"]);
  result.scale = py::cast<std::string>(row["scale"]);
  result.boundary_mode = py::cast<std::string>(row["boundary_mode"]);
  result.dimension = py::cast<std::int32_t>(row["dimension"]);
  for (const py::handle value : py::cast<py::list>(row["axes"])) {
    const py::dict axis = py::cast<py::dict>(value);
    std::vector<double> coefficients;
    for (const py::handle coefficient_value : py::cast<py::list>(axis["coefficients"])) {
      const py::dict coefficient = py::cast<py::dict>(coefficient_value);
      if (coefficient.size() != 1 || !coefficient.contains("binary64"))
        throw std::invalid_argument(
            "AMR Tagger stencil coefficient is not canonical binary64 data");
      const std::string encoded = py::cast<std::string>(coefficient["binary64"]);
      std::size_t consumed = 0;
      const double parsed = std::stod(encoded, &consumed);
      if (consumed != encoded.size() || !std::isfinite(parsed))
        throw std::invalid_argument(
            "AMR Tagger stencil coefficient is not finite canonical binary64 data");
      coefficients.push_back(parsed);
    }
    result.axes.push_back(Program::AxisStencil{
        py::cast<std::int32_t>(axis["axis"]), py::cast<std::int32_t>(axis["derivative_order"]),
        py::cast<std::int32_t>(axis["formal_order"]), py::cast<std::size_t>(axis["ghost_lower"]),
        py::cast<std::size_t>(axis["ghost_upper"]),
        py::cast<std::vector<std::int32_t>>(axis["offsets"]), std::move(coefficients)});
  }
  return result;
}

pops::runtime::amr::PreparedClusteringSpec amr_clustering_spec_from_python(
    const py::dict& row, const py::dict& execution) {
  pops::runtime::amr::PreparedClusteringSpec spec;
  spec.provider_identity = py::cast<std::string>(row["provider_identity"]);
  spec.component_id = py::cast<std::string>(row["component_id"]);
  spec.manifest_identity = py::cast<std::string>(row["component_manifest_identity"]);
  spec.layout_identity = py::cast<std::string>(row["layout_identity"]);
  spec.interface_version = py::cast<std::uint32_t>(row["interface_version"]);
  spec.execution = pops::python::detail::make_component_execution_context(execution);
  return spec;
}

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
             double positivity_floor, double weno_epsilon, bool wave_speed_cache) {
            NewtonOptions newton;
            newton.max_iters = newton_max_iters;
            newton.rel_tol = static_cast<Real>(newton_rel_tol);
            newton.abs_tol = static_cast<Real>(newton_abs_tol);
            newton.fd_eps = static_cast<Real>(newton_fd_eps);
            newton.damping = static_cast<Real>(newton_damping);
            newton.fail_policy =
                newton_fail_policy_from_string(newton_fail_policy, "AmrSystem::add_block");
            s.add_block(name, model, limiter, riemann, recon, time, substeps, stride, implicit_vars,
                        implicit_roles, newton, newton_diagnostics, positivity_floor,
                        weno_epsilon, wave_speed_cache);
          },
          py::arg("name"), py::arg("model"), py::arg("limiter") = "minmod",
          py::arg("riemann") = "rusanov", py::arg("recon") = "conservative",
          py::arg("time") = "explicit", py::arg("substeps") = 1, py::arg("stride") = 1,
          // Partial IMEX mask CARRIED BY THE BLOCK (capstone vii): conserved variables treated
          // implicitly by NAME (implicit_vars) or by physical ROLE (implicit_roles). Empty (default)
          // -> full backward-Euler. Only meaningful with time="imex" and MULTI-BLOCK (cf. add_block).
          py::arg("implicit_vars") = std::vector<std::string>{},
          py::arg("implicit_roles") = std::vector<std::string>{},
          // IMEX Newton options and newton_diagnostics use the native unified AMR runtime at every
          // block count. Compiled .so loaders reject values their flat ABI cannot transport.
          py::arg("newton_max_iters") = kNewtonDefaultMaxIters,
          py::arg("newton_rel_tol") = static_cast<double>(kNewtonDefaultRelTol),
          py::arg("newton_abs_tol") = static_cast<double>(kNewtonDefaultAbsTol),
          py::arg("newton_fd_eps") = static_cast<double>(kNewtonDefaultFdEps),
          py::arg("newton_damping") = static_cast<double>(kNewtonDefaultDamping),
          py::arg("newton_fail_policy") = "none", py::arg("newton_diagnostics") = false,
          // Zhang-Shu positivity floor (ADC-259): Density-role face-state + C/F-ghost-mean floor on
          // the AMR transport. 0 (default) = inactive, bit-identical. Marshaled from spatial.positivity_floor
          // by the AmrSystem.add_block / add_equation Python facade.
          py::arg("positivity_floor") = 0.0,
          py::arg("weno_epsilon") = static_cast<double>(kWenoEpsilon),
          py::arg("wave_speed_cache") = false)
      .def(
          "_install_boundary_plan",
          [](AmrSystem& system, const std::string& name, const std::string& identity,
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
          "Install one resolved per-block ghost-production plan before lazy AMR construction.")
      .def("_install_block_state_route", &AmrSystem::install_block_state_route, py::arg("name"),
           py::arg("state_identity"),
           "Bind one exact state Handle identity to native AMR block storage.")
      .def("_install_boundary_field_route", &AmrSystem::install_boundary_field_route,
           py::arg("field_identity"), py::arg("provider_slot"),
           "Bind one exact boundary field Handle to native provider storage.")
      .def("_discard_boundary_plans", &AmrSystem::discard_boundary_plans,
           "Roll back one failed pre-block boundary authority transaction.")
      .def(
          "_install_amr_tagger_component",
          [](AmrSystem& system, std::shared_ptr<pops::component::LoadedComponent> component,
             const py::dict& binding, const py::dict& execution) {
            system.install_amr_tagger_component(amr_tagger_spec_from_python(binding, execution),
                                                std::move(component));
          },
          py::arg("component"), py::arg("binding"), py::arg("execution_context"))
      .def(
          "_install_amr_clustering_component",
          [](AmrSystem& system, std::shared_ptr<pops::component::LoadedComponent> component,
             const py::dict& binding, const py::dict& execution) {
            system.install_amr_clustering_component(
                amr_clustering_spec_from_python(binding, execution), std::move(component));
          },
          py::arg("component"), py::arg("binding"), py::arg("execution_context"))
      .def("_discard_amr_provider_components", &AmrSystem::discard_amr_provider_components,
           "Roll back one failed external AMR provider transaction.")
      .def(
          "_install_ghost_boundary_component",
          [](AmrSystem& system, const std::string& name,
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
          [](AmrSystem& system, const std::string& name,
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
          [](AmrSystem& system, const std::string& name,
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
          [](AmrSystem& system, std::size_t left_block, std::size_t right_block, int level,
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
      .def("_interface_evaluation_count", &AmrSystem::interface_evaluation_count,
           py::arg("identity"), py::arg("level") = 0)
      .def("_discard_interface_flux_components", &AmrSystem::discard_interface_flux_components,
           "Roll back one failed post-block interface authority transaction.")
      // Newton report (IMEX diagnostics OPT-IN, native AMR runtime): dict {enabled, converged,
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
      // Private production-package seam. Parameters are fixed before AMR closures are built.
      .def("_install_native_block", &AmrSystem::add_native_block, py::arg("name"),
           py::arg("so_path"), py::arg("limiter") = "minmod", py::arg("riemann") = "rusanov",
           py::arg("recon") = "conservative", py::arg("time") = "explicit",
           py::arg("gamma") = static_cast<double>(kPhysicalDefaultGamma), py::arg("substeps") = 1,
           py::arg("params") = std::vector<double>{},
           // Zhang-Shu positivity floor (ADC-322): marshaled down the regenerated .so loader
           // (pops_install_native_amr). 0 (default) = inactive, bit-identical.
           py::arg("positivity_floor") = 0.0,
           py::arg("weno_epsilon") = static_cast<double>(kWenoEpsilon),
           py::arg("wave_speed_cache") = false)
      .def("_install_external_riemann_block", &AmrSystem::add_external_riemann_block,
           py::arg("name"), py::arg("so_path"), py::arg("brick_id"), py::arg("sha256"),
           py::arg("limiter"), py::arg("recon"), py::arg("time"), py::arg("gamma"),
           py::arg("substeps"), py::arg("stride"), py::arg("expected_nvars"),
           py::arg("expected_naux"), py::arg("expected_model_identity"),
           py::arg("positivity_floor") = 0.0,
           py::arg("weno_epsilon") = static_cast<double>(kWenoEpsilon))
      // Regrid criterion: refine where the SELECTED variable exceeds threshold. Default = component 0
      // (historical density), bit-identical 1e30 no-op. ADC-296: select it PER BLOCK by NAME (variable=)
      // or physical ROLE (role=); a block lacking it raises at build (no silent comp-0 fallback).
      // Native and compiled runtime blocks carry the same exact VariableSet descriptor.
      .def("set_refinement", &AmrSystem::set_refinement, py::arg("threshold"),
           py::arg("variable") = "", py::arg("role") = "",
           "Refine where the selected conserved variable exceeds threshold. variable=/role= pick "
           "it per "
           "block by name or physical role (default: component 0, the historical density). "
           "Selecting by "
           "name and role at once, or a name/role absent from a block, raises.")
      .def("_set_bootstrap_refinement", &AmrSystem::set_bootstrap_refinement, py::arg("block"),
           py::arg("variable"), py::arg("threshold"), py::arg("provider_identity"))
      .def(
          "_set_bootstrap_tagging",
          [](AmrSystem& system, const std::vector<std::string>& leaf_blocks,
             const std::vector<std::string>& leaf_variables, const std::vector<int>& leaf_ops,
             const std::vector<double>& leaf_thresholds,
             const std::vector<int>& leaf_stencil_indices, const py::list& stencil_rows,
             const std::vector<std::int32_t>& refine_ops,
             const std::vector<std::int32_t>& refine_args,
             const std::vector<std::int32_t>& coarsen_ops,
             const std::vector<std::int32_t>& coarsen_args, int min_cycles,
             const std::string& equality_policy, const std::string& conflict_policy,
             const std::string& clock_identity, const std::string& provider_identity) {
            std::vector<pops::runtime::amr::PreparedTaggingProgram::Stencil> stencils;
            stencils.reserve(stencil_rows.size());
            for (const py::handle row : stencil_rows)
              stencils.push_back(amr_tagging_stencil_from_python(py::cast<py::dict>(row)));
            system.set_bootstrap_tagging(leaf_blocks, leaf_variables, leaf_ops, leaf_thresholds,
                                         leaf_stencil_indices, stencils, refine_ops, refine_args,
                                         coarsen_ops, coarsen_args, min_cycles, equality_policy,
                                         conflict_policy, clock_identity, provider_identity);
          },
          py::arg("leaf_blocks"), py::arg("leaf_variables"), py::arg("leaf_ops"),
          py::arg("leaf_thresholds"), py::arg("leaf_stencil_indices"), py::arg("stencils"),
          py::arg("refine_ops"), py::arg("refine_args"), py::arg("coarsen_ops"),
          py::arg("coarsen_args"), py::arg("min_cycles"), py::arg("equality_policy"),
          py::arg("conflict_policy"), py::arg("clock_identity"), py::arg("provider_identity"))
      // Shared-potential gradient leaf appended to the prepared regrid graph. It uses the same
      // native Kokkos/MPI Tagger route as model-state criteria; <= 0 keeps the leaf absent.
      .def("set_phi_refinement", &AmrSystem::set_phi_refinement, py::arg("grad_threshold"))
      .def(
          "set_poisson",
          [](AmrSystem& system, const std::string& rhs, const std::string& solver,
             const std::string& bc, const std::string& wall,
             double wall_radius) { system.set_poisson(rhs, solver, bc, wall, wall_radius); },
          "Configures the default AMR field through the registered native provider. The Python "
          "shortcut selects provider defaults; resolved provider-specific options are installed by "
          "the compiled field-plan pipeline.",
          py::arg("rhs") = "charge_density", py::arg("solver") = "geometric_mg",
          py::arg("bc") = "auto", py::arg("wall") = "none", py::arg("wall_radius") = 0.0)
      .def(
          "set_field_solver_plan",
          [](AmrSystem& system, const std::string& provider_slot, const std::string& plan_identity,
             const std::string& provider_identity, const std::string& output_owner_identity,
             const std::string& output_block, const std::string& output_key,
             const std::vector<std::string>& provider_identities,
             const std::vector<std::string>& provider_blocks,
             const std::vector<std::string>& provider_keys,
             const std::vector<double>& provider_coefficients, const std::string& solver,
             const std::string& hierarchy_policy_id,
             std::uint64_t hierarchy_policy_interface_version,
             const std::string& hierarchy_policy_option_schema,
             const py::dict& hierarchy_policy_options, const std::string& schema_identity,
             const py::dict& options) {
            const AmrFieldHierarchyPolicyAuthority hierarchy_policy{
                hierarchy_policy_id,
                hierarchy_policy_interface_version,
                prepared_provider_options_from_python(hierarchy_policy_option_schema,
                                                      hierarchy_policy_options),
            };
            system.set_field_solver_plan(
                provider_slot, plan_identity, provider_identity, output_owner_identity,
                output_block, output_key, provider_identities, provider_blocks, provider_keys,
                provider_coefficients, solver, hierarchy_policy,
                prepared_provider_options_from_python(schema_identity, options));
          },
          py::arg("provider_slot"), py::arg("plan_identity"), py::arg("provider_identity"),
          py::arg("output_owner_identity"), py::arg("output_block"), py::arg("output_key"),
          py::arg("provider_identities"), py::arg("provider_blocks"), py::arg("provider_keys"),
          py::arg("provider_coefficients"), py::arg("solver"), py::arg("hierarchy_policy_id"),
          py::arg("hierarchy_policy_interface_version"), py::arg("hierarchy_policy_option_schema"),
          py::arg("hierarchy_policy_options"), py::arg("schema_identity"), py::arg("options"))
      .def(
          "field_solver_configuration",
          [](const AmrSystem& system, const std::string& provider_slot) {
            const AmrFieldSolverConfiguration config =
                system.field_solver_configuration(provider_slot);
            py::dict result;
            result["schema_version"] = 1;
            result["provider_slot"] = provider_slot;
            result["plan_identity"] = config.plan_identity;
            result["provider_identity"] = config.provider_identity;
            result["solver"] = config.solver;
            py::dict hierarchy_policy;
            hierarchy_policy["policy_id"] = config.hierarchy_policy.policy_id;
            hierarchy_policy["interface_version"] = config.hierarchy_policy.interface_version;
            hierarchy_policy["option_schema"] = config.hierarchy_policy.options.schema_identity;
            hierarchy_policy["options"] =
                prepared_provider_options_to_python(config.hierarchy_policy.options);
            result["hierarchy_policy"] = std::move(hierarchy_policy);
            result["option_schema_identity"] = config.options.schema_identity;
            result["options"] = prepared_provider_options_to_python(config.options);
            return result;
          },
          py::arg("provider_slot"))
      .def("set_field_reaction", &AmrSystem::set_field_reaction, py::arg("provider_slot"),
           py::arg("reaction"))
      .def("_set_field_topology_authority", &AmrSystem::set_field_topology_authority,
           py::arg("provider_slot"), py::arg("provider_kind"), py::arg("provenance"),
           py::arg("topology_digest"))
      .def(
          "_field_topology_report",
          [](const AmrSystem& system, const std::string& provider_slot) {
            py::list report;
            for (const auto& row : system.field_topology_report(provider_slot)) {
              py::dict item;
              item["patch_identity"] = row.patch_identity;
              item["topology_digest"] = row.topology_digest;
              item["provenance"] = row.provenance;
              item["material_points"] = row.material_points;
              item["connected_components"] = row.connected_components;
              report.append(std::move(item));
            }
            return report;
          },
          py::arg("provider_slot"))
      .def("register_elliptic_field", &AmrSystem::register_elliptic_field, py::arg("block"),
           py::arg("field"), py::arg("phi_comp"), py::arg("gx_comp"), py::arg("gy_comp"),
           py::arg("gradient_sign"))
      .def("set_field_boundary_plan", &AmrSystem::set_field_boundary_plan, py::arg("provider_slot"),
           py::arg("kind"), py::arg("alpha"), py::arg("beta"), py::arg("value"))
      .def("set_field_boundary_dependencies", &AmrSystem::set_field_boundary_dependencies,
           py::arg("provider_slot"), py::arg("state_blocks"), py::arg("state_components"),
           py::arg("field_blocks"), py::arg("field_keys"), py::arg("field_components"))
      .def("set_field_boundary_parameters", &AmrSystem::set_field_boundary_parameters,
           py::arg("provider_slot"), py::arg("parameters"))
      .def("set_field_newton_plan", &AmrSystem::set_field_newton_plan, py::arg("provider_slot"),
           py::arg("tolerance"), py::arg("max_iterations"), py::arg("linear_tolerance"),
           py::arg("linear_max_iterations"), py::arg("restart"), py::arg("armijo"),
           py::arg("minimum_step"))
      .def(
          "set_field_nullspace",
          [](AmrSystem& system, const std::string& provider_slot,
             const std::string& provider_identity, const std::string& schema_identity,
             const py::dict& options) {
            system.set_field_nullspace(
                provider_slot, provider_identity,
                prepared_provider_options_from_python(schema_identity, options));
          },
          py::arg("provider_slot"), py::arg("provider_identity"), py::arg("schema_identity"),
          py::arg("options"))
      .def(
          "set_default_field_nullspace",
          [](AmrSystem& system, const std::string& provider_identity,
             const std::string& schema_identity, const py::dict& options) {
            system.set_default_field_nullspace(
                provider_identity, prepared_provider_options_from_python(schema_identity, options));
          },
          py::arg("provider_identity"), py::arg("schema_identity"), py::arg("options"));
}

// Physics wiring: dt bounds, fields, and coupled source stages.
void bind_amr_physics(py::class_<AmrSystem>& cls) {
  cls
      // GLOBAL step bound + ACTIVE bound (AMR StabilityPolicy, System.add_dt_bound parity).
      .def("add_dt_bound", &AmrSystem::add_dt_bound, py::arg("label"), py::arg("fn"))
      .def("last_dt_bound", &AmrSystem::last_dt_bound)
      // Python owns the Cartesian orientation: exactly (ny, nx), then one explicit flattening at
      // the vector-valued C++ boundary.
      .def(
          "set_magnetic_field",
          [](AmrSystem& s, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            require_amr_cell_array_shape(s, arr, "AmrSystem.set_magnetic_field");
            s.set_magnetic_field(flat(arr));
          },
          py::arg("bz"), "Set the coarse magnetic field from exactly one (ny, nx) array.")
      // ADC-291: model-NAMED aux field at a resolved channel component (>= kAuxNamedBase). The Python
      // facade resolves the name -> comp and flattens the exact (ny, nx) Cartesian field.
      .def(
          "set_aux_field_component",
          [](AmrSystem& s, int comp,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            require_amr_cell_array_shape(s, arr, "AmrSystem.set_aux_field_component");
            s.set_aux_field_component(comp, flat(arr));
          },
          py::arg("comp"), py::arg("field"),
          "Set one coarse auxiliary component from exactly one (ny, nx) array.")
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
            require_amr_cell_array_shape(s, arr, "AmrSystem.set_density");
            s.set_density(name, flat(arr));
          },
          py::arg("name"), py::arg("rho"),
          "Set a block's coarse density from exactly one (ny, nx) array.")
      // Full initial conservative state (ncomp, ny, nx) -> starts the AMR from the paper's drift
      // state (rho, rho*u, rho*v) instead of m=0. Keeps ndim==3 EXPLICIT: flat() flattens
      // any C-contiguous array, so a 2D density (ny, nx) passed by mistake would become a
      // 1-component state (comp 0 = density, momentum left at 0) -- a silent density masquerade
      // with the wrong physics. flat() then flattens in component-major c*cells + j*nx + i.
      .def(
          "set_conservative_state",
          [](AmrSystem& s, const std::string& name,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            if (arr.ndim() != 3)
              throw std::runtime_error(
                  "AmrSystem.set_conservative_state: state expected of shape (ncomp, ny, nx); got "
                  "a " +
                  std::to_string(arr.ndim()) +
                  "D array (a 2D density? use "
                  "set_density)");
            if (arr.shape(1) != s.ny() || arr.shape(2) != s.nx())
              throw std::runtime_error(
                  "AmrSystem.set_conservative_state: spatial shape differs from (ny, nx)");
            s.set_conservative_state(name, flat(arr));
          },
          py::arg("name"), py::arg("U"))
      .def("_begin_bootstrap_plan", &AmrSystem::begin_bootstrap_plan)
      .def("_bootstrap_next_level", &AmrSystem::bootstrap_next_level)
      .def("_commit_bootstrap_level", &AmrSystem::commit_bootstrap_level)
      .def("_rollback_bootstrap_level", &AmrSystem::rollback_bootstrap_level)
      .def("_register_bootstrap_transfer_route", &AmrSystem::register_bootstrap_transfer_route,
           py::arg("identity"), py::arg("subjects"), py::arg("provider_identity"), py::arg("space"),
           py::arg("centering"), py::arg("representation"), py::arg("storage"),
           py::arg("operation"), py::arg("kernel"), py::arg("order"), py::arg("ghost_depth"),
           py::arg("dimension"), py::arg("refinement_ratio"))
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
           py::arg("subject"), py::arg("block"), py::arg("space"), py::arg("centering"),
           py::arg("components"))
      .def("_register_analytic_gaussian", &AmrSystem::register_analytic_gaussian,
           py::arg("subject"), py::arg("block"), py::arg("center_x"), py::arg("center_y"),
           py::arg("background"), py::arg("amplitude"), py::arg("inverse_width"))
      .def("_register_analytic_expression", &AmrSystem::register_analytic_expression,
           py::arg("subject"), py::arg("block"), py::arg("space"), py::arg("centering"),
           py::arg("opcodes"), py::arg("literals"))
      .def("_bootstrap_analytic_reproject", &AmrSystem::bootstrap_analytic_reproject,
           py::arg("subject"), py::arg("level"))
      .def("_apply_bootstrap_component_floor", &AmrSystem::apply_bootstrap_component_floor,
           py::arg("subject"), py::arg("level"), py::arg("component"), py::arg("floor"))
      .def("_recompute_bootstrap_field", &AmrSystem::recompute_bootstrap_field, py::arg("subject"),
           py::arg("field_name"))
      .def("_bootstrap_prolong_array", &AmrSystem::bootstrap_prolong_array, py::arg("subject"),
           py::arg("level"))
      .def("_synchronize_bootstrap_state", &AmrSystem::synchronize_bootstrap_state,
           py::arg("subject"), py::arg("fine_level"))
      .def("_bootstrap_array_level", &AmrSystem::bootstrap_array_level, py::arg("subject"),
           py::arg("level"))
      .def("_invalidate_bootstrap_cache", &AmrSystem::invalidate_bootstrap_cache,
           py::arg("subject"), py::arg("level"))
      .def(
          "_rebuild_bootstrap_topology_cache",
          [](AmrSystem& s, const std::string& subject, int level) {
            py::list out;
            for (const pops::PatchBox& b : s.rebuild_bootstrap_topology_cache(subject, level))
              out.append(py::make_tuple(b.level, b.ilo, b.jlo, b.ihi, b.jhi));
            return out;
          },
          py::arg("subject"), py::arg("level"))
      .def("_bootstrap_cache_epoch", &AmrSystem::bootstrap_cache_epoch, py::arg("subject"))
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
      .def("_begin_step_transaction", &AmrSystem::begin_step_transaction)
      .def("_commit_step_transaction", &AmrSystem::commit_step_transaction)
      .def("_step_change_l2", &AmrSystem::step_change_l2)
      .def("_finalize_step_transaction", &AmrSystem::finalize_step_transaction)
      .def("_rollback_step_transaction", &AmrSystem::rollback_step_transaction)
      .def("step_cfl", &AmrSystem::step_cfl,
           "Advances by one AMR macro-step at dt = cfl * dx_coarse / max wave speed (also honors "
           "the substeps/stride cadence in multi-block and the optional bounds). Returns the dt "
           "used. speed_floor (ADC-645): the floor applied to the reduced max wave speed on the "
           "multi-block runtime engine (default = the historical kCflSpeedFloor, bit-identical); "
           "refused non-default on the single-block coupler (no historical floor site there).",
           py::arg("cfl"), py::arg("speed_floor") = static_cast<double>(kCflSpeedFloor),
           py::arg("max_dt") = std::numeric_limits<double>::infinity(), py::arg("min_dt") = 0.0)
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
      .def(
          "profile_snapshot",
          [](AmrSystem& s) { return profile_snapshot_to_dict(s.profiler_handle().snapshot()); },
          "Structured AMR profiling snapshot: schema_version, enabled, scopes and counters.");
}

// Clock + compiled-Program install/introspection + runtime freeze lifecycle.
void bind_amr_program(py::class_<AmrSystem>& cls) {
  cls.def("nx", &AmrSystem::nx)
      .def("ny", &AmrSystem::ny)
      .def("time", &AmrSystem::time)
      // AMR clock (IO v1, System parity): macro-step counter + restoration (t, macro_step) ->
      // the regrid/stride cadence resumes exactly after a set_clock. Prerequisite PR-IO-3.
      .def("macro_step", &AmrSystem::macro_step)
      .def("set_clock", &AmrSystem::set_clock, py::arg("t"), py::arg("macro_step"))
      .def("field_provider_slots", &AmrSystem::field_provider_slots)
      .def("field_provider_levels", &AmrSystem::field_provider_levels, py::arg("provider_slot"))
      .def("set_field_potential", &AmrSystem::set_field_potential, py::arg("provider_slot"),
           py::arg("phi"))
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
      // Internal compiled-kernel cadence seam; the public controller is Program.step_strategy().
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
      // IR hash of the installed compiled Program (the .so's pops_program_hash), or "" if none. Parity
      // System::installed_program_hash (the checkpoint guard).
      .def("installed_program_hash", &AmrSystem::installed_program_hash)
      .def("program_accepted_state",
           [](const AmrSystem& s) {
             const auto bytes = s.program_accepted_state();
             return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
           })
      .def(
          "restore_program_accepted_state",
          [](AmrSystem& s, py::bytes payload) {
            std::string bytes = payload;
            s.restore_program_accepted_state(std::vector<std::uint8_t>(bytes.begin(), bytes.end()));
          },
          py::arg("payload"))
      .def(
          "materialize_program_restart_histories",
          [](AmrSystem& s, py::bytes payload, const std::vector<std::string>& names,
             const std::vector<int>& depths, const std::vector<int>& ncomps) {
            std::string bytes = payload;
            s.materialize_program_restart_histories(
                std::vector<std::uint8_t>(bytes.begin(), bytes.end()), names, depths, ncomps);
          },
          py::arg("payload"), py::arg("names"), py::arg("depths"), py::arg("ncomps"))
      .def("program_accepted_state_manifest", &AmrSystem::program_accepted_state_manifest)
      .def("program_clock_manifest", &AmrSystem::program_clock_manifest)
      .def("program_flux_ledger_manifest", &AmrSystem::program_flux_ledger_manifest)
      .def("program_sync_manifest", &AmrSystem::program_sync_manifest)
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
      // path drives -- exact selected levels, with every coarser selected footprint masked by the
      // next selected finer footprint.
      .def("composite_reduce", &AmrSystem::composite_reduce, py::arg("block"), py::arg("kind"),
           py::arg("comp") = 0, py::arg("levels") = std::vector<int>{})
      .def("composite_reduce_field", &AmrSystem::composite_reduce_field, py::arg("provider_slot"),
           py::arg("kind"), py::arg("comp") = 0, py::arg("levels") = std::vector<int>{})
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
      .def(
          "effective_options_report",
          [](const AmrSystem& s) {
            return effective_options_report_to_dict(s.effective_options_report());
          },
          "Structured effective numerical/solver/physical options for this AmrSystem.")
      .def("n_patches", &AmrSystem::n_patches)
      // Index-space footprints of the fine patches: list of tuples (level, ilo, jlo, ihi, jhi), INCLUSIVE
      // corners, in the axis-resolved index space of the level (ratio 2). SAME
      // source as n_patches() (the GLOBAL fine BoxArray) -> rank-independent, MPI-safe. Query between
      // steps, zero cost on the hot path. The Python wrapper converts with the exact x/y bounds;
      // cf. AmrSystem.patch_rectangles() on the facade side.
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
      .def("density", [](AmrSystem& s) { return to_2d(s.density(), s.ny(), s.nx()); })
      .def(
          "density",
          [](AmrSystem& s, const std::string& name) {
            return to_2d(s.density(name), s.ny(), s.nx());
          },
          py::arg("name"))
      // phi of the coarse (base) level, (ny, nx). SAME observable as System.potential(): level 0
      // covers the whole domain -> enough to sample a median circle (azimuthal FFT). In
      // multi-block, phi results from the SYSTEM Poisson (Sum_b q_b n_b co-located), shared by all.
      .def("potential", [](AmrSystem& s) { return to_2d(s.potential(), s.ny(), s.nx()); })
      // ADC-428: solved potential of a NAMED elliptic field (m.elliptic_field) on the coarse level,
      // (ny, nx). Read-back counterpart of potential() for a second elliptic field; the Python
      // AmrSystem.field(name) resolves the field name to this. Solves the hierarchy if needed.
      .def(
          "named_field_values",
          [](AmrSystem& s, const std::string& field) {
            return to_2d(s.named_field_values(field), s.ny(), s.nx());
          },
          py::arg("field"))
      // AMR CHECKPOINT / RESTART single-rank (ADC-65): full conservative state per level + phi
      // (warm-start) + imposition of the saved fine hierarchy. SERIAL MONO-BLOCK (multi-block: C++
      // rejection; np>1: facade rejection -- per-level gather = future). level_state / level_potential return
      // FLAT fields (c*nf*nf + j*nf + i / nf*nf, nf = nx << k); the facade reshapes. set_*
      // flatten any C-contiguous array (flat). set_hierarchy: list of tuples
      // (level, ilo, jlo, ihi, jhi) like patch_boxes() (the coupler filters level 1).
      .def("n_levels", &AmrSystem::n_levels)
      .def("max_levels", &AmrSystem::max_levels)
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
      .def("field_potential_global", &AmrSystem::field_potential_global, py::arg("provider_slot"))
      .def("field_potential_level_global", &AmrSystem::field_potential_level_global,
           py::arg("provider_slot"), py::arg("level"))
      .def(
          "output_field_local_pieces",
          [](AmrSystem& s, const std::string& provider_slot, int level) {
            return output_pieces_to_python(s.output_field_local_pieces(provider_slot, level));
          },
          py::arg("provider_slot"), py::arg("level"),
          "Exact compact valid-cell pieces of one qualified field owned by this rank.")
      .def(
          "output_field_root_pieces",
          [](AmrSystem& s, const WorldCommunicator& world, const std::string& provider_slot,
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
          [](AmrSystem& s, int level, const std::array<double, 2>& origin,
             const std::array<double, 2>& spacing, const std::array<std::int64_t, 2>& cell_shape,
             int next_refinement_ratio, const std::string& cell_measure) {
            if (level < 0 || level >= s.n_levels())
              throw std::out_of_range("AmrSystem output geometry level is out of range");
            return pops::python::detail::native_output_geometry_snapshot(
                level, s.checkpoint_topology_epoch(), origin, spacing, cell_shape, cell_measure,
                s.output_geometry_boxes(), next_refinement_ratio, true);
          },
          py::arg("level"), py::arg("origin"), py::arg("spacing"), py::arg("cell_shape"),
          py::arg("next_refinement_ratio"), py::arg("cell_measure"),
          "Private Writer geometry view: native, immutable, and topology-versioned.")
      // MULTI-BLOCK per-BLOCK per-level state (ADC-509): the AmrRuntime engine shares the layout +
      // aux, so the per-level STATE is read/restored PER BLOCK (by name) while phi stays shared
      // (level_potential). block_level_state returns a FLAT field (c*nf*nf + j*nf + i); the _global
      // variant gathers under np>1; set_block_level_state flattens any C-contiguous array.
      .def(
          "block_n_vars",
          [](AmrSystem& s, const std::string& name) { return s.block_n_vars(name); },
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
          "output_state_local_pieces",
          [](AmrSystem& s, const std::string& name, int level) {
            return output_pieces_to_python(s.output_state_local_pieces(name, level));
          },
          py::arg("block"), py::arg("level"),
          "Exact compact valid-cell pieces of one qualified state owned by this rank.")
      .def(
          "output_state_root_pieces",
          [](AmrSystem& s, const WorldCommunicator& world, const std::string& name, int level) {
            std::vector<OutputPiece> pieces;
            {
              py::gil_scoped_release release;
              pieces = s.output_state_root_pieces(world, name, level);
            }
            return output_pieces_to_python(pieces);
          },
          py::arg("world"), py::arg("block"), py::arg("level"),
          "Collectively gather compact state pieces in C++; complete only on MPI rank zero.")
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
      .def("checkpoint_temporal_relations", &AmrSystem::checkpoint_temporal_relations)
      .def("set_temporal_relations", &AmrSystem::set_temporal_relations, py::arg("numerators"),
           py::arg("denominators"), py::arg("remainder_policies"))
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
      .def_readwrite("ny", &AmrSystemConfig::ny)
      .def_readwrite("L", &AmrSystemConfig::L)
      .def_readwrite("Ly", &AmrSystemConfig::Ly)
      .def_readwrite("regrid_every", &AmrSystemConfig::regrid_every)
      .def_readwrite("level_count", &AmrSystemConfig::level_count)
      .def_readwrite("regrid_grow", &AmrSystemConfig::regrid_grow)
      .def_readwrite("regrid_margin", &AmrSystemConfig::regrid_margin)
      .def_readwrite("explicit_bootstrap", &AmrSystemConfig::explicit_bootstrap)
      .def_property(
          "periodicity", [](const AmrSystemConfig& config) {
            return periodicity_to_python(config.periodicity);
          },
          [](AmrSystemConfig& config, const py::handle& value) {
            config.periodicity = periodicity_from_python(value, "AmrSystemConfig");
          })
      .def_readwrite("distribute_coarse", &AmrSystemConfig::distribute_coarse)
      .def_readwrite("coarse_max_grid", &AmrSystemConfig::coarse_max_grid)
      // ADC-616: Berger-Rigoutsos clustering params (<= 0 = the historical {0.7, 1, 32} default).
      .def_readwrite("cluster_min_efficiency", &AmrSystemConfig::cluster_min_efficiency)
      .def_readwrite("cluster_min_box_size", &AmrSystemConfig::cluster_min_box_size)
      .def_readwrite("cluster_max_box_size", &AmrSystemConfig::cluster_max_box_size)
      .def_readwrite("xlo", &AmrSystemConfig::xlo)
      .def_readwrite("ylo", &AmrSystemConfig::ylo)
      .def(
          "_set_load_balance_provider",
          [](AmrSystemConfig& config, const std::string& route,
             const std::string& semantic_identity, const std::string& option_schema_identity,
             const py::dict& options) {
            if (route.empty() || semantic_identity.empty())
              throw py::value_error(
                  "AMR load-balance route and semantic identity must be non-empty");
            config.load_balance_route = route;
            config.load_balance_identity = semantic_identity;
            config.load_balance_options =
                prepared_provider_options_from_python(option_schema_identity, options);
          },
          py::arg("route"), py::arg("semantic_identity"),
          py::arg("option_schema_identity"), py::arg("options"));

  // AmrSystem: generic single-species composition on AMR.
  py::class_<AmrSystem> cls(m, "AmrSystem");
  bind_amr_assembly(cls);
  bind_amr_physics(cls);
  bind_amr_stepping(cls);
  bind_amr_program(cls);
  bind_amr_data(cls);
}
