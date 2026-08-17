#include "../bindings_detail.hpp"
#include <pops/parallel/execution_lane.hpp>
#include <pops/parallel/world_communicator.hpp>
#include "boundary_component_install.hpp"
#include "checkpoint_spatial_binding.hpp"
#include "output_geometry_binding.hpp"

#include <pops/runtime/amr/prepared_tagging_execution.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// ADC-365: the AMR (AmrSystemConfig + AmrSystem) bindings.
//
// ADC-593: like init_system, the AmrSystem .def registrations are INTERNAL seams of the bind flow (the
// AMR target reaches them through the typed layout, not as public vocabulary). The AmrSystem chain is
// split into concern-grouped static helpers (assembly / physics / stepping / program / data), each taking
// the class handle and adding its slice. PURE reorganization: same .def names, docstrings, args, and
// RELATIVE order (no overload set is reordered -- every AmrSystem method name here is unique). The class
// name and the .def names are unchanged, so the legacy-name architecture gate still finds them here.
namespace {

using AmrSystem = pops::AmrSystem<pops::kNativeDimension>;
using AmrSystemConfig = pops::AmrSystemConfig<pops::kNativeDimension>;

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

struct PreparedAmrFieldOutputPublication {
  std::string owner_identity;
  std::string block;
  std::string field;
  std::vector<pops::runtime::system::AuxiliaryComponentKey> output_keys;
  int gradient_sign = 1;
};

std::string exact_publication_string(const py::handle& value, std::string_view field) {
  if (!PyUnicode_CheckExact(value.ptr()))
    throw py::type_error("AMR field output publication '" + std::string(field) +
                         "' must be an exact string");
  const std::string result = py::cast<std::string>(value);
  if (result.empty())
    throw py::value_error("AMR field output publication '" + std::string(field) +
                          "' must be non-empty");
  return result;
}

PreparedAmrFieldOutputPublication prepared_amr_field_output_publication_from_python(
    const py::dict& publication) {
  if (!PyDict_CheckExact(publication.ptr()) || publication.size() != 6 ||
      !publication.contains("schema_version") || !publication.contains("owner_identity") ||
      !publication.contains("block") || !publication.contains("field") ||
      !publication.contains("output_keys") || !publication.contains("gradient_sign"))
    throw py::value_error("AMR field output publication must be one exact schema-v1 record");
  const py::handle schema = publication["schema_version"];
  const py::handle sign = publication["gradient_sign"];
  if (!PyLong_CheckExact(schema.ptr()) || py::cast<std::uint32_t>(schema) != 1)
    throw py::value_error("AMR field output publication schema_version must be exactly 1");
  if (!PyLong_CheckExact(sign.ptr()))
    throw py::type_error("AMR field output publication gradient_sign must be an exact int");
  const py::handle rows = publication["output_keys"];
  if (!PyList_CheckExact(rows.ptr()) && !PyTuple_CheckExact(rows.ptr()))
    throw py::type_error("AMR field output publication output_keys must be an exact sequence");

  PreparedAmrFieldOutputPublication result;
  result.owner_identity = exact_publication_string(publication["owner_identity"], "owner_identity");
  result.block = exact_publication_string(publication["block"], "block");
  result.field = exact_publication_string(publication["field"], "field");
  result.gradient_sign = py::cast<int>(sign);
  const py::sequence output_rows = py::reinterpret_borrow<py::sequence>(rows);
  result.output_keys.reserve(py::len(output_rows));
  for (const py::handle item : output_rows) {
    if (!PyDict_CheckExact(item.ptr()))
      throw py::type_error("AMR field output publication keys must be exact dicts");
    const py::dict row = py::reinterpret_borrow<py::dict>(item);
    if (row.size() != 4 || !row.contains("owner_qid") || !row.contains("space_kind") ||
        !row.contains("space_name") || !row.contains("component"))
      throw py::value_error("AMR field output publication requires exact ComponentKey records");
    result.output_keys.push_back(
        {exact_publication_string(row["owner_qid"], "output_keys.owner_qid"),
         exact_publication_string(row["space_kind"], "output_keys.space_kind"),
         exact_publication_string(row["space_name"], "output_keys.space_name"),
         exact_publication_string(row["component"], "output_keys.component")});
  }
  return result;
}

void require_amr_cell_array_shape(const AmrSystem& system, const py::array& array,
                                  std::string_view operation) {
  const std::vector<py::ssize_t> expected = ranked_numpy_shape(system.spatial_shape());
  if (array.ndim() != pops::kNativeDimension)
    throw py::value_error(std::string(operation) +
                          ": cell-array rank differs from the native spatial dimension");
  for (int axis = 0; axis < pops::kNativeDimension; ++axis)
    if (array.shape(axis) != expected[static_cast<std::size_t>(axis)])
      throw py::value_error(std::string(operation) +
                            ": cell-array shape differs from the exact native spatial shape");
}

void require_amr_state_array_shape(const AmrSystem& system, const py::array& array,
                                   std::string_view operation) {
  const std::vector<py::ssize_t> expected = ranked_numpy_shape(system.spatial_shape());
  if (array.ndim() != pops::kNativeDimension + 1 || array.shape(0) < 1)
    throw py::value_error(std::string(operation) +
                          ": state rank must be one component axis plus the native dimension");
  for (int axis = 0; axis < pops::kNativeDimension; ++axis)
    if (array.shape(axis + 1) != expected[static_cast<std::size_t>(axis)])
      throw py::value_error(std::string(operation) +
                            ": state shape differs from the exact native spatial shape");
}

typename pops::runtime::amr::PreparedTaggingProgram<pops::kNativeDimension>::Stencil
amr_tagging_stencil_from_python(const py::dict& row) {
  using Program = pops::runtime::amr::PreparedTaggingProgram<pops::kNativeDimension>;
  Program::Stencil result;
  result.identity = py::cast<std::string>(row["identity"]);
  result.route = py::cast<std::string>(row["route"]);
  result.norm = py::cast<std::string>(row["norm"]);
  result.scale = py::cast<std::string>(row["scale"]);
  result.boundary_mode = py::cast<std::string>(row["boundary_mode"]);
  if (py::cast<std::int32_t>(row["dimension"]) != pops::kNativeDimension)
    throw std::invalid_argument(
        "AMR Tagger stencil dimension differs from the selected native specialization");
  const py::list axes = py::cast<py::list>(row["axes"]);
  if (axes.size() != pops::kNativeDimension)
    throw std::invalid_argument("AMR Tagger stencil has no exact native-rank axis image");
  std::size_t axis_ordinal = 0;
  for (const py::handle value : axes) {
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
    const std::int32_t axis_index = py::cast<std::int32_t>(axis["axis"]);
    if (axis_index != static_cast<std::int32_t>(axis_ordinal))
      throw std::invalid_argument("AMR Tagger stencil axes are not in canonical native order");
    result.axes[axis_ordinal++] =
        Program::AxisStencil{axis_index,
                             py::cast<std::int32_t>(axis["derivative_order"]),
                             py::cast<std::int32_t>(axis["formal_order"]),
                             py::cast<std::size_t>(axis["ghost_lower"]),
                             py::cast<std::size_t>(axis["ghost_upper"]),
                             py::cast<std::vector<std::int32_t>>(axis["offsets"]),
                             std::move(coefficients)};
  }
  return result;
}

void install_amr_interface_flux_provider(AmrSystem& system, const py::list& rows) {
  using Scheduler = pops::runtime::multiblock::InterfaceFluxScheduler<pops::kNativeDimension>;
  using PreparedComponent =
      pops::runtime::multiblock::PreparedInterfaceFluxComponent<pops::kNativeDimension>;
  using Job = pops::python::detail::PreparedInterfaceFluxJob<pops::kNativeDimension>;
  std::vector<Job> jobs;
  const std::string provider_contract =
      pops::python::detail::prepare_interface_flux_jobs<pops::kNativeDimension>(rows, jobs, false);
  system.install_prepared_amr_interface_flux_provider(
      provider_contract, [&system, jobs = std::move(jobs)](Scheduler& scheduler) mutable {
        for (Job& job : jobs) {
          if (job.route.left_block >= static_cast<std::size_t>(system.n_blocks()) ||
              job.route.right_block >= static_cast<std::size_t>(system.n_blocks()) ||
              job.route.level < 0 || job.route.level >= system.n_levels())
            throw std::out_of_range(
                "AMR shared-interface route lies outside the prepared block/level registry");
          auto& left = system.prepared_amr_block_state(static_cast<int>(job.route.left_block),
                                                       job.route.level);
          auto& right = system.prepared_amr_block_state(static_cast<int>(job.route.right_block),
                                                        job.route.level);
          const auto geometry = system.prepared_amr_level_geometry(job.route.level);
          const PopsExecutionContextV1 execution = job.spec.execution->view();
          scheduler.install(
              std::move(job.route), left, geometry, right, geometry, execution,
              [spec = std::move(job.spec), component = std::move(job.component)]() mutable {
                auto prepared =
                    std::make_shared<PreparedComponent>(std::move(spec), std::move(component));
                return pops::runtime::multiblock::InterfaceFluxEvaluator(
                    [prepared](const pops::runtime::multiblock::BoundaryEvaluationPoint& point,
                               const pops::runtime::multiblock::InterfaceFluxBatch& batch) {
                      prepared->evaluate(point, batch);
                    });
              });
        }
      });
}

// Assembly seams: per-block composition, native block, and refinement tagging.
void bind_amr_assembly(py::class_<AmrSystem>& cls) {
  cls.def(py::init<const AmrSystemConfig&>())
      // The lambda assembles the flat preparation controls into NewtonOptions before the C++ call.
      // Failure policy is intentionally absent: every local nonlinear failure is fail-closed.
      .def(
          "add_block",
          [](AmrSystem& s, const std::string& name, const ModelSpec& model,
             const std::string& limiter, const std::string& riemann, const std::string& recon,
             const std::string& time, int substeps, int stride,
             const std::vector<std::string>& implicit_vars,
             const std::vector<std::string>& implicit_roles, int newton_max_iters,
             double newton_rel_tol, double newton_abs_tol, double newton_fd_eps,
             double newton_damping, bool newton_diagnostics, double positivity_floor,
             double weno_epsilon, bool wave_speed_cache) {
            NewtonOptions newton;
            newton.max_iters = newton_max_iters;
            newton.rel_tol = static_cast<Real>(newton_rel_tol);
            newton.abs_tol = static_cast<Real>(newton_abs_tol);
            newton.fd_eps = static_cast<Real>(newton_fd_eps);
            newton.damping = static_cast<Real>(newton_damping);
            s.add_block(name, model, limiter, riemann, recon, time, substeps, stride, implicit_vars,
                        implicit_roles, newton, newton_diagnostics, positivity_floor, weno_epsilon,
                        wave_speed_cache);
          },
          py::arg("name"), py::arg("model"), py::arg("limiter") = "minmod",
          py::arg("riemann") = "rusanov", py::arg("recon") = "conservative",
          py::arg("time") = "explicit", py::arg("substeps") = 1, py::arg("stride") = 1,
          // Compatibility call shape shared with System.add_block. The AMR spatial runtime has no
          // typed local implicit-source Program primitive: every non-empty selector fails closed and
          // is never retained by the block. Empty selectors keep the bare "imex" authoring token as
          // metadata only; they do not enable a hidden backward-Euler step.
          py::arg("implicit_vars") = std::vector<std::string>{},
          py::arg("implicit_roles") = std::vector<std::string>{},
          // Every non-default Newton control and newton_diagnostics=true fails closed until a typed
          // AMR local nonlinear/Newton Program primitive owns both the solve and its report.
          py::arg("newton_max_iters") = kNewtonDefaultMaxIters,
          py::arg("newton_rel_tol") = static_cast<double>(kNewtonDefaultRelTol),
          py::arg("newton_abs_tol") = static_cast<double>(kNewtonDefaultAbsTol),
          py::arg("newton_fd_eps") = static_cast<double>(kNewtonDefaultFdEps),
          py::arg("newton_damping") = static_cast<double>(kNewtonDefaultDamping),
          py::arg("newton_diagnostics") = false,
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
             const std::vector<double>& face_values,
             const std::vector<std::string>& face_identities,
             const std::vector<std::string>& component_roles,
             const std::vector<int>& omitted_interface_faces, const std::string& state_identity,
             const std::vector<std::vector<int>>& periodic_identifications,
             const std::vector<std::string>& face_representations,
             const std::vector<std::string>& face_converter_identities,
             const std::vector<std::vector<std::string>>& face_analytic_opcodes,
             const std::vector<std::vector<double>>& face_analytic_literals,
             const std::vector<std::string>& face_analytic_clocks) {
            reject_unqualified_periodic_identifications<pops::kNativeDimension>(
                periodic_identifications, "ranked Cartesian AMR boundary authority");
            std::vector<bool> omitted(face_types.size(), false);
            for (int ordinal : omitted_interface_faces) {
              if (ordinal < 0 || static_cast<std::size_t>(ordinal) >= face_types.size() ||
                  face_types[static_cast<std::size_t>(ordinal)] != "external" ||
                  omitted[static_cast<std::size_t>(ordinal)])
                throw py::value_error(
                    "every omitted AMR interface face must be one unique external ranked face");
              omitted[static_cast<std::size_t>(ordinal)] = true;
            }
            system.install_hyperbolic_boundary(
                name, identity, required_depth, face_types, face_values, face_identities,
                component_roles, state_identity, face_representations, face_converter_identities,
                face_analytic_opcodes, face_analytic_literals, face_analytic_clocks);
          },
          py::arg("name"), py::arg("identity"), py::arg("required_depth"), py::arg("face_types"),
          py::arg("face_values"), py::arg("face_identities"), py::arg("component_roles"),
          py::arg("omitted_interface_faces") = std::vector<int>{},
          py::arg("state_identity") = std::string{},
          py::arg("periodic_identifications") = std::vector<std::vector<int>>{},
          py::arg("face_representations") = std::vector<std::string>{},
          py::arg("face_converter_identities") = std::vector<std::string>{},
          py::arg("face_analytic_opcodes") = std::vector<std::vector<std::string>>{},
          py::arg("face_analytic_literals") = std::vector<std::vector<double>>{},
          py::arg("face_analytic_clocks") = std::vector<std::string>{},
          "Install one resolved per-block ghost-production plan before lazy AMR construction.")
      .def("_install_block_state_route", &AmrSystem::install_block_state_route, py::arg("name"),
           py::arg("state_identity"),
           "Bind one exact state Handle identity to native AMR block storage.")
      .def("_install_field_storage_route", &AmrSystem::install_field_storage_route,
           py::arg("field_identity"), py::arg("provider_slot"),
           "Bind one exact solved-field Handle to native provider storage.")
      .def("has_package_assembly_lane", &AmrSystem::has_package_assembly_lane,
           "True when the RuntimeInstance package-assembly lane is already staged.")
      .def(
          "_prepare_boundary_execution_lane",
          [](AmrSystem& system, const py::object& communicator_authority,
             const py::dict& execution_data) {
            std::shared_ptr<const pops::component::PreparedExecutionContextV1> execution;
            std::shared_ptr<std::optional<pops::ExecutionLane>> lane_holder;
            std::string lane_identity;
            std::exception_ptr preparation_error;
            try {
              execution = pops::python::detail::make_component_execution_context(execution_data);
              const PopsExecutionContextV1 view = execution->view();
#ifdef POPS_HAS_MPI
              const auto& world = communicator_authority.cast<const pops::WorldCommunicator&>();
              if (&world != &pops::WorldCommunicator::world() ||
                  std::string_view(view.communicator_identity) != world.identity() ||
                  view.communicator_f_handle != world.fortran_handle() ||
                  view.communicator_datatype_f_handle != world.datatype_float64().fortran_handle())
                throw std::invalid_argument(
                    "AMR boundary execution requires the exact native process world");
#else
              if (!communicator_authority.is_none() || view.communicator_f_handle != 0 ||
                  view.communicator_datatype_f_handle != 0 ||
                  std::string_view(view.communicator_identity) != "serial")
                throw std::invalid_argument(
                    "serial AMR boundary execution requires the exact null authority");
#endif
              lane_identity = "pops.amr.package-assembly/" + execution->identity();
              lane_holder = std::make_shared<std::optional<pops::ExecutionLane>>();
            } catch (...) {
              preparation_error = std::current_exception();
            }
#ifdef POPS_HAS_MPI
            const pops::WorldCommunicator& world = pops::WorldCommunicator::world();
            if (pops::all_reduce_max(preparation_error ? 1L : 0L, world.communicator()) != 0) {
              if (world.size() == 1 && preparation_error)
                std::rethrow_exception(preparation_error);
              throw std::runtime_error(
                  "AMR boundary native execution authority failed collectively");
            }
#else
            if (preparation_error)
              std::rethrow_exception(preparation_error);
#endif
            lane_holder->emplace(pops::ExecutionLane::duplicate_world_collectively(lane_identity));
            std::shared_ptr<pops::ExecutionLane> lane(lane_holder, &lane_holder->value());
            std::shared_ptr<const pops::component::PreparedExecutionContextV1> lane_execution;
            std::exception_ptr lane_context_error;
            try {
              lane_execution = std::make_shared<const pops::component::PreparedExecutionContextV1>(
                  execution->for_lane(*lane));
            } catch (...) {
              lane_context_error = std::current_exception();
            }
            if (pops::all_reduce_max(lane_context_error ? 1L : 0L, lane->communicator()) != 0) {
              if (lane->size() == 1 && lane_context_error)
                std::rethrow_exception(lane_context_error);
              throw std::runtime_error(
                  "AMR package-lane execution context preparation failed collectively");
            }
            system.install_prepared_boundary_execution_context(std::move(lane),
                                                               std::move(lane_execution));
          },
          py::arg("communicator_authority"), py::arg("execution_context"),
          "Retain the RuntimeInstance execution authority for the prepared AMR hierarchy lane.")
      .def(
          "_preflight_ghost_boundary_component",
          [](AmrSystem&, const std::shared_ptr<pops::component::LoadedComponent>& component,
             const py::dict& row, const std::string& parameters_json,
             const std::string& target_json, const py::dict& execution_data) {
            auto spec = pops::python::detail::boundary_component_spec_from_python(
                row, parameters_json, target_json, execution_data);
            (void)std::make_shared<pops::PreparedGhostBoundaryComponent>(std::move(spec),
                                                                         component);
          },
          py::arg("component"), py::arg("row"), py::arg("parameters_json"), py::arg("target_json"),
          py::arg("execution_context"))
      .def(
          "_install_ghost_boundary_component",
          [](AmrSystem& system, const std::string& block,
             std::shared_ptr<pops::component::LoadedComponent> component, const py::dict& row,
             const std::string& parameters_json, const std::string& target_json,
             const py::dict& execution_data) {
            auto spec = pops::python::detail::boundary_component_spec_from_python(
                row, parameters_json, target_json, execution_data);
            system.stage_prepared_ghost_boundary_component(
                block, std::make_shared<pops::PreparedGhostBoundaryComponent>(
                           std::move(spec), std::move(component)));
          },
          py::arg("block"), py::arg("component"), py::arg("row"), py::arg("parameters_json"),
          py::arg("target_json"), py::arg("execution_context"))
      .def(
          "_install_boundary_flux_component",
          [](AmrSystem& system, const std::string& block,
             std::shared_ptr<pops::component::LoadedComponent> component, const py::dict& row,
             const std::string& parameters_json, const std::string& target_json,
             const py::dict& execution_data) {
            auto spec = pops::python::detail::boundary_component_spec_from_python(
                row, parameters_json, target_json, execution_data);
            system.stage_prepared_boundary_flux_component(
                block, std::make_shared<pops::PreparedBoundaryFluxComponent>(std::move(spec),
                                                                             std::move(component)));
          },
          py::arg("block"), py::arg("component"), py::arg("row"), py::arg("parameters_json"),
          py::arg("target_json"), py::arg("execution_context"))
      .def(
          "_preflight_boundary_flux_component",
          [](AmrSystem&, const std::shared_ptr<pops::component::LoadedComponent>& component,
             const py::dict& row, const std::string& parameters_json,
             const std::string& target_json, const py::dict& execution_data) {
            auto spec = pops::python::detail::boundary_component_spec_from_python(
                row, parameters_json, target_json, execution_data);
            (void)std::make_shared<pops::PreparedBoundaryFluxComponent>(std::move(spec), component);
          },
          py::arg("component"), py::arg("row"), py::arg("parameters_json"), py::arg("target_json"),
          py::arg("execution_context"))
      .def(
          "_preflight_field_boundary_residual_component",
          [](AmrSystem&, const std::shared_ptr<pops::component::LoadedComponent>& component,
             const py::dict& row, const std::string& parameters_json,
             const std::string& target_json, const py::dict& execution_data) {
            auto spec = pops::python::detail::boundary_component_spec_from_python(
                row, parameters_json, target_json, execution_data);
            (void)std::make_shared<pops::PreparedFieldBoundaryResidualComponent>(std::move(spec),
                                                                                 component);
          },
          py::arg("component"), py::arg("row"), py::arg("parameters_json"), py::arg("target_json"),
          py::arg("execution_context"))
      .def(
          "_preflight_field_boundary_jvp_component",
          [](AmrSystem&, const std::shared_ptr<pops::component::LoadedComponent>& component,
             const py::dict& row, const std::string& parameters_json,
             const std::string& target_json, const py::dict& execution_data) {
            auto spec = pops::python::detail::boundary_component_spec_from_python(
                row, parameters_json, target_json, execution_data);
            (void)std::make_shared<pops::PreparedFieldBoundaryJvpComponent>(std::move(spec),
                                                                            component);
          },
          py::arg("component"), py::arg("row"), py::arg("parameters_json"), py::arg("target_json"),
          py::arg("execution_context"))
      .def(
          "_install_field_boundary_component_pair",
          [](AmrSystem& system, const std::string& block,
             std::shared_ptr<pops::component::LoadedComponent> residual_component,
             const py::dict& residual_row,
             std::shared_ptr<pops::component::LoadedComponent> jvp_component,
             const py::dict& jvp_row, const std::string& parameters_json,
             const std::string& target_json, const py::dict& execution_data) {
            auto residual_spec = pops::python::detail::boundary_component_spec_from_python(
                residual_row, parameters_json, target_json, execution_data);
            auto jvp_spec = pops::python::detail::boundary_component_spec_from_python(
                jvp_row, parameters_json, target_json, execution_data);
            system.stage_prepared_field_boundary_component_pair(
                block,
                std::make_shared<pops::PreparedFieldBoundaryResidualComponent>(
                    std::move(residual_spec), std::move(residual_component)),
                std::make_shared<pops::PreparedFieldBoundaryJvpComponent>(
                    std::move(jvp_spec), std::move(jvp_component)));
          },
          py::arg("block"), py::arg("residual_component"), py::arg("residual_row"),
          py::arg("jvp_component"), py::arg("jvp_row"), py::arg("parameters_json"),
          py::arg("target_json"), py::arg("execution_context"))
      .def("_install_interface_flux_provider", &install_amr_interface_flux_provider,
           py::arg("jobs"),
           "Atomically extend the prepared multi-level shared-interface provider registry.")
      .def("_discard_boundary_plans", &AmrSystem::discard_hyperbolic_boundaries,
           "Roll back one failed pre-block boundary authority transaction.")
      .def(
          "_install_amr_tagger_component",
          [](AmrSystem& system, std::shared_ptr<pops::component::LoadedComponent> component,
             const py::dict& binding, const std::string& parameters_json,
             const std::string& target_json, const py::dict& execution_data) {
            const py::dict capability = py::cast<py::dict>(binding["tagging_capability"]);
            system.install_tagger_component(
                std::move(component), py::cast<std::string>(binding["component_id"]),
                py::cast<std::string>(binding["component_manifest_identity"]),
                py::cast<std::uint32_t>(binding["interface_version"]),
                py::cast<std::string>(binding["provider_identity"]),
                py::cast<std::string>(binding["tagging_graph_identity"]),
                py::cast<std::string>(binding["layout_identity"]),
                py::cast<std::string>(binding["clock_identity"]),
                py::cast<std::string>(capability["execution_mode"]), parameters_json, target_json,
                pops::python::detail::make_component_execution_context(execution_data));
          },
          py::arg("component"), py::arg("binding"), py::arg("parameters_json"),
          py::arg("target_json"), py::arg("execution_context"),
          "Install one authenticated native Tagger candidate evaluator.")
      // Private production-package seam. Parameters are fixed before AMR closures are built.
      .def("_install_native_block", &AmrSystem::add_native_block, py::arg("name"),
           py::arg("so_path"), py::arg("expected_model_identity"),
           py::arg("expected_binary_identity"), py::arg("limiter") = "minmod",
           py::arg("riemann") = "rusanov", py::arg("recon") = "conservative",
           py::arg("time") = "explicit",
           py::arg("gamma") = static_cast<double>(kPhysicalDefaultGamma), py::arg("substeps") = 1,
           py::arg("stride") = 1, py::arg("params") = std::vector<double>{},
           // Zhang-Shu positivity floor (ADC-322): marshaled down the regenerated .so loader
           // (pops_install_native_amr). 0 (default) = inactive, bit-identical.
           py::arg("positivity_floor") = 0.0,
           py::arg("weno_epsilon") = static_cast<double>(kWenoEpsilon),
           py::arg("wave_speed_cache") = false)
      .def("_register_external_riemann_package", &AmrSystem::register_external_riemann_package,
           py::arg("name"), py::arg("so_path"), py::arg("brick_id"), py::arg("expected_sha256"),
           py::arg("expected_nvars"), py::arg("expected_provider_count"),
           py::arg("expected_model_identity"), py::arg("provider_consumer_qid"),
           py::arg("limiter") = "minmod", py::arg("recon") = "conservative",
           py::arg("time") = "explicit",
           py::arg("gamma") = static_cast<double>(kPhysicalDefaultGamma), py::arg("substeps") = 1,
           py::arg("stride") = 1, py::arg("positivity_floor") = 0.0,
           py::arg("weno_epsilon") = static_cast<double>(kWenoEpsilon))
      .def(
          "_set_bootstrap_tagging",
          [](AmrSystem& system, const std::vector<std::string>& leaf_subject_kinds,
             const std::vector<std::string>& leaf_subject_identities,
             const std::vector<std::string>& leaf_blocks,
             const std::vector<std::string>& leaf_variables,
             const std::vector<int>& leaf_field_component_indices, const std::vector<int>& leaf_ops,
             const std::vector<double>& leaf_thresholds,
             const std::vector<int>& leaf_stencil_indices, const py::list& stencil_rows,
             const std::vector<std::int32_t>& refine_ops,
             const std::vector<std::int32_t>& refine_args,
             const std::vector<std::int32_t>& coarsen_ops,
             const std::vector<std::int32_t>& coarsen_args, int min_cycles,
             const std::string& equality_policy, const std::string& conflict_policy,
             const std::string& clock_identity, const std::string& provider_identity) {
            std::vector<typename pops::runtime::amr::PreparedTaggingProgram<
                pops::kNativeDimension>::Stencil>
                stencils;
            stencils.reserve(stencil_rows.size());
            for (const py::handle row : stencil_rows)
              stencils.push_back(amr_tagging_stencil_from_python(py::cast<py::dict>(row)));
            system.set_bootstrap_tagging(
                leaf_subject_kinds, leaf_subject_identities, leaf_blocks, leaf_variables,
                leaf_field_component_indices, leaf_ops, leaf_thresholds, leaf_stencil_indices,
                stencils, refine_ops, refine_args, coarsen_ops, coarsen_args, min_cycles,
                equality_policy, conflict_policy, clock_identity, provider_identity);
          },
          py::arg("leaf_subject_kinds"), py::arg("leaf_subject_identities"), py::arg("leaf_blocks"),
          py::arg("leaf_variables"), py::arg("leaf_field_component_indices"), py::arg("leaf_ops"),
          py::arg("leaf_thresholds"), py::arg("leaf_stencil_indices"), py::arg("stencils"),
          py::arg("refine_ops"), py::arg("refine_args"), py::arg("coarsen_ops"),
          py::arg("coarsen_args"), py::arg("min_cycles"), py::arg("equality_policy"),
          py::arg("conflict_policy"), py::arg("clock_identity"), py::arg("provider_identity"))
      .def(
          "set_poisson",
          [](AmrSystem& system, const std::string& rhs, const std::string& solver,
             const std::string& bc) { system.set_poisson(rhs, solver, bc); },
          "Configures the default AMR field through the registered native provider. The Python "
          "shortcut selects provider defaults; resolved provider-specific options are installed by "
          "the compiled field-plan pipeline.",
          py::arg("rhs") = "charge_density", py::arg("solver") = "geometric_mg",
          py::arg("bc") = "auto")
      .def(
          "set_field_solver_plan",
          [](AmrSystem& system, const std::string& provider_slot, const std::string& plan_identity,
             const std::string& provider_identity, const py::dict& output_publication,
             const std::vector<std::string>& provider_identities,
             const std::vector<std::string>& provider_blocks,
             const std::vector<std::string>& provider_keys,
             const std::vector<double>& provider_coefficients, const std::string& solver,
             const std::string& hierarchy_policy_id,
             std::uint64_t hierarchy_policy_interface_version,
             const std::string& hierarchy_policy_option_schema,
             const py::dict& hierarchy_policy_options, const std::string& schema_identity,
             const py::dict& options) {
            const PreparedAmrFieldOutputPublication publication =
                prepared_amr_field_output_publication_from_python(output_publication);
            const AmrFieldHierarchyPolicyAuthority hierarchy_policy{
                hierarchy_policy_id,
                hierarchy_policy_interface_version,
                prepared_provider_options_from_python(hierarchy_policy_option_schema,
                                                      hierarchy_policy_options),
            };
            system.set_field_solver_plan(
                provider_slot, plan_identity, provider_identity, publication.owner_identity,
                publication.block, publication.field, publication.output_keys,
                publication.gradient_sign, provider_identities, provider_blocks, provider_keys,
                provider_coefficients, solver, hierarchy_policy,
                prepared_provider_options_from_python(schema_identity, options));
          },
          py::arg("provider_slot"), py::arg("plan_identity"), py::arg("provider_identity"),
          py::arg("output_publication"), py::arg("provider_identities"), py::arg("provider_blocks"),
          py::arg("provider_keys"), py::arg("provider_coefficients"), py::arg("solver"),
          py::arg("hierarchy_policy_id"), py::arg("hierarchy_policy_interface_version"),
          py::arg("hierarchy_policy_option_schema"), py::arg("hierarchy_policy_options"),
          py::arg("schema_identity"), py::arg("options"))
      .def(
          "register_field_solver_provider",
          [](AmrSystem& system, const std::string& provider_slot,
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
      .def(
          "register_elliptic_field",
          [](AmrSystem& system, const std::string& block, const std::string& field,
             const py::sequence& output_keys, int gradient_sign) {
            std::vector<runtime::system::AuxiliaryComponentKey> keys;
            keys.reserve(py::len(output_keys));
            for (const py::handle row : output_keys) {
              const py::dict key = py::cast<py::dict>(row);
              keys.push_back({py::cast<std::string>(key["owner_qid"]),
                              py::cast<std::string>(key["space_kind"]),
                              py::cast<std::string>(key["space_name"]),
                              py::cast<std::string>(key["component"])});
            }
            system.register_elliptic_field(block, field, keys, gradient_sign);
          },
          py::arg("block"), py::arg("field"), py::arg("output_keys"), py::arg("gradient_sign"))
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
      // Runtime-private lowering seam for the normalized analytic LevelSet.  The exact-ranked AMR
      // hierarchy owns compilation, collective validation, and per-level rematerialization; no
      // Python callback reaches a cell kernel.
      .def("_set_analytic_level_set", &AmrSystem::set_analytic_level_set, py::arg("opcodes"),
           py::arg("literals"), py::arg("mode") = "none", py::arg("kappa_min") = 0.0,
           py::arg("face_open_eps") = 0.0, py::arg("cut_theta_min") = 0.0)
      .def("set_geometry_mode", &AmrSystem::set_geometry_mode, py::arg("mode"))
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
      // Owner-qualified InputAux only.  The native hierarchy owns every per-level storage group,
      // transfer, derived launch and halo phase; Python never selects a raw carrier component.
      .def(
          "stage_auxiliary_input",
          [](AmrSystem& s, const std::string& owner_qid, const std::string& space_kind,
             const std::string& space_name, const std::string& component,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            require_amr_cell_array_shape(s, arr, "AmrSystem.stage_auxiliary_input");
            s.stage_auxiliary_input({owner_qid, space_kind, space_name, component}, flat(arr));
          },
          py::arg("owner_qid"), py::arg("space_kind"), py::arg("space_name"), py::arg("component"),
          py::arg("values"))
      .def(
          "auxiliary_component",
          [](const AmrSystem& s, const std::string& owner_qid, const std::string& space_kind,
             const std::string& space_name, const std::string& component) {
            return to_ranked_field(
                s.auxiliary_component({owner_qid, space_kind, space_name, component}),
                s.spatial_shape());
          },
          py::arg("owner_qid"), py::arg("space_kind"), py::arg("space_name"), py::arg("component"))
      .def(
          "auxiliary_address",
          [](const AmrSystem& s, const std::string& owner_qid, const std::string& space_kind,
             const std::string& space_name, const std::string& component) {
            const auto address =
                s.auxiliary_address({owner_qid, space_kind, space_name, component});
            return py::make_tuple(address.group, address.component);
          },
          py::arg("owner_qid"), py::arg("space_kind"), py::arg("space_name"), py::arg("component"))
      .def(
          "auxiliary_registry_contract",
          [](const AmrSystem& s) {
            const std::string contract = s.auxiliary_registry_contract();
            return py::bytes(contract.data(), contract.size());
          },
          "Exact auxiliary-registry contract bytes for rollback comparison.")
      .def("capture_auxiliary_checkpoint_accepted_state",
           [](const AmrSystem& s) {
             py::list result;
             for (const auto& state : s.capture_auxiliary_checkpoint_accepted_state()) {
               const auto bytes =
                   pops::runtime::system::serialize_auxiliary_checkpoint_state(state);
               result.append(py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
             }
             return result;
           })
      .def("checkpoint_rank_local_carrier_manifest",
           &AmrSystem::checkpoint_rank_local_carrier_manifest,
           "Collective-free exact rank-local carrier and auxiliary-registry rollback witness.")
      .def("dirty_auxiliary_provider_identities", &AmrSystem::dirty_auxiliary_provider_identities,
           "Exact pending auxiliary-provider identities retained by rollback.")
      .def("_checkpoint_auxiliary_level_capacity", &AmrSystem::checkpoint_auxiliary_level_capacity,
           "Return the sealed AMR per-level auxiliary metadata/scalar checkpoint capacity.")
      .def(
          "_checkpoint_interface_flux_capacity",
          [](const AmrSystem& s) {
            const auto budget = s.prepared_amr_interface_flux_ledger_budget();
            return py::make_tuple(budget.max_fragments_per_window,
                                  budget.max_payload_terms_per_window,
                                  budget.exact_contract.size());
          },
          "Return the artifact-authenticated accepted interface-ledger capacity.")
      .def(
          "_checkpoint_program_flux_capacity",
          [](const AmrSystem& s) {
            const auto& budget = s.prepared_amr_program_flux_expression_budget();
            std::size_t rhs = 0;
            std::size_t coefficients = 0;
            for (const auto& block : budget.blocks) {
              if (block.rhs_basis_bound > std::numeric_limits<std::size_t>::max() - rhs ||
                  block.coefficient_term_bound >
                      std::numeric_limits<std::size_t>::max() - coefficients)
                throw std::overflow_error("AMR Program flux checkpoint capacity overflows size_t");
              rhs += block.rhs_basis_bound;
              coefficients += block.coefficient_term_bound;
            }
            return py::make_tuple(rhs, coefficients, budget.interface_coupling_application_bound,
                                  budget.interface_coupling_identity_character_bound,
                                  budget.exact_contract.size());
          },
          "Return the artifact-authenticated Program face/interface flux capacity.")
      .def("_checkpoint_program_state_capacity", &AmrSystem::checkpoint_program_state_capacity,
           "Return the artifact-authenticated POPSAND4/source-authority byte capacities.")
      .def(
          "restore_restart_auxiliary_checkpoint_accepted_state",
          [](AmrSystem& s, py::object payloads) {
            py::object retained_payload;
            s.restore_restart_auxiliary_checkpoint_accepted_state_bytes(
                [&]() { return static_cast<std::size_t>(py::len(payloads)); },
                [&](std::size_t level) -> std::span<const std::uint8_t> {
                  retained_payload = payloads.attr("__getitem__")(py::int_(level));
                  if (!PyBytes_CheckExact(retained_payload.ptr()))
                    throw py::type_error(
                        "AMR exact auxiliary checkpoint payloads must be bytes objects");
                  char* data = nullptr;
                  Py_ssize_t size = 0;
                  if (PyBytes_AsStringAndSize(retained_payload.ptr(), &data, &size) != 0)
                    throw py::error_already_set();
                  return {reinterpret_cast<const std::uint8_t*>(data),
                          static_cast<std::size_t>(size)};
                });
          },
          py::arg("payloads"))
      .def(
          "set_density",
          [](AmrSystem& s, const std::string& name,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            require_amr_cell_array_shape(s, arr, "AmrSystem.set_density");
            s.set_density(name, flat(arr));
          },
          py::arg("name"), py::arg("rho"),
          "Set a block's coarse density from one exact ranked cell array.")
      // Full initial conservative state keeps one leading component axis. flat() flattens
      // any C-contiguous array, so a density-only array passed by mistake could become a
      // 1-component state (comp 0 = density, momentum left at 0) -- a silent density masquerade
      // with the wrong physics. flat() then flattens in component-major c*cells + j*nx + i.
      .def(
          "set_conservative_state",
          [](AmrSystem& s, const std::string& name,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            require_amr_state_array_shape(s, arr, "AmrSystem.set_conservative_state");
            s.set_conservative_state(name, flat(arr));
          },
          py::arg("name"), py::arg("U"))
      // Generic InitialConditionPlan staging.  The resolved subject must be bound to its
      // authenticated state route before either canonical analytic programs or exact-rank arrays
      // are accepted; there are intentionally no source-specific bootstrap entry points.
      .def("_bind_bootstrap_subject", &AmrSystem::bind_bootstrap_subject, py::arg("subject_id"),
           py::arg("runtime_block"), py::arg("source_route"))
      .def("_stage_bootstrap_analytic_state", &AmrSystem::stage_bootstrap_analytic_state,
           py::arg("subject_id"), py::arg("runtime_block"), py::arg("space"), py::arg("centering"),
           py::arg("projection"), py::arg("opcodes"), py::arg("literals"))
      .def(
          "_stage_bootstrap_array",
          [](AmrSystem& s, const std::string& subject_id, const std::string& runtime_block,
             const std::string& space, const std::string& centering,
             py::array_t<double, py::array::c_style | py::array::forcecast> values) {
            require_amr_state_array_shape(s, values, "AmrSystem.stage_bootstrap_array");
            Extent<pops::kNativeDimension> spatial_shape{};
            for (int axis = 0; axis < pops::kNativeDimension; ++axis)
              spatial_shape[axis] = values.shape(pops::kNativeDimension - axis);
            s.stage_bootstrap_array(subject_id, runtime_block, space, centering,
                                    static_cast<int>(values.shape(0)), spatial_shape, flat(values));
          },
          py::arg("subject_id"), py::arg("runtime_block"), py::arg("space"), py::arg("centering"),
          py::arg("values"))
      .def("_materialize_bootstrap_action", &AmrSystem::materialize_bootstrap_action,
           py::arg("subject_id"), py::arg("action"), py::arg("action_route"), py::arg("level"))
      .def("_recompute_bootstrap_field", &AmrSystem::recompute_bootstrap_field,
           py::arg("subject_id"), py::arg("provider_slot"))
      .def("_synchronize_bootstrap_state", &AmrSystem::synchronize_bootstrap_state,
           py::arg("subject_id"), py::arg("fine_level"))
      .def("_begin_bootstrap_plan", &AmrSystem::begin_bootstrap_plan)
      .def("_bootstrap_next_level", &AmrSystem::bootstrap_next_level)
      .def("_commit_bootstrap_level", &AmrSystem::commit_bootstrap_level)
      .def("_rollback_bootstrap_level", &AmrSystem::rollback_bootstrap_level)
      .def(
          "_register_bootstrap_transfer_route",
          [](AmrSystem& system, const std::string& identity,
             const std::vector<std::string>& subjects, const std::string& provider_identity,
             const std::string& space, const std::string& centering,
             const std::string& representation, const std::string& storage,
             const std::string& operation, const std::string& kernel, int order,
             const py::handle& ghost_depth, const py::handle& refinement_ratio) {
            system.register_bootstrap_transfer_route(
                identity, subjects, provider_identity, space, centering, representation, storage,
                operation, kernel, order,
                ranked_extent_from_python<pops::kNativeDimension>(
                    ghost_depth, "AmrSystem bootstrap transfer ghost_depth", true),
                ranked_extent_from_python<pops::kNativeDimension>(
                    refinement_ratio, "AmrSystem bootstrap transfer refinement_ratio", false));
          },
          py::arg("identity"), py::arg("subjects"), py::arg("provider_identity"), py::arg("space"),
          py::arg("centering"), py::arg("representation"), py::arg("storage"), py::arg("operation"),
          py::arg("kernel"), py::arg("order"), py::arg("ghost_depth"), py::arg("refinement_ratio"))
      .def("_register_bootstrap_oriented_face_subjects",
           &AmrSystem::register_bootstrap_oriented_face_subjects, py::arg("oriented_subjects"));
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
  cls.def("time", &AmrSystem::time)
      // AMR clock (IO v1, System parity): macro-step counter + restoration (t, macro_step) ->
      // the regrid/stride cadence resumes exactly after a set_clock. Prerequisite PR-IO-3.
      .def("macro_step", &AmrSystem::macro_step)
      .def("set_clock", &AmrSystem::set_clock, py::arg("t"), py::arg("macro_step"))
      .def("field_provider_slots", &AmrSystem::field_provider_slots)
      .def("checkpoint_phi_provider_slot", &AmrSystem::checkpoint_phi_provider_slot)
      .def("field_provider_checkpoint_manifest", &AmrSystem::field_provider_checkpoint_manifest,
           "Collective-free immutable manifest for every exact AMR field provider.")
      .def("field_provider_levels", &AmrSystem::field_provider_levels, py::arg("provider_slot"))
      .def("restore_field_potentials", &AmrSystem::restore_field_potentials,
           py::arg("provider_slots"), py::arg("potentials"),
           "Atomically restore one complete all-provider field warm-start image.")
      .def("recompute_fields_after_restart_regrid",
           &AmrSystem::recompute_fields_after_restart_regrid,
           "Recompute typed derived fields after a restart regrid and return its witness.")
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
      // Exact partially accumulated GLOBAL stride window (strict checkpoint/restart state).
      .def("program_cadence_window_dt", &AmrSystem::program_cadence_window_dt)
      .def("program_cadence_window_steps", &AmrSystem::program_cadence_window_steps)
      .def("program_cadence_window_start_time", &AmrSystem::program_cadence_window_start_time)
      // Accepted Program interval provenance used by strict history replay validation.
      .def("program_last_dt", &AmrSystem::program_last_dt)
      .def("restore_program_cadence_window", &AmrSystem::restore_program_cadence_window,
           py::arg("accumulated_dt"), py::arg("held_steps"), py::arg("window_start_time"),
           py::arg("accepted_last_dt"), py::arg("accepted_time"), py::arg("macro_step"))
      // Changes the RUNTIME parameters of a compiled time PROGRAM block WITHOUT recompiling the .so
      // (ADC-508, parity ADC-510). prog_block = the PROGRAM block index (P.state order); values = that
      // block's params in sorted-name order. Python's _install_program_params routes params={name: value}
      // here. cf. AmrSystem::set_program_params.
      .def("set_program_params", &AmrSystem::set_program_params, py::arg("prog_block"),
           py::arg("values"))
      // IR hash of the installed compiled Program (the .so's pops_program_hash), or "" if none. Parity
      // System::installed_program_hash (the checkpoint guard).
      .def("installed_program_hash", &AmrSystem::installed_program_hash)
      // Exact Program-index -> AMR-block-index map established by name at install.  Expose only
      // immutable report metadata, never a structural mutation route.
      .def("program_block_map", &AmrSystem::program_block_map)
      .def(
          "program_param_count",
          [](const AmrSystem& system, int program_block) {
            return system.program_params(program_block).count;
          },
          py::arg("program_block"))
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
          "restore_checkpoint_accepted_state",
          [](AmrSystem& s, py::bytes payload) {
            std::string bytes = payload;
            s.restore_checkpoint_accepted_state(
                std::vector<std::uint8_t>(bytes.begin(), bytes.end()));
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
      .def("program_temporal_partition_manifest", &AmrSystem::program_temporal_partition_manifest)
      .def("program_flux_ledger_manifest", &AmrSystem::program_flux_ledger_manifest)
      .def("program_interface_flux_ledger_manifest",
           &AmrSystem::program_interface_flux_ledger_manifest)
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
      .def("_accepted_balance_terms", &AmrSystem::accepted_balance_terms, py::arg("route"))
      .def("_selected_accepted_balance_terms", &AmrSystem::selected_accepted_balance_terms,
           py::arg("route"), py::arg("block"), py::arg("component"), py::arg("levels"),
           py::arg("automatic_terms"))
      .def("_consume_step_projections", &AmrSystem::consume_step_projections)
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
      .def("variable_names", &AmrSystem::variable_names,
           "Installed variable names of one authenticated AMR block. kind = 'conservative' | "
           "'primitive'.",
           py::arg("name"), py::arg("kind") = "conservative")
      .def(
          "effective_options_report",
          [](const AmrSystem& s) {
            return effective_options_report_to_dict(s.effective_options_report());
          },
          "Structured effective numerical/solver/physical options for this AmrSystem.")
      .def("n_patches", &AmrSystem::n_patches)
      // Exact-ranked index-space footprints of the fine patches. Each item carries the level and
      // inclusive lower/upper Index<Dim> corners in that level's native axis order. SAME
      // source as n_patches() (the GLOBAL fine BoxArray) -> rank-independent, MPI-safe. Query between
      // steps, zero cost on the hot path. The Python wrapper converts with the exact physical
      // bounds; cf. AmrSystem.patch_bounds() on the facade side.
      .def("patch_boxes",
           [](AmrSystem& s) { return ranked_amr_patches_to_python(s.patch_boxes()); })
      .def("spatial_shape",
           [](const AmrSystem& s) { return ranked_extent_to_python(s.spatial_shape()); })
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
      .def("density", [](AmrSystem& s) { return to_ranked_field(s.density(), s.spatial_shape()); })
      .def(
          "density",
          [](AmrSystem& s, const std::string& name) {
            return to_ranked_field(s.density(name), s.spatial_shape());
          },
          py::arg("name"))
      // phi of the coarse (base) level in its exact ranked shape. SAME observable as
      // System.potential(): level 0
      // covers the whole domain -> enough to sample a median circle (azimuthal FFT). In
      // multi-block, phi results from the SYSTEM Poisson (Sum_b q_b n_b co-located), shared by all.
      .def("potential",
           [](AmrSystem& s) { return to_ranked_field(s.potential(), s.spatial_shape()); })
      // ADC-428: solved potential of a NAMED elliptic field (m.elliptic_field) on the coarse level,
      // exact-ranked shape. Read-back counterpart of potential() for a second elliptic field; the Python
      // AmrSystem.field(name) resolves the field name to this. Solves the hierarchy if needed.
      .def(
          "named_field_values",
          [](AmrSystem& s, const std::string& field) {
            return to_ranked_field(s.named_field_values(field), s.spatial_shape());
          },
          py::arg("field"))
      // AMR CHECKPOINT / RESTART single-rank (ADC-65): full conservative state per level + phi
      // (warm-start) + imposition of the saved fine hierarchy. SERIAL MONO-BLOCK (multi-block: C++
      // rejection; np>1: facade rejection -- per-level gather = future). level_state / level_potential return
      // FLAT fields (c*nf*nf + j*nf + i / nf*nf, nf = nx << k); the facade reshapes. set_*
      // flatten any C-contiguous array (flat). set_hierarchy consumes the exact-ranked patch
      // records returned by patch_boxes() (the coupler filters level 1).
      .def("n_levels", &AmrSystem::n_levels)
      .def("max_levels", &AmrSystem::max_levels)
      .def("configured_n_levels", &AmrSystem::configured_n_levels)
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
          [](AmrSystem& s, const py::handle& boxes) {
            s.set_hierarchy(ranked_amr_patches_from_python<pops::kNativeDimension>(
                boxes, "AmrSystem.set_hierarchy"));
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
          [](AmrSystem& s, const ObserverMpiLane& lane, const std::string& provider_slot,
             int level) {
            std::vector<OutputPiece<pops::kNativeDimension>> pieces;
            {
              py::gil_scoped_release release;
              pieces = s.output_field_root_pieces(lane, provider_slot, level);
            }
            return output_pieces_to_python(pieces);
          },
          py::arg("lane"), py::arg("provider_slot"), py::arg("level"),
          "Collectively gather compact field pieces in C++; complete only on MPI rank zero.")
      .def(
          "_output_geometry_snapshot",
          [](AmrSystem& s, int level, const std::array<double, pops::kNativeDimension>& origin,
             const std::array<double, pops::kNativeDimension>& spacing,
             const std::array<std::int64_t, pops::kNativeDimension>& cell_shape,
             const std::array<int, pops::kNativeDimension>& next_refinement_ratio,
             const std::string& cell_measure) {
            if (level < 0 || level >= s.n_levels())
              throw std::out_of_range("AmrSystem output geometry level is out of range");
            std::vector<pops::python::detail::OutputGeometryPatch<pops::kNativeDimension>> patches;
            const std::vector<AmrPatch<pops::kNativeDimension>> native_boxes =
                s.output_geometry_boxes();
            patches.reserve(native_boxes.size());
            for (const AmrPatch<pops::kNativeDimension>& patch : native_boxes)
              patches.push_back({patch.level, patch.box});
            return pops::python::detail::native_output_geometry_snapshot<pops::kNativeDimension>(
                level, s.checkpoint_topology_epoch(), origin, spacing, cell_shape, cell_measure,
                patches, next_refinement_ratio, true);
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
          [](AmrSystem& s, const ObserverMpiLane& lane, const std::string& name, int level) {
            std::vector<OutputPiece<pops::kNativeDimension>> pieces;
            {
              py::gil_scoped_release release;
              pieces = s.output_state_root_pieces(lane, name, level);
            }
            return output_pieces_to_python(pieces);
          },
          py::arg("lane"), py::arg("block"), py::arg("level"),
          "Collectively gather compact state pieces in C++; complete only on MPI rank zero.")
      .def(
          "output_embedded_boundary_local_pieces",
          [](AmrSystem& s, const std::string& name, int level) {
            return output_pieces_to_python(s.output_embedded_boundary_local_pieces(name, level));
          },
          py::arg("name"), py::arg("level"),
          "Exact-ranked prepared AMR embedded-boundary sidecar pieces owned by this rank.")
      .def(
          "output_embedded_boundary_root_pieces",
          [](AmrSystem& s, const ObserverMpiLane& lane, const std::string& name, int level) {
            std::vector<OutputPiece<pops::kNativeDimension>> pieces;
            {
              py::gil_scoped_release release;
              pieces = s.output_embedded_boundary_root_pieces(lane, name, level);
            }
            return output_pieces_to_python(pieces);
          },
          py::arg("lane"), py::arg("name"), py::arg("level"),
          "Collectively gather exact-ranked AMR embedded-boundary sidecars; complete only on MPI "
          "rank zero.")
      .def(
          "set_block_level_state",
          [](AmrSystem& s, const std::string& name, int k,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.set_block_level_state(name, k, flat(arr));
          },
          py::arg("name"), py::arg("k"), py::arg("state"))
      // ADC-542: owner rank per box of a level (the shared ranked ownership), for the v3 checkpoint
      // to reproduce the local-fab iteration order at restart. Empty on the single-block coupler path.
      .def(
          "level_owner_ranks", [](AmrSystem& s, int k) { return s.level_owner_ranks(k); },
          py::arg("k"))
      .def(
          "level_distribution_mode",
          [](const AmrSystem& s, int k) { return s.level_distribution_mode(k); }, py::arg("k"))
      // ADC-542: impose a mid-run MULTI-BLOCK hierarchy from a v3 checkpoint. @p boxes are the
      // level-tagged exact-ranked patch signatures; @p owner_ranks is the per-box owner
      // rank aligned with @p boxes. Routes to AmrRuntime::rebuild_hierarchy (all levels rebuilt).
      .def(
          "rebuild_hierarchy",
          [](AmrSystem& s, const py::handle& boxes, const std::vector<int>& owner_ranks) {
            s.rebuild_hierarchy(ranked_amr_patches_from_python<pops::kNativeDimension>(
                                    boxes, "AmrSystem.rebuild_hierarchy"),
                                owner_ranks);
          },
          py::arg("boxes"), py::arg("owner_ranks"))
      .def(
          "rematerialize_hierarchy_ownership",
          [](AmrSystem& s, const py::handle& boxes, const std::vector<std::string>& level_modes) {
            const std::vector<AmrPatch<pops::kNativeDimension>> bx =
                ranked_amr_patches_from_python<pops::kNativeDimension>(
                    boxes, "AmrSystem.rematerialize_hierarchy_ownership");
            py::gil_scoped_release release;
            return s.rematerialize_hierarchy_ownership(bx, level_modes);
          },
          py::arg("boxes"), py::arg("level_modes"),
          "Collectively rematerialize exact recorded patch ownership for this communicator.")
      .def(
          "program_accepted_state_source_authority",
          [](const AmrSystem& s, const std::vector<std::vector<int>>& source_level_owners,
             const std::vector<std::string>& source_level_modes, int source_rank_count) {
            const auto authority = s.program_accepted_state_source_authority(
                source_level_owners, source_level_modes, source_rank_count);
            return py::bytes(reinterpret_cast<const char*>(authority.data()), authority.size());
          },
          py::arg("source_level_owners"), py::arg("source_level_modes"),
          py::arg("source_rank_count"),
          "Seal the live accepted Program image under its artifact and source ownership.")
      .def(
          "rematerialize_program_accepted_state",
          [](AmrSystem& s, const py::bytes& source_state, int source_rank_count,
             const std::vector<std::vector<int>>& source_level_owners,
             const std::vector<std::vector<int>>& target_level_owners,
             const std::vector<std::string>& source_level_modes,
             const std::vector<std::string>& target_level_modes,
             const py::bytes& source_authority) {
            const std::string state = source_state;
            const std::string authority = source_authority;
            const auto rematerialized = s.rematerialize_program_accepted_state(
                std::vector<std::uint8_t>(state.begin(), state.end()), source_rank_count,
                source_level_owners, target_level_owners, source_level_modes, target_level_modes,
                std::vector<std::uint8_t>(authority.begin(), authority.end()));
            if (rematerialized.empty())
              return py::bytes();
            return py::bytes(reinterpret_cast<const char*>(rematerialized.data()),
                             rematerialized.size());
          },
          py::arg("source_state"), py::arg("source_rank_count"), py::arg("source_level_owners"),
          py::arg("target_level_owners"), py::arg("source_level_modes"),
          py::arg("target_level_modes"), py::arg("source_authority"),
          "Rematerialize one canonical Program image under current ownership.")
      .def(
          "_prepare_checkpoint_spatial_contract",
          [](const AmrSystem&, const py::dict& data) {
            return pops::python::detail::prepare_checkpoint_spatial_contract<kNativeDimension>(
                data);
          },
          py::arg("contract"),
          "Validate the exact rank-generic checkpoint schema before restart state work.")
      .def("begin_restart_transaction", &AmrSystem::begin_restart_transaction)
      .def("commit_restart_transaction", &AmrSystem::commit_restart_transaction)
      .def("finalize_restart_transaction", &AmrSystem::finalize_restart_transaction)
      .def("rollback_restart_transaction", &AmrSystem::rollback_restart_transaction)
      .def("preflight_regrid_on_restart", &AmrSystem::preflight_regrid_on_restart)
      .def("regrid_on_restart", &AmrSystem::regrid_on_restart)
      .def("checkpoint_regrid_count", &AmrSystem::checkpoint_regrid_count)
      .def("checkpoint_topology_epoch", &AmrSystem::checkpoint_topology_epoch)
      .def("restore_checkpoint_counters", &AmrSystem::restore_checkpoint_counters,
           py::arg("regrid_count"), py::arg("topology_epoch"))
      .def("checkpoint_temporal_relations", &AmrSystem::checkpoint_temporal_relations)
      .def("set_temporal_relations", &AmrSystem::set_temporal_relations, py::arg("numerators"),
           py::arg("denominators"), py::arg("remainder_policies"))
      .def("checkpoint_transfer_routes", &AmrSystem::checkpoint_transfer_routes)
      // ADC-631 multistep history rings on the compiled-Program AMR route.  Values and accepted
      // provenance are qualified by level; _system_io_history.py retains the Uniform signatures
      // only for the non-AMR facade.
      .def("history_names", &AmrSystem::history_names)
      .def("history_levels", &AmrSystem::history_levels, py::arg("name"))
      .def("history_depth", &AmrSystem::history_depth, py::arg("name"))
      .def("history_ncomp", &AmrSystem::history_ncomp, py::arg("name"))
      .def(
          "history_global",
          [](const AmrSystem& s, const std::string& name, int level, int slot) {
            return s.history_global(name, level, slot);
          },
          py::arg("name"), py::arg("level"), py::arg("slot"))
      .def("history_initialized", &AmrSystem::history_initialized, py::arg("name"),
           py::arg("level"))
      .def("history_fill_count", &AmrSystem::history_fill_count, py::arg("name"), py::arg("level"))
      .def(
          "restore_history",
          [](AmrSystem& s, const std::string& name, int level, int slot,
             py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            s.restore_history(name, level, slot, flat(arr));
          },
          py::arg("name"), py::arg("level"), py::arg("slot"), py::arg("values"))
      .def("set_history_initialized", &AmrSystem::set_history_initialized, py::arg("name"),
           py::arg("level"), py::arg("initialized"))
      .def("restore_history_fill_count", &AmrSystem::restore_history_fill_count, py::arg("name"),
           py::arg("level"), py::arg("fill_count"))
      .def("restore_history_metadata", &AmrSystem::restore_history_metadata, py::arg("name"),
           py::arg("level"), py::arg("initialized"), py::arg("fill_count"))
      .def(
          "restore_history_provenance",
          [](AmrSystem& s, const std::string& name, int level,
             py::array_t<double, py::array::c_style | py::array::forcecast> slot_dt,
             bool initialized, int fill_count) {
            s.restore_history_provenance(name, level, flat(slot_dt), initialized, fill_count);
          },
          py::arg("name"), py::arg("level"), py::arg("slot_dt"), py::arg("initialized"),
          py::arg("fill_count"))
      .def("history_slot_dt", &AmrSystem::history_slot_dt, py::arg("name"), py::arg("level"),
           py::arg("slot"))
      .def("restore_history_slot_dt", &AmrSystem::restore_history_slot_dt, py::arg("name"),
           py::arg("level"), py::arg("slot"), py::arg("dt"))
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
  using NativeAmrSystem = pops::AmrSystem<pops::kNativeDimension>;
  using NativeAmrSystemConfig = pops::AmrSystemConfig<pops::kNativeDimension>;
  py::class_<NativeAmrSystemConfig>(m, "AmrSystemConfig")
      .def(py::init<>())
      .def_property(
          "shape",
          [](const NativeAmrSystemConfig& config) { return ranked_extent_to_python(config.shape); },
          [](NativeAmrSystemConfig& config, const py::handle& value) {
            config.shape =
                ranked_extent_from_python<kNativeDimension>(value, "AmrSystemConfig.shape");
          })
      .def_property(
          "lower",
          [](const NativeAmrSystemConfig& config) {
            return ranked_real_vector_to_python(config.lower);
          },
          [](NativeAmrSystemConfig& config, const py::handle& value) {
            config.lower =
                ranked_real_vector_from_python<kNativeDimension>(value, "AmrSystemConfig.lower");
          })
      .def_property(
          "upper",
          [](const NativeAmrSystemConfig& config) {
            return ranked_real_vector_to_python(config.upper);
          },
          [](NativeAmrSystemConfig& config, const py::handle& value) {
            config.upper =
                ranked_real_vector_from_python<kNativeDimension>(value, "AmrSystemConfig.upper");
          })
      .def_property(
          "boxes",
          [](const NativeAmrSystemConfig& config) { return ranked_boxes_to_python(config.boxes); },
          [](NativeAmrSystemConfig& config, const py::handle& value) {
            config.boxes =
                ranked_boxes_from_python<kNativeDimension>(value, "AmrSystemConfig.boxes");
          })
      .def_readwrite("coordinate_system", &NativeAmrSystemConfig::coordinate_system)
      .def_readwrite("regrid_every", &NativeAmrSystemConfig::regrid_every)
      .def_readwrite("level_count", &NativeAmrSystemConfig::level_count)
      .def_property(
          "transition_ratios",
          [](const NativeAmrSystemConfig& config) {
            return ranked_extents_to_python(config.transition_ratios);
          },
          [](NativeAmrSystemConfig& config, const py::handle& value) {
            auto ratios = ranked_extents_from_python<kNativeDimension>(
                value, "AmrSystemConfig.transition_ratios", 1);
            for (const auto& ratio : ratios) {
              bool refines_at_least_one_axis = false;
              for (int axis = 0; axis < kNativeDimension; ++axis)
                refines_at_least_one_axis = refines_at_least_one_axis || ratio[axis] > 1;
              if (!refines_at_least_one_axis)
                throw py::value_error(
                    "AmrSystemConfig.transition_ratios: every level transition must refine at "
                    "least one axis");
            }
            config.transition_ratios = std::move(ratios);
          })
      .def_property(
          "transition_buffers",
          [](const NativeAmrSystemConfig& config) {
            return ranked_extents_to_python(config.transition_buffers);
          },
          [](NativeAmrSystemConfig& config, const py::handle& value) {
            config.transition_buffers = ranked_extents_from_python<kNativeDimension>(
                value, "AmrSystemConfig.transition_buffers", 0);
          })
      .def_property(
          "transition_lookaheads",
          [](const NativeAmrSystemConfig& config) {
            return ranked_extents_to_python(config.transition_lookaheads);
          },
          [](NativeAmrSystemConfig& config, const py::handle& value) {
            config.transition_lookaheads = ranked_extents_from_python<kNativeDimension>(
                value, "AmrSystemConfig.transition_lookaheads", 0);
          })
      .def_readwrite("explicit_bootstrap", &NativeAmrSystemConfig::explicit_bootstrap)
      .def_property(
          "periodicity",
          [](const NativeAmrSystemConfig& config) {
            return ranked_periodicity_to_python<pops::kNativeDimension>(config.periodicity);
          },
          [](NativeAmrSystemConfig& config, const py::handle& value) {
            config.periodicity =
                ranked_periodicity_from_python<kNativeDimension>(value, "AmrSystemConfig");
          })
      .def_readwrite("distribute_coarse", &NativeAmrSystemConfig::distribute_coarse)
      .def_property(
          "coarse_max_grid",
          [](const NativeAmrSystemConfig& config) {
            return ranked_extent_to_python(config.coarse_max_grid);
          },
          [](NativeAmrSystemConfig& config, const py::handle& value) {
            config.coarse_max_grid = ranked_extent_from_python<kNativeDimension>(
                value, "AmrSystemConfig.coarse_max_grid", true);
          })
      // ADC-616: Berger-Rigoutsos clustering params (<= 0 = the historical {0.7, 1, 32} default).
      .def_readwrite("cluster_min_efficiency", &NativeAmrSystemConfig::cluster_min_efficiency)
      .def_readwrite("cluster_min_box_size", &NativeAmrSystemConfig::cluster_min_box_size)
      .def_readwrite("cluster_max_box_size", &NativeAmrSystemConfig::cluster_max_box_size)
      .def(
          "_set_load_balance_provider",
          [](NativeAmrSystemConfig& config, const std::string& route,
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
          py::arg("route"), py::arg("semantic_identity"), py::arg("option_schema_identity"),
          py::arg("options"));

  // AmrSystem: generic single-species composition on AMR.
  py::class_<NativeAmrSystem> cls(m, "AmrSystem");
  bind_amr_assembly(cls);
  bind_amr_physics(cls);
  bind_amr_stepping(cls);
  bind_amr_program(cls);
  bind_amr_data(cls);
}
