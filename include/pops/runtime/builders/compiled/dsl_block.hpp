/// @file
/// @brief Exact-ranked generated-package preparation seam for Uniform System blocks.

#pragma once

#include <pops/core/model/physical_model.hpp>
#include <pops/mesh/topology/boundary_topology.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/system.hpp>
#include <pops/runtime/system/native_package_capability.hpp>

#include <cmath>
#include <concepts>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace pops {

/// Authored spatial/time route tokens retained while the generated package prepares its concrete
/// operator.  These values never select a runtime dimension: `Dim` is already a template argument
/// of the request and of every field captured by the resulting block.
struct CompiledSystemBlockRoutes {
  std::string limiter;
  std::string riemann;
  std::string reconstruction;
  std::string time;
  Real positivity_floor = Real(0);
};

/// Complete immutable input handed to a package-owned exact block factory through ADL.
///
/// A generated translation unit defines
/// `prepare_exact_system_block(CompiledSystemBlockPreparation<Dim, ProdModel>)` in the model's
/// namespace.  That function materializes all numerical operators and returns one
/// `PreparedSystemBlock<Dim>`. The public helper below validates and commits that image; no legacy
/// mesh context or dynamic dimension tag crosses this boundary.
template <int Dim, class Model>
struct CompiledSystemBlockPreparation {
  static_assert(Dim >= 1 && Dim <= 3,
                "CompiledSystemBlockPreparation only supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;

  std::string name;
  Model model;
  CompiledSystemBlockRoutes routes;
  Geometry<Dim> geometry;
  BoundaryTopology<Dim> topology;
  std::shared_ptr<const runtime::system::AuxiliaryStorageGroups<Dim>> provider_storage;
  std::shared_ptr<const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>> provider_plan;
};

namespace compiled_system_detail {

template <class Request>
concept PackagePreparedSystemBlock = requires(Request request) {
  {
    prepare_exact_system_block(std::move(request))
  } -> std::same_as<PreparedSystemBlock<Request::dimension>>;
};

template <int Dim, class Model>
PreparedSystemBlock<Dim> invoke_package_preparer(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  // Deliberately unqualified: the only legal implementation lives beside the generated Model and
  // is found by ADL. There is no catalogue/runtime fallback in the PoPS core.
  return prepare_exact_system_block(std::move(request));
}

inline void validate_routes(const CompiledSystemBlockRoutes& routes) {
  if (routes.limiter.empty() || routes.riemann.empty() ||
      (routes.reconstruction != "conservative" && routes.reconstruction != "primitive"))
    throw std::invalid_argument(
        "compiled System block requires explicit limiter, Riemann and reconstruction routes");
  if (routes.time != "explicit" && routes.time != "euler" && routes.time != "ssprk3" &&
      routes.time != "imex" && routes.time != "imexrk_ars222")
    throw std::invalid_argument("compiled System block has an unsupported Program time route");
  if (!std::isfinite(routes.positivity_floor) || routes.positivity_floor < Real(0))
    throw std::invalid_argument(
        "compiled System block positivity floor must be finite and non-negative");
}

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_compiled_system_block_from_authority(
    const std::string& name, Model model, const std::string& limiter, const std::string& riemann,
    const std::string& reconstruction, const std::string& time, double gamma, int substeps,
    bool evolve, int stride, double positivity_floor, Geometry<Dim> geometry,
    std::array<bool, Dim> periodicity,
    std::shared_ptr<const runtime::system::AuxiliaryStorageGroups<Dim>> provider_storage,
    std::shared_ptr<const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>> provider_plan) {
  static_assert(Dim >= 1 && Dim <= 3);
  static_assert(
      requires { Model::dimension; },
      "a generated System model must publish its exact spatial dimension");
  static_assert(Model::dimension == Dim,
                "generated model dimension differs from the target System specialization");
  static_assert(requires {
    Model::n_vars;
    { Model::conservative_vars() } -> std::same_as<VariableSet>;
    { Model::primitive_vars() } -> std::same_as<VariableSet>;
  });

  if (name.empty())
    throw std::invalid_argument("compiled System block name must be non-empty");
  if (!std::isfinite(gamma) || !(gamma > 0.0))
    throw std::invalid_argument("compiled System block gamma must be finite and positive");
  if (substeps < 1 || stride < 1)
    throw std::invalid_argument("compiled System block substeps and stride must be positive");

  CompiledSystemBlockRoutes routes{limiter, riemann, reconstruction, time,
                                   static_cast<Real>(positivity_floor)};
  validate_routes(routes);
  using Request = CompiledSystemBlockPreparation<Dim, Model>;
  static_assert(
      requires(Request request) {
        {
          prepare_exact_system_block(std::move(request))
        } -> std::same_as<PreparedSystemBlock<Dim>>;
      }, "generated model package lacks prepare_exact_system_block for its exact native dimension");

  PreparedSystemBlock<Dim> prepared = invoke_package_preparer(
      Request{name, std::move(model), std::move(routes), std::move(geometry),
              BoundaryTopology<Dim>::axis_periodic(periodicity), std::move(provider_storage),
              std::move(provider_plan)});

  // Authoritative structural metadata comes from compile-time model facts and the resolved install
  // request, never from an independently mutable runtime description.
  prepared.name = name;
  prepared.ncomp = Model::n_vars;
  prepared.conservative_variables = Model::conservative_vars();
  prepared.primitive_variables = Model::primitive_vars();
  prepared.gamma = gamma;
  prepared.substeps = substeps;
  prepared.evolve = evolve;
  prepared.stride = stride;
  if (prepared.provider_components == 0)
    prepared.provider_components = provider_count_for<Model, Dim>();
  return prepared;
}

}  // namespace compiled_system_detail

/// Ask the generated model package to prepare one exact-ranked block image.
///
/// The generated preparer owns every route-specific instantiation. The core supplies only the
/// immutable ranked geometry, nullable accepted provider storage, and one sealed consumer plan.
/// A provider-free model carries an exact empty plan and no storage allocation. A provider that
/// does not support an authored route must reject it while preparing; there is no alternate 2D
/// builder.
template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_compiled_system_block(
    runtime::system::PreparedNativeBlockInstaller<Dim>& installer, const std::string& name,
    const std::string& provider_consumer_qid, Model model, const std::string& limiter,
    const std::string& riemann, const std::string& reconstruction, const std::string& time,
    double gamma, int substeps, bool evolve, int stride, double positivity_floor = 0.0) {
  std::shared_ptr<const runtime::system::AuxiliaryStorageGroups<Dim>> provider_storage;
  std::shared_ptr<const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>> provider_plan;
  if constexpr (provider_count_for<Model, Dim>() > 0) {
    provider_storage = installer.provider_storage();
    provider_plan = installer.consumer_plan(provider_consumer_qid);
    if (!provider_plan)
      throw std::logic_error("compiled System block requires a sealed auxiliary consumer plan");
  }
  return compiled_system_detail::prepare_compiled_system_block_from_authority<Dim>(
      name, std::move(model), limiter, riemann, reconstruction, time, gamma, substeps, evolve,
      stride, positivity_floor, installer.geometry(), installer.periodicity(),
      std::move(provider_storage), std::move(provider_plan));
}

/// Prepare through the direct facade surface retained for in-process and test-owned Systems.
/// This is an exact adapter to the same generated preparer used by package capabilities.
template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_compiled_system_block(
    System<Dim>& system, const std::string& name, Model model, const std::string& limiter,
    const std::string& riemann, const std::string& reconstruction, const std::string& time,
    double gamma, int substeps, bool evolve, int stride, double positivity_floor = 0.0) {
  std::shared_ptr<const runtime::system::AuxiliaryStorageGroups<Dim>> provider_storage;
  std::shared_ptr<const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>> provider_plan;
  if constexpr (provider_count_for<Model, Dim>() > 0) {
    provider_storage = system.prepared_block_provider_storage_owner();
    provider_plan = std::make_shared<runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>>(
        system.prepared_auxiliary_consumer_plan(name));
  }
  return compiled_system_detail::prepare_compiled_system_block_from_authority<Dim>(
      name, std::move(model), limiter, riemann, reconstruction, time, gamma, substeps, evolve,
      stride, positivity_floor, system.prepared_block_geometry(),
      system.prepared_block_periodicity(), std::move(provider_storage), std::move(provider_plan));
}

/// Stage the RuntimeInstance world lane Python bind installs when the C++ convenience path
/// skipped it. Callers that already installed a lane keep that lane.
template <int Dim>
void ensure_compiled_system_execution_lane(System<Dim>& system, const std::string& name) {
  try {
    (void)system.prepared_boundary_execution_lane();
  } catch (const std::logic_error&) {
    system.install_prepared_boundary_execution_lane(std::make_shared<ExecutionLane>(
        ExecutionLane::world("pops.runtime.package." + name + "/lane")));
  }
}

/// Stage the same default block/state identity Python add_equation installs when pops.bind is
/// skipped. Callers that already installed a unique route keep that identity.
template <int Dim>
void ensure_compiled_system_state_route(System<Dim>& system, const std::string& name) {
  try {
    system.install_block_state_route(name, "pops.runtime.package." + name + "/state");
  } catch (const std::runtime_error& error) {
    const std::string_view message = error.what();
    if (message.find("unique") == std::string_view::npos &&
        message.find("duplicate") == std::string_view::npos)
      throw;
  } catch (const std::logic_error& error) {
    const std::string_view message = error.what();
    if (message.find("must be installed before") == std::string_view::npos)
      throw;
  }
}

/// Publish one already prepared block through the facade's atomic structural seam.
template <int Dim>
void install_prepared_block(System<Dim>& system, PreparedSystemBlock<Dim> prepared) {
  ensure_compiled_system_execution_lane(system, prepared.name);
  ensure_compiled_system_state_route(system, prepared.name);
  system.install_prepared_block(std::move(prepared));
}

/// Convenience composition used by generated native loaders.
template <int Dim, class Model>
void add_compiled_model(runtime::system::PreparedNativeBlockInstaller<Dim>& installer,
                        const std::string& name, const std::string& provider_consumer_qid,
                        Model model, const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& reconstruction = "conservative",
                        const std::string& time = "explicit",
                        double gamma = static_cast<double>(kPhysicalDefaultGamma), int substeps = 1,
                        bool evolve = true, int stride = 1, double positivity_floor = 0.0) {
  runtime::system::PreparedNativeSystemPackage<Dim> package;
  package.consumer_qid = provider_consumer_qid;
  package.block = prepare_compiled_system_block<Dim>(
      installer, name, provider_consumer_qid, std::move(model), limiter, riemann, reconstruction,
      time, gamma, substeps, evolve, stride, positivity_floor);
  installer.commit(std::move(package));
}

/// Direct-facade composition retained for in-process generated models.
template <int Dim, class Model>
void add_compiled_model(System<Dim>& system, const std::string& name, Model model,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& reconstruction = "conservative",
                        const std::string& time = "explicit",
                        double gamma = static_cast<double>(kPhysicalDefaultGamma), int substeps = 1,
                        bool evolve = true, int stride = 1, double positivity_floor = 0.0) {
  ensure_compiled_system_execution_lane(system, name);
  ensure_compiled_system_state_route(system, name);
  install_prepared_block(
      system, prepare_compiled_system_block<Dim>(system, name, std::move(model), limiter, riemann,
                                                 reconstruction, time, gamma, substeps, evolve,
                                                 stride, positivity_floor));
}

}  // namespace pops
