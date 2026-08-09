/// @file
/// @brief Exact-ranked generated-package preparation seam for Uniform System blocks.

#pragma once

#include <pops/core/model/physical_model.hpp>
#include <pops/mesh/topology/boundary_topology.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/system.hpp>

#include <cmath>
#include <concepts>
#include <string>
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
  const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage = nullptr;
  const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* provider_plan = nullptr;
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
    System<Dim>& system, const std::string& name, Model model, const std::string& limiter,
    const std::string& riemann, const std::string& reconstruction, const std::string& time,
    double gamma, int substeps, bool evolve, int stride, double positivity_floor = 0.0) {
  static_assert(Dim >= 1 && Dim <= 3);
  static_assert(requires { Model::dimension; },
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
  compiled_system_detail::validate_routes(routes);
  using Request = CompiledSystemBlockPreparation<Dim, Model>;
  static_assert(requires(Request request) {
    {
      prepare_exact_system_block(std::move(request))
    } -> std::same_as<PreparedSystemBlock<Dim>>;
  }, "generated model package lacks prepare_exact_system_block for its exact native dimension");

  const auto* provider_storage = system.prepared_block_provider_storage_groups();
  const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* provider_plan = nullptr;
  if constexpr (provider_count_for<Model, Dim>() > 0)
    provider_plan = &system.prepared_auxiliary_consumer_plan(name);
  PreparedSystemBlock<Dim> prepared = compiled_system_detail::invoke_package_preparer(
      Request{name, std::move(model), std::move(routes), system.prepared_block_geometry(),
              BoundaryTopology<Dim>::axis_periodic(system.prepared_block_periodicity()),
              provider_storage, provider_plan});

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

/// Publish one already prepared block through the facade's atomic structural seam.
template <int Dim>
void install_prepared_block(System<Dim>& system, PreparedSystemBlock<Dim> prepared) {
  system.install_prepared_block(std::move(prepared));
}

/// Convenience composition used by generated native loaders.
template <int Dim, class Model>
void add_compiled_model(System<Dim>& system, const std::string& name, Model model,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& reconstruction = "conservative",
                        const std::string& time = "explicit",
                        double gamma = static_cast<double>(kPhysicalDefaultGamma),
                        int substeps = 1, bool evolve = true, int stride = 1,
                        double positivity_floor = 0.0) {
  install_prepared_block(
      system, prepare_compiled_system_block<Dim>(
                  system, name, std::move(model), limiter, riemann, reconstruction, time, gamma,
                  substeps, evolve, stride, positivity_floor));
}

}  // namespace pops
