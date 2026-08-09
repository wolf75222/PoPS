/// @file
/// @brief Atomic installation of exact-ranked Uniform System providers.

#include "system_impl.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/builders/compiled/native_loader.hpp>
#include <pops/runtime/named_field_output.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {
namespace {

template <int Dim>
std::size_t checked_cells(const Box<Dim>& box, const char* operation) {
  const std::int64_t cells = box.numPts();
  if (cells < 0 || static_cast<std::uint64_t>(cells) >
                       static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error(std::string(operation) + ": ranked box exceeds size_t");
  return static_cast<std::size_t>(cells);
}

template <int Dim>
Index<Dim> unflatten(const Box<Dim>& box, std::size_t linear) {
  Index<Dim> index{};
  const Extent<Dim> extent = box.extent();
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t axis_extent = static_cast<std::size_t>(extent[axis]);
    index[axis] = box.lo[axis] + static_cast<int>(linear % axis_extent);
    linear /= axis_extent;
  }
  return index;
}

template <int Dim>
std::size_t storage_offset(const Index<Dim>& index, const Box<Dim>& storage) {
  const Extent<Dim> extent = storage.extent();
  std::size_t offset = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    offset += static_cast<std::size_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(extent[axis]);
  }
  return offset;
}

/// Copy only valid cells. Ghost values are owned by prepared communication/boundary producers and
/// must be regenerated after storage width changes.
template <int Dim>
void copy_valid_components(const MultiFab<Dim>& source, MultiFab<Dim>& destination, int components,
                           const char* operation) {
  if (source.layout() != destination.layout() ||
      source.distribution() != destination.distribution() ||
      source.local_rank() != destination.local_rank() ||
      source.local_global_indices() != destination.local_global_indices() ||
      source.local_size() != destination.local_size() || components < 0 ||
      components > source.ncomp() || components > destination.ncomp())
    throw std::invalid_argument(std::string(operation) +
                                ": fields do not share one exact ranked layout");

  for (std::size_t local = 0; local < source.local_size(); ++local) {
    const Fab<Dim>& source_fab = source.fab(local);
    Fab<Dim>& destination_fab = destination.fab(local);
    if (source_fab.box() != destination_fab.box())
      throw std::invalid_argument(std::string(operation) + ": local valid boxes differ");
    auto source_host = source_fab.create_host_mirror();
    auto destination_host = destination_fab.create_host_mirror();
    source_fab.copy_to_host(source_host);
    destination_fab.copy_to_host(destination_host);
    const Box<Dim>& valid = source_fab.box();
    const Box<Dim>& source_storage = source_fab.grown_box();
    const Box<Dim>& destination_storage = destination_fab.grown_box();
    const std::size_t source_stride = checked_cells(source_storage, operation);
    const std::size_t destination_stride = checked_cells(destination_storage, operation);
    const std::size_t cells = checked_cells(valid, operation);
    for (int component = 0; component < components; ++component)
      for (std::size_t linear = 0; linear < cells; ++linear) {
        const Index<Dim> index = unflatten(valid, linear);
        destination_host(static_cast<std::size_t>(component) * destination_stride +
                         storage_offset(index, destination_storage)) =
            source_host(static_cast<std::size_t>(component) * source_stride +
                        storage_offset(index, source_storage));
      }
    destination_fab.copy_from_host(destination_host);
  }
}

void validate_variable_set(const VariableSet& variables, VariableKind expected, int ncomp,
                           const char* label) {
  if (variables.kind != expected || variables.size != ncomp ||
      variables.names.size() != static_cast<std::size_t>(ncomp) ||
      variables.roles.size() != static_cast<std::size_t>(ncomp) ||
      (!variables.user_roles.empty() &&
       variables.user_roles.size() != static_cast<std::size_t>(ncomp)))
    throw std::invalid_argument(std::string("prepared System block ") + label +
                                " variable metadata differs from its component count");
  std::set<std::string> names;
  for (const std::string& name : variables.names)
    if (name.empty() || !names.insert(name).second)
      throw std::invalid_argument(std::string("prepared System block ") + label +
                                  " variable names must be unique and non-empty");
}

template <int Dim>
void validate_prepared_block(const PreparedSystemBlock<Dim>& block) {
  if (block.name.empty() || block.provider_identity.empty())
    throw std::invalid_argument(
        "prepared System block requires non-empty block and provider identities");
  if (block.ncomp < 1 || block.aux_components < 1)
    throw std::invalid_argument(
        "prepared System block requires positive state and auxiliary component counts");
  if (!std::isfinite(block.gamma) || !(block.gamma > 0.0) || block.substeps < 1 || block.stride < 1)
    throw std::invalid_argument("prepared System block gamma, substeps, and stride are invalid");
  for (int axis = 0; axis < Dim; ++axis)
    if (block.ghosts[axis] < 1)
      throw std::invalid_argument(
          "prepared System block must declare a positive ghost extent on every native axis");
  validate_variable_set(block.conservative_variables, VariableKind::Conservative, block.ncomp,
                        "conservative");
  validate_variable_set(block.primitive_variables, VariableKind::Primitive, block.ncomp,
                        "primitive");

  const auto& closures = block.closures;
  if (!closures.rhs_into || !closures.rhs_flux_only || !closures.source_only ||
      !closures.rhs_at_point || !closures.rhs_flux_only_at_point || !closures.rhs_core_at_point ||
      !closures.rhs_flux_only_core_at_point || !closures.rhs_core_at_point_prepared ||
      !closures.rhs_flux_only_core_at_point_prepared ||
      !closures.prepare_generated_state_at_point ||
      !closures.prepare_generated_state_at_point_prepared || !block.maximum_speed ||
      !block.poisson_rhs || !block.primitive_to_conservative || !block.conservative_to_primitive ||
      !block.batch_conservative_to_primitive)
    throw std::invalid_argument(
        "prepared System block does not implement the complete exact-ranked execution contract");
}

template <int Dim, class Implementation>
std::optional<MultiFab<Dim>> widened_aux_candidate(Implementation& implementation,
                                                   int requested_components) {
  if (requested_components <= implementation.aux_ncomp_)
    return std::nullopt;
  MultiFab<Dim> candidate(implementation.ba, implementation.dm, implementation.local_rank,
                          requested_components, implementation.aux.ghosts());
  copy_valid_components(implementation.aux, candidate, implementation.aux_ncomp_,
                        "System auxiliary widening");
  return candidate;
}

}  // namespace

template <int Dim>
void System<Dim>::add_block(const std::string&, const ModelSpec&, const std::string&,
                            const std::string&, const std::string&, const std::string&, int, bool,
                            int, const std::vector<std::string>&, const std::vector<std::string>&,
                            const NewtonOptions&, bool, double, bool, double) {
  throw std::logic_error(
      "System::add_block(ModelSpec) was removed from the native core: resolve and compile one "
      "dimension-qualified provider, then install its PreparedSystemBlock<Dim>");
}

template <int Dim>
void System<Dim>::install_block_state_route(const std::string& name,
                                            const std::string& state_identity) {
  require_assembling(p_->lifecycle_, "install_block_state_route");
  if (std::any_of(p_->sp.begin(), p_->sp.end(),
                  [&](const typename Impl::Species& block) { return block.name == name; }))
    throw std::logic_error("System block state route must be installed before its prepared block");
  p_->boundary_registry_.install_state_route(name, state_identity);
}

template <int Dim>
void System<Dim>::install_field_storage_route(const std::string& field_identity,
                                              const std::string& provider_slot) {
  require_assembling(p_->lifecycle_, "install_field_storage_route");
  p_->boundary_registry_.install_field_storage_route(field_identity, provider_slot);
}

template <int Dim>
void System<Dim>::install_hyperbolic_boundary(
    const std::string& name, const std::string& identity, int required_depth,
    const std::vector<std::string>& face_types, const std::vector<double>& face_values,
    const std::vector<std::string>& face_identities,
    const std::vector<std::string>& component_roles, const std::string& state_identity,
    const std::vector<std::string>& face_representations,
    const std::vector<std::string>& face_converter_identities,
    const std::vector<std::vector<std::string>>& face_analytic_opcodes,
    const std::vector<std::vector<double>>& face_analytic_literals,
    const std::vector<std::string>& face_analytic_clocks) {
  require_assembling(p_->lifecycle_, "install_hyperbolic_boundary");
  if (std::any_of(p_->sp.begin(), p_->sp.end(),
                  [&](const typename Impl::Species& block) { return block.name == name; }))
    throw std::logic_error(
        "System hyperbolic boundary must be prepared before its block is committed");
  p_->boundary_registry_.install_boundary(
      name, identity, required_depth, face_types, face_values, face_identities, component_roles,
      state_identity, face_representations, face_converter_identities, face_analytic_opcodes,
      face_analytic_literals, face_analytic_clocks);
}

template <int Dim>
void System<Dim>::install_prepared_hyperbolic_boundary(
    const std::string& name, const std::string& identity, int required_depth,
    const std::string& state_identity, std::shared_ptr<const HyperbolicBoundary> boundary) {
  require_assembling(p_->lifecycle_, "install_prepared_hyperbolic_boundary");
  if (std::any_of(p_->sp.begin(), p_->sp.end(),
                  [&](const typename Impl::Species& block) { return block.name == name; }))
    throw std::logic_error(
        "System hyperbolic boundary must be prepared before its block is committed");
  p_->boundary_registry_.install_boundary(name, identity, required_depth, state_identity,
                                          std::move(boundary));
}

template <int Dim>
void System<Dim>::discard_hyperbolic_boundaries() {
  require_assembling(p_->lifecycle_, "discard_hyperbolic_boundaries");
  for (typename Impl::Species& block : p_->sp) {
    block.boundary.reset();
    block.state_identity.clear();
  }
  p_->boundary_registry_.discard_transaction();
}

template <int Dim>
void System<Dim>::install_interface_provider(SystemInterfaceProvider<Dim> provider) {
  require_assembling(p_->lifecycle_, "install_interface_provider");
  p_->blocks_.install_interface_provider(std::move(provider));
}

template <int Dim>
void System<Dim>::discard_interface_flux_components() {
  require_assembling(p_->lifecycle_, "discard_interface_flux_components");
  p_->blocks_.discard_interface_fluxes();
}

template <int Dim>
std::size_t System<Dim>::interface_evaluation_count(const std::string& identity, int level) const {
  if (identity.empty() || level < 0)
    throw std::invalid_argument(
        "System interface evaluation query requires an identity and non-negative level");
  return p_->blocks_.interface_evaluation_count(identity, level);
}

template <int Dim>
Geometry<Dim> System<Dim>::prepared_block_geometry() const {
  return p_->geom;
}

template <int Dim>
std::array<bool, Dim> System<Dim>::prepared_block_periodicity() const {
  return p_->periodicity;
}

template <int Dim>
void System<Dim>::install_prepared_block(PreparedSystemBlock<Dim> prepared) {
  require_assembling(p_->lifecycle_, "install_prepared_block");
  validate_prepared_block(prepared);
  if (std::any_of(p_->sp.begin(), p_->sp.end(),
                  [&](const typename Impl::Species& block) { return block.name == prepared.name; }))
    throw std::invalid_argument("System prepared block name is already installed");

  const auto route = p_->boundary_registry_.state_routes().find(prepared.name);
  if (route == p_->boundary_registry_.state_routes().end())
    throw std::runtime_error("System prepared block lacks its exact pre-installed state identity");
  const std::string state_identity = route->second;

  const auto* installed_boundary = p_->boundary_for(prepared.name);
  std::shared_ptr<const HyperbolicBoundary> boundary;
  if (installed_boundary != nullptr) {
    if (installed_boundary->state_identity != state_identity ||
        installed_boundary->authority->ncomp() != prepared.ncomp ||
        installed_boundary->authority->periodic_axes() != p_->periodicity)
      throw std::invalid_argument(
          "System prepared block boundary differs from its exact state/domain contract");
    for (int axis = 0; axis < Dim; ++axis)
      if (prepared.ghosts[axis] < installed_boundary->required_depth)
        throw std::invalid_argument(
            "System prepared block ghosts are narrower than its boundary requirement");
    boundary = std::make_shared<HyperbolicBoundary>(
        installed_boundary->authority->with_converted_fixed_states(
            prepared.primitive_to_conservative));
  }

  std::optional<MultiFab<Dim>> aux_candidate =
      widened_aux_candidate<Dim>(*p_, prepared.aux_components);

  typename Impl::Species candidate;
  candidate.name = prepared.name;
  candidate.U = MultiFab<Dim>(p_->ba, p_->dm, p_->local_rank, prepared.ncomp, prepared.ghosts);
  candidate.ncomp = prepared.ncomp;
  candidate.substeps = prepared.substeps;
  candidate.evolve = prepared.evolve;
  candidate.stride = prepared.stride;
  candidate.gamma = prepared.gamma;
  candidate.rhs_into = std::move(prepared.closures.rhs_into);
  candidate.max_speed = std::move(prepared.maximum_speed);
  candidate.add_poisson_rhs = std::move(prepared.poisson_rhs);
  candidate.cons_vars = std::move(prepared.conservative_variables);
  candidate.prim_vars = std::move(prepared.primitive_variables);
  candidate.prim_to_cons = std::move(prepared.primitive_to_conservative);
  candidate.cons_to_prim = std::move(prepared.conservative_to_primitive);
  candidate.batch_cons_to_prim = std::move(prepared.batch_conservative_to_primitive);
  candidate.source_frequency = std::move(prepared.source_frequency);
  candidate.stability_dt = std::move(prepared.stability_dt);
  candidate.hotspot = std::move(prepared.closures.hotspot);
  candidate.project = std::move(prepared.closures.project);
  candidate.project_masked = std::move(prepared.closures.project_masked);
  candidate.rhs_flux_only = std::move(prepared.closures.rhs_flux_only);
  candidate.source_only = std::move(prepared.closures.source_only);
  candidate.source_only_masked = std::move(prepared.closures.source_only_masked);
  candidate.staircase_residuals = std::move(prepared.closures.staircase);
  candidate.cutcell_residuals = std::move(prepared.closures.cut_cell);
  candidate.rhs_at_point = std::move(prepared.closures.rhs_at_point);
  candidate.rhs_flux_only_at_point = std::move(prepared.closures.rhs_flux_only_at_point);
  candidate.rhs_without_prepared_interfaces =
      std::move(prepared.closures.rhs_without_prepared_interfaces);
  candidate.rhs_flux_only_without_prepared_interfaces =
      std::move(prepared.closures.rhs_flux_only_without_prepared_interfaces);
  candidate.rhs_core_at_point = std::move(prepared.closures.rhs_core_at_point);
  candidate.rhs_flux_only_core_at_point = std::move(prepared.closures.rhs_flux_only_core_at_point);
  candidate.boundary_residual_at_point = std::move(prepared.closures.boundary_residual_at_point);
  candidate.boundary_jvp_at_point = std::move(prepared.closures.boundary_jvp_at_point);
  candidate.rhs_core_at_point_prepared = std::move(prepared.closures.rhs_core_at_point_prepared);
  candidate.rhs_flux_only_core_at_point_prepared =
      std::move(prepared.closures.rhs_flux_only_core_at_point_prepared);
  candidate.boundary_residual_at_point_prepared =
      std::move(prepared.closures.boundary_residual_at_point_prepared);
  candidate.boundary_jvp_at_point_prepared =
      std::move(prepared.closures.boundary_jvp_at_point_prepared);
  candidate.prepare_generated_state_at_point =
      std::move(prepared.closures.prepare_generated_state_at_point);
  candidate.prepare_generated_state_at_point_prepared =
      std::move(prepared.closures.prepare_generated_state_at_point_prepared);
  candidate.boundary = boundary;
  candidate.state_identity = state_identity;

  // Every operation above is preparatory. These moves are noexcept after the one vector growth.
  p_->sp.push_back(std::move(candidate));
  if (aux_candidate) {
    p_->aux = std::move(*aux_candidate);
    p_->aux_ncomp_ = prepared.aux_components;
  }
  if (installed_boundary != nullptr)
    p_->boundary_registry_.boundary(prepared.name).authority = std::move(boundary);
}

template <int Dim>
void System<Dim>::ensure_aux_width(int ncomp) {
  require_assembling(p_->lifecycle_, "ensure_aux_width");
  if (ncomp < 1)
    throw std::invalid_argument("System auxiliary width must be positive");
  std::optional<MultiFab<Dim>> candidate = widened_aux_candidate<Dim>(*p_, ncomp);
  if (!candidate)
    return;
  p_->aux = std::move(*candidate);
  p_->aux_ncomp_ = ncomp;
}

template <int Dim>
void System<Dim>::register_elliptic_field(const std::string& block, const std::string& field,
                                          const std::vector<int>& output_components,
                                          int gradient_sign) {
  require_assembling(p_->lifecycle_, "register_elliptic_field");
  if (field.empty())
    throw std::invalid_argument("System named elliptic field identity must be non-empty");
  (void)p_->find(block);
  runtime::field::NamedFieldOutput<Dim> output(output_components, gradient_sign);
  output.validate_width(p_->aux_ncomp_, "System");

  auto selected = p_->field_plans_.end();
  for (auto plan = p_->field_plans_.begin(); plan != p_->field_plans_.end(); ++plan) {
    if (plan->second.output_block != block || plan->second.output_key != field)
      continue;
    if (selected != p_->field_plans_.end())
      throw std::runtime_error(
          "System named elliptic output resolves to multiple qualified provider slots");
    selected = plan;
  }
  if (selected == p_->field_plans_.end())
    throw std::invalid_argument(
        "System named elliptic output has no resolved exact-ranked field plan");
  const std::string& provider_slot = selected->first;
  const typename Impl::FieldPlan& plan = selected->second;
  if (p_->named_fields_.contains(provider_slot))
    throw std::invalid_argument("System named elliptic field is already registered: " +
                                provider_slot);
  if ((!plan.boundary_state_blocks.empty() || !plan.boundary_field_blocks.empty()) &&
      !plan.boundary_kernel)
    throw std::logic_error(
        "System field boundary dependencies require one compiled exact-ranked kernel");
  const BoundaryTopology<Dim> topology = BoundaryTopology<Dim>::axis_periodic(p_->periodicity);
  elliptic::nd::CartesianBoundaryKind physical = elliptic::nd::CartesianBoundaryKind::dirichlet;
  if (p_->poisson_bc_ == "neumann")
    physical = elliptic::nd::CartesianBoundaryKind::neumann;
  else if (p_->poisson_bc_ != "auto" && p_->poisson_bc_ != "dirichlet" &&
           p_->poisson_bc_ != "periodic")
    throw std::invalid_argument("System Poisson boundary mode is unknown");
  auto operator_options =
      elliptic::nd::CartesianPoissonOptions<Dim>::from_topology(topology, physical);
  if (!plan.boundary_kind.empty()) {
    for (int axis = 0; axis < Dim; ++axis) {
      for (int side = 0; side < 2; ++side) {
        const std::size_t face = static_cast<std::size_t>(2 * axis + side);
        const std::string& kind = plan.boundary_kind[face];
        const bool periodic = kind == "periodic";
        if (periodic != p_->periodicity[static_cast<std::size_t>(axis)])
          throw std::invalid_argument(
              "System field boundary periodicity differs from its exact domain topology");
        if (kind == "periodic")
          operator_options.boundaries[face] = elliptic::nd::CartesianBoundaryKind::periodic;
        else if (kind == "dirichlet")
          operator_options.boundaries[face] = elliptic::nd::CartesianBoundaryKind::dirichlet;
        else if (kind == "neumann")
          operator_options.boundaries[face] = elliptic::nd::CartesianBoundaryKind::neumann;
        else if (kind == "mixed")
          operator_options.boundaries[face] = elliptic::nd::CartesianBoundaryKind::mixed;
        else
          throw std::logic_error("System exact-ranked field boundary kind is unsupported");
        operator_options.boundary_alpha[face] = static_cast<Real>(plan.boundary_alpha[face]);
        operator_options.boundary_beta[face] = static_cast<Real>(plan.boundary_beta[face]);
        operator_options.boundary_values[face] = static_cast<Real>(plan.boundary_value[face]);
      }
    }
  }
  std::unique_ptr<runtime::system::ExactFieldSolverBackend<Dim>> backend;
  const auto component = p_->component_field_solver_providers_.find(plan.backend_provider_route);
  const auto configured = p_->configured_field_solver_providers_.find(plan.backend_provider_route);
  if ((component == p_->component_field_solver_providers_.end()) ==
      (configured == p_->configured_field_solver_providers_.end()))
    throw std::runtime_error(
        "System field plan must select exactly one installed exact-ranked backend route");

  if (component != p_->component_field_solver_providers_.end()) {
    if (plan.boundary_kernel)
      throw std::logic_error(
          "external exact field components must own dynamic boundaries in their component ABI");
    if (plan.has_reaction)
      throw std::logic_error(
          "external field reaction must be carried by the component's exact operator contract");
    if (!plan.boundary_kind.empty() &&
        std::any_of(plan.boundary_kind.begin(), plan.boundary_kind.end(),
                    [](const std::string& kind) { return kind != "periodic"; }))
      throw std::logic_error(
          "external field components currently require a fully periodic exact topology");
    backend = std::make_unique<runtime::system::ComponentFieldSolverBackend<Dim>>(
        std::string(component->second->provider_identity()), p_->geom, p_->ba, p_->dm,
        p_->local_rank, topology, p_->periodicity, component->second);
  } else {
    if (plan.has_reaction)
      throw std::logic_error(
          "configured exact Cartesian field solver does not implement a reaction operator");
    operator_options.absolute_tolerance = static_cast<Real>(configured->second.absolute_tolerance);
    operator_options.relative_tolerance = static_cast<Real>(configured->second.relative_tolerance);
    operator_options.maximum_iterations = configured->second.maximum_iterations;
    backend = std::make_unique<runtime::system::CartesianCgFieldSolverBackend<Dim>>(
        p_->geom, p_->ba, p_->dm, p_->local_rank, topology, operator_options,
        configured->second.exact_identity);
  }

  auto prepared = std::make_shared<typename System<Dim>::Impl::exact_field_type>(
      provider_slot, block, output, p_->geom, p_->ba, p_->dm, p_->local_rank, std::move(backend),
      p_->sp.size());
  if (plan.boundary_kernel)
    prepared->install_boundary_kernel(*plan.boundary_kernel);
  if (plan.newton)
    prepared->install_newton(*plan.newton);
  const FieldNullspaceProviderSelection nullspace_selection{
      plan.nullspace_provider_identity.empty() ? p_->default_nullspace_provider_identity_
                                               : plan.nullspace_provider_identity,
      plan.nullspace_provider_identity.empty() ? p_->default_nullspace_options_
                                               : plan.nullspace_options};
  const std::string topology_identity = plan.topology_digest.empty()
                                            ? plan.plan_identity + ":uniform-topology"
                                            : plan.topology_digest;
  prepared->install_nullspace(
      p_->prepare_uniform_field_nullspace(plan.plan_identity, topology_identity,
                                          nullspace_selection, operator_options,
                                          prepared->accepted_potential(), plan.has_reaction),
      PreparedVectorDistribution<Dim>::distributed());
  p_->named_fields_.emplace(provider_slot, std::move(prepared));
}

template <int Dim>
void System<Dim>::set_block_elliptic_field(
    const std::string& block_name, const std::string& field,
    std::function<void(const MultiFab<Dim>&, MultiFab<Dim>&)> rhs) {
  require_assembling(p_->lifecycle_, "set_block_elliptic_field");
  if (field.empty() || !rhs)
    throw std::invalid_argument(
        "System named elliptic RHS requires a field identity and prepared closure");
  const int block = p_->index(block_name);
  auto selected = p_->field_plans_.end();
  Real coefficient = Real(0);
  for (auto plan = p_->field_plans_.begin(); plan != p_->field_plans_.end(); ++plan) {
    if (plan->second.output_key != field)
      continue;
    Real candidate = Real(0);
    bool contributes = false;
    for (const typename Impl::FieldProviderBinding& binding : plan->second.providers) {
      // ``field`` identifies the qualified output route selected above.  A source provider key is
      // deliberately independent (for example electron_charge -> electrostatic); generated block
      // installation supplies the one RHS closure owned by this block, so its exact coefficient is
      // resolved by block ownership rather than by equating input and output names.
      if (binding.block == block_name) {
        candidate += static_cast<Real>(binding.coefficient);
        contributes = true;
      }
    }
    if (!contributes)
      continue;
    if (selected != p_->field_plans_.end())
      throw std::runtime_error("System elliptic RHS resolves to multiple qualified provider slots");
    selected = plan;
    coefficient = candidate;
  }
  if (selected == p_->field_plans_.end())
    throw std::invalid_argument("System elliptic RHS has no resolved provider binding");
  const auto provider = p_->named_fields_.find(selected->first);
  if (provider == p_->named_fields_.end())
    throw std::invalid_argument("System named elliptic field is not registered: " +
                                selected->first);
  provider->second->add_rhs(static_cast<std::size_t>(block), std::move(rhs), coefficient);
}

template <int Dim>
void System<Dim>::add_dt_bound(const std::string& label, std::function<double()> function) {
  require_assembling(p_->lifecycle_, "add_dt_bound");
  if (label.empty() || !function)
    throw std::invalid_argument("System global dt bound requires a non-empty label and provider");
  p_->coupling_.dt_bounds.push_back({label, std::move(function)});
}

template <int Dim>
std::string System<Dim>::last_dt_bound() const {
  return p_->last_dt_reason_;
}

template <int Dim>
typename System<Dim>::SourceNewtonReport System<Dim>::newton_report(const std::string&) const {
  throw std::logic_error(
      "System Newton reports belong to an installed typed nonlinear Program provider");
}

template <int Dim>
void System<Dim>::add_native_block(const std::string& name, const std::string& so_path,
                                   const std::string& limiter, const std::string& riemann,
                                   const std::string& recon, const std::string& time, double gamma,
                                   int substeps, bool evolve, int stride,
                                   const std::vector<double>& params, double positivity_floor) {
  require_assembling(p_->lifecycle_, "add_native_block");
  native_loader::add_native_block<Dim>(this, name, so_path, limiter, riemann, recon, time, gamma,
                                       substeps, evolve, stride, params, positivity_floor);
}

template <int Dim>
void System<Dim>::add_coupled_source(const CoupledSourceProgram&, double, const std::string&) {
  throw std::logic_error(
      "System coupled-source bytecode is not an execution provider; install one prepared "
      "dimension-qualified coupling operator");
}

template <int Dim>
void System<Dim>::add_coupling_operator(const CouplingOperator&) {
  throw std::logic_error(
      "System CouplingOperator metadata is not executable; install its prepared exact-ranked "
      "operator and inspect view together");
}

template <int Dim>
void System<Dim>::install_prepared_coupling_operator(
    const std::string& label, CouplingOperatorView view,
    std::function<void(Real, const std::vector<MultiFab<Dim>*>&)> operation,
    double constant_frequency, std::function<Real()> maximum_frequency) {
  require_assembling(p_->lifecycle_, "install_prepared_coupling_operator");
  if (label.empty() || !operation || !std::isfinite(constant_frequency) || constant_frequency < 0.0)
    throw std::invalid_argument(
        "prepared System coupling requires an identity, executable provider, and finite frequency");
  if (!view.label.empty() && view.label != label)
    throw std::invalid_argument(
        "prepared System coupling inspect identity differs from its executable provider");
  view.label = label;
  view.frequency.constant_mu = constant_frequency;
  view.frequency.per_cell = static_cast<bool>(maximum_frequency);

  auto candidate = p_->coupling_;
  candidate.operators.push_back(std::move(operation));
  candidate.coupled_operators.push_back(std::move(view));
  if (constant_frequency > 0.0)
    candidate.coupled_freqs.push_back({label, constant_frequency});
  if (maximum_frequency)
    candidate.coupled_frequencies.push_back({label, std::move(maximum_frequency)});
  p_->coupling_ = std::move(candidate);
}

template <int Dim>
const std::vector<CouplingOperatorView>& System<Dim>::coupled_operators() const {
  return p_->coupling_.coupled_operators;
}

template void System<kNativeDimension>::add_block(const std::string&, const ModelSpec&,
                                                  const std::string&, const std::string&,
                                                  const std::string&, const std::string&, int, bool,
                                                  int, const std::vector<std::string>&,
                                                  const std::vector<std::string>&,
                                                  const NewtonOptions&, bool, double, bool, double);
template void System<kNativeDimension>::install_block_state_route(const std::string&,
                                                                  const std::string&);
template void System<kNativeDimension>::install_field_storage_route(const std::string&,
                                                                    const std::string&);
template void System<kNativeDimension>::install_hyperbolic_boundary(
    const std::string&, const std::string&, int, const std::vector<std::string>&,
    const std::vector<double>&, const std::vector<std::string>&, const std::vector<std::string>&,
    const std::string&, const std::vector<std::string>&, const std::vector<std::string>&,
    const std::vector<std::vector<std::string>>&, const std::vector<std::vector<double>>&,
    const std::vector<std::string>&);
template void System<kNativeDimension>::install_prepared_hyperbolic_boundary(
    const std::string&, const std::string&, int, const std::string&,
    std::shared_ptr<const System<kNativeDimension>::HyperbolicBoundary>);
template void System<kNativeDimension>::discard_hyperbolic_boundaries();
template void System<kNativeDimension>::install_interface_provider(
    SystemInterfaceProvider<kNativeDimension>);
template void System<kNativeDimension>::discard_interface_flux_components();
template std::size_t System<kNativeDimension>::interface_evaluation_count(const std::string&,
                                                                          int) const;
template Geometry<kNativeDimension> System<kNativeDimension>::prepared_block_geometry() const;
template std::array<bool, kNativeDimension> System<kNativeDimension>::prepared_block_periodicity()
    const;
template void System<kNativeDimension>::install_prepared_block(
    PreparedSystemBlock<kNativeDimension>);
template void System<kNativeDimension>::ensure_aux_width(int);
template void System<kNativeDimension>::register_elliptic_field(const std::string&,
                                                                const std::string&,
                                                                const std::vector<int>&, int);
template void System<kNativeDimension>::set_block_elliptic_field(
    const std::string&, const std::string&,
    std::function<void(const MultiFab<kNativeDimension>&, MultiFab<kNativeDimension>&)>);
template void System<kNativeDimension>::add_dt_bound(const std::string&, std::function<double()>);
template std::string System<kNativeDimension>::last_dt_bound() const;
template System<kNativeDimension>::SourceNewtonReport System<kNativeDimension>::newton_report(
    const std::string&) const;
template void System<kNativeDimension>::add_native_block(const std::string&, const std::string&,
                                                         const std::string&, const std::string&,
                                                         const std::string&, const std::string&,
                                                         double, int, bool, int,
                                                         const std::vector<double>&, double);
template void System<kNativeDimension>::add_coupled_source(const CoupledSourceProgram&, double,
                                                           const std::string&);
template void System<kNativeDimension>::add_coupling_operator(const CouplingOperator&);
template void System<kNativeDimension>::install_prepared_coupling_operator(
    const std::string&, CouplingOperatorView,
    std::function<void(Real, const std::vector<MultiFab<kNativeDimension>*>&)>, double,
    std::function<Real()>);
template const std::vector<CouplingOperatorView>& System<kNativeDimension>::coupled_operators()
    const;

}  // namespace pops
