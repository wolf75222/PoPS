/// @file
/// @brief Atomic installation of exact-ranked Uniform System providers.

#include "system_impl.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/coupling/source/coupled_source_program.hpp>
#include <pops/runtime/builders/compiled/native_loader.hpp>
#include <pops/runtime/named_field_output.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
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

template <int Dim>
mesh::BoxArrayValidationBudget exact_layout_budget(const mesh::BoxArray<Dim>& layout) {
  const std::size_t boxes = layout.size();
  if (boxes > 1 && boxes - 1 > std::numeric_limits<std::size_t>::max() / boxes)
    throw std::length_error("FFT field layout validation budget exceeds size_t");
  return {boxes, boxes < 2 ? 0 : boxes * (boxes - 1) / 2};
}

template <int Dim>
PhysicalBoundaryConditions<Dim> fft_periodic_boundary(const Geometry<Dim>& geometry) {
  std::array<bool, Dim> periodic{};
  std::array<PhysicalBoundaryFace, 2 * Dim> faces{};
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    periodic[axis] = true;
    spacing[axis] = geometry.spacing(axis);
  }
  return {BoundaryTopology<Dim>::axis_periodic(periodic), faces, spacing};
}

template <int Dim>
EllipticBuildRequest<Dim> fft_build_request(const Geometry<Dim>& geometry,
                                            const mesh::BoxArray<Dim>& layout,
                                            const mesh::Distribution<Dim>& distribution,
                                            Index<Dim> local_rank) {
  Extent<Dim> rhs_ghosts{};
  Extent<Dim> phi_ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    phi_ghosts[axis] = 1;
  return {geometry,
          layout,
          distribution,
          local_rank,
          fft_periodic_boundary(geometry),
          rhs_ghosts,
          phi_ghosts,
          exact_layout_budget(layout)};
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
  if (block.ncomp < 1 || block.provider_components < 0)
    throw std::invalid_argument(
        "prepared System block requires a positive state and non-negative provider value count");
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
void System<Dim>::install_prepared_boundary_execution_lane(std::shared_ptr<ExecutionLane> lane) {
  require_assembling(p_->lifecycle_, "install_prepared_boundary_execution_lane");
  if (!lane || lane->identity().empty() ||
      lane->size() != static_cast<int>(p_->dm.rank_space().size()) ||
      lane->rank() != static_cast<int>(p_->dm.rank_space().linear_rank(p_->local_rank)))
    throw std::invalid_argument(
        "prepared System boundary lane differs from its exact runtime rank space");
  if (prepared_boundary_execution_lane_)
    throw std::logic_error("prepared System boundary lane is already installed");
  prepared_boundary_execution_lane_ = std::move(lane);
}

template <int Dim>
void System<Dim>::stage_prepared_ghost_boundary_component(
    const std::string& block, std::shared_ptr<PreparedGhostBoundaryComponent> component) {
  require_assembling(p_->lifecycle_, "stage_prepared_ghost_boundary_component");
  if (block.empty() || !component)
    throw std::invalid_argument(
        "prepared System GhostBoundary staging requires a block and typed component");
  const auto& spec = component->spec();
  if (spec.region.dimension != Dim ||
      spec.state_identity != p_->boundary_registry_.state_route(block))
    throw std::invalid_argument(
        "prepared System GhostBoundary differs from its exact block/rank route");
  const std::string package_identity =
      "~pops.system.boundary-component.v1\n" + block + "\n" + spec.target_identity;
  stage_prepared_native_package(
      package_identity,
      [this, block, component] {
        typename Impl::Species& selected = p_->blocks_.find(block);
        if (!selected.boundary || selected.state_identity != component->spec().state_identity)
          throw std::invalid_argument(
              "prepared System GhostBoundary block was not materialized with its exact boundary");
        // The RuntimeInstance lane was materialized from its authenticated communicator before
        // plan publication. Each invocation borrows that exact lane through ProgramContext.
        const Geometry<Dim> geometry = p_->geom;
        if (!selected.boundary_full_at_point_prepared ||
            !selected.boundary_core_at_point_prepared ||
            !selected.boundary_flux_full_at_point_prepared ||
            !selected.boundary_flux_core_at_point_prepared)
          throw std::invalid_argument(
              "prepared System GhostBoundary requires complete compiled full and core closures");
        auto wrap_component = [component, geometry](auto compiled) {
          return [component, geometry, compiled = std::move(compiled)](
                     const auto& point, MultiFab<Dim>& state, MultiFab<Dim>& result,
                     const PreparedHyperbolicBoundary<Dim>& boundary, const ExecutionLane& lane,
                     const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
            std::exception_ptr component_error;
            try {
              component->template apply_ghost_region<Dim>(point, state, geometry, lane);
            } catch (...) {
              component_error = std::current_exception();
            }
            if (all_reduce_max(component_error ? 1L : 0L, lane) != 0) {
              if (lane.size() == 1 && component_error)
                std::rethrow_exception(component_error);
              throw std::runtime_error(
                  "prepared System GhostBoundary execution failed collectively");
            }
            compiled(point, state, result, boundary, lane, transport);
          };
        };
        selected.boundary_full_at_point_prepared =
            wrap_component(std::move(selected.boundary_full_at_point_prepared));
        selected.boundary_flux_full_at_point_prepared =
            wrap_component(std::move(selected.boundary_flux_full_at_point_prepared));
        selected.boundary_residual_at_point_prepared = make_prepared_boundary_residual<Dim>(
            selected.boundary_full_at_point_prepared, selected.boundary_core_at_point_prepared);
        selected.boundary_jvp_at_point_prepared =
            make_prepared_boundary_jvp<Dim>(selected.boundary_residual_at_point_prepared);
      },
      component);
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
  std::erase_if(p_->pending_native_packages_, [](const auto& package) {
    return package.identity.starts_with("~pops.system.boundary-component.v1\n");
  });
  prepared_boundary_execution_lane_.reset();
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

  if (prepared.provider_components != 0 && p_->auxiliary_registry_.sealed()) {
    const auto& plan = p_->auxiliary_registry_.consumer_plan(prepared.name);
    if (plan.value_count() != static_cast<std::size_t>(prepared.provider_components))
      throw std::invalid_argument(
          "prepared System block provider count differs from its resolved consumer plan");
  } else if (prepared.provider_components != 0) {
    throw std::logic_error(
        "prepared System block with provider values requires the sealed global provider registry");
  }

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
  candidate.rhs_core_at_point_prepared = std::move(prepared.closures.rhs_core_at_point_prepared);
  candidate.rhs_flux_only_core_at_point_prepared =
      std::move(prepared.closures.rhs_flux_only_core_at_point_prepared);
  candidate.boundary_full_at_point_prepared =
      std::move(prepared.closures.boundary_full_at_point_prepared);
  candidate.boundary_core_at_point_prepared =
      std::move(prepared.closures.boundary_core_at_point_prepared);
  candidate.boundary_flux_full_at_point_prepared =
      std::move(prepared.closures.boundary_flux_full_at_point_prepared);
  candidate.boundary_flux_core_at_point_prepared =
      std::move(prepared.closures.boundary_flux_core_at_point_prepared);
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
  if (candidate.boundary &&
      (!candidate.boundary_full_at_point_prepared || !candidate.boundary_core_at_point_prepared ||
       !candidate.boundary_flux_full_at_point_prepared ||
       !candidate.boundary_flux_core_at_point_prepared ||
       !candidate.boundary_residual_at_point_prepared || !candidate.boundary_jvp_at_point_prepared))
    throw std::invalid_argument(
        "prepared System boundary lacks its complete full/core/residual/JVP authority");

  // Every operation above is preparatory. These moves are noexcept after the one vector growth.
  p_->sp.push_back(std::move(candidate));
  if (installed_boundary != nullptr)
    p_->boundary_registry_.boundary(prepared.name).authority = std::move(boundary);
}

template <int Dim>
void System<Dim>::register_elliptic_field(
    const std::string& block, const std::string& field,
    const std::vector<runtime::system::AuxiliaryComponentKey>& output_keys, int gradient_sign) {
  require_assembling(p_->lifecycle_, "register_elliptic_field");
  if (field.empty())
    throw std::invalid_argument("System named elliptic field identity must be non-empty");
  (void)p_->find(block);
  runtime::field::NamedFieldOutput<Dim> output(output_keys.size(), gradient_sign);
  if (!p_->auxiliary_registry_.sealed())
    throw std::logic_error(
        "System named elliptic outputs require a sealed auxiliary provider registry");
  std::vector<std::string> exact_output_keys;
  exact_output_keys.reserve(output_keys.size());
  for (const auto& key : output_keys) {
    key.validate();
    const std::string exact_key = key.exact_key();
    if (std::find(exact_output_keys.begin(), exact_output_keys.end(), exact_key) !=
        exact_output_keys.end())
      throw std::invalid_argument("System named elliptic output keys must be unique");
    exact_output_keys.push_back(exact_key);
    if (p_->auxiliary_registry_.provider_for_key(key).kind() !=
        runtime::system::AuxiliaryProviderKind::field_output)
      throw std::invalid_argument(
          "System named elliptic output key is not owned by a field-output provider");
  }

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
    if (configured->second.family_route == "fft") {
      if (plan.has_reaction)
        throw std::logic_error(
            "FFT field solver is a constant-coefficient Poisson operator and rejects reaction");
      if (plan.boundary_kernel)
        throw std::logic_error(
            "FFT field solver has a fixed periodic boundary contract and rejects dynamic kernels");
      if (plan.newton)
        throw std::logic_error(
            "FFT field solver is direct linear and rejects Newton configuration");
      for (int axis = 0; axis < Dim; ++axis) {
        if (!p_->periodicity[static_cast<std::size_t>(axis)])
          throw std::logic_error(
              "FFT field solver requires periodic System topology on every axis");
        for (int side = 0; side < 2; ++side) {
          const std::size_t face = static_cast<std::size_t>(2 * axis + side);
          if (operator_options.boundaries[face] != elliptic::nd::CartesianBoundaryKind::periodic)
            throw std::logic_error(
                "FFT field solver requires periodic exact field boundaries on every face");
        }
      }
      const ExecutionLane world = ExecutionLane::world();
      std::optional<EllipticBuildRequest<Dim>> request;
      std::exception_ptr request_error;
      try {
        request.emplace(fft_build_request(p_->geom, p_->ba, p_->dm, p_->local_rank));
      } catch (...) {
        request_error = std::current_exception();
      }
      if (all_reduce_max(request_error ? 1L : 0L, world.communicator()) != 0) {
        if (world.size() == 1 && request_error)
          std::rethrow_exception(request_error);
        throw std::runtime_error("FFT field solver request allocation failed collectively");
      }
      backend = runtime::system::PoissonFftFieldSolverBackend<Dim>::prepare_collectively(
          std::move(*request), configured->second.exact_identity);
    } else if (configured->second.family_route == "cartesian_cg") {
      if (plan.has_reaction)
        throw std::logic_error(
            "configured exact Cartesian field solver does not implement a reaction operator");
      operator_options.absolute_tolerance =
          static_cast<Real>(configured->second.absolute_tolerance);
      operator_options.relative_tolerance =
          static_cast<Real>(configured->second.relative_tolerance);
      operator_options.maximum_iterations = configured->second.maximum_iterations;
      backend = std::make_unique<runtime::system::CartesianCgFieldSolverBackend<Dim>>(
          p_->geom, p_->ba, p_->dm, p_->local_rank, topology, operator_options,
          configured->second.exact_identity);
    } else {
      throw std::logic_error("configured exact field solver route was not decoded exactly");
    }
  }

  auto prepared = std::make_shared<typename System<Dim>::Impl::exact_field_type>(
      provider_slot, block, output, p_->geom, p_->ba, p_->dm, p_->local_rank, std::move(backend),
      p_->sp.size(), output_keys);
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
void System<Dim>::register_native_package(const std::string& name, const std::string& so_path,
                                          const std::string& limiter, const std::string& riemann,
                                          const std::string& recon, const std::string& time,
                                          double gamma, int substeps, bool evolve, int stride,
                                          const std::vector<double>& params,
                                          double positivity_floor) {
  require_assembling(p_->lifecycle_, "register_native_package");
  native_loader::register_native_package<Dim>(this, name, so_path, limiter, riemann, recon, time,
                                              gamma, substeps, evolve, stride, params,
                                              positivity_floor);
}

template <int Dim>
void System<Dim>::stage_prepared_native_package(std::string identity,
                                                std::function<void()> installer,
                                                std::shared_ptr<void> package_lifetime) {
  require_assembling(p_->lifecycle_, "stage_prepared_native_package");
  if (identity.empty() || !installer || !package_lifetime)
    throw std::invalid_argument(
        "System native package staging requires an identity, installer, and DSO lifetime");
  for (const auto& package : p_->pending_native_packages_)
    if (package.identity == identity)
      throw std::invalid_argument("System native package identity is registered more than once");
  for (const auto& package : p_->installed_native_packages_)
    if (package.identity == identity)
      throw std::invalid_argument("System native package identity was already finalized");
  p_->pending_native_packages_.push_back(
      {std::move(identity), std::move(installer), std::move(package_lifetime)});
}

template <int Dim>
void System<Dim>::finalize_native_packages() {
  require_assembling(p_->lifecycle_, "finalize_native_packages");
  if (p_->pending_native_packages_.empty())
    throw std::logic_error(
        "System native package finalization requires at least one staged package");

  std::vector<typename Impl::PendingNativePackage> packages =
      std::move(p_->pending_native_packages_);
  p_->pending_native_packages_.clear();
  std::sort(packages.begin(), packages.end(),
            [](const auto& left, const auto& right) { return left.identity < right.identity; });

  std::vector<ExactOrderedBytePair> exact_packages;
  exact_packages.reserve(packages.size());
  for (const auto& package : packages)
    exact_packages.emplace_back("system-native-package", package.identity);
  if (!all_ranks_agree_exact_ordered_byte_pairs(exact_packages))
    throw std::runtime_error("System staged native packages differ across MPI ranks");

  std::optional<typename Impl::NativePackageFinalizeSnapshot> snapshot;
  std::exception_ptr snapshot_error;
  try {
    snapshot.emplace(*p_);
  } catch (...) {
    snapshot_error = std::current_exception();
  }
  if (all_reduce_max(snapshot_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && snapshot_error)
      std::rethrow_exception(snapshot_error);
    throw std::runtime_error("System native package rollback snapshot failed collectively");
  }

  std::exception_ptr failure;
  try {
    // The sole global seal happens after every typed route registration and before any block
    // constructor captures a provider pointer or local consumer-plan image.
    seal_auxiliary_providers();
    for (const auto& package : packages) {
      std::exception_ptr local_error;
      try {
        package.install();
      } catch (...) {
        local_error = std::current_exception();
      }
      if (all_reduce_max(local_error ? 1L : 0L) != 0) {
        if (n_ranks() == 1 && local_error)
          std::rethrow_exception(local_error);
        throw std::runtime_error("System native package installer failed collectively: " +
                                 package.identity);
      }
    }
  } catch (...) {
    failure = std::current_exception();
  }

  const long failed = failure ? 1L : 0L;
  if (all_reduce_max(failed) != 0) {
    // Every rank restores the same pre-finalization image.  ``packages`` then drops its DSO owners,
    // so no failed package code can remain reachable from the restored System.
    snapshot->restore(*p_);
    if (n_ranks() == 1 && failure)
      std::rethrow_exception(failure);
    throw std::runtime_error("System native package finalization rolled back collectively");
  }

  p_->installed_native_packages_.insert(p_->installed_native_packages_.end(),
                                        std::make_move_iterator(packages.begin()),
                                        std::make_move_iterator(packages.end()));
}

template <int Dim>
void System<Dim>::add_coupled_source_prepared_(const CoupledSourceProgram& description,
                                               double frequency, const std::string& label,
                                               CouplingOperatorView inspect) {
  struct InputRef {
    int block = -1;
    int component = -1;
  };
  struct OutputRef {
    int block = -1;
    int component = -1;
    CsProgram program{};
  };

  std::vector<InputRef> inputs;
  std::vector<OutputRef> outputs;
  std::vector<Real> constants;
  CsProgram frequency_program{};
  bool has_frequency_program = false;
  std::exception_ptr local_error;
  try {
    require_assembling(p_->lifecycle_, "add_coupled_source");
    if (label.empty() || !std::isfinite(frequency) || frequency < 0.0)
      throw std::invalid_argument(
          "System coupled source requires a non-empty label and finite non-negative frequency");
    if (description.in_blocks.size() != description.in_roles.size() ||
        description.out_blocks.size() != description.out_roles.size() ||
        description.out_blocks.size() != description.prog_lens.size() ||
        description.prog_ops.size() != description.prog_args.size() ||
        description.freq_prog_ops.size() != description.freq_prog_args.size())
      throw std::invalid_argument("System coupled-source arrays have inconsistent lengths");
    if (description.out_blocks.empty())
      throw std::invalid_argument("System coupled source requires at least one output term");
    if (description.in_blocks.size() + description.consts.size() >
            static_cast<std::size_t>(kCsMaxReg) ||
        description.out_blocks.size() > static_cast<std::size_t>(kCsMaxTerms))
      throw std::length_error("System coupled source exceeds its device register/term capacity");

    const auto resolve = [&](const std::string& block_name, const std::string& token) -> InputRef {
      const int block = p_->blocks_.index(block_name);
      const VariableSet& variables = p_->sp[static_cast<std::size_t>(block)].cons_vars;
      validate_variable_semantics<Dim>(variables, "System::add_coupled_source", block_name);
      if (!variables.user_roles.empty() && variables.user_roles.size() != variables.roles.size())
        throw std::invalid_argument("System block '" + block_name +
                                    "' has incomplete custom-role metadata");
      int component = -1;
      for (int candidate = 0; candidate < variables.size; ++candidate) {
        const VariableSemantic semantic = variables.roles[static_cast<std::size_t>(candidate)];
        bool matches = false;
        if (semantic == VariableSemantic::Custom) {
          const std::string_view label =
              variables.user_roles.empty()
                  ? std::string_view{}
                  : std::string_view(variables.user_roles[static_cast<std::size_t>(candidate)]);
          matches = label.empty() ? token == "custom" : label == token;
        } else {
          matches = role_name(semantic) == token;
        }
        if (!matches)
          continue;
        if (component >= 0)
          throw std::invalid_argument("System block '" + block_name + "' declares role token '" +
                                      token + "' more than once");
        component = candidate;
      }
      if (component < 0)
        throw std::invalid_argument("System block '" + block_name +
                                    "' does not declare role token '" + token +
                                    "' (declared: " + roles_csv(variables) + ")");
      return {block, component};
    };

    inputs.reserve(description.in_blocks.size());
    for (std::size_t index = 0; index < description.in_blocks.size(); ++index)
      inputs.push_back(resolve(description.in_blocks[index], description.in_roles[index]));

    outputs.reserve(description.out_blocks.size());
    std::size_t offset = 0;
    const int register_count =
        static_cast<int>(description.in_blocks.size() + description.consts.size());
    for (std::size_t term = 0; term < description.out_blocks.size(); ++term) {
      const InputRef target = resolve(description.out_blocks[term], description.out_roles[term]);
      const int length = description.prog_lens[term];
      if (length < 0 || length > kCsMaxProg || offset > description.prog_ops.size() ||
          static_cast<std::size_t>(length) > description.prog_ops.size() - offset)
        throw std::invalid_argument("System coupled-source term program has an invalid length");
      CsProgram program{};
      program.len = length;
      for (int instruction = 0; instruction < length; ++instruction) {
        const int opcode = description.prog_ops[offset + static_cast<std::size_t>(instruction)];
        const int argument = description.prog_args[offset + static_cast<std::size_t>(instruction)];
        if (opcode < 0 || opcode > static_cast<int>(CsOp::Sqrt) ||
            (opcode == static_cast<int>(CsOp::PushReg) &&
             (argument < 0 || argument >= register_count)))
          throw std::invalid_argument("System coupled-source term contains an invalid instruction");
        program.op[instruction] = opcode;
        program.arg[instruction] = argument;
      }
      validate_cs_program_stack(program, "System::add_coupled_source term " + std::to_string(term));
      outputs.push_back({target.block, target.component, program});
      offset += static_cast<std::size_t>(length);
    }
    if (offset != description.prog_ops.size())
      throw std::invalid_argument("System coupled-source term lengths do not consume the program");

    has_frequency_program = !description.freq_prog_ops.empty();
    if (has_frequency_program) {
      if (description.freq_prog_ops.size() > static_cast<std::size_t>(kCsMaxProg))
        throw std::length_error("System coupled-source frequency program is too long");
      frequency_program.len = static_cast<int>(description.freq_prog_ops.size());
      for (int instruction = 0; instruction < frequency_program.len; ++instruction) {
        const int opcode = description.freq_prog_ops[static_cast<std::size_t>(instruction)];
        const int argument = description.freq_prog_args[static_cast<std::size_t>(instruction)];
        if (opcode < 0 || opcode > static_cast<int>(CsOp::Sqrt) ||
            (opcode == static_cast<int>(CsOp::PushReg) &&
             (argument < 0 || argument >= register_count)))
          throw std::invalid_argument(
              "System coupled-source frequency contains an invalid instruction");
        frequency_program.op[instruction] = opcode;
        frequency_program.arg[instruction] = argument;
      }
      validate_cs_program_stack(frequency_program, "System::add_coupled_source frequency");
    }
    constants.assign(description.consts.begin(), description.consts.end());
    if (std::any_of(constants.begin(), constants.end(),
                    [](Real value) { return !std::isfinite(value); }))
      throw std::invalid_argument("System coupled-source constants must be finite native reals");
    if (!inspect.label.empty()) {
      if (inspect.label != label || inspect.frequency.constant_mu != frequency ||
          inspect.frequency.per_cell != has_frequency_program)
        throw std::invalid_argument(
            "typed System coupling inspect metadata differs from its executable program");
    } else {
      inspect.label = label;
      inspect.frequency.constant_mu = frequency;
      inspect.frequency.per_cell = has_frequency_program;
    }
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("System coupled-source preparation failed collectively");
  }

  ExactContractBuilder contract;
  contract.text("pops.system.coupled-source")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .text(label)
      .scalar(frequency)
      .sequence(description.in_blocks,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(description.in_roles,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(description.consts)
      .sequence(description.out_blocks,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(description.out_roles,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(description.prog_ops)
      .sequence(description.prog_args)
      .sequence(description.prog_lens)
      .sequence(description.freq_prog_ops)
      .sequence(description.freq_prog_args)
      .sequence(inspect.conservation.conserved_roles,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .sequence(inspect.conservation.created_roles,
                [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
      .scalar(static_cast<std::uint8_t>(inspect.frequency.per_cell ? 1 : 0));
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("system-coupled-source"), contract.view()}}))
    throw std::invalid_argument("System coupled-source descriptions differ between MPI ranks");

  const int input_count = static_cast<int>(inputs.size());
  const int constant_count = static_cast<int>(constants.size());
  const int output_count = static_cast<int>(outputs.size());
  auto operation = [inputs, outputs, constants, input_count, constant_count, output_count](
                       Real dt, const std::vector<MultiFab<Dim>*>& states) {
    const int reference_block = input_count != 0 ? inputs.front().block : outputs.front().block;
    MultiFab<Dim>& reference = *states[static_cast<std::size_t>(reference_block)];
    for (const InputRef& input : inputs)
      if (states[static_cast<std::size_t>(input.block)]->local_global_indices() !=
          reference.local_global_indices())
        throw std::invalid_argument("System coupled-source input layouts are not co-located");
    for (const OutputRef& output : outputs)
      if (states[static_cast<std::size_t>(output.block)]->local_global_indices() !=
          reference.local_global_indices())
        throw std::invalid_argument("System coupled-source output layouts are not co-located");
    for (std::size_t local = 0; local < reference.local_size(); ++local) {
      CoupledSourceKernel<Dim> kernel{};
      kernel.dt = dt;
      kernel.n_in = input_count;
      kernel.n_const = constant_count;
      kernel.n_terms = output_count;
      for (int input = 0; input < input_count; ++input) {
        const InputRef& ref = inputs[static_cast<std::size_t>(input)];
        const Fab<Dim>& fab = states[static_cast<std::size_t>(ref.block)]->fab(local);
        kernel.in[input] = fab.view();
        kernel.in_comp[input] = ref.component;
      }
      for (int constant = 0; constant < constant_count; ++constant)
        kernel.consts[constant] = constants[static_cast<std::size_t>(constant)];
      for (int output = 0; output < output_count; ++output) {
        const OutputRef& ref = outputs[static_cast<std::size_t>(output)];
        kernel.out[output] = states[static_cast<std::size_t>(ref.block)]->fab(local).view();
        kernel.out_comp[output] = ref.component;
        kernel.prog[output] = ref.program;
      }
      for_each_cell(reference.box(local), kernel);
    }
  };

  std::function<Real()> maximum_frequency;
  if (has_frequency_program) {
    Impl* implementation = p_.get();
    maximum_frequency = [implementation, inputs, constants, frequency_program, input_count,
                         constant_count]() {
      if (input_count == 0)
        throw std::logic_error("state-dependent coupling frequency requires at least one input");
      const MultiFab<Dim>& reference =
          implementation->sp[static_cast<std::size_t>(inputs.front().block)].U;
      Real local_maximum = Real(0);
      for (std::size_t local = 0; local < reference.local_size(); ++local) {
        CoupledFreqKernel<Dim> kernel{};
        kernel.n_in = input_count;
        kernel.n_const = constant_count;
        kernel.prog = frequency_program;
        for (int input = 0; input < input_count; ++input) {
          const InputRef& ref = inputs[static_cast<std::size_t>(input)];
          const Fab<Dim>& fab =
              implementation->sp[static_cast<std::size_t>(ref.block)].U.fab(local);
          kernel.in[input] = fab.view();
          kernel.in_comp[input] = ref.component;
        }
        for (int constant = 0; constant < constant_count; ++constant)
          kernel.consts[constant] = constants[static_cast<std::size_t>(constant)];
        local_maximum = std::max(
            local_maximum,
            for_each_cell_reduce_max(reference.box(local), [=] POPS_HD(const Index<Dim>& index) {
              Real value = std::numeric_limits<Real>::lowest();
              kernel(index, value);
              return value;
            }));
      }
      return all_reduce_max(local_maximum);
    };
  }
  install_prepared_coupling_operator(label, std::string(contract.view()), std::move(inspect),
                                     std::move(operation), frequency, std::move(maximum_frequency));
}

template <int Dim>
void System<Dim>::add_coupled_source(const CoupledSourceProgram& description, double frequency,
                                     const std::string& label) {
  add_coupled_source_prepared_(description, frequency, label, {});
}

template <int Dim>
void System<Dim>::add_coupling_operator(const CouplingOperator& op) {
  validate_coupling_contract(op, "System::add_coupling_operator");
  add_coupled_source_prepared_(op.program, op.frequency.constant_mu, op.label,
                               CouplingOperatorView{op.label, op.conservation, op.frequency});
}

template <int Dim>
void System<Dim>::install_prepared_coupling_operator(
    const std::string& label, const std::string& provider_contract, CouplingOperatorView view,
    std::function<void(Real, const std::vector<MultiFab<Dim>*>&)> operation,
    double constant_frequency, std::function<Real()> maximum_frequency) {
  std::exception_ptr local_error;
  runtime::system::SystemCouplingRegistry<Dim> candidate;
  std::string exact;
  try {
    require_assembling(p_->lifecycle_, "install_prepared_coupling_operator");
    if (label.empty() || provider_contract.empty() || !operation ||
        !std::isfinite(constant_frequency) || constant_frequency < 0.0)
      throw std::invalid_argument(
          "prepared System coupling requires exact provider identity, executable, and finite "
          "frequency");
    if (!view.label.empty() && view.label != label)
      throw std::invalid_argument(
          "prepared System coupling inspect identity differs from its executable provider");
    if (p_->coupling_.operator_contracts.size() != p_->coupling_.operators.size() ||
        p_->coupling_.operators.size() != p_->coupling_.coupled_operators.size())
      throw std::logic_error("prepared System coupling registry is inconsistent");
    view.label = label;
    view.frequency.constant_mu = constant_frequency;
    view.frequency.per_cell = static_cast<bool>(maximum_frequency);

    ExactContractBuilder contract;
    contract.text("pops.system.prepared-coupling")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(label)
        .bytes(provider_contract)
        .scalar(constant_frequency)
        .scalar(static_cast<std::uint8_t>(maximum_frequency ? 1 : 0))
        .sequence(view.conservation.conserved_roles,
                  [](ExactContractBuilder& item, const std::string& role) { item.text(role); })
        .sequence(view.conservation.created_roles,
                  [](ExactContractBuilder& item, const std::string& role) { item.text(role); })
        .scalar(static_cast<std::uint64_t>(p_->sp.size()));
    for (const typename Impl::Species& block : p_->sp) {
      if (block.name.empty() || block.state_identity.empty())
        throw std::invalid_argument(
            "prepared System coupling block map lacks an exact state identity");
      contract.text(block.name)
          .text(block.state_identity)
          .scalar(block.ncomp)
          .scalar(static_cast<std::uint64_t>(block.cons_vars.roles.size()));
      for (const VariableSemantic role : block.cons_vars.roles)
        contract.scalar(static_cast<std::uint8_t>(role.kind)).scalar(role.axis);
      contract.sequence(
          block.cons_vars.user_roles,
          [](ExactContractBuilder& item, const std::string& role) { item.text(role); });
    }
    exact = std::move(contract).release();

    // Copy and allocate the complete candidate registry before consensus.  A rank-local allocation
    // failure therefore cannot publish a shorter provider list on another rank.
    candidate = p_->coupling_;
    candidate.operators.push_back(std::move(operation));
    candidate.operator_contracts.push_back(exact);
    candidate.coupled_operators.push_back(std::move(view));
    if (constant_frequency > 0.0)
      candidate.coupled_freqs.push_back({label, constant_frequency});
    if (maximum_frequency)
      candidate.coupled_frequencies.push_back({label, std::move(maximum_frequency)});
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("prepared System coupling installation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("system-prepared-coupling"), exact}}))
    throw std::invalid_argument("prepared System coupling contract differs between MPI ranks");
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
template void System<kNativeDimension>::install_prepared_boundary_execution_lane(
    std::shared_ptr<ExecutionLane>);
template void System<kNativeDimension>::stage_prepared_ghost_boundary_component(
    const std::string&, std::shared_ptr<PreparedGhostBoundaryComponent>);
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
template void System<kNativeDimension>::register_elliptic_field(
    const std::string&, const std::string&,
    const std::vector<runtime::system::AuxiliaryComponentKey>&, int);
template void System<kNativeDimension>::set_block_elliptic_field(
    const std::string&, const std::string&,
    std::function<void(const MultiFab<kNativeDimension>&, MultiFab<kNativeDimension>&)>);
template void System<kNativeDimension>::add_dt_bound(const std::string&, std::function<double()>);
template std::string System<kNativeDimension>::last_dt_bound() const;
template void System<kNativeDimension>::register_native_package(
    const std::string&, const std::string&, const std::string&, const std::string&,
    const std::string&, const std::string&, double, int, bool, int, const std::vector<double>&,
    double);
template void System<kNativeDimension>::stage_prepared_native_package(std::string,
                                                                      std::function<void()>,
                                                                      std::shared_ptr<void>);
template void System<kNativeDimension>::finalize_native_packages();
template void System<kNativeDimension>::add_coupled_source(const CoupledSourceProgram&, double,
                                                           const std::string&);
template void System<kNativeDimension>::add_coupling_operator(const CouplingOperator&);
template void System<kNativeDimension>::install_prepared_coupling_operator(
    const std::string&, const std::string&, CouplingOperatorView,
    std::function<void(Real, const std::vector<MultiFab<kNativeDimension>*>&)>, double,
    std::function<Real()>);
template const std::vector<CouplingOperatorView>& System<kNativeDimension>::coupled_operators()
    const;

}  // namespace pops
