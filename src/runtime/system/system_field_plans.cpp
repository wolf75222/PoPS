/// @file
/// @brief Exact-ranked field-plan registry and backend provider installation for System.

#include "system_impl.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/identity/sha256.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace pops {
namespace {

template <class Value>
Value exact_option(const PreparedProviderOptions& options, std::string_view name) {
  const auto found = options.values.find(std::string(name));
  if (found == options.values.end() || !std::holds_alternative<Value>(found->second))
    throw std::invalid_argument("field solver option '" + std::string(name) +
                                "' is missing or has the wrong exact type");
  return std::get<Value>(found->second);
}

int exact_int_option(const PreparedProviderOptions& options, std::string_view name) {
  const std::int64_t value = exact_option<std::int64_t>(options, name);
  if (value < static_cast<std::int64_t>(std::numeric_limits<int>::min()) ||
      value > static_cast<std::int64_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("field solver option '" + std::string(name) +
                                "' is outside the native integer range");
  return static_cast<int>(value);
}

struct ConfiguredFieldOptions {
  double relative_tolerance = 0.0;
  double absolute_tolerance = 0.0;
  int maximum_iterations = 0;
};

ConfiguredFieldOptions decode_configured_field_options(std::string_view family_route,
                                                       const PreparedProviderOptions& options) {
  if (family_route == "geometric_mg")
    throw std::invalid_argument(
        "GeometricMG is reserved for the AMR MG/FAC route; uniform System requires cartesian_cg");
  if (family_route != "cartesian_cg")
    throw std::invalid_argument("System exact-ranked field solver family is unknown: " +
                                std::string(family_route));
  if (options.schema_identity != "pops.system.cartesian-cg-options@1" || options.values.size() != 3)
    throw std::invalid_argument(
        "System exact-ranked field solver received an incompatible option schema");

  ConfiguredFieldOptions result;
  result.relative_tolerance = exact_option<double>(options, "rel_tol");
  result.absolute_tolerance = exact_option<double>(options, "abs_tol");
  result.maximum_iterations = exact_int_option(options, "max_iterations");
  if (!std::isfinite(result.relative_tolerance) || result.relative_tolerance <= 0.0 ||
      !std::isfinite(result.absolute_tolerance) || result.absolute_tolerance < 0.0 ||
      result.maximum_iterations < 1)
    throw std::invalid_argument(
        "System exact-ranked field solver options are outside their exact domain");
  return result;
}

std::string configured_provider_identity(std::string_view family_route,
                                         const PreparedProviderOptions& options) {
  const std::string contract = options.exact_contract();
  const std::vector<std::uint8_t> bytes(contract.begin(), contract.end());
  return "pops.field-solver." + std::string(family_route) +
         ".configured:sha256:" + identity::sha256_hex(bytes);
}

template <class Implementation>
void require_unmaterialized_field_plan(const Implementation& implementation,
                                       const std::string& provider_slot, const char* operation) {
  if (implementation.named_fields_.contains(provider_slot))
    throw std::logic_error(std::string("System::") + operation +
                           ": field backend is already materialized");
}

}  // namespace

template <int Dim>
std::string System<Dim>::register_configured_field_solver_provider(
    const std::string& family_route, const std::string& provider_route,
    const PreparedProviderOptions& options) {
  require_assembling(p_->lifecycle_, "register_configured_field_solver_provider");
  if (provider_route.empty())
    throw std::invalid_argument("configured field solver provider route must be non-empty");
  if (p_->configured_field_solver_providers_.contains(provider_route) ||
      p_->component_field_solver_providers_.contains(provider_route))
    throw std::runtime_error("field solver provider route is already registered: " +
                             provider_route);
  const ConfiguredFieldOptions decoded = decode_configured_field_options(family_route, options);
  const std::string exact_identity = configured_provider_identity(family_route, options);
  p_->configured_field_solver_providers_.emplace(
      provider_route, typename Impl::ConfiguredFieldSolverProvider{
                          family_route, exact_identity, options, decoded.relative_tolerance,
                          decoded.absolute_tolerance, decoded.maximum_iterations});
  p_->field_plan_consensus_verified_ = false;
  return exact_identity;
}

template <int Dim>
void System<Dim>::set_field_solver_plan(
    const std::string& provider_slot, const std::string& plan_identity,
    const std::string& provider_identity, const std::string& output_owner_identity,
    const std::string& output_block, const std::string& output_key,
    const std::vector<std::string>& provider_identities,
    const std::vector<std::string>& provider_blocks, const std::vector<std::string>& provider_keys,
    const std::vector<double>& provider_coefficients, const std::string& backend_provider_route) {
  require_assembling(p_->lifecycle_, "set_field_solver_plan");
  const std::size_t count = provider_identities.size();
  if (provider_slot.empty() || plan_identity.empty() || provider_identity.empty() ||
      output_owner_identity.empty() || output_block.empty() || output_key.empty() ||
      backend_provider_route.empty() || count == 0 || provider_blocks.size() != count ||
      provider_keys.size() != count || provider_coefficients.size() != count)
    throw std::invalid_argument("System field solver plan is incomplete");
  if (p_->field_plans_.contains(provider_slot))
    throw std::runtime_error("System field solver plan slot is already installed: " +
                             provider_slot);
  for (const auto& [existing_slot, existing] : p_->field_plans_) {
    (void)existing_slot;
    if (existing.output_block == output_block && existing.output_key == output_key)
      throw std::runtime_error(
          "System field solver output block/key is already owned by another qualified slot");
  }

  typename Impl::FieldPlan plan;
  plan.plan_identity = plan_identity;
  plan.provider_identity = provider_identity;
  plan.output_owner_identity = output_owner_identity;
  plan.output_block = output_block;
  plan.output_key = output_key;
  plan.backend_provider_route = backend_provider_route;
  plan.providers.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    if (provider_identities[index].empty() || provider_blocks[index].empty() ||
        provider_keys[index].empty() || !std::isfinite(provider_coefficients[index]))
      throw std::invalid_argument("System field solver provider binding is incomplete");
    plan.providers.push_back({provider_identities[index], provider_blocks[index],
                              provider_keys[index], provider_coefficients[index]});
  }
  p_->field_plans_.emplace(provider_slot, std::move(plan));
  p_->field_plan_consensus_verified_ = false;
}

template <int Dim>
void System<Dim>::set_field_reaction(const std::string& provider_slot, double reaction) {
  require_assembling(p_->lifecycle_, "set_field_reaction");
  require_unmaterialized_field_plan(*p_, provider_slot, "set_field_reaction");
  if (!std::isfinite(reaction) || reaction <= 0.0)
    throw std::invalid_argument(
        "screened field reaction coefficient must be finite and strictly positive");
  const auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("screened field reaction names an unknown provider slot");
  found->second.reaction = reaction;
  found->second.has_reaction = true;
  p_->field_plan_consensus_verified_ = false;
}

template <int Dim>
std::string System<Dim>::register_field_solver_provider(
    const std::string& provider_slot, runtime::field::PreparedFieldSolverSpec spec,
    std::shared_ptr<component::LoadedComponent> topology,
    std::shared_ptr<component::LoadedComponent> solver) {
  require_assembling(p_->lifecycle_, "register_field_solver_provider");
  if (provider_slot.empty() || spec.provider_slot != provider_slot)
    throw std::invalid_argument("external field solver provider slot mismatch");
  if (p_->configured_field_solver_providers_.contains(provider_slot) ||
      p_->component_field_solver_providers_.contains(provider_slot))
    throw std::runtime_error("field solver provider route is already registered: " + provider_slot);
  auto component = std::make_shared<typename Impl::component_field_solver_type>(
      std::move(spec), std::move(topology), std::move(solver));
  const std::string identity(component->provider_identity());
  p_->component_field_solver_providers_.emplace(provider_slot, std::move(component));
  p_->field_plan_consensus_verified_ = false;
  return identity;
}

template <int Dim>
void System<Dim>::register_field_nullspace_provider(
    std::shared_ptr<const FieldNullspaceProvider<Dim>> provider) {
  require_assembling(p_->lifecycle_, "register_field_nullspace_provider");
  if (!p_->field_nullspace_providers_)
    throw std::logic_error("System field-nullspace registry is absent");
  p_->field_nullspace_providers_->add(std::move(provider));
  p_->field_plan_consensus_verified_ = false;
}

template <int Dim>
void System<Dim>::set_default_field_nullspace(const std::string& nullspace_provider_identity,
                                              const PreparedProviderOptions& options) {
  require_assembling(p_->lifecycle_, "set_default_field_nullspace");
  if (nullspace_provider_identity.empty())
    throw std::invalid_argument("default field nullspace provider identity must be non-empty");
  (void)options.exact_contract();
  p_->default_nullspace_provider_identity_ = nullspace_provider_identity;
  p_->default_nullspace_options_ = options;
  p_->default_field_.reset();
  p_->field_plan_consensus_verified_ = false;
}

template <int Dim>
void System<Dim>::set_field_topology_authority(const std::string& provider_slot,
                                               const std::string& provider_kind,
                                               const std::string& provenance,
                                               const std::string& topology_digest) {
  require_assembling(p_->lifecycle_, "set_field_topology_authority");
  require_unmaterialized_field_plan(*p_, provider_slot, "set_field_topology_authority");
  if (provider_kind.empty() || provenance.empty() || topology_digest.empty())
    throw std::invalid_argument("field topology authority is incomplete");
  const auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("field topology authority names an unknown provider slot");
  found->second.topology_provider_kind = provider_kind;
  found->second.topology_provenance = provenance;
  found->second.topology_digest = topology_digest;
  p_->field_plan_consensus_verified_ = false;
}

template <int Dim>
std::vector<runtime::field::FieldTopologyReportRow> System<Dim>::field_topology_report(
    const std::string& provider_slot) const {
  const auto field = p_->named_fields_.find(provider_slot);
  if (field != p_->named_fields_.end())
    return field->second->topology_report();
  if (!p_->field_plans_.contains(provider_slot))
    throw std::runtime_error("unknown qualified field provider slot");
  return {};
}

template <int Dim>
void System<Dim>::set_field_boundary_plan(const std::string& provider_slot,
                                          const std::vector<std::string>& kind,
                                          const std::vector<double>& alpha,
                                          const std::vector<double>& beta,
                                          const std::vector<double>& value) {
  require_assembling(p_->lifecycle_, "set_field_boundary_plan");
  require_unmaterialized_field_plan(*p_, provider_slot, "set_field_boundary_plan");
  const std::size_t faces = static_cast<std::size_t>(2 * Dim);
  if (kind.size() != faces || alpha.size() != faces || beta.size() != faces ||
      value.size() != faces)
    throw std::invalid_argument(
        "System field boundary plan must cover both faces of every exact axis");
  for (std::size_t face = 0; face < faces; ++face) {
    if (kind[face] != "periodic" && kind[face] != "dirichlet" && kind[face] != "neumann" &&
        kind[face] != "mixed")
      throw std::invalid_argument("System field boundary kind is unknown");
    if (!std::isfinite(alpha[face]) || !std::isfinite(beta[face]) || !std::isfinite(value[face]))
      throw std::invalid_argument("System field boundary coefficients must be finite");
  }
  const auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("field boundary plan names an unknown provider slot");
  found->second.boundary_kind = kind;
  found->second.boundary_alpha = alpha;
  found->second.boundary_beta = beta;
  found->second.boundary_value = value;
  p_->field_plan_consensus_verified_ = false;
}

template <int Dim>
void System<Dim>::set_field_boundary_dependencies(const std::string& provider_slot,
                                                  const std::vector<std::string>& state_blocks,
                                                  const std::vector<int>& state_components,
                                                  const std::vector<std::string>& field_blocks,
                                                  const std::vector<std::string>& field_keys,
                                                  const std::vector<int>& field_components) {
  require_assembling(p_->lifecycle_, "set_field_boundary_dependencies");
  require_unmaterialized_field_plan(*p_, provider_slot, "set_field_boundary_dependencies");
  if (state_blocks.size() != state_components.size() || field_blocks.size() != field_keys.size() ||
      field_blocks.size() != field_components.size())
    throw std::invalid_argument("System field boundary dependency vectors differ in length");
  const auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("field boundary dependencies name an unknown provider slot");
  found->second.boundary_state_blocks = state_blocks;
  found->second.boundary_state_components = state_components;
  found->second.boundary_field_blocks = field_blocks;
  found->second.boundary_field_keys = field_keys;
  found->second.boundary_field_components = field_components;
  p_->field_plan_consensus_verified_ = false;
}

template <int Dim>
void System<Dim>::set_field_boundary_kernel(const std::string& provider_slot,
                                            const CompiledFieldBoundaryKernel<Dim>& kernel) {
  require_assembling(p_->lifecycle_, "set_field_boundary_kernel");
  const auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("field boundary kernel names an unknown provider slot");
  kernel.validate();
  // A generated DSO installer is an untrusted publication producer: while the loader owns a
  // staging transaction this setter records only its candidate authority.  In particular it may
  // target a field that was necessarily materialized by the block loader before install_program;
  // the live plan and solver remain untouched until the complete artifact registry is validated.
  if (p_->program_.artifact_field_boundary_stage_) {
    auto& stage = *p_->program_.artifact_field_boundary_stage_;
    const auto candidate = stage.authorities.find(provider_slot);
    if (candidate == stage.authorities.end())
      throw std::runtime_error(
          "field boundary artifact candidate does not cover the complete field-plan registry");
    if (!stage.kernel_slots.insert(provider_slot).second)
      throw std::logic_error(
          "field boundary artifact installed the same qualified provider slot more than once");
    if (candidate->second.kernel)
      throw std::logic_error(
          "field boundary artifact cannot replace a statically authored boundary kernel");
    candidate->second.kernel = kernel;
    return;
  }
  require_unmaterialized_field_plan(*p_, provider_slot, "set_field_boundary_kernel");
  if (found->second.boundary_kernel)
    throw std::logic_error("field boundary kernel is already installed for this provider slot");
  found->second.boundary_kernel = kernel;
  p_->field_plan_consensus_verified_ = false;
}

template <int Dim>
void System<Dim>::set_field_logical_timepoint(const std::string& provider_slot,
                                              const FieldLogicalTimePoint& point) {
  const auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("field logical timepoint names an unknown provider slot");
  const bool invalid = !std::isfinite(static_cast<double>(point.time)) ||
                       !std::isfinite(static_cast<double>(point.dt)) || point.dt <= Real(0) ||
                       point.clock_slot < 0 || point.partition_slot < 0 || point.stage_slot < 0 ||
                       point.level != 0 || point.step < 0 || point.substep < 0 ||
                       point.iteration < 0;
  // Artifact installation is deliberately non-collective inside the untrusted DSO callback: one
  // rank may reject, omit, or duplicate a setter without stranding peers in a communicator call.
  // The enclosing loader transaction performs the collective failure reduction and exact ordered
  // candidate comparison after every rank has returned from the callback.
  if (p_->program_.artifact_field_boundary_stage_) {
    if (invalid)
      throw std::invalid_argument(
          "System field logical timepoint must be one complete level-zero coordinate");
    auto& stage = *p_->program_.artifact_field_boundary_stage_;
    const auto candidate = stage.authorities.find(provider_slot);
    if (candidate == stage.authorities.end())
      throw std::runtime_error(
          "field boundary artifact timepoint does not cover the complete field-plan registry");
    if (!stage.point_slots.insert(provider_slot).second)
      throw std::logic_error(
          "field boundary artifact installed the same logical timepoint more than once");
    candidate->second.point = point;
    return;
  }
  if (all_reduce_max(invalid ? 1L : 0L) != 0)
    throw std::invalid_argument(
        "System field logical timepoint must be one complete level-zero coordinate");
  ExactContractBuilder contract;
  contract.text("pops.system.field-logical-timepoint")
      .scalar(std::uint32_t{1})
      .text(provider_slot)
      .scalar(point.time)
      .scalar(point.dt)
      .scalar(point.clock_slot)
      .scalar(point.partition_slot)
      .scalar(point.stage_slot)
      .scalar(point.level)
      .scalar(point.step)
      .scalar(point.substep)
      .scalar(point.iteration);
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-field-logical-timepoint", std::move(contract).release()}}))
    throw std::invalid_argument(
        "System field logical timepoint differs between communicator ranks");
  found->second.boundary_point = point;
}

template <int Dim>
void System<Dim>::set_field_boundary_parameters(const std::string& provider_slot,
                                                const std::vector<double>& parameters) {
  require_assembling(p_->lifecycle_, "set_field_boundary_parameters");
  if (!std::all_of(parameters.begin(), parameters.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("field boundary parameters must be finite");
  const auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("field boundary parameters name an unknown provider slot");
  if (p_->program_.artifact_field_boundary_stage_) {
    auto& stage = *p_->program_.artifact_field_boundary_stage_;
    const auto candidate = stage.authorities.find(provider_slot);
    if (candidate == stage.authorities.end())
      throw std::runtime_error(
          "field boundary artifact parameters do not cover the complete field-plan registry");
    if (!stage.parameter_slots.insert(provider_slot).second)
      throw std::logic_error(
          "field boundary artifact installed the same parameter pack more than once");
    candidate->second.parameters.assign(parameters.begin(), parameters.end());
    return;
  }
  require_unmaterialized_field_plan(*p_, provider_slot, "set_field_boundary_parameters");
  found->second.boundary_parameters.assign(parameters.begin(), parameters.end());
  p_->field_plan_consensus_verified_ = false;
}

template <int Dim>
void System<Dim>::set_field_newton_plan(const std::string& provider_slot, double tolerance,
                                        int max_iterations, double linear_tolerance,
                                        int linear_max_iterations, int restart, double armijo,
                                        double minimum_step) {
  require_assembling(p_->lifecycle_, "set_field_newton_plan");
  require_unmaterialized_field_plan(*p_, provider_slot, "set_field_newton_plan");
  const FieldNewtonOptions options{
      static_cast<Real>(tolerance),   max_iterations, static_cast<Real>(linear_tolerance),
      linear_max_iterations,          restart,        static_cast<Real>(armijo),
      static_cast<Real>(minimum_step)};
  validate_field_newton_options(options);
  const auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("field Newton plan names an unknown provider slot");
  found->second.newton = options;
  p_->field_plan_consensus_verified_ = false;
}

template <int Dim>
void System<Dim>::set_field_nullspace(const std::string& provider_slot,
                                      const std::string& nullspace_provider_identity,
                                      const PreparedProviderOptions& options) {
  require_assembling(p_->lifecycle_, "set_field_nullspace");
  require_unmaterialized_field_plan(*p_, provider_slot, "set_field_nullspace");
  if (nullspace_provider_identity.empty())
    throw std::invalid_argument("field nullspace provider identity must be non-empty");
  (void)options.exact_contract();
  const auto found = p_->field_plans_.find(provider_slot);
  if (found == p_->field_plans_.end())
    throw std::runtime_error("field nullspace names an unknown provider slot");
  found->second.nullspace_provider_identity = nullspace_provider_identity;
  found->second.nullspace_options = options;
  p_->field_plan_consensus_verified_ = false;
}

template std::string System<kNativeDimension>::register_configured_field_solver_provider(
    const std::string&, const std::string&, const PreparedProviderOptions&);
template void System<kNativeDimension>::set_field_solver_plan(
    const std::string&, const std::string&, const std::string&, const std::string&,
    const std::string&, const std::string&, const std::vector<std::string>&,
    const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<double>&,
    const std::string&);
template void System<kNativeDimension>::set_field_reaction(const std::string&, double);
template std::string System<kNativeDimension>::register_field_solver_provider(
    const std::string&, runtime::field::PreparedFieldSolverSpec,
    std::shared_ptr<component::LoadedComponent>, std::shared_ptr<component::LoadedComponent>);
template void System<kNativeDimension>::register_field_nullspace_provider(
    std::shared_ptr<const FieldNullspaceProvider<kNativeDimension>>);
template void System<kNativeDimension>::set_default_field_nullspace(const std::string&,
                                                                    const PreparedProviderOptions&);
template void System<kNativeDimension>::set_field_topology_authority(const std::string&,
                                                                     const std::string&,
                                                                     const std::string&,
                                                                     const std::string&);
template std::vector<runtime::field::FieldTopologyReportRow>
System<kNativeDimension>::field_topology_report(const std::string&) const;
template void System<kNativeDimension>::set_field_boundary_plan(const std::string&,
                                                                const std::vector<std::string>&,
                                                                const std::vector<double>&,
                                                                const std::vector<double>&,
                                                                const std::vector<double>&);
template void System<kNativeDimension>::set_field_boundary_dependencies(
    const std::string&, const std::vector<std::string>&, const std::vector<int>&,
    const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<int>&);
template void System<kNativeDimension>::set_field_boundary_kernel(
    const std::string&, const CompiledFieldBoundaryKernel<kNativeDimension>&);
template void System<kNativeDimension>::set_field_logical_timepoint(const std::string&,
                                                                    const FieldLogicalTimePoint&);
template void System<kNativeDimension>::set_field_boundary_parameters(const std::string&,
                                                                      const std::vector<double>&);
template void System<kNativeDimension>::set_field_newton_plan(const std::string&, double, int,
                                                              double, int, int, double, double);
template void System<kNativeDimension>::set_field_nullspace(const std::string&, const std::string&,
                                                            const PreparedProviderOptions&);

}  // namespace pops
