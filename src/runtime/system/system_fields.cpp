/// @file
/// @brief Exact compile-time-ranked System state and elliptic-field surface.

#include "system_impl.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/parallel/solve_report_consensus.hpp>
#include <pops/runtime/analytic/collective_preflight.hpp>
#include <pops/runtime/output_piece_collective.hpp>
#include <pops/runtime/system/auxiliary_ghost_fill.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace pops {
namespace {

template <int Dim>
void copy_valid(const MultiFab<Dim>& source, MultiFab<Dim>& destination) {
  if (source.layout() != destination.layout() ||
      source.distribution() != destination.distribution() ||
      source.local_rank() != destination.local_rank() || source.ncomp() != destination.ncomp())
    throw std::invalid_argument("System valid-cell copy requires one exact ND layout");
  for (int component = 0; component < source.ncomp(); ++component)
    elliptic::nd::detail::copy_component(source, component, destination, component);
  Kokkos::fence();
}

template <int Dim>
void copy_field_outputs_to_provider_candidate(
    const MultiFab<Dim>& outputs, const std::vector<runtime::system::AuxiliaryComponentKey>& keys,
    const runtime::system::ExactAuxiliaryRegistry<Dim>& registry,
    runtime::system::AuxiliaryStorageGroups<Dim>& candidate) {
  if (keys.size() != static_cast<std::size_t>(outputs.ncomp()))
    throw std::invalid_argument(
        "System field-output key count differs from its compact candidate width");
  for (std::size_t slot = 0; slot < keys.size(); ++slot) {
    const auto address = registry.address_of(keys[slot]);
    MultiFab<Dim>* destination = candidate.find(address.group);
    if (destination == nullptr ||
        address.component >= static_cast<std::size_t>(destination->ncomp()))
      throw std::logic_error(
          "System field-output candidate lacks its resolved provider storage address");
    if (destination->layout() != outputs.layout() ||
        destination->distribution() != outputs.distribution() ||
        destination->local_rank() != outputs.local_rank())
      throw std::invalid_argument(
          "System field-output provider group differs from the solved field layout");
    elliptic::nd::detail::copy_component(outputs, static_cast<int>(slot), *destination,
                                         static_cast<int>(address.component));
  }
  Kokkos::fence();
}

template <int Dim, class Implementation>
runtime::system::AuxiliaryEvaluationPoint field_output_evaluation_point(
    const Implementation& implementation) {
  if (implementation.macro_step_ < 0)
    throw std::logic_error("System field-output publication has a negative accepted step");
  runtime::system::AuxiliaryEvaluationPoint point;
  point.clock = "pops.system.accepted";
  point.accepted_step = static_cast<std::uint64_t>(implementation.macro_step_);
  point.layout_generation = implementation.embedded_boundary_generation_;
  point.event = runtime::system::AuxiliaryEvaluationEvent::before_residual;
  ExactContractBuilder exact;
  point.serialize_exact(exact);
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-field-output-evaluation-point", std::move(exact).release()}}))
    throw std::runtime_error("System field-output evaluation point differs across MPI ranks");
  return point;
}

using runtime::system::marshaling::checked_cell_count;
using runtime::system::marshaling::domain_ordinal;
using runtime::system::marshaling::for_each_host_index;
using runtime::system::marshaling::gather_global;
using runtime::system::marshaling::gather_local_compact;
using runtime::system::marshaling::storage_ordinal;
using runtime::system::marshaling::write_global;

template <int Dim>
RelativeCellMeasure<Dim> embedded_cell_measure(
    const std::shared_ptr<const runtime::system::PreparedEmbeddedBoundaryGeometry<Dim>>& embedded) {
  RelativeCellMeasure<Dim> measure;
  if (!embedded || embedded->mode() == runtime::system::PreparedEmbeddedBoundaryMode::inactive)
    return measure;
  measure.active_cells = &embedded->active_mask();
  if (embedded->mode() == runtime::system::PreparedEmbeddedBoundaryMode::cut_cell)
    measure.inverse_volume_fraction = &embedded->volume_fraction();
  return measure;
}

template <int Dim, class Implementation>
const MultiFab<Dim>& embedded_boundary_output_field(const Implementation& implementation,
                                                    std::string_view name) {
  if (!implementation.embedded_boundary_)
    throw std::runtime_error("System has no prepared embedded-boundary output sidecar");
  if (name == "pops_active")
    return implementation.embedded_boundary_->active_mask();
  if (name == "pops_phi")
    return implementation.embedded_boundary_->phi();
  if (name == "pops_kappa")
    return implementation.embedded_boundary_->volume_fraction();
  throw std::invalid_argument(
      "System embedded-boundary output name must be pops_active, pops_phi, or pops_kappa");
}

template <int Dim, class Species>
void require_recoverable_candidate(const Species& state, const MultiFab<Dim>& candidate,
                                   std::string_view operation) {
  if (candidate.layout() != state.U.layout() ||
      candidate.distribution() != state.U.distribution() ||
      candidate.local_rank() != state.U.local_rank() || candidate.ncomp() != state.U.ncomp() ||
      candidate.ghosts() != state.U.ghosts())
    throw std::invalid_argument(std::string(operation) + ": candidate layout differs from block");
  if (all_reduce_max(state.cons_to_prim ? 0L : 1L) != 0)
    throw std::runtime_error(std::string(operation) +
                             ": block has no prepared variable-recovery authority");

  std::vector<double> conservative(static_cast<std::size_t>(state.ncomp));
  std::vector<double> primitive(static_cast<std::size_t>(state.ncomp));
  long failures = 0;
  for (std::size_t local = 0; local < candidate.local_size(); ++local) {
    const Fab<Dim>& fab = candidate.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for_each_host_index(fab.box(), [&](const Index<Dim>& index, std::size_t) {
      for (int component = 0; component < state.ncomp; ++component)
        conservative[static_cast<std::size_t>(component)] =
            static_cast<double>(host(storage_ordinal(fab, index, component)));
      try {
        const RecoveryReport report = state.cons_to_prim(conservative.data(), primitive.data());
        const bool finite = std::all_of(conservative.begin(), conservative.end(),
                                        [](double value) { return std::isfinite(value); }) &&
                            std::all_of(primitive.begin(), primitive.end(),
                                        [](double value) { return std::isfinite(value); });
        if (!report.publication_permitted() || !finite)
          ++failures;
      } catch (...) {
        ++failures;
      }
    });
  }
  failures = all_reduce_sum(failures);
  if (failures != 0)
    throw std::runtime_error(std::string(operation) +
                             ": variable recovery rejected the candidate (failed cells=" +
                             std::to_string(failures) + ")");
}

template <int Dim, class Species>
void publish_recovered_candidate(Species& state, MultiFab<Dim>& candidate,
                                 std::string_view operation) {
  require_recoverable_candidate<Dim>(state, candidate, operation);
  copy_valid(candidate, state.U);
}

template <int Dim>
void require_exact_field_evaluation_request(
    const runtime::multiblock::BoundaryEvaluationPoint& point, std::string_view provider_slot,
    std::string_view request_kind) {
  const bool invalid =
      request_kind.empty() || provider_slot.empty() || point.clock.empty() || point.tick < 0 ||
      point.level != 0 || point.substep < 0 || point.stage < 0 || !std::isfinite(point.dt) ||
      point.dt <= 0.0 || !std::isfinite(point.physical_time) ||
      point.stage_fraction < amr::Rational(0, 1) || amr::Rational(1, 1) < point.stage_fraction;
  if (all_reduce_max(invalid ? 1L : 0L) != 0)
    throw std::invalid_argument(
        "System exact field evaluation requires one complete level-zero point and provider slot");
  ExactContractBuilder request;
  request.text("pops.system.exact-field-evaluation")
      .scalar(std::uint32_t{1})
      .text(request_kind)
      .text(provider_slot)
      .text(point.clock)
      .scalar(point.tick)
      .scalar(static_cast<std::int32_t>(point.level))
      .scalar(static_cast<std::int32_t>(point.substep))
      .scalar(static_cast<std::int32_t>(point.stage))
      .scalar(point.stage_fraction.numerator)
      .scalar(point.stage_fraction.denominator)
      .scalar(point.dt)
      .scalar(point.physical_time);
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"system-exact-field-evaluation", std::move(request).release()}}))
    throw std::invalid_argument(
        "System exact field evaluation point differs between communicator ranks");
}

template <int Dim>
elliptic::nd::CartesianPoissonOptions<Dim> poisson_options(const BoundaryTopology<Dim>& topology,
                                                           std::string_view mode,
                                                           double relative_tolerance,
                                                           double absolute_tolerance,
                                                           int maximum_iterations) {
  elliptic::nd::CartesianBoundaryKind physical = elliptic::nd::CartesianBoundaryKind::dirichlet;
  if (mode == "neumann")
    physical = elliptic::nd::CartesianBoundaryKind::neumann;
  else if (mode != "auto" && mode != "dirichlet" && mode != "periodic")
    throw std::invalid_argument("System Poisson boundary mode is unknown");
  if (mode == "periodic" && topology.periodic_pair_count() != static_cast<std::size_t>(Dim))
    throw std::invalid_argument(
        "System periodic Poisson requires every exact topology axis to be periodic");
  auto result = elliptic::nd::CartesianPoissonOptions<Dim>::from_topology(topology, physical);
  result.relative_tolerance = static_cast<Real>(relative_tolerance);
  result.absolute_tolerance = static_cast<Real>(absolute_tolerance);
  result.maximum_iterations = maximum_iterations;
  return result;
}

template <int Dim, class Implementation>
std::shared_ptr<runtime::system::ExactNamedField<Dim>> prepare_default_field(
    Implementation& implementation) {
  if (implementation.default_field_)
    return implementation.default_field_;
  runtime::field::NamedFieldOutput<Dim> output(static_cast<std::size_t>(Dim + 1), 1);
  const BoundaryTopology<Dim> topology =
      BoundaryTopology<Dim>::axis_periodic(implementation.periodicity);
  const auto operator_options =
      poisson_options(topology, implementation.poisson_bc_, implementation.poisson_rel_tol_,
                      implementation.poisson_abs_tol_, implementation.poisson_max_iterations_);
  auto prepared = std::make_shared<runtime::system::ExactNamedField<Dim>>(
      "pops.system.default-field", implementation.sp.empty() ? "system" : implementation.sp[0].name,
      output, implementation.geom, implementation.ba, implementation.dm, implementation.local_rank,
      topology, operator_options, implementation.sp.size());
  bool has_rhs = false;
  for (std::size_t block = 0; block < implementation.sp.size(); ++block) {
    if (!implementation.sp[block].add_poisson_rhs)
      continue;
    prepared->add_rhs(block, implementation.sp[block].add_poisson_rhs, Real(1));
    has_rhs = true;
  }
  if (!has_rhs)
    throw std::runtime_error("System default elliptic field has no prepared RHS provider");
  const FieldNullspaceProviderSelection nullspace_selection{
      implementation.default_nullspace_provider_identity_,
      implementation.default_nullspace_options_};
  prepared->install_nullspace(
      implementation.prepare_uniform_field_nullspace(
          "pops.system.default-field", "pops.system.default-uniform-topology", nullspace_selection,
          operator_options, prepared->accepted_potential(), false),
      PreparedVectorDistribution<Dim>::distributed());
  implementation.default_field_ = prepared;
  return prepared;
}

template <int Dim, class Blocks>
std::vector<const MultiFab<Dim>*> select_states(
    const Blocks& blocks, const std::vector<const MultiFab<Dim>*>& overrides) {
  if (!overrides.empty() && overrides.size() != blocks.size())
    throw std::invalid_argument("System field stage vector does not cover every block");
  std::vector<const MultiFab<Dim>*> result;
  result.reserve(blocks.size());
  for (std::size_t block = 0; block < blocks.size(); ++block)
    result.push_back(overrides.empty() || overrides[block] == nullptr ? &blocks[block].U
                                                                      : overrides[block]);
  return result;
}

template <int Dim>
struct PreparedBoundaryContext {
  std::vector<const MultiFab<Dim>*> states;
  std::vector<FieldDistribution> state_distributions;
  std::vector<std::string> state_identities;
  std::vector<const MultiFab<Dim>*> fields;
  std::vector<FieldDistribution> field_distributions;
  std::vector<std::string> field_identities;
  const std::vector<Real>* parameters = nullptr;
  FieldLogicalTimePoint point{};
  FieldBoundaryFailure<Dim> failure{};

  FieldBoundaryExecutionContext<Dim> view() {
    if (states.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        fields.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        (parameters != nullptr &&
         parameters->size() > static_cast<std::size_t>(std::numeric_limits<int>::max())))
      throw std::overflow_error("System field boundary dependency pack exceeds native int");
    return {point,
            states.empty() ? nullptr : states.data(),
            state_distributions.empty() ? nullptr : state_distributions.data(),
            state_identities.empty() ? nullptr : state_identities.data(),
            static_cast<int>(states.size()),
            fields.empty() ? nullptr : fields.data(),
            field_distributions.empty() ? nullptr : field_distributions.data(),
            field_identities.empty() ? nullptr : field_identities.data(),
            static_cast<int>(fields.size()),
            parameters,
            parameters == nullptr ? 0 : static_cast<int>(parameters->size()),
            &failure};
  }
};

template <int Dim, class Implementation>
std::optional<PreparedBoundaryContext<Dim>> prepare_boundary_context(
    Implementation& implementation, const std::string& provider_slot,
    const std::vector<const MultiFab<Dim>*>& selected_states) {
  const auto plan_entry = implementation.field_plans_.find(provider_slot);
  if (plan_entry == implementation.field_plans_.end() || !plan_entry->second.boundary_kernel)
    return std::nullopt;
  const auto& plan = plan_entry->second;
  if (selected_states.size() != implementation.sp.size())
    throw std::invalid_argument(
        "System dynamic field boundary state selection does not cover every block");

  PreparedBoundaryContext<Dim> prepared;
  prepared.parameters = &plan.boundary_parameters;
  if (plan.boundary_point)
    prepared.point = *plan.boundary_point;
  prepared.states.reserve(plan.boundary_state_blocks.size());
  prepared.state_distributions.reserve(plan.boundary_state_blocks.size());
  prepared.state_identities.reserve(plan.boundary_state_blocks.size());
  for (std::size_t dependency = 0; dependency < plan.boundary_state_blocks.size(); ++dependency) {
    const int block = implementation.index(plan.boundary_state_blocks[dependency]);
    const auto* state = selected_states[static_cast<std::size_t>(block)];
    const int component = plan.boundary_state_components[dependency];
    if (state == nullptr || component < 0 || component >= state->ncomp())
      throw std::invalid_argument(
          "System dynamic field boundary state dependency is not materialized exactly");
    prepared.states.push_back(state);
    prepared.state_distributions.push_back(FieldDistribution::Distributed);
    prepared.state_identities.push_back(
        implementation.sp[static_cast<std::size_t>(block)].state_identity);
  }

  prepared.fields.reserve(plan.boundary_field_blocks.size());
  prepared.field_distributions.reserve(plan.boundary_field_blocks.size());
  prepared.field_identities.reserve(plan.boundary_field_blocks.size());
  for (std::size_t dependency = 0; dependency < plan.boundary_field_blocks.size(); ++dependency) {
    if (plan.boundary_field_components[dependency] != 0)
      throw std::invalid_argument(
          "System scalar field boundary dependency must select component zero");
    const auto dependency_plan = std::find_if(
        implementation.field_plans_.begin(), implementation.field_plans_.end(),
        [&](const auto& candidate) {
          return candidate.second.output_block == plan.boundary_field_blocks[dependency] &&
                 candidate.second.output_key == plan.boundary_field_keys[dependency];
        });
    if (dependency_plan == implementation.field_plans_.end())
      throw std::runtime_error(
          "System dynamic boundary names an unknown owner-qualified field dependency");
    const auto field = implementation.named_fields_.find(dependency_plan->first);
    if (field == implementation.named_fields_.end())
      throw std::logic_error(
          "System dynamic boundary field dependency has not been materialized before its use");
    prepared.fields.push_back(&field->second->dependency_potential());
    prepared.field_distributions.push_back(FieldDistribution::Distributed);
    prepared.field_identities.push_back(field->second->identity());
  }
  return prepared;
}

template <int Dim, class Implementation>
SolveReport solve_field_candidate(
    Implementation& implementation, const std::string& provider_slot,
    const std::shared_ptr<runtime::system::ExactNamedField<Dim>>& field,
    std::vector<const MultiFab<Dim>*> states) {
  auto storage = prepare_boundary_context<Dim>(implementation, provider_slot, states);
  std::optional<FieldBoundaryExecutionContext<Dim>> context;
  if (storage)
    context = storage->view();
  return field->solve_candidate(states, context ? &*context : nullptr);
}

}  // namespace

template <int Dim>
void System<Dim>::validate_program_state_publication_candidate(
    int block, const MultiFab<Dim>& candidate) const {
  if (all_reduce_max(block >= 0 && block < static_cast<int>(p_->sp.size()) ? 0L : 1L) != 0)
    throw std::out_of_range(
        "System Program state publication block index differs across communicator ranks");
  require_recoverable_candidate<Dim>(p_->sp[static_cast<std::size_t>(block)], candidate,
                                     "System Program terminal state publication");
}

template <int Dim>
void System<Dim>::set_density(const std::string& name, const std::vector<double>& density_values) {
  typename Impl::Species& block = p_->find(name);
  const std::size_t cells = checked_cell_count(p_->dom);
  if (density_values.size() != cells)
    throw std::invalid_argument("System::set_density payload does not match the exact domain");
  std::vector<double> state(static_cast<std::size_t>(block.ncomp) * cells, 0.0);
  std::copy(density_values.begin(), density_values.end(), state.begin());
  if (block.ncomp == Dim + 2) {
    const double denominator = block.gamma - 1.0;
    if (!(denominator > 0.0))
      throw std::invalid_argument("System::set_density requires gamma > 1 for an energy state");
    for (std::size_t cell = 0; cell < cells; ++cell)
      state[static_cast<std::size_t>(block.ncomp - 1) * cells + cell] =
          density_values[cell] / denominator;
  }
  write_global(block.U, p_->dom, state, block.ncomp);
}

template <int Dim>
void System<Dim>::set_primitive_state(const std::string& name,
                                      const std::vector<double>& primitive) {
  typename Impl::Species& block = p_->find(name);
  const std::size_t cells = checked_cell_count(p_->dom);
  if (primitive.size() != static_cast<std::size_t>(block.ncomp) * cells)
    throw std::invalid_argument(
        "System::set_primitive_state payload does not match the exact state shape");
  if (!block.prim_to_cons || !block.cons_to_prim)
    throw std::runtime_error(
        "System::set_primitive_state requires both prepared variable-conversion authorities");
  std::vector<double> conservative(primitive.size(), 0.0);
  std::vector<double> input(static_cast<std::size_t>(block.ncomp));
  std::vector<double> output(static_cast<std::size_t>(block.ncomp));
  std::vector<double> recovered(static_cast<std::size_t>(block.ncomp));
  long failures = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    for (int component = 0; component < block.ncomp; ++component)
      input[static_cast<std::size_t>(component)] =
          primitive[static_cast<std::size_t>(component) * cells + cell];
    try {
      block.prim_to_cons(input.data(), output.data());
      const RecoveryReport report = block.cons_to_prim(output.data(), recovered.data());
      bool finite = report.publication_permitted();
      for (double value : output)
        finite = finite && std::isfinite(value);
      if (!finite) {
        ++failures;
        continue;
      }
      for (int component = 0; component < block.ncomp; ++component)
        conservative[static_cast<std::size_t>(component) * cells + cell] =
            output[static_cast<std::size_t>(component)];
    } catch (...) {
      ++failures;
    }
  }
  if (all_reduce_max(failures) != 0)
    throw std::runtime_error(
        "System::set_primitive_state variable recovery rejected the candidate");
  write_global(block.U, p_->dom, conservative, block.ncomp);
}

template <int Dim>
std::vector<double> System<Dim>::get_primitive_state(const std::string& name) {
  typename Impl::Species& block = p_->find(name);
  if (!block.batch_cons_to_prim)
    throw std::runtime_error(
        "System::get_primitive_state requires a generation-qualified batch recovery provider");
  const std::vector<double> conservative = gather_global(block.U, p_->dom, block.ncomp);
  std::vector<double> primitive;
  const UniformRecoveryBatchReport report = block.batch_cons_to_prim(conservative, primitive);
  if (!report.publication_permitted())
    throw std::runtime_error("System::get_primitive_state batch variable recovery failed");
  return primitive;
}

template <int Dim>
void System<Dim>::set_poisson(const std::string& rhs, const std::string& solver,
                              const std::string& bc, double abs_tol, double rel_tol,
                              int max_iterations) {
  require_assembling(p_->lifecycle_, "set_poisson");
  if (rhs != "charge_density" && rhs != "composite")
    throw std::invalid_argument(
        "System exact Poisson supports charge_density or composite RHS routes");
  if (solver == "geometric_mg")
    throw std::invalid_argument(
        "GeometricMG is reserved for AMR MG/FAC; uniform System uses cartesian_cg");
  if (solver != "cartesian_cg")
    throw std::invalid_argument(
        "System exact ND Poisson supports only the cartesian_cg provider route");
  if (!std::isfinite(abs_tol) || abs_tol < 0.0 || !std::isfinite(rel_tol) || rel_tol <= 0.0 ||
      max_iterations < 1)
    throw std::invalid_argument("System exact Poisson controls are invalid");
  const BoundaryTopology<Dim> topology = BoundaryTopology<Dim>::axis_periodic(p_->periodicity);
  (void)poisson_options(topology, bc, rel_tol, abs_tol, max_iterations);
  p_->poisson_solver_ = solver;
  p_->poisson_bc_ = bc;
  p_->poisson_abs_tol_ = abs_tol;
  p_->poisson_rel_tol_ = rel_tol;
  p_->poisson_max_iterations_ = max_iterations;
  p_->default_field_.reset();
}

template <int Dim>
SolveReport System<Dim>::solve_fields_in_place_() {
  const auto field = prepare_default_field<Dim>(*p_);
  p_->active_field_ = field;
  return solve_field_candidate<Dim>(*p_, field->identity(), field, select_states<Dim>(p_->sp, {}));
}

template <int Dim>
SolveReport System<Dim>::solve_fields_from_state_in_place_(int block_index,
                                                           const MultiFab<Dim>& stage) {
  if (block_index < 0 || block_index >= static_cast<int>(p_->sp.size()))
    throw std::out_of_range("System field stage block index is outside the registry");
  std::vector<const MultiFab<Dim>*> overrides(p_->sp.size(), nullptr);
  overrides[static_cast<std::size_t>(block_index)] = &stage;
  const auto field = prepare_default_field<Dim>(*p_);
  p_->active_field_ = field;
  return solve_field_candidate<Dim>(*p_, field->identity(), field,
                                    select_states<Dim>(p_->sp, overrides));
}

template <int Dim>
SolveReport System<Dim>::solve_fields_from_state_at_in_place_(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
    int block_index, const MultiFab<Dim>& stage) {
  require_exact_field_evaluation_request<Dim>(point, provider_slot, "single-stage");
  if (provider_slot == "pops.system.default-field")
    return solve_fields_from_state_in_place_(block_index, stage);
  return solve_fields_from_state_in_place_(provider_slot, block_index, stage);
}

template <int Dim>
SolveReport System<Dim>::solve_fields_from_blocks_in_place_(
    const std::vector<const MultiFab<Dim>*>& stages) {
  const auto field = prepare_default_field<Dim>(*p_);
  p_->active_field_ = field;
  return solve_field_candidate<Dim>(*p_, field->identity(), field,
                                    select_states<Dim>(p_->sp, stages));
}

template <int Dim>
SolveReport System<Dim>::solve_fields_from_state_in_place_(const std::string& field,
                                                           int block_index,
                                                           const MultiFab<Dim>& stage) {
  if (block_index < 0 || block_index >= static_cast<int>(p_->sp.size()))
    throw std::out_of_range("System named-field block index is outside the registry");
  std::vector<const MultiFab<Dim>*> stages(p_->sp.size(), nullptr);
  stages[static_cast<std::size_t>(block_index)] = &stage;
  return solve_fields_from_blocks_in_place_(field, stages);
}

template <int Dim>
SolveReport System<Dim>::solve_fields_from_blocks_in_place_(
    const std::string& field, const std::vector<const MultiFab<Dim>*>& stages) {
  const std::string provider_slot = p_->resolve_named_field_slot(field);
  const auto found = p_->named_fields_.find(provider_slot);
  if (found == p_->named_fields_.end())
    throw std::out_of_range("System named elliptic field is not registered: " + field);
  p_->active_field_ = found->second;
  return solve_field_candidate<Dim>(*p_, provider_slot, found->second,
                                    select_states<Dim>(p_->sp, stages));
}

template <int Dim>
SolveReport System<Dim>::solve_fields_from_blocks_at_in_place_(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& field,
    const std::vector<const MultiFab<Dim>*>& stages) {
  require_exact_field_evaluation_request<Dim>(point, field, "simultaneous-stages");
  return solve_fields_from_blocks_in_place_(field, stages);
}

template <int Dim>
SolveOutcome System<Dim>::run_field_publication_outcome_(
    const std::function<SolveReport()>& solve) {
  begin_field_publication_outcome_();
  SolveReport report;
  std::exception_ptr local_error;
  try {
    report = solve();
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L) != 0) {
    rollback_field_publication_transaction();
    if (n_ranks() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("System exact field solver failed on at least one MPI rank");
  }
  return stage_field_publication_outcome_(std::move(report));
}

template <int Dim>
void System<Dim>::begin_field_publication_outcome_() {
  const bool active = p_->active_field_ || p_->active_field_provider_candidate_ ||
                      p_->active_field_auxiliary_publication_ ||
                      !p_->active_field_stale_auxiliary_providers_.empty();
  if (all_reduce_max(active ? 1L : 0L) != 0)
    throw std::logic_error(
        "System field solves are sequential until their prior SolveOutcome is consumed");
}

template <int Dim>
SolveOutcome System<Dim>::stage_field_publication_outcome_(SolveReport report) {
  if (!solve_report_is_publishable(report, p_->active_field_
                                               ? p_->active_field_->maximum_iterations()
                                               : std::numeric_limits<int>::max())) {
    rollback_field_publication_transaction();
    throw std::runtime_error("System exact field solver published a malformed SolveReport");
  }
  ExactSolveReportConsensusScratch consensus;
  if (!consensus.agrees(report)) {
    rollback_field_publication_transaction();
    throw std::runtime_error("System exact field solver report differs between MPI ranks");
  }
  if (!report.solved_value_available()) {
    rollback_field_publication_transaction();
    return SolveOutcome::collective_world(std::move(report));
  }
  try {
    stage_field_publication_candidate();
  } catch (...) {
    rollback_field_publication_transaction();
    throw;
  }
  return SolveOutcome::collective_world(
      std::move(report),
      SolveOutcome::PublicationHooks{
          this,
          [](void* context) noexcept {
            static_cast<System<Dim>*>(context)->accept_field_publication_candidate();
          },
          nullptr,
          [](void* context) noexcept {
            try {
              static_cast<System<Dim>*>(context)->rollback_field_publication_transaction();
            } catch (...) {
              std::terminate();
            }
          },
          {},
          [](void* context) {
            static_cast<System<Dim>*>(context)->validate_field_publication_candidate();
          }});
}

template <int Dim>
void System<Dim>::prepare_default_field_publication_storage_() {
  (void)prepare_default_field<Dim>(*p_);
}

template <int Dim>
void System<Dim>::prepare_named_field_publication_storage_(const std::string& field) {
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"system-named-field-publication", field}}))
    throw std::invalid_argument("System named field request differs between MPI ranks");
  if (p_->named_fields_.find(p_->resolve_named_field_slot(field)) == p_->named_fields_.end())
    throw std::out_of_range("System named elliptic field is not registered: " + field);
}

template <int Dim>
SolveOutcome System<Dim>::solve_fields() {
  prepare_default_field_publication_storage_();
  return run_field_publication_outcome_([this] { return solve_fields_in_place_(); });
}

template <int Dim>
SolveOutcome System<Dim>::solve_fields_from_state(int block_index, const MultiFab<Dim>& stage) {
  prepare_default_field_publication_storage_();
  return run_field_publication_outcome_([this, block_index, &stage] {
    return solve_fields_from_state_in_place_(block_index, stage);
  });
}

template <int Dim>
SolveOutcome System<Dim>::solve_fields_from_state_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
    int block_index, const MultiFab<Dim>& stage) {
  return run_field_publication_outcome_([this, &point, &provider_slot, block_index, &stage] {
    return solve_fields_from_state_at_in_place_(point, provider_slot, block_index, stage);
  });
}

template <int Dim>
SolveOutcome System<Dim>::solve_fields_from_blocks(
    const std::vector<const MultiFab<Dim>*>& stages) {
  prepare_default_field_publication_storage_();
  return run_field_publication_outcome_(
      [this, &stages] { return solve_fields_from_blocks_in_place_(stages); });
}

template <int Dim>
SolveOutcome System<Dim>::solve_fields_from_state(const std::string& field, int block_index,
                                                  const MultiFab<Dim>& stage) {
  prepare_named_field_publication_storage_(field);
  return run_field_publication_outcome_([this, &field, block_index, &stage] {
    return solve_fields_from_state_in_place_(field, block_index, stage);
  });
}

template <int Dim>
SolveOutcome System<Dim>::solve_fields_from_blocks(
    const std::string& field, const std::vector<const MultiFab<Dim>*>& stages) {
  prepare_named_field_publication_storage_(field);
  return run_field_publication_outcome_(
      [this, &field, &stages] { return solve_fields_from_blocks_in_place_(field, stages); });
}

template <int Dim>
void System<Dim>::begin_field_publication_transaction() {
  if (p_->active_field_ || p_->active_field_provider_candidate_ ||
      p_->active_field_auxiliary_publication_ ||
      !p_->active_field_stale_auxiliary_providers_.empty())
    throw std::logic_error("System field publication transaction is already active");
}

template <int Dim>
void System<Dim>::stage_field_publication_candidate() {
  std::vector<runtime::system::AuxiliaryComponentKey> output_keys;
  std::string output_contract;
  std::exception_ptr preflight_error;
  try {
    if (!p_->active_field_)
      throw std::logic_error("System field solve did not stage an exact provider candidate");
    p_->active_field_->validate_candidate();
    output_keys = p_->active_field_->output_keys();
    if (!output_keys.empty() && (!p_->auxiliary_registry_.sealed() || !p_->provider_carrier_))
      throw std::logic_error("System named field output requires a sealed provider carrier");
    if (p_->active_field_provider_candidate_ || p_->active_field_auxiliary_publication_)
      throw std::logic_error("System field provider publication candidate is already active");
    ExactContractBuilder exact;
    exact.text("pops.system-field-output-publication")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(p_->active_field_->identity())
        .sequence(output_keys,
                  [](ExactContractBuilder& item, const auto& key) { key.serialize_exact(item); });
    output_contract = std::move(exact).release();
  } catch (...) {
    preflight_error = std::current_exception();
  }
  runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
      preflight_error, nullptr,
      "System field-output preflight failed collectively before transport preparation");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("pops.system-field-output-publication"), output_contract}}))
    throw std::invalid_argument(
        "System field-output publication contract differs between communicator ranks");
  if (output_keys.empty())
    return;

  std::vector<std::string> provider_identities;
  try {
    std::exception_ptr materialization_error;
    try {
      provider_identities.reserve(output_keys.size());
      for (const auto& key : output_keys) {
        const auto& provider = p_->auxiliary_registry_.provider_for_key(key);
        if (provider.kind() != runtime::system::AuxiliaryProviderKind::field_output)
          throw std::logic_error("System field output lost its field-output provider authority");
        if (std::find(provider_identities.begin(), provider_identities.end(),
                      provider.identity()) == provider_identities.end())
          provider_identities.push_back(provider.identity());
      }
      p_->active_field_provider_candidate_ = *p_->provider_carrier_;
      copy_field_outputs_to_provider_candidate(p_->active_field_->candidate_outputs(), output_keys,
                                               p_->auxiliary_registry_,
                                               *p_->active_field_provider_candidate_);
    } catch (...) {
      materialization_error = std::current_exception();
    }
    runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
        materialization_error, nullptr,
        "System field-output materialization failed collectively before transport preparation");

    const auto evaluation_point = field_output_evaluation_point<Dim>(*p_);
    std::exception_ptr publication_error;
    try {
      p_->active_field_auxiliary_publication_.emplace(
          p_->auxiliary_registry_.begin_external_publication(evaluation_point,
                                                             provider_identities));
      for (const std::string& identity : provider_identities)
        p_->active_field_auxiliary_publication_->stage_external(identity);
    } catch (...) {
      publication_error = std::current_exception();
    }
    runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
        publication_error, nullptr,
        "System field-output publication staging failed collectively before transport preparation");
    if (!p_->auxiliary_ghost_lane_)
      p_->auxiliary_ghost_lane_.emplace(
          ExecutionLane::duplicate_world_collectively("pops.system.auxiliary-ghosts"));
    if (!p_->auxiliary_ghost_transport_)
      p_->auxiliary_ghost_transport_.emplace(runtime::system::prepare_auxiliary_ghost_transport(
          *p_->provider_carrier_, p_->auxiliary_registry_, p_->dom, p_->geom,
          BoundaryTopology<Dim>::axis_periodic(p_->periodicity), &*p_->auxiliary_ghost_lane_));
    const auto refresh_candidate_ghosts = [&] {
      p_->auxiliary_ghost_transport_->execute(*p_->active_field_provider_candidate_);
    };
    refresh_candidate_ghosts();
    p_->active_field_auxiliary_publication_->launch_ready_native(
        {&*p_->provider_carrier_, &*p_->active_field_provider_candidate_},
        [&](const auto&, std::exception_ptr local_error) {
          runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
              local_error, &*p_->auxiliary_ghost_lane_,
              "System field-output auxiliary launch failed collectively before ghost fill");
          refresh_candidate_ghosts();
        });
    std::exception_ptr fence_error;
    try {
      Kokkos::fence();
    } catch (...) {
      fence_error = std::current_exception();
    }
    runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
        fence_error, &*p_->auxiliary_ghost_lane_,
        "System field-output device fence failed collectively");
    runtime::system::require_finite_auxiliary_groups(*p_->active_field_provider_candidate_,
                                                     &*p_->auxiliary_ghost_lane_,
                                                     "System field-output publication");
    p_->active_field_auxiliary_publication_->validate_complete();
    p_->active_field_stale_auxiliary_providers_ =
        p_->auxiliary_registry_.dependent_provider_identities(provider_identities);
  } catch (...) {
    if (p_->active_field_auxiliary_publication_)
      p_->active_field_auxiliary_publication_->reject();
    p_->active_field_auxiliary_publication_.reset();
    p_->active_field_provider_candidate_.reset();
    p_->active_field_stale_auxiliary_providers_.clear();
    throw;
  }
}

template <int Dim>
void System<Dim>::validate_field_publication_candidate() {
  if (!p_->active_field_)
    throw std::logic_error("System field publication candidate is absent");
  p_->active_field_->validate_candidate();
  if (p_->active_field_auxiliary_publication_) {
    if (!p_->active_field_provider_candidate_)
      throw std::logic_error("System field provider candidate storage is absent");
    p_->active_field_auxiliary_publication_->validate_complete();
  } else if (p_->active_field_provider_candidate_) {
    throw std::logic_error("System field provider metadata candidate is absent");
  }
}

template <int Dim>
void System<Dim>::accept_field_publication_candidate() noexcept {
  if (!p_->active_field_)
    std::terminate();
  try {
    p_->active_field_->accept_candidate();
    if (p_->active_field_auxiliary_publication_) {
      if (!p_->active_field_provider_candidate_ || !p_->provider_carrier_)
        std::terminate();
      p_->active_field_auxiliary_publication_->accept();
      p_->provider_carrier_.swap(p_->active_field_provider_candidate_);
      for (const std::string& identity : p_->active_field_stale_auxiliary_providers_)
        if (std::find(p_->dirty_auxiliary_providers_.begin(), p_->dirty_auxiliary_providers_.end(),
                      identity) == p_->dirty_auxiliary_providers_.end())
          p_->dirty_auxiliary_providers_.push_back(identity);
    }
    p_->active_field_auxiliary_publication_.reset();
    p_->active_field_provider_candidate_.reset();
    p_->active_field_stale_auxiliary_providers_.clear();
    p_->active_field_.reset();
  } catch (...) {
    std::terminate();
  }
}

template <int Dim>
void System<Dim>::rollback_field_publication_transaction() {
  if (p_->active_field_auxiliary_publication_)
    p_->active_field_auxiliary_publication_->reject();
  p_->active_field_auxiliary_publication_.reset();
  p_->active_field_provider_candidate_.reset();
  p_->active_field_stale_auxiliary_providers_.clear();
  if (!p_->active_field_)
    return;
  p_->active_field_->reject_candidate();
  p_->active_field_.reset();
}

template <int Dim>
bool System<Dim>::field_publication_transaction_active_() const noexcept {
  return static_cast<bool>(p_->active_field_) ||
         static_cast<bool>(p_->active_field_provider_candidate_) ||
         static_cast<bool>(p_->active_field_auxiliary_publication_) ||
         !p_->active_field_stale_auxiliary_providers_.empty();
}

template <int Dim>
void System<Dim>::set_potential(const std::vector<double>& potential_values) {
  auto field = prepare_default_field<Dim>(*p_);
  write_global(field->accepted_potential_for_restore(), p_->dom, potential_values, 1);
}

template <int Dim>
std::vector<std::string> System<Dim>::field_provider_slots() const {
  std::vector<std::string> result;
  if (p_->default_field_)
    result.push_back("pops.system.default-field");
  result.reserve(result.size() + p_->named_fields_.size());
  for (const auto& [identity, field] : p_->named_fields_) {
    (void)field;
    result.push_back(identity);
  }
  return result;
}

template <int Dim>
void System<Dim>::set_field_potential(const std::string& provider_slot,
                                      const std::vector<double>& potential_values) {
  std::shared_ptr<runtime::system::ExactNamedField<Dim>> field;
  if (provider_slot == "pops.system.default-field")
    field = prepare_default_field<Dim>(*p_);
  else {
    const auto found = p_->named_fields_.find(provider_slot);
    if (found == p_->named_fields_.end())
      throw std::out_of_range("System field provider slot is unknown: " + provider_slot);
    field = found->second;
  }
  write_global(field->accepted_potential_for_restore(), p_->dom, potential_values, 1);
}

template <int Dim>
std::vector<double> System<Dim>::eval_rhs(const std::string& name) {
  typename Impl::Species& block = p_->find(name);
  MultiFab<Dim> residual(p_->ba, p_->dm, p_->local_rank, block.ncomp, Extent<Dim>{});
  block_rhs_into(p_->index(name), block.U, residual);
  return gather_local_compact(residual, block.ncomp);
}

template <int Dim>
double System<Dim>::reduce_component(const std::string& block_name, const std::string& kind,
                                     int component) const {
  const typename Impl::Species& block = p_->find(block_name);
  if (component < 0 || component >= block.ncomp)
    throw std::out_of_range("System reduction component is outside the block state");
  const MultiFab<Dim>& values = block.U;
  const RelativeCellMeasure<Dim> measure = embedded_cell_measure<Dim>(p_->embedded_boundary_);
  if (kind == "sum")
    return static_cast<double>(reduce_sum(values, component, measure));
  if (kind == "min")
    return static_cast<double>(reduce_min(values, component, measure));
  if (kind == "max")
    return static_cast<double>(reduce_max(values, component, measure));
  if (kind == "abs_sum")
    return static_cast<double>(reduce_abs_sum(values, component, measure));
  if (kind == "sum_sq")
    return static_cast<double>(dot(values, values, component, measure));
  if (kind == "abs_max")
    return static_cast<double>(reduce_norm_inf(values, component, measure));
  if (kind == "sum_all" || kind == "abs_sum_all" || kind == "abs_max_all") {
    double result = 0.0;
    for (int current = 0; current < block.ncomp; ++current) {
      const double value = kind == "sum_all" ? static_cast<double>(reduce_sum(values, current))
                           : kind == "abs_sum_all"
                               ? static_cast<double>(reduce_abs_sum(values, current))
                               : static_cast<double>(reduce_norm_inf(values, current));
      result = kind == "abs_max_all" ? std::max(result, value) : result + value;
    }
    return result;
  }
  if (kind == "sum_sq_all")
    return static_cast<double>(dot_all(values, values));
  throw std::invalid_argument("System reduction kind is unknown: " + kind);
}

template <int Dim>
MultiFab<Dim> System<Dim>::alloc_scalar_field(int components, int ghosts) {
  if (components < 1 || ghosts < 0)
    throw std::invalid_argument("System scalar-field allocation shape is invalid");
  Extent<Dim> ghost_extent{};
  for (int axis = 0; axis < Dim; ++axis)
    ghost_extent[axis] = ghosts;
  return MultiFab<Dim>(p_->ba, p_->dm, p_->local_rank, components, ghost_extent);
}

template <int Dim>
MultiFab<Dim>& System<Dim>::register_history(const std::string& name, int lag, int ncomp, int owner,
                                             const std::string& state_identity,
                                             const std::string& space_identity,
                                             const std::string& clock_identity,
                                             const std::string& interpolation_identity) {
  if (lag < 1 || p_->sp.empty())
    throw std::invalid_argument("System history requires a block and lag >= 1");
  const bool qualified = owner >= 0;
  if (qualified &&
      (owner >= static_cast<int>(p_->sp.size()) || state_identity.empty() ||
       space_identity.empty() || clock_identity.empty() || interpolation_identity.empty()))
    throw std::invalid_argument("System qualified history identity is incomplete");
  const int depth = lag + 1;
  auto& histories = p_->program_.hist_;
  auto found = histories.histories.find(name);
  if (found != histories.histories.end()) {
    if ((qualified &&
         (histories.owner[name] != owner || histories.state_identity[name] != state_identity ||
          histories.space_identity[name] != space_identity ||
          histories.clock_identity[name] != clock_identity ||
          histories.interpolation_identity[name] != interpolation_identity)) ||
        (!qualified && histories.owner[name] != -1))
      throw std::invalid_argument("System history cannot be requalified");
    if (ncomp >= 1 && found->second.front().ncomp() != ncomp)
      throw std::invalid_argument("System history component count changed");
    while (static_cast<int>(found->second.size()) < depth)
      found->second.emplace_back(p_->ba, p_->dm, p_->local_rank, found->second.front().ncomp(),
                                 found->second.front().ghosts());
    histories.depth[name] = static_cast<int>(found->second.size());
    histories.slot_dt[name].resize(found->second.size(), Real(0));
    return found->second.front();
  }
  const int components =
      ncomp < 0 ? p_->sp[qualified ? static_cast<std::size_t>(owner) : 0].ncomp : ncomp;
  if (components < 1)
    throw std::invalid_argument("System history component count must be positive");
  Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    ghosts[axis] = 1;
  std::vector<MultiFab<Dim>> ring;
  ring.reserve(static_cast<std::size_t>(depth));
  for (int slot = 0; slot < depth; ++slot)
    ring.emplace_back(p_->ba, p_->dm, p_->local_rank, components, ghosts);
  auto& stored = histories.histories.emplace(name, std::move(ring)).first->second;
  histories.depth[name] = depth;
  histories.initialized[name] = false;
  histories.fill_count[name] = 0;
  histories.store_pending[name] = false;
  histories.owner[name] = qualified ? owner : -1;
  histories.slot_dt[name] = std::vector<Real>(static_cast<std::size_t>(depth), Real(0));
  if (qualified) {
    histories.state_identity[name] = state_identity;
    histories.space_identity[name] = space_identity;
    histories.clock_identity[name] = clock_identity;
    histories.interpolation_identity[name] = interpolation_identity;
  }
  return stored.front();
}

template <int Dim>
MultiFab<Dim>& System<Dim>::read_history(const std::string& name, int lag) {
  auto& histories = p_->program_.hist_;
  const auto found = histories.histories.find(name);
  if (found == histories.histories.end() || lag < 0 || lag >= histories.depth[name])
    throw std::out_of_range("System history read is outside the registered ring");
  if (!histories.initialized[name])
    throw std::runtime_error("System history was requested before its first store");
  return found->second[static_cast<std::size_t>(lag)];
}

template <int Dim>
std::vector<double> System<Dim>::get_state(const std::string& name) {
  const typename Impl::Species& block = p_->find(name);
  return gather_local_compact(block.U, block.ncomp);
}

template <int Dim>
void System<Dim>::set_state(const std::string& name, const std::vector<double>& state) {
  typename Impl::Species& block = p_->find(name);
  write_global(block.U, p_->dom, state, block.ncomp);
}

template <int Dim>
std::int64_t System<Dim>::set_analytic_expression_state(
    const std::string& name, const std::string& space, const std::string& centering,
    const std::string& projection, const std::vector<std::vector<std::string>>& opcodes,
    const std::vector<std::vector<double>>& literals) {
  auto prepared = analytic::collectively_prepare_analytic_request(
      "System::set_analytic_expression_state",
      {{"centering", centering}, {"name", name}, {"projection", projection}, {"space", space}}, {},
      opcodes, literals, [&] {
        require_assembling(p_->lifecycle_, "set_analytic_expression_state");
        if (space != "cell" || centering != "cell" || projection != "conservative_cell_average")
          throw std::invalid_argument(
              "System analytic state requires cell conservative_cell_average projection");
        typename Impl::Species& block = p_->find(name);
        auto programs = analytic::compile_component_programs(opcodes, literals);
        if (programs.size() != static_cast<std::size_t>(block.ncomp))
          throw std::invalid_argument("System analytic expression component count differs");
        return std::pair<typename Impl::Species*, std::vector<analytic::AnalyticProgram>>{
            &block, std::move(programs)};
      });
  MultiFab<Dim> candidate(prepared.first->U.layout(), prepared.first->U.distribution(),
                          prepared.first->U.local_rank(), prepared.first->U.ncomp(),
                          prepared.first->U.ghosts());
  const std::int64_t count =
      analytic::materialize_cell_average(candidate, p_->geom, prepared.second);
  publish_recovered_candidate<Dim>(*prepared.first, candidate,
                                   "System::set_analytic_expression_state");
  return count;
}

template <int Dim>
std::int64_t System<Dim>::set_analytic_mapped_state(
    const std::string& name, const std::vector<std::vector<std::string>>& opcodes,
    const std::vector<std::vector<double>>& literals,
    const std::vector<runtime::system::AnalyticMappedInput>& inputs,
    const std::string& consumer_qid) {
  ExactContractBuilder input_contract;
  input_contract.text("pops.system.analytic-mapped-inputs")
      .scalar(std::uint32_t{1})
      .text(consumer_qid)
      .scalar(static_cast<std::uint64_t>(inputs.size()));
  for (const auto& input : inputs) {
    input_contract.scalar(static_cast<std::uint8_t>(input.kind))
        .scalar(input.state_component)
        .text(input.provider_key.owner_qid)
        .text(input.provider_key.space_kind)
        .text(input.provider_key.space_name)
        .text(input.provider_key.component);
  }
  const std::string exact_inputs = std::move(input_contract).release();
  auto prepared = analytic::collectively_prepare_analytic_request(
      "System::set_analytic_mapped_state",
      {{"consumer_qid", consumer_qid}, {"inputs", exact_inputs}, {"name", name}}, {}, opcodes,
      literals, [&] {
        require_assembling(p_->lifecycle_, "set_analytic_mapped_state");
        typename Impl::Species& block = p_->find(name);
        auto programs = analytic::compile_component_programs(opcodes, literals);
        if (consumer_qid.empty() || programs.size() != static_cast<std::size_t>(block.ncomp) ||
            inputs.size() > analytic::kAnalyticMaxStack)
          throw std::invalid_argument("System mapped analytic request shape is invalid");
        runtime::system::AuxiliaryConsumerProviderPlan<Dim> expected_plan;
        expected_plan.consumer_qid = consumer_qid;
        std::vector<runtime::system::AuxiliaryComponentKey> provider_keys;
        for (const auto& input : inputs) {
          input.validate();
          if (input.kind == runtime::system::AnalyticMappedInputKind::state_component &&
              input.state_component >= block.ncomp)
            throw std::out_of_range(
                "System mapped analytic state input component is outside the target state");
          if (input.kind != runtime::system::AnalyticMappedInputKind::provider_component)
            continue;
          if (!p_->auxiliary_registry_.sealed())
            throw std::logic_error(
                "System mapped analytic providers require a sealed auxiliary registry");
          const auto& provider = p_->auxiliary_registry_.provider_for_key(input.provider_key);
          const auto output = std::find_if(
              provider.outputs().begin(), provider.outputs().end(),
              [&](const auto& candidate) { return candidate.key == input.provider_key; });
          if (output == provider.outputs().end())
            throw std::logic_error(
                "System sealed auxiliary provider lost its declared component output");
          expected_plan.values.push_back(
              {{output->key, output->contract, output->shape}, provider_keys.size()});
          provider_keys.push_back(input.provider_key);
        }
        return std::tuple<typename Impl::Species*, std::vector<analytic::AnalyticProgram>,
                          runtime::system::AuxiliaryConsumerProviderPlan<Dim>,
                          std::vector<runtime::system::AuxiliaryComponentKey>>{
            &block, std::move(programs), std::move(expected_plan), std::move(provider_keys)};
      });
  typename Impl::Species* block = std::get<0>(prepared);
  auto& programs = std::get<1>(prepared);
  auto expected_plan = std::move(std::get<2>(prepared));
  const auto& provider_keys = std::get<3>(prepared);

  const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* resolved_plan = nullptr;
  if (!provider_keys.empty()) {
    try {
      resolved_plan = &p_->auxiliary_registry_.consumer_plan(consumer_qid);
    } catch (const std::out_of_range&) {
      install_auxiliary_consumer_plan(std::move(expected_plan));
      resolved_plan = &p_->auxiliary_registry_.consumer_plan(consumer_qid);
    }
  }
  MultiFab<Dim> seed = block->U;
  MultiFab<Dim> candidate(block->U.layout(), block->U.distribution(), block->U.local_rank(),
                          block->U.ncomp(), block->U.ghosts());
  std::vector<analytic::detail::AnalyticMappedInputField<Dim, typename MultiFab<Dim>::memory_space>>
      resolved_inputs;
  std::exception_ptr resolution_error;
  try {
    if (!provider_keys.empty() && resolved_plan == nullptr)
      throw std::logic_error("System mapped analytic provider inputs have no resolved plan");
    if (resolved_plan != nullptr && resolved_plan->value_count() != provider_keys.size())
      throw std::invalid_argument(
          "System mapped analytic consumer plan has a different provider count");
    for (std::size_t slot = 0; slot < provider_keys.size(); ++slot)
      if (resolved_plan->values[slot].consumer_slot != slot ||
          resolved_plan->values[slot].key != provider_keys[slot])
        throw std::invalid_argument(
            "System mapped analytic consumer plan differs from its exact input order");
    if (!provider_keys.empty() && !p_->provider_carrier_)
      throw std::logic_error(
          "System mapped analytic provider inputs have no accepted storage groups");
    for (const auto& key : provider_keys) {
      const auto& provider = p_->auxiliary_registry_.provider_for_key(key);
      if (!p_->auxiliary_registry_.last_accepted_point(provider.identity()))
        throw std::logic_error(
            "System mapped analytic provider input has never published an accepted generation");
      if (std::find(p_->dirty_auxiliary_providers_.begin(), p_->dirty_auxiliary_providers_.end(),
                    provider.identity()) != p_->dirty_auxiliary_providers_.end())
        throw std::logic_error(
            "System mapped analytic provider input is stale at the requested generation");
    }
    resolved_inputs.reserve(inputs.size());
    std::size_t provider_slot = 0;
    for (const auto& input : inputs) {
      if (input.kind == runtime::system::AnalyticMappedInputKind::state_component) {
        resolved_inputs.push_back({&seed, input.state_component});
        continue;
      }
      const auto& value = resolved_plan->values[provider_slot++];
      const MultiFab<Dim>* group = p_->provider_carrier_->find(value.address.group);
      if (group == nullptr || value.address.component >= static_cast<std::size_t>(group->ncomp()))
        throw std::logic_error(
            "System mapped analytic provider input lacks its resolved storage address");
      if (value.contract.centering != "cell" || value.shape.value_components != 1)
        throw std::invalid_argument(
            "System mapped analytic provider input requires a scalar cell-centred projection");
      resolved_inputs.push_back({group, static_cast<int>(value.address.component)});
    }
  } catch (...) {
    resolution_error = std::current_exception();
  }
  if (all_reduce_max(resolution_error ? 1L : 0L) != 0) {
    if (n_ranks() == 1 && resolution_error)
      std::rethrow_exception(resolution_error);
    throw std::runtime_error("System mapped analytic input resolution failed collectively");
  }
  const std::int64_t count =
      analytic::materialize_discrete_mapped_state(candidate, p_->geom, programs, resolved_inputs);
  publish_recovered_candidate<Dim>(*block, candidate, "System::set_analytic_mapped_state");
  return count;
}

template <int Dim>
std::int64_t System<Dim>::set_analytic_gaussian_state(const std::string& name,
                                                      const RealVector<Dim>& center,
                                                      double background, double amplitude,
                                                      double inverse_width) {
  require_assembling(p_->lifecycle_, "set_analytic_gaussian_state");
  typename Impl::Species& block = p_->find(name);
  if (block.ncomp != 1)
    throw std::invalid_argument("System Gaussian initializer requires a scalar block");
  MultiFab<Dim> candidate(block.U.layout(), block.U.distribution(), block.U.local_rank(), 1,
                          block.U.ghosts());
  const std::int64_t count = analytic::materialize_gaussian_cell_average(
      candidate, p_->geom, center, static_cast<Real>(background), static_cast<Real>(amplitude),
      static_cast<Real>(inverse_width));
  publish_recovered_candidate<Dim>(block, candidate, "System::set_analytic_gaussian_state");
  return count;
}

template <int Dim>
int System<Dim>::n_vars(const std::string& name) const {
  return p_->find(name).ncomp;
}

template <int Dim>
std::vector<std::string> System<Dim>::variable_names(const std::string& name,
                                                     const std::string& kind) const {
  const typename Impl::Species& block = p_->find(name);
  if (kind == "conservative")
    return block.cons_vars.names;
  if (kind == "primitive")
    return block.prim_vars.names;
  throw std::invalid_argument("System variable kind is neither conservative nor primitive");
}

template <int Dim>
std::vector<std::string> System<Dim>::variable_roles(const std::string& name,
                                                     const std::string& kind) const {
  const typename Impl::Species& block = p_->find(name);
  const VariableSet* variables = kind == "conservative" ? &block.cons_vars
                                 : kind == "primitive"  ? &block.prim_vars
                                                        : nullptr;
  if (variables == nullptr)
    throw std::invalid_argument("System variable role kind is invalid");
  std::vector<std::string> result;
  result.reserve(static_cast<std::size_t>(variables->size));
  for (int index = 0; index < variables->size; ++index)
    result.emplace_back(role_name(variables->at(index).role));
  return result;
}

template <int Dim>
double System<Dim>::block_gamma(const std::string& name) const {
  return p_->find(name).gamma;
}

template <int Dim>
double System<Dim>::mass(const std::string& name) const {
  return static_cast<double>(
      reduce_sum(p_->find(name).U, 0, embedded_cell_measure<Dim>(p_->embedded_boundary_)));
}

template <int Dim>
std::vector<double> System<Dim>::density(const std::string& name) const {
  return gather_local_compact(p_->find(name).U, 1);
}

template <int Dim>
std::vector<double> System<Dim>::potential() {
  return gather_local_compact(prepare_default_field<Dim>(*p_)->accepted_potential(), 1);
}

template <int Dim>
std::vector<double> System<Dim>::density_global(const std::string& name) const {
  return gather_global(p_->find(name).U, p_->dom, 1);
}

template <int Dim>
std::vector<double> System<Dim>::state_global(const std::string& name) const {
  const typename Impl::Species& block = p_->find(name);
  return gather_global(block.U, p_->dom, block.ncomp);
}

template <int Dim>
std::vector<double> System<Dim>::potential_global() {
  return gather_global(prepare_default_field<Dim>(*p_)->accepted_potential(), p_->dom, 1);
}

template <int Dim>
std::vector<double> System<Dim>::field_potential_global(const std::string& provider_slot) {
  if (provider_slot == "pops.system.default-field")
    return potential_global();
  const auto found = p_->named_fields_.find(provider_slot);
  if (found == p_->named_fields_.end())
    throw std::out_of_range("System field provider slot is unknown: " + provider_slot);
  return gather_global(found->second->accepted_potential(), p_->dom, 1);
}

template <int Dim>
std::vector<OutputPiece<Dim>> System<Dim>::output_state_local_pieces(const std::string& name,
                                                                     int level) const {
  if (level != 0)
    throw std::out_of_range("Uniform System output has only level zero");
  return output_local_pieces(p_->find(name).U, 0, false);
}

template <int Dim>
std::vector<OutputPiece<Dim>> System<Dim>::output_field_local_pieces(
    const std::string& provider_slot, int level) {
  if (level != 0)
    throw std::out_of_range("Uniform System output has only level zero");
  if (provider_slot == "pops.system.default-field")
    return output_local_pieces(prepare_default_field<Dim>(*p_)->accepted_potential(), 0, false);
  const auto found = p_->named_fields_.find(provider_slot);
  if (found == p_->named_fields_.end())
    throw std::out_of_range("System field provider slot is unknown: " + provider_slot);
  return output_local_pieces(found->second->accepted_potential(), 0, false);
}

template <int Dim>
std::vector<OutputPiece<Dim>> System<Dim>::output_embedded_boundary_local_pieces(
    const std::string& name, int level) const {
  if (level != 0)
    throw std::out_of_range("Uniform System output has only level zero");
  return output_local_pieces(embedded_boundary_output_field<Dim>(*p_, name), 0, false);
}

template <int Dim>
std::vector<OutputPiece<Dim>> System<Dim>::output_state_root_pieces(const ObserverMpiLane& lane,
                                                                    const std::string& name,
                                                                    int level) const {
  return output_pieces_to_root(lane,
                               detail::output_collective_identity("System", "state", name, level),
                               [&] { return output_state_local_pieces(name, level); });
}

template <int Dim>
std::vector<OutputPiece<Dim>> System<Dim>::output_field_root_pieces(
    const ObserverMpiLane& lane, const std::string& provider_slot, int level) {
  return output_pieces_to_root(
      lane, detail::output_collective_identity("System", "field", provider_slot, level),
      [&] { return output_field_local_pieces(provider_slot, level); });
}

template <int Dim>
std::vector<OutputPiece<Dim>> System<Dim>::output_embedded_boundary_root_pieces(
    const ObserverMpiLane& lane, const std::string& name, int level) const {
  return output_pieces_to_root(
      lane, detail::output_collective_identity("System", "embedded-boundary", name, level),
      [&] { return output_embedded_boundary_local_pieces(name, level); });
}

template <int Dim>
std::vector<Box<Dim>> System<Dim>::local_boxes(const std::string& name) const {
  const MultiFab<Dim>& state = p_->find(name).U;
  std::vector<Box<Dim>> result;
  result.reserve(state.local_size());
  for (std::size_t local = 0; local < state.local_size(); ++local)
    result.push_back(state.box(local));
  return result;
}

template <int Dim>
std::vector<double> System<Dim>::local_state(const std::string& name, int local_index) const {
  const typename Impl::Species& block = p_->find(name);
  if (local_index < 0 || static_cast<std::size_t>(local_index) >= block.U.local_size())
    throw std::out_of_range("System local state index is outside the owned patch list");
  const Fab<Dim>& fab = block.U.fab(static_cast<std::size_t>(local_index));
  const std::size_t cells = checked_cell_count(fab.box());
  std::vector<double> result(static_cast<std::size_t>(block.ncomp) * cells, 0.0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  for_each_host_index(fab.box(), [&](const Index<Dim>& index, std::size_t linear) {
    for (int component = 0; component < block.ncomp; ++component)
      result[static_cast<std::size_t>(component) * cells + linear] =
          static_cast<double>(host(storage_ordinal(fab, index, component)));
  });
  return result;
}

template void System<kNativeDimension>::validate_program_state_publication_candidate(
    int, const MultiFab<kNativeDimension>&) const;
template void System<kNativeDimension>::set_density(const std::string&, const std::vector<double>&);
template void System<kNativeDimension>::set_primitive_state(const std::string&,
                                                            const std::vector<double>&);
template std::vector<double> System<kNativeDimension>::get_primitive_state(const std::string&);
template void System<kNativeDimension>::set_poisson(const std::string&, const std::string&,
                                                    const std::string&, double, double, int);
template SolveReport System<kNativeDimension>::solve_fields_in_place_();
template SolveReport System<kNativeDimension>::solve_fields_from_state_in_place_(
    int, const MultiFab<kNativeDimension>&);
template SolveReport System<kNativeDimension>::solve_fields_from_state_at_in_place_(
    const runtime::multiblock::BoundaryEvaluationPoint&, const std::string&, int,
    const MultiFab<kNativeDimension>&);
template SolveReport System<kNativeDimension>::solve_fields_from_blocks_in_place_(
    const std::vector<const MultiFab<kNativeDimension>*>&);
template SolveReport System<kNativeDimension>::solve_fields_from_state_in_place_(
    const std::string&, int, const MultiFab<kNativeDimension>&);
template SolveReport System<kNativeDimension>::solve_fields_from_blocks_in_place_(
    const std::string&, const std::vector<const MultiFab<kNativeDimension>*>&);
template SolveReport System<kNativeDimension>::solve_fields_from_blocks_at_in_place_(
    const runtime::multiblock::BoundaryEvaluationPoint&, const std::string&,
    const std::vector<const MultiFab<kNativeDimension>*>&);
template SolveOutcome System<kNativeDimension>::solve_fields();
template SolveOutcome System<kNativeDimension>::solve_fields_from_state(
    int, const MultiFab<kNativeDimension>&);
template SolveOutcome System<kNativeDimension>::solve_fields_from_state_at(
    const runtime::multiblock::BoundaryEvaluationPoint&, const std::string&, int,
    const MultiFab<kNativeDimension>&);
template SolveOutcome System<kNativeDimension>::solve_fields_from_blocks(
    const std::vector<const MultiFab<kNativeDimension>*>&);
template SolveOutcome System<kNativeDimension>::solve_fields_from_state(
    const std::string&, int, const MultiFab<kNativeDimension>&);
template SolveOutcome System<kNativeDimension>::solve_fields_from_blocks(
    const std::string&, const std::vector<const MultiFab<kNativeDimension>*>&);
template void System<kNativeDimension>::prepare_default_field_publication_storage_();
template void System<kNativeDimension>::prepare_named_field_publication_storage_(
    const std::string&);
template void System<kNativeDimension>::begin_field_publication_transaction();
template void System<kNativeDimension>::stage_field_publication_candidate();
template void System<kNativeDimension>::validate_field_publication_candidate();
template void System<kNativeDimension>::accept_field_publication_candidate() noexcept;
template void System<kNativeDimension>::rollback_field_publication_transaction();
template bool System<kNativeDimension>::field_publication_transaction_active_() const noexcept;
template void System<kNativeDimension>::begin_field_publication_outcome_();
template SolveOutcome System<kNativeDimension>::stage_field_publication_outcome_(SolveReport);
template SolveOutcome System<kNativeDimension>::run_field_publication_outcome_(
    const std::function<SolveReport()>&);
template void System<kNativeDimension>::set_potential(const std::vector<double>&);
template std::vector<std::string> System<kNativeDimension>::field_provider_slots() const;
template void System<kNativeDimension>::set_field_potential(const std::string&,
                                                            const std::vector<double>&);
template std::vector<double> System<kNativeDimension>::eval_rhs(const std::string&);
template double System<kNativeDimension>::reduce_component(const std::string&, const std::string&,
                                                           int) const;
template MultiFab<kNativeDimension> System<kNativeDimension>::alloc_scalar_field(int, int);
template MultiFab<kNativeDimension>& System<kNativeDimension>::register_history(
    const std::string&, int, int, int, const std::string&, const std::string&, const std::string&,
    const std::string&);
template MultiFab<kNativeDimension>& System<kNativeDimension>::read_history(const std::string&,
                                                                            int);
template std::vector<double> System<kNativeDimension>::get_state(const std::string&);
template void System<kNativeDimension>::set_state(const std::string&, const std::vector<double>&);
template std::int64_t System<kNativeDimension>::set_analytic_expression_state(
    const std::string&, const std::string&, const std::string&, const std::string&,
    const std::vector<std::vector<std::string>>&, const std::vector<std::vector<double>>&);
template std::int64_t System<kNativeDimension>::set_analytic_mapped_state(
    const std::string&, const std::vector<std::vector<std::string>>&,
    const std::vector<std::vector<double>>&,
    const std::vector<runtime::system::AnalyticMappedInput>&, const std::string&);
template std::int64_t System<kNativeDimension>::set_analytic_gaussian_state(
    const std::string&, const RealVector<kNativeDimension>&, double, double, double);
template int System<kNativeDimension>::n_vars(const std::string&) const;
template std::vector<std::string> System<kNativeDimension>::variable_names(
    const std::string&, const std::string&) const;
template std::vector<std::string> System<kNativeDimension>::variable_roles(
    const std::string&, const std::string&) const;
template double System<kNativeDimension>::block_gamma(const std::string&) const;
template double System<kNativeDimension>::mass(const std::string&) const;
template std::vector<double> System<kNativeDimension>::density(const std::string&) const;
template std::vector<double> System<kNativeDimension>::potential();
template std::vector<double> System<kNativeDimension>::density_global(const std::string&) const;
template std::vector<double> System<kNativeDimension>::state_global(const std::string&) const;
template std::vector<double> System<kNativeDimension>::potential_global();
template std::vector<double> System<kNativeDimension>::field_potential_global(const std::string&);
template std::vector<OutputPiece<kNativeDimension>>
System<kNativeDimension>::output_state_local_pieces(const std::string&, int) const;
template std::vector<OutputPiece<kNativeDimension>>
System<kNativeDimension>::output_field_local_pieces(const std::string&, int);
template std::vector<OutputPiece<kNativeDimension>>
System<kNativeDimension>::output_embedded_boundary_local_pieces(const std::string&, int) const;
template std::vector<OutputPiece<kNativeDimension>>
System<kNativeDimension>::output_state_root_pieces(const ObserverMpiLane&, const std::string&,
                                                   int) const;
template std::vector<OutputPiece<kNativeDimension>>
System<kNativeDimension>::output_field_root_pieces(const ObserverMpiLane&, const std::string&, int);
template std::vector<OutputPiece<kNativeDimension>>
System<kNativeDimension>::output_embedded_boundary_root_pieces(const ObserverMpiLane&,
                                                               const std::string&, int) const;
template std::vector<Box<kNativeDimension>> System<kNativeDimension>::local_boxes(
    const std::string&) const;
template std::vector<double> System<kNativeDimension>::local_state(const std::string&, int) const;

}  // namespace pops
