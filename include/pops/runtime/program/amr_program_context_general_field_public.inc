/// Install an exact field-owned hierarchy problem in the existing prepared solver registry.
/// The key is a compiled solve identity, never the index of a contributing species.
void configure_hierarchy_field_solver(
    std::int64_t key, int components, const std::string& field_identity,
    const std::string& provider_identity, const std::string& plan_identity,
    const std::string& operator_contract_identity, const std::vector<std::string>& assembly_slots,
    const std::string& solution_slot, const PreparedProviderOptions& options) const {
  require_facade_execution_();
  const ExecutionLane& lane = prepared_execution_lane();
  std::map<std::int64_t, HierarchyFieldResource> staged;
  std::exception_ptr error;
  long invalid = 0;
  try {
    if (key < 0 || components < 1 || field_identity.empty() || provider_identity.empty() ||
        plan_identity.empty() || operator_contract_identity.empty() || assembly_slots.empty() ||
        solution_slot.empty() || !hierarchy_tensor_solver_registry_)
      throw std::invalid_argument(
          "hierarchy field requires complete independent storage authority");
    ExactContractBuilder contract;
    contract.text("pops.hierarchy.field-storage")
        .scalar(std::uint32_t{1})
        .scalar(key)
        .scalar(components)
        .text(field_identity)
        .text(provider_identity)
        .text(plan_identity)
        .text(operator_contract_identity)
        .text(solution_slot)
        .bytes(options.exact_contract())
        .sequence(assembly_slots,
                  [](ExactContractBuilder& item, const std::string& slot) { item.text(slot); });
    HierarchyTensorSelection selection{-1,
                                       components,
                                       provider_identity,
                                       plan_identity,
                                       operator_contract_identity,
                                       assembly_slots,
                                       solution_slot,
                                       options,
                                       std::move(contract).release()};
    const auto found = hierarchy_field_resources_.find(key);
    if (found != hierarchy_field_resources_.end() &&
        found->second.selection.exact_contract != selection.exact_contract)
      throw std::invalid_argument(
          "hierarchy field solve key was rebound to a different physical problem");
    staged.emplace(key, HierarchyFieldResource{std::move(selection), field_identity, nullptr});
  } catch (...) {
    invalid = 1;
    error = std::current_exception();
  }
  if (all_reduce_max(invalid, lane) != 0) {
    if (error)
      std::rethrow_exception(error);
    throw std::invalid_argument("hierarchy field storage configuration failed on another rank");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"hierarchy-field-storage", staged.begin()->second.selection.exact_contract}}, lane))
    throw std::invalid_argument("hierarchy field storage authority differs between ranks");
  const long present = hierarchy_field_resources_.contains(key) ? 1L : 0L;
  if (all_reduce_min(present, lane) != all_reduce_max(present, lane))
    throw std::invalid_argument("hierarchy field resource presence differs between ranks");
  if (present == 0) {
    auto& resource = staged.begin()->second;
    resource.solver = prepare_hierarchy_field_solver_(resource);
    resource.topology_epoch = runtime_->topology_epoch();
    resource.generation = runtime_->materialization_generation();
    hierarchy_field_resources_.merge(staged);  // Allocation-free after the collective preparation.
  } else {
    (void)configured_hierarchy_field_solver_(key);
  }
}

field_type& hierarchy_field_assembly(std::int64_t key, std::string_view slot) const {
  auto& solver = configured_hierarchy_field_solver_(key);
  const auto& selection = hierarchy_field_resources_.at(key).selection;
  const long invalid =
      std::find(selection.assembly_field_slots.begin(), selection.assembly_field_slots.end(),
                slot) == selection.assembly_field_slots.end()
          ? 1L
          : 0L;
  if (all_reduce_max(invalid, prepared_execution_lane()) != 0)
    throw std::invalid_argument(
        "hierarchy field assembly slot differs from its exact problem on a rank");
  return solver.assembly_target(slot, active_level_);
}

field_type& hierarchy_field_solution(std::int64_t key) const {
  return configured_hierarchy_field_solver_(key).solution(active_level_);
}

void stage_hierarchy_field_initial_guess(std::int64_t key) const {
  configured_hierarchy_field_solver_(key).stage_initial_guess(active_level_, nullptr);
}
void stage_hierarchy_field_initial_guess(std::int64_t key, const field_type& guess) const {
  configured_hierarchy_field_solver_(key).stage_initial_guess(active_level_, &guess);
}

SolveOutcome solve_hierarchy_field(std::int64_t key, Real relative, Real absolute,
                                   int maximum) const {
  return solve_prepared_hierarchy_tensor_collectively(
      configured_hierarchy_field_solver_(key),
      HierarchyTensorSolveControls{relative, absolute, maximum}, prepared_execution_lane());
}

using ProgramFieldComponent = typename facade_type::ProgramFieldComponent;
using ProgramFieldLevel = typename facade_type::ProgramFieldLevel;

void begin_staged_field_publications() const {
  staged_field_publications_.clear();
}
void stage_field_components(int evaluation_id, const std::string& identity,
                            std::initializer_list<ProgramFieldComponent> components) const {
  std::exception_ptr error;
  try {
    auto& levels = staged_field_publications_[identity];
    if (identity.empty() || levels.size() != static_cast<std::size_t>(active_level_))
      throw std::invalid_argument(
          "hierarchy field publication levels must be staged once in order");
    levels.push_back(
        {boundary_evaluation_point(evaluation_id), std::vector<ProgramFieldComponent>(components)});
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, prepared_execution_lane()) != 0) {
    staged_field_publications_.clear();
    if (error)
      std::rethrow_exception(error);
    throw std::runtime_error("hierarchy field publication staging failed collectively");
  }
}
void publish_staged_field_components() const {
  try {
    std::string bytes;
    std::exception_ptr error;
    try {
      ExactContractBuilder contract;
      contract.text("pops.hierarchy.field-publications");
      for (const auto& [identity, levels] : staged_field_publications_)
        contract.text(identity).scalar(static_cast<std::uint64_t>(levels.size()));
      bytes = std::move(contract).release();
    } catch (...) {
      error = std::current_exception();
    }
    if (all_reduce_max(error ? 1L : 0L, prepared_execution_lane()) != 0) {
      if (error)
        std::rethrow_exception(error);
      throw std::runtime_error("hierarchy field publication contract failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs({{"hierarchy-field-publications", bytes}},
                                                  prepared_execution_lane()))
      throw std::invalid_argument("hierarchy field publication identities differ between ranks");
    for (const auto& [identity, levels] : staged_field_publications_)
      facade_->publish_program_field_components(identity, levels);
    staged_field_publications_.clear();
  } catch (...) {
    staged_field_publications_.clear();
    throw;
  }
}

void observe_hierarchy_field_gradient(std::int64_t key, field_type& output, int component) const {
  const auto& input = configured_hierarchy_field_solver_(key).solution(active_level_);
  long invalid = component < 0 || component >= input.ncomp() || output.ncomp() != Dim ||
                 output.layout() != input.layout() ||
                 output.distribution() != input.distribution() ||
                 output.local_rank() != input.local_rank() || output.shares_storage_with(input);
  for (int axis = 0; axis < Dim; ++axis)
    invalid |= input.ghosts()[axis] < 1 ? 1L : 0L;
  if (all_reduce_max(invalid, prepared_execution_lane()) != 0)
    throw std::invalid_argument("hierarchy field gradient requires exact solved field storage");
  const auto geom = geometry();
  for (std::size_t local = 0; local < output.local_size(); ++local) {
    const auto result = output.fab(local).view();
    const auto value = input.fab(local).view();
    for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      for (int axis = 0; axis < Dim; ++axis) {
        Index<Dim> lower = cell, upper = cell;
        --lower[axis];
        ++upper[axis];
        result(cell, axis) =
            (value(upper, component) - value(lower, component)) / (Real(2) * geom.spacing(axis));
      }
    });
  }
  count_kernel_();
}

/// Raw active-domain algebra, matching owner-qualified AMR Program reductions.
/// Covered coarse cells are excluded exactly once; these are not volume integrals.
Real reduce_hierarchy_field_component(std::int64_t key, int component,
                                      std::string_view kind) const {
  auto& solver = configured_hierarchy_field_solver_(key);
  const auto& lane = prepared_execution_lane();
  const bool maximum = kind == "max", minimum = kind == "min";
  Real local = maximum   ? -std::numeric_limits<Real>::infinity()
               : minimum ? std::numeric_limits<Real>::infinity()
                         : Real(0);
  std::exception_ptr error;
  try {
    if (kind != "sum" && kind != "abs_sum" && !maximum && !minimum)
      throw std::invalid_argument("unknown hierarchy field observation reduction");
    for (int level = 0; level < solver.level_count(); ++level) {
      const auto& field = solver.solution(level);
      const auto& active = solver.active_cell_mask(level);
      if (component < 0 || component >= field.ncomp())
        throw std::invalid_argument(
            "hierarchy field observation component is outside solved tuple");
      if (kind == "sum")
        local += pops::reduce_active_sum_local(field, component, &active);
      else if (kind == "abs_sum")
        local += pops::reduce_active_abs_sum_local(field, component, &active);
      else if (pops::reduce_masked_max_local(field, component, &active).has_active) {
        if (maximum)
          local = std::max(local, pops::reduce_active_max_local(field, component, &active));
        else
          local = std::min(local, pops::reduce_active_min_local(field, component, &active));
      }
    }
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    if (error)
      std::rethrow_exception(error);
    throw std::runtime_error("hierarchy field observation reduction failed collectively");
  }
  if (maximum)
    return static_cast<Real>(all_reduce_max(static_cast<double>(local), lane));
  if (minimum)
    return static_cast<Real>(all_reduce_min(static_cast<double>(local), lane));
  return static_cast<Real>(all_reduce_sum(static_cast<double>(local), lane));
}
