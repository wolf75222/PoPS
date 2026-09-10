std::unique_ptr<hierarchy_tensor_solver_type> prepare_hierarchy_field_solver_(
    const HierarchyFieldResource& resource) const {
  const auto& selection = resource.selection;
  hierarchy_tensor_request_type request;
  std::exception_ptr error;
  long invalid = 0;
  try {
    request.block =
        std::numeric_limits<std::size_t>::max();  // No contributing state owns this field.
    request.components = selection.components;
    request.plan_identity = selection.plan_identity + ":" + resource.field_identity;
    request.operator_contract_identity = selection.operator_contract_identity;
    request.assembly_field_slots = selection.assembly_field_slots;
    request.solution_field_slot = selection.solution_field_slot;
    request.options = selection.options;
    const auto topology = facade_->prepared_amr_boundary_topology();
    for (std::size_t level = 0; level < runtime_->hierarchy().num_levels(); ++level) {
      const auto& prototype = runtime_->hierarchy().state(level);
      const auto geometry = facade_->prepared_amr_level_geometry(static_cast<int>(level));
      std::array<PhysicalBoundaryFace, 2 * Dim> faces{};
      RealVector<Dim> spacing{};
      for (int axis = 0; axis < Dim; ++axis) {
        spacing[axis] = geometry.spacing(axis);
        for (int side = 0; side < 2; ++side) {
          const Face<Dim> face{axis, side == 0 ? BoundarySide::lower : BoundarySide::upper};
          const auto& law = std::get<std::string>(selection.options.values.at(
              "physical." + std::to_string(axis) + "." + std::to_string(side)));
          if ((law == "periodic") != topology.is_periodic(face) ||
              (law != "periodic" && law != "neumann"))
            throw std::invalid_argument(
                "authored hierarchy field boundary differs from prepared topology");
          if (law == "neumann")
            faces[static_cast<std::size_t>(face.ordinal())] = {PhysicalBoundaryKind::neumann,
                                                               Real(0), Real(0), Real(1)};
        }
      }
      PhysicalBoundaryConditions<Dim> boundary{topology, faces, spacing};
      request.levels.push_back({geometry, boundary, prototype.layout(), prototype.distribution(),
                                prototype.local_rank()});
      if (level != 0)
        request.ratios.push_back(runtime_->hierarchy().layout(level).ratio_from_parent());
    }
  } catch (...) {
    invalid = 1;
    error = std::current_exception();
  }
  const auto& lane = prepared_execution_lane();
  if (all_reduce_max(invalid, lane) != 0) {
    if (error)
      std::rethrow_exception(error);
    throw std::invalid_argument("hierarchy field request failed on another rank");
  }
  return prepare_hierarchy_tensor_solver_collectively(
      *hierarchy_tensor_solver_registry_, selection.provider_identity, std::move(request), lane);
}

hierarchy_tensor_solver_type& configured_hierarchy_field_solver_(std::int64_t key) const {
  refresh_resources_();
  auto found = hierarchy_field_resources_.find(key);
  const auto& lane = prepared_execution_lane();
  if (all_reduce_max(found == hierarchy_field_resources_.end() ? 1L : 0L, lane) != 0)
    throw std::logic_error(
        "hierarchy field must be installed on every rank before assembly or observation");
  auto& resource = found->second;
  const long stale = !resource.solver || resource.topology_epoch != runtime_->topology_epoch() ||
                             resource.generation != runtime_->materialization_generation()
                         ? 1L
                         : 0L;
  if (all_reduce_max(stale, lane) != 0) {
    if (all_reduce_max(mapping_continuation_ && !resource.scratches.empty() ? 1L : 0L, lane) != 0)
      throw std::logic_error("hierarchy field scratch generation changed during a live continuation");
    auto prepared = prepare_hierarchy_field_solver_(resource);
    staged_field_publications_.clear();
    resource.scratches.clear();
    resource.solver = std::move(prepared);
    resource.topology_epoch = runtime_->topology_epoch();
    resource.generation = runtime_->materialization_generation();
  }
  return *resource.solver;
}
