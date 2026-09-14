/// Numerical observations belong to their exact prepared solve, independently of state owners.
field_type& hierarchy_field_scratch(std::int64_t solve_key, std::int64_t value_id, int subslot,
                                    int ncomp = 1, int ghost_depth = 1) const {
  const auto& lane = prepared_execution_lane();
  auto& solver = configured_hierarchy_field_solver_(solve_key);
  auto& resource = hierarchy_field_resources_.at(solve_key);
  using Key = std::tuple<int, std::int64_t, int>;
  const Key key{active_level_, value_id, subslot};
  std::exception_ptr error;
  std::string contract;
  const field_type* prototype = nullptr;
  Extent<Dim> ghosts{};
  try {
    if (value_id < 0 || subslot < 0 || ncomp < 1 || ghost_depth < 0 || active_level_ < 0 ||
        static_cast<std::size_t>(active_level_) >= runtime_->hierarchy().num_levels())
      throw std::invalid_argument("hierarchy field scratch requires an exact value/level/shape");
    prototype = &solver.solution(active_level_);
    const auto& level = runtime_->hierarchy().state(static_cast<std::size_t>(active_level_));
    if (prototype->ncomp() != resource.selection.components ||
        prototype->layout() != level.layout() ||
        prototype->distribution() != level.distribution() ||
        prototype->local_rank() != level.local_rank() ||
        prototype->local_size() != level.local_size())
      throw std::invalid_argument(
          "hierarchy field scratch provider changed its exact level layout");
    ghosts = uniform_ghosts_(ghost_depth);
    ExactContractBuilder authority;
    authority.text("pops.hierarchy.field-scratch")
        .scalar(std::uint32_t{1})
        .scalar(solve_key)
        .scalar(active_level_)
        .scalar(value_id)
        .scalar(subslot)
        .scalar(ncomp)
        .scalar(ghost_depth)
        .scalar(resource.topology_epoch)
        .scalar(resource.generation)
        .bytes(resource.selection.exact_contract);
    contract = std::move(authority).release();
    const auto found = resource.scratches.find(key);
    if (found != resource.scratches.end()) {
      const auto& result = found->second;
      if (result.layout() != prototype->layout() ||
          result.distribution() != prototype->distribution() ||
          result.local_rank() != prototype->local_rank() ||
          result.local_size() != prototype->local_size() || result.ncomp() != ncomp ||
          result.ghosts() != ghosts || result.shares_storage_with(*prototype))
        throw std::invalid_argument(
            "hierarchy field scratch identity changed its exact field contract");
    }
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane,
                                 "hierarchy field scratch validation failed collectively");
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"hierarchy-field-scratch", contract}}, lane))
    throw std::invalid_argument("hierarchy field scratch identity differs between ranks");

  // Allocate in a detached map. A rank-local allocation failure cannot publish only some ranks'
  // resources or skip the caller's next collective. Existing map references remain stable.
  std::map<Key, field_type> staged;
  try {
    const auto found = resource.scratches.find(key);
    if (found == resource.scratches.end())
      staged.emplace(key, make_scratch_(*prototype, ncomp, ghosts));
    else
      found->second.set_val(Real(0));
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane,
                                 "hierarchy field scratch allocation failed collectively");
  resource.scratches.merge(staged);
  return resource.scratches.find(key)->second;
}
