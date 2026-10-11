// Native bind-time field/map stages over one fully seeded hierarchy.
template <int Dim>
std::pair<std::size_t, SolveReport> AmrSystem<Dim>::solve_bootstrap_initial_field(const std::string& subject_id,
                                                          const std::string& provider_slot,
                                                          int ordinal) {
  if (!p_->bootstrap_transaction || !p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error(
        "native initial field solve requires its prepared bootstrap transaction");
  const auto& lane = *p_->prepared_hierarchy->lane;
  const bool topology_stale =
      p_->last_topology_rematerialization_epoch != p_->engine->topology_epoch() ||
      p_->last_topology_rematerialization_generation != p_->engine->materialization_generation();
  (void)analytic::collectively_prepare_exact_analytic_request(
      "AmrSystem::solve_bootstrap_initial_field",
      [&] {
        const auto source = p_->bootstrap_sources.find(subject_id);
        const auto plan = p_->field_plans.find(provider_slot);
        if (source == p_->bootstrap_sources.end() ||
            source->second.kind != Impl::BootstrapSourceKind::program_seed ||
            plan == p_->field_plans.end() ||
            plan->second.output_block != source->second.runtime_block || ordinal < 1 ||
            p_->macro_step != 0 || p_->accepted_time != 0.0 ||
            p_->program.checkpoint_metadata_.primary_clock_identity.empty())
          throw std::invalid_argument(
              "native initial field solve lost its exact slot/source/zero-time authority");
        if (ordinal > 1 && topology_stale)
          throw std::logic_error(
              "native initial program topology changed between its field stages");
        return true;
      },
      [&] {
        ExactContractBuilder contract;
        contract.text(subject_id)
            .text(provider_slot)
            .scalar(ordinal)
            .scalar(p_->engine->topology_epoch())
            .scalar(p_->engine->materialization_generation());
        return std::move(contract).release();
      },
      lane.communicator());
  // The topology rematerializer already performs the first real composite solve. Do not
  // manufacture an extra first solve. The second stage must execute even on unchanged topology.
  const std::size_t cells = recompute_bootstrap_field(subject_id, provider_slot);
  if (ordinal == 1 && topology_stale) {
    const auto report = field_solve_report(provider_slot);
    if (!report)
      throw std::logic_error("native bootstrap field rematerialization lost its accepted report");
    return {cells, *report};
  }
  runtime::multiblock::BoundaryEvaluationPoint point;
  point.clock = p_->program.checkpoint_metadata_.primary_clock_identity;
  point.tick = 0;
  point.level = 0;
  point.substep = 0;
  point.stage = 0;
  point.stage_fraction = {0, 1};
  point.dt = 0.0;
  point.physical_time = 0.0;
  std::optional<runtime::PreparedAcceptedInitialFieldPointV1> authority;
  std::vector<const MultiFab<Dim>*> states;
  std::exception_ptr error;
  try {
    states.resize(p_->blocks.size(), nullptr);
    if (p_->active_initial_field_point)
      throw std::logic_error("native initial field authority cannot be nested");
    authority.emplace(point, p_->program.checkpoint_metadata_.primary_clock_identity,
                      p_->macro_step, p_->accepted_time, 0, true);
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane.communicator(),
                                 "native initial field stage allocation failed collectively");
  struct Reset {
    Impl& owner;
    ~Reset() { owner.active_initial_field_point = nullptr; }
  } reset{*p_};
  p_->active_initial_field_point = &*authority;
  const SolveReport report = consume_solve_outcome(
      solve_program_field_from_blocks_on_prepared_lane(point, provider_slot, 0, states));
  return {cells, report};
}

template <int Dim>
std::size_t AmrSystem<Dim>::map_bootstrap_state_hierarchy(
    const std::string& subject_id, const analytic::AnalyticOpcodeRows& opcodes,
    const analytic::AnalyticLiteralRows& literals,
    const std::vector<runtime::system::AnalyticMappedInput>& inputs,
    const std::string& consumer_qid, const std::string& provider_slot) {
  if (!p_->bootstrap_transaction || !p_->prepared_hierarchy || !p_->prepared_hierarchy->lane)
    throw std::logic_error("native initial map requires the prepared bootstrap transaction");
  const auto& lane = *p_->prepared_hierarchy->lane;
  using field_type = typename Impl::field_type;
  using input_type =
      analytic::detail::AnalyticMappedInputField<Dim, typename field_type::memory_space>;
  std::size_t block_index = 0;
  std::vector<analytic::AnalyticProgram> programs;
  (void)analytic::collectively_prepare_exact_analytic_request(
      "AmrSystem::map_bootstrap_state_hierarchy",
      [&] {
        if (consumer_qid.empty() || p_->macro_step != 0 || p_->accepted_time != 0.0)
          throw std::invalid_argument("native initial map changed its consumer or zero-time point");
        const auto source = p_->bootstrap_sources.find(subject_id);
        const auto field = p_->field_plans.find(provider_slot);
        if (source == p_->bootstrap_sources.end() ||
            source->second.kind != Impl::BootstrapSourceKind::program_seed ||
            field == p_->field_plans.end() || !field->second.materialized_for(*p_->engine) ||
            field->second.candidate_ready || field->second.output_keys.empty())
          throw std::invalid_argument(
              "native initial map has no complete accepted solved field family");
        const auto block = std::find_if(
            p_->blocks.begin(), p_->blocks.end(),
            [&](const auto& value) { return value.name == source->second.runtime_block; });
        if (block == p_->blocks.end() || field->second.output_block != block->name)
          throw std::invalid_argument("native initial map changed its field/State owner");
        block_index = static_cast<std::size_t>(std::distance(p_->blocks.begin(), block));
        programs = analytic::compile_component_programs(opcodes, literals);
        if (programs.size() != static_cast<std::size_t>(block->ncomp) ||
            inputs.size() > analytic::kAnalyticMaxStack)
          throw std::invalid_argument("native initial map changed its complete conservative shape");
        for (const auto& input : inputs) {
          input.validate();
          if (input.kind == runtime::system::AnalyticMappedInputKind::state_component) {
            if (input.state_component >= block->ncomp)
              throw std::out_of_range("initial map State component is outside its owner");
          } else {
            (void)p_->auxiliary_registry.provider_for_key(input.provider_key);
            if (input.kind == runtime::system::AnalyticMappedInputKind::provider_derivative &&
                (input.derivative_axis >= Dim ||
                 input.provider_key != field->second.output_keys.front()))
              throw std::invalid_argument(
                  "initial CF derivative requires the exact solved potential component");
          }
        }
        return true;
      },
      [&] {
        ExactContractBuilder contract;
        contract.text(subject_id)
            .text(consumer_qid)
            .text(provider_slot)
            .scalar(p_->engine->topology_epoch())
            .scalar(p_->engine->materialization_generation())
            .scalar(static_cast<std::uint64_t>(inputs.size()));
        for (const auto& input : inputs)
          input.serialize_exact(contract);
        for (std::size_t row = 0; row < opcodes.size(); ++row) {
          contract.scalar(static_cast<std::uint64_t>(opcodes[row].size()));
          for (std::size_t token = 0; token < opcodes[row].size(); ++token)
            contract.text(opcodes[row][token]).scalar(literals.at(row).at(token));
        }
        return std::move(contract).release();
      },
      lane.communicator());
  const auto& plan = p_->field_plans.at(provider_slot);
  const std::size_t levels = p_->engine->hierarchy().num_levels();
  std::vector<std::unique_ptr<field_type>> candidates;
  std::vector<std::vector<input_type>> resolved;
  std::vector<std::unique_ptr<field_type>> derivatives;
  std::exception_ptr error;
  try {
    if (plan.accepted_potential.size() != levels)
      throw std::logic_error("native initial potential family is incomplete");
    candidates.reserve(levels);
    resolved.resize(levels);
    derivatives.reserve(checked_size_product(
        levels, inputs.size(), "native initial derivative family exceeds size_t"));
    for (std::size_t level = 0; level < levels; ++level) {
      const auto& state = p_->block_state(block_index, level);
      if (!p_->bootstrap_materialized_actions.contains({subject_id, static_cast<int>(level)}))
        throw std::logic_error("native initial map encountered an unseeded hierarchy level");
      candidates.push_back(std::make_unique<field_type>(
          state.layout(), state.distribution(), state.local_rank(), state.ncomp(), state.ghosts()));
      copy_full_field_in_place(state, *candidates.back());
      resolved[level].reserve(inputs.size());
      for (const auto& input : inputs) {
        if (input.kind == runtime::system::AnalyticMappedInputKind::state_component)
          resolved[level].push_back({&state, input.state_component});
        else if (input.provider_key == plan.output_keys.front()) {
          if (!plan.accepted_potential[level])
            throw std::logic_error("initial potential level is empty");
          resolved[level].push_back({plan.accepted_potential[level].get(), 0});
        } else {
          const auto address = p_->auxiliary_registry.address_of(input.provider_key);
          const auto* group = p_->prepared_hierarchy->provider_storage[level]->find(address.group);
          if (!group || address.component >= static_cast<std::size_t>(group->ncomp()))
            throw std::logic_error("initial map lost its exact provider storage component");
          resolved[level].push_back({group, static_cast<int>(address.component)});
        }
      }
    }
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane.communicator(),
                                 "native initial family allocation failed collectively");
  std::size_t count = 0;
  // All reads refer to phi1 and unmodified conservative State. No level is published early.
  for (std::size_t level = 0; level < levels; ++level) {
    const auto geometry = Geometry<Dim>::from_bounds(p_->engine->hierarchy().layout(level).domain(),
                                                     p_->cfg.lower, p_->cfg.upper);
    for (std::size_t slot = 0; slot < inputs.size(); ++slot) {
      if (inputs[slot].kind != runtime::system::AnalyticMappedInputKind::provider_derivative)
        continue;
      auto derivative = analytic::materialize_initial_derivative_from_prepared_halo(
          *resolved[level][slot].field, geometry, inputs[slot].derivative_axis, lane);
      resolved[level][slot] = {derivative.get(), 0};
      derivatives.push_back(std::move(derivative));
    }
    count = checked_size_sum(
        count,
        static_cast<std::size_t>(analytic::materialize_discrete_mapped_state(
            *candidates[level], geometry, programs, resolved[level], lane.communicator())),
        "native mapped family materialization exceeds size_t");
    validate_prepared_amr_state_publication_candidate(static_cast<int>(block_index),
                                                      static_cast<int>(level), *candidates[level]);
  }
  error = {};
  try {
    for (std::size_t level = 0; level < levels; ++level)
      copy_full_field_in_place(*candidates[level], p_->block_state(block_index, level));
    Kokkos::fence();
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane.communicator(),
                                 "native initial State family publication failed collectively");
  p_->discard_level_evaluations();
  return count;
}
