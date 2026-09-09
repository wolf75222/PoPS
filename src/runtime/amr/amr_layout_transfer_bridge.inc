// Included in amr_system.cpp after the complete facade implementation; no second runtime owner.
namespace amr_layout_transfer_bridge_detail {
template <class Function>
void collective(const ExecutionLane& lane, const char* operation, Function&& function) {
  std::exception_ptr error;
  try {
    function();
  } catch (...) {
    error = std::current_exception();
  }
  if (all_reduce_max(error ? 1L : 0L, lane)) {
    if (lane.size() == 1 && error)
      std::rethrow_exception(error);
    throw std::runtime_error(std::string(operation) + " failed on a hierarchy rank");
  }
}
}  // namespace amr_layout_transfer_bridge_detail

template <int Dim>
struct PreparedAmrSystemLayoutTransfer<Dim>::Impl {
  AmrSystem<Dim>* source;
  AmrSystem<Dim>* target;
  std::shared_ptr<component::LoadedComponent> provider;
  AmrPhysicalTransferSpec<Dim> spec;
  SystemLayoutTransferExecution execution;
  std::shared_ptr<PreparedAmrLayoutTransfer<Dim>> transfer;
  std::string prepared_source_hierarchy, prepared_target_hierarchy;
  std::uint64_t prepared_source_epoch = 0, prepared_target_epoch = 0;
  std::uint64_t active_generation = 0;

  Impl(AmrSystem<Dim>& source_system, AmrSystem<Dim>& target_system,
       std::shared_ptr<component::LoadedComponent> component, AmrPhysicalTransferSpec<Dim> contract,
       SystemLayoutTransferExecution context)
      : source(&source_system),
        target(&target_system),
        provider(std::move(component)),
        spec(std::move(contract)),
        execution(std::move(context)) {}

  const ExecutionLane& lane() const {
    return source->p_->require_prepared_engine_lane("AMR layout-transfer bridge");
  }

  void require_transactions() const {
    if (!source->p_->external_step_transaction || !target->p_->external_step_transaction ||
        source->p_->external_step_committed || target->p_->external_step_committed)
      throw std::logic_error(
          "AMR layout transfer requires both native Systems in uncommitted transactions");
  }

  AmrTransferEndpoint<Dim> endpoint(AmrSystem<Dim>& system, bool input) const {
    const auto& impl = *system.p_;
    if (!impl.engine || !impl.prepared_hierarchy || !impl.prepared_hierarchy->lane)
      throw std::logic_error("AMR transfer endpoint requires a materialized hierarchy");
    const auto& name = input ? spec.authentication.source_block : spec.authentication.target_block;
    const auto block = impl.block_index(name);
    AmrTransferEndpoint<Dim> result;
    result.layout_identity = input ? spec.authentication.source_layout_identity
                                   : spec.authentication.target_layout_identity;
    result.state_identity = impl.boundary_registry.state_route(name);
    result.hierarchy_identity = prefixed_sha256("pops.amr-layout-transfer.hierarchy.v1:sha256:",
                                                impl.engine->spatial_contract());
    result.hierarchy_generation = impl.engine->materialization_generation();
    result.periodicity = impl.cfg.periodicity;
    const auto count = impl.engine->hierarchy().num_levels();
    result.levels.reserve(count);
    for (std::size_t level = 0; level < count; ++level) {
      const auto& embedded = impl.prepared_hierarchy->embedded_boundary[level];
      const MultiFab<Dim>* active = nullptr;
      const MultiFab<Dim>* measure = nullptr;
      if (embedded && embedded->mode() != runtime::system::PreparedEmbeddedBoundaryMode::inactive) {
        active = &embedded->active_mask();
        measure = embedded->mode() == runtime::system::PreparedEmbeddedBoundaryMode::staircase
                      ? active
                      : &embedded->volume_fraction();
        result.measure_identity = "pops://measure/cellwise-constant-relative-volume@1";
      }
      result.levels.push_back(
          {static_cast<int>(level),
           Geometry<Dim>::from_bounds(impl.engine->hierarchy().layout(level).domain(),
                                      impl.cfg.lower, impl.cfg.upper),
           &impl.block_state(block, level), impl.prepared_hierarchy->active_coverage[level].get(),
           active, measure});
    }
    return result;
  }

  void refresh_transfer() {
    AmrTransferEndpoint<Dim> from, to;
    amr_layout_transfer_bridge_detail::collective(lane(), "AMR transfer endpoint assembly", [&] {
      from = endpoint(*source, true);
      to = endpoint(*target, false);
    });
    const bool stale = !transfer || prepared_source_hierarchy != from.hierarchy_identity ||
                       prepared_target_hierarchy != to.hierarchy_identity ||
                       prepared_source_epoch != from.hierarchy_generation ||
                       prepared_target_epoch != to.hierarchy_generation;
    if (all_reduce_min(stale ? 1L : 0L, lane()) != all_reduce_max(stale ? 1L : 0L, lane()))
      throw std::logic_error(
          "AMR layout transfer hierarchy refresh decision differs between ranks");
    if (!stale)
      return;
    auto replacement =
        PreparedAmrLayoutTransfer<Dim>::prepare(from, to, spec, provider, execution, lane());
    amr_layout_transfer_bridge_detail::collective(lane(), "AMR transfer hierarchy provenance", [&] {
      prepared_source_hierarchy = from.hierarchy_identity;
      prepared_target_hierarchy = to.hierarchy_identity;
    });
    prepared_source_epoch = from.hierarchy_generation;
    prepared_target_epoch = to.hierarchy_generation;
    transfer = std::move(replacement);
  }
};

template <int Dim>
PreparedAmrSystemLayoutTransfer<Dim>::PreparedAmrSystemLayoutTransfer(
    std::unique_ptr<Impl> impl) noexcept
    : p_(std::move(impl)) {}
template <int Dim>
PreparedAmrSystemLayoutTransfer<Dim>::~PreparedAmrSystemLayoutTransfer() = default;

template <int Dim>
std::shared_ptr<PreparedAmrSystemLayoutTransfer<Dim>> PreparedAmrSystemLayoutTransfer<Dim>::prepare(
    AmrSystem<Dim>& source, AmrSystem<Dim>& target,
    std::shared_ptr<component::LoadedComponent> provider, AmrPhysicalTransferSpec<Dim> spec,
    SystemLayoutTransferExecution execution) {
  source.p_->ensure_engine();
  target.p_->ensure_engine();
  const ExecutionLane& lane =
      source.p_->require_prepared_engine_lane("AMR layout transfer preparation");
  std::shared_ptr<PreparedAmrSystemLayoutTransfer> result;
  amr_layout_transfer_bridge_detail::collective(
      lane, "AMR layout transfer facade preparation", [&] {
        if (&source == &target)
          throw std::invalid_argument("AMR layout transfer requires distinct live Systems");
        if (!spec.authentication.program_invocation.empty())
          throw std::invalid_argument(
              "AMR Program-stage transfer must supply qualified native stage views");
        const ExecutionLane& target_lane =
            target.p_->require_prepared_engine_lane("AMR layout transfer target");
        if (!lane.congruent_with(target_lane))
          throw std::invalid_argument(
              "AMR layout transfer changes the native hierarchy rank space");
        auto pending = std::make_unique<Impl>(source, target, std::move(provider), std::move(spec),
                                              std::move(execution));
        // Complete local owner/control-block allocation before refresh prepares an MPI session.
        result = std::shared_ptr<PreparedAmrSystemLayoutTransfer>(
            new PreparedAmrSystemLayoutTransfer(std::move(pending)));
      });
  result->p_->refresh_transfer();
  return result;
}

template <int Dim>
AmrLayoutTransferReceipt PreparedAmrSystemLayoutTransfer<Dim>::expected_receipt_contract() const {
  AmrTransferEndpoint<Dim> from, to;
  amr_layout_transfer_bridge_detail::collective(
      p_->lane(), "AMR expected receipt endpoint assembly", [&] {
        p_->require_transactions();
        if (!p_->active_generation)
          throw std::logic_error(
              "AMR expected receipt requires the current prepared transfer transaction");
        from = p_->endpoint(*p_->source, true);
        to = p_->endpoint(*p_->target, false);
      });
  return p_->transfer->expected_receipt_contract(from, to);
}

template <int Dim>
void PreparedAmrSystemLayoutTransfer<Dim>::begin_transaction(std::uint64_t generation) {
  amr_layout_transfer_bridge_detail::collective(p_->lane(), "AMR layout transfer begin", [&] {
    p_->require_transactions();
    if (p_->active_generation)
      throw std::logic_error("AMR layout transfer already has an active transaction");
  });
  p_->refresh_transfer();
  p_->transfer->begin_transaction(generation);
  p_->active_generation = generation;
}

template <int Dim>
void PreparedAmrSystemLayoutTransfer<Dim>::capture(std::uint64_t generation,
                                                   std::uint64_t attempt) {
  AmrTransferEndpoint<Dim> from;
  amr_layout_transfer_bridge_detail::collective(p_->lane(), "AMR layout transfer capture endpoint",
                                                [&] {
                                                  p_->require_transactions();
                                                  from = p_->endpoint(*p_->source, true);
                                                });
  p_->transfer->capture(from, generation, attempt);
}

template <int Dim>
AmrLayoutTransferReceipt PreparedAmrSystemLayoutTransfer<Dim>::apply(std::uint64_t generation,
                                                                     std::uint64_t attempt) {
  AmrTransferEndpoint<Dim> to;
  std::vector<MultiFab<Dim>> candidates;
  std::vector<MultiFab<Dim>*> candidate_views;
  const auto& lane = p_->lane();
  amr_layout_transfer_bridge_detail::collective(lane, "AMR layout transfer candidate staging", [&] {
    p_->require_transactions();
    to = p_->endpoint(*p_->target, false);
    candidates.reserve(to.levels.size());
    candidate_views.reserve(to.levels.size());
    for (const auto& level : to.levels)
      candidates.emplace_back(*level.state);
    for (auto& candidate : candidates)
      candidate_views.push_back(&candidate);
  });
  auto receipt = p_->transfer->apply(to, candidate_views, generation, attempt);
  auto& target = *p_->target;
  const auto block = target.p_->block_index(p_->spec.authentication.target_block);
  const ExecutionLane& target_lane =
      target.p_->require_prepared_engine_lane("AMR transfer covered-cell synchronization");
  for (std::size_t fine = candidates.size(); fine > 1; --fine) {
    const auto child = fine - 1, parent = fine - 2;
    if (to.measure_identity == "pops://measure/cartesian-cells@1") {
      numerics::time::amr::execute_average_down_collectively(
          target.p_->multiblock_hierarchy->topology_runtime(), child, candidates[child],
          candidates[parent], target_lane, &target.p_->program.profiler_);
      continue;
    }
    // Restrict the explicitly declared relative measure, then divide by the parent measure.
    // The numerical restriction and inter-rank movement stay with the existing AMR authority.
    std::optional<MultiFab<Dim>> weighted_child, weighted_parent;
    amr_layout_transfer_bridge_detail::collective(
        target_lane, "AMR physical measure restriction staging", [&] {
          weighted_child.emplace(candidates[child]);
          weighted_parent.emplace(candidates[parent]);
          for (std::size_t local = 0; local < weighted_child->local_size(); ++local) {
            auto values = weighted_child->fab(local).view();
            const auto global = weighted_child->global_index(local);
            const auto active = to.levels[child].activity->fab_global(global).view();
            const auto measure = to.levels[child].relative_measure->fab_global(global).view();
            const int width = weighted_child->ncomp();
            for_each_cell(
                weighted_child->fab(local).box(), KOKKOS_LAMBDA(const Index<Dim>& cell) {
                  const Real fraction = active(cell) == Real(0) ? Real(0) : measure(cell);
                  for (int component = 0; component < width; ++component)
                    values(cell, component) =
                        fraction == Real(0) ? Real(0) : values(cell, component) * fraction;
                });
          }
          Kokkos::fence();
        });
    numerics::time::amr::execute_average_down_collectively(
        target.p_->multiblock_hierarchy->topology_runtime(), child, *weighted_child,
        *weighted_parent, target_lane, &target.p_->program.profiler_);
    amr_layout_transfer_bridge_detail::collective(
        target_lane, "AMR physical measure restriction normalization", [&] {
          for (std::size_t local = 0; local < candidates[parent].local_size(); ++local) {
            const auto global = candidates[parent].global_index(local);
            auto values = candidates[parent].fab(local).view();
            const auto restricted = weighted_parent->fab_global(global).view();
            const auto active = to.levels[parent].activity->fab_global(global).view();
            const auto measure = to.levels[parent].relative_measure->fab_global(global).view();
            const auto coverage = to.levels[parent].coverage->fab_global(global).view();
            const int width = candidates[parent].ncomp();
            for_each_cell(
                candidates[parent].fab(local).box(), KOKKOS_LAMBDA(const Index<Dim>& cell) {
                  if (coverage(cell) != Real(0) || active(cell) == Real(0))
                    return;
                  for (int component = 0; component < width; ++component)
                    values(cell, component) = restricted(cell, component) / measure(cell);
                });
          }
          Kokkos::fence();
        });
  }
  // Every level passes model recovery before the first accepted-state publication.
  for (std::size_t level = 0; level < candidates.size(); ++level)
    target.validate_prepared_amr_state_publication_candidate(
        static_cast<int>(block), static_cast<int>(level), candidates[level]);

  std::vector<std::string> names;
  amr_layout_transfer_bridge_detail::collective(target_lane,
                                                "AMR layout transfer publication identities", [&] {
                                                  for (const auto& entry : target.p_->blocks)
                                                    names.push_back(entry.name);
                                                });
  const auto block_map = target.p_->multiblock_hierarchy->prepare_program_block_map(names);
  // The explicit child transaction makes multi-level publication one rollback unit. It remains
  // provisional in the enclosing multi-layout transaction after this child has committed.
  target.begin_nested_step_transaction();
  try {
    for (std::size_t level = 0; level < candidates.size(); ++level) {
      std::vector<MultiFab<Dim>> other_blocks;
      std::vector<MultiFab<Dim>*> pack;
      amr_layout_transfer_bridge_detail::collective(
          target_lane, "AMR layout transfer all-block candidate pack", [&] {
            other_blocks.reserve(names.size());
            pack.resize(names.size());
            for (std::size_t owner = 0; owner < names.size(); ++owner) {
              if (owner == block) {
                pack[owner] = &candidates[level];
                continue;
              }
              other_blocks.emplace_back(target.p_->block_state(owner, level));
              pack[owner] = &other_blocks.back();
            }
          });
      target.p_->multiblock_hierarchy->publish_program_candidates(block_map, level, pack);
    }
    target.p_->discard_level_evaluations();
    target.commit_step_transaction();
    target.finalize_step_transaction();
  } catch (...) {
    target.rollback_step_transaction();
    throw;
  }
  return receipt;
}

template <int Dim>
void PreparedAmrSystemLayoutTransfer<Dim>::reject_attempt(std::uint64_t generation,
                                                          std::uint64_t attempt) {
  p_->transfer->reject_attempt(generation, attempt);
}
template <int Dim>
void PreparedAmrSystemLayoutTransfer<Dim>::finalize_transaction(std::uint64_t generation) noexcept {
  if (p_->active_generation != generation)
    return;
  p_->transfer->finalize_transaction(generation);
  p_->active_generation = 0;
}
template <int Dim>
void PreparedAmrSystemLayoutTransfer<Dim>::rollback_transaction(std::uint64_t generation) noexcept {
  if (p_->active_generation != generation)
    return;
  p_->transfer->rollback_transaction(generation);
  p_->active_generation = 0;
}

template class PreparedAmrSystemLayoutTransfer<kNativeDimension>;
