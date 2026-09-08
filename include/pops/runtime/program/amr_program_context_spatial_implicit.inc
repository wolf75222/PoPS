// Composite temporal solves own provisional closure; residual/JVP evaluations create no FluxBasis.
template <class Prepare>
void prepare_spatial_collectively(Prepare&& prepare) const {
  std::exception_ptr error;
  try {
    prepare();
  } catch (...) {
    error = std::current_exception();
  }
  const auto& lane = prepared_execution_lane();
  if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && error)
      std::rethrow_exception(error);
    throw std::runtime_error("composite spatial preparation failed collectively");
  }
}

std::shared_ptr<PreparedAmrSpatialResidual<Dim>> prepare_spatial_hierarchy(
    int identity, int program_block, FieldNewtonOptions options, Real difference_step) const {
  const auto epoch = runtime_->topology_epoch(),
             generation = runtime_->materialization_generation();
  auto found = spatial_hierarchy_resources_.find(identity);
  const bool hit = found != spatial_hierarchy_resources_.end() && found->second.epoch == epoch &&
                   found->second.generation == generation;
  std::string cache_contract;
  prepare_spatial_collectively([&] {
    ExactContractBuilder exact;
    exact.scalar(epoch)
        .scalar(generation)
        .scalar(std::int32_t{identity})
        .scalar(std::int32_t{program_block})
        .scalar(std::uint32_t{hit ? 1u : 0u});
    cache_contract = std::move(exact).release();
  });
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"pops.composite-spatial-cache", cache_contract}},
                                                prepared_execution_lane()))
    throw std::logic_error("composite spatial cache generations disagree across execution ranks");
  if (hit) {
    prepare_spatial_collectively([&] {
      if (found->second.block != program_block)
        throw std::logic_error("composite spatial resource changed its block owner");
    });
    return found->second.workspace;
  }
  std::shared_ptr<PreparedAmrSpatialResidual<Dim>> workspace;
  prepare_spatial_collectively([&] {
    std::vector<const field_type*> layouts, masks;
    std::vector<Real> measures;
    const auto count = program_resource_topology().levels;
    for (int level = 0; level < count; ++level) {
      with_program_resource_level(level, [&] { layouts.push_back(&state(program_block)); });
      masks.push_back(
          facade_->prepared_amr_block_level_active_mask(sys_block(program_block), level));
      measures.push_back(cell_measure_(facade_->prepared_amr_level_geometry(level)));
    }
    workspace = std::make_shared<PreparedAmrSpatialResidual<Dim>>(layouts, masks, measures, options,
                                                                  difference_step);
    for (int level = 0; level + 1 < count; ++level) {
      const auto& coarse = runtime_->hierarchy().layout(level);
      const auto& fine = runtime_->hierarchy().layout(level + 1);
      const auto ratio = fine.ratio_from_parent();
      const auto coarse_geom = facade_->prepared_amr_level_geometry(level);
      const auto fine_geom = facade_->prepared_amr_level_geometry(level + 1);
      const ::pops::amr::reflux::FaceRefinementMapping<Dim> mapping{coarse.domain().lo,
                                                                    fine.domain().lo};
      std::size_t maximum_faces = 1;
      for (int axis = 0; axis < Dim; ++axis)
        maximum_faces = checked_product_(maximum_faces, static_cast<std::size_t>(ratio[axis]),
                                         "composite diffusion faces");
      const ::pops::amr::reflux::MetricRefluxBudget budget{maximum_faces, maximum_faces, 1};
      const auto runtime_block = static_cast<std::size_t>(sys_block(program_block));
      ::pops::amr::reflux::CoarseFaceRefluxKey<Dim> query;
      query.owner = facade_->prepared_amr_multiblock_hierarchy_().block_identity(runtime_block);
      query.state = facade_->prepared_amr_block_state_identity_(runtime_block);
      query.levels = {level, level + 1};
      for (const auto& face : program_interface_faces_(static_cast<std::size_t>(level))) {
        query.axis = face.axis;
        query.coarse_face = face.coarse_face;
        workspace->interfaces.push_back(
            {static_cast<std::size_t>(level), face.axis, face.coarse_face, face.coarse_cell,
             ::pops::amr::reflux::fine_faces_for_coarse_face(query, ratio, mapping, budget),
             face_measure_(coarse_geom, face.axis), face_measure_(fine_geom, face.axis),
             Real(1) / cell_measure_(coarse_geom),
             face.side == ::pops::amr::reflux::CoarseCellFaceSide::Lower ? Real(-1) : Real(1)});
      }
    }
  });
  prepare_spatial_collectively([&] {
    using sample_key = std::tuple<std::size_t, int, std::array<int, Dim>>;
    std::map<sample_key, std::size_t> samples;
    auto sample = [&](std::size_t level, int axis, const Index<Dim>& face) {
      const sample_key key{level, axis, index_key_(face)};
      if (const auto existing = samples.find(key); existing != samples.end())
        return existing->second;
      const auto& prototype = workspace->candidate(level);
      std::size_t selected = prototype.layout().size();
      for (std::size_t patch = 0; patch < prototype.layout().size(); ++patch)
        if (nd::face_box(prototype.layout()[patch], axis).contains(face)) {
          selected = patch;
          break;
        }
      if (selected == prototype.layout().size())
        throw std::logic_error("composite diffusion interface has no exact face owner");
      const auto owner = prototype.distribution().replicated()
                             ? prototype.rank_space().coordinate(0)
                             : prototype.distribution().owner(selected);
      const bool owned = owner == prototype.local_rank();
      const auto local = owned ? prototype.local_index_of(selected) : field_type::not_local;
      if (owned && local == field_type::not_local)
        throw std::logic_error("composite diffusion interface lost its canonical local patch");
      const auto index = workspace->face_samples.size();
      workspace->face_samples.push_back({level, local, axis, face, owned});
      samples.emplace(key, index);
      return index;
    };
    for (auto& face : workspace->interfaces) {
      face.coarse_sample = sample(face.parent, face.axis, face.coarse_face);
      face.fine_samples.reserve(face.fine_faces.size());
      for (const auto& coordinate : face.fine_faces)
        face.fine_samples.push_back(sample(face.parent + 1, face.axis, coordinate));
    }
    const auto count = workspace->face_samples.size();
    workspace->face_gather =
        decltype(workspace->face_gather)("composite_spatial_face_gather", count);
    workspace->face_gather_host =
        decltype(workspace->face_gather_host)("composite_spatial_face_gather_host", count);
    workspace->face_values =
        decltype(workspace->face_values)("composite_spatial_face_values", count);
    workspace->face_values_host =
        decltype(workspace->face_values_host)("composite_spatial_face_values_host", count);
  });
  // Publish the provisional cache node without allocating or copying its prepared resource.
  decltype(spatial_hierarchy_resources_) pending;
  prepare_spatial_collectively([&] {
    pending.emplace(identity,
                    SpatialHierarchyResource{epoch, generation, program_block, workspace,
                                             std::vector<FluxExpression>(workspace->levels())});
  });
  auto node = pending.extract(identity);
  if (found == spatial_hierarchy_resources_.end())
    spatial_hierarchy_resources_.insert(std::move(node));
  else {
    static_assert(std::is_nothrow_move_assignable_v<SpatialHierarchyResource>);
    found->second = std::move(node.mapped());
  }
  return workspace;
}

// Use the current conserved parent; failures restore the untouched accepted/old envelope.
template <class Evaluate>
auto with_spatial_parent(int program_block, const field_type* parent, Evaluate&& evaluate) const {
  const auto block = static_cast<std::size_t>(sys_block(program_block));
  if (active_attempt_states_.empty() || block >= active_staged_parents_.size())
    throw std::logic_error("composite spatial evaluation requires a synchronized attempt envelope");
  const field_type* previous = active_staged_parents_[block];
  active_staged_parents_[block] = parent;
  try {
    auto result = evaluate();
    active_staged_parents_[block] = previous;
    return result;
  } catch (...) {
    active_staged_parents_[block] = previous;
    throw;
  }
}

void prepare_spatial_face_gather(PreparedAmrSpatialResidual<Dim>& workspace) const {
  prepare_spatial_collectively([&] {
    for (std::size_t index = 0; index < workspace.face_samples.size(); ++index) {
      const auto& sample = workspace.face_samples[index];
      auto& entry = workspace.face_gather_host(index);
      entry.face = sample.face;
      entry.owned = sample.owned;
      if (!sample.owned)
        continue;
      const auto& faces = *workspace.faces.at(sample.level);
      if (sample.local_patch >= faces.size())
        throw std::logic_error("composite diffusion lost a prepared field-owned face buffer");
      entry.density = faces[sample.local_patch].view().axes[sample.axis];
    }
    Kokkos::deep_copy(workspace.face_gather, workspace.face_gather_host);
  });
}

void gather_spatial_face_densities(PreparedAmrSpatialResidual<Dim>& workspace) const {
  prepare_spatial_collectively([&] {
    const auto entries = workspace.face_gather;
    const auto values = workspace.face_values;
    Kokkos::parallel_for(
        "pops_composite_diffusion_faces", Kokkos::RangePolicy<>(0, entries.extent(0)),
        KOKKOS_LAMBDA(const std::size_t index) {
          const auto entry = entries(index);
          values(index) = entry.owned ? entry.density(entry.face, 0) : Real(0);
        });
    Kokkos::deep_copy(workspace.face_values_host, values);
  });
  all_reduce_sum_inplace(workspace.face_values_host.data(), workspace.face_values_host.extent(0),
                         prepared_execution_lane().communicator());
}

SolveReport solve_spatial_hierarchy(PreparedAmrSpatialResidual<Dim>& workspace) const {
  using hierarchy_type = typename PreparedAmrSpatialResidual<Dim>::hierarchy_type;
  prepare_spatial_face_gather(workspace);
  auto evaluate = [&](const hierarchy_type& q, const hierarchy_type& previous,
                      hierarchy_type& result, int) {
    // Map solver coordinates, then restrict the fine iterate; covered coarse cells are not unknowns.
    for (std::size_t level = 0; level < q.size(); ++level)
      with_program_attempt_level(static_cast<int>(level), [&] {
        workspace.conserved[level] = workspace.mappings.at(level)(q[level]);
      });
    for (std::size_t child = q.size(); child-- > 1;)
      ::pops::numerics::time::amr::execute_average_down_collectively(
          *runtime_, child, *workspace.conserved[child], *workspace.conserved[child - 1],
          prepared_execution_lane());
    const field_type* parent = nullptr;
    for (std::size_t level = 0; level < q.size(); ++level)
      with_program_attempt_level(static_cast<int>(level), [&] {
        parent = workspace.evaluations.at(level)(*workspace.conserved[level], previous[level],
                                                 result[level], parent);
      });
    // R=div(A grad W). Replace its coarse face density by the fine area sum inside F.
    // The generated weight already includes -tau and the occurrence coefficient exactly once.
    gather_spatial_face_densities(workspace);
    for (const auto& face : workspace.interfaces) {
      const auto level = face.parent;
      const Real coarse = workspace.face_values_host(face.coarse_sample);
      Real fine = Real(0);
      for (const auto index : face.fine_samples)
        fine += workspace.face_values_host(index);
      const Real correction = workspace.residual_face_weight * face.divergence_sign *
                              face.inverse_volume *
                              (face.fine_area * fine - face.coarse_area * coarse);
      for (std::size_t local = 0; local < result[level].local_size(); ++local)
        if (result[level].box(local).contains(face.coarse_cell)) {
          const auto destination = result[level].fab(local).view();
          for_each_cell(
              Box<Dim>{face.coarse_cell, face.coarse_cell},
              [=] POPS_HD(const Index<Dim>& cell) { destination(cell, 0) += correction; });
        }
    }
    device_fence();
  };
  return workspace.solve(evaluate, prepared_execution_lane());
}
