// The composite temporal solve owns provisional interface closure, independently of the
// accepted transport reflux ledger. No residual/JVP evaluation creates a FluxBasis.
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
  if (found != spatial_hierarchy_resources_.end() && found->second.epoch == epoch &&
      found->second.generation == generation) {
    if (found->second.block != program_block)
      throw std::logic_error("composite spatial resource changed its block owner");
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
      for (const auto& face : program_interface_faces_(static_cast<std::size_t>(level))) {
        ::pops::amr::reflux::CoarseFaceRefluxKey<Dim> query;
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
    spatial_hierarchy_resources_.insert_or_assign(
        identity, SpatialHierarchyResource{epoch, generation, program_block, workspace});
  });
  return workspace;
}

// A deferred explicit face correction must be applied to U_n before the implicit
// inverse, not dropped or appended to the solved state. An accepted state is already
// reconciled; only a newly produced, unreconciled stage expression is refused here.
void require_spatial_previous_reconciled(const field_type& previous) const {
  prepare_spatial_collectively([&] {
    if (!active_flux_expression_(previous).empty())
      throw std::invalid_argument(
          "composite implicit previous stage has unreconciled explicit face fluxes; "
          "a conservative synchronization barrier is required before the implicit solve");
  });
}

// The exact current conserved parent is an evaluation input. The accepted/old parent envelope
// remains unchanged and is restored even when a constitutive law rejects the candidate.
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

Real spatial_face_density(const typename PreparedAmrSpatialResidual<Dim>::faces_type& faces,
                          const field_type& prototype, int axis, const Index<Dim>& face) const {
  Real value = Real(0);
  prepare_spatial_collectively([&] {
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
    if (owner == prototype.local_rank()) {
      const auto local = prototype.local_index_of(selected);
      if (local == field_type::not_local || local >= faces.size())
        throw std::logic_error("composite diffusion interface lost its field-owned face buffer");
      const auto view = faces[local].view().axes[axis];
      Kokkos::parallel_reduce(
          "pops_composite_diffusion_face", Kokkos::RangePolicy<>(0, 1),
          [=] POPS_HD(int, Real& sum) { sum += view(face, 0); }, value);
      device_fence();
      if (!std::isfinite(value))
        throw std::runtime_error("composite diffusion face is non-finite");
    }
  });
  return all_reduce_sum(value, prepared_execution_lane());
}

SolveReport solve_spatial_hierarchy(PreparedAmrSpatialResidual<Dim>& workspace) const {
  using hierarchy_type = typename PreparedAmrSpatialResidual<Dim>::hierarchy_type;
  auto evaluate = [&](const hierarchy_type& q, const hierarchy_type& previous,
                      hierarchy_type& result, int) {
    // Map solver coordinates first, then restrict the current conserved fine iterate.
    // Covered coarse cells are not independent Newton unknowns or old-state ghost data.
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
    for (const auto& face : workspace.interfaces) {
      const auto level = face.parent;
      const Real coarse =
          spatial_face_density(*workspace.faces.at(level), q[level], face.axis, face.coarse_face);
      Real fine = Real(0);
      for (const auto& coordinate : face.fine_faces)
        fine += spatial_face_density(*workspace.faces.at(level + 1), q[level + 1], face.axis,
                                     coordinate);
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
