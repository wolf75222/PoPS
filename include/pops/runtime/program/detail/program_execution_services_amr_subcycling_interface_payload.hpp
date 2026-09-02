// Interface-route construction is cold; payload collection is the paired prepared hot path.
// This is intentionally a class-definition fragment included by subcycling_runtime.hpp.

void stage_prepared_publication_candidates_(std::size_t level, std::span<field_type*> pack) const {
  const ExecutionLane& lane = prepared_execution_lane();
  std::size_t blocks = 0;
  std::exception_ptr lookup_error;
  try {
    blocks = static_cast<std::size_t>(n_blocks());
    if (pack.size() != blocks)
      throw std::invalid_argument(
          "AMR Program publication staging pack lost its complete block set");
    hot_path_workspace_.require_bound(blocks, "AMR Program publication staging");
    if (level >= hot_path_workspace_.level_capacity)
      throw std::invalid_argument(
          "AMR Program publication staging level exceeds its prepared arena");
    for (std::size_t block = 0; block < blocks; ++block)
      if (pack[block] == nullptr)
        throw std::invalid_argument("AMR Program publication staging pack contains a null state");
  } catch (...) {
    lookup_error = std::current_exception();
  }
  if (all_reduce_max(lookup_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && lookup_error)
      std::rethrow_exception(lookup_error);
    throw std::runtime_error("AMR Program publication staging mask lookup failed collectively");
  }
  for (std::size_t block = 0; block < blocks; ++block) {
    const field_type* active = nullptr;
    lookup_error = nullptr;
    try {
      active = facade_->program_prepared_amr_block_level_active_mask_(static_cast<int>(block),
                                                                      static_cast<int>(level));
    } catch (...) {
      lookup_error = std::current_exception();
    }
    if (all_reduce_max(lookup_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && lookup_error)
        std::rethrow_exception(lookup_error);
      throw std::runtime_error("AMR Program publication staging mask lookup failed collectively");
    }
    const long masked = active != nullptr ? 1L : 0L;
    if (all_reduce_min(masked, lane) != all_reduce_max(masked, lane))
      throw std::runtime_error(
          "AMR Program publication staging mask classification differs between ranks");
    if (active == nullptr)
      continue;
    std::exception_ptr stage_error;
    try {
      const int runtime_block = static_cast<int>(block);
      const int live_level = static_cast<int>(level);
      const field_type& accepted =
          facade_->program_prepared_amr_block_state_(runtime_block, live_level);
      require_same_field_contract_(*pack[block], accepted,
                                   "AMR Program publication staging candidate");
      require_same_layout_(*active, accepted, "AMR Program publication staging active mask");
      if (active->ncomp() != 1)
        throw std::invalid_argument(
            "AMR Program publication staging active mask requires one component");
      field_type& staged = hot_path_workspace_.publication_candidates.at(
          level * hot_path_workspace_.block_capacity + block);
      copy_full_(accepted, staged);
      const int components = pack[block]->ncomp();
      for (std::size_t local = 0; local < pack[block]->local_size(); ++local) {
        const auto& source_fab = pack[block]->fab(local);
        auto& destination_fab = staged.fab(local);
        const auto& active_fab = active->fab(local);
        if (pack[block]->global_index(local) != accepted.global_index(local) ||
            pack[block]->global_index(local) != active->global_index(local) ||
            source_fab.box() != destination_fab.box() || source_fab.box() != active_fab.box())
          throw std::invalid_argument(
              "AMR Program publication staging encountered different exact local patch ownership");
        const FieldView<const Real, Dim> source = std::as_const(source_fab).view();
        const FieldView<Real, Dim> destination = destination_fab.view();
        const FieldView<const Real, Dim> mask = std::as_const(active_fab).view();
        for_each_cell(source_fab.box(), [=] POPS_HD(const Index<Dim>& cell) {
          if (mask(cell, 0) < Real(0.5))
            return;
          for (int component = 0; component < components; ++component)
            destination(cell, component) = source(cell, component);
        });
      }
      ::pops::device_fence(prepared_hot_fence_label_());
      copy_full_(staged, *pack[block]);
    } catch (...) {
      stage_error = std::current_exception();
    }
    if (all_reduce_max(stage_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && stage_error)
        std::rethrow_exception(stage_error);
      throw std::runtime_error("AMR Program publication staging failed collectively");
    }
  }
}

void clear_active_multiblock_group_() const noexcept {
  active_attempt_states_.clear();
  active_staged_parents_.clear();
  active_incoming_flux_.clear();
  active_outgoing_flux_.clear();
  active_block_identities_.clear();
  active_flux_basis_counts_.clear();
  active_flux_expressions_.clear();
  reset_static_flux_active_state_();
  next_active_flux_basis_identity_ = 0;
  active_subcycling_attempt_ = 0;
}

static std::array<int, Dim> index_key_(const Index<Dim>& index) {
  std::array<int, Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[static_cast<std::size_t>(axis)] = index[axis];
  return result;
}

static std::vector<Index<Dim>> cells_in_box_(const Box<Dim>& box) {
  const std::size_t cells = static_cast<std::size_t>(box.numPts());
  std::vector<Index<Dim>> result;
  result.reserve(cells);
  for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
    std::size_t remainder = ordinal;
    Index<Dim> cell{};
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t length = static_cast<std::size_t>(box.length(axis));
      cell[axis] = box.lo[axis] + static_cast<int>(remainder % length);
      remainder /= length;
    }
    result.push_back(cell);
  }
  return result;
}

static std::vector<ProgramInterfaceFace> program_interface_faces_(
    const hierarchy_type& hierarchy, const BoundaryTopology<Dim>& topology,
    std::size_t parent_level) {
  if (parent_level + 1 >= hierarchy.num_levels())
    throw std::out_of_range("AMR Program interface faces exceed the prepared hierarchy");
  const auto& parent = hierarchy.layout(parent_level);
  const auto& child = hierarchy.layout(parent_level + 1);
  Extent<Dim> ratio{};
  for (int axis = 0; axis < Dim; ++axis)
    ratio[axis] = child.ratio_from_parent()[axis];
  std::set<std::array<int, Dim>> covered;
  for (const Box<Dim>& fine_patch : child.patches().boxes())
    for (const Index<Dim>& cell : cells_in_box_(pops::coarsen(fine_patch, ratio)))
      covered.insert(index_key_(cell));

  std::vector<ProgramInterfaceFace> result;
  for (const auto& coordinate : covered) {
    Index<Dim> inside{};
    for (int axis = 0; axis < Dim; ++axis)
      inside[axis] = coordinate[static_cast<std::size_t>(axis)];
    for (int axis = 0; axis < Dim; ++axis) {
      for (int direction : {-1, 1}) {
        Index<Dim> outside = inside;
        outside[axis] += direction;
        if (!parent.domain().contains(outside)) {
          const BoundarySide side = direction < 0 ? BoundarySide::lower : BoundarySide::upper;
          if (!topology.is_periodic(Face<Dim>{axis, side}))
            continue;
          outside[axis] = direction < 0 ? parent.domain().hi[axis] : parent.domain().lo[axis];
        }
        const bool parent_cell =
            std::any_of(parent.patches().boxes().begin(), parent.patches().boxes().end(),
                        [&](const Box<Dim>& patch) { return patch.contains(outside); });
        if (!parent_cell || covered.contains(index_key_(outside)))
          continue;
        Index<Dim> face = inside;
        if (direction > 0)
          ++face[axis];
        result.push_back({axis, face, outside,
                          direction > 0 ? ::pops::amr::reflux::CoarseCellFaceSide::Lower
                                        : ::pops::amr::reflux::CoarseCellFaceSide::Upper});
      }
    }
  }
  return result;
}

std::vector<ProgramInterfaceFace> program_interface_faces_(std::size_t parent_level) const {
  if (runtime_ == nullptr)
    throw std::logic_error("AMR Program interface faces require one prepared runtime");
  if (preparation_view_ != nullptr) {
    std::array<bool, Dim> periodic{};
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t lower = static_cast<std::size_t>(2 * axis);
      const std::size_t upper = lower + 1U;
      if (preparation_view_->periodic_faces.at(lower) !=
          preparation_view_->periodic_faces.at(upper))
        throw std::logic_error("AMR Program preparation has asymmetric periodic faces");
      periodic[static_cast<std::size_t>(axis)] = preparation_view_->periodic_faces.at(lower);
    }
    return program_interface_faces_(runtime_->hierarchy(),
                                    BoundaryTopology<Dim>::axis_periodic(periodic), parent_level);
  }
  require_facade_execution_();
  return program_interface_faces_(runtime_->hierarchy(),
                                  facade_->program_prepared_amr_boundary_topology_(), parent_level);
}

void collective_face_payload_into_(const level_evaluation_type& evaluation, const field_type& field,
                                   int axis, const Index<Dim>& face,
                                   std::span<Real> payload) const {
  std::exception_ptr local_error;
  try {
    std::size_t selected = field.layout().size();
    for (std::size_t global = 0; global < field.layout().size(); ++global)
      if (nd::face_box(field.layout()[global], axis).contains(face)) {
        selected = global;
        break;
      }
    if (selected == field.layout().size())
      throw std::out_of_range("AMR Program interface face has no level flux patch");
    const Index<Dim> owner = field.distribution().replicated()
                                 ? field.rank_space().coordinate(0)
                                 : field.distribution().owner(selected);
    if (payload.size() != static_cast<std::size_t>(field.ncomp()))
      throw std::invalid_argument(
          "AMR Program prepared face payload has the wrong component count");
    std::fill(payload.begin(), payload.end(), Real(0));
    if (owner == field.local_rank()) {
      const std::size_t local = field.local_index_of(selected);
      if (local == field_type::not_local || local >= evaluation.integrated_face_fluxes.size())
        throw std::runtime_error("AMR Program interface flux storage lost its local patch");
      const FieldView<const Real, Dim> values =
          evaluation.integrated_face_fluxes[local].view().axes[axis];
      hot_path_workspace_.require_sum_reduction("AMR Program prepared interface-face payload");
      const typename PreparedHotPathWorkspace::sum_execution_space execution{};
      const Box<Dim> face_point{face, face};
      for (int component = 0; component < field.ncomp(); ++component) {
        payload[static_cast<std::size_t>(component)] = ::pops::for_each_cell_reduce_sum(
            execution, hot_path_workspace_.sum_reduction, face_point,
            [=] POPS_HD(const Index<Dim>& sample) { return values(sample, component); });
      }
    }
  } catch (...) {
    local_error = std::current_exception();
  }
  const ExecutionLane& lane = prepared_execution_lane();
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program interface flux failed on another MPI rank");
  }
  for (Real& value : payload)
    value = all_reduce_sum(value, lane);
}

std::vector<Real> collective_face_payload_(const level_evaluation_type& evaluation,
                                           const field_type& field, int axis,
                                           const Index<Dim>& face) const {
  std::vector<Real> payload(static_cast<std::size_t>(field.ncomp()));
  collective_face_payload_into_(evaluation, field, axis, face, payload);
  return payload;
}

POPS_HD static Real named_flux_face_value_(const FieldView<const Real, Dim>& flux,
                                           const Index<Dim>& face, int axis, int component) {
  Index<Dim> lower = face;
  --lower[axis];
  return Real(0.5) * (flux(lower, component) + flux(face, component));
}

void collective_named_face_payload_into_(const std::array<field_type*, Dim>& cell_fluxes,
                                         const field_type& field, int axis, const Index<Dim>& face,
                                         std::span<Real> payload) const {
  std::exception_ptr local_error;
  try {
    if (axis < 0 || axis >= Dim || cell_fluxes[static_cast<std::size_t>(axis)] == nullptr)
      throw std::out_of_range("AMR Program named face-flux axis is outside its exact rank");
    const field_type& flux = *cell_fluxes[static_cast<std::size_t>(axis)];
    std::size_t selected = field.layout().size();
    for (std::size_t global = 0; global < field.layout().size(); ++global)
      if (nd::face_box(field.layout()[global], axis).contains(face)) {
        selected = global;
        break;
      }
    if (selected == field.layout().size())
      throw std::out_of_range("AMR Program named interface face has no level flux patch");
    const Index<Dim> owner = field.distribution().replicated()
                                 ? field.rank_space().coordinate(0)
                                 : field.distribution().owner(selected);
    if (payload.size() != static_cast<std::size_t>(field.ncomp()))
      throw std::invalid_argument(
          "AMR Program prepared named-face payload has the wrong component count");
    std::fill(payload.begin(), payload.end(), Real(0));
    if (owner == field.local_rank()) {
      const std::size_t local = flux.local_index_of(selected);
      if (local == field_type::not_local || local >= flux.local_size())
        throw std::runtime_error("AMR Program named interface flux lost its local patch");
      const FieldView<const Real, Dim> values = std::as_const(flux).fab(local).view();
      hot_path_workspace_.require_sum_reduction("AMR Program prepared named-interface payload");
      const typename PreparedHotPathWorkspace::sum_execution_space execution{};
      const Box<Dim> face_point{face, face};
      for (int component = 0; component < field.ncomp(); ++component) {
        payload[static_cast<std::size_t>(component)] = ::pops::for_each_cell_reduce_sum(
            execution, hot_path_workspace_.sum_reduction, face_point,
            [=] POPS_HD(const Index<Dim>& sample) {
              return named_flux_face_value_(values, sample, axis, component);
            });
      }
    }
  } catch (...) {
    local_error = std::current_exception();
  }
  const ExecutionLane& lane = prepared_execution_lane();
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program named interface flux failed collectively");
  }
  for (Real& value : payload)
    value = all_reduce_sum(value, lane);
}

std::vector<Real> collective_named_face_payload_(const std::array<field_type*, Dim>& cell_fluxes,
                                                 const field_type& field, int axis,
                                                 const Index<Dim>& face) const {
  std::vector<Real> payload(static_cast<std::size_t>(field.ncomp()));
  collective_named_face_payload_into_(cell_fluxes, field, axis, face, payload);
  return payload;
}

void collective_face_payload_into_(const std::array<field_type, Dim>& integrated_face_fluxes,
                                   const field_type& field, int axis, const Index<Dim>& face,
                                   std::span<Real> payload) const {
  std::exception_ptr local_error;
  try {
    if (axis < 0 || axis >= Dim)
      throw std::out_of_range("AMR Program exact face-flux axis is outside its rank");
    const field_type& faces = integrated_face_fluxes[static_cast<std::size_t>(axis)];
    std::size_t selected = field.layout().size();
    for (std::size_t global = 0; global < field.layout().size(); ++global)
      if (nd::face_box(field.layout()[global], axis).contains(face)) {
        selected = global;
        break;
      }
    if (selected == field.layout().size())
      throw std::out_of_range("AMR Program exact interface face has no level flux patch");
    const Index<Dim> owner = field.distribution().replicated()
                                 ? field.rank_space().coordinate(0)
                                 : field.distribution().owner(selected);
    if (payload.size() != static_cast<std::size_t>(field.ncomp()))
      throw std::invalid_argument(
          "AMR Program prepared exact-face payload has the wrong component count");
    std::fill(payload.begin(), payload.end(), Real(0));
    if (owner == field.local_rank()) {
      const std::size_t local = faces.local_index_of(selected);
      if (local == field_type::not_local || local >= faces.local_size())
        throw std::runtime_error("AMR Program exact interface flux lost its local patch");
      const FieldView<const Real, Dim> values = faces.fab(local).view();
      hot_path_workspace_.require_sum_reduction("AMR Program prepared exact-interface payload");
      const typename PreparedHotPathWorkspace::sum_execution_space execution{};
      const Box<Dim> face_point{face, face};
      for (int component = 0; component < field.ncomp(); ++component) {
        payload[static_cast<std::size_t>(component)] = ::pops::for_each_cell_reduce_sum(
            execution, hot_path_workspace_.sum_reduction, face_point,
            [=] POPS_HD(const Index<Dim>& sample) { return values(sample, component); });
      }
    }
  } catch (...) {
    local_error = std::current_exception();
  }
  const ExecutionLane& lane = prepared_execution_lane();
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program exact interface flux failed on another MPI rank");
  }
  for (Real& value : payload)
    value = all_reduce_sum(value, lane);
}

std::vector<Real> collective_face_payload_(
    const std::array<field_type, Dim>& integrated_face_fluxes, const field_type& field, int axis,
    const Index<Dim>& face) const {
  std::vector<Real> payload(static_cast<std::size_t>(field.ncomp()));
  collective_face_payload_into_(integrated_face_fluxes, field, axis, face, payload);
  return payload;
}

static void payload_axpy_(std::vector<Real>& destination, double factor,
                          const std::vector<Real>& source) {
  if (destination.empty())
    destination.assign(source.size(), Real(0));
  if (destination.size() != source.size())
    throw std::invalid_argument("AMR Program reflux payload component counts differ");
  for (std::size_t component = 0; component < source.size(); ++component)
    destination[component] += static_cast<Real>(factor) * source[component];
}

static double face_measure_(const Geometry<Dim>& geometry, int normal_axis) {
  double result = 1.0;
  for (int axis = 0; axis < Dim; ++axis)
    if (axis != normal_axis)
      result *= static_cast<double>(geometry.spacing(axis));
  return result;
}

static double cell_measure_(const Geometry<Dim>& geometry) {
  double result = 1.0;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<double>(geometry.spacing(axis));
  return result;
}
