// Invocation-owned path residuals. Conservative flux, signed recipient terms and speed remain
// separate until all levels of this exact synchronous SSA stage have reached the barrier.
private:
struct StagedPathRhs {
  int block = -1, runtime_block = -1, level = -1, rate = -1;
  field_type* input = nullptr;
  field_type* output = nullptr;
  std::optional<RhsInputTrace> trace;
  runtime::multiblock::BoundaryEvaluationPoint point;
  std::string temporal_family, operator_identity;
  double courant = 0;
};
mutable std::map<int, StagedPathRhs> staged_path_rhs_;

public:
double path_rhs_courant() const { return facade_->active_program_step_courant_(); }

void stage_path_rhs(int program_block, field_type& input, field_type& output, int rate_id,
                    std::string_view temporal_family, const RhsInputTrace* input_trace,
                    double courant) const {
  const ExecutionLane& lane = prepared_execution_lane();
  // These facade getters may materialize/enter collectives, so call them in common order.
  const bool shared = facade_->prepared_amr_multiblock_hierarchy_().has_interface_flux_provider();
  int block = -1;
  std::string path_identity;
  std::optional<StagedPathRhs> candidate;
  std::string exact;
  std::exception_ptr error;
  try {
    block = sys_block(program_block);
    path_identity = facade_->prepared_amr_block_path_operator_identity_(block, active_level_);
    require_rate_identity_(rate_id);
    if (!mapping_continuation_ || !mapping_continuation_->executing ||
        !multiblock_subcycling_ || !multiblock_subcycling_->has_synchronized_groups() ||
        active_attempt_states_.empty() || active_level_ < 0 || active_level_ >= nlev() ||
        logical_substep_ != 0 || cell_temporal_configuration_ || shared ||
        staged_path_rhs_.contains(active_level_) || path_identity.empty())
      throw std::invalid_argument("AMR path RHS requires one unique synchronous level stage");
    if (facade_->prepared_amr_block_level_active_mask(block, active_level_) != nullptr)
      throw std::invalid_argument("AMR path RHS has no embedded-boundary side authority");
    for (const auto& relation : facade_->prepared_program_temporal_relations())
      if (relation.temporal_ratio() != ::pops::amr::Rational(1, 1))
        throw std::invalid_argument("AMR path RHS requires equal parent/child clock windows");
    if (!std::isfinite(courant) || !(courant > 0.0) ||
        courant != facade_->active_program_step_courant_())
      throw std::invalid_argument("AMR path RHS Courant differs from its authored invocation");
    const auto destination = scratches_.find(
        ScratchKey{ScratchKind::Rhs, active_level_, block, rate_id, 0});
    if (input_trace == nullptr || input.shares_storage_with(output) ||
        scratch_prototype_owner_(input) != block || destination == scratches_.end() ||
        &destination->second != &output)
      throw std::invalid_argument("AMR path RHS requires owned immutable SSA input and output");
    require_same_field_contract_(input, output, "AMR path RHS output");
    if (staged_path_rhs_.size() >= static_cast<std::size_t>(nlev()))
      throw std::length_error("AMR path RHS exceeds its exact hierarchy stage capacity");
    candidate.emplace(StagedPathRhs{program_block, block, active_level_, rate_id, &input, &output,
                                   *input_trace, boundary_evaluation_point(rate_id),
                                   std::string(temporal_family), path_identity, courant});
    ExactContractBuilder contract;
    contract.text("pops.amr-path-rhs-stage.v1")
        .scalar(block).scalar(active_level_).scalar(rate_id).scalar(courant)
        .scalar(active_subcycling_attempt_).scalar(current_dt_)
        .scalar(stage_time_.numerator).scalar(stage_time_.denominator)
        .text(temporal_family).bytes(path_identity)
        .bytes(mapping_continuation_->hierarchy_authority);
    exact = std::move(contract).release();
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "AMR path RHS staging failed collectively");
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"path-rhs-stage", exact}}, lane))
    throw std::invalid_argument("AMR path RHS stage differs between ranks");
  // Authentication and snapshot ownership are independent of the caller's live scratch contents.
  (void)rhs_input_parent_(block, input, rate_id, &*candidate->trace);
  std::map<int, StagedPathRhs> next;
  error = nullptr;
  try {
    // Immutable trace tokens support copy construction, never assignment to an existing token.
    next.insert(staged_path_rhs_.begin(), staged_path_rhs_.end());
    next.emplace(active_level_, std::move(*candidate));
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "AMR path RHS reservation failed collectively");
  staged_path_rhs_.swap(next);
}

private:
// One owner supplies the whole tuple. A source failure is synchronized before payload MPI.
std::vector<Real> collective_path_face_(const level_evaluation_type& evaluation,
                                        const field_type& field, int axis,
                                        const Index<Dim>& face) const {
  const ExecutionLane& lane = prepared_execution_lane();
  std::vector<Real> result;
  long supplied = 0;
  std::exception_ptr error;
  try {
    if (!evaluation.path_faces || axis < 0 || axis >= Dim)
      throw std::invalid_argument("AMR path face query has no typed evaluation");
    const auto& path = *evaluation.path_faces;
    path.require_layout(field, path.operator_identity);
    const std::size_t components = static_cast<std::size_t>(field.ncomp());
    const std::size_t width = checked_product_(components, 3, "AMR path tuple width") + 1;
    result.assign(width, Real(0));
    std::size_t selected = field.layout().size();
    for (std::size_t global = 0; global < field.layout().size(); ++global)
      if (nd::face_box(field.layout()[global], axis).contains(face)) {
        selected = global;
        break;
      }
    if (selected == field.layout().size())
      throw std::out_of_range("AMR path face has no canonical patch owner");
    const Index<Dim> owner = field.distribution().replicated()
                                 ? field.rank_space().coordinate(0)
                                 : field.distribution().owner(selected);
    if (owner == field.local_rank()) {
      const std::size_t local = field.local_index_of(selected);
      if (local == field_type::not_local || local >= evaluation.integrated_face_fluxes.size())
        throw std::logic_error("AMR path face lost its canonical local owner");
      const auto flux = evaluation.integrated_face_fluxes[local].view().axes[axis];
      const auto left = path.integrated_left_ncp[local].view().axes[axis];
      const auto right = path.integrated_right_ncp[local].view().axes[axis];
      const auto speed = path.speed_bounds[local].view().axes[axis];
      Kokkos::View<Real*, MemorySpace> packed("pops_path_face_tuple", width);
      Kokkos::parallel_for("pops_path_face_tuple_gather", Kokkos::RangePolicy<>(0, width),
                          [=] POPS_HD(std::size_t entry) {
        packed(entry) = entry < components ? flux(face, static_cast<int>(entry))
                        : entry < 2 * components
                            ? left(face, static_cast<int>(entry - components))
                        : entry < 3 * components
                            ? right(face, static_cast<int>(entry - 2 * components))
                            : speed(face, 0);
      });
      const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, packed);
      for (std::size_t entry = 0; entry < width; ++entry)
        result[entry] = host(entry);
      supplied = 1;
    }
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "AMR path face gathering failed collectively");
  if (all_reduce_sum(supplied, lane) != 1)
    throw std::invalid_argument("AMR path face has missing or duplicate canonical ownership");
  all_reduce_sum_inplace(result.data(), result.size(), lane);
  for (Real value : result)
    if (!std::isfinite(static_cast<double>(value)))
      throw std::runtime_error("AMR path face tuple contains a non-finite value");
  if (result.back() < Real(0))
    throw std::runtime_error("AMR path face tuple contains a negative speed bound");
  return result;
}

void replace_path_face_(level_evaluation_type& evaluation, const field_type& field, int axis,
                        const Index<Dim>& face, const std::vector<Real>& tuple,
                        const Geometry<Dim>& geometry) const {
  const int components = field.ncomp();
  if (!evaluation.path_faces || tuple.size() != static_cast<std::size_t>(3 * components + 1))
    throw std::invalid_argument("AMR path replacement has a foreign typed tuple");
  std::vector<Index<Dim>> aliases{face};
  const auto topology = facade_->prepared_amr_boundary_topology();
  if (topology.is_periodic(Face<Dim>{axis, BoundarySide::lower})) {
    Index<Dim> alias = face;
    if (face[axis] == geometry.domain().lo[axis])
      alias[axis] = geometry.domain().hi[axis] + 1;
    else if (face[axis] == geometry.domain().hi[axis] + 1)
      alias[axis] = geometry.domain().lo[axis];
    if (alias != face)
      aliases.push_back(alias);
  }
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& path = *evaluation.path_faces;
    const auto flux = evaluation.integrated_face_fluxes[local].view().axes[axis];
    const auto left = path.integrated_left_ncp[local].view().axes[axis];
    const auto right = path.integrated_right_ncp[local].view().axes[axis];
    const auto speed = path.speed_bounds[local].view().axes[axis];
    for (const auto& target : aliases) {
      if (!nd::face_box(field.box(local), axis).contains(target))
        continue;
      for (int component = 0; component < components; ++component) {
        const Real f = tuple[static_cast<std::size_t>(component)];
        const Real l = tuple[static_cast<std::size_t>(components + component)];
        const Real r = tuple[static_cast<std::size_t>(2 * components + component)];
        const Real a = tuple.back();
        for_each_cell(Box<Dim>{target, target}, [=] POPS_HD(const Index<Dim>& index) {
          flux(index, component) = f;
          left(index, component) = l;
          right(index, component) = r;
          if (component == 0)
            speed(index, 0) = a;
        });
      }
    }
  }
}

void canonicalize_path_interfaces_(int runtime_block,
                                    const std::vector<level_evaluation_type*>& evaluations) const {
  const ExecutionLane& lane = prepared_execution_lane();
  for (std::size_t child = evaluations.size(); child-- > 1;) {
    const std::size_t parent = child - 1;
    auto& coarse = *evaluations[parent];
    const auto& fine = *evaluations[child];
    const auto& coarse_field = coarse.residual;
    const auto& fine_field = fine.residual;
    const auto coarse_geometry = facade_->prepared_amr_level_geometry(static_cast<int>(parent));
    const auto fine_geometry = facade_->prepared_amr_level_geometry(static_cast<int>(child));
    struct Query {
      ProgramInterfaceFace interface;
      std::vector<Index<Dim>> fine_faces;
      double fine_area = 0, coarse_area = 0, volume = 0;
    };
    std::vector<Query> queries;
    // Allocate/enumerate the entire common schedule before the first tuple MPI. No rank may
    // leave this phase while another has already begun gathering a face payload.
    prepare_spatial_collectively([&] {
      const auto& coarse_layout = runtime_->hierarchy().layout(parent);
      const auto& fine_layout = runtime_->hierarchy().layout(child);
      const auto ratio = fine_layout.ratio_from_parent();
      const ::pops::amr::reflux::FaceRefinementMapping<Dim> mapping{
          coarse_layout.domain().lo, fine_layout.domain().lo};
      std::vector<Box<Dim>> covered_fine;
      if (child + 1 < evaluations.size()) {
        const auto& descendant = runtime_->hierarchy().layout(child + 1);
        Extent<Dim> descendant_ratio{};
        for (int axis = 0; axis < Dim; ++axis)
          descendant_ratio[axis] = descendant.ratio_from_parent()[axis];
        for (const auto& patch : descendant.patches().boxes())
          covered_fine.push_back(pops::coarsen(patch, descendant_ratio));
      }
      const auto interfaces = program_interface_faces_(parent);
      queries.reserve(interfaces.size());
      for (const auto& interface : interfaces) {
        ::pops::amr::reflux::CoarseFaceRefluxKey<Dim> key;
        key.owner = std::string(active_block_identities_.at(static_cast<std::size_t>(runtime_block)));
        key.state = facade_->prepared_amr_block_state_identity_(runtime_block);
        key.levels = {static_cast<int>(parent), static_cast<int>(child)};
        key.axis = interface.axis;
        key.coarse_face = interface.coarse_face;
        key.attempt = active_subcycling_attempt_;
        key.macro_step = active_subcycling_window_.begin.macro_step;
        key.window_begin = active_subcycling_window_.begin.phase;
        key.window_end = active_subcycling_window_.end.phase;
        std::size_t maximum_faces = 1;
        for (int axis = 0; axis < Dim; ++axis)
          if (axis != interface.axis)
            maximum_faces = checked_product_(maximum_faces, static_cast<std::size_t>(ratio[axis]),
                                             "AMR path subface budget");
        const ::pops::amr::reflux::MetricRefluxBudget budget{maximum_faces, maximum_faces, 1};
        Query query{interface,
                    ::pops::amr::reflux::fine_faces_for_coarse_face(key, ratio, mapping, budget),
                    face_measure_(fine_geometry, interface.axis),
                    face_measure_(coarse_geometry, interface.axis), cell_measure_(coarse_geometry)};
        if (!std::isfinite(query.fine_area) || !(query.fine_area > 0) ||
            !std::isfinite(query.coarse_area) || !(query.coarse_area > 0) ||
            !std::isfinite(query.volume) || !(query.volume > 0))
          throw std::invalid_argument("AMR path replacement has invalid prepared metric measures");
        for (auto fine_cell : query.fine_faces) {
          if (interface.side == ::pops::amr::reflux::CoarseCellFaceSide::Lower)
            --fine_cell[interface.axis];
          const bool present = std::any_of(fine_layout.patches().boxes().begin(),
              fine_layout.patches().boxes().end(),
              [&](const Box<Dim>& patch) { return patch.contains(fine_cell); });
          if (!present || std::any_of(covered_fine.begin(), covered_fine.end(),
                  [&](const Box<Dim>& patch) { return patch.contains(fine_cell); }))
            throw std::invalid_argument(
                "AMR path interface requires an active adjacent fine recipient; level jumps "
                "need a different canonical interface authority");
        }
        queries.push_back(std::move(query));
      }
    });
    for (const auto& query : queries) {
      const auto& interface = query.interface;
      Index<Dim> recipient_face = interface.coarse_cell;
      if (interface.side == ::pops::amr::reflux::CoarseCellFaceSide::Upper)
        ++recipient_face[interface.axis];
      const auto prior = collective_path_face_(coarse, coarse_field, interface.axis, recipient_face);
      std::vector<Real> replacement;
      prepare_spatial_collectively([&] { replacement.assign(prior.size(), Real(0)); });
      for (const auto& subface : query.fine_faces) {
        const auto sample = collective_path_face_(fine, fine_field, interface.axis, subface);
        for (std::size_t component = 0; component + 1 < sample.size(); ++component)
          replacement[component] += sample[component];
        replacement.back() += static_cast<Real>(query.fine_area / query.coarse_area) * sample.back();
      }
      std::exception_ptr error;
      try {
        for (Real value : replacement)
          if (!std::isfinite(static_cast<double>(value)))
            throw std::runtime_error("AMR path subface sum overflowed");
        const int components = coarse_field.ncomp();
        const bool left = interface.side == ::pops::amr::reflux::CoarseCellFaceSide::Upper;
        const std::size_t side_offset = static_cast<std::size_t>((left ? 1 : 2) * components);
        const Real sign = left ? Real(-1) : Real(1);
        std::vector<Real> correction(static_cast<std::size_t>(components));
        for (int component = 0; component < components; ++component) {
          const std::size_t k = static_cast<std::size_t>(component);
          correction[k] = (sign * (replacement[k] - prior[k]) +
                           replacement[side_offset + k] - prior[side_offset + k]) /
                          static_cast<Real>(query.volume);
          if (!std::isfinite(static_cast<double>(correction[k])))
            throw std::runtime_error("AMR path coarse residual correction overflowed");
        }
        replace_path_face_(coarse, coarse_field, interface.axis, interface.coarse_face,
                           replacement, coarse_geometry);
        apply_reflux_payload_(coarse.residual, interface.coarse_cell, correction);
        device_fence();
      } catch (...) {
        error = std::current_exception();
      }
      collectively_rethrow_exception(error, lane,
                                     "AMR canonical path replacement failed collectively");
    }
  }
}

void validate_path_stage_values_(const StagedPathRhs& request,
                                 const level_evaluation_type& evaluation) const {
  const ExecutionLane& lane = prepared_execution_lane();
  const auto geometry = facade_->prepared_amr_level_geometry(request.level);
  Real local_frequency = Real(0);
  std::exception_ptr error;
  try {
    if (!evaluation.path_faces ||
        evaluation.path_faces->operator_identity != request.operator_identity)
      throw std::invalid_argument("AMR path RHS lost its exact path operator identity");
    const auto& path = *evaluation.path_faces;
    path.require_layout(evaluation.residual, request.operator_identity);
    const Real volume = static_cast<Real>(cell_measure_(geometry));
    if (!Kokkos::isfinite(volume) || !(volume > Real(0)))
      throw std::invalid_argument("AMR path CFL has an invalid cell measure");
    Real areas[Dim]{};
    for (int axis = 0; axis < Dim; ++axis) {
      areas[axis] = static_cast<Real>(face_measure_(geometry, axis));
      if (!Kokkos::isfinite(areas[axis]) || !(areas[axis] > Real(0)))
        throw std::invalid_argument("AMR path CFL has an invalid face measure");
    }
    const auto& coverage = facade_->prepared_amr_block_level_coverage_mask(request.runtime_block,
                                                                          request.level);
    for (std::size_t local = 0; local < evaluation.residual.local_size(); ++local) {
      const auto rhs = std::as_const(evaluation.residual.fab(local)).view();
      const auto mask = std::as_const(coverage.fab(local)).view();
      const auto speeds = path.speed_bounds[local].view();
      const int components = evaluation.residual.ncomp();
      const Real maximum = for_each_cell_reduce_max(evaluation.residual.box(local),
          [=] POPS_HD(const Index<Dim>& cell) {
        for (int component = 0; component < components; ++component)
          if (!Kokkos::isfinite(rhs(cell, component)))
            return std::numeric_limits<Real>::infinity();
        if (mask(cell, 0) < Real(0.5))
          return Real(0);
        Real frequency = Real(0);
        for (int axis = 0; axis < Dim; ++axis) {
          Index<Dim> upper = cell;
          ++upper[axis];
          const Real lower_speed = speeds.axes[axis](cell, 0);
          const Real upper_speed = speeds.axes[axis](upper, 0);
          if (!Kokkos::isfinite(lower_speed) || !Kokkos::isfinite(upper_speed) ||
              lower_speed < Real(0) || upper_speed < Real(0))
            return std::numeric_limits<Real>::infinity();
          frequency += areas[axis] * (lower_speed + upper_speed) / volume;
        }
        return Kokkos::isfinite(frequency) ? frequency : std::numeric_limits<Real>::infinity();
      });
      local_frequency = std::max(local_frequency, maximum);
    }
    device_fence();
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "AMR path stage validation failed collectively");
  const double frequency = static_cast<double>(all_reduce_max(local_frequency, lane));
  const double scaled = request.point.dt * frequency;
  if (!std::isfinite(frequency) || !std::isfinite(scaled) || scaled > request.courant) {
    std::ostringstream message;
    message << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "AMR path RHS refused publication: actual face CFL exceeds authored Courant; level="
            << request.level << " dt=" << request.point.dt << " frequency=" << frequency
            << " dt_frequency=" << scaled << " courant=" << request.courant;
    throw std::runtime_error(message.str());
  }
}

public:
void publish_staged_path_rhs(int rate_id) const {
  const ExecutionLane& lane = prepared_execution_lane();
  struct Publication {
    StagedPathRhs* request = nullptr;
    level_evaluation_type* evaluation = nullptr;
    field_type backup;
    FluxExpressionRegistry registry;
    std::vector<std::size_t> counts;
    std::uint64_t next_identity = 0;
    FluxExpressionRegistry* registry_target = nullptr;
    std::vector<std::size_t>* counts_target = nullptr;
    std::uint64_t* identity_target = nullptr;
  };
  std::vector<Publication> publications;
  std::vector<level_evaluation_type*> evaluations;
  std::vector<std::pair<int, int>> targets;
  try {
    prepare_spatial_collectively([&] {
      if (!mapping_continuation_ || mapping_continuation_->executing || active_level_ != 0 ||
          mapping_continuation_->ports.empty() ||
          mapping_continuation_->ports.front().kind != MappingPortKind::spatial_rhs ||
          !multiblock_subcycling_ || !multiblock_subcycling_->has_synchronized_groups() ||
          staged_path_rhs_.size() != static_cast<std::size_t>(nlev()))
        throw std::logic_error("AMR path publication requires its complete unique hierarchy barrier");
      publications.resize(static_cast<std::size_t>(nlev()));
      evaluations.resize(publications.size());
      targets.reserve(publications.size());
      const auto& first = staged_path_rhs_.at(0);
      if (!first.trace || !first.trace->current_)
        throw std::invalid_argument("AMR path hierarchy has no authenticated root trace");
      for (int level = 0; level < nlev(); ++level) {
        auto& request = staged_path_rhs_.at(level);
        if (request.level != level || request.rate != rate_id ||
            request.runtime_block != first.runtime_block || request.operator_identity != first.operator_identity ||
            request.temporal_family != first.temporal_family || request.courant != first.courant ||
            request.point.physical_time != first.point.physical_time ||
            request.point.dt != first.point.dt || !std::isfinite(request.point.dt) ||
            !(request.point.dt > 0) ||
            !request.trace || !request.trace->current_ ||
            !same_rhs_input_window_(*first.trace->current_, *request.trace->current_))
          throw std::invalid_argument("AMR path hierarchy contains different RHS occurrences");
        publications[static_cast<std::size_t>(level)].request = &request;
        targets.emplace_back(request.runtime_block, level);
      }
    });
    auto at_level = [&](int level, auto&& operation) {
      // The once callback already owns the root envelope; it must not try to acquire it twice.
      if (level == active_level_)
        std::forward<decltype(operation)>(operation)();
      else
        with_program_attempt_level(level, std::forward<decltype(operation)>(operation));
    };
    for (int level = 0; level < nlev(); ++level)
      at_level(level, [&] {
        auto& publication = publications[static_cast<std::size_t>(level)];
        auto& request = *publication.request;
        const auto* parent = rhs_input_parent_(request.runtime_block, *request.input, rate_id,
                                               &*request.trace);
        field_type immutable_input;
        prepare_spatial_collectively([&] { immutable_input = request.trace->current_->snapshot; });
        facade_->validate_prepared_amr_state_publication_candidate(request.runtime_block, level,
                                                                   immutable_input);
        publication.evaluation = &facade_->prepare_prepared_amr_block_level_path_rhs_at(
            request.runtime_block, request.point, immutable_input, level - 1, parent);
        evaluations[static_cast<std::size_t>(level)] = publication.evaluation;
      });
    canonicalize_path_interfaces_(publications.front().request->runtime_block, evaluations);
    for (int level = 0; level < nlev(); ++level)
      at_level(level, [&] {
        auto& publication = publications[static_cast<std::size_t>(level)];
        auto& request = *publication.request;
        validate_path_stage_values_(request, *publication.evaluation);
        prepare_spatial_collectively([&] {
          publication.backup = rhs_scratch_like(*request.output);
          copy_valid_(*request.output, publication.backup);
          publication.registry = active_flux_expressions_;
          publication.counts = active_flux_basis_counts_;
          publication.next_identity = next_active_flux_basis_identity_;
        });
        const auto& evaluation = *publication.evaluation;
        prepare_active_flux_basis_impl_(request.runtime_block, evaluation.point, rate_id,
            FluxBasisProvider::PreparedDefaultFlux, request.temporal_family,
            evaluation.topology_epoch, evaluation.materialization_generation, *request.output,
            &evaluation, nullptr, nullptr, active_subcycling_window_, publication.registry,
            publication.counts, publication.next_identity);
      });
    facade_->validate_prepared_amr_block_level_batch(targets);
    prepare_spatial_collectively([&] {
      for (int level = 0; level < nlev(); ++level) {
        auto& publication = publications[static_cast<std::size_t>(level)];
        if (level == active_level_) {
          publication.registry_target = &active_flux_expressions_;
          publication.counts_target = &active_flux_basis_counts_;
          publication.identity_target = &next_active_flux_basis_identity_;
        } else {
          auto& envelope = synchronized_level_envelopes_.at(level);
          publication.registry_target = &envelope.active_flux_expressions_;
          publication.counts_target = &envelope.active_flux_basis_counts_;
          publication.identity_target = &envelope.next_active_flux_basis_identity_;
        }
      }
      count_kernel_(static_cast<std::int64_t>(publications.size()));
      device_fence();
    });
    std::exception_ptr publication_error;
    try {
      for (auto& publication : publications)
        copy_valid_(publication.evaluation->residual, *publication.request->output);
      device_fence();
    } catch (...) {
      publication_error = std::current_exception();
    }
    if (all_reduce_max(publication_error ? 1L : 0L, lane) != 0) {
      prepare_spatial_collectively([&] {
        for (auto& publication : publications)
          copy_valid_(publication.backup, *publication.request->output);
        device_fence();
      });
      collectively_rethrow_exception(publication_error, lane,
                                     "AMR path RHS output publication failed collectively");
    }
    static_assert(std::is_nothrow_swappable_v<FluxExpressionRegistry>);
    static_assert(std::is_nothrow_swappable_v<std::vector<std::size_t>>);
    for (auto& publication : publications) {
      publication.registry_target->swap(publication.registry);
      publication.counts_target->swap(publication.counts);
      *publication.identity_target = publication.next_identity;
    }
    facade_->publish_prepared_amr_block_level_batch(targets);
    staged_path_rhs_.clear();
  } catch (...) {
    staged_path_rhs_.clear();
    throw;
  }
}

public:
