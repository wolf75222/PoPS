using HistoryFluxSnapshot = history_flux::Snapshot<Dim>;

std::shared_ptr<const HistoryFluxSnapshot> capture_history_flux_snapshot_(
    std::size_t block, std::uint64_t basis_identity,
    const runtime::multiblock::BoundaryEvaluationPoint& point, int rhs_identity,
    FluxBasisProvider provider, std::uint64_t topology_epoch, std::uint64_t generation,
    const field_type& rhs, const level_evaluation_type* evaluation,
    const std::array<field_type, Dim>* exact, const std::array<field_type*, Dim>* named) const {
  const auto& owners = runtime_state().hist_.owner;
  if (std::none_of(owners.begin(), owners.end(),
                   [&](const auto& entry) { return entry.second == static_cast<int>(block); }))
    return {};  // No history: retain the existing interface-only capture and communication cost.
  const auto& lane = prepared_execution_lane();
  std::shared_ptr<HistoryFluxSnapshot> snapshot;
  std::vector<char> digests;
  std::string contract;
  prepare_spatial_collectively([&] {
    snapshot = std::make_shared<HistoryFluxSnapshot>();
    snapshot->identity = "pending-physical-source-digest";
    snapshot->level = active_level_;
    snapshot->components = rhs.ncomp();
    const auto geometry = facade_->prepared_amr_level_geometry(active_level_);
    snapshot->domain = geometry.domain();
    snapshot->rank_count = lane.size();
    snapshot->local_rank = lane.rank();
    checkpoint_detail::Writer source;
    source.string("pops.amr.history-face-source-point.v1");
    source.u64(block);
    source.u64(basis_identity);
    source.i32(rhs_identity);
    source.u64(static_cast<std::uint64_t>(provider));
    source.u64(topology_epoch);
    source.u64(generation);
    source.string(point.clock);
    source.i64(point.tick);
    source.i32(point.level);
    source.i32(point.substep);
    source.i32(point.stage);
    source.i64(point.stage_fraction.numerator);
    source.i64(point.stage_fraction.denominator);
    source.real(point.dt);
    source.real(point.physical_time);
    source.string(point.graph_identity);
    source.string(point.rate_identity);
    source.string(point.application_identity);
    snapshot->source_identity = "pops.amr.history-face-source-point.v1:sha256:" +
                                identity::sha256_hex(std::move(source).take());
    for (int axis = 0; axis < Dim; ++axis)
      snapshot->cell_size[axis] = static_cast<double>(geometry.spacing(axis));
    for (std::size_t global = 0; global < rhs.layout().size(); ++global) {
      snapshot->patches.push_back(rhs.layout()[global]);
      snapshot->owners.push_back(
          rhs.distribution().replicated()
              ? 0
              : static_cast<int>(rhs.rank_space().linear_rank(rhs.distribution().owner(global))));
      if (snapshot->owners.back() != lane.rank())
        continue;
      const auto local = rhs.local_index_of(global);
      if (local == field_type::not_local)
        throw std::logic_error("AMR history flux source lost its owned patch");
      history_flux::OwnedPatch<Dim> patch;
      patch.global_patch = global;
      for (int axis = 0; axis < Dim; ++axis) {
        const auto face_box = nd::face_box(rhs.layout()[global], axis);
        const auto cells = history_flux::detail::points(face_box);
        const auto count = checked_product_(cells, static_cast<std::size_t>(rhs.ncomp()),
                                            "AMR owned history face sample count");
        FieldView<const Real, Dim> values;
        if (named != nullptr) {
          const auto& field = *(*named)[axis];
          const auto source_local = field.local_index_of(global);
          if (source_local == field_type::not_local)
            throw std::logic_error("AMR named history source lacks its owned patch");
          values = field.fab(source_local).view();
        } else if (exact != nullptr) {
          const auto& field = (*exact)[axis];
          const auto source_local = field.local_index_of(global);
          if (source_local == field_type::not_local)
            throw std::logic_error("AMR exact history source lacks its owned patch");
          values = field.fab(source_local).view();
        } else {
          values = evaluation->integrated_face_fluxes.at(local).view().axes[axis];
        }
        Kokkos::View<Real*, MemorySpace> packed("pops_owned_history_face_snapshot", count);
        const bool cell_provider = named != nullptr;
        const Real measure = static_cast<Real>(face_measure_(geometry, axis));
        const bool integrated = named == nullptr && provider != FluxBasisProvider::DiffusiveFace;
        Kokkos::parallel_for(
            "pops_pack_owned_history_faces",
            Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0, count),
            [=] POPS_HD(std::size_t ordinal) {
              std::size_t remainder = ordinal % cells;
              Index<Dim> face = face_box.lo;
              for (int direction = 0; direction < Dim; ++direction) {
                const auto width = static_cast<std::size_t>(face_box.length(direction));
                face[direction] += static_cast<int>(remainder % width);
                remainder /= width;
              }
              const int component = static_cast<int>(ordinal / cells);
              Real value = cell_provider ? named_flux_face_value_(values, face, axis, component)
                                         : values(face, component);
              packed(ordinal) = integrated ? value / measure : value;
            });
        const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, packed);
        patch.density[axis].assign(host.data(), host.data() + count);
      }
      snapshot->owned.push_back(std::move(patch));
    }
    history_flux::certify_prepared_layout(
        *snapshot, runtime_->hierarchy().layout(static_cast<std::size_t>(active_level_)));
    history_flux::validate_snapshot(*snapshot, facade_->configured_n_levels());
    digests.assign(checked_product_(snapshot->patches.size(), std::size_t{64},
                                    "AMR history source digest count"),
                   char{0});
    for (const auto& patch : snapshot->owned) {
      const auto digest = history_flux::patch_digest(patch);
      std::copy(digest.begin(), digest.end(), digests.begin() + patch.global_patch * 64);
    }
    checkpoint_detail::Writer metadata;
    history_flux::write_physical_metadata(metadata, *snapshot);
    metadata.i32(snapshot->rank_count);
    for (int owner : snapshot->owners)
      metadata.i32(owner);
    auto bytes = std::move(metadata).take();
    contract.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  });
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"history-face-source", contract}}, lane))
    throw std::invalid_argument("AMR history face source layout differs across execution ranks");
  // Only fixed-size per-patch digests communicate during RHS capture. Physical faces stay owned.
  all_reduce_or_inplace(digests.data(), digests.size(), lane);
  prepare_spatial_collectively([&] {
    std::vector<std::string> patch_digests;
    for (std::size_t patch = 0; patch < snapshot->patches.size(); ++patch)
      patch_digests.emplace_back(digests.data() + patch * 64, 64);
    snapshot->identity = history_flux::content_identity(*snapshot, patch_digests);
  });
  return snapshot;
}

std::vector<FluxBasisFace> history_faces_for_current_interfaces_(
    const std::shared_ptr<const HistoryFluxSnapshot>& snapshot,
    const std::vector<FluxBasisFace>& current_faces) const {
  if (!snapshot)
    throw std::invalid_argument("AMR moved history interface lacks physical source samples");
  const auto& lane = prepared_execution_lane();
  using Query = history_flux::FaceQuery<Dim>;
  using Key = std::pair<std::string, Query>;
  std::map<Key, std::shared_ptr<const HistoryFluxSnapshot>> queries;
  std::vector<std::vector<history_flux::SourceTerm<Dim>>> terms;
  std::vector<Real> samples;
  std::vector<Real> counts;
  std::string contract;
  prepare_spatial_collectively([&] {
    for (const auto& face : current_faces) {
      terms.push_back(history_flux::source_queries(snapshot, {face.axis, face.face},
                                                   facade_->configured_n_levels()));
      for (const auto& term : terms.back())
        queries.emplace(Key{term.source->identity, term.query}, term.source);
    }
    samples.assign(checked_product_(queries.size(), static_cast<std::size_t>(snapshot->components),
                                    "AMR sparse history face query payload"),
                   Real(0));
    counts.assign(queries.size(), Real(0));
    checkpoint_detail::Writer exact;
    exact.string(snapshot->identity);
    exact.size(queries.size());
    std::size_t ordinal = 0;
    for (const auto& [key, source] : queries) {
      exact.string(key.first);
      exact.i32(key.second.axis);
      for (int axis = 0; axis < Dim; ++axis)
        exact.i32(key.second.face[axis]);
      if (const auto payload = history_flux::local_payload(*source, key.second)) {
        counts[ordinal] = Real(1);
        std::copy(payload->begin(), payload->end(),
                  samples.begin() + ordinal * static_cast<std::size_t>(snapshot->components));
      }
      ++ordinal;
    }
    auto bytes = std::move(exact).take();
    contract.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  });
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"history-face-queries", contract}}, lane))
    throw std::invalid_argument("AMR history sparse query plan differs across ranks");
  all_reduce_sum_inplace(counts.data(), counts.size(), lane);
  all_reduce_sum_inplace(samples.data(), samples.size(), lane);
  std::vector<FluxBasisFace> result;
  prepare_spatial_collectively([&] {
    if (!std::all_of(counts.begin(), counts.end(), [](Real count) { return count == Real(1); }))
      throw std::invalid_argument("AMR historical face query has missing or duplicate ownership");
    std::map<Key, std::size_t> ordinals;
    for (const auto& [key, source] : queries) {
      (void)source;
      ordinals.emplace(key, ordinals.size());
    }
    result = current_faces;
    for (std::size_t face = 0; face < result.size(); ++face) {
      result[face].flux_density.assign(static_cast<std::size_t>(snapshot->components), Real(0));
      for (const auto& term : terms[face]) {
        const auto ordinal = ordinals.at(Key{term.source->identity, term.query});
        for (int component = 0; component < snapshot->components; ++component)
          result[face].flux_density[static_cast<std::size_t>(component)] +=
              static_cast<Real>(term.weight.value()) *
              samples[ordinal * static_cast<std::size_t>(snapshot->components) + component];
      }
    }
  });
  return result;
}

std::vector<FluxBasisFace> current_history_interface_geometry_(int level) const {
  using fragment_role_type = ::pops::amr::reflux::FaceLedgerRole;
  std::vector<FluxBasisFace> faces;
  const auto geometry = facade_->prepared_amr_level_geometry(level);
  if (level + 1 < nlev())
    for (const auto& interface : program_interface_faces_(static_cast<std::size_t>(level)))
      faces.push_back({fragment_role_type::Coarse,
                       interface.axis,
                       interface.coarse_face,
                       interface.coarse_face,
                       face_measure_(geometry, interface.axis),
                       {}});
  if (level > 0) {
    const auto& hierarchy = runtime_->hierarchy();
    const auto& fine = hierarchy.layout(static_cast<std::size_t>(level));
    const auto& coarse = hierarchy.layout(static_cast<std::size_t>(level - 1));
    const auto ratio = fine.ratio_from_parent();
    for (const auto& interface : program_interface_faces_(static_cast<std::size_t>(level - 1))) {
      std::size_t count = 1;
      for (int axis = 0; axis < Dim; ++axis)
        if (axis != interface.axis)
          count = checked_product_(count, static_cast<std::size_t>(ratio[axis]),
                                   "AMR historical fine interface enumeration");
      for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
        auto remainder = ordinal;
        Index<Dim> face;
        for (int axis = 0; axis < Dim; ++axis) {
          face[axis] = fine.domain().lo[axis] +
                       (interface.coarse_face[axis] - coarse.domain().lo[axis]) * ratio[axis];
          if (axis != interface.axis) {
            face[axis] += static_cast<int>(remainder % static_cast<std::size_t>(ratio[axis]));
            remainder /= static_cast<std::size_t>(ratio[axis]);
          }
        }
        faces.push_back({fragment_role_type::Fine,
                         interface.axis,
                         face,
                         interface.coarse_face,
                         face_measure_(geometry, interface.axis),
                         {}});
      }
    }
  }
  return faces;
}

std::shared_ptr<const HistoryFluxSnapshot> project_history_flux_snapshot_(
    const std::shared_ptr<const HistoryFluxSnapshot>& parent, int level) const {
  if (!parent)
    throw std::invalid_argument(
        "AMR parent history transfer lacks its authenticated complete face source");
  auto projected = std::make_shared<HistoryFluxSnapshot>();
  projected->identity = "pending-conservative-projection-digest";
  projected->source_identity = parent->source_identity;
  projected->parent = parent;
  projected->level = level;
  projected->components = parent->components;
  projected->rank_count = parent->rank_count;
  projected->local_rank = parent->local_rank;
  const auto& layout = runtime_->hierarchy().layout(static_cast<std::size_t>(level));
  const auto geometry = facade_->prepared_amr_level_geometry(level);
  projected->domain = layout.domain();
  projected->patches = layout.patches().boxes();
  // Projection nodes own no payload: ownership belongs exclusively to immutable raw leaves.
  projected->owners.assign(projected->patches.size(), 0);
  for (int axis = 0; axis < Dim; ++axis) {
    projected->ratio[axis] = layout.ratio_from_parent()[axis];
    projected->cell_size[axis] = static_cast<double>(geometry.spacing(axis));
  }
  history_flux::certify_prepared_layout(*projected, layout);
  history_flux::validate_snapshot(*projected, facade_->configured_n_levels());
  projected->identity = history_flux::projection_identity(*projected);
  return projected;
}

// Called only at the accepted regrid boundary. Every query plan is agreed before payload MPI;
// history reads themselves merely consume this precomputed current-interface view.
void prepare_remapped_history_flux_faces_(
    std::map<std::string, std::vector<FluxExpression>>& expressions) const {
  using CacheKey = std::pair<std::string, int>;
  std::map<CacheKey, std::vector<FluxBasisFace>> views;
  for (auto& [key, slots] : expressions) {
    const int level = decode_history_key_(key)->first;
    std::vector<FluxBasisFace> geometry;
    prepare_spatial_collectively([&] { geometry = current_history_interface_geometry_(level); });
    for (auto& expression : slots)
      for (auto& [identity, term] : expression) {
        (void)identity;
        std::shared_ptr<FluxBasis> candidate;
        prepare_spatial_collectively([&] {
          if (!term.basis)
            throw std::logic_error("AMR history remap has an absent flux basis");
          candidate = std::make_shared<FluxBasis>(*term.basis);
          candidate->level = level;
          for (const auto& sample : candidate->shared_samples)
            if (!sample || sample->level != level)
              throw std::invalid_argument("AMR shared history flux cannot transfer across levels");
        });
        if (candidate->history_snapshot) {
          const CacheKey cache{candidate->history_snapshot->identity, level};
          auto found = views.find(cache);
          if (found == views.end()) {
            auto faces =
                history_faces_for_current_interfaces_(candidate->history_snapshot, geometry);
            prepare_spatial_collectively(
                [&] { found = views.emplace(cache, std::move(faces)).first; });
          }
          prepare_spatial_collectively([&] { candidate->faces = found->second; });
        } else {
          // Legacy archives authenticate only their old interface view. Static interfaces may
          // continue until these slots rotate out; moved or parent-projected faces must refuse.
          prepare_spatial_collectively([&] {
            std::vector<FluxBasisFace> faces;
            for (const auto& face : geometry) {
              const auto old = std::find_if(
                  candidate->faces.begin(), candidate->faces.end(), [&](const auto& previous) {
                    return face.role == previous.role && face.axis == previous.axis &&
                           face.face == previous.face && face.coarse_face == previous.coarse_face &&
                           face.face_measure == previous.face_measure;
                  });
              if (old == candidate->faces.end())
                throw std::invalid_argument(
                    "AMR moved legacy history interface lacks authenticated source samples");
              faces.push_back(*old);
            }
            candidate->faces = std::move(faces);
          });
        }
        term.basis = std::move(candidate);
      }
  }
}

void publish_history_flux_snapshots_() const {
  if (history_flux_expressions_.empty() && facade_->program_history_flux_snapshots().empty())
    return;
  prepare_spatial_collectively([&] {
    history_flux::SnapshotMap<Dim> raw;
    for (const auto& [key, slots] : history_flux_expressions_) {
      (void)key;
      for (const auto& expression : slots)
        for (const auto& [identity, term] : expression) {
          (void)identity;
          auto snapshot = term.basis ? term.basis->history_snapshot : nullptr;
          int remaining = facade_->configured_n_levels();
          while (snapshot && snapshot->parent) {
            if (--remaining <= 0)
              throw std::logic_error("AMR history snapshot lineage exceeds authored levels");
            snapshot = snapshot->parent;
          }
          if (snapshot) {
            const auto source_identity = snapshot->identity;
            raw.emplace(source_identity, std::move(snapshot));
          }
        }
    }
    // Replacing the reachable set drops retired/rejected ancestry; shared slots never duplicate data.
    facade_->publish_program_history_flux_snapshots(std::move(raw));
  });
}
